#include "helpers.h"
#include "Eigensolver/eig_workspace.h"
#include "punmtr_he2hb_grouped.h"
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "ddla_desc.h"
#include "comm_traits.h"
#include "BLAS/blas_impl.h"
#include "plan_util.h"
#include "punmtr_impl.h"
#include "instantiate.h"

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace ddla {
namespace detail {
namespace {
struct Panel
{
    int slot;
    int start;
    int width;
    int prefix;
    int rows;
    int compact_ld;
    int root;
    int column;
};

struct GramBlock
{
    int i;
    int j;
    std::size_t offset;
};

#if DDLA_HAS_GPU
// Copy masked tails/unit anchors and diagonal upper factors in one stream operation.
// A panel's source block column has one owner; row broadcasts follow this pack.
template <bool PackFactor, class D>
__global__ void grouped_pack(
    Panel p0,
    Panel p1,
    Panel p2,
    int count,
    int K,
    int ldv,
    int nb,
    int band,
    int pr,
    int pc,
    int mr,
    int mc,
    int lda,
    const D* A,
    const D* tmat,
    D* V,
    D* S)
{
    const std::size_t vsize = std::size_t(ldv) * K;
    const std::size_t ssize = PackFactor ? std::size_t(K) * K : 0;
    const std::size_t total = vsize > ssize ? vsize : ssize;
    for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += std::size_t(blockDim.x) * gridDim.x) {
        if (index < vsize) {
            const int row = int(index % ldv), col = int(index / ldv);
            const Panel p = count > 2 && col >= p2.column   ? p2
                            : count > 1 && col >= p1.column ? p1
                                                            : p0;
            D value = D(0);
            if (mc == p.root) {
                const int global = (row / nb * pr + mr) * nb + row % nb;
                const int pivot = p.start + col - p.column - band;
                if (global < pivot) {
                    const int local_col = p.start / nb / pc * nb + p.start % nb + col - p.column;
                    value = A[row + std::size_t(local_col) * lda];
                } else if (global == pivot)
                    value = D(1);
            }
            V[index] = value;
        }
        if constexpr (PackFactor) {
            if (index < ssize) {
                const int row = int(index % K), col = int(index / K);
                const Panel p = count > 2 && col >= p2.column   ? p2
                                : count > 1 && col >= p1.column ? p1
                                                                : p0;
                S[index] = row >= p.column && row <= col
                               ? tmat[std::size_t(p.slot) * band * band + row - p.column +
                                      std::size_t(col - p.column) * band]
                               : D(0);
            }
        }
    }
}
#endif

// Element counts of V, W, W2, S, G and G2 (carving order). K, the widest
// group's columns, is at most min(group_cap, steps) panel widths.
void grouped_counts(
    const GroupedStage5Plan& p, bool compact_single, std::size_t count[6])
{
    const int panels = std::min(p.options.group_cap, p.steps);
    if (panels == 1) {
        // Borrow the original T slot without padded dummy S/Gram buffers.
        // CPU retains the old sequential pitches. GPU keeps its existing
        // compact width for a partial panel, avoiding larger tiny-case scratch.
        const int width = compact_single ? std::min(p.nbw, p.n - p.nbw) : p.nbw;
        count[0] = std::size_t(std::max(1, p.max_rows)) * width;
        count[1] = count[2] = std::size_t(width) * std::max(1, p.zcols);
        count[3] = count[4] = count[5] = 0;
        return;
    }
    const auto capacity = std::size_t(std::min(panels * p.nbw, p.n - p.nbw));
    const auto rows = std::size_t(std::max(1, p.max_rows));
    const auto columns = std::size_t(std::max(1, p.zcols));
    const auto gram =
        std::size_t(panels * (panels - 1) / 2) * std::size_t(p.nbw) * std::size_t(p.nbw);
    count[0] = rows * capacity;
    count[1] = count[2] = capacity * columns;
    count[3] = capacity * capacity;
    count[4] = count[5] = std::max(std::size_t(1), gram);
}
} // namespace

GroupedStage5Plan grouped_stage5_geometry(
    const spectral_impl::UnmtrHe2hbPlan& common, const GroupedStage5Options& options)
{
    GroupedStage5Plan p;
    p.trans = common.trans;
    p.n = common.n;
    p.nbw = common.nbw;
    p.nb = common.nb;
    p.steps = common.nsteps;
    p.options = options;
    p.pr = common.nprows;
    p.pc = common.npcols;
    p.mr = common.myprow;
    p.mc = common.mypcol;
    p.zcols = common.zcols;
    p.max_rows = common.ldv;
    p.lldA = common.lldA;
    p.lldZ = common.lldZ;
    return p;
}

template <class T>
GroupedStage5Plan grouped_stage5_geometry(
    const DdlaHandle_t& h,
    char trans,
    int n,
    int nbw,
    const int* da,
    const int* dz,
    const GroupedStage5Options& options)
{
    return grouped_stage5_geometry(
        spectral_impl::punmtr_he2hb_geometry(h, ddlaGetBackend(h), trans, n, nbw, da, dz), options);
}

template <class T>
GroupedStage5Plan grouped_stage5_check(
    const DdlaHandle_t& handle,
    char trans,
    int n,
    int nbw,
    const T* A,
    const int* descA,
    const T* tmat,
    T* Z,
    const int* descZ,
    const GroupedStage5Options& options)
{
    if (!handle)
        throw std::invalid_argument("grouped band-to-full: null handle");
    const auto common = spectral_impl::punmtr_he2hb_check(
        handle, ddlaGetBackend(handle), trans, n, nbw, A, descA, tmat, Z, descZ);
    if (options.group_cap < 1 || options.group_cap > 3)
        throw std::invalid_argument("grouped band-to-full: group cap must be 1, 2 or 3");
    if (common.nsteps) {
        const int rows = num_loc(n, common.nb, common.myprow, 0, common.nprows);
        const int cols = num_loc(n, common.nb, common.mypcol, 0, common.npcols);
        const std::size_t t_elements = std::size_t(nbw) * nbw * common.nsteps;
        const std::size_t z_elements =
            rows && common.zcols ? std::size_t(common.zcols - 1) * common.lldZ + rows : 0;
        if (matrix_overlaps(A, common.lldA, rows, cols, Z, common.lldZ, rows, common.zcols) ||
            array_overlaps(tmat, t_elements, Z, z_elements))
            throw std::invalid_argument("grouped band-to-full: overlapping factors/output");
    }
    return grouped_stage5_geometry(common, options);
}

template <class T>
void grouped_stage5_workspace_size(
    const DdlaHandle_t& h, const GroupedStage5Plan& p, std::size_t* backend_bytes)
{
    *backend_bytes = 0;
    if (!p.steps)
        return;
    std::size_t count[6];
    grouped_counts(p, ddlaGetBackend(h) == DdlaBackend::GPU, count);
    for (int i = 0; i < 6; ++i)
        if (count[i])
            *backend_bytes += padded_bytes<T>(count[i]);
}

namespace {
// One sweep body, specialized once for a single-panel plan. Multi-panel plans
// keep their original full-factor path even when the last group has one panel.
template <DdlaBackend Backend, class T, bool SinglePanel>
void grouped_stage5_run_impl(
    const DdlaHandle_t& handle,
    const GroupedStage5Plan& p,
    const T* A,
    const T* tmat,
    T* Z,
    int& info,
    void* backend_work,
    std::size_t backend_bytes)
{
    info = 0;
    if (!p.steps)
        return;
    const auto& options = p.options;
    const int n = p.n;
    const int nbw = p.nbw;
    const int nb = p.nb;
    const int lldA = p.lldA;
    const int lldZ = p.lldZ;
    const int steps = p.steps;
    const int zcols = p.zcols;
    const int pr = p.pr;
    const int pc = p.pc;
    const int mr = p.mr;
    const int mc = p.mc;
    const char trans = p.trans;

    std::size_t counts[6];
    grouped_counts(p, Backend == DdlaBackend::GPU, counts);
    char* work = static_cast<char*>(backend_work);
    T* const V = take<T>(work, counts[0]);
    T* const W = take<T>(work, counts[1]);
    T* const W2 = take<T>(work, counts[2]);
    T* const S = counts[3] ? take<T>(work, counts[3]) : nullptr;
    T* const G = counts[4] ? take<T>(work, counts[4]) : nullptr;
    T* const G2 = counts[5] ? take<T>(work, counts[5]) : nullptr;
    assert(std::size_t(work - static_cast<char*>(backend_work)) <= backend_bytes);
    (void)backend_bytes;
    const bool adjoint = trans != 'N';
    const char conjugate_op = std::is_same<T, typename real_type<T>::type>::value ? 'T' : 'C';
    const char factor_op = adjoint ? conjugate_op : 'N';
    int cursor = adjoint ? 0 : steps;
    // Workspace is allocated once, before processing the groups.
    while (adjoint ? cursor < steps : cursor > 0) {
        const int count =
            SinglePanel ? 1 : std::min(options.group_cap, adjoint ? steps - cursor : cursor);
        const int lo = adjoint ? cursor : cursor - count;
        const int hi = lo + count;
        Panel panels[3]{};
        GramBlock pairs[3];
        int K = 0;
        int pair_count = 0;
        std::size_t q = 0;
        for (int j = 0; j < count; ++j) {
            const int slot = lo + j;
            const int start = (slot + 1) * nbw;
            const int width = std::min(nbw, n - start);
            const int prefix = start + width - nbw;
            const int rows = num_loc(prefix, nb, mr, 0, pr);
            panels[j] = Panel{
                slot, start, width, prefix, rows, std::max(1, rows), indxg2p(start, nb, 0, pc), K};
            K += width;
            for (int i = 0; i < j; ++i) {
                pairs[pair_count++] = GramBlock{i, j, q};
                q += std::size_t(panels[i].width) * width;
            }
        }
        const int rows = panels[count - 1].rows;
        const int ldv = std::max(1, rows);
        const std::size_t w_count = std::size_t(K) * zcols;
        if constexpr (Backend == DdlaBackend::CPU) {
            for (int j = 0; j < count; ++j) {
                const Panel& p = panels[j];
                T* base = V + std::size_t(p.column) * ldv;
                std::fill_n(base, std::size_t(p.compact_ld) * p.width, T(0));
                if (p.root == mc) {
                    const int lc = indxg2l(p.start, nb, pc);
                    for (int c = 0; c < p.width; ++c) {
                        const int nr = num_loc(p.start + c - nbw, nb, mr, 0, pr);
                        if (nr > 0)
                            std::memcpy(base + std::size_t(c) * p.compact_ld,
                                        A + std::size_t(lc + c) * lldA,
                                        std::size_t(nr) * sizeof(T));
                    }
                }
            }
            for (int j = 0; j < count; ++j) {
                const Panel& p = panels[j];
                T* base = V + std::size_t(p.column) * ldv;
                const std::size_t payload = std::size_t(p.compact_ld) * p.width;
                commBcast<DdlaBackend::CPU>(handle, CommScope::Row, base, payload, p.root);
            }
            for (int j = 0; j < count; ++j) {
                const Panel& p = panels[j];
                T* base = V + std::size_t(p.column) * ldv;
                if (p.compact_ld != ldv) {
                    for (int c = p.width - 1; c >= 0; --c) {
                        T* target = base + std::size_t(c) * ldv;
                        std::memmove(target,
                                     base + std::size_t(c) * p.compact_ld,
                                     std::size_t(p.compact_ld) * sizeof(T));
                        std::fill(target + p.compact_ld, target + ldv, T(0));
                    }
                }
                for (int c = 0; c < p.width; ++c) {
                    const int pivot = p.start + c - nbw;
                    if (mr == indxg2p(pivot, nb, 0, pr))
                        base[num_loc(pivot, nb, mr, 0, pr) + std::size_t(c) * ldv] = T(1);
                }
            }
        } else {
#if DDLA_HAS_GPU
            using D = DeviceScalarT<T>;
            const std::size_t v_items = std::size_t(ldv) * K;
            const std::size_t items = SinglePanel ? v_items : std::max(v_items, std::size_t(K) * K);
            const int blocks = int(std::min<std::size_t>((items + 255) / 256, 65535));
            // Single-panel plans pack V only and borrow T; multi-panel plans
            // keep the original factor pack without a per-group mode branch.
            grouped_pack<!SinglePanel, D><<<blocks, 256, 0, static_cast<gpuStream_t>(handle->stream)>>>(
                panels[0],
                panels[1],
                panels[2],
                count,
                K,
                ldv,
                nb,
                nbw,
                pr,
                pc,
                mr,
                mc,
                lldA,
                reinterpret_cast<const D*>(A),
                reinterpret_cast<const D*>(tmat),
                reinterpret_cast<D*>(V),
                reinterpret_cast<D*>(S));
            GPU_CHECK(gpuGetLastError());
            // Common leading dimension avoids an overlapping device expansion.
            for (int j = 0; j < count; ++j) {
                const Panel& panel = panels[j];
                commBcast<Backend>(handle,
                                   CommScope::Row,
                                   V + std::size_t(panel.column) * ldv,
                                   std::size_t(ldv) * panel.width,
                                   panel.root);
            }
#endif
        }
        if constexpr (SinglePanel) {
            const T* factor = tmat + std::size_t(panels[0].slot) * nbw * nbw;
            // Borrow T with its original nbw pitch. CPU preserves the original
            // W pitch; GPU retains its existing compact partial-panel pitch.
            // No Gram work, combined factor or identity reduction is needed.
            const int ldw = Backend == DdlaBackend::CPU ? nbw : K;
            punmtr_run<Backend, T>(handle,
                                   'L',
                                   'U',
                                   factor_op,
                                   rows,
                                   zcols,
                                   K,
                                   ldv,
                                   nbw,
                                   lldZ,
                                   ldw,
                                   pr > 1,
                                   V,
                                   factor,
                                   Z,
                                   W,
                                   W2);
        } else {
            for (int j = 0; j < pair_count; ++j) {
                const GramBlock& g = pairs[j];
                const Panel& a = panels[g.i];
                const Panel& b = panels[g.j];
                if (a.rows > 0)
                    blas_impl::gemm_run<Backend, T>(handle,
                                                    conjugate_op,
                                                    'N',
                                                    a.width,
                                                    b.width,
                                                    a.rows,
                                                    T(1),
                                                    V + std::size_t(a.column) * ldv,
                                                    ldv,
                                                    V + std::size_t(b.column) * ldv,
                                                    ldv,
                                                    T(0),
                                                    G + g.offset,
                                                    a.width);
                else if constexpr (Backend == DdlaBackend::CPU)
                    std::fill_n(G + g.offset, std::size_t(a.width) * b.width, T(0));
                else
                    DDLA_CHECK(ddlaMemset(
                        handle, G + g.offset, 0, std::size_t(a.width) * b.width * sizeof(T)));
            }
            if (q)
                commAllReduce<Backend>(handle, CommScope::Col, G, G2, int(q), cclSum);
            int previous = 0;
            for (int j = 0; j < count; ++j) {
                const Panel& p = panels[j];
                const T* block = tmat + std::size_t(p.slot) * nbw * nbw;
                if constexpr (Backend == DdlaBackend::CPU) {
                    for (int c = 0; c < p.width; ++c)
                        for (int r = 0; r <= c; ++r)
                            S[previous + r + std::size_t(previous + c) * K] =
                                block[r + std::size_t(c) * nbw];
                }
                if (previous) {
                    for (int z = 0; z < pair_count; ++z)
                        if (pairs[z].j == j) {
                            const Panel& a = panels[pairs[z].i];
                            if constexpr (Backend == DdlaBackend::CPU) {
                                for (int c = 0; c < p.width; ++c)
                                    std::copy_n(G2 + pairs[z].offset + std::size_t(c) * a.width,
                                                a.width,
                                                G + a.column + std::size_t(c) * previous);
                            } else {
                                blas_impl::geam_run<Backend, T>(handle,
                                                                'N',
                                                                'N',
                                                                a.width,
                                                                p.width,
                                                                T(1),
                                                                G2 + pairs[z].offset,
                                                                a.width,
                                                                T(0),
                                                                nullptr,
                                                                1,
                                                                G + a.column,
                                                                previous);
                            }
                        }
                    T* off = S + std::size_t(previous) * K;
                    blas_impl::trmm_run<Backend, T>(
                        handle, 'L', 'U', 'N', 'N', previous, p.width, T(-1), S, K, G, previous);
                    blas_impl::trmm_run<Backend, T>(
                        handle, 'R', 'U', 'N', 'N', previous, p.width, T(1), block, nbw, G, previous);
                    if constexpr (Backend == DdlaBackend::CPU) {
                        for (int c = 0; c < p.width; ++c)
                            std::copy_n(
                                G + std::size_t(c) * previous, previous, off + std::size_t(c) * K);
                    } else {
                        blas_impl::geam_run<Backend, T>(handle,
                                                        'N',
                                                        'N',
                                                        previous,
                                                        p.width,
                                                        T(1),
                                                        G,
                                                        previous,
                                                        T(0),
                                                        nullptr,
                                                        1,
                                                        off,
                                                        K);
                    }
                }
                previous += p.width;
            }

            punmtr_project_run<Backend, T>(handle, 'L', rows, zcols, K, ldv, lldZ, K, V, Z, W);
            commAllReduce<Backend>(handle, CommScope::Col, W, W2, int(w_count), cclSum);
            if (zcols > 0)
                blas_impl::trmm_run<Backend, T>(
                    handle, 'L', 'U', factor_op, 'N', K, zcols, T(1), S, K, W2, K);
            punmtr_update_run<Backend, T>(handle, 'L', rows, zcols, K, ldv, lldZ, K, V, W2, Z);
        }
        cursor = adjoint ? hi : lo;
    }
}

} // namespace

template <DdlaBackend Backend, class T>
void grouped_stage5_run(
    const DdlaHandle_t& handle,
    const GroupedStage5Plan& p,
    const T* A,
    const T* tmat,
    T* Z,
    int& info,
    void* backend_work,
    std::size_t backend_bytes)
{
    if (std::min(p.options.group_cap, p.steps) == 1)
        grouped_stage5_run_impl<Backend, T, true>(
            handle, p, A, tmat, Z, info, backend_work, backend_bytes);
    else
        grouped_stage5_run_impl<Backend, T, false>(
            handle, p, A, tmat, Z, info, backend_work, backend_bytes);
}

template <DdlaBackend Backend, class T>
void punmtr_he2hb_grouped(
    const DdlaHandle_t& h,
    char trans,
    int n,
    int nbw,
    const T* A,
    const int* da,
    const T* tmat,
    T* Z,
    const int* dz,
    const GroupedStage5Options& options,
    int& info)
{
    info = 0;
    GroupedStage5Plan p;
    try {
        p = grouped_stage5_check<T>(h, trans, n, nbw, A, da, tmat, Z, dz, options);
        if (ddlaGetBackend(h) != Backend)
            throw std::invalid_argument("grouped band-to-full: template backend mismatch");
    } catch (...) {
        info = -1;
        throw;
    }
    std::size_t backend_bytes = 0;
    ddla::vector<char> backend_work(h);
    try {
        grouped_stage5_workspace_size<T>(h, p, &backend_bytes);
        backend_work.resize(backend_bytes);
    } catch (...) {
        info = -2;
        throw;
    }
    try {
        grouped_stage5_run<Backend, T>(h, p, A, tmat, Z, info, backend_work.data(), backend_bytes);
        if constexpr (Backend == DdlaBackend::GPU)
            DDLA_CHECK(ddlaSynchronize(h));
    } catch (...) {
        if constexpr (Backend == DdlaBackend::GPU) {
            try {
                (void)ddlaSynchronize(h);
            } catch (...) {
            }
        }
        info = -3;
        throw;
    }
}

#define INSTANTIATE_GEOMETRY(T)                                                                    \
    template GroupedStage5Plan grouped_stage5_geometry<T>(                                         \
        const DdlaHandle_t&, char, int, int, const int*, const int*, const GroupedStage5Options&); \
    template GroupedStage5Plan grouped_stage5_check<T>(const DdlaHandle_t&,                        \
                                                       char,                                       \
                                                       int,                                        \
                                                       int,                                        \
                                                       const T*,                                   \
                                                       const int*,                                 \
                                                       const T*,                                   \
                                                       T*,                                         \
                                                       const int*,                                 \
                                                       const GroupedStage5Options&);               \
    template void grouped_stage5_workspace_size<T>(                                                \
        const DdlaHandle_t&, const GroupedStage5Plan&, std::size_t*)
#define INSTANTIATE_RUN(B, T)                                                                      \
    template void grouped_stage5_run<B, T>(const DdlaHandle_t&,                                    \
                                           const GroupedStage5Plan&,                               \
                                           const T*,                                               \
                                           const T*,                                               \
                                           T*,                                                     \
                                           int&,                                                   \
                                           void*,                                                  \
                                           std::size_t);                                           \
    template void punmtr_he2hb_grouped<B, T>(const DdlaHandle_t&,                                  \
                                             char,                                                 \
                                             int,                                                  \
                                             int,                                                  \
                                             const T*,                                             \
                                             const int*,                                           \
                                             const T*,                                             \
                                             T*,                                                   \
                                             const int*,                                           \
                                             const GroupedStage5Options&,                          \
                                             int&)
DDLA_FOR_EACH_TYPE(INSTANTIATE_GEOMETRY)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_RUN)
#undef INSTANTIATE_RUN
#undef INSTANTIATE_GEOMETRY
}
}
