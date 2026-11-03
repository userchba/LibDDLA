#include <ddla/ddla.h>
#include "ddla_desc.h"
#include "comm_traits.h"
#include "require_backend.h"
#include "transport_block.h"
#include "check_agree.h"
#include "plan_util.h"
#include "ptrmm_impl.h"
#include "pgeadd_impl.h"
#include "pblas_workspace.h"
#include "BLAS/blas_impl.h"
#include <ddla/ddla_vector.h>
#include <algorithm>
#include <cassert>
#include <cctype>
#include <stdexcept>
#include <vector>

namespace ddla {
namespace ptrmm_impl {
namespace {

// Element counts of ptrmm's own buffers, in carving order: the staged
// diagonal block | a block row/column of the original B | the off-diagonal
// panel of A | the buffer the panel is exchanged or transposed into.
void scratch_counts(const DdlaHandle_t& h, bool left, int m, int n, const int* descA, const int* descB,
                    std::size_t (&count)[4])
{
    const int order = left ? m : n;
    const int nb = descA[DDLA_MB_];
    const int pr = h->nprows_, pc = h->npcols_;
    const int row = h->myprow_, col = h->mypcol_;
    const int ml = num_loc(m, descB[DDLA_MB_], row, descB[DDLA_RSRC_], pr);
    const int nl = num_loc(n, descB[DDLA_NB_], col, descB[DDLA_CSRC_], pc);
    const int along = left ? ml : nl;
    const int source_along = num_loc(order, nb, left ? row : col,
                                     descA[left ? DDLA_RSRC_ : DDLA_CSRC_], left ? pr : pc);
    count[0] = std::size_t(nb) * nb;
    count[1] = std::size_t(nb) * std::max(1, left ? nl : ml);
    count[2] = std::size_t(nb) * std::max({1, along, source_along});
    count[3] = count[2];
}

// The pgeadd of a transposed panel: the sweep's longest panel is `order` long
// and one block wide, so it is added as order x nb (side 'L') or nb x order
// (side 'R').
template <typename T>
void panel_add_bytes(const DdlaHandle_t& h, bool left, char trans, int order, int nb,
                     std::size_t* backend_bytes, std::size_t* host_bytes)
{
    *backend_bytes = *host_bytes = 0;
    if (trans == 'N') return;
    int d[DDLA_DLEN_];
    pblas::describe(d, h, left ? order : nb, left ? nb : order, nb, nb, 0, 0);
    pgeadd_impl::pgeadd_workspace_size<T>(h, trans, 'N', true, false, d[DDLA_M_], d[DDLA_N_], d,
                                          backend_bytes, host_bytes);
}

// Extract only the off-diagonal panel, in op(A)'s orientation and B's
// distribution along the long dimension. The short dimension is replicated.
// An exchanged panel swaps panel and work; a transposed one is added into
// work by pgeadd on the rest of the scratch (add_work / add_host).
template <DdlaBackend Backend, typename T>
void triangular_panel(const DdlaHandle_t& h, bool left, char trans,
                      int k, int width, int first, int length,
                      const T* A, const int* da, const int* db, T*& panel, T*& work,
                      void* add_work, std::size_t add_bytes, void* add_host, std::size_t add_host_bytes)
{
    const int nb = da[DDLA_MB_];
    const int coord = left ? h->myprow_ : h->mypcol_;
    const int procs = left ? h->nprows_ : h->npcols_;
    const int target = db[left ? DDLA_RSRC_ : DDLA_CSRC_];
    if (trans == 'N') {
        transport_impl::transport_run<Backend, T>(h, left ? 'C' : 'R', 'N',
            left ? length : width, left ? width : length,
            A, left ? first : k, left ? k : first, da, panel, nullptr, 0, nullptr, 0);
        const int source = da[left ? DDLA_RSRC_ : DDLA_CSRC_];
        if (source != target) {
            const int send_length = num_loc(first + length, nb, coord, source, procs)
                                  - num_loc(first, nb, coord, source, procs);
            const int recv_length = num_loc(first + length, nb, coord, target, procs)
                                  - num_loc(first, nb, coord, target, procs);
            const int shift = (target - source + procs) % procs;
            commSendrecv<Backend>(h, left ? CommScope::Col : CommScope::Row,
                panel, std::size_t(send_length) * width, (coord + shift) % procs,
                work, std::size_t(recv_length) * width, (coord - shift + procs) % procs);
            std::swap(panel, work);
        }
        return;
    }

    // pgeadd handles rectangular grids and independent source coordinates.
    // A compact descriptor preserves the block phase of the cropped panel;
    // the origin is block-aligned and its short dimension is one block.
    int d[DDLA_DLEN_];
    const int owner = indxg2p(first, nb, target, procs);
    pblas::describe(d,h,left?length:width,left?width:length,nb,nb,left?owner:0,left?0:owner);
    pgeadd_impl::pgeadd_run<Backend, T>(h, trans, 'N', d[DDLA_M_], d[DDLA_N_],
        T(1), A, (left ? k : first) + 1, (left ? first : k) + 1, da,
        T(0), static_cast<const T*>(nullptr), 1, 1, d, work, 1, 1, d,
        add_work, add_bytes, add_host, add_host_bytes);
    transport_impl::transport_run<Backend, T>(h, left ? 'C' : 'R', 'N',
                            d[DDLA_M_], d[DDLA_N_], work, 0, 0, d, panel, nullptr, 0, nullptr, 0);
}

} // namespace

// Component 1.
template <DdlaBackend Backend>
bool ptrmm_check(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag,
                      int m, int n, const int* descA, const int* descB)
{
    return detail::agree_check(handle, "ptrmm", [&](detail::Agree&) {
        detail::require_backend<Backend>(handle, "ptrmm");
        check_desc(descA, handle);
        check_desc(descB, handle);
        side = static_cast<char>(std::toupper(static_cast<unsigned char>(side)));
        uplo = static_cast<char>(std::toupper(static_cast<unsigned char>(uplo)));
        trans = static_cast<char>(std::toupper(static_cast<unsigned char>(trans)));
        diag = static_cast<char>(std::toupper(static_cast<unsigned char>(diag)));
        if ((side != 'L' && side != 'R') || (uplo != 'L' && uplo != 'U') ||
            (trans != 'N' && trans != 'T' && trans != 'C') ||
            (diag != 'N' && diag != 'U'))
            throw std::invalid_argument("ptrmm: invalid side, uplo, trans or diag");
        const bool left = side == 'L';
        const int order = left ? m : n;
        if (m < 0 || n < 0 || order > descA[DDLA_M_] || order > descA[DDLA_N_] ||
            m > descB[DDLA_M_] || n > descB[DDLA_N_])
            throw std::invalid_argument("ptrmm: dimensions outside descriptors");
        const int nb = descA[DDLA_MB_];
        if (nb != descA[DDLA_NB_] || nb != descB[left ? DDLA_MB_ : DDLA_NB_])
            throw std::invalid_argument("ptrmm: square A blocks must match the multiplied dimension of B");
        return m == 0 || n == 0;
    });
}

// Component 2.  Backend [ diagonal | block of B | panel | work
// (scratch_counts) | the panel's pgeadd (panel_add_bytes) ], host [ the
// panel's pgeadd ].
template <typename T>
void ptrmm_workspace_size(const DdlaHandle_t& handle, char side, char trans, int m, int n,
                          bool alpha_is_zero, const int* descA, const int* descB,
                          std::size_t* backend_bytes, std::size_t* host_bytes)
{
    *backend_bytes = *host_bytes = 0;
    if (alpha_is_zero || m == 0 || n == 0) return;
    const bool left = side == 'L';
    std::size_t count[4], add_bytes = 0;
    scratch_counts(handle, left, m, n, descA, descB, count);
    panel_add_bytes<T>(handle, left, trans, left ? m : n, descA[DDLA_MB_], &add_bytes, host_bytes);
    for (const std::size_t c : count) *backend_bytes += detail::padded_bytes<T>(c);
    *backend_bytes += add_bytes;
}

// Component 3.
template <DdlaBackend Backend, typename T>
void ptrmm_run(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag,
               int m, int n, const T& alpha, const T* A, const int* descA, T* B, const int* descB,
               void* backend_work, [[maybe_unused]] std::size_t backend_bytes,
               void* host_work, std::size_t host_bytes)
{
    try {
        if (m == 0 || n == 0) return;
        const bool left = side == 'L';
        const int order = left ? m : n;
        const int nb = descA[DDLA_MB_];

        const int pr = handle->nprows_, pc = handle->npcols_;
        const int row = handle->myprow_, col = handle->mypcol_;
        const int ml = num_loc(m, descB[DDLA_MB_], row, descB[DDLA_RSRC_], pr);
        const int nl = num_loc(n, descB[DDLA_NB_], col, descB[DDLA_CSRC_], pc);
        const int ldb = descB[DDLA_LLD_];
        if (alpha == T(0)) {
            // BLAS zero-alpha semantics: neither A nor the old B is read.
            if (ml > 0) for (int j = 0; j < nl; ++j)
                DDLA_CHECK(ddlaMemset(handle, B + std::size_t(j) * ldb, 0, std::size_t(ml) * sizeof(T)));
            return;
        }

        // The scratch, in the order ptrmm_workspace_size counts it; the rest is
        // the panel's pgeadd.  Every operation and collective uses handle's stream.
        std::size_t count[4];
        scratch_counts(handle, left, m, n, descA, descB, count);
        char* w = static_cast<char*>(backend_work);
        T* const diagonal = detail::take<T>(w, count[0]);
        T* const block_B = detail::take<T>(w, count[1]);
        T* panel = detail::take<T>(w, count[2]);
        T* work = detail::take<T>(w, count[3]);
        const std::size_t used = static_cast<std::size_t>(w - static_cast<char*>(backend_work));
        assert(used <= backend_bytes);

        const bool lower = (uplo == 'L') == (trans == 'N');
        const bool forward = left ? !lower : lower;
        const int blocks = (order - 1) / nb + 1;
        for (int step = 0; step < blocks; ++step) {
            const int k = (forward ? step : blocks - 1 - step) * nb;
            const int width = std::min(nb, order - k);
            const int ar = indxg2p(k, nb, descA[DDLA_RSRC_], pr);
            const int ac = indxg2p(k, nb, descA[DDLA_CSRC_], pc);
            if (row == ar && col == ac) {
                const int i = num_loc(k, nb, row, descA[DDLA_RSRC_], pr);
                const int j = num_loc(k, nb, col, descA[DDLA_CSRC_], pc);
                DDLA_CHECK(ddlaMemcpy2D(handle, diagonal, std::size_t(width) * sizeof(T),
                    A + i + std::size_t(j) * descA[DDLA_LLD_], std::size_t(descA[DDLA_LLD_]) * sizeof(T),
                    std::size_t(width) * sizeof(T), width));
            }
            commBcast<Backend>(handle, CommScope::Grid, diagonal,
                              std::size_t(width) * width, handle->rc_to_rank(ar, ac));

            // Save this original B panel before applying its diagonal block. The
            // sweep updates only panels already visited, so every input is used
            // before it is overwritten and alpha is applied exactly once. Local TRMM
            // overwrites B itself; block_B stays original for the GEMM updates.
            transport_impl::transport_run<Backend, T>(handle, left ? 'R' : 'C', 'N',
                left ? width : m, left ? n : width,
                B, left ? k : 0, left ? 0 : k, descB, block_B, nullptr, 0, nullptr, 0);
            const int coord = left ? row : col, procs = left ? pr : pc;
            const int source = descB[left ? DDLA_RSRC_ : DDLA_CSRC_];
            const int local_k = num_loc(k, nb, coord, source, procs);
            if (coord == indxg2p(k, nb, source, procs) && (left ? nl : ml) > 0) {
                blas_impl::trmm_run<Backend>(handle, side, uplo, trans, diag, left ? width : ml, left ? nl : width,
                    alpha, diagonal, width,
                    B + (left ? std::size_t(local_k) : std::size_t(local_k) * ldb), ldb);
            }

            const int first = forward ? 0 : k + width;
            const int length = forward ? k : order - first;
            if (length == 0) continue;
            triangular_panel<Backend, T>(handle, left, trans, k, width, first, length,
                                         A, descA, descB, panel, work,
                                         w, backend_bytes - used, host_work, host_bytes);
            const int start = num_loc(first, nb, coord, source, procs);
            const int count = num_loc(first + length, nb, coord, source, procs) - start;
            if (count > 0 && (left ? nl : ml) > 0) {
                if (left)
                    blas_impl::gemm_run<Backend>(handle, 'N', 'N', count, nl, width, alpha,
                        panel, count, block_B, width, T(1), B + start, ldb);
                else
                    blas_impl::gemm_run<Backend>(handle, 'N', 'N', ml, count, width, alpha,
                        block_B, ml, panel, width, T(1), B + std::size_t(start) * ldb, ldb);
            }
        }
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(handle) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(handle); } catch (...) {}
        throw;
    }
}

} // namespace ptrmm_impl

template <DdlaBackend Backend, typename T>
void ptrmm(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag,
           int m, int n, const T& alpha,
           const T* A, const int* descA, T* B, const int* descB)
{
    const bool noop =
        ptrmm_impl::ptrmm_check<Backend>(handle, side, uplo, trans, diag, m, n, descA, descB);
    if (noop) return;
    side=static_cast<char>(std::toupper(static_cast<unsigned char>(side)));
    uplo=static_cast<char>(std::toupper(static_cast<unsigned char>(uplo)));
    trans=static_cast<char>(std::toupper(static_cast<unsigned char>(trans)));
    diag=static_cast<char>(std::toupper(static_cast<unsigned char>(diag)));
    std::size_t backend_bytes = 0, host_bytes = 0;
    ptrmm_impl::ptrmm_workspace_size<T>(handle, side, trans, m, n, alpha == T(0), descA, descB,
                                        &backend_bytes, &host_bytes);
    vector<char> backend_work(handle, backend_bytes);
    std::vector<char> host_work(host_bytes);
    ptrmm_impl::ptrmm_run<Backend, T>(handle, side, uplo, trans, diag, m, n,
                                      alpha, A, descA, B, descB,
                                      backend_work.data(), backend_bytes, host_work.data(), host_bytes);
}

#define INSTANTIATE_PTRMM(BACKEND, TYPE) \
    template void ptrmm<BACKEND, TYPE>(const DdlaHandle_t&, char, char, char, char, \
        int, int, const TYPE&, const TYPE*, const int*, TYPE*, const int*); \
    template void ptrmm_impl::ptrmm_run<BACKEND, TYPE>(const DdlaHandle_t&, \
        char, char, char, char, int, int, const TYPE&, const TYPE*, const int*, TYPE*, const int*, \
        void*, std::size_t, void*, std::size_t)
#define INSTANTIATE_PTRMM_CHECK(BACKEND) \
    template bool ptrmm_impl::ptrmm_check<BACKEND>(const DdlaHandle_t&, \
        char, char, char, char, int, int, const int*, const int*)
#define INSTANTIATE_PTRMM_SIZE(TYPE) \
    template void ptrmm_impl::ptrmm_workspace_size<TYPE>(const DdlaHandle_t&, \
        char, char, int, int, bool, const int*, const int*, std::size_t*, std::size_t*)
INSTANTIATE_PTRMM_CHECK(DdlaBackend::CPU);
INSTANTIATE_PTRMM(DdlaBackend::CPU, float);
INSTANTIATE_PTRMM(DdlaBackend::CPU, double);
INSTANTIATE_PTRMM(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PTRMM(DdlaBackend::CPU, std::complex<double>);
#if DDLA_HAS_GPU
INSTANTIATE_PTRMM_CHECK(DdlaBackend::GPU);
INSTANTIATE_PTRMM(DdlaBackend::GPU, float);
INSTANTIATE_PTRMM(DdlaBackend::GPU, double);
INSTANTIATE_PTRMM(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PTRMM(DdlaBackend::GPU, std::complex<double>);
#endif
INSTANTIATE_PTRMM_SIZE(float);
INSTANTIATE_PTRMM_SIZE(double);
INSTANTIATE_PTRMM_SIZE(std::complex<float>);
INSTANTIATE_PTRMM_SIZE(std::complex<double>);
#undef INSTANTIATE_PTRMM_SIZE
#undef INSTANTIATE_PTRMM_CHECK
#undef INSTANTIATE_PTRMM

} // namespace ddla
