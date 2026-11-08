// ---------------------------------------------------------------------------
// Stage 5 of the two-stage Hermitian eigensolver: apply the band-stage
// orthogonal transform to Z (ELPA2 trans_ev_band_to_full, compact WY within each panel).
//
// phe2hb produced A_out = Q_total A_pre Q_total^H, i.e. A_pre = Q_total^H
// A_out Q_total, with Q_total = Qp_1 Qp_2 ... Qp_nsteps and
// Qp_i = I - V_i T_i V_i^H (panel i's reflectors, panels combined in
// ascending istep order).  Eigenvectors therefore map as
// Y (of A_out) -> Q_total^H Y (of A_pre), and this routine offers both
// directions:
//
//   trans = 'N'  Z := Q_total   Z  -- loop istep = nsteps .. 1 (current
//                                       first-applied panel is the largest
//                                       istep), each Z := Z - V (T (V^H Z));
//   trans = 'C'  Z := Q_total^H Z  -- loop istep = 1 .. nsteps, each
//                                       Z := Z - V (T^H (V^H Z)).
//
// Both orderings are needed: the eigensolver driver back-transforms with 'C',
// while 'N' is the inverse direction (and the mode the isolation test pins by
// checking that Z = I maps to Q_total).
//
// V is gathered row-distributed from A's reflector-storage region and T is
// read from the replicated tmat slots.
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>

#include "Eigensolver/pvector_redist.h"
#include "Eigensolver/eig_workspace.h"
#include "ddla_connector.h"
#include "BLAS/blas_impl.h"
#include "comm_traits.h"
#include "ddla_stream.h"
#include "ddla_desc.h"
#include "plan_util.h"
#include "punmtr_impl.h"
#include "punmtr_he2hb_grouped.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ddla {

namespace spectral_impl {
UnmtrHe2hbPlan punmtr_he2hb_geometry(
    const DdlaHandle_t& h,
    DdlaBackend backend,
    char trans,
    int n,
    int nbw,
    const int* a,
    const int* z)
{
    UnmtrHe2hbPlan p;
    p.backend = backend;
    p.trans = trans;
    p.n = n;
    p.nbw = nbw;
    p.nb = a[DDLA_NB_];
    p.lldA = a[DDLA_LLD_];
    p.lldZ = z[DDLA_LLD_];
    p.nprows = h->nprows_;
    p.npcols = h->npcols_;
    p.myprow = h->myprow_;
    p.mypcol = h->mypcol_;
    p.nsteps = n ? (n - 1) / nbw : 0;
    p.zcols = num_loc(n, z[DDLA_NB_], p.mypcol, z[DDLA_CSRC_], p.npcols);
    if (p.nsteps) {
        const int last = p.nsteps * nbw;
        const int nc = std::min(n, last + nbw) - last;
        p.ldv = std::max(num_loc(last + nc - nbw, p.nb, p.myprow, 0, p.nprows), 1);
    }
    return p;
}

UnmtrHe2hbPlan punmtr_he2hb_check(
    const DdlaHandle_t& handle,
    DdlaBackend backend,
    char trans,
    int n,
    int nbw,
    const void* A,
    const int* descA,
    const void* tmat,
    const void* Z,
    const int* descZ)
{
    if (!handle)
        throw std::runtime_error("punmtr_he2hb: null handle");
    if (ddlaGetBackend(handle) != backend)
        throw std::runtime_error("punmtr_he2hb: template backend does not match handle backend");
    if (trans != 'N' && trans != 'T' && trans != 'C')
        throw std::invalid_argument("punmtr_he2hb: invalid trans");
    check_desc(descA, handle);
    check_desc(descZ, handle);
    const int nb = descA[DDLA_NB_];
    if (descA[DDLA_MB_] != nb)
        throw std::runtime_error("punmtr_he2hb: requires square blocks for A");
    if (descA[DDLA_RSRC_] != 0 || descA[DDLA_CSRC_] != 0)
        throw std::runtime_error("punmtr_he2hb: requires RSRC == CSRC == 0 for A");
    if (descZ[DDLA_MB_] != nb || descZ[DDLA_RSRC_] != descA[DDLA_RSRC_])
        throw std::runtime_error("punmtr_he2hb: Z row block size and source must match A");
    if (nbw <= 0 || nbw > nb || nb % nbw != 0)
        throw std::runtime_error("punmtr_he2hb: nbw must divide the block size nb");
    if (n < 0 || n > descA[DDLA_M_] || n > descA[DDLA_N_] || n > descZ[DDLA_M_] ||
        n > descZ[DDLA_N_])
        throw std::invalid_argument("punmtr_he2hb: order exceeds descriptor");
    const auto p = punmtr_he2hb_geometry(handle, backend, trans, n, nbw, descA, descZ);
    if (p.nsteps) {
        const int rows = num_loc(n, nb, p.myprow, 0, p.nprows);
        const int cols = num_loc(n, nb, p.mypcol, 0, p.npcols);
        if (!tmat || (rows > 0 && cols > 0 && !A) || (rows > 0 && p.zcols > 0 && !Z))
            throw std::invalid_argument("punmtr_he2hb: missing active input/output");
    }
    return p;
}

namespace {
detail::GroupedStage5Plan gpu_grouped_plan(
    const UnmtrHe2hbPlan& p)
{
    // Pair panels to bound the additional Gram/TRMM work and workspace.
    detail::GroupedStage5Options options;
    options.group_cap = 2;
    return detail::grouped_stage5_geometry(p, options);
}
} // namespace

template <class T>
void punmtr_he2hb_workspace_size(
    const DdlaHandle_t& h, const UnmtrHe2hbPlan& p, std::size_t* backend_bytes)
{
    if (p.backend == DdlaBackend::GPU)
        detail::grouped_stage5_workspace_size<T>(h, gpu_grouped_plan(p), backend_bytes);
    else
        punmtr_he2hb_sequential_workspace_size<T>(h, p, backend_bytes);
}

// [ V | W | W2 ].  V's window is largest for the last panel (istep = nsteps); each panel keeps its own leading
// dimension ld_v = max(l_rows, 1) inside it.  W and W2 are panel-independent.
template <class T>
void punmtr_he2hb_sequential_workspace_size(
    const DdlaHandle_t&, const UnmtrHe2hbPlan& p, std::size_t* backend_bytes)
{
    *backend_bytes = 0;
    if (!p.nsteps)
        return;
    const std::size_t w_elements = static_cast<std::size_t>(p.nbw) * std::max(p.zcols, 1);
    *backend_bytes = detail::padded_bytes<T>(static_cast<std::size_t>(p.ldv) * p.nbw) +
                     2 * detail::padded_bytes<T>(w_elements);
}

template <DdlaBackend Backend, class T>
void punmtr_he2hb_sequential_run(
    const DdlaHandle_t& handle,
    const UnmtrHe2hbPlan& p,
    const T* A,
    const int*,
    const T* tmat,
    T* Z,
    const int*,
    int& info,
    void* backend_work,
    std::size_t backend_bytes)
{
    try {
        static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                      "GPU punmtr_he2hb is not available");
        info = 0;
        if (!p.nsteps)
            return;
        const char trans = p.trans;
        const int n = p.n;
        const int nbw = p.nbw;
        const int nb = p.nb;
        const int lldA = p.lldA;
        const int lldZ = p.lldZ;
        const int nprows = p.nprows;
        const int npcols = p.npcols;
        const int myprow = p.myprow;
        const int mypcol = p.mypcol;
        auto sync_stream = [&]() {
            if constexpr (Backend != DdlaBackend::CPU)
                DDLA_CHECK(ddlaSynchronize(handle));
        };
        // Source for the v(nrow)=1 anchors issued below.  Hoisted out of the panel
        // loop on purpose: the H2D that reads it is issued asynchronously, so a
        // constant local to the loop body would be read after its scope had ended.
        // That only ever worked because CUDA stages small pageable H2D copies
        // synchronously; it is not something to keep relying on.
        const T one_anchor = T(1);

        const int nsteps = p.nsteps;
        const int l_zcols = p.zcols;
        const int ld_w = nbw;

        // Q_total = Qp_1 ... Qp_nsteps, so applying Q_total^H means consuming the
        // panels in ascending istep order with T^H; applying Q_total reverses both.
        const bool adjoint = (trans != 'N');
        const int istep_begin = adjoint ? 1 : nsteps;
        const int istep_end = adjoint ? nsteps + 1 : 0; // exclusive
        const int istep_step = adjoint ? 1 : -1;
        // 'C' and 'T' agree for a real scalar type, but the vendor GEMM entry
        // points are inconsistent about accepting 'C' on real operands (cf. the
        // xSYR2K note in src/her2k.h), so spell the real case explicitly.
        const char t_op =
            adjoint ? (std::is_same<T, typename real_type<T>::type>::value ? 'T' : 'C') : 'N';

        // Panel scratch, sized once for the largest panel instead of allocated
        // once per panel: per-panel cudaMallocAsync/cudaFreeAsync pairs are
        // trimmed by the default pool at every synchronize (release threshold 0)
        // and re-mapped by the next panel (see phe2hb).
        const std::size_t w_elements = static_cast<std::size_t>(ld_w) * std::max(l_zcols, 1);
        char* w = static_cast<char*>(backend_work);
        T* const V = detail::take<T>(
            w, static_cast<std::size_t>(p.ldv) * nbw); // gathered reflectors: l_rows x n_cols
        T* const W =
            detail::take<T>(w, w_elements); // W = V^H Z partial / T W result: nbw x l_zcols
        T* const W2 = detail::take<T>(w, w_elements); // W allreduce recv: nbw x l_zcols
        assert(static_cast<std::size_t>(w - static_cast<char*>(backend_work)) <= backend_bytes);
        (void)backend_bytes;

        for (int istep = istep_begin; istep != istep_end; istep += istep_step) {
            const int panel_start = istep * nbw;
            const int n_cols = std::min(n, panel_start + nbw) - panel_start;
            const int lr_end = panel_start + n_cols - nbw;
            const int l_rows = ddla::num_loc(lr_end, nb, myprow, 0, nprows);
            const int ld_v = std::max(l_rows, 1);

            // ---- gather V from A's storage region (ONE Row bcast) ---------------
            // Zero the whole buffer first: rows beyond a column's support must
            // contribute zero to W = V^H Z.  The panel lies inside one block column, so
            // its n_cols columns have one owner process column and are consecutive local
            // columns of A; V is column-major with stride ld_v, so they form one
            // contiguous broadcast payload.
            DDLA_CHECK(
                ddlaMemset(handle, V, 0, static_cast<std::size_t>(ld_v) * n_cols * sizeof(T)));
            const int root = ddla::indxg2p(panel_start, nb, 0, npcols);
            if (root == mypcol) {
                const int j0 = ddla::indxg2l(panel_start, nb, npcols);
                for (int c = 0; c < n_cols; ++c) {
                    const int nrow_c = (panel_start + c) - nbw;
                    const int lp_c = ddla::num_loc(nrow_c, nb, myprow, 0, nprows);
                    T* vcol = V + static_cast<std::size_t>(c) * ld_v;
                    DDLA_CHECK(ddlaMemcpy(handle,
                                          vcol,
                                          A + static_cast<std::size_t>(j0 + c) * lldA,
                                          sizeof(T) * lp_c));
                }
            }
            // The payload is every column's support (ld_v rows), which is exactly what the
            // non-owner ranks need.
            commBcast<Backend>(
                handle, CommScope::Row, V, static_cast<std::size_t>(ld_v) * n_cols, root);
            // Explicit anchor: v(nrow) = 1 on the pivot-row owner.
            for (int c = 0; c < n_cols; ++c) {
                const int nrow = (panel_start + c) - nbw;
                if (myprow == ddla::indxg2p(nrow, nb, 0, nprows)) {
                    const int lp = ddla::num_loc(nrow, nb, myprow, 0, nprows);
                    T* vcol = V + static_cast<std::size_t>(c) * ld_v;
                    if constexpr (Backend == DdlaBackend::CPU) {
                        vcol[lp] = T(1);
                    } else {
                        DDLA_CHECK(ddlaMemcpy(handle, vcol + lp, &one_anchor, sizeof(T)));
                    }
                }
            }
            sync_stream();

            const T* Tstep = tmat + static_cast<std::size_t>(istep - 1) * nbw * nbw;

            detail::punmtr_run<Backend, T>(handle,
                                           'L',
                                           'U',
                                           t_op,
                                           l_rows,
                                           l_zcols,
                                           n_cols,
                                           ld_v,
                                           nbw,
                                           lldZ,
                                           ld_w,
                                           ('L' == 'L' ? handle->nprows_ : handle->npcols_) > 1,
                                           V,
                                           Tstep,
                                           Z,
                                           W,
                                           W2);
            sync_stream();
        }

    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(handle) == DdlaBackend::GPU)
            try {
                (void)ddlaSynchronize(handle);
            } catch (...) {
            }
        throw;
    }
}

template <DdlaBackend Backend, class T>
void punmtr_he2hb_run(
    const DdlaHandle_t& h,
    const UnmtrHe2hbPlan& p,
    const T* A,
    const int* da,
    const T* tmat,
    T* Z,
    const int* dz,
    int& info,
    void* work,
    std::size_t bytes)
{
    if constexpr (Backend == DdlaBackend::CPU) {
        punmtr_he2hb_sequential_run<Backend, T>(h, p, A, da, tmat, Z, dz, info, work, bytes);
    } else {
        try {
            detail::grouped_stage5_run<Backend, T>(
                h, gpu_grouped_plan(p), A, tmat, Z, info, work, bytes);
            DDLA_CHECK(ddlaSynchronize(h));
        } catch (...) {
            try {
                (void)ddlaSynchronize(h);
            } catch (...) {
            }
            throw;
        }
    }
}

} // namespace spectral_impl

template <DdlaBackend Backend, class T>
void punmtr_he2hb(
    const DdlaHandle_t& handle,
    char trans,
    int n,
    int nbw,
    const T* A,
    const int* descA,
    const T* tmat,
    T* Z,
    const int* descZ,
    int& info)
{
    info = 0;
    const auto p =
        spectral_impl::punmtr_he2hb_check(handle, Backend, trans, n, nbw, A, descA, tmat, Z, descZ);
    std::size_t backend_bytes = 0;
    spectral_impl::punmtr_he2hb_workspace_size<T>(handle, p, &backend_bytes);
    vector<char> backend_work(handle, backend_bytes);
    spectral_impl::punmtr_he2hb_run<Backend, T>(
        handle, p, A, descA, tmat, Z, descZ, info, backend_work.data(), backend_bytes);
}

#define INSTANTIATE_PUNMTR_HE2HB(BACKEND, TYPE)                                                    \
    template void punmtr_he2hb<BACKEND, TYPE>(const DdlaHandle_t&,                                 \
                                              char,                                                \
                                              int,                                                 \
                                              int,                                                 \
                                              const TYPE*,                                         \
                                              const int*,                                          \
                                              const TYPE*,                                         \
                                              TYPE*,                                               \
                                              const int*,                                          \
                                              int&);                                               \
    template void spectral_impl::punmtr_he2hb_run<BACKEND, TYPE>(                                  \
        const DdlaHandle_t&,                                                                       \
        const spectral_impl::UnmtrHe2hbPlan&,                                                      \
        const TYPE*,                                                                               \
        const int*,                                                                                \
        const TYPE*,                                                                               \
        TYPE*,                                                                                     \
        const int*,                                                                                \
        int&,                                                                                      \
        void*,                                                                                     \
        std::size_t);                                                                              \
    template void spectral_impl::punmtr_he2hb_sequential_run<BACKEND, TYPE>(                       \
        const DdlaHandle_t&,                                                                       \
        const spectral_impl::UnmtrHe2hbPlan&,                                                      \
        const TYPE*,                                                                               \
        const int*,                                                                                \
        const TYPE*,                                                                               \
        TYPE*,                                                                                     \
        const int*,                                                                                \
        int&,                                                                                      \
        void*,                                                                                     \
        std::size_t)
#define INSTANTIATE_PUNMTR_HE2HB_SIZE(TYPE)                                                        \
    template void spectral_impl::punmtr_he2hb_workspace_size<TYPE>(                                \
        const DdlaHandle_t&, const spectral_impl::UnmtrHe2hbPlan&, std::size_t*);                  \
    template void spectral_impl::punmtr_he2hb_sequential_workspace_size<TYPE>(                     \
        const DdlaHandle_t&, const spectral_impl::UnmtrHe2hbPlan&, std::size_t*)

INSTANTIATE_PUNMTR_HE2HB(DdlaBackend::CPU, float);
INSTANTIATE_PUNMTR_HE2HB(DdlaBackend::CPU, double);
INSTANTIATE_PUNMTR_HE2HB(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PUNMTR_HE2HB(DdlaBackend::CPU, std::complex<double>);

#if DDLA_HAS_GPU
INSTANTIATE_PUNMTR_HE2HB(DdlaBackend::GPU, float);
INSTANTIATE_PUNMTR_HE2HB(DdlaBackend::GPU, double);
INSTANTIATE_PUNMTR_HE2HB(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PUNMTR_HE2HB(DdlaBackend::GPU, std::complex<double>);
#endif

INSTANTIATE_PUNMTR_HE2HB_SIZE(float);
INSTANTIATE_PUNMTR_HE2HB_SIZE(double);
INSTANTIATE_PUNMTR_HE2HB_SIZE(std::complex<float>);
INSTANTIATE_PUNMTR_HE2HB_SIZE(std::complex<double>);

#undef INSTANTIATE_PUNMTR_HE2HB_SIZE
#undef INSTANTIATE_PUNMTR_HE2HB

} // namespace ddla
