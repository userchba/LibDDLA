// ---------------------------------------------------------------------------
// Stage-1 panel of the two-stage Hermitian eigensolver (ELPA2 bandred panel
// step) -- a thin adapter over the distributed QL factorization.
//
// The panel A(0:lr_end, panel_start : panel_start + n_cols), lr_end =
// panel_start + n_cols - nbw, IS a QL factorization: panel column lc has its
// reflector's unit entry at row lr_end - n_cols + lc = panel_start + lc - nbw
// (the band edge), i.e. LAPACK ?GEQL2's pivot row m - n + lc.  So the panel is
// factored by qr_block_run with uplo 'L' (pgeqrf.cpp): the window is gathered to ONE
// root rank, factored there by the single-process ddla::geqrf (uplo 'L': LAPACK ?geqlf on the
// CPU, the vendor geqrf on the J-reversed window on the GPU; no communication between
// columns) and scattered back.  DDLA_QR_PANEL=inplace selects the other panel variant of
// qr_block_run instead (columns factored in place on the distributed window, no gather;
// slower on the GPU).  This replaces the former per-column code (two
// Col allreduces + a D2H sync + a host Larfg per panel column), whose
// per-column latency was stage 1's only deficit against ELPA.
//
// Convention map (verified by a serial numpy replica, 2026-10-15): phe2hb's
// Q = I - V T V^H = H_1...H_m equals Q_L^H of the QL factorization, i.e.
//     tau_phe2hb = conj(tau_LAPACK),   T_phe2hb = T_L^H,
// with the same V (v = x / (alpha - beta), unit entry at the band edge) and the
// same real beta.  Real types: no conjugation.  The panel's T_L is on every rank after
// the [V|T] broadcast, so the driver stores T_L^H straight into the caller's tmat slot
// (qr_block_run's tout): stage 1 builds no T of its own.  Output differs from the old
// ELPA-style panel in rounding and in one degenerate case (x == 0 with a real
// negative alpha: LAPACK keeps tau = 0 and alpha, ELPA flipped it with tau = 2);
// every consumer only needs a real band and a consistent (V, tau).
// ---------------------------------------------------------------------------
#include "phe2hb_impl.h"

#include "ddla_connector.h"
#include "ddla_desc.h"
#include "QR/pgeqrf_impl.h"
#include "instantiate.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>

namespace ddla {
namespace spectral_impl {

// Every panel's own window: the window grows with the panel, and the root rank that makes
// the vendor query rotates over the grid, so each panel is sized and the largest is kept.
template <typename T>
void phe2hb_panel_workspace_size(const DdlaHandle_t& handle, const int* descA, int n, int nbw,
                                 bool inplace, T* A, T* tau,
                                 std::size_t* backend_bytes, std::size_t* host_bytes)
{
    *backend_bytes = *host_bytes = 0;
    for (int step = n > 0 ? (n - 1) / nbw : 0; step >= 1; --step) {
        const int panel_start = step * nbw;
        const int n_cols = std::min(n - panel_start, nbw);
        const qr_impl::QrPlan p = qr_impl::qr_geometry(handle, descA, 'L', panel_start + n_cols - nbw,
                                                       n_cols, panel_start);
        std::size_t bytes = 0, host = 0;
        qr_impl::qr_block_workspace_size<T>(handle, p, true, true, inplace, A, descA[DDLA_LLD_], tau, &bytes, &host);
        *backend_bytes = std::max(*backend_bytes, bytes);
        *host_bytes = std::max(*host_bytes, host);
    }
}

template <DdlaBackend Backend, typename T>
void phe2hb_panel_run(const DdlaHandle_t& handle, T* A, const int* descA,
                      int panel_start, int n_cols, int nbw, bool inplace,
                      T* vmr, int ld_vmr, T* tslot,
                      void* backend_work, std::size_t backend_bytes,
                      void* host_work, std::size_t host_bytes)
{
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU phe2hb_panel_run is not available in this LibDDLA build");
    // Window rows [0, lr_end); the QL block is A(0:lr_end, panel columns).  The panel
    // lies inside one block column (nbw divides nb): ONE QL panel, nothing to update
    // inside the block.
    const int lr_end = panel_start + n_cols - nbw;
    const qr_impl::QrPlan p = qr_impl::qr_geometry(handle, descA, 'L', lr_end, n_cols, panel_start);

    // vmr is zeroed once per call; every panel's V fills its column support.
    DDLA_CHECK(ddlaMemset(handle, vmr, 0, static_cast<std::size_t>(ld_vmr) * n_cols * sizeof(T)));

    // The one panel; [V|T_L] is Row-broadcast because the other process columns need V
    // (copied into vmr) and T_L (stored as T_L^H in tslot).
    qr_impl::qr_block_run<Backend, T>(handle, A, descA, p, true, true, inplace, vmr, ld_vmr, tslot, nbw,
                                      backend_work, backend_bytes, host_work, host_bytes);
}

} // namespace spectral_impl

namespace detail {
bool phe2hb_cpu_qr_inplace(const DdlaHandle_t& h,int n,const int* desc,int nbw)
{
    int pr=0,pc=0;ddlaGetGridDims(h,pr,pc);
    const char* env=std::getenv("DDLA_QR_PANEL");
    // Read the existing parser for its recognized values and invalid-value
    // warning. Only an absent/empty override enables the new private default.
    bool inplace=qr_impl::qr_panel_inplace_env();
    if(!env||!*env)
        inplace=phe2hb_cpu_qr_inplace_eligible(n,desc[DDLA_NB_],nbw,pr,pc);
    if(pr>1||pc>1) {
        int lo=inplace?1:0,hi=lo;
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,&lo,1,MPI_INT,MPI_MIN,ddlaGetCommunicator(h)));
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,&hi,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h)));
        if(lo!=hi)throw std::invalid_argument("CPU band reduction QR panel policy differs across ranks");
    }
    return inplace;
}
} // namespace detail

#define INSTANTIATE_PHE2HB_PANEL_SIZE(TYPE)                                             \
    template void spectral_impl::phe2hb_panel_workspace_size<TYPE>(                     \
        const DdlaHandle_t&, const int*, int, int, bool, TYPE*, TYPE*, std::size_t*, std::size_t*)
#define INSTANTIATE_PHE2HB_PANEL(BACKEND, TYPE)                                         \
    template void spectral_impl::phe2hb_panel_run<BACKEND, TYPE>(                       \
        const DdlaHandle_t&, TYPE*, const int*, int, int, int, bool, TYPE*, int, TYPE*, \
        void*, std::size_t, void*, std::size_t)

DDLA_FOR_EACH_TYPE(INSTANTIATE_PHE2HB_PANEL_SIZE)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_PHE2HB_PANEL)

#undef INSTANTIATE_PHE2HB_PANEL
#undef INSTANTIATE_PHE2HB_PANEL_SIZE

} // namespace ddla
