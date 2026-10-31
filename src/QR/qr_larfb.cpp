#include "BLAS/blas_impl.h"
// ---------------------------------------------------------------------------
// BLAS-3 update step of pgeqrf (moved out of pgeqrf.cpp, 2026-10-11): see pgeqrf_impl.h.
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>

#include "ddla_connector.h"
#include "comm_traits.h"
#include "pgeqrf_impl.h"

#include <cstddef>
#include <complex>

namespace ddla {
namespace qr_impl {

// qr_larfb_left_adj: C := Q_panel^H C = C - V T^H (V^H C) on the local
// trailing columns (ZLARFB with TRANS='C'/'T').  The W allreduce is
// Col-scope (W's rows sum over the process-column's process rows) and is
// gated only on nct, which is uniform within every process column.
template <DdlaBackend Backend, typename T>
void qr_larfb_left_adj(const QrCtx<T>& ctx, QrWorkspace<T>& ws,
                       int jb, int lr0, int lrp, int ldv, int lct, int nct,
                       T* V, T* T_panel, char t_uplo)
{
    const DdlaHandle_t& handle = ctx.handle;
    T* const A = ctx.A;
    const int lld = ctx.lld, nprows = ctx.nprows;
    const char adj = ctx.adj;
    T* const W = ws.W;
    T* const Wp = ws.Wp;

    T* const Cblk = A + static_cast<std::size_t>(lr0) + static_cast<std::size_t>(lct) * lld;
    // Single-process-row grid (nprows == 1): the owner column IS every
    // process row, so V is replicated everywhere by construction and
    // W = V^H C is already the complete sum -- skip the zero-fill (the
    // gemm overwrites W entirely: beta = 0), the Col allreduce and the
    // Wp double-buffer.  The allreduce/memset stay on multi-row grids;
    // the memset remains only when the local gemm is skipped (lrp == 0)
    // and the allreduce still runs.
    const bool solo = (nprows == 1);
    // A 1 x P rank owns window rows whenever the panel exists (lrp > 0
    // for every panel it hosts; the trailing columns' rows are the same
    // local rows), so the solo gemm is unconditional in practice; the
    // lrp == 0 guard keeps a hypothetical row-less rank correct.
    if (lrp > 0) {
        // gemm with beta = 0 overwrites W completely: no zero-fill.
        blas_impl::gemm_run<Backend,T>(handle, adj, 'N', jb, nct, lrp, T(1),
                         V, ldv, Cblk, lld, T(0), W, jb);
    } else if (!solo) {
        DDLA_CHECK(ddlaMemset(handle, W, 0, static_cast<std::size_t>(jb) * nct * sizeof(T)));
    }
    if (!solo)
        commAllReduce<Backend>(handle, CommScope::Col, W, Wp, jb * nct, cclSum);
    // The unreduced solo result and the multi-row reduced result are each
    // consumed only by TRMM and the update, so use that live buffer in place.
    T* const product = solo ? W : Wp;
    blas_impl::trmm_run<Backend,T>(handle, 'L', t_uplo, adj, 'N', jb, nct, T(1),
                     T_panel, jb, product, jb);
    if (lrp > 0) {
        blas_impl::gemm_run<Backend,T>(handle, 'N', 'N', lrp, nct, jb, T(-1),
                         V, ldv, product, jb, T(1), Cblk, lld);
    }
}

#define INSTANTIATE_QR_LARFB(BACKEND, TYPE)                                         \
    template void qr_larfb_left_adj<BACKEND, TYPE>(                                 \
        const QrCtx<TYPE>&, QrWorkspace<TYPE>&, int, int, int, int, int, int,       \
        TYPE*, TYPE*, char)

#if DDLA_HAS_CPU
INSTANTIATE_QR_LARFB(DdlaBackend::CPU, float);
INSTANTIATE_QR_LARFB(DdlaBackend::CPU, double);
INSTANTIATE_QR_LARFB(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_QR_LARFB(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_QR_LARFB(DdlaBackend::GPU, float);
INSTANTIATE_QR_LARFB(DdlaBackend::GPU, double);
INSTANTIATE_QR_LARFB(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_QR_LARFB(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_QR_LARFB

} // namespace qr_impl
} // namespace ddla
