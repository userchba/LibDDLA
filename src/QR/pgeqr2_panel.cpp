// ---------------------------------------------------------------------------
// Panel step of pgeqrf and of the stage-1 panel (phe2hb_factor_panel) for QR (uplo 'U')
// AND QL (uplo 'L'): qr_panel_gathered (the default: gather, the single-process
// ddla::geqrf, T, scatter), qr_panel_distributed (the opt-in in-place variant) and
// qr_store_T_adjoint.  See pgeqrf_impl.h.  The file is named after LAPACK's column
// algorithm (xGEQR2); there is no public one-panel routine.  (Moved out of pgeqrf.cpp
// 2026-10-11; the QL panel step was folded in 2026-10-17; the gathered root became a
// call of the library QR / QL, geqrf.cpp, 2026-10-21; its T became a call of the library larft,
// larft.cpp, 2026-10-22.)
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>

#include "ddla_connector.h"
#include "comm_traits.h"
#include "ddla_stream.h"
#include "ddla_desc.h"
#include "pgeqrf_kernels.h"
#include "pgeqrf_impl.h"
#include "matrix_transfer.h"

#include <algorithm>
#include <cstddef>
#include <cassert>
#include <cmath>
#include <complex>
#include <type_traits>
#include <vector>

namespace ddla {
namespace qr_impl {

// ---------------------------------------------------------------------------
// qr_tau_from_T: native taus out of diag(T) -- one memcpy2D with width sizeof(T), height jb and
// source pitch (jb+1) elements.  Used by every peer of the owner column (the root's taus come
// straight out of geqrf; T's diagonal is tau for the forward and the backward T alike).
// ---------------------------------------------------------------------------
namespace {
template <typename T>
void qr_tau_from_T(const DdlaHandle_t& handle, T* taup, const T* T_panel, int jb)
{
    if (jb <= 0) return;
    DDLA_CHECK(ddlaMemcpy2D(handle, taup, sizeof(T), T_panel, static_cast<std::size_t>(jb + 1) * sizeof(T),
                            sizeof(T), jb));
}
} // namespace

// ---------------------------------------------------------------------------
// qr_store_T_adjoint: out := T^H (conjugate transpose of the jb x jb block, CPU host loop /
// GPU one-block kernel).  Used by the stage-1 hook (ws.tout).
// ---------------------------------------------------------------------------
template <DdlaBackend Backend, typename T>
void qr_store_T_adjoint(const DdlaHandle_t& handle, T* out, int ldout, const T* T_panel, int jb)
{
    if (jb <= 0) return;
    if constexpr (Backend == DdlaBackend::GPU) {
        detail::qr_adjoint_T<T>(handle, out, ldout, T_panel, jb);
    } else {
        for (int c = 0; c < jb; ++c)
            for (int r = 0; r < jb; ++r)
                out[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * ldout] =
                    conj_h(T_panel[static_cast<std::size_t>(c) + static_cast<std::size_t>(r) * jb]);
    }
}

// ---------------------------------------------------------------------------
// qr_panel_gathered: the PANEL step of pgeqrf and phe2hb_factor_panel, BOTH factorizations --
// factor the jb panel columns on the owner process column.  The window is the
// block of rows [g0, g0 + m_w) of the panel columns, in natural order:
//   uplo 'U' (QR, A = Q R): g0 = j0 (the panel's first column), m_w = m - j0;
//            root = the process row owning global row g0;
//   uplo 'L' (QL, A = Q L): g0 = 0, m_w = the window height (the reflector of the
//            LAST panel column has its unit entry at row m_w - 1); root = the
//            owner of window row m_w - 1.
// On a single process row the root factors A itself (QR and QL alike); otherwise it gathers
// the window rows of its peers into Wg.  The root then factors the whole window with the
// single-process ddla::geqrf(uplo) (LAPACK ?geqrf / ?geqlf on the CPU, the vendor geqrf on
// the GPU; geqrf.cpp), which leaves R / L and V in LAPACK storage and the taus at taup, and
// builds T (the LARFT part: the library larft, 'F' for QR / 'B' for QL; larft.cpp: LAPACK ?larft on
// the CPU, herk + the qr_build_T kernel on the GPU), and scatters the final rows back and bcasts
// T over the column (one Col bcast from the root: every rank takes the same matrix); every
// owner-column rank ends with its explicit V window, T, and the jb taus at taup (QR and QL: the
// root has them from geqrf, every peer takes them from diag(T)).  need_T == false skips the T
// build.  The panel is ONE block column of the descriptor (jb <= nb, never crossing a block
// boundary): the gather/scatter stays on the owner process column's Col communicator.
// ---------------------------------------------------------------------------
template <DdlaBackend Backend, typename T>
void qr_panel_gathered(const QrCtx<T>& ctx, QrWorkspace<T>& ws, char uplo,
                       int g0, int m_w, int jb, int j0, int lc0, int lr0_, int lrp_, int ldv,
                       T* V, T* T_panel, T* taup, bool need_T)
{
    const DdlaHandle_t& handle = ctx.handle;
    T* const A = ctx.A;
    const int mb = ctx.mb, rsrc = ctx.rsrc, lld = ctx.lld;
    const int nprows = ctx.nprows, myprow = ctx.myprow;
    T* const Wg = ws.Wg;
    T* const S = ws.S;
    const bool ql = (uplo == 'L');
    assert(uplo == 'U' || uplo == 'L');
    assert(!ql || g0 == 0);
    // One process row factors A itself (no gathered window); everything else goes through Wg.
    const bool inplace = nprows == 1;
    const int pr0 = indxg2p(ql ? g0 + m_w - 1 : g0, mb, rsrc, nprows);
    T* const Apan = A + static_cast<std::size_t>(lr0_) + static_cast<std::size_t>(lc0) * lld;
    T* const W = inplace ? Apan : Wg;
    const int ldw = inplace ? lld : m_w;
    const bool root = (myprow == pr0);
    const int pc = indxg2p(j0, ctx.descA[DDLA_NB_], ctx.descA[DDLA_CSRC_], handle->npcols_);
    const int root_rank = handle->rc_to_rank(pr0, pc);
    T* const packed = root ? S : V;
    const std::size_t packed_capacity = root ? (nprows > 1 ? std::size_t(m_w) * jb : 0)
                                              : std::size_t(ldv) * jb;
    matrix_impl::gather<Backend,T>(handle, m_w, jb, A, g0+1, j0+1, ctx.descA, root_rank,
                                   W, ldw, packed, packed_capacity);

    // ---- factor the window on the root (every rank of a single-process-row grid): the library
    // QR / QL, then T (larft writes the whole jb x jb block, the other triangle as zeros: T is
    // shipped and transposed as a whole block, the stage-1 hook stores T^H).
    if (root) {
        geqrf<Backend, T>(handle, uplo, m_w, jb, W, ldw, taup);
        if (need_T) larft<Backend, T>(handle, ql ? 'B' : 'F', 'C', m_w, jb, W, ldw, taup, T_panel, jb);
    }

    // The caller groups the scatter sends on root; peers complete the shared scatter
    // receive before unpacking their local rows.  T is ONE matrix for the whole owner
    // column -- every peer takes the same bytes (taus off its diagonal, then the driver
    // row-broadcasts [V | T]) -- so a single Col bcast from the panel root replaces the
    // root's per-peer send loop and the peers' matching receives.
    if (nprows > 1) {
        if (root) commGroupStart<Backend>(handle);
        matrix_impl::scatter<Backend,T>(handle, m_w, jb, A, g0+1, j0+1, ctx.descA, root_rank,
                                        W, ldw, packed, packed_capacity, false);
        if (root) commGroupEnd<Backend>(handle);
        commBcast<Backend>(handle, CommScope::Col, T_panel, std::size_t(jb)*jb, pr0);
        if (!root && lrp_ == 0) DDLA_CHECK(ddlaMemset(handle, V, 0, std::size_t(jb)*sizeof(T)));
    }

    // ---- explicit V and tau on every owner-column rank.  The root (and
    // every rank of a single-process-row grid) builds V straight from the
    // window; peers with rows convert their received block in place.
    const bool from_window = root;   // nprows == 1: the only rank is the root
    if (from_window || lrp_ > 0) {
        if constexpr (Backend == DdlaBackend::GPU) {
            if (from_window)
                detail::qr_build_v<T>(handle, false, ql ? 1 : 0, V, ldv, W, ldw, lrp_, jb, g0, m_w,
                                      lr0_, mb, myprow, rsrc, nprows);
            else
                detail::qr_build_v<T>(handle, true, ql ? 1 : 0, V, ldv, V, ldv, lrp_, jb, g0, m_w,
                                      lr0_, mb, myprow, rsrc, nprows);
        } else {
            const long long total = static_cast<long long>(lrp_) * jb;
            for (long long kk = 0; kk < total; ++kk) {
                const int i = static_cast<int>(kk % lrp_);
                const int c = static_cast<int>(kk / lrp_);
                const int g = detail::qr_l2g(lr0_ + i, mb, myprow, rsrc, nprows);
                const int d = g - g0;
                const int piv = ql ? g0 + m_w - jb + c : g0 + c;
                const std::size_t vo = static_cast<std::size_t>(i)
                                     + static_cast<std::size_t>(c) * ldv;
                T out;
                if (g == piv) out = T(1);
                else if (ql ? g > piv : g < piv) out = T(0);
                else if (from_window)
                    out = W[static_cast<std::size_t>(d) + static_cast<std::size_t>(c) * ldw];
                else out = V[vo];
                V[vo] = out;
            }
        }
    }
    // Non-root owner-column ranks without rows keep the zeroed 1 x jb V slot from above.  The root
    // wrote the taus (geqrf); EVERY peer takes them from T's diagonal -- the lrp == 0 ranks own tau
    // slots too, and the root only wrote its own.
    if (!from_window) qr_tau_from_T<T>(handle, taup, T_panel, jb);
}

// ---------------------------------------------------------------------------
// qr_panel_distributed: the IN-PLACE panel step, the opt-in alternative to qr_panel_gathered
// (DDLA_QR_PANEL=inplace; see pgeqrf_impl.h).  Same arguments, same outputs.
//
// The panel columns are factored IN PLACE on the distributed A, one column at a time, on the
// owner process column's Col communicator (nprows ranks; on a single-process-row grid the
// collectives are skipped, nothing else differs).  Per column (QR; QL is the mirror image, see
// below), anchor = the unit entry's row, x = the rows below it:
//
//   1. every rank: its local x^H x with the vendor dot into its own slot of the record
//      [ar, ai, slot_0 .. slot_{P-1}]; the anchor owner copies alpha into the head of the record.
//      The record is -0.0 padded, so ONE Col SUM allreduce is an exact allgather (v + (-0.0) == v
//      for every v, signed zeros included: alpha = -0.0 keeps its sign);
//   2. the record goes to the host (the one sync per column) and every rank folds it the same
//      way: ||x|| = sqrt of the sum of the slots -- or, when that sum could have under- or
//      overflowed (outside [safmin, 1/safmin], NaN), a second round with the scale-safe vendor
//      nrm2 and a lapy2 chain over its slots -- then the exact xLARFG scalar path
//      (qr_larfg_from_norms: beta = -copysign(lapy3(alpha, ||x||), Re alpha), tau, 1 / (alpha -
//      beta)); tau and beta are therefore bitwise identical on every rank.  The rescale loop of
//      xLARFG (|beta| < safmin: only for norms within 2^53 of the subnormal range) is the
//      distributed one of PxLARFG: scale x by safmin^-1 (an exact power of two), one more norm
//      round for ||safmin^-1 x||, redo the scalars;
//   3. scal: v = x * (1 / (alpha - beta)) on the local rows of x;
//   4. the other panel columns: unit entry planted at the anchor, w = A_rest^H v (gemv on the
//      local rows), ONE Col allreduce of w, A_rest -= conj(tau) v w^H (gerc on the local rows).
//      Nothing is gathered: every rank touches only the rows of A it owns;
//   5. beta replaces the anchor entry (LAPACK ?GEQRF storage: R on and above the diagonal, V
//      below with the unit entry implicit).
//
// After the columns: the explicit native V of the panel's local window rows (qr_build_v, 1 on the
// unit row, 0 beyond it), V^H V by herk on the local V and ONE Col allreduce, and T (LARFT) by the
// textbook recurrence on the host from that jb x jb matrix and the taus -- the same bits on every
// rank of the owner column; T and the taus are copied back to the device.  The Row broadcast of
// [V | T] to the other process columns and the trailing update are the driver's and qr_larfb.cpp's.
//
// QL (A = Q L, ?GEQLF): the column jj of the panel has its unit entry at window row m_w - jb + jj,
// x ABOVE it, and the columns are processed from jb - 1 down to 0, each updating the panel
// columns to its LEFT; V is unit-upper-trapezoidal and T lower triangular (LARFT DIRECT = 'B').
// No reversal of the window is needed: the local row ranges are simply the mirror images.
// ---------------------------------------------------------------------------

namespace {

// LAPACK xLARFT DIRECT = 'F', STOREV = 'C' from vv = V^H V (only the upper triangle is read; ld
// jb): T upper triangular, T(i,i) = tau_i, T(0:i-1, i) = -tau_i T(0:i-1, 0:i-1) vv(0:i-1, i).  Loops
// run down the columns of T (contiguous); the per-element summation order is the textbook one.
template <typename T>
void host_larft_forward(int jb, const T* vv, const T* tau, T* t)
{
    std::fill(t, t + static_cast<std::size_t>(jb) * jb, T(0));
    std::vector<T> y(static_cast<std::size_t>(jb));
    for (int i = 0; i < jb; ++i) {
        if (i > 0 && tau[i] != T(0)) {
            std::fill(y.begin(), y.begin() + i, T(0));
            for (int p = 0; p < i; ++p) {
                const T x = vv[static_cast<std::size_t>(p) + static_cast<std::size_t>(i) * jb];
                const T* const tc = t + static_cast<std::size_t>(p) * jb;
                for (int r = 0; r <= p; ++r) y[r] += tc[r] * x;
            }
            for (int r = 0; r < i; ++r)
                t[static_cast<std::size_t>(r) + static_cast<std::size_t>(i) * jb] = -tau[i] * y[r];
        }
        t[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * jb] = tau[i];
    }
}

// LAPACK xLARFT DIRECT = 'B', STOREV = 'C' (the QL panel): T lower triangular, T(i,i) = tau_i,
// T(i+1:jb-1, i) = -tau_i T(i+1:jb-1, i+1:jb-1) (V(:, i+1:jb-1)^H v_i), where the entries
// v_p^H v_i (p > i) are the conjugates of the upper triangle of vv.
template <typename T>
void host_larft_backward(int jb, const T* vv, const T* tau, T* t)
{
    std::fill(t, t + static_cast<std::size_t>(jb) * jb, T(0));
    std::vector<T> y(static_cast<std::size_t>(jb));
    for (int i = jb - 1; i >= 0; --i) {
        if (i < jb - 1 && tau[i] != T(0)) {
            std::fill(y.begin() + (i + 1), y.end(), T(0));
            for (int p = i + 1; p < jb; ++p) {
                const T x = conj_h(vv[static_cast<std::size_t>(i) + static_cast<std::size_t>(p) * jb]);
                const T* const tc = t + static_cast<std::size_t>(p) * jb;
                for (int r = p; r < jb; ++r) y[r] += tc[r] * x;
            }
            for (int r = i + 1; r < jb; ++r)
                t[static_cast<std::size_t>(r) + static_cast<std::size_t>(i) * jb] = -tau[i] * y[r];
        }
        t[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * jb] = tau[i];
    }
}

} // namespace

// ---------------------------------------------------------------------------
// qr_panel_distributed: see the header comment of this file and pgeqrf_impl.h.
// ---------------------------------------------------------------------------
template <DdlaBackend Backend, typename T>
void qr_panel_distributed(const QrCtx<T>& ctx, QrWorkspace<T>& ws, char uplo,
                          int g0, int m_w, int jb, int lc0, int lr0, int lrp, int ldv,
                          T* V, T* T_panel, T* taup, bool need_T)
{
    using rt = typename real_type<T>::type;
    constexpr bool CPLX = !std::is_same<rt, T>::value;
    constexpr int NR = CPLX ? 2 : 1;   // rt entries per T
    const DdlaHandle_t& handle = ctx.handle;
    T* const A = ctx.A;
    const int mb = ctx.mb, rsrc = ctx.rsrc, lld = ctx.lld;
    const int nprows = ctx.nprows, myprow = ctx.myprow;
    const char adj = ctx.adj;
    const bool ql = (uplo == 'L');
    assert(uplo == 'U' || uplo == 'L');
    assert(!ql || g0 == 0);
    const int g_end = g0 + m_w;
    const int nrec = 2 + nprows * NR;   // [ar, ai, one slot of NR entries per process row]

    // Local row count below a global row index (rows [0, g) of the matrix this rank owns).
    auto loc = [&](int g) { return num_loc(g, mb, myprow, rsrc, nprows); };

    // ---- host state of the panel -------------------------------------------------------------
    std::vector<rt> rec_h(static_cast<std::size_t>(nrec));
    std::vector<T> tau_h(static_cast<std::size_t>(jb), T(0));
    const rt zero2[2] = {rt(0), rt(0)};
    const rt negz[2] = {-rt(0), -rt(0)};
    const T one = T(1);

    // The record sent: every slot -0.0 (the additive identity of the SUM allreduce); a column
    // rewrites only the slots this rank contributes.
    {
        const std::vector<rt> init(static_cast<std::size_t>(nrec), -rt(0));
        DDLA_CHECK(ddlaMemcpy(handle, ws.rec, init.data(), static_cast<std::size_t>(nrec) * sizeof(rt)));
    }

    // One norm round: this rank's rows of x reduced into its record slot -- x^H x by the vendor
    // dot (use_dot) or ||x||_2 by the vendor nrm2 -- the record summed over the owner process
    // column and fetched to the host.  The one blocking wait of a column.
    auto norm_round = [&](const T* x, int nx, bool use_dot) {
        rt* const slot = ws.rec + 2 + NR * myprow;
        if (nx > 0) {
            if (use_dot) dotc<Backend, T>(handle, nx, x, 1, x, 1, reinterpret_cast<T*>(slot));
            else nrm2<Backend, T>(handle, nx, x, 1, slot);
        } else {
            DDLA_CHECK(ddlaMemcpy(handle, slot, zero2, static_cast<std::size_t>(NR) * sizeof(rt)));
        }
        const rt* red = ws.rec;
        if (nprows > 1) {
            commAllReduce<Backend>(handle, CommScope::Col, ws.rec, ws.recr,
                                   static_cast<std::size_t>(nrec), cclSum);
            red = ws.recr;
        }
        DDLA_CHECK(ddlaMemcpy(handle, rec_h.data(), red, static_cast<std::size_t>(nrec) * sizeof(rt)));
        DDLA_CHECK(ddlaSynchronize(handle));
    };
    // ||x||_2 over the whole column.  The first round is the vendor dot (x^H x, summed over the
    // process rows from the last down -- the fixed order every rank uses): two light kernels, but
    // its square can under- or overflow, so a sum outside [safmin, 1/safmin] (or NaN) -- an x
    // with a tiny, huge or non-finite norm -- is redone with the scale-safe vendor nrm2 (slots
    // folded by lapy2).  nx_global == 0 (no x row anywhere) is exactly zero.
    const rt safmin_r = detail::qr_safmin<rt>();
    auto column_norm = [&](const T* x, int nx, int nx_global) -> rt {
        norm_round(x, nx, true);
        rt sumsq = rt(0);
        for (int p = nprows - 1; p >= 0; --p) sumsq += rec_h[2 + NR * p];
        if (sumsq >= safmin_r && sumsq <= rt(1) / safmin_r) return std::sqrt(sumsq);
        if (nx_global <= 0) return rt(0);
        norm_round(x, nx, false);
        rt xn = rt(0);
        for (int p = nprows - 1; p >= 0; --p) xn = detail::qr_lapy2<rt>(xn, rec_h[2 + NR * p]);
        return xn;
    };

    // ---- the columns, one after the other -------------------------------------------------------
    for (int step = 0; step < jb; ++step) {
        const int jj = ql ? jb - 1 - step : step;
        const int ga = ql ? g_end - jb + jj : g0 + jj;          // global row of the unit entry
        const bool mine = (indxg2p(ga, mb, rsrc, nprows) == myprow);
        const int la = indxg2l(ga, mb, nprows);                  // local row of it (owner only)
        // Local row ranges: the reflector's support [s_lo, s_hi) (unit entry + x) and x itself.
        const int s_lo = ql ? loc(g0) : loc(ga);
        const int s_hi = ql ? loc(ga + 1) : loc(g_end);
        const int x_lo = ql ? loc(g0) : loc(ga + 1);
        const int x_hi = ql ? loc(ga) : loc(g_end);
        const int nx = x_hi - x_lo, ns = s_hi - s_lo;
        T* const Acol = A + static_cast<std::size_t>(lc0 + jj) * lld;
        T* const x = Acol + x_lo;

        // alpha into the head of the record (the anchor owner), -0.0 from everyone else.
        if (mine) DDLA_CHECK(ddlaMemcpy(handle, ws.rec, Acol + la, sizeof(T)));
        else DDLA_CHECK(ddlaMemcpy(handle, ws.rec, negz, static_cast<std::size_t>(NR) * sizeof(rt)));

        // ---- the reflector: xLARFG on the distributed column ------------------------------------
        const int nx_global = ql ? ga - g0 : std::max(0, g_end - (ga + 1));
        rt xn = column_norm(x, nx, nx_global);
        const rt ar = rec_h[0];
        const rt ai = CPLX ? rec_h[1] : rt(0);
        detail::QrLarfg<rt, CPLX> s;
        // xns (the norm of the rescaled x) is read only by the trivial test and the rescale path.
        detail::qr_larfg_from_norms<rt, CPLX>(ar, ai, xn, xn, s);
        // The vendor norms are accurate on normal numbers but lose (or flush) subnormal entries, so
        // two cases take a second norm on x scaled up by the power of two rsafmn (exact; the
        // scaled entries are normal again): the xLARFG rescale (|beta| < safmin), and a zero norm
        // next to a tiny alpha, which could be subnormals read as zero.
        if ((!s.trivial && s.knt > 0)
            || (s.trivial && std::max(std::fabs(ar), std::fabs(ai)) < s.safmin)) {
            if (nx > 0) scal<Backend, T>(handle, nx, T(s.rsafmn), x, 1);
            const rt xns = column_norm(x, nx, nx_global);
            xn = xns * s.safmin;   // the true norm of x (a power-of-two scaling)
            detail::qr_larfg_from_norms<rt, CPLX>(ar, ai, xn, xns, s);
            if (!s.trivial && s.knt == 0 && nx > 0)   // scaled although beta turned out fine: undo
                scal<Backend, T>(handle, nx, T(s.safmin), x, 1);
            assert(s.knt <= 1);
        }
        T tau_v;
        if constexpr (CPLX) tau_v = T(s.tau_r, s.tau_i);
        else tau_v = T(s.tau_r);
        tau_h[jj] = tau_v;
        if (s.trivial) continue;   // H = I: tau = 0, the column (alpha and x) stays as it is

        if (nx > 0) {
            T vfac;
            if constexpr (CPLX) vfac = T(s.vfac_r, s.vfac_i);
            else vfac = T(s.vfac_r);
            scal<Backend, T>(handle, nx, vfac, x, 1);   // v = x / (alpha - beta)
        }

        // ---- the other panel columns: A_rest -= conj(tau) v (v^H A_rest) ------------------------
        const int ncr = ql ? jj : jb - jj - 1;
        if (ncr > 0) {
            T* const A_rest = A + static_cast<std::size_t>(s_lo)
                            + static_cast<std::size_t>(ql ? lc0 : lc0 + jj + 1) * lld;
            if (mine) DDLA_CHECK(ddlaMemcpy(handle, Acol + la, &one, sizeof(T)));   // v's unit entry
            if (ns > 0)
                gemv<Backend, T>(handle, adj, ns, ncr, T(1), A_rest, lld, Acol + s_lo, 1, T(0), ws.w, 1);
            else
                DDLA_CHECK(ddlaMemset(handle, ws.w, 0, static_cast<std::size_t>(ncr) * sizeof(T)));
            const T* w = ws.w;
            if (nprows > 1) {
                commAllReduce<Backend>(handle, CommScope::Col, ws.w, ws.wr,
                                       static_cast<std::size_t>(ncr), cclSum);
                w = ws.wr;
            }
            if (ns > 0)
                gerc<Backend, T>(handle, ns, ncr, T(-1) * conj_h(tau_v), Acol + s_lo, 1, w, 1, A_rest, lld);
        }
        // beta replaces the unit entry / alpha.
        if (mine) {
            const T beta_t = T(s.beta);
            DDLA_CHECK(ddlaMemcpy(handle, Acol + la, &beta_t, sizeof(T)));
        }
    }

    // ---- explicit V, the taus, T ---------------------------------------------------------------
    // A rank without window rows keeps a zeroed 1 x jb V slot: the Row broadcast reads it.
    const int piv0 = ql ? g_end - jb : g0;   // global row of column 0's unit entry (CPU V loop)
    T* const Apan = A + static_cast<std::size_t>(lr0) + static_cast<std::size_t>(lc0) * lld;
    if (lrp > 0) {
        if constexpr (Backend == DdlaBackend::GPU) {
            // in_place: every entry reads only its own native element (i, c) of Apan (leading
            // dimension lld); rev == ql selects the unit-upper layout of the QL panel
            detail::qr_build_v<T>(handle, true, ql ? 1 : 0, V, ldv, Apan, lld, lrp, jb, g0, m_w, lr0,
                                  mb, myprow, rsrc, nprows);
        } else {
            for (int c = 0; c < jb; ++c)
                for (int i = 0; i < lrp; ++i) {
                    const int g = detail::qr_l2g(lr0 + i, mb, myprow, rsrc, nprows);
                    const int piv = piv0 + c;
                    T out;
                    if (g == piv) out = T(1);
                    else if (ql ? g > piv : g < piv) out = T(0);
                    else out = Apan[static_cast<std::size_t>(i) + static_cast<std::size_t>(c) * lld];
                    V[static_cast<std::size_t>(i) + static_cast<std::size_t>(c) * ldv] = out;
                }
        }
    } else {
        DDLA_CHECK(ddlaMemset(handle, V, 0, static_cast<std::size_t>(ldv) * jb * sizeof(T)));
    }

    DDLA_CHECK(ddlaMemcpy(handle, taup, tau_h.data(), static_cast<std::size_t>(jb) * sizeof(T)));

    if (!need_T) return;
    // VV = V^H V over the whole window: the local part by herk, summed over the owner column.
    // (The zero fill keeps the lower triangle -- never read -- defined for the allreduce.)
    DDLA_CHECK(ddlaMemset(handle, ws.VV, 0, static_cast<std::size_t>(jb) * jb * sizeof(T)));
    if (lrp > 0)
        herk<Backend, T>(handle, 'U', 'C', jb, lrp, rt(1), V, ldv, rt(0), ws.VV, jb);
    const T* vv = ws.VV;
    if (nprows > 1) {
        commAllReduce<Backend>(handle, CommScope::Col, ws.VV, ws.VVr,
                               static_cast<std::size_t>(jb) * jb, cclSum);
        vv = ws.VVr;
    }
    std::vector<T> vv_h(static_cast<std::size_t>(jb) * jb), t_h(static_cast<std::size_t>(jb) * jb);
    DDLA_CHECK(ddlaMemcpy(handle, vv_h.data(), vv, vv_h.size() * sizeof(T)));
    DDLA_CHECK(ddlaSynchronize(handle));
    if (ql) host_larft_backward<T>(jb, vv_h.data(), tau_h.data(), t_h.data());
    else host_larft_forward<T>(jb, vv_h.data(), tau_h.data(), t_h.data());
    DDLA_CHECK(ddlaMemcpy(handle, T_panel, t_h.data(), t_h.size() * sizeof(T)));
}

#define INSTANTIATE_QR_PANEL_DIST(BACKEND, TYPE)                                    \
    template void qr_panel_distributed<BACKEND, TYPE>(                              \
        const QrCtx<TYPE>&, QrWorkspace<TYPE>&, char, int, int, int, int, int, int, \
        int, TYPE*, TYPE*, TYPE*, bool)

#define INSTANTIATE_QR_PANEL(BACKEND, TYPE)                                         \
    template void qr_panel_gathered<BACKEND, TYPE>(                                 \
        const QrCtx<TYPE>&, QrWorkspace<TYPE>&, char, int, int, int, int, int, int, int, \
        int, TYPE*, TYPE*, TYPE*, bool);                                            \
    template void qr_store_T_adjoint<BACKEND, TYPE>(const DdlaHandle_t&, TYPE*, int, \
                                                    const TYPE*, int)

#if DDLA_HAS_CPU
INSTANTIATE_QR_PANEL_DIST(DdlaBackend::CPU, float);
INSTANTIATE_QR_PANEL_DIST(DdlaBackend::CPU, double);
INSTANTIATE_QR_PANEL_DIST(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_QR_PANEL_DIST(DdlaBackend::CPU, std::complex<double>);
INSTANTIATE_QR_PANEL(DdlaBackend::CPU, float);
INSTANTIATE_QR_PANEL(DdlaBackend::CPU, double);
INSTANTIATE_QR_PANEL(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_QR_PANEL(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_QR_PANEL_DIST(DdlaBackend::GPU, float);
INSTANTIATE_QR_PANEL_DIST(DdlaBackend::GPU, double);
INSTANTIATE_QR_PANEL_DIST(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_QR_PANEL_DIST(DdlaBackend::GPU, std::complex<double>);
INSTANTIATE_QR_PANEL(DdlaBackend::GPU, float);
INSTANTIATE_QR_PANEL(DdlaBackend::GPU, double);
INSTANTIATE_QR_PANEL(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_QR_PANEL(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_QR_PANEL
#undef INSTANTIATE_QR_PANEL_DIST

} // namespace qr_impl
} // namespace ddla
