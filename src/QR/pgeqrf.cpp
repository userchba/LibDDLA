// ---------------------------------------------------------------------------
// Distributed Householder QR (uplo = 'U', ScaLAPACK PxGEQRF semantics) and QL
// (uplo = 'L', PxGEQLF) factorization -- ONE implementation, see the end of this
// header for the source layout.  The description below is written for QR; QL is
// the same machinery run right to left.  Classic right-looking structure, the same
// loop on every grid (a 1x1 grid included):
//
//     for each block column j0 (width jb <= nb):
//         gathered panel -- the owner process column sends its window rows
//                           to ONE rank (the process row owning global row
//                           j0), the root factors the whole window there with
//                           the single-process ddla::geqrf(uplo) (no communication),
//                           builds T, and sends the rows and T back; tau
//                           travels in T's diagonal
//         Row bcast      -- V and T shared with every process column
//         larfb          -- C := Q_panel^H C on the trailing columns
//
// Output convention (LAPACK ?GEQRF): R occupies the upper triangle of the
// leading block; below the diagonal sits V with an implicit unit diagonal,
// so Q = H(1)...H(k), k = min(m,n), with H = I - tau v v^H.  Complex R has a
// real diagonal, and tau entries for global columns >= k are left untouched.
//
// Two panel variants, chosen per call by DDLA_QR_PANEL (every rank must agree; checked here):
//   'gather' (the default)  the panel is gathered on one rank and factored there, described below;
//   'inplace'               the panel is factored column by column IN PLACE on the distributed A
//                           (qr_panel_distributed, pgeqr2_panel.cpp): per column a local dot (nrm2
//                           when the sum of squares could under- or overflow) + ONE Col allreduce
//                           for the norm, the same exact xLARFG scalars on every rank, vendor scal,
//                           and gemv + ONE Col allreduce + gerc for the in-panel update; V^H V by
//                           herk + one allreduce and T by the host LARFT recurrence.  Nothing is
//                           gathered, but every column costs a host round trip: on the V100 it is
//                           1.3-4.6x slower than the gathered panel (2026-10-20 section of
//                           pgeqrf-development.md), so it is the opt-in.
// Everything else (the Row bcast of [V|T], the blocked update) is shared.
//
// Per panel the owner column costs exactly two point-to-point rounds (gather,
// scatter) with the root plus the one Col-scope allreduce of larfb; the jb
// panel columns themselves run without any collective.  On a single-process-row
// grid (or a 1x1 handle) the root factors A in place, QR and QL alike.
//
// Root panel factorization: the window (m_w x jb, natural order, on Wg or in A
// itself) is factored by ddla::geqrf(uplo, ...) (src/geqrf.cpp): LAPACK ?geqrf /
// ?geqlf on the CPU, the vendor geqrf on the GPU (cuSOLVER / hipSOLVER have no
// ?geqlf, so uplo = 'L' runs the QR of the J-reversed copy there).  It leaves R / L
// and V in LAPACK storage and the jb taus, so the window needs no post-processing
// before the scatter.  T (the LARFT part): ddla::larft(direct, ...) on the window ('F' for
// QR, 'B' for QL; src/larft.cpp: LAPACK ?larft on the CPU, V^H V by herk + the qr_build_T
// kernel on the GPU).  T is skipped where nothing consumes it (see qr_panel_geo);
// peers take tau from T's diagonal.
//
// The block-reflector apply is a private helper (no public punmqr/pungqr);
// it follows ZLARFB with TRANS='C'/'T' (zlarfb.f): W = V^H C (Col allreduce),
// W = T^H W (out-of-place TRMM), C -= V W.  V is replicated across the owner
// column by construction (every rank builds its own copy from identical
// gathered data) and broadcast along the process row together with T.
//
// At the end of the call one D2H copies the local tau prefix back (the only
// blocking host wait of the whole factorization on the GPU path).
//
// Source layout: this file holds the shared block driver qr_impl::qr_block_plain
// (panel -> Row broadcast -> update), plan / panel list / workspace layout, and
// the public pgeqrf entry (checks, panel variant, workspace, driver, tau copy-out).
// Every panel is one block column of the descriptor or the part inside the
// reflector range.  Its gather/scatter stays on the owner column's communicator;
// a wider panel on a 1x1 grid is obtained by passing a larger descriptor nb.
// The panel implementations live in pgeqr2_panel.cpp and the BLAS-3 trailing
// update in qr_larfb.cpp; both use QrCtx / QrWorkspace from pgeqrf_impl.h.
//
// QL: A(0:m, ja:ja+n) = Q L, LAPACK ?GEQLF storage: the reflector of block column jj
// has its unit entry at row m - n + jj, V above it, L lower triangular in the bottom
// n x n block (for m < n only the last min(m, n) columns carry reflectors).  The
// panels run RIGHT to LEFT; panel columns [j_lo, j_hi) factor the window rows
// [0, m - n + j_hi - ja) with geqrf('L') (T_L lower triangular, tau_L(c) = diag(T_L)(c)),
// and the block columns LEFT of the panel are updated by Q_panel^H from the left with
// the lower-triangular T_L.  The stage-1 panel of the Hermitian
// eigensolver (phe2hb_factor_panel) runs qr_block_plain with uplo 'L',
// bcast_all and a V copy-out (one panel per call: nbw divides the block size).
// ---------------------------------------------------------------------------

#include <ddla/ddla.h>

#include "ddla_connector.h"
#include "comm_traits.h"
#include "ddla_desc.h"
#include "pgeqrf_impl.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <complex>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ddla {

namespace qr_impl {

// DDLA_QR_PANEL: see pgeqrf_impl.h.
bool qr_panel_inplace_env()
{
    const char* v = std::getenv("DDLA_QR_PANEL");
    if (v == nullptr || *v == '\0' || std::strcmp(v, "gather") == 0) return false;
    if (std::strcmp(v, "inplace") == 0) return true;
    std::fprintf(stderr, "pgeqrf: ignoring DDLA_QR_PANEL=%s (need 'gather' or 'inplace')\n", v);
    return false;
}

// ---- plan ---------------------------------------------------------------------

QrPlan qr_make_plan(const DdlaHandle_t& handle, const int* descA, char uplo,
                    int m, int n, int ja, const char* who)
{
    if (handle == nullptr)
        throw std::runtime_error(std::string(who) + ": null handle");
    check_desc(descA, handle);
    if (m < 0 || n < 0 || ja < 0 || m > descA[DDLA_M_] || ja + n > descA[DDLA_N_])
        throw std::runtime_error(std::string(who)
                                 + ": m/n/ja outside the descriptor dimensions");
    QrPlan p;
    p.uplo = uplo;
    p.m = m; p.n = n; p.ja = ja;
    p.mb = descA[DDLA_MB_]; p.nb = descA[DDLA_NB_];
    p.rsrc = descA[DDLA_RSRC_]; p.csrc = descA[DDLA_CSRC_];
    ddlaGetGridDims(handle, p.nprows, p.npcols);
    ddlaGetGridCoords(handle, p.myprow, p.mypcol);
    const int k = std::min(m, n);
    p.jr_lo = uplo == 'L' ? ja + n - k : ja;
    p.jr_hi = uplo == 'L' ? ja + n : ja + k;
    // Workspace panel width (it fixes buffer offsets, and with them operand alignment): QR
    // sizes every call by the descriptor nb, QL by the widest actual panel.
    p.nbws = 1;
    if (uplo == 'U') {
        p.nbws = p.nb;
    } else {
        for (const QrPanel& q : qr_panels(p)) p.nbws = std::max(p.nbws, q.j_hi - q.j_lo);
    }
    return p;
}

std::vector<QrPanel> qr_panels(const QrPlan& p)
{
    std::vector<QrPanel> v;
    if (p.uplo == 'U') {
        for (int j = p.jr_lo; j < p.jr_hi;) {
            const int g0 = j - p.ja;
            const int j_hi = std::min((j / p.nb + 1) * p.nb, p.jr_hi);   // end of this block column
            v.push_back(QrPanel{j, j_hi, g0, p.m - g0});
            j = j_hi;
        }
    } else {
        for (int j_hi = p.jr_hi; j_hi > p.jr_lo;) {
            const int j_lo = std::max(p.jr_lo, ((j_hi - 1) / p.nb) * p.nb);
            // Window rows [0, mw): the reflector of the panel's LAST column has its
            // unit entry at row mw - 1 (= m - n + (j_hi - 1 - ja)).
            v.push_back(QrPanel{j_lo, j_hi, 0, p.m - p.n + (j_hi - p.ja)});
            j_hi = j_lo;
        }
    }
    return v;
}

// ---- workspace ----------------------------------------------------------------

namespace {

inline std::size_t qr_align(std::size_t bytes)
{
    constexpr std::size_t kAlign = 256;
    return (bytes + kAlign - 1) / kAlign * kAlign;
}

// Elements of the [V|T] slot: V (ldv_max x nbws) followed by T (nbws x nbws).
inline std::size_t qr_slot_elems(const QrPlan& p)
{
    const std::size_t ldv_max = static_cast<std::size_t>(
        std::max(1, num_loc(p.m, p.mb, p.myprow, p.rsrc, p.nprows)));
    const std::size_t jb = static_cast<std::size_t>(p.nbws);
    return ldv_max * jb + jb * jb;
}

struct QrLayout {
    std::size_t wg, s, vt, vv, rec, w, W, Wp, tau, vvr, recr, wr, total;
    std::size_t slot;   // elements per [V|T] slot
};

template <typename T>
QrLayout qr_layout(const QrPlan& p, const QrWorkOpts& o)
{
    using rt = typename real_type<T>::type;
    QrLayout L{};
    const std::size_t jb = static_cast<std::size_t>(p.nbws);
    const std::size_t m = static_cast<std::size_t>(std::max(p.m, 1));
    L.slot = qr_slot_elems(p);
    std::size_t off = 0;
    auto take = [&](std::size_t& slot, std::size_t bytes) {
        slot = off;
        off += qr_align(bytes);
    };
    // The gathered panel: a single process row factors A in place (QR and QL alike, no gathered
    // window); every other call gathers into Wg.  The gathered panel's T (ddla::larft) needs no
    // buffer of this arena (the GPU arm takes its k x k scratch stream-ordered).  The
    // in-place panel (o.inplace) gathers nothing: it needs a (2 + P * NR)-entry norm record (NR =
    // rt entries per T), the local V^H V (VV), w and the allreduce receive buffers placed
    // after the shared ones below; in gathered mode those take zero bytes.
    const bool ip = o.inplace;
    const std::size_t nr = sizeof(T) / sizeof(rt);
    const std::size_t nrec_ip = 2 + static_cast<std::size_t>(p.nprows) * nr;
    take(L.wg, ((!ip && p.nprows > 1) ? m * jb : 1) * sizeof(T));
    take(L.s, ((!ip && p.nprows > 1) ? m * jb : 1) * sizeof(T));
    take(L.vt, L.slot * sizeof(T));
    take(L.vv, (ip ? jb * jb : 1) * sizeof(T));
    take(L.rec, (ip ? nrec_ip : 1) * sizeof(rt));
    take(L.w, (ip ? jb : 1) * sizeof(T));
    const std::size_t wcols = o.upd ? jb * static_cast<std::size_t>(std::max(1, o.nloc_blk)) : 1;
    take(L.W, wcols * sizeof(T));
    take(L.Wp, wcols * sizeof(T));
    const std::size_t ntau = o.tau_by_block
        ? static_cast<std::size_t>(std::max(1, p.n))
        : static_cast<std::size_t>(std::max(1, num_loc(p.jr_hi, p.nb, p.mypcol, p.csrc, p.npcols)));
    take(L.tau, ntau * sizeof(T));
    take(L.vvr, ip ? jb * jb * sizeof(T) : 0);
    take(L.recr, ip ? nrec_ip * sizeof(rt) : 0);
    take(L.wr, ip ? jb * sizeof(T) : 0);
    L.total = off;
    return L;
}

} // namespace

template <typename T>
std::size_t qr_work_bytes(const QrPlan& p, const QrWorkOpts& o)
{
    return qr_layout<T>(p, o).total;
}

template <typename T>
void qr_carve_work(const QrPlan& p, const QrWorkOpts& o, void* base,
                   QrWorkspace<T>& ws, T*& VT)
{
    using rt = typename real_type<T>::type;
    const QrLayout L = qr_layout<T>(p, o);
    char* const b = static_cast<char*>(base);
    ws = QrWorkspace<T>{};
    ws.Wg = reinterpret_cast<T*>(b + L.wg);
    ws.S = reinterpret_cast<T*>(b + L.s);
    ws.VV = reinterpret_cast<T*>(b + L.vv);
    ws.rec = reinterpret_cast<rt*>(b + L.rec);
    ws.w = reinterpret_cast<T*>(b + L.w);
    ws.W = reinterpret_cast<T*>(b + L.W);
    ws.Wp = reinterpret_cast<T*>(b + L.Wp);
    ws.tau = reinterpret_cast<T*>(b + L.tau);
    if (o.inplace) {
        ws.VVr = reinterpret_cast<T*>(b + L.vvr);
        ws.recr = reinterpret_cast<rt*>(b + L.recr);
        ws.wr = reinterpret_cast<T*>(b + L.wr);
    }
    VT = reinterpret_cast<T*>(b + L.vt);
}

template <typename T>
QrCtx<T> qr_make_ctx(const DdlaHandle_t& handle, T* A, const QrPlan& p, int lld, const int* descA)
{
    using rt = typename real_type<T>::type;
    const char adj = std::is_same<rt, T>::value ? 'T' : 'C';
    return QrCtx<T>{handle, A, p.mb, p.rsrc, lld, p.nprows, p.myprow, descA, adj};
}

// ---- the block driver: per-panel steps and the loop ------------------------------

namespace {

// Geometry of one panel on this rank (identical formulas on every rank).
template <typename T>
struct QrPanelGeo {
    int jb;           // panel width
    int pc, lc0;      // owner process column, local index of the panel's first column
    int lr0, lrp;     // this rank's first local window row and number of window rows
    int ldv;          // max(1, lrp)
    T *V, *T_panel;     // the [V | T] slot of the panel: V (ldv x jb), then T (jb x jb)
    T* taup;        // the panel's taus inside ws.tau
    bool trailing;    // the panel has columns to update (QR right of it, QL left of it)
    bool need_T;      // the panel step builds T
};

template <typename T>
QrPanelGeo<T> qr_panel_geo(const QrPlan& p, const QrPanel& q, const QrWorkspace<T>& ws,
                           const QrBlockOpts& o, T* slot)
{
    QrPanelGeo<T> g;
    g.jb = q.j_hi - q.j_lo;
    // One block column per panel: the owner process column below is then the owner of
    // EVERY column of the panel, and its columns are contiguous locally.
    if (q.j_lo / p.nb != (q.j_hi - 1) / p.nb)
        throw std::logic_error("pgeqrf: a panel crosses a block-column boundary");
    g.pc = indxg2p(q.j_lo, p.nb, p.csrc, p.npcols);
    g.lc0 = indxg2l(q.j_lo, p.nb, p.npcols);
    g.lr0 = num_loc(q.g0, p.mb, p.myprow, p.rsrc, p.nprows);
    g.lrp = num_loc(q.g0 + q.m_w, p.mb, p.myprow, p.rsrc, p.nprows) - g.lr0;
    g.ldv = std::max(1, g.lrp);
    g.V = slot;
    g.T_panel = g.V + static_cast<std::size_t>(g.ldv) * g.jb;
    g.taup = ws.tau + (o.tau_by_block ? q.j_lo - p.ja : g.lc0);
    g.trailing = qr_has_trailing(p, q);   // global: all ranks agree
    // T is consumed by the trailing larfb, by the Row bcast (npcols > 1), by the peers
    // (they take tau from diag(T), so nprows > 1) and by the stage-1 T^H hook (tout); a
    // panel where none of these applies (a 1x1 grid's last panel) skips the T build.
    g.need_T = g.trailing || p.nprows > 1 || p.npcols > 1 || ws.tout != nullptr;
    return g;
}

// Step 1, the PANEL (the GEQR2 + LARFT part): the gathered panel (gather, factor on the root,
// scatter) or, with `inplace`, the in-place column algorithm; leaves V, T and tau on every rank
// of the owner process column.
template <DdlaBackend Backend, typename T>
void qr_block_factor(const QrCtx<T>& ctx, QrWorkspace<T>& ws, const QrPlan& p,
                     const QrPanel& q, const QrPanelGeo<T>& g, bool inplace)
{
    if (inplace)
        qr_panel_distributed<Backend, T>(ctx, ws, p.uplo, q.g0, q.m_w, g.jb, g.lc0, g.lr0, g.lrp,
                                         g.ldv, g.V, g.T_panel, g.taup, g.need_T);
    else
        qr_panel_gathered<Backend, T>(ctx, ws, p.uplo, q.g0, q.m_w, g.jb, q.j_lo, g.lc0, g.lr0, g.lrp,
                                      g.ldv, g.V, g.T_panel, g.taup, g.need_T);
}

// Step 2, SHARE the panel: Row-broadcast [V | T] from the owner process column when
// another process column has columns to update (or bcast_all), plus the phe2hb hooks
// (V copied out into vmr, T^H stored into the caller's T slot ws.tout).
template <DdlaBackend Backend, typename T>
void qr_block_share(const QrCtx<T>& ctx, const QrWorkspace<T>& ws, const QrPlan& p, const QrPanel& q,
                    const QrPanelGeo<T>& g, const QrBlockOpts& o, T* vmr, int ld_vmr)
{
    const DdlaHandle_t& handle = ctx.handle;
    const bool bc = p.npcols > 1 && (g.trailing || o.bcast_all);
    if (bc) {
        // V and T travel together (contiguous in the slot); the count is
        // uniform on every rank of the row communicator (same myprow).
        commBcast<Backend>(handle, CommScope::Row, g.V,
                           static_cast<std::size_t>(g.ldv) * g.jb
                             + static_cast<std::size_t>(g.jb) * g.jb, g.pc);
    }
    if (ws.tout != nullptr) {
        DDLA_CHECK(ddlaMemset(handle, ws.tout, 0,
                              static_cast<std::size_t>(ws.ld_tout) * ws.ld_tout * sizeof(T)));
        qr_store_T_adjoint<Backend, T>(handle, ws.tout, ws.ld_tout, g.T_panel, g.jb);
    }
    if (vmr != nullptr && g.lrp > 0 && g.jb > 0)
        DDLA_CHECK(ddlaMemcpy2D(handle, vmr + static_cast<std::size_t>(q.j_lo - p.ja) * ld_vmr,
                                static_cast<std::size_t>(ld_vmr) * sizeof(T), g.V,
                                static_cast<std::size_t>(g.ldv) * sizeof(T),
                                static_cast<std::size_t>(g.lrp) * sizeof(T), static_cast<std::size_t>(g.jb)));
}

// Step 3's column range: this rank's local columns that panel q updates -- QR right
// of the panel, QL left of it (rows of the panel's window either way).  Returns false
// when the panel has no columns to update or this rank owns none of them.
inline bool qr_block_trailing(const QrPlan& p, const QrPanel& q, int& lct, int& nct)
{
    if (!qr_has_trailing(p, q)) return false;
    const bool ql = (p.uplo == 'L');
    lct = ql ? num_loc(p.ja, p.nb, p.mypcol, p.csrc, p.npcols)
             : num_loc(q.j_hi, p.nb, p.mypcol, p.csrc, p.npcols);
    nct = ql ? num_loc(q.j_lo, p.nb, p.mypcol, p.csrc, p.npcols) - lct
             : num_loc(p.ja + p.n, p.nb, p.mypcol, p.csrc, p.npcols) - lct;
    return nct > 0;
}

} // namespace

// ---- the loop: panel -> share -> update ----------------------------------------
// Every grid, a 1x1 grid included.  One [V | T] slot, one stream, one larfb per panel (on a
// multi-rank grid the update's Col allreduce must not be split).
template <DdlaBackend Backend, typename T>
void qr_block_plain(const QrCtx<T>& ctx, QrWorkspace<T>& ws, T* VT, const QrPlan& p,
                    const std::vector<QrPanel>& panels, const QrBlockOpts& o,
                    T* vmr, int ld_vmr)
{
    for (const QrPanel& q : panels) {
        const QrPanelGeo<T> g = qr_panel_geo(p, q, ws, o, VT);
        if (p.mypcol == g.pc)
            qr_block_factor<Backend, T>(ctx, ws, p, q, g, o.inplace);   // 1) panel
        qr_block_share<Backend, T>(ctx, ws, p, q, g, o, vmr, ld_vmr); // 2) [V | T] along the row
        int lct = 0, nct = 0;
        if (qr_block_trailing(p, q, lct, nct))                          // 3) BLAS-3 update
            qr_larfb_left_adj<Backend, T>(ctx, ws, g.jb, g.lr0, g.lrp, g.ldv, lct, nct,
                                          g.V, g.T_panel, p.uplo);
    }
}

#define INSTANTIATE_QR_SETUP(TYPE)                                               \
    template std::size_t qr_work_bytes<TYPE>(const QrPlan&, const QrWorkOpts&);     \
    template void qr_carve_work<TYPE>(const QrPlan&, const QrWorkOpts&, void*,     \
                                      QrWorkspace<TYPE>&, TYPE*&);              \
    template QrCtx<TYPE> qr_make_ctx<TYPE>(const DdlaHandle_t&, TYPE*,            \
                                          const QrPlan&, int, const int*)

INSTANTIATE_QR_SETUP(float);
INSTANTIATE_QR_SETUP(double);
INSTANTIATE_QR_SETUP(std::complex<float>);
INSTANTIATE_QR_SETUP(std::complex<double>);

#undef INSTANTIATE_QR_SETUP

#define INSTANTIATE_QR_BLOCK(BACKEND, TYPE)                                      \
    template void qr_block_plain<BACKEND, TYPE>(const QrCtx<TYPE>&,               \
        QrWorkspace<TYPE>&, TYPE*, const QrPlan&, const std::vector<QrPanel>&,    \
        const QrBlockOpts&, TYPE*, int)

#if DDLA_HAS_CPU
INSTANTIATE_QR_BLOCK(DdlaBackend::CPU, float);
INSTANTIATE_QR_BLOCK(DdlaBackend::CPU, double);
INSTANTIATE_QR_BLOCK(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_QR_BLOCK(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_QR_BLOCK(DdlaBackend::GPU, float);
INSTANTIATE_QR_BLOCK(DdlaBackend::GPU, double);
INSTANTIATE_QR_BLOCK(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_QR_BLOCK(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_QR_BLOCK

} // namespace qr_impl

// ---- public entry ---------------------------------------------------------------
template <DdlaBackend Backend, typename T>
void pgeqrf(
    const DdlaHandle_t& handle, char uplo,
    int m, int n,
    T* A, const int* descA,
    T* tau,
    int& info)
{
    info = 0;
    if (handle == nullptr)
        throw std::runtime_error("pgeqrf: null handle");
    using namespace qr_impl;
    const char normalized_uplo = normalize_uplo(uplo, "pgeqrf");
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU pgeqrf is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU pgeqrf is not available in this LibDDLA build");
    if (ddlaGetBackend(handle) != Backend)
        throw std::runtime_error("pgeqrf: template backend does not match handle backend");
    const QrPlan p = qr_make_plan(handle, descA, normalized_uplo, m, n, /*ja=*/0, "pgeqrf");
    const int k = p.jr_hi - p.jr_lo;   // min(m, n) reflectors

    // The panel is one block column of the descriptor (jb <= nb on every grid; on a 1x1 grid
    // pass a larger descriptor nb for a wider panel).

    // Ranks that own no reflector column never write tau, so a caller sizing tau as
    // exactly the owned columns may hand them a null pointer.
    const int lo = num_loc(p.jr_lo, p.nb, p.mypcol, p.csrc, p.npcols);
    const int hi = num_loc(p.jr_hi, p.nb, p.mypcol, p.csrc, p.npcols);
    if (k > 0 && tau == nullptr && hi > lo)
        throw std::runtime_error("pgeqrf: null tau array");
    if (k == 0) return;

    const std::vector<QrPanel> panels = qr_panels(p);
    bool upd = false;
    for (const QrPanel& q : panels) upd = upd || qr_has_trailing(p, q);

    // The panel variant (DDLA_QR_PANEL).  Every rank must run the same collectives, so a value that
    // differs between the ranks is an error on all of them (two small allreduces on a multi-rank grid).
    const bool inplace = qr_panel_inplace_env();
    if (p.nprows * p.npcols > 1) {
        int lo = inplace ? 1 : 0, hi = lo;
        MPI_Comm comm = ddlaGetCommunicator(handle);
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &hi, 1, MPI_INT, MPI_MAX, comm));
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &lo, 1, MPI_INT, MPI_MIN, comm));
        if (hi != lo)
            throw std::runtime_error("pgeqrf: DDLA_QR_PANEL differs between the ranks");
    }

    QrWorkOpts wo;
    wo.upd = upd;
    wo.nloc_blk = num_loc(n, p.nb, p.mypcol, p.csrc, p.npcols);
    wo.inplace = inplace;
    void* base = nullptr;
    DDLA_CHECK(ddlaMalloc(handle, &base, qr_work_bytes<T>(p, wo)));
    QrWorkspace<T> ws;
    T* VT = nullptr;
    qr_carve_work<T>(p, wo, base, ws, VT);
    const QrCtx<T> ctx = qr_make_ctx<T>(handle, A, p, descA[DDLA_LLD_], descA);

    QrBlockOpts bo;
    bo.inplace = inplace;
    qr_block_plain<Backend, T>(ctx, ws, VT, p, panels, bo, nullptr, 0);

    // Host tau: one copy of the local reflector-column range (local columns [lo, hi) are
    // contiguous), then one sync on EVERY rank -- on the GPU the only blocking host wait of
    // the call (a CPU handle copies on the host and the sync is a no-op).
    if (hi > lo)
        DDLA_CHECK(ddlaMemcpy(handle, tau + lo, ws.tau + lo, static_cast<std::size_t>(hi - lo) * sizeof(T)));
    DDLA_CHECK(ddlaSynchronize(handle));
    DDLA_CHECK(ddlaFree(handle, base));
}

#define INSTANTIATE_PGEQRF(BACKEND, TYPE)                                          \
    template void pgeqrf<BACKEND, TYPE>(                                           \
        const DdlaHandle_t&, char, int, int, TYPE*,           \
        const int*, TYPE*, int&)

#if DDLA_HAS_CPU
INSTANTIATE_PGEQRF(DdlaBackend::CPU, float);
INSTANTIATE_PGEQRF(DdlaBackend::CPU, double);
INSTANTIATE_PGEQRF(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PGEQRF(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_PGEQRF(DdlaBackend::GPU, float);
INSTANTIATE_PGEQRF(DdlaBackend::GPU, double);
INSTANTIATE_PGEQRF(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PGEQRF(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_PGEQRF

} // namespace ddla
