// ---------------------------------------------------------------------------
// Private implementation bundle of pgeqrf (NOT installed): ONE
// implementation of the distributed Householder QR (uplo = 'U', A = Q R) and QL
// (uplo = 'L', A = Q L) factorizations and of the stage-1 panel of the Hermitian
// eigensolver (phe2hb_factor_panel).
//
//   qr_panel_gathered   (pgeqr2_panel.cpp) -- the PANEL step, both uplo, DEFAULT:
//                       gather the panel rows to the root, factor the whole m_w x jb
//                       window there with the single-process ddla::geqrf(uplo) (LAPACK
//                       ?geqrf / ?geqlf on the CPU, cuSOLVER / hipSOLVER on the GPU),
//                       form T (the LARFT part: ddla::larft, src/larft.cpp: ?larft on the
//                       CPU, herk + qr_build_T on the GPU), scatter the rows and T back,
//                       build V and tau.
//   qr_panel_distributed (pgeqr2_panel.cpp) -- the same PANEL step, both uplo, as an
//                       OPT-IN (DDLA_QR_PANEL=inplace): factor the jb columns IN
//                       PLACE on the distributed A, one column at a time (local dot
//                       or nrm2 + one Col allreduce for the norm, vendor scal / gemv /
//                       gerc, one more Col allreduce for the in-panel update), then V
//                       and T (herk + a Col allreduce + the host recurrence).  No
//                       panel column is ever gathered; every column costs a host
//                       round trip, so it is slower than the gathered panel on the
//                       GPU (see the 2026-10-20 section of pgeqrf-development.md).
//   qr_larfb_left_adj   (qr_larfb.cpp)     -- the BLAS-3 UPDATE step:
//                       C := Q_panel^H C = C - V T^H (V^H C) on the local
//                       trailing columns (ZLARFB with TRANS = 'C'/'T'); T is
//                       upper triangular for QR, lower for QL.
//   qr_block_plain      (pgeqrf.cpp)       -- the block DRIVER: the panel loop
//                       (QR: left to right, trailing columns to the right of the
//                       panel; QL: right to left, trailing columns to its left),
//                       and the Row broadcast of [V|T].
//   qr_make_plan / qr_panels / qr_work_bytes / qr_carve_work / qr_make_ctx
//                       (pgeqrf.cpp) -- geometry, panel list and workspace layout,
//                       also used by phe2hb_factor_panel (phe2hb_panel.cpp).
//
// Block A(0:m, ja:ja+n), LAPACK conventions.  QR: the reflector of block column
// jj (0-based) has its unit entry at row jj, V below it, R in the upper
// triangle.  QL (?GEQLF): the unit entry is at row m - n + jj, V above it, L
// lower triangular in the bottom n x n block; for m < n only the last
// min(m, n) columns carry reflectors.

// ---------------------------------------------------------------------------
#ifndef DDLA_PGEQRF_IMPL_H
#define DDLA_PGEQRF_IMPL_H

#include <ddla/ddla.h>

#include <cctype>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace ddla {
namespace qr_impl {

// Host conjugation overloads (std::conj of a real argument returns complex).
template <typename U> U conj_h(U v) { return v; }
template <typename U> std::complex<U> conj_h(std::complex<U> v) { return std::conj(v); }

// Geometry of one pgeqrf call: everything the two step functions read besides
// the panel coordinates they are handed.
template <typename T>
struct QrCtx {
    const DdlaHandle_t& handle;
    T* A;           // the distributed matrix (local part)
    int mb, rsrc;     // row block size, first process row
    int lld;          // local leading dimension of A
    int nprows, myprow;
    const int* descA; // full distribution for shared matrix transport
    char adj;         // 'T' (real) or 'C' (complex): the adjoint flag of the BLAS calls
};

// Backend buffers shared by the panel and update steps (allocated and freed by
// the driver).  tau holds the reflector scalars the panels produce: indexed by
// LOCAL column on the public path, by j - ja for the stage-1 adapter (tau_by_block).
template <typename T>
struct QrWorkspace {
    using rt = typename real_type<T>::type;
    // Gathered panel (qr_panel_gathered):
    T* Wg = nullptr;     // gathered panel window (natural order) on the root, when nprows > 1
    T* S = nullptr;      // gather/scatter block staging
    T* tau = nullptr;    // reflector-scalar prefix in the selected backend
    // In-place panel only (qr_panel_distributed; QrWorkOpts::inplace): rec is the norm record
    // [ar, ai, one T-wide slot per process row], w this rank's partial A_rest^H v (jb), VV
    // the local V^H V, and the receive buffers of the Col allreduces:
    T* VV = nullptr;     // in-place panel: this rank's local V^H V: nb*nb
    T* w = nullptr;      // in-place panel: nb
    T* VVr = nullptr;    // V^H V summed over the owner process column: nb*nb
    T* wr = nullptr;     // w summed over the owner process column: nb
    rt* recr = nullptr;  // the record summed over the owner process column
    rt* rec = nullptr;   // the record sent
    T* W = nullptr;      // larfb W: nb * max(1, nloc)
    T* Wp = nullptr;     // larfb Wp: nb * max(1, nloc)
    // Stage-1 hook (phe2hb): when set, every panel's T^H -- the upper-triangular T of
    // phe2hb, T_phe2hb = T_L^H -- is stored here after the [V|T] share step: the slot is
    // ld_tout x ld_tout (zeroed first), the panel fills its jb x jb leading block.
    T* tout = nullptr;
    int ld_tout = 0;
};

// out := T^H (jb x jb; out has leading dimension ldout; entries of out outside the leading
// jb x jb block are left alone).  Stream-ordered on the GPU, no host synchronization.
template <DdlaBackend Backend, typename T>
void qr_store_T_adjoint(const DdlaHandle_t& handle, T* out, int ldout, const T* T_panel, int jb);

// PANEL step on the owner process column: factor the jb panel columns whose
// window is the rows [g0, g0 + m_w) (QR: g0 = first panel column - ja, m_w = m -
// g0; QL: g0 = 0, m_w = rows up to and including the last column's unit entry);
// lc0 = local index of the first panel column, lr0_/lrp_ = this rank's first
// local window row and local window row count, ldv = max(1, lrp_).  Leaves, on
// every owner-column rank: A's panel window in LAPACK storage, the explicit
// native V in V (ldv x jb), T in T_panel (jb x jb; upper for QR, lower for QL) and
// the jb taus at taup (see the long comment in pgeqr2_panel.cpp).  need_T ==
// false skips the T build; it must be true whenever a peer takes tau from diag(T)
// (nprows > 1), anything consumes T (the update, the Row bcast, the stage-1 T^H
// hook) -- see qr_panel_geo.
template <DdlaBackend Backend, typename T>
void qr_panel_gathered(const QrCtx<T>& ctx, QrWorkspace<T>& ws, char uplo,
                       int g0, int m_w, int jb, int j0, int lc0, int lr0_, int lrp_, int ldv,
                       T* V, T* T_panel, T* taup, bool need_T);

// The same panel step, IN PLACE on the distributed A (the opt-in DDLA_QR_PANEL=inplace; same
// arguments and same outputs as qr_panel_gathered: A's panel window in LAPACK storage, the
// explicit V in V, T in T_panel, the jb taus at taup, identical on every owner-column rank).
// The window rows [g0, g0 + m_w) are factored column by column on the owner column's ranks
// with no gather: QR column jj has its unit entry at window row jj (x below it, the panel
// columns to its right are updated); QL runs jb - 1 .. 0 with x above the unit entry at
// window row m_w - jb + jj and updates the columns to its left.  need_T == false skips the T
// build.  Needs the in-place workspace layout (QrWorkOpts::inplace).
template <DdlaBackend Backend, typename T>
void qr_panel_distributed(const QrCtx<T>& ctx, QrWorkspace<T>& ws, char uplo,
                          int g0, int m_w, int jb, int lc0, int lr0_, int lrp_, int ldv,
                          T* V, T* T_panel, T* taup, bool need_T);

// DDLA_QR_PANEL: 'gather' (or unset; the default) selects qr_panel_gathered, 'inplace' selects
// qr_panel_distributed.  Anything else warns on stderr and keeps the default.  Every rank must
// use the same value (a mismatch would run different collectives): pgeqrf checks it with two
// small allreduces, the stage-1 panel reads it per call without a check.
bool qr_panel_inplace_env();

// BLAS-3 UPDATE step: C := Q_panel^H C on the local trailing columns.  t_uplo
// is the triangle of the T factor: 'U' for the QR panel's forward T, 'L' for
// the QL panel's backward T_L (C := C - V T^H (V^H C) either way).
template <DdlaBackend Backend, typename T>
void qr_larfb_left_adj(const QrCtx<T>& ctx, QrWorkspace<T>& ws,
                       int jb, int lr0, int lrp, int ldv, int lct, int nct,
                       T* V, T* T_panel, char t_uplo);

// uplo flag of the public pgeqrf: 'U'/'u' -> 'U' (A = Q R), 'L'/'l' -> 'L'
// (A = Q L); anything else throws.  Every rank sees the same flag, so the throw
// happens on all ranks before any collective.
inline char normalize_uplo(char uplo, const char* who)
{
    const char u = static_cast<char>(std::toupper(static_cast<unsigned char>(uplo)));
    if (u != 'U' && u != 'L')
        throw std::runtime_error(std::string(who) + ": uplo must be 'U' or 'L'");
    return u;
}

// ---- plan: the geometry of one call (identical formulas on every rank) --------

struct QrPlan {
    char uplo = 'U';
    int m = 0, n = 0, ja = 0;                 // the block A(0:m, ja:ja+n)
    int mb = 1, nb = 1, rsrc = 0, csrc = 0;
    int nprows = 1, npcols = 1, myprow = 0, mypcol = 0;
    int jr_lo = 0, jr_hi = 0;                 // global reflector columns [jr_lo, jr_hi):
                                              // QR [ja, ja+k), QL [ja+n-k, ja+n), k = min(m, n)
    int nbws = 1;                             // widest panel the workspaces are sized for:
                                              // QR the descriptor nb, QL the widest actual panel
};

// One panel: global columns [j_lo, j_hi) and its window rows [g0, g0 + m_w).
struct QrPanel { int j_lo, j_hi, g0, m_w; };

// Validate the descriptor against the block and fill the plan (throws on misuse,
// never communicates, never reads the environment).  Every panel is ONE block column of
// the descriptor (or the part of it inside the reflector range), so the panel's
// gather/scatter stays on the owner process column's Col communicator.
QrPlan qr_make_plan(const DdlaHandle_t& handle, const int* descA, char uplo,
                    int m, int n, int ja, const char* who);

// The panels in execution order: QR left to right, QL right to left, each one the
// intersection of a block column [b*nb, (b+1)*nb) with the reflector range.
std::vector<QrPanel> qr_panels(const QrPlan& p);

// Does panel q have columns to update?  QR: columns right of it inside the block;
// QL: columns left of it inside the block.
inline bool qr_has_trailing(const QrPlan& p, const QrPanel& q)
{
    return p.uplo == 'L' ? q.j_lo > p.ja : q.j_hi < p.ja + p.n;
}

// ---- workspace -------------------------------------------------------------

struct QrWorkOpts {
    bool tau_by_block = false;  // tau holds the block's taus indexed j - ja (stage-1
                                // adapter) instead of the LOCAL column index
    bool upd = false;           // allocate the larfb buffers (any panel has trailing columns)
    int nloc_blk = 0;           // local columns of the block (an upper bound is fine)
    bool inplace = false;       // lay out the buffers of qr_panel_distributed instead of
                                // those of qr_panel_gathered
};

// Per-rank workspace of a call: every buffer 256-byte aligned inside ONE
// allocation.  QL layout is monotone in the window rows, the panel width and
// nloc_blk (phe2hb_panel_work_bytes bounds every panel with one plan).
template <typename T>
std::size_t qr_work_bytes(const QrPlan& p, const QrWorkOpts& o);

// Carve ws and the [V|T] slot VT out of `base` (>= qr_work_bytes bytes).
// ws.tau is a workspace buffer on both backends (the public path copies it out to
// the caller's tau).
template <typename T>
void qr_carve_work(const QrPlan& p, const QrWorkOpts& o, void* base,
                   QrWorkspace<T>& ws, T*& VT);

// QrCtx of a call.
template <typename T>
QrCtx<T> qr_make_ctx(const DdlaHandle_t& handle, T* A, const QrPlan& p, int lld, const int* descA);

// ---- the block driver ---------------------------------------------------------

struct QrBlockOpts {
    bool bcast_all = false;     // Row-broadcast every panel's [V|T] even with nothing to update
    bool tau_by_block = false;  // taus at ws.tau + (j_lo - ja) (else + local column)
    bool inplace = false;       // panel step = qr_panel_distributed (workspace laid out with
                                // QrWorkOpts::inplace) instead of qr_panel_gathered
};

// Run the supplied panels in execution order: factor on the owner process column, Row-bcast
// [V|T] (when there are columns to update or bcast_all),
// optionally copy each panel's V into columns [j_lo - ja, j_hi - ja) of vmr
// (ld_vmr >= local window rows; may be null) and store T^H into ws.tout (when set),
// then update the columns to be updated with qr_larfb_left_adj.  VT: the [V|T] slot
// from qr_carve_work.
template <DdlaBackend Backend, typename T>
void qr_block_plain(const QrCtx<T>& ctx, QrWorkspace<T>& ws, T* VT, const QrPlan& p,
                    const std::vector<QrPanel>& panels, const QrBlockOpts& o,
                    T* vmr, int ld_vmr);

} // namespace qr_impl
} // namespace ddla

#endif // DDLA_PGEQRF_IMPL_H
