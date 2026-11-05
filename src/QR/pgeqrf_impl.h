// ---------------------------------------------------------------------------
// Private implementation bundle of pgeqrf (NOT installed): ONE
// implementation of the distributed Householder QR (uplo = 'U', A = Q R) and QL
// (uplo = 'L', A = Q L) factorizations and of the stage-1 panel of the Hermitian
// eigensolver (phe2hb).
//
//   qr_panel_gathered   (pgeqr2_panel.cpp) -- the PANEL step, both uplo, DEFAULT:
//                       gather the panel rows to the root, factor the whole m_w x jb
//                       window there with the single-process geqrf(uplo) (LAPACK
//                       ?geqrf / ?geqlf on the CPU, cuSOLVER / hipSOLVER on the GPU),
//                       form T (the LARFT part: larft, src/larft.cpp: ?larft on the
//                       CPU, herk + reflector_build_T on the GPU), scatter the rows and T back,
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
//   punmtr_run         (punmtr.cpp)     -- the BLAS-3 UPDATE step:
//                       C := Q_panel^H C = C - V T^H (V^H C) on the local
//                       trailing columns (ZLARFB with TRANS = 'C'/'T'); T is
//                       upper triangular for QR, lower for QL.
//   qr_block_workspace_size / qr_block_run (pgeqrf.cpp) -- the block DRIVER: the
//                       panel loop (QR: left to right, trailing columns to the right
//                       of the panel; QL: right to left, trailing columns to its
//                       left), and the Row broadcast of [V|T]; its scratch in bytes.
//   qr_make_plan / qr_geometry / qr_panels
//                       (pgeqrf.cpp) -- geometry and panel list, also used by the
//                       stage-1 panel (phe2hb).
//
// Block A(0:m, ja:ja+n), LAPACK conventions.  QR: the reflector of block column
// jj (0-based) has its unit entry at row jj, V below it, R in the upper
// triangle.  QL (?GEQLF): the unit entry is at row m - n + jj, V above it, L
// lower triangular in the bottom n x n block; for m < n only the last
// min(m, n) columns carry reflectors.

// ---------------------------------------------------------------------------
#ifndef DDLA_PGEQRF_IMPL_H
#define DDLA_PGEQRF_IMPL_H

#include "helpers.h"
#include <ddla/ddla.h>

#include <algorithm>
#include <cctype>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace ddla {
namespace qr_impl {

// Buffers shared by the panel and update steps, carved by qr_block_run from the
// caller's scratch (see qr_block_workspace_size).  tau holds the reflector scalars the
// panels produce: indexed by LOCAL column on the public path, by j - ja for the
// stage-1 adapter (tau_by_block).
template <typename T>
struct QrWorkspace {
    using rt = typename real_type<T>::type;
    // Gathered panel (qr_panel_gathered):
    T* Wg = nullptr;     // gathered panel window (natural order) on the root, when nprows > 1
    T* S = nullptr;      // gather/scatter block staging on the root
    void* lib = nullptr;          // the root's geqrf / larft scratch (backend; W and Wp overlay it)
    std::size_t lib_bytes = 0;
    void* blocks = nullptr;       // host: the gather/scatter peer list
    std::size_t blocks_bytes = 0;
    T* tau = nullptr;    // reflector-scalar prefix in the selected backend
    // In-place panel only (qr_panel_distributed; inplace): rec is the norm record
    // [ar, ai, one T-wide slot per process row], w this rank's partial A_rest^H v (jb), VV
    // the local V^H V, and the receive buffers of the Col allreduces:
    T* VV = nullptr;     // in-place panel: this rank's local V^H V: nb*nb
    T* w = nullptr;      // in-place panel: nb
    T* VVr = nullptr;    // V^H V summed over the owner process column: nb*nb
    T* wr = nullptr;     // w summed over the owner process column: nb
    rt* recr = nullptr;  // the record summed over the owner process column
    rt* rec = nullptr;   // the record sent
    // ... and its host state: the record as fetched, the panel's taus, V^H V and T.
    rt* rec_h = nullptr; // 2 + nprows * (sizeof(T) / sizeof(rt))
    T* tau_h = nullptr;  // nb
    T* vv_h = nullptr;   // nb*nb
    T* t_h = nullptr;    // nb*nb
    T* W = nullptr;      // larfb W: nb * max(1, nloc)
    T* Wp = nullptr;     // larfb Wp: nb * max(1, nloc)
    // Stage-1 hook (phe2hb): when set, every panel's T^H -- the upper-triangular T of
    // phe2hb, T_phe2hb = T_L^H -- is stored here after the [V|T] share step: the slot is
    // ld_tout x ld_tout (zeroed first), the panel fills its jb x jb leading block.
    T* tout = nullptr;
    int ld_tout = 0;
};

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
void qr_panel_gathered(const DdlaHandle_t& handle, T* A, const int* descA, QrWorkspace<T>& ws, char uplo,
                       int g0, int m_w, int jb, int j0, int lc0, int lr0_, int lrp_, int ldv,
                       T* V, T* T_panel, T* taup, bool need_T);

// The same panel step, IN PLACE on the distributed A (the opt-in DDLA_QR_PANEL=inplace; same
// arguments and same outputs as qr_panel_gathered: A's panel window in LAPACK storage, the
// explicit V in V, T in T_panel, the jb taus at taup, identical on every owner-column rank).
// The window rows [g0, g0 + m_w) are factored column by column on the owner column's ranks
// with no gather: QR column jj has its unit entry at window row jj (x below it, the panel
// columns to its right are updated); QL runs jb - 1 .. 0 with x above the unit entry at
// window row m_w - jb + jj and updates the columns to its left.  need_T == false skips the T
// build.  Needs the in-place buffers of ws (inplace).
template <DdlaBackend Backend, typename T>
void qr_panel_distributed(const DdlaHandle_t& handle, T* A, const int* descA, QrWorkspace<T>& ws, char uplo,
                          int g0, int m_w, int jb, int lc0, int lr0_, int lrp_, int ldv,
                          T* V, T* T_panel, T* taup, bool need_T);

// DDLA_QR_PANEL: 'gather' (or unset; the default) selects qr_panel_gathered, 'inplace' selects
// qr_panel_distributed.  Anything else warns on stderr and keeps the default.  Every rank must
// use the same value (a mismatch would run different collectives): pgeqrf checks it with two
// small allreduces, the stage-1 panel reads it per call without a check.
bool qr_panel_inplace_env();

// Private QR/QL selector: 'U'/'u' -> 'U' (A = Q R), 'L'/'l' -> 'L'
// (A = Q L); anything else throws. The public entries supply a fixed selector,
// which the collective check agrees before choosing a communication schedule.
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
    int nbws = 1;                             // widest panel the buffers are sized for:
                                              // QR the descriptor nb, QL the widest actual panel
};

// One panel: global columns [j_lo, j_hi) and its window rows [g0, g0 + m_w).
struct QrPanel { int j_lo, j_hi, g0, m_w; };

// The panels in execution order, handed to f one by one: QR left to right, QL right to
// left, each one the intersection of a block column [b*nb, (b+1)*nb) with the reflector
// range.  The QL window rows [0, m_w): the reflector of the panel's LAST column has its
// unit entry at row m_w - 1 (= m - n + (j_hi - 1 - ja)).
template <class F>
inline void qr_for_each_panel(const QrPlan& p, F&& f)
{
    if (p.uplo == 'U') {
        for (int j = p.jr_lo; j < p.jr_hi;) {
            const int j_hi = static_cast<int>(std::min<long long>(
                (static_cast<long long>(j) / p.nb + 1) * p.nb, p.jr_hi));   // end of this block column
            f(QrPanel{j, j_hi, j - p.ja, p.m - (j - p.ja)});
            j = j_hi;
        }
    } else {
        for (int j_hi = p.jr_hi; j_hi > p.jr_lo;) {
            const int j_lo = std::max(p.jr_lo, ((j_hi - 1) / p.nb) * p.nb);
            f(QrPanel{j_lo, j_hi, 0, p.m - p.n + (j_hi - p.ja)});
            j_hi = j_lo;
        }
    }
}

// Validate the descriptor against the block and fill the plan (throws on misuse,
// never communicates, never reads the environment).  Every panel is ONE block column of
// the descriptor (or the part of it inside the reflector range), so the panel's
// gather/scatter stays on the owner process column's Col communicator.
// qr_make_plan = qr_check_args (the throws) + qr_geometry (the plan, no validation).
QrPlan qr_make_plan(const DdlaHandle_t& handle, const int* descA, char uplo,
                    int m, int n, int ja, const char* who);
void qr_check_args(const DdlaHandle_t& handle, const int* descA,
                   int m, int n, int ja, const char* who);
QrPlan qr_geometry(const DdlaHandle_t& handle, const int* descA, char uplo,
                   int m, int n, int ja);

// The panels of qr_for_each_panel as a list.
std::vector<QrPanel> qr_panels(const QrPlan& p);

// Does panel q have columns to update?  QR: columns right of it inside the block;
// QL: columns left of it inside the block.
inline bool qr_has_trailing(const QrPlan& p, const QrPanel& q)
{
    return p.uplo == 'L' ? q.j_lo > p.ja : q.j_hi < p.ja + p.n;
}

// ---- the block driver ---------------------------------------------------------


// The scratch of qr_block_run for plan p on this rank.  A (leading dimension lld) and tau
// are what the LAPACK / vendor workspace queries of the gathered root are made with.  A
// caller that runs several plans on one scratch (the stage-1 panel loop) sizes every plan
// and keeps the largest.  An empty plan needs nothing.
template <typename T>
void qr_block_workspace_size(const DdlaHandle_t& handle, const QrPlan& p, bool bcast_all, bool tau_by_block, bool inplace,
                             T* A, int lld, T* tau,
                             std::size_t* backend_bytes, std::size_t* host_bytes);

// Run every panel of p in execution order: factor on the owner process column, Row-bcast
// [V|T] (when there are columns to update or bcast_all), optionally copy each panel's V
// into columns [j_lo - ja, j_hi - ja) of vmr (ld_vmr >= local window rows; may be null)
// and store its T^H into the tout slot (ld_tout x ld_tout; may be null), then update the
// columns to be updated with the shared punmtr_run.  Returns where the panels left their taus
// (backend memory inside backend_work; see tau_by_block).  No validation, no
// allocation.
template <DdlaBackend Backend, typename T>
T* qr_block_run(const DdlaHandle_t& handle, T* A, const int* descA, const QrPlan& p, bool bcast_all, bool tau_by_block, bool inplace,
                T* vmr, int ld_vmr, T* tout, int ld_tout,
                void* backend_work, std::size_t backend_bytes,
                void* host_work, std::size_t host_bytes);

// ---- pgeqrf = check -> workspace size -> run -------------------------------------


// Component 1 (collective): the arguments on every rank and the DDLA_QR_PANEL agreement.
template <DdlaBackend Backend>
bool pgeqrf_check(const DdlaHandle_t& handle, char uplo, int m, int n,
                        const int* descA, const void* tau);

// Component 2: the block driver's scratch for A(0:m, 0:n) (qr_block_workspace_size);
// nothing for an empty call.
template <typename T>
void pgeqrf_workspace_size(const DdlaHandle_t& handle, char uplo, bool inplace, int m, int n,
                           T* A, const int* descA, T* tau,
                           std::size_t* backend_bytes, std::size_t* host_bytes);

// Component 3: the factorization and the copy of the local taus to tau; ends with a
// synchronize.  No validation, no allocation.
template <DdlaBackend Backend, typename T>
void pgeqrf_run(const DdlaHandle_t& handle, char uplo, bool inplace, int m, int n,
                T* A, const int* descA, T* tau,
                void* backend_work, std::size_t backend_bytes,
                void* host_work, std::size_t host_bytes);

} // namespace qr_impl
} // namespace ddla

#endif // DDLA_PGEQRF_IMPL_H
