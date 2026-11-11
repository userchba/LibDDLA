// ---------------------------------------------------------------------------
// Distributed Householder QR (pgeqrf, ScaLAPACK PxGEQRF) and QL
// (pgeqlf, PxGEQLF) factorization -- ONE implementation, see the end of this
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
// QR, 'B' for QL; src/larft.cpp: LAPACK ?larft on the CPU, V^H V by herk + the reflector_build_T
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
// Source layout: this file holds the shared block driver qr_impl::qr_block_run
// (panel -> Row broadcast -> update) and its scratch size, plan / panel list, and
// the public pgeqrf / pgeqlf entries (checks, scratch, driver, tau copy-out).
// Every panel is one block column of the descriptor or the part inside the
// reflector range.  Its gather/scatter stays on the owner column's communicator;
// a wider panel on a 1x1 grid is obtained by passing a larger descriptor nb.
// The panel implementations live in pgeqr2_panel.cpp and the BLAS-3 trailing
// shared update in punmtr.cpp; both use matrix arguments and QrWorkspace from pgeqrf_impl.h.
//
// QL: A(0:m, ja:ja+n) = Q L, LAPACK ?GEQLF storage: the reflector of block column jj
// has its unit entry at row m - n + jj, V above it, L lower triangular in the bottom
// n x n block (for m < n only the last min(m, n) columns carry reflectors).  The
// panels run RIGHT to LEFT; panel columns [j_lo, j_hi) factor the window rows
// [0, m - n + j_hi - ja) with geqrf('L') (T_L lower triangular, tau_L(c) = diag(T_L)(c)),
// and the block columns LEFT of the panel are updated by Q_panel^H from the left with
// the lower-triangular T_L.  The stage-1 panel of the Hermitian
// eigensolver (phe2hb) runs qr_block_run with uplo 'L',
// bcast_all and a V copy-out (one panel per call: nbw divides the block size).
// ---------------------------------------------------------------------------

#include <ddla/ddla.h>

#include "ddla_connector.h"
#include "comm_traits.h"
#include "ddla_desc.h"
#include "pgeqrf_impl.h"
#include "punmtr_impl.h"
#if DDLA_HAS_GPU
#include "pgeqrf_kernels.h"
#endif
#include "check_agree.h"
#include "lapack_impl.h"
#include "matrix_transfer.h"
#include "plan_util.h"
#include "require_backend.h"
#include "instantiate.h"

#include <algorithm>
#include <cassert>
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

void qr_check_args(const DdlaHandle_t& handle, const int* descA,
                   int m, int n, int ja, const char* who)
{
    if (handle == nullptr)
        throw std::runtime_error(std::string(who) + ": null handle");
    check_desc(descA, handle);
    if (m < 0 || n < 0 || ja < 0 || m > descA[DDLA_M_] || static_cast<long long>(ja) + n > descA[DDLA_N_])
        throw std::runtime_error(std::string(who)
                                 + ": m/n/ja outside the descriptor dimensions");
}

QrPlan qr_make_plan(const DdlaHandle_t& handle, const int* descA, char uplo,
                    int m, int n, int ja, const char* who)
{
    qr_check_args(handle, descA, m, n, ja, who);
    return qr_geometry(handle, descA, uplo, m, n, ja);
}

QrPlan qr_geometry(const DdlaHandle_t& handle, const int* descA, char uplo,
                   int m, int n, int ja)
{
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
        qr_for_each_panel(p, [&](const QrPanel& q) { p.nbws = std::max(p.nbws, q.j_hi - q.j_lo); });
    }
    return p;
}

std::vector<QrPanel> qr_panels(const QrPlan& p)
{
    std::vector<QrPanel> v;
    qr_for_each_panel(p, [&](const QrPanel& q) { v.push_back(q); });
    return v;
}

// ---- workspace ----------------------------------------------------------------

namespace {

// Elements of the [V|T] slot: V (ldv_max x nbws) followed by T (nbws x nbws).
inline std::size_t qr_slot_elems(const QrPlan& p)
{
    const std::size_t ldv_max = static_cast<std::size_t>(
        std::max(1, num_loc(p.m, p.mb, p.myprow, p.rsrc, p.nprows)));
    const std::size_t jb = static_cast<std::size_t>(p.nbws);
    return ldv_max * jb + jb * jb;
}

// Elements of the gathered window Wg and of the root's block staging S, each: the m x nbws
// window when the gathered panel runs on more than one process row.  A single process row
// factors A in place (QR and QL alike, no gathered window); the in-place panel gathers
// nothing.
inline std::size_t qr_window_elems(const QrPlan& p, bool inplace)
{
    return !inplace && p.nprows > 1
        ? static_cast<std::size_t>(std::max(p.m, 1)) * static_cast<std::size_t>(p.nbws) : 0;
}

// Elements of tau: the block's taus indexed j - ja (tau_by_block), else this rank's local
// columns up to the last reflector column.
inline std::size_t qr_tau_elems(const QrPlan& p, bool tau_by_block)
{
    return tau_by_block
        ? static_cast<std::size_t>(std::max(1, p.n))
        : static_cast<std::size_t>(std::max(1, num_loc(p.jr_hi, p.nb, p.mypcol, p.csrc, p.npcols)));
}

// Entries (rt) of the in-place panel's norm record: [ar, ai, one T-wide slot per process row].
template <typename T>
inline std::size_t qr_record_elems(const QrPlan& p)
{
    using rt = typename real_type<T>::type;
    return 2 + static_cast<std::size_t>(p.nprows) * (sizeof(T) / sizeof(rt));
}

// Elements of the update's W and of its reduced copy Wp, each: nbws x the block's local
// columns, when any panel has columns to update.
inline std::size_t qr_update_elems(const QrPlan& p)
{
    bool upd = false;
    qr_for_each_panel(p, [&](const QrPanel& q) { upd = upd || qr_has_trailing(p, q); });
    if (!upd) return 0;
    const int nloc = num_loc(p.ja + p.n, p.nb, p.mypcol, p.csrc, p.npcols)
                   - num_loc(p.ja, p.nb, p.mypcol, p.csrc, p.npcols);
    return static_cast<std::size_t>(p.nbws) * static_cast<std::size_t>(std::max(1, nloc));
}

} // namespace

// Backend, in carving order: Wg and S (the gathered panel on more than one process row), the
// [V|T] slot, tau, the in-place panel's VV, rec and w with their Col-allreduce receive
// buffers VVr, recr and wr, then ONE region for what is never live at the same time: the
// geqrf and the larft scratch of every gathered panel this rank roots, and the update's
// W | Wp.  Host: the gather/scatter peer list (h->nprocs - 1 blocks bound every window), or
// the in-place panel's record, taus, V^H V and T.
template <typename T>
void qr_block_workspace_size(const DdlaHandle_t& handle, const QrPlan& p, bool bcast_all, bool tau_by_block, bool inplace,
                             T* A, int lld, T* tau,
                             std::size_t* backend_bytes, std::size_t* host_bytes)
{
    using rt = typename real_type<T>::type;
    *backend_bytes = *host_bytes = 0;
    if (p.jr_lo == p.jr_hi) return;
    const std::size_t jb = static_cast<std::size_t>(p.nbws);
    std::size_t bytes = detail::padded_bytes<T>(qr_window_elems(p, inplace))
                      + detail::padded_bytes<T>(qr_window_elems(p, inplace))
                      + detail::padded_bytes<T>(qr_slot_elems(p))
                      + detail::padded_bytes<T>(qr_tau_elems(p, tau_by_block));
    if (inplace)
        bytes += detail::padded_bytes<T>(jb * jb) + detail::padded_bytes<rt>(qr_record_elems<T>(p))
               + detail::padded_bytes<T>(jb)
               + detail::padded_bytes<T>(jb * jb) + detail::padded_bytes<rt>(qr_record_elems<T>(p))
               + detail::padded_bytes<T>(jb);
    std::size_t shared = 2 * detail::padded_bytes<T>(qr_update_elems(p));
    if (!inplace) {
        // The root of a gathered panel: the rank of the owner process column holding the
        // window's anchor row (QR its first row, QL its last); it factors an m_w x jb window
        // with leading dimension m_w (Wg) or, on one process row, lld (A itself).
        qr_for_each_panel(p, [&](const QrPanel& q) {
            const int pc = indxg2p(q.j_lo, p.nb, p.csrc, p.npcols);
            const int pr = indxg2p(p.uplo == 'L' ? q.g0 + q.m_w - 1 : q.g0, p.mb, p.rsrc, p.nprows);
            if (p.mypcol != pc || p.myprow != pr) return;
            std::size_t lib = 0;
            lapack_impl::geqrf_workspace_size<T>(handle, p.uplo, q.m_w, q.j_hi - q.j_lo, A,
                                                 p.nprows > 1 ? q.m_w : lld, tau, &lib);
            shared = std::max(shared, lib);
            lapack_impl::larft_workspace_size<T>(handle, q.m_w, q.j_hi - q.j_lo, &lib);
            shared = std::max(shared, lib);
        });
    }
    *backend_bytes = bytes + shared;
    if (inplace)
        *host_bytes = detail::padded_bytes<rt>(qr_record_elems<T>(p)) + detail::padded_bytes<T>(jb)
                    + detail::padded_bytes<T>(jb * jb) + detail::padded_bytes<T>(jb * jb);
    else
        *host_bytes = detail::padded_bytes<matrix_impl::Block<T>>(static_cast<std::size_t>(handle->nprocs - 1));
}


namespace {

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
// multi-rank grid the update's Col allreduce must not be split).  The buffers are carved in
// the order qr_block_workspace_size counts them.
template <DdlaBackend Backend, typename T>
T* qr_block_run(const DdlaHandle_t& handle, T* A, const int* descA, const QrPlan& p, bool bcast_all, bool tau_by_block, bool inplace,
                T* vmr, int ld_vmr, T* tout, int ld_tout,
                void* backend_work, std::size_t backend_bytes,
                void* host_work, std::size_t host_bytes)
{
    using rt = typename real_type<T>::type;
    if (p.jr_lo == p.jr_hi) return nullptr;
    const std::size_t jb = static_cast<std::size_t>(p.nbws);
    QrWorkspace<T> ws;
    char* const base = static_cast<char*>(backend_work);
    char* w = base;
    ws.Wg = detail::take<T>(w, qr_window_elems(p, inplace));
    ws.S = detail::take<T>(w, qr_window_elems(p, inplace));
    T* const VT = detail::take<T>(w, qr_slot_elems(p));
    ws.tau = detail::take<T>(w, qr_tau_elems(p, tau_by_block));
    if (inplace) {
        ws.VV = detail::take<T>(w, jb * jb);
        ws.rec = detail::take<rt>(w, qr_record_elems<T>(p));
        ws.w = detail::take<T>(w, jb);
        ws.VVr = detail::take<T>(w, jb * jb);
        ws.recr = detail::take<rt>(w, qr_record_elems<T>(p));
        ws.wr = detail::take<T>(w, jb);
    }
    // The rest: the root's geqrf / larft scratch, and the update's W | Wp from the same start.
    ws.lib = w;
    ws.lib_bytes = backend_bytes - static_cast<std::size_t>(w - base);
    ws.W = detail::take<T>(w, qr_update_elems(p));
    ws.Wp = detail::take<T>(w, qr_update_elems(p));
    assert(static_cast<std::size_t>(w - base) <= backend_bytes);

    char* const hbase = static_cast<char*>(host_work);
    char* hw = hbase;
    if (inplace) {
        ws.rec_h = detail::take<rt>(hw, qr_record_elems<T>(p));
        ws.tau_h = detail::take<T>(hw, jb);
        ws.vv_h = detail::take<T>(hw, jb * jb);
        ws.t_h = detail::take<T>(hw, jb * jb);
    } else {
        ws.blocks = detail::take<matrix_impl::Block<T>>(hw, static_cast<std::size_t>(handle->nprocs - 1));
        ws.blocks_bytes = host_bytes;
    }
    assert(static_cast<std::size_t>(hw - hbase) <= host_bytes);
    ws.tout = tout;
    ws.ld_tout = ld_tout;

    qr_for_each_panel(p, [&](const QrPanel& q) {
        const int jb = q.j_hi - q.j_lo;
        // One block column per panel: the owner process column below is then the owner of
        // EVERY column of the panel, and its columns are contiguous locally.
        assert(q.j_lo / p.nb == (q.j_hi - 1) / p.nb);
        const int pc = indxg2p(q.j_lo, p.nb, p.csrc, p.npcols);
        const int lc0 = indxg2l(q.j_lo, p.nb, p.npcols);
        const int lr0 = num_loc(q.g0, p.mb, p.myprow, p.rsrc, p.nprows);
        const int lrp = num_loc(q.g0 + q.m_w, p.mb, p.myprow, p.rsrc, p.nprows) - lr0;
        const int ldv = std::max(1, lrp);
        T* const V = VT;
        T* const T_panel = V + static_cast<std::size_t>(ldv) * jb;
        T* const taup = ws.tau + (tau_by_block ? q.j_lo - p.ja : lc0);
        const bool trailing = qr_has_trailing(p, q);   // global: all ranks agree
        // T is consumed by the trailing larfb, by the Row bcast (npcols > 1), by the peers
        // (they take tau from diag(T), so nprows > 1) and by the stage-1 T^H hook (tout); a
        // panel where none of these applies (a 1x1 grid's last panel) skips the T build.
        const bool need_T = trailing || p.nprows > 1 || p.npcols > 1 || ws.tout != nullptr;
        if (p.mypcol == pc) {
            if (inplace)
                qr_panel_distributed<Backend, T>(handle, A, descA, ws, p.uplo, q.g0, q.m_w, jb, lc0, lr0, lrp,
                                                 ldv, V, T_panel, taup, need_T);
            else
                qr_panel_gathered<Backend, T>(handle, A, descA, ws, p.uplo, q.g0, q.m_w, jb, q.j_lo, lc0, lr0, lrp,
                                              ldv, V, T_panel, taup, need_T);
        }
        const bool bc = p.npcols > 1 && (trailing || bcast_all);
        if (bc) {
            // V and T travel together (contiguous in the slot); the count is
            // uniform on every rank of the row communicator (same myprow).
            commBcast<Backend>(handle, CommScope::Row, V,
                               static_cast<std::size_t>(ldv) * jb
                                 + static_cast<std::size_t>(jb) * jb, pc);
        }
        if (ws.tout != nullptr) {
            DDLA_CHECK(ddlaMemset(handle, ws.tout, 0,
                                  static_cast<std::size_t>(ws.ld_tout) * ws.ld_tout * sizeof(T)));
            if constexpr (Backend == DdlaBackend::CPU) {
                geam<Backend, T>(handle, 'C', 'N', jb, jb,
                                 T(1), T_panel, jb, T(0), nullptr, jb,
                                 ws.tout, ws.ld_tout);
            } else {
    #if DDLA_HAS_GPU
                // Vendor GEAM changes signed zeros/nonfinite complex components;
                // T^H is an exact copy, so retain the existing GPU primitive.
                detail::qr_adjoint_T<T>(handle, ws.tout, ws.ld_tout, T_panel, jb);
    #endif
            }
        }
        if (vmr != nullptr && lrp > 0 && jb > 0)
            DDLA_CHECK(ddlaMemcpy2D(handle, vmr + static_cast<std::size_t>(q.j_lo - p.ja) * ld_vmr,
                                    static_cast<std::size_t>(ld_vmr) * sizeof(T), V,
                                    static_cast<std::size_t>(ldv) * sizeof(T),
                                    static_cast<std::size_t>(lrp) * sizeof(T), static_cast<std::size_t>(jb)));
        int lct = 0, nct = 0;
        if (qr_block_trailing(p, q, lct, nct)) {                        // 3) BLAS-3 update
            T* block = A;
            if (lrp > 0) block += static_cast<std::size_t>(lr0) + static_cast<std::size_t>(lct) * descA[DDLA_LLD_];

            detail::punmtr_run<Backend,T>(handle,'L',p.uplo,(std::is_same<T, typename real_type<T>::type>::value ? 'T' : 'C'),lrp,nct,jb,ldv,jb,descA[DDLA_LLD_],jb,('L'=='L'?handle->nprows_:handle->npcols_)>1, V, T_panel, block, ws.W, ws.Wp);
        }
    });
    return ws.tau;
}

#define INSTANTIATE_QR_SETUP(TYPE)                                                    \
    template void qr_block_workspace_size<TYPE>(const DdlaHandle_t&, const QrPlan&,   \
        bool, bool, bool, TYPE*, int, TYPE*, std::size_t*, std::size_t*)

DDLA_FOR_EACH_TYPE(INSTANTIATE_QR_SETUP)

#undef INSTANTIATE_QR_SETUP

#define INSTANTIATE_QR_BLOCK(BACKEND, TYPE)                                           \
    template TYPE* qr_block_run<BACKEND, TYPE>(const DdlaHandle_t&, TYPE*, const int*, const QrPlan&,     \
        bool, bool, bool, TYPE*, int, TYPE*, int, void*, std::size_t, void*, std::size_t)

DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_QR_BLOCK)

#undef INSTANTIATE_QR_BLOCK

} // namespace qr_impl

// ---- public entry ---------------------------------------------------------------
//
// pgeqrf = check -> workspace size -> run (the same three components as every routine; the
// stage-1 adapter of the eigensolver uses the block-level functions above directly).
namespace qr_impl {

// Component 1.  Collective: a bad argument on any rank (a null tau on a rank that owns reflector columns, say)
// makes every rank throw, and DDLA_QR_PANEL -- which selects different collectives -- must agree across ranks.
template <DdlaBackend Backend>
bool pgeqrf_check(const DdlaHandle_t& handle, char uplo, int m, int n, const int* descA, const void* tau)
{
    const char* name = uplo == 'L' || uplo == 'l' ? "pgeqlf" : "pgeqrf";
    return detail::agree_check(handle, name, [&](detail::Agree& agree) {
        detail::require_backend<Backend>(handle, name);
        uplo = normalize_uplo(uplo, name);
        agree.mode("QR/QL factorization", int(uplo));
        qr_check_args(handle, descA, m, n, /*ja=*/0, name);
        const QrPlan p = qr_geometry(handle, descA, uplo, m, n, /*ja=*/0);
        const int k = p.jr_hi - p.jr_lo;   // min(m, n) reflectors
        // Ranks that own no reflector column never write tau, so a caller sizing tau as
        // exactly the owned columns may hand them a null pointer.
        const int lo = num_loc(p.jr_lo, p.nb, p.mypcol, p.csrc, p.npcols);
        const int hi = num_loc(p.jr_hi, p.nb, p.mypcol, p.csrc, p.npcols);
        if (k > 0 && tau == nullptr && hi > lo)
            throw std::runtime_error(std::string(name) + ": null tau array");
        bool inplace = false;
        if (k != 0) {
            // The panel variant (DDLA_QR_PANEL).  Every rank must run the same collectives.
            inplace = qr_panel_inplace_env();
            agree.mode("DDLA_QR_PANEL", inplace ? 1 : 0);
        }
        return inplace;
    });
}

// Component 2: the block driver's scratch over A(0:m, 0:n) (qr_block_workspace_size: backend
// Wg | S | [V|T] | tau | the in-place panel's buffers | the root's geqrf / larft scratch
// overlaid by W | Wp; host the peer list or the in-place panel's host state).  The panel is one
// block column of the descriptor (jb <= nb on every grid; on a 1x1 grid pass a larger
// descriptor nb for a wider panel).  The LAPACK / vendor queries of the gathered root are made
// with A (leading dimension LLD) and tau.
template <typename T>
void pgeqrf_workspace_size(const DdlaHandle_t& handle, char uplo, bool inplace, int m, int n,
                           T* A, const int* descA, T* tau,
                           std::size_t* backend_bytes, std::size_t* host_bytes)
{
    *backend_bytes = *host_bytes = 0;
    if (m == 0 || n == 0) return;
    qr_block_workspace_size<T>(handle, qr_geometry(handle, descA, uplo, m, n, /*ja=*/0), false, false, inplace,
                               A, descA[DDLA_LLD_], tau, backend_bytes, host_bytes);
}

// Component 3.
template <DdlaBackend Backend, typename T>
void pgeqrf_run(const DdlaHandle_t& handle, char uplo, bool inplace, int m, int n,
                T* A, const int* descA, T* tau,
                void* backend_work, std::size_t backend_bytes,
                void* host_work, std::size_t host_bytes)
{
    try {
        if (m == 0 || n == 0) return;
        const QrPlan p = qr_geometry(handle, descA, uplo, m, n, /*ja=*/0);

        const T* const taus = qr_block_run<Backend, T>(handle, A, descA, p, false, false, inplace, nullptr, 0, nullptr, 0,
                                                       backend_work, backend_bytes, host_work, host_bytes);

        // Host tau: one copy of the local reflector-column range (local columns [lo, hi) are
        // contiguous), then one sync on EVERY rank -- on the GPU the only blocking host wait of
        // the call (a CPU handle copies on the host and the sync is a no-op).
        const int lo = num_loc(p.jr_lo, p.nb, p.mypcol, p.csrc, p.npcols);
        const int hi = num_loc(p.jr_hi, p.nb, p.mypcol, p.csrc, p.npcols);
        if (hi > lo)
            DDLA_CHECK(ddlaMemcpy(handle, tau + lo, taus + lo, static_cast<std::size_t>(hi - lo) * sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(handle));
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(handle) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(handle); } catch (...) {}
        throw;
    }
}

} // namespace qr_impl

namespace {

template <DdlaBackend Backend, typename T>
void factorize(
    const DdlaHandle_t& handle, char uplo,
    int m, int n,
    T* A, const int* descA,
    T* tau,
    int& info)
{
    info = 0;
    const bool inplace = qr_impl::pgeqrf_check<Backend>(handle, uplo, m, n, descA, tau);
    if (m == 0 || n == 0) return;
    std::size_t backend_bytes = 0, host_bytes = 0;
    qr_impl::pgeqrf_workspace_size<T>(handle, uplo, inplace, m, n, A, descA, tau, &backend_bytes, &host_bytes);
    vector<char> backend_work(handle, backend_bytes);
    std::vector<char> host_work(host_bytes);
    qr_impl::pgeqrf_run<Backend, T>(handle, uplo, inplace, m, n, A, descA, tau,
                                    backend_work.data(), backend_bytes, host_work.data(), host_bytes);
}

} // namespace

template <DdlaBackend Backend, typename T>
void pgeqrf(const DdlaHandle_t& handle, int m, int n,
            T* A, const int* descA, T* tau, int& info)
{
    factorize<Backend, T>(handle, 'U', m, n, A, descA, tau, info);
}

template <DdlaBackend Backend, typename T>
void pgeqlf(const DdlaHandle_t& handle, int m, int n,
            T* A, const int* descA, T* tau, int& info)
{
    factorize<Backend, T>(handle, 'L', m, n, A, descA, tau, info);
}

#define INSTANTIATE_PGEQRF(BACKEND, TYPE)                                          \
    template void pgeqrf<BACKEND, TYPE>(                                           \
        const DdlaHandle_t&, int, int, TYPE*, const int*, TYPE*, int&);             \
    template void pgeqlf<BACKEND, TYPE>(                                           \
        const DdlaHandle_t&, int, int, TYPE*, const int*, TYPE*, int&);                                                  \
    template void qr_impl::pgeqrf_run<BACKEND, TYPE>(                              \
        const DdlaHandle_t&, char, bool, int, int, TYPE*,          \
        const int*, TYPE*, void*, std::size_t, void*, std::size_t)
#define INSTANTIATE_PGEQRF_SIZE(TYPE)                                              \
    template void qr_impl::pgeqrf_workspace_size<TYPE>(                            \
        const DdlaHandle_t&, char, bool, int, int, TYPE*,          \
        const int*, TYPE*, std::size_t*, std::size_t*)

template bool qr_impl::pgeqrf_check<DdlaBackend::CPU>(
    const DdlaHandle_t&, char, int, int, const int*, const void*);
INSTANTIATE_PGEQRF(DdlaBackend::CPU, float);
INSTANTIATE_PGEQRF(DdlaBackend::CPU, double);
INSTANTIATE_PGEQRF(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PGEQRF(DdlaBackend::CPU, std::complex<double>);

#if DDLA_HAS_GPU
template bool qr_impl::pgeqrf_check<DdlaBackend::GPU>(
    const DdlaHandle_t&, char, int, int, const int*, const void*);
INSTANTIATE_PGEQRF(DdlaBackend::GPU, float);
INSTANTIATE_PGEQRF(DdlaBackend::GPU, double);
INSTANTIATE_PGEQRF(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PGEQRF(DdlaBackend::GPU, std::complex<double>);
#endif

DDLA_FOR_EACH_TYPE(INSTANTIATE_PGEQRF_SIZE)

#undef INSTANTIATE_PGEQRF_SIZE
#undef INSTANTIATE_PGEQRF

} // namespace ddla
