// ---------------------------------------------------------------------------
// Distributed Householder bidiagonalization (ScaLAPACK PxGEBRD semantics):
// A = Q B P^H with B upper bidiagonal (m >= n) or lower bidiagonal (m < n),
// Q = H(0)...H(nq-1) and P = G(0)...G(np-1).
//
// Panels are bounded by min(descriptor NB, 256), independent of matrix size.
// Blocked like LAPACK ?GEBRD: each panel of jb steps runs the ?LABRD
// recurrence, which at step c updates and reflects only column c and row c
// and defers everything else to four panel buffers; the trailing matrix then
// takes ONE local rank-2jb update.
//
// Panel buffers, INTERLEAVED in production order (per step i one left and
// one right column; the pairing VX_i <-> YU_i makes every deferred update
// ONE GEMM over two interleaved views; each Y/X step ONE GEMV over the
// interleaved block):
//   m >= n (upper): VX = (V_0, X_0, V_1, X_1, ...), ld = mloc; YU = (Y_0,
//        Ur_0, Y_1, Ur_1, ...), ld = nloc.  Step i: V col 2i, X col 2i+1,
//        Y col 2i, Ur col 2i+1.
//   m <  n (lower): step i's left reflector is the ROW one:
//        VX = (X_0, V_0, X_1, V_1, ...), YU = (Ur_0, Y_0, ...).  Step i:
//        X col 2i, V col 2i+1, Ur col 2i, Y col 2i+1.
// V holds the left reflectors with their unit and zeros above it; Ur holds
// the right reflectors as the UNCONJUGATED ?LARFG vectors (A itself stores
// their conjugate, the ?GELQF convention ?UNGBR/?UNMBR expect); X and Y are
// ?LABRD's update factors.  Every panel column carries TWO padding T rows
// after its vector rows: [ld] = tau, [ld+1] = the reflector's real beta.
// No GEMM/GEMV reads past the ld vector rows, so ONE commBcast per column
// carries vector + scalars together; consumers read them from the padding.
//
// Per step (upper case), ?LABRD's recurrence over the interleaved pairs:
//   owner's process column: A(g0:, c) -= (V,X)(g0:, :2i) * (Y,Ur)(c, :2i)^H
//                           (ONE GEMM, k = 2i), then the column reflector;
//   every rank:             Y(:, i) from ONE Col-scope all-reduce of
//                           A^H v - (Y,Ur)(gy:, :2i) * (partial)^H
//                           (ONE GEMV over 2i rows... see the code), scaled
//                           by tau_v on DEVICE (?scal with device tau);
//   owner's process row:    r = A(c, g0:)^H - (Y,Ur)(g0:, :2i+1) * (corre-
//                           sponding)^H, ?LARFG -> Ur(:, i), conj tail back
//                           into A(c, g0+1:);
//   every rank:             X(:, i) from ONE Row-scope all-reduce, scaled
//                           by tau_r on device.
// (lower case: row reflector first, then column; widths i/i or i/i+1 as in
// ?LABRD's second branch.)
//
// GPU arm (device-resident): scalars never touch the host inside the step loop,
// and each half-step (column reflector + its Y, row reflector + its X) is ONE
// all-reduce.  Its input is the reflector's UNSCALED vector x, which every
// rank of the all-reduce's scope already holds: the owner brings the vector up
// to date and broadcasts it only at the start of a panel; inside a panel the
// vector rides in the PREVIOUS half-step's all-reduce as a single-contributor
// partial (the newest Y / X column, which that all-reduce produces, is the one
// term still missing: x = partial - Y_i resp. - X_i, since the newest
// reflector's anchor entry is exactly 1).  Every rank forms, in one launch
// (brd_mv), the -0.0-padded norm record [ar, ai, n_P, ns_P] (IEEE's additive
// identity makes a SUM allreduce reproduce every contribution bit-exactly)
// together with the partials below.  The
// reflector is v = e_anchor + s * x (s = xs * vfac from the ?LARFG scalar
// path) -- LINEAR in s -- so Y = tau * (A^H v - YU (V,X)^H v) splits into
// p1 + s * p2 and p4 + s * p3, with p1/p4 the anchor-row terms and p2/p3 two
// GEMVs on the unscaled x that need no reflector scalar.  ONE allreduce
// carries [record | p1 | p2 | p3 | p4]; brd_fold then runs the exact scalar
// path (qr_larfg_from_norms: identical bits on every rank), writes v, the
// {tau, beta} padding, A's reflector storage and tau * (p1 + f p2) /
// tau * (p4 + f p3) (f = s / sigma), leaving ONE GEMV for the correction.
// sigma, a power of two with sigma * max|A| in [0.5, 1) (one Grid allreduce
// per call), keeps the speculative GEMVs clear of under/overflow at any scale
// of A.  The host launch cost, not the GPU, bounds a column (a cuBLAS GEMV is ~100 us of host
// time for ~10 us of GPU time), so a half-step is exactly TWO kernel launches around its
// all-reduce: brd_mv (record + every speculative partial, one launch) and brd_fold2 (scalar path,
// reflector writes, the complete Y / X column, the carried vector).  Per column: 2 collectives,
// 4 kernels, ZERO host syncs (NCCL build; the tunnel comm wrappers are synchronous).
//
// CPU arm: the same record/fold/scalar math as host loops on the same
// QR_HD helpers: one MPI_Allgather of the per-rank {ar, ai, n, ns} + the
// fixed-order fold + qr_larfg_from_norms on every rank, host tau for scal.
//
// Per panel: ONE payload kernel packs (d, e, tauq, taup) over the panel's
// global column range from the padding slots (never touching e[k-1]); ONE
// D2H + sync at the END fills the HOST outputs -- the API is unchanged:
// d, e, tauq, taup are HOST arrays of GLOBAL length replicated on every
// rank.
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>

#include "ddla_connector.h"
#include "check_agree.h"
#include "BLAS/blas_impl.h"
#include "comm_traits.h"
#include "ddla_stream.h"
#include "ddla_desc.h"
#include "pgebrd_kernels.h"
#include "QR/pgeqrf_kernels.h"
#include "mpi_datatype.h"
#include "plan_util.h"
#include "svd_workspace.h"
#include "instantiate.h"
#include "require_backend.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace ddla {

namespace {

// Length of one all-reduce buffer of a fused half-step, [record | p1 | p2 | p3 | p4 | carried partial]:
// the record <= brd_rec_count(P) reals, p1 / p2 / the carried partial <= max(mloc, nloc), p3 / p4 <= 2 nbp + 1 each.
std::size_t brd_reduce_count(const spectral_impl::PgebrdPlan& p)
{
    return static_cast<std::size_t>(detail::brd_rec_count(std::max(p.nprows, p.npcols))) +
           3 * static_cast<std::size_t>(std::max(p.mloc, p.nloc)) + 4 * static_cast<std::size_t>(p.nbp) + 8;
}

// CPU arm's ?LARFG over one grid dimension as host loops (same math as the
// device kernels): per-rank contribution {ar, ai, n, ns} Allgathered, the
// fixed-order fold, qr_larfg_from_norms, in-place tail scale.  Identical
// {tau, beta} on every rank of comm; x holds the scaled tail.
template <typename T>
void host_larfg(MPI_Comm comm, int comm_size, int comm_rank, int owner,
                T* x, int nx, T* x_alpha, T& tau, typename real_type<T>::type& beta,
                typename real_type<T>::type* recv)
{
    using rt = typename real_type<T>::type;
    constexpr bool cplx = !std::is_same<T, rt>::value;

    rt local[4] = {-rt(0), -rt(0), -rt(0), -rt(0)};
    if (comm_rank == owner && x_alpha != nullptr) {
        local[0] = std::real(x_alpha[0]);
        if constexpr (cplx) local[1] = std::imag(x_alpha[0]);
    }
    if (nx > 0)
        detail::qr_norm_pair_serial<rt, cplx>(reinterpret_cast<const rt*>(x), nx,
                                      local[2], local[3]);
    MPI_CHECK(MPI_Allgather(local, 4, detail::mpi_datatype<rt>(), recv, 4,
                            detail::mpi_datatype<rt>(), comm));
    rt xn = rt(0), xns = rt(0);
    for (int p = comm_size - 1; p >= 0; --p) {
        xn = detail::qr_lapy2(xn, recv[static_cast<std::size_t>(p) * 4 + 2]);
        xns = detail::qr_lapy2(xns, recv[static_cast<std::size_t>(p) * 4 + 3]);
    }
    detail::QrLarfg<rt, cplx> s;
    detail::qr_larfg_from_norms<rt, cplx>(recv[static_cast<std::size_t>(owner) * 4],
                                  recv[static_cast<std::size_t>(owner) * 4 + 1],
                                  xn, xns, s);
    if constexpr (cplx) tau = T(s.tau_r, s.tau_i);
    else tau = T(s.tau_r);
    beta = s.beta;
    if (nx > 0 && !s.trivial) {
        rt xs = rt(1);
        for (int j = 0; j < s.knt; ++j) xs *= s.rsafmn;
        for (int i = 0; i < nx; ++i) {
            T* p = x + i;
            if constexpr (cplx) {
                const rt re = std::real(p[0]) * xs, im = std::imag(p[0]) * xs;
                p[0] = T(re * s.vfac_r - im * s.vfac_i, re * s.vfac_i + im * s.vfac_r);
            } else {
                p[0] = T((std::real(p[0]) * xs) * s.vfac_r);
            }
        }
    }
}

} // namespace

namespace spectral_impl {
PgebrdPlan pgebrd_geometry(const DdlaHandle_t& h,DdlaBackend backend,int m,int n,const int* desc)
{
    PgebrdPlan p;p.backend=backend;p.m=m;p.n=n;p.k=std::min(m,n);
    if(!p.k)return p;
    p.mb=desc[DDLA_MB_];p.nb=desc[DDLA_NB_];p.rsrc=desc[DDLA_RSRC_];p.csrc=desc[DDLA_CSRC_];p.lld=desc[DDLA_LLD_];
    ddlaGetGridDims(h,p.nprows,p.npcols);ddlaGetGridCoords(h,p.myprow,p.mypcol);
    p.mloc=num_loc(m,p.mb,p.myprow,p.rsrc,p.nprows);p.nloc=num_loc(n,p.nb,p.mypcol,p.csrc,p.npcols);
    p.nbp=std::max(1,std::min({p.nb,p.k,kSvdPanelWidth}));p.ldv=std::max(1,p.mloc)+2;p.ldy=std::max(1,p.nloc)+2;
    return p;
}
PgebrdPlan pgebrd_check(const DdlaHandle_t& handle,DdlaBackend backend,int m,int n,
    const void* A,const int* descA,const void* d,const void* e,const void* tauq,const void* taup)
{
    const auto p=detail::agree_check(handle,"pgebrd",[&](detail::Agree& agreement)->PgebrdPlan {
        agreement.mode("spectral operation",3101);
    detail::require_backend(handle,backend,"pgebrd");
    check_desc(descA,handle);
    if(m<0||m>descA[DDLA_M_]||n<0||n>descA[DDLA_N_])
        throw std::runtime_error("pgebrd: m/n outside the descriptor dimensions");
    const auto p=pgebrd_geometry(handle,backend,m,n,descA);
    if(!p.k)return p;
    if(!d||!tauq||!taup||(p.k>1&&!e))
        throw std::runtime_error("pgebrd: null d/e/tauq/taup array");
    if(p.mloc>0&&p.nloc>0&&!A)throw std::invalid_argument("pgebrd: null active input");
    return p;
    });
    int signature[13]={int(backend),m,n,handle->nprows_,handle->npcols_,int(handle->major)};
    detail::descriptor_signature(signature+6,descA);
    detail::agree_signature(handle,"pgebrd",signature);
    return p;
}

// Component 2 (svd_workspace.h).
template <typename T>
void pgebrd_workspace_size(const DdlaHandle_t&, const PgebrdPlan& p, std::size_t* backend_bytes)
{
    using rt = typename real_type<T>::type;
    *backend_bytes = 0;
    if (!p.k) return;
    const std::size_t nred = brd_reduce_count(p);
    std::size_t bytes = detail::padded_bytes<T>(static_cast<std::size_t>(p.ldv) * 2 * p.nbp)
                      + detail::padded_bytes<T>(static_cast<std::size_t>(p.ldy) * 2 * p.nbp)
                      + 2 * detail::padded_bytes<T>(nred);
    if (p.backend == DdlaBackend::CPU) {
        bytes += detail::padded_bytes<T>(p.ldy)
               + detail::padded_bytes<rt>(4 * static_cast<std::size_t>(std::max(p.nprows, p.npcols)));
    } else {
        bytes += detail::padded_bytes<rt>(1) + detail::padded_bytes<rt>(static_cast<std::size_t>(p.nprows) * p.npcols)
               + detail::padded_bytes<rt>(detail::kBrdAmaxBlocks) + detail::padded_bytes<rt>(p.k)
               + detail::padded_bytes<rt>(std::max(1, p.k - 1)) + 2 * detail::padded_bytes<T>(p.k);
    }
    *backend_bytes = bytes;
}

// Component 3.
template <DdlaBackend Backend, typename T>
void pgebrd_run(const DdlaHandle_t& handle,const PgebrdPlan& p,T* A,const int*,
    typename real_type<T>::type* d,typename real_type<T>::type* e,T* tauq,T* taup,int& info,
    void* backend_work,[[maybe_unused]] std::size_t backend_bytes)
{
    using rt=typename real_type<T>::type;

    static_assert(Backend!=DdlaBackend::GPU||DDLA_HAS_GPU,"GPU pgebrd is not available");
    info=0;if(!p.k)return;
    const int m=p.m,n=p.n,k=p.k,mb=p.mb,nb=p.nb,rsrc=p.rsrc,csrc=p.csrc,lld=p.lld;
    const int nprows=p.nprows,npcols=p.npcols,myprow=p.myprow,mypcol=p.mypcol;
    const int mloc=p.mloc,nloc=p.nloc,nbp=p.nbp,ldv=p.ldv,ldy=p.ldy;
    const bool upper=m>=n;[[maybe_unused]] const bool cplx=!std::is_same<rt,T>::value;
    const char adj=std::is_same<rt,T>::value?'T':'C';
    const std::size_t pv = static_cast<std::size_t>(ldv) * 2 * nbp;   // interleaved pairs
    const std::size_t py = static_cast<std::size_t>(ldy) * 2 * nbp;
    const std::size_t nred = brd_reduce_count(p);

    // ---- scratch carved once (never allocated inside the step loop) -------
    char* const base = static_cast<char*>(backend_work);
    char* w = base;
    T* const VX = detail::take<T>(w, pv);
    T* const YU = detail::take<T>(w, py);
    T* const reduce_send = detail::take<T>(w, nred);                  // all-reduce send/recv
    T* const reduce_recv = detail::take<T>(w, nred);
    auto zero = [&](T* ptr, std::size_t count) {
        if constexpr (Backend == DdlaBackend::CPU) std::fill_n(ptr, count, T(0));
        else DDLA_CHECK(ddlaMemset(handle, ptr, 0, count * sizeof(T)));
    };
    auto copy = [&](T* dst, const T* src, int count) {
        if constexpr (Backend == DdlaBackend::CPU) std::copy_n(src, count, dst);
        else DDLA_CHECK(ddlaMemcpy(handle, dst, src, static_cast<std::size_t>(count) * sizeof(T)));
    };

    // slot pointers of step i (local slots within the panel buffers)
    auto vcol_of = [&](int i) {  // V_i column (left reflector's v)
        return VX + static_cast<std::size_t>(upper ? 2 * i : 2 * i + 1) * ldv;
    };
    auto urcol_of = [&](int i) { // Ur_i column (right reflector's u)
        return YU + static_cast<std::size_t>(upper ? 2 * i + 1 : 2 * i) * ldy;
    };
    auto ycol_of = [&](int i) {
        return YU + static_cast<std::size_t>(upper ? 2 * i : 2 * i + 1) * ldy;
    };
    auto xcol_of = [&](int i) {
        return VX + static_cast<std::size_t>(upper ? 2 * i + 1 : 2 * i) * ldv;
    };

    // Both backends defer the same trailing rank-2jb update until panel end.
    auto update_trailing = [&](int first, int width) {
        const int t = first + width;
        if (t >= m || t >= n) return;
        const int lr = num_loc(t, mb, myprow, rsrc, nprows);
        const int lc = num_loc(t, nb, mypcol, csrc, npcols);
        const int mr = mloc - lr, nc = nloc - lc;
        if (mr > 0 && nc > 0)
            blas_impl::gemm_run<Backend, T>(handle, 'N', adj, mr, nc, 2 * width, T(-1),
                VX + lr, ldv, YU + lc, ldy, T(1), A + lr + static_cast<std::size_t>(lc) * lld, lld);
    };

    // ===================================================================
    // GPU arm: device-resident reflectors, interleaved buffers.
    // ===================================================================
    if constexpr (Backend != DdlaBackend::CPU) {
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)

        const int nprocs = nprows * npcols;
        rt* const d_sigma = detail::take<rt>(w, 1);                                   // sigma, device scalar
        rt* const d_slot = detail::take<rt>(w, static_cast<std::size_t>(nprocs));
        rt* const d_part = detail::take<rt>(w, static_cast<std::size_t>(detail::kBrdAmaxBlocks));
        rt* const d_d = detail::take<rt>(w, static_cast<std::size_t>(k));
        // e's device mirror holds only k-1 entries: the payload kernel never
        // writes e[k-1] and the final D2H must not overrun the caller's e either.
        rt* const d_e = detail::take<rt>(w, static_cast<std::size_t>(std::max(1, k - 1)));
        T* const d_tq = detail::take<T>(w, static_cast<std::size_t>(k));
        T* const d_tp = detail::take<T>(w, static_cast<std::size_t>(k));
        assert(static_cast<std::size_t>(w - base) <= backend_bytes);

        // ---- range guard: sigma = 2^-e with sigma * max|A| in [0.5, 1) (1 for a zero / non-finite A),
        // folded on the device from ONE Grid allreduce of the per-rank maxima.  Every speculative
        // GEMV below consumes sigma * x, so neither A^H x nor (V,X)^H x can under/overflow.
        {
            detail::brd_amax_local<T>(handle, A, mloc, nloc, lld, d_part, d_slot, myprow * npcols + mypcol,
                                      nprocs);
            commAllReduce<Backend>(handle, CommScope::Grid, d_slot, d_slot, nprocs, cclSum);
            detail::brd_sigma<T>(handle, d_slot, nprocs, d_sigma);
        }
        // T slots of the norm record at the head of a fused all-reduce buffer (the record is rt-typed;
        // complex T rounds it up to whole elements, brd_mv's record fill covers the extra real)
        auto rec_slots = [&](int P) {
            return (detail::brd_rec_count(P) + (cplx ? 1 : 0)) / (cplx ? 2 : 1);
        };

        using MvOp = detail::BrdMvOp<T>;
        // brd_mv ops: T form (A^H x style), N form (A x style + carried partials), strided copy / zero
        auto op_t = [](const T* M, int ld, int nrow, int ncol, const T* x, T* out) {
            MvOp o;
            o.kind = 1; o.M = M; o.ld = ld; o.nrow = nrow; o.ncol = ncol;
            o.x = x; o.incx = 1; o.scalex = true; o.out = out;
            return o;
        };
        auto op_n = [](T* out, int nout, const T* M, int ld, int ncol, const T* x, int incx, bool conjx,
                       bool scalex, const T* init, int incinit, bool conjinit, int sign) {
            MvOp o;
            o.kind = 2; o.out = out; o.nout = nout; o.M = M; o.ld = ld; o.ncol = ncol;
            o.x = x; o.incx = incx; o.conjx = conjx; o.scalex = scalex;
            o.init = init; o.incinit = incinit; o.conjinit = conjinit; o.sign = sign;
            return o;
        };
        auto op_fill = [](T* out, int n, const T* src, int stride, int mode) {
            MvOp o;
            o.kind = 3; o.out = out; o.nout = n; o.init = src; o.incinit = stride; o.mode = mode;
            return o;
        };

        // ---- column half: the column reflector H(c) (anchor row g0: upper c, lower c+1) and, when
        // with_y, Y(gy:, i) with gy = c+1 -- ONE Col all-reduce.  ny/nx_ = the widths of the update
        // terms (upper: i, i; lower: i, i+1).  ride_in: the unscaled column is already in the V slot
        // (the previous row half's all-reduce delivered it); otherwise the owner process column
        // updates it and ONE Row bcast hands it out.  carry: the all-reduce also carries the owner
        // process row's partial of the NEXT row reflector's vector (see the file header), finished
        // by brd_fold2 as r = partial - Y_i into that row half's Ur slot.
        auto column_half = [&](int c, int i, int g0, int gy, int ny, int nx_, bool with_y,
                               bool ride_in, bool carry) {
            const int pcc = indxg2p(c, nb, csrc, npcols);
            const int pr0 = indxg2p(g0, mb, rsrc, nprows);
            const int r0 = num_loc(g0, mb, myprow, rsrc, nprows);
            const int r1 = num_loc(g0 + 1, mb, myprow, rsrc, nprows);
            const int q0 = num_loc(gy, nb, mypcol, csrc, npcols);
            const int lc = indxg2l(c, nb, npcols);
            const int nr = mloc - r0;                          // rows from the anchor down
            const int nt = mloc - r1;                          // tail rows strictly below it
            const int nq = with_y ? nloc - q0 : 0;             // Y columns of this rank
            const int kw = with_y ? ny + nx_ : 0;              // produced (V,X)/(Y,Ur) columns the corrections ride on
            T* const acol = A + static_cast<std::size_t>(lc) * lld;
            T* const vcol = vcol_of(i);
            const bool owner = (mypcol == pcc);
            const bool anchor = (myprow == pr0);
            if (!ride_in) {
                // owner: A(g0:, c) -= (V,X)(g0:, :ny+nx_) * (Y,Ur)(c, :ny+nx_)^H (ONE GEMM), then the
                // updated column goes into the V slot; ONE Row bcast hands the UNSCALED column (anchor
                // value at r0 included) to every process column.  nr is uniform over a Row communicator.
                if (owner && nr > 0) {
                    if (ny + nx_ > 0)
                        blas_impl::gemm_run<Backend, T>(handle, 'N', adj, nr, 1, ny + nx_, T(-1),
                                         VX + r0, ldv, YU + lc, ldy, T(1), acol + r0, lld);
                    DDLA_CHECK(ddlaMemcpy(handle, vcol + r0, acol + r0, static_cast<std::size_t>(nr) * sizeof(T)));
                }
                if (npcols > 1 && nr > 0)
                    commBcast<Backend>(handle, CommScope::Row, vcol + r0, static_cast<std::size_t>(nr), pcc);
            }

            // record of this rank's tail + every speculative partial, ONE launch
            const int rs = rec_slots(nprows);
            T* const p1 = reduce_send + rs;
            T* const p2 = p1 + nq;
            T* const p3 = p2 + nq;
            T* const p4 = p3 + kw;
            T* const pc = p4 + kw;                             // carried row-reflector partial (nq)
            detail::BrdMvArgs<T> mv;
            mv.rx = vcol + r1; mv.rnx = nt; mv.ralpha = anchor ? vcol + r0 : nullptr;
            mv.rme = myprow; mv.rP = nprows;
            mv.rec = reinterpret_cast<rt*>(reduce_send); mv.rec_fill = rs * (cplx ? 2 : 1);
            mv.sigma = d_sigma;
            if (with_y) {
                // p2 = A^H x~, p3 = (V,X)^H x~ over the tail; p1 = conj(A(g0, gy:)), p4 = conj(VX(g0, :kw))
                // on the anchor's process row (the e_anchor terms of A^H v and (V,X)^H v), 0 elsewhere.
                mv.op[0] = op_t(A + r1 + static_cast<std::size_t>(q0) * lld, lld, nt, nq, vcol + r1, p2);
                mv.op[1] = op_t(VX + r1, ldv, nt, kw, vcol + r1, p3);
                mv.op[2] = op_fill(p1, nq, A + r0 + static_cast<std::size_t>(q0) * lld, lld, anchor ? 2 : 3);
                mv.op[3] = op_fill(p4, kw, VX + r0, ldv, anchor ? 2 : 3);
                if (carry) {
                    // r_partial = conj(A(row, gy:)) - (Y,Ur)(gy:, :kr) * (V,X)(row, :kr)^H, kr = Y_i's
                    // slot, on the process row owning the next row reflector's row (upper: row c,
                    // lower: row c+1); zeros elsewhere
                    const int rowc = upper ? c : c + 1;
                    const bool rowown = (myprow == indxg2p(rowc, mb, rsrc, nprows));
                    const int lrr = rowown ? indxg2l(rowc, mb, nprows) : 0;
                    const int kr = upper ? 2 * i : 2 * i + 1;
                    mv.op[4] = op_n(pc, nq, YU + q0, ldy, rowown ? kr : 0, VX + lrr, ldv, true, false,
                                    rowown ? A + lrr + static_cast<std::size_t>(q0) * lld : nullptr, lld,
                                    true, -1);
                }
            }
            detail::brd_mv<T>(handle, mv);
            const int cn = (with_y && carry) ? nq : 0;
            commAllReduce<Backend>(handle, CommScope::Col, reduce_send, reduce_recv, rs + 2 * nq + 2 * kw + cn, cclSum);

            // the reflector from the reduced record: v (V slot, + A's tail on the owner column), beta,
            // {tau, beta} padding on EVERY rank, the complete Y column and the carried r vector
            T* const q1 = reduce_recv + rs;
            T* const q2 = q1 + nq;
            T* const q3 = q2 + nq;
            T* const q4 = q3 + kw;
            T* const qc = q4 + kw;
            detail::brd_fold2<T>(handle, reinterpret_cast<const rt*>(reduce_recv), nprows,
                                 vcol + r1, nt, anchor ? vcol + r0 : nullptr,
                                 owner ? acol + r1 : nullptr, 1, false,
                                 (owner && anchor) ? acol + r0 : nullptr,
                                 vcol + mloc, vcol + mloc + 1,
                                 with_y ? ycol_of(i) + q0 : nullptr, nq, q1, q2, q3, q4, kw,
                                 YU + q0, ldy,
                                 (with_y && carry) ? qc : nullptr,
                                 (with_y && carry && nq > 0) ? urcol_of(upper ? i : i + 1) + q0 : nullptr,
                                 d_sigma);
        };

        // ---- row half: the row reflector G(c) (anchor column g0: upper c+1, lower c) and, when with_x,
        // X(gx:, i) with gx = c+1 -- ONE Row all-reduce.  nv/nx_ = the widths of the update terms
        // (upper: i+1, i; lower: i, i).  ride_in: the unscaled row is already in the Ur slot (the
        // column half's all-reduce delivered it); otherwise the owner process row forms it in the Ur
        // slot and ONE Col bcast hands it out.  carry: the all-reduce also carries the owner process
        // column's partial of the NEXT column reflector's vector, finished as x = partial - X_i.
        auto row_half = [&](int c, int i, int g0, int gx, int nv, int nx_, bool with_x,
                            bool ride_in, bool carry) {
            const int prc = indxg2p(c, mb, rsrc, nprows);
            const int pc0 = indxg2p(g0, nb, csrc, npcols);
            const int q0 = num_loc(g0, nb, mypcol, csrc, npcols);
            const int r0 = num_loc(gx, mb, myprow, rsrc, nprows);
            const int lrc = indxg2l(c, mb, nprows);
            const int nq = nloc - q0;                          // columns from the anchor on
            const int off = (mypcol == pc0) ? 1 : 0;           // the anchor column is mine
            const int nt = nq - off;                           // tail columns strictly beyond it
            const int nr = with_x ? mloc - r0 : 0;             // X rows of this rank
            const int kw = with_x ? nv + nx_ : 0;
            T* const ucol = urcol_of(i);
            const bool owner = (myprow == prc);
            const bool anchor = (off == 1);
            if (!ride_in) {
                // owner: r = conj(A(c, g0:)) - (Y,Ur)(g0:, :nv+nx_) * (V,X)(c, :nv+nx_)^H, formed IN the Ur
                // slot (ONE conj-copy + ONE GEMM); ONE Col bcast hands the unscaled r to every process row.
                // nq is uniform over a Col communicator.
                if (owner && nq > 0) {
                    T* const rtile = ucol + q0;
                    blas_impl::geam_run<Backend, T>(handle, 'C', 'N', nq, 1, T(1),
                                     A + lrc + static_cast<std::size_t>(q0) * lld, lld, T(0), nullptr, 1, rtile, nq);
                    if (nv + nx_ > 0)
                        blas_impl::gemm_run<Backend, T>(handle, 'N', adj, nq, 1, nv + nx_, T(-1),
                                         YU + q0, ldy, VX + lrc, ldv, T(1), rtile, nq);
                }
                if (nprows > 1 && nq > 0)
                    commBcast<Backend>(handle, CommScope::Col, ucol + q0, static_cast<std::size_t>(nq), prc);
            }

            const int rs = rec_slots(npcols);
            T* const p1 = reduce_send + rs;
            T* const p2 = p1 + nr;
            T* const p3 = p2 + nr;
            T* const p4 = p3 + kw;
            T* const pc = p4 + kw;                             // carried column-reflector partial (nr)
            detail::BrdMvArgs<T> mv;
            mv.rx = ucol + q0 + off; mv.rnx = nt; mv.ralpha = anchor ? ucol + q0 : nullptr;
            mv.rme = mypcol; mv.rP = npcols;
            mv.rec = reinterpret_cast<rt*>(reduce_send); mv.rec_fill = rs * (cplx ? 2 : 1);
            mv.sigma = d_sigma;
            if (with_x) {
                // p2 = A r~, p3 = (Y,Ur)^H r~ over the tail; p1 = A(gx:, g0), p4 = conj(YU(g0, :kw)) on the
                // anchor's process column (the e_anchor terms of A u and (Y,Ur)^H u), 0 elsewhere.
                mv.op[0] = op_n(p2, nr, A + r0 + static_cast<std::size_t>(q0 + off) * lld, lld, nt,
                                ucol + q0 + off, 1, false, true, nullptr, 1, false, 1);
                mv.op[1] = op_t(YU + q0 + off, ldy, nt, kw, ucol + q0 + off, p3);
                mv.op[2] = op_fill(p1, nr, A + r0 + static_cast<std::size_t>(q0) * lld, 1, anchor ? 1 : 3);
                mv.op[3] = op_fill(p4, kw, YU + q0, ldy, anchor ? 2 : 3);
                if (carry) {
                    // x_partial = A(gx:, col) - (V,X)(gx:, :kx) * (Y,Ur)(col, :kx)^H, kx = X_i's slot, on the
                    // process column owning the next column reflector's column (upper: c+1, lower: c);
                    // zeros elsewhere
                    const int colc = upper ? c + 1 : c;
                    const bool colown = (mypcol == indxg2p(colc, nb, csrc, npcols));
                    const int lcx = colown ? indxg2l(colc, nb, npcols) : 0;
                    const int kx = upper ? 2 * i + 1 : 2 * i;
                    mv.op[4] = op_n(pc, nr, VX + r0, ldv, colown ? kx : 0, YU + lcx, ldy, true, false,
                                    colown ? A + r0 + static_cast<std::size_t>(lcx) * lld : nullptr, 1,
                                    false, -1);
                }
            }
            detail::brd_mv<T>(handle, mv);
            const int cn = (with_x && carry) ? nr : 0;
            commAllReduce<Backend>(handle, CommScope::Row, reduce_send, reduce_recv, rs + 2 * nr + 2 * kw + cn, cclSum);

            T* const q1 = reduce_recv + rs;
            T* const q2 = q1 + nr;
            T* const q3 = q2 + nr;
            T* const q4 = q3 + kw;
            T* const qc = q4 + kw;
            // u (Ur slot, unconjugated), conj(u tail) into A's row on the owner row, beta into A(c, g0),
            // the complete X column and the carried x vector (the next column half's V slot)
            detail::brd_fold2<T>(handle, reinterpret_cast<const rt*>(reduce_recv), npcols,
                                 ucol + q0 + off, nt, anchor ? ucol + q0 : nullptr,
                                 owner ? A + lrc + static_cast<std::size_t>(q0 + off) * lld : nullptr, lld, true,
                                 (owner && anchor) ? A + lrc + static_cast<std::size_t>(q0) * lld : nullptr,
                                 ucol + nloc, ucol + nloc + 1,
                                 with_x ? xcol_of(i) + r0 : nullptr, nr, q1, q2, q3, q4, kw,
                                 VX + r0, ldv,
                                 (with_x && carry) ? qc : nullptr,
                                 (with_x && carry && nr > 0) ? vcol_of(upper ? i + 1 : i) + r0 : nullptr,
                                 d_sigma);
        };

        for (int k0 = 0; k0 < k; k0 += nbp) {
            const int jb = std::min(nbp, k - k0);
            zero(VX, pv);
            zero(YU, py);
            for (int i = 0; i < jb; ++i) {
                const int c = k0 + i;
                // ride_in / carry: inside a panel the vector of each half-step rides in the all-reduce
                // of the half-step before it (upper: column -> row -> next column; lower: row -> column
                // -> next row); a panel's first half-step starts from the owner + bcast.
                if (upper) {
                    column_half(c, i, c, c + 1, i, i, c < n - 1, i > 0, c < n - 1);
                    if (c < n - 1) row_half(c, i, c + 1, c + 1, i + 1, i, true, true, i + 1 < jb);
                } else {
                    row_half(c, i, c, c + 1, i, i, c < m - 1, i > 0, c < m - 1);
                    if (c < m - 1) column_half(c, i, c + 1, c + 1, i, i + 1, true, true, i + 1 < jb);
                }
            }
            // A22 -= VX(:, 2i0:2i0+2jb) * YU(:, 2i0:2i0+2jb)^H (one GEMM)
            update_trailing(k0, jb);
            // payload: d/e/tauq/taup for this panel from the padding slots
            // (count = the steps whose SECOND reflector ran).
            const int count_row_steps = std::min(jb, k - 1 - k0);
            detail::brd_panel_payload<T>(handle, upper, jb, k0, count_row_steps,
                                         ldv, ldy, mloc, nloc,
                                         VX, YU, d_d, d_e, d_tq, d_tp);
        }
        // d, e, tauq, taup to the HOST outputs: four stream-ordered copies behind
        // the last payload kernel, then ONE stream sync (the only one of the call).
        DDLA_CHECK(ddlaMemcpy(handle, d, d_d, static_cast<std::size_t>(k) * sizeof(rt)));
        if (k > 1)
            DDLA_CHECK(ddlaMemcpy(handle, e, d_e, static_cast<std::size_t>(k - 1) * sizeof(rt)));
        DDLA_CHECK(ddlaMemcpy(handle, tauq, d_tq, static_cast<std::size_t>(k) * sizeof(T)));
        DDLA_CHECK(ddlaMemcpy(handle, taup, d_tp, static_cast<std::size_t>(k) * sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(handle));
#endif
    }

    // ===================================================================
    // CPU arm: host loops, same layout, host tau.
    // ===================================================================
    if constexpr (Backend == DdlaBackend::CPU) {
        T* const hr = detail::take<T>(w, static_cast<std::size_t>(ldy));    // CPU arm's work vector, host memory
        // Column and row reflectors run serially and can reuse one receive
        // buffer.  It is carved outside the panel/reflector loops.
        rt* const norm_recv = detail::take<rt>(w, static_cast<std::size_t>(4) * std::max(nprows, npcols));
        assert(static_cast<std::size_t>(w - base) <= backend_bytes);
        auto column_step_cpu = [&](int c, int i, int g0, int ny, int nx_) {
            const int pcc = indxg2p(c, nb, csrc, npcols);
            const int pr0 = indxg2p(g0, mb, rsrc, nprows);
            const int r0 = num_loc(g0, mb, myprow, rsrc, nprows);
            const int r1 = num_loc(g0 + 1, mb, myprow, rsrc, nprows);
            const int lc = indxg2l(c, nb, npcols);
            T* const acol = A + static_cast<std::size_t>(lc) * lld;
            T* const vcol = vcol_of(i);
            if (mypcol == pcc) {
                const int nr = mloc - r0;
                const int nt = mloc - r1;
                if (nr > 0 && ny + nx_ > 0)
                    blas_impl::gemm_run<Backend, T>(handle, 'N', adj, nr, 1, ny + nx_, T(-1),
                                     VX + r0, ldv, YU + lc, ldy, T(1), acol + r0, lld);
                T tq{}; rt beta = rt(0);
                host_larfg<T>(handle->col_comm, nprows, myprow, pr0,
                              acol + r1, nt, acol + r0, tq, beta, norm_recv);
                if (myprow == pr0) acol[r0] = T(beta);
                // copy the tail into the V column (with the anchor at r0)
                std::copy_n(acol + r1, std::max(0, nt), vcol + r1);
                if (myprow == pr0 && r0 < ldv) vcol[r0] = T(1);
                vcol[mloc] = tq;
                vcol[mloc + 1] = T(beta);
            }
            // Even an empty local vector carries tau and beta in its padding.
            if (npcols > 1)
                commBcast<Backend>(handle, CommScope::Row, vcol,
                                   static_cast<std::size_t>(mloc) + 2, pcc);
        };

        auto row_step_cpu = [&](int c, int i, int g0, int nv, int nx_) {
            const int prc = indxg2p(c, mb, rsrc, nprows);
            const int pc0 = indxg2p(g0, nb, csrc, npcols);
            const int q0 = num_loc(g0, nb, mypcol, csrc, npcols);
            const int nq = nloc - q0;
            T* const ucol = urcol_of(i);
            if (myprow == prc) {
                const int lrc = indxg2l(c, mb, nprows);
                if (nq > 0) {
                    blas_impl::geam_run<Backend, T>(handle, 'C', 'N', nq, 1, T(1),
                                     A + lrc + static_cast<std::size_t>(q0) * lld, lld, T(0), nullptr, 1, hr, nq);
                    if (nv + nx_ > 0)
                        blas_impl::gemm_run<Backend, T>(handle, 'N', adj, nq, 1, nv + nx_, T(-1),
                                         YU + q0, ldy, VX + lrc, ldv, T(1), hr, nq);
                }
                const int off = (mypcol == pc0) ? 1 : 0;
                T tp{}; rt beta = rt(0);
                host_larfg<T>(handle->row_comm, npcols, mypcol, pc0,
                              hr + off, nq - off, hr, tp, beta, norm_recv);
                std::copy_n(hr, std::max(0, nq), ucol + q0);
                if (off) {
                    ucol[q0] = T(1);
                    A[lrc + static_cast<std::size_t>(q0) * lld] = T(beta);
                }
                if (nq - off > 0)
                    blas_impl::geam_run<Backend, T>(handle, 'C', 'N', 1, nq - off, T(1), hr + off, nq - off, T(0), nullptr, 1,
                                     A + lrc + static_cast<std::size_t>(q0 + off) * lld, lld);
                ucol[nloc] = tp;
                ucol[nloc + 1] = T(beta);
            }
            if (nprows > 1)
                commBcast<Backend>(handle, CommScope::Col, ucol,
                                   static_cast<std::size_t>(nloc) + 2, prc);
        };

        auto y_step_cpu = [&](int i, int gv, int gy, int ny, int nx_, const T& tq) {
            const int r0 = num_loc(gv, mb, myprow, rsrc, nprows);
            const int q0 = num_loc(gy, nb, mypcol, csrc, npcols);
            const int nr = mloc - r0;
            const int nq = nloc - q0;
            const T* const vcol = vcol_of(i) + r0;
            // produced-pairs width (slots [0, ny+nx_)); V_i's own slot excluded
            // (the add-back below writes Y_i = slot 2i while it runs).
            const int kw = ny + nx_;
            const int cnt = kw + nq;
            if (cnt == 0) return;
            zero(reduce_send, static_cast<std::size_t>(cnt));
            if (nr > 0 && nq > 0)
                blas_impl::gemv_run<Backend, T>(handle, adj, nr, nq, T(1),
                                 A + r0 + static_cast<std::size_t>(q0) * lld, lld,
                                 vcol, 1, T(0), reduce_send, 1);
            if (nr > 0 && kw > 0)
                blas_impl::gemv_run<Backend, T>(handle, adj, nr, kw, T(1),
                                 VX + r0, ldv, vcol, 1, T(0), reduce_send + nq, 1);
            commAllReduce<Backend>(handle, CommScope::Col, reduce_send, reduce_recv, cnt, cclSum);
            if (nq > 0) {
                T* const ycol = ycol_of(i) + q0;
                copy(ycol, reduce_recv, nq);
                if (kw > 0)
                    blas_impl::gemv_run<Backend, T>(handle, 'N', nq, kw, T(-1),
                                     YU + q0, ldy, reduce_recv + nq, 1, T(1), ycol, 1);
                blas_impl::scal_run<Backend, T>(handle, nq, tq, ycol, 1);
            }
        };

        auto x_step_cpu = [&](int i, int gu, int gx, int nv, int nx_, const T& tp) {
            const int q0 = num_loc(gu, nb, mypcol, csrc, npcols);
            const int r0 = num_loc(gx, mb, myprow, rsrc, nprows);
            const int nq = nloc - q0;
            const int nr = mloc - r0;
            const int kw = nv + nx_;
            const int cnt = kw + nr;
            if (cnt == 0) return;
            const T* const ucol = urcol_of(i) + q0;
            zero(reduce_send, static_cast<std::size_t>(cnt));
            if (nq > 0 && nr > 0)
                blas_impl::gemv_run<Backend, T>(handle, 'N', nr, nq, T(1),
                                 A + r0 + static_cast<std::size_t>(q0) * lld, lld,
                                 ucol, 1, T(0), reduce_send, 1);
            if (nq > 0 && kw > 0)
                blas_impl::gemv_run<Backend, T>(handle, adj, nq, kw, T(1),
                                 YU + q0, ldy, ucol, 1, T(0), reduce_send + nr, 1);
            commAllReduce<Backend>(handle, CommScope::Row, reduce_send, reduce_recv, cnt, cclSum);
            if (nr > 0) {
                T* const xcol = xcol_of(i) + r0;
                copy(xcol, reduce_recv, nr);
                if (kw > 0)
                    blas_impl::gemv_run<Backend, T>(handle, 'N', nr, kw, T(-1),
                                     VX + r0, ldv, reduce_recv + nr, 1, T(1), xcol, 1);
                blas_impl::scal_run<Backend, T>(handle, nr, tp, xcol, 1);   // unconditional (as above)
            }
        };

        for (int k0 = 0; k0 < k; k0 += nbp) {
            const int jb = std::min(nbp, k - k0);
            zero(VX, pv);
            zero(YU, py);
            for (int i = 0; i < jb; ++i) {
                const int c = k0 + i;
                if (upper) {
                    column_step_cpu(c, i, c, i, i);
                    d[c] = std::real(vcol_of(i)[mloc + 1]);
                    tauq[c] = vcol_of(i)[mloc];
                    if (c < n - 1) {
                        y_step_cpu(i, c, c + 1, i, i, tauq[c]);
                        row_step_cpu(c, i, c + 1, i + 1, i);
                        e[c] = std::real(urcol_of(i)[nloc + 1]);
                        taup[c] = urcol_of(i)[nloc];
                        x_step_cpu(i, c + 1, c + 1, i + 1, i, taup[c]);
                    } else {
                        taup[c] = T(0);
                    }
                } else {
                    row_step_cpu(c, i, c, i, i);
                    d[c] = std::real(urcol_of(i)[nloc + 1]);
                    taup[c] = urcol_of(i)[nloc];
                    if (c < m - 1) {
                        x_step_cpu(i, c, c + 1, i, i, taup[c]);
                        column_step_cpu(c, i, c + 1, i, i + 1);
                        e[c] = std::real(vcol_of(i)[mloc + 1]);
                        tauq[c] = vcol_of(i)[mloc];
                        y_step_cpu(i, c + 1, c + 1, i, i + 1, tauq[c]);
                    } else {
                        tauq[c] = T(0);
                    }
                }
            }
            update_trailing(k0, jb);
        }
        DDLA_CHECK(ddlaSynchronize(handle));
    }
}

} // namespace spectral_impl

template<DdlaBackend Backend,class T>
void pgebrd(const DdlaHandle_t& handle,int m,int n,T* A,const int* descA,
    typename real_type<T>::type* d,typename real_type<T>::type* e,T* tauq,T* taup,int& info)
{
    info=0;
    const auto p=spectral_impl::pgebrd_check(handle,Backend,m,n,A,descA,d,e,tauq,taup);
    std::size_t backend_bytes=0;
    spectral_impl::pgebrd_workspace_size<T>(handle,p,&backend_bytes);
    vector<char> backend_work(handle,backend_bytes);
    spectral_impl::pgebrd_run<Backend,T>(handle,p,A,descA,d,e,tauq,taup,info,backend_work.data(),backend_bytes);
}

#define INSTANTIATE_PGEBRD(BACKEND, TYPE)                                          \
    template void pgebrd<BACKEND, TYPE>(                                           \
        const DdlaHandle_t&, int, int, TYPE*, const int*,            \
        real_type<TYPE>::type*, real_type<TYPE>::type*, TYPE*, TYPE*, int&); \
    template void spectral_impl::pgebrd_run<BACKEND,TYPE>(const DdlaHandle_t&, \
        const spectral_impl::PgebrdPlan&,TYPE*,const int*,real_type<TYPE>::type*,real_type<TYPE>::type*, \
        TYPE*,TYPE*,int&,void*,std::size_t)
#define INSTANTIATE_PGEBRD_SIZE(TYPE) \
    template void spectral_impl::pgebrd_workspace_size<TYPE>(const DdlaHandle_t&, \
        const spectral_impl::PgebrdPlan&,std::size_t*)

DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_PGEBRD)
DDLA_FOR_EACH_TYPE(INSTANTIATE_PGEBRD_SIZE)

#undef INSTANTIATE_PGEBRD_SIZE
#undef INSTANTIATE_PGEBRD

} // namespace ddla
