#ifndef DDLA_PGEBRD_KERNELS_H
#define DDLA_PGEBRD_KERNELS_H

// Public <ddla/ddla.h> first (hipcc default-argument rule, see src/ptran.h);
// pgebrd_kernels.cpp is a GPU-vendor TU and needs the runtime aliases.
#include <ddla/ddla.h>

#include "QR/pgeqrf_kernels.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <type_traits>

// Record length of brd_prep: {ar, ai} + n_P + ns_P.  Host-visible to both
// arms (the CPU arm sizes its brec with it too).
namespace ddla { namespace detail {
constexpr int brd_rec_count(int P) { return 2 + 2 * P; }
// Scratch reals brd_amax_local needs (one partial maximum per block).
constexpr int kBrdAmaxBlocks = 160;

// Padding slots of a panel-buffer column: the column has ld+2 rt-reals per
// step when complex / ld+2 when real, laid out as
//   [0, ld)   the vector v (columns of V or Ur; rows of the panel buffer)
//   [ld]      slot tau
//   [ld + 1]  slot beta (real type)
// No GEMM/GEMV reads past the ld rows of the column.
template <typename T>
constexpr int brd_pad_count()
{
    return std::is_same<typename real_type<T>::type, T>::value ? 1 : 2;
}
} }

#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)

// ---------------------------------------------------------------------------
// Device kernels of the device-resident pgebrd (schedule: src/pgebrd.cpp).  The
// LAPACK-accurate scalar path is ddla::detail's qr_larfg_from_norms (and
// qr_lapy2, ...) from src/pgeqrf_kernels.h, shared read-only.
//
// One reflector (column or row) per half-step:
//
//   brd_mv:     ONE launch before a half-step's all-reduce: the norm record of the reflector's
//               UNSCALED vector (-0.0-padded [ar, ai, n_0..n_{P-1}, ns_0..ns_{P-1}]: ONE SUM
//               all-reduce reproduces every contribution bit-exactly, so every rank folds the
//               same record in ScaLAPACK's DCOMBNRM2 p-order) plus the speculative GEMV-shaped
//               partials of Y / X (A^H x, (V,X)^H x, A x, ...), anchor-row copies and the carried
//               partial of the next vector.
//   brd_fold2:  ONE launch after it: the exact ?LARFG scalar path (identical bits on every rank;
//               the knt rescale rides in the carried ns norm), v, the unit anchor, beta, the
//               {tau, beta} padding, A's reflector storage, the complete Y / X column and the
//               carried vector.
//   brd_amax_local / brd_sigma: the range guard sigma, a power of two with
//               sigma * max|A| in [0.5, 1), from one Grid all-reduce of maxima.
//   brd_unit_trapezoid: punmbr's reflector-panel structure (zero above the unit).
//   brd_panel_payload: d / e / tauq / taup from the panel's padding slots.
//
// Kernel rules: fixed-order shared-memory reductions, no warp shuffles, templated
// <rt, Cplx> over interleaved rt buffers, no complex object passed by value, scalars
// stay in device memory across the kernels.
//
// pgebrd_kernels.cpp compiles with FMA contraction disabled (-fmad=false /
// -ffp-contract=off), like pgeqrf_kernels.cpp: the qr_* helpers are instantiated in
// two TUs (pgeqrf_kernels.cpp and this one), and only a contraction-free compile
// guarantees the fp chains reassociate identically.
// ---------------------------------------------------------------------------

namespace ddla {
namespace detail {

// --- GPU launchers (GPU-vendor TUs only) ------------------------------------

constexpr int kBrdMvOps = 6;

// One op of brd_mv (all pointers device memory, strides in elements of T):
//   kind 1 (T form): out[j] = sum_{i<nrow} conj(M(i, j)) * (sigma * x[i * incx]),  j < ncol
//   kind 2 (N form): out[i] = init'(i) + sign * sum_{j<ncol} M(i, j) * x'(j),      i < nout
//                    x'(j) = x[j * incx] (conjugated if conjx, times sigma if scalex),
//                    init'(i) = init[i * incinit] (conjugated if conjinit; init == null: 0)
//   kind 3 (fill)  : out[i] = init[i * incinit] (mode 1), its conjugate (mode 2) or 0 (mode 3), i < nout
// M is column-major with leading dimension ld.
template <typename T>
struct BrdMvOp {
    int kind = 0;                 // 0: unused
    const T* M = nullptr;
    int ld = 0, nrow = 0, ncol = 0;
    const T* x = nullptr;
    int incx = 1;
    bool conjx = false, scalex = false;
    const T* init = nullptr;
    int incinit = 1;
    bool conjinit = false;
    int mode = 1, sign = 1;
    T* out = nullptr;
    int nout = 0;
};

// ONE launch of a fused half-step: block 0 builds this rank's slice of the norm record
// [ar, ai, n_0..n_{P-1}, ns_0..ns_{P-1}] of the reflector tail rx (rnx contiguous elements; slot
// rme of rP; alpha at ralpha on the pivot owner, null elsewhere) at the head of the all-reduce
// buffer `rec`, filling rec_fill reals with -0.0 first (IEEE's additive identity makes the SUM
// all-reduce reproduce every contribution bit-exactly); the other blocks run the ops.  sigma: the
// device range-guard scalar (see brd_amax_local).
template <typename T>
struct BrdMvArgs {
    const T* rx = nullptr;
    int rnx = 0;
    const T* ralpha = nullptr;
    int rme = 0, rP = 1, rec_fill = 0;
    typename real_type<T>::type* rec = nullptr;
    const typename real_type<T>::type* sigma = nullptr;
    BrdMvOp<T> op[kBrdMvOps];
};
template <typename T>
void brd_mv(const DdlaHandle_t& handle, const BrdMvArgs<T>& args);

// Everything after the ONE reduced record of a fused half-step, in one launch: the exact ?LARFG
// scalar path (identical bits on every rank), the reflector tail v (in place, x -> (x * rsafmn^knt)
// * vfac; also stored into A through pA, conjugated when pA_conj), the unit anchor (panchor), beta
// (pbeta) and the {tau, beta} pads on EVERY rank, and the complete Y / X column
//   yout(j) = tau * (p1 + f * p2)(j) - sum_{k<kw} M(j, k) * tau * (p4 + f * p3)(k),  f = xs*vfac/sigma
// (ny rows), plus the carried vector rout = rp - yout when rout != null.  Null pointers skip the
// corresponding write.
template <typename T>
void brd_fold2(const DdlaHandle_t& handle, const typename real_type<T>::type* rec_red, int P,
               T* v, int nt, T* panchor,
               T* pA, int pA_stride, bool pA_conj, T* pbeta,
               T* ptau_slot, T* pbeta_slot,
               T* yout, int ny, const T* p1, const T* p2, const T* p3, const T* p4, int kw,
               const T* M, int ldM, const T* rp, T* rout,
               const typename real_type<T>::type* sigma);

// Scale guard: local max|A| (real and imaginary components) over the mloc x nloc local block into
// slot `me` of the P-slot -0.0-padded array `slots` (part: >= kBrdAmaxBlocks reals of scratch);
// after ONE Grid SUM of `slots`, brd_sigma folds it into sigma = 2^-e with sigma * max|A| in
// [0.5, 1).
template <typename T>
void brd_amax_local(const DdlaHandle_t& handle, const T* a, int mloc, int nloc, int lld,
                    typename real_type<T>::type* part, typename real_type<T>::type* slots,
                    int me, int P);
template <typename T>
void brd_sigma(const DdlaHandle_t& handle, const typename real_type<T>::type* slots, int P, typename real_type<T>::type* sigma);

// punmbr: restore the unit-trapezoid structure of a reflector panel copied from A (see the kernel):
// the ld x jb block's element (i, jj) at global index g(i) of the applied dimension (block-cyclic,
// block blk, process me of np, first process src) becomes 0 for g < u0 + jj, 1 for g == u0 + jj.
template <typename T>
void brd_unit_trapezoid(const DdlaHandle_t& handle, T* v, int ld, int rows, int jb, int u0, int blk, int me, int src, int np);

// Panel payload: one block per step, scattering {tau, beta} from the panel
// columns' padding slots into DEVICE d/e/tauq/taup over the panel's global
// column range [c0, c0 + jb); count_row_steps = the steps whose SECOND
// reflector ran (row for upper, column for lower): e is written only for
// those, and the trailing tauq (lower) / taup (upper) entry is zeroed --
// exactly ?GEBRD's tail convention.  Never writes e[k-1].  ldv/ldy: the
// buffer row counts; mloc/nloc: the LOCAL row extents (a column's padding
// slots sit at local rows mloc / nloc).
template <typename T>
void brd_panel_payload(const DdlaHandle_t& handle, bool upper, int jb, int c0, int count_row_steps,
                       int ldv, int ldy, int mloc, int nloc,
                       const T* vx, const T* yu,
                       typename real_type<T>::type* d_out,
                       typename real_type<T>::type* e_out,
                       T* tauq_out, T* taup_out);

} // namespace detail
} // namespace ddla

#endif // defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)

#endif // DDLA_PGEBRD_KERNELS_H