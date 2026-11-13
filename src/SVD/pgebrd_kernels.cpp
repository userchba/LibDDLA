#include "ddla_connector.h"
// ---------------------------------------------------------------------------
// Device kernels of the device-resident pgebrd: brd_mv, brd_fold2,
// brd_amax_local/brd_sigma, brd_unit_trapezoid, brd_panel_payload.
// Contract: src/pgebrd_kernels.h.  Kernel discipline mirrors
// src/pgeqrf_kernels.cpp (dev1 export), with the qr_* LAPACK-accurate
// arithmetic from pgeqrf_kernels.h shared read-only.
//
// This TU is compiled as CUDA/HIP LANGUAGE with FMA contraction disabled
// (mirror of pgeqrf_kernels.cpp's -fmad=false / -ffp-contract=off): the qr_*
// helpers are instantiated here AND in pgeqrf_kernels.cpp, and only a
// contraction-free compile guarantees identical fp chains from both sites.
//
// Precedent for the launcher/kernel split: src/pgetf2_kernels.cpp.
// ---------------------------------------------------------------------------
#include "pgebrd_kernels.h"
#include "ddla_stream.h"
#include "instantiate.h"

#include <algorithm>

#if DDLA_HAS_GPU

namespace ddla {
namespace detail {

namespace {

constexpr int kBrdThreads = 128;

// ---------------------------------------------------------------------------
// brd_prep_body (block 0 of brd_mv): fill the record region with -0.0 (IEEE's exact additive
// identity, so the SUM allreduce reproduces every contributed value bit-exactly), then write
// this rank's slots: robust local norm + pre-scaled norm at 2+me / 2+P+me, alpha at 0/1 on the
// pivot owner.  x walks `nx` contiguous elements (interleaved rt for complex).  Needs exactly
// kBrdThreads threads.
// ---------------------------------------------------------------------------
template <typename rt, bool Cplx>
__device__ void brd_prep_body(const rt* x, int nx, const rt* alpha_or_null, int me, int P, rt* rec,
                              int rec_fill)
{
    constexpr int xr = Cplx ? 2 : 1;       // contiguous reflector tail, interleaved rt for complex
    __shared__ rt sh_red[kBrdThreads];
    __shared__ bool sh_nan[kBrdThreads];
    __shared__ bool sh_inf[kBrdThreads];
    __shared__ bool sh_finite;
    __shared__ int sh_e;
    __shared__ rt sh_n, sh_ns;

    const int tid = threadIdx.x;
    const int nrec = 2 + 2 * P;
    const int nfill = rec_fill > nrec ? rec_fill : nrec;   // fused path: whole T-padded record region
    for (int i = tid; i < nfill; i += kBrdThreads) rec[i] = -rt(0);
    __syncthreads();

    if (nx <= 0) {
        if (tid == 0) { sh_n = rt(0); sh_ns = rt(0); }
    } else {
        // Pass 1: NaN-sticky max over the components (same as the fused QR
        // record; components of x are x[el * xstride (+1 if Cplx)]).
        rt m = rt(0);
        bool nan_seen = false, inf_seen = false;
        for (int i = tid; i < nx; i += kBrdThreads) {
            const rt* p = x + static_cast<std::size_t>(i) * xr;
            const rt re = qr_abs(p[0]);
            if (qr_isnan(re)) nan_seen = true;
            if (qr_isinf(re)) inf_seen = true;
            if (re > m) m = re;
            if constexpr (Cplx) {
                const rt im = qr_abs(p[1]);
                if (qr_isnan(im)) nan_seen = true;
                if (qr_isinf(im)) inf_seen = true;
                if (im > m) m = im;
            }
        }
        sh_red[tid] = m;
        sh_nan[tid] = nan_seen;
        sh_inf[tid] = inf_seen;
        __syncthreads();
        for (int stride = kBrdThreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if (sh_nan[tid + stride]) sh_nan[tid] = true;
                if (sh_inf[tid + stride]) sh_inf[tid] = true;
                if (sh_red[tid + stride] > sh_red[tid]) sh_red[tid] = sh_red[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            if (sh_nan[0]) {
                sh_finite = false;
                sh_n = qr_nan<rt>();
                sh_ns = sh_n;
            } else if (sh_inf[0] || qr_isinf(sh_red[0])) {
                sh_finite = false;
                sh_n = qr_inf<rt>();
                sh_ns = sh_n;
            } else {
                sh_finite = true;
            }
        }
        __syncthreads();
        rt acc = rt(0);
        if (sh_finite) {
            if (tid == 0) {
                int e = 0;
                qr_frexp(sh_red[0], &e);
                sh_e = e;
            }
            __syncthreads();
            rt scl1, scl2;
            qr_norm_scale(sh_e, scl1, scl2);   // sh_e is uniform
            for (int i = tid; i < nx; i += kBrdThreads) {
                const rt* p = x + static_cast<std::size_t>(i) * xr;
                const rt re = (p[0] * scl1) * scl2;
                acc += re * re;
                if constexpr (Cplx) {
                    const rt im = (p[1] * scl1) * scl2;
                    acc += im * im;
                }
            }
        }
        sh_red[tid] = acc;
        __syncthreads();
        for (int stride = kBrdThreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) sh_red[tid] += sh_red[tid + stride];
            __syncthreads();
        }
        if (tid == 0 && sh_finite) {
            qr_norm_finish(qr_sqrt(sh_red[0]), sh_e, sh_n, sh_ns);
        }
        __syncthreads();
    }
    if (tid == 0) {
        rec[2 + me] = sh_n;
        rec[2 + P + me] = sh_ns;
        if (alpha_or_null != nullptr) {
            if constexpr (Cplx) { rec[0] = alpha_or_null[0]; rec[1] = alpha_or_null[1]; }
            else rec[0] = alpha_or_null[0];
        }
    }
}

// ---------------------------------------------------------------------------
// brd_panel_payload: one block per panel step.  The produced {tau, beta}
// pairs live in the padding slots of the panel columns (every rank's copy is
// identical after the bcasts); this kernel scatters them into the device
// d/e/tauq/taup arrays over the panel's global column range, never writing
// e[k-1] (the driver passes count_e_row = the number of steps whose row
// reflector ran, guarded inside).
// Padding layout of a column with L vector rows, in T units:  L = tau (1 T),
// L+1 = beta (1 rt, stored in beta_slot's real part for complex).
//   upper: V_i = VX slot 2i, Ur_i = YU slot 2i+1; d = beta_v, e = beta_ur,
//          tauq = tau_v, taup = tau_ur.
//   lower: Ur_i = YU slot 2i, V_i = VX slot 2i+1; d = beta_ur, e = beta_v,
//          tauq = tau_v, taup = tau_ur.
// count_col_row: steps whose SECOND reflector ran (row for upper, column for
// lower): e[j] is written only for i < count_col_row.
// ---------------------------------------------------------------------------
template <typename rt, bool Cplx>
__global__ void brd_panel_payload_kernel(bool upper, int jb, int c0,
                                         int count_col_row,
                                         int ldvp, int ldyp, int mloc, int nloc,
                                         const rt* vx, const rt* yu,
                                         rt* d_out, rt* e_out,
                                         rt* tauq_out, rt* taup_out)
{
    const int i = blockIdx.x;
    if (i >= jb) return;
    const int c = c0 + i;
    // slot of the left (column, V) and right (row, Ur) reflectors
    const int slot_v = upper ? 2 * i : 2 * i + 1;     // VX column index
    const int slot_u = upper ? 2 * i + 1 : 2 * i;     // YU column index
    // padding entries, T units -> rt units: tau at slot*ld, beta at slot*ld+1
    // (real) / slot*ld*(2) + [2,3] interleaved (complex).  The buffers are
    // T-typed; rt reinterpretation matches the interleaved layout.
    const int tau_stride = Cplx ? 2 : 1;
    const long long voff = (long long)slot_v * ldvp * tau_stride;
    const long long uoff = (long long)slot_u * ldyp * tau_stride;
    const rt tau_vr = vx[voff + (long long)mloc * tau_stride];
    const rt beta_v = vx[voff + (long long)mloc * tau_stride + tau_stride];
    const rt tau_ur = yu[uoff + (long long)nloc * tau_stride];
    const rt beta_ur = yu[uoff + (long long)nloc * tau_stride + tau_stride];
    if (!Cplx) {
        d_out[c] = upper ? beta_v : beta_ur;
        tauq_out[c] = tau_vr;
        taup_out[c] = tau_ur;
        if (i < count_col_row) e_out[c] = upper ? beta_ur : beta_v;
        else if (upper) taup_out[c] = (rt)0;
        else tauq_out[c] = (rt)0;
    } else {
        // complex tauq/taup are T-typed (interleaved rt pairs)
        rt* tq = tauq_out + 2 * c;
        rt* tp = taup_out + 2 * c;
        tq[0] = tau_vr; tq[1] = vx[voff + (long long)mloc * tau_stride + 1];
        tp[0] = tau_ur; tp[1] = yu[uoff + (long long)nloc * tau_stride + 1];
        d_out[c] = upper ? beta_v : beta_ur;
        if (i < count_col_row) e_out[c] = upper ? beta_ur : beta_v;
        else if (upper) { tp[0] = (rt)0; tp[1] = (rt)0; }
        else { tq[0] = (rt)0; tq[1] = (rt)0; }
    }
}

// ---------------------------------------------------------------------------
// Fused half-step kernels (pgebrd.cpp).  The host launch cost is pgebrd's bottleneck (a cuBLAS
// GEMV costs ~100 us of host time for ~10 us of GPU time), so a half-step is TWO launches around
// its single all-reduce: brd_mv before it, brd_fold2 after it.
// ---------------------------------------------------------------------------

// brd_mv: block 0 builds the norm record of the reflector tail; the other blocks run up to
// kBrdMvOps independent ops on the UNSCALED vector (inputs no op of the launch writes):
//   kind 1 (T form): out[j] = sum_i conj(M(i, j)) * (sigma * x[i])   -- A^H x, (V,X)^H x, (Y,Ur)^H x
//   kind 2 (N form): out[i] = init'(i) + sign * sum_j M(i, j) * x'(j) -- A x and the carried partials
//   kind 3 (fill)  : out[i] = src(i * stride), its conjugate, or 0
// Reductions are fixed-order shared-memory trees (deterministic, no warp shuffles).
template <typename rt>
struct MvOpR {
    int kind, nrow, ncol, conjx, scalex, conjinit, mode, sign, nout;
    const rt* M;
    long long ld;
    const rt* x;
    long long incx;
    const rt* init;
    long long incinit;
    rt* out;
};
template <typename rt>
struct MvArgsR {
    const rt* rx;
    int rnx;
    const rt* ralpha;
    int rme, rP, rec_fill;
    rt* rec;
    const rt* sigma;
    int blk[kBrdMvOps + 1];
    MvOpR<rt> op[kBrdMvOps];
};

template <typename rt, bool Cplx>
__global__ void __launch_bounds__(kBrdThreads) brd_mv_kernel(const MvArgsR<rt> a)
{
    constexpr int W = Cplx ? 2 : 1;
    __shared__ rt shr[4][32][W];
    const int bid = blockIdx.x;
    if (bid == 0) {
        brd_prep_body<rt, Cplx>(a.rx, a.rnx, a.ralpha, a.rme, a.rP, a.rec, a.rec_fill);
        return;
    }
    int o = 0;
    while (o + 1 < kBrdMvOps && bid >= a.blk[o + 1]) ++o;
    const MvOpR<rt>& op = a.op[o];
    const int lb = bid - a.blk[o];
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;

    if (op.kind == 1) {                                   // T form: one warp per output column
        const int j = lb * 4 + w;
        const bool valid = j < op.ncol;
        rt ar = rt(0), ai = rt(0);
        if (valid) {
            const rt sg = op.scalex ? a.sigma[0] : rt(1);
            const rt* mc = op.M + static_cast<long long>(j) * op.ld;
            #pragma unroll 4
            for (int i = lane; i < op.nrow; i += 32) {
                const rt* xp = op.x + static_cast<long long>(i) * op.incx;
                const rt* mp = mc + static_cast<long long>(i) * W;
                if constexpr (Cplx) {
                    const rt xr = xp[0] * sg, xi = xp[1] * sg;
                    const rt mr = mp[0], mi = mp[1];
                    ar += mr * xr + mi * xi;
                    ai += mr * xi - mi * xr;
                } else {
                    ar += mp[0] * (xp[0] * sg);
                }
            }
        }
        shr[w][lane][0] = ar;
        if constexpr (Cplx) shr[w][lane][1] = ai;
        __syncthreads();
        for (int s = 16; s > 0; s >>= 1) {
            if (lane < s) {
                shr[w][lane][0] += shr[w][lane + s][0];
                if constexpr (Cplx) shr[w][lane][1] += shr[w][lane + s][1];
            }
            __syncthreads();
        }
        if (valid && lane == 0) {
            op.out[static_cast<long long>(j) * W] = shr[w][0][0];
            if constexpr (Cplx) op.out[static_cast<long long>(j) * W + 1] = shr[w][0][1];
        }
        return;
    }
    if (op.kind == 2) {                                   // N form: 32 rows x 4 column groups per block
        const int row = lb * 32 + lane;
        rt ar = rt(0), ai = rt(0);
        if (row < op.nout) {
            const rt sg = op.scalex ? a.sigma[0] : rt(1);
            const rt* mp = op.M + static_cast<long long>(row) * W;
            #pragma unroll 4
            for (int j = w; j < op.ncol; j += 4) {
                const rt* xp = op.x + static_cast<long long>(j) * op.incx;
                const rt* mq = mp + static_cast<long long>(j) * op.ld;
                if constexpr (Cplx) {
                    const rt xr = xp[0] * sg;
                    const rt xi = op.conjx ? -(xp[1] * sg) : xp[1] * sg;
                    const rt mr = mq[0], mi = mq[1];
                    ar += mr * xr - mi * xi;
                    ai += mr * xi + mi * xr;
                } else {
                    ar += mq[0] * (xp[0] * sg);
                }
            }
        }
        shr[w][lane][0] = ar;
        if constexpr (Cplx) shr[w][lane][1] = ai;
        __syncthreads();
        if (w == 0 && row < op.nout) {
            const rt sr = ((shr[0][lane][0] + shr[1][lane][0]) + shr[2][lane][0]) + shr[3][lane][0];
            rt ir = rt(0), ii = rt(0);
            if (op.init != nullptr) {
                const rt* ip = op.init + static_cast<long long>(row) * op.incinit;
                ir = ip[0];
                if constexpr (Cplx) ii = op.conjinit ? -ip[1] : ip[1];
            }
            op.out[static_cast<long long>(row) * W] = ir + rt(op.sign) * sr;
            if constexpr (Cplx) {
                const rt si = ((shr[0][lane][1] + shr[1][lane][1]) + shr[2][lane][1]) + shr[3][lane][1];
                op.out[static_cast<long long>(row) * W + 1] = ii + rt(op.sign) * si;
            }
        }
        return;
    }
    // fill: grid-stride over this op's blocks
    const int nb = a.blk[o + 1] - a.blk[o];
    for (int i = lb * kBrdThreads + threadIdx.x; i < op.nout; i += nb * kBrdThreads) {
        rt re = rt(0), im = rt(0);
        if (op.mode != 3 && op.init != nullptr) {
            const rt* p = op.init + static_cast<long long>(i) * op.incinit;
            re = p[0];
            if constexpr (Cplx) im = (op.mode == 2) ? -p[1] : p[1];
        }
        rt* d = op.out + static_cast<long long>(i) * W;
        d[0] = re;
        if constexpr (Cplx) d[1] = im;
    }
}

// brd_fold2: everything that follows ONE reduced record.  Every block recomputes the exact ?LARFG
// scalar path (thread 0, shared) from the record -- deterministic, identical bits on every rank and
// block -- and the correction coefficients corr = tau * (p4 + f * p3) (f = xs * vfac / sigma, kw of
// them, in dynamic shared memory), then, grid-stride:
//   * the reflector tail  v = (x * rsafmn^knt) * vfac  IN PLACE (two roundings), and conj(v) / v into
//     the owner's A tail through pA (the trivial tau = 0 path stores the tail unchanged);
//   * y(j) = tau * (p1 + f * p2)(j) - sum_k M(j, k) * corr(k)   (the Y / X column, complete), and the
//     carried vector rout(j) = rp(j) - y(j);
// block 0 / thread 0 writes the unit anchor, beta (A's alpha slot) and the {tau, beta} pads on EVERY
// rank.
template <typename rt, bool Cplx>
__global__ void brd_fold2_kernel(const rt* rec_red, int P, rt* v, int nt, rt* panchor,
                                 rt* pA, long long pA_stride, int pA_conj, rt* pbeta,
                                 rt* ptau_slot, rt* pbeta_slot,
                                 rt* yout, int ny, const rt* p1, const rt* p2, const rt* p3, const rt* p4,
                                 int kw, const rt* M, long long ldM, const rt* rp, rt* rout,
                                 const rt* sigma)
{
    constexpr int W = Cplx ? 2 : 1;
    extern __shared__ __align__(16) unsigned char brd_fold_smem[];
    rt* const corr = reinterpret_cast<rt*>(brd_fold_smem);
    __shared__ bool sh_trivial;
    __shared__ rt sh_beta, sh_vfac_r, sh_vfac_i, sh_xs, sh_tau_r, sh_tau_i, sh_f_r, sh_f_i;
    const int tid = threadIdx.x;
    if (tid == 0) {
        QrLarfg<rt, Cplx> s;
        rt xn = rt(0), xns = rt(0);
        for (int p = P - 1; p >= 0; --p) {
            xn = qr_lapy2(xn, rec_red[2 + p]);
            xns = qr_lapy2(xns, rec_red[2 + P + p]);
        }
        qr_larfg_from_norms<rt, Cplx>(rec_red[0], rec_red[1], xn, xns, s);
        rt xs = rt(1);
        for (int j = 0; j < s.knt; ++j) xs *= s.rsafmn;
        sh_trivial = s.trivial;
        sh_beta = s.beta;
        sh_xs = xs;
        sh_tau_r = s.tau_r;
        sh_vfac_r = s.vfac_r;
        const rt isig = rt(1) / sigma[0];
        sh_f_r = s.trivial ? rt(0) : (xs * s.vfac_r) * isig;
        if constexpr (Cplx) {
            sh_tau_i = s.tau_i;
            sh_vfac_i = s.vfac_i;
            sh_f_i = s.trivial ? rt(0) : (xs * s.vfac_i) * isig;
        } else {
            sh_tau_i = rt(0);
            sh_vfac_i = rt(0);
            sh_f_i = rt(0);
        }
    }
    __syncthreads();
    for (int k = tid; k < kw; k += blockDim.x) {
        if constexpr (Cplx) {
            const rt yr = p4[2 * k] + (sh_f_r * p3[2 * k] - sh_f_i * p3[2 * k + 1]);
            const rt yi = p4[2 * k + 1] + (sh_f_r * p3[2 * k + 1] + sh_f_i * p3[2 * k]);
            corr[2 * k] = sh_tau_r * yr - sh_tau_i * yi;
            corr[2 * k + 1] = sh_tau_r * yi + sh_tau_i * yr;
        } else {
            corr[k] = sh_tau_r * (p4[k] + sh_f_r * p3[k]);
        }
    }
    // nvcc 11.8 can hoist the unrolled dot product's final shared loads
    // beyond kw even when their arithmetic is predicated off. Keep that
    // four-coefficient tail inside initialized shared storage.
    const int corr_cols = ((kw > 0 ? kw : 1) + 3) & ~3;
    for (int k = kw * W + tid; k < corr_cols * W; k += blockDim.x)
        corr[k] = rt(0);
    __syncthreads();

    const int gid = blockIdx.x * blockDim.x + tid;
    const int gstride = gridDim.x * blockDim.x;
    for (int i = gid; i < nt; i += gstride) {
        rt* vp = v + static_cast<std::size_t>(i) * W;
        if constexpr (Cplx) {
            rt vre = vp[0], vim = vp[1];
            if (!sh_trivial) {
                const rt re = vre * sh_xs;
                const rt im = vim * sh_xs;
                vre = re * sh_vfac_r - im * sh_vfac_i;
                vim = re * sh_vfac_i + im * sh_vfac_r;
            }
            vp[0] = vre; vp[1] = vim;
            if (pA != nullptr) {
                rt* ap = pA + static_cast<long long>(i) * pA_stride * W;
                ap[0] = vre;
                ap[1] = pA_conj ? -vim : vim;
            }
        } else {
            rt x = vp[0];
            if (!sh_trivial) x = (x * sh_xs) * sh_vfac_r;
            vp[0] = x;
            if (pA != nullptr) pA[static_cast<long long>(i) * pA_stride] = x;
        }
    }
    for (int j = gid; j < ny; j += gstride) {
        rt acr = rt(0), aci = rt(0);
        const rt* mr_ = M + static_cast<long long>(j) * W;
        for (int k = 0; k < kw; ++k) {
            const rt* mp = mr_ + static_cast<long long>(k) * ldM;
            if constexpr (Cplx) {
                const rt cr = corr[2 * k], ci = corr[2 * k + 1];
                acr += mp[0] * cr - mp[1] * ci;
                aci += mp[0] * ci + mp[1] * cr;
            } else {
                acr += mp[0] * corr[k];
            }
        }
        if constexpr (Cplx) {
            const rt* a = p1 + 2 * static_cast<std::size_t>(j);
            const rt* b = p2 + 2 * static_cast<std::size_t>(j);
            const rt yr = a[0] + (sh_f_r * b[0] - sh_f_i * b[1]);
            const rt yi = a[1] + (sh_f_r * b[1] + sh_f_i * b[0]);
            const rt outr = (sh_tau_r * yr - sh_tau_i * yi) - acr;
            const rt outi = (sh_tau_r * yi + sh_tau_i * yr) - aci;
            yout[2 * static_cast<std::size_t>(j)] = outr;
            yout[2 * static_cast<std::size_t>(j) + 1] = outi;
            if (rout != nullptr) {
                rout[2 * static_cast<std::size_t>(j)] = rp[2 * static_cast<std::size_t>(j)] - outr;
                rout[2 * static_cast<std::size_t>(j) + 1] = rp[2 * static_cast<std::size_t>(j) + 1] - outi;
            }
        } else {
            const rt outr = sh_tau_r * (p1[j] + sh_f_r * p2[j]) - acr;
            yout[j] = outr;
            if (rout != nullptr) rout[j] = rp[j] - outr;
        }
    }
    if (blockIdx.x == 0 && tid == 0) {
        if constexpr (Cplx) { ptau_slot[0] = sh_tau_r; ptau_slot[1] = sh_tau_i; }
        else ptau_slot[0] = sh_tau_r;
        pbeta_slot[0] = sh_beta;
        if (pbeta != nullptr) {
            if constexpr (Cplx) { pbeta[0] = sh_beta; pbeta[1] = rt(0); }
            else pbeta[0] = sh_beta;
        }
        if (panchor != nullptr) {
            if constexpr (Cplx) { panchor[0] = rt(1); panchor[1] = rt(0); }
            else panchor[0] = rt(1);
        }
    }
}

// Scale guard: sigma = a power of two with sigma * max|A| in [0.5, 1) (1 for 0 / Inf / NaN).
// Local pass: per-block partial maxima (one block per slice of the flat local matrix) ...
template <typename rt, bool Cplx>
__global__ void brd_amax_kernel(const rt* a, int mloc, int nloc, int lld, rt* part)
{
    constexpr int W = Cplx ? 2 : 1;
    __shared__ rt sh[kBrdThreads];
    const std::size_t total = static_cast<std::size_t>(mloc) * static_cast<std::size_t>(nloc);
    rt m = rt(0);
    for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const std::size_t i = idx % static_cast<std::size_t>(mloc);
        const std::size_t j = idx / static_cast<std::size_t>(mloc);
        const rt* p = a + (i + j * static_cast<std::size_t>(lld)) * W;
        rt v = qr_abs(p[0]);
        if (v > m) m = v;
        if constexpr (Cplx) {
            v = qr_abs(p[1]);
            if (v > m) m = v;
        }
    }
    sh[threadIdx.x] = m;
    __syncthreads();
    for (int stride = kBrdThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && sh[threadIdx.x + stride] > sh[threadIdx.x]) sh[threadIdx.x] = sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) part[blockIdx.x] = sh[0];
}

// ... the P-slot -0.0 padded array (this rank's maximum in slot `me`; ONE Grid SUM reproduces all) ...
template <typename rt>
__global__ void brd_amax_slot_kernel(const rt* part, int nparts, rt* slots, int me, int P)
{
    for (int i = threadIdx.x; i < P; i += blockDim.x) slots[i] = -rt(0);
    __syncthreads();
    if (threadIdx.x == 0) {
        rt m = rt(0);
        for (int i = 0; i < nparts; ++i) if (part[i] > m) m = part[i];
        slots[me] = m;
    }
}

// ... and the final fold: sigma = 2^-e with max = f * 2^e, f in [0.5, 1) (e clamped so sigma and
// 1/sigma stay normal).
template <typename rt>
__global__ void brd_sigma_kernel(const rt* slots, int P, rt* sigma)
{
    if (threadIdx.x != 0) return;
    rt m = rt(0);
    for (int i = 0; i < P; ++i) if (slots[i] > m) m = slots[i];
    rt sg = rt(1);
    if (m > rt(0) && !qr_isinf(m) && !qr_isnan(m)) {
        int e = 0;
        qr_frexp(m, &e);
        const int emax = (sizeof(rt) == 8) ? 1020 : 124;
        if (e > emax) e = emax;
        if (e < -emax) e = -emax;
        sg = rt(1);
        const rt two = rt(2);
        const rt half = rt(0.5);
        for (int i = 0; i < (e > 0 ? e : -e); ++i) sg *= (e > 0 ? half : two);
    }
    sigma[0] = sg;
}

// punmbr panel structure: element (i, jj) of the ld x jb block sits at GLOBAL index g(i) along the
// applied dimension (block-cyclic: block blk, np processes, this one `rel` places after the first);
// reflector jj has its unit at u0 + jj: g < u -> 0, g == u -> 1, g > u keeps the copied tail.
template <typename rt, bool Cplx>
__global__ void brd_unit_trap_kernel(rt* v, int ld, int rows, int jb, int u0, int blk, int rel, int np)
{
    constexpr int W = Cplx ? 2 : 1;
    const std::size_t total = static_cast<std::size_t>(rows) * static_cast<std::size_t>(jb);
    for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const int i = static_cast<int>(idx % static_cast<std::size_t>(rows));
        const int jj = static_cast<int>(idx / static_cast<std::size_t>(rows));
        const int g = ((i / blk) * np + rel) * blk + i % blk;
        const int u = u0 + jj;
        if (g > u) continue;
        rt* p = v + (static_cast<std::size_t>(i) + static_cast<std::size_t>(jj) * static_cast<std::size_t>(ld)) * W;
        p[0] = (g == u) ? rt(1) : rt(0);
        if constexpr (Cplx) p[1] = rt(0);
    }
}

} // namespace

// --- launcher wrappers -------------------------------------------------------

template <typename T>
void brd_mv(const DdlaHandle_t& handle, const BrdMvArgs<T>& args)
{
    const gpuStream_t stream = static_cast<gpuStream_t>(handle->stream);
    using rt = typename real_type<T>::type;
    constexpr bool Cplx = !std::is_same<rt, T>::value;
    constexpr int W = Cplx ? 2 : 1;
    MvArgsR<rt> a;
    a.rx = reinterpret_cast<const rt*>(args.rx);
    a.rnx = args.rnx;
    a.ralpha = reinterpret_cast<const rt*>(args.ralpha);
    a.rme = args.rme;
    a.rP = args.rP;
    a.rec_fill = args.rec_fill;
    a.rec = args.rec;
    a.sigma = args.sigma;
    int blocks = 1;                                        // block 0: the record
    for (int o = 0; o < kBrdMvOps; ++o) {
        const BrdMvOp<T>& s = args.op[o];
        MvOpR<rt>& d = a.op[o];
        d.kind = s.kind;
        d.nrow = s.nrow;
        d.ncol = s.ncol;
        d.conjx = s.conjx ? 1 : 0;
        d.scalex = s.scalex ? 1 : 0;
        d.conjinit = s.conjinit ? 1 : 0;
        d.mode = s.mode;
        d.sign = s.sign;
        d.nout = s.nout;
        d.M = reinterpret_cast<const rt*>(s.M);
        d.ld = static_cast<long long>(s.ld) * W;
        d.x = reinterpret_cast<const rt*>(s.x);
        d.incx = static_cast<long long>(s.incx) * W;
        d.init = reinterpret_cast<const rt*>(s.init);
        d.incinit = static_cast<long long>(s.incinit) * W;
        d.out = reinterpret_cast<rt*>(s.out);
        a.blk[o] = blocks;
        int nb = 0;
        if (s.kind == 1) nb = (s.ncol + 3) / 4;
        else if (s.kind == 2) nb = (s.nout + 31) / 32;
        else if (s.kind == 3) nb = std::min((s.nout + kBrdThreads - 1) / kBrdThreads, 16);
        blocks += std::max(nb, 0);
    }
    a.blk[kBrdMvOps] = blocks;
    brd_mv_kernel<rt, Cplx><<<blocks, kBrdThreads, 0, stream>>>(a);
    DDLA_CHECK(ddlaGetLastError(handle));
}

template <typename T>
void brd_fold2(const DdlaHandle_t& handle, const typename real_type<T>::type* rec_red, int P,
               T* v, int nt, T* panchor,
               T* pA, int pA_stride, bool pA_conj, T* pbeta,
               T* ptau_slot, T* pbeta_slot,
               T* yout, int ny, const T* p1, const T* p2, const T* p3, const T* p4, int kw,
               const T* M, int ldM, const T* rp, T* rout,
               const typename real_type<T>::type* sigma)
{
    const gpuStream_t stream = static_cast<gpuStream_t>(handle->stream);
    using rt = typename real_type<T>::type;
    constexpr bool Cplx = !std::is_same<rt, T>::value;
    constexpr int W = Cplx ? 2 : 1;
    const int maxn = std::max(1, std::max(nt, ny));
    const int blocks = std::min((maxn + kBrdThreads - 1) / kBrdThreads, 160);
    const std::size_t corr_cols = (static_cast<std::size_t>(std::max(kw, 1)) + 3) & ~std::size_t(3);
    const std::size_t smem = corr_cols * W * sizeof(rt);
    brd_fold2_kernel<rt, Cplx><<<blocks, kBrdThreads, smem, stream>>>(
        rec_red, P, reinterpret_cast<rt*>(v), nt, reinterpret_cast<rt*>(panchor),
        reinterpret_cast<rt*>(pA), static_cast<long long>(pA_stride), pA_conj ? 1 : 0,
        reinterpret_cast<rt*>(pbeta), reinterpret_cast<rt*>(ptau_slot), reinterpret_cast<rt*>(pbeta_slot),
        reinterpret_cast<rt*>(yout), ny, reinterpret_cast<const rt*>(p1), reinterpret_cast<const rt*>(p2),
        reinterpret_cast<const rt*>(p3), reinterpret_cast<const rt*>(p4), kw,
        reinterpret_cast<const rt*>(M), static_cast<long long>(ldM) * W,
        reinterpret_cast<const rt*>(rp), reinterpret_cast<rt*>(rout), sigma);
    DDLA_CHECK(ddlaGetLastError(handle));
}

template <typename T>
void brd_amax_local(const DdlaHandle_t& handle, const T* a, int mloc, int nloc, int lld,
                    typename real_type<T>::type* part, typename real_type<T>::type* slots,
                    int me, int P)
{
    const gpuStream_t stream = static_cast<gpuStream_t>(handle->stream);
    using rt = typename real_type<T>::type;
    constexpr bool Cplx = !std::is_same<rt, T>::value;
    const std::size_t total = static_cast<std::size_t>(std::max(mloc, 0)) * static_cast<std::size_t>(std::max(nloc, 0));
    const int blocks = static_cast<int>(std::min<std::size_t>((total + kBrdThreads - 1) / kBrdThreads, kBrdAmaxBlocks));
    const int nblk = std::max(blocks, 1);
    brd_amax_kernel<rt, Cplx><<<nblk, kBrdThreads, 0, stream>>>(
        reinterpret_cast<const rt*>(a), std::max(mloc, 1), total == 0 ? 0 : nloc, lld, part);
    DDLA_CHECK(ddlaGetLastError(handle));
    brd_amax_slot_kernel<rt><<<1, 64, 0, stream>>>(part, nblk, slots, me, P);
    DDLA_CHECK(ddlaGetLastError(handle));
}

template <typename T>
void brd_unit_trapezoid(const DdlaHandle_t& handle, T* v, int ld, int rows, int jb, int u0, int blk, int me, int src, int np)
{
    const gpuStream_t stream = static_cast<gpuStream_t>(handle->stream);
    using rt = typename real_type<T>::type;
    constexpr bool Cplx = !std::is_same<rt, T>::value;
    if (rows <= 0 || jb <= 0) return;
    const std::size_t total = static_cast<std::size_t>(rows) * static_cast<std::size_t>(jb);
    const int blocks = static_cast<int>(std::min<std::size_t>((total + kBrdThreads - 1) / kBrdThreads, 1024));
    brd_unit_trap_kernel<rt, Cplx><<<blocks, kBrdThreads, 0, stream>>>(
        reinterpret_cast<rt*>(v), ld, rows, jb, u0, blk, (me - src + np) % np, np);
    DDLA_CHECK(ddlaGetLastError(handle));
}

template <typename T>
void brd_sigma(const DdlaHandle_t& handle, const typename real_type<T>::type* slots, int P, typename real_type<T>::type* sigma)
{
    const gpuStream_t stream = static_cast<gpuStream_t>(handle->stream);
    using rt = typename real_type<T>::type;
    brd_sigma_kernel<rt><<<1, 32, 0, stream>>>(slots, P, sigma);
    DDLA_CHECK(ddlaGetLastError(handle));
}

// Explicit instantiations for the four scalar types.

// --- brd_panel_payload launcher ---------------------------------------------
template <typename T>
void brd_panel_payload(const DdlaHandle_t& handle, bool upper, int jb, int c0, int count_row_steps,
                       int ldv, int ldy, int mloc, int nloc,
                       const T* vx, const T* yu,
                       typename real_type<T>::type* d_out,
                       typename real_type<T>::type* e_out,
                       T* tauq_out, T* taup_out)
{
    const gpuStream_t stream = static_cast<gpuStream_t>(handle->stream);
    using rt = typename real_type<T>::type;
    constexpr bool Cplx = !std::is_same<rt, T>::value;
    if (jb <= 0) return;
    brd_panel_payload_kernel<rt, Cplx><<<jb, 1, 0, stream>>>(
        upper, jb, c0, count_row_steps, ldv, ldy, mloc, nloc,
        reinterpret_cast<const rt*>(vx), reinterpret_cast<const rt*>(yu),
        d_out, e_out, reinterpret_cast<rt*>(tauq_out),
        reinterpret_cast<rt*>(taup_out));
    DDLA_CHECK(ddlaGetLastError(handle));
}

template void brd_panel_payload<float>(const DdlaHandle_t&, bool, int, int, int, int, int, int, int,
                                       const float*, const float*, float*, float*,
                                       float*, float*);
template void brd_panel_payload<double>(const DdlaHandle_t&, bool, int, int, int, int, int, int, int,
                                        const double*, const double*, double*, double*,
                                        double*, double*);
template void brd_panel_payload<std::complex<float>>(const DdlaHandle_t&, bool, int, int, int, int, int,
                                                     int, int,
                                                     const std::complex<float>*,
                                                     const std::complex<float>*,
                                                     float*, float*,
                                                     std::complex<float>*,
                                                     std::complex<float>*);
template void brd_panel_payload<std::complex<double>>(const DdlaHandle_t&, bool, int, int, int, int, int,
                                                      int, int,
                                                      const std::complex<double>*,
                                                      const std::complex<double>*,
                                                      double*, double*,
                                                      std::complex<double>*,
                                                      std::complex<double>*);


// --- launcher explicit instantiations --------------------------------------------
#define BRD_INST_FUSED(T)                                                                          \
    template void brd_mv<T>(const DdlaHandle_t&, const BrdMvArgs<T>&);                                 \
    template void brd_fold2<T>(const DdlaHandle_t&, const real_type<T>::type*, int, T*, int, T*, T*, int, bool, T*,      \
                               T*, T*, T*, int, const T*, const T*, const T*, const T*, int,       \
                               const T*, int, const T*, T*, const real_type<T>::type*);                                                   \
    template void brd_amax_local<T>(const DdlaHandle_t&, const T*, int, int, int, real_type<T>::type*,                   \
                                    real_type<T>::type*, int, int);                \
    template void brd_sigma<T>(const DdlaHandle_t&, const real_type<T>::type*, int, real_type<T>::type*);   \
    template void brd_unit_trapezoid<T>(const DdlaHandle_t&, T*, int, int, int, int, int, int, int, int)
DDLA_FOR_EACH_TYPE(BRD_INST_FUSED)
#undef BRD_INST_FUSED

} // namespace detail
} // namespace ddla

#endif // DDLA_HAS_GPU