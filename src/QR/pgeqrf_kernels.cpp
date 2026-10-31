#include "ddla_connector.h"
// ---------------------------------------------------------------------------
// Device kernels of the pgeqrf panel step on the GPU.  The panel itself is factored by the vendor
// geqrf (src/geqrf.cpp); what is left here is index and layout work, all of it run once per panel:
//
//   qr_reverse    -- local J-reversal for GPU geqrf with uplo = 'L'
//   qr_build_v    -- the explicit V (unit entries and zeros planted) from the window or in place
//   qr_adjoint_T  -- T -> T^H for the stage-1 hook
//   qr_build_T    -- T from V^H V and tau (ZLARFT 'F' for QR, 'B' for QL); the GPU arm of ddla::larft
//                    (src/larft.cpp) calls it
//
// Precedent: src/pgetf2_kernels.cpp (launcher + kernel split, explicit instantiation for the four
// scalar types).
// ---------------------------------------------------------------------------
#include "pgeqrf_kernels.h"
#include "ddla_stream.h"

#include <algorithm>

#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)

namespace ddla {
namespace detail {

namespace {

constexpr int kQrThreads = 128;
constexpr int kQrMaxBlocks = 1024;
// The fixed-order reduction trees halve the stride down from blockDim and the
// 2D launch grids divide row ranges by blockDim: both need a power of two.
static_assert(kQrThreads > 0 && (kQrThreads & (kQrThreads - 1)) == 0,
              "kQrThreads must be a power of two");

// Window element (r, col) of a <rt, Cplx>-templated panel: complex types are
// interleaved (rt pairs), real types plain.
template <bool Cplx>
DDLA_QR_HD inline std::size_t qr_elem(int r, int col, int ldw)
{
    const std::size_t e = static_cast<std::size_t>(r)
                        + static_cast<std::size_t>(col) * ldw;
    return Cplx ? 2 * e : e;
}

// Local row/column reversal for vendor QL (A' = J A J). Distributed
// panel packing and block-cyclic mapping live in matrix_transfer instead.
template <typename T>
__global__ void qr_reverse_kernel(int m, int n, const T* src, int ldsrc, T* dst, int lddst)
{
    for (int c = blockIdx.y; c < n; c += gridDim.y)
        for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < m;
             r += gridDim.x * blockDim.x)
            dst[std::size_t(m - 1 - r) + std::size_t(n - 1 - c) * lddst] =
                src[std::size_t(r) + std::size_t(c) * ldsrc];
}

// ---------------------------------------------------------------------------
// qr_build_v: explicit native V from a panel window (or in place from the
// freshly received window).  The window is in LAPACK storage, natural order.
// rev == 0 (QR, unit-lower V): entry (i, c) is 1 on the global diagonal
// (qr_l2g(lr0 + i) == j0 + c), 0 above it, the window value (g - j0, c) below
// it.  rev == 1 (QL, unit-UPPER V): the unit entry of column c sits at global
// row j0 + m_w - jb + c, 0 below it, the window value (g - j0, c) above it.
// in_place: the source is v itself, every entry reads only its own element.
// 2D grid (row block, column).
// ---------------------------------------------------------------------------
template <typename rt, bool Cplx>
__global__ void qr_build_v_kernel(int in_place, int rev, rt* v, int ldv, const rt* src,
                                  int ldsrc, int lrp, int jb, int j0, int m_w, int lr0,
                                  int mb, int myprow, int rsrc, int nprows)
{
    for (int c = blockIdx.y; c < jb; c += gridDim.y) {
        const int piv = rev ? j0 + m_w - jb + c : j0 + c;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < lrp;
             i += gridDim.x * blockDim.x) {
            const int g = qr_l2g(lr0 + i, mb, myprow, rsrc, nprows);
            const int d = g - j0;
            const int srow = in_place ? i : d;
            const int scol = c;
            rt* const p = v + qr_elem<Cplx>(i, c, ldv);
            const rt* const q = src + qr_elem<Cplx>(srow, scol, ldsrc);
            if (g == piv) {
                p[0] = rt(1);
                if constexpr (Cplx) p[1] = rt(0);
            } else if (rev ? g > piv : g < piv) {
                p[0] = rt(0);
                if constexpr (Cplx) p[1] = rt(0);
            } else {
                p[0] = q[0];
                if constexpr (Cplx) p[1] = q[1];
            }
        }
    }
}

// qr_adjoint_T: out(r, c) = conj(in(c, r)), r, c < jb.  One block; every output element is
// written by exactly one thread.  (A complex zero imaginary part becomes -0.0, which no
// consumer distinguishes.)
template <typename rt, bool Cplx>
__global__ void qr_adjoint_T_kernel(rt* out, int ldout, const rt* in, int jb)
{
    for (int idx = threadIdx.x; idx < jb * jb; idx += blockDim.x) {
        const int r = idx % jb, c = idx / jb;
        const rt* const q = in + qr_elem<Cplx>(c, r, jb);
        rt* const p = out + qr_elem<Cplx>(r, c, ldout);
        p[0] = q[0];
        if constexpr (Cplx) p[1] = -q[1];
    }
}

// ---------------------------------------------------------------------------
// qr_build_T: one block per row r of T.  bwd == 0: T upper triangular (ZLARFT 'F'):
// T(r,r) = tau_r, T(r,i) = -tau_i sum_{p=r}^{i-1} T(r,p) VV(p,i) for i = r+1 .. jb-1.
// bwd == 1: T lower triangular (ZLARFT 'B', the QL panel): T(r,i) = -tau_i sum_{p=i+1}^{r}
// T(r,p) VV(p,i) for i = r-1 .. 0, with VV(p,i) = conj(VV(i,p)) read from the stored UPPER
// triangle.  The row lives in shared memory, tau is
// EITHER preloaded into shared memory next to it (when the doubled buffer
// fits under the launch-time smem request; every step then reads it there)
// or read from global memory step by step by thread 0 (huge jb keeps the
// launch legal under the device's shared-memory limit).  The p-sums run with
// a fixed-order shared-memory tree; T(r,i) is produced sequentially in i
// (the recurrence consumes earlier entries of the row).
// ---------------------------------------------------------------------------
template <typename rt, bool Cplx>
__global__ void qr_build_T_kernel(int jb, const rt* __restrict__ vv,
                                  const rt* __restrict__ tau,
                                  rt* __restrict__ tmat, int ldt, int tau_shared, int bwd)
{
    extern __shared__ unsigned char ddla_qr_smem[];
    rt* sh_row = reinterpret_cast<rt*>(ddla_qr_smem);
    // tau_shared == 0 leaves this region unsized: sh_tau is never touched
    // then (every read is guarded by tau_shared).
    rt* sh_tau = sh_row + jb * (Cplx ? 2 : 1);   // jb entries (2*jb for rt pairs)
    __shared__ rt sh_red[2 * kQrThreads];

    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    if (tau_shared) {
        for (int i = tid; i < jb; i += kQrThreads) {
            if constexpr (Cplx) {
                sh_row[2 * i] = rt(0);
                sh_row[2 * i + 1] = rt(0);
                sh_tau[2 * i] = tau[2 * i];
                sh_tau[2 * i + 1] = tau[2 * i + 1];
            } else {
                sh_row[i] = rt(0);
                sh_tau[i] = tau[i];
            }
        }
    } else {
        for (int i = tid; i < jb; i += kQrThreads) {
            if constexpr (Cplx) { sh_row[2 * i] = rt(0); sh_row[2 * i + 1] = rt(0); }
            else sh_row[i] = rt(0);
        }
    }
    // The zero-fill (and the tau preload above) run on all warps: a barrier
    // is required before thread 0 plants the diagonal, or another warp's
    // writes could land after it (racecheck-flagged; matters for jb >
    // blockDim warps' reach).
    __syncthreads();
    if (tid == 0) {
        if constexpr (Cplx) {
            if (tau_shared) { sh_row[2 * r] = sh_tau[2 * r]; sh_row[2 * r + 1] = sh_tau[2 * r + 1]; }
            else { sh_row[2 * r] = tau[2 * r]; sh_row[2 * r + 1] = tau[2 * r + 1]; }
        } else {
            sh_row[r] = tau_shared ? sh_tau[r] : tau[r];
        }
    }
    __syncthreads();

    const int nsteps = bwd ? r : jb - 1 - r;
    for (int s = 0; s < nsteps; ++s) {
        const int i = bwd ? r - 1 - s : r + 1 + s;
        const int p_lo = bwd ? i + 1 : r;
        const int p_hi = bwd ? r + 1 : i;   // exclusive
        rt ar = rt(0), ai = rt(0);
        for (int p = p_lo + tid; p < p_hi; p += kQrThreads) {
            // forward VV(p, i), backward conj(VV(i, p)): both entries sit in the stored upper triangle
            const std::size_t vi = bwd ? static_cast<std::size_t>(i) + static_cast<std::size_t>(p) * jb
                                       : static_cast<std::size_t>(p) + static_cast<std::size_t>(i) * jb;
            if constexpr (Cplx) {
                const rt tr = sh_row[2 * p], ti = sh_row[2 * p + 1];
                const rt vr = vv[2 * vi], vi2 = bwd ? -vv[2 * vi + 1] : vv[2 * vi + 1];
                ar += tr * vr - ti * vi2;
                ai += tr * vi2 + ti * vr;
            } else {
                ar += sh_row[p] * vv[vi];
            }
        }
        sh_red[2 * tid] = ar;
        sh_red[2 * tid + 1] = ai;
        __syncthreads();
        for (int stride = kQrThreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sh_red[2 * tid] += sh_red[2 * (tid + stride)];
                sh_red[2 * tid + 1] += sh_red[2 * (tid + stride) + 1];
            }
            __syncthreads();
        }
        if (tid == 0) {
            // tau (global) or sh_tau (shared) is uniform across the block
            // here: the barrier above orders thread 0's read after every
            // thread's tree finish, and only thread 0 consumes it.
            if constexpr (Cplx) {
                const rt tr = tau_shared ? sh_tau[2 * i] : tau[2 * i];
                const rt ti = tau_shared ? sh_tau[2 * i + 1] : tau[2 * i + 1];
                sh_row[2 * i] = -(tr * sh_red[0] - ti * sh_red[1]);
                sh_row[2 * i + 1] = -(tr * sh_red[1] + ti * sh_red[0]);
            } else {
                sh_row[i] = -(tau_shared ? sh_tau[i] : tau[i]) * sh_red[0];
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < jb; i += kQrThreads) {
        const std::size_t ti = static_cast<std::size_t>(r) + static_cast<std::size_t>(i) * ldt;
        if constexpr (Cplx) { tmat[2 * ti] = sh_row[2 * i]; tmat[2 * ti + 1] = sh_row[2 * i + 1]; }
        else tmat[ti] = sh_row[i];
    }
}

} // namespace

// --- launcher wrappers ------------------------------------------------------

// 2D launch geometry shared by the row-permutation kernels: blocks_x covers
// the rows, blocks_y the columns (capped, with an in-kernel stride loop as
// the gridDim.y guard).
static dim3 qr_rowcol_grid(int rows, int cols)
{
    const int bx = static_cast<int>(std::min<long long>(
        (static_cast<long long>(rows) + kQrThreads - 1) / kQrThreads,
        kQrMaxBlocks));
    const int by = cols < 65535 ? cols : 65535;
    return dim3(bx > 0 ? bx : 1, by > 0 ? by : 1);
}

template <typename T>
void qr_reverse(const DdlaHandle_t& handle, int m, int n, const T* src, int ldsrc, T* dst, int lddst)
{
    if (m <= 0 || n <= 0) return;
    qr_reverse_kernel<T><<<qr_rowcol_grid(m, n), kQrThreads, 0, static_cast<gpuStream_t>(handle->stream)>>>(m, n, src, ldsrc, dst, lddst);
    DDLA_CHECK(ddlaGetLastError(handle));
}

template <typename T>
void qr_build_v(const DdlaHandle_t& handle, bool in_place, int rev, T* v, int ldv, const T* src, int ldsrc,
                int lrp, int jb, int j0, int m_w, int lr0, int mb, int myprow,
                int rsrc, int nprows)
{
    const gpuStream_t stream = static_cast<gpuStream_t>(handle->stream);
    using rt = typename qr_rt_of<T>::type;
    constexpr bool Cplx = !std::is_same<rt, T>::value;
    if (lrp <= 0 || jb <= 0) return;
    qr_build_v_kernel<rt, Cplx><<<qr_rowcol_grid(lrp, jb), kQrThreads, 0, stream>>>(
        in_place ? 1 : 0, rev, reinterpret_cast<rt*>(v), ldv,
        reinterpret_cast<const rt*>(src), ldsrc, lrp, jb, j0, m_w, lr0, mb, myprow,
        rsrc, nprows);
    DDLA_CHECK(ddlaGetLastError(handle));
}

template <typename T>
void qr_adjoint_T(const DdlaHandle_t& handle, T* out, int ldout, const T* in, int jb)
{
    const gpuStream_t stream = static_cast<gpuStream_t>(handle->stream);
    using rt = typename qr_rt_of<T>::type;
    constexpr bool Cplx = !std::is_same<rt, T>::value;
    if (jb <= 0) return;
    qr_adjoint_T_kernel<rt, Cplx><<<1, kQrThreads, 0, stream>>>(
        reinterpret_cast<rt*>(out), ldout, reinterpret_cast<const rt*>(in), jb);
    DDLA_CHECK(ddlaGetLastError(handle));
}

template <typename T>
void qr_build_T(const DdlaHandle_t& handle, int jb, const T* vv, const T* tau, T* tmat, int ldt, bool backward)
{
    const gpuStream_t stream = static_cast<gpuStream_t>(handle->stream);
    using rt = typename qr_rt_of<T>::type;
    constexpr bool Cplx = !std::is_same<rt, T>::value;
    if (jb <= 0) return;
    // Preferred layout: shared memory holds the T row AND the preloaded tau
    // (twice jb entries).  The launch must stay under the device's
    // shared-memory limit even on the default 48 KiB cap, so a jb that
    // doubles past it falls back to reading tau from global memory (the
    // row buffer alone still has to fit; at the point where THAT fails the
    // kernel cannot launch at all, and the caller's nb is far beyond any
    // sane block size -- the grid test pins the fallback at nb=2048).
    const long long row_bytes = static_cast<long long>(jb) * (Cplx ? 2 : 1)
                              * static_cast<long long>(sizeof(rt));
    const long long smem_dbl = 2 * row_bytes;
// The limit must be the PER-BLOCK maximum WITHOUT opting in: dynamic
// shared memory beyond 48 KiB requires cudaFuncSetAttribute
// (MaxDynamicSharedMemorySize), which is not called here, so sizing with
// the Optin attribute (96 KiB on V100) would let a 64 KiB request through
// and the launch would FAIL -- the opposite of the intended fallback
// (round-4 review finding 4).  cudaDevAttrMaxSharedMemoryPerBlock
// (hipDeviceAttributeMaxSharedMemoryPerBlock) is exactly the no-optin cap;
// the kernel's STATIC shared arrays (sh_red) come out of the same budget,
// so the row buffer must leave them room.
    int dev = 0, perblock = 0;
    const bool have_dev = gpuGetDevice(&dev) == gpuSuccess
                       && gpuDeviceGetAttribute(&perblock,
                             gpuDevAttrMaxSharedMemoryPerBlock, dev) == gpuSuccess;
    const long long limit = have_dev ? perblock - 2 * kQrThreads * (long long)sizeof(rt)  // sh_red
                                     : GPU_DEFAULT_SHARED_MEM_PER_BLOCK - 2 * kQrThreads * (long long)sizeof(rt);
    const bool tau_shared = smem_dbl <= limit;
    const int smem = static_cast<int>(tau_shared ? smem_dbl : row_bytes);
    qr_build_T_kernel<rt, Cplx><<<jb, kQrThreads, smem, stream>>>(
        jb, reinterpret_cast<const rt*>(vv), reinterpret_cast<const rt*>(tau),
        reinterpret_cast<rt*>(tmat), ldt, tau_shared ? 1 : 0, backward ? 1 : 0);
    DDLA_CHECK(ddlaGetLastError(handle));
}

// Explicit instantiations for the four scalar types.
#define DDLA_QR_INST_MAP_KERNELS(T)                                                \
    template void qr_reverse<T>(const DdlaHandle_t&, int, int, const T*, int, T*, int); \
    template void qr_build_v<T>(const DdlaHandle_t&, bool, int, T*, int, const T*, int, int, int, int,  \
                                int, int, int, int, int, int);    \
    template void qr_adjoint_T<T>(const DdlaHandle_t&, T*, int, const T*, int);

DDLA_QR_INST_MAP_KERNELS(float)
DDLA_QR_INST_MAP_KERNELS(double)
DDLA_QR_INST_MAP_KERNELS(std::complex<float>)
DDLA_QR_INST_MAP_KERNELS(std::complex<double>)

#undef DDLA_QR_INST_MAP_KERNELS

template void qr_build_T<float>(const DdlaHandle_t&, int, const float*, const float*, float*, int, bool);
template void qr_build_T<double>(const DdlaHandle_t&, int, const double*, const double*, double*, int, bool);
template void qr_build_T<std::complex<float>>(const DdlaHandle_t&, int, const std::complex<float>*,
                                              const std::complex<float>*,
                                              std::complex<float>*, int, bool);
template void qr_build_T<std::complex<double>>(const DdlaHandle_t&, int, const std::complex<double>*,
                                               const std::complex<double>*,
                                               std::complex<double>*, int, bool);

} // namespace detail
} // namespace ddla

#endif // defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
