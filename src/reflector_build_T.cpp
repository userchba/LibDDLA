#include "reflector_build_T.h"
#include "ddla_connector.h"
#include "ddla_stream.h"

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstddef>
#include <type_traits>

namespace ddla {
namespace detail {

#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
namespace {
constexpr int kReflectorThreads = 128;

// ---------------------------------------------------------------------------
// reflector_build_T: one block per row r of T.  bwd == 0: T upper triangular (ZLARFT 'F'):
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
__global__ void reflector_build_T_kernel(int jb, const rt* __restrict__ vv,
                                         const rt* __restrict__ tau,
                                         rt* __restrict__ tmat, int ldt, int tau_shared, int bwd)
{
    extern __shared__ unsigned char ddla_reflector_smem[];
    rt* sh_row = reinterpret_cast<rt*>(ddla_reflector_smem);
    // tau_shared == 0 leaves this region unsized: sh_tau is never touched
    // then (every read is guarded by tau_shared).
    rt* sh_tau = sh_row + jb * (Cplx ? 2 : 1);   // jb entries (2*jb for rt pairs)
    __shared__ rt sh_red[2 * kReflectorThreads];

    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    if (tau_shared) {
        for (int i = tid; i < jb; i += kReflectorThreads) {
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
        for (int i = tid; i < jb; i += kReflectorThreads) {
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
        for (int p = p_lo + tid; p < p_hi; p += kReflectorThreads) {
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
        for (int stride = kReflectorThreads / 2; stride > 0; stride >>= 1) {
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

    for (int i = tid; i < jb; i += kReflectorThreads) {
        const std::size_t ti = static_cast<std::size_t>(r) + static_cast<std::size_t>(i) * ldt;
        if constexpr (Cplx) { tmat[2 * ti] = sh_row[2 * i]; tmat[2 * ti + 1] = sh_row[2 * i + 1]; }
        else tmat[ti] = sh_row[i];
    }
}

} // namespace
#endif

template <DdlaBackend Backend, typename T>
void reflector_build_T(const DdlaHandle_t& handle, int jb, const T* gram,
                       const T* tau, T* tmat, int ldt, bool backward)
{
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU reflector_build_T is not available in this build");
    if (jb <= 0) return;
    assert(gram && tau && tmat && ldt >= jb);
    if constexpr (Backend == DdlaBackend::CPU) {
        (void)handle;
        for (int c = 0; c < jb; ++c)
            std::fill_n(tmat + static_cast<std::size_t>(c) * ldt, jb, T(0));
        for (int step = 0; step < jb; ++step) {
            const int c = backward ? jb - 1 - step : step;
            const T tc = tau[c];
            tmat[c + static_cast<std::size_t>(c) * ldt] = tc;
            if (tc == T(0)) continue;
            const int first = backward ? c + 1 : 0, last = backward ? jb : c;
            for (int r = first; r < last; ++r) {
                T sum = 0;
                const int lo = backward ? c + 1 : r, hi = backward ? r + 1 : c;
                for (int p = lo; p < hi; ++p) {
                    T value = backward ? gram[c + static_cast<std::size_t>(p) * jb]
                                       : gram[p + static_cast<std::size_t>(c) * jb];
                    if constexpr (!std::is_same<T, typename real_type<T>::type>::value)
                        if (backward) value = std::conj(value);
                    sum += tmat[r + static_cast<std::size_t>(p) * ldt] * value;
                }
                tmat[r + static_cast<std::size_t>(c) * ldt] = -tc * sum;
            }
        }
    } else {
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
        const gpuStream_t stream = static_cast<gpuStream_t>(handle->stream);
        using rt = typename real_type<T>::type;
        constexpr bool Cplx = !std::is_same<rt, T>::value;
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
        const long long limit = have_dev ? perblock - 2 * kReflectorThreads * (long long)sizeof(rt)  // sh_red
                                         : GPU_DEFAULT_SHARED_MEM_PER_BLOCK - 2 * kReflectorThreads * (long long)sizeof(rt);
        const bool tau_shared = smem_dbl <= limit;
        const int smem = static_cast<int>(tau_shared ? smem_dbl : row_bytes);
        reflector_build_T_kernel<rt, Cplx><<<jb, kReflectorThreads, smem, stream>>>(
            jb, reinterpret_cast<const rt*>(gram), reinterpret_cast<const rt*>(tau),
            reinterpret_cast<rt*>(tmat), ldt, tau_shared ? 1 : 0, backward ? 1 : 0);
        DDLA_CHECK(ddlaGetLastError(handle));
#endif
    }
}

#define DDLA_INSTANTIATE_REFLECTOR_BUILD_T(B, T) \
    template void reflector_build_T<B, T>(const DdlaHandle_t&, int, const T*, const T*, T*, int, bool)
#define DDLA_INSTANTIATE_REFLECTOR_BUILD_T_TYPES(B) \
    DDLA_INSTANTIATE_REFLECTOR_BUILD_T(B, float); \
    DDLA_INSTANTIATE_REFLECTOR_BUILD_T(B, double); \
    DDLA_INSTANTIATE_REFLECTOR_BUILD_T(B, std::complex<float>); \
    DDLA_INSTANTIATE_REFLECTOR_BUILD_T(B, std::complex<double>)

// Host fallback is needed even when the public CPU backend is disabled.
DDLA_INSTANTIATE_REFLECTOR_BUILD_T_TYPES(DdlaBackend::CPU);
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
DDLA_INSTANTIATE_REFLECTOR_BUILD_T_TYPES(DdlaBackend::GPU);
#endif

#undef DDLA_INSTANTIATE_REFLECTOR_BUILD_T_TYPES
#undef DDLA_INSTANTIATE_REFLECTOR_BUILD_T

} // namespace detail
} // namespace ddla
