#ifndef DDLA_GPU_CUDA_H
#define DDLA_GPU_CUDA_H

// CUDA mapping of the vendor-neutral gpu* names (see gpu.h).
//
// The names that differ from HIP only by the family prefix are listed once, in gpu_alias.h; this file defines the
// CUDA prefixes for that list and everything CUDA spells or declares differently.  It and gpu_hip.h keep their own
// entries in the SAME order, so a side-by-side diff shows every vendor difference.  Only gpu/*.h may spell vendor
// API names; the rest of the library uses gpu*.

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#if CUBLAS_VERSION >= 111103
#include <cublasLt.h>
#define DDLA_GPUBLAS_HAS_HEURISTICS_CACHE 1
#else
#define DDLA_GPUBLAS_HAS_HEURISTICS_CACHE 0
#endif
#include <curand.h>
#ifdef DDLA_USE_CCL
#include <nccl.h>
#endif

// ---- the family prefixes of the shared alias list ---------------------------------------------
#define DDLA_GPU_RT(name) cuda##name
#define DDLA_GPU_RT_CONST(name) CUDA_##name
#define DDLA_GPU_BLAS(name) cublas##name
#define DDLA_GPU_BLAS_CONST(name) CUBLAS_##name
#define DDLA_GPU_SOLVER(name) cusolverDn##name
#define DDLA_GPU_SOLVER_BASE(name) cusolver##name
#define DDLA_GPU_SOLVER_CONST(name) CUSOLVER_##name
#define DDLA_GPU_RAND(name) curand##name
#define DDLA_GPU_RAND_CONST(name) CURAND_##name
#include "gpu_alias.h"

namespace ddla {

// ---- runtime -----------------------------------------------------------------------------------
using gpuDataType_t = cudaDataType_t;
using gpuPointerAttributes = cudaPointerAttributes;
constexpr auto gpuDevAttrComputeCapabilityMajor = cudaDevAttrComputeCapabilityMajor;
constexpr auto gpuDevAttrComputeCapabilityMinor = cudaDevAttrComputeCapabilityMinor;
constexpr auto gpuDevAttrMaxSharedMemoryPerBlock = cudaDevAttrMaxSharedMemoryPerBlock;
constexpr auto gpuHostAllocDefault = cudaHostAllocDefault;

// ---- BLAS --------------------------------------------------------------------------------------
using gpublasDoubleComplex = cuDoubleComplex;
using gpublasComplex = cuFloatComplex;

// ---- solver ------------------------------------------------------------------------------------
using gpusolverDoubleComplex = cuDoubleComplex;
using gpusolverComplex = cuFloatComplex;
// cuSOLVER takes the cuBLAS fill mode directly: the converter is the identity.
using gpusolverFillMode_t = gpublasFillMode_t;
constexpr auto GPUSOLVER_FILL_MODE_LOWER = GPUBLAS_FILL_MODE_LOWER;
constexpr auto GPUSOLVER_FILL_MODE_UPPER = GPUBLAS_FILL_MODE_UPPER;
inline gpusolverFillMode_t gpusolverFillMode(gpublasFillMode_t uplo) { return uplo; }

} // namespace ddla

// ---- functions (global macros) ------------------------------------------------------------

// runtime
#define gpuFreeHost cudaFreeHost
#define gpuHostAlloc cudaHostAlloc

// BLAS
#if DDLA_GPUBLAS_HAS_HEURISTICS_CACHE
#define gpublasHeuristicsCacheGetCapacity cublasLtHeuristicsCacheGetCapacity
#define gpublasHeuristicsCacheSetCapacity cublasLtHeuristicsCacheSetCapacity
#endif

// solver
#define gpusolverClaswp cusolverDnClaswp
#define gpusolverDlaswp cusolverDnDlaswp
#define gpusolverSlaswp cusolverDnSlaswp
#define gpusolverZlaswp cusolverDnZlaswp
// cuSOLVER's getrf has no Lwork argument (the workspace size is implied by the bufferSize call);
// hipSOLVER's has.  The neutral signature is the hipSOLVER one, so call sites are identical.
#define gpusolverSgetrf(h, m, n, A, lda, work, lwork, ipiv, info) \
    cusolverDnSgetrf(h, m, n, A, lda, work, ipiv, info)
#define gpusolverDgetrf(h, m, n, A, lda, work, lwork, ipiv, info) \
    cusolverDnDgetrf(h, m, n, A, lda, work, ipiv, info)
#define gpusolverCgetrf(h, m, n, A, lda, work, lwork, ipiv, info) \
    cusolverDnCgetrf(h, m, n, A, lda, work, ipiv, info)
#define gpusolverZgetrf(h, m, n, A, lda, work, lwork, ipiv, info) \
    cusolverDnZgetrf(h, m, n, A, lda, work, ipiv, info)

// ---- vendor features and launch helpers --------------------------------------------------
// cuSOLVER has laswp, hipSOLVER does not (HIP falls back to a row-swap loop, see laswp.cpp).
#define GPU_HAS_SOLVER_LASWP 1
#define GPU_DEFAULT_SHARED_MEM_PER_BLOCK (48 * 1024)   // when the device query fails

// Pointer-attribute memory type: cudaPointerAttributes::type (CUDA >= 11) and
// hipPointerAttribute_t::type (ROCm >= 6) replace the older ::memoryType member.
#define GPU_POINTER_MEMTYPE(attr) ((attr).type)

// Launch a kernel named by a parenthesised type tuple (template kernels whose names contain commas):
//   gpuLaunchKernelGGL((my_kernel<T, 4>), grid, block, shmem, stream, args...)
#define GPU_UNPAREN_KERNEL(...) __VA_ARGS__
#define gpuLaunchKernelGGL(kernel_tuple, grid, block, shmem, stream, ...) \
    GPU_UNPAREN_KERNEL kernel_tuple<<<grid, block, shmem, stream>>>(__VA_ARGS__)

namespace ddla {
namespace gpu_detail {
template <typename T> struct identity { using type = T; };
} // namespace gpu_detail

// Launch a kernel through a function pointer with by-value arguments.  Returns the launch status.
template <typename... Params>
inline gpuError_t gpuLaunchKernelPtr(void (*func)(Params...), dim3 grid, dim3 block, std::size_t shmem,
                                     gpuStream_t stream, typename gpu_detail::identity<Params>::type... args)
{
    void* argv[] = {static_cast<void*>(&args)...};
    return cudaLaunchKernel(reinterpret_cast<const void*>(func), grid, block, argv, shmem, stream);
}

#if defined(__CUDACC__) || defined(__HIP__)
// 32-lane shuffle down (logical width 32 on both vendors; every lane of the group must reach the call).
template <typename T>
__device__ __forceinline__ T gpuShflDown32(T v, int off)
{
    return __shfl_down_sync(0xffffffffu, v, off, 32);
}
#endif

} // namespace ddla

#endif // DDLA_GPU_CUDA_H

