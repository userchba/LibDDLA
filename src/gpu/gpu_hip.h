#ifndef DDLA_GPU_HIP_H
#define DDLA_GPU_HIP_H

// HIP mapping of the vendor-neutral gpu* names (see gpu.h).
//
// The names that differ from CUDA only by the family prefix are listed once, in gpu_alias.h; this file defines the
// HIP prefixes for that list and everything HIP spells or declares differently.  It and gpu_cuda.h keep their own
// entries in the SAME order, so a side-by-side diff shows every vendor difference.  Only gpu/*.h may spell vendor
// API names; the rest of the library uses gpu*.

#ifdef DDLA_USE_CCL
#include <rccl/rccl.h>
#endif
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>
#include <hiprand/hiprand.h>
// This vendor cache workaround has no HIP counterpart.
#define DDLA_GPUBLAS_HAS_HEURISTICS_CACHE 0

// ---- the family prefixes of the shared alias list ---------------------------------------------
#define DDLA_GPU_RT(name) hip##name
#define DDLA_GPU_RT_CONST(name) HIP_##name
#define DDLA_GPU_BLAS(name) hipblas##name
#define DDLA_GPU_BLAS_CONST(name) HIPBLAS_##name
#define DDLA_GPU_SOLVER(name) hipsolver##name
#define DDLA_GPU_SOLVER_BASE(name) hipsolver##name
#define DDLA_GPU_SOLVER_CONST(name) HIPSOLVER_##name
#define DDLA_GPU_RAND(name) hiprand##name
#define DDLA_GPU_RAND_CONST(name) HIPRAND_##name
#include "gpu_alias.h"

namespace ddla {

// ---- runtime -----------------------------------------------------------------------------------
using gpuDataType_t = hipDataType;
using gpuPointerAttributes = hipPointerAttribute_t;
constexpr auto gpuDevAttrComputeCapabilityMajor = hipDeviceAttributeComputeCapabilityMajor;
constexpr auto gpuDevAttrComputeCapabilityMinor = hipDeviceAttributeComputeCapabilityMinor;
constexpr auto gpuDevAttrMaxSharedMemoryPerBlock = hipDeviceAttributeMaxSharedMemoryPerBlock;
constexpr auto gpuHostAllocDefault = hipHostMallocDefault;

// ---- BLAS --------------------------------------------------------------------------------------
using gpublasDoubleComplex = hipblasDoubleComplex;
using gpublasComplex = hipblasComplex;

// ---- solver ------------------------------------------------------------------------------------
using gpusolverDoubleComplex = hipDoubleComplex;
using gpusolverComplex = hipFloatComplex;
// hipsolverFillMode_t is a distinct enum from hipblasFillMode_t on current ROCm (older hipSOLVER and
// DTK alias it and expose the HIPSOLVER_FILL_MODE_* names as macros).  Mapping through the named
// constants of each family is correct under both layouts and survives any enumerator renumbering.
using gpusolverFillMode_t = hipsolverFillMode_t;
constexpr auto GPUSOLVER_FILL_MODE_LOWER = HIPSOLVER_FILL_MODE_LOWER;
constexpr auto GPUSOLVER_FILL_MODE_UPPER = HIPSOLVER_FILL_MODE_UPPER;
inline gpusolverFillMode_t gpusolverFillMode(gpublasFillMode_t uplo)
{
    if (uplo == GPUBLAS_FILL_MODE_LOWER)
        return GPUSOLVER_FILL_MODE_LOWER;
    if (uplo == GPUBLAS_FILL_MODE_UPPER)
        return GPUSOLVER_FILL_MODE_UPPER;
    throw std::runtime_error("gpusolverFillMode: invalid fill mode");
}

} // namespace ddla

// ---- functions (global macros) ------------------------------------------------------------

// runtime
#define gpuFreeHost hipHostFree
#define gpuHostAlloc hipHostMalloc

// BLAS

// solver
#define gpusolverCgetrf hipsolverCgetrf
#define gpusolverDgetrf hipsolverDgetrf
#define gpusolverSgetrf hipsolverSgetrf
#define gpusolverZgetrf hipsolverZgetrf

// ---- vendor features and launch helpers --------------------------------------------------
// cuSOLVER has laswp, hipSOLVER does not (HIP falls back to a row-swap loop, see laswp.cpp).
#define GPU_HAS_SOLVER_LASWP 0
#define GPU_DEFAULT_SHARED_MEM_PER_BLOCK (64 * 1024)   // when the device query fails

// Pointer-attribute memory type: cudaPointerAttributes::type (CUDA >= 11) and
// hipPointerAttribute_t::type (ROCm >= 6) replace the older ::memoryType member.
#if defined(HIP_VERSION_MAJOR) && HIP_VERSION_MAJOR >= 6
#define GPU_POINTER_MEMTYPE(attr) ((attr).type)
#else
#define GPU_POINTER_MEMTYPE(attr) ((attr).memoryType)
#endif

// Launch a kernel named by a parenthesised type tuple (template kernels whose names contain commas):
//   gpuLaunchKernelGGL((my_kernel<T, 4>), grid, block, shmem, stream, args...)
#define GPU_UNPAREN_KERNEL(...) __VA_ARGS__
#define gpuLaunchKernelGGL(kernel_tuple, grid, block, shmem, stream, ...) \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(GPU_UNPAREN_KERNEL kernel_tuple), grid, block, shmem, stream, \
                       __VA_ARGS__)

namespace ddla {
namespace gpu_detail {
template <typename T> struct identity { using type = T; };
} // namespace gpu_detail

// Launch a kernel through a function pointer with by-value arguments.  Returns the launch status.
template <typename... Params>
inline gpuError_t gpuLaunchKernelPtr(void (*func)(Params...), dim3 grid, dim3 block, std::size_t shmem,
                                     gpuStream_t stream, typename gpu_detail::identity<Params>::type... args)
{
    hipLaunchKernelGGL(func, grid, block, shmem, stream, args...);
    return hipGetLastError();
}

#if defined(__CUDACC__) || defined(__HIP__)
// 32-lane shuffle down (logical width 32 on both vendors; every lane of the group must reach the call).
template <typename T>
__device__ __forceinline__ T gpuShflDown32(T v, int off)
{
    return __shfl_down(v, off, 32);
}
#endif

} // namespace ddla

#endif // DDLA_GPU_HIP_H

