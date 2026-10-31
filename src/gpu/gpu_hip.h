#ifndef DDLA_GPU_HIP_H
#define DDLA_GPU_HIP_H

// HIP mapping of the vendor-neutral gpu* names (see gpu.h).
//
// This file and gpu_cuda.h list the SAME names in the SAME order, so a side-by-side diff shows every
// vendor difference.  Only gpu/*.h may spell vendor API names; the rest of the library uses gpu*.
//   types      -> using          (inside namespace ddla)
//   constants  -> constexpr auto (inside namespace ddla)
//   functions  -> #define        (vendor functions can be overloaded or macros themselves)

#ifdef DDLA_USE_CCL
#include <rccl/rccl.h>
#endif
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>
#include <hiprand/hiprand.h>

namespace ddla {

// ---- runtime -----------------------------------------------------------------------------------
using gpuDataType_t = hipDataType;
using gpuError_t = hipError_t;
using gpuEvent_t = hipEvent_t;
using gpuMemcpyKind = hipMemcpyKind;
using gpuPointerAttributes = hipPointerAttribute_t;
using gpuStream_t = hipStream_t;
constexpr auto GPU_C_32F = HIP_C_32F;
constexpr auto GPU_C_64F = HIP_C_64F;
constexpr auto GPU_R_32F = HIP_R_32F;
constexpr auto GPU_R_64F = HIP_R_64F;
constexpr auto gpuDevAttrMaxSharedMemoryPerBlock = hipDeviceAttributeMaxSharedMemoryPerBlock;
constexpr auto gpuErrorInvalidValue = hipErrorInvalidValue;
constexpr auto gpuErrorNotReady = hipErrorNotReady;
constexpr auto gpuHostAllocDefault = hipHostMallocDefault;
constexpr auto gpuMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;
constexpr auto gpuMemcpyDeviceToHost = hipMemcpyDeviceToHost;
constexpr auto gpuMemcpyHostToDevice = hipMemcpyHostToDevice;
constexpr auto gpuMemoryTypeDevice = hipMemoryTypeDevice;
constexpr auto gpuMemoryTypeManaged = hipMemoryTypeManaged;
constexpr auto gpuStreamNonBlocking = hipStreamNonBlocking;
constexpr auto gpuSuccess = hipSuccess;

// ---- BLAS --------------------------------------------------------------------------------------
using gpublasDiagType_t = hipblasDiagType_t;
using gpublasFillMode_t = hipblasFillMode_t;
using gpublasHandle_t = hipblasHandle_t;
using gpublasOperation_t = hipblasOperation_t;
using gpublasPointerMode_t = hipblasPointerMode_t;
using gpublasSideMode_t = hipblasSideMode_t;
using gpublasStatus_t = hipblasStatus_t;
constexpr auto GPUBLAS_DIAG_NON_UNIT = HIPBLAS_DIAG_NON_UNIT;
constexpr auto GPUBLAS_DIAG_UNIT = HIPBLAS_DIAG_UNIT;
constexpr auto GPUBLAS_FILL_MODE_LOWER = HIPBLAS_FILL_MODE_LOWER;
constexpr auto GPUBLAS_FILL_MODE_UPPER = HIPBLAS_FILL_MODE_UPPER;
constexpr auto GPUBLAS_OP_C = HIPBLAS_OP_C;
constexpr auto GPUBLAS_OP_N = HIPBLAS_OP_N;
constexpr auto GPUBLAS_OP_T = HIPBLAS_OP_T;
constexpr auto GPUBLAS_POINTER_MODE_DEVICE = HIPBLAS_POINTER_MODE_DEVICE;
constexpr auto GPUBLAS_SIDE_LEFT = HIPBLAS_SIDE_LEFT;
constexpr auto GPUBLAS_SIDE_RIGHT = HIPBLAS_SIDE_RIGHT;
constexpr auto GPUBLAS_STATUS_SUCCESS = HIPBLAS_STATUS_SUCCESS;
using gpublasDoubleComplex = hipblasDoubleComplex;
using gpublasComplex = hipblasComplex;

// ---- solver ------------------------------------------------------------------------------------
using gpusolverHandle_t = hipsolverHandle_t;
using gpusolverStatus_t = hipsolverStatus_t;
constexpr auto GPUSOLVER_STATUS_SUCCESS = HIPSOLVER_STATUS_SUCCESS;
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

// ---- rand --------------------------------------------------------------------------------------
using gpurandGenerator_t = hiprandGenerator_t;
using gpurandRngType = hiprandRngType;
using gpurandStatus_t = hiprandStatus_t;
constexpr auto GPURAND_RNG_PSEUDO_DEFAULT = HIPRAND_RNG_PSEUDO_DEFAULT;
constexpr auto GPURAND_STATUS_SUCCESS = HIPRAND_STATUS_SUCCESS;

// ---- collective communication (NCCL / RCCL) ----------------------------------------------------
#ifdef DDLA_USE_CCL
using gpucclComm_t = ncclComm_t;
using gpucclRedOp_t = ncclRedOp_t;
using gpucclResult_t = ncclResult_t;
using gpucclUniqueId = ncclUniqueId;
constexpr auto gpucclFloat32 = ncclFloat32;
constexpr auto gpucclFloat64 = ncclFloat64;
constexpr auto gpucclInt8 = ncclInt8;
constexpr auto gpucclMax = ncclMax;
constexpr auto gpucclMin = ncclMin;
constexpr auto gpucclProd = ncclProd;
constexpr auto gpucclSuccess = ncclSuccess;
constexpr auto gpucclSum = ncclSum;
#endif

} // namespace ddla

// ---- functions (global macros) ------------------------------------------------------------

// runtime
#define gpuDeviceGetAttribute hipDeviceGetAttribute
#define gpuDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuEventCreate hipEventCreate
#define gpuEventDestroy hipEventDestroy
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize
#define gpuFree hipFree
#define gpuFreeAsync hipFreeAsync
#define gpuFreeHost hipHostFree
#define gpuGetDevice hipGetDevice
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuGetErrorString hipGetErrorString
#define gpuGetLastError hipGetLastError
#define gpuHostAlloc hipHostMalloc
#define gpuMalloc hipMalloc
#define gpuMallocAsync hipMallocAsync
#define gpuMallocManaged hipMallocManaged
#define gpuMemGetInfo hipMemGetInfo
#define gpuMemcpy hipMemcpy
#define gpuMemcpy2DAsync hipMemcpy2DAsync
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemsetAsync hipMemsetAsync
#define gpuPointerGetAttributes hipPointerGetAttributes
#define gpuSetDevice hipSetDevice
#define gpuStreamCreate hipStreamCreate
#define gpuStreamCreateWithFlags hipStreamCreateWithFlags
#define gpuStreamCreateWithPriority hipStreamCreateWithPriority
#define gpuStreamDestroy hipStreamDestroy
#define gpuStreamQuery hipStreamQuery
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuStreamWaitEvent hipStreamWaitEvent

// BLAS
#define gpublasCaxpy hipblasCaxpy
#define gpublasCdotc hipblasCdotc
#define gpublasCgeam hipblasCgeam
#define gpublasCgemm hipblasCgemm
#define gpublasCgemmBatched hipblasCgemmBatched
#define gpublasCgemv hipblasCgemv
#define gpublasCgerc hipblasCgerc
#define gpublasCgeru hipblasCgeru
#define gpublasChemm hipblasChemm
#define gpublasChemv hipblasChemv
#define gpublasCher2 hipblasCher2
#define gpublasCher2k hipblasCher2k
#define gpublasCherk hipblasCherk
#define gpublasCreate hipblasCreate
#define gpublasCscal hipblasCscal
#define gpublasCsscal hipblasCsscal
#define gpublasCswap hipblasCswap
#define gpublasCtrmm hipblasCtrmm
#define gpublasCtrmv hipblasCtrmv
#define gpublasCtrsm hipblasCtrsm
#define gpublasDaxpy hipblasDaxpy
#define gpublasDdot hipblasDdot
#define gpublasDestroy hipblasDestroy
#define gpublasDgeam hipblasDgeam
#define gpublasDgemm hipblasDgemm
#define gpublasDgemmBatched hipblasDgemmBatched
#define gpublasDgemv hipblasDgemv
#define gpublasDger hipblasDger
#define gpublasDnrm2 hipblasDnrm2
#define gpublasDscal hipblasDscal
#define gpublasDswap hipblasDswap
#define gpublasDsymm hipblasDsymm
#define gpublasDsymv hipblasDsymv
#define gpublasDsyr2 hipblasDsyr2
#define gpublasDsyr2k hipblasDsyr2k
#define gpublasDsyrk hipblasDsyrk
#define gpublasDtrmm hipblasDtrmm
#define gpublasDtrmv hipblasDtrmv
#define gpublasDtrsm hipblasDtrsm
#define gpublasDznrm2 hipblasDznrm2
#define gpublasGetPointerMode hipblasGetPointerMode
#define gpublasIcamax hipblasIcamax
#define gpublasIdamax hipblasIdamax
#define gpublasIsamax hipblasIsamax
#define gpublasIzamax hipblasIzamax
#define gpublasSaxpy hipblasSaxpy
#define gpublasScnrm2 hipblasScnrm2
#define gpublasSdot hipblasSdot
#define gpublasSetPointerMode hipblasSetPointerMode
#define gpublasSetStream hipblasSetStream
#define gpublasSgeam hipblasSgeam
#define gpublasSgemm hipblasSgemm
#define gpublasSgemmBatched hipblasSgemmBatched
#define gpublasSgemv hipblasSgemv
#define gpublasSger hipblasSger
#define gpublasSnrm2 hipblasSnrm2
#define gpublasSscal hipblasSscal
#define gpublasSswap hipblasSswap
#define gpublasSsymm hipblasSsymm
#define gpublasSsymv hipblasSsymv
#define gpublasSsyr2 hipblasSsyr2
#define gpublasSsyr2k hipblasSsyr2k
#define gpublasSsyrk hipblasSsyrk
#define gpublasStrmm hipblasStrmm
#define gpublasStrmv hipblasStrmv
#define gpublasStrsm hipblasStrsm
#define gpublasZaxpy hipblasZaxpy
#define gpublasZdotc hipblasZdotc
#define gpublasZdscal hipblasZdscal
#define gpublasZgeam hipblasZgeam
#define gpublasZgemm hipblasZgemm
#define gpublasZgemmBatched hipblasZgemmBatched
#define gpublasZgemv hipblasZgemv
#define gpublasZgerc hipblasZgerc
#define gpublasZgeru hipblasZgeru
#define gpublasZhemm hipblasZhemm
#define gpublasZhemv hipblasZhemv
#define gpublasZher2 hipblasZher2
#define gpublasZher2k hipblasZher2k
#define gpublasZherk hipblasZherk
#define gpublasZscal hipblasZscal
#define gpublasZswap hipblasZswap
#define gpublasZtrmm hipblasZtrmm
#define gpublasZtrmv hipblasZtrmv
#define gpublasZtrsm hipblasZtrsm

// solver
#define gpusolverCgeqrf hipsolverCgeqrf
#define gpusolverCgeqrf_bufferSize hipsolverCgeqrf_bufferSize
#define gpusolverCgetrf hipsolverCgetrf
#define gpusolverCgetrf_bufferSize hipsolverCgetrf_bufferSize
#define gpusolverCpotrf hipsolverCpotrf
#define gpusolverCpotrf_bufferSize hipsolverCpotrf_bufferSize
#define gpusolverCreate hipsolverCreate
#define gpusolverDestroy hipsolverDestroy
#define gpusolverDgeqrf hipsolverDgeqrf
#define gpusolverDgeqrf_bufferSize hipsolverDgeqrf_bufferSize
#define gpusolverDgetrf hipsolverDgetrf
#define gpusolverDgetrf_bufferSize hipsolverDgetrf_bufferSize
#define gpusolverDpotrf hipsolverDpotrf
#define gpusolverDpotrf_bufferSize hipsolverDpotrf_bufferSize
#define gpusolverGetStream hipsolverGetStream
#define gpusolverSetStream hipsolverSetStream
#define gpusolverSgeqrf hipsolverSgeqrf
#define gpusolverSgeqrf_bufferSize hipsolverSgeqrf_bufferSize
#define gpusolverSgetrf hipsolverSgetrf
#define gpusolverSgetrf_bufferSize hipsolverSgetrf_bufferSize
#define gpusolverSpotrf hipsolverSpotrf
#define gpusolverSpotrf_bufferSize hipsolverSpotrf_bufferSize
#define gpusolverZgeqrf hipsolverZgeqrf
#define gpusolverZgeqrf_bufferSize hipsolverZgeqrf_bufferSize
#define gpusolverZgetrf hipsolverZgetrf
#define gpusolverZgetrf_bufferSize hipsolverZgetrf_bufferSize
#define gpusolverZpotrf hipsolverZpotrf
#define gpusolverZpotrf_bufferSize hipsolverZpotrf_bufferSize

// rand
#define gpurandCreateGenerator hiprandCreateGenerator
#define gpurandDestroyGenerator hiprandDestroyGenerator
#define gpurandGenerateUniform hiprandGenerateUniform
#define gpurandGenerateUniformDouble hiprandGenerateUniformDouble
#define gpurandSetPseudoRandomGeneratorSeed hiprandSetPseudoRandomGeneratorSeed
#define gpurandSetStream hiprandSetStream

// collective communication (NCCL / RCCL)
#ifdef DDLA_USE_CCL
#define gpucclAllReduce ncclAllReduce
#define gpucclReduce ncclReduce
#define gpucclBcast ncclBcast
#define gpucclCommDestroy ncclCommDestroy
#define gpucclCommInitRank ncclCommInitRank
#define gpucclGetUniqueId ncclGetUniqueId
#define gpucclGroupEnd ncclGroupEnd
#define gpucclGroupStart ncclGroupStart
#define gpucclRecv ncclRecv
#define gpucclSend ncclSend
#endif

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

#define gpublasCsymm hipblasCsymm
#define gpublasCsyrk hipblasCsyrk
#define gpublasCsyr2k hipblasCsyr2k
#define gpublasZsymm hipblasZsymm
#define gpublasZsyrk hipblasZsyrk
#define gpublasZsyr2k hipblasZsyr2k

#endif // DDLA_GPU_HIP_H
