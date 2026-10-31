#ifndef DDLA_GPU_CUDA_H
#define DDLA_GPU_CUDA_H

// CUDA mapping of the vendor-neutral gpu* names (see gpu.h).
//
// This file and gpu_hip.h list the SAME names in the SAME order, so a side-by-side diff shows every
// vendor difference.  Only gpu/*.h may spell vendor API names; the rest of the library uses gpu*.
//   types      -> using          (inside namespace ddla)
//   constants  -> constexpr auto (inside namespace ddla)
//   functions  -> #define        (vendor functions can be overloaded or macros themselves)

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <curand.h>
#ifdef DDLA_USE_CCL
#include <nccl.h>
#endif

namespace ddla {

// ---- runtime -----------------------------------------------------------------------------------
using gpuDataType_t = cudaDataType_t;
using gpuError_t = cudaError_t;
using gpuEvent_t = cudaEvent_t;
using gpuMemcpyKind = cudaMemcpyKind;
using gpuPointerAttributes = cudaPointerAttributes;
using gpuStream_t = cudaStream_t;
constexpr auto GPU_C_32F = CUDA_C_32F;
constexpr auto GPU_C_64F = CUDA_C_64F;
constexpr auto GPU_R_32F = CUDA_R_32F;
constexpr auto GPU_R_64F = CUDA_R_64F;
constexpr auto gpuDevAttrMaxSharedMemoryPerBlock = cudaDevAttrMaxSharedMemoryPerBlock;
constexpr auto gpuErrorInvalidValue = cudaErrorInvalidValue;
constexpr auto gpuErrorNotReady = cudaErrorNotReady;
constexpr auto gpuHostAllocDefault = cudaHostAllocDefault;
constexpr auto gpuMemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;
constexpr auto gpuMemcpyDeviceToHost = cudaMemcpyDeviceToHost;
constexpr auto gpuMemcpyHostToDevice = cudaMemcpyHostToDevice;
constexpr auto gpuMemoryTypeDevice = cudaMemoryTypeDevice;
constexpr auto gpuMemoryTypeManaged = cudaMemoryTypeManaged;
constexpr auto gpuStreamNonBlocking = cudaStreamNonBlocking;
constexpr auto gpuSuccess = cudaSuccess;

// ---- BLAS --------------------------------------------------------------------------------------
using gpublasDiagType_t = cublasDiagType_t;
using gpublasFillMode_t = cublasFillMode_t;
using gpublasHandle_t = cublasHandle_t;
using gpublasOperation_t = cublasOperation_t;
using gpublasPointerMode_t = cublasPointerMode_t;
using gpublasSideMode_t = cublasSideMode_t;
using gpublasStatus_t = cublasStatus_t;
constexpr auto GPUBLAS_DIAG_NON_UNIT = CUBLAS_DIAG_NON_UNIT;
constexpr auto GPUBLAS_DIAG_UNIT = CUBLAS_DIAG_UNIT;
constexpr auto GPUBLAS_FILL_MODE_LOWER = CUBLAS_FILL_MODE_LOWER;
constexpr auto GPUBLAS_FILL_MODE_UPPER = CUBLAS_FILL_MODE_UPPER;
constexpr auto GPUBLAS_OP_C = CUBLAS_OP_C;
constexpr auto GPUBLAS_OP_N = CUBLAS_OP_N;
constexpr auto GPUBLAS_OP_T = CUBLAS_OP_T;
constexpr auto GPUBLAS_POINTER_MODE_DEVICE = CUBLAS_POINTER_MODE_DEVICE;
constexpr auto GPUBLAS_SIDE_LEFT = CUBLAS_SIDE_LEFT;
constexpr auto GPUBLAS_SIDE_RIGHT = CUBLAS_SIDE_RIGHT;
constexpr auto GPUBLAS_STATUS_SUCCESS = CUBLAS_STATUS_SUCCESS;
using gpublasDoubleComplex = cuDoubleComplex;
using gpublasComplex = cuFloatComplex;

// ---- solver ------------------------------------------------------------------------------------
using gpusolverHandle_t = cusolverDnHandle_t;
using gpusolverStatus_t = cusolverStatus_t;
constexpr auto GPUSOLVER_STATUS_SUCCESS = CUSOLVER_STATUS_SUCCESS;
using gpusolverDoubleComplex = cuDoubleComplex;
using gpusolverComplex = cuFloatComplex;
// cuSOLVER takes the cuBLAS fill mode directly: the converter is the identity.
using gpusolverFillMode_t = gpublasFillMode_t;
constexpr auto GPUSOLVER_FILL_MODE_LOWER = GPUBLAS_FILL_MODE_LOWER;
constexpr auto GPUSOLVER_FILL_MODE_UPPER = GPUBLAS_FILL_MODE_UPPER;
inline gpusolverFillMode_t gpusolverFillMode(gpublasFillMode_t uplo) { return uplo; }

// ---- rand --------------------------------------------------------------------------------------
using gpurandGenerator_t = curandGenerator_t;
using gpurandRngType = curandRngType;
using gpurandStatus_t = curandStatus_t;
constexpr auto GPURAND_RNG_PSEUDO_DEFAULT = CURAND_RNG_PSEUDO_DEFAULT;
constexpr auto GPURAND_STATUS_SUCCESS = CURAND_STATUS_SUCCESS;

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
#define gpuDeviceGetAttribute cudaDeviceGetAttribute
#define gpuDeviceGetStreamPriorityRange cudaDeviceGetStreamPriorityRange
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuEventCreate cudaEventCreate
#define gpuEventDestroy cudaEventDestroy
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuFree cudaFree
#define gpuFreeAsync cudaFreeAsync
#define gpuFreeHost cudaFreeHost
#define gpuGetDevice cudaGetDevice
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuGetErrorString cudaGetErrorString
#define gpuGetLastError cudaGetLastError
#define gpuHostAlloc cudaHostAlloc
#define gpuMalloc cudaMalloc
#define gpuMallocAsync cudaMallocAsync
#define gpuMallocManaged cudaMallocManaged
#define gpuMemGetInfo cudaMemGetInfo
#define gpuMemcpy cudaMemcpy
#define gpuMemcpy2DAsync cudaMemcpy2DAsync
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemsetAsync cudaMemsetAsync
#define gpuPointerGetAttributes cudaPointerGetAttributes
#define gpuSetDevice cudaSetDevice
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamCreateWithFlags cudaStreamCreateWithFlags
#define gpuStreamCreateWithPriority cudaStreamCreateWithPriority
#define gpuStreamDestroy cudaStreamDestroy
#define gpuStreamQuery cudaStreamQuery
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuStreamWaitEvent cudaStreamWaitEvent

// BLAS
#define gpublasCaxpy cublasCaxpy
#define gpublasCdotc cublasCdotc
#define gpublasCgeam cublasCgeam
#define gpublasCgemm cublasCgemm
#define gpublasCgemmBatched cublasCgemmBatched
#define gpublasCgemv cublasCgemv
#define gpublasCgerc cublasCgerc
#define gpublasCgeru cublasCgeru
#define gpublasChemm cublasChemm
#define gpublasChemv cublasChemv
#define gpublasCher2 cublasCher2
#define gpublasCher2k cublasCher2k
#define gpublasCherk cublasCherk
#define gpublasCreate cublasCreate
#define gpublasCscal cublasCscal
#define gpublasCsscal cublasCsscal
#define gpublasCswap cublasCswap
#define gpublasCtrmm cublasCtrmm
#define gpublasCtrmv cublasCtrmv
#define gpublasCtrsm cublasCtrsm
#define gpublasDaxpy cublasDaxpy
#define gpublasDdot cublasDdot
#define gpublasDestroy cublasDestroy
#define gpublasDgeam cublasDgeam
#define gpublasDgemm cublasDgemm
#define gpublasDgemmBatched cublasDgemmBatched
#define gpublasDgemv cublasDgemv
#define gpublasDger cublasDger
#define gpublasDnrm2 cublasDnrm2
#define gpublasDscal cublasDscal
#define gpublasDswap cublasDswap
#define gpublasDsymm cublasDsymm
#define gpublasDsymv cublasDsymv
#define gpublasDsyr2 cublasDsyr2
#define gpublasDsyr2k cublasDsyr2k
#define gpublasDsyrk cublasDsyrk
#define gpublasDtrmm cublasDtrmm
#define gpublasDtrmv cublasDtrmv
#define gpublasDtrsm cublasDtrsm
#define gpublasDznrm2 cublasDznrm2
#define gpublasGetPointerMode cublasGetPointerMode
#define gpublasIcamax cublasIcamax
#define gpublasIdamax cublasIdamax
#define gpublasIsamax cublasIsamax
#define gpublasIzamax cublasIzamax
#define gpublasSaxpy cublasSaxpy
#define gpublasScnrm2 cublasScnrm2
#define gpublasSdot cublasSdot
#define gpublasSetPointerMode cublasSetPointerMode
#define gpublasSetStream cublasSetStream
#define gpublasSgeam cublasSgeam
#define gpublasSgemm cublasSgemm
#define gpublasSgemmBatched cublasSgemmBatched
#define gpublasSgemv cublasSgemv
#define gpublasSger cublasSger
#define gpublasSnrm2 cublasSnrm2
#define gpublasSscal cublasSscal
#define gpublasSswap cublasSswap
#define gpublasSsymm cublasSsymm
#define gpublasSsymv cublasSsymv
#define gpublasSsyr2 cublasSsyr2
#define gpublasSsyr2k cublasSsyr2k
#define gpublasSsyrk cublasSsyrk
#define gpublasStrmm cublasStrmm
#define gpublasStrmv cublasStrmv
#define gpublasStrsm cublasStrsm
#define gpublasZaxpy cublasZaxpy
#define gpublasZdotc cublasZdotc
#define gpublasZdscal cublasZdscal
#define gpublasZgeam cublasZgeam
#define gpublasZgemm cublasZgemm
#define gpublasZgemmBatched cublasZgemmBatched
#define gpublasZgemv cublasZgemv
#define gpublasZgerc cublasZgerc
#define gpublasZgeru cublasZgeru
#define gpublasZhemm cublasZhemm
#define gpublasZhemv cublasZhemv
#define gpublasZher2 cublasZher2
#define gpublasZher2k cublasZher2k
#define gpublasZherk cublasZherk
#define gpublasZscal cublasZscal
#define gpublasZswap cublasZswap
#define gpublasZtrmm cublasZtrmm
#define gpublasZtrmv cublasZtrmv
#define gpublasZtrsm cublasZtrsm

// solver
#define gpusolverCgeqrf cusolverDnCgeqrf
#define gpusolverCgeqrf_bufferSize cusolverDnCgeqrf_bufferSize
#define gpusolverCgetrf_bufferSize cusolverDnCgetrf_bufferSize
#define gpusolverClaswp cusolverDnClaswp
#define gpusolverCpotrf cusolverDnCpotrf
#define gpusolverCpotrf_bufferSize cusolverDnCpotrf_bufferSize
#define gpusolverCreate cusolverDnCreate
#define gpusolverDestroy cusolverDnDestroy
#define gpusolverDgeqrf cusolverDnDgeqrf
#define gpusolverDgeqrf_bufferSize cusolverDnDgeqrf_bufferSize
#define gpusolverDgetrf_bufferSize cusolverDnDgetrf_bufferSize
#define gpusolverDlaswp cusolverDnDlaswp
#define gpusolverDpotrf cusolverDnDpotrf
#define gpusolverDpotrf_bufferSize cusolverDnDpotrf_bufferSize
#define gpusolverGetStream cusolverDnGetStream
#define gpusolverSetStream cusolverDnSetStream
#define gpusolverSgeqrf cusolverDnSgeqrf
#define gpusolverSgeqrf_bufferSize cusolverDnSgeqrf_bufferSize
#define gpusolverSgetrf_bufferSize cusolverDnSgetrf_bufferSize
#define gpusolverSlaswp cusolverDnSlaswp
#define gpusolverSpotrf cusolverDnSpotrf
#define gpusolverSpotrf_bufferSize cusolverDnSpotrf_bufferSize
#define gpusolverZgeqrf cusolverDnZgeqrf
#define gpusolverZgeqrf_bufferSize cusolverDnZgeqrf_bufferSize
#define gpusolverZgetrf_bufferSize cusolverDnZgetrf_bufferSize
#define gpusolverZlaswp cusolverDnZlaswp
#define gpusolverZpotrf cusolverDnZpotrf
#define gpusolverZpotrf_bufferSize cusolverDnZpotrf_bufferSize
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

// rand
#define gpurandCreateGenerator curandCreateGenerator
#define gpurandDestroyGenerator curandDestroyGenerator
#define gpurandGenerateUniform curandGenerateUniform
#define gpurandGenerateUniformDouble curandGenerateUniformDouble
#define gpurandSetPseudoRandomGeneratorSeed curandSetPseudoRandomGeneratorSeed
#define gpurandSetStream curandSetStream

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

#define gpublasCsymm cublasCsymm
#define gpublasCsyrk cublasCsyrk
#define gpublasCsyr2k cublasCsyr2k
#define gpublasZsymm cublasZsymm
#define gpublasZsyrk cublasZsyrk
#define gpublasZsyr2k cublasZsyr2k

#endif // DDLA_GPU_CUDA_H
