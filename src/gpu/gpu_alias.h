#ifndef DDLA_GPU_ALIAS_H
#define DDLA_GPU_ALIAS_H

// The gpu* names that differ between the vendors only by the family prefix (see gpu.h).  gpu_cuda.h and gpu_hip.h
// define the prefix macros and include this list; whatever is spelled differently, has a different signature or
// exists for one vendor only stays in the vendor header.
//
//   DDLA_GPU_RT(Name)          cudaName        / hipName          DDLA_GPU_RT_CONST(NAME)      CUDA_NAME      / HIP_NAME
//   DDLA_GPU_BLAS(Name)        cublasName      / hipblasName      DDLA_GPU_BLAS_CONST(NAME)    CUBLAS_NAME    / HIPBLAS_NAME
//   DDLA_GPU_SOLVER(Name)      cusolverDnName  / hipsolverName    DDLA_GPU_SOLVER_CONST(NAME)  CUSOLVER_NAME  / HIPSOLVER_NAME
//   DDLA_GPU_SOLVER_BASE(Name) cusolverName    / hipsolverName
//   DDLA_GPU_RAND(Name)        curandName      / hiprandName      DDLA_GPU_RAND_CONST(NAME)    CURAND_NAME    / HIPRAND_NAME
//
// The collective-communication names are the same for NCCL and RCCL.
//   types      -> using          (inside namespace ddla)
//   constants  -> constexpr auto (inside namespace ddla)
//   functions  -> #define        (vendor functions can be overloaded or macros themselves)

namespace ddla {

// ---- runtime -----------------------------------------------------------------------------------
using gpuError_t = DDLA_GPU_RT(Error_t);
using gpuEvent_t = DDLA_GPU_RT(Event_t);
using gpuMemcpyKind = DDLA_GPU_RT(MemcpyKind);
using gpuStream_t = DDLA_GPU_RT(Stream_t);
constexpr auto GPU_C_32F = DDLA_GPU_RT_CONST(C_32F);
constexpr auto GPU_C_64F = DDLA_GPU_RT_CONST(C_64F);
constexpr auto GPU_R_32F = DDLA_GPU_RT_CONST(R_32F);
constexpr auto GPU_R_64F = DDLA_GPU_RT_CONST(R_64F);
constexpr auto gpuErrorInvalidValue = DDLA_GPU_RT(ErrorInvalidValue);
constexpr auto gpuErrorNotReady = DDLA_GPU_RT(ErrorNotReady);
constexpr auto gpuMemcpyDeviceToDevice = DDLA_GPU_RT(MemcpyDeviceToDevice);
constexpr auto gpuMemcpyDeviceToHost = DDLA_GPU_RT(MemcpyDeviceToHost);
constexpr auto gpuMemcpyHostToDevice = DDLA_GPU_RT(MemcpyHostToDevice);
constexpr auto gpuMemoryTypeDevice = DDLA_GPU_RT(MemoryTypeDevice);
constexpr auto gpuMemoryTypeManaged = DDLA_GPU_RT(MemoryTypeManaged);
constexpr auto gpuStreamNonBlocking = DDLA_GPU_RT(StreamNonBlocking);
constexpr auto gpuSuccess = DDLA_GPU_RT(Success);

// ---- BLAS --------------------------------------------------------------------------------------
using gpublasDiagType_t = DDLA_GPU_BLAS(DiagType_t);
using gpublasFillMode_t = DDLA_GPU_BLAS(FillMode_t);
using gpublasHandle_t = DDLA_GPU_BLAS(Handle_t);
using gpublasOperation_t = DDLA_GPU_BLAS(Operation_t);
using gpublasPointerMode_t = DDLA_GPU_BLAS(PointerMode_t);
using gpublasSideMode_t = DDLA_GPU_BLAS(SideMode_t);
using gpublasStatus_t = DDLA_GPU_BLAS(Status_t);
constexpr auto GPUBLAS_DIAG_NON_UNIT = DDLA_GPU_BLAS_CONST(DIAG_NON_UNIT);
constexpr auto GPUBLAS_DIAG_UNIT = DDLA_GPU_BLAS_CONST(DIAG_UNIT);
constexpr auto GPUBLAS_FILL_MODE_LOWER = DDLA_GPU_BLAS_CONST(FILL_MODE_LOWER);
constexpr auto GPUBLAS_FILL_MODE_UPPER = DDLA_GPU_BLAS_CONST(FILL_MODE_UPPER);
constexpr auto GPUBLAS_OP_C = DDLA_GPU_BLAS_CONST(OP_C);
constexpr auto GPUBLAS_OP_N = DDLA_GPU_BLAS_CONST(OP_N);
constexpr auto GPUBLAS_OP_T = DDLA_GPU_BLAS_CONST(OP_T);
constexpr auto GPUBLAS_POINTER_MODE_DEVICE = DDLA_GPU_BLAS_CONST(POINTER_MODE_DEVICE);
constexpr auto GPUBLAS_SIDE_LEFT = DDLA_GPU_BLAS_CONST(SIDE_LEFT);
constexpr auto GPUBLAS_SIDE_RIGHT = DDLA_GPU_BLAS_CONST(SIDE_RIGHT);
constexpr auto GPUBLAS_STATUS_SUCCESS = DDLA_GPU_BLAS_CONST(STATUS_SUCCESS);

// ---- solver ------------------------------------------------------------------------------------
using gpusolverHandle_t = DDLA_GPU_SOLVER(Handle_t);
using gpusolverStatus_t = DDLA_GPU_SOLVER_BASE(Status_t);
using gpusolverEigMode_t = DDLA_GPU_SOLVER_BASE(EigMode_t);
constexpr auto GPUSOLVER_EIG_MODE_VECTOR = DDLA_GPU_SOLVER_CONST(EIG_MODE_VECTOR);
constexpr auto GPUSOLVER_EIG_MODE_NOVECTOR = DDLA_GPU_SOLVER_CONST(EIG_MODE_NOVECTOR);
constexpr auto GPUSOLVER_STATUS_SUCCESS = DDLA_GPU_SOLVER_CONST(STATUS_SUCCESS);

// ---- rand --------------------------------------------------------------------------------------
using gpurandGenerator_t = DDLA_GPU_RAND(Generator_t);
using gpurandRngType = DDLA_GPU_RAND(RngType);
using gpurandStatus_t = DDLA_GPU_RAND(Status_t);
constexpr auto GPURAND_RNG_PSEUDO_DEFAULT = DDLA_GPU_RAND_CONST(RNG_PSEUDO_DEFAULT);
constexpr auto GPURAND_STATUS_SUCCESS = DDLA_GPU_RAND_CONST(STATUS_SUCCESS);

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
#define gpuDeviceGetAttribute DDLA_GPU_RT(DeviceGetAttribute)
#define gpuDeviceGetStreamPriorityRange DDLA_GPU_RT(DeviceGetStreamPriorityRange)
#define gpuDeviceSynchronize DDLA_GPU_RT(DeviceSynchronize)
#define gpuEventCreate DDLA_GPU_RT(EventCreate)
#define gpuEventDestroy DDLA_GPU_RT(EventDestroy)
#define gpuEventRecord DDLA_GPU_RT(EventRecord)
#define gpuEventSynchronize DDLA_GPU_RT(EventSynchronize)
#define gpuFree DDLA_GPU_RT(Free)
#define gpuFreeAsync DDLA_GPU_RT(FreeAsync)
#define gpuGetDevice DDLA_GPU_RT(GetDevice)
#define gpuGetDeviceCount DDLA_GPU_RT(GetDeviceCount)
#define gpuGetErrorString DDLA_GPU_RT(GetErrorString)
#define gpuGetLastError DDLA_GPU_RT(GetLastError)
#define gpuMalloc DDLA_GPU_RT(Malloc)
#define gpuMallocAsync DDLA_GPU_RT(MallocAsync)
#define gpuMallocManaged DDLA_GPU_RT(MallocManaged)
#define gpuMemGetInfo DDLA_GPU_RT(MemGetInfo)
#define gpuMemcpy DDLA_GPU_RT(Memcpy)
#define gpuMemcpy2DAsync DDLA_GPU_RT(Memcpy2DAsync)
#define gpuMemcpyAsync DDLA_GPU_RT(MemcpyAsync)
#define gpuMemsetAsync DDLA_GPU_RT(MemsetAsync)
#define gpuPointerGetAttributes DDLA_GPU_RT(PointerGetAttributes)
#define gpuSetDevice DDLA_GPU_RT(SetDevice)
#define gpuStreamCreate DDLA_GPU_RT(StreamCreate)
#define gpuStreamCreateWithFlags DDLA_GPU_RT(StreamCreateWithFlags)
#define gpuStreamCreateWithPriority DDLA_GPU_RT(StreamCreateWithPriority)
#define gpuStreamDestroy DDLA_GPU_RT(StreamDestroy)
#define gpuStreamQuery DDLA_GPU_RT(StreamQuery)
#define gpuStreamSynchronize DDLA_GPU_RT(StreamSynchronize)
#define gpuStreamWaitEvent DDLA_GPU_RT(StreamWaitEvent)

// BLAS
#define gpublasCaxpy DDLA_GPU_BLAS(Caxpy)
#define gpublasCdotc DDLA_GPU_BLAS(Cdotc)
#define gpublasCgeam DDLA_GPU_BLAS(Cgeam)
#define gpublasCgemm DDLA_GPU_BLAS(Cgemm)
#define gpublasCgemmBatched DDLA_GPU_BLAS(CgemmBatched)
#define gpublasCgemv DDLA_GPU_BLAS(Cgemv)
#define gpublasCgerc DDLA_GPU_BLAS(Cgerc)
#define gpublasCgeru DDLA_GPU_BLAS(Cgeru)
#define gpublasChemm DDLA_GPU_BLAS(Chemm)
#define gpublasChemv DDLA_GPU_BLAS(Chemv)
#define gpublasCher2 DDLA_GPU_BLAS(Cher2)
#define gpublasCher2k DDLA_GPU_BLAS(Cher2k)
#define gpublasCherk DDLA_GPU_BLAS(Cherk)
#define gpublasCreate DDLA_GPU_BLAS(Create)
#define gpublasCscal DDLA_GPU_BLAS(Cscal)
#define gpublasCsscal DDLA_GPU_BLAS(Csscal)
#define gpublasCswap DDLA_GPU_BLAS(Cswap)
#define gpublasCtrmm DDLA_GPU_BLAS(Ctrmm)
#define gpublasCtrmv DDLA_GPU_BLAS(Ctrmv)
#define gpublasCtrsm DDLA_GPU_BLAS(Ctrsm)
#define gpublasDaxpy DDLA_GPU_BLAS(Daxpy)
#define gpublasDdot DDLA_GPU_BLAS(Ddot)
#define gpublasDestroy DDLA_GPU_BLAS(Destroy)
#define gpublasDgeam DDLA_GPU_BLAS(Dgeam)
#define gpublasDgemm DDLA_GPU_BLAS(Dgemm)
#define gpublasDgemmBatched DDLA_GPU_BLAS(DgemmBatched)
#define gpublasDgemv DDLA_GPU_BLAS(Dgemv)
#define gpublasDger DDLA_GPU_BLAS(Dger)
#define gpublasDnrm2 DDLA_GPU_BLAS(Dnrm2)
#define gpublasDscal DDLA_GPU_BLAS(Dscal)
#define gpublasDswap DDLA_GPU_BLAS(Dswap)
#define gpublasDsymm DDLA_GPU_BLAS(Dsymm)
#define gpublasDsymv DDLA_GPU_BLAS(Dsymv)
#define gpublasDsyr2 DDLA_GPU_BLAS(Dsyr2)
#define gpublasDsyr2k DDLA_GPU_BLAS(Dsyr2k)
#define gpublasDsyrk DDLA_GPU_BLAS(Dsyrk)
#define gpublasDtrmm DDLA_GPU_BLAS(Dtrmm)
#define gpublasDtrmv DDLA_GPU_BLAS(Dtrmv)
#define gpublasDtrsm DDLA_GPU_BLAS(Dtrsm)
#define gpublasDznrm2 DDLA_GPU_BLAS(Dznrm2)
#define gpublasGetPointerMode DDLA_GPU_BLAS(GetPointerMode)
#define gpublasGetVersion DDLA_GPU_BLAS(GetVersion)
#define gpublasIcamax DDLA_GPU_BLAS(Icamax)
#define gpublasIdamax DDLA_GPU_BLAS(Idamax)
#define gpublasIsamax DDLA_GPU_BLAS(Isamax)
#define gpublasIzamax DDLA_GPU_BLAS(Izamax)
#define gpublasSaxpy DDLA_GPU_BLAS(Saxpy)
#define gpublasScnrm2 DDLA_GPU_BLAS(Scnrm2)
#define gpublasSdot DDLA_GPU_BLAS(Sdot)
#define gpublasSetPointerMode DDLA_GPU_BLAS(SetPointerMode)
#define gpublasSetStream DDLA_GPU_BLAS(SetStream)
#define gpublasSgeam DDLA_GPU_BLAS(Sgeam)
#define gpublasSgemm DDLA_GPU_BLAS(Sgemm)
#define gpublasSgemmBatched DDLA_GPU_BLAS(SgemmBatched)
#define gpublasSgemv DDLA_GPU_BLAS(Sgemv)
#define gpublasSger DDLA_GPU_BLAS(Sger)
#define gpublasSnrm2 DDLA_GPU_BLAS(Snrm2)
#define gpublasSscal DDLA_GPU_BLAS(Sscal)
#define gpublasSswap DDLA_GPU_BLAS(Sswap)
#define gpublasSsymm DDLA_GPU_BLAS(Ssymm)
#define gpublasSsymv DDLA_GPU_BLAS(Ssymv)
#define gpublasSsyr2 DDLA_GPU_BLAS(Ssyr2)
#define gpublasSsyr2k DDLA_GPU_BLAS(Ssyr2k)
#define gpublasSsyrk DDLA_GPU_BLAS(Ssyrk)
#define gpublasStrmm DDLA_GPU_BLAS(Strmm)
#define gpublasStrmv DDLA_GPU_BLAS(Strmv)
#define gpublasStrsm DDLA_GPU_BLAS(Strsm)
#define gpublasZaxpy DDLA_GPU_BLAS(Zaxpy)
#define gpublasZdotc DDLA_GPU_BLAS(Zdotc)
#define gpublasZdscal DDLA_GPU_BLAS(Zdscal)
#define gpublasZgeam DDLA_GPU_BLAS(Zgeam)
#define gpublasZgemm DDLA_GPU_BLAS(Zgemm)
#define gpublasZgemmBatched DDLA_GPU_BLAS(ZgemmBatched)
#define gpublasZgemv DDLA_GPU_BLAS(Zgemv)
#define gpublasZgerc DDLA_GPU_BLAS(Zgerc)
#define gpublasZgeru DDLA_GPU_BLAS(Zgeru)
#define gpublasZhemm DDLA_GPU_BLAS(Zhemm)
#define gpublasZhemv DDLA_GPU_BLAS(Zhemv)
#define gpublasZher2 DDLA_GPU_BLAS(Zher2)
#define gpublasZher2k DDLA_GPU_BLAS(Zher2k)
#define gpublasZherk DDLA_GPU_BLAS(Zherk)
#define gpublasZscal DDLA_GPU_BLAS(Zscal)
#define gpublasZswap DDLA_GPU_BLAS(Zswap)
#define gpublasZtrmm DDLA_GPU_BLAS(Ztrmm)
#define gpublasZtrmv DDLA_GPU_BLAS(Ztrmv)
#define gpublasZtrsm DDLA_GPU_BLAS(Ztrsm)

#define gpublasCsymm DDLA_GPU_BLAS(Csymm)
#define gpublasCsyrk DDLA_GPU_BLAS(Csyrk)
#define gpublasCsyr2k DDLA_GPU_BLAS(Csyr2k)
#define gpublasZsymm DDLA_GPU_BLAS(Zsymm)
#define gpublasZsyrk DDLA_GPU_BLAS(Zsyrk)
#define gpublasZsyr2k DDLA_GPU_BLAS(Zsyr2k)
// Diagonal matrix multiplication.
#define gpublasSdgmm DDLA_GPU_BLAS(Sdgmm)
#define gpublasDdgmm DDLA_GPU_BLAS(Ddgmm)
#define gpublasCdgmm DDLA_GPU_BLAS(Cdgmm)
#define gpublasZdgmm DDLA_GPU_BLAS(Zdgmm)

// solver
#define gpusolverCgeqrf DDLA_GPU_SOLVER(Cgeqrf)
#define gpusolverCgeqrf_bufferSize DDLA_GPU_SOLVER(Cgeqrf_bufferSize)
#define gpusolverCgetrf_bufferSize DDLA_GPU_SOLVER(Cgetrf_bufferSize)
#define gpusolverCpotrf DDLA_GPU_SOLVER(Cpotrf)
#define gpusolverCpotrf_bufferSize DDLA_GPU_SOLVER(Cpotrf_bufferSize)
#define gpusolverCreate DDLA_GPU_SOLVER(Create)
#define gpusolverDestroy DDLA_GPU_SOLVER(Destroy)
#define gpusolverDgeqrf DDLA_GPU_SOLVER(Dgeqrf)
#define gpusolverDgeqrf_bufferSize DDLA_GPU_SOLVER(Dgeqrf_bufferSize)
#define gpusolverDgetrf_bufferSize DDLA_GPU_SOLVER(Dgetrf_bufferSize)
#define gpusolverDpotrf DDLA_GPU_SOLVER(Dpotrf)
#define gpusolverDpotrf_bufferSize DDLA_GPU_SOLVER(Dpotrf_bufferSize)
#define gpusolverGetStream DDLA_GPU_SOLVER(GetStream)
#define gpusolverSetStream DDLA_GPU_SOLVER(SetStream)
#define gpusolverSgeqrf DDLA_GPU_SOLVER(Sgeqrf)
#define gpusolverSgeqrf_bufferSize DDLA_GPU_SOLVER(Sgeqrf_bufferSize)
#define gpusolverSgetrf_bufferSize DDLA_GPU_SOLVER(Sgetrf_bufferSize)
#define gpusolverSpotrf DDLA_GPU_SOLVER(Spotrf)
#define gpusolverSpotrf_bufferSize DDLA_GPU_SOLVER(Spotrf_bufferSize)
#define gpusolverZgeqrf DDLA_GPU_SOLVER(Zgeqrf)
#define gpusolverZgeqrf_bufferSize DDLA_GPU_SOLVER(Zgeqrf_bufferSize)
#define gpusolverZgetrf_bufferSize DDLA_GPU_SOLVER(Zgetrf_bufferSize)
#define gpusolverZpotrf DDLA_GPU_SOLVER(Zpotrf)
#define gpusolverZpotrf_bufferSize DDLA_GPU_SOLVER(Zpotrf_bufferSize)

#define gpusolverSsyevd DDLA_GPU_SOLVER(Ssyevd)
#define gpusolverDsyevd DDLA_GPU_SOLVER(Dsyevd)
#define gpusolverCheevd DDLA_GPU_SOLVER(Cheevd)
#define gpusolverZheevd DDLA_GPU_SOLVER(Zheevd)
#define gpusolverSsyevd_bufferSize DDLA_GPU_SOLVER(Ssyevd_bufferSize)
#define gpusolverDsyevd_bufferSize DDLA_GPU_SOLVER(Dsyevd_bufferSize)
#define gpusolverCheevd_bufferSize DDLA_GPU_SOLVER(Cheevd_bufferSize)
#define gpusolverZheevd_bufferSize DDLA_GPU_SOLVER(Zheevd_bufferSize)

#define gpusolverSgesvd DDLA_GPU_SOLVER(Sgesvd)
#define gpusolverDgesvd DDLA_GPU_SOLVER(Dgesvd)
#define gpusolverCgesvd DDLA_GPU_SOLVER(Cgesvd)
#define gpusolverZgesvd DDLA_GPU_SOLVER(Zgesvd)

#define gpusolverSgebrd DDLA_GPU_SOLVER(Sgebrd)
#define gpusolverSgebrd_bufferSize DDLA_GPU_SOLVER(Sgebrd_bufferSize)
#define gpusolverDgebrd DDLA_GPU_SOLVER(Dgebrd)
#define gpusolverDgebrd_bufferSize DDLA_GPU_SOLVER(Dgebrd_bufferSize)
#define gpusolverCgebrd DDLA_GPU_SOLVER(Cgebrd)
#define gpusolverCgebrd_bufferSize DDLA_GPU_SOLVER(Cgebrd_bufferSize)
#define gpusolverZgebrd DDLA_GPU_SOLVER(Zgebrd)
#define gpusolverZgebrd_bufferSize DDLA_GPU_SOLVER(Zgebrd_bufferSize)

// rand
#define gpurandCreateGenerator DDLA_GPU_RAND(CreateGenerator)
#define gpurandDestroyGenerator DDLA_GPU_RAND(DestroyGenerator)
#define gpurandGenerateUniform DDLA_GPU_RAND(GenerateUniform)
#define gpurandGenerateUniformDouble DDLA_GPU_RAND(GenerateUniformDouble)
#define gpurandSetPseudoRandomGeneratorSeed DDLA_GPU_RAND(SetPseudoRandomGeneratorSeed)
#define gpurandSetStream DDLA_GPU_RAND(SetStream)

// collective communication (NCCL / RCCL)
#ifdef DDLA_USE_CCL
#define gpucclAllReduce ncclAllReduce
#define gpucclAllGather ncclAllGather
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

#endif // DDLA_GPU_ALIAS_H
