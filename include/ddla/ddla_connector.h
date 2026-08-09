#ifndef DDLA_CONNECTOR_H
#define DDLA_CONNECTOR_H

#include <mpi.h>
#include <iostream>
#ifdef DDLA_USE_CUDA
#ifdef DDLA_USE_CCL
#include <nccl.h>
#endif
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <curand.h>
#endif
#ifdef DDLA_USE_HIP
#ifdef DDLA_USE_CCL
#include <rccl/rccl.h>
#endif
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>
#include <hiprand/hiprand.h>
#endif

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <new>
#include <complex>
#include <type_traits>
#include <stdexcept>
#include <string>
#include "ddla_handle_t.h"

namespace ddla{

#ifdef DDLA_USE_CCL
using cclOp=ncclRedOp_t;
const auto cclSum=ncclRedOp_t::ncclSum;
#else
using cclOp=MPI_Op;
const auto cclSum=MPI_SUM;
#endif

// ---------------------------------------------------------------------------
// Runtime abstraction: a DdlaBackend-templated trait class replaces the old
// per-vendor #define family. Every TU compiles with exactly one of
// DDLA_USE_CUDA / DDLA_USE_HIP / DDLA_USE_CPU active (see src/CMakeLists.txt);
// `local_backend_v` reflects *that*, not the project-wide DDLA_HAS_GPU/
// DDLA_HAS_CPU capability flags (`ddla::default_backend_v` in
// ddla_handle_t.h). `RuntimeTraits<DdlaBackend::GPU>` is only ever *defined*
// in a TU that actually compiled in a GPU vendor (guarded by the same
// DDLA_USE_CUDA/DDLA_USE_HIP check, right above) -- so the runtime functions
// below must default their Backend template parameter to this TU-local
// constant, not the project-wide one, or a TU compiled without a GPU vendor
// active would try to instantiate the (undefined) GPU specialization and
// fail to compile.
// ---------------------------------------------------------------------------
namespace detail {

#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
constexpr DdlaBackend local_backend_v = DdlaBackend::GPU;
#elif defined(DDLA_USE_CPU)
constexpr DdlaBackend local_backend_v = DdlaBackend::CPU;
#endif

template <DdlaBackend Backend> struct RuntimeTraits;   // primary: intentionally undefined

template <>
struct RuntimeTraits<DdlaBackend::CPU> {
    using stream_t = int;
    using error_t = int;
    using event_t = int;
    using memcpy_kind_t = int;
    using data_type_t = int;
    static constexpr error_t success = 0;
    static constexpr memcpy_kind_t host_to_device = 0;
    static constexpr memcpy_kind_t device_to_host = 1;
    static constexpr memcpy_kind_t device_to_device = 2;
    static constexpr data_type_t r_64f = 0;
    static constexpr data_type_t c_64f = 0;
    static constexpr data_type_t r_32f = 0;
    static constexpr data_type_t c_32f = 0;
};

#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
template <>
struct RuntimeTraits<DdlaBackend::GPU> {
#if defined(DDLA_USE_CUDA)
    using stream_t = cudaStream_t;
    using error_t = cudaError_t;
    using event_t = cudaEvent_t;
    using memcpy_kind_t = cudaMemcpyKind;
    using data_type_t = cudaDataType_t;
    static constexpr error_t success = cudaError_t::cudaSuccess;
    static constexpr memcpy_kind_t host_to_device = memcpy_kind_t::cudaMemcpyHostToDevice;
    static constexpr memcpy_kind_t device_to_host = memcpy_kind_t::cudaMemcpyDeviceToHost;
    static constexpr memcpy_kind_t device_to_device = memcpy_kind_t::cudaMemcpyDeviceToDevice;
    static constexpr data_type_t r_64f = data_type_t::CUDA_R_64F;
    static constexpr data_type_t c_64f = data_type_t::CUDA_C_64F;
    static constexpr data_type_t r_32f = data_type_t::CUDA_R_32F;
    static constexpr data_type_t c_32f = data_type_t::CUDA_C_32F;
#elif defined(DDLA_USE_HIP)
    using stream_t = hipStream_t;
    using error_t = hipError_t;
    using event_t = hipEvent_t;
    using memcpy_kind_t = hipMemcpyKind;
    using data_type_t = hipDataType;
    static constexpr error_t success = hipError_t::hipSuccess;
    static constexpr memcpy_kind_t host_to_device = memcpy_kind_t::hipMemcpyHostToDevice;
    static constexpr memcpy_kind_t device_to_host = memcpy_kind_t::hipMemcpyDeviceToHost;
    static constexpr memcpy_kind_t device_to_device = memcpy_kind_t::hipMemcpyDeviceToDevice;
    static constexpr data_type_t r_64f = data_type_t::HIP_R_64F;
    static constexpr data_type_t c_64f = data_type_t::HIP_C_64F;
    static constexpr data_type_t r_32f = data_type_t::HIP_R_32F;
    static constexpr data_type_t c_32f = data_type_t::HIP_C_32F;
#endif
};
#endif // defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
// RuntimeTraits<DdlaBackend::GPU> is only ever *defined* in a TU that
// actually compiled in a GPU vendor. Naming it in a CPU-only TU is an
// incomplete-type compile error -- the same "catch backend mismatch at
// compile time" property the old per-TU #ifdef selection already had.

} // namespace detail

// ---------------------------------------------------------------------------
// deblas* / desolver* / derand* families: vendor BLAS/solver/RNG handle and
// status types, unchanged by this refactor (already correctly namespaced,
// not part of the runtime-abstraction rework).
// ---------------------------------------------------------------------------
#ifdef DDLA_USE_CUDA
using deblasStatus_t = cublasStatus_t;
constexpr auto DEBLAS_STATUS_SUCCESS = deblasStatus_t::CUBLAS_STATUS_SUCCESS;
using deblasHandle_t = cublasHandle_t;
using desolverHandle_t = cusolverDnHandle_t;
using desolverStatus_t = cusolverStatus_t;
constexpr auto DESOLVER_STATUS_SUCCESS = desolverStatus_t::CUSOLVER_STATUS_SUCCESS;
#define desolverGetStream cusolverDnGetStream
using derandGenerator_t = curandGenerator_t;
using derandStatus_t = curandStatus_t;
constexpr auto DERAND_STATUS_SUCCESS = derandStatus_t::CURAND_STATUS_SUCCESS;
#define derandCreateGenerator curandCreateGenerator
#define derandSetPseudoRandomGeneratorSeed curandSetPseudoRandomGeneratorSeed
#define derandGenerateUniform curandGenerateUniform
#define derandGenerateUniformDouble curandGenerateUniformDouble
#define derandDestroyGenerator curandDestroyGenerator
using derandRngType = curandRngType;
constexpr auto DERAND_RNG_PSEUDO_DEFAULT = derandRngType::CURAND_RNG_PSEUDO_DEFAULT;
using deblasSideMode_t = cublasSideMode_t;
constexpr auto DEBLAS_SIDE_LEFT = deblasSideMode_t::CUBLAS_SIDE_LEFT;
constexpr auto DEBLAS_SIDE_RIGHT = deblasSideMode_t::CUBLAS_SIDE_RIGHT;
using deblasFillMode_t = cublasFillMode_t;
constexpr auto DEBLAS_FILL_MODE_LOWER = deblasFillMode_t::CUBLAS_FILL_MODE_LOWER;
constexpr auto DEBLAS_FILL_MODE_UPPER = deblasFillMode_t::CUBLAS_FILL_MODE_UPPER;
using deblasDiagType_t = cublasDiagType_t;
constexpr auto DEBLAS_DIAG_UNIT = deblasDiagType_t::CUBLAS_DIAG_UNIT;
constexpr auto DEBLAS_DIAG_NON_UNIT = deblasDiagType_t::CUBLAS_DIAG_NON_UNIT;
using deblasOperation_t = cublasOperation_t;
constexpr auto DEBLAS_OP_N = deblasOperation_t::CUBLAS_OP_N;
constexpr auto DEBLAS_OP_T = deblasOperation_t::CUBLAS_OP_T;
constexpr auto DEBLAS_OP_C = deblasOperation_t::CUBLAS_OP_C;

#endif
#ifdef DDLA_USE_HIP
using deblasStatus_t = hipblasStatus_t;
constexpr auto DEBLAS_STATUS_SUCCESS = deblasStatus_t::HIPBLAS_STATUS_SUCCESS;
using deblasHandle_t = hipblasHandle_t;
using desolverHandle_t = hipsolverHandle_t;
using desolverStatus_t = hipsolverStatus_t;
constexpr auto DESOLVER_STATUS_SUCCESS = desolverStatus_t::HIPSOLVER_STATUS_SUCCESS;
#define desolverGetStream hipsolverGetStream
using derandGenerator_t = hiprandGenerator_t;
using derandStatus_t = hiprandStatus_t;
constexpr auto DERAND_STATUS_SUCCESS = derandStatus_t::HIPRAND_STATUS_SUCCESS;
#define derandCreateGenerator hiprandCreateGenerator
#define derandSetPseudoRandomGeneratorSeed hiprandSetPseudoRandomGeneratorSeed
#define derandGenerateUniform hiprandGenerateUniform
#define derandGenerateUniformDouble hiprandGenerateUniformDouble
#define derandDestroyGenerator hiprandDestroyGenerator
using derandRngType = hiprandRngType;
constexpr auto DERAND_RNG_PSEUDO_DEFAULT = derandRngType::HIPRAND_RNG_PSEUDO_DEFAULT;
using deblasSideMode_t = hipblasSideMode_t;
constexpr auto DEBLAS_SIDE_LEFT = deblasSideMode_t::HIPBLAS_SIDE_LEFT;
constexpr auto DEBLAS_SIDE_RIGHT = deblasSideMode_t::HIPBLAS_SIDE_RIGHT;
using deblasFillMode_t = hipblasFillMode_t;
constexpr auto DEBLAS_FILL_MODE_LOWER = deblasFillMode_t::HIPBLAS_FILL_MODE_LOWER;
constexpr auto DEBLAS_FILL_MODE_UPPER = deblasFillMode_t::HIPBLAS_FILL_MODE_UPPER;
using deblasDiagType_t = hipblasDiagType_t;
constexpr auto DEBLAS_DIAG_UNIT = deblasDiagType_t::HIPBLAS_DIAG_UNIT;
constexpr auto DEBLAS_DIAG_NON_UNIT = deblasDiagType_t::HIPBLAS_DIAG_NON_UNIT;
using deblasOperation_t = hipblasOperation_t;
constexpr auto DEBLAS_OP_N = deblasOperation_t::HIPBLAS_OP_N;
constexpr auto DEBLAS_OP_T = deblasOperation_t::HIPBLAS_OP_T;
constexpr auto DEBLAS_OP_C = deblasOperation_t::HIPBLAS_OP_C;
#endif
#ifdef DDLA_USE_CPU
using deblasStatus_t = int;
constexpr auto DEBLAS_STATUS_SUCCESS = 0;
using deblasHandle_t = void*;
using desolverHandle_t = void*;
using desolverStatus_t = int;
constexpr auto DESOLVER_STATUS_SUCCESS = 0;
#define desolverGetStream(solverH, stream) ((void)0)
using derandGenerator_t = void*;
using derandStatus_t = int;
constexpr auto DERAND_STATUS_SUCCESS = 0;
#define derandCreateGenerator(gen, rng) ((void)0)
#define derandSetPseudoRandomGeneratorSeed(gen, seed) ((void)0)
#define derandGenerateUniform(gen, data, n) ((void)0)
#define derandGenerateUniformDouble(gen, data, n) ((void)0)
#define derandDestroyGenerator(gen) ((void)0)
using derandRngType = int;
constexpr auto DERAND_RNG_PSEUDO_DEFAULT = 0;
using deblasSideMode_t = int;
constexpr auto DEBLAS_SIDE_LEFT = 0;
constexpr auto DEBLAS_SIDE_RIGHT = 1;
using deblasFillMode_t = int;
constexpr auto DEBLAS_FILL_MODE_LOWER = 0;
constexpr auto DEBLAS_FILL_MODE_UPPER = 1;
using deblasDiagType_t = int;
constexpr auto DEBLAS_DIAG_UNIT = 0;
constexpr auto DEBLAS_DIAG_NON_UNIT = 1;
using deblasOperation_t = char;
constexpr auto DEBLAS_OP_N = 'N';
constexpr auto DEBLAS_OP_T = 'T';
constexpr auto DEBLAS_OP_C = 'C';
#endif

// ---------------------------------------------------------------------------
// Bare runtime type/constant names, resolved once per TU via RuntimeTraits
// at the local (per-TU) backend. Every existing call site that spells these
// bare names (e.g. `runtimeStream_t stream = handle->stream;`) keeps working
// unchanged -- these are plain aliases, not alias templates, so no `<...>`
// is ever required at the use site.
// ---------------------------------------------------------------------------
using runtimeStream_t = detail::RuntimeTraits<detail::local_backend_v>::stream_t;
using runtimeError_t = detail::RuntimeTraits<detail::local_backend_v>::error_t;
using runtimeEvent_t = detail::RuntimeTraits<detail::local_backend_v>::event_t;
using runtimeDataType_t = detail::RuntimeTraits<detail::local_backend_v>::data_type_t;
using runtimeMemcpyKind = detail::RuntimeTraits<detail::local_backend_v>::memcpy_kind_t;

constexpr auto runtimeSuccess = detail::RuntimeTraits<detail::local_backend_v>::success;
constexpr auto RUNTIME_R_64F = detail::RuntimeTraits<detail::local_backend_v>::r_64f;
constexpr auto RUNTIME_C_64F = detail::RuntimeTraits<detail::local_backend_v>::c_64f;
constexpr auto RUNTIME_R_32F = detail::RuntimeTraits<detail::local_backend_v>::r_32f;
constexpr auto RUNTIME_C_32F = detail::RuntimeTraits<detail::local_backend_v>::c_32f;
constexpr auto runtimeMemcpyHostToDevice = detail::RuntimeTraits<detail::local_backend_v>::host_to_device;
constexpr auto runtimeMemcpyDeviceToHost = detail::RuntimeTraits<detail::local_backend_v>::device_to_host;
constexpr auto runtimeMemcpyDeviceToDevice = detail::RuntimeTraits<detail::local_backend_v>::device_to_device;

// Unconditionally defined (not guarded by #ifdef DDLA_USE_CPU): its body has
// no vendor dependency, and it must stay name-lookup-visible in every TU --
// RuntimeImpl<DdlaBackend::CPU>::memcpy2DAsync below calls it by name. Only
// ever *called* from the CPU implementation; in a GPU-only TU it stays
// declared but unused.
static inline runtimeError_t cpuMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
    size_t width, size_t height, runtimeMemcpyKind kind, runtimeStream_t stream) {
    (void)kind; (void)stream;
    if (width == 0 || height == 0) return runtimeSuccess;
    // Contiguous fast path: when the source and destination row pitches equal
    // the row width, the whole region is one contiguous block -- a single
    // memcpy avoids per-row call overhead (important for stride-1 vectors
    // where height can be very large).
    if (spitch == width && dpitch == width) {
        std::memcpy(dst, src, width * height);
        return runtimeSuccess;
    }
    for (size_t i = 0; i < height; ++i)
        std::memcpy((char*)dst + i * dpitch, (const char*)src + i * spitch, width);
    return runtimeSuccess;
}

// ---------------------------------------------------------------------------
// RuntimeImpl<Backend>: per-backend implementations of the runtime* family
// below. C++11 has no `if constexpr`, so the CPU/GPU branch bodies move into
// these specializations -- a plain runtime `if` would require every branch to
// compile in every TU, which fails because e.g. cudaMalloc is undeclared in a
// CPU-only TU. RuntimeImpl<DdlaBackend::GPU> is only *defined* in a TU that
// actually compiled in a GPU vendor, exactly like RuntimeTraits<Backend>: the
// runtime* wrappers below default Backend to this TU-local constant, so a TU
// compiled without a GPU vendor never names the undefined GPU specialization.
// ---------------------------------------------------------------------------
namespace detail {

template <DdlaBackend Backend> struct RuntimeImpl;   // primary: intentionally undefined

template <>
struct RuntimeImpl<DdlaBackend::CPU> {
    using traits = RuntimeTraits<DdlaBackend::CPU>;
    using error_t = typename traits::error_t;
    using stream_t = typename traits::stream_t;
    using memcpy_kind_t = typename traits::memcpy_kind_t;
    using event_t = typename traits::event_t;

    static error_t malloc(void** ptr, std::size_t bytes)
    {
        if (bytes == 0 && ptr != nullptr) {
            *ptr = nullptr;
            return traits::success;
        }
        *ptr = std::malloc(bytes);
        return (*ptr != nullptr || bytes == 0) ? traits::success : 1;
    }
    static error_t mallocAsync(void** ptr, std::size_t bytes, stream_t stream)
    {
        (void)stream;
        if (bytes == 0 && ptr != nullptr) {
            *ptr = nullptr;
            return traits::success;
        }
        *ptr = std::malloc(bytes);
        return (*ptr != nullptr || bytes == 0) ? traits::success : 1;
    }
    static error_t free(void* ptr)
    {
        if (ptr == nullptr) return traits::success;
        std::free(ptr);
        return traits::success;
    }
    static error_t freeAsync(void* ptr, stream_t stream)
    {
        (void)stream;
        if (ptr == nullptr) return traits::success;
        std::free(ptr);
        return traits::success;
    }
    static error_t streamSynchronize(stream_t stream) { (void)stream; return traits::success; }
    static error_t deviceSynchronize() { return traits::success; }
    static error_t getDeviceCount(int* count) { *count = 1; return traits::success; }
    static const char* getErrorString(error_t status) { (void)status; return ""; }
    static error_t getLastError() { return traits::success; }
    static error_t memcpyAsync(void* dst, const void* src, std::size_t count,
                               memcpy_kind_t kind, stream_t stream)
    {
        (void)kind; (void)stream;
        std::memcpy(dst, src, count);
        return traits::success;
    }
    static error_t memcpy(void* dst, const void* src, std::size_t count, memcpy_kind_t kind)
    {
        (void)kind;
        std::memcpy(dst, src, count);
        return traits::success;
    }
    static error_t memcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                                 size_t width, size_t height, memcpy_kind_t kind, stream_t stream)
    {
        return cpuMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
    }
    static error_t memsetAsync(void* ptr, int value, std::size_t count, stream_t stream)
    {
        (void)stream;
        std::memset(ptr, value, count);
        return traits::success;
    }
    static error_t eventCreate(event_t* event) { (void)event; return traits::success; }
    static error_t eventDestroy(event_t event) { (void)event; return traits::success; }
    static error_t eventRecord(event_t event, stream_t stream) { (void)event; (void)stream; return traits::success; }
    static error_t streamWaitEvent(stream_t stream, event_t event, unsigned int flags)
    {
        (void)stream; (void)event; (void)flags;
        return traits::success;
    }
    static error_t memGetInfo(std::size_t* free, std::size_t* total) { *free = 0; *total = 0; return traits::success; }
};

#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
template <>
struct RuntimeImpl<DdlaBackend::GPU> {
    using traits = RuntimeTraits<DdlaBackend::GPU>;
    using error_t = typename traits::error_t;
    using stream_t = typename traits::stream_t;
    using memcpy_kind_t = typename traits::memcpy_kind_t;
    using event_t = typename traits::event_t;

    static error_t malloc(void** ptr, std::size_t bytes)
    {
        if (bytes == 0 && ptr != nullptr) {
            *ptr = nullptr;
            return traits::success;
        }
#if defined(DDLA_USE_CUDA)
        return cudaMalloc(reinterpret_cast<void**>(ptr), bytes);
#elif defined(DDLA_USE_HIP)
        return hipMalloc(reinterpret_cast<void**>(ptr), bytes);
#endif
    }
    static error_t mallocAsync(void** ptr, std::size_t bytes, stream_t stream)
    {
        if (bytes == 0 && ptr != nullptr) {
            *ptr = nullptr;
            return traits::success;
        }
#if defined(DDLA_USE_CUDA)
        return cudaMallocAsync(reinterpret_cast<void**>(ptr), bytes, stream);
#elif defined(DDLA_USE_HIP)
        return hipMallocAsync(reinterpret_cast<void**>(ptr), bytes, stream);
#endif
    }
    static error_t free(void* ptr)
    {
        if (ptr == nullptr) return traits::success;
#if defined(DDLA_USE_CUDA)
        return cudaFree(ptr);
#elif defined(DDLA_USE_HIP)
        return hipFree(ptr);
#endif
    }
    static error_t freeAsync(void* ptr, stream_t stream)
    {
        if (ptr == nullptr) return traits::success;
#if defined(DDLA_USE_CUDA)
        return cudaFreeAsync(ptr, stream);
#elif defined(DDLA_USE_HIP)
        return hipFreeAsync(ptr, stream);
#endif
    }
    static error_t streamSynchronize(stream_t stream)
    {
#if defined(DDLA_USE_CUDA)
        return cudaStreamSynchronize(stream);
#elif defined(DDLA_USE_HIP)
        return hipStreamSynchronize(stream);
#endif
    }
    static error_t deviceSynchronize()
    {
#if defined(DDLA_USE_CUDA)
        return cudaDeviceSynchronize();
#elif defined(DDLA_USE_HIP)
        return hipDeviceSynchronize();
#endif
    }
    static error_t getDeviceCount(int* count)
    {
#if defined(DDLA_USE_CUDA)
        return cudaGetDeviceCount(count);
#elif defined(DDLA_USE_HIP)
        return hipGetDeviceCount(count);
#endif
    }
    static const char* getErrorString(error_t status)
    {
#if defined(DDLA_USE_CUDA)
        return cudaGetErrorString(status);
#elif defined(DDLA_USE_HIP)
        return hipGetErrorString(status);
#endif
    }
    static error_t getLastError()
    {
#if defined(DDLA_USE_CUDA)
        return cudaGetLastError();
#elif defined(DDLA_USE_HIP)
        return hipGetLastError();
#endif
    }
    static error_t memcpyAsync(void* dst, const void* src, std::size_t count,
                               memcpy_kind_t kind, stream_t stream)
    {
#if defined(DDLA_USE_CUDA)
        return cudaMemcpyAsync(dst, src, count, kind, stream);
#elif defined(DDLA_USE_HIP)
        return hipMemcpyAsync(dst, src, count, kind, stream);
#endif
    }
    static error_t memcpy(void* dst, const void* src, std::size_t count, memcpy_kind_t kind)
    {
#if defined(DDLA_USE_CUDA)
        return cudaMemcpy(dst, src, count, kind);
#elif defined(DDLA_USE_HIP)
        return hipMemcpy(dst, src, count, kind);
#endif
    }
    static error_t memcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                                 size_t width, size_t height, memcpy_kind_t kind, stream_t stream)
    {
#if defined(DDLA_USE_CUDA)
        return cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
#elif defined(DDLA_USE_HIP)
        return hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
#endif
    }
    static error_t memsetAsync(void* ptr, int value, std::size_t count, stream_t stream)
    {
#if defined(DDLA_USE_CUDA)
        return cudaMemsetAsync(ptr, value, count, stream);
#elif defined(DDLA_USE_HIP)
        return hipMemsetAsync(ptr, value, count, stream);
#endif
    }
    static error_t eventCreate(event_t* event)
    {
#if defined(DDLA_USE_CUDA)
        return cudaEventCreate(event);
#elif defined(DDLA_USE_HIP)
        return hipEventCreate(event);
#endif
    }
    static error_t eventDestroy(event_t event)
    {
#if defined(DDLA_USE_CUDA)
        return cudaEventDestroy(event);
#elif defined(DDLA_USE_HIP)
        return hipEventDestroy(event);
#endif
    }
    static error_t eventRecord(event_t event, stream_t stream)
    {
#if defined(DDLA_USE_CUDA)
        return cudaEventRecord(event, stream);
#elif defined(DDLA_USE_HIP)
        return hipEventRecord(event, stream);
#endif
    }
    static error_t streamWaitEvent(stream_t stream, event_t event, unsigned int flags)
    {
#if defined(DDLA_USE_CUDA)
        return cudaStreamWaitEvent(stream, event, flags);
#elif defined(DDLA_USE_HIP)
        return hipStreamWaitEvent(stream, event, flags);
#endif
    }
    static error_t memGetInfo(std::size_t* free, std::size_t* total)
    {
#if defined(DDLA_USE_CUDA)
        return cudaMemGetInfo(free, total);
#elif defined(DDLA_USE_HIP)
        return hipMemGetInfo(free, total);
#endif
    }
};
#endif // defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)

} // namespace detail

// ---------------------------------------------------------------------------
// Runtime functions: real DdlaBackend-templated functions (no #define), each
// defaulted to this TU's local backend so existing bracket-free call sites
// (e.g. `runtimeMallocAsync(&p, n, stream)`) keep compiling unchanged --
// Backend is a non-deduced template parameter here, so a call with no
// explicit <...> always resolves it from the default argument. Each function
// forwards to detail::RuntimeImpl<Backend>, the per-backend implementation
// selected at compile time.
// ---------------------------------------------------------------------------

template <DdlaBackend Backend = detail::local_backend_v>
static inline typename detail::RuntimeTraits<Backend>::error_t
runtimeMalloc(void** ptr, std::size_t bytes)
{
    return detail::RuntimeImpl<Backend>::malloc(ptr, bytes);
}

// `typename = enable_if<!is_void<T>>` excludes T=void from this overload set:
// without it, a call with a plain `void**` argument would deduce T=void here
// and become an exact-match tie against the void**-overload above -- and
// since *both* are now function templates (not one template + one plain
// function), the old "prefer the non-template" tiebreaker no longer applies,
// making the call ambiguous. Restricting T to non-void keeps the two
// overloads' argument sets disjoint.
template <DdlaBackend Backend = detail::local_backend_v, typename T,
          typename = typename std::enable_if<!std::is_void<T>::value>::type>
static inline typename detail::RuntimeTraits<Backend>::error_t
runtimeMalloc(T** ptr, std::size_t bytes)
{
    return runtimeMalloc<Backend>(reinterpret_cast<void**>(ptr), bytes);
}

template <DdlaBackend Backend = detail::local_backend_v>
static inline typename detail::RuntimeTraits<Backend>::error_t
runtimeMallocAsync(void** ptr, std::size_t bytes, typename detail::RuntimeTraits<Backend>::stream_t stream)
{
    return detail::RuntimeImpl<Backend>::mallocAsync(ptr, bytes, stream);
}

// Same T=void exclusion as runtimeMalloc's T** overload above, and for the
// same reason (both overloads here are now function templates).
template <DdlaBackend Backend = detail::local_backend_v, typename T,
          typename = typename std::enable_if<!std::is_void<T>::value>::type>
static inline typename detail::RuntimeTraits<Backend>::error_t
runtimeMallocAsync(T** ptr, std::size_t bytes, typename detail::RuntimeTraits<Backend>::stream_t stream)
{
    return runtimeMallocAsync<Backend>(reinterpret_cast<void**>(ptr), bytes, stream);
}

template <DdlaBackend Backend = detail::local_backend_v>
static inline typename detail::RuntimeTraits<Backend>::error_t
runtimeFree(void* ptr)
{
    return detail::RuntimeImpl<Backend>::free(ptr);
}

template <DdlaBackend Backend = detail::local_backend_v>
static inline typename detail::RuntimeTraits<Backend>::error_t
runtimeFreeAsync(void* ptr, typename detail::RuntimeTraits<Backend>::stream_t stream)
{
    return detail::RuntimeImpl<Backend>::freeAsync(ptr, stream);
}

template <DdlaBackend Backend = detail::local_backend_v>
static inline typename detail::RuntimeTraits<Backend>::error_t
runtimeStreamSynchronize(typename detail::RuntimeTraits<Backend>::stream_t stream) {
    return detail::RuntimeImpl<Backend>::streamSynchronize(stream);
}

template <DdlaBackend Backend = detail::local_backend_v>
static inline typename detail::RuntimeTraits<Backend>::error_t
runtimeDeviceSynchronize(){
    return detail::RuntimeImpl<Backend>::deviceSynchronize();
}

template <DdlaBackend Backend = detail::local_backend_v>
static inline typename detail::RuntimeTraits<Backend>::error_t
runtimeGetDeviceCount(int* count){
    return detail::RuntimeImpl<Backend>::getDeviceCount(count);
}

template <DdlaBackend Backend = detail::local_backend_v>
static inline const char*
runtimeGetErrorString(typename detail::RuntimeTraits<Backend>::error_t status)
{
    return detail::RuntimeImpl<Backend>::getErrorString(status);
}

template <DdlaBackend Backend = detail::local_backend_v>
static inline typename detail::RuntimeTraits<Backend>::error_t
runtimeGetLastError()
{
    return detail::RuntimeImpl<Backend>::getLastError();
}

template <DdlaBackend Backend = detail::local_backend_v>
static inline typename detail::RuntimeTraits<Backend>::error_t
runtimeMemcpyAsync(void* dst, const void* src, std::size_t count,
                   typename detail::RuntimeTraits<Backend>::memcpy_kind_t kind,
                   typename detail::RuntimeTraits<Backend>::stream_t stream)
{
    return detail::RuntimeImpl<Backend>::memcpyAsync(dst, src, count, kind, stream);
}

template <DdlaBackend Backend = detail::local_backend_v>
static inline typename detail::RuntimeTraits<Backend>::error_t
runtimeMemcpy(void* dst, const void* src, std::size_t count,
              typename detail::RuntimeTraits<Backend>::memcpy_kind_t kind)
{
    return detail::RuntimeImpl<Backend>::memcpy(dst, src, count, kind);
}

template <DdlaBackend Backend = detail::local_backend_v>
static inline typename detail::RuntimeTraits<Backend>::error_t
runtimeMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                      size_t width, size_t height,
                      typename detail::RuntimeTraits<Backend>::memcpy_kind_t kind,
                      typename detail::RuntimeTraits<Backend>::stream_t stream)
{
    return detail::RuntimeImpl<Backend>::memcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
}

template <DdlaBackend Backend = detail::local_backend_v>
static inline typename detail::RuntimeTraits<Backend>::error_t
runtimeMemsetAsync(void* ptr, int value, std::size_t count,
                    typename detail::RuntimeTraits<Backend>::stream_t stream)
{
    return detail::RuntimeImpl<Backend>::memsetAsync(ptr, value, count, stream);
}

template <DdlaBackend Backend = detail::local_backend_v>
static inline typename detail::RuntimeTraits<Backend>::error_t
runtimeEventCreate(typename detail::RuntimeTraits<Backend>::event_t* event)
{
    return detail::RuntimeImpl<Backend>::eventCreate(event);
}

template <DdlaBackend Backend = detail::local_backend_v>
static inline typename detail::RuntimeTraits<Backend>::error_t
runtimeEventDestroy(typename detail::RuntimeTraits<Backend>::event_t event)
{
    return detail::RuntimeImpl<Backend>::eventDestroy(event);
}

template <DdlaBackend Backend = detail::local_backend_v>
static inline typename detail::RuntimeTraits<Backend>::error_t
runtimeEventRecord(typename detail::RuntimeTraits<Backend>::event_t event,
                    typename detail::RuntimeTraits<Backend>::stream_t stream)
{
    return detail::RuntimeImpl<Backend>::eventRecord(event, stream);
}

template <DdlaBackend Backend = detail::local_backend_v>
static inline typename detail::RuntimeTraits<Backend>::error_t
runtimeStreamWaitEvent(typename detail::RuntimeTraits<Backend>::stream_t stream,
                        typename detail::RuntimeTraits<Backend>::event_t event,
                        unsigned int flags)
{
    return detail::RuntimeImpl<Backend>::streamWaitEvent(stream, event, flags);
}

template <DdlaBackend Backend = detail::local_backend_v>
static inline typename detail::RuntimeTraits<Backend>::error_t
runtimeMemGetInfo(std::size_t* free, std::size_t* total)
{
    return detail::RuntimeImpl<Backend>::memGetInfo(free, total);
}


static inline void MPI_CHECK(int status, const char* file = __builtin_FILE(), int line = __builtin_LINE())
{
    if (status != MPI_SUCCESS)
    {
        fprintf(stderr, "mpi error at %s:%d : %d\n", file, line, status);
        int mpi_initialized = 0;
        int mpi_finalized = 0;
        MPI_Initialized(&mpi_initialized);
        MPI_Finalized(&mpi_finalized);
        // Abort the whole MPI job when possible so the other ranks do not
        // hang waiting in collectives.  MPI_Abort is only valid between
        // MPI_Init and MPI_Finalize; after finalization fall through to the
        // throw below.
        if (mpi_initialized && !mpi_finalized) {
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        throw std::runtime_error("ddla: MPI error at " + std::string(file) +
                                 ":" + std::to_string(line));
    }
}

// DdlaBackend-templated, mirroring the runtime* family above: the status
// parameter's type comes from RuntimeTraits<Backend>, which is a non-deduced
// context, so every existing bracket-free `RUNTIME_CHECK(expr)` still resolves
// Backend from the default and behaves exactly as before. Explicitly writing
// `RUNTIME_CHECK<DdlaBackend::CPU>(...)` is now possible too, which matters in
// a dual build where RuntimeTraits<CPU>::error_t (int) and
// RuntimeTraits<GPU>::error_t (cudaError_t/hipError_t) are different types and
// the untemplated version only ever accepted the GPU one.
// Error-propagation model for the CHECK family below: on failure these throw
// std::runtime_error (previously they called exit(EXIT_FAILURE)).  An
// uncaught exception still terminates the rank, so uncaught it behaves like
// the old exit(); but if a caller catches the exception on one rank only,
// the remaining ranks can hang in MPI/NCCL/RCCL collectives.  Callers that
// catch must therefore abort the whole job themselves (e.g. MPI_Abort) when
// the error could leave peers blocked in a collective.  MPI_CHECK already
// aborts the job directly (see above); the other CHECK macros cannot, since
// they have no communicator available.
template <DdlaBackend Backend = detail::local_backend_v>
static inline void RUNTIME_CHECK(typename detail::RuntimeTraits<Backend>::error_t status,
                                 const char* file = __builtin_FILE(), int line = __builtin_LINE())
{
    if (status != detail::RuntimeTraits<Backend>::success)
    {
        fprintf(stderr, "runtime error at %s:%d : %s\n", file, line,
                runtimeGetErrorString<Backend>(status));
        throw std::runtime_error("ddla: runtime error at " + std::string(file) +
                                 ":" + std::to_string(line) + " : " +
                                 runtimeGetErrorString<Backend>(status));
    }
}

static inline void BLAS_CHECK(deblasStatus_t err_, const char* file = __builtin_FILE(), int line = __builtin_LINE())
{
    if (err_ != DEBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "deblas error %d at %s:%d\n", err_, file, line);
        throw std::runtime_error("ddla: deblas error " + std::to_string(err_) +
                                 " at " + std::string(file) + ":" + std::to_string(line));
    }
}

static inline void SOLVER_CHECK(desolverStatus_t err_, const char* file = __builtin_FILE(), int line = __builtin_LINE())
{
    if (err_ != DESOLVER_STATUS_SUCCESS)
    {
        fprintf(stderr, "cusolver error %d at %s:%d\n", err_, file, line);
        throw std::runtime_error("ddla: solver error " + std::to_string(err_) +
                                 " at " + std::string(file) + ":" + std::to_string(line));
    }
}

#ifdef DDLA_USE_CCL
static inline void CCL_CHECK(ncclResult_t status, const char* file = __builtin_FILE(), int line = __builtin_LINE())
{
    if (status != ncclSuccess)
    {
        fprintf(stderr, "nccl error at %s:%d : %d\n", file, line, status);
        throw std::runtime_error("ddla: nccl error " + std::to_string(status) +
                                 " at " + std::string(file) + ":" + std::to_string(line));
    }
}
#else
static inline void CCL_CHECK(int status, const char* file = __builtin_FILE(), int line = __builtin_LINE())
{
    MPI_CHECK(status, file, line);
}
#endif

static inline void DERAND_CHECK(derandStatus_t status, const char* file = __builtin_FILE(), int line = __builtin_LINE())
{
    if (status != DERAND_STATUS_SUCCESS)
    {
        fprintf(stderr, "derand error at %s:%d : %d\n", file, line, status);
        throw std::runtime_error("ddla: derand error " + std::to_string(status) +
                                 " at " + std::string(file) + ":" + std::to_string(line));
    }
}


} // namespace ddla

#endif // DDLA_CONNECTOR_H
