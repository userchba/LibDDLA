// ---------------------------------------------------------------------------
// Backend-templated Cholesky factorization of a Hermitian positive-definite matrix (LAPACK ?potrf):
//     A = U^H * U  (uplo = 'U')        or        A = L * L^H  (uplo = 'L').
//
// Structure follows src/getrf.cpp: one definition covering CPU (Fortran LAPACK ?potrf on host pointers) and GPU
// (cuSOLVER / hipSOLVER potrf on device pointers, enqueued on the handle's stream), selected with `if constexpr`
// on the Backend template parameter, the arms guarded by DDLA_HAS_CPU / DDLA_HAS_GPU.  The vendor calls are private
// to this translation unit; the public interface is the templated ddla::potrf declared in <ddla/ddla.h>.
//
// Three components (plan_util.h): potrf_check; potrf_workspace_size (backend bytes of the vendor workspace);
// potrf_run on the caller's scratch.
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

#include "blas_enums.h"
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "lapack_impl.h"

#if DDLA_HAS_CPU
// Host LAPACK ?potrf (Fortran ABI; the CPU BLAS library must provide LAPACK -- OpenBLAS and MKL do).
extern "C" {
void spotrf_(const char*, const int*, float*, const int*, int*);
void dpotrf_(const char*, const int*, double*, const int*, int*);
void cpotrf_(const char*, const int*, std::complex<float>*, const int*, int*);
void zpotrf_(const char*, const int*, std::complex<double>*, const int*, int*);
}
#endif

namespace ddla {

namespace {

inline const char* potrf_backend_name(DdlaBackend backend)
{
    return backend == DdlaBackend::CPU ? "CPU" : "GPU";
}

#if DDLA_HAS_CPU
inline void cpu_potrf(char uplo, int n, float* A, int lda, int* info)
{
    spotrf_(&uplo, &n, A, &lda, info);
}

inline void cpu_potrf(char uplo, int n, double* A, int lda, int* info)
{
    dpotrf_(&uplo, &n, A, &lda, info);
}

inline void cpu_potrf(char uplo, int n, std::complex<float>* A, int lda, int* info)
{
    cpotrf_(&uplo, &n, A, &lda, info);
}

inline void cpu_potrf(char uplo, int n, std::complex<double>* A, int lda, int* info)
{
    zpotrf_(&uplo, &n, A, &lda, info);
}
#endif // DDLA_HAS_CPU

#if DDLA_HAS_GPU
#if DDLA_GPU_VENDOR

// Per scalar type: the buffer-size query and the factorization itself.
inline gpusolverStatus_t potrf_buffer_size(gpusolverHandle_t h, gpublasFillMode_t uplo, int n, float* A, int lda, int* lwork)
{
    return gpusolverSpotrf_bufferSize(h, gpusolverFillMode(uplo), n, A, lda, lwork);
}
inline gpusolverStatus_t potrf_buffer_size(gpusolverHandle_t h, gpublasFillMode_t uplo, int n, double* A, int lda, int* lwork)
{
    return gpusolverDpotrf_bufferSize(h, gpusolverFillMode(uplo), n, A, lda, lwork);
}
inline gpusolverStatus_t potrf_buffer_size(gpusolverHandle_t h, gpublasFillMode_t uplo, int n, std::complex<float>* A,
                                           int lda, int* lwork)
{
    return gpusolverCpotrf_bufferSize(h, gpusolverFillMode(uplo), n, (gpusolverComplex*)A, lda, lwork);
}
inline gpusolverStatus_t potrf_buffer_size(gpusolverHandle_t h, gpublasFillMode_t uplo, int n, std::complex<double>* A,
                                           int lda, int* lwork)
{
    return gpusolverZpotrf_bufferSize(h, gpusolverFillMode(uplo), n, (gpusolverDoubleComplex*)A, lda, lwork);
}

inline gpusolverStatus_t potrf_call(gpusolverHandle_t h, gpublasFillMode_t uplo, int n, float* A, int lda, float* work,
                                    int lwork, int* info)
{
    return gpusolverSpotrf(h, gpusolverFillMode(uplo), n, A, lda, work, lwork, info);
}
inline gpusolverStatus_t potrf_call(gpusolverHandle_t h, gpublasFillMode_t uplo, int n, double* A, int lda, double* work,
                                    int lwork, int* info)
{
    return gpusolverDpotrf(h, gpusolverFillMode(uplo), n, A, lda, work, lwork, info);
}
inline gpusolverStatus_t potrf_call(gpusolverHandle_t h, gpublasFillMode_t uplo, int n, std::complex<float>* A, int lda,
                                    std::complex<float>* work, int lwork, int* info)
{
    return gpusolverCpotrf(h, gpusolverFillMode(uplo), n, (gpusolverComplex*)A, lda, (gpusolverComplex*)work, lwork, info);
}
inline gpusolverStatus_t potrf_call(gpusolverHandle_t h, gpublasFillMode_t uplo, int n, std::complex<double>* A, int lda,
                                    std::complex<double>* work, int lwork, int* info)
{
    return gpusolverZpotrf(h, gpusolverFillMode(uplo), n, (gpusolverDoubleComplex*)A, lda,
                           (gpusolverDoubleComplex*)work, lwork, info);
}

// Factor on the caller's backend scratch; lwork is what the scratch holds.
template <typename T>
gpusolverStatus_t gpu_potrf(const DdlaHandle_t& handle, void* work, std::size_t work_bytes, gpublasFillMode_t uplo,
                            int n, T* A, int lda, int* info)
{
    const int lwork = static_cast<int>(std::min<std::size_t>(work_bytes / sizeof(T), std::numeric_limits<int>::max()));
    return potrf_call(static_cast<gpusolverHandle_t>(handle->solverH), uplo, n, A, lda, static_cast<T*>(work), lwork,
                      info);
}

#else // DDLA_GPU_VENDOR

template <typename T>
gpusolverStatus_t gpu_potrf(const DdlaHandle_t&, void*, std::size_t, gpublasFillMode_t, int, T*, int, int*)
{
    throw std::runtime_error("potrf: no GPU vendor selected at compile time");
}

#endif // DDLA_GPU_VENDOR
#endif // DDLA_HAS_GPU

} // anonymous namespace

namespace lapack_impl {

// Component 1.
template <DdlaBackend Backend>
char potrf_check(const DdlaHandle_t& handle, char uplo, int n, const void* A, int lda, const int* info)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU potrf is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU potrf is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("potrf: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("potrf: template backend ") + potrf_backend_name(Backend) +
            " does not match handle backend " + potrf_backend_name(actual));
    }
    // 'U' / 'L' in either case; anything else throws instead of silently picking a triangle.
    const gpublasFillMode_t fill = detail::to_fill_mode(uplo);
    if (n < 0) throw std::runtime_error("potrf: negative matrix size");
    if (info == nullptr) throw std::runtime_error("potrf: null info");

    const char u = fill == GPUBLAS_FILL_MODE_UPPER ? 'U' : 'L';
    if (n == 0) return u;   // nothing to factor: the pointers are not read
    if (A == nullptr) throw std::runtime_error("potrf: null A");
    if (lda < std::max(1, n)) throw std::runtime_error("potrf: lda too small");
    return u;
}

// Component 2: the vendor scratch of an order-n factorization (uplo normalized).  CPU handles (host LAPACK) need
// none.
template <typename T>
void potrf_workspace_size(const DdlaHandle_t& handle, char uplo, int n, T* A, int lda, std::size_t* backend_bytes)
{
    *backend_bytes = 0;
#if DDLA_HAS_GPU && DDLA_GPU_VENDOR
    if (handle->backend == DdlaBackend::GPU && n > 0) {
        const gpublasFillMode_t fill = uplo == 'U' ? GPUBLAS_FILL_MODE_UPPER : GPUBLAS_FILL_MODE_LOWER;
        int lwork = 0;
        SOLVER_CHECK(potrf_buffer_size(static_cast<gpusolverHandle_t>(handle->solverH), fill, n, A, lda, &lwork));
        *backend_bytes = static_cast<std::size_t>(lwork) * sizeof(T);
    }
#else
    (void)handle; (void)uplo; (void)n; (void)A; (void)lda;
#endif
}

// Component 3.
template <DdlaBackend Backend, typename T>
void potrf_run(const DdlaHandle_t& handle, char uplo, int n, T* A, int lda, int* info,
               [[maybe_unused]] void* backend_work, [[maybe_unused]] std::size_t backend_bytes)
{
    if (n <= 0) {
        // Nothing to factor: success, reported where the caller will look for it.
        if constexpr (Backend == DdlaBackend::CPU) {
            *info = 0;
        } else {
            DDLA_CHECK(ddlaMemset(handle, info, 0, sizeof(int)));
        }
        return;
    }

    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        cpu_potrf(uplo, n, A, lda, info);
#endif
    } else {
#if DDLA_HAS_GPU
        SOLVER_CHECK(gpu_potrf(handle, backend_work, backend_bytes,
                               uplo == 'U' ? GPUBLAS_FILL_MODE_UPPER : GPUBLAS_FILL_MODE_LOWER, n, A, lda, info));
#endif
    }
}

} // namespace lapack_impl

template <DdlaBackend Backend, typename T>
void potrf(const DdlaHandle_t& handle, char uplo, int n, T* A, int lda, int* info)
{
    // 1. parameter check (also normalizes uplo).
    const char u = lapack_impl::potrf_check<Backend>(handle, uplo, n, A, lda, info);
    // 2. the vendor scratch of this order (an empty problem needs none).
    std::size_t backend_bytes = 0;
    lapack_impl::potrf_workspace_size<T>(handle, u, n, A, lda, &backend_bytes);
    vector<char> work(handle, backend_bytes);
    // 3. the factorization (an empty problem only reports info = 0).
    lapack_impl::potrf_run<Backend, T>(handle, u, n, A, lda, info, work.data(), backend_bytes);
}

#define INSTANTIATE_POTRF(BACKEND, TYPE)                                     \
    template void potrf<BACKEND, TYPE>(                                      \
        const DdlaHandle_t&, char, int, TYPE*, int, int*);                   \
    template void lapack_impl::potrf_run<BACKEND, TYPE>(                     \
        const DdlaHandle_t&, char, int, TYPE*, int, int*, void*, std::size_t)
#define INSTANTIATE_POTRF_SIZE(TYPE)                                         \
    template void lapack_impl::potrf_workspace_size<TYPE>(                   \
        const DdlaHandle_t&, char, int, TYPE*, int, std::size_t*)

#if DDLA_HAS_CPU
template char lapack_impl::potrf_check<DdlaBackend::CPU>(
    const DdlaHandle_t&, char, int, const void*, int, const int*);
INSTANTIATE_POTRF(DdlaBackend::CPU, float);
INSTANTIATE_POTRF(DdlaBackend::CPU, double);
INSTANTIATE_POTRF(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_POTRF(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
template char lapack_impl::potrf_check<DdlaBackend::GPU>(
    const DdlaHandle_t&, char, int, const void*, int, const int*);
INSTANTIATE_POTRF(DdlaBackend::GPU, float);
INSTANTIATE_POTRF(DdlaBackend::GPU, double);
INSTANTIATE_POTRF(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_POTRF(DdlaBackend::GPU, std::complex<double>);
#endif

INSTANTIATE_POTRF_SIZE(float);
INSTANTIATE_POTRF_SIZE(double);
INSTANTIATE_POTRF_SIZE(std::complex<float>);
INSTANTIATE_POTRF_SIZE(std::complex<double>);

#undef INSTANTIATE_POTRF_SIZE
#undef INSTANTIATE_POTRF

} // namespace ddla
