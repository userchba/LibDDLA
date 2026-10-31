// ---------------------------------------------------------------------------
// Backend-templated Cholesky factorization of a Hermitian positive-definite matrix (LAPACK ?potrf):
//     A = U^H * U  (uplo = 'U')        or        A = L * L^H  (uplo = 'L').
//
// Structure follows src/getrf.cpp: one definition covering CPU (Fortran LAPACK ?potrf on host pointers) and GPU
// (cuSOLVER / hipSOLVER potrf on device pointers, enqueued on the handle's stream), selected with `if constexpr`
// on the Backend template parameter, the arms guarded by DDLA_HAS_CPU / DDLA_HAS_GPU.  The vendor calls are private
// to this translation unit; the public interface is the templated ddla::potrf declared in <ddla/ddla.h>.
//
// Three components (src/lapack_impl.h): potrf_check validates, potrf_plan sizes and allocates the vendor scratch,
// potrf_run factorizes; the public potrf is check -> plan -> run.
//
// GPU arm: scratch sized with ?potrf_bufferSize by potrf_plan (the maximum over the planned orders), allocated
// stream-ordered on the handle's stream (the stream the solver is bound to); potrf_run re-queries the size of its
// own order, so the call stays asynchronous; `info` is valid once the caller synchronizes the stream.
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>

#include "blas_enums.h"
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "lapack_impl.h"
#include "plan_util.h"

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

// The factorization on the device: query the exact lwork of this order (a host call), check that the planned scratch
// holds it, run.  The scratch lives in the workspace, on the stream the solver runs on.
template <typename T>
gpusolverStatus_t gpu_potrf(const DdlaHandle_t& handle, vector<T>& work, gpublasFillMode_t uplo, int n,
                            T* A, int lda, int* info)
{
    int lwork = 0;
    SOLVER_CHECK(potrf_buffer_size(static_cast<gpusolverHandle_t>(handle->solverH), uplo, n, A, lda, &lwork));
    assert(static_cast<std::size_t>(lwork) <= work.capacity());
    return potrf_call(static_cast<gpusolverHandle_t>(handle->solverH), uplo, n, A, lda, lwork > 0 ? work.data() : nullptr, lwork, info);
}

#else // DDLA_GPU_VENDOR

template <typename T>
gpusolverStatus_t gpu_potrf(const DdlaHandle_t&, vector<T>&, gpublasFillMode_t, int, T*, int, int*)
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

// Component 2: the vendor scratch.  CPU handles (host LAPACK) need none.
template <typename T>
vector<T> potrf_plan(const DdlaHandle_t& handle, char uplo, const int* orders, std::size_t count, T* A, int lda)
{
    vector<T> work(handle);
#if DDLA_HAS_GPU && DDLA_GPU_VENDOR
    if (ddlaGetBackend(handle) == DdlaBackend::GPU) {
        const gpublasFillMode_t fill = detail::to_fill_mode(uplo);
        int lwork = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const int n = orders[i];
            if (n <= 0) continue;
            int need = 0;
            SOLVER_CHECK(potrf_buffer_size(static_cast<gpusolverHandle_t>(handle->solverH), fill, n, A, lda, &need));
            lwork = std::max(lwork, need);
        }
        work.resize(static_cast<std::size_t>(lwork));
    }
#else
    (void)uplo; (void)orders; (void)count; (void)A; (void)lda;
#endif
    return work;
}

// Component 3.
template <DdlaBackend Backend, typename T>
void potrf_run(const DdlaHandle_t& handle, [[maybe_unused]] vector<T>& work, char uplo, int n, T* A, int lda,
               int* info)
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
        SOLVER_CHECK(gpu_potrf(handle, work, detail::to_fill_mode(uplo), n, A, lda, info));
#endif
    }
}

} // namespace lapack_impl

template <DdlaBackend Backend, typename T>
void potrf(const DdlaHandle_t& handle, char uplo, int n, T* A, int lda, int* info)
{
    // 1. parameter check (also normalizes uplo).
    const char u = lapack_impl::potrf_check<Backend>(handle, uplo, n, A, lda, info);
    // 2. the vendor scratch of this one order (an empty problem plans nothing).
    vector<T> work = lapack_impl::potrf_plan<T>(handle, u, {n}, A, lda);
    // 3. the factorization (an empty problem only reports info = 0).
    lapack_impl::potrf_run<Backend, T>(handle, work, u, n, A, lda, info);
}

#define INSTANTIATE_POTRF(BACKEND, TYPE)                                     \
    template void potrf<BACKEND, TYPE>(                                      \
        const DdlaHandle_t&, char, int, TYPE*, int, int*);                   \
    template void lapack_impl::potrf_run<BACKEND, TYPE>(                     \
        const DdlaHandle_t&, vector<TYPE>&, char, int, TYPE*, int, int*)
#define INSTANTIATE_POTRF_PLAN(TYPE)                                         \
    template vector<TYPE> lapack_impl::potrf_plan<TYPE>(      \
        const DdlaHandle_t&, char, const int*, std::size_t, TYPE*, int)

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

INSTANTIATE_POTRF_PLAN(float);
INSTANTIATE_POTRF_PLAN(double);
INSTANTIATE_POTRF_PLAN(std::complex<float>);
INSTANTIATE_POTRF_PLAN(std::complex<double>);

#undef INSTANTIATE_POTRF_PLAN
#undef INSTANTIATE_POTRF

} // namespace ddla
