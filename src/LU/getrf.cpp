// ---------------------------------------------------------------------------
// Backend-templated LU factorization with partial pivoting (LAPACK ?getrf):  A = P * L * U.
//
// Dispatch follows the Backend template parameter. CPU code is always compiled;
// DDLA_HAS_GPU enables GPU code, and DDLA_USE_CUDA / DDLA_USE_HIP select its vendor.
//
// Three components (plan_util.h): getrf_check; getrf_workspace_size (backend bytes of the vendor workspace);
// getrf_run on the caller's scratch.
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

#include "ddla_connector.h"
#include "ddla_stream.h"
#include "lapack_impl.h"
#include "instantiate.h"
#include "require_backend.h"

// Host LAPACK ?getrf (Fortran ABI; the CPU BLAS library must provide LAPACK -- OpenBLAS and MKL do).
extern "C" {
void sgetrf_(const int*, const int*, float*, const int*, int*, int*);
void dgetrf_(const int*, const int*, double*, const int*, int*, int*);
void cgetrf_(const int*, const int*, std::complex<float>*, const int*, int*, int*);
void zgetrf_(const int*, const int*, std::complex<double>*, const int*, int*, int*);
}

namespace ddla {

namespace {

inline void cpu_getrf(int m, int n, float* A, int lda, int* ipiv, int* info)
{
    sgetrf_(&m, &n, A, &lda, ipiv, info);
}

inline void cpu_getrf(int m, int n, double* A, int lda, int* ipiv, int* info)
{
    dgetrf_(&m, &n, A, &lda, ipiv, info);
}

inline void cpu_getrf(int m, int n, std::complex<float>* A, int lda, int* ipiv, int* info)
{
    cgetrf_(&m, &n, A, &lda, ipiv, info);
}

inline void cpu_getrf(int m, int n, std::complex<double>* A, int lda, int* ipiv, int* info)
{
    zgetrf_(&m, &n, A, &lda, ipiv, info);
}

#if DDLA_HAS_GPU

// Per scalar type: the buffer-size query and the factorization itself.
inline gpusolverStatus_t getrf_buffer_size(gpusolverHandle_t h, int m, int n, float* A, int lda, int* lwork)
{
    return gpusolverSgetrf_bufferSize(h, m, n, A, lda, lwork);
}
inline gpusolverStatus_t getrf_buffer_size(gpusolverHandle_t h, int m, int n, double* A, int lda, int* lwork)
{
    return gpusolverDgetrf_bufferSize(h, m, n, A, lda, lwork);
}
inline gpusolverStatus_t getrf_buffer_size(gpusolverHandle_t h, int m, int n, std::complex<float>* A, int lda, int* lwork)
{
    return gpusolverCgetrf_bufferSize(h, m, n, (gpusolverComplex*)A, lda, lwork);
}
inline gpusolverStatus_t getrf_buffer_size(gpusolverHandle_t h, int m, int n, std::complex<double>* A, int lda, int* lwork)
{
    return gpusolverZgetrf_bufferSize(h, m, n, (gpusolverDoubleComplex*)A, lda, lwork);
}

inline gpusolverStatus_t getrf_call(gpusolverHandle_t h, int m, int n, float* A, int lda, float* work, int lwork,
                                    int* ipiv, int* info)
{
    return gpusolverSgetrf(h, m, n, A, lda, work, lwork, ipiv, info);
}
inline gpusolverStatus_t getrf_call(gpusolverHandle_t h, int m, int n, double* A, int lda, double* work, int lwork,
                                    int* ipiv, int* info)
{
    return gpusolverDgetrf(h, m, n, A, lda, work, lwork, ipiv, info);
}
inline gpusolverStatus_t getrf_call(gpusolverHandle_t h, int m, int n, std::complex<float>* A, int lda,
                                    std::complex<float>* work, int lwork, int* ipiv, int* info)
{
    return gpusolverCgetrf(h, m, n, (gpusolverComplex*)A, lda, (gpusolverComplex*)work, lwork, ipiv, info);
}
inline gpusolverStatus_t getrf_call(gpusolverHandle_t h, int m, int n, std::complex<double>* A, int lda,
                                    std::complex<double>* work, int lwork, int* ipiv, int* info)
{
    return gpusolverZgetrf(h, m, n, (gpusolverDoubleComplex*)A, lda, (gpusolverDoubleComplex*)work, lwork, ipiv, info);
}

// Factor on the caller's backend scratch; lwork is what the scratch holds.
template <typename T>
gpusolverStatus_t gpu_getrf(const DdlaHandle_t& handle, void* work, std::size_t work_bytes, int m, int n, T* A,
                            int lda, int* ipiv, int* info)
{
    const int lwork = static_cast<int>(std::min<std::size_t>(work_bytes / sizeof(T), std::numeric_limits<int>::max()));
    return getrf_call(static_cast<gpusolverHandle_t>(handle->solverH), m, n, A, lda, static_cast<T*>(work), lwork,
                      ipiv, info);
}

#endif // DDLA_HAS_GPU

} // anonymous namespace

namespace lapack_impl {

// Component 1.
template <DdlaBackend Backend>
void getrf_check(const DdlaHandle_t& handle, int m, int n, const void* A, int lda,
                 const int* ipiv, const int* info)
{
    detail::require_backend<Backend>(handle, "getrf");
    if (m < 0 || n < 0) throw std::runtime_error("getrf: negative matrix size");
    if (info == nullptr) throw std::runtime_error("getrf: null info");

    if (m == 0 || n == 0) return;   // nothing to factor: the pointers are not read
    if (A == nullptr) throw std::runtime_error("getrf: null A");
    if (ipiv == nullptr) throw std::runtime_error("getrf: null ipiv");
    if (lda < std::max(1, m)) throw std::runtime_error("getrf: lda too small");
}

// Component 2: the vendor workspace of an m x n factorization.  CPU handles (host LAPACK) need none.
template <typename T>
void getrf_workspace_size(const DdlaHandle_t& handle, int m, int n, T* A, int lda, std::size_t* backend_bytes)
{
    *backend_bytes = 0;
#if DDLA_HAS_GPU
    if (handle->backend == DdlaBackend::GPU && m > 0 && n > 0) {
        int lwork = 0;
        SOLVER_CHECK(getrf_buffer_size(static_cast<gpusolverHandle_t>(handle->solverH), m, n, A, lda, &lwork));
        *backend_bytes = static_cast<std::size_t>(lwork) * sizeof(T);
    }
#else
    (void)handle; (void)m; (void)n; (void)A; (void)lda;
#endif
}

// Component 3.
template <DdlaBackend Backend, typename T>
void getrf_run(const DdlaHandle_t& handle, int m, int n, T* A, int lda, int* ipiv, int* info,
               [[maybe_unused]] void* backend_work, [[maybe_unused]] std::size_t backend_bytes)
{
    if (m <= 0 || n <= 0) {
        // Nothing to factor: success, reported where the caller will look for it.
        if constexpr (Backend == DdlaBackend::CPU) {
            *info = 0;
        } else {
            DDLA_CHECK(ddlaMemset(handle, info, 0, sizeof(int)));
        }
        return;
    }

    if constexpr (Backend == DdlaBackend::CPU) {
        cpu_getrf(m, n, A, lda, ipiv, info);
    } else {
#if DDLA_HAS_GPU
        SOLVER_CHECK(gpu_getrf(handle, backend_work, backend_bytes, m, n, A, lda, ipiv, info));
#endif
    }
}

} // namespace lapack_impl

template <DdlaBackend Backend, typename T>
void getrf(const DdlaHandle_t& handle, int m, int n, T* A, int lda, int* ipiv, int* info)
{
    // 1. parameter check.
    lapack_impl::getrf_check<Backend>(handle, m, n, A, lda, ipiv, info);
    // 2. the vendor workspace of this shape (an empty shape needs none).
    std::size_t backend_bytes = 0;
    lapack_impl::getrf_workspace_size<T>(handle, m, n, A, lda, &backend_bytes);
    vector<char> work(handle, backend_bytes);
    // 3. the factorization (an empty problem only reports info = 0).
    lapack_impl::getrf_run<Backend, T>(handle, m, n, A, lda, ipiv, info, work.data(), backend_bytes);
}

#define INSTANTIATE_GETRF(BACKEND, TYPE)                                     \
    template void getrf<BACKEND, TYPE>(                                      \
        const DdlaHandle_t&, int, int, TYPE*, int, int*, int*);              \
    template void lapack_impl::getrf_run<BACKEND, TYPE>(                     \
        const DdlaHandle_t&, int, int, TYPE*, int, int*, int*, void*, std::size_t)
#define INSTANTIATE_GETRF_SIZE(TYPE)                                         \
    template void lapack_impl::getrf_workspace_size<TYPE>(                   \
        const DdlaHandle_t&, int, int, TYPE*, int, std::size_t*)

template void lapack_impl::getrf_check<DdlaBackend::CPU>(
    const DdlaHandle_t&, int, int, const void*, int, const int*, const int*);
INSTANTIATE_GETRF(DdlaBackend::CPU, float);
INSTANTIATE_GETRF(DdlaBackend::CPU, double);
INSTANTIATE_GETRF(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_GETRF(DdlaBackend::CPU, std::complex<double>);

#if DDLA_HAS_GPU
template void lapack_impl::getrf_check<DdlaBackend::GPU>(
    const DdlaHandle_t&, int, int, const void*, int, const int*, const int*);
INSTANTIATE_GETRF(DdlaBackend::GPU, float);
INSTANTIATE_GETRF(DdlaBackend::GPU, double);
INSTANTIATE_GETRF(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_GETRF(DdlaBackend::GPU, std::complex<double>);
#endif

DDLA_FOR_EACH_TYPE(INSTANTIATE_GETRF_SIZE)

#undef INSTANTIATE_GETRF_SIZE
#undef INSTANTIATE_GETRF

} // namespace ddla
