#include "blas_impl.h"
// ---------------------------------------------------------------------------
// Unified backend-templated matrix-vector product
// (y := alpha*blas_impl::op(A)*x + beta*y).
//
// Dispatch follows the Backend template parameter. CPU code is always compiled;
// DDLA_HAS_GPU enables GPU code, and DDLA_USE_CUDA / DDLA_USE_HIP select its vendor.
//
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <complex>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "blas_enums.h"
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "instantiate.h"
#include "require_backend.h"

extern "C" {
// xGEMV takes the transpose character directly ('N', 'T' or 'C' -- the complex
// types accept the conjugate transpose), so the CPU arm forwards the caller's
// flag unchanged and needs no conversion.
void sgemv_(const char*, const int*, const int*, const float*,
            const float*, const int*, const float*, const int*,
            const float*, float*, const int*);
void dgemv_(const char*, const int*, const int*, const double*,
            const double*, const int*, const double*, const int*,
            const double*, double*, const int*);
void cgemv_(const char*, const int*, const int*, const std::complex<float>*,
            const std::complex<float>*, const int*, const std::complex<float>*, const int*,
            const std::complex<float>*, std::complex<float>*, const int*);
void zgemv_(const char*, const int*, const int*, const std::complex<double>*,
            const std::complex<double>*, const int*, const std::complex<double>*, const int*,
            const std::complex<double>*, std::complex<double>*, const int*);
}

// ---------------------------------------------------------------------------
// GPU arm: the vendor calls.  Kept as free gpublasGemv overloads so the dispatch
// above stays a single BLAS_CHECK call per scalar type, exactly as
// ddla::gemm does it inline.  Private to this translation unit -- the public
// interface is the templated ddla::gemv declared in <ddla/ddla.h>.
// ---------------------------------------------------------------------------
#if DDLA_HAS_GPU

namespace ddla {

inline gpublasStatus_t gpublasGemv(
    gpublasHandle_t handle, gpublasOperation_t trans,
    int m, int n,
    const float& alpha,
    const float* A, int lda,
    const float* x, int incx,
    const float& beta,
    float* y, int incy
)
{
    #if DDLA_GPU_VENDOR
    return gpublasSgemv(handle, trans, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
    #else
    // Neither vendor macro: this can only happen if the translation unit was
    // built without a GPU language while DDLA_HAS_GPU is set, which the build
    // system does not do (see the LANGUAGE CUDA/HIP source lists).  Throw
    // rather than fall through to a host pointer.
    throw std::runtime_error("gpublasGemv: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasGemv(
    gpublasHandle_t handle, gpublasOperation_t trans,
    int m, int n,
    const double& alpha,
    const double* A, int lda,
    const double* x, int incx,
    const double& beta,
    double* y, int incy
)
{
    #if DDLA_GPU_VENDOR
    return gpublasDgemv(handle, trans, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
    #else
    throw std::runtime_error("gpublasGemv: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasGemv(
    gpublasHandle_t handle, gpublasOperation_t trans,
    int m, int n,
    const std::complex<float>& alpha,
    const std::complex<float>* A, int lda,
    const std::complex<float>* x, int incx,
    const std::complex<float>& beta,
    std::complex<float>* y, int incy
)
{
    #if DDLA_GPU_VENDOR
    return gpublasCgemv(handle, trans, m, n,
        (gpublasComplex*)&alpha, (gpublasComplex*)A, lda,
        (gpublasComplex*)x, incx, (gpublasComplex*)&beta, (gpublasComplex*)y, incy);
    #else
    throw std::runtime_error("gpublasGemv: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasGemv(
    gpublasHandle_t handle, gpublasOperation_t trans,
    int m, int n,
    const std::complex<double>& alpha,
    const std::complex<double>* A, int lda,
    const std::complex<double>* x, int incx,
    const std::complex<double>& beta,
    std::complex<double>* y, int incy
)
{
    #if DDLA_GPU_VENDOR
    return gpublasZgemv(handle, trans, m, n,
        (gpublasDoubleComplex*)&alpha, (gpublasDoubleComplex*)A, lda,
        (gpublasDoubleComplex*)x, incx, (gpublasDoubleComplex*)&beta, (gpublasDoubleComplex*)y, incy);
    #else
    throw std::runtime_error("gpublasGemv: no GPU vendor selected at compile time");
    #endif
}

} // namespace ddla

#endif // DDLA_HAS_GPU

namespace ddla {

namespace blas_impl {

template <DdlaBackend Backend, typename T>
void gemv_check(
    const DdlaHandle_t& handle,
    char trans, int m, int n,
    const T& alpha,
    const T* A, int lda,
    const T* x, int incx,
    const T& beta,
    T* y, int incy)
{
    detail::require_backend<Backend>(handle, "gemv");

    if constexpr (Backend == DdlaBackend::GPU) {
        (void)detail::to_blas_op(trans);
    }
}

template <DdlaBackend Backend, typename T>
void gemv_run(
    const DdlaHandle_t& handle,
    char trans, int m, int n,
    const T& alpha,
    const T* A, int lda,
    const T* x, int incx,
    const T& beta,
    T* y, int incy)
{
    if constexpr (Backend == DdlaBackend::CPU) {
        // xGEMV consumes the transpose character directly, so the caller's
        // flag ('N', 'T' or 'C') is forwarded unchanged.
        if constexpr (std::is_same_v<T, float>) {
            sgemv_(&trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
        } else if constexpr (std::is_same_v<T, double>) {
            dgemv_(&trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            cgemv_(&trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            zgemv_(&trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
        } else {
            static_assert(sizeof(T) == 0, "gemv: unsupported scalar type T");
        }
    } else {
#if DDLA_HAS_GPU
        if constexpr (std::is_same_v<T, float> ||
                      std::is_same_v<T, double> ||
                      std::is_same_v<T, std::complex<float>> ||
                      std::is_same_v<T, std::complex<double>>) {
            BLAS_CHECK(gpublasGemv(static_cast<gpublasHandle_t>(handle->blasH),
                                   blas_impl::op(trans),
                                   m, n, alpha, A, lda, x, incx, beta, y, incy));
        } else {
            static_assert(sizeof(T) == 0, "gemv: unsupported scalar type T");
        }
#else
        (void)handle; (void)trans; (void)m; (void)n; (void)alpha; (void)A; (void)lda;
        (void)x; (void)incx; (void)beta; (void)y; (void)incy;
#endif
    }
}

} // namespace blas_impl

template <DdlaBackend Backend, typename T>
void gemv(
    const DdlaHandle_t& handle,
    char trans, int m, int n,
    const T& alpha,
    const T* A, int lda,
    const T* x, int incx,
    const T& beta,
    T* y, int incy)
{
    blas_impl::gemv_check<Backend, T>(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    blas_impl::gemv_run<Backend, T>(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

#define INSTANTIATE_GEMV(BACKEND, TYPE)                                        \
    template void gemv<BACKEND, TYPE>(                                         \
        const DdlaHandle_t&, char, int, int,                                   \
        const TYPE&, const TYPE*, int,                                         \
        const TYPE*, int, const TYPE&, TYPE*, int)

DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_GEMV)

#undef INSTANTIATE_GEMV


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::gemv_check<BACKEND, TYPE>( const DdlaHandle_t& handle, char trans, int m, int n, const TYPE& alpha, const TYPE* A, int lda, const TYPE* x, int incx, const TYPE& beta, TYPE* y, int incy); \
    template void blas_impl::gemv_run<BACKEND, TYPE>( const DdlaHandle_t& handle, char trans, int m, int n, const TYPE& alpha, const TYPE* A, int lda, const TYPE* x, int incx, const TYPE& beta, TYPE* y, int incy)
DDLA_FOR_EACH_BACKEND_TYPE(DDLA_INSTANTIATE_PHASES)
#undef DDLA_INSTANTIATE_PHASES
} // namespace ddla
