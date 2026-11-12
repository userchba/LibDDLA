#include "blas_impl.h"
// ---------------------------------------------------------------------------
// Unified backend-templated Hermitian matrix-vector product
//   y := alpha*A*x + beta*y.
//
// Dispatch follows the Backend template parameter. CPU code is always compiled;
// DDLA_HAS_GPU enables GPU code, and DDLA_USE_CUDA / DDLA_USE_HIP select its vendor.
//
// Scalar convention: xHEMV takes alpha and beta of the operands' own type T
// (both real for the real scalar types, complex for the complex ones), so the
// signature takes `const T&` for both.  For the real scalar types the routine
// is xSYMV; there is no transpose flag in this family.
// ---------------------------------------------------------------------------
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
// Real: xHEMV has no real-scalar BLAS counterpart, so the real types use xSYMV.
void ssymv_(const char*, const int*, const float*, const float*, const int*,
            const float*, const int*, const float*, float*, const int*);
void dsymv_(const char*, const int*, const double*, const double*, const int*,
            const double*, const int*, const double*, double*, const int*);
// Complex: xHEMV with complex alpha/beta.
void chemv_(const char*, const int*, const std::complex<float>*,
            const std::complex<float>*, const int*, const std::complex<float>*, const int*,
            const std::complex<float>*, std::complex<float>*, const int*);
void zhemv_(const char*, const int*, const std::complex<double>*,
            const std::complex<double>*, const int*, const std::complex<double>*, const int*,
            const std::complex<double>*, std::complex<double>*, const int*);
}

// ---------------------------------------------------------------------------
// GPU arm: the vendor calls.  Kept as free gpublasHemv overloads so the
// dispatch above stays a single BLAS_CHECK call per scalar type, exactly as
// ddla::gemm does it inline.  Private to this translation unit -- the public
// interface is the templated ddla::hemv declared in <ddla/ddla.h>.
// ---------------------------------------------------------------------------
#if DDLA_HAS_GPU

namespace ddla {

inline gpublasStatus_t gpublasHemv(
    gpublasHandle_t handle, gpublasFillMode_t uplo, int n,
    const float& alpha,
    const float* A, int lda,
    const float* x, int incx,
    const float& beta,
    float* y, int incy
)
{
    // Real HEMV == SYMV.
    return gpublasSsymv(handle, uplo, n, &alpha, A, lda, x, incx, &beta, y, incy);
}

inline gpublasStatus_t gpublasHemv(
    gpublasHandle_t handle, gpublasFillMode_t uplo, int n,
    const double& alpha,
    const double* A, int lda,
    const double* x, int incx,
    const double& beta,
    double* y, int incy
)
{
    return gpublasDsymv(handle, uplo, n, &alpha, A, lda, x, incx, &beta, y, incy);
}

inline gpublasStatus_t gpublasHemv(
    gpublasHandle_t handle, gpublasFillMode_t uplo, int n,
    const std::complex<float>& alpha,
    const std::complex<float>* A, int lda,
    const std::complex<float>* x, int incx,
    const std::complex<float>& beta,
    std::complex<float>* y, int incy
)
{
    return gpublasChemv(handle, uplo, n,
        (gpublasComplex*)&alpha, (gpublasComplex*)A, lda,
        (gpublasComplex*)x, incx, (gpublasComplex*)&beta, (gpublasComplex*)y, incy);
}

inline gpublasStatus_t gpublasHemv(
    gpublasHandle_t handle, gpublasFillMode_t uplo, int n,
    const std::complex<double>& alpha,
    const std::complex<double>* A, int lda,
    const std::complex<double>* x, int incx,
    const std::complex<double>& beta,
    std::complex<double>* y, int incy
)
{
    return gpublasZhemv(handle, uplo, n,
        (gpublasDoubleComplex*)&alpha, (gpublasDoubleComplex*)A, lda,
        (gpublasDoubleComplex*)x, incx, (gpublasDoubleComplex*)&beta, (gpublasDoubleComplex*)y, incy);
}

} // namespace ddla

#endif // DDLA_HAS_GPU

namespace ddla {

namespace blas_impl {

template <DdlaBackend Backend, typename T>
void hemv_check(
    const DdlaHandle_t& handle,
    char uplo,
    int n,
    const T& alpha,
    const T* A, int lda,
    const T* x, int incx,
    const T& beta,
    T* y, int incy)
{
    detail::require_backend<Backend>(handle, "hemv");

    if constexpr (Backend == DdlaBackend::GPU) {
        (void)detail::check_uplo(uplo);
    }
}

template <DdlaBackend Backend, typename T>
void hemv_run(
    const DdlaHandle_t& handle,
    char uplo,
    int n,
    const T& alpha,
    const T* A, int lda,
    const T* x, int incx,
    const T& beta,
    T* y, int incy)
{
    if constexpr (Backend == DdlaBackend::CPU) {
        // Real data: HEMV == SYMV; complex data calls xHEMV.  There is no
        // transpose flag in this family, so nothing needs normalising.
        if constexpr (std::is_same_v<T, float>) {
            const char u = uplo;
            ssymv_(&u, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
        } else if constexpr (std::is_same_v<T, double>) {
            const char u = uplo;
            dsymv_(&u, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            const char u = uplo;
            chemv_(&u, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            const char u = uplo;
            zhemv_(&u, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
        } else {
            static_assert(sizeof(T) == 0, "hemv: unsupported scalar type T");
        }
    } else {
#if DDLA_HAS_GPU
        // The real types map onto xSYMV inside gpublasHemv.
        if constexpr (std::is_same_v<T, float> ||
                      std::is_same_v<T, double> ||
                      std::is_same_v<T, std::complex<float>> ||
                      std::is_same_v<T, std::complex<double>>) {
            BLAS_CHECK(gpublasHemv(static_cast<gpublasHandle_t>(handle->blasH),
                                   blas_impl::fill(uplo),
                                   n, alpha, A, lda, x, incx, beta, y, incy));
        } else {
            static_assert(sizeof(T) == 0, "hemv: unsupported scalar type T");
        }
#else
        (void)handle; (void)uplo; (void)n; (void)alpha; (void)A; (void)lda;
        (void)x; (void)incx; (void)beta; (void)y; (void)incy;
#endif
    }
}

} // namespace blas_impl

template <DdlaBackend Backend, typename T>
void hemv(
    const DdlaHandle_t& handle,
    char uplo,
    int n,
    const T& alpha,
    const T* A, int lda,
    const T* x, int incx,
    const T& beta,
    T* y, int incy)
{
    blas_impl::hemv_check<Backend, T>(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
    blas_impl::hemv_run<Backend, T>(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

#define INSTANTIATE_HEMV(BACKEND, TYPE)                                        \
    template void hemv<BACKEND, TYPE>(                                         \
        const DdlaHandle_t&, char, int,                                        \
        const TYPE&, const TYPE*, int, const TYPE*, int,                       \
        const TYPE&, TYPE*, int)

DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_HEMV)

#undef INSTANTIATE_HEMV


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::hemv_check<BACKEND, TYPE>( const DdlaHandle_t& handle, char uplo, int n, const TYPE& alpha, const TYPE* A, int lda, const TYPE* x, int incx, const TYPE& beta, TYPE* y, int incy); \
    template void blas_impl::hemv_run<BACKEND, TYPE>( const DdlaHandle_t& handle, char uplo, int n, const TYPE& alpha, const TYPE* A, int lda, const TYPE* x, int incx, const TYPE& beta, TYPE* y, int incy)
DDLA_FOR_EACH_BACKEND_TYPE(DDLA_INSTANTIATE_PHASES)
#undef DDLA_INSTANTIATE_PHASES
} // namespace ddla
