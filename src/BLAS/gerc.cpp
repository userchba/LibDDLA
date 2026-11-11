#include "blas_impl.h"
// ---------------------------------------------------------------------------
// Unified backend-templated conjugate rank-1 update (A := alpha*x*y^H + A).
//
// This is the routine LAPACK's xLARF uses to apply a Householder reflector
// from the left:  A := A - tau * v * w^H.  The update touches the leading
// m x n submatrix of A only.
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
// Real: xGER (same signature for real and complex; real is plain xGER).
void sger_(const int*, const int*, const float*,
           const float*, const int*, const float*, const int*,
           float*, const int*);
void dger_(const int*, const int*, const double*,
           const double*, const int*, const double*, const int*,
           double*, const int*);
// Complex: xGERC (conjugated).
void cgerc_(const int*, const int*, const std::complex<float>*,
            const std::complex<float>*, const int*,
            const std::complex<float>*, const int*,
            std::complex<float>*, const int*);
void zgerc_(const int*, const int*, const std::complex<double>*,
            const std::complex<double>*, const int*,
            const std::complex<double>*, const int*,
            std::complex<double>*, const int*);
}

// ---------------------------------------------------------------------------
// GPU arm: the vendor calls.  Kept as free gpublasGerc overloads so the dispatch
// above stays a single BLAS_CHECK call per scalar type, exactly as
// ddla::gemm does it inline.  Private to this translation unit -- the public
// interface is the templated ddla::gerc declared in <ddla/ddla.h>.
// ---------------------------------------------------------------------------
#if DDLA_HAS_GPU

namespace ddla {

inline gpublasStatus_t gpublasGerc(
    gpublasHandle_t handle, int m, int n,
    const float& alpha, const float* x, int incx,
    const float* y, int incy, float* A, int lda)
{
    #if DDLA_GPU_VENDOR
    return gpublasSger(handle, m, n, &alpha, x, incx, y, incy, A, lda);
    #else
    throw std::runtime_error("gpublasGerc: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasGerc(
    gpublasHandle_t handle, int m, int n,
    const double& alpha, const double* x, int incx,
    const double* y, int incy, double* A, int lda)
{
    #if DDLA_GPU_VENDOR
    return gpublasDger(handle, m, n, &alpha, x, incx, y, incy, A, lda);
    #else
    throw std::runtime_error("gpublasGerc: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasGerc(
    gpublasHandle_t handle, int m, int n,
    const std::complex<float>& alpha, const std::complex<float>* x, int incx,
    const std::complex<float>* y, int incy, std::complex<float>* A, int lda)
{
    #if DDLA_GPU_VENDOR
    return gpublasCgerc(handle, m, n,
                        (gpublasComplex*)&alpha,
                        (gpublasComplex*)x, incx,
                        (gpublasComplex*)y, incy,
                        (gpublasComplex*)A, lda);
    #else
    throw std::runtime_error("gpublasGerc: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasGerc(
    gpublasHandle_t handle, int m, int n,
    const std::complex<double>& alpha, const std::complex<double>* x, int incx,
    const std::complex<double>* y, int incy, std::complex<double>* A, int lda)
{
    #if DDLA_GPU_VENDOR
    return gpublasZgerc(handle, m, n,
                      (gpublasDoubleComplex*)&alpha,
                      (gpublasDoubleComplex*)x, incx,
                      (gpublasDoubleComplex*)y, incy,
                      (gpublasDoubleComplex*)A, lda);
    #else
    throw std::runtime_error("gpublasGerc: no GPU vendor selected at compile time");
    #endif
}

} // namespace ddla

#endif // DDLA_HAS_GPU

namespace ddla {

// ---------------------------------------------------------------------------
// GPU dispatch (single BLAS_CHECK call per scalar type).
// ---------------------------------------------------------------------------
namespace blas_impl {

template <DdlaBackend Backend, typename T>
void gerc_check(const DdlaHandle_t& handle, int m, int n,
          const T& alpha, const T* x, int incx,
          const T* y, int incy, T* A, int lda)
{
    detail::require_backend<Backend>(handle, "gerc");

    if (m <= 0 || n <= 0) return;

    if constexpr (Backend == DdlaBackend::CPU) {
        if (incx == 0 || incy == 0) throw std::runtime_error("gerc: incx and incy must be nonzero");
    }
}

template <DdlaBackend Backend, typename T>
void gerc_run(const DdlaHandle_t& handle, int m, int n,
          const T& alpha, const T* x, int incx,
          const T* y, int incy, T* A, int lda)
{
    if (m <= 0 || n <= 0) return;
    if constexpr (Backend == DdlaBackend::CPU) {
if constexpr (std::is_same_v<T, float>) {
            sger_(&m, &n, &alpha, x, &incx, y, &incy, A, &lda);
        } else if constexpr (std::is_same_v<T, double>) {
            dger_(&m, &n, &alpha, x, &incx, y, &incy, A, &lda);
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            cgerc_(&m, &n, &alpha, x, &incx, y, &incy, A, &lda);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            zgerc_(&m, &n, &alpha, x, &incx, y, &incy, A, &lda);
        } else {
            static_assert(sizeof(T) == 0, "gerc: unsupported scalar type T");
        }
    } else {
        #if DDLA_HAS_GPU
        BLAS_CHECK(gpublasGerc(static_cast<gpublasHandle_t>(handle->blasH), m, n, alpha, x, incx, y, incy, A, lda));
        #endif
    }
}

} // namespace blas_impl

template <DdlaBackend Backend, typename T>
void gerc(const DdlaHandle_t& handle, int m, int n,
          const T& alpha, const T* x, int incx,
          const T* y, int incy, T* A, int lda)
{
    blas_impl::gerc_check<Backend, T>(handle, m, n, alpha, x, incx, y, incy, A, lda);
    blas_impl::gerc_run<Backend, T>(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

#define INSTANTIATE_GERC(BACKEND, TYPE)                                      \
    template void gerc<BACKEND, TYPE>(                                       \
        const DdlaHandle_t&, int, int, const TYPE&, const TYPE*, int,        \
        const TYPE*, int, TYPE*, int)

DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_GERC)

#undef INSTANTIATE_GERC


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::gerc_check<BACKEND, TYPE>(const DdlaHandle_t& handle, int m, int n, const TYPE& alpha, const TYPE* x, int incx, const TYPE* y, int incy, TYPE* A, int lda); \
    template void blas_impl::gerc_run<BACKEND, TYPE>(const DdlaHandle_t& handle, int m, int n, const TYPE& alpha, const TYPE* x, int incx, const TYPE* y, int incy, TYPE* A, int lda)
DDLA_FOR_EACH_BACKEND_TYPE(DDLA_INSTANTIATE_PHASES)
#undef DDLA_INSTANTIATE_PHASES
} // namespace ddla
