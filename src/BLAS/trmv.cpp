#include "blas_impl.h"
// ---------------------------------------------------------------------------
// Unified backend-templated triangular matrix-vector product (x := blas_impl::op(A)*x,
// in place).
//
// Dispatch follows the Backend template parameter. CPU code is always compiled;
// DDLA_HAS_GPU enables GPU code, and DDLA_USE_CUDA / DDLA_USE_HIP select its vendor.
//
// Note on the operand: `x` is modified IN PLACE.  The templated signature
// takes `char uplo/diag` (LAPACK convention) and converts inside the GPU arm;
// the Fortran arm needs the characters and passes them straight through.
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
// xTRMV takes uplo, trans and diag as characters, so the CPU arm forwards all
// three unchanged and needs no conversion.
void strmv_(const char*, const char*, const char*, const int*,
            const float*, const int*, float*, const int*);
void dtrmv_(const char*, const char*, const char*, const int*,
            const double*, const int*, double*, const int*);
void ctrmv_(const char*, const char*, const char*, const int*,
            const std::complex<float>*, const int*, std::complex<float>*, const int*);
void ztrmv_(const char*, const char*, const char*, const int*,
            const std::complex<double>*, const int*, std::complex<double>*, const int*);
}

// ---------------------------------------------------------------------------
// GPU arm: the vendor calls.  Kept as free gpublasTrmv overloads so the
// dispatch above stays a single BLAS_CHECK call per scalar type, exactly as
// ddla::gemm does it inline.  Private to this translation unit -- the public
// interface is the templated ddla::trmv declared in <ddla/ddla.h>.
//
// The complex overloads cast the const `A` away: the vendor prototypes take a
// non-const operand even though xTRMV only reads the triangle.  This matches
// the pre-unification header (src/trmv.h) verbatim.
// ---------------------------------------------------------------------------
#if DDLA_HAS_GPU

namespace ddla {

inline gpublasStatus_t gpublasTrmv(
    gpublasHandle_t handle, gpublasFillMode_t uplo,
    gpublasOperation_t trans, gpublasDiagType_t diag,
    int n,
    const float* A, int lda,
    float* x, int incx
)
{
    return gpublasStrmv(handle, uplo, trans, diag, n, A, lda, x, incx);
}

inline gpublasStatus_t gpublasTrmv(
    gpublasHandle_t handle, gpublasFillMode_t uplo,
    gpublasOperation_t trans, gpublasDiagType_t diag,
    int n,
    const double* A, int lda,
    double* x, int incx
)
{
    return gpublasDtrmv(handle, uplo, trans, diag, n, A, lda, x, incx);
}

inline gpublasStatus_t gpublasTrmv(
    gpublasHandle_t handle, gpublasFillMode_t uplo,
    gpublasOperation_t trans, gpublasDiagType_t diag,
    int n,
    const std::complex<float>* A, int lda,
    std::complex<float>* x, int incx
)
{
    return gpublasCtrmv(handle, uplo, trans, diag, n,
        (gpublasComplex*)A, lda, (gpublasComplex*)x, incx);
}

inline gpublasStatus_t gpublasTrmv(
    gpublasHandle_t handle, gpublasFillMode_t uplo,
    gpublasOperation_t trans, gpublasDiagType_t diag,
    int n,
    const std::complex<double>* A, int lda,
    std::complex<double>* x, int incx
)
{
    return gpublasZtrmv(handle, uplo, trans, diag, n,
        (gpublasDoubleComplex*)A, lda, (gpublasDoubleComplex*)x, incx);
}

} // namespace ddla

#endif // DDLA_HAS_GPU

namespace ddla {

namespace blas_impl {

template <DdlaBackend Backend, typename T>
void trmv_check(
    const DdlaHandle_t& handle,
    char uplo, char trans, char diag,
    int n,
    const T* A, int lda,
    T* x, int incx)
{
    detail::require_backend<Backend>(handle, "trmv");

    if constexpr (Backend == DdlaBackend::GPU) {
        (void)detail::check_uplo(uplo);
        (void)detail::check_trans(trans);
        (void)detail::check_diag(diag);
    }
}

template <DdlaBackend Backend, typename T>
void trmv_run(
    const DdlaHandle_t& handle,
    char uplo, char trans, char diag,
    int n,
    const T* A, int lda,
    T* x, int incx)
{
    if constexpr (Backend == DdlaBackend::CPU) {
        // xTRMV takes uplo/trans/diag as characters; forward them unchanged.
        if constexpr (std::is_same_v<T, float>) {
            strmv_(&uplo, &trans, &diag, &n, A, &lda, x, &incx);
        } else if constexpr (std::is_same_v<T, double>) {
            dtrmv_(&uplo, &trans, &diag, &n, A, &lda, x, &incx);
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            ctrmv_(&uplo, &trans, &diag, &n, A, &lda, x, &incx);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            ztrmv_(&uplo, &trans, &diag, &n, A, &lda, x, &incx);
        } else {
            static_assert(sizeof(T) == 0, "trmv: unsupported scalar type T");
        }
    } else {
#if DDLA_HAS_GPU
        if constexpr (std::is_same_v<T, float> ||
                      std::is_same_v<T, double> ||
                      std::is_same_v<T, std::complex<float>> ||
                      std::is_same_v<T, std::complex<double>>) {
            BLAS_CHECK(gpublasTrmv(static_cast<gpublasHandle_t>(handle->blasH),
                                   blas_impl::fill(uplo),
                                   blas_impl::op(trans),
                                   blas_impl::diag(diag),
                                   n, A, lda, x, incx));
        } else {
            static_assert(sizeof(T) == 0, "trmv: unsupported scalar type T");
        }
#else
        (void)handle; (void)uplo; (void)trans; (void)diag; (void)n;
        (void)A; (void)lda; (void)x; (void)incx;
#endif
    }
}

} // namespace blas_impl

template <DdlaBackend Backend, typename T>
void trmv(
    const DdlaHandle_t& handle,
    char uplo, char trans, char diag,
    int n,
    const T* A, int lda,
    T* x, int incx)
{
    blas_impl::trmv_check<Backend, T>(handle, uplo, trans, diag, n, A, lda, x, incx);
    blas_impl::trmv_run<Backend, T>(handle, uplo, trans, diag, n, A, lda, x, incx);
}

#define INSTANTIATE_TRMV(BACKEND, TYPE)                                        \
    template void trmv<BACKEND, TYPE>(                                         \
        const DdlaHandle_t&, char, char, char, int,                            \
        const TYPE*, int, TYPE*, int)

DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_TRMV)

#undef INSTANTIATE_TRMV


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::trmv_check<BACKEND, TYPE>( const DdlaHandle_t& handle, char uplo, char trans, char diag, int n, const TYPE* A, int lda, TYPE* x, int incx); \
    template void blas_impl::trmv_run<BACKEND, TYPE>( const DdlaHandle_t& handle, char uplo, char trans, char diag, int n, const TYPE* A, int lda, TYPE* x, int incx)
DDLA_FOR_EACH_BACKEND_TYPE(DDLA_INSTANTIATE_PHASES)
#undef DDLA_INSTANTIATE_PHASES
} // namespace ddla
