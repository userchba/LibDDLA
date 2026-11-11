#include "blas_impl.h"
// ---------------------------------------------------------------------------
// Unified backend-templated triangular solve (side):
//   side='L': B := alpha*blas_impl::op(A)^{-1}*B   (A is m-by-m)
//   side='R': B := alpha*B*blas_impl::op(A)^{-1}   (A is n-by-n)
//
// Dispatch follows the Backend template parameter. CPU code is always compiled;
// DDLA_HAS_GPU enables GPU code, and DDLA_USE_CUDA / DDLA_USE_HIP select its vendor.
//
// The public entry point (include/ddla/ddla.h) takes LAPACK-style characters,
// which the GPU arm converts with the blas_enums.h helpers; the CPU arm consumes
// the characters directly.
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
void strsm_(const char*, const char*, const char*, const char*, const int*, const int*,
            const float*, const float*, const int*, float*, const int*);
void dtrsm_(const char*, const char*, const char*, const char*, const int*, const int*,
            const double*, const double*, const int*, double*, const int*);
void ctrsm_(const char*, const char*, const char*, const char*, const int*, const int*,
            const std::complex<float>*, const std::complex<float>*, const int*,
            std::complex<float>*, const int*);
void ztrsm_(const char*, const char*, const char*, const char*, const int*, const int*,
            const std::complex<double>*, const std::complex<double>*, const int*,
            std::complex<double>*, const int*);
}

// ---------------------------------------------------------------------------
// GPU arm: the vendor calls.  Kept as free gpublasTrsm overloads so the dispatch
// below stays a single BLAS_CHECK call per scalar type, exactly as
// ddla::gemm does it inline.  Private to this translation unit -- the public
// interface is the templated ddla::trsm declared in <ddla/ddla.h>.
//
// A and B are deliberately NON-const: the vendor prototypes take non-const
// pointers even though xTRSM only reads A and reads/writes B.
// ---------------------------------------------------------------------------
#if DDLA_HAS_GPU

namespace ddla {

inline gpublasStatus_t gpublasTrsm(
    gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo, gpublasOperation_t trans, gpublasDiagType_t diag,
    int m, int n,
    const std::complex<double>& alpha,
    std::complex<double> *A, int lda,
    std::complex<double> *B, int ldb
)
{
    #if DDLA_GPU_VENDOR
    return gpublasZtrsm(handle, side, uplo, trans, diag, m, n, (gpublasDoubleComplex*)&alpha, (gpublasDoubleComplex*)A, lda, (gpublasDoubleComplex*)B, ldb);
    #else
    // Neither vendor macro: this can only happen if the translation unit was
    // built without a GPU language while DDLA_HAS_GPU is set, which the build
    // system does not do (see the LANGUAGE CUDA/HIP source lists).  Throw
    // rather than fall through to a host pointer.
    throw std::runtime_error("gpublasTrsm: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasTrsm(
    gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo, gpublasOperation_t trans, gpublasDiagType_t diag,
    int m, int n,
    const std::complex<float>& alpha,
    std::complex<float> *A, int lda,
    std::complex<float> *B, int ldb
)
{
    #if DDLA_GPU_VENDOR
    return gpublasCtrsm(handle, side, uplo, trans, diag, m, n, (gpublasComplex*)&alpha, (gpublasComplex*)A, lda, (gpublasComplex*)B, ldb);
    #else
    throw std::runtime_error("gpublasTrsm: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasTrsm(
    gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo, gpublasOperation_t trans, gpublasDiagType_t diag,
    int m, int n,
    const float& alpha,
    float *A, int lda,
    float *B, int ldb
)
{
    #if DDLA_GPU_VENDOR
    return gpublasStrsm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb);
    #else
    throw std::runtime_error("gpublasTrsm: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasTrsm(
    gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo, gpublasOperation_t trans, gpublasDiagType_t diag,
    int m, int n,
    const double& alpha,
    double *A, int lda,
    double *B, int ldb
)
{
    #if DDLA_GPU_VENDOR
    return gpublasDtrsm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb);
    #else
    throw std::runtime_error("gpublasTrsm: no GPU vendor selected at compile time");
    #endif
}

} // namespace ddla

#endif // DDLA_HAS_GPU

namespace ddla {

namespace blas_impl {

template <DdlaBackend Backend, typename T>
void trsm_check(
    const DdlaHandle_t& handle,
    char side, char uplo, char trans, char diag,
    int m, int n,
    const T& alpha,
    T* A, int lda,
    T* B, int ldb)
{
    detail::require_backend<Backend>(handle, "trsm");

    if constexpr (Backend == DdlaBackend::GPU) {
        (void)detail::to_side_mode(side);
        (void)detail::to_fill_mode(uplo);
        (void)detail::to_blas_op(trans);
        (void)detail::to_diag_type(diag);
    }
}

template <DdlaBackend Backend, typename T>
void trsm_run(
    const DdlaHandle_t& handle,
    char side, char uplo, char trans, char diag,
    int m, int n,
    const T& alpha,
    T* A, int lda,
    T* B, int ldb)
{
    if constexpr (Backend == DdlaBackend::CPU) {
        // The Fortran entry points take the LAPACK characters directly, so no
        // conversion is needed on this path.
        if constexpr (std::is_same_v<T, float>) {
            strsm_(&side, &uplo, &trans, &diag, &m, &n, &alpha, A, &lda, B, &ldb);
        } else if constexpr (std::is_same_v<T, double>) {
            dtrsm_(&side, &uplo, &trans, &diag, &m, &n, &alpha, A, &lda, B, &ldb);
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            ctrsm_(&side, &uplo, &trans, &diag, &m, &n, &alpha, A, &lda, B, &ldb);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            ztrsm_(&side, &uplo, &trans, &diag, &m, &n, &alpha, A, &lda, B, &ldb);
        } else {
            static_assert(sizeof(T) == 0, "trsm: unsupported scalar type T");
        }
    } else {
#if DDLA_HAS_GPU
        if constexpr (std::is_same_v<T, float> ||
                      std::is_same_v<T, double> ||
                      std::is_same_v<T, std::complex<float>> ||
                      std::is_same_v<T, std::complex<double>>) {
            BLAS_CHECK(gpublasTrsm(static_cast<gpublasHandle_t>(handle->blasH),
                                   blas_impl::side(side),
                                   blas_impl::fill(uplo),
                                   blas_impl::op(trans),
                                   blas_impl::diag(diag),
                                   m, n, alpha, A, lda, B, ldb));
        } else {
            static_assert(sizeof(T) == 0, "trsm: unsupported scalar type T");
        }
#else
        (void)handle; (void)side; (void)uplo; (void)trans; (void)diag;
        (void)m; (void)n; (void)alpha; (void)A; (void)lda; (void)B; (void)ldb;
#endif
    }
}

} // namespace blas_impl

template <DdlaBackend Backend, typename T>
void trsm(
    const DdlaHandle_t& handle,
    char side, char uplo, char trans, char diag,
    int m, int n,
    const T& alpha,
    T* A, int lda,
    T* B, int ldb)
{
    blas_impl::trsm_check<Backend, T>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    blas_impl::trsm_run<Backend, T>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

#define INSTANTIATE_TRSM(BACKEND, TYPE)                                        \
    template void trsm<BACKEND, TYPE>(                                         \
        const DdlaHandle_t&, char, char, char, char, int, int,                 \
        const TYPE&, TYPE*, int, TYPE*, int)

DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_TRSM)

#undef INSTANTIATE_TRSM


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::trsm_check<BACKEND, TYPE>( const DdlaHandle_t& handle, char side, char uplo, char trans, char diag, int m, int n, const TYPE& alpha, TYPE* A, int lda, TYPE* B, int ldb); \
    template void blas_impl::trsm_run<BACKEND, TYPE>( const DdlaHandle_t& handle, char side, char uplo, char trans, char diag, int m, int n, const TYPE& alpha, TYPE* A, int lda, TYPE* B, int ldb)
DDLA_FOR_EACH_BACKEND_TYPE(DDLA_INSTANTIATE_PHASES)
#undef DDLA_INSTANTIATE_PHASES
} // namespace ddla
