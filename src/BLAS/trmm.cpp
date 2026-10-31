#include "blas_impl.h"
// ---------------------------------------------------------------------------
// Unified backend-templated triangular matrix multiply
// (B := alpha*blas_impl::op(A)*B or B := alpha*B*blas_impl::op(A), per @p side).
//
// Structure follows src/herk.cpp exactly: one definition covering CPU (Fortran
// BLAS) and GPU (cuBLAS/hipBLAS), selected with `if constexpr` on the Backend
// template parameter, with the arms themselves guarded by the BUILD CAPABILITY
// macros (DDLA_HAS_CPU / DDLA_HAS_GPU) and the vendor chosen per scalar arm by
// the PER-TU macros (DDLA_USE_CUDA / DDLA_USE_HIP).
//
// Why the capability guard matters here: the previous design put a `#else`
// CPU arm inside src/trmm.h selected by `!defined(DDLA_USE_CUDA) &&
// !defined(DDLA_USE_HIP)`.  In a dual build (DDLA_USE_CPU + DDLA_USE_CUDA)
// every TU defines DDLA_USE_CUDA, so that arm was never compiled -- while
// DDLA_HAS_CPU was 1 and the CPU instantiation of this function still existed.
// A CPU-backend call would therefore have reached cublasStrmm with HOST
// pointers.  Guarding on DDLA_HAS_CPU removes that class of bug.
//
// The public interface follows BLAS and overwrites B. The GPU arm uses the
// vendor-supported in-place case by passing B and ldb as both input and output.
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <complex>
#include <cstddef>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "blas_enums.h"
#include "ddla_connector.h"
#include "ddla_stream.h"

#if DDLA_HAS_CPU
extern "C" {
void strmm_(const char*, const char*, const char*, const char*, const int*, const int*,
            const float*, const float*, const int*, float*, const int*);
void dtrmm_(const char*, const char*, const char*, const char*, const int*, const int*,
            const double*, const double*, const int*, double*, const int*);
void ctrmm_(const char*, const char*, const char*, const char*, const int*, const int*,
            const std::complex<float>*, const std::complex<float>*, const int*,
            std::complex<float>*, const int*);
void ztrmm_(const char*, const char*, const char*, const char*, const int*, const int*,
            const std::complex<double>*, const std::complex<double>*, const int*,
            std::complex<double>*, const int*);
}
#endif

// ---------------------------------------------------------------------------
// GPU arm: the vendor calls.  Kept as free gpublasTrmm overloads so the
// dispatch below stays a single BLAS_CHECK call per scalar type, exactly as
// ddla::gemm does it inline.  Private to this translation unit -- the public
// interface is the templated ddla::trmm declared in <ddla/ddla.h>.
// ---------------------------------------------------------------------------
#if DDLA_HAS_GPU

namespace ddla {

inline gpublasStatus_t gpublasTrmm(
    gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo,
    gpublasOperation_t trans, gpublasDiagType_t diag,
    int m, int n,
    const float& alpha,
    const float* A, int lda,
    float* B, int ldb
)
{
#if DDLA_GPU_VENDOR
    return gpublasStrmm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb, B, ldb);
#else
    // Neither vendor macro: this can only happen if the translation unit was
    // built without a GPU language while DDLA_HAS_GPU is set, which the build
    // system does not do (see the LANGUAGE CUDA/HIP source lists).  Throw
    // rather than fall through to a host pointer.
    throw std::runtime_error("gpublasTrmm: no GPU vendor selected at compile time");
#endif
}

inline gpublasStatus_t gpublasTrmm(
    gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo,
    gpublasOperation_t trans, gpublasDiagType_t diag,
    int m, int n,
    const double& alpha,
    const double* A, int lda,
    double* B, int ldb
)
{
#if DDLA_GPU_VENDOR
    return gpublasDtrmm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb, B, ldb);
#else
    throw std::runtime_error("gpublasTrmm: no GPU vendor selected at compile time");
#endif
}

inline gpublasStatus_t gpublasTrmm(
    gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo,
    gpublasOperation_t trans, gpublasDiagType_t diag,
    int m, int n,
    const std::complex<float>& alpha,
    const std::complex<float>* A, int lda,
    std::complex<float>* B, int ldb
)
{
#if DDLA_GPU_VENDOR
    return gpublasCtrmm(handle, side, uplo, trans, diag, m, n,
                        reinterpret_cast<const gpublasComplex*>(&alpha),
                        reinterpret_cast<const gpublasComplex*>(A), lda,
                        reinterpret_cast<gpublasComplex*>(B), ldb,
                        reinterpret_cast<gpublasComplex*>(B), ldb);
#else
    throw std::runtime_error("gpublasTrmm: no GPU vendor selected at compile time");
#endif
}

inline gpublasStatus_t gpublasTrmm(
    gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo,
    gpublasOperation_t trans, gpublasDiagType_t diag,
    int m, int n,
    const std::complex<double>& alpha,
    const std::complex<double>* A, int lda,
    std::complex<double>* B, int ldb
)
{
#if DDLA_GPU_VENDOR
    return gpublasZtrmm(handle, side, uplo, trans, diag, m, n,
                        reinterpret_cast<const gpublasDoubleComplex*>(&alpha),
                        reinterpret_cast<const gpublasDoubleComplex*>(A), lda,
                        reinterpret_cast<gpublasDoubleComplex*>(B), ldb,
                        reinterpret_cast<gpublasDoubleComplex*>(B), ldb);
#else
    throw std::runtime_error("gpublasTrmm: no GPU vendor selected at compile time");
#endif
}

} // namespace ddla

#endif // DDLA_HAS_GPU

namespace ddla {

namespace blas_impl {

template <DdlaBackend Backend, typename T>
void trmm_check(
    const DdlaHandle_t& handle,
    char side, char uplo, char trans, char diag,
    int m, int n,
    const T& alpha,
    const T* A, int lda,
    T* B, int ldb)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU trmm is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU trmm is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("trmm: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("trmm: template backend ") +
            (Backend == DdlaBackend::CPU ? "CPU" : "GPU") +
            " does not match handle backend " +
            (actual == DdlaBackend::CPU ? "CPU" : "GPU"));
    }

    // Check the BLAS shape even on quick-return paths. Scalar zero must not
    // read A or the old B, and must leave leading-dimension padding untouched.
    (void)detail::to_side_mode(side);
    (void)detail::to_fill_mode(uplo);
    (void)detail::to_blas_op(trans);
    (void)detail::to_diag_type(diag);
    const int order = (side == 'L' || side == 'l') ? m : n;
    if (m < 0 || n < 0 || lda < std::max(1, order) || ldb < std::max(1, m))
        throw std::invalid_argument("trmm: invalid dimensions or leading dimension");
    if (m == 0 || n == 0) return;
    if (B == nullptr) throw std::invalid_argument("trmm: null B");
    if (alpha != T(0) && A == nullptr) throw std::invalid_argument("trmm: null A");

}

template <DdlaBackend Backend, typename T>
void trmm_run(
    const DdlaHandle_t& handle,
    char side, char uplo, char trans, char diag,
    int m, int n,
    const T& alpha,
    const T* A, int lda,
    T* B, int ldb)
{
    if (m == 0 || n == 0) return;
    if (alpha == T(0)) {
        for (int j = 0; j < n; ++j)
            DDLA_CHECK(ddlaMemset(handle, B + static_cast<std::size_t>(j) * ldb,
                                  0, static_cast<std::size_t>(m) * sizeof(T)));
        return;
    }
    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        if constexpr (std::is_same_v<T, float>) {
            strmm_(&side, &uplo, &trans, &diag, &m, &n, &alpha, A, &lda, B, &ldb);
        } else if constexpr (std::is_same_v<T, double>) {
            dtrmm_(&side, &uplo, &trans, &diag, &m, &n, &alpha, A, &lda, B, &ldb);
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            ctrmm_(&side, &uplo, &trans, &diag, &m, &n, &alpha, A, &lda, B, &ldb);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            ztrmm_(&side, &uplo, &trans, &diag, &m, &n, &alpha, A, &lda, B, &ldb);
        } else {
            static_assert(sizeof(T) == 0, "trmm: unsupported scalar type T");
        }
#else
        (void)handle; (void)side; (void)uplo; (void)trans; (void)diag;
        (void)m; (void)n; (void)alpha; (void)A; (void)lda;
        (void)B; (void)ldb;
#endif
    } else {
#if DDLA_HAS_GPU
        if constexpr (std::is_same_v<T, float> ||
                      std::is_same_v<T, double> ||
                      std::is_same_v<T, std::complex<float>> ||
                      std::is_same_v<T, std::complex<double>>) {
            BLAS_CHECK(gpublasTrmm(static_cast<gpublasHandle_t>(handle->blasH),
                                   blas_impl::side(side),
                                   blas_impl::fill(uplo),
                                   blas_impl::op(trans),
                                   blas_impl::diag(diag),
                                   m, n, alpha, A, lda,
                                   B, ldb));
        } else {
            static_assert(sizeof(T) == 0, "trmm: unsupported scalar type T");
        }
#else
        (void)handle; (void)side; (void)uplo; (void)trans; (void)diag;
        (void)m; (void)n; (void)alpha; (void)A; (void)lda;
        (void)B; (void)ldb;
#endif
    }
}

} // namespace blas_impl

template <DdlaBackend Backend, typename T>
void trmm(
    const DdlaHandle_t& handle,
    char side, char uplo, char trans, char diag,
    int m, int n,
    const T& alpha,
    const T* A, int lda,
    T* B, int ldb)
{
    blas_impl::trmm_check<Backend, T>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    blas_impl::trmm_run<Backend, T>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

#define INSTANTIATE_TRMM(BACKEND, TYPE)                                        \
    template void trmm<BACKEND, TYPE>(                                         \
        const DdlaHandle_t&, char, char, char, char, int, int,                 \
        const TYPE&, const TYPE*, int, TYPE*, int)

#if DDLA_HAS_CPU
INSTANTIATE_TRMM(DdlaBackend::CPU, float);
INSTANTIATE_TRMM(DdlaBackend::CPU, double);
INSTANTIATE_TRMM(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_TRMM(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_TRMM(DdlaBackend::GPU, float);
INSTANTIATE_TRMM(DdlaBackend::GPU, double);
INSTANTIATE_TRMM(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_TRMM(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_TRMM


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::trmm_check<BACKEND, TYPE>( const DdlaHandle_t& handle, char side, char uplo, char trans, char diag, int m, int n, const TYPE& alpha, const TYPE* A, int lda, TYPE* B, int ldb); \
    template void blas_impl::trmm_run<BACKEND, TYPE>( const DdlaHandle_t& handle, char side, char uplo, char trans, char diag, int m, int n, const TYPE& alpha, const TYPE* A, int lda, TYPE* B, int ldb);
#if DDLA_HAS_CPU
DDLA_INSTANTIATE_PHASES(DdlaBackend::CPU, float)
DDLA_INSTANTIATE_PHASES(DdlaBackend::CPU, double)
DDLA_INSTANTIATE_PHASES(DdlaBackend::CPU, std::complex<float>)
DDLA_INSTANTIATE_PHASES(DdlaBackend::CPU, std::complex<double>)
#endif
#if DDLA_HAS_GPU
DDLA_INSTANTIATE_PHASES(DdlaBackend::GPU, float)
DDLA_INSTANTIATE_PHASES(DdlaBackend::GPU, double)
DDLA_INSTANTIATE_PHASES(DdlaBackend::GPU, std::complex<float>)
DDLA_INSTANTIATE_PHASES(DdlaBackend::GPU, std::complex<double>)
#endif
#undef DDLA_INSTANTIATE_PHASES
} // namespace ddla
