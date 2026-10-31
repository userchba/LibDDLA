#include "blas_impl.h"
// ---------------------------------------------------------------------------
// Unified backend-templated Hermitian matrix-matrix product
//   C := alpha*A*B + beta*C    (side = 'L')   or
//   C := alpha*B*A + beta*C    (side = 'R').
//
// Structure follows src/gemm.cpp and src/herk.cpp exactly: one definition
// covering CPU (Fortran BLAS) and GPU (cuBLAS/hipBLAS), selected with
// `if constexpr` on the Backend template parameter, with the arms themselves
// guarded by the BUILD CAPABILITY macros (DDLA_HAS_CPU / DDLA_HAS_GPU) and the
// vendor chosen per scalar arm by the PER-TU macros (DDLA_USE_CUDA /
// DDLA_USE_HIP).
//
// Why the capability guard matters here: the previous design put a `#else`
// CPU arm inside src/hemm.h selected by `!defined(DDLA_USE_CUDA) &&
// !defined(DDLA_USE_HIP)`.  In a dual build (DDLA_USE_CPU + DDLA_USE_CUDA)
// every TU defines DDLA_USE_CUDA, so that arm was never compiled -- while
// DDLA_HAS_CPU was 1 and the CPU instantiation of this function still existed.
// A CPU-backend call would therefore have reached cublasChemm with HOST
// pointers.  Guarding on DDLA_HAS_CPU removes that class of bug.
//
// Scalar convention: xHEMM takes alpha and beta of the operands' own type T
// (both real for the real scalar types, complex for the complex ones), so the
// signature takes `const T&` for both.  For the real scalar types the routine
// is xSYMM; there is no transpose flag in this family.
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

#if DDLA_HAS_CPU
extern "C" {
// Real: xHEMM has no real-scalar BLAS counterpart, so the real types use xSYMM.
void ssymm_(const char*, const char*, const int*, const int*, const float*,
            const float*, const int*, const float*, const int*,
            const float*, float*, const int*);
void dsymm_(const char*, const char*, const int*, const int*, const double*,
            const double*, const int*, const double*, const int*,
            const double*, double*, const int*);
// Complex: xHEMM with complex alpha/beta.
void chemm_(const char*, const char*, const int*, const int*, const std::complex<float>*,
            const std::complex<float>*, const int*, const std::complex<float>*, const int*,
            const std::complex<float>*, std::complex<float>*, const int*);
void zhemm_(const char*, const char*, const int*, const int*, const std::complex<double>*,
            const std::complex<double>*, const int*, const std::complex<double>*, const int*,
            const std::complex<double>*, std::complex<double>*, const int*);
}
#endif

// ---------------------------------------------------------------------------
// GPU arm: the vendor calls.  Kept as free gpublasHemm overloads so the
// dispatch above stays a single BLAS_CHECK call per scalar type, exactly as
// ddla::gemm does it inline.  Private to this translation unit -- the public
// interface is the templated ddla::hemm declared in <ddla/ddla.h>.
// ---------------------------------------------------------------------------
#if DDLA_HAS_GPU

namespace ddla {

inline gpublasStatus_t gpublasHemm(
    gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo,
    int m, int n,
    const float& alpha,
    const float* A, int lda,
    const float* B, int ldb,
    const float& beta,
    float* C, int ldc
)
{
    // Real HEMM == SYMM.
    #if DDLA_GPU_VENDOR
    return gpublasSsymm(handle, side, uplo, m, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
    #else
    throw std::runtime_error("gpublasHemm: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasHemm(
    gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo,
    int m, int n,
    const double& alpha,
    const double* A, int lda,
    const double* B, int ldb,
    const double& beta,
    double* C, int ldc
)
{
    #if DDLA_GPU_VENDOR
    return gpublasDsymm(handle, side, uplo, m, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
    #else
    throw std::runtime_error("gpublasHemm: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasHemm(
    gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo,
    int m, int n,
    const std::complex<float>& alpha,
    const std::complex<float>* A, int lda,
    const std::complex<float>* B, int ldb,
    const std::complex<float>& beta,
    std::complex<float>* C, int ldc
)
{
    #if DDLA_GPU_VENDOR
    return gpublasChemm(handle, side, uplo, m, n,
        (gpublasComplex*)&alpha, (gpublasComplex*)A, lda,
        (gpublasComplex*)B, ldb, (gpublasComplex*)&beta, (gpublasComplex*)C, ldc);
    #else
    throw std::runtime_error("gpublasHemm: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasHemm(
    gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo,
    int m, int n,
    const std::complex<double>& alpha,
    const std::complex<double>* A, int lda,
    const std::complex<double>* B, int ldb,
    const std::complex<double>& beta,
    std::complex<double>* C, int ldc
)
{
    #if DDLA_GPU_VENDOR
    return gpublasZhemm(handle, side, uplo, m, n,
        (gpublasDoubleComplex*)&alpha, (gpublasDoubleComplex*)A, lda,
        (gpublasDoubleComplex*)B, ldb, (gpublasDoubleComplex*)&beta, (gpublasDoubleComplex*)C, ldc);
    #else
    throw std::runtime_error("gpublasHemm: no GPU vendor selected at compile time");
    #endif
}

} // namespace ddla

#endif // DDLA_HAS_GPU

namespace ddla {

namespace blas_impl {

template <DdlaBackend Backend, typename T>
void hemm_check(
    const DdlaHandle_t& handle,
    char side, char uplo,
    int m, int n,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const T& beta,
    T* C, int ldc)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU hemm is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU hemm is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("hemm: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("hemm: template backend ") +
            (Backend == DdlaBackend::CPU ? "CPU" : "GPU") +
            " does not match handle backend " +
            (actual == DdlaBackend::CPU ? "CPU" : "GPU"));
    }

    if constexpr (Backend == DdlaBackend::GPU) {
        (void)detail::to_side_mode(side);
        (void)detail::to_fill_mode(uplo);
    }
}

template <DdlaBackend Backend, typename T>
void hemm_run(
    const DdlaHandle_t& handle,
    char side, char uplo,
    int m, int n,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const T& beta,
    T* C, int ldc)
{
    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        // Real data: HEMM == SYMM; complex data calls xHEMM.  There is no
        // transpose flag in this family, so nothing needs normalising.
        if constexpr (std::is_same_v<T, float>) {
            const char s = side, u = uplo;
            ssymm_(&s, &u, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        } else if constexpr (std::is_same_v<T, double>) {
            const char s = side, u = uplo;
            dsymm_(&s, &u, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            const char s = side, u = uplo;
            chemm_(&s, &u, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            const char s = side, u = uplo;
            zhemm_(&s, &u, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        } else {
            static_assert(sizeof(T) == 0, "hemm: unsupported scalar type T");
        }
#else
        (void)handle; (void)side; (void)uplo; (void)m; (void)n;
        (void)alpha; (void)A; (void)lda; (void)B; (void)ldb;
        (void)beta; (void)C; (void)ldc;
#endif
    } else {
#if DDLA_HAS_GPU
        // The real types map onto xSYMM inside gpublasHemm.
        if constexpr (std::is_same_v<T, float> ||
                      std::is_same_v<T, double> ||
                      std::is_same_v<T, std::complex<float>> ||
                      std::is_same_v<T, std::complex<double>>) {
            BLAS_CHECK(gpublasHemm(static_cast<gpublasHandle_t>(handle->blasH),
                                   blas_impl::side(side),
                                   blas_impl::fill(uplo),
                                   m, n, alpha, A, lda, B, ldb, beta, C, ldc));
        } else {
            static_assert(sizeof(T) == 0, "hemm: unsupported scalar type T");
        }
#else
        (void)handle; (void)side; (void)uplo; (void)m; (void)n;
        (void)alpha; (void)A; (void)lda; (void)B; (void)ldb;
        (void)beta; (void)C; (void)ldc;
#endif
    }
}

} // namespace blas_impl

template <DdlaBackend Backend, typename T>
void hemm(
    const DdlaHandle_t& handle,
    char side, char uplo,
    int m, int n,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const T& beta,
    T* C, int ldc)
{
    blas_impl::hemm_check<Backend, T>(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    blas_impl::hemm_run<Backend, T>(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

#define INSTANTIATE_HEMM(BACKEND, TYPE)                                        \
    template void hemm<BACKEND, TYPE>(                                         \
        const DdlaHandle_t&, char, char, int, int,                             \
        const TYPE&, const TYPE*, int, const TYPE*, int,                       \
        const TYPE&, TYPE*, int)

#if DDLA_HAS_CPU
INSTANTIATE_HEMM(DdlaBackend::CPU, float);
INSTANTIATE_HEMM(DdlaBackend::CPU, double);
INSTANTIATE_HEMM(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_HEMM(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_HEMM(DdlaBackend::GPU, float);
INSTANTIATE_HEMM(DdlaBackend::GPU, double);
INSTANTIATE_HEMM(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_HEMM(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_HEMM


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::hemm_check<BACKEND, TYPE>( const DdlaHandle_t& handle, char side, char uplo, int m, int n, const TYPE& alpha, const TYPE* A, int lda, const TYPE* B, int ldb, const TYPE& beta, TYPE* C, int ldc); \
    template void blas_impl::hemm_run<BACKEND, TYPE>( const DdlaHandle_t& handle, char side, char uplo, int m, int n, const TYPE& alpha, const TYPE* A, int lda, const TYPE* B, int ldb, const TYPE& beta, TYPE* C, int ldc);
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
