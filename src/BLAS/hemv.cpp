#include "blas_impl.h"
// ---------------------------------------------------------------------------
// Unified backend-templated Hermitian matrix-vector product
//   y := alpha*A*x + beta*y.
//
// Structure follows src/gemm.cpp and src/herk.cpp exactly: one definition
// covering CPU (Fortran BLAS) and GPU (cuBLAS/hipBLAS), selected with
// `if constexpr` on the Backend template parameter, with the arms themselves
// guarded by the BUILD CAPABILITY macros (DDLA_HAS_CPU / DDLA_HAS_GPU) and the
// vendor chosen per scalar arm by the PER-TU macros (DDLA_USE_CUDA /
// DDLA_USE_HIP).
//
// Why the capability guard matters here: the previous design put a `#else`
// CPU arm inside src/hemv.h selected by `!defined(DDLA_USE_CUDA) &&
// !defined(DDLA_USE_HIP)`.  In a dual build (DDLA_USE_CPU + DDLA_USE_CUDA)
// every TU defines DDLA_USE_CUDA, so that arm was never compiled -- while
// DDLA_HAS_CPU was 1 and the CPU instantiation of this function still existed.
// A CPU-backend call would therefore have reached cublasChemv with HOST
// pointers.  Guarding on DDLA_HAS_CPU removes that class of bug.
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

#if DDLA_HAS_CPU
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
#endif

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
    #if DDLA_GPU_VENDOR
    return gpublasSsymv(handle, uplo, n, &alpha, A, lda, x, incx, &beta, y, incy);
    #else
    throw std::runtime_error("gpublasHemv: no GPU vendor selected at compile time");
    #endif
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
    #if DDLA_GPU_VENDOR
    return gpublasDsymv(handle, uplo, n, &alpha, A, lda, x, incx, &beta, y, incy);
    #else
    throw std::runtime_error("gpublasHemv: no GPU vendor selected at compile time");
    #endif
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
    #if DDLA_GPU_VENDOR
    return gpublasChemv(handle, uplo, n,
        (gpublasComplex*)&alpha, (gpublasComplex*)A, lda,
        (gpublasComplex*)x, incx, (gpublasComplex*)&beta, (gpublasComplex*)y, incy);
    #else
    throw std::runtime_error("gpublasHemv: no GPU vendor selected at compile time");
    #endif
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
    #if DDLA_GPU_VENDOR
    return gpublasZhemv(handle, uplo, n,
        (gpublasDoubleComplex*)&alpha, (gpublasDoubleComplex*)A, lda,
        (gpublasDoubleComplex*)x, incx, (gpublasDoubleComplex*)&beta, (gpublasDoubleComplex*)y, incy);
    #else
    throw std::runtime_error("gpublasHemv: no GPU vendor selected at compile time");
    #endif
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
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU hemv is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU hemv is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("hemv: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("hemv: template backend ") +
            (Backend == DdlaBackend::CPU ? "CPU" : "GPU") +
            " does not match handle backend " +
            (actual == DdlaBackend::CPU ? "CPU" : "GPU"));
    }

    if constexpr (Backend == DdlaBackend::GPU) {
        (void)detail::to_fill_mode(uplo);
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
#if DDLA_HAS_CPU
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
#else
        (void)handle; (void)uplo; (void)n; (void)alpha; (void)A; (void)lda;
        (void)x; (void)incx; (void)beta; (void)y; (void)incy;
#endif
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

#if DDLA_HAS_CPU
INSTANTIATE_HEMV(DdlaBackend::CPU, float);
INSTANTIATE_HEMV(DdlaBackend::CPU, double);
INSTANTIATE_HEMV(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_HEMV(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_HEMV(DdlaBackend::GPU, float);
INSTANTIATE_HEMV(DdlaBackend::GPU, double);
INSTANTIATE_HEMV(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_HEMV(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_HEMV


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::hemv_check<BACKEND, TYPE>( const DdlaHandle_t& handle, char uplo, int n, const TYPE& alpha, const TYPE* A, int lda, const TYPE* x, int incx, const TYPE& beta, TYPE* y, int incy); \
    template void blas_impl::hemv_run<BACKEND, TYPE>( const DdlaHandle_t& handle, char uplo, int n, const TYPE& alpha, const TYPE* A, int lda, const TYPE* x, int incx, const TYPE& beta, TYPE* y, int incy);
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
