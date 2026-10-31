#include "blas_impl.h"
// ---------------------------------------------------------------------------
// Unified backend-templated matrix-vector product
// (y := alpha*blas_impl::op(A)*x + beta*y).
//
// Structure follows src/herk.cpp exactly: one definition covering CPU (Fortran
// BLAS) and GPU (cuBLAS/hipBLAS), selected with `if constexpr` on the Backend
// template parameter, with the arms themselves guarded by the BUILD CAPABILITY
// macros (DDLA_HAS_CPU / DDLA_HAS_GPU) and the vendor chosen per scalar arm by
// the PER-TU macros (DDLA_USE_CUDA / DDLA_USE_HIP).
//
// Why the capability guard matters here: the previous design put a `#else`
// CPU arm inside src/gemv.h selected by `!defined(DDLA_USE_CUDA) &&
// !defined(DDLA_USE_HIP)`.  In a dual build (DDLA_USE_CPU + DDLA_USE_CUDA)
// every TU defines DDLA_USE_CUDA, so that arm was never compiled -- while
// DDLA_HAS_CPU was 1 and the CPU instantiation of this function still existed.
// A CPU-backend call would therefore have reached cublasSgemv with HOST
// pointers.  Guarding on DDLA_HAS_CPU removes that class of bug.
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
#endif

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
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU gemv is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU gemv is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("gemv: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("gemv: template backend ") +
            (Backend == DdlaBackend::CPU ? "CPU" : "GPU") +
            " does not match handle backend " +
            (actual == DdlaBackend::CPU ? "CPU" : "GPU"));
    }

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
#if DDLA_HAS_CPU
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
#else
        (void)handle; (void)trans; (void)m; (void)n; (void)alpha; (void)A; (void)lda;
        (void)x; (void)incx; (void)beta; (void)y; (void)incy;
#endif
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

#if DDLA_HAS_CPU
INSTANTIATE_GEMV(DdlaBackend::CPU, float);
INSTANTIATE_GEMV(DdlaBackend::CPU, double);
INSTANTIATE_GEMV(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_GEMV(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_GEMV(DdlaBackend::GPU, float);
INSTANTIATE_GEMV(DdlaBackend::GPU, double);
INSTANTIATE_GEMV(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_GEMV(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_GEMV


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::gemv_check<BACKEND, TYPE>( const DdlaHandle_t& handle, char trans, int m, int n, const TYPE& alpha, const TYPE* A, int lda, const TYPE* x, int incx, const TYPE& beta, TYPE* y, int incy); \
    template void blas_impl::gemv_run<BACKEND, TYPE>( const DdlaHandle_t& handle, char trans, int m, int n, const TYPE& alpha, const TYPE* A, int lda, const TYPE* x, int incx, const TYPE& beta, TYPE* y, int incy);
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
