#include "blas_impl.h"
// ---------------------------------------------------------------------------
// Unified backend-templated Hermitian rank-2 update
//   A := alpha*x*y^H + conj(alpha)*y*x^H + A.
//
// Structure follows src/gemm.cpp and src/herk.cpp exactly: one definition
// covering CPU (Fortran BLAS) and GPU (cuBLAS/hipBLAS), selected with
// `if constexpr` on the Backend template parameter, with the arms themselves
// guarded by the BUILD CAPABILITY macros (DDLA_HAS_CPU / DDLA_HAS_GPU) and the
// vendor chosen per scalar arm by the PER-TU macros (DDLA_USE_CUDA /
// DDLA_USE_HIP).
//
// Why the capability guard matters here: the previous design put a `#else`
// CPU arm inside src/her2.h selected by `!defined(DDLA_USE_CUDA) &&
// !defined(DDLA_USE_HIP)`.  In a dual build (DDLA_USE_CPU + DDLA_USE_CUDA)
// every TU defines DDLA_USE_CUDA, so that arm was never compiled -- while
// DDLA_HAS_CPU was 1 and the CPU instantiation of this function still existed.
// A CPU-backend call would therefore have reached cublasCher2 with HOST
// pointers.  Guarding on DDLA_HAS_CPU removes that class of bug.
//
// Scalar convention: xHER2 has no beta and no transpose flag -- the update is
// applied in place to A, which is therefore the LAST argument (matching the
// xHER2/cuBLAS v2 argument order), with a COMPLEX alpha for the complex types.
// For the real scalar types the routine is xSYR2 and alpha is real; nothing
// needs normalising here because this family carries no transpose flag.
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
// Real: xHER2 has no real-scalar BLAS counterpart, so the real types use xSYR2.
void ssyr2_(const char*, const int*, const float*, const float*, const int*,
            const float*, const int*, float*, const int*);
void dsyr2_(const char*, const int*, const double*, const double*, const int*,
            const double*, const int*, double*, const int*);
// Complex: xHER2 with a complex alpha.
void cher2_(const char*, const int*, const std::complex<float>*,
            const std::complex<float>*, const int*, const std::complex<float>*, const int*,
            std::complex<float>*, const int*);
void zher2_(const char*, const int*, const std::complex<double>*,
            const std::complex<double>*, const int*, const std::complex<double>*, const int*,
            std::complex<double>*, const int*);
}
#endif

// ---------------------------------------------------------------------------
// GPU arm: the vendor calls.  Kept as free gpublasHer2 overloads so the
// dispatch above stays a single BLAS_CHECK call per scalar type, exactly as
// ddla::gemm does it inline.  Private to this translation unit -- the public
// interface is the templated ddla::her2 declared in <ddla/ddla.h>.
// ---------------------------------------------------------------------------
#if DDLA_HAS_GPU

namespace ddla {

inline gpublasStatus_t gpublasHer2(
    gpublasHandle_t handle, gpublasFillMode_t uplo, int n,
    const float& alpha,
    const float* x, int incx,
    const float* y, int incy,
    float* A, int lda
)
{
    // Real HER2 == SYR2.
    #if DDLA_GPU_VENDOR
    return gpublasSsyr2(handle, uplo, n, &alpha, x, incx, y, incy, A, lda);
    #else
    throw std::runtime_error("gpublasHer2: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasHer2(
    gpublasHandle_t handle, gpublasFillMode_t uplo, int n,
    const double& alpha,
    const double* x, int incx,
    const double* y, int incy,
    double* A, int lda
)
{
    #if DDLA_GPU_VENDOR
    return gpublasDsyr2(handle, uplo, n, &alpha, x, incx, y, incy, A, lda);
    #else
    throw std::runtime_error("gpublasHer2: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasHer2(
    gpublasHandle_t handle, gpublasFillMode_t uplo, int n,
    const std::complex<float>& alpha,
    const std::complex<float>* x, int incx,
    const std::complex<float>* y, int incy,
    std::complex<float>* A, int lda
)
{
    #if DDLA_GPU_VENDOR
    return gpublasCher2(handle, uplo, n, (gpublasComplex*)&alpha,
        (gpublasComplex*)x, incx, (gpublasComplex*)y, incy, (gpublasComplex*)A, lda);
    #else
    throw std::runtime_error("gpublasHer2: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasHer2(
    gpublasHandle_t handle, gpublasFillMode_t uplo, int n,
    const std::complex<double>& alpha,
    const std::complex<double>* x, int incx,
    const std::complex<double>* y, int incy,
    std::complex<double>* A, int lda
)
{
    #if DDLA_GPU_VENDOR
    return gpublasZher2(handle, uplo, n, (gpublasDoubleComplex*)&alpha,
        (gpublasDoubleComplex*)x, incx, (gpublasDoubleComplex*)y, incy, (gpublasDoubleComplex*)A, lda);
    #else
    throw std::runtime_error("gpublasHer2: no GPU vendor selected at compile time");
    #endif
}

} // namespace ddla

#endif // DDLA_HAS_GPU

namespace ddla {

namespace blas_impl {

template <DdlaBackend Backend, typename T>
void her2_check(
    const DdlaHandle_t& handle,
    char uplo,
    int n,
    const T& alpha,
    const T* x, int incx,
    const T* y, int incy,
    T* A, int lda)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU her2 is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU her2 is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("her2: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("her2: template backend ") +
            (Backend == DdlaBackend::CPU ? "CPU" : "GPU") +
            " does not match handle backend " +
            (actual == DdlaBackend::CPU ? "CPU" : "GPU"));
    }

    if constexpr (Backend == DdlaBackend::GPU) {
        (void)detail::to_fill_mode(uplo);
    }
}

template <DdlaBackend Backend, typename T>
void her2_run(
    const DdlaHandle_t& handle,
    char uplo,
    int n,
    const T& alpha,
    const T* x, int incx,
    const T* y, int incy,
    T* A, int lda)
{
    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        // Real data: HER2 == SYR2; complex data calls xHER2.  No transpose
        // flag exists in this family, so nothing needs normalising.
        if constexpr (std::is_same_v<T, float>) {
            const char u = uplo;
            ssyr2_(&u, &n, &alpha, x, &incx, y, &incy, A, &lda);
        } else if constexpr (std::is_same_v<T, double>) {
            const char u = uplo;
            dsyr2_(&u, &n, &alpha, x, &incx, y, &incy, A, &lda);
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            const char u = uplo;
            cher2_(&u, &n, &alpha, x, &incx, y, &incy, A, &lda);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            const char u = uplo;
            zher2_(&u, &n, &alpha, x, &incx, y, &incy, A, &lda);
        } else {
            static_assert(sizeof(T) == 0, "her2: unsupported scalar type T");
        }
#else
        (void)handle; (void)uplo; (void)n; (void)alpha;
        (void)x; (void)incx; (void)y; (void)incy; (void)A; (void)lda;
#endif
    } else {
#if DDLA_HAS_GPU
        // The real types map onto xSYR2 inside gpublasHer2.
        if constexpr (std::is_same_v<T, float> ||
                      std::is_same_v<T, double> ||
                      std::is_same_v<T, std::complex<float>> ||
                      std::is_same_v<T, std::complex<double>>) {
            BLAS_CHECK(gpublasHer2(static_cast<gpublasHandle_t>(handle->blasH),
                                   blas_impl::fill(uplo),
                                   n, alpha, x, incx, y, incy, A, lda));
        } else {
            static_assert(sizeof(T) == 0, "her2: unsupported scalar type T");
        }
#else
        (void)handle; (void)uplo; (void)n; (void)alpha;
        (void)x; (void)incx; (void)y; (void)incy; (void)A; (void)lda;
#endif
    }
}

} // namespace blas_impl

template <DdlaBackend Backend, typename T>
void her2(
    const DdlaHandle_t& handle,
    char uplo,
    int n,
    const T& alpha,
    const T* x, int incx,
    const T* y, int incy,
    T* A, int lda)
{
    blas_impl::her2_check<Backend, T>(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
    blas_impl::her2_run<Backend, T>(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
}

#define INSTANTIATE_HER2(BACKEND, TYPE)                                        \
    template void her2<BACKEND, TYPE>(                                         \
        const DdlaHandle_t&, char, int,                                        \
        const TYPE&, const TYPE*, int, const TYPE*, int,                       \
        TYPE*, int)

#if DDLA_HAS_CPU
INSTANTIATE_HER2(DdlaBackend::CPU, float);
INSTANTIATE_HER2(DdlaBackend::CPU, double);
INSTANTIATE_HER2(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_HER2(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_HER2(DdlaBackend::GPU, float);
INSTANTIATE_HER2(DdlaBackend::GPU, double);
INSTANTIATE_HER2(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_HER2(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_HER2


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::her2_check<BACKEND, TYPE>( const DdlaHandle_t& handle, char uplo, int n, const TYPE& alpha, const TYPE* x, int incx, const TYPE* y, int incy, TYPE* A, int lda); \
    template void blas_impl::her2_run<BACKEND, TYPE>( const DdlaHandle_t& handle, char uplo, int n, const TYPE& alpha, const TYPE* x, int incx, const TYPE* y, int incy, TYPE* A, int lda);
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
