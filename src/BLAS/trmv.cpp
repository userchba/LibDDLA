#include "blas_impl.h"
// ---------------------------------------------------------------------------
// Unified backend-templated triangular matrix-vector product (x := blas_impl::op(A)*x,
// in place).
//
// Structure follows src/herk.cpp exactly: one definition covering CPU (Fortran
// BLAS) and GPU (cuBLAS/hipBLAS), selected with `if constexpr` on the Backend
// template parameter, with the arms themselves guarded by the BUILD CAPABILITY
// macros (DDLA_HAS_CPU / DDLA_HAS_GPU) and the vendor chosen per scalar arm by
// the PER-TU macros (DDLA_USE_CUDA / DDLA_USE_HIP).
//
// Why the capability guard matters here: the previous design put a `#else`
// CPU arm inside src/trmv.h selected by `!defined(DDLA_USE_CUDA) &&
// !defined(DDLA_USE_HIP)`.  In a dual build (DDLA_USE_CPU + DDLA_USE_CUDA)
// every TU defines DDLA_USE_CUDA, so that arm was never compiled -- while
// DDLA_HAS_CPU was 1 and the CPU instantiation of this function still existed.
// A CPU-backend call would therefore have reached cublasStrmv with HOST
// pointers.  Guarding on DDLA_HAS_CPU removes that class of bug.
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

#if DDLA_HAS_CPU
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
#endif

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
    #if DDLA_GPU_VENDOR
    return gpublasStrmv(handle, uplo, trans, diag, n, A, lda, x, incx);
    #else
    // Neither vendor macro: this can only happen if the translation unit was
    // built without a GPU language while DDLA_HAS_GPU is set, which the build
    // system does not do (see the LANGUAGE CUDA/HIP source lists).  Throw
    // rather than fall through to a host pointer.
    throw std::runtime_error("gpublasTrmv: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasTrmv(
    gpublasHandle_t handle, gpublasFillMode_t uplo,
    gpublasOperation_t trans, gpublasDiagType_t diag,
    int n,
    const double* A, int lda,
    double* x, int incx
)
{
    #if DDLA_GPU_VENDOR
    return gpublasDtrmv(handle, uplo, trans, diag, n, A, lda, x, incx);
    #else
    throw std::runtime_error("gpublasTrmv: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasTrmv(
    gpublasHandle_t handle, gpublasFillMode_t uplo,
    gpublasOperation_t trans, gpublasDiagType_t diag,
    int n,
    const std::complex<float>* A, int lda,
    std::complex<float>* x, int incx
)
{
    #if DDLA_GPU_VENDOR
    return gpublasCtrmv(handle, uplo, trans, diag, n,
        (gpublasComplex*)A, lda, (gpublasComplex*)x, incx);
    #else
    throw std::runtime_error("gpublasTrmv: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasTrmv(
    gpublasHandle_t handle, gpublasFillMode_t uplo,
    gpublasOperation_t trans, gpublasDiagType_t diag,
    int n,
    const std::complex<double>* A, int lda,
    std::complex<double>* x, int incx
)
{
    #if DDLA_GPU_VENDOR
    return gpublasZtrmv(handle, uplo, trans, diag, n,
        (gpublasDoubleComplex*)A, lda, (gpublasDoubleComplex*)x, incx);
    #else
    throw std::runtime_error("gpublasTrmv: no GPU vendor selected at compile time");
    #endif
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
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU trmv is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU trmv is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("trmv: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("trmv: template backend ") +
            (Backend == DdlaBackend::CPU ? "CPU" : "GPU") +
            " does not match handle backend " +
            (actual == DdlaBackend::CPU ? "CPU" : "GPU"));
    }

    if constexpr (Backend == DdlaBackend::GPU) {
        (void)detail::to_fill_mode(uplo);
        (void)detail::to_blas_op(trans);
        (void)detail::to_diag_type(diag);
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
#if DDLA_HAS_CPU
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
#else
        (void)handle; (void)uplo; (void)trans; (void)diag; (void)n;
        (void)A; (void)lda; (void)x; (void)incx;
#endif
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

#if DDLA_HAS_CPU
INSTANTIATE_TRMV(DdlaBackend::CPU, float);
INSTANTIATE_TRMV(DdlaBackend::CPU, double);
INSTANTIATE_TRMV(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_TRMV(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_TRMV(DdlaBackend::GPU, float);
INSTANTIATE_TRMV(DdlaBackend::GPU, double);
INSTANTIATE_TRMV(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_TRMV(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_TRMV


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::trmv_check<BACKEND, TYPE>( const DdlaHandle_t& handle, char uplo, char trans, char diag, int n, const TYPE* A, int lda, TYPE* x, int incx); \
    template void blas_impl::trmv_run<BACKEND, TYPE>( const DdlaHandle_t& handle, char uplo, char trans, char diag, int n, const TYPE* A, int lda, TYPE* x, int incx);
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
