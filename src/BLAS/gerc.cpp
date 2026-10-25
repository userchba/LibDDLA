// ---------------------------------------------------------------------------
// Unified backend-templated conjugate rank-1 update (A := alpha*x*y^H + A).
//
// This is the routine LAPACK's xLARF uses to apply a Householder reflector
// from the left:  A := A - tau * v * w^H.  The update touches the leading
// m x n submatrix of A only.
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
#include "ddla_stream_impl.h"

#if DDLA_HAS_CPU
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
#endif

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
template <DdlaBackend Backend, typename T>
void gerc(const DdlaHandle_t& handle, int m, int n,
          const T& alpha, const T* x, int incx,
          const T* y, int incy, T* A, int lda)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU gerc is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU gerc is not available in this LibDDLA build");

    if (handle == nullptr)
        throw std::runtime_error("gerc: null handle");

    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("gerc: template backend ") +
            (Backend == DdlaBackend::CPU ? "CPU" : "GPU") +
            " does not match handle backend " +
            (actual == DdlaBackend::CPU ? "CPU" : "GPU"));
    }

    if (m <= 0 || n <= 0) return;

    if constexpr (Backend == DdlaBackend::CPU) {
        #if DDLA_HAS_CPU
        if (incx == 0 || incy == 0)
            throw std::runtime_error("gerc: incx and incy must be nonzero");
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
        #endif
    } else {
        #if DDLA_HAS_GPU
        BLAS_CHECK(gpublasGerc(handle->blasH, m, n, alpha, x, incx, y, incy, A, lda));
        #endif
    }
}

#define INSTANTIATE_GERC(BACKEND, TYPE)                                      \
    template void gerc<BACKEND, TYPE>(                                       \
        const DdlaHandle_t&, int, int, const TYPE&, const TYPE*, int,        \
        const TYPE*, int, TYPE*, int)

#if DDLA_HAS_CPU
INSTANTIATE_GERC(DdlaBackend::CPU, float);
INSTANTIATE_GERC(DdlaBackend::CPU, double);
INSTANTIATE_GERC(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_GERC(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_GERC(DdlaBackend::GPU, float);
INSTANTIATE_GERC(DdlaBackend::GPU, double);
INSTANTIATE_GERC(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_GERC(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_GERC

} // namespace ddla
