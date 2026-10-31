#include "blas_impl.h"
// ---------------------------------------------------------------------------
// Unified backend-templated conjugate dot product (result := x^H * y).
//
// Structure follows src/herk.cpp exactly: one definition covering CPU (Fortran
// BLAS) and GPU (cuBLAS/hipBLAS), selected with `if constexpr` on the Backend
// template parameter, with the arms themselves guarded by the BUILD CAPABILITY
// macros (DDLA_HAS_CPU / DDLA_HAS_GPU) and the vendor chosen per scalar arm by
// the PER-TU macros (DDLA_USE_CUDA / DDLA_USE_HIP).
//
// Why the capability guard matters here: the previous design put a `#else`
// CPU arm inside src/dot.h selected by `!defined(DDLA_USE_CUDA) &&
// !defined(DDLA_USE_HIP)`.  In a dual build (DDLA_USE_CPU + DDLA_USE_CUDA)
// every TU defines DDLA_USE_CUDA, so that arm was never compiled -- while
// DDLA_HAS_CPU was 1 and the CPU instantiation of this function still existed.
// A CPU-backend call would therefore have reached cublasDdotc with HOST
// pointers.  Guarding on DDLA_HAS_CPU removes that class of bug.
//
// POINTER SEMANTICS -- the result is written THROUGH A POINTER, not returned
// by value, and the pointer's address space depends on the backend:
//   * GPU handles: `result` is a DEVICE pointer.  cuBLAS/hipBLAS default to
//     HOST pointer mode, in which the vendor routine dereferences the argument
//     as a host address; a non-host-mapped device buffer then faults.  Every
//     GPU overload therefore opens a detail::blas_device_pointer_mode_scope,
//     which switches the handle to DEVICE mode for the duration of the call
//     and restores the previous mode on scope exit.
//   * CPU handles: `result` is a HOST pointer.
//
// The file is named dot.cpp to mirror the private header it replaces
// (src/dot.h); the wrapper family itself is Dotc (conjugate in the first
// argument) and the templated entry point is ddla::dotc, so this TU defines
// gpublasDotc.  For the real scalar types xDOTC degenerates to the plain dot
// product.
//
// CPU complex results use the required CBLAS output-pointer interface, avoiding
// a compiler-dependent complex-valued Fortran function return. CMake checks
// that the selected CPU BLAS provides both complex DOT entry points.
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <complex>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "blas_enums.h"
#include "blas_pointer_mode.h"
#include "ddla_connector.h"
#include "ddla_stream.h"

#if DDLA_HAS_CPU
extern "C" {
float  sdot_(const int*, const float*, const int*, const float*, const int*);
double ddot_(const int*, const double*, const int*, const double*, const int*);
void cblas_cdotc_sub(int, const void*, int, const void*, int, void*);
void cblas_zdotc_sub(int, const void*, int, const void*, int, void*);
}
#endif

// ---------------------------------------------------------------------------
// GPU arm: the vendor calls.  Kept as free gpublasDotc overloads so the
// dispatch above stays a single BLAS_CHECK call per scalar type, exactly as
// ddla::gemm does it inline.  Private to this translation unit -- the public
// interface is the templated ddla::dotc declared in <ddla/ddla.h>.
//
// The complex overloads cast the const inputs away: the vendor prototypes take
// non-const operands even though xDOTC only reads them.  This matches the
// pre-unification header (src/dot.h) verbatim.
//
// The pointer-mode scope is the FIRST statement of each vendor arm so that the
// handle is in DEVICE mode before the vendor routine dereferences `result`,
// and the destructor restores the caller's mode on the way out (including on
// the throwing path, since the scope is an automatic object).
// ---------------------------------------------------------------------------
#if DDLA_HAS_GPU

namespace ddla {

inline gpublasStatus_t gpublasDotc(
    gpublasHandle_t handle, int n,
    const float* x, int incx,
    const float* y, int incy,
    float* result
)
{
    #if DDLA_GPU_VENDOR
    detail::blas_device_pointer_mode_scope pm(handle);
    return gpublasSdot(handle, n, x, incx, y, incy, result);
    #else
    // Neither vendor macro: this can only happen if the translation unit was
    // built without a GPU language while DDLA_HAS_GPU is set, which the build
    // system does not do (see the LANGUAGE CUDA/HIP source lists).  Throw
    // rather than fall through to a host pointer.
    throw std::runtime_error("gpublasDotc: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasDotc(
    gpublasHandle_t handle, int n,
    const double* x, int incx,
    const double* y, int incy,
    double* result
)
{
    #if DDLA_GPU_VENDOR
    detail::blas_device_pointer_mode_scope pm(handle);
    return gpublasDdot(handle, n, x, incx, y, incy, result);
    #else
    throw std::runtime_error("gpublasDotc: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasDotc(
    gpublasHandle_t handle, int n,
    const std::complex<float>* x, int incx,
    const std::complex<float>* y, int incy,
    std::complex<float>* result
)
{
    #if DDLA_GPU_VENDOR
    detail::blas_device_pointer_mode_scope pm(handle);
    return gpublasCdotc(handle, n, (gpublasComplex*)x, incx, (gpublasComplex*)y, incy,
                        (gpublasComplex*)result);
    #else
    throw std::runtime_error("gpublasDotc: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasDotc(
    gpublasHandle_t handle, int n,
    const std::complex<double>* x, int incx,
    const std::complex<double>* y, int incy,
    std::complex<double>* result
)
{
    #if DDLA_GPU_VENDOR
    detail::blas_device_pointer_mode_scope pm(handle);
    return gpublasZdotc(handle, n, (gpublasDoubleComplex*)x, incx, (gpublasDoubleComplex*)y, incy,
                        (gpublasDoubleComplex*)result);
    #else
    throw std::runtime_error("gpublasDotc: no GPU vendor selected at compile time");
    #endif
}

} // namespace ddla

#endif // DDLA_HAS_GPU

namespace ddla {

namespace blas_impl {

template <DdlaBackend Backend, typename T>
void dotc_check(
    const DdlaHandle_t& handle,
    int n,
    const T* x, int incx,
    const T* y, int incy,
    T* result)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU dotc is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU dotc is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("dotc: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("dotc: template backend ") +
            (Backend == DdlaBackend::CPU ? "CPU" : "GPU") +
            " does not match handle backend " +
            (actual == DdlaBackend::CPU ? "CPU" : "GPU"));
    }

}

template <DdlaBackend Backend, typename T>
void dotc_run(
    const DdlaHandle_t& handle,
    int n,
    const T* x, int incx,
    const T* y, int incy,
    T* result)
{
    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        if constexpr (std::is_same_v<T, float>) {
            *result = sdot_(&n, x, &incx, y, &incy);
        } else if constexpr (std::is_same_v<T, double>) {
            *result = ddot_(&n, x, &incx, y, &incy);
        } else if constexpr (std::is_same_v<T, std::complex<float>> ||
                             std::is_same_v<T, std::complex<double>>) {
            if constexpr (std::is_same_v<T, std::complex<float>>)
                cblas_cdotc_sub(n, x, incx, y, incy, result);
            else
                cblas_zdotc_sub(n, x, incx, y, incy, result);
        } else {
            static_assert(sizeof(T) == 0, "dotc: unsupported scalar type T");
        }
#else
        (void)handle; (void)n; (void)x; (void)incx; (void)y; (void)incy; (void)result;
#endif
    } else {
#if DDLA_HAS_GPU
        // DEVICE result pointer: the vendor arm forces DEVICE pointer mode.
        if constexpr (std::is_same_v<T, float> ||
                      std::is_same_v<T, double> ||
                      std::is_same_v<T, std::complex<float>> ||
                      std::is_same_v<T, std::complex<double>>) {
            BLAS_CHECK(gpublasDotc(static_cast<gpublasHandle_t>(handle->blasH), n, x, incx, y, incy, result));
        } else {
            static_assert(sizeof(T) == 0, "dotc: unsupported scalar type T");
        }
#else
        (void)handle; (void)n; (void)x; (void)incx; (void)y; (void)incy; (void)result;
#endif
    }
}

} // namespace blas_impl

template <DdlaBackend Backend, typename T>
void dotc(
    const DdlaHandle_t& handle,
    int n,
    const T* x, int incx,
    const T* y, int incy,
    T* result)
{
    blas_impl::dotc_check<Backend, T>(handle, n, x, incx, y, incy, result);
    blas_impl::dotc_run<Backend, T>(handle, n, x, incx, y, incy, result);
}

#define INSTANTIATE_DOTC(BACKEND, TYPE)                                        \
    template void dotc<BACKEND, TYPE>(                                         \
        const DdlaHandle_t&, int, const TYPE*, int, const TYPE*, int, TYPE*)

#if DDLA_HAS_CPU
INSTANTIATE_DOTC(DdlaBackend::CPU, float);
INSTANTIATE_DOTC(DdlaBackend::CPU, double);
INSTANTIATE_DOTC(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_DOTC(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_DOTC(DdlaBackend::GPU, float);
INSTANTIATE_DOTC(DdlaBackend::GPU, double);
INSTANTIATE_DOTC(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_DOTC(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_DOTC


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::dotc_check<BACKEND, TYPE>( const DdlaHandle_t& handle, int n, const TYPE* x, int incx, const TYPE* y, int incy, TYPE* result); \
    template void blas_impl::dotc_run<BACKEND, TYPE>( const DdlaHandle_t& handle, int n, const TYPE* x, int incx, const TYPE* y, int incy, TYPE* result);
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
