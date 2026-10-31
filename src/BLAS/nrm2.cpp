#include "blas_impl.h"
// ---------------------------------------------------------------------------
// Unified backend-templated Euclidean norm (result := ||x||_2).
//
// Structure follows src/herk.cpp exactly: one definition covering CPU (Fortran
// BLAS) and GPU (cuBLAS/hipBLAS), selected with `if constexpr` on the Backend
// template parameter, with the arms themselves guarded by the BUILD CAPABILITY
// macros (DDLA_HAS_CPU / DDLA_HAS_GPU) and the vendor chosen per scalar arm by
// the PER-TU macros (DDLA_USE_CUDA / DDLA_USE_HIP).
//
// Why the capability guard matters here: the previous design put a `#else`
// CPU arm inside src/nrm2.h selected by `!defined(DDLA_USE_CUDA) &&
// !defined(DDLA_USE_HIP)`.  In a dual build (DDLA_USE_CPU + DDLA_USE_CUDA)
// every TU defines DDLA_USE_CUDA, so that arm was never compiled -- while
// DDLA_HAS_CPU was 1 and the CPU instantiation of this function still existed.
// A CPU-backend call would therefore have reached cublasDnrm2 with HOST
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
//   * CPU handles: `result` is a HOST pointer (the Fortran ABI returns the
//     value, and the CPU arm stores it).
//
// The returned type is real even for complex `x`: xNRM2 / xCNRM2 / xZNRM2 all
// produce a real scalar, hence `typename real_type<T>::type*`.
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
// Real-valued function returns: safe ABI (unlike the complex xDOTC family,
// whose hidden-argument return convention is compiler-dependent).  The CPU arm
// stores the returned value through the caller's HOST result pointer.
float  snrm2_(const int*, const float*, const int*);
double dnrm2_(const int*, const double*, const int*);
float  scnrm2_(const int*, const std::complex<float>*, const int*);
double dznrm2_(const int*, const std::complex<double>*, const int*);
}
#endif

// ---------------------------------------------------------------------------
// GPU arm: the vendor calls.  Kept as free gpublasNrm2 overloads so the
// dispatch above stays a single BLAS_CHECK call per scalar type, exactly as
// ddla::gemm does it inline.  Private to this translation unit -- the public
// interface is the templated ddla::nrm2 declared in <ddla/ddla.h>.
//
// The pointer-mode scope is the FIRST statement of each vendor arm so that the
// handle is in DEVICE mode before the vendor routine dereferences `result`,
// and the destructor restores the caller's mode on the way out (including on
// the throwing path, since the scope is an automatic object).
// ---------------------------------------------------------------------------
#if DDLA_HAS_GPU

namespace ddla {

inline gpublasStatus_t gpublasNrm2(
    gpublasHandle_t handle, int n,
    const float* x, int incx,
    float* result
)
{
    #if DDLA_GPU_VENDOR
    detail::blas_device_pointer_mode_scope pm(handle);
    return gpublasSnrm2(handle, n, x, incx, result);
    #else
    // Neither vendor macro: this can only happen if the translation unit was
    // built without a GPU language while DDLA_HAS_GPU is set, which the build
    // system does not do (see the LANGUAGE CUDA/HIP source lists).  Throw
    // rather than fall through to a host pointer.
    throw std::runtime_error("gpublasNrm2: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasNrm2(
    gpublasHandle_t handle, int n,
    const double* x, int incx,
    double* result
)
{
    #if DDLA_GPU_VENDOR
    detail::blas_device_pointer_mode_scope pm(handle);
    return gpublasDnrm2(handle, n, x, incx, result);
    #else
    throw std::runtime_error("gpublasNrm2: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasNrm2(
    gpublasHandle_t handle, int n,
    const std::complex<float>* x, int incx,
    float* result
)
{
    #if DDLA_GPU_VENDOR
    detail::blas_device_pointer_mode_scope pm(handle);
    return gpublasScnrm2(handle, n, (gpublasComplex*)x, incx, result);
    #else
    throw std::runtime_error("gpublasNrm2: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasNrm2(
    gpublasHandle_t handle, int n,
    const std::complex<double>* x, int incx,
    double* result
)
{
    #if DDLA_GPU_VENDOR
    detail::blas_device_pointer_mode_scope pm(handle);
    return gpublasDznrm2(handle, n, (gpublasDoubleComplex*)x, incx, result);
    #else
    throw std::runtime_error("gpublasNrm2: no GPU vendor selected at compile time");
    #endif
}

} // namespace ddla

#endif // DDLA_HAS_GPU

namespace ddla {

namespace blas_impl {

template <DdlaBackend Backend, typename T>
void nrm2_check(
    const DdlaHandle_t& handle,
    int n,
    const T* x, int incx,
    typename real_type<T>::type* result)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU nrm2 is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU nrm2 is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("nrm2: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("nrm2: template backend ") +
            (Backend == DdlaBackend::CPU ? "CPU" : "GPU") +
            " does not match handle backend " +
            (actual == DdlaBackend::CPU ? "CPU" : "GPU"));
    }

}

template <DdlaBackend Backend, typename T>
void nrm2_run(
    const DdlaHandle_t& handle,
    int n,
    const T* x, int incx,
    typename real_type<T>::type* result)
{
    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        // HOST result pointer: the Fortran function returns the value and the
        // CPU arm performs the store.
        if constexpr (std::is_same_v<T, float>) {
            *result = snrm2_(&n, x, &incx);
        } else if constexpr (std::is_same_v<T, double>) {
            *result = dnrm2_(&n, x, &incx);
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            *result = scnrm2_(&n, x, &incx);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            *result = dznrm2_(&n, x, &incx);
        } else {
            static_assert(sizeof(T) == 0, "nrm2: unsupported scalar type T");
        }
#else
        (void)handle; (void)n; (void)x; (void)incx; (void)result;
#endif
    } else {
#if DDLA_HAS_GPU
        // DEVICE result pointer: the vendor arm forces DEVICE pointer mode.
        if constexpr (std::is_same_v<T, float> ||
                      std::is_same_v<T, double> ||
                      std::is_same_v<T, std::complex<float>> ||
                      std::is_same_v<T, std::complex<double>>) {
            BLAS_CHECK(gpublasNrm2(static_cast<gpublasHandle_t>(handle->blasH), n, x, incx, result));
        } else {
            static_assert(sizeof(T) == 0, "nrm2: unsupported scalar type T");
        }
#else
        (void)handle; (void)n; (void)x; (void)incx; (void)result;
#endif
    }
}

} // namespace blas_impl

template <DdlaBackend Backend, typename T>
void nrm2(
    const DdlaHandle_t& handle,
    int n,
    const T* x, int incx,
    typename real_type<T>::type* result)
{
    blas_impl::nrm2_check<Backend, T>(handle, n, x, incx, result);
    blas_impl::nrm2_run<Backend, T>(handle, n, x, incx, result);
}

#define INSTANTIATE_NRM2(BACKEND, TYPE)                                        \
    template void nrm2<BACKEND, TYPE>(                                         \
        const DdlaHandle_t&, int, const TYPE*, int,                            \
        typename real_type<TYPE>::type*)

#if DDLA_HAS_CPU
INSTANTIATE_NRM2(DdlaBackend::CPU, float);
INSTANTIATE_NRM2(DdlaBackend::CPU, double);
INSTANTIATE_NRM2(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_NRM2(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_NRM2(DdlaBackend::GPU, float);
INSTANTIATE_NRM2(DdlaBackend::GPU, double);
INSTANTIATE_NRM2(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_NRM2(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_NRM2


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::nrm2_check<BACKEND, TYPE>( const DdlaHandle_t& handle, int n, const TYPE* x, int incx, typename real_type<TYPE>::type* result); \
    template void blas_impl::nrm2_run<BACKEND, TYPE>( const DdlaHandle_t& handle, int n, const TYPE* x, int incx, typename real_type<TYPE>::type* result);
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
