#include "blas_impl.h"
// ---------------------------------------------------------------------------
// Unified backend-templated Euclidean norm (result := ||x||_2).
//
// Dispatch follows the Backend template parameter. CPU code is always compiled;
// DDLA_HAS_GPU enables GPU code, and DDLA_USE_CUDA / DDLA_USE_HIP select its vendor.
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
#include "instantiate.h"
#include "require_backend.h"

extern "C" {
// Real-valued function returns: safe ABI (unlike the complex xDOTC family,
// whose hidden-argument return convention is compiler-dependent).  The CPU arm
// stores the returned value through the caller's HOST result pointer.
float  snrm2_(const int*, const float*, const int*);
double dnrm2_(const int*, const double*, const int*);
float  scnrm2_(const int*, const std::complex<float>*, const int*);
double dznrm2_(const int*, const std::complex<double>*, const int*);
}

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
    detail::blas_device_pointer_mode_scope pm(handle);
    return gpublasSnrm2(handle, n, x, incx, result);
}

inline gpublasStatus_t gpublasNrm2(
    gpublasHandle_t handle, int n,
    const double* x, int incx,
    double* result
)
{
    detail::blas_device_pointer_mode_scope pm(handle);
    return gpublasDnrm2(handle, n, x, incx, result);
}

inline gpublasStatus_t gpublasNrm2(
    gpublasHandle_t handle, int n,
    const std::complex<float>* x, int incx,
    float* result
)
{
    detail::blas_device_pointer_mode_scope pm(handle);
    return gpublasScnrm2(handle, n, (gpublasComplex*)x, incx, result);
}

inline gpublasStatus_t gpublasNrm2(
    gpublasHandle_t handle, int n,
    const std::complex<double>* x, int incx,
    double* result
)
{
    detail::blas_device_pointer_mode_scope pm(handle);
    return gpublasDznrm2(handle, n, (gpublasDoubleComplex*)x, incx, result);
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
    detail::require_backend<Backend>(handle, "nrm2");

}

template <DdlaBackend Backend, typename T>
void nrm2_run(
    const DdlaHandle_t& handle,
    int n,
    const T* x, int incx,
    typename real_type<T>::type* result)
{
    if constexpr (Backend == DdlaBackend::CPU) {
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

DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_NRM2)

#undef INSTANTIATE_NRM2


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::nrm2_check<BACKEND, TYPE>( const DdlaHandle_t& handle, int n, const TYPE* x, int incx, typename real_type<TYPE>::type* result); \
    template void blas_impl::nrm2_run<BACKEND, TYPE>( const DdlaHandle_t& handle, int n, const TYPE* x, int incx, typename real_type<TYPE>::type* result)
DDLA_FOR_EACH_BACKEND_TYPE(DDLA_INSTANTIATE_PHASES)
#undef DDLA_INSTANTIATE_PHASES
} // namespace ddla
