#include "blas_impl.h"
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <complex>
#include <stdexcept>
#include <string>

#include "ddla_connector.h"
#include "ddla_stream.h"
#include "instantiate.h"
#include "require_backend.h"

extern "C" {
int isamax_(const int*, const float*, const int*);
int idamax_(const int*, const double*, const int*);
int icamax_(const int*, const std::complex<float>*, const int*);
int izamax_(const int*, const std::complex<double>*, const int*);
}

namespace ddla {

namespace {

inline int cpu_iamax(int n, const float* x, int incx)
{
    return isamax_(&n, x, &incx);
}

inline int cpu_iamax(int n, const double* x, int incx)
{
    return idamax_(&n, x, &incx);
}

inline int cpu_iamax(int n, const std::complex<float>* x, int incx)
{
    return icamax_(&n, x, &incx);
}

inline int cpu_iamax(int n, const std::complex<double>* x, int incx)
{
    return izamax_(&n, x, &incx);
}

#if DDLA_HAS_GPU
// GPU arm: the vendor calls, one overload per scalar type (cuBLAS / hipBLAS iamax).  Private to this translation unit --
// the public interface is the templated ddla::iamax declared in <ddla/ddla.h>.
inline gpublasStatus_t gpublasIamax(gpublasHandle_t handle, int n, const float *x, int incx, int *result)
{
#if DDLA_GPU_VENDOR
    return gpublasIsamax(handle, n, x, incx, result);
#else
    throw std::runtime_error("iamax: no GPU vendor selected at compile time");
#endif
}

inline gpublasStatus_t gpublasIamax(gpublasHandle_t handle, int n, const double *x, int incx, int *result)
{
#if DDLA_GPU_VENDOR
    return gpublasIdamax(handle, n, x, incx, result);
#else
    throw std::runtime_error("iamax: no GPU vendor selected at compile time");
#endif
}

inline gpublasStatus_t gpublasIamax(gpublasHandle_t handle, int n, const std::complex<float> *x, int incx, int *result)
{
#if DDLA_GPU_VENDOR
    return gpublasIcamax(handle, n, (gpublasComplex*)x, incx, result);
#else
    throw std::runtime_error("iamax: no GPU vendor selected at compile time");
#endif
}

inline gpublasStatus_t gpublasIamax(gpublasHandle_t handle, int n, const std::complex<double> *x, int incx, int *result)
{
#if DDLA_GPU_VENDOR
    return gpublasIzamax(handle, n, (gpublasDoubleComplex*)x, incx, result);
#else
    throw std::runtime_error("iamax: no GPU vendor selected at compile time");
#endif
}
#endif // DDLA_HAS_GPU

} // anonymous namespace

namespace blas_impl {

template <DdlaBackend Backend, typename T>
void iamax_check(const DdlaHandle_t& handle, int n, const T* x, int incx, int& result)
{
    detail::require_backend<Backend>(handle, "iamax");
    if (n <= 0) {
        return;
    }

}

template <DdlaBackend Backend, typename T>
void iamax_run(const DdlaHandle_t& handle, int n, const T* x, int incx, int& result)
{
    if (n <= 0) { result = 0; return; }
    if constexpr (Backend == DdlaBackend::CPU) {
        result = cpu_iamax(n, x, incx);
    } else {
#if DDLA_HAS_GPU
        BLAS_CHECK(gpublasIamax(static_cast<gpublasHandle_t>(handle->blasH), n, x, incx, &result));
#endif
    }
}

} // namespace blas_impl

template <DdlaBackend Backend, typename T>
void iamax(const DdlaHandle_t& handle, int n, const T* x, int incx, int& result)
{
    blas_impl::iamax_check<Backend, T>(handle, n, x, incx, result);
    blas_impl::iamax_run<Backend, T>(handle, n, x, incx, result);
}

#define INSTANTIATE_IAMAX(BACKEND, TYPE)                                     \
    template void iamax<BACKEND, TYPE>(                                      \
        const DdlaHandle_t&, int, const TYPE*, int, int&)

DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_IAMAX)

#undef INSTANTIATE_IAMAX


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::iamax_check<BACKEND, TYPE>(const DdlaHandle_t& handle, int n, const TYPE* x, int incx, int& result); \
    template void blas_impl::iamax_run<BACKEND, TYPE>(const DdlaHandle_t& handle, int n, const TYPE* x, int incx, int& result)
DDLA_FOR_EACH_BACKEND_TYPE(DDLA_INSTANTIATE_PHASES)
#undef DDLA_INSTANTIATE_PHASES
} // namespace ddla
