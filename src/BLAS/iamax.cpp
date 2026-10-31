#include "blas_impl.h"
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <complex>
#include <stdexcept>
#include <string>

#include "ddla_connector.h"
#include "ddla_stream.h"

#if DDLA_HAS_CPU
extern "C" {
int isamax_(const int*, const float*, const int*);
int idamax_(const int*, const double*, const int*);
int icamax_(const int*, const std::complex<float>*, const int*);
int izamax_(const int*, const std::complex<double>*, const int*);
}
#endif

namespace ddla {

namespace {

inline const char* iamax_backend_name(DdlaBackend backend)
{
    return backend == DdlaBackend::CPU ? "CPU" : "GPU";
}

#if DDLA_HAS_CPU
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
#endif // DDLA_HAS_CPU

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
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU iamax is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU iamax is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("iamax: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("iamax: template backend ") + iamax_backend_name(Backend) +
            " does not match handle backend " + iamax_backend_name(actual));
    }
    if (n <= 0) {
        return;
    }

}

template <DdlaBackend Backend, typename T>
void iamax_run(const DdlaHandle_t& handle, int n, const T* x, int incx, int& result)
{
    if (n <= 0) { result = 0; return; }
    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        result = cpu_iamax(n, x, incx);
#endif
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

#if DDLA_HAS_CPU
INSTANTIATE_IAMAX(DdlaBackend::CPU, float);
INSTANTIATE_IAMAX(DdlaBackend::CPU, double);
INSTANTIATE_IAMAX(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_IAMAX(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_IAMAX(DdlaBackend::GPU, float);
INSTANTIATE_IAMAX(DdlaBackend::GPU, double);
INSTANTIATE_IAMAX(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_IAMAX(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_IAMAX


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::iamax_check<BACKEND, TYPE>(const DdlaHandle_t& handle, int n, const TYPE* x, int incx, int& result); \
    template void blas_impl::iamax_run<BACKEND, TYPE>(const DdlaHandle_t& handle, int n, const TYPE* x, int incx, int& result);
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
