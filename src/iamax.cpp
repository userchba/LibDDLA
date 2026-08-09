#include <ddla/iamax.h>

#include <complex>
#include <stdexcept>
#include <string>

#include "ddla_stream_impl.h"

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

} // anonymous namespace

template <DdlaBackend Backend, typename T>
void iamax(const DdlaHandle_t& handle, int n, const T* x, int incx, int& result)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU iamax is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU iamax is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("iamax: null handle");
    }
    const DdlaBackend actual = ddla_get_backend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("iamax: template backend ") + iamax_backend_name(Backend) +
            " does not match handle backend " + iamax_backend_name(actual));
    }
    if (n <= 0) {
        result = 0;
        return;
    }

    if (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        result = cpu_iamax(n, x, incx);
#endif
    } else {
#if DDLA_HAS_GPU
        BLAS_CHECK(deblasIamax(handle->blasH, n, x, incx, &result));
#endif
    }
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

} // namespace ddla
