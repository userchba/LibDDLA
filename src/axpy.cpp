#include <ddla/axpy.h>

#include <complex>
#include <stdexcept>
#include <string>

#include "ddla_stream_impl.h"

#if DDLA_HAS_CPU
extern "C" {
void saxpy_(const int*, const float*, const float*, const int*, float*, const int*);
void daxpy_(const int*, const double*, const double*, const int*, double*, const int*);
void caxpy_(const int*, const std::complex<float>*, const std::complex<float>*, const int*,
            std::complex<float>*, const int*);
void zaxpy_(const int*, const std::complex<double>*, const std::complex<double>*, const int*,
            std::complex<double>*, const int*);
}
#endif

namespace ddla {

namespace {

inline const char* axpy_backend_name(DdlaBackend backend)
{
    return backend == DdlaBackend::CPU ? "CPU" : "GPU";
}

#if DDLA_HAS_CPU
inline void cpu_axpy(int n, const float& alpha, const float* x, int incx,
                     float* y, int incy)
{
    saxpy_(&n, &alpha, x, &incx, y, &incy);
}

inline void cpu_axpy(int n, const double& alpha, const double* x, int incx,
                     double* y, int incy)
{
    daxpy_(&n, &alpha, x, &incx, y, &incy);
}

inline void cpu_axpy(int n, const std::complex<float>& alpha,
                     const std::complex<float>* x, int incx,
                     std::complex<float>* y, int incy)
{
    caxpy_(&n, &alpha, x, &incx, y, &incy);
}

inline void cpu_axpy(int n, const std::complex<double>& alpha,
                     const std::complex<double>* x, int incx,
                     std::complex<double>* y, int incy)
{
    zaxpy_(&n, &alpha, x, &incx, y, &incy);
}
#endif // DDLA_HAS_CPU

} // anonymous namespace

template <DdlaBackend Backend, typename T>
void axpy(const DdlaHandle_t& handle, int n, const T& alpha,
          const T* x, int incx, T* y, int incy)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU axpy is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU axpy is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("axpy: null handle");
    }
    const DdlaBackend actual = ddla_get_backend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("axpy: template backend ") + axpy_backend_name(Backend) +
            " does not match handle backend " + axpy_backend_name(actual));
    }
    if (n <= 0) return;

    if (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        cpu_axpy(n, alpha, x, incx, y, incy);
#endif
    } else {
#if DDLA_HAS_GPU
        BLAS_CHECK(deblasAxpy(handle->blasH, static_cast<int64_t>(n), alpha, x,
                              incx, y, incy));
#endif
    }
}

#define INSTANTIATE_AXPY(BACKEND, TYPE)                                      \
    template void axpy<BACKEND, TYPE>(                                       \
        const DdlaHandle_t&, int, const TYPE&, const TYPE*, int, TYPE*, int)

#if DDLA_HAS_CPU
INSTANTIATE_AXPY(DdlaBackend::CPU, float);
INSTANTIATE_AXPY(DdlaBackend::CPU, double);
INSTANTIATE_AXPY(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_AXPY(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_AXPY(DdlaBackend::GPU, float);
INSTANTIATE_AXPY(DdlaBackend::GPU, double);
INSTANTIATE_AXPY(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_AXPY(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_AXPY

} // namespace ddla
