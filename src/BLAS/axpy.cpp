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
void saxpy_(const int*, const float*, const float*, const int*, float*, const int*);
void daxpy_(const int*, const double*, const double*, const int*, double*, const int*);
void caxpy_(const int*, const std::complex<float>*, const std::complex<float>*, const int*,
            std::complex<float>*, const int*);
void zaxpy_(const int*, const std::complex<double>*, const std::complex<double>*, const int*,
            std::complex<double>*, const int*);
}

namespace ddla {

namespace {

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

#if DDLA_HAS_GPU
// GPU arm: the vendor calls, one overload per scalar type (cuBLAS / hipBLAS axpy).  Private to this translation unit --
// the public interface is the templated ddla::axpy declared in <ddla/ddla.h>.
inline gpublasStatus_t gpublasAxpy(gpublasHandle_t handle, const int64_t& n, const float& alpha, const float *x, int incx, float *y, int incy)
{
#if DDLA_GPU_VENDOR
    return gpublasSaxpy(handle, n, &alpha, x, incx, y, incy);
#else
    throw std::runtime_error("axpy: no GPU vendor selected at compile time");
#endif
}

inline gpublasStatus_t gpublasAxpy(gpublasHandle_t handle, const int64_t& n, const double& alpha, const double *x, int incx, double *y, int incy)
{
#if DDLA_GPU_VENDOR
    return gpublasDaxpy(handle, n, &alpha, x, incx, y, incy);
#else
    throw std::runtime_error("axpy: no GPU vendor selected at compile time");
#endif
}

inline gpublasStatus_t gpublasAxpy(gpublasHandle_t handle, const int64_t& n, const std::complex<float>& alpha, const std::complex<float> *x, int incx, std::complex<float> *y, int incy)
{
#if DDLA_GPU_VENDOR
    return gpublasCaxpy(handle, n, (gpublasComplex*)&alpha, (gpublasComplex*)x, incx, (gpublasComplex*)y, incy);
#else
    throw std::runtime_error("axpy: no GPU vendor selected at compile time");
#endif
}

inline gpublasStatus_t gpublasAxpy(gpublasHandle_t handle, const int64_t& n, const std::complex<double>& alpha, const std::complex<double> *x, int incx, std::complex<double> *y, int incy)
{
#if DDLA_GPU_VENDOR
    return gpublasZaxpy(handle, n, (gpublasDoubleComplex*)&alpha, (gpublasDoubleComplex*)x, incx, (gpublasDoubleComplex*)y, incy);
#else
    throw std::runtime_error("axpy: no GPU vendor selected at compile time");
#endif
}
#endif // DDLA_HAS_GPU

} // anonymous namespace

namespace blas_impl {

template <DdlaBackend Backend, typename T>
void axpy_check(const DdlaHandle_t& handle, int n, const T& alpha,
          const T* x, int incx, T* y, int incy)
{
    detail::require_backend<Backend>(handle, "axpy");
    if (n <= 0) return;

}

template <DdlaBackend Backend, typename T>
void axpy_run(const DdlaHandle_t& handle, int n, const T& alpha,
          const T* x, int incx, T* y, int incy)
{
    if (n <= 0) return;
    if constexpr (Backend == DdlaBackend::CPU) {
        cpu_axpy(n, alpha, x, incx, y, incy);
    } else {
#if DDLA_HAS_GPU
        BLAS_CHECK(gpublasAxpy(static_cast<gpublasHandle_t>(handle->blasH), static_cast<int64_t>(n), alpha, x,
                               incx, y, incy));
#endif
    }
}

} // namespace blas_impl

template <DdlaBackend Backend, typename T>
void axpy(const DdlaHandle_t& handle, int n, const T& alpha,
          const T* x, int incx, T* y, int incy)
{
    blas_impl::axpy_check<Backend, T>(handle, n, alpha, x, incx, y, incy);
    blas_impl::axpy_run<Backend, T>(handle, n, alpha, x, incx, y, incy);
}

#define INSTANTIATE_AXPY(BACKEND, TYPE)                                      \
    template void axpy<BACKEND, TYPE>(                                       \
        const DdlaHandle_t&, int, const TYPE&, const TYPE*, int, TYPE*, int)

DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_AXPY)

#undef INSTANTIATE_AXPY


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::axpy_check<BACKEND, TYPE>(const DdlaHandle_t& handle, int n, const TYPE& alpha, const TYPE* x, int incx, TYPE* y, int incy); \
    template void blas_impl::axpy_run<BACKEND, TYPE>(const DdlaHandle_t& handle, int n, const TYPE& alpha, const TYPE* x, int incx, TYPE* y, int incy)
DDLA_FOR_EACH_BACKEND_TYPE(DDLA_INSTANTIATE_PHASES)
#undef DDLA_INSTANTIATE_PHASES
} // namespace ddla
