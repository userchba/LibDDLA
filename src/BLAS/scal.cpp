#include "blas_impl.h"
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <complex>
#include <stdexcept>
#include <string>

#include "ddla_connector.h"
#include "ddla_stream.h"

#if DDLA_HAS_CPU
// CPU BLAS Level-1 SCAL bindings (Fortran ABI), declared by hand rather than
// via a vendor header -- same convention as gemm.cpp's sgemm_/dgemm_/cgemm_/
// zgemm_ block.
extern "C" {
void sscal_(const int*, const float*, float*, const int*);
void dscal_(const int*, const double*, double*, const int*);
void cscal_(const int*, const std::complex<float>*, std::complex<float>*, const int*);
void zscal_(const int*, const std::complex<double>*, std::complex<double>*, const int*);
}
#endif

namespace ddla {

#if DDLA_HAS_CPU
inline void cpu_scal(int n, const float& alpha, float* x, int incx)
{
    sscal_(&n, &alpha, x, &incx);
}

inline void cpu_scal(int n, const double& alpha, double* x, int incx)
{
    dscal_(&n, &alpha, x, &incx);
}

inline void cpu_scal(int n, const std::complex<float>& alpha, std::complex<float>* x, int incx)
{
    cscal_(&n, &alpha, x, &incx);
}

inline void cpu_scal(int n, const std::complex<double>& alpha, std::complex<double>* x, int incx)
{
    zscal_(&n, &alpha, x, &incx);
}
#endif

#if DDLA_HAS_GPU
namespace {

// GPU arm: the vendor calls, one overload per scalar type (cuBLAS / hipBLAS scal).  Private to this translation unit --
// the public interface is the templated ddla::scal declared in <ddla/ddla.h>.
inline gpublasStatus_t gpublasScal(gpublasHandle_t handle, int64_t n, const float& alpha, float *x, int64_t incx)
{
#if DDLA_GPU_VENDOR
    return gpublasSscal(handle, n, &alpha, x, incx);
#else
    throw std::runtime_error("scal: no GPU vendor selected at compile time");
#endif
}

inline gpublasStatus_t gpublasScal(gpublasHandle_t handle, int64_t n, const double& alpha, double *x, int64_t incx)
{
#if DDLA_GPU_VENDOR
    return gpublasDscal(handle, n, &alpha, x, incx);
#else
    throw std::runtime_error("scal: no GPU vendor selected at compile time");
#endif
}

inline gpublasStatus_t gpublasScal(gpublasHandle_t handle, int64_t n, const std::complex<float>& alpha, std::complex<float> *x, int64_t incx)
{
#if DDLA_GPU_VENDOR
    return gpublasCscal(handle, n, (gpublasComplex*)&alpha, (gpublasComplex*)x, incx);
#else
    throw std::runtime_error("scal: no GPU vendor selected at compile time");
#endif
}

inline gpublasStatus_t gpublasScal(gpublasHandle_t handle, int64_t n, const std::complex<double>& alpha, std::complex<double> *x, int64_t incx)
{
#if DDLA_GPU_VENDOR
    return gpublasZscal(handle, n, (gpublasDoubleComplex*)&alpha, (gpublasDoubleComplex*)x, incx);
#else
    throw std::runtime_error("scal: no GPU vendor selected at compile time");
#endif
}

} // anonymous namespace
#endif // DDLA_HAS_GPU

inline const char* scal_backend_name(DdlaBackend backend)
{
    return backend == DdlaBackend::CPU ? "CPU" : "GPU";
}

namespace blas_impl {

template <DdlaBackend Backend, typename T>
void scal_check(const DdlaHandle_t& handle, int n, const T& alpha, T* x, int incx)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU scal is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU scal is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("scal: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("scal: template backend ") + scal_backend_name(Backend) +
            " does not match handle backend " + scal_backend_name(actual));
    }
    if (n <= 0) return;

}

template <DdlaBackend Backend, typename T>
void scal_run(const DdlaHandle_t& handle, int n, const T& alpha, T* x, int incx)
{
    if (n <= 0) return;
    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        cpu_scal(n, alpha, x, incx);
#endif
    } else {
#if DDLA_HAS_GPU
        BLAS_CHECK(gpublasScal(static_cast<gpublasHandle_t>(handle->blasH), static_cast<int64_t>(n), alpha, x,
                               static_cast<int64_t>(incx)));
#endif
    }
}

} // namespace blas_impl

template <DdlaBackend Backend, typename T>
void scal(const DdlaHandle_t& handle, int n, const T& alpha, T* x, int incx)
{
    blas_impl::scal_check<Backend, T>(handle, n, alpha, x, incx);
    blas_impl::scal_run<Backend, T>(handle, n, alpha, x, incx);
}

#define INSTANTIATE_SCAL(BACKEND, TYPE)                                      \
    template void scal<BACKEND, TYPE>(                                       \
        const DdlaHandle_t&, int, const TYPE&, TYPE*, int)

#if DDLA_HAS_CPU
INSTANTIATE_SCAL(DdlaBackend::CPU, float);
INSTANTIATE_SCAL(DdlaBackend::CPU, double);
INSTANTIATE_SCAL(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_SCAL(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_SCAL(DdlaBackend::GPU, float);
INSTANTIATE_SCAL(DdlaBackend::GPU, double);
INSTANTIATE_SCAL(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_SCAL(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_SCAL


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::scal_check<BACKEND, TYPE>(const DdlaHandle_t& handle, int n, const TYPE& alpha, TYPE* x, int incx); \
    template void blas_impl::scal_run<BACKEND, TYPE>(const DdlaHandle_t& handle, int n, const TYPE& alpha, TYPE* x, int incx);
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
