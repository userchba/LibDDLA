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
void sger_(const int*, const int*, const float*, const float*, const int*,
           const float*, const int*, float*, const int*);
void dger_(const int*, const int*, const double*, const double*, const int*,
           const double*, const int*, double*, const int*);
void cgeru_(const int*, const int*, const std::complex<float>*,
            const std::complex<float>*, const int*,
            const std::complex<float>*, const int*,
            std::complex<float>*, const int*);
void zgeru_(const int*, const int*, const std::complex<double>*,
            const std::complex<double>*, const int*,
            const std::complex<double>*, const int*,
            std::complex<double>*, const int*);
}
#endif

namespace ddla {

namespace {

inline const char* geru_backend_name(DdlaBackend backend)
{
    return backend == DdlaBackend::CPU ? "CPU" : "GPU";
}

#if DDLA_HAS_CPU
inline void cpu_geru(int m, int n, const float& alpha,
                     const float* x, int incx, const float* y, int incy,
                     float* A, int lda)
{
    sger_(&m, &n, &alpha, x, &incx, y, &incy, A, &lda);
}

inline void cpu_geru(int m, int n, const double& alpha,
                     const double* x, int incx, const double* y, int incy,
                     double* A, int lda)
{
    dger_(&m, &n, &alpha, x, &incx, y, &incy, A, &lda);
}

inline void cpu_geru(int m, int n, const std::complex<float>& alpha,
                     const std::complex<float>* x, int incx,
                     const std::complex<float>* y, int incy,
                     std::complex<float>* A, int lda)
{
    cgeru_(&m, &n, &alpha, x, &incx, y, &incy, A, &lda);
}

inline void cpu_geru(int m, int n, const std::complex<double>& alpha,
                     const std::complex<double>* x, int incx,
                     const std::complex<double>* y, int incy,
                     std::complex<double>* A, int lda)
{
    zgeru_(&m, &n, &alpha, x, &incx, y, &incy, A, &lda);
}
#endif // DDLA_HAS_CPU

#if DDLA_HAS_GPU
// GPU arm: the vendor calls, one overload per scalar type (cuBLAS / hipBLAS ger / geru).  Private to this translation
// unit -- the public interface is the templated ddla::geru declared in <ddla/ddla.h>.
inline gpublasStatus_t gpublasGeru(gpublasHandle_t handle, int m, int n, const float& alpha, const float *x, int incx, const float *y, int incy, float *A, int lda)
{
#if DDLA_GPU_VENDOR
    return gpublasSger(handle, m, n, &alpha, x, incx, y, incy, A, lda);
#else
    throw std::runtime_error("geru: no GPU vendor selected at compile time");
#endif
}

inline gpublasStatus_t gpublasGeru(gpublasHandle_t handle, int m, int n, const double& alpha, const double *x, int incx, const double *y, int incy, double *A, int lda)
{
#if DDLA_GPU_VENDOR
    return gpublasDger(handle, m, n, &alpha, x, incx, y, incy, A, lda);
#else
    throw std::runtime_error("geru: no GPU vendor selected at compile time");
#endif
}

inline gpublasStatus_t gpublasGeru(gpublasHandle_t handle, int m, int n, const std::complex<float>& alpha, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy, std::complex<float> *A, int lda)
{
#if DDLA_GPU_VENDOR
    return gpublasCgeru(handle, m, n, (gpublasComplex*)&alpha, (gpublasComplex*)x, incx, (gpublasComplex*)y, incy, (gpublasComplex*)A, lda);
#else
    throw std::runtime_error("geru: no GPU vendor selected at compile time");
#endif
}

inline gpublasStatus_t gpublasGeru(gpublasHandle_t handle, int m, int n, const std::complex<double>& alpha, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy, std::complex<double> *A, int lda)
{
#if DDLA_GPU_VENDOR
    return gpublasZgeru(handle, m, n, (gpublasDoubleComplex*)&alpha, (gpublasDoubleComplex*)x, incx, (gpublasDoubleComplex*)y, incy, (gpublasDoubleComplex*)A, lda);
#else
    throw std::runtime_error("geru: no GPU vendor selected at compile time");
#endif
}
#endif // DDLA_HAS_GPU

} // anonymous namespace

namespace blas_impl {

template <DdlaBackend Backend, typename T>
void geru_check(const DdlaHandle_t& handle, int m, int n, const T& alpha,
          const T* x, int incx, const T* y, int incy, T* A, int lda)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU geru is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU geru is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("geru: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("geru: template backend ") + geru_backend_name(Backend) +
            " does not match handle backend " + geru_backend_name(actual));
    }
    if (m <= 0 || n <= 0) return;

    if constexpr (Backend == DdlaBackend::CPU) {
        if (incx == 0 || incy == 0) throw std::runtime_error("geru: incx and incy must be nonzero");
    }
}

template <DdlaBackend Backend, typename T>
void geru_run(const DdlaHandle_t& handle, int m, int n, const T& alpha,
          const T* x, int incx, const T* y, int incy, T* A, int lda)
{
    if (m <= 0 || n <= 0) return;
    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        cpu_geru(m, n, alpha, x, incx, y, incy, A, lda);
#endif
    } else {
#if DDLA_HAS_GPU
        BLAS_CHECK(gpublasGeru(static_cast<gpublasHandle_t>(handle->blasH), m, n, alpha, x, incx, y, incy, A, lda));
#endif
    }
}

} // namespace blas_impl

template <DdlaBackend Backend, typename T>
void geru(const DdlaHandle_t& handle, int m, int n, const T& alpha,
          const T* x, int incx, const T* y, int incy, T* A, int lda)
{
    blas_impl::geru_check<Backend, T>(handle, m, n, alpha, x, incx, y, incy, A, lda);
    blas_impl::geru_run<Backend, T>(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

#define INSTANTIATE_GERU(BACKEND, TYPE)                                      \
    template void geru<BACKEND, TYPE>(                                       \
        const DdlaHandle_t&, int, int, const TYPE&, const TYPE*, int,        \
        const TYPE*, int, TYPE*, int)

#if DDLA_HAS_CPU
INSTANTIATE_GERU(DdlaBackend::CPU, float);
INSTANTIATE_GERU(DdlaBackend::CPU, double);
INSTANTIATE_GERU(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_GERU(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_GERU(DdlaBackend::GPU, float);
INSTANTIATE_GERU(DdlaBackend::GPU, double);
INSTANTIATE_GERU(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_GERU(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_GERU


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::geru_check<BACKEND, TYPE>(const DdlaHandle_t& handle, int m, int n, const TYPE& alpha, const TYPE* x, int incx, const TYPE* y, int incy, TYPE* A, int lda); \
    template void blas_impl::geru_run<BACKEND, TYPE>(const DdlaHandle_t& handle, int m, int n, const TYPE& alpha, const TYPE* x, int incx, const TYPE* y, int incy, TYPE* A, int lda);
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
