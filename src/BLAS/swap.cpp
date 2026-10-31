#include "blas_impl.h"
// ---------------------------------------------------------------------------
// Backend-templated BLAS Level-1 swap:  x <-> y.
//
// Structure follows src/scal.cpp: one definition covering CPU (Fortran ?swap on host pointers) and GPU (cuBLAS / hipBLAS
// ?swap on device pointers, enqueued on the handle's stream), selected with `if constexpr` on the Backend template
// parameter, the arms guarded by the BUILD CAPABILITY macros (DDLA_HAS_CPU / DDLA_HAS_GPU), never by the per-TU vendor
// macro alone.  The vendor overloads are private to this translation unit; the public interface is the templated
// ddla::swap declared in <ddla/ddla.h>.
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <complex>
#include <stdexcept>
#include <string>

#include "ddla_connector.h"
#include "ddla_stream.h"

#if DDLA_HAS_CPU
// CPU BLAS Level-1 SWAP bindings (Fortran ABI), declared by hand like gemm.cpp / scal.cpp do.
extern "C" {
void sswap_(const int*, float*, const int*, float*, const int*);
void dswap_(const int*, double*, const int*, double*, const int*);
void cswap_(const int*, std::complex<float>*, const int*, std::complex<float>*, const int*);
void zswap_(const int*, std::complex<double>*, const int*, std::complex<double>*, const int*);
}
#endif

namespace ddla {

namespace {

inline const char* swap_backend_name(DdlaBackend backend)
{
    return backend == DdlaBackend::CPU ? "CPU" : "GPU";
}

#if DDLA_HAS_CPU
inline void cpu_swap(int n, float* x, int incx, float* y, int incy)
{
    sswap_(&n, x, &incx, y, &incy);
}

inline void cpu_swap(int n, double* x, int incx, double* y, int incy)
{
    dswap_(&n, x, &incx, y, &incy);
}

inline void cpu_swap(int n, std::complex<float>* x, int incx, std::complex<float>* y, int incy)
{
    cswap_(&n, x, &incx, y, &incy);
}

inline void cpu_swap(int n, std::complex<double>* x, int incx, std::complex<double>* y, int incy)
{
    zswap_(&n, x, &incx, y, &incy);
}
#endif // DDLA_HAS_CPU

#if DDLA_HAS_GPU
inline gpublasStatus_t gpublasSwap(gpublasHandle_t handle, int n, float* x, int incx, float* y, int incy)
{
#if DDLA_GPU_VENDOR
    return gpublasSswap(handle, n, x, incx, y, incy);
#else
    throw std::runtime_error("swap: no GPU vendor selected at compile time");
#endif
}

inline gpublasStatus_t gpublasSwap(gpublasHandle_t handle, int n, double* x, int incx, double* y, int incy)
{
#if DDLA_GPU_VENDOR
    return gpublasDswap(handle, n, x, incx, y, incy);
#else
    throw std::runtime_error("swap: no GPU vendor selected at compile time");
#endif
}

inline gpublasStatus_t gpublasSwap(gpublasHandle_t handle, int n, std::complex<float>* x, int incx,
                                   std::complex<float>* y, int incy)
{
#if DDLA_GPU_VENDOR
    return gpublasCswap(handle, n, (gpublasComplex*)x, incx, (gpublasComplex*)y, incy);
#else
    throw std::runtime_error("swap: no GPU vendor selected at compile time");
#endif
}

inline gpublasStatus_t gpublasSwap(gpublasHandle_t handle, int n, std::complex<double>* x, int incx,
                                   std::complex<double>* y, int incy)
{
#if DDLA_GPU_VENDOR
    return gpublasZswap(handle, n, (gpublasDoubleComplex*)x, incx, (gpublasDoubleComplex*)y, incy);
#else
    throw std::runtime_error("swap: no GPU vendor selected at compile time");
#endif
}
#endif // DDLA_HAS_GPU

} // anonymous namespace

namespace blas_impl {

template <DdlaBackend Backend, typename T>
void swap_check(const DdlaHandle_t& handle, int n, T* x, int incx, T* y, int incy)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU swap is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU swap is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("swap: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("swap: template backend ") + swap_backend_name(Backend) +
            " does not match handle backend " + swap_backend_name(actual));
    }
    if (n <= 0) return;
    if (x == nullptr || y == nullptr) {
        throw std::runtime_error("swap: null x or y");
    }
    if (incx == 0 || incy == 0) {
        throw std::runtime_error("swap: incx and incy must be nonzero");
    }

}

template <DdlaBackend Backend, typename T>
void swap_run(const DdlaHandle_t& handle, int n, T* x, int incx, T* y, int incy)
{
    if (n <= 0) return;
    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        cpu_swap(n, x, incx, y, incy);
#endif
    } else {
#if DDLA_HAS_GPU
        BLAS_CHECK(gpublasSwap(static_cast<gpublasHandle_t>(handle->blasH), n, x, incx, y, incy));
#endif
    }
}

} // namespace blas_impl

template <DdlaBackend Backend, typename T>
void swap(const DdlaHandle_t& handle, int n, T* x, int incx, T* y, int incy)
{
    blas_impl::swap_check<Backend, T>(handle, n, x, incx, y, incy);
    blas_impl::swap_run<Backend, T>(handle, n, x, incx, y, incy);
}

#define INSTANTIATE_SWAP(BACKEND, TYPE)                                      \
    template void swap<BACKEND, TYPE>(                                       \
        const DdlaHandle_t&, int, TYPE*, int, TYPE*, int)

#if DDLA_HAS_CPU
INSTANTIATE_SWAP(DdlaBackend::CPU, float);
INSTANTIATE_SWAP(DdlaBackend::CPU, double);
INSTANTIATE_SWAP(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_SWAP(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_SWAP(DdlaBackend::GPU, float);
INSTANTIATE_SWAP(DdlaBackend::GPU, double);
INSTANTIATE_SWAP(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_SWAP(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_SWAP


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::swap_check<BACKEND, TYPE>(const DdlaHandle_t& handle, int n, TYPE* x, int incx, TYPE* y, int incy); \
    template void blas_impl::swap_run<BACKEND, TYPE>(const DdlaHandle_t& handle, int n, TYPE* x, int incx, TYPE* y, int incy);
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
