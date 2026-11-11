#include "blas_impl.h"
// ---------------------------------------------------------------------------
// Backend-templated BLAS Level-1 swap:  x <-> y.
//
// Dispatch follows the Backend template parameter. CPU code is always compiled;
// DDLA_HAS_GPU enables GPU code, and DDLA_USE_CUDA / DDLA_USE_HIP select its vendor.
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <complex>
#include <stdexcept>
#include <string>

#include "ddla_connector.h"
#include "ddla_stream.h"
#include "instantiate.h"
#include "require_backend.h"

// CPU BLAS Level-1 SWAP bindings (Fortran ABI), declared by hand like gemm.cpp / scal.cpp do.
extern "C" {
void sswap_(const int*, float*, const int*, float*, const int*);
void dswap_(const int*, double*, const int*, double*, const int*);
void cswap_(const int*, std::complex<float>*, const int*, std::complex<float>*, const int*);
void zswap_(const int*, std::complex<double>*, const int*, std::complex<double>*, const int*);
}

namespace ddla {

namespace {

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
    detail::require_backend<Backend>(handle, "swap");
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
        cpu_swap(n, x, incx, y, incy);
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

DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_SWAP)

#undef INSTANTIATE_SWAP


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::swap_check<BACKEND, TYPE>(const DdlaHandle_t& handle, int n, TYPE* x, int incx, TYPE* y, int incy); \
    template void blas_impl::swap_run<BACKEND, TYPE>(const DdlaHandle_t& handle, int n, TYPE* x, int incx, TYPE* y, int incy)
DDLA_FOR_EACH_BACKEND_TYPE(DDLA_INSTANTIATE_PHASES)
#undef DDLA_INSTANTIATE_PHASES
} // namespace ddla
