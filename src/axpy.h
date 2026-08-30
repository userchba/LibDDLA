#ifndef AXPY_H
#define AXPY_H

// Public <ddla/ddla.h> must be included before any redeclaration: it
// owns the public default arguments (Backend = default_backend_v), and
// hipcc/clang rejects redeclarations that introduce defaults later.
#include <ddla/ddla.h>
#include "ddla_connector.h"
#include <ddla/ddla_handle_t.h>

namespace ddla{

inline deblasStatus_t deblasAxpy(deblasHandle_t handle, const int64_t& n, const float& alpha, const float *x, int incx, float *y, int incy) {
#if defined(DDLA_USE_CUDA)
    return cublasSaxpy(handle, n, &alpha, x, incx, y, incy);
#elif defined(DDLA_USE_HIP)
    return hipblasSaxpy(handle, n, &alpha, x, incx, y, incy);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasAxpy(deblasHandle_t handle, const int64_t& n, const double& alpha, const double *x, int incx, double *y, int incy) {
#if defined(DDLA_USE_CUDA)
    return cublasDaxpy(handle, n, &alpha, x, incx, y, incy);
#elif defined(DDLA_USE_HIP)
    return hipblasDaxpy(handle, n, &alpha, x, incx, y, incy);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasAxpy(deblasHandle_t handle, const int64_t& n, const std::complex<float>& alpha, const std::complex<float> *x, int incx, std::complex<float> *y, int incy) {
#if defined(DDLA_USE_CUDA)
    return cublasCaxpy(handle, n, (cuFloatComplex*)&alpha, (cuFloatComplex*)x, incx, (cuFloatComplex*)y, incy);
#elif defined(DDLA_USE_HIP)
    return hipblasCaxpy(handle, n, (hipblasComplex*)&alpha, (hipblasComplex*)x, incx, (hipblasComplex*)y, incy);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasAxpy(deblasHandle_t handle, const int64_t& n, const std::complex<double>& alpha, const std::complex<double> *x, int incx, std::complex<double> *y, int incy) {
#if defined(DDLA_USE_CUDA)
    return cublasZaxpy(handle, n, (cuDoubleComplex*)&alpha, (cuDoubleComplex*)x, incx, (cuDoubleComplex*)y, incy);
#elif defined(DDLA_USE_HIP)
    return hipblasZaxpy(handle, n, (hipblasDoubleComplex*)&alpha, (hipblasDoubleComplex*)x, incx, (hipblasDoubleComplex*)y, incy);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}


/**
 * @brief Backend-neutral BLAS Level-1 axpy: y := alpha * x + y.
 *
 * CPU specializations consume host pointers and run a plain vendor-neutral
 * host loop; GPU specializations consume device pointers and call
 * cuBLAS/hipBLAS via deblasAxpy. Mirrors ddla::scal's shape.
 */
template <DdlaBackend Backend, typename T>
void axpy(const DdlaHandle_t& handle, int n, const T& alpha,
          const T* x, int incx, T* y, int incy);

} // namespace ddla

#endif // AXPY_H
