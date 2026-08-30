#ifndef GERU_H
#define GERU_H

// Public <ddla/ddla.h> must be included before any redeclaration: it
// owns the public default arguments (Backend = default_backend_v), and
// hipcc/clang rejects redeclarations that introduce defaults later.
#include <ddla/ddla.h>
#include "ddla_connector.h"
#include <ddla/ddla_handle_t.h>

namespace ddla{

inline deblasStatus_t deblasGeru(deblasHandle_t handle, int m, int n, const float& alpha, const float *x, int incx, const float *y, int incy, float *A, int lda) {
#if defined(DDLA_USE_CUDA)
    return cublasSger(handle, m, n, &alpha, x, incx, y, incy, A, lda);
#elif defined(DDLA_USE_HIP)
    return hipblasSger(handle, m, n, &alpha, x, incx, y, incy, A, lda);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasGeru(deblasHandle_t handle, int m, int n, const double& alpha, const double *x, int incx, const double *y, int incy, double *A, int lda) {
#if defined(DDLA_USE_CUDA)
    return cublasDger(handle, m, n, &alpha, x, incx, y, incy, A, lda);
#elif defined(DDLA_USE_HIP)
    return hipblasDger(handle, m, n, &alpha, x, incx, y, incy, A, lda);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasGeru(deblasHandle_t handle, int m, int n, const std::complex<float>& alpha, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy, std::complex<float> *A, int lda) {
#if defined(DDLA_USE_CUDA)
    return cublasCgeru(handle, m, n, (cuFloatComplex*)&alpha, (cuFloatComplex*)x, incx, (cuFloatComplex*)y, incy, (cuFloatComplex*)A, lda);
#elif defined(DDLA_USE_HIP)
    return hipblasCgeru(handle, m, n, (hipblasComplex*)&alpha, (hipblasComplex*)x, incx, (hipblasComplex*)y, incy, (hipblasComplex*)A, lda);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasGeru(deblasHandle_t handle, int m, int n, const std::complex<double>& alpha, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy, std::complex<double> *A, int lda) {
#if defined(DDLA_USE_CUDA)
    return cublasZgeru(handle, m, n, (cuDoubleComplex*)&alpha, (cuDoubleComplex*)x, incx, (cuDoubleComplex*)y, incy, (cuDoubleComplex*)A, lda);
#elif defined(DDLA_USE_HIP)
    return hipblasZgeru(handle, m, n, (hipblasDoubleComplex*)&alpha, (hipblasDoubleComplex*)x, incx, (hipblasDoubleComplex*)y, incy, (hipblasDoubleComplex*)A, lda);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

/**
 * @brief Backend-neutral BLAS Level-2 geru: A := alpha * x * y^T + A
 * (unconjugated outer product).
 *
 * CPU specializations consume host pointers and run a plain vendor-neutral
 * host loop; GPU specializations consume device pointers and call
 * cuBLAS/hipBLAS via deblasGeru.
 */
template <DdlaBackend Backend, typename T>
void geru(const DdlaHandle_t& handle, int m, int n, const T& alpha,
          const T* x, int incx, const T* y, int incy, T* A, int lda);

} // namespace ddla

#endif // GERU_H