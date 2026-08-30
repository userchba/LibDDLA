#ifndef SCAL_H
#define SCAL_H

// Public <ddla/ddla.h> must be included before any redeclaration: it
// owns the public default arguments (Backend = default_backend_v), and
// hipcc/clang rejects redeclarations that introduce defaults later.
#include <ddla/ddla.h>
#include "ddla_connector.h"
#include <ddla/ddla_handle_t.h>

namespace ddla{

inline deblasStatus_t deblasScal(deblasHandle_t handle, int64_t n, const float& alpha, float *x, int64_t incx)
{
    #if defined(DDLA_USE_CUDA)
    return cublasSscal(handle, n, &alpha, x, incx);
    #elif defined(DDLA_USE_HIP)
    return hipblasSscal(handle, n, &alpha, x, incx);
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif
}

inline deblasStatus_t deblasScal(deblasHandle_t handle, int64_t n, const double& alpha, double *x, int64_t incx)
{
#if defined(DDLA_USE_CUDA)
    return cublasDscal(handle, n, &alpha, x, incx);
#elif defined(DDLA_USE_HIP)
    return hipblasDscal(handle, n, &alpha, x, incx);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasScal(deblasHandle_t handle, int64_t n, const float& alpha, std::complex<float> *x, int64_t incx)
{
#if defined(DDLA_USE_CUDA)
    return cublasCsscal(handle, n, &alpha, (cuFloatComplex*)x, incx);
#elif defined(DDLA_USE_HIP)
    return hipblasCsscal(handle, n, &alpha, (hipblasComplex*)x, incx);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasScal(deblasHandle_t handle, int64_t n, const double& alpha, std::complex<double> *x, int64_t incx)
{
#if defined(DDLA_USE_CUDA)
    return cublasZdscal(handle, n, &alpha, (cuDoubleComplex*)x, incx);
#elif defined(DDLA_USE_HIP)
    return hipblasZdscal(handle, n, &alpha, (hipblasDoubleComplex*)x, incx);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasScal(deblasHandle_t handle, int64_t n, const std::complex<float>& alpha, std::complex<float> *x, int64_t incx) {
#if defined(DDLA_USE_CUDA)
    return cublasCscal(handle, n, (cuFloatComplex*)&alpha, (cuFloatComplex*)x, incx);
#elif defined(DDLA_USE_HIP)
    return hipblasCscal(handle, n, (hipblasComplex*)&alpha, (hipblasComplex*)x, incx);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasScal(deblasHandle_t handle, int64_t n, const std::complex<double>& alpha, std::complex<double> *x, int64_t incx) {
#if defined(DDLA_USE_CUDA)
    return cublasZscal(handle, n, (cuDoubleComplex*)&alpha, (cuDoubleComplex*)x, incx);
#elif defined(DDLA_USE_HIP)
    return hipblasZscal(handle, n, (hipblasDoubleComplex*)&alpha, (hipblasDoubleComplex*)x, incx);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

/**
 * @brief Backend-neutral BLAS Level-1 scale: x := alpha * x.
 *
 * Mirrors ddla::gemm's shape: one function, backend chosen by the template
 * argument, `if constexpr` dispatch inside. CPU specializations consume host
 * pointers and call the linked BLAS (?scal); GPU specializations consume
 * device pointers and call cuBLAS/hipBLAS via deblasScal.
 *
 * Note: this covers the same-type-alpha case (alpha and x share type T).
 * The mixed real-alpha/complex-x variants (csscal/zdscal) are still reached
 * only through the raw deblasScal overloads above.
 */
template <DdlaBackend Backend, typename T>
void scal(const DdlaHandle_t& handle, int n, const T& alpha, T* x, int incx);

} // namespace ddla

#endif // SCAL_H
