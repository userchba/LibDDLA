#ifndef IAMAX_H
#define IAMAX_H

// Public <ddla/ddla.h> must be included before any redeclaration: it
// owns the public default arguments (Backend = default_backend_v), and
// hipcc/clang rejects redeclarations that introduce defaults later.
#include <ddla/ddla.h>
#include "ddla_connector.h"
#include <ddla/ddla_handle_t.h>

namespace ddla{

inline deblasStatus_t deblasIamax(deblasHandle_t handle, int n, const float *x, int incx, int *result) {
#if defined(DDLA_USE_CUDA)
    return cublasIsamax(handle, n, x, incx, result);
#elif defined(DDLA_USE_HIP)
    return hipblasIsamax(handle, n, x, incx, result);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}


inline deblasStatus_t deblasIamax(deblasHandle_t handle, int n, const double *x, int incx, int *result) {
#if defined(DDLA_USE_CUDA)
    return cublasIdamax(handle, n, x, incx, result);
#elif defined(DDLA_USE_HIP)
    return hipblasIdamax(handle, n, x, incx, result);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasIamax(deblasHandle_t handle, int n, const std::complex<float> *x, int incx, int *result) {
#if defined(DDLA_USE_CUDA)
    return cublasIcamax(handle, n, (cuFloatComplex*)x, incx, result);
#elif defined(DDLA_USE_HIP)
    return hipblasIcamax(handle, n, (hipblasComplex*)x, incx, result);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasIamax(deblasHandle_t handle, int n, const std::complex<double> *x, int incx, int *result) {
#if defined(DDLA_USE_CUDA)
    return cublasIzamax(handle, n, (cuDoubleComplex*)x, incx, result);
#elif defined(DDLA_USE_HIP)
    return hipblasIzamax(handle, n, (hipblasDoubleComplex*)x, incx, result);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

/**
 * @brief Backend-neutral BLAS Level-1 iamax: 1-based index of the element
 * with largest magnitude (|Re| + |Im| for complex, matching cuBLAS/hipBLAS
 * iamax semantics).
 *
 * CPU specializations consume host pointers and run a plain vendor-neutral
 * host loop; GPU specializations consume device pointers and call
 * cuBLAS/hipBLAS via deblasIamax.
 */
template <DdlaBackend Backend, typename T>
void iamax(const DdlaHandle_t& handle, int n, const T* x, int incx, int& result);

} // namespace ddla

#endif // IAMAX_H