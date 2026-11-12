// Public handle-driven entry points. Backend-specialized implementations
// and all inner calls remain in their original translation units.
#include "backend_api.h"
#include "ddla_stream.h"
#include <stdexcept>
#include <string>

namespace ddla {

template <typename T, typename U>
void dgmm(
    const DdlaHandle_t& handle, char side, int m, int n, const T* A, int lda, const U* x, int incx, T* C, int ldc)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return dgmm<DdlaBackend::GPU, T, U>(handle, side, m, n, A, lda, x, incx, C, ldc);
#endif
    return dgmm<DdlaBackend::CPU, T, U>(handle, side, m, n, A, lda, x, incx, C, ldc);
}


template void dgmm<float, float>(const DdlaHandle_t& handle, char side, int m, int n, const float* A, int lda, const float* x, int incx, float* C, int ldc);
template void dgmm<double, double>(const DdlaHandle_t& handle, char side, int m, int n, const double* A, int lda, const double* x, int incx, double* C, int ldc);
template void dgmm<std::complex<float>, std::complex<float>>(const DdlaHandle_t& handle, char side, int m, int n, const std::complex<float>* A, int lda, const std::complex<float>* x, int incx, std::complex<float>* C, int ldc);
template void dgmm<std::complex<double>, std::complex<double>>(const DdlaHandle_t& handle, char side, int m, int n, const std::complex<double>* A, int lda, const std::complex<double>* x, int incx, std::complex<double>* C, int ldc);
template void dgmm<std::complex<float>, float>(const DdlaHandle_t& handle, char side, int m, int n, const std::complex<float>* A, int lda, const float* x, int incx, std::complex<float>* C, int ldc);
template void dgmm<std::complex<double>, double>(const DdlaHandle_t& handle, char side, int m, int n, const std::complex<double>* A, int lda, const double* x, int incx, std::complex<double>* C, int ldc);

template <typename T, typename U>
void pdgmm(
    const DdlaHandle_t& handle, char side, int m, int n, const T* A, const int* descA, const U* x, int incx, T* C, const int* descC)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pdgmm<DdlaBackend::GPU, T, U>(handle, side, m, n, A, descA, x, incx, C, descC);
#endif
    return pdgmm<DdlaBackend::CPU, T, U>(handle, side, m, n, A, descA, x, incx, C, descC);
}


template void pdgmm<float, float>(const DdlaHandle_t& handle, char side, int m, int n, const float* A, const int* descA, const float* x, int incx, float* C, const int* descC);
template void pdgmm<double, double>(const DdlaHandle_t& handle, char side, int m, int n, const double* A, const int* descA, const double* x, int incx, double* C, const int* descC);
template void pdgmm<std::complex<float>, std::complex<float>>(const DdlaHandle_t& handle, char side, int m, int n, const std::complex<float>* A, const int* descA, const std::complex<float>* x, int incx, std::complex<float>* C, const int* descC);
template void pdgmm<std::complex<double>, std::complex<double>>(const DdlaHandle_t& handle, char side, int m, int n, const std::complex<double>* A, const int* descA, const std::complex<double>* x, int incx, std::complex<double>* C, const int* descC);
template void pdgmm<std::complex<float>, float>(const DdlaHandle_t& handle, char side, int m, int n, const std::complex<float>* A, const int* descA, const float* x, int incx, std::complex<float>* C, const int* descC);
template void pdgmm<std::complex<double>, double>(const DdlaHandle_t& handle, char side, int m, int n, const std::complex<double>* A, const int* descA, const double* x, int incx, std::complex<double>* C, const int* descC);

template <typename T>
void ptrtrs(
    const DdlaHandle_t& handle, char side, char uplo, char trans, char diag, int m, int n, T* A, const int* descA, T* B, const int* descB)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return ptrtrs<DdlaBackend::GPU, T>(handle, side, uplo, trans, diag, m, n, A, descA, B, descB);
#endif
    return ptrtrs<DdlaBackend::CPU, T>(handle, side, uplo, trans, diag, m, n, A, descA, B, descB);
}


template void ptrtrs<float>(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag, int m, int n, float* A, const int* descA, float* B, const int* descB);
template void ptrtrs<double>(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag, int m, int n, double* A, const int* descA, double* B, const int* descB);
template void ptrtrs<std::complex<float>>(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag, int m, int n, std::complex<float>* A, const int* descA, std::complex<float>* B, const int* descB);
template void ptrtrs<std::complex<double>>(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag, int m, int n, std::complex<double>* A, const int* descA, std::complex<double>* B, const int* descB);

template <typename T>
void ptrmm(
    const DdlaHandle_t& handle, char side, char uplo, char trans, char diag, int m, int n, const T& alpha, const T* A, const int* descA, T* B, const int* descB)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return ptrmm<DdlaBackend::GPU, T>(handle, side, uplo, trans, diag, m, n, alpha, A, descA, B, descB);
#endif
    return ptrmm<DdlaBackend::CPU, T>(handle, side, uplo, trans, diag, m, n, alpha, A, descA, B, descB);
}


template void ptrmm<float>(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag, int m, int n, const float& alpha, const float* A, const int* descA, float* B, const int* descB);
template void ptrmm<double>(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag, int m, int n, const double& alpha, const double* A, const int* descA, double* B, const int* descB);
template void ptrmm<std::complex<float>>(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag, int m, int n, const std::complex<float>& alpha, const std::complex<float>* A, const int* descA, std::complex<float>* B, const int* descB);
template void ptrmm<std::complex<double>>(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag, int m, int n, const std::complex<double>& alpha, const std::complex<double>* A, const int* descA, std::complex<double>* B, const int* descB);

template <typename T>
void phemm(
    const DdlaHandle_t& handle, char side, char uplo, int m, int n, const T& alpha, const T* A, const int* descA, const T* B, const int* descB, const T& beta, T* C, const int* descC)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return phemm<DdlaBackend::GPU, T>(handle, side, uplo, m, n, alpha, A, descA, B, descB, beta, C, descC);
#endif
    return phemm<DdlaBackend::CPU, T>(handle, side, uplo, m, n, alpha, A, descA, B, descB, beta, C, descC);
}


template void phemm<float>(const DdlaHandle_t& handle, char side, char uplo, int m, int n, const float& alpha, const float* A, const int* descA, const float* B, const int* descB, const float& beta, float* C, const int* descC);
template void phemm<double>(const DdlaHandle_t& handle, char side, char uplo, int m, int n, const double& alpha, const double* A, const int* descA, const double* B, const int* descB, const double& beta, double* C, const int* descC);
template void phemm<std::complex<float>>(const DdlaHandle_t& handle, char side, char uplo, int m, int n, const std::complex<float>& alpha, const std::complex<float>* A, const int* descA, const std::complex<float>* B, const int* descB, const std::complex<float>& beta, std::complex<float>* C, const int* descC);
template void phemm<std::complex<double>>(const DdlaHandle_t& handle, char side, char uplo, int m, int n, const std::complex<double>& alpha, const std::complex<double>* A, const int* descA, const std::complex<double>* B, const int* descB, const std::complex<double>& beta, std::complex<double>* C, const int* descC);

template <typename R>
void phemm(
    const DdlaHandle_t& handle, char side, char uplo, int m, int n, const std::complex<R>& alpha, const R* A, const int* descA, const std::complex<R>* B, const int* descB, const std::complex<R>& beta, std::complex<R>* C, const int* descC)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return phemm<DdlaBackend::GPU, R>(handle, side, uplo, m, n, alpha, A, descA, B, descB, beta, C, descC);
#endif
    return phemm<DdlaBackend::CPU, R>(handle, side, uplo, m, n, alpha, A, descA, B, descB, beta, C, descC);
}


template void phemm<float>(const DdlaHandle_t& handle, char side, char uplo, int m, int n, const std::complex<float>& alpha, const float* A, const int* descA, const std::complex<float>* B, const int* descB, const std::complex<float>& beta, std::complex<float>* C, const int* descC);
template void phemm<double>(const DdlaHandle_t& handle, char side, char uplo, int m, int n, const std::complex<double>& alpha, const double* A, const int* descA, const std::complex<double>* B, const int* descB, const std::complex<double>& beta, std::complex<double>* C, const int* descC);

template <typename R>
void phemm(
    const DdlaHandle_t& handle, char side, char uplo, int m, int n, const std::complex<R>& alpha, const std::complex<R>* A, const int* descA, const R* B, const int* descB, const std::complex<R>& beta, std::complex<R>* C, const int* descC)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return phemm<DdlaBackend::GPU, R>(handle, side, uplo, m, n, alpha, A, descA, B, descB, beta, C, descC);
#endif
    return phemm<DdlaBackend::CPU, R>(handle, side, uplo, m, n, alpha, A, descA, B, descB, beta, C, descC);
}


template void phemm<float>(const DdlaHandle_t& handle, char side, char uplo, int m, int n, const std::complex<float>& alpha, const std::complex<float>* A, const int* descA, const float* B, const int* descB, const std::complex<float>& beta, std::complex<float>* C, const int* descC);
template void phemm<double>(const DdlaHandle_t& handle, char side, char uplo, int m, int n, const std::complex<double>& alpha, const std::complex<double>* A, const int* descA, const double* B, const int* descB, const std::complex<double>& beta, std::complex<double>* C, const int* descC);

template <typename T>
void phemmt(
    const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const T& alpha, const T* A, const int* descA, const T* B, const int* descB, const T& beta, T* C, const int* descC)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return phemmt<DdlaBackend::GPU, T>(handle, side, uploA, uploC, n, alpha, A, descA, B, descB, beta, C, descC);
#endif
    return phemmt<DdlaBackend::CPU, T>(handle, side, uploA, uploC, n, alpha, A, descA, B, descB, beta, C, descC);
}


template void phemmt<float>(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const float& alpha, const float* A, const int* descA, const float* B, const int* descB, const float& beta, float* C, const int* descC);
template void phemmt<double>(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const double& alpha, const double* A, const int* descA, const double* B, const int* descB, const double& beta, double* C, const int* descC);
template void phemmt<std::complex<float>>(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const std::complex<float>& alpha, const std::complex<float>* A, const int* descA, const std::complex<float>* B, const int* descB, const std::complex<float>& beta, std::complex<float>* C, const int* descC);
template void phemmt<std::complex<double>>(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const std::complex<double>& alpha, const std::complex<double>* A, const int* descA, const std::complex<double>* B, const int* descB, const std::complex<double>& beta, std::complex<double>* C, const int* descC);

template <typename R>
void phemmt(
    const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const std::complex<R>& alpha, const R* A, const int* descA, const std::complex<R>* B, const int* descB, const std::complex<R>& beta, std::complex<R>* C, const int* descC)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return phemmt<DdlaBackend::GPU, R>(handle, side, uploA, uploC, n, alpha, A, descA, B, descB, beta, C, descC);
#endif
    return phemmt<DdlaBackend::CPU, R>(handle, side, uploA, uploC, n, alpha, A, descA, B, descB, beta, C, descC);
}


template void phemmt<float>(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const std::complex<float>& alpha, const float* A, const int* descA, const std::complex<float>* B, const int* descB, const std::complex<float>& beta, std::complex<float>* C, const int* descC);
template void phemmt<double>(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const std::complex<double>& alpha, const double* A, const int* descA, const std::complex<double>* B, const int* descB, const std::complex<double>& beta, std::complex<double>* C, const int* descC);

template <typename R>
void phemmt(
    const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const std::complex<R>& alpha, const std::complex<R>* A, const int* descA, const R* B, const int* descB, const std::complex<R>& beta, std::complex<R>* C, const int* descC)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return phemmt<DdlaBackend::GPU, R>(handle, side, uploA, uploC, n, alpha, A, descA, B, descB, beta, C, descC);
#endif
    return phemmt<DdlaBackend::CPU, R>(handle, side, uploA, uploC, n, alpha, A, descA, B, descB, beta, C, descC);
}


template void phemmt<float>(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const std::complex<float>& alpha, const std::complex<float>* A, const int* descA, const float* B, const int* descB, const std::complex<float>& beta, std::complex<float>* C, const int* descC);
template void phemmt<double>(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const std::complex<double>& alpha, const std::complex<double>* A, const int* descA, const double* B, const int* descB, const std::complex<double>& beta, std::complex<double>* C, const int* descC);

template <typename T>
void psymm(
    const DdlaHandle_t& handle, char side, char uplo, int m, int n, const T& alpha, const T* A, const int* descA, const T* B, const int* descB, const T& beta, T* C, const int* descC)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return psymm<DdlaBackend::GPU, T>(handle, side, uplo, m, n, alpha, A, descA, B, descB, beta, C, descC);
#endif
    return psymm<DdlaBackend::CPU, T>(handle, side, uplo, m, n, alpha, A, descA, B, descB, beta, C, descC);
}


template void psymm<float>(const DdlaHandle_t& handle, char side, char uplo, int m, int n, const float& alpha, const float* A, const int* descA, const float* B, const int* descB, const float& beta, float* C, const int* descC);
template void psymm<double>(const DdlaHandle_t& handle, char side, char uplo, int m, int n, const double& alpha, const double* A, const int* descA, const double* B, const int* descB, const double& beta, double* C, const int* descC);
template void psymm<std::complex<float>>(const DdlaHandle_t& handle, char side, char uplo, int m, int n, const std::complex<float>& alpha, const std::complex<float>* A, const int* descA, const std::complex<float>* B, const int* descB, const std::complex<float>& beta, std::complex<float>* C, const int* descC);
template void psymm<std::complex<double>>(const DdlaHandle_t& handle, char side, char uplo, int m, int n, const std::complex<double>& alpha, const std::complex<double>* A, const int* descA, const std::complex<double>* B, const int* descB, const std::complex<double>& beta, std::complex<double>* C, const int* descC);

template <typename T>
void psymmt(
    const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const T& alpha, const T* A, const int* descA, const T* B, const int* descB, const T& beta, T* C, const int* descC)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return psymmt<DdlaBackend::GPU, T>(handle, side, uploA, uploC, n, alpha, A, descA, B, descB, beta, C, descC);
#endif
    return psymmt<DdlaBackend::CPU, T>(handle, side, uploA, uploC, n, alpha, A, descA, B, descB, beta, C, descC);
}


template void psymmt<float>(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const float& alpha, const float* A, const int* descA, const float* B, const int* descB, const float& beta, float* C, const int* descC);
template void psymmt<double>(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const double& alpha, const double* A, const int* descA, const double* B, const int* descB, const double& beta, double* C, const int* descC);
template void psymmt<std::complex<float>>(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const std::complex<float>& alpha, const std::complex<float>* A, const int* descA, const std::complex<float>* B, const int* descB, const std::complex<float>& beta, std::complex<float>* C, const int* descC);
template void psymmt<std::complex<double>>(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const std::complex<double>& alpha, const std::complex<double>* A, const int* descA, const std::complex<double>* B, const int* descB, const std::complex<double>& beta, std::complex<double>* C, const int* descC);

template <typename T>
void pgemmt(
    const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k, const T& alpha, const T* A, const int* descA, const T* B, const int* descB, const T& beta, T* C, const int* descC)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pgemmt<DdlaBackend::GPU, T>(handle, uplo, transa, transb, n, k, alpha, A, descA, B, descB, beta, C, descC);
#endif
    return pgemmt<DdlaBackend::CPU, T>(handle, uplo, transa, transb, n, k, alpha, A, descA, B, descB, beta, C, descC);
}


template void pgemmt<float>(const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k, const float& alpha, const float* A, const int* descA, const float* B, const int* descB, const float& beta, float* C, const int* descC);
template void pgemmt<double>(const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k, const double& alpha, const double* A, const int* descA, const double* B, const int* descB, const double& beta, double* C, const int* descC);
template void pgemmt<std::complex<float>>(const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k, const std::complex<float>& alpha, const std::complex<float>* A, const int* descA, const std::complex<float>* B, const int* descB, const std::complex<float>& beta, std::complex<float>* C, const int* descC);
template void pgemmt<std::complex<double>>(const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k, const std::complex<double>& alpha, const std::complex<double>* A, const int* descA, const std::complex<double>* B, const int* descB, const std::complex<double>& beta, std::complex<double>* C, const int* descC);

template <typename R>
void pgemmt(
    const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k, const std::complex<R>& alpha, const R* A, const int* descA, const std::complex<R>* B, const int* descB, const std::complex<R>& beta, std::complex<R>* C, const int* descC)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pgemmt<DdlaBackend::GPU, R>(handle, uplo, transa, transb, n, k, alpha, A, descA, B, descB, beta, C, descC);
#endif
    return pgemmt<DdlaBackend::CPU, R>(handle, uplo, transa, transb, n, k, alpha, A, descA, B, descB, beta, C, descC);
}


template void pgemmt<float>(const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k, const std::complex<float>& alpha, const float* A, const int* descA, const std::complex<float>* B, const int* descB, const std::complex<float>& beta, std::complex<float>* C, const int* descC);
template void pgemmt<double>(const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k, const std::complex<double>& alpha, const double* A, const int* descA, const std::complex<double>* B, const int* descB, const std::complex<double>& beta, std::complex<double>* C, const int* descC);

template <typename R>
void pgemmt(
    const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k, const std::complex<R>& alpha, const std::complex<R>* A, const int* descA, const R* B, const int* descB, const std::complex<R>& beta, std::complex<R>* C, const int* descC)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pgemmt<DdlaBackend::GPU, R>(handle, uplo, transa, transb, n, k, alpha, A, descA, B, descB, beta, C, descC);
#endif
    return pgemmt<DdlaBackend::CPU, R>(handle, uplo, transa, transb, n, k, alpha, A, descA, B, descB, beta, C, descC);
}


template void pgemmt<float>(const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k, const std::complex<float>& alpha, const std::complex<float>* A, const int* descA, const float* B, const int* descB, const std::complex<float>& beta, std::complex<float>* C, const int* descC);
template void pgemmt<double>(const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k, const std::complex<double>& alpha, const std::complex<double>* A, const int* descA, const double* B, const int* descB, const std::complex<double>& beta, std::complex<double>* C, const int* descC);

template <typename T>
void psyrk(
    const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const T& alpha, const T* A, const int* descA, const T& beta, T* C, const int* descC)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return psyrk<DdlaBackend::GPU, T>(handle, uplo, trans, n, k, alpha, A, descA, beta, C, descC);
#endif
    return psyrk<DdlaBackend::CPU, T>(handle, uplo, trans, n, k, alpha, A, descA, beta, C, descC);
}


template void psyrk<float>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const float& alpha, const float* A, const int* descA, const float& beta, float* C, const int* descC);
template void psyrk<double>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const double& alpha, const double* A, const int* descA, const double& beta, double* C, const int* descC);
template void psyrk<std::complex<float>>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const std::complex<float>& alpha, const std::complex<float>* A, const int* descA, const std::complex<float>& beta, std::complex<float>* C, const int* descC);
template void psyrk<std::complex<double>>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const std::complex<double>& alpha, const std::complex<double>* A, const int* descA, const std::complex<double>& beta, std::complex<double>* C, const int* descC);

template <typename T>
void psyr2k(
    const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const T& alpha, const T* A, const int* descA, const T* B, const int* descB, const T& beta, T* C, const int* descC)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return psyr2k<DdlaBackend::GPU, T>(handle, uplo, trans, n, k, alpha, A, descA, B, descB, beta, C, descC);
#endif
    return psyr2k<DdlaBackend::CPU, T>(handle, uplo, trans, n, k, alpha, A, descA, B, descB, beta, C, descC);
}


template void psyr2k<float>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const float& alpha, const float* A, const int* descA, const float* B, const int* descB, const float& beta, float* C, const int* descC);
template void psyr2k<double>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const double& alpha, const double* A, const int* descA, const double* B, const int* descB, const double& beta, double* C, const int* descC);
template void psyr2k<std::complex<float>>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const std::complex<float>& alpha, const std::complex<float>* A, const int* descA, const std::complex<float>* B, const int* descB, const std::complex<float>& beta, std::complex<float>* C, const int* descC);
template void psyr2k<std::complex<double>>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const std::complex<double>& alpha, const std::complex<double>* A, const int* descA, const std::complex<double>* B, const int* descB, const std::complex<double>& beta, std::complex<double>* C, const int* descC);

template <typename T>
void pherk(
    const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const typename real_type<T>::type& alpha, const T* A, const int* descA, const typename real_type<T>::type& beta, T* C, const int* descC)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pherk<DdlaBackend::GPU, T>(handle, uplo, trans, n, k, alpha, A, descA, beta, C, descC);
#endif
    return pherk<DdlaBackend::CPU, T>(handle, uplo, trans, n, k, alpha, A, descA, beta, C, descC);
}


template void pherk<std::complex<float>>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const typename real_type<std::complex<float>>::type& alpha, const std::complex<float>* A, const int* descA, const typename real_type<std::complex<float>>::type& beta, std::complex<float>* C, const int* descC);
template void pherk<std::complex<double>>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const typename real_type<std::complex<double>>::type& alpha, const std::complex<double>* A, const int* descA, const typename real_type<std::complex<double>>::type& beta, std::complex<double>* C, const int* descC);

template <typename T>
void pher2k(
    const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const T& alpha, const T* A, const int* descA, const T* B, const int* descB, const typename real_type<T>::type& beta, T* C, const int* descC)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pher2k<DdlaBackend::GPU, T>(handle, uplo, trans, n, k, alpha, A, descA, B, descB, beta, C, descC);
#endif
    return pher2k<DdlaBackend::CPU, T>(handle, uplo, trans, n, k, alpha, A, descA, B, descB, beta, C, descC);
}


template void pher2k<std::complex<float>>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const std::complex<float>& alpha, const std::complex<float>* A, const int* descA, const std::complex<float>* B, const int* descB, const typename real_type<std::complex<float>>::type& beta, std::complex<float>* C, const int* descC);
template void pher2k<std::complex<double>>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const std::complex<double>& alpha, const std::complex<double>* A, const int* descA, const std::complex<double>* B, const int* descB, const typename real_type<std::complex<double>>::type& beta, std::complex<double>* C, const int* descC);

template <typename T>
void symm(
    const DdlaHandle_t& handle, char side, char uplo, int m, int n, const T& alpha, const T* A, int lda, const T* B, int ldb, const T& beta, T* C, int ldc)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return symm<DdlaBackend::GPU, T>(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
    return symm<DdlaBackend::CPU, T>(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}


template void symm<float>(const DdlaHandle_t& handle, char side, char uplo, int m, int n, const float& alpha, const float* A, int lda, const float* B, int ldb, const float& beta, float* C, int ldc);
template void symm<double>(const DdlaHandle_t& handle, char side, char uplo, int m, int n, const double& alpha, const double* A, int lda, const double* B, int ldb, const double& beta, double* C, int ldc);
template void symm<std::complex<float>>(const DdlaHandle_t& handle, char side, char uplo, int m, int n, const std::complex<float>& alpha, const std::complex<float>* A, int lda, const std::complex<float>* B, int ldb, const std::complex<float>& beta, std::complex<float>* C, int ldc);
template void symm<std::complex<double>>(const DdlaHandle_t& handle, char side, char uplo, int m, int n, const std::complex<double>& alpha, const std::complex<double>* A, int lda, const std::complex<double>* B, int ldb, const std::complex<double>& beta, std::complex<double>* C, int ldc);

template <typename T>
void syr2k(
    const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const T& alpha, const T* A, int lda, const T* B, int ldb, const T& beta, T* C, int ldc)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return syr2k<DdlaBackend::GPU, T>(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
    return syr2k<DdlaBackend::CPU, T>(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}


template void syr2k<float>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const float& alpha, const float* A, int lda, const float* B, int ldb, const float& beta, float* C, int ldc);
template void syr2k<double>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const double& alpha, const double* A, int lda, const double* B, int ldb, const double& beta, double* C, int ldc);
template void syr2k<std::complex<float>>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const std::complex<float>& alpha, const std::complex<float>* A, int lda, const std::complex<float>* B, int ldb, const std::complex<float>& beta, std::complex<float>* C, int ldc);
template void syr2k<std::complex<double>>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const std::complex<double>& alpha, const std::complex<double>* A, int lda, const std::complex<double>* B, int ldb, const std::complex<double>& beta, std::complex<double>* C, int ldc);

template <typename T>
void plapiv(
    const DdlaHandle_t& handle, char direc, char rowcol, char pivroc, int m, int n, T* A, const int* descA, const int* ipiv, const int* descIP, int* iwork)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return plapiv<DdlaBackend::GPU, T>(handle, direc, rowcol, pivroc, m, n, A, descA, ipiv, descIP, iwork);
#endif
    return plapiv<DdlaBackend::CPU, T>(handle, direc, rowcol, pivroc, m, n, A, descA, ipiv, descIP, iwork);
}


template void plapiv<float>(const DdlaHandle_t& handle, char direc, char rowcol, char pivroc, int m, int n, float* A, const int* descA, const int* ipiv, const int* descIP, int* iwork);
template void plapiv<double>(const DdlaHandle_t& handle, char direc, char rowcol, char pivroc, int m, int n, double* A, const int* descA, const int* ipiv, const int* descIP, int* iwork);
template void plapiv<std::complex<float>>(const DdlaHandle_t& handle, char direc, char rowcol, char pivroc, int m, int n, std::complex<float>* A, const int* descA, const int* ipiv, const int* descIP, int* iwork);
template void plapiv<std::complex<double>>(const DdlaHandle_t& handle, char direc, char rowcol, char pivroc, int m, int n, std::complex<double>* A, const int* descA, const int* ipiv, const int* descIP, int* iwork);

template <typename T>
void pswap(
    const DdlaHandle_t& handle, int N, T* A, int ia, int ja, const int* descA, int inca, T* B, int ib, int jb, const int* descB, int incb)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pswap<DdlaBackend::GPU, T>(handle, N, A, ia, ja, descA, inca, B, ib, jb, descB, incb);
#endif
    return pswap<DdlaBackend::CPU, T>(handle, N, A, ia, ja, descA, inca, B, ib, jb, descB, incb);
}


template void pswap<float>(const DdlaHandle_t& handle, int N, float* A, int ia, int ja, const int* descA, int inca, float* B, int ib, int jb, const int* descB, int incb);
template void pswap<double>(const DdlaHandle_t& handle, int N, double* A, int ia, int ja, const int* descA, int inca, double* B, int ib, int jb, const int* descB, int incb);
template void pswap<std::complex<float>>(const DdlaHandle_t& handle, int N, std::complex<float>* A, int ia, int ja, const int* descA, int inca, std::complex<float>* B, int ib, int jb, const int* descB, int incb);
template void pswap<std::complex<double>>(const DdlaHandle_t& handle, int N, std::complex<double>* A, int ia, int ja, const int* descA, int inca, std::complex<double>* B, int ib, int jb, const int* descB, int incb);

template <typename T>
void gather(
    const DdlaHandle_t& handle, int m, int n, const T* A, int ia, int ja, const int* descA, int root, T* dst, int lddst)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return gather<DdlaBackend::GPU, T>(handle, m, n, A, ia, ja, descA, root, dst, lddst);
#endif
    return gather<DdlaBackend::CPU, T>(handle, m, n, A, ia, ja, descA, root, dst, lddst);
}


template void gather<float>(const DdlaHandle_t& handle, int m, int n, const float* A, int ia, int ja, const int* descA, int root, float* dst, int lddst);
template void gather<double>(const DdlaHandle_t& handle, int m, int n, const double* A, int ia, int ja, const int* descA, int root, double* dst, int lddst);
template void gather<std::complex<float>>(const DdlaHandle_t& handle, int m, int n, const std::complex<float>* A, int ia, int ja, const int* descA, int root, std::complex<float>* dst, int lddst);
template void gather<std::complex<double>>(const DdlaHandle_t& handle, int m, int n, const std::complex<double>* A, int ia, int ja, const int* descA, int root, std::complex<double>* dst, int lddst);

template <typename T>
void scatter(
    const DdlaHandle_t& handle, int m, int n, T* A, int ia, int ja, const int* descA, int root, const T* src, int ldsrc)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return scatter<DdlaBackend::GPU, T>(handle, m, n, A, ia, ja, descA, root, src, ldsrc);
#endif
    return scatter<DdlaBackend::CPU, T>(handle, m, n, A, ia, ja, descA, root, src, ldsrc);
}


template void scatter<float>(const DdlaHandle_t& handle, int m, int n, float* A, int ia, int ja, const int* descA, int root, const float* src, int ldsrc);
template void scatter<double>(const DdlaHandle_t& handle, int m, int n, double* A, int ia, int ja, const int* descA, int root, const double* src, int ldsrc);
template void scatter<std::complex<float>>(const DdlaHandle_t& handle, int m, int n, std::complex<float>* A, int ia, int ja, const int* descA, int root, const std::complex<float>* src, int ldsrc);
template void scatter<std::complex<double>>(const DdlaHandle_t& handle, int m, int n, std::complex<double>* A, int ia, int ja, const int* descA, int root, const std::complex<double>* src, int ldsrc);

template <typename T>
void pgetrf(
    const DdlaHandle_t& handle, int m, int n, T* A, int ia, int ja, const int* descA, int* ipiv, int& info)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pgetrf<DdlaBackend::GPU, T>(handle, m, n, A, ia, ja, descA, ipiv, info);
#endif
    return pgetrf<DdlaBackend::CPU, T>(handle, m, n, A, ia, ja, descA, ipiv, info);
}


template void pgetrf<float>(const DdlaHandle_t& handle, int m, int n, float* A, int ia, int ja, const int* descA, int* ipiv, int& info);
template void pgetrf<double>(const DdlaHandle_t& handle, int m, int n, double* A, int ia, int ja, const int* descA, int* ipiv, int& info);
template void pgetrf<std::complex<float>>(const DdlaHandle_t& handle, int m, int n, std::complex<float>* A, int ia, int ja, const int* descA, int* ipiv, int& info);
template void pgetrf<std::complex<double>>(const DdlaHandle_t& handle, int m, int n, std::complex<double>* A, int ia, int ja, const int* descA, int* ipiv, int& info);

template <typename T>
void pgetrf_nopiv(
    const DdlaHandle_t& handle, int m, int n, T* A, const int* descA, int& info)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pgetrf_nopiv<DdlaBackend::GPU, T>(handle, m, n, A, descA, info);
#endif
    return pgetrf_nopiv<DdlaBackend::CPU, T>(handle, m, n, A, descA, info);
}


template void pgetrf_nopiv<float>(const DdlaHandle_t& handle, int m, int n, float* A, const int* descA, int& info);
template void pgetrf_nopiv<double>(const DdlaHandle_t& handle, int m, int n, double* A, const int* descA, int& info);
template void pgetrf_nopiv<std::complex<float>>(const DdlaHandle_t& handle, int m, int n, std::complex<float>* A, const int* descA, int& info);
template void pgetrf_nopiv<std::complex<double>>(const DdlaHandle_t& handle, int m, int n, std::complex<double>* A, const int* descA, int& info);

template <typename T>
void pgetrs(
    const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, T* A, const int* descA, const int* ipiv, T* B, const int* descB)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pgetrs<DdlaBackend::GPU, T>(handle, side, trans, n, nrhs, A, descA, ipiv, B, descB);
#endif
    return pgetrs<DdlaBackend::CPU, T>(handle, side, trans, n, nrhs, A, descA, ipiv, B, descB);
}


template void pgetrs<float>(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, float* A, const int* descA, const int* ipiv, float* B, const int* descB);
template void pgetrs<double>(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, double* A, const int* descA, const int* ipiv, double* B, const int* descB);
template void pgetrs<std::complex<float>>(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, std::complex<float>* A, const int* descA, const int* ipiv, std::complex<float>* B, const int* descB);
template void pgetrs<std::complex<double>>(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, std::complex<double>* A, const int* descA, const int* ipiv, std::complex<double>* B, const int* descB);

template <typename T>
void pgetrs_nopiv(
    const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, T* A, const int* descA, T* B, const int* descB)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pgetrs_nopiv<DdlaBackend::GPU, T>(handle, side, trans, n, nrhs, A, descA, B, descB);
#endif
    return pgetrs_nopiv<DdlaBackend::CPU, T>(handle, side, trans, n, nrhs, A, descA, B, descB);
}


template void pgetrs_nopiv<float>(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, float* A, const int* descA, float* B, const int* descB);
template void pgetrs_nopiv<double>(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, double* A, const int* descA, double* B, const int* descB);
template void pgetrs_nopiv<std::complex<float>>(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, std::complex<float>* A, const int* descA, std::complex<float>* B, const int* descB);
template void pgetrs_nopiv<std::complex<double>>(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, std::complex<double>* A, const int* descA, std::complex<double>* B, const int* descB);

template <typename T>
void pgesv(
    const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, T* A, const int* descA, T* B, const int* descB)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pgesv<DdlaBackend::GPU, T>(handle, side, trans, n, nrhs, A, descA, B, descB);
#endif
    return pgesv<DdlaBackend::CPU, T>(handle, side, trans, n, nrhs, A, descA, B, descB);
}


template void pgesv<float>(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, float* A, const int* descA, float* B, const int* descB);
template void pgesv<double>(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, double* A, const int* descA, double* B, const int* descB);
template void pgesv<std::complex<float>>(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, std::complex<float>* A, const int* descA, std::complex<float>* B, const int* descB);
template void pgesv<std::complex<double>>(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, std::complex<double>* A, const int* descA, std::complex<double>* B, const int* descB);

template <typename T>
void pgesv_nopiv(
    const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, T* A, const int* descA, T* B, const int* descB)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pgesv_nopiv<DdlaBackend::GPU, T>(handle, side, trans, n, nrhs, A, descA, B, descB);
#endif
    return pgesv_nopiv<DdlaBackend::CPU, T>(handle, side, trans, n, nrhs, A, descA, B, descB);
}


template void pgesv_nopiv<float>(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, float* A, const int* descA, float* B, const int* descB);
template void pgesv_nopiv<double>(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, double* A, const int* descA, double* B, const int* descB);
template void pgesv_nopiv<std::complex<float>>(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, std::complex<float>* A, const int* descA, std::complex<float>* B, const int* descB);
template void pgesv_nopiv<std::complex<double>>(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, std::complex<double>* A, const int* descA, std::complex<double>* B, const int* descB);

template <typename T>
void pgetrf_bpiv(
    const DdlaHandle_t& handle, int m, int n, T* A, const int* descA, int* pivots, int& info)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pgetrf_bpiv<DdlaBackend::GPU, T>(handle, m, n, A, descA, pivots, info);
#endif
    return pgetrf_bpiv<DdlaBackend::CPU, T>(handle, m, n, A, descA, pivots, info);
}


template void pgetrf_bpiv<float>(const DdlaHandle_t& handle, int m, int n, float* A, const int* descA, int* pivots, int& info);
template void pgetrf_bpiv<double>(const DdlaHandle_t& handle, int m, int n, double* A, const int* descA, int* pivots, int& info);
template void pgetrf_bpiv<std::complex<float>>(const DdlaHandle_t& handle, int m, int n, std::complex<float>* A, const int* descA, int* pivots, int& info);
template void pgetrf_bpiv<std::complex<double>>(const DdlaHandle_t& handle, int m, int n, std::complex<double>* A, const int* descA, int* pivots, int& info);

template <typename T>
void pgetrs_bpiv(
    const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, T* A, const int* descA, int* pivots, T* B, const int* descB)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pgetrs_bpiv<DdlaBackend::GPU, T>(handle, side, trans, n, nrhs, A, descA, pivots, B, descB);
#endif
    return pgetrs_bpiv<DdlaBackend::CPU, T>(handle, side, trans, n, nrhs, A, descA, pivots, B, descB);
}


template void pgetrs_bpiv<float>(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, float* A, const int* descA, int* pivots, float* B, const int* descB);
template void pgetrs_bpiv<double>(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, double* A, const int* descA, int* pivots, double* B, const int* descB);
template void pgetrs_bpiv<std::complex<float>>(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, std::complex<float>* A, const int* descA, int* pivots, std::complex<float>* B, const int* descB);
template void pgetrs_bpiv<std::complex<double>>(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, std::complex<double>* A, const int* descA, int* pivots, std::complex<double>* B, const int* descB);

template <typename T>
void pgesv_bpiv(
    const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, T* A, const int* descA, T* B, const int* descB)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pgesv_bpiv<DdlaBackend::GPU, T>(handle, side, trans, n, nrhs, A, descA, B, descB);
#endif
    return pgesv_bpiv<DdlaBackend::CPU, T>(handle, side, trans, n, nrhs, A, descA, B, descB);
}


template void pgesv_bpiv<float>(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, float* A, const int* descA, float* B, const int* descB);
template void pgesv_bpiv<double>(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, double* A, const int* descA, double* B, const int* descB);
template void pgesv_bpiv<std::complex<float>>(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, std::complex<float>* A, const int* descA, std::complex<float>* B, const int* descB);
template void pgesv_bpiv<std::complex<double>>(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs, std::complex<double>* A, const int* descA, std::complex<double>* B, const int* descB);

template <typename T>
void pgemm(
    const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const T& alpha, const T* A, const int* descA, const T* B, const int* descB, const T& beta, T* C, const int* descC)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pgemm<DdlaBackend::GPU, T>(handle, transa, transb, m, n, k, alpha, A, descA, B, descB, beta, C, descC);
#endif
    return pgemm<DdlaBackend::CPU, T>(handle, transa, transb, m, n, k, alpha, A, descA, B, descB, beta, C, descC);
}


template void pgemm<float>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const float& alpha, const float* A, const int* descA, const float* B, const int* descB, const float& beta, float* C, const int* descC);
template void pgemm<double>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const double& alpha, const double* A, const int* descA, const double* B, const int* descB, const double& beta, double* C, const int* descC);
template void pgemm<std::complex<float>>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const std::complex<float>& alpha, const std::complex<float>* A, const int* descA, const std::complex<float>* B, const int* descB, const std::complex<float>& beta, std::complex<float>* C, const int* descC);
template void pgemm<std::complex<double>>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const std::complex<double>& alpha, const std::complex<double>* A, const int* descA, const std::complex<double>* B, const int* descB, const std::complex<double>& beta, std::complex<double>* C, const int* descC);

template <typename T>
void pgeqrf(
    const DdlaHandle_t& handle, int m, int n, T* A, const int* descA, T* tau, int& info)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pgeqrf<DdlaBackend::GPU, T>(handle, m, n, A, descA, tau, info);
#endif
    return pgeqrf<DdlaBackend::CPU, T>(handle, m, n, A, descA, tau, info);
}


template void pgeqrf<float>(const DdlaHandle_t& handle, int m, int n, float* A, const int* descA, float* tau, int& info);
template void pgeqrf<double>(const DdlaHandle_t& handle, int m, int n, double* A, const int* descA, double* tau, int& info);
template void pgeqrf<std::complex<float>>(const DdlaHandle_t& handle, int m, int n, std::complex<float>* A, const int* descA, std::complex<float>* tau, int& info);
template void pgeqrf<std::complex<double>>(const DdlaHandle_t& handle, int m, int n, std::complex<double>* A, const int* descA, std::complex<double>* tau, int& info);

template <typename T>
void pgeqrs(
    const DdlaHandle_t& handle, int m, int n, int nrhs, const T* A, const int* descA, const T* tau, T* B, const int* descB, int& info)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pgeqrs<DdlaBackend::GPU, T>(handle, m, n, nrhs, A, descA, tau, B, descB, info);
#endif
    return pgeqrs<DdlaBackend::CPU, T>(handle, m, n, nrhs, A, descA, tau, B, descB, info);
}


template void pgeqrs<float>(const DdlaHandle_t& handle, int m, int n, int nrhs, const float* A, const int* descA, const float* tau, float* B, const int* descB, int& info);
template void pgeqrs<double>(const DdlaHandle_t& handle, int m, int n, int nrhs, const double* A, const int* descA, const double* tau, double* B, const int* descB, int& info);
template void pgeqrs<std::complex<float>>(const DdlaHandle_t& handle, int m, int n, int nrhs, const std::complex<float>* A, const int* descA, const std::complex<float>* tau, std::complex<float>* B, const int* descB, int& info);
template void pgeqrs<std::complex<double>>(const DdlaHandle_t& handle, int m, int n, int nrhs, const std::complex<double>* A, const int* descA, const std::complex<double>* tau, std::complex<double>* B, const int* descB, int& info);

template <typename T>
void pbdsdc(
    const DdlaHandle_t& handle, char uplo, int n, T* d, T* e, T* s, T* U, T* V, int ld, int& info)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pbdsdc<DdlaBackend::GPU, T>(handle, uplo, n, d, e, s, U, V, ld, info);
#endif
    return pbdsdc<DdlaBackend::CPU, T>(handle, uplo, n, d, e, s, U, V, ld, info);
}


template void pbdsdc<float>(const DdlaHandle_t& handle, char uplo, int n, float* d, float* e, float* s, float* U, float* V, int ld, int& info);
template void pbdsdc<double>(const DdlaHandle_t& handle, char uplo, int n, double* d, double* e, double* s, double* U, double* V, int ld, int& info);

template <typename T>
void pgebrd(
    const DdlaHandle_t& handle, int m, int n, T* A, const int* descA, typename real_type<T>::type* d, typename real_type<T>::type* e, T* tauq, T* taup, int& info)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pgebrd<DdlaBackend::GPU, T>(handle, m, n, A, descA, d, e, tauq, taup, info);
#endif
    return pgebrd<DdlaBackend::CPU, T>(handle, m, n, A, descA, d, e, tauq, taup, info);
}


template void pgebrd<float>(const DdlaHandle_t& handle, int m, int n, float* A, const int* descA, typename real_type<float>::type* d, typename real_type<float>::type* e, float* tauq, float* taup, int& info);
template void pgebrd<double>(const DdlaHandle_t& handle, int m, int n, double* A, const int* descA, typename real_type<double>::type* d, typename real_type<double>::type* e, double* tauq, double* taup, int& info);
template void pgebrd<std::complex<float>>(const DdlaHandle_t& handle, int m, int n, std::complex<float>* A, const int* descA, typename real_type<std::complex<float>>::type* d, typename real_type<std::complex<float>>::type* e, std::complex<float>* tauq, std::complex<float>* taup, int& info);
template void pgebrd<std::complex<double>>(const DdlaHandle_t& handle, int m, int n, std::complex<double>* A, const int* descA, typename real_type<std::complex<double>>::type* d, typename real_type<std::complex<double>>::type* e, std::complex<double>* tauq, std::complex<double>* taup, int& info);

template <typename T>
void pgesvd(
    const DdlaHandle_t& handle, char jobu, char jobvt, int m, int n, T* A, const int* descA, typename real_type<T>::type* s, T* U, const int* descU, T* VT, const int* descVT, int& info)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pgesvd<DdlaBackend::GPU, T>(handle, jobu, jobvt, m, n, A, descA, s, U, descU, VT, descVT, info);
#endif
    return pgesvd<DdlaBackend::CPU, T>(handle, jobu, jobvt, m, n, A, descA, s, U, descU, VT, descVT, info);
}


template void pgesvd<float>(const DdlaHandle_t& handle, char jobu, char jobvt, int m, int n, float* A, const int* descA, typename real_type<float>::type* s, float* U, const int* descU, float* VT, const int* descVT, int& info);
template void pgesvd<double>(const DdlaHandle_t& handle, char jobu, char jobvt, int m, int n, double* A, const int* descA, typename real_type<double>::type* s, double* U, const int* descU, double* VT, const int* descVT, int& info);
template void pgesvd<std::complex<float>>(const DdlaHandle_t& handle, char jobu, char jobvt, int m, int n, std::complex<float>* A, const int* descA, typename real_type<std::complex<float>>::type* s, std::complex<float>* U, const int* descU, std::complex<float>* VT, const int* descVT, int& info);
template void pgesvd<std::complex<double>>(const DdlaHandle_t& handle, char jobu, char jobvt, int m, int n, std::complex<double>* A, const int* descA, typename real_type<std::complex<double>>::type* s, std::complex<double>* U, const int* descU, std::complex<double>* VT, const int* descVT, int& info);

template <typename T>
void punmbr(
    const DdlaHandle_t& handle, char vect, char side, char trans, int m, int n, int k, const T* A, const int* descA, const T* tau, T* C, const int* descC, int& info)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return punmbr<DdlaBackend::GPU, T>(handle, vect, side, trans, m, n, k, A, descA, tau, C, descC, info);
#endif
    return punmbr<DdlaBackend::CPU, T>(handle, vect, side, trans, m, n, k, A, descA, tau, C, descC, info);
}


template void punmbr<float>(const DdlaHandle_t& handle, char vect, char side, char trans, int m, int n, int k, const float* A, const int* descA, const float* tau, float* C, const int* descC, int& info);
template void punmbr<double>(const DdlaHandle_t& handle, char vect, char side, char trans, int m, int n, int k, const double* A, const int* descA, const double* tau, double* C, const int* descC, int& info);
template void punmbr<std::complex<float>>(const DdlaHandle_t& handle, char vect, char side, char trans, int m, int n, int k, const std::complex<float>* A, const int* descA, const std::complex<float>* tau, std::complex<float>* C, const int* descC, int& info);
template void punmbr<std::complex<double>>(const DdlaHandle_t& handle, char vect, char side, char trans, int m, int n, int k, const std::complex<double>* A, const int* descA, const std::complex<double>* tau, std::complex<double>* C, const int* descC, int& info);

template <typename T>
void pgeqlf(
    const DdlaHandle_t& handle, int m, int n, T* A, const int* descA, T* tau, int& info)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pgeqlf<DdlaBackend::GPU, T>(handle, m, n, A, descA, tau, info);
#endif
    return pgeqlf<DdlaBackend::CPU, T>(handle, m, n, A, descA, tau, info);
}


template void pgeqlf<float>(const DdlaHandle_t& handle, int m, int n, float* A, const int* descA, float* tau, int& info);
template void pgeqlf<double>(const DdlaHandle_t& handle, int m, int n, double* A, const int* descA, double* tau, int& info);
template void pgeqlf<std::complex<float>>(const DdlaHandle_t& handle, int m, int n, std::complex<float>* A, const int* descA, std::complex<float>* tau, int& info);
template void pgeqlf<std::complex<double>>(const DdlaHandle_t& handle, int m, int n, std::complex<double>* A, const int* descA, std::complex<double>* tau, int& info);

template <typename T>
void pgeadd(
    const DdlaHandle_t& handle, char transa, char transb, int m, int n, const T& alpha, const T* A, int ia, int ja, const int* descA, const T& beta, const T* B, int ib, int jb, const int* descB, T* C, int ic, int jc, const int* descC)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pgeadd<DdlaBackend::GPU, T>(handle, transa, transb, m, n, alpha, A, ia, ja, descA, beta, B, ib, jb, descB, C, ic, jc, descC);
#endif
    return pgeadd<DdlaBackend::CPU, T>(handle, transa, transb, m, n, alpha, A, ia, ja, descA, beta, B, ib, jb, descB, C, ic, jc, descC);
}


template void pgeadd<float>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, const float& alpha, const float* A, int ia, int ja, const int* descA, const float& beta, const float* B, int ib, int jb, const int* descB, float* C, int ic, int jc, const int* descC);
template void pgeadd<double>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, const double& alpha, const double* A, int ia, int ja, const int* descA, const double& beta, const double* B, int ib, int jb, const int* descB, double* C, int ic, int jc, const int* descC);
template void pgeadd<std::complex<float>>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, const std::complex<float>& alpha, const std::complex<float>* A, int ia, int ja, const int* descA, const std::complex<float>& beta, const std::complex<float>* B, int ib, int jb, const int* descB, std::complex<float>* C, int ic, int jc, const int* descC);
template void pgeadd<std::complex<double>>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, const std::complex<double>& alpha, const std::complex<double>* A, int ia, int ja, const int* descA, const std::complex<double>& beta, const std::complex<double>* B, int ib, int jb, const int* descB, std::complex<double>* C, int ic, int jc, const int* descC);

template <typename T1, typename T2>
void pdam(
    const DdlaHandle_t& handle, const T1& alpha, T2* A, const int* descA, int n)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pdam<DdlaBackend::GPU, T1, T2>(handle, alpha, A, descA, n);
#endif
    return pdam<DdlaBackend::CPU, T1, T2>(handle, alpha, A, descA, n);
}


template void pdam<float, float>(const DdlaHandle_t& handle, const float& alpha, float* A, const int* descA, int n);
template void pdam<double, double>(const DdlaHandle_t& handle, const double& alpha, double* A, const int* descA, int n);
template void pdam<float, std::complex<float>>(const DdlaHandle_t& handle, const float& alpha, std::complex<float>* A, const int* descA, int n);
template void pdam<std::complex<float>, std::complex<float>>(const DdlaHandle_t& handle, const std::complex<float>& alpha, std::complex<float>* A, const int* descA, int n);
template void pdam<double, std::complex<double>>(const DdlaHandle_t& handle, const double& alpha, std::complex<double>* A, const int* descA, int n);
template void pdam<std::complex<double>, std::complex<double>>(const DdlaHandle_t& handle, const std::complex<double>& alpha, std::complex<double>* A, const int* descA, int n);

template <typename T>
void ppotrf(
    const DdlaHandle_t& handle, char uplo, int n, T* A, int ia, int ja, const int* descA, int& info)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return ppotrf<DdlaBackend::GPU, T>(handle, uplo, n, A, ia, ja, descA, info);
#endif
    return ppotrf<DdlaBackend::CPU, T>(handle, uplo, n, A, ia, ja, descA, info);
}


template void ppotrf<float>(const DdlaHandle_t& handle, char uplo, int n, float* A, int ia, int ja, const int* descA, int& info);
template void ppotrf<double>(const DdlaHandle_t& handle, char uplo, int n, double* A, int ia, int ja, const int* descA, int& info);
template void ppotrf<std::complex<float>>(const DdlaHandle_t& handle, char uplo, int n, std::complex<float>* A, int ia, int ja, const int* descA, int& info);
template void ppotrf<std::complex<double>>(const DdlaHandle_t& handle, char uplo, int n, std::complex<double>* A, int ia, int ja, const int* descA, int& info);

template <typename T>
void potrf_bottom_right(
    char uplo, int n, T* A, int lda, int& info, const DdlaHandle_t& handle)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return potrf_bottom_right<DdlaBackend::GPU, T>(uplo, n, A, lda, info, handle);
#endif
    return potrf_bottom_right<DdlaBackend::CPU, T>(uplo, n, A, lda, info, handle);
}


template void potrf_bottom_right<float>(char uplo, int n, float* A, int lda, int& info, const DdlaHandle_t& handle);
template void potrf_bottom_right<double>(char uplo, int n, double* A, int lda, int& info, const DdlaHandle_t& handle);
template void potrf_bottom_right<std::complex<float>>(char uplo, int n, std::complex<float>* A, int lda, int& info, const DdlaHandle_t& handle);
template void potrf_bottom_right<std::complex<double>>(char uplo, int n, std::complex<double>* A, int lda, int& info, const DdlaHandle_t& handle);

template <typename T>
void ppotrf_bottom_right(
    const DdlaHandle_t& handle, char uplo, int n, T* A, const int* descA, int& info)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return ppotrf_bottom_right<DdlaBackend::GPU, T>(handle, uplo, n, A, descA, info);
#endif
    return ppotrf_bottom_right<DdlaBackend::CPU, T>(handle, uplo, n, A, descA, info);
}


template void ppotrf_bottom_right<float>(const DdlaHandle_t& handle, char uplo, int n, float* A, const int* descA, int& info);
template void ppotrf_bottom_right<double>(const DdlaHandle_t& handle, char uplo, int n, double* A, const int* descA, int& info);
template void ppotrf_bottom_right<std::complex<float>>(const DdlaHandle_t& handle, char uplo, int n, std::complex<float>* A, const int* descA, int& info);
template void ppotrf_bottom_right<std::complex<double>>(const DdlaHandle_t& handle, char uplo, int n, std::complex<double>* A, const int* descA, int& info);

template <typename T>
void ppotrs(
    const DdlaHandle_t& handle, char side, char uplo, char trans, int n, int nrhs, const T* A, const int* descA, T* B, const int* descB)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return ppotrs<DdlaBackend::GPU, T>(handle, side, uplo, trans, n, nrhs, A, descA, B, descB);
#endif
    return ppotrs<DdlaBackend::CPU, T>(handle, side, uplo, trans, n, nrhs, A, descA, B, descB);
}


template void ppotrs<float>(const DdlaHandle_t& handle, char side, char uplo, char trans, int n, int nrhs, const float* A, const int* descA, float* B, const int* descB);
template void ppotrs<double>(const DdlaHandle_t& handle, char side, char uplo, char trans, int n, int nrhs, const double* A, const int* descA, double* B, const int* descB);
template void ppotrs<std::complex<float>>(const DdlaHandle_t& handle, char side, char uplo, char trans, int n, int nrhs, const std::complex<float>* A, const int* descA, std::complex<float>* B, const int* descB);
template void ppotrs<std::complex<double>>(const DdlaHandle_t& handle, char side, char uplo, char trans, int n, int nrhs, const std::complex<double>* A, const int* descA, std::complex<double>* B, const int* descB);

template <typename T>
void pposv(
    const DdlaHandle_t& handle, char side, char uplo, char trans, int n, int nrhs, T* A, int ia, int ja, const int* descA, T* B, int ib, int jb, const int* descB, int& info)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pposv<DdlaBackend::GPU, T>(handle, side, uplo, trans, n, nrhs, A, ia, ja, descA, B, ib, jb, descB, info);
#endif
    return pposv<DdlaBackend::CPU, T>(handle, side, uplo, trans, n, nrhs, A, ia, ja, descA, B, ib, jb, descB, info);
}


template void pposv<float>(const DdlaHandle_t& handle, char side, char uplo, char trans, int n, int nrhs, float* A, int ia, int ja, const int* descA, float* B, int ib, int jb, const int* descB, int& info);
template void pposv<double>(const DdlaHandle_t& handle, char side, char uplo, char trans, int n, int nrhs, double* A, int ia, int ja, const int* descA, double* B, int ib, int jb, const int* descB, int& info);
template void pposv<std::complex<float>>(const DdlaHandle_t& handle, char side, char uplo, char trans, int n, int nrhs, std::complex<float>* A, int ia, int ja, const int* descA, std::complex<float>* B, int ib, int jb, const int* descB, int& info);
template void pposv<std::complex<double>>(const DdlaHandle_t& handle, char side, char uplo, char trans, int n, int nrhs, std::complex<double>* A, int ia, int ja, const int* descA, std::complex<double>* B, int ib, int jb, const int* descB, int& info);

template <typename T>
void gemm(
    const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const T& alpha, const T* A, int lda, const T* B, int ldb, const T& beta, T* C, int ldc)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return gemm<DdlaBackend::GPU, T>(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
    return gemm<DdlaBackend::CPU, T>(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}


template void gemm<float>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const float& alpha, const float* A, int lda, const float* B, int ldb, const float& beta, float* C, int ldc);
template void gemm<double>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const double& alpha, const double* A, int lda, const double* B, int ldb, const double& beta, double* C, int ldc);
template void gemm<std::complex<float>>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const std::complex<float>& alpha, const std::complex<float>* A, int lda, const std::complex<float>* B, int ldb, const std::complex<float>& beta, std::complex<float>* C, int ldc);
template void gemm<std::complex<double>>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const std::complex<double>& alpha, const std::complex<double>* A, int lda, const std::complex<double>* B, int ldb, const std::complex<double>& beta, std::complex<double>* C, int ldc);

template <typename R>
void gemm(
    const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const std::complex<R>& alpha, const R* A, int lda, const std::complex<R>* B, int ldb, const std::complex<R>& beta, std::complex<R>* C, int ldc)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return gemm<DdlaBackend::GPU, R>(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
    return gemm<DdlaBackend::CPU, R>(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}


template void gemm<float>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const std::complex<float>& alpha, const float* A, int lda, const std::complex<float>* B, int ldb, const std::complex<float>& beta, std::complex<float>* C, int ldc);
template void gemm<double>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const std::complex<double>& alpha, const double* A, int lda, const std::complex<double>* B, int ldb, const std::complex<double>& beta, std::complex<double>* C, int ldc);

template <typename T>
void herk(
    const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const typename real_type<T>::type& alpha, const T* A, int lda, const typename real_type<T>::type& beta, T* C, int ldc)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return herk<DdlaBackend::GPU, T>(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
#endif
    return herk<DdlaBackend::CPU, T>(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}


template void herk<float>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const typename real_type<float>::type& alpha, const float* A, int lda, const typename real_type<float>::type& beta, float* C, int ldc);
template void herk<double>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const typename real_type<double>::type& alpha, const double* A, int lda, const typename real_type<double>::type& beta, double* C, int ldc);
template void herk<std::complex<float>>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const typename real_type<std::complex<float>>::type& alpha, const std::complex<float>* A, int lda, const typename real_type<std::complex<float>>::type& beta, std::complex<float>* C, int ldc);
template void herk<std::complex<double>>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const typename real_type<std::complex<double>>::type& alpha, const std::complex<double>* A, int lda, const typename real_type<std::complex<double>>::type& beta, std::complex<double>* C, int ldc);

template <typename T>
void her2k(
    const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const T& alpha, const T* A, int lda, const T* B, int ldb, const typename real_type<T>::type& beta, T* C, int ldc)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return her2k<DdlaBackend::GPU, T>(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
    return her2k<DdlaBackend::CPU, T>(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}


template void her2k<float>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const float& alpha, const float* A, int lda, const float* B, int ldb, const typename real_type<float>::type& beta, float* C, int ldc);
template void her2k<double>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const double& alpha, const double* A, int lda, const double* B, int ldb, const typename real_type<double>::type& beta, double* C, int ldc);
template void her2k<std::complex<float>>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const std::complex<float>& alpha, const std::complex<float>* A, int lda, const std::complex<float>* B, int ldb, const typename real_type<std::complex<float>>::type& beta, std::complex<float>* C, int ldc);
template void her2k<std::complex<double>>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const std::complex<double>& alpha, const std::complex<double>* A, int lda, const std::complex<double>* B, int ldb, const typename real_type<std::complex<double>>::type& beta, std::complex<double>* C, int ldc);

template <typename T>
void hemv(
    const DdlaHandle_t& handle, char uplo, int n, const T& alpha, const T* A, int lda, const T* x, int incx, const T& beta, T* y, int incy)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return hemv<DdlaBackend::GPU, T>(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
#endif
    return hemv<DdlaBackend::CPU, T>(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}


template void hemv<float>(const DdlaHandle_t& handle, char uplo, int n, const float& alpha, const float* A, int lda, const float* x, int incx, const float& beta, float* y, int incy);
template void hemv<double>(const DdlaHandle_t& handle, char uplo, int n, const double& alpha, const double* A, int lda, const double* x, int incx, const double& beta, double* y, int incy);
template void hemv<std::complex<float>>(const DdlaHandle_t& handle, char uplo, int n, const std::complex<float>& alpha, const std::complex<float>* A, int lda, const std::complex<float>* x, int incx, const std::complex<float>& beta, std::complex<float>* y, int incy);
template void hemv<std::complex<double>>(const DdlaHandle_t& handle, char uplo, int n, const std::complex<double>& alpha, const std::complex<double>* A, int lda, const std::complex<double>* x, int incx, const std::complex<double>& beta, std::complex<double>* y, int incy);

template <typename T>
void hemm(
    const DdlaHandle_t& handle, char side, char uplo, int m, int n, const T& alpha, const T* A, int lda, const T* B, int ldb, const T& beta, T* C, int ldc)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return hemm<DdlaBackend::GPU, T>(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
    return hemm<DdlaBackend::CPU, T>(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}


template void hemm<float>(const DdlaHandle_t& handle, char side, char uplo, int m, int n, const float& alpha, const float* A, int lda, const float* B, int ldb, const float& beta, float* C, int ldc);
template void hemm<double>(const DdlaHandle_t& handle, char side, char uplo, int m, int n, const double& alpha, const double* A, int lda, const double* B, int ldb, const double& beta, double* C, int ldc);
template void hemm<std::complex<float>>(const DdlaHandle_t& handle, char side, char uplo, int m, int n, const std::complex<float>& alpha, const std::complex<float>* A, int lda, const std::complex<float>* B, int ldb, const std::complex<float>& beta, std::complex<float>* C, int ldc);
template void hemm<std::complex<double>>(const DdlaHandle_t& handle, char side, char uplo, int m, int n, const std::complex<double>& alpha, const std::complex<double>* A, int lda, const std::complex<double>* B, int ldb, const std::complex<double>& beta, std::complex<double>* C, int ldc);

template <typename R>
void hemm(
    const DdlaHandle_t& handle, char side, char uplo, int m, int n, const std::complex<R>& alpha, const R* A, int lda, const std::complex<R>* B, int ldb, const std::complex<R>& beta, std::complex<R>* C, int ldc)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return hemm<DdlaBackend::GPU, R>(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
    return hemm<DdlaBackend::CPU, R>(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}


template void hemm<float>(const DdlaHandle_t& handle, char side, char uplo, int m, int n, const std::complex<float>& alpha, const float* A, int lda, const std::complex<float>* B, int ldb, const std::complex<float>& beta, std::complex<float>* C, int ldc);
template void hemm<double>(const DdlaHandle_t& handle, char side, char uplo, int m, int n, const std::complex<double>& alpha, const double* A, int lda, const std::complex<double>* B, int ldb, const std::complex<double>& beta, std::complex<double>* C, int ldc);

template <typename R>
void hemm(
    const DdlaHandle_t& handle, char side, char uplo, int m, int n, const std::complex<R>& alpha, const std::complex<R>* A, int lda, const R* B, int ldb, const std::complex<R>& beta, std::complex<R>* C, int ldc)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return hemm<DdlaBackend::GPU, R>(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
    return hemm<DdlaBackend::CPU, R>(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}


template void hemm<float>(const DdlaHandle_t& handle, char side, char uplo, int m, int n, const std::complex<float>& alpha, const std::complex<float>* A, int lda, const float* B, int ldb, const std::complex<float>& beta, std::complex<float>* C, int ldc);
template void hemm<double>(const DdlaHandle_t& handle, char side, char uplo, int m, int n, const std::complex<double>& alpha, const std::complex<double>* A, int lda, const double* B, int ldb, const std::complex<double>& beta, std::complex<double>* C, int ldc);

template <typename T>
void hemmt(
    const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const T& alpha, const T* A, int lda, const T* B, int ldb, const T& beta, T* C, int ldc)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return hemmt<DdlaBackend::GPU, T>(handle, side, uploA, uploC, n, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
    return hemmt<DdlaBackend::CPU, T>(handle, side, uploA, uploC, n, alpha, A, lda, B, ldb, beta, C, ldc);
}


template void hemmt<float>(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const float& alpha, const float* A, int lda, const float* B, int ldb, const float& beta, float* C, int ldc);
template void hemmt<double>(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const double& alpha, const double* A, int lda, const double* B, int ldb, const double& beta, double* C, int ldc);
template void hemmt<std::complex<float>>(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const std::complex<float>& alpha, const std::complex<float>* A, int lda, const std::complex<float>* B, int ldb, const std::complex<float>& beta, std::complex<float>* C, int ldc);
template void hemmt<std::complex<double>>(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const std::complex<double>& alpha, const std::complex<double>* A, int lda, const std::complex<double>* B, int ldb, const std::complex<double>& beta, std::complex<double>* C, int ldc);

template <typename R>
void hemmt(
    const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const std::complex<R>& alpha, const R* A, int lda, const std::complex<R>* B, int ldb, const std::complex<R>& beta, std::complex<R>* C, int ldc)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return hemmt<DdlaBackend::GPU, R>(handle, side, uploA, uploC, n, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
    return hemmt<DdlaBackend::CPU, R>(handle, side, uploA, uploC, n, alpha, A, lda, B, ldb, beta, C, ldc);
}


template void hemmt<float>(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const std::complex<float>& alpha, const float* A, int lda, const std::complex<float>* B, int ldb, const std::complex<float>& beta, std::complex<float>* C, int ldc);
template void hemmt<double>(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const std::complex<double>& alpha, const double* A, int lda, const std::complex<double>* B, int ldb, const std::complex<double>& beta, std::complex<double>* C, int ldc);

template <typename R>
void hemmt(
    const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const std::complex<R>& alpha, const std::complex<R>* A, int lda, const R* B, int ldb, const std::complex<R>& beta, std::complex<R>* C, int ldc)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return hemmt<DdlaBackend::GPU, R>(handle, side, uploA, uploC, n, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
    return hemmt<DdlaBackend::CPU, R>(handle, side, uploA, uploC, n, alpha, A, lda, B, ldb, beta, C, ldc);
}


template void hemmt<float>(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const std::complex<float>& alpha, const std::complex<float>* A, int lda, const float* B, int ldb, const std::complex<float>& beta, std::complex<float>* C, int ldc);
template void hemmt<double>(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const std::complex<double>& alpha, const std::complex<double>* A, int lda, const double* B, int ldb, const std::complex<double>& beta, std::complex<double>* C, int ldc);

template <typename T>
void symmt(
    const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const T& alpha, const T* A, int lda, const T* B, int ldb, const T& beta, T* C, int ldc)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return symmt<DdlaBackend::GPU, T>(handle, side, uploA, uploC, n, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
    return symmt<DdlaBackend::CPU, T>(handle, side, uploA, uploC, n, alpha, A, lda, B, ldb, beta, C, ldc);
}


template void symmt<float>(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const float& alpha, const float* A, int lda, const float* B, int ldb, const float& beta, float* C, int ldc);
template void symmt<double>(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const double& alpha, const double* A, int lda, const double* B, int ldb, const double& beta, double* C, int ldc);
template void symmt<std::complex<float>>(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const std::complex<float>& alpha, const std::complex<float>* A, int lda, const std::complex<float>* B, int ldb, const std::complex<float>& beta, std::complex<float>* C, int ldc);
template void symmt<std::complex<double>>(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n, const std::complex<double>& alpha, const std::complex<double>* A, int lda, const std::complex<double>* B, int ldb, const std::complex<double>& beta, std::complex<double>* C, int ldc);

template <typename T>
void her2(
    const DdlaHandle_t& handle, char uplo, int n, const T& alpha, const T* x, int incx, const T* y, int incy, T* A, int lda)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return her2<DdlaBackend::GPU, T>(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
#endif
    return her2<DdlaBackend::CPU, T>(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
}


template void her2<float>(const DdlaHandle_t& handle, char uplo, int n, const float& alpha, const float* x, int incx, const float* y, int incy, float* A, int lda);
template void her2<double>(const DdlaHandle_t& handle, char uplo, int n, const double& alpha, const double* x, int incx, const double* y, int incy, double* A, int lda);
template void her2<std::complex<float>>(const DdlaHandle_t& handle, char uplo, int n, const std::complex<float>& alpha, const std::complex<float>* x, int incx, const std::complex<float>* y, int incy, std::complex<float>* A, int lda);
template void her2<std::complex<double>>(const DdlaHandle_t& handle, char uplo, int n, const std::complex<double>& alpha, const std::complex<double>* x, int incx, const std::complex<double>* y, int incy, std::complex<double>* A, int lda);

template <typename T>
void gemv(
    const DdlaHandle_t& handle, char trans, int m, int n, const T& alpha, const T* A, int lda, const T* x, int incx, const T& beta, T* y, int incy)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return gemv<DdlaBackend::GPU, T>(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
#endif
    return gemv<DdlaBackend::CPU, T>(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}


template void gemv<float>(const DdlaHandle_t& handle, char trans, int m, int n, const float& alpha, const float* A, int lda, const float* x, int incx, const float& beta, float* y, int incy);
template void gemv<double>(const DdlaHandle_t& handle, char trans, int m, int n, const double& alpha, const double* A, int lda, const double* x, int incx, const double& beta, double* y, int incy);
template void gemv<std::complex<float>>(const DdlaHandle_t& handle, char trans, int m, int n, const std::complex<float>& alpha, const std::complex<float>* A, int lda, const std::complex<float>* x, int incx, const std::complex<float>& beta, std::complex<float>* y, int incy);
template void gemv<std::complex<double>>(const DdlaHandle_t& handle, char trans, int m, int n, const std::complex<double>& alpha, const std::complex<double>* A, int lda, const std::complex<double>* x, int incx, const std::complex<double>& beta, std::complex<double>* y, int incy);

template <typename T>
void trmv(
    const DdlaHandle_t& handle, char uplo, char trans, char diag, int n, const T* A, int lda, T* x, int incx)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return trmv<DdlaBackend::GPU, T>(handle, uplo, trans, diag, n, A, lda, x, incx);
#endif
    return trmv<DdlaBackend::CPU, T>(handle, uplo, trans, diag, n, A, lda, x, incx);
}


template void trmv<float>(const DdlaHandle_t& handle, char uplo, char trans, char diag, int n, const float* A, int lda, float* x, int incx);
template void trmv<double>(const DdlaHandle_t& handle, char uplo, char trans, char diag, int n, const double* A, int lda, double* x, int incx);
template void trmv<std::complex<float>>(const DdlaHandle_t& handle, char uplo, char trans, char diag, int n, const std::complex<float>* A, int lda, std::complex<float>* x, int incx);
template void trmv<std::complex<double>>(const DdlaHandle_t& handle, char uplo, char trans, char diag, int n, const std::complex<double>* A, int lda, std::complex<double>* x, int incx);

template <typename T>
void nrm2(
    const DdlaHandle_t& handle, int n, const T* x, int incx, typename real_type<T>::type* result)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return nrm2<DdlaBackend::GPU, T>(handle, n, x, incx, result);
#endif
    return nrm2<DdlaBackend::CPU, T>(handle, n, x, incx, result);
}


template void nrm2<float>(const DdlaHandle_t& handle, int n, const float* x, int incx, typename real_type<float>::type* result);
template void nrm2<double>(const DdlaHandle_t& handle, int n, const double* x, int incx, typename real_type<double>::type* result);
template void nrm2<std::complex<float>>(const DdlaHandle_t& handle, int n, const std::complex<float>* x, int incx, typename real_type<std::complex<float>>::type* result);
template void nrm2<std::complex<double>>(const DdlaHandle_t& handle, int n, const std::complex<double>* x, int incx, typename real_type<std::complex<double>>::type* result);

template <typename T>
void dotc(
    const DdlaHandle_t& handle, int n, const T* x, int incx, const T* y, int incy, T* result)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return dotc<DdlaBackend::GPU, T>(handle, n, x, incx, y, incy, result);
#endif
    return dotc<DdlaBackend::CPU, T>(handle, n, x, incx, y, incy, result);
}


template void dotc<float>(const DdlaHandle_t& handle, int n, const float* x, int incx, const float* y, int incy, float* result);
template void dotc<double>(const DdlaHandle_t& handle, int n, const double* x, int incx, const double* y, int incy, double* result);
template void dotc<std::complex<float>>(const DdlaHandle_t& handle, int n, const std::complex<float>* x, int incx, const std::complex<float>* y, int incy, std::complex<float>* result);
template void dotc<std::complex<double>>(const DdlaHandle_t& handle, int n, const std::complex<double>* x, int incx, const std::complex<double>* y, int incy, std::complex<double>* result);

template <typename T>
void trsm(
    const DdlaHandle_t& handle, char side, char uplo, char trans, char diag, int m, int n, const T& alpha, T* A, int lda, T* B, int ldb)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return trsm<DdlaBackend::GPU, T>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
#endif
    return trsm<DdlaBackend::CPU, T>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}


template void trsm<float>(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag, int m, int n, const float& alpha, float* A, int lda, float* B, int ldb);
template void trsm<double>(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag, int m, int n, const double& alpha, double* A, int lda, double* B, int ldb);
template void trsm<std::complex<float>>(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag, int m, int n, const std::complex<float>& alpha, std::complex<float>* A, int lda, std::complex<float>* B, int ldb);
template void trsm<std::complex<double>>(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag, int m, int n, const std::complex<double>& alpha, std::complex<double>* A, int lda, std::complex<double>* B, int ldb);

template <typename T>
void syrk(
    const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const T& alpha, const T* A, int lda, const T& beta, T* C, int ldc)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return syrk<DdlaBackend::GPU, T>(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
#endif
    return syrk<DdlaBackend::CPU, T>(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}


template void syrk<float>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const float& alpha, const float* A, int lda, const float& beta, float* C, int ldc);
template void syrk<double>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const double& alpha, const double* A, int lda, const double& beta, double* C, int ldc);
template void syrk<std::complex<float>>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const std::complex<float>& alpha, const std::complex<float>* A, int lda, const std::complex<float>& beta, std::complex<float>* C, int ldc);
template void syrk<std::complex<double>>(const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const std::complex<double>& alpha, const std::complex<double>* A, int lda, const std::complex<double>& beta, std::complex<double>* C, int ldc);

template <typename T>
void trmm(
    const DdlaHandle_t& handle, char side, char uplo, char trans, char diag, int m, int n, const T& alpha, const T* A, int lda, T* B, int ldb)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return trmm<DdlaBackend::GPU, T>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
#endif
    return trmm<DdlaBackend::CPU, T>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}


template void trmm<float>(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag, int m, int n, const float& alpha, const float* A, int lda, float* B, int ldb);
template void trmm<double>(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag, int m, int n, const double& alpha, const double* A, int lda, double* B, int ldb);
template void trmm<std::complex<float>>(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag, int m, int n, const std::complex<float>& alpha, const std::complex<float>* A, int lda, std::complex<float>* B, int ldb);
template void trmm<std::complex<double>>(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag, int m, int n, const std::complex<double>& alpha, const std::complex<double>* A, int lda, std::complex<double>* B, int ldb);

template <typename T>
void trtrm(
    const DdlaHandle_t& handle, char uplo_a, char transa, char uplo_b, char transb, int n, const T& alpha, const T* A, int lda, const T* B, int ldb, const T& beta, T* C, int ldc)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return trtrm<DdlaBackend::GPU, T>(handle, uplo_a, transa, uplo_b, transb, n, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
    return trtrm<DdlaBackend::CPU, T>(handle, uplo_a, transa, uplo_b, transb, n, alpha, A, lda, B, ldb, beta, C, ldc);
}


template void trtrm<float>(const DdlaHandle_t& handle, char uplo_a, char transa, char uplo_b, char transb, int n, const float& alpha, const float* A, int lda, const float* B, int ldb, const float& beta, float* C, int ldc);
template void trtrm<double>(const DdlaHandle_t& handle, char uplo_a, char transa, char uplo_b, char transb, int n, const double& alpha, const double* A, int lda, const double* B, int ldb, const double& beta, double* C, int ldc);
template void trtrm<std::complex<float>>(const DdlaHandle_t& handle, char uplo_a, char transa, char uplo_b, char transb, int n, const std::complex<float>& alpha, const std::complex<float>* A, int lda, const std::complex<float>* B, int ldb, const std::complex<float>& beta, std::complex<float>* C, int ldc);
template void trtrm<std::complex<double>>(const DdlaHandle_t& handle, char uplo_a, char transa, char uplo_b, char transb, int n, const std::complex<double>& alpha, const std::complex<double>* A, int lda, const std::complex<double>* B, int ldb, const std::complex<double>& beta, std::complex<double>* C, int ldc);

template <typename T>
void gemmBatched(
    const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const T& alpha, const T* const Aarray[], int lda, const T* const Barray[], int ldb, const T& beta, T* const Carray[], int ldc, int batchCount)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return gemmBatched<DdlaBackend::GPU, T>(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
#endif
    return gemmBatched<DdlaBackend::CPU, T>(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}


template void gemmBatched<float>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const float& alpha, const float* const Aarray[], int lda, const float* const Barray[], int ldb, const float& beta, float* const Carray[], int ldc, int batchCount);
template void gemmBatched<double>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const double& alpha, const double* const Aarray[], int lda, const double* const Barray[], int ldb, const double& beta, double* const Carray[], int ldc, int batchCount);
template void gemmBatched<std::complex<float>>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const std::complex<float>& alpha, const std::complex<float>* const Aarray[], int lda, const std::complex<float>* const Barray[], int ldb, const std::complex<float>& beta, std::complex<float>* const Carray[], int ldc, int batchCount);
template void gemmBatched<std::complex<double>>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const std::complex<double>& alpha, const std::complex<double>* const Aarray[], int lda, const std::complex<double>* const Barray[], int ldb, const std::complex<double>& beta, std::complex<double>* const Carray[], int ldc, int batchCount);

template <typename R>
void gemmBatched(
    const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const std::complex<R>& alpha, const R* const Aarray[], int lda, const std::complex<R>* const Barray[], int ldb, const std::complex<R>& beta, std::complex<R>* const Carray[], int ldc, int batchCount)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return gemmBatched<DdlaBackend::GPU, R>(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
#endif
    return gemmBatched<DdlaBackend::CPU, R>(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}


template void gemmBatched<float>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const std::complex<float>& alpha, const float* const Aarray[], int lda, const std::complex<float>* const Barray[], int ldb, const std::complex<float>& beta, std::complex<float>* const Carray[], int ldc, int batchCount);
template void gemmBatched<double>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const std::complex<double>& alpha, const double* const Aarray[], int lda, const std::complex<double>* const Barray[], int ldb, const std::complex<double>& beta, std::complex<double>* const Carray[], int ldc, int batchCount);

template <typename R>
void gemmBatched(
    const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const std::complex<R>& alpha, const std::complex<R>* const Aarray[], int lda, const R* const Barray[], int ldb, const std::complex<R>& beta, std::complex<R>* const Carray[], int ldc, int batchCount)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return gemmBatched<DdlaBackend::GPU, R>(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
#endif
    return gemmBatched<DdlaBackend::CPU, R>(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}


template void gemmBatched<float>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const std::complex<float>& alpha, const std::complex<float>* const Aarray[], int lda, const float* const Barray[], int ldb, const std::complex<float>& beta, std::complex<float>* const Carray[], int ldc, int batchCount);
template void gemmBatched<double>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const std::complex<double>& alpha, const std::complex<double>* const Aarray[], int lda, const double* const Barray[], int ldb, const std::complex<double>& beta, std::complex<double>* const Carray[], int ldc, int batchCount);

template <typename T>
void geam(
    const DdlaHandle_t& handle, char transa, char transb, int m, int n, const T& alpha, const T* A, int lda, const T& beta, const T* B, int ldb, T* C, int ldc)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return geam<DdlaBackend::GPU, T>(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
#endif
    return geam<DdlaBackend::CPU, T>(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}


template void geam<float>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, const float& alpha, const float* A, int lda, const float& beta, const float* B, int ldb, float* C, int ldc);
template void geam<double>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, const double& alpha, const double* A, int lda, const double& beta, const double* B, int ldb, double* C, int ldc);
template void geam<std::complex<float>>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, const std::complex<float>& alpha, const std::complex<float>* A, int lda, const std::complex<float>& beta, const std::complex<float>* B, int ldb, std::complex<float>* C, int ldc);
template void geam<std::complex<double>>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, const std::complex<double>& alpha, const std::complex<double>* A, int lda, const std::complex<double>& beta, const std::complex<double>* B, int ldb, std::complex<double>* C, int ldc);

template <typename T>
void gemmVbatched(
    const DdlaHandle_t& handle, char transA, char transB, int* d_m, int* d_n, int* d_k, T alpha, const T* const* d_A_array, int* d_lda, const T* const* d_B_array, int* d_ldb, T beta, T** d_C_array, int* d_ldc, int batch_count)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return gemmVbatched<DdlaBackend::GPU, T>(handle, transA, transB, d_m, d_n, d_k, alpha, d_A_array, d_lda, d_B_array, d_ldb, beta, d_C_array, d_ldc, batch_count);
    return gemmVbatched<DdlaBackend::CPU, T>(handle, transA, transB, d_m, d_n, d_k, alpha, d_A_array, d_lda, d_B_array, d_ldb, beta, d_C_array, d_ldc, batch_count);
#else
    if (!handle)
        throw std::invalid_argument("gemmVbatched: invalid DDLA handle");
    throw std::invalid_argument("gemmVbatched: only the GPU backend is implemented");
#endif
}


template void gemmVbatched<float>(const DdlaHandle_t& handle, char transA, char transB, int* d_m, int* d_n, int* d_k, float alpha, const float* const* d_A_array, int* d_lda, const float* const* d_B_array, int* d_ldb, float beta, float** d_C_array, int* d_ldc, int batch_count);
template void gemmVbatched<double>(const DdlaHandle_t& handle, char transA, char transB, int* d_m, int* d_n, int* d_k, double alpha, const double* const* d_A_array, int* d_lda, const double* const* d_B_array, int* d_ldb, double beta, double** d_C_array, int* d_ldc, int batch_count);
template void gemmVbatched<std::complex<float>>(const DdlaHandle_t& handle, char transA, char transB, int* d_m, int* d_n, int* d_k, std::complex<float> alpha, const std::complex<float>* const* d_A_array, int* d_lda, const std::complex<float>* const* d_B_array, int* d_ldb, std::complex<float> beta, std::complex<float>** d_C_array, int* d_ldc, int batch_count);
template void gemmVbatched<std::complex<double>>(const DdlaHandle_t& handle, char transA, char transB, int* d_m, int* d_n, int* d_k, std::complex<double> alpha, const std::complex<double>* const* d_A_array, int* d_lda, const std::complex<double>* const* d_B_array, int* d_ldb, std::complex<double> beta, std::complex<double>** d_C_array, int* d_ldc, int batch_count);

template <typename T>
void gemmVbatched2s(
    const DdlaHandle_t& handle, char transA_0, char transB_0, int* d_m_0, int* d_n_0, int* d_k_0, T alpha_0, const T* const* d_A_array_0, int* d_lda_0, const T* const* d_B_array_0, int* d_ldb_0, T beta_0, T** d_C_array_0, int* d_ldc_0, char transA_1, char transB_1, int* d_m_1, int* d_n_1, int* d_k_1, T alpha_1, const T* const* d_AB_array_1, int* d_lda_1, int* d_ldb_1, T beta_1, T** d_C_array_1, int* d_ldc_1, bool C0_left, int batch_count, const int* segment_sizes, int segment_count)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return gemmVbatched2s<DdlaBackend::GPU, T>(handle, transA_0, transB_0, d_m_0, d_n_0, d_k_0, alpha_0, d_A_array_0, d_lda_0, d_B_array_0, d_ldb_0, beta_0, d_C_array_0, d_ldc_0, transA_1, transB_1, d_m_1, d_n_1, d_k_1, alpha_1, d_AB_array_1, d_lda_1, d_ldb_1, beta_1, d_C_array_1, d_ldc_1, C0_left, batch_count, segment_sizes, segment_count);
    return gemmVbatched2s<DdlaBackend::CPU, T>(handle, transA_0, transB_0, d_m_0, d_n_0, d_k_0, alpha_0, d_A_array_0, d_lda_0, d_B_array_0, d_ldb_0, beta_0, d_C_array_0, d_ldc_0, transA_1, transB_1, d_m_1, d_n_1, d_k_1, alpha_1, d_AB_array_1, d_lda_1, d_ldb_1, beta_1, d_C_array_1, d_ldc_1, C0_left, batch_count, segment_sizes, segment_count);
#else
    if (!handle)
        throw std::invalid_argument("gemmVbatched2s: invalid DDLA handle");
    throw std::invalid_argument("gemmVbatched2s: only the GPU backend is implemented");
#endif
}


template void gemmVbatched2s<float>(const DdlaHandle_t& handle, char transA_0, char transB_0, int* d_m_0, int* d_n_0, int* d_k_0, float alpha_0, const float* const* d_A_array_0, int* d_lda_0, const float* const* d_B_array_0, int* d_ldb_0, float beta_0, float** d_C_array_0, int* d_ldc_0, char transA_1, char transB_1, int* d_m_1, int* d_n_1, int* d_k_1, float alpha_1, const float* const* d_AB_array_1, int* d_lda_1, int* d_ldb_1, float beta_1, float** d_C_array_1, int* d_ldc_1, bool C0_left, int batch_count, const int* segment_sizes, int segment_count);
template void gemmVbatched2s<double>(const DdlaHandle_t& handle, char transA_0, char transB_0, int* d_m_0, int* d_n_0, int* d_k_0, double alpha_0, const double* const* d_A_array_0, int* d_lda_0, const double* const* d_B_array_0, int* d_ldb_0, double beta_0, double** d_C_array_0, int* d_ldc_0, char transA_1, char transB_1, int* d_m_1, int* d_n_1, int* d_k_1, double alpha_1, const double* const* d_AB_array_1, int* d_lda_1, int* d_ldb_1, double beta_1, double** d_C_array_1, int* d_ldc_1, bool C0_left, int batch_count, const int* segment_sizes, int segment_count);
template void gemmVbatched2s<std::complex<float>>(const DdlaHandle_t& handle, char transA_0, char transB_0, int* d_m_0, int* d_n_0, int* d_k_0, std::complex<float> alpha_0, const std::complex<float>* const* d_A_array_0, int* d_lda_0, const std::complex<float>* const* d_B_array_0, int* d_ldb_0, std::complex<float> beta_0, std::complex<float>** d_C_array_0, int* d_ldc_0, char transA_1, char transB_1, int* d_m_1, int* d_n_1, int* d_k_1, std::complex<float> alpha_1, const std::complex<float>* const* d_AB_array_1, int* d_lda_1, int* d_ldb_1, std::complex<float> beta_1, std::complex<float>** d_C_array_1, int* d_ldc_1, bool C0_left, int batch_count, const int* segment_sizes, int segment_count);
template void gemmVbatched2s<std::complex<double>>(const DdlaHandle_t& handle, char transA_0, char transB_0, int* d_m_0, int* d_n_0, int* d_k_0, std::complex<double> alpha_0, const std::complex<double>* const* d_A_array_0, int* d_lda_0, const std::complex<double>* const* d_B_array_0, int* d_ldb_0, std::complex<double> beta_0, std::complex<double>** d_C_array_0, int* d_ldc_0, char transA_1, char transB_1, int* d_m_1, int* d_n_1, int* d_k_1, std::complex<double> alpha_1, const std::complex<double>* const* d_AB_array_1, int* d_lda_1, int* d_ldb_1, std::complex<double> beta_1, std::complex<double>** d_C_array_1, int* d_ldc_1, bool C0_left, int batch_count, const int* segment_sizes, int segment_count);

template <typename T>
void random_generate(
    const DdlaHandle_t& handle, T* data, const int64_t& lengthOfData)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return random_generate<DdlaBackend::GPU, T>(handle, data, lengthOfData);
#endif
    return random_generate<DdlaBackend::CPU, T>(handle, data, lengthOfData);
}


template void random_generate<float>(const DdlaHandle_t& handle, float* data, const int64_t& lengthOfData);
template void random_generate<double>(const DdlaHandle_t& handle, double* data, const int64_t& lengthOfData);
template void random_generate<std::complex<float>>(const DdlaHandle_t& handle, std::complex<float>* data, const int64_t& lengthOfData);
template void random_generate<std::complex<double>>(const DdlaHandle_t& handle, std::complex<double>* data, const int64_t& lengthOfData);

template <typename T>
void scal(
    const DdlaHandle_t& handle, int n, const T& alpha, T* x, int incx)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return scal<DdlaBackend::GPU, T>(handle, n, alpha, x, incx);
#endif
    return scal<DdlaBackend::CPU, T>(handle, n, alpha, x, incx);
}


template void scal<float>(const DdlaHandle_t& handle, int n, const float& alpha, float* x, int incx);
template void scal<double>(const DdlaHandle_t& handle, int n, const double& alpha, double* x, int incx);
template void scal<std::complex<float>>(const DdlaHandle_t& handle, int n, const std::complex<float>& alpha, std::complex<float>* x, int incx);
template void scal<std::complex<double>>(const DdlaHandle_t& handle, int n, const std::complex<double>& alpha, std::complex<double>* x, int incx);

template <typename T>
void scal(
    const DdlaHandle_t& handle, int m, int n, const T& alpha, T* A, int lda)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return scal<DdlaBackend::GPU, T>(handle, m, n, alpha, A, lda);
#endif
    return scal<DdlaBackend::CPU, T>(handle, m, n, alpha, A, lda);
}


template void scal<float>(const DdlaHandle_t& handle, int m, int n, const float& alpha, float* A, int lda);
template void scal<double>(const DdlaHandle_t& handle, int m, int n, const double& alpha, double* A, int lda);
template void scal<std::complex<float>>(const DdlaHandle_t& handle, int m, int n, const std::complex<float>& alpha, std::complex<float>* A, int lda);
template void scal<std::complex<double>>(const DdlaHandle_t& handle, int m, int n, const std::complex<double>& alpha, std::complex<double>* A, int lda);

template <typename T>
void pscal(
    const DdlaHandle_t& handle, int m, int n, const T& alpha, T* A, const int* descA)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pscal<DdlaBackend::GPU, T>(handle, m, n, alpha, A, descA);
#endif
    return pscal<DdlaBackend::CPU, T>(handle, m, n, alpha, A, descA);
}


template void pscal<float>(const DdlaHandle_t& handle, int m, int n, const float& alpha, float* A, const int* descA);
template void pscal<double>(const DdlaHandle_t& handle, int m, int n, const double& alpha, double* A, const int* descA);
template void pscal<std::complex<float>>(const DdlaHandle_t& handle, int m, int n, const std::complex<float>& alpha, std::complex<float>* A, const int* descA);
template void pscal<std::complex<double>>(const DdlaHandle_t& handle, int m, int n, const std::complex<double>& alpha, std::complex<double>* A, const int* descA);

template <typename T>
void scalBatched(
    const DdlaHandle_t& handle, int m, int n, const T& alpha, T* const Aarray[], int lda, int batchCount)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return scalBatched<DdlaBackend::GPU, T>(handle, m, n, alpha, Aarray, lda, batchCount);
#endif
    return scalBatched<DdlaBackend::CPU, T>(handle, m, n, alpha, Aarray, lda, batchCount);
}


template void scalBatched<float>(const DdlaHandle_t& handle, int m, int n, const float& alpha, float* const Aarray[], int lda, int batchCount);
template void scalBatched<double>(const DdlaHandle_t& handle, int m, int n, const double& alpha, double* const Aarray[], int lda, int batchCount);
template void scalBatched<std::complex<float>>(const DdlaHandle_t& handle, int m, int n, const std::complex<float>& alpha, std::complex<float>* const Aarray[], int lda, int batchCount);
template void scalBatched<std::complex<double>>(const DdlaHandle_t& handle, int m, int n, const std::complex<double>& alpha, std::complex<double>* const Aarray[], int lda, int batchCount);

template <typename T, typename Alpha>
std::enable_if_t<std::is_same_v<T,std::complex<Alpha>>,void> scal(
    const DdlaHandle_t& h, int m, int n, const Alpha& alpha, T* A, int lda)
{
#if DDLA_HAS_GPU
    if (h && h->backend == DdlaBackend::GPU)
        return scal<DdlaBackend::GPU, T, Alpha>(h, m, n, alpha, A, lda);
#endif
    return scal<DdlaBackend::CPU, T, Alpha>(h, m, n, alpha, A, lda);
}


template std::enable_if_t<std::is_same_v<std::complex<float>,std::complex<float>>,void> scal<std::complex<float>, float>(const DdlaHandle_t& h, int m, int n, const float& alpha, std::complex<float>* A, int lda);
template std::enable_if_t<std::is_same_v<std::complex<double>,std::complex<double>>,void> scal<std::complex<double>, double>(const DdlaHandle_t& h, int m, int n, const double& alpha, std::complex<double>* A, int lda);

template <typename T, typename Alpha>
std::enable_if_t<std::is_same_v<T,std::complex<Alpha>>,void> pscal(
    const DdlaHandle_t& h, int m, int n, const Alpha& alpha, T* A, const int* descA)
{
#if DDLA_HAS_GPU
    if (h && h->backend == DdlaBackend::GPU)
        return pscal<DdlaBackend::GPU, T, Alpha>(h, m, n, alpha, A, descA);
#endif
    return pscal<DdlaBackend::CPU, T, Alpha>(h, m, n, alpha, A, descA);
}


template std::enable_if_t<std::is_same_v<std::complex<float>,std::complex<float>>,void> pscal<std::complex<float>, float>(const DdlaHandle_t& h, int m, int n, const float& alpha, std::complex<float>* A, const int* descA);
template std::enable_if_t<std::is_same_v<std::complex<double>,std::complex<double>>,void> pscal<std::complex<double>, double>(const DdlaHandle_t& h, int m, int n, const double& alpha, std::complex<double>* A, const int* descA);

template <typename T, typename Alpha>
std::enable_if_t<std::is_same_v<T,std::complex<Alpha>>,void> scalBatched(
    const DdlaHandle_t& h, int m, int n, const Alpha& alpha, T* const Aarray[], int lda, int count)
{
#if DDLA_HAS_GPU
    if (h && h->backend == DdlaBackend::GPU)
        return scalBatched<DdlaBackend::GPU, T, Alpha>(h, m, n, alpha, Aarray, lda, count);
#endif
    return scalBatched<DdlaBackend::CPU, T, Alpha>(h, m, n, alpha, Aarray, lda, count);
}


template std::enable_if_t<std::is_same_v<std::complex<float>,std::complex<float>>,void> scalBatched<std::complex<float>, float>(const DdlaHandle_t& h, int m, int n, const float& alpha, std::complex<float>* const Aarray[], int lda, int count);
template std::enable_if_t<std::is_same_v<std::complex<double>,std::complex<double>>,void> scalBatched<std::complex<double>, double>(const DdlaHandle_t& h, int m, int n, const double& alpha, std::complex<double>* const Aarray[], int lda, int count);

template <typename T>
void axpy(
    const DdlaHandle_t& handle, int n, const T& alpha, const T* x, int incx, T* y, int incy)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return axpy<DdlaBackend::GPU, T>(handle, n, alpha, x, incx, y, incy);
#endif
    return axpy<DdlaBackend::CPU, T>(handle, n, alpha, x, incx, y, incy);
}


template void axpy<float>(const DdlaHandle_t& handle, int n, const float& alpha, const float* x, int incx, float* y, int incy);
template void axpy<double>(const DdlaHandle_t& handle, int n, const double& alpha, const double* x, int incx, double* y, int incy);
template void axpy<std::complex<float>>(const DdlaHandle_t& handle, int n, const std::complex<float>& alpha, const std::complex<float>* x, int incx, std::complex<float>* y, int incy);
template void axpy<std::complex<double>>(const DdlaHandle_t& handle, int n, const std::complex<double>& alpha, const std::complex<double>* x, int incx, std::complex<double>* y, int incy);

template <typename T>
void iamax(
    const DdlaHandle_t& handle, int n, const T* x, int incx, int& result)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return iamax<DdlaBackend::GPU, T>(handle, n, x, incx, result);
#endif
    return iamax<DdlaBackend::CPU, T>(handle, n, x, incx, result);
}


template void iamax<float>(const DdlaHandle_t& handle, int n, const float* x, int incx, int& result);
template void iamax<double>(const DdlaHandle_t& handle, int n, const double* x, int incx, int& result);
template void iamax<std::complex<float>>(const DdlaHandle_t& handle, int n, const std::complex<float>* x, int incx, int& result);
template void iamax<std::complex<double>>(const DdlaHandle_t& handle, int n, const std::complex<double>* x, int incx, int& result);

template <typename T>
void swap(
    const DdlaHandle_t& handle, int n, T* x, int incx, T* y, int incy)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return swap<DdlaBackend::GPU, T>(handle, n, x, incx, y, incy);
#endif
    return swap<DdlaBackend::CPU, T>(handle, n, x, incx, y, incy);
}


template void swap<float>(const DdlaHandle_t& handle, int n, float* x, int incx, float* y, int incy);
template void swap<double>(const DdlaHandle_t& handle, int n, double* x, int incx, double* y, int incy);
template void swap<std::complex<float>>(const DdlaHandle_t& handle, int n, std::complex<float>* x, int incx, std::complex<float>* y, int incy);
template void swap<std::complex<double>>(const DdlaHandle_t& handle, int n, std::complex<double>* x, int incx, std::complex<double>* y, int incy);

template <typename T>
void geru(
    const DdlaHandle_t& handle, int m, int n, const T& alpha, const T* x, int incx, const T* y, int incy, T* A, int lda)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return geru<DdlaBackend::GPU, T>(handle, m, n, alpha, x, incx, y, incy, A, lda);
#endif
    return geru<DdlaBackend::CPU, T>(handle, m, n, alpha, x, incx, y, incy, A, lda);
}


template void geru<float>(const DdlaHandle_t& handle, int m, int n, const float& alpha, const float* x, int incx, const float* y, int incy, float* A, int lda);
template void geru<double>(const DdlaHandle_t& handle, int m, int n, const double& alpha, const double* x, int incx, const double* y, int incy, double* A, int lda);
template void geru<std::complex<float>>(const DdlaHandle_t& handle, int m, int n, const std::complex<float>& alpha, const std::complex<float>* x, int incx, const std::complex<float>* y, int incy, std::complex<float>* A, int lda);
template void geru<std::complex<double>>(const DdlaHandle_t& handle, int m, int n, const std::complex<double>& alpha, const std::complex<double>* x, int incx, const std::complex<double>* y, int incy, std::complex<double>* A, int lda);

template <typename T>
void gerc(
    const DdlaHandle_t& handle, int m, int n, const T& alpha, const T* x, int incx, const T* y, int incy, T* A, int lda)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return gerc<DdlaBackend::GPU, T>(handle, m, n, alpha, x, incx, y, incy, A, lda);
#endif
    return gerc<DdlaBackend::CPU, T>(handle, m, n, alpha, x, incx, y, incy, A, lda);
}


template void gerc<float>(const DdlaHandle_t& handle, int m, int n, const float& alpha, const float* x, int incx, const float* y, int incy, float* A, int lda);
template void gerc<double>(const DdlaHandle_t& handle, int m, int n, const double& alpha, const double* x, int incx, const double* y, int incy, double* A, int lda);
template void gerc<std::complex<float>>(const DdlaHandle_t& handle, int m, int n, const std::complex<float>& alpha, const std::complex<float>* x, int incx, const std::complex<float>* y, int incy, std::complex<float>* A, int lda);
template void gerc<std::complex<double>>(const DdlaHandle_t& handle, int m, int n, const std::complex<double>& alpha, const std::complex<double>* x, int incx, const std::complex<double>* y, int incy, std::complex<double>* A, int lda);

template <typename T>
void getrf(
    const DdlaHandle_t& handle, int m, int n, T* A, int lda, int* ipiv, int* info)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return getrf<DdlaBackend::GPU, T>(handle, m, n, A, lda, ipiv, info);
#endif
    return getrf<DdlaBackend::CPU, T>(handle, m, n, A, lda, ipiv, info);
}


template void getrf<float>(const DdlaHandle_t& handle, int m, int n, float* A, int lda, int* ipiv, int* info);
template void getrf<double>(const DdlaHandle_t& handle, int m, int n, double* A, int lda, int* ipiv, int* info);
template void getrf<std::complex<float>>(const DdlaHandle_t& handle, int m, int n, std::complex<float>* A, int lda, int* ipiv, int* info);
template void getrf<std::complex<double>>(const DdlaHandle_t& handle, int m, int n, std::complex<double>* A, int lda, int* ipiv, int* info);

template <typename T>
void potrf(
    const DdlaHandle_t& handle, char uplo, int n, T* A, int lda, int* info)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return potrf<DdlaBackend::GPU, T>(handle, uplo, n, A, lda, info);
#endif
    return potrf<DdlaBackend::CPU, T>(handle, uplo, n, A, lda, info);
}


template void potrf<float>(const DdlaHandle_t& handle, char uplo, int n, float* A, int lda, int* info);
template void potrf<double>(const DdlaHandle_t& handle, char uplo, int n, double* A, int lda, int* info);
template void potrf<std::complex<float>>(const DdlaHandle_t& handle, char uplo, int n, std::complex<float>* A, int lda, int* info);
template void potrf<std::complex<double>>(const DdlaHandle_t& handle, char uplo, int n, std::complex<double>* A, int lda, int* info);

template <typename T>
void laswp(
    const DdlaHandle_t& handle, int n, T* A, int lda, int k1, int k2, const int* ipiv, int incx)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return laswp<DdlaBackend::GPU, T>(handle, n, A, lda, k1, k2, ipiv, incx);
#endif
    return laswp<DdlaBackend::CPU, T>(handle, n, A, lda, k1, k2, ipiv, incx);
}


template void laswp<float>(const DdlaHandle_t& handle, int n, float* A, int lda, int k1, int k2, const int* ipiv, int incx);
template void laswp<double>(const DdlaHandle_t& handle, int n, double* A, int lda, int k1, int k2, const int* ipiv, int incx);
template void laswp<std::complex<float>>(const DdlaHandle_t& handle, int n, std::complex<float>* A, int lda, int k1, int k2, const int* ipiv, int incx);
template void laswp<std::complex<double>>(const DdlaHandle_t& handle, int n, std::complex<double>* A, int lda, int k1, int k2, const int* ipiv, int incx);

template <typename T>
void geqrf(
    const DdlaHandle_t& handle, char uplo, int m, int n, T* A, int lda, T* tau)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return geqrf<DdlaBackend::GPU, T>(handle, uplo, m, n, A, lda, tau);
#endif
    return geqrf<DdlaBackend::CPU, T>(handle, uplo, m, n, A, lda, tau);
}


template void geqrf<float>(const DdlaHandle_t& handle, char uplo, int m, int n, float* A, int lda, float* tau);
template void geqrf<double>(const DdlaHandle_t& handle, char uplo, int m, int n, double* A, int lda, double* tau);
template void geqrf<std::complex<float>>(const DdlaHandle_t& handle, char uplo, int m, int n, std::complex<float>* A, int lda, std::complex<float>* tau);
template void geqrf<std::complex<double>>(const DdlaHandle_t& handle, char uplo, int m, int n, std::complex<double>* A, int lda, std::complex<double>* tau);

template <typename T>
void larft(
    const DdlaHandle_t& handle, char direct, char storev, int n, int k, T* V, int ldv, const T* tau, T* Tmat, int ldt)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return larft<DdlaBackend::GPU, T>(handle, direct, storev, n, k, V, ldv, tau, Tmat, ldt);
#endif
    return larft<DdlaBackend::CPU, T>(handle, direct, storev, n, k, V, ldv, tau, Tmat, ldt);
}


template void larft<float>(const DdlaHandle_t& handle, char direct, char storev, int n, int k, float* V, int ldv, const float* tau, float* Tmat, int ldt);
template void larft<double>(const DdlaHandle_t& handle, char direct, char storev, int n, int k, double* V, int ldv, const double* tau, double* Tmat, int ldt);
template void larft<std::complex<float>>(const DdlaHandle_t& handle, char direct, char storev, int n, int k, std::complex<float>* V, int ldv, const std::complex<float>* tau, std::complex<float>* Tmat, int ldt);
template void larft<std::complex<double>>(const DdlaHandle_t& handle, char direct, char storev, int n, int k, std::complex<double>* V, int ldv, const std::complex<double>* tau, std::complex<double>* Tmat, int ldt);

template <typename R>
void pgemm(
    const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const std::complex<R>& alpha, const R* A, const int* descA, const std::complex<R>* B, const int* descB, const std::complex<R>& beta, std::complex<R>* C, const int* descC)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pgemm<DdlaBackend::GPU, R>(handle, transa, transb, m, n, k, alpha, A, descA, B, descB, beta, C, descC);
#endif
    return pgemm<DdlaBackend::CPU, R>(handle, transa, transb, m, n, k, alpha, A, descA, B, descB, beta, C, descC);
}


template void pgemm<float>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const std::complex<float>& alpha, const float* A, const int* descA, const std::complex<float>* B, const int* descB, const std::complex<float>& beta, std::complex<float>* C, const int* descC);
template void pgemm<double>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const std::complex<double>& alpha, const double* A, const int* descA, const std::complex<double>* B, const int* descB, const std::complex<double>& beta, std::complex<double>* C, const int* descC);

template <typename R>
void pgemm(
    const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const std::complex<R>& alpha, const std::complex<R>* A, const int* descA, const R* B, const int* descB, const std::complex<R>& beta, std::complex<R>* C, const int* descC)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pgemm<DdlaBackend::GPU, R>(handle, transa, transb, m, n, k, alpha, A, descA, B, descB, beta, C, descC);
#endif
    return pgemm<DdlaBackend::CPU, R>(handle, transa, transb, m, n, k, alpha, A, descA, B, descB, beta, C, descC);
}


template void pgemm<float>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const std::complex<float>& alpha, const std::complex<float>* A, const int* descA, const float* B, const int* descB, const std::complex<float>& beta, std::complex<float>* C, const int* descC);
template void pgemm<double>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const std::complex<double>& alpha, const std::complex<double>* A, const int* descA, const double* B, const int* descB, const std::complex<double>& beta, std::complex<double>* C, const int* descC);

template <typename T>
void phe2hb(
    const DdlaHandle_t& handle, int n, T* A, const int* descA, int nbw, T* tmat, int& info)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return phe2hb<DdlaBackend::GPU, T>(handle, n, A, descA, nbw, tmat, info);
#endif
    return phe2hb<DdlaBackend::CPU, T>(handle, n, A, descA, nbw, tmat, info);
}


template void phe2hb<float>(const DdlaHandle_t& handle, int n, float* A, const int* descA, int nbw, float* tmat, int& info);
template void phe2hb<double>(const DdlaHandle_t& handle, int n, double* A, const int* descA, int nbw, double* tmat, int& info);
template void phe2hb<std::complex<float>>(const DdlaHandle_t& handle, int n, std::complex<float>* A, const int* descA, int nbw, std::complex<float>* tmat, int& info);
template void phe2hb<std::complex<double>>(const DdlaHandle_t& handle, int n, std::complex<double>* A, const int* descA, int nbw, std::complex<double>* tmat, int& info);

template <typename T>
void punmtr_he2hb(
    const DdlaHandle_t& handle, char trans, int n, int nbw, const T* A, const int* descA, const T* tmat, T* Z, const int* descZ, int& info)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return punmtr_he2hb<DdlaBackend::GPU, T>(handle, trans, n, nbw, A, descA, tmat, Z, descZ, info);
#endif
    return punmtr_he2hb<DdlaBackend::CPU, T>(handle, trans, n, nbw, A, descA, tmat, Z, descZ, info);
}


template void punmtr_he2hb<float>(const DdlaHandle_t& handle, char trans, int n, int nbw, const float* A, const int* descA, const float* tmat, float* Z, const int* descZ, int& info);
template void punmtr_he2hb<double>(const DdlaHandle_t& handle, char trans, int n, int nbw, const double* A, const int* descA, const double* tmat, double* Z, const int* descZ, int& info);
template void punmtr_he2hb<std::complex<float>>(const DdlaHandle_t& handle, char trans, int n, int nbw, const std::complex<float>* A, const int* descA, const std::complex<float>* tmat, std::complex<float>* Z, const int* descZ, int& info);
template void punmtr_he2hb<std::complex<double>>(const DdlaHandle_t& handle, char trans, int n, int nbw, const std::complex<double>* A, const int* descA, const std::complex<double>* tmat, std::complex<double>* Z, const int* descZ, int& info);

template <typename T>
void pherm_complete(
    const DdlaHandle_t& handle, char uplo, int n, T* A, const int* descA)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pherm_complete<DdlaBackend::GPU, T>(handle, uplo, n, A, descA);
#endif
    return pherm_complete<DdlaBackend::CPU, T>(handle, uplo, n, A, descA);
}


template void pherm_complete<float>(const DdlaHandle_t& handle, char uplo, int n, float* A, const int* descA);
template void pherm_complete<double>(const DdlaHandle_t& handle, char uplo, int n, double* A, const int* descA);
template void pherm_complete<std::complex<float>>(const DdlaHandle_t& handle, char uplo, int n, std::complex<float>* A, const int* descA);
template void pherm_complete<std::complex<double>>(const DdlaHandle_t& handle, char uplo, int n, std::complex<double>* A, const int* descA);

template <typename T>
void phetrd(
    const DdlaHandle_t& handle, char uplo, int n, T* A, const int* descA, typename real_type<T>::type* d, typename real_type<T>::type* e, T* tau, int& info)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return phetrd<DdlaBackend::GPU, T>(handle, uplo, n, A, descA, d, e, tau, info);
#endif
    return phetrd<DdlaBackend::CPU, T>(handle, uplo, n, A, descA, d, e, tau, info);
}


template void phetrd<float>(const DdlaHandle_t& handle, char uplo, int n, float* A, const int* descA, typename real_type<float>::type* d, typename real_type<float>::type* e, float* tau, int& info);
template void phetrd<double>(const DdlaHandle_t& handle, char uplo, int n, double* A, const int* descA, typename real_type<double>::type* d, typename real_type<double>::type* e, double* tau, int& info);
template void phetrd<std::complex<float>>(const DdlaHandle_t& handle, char uplo, int n, std::complex<float>* A, const int* descA, typename real_type<std::complex<float>>::type* d, typename real_type<std::complex<float>>::type* e, std::complex<float>* tau, int& info);
template void phetrd<std::complex<double>>(const DdlaHandle_t& handle, char uplo, int n, std::complex<double>* A, const int* descA, typename real_type<std::complex<double>>::type* d, typename real_type<std::complex<double>>::type* e, std::complex<double>* tau, int& info);

template <typename T>
void punmtr(
    const DdlaHandle_t& handle, char side, char uplo, char trans, int m, int n, const T* A, const int* descA, const T* tau, T* C, const int* descC)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return punmtr<DdlaBackend::GPU, T>(handle, side, uplo, trans, m, n, A, descA, tau, C, descC);
#endif
    return punmtr<DdlaBackend::CPU, T>(handle, side, uplo, trans, m, n, A, descA, tau, C, descC);
}


template void punmtr<float>(const DdlaHandle_t& handle, char side, char uplo, char trans, int m, int n, const float* A, const int* descA, const float* tau, float* C, const int* descC);
template void punmtr<double>(const DdlaHandle_t& handle, char side, char uplo, char trans, int m, int n, const double* A, const int* descA, const double* tau, double* C, const int* descC);
template void punmtr<std::complex<float>>(const DdlaHandle_t& handle, char side, char uplo, char trans, int m, int n, const std::complex<float>* A, const int* descA, const std::complex<float>* tau, std::complex<float>* C, const int* descC);
template void punmtr<std::complex<double>>(const DdlaHandle_t& handle, char side, char uplo, char trans, int m, int n, const std::complex<double>* A, const int* descA, const std::complex<double>* tau, std::complex<double>* C, const int* descC);

template <typename T>
void punmtr(
    const DdlaHandle_t& handle, char side, char uplo, char trans, int m, int n, int k, const T* V, int ldv, const T* Tmat, int ldt, T* C, const int* descC)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return punmtr<DdlaBackend::GPU, T>(handle, side, uplo, trans, m, n, k, V, ldv, Tmat, ldt, C, descC);
#endif
    return punmtr<DdlaBackend::CPU, T>(handle, side, uplo, trans, m, n, k, V, ldv, Tmat, ldt, C, descC);
}


template void punmtr<float>(const DdlaHandle_t& handle, char side, char uplo, char trans, int m, int n, int k, const float* V, int ldv, const float* Tmat, int ldt, float* C, const int* descC);
template void punmtr<double>(const DdlaHandle_t& handle, char side, char uplo, char trans, int m, int n, int k, const double* V, int ldv, const double* Tmat, int ldt, double* C, const int* descC);
template void punmtr<std::complex<float>>(const DdlaHandle_t& handle, char side, char uplo, char trans, int m, int n, int k, const std::complex<float>* V, int ldv, const std::complex<float>* Tmat, int ldt, std::complex<float>* C, const int* descC);
template void punmtr<std::complex<double>>(const DdlaHandle_t& handle, char side, char uplo, char trans, int m, int n, int k, const std::complex<double>* V, int ldv, const std::complex<double>* Tmat, int ldt, std::complex<double>* C, const int* descC);

template <typename R>
void gemm(
    const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const std::complex<R>& alpha, const std::complex<R>* A, int lda, const R* B, int ldb, const std::complex<R>& beta, std::complex<R>* C, int ldc)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return gemm<DdlaBackend::GPU, R>(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
    return gemm<DdlaBackend::CPU, R>(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}


template void gemm<float>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const std::complex<float>& alpha, const std::complex<float>* A, int lda, const float* B, int ldb, const std::complex<float>& beta, std::complex<float>* C, int ldc);
template void gemm<double>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const std::complex<double>& alpha, const std::complex<double>* A, int lda, const double* B, int ldb, const std::complex<double>& beta, std::complex<double>* C, int ldc);

template <typename T>
void syevd(
    const DdlaHandle_t& handle, char jobz, char uplo, int n, T* A, int lda, T* w, int* info)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return syevd<DdlaBackend::GPU, T>(handle, jobz, uplo, n, A, lda, w, info);
#endif
    return syevd<DdlaBackend::CPU, T>(handle, jobz, uplo, n, A, lda, w, info);
}


template void syevd<float>(const DdlaHandle_t& handle, char jobz, char uplo, int n, float* A, int lda, float* w, int* info);
template void syevd<double>(const DdlaHandle_t& handle, char jobz, char uplo, int n, double* A, int lda, double* w, int* info);

template <typename T>
void heevd(
    const DdlaHandle_t& handle, char jobz, char uplo, int n, T* A, int lda, typename real_type<T>::type* w, int* info)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return heevd<DdlaBackend::GPU, T>(handle, jobz, uplo, n, A, lda, w, info);
#endif
    return heevd<DdlaBackend::CPU, T>(handle, jobz, uplo, n, A, lda, w, info);
}


template void heevd<std::complex<float>>(const DdlaHandle_t& handle, char jobz, char uplo, int n, std::complex<float>* A, int lda, typename real_type<std::complex<float>>::type* w, int* info);
template void heevd<std::complex<double>>(const DdlaHandle_t& handle, char jobz, char uplo, int n, std::complex<double>* A, int lda, typename real_type<std::complex<double>>::type* w, int* info);

template <typename T>
void pheevd(
    const DdlaHandle_t& handle, char jobz, char uplo, int n, T* A, const int* descA, typename real_type<T>::type* w, T* Z, const int* descZ, int& info)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pheevd<DdlaBackend::GPU, T>(handle, jobz, uplo, n, A, descA, w, Z, descZ, info);
#endif
    return pheevd<DdlaBackend::CPU, T>(handle, jobz, uplo, n, A, descA, w, Z, descZ, info);
}


template void pheevd<float>(const DdlaHandle_t& handle, char jobz, char uplo, int n, float* A, const int* descA, typename real_type<float>::type* w, float* Z, const int* descZ, int& info);
template void pheevd<double>(const DdlaHandle_t& handle, char jobz, char uplo, int n, double* A, const int* descA, typename real_type<double>::type* w, double* Z, const int* descZ, int& info);
template void pheevd<std::complex<float>>(const DdlaHandle_t& handle, char jobz, char uplo, int n, std::complex<float>* A, const int* descA, typename real_type<std::complex<float>>::type* w, std::complex<float>* Z, const int* descZ, int& info);
template void pheevd<std::complex<double>>(const DdlaHandle_t& handle, char jobz, char uplo, int n, std::complex<double>* A, const int* descA, typename real_type<std::complex<double>>::type* w, std::complex<double>* Z, const int* descZ, int& info);

template <typename T>
void pheevd_two_stage(
    const DdlaHandle_t& handle, char jobz, char uplo, int n, T* A, const int* descA, typename real_type<T>::type* w, T* Z, const int* descZ, int& info)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pheevd_two_stage<DdlaBackend::GPU, T>(handle, jobz, uplo, n, A, descA, w, Z, descZ, info);
#endif
    return pheevd_two_stage<DdlaBackend::CPU, T>(handle, jobz, uplo, n, A, descA, w, Z, descZ, info);
}


template void pheevd_two_stage<float>(const DdlaHandle_t& handle, char jobz, char uplo, int n, float* A, const int* descA, typename real_type<float>::type* w, float* Z, const int* descZ, int& info);
template void pheevd_two_stage<double>(const DdlaHandle_t& handle, char jobz, char uplo, int n, double* A, const int* descA, typename real_type<double>::type* w, double* Z, const int* descZ, int& info);
template void pheevd_two_stage<std::complex<float>>(const DdlaHandle_t& handle, char jobz, char uplo, int n, std::complex<float>* A, const int* descA, typename real_type<std::complex<float>>::type* w, std::complex<float>* Z, const int* descZ, int& info);
template void pheevd_two_stage<std::complex<double>>(const DdlaHandle_t& handle, char jobz, char uplo, int n, std::complex<double>* A, const int* descA, typename real_type<std::complex<double>>::type* w, std::complex<double>* Z, const int* descZ, int& info);

template <typename T>
void pheevd_one_stage(
    const DdlaHandle_t& handle, char jobz, char uplo, int n, T* A, const int* descA, typename real_type<T>::type* w, T* Z, const int* descZ, int& info)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return pheevd_one_stage<DdlaBackend::GPU, T>(handle, jobz, uplo, n, A, descA, w, Z, descZ, info);
#endif
    return pheevd_one_stage<DdlaBackend::CPU, T>(handle, jobz, uplo, n, A, descA, w, Z, descZ, info);
}


template void pheevd_one_stage<float>(const DdlaHandle_t& handle, char jobz, char uplo, int n, float* A, const int* descA, typename real_type<float>::type* w, float* Z, const int* descZ, int& info);
template void pheevd_one_stage<double>(const DdlaHandle_t& handle, char jobz, char uplo, int n, double* A, const int* descA, typename real_type<double>::type* w, double* Z, const int* descZ, int& info);
template void pheevd_one_stage<std::complex<float>>(const DdlaHandle_t& handle, char jobz, char uplo, int n, std::complex<float>* A, const int* descA, typename real_type<std::complex<float>>::type* w, std::complex<float>* Z, const int* descZ, int& info);
template void pheevd_one_stage<std::complex<double>>(const DdlaHandle_t& handle, char jobz, char uplo, int n, std::complex<double>* A, const int* descA, typename real_type<std::complex<double>>::type* w, std::complex<double>* Z, const int* descZ, int& info);

template <typename T>
void gemmt(
    const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k, const T& alpha, const T* A, int lda, const T* B, int ldb, const T& beta, T* C, int ldc)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return gemmt<DdlaBackend::GPU, T>(handle, uplo, transa, transb, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
    return gemmt<DdlaBackend::CPU, T>(handle, uplo, transa, transb, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}


template void gemmt<float>(const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k, const float& alpha, const float* A, int lda, const float* B, int ldb, const float& beta, float* C, int ldc);
template void gemmt<double>(const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k, const double& alpha, const double* A, int lda, const double* B, int ldb, const double& beta, double* C, int ldc);
template void gemmt<std::complex<float>>(const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k, const std::complex<float>& alpha, const std::complex<float>* A, int lda, const std::complex<float>* B, int ldb, const std::complex<float>& beta, std::complex<float>* C, int ldc);
template void gemmt<std::complex<double>>(const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k, const std::complex<double>& alpha, const std::complex<double>* A, int lda, const std::complex<double>* B, int ldb, const std::complex<double>& beta, std::complex<double>* C, int ldc);

template <typename R>
void gemmt(
    const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k, const std::complex<R>& alpha, const R* A, int lda, const std::complex<R>* B, int ldb, const std::complex<R>& beta, std::complex<R>* C, int ldc)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return gemmt<DdlaBackend::GPU, R>(handle, uplo, transa, transb, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
    return gemmt<DdlaBackend::CPU, R>(handle, uplo, transa, transb, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}


template void gemmt<float>(const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k, const std::complex<float>& alpha, const float* A, int lda, const std::complex<float>* B, int ldb, const std::complex<float>& beta, std::complex<float>* C, int ldc);
template void gemmt<double>(const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k, const std::complex<double>& alpha, const double* A, int lda, const std::complex<double>* B, int ldb, const std::complex<double>& beta, std::complex<double>* C, int ldc);

template <typename R>
void gemmt(
    const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k, const std::complex<R>& alpha, const std::complex<R>* A, int lda, const R* B, int ldb, const std::complex<R>& beta, std::complex<R>* C, int ldc)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return gemmt<DdlaBackend::GPU, R>(handle, uplo, transa, transb, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
    return gemmt<DdlaBackend::CPU, R>(handle, uplo, transa, transb, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}


template void gemmt<float>(const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k, const std::complex<float>& alpha, const std::complex<float>* A, int lda, const float* B, int ldb, const std::complex<float>& beta, std::complex<float>* C, int ldc);
template void gemmt<double>(const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k, const std::complex<double>& alpha, const std::complex<double>* A, int lda, const double* B, int ldb, const std::complex<double>& beta, std::complex<double>* C, int ldc);

template <typename T>
void ptrtrm(
    const DdlaHandle_t& handle, char uplo_a, char transa, char uplo_b, char transb, int n, const T& alpha, const T* A, const int* descA, const T* B, const int* descB, const T& beta, T* C, const int* descC)
{
#if DDLA_HAS_GPU
    if (handle && handle->backend == DdlaBackend::GPU)
        return ptrtrm<DdlaBackend::GPU, T>(handle, uplo_a, transa, uplo_b, transb, n, alpha, A, descA, B, descB, beta, C, descC);
#endif
    return ptrtrm<DdlaBackend::CPU, T>(handle, uplo_a, transa, uplo_b, transb, n, alpha, A, descA, B, descB, beta, C, descC);
}


template void ptrtrm<float>(const DdlaHandle_t& handle, char uplo_a, char transa, char uplo_b, char transb, int n, const float& alpha, const float* A, const int* descA, const float* B, const int* descB, const float& beta, float* C, const int* descC);
template void ptrtrm<double>(const DdlaHandle_t& handle, char uplo_a, char transa, char uplo_b, char transb, int n, const double& alpha, const double* A, const int* descA, const double* B, const int* descB, const double& beta, double* C, const int* descC);
template void ptrtrm<std::complex<float>>(const DdlaHandle_t& handle, char uplo_a, char transa, char uplo_b, char transb, int n, const std::complex<float>& alpha, const std::complex<float>* A, const int* descA, const std::complex<float>* B, const int* descB, const std::complex<float>& beta, std::complex<float>* C, const int* descC);
template void ptrtrm<std::complex<double>>(const DdlaHandle_t& handle, char uplo_a, char transa, char uplo_b, char transb, int n, const std::complex<double>& alpha, const std::complex<double>* A, const int* descA, const std::complex<double>* B, const int* descB, const std::complex<double>& beta, std::complex<double>* C, const int* descC);

template <typename T>
void sytrf(
    const DdlaHandle_t& h, char uplo, int n, T* A, int lda, int* ipiv, int& info)
{
#if DDLA_HAS_GPU
    if (h && h->backend == DdlaBackend::GPU)
        return sytrf<DdlaBackend::GPU, T>(h, uplo, n, A, lda, ipiv, info);
#endif
    return sytrf<DdlaBackend::CPU, T>(h, uplo, n, A, lda, ipiv, info);
}


template void sytrf<float>(const DdlaHandle_t& h, char uplo, int n, float* A, int lda, int* ipiv, int& info);
template void sytrf<double>(const DdlaHandle_t& h, char uplo, int n, double* A, int lda, int* ipiv, int& info);
template void sytrf<std::complex<float>>(const DdlaHandle_t& h, char uplo, int n, std::complex<float>* A, int lda, int* ipiv, int& info);
template void sytrf<std::complex<double>>(const DdlaHandle_t& h, char uplo, int n, std::complex<double>* A, int lda, int* ipiv, int& info);

template <typename T>
void sytrs(
    const DdlaHandle_t& h, char uplo, int n, int nrhs, const T* A, int lda, const int* ipiv, T* B, int ldb, int& info)
{
#if DDLA_HAS_GPU
    if (h && h->backend == DdlaBackend::GPU)
        return sytrs<DdlaBackend::GPU, T>(h, uplo, n, nrhs, A, lda, ipiv, B, ldb, info);
#endif
    return sytrs<DdlaBackend::CPU, T>(h, uplo, n, nrhs, A, lda, ipiv, B, ldb, info);
}


template void sytrs<float>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const float* A, int lda, const int* ipiv, float* B, int ldb, int& info);
template void sytrs<double>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const double* A, int lda, const int* ipiv, double* B, int ldb, int& info);
template void sytrs<std::complex<float>>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const std::complex<float>* A, int lda, const int* ipiv, std::complex<float>* B, int ldb, int& info);
template void sytrs<std::complex<double>>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const std::complex<double>* A, int lda, const int* ipiv, std::complex<double>* B, int ldb, int& info);

template <typename T>
void sytrf_nopiv(
    const DdlaHandle_t& h, char uplo, int n, T* A, int lda, int& info)
{
#if DDLA_HAS_GPU
    if (h && h->backend == DdlaBackend::GPU)
        return sytrf_nopiv<DdlaBackend::GPU, T>(h, uplo, n, A, lda, info);
#endif
    return sytrf_nopiv<DdlaBackend::CPU, T>(h, uplo, n, A, lda, info);
}


template void sytrf_nopiv<float>(const DdlaHandle_t& h, char uplo, int n, float* A, int lda, int& info);
template void sytrf_nopiv<double>(const DdlaHandle_t& h, char uplo, int n, double* A, int lda, int& info);
template void sytrf_nopiv<std::complex<float>>(const DdlaHandle_t& h, char uplo, int n, std::complex<float>* A, int lda, int& info);
template void sytrf_nopiv<std::complex<double>>(const DdlaHandle_t& h, char uplo, int n, std::complex<double>* A, int lda, int& info);

template <typename T>
void sytrs_nopiv(
    const DdlaHandle_t& h, char uplo, int n, int nrhs, const T* A, int lda, T* B, int ldb, int& info)
{
#if DDLA_HAS_GPU
    if (h && h->backend == DdlaBackend::GPU)
        return sytrs_nopiv<DdlaBackend::GPU, T>(h, uplo, n, nrhs, A, lda, B, ldb, info);
#endif
    return sytrs_nopiv<DdlaBackend::CPU, T>(h, uplo, n, nrhs, A, lda, B, ldb, info);
}


template void sytrs_nopiv<float>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const float* A, int lda, float* B, int ldb, int& info);
template void sytrs_nopiv<double>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const double* A, int lda, double* B, int ldb, int& info);
template void sytrs_nopiv<std::complex<float>>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const std::complex<float>* A, int lda, std::complex<float>* B, int ldb, int& info);
template void sytrs_nopiv<std::complex<double>>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const std::complex<double>* A, int lda, std::complex<double>* B, int ldb, int& info);

template <typename T>
void hetrf(
    const DdlaHandle_t& h, char uplo, int n, T* A, int lda, int* ipiv, int& info)
{
#if DDLA_HAS_GPU
    if (h && h->backend == DdlaBackend::GPU)
        return hetrf<DdlaBackend::GPU, T>(h, uplo, n, A, lda, ipiv, info);
#endif
    return hetrf<DdlaBackend::CPU, T>(h, uplo, n, A, lda, ipiv, info);
}


template void hetrf<float>(const DdlaHandle_t& h, char uplo, int n, float* A, int lda, int* ipiv, int& info);
template void hetrf<double>(const DdlaHandle_t& h, char uplo, int n, double* A, int lda, int* ipiv, int& info);
template void hetrf<std::complex<float>>(const DdlaHandle_t& h, char uplo, int n, std::complex<float>* A, int lda, int* ipiv, int& info);
template void hetrf<std::complex<double>>(const DdlaHandle_t& h, char uplo, int n, std::complex<double>* A, int lda, int* ipiv, int& info);

template <typename T>
void hetrs(
    const DdlaHandle_t& h, char uplo, int n, int nrhs, const T* A, int lda, const int* ipiv, T* B, int ldb, int& info)
{
#if DDLA_HAS_GPU
    if (h && h->backend == DdlaBackend::GPU)
        return hetrs<DdlaBackend::GPU, T>(h, uplo, n, nrhs, A, lda, ipiv, B, ldb, info);
#endif
    return hetrs<DdlaBackend::CPU, T>(h, uplo, n, nrhs, A, lda, ipiv, B, ldb, info);
}


template void hetrs<float>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const float* A, int lda, const int* ipiv, float* B, int ldb, int& info);
template void hetrs<double>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const double* A, int lda, const int* ipiv, double* B, int ldb, int& info);
template void hetrs<std::complex<float>>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const std::complex<float>* A, int lda, const int* ipiv, std::complex<float>* B, int ldb, int& info);
template void hetrs<std::complex<double>>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const std::complex<double>* A, int lda, const int* ipiv, std::complex<double>* B, int ldb, int& info);

template <typename T>
void hetrf_nopiv(
    const DdlaHandle_t& h, char uplo, int n, T* A, int lda, int& info)
{
#if DDLA_HAS_GPU
    if (h && h->backend == DdlaBackend::GPU)
        return hetrf_nopiv<DdlaBackend::GPU, T>(h, uplo, n, A, lda, info);
#endif
    return hetrf_nopiv<DdlaBackend::CPU, T>(h, uplo, n, A, lda, info);
}


template void hetrf_nopiv<float>(const DdlaHandle_t& h, char uplo, int n, float* A, int lda, int& info);
template void hetrf_nopiv<double>(const DdlaHandle_t& h, char uplo, int n, double* A, int lda, int& info);
template void hetrf_nopiv<std::complex<float>>(const DdlaHandle_t& h, char uplo, int n, std::complex<float>* A, int lda, int& info);
template void hetrf_nopiv<std::complex<double>>(const DdlaHandle_t& h, char uplo, int n, std::complex<double>* A, int lda, int& info);

template <typename T>
void hetrs_nopiv(
    const DdlaHandle_t& h, char uplo, int n, int nrhs, const T* A, int lda, T* B, int ldb, int& info)
{
#if DDLA_HAS_GPU
    if (h && h->backend == DdlaBackend::GPU)
        return hetrs_nopiv<DdlaBackend::GPU, T>(h, uplo, n, nrhs, A, lda, B, ldb, info);
#endif
    return hetrs_nopiv<DdlaBackend::CPU, T>(h, uplo, n, nrhs, A, lda, B, ldb, info);
}


template void hetrs_nopiv<float>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const float* A, int lda, float* B, int ldb, int& info);
template void hetrs_nopiv<double>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const double* A, int lda, double* B, int ldb, int& info);
template void hetrs_nopiv<std::complex<float>>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const std::complex<float>* A, int lda, std::complex<float>* B, int ldb, int& info);
template void hetrs_nopiv<std::complex<double>>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const std::complex<double>* A, int lda, std::complex<double>* B, int ldb, int& info);

template <typename T>
void psytrf(
    const DdlaHandle_t& h, char uplo, int n, T* A, const int* descA, int* ipiv, int& info)
{
#if DDLA_HAS_GPU
    if (h && h->backend == DdlaBackend::GPU)
        return psytrf<DdlaBackend::GPU, T>(h, uplo, n, A, descA, ipiv, info);
#endif
    return psytrf<DdlaBackend::CPU, T>(h, uplo, n, A, descA, ipiv, info);
}


template void psytrf<float>(const DdlaHandle_t& h, char uplo, int n, float* A, const int* descA, int* ipiv, int& info);
template void psytrf<double>(const DdlaHandle_t& h, char uplo, int n, double* A, const int* descA, int* ipiv, int& info);
template void psytrf<std::complex<float>>(const DdlaHandle_t& h, char uplo, int n, std::complex<float>* A, const int* descA, int* ipiv, int& info);
template void psytrf<std::complex<double>>(const DdlaHandle_t& h, char uplo, int n, std::complex<double>* A, const int* descA, int* ipiv, int& info);

template <typename T>
void psytrs(
    const DdlaHandle_t& h, char uplo, int n, int nrhs, const T* A, const int* descA, const int* ipiv, T* B, const int* descB, int& info)
{
#if DDLA_HAS_GPU
    if (h && h->backend == DdlaBackend::GPU)
        return psytrs<DdlaBackend::GPU, T>(h, uplo, n, nrhs, A, descA, ipiv, B, descB, info);
#endif
    return psytrs<DdlaBackend::CPU, T>(h, uplo, n, nrhs, A, descA, ipiv, B, descB, info);
}


template void psytrs<float>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const float* A, const int* descA, const int* ipiv, float* B, const int* descB, int& info);
template void psytrs<double>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const double* A, const int* descA, const int* ipiv, double* B, const int* descB, int& info);
template void psytrs<std::complex<float>>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const std::complex<float>* A, const int* descA, const int* ipiv, std::complex<float>* B, const int* descB, int& info);
template void psytrs<std::complex<double>>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const std::complex<double>* A, const int* descA, const int* ipiv, std::complex<double>* B, const int* descB, int& info);

template <typename T>
void psytrf_nopiv(
    const DdlaHandle_t& h, char uplo, int n, T* A, const int* descA, int& info)
{
#if DDLA_HAS_GPU
    if (h && h->backend == DdlaBackend::GPU)
        return psytrf_nopiv<DdlaBackend::GPU, T>(h, uplo, n, A, descA, info);
#endif
    return psytrf_nopiv<DdlaBackend::CPU, T>(h, uplo, n, A, descA, info);
}


template void psytrf_nopiv<float>(const DdlaHandle_t& h, char uplo, int n, float* A, const int* descA, int& info);
template void psytrf_nopiv<double>(const DdlaHandle_t& h, char uplo, int n, double* A, const int* descA, int& info);
template void psytrf_nopiv<std::complex<float>>(const DdlaHandle_t& h, char uplo, int n, std::complex<float>* A, const int* descA, int& info);
template void psytrf_nopiv<std::complex<double>>(const DdlaHandle_t& h, char uplo, int n, std::complex<double>* A, const int* descA, int& info);

template <typename T>
void psytrs_nopiv(
    const DdlaHandle_t& h, char uplo, int n, int nrhs, const T* A, const int* descA, T* B, const int* descB, int& info)
{
#if DDLA_HAS_GPU
    if (h && h->backend == DdlaBackend::GPU)
        return psytrs_nopiv<DdlaBackend::GPU, T>(h, uplo, n, nrhs, A, descA, B, descB, info);
#endif
    return psytrs_nopiv<DdlaBackend::CPU, T>(h, uplo, n, nrhs, A, descA, B, descB, info);
}


template void psytrs_nopiv<float>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const float* A, const int* descA, float* B, const int* descB, int& info);
template void psytrs_nopiv<double>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const double* A, const int* descA, double* B, const int* descB, int& info);
template void psytrs_nopiv<std::complex<float>>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const std::complex<float>* A, const int* descA, std::complex<float>* B, const int* descB, int& info);
template void psytrs_nopiv<std::complex<double>>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const std::complex<double>* A, const int* descA, std::complex<double>* B, const int* descB, int& info);

template <typename T>
void phetrf(
    const DdlaHandle_t& h, char uplo, int n, T* A, const int* descA, int* ipiv, int& info)
{
#if DDLA_HAS_GPU
    if (h && h->backend == DdlaBackend::GPU)
        return phetrf<DdlaBackend::GPU, T>(h, uplo, n, A, descA, ipiv, info);
#endif
    return phetrf<DdlaBackend::CPU, T>(h, uplo, n, A, descA, ipiv, info);
}


template void phetrf<float>(const DdlaHandle_t& h, char uplo, int n, float* A, const int* descA, int* ipiv, int& info);
template void phetrf<double>(const DdlaHandle_t& h, char uplo, int n, double* A, const int* descA, int* ipiv, int& info);
template void phetrf<std::complex<float>>(const DdlaHandle_t& h, char uplo, int n, std::complex<float>* A, const int* descA, int* ipiv, int& info);
template void phetrf<std::complex<double>>(const DdlaHandle_t& h, char uplo, int n, std::complex<double>* A, const int* descA, int* ipiv, int& info);

template <typename T>
void phetrs(
    const DdlaHandle_t& h, char uplo, int n, int nrhs, const T* A, const int* descA, const int* ipiv, T* B, const int* descB, int& info)
{
#if DDLA_HAS_GPU
    if (h && h->backend == DdlaBackend::GPU)
        return phetrs<DdlaBackend::GPU, T>(h, uplo, n, nrhs, A, descA, ipiv, B, descB, info);
#endif
    return phetrs<DdlaBackend::CPU, T>(h, uplo, n, nrhs, A, descA, ipiv, B, descB, info);
}


template void phetrs<float>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const float* A, const int* descA, const int* ipiv, float* B, const int* descB, int& info);
template void phetrs<double>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const double* A, const int* descA, const int* ipiv, double* B, const int* descB, int& info);
template void phetrs<std::complex<float>>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const std::complex<float>* A, const int* descA, const int* ipiv, std::complex<float>* B, const int* descB, int& info);
template void phetrs<std::complex<double>>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const std::complex<double>* A, const int* descA, const int* ipiv, std::complex<double>* B, const int* descB, int& info);

template <typename T>
void phetrf_nopiv(
    const DdlaHandle_t& h, char uplo, int n, T* A, const int* descA, int& info)
{
#if DDLA_HAS_GPU
    if (h && h->backend == DdlaBackend::GPU)
        return phetrf_nopiv<DdlaBackend::GPU, T>(h, uplo, n, A, descA, info);
#endif
    return phetrf_nopiv<DdlaBackend::CPU, T>(h, uplo, n, A, descA, info);
}


template void phetrf_nopiv<float>(const DdlaHandle_t& h, char uplo, int n, float* A, const int* descA, int& info);
template void phetrf_nopiv<double>(const DdlaHandle_t& h, char uplo, int n, double* A, const int* descA, int& info);
template void phetrf_nopiv<std::complex<float>>(const DdlaHandle_t& h, char uplo, int n, std::complex<float>* A, const int* descA, int& info);
template void phetrf_nopiv<std::complex<double>>(const DdlaHandle_t& h, char uplo, int n, std::complex<double>* A, const int* descA, int& info);

template <typename T>
void phetrs_nopiv(
    const DdlaHandle_t& h, char uplo, int n, int nrhs, const T* A, const int* descA, T* B, const int* descB, int& info)
{
#if DDLA_HAS_GPU
    if (h && h->backend == DdlaBackend::GPU)
        return phetrs_nopiv<DdlaBackend::GPU, T>(h, uplo, n, nrhs, A, descA, B, descB, info);
#endif
    return phetrs_nopiv<DdlaBackend::CPU, T>(h, uplo, n, nrhs, A, descA, B, descB, info);
}


template void phetrs_nopiv<float>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const float* A, const int* descA, float* B, const int* descB, int& info);
template void phetrs_nopiv<double>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const double* A, const int* descA, double* B, const int* descB, int& info);
template void phetrs_nopiv<std::complex<float>>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const std::complex<float>* A, const int* descA, std::complex<float>* B, const int* descB, int& info);
template void phetrs_nopiv<std::complex<double>>(const DdlaHandle_t& h, char uplo, int n, int nrhs, const std::complex<double>* A, const int* descA, std::complex<double>* B, const int* descB, int& info);

template <typename T>
void ormqr(
    const DdlaHandle_t& h, char side, char trans, int m, int n, int k, const T* A, int lda, const T* tau, T* C, int ldc, int& info)
{
#if DDLA_HAS_GPU
    if (h && h->backend == DdlaBackend::GPU)
        return ormqr<DdlaBackend::GPU, T>(h, side, trans, m, n, k, A, lda, tau, C, ldc, info);
#endif
    return ormqr<DdlaBackend::CPU, T>(h, side, trans, m, n, k, A, lda, tau, C, ldc, info);
}


template void ormqr<float>(const DdlaHandle_t& h, char side, char trans, int m, int n, int k, const float* A, int lda, const float* tau, float* C, int ldc, int& info);
template void ormqr<double>(const DdlaHandle_t& h, char side, char trans, int m, int n, int k, const double* A, int lda, const double* tau, double* C, int ldc, int& info);
template void ormqr<std::complex<float>>(const DdlaHandle_t& h, char side, char trans, int m, int n, int k, const std::complex<float>* A, int lda, const std::complex<float>* tau, std::complex<float>* C, int ldc, int& info);
template void ormqr<std::complex<double>>(const DdlaHandle_t& h, char side, char trans, int m, int n, int k, const std::complex<double>* A, int lda, const std::complex<double>* tau, std::complex<double>* C, int ldc, int& info);

template <typename T>
void pormqr(
    const DdlaHandle_t& h, char side, char trans, int m, int n, int k, const T* A, const int* descA, const T* tau, T* C, const int* descC, int& info)
{
#if DDLA_HAS_GPU
    if (h && h->backend == DdlaBackend::GPU)
        return pormqr<DdlaBackend::GPU, T>(h, side, trans, m, n, k, A, descA, tau, C, descC, info);
#endif
    return pormqr<DdlaBackend::CPU, T>(h, side, trans, m, n, k, A, descA, tau, C, descC, info);
}


template void pormqr<float>(const DdlaHandle_t& h, char side, char trans, int m, int n, int k, const float* A, const int* descA, const float* tau, float* C, const int* descC, int& info);
template void pormqr<double>(const DdlaHandle_t& h, char side, char trans, int m, int n, int k, const double* A, const int* descA, const double* tau, double* C, const int* descC, int& info);
template void pormqr<std::complex<float>>(const DdlaHandle_t& h, char side, char trans, int m, int n, int k, const std::complex<float>* A, const int* descA, const std::complex<float>* tau, std::complex<float>* C, const int* descC, int& info);
template void pormqr<std::complex<double>>(const DdlaHandle_t& h, char side, char trans, int m, int n, int k, const std::complex<double>* A, const int* descA, const std::complex<double>* tau, std::complex<double>* C, const int* descC, int& info);

template <typename T>
void orgqr(
    const DdlaHandle_t& h, int m, int n, int k, T* A, int lda, const T* tau, int& info)
{
#if DDLA_HAS_GPU
    if (h && h->backend == DdlaBackend::GPU)
        return orgqr<DdlaBackend::GPU, T>(h, m, n, k, A, lda, tau, info);
#endif
    return orgqr<DdlaBackend::CPU, T>(h, m, n, k, A, lda, tau, info);
}


template void orgqr<float>(const DdlaHandle_t& h, int m, int n, int k, float* A, int lda, const float* tau, int& info);
template void orgqr<double>(const DdlaHandle_t& h, int m, int n, int k, double* A, int lda, const double* tau, int& info);
template void orgqr<std::complex<float>>(const DdlaHandle_t& h, int m, int n, int k, std::complex<float>* A, int lda, const std::complex<float>* tau, int& info);
template void orgqr<std::complex<double>>(const DdlaHandle_t& h, int m, int n, int k, std::complex<double>* A, int lda, const std::complex<double>* tau, int& info);

template <typename T>
void porgqr(
    const DdlaHandle_t& h, int m, int n, int k, T* A, const int* descA, const T* tau, int& info)
{
#if DDLA_HAS_GPU
    if (h && h->backend == DdlaBackend::GPU)
        return porgqr<DdlaBackend::GPU, T>(h, m, n, k, A, descA, tau, info);
#endif
    return porgqr<DdlaBackend::CPU, T>(h, m, n, k, A, descA, tau, info);
}


template void porgqr<float>(const DdlaHandle_t& h, int m, int n, int k, float* A, const int* descA, const float* tau, int& info);
template void porgqr<double>(const DdlaHandle_t& h, int m, int n, int k, double* A, const int* descA, const double* tau, int& info);
template void porgqr<std::complex<float>>(const DdlaHandle_t& h, int m, int n, int k, std::complex<float>* A, const int* descA, const std::complex<float>* tau, int& info);
template void porgqr<std::complex<double>>(const DdlaHandle_t& h, int m, int n, int k, std::complex<double>* A, const int* descA, const std::complex<double>* tau, int& info);

} // namespace ddla
