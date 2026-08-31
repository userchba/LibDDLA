#ifndef POTRF_H
#define POTRF_H

#include "ddla_connector.h"
#include <stdexcept>

namespace ddla{

inline desolverStatus_t desolverPotrf(
    desolverHandle_t handle,
    deblasFillMode_t uplo,
    int n,
    std::complex<double> *A,
    int lda,
    int *devInfo
)
{
    std::complex<double> *Workspace = nullptr;
    int Lwork;

    runtimeStream_t stream;
    SOLVER_CHECK(desolverGetStream(handle, &stream));

    #if defined(DDLA_USE_CUDA)
    SOLVER_CHECK(cusolverDnZpotrf_bufferSize(handle, uplo, n, (cuDoubleComplex*)A, lda, &Lwork));
    #elif defined(DDLA_USE_HIP)
    SOLVER_CHECK(hipsolverZpotrf_bufferSize(handle, desolverFillMode(uplo), n, (hipDoubleComplex*)A, lda, &Lwork));
    #else
    throw std::runtime_error("not ENABLE CUDA and ENABLE HIP\n");
    #endif

    if(Lwork > 0)
        RUNTIME_CHECK(runtimeMallocAsync(&Workspace, Lwork*sizeof(std::complex<double>), stream));

    desolverStatus_t status{};
    #if defined(DDLA_USE_CUDA)
    status = cusolverDnZpotrf(handle, uplo, n, (cuDoubleComplex*)A, lda, (cuDoubleComplex*)Workspace, Lwork, devInfo);
    #elif defined(DDLA_USE_HIP)
    status = hipsolverZpotrf(handle, desolverFillMode(uplo), n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)Workspace, Lwork, devInfo);
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif

    if(Lwork > 0)
        RUNTIME_CHECK(runtimeFreeAsync(Workspace, stream));
    return status;
}

inline desolverStatus_t desolverPotrf(
    desolverHandle_t handle,
    deblasFillMode_t uplo,
    int n,
    std::complex<float> *A,
    int lda,
    int *devInfo
)
{
    std::complex<float> *Workspace = nullptr;
    int Lwork;

    runtimeStream_t stream;
    SOLVER_CHECK(desolverGetStream(handle, &stream));

    #if defined(DDLA_USE_CUDA)
    SOLVER_CHECK(cusolverDnCpotrf_bufferSize(handle, uplo, n, (cuFloatComplex*)A, lda, &Lwork));
    #elif defined(DDLA_USE_HIP)
    SOLVER_CHECK(hipsolverCpotrf_bufferSize(handle, desolverFillMode(uplo), n, (hipFloatComplex*)A, lda, &Lwork));
    #else
    throw std::runtime_error("not ENABLE CUDA and ENABLE HIP\n");
    #endif

    if(Lwork > 0)
        RUNTIME_CHECK(runtimeMallocAsync(&Workspace, Lwork*sizeof(std::complex<float>), stream));

    desolverStatus_t status{};
    #if defined(DDLA_USE_CUDA)
    status = cusolverDnCpotrf(handle, uplo, n, (cuFloatComplex*)A, lda, (cuFloatComplex*)Workspace, Lwork, devInfo);
    #elif defined(DDLA_USE_HIP)
    status = hipsolverCpotrf(handle, desolverFillMode(uplo), n, (hipFloatComplex*)A, lda, (hipFloatComplex*)Workspace, Lwork, devInfo);
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif

    if(Lwork > 0)
        RUNTIME_CHECK(runtimeFreeAsync(Workspace, stream));
    return status;
}

inline desolverStatus_t desolverPotrf(
    desolverHandle_t handle,
    deblasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    int *devInfo
)
{
    float *Workspace = nullptr;
    int Lwork;

    runtimeStream_t stream;
    SOLVER_CHECK(desolverGetStream(handle, &stream));

    #if defined(DDLA_USE_CUDA)
    SOLVER_CHECK(cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, &Lwork));
    #elif defined(DDLA_USE_HIP)
    SOLVER_CHECK(hipsolverSpotrf_bufferSize(handle, desolverFillMode(uplo), n, A, lda, &Lwork));
    #else
    throw std::runtime_error("not ENABLE CUDA and ENABLE HIP\n");
    #endif

    if(Lwork > 0)
        RUNTIME_CHECK(runtimeMallocAsync(&Workspace, Lwork*sizeof(float), stream));

    desolverStatus_t status{};
    #if defined(DDLA_USE_CUDA)
    status = cusolverDnSpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
    #elif defined(DDLA_USE_HIP)
    status = hipsolverSpotrf(handle, desolverFillMode(uplo), n, A, lda, Workspace, Lwork, devInfo);
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif

    if(Lwork > 0)
        RUNTIME_CHECK(runtimeFreeAsync(Workspace, stream));
    return status;
}

inline desolverStatus_t desolverPotrf(
    desolverHandle_t handle,
    deblasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    int *devInfo
)
{
    double *Workspace = nullptr;
    int Lwork;

    runtimeStream_t stream;
    SOLVER_CHECK(desolverGetStream(handle, &stream));

    #if defined(DDLA_USE_CUDA)
    SOLVER_CHECK(cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, &Lwork));
    #elif defined(DDLA_USE_HIP)
    SOLVER_CHECK(hipsolverDpotrf_bufferSize(handle, desolverFillMode(uplo), n, A, lda, &Lwork));
    #else
    throw std::runtime_error("not ENABLE CUDA and ENABLE HIP\n");
    #endif

    if(Lwork > 0)
        RUNTIME_CHECK(runtimeMallocAsync(&Workspace, Lwork*sizeof(double), stream));

    desolverStatus_t status{};
    #if defined(DDLA_USE_CUDA)
    status = cusolverDnDpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
    #elif defined(DDLA_USE_HIP)
    status = hipsolverDpotrf(handle, desolverFillMode(uplo), n, A, lda, Workspace, Lwork, devInfo);
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif

    if(Lwork > 0)
        RUNTIME_CHECK(runtimeFreeAsync(Workspace, stream));
    return status;
}

}


#endif
