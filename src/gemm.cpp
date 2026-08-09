#include <ddla/gemm.h>

#include <complex>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "ddla_stream_impl.h"

#if DDLA_HAS_CPU
extern "C" {
void sgemm_(const char*, const char*, const int*, const int*, const int*,
            const float*, const float*, const int*, const float*, const int*,
            const float*, float*, const int*);
void dgemm_(const char*, const char*, const int*, const int*, const int*,
            const double*, const double*, const int*, const double*, const int*,
            const double*, double*, const int*);
void cgemm_(const char*, const char*, const int*, const int*, const int*,
            const std::complex<float>*, const std::complex<float>*, const int*,
            const std::complex<float>*, const int*,
            const std::complex<float>*, std::complex<float>*, const int*);
void zgemm_(const char*, const char*, const int*, const int*, const int*,
            const std::complex<double>*, const std::complex<double>*, const int*,
            const std::complex<double>*, const int*,
            const std::complex<double>*, std::complex<double>*, const int*);
}
#endif

namespace ddla {

namespace {

// Scalar-type overload dispatch for the CPU and GPU gemm paths. C++11 has no
// `if constexpr`, so the former is_same_v<> chains become plain overloads on
// the scalar pointer type: overload resolution picks the right BLAS entry
// point at compile time, and an unsupported T simply has no matching
// overload (the same guard the old `static_assert(sizeof(T)==0)` tail gave).

#if DDLA_HAS_CPU
inline void cpu_gemm_call(char transa, char transb, int m, int n, int k,
                          const float& alpha, const float* A, int lda,
                          const float* B, int ldb, const float& beta,
                          float* C, int ldc)
{
    sgemm_(&transa, &transb, &m, &n, &k,
           &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}
inline void cpu_gemm_call(char transa, char transb, int m, int n, int k,
                          const double& alpha, const double* A, int lda,
                          const double* B, int ldb, const double& beta,
                          double* C, int ldc)
{
    dgemm_(&transa, &transb, &m, &n, &k,
           &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}
inline void cpu_gemm_call(char transa, char transb, int m, int n, int k,
                          const std::complex<float>& alpha, const std::complex<float>* A, int lda,
                          const std::complex<float>* B, int ldb, const std::complex<float>& beta,
                          std::complex<float>* C, int ldc)
{
    cgemm_(&transa, &transb, &m, &n, &k,
           &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}
inline void cpu_gemm_call(char transa, char transb, int m, int n, int k,
                          const std::complex<double>& alpha, const std::complex<double>* A, int lda,
                          const std::complex<double>* B, int ldb, const std::complex<double>& beta,
                          std::complex<double>* C, int ldc)
{
    zgemm_(&transa, &transb, &m, &n, &k,
           &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}
#endif // DDLA_HAS_CPU

#if DDLA_HAS_GPU
inline void gpu_gemm_call(deblasHandle_t blasH, deblasOperation_t opA, deblasOperation_t opB,
                          int m, int n, int k,
                          const float& alpha, const float* A, int lda,
                          const float* B, int ldb, const float& beta,
                          float* C, int ldc)
{
#if defined(DDLA_USE_CUDA)
    BLAS_CHECK(cublasSgemm(blasH, opA, opB, m, n, k,
                           &alpha, A, lda, B, ldb, &beta, C, ldc));
#elif defined(DDLA_USE_HIP)
    BLAS_CHECK(hipblasSgemm(blasH, opA, opB, m, n, k,
                            &alpha, A, lda, B, ldb, &beta, C, ldc));
#else
    throw std::runtime_error(
        "gemm: GPU backend requires DDLA_USE_CUDA or DDLA_USE_HIP");
#endif
}
inline void gpu_gemm_call(deblasHandle_t blasH, deblasOperation_t opA, deblasOperation_t opB,
                          int m, int n, int k,
                          const double& alpha, const double* A, int lda,
                          const double* B, int ldb, const double& beta,
                          double* C, int ldc)
{
#if defined(DDLA_USE_CUDA)
    BLAS_CHECK(cublasDgemm(blasH, opA, opB, m, n, k,
                           &alpha, A, lda, B, ldb, &beta, C, ldc));
#elif defined(DDLA_USE_HIP)
    BLAS_CHECK(hipblasDgemm(blasH, opA, opB, m, n, k,
                            &alpha, A, lda, B, ldb, &beta, C, ldc));
#else
    throw std::runtime_error(
        "gemm: GPU backend requires DDLA_USE_CUDA or DDLA_USE_HIP");
#endif
}
inline void gpu_gemm_call(deblasHandle_t blasH, deblasOperation_t opA, deblasOperation_t opB,
                          int m, int n, int k,
                          const std::complex<float>& alpha, const std::complex<float>* A, int lda,
                          const std::complex<float>* B, int ldb, const std::complex<float>& beta,
                          std::complex<float>* C, int ldc)
{
#if defined(DDLA_USE_CUDA)
    BLAS_CHECK(cublasCgemm(blasH, opA, opB, m, n, k,
                           (cuFloatComplex*)&alpha, (cuFloatComplex*)A, lda,
                           (cuFloatComplex*)B, ldb,
                           (cuFloatComplex*)&beta, (cuFloatComplex*)C, ldc));
#elif defined(DDLA_USE_HIP)
    BLAS_CHECK(hipblasCgemm(blasH, opA, opB, m, n, k,
                            (hipblasComplex*)&alpha, (hipblasComplex*)A, lda,
                            (hipblasComplex*)B, ldb,
                            (hipblasComplex*)&beta, (hipblasComplex*)C, ldc));
#else
    throw std::runtime_error(
        "gemm: GPU backend requires DDLA_USE_CUDA or DDLA_USE_HIP");
#endif
}
inline void gpu_gemm_call(deblasHandle_t blasH, deblasOperation_t opA, deblasOperation_t opB,
                          int m, int n, int k,
                          const std::complex<double>& alpha, const std::complex<double>* A, int lda,
                          const std::complex<double>* B, int ldb, const std::complex<double>& beta,
                          std::complex<double>* C, int ldc)
{
#if defined(DDLA_USE_CUDA)
    BLAS_CHECK(cublasZgemm(blasH, opA, opB, m, n, k,
                           (cuDoubleComplex*)&alpha, (cuDoubleComplex*)A, lda,
                           (cuDoubleComplex*)B, ldb,
                           (cuDoubleComplex*)&beta, (cuDoubleComplex*)C, ldc));
#elif defined(DDLA_USE_HIP)
    BLAS_CHECK(hipblasZgemm(blasH, opA, opB, m, n, k,
                            (hipblasDoubleComplex*)&alpha, (hipblasDoubleComplex*)A, lda,
                            (hipblasDoubleComplex*)B, ldb,
                            (hipblasDoubleComplex*)&beta, (hipblasDoubleComplex*)C, ldc));
#else
    throw std::runtime_error(
        "gemm: GPU backend requires DDLA_USE_CUDA or DDLA_USE_HIP");
#endif
}
#endif // DDLA_HAS_GPU

} // anonymous namespace

inline const char* backend_name(DdlaBackend backend)
{
    return backend == DdlaBackend::CPU ? "CPU" : "GPU";
}

template <DdlaBackend Backend, typename T>
void gemm(
    const DdlaHandle_t& handle,
    char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const T& beta,
    T* C, int ldc)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU gemm is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU gemm is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("gemm: null handle");
    }
    const DdlaBackend actual = ddla_get_backend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("gemm: template backend ") + backend_name(Backend) +
            " does not match handle backend " + backend_name(actual));
    }

    if (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        cpu_gemm_call(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
    } else {
#if DDLA_HAS_GPU
        const deblasOperation_t opA = transa == 'N' ? DEBLAS_OP_N :
                                      transa == 'T' ? DEBLAS_OP_T : DEBLAS_OP_C;
        const deblasOperation_t opB = transb == 'N' ? DEBLAS_OP_N :
                                      transb == 'T' ? DEBLAS_OP_T : DEBLAS_OP_C;
        gpu_gemm_call(handle->blasH, opA, opB, m, n, k,
                      alpha, A, lda, B, ldb, beta, C, ldc);
#endif
    }
}

#define INSTANTIATE_GEMM(BACKEND, TYPE)                                      \
    template void gemm<BACKEND, TYPE>(                               \
        const DdlaHandle_t&, char, char, int, int, int,                       \
        const TYPE&, const TYPE*, int, const TYPE*, int,                      \
        const TYPE&, TYPE*, int)

#if DDLA_HAS_CPU
INSTANTIATE_GEMM(DdlaBackend::CPU, float);
INSTANTIATE_GEMM(DdlaBackend::CPU, double);
INSTANTIATE_GEMM(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_GEMM(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_GEMM(DdlaBackend::GPU, float);
INSTANTIATE_GEMM(DdlaBackend::GPU, double);
INSTANTIATE_GEMM(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_GEMM(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_GEMM

} // namespace ddla
