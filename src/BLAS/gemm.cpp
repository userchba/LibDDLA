#include "blas_impl.h"
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <complex>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "ddla_connector.h"
#include "ddla_stream.h"

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

inline const char* backend_name(DdlaBackend backend)
{
    return backend == DdlaBackend::CPU ? "CPU" : "GPU";
}

namespace blas_impl {

template <DdlaBackend Backend, typename T>
void gemm_check(
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
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("gemm: template backend ") + backend_name(Backend) +
            " does not match handle backend " + backend_name(actual));
    }

}

template <DdlaBackend Backend, typename T>
void gemm_run(
    const DdlaHandle_t& handle,
    char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const T& beta,
    T* C, int ldc)
{
    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        if constexpr (std::is_same_v<T, float>)
            sgemm_(&transa, &transb, &m, &n, &k,
                   &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        else if constexpr (std::is_same_v<T, double>)
            dgemm_(&transa, &transb, &m, &n, &k,
                   &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        else if constexpr (std::is_same_v<T, std::complex<float>>)
            cgemm_(&transa, &transb, &m, &n, &k,
                   &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        else if constexpr (std::is_same_v<T, std::complex<double>>)
            zgemm_(&transa, &transb, &m, &n, &k,
                   &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        else
            static_assert(sizeof(T) == 0, "gemm: unsupported scalar type T");
#endif
    } else {
#if DDLA_HAS_GPU
        const gpublasOperation_t opA = transa == 'N' ? GPUBLAS_OP_N :
                                       transa == 'T' ? GPUBLAS_OP_T : GPUBLAS_OP_C;
        const gpublasOperation_t opB = transb == 'N' ? GPUBLAS_OP_N :
                                       transb == 'T' ? GPUBLAS_OP_T : GPUBLAS_OP_C;
        if constexpr (std::is_same_v<T, float>) {
#if DDLA_GPU_VENDOR
            BLAS_CHECK(gpublasSgemm(static_cast<gpublasHandle_t>(handle->blasH), opA, opB, m, n, k,
                                    &alpha, A, lda, B, ldb, &beta, C, ldc));
#else
            throw std::runtime_error(
                "gemm: GPU backend requires DDLA_USE_CUDA or DDLA_USE_HIP");
#endif
        } else if constexpr (std::is_same_v<T, double>) {
#if DDLA_GPU_VENDOR
            BLAS_CHECK(gpublasDgemm(static_cast<gpublasHandle_t>(handle->blasH), opA, opB, m, n, k,
                                    &alpha, A, lda, B, ldb, &beta, C, ldc));
#else
            throw std::runtime_error(
                "gemm: GPU backend requires DDLA_USE_CUDA or DDLA_USE_HIP");
#endif
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
#if DDLA_GPU_VENDOR
            BLAS_CHECK(gpublasCgemm(static_cast<gpublasHandle_t>(handle->blasH), opA, opB, m, n, k,
                                    (gpublasComplex*)&alpha, (gpublasComplex*)A, lda,
                                    (gpublasComplex*)B, ldb,
                                    (gpublasComplex*)&beta, (gpublasComplex*)C, ldc));
#else
            throw std::runtime_error(
                "gemm: GPU backend requires DDLA_USE_CUDA or DDLA_USE_HIP");
#endif
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
#if DDLA_GPU_VENDOR
            BLAS_CHECK(gpublasZgemm(static_cast<gpublasHandle_t>(handle->blasH), opA, opB, m, n, k,
                                    (gpublasDoubleComplex*)&alpha, (gpublasDoubleComplex*)A, lda,
                                    (gpublasDoubleComplex*)B, ldb,
                                    (gpublasDoubleComplex*)&beta, (gpublasDoubleComplex*)C, ldc));
#else
            throw std::runtime_error(
                "gemm: GPU backend requires DDLA_USE_CUDA or DDLA_USE_HIP");
#endif
        } else {
            static_assert(sizeof(T) == 0, "gemm: unsupported scalar type T");
        }
#endif
    }
}

} // namespace blas_impl

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
    blas_impl::gemm_check<Backend, T>(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    blas_impl::gemm_run<Backend, T>(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::gemm_check<BACKEND, TYPE>( const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const TYPE& alpha, const TYPE* A, int lda, const TYPE* B, int ldb, const TYPE& beta, TYPE* C, int ldc); \
    template void blas_impl::gemm_run<BACKEND, TYPE>( const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const TYPE& alpha, const TYPE* A, int lda, const TYPE* B, int ldb, const TYPE& beta, TYPE* C, int ldc);
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
