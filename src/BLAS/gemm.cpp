#include "blas_impl.h"
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <complex>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "ddla_connector.h"
#include "ddla_stream.h"
#include "instantiate.h"
#include "require_backend.h"

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
    detail::require_backend<Backend>(handle, "gemm");
    // Fortran LSAME is case-insensitive, so accept either case on both
    // backends; anything else would otherwise reach cuBLAS as OP_C.
    const char ta = upper(transa), tb = upper(transb);
    if ((ta != 'N' && ta != 'T' && ta != 'C') || (tb != 'N' && tb != 'T' && tb != 'C'))
        throw std::invalid_argument("gemm: transa/transb must be 'N', 'T' or 'C'");
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
    } else {
#if DDLA_HAS_GPU
        const gpublasOperation_t opA = op(transa);
        const gpublasOperation_t opB = op(transb);
        if constexpr (std::is_same_v<T, float>) {
            BLAS_CHECK(gpublasSgemm(static_cast<gpublasHandle_t>(handle->blasH), opA, opB, m, n, k,
                                    &alpha, A, lda, B, ldb, &beta, C, ldc));
        } else if constexpr (std::is_same_v<T, double>) {
            BLAS_CHECK(gpublasDgemm(static_cast<gpublasHandle_t>(handle->blasH), opA, opB, m, n, k,
                                    &alpha, A, lda, B, ldb, &beta, C, ldc));
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            BLAS_CHECK(gpublasCgemm(static_cast<gpublasHandle_t>(handle->blasH), opA, opB, m, n, k,
                                    (gpublasComplex*)&alpha, (gpublasComplex*)A, lda,
                                    (gpublasComplex*)B, ldb,
                                    (gpublasComplex*)&beta, (gpublasComplex*)C, ldc));
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            BLAS_CHECK(gpublasZgemm(static_cast<gpublasHandle_t>(handle->blasH), opA, opB, m, n, k,
                                    (gpublasDoubleComplex*)&alpha, (gpublasDoubleComplex*)A, lda,
                                    (gpublasDoubleComplex*)B, ldb,
                                    (gpublasDoubleComplex*)&beta, (gpublasDoubleComplex*)C, ldc));
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

DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_GEMM)

#undef INSTANTIATE_GEMM


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::gemm_check<BACKEND, TYPE>( const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const TYPE& alpha, const TYPE* A, int lda, const TYPE* B, int ldb, const TYPE& beta, TYPE* C, int ldc); \
    template void blas_impl::gemm_run<BACKEND, TYPE>( const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const TYPE& alpha, const TYPE* A, int lda, const TYPE* B, int ldb, const TYPE& beta, TYPE* C, int ldc)
DDLA_FOR_EACH_BACKEND_TYPE(DDLA_INSTANTIATE_PHASES)
#undef DDLA_INSTANTIATE_PHASES
} // namespace ddla
