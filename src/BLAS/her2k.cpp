#include "blas_impl.h"
// ---------------------------------------------------------------------------
// Unified backend-templated Hermitian rank-2k update
//   C := alpha*A*B^H + conj(alpha)*B*A^H + beta*C    (trans = 'N')   or
//   C := alpha*A^H*B + conj(alpha)*B^H*A + beta*C    (trans = 'C').
//
// Structure follows src/gemm.cpp and src/herk.cpp exactly: one definition
// covering CPU (Fortran BLAS) and GPU (cuBLAS/hipBLAS), selected with
// `if constexpr` on the Backend template parameter, with the arms themselves
// guarded by the BUILD CAPABILITY macros (DDLA_HAS_CPU / DDLA_HAS_GPU) and the
// vendor chosen per scalar arm by the PER-TU macros (DDLA_USE_CUDA /
// DDLA_USE_HIP).
//
// Why the capability guard matters here: the previous design put a `#else`
// CPU arm inside src/her2k.h selected by `!defined(DDLA_USE_CUDA) &&
// !defined(DDLA_USE_HIP)`.  In a dual build (DDLA_USE_CPU + DDLA_USE_CUDA)
// every TU defines DDLA_USE_CUDA, so that arm was never compiled -- while
// DDLA_HAS_CPU was 1 and the CPU instantiation of this function still existed.
// A CPU-backend call would therefore have reached cublasCher2k with HOST
// pointers.  Guarding on DDLA_HAS_CPU removes that class of bug.
//
// Scalar convention: xHER2K (and cuBLAS v2 / hipBLAS) takes a COMPLEX alpha
// with a REAL beta.  The signature mirrors that: `const T& alpha` and
// `const typename real_type<T>::type& beta`.  For the real scalar types this
// is xSYR2K, where both scalars are real and a conjugate-transpose request is
// normalised to a plain transpose.
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <complex>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "blas_enums.h"
#include "ddla_connector.h"
#include "ddla_stream.h"

#if DDLA_HAS_CPU
extern "C" {
// Real: xHER2K has no real-scalar BLAS counterpart, so the real types use
// xSYR2K (alpha and beta are real in both families).
void ssyr2k_(const char*, const char*, const int*, const int*, const float*,
             const float*, const int*, const float*, const int*,
             const float*, float*, const int*);
void dsyr2k_(const char*, const char*, const int*, const int*, const double*,
             const double*, const int*, const double*, const int*,
             const double*, double*, const int*);
// Complex: xHER2K takes a COMPLEX alpha and a REAL beta.
void cher2k_(const char*, const char*, const int*, const int*,
             const std::complex<float>*, const std::complex<float>*, const int*,
             const std::complex<float>*, const int*, const float*,
             std::complex<float>*, const int*);
void zher2k_(const char*, const char*, const int*, const int*,
             const std::complex<double>*, const std::complex<double>*, const int*,
             const std::complex<double>*, const int*, const double*,
             std::complex<double>*, const int*);
}
#endif

// ---------------------------------------------------------------------------
// GPU arm: the vendor calls.  Kept as free gpublasHer2k overloads so the
// dispatch above stays a single BLAS_CHECK call per scalar type, exactly as
// ddla::gemm does it inline.  Private to this translation unit -- the public
// interface is the templated ddla::her2k declared in <ddla/ddla.h>.
// ---------------------------------------------------------------------------
#if DDLA_HAS_GPU

namespace ddla {

inline gpublasStatus_t gpublasHer2k(
    gpublasHandle_t handle,
    gpublasFillMode_t uplo, gpublasOperation_t trans,
    int n, int k,
    const float& alpha,
    const float* A, int lda,
    const float* B, int ldb,
    const float& beta,
    float* C, int ldc
)
{
    // Real HER2K == SYR2K.  For real data a conjugate transpose is identical
    // to a plain transpose, so map GPUBLAS_OP_C -> transpose as well; cuBLAS
    // and hipBLAS reject 'C' outright and only the reference SSYR2K tolerates
    // it.  (Same mapping as gpublasHerk's real overloads.)
    const gpublasOperation_t trans_real = (trans == GPUBLAS_OP_N) ? GPUBLAS_OP_N : GPUBLAS_OP_T;
    #if DDLA_GPU_VENDOR
    return gpublasSsyr2k(handle, uplo, trans_real, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    #else
    throw std::runtime_error("gpublasHer2k: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasHer2k(
    gpublasHandle_t handle,
    gpublasFillMode_t uplo, gpublasOperation_t trans,
    int n, int k,
    const double& alpha,
    const double* A, int lda,
    const double* B, int ldb,
    const double& beta,
    double* C, int ldc
)
{
    const gpublasOperation_t trans_real = (trans == GPUBLAS_OP_N) ? GPUBLAS_OP_N : GPUBLAS_OP_T;
    #if DDLA_GPU_VENDOR
    return gpublasDsyr2k(handle, uplo, trans_real, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    #else
    throw std::runtime_error("gpublasHer2k: no GPU vendor selected at compile time");
    #endif
}

// cuBLAS v2 quirk (mirrored by hipBLAS): xher2k takes a COMPLEX alpha with a
// REAL beta -- same as Fortran xHER2K.  (xherk, by contrast, takes a real
// alpha, which is what gpublasHerk's real-alpha interface follows.)
inline gpublasStatus_t gpublasHer2k(
    gpublasHandle_t handle,
    gpublasFillMode_t uplo, gpublasOperation_t trans,
    int n, int k,
    const std::complex<float>& alpha,
    const std::complex<float>* A, int lda,
    const std::complex<float>* B, int ldb,
    const float& beta,
    std::complex<float>* C, int ldc
)
{
    #if DDLA_GPU_VENDOR
    return gpublasCher2k(handle, uplo, trans, n, k,
        (gpublasComplex*)&alpha, (gpublasComplex*)A, lda, (gpublasComplex*)B, ldb,
        &beta, (gpublasComplex*)C, ldc);
    #else
    throw std::runtime_error("gpublasHer2k: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasHer2k(
    gpublasHandle_t handle,
    gpublasFillMode_t uplo, gpublasOperation_t trans,
    int n, int k,
    const std::complex<double>& alpha,
    const std::complex<double>* A, int lda,
    const std::complex<double>* B, int ldb,
    const double& beta,
    std::complex<double>* C, int ldc
)
{
    #if DDLA_GPU_VENDOR
    return gpublasZher2k(handle, uplo, trans, n, k,
        (gpublasDoubleComplex*)&alpha, (gpublasDoubleComplex*)A, lda, (gpublasDoubleComplex*)B, ldb,
        &beta, (gpublasDoubleComplex*)C, ldc);
    #else
    throw std::runtime_error("gpublasHer2k: no GPU vendor selected at compile time");
    #endif
}

} // namespace ddla

#endif // DDLA_HAS_GPU

namespace ddla {

namespace blas_impl {

template <DdlaBackend Backend, typename T>
void her2k_check(
    const DdlaHandle_t& handle,
    char uplo, char trans,
    int n, int k,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const typename real_type<T>::type& beta,
    T* C, int ldc)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU her2k is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU her2k is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("her2k: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("her2k: template backend ") +
            (Backend == DdlaBackend::CPU ? "CPU" : "GPU") +
            " does not match handle backend " +
            (actual == DdlaBackend::CPU ? "CPU" : "GPU"));
    }

    if constexpr (Backend == DdlaBackend::GPU) {
        (void)detail::to_fill_mode(uplo);
        (void)detail::to_blas_op(trans);
    }
}

template <DdlaBackend Backend, typename T>
void her2k_run(
    const DdlaHandle_t& handle,
    char uplo, char trans,
    int n, int k,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const typename real_type<T>::type& beta,
    T* C, int ldc)
{
    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        // Real data: HER2K == SYR2K, and 'C' must be spelled 'T' because some
        // vendor copies reject 'C' on real operands.  Complex data keeps the
        // caller's flag ('N' or 'C') -- xHER2K accepts both.
        if constexpr (std::is_same_v<T, float>) {
            const char u = uplo, t = detail::trans_real(trans);
            ssyr2k_(&u, &t, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        } else if constexpr (std::is_same_v<T, double>) {
            const char u = uplo, t = detail::trans_real(trans);
            dsyr2k_(&u, &t, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            const char u = uplo;
            cher2k_(&u, &trans, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            const char u = uplo;
            zher2k_(&u, &trans, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        } else {
            static_assert(sizeof(T) == 0, "her2k: unsupported scalar type T");
        }
#else
        (void)handle; (void)uplo; (void)trans; (void)n; (void)k;
        (void)alpha; (void)A; (void)lda; (void)B; (void)ldb;
        (void)beta; (void)C; (void)ldc;
#endif
    } else {
#if DDLA_HAS_GPU
        // The real types map onto xSYR2K with a normalised transpose inside
        // gpublasHer2k; the complex types take the caller's flag unchanged.
        if constexpr (std::is_same_v<T, float> ||
                      std::is_same_v<T, double> ||
                      std::is_same_v<T, std::complex<float>> ||
                      std::is_same_v<T, std::complex<double>>) {
            BLAS_CHECK(gpublasHer2k(static_cast<gpublasHandle_t>(handle->blasH),
                                    blas_impl::fill(uplo),
                                    blas_impl::op(trans),
                                    n, k, alpha, A, lda, B, ldb, beta, C, ldc));
        } else {
            static_assert(sizeof(T) == 0, "her2k: unsupported scalar type T");
        }
#else
        (void)handle; (void)uplo; (void)trans; (void)n; (void)k;
        (void)alpha; (void)A; (void)lda; (void)B; (void)ldb;
        (void)beta; (void)C; (void)ldc;
#endif
    }
}

} // namespace blas_impl

template <DdlaBackend Backend, typename T>
void her2k(
    const DdlaHandle_t& handle,
    char uplo, char trans,
    int n, int k,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const typename real_type<T>::type& beta,
    T* C, int ldc)
{
    blas_impl::her2k_check<Backend, T>(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    blas_impl::her2k_run<Backend, T>(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

#define INSTANTIATE_HER2K(BACKEND, TYPE)                                       \
    template void her2k<BACKEND, TYPE>(                                        \
        const DdlaHandle_t&, char, char, int, int,                             \
        const TYPE&, const TYPE*, int, const TYPE*, int,                       \
        const typename real_type<TYPE>::type&, TYPE*, int)

#if DDLA_HAS_CPU
INSTANTIATE_HER2K(DdlaBackend::CPU, float);
INSTANTIATE_HER2K(DdlaBackend::CPU, double);
INSTANTIATE_HER2K(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_HER2K(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_HER2K(DdlaBackend::GPU, float);
INSTANTIATE_HER2K(DdlaBackend::GPU, double);
INSTANTIATE_HER2K(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_HER2K(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_HER2K


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::her2k_check<BACKEND, TYPE>( const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const TYPE& alpha, const TYPE* A, int lda, const TYPE* B, int ldb, const typename real_type<TYPE>::type& beta, TYPE* C, int ldc); \
    template void blas_impl::her2k_run<BACKEND, TYPE>( const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const TYPE& alpha, const TYPE* A, int lda, const TYPE* B, int ldb, const typename real_type<TYPE>::type& beta, TYPE* C, int ldc);
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
