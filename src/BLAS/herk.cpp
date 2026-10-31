#include "blas_impl.h"
// ---------------------------------------------------------------------------
// Unified backend-templated Hermitian rank-k update (C := alpha*A*A^H + beta*C).
//
// Structure follows src/gemm.cpp exactly: one definition covering CPU (Fortran
// BLAS) and GPU (cuBLAS/hipBLAS), selected with `if constexpr` on the Backend
// template parameter, with the arms themselves guarded by the BUILD CAPABILITY
// macros (DDLA_HAS_CPU / DDLA_HAS_GPU) and the vendor chosen per scalar arm by
// the PER-TU macros (DDLA_USE_CUDA / DDLA_USE_HIP).
//
// Why the capability guard matters here: the previous design put a `#else`
// CPU arm inside src/herk.h selected by `!defined(DDLA_USE_CUDA) &&
// !defined(DDLA_USE_HIP)`.  In a dual build (DDLA_USE_CPU + DDLA_USE_CUDA)
// every TU defines DDLA_USE_CUDA, so that arm was never compiled -- while
// DDLA_HAS_CPU was 1 and the CPU instantiation of this function still existed.
// A CPU-backend call would therefore have reached cublasCherk with HOST
// pointers.  Guarding on DDLA_HAS_CPU removes that class of bug.
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
// Real: xHERK has no real-scalar BLAS counterpart, so the real types use xSYRK
// (alpha/beta are real in both families, which is why the signature takes a
// real alpha for every T).
void ssyrk_(const char*, const char*, const int*, const int*,
            const float*, const float*, const int*, const float*, float*, const int*);
void dsyrk_(const char*, const char*, const int*, const int*,
            const double*, const double*, const int*, const double*, double*, const int*);
// Complex: xHERK takes a REAL alpha and REAL beta.
void cherk_(const char*, const char*, const int*, const int*,
            const float*, const std::complex<float>*, const int*, const float*,
            std::complex<float>*, const int*);
void zherk_(const char*, const char*, const int*, const int*,
            const double*, const std::complex<double>*, const int*, const double*,
            std::complex<double>*, const int*);
}
#endif

// ---------------------------------------------------------------------------
// GPU arm: the vendor calls.  Kept as free gpublasHerk overloads so the
// dispatch above stays a single BLAS_CHECK call per scalar type, exactly as
// ddla::gemm does it inline.  Private to this translation unit -- the public
// interface is the templated ddla::herk declared in <ddla/ddla.h>.
// ---------------------------------------------------------------------------
#if DDLA_HAS_GPU

namespace ddla {

inline gpublasStatus_t gpublasHerk(
    gpublasHandle_t handle,
    gpublasFillMode_t uplo, gpublasOperation_t trans,
    int n, int k,
    const double& alpha,
    const std::complex<double> *A, int lda,
    const double& beta,
    std::complex<double> *C, int ldc
)
{
    #if DDLA_GPU_VENDOR
    return gpublasZherk(
        handle, uplo, trans, n, k, 
        &alpha, (gpublasDoubleComplex*)A, lda, 
        &beta, (gpublasDoubleComplex*)C, ldc
    );
    #else
    // Neither vendor macro: this can only happen if the translation unit was
    // built without a GPU language while DDLA_HAS_GPU is set, which the build
    // system does not do (see the LANGUAGE CUDA/HIP source lists).  Throw
    // rather than fall through to a host pointer.
    throw std::runtime_error("gpublasHerk: no GPU vendor selected at compile time");
    #endif
}


inline gpublasStatus_t gpublasHerk(
    gpublasHandle_t handle,
    gpublasFillMode_t uplo, gpublasOperation_t trans,
    int n, int k,
    const float& alpha,
    const float* A, int lda,
    const float& beta,
    float* C, int ldc
)
{
    // Real HERK == SYRK.  For real data a conjugate transpose is identical
    // to a plain transpose, so map GPUBLAS_OP_C -> transpose as well.
    gpublasOperation_t trans_real = (trans == GPUBLAS_OP_N) ? GPUBLAS_OP_N : GPUBLAS_OP_T;
    #if DDLA_GPU_VENDOR
    return gpublasSsyrk(handle, uplo, trans_real, n, k,
        &alpha, A, lda, &beta, C, ldc);
    #else
    throw std::runtime_error("gpublasHerk: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasHerk(
    gpublasHandle_t handle,
    gpublasFillMode_t uplo, gpublasOperation_t trans,
    int n, int k,
    const double& alpha,
    const double* A, int lda,
    const double& beta,
    double* C, int ldc
)
{
    gpublasOperation_t trans_real = (trans == GPUBLAS_OP_N) ? GPUBLAS_OP_N : GPUBLAS_OP_T;
    #if DDLA_GPU_VENDOR
    return gpublasDsyrk(handle, uplo, trans_real, n, k,
        &alpha, A, lda, &beta, C, ldc);
    #else
    throw std::runtime_error("gpublasHerk: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasHerk(
    gpublasHandle_t handle,
    gpublasFillMode_t uplo, gpublasOperation_t trans,
    int n, int k,
    const float& alpha,
    const std::complex<float> *A, int lda,
    const float& beta,
    std::complex<float> *C, int ldc
)
{
    #if DDLA_GPU_VENDOR
    return gpublasCherk(
        handle, uplo, trans, n, k, 
        &alpha, (gpublasComplex*)A, lda,
        &beta, (gpublasComplex*)C, ldc
    );
    #else
    throw std::runtime_error("gpublasHerk: no GPU vendor selected at compile time");
    #endif
}

} // namespace ddla

#endif // DDLA_HAS_GPU

namespace ddla {

namespace blas_impl {

template <DdlaBackend Backend, typename T>
void herk_check(
    const DdlaHandle_t& handle,
    char uplo, char trans,
    int n, int k,
    const typename real_type<T>::type& alpha,
    const T* A, int lda,
    const typename real_type<T>::type& beta,
    T* C, int ldc)
{
    using rt = typename real_type<T>::type;

    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU herk is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU herk is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("herk: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("herk: template backend ") +
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
void herk_run(
    const DdlaHandle_t& handle,
    char uplo, char trans,
    int n, int k,
    const typename real_type<T>::type& alpha,
    const T* A, int lda,
    const typename real_type<T>::type& beta,
    T* C, int ldc)
{
    using rt = typename real_type<T>::type;
    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        // Real data: HERK == SYRK, and 'C' must be spelled 'T' because some
        // vendor copies reject 'C' on real operands.  Complex data keeps the
        // caller's flag ('N' or 'C') -- xHERK accepts both.
        if constexpr (std::is_same_v<T, float>) {
            const char u = uplo, t = detail::trans_real(trans);
            ssyrk_(&u, &t, &n, &k, &alpha, A, &lda, &beta, C, &ldc);
        } else if constexpr (std::is_same_v<T, double>) {
            const char u = uplo, t = detail::trans_real(trans);
            dsyrk_(&u, &t, &n, &k, &alpha, A, &lda, &beta, C, &ldc);
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            const char u = uplo;
            cherk_(&u, &trans, &n, &k, &alpha, A, &lda, &beta, C, &ldc);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            const char u = uplo;
            zherk_(&u, &trans, &n, &k, &alpha, A, &lda, &beta, C, &ldc);
        } else {
            static_assert(sizeof(T) == 0, "herk: unsupported scalar type T");
        }
        (void)sizeof(rt);   // rt is named for clarity; unused on this path
#else
        (void)handle; (void)uplo; (void)trans; (void)n; (void)k;
        (void)alpha; (void)A; (void)lda; (void)beta; (void)C; (void)ldc;
#endif
    } else {
#if DDLA_HAS_GPU
        // The real types map onto xSYRK with a normalised transpose inside
        // gpublasHerk; the complex types take the caller's flag unchanged.
        if constexpr (std::is_same_v<T, float> ||
                      std::is_same_v<T, double> ||
                      std::is_same_v<T, std::complex<float>> ||
                      std::is_same_v<T, std::complex<double>>) {
            BLAS_CHECK(gpublasHerk(static_cast<gpublasHandle_t>(handle->blasH),
                                   blas_impl::fill(uplo),
                                   blas_impl::op(trans),
                                   n, k, alpha, A, lda, beta, C, ldc));
        } else {
            static_assert(sizeof(T) == 0, "herk: unsupported scalar type T");
        }
#else
        (void)handle; (void)uplo; (void)trans; (void)n; (void)k;
        (void)alpha; (void)A; (void)lda; (void)beta; (void)C; (void)ldc;
#endif
    }
}

} // namespace blas_impl

template <DdlaBackend Backend, typename T>
void herk(
    const DdlaHandle_t& handle,
    char uplo, char trans,
    int n, int k,
    const typename real_type<T>::type& alpha,
    const T* A, int lda,
    const typename real_type<T>::type& beta,
    T* C, int ldc)
{
    blas_impl::herk_check<Backend, T>(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
    blas_impl::herk_run<Backend, T>(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

#define INSTANTIATE_HERK(BACKEND, TYPE)                                        \
    template void herk<BACKEND, TYPE>(                                         \
        const DdlaHandle_t&, char, char, int, int,                             \
        const typename real_type<TYPE>::type&, const TYPE*, int,               \
        const typename real_type<TYPE>::type&, TYPE*, int)

#if DDLA_HAS_CPU
INSTANTIATE_HERK(DdlaBackend::CPU, float);
INSTANTIATE_HERK(DdlaBackend::CPU, double);
INSTANTIATE_HERK(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_HERK(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_HERK(DdlaBackend::GPU, float);
INSTANTIATE_HERK(DdlaBackend::GPU, double);
INSTANTIATE_HERK(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_HERK(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_HERK


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::herk_check<BACKEND, TYPE>( const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const typename real_type<TYPE>::type& alpha, const TYPE* A, int lda, const typename real_type<TYPE>::type& beta, TYPE* C, int ldc); \
    template void blas_impl::herk_run<BACKEND, TYPE>( const DdlaHandle_t& handle, char uplo, char trans, int n, int k, const typename real_type<TYPE>::type& alpha, const TYPE* A, int lda, const typename real_type<TYPE>::type& beta, TYPE* C, int ldc);
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
