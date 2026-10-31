#include "blas_impl.h"
// ---------------------------------------------------------------------------
// Unified backend-templated batched matrix multiply over arrays of pointers:
// for i in [0, batchCount):  C[i] := alpha*blas_impl::op(A[i])*blas_impl::op(B[i]) + beta*C[i].
//
// Structure follows src/herk.cpp exactly: one definition covering CPU (Fortran
// BLAS) and GPU (cuBLAS/hipBLAS), selected with `if constexpr` on the Backend
// template parameter, with the arms themselves guarded by the BUILD CAPABILITY
// macros (DDLA_HAS_CPU / DDLA_HAS_GPU) and the vendor chosen per scalar arm by
// the PER-TU macros (DDLA_USE_CUDA / DDLA_USE_HIP).
//
// Why the capability guard matters here: the previous design put a `#else`
// CPU arm inside src/gemmBatched.h selected by `!defined(DDLA_USE_CUDA) &&
// !defined(DDLA_USE_HIP)`.  In a dual build (DDLA_USE_CPU + DDLA_USE_CUDA)
// every TU defines DDLA_USE_CUDA, so that arm was never compiled -- while
// DDLA_HAS_CPU was 1 and the CPU instantiation of this function still existed.
// A CPU-backend call would therefore have reached cublasZgemmBatched with HOST
// pointers.  Guarding on DDLA_HAS_CPU removes that class of bug.
//
// The array-of-pointer arguments keep the `const T* const Aarray[]` shape: the
// pointers themselves are read-only to this wrapper (the vendor prototypes and
// the Fortran declaration differ on constness, which is why the GPU arm casts
// -- exactly as the pre-unification wrapper did).
//
// CPU arm: Fortran BLAS has no batched entry point, so it is a loop of per-matrix gemm calls, and that loop runs on the handle's host-
// thread budget (DdlaStream::host_threads, the same policy as geam): every OpenMP thread computes WHOLE products, so every core handles one
// matrix multiplication at a time.  The BLAS library's own thread count is held at one for the duration of the loop and restored after it
// (detail::ScopedBlasThreads, the way LibRI wraps its MKL gemm loop), otherwise a pthreads OpenBLAS would start a thread pool inside every
// call.  The matrices of one batch must not overlap in C (the cuBLAS contract; every caller writes one tile per entry).
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <algorithm>
#include <complex>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "blas_enums.h"
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "host_threads.h"

#if DDLA_HAS_CPU
extern "C" {
// Fortran BLAS entry points for the CPU arm below.  Guarded by the BUILD
// CAPABILITY (DDLA_HAS_CPU), not the per-TU vendor macro: in a dual build
// every TU defines DDLA_USE_CUDA/DDLA_USE_HIP while still instantiating the
// CPU templates, so `#if !defined(DDLA_USE_CUDA)` would leave these
// declarations out exactly where they are needed.
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
namespace {

// Calls call(i) for every product i of the batch.  With fewer than two products or two threads there is nothing to spread and the loop is
// serial with the BLAS keeping its own threads (one big product still gets a threaded BLAS); otherwise min(nthreads, count) OpenMP threads
// each run whole products while the BLAS is pinned to one thread.  Without OpenMP the loop stays serial and the BLAS is left alone.
template <typename Call>
void for_each_product(int nthreads, int count, Call&& call)
{
#ifdef _OPENMP
    const int nt = std::min(nthreads, count);
#else
    const int nt = 1;
    (void)nthreads;
#endif
    if (nt < 2) {
        for (int i = 0; i < count; ++i) call(i);
        return;
    }
    detail::ScopedBlasThreads one_blas_thread(1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nt)
#endif
    for (int i = 0; i < count; ++i) call(i);
}

} // namespace
} // namespace ddla
#endif // DDLA_HAS_CPU

// ---------------------------------------------------------------------------
// GPU arm: the vendor calls.  Kept as free gpublasGemmBatched overloads so the
// dispatch below stays a single BLAS_CHECK call per scalar type, exactly as
// ddla::gemm does it inline.  Private to this translation unit -- the public
// interface is the templated ddla::gemmBatched declared in <ddla/ddla.h>.
// ---------------------------------------------------------------------------
#if DDLA_HAS_GPU

namespace ddla {

inline gpublasStatus_t gpublasGemmBatched(
    gpublasHandle_t handle, gpublasOperation_t transa, gpublasOperation_t transb,
    int m, int n, int k,
    const std::complex<double>& alpha,
    const std::complex<double> *const Aarray[], int lda,
    const std::complex<double> *const Barray[], int ldb,
    const std::complex<double>& beta,
    std::complex<double> *const Carray[], int ldc,
    int batchCount
)
{
    #if DDLA_GPU_VENDOR
    return gpublasZgemmBatched(
        handle, transa, transb, m, n, k,
        (gpublasDoubleComplex*)(&alpha),
        (gpublasDoubleComplex**)(Aarray), lda,
        (gpublasDoubleComplex**)(Barray), ldb,
        (gpublasDoubleComplex*)(&beta),
        (gpublasDoubleComplex**)(Carray), ldc,
        batchCount
    );
    #else
    // Neither vendor macro: this can only happen if the translation unit was
    // built without a GPU language while DDLA_HAS_GPU is set, which the build
    // system does not do (see the LANGUAGE CUDA/HIP source lists).  Throw
    // rather than fall through to a host pointer.
    throw std::runtime_error("gpublasGemmBatched: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasGemmBatched(
    gpublasHandle_t handle, gpublasOperation_t transa, gpublasOperation_t transb,
    int m, int n, int k,
    const std::complex<float>& alpha,
    const std::complex<float> *const Aarray[], int lda,
    const std::complex<float> *const Barray[], int ldb,
    const std::complex<float>& beta,
    std::complex<float> *const Carray[], int ldc,
    int batchCount
)
{
    #if DDLA_GPU_VENDOR
    return gpublasCgemmBatched(
        handle, transa, transb, m, n, k,
        (gpublasComplex*)(&alpha),
        (gpublasComplex**)(Aarray), lda,
        (gpublasComplex**)(Barray), ldb,
        (gpublasComplex*)(&beta),
        (gpublasComplex**)(Carray), ldc,
        batchCount
    );
    #else
    throw std::runtime_error("gpublasGemmBatched: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasGemmBatched(
    gpublasHandle_t handle, gpublasOperation_t transa, gpublasOperation_t transb,
    int m, int n, int k,
    const float& alpha,
    const float *const Aarray[], int lda,
    const float *const Barray[], int ldb,
    const float& beta,
    float *const Carray[], int ldc,
    int batchCount
)
{
    #if DDLA_GPU_VENDOR
    return gpublasSgemmBatched(
        handle, transa, transb, m, n, k,
        &alpha,
        Aarray, lda,
        Barray, ldb,
        &beta,
        Carray, ldc,
        batchCount
    );
    #else
    throw std::runtime_error("gpublasGemmBatched: no GPU vendor selected at compile time");
    #endif
}

inline gpublasStatus_t gpublasGemmBatched(
    gpublasHandle_t handle, gpublasOperation_t transa, gpublasOperation_t transb,
    int m, int n, int k,
    const double& alpha,
    const double *const Aarray[], int lda,
    const double *const Barray[], int ldb,
    const double& beta,
    double *const Carray[], int ldc,
    int batchCount
)
{
    #if DDLA_GPU_VENDOR
    return gpublasDgemmBatched(
        handle, transa, transb, m, n, k,
        &alpha,
        Aarray, lda,
        Barray, ldb,
        &beta,
        Carray, ldc,
        batchCount
    );
    #else
    throw std::runtime_error("gpublasGemmBatched: no GPU vendor selected at compile time");
    #endif
}

} // namespace ddla

#endif // DDLA_HAS_GPU

namespace ddla {

namespace blas_impl {

template <DdlaBackend Backend, typename T>
void gemmBatched_check(
    const DdlaHandle_t& handle,
    char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const T* const Aarray[], int lda,
    const T* const Barray[], int ldb,
    const T& beta,
    T* const Carray[], int ldc,
    int batchCount)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU gemmBatched is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU gemmBatched is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("gemmBatched: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("gemmBatched: template backend ") +
            (Backend == DdlaBackend::CPU ? "CPU" : "GPU") +
            " does not match handle backend " +
            (actual == DdlaBackend::CPU ? "CPU" : "GPU"));
    }

    if constexpr (Backend == DdlaBackend::GPU) {
        (void)detail::to_blas_op(transa);
        (void)detail::to_blas_op(transb);
    }
}

template <DdlaBackend Backend, typename T>
void gemmBatched_run(
    const DdlaHandle_t& handle,
    char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const T* const Aarray[], int lda,
    const T* const Barray[], int ldb,
    const T& beta,
    T* const Carray[], int ldc,
    int batchCount)
{
    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        // One Fortran gemm per matrix, spread over the handle's host threads (see for_each_product).  The characters go to
        // the Fortran entry points unconverted.
        const int nthreads = handle->host_threads;
        if constexpr (std::is_same_v<T, float>) {
            for_each_product(nthreads, batchCount, [&](int i) {
                sgemm_(&transa, &transb, &m, &n, &k, &alpha, Aarray[i], &lda,
                       Barray[i], &ldb, &beta, Carray[i], &ldc);
            });
        } else if constexpr (std::is_same_v<T, double>) {
            for_each_product(nthreads, batchCount, [&](int i) {
                dgemm_(&transa, &transb, &m, &n, &k, &alpha, Aarray[i], &lda,
                       Barray[i], &ldb, &beta, Carray[i], &ldc);
            });
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            for_each_product(nthreads, batchCount, [&](int i) {
                cgemm_(&transa, &transb, &m, &n, &k, &alpha, Aarray[i], &lda,
                       Barray[i], &ldb, &beta, Carray[i], &ldc);
            });
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            for_each_product(nthreads, batchCount, [&](int i) {
                zgemm_(&transa, &transb, &m, &n, &k, &alpha, Aarray[i], &lda,
                       Barray[i], &ldb, &beta, Carray[i], &ldc);
            });
        } else {
            static_assert(sizeof(T) == 0, "gemmBatched: unsupported scalar type T");
        }
#else
        (void)handle; (void)transa; (void)transb; (void)m; (void)n; (void)k;
        (void)alpha; (void)Aarray; (void)lda; (void)Barray; (void)ldb;
        (void)beta; (void)Carray; (void)ldc; (void)batchCount;
#endif
    } else {
#if DDLA_HAS_GPU
        if constexpr (std::is_same_v<T, float> ||
                      std::is_same_v<T, double> ||
                      std::is_same_v<T, std::complex<float>> ||
                      std::is_same_v<T, std::complex<double>>) {
            BLAS_CHECK(gpublasGemmBatched(static_cast<gpublasHandle_t>(handle->blasH),
                                          blas_impl::op(transa),
                                          blas_impl::op(transb),
                                          m, n, k, alpha,
                                          Aarray, lda, Barray, ldb, beta,
                                          Carray, ldc, batchCount));
        } else {
            static_assert(sizeof(T) == 0, "gemmBatched: unsupported scalar type T");
        }
#else
        (void)handle; (void)transa; (void)transb; (void)m; (void)n; (void)k;
        (void)alpha; (void)Aarray; (void)lda; (void)Barray; (void)ldb;
        (void)beta; (void)Carray; (void)ldc; (void)batchCount;
#endif
    }
}

} // namespace blas_impl

template <DdlaBackend Backend, typename T>
void gemmBatched(
    const DdlaHandle_t& handle,
    char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const T* const Aarray[], int lda,
    const T* const Barray[], int ldb,
    const T& beta,
    T* const Carray[], int ldc,
    int batchCount)
{
    blas_impl::gemmBatched_check<Backend, T>(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
    blas_impl::gemmBatched_run<Backend, T>(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

#define INSTANTIATE_GEMM_BATCHED(BACKEND, TYPE)                                \
    template void gemmBatched<BACKEND, TYPE>(                                  \
        const DdlaHandle_t&, char, char, int, int, int,                        \
        const TYPE&, const TYPE* const[], int, const TYPE* const[], int,       \
        const TYPE&, TYPE* const[], int, int)

#if DDLA_HAS_CPU
INSTANTIATE_GEMM_BATCHED(DdlaBackend::CPU, float);
INSTANTIATE_GEMM_BATCHED(DdlaBackend::CPU, double);
INSTANTIATE_GEMM_BATCHED(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_GEMM_BATCHED(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_GEMM_BATCHED(DdlaBackend::GPU, float);
INSTANTIATE_GEMM_BATCHED(DdlaBackend::GPU, double);
INSTANTIATE_GEMM_BATCHED(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_GEMM_BATCHED(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_GEMM_BATCHED


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::gemmBatched_check<BACKEND, TYPE>( const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const TYPE& alpha, const TYPE* const Aarray[], int lda, const TYPE* const Barray[], int ldb, const TYPE& beta, TYPE* const Carray[], int ldc, int batchCount); \
    template void blas_impl::gemmBatched_run<BACKEND, TYPE>( const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const TYPE& alpha, const TYPE* const Aarray[], int lda, const TYPE* const Barray[], int ldb, const TYPE& beta, TYPE* const Carray[], int ldc, int batchCount);
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
