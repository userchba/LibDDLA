#ifndef DDLA_BLAS_IMPL_H
#define DDLA_BLAS_IMPL_H
#include <ddla/ddla.h>
#include "ddla_connector.h"
#include "plan_util.h"
#include <algorithm>
#include <type_traits>
#include <cctype>
#include <complex>
#include <cstddef>
namespace ddla { namespace blas_impl {
// Scratch-free operations expose check/run only. Mixed GEMM, GEMMT and TRTRM
// need scratch: a size function (backend bytes) and a run that takes them.
// These conversions assume the caller has already checked the flags.
inline char upper(char c) noexcept { return static_cast<char>(std::toupper(static_cast<unsigned char>(c))); }
#if DDLA_HAS_GPU
inline gpublasOperation_t op(char c) noexcept { c=upper(c); return c=='N' ? GPUBLAS_OP_N : c=='T' ? GPUBLAS_OP_T : GPUBLAS_OP_C; }
inline gpublasFillMode_t fill(char c) noexcept { return upper(c)=='U' ? GPUBLAS_FILL_MODE_UPPER : GPUBLAS_FILL_MODE_LOWER; }
inline gpublasSideMode_t side(char c) noexcept { return upper(c)=='L' ? GPUBLAS_SIDE_LEFT : GPUBLAS_SIDE_RIGHT; }
inline gpublasDiagType_t diag(char c) noexcept { return upper(c)=='U' ? GPUBLAS_DIAG_UNIT : GPUBLAS_DIAG_NON_UNIT; }
#endif
// diagonal_length describes the complete replicated vector for overlap checks.
template <DdlaBackend Backend, typename T, typename U>
void dgmm_check(const DdlaHandle_t& h, char side, int m, int n,
    const T* A, int lda, const U* x, int incx, T* C, int ldc, int diagonal_length);
template <DdlaBackend Backend, typename T, typename U>
void dgmm_run(const DdlaHandle_t& h, char side, int m, int n,
    const T* A, int lda, const U* x, int incx, T* C, int ldc);

template <DdlaBackend Backend, typename T>
void gemm_check(
    const DdlaHandle_t& handle,
    char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const T& beta,
    T* C, int ldc);
template <DdlaBackend Backend, typename T>
void gemm_run(
    const DdlaHandle_t& handle,
    char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const T& beta,
    T* C, int ldc);

template <DdlaBackend Backend, typename T, typename TA = T, typename TB = T>
void gemmBatched_check(
    const DdlaHandle_t& handle,
    char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const TA* const Aarray[], int lda,
    const TB* const Barray[], int ldb,
    const T& beta,
    T* const Carray[], int ldc,
    int batchCount);
template <DdlaBackend Backend, typename T, typename TA = T, typename TB = T>
void gemmBatched_run(
    const DdlaHandle_t& handle,
    char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const TA* const Aarray[], int lda,
    const TB* const Barray[], int ldb,
    const T& beta,
    T* const Carray[], int ldc,
    int batchCount, void* work = nullptr, std::size_t bytes = 0);

template <DdlaBackend Backend, typename T>
void geam_check(const DdlaHandle_t& handle, char transa, char transb, int m, int n,
          const T& alpha, const T* A, int lda,
          const T& beta, const T* B, int ldb,
          T* C, int ldc);
template <DdlaBackend Backend, typename T>
void geam_run(const DdlaHandle_t& handle, char transa, char transb, int m, int n,
          const T& alpha, const T* A, int lda,
          const T& beta, const T* B, int ldb,
          T* C, int ldc);

// Prechecked matrix scaling shared by ordinary and real-coefficient calls.
template <DdlaBackend Backend, typename T, typename Alpha = T>
void matrix_scal_run(const DdlaHandle_t& h,int m,int n,const Alpha& alpha,T* a,int lda);
// Matrix-region scaling. GPU arrays remain backend-resident; no workspace.
template <typename T, typename Alpha = T>
void matrix_scal_gpu_run(const DdlaHandle_t& h, int m, int n, const Alpha& alpha,
                         T* a, int lda, T* const* arrays, int count);
template <DdlaBackend Backend, typename T, typename Alpha = T>
void scalBatched_run(const DdlaHandle_t& h, int m, int n, const Alpha& alpha,
                     T* const* arrays, int lda, int count);

template <DdlaBackend Backend, typename T>
void axpy_check(const DdlaHandle_t& handle, int n, const T& alpha,
          const T* x, int incx, T* y, int incy);
template <DdlaBackend Backend, typename T>
void axpy_run(const DdlaHandle_t& handle, int n, const T& alpha,
          const T* x, int incx, T* y, int incy);

template <DdlaBackend Backend, typename T>
void dotc_check(
    const DdlaHandle_t& handle,
    int n,
    const T* x, int incx,
    const T* y, int incy,
    T* result);
template <DdlaBackend Backend, typename T>
void dotc_run(
    const DdlaHandle_t& handle,
    int n,
    const T* x, int incx,
    const T* y, int incy,
    T* result);

template <DdlaBackend Backend, typename T>
void scal_check(const DdlaHandle_t& handle, int n, const T& alpha, T* x, int incx);
template <DdlaBackend Backend, typename T>
void scal_run(const DdlaHandle_t& handle, int n, const T& alpha, T* x, int incx);

template <DdlaBackend Backend, typename T>
void swap_check(const DdlaHandle_t& handle, int n, T* x, int incx, T* y, int incy);
template <DdlaBackend Backend, typename T>
void swap_run(const DdlaHandle_t& handle, int n, T* x, int incx, T* y, int incy);

template <DdlaBackend Backend, typename T>
void iamax_check(const DdlaHandle_t& handle, int n, const T* x, int incx, int& result);
template <DdlaBackend Backend, typename T>
void iamax_run(const DdlaHandle_t& handle, int n, const T* x, int incx, int& result);

template <DdlaBackend Backend, typename T>
void geru_check(const DdlaHandle_t& handle, int m, int n, const T& alpha,
          const T* x, int incx, const T* y, int incy, T* A, int lda);
template <DdlaBackend Backend, typename T>
void geru_run(const DdlaHandle_t& handle, int m, int n, const T& alpha,
          const T* x, int incx, const T* y, int incy, T* A, int lda);

template <DdlaBackend Backend, typename T>
void gerc_check(const DdlaHandle_t& handle, int m, int n,
          const T& alpha, const T* x, int incx,
          const T* y, int incy, T* A, int lda);
template <DdlaBackend Backend, typename T>
void gerc_run(const DdlaHandle_t& handle, int m, int n,
          const T& alpha, const T* x, int incx,
          const T* y, int incy, T* A, int lda);

template <DdlaBackend Backend, typename T>
void nrm2_check(
    const DdlaHandle_t& handle,
    int n,
    const T* x, int incx,
    typename real_type<T>::type* result);
template <DdlaBackend Backend, typename T>
void nrm2_run(
    const DdlaHandle_t& handle,
    int n,
    const T* x, int incx,
    typename real_type<T>::type* result);

template <DdlaBackend Backend, typename T>
void gemv_check(
    const DdlaHandle_t& handle,
    char trans, int m, int n,
    const T& alpha,
    const T* A, int lda,
    const T* x, int incx,
    const T& beta,
    T* y, int incy);
template <DdlaBackend Backend, typename T>
void gemv_run(
    const DdlaHandle_t& handle,
    char trans, int m, int n,
    const T& alpha,
    const T* A, int lda,
    const T* x, int incx,
    const T& beta,
    T* y, int incy);

template <DdlaBackend Backend, typename T>
void herk_check(
    const DdlaHandle_t& handle,
    char uplo, char trans,
    int n, int k,
    const typename real_type<T>::type& alpha,
    const T* A, int lda,
    const typename real_type<T>::type& beta,
    T* C, int ldc);
template <DdlaBackend Backend, typename T>
void herk_run(
    const DdlaHandle_t& handle,
    char uplo, char trans,
    int n, int k,
    const typename real_type<T>::type& alpha,
    const T* A, int lda,
    const typename real_type<T>::type& beta,
    T* C, int ldc);

template <DdlaBackend Backend, typename T>
void her2k_check(
    const DdlaHandle_t& handle,
    char uplo, char trans,
    int n, int k,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const typename real_type<T>::type& beta,
    T* C, int ldc);
template <DdlaBackend Backend, typename T>
void her2k_run(
    const DdlaHandle_t& handle,
    char uplo, char trans,
    int n, int k,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const typename real_type<T>::type& beta,
    T* C, int ldc);

template <DdlaBackend Backend, typename T>
void hemv_check(
    const DdlaHandle_t& handle,
    char uplo,
    int n,
    const T& alpha,
    const T* A, int lda,
    const T* x, int incx,
    const T& beta,
    T* y, int incy);
template <DdlaBackend Backend, typename T>
void hemv_run(
    const DdlaHandle_t& handle,
    char uplo,
    int n,
    const T& alpha,
    const T* A, int lda,
    const T* x, int incx,
    const T& beta,
    T* y, int incy);

template <DdlaBackend Backend, typename T>
void hemm_check(
    const DdlaHandle_t& handle,
    char side, char uplo,
    int m, int n,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const T& beta,
    T* C, int ldc);
template <DdlaBackend Backend, typename T>
void hemm_run(
    const DdlaHandle_t& handle,
    char side, char uplo,
    int m, int n,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const T& beta,
    T* C, int ldc);

template <DdlaBackend Backend, typename T>
void her2_check(
    const DdlaHandle_t& handle,
    char uplo,
    int n,
    const T& alpha,
    const T* x, int incx,
    const T* y, int incy,
    T* A, int lda);
template <DdlaBackend Backend, typename T>
void her2_run(
    const DdlaHandle_t& handle,
    char uplo,
    int n,
    const T& alpha,
    const T* x, int incx,
    const T* y, int incy,
    T* A, int lda);

template <DdlaBackend Backend, typename T>
void trmv_check(
    const DdlaHandle_t& handle,
    char uplo, char trans, char diag,
    int n,
    const T* A, int lda,
    T* x, int incx);
template <DdlaBackend Backend, typename T>
void trmv_run(
    const DdlaHandle_t& handle,
    char uplo, char trans, char diag,
    int n,
    const T* A, int lda,
    T* x, int incx);

template <DdlaBackend Backend, typename T>
void trsm_check(
    const DdlaHandle_t& handle,
    char side, char uplo, char trans, char diag,
    int m, int n,
    const T& alpha,
    T* A, int lda,
    T* B, int ldb);
template <DdlaBackend Backend, typename T>
void trsm_run(
    const DdlaHandle_t& handle,
    char side, char uplo, char trans, char diag,
    int m, int n,
    const T& alpha,
    T* A, int lda,
    T* B, int ldb);

template <DdlaBackend Backend, typename T>
void trmm_check(
    const DdlaHandle_t& handle,
    char side, char uplo, char trans, char diag,
    int m, int n,
    const T& alpha,
    const T* A, int lda,
    T* B, int ldb);
template <DdlaBackend Backend, typename T>
void trmm_run(
    const DdlaHandle_t& handle,
    char side, char uplo, char trans, char diag,
    int m, int n,
    const T& alpha,
    const T* A, int lda,
    T* B, int ldb);

template <DdlaBackend Backend, typename R>
void gemm_check(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
          const std::complex<R>& alpha, const R* A, int lda,
          const std::complex<R>* B, int ldb, const std::complex<R>& beta,
          std::complex<R>* C, int ldc);
template <DdlaBackend Backend, typename R>
void gemm_run(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
          const std::complex<R>& alpha, const R* A, int lda,
          const std::complex<R>* B, int ldb, const std::complex<R>& beta,
          std::complex<R>* C, int ldc, void* backend_work, std::size_t backend_bytes);
template <DdlaBackend Backend, typename R>
void gemm_check(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
          const std::complex<R>& alpha, const std::complex<R>* A, int lda,
          const R* B, int ldb, const std::complex<R>& beta,
          std::complex<R>* C, int ldc);
template <DdlaBackend Backend, typename R>
void gemm_run(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
          const std::complex<R>& alpha, const std::complex<R>* A, int lda,
          const R* B, int ldb, const std::complex<R>& beta,
          std::complex<R>* C, int ldc, void* backend_work, std::size_t backend_bytes);
// Backend bytes of an m x n x k mixed product (complex_left: complex op(A) times
// real op(B)), for either gemm_run above. CPU handles: per-worker slots
// [ packed operand | real result ] of the tiling for h->host_threads (the run
// re-derives it, so size and run under the same thread budget); GPU handles:
// one [ packed operand | real product ] region of a <= 512-column panel, nothing
// for the fused small real-by-complex kernel. m, n or k <= 0 needs nothing; a
// caller that runs several shapes on one buffer keeps the largest.
template <typename R>
void mixed_gemm_workspace_size(const DdlaHandle_t& h, int m, int n, int k, bool complex_left,
                               std::size_t* backend_bytes);
template <DdlaBackend Backend, typename T>
void symm_check(const DdlaHandle_t& h, char side, char uplo, int m, int n,
    const T& alpha, const T* A, int lda, const T* B, int ldb, const T& beta, T* C, int ldc);
template <DdlaBackend Backend, typename T>
void symm_run(const DdlaHandle_t& h, char side, char uplo, int m, int n,
    const T& alpha, const T* A, int lda, const T* B, int ldb, const T& beta, T* C, int ldc);
template <DdlaBackend Backend, typename T>
void syrk_check(const DdlaHandle_t& h, char uplo, char trans, int n, int k,
    const T& alpha, const T* A, int lda, const T& beta, T* C, int ldc);
template <DdlaBackend Backend, typename T>
void syrk_run(const DdlaHandle_t& h, char uplo, char trans, int n, int k,
    const T& alpha, const T* A, int lda, const T& beta, T* C, int ldc);
template <DdlaBackend Backend, typename T>
void syr2k_check(const DdlaHandle_t& h, char uplo, char trans, int n, int k,
    const T& alpha, const T* A, int lda, const T* B, int ldb, const T& beta, T* C, int ldc);
template <DdlaBackend Backend, typename T>
void syr2k_run(const DdlaHandle_t& h, char uplo, char trans, int n, int k,
    const T& alpha, const T* A, int lda, const T* B, int ldb, const T& beta, T* C, int ldc);
// Backend bytes of an order-n trtrm: on a CPU handle the product panel
// (n x min(n, 256) elements of T); a GPU handle needs none. The run below
// uses the panel only when alpha != 0.
template <typename T> void trtrm_workspace_size(const DdlaHandle_t& handle, int n, std::size_t* backend_bytes);
template <DdlaBackend Backend, typename T>
void trtrm_check(const DdlaHandle_t& handle, char uplo_a, char transa, char uplo_b, char transb,
           int n, const T& alpha, const T* A, int lda, const T* B, int ldb,
           const T& beta, T* C, int ldc);
template <DdlaBackend Backend, typename T>
void trtrm_run(const DdlaHandle_t& handle, char uplo_a, char transa, char uplo_b, char transb,
           int n, const T& alpha, const T* A, int lda, const T* B, int ldb,
           const T& beta, T* C, int ldc, void* backend_work, std::size_t backend_bytes);
template <DdlaBackend Backend, typename T, typename TA = T, typename TB = T>
void gemmt_check(const DdlaHandle_t& h, char uplo, char ta, char tb,
    int n, int k, const T& alpha, const TA* a, int lda, const TB* b, int ldb,
    const T& beta, T* c, int ldc);
// Backend scratch bytes only; this query performs no allocation.
template <typename T, typename TA = T, typename TB = T>
void gemmt_workspace_size(const DdlaHandle_t& h, char transa, char transb,
    int n, int k, std::size_t* backend_bytes);
template <DdlaBackend Backend, typename T, typename TA = T, typename TB = T>
void gemmt_run(const DdlaHandle_t& h, char uplo, char ta, char tb,
    int n, int k, const T& alpha, const TA* a, int lda, const TB* b, int ldb,
    const T& beta, T* c, int ldc, void* backend_work, std::size_t backend_bytes);
// Local products shared by distributed and structured operations.  A same-type product is the plain BLAS
// call and needs no scratch; a mixed real/complex product takes its scratch
// from work (bytes), sized with local_gemm_workspace_size for every shape the
// caller runs on it (the largest wins).
template <typename T, typename TA, typename TB>
void local_gemm_workspace_size([[maybe_unused]] const DdlaHandle_t& h, [[maybe_unused]] int m,
                               [[maybe_unused]] int n, [[maybe_unused]] int k, std::size_t* backend_bytes)
{
    *backend_bytes=0;
    if constexpr (!std::is_same_v<T,TA> || !std::is_same_v<T,TB>)
        mixed_gemm_workspace_size<typename real_type<T>::type>(
            h,m,n,k,!std::is_same_v<TA,typename real_type<T>::type>,backend_bytes);
}

template <DdlaBackend Backend, typename T, typename TA, typename TB>
void local_gemm_run(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k, const T& alpha,
                    const TA* a, int lda, const TB* b, int ldb, const T& beta, T* c, int ldc,
                    [[maybe_unused]] void* work, [[maybe_unused]] std::size_t bytes)
{
    if constexpr (std::is_same_v<T,TA> && std::is_same_v<T,TB>)
        gemm_run<Backend,T>(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
    else
        gemm_run<Backend,typename real_type<T>::type>(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,
                                                                 work,bytes);
}


// Mixed batches keep device pointer tables on device and use bounded chunks.
// Small GPU products reuse the existing fused mixed-GEMM tile kernel.
template <typename R>
int mixed_gemmBatched_capacity(int threads,int m,int n,int k,bool complex_left,int count,int& depth,std::size_t parallel_bytes=0)
{
    const std::size_t panel=std::min(n,512);
    const auto slot=[&](int kk) {
        return (2*(complex_left ? std::size_t(m)*kk : std::size_t(kk)*panel)
                +2*std::size_t(m)*panel)*sizeof(R);
    };
    const std::size_t full=slot(k);
    const int original=int(std::min<std::size_t>(std::max(0,count),
        std::min<std::size_t>(32,std::max<std::size_t>(1,(32u<<20)/std::max<std::size_t>(1,full)))));
    depth=k;
    const std::size_t original_bytes=detail::padded_bytes<R>(std::size_t(original)*full/sizeof(R))
        +3*detail::padded_bytes<R*>(original);
    // A few large matrices can reuse the existing full-team scalar path.
    if(threads>0 && parallel_bytes>0 && parallel_bytes<=original_bytes) {
        depth=0;return 1;
    }
    if(threads<=0)return original;
    if(m>256 || n>256 || k<=128)return std::min(original,std::max(1,threads));
    // Long dot products otherwise leave only one or two worker slots in the
    // packing budget. Stream K panels and keep each worker's real sum until
    // the final alpha/beta update, without increasing the original allocation.
    depth=128;
    const std::size_t capacity=std::max<std::size_t>(1,std::size_t(original)*full/slot(depth));
    int slots=int(std::min<std::size_t>(std::max(0,count),
        std::min<std::size_t>(std::max(1,threads),capacity)));
    while(slots>1 && detail::padded_bytes<R>(std::size_t(slots)*slot(depth)/sizeof(R))
            +3*detail::padded_bytes<R*>(slots)>original_bytes)--slots;
    return slots;
}

template <typename R>
void mixed_gemmBatched_workspace_size(const DdlaHandle_t& h,int m,int n,int k,
    bool complex_left,int count,std::size_t* bytes);
template <typename R, bool ComplexLeft>
void mixed_gemmBatched_gpu_run(const DdlaHandle_t& h,char ta,char tb,int m,int n,int k,
    const std::complex<R>& alpha,const void* const* a,int lda,const void* const* b,int ldb,
    const std::complex<R>& beta,std::complex<R>* const* c,int ldc,int count,void* work,std::size_t bytes);


// Extra arithmetic workspace is zero for same-type batches.
template <typename T,typename TA=T,typename TB=T>
void local_gemmBatched_workspace_size(const DdlaHandle_t& h,int m,int n,int k,int count,std::size_t* bytes)
{
    *bytes=0;
    if constexpr (!std::is_same_v<T,TA> || !std::is_same_v<T,TB>)
        mixed_gemmBatched_workspace_size<typename real_type<T>::type>(h,m,n,k,std::is_same_v<T,TA>,count,bytes);
}

}} // namespace ddla::blas_impl
#endif
