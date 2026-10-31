#ifndef DDLA_BLAS_IMPL_H
#define DDLA_BLAS_IMPL_H
#include <ddla/ddla.h>
#include "workspace.h"
#include "gemm_mixed_cpu.h"
#include "ddla_connector.h"
#include <cctype>
namespace ddla { namespace blas_impl {
// Scratch-free operations expose check/run only. Size queries below belong
// only to mixed GEMM and TRTRM, whose supported paths require temporary memory.
// These conversions assume the caller has already checked the flags.
inline char upper(char c) noexcept { return static_cast<char>(std::toupper(static_cast<unsigned char>(c))); }
inline gpublasOperation_t op(char c) noexcept { c=upper(c); return c=='N' ? GPUBLAS_OP_N : c=='T' ? GPUBLAS_OP_T : GPUBLAS_OP_C; }
inline gpublasFillMode_t fill(char c) noexcept { return upper(c)=='U' ? GPUBLAS_FILL_MODE_UPPER : GPUBLAS_FILL_MODE_LOWER; }
inline gpublasSideMode_t side(char c) noexcept { return upper(c)=='L' ? GPUBLAS_SIDE_LEFT : GPUBLAS_SIDE_RIGHT; }
inline gpublasDiagType_t diag(char c) noexcept { return upper(c)=='U' ? GPUBLAS_DIAG_UNIT : GPUBLAS_DIAG_NON_UNIT; }
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
    int batchCount);
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
    int batchCount);

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
void gemm_run(const DdlaHandle_t& h, Workspace workspace, const detail::MixedGemmWorkspaceSize& query, char ta, char tb, int m, int n, int k,
          const std::complex<R>& alpha, const R* A, int lda,
          const std::complex<R>* B, int ldb, const std::complex<R>& beta,
          std::complex<R>* C, int ldc);
template <DdlaBackend Backend, typename R>
void gemm_check(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
          const std::complex<R>& alpha, const std::complex<R>* A, int lda,
          const R* B, int ldb, const std::complex<R>& beta,
          std::complex<R>* C, int ldc);
template <DdlaBackend Backend, typename R>
void gemm_run(const DdlaHandle_t& h, Workspace workspace, const detail::MixedGemmWorkspaceSize& query, char ta, char tb, int m, int n, int k,
          const std::complex<R>& alpha, const std::complex<R>* A, int lda,
          const R* B, int ldb, const std::complex<R>& beta,
          std::complex<R>* C, int ldc);
template <DdlaBackend Backend, typename R>
detail::MixedGemmWorkspaceSize mixed_gemm_workspace_size(const DdlaHandle_t&,int,int,int,bool,bool);
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
// Start disposable host scratch lifetimes once, outside repeated execution.
template <typename T> inline void trtrm_prepare_workspace(Workspace work) noexcept
{ detail::workspace_start_objects(detail::workspace_at<T>(work.host,0,work.host.size()/sizeof(T))); }
template <DdlaBackend Backend, typename T> WorkspaceSize trtrm_workspace_size(int,const T&);
template <DdlaBackend Backend, typename T>
void trtrm_check(const DdlaHandle_t& handle, char uplo, int n, const T& alpha,
           const T* A, int lda, const T* B, int ldb,
           const T& beta, T* C, int ldc);
template <DdlaBackend Backend, typename T>
void trtrm_check(const DdlaHandle_t& handle, char uplo_a, char transa, char uplo_b, char transb,
           int n, const T& alpha, const T* A, int lda, const T* B, int ldb,
           const T& beta, T* C, int ldc);
template <DdlaBackend Backend, typename T>
void trtrm_run(const DdlaHandle_t& handle, Workspace workspace, char uplo, int n, const T& alpha,
           const T* A, int lda, const T* B, int ldb,
           const T& beta, T* C, int ldc);
template <DdlaBackend Backend, typename T>
void trtrm_run(const DdlaHandle_t& handle, Workspace workspace, char uplo_a, char transa, char uplo_b, char transb,
           int n, const T& alpha, const T* A, int lda, const T* B, int ldb,
           const T& beta, T* C, int ldc);
template <DdlaBackend Backend, typename T>
void gemm_sym_check(const DdlaHandle_t& h, char symmetry, char uplo, char ta, char tb,
    int n, int k, const T& alpha, const T* a, int lda, const T* b, int ldb,
    const T& beta, T* c, int ldc);
template <DdlaBackend Backend, typename T>
void gemm_sym_run(const DdlaHandle_t& h, char symmetry, char uplo, char ta, char tb,
    int n, int k, const T& alpha, const T* a, int lda, const T* b, int ldb,
    const T& beta, T* c, int ldc);
}} // namespace ddla::blas_impl
#endif
