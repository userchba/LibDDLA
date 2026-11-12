#ifndef DDLA_BACKEND_API_H
#define DDLA_BACKEND_API_H

// Internal backend-specific overloads. This file is never installed.
// Public callers use <ddla/ddla.h>; only implementation code selects Backend.
#include <ddla/ddla.h>

namespace ddla {

template <DdlaBackend Backend, typename T, typename U>
void dgmm(const DdlaHandle_t& handle, char side, int m, int n,
    const T* A, int lda, const U* x, int incx, T* C, int ldc);

template <DdlaBackend Backend, typename T, typename U>
void pdgmm(const DdlaHandle_t& handle, char side, int m, int n,
    const T* A, const int* descA, const U* x, int incx, T* C, const int* descC);

template <DdlaBackend Backend, typename T>
void ptrtrs(
    const DdlaHandle_t& handle,
    char side, char uplo, char trans, char diag,
    int m, int n,
    T* A, const int* descA,
    T* B, const int* descB
);

template <DdlaBackend Backend, typename T>
void ptrmm(
    const DdlaHandle_t& handle, char side, char uplo, char trans, char diag,
    int m, int n, const T& alpha,
    const T* A, const int* descA, T* B, const int* descB
);

template <DdlaBackend Backend, typename T>
void phemm(const DdlaHandle_t& handle, char side, char uplo, int m, int n,
    const T& alpha, const T* A, const int* descA,
    const T* B, const int* descB, const T& beta, T* C, const int* descC);

template <DdlaBackend Backend, typename R>
void phemm(const DdlaHandle_t& handle, char side, char uplo, int m, int n,
    const std::complex<R>& alpha, const R* A, const int* descA,
    const std::complex<R>* B, const int* descB, const std::complex<R>& beta,
    std::complex<R>* C, const int* descC);

template <DdlaBackend Backend, typename R>
void phemm(const DdlaHandle_t& handle, char side, char uplo, int m, int n,
    const std::complex<R>& alpha, const std::complex<R>* A, const int* descA,
    const R* B, const int* descB, const std::complex<R>& beta,
    std::complex<R>* C, const int* descC);

template <DdlaBackend Backend, typename T>
void phemmt(const DdlaHandle_t& handle,char side,char uploA,char uploC,int n,
    const T& alpha,const T* A,const int* descA,const T* B,const int* descB,
    const T& beta,T* C,const int* descC);

template <DdlaBackend Backend, typename R>
void phemmt(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n,
    const std::complex<R>& alpha, const R* A, const int* descA,
    const std::complex<R>* B, const int* descB, const std::complex<R>& beta,
    std::complex<R>* C, const int* descC);

template <DdlaBackend Backend, typename R>
void phemmt(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n,
    const std::complex<R>& alpha, const std::complex<R>* A, const int* descA,
    const R* B, const int* descB, const std::complex<R>& beta,
    std::complex<R>* C, const int* descC);

template <DdlaBackend Backend, typename T>
void psymm(const DdlaHandle_t& handle, char side, char uplo, int m, int n,
    const T& alpha, const T* A, const int* descA,
    const T* B, const int* descB, const T& beta, T* C, const int* descC);

template <DdlaBackend Backend, typename T>
void psymmt(const DdlaHandle_t& handle,char side,char uploA,char uploC,int n,
    const T& alpha,const T* A,const int* descA,const T* B,const int* descB,
    const T& beta,T* C,const int* descC);

template <DdlaBackend Backend, typename T>
void pgemmt(const DdlaHandle_t& handle, char uplo,
    char transa, char transb, int n, int k,
    const T& alpha, const T* A, const int* descA,
    const T* B, const int* descB, const T& beta, T* C, const int* descC);

template <DdlaBackend Backend, typename R>
void pgemmt(const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k,
    const std::complex<R>& alpha, const R* A, const int* descA,
    const std::complex<R>* B, const int* descB, const std::complex<R>& beta,
    std::complex<R>* C, const int* descC);

template <DdlaBackend Backend, typename R>
void pgemmt(const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k,
    const std::complex<R>& alpha, const std::complex<R>* A, const int* descA,
    const R* B, const int* descB, const std::complex<R>& beta,
    std::complex<R>* C, const int* descC);

template <DdlaBackend Backend, typename T>
void psyrk(const DdlaHandle_t& handle, char uplo, char trans, int n, int k,
    const T& alpha, const T* A, const int* descA,
    const T& beta, T* C, const int* descC);

template <DdlaBackend Backend, typename T>
void psyr2k(const DdlaHandle_t& handle, char uplo, char trans, int n, int k,
    const T& alpha, const T* A, const int* descA, const T* B, const int* descB,
    const T& beta, T* C, const int* descC);

template <DdlaBackend Backend, typename T>
void pherk(const DdlaHandle_t& handle, char uplo, char trans, int n, int k,
    const typename real_type<T>::type& alpha, const T* A, const int* descA,
    const typename real_type<T>::type& beta, T* C, const int* descC);

template <DdlaBackend Backend, typename T>
void pher2k(const DdlaHandle_t& handle, char uplo, char trans, int n, int k,
    const T& alpha, const T* A, const int* descA, const T* B, const int* descB,
    const typename real_type<T>::type& beta, T* C, const int* descC);

template <DdlaBackend Backend, typename T>
void symm(const DdlaHandle_t& handle, char side, char uplo, int m, int n,
    const T& alpha, const T* A, int lda, const T* B, int ldb,
    const T& beta, T* C, int ldc);

template <DdlaBackend Backend, typename T>
void syr2k(const DdlaHandle_t& handle, char uplo, char trans, int n, int k,
    const T& alpha, const T* A, int lda, const T* B, int ldb,
    const T& beta, T* C, int ldc);

template <DdlaBackend Backend, typename T>
void plapiv(
    const DdlaHandle_t& handle,
    char direc, char rowcol, char pivroc,
    int m, int n,
    T* A, const int* descA,
    const int* ipiv, const int* descIP,
    int* iwork
);

template <DdlaBackend Backend, typename T>
void pswap(
    const DdlaHandle_t& handle,
    int N,
    T* A, int ia, int ja, const int* descA, int inca,
    T* B, int ib, int jb, const int* descB, int incb
);

template <DdlaBackend Backend, typename T>
void gather(const DdlaHandle_t& handle, int m, int n,
            const T* A, int ia, int ja, const int* descA,
            int root, T* dst, int lddst);

template <DdlaBackend Backend, typename T>
void scatter(const DdlaHandle_t& handle, int m, int n,
             T* A, int ia, int ja, const int* descA,
             int root, const T* src, int ldsrc);

template <DdlaBackend Backend, typename T>
void pgetrf(
    const DdlaHandle_t& handle,
    int m, int n,
    T* A, int ia, int ja, const int* descA,
    int* ipiv, // host
    int& info  // host
);

template <DdlaBackend Backend, typename T>
void pgetrf_nopiv(
    const DdlaHandle_t& handle,
    int m, int n,
    T* A, const int* descA,
    int& info  // host
);

template <DdlaBackend Backend, typename T>
void pgetrs(
    const DdlaHandle_t& handle,
    char side, char trans, int n, int nrhs,
    T* A, const int* descA,
    const int* ipiv, // host
    T* B, const int* descB
);

template <DdlaBackend Backend, typename T>
void pgetrs_nopiv(
    const DdlaHandle_t& handle,
    char side, char trans, int n, int nrhs,
    T* A, const int* descA,
    T* B, const int* descB
);

template <DdlaBackend Backend, typename T>
void pgesv(
    const DdlaHandle_t& handle,
    char side, char trans, int n, int nrhs,
    T* A, const int* descA,
    T* B, const int* descB
);

template <DdlaBackend Backend, typename T>
void pgesv_nopiv(
    const DdlaHandle_t& handle,
    char side, char trans, int n, int nrhs,
    T* A, const int* descA,
    T* B, const int* descB
);

template <DdlaBackend Backend, typename T>
void pgetrf_bpiv(
    const DdlaHandle_t& handle,
    int m, int n,
    T* A, const int* descA,
    int* pivots, // host for CPU, device for GPU
    int& info  // host
);

template <DdlaBackend Backend, typename T>
void pgetrs_bpiv(
    const DdlaHandle_t& handle,
    char side, char trans, int n, int nrhs,
    T* A, const int* descA,
    int* pivots, // host for CPU, device for GPU
    T* B, const int* descB
);

template <DdlaBackend Backend, typename T>
void pgesv_bpiv(
    const DdlaHandle_t& handle,
    char side, char trans, int n, int nrhs,
    T* A, const int* descA,
    T* B, const int* descB
);

template <DdlaBackend Backend, typename T>
void pgemm(
    const DdlaHandle_t& handle,
    char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const T* A, const int* descA,
    const T* B, const int* descB,
    const T& beta,
    T* C, const int* descC
);

template <DdlaBackend Backend, typename T>
void pgeqrf(const DdlaHandle_t& handle, int m, int n,
            T* A, const int* descA, T* tau, int& info);

template <DdlaBackend Backend, typename T>
void pgeqrs(const DdlaHandle_t& handle, int m, int n, int nrhs,
            const T* A, const int* descA, const T* tau,
            T* B, const int* descB, int& info);

template <DdlaBackend Backend, typename T>
void pbdsdc(const DdlaHandle_t& handle, char uplo, int n,
            T* d, T* e, T* s, T* U, T* V, int ld, int& info);

template <DdlaBackend Backend, typename T>
void pgebrd(const DdlaHandle_t& handle, int m, int n,
            T* A, const int* descA,
            typename real_type<T>::type* d, typename real_type<T>::type* e,
            T* tauq, T* taup,
            int& info);

template <DdlaBackend Backend, typename T>
void pgesvd(const DdlaHandle_t& handle,
            char jobu, char jobvt,
            int m, int n,
            T* A, const int* descA,
            typename real_type<T>::type* s,
            T* U, const int* descU,
            T* VT, const int* descVT,
            int& info);

template <DdlaBackend Backend, typename T>
void punmbr(const DdlaHandle_t& handle,
            char vect, char side, char trans,
            int m, int n, int k,
            const T* A, const int* descA,
            const T* tau,
            T* C, const int* descC,
            int& info);

template <DdlaBackend Backend, typename T>
void pgeqlf(const DdlaHandle_t& handle, int m, int n,
            T* A, const int* descA, T* tau, int& info);

template <DdlaBackend Backend, typename T>
void pgeadd(
    const DdlaHandle_t& handle, char transa, char transb,
    int m, int n,
    const T& alpha, const T* A, int ia, int ja, const int* descA,
    const T& beta, const T* B, int ib, int jb, const int* descB,
    T* C, int ic, int jc, const int* descC
);

template <DdlaBackend Backend, typename T1, typename T2>
void pdam(const DdlaHandle_t& handle, const T1& alpha, T2* A, const int* descA, int n = -1);

template <DdlaBackend Backend, typename T>
void ppotrf(
    const DdlaHandle_t& handle,
    char uplo, int n,
    T* A, int ia, int ja, const int* descA,
    int& info // host pointer
);

template <DdlaBackend Backend, typename T>
void potrf_bottom_right(
    char uplo, int n, T* A, int lda,
    int& info, const DdlaHandle_t& handle
);

template <DdlaBackend Backend, typename T>
void ppotrf_bottom_right(
    const DdlaHandle_t& handle,
    char uplo, int n, T* A,
    const int* descA, int& info
);

template <DdlaBackend Backend, typename T>
void ppotrs(
    const DdlaHandle_t& handle,
    char side, char uplo, char trans,
    int n, int nrhs,
    const T* A, const int* descA,
    T* B, const int* descB
);

template <DdlaBackend Backend, typename T>
void pposv(
    const DdlaHandle_t& handle,
    char side, char uplo, char trans,
    int n, int nrhs,
    T* A, int ia, int ja, const int* descA,
    T* B, int ib, int jb, const int* descB,
    int& info // host pointer
);

template <DdlaBackend Backend, typename T>
void gemm(
    const DdlaHandle_t& handle,
    char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const T& beta,
    T* C, int ldc);

template <DdlaBackend Backend, typename R>
void gemm(
    const DdlaHandle_t& handle,
    char transa, char transb,
    int m, int n, int k,
    const std::complex<R>& alpha,
    const R* A, int lda,
    const std::complex<R>* B, int ldb,
    const std::complex<R>& beta,
    std::complex<R>* C, int ldc);

template <DdlaBackend Backend, typename T>
void herk(
    const DdlaHandle_t& handle,
    char uplo, char trans,
    int n, int k,
    const typename real_type<T>::type& alpha,
    const T* A, int lda,
    const typename real_type<T>::type& beta,
    T* C, int ldc);

template <DdlaBackend Backend, typename T>
void her2k(
    const DdlaHandle_t& handle,
    char uplo, char trans,
    int n, int k,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const typename real_type<T>::type& beta,
    T* C, int ldc);

template <DdlaBackend Backend, typename T>
void hemv(
    const DdlaHandle_t& handle,
    char uplo,
    int n,
    const T& alpha,
    const T* A, int lda,
    const T* x, int incx,
    const T& beta,
    T* y, int incy);

template <DdlaBackend Backend, typename T>
void hemm(
    const DdlaHandle_t& handle,
    char side, char uplo,
    int m, int n,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const T& beta,
    T* C, int ldc);

template <DdlaBackend Backend, typename R>
void hemm(const DdlaHandle_t& handle, char side, char uplo, int m, int n,
    const std::complex<R>& alpha, const R* A, int lda,
    const std::complex<R>* B, int ldb, const std::complex<R>& beta,
    std::complex<R>* C, int ldc);

template <DdlaBackend Backend, typename R>
void hemm(const DdlaHandle_t& handle, char side, char uplo, int m, int n,
    const std::complex<R>& alpha, const std::complex<R>* A, int lda,
    const R* B, int ldb, const std::complex<R>& beta,
    std::complex<R>* C, int ldc);

template <DdlaBackend Backend, typename T>
void hemmt(const DdlaHandle_t& handle,char side,char uploA,char uploC,int n,
    const T& alpha,const T* A,int lda,const T* B,int ldb,const T& beta,T* C,int ldc);

template <DdlaBackend Backend, typename R>
void hemmt(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n,
    const std::complex<R>& alpha, const R* A, int lda,
    const std::complex<R>* B, int ldb, const std::complex<R>& beta,
    std::complex<R>* C, int ldc);

template <DdlaBackend Backend, typename R>
void hemmt(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n,
    const std::complex<R>& alpha, const std::complex<R>* A, int lda,
    const R* B, int ldb, const std::complex<R>& beta,
    std::complex<R>* C, int ldc);

template <DdlaBackend Backend, typename T>
void symmt(const DdlaHandle_t& handle,char side,char uploA,char uploC,int n,
    const T& alpha,const T* A,int lda,const T* B,int ldb,const T& beta,T* C,int ldc);

template <DdlaBackend Backend, typename T>
void her2(
    const DdlaHandle_t& handle,
    char uplo,
    int n,
    const T& alpha,
    const T* x, int incx,
    const T* y, int incy,
    T* A, int lda);

template <DdlaBackend Backend, typename T>
void gemv(
    const DdlaHandle_t& handle,
    char trans, int m, int n,
    const T& alpha,
    const T* A, int lda,
    const T* x, int incx,
    const T& beta,
    T* y, int incy);

template <DdlaBackend Backend, typename T>
void trmv(
    const DdlaHandle_t& handle,
    char uplo, char trans, char diag,
    int n,
    const T* A, int lda,
    T* x, int incx);

template <DdlaBackend Backend, typename T>
void nrm2(
    const DdlaHandle_t& handle,
    int n,
    const T* x, int incx,
    typename real_type<T>::type* result);

template <DdlaBackend Backend, typename T>
void dotc(
    const DdlaHandle_t& handle,
    int n,
    const T* x, int incx,
    const T* y, int incy,
    T* result);

template <DdlaBackend Backend, typename T>
void trsm(
    const DdlaHandle_t& handle,
    char side, char uplo, char trans, char diag,
    int m, int n,
    const T& alpha,
    T* A, int lda,
    T* B, int ldb);

template <DdlaBackend Backend, typename T>
void syrk(
    const DdlaHandle_t& handle,
    char uplo, char trans,
    int n, int k,
    const T& alpha,
    const T* A, int lda,
    const T& beta,
    T* C, int ldc);

template <DdlaBackend Backend, typename T>
void trmm(
    const DdlaHandle_t& handle,
    char side, char uplo, char trans, char diag,
    int m, int n,
    const T& alpha,
    const T* A, int lda,
    T* B, int ldb);

template <DdlaBackend Backend, typename T>
void trtrm(const DdlaHandle_t& handle, char uplo_a, char transa, char uplo_b, char transb,
           int n, const T& alpha, const T* A, int lda, const T* B, int ldb,
           const T& beta, T* C, int ldc);

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
    int batchCount);

template <DdlaBackend Backend, typename R>
void gemmBatched(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k,
    const std::complex<R>& alpha, const R* const Aarray[], int lda,
    const std::complex<R>* const Barray[], int ldb, const std::complex<R>& beta,
    std::complex<R>* const Carray[], int ldc, int batchCount);

template <DdlaBackend Backend, typename R>
void gemmBatched(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k,
    const std::complex<R>& alpha, const std::complex<R>* const Aarray[], int lda,
    const R* const Barray[], int ldb, const std::complex<R>& beta,
    std::complex<R>* const Carray[], int ldc, int batchCount);

template <DdlaBackend Backend, typename T>
void geam(const DdlaHandle_t& handle, char transa, char transb, int m, int n,
          const T& alpha, const T* A, int lda,
          const T& beta, const T* B, int ldb,
          T* C, int ldc);

template <DdlaBackend Backend, typename T>
void gemmVbatched(
    const DdlaHandle_t& handle,
    char transA, char transB,
    int* d_m, int* d_n, int* d_k,
    T alpha,
    const T* const* d_A_array, int* d_lda,
    const T* const* d_B_array, int* d_ldb,
    T beta,
    T** d_C_array, int* d_ldc,
    int batch_count);

template <DdlaBackend Backend, typename T>
void gemmVbatched2s(
    const DdlaHandle_t& handle,
    char transA_0, char transB_0,
    int* d_m_0, int* d_n_0, int* d_k_0,
    T alpha_0,
    const T* const* d_A_array_0, int* d_lda_0,
    const T* const* d_B_array_0, int* d_ldb_0,
    T beta_0,
    T** d_C_array_0, int* d_ldc_0,
    char transA_1, char transB_1,
    int* d_m_1, int* d_n_1, int* d_k_1,
    T alpha_1,
    const T* const* d_AB_array_1,
    int* d_lda_1, int* d_ldb_1,
    T beta_1,
    T** d_C_array_1, int* d_ldc_1,
    bool C0_left,
    int batch_count,
    const int* segment_sizes,
    int segment_count);

template <DdlaBackend Backend, typename T>
void random_generate(const DdlaHandle_t& handle, T* data, const int64_t& lengthOfData);

template <DdlaBackend Backend, typename T>
void write_matrix(const T* A, int m, int n, const char* filename);

template <DdlaBackend Backend, typename T>
void scal(const DdlaHandle_t& handle, int n, const T& alpha, T* x, int incx);

template <DdlaBackend Backend, typename T>
void scal(const DdlaHandle_t& handle, int m, int n, const T& alpha, T* A, int lda);

template <DdlaBackend Backend, typename T>
void pscal(const DdlaHandle_t& handle, int m, int n, const T& alpha,
           T* A, const int* descA);

template <DdlaBackend Backend, typename T>
void scalBatched(const DdlaHandle_t& handle, int m, int n, const T& alpha,
                 T* const Aarray[], int lda, int batchCount);

template <DdlaBackend Backend, typename T, typename Alpha>
typename std::enable_if<
    std::is_same<T, std::complex<Alpha>>::value,
    void
>::type
scal(const DdlaHandle_t& h,int m,int n,const Alpha& alpha,T* A,int lda);

template <DdlaBackend Backend, typename T, typename Alpha>
typename std::enable_if<
    std::is_same<T, std::complex<Alpha>>::value,
    void
>::type
pscal(const DdlaHandle_t& h,int m,int n,const Alpha& alpha,T* A,const int* descA);

template <DdlaBackend Backend, typename T, typename Alpha>
typename std::enable_if<
    std::is_same<T, std::complex<Alpha>>::value,
    void
>::type
scalBatched(const DdlaHandle_t& h,int m,int n,const Alpha& alpha,T* const Aarray[],int lda,int count);

template <DdlaBackend Backend, typename T>
void axpy(const DdlaHandle_t& handle, int n, const T& alpha,
          const T* x, int incx, T* y, int incy);

template <DdlaBackend Backend, typename T>
void iamax(const DdlaHandle_t& handle, int n, const T* x, int incx, int& result);

template <DdlaBackend Backend, typename T>
void swap(const DdlaHandle_t& handle, int n, T* x, int incx, T* y, int incy);

template <DdlaBackend Backend, typename T>
void geru(const DdlaHandle_t& handle, int m, int n, const T& alpha,
          const T* x, int incx, const T* y, int incy, T* A, int lda);

template <DdlaBackend Backend, typename T>
void gerc(const DdlaHandle_t& handle, int m, int n, const T& alpha,
          const T* x, int incx, const T* y, int incy, T* A, int lda);

template <DdlaBackend Backend, typename T>
void getrf(const DdlaHandle_t& handle, int m, int n, T* A, int lda, int* ipiv, int* info);

template <DdlaBackend Backend, typename T>
void potrf(const DdlaHandle_t& handle, char uplo, int n, T* A, int lda, int* info);

template <DdlaBackend Backend, typename T>
void laswp(const DdlaHandle_t& handle, int n, T* A, int lda, int k1, int k2, const int* ipiv, int incx);

template <DdlaBackend Backend, typename T>
void geqrf(const DdlaHandle_t& handle, char uplo, int m, int n, T* A, int lda, T* tau);

template <DdlaBackend Backend, typename T>
void larft(const DdlaHandle_t& handle, char direct, char storev, int n, int k, T* V, int ldv, const T* tau, T* Tmat,
           int ldt);

template <DdlaBackend Backend, typename R>
void pgemm(const DdlaHandle_t& handle, char transa, char transb,
           int m, int n, int k, const std::complex<R>& alpha,
           const R* A, const int* descA,
           const std::complex<R>* B, const int* descB,
           const std::complex<R>& beta, std::complex<R>* C, const int* descC);

template <DdlaBackend Backend, typename R>
void pgemm(const DdlaHandle_t& handle, char transa, char transb,
           int m, int n, int k, const std::complex<R>& alpha,
           const std::complex<R>* A, const int* descA,
           const R* B, const int* descB,
           const std::complex<R>& beta, std::complex<R>* C, const int* descC);

template <DdlaBackend Backend, typename T>
void phe2hb(const DdlaHandle_t& handle, int n, T* A, const int* descA,
            int nbw, T* tmat, int& info);

template <DdlaBackend Backend, typename T>
void punmtr_he2hb(
    const DdlaHandle_t& handle,
    char trans,
    int n, int nbw,
    const T* A, const int* descA,
    const T* tmat,
    T* Z, const int* descZ,
    int& info
);

template <DdlaBackend Backend, typename T>
void pherm_complete(
    const DdlaHandle_t& handle,
    char uplo, int n,
    T* A, const int* descA
);

template <DdlaBackend Backend, typename T>
void phetrd(const DdlaHandle_t& handle, char uplo, int n, T* A, const int* descA,
    typename real_type<T>::type* d, typename real_type<T>::type* e, T* tau, int& info);

template <DdlaBackend Backend, typename T>
void punmtr(const DdlaHandle_t& handle, char side, char uplo, char trans, int m, int n,
    const T* A, const int* descA, const T* tau, T* C, const int* descC);

template <DdlaBackend Backend, typename T>
void punmtr(const DdlaHandle_t& handle, char side, char uplo, char trans,
    int m, int n, int k, const T* V, int ldv, const T* Tmat, int ldt,
    T* C, const int* descC);

template <DdlaBackend Backend, typename R>
void gemm(
    const DdlaHandle_t& handle,
    char transa, char transb,
    int m, int n, int k,
    const std::complex<R>& alpha,
    const std::complex<R>* A, int lda,
    const R* B, int ldb,
    const std::complex<R>& beta,
    std::complex<R>* C, int ldc);

template <DdlaBackend Backend, typename T>
void syevd(const DdlaHandle_t& handle, char jobz, char uplo, int n,
           T* A, int lda, T* w, int* info);

template <DdlaBackend Backend, typename T>
void heevd(const DdlaHandle_t& handle, char jobz, char uplo, int n,
           T* A, int lda, typename real_type<T>::type* w, int* info);

template <DdlaBackend Backend, typename T>
void pheevd(
    const DdlaHandle_t& handle,
    char jobz, char uplo, int n,
    T* A, const int* descA,
    typename real_type<T>::type* w,
    T* Z, const int* descZ,
    int& info
);

template <DdlaBackend Backend, typename T>
void pheevd_two_stage(const DdlaHandle_t& handle, char jobz, char uplo, int n,
    T* A, const int* descA, typename real_type<T>::type* w,
    T* Z, const int* descZ, int& info);

template <DdlaBackend Backend, typename T>
void pheevd_one_stage(const DdlaHandle_t& handle, char jobz, char uplo, int n,
    T* A, const int* descA, typename real_type<T>::type* w,
    T* Z, const int* descZ, int& info);

template <DdlaBackend Backend, typename T>
void gemmt(const DdlaHandle_t& handle, char uplo,
    char transa, char transb, int n, int k,
    const T& alpha, const T* A, int lda, const T* B, int ldb,
    const T& beta, T* C, int ldc);

template <DdlaBackend Backend, typename R>
void gemmt(const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k,
    const std::complex<R>& alpha, const R* A, int lda,
    const std::complex<R>* B, int ldb, const std::complex<R>& beta,
    std::complex<R>* C, int ldc);

template <DdlaBackend Backend, typename R>
void gemmt(const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k,
    const std::complex<R>& alpha, const std::complex<R>* A, int lda,
    const R* B, int ldb, const std::complex<R>& beta,
    std::complex<R>* C, int ldc);

template <DdlaBackend Backend, typename T>
void ptrtrm(const DdlaHandle_t& handle, char uplo_a, char transa, char uplo_b, char transb,
    int n, const T& alpha, const T* A, const int* descA, const T* B, const int* descB,
    const T& beta, T* C, const int* descC);

template <DdlaBackend Backend, typename T>
void sytrf(const DdlaHandle_t& h, char uplo, int n, T* A, int lda, int* ipiv, int& info);

template <DdlaBackend Backend, typename T>
void sytrs(const DdlaHandle_t& h, char uplo, int n, int nrhs, const T* A, int lda, const int* ipiv, T* B, int ldb, int& info);

template <DdlaBackend Backend, typename T>
void sytrf_nopiv(const DdlaHandle_t& h, char uplo, int n, T* A, int lda, int& info);

template <DdlaBackend Backend, typename T>
void sytrs_nopiv(const DdlaHandle_t& h, char uplo, int n, int nrhs, const T* A, int lda, T* B, int ldb, int& info);

template <DdlaBackend Backend, typename T>
void hetrf(const DdlaHandle_t& h, char uplo, int n, T* A, int lda, int* ipiv, int& info);

template <DdlaBackend Backend, typename T>
void hetrs(const DdlaHandle_t& h, char uplo, int n, int nrhs, const T* A, int lda, const int* ipiv, T* B, int ldb, int& info);

template <DdlaBackend Backend, typename T>
void hetrf_nopiv(const DdlaHandle_t& h, char uplo, int n, T* A, int lda, int& info);

template <DdlaBackend Backend, typename T>
void hetrs_nopiv(const DdlaHandle_t& h, char uplo, int n, int nrhs, const T* A, int lda, T* B, int ldb, int& info);

template <DdlaBackend Backend, typename T>
void psytrf(const DdlaHandle_t& h, char uplo, int n, T* A, const int* descA, int* ipiv, int& info);

template <DdlaBackend Backend, typename T>
void psytrs(const DdlaHandle_t& h, char uplo, int n, int nrhs, const T* A, const int* descA, const int* ipiv, T* B, const int* descB, int& info);

template <DdlaBackend Backend, typename T>
void psytrf_nopiv(const DdlaHandle_t& h, char uplo, int n, T* A, const int* descA, int& info);

template <DdlaBackend Backend, typename T>
void psytrs_nopiv(const DdlaHandle_t& h, char uplo, int n, int nrhs, const T* A, const int* descA, T* B, const int* descB, int& info);

template <DdlaBackend Backend, typename T>
void phetrf(const DdlaHandle_t& h, char uplo, int n, T* A, const int* descA, int* ipiv, int& info);

template <DdlaBackend Backend, typename T>
void phetrs(const DdlaHandle_t& h, char uplo, int n, int nrhs, const T* A, const int* descA, const int* ipiv, T* B, const int* descB, int& info);

template <DdlaBackend Backend, typename T>
void phetrf_nopiv(const DdlaHandle_t& h, char uplo, int n, T* A, const int* descA, int& info);

template <DdlaBackend Backend, typename T>
void phetrs_nopiv(const DdlaHandle_t& h, char uplo, int n, int nrhs, const T* A, const int* descA, T* B, const int* descB, int& info);

template <DdlaBackend Backend, typename T>
void ormqr(const DdlaHandle_t& h,char side,char trans,int m,int n,int k,
           const T* A,int lda,const T* tau,T* C,int ldc,int& info);

template <DdlaBackend Backend, typename T>
void pormqr(const DdlaHandle_t& h,char side,char trans,int m,int n,int k,
            const T* A,const int* descA,const T* tau,T* C,const int* descC,int& info);

template <DdlaBackend Backend, typename T>
void orgqr(const DdlaHandle_t& h,int m,int n,int k,T* A,int lda,const T* tau,int& info);

template <DdlaBackend Backend, typename T>
void porgqr(const DdlaHandle_t& h,int m,int n,int k,T* A,const int* descA,const T* tau,int& info);

} // namespace ddla

#endif // DDLA_BACKEND_API_H
