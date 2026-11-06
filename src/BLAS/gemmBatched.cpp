#include "blas_impl.h"
#ifdef _OPENMP
#include <omp.h>
#endif
// ---------------------------------------------------------------------------
// Unified backend-templated batched matrix multiply over arrays of pointers:
// for i in [0, batchCount):  C[i] := alpha*blas_impl::op(A[i])*blas_impl::op(B[i]) + beta*C[i].
//
// Dispatch follows the Backend template parameter. CPU code is always compiled;
// DDLA_HAS_GPU enables GPU code, and DDLA_USE_CUDA / DDLA_USE_HIP select its vendor.
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
#include "plan_util.h"
#include "gemm_mixed_cpu.h"
#include "helpers.h"
#include "require_backend.h"
#include <cassert>

extern "C" {
// Fortran BLAS entry points for the CPU arm, available in every build.
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

template <DdlaBackend Backend, typename T, typename TA, typename TB>
void gemmBatched_check(
    const DdlaHandle_t& handle,
    char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const TA* const Aarray[], int lda,
    const TB* const Barray[], int ldb,
    const T& beta,
    T* const Carray[], int ldc,
    int batchCount)
{
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

    if constexpr (!std::is_same_v<T,TA> || !std::is_same_v<T,TB>) {
        const char ta=upper(transa),tb=upper(transb);
        if ((ta!='N' && ta!='T' && ta!='C') || (tb!='N' && tb!='T' && tb!='C') ||
            m<0 || n<0 || k<0 || batchCount<0 || lda<std::max(1,ta=='N'?m:k) ||
            ldb<std::max(1,tb=='N'?k:n) || ldc<std::max(1,m))
            throw std::invalid_argument("gemmBatched(mixed): invalid flags, dimensions or pitches");
        const bool product=k>0 && alpha!=T(0);
        if (m && n && batchCount && (product || beta!=T(1))) {
            if (!Carray || (product && (!Aarray || !Barray)))
                throw std::invalid_argument("gemmBatched(mixed): null pointer array");
            if constexpr (Backend==DdlaBackend::CPU) for (int i=0;i<batchCount;++i) {
                if (!Carray[i] || (product && (!Aarray[i] || !Barray[i])))
                    throw std::invalid_argument("gemmBatched(mixed): null matrix");
                if (product && (detail::matrix_overlaps(Aarray[i],lda,ta=='N'?m:k,ta=='N'?k:m,Carray[i],ldc,m,n) ||
                    detail::matrix_overlaps(Barray[i],ldb,tb=='N'?k:n,tb=='N'?n:k,Carray[i],ldc,m,n)))
                    throw std::invalid_argument("gemmBatched(mixed): output overlaps input");
            }
        }
    }

    if constexpr (Backend == DdlaBackend::GPU) {
        (void)detail::to_blas_op(transa);
        (void)detail::to_blas_op(transb);
    }
}

template <DdlaBackend Backend, typename T, typename TA, typename TB>
void gemmBatched_run(
    const DdlaHandle_t& handle,
    char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const TA* const Aarray[], int lda,
    const TB* const Barray[], int ldb,
    const T& beta,
    T* const Carray[], int ldc,
    int batchCount, void* work, std::size_t bytes)
{
    if constexpr (!std::is_same_v<T,TA> || !std::is_same_v<T,TB>) {
        using R=typename real_type<T>::type;
        constexpr bool complex_left=std::is_same_v<T,TA>;
        transa=upper(transa);transb=upper(transb);
        if constexpr (std::is_arithmetic_v<TA>) if (transa=='C') transa='T';
        if constexpr (std::is_arithmetic_v<TB>) if (transb=='C') transb='T';
        if (m==0 || n==0 || batchCount==0) return;
        const bool product=k>0 && alpha!=T(0);
        if (!product && beta==T(1)) return;
        if constexpr (Backend==DdlaBackend::GPU) {
#if DDLA_HAS_GPU
            mixed_gemmBatched_gpu_run<R,complex_left>(handle,transa,transb,m,n,k,alpha,
                reinterpret_cast<const void* const*>(Aarray),lda,reinterpret_cast<const void* const*>(Barray),ldb,
                beta,Carray,ldc,batchCount,work,bytes);
#endif
        } else {
            if (!product) {
                for_each_product(handle->host_threads,batchCount,[&](int q) {
                    for (int j=0;j<n;++j) for (int i=0;i<m;++i) {
                        T& out=Carray[q][i+std::size_t(j)*ldc];
                        out=beta==T(0)?T(0):beta*out;
                    }
                });
                return;
            }
            int depth=k;
            std::size_t parallel=0;
            if(m>=1024 && n>=1024 && batchCount<=std::max(1,handle->host_threads/2))
                detail::mixed_gemm_cpu_workspace_size<R>(handle,m,n,k,complex_left,&parallel);
            const int panel=std::min(n,512),cap=mixed_gemmBatched_capacity<R>(handle->host_threads,m,n,k,complex_left,batchCount,depth,parallel);
            if(depth==0) {
                for(int q=0;q<batchCount;++q)
                    gemm_run<Backend,R>(handle,transa,transb,m,n,k,alpha,Aarray[q],lda,Barray[q],ldb,beta,Carray[q],ldc,work,bytes);
                return;
            }
            const std::size_t packed=2*(complex_left ? std::size_t(m)*depth : std::size_t(depth)*panel);
            const std::size_t result=2*std::size_t(m)*panel,stride=packed+result;
            char* cursor=static_cast<char*>(work);
            R* data=detail::take<R>(cursor,stride*cap);
            const R** ap=detail::take<const R*>(cursor,cap);
            const R** bp=detail::take<const R*>(cursor,cap);
            R** cp=detail::take<R*>(cursor,cap);
            assert(std::size_t(cursor-static_cast<char*>(work))<=bytes);
            // Reuse one slot per worker across the entire batch. Chunking the
            // product list would repeatedly wake OpenMP/BLAS teams and retain
            // scratch for queued products that cannot execute concurrently.
            for_each_product(cap,batchCount,[&](int product) {
                int q=0;
#ifdef _OPENMP
                q=cap>1?omp_get_thread_num():0;
#endif
                R* p=data+q*stride;
                if constexpr (complex_left) if(depth==k && transa!='N')
                    detail::mixed_pack_left<R>(transa,m,k,Aarray[product],lda,p);
                for (int j0=0;j0<n;) {
                    const int nj=std::min(panel,n-j0);
                    cp[q]=p+packed;
                    if constexpr(complex_left) if(transa=='N') {
                        // The complex row pairs can go straight to real BLAS.
                        gemm_run<Backend,R>(handle,'N',transb,2*m,nj,k,R(1),
                            reinterpret_cast<const R*>(Aarray[product]),2*lda,
                            Barray[product]+(transb=='N'?std::size_t(j0)*ldb:std::size_t(j0)),ldb,
                            R(0),cp[q],2*m);
                        T* output=Carray[product]+std::size_t(j0)*ldc;
                        geam_run<Backend,T>(handle,'N','N',m,nj,alpha,
                            reinterpret_cast<const T*>(cp[q]),m,beta,output,ldc,output,ldc);
                        j0+=nj;continue;
                    }
                    for(int l0=0;l0<k;) {
                        const int kk=std::min(depth,k-l0);
                        if constexpr (complex_left) {
                            if(depth<k)detail::mixed_pack_left<R>(transa,m,kk,
                                Aarray[product]+(transa=='N'?std::size_t(l0)*lda:std::size_t(l0)),lda,p);
                            ap[q]=p;
                            bp[q]=Barray[product]+(transb=='N'?std::size_t(l0)+std::size_t(j0)*ldb
                                                                          :std::size_t(j0)+std::size_t(l0)*ldb);
                        } else {
                            const TB* input=Barray[product]+(transb=='N'?std::size_t(l0)+std::size_t(j0)*ldb
                                                                                     :std::size_t(j0)+std::size_t(l0)*ldb);
                            detail::mixed_pack_right<R>(transb,kk,nj,input,ldb,p);
                            ap[q]=Aarray[product]+(transa=='N'?std::size_t(l0)*lda:std::size_t(l0));bp[q]=p;
                        }
                        gemm_run<Backend,R>(handle,complex_left?'N':transa,complex_left?transb:'N',
                            complex_left?2*m:m,complex_left?nj:2*nj,kk,R(1),ap[q],complex_left?2*m:lda,
                            bp[q],complex_left?ldb:kk,l0==0?R(0):R(1),cp[q],complex_left?2*m:m);
                        l0+=kk;
                    }
                    detail::mixed_combine<R,complex_left>(m,nj,cp[q],alpha,beta,
                        Carray[product]+std::size_t(j0)*ldc,ldc);
                    j0+=nj;
                }
            });
        }
    } else
    if constexpr (Backend == DdlaBackend::CPU) {
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

INSTANTIATE_GEMM_BATCHED(DdlaBackend::CPU, float);
INSTANTIATE_GEMM_BATCHED(DdlaBackend::CPU, double);
INSTANTIATE_GEMM_BATCHED(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_GEMM_BATCHED(DdlaBackend::CPU, std::complex<double>);

#if DDLA_HAS_GPU
INSTANTIATE_GEMM_BATCHED(DdlaBackend::GPU, float);
INSTANTIATE_GEMM_BATCHED(DdlaBackend::GPU, double);
INSTANTIATE_GEMM_BATCHED(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_GEMM_BATCHED(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_GEMM_BATCHED


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::gemmBatched_check<BACKEND, TYPE>( const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const TYPE& alpha, const TYPE* const Aarray[], int lda, const TYPE* const Barray[], int ldb, const TYPE& beta, TYPE* const Carray[], int ldc, int batchCount); \
    template void blas_impl::gemmBatched_run<BACKEND, TYPE>( const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k, const TYPE& alpha, const TYPE* const Aarray[], int lda, const TYPE* const Barray[], int ldb, const TYPE& beta, TYPE* const Carray[], int ldc, int batchCount, void*, std::size_t);
DDLA_INSTANTIATE_PHASES(DdlaBackend::CPU, float)
DDLA_INSTANTIATE_PHASES(DdlaBackend::CPU, double)
DDLA_INSTANTIATE_PHASES(DdlaBackend::CPU, std::complex<float>)
DDLA_INSTANTIATE_PHASES(DdlaBackend::CPU, std::complex<double>)
#if DDLA_HAS_GPU
DDLA_INSTANTIATE_PHASES(DdlaBackend::GPU, float)
DDLA_INSTANTIATE_PHASES(DdlaBackend::GPU, double)
DDLA_INSTANTIATE_PHASES(DdlaBackend::GPU, std::complex<float>)
DDLA_INSTANTIATE_PHASES(DdlaBackend::GPU, std::complex<double>)
#endif
#undef DDLA_INSTANTIATE_PHASES
template void blas_impl::gemmBatched_check<DdlaBackend::CPU,std::complex<float>,float,std::complex<float>>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<float>&,const float* const*,int,const std::complex<float>* const*,int,const std::complex<float>&,std::complex<float>* const*,int,int);
template void blas_impl::gemmBatched_run<DdlaBackend::CPU,std::complex<float>,float,std::complex<float>>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<float>&,const float* const*,int,const std::complex<float>* const*,int,const std::complex<float>&,std::complex<float>* const*,int,int,void*,std::size_t);
template void blas_impl::gemmBatched_check<DdlaBackend::CPU,std::complex<float>,std::complex<float>,float>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<float>&,const std::complex<float>* const*,int,const float* const*,int,const std::complex<float>&,std::complex<float>* const*,int,int);
template void blas_impl::gemmBatched_run<DdlaBackend::CPU,std::complex<float>,std::complex<float>,float>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<float>&,const std::complex<float>* const*,int,const float* const*,int,const std::complex<float>&,std::complex<float>* const*,int,int,void*,std::size_t);
template void blas_impl::gemmBatched_check<DdlaBackend::CPU,std::complex<double>,double,std::complex<double>>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<double>&,const double* const*,int,const std::complex<double>* const*,int,const std::complex<double>&,std::complex<double>* const*,int,int);
template void blas_impl::gemmBatched_run<DdlaBackend::CPU,std::complex<double>,double,std::complex<double>>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<double>&,const double* const*,int,const std::complex<double>* const*,int,const std::complex<double>&,std::complex<double>* const*,int,int,void*,std::size_t);
template void blas_impl::gemmBatched_check<DdlaBackend::CPU,std::complex<double>,std::complex<double>,double>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<double>&,const std::complex<double>* const*,int,const double* const*,int,const std::complex<double>&,std::complex<double>* const*,int,int);
template void blas_impl::gemmBatched_run<DdlaBackend::CPU,std::complex<double>,std::complex<double>,double>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<double>&,const std::complex<double>* const*,int,const double* const*,int,const std::complex<double>&,std::complex<double>* const*,int,int,void*,std::size_t);
#if DDLA_HAS_GPU
template void blas_impl::gemmBatched_check<DdlaBackend::GPU,std::complex<float>,float,std::complex<float>>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<float>&,const float* const*,int,const std::complex<float>* const*,int,const std::complex<float>&,std::complex<float>* const*,int,int);
template void blas_impl::gemmBatched_run<DdlaBackend::GPU,std::complex<float>,float,std::complex<float>>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<float>&,const float* const*,int,const std::complex<float>* const*,int,const std::complex<float>&,std::complex<float>* const*,int,int,void*,std::size_t);
template void blas_impl::gemmBatched_check<DdlaBackend::GPU,std::complex<float>,std::complex<float>,float>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<float>&,const std::complex<float>* const*,int,const float* const*,int,const std::complex<float>&,std::complex<float>* const*,int,int);
template void blas_impl::gemmBatched_run<DdlaBackend::GPU,std::complex<float>,std::complex<float>,float>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<float>&,const std::complex<float>* const*,int,const float* const*,int,const std::complex<float>&,std::complex<float>* const*,int,int,void*,std::size_t);
template void blas_impl::gemmBatched_check<DdlaBackend::GPU,std::complex<double>,double,std::complex<double>>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<double>&,const double* const*,int,const std::complex<double>* const*,int,const std::complex<double>&,std::complex<double>* const*,int,int);
template void blas_impl::gemmBatched_run<DdlaBackend::GPU,std::complex<double>,double,std::complex<double>>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<double>&,const double* const*,int,const std::complex<double>* const*,int,const std::complex<double>&,std::complex<double>* const*,int,int,void*,std::size_t);
template void blas_impl::gemmBatched_check<DdlaBackend::GPU,std::complex<double>,std::complex<double>,double>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<double>&,const std::complex<double>* const*,int,const double* const*,int,const std::complex<double>&,std::complex<double>* const*,int,int);
template void blas_impl::gemmBatched_run<DdlaBackend::GPU,std::complex<double>,std::complex<double>,double>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<double>&,const std::complex<double>* const*,int,const double* const*,int,const std::complex<double>&,std::complex<double>* const*,int,int,void*,std::size_t);
#endif
} // namespace ddla
