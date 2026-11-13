#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "helpers.h"
#include "host_threads.h"
#include "instantiate.h"
#include "require_backend.h"
#include <algorithm>
#include <cctype>
#include <complex>
#include <stdexcept>
#include <type_traits>
#include <vector>

extern "C" {
void ssygvd_(const int*,const char*,const char*,const int*,float*,const int*,float*,const int*,float*,float*,const int*,int*,const int*,int*);
void dsygvd_(const int*,const char*,const char*,const int*,double*,const int*,double*,const int*,double*,double*,const int*,int*,const int*,int*);
void chegvd_(const int*,const char*,const char*,const int*,std::complex<float>*,const int*,std::complex<float>*,const int*,float*,std::complex<float>*,const int*,float*,const int*,int*,const int*,int*);
void zhegvd_(const int*,const char*,const char*,const int*,std::complex<double>*,const int*,std::complex<double>*,const int*,double*,std::complex<double>*,const int*,double*,const int*,int*,const int*,int*);
}

namespace ddla {
namespace {
template<class T>
void cpu_hegvd(char job,char uplo,int n,T* a,int lda,T* b,int ldb,
    typename real_type<T>::type* w,T* work,int lwork,
    typename real_type<T>::type* rwork,int lrwork,int* iwork,int liwork,int* info)
{
    const int type=1;
    if constexpr(std::is_same_v<T,float>)
        ssygvd_(&type,&job,&uplo,&n,a,&lda,b,&ldb,w,work,&lwork,iwork,&liwork,info);
    else if constexpr(std::is_same_v<T,double>)
        dsygvd_(&type,&job,&uplo,&n,a,&lda,b,&ldb,w,work,&lwork,iwork,&liwork,info);
    else if constexpr(std::is_same_v<T,std::complex<float>>)
        chegvd_(&type,&job,&uplo,&n,a,&lda,b,&ldb,w,work,&lwork,rwork,&lrwork,iwork,&liwork,info);
    else
        zhegvd_(&type,&job,&uplo,&n,a,&lda,b,&ldb,w,work,&lwork,rwork,&lrwork,iwork,&liwork,info);
}

#if DDLA_HAS_GPU
#define GPU_HEGVD(T,R,V,NAME) \
gpusolverStatus_t gpu_size(gpusolverHandle_t h,gpusolverEigMode_t job,gpusolverFillMode_t uplo, \
    int n,T* a,int lda,T* b,int ldb,R* w,int* count) \
{ return NAME##_bufferSize(h,GPUSOLVER_EIG_TYPE_1,job,uplo,n,reinterpret_cast<V*>(a),lda, \
    reinterpret_cast<V*>(b),ldb,w,count); } \
gpusolverStatus_t gpu_call(gpusolverHandle_t h,gpusolverEigMode_t job,gpusolverFillMode_t uplo, \
    int n,T* a,int lda,T* b,int ldb,R* w,T* work,int count,int* info) \
{ return NAME(h,GPUSOLVER_EIG_TYPE_1,job,uplo,n,reinterpret_cast<V*>(a),lda, \
    reinterpret_cast<V*>(b),ldb,w,reinterpret_cast<V*>(work),count,info); }
GPU_HEGVD(float,float,float,gpusolverSsygvd)
GPU_HEGVD(double,double,double,gpusolverDsygvd)
GPU_HEGVD(std::complex<float>,float,gpusolverComplex,gpusolverChegvd)
GPU_HEGVD(std::complex<double>,double,gpusolverDoubleComplex,gpusolverZhegvd)
#undef GPU_HEGVD
#endif

template<DdlaBackend B,class T>
void local_hegvd(const DdlaHandle_t& h,char job,char uplo,int n,T* a,int lda,T* b,int ldb,
                 typename real_type<T>::type* w,int* info)
{
    using R=typename real_type<T>::type;
    detail::require_backend<B>(h,"hegvd/sygvd");
    job=static_cast<char>(std::toupper(static_cast<unsigned char>(job)));
    uplo=static_cast<char>(std::toupper(static_cast<unsigned char>(uplo)));
    if(n<0 || (job!='N' && job!='V') || (uplo!='U' && uplo!='L') || !info ||
       lda<std::max(1,n) || ldb<std::max(1,n) || (n && (!a || !b || !w)))
        throw std::invalid_argument("hegvd/sygvd: invalid order/options/pointers/leading dimensions");
    if(detail::matrix_overlaps(a,lda,n,n,b,ldb,n,n))
        throw std::invalid_argument("hegvd/sygvd: A and B must not overlap");
    if(!n) {
        if constexpr(B==DdlaBackend::CPU) *info=0;
        else DDLA_CHECK(ddlaMemset(h,info,0,sizeof(int)));
        return;
    }
    if constexpr(B==DdlaBackend::CPU) {
        detail::ScopedBlasThreads threads(h->host_threads);
        T query{};R real_query{};int int_query=0,status=0;
        cpu_hegvd(job,uplo,n,a,lda,b,ldb,w,&query,-1,&real_query,-1,&int_query,-1,&status);
        if(status)throw std::runtime_error("hegvd/sygvd: LAPACK workspace query failed");
        const int lwork=static_cast<int>(std::real(query));
        int lrwork=0;
        if constexpr(!std::is_same_v<T,R>) lrwork=static_cast<int>(real_query);
        std::vector<T> work(lwork);
        std::vector<R> rwork(lrwork);
        std::vector<int> iwork(int_query);
        cpu_hegvd(job,uplo,n,a,lda,b,ldb,w,work.data(),lwork,rwork.data(),lrwork,
                  iwork.data(),int_query,info);
    } else {
#if DDLA_HAS_GPU
        const auto solver=static_cast<gpusolverHandle_t>(h->solverH);
        const auto mode=job=='V'?GPUSOLVER_EIG_MODE_VECTOR:GPUSOLVER_EIG_MODE_NOVECTOR;
        const auto triangle=uplo=='U'?GPUSOLVER_FILL_MODE_UPPER:GPUSOLVER_FILL_MODE_LOWER;
        int count=0;
        SOLVER_CHECK(gpu_size(solver,mode,triangle,n,a,lda,b,ldb,w,&count));
        ddla::vector<T> work(h,count);
        const std::size_t apad=std::size_t(lda-n)*n,bpad=std::size_t(ldb-n)*n;
        ddla::vector<T> padding(h,apad+bpad);
        auto copy_padding=[&](bool restore) {
            auto copy=[&](T* matrix,int ld,T* saved) {
                const std::size_t width=std::size_t(ld-n)*sizeof(T);
                if(!width)return;
                if(restore)DDLA_CHECK(ddlaMemcpy2D(h,matrix+n,std::size_t(ld)*sizeof(T),saved,width,width,n));
                else DDLA_CHECK(ddlaMemcpy2D(h,saved,width,matrix+n,std::size_t(ld)*sizeof(T),width,n));
            };
            if(apad)copy(a,lda,padding.data());
            if(bpad)copy(b,ldb,padding.data()+apad);
        };
        copy_padding(false);
        try {
#if defined(DDLA_USE_CUDA)
            // CUDA 11.8 SYGST/HEGST reads caller workspace before writing it
            // (initcheck: batch_symmetrize_kernel, including real float).
            DDLA_CHECK(ddlaMemset(h,work.data(),0,std::size_t(count)*sizeof(T)));
#endif
            SOLVER_CHECK(gpu_call(solver,mode,triangle,n,a,lda,b,ldb,w,work.data(),count,info));
            copy_padding(true);
            DDLA_CHECK(ddlaSynchronize(h));
        } catch(...) {
            copy_padding(true);
            (void)ddlaSynchronize(h);
            throw;
        }
#endif
    }
}
} // namespace

template<DdlaBackend B,class T>
void hegvd(const DdlaHandle_t& h,char job,char uplo,int n,T* a,int lda,T* b,int ldb,
           typename real_type<T>::type* w,int* info)
{
    static_assert(!std::is_same_v<T,typename real_type<T>::type>,"hegvd requires complex scalars");
    local_hegvd<B,T>(h,job,uplo,n,a,lda,b,ldb,w,info);
}
template<DdlaBackend B,class T>
void sygvd(const DdlaHandle_t& h,char job,char uplo,int n,T* a,int lda,T* b,int ldb,T* w,int* info)
{
    static_assert(std::is_same_v<T,float> || std::is_same_v<T,double>,"sygvd requires real scalars");
    local_hegvd<B,T>(h,job,uplo,n,a,lda,b,ldb,w,info);
}
#define LOCAL(B) \
template void hegvd<B,std::complex<float>>(const DdlaHandle_t&,char,char,int,std::complex<float>*,int,std::complex<float>*,int,float*,int*); \
template void hegvd<B,std::complex<double>>(const DdlaHandle_t&,char,char,int,std::complex<double>*,int,std::complex<double>*,int,double*,int*); \
template void sygvd<B,float>(const DdlaHandle_t&,char,char,int,float*,int,float*,int,float*,int*); \
template void sygvd<B,double>(const DdlaHandle_t&,char,char,int,double*,int,double*,int,double*,int*)
DDLA_FOR_EACH_BACKEND(LOCAL)
#undef LOCAL
} // namespace ddla
