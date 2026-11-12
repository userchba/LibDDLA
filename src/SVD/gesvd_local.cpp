#include "gesvd_local.h"
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "host_threads.h"
#include "instantiate.h"
#include "plan_util.h"
#include <algorithm>
#include <complex>
#include <type_traits>
extern "C" {
void sgesvd_(const char*,const char*,const int*,const int*,float*,const int*,float*,float*,const int*,float*,const int*,float*,const int*,int*);
void dgesvd_(const char*,const char*,const int*,const int*,double*,const int*,double*,double*,const int*,double*,const int*,double*,const int*,int*);
void cgesvd_(const char*,const char*,const int*,const int*,std::complex<float>*,const int*,float*,std::complex<float>*,const int*,std::complex<float>*,const int*,std::complex<float>*,const int*,float*,int*);
void zgesvd_(const char*,const char*,const int*,const int*,std::complex<double>*,const int*,double*,std::complex<double>*,const int*,std::complex<double>*,const int*,std::complex<double>*,const int*,double*,int*);
}
namespace ddla { namespace spectral_impl {
namespace {
template<class T>
void cpu_gesvd(char ju,char jv,int m,int n,T* a,int lda,typename real_type<T>::type* s,
    T* u,int ldu,T* vt,int ldvt,T* work,int count,typename real_type<T>::type* real_work,int* info)
{
    if constexpr(std::is_same_v<T,float>)
        sgesvd_(&ju,&jv,&m,&n,a,&lda,s,u,&ldu,vt,&ldvt,work,&count,info);
    else if constexpr(std::is_same_v<T,double>)
        dgesvd_(&ju,&jv,&m,&n,a,&lda,s,u,&ldu,vt,&ldvt,work,&count,info);
    else if constexpr(std::is_same_v<T,std::complex<float>>)
        cgesvd_(&ju,&jv,&m,&n,a,&lda,s,u,&ldu,vt,&ldvt,work,&count,real_work,info);
    else
        zgesvd_(&ju,&jv,&m,&n,a,&lda,s,u,&ldu,vt,&ldvt,work,&count,real_work,info);
}
#if DDLA_HAS_GPU
#define GPU_GESVD(T,R,V,NAME) \
gpusolverStatus_t gpu_gesvd_size(gpusolverHandle_t h,char ju,char jv,int m,int n,T*,int* count) \
{ return NAME##_bufferSize(h,ju,jv,m,n,count); } \
gpusolverStatus_t gpu_gesvd_call(gpusolverHandle_t h,char ju,char jv,int m,int n,T* a,int lda,R* s,T* u,int ldu,T* vt,int ldvt,T* work,int count,R* real_work,int* info) \
{ return NAME(h,ju,jv,m,n,reinterpret_cast<V*>(a),lda,s,reinterpret_cast<V*>(u),ldu,reinterpret_cast<V*>(vt),ldvt,reinterpret_cast<V*>(work),count,real_work,info); }
GPU_GESVD(float,float,float,gpusolverSgesvd)
GPU_GESVD(double,double,double,gpusolverDgesvd)
GPU_GESVD(std::complex<float>,float,gpusolverComplex,gpusolverCgesvd)
GPU_GESVD(std::complex<double>,double,gpusolverDoubleComplex,gpusolverZgesvd)
#undef GPU_GESVD
#endif
}
template<class T>
void local_gesvd_workspace_size(const DdlaHandle_t& h,char ju,char jv,int m,int n,
    std::size_t* backend,std::size_t* host)
{
    using R=typename real_type<T>::type;
    *backend=*host=0;
    const int k=std::min(m,n);
    if(!k)return;
    ju=ju=='V'?'S':'N';jv=jv=='V'?'S':'N';
    if(h->backend==DdlaBackend::CPU) {
        T dummy{},query{};R real_dummy{};int info=0;
        cpu_gesvd(ju,jv,m,n,&dummy,std::max(1,m),&real_dummy,&dummy,std::max(1,m),
            &dummy,std::max(1,k),&query,-1,&real_dummy,&info);
        if(info)throw std::runtime_error("pgesvd: LAPACK workspace query failed");
        *backend=std::size_t(std::real(query))*sizeof(T);
        if constexpr(!std::is_same_v<T,R>)*host=detail::padded_bytes<R>(5*std::size_t(k));
    } else {
#if DDLA_HAS_GPU
        int count=0;
        SOLVER_CHECK(gpu_gesvd_size(static_cast<gpusolverHandle_t>(h->solverH),ju,jv,m,n,
            static_cast<T*>(nullptr),&count));
        *backend=detail::padded_bytes<R>(k)+detail::padded_bytes<R>(std::max(1,k-1))+
            detail::padded_bytes<int>(1)+std::size_t(count)*sizeof(T);
#endif
    }
}

template<DdlaBackend B,class T>
void local_gesvd_run(const DdlaHandle_t& h,char ju,char jv,int m,int n,T* a,int lda,
    typename real_type<T>::type* s,T* u,int ldu,T* vt,int ldvt,int& info,
    void* backend,std::size_t bytes,void* host,std::size_t)
{
    using R=typename real_type<T>::type;
    info=0;const int k=std::min(m,n);if(!k)return;
    ju=ju=='V'?'S':'N';jv=jv=='V'?'S':'N';
    if(!u)u=a;
    if(!vt)vt=a;
    if constexpr(B==DdlaBackend::CPU) {
        detail::ScopedBlasThreads threads(h->host_threads);
        cpu_gesvd(ju,jv,m,n,a,lda,s,u,ldu,vt,ldvt,static_cast<T*>(backend),int(bytes/sizeof(T)),static_cast<R*>(host),&info);
    } else {
#if DDLA_HAS_GPU
        char* work=static_cast<char*>(backend);
        R* values=detail::take<R>(work,k);
        R* real_work=detail::take<R>(work,std::max(1,k-1));
        int* status=detail::take<int>(work,1);
        const std::size_t used=detail::padded_bytes<R>(k)+detail::padded_bytes<R>(std::max(1,k-1))+detail::padded_bytes<int>(1);
        SOLVER_CHECK(gpu_gesvd_call(static_cast<gpusolverHandle_t>(h->solverH),ju,jv,m,n,a,lda,
            values,u,ldu,vt,ldvt,reinterpret_cast<T*>(work),int((bytes-used)/sizeof(T)),real_work,status));
        DDLA_CHECK(ddlaMemcpy(h,&info,status,sizeof(int)));
        DDLA_CHECK(ddlaMemcpy(h,s,values,std::size_t(k)*sizeof(R)));
        DDLA_CHECK(ddlaSynchronize(h));
#endif
    }
}
#define SIZE(T) template void local_gesvd_workspace_size<T>(const DdlaHandle_t&,char,char,int,int,std::size_t*,std::size_t*)
#define RUN(B,T) template void local_gesvd_run<B,T>(const DdlaHandle_t&,char,char,int,int,T*,int,typename real_type<T>::type*,T*,int,T*,int,int&,void*,std::size_t,void*,std::size_t)
DDLA_FOR_EACH_TYPE(SIZE)
DDLA_FOR_EACH_BACKEND_TYPE(RUN)
#undef RUN
#undef SIZE
} }
