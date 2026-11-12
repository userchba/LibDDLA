#include "gebrd_local.h"
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "host_threads.h"
#include "instantiate.h"
#include "plan_util.h"
#include <algorithm>
#include <complex>
#include <type_traits>
extern "C" {
void sgebrd_(const int*,const int*,float*,const int*,float*,float*,float*,float*,float*,const int*,int*);
void dgebrd_(const int*,const int*,double*,const int*,double*,double*,double*,double*,double*,const int*,int*);
void cgebrd_(const int*,const int*,std::complex<float>*,const int*,float*,float*,std::complex<float>*,std::complex<float>*,std::complex<float>*,const int*,int*);
void zgebrd_(const int*,const int*,std::complex<double>*,const int*,double*,double*,std::complex<double>*,std::complex<double>*,std::complex<double>*,const int*,int*);
}
namespace ddla { namespace spectral_impl {
namespace {
template<class T>
void cpu_gebrd(int m,int n,T* a,int lda,typename real_type<T>::type* d,
    typename real_type<T>::type* e,T* tq,T* tp,T* work,int count,int* info)
{
    if constexpr(std::is_same_v<T,float>)sgebrd_(&m,&n,a,&lda,d,e,tq,tp,work,&count,info);
    else if constexpr(std::is_same_v<T,double>)dgebrd_(&m,&n,a,&lda,d,e,tq,tp,work,&count,info);
    else if constexpr(std::is_same_v<T,std::complex<float>>)cgebrd_(&m,&n,a,&lda,d,e,tq,tp,work,&count,info);
    else zgebrd_(&m,&n,a,&lda,d,e,tq,tp,work,&count,info);
}
#if DDLA_HAS_GPU
#define GPU_GEBRD(T,R,V,NAME) \
gpusolverStatus_t gpu_gebrd_size(gpusolverHandle_t h,int m,int n,T*,int* count) \
{return NAME##_bufferSize(h,m,n,count);} \
gpusolverStatus_t gpu_gebrd_call(gpusolverHandle_t h,int m,int n,T* a,int lda,R* d,R* e,T* tq,T* tp,T* work,int count,int* info) \
{return NAME(h,m,n,reinterpret_cast<V*>(a),lda,d,e,reinterpret_cast<V*>(tq),reinterpret_cast<V*>(tp),reinterpret_cast<V*>(work),count,info);}
GPU_GEBRD(float,float,float,gpusolverSgebrd)
GPU_GEBRD(double,double,double,gpusolverDgebrd)
GPU_GEBRD(std::complex<float>,float,gpusolverComplex,gpusolverCgebrd)
GPU_GEBRD(std::complex<double>,double,gpusolverDoubleComplex,gpusolverZgebrd)
#undef GPU_GEBRD
#endif
}
template<class T>
void local_gebrd_workspace_size(const DdlaHandle_t& h,int m,int n,std::size_t* bytes)
{
    using R=typename real_type<T>::type;
    *bytes=0;const int k=std::min(m,n);if(!k)return;
    if(h->backend==DdlaBackend::CPU) {
        T dummy{},query{};R real_dummy{};int info=0;
        cpu_gebrd(m,n,&dummy,std::max(1,m),&real_dummy,&real_dummy,&dummy,&dummy,&query,-1,&info);
        if(info)throw std::runtime_error("pgebrd: LAPACK workspace query failed");
        *bytes=std::size_t(std::real(query))*sizeof(T);
    } else {
#if DDLA_HAS_GPU
        int count=0;SOLVER_CHECK(gpu_gebrd_size(static_cast<gpusolverHandle_t>(h->solverH),m,n,static_cast<T*>(nullptr),&count));
        *bytes=2*detail::padded_bytes<R>(k)+2*detail::padded_bytes<T>(k)+detail::padded_bytes<int>(1)+std::size_t(count)*sizeof(T);
#endif
    }
}
template<DdlaBackend B,class T>
void local_gebrd_run(const DdlaHandle_t& h,int m,int n,T* a,int lda,
    typename real_type<T>::type* d,typename real_type<T>::type* e,T* tq,T* tp,
    int& info,void* backend,std::size_t bytes)
{
    using R=typename real_type<T>::type;
    info=0;const int k=std::min(m,n);if(!k)return;
    if constexpr(B==DdlaBackend::CPU) {
        detail::ScopedBlasThreads threads(h->host_threads);
        cpu_gebrd(m,n,a,lda,d,e,tq,tp,static_cast<T*>(backend),int(bytes/sizeof(T)),&info);
    } else {
#if DDLA_HAS_GPU
        char* work=static_cast<char*>(backend);
        R* diagonal=detail::take<R>(work,k);R* off=detail::take<R>(work,k);
        T* tauq=detail::take<T>(work,k);T* taup=detail::take<T>(work,k);
        int* status=detail::take<int>(work,1);
        const auto used=2*detail::padded_bytes<R>(k)+2*detail::padded_bytes<T>(k)+detail::padded_bytes<int>(1);
        SOLVER_CHECK(gpu_gebrd_call(static_cast<gpusolverHandle_t>(h->solverH),m,n,a,lda,diagonal,off,tauq,taup,reinterpret_cast<T*>(work),int((bytes-used)/sizeof(T)),status));
        DDLA_CHECK(ddlaMemcpy(h,&info,status,sizeof(int)));
        DDLA_CHECK(ddlaMemcpy(h,d,diagonal,std::size_t(k)*sizeof(R)));
        if(k>1)DDLA_CHECK(ddlaMemcpy(h,e,off,std::size_t(k-1)*sizeof(R)));
        DDLA_CHECK(ddlaMemcpy(h,tq,tauq,std::size_t(k)*sizeof(T)));
        DDLA_CHECK(ddlaMemcpy(h,tp,taup,std::size_t(k)*sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
#endif
    }
}
#define SIZE(T) template void local_gebrd_workspace_size<T>(const DdlaHandle_t&,int,int,std::size_t*)
#define RUN(B,T) template void local_gebrd_run<B,T>(const DdlaHandle_t&,int,int,T*,int,typename real_type<T>::type*,typename real_type<T>::type*,T*,T*,int&,void*,std::size_t)
DDLA_FOR_EACH_TYPE(SIZE)
DDLA_FOR_EACH_BACKEND_TYPE(RUN)
#undef RUN
#undef SIZE
} }
