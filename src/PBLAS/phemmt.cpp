#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "phemm_impl.h"
#include "BLAS/blas_impl.h"
#include "instantiate.h"
#include <vector>
namespace ddla {
template <DdlaBackend Backend,typename T>
void phemmt(const DdlaHandle_t& h,char side,char uploA,char uploC,int n,
    const T& alpha,const T* A,const int* da,const T* B,const int* db,const T& beta,T* C,const int* dc)
{
    uploC=blas_impl::upper(uploC);
    const char triangle=(uploC=='U' || uploC=='L') ? uploC : '?';
    const bool noop=phemm_impl::phemm_check<Backend,T>(h,side,uploA,n,n,alpha,A,da,B,db,beta,C,dc,true,triangle);
    if (noop) return;
    side=blas_impl::upper(side);uploA=blas_impl::upper(uploA);uploC=blas_impl::upper(uploC);
    std::size_t bytes=0,host_bytes=0;
    phemm_impl::phemm_workspace_size<T>(h,side,n,n,alpha==T(0),da,db,dc,true,&bytes,&host_bytes,uploC);
    vector<char> work(h,bytes);std::vector<char> host(host_bytes);
    phemm_impl::phemm_run<Backend,T>(h,side,uploA,n,n,alpha,A,da,B,db,beta,C,dc,true,
        work.data(),bytes,host.data(),host_bytes,uploC);
}
#define INSTANTIATE(B,T) template void phemmt<B,T>(const DdlaHandle_t&,char,char,char,int,const T&,const T*,const int*,const T*,const int*,const T&,T*,const int*)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE)
#undef INSTANTIATE
template <DdlaBackend Backend,typename R>
void phemmt(const DdlaHandle_t& h,char side,char ua,char uc,int n,
    const std::complex<R>& alpha,const R* A,const int* da,const std::complex<R>* B,const int* db,
    const std::complex<R>& beta,std::complex<R>* C,const int* dc)
{
    using T=std::complex<R>;
    uc=blas_impl::upper(uc);
    const char triangle=(uc=='U' || uc=='L')?uc:'?';
    if (phemm_impl::phemm_check<Backend,T>(h,side,ua,n,n,alpha,A,da,B,db,beta,C,dc,true,triangle)) return;
    side=blas_impl::upper(side);ua=blas_impl::upper(ua);
    std::size_t bytes=0,host_bytes=0;
    phemm_impl::phemm_workspace_size<T,R,std::complex<R>>(h,side,n,n,alpha==T(0),da,db,dc,true,&bytes,&host_bytes,uc);
    vector<char> work(h,bytes);std::vector<char> host(host_bytes);
    phemm_impl::phemm_run<Backend,T>(h,side,ua,n,n,alpha,A,da,B,db,beta,C,dc,true,
        work.data(),bytes,host.data(),host_bytes,uc);
}

template <DdlaBackend Backend,typename R>
void phemmt(const DdlaHandle_t& h,char side,char ua,char uc,int n,
    const std::complex<R>& alpha,const std::complex<R>* A,const int* da,const R* B,const int* db,
    const std::complex<R>& beta,std::complex<R>* C,const int* dc)
{
    using T=std::complex<R>;
    uc=blas_impl::upper(uc);
    const char triangle=(uc=='U' || uc=='L')?uc:'?';
    if (phemm_impl::phemm_check<Backend,T>(h,side,ua,n,n,alpha,A,da,B,db,beta,C,dc,true,triangle)) return;
    side=blas_impl::upper(side);ua=blas_impl::upper(ua);
    std::size_t bytes=0,host_bytes=0;
    phemm_impl::phemm_workspace_size<T,std::complex<R>,R>(h,side,n,n,alpha==T(0),da,db,dc,true,&bytes,&host_bytes,uc);
    vector<char> work(h,bytes);std::vector<char> host(host_bytes);
    phemm_impl::phemm_run<Backend,T>(h,side,ua,n,n,alpha,A,da,B,db,beta,C,dc,true,
        work.data(),bytes,host.data(),host_bytes,uc);
}

template void phemmt<DdlaBackend::CPU,float>(const DdlaHandle_t&,char,char,char,int,const std::complex<float>&,const float*,const int*,const std::complex<float>*,const int*,const std::complex<float>&,std::complex<float>*,const int*);
template void phemmt<DdlaBackend::CPU,float>(const DdlaHandle_t&,char,char,char,int,const std::complex<float>&,const std::complex<float>*,const int*,const float*,const int*,const std::complex<float>&,std::complex<float>*,const int*);
template void phemmt<DdlaBackend::CPU,double>(const DdlaHandle_t&,char,char,char,int,const std::complex<double>&,const double*,const int*,const std::complex<double>*,const int*,const std::complex<double>&,std::complex<double>*,const int*);
template void phemmt<DdlaBackend::CPU,double>(const DdlaHandle_t&,char,char,char,int,const std::complex<double>&,const std::complex<double>*,const int*,const double*,const int*,const std::complex<double>&,std::complex<double>*,const int*);
#if DDLA_HAS_GPU
template void phemmt<DdlaBackend::GPU,float>(const DdlaHandle_t&,char,char,char,int,const std::complex<float>&,const float*,const int*,const std::complex<float>*,const int*,const std::complex<float>&,std::complex<float>*,const int*);
template void phemmt<DdlaBackend::GPU,float>(const DdlaHandle_t&,char,char,char,int,const std::complex<float>&,const std::complex<float>*,const int*,const float*,const int*,const std::complex<float>&,std::complex<float>*,const int*);
template void phemmt<DdlaBackend::GPU,double>(const DdlaHandle_t&,char,char,char,int,const std::complex<double>&,const double*,const int*,const std::complex<double>*,const int*,const std::complex<double>&,std::complex<double>*,const int*);
template void phemmt<DdlaBackend::GPU,double>(const DdlaHandle_t&,char,char,char,int,const std::complex<double>&,const std::complex<double>*,const int*,const double*,const int*,const std::complex<double>&,std::complex<double>*,const int*);
#endif
}
