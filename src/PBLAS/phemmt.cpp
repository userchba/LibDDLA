#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "phemm_impl.h"
#include "BLAS/blas_impl.h"
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
#define TYPES(B) \
 INSTANTIATE(B,float); INSTANTIATE(B,double); INSTANTIATE(B,std::complex<float>); INSTANTIATE(B,std::complex<double>)
TYPES(DdlaBackend::CPU);
#if DDLA_HAS_GPU
TYPES(DdlaBackend::GPU);
#endif
#undef TYPES
#undef INSTANTIATE
}
