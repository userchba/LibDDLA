#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "hemmt_impl.h"
#include "blas_impl.h"

namespace ddla {
template <DdlaBackend Backend,typename T>
void symmt(const DdlaHandle_t& h,char side,char uploA,char uploC,int n,
    const T& alpha,const T* A,int lda,const T* B,int ldb,const T& beta,T* C,int ldc)
{
    if (blas_impl::hemmt_check<Backend,T>(h,side,uploA,uploC,n,alpha,A,lda,B,ldb,beta,C,ldc,"symmt")) return;
    std::size_t bytes=0;
    blas_impl::hemmt_workspace_size<T>(h,n,alpha!=T(0),&bytes);
    vector<char> work(h,bytes);
    blas_impl::hemmt_run<Backend,T>(h,blas_impl::upper(side),blas_impl::upper(uploA),blas_impl::upper(uploC),
        n,alpha,A,lda,B,ldb,beta,C,ldc,work.data(),bytes,false);
}
#define INSTANTIATE(B,T) template void symmt<B,T>(const DdlaHandle_t&,char,char,char,int,const T&,const T*,int,const T*,int,const T&,T*,int)
#define TYPES(B) \
 INSTANTIATE(B,float); INSTANTIATE(B,double); INSTANTIATE(B,std::complex<float>); INSTANTIATE(B,std::complex<double>)
TYPES(DdlaBackend::CPU);
#if DDLA_HAS_GPU
TYPES(DdlaBackend::GPU);
#endif
#undef TYPES
#undef INSTANTIATE
}
