#include <ddla/ddla.h>
#include "prankk_impl.h"

namespace ddla {

template <DdlaBackend Backend, typename T>
void psyrk(const DdlaHandle_t& h, char uplo, char trans, int n, int k,
    const T& alpha, const T* A, const int* da,
    const T& beta, T* C, const int* dc)
{
    prankk_impl::execute<false,false,Backend,T>(h,uplo,trans,n,k,
        T(alpha),A,da,A,da,T(beta),C,dc);
}

#define INSTANTIATE(B,T) \
 template void psyrk<B,T>(const DdlaHandle_t&,char,char,int,int,const T&,const T*,const int*,const T&,T*,const int*)
#define INSTANTIATE_TYPES(B) \
 INSTANTIATE(B,float); INSTANTIATE(B,double); \
 INSTANTIATE(B,std::complex<float>); INSTANTIATE(B,std::complex<double>)
INSTANTIATE_TYPES(DdlaBackend::CPU);
#if DDLA_HAS_GPU
INSTANTIATE_TYPES(DdlaBackend::GPU);
#endif
#undef INSTANTIATE_TYPES
#undef INSTANTIATE
} // namespace ddla
