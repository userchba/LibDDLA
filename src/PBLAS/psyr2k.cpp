#include <ddla/ddla.h>
#include "prankk_impl.h"
#include "instantiate.h"

namespace ddla {

template <DdlaBackend Backend, typename T>
void psyr2k(const DdlaHandle_t& h, char uplo, char trans, int n, int k,
    const T& alpha, const T* A, const int* da, const T* B, const int* db,
    const T& beta, T* C, const int* dc)
{
    prankk_impl::execute<false,true,Backend,T>(h,uplo,trans,n,k,
        T(alpha),A,da,B,db,T(beta),C,dc);
}

#define INSTANTIATE(B,T) \
 template void psyr2k<B,T>(const DdlaHandle_t&,char,char,int,int,const T&,const T*,const int*,const T*,const int*,const T&,T*,const int*)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE)
#undef INSTANTIATE
} // namespace ddla
