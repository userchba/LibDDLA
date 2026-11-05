#include <ddla/ddla.h>
#include "prankk_impl.h"

namespace ddla {

template <DdlaBackend Backend, typename T>
void pher2k(const DdlaHandle_t& h, char uplo, char trans, int n, int k,
    const T& alpha, const T* A, const int* da, const T* B, const int* db,
    const typename real_type<T>::type& beta, T* C, const int* dc)
{
    static_assert(!std::is_arithmetic_v<T>, "pher2k: complex scalar type required");
    prankk_impl::execute<true,true,Backend,T>(h,uplo,trans,n,k,
        T(alpha),A,da,B,db,T(beta),C,dc);
}

#define INSTANTIATE(B,T) \
 template void pher2k<B,T>(const DdlaHandle_t&,char,char,int,int,const T&,const T*,const int*,const T*,const int*,const typename real_type<T>::type&,T*,const int*)
#define INSTANTIATE_TYPES(B) \
 INSTANTIATE(B,std::complex<float>); INSTANTIATE(B,std::complex<double>)
INSTANTIATE_TYPES(DdlaBackend::CPU);
#if DDLA_HAS_GPU
INSTANTIATE_TYPES(DdlaBackend::GPU);
#endif
#undef INSTANTIATE_TYPES
#undef INSTANTIATE
} // namespace ddla
