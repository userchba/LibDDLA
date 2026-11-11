#include <ddla/ddla.h>
#include "prankk_impl.h"
#include "instantiate.h"

namespace ddla {

template <DdlaBackend Backend, typename T>
void pherk(const DdlaHandle_t& h, char uplo, char trans, int n, int k,
    const typename real_type<T>::type& alpha, const T* A, const int* da,
    const typename real_type<T>::type& beta, T* C, const int* dc)
{
    static_assert(!std::is_arithmetic_v<T>, "pherk: complex scalar type required");
    prankk_impl::execute<true,false,Backend,T>(h,uplo,trans,n,k,
        T(alpha),A,da,A,da,T(beta),C,dc);
}

#define INSTANTIATE(B,T) \
 template void pherk<B,T>(const DdlaHandle_t&,char,char,int,int,const typename real_type<T>::type&,const T*,const int*,const typename real_type<T>::type&,T*,const int*)
#define INSTANTIATE_TYPES(B) \
 INSTANTIATE(B,std::complex<float>); INSTANTIATE(B,std::complex<double>)
DDLA_FOR_EACH_BACKEND(INSTANTIATE_TYPES)
#undef INSTANTIATE_TYPES
#undef INSTANTIATE
} // namespace ddla
