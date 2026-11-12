#include <ddla/ddla.h>
#include "lapack_impl.h"
#include "instantiate.h"
#include <type_traits>

namespace ddla {
template<DdlaBackend B,class T>
void syevd(const DdlaHandle_t& h,char job,char uplo,int n,T* a,int lda,T* w,int* info)
{
    static_assert(std::is_same_v<T,float> || std::is_same_v<T,double>,"syevd requires real scalars");
    lapack_impl::heevd_allocate<B,T>(h,job,uplo,n,a,lda,w,info);
}
#define SYEVD(B,T) template void syevd<B,T>(const DdlaHandle_t&,char,char,int,T*,int,T*,int*)
#define SYEVD_TYPES(B) SYEVD(B,float); SYEVD(B,double)
DDLA_FOR_EACH_BACKEND(SYEVD_TYPES)
#undef SYEVD_TYPES
#undef SYEVD
} // namespace ddla
