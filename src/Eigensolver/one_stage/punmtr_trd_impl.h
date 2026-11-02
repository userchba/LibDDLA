#ifndef DDLA_PUNMTR_TRD_IMPL_H
#define DDLA_PUNMTR_TRD_IMPL_H
#include "phetrd_impl.h"
namespace ddla { namespace spectral_impl {
struct UnmtrTrdPlan {
    DdlaBackend backend{};
    char side='L',uplo='U',trans='N';
    int m=0,n=0,nq=0;
    int pr=1,pc=1,mr=0,mc=0;
    std::array<int,DDLA_DLEN_> a{},c{};
};
UnmtrTrdPlan punmtr_trd_geometry(const DdlaHandle_t&,DdlaBackend,char,char,char,int,int,const int*,const int*);
template<class T>UnmtrTrdPlan punmtr_trd_check(const DdlaHandle_t&,DdlaBackend,char,char,char,int,int,
    const T*,const int*,const T*,T*,const int*);
template<class T>void punmtr_trd_workspace_size(const DdlaHandle_t&,const UnmtrTrdPlan&,std::size_t*);
template<DdlaBackend B,class T>void punmtr_trd_run(const DdlaHandle_t&,const UnmtrTrdPlan&,
    const T* A,const T* tau,T* C,void*,std::size_t);
static_assert(std::is_trivially_copyable<UnmtrTrdPlan>::value,"stored-reflector plan owns no resources");
} }
#endif
