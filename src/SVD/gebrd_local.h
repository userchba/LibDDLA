#ifndef DDLA_GEBRD_LOCAL_H
#define DDLA_GEBRD_LOCAL_H
#include <ddla/ddla.h>
namespace ddla { namespace spectral_impl {
template<class T>
void local_gebrd_workspace_size(const DdlaHandle_t&,int,int,std::size_t*);
template<DdlaBackend B,class T>
void local_gebrd_run(const DdlaHandle_t&,int,int,T*,int,
    typename real_type<T>::type*,typename real_type<T>::type*,T*,T*,int&,void*,std::size_t);
} }
#endif
