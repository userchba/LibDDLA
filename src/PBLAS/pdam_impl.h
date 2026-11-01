#ifndef DDLA_PDAM_IMPL_H
#define DDLA_PDAM_IMPL_H
#include <ddla/ddla.h>
namespace ddla { namespace pdam_impl {
// This operation is scratch-free on every backend: check and execution only.
template <DdlaBackend Backend>
int pdam_check(const DdlaHandle_t&,int,const int*);
template <DdlaBackend Backend,typename T1,typename T2>
void pdam_run(const DdlaHandle_t&,int,const T1&,T2*,const int*);
}}
#endif
