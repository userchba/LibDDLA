#ifndef DDLA_PHEEVD_ONE_STAGE_H
#define DDLA_PHEEVD_ONE_STAGE_H
#include <ddla/ddla.h>
namespace ddla { namespace detail {
// Optional private benchmark result. Timings synchronize the handle only when
// requested; the public entry has no instrumentation or additional barriers.
struct OneStageTimes { double prepare=0, reduction=0, solve=0, redistribution=0, backtransform=0; };
template<DdlaBackend Backend, class T>
void pheevd_one_stage_timed(const DdlaHandle_t&,char,char,int,T*,const int*,
    typename real_type<T>::type*,T*,const int*,int&,OneStageTimes*);
} }
#endif
