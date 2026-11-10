#ifndef DDLA_HB2ST_HOUSEHOLDER_GPU_H
#define DDLA_HB2ST_HOUSEHOLDER_GPU_H
#include <ddla/ddla.h>
namespace ddla { namespace detail {
// One bounded panel in application order: [tau, vector tail], band scalars
// per reflector, implicit leading one. rows is a device row-major strip.
template<class T>
void hh_gpu_panel_run(const DdlaHandle_t&, T* rows, int ld, int columns,
                      int n, int band, int first, int count, const T* packed);
// Contiguous owned band columns plus a bounded temporal halo. Only bounded
// wavefront packets are retained on the device; history remains owner-partitioned.
template<class T>
void hh_gpu_reduce_levels(const DdlaHandle_t&,int,int,int,int,int,int,int,int,T*,T*,int*);
} }
#endif
