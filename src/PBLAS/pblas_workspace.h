#ifndef DDLA_PBLAS_WORKSPACE_H
#define DDLA_PBLAS_WORKSPACE_H

#include <ddla/ddla.h>
#include <ddla/ddla_desc.h>
#include "ddla_stream.h"

#include <algorithm>

namespace ddla { namespace pblas {

// Derived descriptors in a trusted algorithm need index geometry only. The
// public descriptor constructor remains responsible for validating user input.
inline void describe(int* d,DdlaHandle_t h,int m,int n,int mb,int nb,int rs,int cs,int lld=-1) noexcept
{
    d[DDLA_DTYPE_]=DDLA_BLOCK_CYCLIC_2D;d[DDLA_CTXT_]=0;
    d[DDLA_M_]=m;d[DDLA_N_]=n;d[DDLA_MB_]=mb;d[DDLA_NB_]=nb;
    d[DDLA_RSRC_]=rs;d[DDLA_CSRC_]=cs;
    d[DDLA_LLD_]=std::max(lld,std::max(1,num_loc(m,mb,h->myprow_,rs,h->nprows_)));
}

}}
#endif
