#ifndef DDLA_PBDSDC_IMPL_H
#define DDLA_PBDSDC_IMPL_H
#include <ddla/ddla.h>
#include "dc_threads.h"
#include <algorithm>
#include <cstddef>

namespace ddla { namespace spectral_impl {
// The checked call and its frozen policy: the leaf size (DDLA_SVD_DC_LEAF, read
// on rank 0) and the host threads of the secular work, agreed by the check.
struct PbdsdcPlan {
    DdlaBackend backend{};char uplo='U';
    int n=0,ranks=1,rank=0,threads=1,leaf=128,ld=1;
};
void pbdsdc_freeze_policy(const DdlaHandle_t&,PbdsdcPlan&);
PbdsdcPlan pbdsdc_check(const DdlaHandle_t&,DdlaBackend,char,int,const void*,const void*,
    const void*,const void*,const void*,int);

// Value-dependent splits can put large leaves on one rank and singleton
// leaves on its peers. Bound that skew, rather than assuming 1/P storage.
// A positive leaf a<=L,m<=a+1 retains a+a*a+m*m <= 2*a*a+4*a
// scalars. For c owned leaves, at least rank+(c-1)*(P-1) other positive
// leaves consume the global n budget. Convexity maximizes sum(a*a) by
// filling owned leaves to L before the next one. L=2 may produce empty
// children, so it uses the safe global (2*L+4)*n bound. The bound is O(n*L)
// per rank under skew, L<=256, never a new replicated n*n matrix.
// The result counts reals: this rank's retained leaf factors.
inline std::size_t pbdsdc_retained_bound(const PbdsdcPlan& p)
{
    const std::size_t n=p.n,L=std::min(p.n,p.leaf);
    if(!n||static_cast<std::size_t>(p.rank)>=n)return 0;
    if(L<=2)return (2*L+4)*n;
    std::size_t bound=0;
    const auto max_count=1+(n-1-p.rank)/p.ranks;
    for(std::size_t count=1;count<=max_count;++count) {
        const auto other=p.rank+(count-1)*(p.ranks-1);
        if(other>=n||n-other<count)break;
        const auto total=std::min(count*L,n-other);
        const auto extra=total-count,full=extra/(L-1),tail=extra%(L-1);
        auto squares=full*(L*L);
        if(full<count)squares+=(tail+1)*(tail+1)+(count-full-1);
        bound=std::max(bound,2*squares+4*total);
    }
    return bound;
}

// The scratch of pbdsdc_run on this rank (none for n == 0), in carving order.
// Host: the independent trees (n records) and their index arrays (4n ints); then the
// leaf phase and the merge phase share the rest.  Leaf phase: the leaf records (n),
// this rank's retained leaf factors (pbdsdc_retained_bound), the receive buffer of one
// leaf, per-worker error slots and private DLASDQ scratch for each prepared
// worker (at most min(threads, owned leaves)). Merge phase: the 1-based
// vectors of DLASD1 / DLASD2 / DLASD3 (n + 3 each), the secular columns of this rank's share, the decision and
// permutation records and the deflation rotations.  Backend: the merge slabs U2 / V2
// (rows x n each), the coefficient panels (2n x min(ceil(n/P), 512)) and, on a GPU handle, the
// device permutation and rotations.
template<class R>
void pbdsdc_workspace_size(const DdlaHandle_t&,const PbdsdcPlan&,
    std::size_t* backend_bytes,std::size_t* host_bytes);
template<DdlaBackend B,class R>
void pbdsdc_run(const DdlaHandle_t&,const PbdsdcPlan&,detail::DcHostTeam&,
    R* d,R* e,R* s,R* U,R* V,int& info,
    void* backend_work,std::size_t backend_bytes,void* host_work,std::size_t host_bytes);
} }
#endif
