#ifndef DDLA_ONE_STAGE_WORKSPACE_H
#define DDLA_ONE_STAGE_WORKSPACE_H

#include <ddla/ddla.h>
#include "pheevd_one_stage.h"
#include "phetrd_impl.h"
#include "Eigensolver/eig_workspace.h"
#include "Eigensolver/pstedc_impl.h"
#include <optional>
#include <array>
#include <cstddef>
#include <type_traits>

namespace ddla { namespace spectral_impl {

// All geometry, collective policy and alias decisions are frozen by check.
// Descriptors retain their rank-local LLD; only shared fields are agreed.
struct OneStagePlan : HetrdPlan {
    char job='N';bool backup=false;
    std::array<int,DDLA_DLEN_> z{},saved{},tridiagonal_vectors{};
    std::size_t output_scratch=0;
    HermitianPlan complete{};
    PstedcVectorPlan vectors{};PstedcPlan values{};
};
template<class T> OneStagePlan one_stage_check(const DdlaHandle_t&,DdlaBackend,bool,char,char,int,
    const T*,const int*,const typename real_type<T>::type*,const T*,const int*);

// The scratch of one_stage_run on this rank (none for n == 0).  Backend, in carving
// order: the reflector scalars tau (n - 1), the copy of the reflectors when Z overlaps A,
// then one region that the phases take in turn: the Hermitian completion (full storage),
// the reduction, then the eigenvalue-only D&C, or row-slab D&C followed by
// redistribution (plus a compact real 2D tile for complex output), then the
// back-transform. The row slab stays live through redistribution; their scratch
// regions are reused. Host: the tridiagonal
// (d | e, 2n), then the phases' host scratch, the largest of them.
template<class T> void one_stage_workspace_size(const DdlaHandle_t&,const OneStagePlan&,
    std::size_t* backend_bytes,std::size_t* host_bytes);

// Prepare both the reduction cache policy and the row solver's upload fence.
// Inherited reduction controls remain available to direct workspace callers.
class OneStageResources : public HetrdResources {
public:
    std::optional<PstedcResources> vectors;
    OneStageResources(const DdlaHandle_t& h,const OneStagePlan& p):HetrdResources(h,p) {
        if(p.n&&p.job=='V')vectors.emplace(h,p.vectors);
    }
};
// The solve on the caller's scratch; times (may be null) receives the phase timings.
// No validation, no allocation.
template<DdlaBackend B,class T>
void one_stage_run(const DdlaHandle_t&,const OneStagePlan&,OneStageResources&,T* a,
    typename real_type<T>::type* eval,T* z,int& info,detail::OneStageTimes* times,
    void* backend_work,std::size_t backend_bytes,void* host_work,std::size_t host_bytes);

static_assert(std::is_trivially_copyable<OneStagePlan>::value,"one-stage check owns no resources");
} }
#endif
