#ifndef DDLA_PHETRD_IMPL_H
#define DDLA_PHETRD_IMPL_H
#include <ddla/ddla.h>
#include "ddla_desc.h"
#include "ddla_stream.h"
#include "comm_traits.h"
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace ddla { namespace detail {
// Lower storage is the conjugate-transposed view of the same Hermitian
// matrix. Both storage triangles use the same prefix/QL reflector convention.
inline std::size_t triangle_offset(bool lower,int ld,int r,int c) {
    return lower?c+static_cast<std::size_t>(r)*ld:r+static_cast<std::size_t>(c)*ld;
}
} namespace spectral_impl {
struct HetrdPlan {
    DdlaBackend backend{};
    char uplo='U';bool stored_only=true;
    int n=0,pr=1,pc=1,mr=0,mc=0;
    std::array<int,DDLA_DLEN_> a{};
};
HetrdPlan phetrd_geometry(const DdlaHandle_t&,DdlaBackend,char,int,const int*,bool stored_only=true);
template<class T>HetrdPlan phetrd_check(const DdlaHandle_t&,DdlaBackend,char,int,const T*,const int*,
    const typename real_type<T>::type*,const typename real_type<T>::type*,const T*);
template<class T>void phetrd_workspace_size(const DdlaHandle_t&,const HetrdPlan&,std::size_t*,std::size_t*);
// Construct outside numerical runs. Cache changes are scoped to reduction.
class HetrdResources {
    std::size_t cache_capacity_=0;
    bool cache_eligible_=false,cache_changed_=false;
public:
    HetrdResources(const DdlaHandle_t&,const HetrdPlan&);
    ~HetrdResources();
    HetrdResources(const HetrdResources&)=delete;
    HetrdResources& operator=(const HetrdResources&)=delete;
    void begin_reduction();void end_reduction();void cancel_reduction() noexcept;
};
// tau: n-1 backend entries; de: contiguous replicated host [d(n)|e(n)].
// Scratch contains only reduction temporaries; no allocation/check/query/setup.
template<DdlaBackend B,class T>void phetrd_run(const DdlaHandle_t&,const HetrdPlan&,HetrdResources&,
    T* A,T* tau,typename real_type<T>::type* de,int& info,void*,std::size_t,void*,std::size_t);
static_assert(std::is_trivially_copyable<HetrdPlan>::value,"phetrd plan owns no resources");
} }
#endif
