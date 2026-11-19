#ifndef DDLA_PHEEVD_DRIVER_H
#define DDLA_PHEEVD_DRIVER_H

// Private composition of the existing three routes; no numerical policy here.
#include "Eigensolver/pheevd_impl.h"
#include "Eigensolver/one_stage/one_stage_workspace.h"
#include "Eigensolver/two_stage/pheevd_impl.h"
#include <ddla/ddla_vector.h>
#include <optional>
#include <stdexcept>

namespace ddla { namespace spectral_impl {
struct PheevdDriverPlan {
    detail::PheevdRoute route=detail::PheevdRoute::Local;
    DdlaBackend backend{};
    DdlaCommunication communication{};
    int n=0;
    std::optional<detail::LocalPheevdPlan> local;
    std::optional<OneStagePlan> one;
    std::optional<PheevdPlan> two;
};
struct PheevdDriverWorkspace {
    std::size_t backend_bytes=0, host_bytes=0;
    detail::LocalPheevdWorkspace local{};
};
// A two-stage plan borrows topology state. Its owner must outlive the plan,
// resources and every run, and release collectively after resources.release().
template<DdlaBackend B,class T>
PheevdDriverPlan pheevd_driver_check(const DdlaHandle_t& h,detail::PheevdRoute route,
    char job,char uplo,int n,int nev,T* a,const int* da,typename real_type<T>::type* w,
    T* z,const int* dz,const Phb2stTopology* topology)
{
    PheevdDriverPlan p;
    p.route=route; p.backend=B; p.communication=ddlaGetCommunication(h); p.n=n;
    if (route==detail::PheevdRoute::Local)
        p.local=detail::pheevd_local_check<B,T>(h,job,uplo,n,nev,a,da,w,z,dz);
    else if (route==detail::PheevdRoute::OneStage)
        p.one=one_stage_check<T>(h,B,job,uplo,n,a,da,w,z,dz,nev);
    else {
        if (!topology) throw std::invalid_argument("pheevd: missing prepared topology");
        p.two=pheevd_check<T>(h,B,job,uplo,n,a,da,w,z,dz,*topology,'A',nullptr,DdlaBackend::CPU,nev);
    }
    return p;
}

template<class T> class PheevdDriverResources {
    using R=typename real_type<T>::type;
public:
    ddla::vector<R> local_values;
    ddla::vector<int> local_info;
    std::optional<OneStageResources> one;
    std::optional<PheevdResources<T>> two;
    PheevdDriverResources(const DdlaHandle_t& h,const PheevdDriverPlan& p,
                          detail::EigMemProfile* profile=nullptr)
        :local_values(h),local_info(h)
    {
        if (p.local && p.n && p.backend==DdlaBackend::GPU) {
            local_values.resize(p.n); local_info.resize(1);
        }
        if (p.one) one.emplace(h,*p.one);
        if (p.two) two.emplace(h,*p.two,profile);
    }
    // Query/value resources are retained independently of the phase arena.
    std::size_t retained_backend_bytes() const noexcept {
        return local_values.size()*sizeof(R)+local_info.size()*sizeof(int);
    }
    void release() { if (two) two->release(); }
};

inline void pheevd_driver_transport(const DdlaHandle_t& h,const PheevdDriverPlan& p)
{
    if (p.backend==DdlaBackend::GPU && p.communication!=ddlaGetCommunication(h))
        throw std::invalid_argument("pheevd: communication changed; prepare a new composite plan");
}
template<class T>
PheevdDriverWorkspace pheevd_driver_workspace_size(const DdlaHandle_t& h,const PheevdDriverPlan& p,
    const PheevdDriverResources<T>& resources,T* a,typename real_type<T>::type* w)
{
    pheevd_driver_transport(h,p);
    PheevdDriverWorkspace q;
    if (p.local) {
        // The vendor query needs an actual backend eigenvalue address.
        auto* values=p.backend==DdlaBackend::GPU?const_cast<typename real_type<T>::type*>(resources.local_values.data()):w;
        q.local=detail::pheevd_local_workspace_size<T>(h,*p.local,a,values);
        q.backend_bytes=q.local.backend_bytes; q.host_bytes=q.local.host_bytes;
    } else if (p.one)
        one_stage_workspace_size<T>(h,*p.one,&q.backend_bytes,&q.host_bytes);
    else if (p.two)
        pheevd_workspace_size<T>(h,*p.two,a,&q.backend_bytes,&q.host_bytes);
    return q;
}
// No native workspace query, allocation or communication. Resources must be
// prepared first (TwoStage's GPU halo uses its frozen local geometry).
template<class T>
std::array<std::size_t,2> pheevd_driver_staging_bytes(const DdlaHandle_t& h,const PheevdDriverPlan& p,
    const PheevdDriverResources<T>& resources)
{
    pheevd_driver_transport(h,p);
    if(p.one)return one_stage_staging_bytes<T>(h,*p.one);
    if(p.two){
        if(!resources.two)throw std::invalid_argument("pheevd: missing prepared two-stage resources");
        return pheevd_staging_bytes<T>(h,*p.two,*resources.two);
    }
    return {}; // The Local route never communicates backend matrices.
}
template<DdlaBackend B,class T>
void pheevd_driver_run(const DdlaHandle_t& h,const PheevdDriverPlan& p,PheevdDriverResources<T>& r,
    const PheevdDriverWorkspace& q,T* a,typename real_type<T>::type* w,T* z,int& info,
    void* backend,std::size_t backend_bytes,void* host,std::size_t host_bytes)
{
    if constexpr (B==DdlaBackend::GPU)
        detail::agree_local(h,"pheevd composite transport",[&]{pheevd_driver_transport(h,p);});
    if (p.local)
        detail::pheevd_local_run<B,T>(h,*p.local,q.local,a,w,z,info,r.local_values.data(),r.local_info.data(),
            backend,backend_bytes,host,host_bytes);
    else if (p.one)
        one_stage_run<B,T>(h,*p.one,*r.one,a,w,z,info,nullptr,backend,backend_bytes,host,host_bytes);
    else
        pheevd_run<B,T>(h,*p.two,*r.two,a,w,z,info,backend,backend_bytes,host,host_bytes);
}
} }
#endif
