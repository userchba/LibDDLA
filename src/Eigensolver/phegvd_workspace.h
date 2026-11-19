#ifndef DDLA_PHEGVD_WORKSPACE_H
#define DDLA_PHEGVD_WORKSPACE_H
#include "Eigensolver/pheevd_driver.h"
#include "Eigensolver/hegst_impl.h"
#include <array>

namespace ddla { namespace spectral_impl {
struct PhegvdPlan {
    DdlaBackend backend{};
    DdlaCommunication communication{};
    char job='N', uplo='U';
    int n=0, nev=0;
    bool vectors=false;
    std::array<int,DDLA_DLEN_> a{},b{},z{};
    HermitianPlan complete{};
};
struct PhegvdWorkspace {
    std::size_t factor_solve_backend=0, factor_solve_host=0;
    std::size_t congruence_backend=0, congruence_host=0;
    PheevdDriverWorkspace eigen{};
    std::size_t backend_bytes=0, host_bytes=0;
};
template<DdlaBackend B,class T>
PhegvdPlan phegvd_check(const DdlaHandle_t&,char,char,int,int,T*,const int*,T*,const int*,
    typename real_type<T>::type*,T*,const int*);
// Preparation is intentionally separate: the public wrapper does not read
// method/tuning options or prepare eigensolver resources before B succeeds.
// Prepared raw callers may create all plans/resources before factorization.
template<class T>
void phegvd_factor_solve_workspace_size(const DdlaHandle_t&,const PhegvdPlan&,T*,std::size_t*,std::size_t*);
template<DdlaBackend B,class T>
void phegvd_factor_run(const DdlaHandle_t&,const PhegvdPlan&,T*,int&,void*,std::size_t,void*,std::size_t);
template<DdlaBackend B,class T>
void phegvd_back_run(const DdlaHandle_t&,const PhegvdPlan&,T*,T*,void*,std::size_t,void*,std::size_t);
// Pure domain-wise maximum. The caller has checked p.complete and prepared
// the selected eigen plan/resources and topology; no owner is created here.
template<class T>
PhegvdWorkspace phegvd_workspace_size(const DdlaHandle_t&,const PhegvdPlan&,const PheevdDriverPlan&,
    const PheevdDriverResources<T>&,T*,T*,typename real_type<T>::type*);
// Pure transport-only queries, separate from native workspace sizing and
// retained resources. A successful execution attains these geometry maxima;
// an early non-positive-B/numerical return can communicate less. The complete
// query requires all checked/prepared future plans, but reads no matrix data.
template<class T>std::array<std::size_t,2> phegvd_factor_staging_bytes(const DdlaHandle_t&,const PhegvdPlan&);
template<class T>std::array<std::size_t,2> phegvd_back_staging_bytes(const DdlaHandle_t&,const PhegvdPlan&);
template<class T>std::array<std::size_t,2> phegvd_staging_bytes(const DdlaHandle_t&,const PhegvdPlan&,
    const PheevdDriverPlan&,const PheevdDriverResources<T>&);
// Explicit preparation outside pure queries and raw runs; failures are agreed
// before any destructive phase. Cache ownership stays on the handle and its
// two retained capacities are NOT part of backend_bytes/host_bytes.
void phegvd_prepare_staging(const DdlaHandle_t&,const std::array<std::size_t,2>&);
// Fully prepared execution. Unlike public phased preparation, future method
// checks/resources already exist on entry. Before the first run, explicitly
// reserve phegvd_staging_bytes with phegvd_prepare_staging (cold handles too).
// No allocation/query; borrowed work
// and input/output spans must be disjoint. Bound to the prepared handle state.
template<DdlaBackend B,class T>
void phegvd_run(const DdlaHandle_t&,const PhegvdPlan&,const PheevdDriverPlan&,PheevdDriverResources<T>&,
    const PhegvdWorkspace&,T*,T*,typename real_type<T>::type*,T*,int&,
    void*,std::size_t,void*,std::size_t);
} }
#endif
