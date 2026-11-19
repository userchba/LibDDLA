#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "Eigensolver/phegvd_workspace.h"
#include "Eigensolver/eig_params.h"
#include "Cholesky/cholesky_workspace.h"
#include "PBLAS/ptrtrs_impl.h"
#include "check_agree.h"
#include "ddla_desc.h"
#include "ddla_stream.h"
#include "helpers.h"
#include "instantiate.h"
#include "require_backend.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <memory>
#include <optional>
#include <vector>

namespace ddla { namespace spectral_impl {
template<DdlaBackend B,class T>
PhegvdPlan phegvd_check(const DdlaHandle_t& h,char job,char uplo,int n,int nev,T* a,const int* da,
    T* b,const int* db,typename real_type<T>::type* w,T* z,const int* dz)
{
    job=static_cast<char>(std::toupper(static_cast<unsigned char>(job)));
    uplo=static_cast<char>(std::toupper(static_cast<unsigned char>(uplo)));
    const bool vectors=job=='V' && nev>0;
    // Validate all outputs and factor/input aliasing before destroying B.
    detail::agree_check(h,"phegvd",[&](detail::Agree&) {
        detail::require_backend<B>(h,"phegvd");
        if(n<0 || nev<0 || nev>n || (job!='N' && job!='V') || (uplo!='U' && uplo!='L'))
            throw std::invalid_argument("phegvd: invalid order/count/options");
        check_desc(da,h);check_desc(db,h);
        const int nb=da[DDLA_MB_];
        for(const int* d:{da,db}) {
            if(n>d[DDLA_M_] || n>d[DDLA_N_] || d[DDLA_MB_]!=nb || d[DDLA_NB_]!=nb ||
               d[DDLA_RSRC_]!=0 || d[DDLA_CSRC_]!=0)
                throw std::invalid_argument("phegvd: A/B need matching square source-zero blocks and n-by-n extents");
        }
        const int ar=num_loc(n,nb,h->myprow_,0,h->nprows_);
        const int ac=num_loc(n,nb,h->mypcol_,0,h->npcols_);
        if((n && !w) || (ar && ac && (!a || !b)))
            throw std::invalid_argument("phegvd: missing active input/eigenvalues");
        if(detail::matrix_overlaps(a,da[DDLA_LLD_],ar,ac,b,db[DDLA_LLD_],ar,ac))
            throw std::invalid_argument("phegvd: A and B must not overlap");
        if(vectors) {
            check_desc(dz,h);
            if(n>dz[DDLA_M_] || nev>dz[DDLA_N_] || dz[DDLA_MB_]!=nb || dz[DDLA_NB_]!=nb || dz[DDLA_RSRC_]!=0)
                throw std::invalid_argument("phegvd: incompatible eigenvector descriptor");
            const int zc=num_loc(nev,nb,h->mypcol_,dz[DDLA_CSRC_],h->npcols_);
            if(ar && zc && !z)throw std::invalid_argument("phegvd: missing active output");
            if(detail::matrix_overlaps(b,db[DDLA_LLD_],ar,ac,z,dz[DDLA_LLD_],ar,zc))
                throw std::invalid_argument("phegvd: B and Z must not overlap");
        }
    });
    int signature[26]={n,nev,job,uplo,int(B)};
    detail::descriptor_signature(signature+5,da);
    detail::descriptor_signature(signature+12,db);
    if(vectors)detail::descriptor_signature(signature+19,dz);
    detail::agree_signature(h,"phegvd",signature);
    PhegvdPlan p;
    p.backend=B; p.communication=ddlaGetCommunication(h);
    p.job=job; p.uplo=uplo; p.n=n; p.nev=nev; p.vectors=vectors;
    std::copy_n(da,DDLA_DLEN_,p.a.begin());
    std::copy_n(db,DDLA_DLEN_,p.b.begin());
    if (vectors) std::copy_n(dz,DDLA_DLEN_,p.z.begin());
    return p;

}

template<class T>
void phegvd_factor_solve_workspace_size(const DdlaHandle_t& h,const PhegvdPlan& p,T* b,
    std::size_t* bytes,std::size_t* host_bytes)
{
    *bytes=*host_bytes=0;
    if (!p.n) return;
    cholesky_impl::ppotrf_workspace_size<T>(h,p.uplo,p.n,b,p.b.data(),bytes,host_bytes);
    std::size_t backend=0,host=0;
    hegst_impl::congruence_solve_workspace_size<T>(h,p.uplo,p.n,p.a.data(),p.b.data(),&backend,&host);
    *bytes=std::max(*bytes,backend);*host_bytes=std::max(*host_bytes,host);
    if(p.vectors) {
        ptrtrs_impl::ptrtrs_workspace_size<T>(h,'L',p.uplo=='U'?'N':'C',p.n,p.nev,p.b.data(),p.z.data(),&backend,&host);
        *bytes=std::max(*bytes,backend);*host_bytes=std::max(*host_bytes,host);
    }
}
template<DdlaBackend B,class T>
void phegvd_factor_run(const DdlaHandle_t& h,const PhegvdPlan& p,T* b,int& info,
    void* work,std::size_t bytes,void* host,std::size_t host_bytes)
{
    info=0;
    if (!p.n) return;
    cholesky_impl::ppotrf_run<B,T>(h,p.uplo,p.n,b,1,1,p.b.data(),info,work,bytes,host,host_bytes);
    if (info>0) info+=p.n;
}
template<DdlaBackend B,class T>
void phegvd_back_run(const DdlaHandle_t& h,const PhegvdPlan& p,T* b,T* z,
    void* work,std::size_t bytes,void* host,std::size_t host_bytes)
{
    if (p.n && p.vectors)
        ptrtrs_impl::ptrtrs_run<B,T>(h,'L',p.uplo,p.uplo=='U'?'N':'C','N',p.n,p.nev,
            b,p.b.data(),z,p.z.data(),work,bytes,host,host_bytes);
}
template<class T>
PhegvdWorkspace phegvd_workspace_size(const DdlaHandle_t& h,const PhegvdPlan& p,
    const PheevdDriverPlan& eig,const PheevdDriverResources<T>& resources,T* a,T* b,
    typename real_type<T>::type* w)
{
    PhegvdWorkspace q;
    if (!p.n) return q;
    phegvd_factor_solve_workspace_size<T>(h,p,b,&q.factor_solve_backend,&q.factor_solve_host);
    const auto congruence=hegst_impl::congruence_workspace_size<T>(h,p.complete,p.a.data(),p.b.data());
    q.congruence_backend=congruence.backend_bytes;q.congruence_host=congruence.host_bytes;
    q.eigen=pheevd_driver_workspace_size<T>(h,eig,resources,a,w);
    q.backend_bytes=std::max({q.factor_solve_backend,q.congruence_backend,q.eigen.backend_bytes});
    q.host_bytes=std::max({q.factor_solve_host,q.congruence_host,q.eigen.host_bytes});
    return q;
}
template<class T>std::array<std::size_t,2> phegvd_factor_staging_bytes(const DdlaHandle_t& h,const PhegvdPlan& p)
{
    return cholesky_impl::ppotrf_staging_bytes(h,p.uplo,p.n,p.b.data(),sizeof(T));
}
template<class T>std::array<std::size_t,2> phegvd_back_staging_bytes(const DdlaHandle_t& h,const PhegvdPlan& p)
{
    if(!p.n || !p.vectors)return {};
    return ptrtrs_impl::ptrtrs_staging_bytes(h,'L',p.uplo,p.uplo=='U'?'N':'C',
        p.n,p.nev,p.b.data(),p.z.data(),sizeof(T));
}
template<class T>std::array<std::size_t,2> phegvd_staging_bytes(const DdlaHandle_t& h,const PhegvdPlan& p,
    const PheevdDriverPlan& eig,const PheevdDriverResources<T>& resources)
{
    auto high=phegvd_factor_staging_bytes<T>(h,p);
    pblas::staging_max(high,hegst_impl::congruence_staging_bytes<T>(h,p.complete,p.a.data(),p.b.data()));
    pblas::staging_max(high,pheevd_driver_staging_bytes<T>(h,eig,resources));
    pblas::staging_max(high,phegvd_back_staging_bytes<T>(h,p));
    return high;
}
void phegvd_prepare_staging(const DdlaHandle_t& h,const std::array<std::size_t,2>& bytes)
{
    detail::agree_local(h,"phegvd staging preparation",[&]{
        if(bytes[0]||bytes[1])h->host_buffers<std::byte>(bytes[0],bytes[1]);
    });
}

template<DdlaBackend B,class T>
void phegvd_run(const DdlaHandle_t& h,const PhegvdPlan& p,const PheevdDriverPlan& eig,
    PheevdDriverResources<T>& resources,const PhegvdWorkspace& q,T* a,T* b,
    typename real_type<T>::type* w,T* z,int& info,void* work,std::size_t bytes,
    void* host,std::size_t host_bytes)
{
    info=0;
    if (!p.n) return;
    if constexpr (B==DdlaBackend::GPU)
        detail::agree_local(h,"phegvd prepared transport",[&] {
            if (p.communication!=ddlaGetCommunication(h))
                throw std::invalid_argument("phegvd: communication changed; prepare a new composite plan");
            pheevd_driver_transport(h,eig);
        });
    assert(bytes>=q.backend_bytes && host_bytes>=q.host_bytes);
    try {
        phegvd_factor_run<B,T>(h,p,b,info,work,bytes,host,host_bytes);
        if (info) return;
        hegst_impl::congruence_run<B,T>(h,p.complete,a,p.a.data(),b,p.b.data(),work,bytes,host,host_bytes);
        pheevd_driver_run<B,T>(h,eig,resources,q.eigen,a,w,z,info,work,bytes,host,host_bytes);
        if (!info) phegvd_back_run<B,T>(h,p,b,z,work,bytes,host,host_bytes);
        DDLA_CHECK(ddlaSynchronize(h));
    } catch (...) {
        try { (void)ddlaSynchronize(h); } catch (...) {}
        throw;
    }
}
#define SIZES(T) \
    template std::array<std::size_t,2> phegvd_factor_staging_bytes<T>(const DdlaHandle_t&,const PhegvdPlan&); \
    template std::array<std::size_t,2> phegvd_back_staging_bytes<T>(const DdlaHandle_t&,const PhegvdPlan&); \
    template std::array<std::size_t,2> phegvd_staging_bytes<T>(const DdlaHandle_t&,const PhegvdPlan&,const PheevdDriverPlan&,const PheevdDriverResources<T>&); \
    template void phegvd_factor_solve_workspace_size<T>( \
        const DdlaHandle_t&,const PhegvdPlan&,T*,std::size_t*,std::size_t*); \
    template PhegvdWorkspace phegvd_workspace_size<T>( \
        const DdlaHandle_t&,const PhegvdPlan&,const PheevdDriverPlan&, \
        const PheevdDriverResources<T>&,T*,T*,typename real_type<T>::type*)
DDLA_FOR_EACH_TYPE(SIZES)
#undef SIZES
#define RAW(B,T) \
    template PhegvdPlan phegvd_check<B,T>( \
        const DdlaHandle_t&,char,char,int,int,T*,const int*,T*,const int*, \
        typename real_type<T>::type*,T*,const int*); \
    template void phegvd_factor_run<B,T>( \
        const DdlaHandle_t&,const PhegvdPlan&,T*,int&,void*,std::size_t,void*,std::size_t); \
    template void phegvd_back_run<B,T>( \
        const DdlaHandle_t&,const PhegvdPlan&,T*,T*,void*,std::size_t,void*,std::size_t); \
    template void phegvd_run<B,T>( \
        const DdlaHandle_t&,const PhegvdPlan&,const PheevdDriverPlan&, \
        PheevdDriverResources<T>&,const PhegvdWorkspace&,T*,T*, \
        typename real_type<T>::type*,T*,int&,void*,std::size_t,void*,std::size_t)
DDLA_FOR_EACH_BACKEND_TYPE(RAW)
#undef RAW
} // namespace spectral_impl

// Phased public preparation preserves the failed-B early return. The single
// arena grows only between retired phases; reset before growth avoids retaining
// the old allocation or copying dead scratch during ddla::vector::reserve.
template<DdlaBackend B,class T>
void phegvd(const DdlaHandle_t& h,char job,char uplo,int n,int nev,T* a,const int* da,
    T* b,const int* db,typename real_type<T>::type* w,T* z,const int* dz,int& info)
{
    using namespace spectral_impl;
    auto p=phegvd_check<B,T>(h,job,uplo,n,nev,a,da,b,db,w,z,dz);
    info=0;
    if (!n) return;
    ddla::vector<char> work(h);
    std::vector<char> host;
    const auto grow=[&](std::size_t bytes,std::size_t host_bytes,const std::array<std::size_t,2>& staging) {
        detail::agree_local(h,"phegvd workspace",[&] {
            if (bytes>work.capacity()) {
                if (work.capacity()) DDLA_CHECK(ddlaSynchronize(h));
                work.reset(); work.resize(bytes);
            }
            if (host_bytes>host.capacity()) {
                if (host.capacity()) DDLA_CHECK(ddlaSynchronize(h));
                std::vector<char>().swap(host); host.resize(host_bytes);
            }
            if(staging[0]||staging[1])h->host_buffers<std::byte>(staging[0],staging[1]);
        });
    };
    std::size_t bytes=0,host_bytes=0;
    std::array<std::size_t,2> staging{};
    detail::agree_local(h,"phegvd workspace query",[&] {
        phegvd_factor_solve_workspace_size<T>(h,p,b,&bytes,&host_bytes);
        staging=phegvd_factor_staging_bytes<T>(h,p);
    });
    grow(bytes,host_bytes,staging);
    phegvd_factor_run<B,T>(h,p,b,info,work.data(),work.size(),host.data(),host.size());
    if (info) return;
    p.complete=pherm_complete_check(h,B,p.uplo,n,a,p.a.data());
    detail::agree_local(h,"phegvd congruence query",[&] {
        const auto congruence=hegst_impl::congruence_workspace_size<T>(h,p.complete,p.a.data(),p.b.data());
        bytes=std::max(bytes,congruence.backend_bytes);host_bytes=std::max(host_bytes,congruence.host_bytes);
        staging=congruence.staging_bytes;
    });
    grow(bytes,host_bytes,staging);
    hegst_impl::congruence_run<B,T>(h,p.complete,a,p.a.data(),b,p.b.data(),work.data(),work.size(),host.data(),host.size());
    // Selection is intentionally here, matching the original public order.
    const auto route=detail::pheevd_select<B,T>(h,p.vectors?'V':'N',n,nev,p.a.data(),p.vectors?p.z.data():nullptr);
    std::unique_ptr<Phb2stTopology> topology;
    std::optional<PheevdDriverResources<T>> resources;
    std::optional<detail::EigMemProfile> profile;
    PheevdDriverPlan eig;
    try {
        if (route==detail::PheevdRoute::TwoStage) {
            detail::agree_local(h,"phegvd eigensolver profile",[&] {
                profile.emplace(h,eig_profile_enabled(),B==DdlaBackend::GPU);
            });
            if (eig_profile_enabled()) profile->phase("start");
            topology=phb2st_topology_prepare(h);
        }
        eig=pheevd_driver_check<B,T>(h,route,p.vectors?'V':'N',p.uplo,n,nev,a,p.a.data(),w,z,
            p.vectors?p.z.data():nullptr,topology.get());
        detail::agree_local(h,"phegvd eigensolver resources",[&] {
            resources.emplace(h,eig,profile?&*profile:nullptr);
        });
        PheevdDriverWorkspace eq;
        detail::agree_local(h,"phegvd eigensolver query",[&] {
            eq=pheevd_driver_workspace_size<T>(h,eig,*resources,a,w);
            staging=pheevd_driver_staging_bytes<T>(h,eig,*resources);
            pblas::staging_max(staging,phegvd_back_staging_bytes<T>(h,p));
        });
        bytes=std::max(bytes,eq.backend_bytes); host_bytes=std::max(host_bytes,eq.host_bytes);
        grow(bytes,host_bytes,staging);
        if (profile && eig_profile_enabled()) profile->phase("prepare");
        try {
            pheevd_driver_run<B,T>(h,eig,*resources,eq,a,w,z,info,
                work.data(),work.size(),host.data(),host.size());
            if (!info) phegvd_back_run<B,T>(h,p,b,z,work.data(),work.size(),host.data(),host.size());
            DDLA_CHECK(ddlaSynchronize(h));
        } catch (...) {
            resources->release();
            throw;
        }
        resources->release();
        if (topology) topology->release();
    } catch (...) {
        if (topology) topology->release();
        throw;
    }
    resources.reset();
    topology.reset();
    work=ddla::vector<char>(h);
    std::vector<char>().swap(host);
    if (profile && eig_profile_enabled()) {
        profile->phase("free");
        profile->report_timing(h->myid,n,eig.two->band);
        profile->report(h->myid,n);
    }
}
#define GENERALIZED(B,T) template void phegvd<B,T>(const DdlaHandle_t&,char,char,int,int,T*,const int*,T*,const int*,typename real_type<T>::type*,T*,const int*,int&)
DDLA_FOR_EACH_BACKEND_TYPE(GENERALIZED)
#undef GENERALIZED
} // namespace ddla
