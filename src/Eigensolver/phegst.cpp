#include "hegst_impl.h"
#include <ddla/ddla_vector.h>
#include "BLAS/blas_impl.h"
#include "PBLAS/ptrtrs_impl.h"
#include "PBLAS/pblas_workspace.h"
#include "check_agree.h"
#include "ddla_desc.h"
#include "ddla_stream.h"
#include "instantiate.h"
#include "require_backend.h"
#include <algorithm>
#include <cctype>
#include <climits>
#include <exception>
#include <stdexcept>
#include <vector>

namespace ddla { namespace hegst_impl {
namespace {
void current(const DdlaHandle_t& h,const Plan& p)
{
    const auto& c=p.completion;
    if(!h || h->backend!=c.backend || h->communication()!=p.transport || h->comm!=p.comm ||
       h->nprows_!=c.pr || h->npcols_!=c.pc || h->myprow_!=c.mr ||
       h->mypcol_!=c.mc || h->major!=c.major || h->host_threads!=c.threads ||
       h->stream!=p.compute || h->stream_data!=p.data)
        throw std::invalid_argument("phegst: prepared handle configuration changed");
    if(c.n && c.backend==DdlaBackend::GPU && p.transport==DdlaCommunication::CCL &&
       h->communication_group_depth)
        throw std::invalid_argument("phegst: close the active CCL group first");
}
template<DdlaBackend B> void drain(const DdlaHandle_t& h)
{
    std::exception_ptr first;
    try { DDLA_CHECK(ddlaSynchronize(h)); }
    catch(...) { first=std::current_exception(); }
    if constexpr(B==DdlaBackend::GPU) {
        try { DDLA_CHECK(detail::synchronize_on(h,h->stream_data)); }
        catch(...) { if(!first)first=std::current_exception(); }
    }
    if(first)std::rethrow_exception(first);
}
}

template<class T>
Plan check(const ddla::DdlaHandle_t& h,ddla::DdlaBackend backend,char uplo,int n,
           T* a,const int* da,const T* factor,const int* df)
{
    using namespace ddla;
    uplo=static_cast<char>(std::toupper(static_cast<unsigned char>(uplo)));
    const auto p=detail::agree_check(h,"phegst",[&](detail::Agree&)->Plan {
        detail::require_backend(h,backend,"phegst");
        if (n<0 || n>INT_MAX/8 || (uplo!='L' && uplo!='U'))
            throw std::invalid_argument("phegst: invalid order or triangle");
        check_desc(da,h);
        check_desc(df,h);
        const int block=da[DDLA_MB_];
        for (const int* d:{da,df})
            if (d[DDLA_MB_]!=block || d[DDLA_NB_]!=block ||
                d[DDLA_RSRC_]!=0 || d[DDLA_CSRC_]!=0 || n>d[DDLA_M_] || n>d[DDLA_N_])
                throw std::invalid_argument("phegst: matching square source-zero descriptors required");
        const int nr=num_loc(n,block,h->myprow_,0,h->nprows_);
        const int nc=num_loc(n,block,h->mypcol_,0,h->npcols_);
        if (nr && nc && (!a || !factor))
            throw std::invalid_argument("phegst: missing active matrix or factor");
        if (detail::matrix_overlaps(a,da[DDLA_LLD_],nr,nc,factor,df[DDLA_LLD_],nr,nc))
            throw std::invalid_argument("phegst: A and factor must be disjoint");
        if (n && backend==DdlaBackend::GPU && h->communication()==DdlaCommunication::CCL &&
            h->communication_group_depth)
            throw std::invalid_argument("phegst: close the active CCL group first");
        Plan q;
        q.transport=h->communication();
        q.comm=h->comm;
        q.compute=h->stream;
        q.data=h->stream_data;
        std::copy_n(da,DDLA_DLEN_,q.a);
        std::copy_n(df,DDLA_DLEN_,q.factor);
        q.completion={backend,uplo=='U',true,n,block,da[DDLA_LLD_],
                      h->nprows_,h->npcols_,h->myprow_,h->mypcol_,h->host_threads,h->major};
        return q;
    });
    int signature[18]={int(backend),int(p.transport),int(uplo),n};
    detail::descriptor_signature(signature+4,da);
    detail::descriptor_signature(signature+11,df);
    detail::agree_signature(h,"phegst",signature);
    return p;
}

template<class T>
void congruence_solve_workspace_size(const DdlaHandle_t& h,char uplo,int n,
    const int* a,const int* factor,std::size_t* backend,std::size_t* host)
{
    *backend=*host=0;
    // Matching square source-zero descriptors put the whole active matrix on
    // (0,0) when it fits in one block. Local TRSM needs no borrowed scratch.
    if(!n || n<=a[DDLA_MB_])return;
    std::size_t b=0,s=0;
    ptrtrs_impl::ptrtrs_workspace_size<T>(h,'L',uplo=='U'?'C':'N',n,n,factor,a,backend,host);
    ptrtrs_impl::ptrtrs_workspace_size<T>(h,'R',uplo=='U'?'N':'C',n,n,factor,a,&b,&s);
    *backend=std::max(*backend,b);*host=std::max(*host,s);
}
template<class T>
std::array<std::size_t,2> congruence_staging_bytes(const DdlaHandle_t& h,
    const spectral_impl::HermitianPlan& p,const int* a,const int* factor)
{
    std::array<std::size_t,2> bytes{};
    if(!p.n || p.n<=p.nb || p.backend!=DdlaBackend::GPU ||
       h->communication()!=DdlaCommunication::HostStagedMPI)return bytes;
    std::size_t pack=0,local=0,unpack=0,diagonal=0,send=0,receive=0;
    spectral_impl::pherm_complete_geometry(p,pack,local,unpack,diagonal,send,receive);
    bytes={pblas::communication_bytes(send,sizeof(T)),pblas::communication_bytes(receive,sizeof(T))};
    const char uplo=p.upper?'U':'L';
    pblas::staging_max(bytes,ptrtrs_impl::ptrtrs_staging_bytes(h,'L',uplo,uplo=='U'?'C':'N',p.n,p.n,factor,a,sizeof(T)));
    pblas::staging_max(bytes,ptrtrs_impl::ptrtrs_staging_bytes(h,'R',uplo,uplo=='U'?'N':'C',p.n,p.n,factor,a,sizeof(T)));
    return bytes;
}
template<class T>
Workspace congruence_workspace_size(const DdlaHandle_t& h,const spectral_impl::HermitianPlan& p,
    const int* a,const int* factor)
{
    Workspace q;
    if(!p.n || p.n<=p.nb)return q;
    congruence_solve_workspace_size<T>(h,p.upper?'U':'L',p.n,a,factor,&q.backend_bytes,&q.host_bytes);
    std::size_t b=0,s=0;
    spectral_impl::pherm_complete_workspace_size<T>(h,p,0,&b,&s);
    q.backend_bytes=std::max(q.backend_bytes,b);q.host_bytes=std::max(q.host_bytes,s);
    q.staging_bytes=congruence_staging_bytes<T>(h,p,a,factor);
    return q;
}
template<class T>
Workspace workspace_size(const DdlaHandle_t& h,const Plan& p)
{
    current(h,p);
    return congruence_workspace_size<T>(h,p.completion,p.a,p.factor);
}

template<DdlaBackend B,class T>
void congruence_run(const DdlaHandle_t& h,const spectral_impl::HermitianPlan& p,T* a,const int* da,
    const T* factor,const int* df,void* work,std::size_t bytes,void* host,std::size_t host_bytes)
{
    if(!p.n)return;
    const char uplo=p.upper?'U':'L';
    if(p.n<=p.nb) {
        if(!p.mr && !p.mc) {
            // The legacy TRSM adapter takes a mutable factor pointer, but its
            // BLAS contract only reads it. No factor copy or broadcast.
            auto* f=const_cast<T*>(factor);
            ddla::herm_complete<B,T>(h,uplo,p.n,a,da[DDLA_LLD_]);
            blas_impl::trsm_run<B,T>(h,'L',uplo,uplo=='U'?'C':'N','N',
                p.n,p.n,T(1),f,df[DDLA_LLD_],a,da[DDLA_LLD_]);
            blas_impl::trsm_run<B,T>(h,'R',uplo,uplo=='U'?'N':'C','N',
                p.n,p.n,T(1),f,df[DDLA_LLD_],a,da[DDLA_LLD_]);
        }
        // PHEGVD proceeds directly to its eigen stage. Canonical completion
        // and success retirement belong only to the standalone PHEGST entry.
        return;
    }
    spectral_impl::pherm_complete_run<B,T>(h,p,a,nullptr,0,work,bytes,host,host_bytes);
    ptrtrs_impl::ptrtrs_run<B,T>(h,'L',uplo,uplo=='U'?'C':'N','N',p.n,p.n,
        factor,df,a,da,work,bytes,host,host_bytes);
    ptrtrs_impl::ptrtrs_run<B,T>(h,'R',uplo,uplo=='U'?'N':'C','N',p.n,p.n,
        factor,df,a,da,work,bytes,host,host_bytes);
}
template<DdlaBackend B,class T>
void run(const DdlaHandle_t& h,const Plan& p,T* a,const T* factor,
    void* backend,std::size_t backend_bytes,void* host,std::size_t host_bytes)
{
    current(h,p);
    if(!p.completion.n)return;
    try {
        congruence_run<B,T>(h,p.completion,a,p.a,factor,p.factor,backend,backend_bytes,host,host_bytes);
        if(p.completion.n<=p.completion.nb) {
            if(!p.completion.mr && !p.completion.mc)
                ddla::herm_complete<B,T>(h,p.completion.upper?'U':'L',p.completion.n,a,p.a[DDLA_LLD_]);
        } else
            spectral_impl::pherm_complete_run<B,T>(h,p.completion,a,nullptr,0,backend,backend_bytes,host,host_bytes);
        drain<B>(h);
    } catch(...) {
        try { drain<B>(h); } catch(...) {}
        throw;
    }
}
#define SIZES(T) \
    template Plan check<T>(const DdlaHandle_t&,DdlaBackend,char,int,T*,const int*,const T*,const int*); \
    template Workspace workspace_size<T>(const DdlaHandle_t&,const Plan&); \
    template void congruence_solve_workspace_size<T>(const DdlaHandle_t&,char,int,const int*,const int*,std::size_t*,std::size_t*); \
    template Workspace congruence_workspace_size<T>(const DdlaHandle_t&,const spectral_impl::HermitianPlan&,const int*,const int*); \
    template std::array<std::size_t,2> congruence_staging_bytes<T>(const DdlaHandle_t&,const spectral_impl::HermitianPlan&,const int*,const int*)
DDLA_FOR_EACH_TYPE(SIZES)
#undef SIZES
#define RUNS(B,T) \
    template void congruence_run<B,T>(const DdlaHandle_t&,const spectral_impl::HermitianPlan&,T*,const int*,const T*,const int*,void*,std::size_t,void*,std::size_t); \
    template void run<B,T>(const DdlaHandle_t&,const Plan&,T*,const T*,void*,std::size_t,void*,std::size_t)
DDLA_FOR_EACH_BACKEND_TYPE(RUNS)
#undef RUNS
} }

namespace ddla {
template<DdlaBackend Backend,class T>
void phegst(const DdlaHandle_t& h,char uplo,int n,T* a,const int* da,const T* factor,const int* df)
{
    const auto p=hegst_impl::check<T>(h,Backend,uplo,n,a,da,factor,df);
    if(!n)return;
    hegst_impl::Workspace q;
    ddla::vector<char> backend(h);std::vector<char> host;
    detail::agree_local(h,"phegst workspace",[&] {
        q=hegst_impl::workspace_size<T>(h,p);
        backend.resize(q.backend_bytes);host.resize(q.host_bytes);
        if(q.staging_bytes[0]||q.staging_bytes[1])h->host_buffers<std::byte>(q.staging_bytes[0],q.staging_bytes[1]);
    });
    hegst_impl::run<Backend,T>(h,p,a,factor,backend.data(),q.backend_bytes,host.data(),q.host_bytes);
}
#define PUBLIC(B,T) template void phegst<B,T>(const DdlaHandle_t&,char,int,T*,const int*,const T*,const int*)
DDLA_FOR_EACH_BACKEND_TYPE(PUBLIC)
#undef PUBLIC
}
