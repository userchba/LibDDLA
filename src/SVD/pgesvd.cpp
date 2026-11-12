#include "matrix_transfer.h"
// ---------------------------------------------------------------------------
// Distributed singular value decomposition A = U diag(s) V^H (ScaLAPACK
// PxGESVD semantics: economy-size U (m x k) and VT (k x n), k = min(m,n),
// singular values in descending order).
//
// Three stages:
//   1. pgebrd: A = Q B P^H, with real upper/lower bidiagonal B.
//   2. pbdsdc on CPU and GPU. Real U_s and V_s and their
//      merge scratch are row-distributed over ALL ranks, O(k^2/P) memory.
//      Scalar vectors are replicated; coefficient broadcasts have bounded
//      width. Slabs are scattered into the output descriptors and released
//      before back-transforms. Both bidiagonal factors are needed internally.
//      Without vectors, every rank runs dqds with O(k) storage.
//   3. punmbr: U = Q [U_s; 0], VT = [VT_s 0] P^H.
//
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>

#include "ddla_connector.h"
#include "ddla_stream.h"
#include "ddla_desc.h"
#include "mpi_datatype.h"
#include "pgesvd_workspace.h"
#include "gesvd_local.h"
#include "check_agree.h"
#include <optional>
#include "plan_util.h"
#include "row_share.h"
#include "instantiate.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <complex>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <type_traits>
#include <vector>

// Host ?BDSQR through the Fortran ABI (src/pstedc.cpp's style: the
// hidden trailing CHARACTER lengths are omitted, correct on this project's
// gfortran + x86-64 SysV toolchain).
extern "C" {
void sbdsqr_(const char* uplo, const int* n, const int* ncvt, const int* nru, const int* ncc,
             float* d, float* e, float* vt, const int* ldvt, float* u, const int* ldu,
             float* c, const int* ldc, float* work, int* info);
void dbdsqr_(const char* uplo, const int* n, const int* ncvt, const int* nru, const int* ncc,
             double* d, double* e, double* vt, const int* ldvt, double* u, const int* ldu,
             double* c, const int* ldc, double* work, int* info);
}

namespace ddla {

namespace {

// With no vectors, BDSQR selects dqds and needs only O(k) work.
void bdsqr_values(char uplo, int n, float* d, float* e, float* work, int& info)
{
    const int zero = 0, one = 1;
    float dummy = 0.0f;
    sbdsqr_(&uplo, &n, &zero, &zero, &zero, d, e, &dummy, &one,
            &dummy, &one, &dummy, &one, work, &info);
}

void bdsqr_values(char uplo, int n, double* d, double* e, double* work, int& info)
{
    const int zero = 0, one = 1;
    double dummy = 0.0;
    dbdsqr_(&uplo, &n, &zero, &zero, &zero, d, e, &dummy, &one,
            &dummy, &one, &dummy, &one, work, &info);
}

} // namespace

namespace spectral_impl {
template<class T>PgesvdPlan pgesvd_check(const DdlaHandle_t& h,DdlaBackend backend,char jobu,char jobvt,
    int m,int n,const T* A,const int* descA,const typename real_type<T>::type* values,
    const T* U,const int* descU,const T* VT,const int* descVT)
{
    using R=typename real_type<T>::type;
    auto p=detail::agree_check(h,"pgesvd",[&](detail::Agree& agreement) {
        agreement.mode("spectral operation",3121);PgesvdPlan result;
        if(!h||ddlaGetBackend(h)!=backend)throw std::invalid_argument("pgesvd: invalid handle/backend");
        result.backend=backend;result.m=m;result.n=n;result.k=std::min(m,n);
        result.jobu=char(std::toupper(static_cast<unsigned char>(jobu)));
        result.jobvt=char(std::toupper(static_cast<unsigned char>(jobvt)));
        if((result.jobu!='V'&&result.jobu!='N')||(result.jobvt!='V'&&result.jobvt!='N'))
            throw std::invalid_argument("pgesvd: jobu and jobvt must be V or N");
        const bool wantu=result.jobu=='V',wantvt=result.jobvt=='V';
        check_desc(descA,h);
        if(m<0||n<0||m>descA[DDLA_M_]||n>descA[DDLA_N_])
            throw std::invalid_argument("pgesvd: m/n outside A descriptor dimensions");
        std::copy_n(descA,DDLA_DLEN_,result.descA.data());
        if(wantu) {
            check_desc(descU,h);
            if(m>descU[DDLA_M_]||result.k>descU[DDLA_N_]||descU[DDLA_MB_]!=descA[DDLA_MB_]||descU[DDLA_RSRC_]!=descA[DDLA_RSRC_])
                throw std::invalid_argument("pgesvd: invalid U dimensions or row distribution");
            std::copy_n(descU,DDLA_DLEN_,result.descU.data());
        }
        if(wantvt) {
            check_desc(descVT,h);
            if(result.k>descVT[DDLA_M_]||n>descVT[DDLA_N_]||descVT[DDLA_NB_]!=descA[DDLA_NB_]||descVT[DDLA_CSRC_]!=descA[DDLA_CSRC_])
                throw std::invalid_argument("pgesvd: invalid VT dimensions or column distribution");
            std::copy_n(descVT,DDLA_DLEN_,result.descVT.data());
        }
        if(result.k&&(!values||(wantu&&!U)||(wantvt&&!VT)))
            throw std::invalid_argument("pgesvd: null active s/U/VT pointer");
        result.ranks=h->nprocs;result.rank=h->myid;result.uplo=m>=n?'U':'L';
        result.brd=pgebrd_geometry(h,backend,m,n,descA);
        if(result.brd.mloc>0&&result.brd.nloc>0&&!A)throw std::invalid_argument("pgesvd: null active A");
        if(!result.k)return result;
        if(wantu) {
            result.unm_u=punmbr_geometry(h,backend,'Q','L','N',m,result.k,n,descA,descU);
        }
        if(wantvt) {
            result.unm_v=punmbr_geometry(h,backend,'P','R','C',result.k,n,m,descA,descVT);
        }
        result.row_ld=std::max(1,detail::RowShare(result.k,result.ranks).count(result.rank));
        return result;
    });
    int signature[32]={3121,int(backend),int(p.jobu),int(p.jobvt),m,n,int(sizeof(R)),std::is_same<T,R>::value?1:0,
        h->nprows_,h->npcols_,int(h->major)};
    detail::descriptor_signature(signature+11,p.descA.data());
    if(p.jobu=='V')detail::descriptor_signature(signature+18,p.descU.data());
    if(p.jobvt=='V')detail::descriptor_signature(signature+25,p.descVT.data());
    detail::agree_signature(h,"pgesvd",signature);
    if(!p.k)return p;
    const bool vectors=p.jobu=='V'||p.jobvt=='V';
    if(vectors) {
        // Only dimensions, backend and policy enter this child plan. Its
        // numerical arrays are internal: pgesvd_run carves and passes them.
        p.dc.backend=backend;p.dc.uplo=p.uplo;p.dc.n=p.k;p.dc.ranks=p.ranks;p.dc.rank=p.rank;p.dc.ld=p.row_ld;
        pbdsdc_freeze_policy(h,p.dc);
    }
    return p;
}

// Component 2 (pgesvd_workspace.h).
template<class T>
void pgesvd_workspace_size(const DdlaHandle_t& h,const PgesvdPlan& p,std::size_t* backend_bytes,std::size_t* host_bytes)
{
    using R=typename real_type<T>::type;using detail::padded_bytes;
    *backend_bytes=0;*host_bytes=0;
    if(!p.k)return;
    // The GPU row-D&C driver is faster than vendor GESVD for full vectors.
    if (h->nprocs==1 && p.backend==DdlaBackend::CPU) {
        local_gesvd_workspace_size<T>(h,p.jobu,p.jobvt,p.m,p.n,backend_bytes,host_bytes);
        return;
    }
    const bool wantu=p.jobu=='V',wantvt=p.jobvt=='V',vectors=wantu||wantvt;
    const std::size_t k=p.k;
    std::size_t brd=0,unm_u=0,unm_u_host=0,unm_v=0,unm_v_host=0;
    pgebrd_workspace_size<T>(h,p.brd,&brd);
    if(wantu)punmbr_workspace_size<T>(h,p.unm_u,&unm_u,&unm_u_host);
    if(wantvt)punmbr_workspace_size<T>(h,p.unm_v,&unm_v,&unm_v_host);
    // the stages run one after the other: each starts both regions afresh
    std::size_t backend=std::max({brd,unm_u,unm_v}),stage=std::max(unm_u_host,unm_v_host);
    if(vectors) {
        std::size_t dc=0,dc_host=0,scatter=0,scatter_host=0;
        pbdsdc_workspace_size<R>(h,p.dc,&dc,&dc_host);
        matrix_impl::scatter_rows_workspace_size<T>(h,p.k,&scatter,&scatter_host);
        backend=std::max(backend,2*padded_bytes<R>(std::size_t(p.row_ld)*k)+std::max(dc,scatter));
        stage=std::max({stage,dc_host,scatter_host});
    } else {
        stage=std::max(stage,padded_bytes<R>(4*k+4));
    }
    *backend_bytes=backend;
    *host_bytes=padded_bytes<R>(k)+padded_bytes<R>(std::max<std::size_t>(1,k-1))+2*padded_bytes<T>(k)+stage;
}

// Component 3.
template<DdlaBackend Backend,class T>
void pgesvd_run(const DdlaHandle_t& handle,const PgesvdPlan& p,detail::DcHostTeam* team,T* A,
    typename real_type<T>::type* s,T* U,T* VT,int& info,
    void* backend_work,std::size_t backend_bytes,void* host_work,std::size_t host_bytes)
{
    try {
        using rt = typename real_type<T>::type;
        static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                      "GPU pgesvd is not available in this LibDDLA build");
        info=0;const int m=p.m,n=p.n,k=p.k;if(!k)return;
        if (handle->nprocs==1 && Backend==DdlaBackend::CPU) {
            local_gesvd_run<Backend,T>(handle,p.jobu,p.jobvt,m,n,A,p.descA[DDLA_LLD_],s,
                U,p.jobu=='V'?p.descU[DDLA_LLD_]:1,VT,p.jobvt=='V'?p.descVT[DDLA_LLD_]:1,
                info,backend_work,backend_bytes,host_work,host_bytes);
            return;
        }
        const bool wantu=p.jobu=='V',wantvt=p.jobvt=='V';
        const int* descA=p.descA.data();const int* descU=p.descU.data();const int* descVT=p.descVT.data();
        MPI_Comm comm=handle->comm;
        char* const hbase=static_cast<char*>(host_work);
        char* hw=hbase;
        const auto host_rest=[&](const char* at){return host_bytes-static_cast<std::size_t>(at-hbase);};
        // ---- 1. bidiagonalization ----------------------------------------------
        // d, e, tauq and taup live through every stage; the stages share the rest.
        rt* const d=detail::take<rt>(hw,k);rt* const e=detail::take<rt>(hw,std::max(1,k-1));
        T* const tauq=detail::take<T>(hw,k);T* const taup=detail::take<T>(hw,k);
        assert(static_cast<std::size_t>(hw-hbase)<=host_bytes);
        char* const stage=hw;
        // PGEBRD writes all active e/tau entries, including the terminal zero tau.
        pgebrd_run<Backend,T>(handle,p.brd,A,descA,d,e,tauq,taup,info,backend_work,backend_bytes);

        // ---- 2. bidiagonal SVD -------------------------------------------------
        const char uplo=p.uplo;
        const bool vectors = wantu || wantvt;
        // Distribute quadratic factors over the whole communicator.
        if (vectors) {
            assert(team);
            char* const bbase=static_cast<char*>(backend_work);
            char* bw=bbase;
            const std::size_t count=static_cast<std::size_t>(p.row_ld)*k;
            rt* const u=detail::take<rt>(bw,count);rt* const v=detail::take<rt>(bw,count);
            assert(static_cast<std::size_t>(bw-bbase)<=backend_bytes);
            const std::size_t backend_rest=backend_bytes-static_cast<std::size_t>(bw-bbase);
            pbdsdc_run<Backend,rt>(handle,p.dc,*team,d,e,s,u,v,info,bw,backend_rest,stage,host_rest(stage));
            if(info)return;
            if(wantu)matrix_impl::scatter_rows_run<Backend,T>(handle,k,u,p.row_ld,false,m,k,U,descU,
                bw,backend_rest,stage,host_rest(stage));
            if(wantvt)matrix_impl::scatter_rows_run<Backend,T>(handle,k,v,p.row_ld,true,k,n,VT,descVT,
                bw,backend_rest,stage,host_rest(stage));
            // Both factors and the child's scratch expire before either transform;
            // the same backend region is reused, never added to peak storage.
            if(wantu) {
                punmbr_run<Backend,T>(handle,p.unm_u,A,descA,tauq,U,descU,info,
                    backend_work,backend_bytes,stage,host_rest(stage));
                if(info)return;
            }
            if(wantvt)punmbr_run<Backend,T>(handle,p.unm_v,A,descA,taup,VT,descVT,info,
                backend_work,backend_bytes,stage,host_rest(stage));
            return;
        }

        char* w=stage;
        rt* const work=detail::take<rt>(w,4*std::size_t(k)+4);
        assert(static_cast<std::size_t>(w-hbase)<=host_bytes);
        int binfo=0,ginfo=0;
        std::copy_n(d,k,s);
        bdsqr_values(uplo,k,s,e,work,binfo);
        MPI_CHECK(MPI_Allreduce(&binfo,&ginfo,1,MPI_INT,MPI_MAX,comm));
        if(ginfo==0) {
            // Reuse dqds work to retain the numerical heterogeneity check.
            rt* const smin=work;rt* const smax=work+k;
            std::copy_n(s,k,smin);std::copy_n(s,k,smax);
            MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,smin,k,detail::mpi_datatype<rt>(),MPI_MIN,comm));
            MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,smax,k,detail::mpi_datatype<rt>(),MPI_MAX,comm));
            for(int i=0;i<k;++i)if(smin[i]!=smax[i])ginfo=k+1;
        }
        info=ginfo;
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(handle) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(handle); } catch (...) {}
        throw;
    }
}

} // namespace spectral_impl

template <DdlaBackend Backend, typename T>
void pgesvd(
    const DdlaHandle_t& handle,
    char jobu, char jobvt,
    int m, int n,
    T* A, const int* descA,
    typename real_type<T>::type* s,
    T* U, const int* descU,
    T* VT, const int* descVT,
    int& info)
{
    info=0;const auto p=spectral_impl::pgesvd_check<T>(handle,Backend,jobu,jobvt,m,n,A,descA,s,U,descU,VT,descVT);
    if(!p.k)return;
    std::size_t backend_bytes=0,host_bytes=0;
    vector<char> backend_work(handle);std::vector<char> host_work;std::optional<detail::DcHostTeam> team;
    std::exception_ptr failure;
    try{spectral_impl::pgesvd_workspace_size<T>(handle,p,&backend_bytes,&host_bytes);
        backend_work.resize(backend_bytes);host_work.resize(host_bytes);
        if(handle->nprocs!=1 || Backend==DdlaBackend::GPU)
            if(const int threads=spectral_impl::pgesvd_team_threads(p))team.emplace(threads);}
    catch(...){failure=std::current_exception();}
    detail::agree_failure(handle->comm,failure,"pgesvd: workspace preparation");
    spectral_impl::pgesvd_run<Backend,T>(handle,p,team?&*team:nullptr,A,s,U,VT,info,
        backend_work.data(),backend_bytes,host_work.data(),host_bytes);
}

#define INSTANTIATE_PGESVD(BACKEND, TYPE)                                          \
    template void pgesvd<BACKEND, TYPE>(                                           \
        const DdlaHandle_t&, char, char, int, int,     \
        TYPE*, const int*, real_type<TYPE>::type*, TYPE*, const int*, TYPE*,       \
        const int*, int&); \
    template void spectral_impl::pgesvd_run<BACKEND,TYPE>(const DdlaHandle_t&,const spectral_impl::PgesvdPlan&, \
        detail::DcHostTeam*,TYPE*,real_type<TYPE>::type*,TYPE*,TYPE*,int&,void*,std::size_t,void*,std::size_t)

DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_PGESVD)

#undef INSTANTIATE_PGESVD
#define INSTANTIATE_PGESVD_CHECK(T) template spectral_impl::PgesvdPlan spectral_impl::pgesvd_check<T>(const DdlaHandle_t&,DdlaBackend,char,char,int,int,const T*,const int*,const real_type<T>::type*,const T*,const int*,const T*,const int*); \
    template void spectral_impl::pgesvd_workspace_size<T>(const DdlaHandle_t&,const spectral_impl::PgesvdPlan&,std::size_t*,std::size_t*)
DDLA_FOR_EACH_TYPE(INSTANTIATE_PGESVD_CHECK)
#undef INSTANTIATE_PGESVD_CHECK

} // namespace ddla
