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
#include "QR/pgeqrf_impl.h"
#include "PBLAS/pgeadd_impl.h"
#include "PBLAS/pblas_workspace.h"
#include "BLAS/blas_impl.h"
#include "helpers.h"
#include <limits>
#include <optional>
#include "plan_util.h"
#include "row_share.h"
#include "instantiate.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cstring>
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
namespace {
constexpr int kShapePanelWidth=256;

void compact_descriptor(const DdlaHandle_t& h,const int* source,int m,int n,int* result)
{
    std::copy_n(source,DDLA_DLEN_,result);
    result[DDLA_M_]=m;result[DDLA_N_]=n;
    result[DDLA_LLD_]=std::max(1,num_loc(m,source[DDLA_MB_],h->myprow_,source[DDLA_RSRC_],h->nprows_));
}

int maximum_local_extent(int length,int block,int source,int peers)
{
    int result=0;
    for(int coordinate=0;coordinate<peers;++coordinate)
        result=std::max(result,num_loc(length,block,coordinate,source,peers));
    return result;
}

bool shape_memory_eligible(const DdlaHandle_t& h,int m,int n,const int* d,bool wide)
{
    const int k=std::min(m,n);
    // Bound root gathered windows AND replicated V slots by a block-rounded
    // input share. Keep the user's QR policy; never force the slower inplace QR.
    if(h->nprocs>1 && k/d[DDLA_NB_]<h->nprocs)return false;
    // The compact k-by-k factor must also span the row grid. Keeping A's
    // blocks would otherwise move an entire square R onto one process row.
    if(h->nprows_>1 && k/d[DDLA_MB_]<h->nprows_)return false;
    if(!wide)return true;
    const auto max_rows=[&](int rows) {
        return maximum_local_extent(rows,d[DDLA_MB_],d[DDLA_RSRC_],h->nprows_);
    };
    const auto max_cols=[&](int cols) {
        return maximum_local_extent(cols,d[DDLA_NB_],d[DDLA_CSRC_],h->npcols_);
    };
    // A skinny transposed matrix can occupy fewer process columns. Do not
    // introduce a matrix-sized concentration beyond the input's block rounding.
    return std::size_t(max_rows(n))*max_cols(k)<=std::size_t(max_rows(m))*max_cols(n);
}

std::size_t matrix_elements(const DdlaHandle_t& h,const int* d)
{
    return std::size_t(d[DDLA_LLD_])*num_loc(d[DDLA_N_],d[DDLA_NB_],h->mypcol_,d[DDLA_CSRC_],h->npcols_);
}

std::size_t shape_sum(std::size_t a,std::size_t b)
{
    if(b>std::numeric_limits<std::size_t>::max()-a)
        throw std::length_error("pgesvd: shape workspace size overflow");
    return a+b;
}

template<class T>std::size_t shape_bytes(std::size_t count)
{
    if(count>(std::numeric_limits<std::size_t>::max()-255)/sizeof(T))
        throw std::length_error("pgesvd: shape workspace element count overflow");
    return detail::padded_bytes<T>(count);
}

PgesvdPlan square_child(const DdlaHandle_t& h,const PgesvdPlan& p)
{
    PgesvdPlan c=p;
    c.route=PgesvdRoute::Direct;c.m=c.n=c.k=p.k;c.uplo='U';c.dc.uplo='U';
    c.descA=p.descR;
    if(p.route==PgesvdRoute::WideQr) {
        c.jobu=p.jobvt;c.jobvt=p.jobu;c.descU=c.descVT=p.descR;
    }
    c.brd=pgebrd_geometry(h,p.backend,p.k,p.k,c.descA.data());
    if(c.jobu=='V')c.unm_u=punmbr_geometry(h,p.backend,'Q','L','N',p.k,p.k,p.k,c.descA.data(),c.descU.data());
    if(c.jobvt=='V')c.unm_v=punmbr_geometry(h,p.backend,'P','R','C',p.k,p.k,p.k,c.descA.data(),c.descVT.data());
    return c;
}

template<class T>
void adjoint_workspace(const DdlaHandle_t& h,int m,int n,const int* source,const int* target,
                       std::size_t& backend,std::size_t& host)
{
    const int rows=m>=n?std::min(m,kShapePanelWidth):m;
    const int cols=m>=n?n:std::min(n,kShapePanelWidth);
    std::size_t b=0,cpu=0;
    pgeadd_impl::pgeadd_workspace_size<T>(h,'C','N',true,false,rows,cols,target,&b,&cpu,source,target,true);
    backend=std::max(backend,b);host=std::max(host,cpu);
}

template<class T>
void shape_workspace(const DdlaHandle_t& h,const PgesvdPlan& p,std::size_t* backend,std::size_t* host)
{
    const auto child=square_child(h,p);
    std::size_t child_b=0,child_h=0,apply_b=0,apply_h=0,copy_b=0,copy_h=0;
    pgesvd_workspace_size<T>(h,child,&child_b,&child_h);
    const bool wide=p.route==PgesvdRoute::WideQr;
    const bool apply=wide?p.jobvt=='V':p.jobu=='V';
    if(apply)punmbr_workspace_size<T>(h,wide?p.unm_v:p.unm_u,&apply_b,&apply_h);
    const std::size_t r=shape_bytes<T>(matrix_elements(h,p.descR.data()));
    std::size_t b=std::max(p.qr_backend_bytes,shape_sum(r,child_b));
    std::size_t stage_h=std::max({p.qr_host_bytes,child_h,apply_h});
    if(wide) {
        const std::size_t f=shape_bytes<T>(matrix_elements(h,p.descF.data()));
        adjoint_workspace<T>(h,p.n,p.k,p.descA.data(),p.descF.data(),copy_b,copy_h);
        b=std::max(b,shape_sum(f,std::max(p.qr_backend_bytes,copy_b)));
        stage_h=std::max(stage_h,copy_h);
        copy_b=0;copy_h=0;
        if(p.jobvt=='V')adjoint_workspace<T>(h,p.k,p.n,p.descF.data(),p.descA.data(),copy_b,copy_h);
        b=std::max(b,shape_sum(shape_sum(f,r),copy_b));
        stage_h=std::max(stage_h,copy_h);
        copy_b=0;copy_h=0;
        if(p.jobu=='V')adjoint_workspace<T>(h,p.k,p.k,p.descR.data(),p.descU.data(),copy_b,copy_h);
        if(p.jobvt=='V')adjoint_workspace<T>(h,p.k,p.k,p.descR.data(),p.descVT.data(),copy_b,copy_h);
        const std::size_t factors=shape_sum(p.jobu=='V'?r:0,p.jobvt=='V'?r:0);
        // Retain R's prefix during output adjoints to avoid two extra square copies.
        b=std::max(b,shape_sum(shape_sum(r,factors),std::max(child_b,copy_b)));
        stage_h=std::max(stage_h,copy_h);
    }
    *backend=std::max(b,apply_b);
    const auto* qd=wide?p.descF.data():p.descA.data();
    const std::size_t local_tau=num_loc(p.k,qd[DDLA_NB_],h->mypcol_,qd[DDLA_CSRC_],h->npcols_);
    *host=shape_sum(shape_bytes<T>(local_tau),shape_sum(apply?shape_bytes<T>(p.k):0,stage_h));
}

template<DdlaBackend B,class T>
void zero_active(const DdlaHandle_t& h,int m,int n,T* output,const int* d)
{
    const int rows=num_loc(m,d[DDLA_MB_],h->myprow_,d[DDLA_RSRC_],h->nprows_);
    const int cols=num_loc(n,d[DDLA_NB_],h->mypcol_,d[DDLA_CSRC_],h->npcols_);
    blas_impl::matrix_scal_run<B,T>(h,rows,cols,T(0),output,d[DDLA_LLD_]);
}

template<DdlaBackend B,class T>
void adjoint_panels(const DdlaHandle_t& h,int m,int n,const T* input,const int* di,T* output,const int* dout,
    void* backend,std::size_t bytes,void* host,std::size_t host_bytes)
{
    const bool rows=m>=n;
    const int long_dimension=rows?m:n;
    for(int first=0;first<long_dimension;) {
        const int width=std::min(kShapePanelWidth,long_dimension-first);
        pgeadd_impl::pgeadd_run<B,T>(h,'C','N',rows?width:m,rows?n:width,
            T(1),input,rows?1:first+1,rows?first+1:1,di,
            T(0),nullptr,rows?first+1:1,rows?1:first+1,dout,
            output,rows?first+1:1,rows?1:first+1,dout,backend,bytes,host,host_bytes);
        first+=width;
    }
}

template<class T>
void extract_upper_r(const DdlaHandle_t& h,int k,const T* factor,const int* df,T* r,const int* dr)
{
    const auto count=matrix_elements(h,dr);
    if(count)DDLA_CHECK(ddlaMemset(h,r,0,count*sizeof(T)));
    for(int j=0;j<k;) {
        const int width=std::min({df[DDLA_MB_]-j%df[DDLA_MB_],df[DDLA_NB_]-j%df[DDLA_NB_],k-j});
        if(h->mypcol_==indxg2p(j,df[DDLA_NB_],df[DDLA_CSRC_],h->npcols_)) {
            const int col=num_loc(j,df[DDLA_NB_],h->mypcol_,df[DDLA_CSRC_],h->npcols_);
            const int prefix=num_loc(j,df[DDLA_MB_],h->myprow_,df[DDLA_RSRC_],h->nprows_);
            if(prefix)DDLA_CHECK(ddlaMemcpy2D(h,r+std::size_t(col)*dr[DDLA_LLD_],std::size_t(dr[DDLA_LLD_])*sizeof(T),
                factor+std::size_t(col)*df[DDLA_LLD_],std::size_t(df[DDLA_LLD_])*sizeof(T),std::size_t(prefix)*sizeof(T),width));
            if(h->myprow_==indxg2p(j,df[DDLA_MB_],df[DDLA_RSRC_],h->nprows_))
                DDLA_CHECK(ddlaMemcpyTri(h,'U',width,width,r+prefix+std::size_t(col)*dr[DDLA_LLD_],dr[DDLA_LLD_],
                    factor+prefix+std::size_t(col)*df[DDLA_LLD_],df[DDLA_LLD_],true));
        }
        j+=width;
    }
}

template<DdlaBackend B,class T>
void shape_run(const DdlaHandle_t& h,const PgesvdPlan& p,detail::DcHostTeam* team,T* a,
    typename real_type<T>::type* values,T* u,T* vt,int& info,void* backend,std::size_t bytes,void* host,std::size_t host_bytes)
{
    const bool wide=p.route==PgesvdRoute::WideQr;
    const bool apply=wide?p.jobvt=='V':p.jobu=='V';
    const auto* qd=wide?p.descF.data():p.descA.data();
    const int long_dimension=wide?p.n:p.m;
    const std::size_t local_tau=num_loc(p.k,qd[DDLA_NB_],h->mypcol_,qd[DDLA_CSRC_],h->npcols_);
    char* hw=static_cast<char*>(host);
    T* tau=detail::take<T>(hw,local_tau);
    T* complete_tau=apply?detail::take<T>(hw,p.k):nullptr;
    const std::size_t rest_h=host_bytes-std::size_t(hw-static_cast<char*>(host));
    char* base=static_cast<char*>(backend);
    const std::size_t fbytes=wide?shape_bytes<T>(matrix_elements(h,p.descF.data())):0;
    T* factor=wide?reinterpret_cast<T*>(base):a;
    if(wide)adjoint_panels<B,T>(h,p.n,p.k,a,p.descA.data(),factor,qd,base+fbytes,bytes-fbytes,hw,rest_h);
    qr_impl::pgeqrf_run<B,T>(h,'U',p.qr_inplace,long_dimension,p.k,factor,qd,tau,
        base+fbytes,bytes-fbytes,hw,rest_h);
    if(apply) {
        std::fill_n(complete_tau,p.k,T(0));
        for(int j=0;j<int(local_tau);++j) {
            const int global=indxl2g(j,qd[DDLA_NB_],h->mypcol_,qd[DDLA_CSRC_],h->npcols_);
            if(global<p.k)complete_tau[global]=tau[j];
        }
        // tau lives in HOST memory even for GPU handles; do not pass it to CCL.
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,complete_tau,p.k,detail::mpi_datatype<T>(),MPI_SUM,h->row_comm));
    }
    T* r=reinterpret_cast<T*>(base+fbytes);
    extract_upper_r(h,p.k,factor,qd,r,p.descR.data());
    const std::size_t rbytes=shape_bytes<T>(matrix_elements(h,p.descR.data()));
    if(wide) {
        if(p.jobvt=='V')adjoint_panels<B,T>(h,p.k,p.n,factor,qd,a,p.descA.data(),base+fbytes+rbytes,bytes-fbytes-rbytes,hw,rest_h);
        if(rbytes)DDLA_CHECK(ddlaMemcpy(h,base,r,matrix_elements(h,p.descR.data())*sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
        r=reinterpret_cast<T*>(base);
    }
    const auto child=square_child(h,p);
    char* child_work=base+rbytes;
    T* ur=u;T* vr=vt;
    if(wide) {
        ur=p.jobvt=='V'?detail::take<T>(child_work,matrix_elements(h,p.descR.data())):nullptr;
        vr=p.jobu=='V'?detail::take<T>(child_work,matrix_elements(h,p.descR.data())):nullptr;
    } else if(p.jobu=='V')zero_active<B,T>(h,p.m,p.k,u,p.descU.data());
    const std::size_t child_bytes=bytes-std::size_t(child_work-base);
    pgesvd_run<B,T>(h,child,team,r,values,ur,vr,info,child_work,child_bytes,hw,rest_h);
    if(info)return;
    DDLA_CHECK(ddlaSynchronize(h));
    if(wide) {
        if(p.jobu=='V')adjoint_panels<B,T>(h,p.k,p.k,vr,p.descR.data(),u,p.descU.data(),child_work,child_bytes,hw,rest_h);
        if(p.jobvt=='V') {
            zero_active<B,T>(h,p.k,p.n,vt,p.descVT.data());
            adjoint_panels<B,T>(h,p.k,p.k,ur,p.descR.data(),vt,p.descVT.data(),child_work,child_bytes,hw,rest_h);
        }
    }
    if(apply)punmbr_run<B,T>(h,wide?p.unm_v:p.unm_u,a,p.descA.data(),complete_tau,
        wide?vt:u,wide?p.descVT.data():p.descU.data(),info,backend,bytes,hw,rest_h);
    DDLA_CHECK(ddlaSynchronize(h));
}
} // namespace

template<class T>PgesvdPlan pgesvd_check(const DdlaHandle_t& h,DdlaBackend backend,char jobu,char jobvt,
    int m,int n,const T* A,const int* descA,const typename real_type<T>::type* values,
    const T* U,const int* descU,const T* VT,const int* descVT)
{
    using R=typename real_type<T>::type;
    auto p=detail::agree_check(h,"pgesvd",[&](detail::Agree& agreement) {
        agreement.mode("spectral operation",3121);PgesvdPlan result;
        if(!h||ddlaGetBackend(h)!=backend)throw std::invalid_argument("pgesvd: invalid handle/backend");
        const char* shape=std::getenv("DDLA_SVD_SHAPE");
        const bool shape_qr=shape && std::strcmp(shape,"qr")==0;
        if(shape && *shape && !shape_qr && std::strcmp(shape,"direct")!=0)
            throw std::invalid_argument("pgesvd: DDLA_SVD_SHAPE must be direct or qr");
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
        const int ar=num_loc(m,descA[DDLA_MB_],h->myprow_,descA[DDLA_RSRC_],h->nprows_);
        const int ac=num_loc(n,descA[DDLA_NB_],h->mypcol_,descA[DDLA_CSRC_],h->npcols_);
        const int ur=wantu?num_loc(m,descU[DDLA_MB_],h->myprow_,descU[DDLA_RSRC_],h->nprows_):0;
        const int uc=wantu?num_loc(result.k,descU[DDLA_NB_],h->mypcol_,descU[DDLA_CSRC_],h->npcols_):0;
        const int vr=wantvt?num_loc(result.k,descVT[DDLA_MB_],h->myprow_,descVT[DDLA_RSRC_],h->nprows_):0;
        const int vc=wantvt?num_loc(n,descVT[DDLA_NB_],h->mypcol_,descVT[DDLA_CSRC_],h->npcols_):0;
        if((wantu && detail::matrix_overlaps(A,descA[DDLA_LLD_],ar,ac,U,descU[DDLA_LLD_],ur,uc)) ||
           (wantvt && detail::matrix_overlaps(A,descA[DDLA_LLD_],ar,ac,VT,descVT[DDLA_LLD_],vr,vc)) ||
           (wantu && wantvt && detail::matrix_overlaps(U,descU[DDLA_LLD_],ur,uc,VT,descVT[DDLA_LLD_],vr,vc)))
            throw std::invalid_argument("pgesvd: active A/U/VT storage must be disjoint");
        if(shape_qr && !(backend==DdlaBackend::CPU && h->nprocs==1)) {
            if(m>=n && m/n>=4 && shape_memory_eligible(h,m,n,descA,false))result.route=PgesvdRoute::TallQr;
            else if(n>m && n/m>=4 && shape_memory_eligible(h,m,n,descA,true))result.route=PgesvdRoute::WideQr;
        }
        if(result.route!=PgesvdRoute::Direct) {
            if(backend==DdlaBackend::GPU && h->communication()==DdlaCommunication::CCL && h->communication_group_depth)
                throw std::invalid_argument("pgesvd: close the active CCL group before a shape reduction");
            compact_descriptor(h,descA,result.k,result.k,result.descR.data());
            if(result.route==PgesvdRoute::WideQr)
                compact_descriptor(h,descA,n,result.k,result.descF.data());
            result.qr_inplace=qr_impl::qr_panel_inplace_env();
        }
        if(wantu) {
            result.unm_u=punmbr_geometry(h,backend,'Q','L','N',m,result.k,
                result.route==PgesvdRoute::TallQr?result.k:n,descA,descU);
        }
        if(wantvt) {
            result.unm_v=punmbr_geometry(h,backend,'P','R','C',result.k,n,m,descA,descVT);
        }
        result.row_ld=std::max(1,detail::RowShare(result.k,result.ranks).count(result.rank));
        return result;
    });
    int signature[34]={3121,int(backend),int(p.jobu),int(p.jobvt),m,n,int(sizeof(R)),std::is_same<T,R>::value?1:0,
        h->nprows_,h->npcols_,int(h->major)};
    detail::descriptor_signature(signature+11,p.descA.data());
    if(p.jobu=='V')detail::descriptor_signature(signature+18,p.descU.data());
    if(p.jobvt=='V')detail::descriptor_signature(signature+25,p.descVT.data());
    signature[32]=int(p.route);signature[33]=p.qr_inplace?1:0;
    detail::agree_signature(h,"pgesvd",signature);
    if(!p.k)return p;
    const bool vectors=p.jobu=='V'||p.jobvt=='V';
    if(vectors) {
        // Only dimensions, backend and policy enter this child plan. Its
        // numerical arrays are internal: pgesvd_run carves and passes them.
        p.dc.backend=backend;p.dc.uplo=p.route==PgesvdRoute::Direct?p.uplo:'U';p.dc.n=p.k;p.dc.ranks=p.ranks;p.dc.rank=p.rank;p.dc.ld=p.row_ld;
        pbdsdc_freeze_policy(h,p.dc);
    }
    if(p.route!=PgesvdRoute::Direct) {
        const int* qd=p.route==PgesvdRoute::WideQr?p.descF.data():p.descA.data();
        detail::agree_local(h,"pgesvd QR workspace preparation",[&] {
            // The gathered roots own original A(j,j), j<k. GPU buffer-size
            // queries do not inspect tau; CPU work queries do not write it.
            T* placeholder=const_cast<T*>(A);
            qr_impl::pgeqrf_workspace_size<T>(h,'U',p.qr_inplace,
                p.route==PgesvdRoute::WideQr?p.n:p.m,p.k,placeholder,qd,placeholder,
                &p.qr_backend_bytes,&p.qr_host_bytes);
        });
    }
    return p;
}

// Component 2 (pgesvd_workspace.h).
template<class T>std::array<std::size_t,2> pgesvd_staging_bytes(const DdlaHandle_t& h,const PgesvdPlan& p)
{
    std::array<std::size_t,2> high{};
    if(p.backend!=DdlaBackend::GPU || h->communication()!=DdlaCommunication::HostStagedMPI || !p.k)return high;
    using R=typename real_type<T>::type;
    if(p.route!=PgesvdRoute::Direct) {
        const bool wide=p.route==PgesvdRoute::WideQr,apply=wide?p.jobvt=='V':p.jobu=='V';
        const int* qd=wide?p.descF.data():p.descA.data();
        pblas::staging_max(high,qr_impl::pgeqrf_staging_bytes<T>(h,'U',p.qr_inplace,wide?p.n:p.m,p.k,qd));
        auto adjoint=[&](int m,int n,const int* source,const int* target) {
            const bool rows=m>=n;const int long_dim=rows?m:n;
            for(int first=0;first<long_dim;) {
                const int width=std::min(kShapePanelWidth,long_dim-first);
                pblas::staging_max(high,pgeadd_impl::pgeadd_staging_bytes(h,'C','N',true,false,
                    rows?width:m,rows?n:width,source,rows?1:first+1,rows?first+1:1,
                    target,1,1,target,rows?first+1:1,rows?1:first+1,sizeof(T)));
                first+=width;
            }
        };
        if(wide) {
            adjoint(p.n,p.k,p.descA.data(),qd);
            if(p.jobvt=='V')adjoint(p.k,p.n,qd,p.descA.data());
            if(p.jobu=='V')adjoint(p.k,p.k,p.descR.data(),p.descU.data());
            if(p.jobvt=='V')adjoint(p.k,p.k,p.descR.data(),p.descVT.data());
        }
        const auto child=square_child(h,p);
        pblas::staging_max(high,pgesvd_staging_bytes<T>(h,child));
        if(apply)pblas::staging_max(high,punmbr_staging_bytes<T>(h,wide?p.unm_v:p.unm_u));
        return high;
    }
    pblas::staging_max(high,pgebrd_staging_bytes<T>(h,p.brd));
    if(p.jobu=='V'||p.jobvt=='V')pblas::staging_max(high,pbdsdc_staging_bound<R>(h,p.dc));
    if(p.jobu=='V') {
        pblas::staging_max(high,matrix_impl::scatter_rows_staging_bytes(h,p.k,false,p.m,p.k,p.descU.data(),sizeof(T)));
        pblas::staging_max(high,punmbr_staging_bytes<T>(h,p.unm_u));
    }
    if(p.jobvt=='V') {
        pblas::staging_max(high,matrix_impl::scatter_rows_staging_bytes(h,p.k,true,p.k,p.n,p.descVT.data(),sizeof(T)));
        pblas::staging_max(high,punmbr_staging_bytes<T>(h,p.unm_v));
    }
    return high;
}
#define STAGING_SIZE(T) template std::array<std::size_t,2> pgesvd_staging_bytes<T>(const DdlaHandle_t&,const PgesvdPlan&)
DDLA_FOR_EACH_TYPE(STAGING_SIZE)
#undef STAGING_SIZE

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
    if(p.route!=PgesvdRoute::Direct) {
        shape_workspace<T>(h,p,backend_bytes,host_bytes);
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
    // A CCL group can hold dependent submissions; reject before the catch
    // path tries to drain any queue. Raw callers retain the checked precondition.
    if constexpr(Backend==DdlaBackend::GPU) {
        if(p.k && p.route!=PgesvdRoute::Direct && handle->communication()==DdlaCommunication::CCL && handle->communication_group_depth)
            throw std::invalid_argument("pgesvd: close the active CCL group before a shape reduction");
    }
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
        if(p.route!=PgesvdRoute::Direct) {
            shape_run<Backend,T>(handle,p,team,A,s,U,VT,info,
                backend_work,backend_bytes,host_work,host_bytes);
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
        const auto slots=spectral_impl::pgesvd_staging_bytes<T>(handle,p);
        if(slots[0]||slots[1])handle->host_buffers<std::byte>(slots[0],slots[1]);
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
