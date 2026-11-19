// Experimental shape-route validation. All matrix-sized test buffers are local.
#include "api_grid_test_common.h"
#include "workspace_test_common.h"
#include "SVD/pgesvd_workspace.h"
#include "ddla_stream.h"
#include "mpi_datatype.h"
#include <ddla/ddla_vector.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>
#include <cstring>
#include <limits>
#include <optional>

namespace {
using namespace ddla;
using namespace api_grid_test;
using namespace ddla::spectral_impl;
std::string transport="default";
bool quick=false;

template<class T>T conjugate(T value) {
    if constexpr(std::is_same<T,typename real_type<T>::type>::value)return value;
    else return std::conj(value);
}
template<class T>T make_scalar(double re,double im=0) {
    if constexpr(std::is_same<T,typename real_type<T>::type>::value)return T(re);
    else return T(re,im);
}
template<class T>T basis(int count,int row,int column,int pass,bool right) {
    using R=typename real_type<T>::type;
    const int mode=(column+(right?2:1)*pass)%count;
    const double pi=std::acos(-1.);
    const R value=R(std::sqrt((mode?2.:1.)/count)*std::cos(pi*(row+.5)*mode/count));
    if constexpr(std::is_same<T,R>::value)return value;
    else {
        const double angle=(right?.27:.39)*(row+1)*(pass+1);
        return value*make_scalar<T>(std::cos(angle),std::sin(angle));
    }
}
template<class R>R scale_for(int kind,int pass) {
    if(kind==4)return std::sqrt(std::numeric_limits<R>::max())/R(32);
    if(kind==5)return std::sqrt(std::numeric_limits<R>::min())*R(32);
    return pass?R(1.125):R(1);
}
template<class R>R weight(int kind,int k,int column) {
    if(kind==1)return column<k/2?R(1)/R(column+1):R(0);
    if(kind==2)return R(1);
    if(kind==3)return R(0);
    return R(1)/R(column+1);
}
template<class T>T entry(int m,int n,int i,int j,int kind,int pass) {
    using R=typename real_type<T>::type;
    T value=0;const int k=std::min(m,n);
    for(int l=0;l<k;++l)value+=basis<T>(m,i,l,pass,false)*weight<R>(kind,k,l)*conjugate(basis<T>(n,j,l,pass,true));
    return value;
}
void describe(DdlaHandle_t h,int m,int n,int mb,int nb,int rs,int cs,int* d) {
    DDLA_CHECK(ddlaDescInit(d,h,m+2,n+3,mb,nb,rs,cs));
    d[DDLA_LLD_]+=2+h->myid;
}
template<class T,class F>std::vector<T> local(DdlaHandle_t h,const int* d,int m,int n,F&& fill) {
    std::vector<T> result(std::max<std::size_t>(1,local_size(h,d)),make_scalar<T>(-717,23));
    const int nr=num_loc(d[DDLA_M_],d[DDLA_MB_],h->myprow_,d[DDLA_RSRC_],h->nprows_);
    const int nc=num_loc(d[DDLA_N_],d[DDLA_NB_],h->mypcol_,d[DDLA_CSRC_],h->npcols_);
    for(int j=0;j<nc;++j) {
        const int gj=indxl2g(j,d[DDLA_NB_],h->mypcol_,d[DDLA_CSRC_],h->npcols_);
        for(int i=0;i<nr;++i) {
            const int gi=indxl2g(i,d[DDLA_MB_],h->myprow_,d[DDLA_RSRC_],h->nprows_);
            if(gi<m&&gj<n)result[i+std::size_t(j)*d[DDLA_LLD_]]=fill(gi,gj);
        }
    }
    return result;
}
template<class T>bool same(const std::vector<T>& a,const std::vector<T>& b) {
    return a.size()==b.size()&&(!a.size()||!std::memcmp(a.data(),b.data(),a.size()*sizeof(T)));
}
template<class T>bool padding(DdlaHandle_t h,const std::vector<T>& x,const std::vector<T>& initial,const int* d,int m,int n) {
    const int rows=num_loc(d[DDLA_M_],d[DDLA_MB_],h->myprow_,d[DDLA_RSRC_],h->nprows_);
    const int cols=num_loc(d[DDLA_N_],d[DDLA_NB_],h->mypcol_,d[DDLA_CSRC_],h->npcols_);
    if(!cols)return same(x,initial);
    for(int j=0;j<cols;++j) {
        const int gj=indxl2g(j,d[DDLA_NB_],h->mypcol_,d[DDLA_CSRC_],h->npcols_);
        for(int i=0;i<d[DDLA_LLD_];++i) {
            const int gi=i<rows?indxl2g(i,d[DDLA_MB_],h->myprow_,d[DDLA_RSRC_],h->nprows_):d[DDLA_M_];
            const auto at=i+std::size_t(j)*d[DDLA_LLD_];
            if((gi>=m||gj>=n)&&x[at]!=initial[at])return false;
        }
    }
    return true;
}
struct Scratch {
    static constexpr std::size_t guard=256;
    DdlaHandle_t h;std::size_t bytes,host_bytes;ddla::vector<char> device;std::vector<char> cpu;
    Scratch(DdlaHandle_t handle,std::size_t b,std::size_t hb):h(handle),bytes(b),host_bytes(hb),device(h,b+2*guard),cpu(hb+2*guard,char(0x5a)) {
        DDLA_CHECK(ddlaMemset(h,device.data(),0x5a,device.size()));
    }
    void* backend(){return device.data()+guard;}
    void* host(){return cpu.data()+guard;}
    void verify() {
        std::array<char,2*guard> end{};
        DDLA_CHECK(ddlaMemcpy(h,end.data(),device.data(),guard));
        DDLA_CHECK(ddlaMemcpy(h,end.data()+guard,device.data()+guard+bytes,guard));
        DDLA_CHECK(ddlaSynchronize(h));
        bool okay=true;
        for(std::size_t i=0;i<guard;++i)okay=okay&&end[i]==char(0x5a)&&end[guard+i]==char(0x5a)&&cpu[i]==char(0x5a)&&cpu[guard+host_bytes+i]==char(0x5a);
        require_close(h,"shape exact scratch guards",okay?0:1,0);
    }
};
template<class S>bool no_allocation(const S& s) {
    return !s.ddla_alloc&&!s.ddla_free&&!s.pinned_alloc&&!s.raw_gpu_alloc&&!s.host_new&&!s.vendor_query&&!s.resource_setup;
}
template<class T>std::vector<T> vector_output(DdlaHandle_t h,const std::vector<T>& x,const int* d,int length,int index,bool row) {
    std::vector<T> result(length,T(0));
    if(row) {
        if(h->myprow_==indxg2p(index,d[DDLA_MB_],d[DDLA_RSRC_],h->nprows_)) {
            const int i=indxg2l(index,d[DDLA_MB_],h->nprows_);
            const int nc=num_loc(length,d[DDLA_NB_],h->mypcol_,d[DDLA_CSRC_],h->npcols_);
            for(int j=0;j<nc;++j) {
                const int g=indxl2g(j,d[DDLA_NB_],h->mypcol_,d[DDLA_CSRC_],h->npcols_);
                result[g]=x[i+std::size_t(j)*d[DDLA_LLD_]];
            }
        }
    } else if(h->mypcol_==indxg2p(index,d[DDLA_NB_],d[DDLA_CSRC_],h->npcols_)) {
        const int j=indxg2l(index,d[DDLA_NB_],h->npcols_);
        const int nr=num_loc(length,d[DDLA_MB_],h->myprow_,d[DDLA_RSRC_],h->nprows_);
        for(int i=0;i<nr;++i) {
            const int g=indxl2g(i,d[DDLA_MB_],h->myprow_,d[DDLA_RSRC_],h->nprows_);
            result[g]=x[i+std::size_t(j)*d[DDLA_LLD_]];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE,result.data(),length,detail::mpi_datatype<T>(),MPI_SUM,h->comm);
    return result;
}
// Gather one output vector at a time, never a full global singular-vector matrix.
template<class T>
void numerical(DdlaHandle_t h,int m,int n,int kind,int pass,const std::vector<typename real_type<T>::type>& s,
    const std::vector<T>& u,const int* du,const std::vector<T>& vt,const int* dv,const int* da,const std::vector<T>& original,bool wantu,bool wantv) {
    using R=typename real_type<T>::type;const int k=std::min(m,n);
    const R scale=scale_for<R>(kind,pass);
    const double tolerance=150*std::max(m,n)*double(std::numeric_limits<R>::epsilon());
    double error=0;bool sorted=true;
    for(int l=0;l<k;++l) {
        error=std::max(error,std::abs(double(s[l]/scale-weight<R>(kind,k,l))));
        sorted=sorted&&std::isfinite(s[l])&&s[l]>=0&&(!l||s[l-1]>=s[l]);
    }
    require_close(h,"shape known singular values",error,tolerance);
    require_close(h,"shape sorted finite values",sorted?0:1,0);
    auto reconstructed=local<T>(h,da,m,n,[](int,int){return T(0);});
    const int nr=num_loc(m,da[DDLA_MB_],h->myprow_,da[DDLA_RSRC_],h->nprows_);
    const int nc=num_loc(n,da[DDLA_NB_],h->mypcol_,da[DDLA_CSRC_],h->npcols_);
    for(int l=0;l<k;++l) {
        auto left=wantu?vector_output(h,u,du,m,l,false):std::vector<T>{};
        auto right=wantv?vector_output(h,vt,dv,n,l,true):std::vector<T>{};
        if(wantu&&wantv) {
            const int nr=num_loc(m,da[DDLA_MB_],h->myprow_,da[DDLA_RSRC_],h->nprows_);
            const int nc=num_loc(n,da[DDLA_NB_],h->mypcol_,da[DDLA_CSRC_],h->npcols_);
            for(int j=0;j<nc;++j)for(int i=0;i<nr;++i) {
                const int gi=indxl2g(i,da[DDLA_MB_],h->myprow_,da[DDLA_RSRC_],h->nprows_);
                const int gj=indxl2g(j,da[DDLA_NB_],h->mypcol_,da[DDLA_CSRC_],h->npcols_);
                reconstructed[i+std::size_t(j)*da[DDLA_LLD_]]+=left[gi]*(s[l]/scale)*right[gj];
            }
        }
        for(int j=0;j<k;++j) {
            T inner=0;
            if(wantu && h->mypcol_==indxg2p(j,du[DDLA_NB_],du[DDLA_CSRC_],h->npcols_)) {
                const int jc=indxg2l(j,du[DDLA_NB_],h->npcols_);
                const int nr=num_loc(m,du[DDLA_MB_],h->myprow_,du[DDLA_RSRC_],h->nprows_);
                for(int i=0;i<nr;++i)inner+=conjugate(left[indxl2g(i,du[DDLA_MB_],h->myprow_,du[DDLA_RSRC_],h->nprows_)])*u[i+std::size_t(jc)*du[DDLA_LLD_]];
            }
            MPI_Allreduce(MPI_IN_PLACE,&inner,1,detail::mpi_datatype<T>(),MPI_SUM,h->comm);
            if(wantu)require_close(h,"shape U orthogonality",std::abs(inner-T(j==l)),tolerance);
            inner=0;
            if(wantv && h->myprow_==indxg2p(j,dv[DDLA_MB_],dv[DDLA_RSRC_],h->nprows_)) {
                const int jr=indxg2l(j,dv[DDLA_MB_],h->nprows_);
                const int nc=num_loc(n,dv[DDLA_NB_],h->mypcol_,dv[DDLA_CSRC_],h->npcols_);
                for(int c=0;c<nc;++c)inner+=vt[jr+std::size_t(c)*dv[DDLA_LLD_]]*conjugate(right[indxl2g(c,dv[DDLA_NB_],h->mypcol_,dv[DDLA_CSRC_],h->npcols_)]);
            }
            MPI_Allreduce(MPI_IN_PLACE,&inner,1,detail::mpi_datatype<T>(),MPI_SUM,h->comm);
            if(wantv)require_close(h,"shape VT orthogonality",std::abs(inner-T(j==l)),tolerance);
        }
        // Verify both one-sided equations using scaled A; no large/small square.
        if(wantu) {
            std::vector<T> ax(n,T(0)),normal(m,T(0));
            for(int j=0;j<nc;++j)for(int i=0;i<nr;++i) {
                const int gi=indxl2g(i,da[DDLA_MB_],h->myprow_,da[DDLA_RSRC_],h->nprows_);
                const int gj=indxl2g(j,da[DDLA_NB_],h->mypcol_,da[DDLA_CSRC_],h->npcols_);
                ax[gj]+=conjugate(original[i+std::size_t(j)*da[DDLA_LLD_]]/scale)*left[gi];
            }
            MPI_Allreduce(MPI_IN_PLACE,ax.data(),n,detail::mpi_datatype<T>(),MPI_SUM,h->comm);
            for(int j=0;j<nc;++j)for(int i=0;i<nr;++i) {
                const int gi=indxl2g(i,da[DDLA_MB_],h->myprow_,da[DDLA_RSRC_],h->nprows_);
                const int gj=indxl2g(j,da[DDLA_NB_],h->mypcol_,da[DDLA_CSRC_],h->npcols_);
                normal[gi]+=(original[i+std::size_t(j)*da[DDLA_LLD_]]/scale)*ax[gj];
            }
            MPI_Allreduce(MPI_IN_PLACE,normal.data(),m,detail::mpi_datatype<T>(),MPI_SUM,h->comm);
            double residual=0;const R lambda=(s[l]/scale)*(s[l]/scale);
            for(int i=0;i<m;++i)residual=std::max(residual,double(std::abs(normal[i]-lambda*left[i])));
            require_close(h,"shape left singular equation",residual,tolerance);
        }
        if(wantv) {
            std::vector<T> ax(m,T(0)),normal(n,T(0));
            for(int j=0;j<nc;++j)for(int i=0;i<nr;++i) {
                const int gi=indxl2g(i,da[DDLA_MB_],h->myprow_,da[DDLA_RSRC_],h->nprows_);
                const int gj=indxl2g(j,da[DDLA_NB_],h->mypcol_,da[DDLA_CSRC_],h->npcols_);
                ax[gi]+=(original[i+std::size_t(j)*da[DDLA_LLD_]]/scale)*conjugate(right[gj]);
            }
            MPI_Allreduce(MPI_IN_PLACE,ax.data(),m,detail::mpi_datatype<T>(),MPI_SUM,h->comm);
            for(int j=0;j<nc;++j)for(int i=0;i<nr;++i) {
                const int gi=indxl2g(i,da[DDLA_MB_],h->myprow_,da[DDLA_RSRC_],h->nprows_);
                const int gj=indxl2g(j,da[DDLA_NB_],h->mypcol_,da[DDLA_CSRC_],h->npcols_);
                normal[gj]+=conjugate(original[i+std::size_t(j)*da[DDLA_LLD_]]/scale)*ax[gi];
            }
            MPI_Allreduce(MPI_IN_PLACE,normal.data(),n,detail::mpi_datatype<T>(),MPI_SUM,h->comm);
            double residual=0;const R lambda=(s[l]/scale)*(s[l]/scale);
            for(int j=0;j<n;++j)residual=std::max(residual,double(std::abs(normal[j]-lambda*conjugate(right[j]))));
            require_close(h,"shape right singular equation",residual,tolerance);
        }
    }
    if(wantu&&wantv) {
        double residual=0;
        const int nr=num_loc(m,da[DDLA_MB_],h->myprow_,da[DDLA_RSRC_],h->nprows_);
        const int nc=num_loc(n,da[DDLA_NB_],h->mypcol_,da[DDLA_CSRC_],h->npcols_);
        for(int j=0;j<nc;++j)for(int i=0;i<nr;++i)
            residual=std::max(residual,double(std::abs(reconstructed[i+std::size_t(j)*da[DDLA_LLD_]]-original[i+std::size_t(j)*da[DDLA_LLD_]]/scale)));
        require_close(h,"shape full reconstruction",residual,tolerance);
    }
}

template<DdlaBackend B,class T>
void exercise(DdlaHandle_t h,int m,int n,char ju,char jv,int kind,bool rectangular=true) {
    using R=typename real_type<T>::type;const int k=std::min(m,n);
    const int mb=rectangular?3:4,nb=rectangular?2:4,rs=h->nprows_-1,cs=h->npcols_-1;
    int da[DDLA_DLEN_],du[DDLA_DLEN_],dv[DDLA_DLEN_];
    describe(h,m,n,mb,nb,rs,cs,da);describe(h,m,k,mb,5,rs,0,du);describe(h,k,n,4,nb,0,cs,dv);
    auto initial=local<T>(h,da,m,n,[&](int i,int j){return scale_for<R>(kind,0)*entry<T>(m,n,i,j,kind,0);});
    auto initial_u=local<T>(h,du,0,0,[](int,int){return T(0);});
    auto initial_v=local<T>(h,dv,0,0,[](int,int){return T(0);});
    DeviceBuffer<T> a(h,initial.size()),u(h,initial_u.size()),v(h,initial_v.size());
    upload(h,a.ptr,initial);upload(h,u.ptr,initial_u);upload(h,v.ptr,initial_v);
    std::vector<R> values(k+2,R(-919));
    const auto p=pgesvd_check<T>(h,B,ju,jv,m,n,a.ptr,da,values.data()+1,ju=='V'?u.ptr:nullptr,ju=='V'?du:nullptr,jv=='V'?v.ptr:nullptr,jv=='V'?dv:nullptr);
    require_close(h,"shape checker preserves query matrix",same(initial,download(h,a.ptr,initial.size()))?0:1,0);
    require_close(h,"shape checker preserves query outputs",same(initial_u,download(h,u.ptr,initial_u.size()))&&
        same(initial_v,download(h,v.ptr,initial_v.size()))&&std::all_of(values.begin(),values.end(),[](R x){return x==R(-919);})?0:1,0);
    PgesvdRoute expected=PgesvdRoute::Direct;
    if(k && !(B==DdlaBackend::CPU&&h->nprocs==1) && (h->nprocs==1 || k/nb>=h->nprocs) &&
        (h->nprows_==1 || k/mb>=h->nprows_)) {
        if(m>=n && m/n>=4)expected=PgesvdRoute::TallQr;
        else if(n>m && n/m>=4) {
            int a_rows=0,f_rows=0,a_cols=0,f_cols=0;
            for(int r=0;r<h->nprows_;++r) {
                a_rows=std::max(a_rows,num_loc(m,mb,r,rs,h->nprows_));
                f_rows=std::max(f_rows,num_loc(n,mb,r,rs,h->nprows_));
            }
            for(int c=0;c<h->npcols_;++c) {
                a_cols=std::max(a_cols,num_loc(n,nb,c,cs,h->npcols_));
                f_cols=std::max(f_cols,num_loc(k,nb,c,cs,h->npcols_));
            }
            if(std::size_t(f_rows)*f_cols<=std::size_t(a_rows)*a_cols)expected=PgesvdRoute::WideQr;
        }
    }
    require_close(h,"shape frozen route and memory eligibility",p.route==expected?0:1,0);
    const char* prior_shape=std::getenv("DDLA_SVD_SHAPE");
    const bool had_shape=prior_shape!=nullptr;const std::string saved_shape=prior_shape?prior_shape:"";
    // Neither pure sizing nor raw execution may re-read the selection policy.
    setenv("DDLA_SVD_SHAPE","invalid-after-check",1);
    std::size_t bytes=0,host_bytes=0;
    MPI_Barrier(h->comm);bool pure=true;
    if(!h->myid) {
        workspace_test::Scope scope(true);pgesvd_workspace_size<T>(h,p,&bytes,&host_bytes);const auto counts=scope.finish();
        pure=no_allocation(counts)&&!counts.copies&&!counts.launches&&!counts.syncs&&!counts.collectives;
    }
    MPI_Barrier(h->comm);
    require_close(h,"shape rank-local pure size",pure?0:1,0);
    pgesvd_workspace_size<T>(h,p,&bytes,&host_bytes);
    Scratch scratch(h,bytes,host_bytes);
    std::optional<detail::DcHostTeam> team;
    if(!(B==DdlaBackend::CPU&&h->nprocs==1))if(const int threads=pgesvd_team_threads(p))team.emplace(threads);
    const char* prior_qr=std::getenv("DDLA_QR_PANEL");
    const std::string saved_qr=prior_qr?prior_qr:"";
    const bool had_qr=prior_qr!=nullptr;
    if(p.route!=PgesvdRoute::Direct)setenv("DDLA_QR_PANEL",p.qr_inplace?"gather":"inplace",1);
    // Warm native/runtime caches before the two changed-input raw-run traces.
    for(int repeat=-1;repeat<2;++repeat) {
        const int pass=std::max(0,repeat);
        auto input=local<T>(h,da,m,n,[&](int i,int j){return scale_for<R>(kind,pass)*entry<T>(m,n,i,j,kind,pass);});
        upload(h,a.ptr,input);upload(h,u.ptr,initial_u);upload(h,v.ptr,initial_v);DDLA_CHECK(ddlaSynchronize(h));
        int info=-1;workspace_test::Scope scope;
        pgesvd_run<B,T>(h,p,team?&*team:nullptr,a.ptr,values.data()+1,ju=='V'?u.ptr:nullptr,jv=='V'?v.ptr:nullptr,info,
            scratch.backend(),bytes,scratch.host(),host_bytes);
        DDLA_CHECK(ddlaSynchronize(h));const auto counts=scope.finish();
        if(repeat>=0)require_close(h,"shape borrowed run no allocation/query/setup",no_allocation(counts)?0:1,0);
        require_close(h,"shape info",info?1:0,0);
        auto aa=download(h,a.ptr,input.size()),uu=download(h,u.ptr,initial_u.size()),vv=download(h,v.ptr,initial_v.size());
        require_close(h,"shape input padding",padding(h,aa,input,da,m,n)?0:1,0);
        require_close(h,"shape U padding",padding(h,uu,initial_u,du,ju=='V'?m:0,ju=='V'?k:0)?0:1,0);
        require_close(h,"shape VT padding",padding(h,vv,initial_v,dv,jv=='V'?k:0,jv=='V'?n:0)?0:1,0);
        require_close(h,"shape singular value guards",values.front()==R(-919)&&values.back()==R(-919)?0:1,0);
        if(repeat>=0) {
            const std::vector<R> active_values(values.begin()+1,values.end()-1);
            auto root_values=active_values;
            MPI_Bcast(root_values.data(),k,detail::mpi_datatype<R>(),0,h->comm);
            require_close(h,"shape all-rank singular values",same(active_values,root_values)?0:1,0);
            numerical(h,m,n,kind,pass,active_values,uu,du,vv,dv,da,input,ju=='V',jv=='V');
        }
        scratch.verify();
    }
    if(had_qr)setenv("DDLA_QR_PANEL",saved_qr.c_str(),1);else unsetenv("DDLA_QR_PANEL");
    if(had_shape)setenv("DDLA_SVD_SHAPE",saved_shape.c_str(),1);else unsetenv("DDLA_SVD_SHAPE");
    if(!h->myid)std::cout<<"SVD_SHAPE_CASE m="<<m<<" n="<<n<<" ju="<<ju<<" jv="<<jv<<" kind="<<kind
        <<" scalar_bytes="<<sizeof(T)<<" route="<<int(p.route)<<" backend_bytes="<<bytes<<" host_bytes="<<host_bytes
        <<" frozen_shape_policy=1"<<std::endl;
}

template<DdlaBackend B,class T>void shape_policy(DdlaHandle_t h) {
    using R=typename real_type<T>::type;
    const int k=4*h->nprocs,m=4*k;int d[DDLA_DLEN_],du[DDLA_DLEN_],dv[DDLA_DLEN_];
    describe(h,m,k,1,1,0,0,d);describe(h,m,k,1,1,0,0,du);describe(h,k,k,1,1,0,0,dv);
    const auto input=local<T>(h,d,m,k,[](int i,int j){return make_scalar<T>(double(i==j)+.01*(i+j));});
    const auto initial_u=local<T>(h,du,m,k,[](int,int){return make_scalar<T>(-717,23);});
    const auto initial_v=local<T>(h,dv,k,k,[](int,int){return make_scalar<T>(-818,31);});
    DeviceBuffer<T> a(h,input.size()),u(h,initial_u.size()),v(h,initial_v.size());
    upload(h,a.ptr,input);upload(h,u.ptr,initial_u);upload(h,v.ptr,initial_v);std::vector<R> values(k,R(-919));
    const char* prior=std::getenv("DDLA_SVD_SHAPE");const bool had_prior=prior!=nullptr;
    const std::string saved=prior?prior:"";
    const auto unchanged=[&]{return same(input,download(h,a.ptr,input.size()))&&
        same(initial_u,download(h,u.ptr,initial_u.size()))&&same(initial_v,download(h,v.ptr,initial_v.size()))&&
        std::all_of(values.begin(),values.end(),[](R x){return x==R(-919);});};
    const auto check=[&]{return pgesvd_check<T>(h,B,'V','V',m,k,a.ptr,d,values.data(),u.ptr,du,v.ptr,dv);};
    for(const char* mode:{static_cast<const char*>(nullptr),"","direct","qr"}) {
        if(mode)setenv("DDLA_SVD_SHAPE",mode,1);else unsetenv("DDLA_SVD_SHAPE");
        // Check deliberately agrees arguments/policy across ranks. Query mode
        // rejects every MPI call; it belongs to pure sizing, not this phase.
        workspace_test::Scope scope;const auto p=check();const auto seen=scope.finish();
        require_close(h,"shape policy check retains observed agreement collectives",
            h->nprocs>1 && !seen.collectives?1:0,0);
        const bool qr=mode && std::string(mode)=="qr" && !(B==DdlaBackend::CPU&&h->nprocs==1);
        require_close(h,"shape absent/empty/direct default and explicit QR",p.route==(qr?PgesvdRoute::TallQr:PgesvdRoute::Direct)?0:1,0);
        require_close(h,"shape policy check preserves input and values",unchanged()?0:1,0);
        if(!qr)require_close(h,"direct policy geometry has no vendor query or matrix work",
            !seen.vendor_query&&!seen.copies&&!seen.launches&&!seen.syncs?0:1,0);
    }
    // Invalid values fail in the existing agreed check, before native workspace
    // queries, resource setup or matrix mutation. Owning INFO still resets to 0.
    for(int mode=0;mode<3;++mode) {
        setenv("DDLA_SVD_SHAPE",mode==0?"QR":mode==1?"direct ":h->myid?"qr":"invalid",1);
        bool rejected=false;std::string why;workspace_test::Scope scope;
        try{(void)check();}catch(const std::exception& e){rejected=true;why=e.what();}
        const auto seen=scope.finish();
        const bool shape_reason=why.find("DDLA_SVD_SHAPE")!=std::string::npos ||
            why.find("invalid argument on another rank")!=std::string::npos;
        require_close(h,"invalid shape policy rejects collectively before native work",rejected&&
            shape_reason&&!seen.vendor_query&&!seen.copies&&!seen.launches&&!seen.syncs&&!seen.resource_setup?0:1,0);
        std::cout<<"SVD_SHAPE_POLICY_REJECTION rank="<<h->myid<<" scalar_bytes="<<sizeof(T)
            <<" mode="<<mode<<" reason="<<why<<std::endl;
        int info=-17;rejected=false;why.clear();workspace_test::Scope owning_scope;
        try{pgesvd<T>(h,'V','V',m,k,a.ptr,d,values.data(),u.ptr,du,v.ptr,dv,info);}
        catch(const std::exception& e){rejected=true;why=e.what();}
        const auto owning_seen=owning_scope.finish();
        const bool owning_reason=why.find("DDLA_SVD_SHAPE")!=std::string::npos ||
            why.find("invalid argument on another rank")!=std::string::npos;
        require_close(h,"invalid owning shape policy preserves matrices/values and INFO contract",
            rejected&&owning_reason&&info==0&&!owning_seen.vendor_query&&!owning_seen.copies&&
            !owning_seen.launches&&!owning_seen.syncs&&!owning_seen.resource_setup&&unchanged()?0:1,0);
        std::cout<<"SVD_SHAPE_POLICY_REJECTION rank="<<h->myid<<" scalar_bytes="<<sizeof(T)
            <<" owning=1 mode="<<mode<<" reason="<<why<<std::endl;
    }
    if(h->nprocs>1) {
        setenv("DDLA_SVD_SHAPE",h->myid?"qr":"direct",1);
        bool rejected=false;std::string why;workspace_test::Scope scope;
        try{(void)check();}catch(const std::exception& e){rejected=true;why=e.what();}
        const auto seen=scope.finish();
        require_close(h,"eligible QR/direct rank mismatch rejects before native preparation",rejected&&
            why.find("global metadata differs between ranks")!=std::string::npos&&
            !seen.vendor_query&&!seen.copies&&!seen.launches&&!seen.syncs&&!seen.resource_setup&&unchanged()?0:1,0);
        std::cout<<"SVD_SHAPE_POLICY_REJECTION rank="<<h->myid<<" scalar_bytes="<<sizeof(T)
            <<" mode=eligible-mismatch reason="<<why<<std::endl;
    }
    if(had_prior)setenv("DDLA_SVD_SHAPE",saved.c_str(),1);else unsetenv("DDLA_SVD_SHAPE");
    if(!h->myid)std::cout<<"SVD_SHAPE_POLICY_PASS scalar_bytes="<<sizeof(T)
        <<" default_direct=1 explicit_qr=1 invalid_collective=1 unchanged=1 eligible_mismatch="<<(h->nprocs>1)<<std::endl;
}

template<DdlaBackend B>void aliases(DdlaHandle_t h) {
    constexpr int n=7;int d[DDLA_DLEN_];describe(h,n,n,4,4,0,0,d);
    auto initial=local<double>(h,d,n,n,[](int i,int j){return double(i==j);});
    DeviceBuffer<double> a(h,initial.size()),u(h,initial.size());upload(h,a.ptr,initial);upload(h,u.ptr,initial);
    std::vector<double> w(n,-919);
    for(int kind=0;kind<3;++kind) {
        bool caught=false;
        try {
            (void)pgesvd_check<double>(h,B,'V','V',n,n,a.ptr,d,w.data(),kind==0?a.ptr:u.ptr,d,
                kind==1?a.ptr:u.ptr,d);
        } catch(const std::exception&) {caught=true;}
        require_close(h,"shape alias rejected collectively",caught?0:1,0);
        require_close(h,"shape alias rejection unchanged A",same(initial,download(h,a.ptr,initial.size()))?0:1,0);
        require_close(h,"shape alias rejection unchanged output",same(initial,download(h,u.ptr,initial.size()))?0:1,0);
        require_close(h,"shape alias rejection unchanged values",std::all_of(w.begin(),w.end(),[](double x){return x==-919;})?0:1,0);
    }
}
template<DdlaBackend B>void qr_policy_mismatch(DdlaHandle_t h) {
    if(h->nprocs<2)return;
    const char* shape=std::getenv("DDLA_SVD_SHAPE");
    require_close(h,"independent QR policy mismatch starts with explicit QR route",shape&&std::strcmp(shape,"qr")==0?0:1,0);
    const int k=4*h->nprocs,m=4*k;int d[DDLA_DLEN_];describe(h,m,k,3,2,0,0,d);
    auto input=local<double>(h,d,m,k,[](int i,int j){return double(i==j)+.01*(i+j);});
    DeviceBuffer<double> a(h,input.size());upload(h,a.ptr,input);std::vector<double> values(k,-919);
    const char* prior=std::getenv("DDLA_QR_PANEL");const bool had_prior=prior!=nullptr;
    const std::string saved=prior?prior:"";
    setenv("DDLA_QR_PANEL",h->myid?"gather":"inplace",1);
    bool caught=false;
    try {(void)pgesvd_check<double>(h,B,'N','N',m,k,a.ptr,d,values.data(),nullptr,nullptr,nullptr,nullptr);}
    catch(const std::exception&) {caught=true;}
    if(had_prior)setenv("DDLA_QR_PANEL",saved.c_str(),1);else unsetenv("DDLA_QR_PANEL");
    require_close(h,"shape QR policy mismatch rejected collectively",caught?0:1,0);
    require_close(h,"shape policy mismatch preserves inputs",same(input,download(h,a.ptr,input.size()))&&
        std::all_of(values.begin(),values.end(),[](double x){return x==-919;})?0:1,0);
}

template<DdlaBackend B,class T>void cases(DdlaHandle_t h) {
    const int k=4*h->nprocs;
    if(quick) {
        for(char ju:{'N','V'})for(char jv:{'N','V'}) {
            exercise<B,T>(h,4*k,k,ju,jv,0);
            exercise<B,T>(h,k,4*k,ju,jv,0);
        }
        exercise<B,T>(h,4,1,'V','V',0,false);
        exercise<B,T>(h,1,4,'V','V',0,false);
        exercise<B,T>(h,5,5,'V','V',0);
        return;
    }
    for(int ratio:{4,8,16})for(char ju:{'N','V'})for(char jv:{'N','V'}) {
        exercise<B,T>(h,ratio*k,k,ju,jv,0);
        exercise<B,T>(h,k,ratio*k,ju,jv,0);
    }
    for(int kind:{1,2,3,4,5}) {
        exercise<B,T>(h,4*k,k,'V','V',kind);
        exercise<B,T>(h,k,4*k,'V','V',kind);
    }
    exercise<B,T>(h,5,5,'V','V',0);exercise<B,T>(h,7,5,'V','V',0);
    exercise<B,T>(h,2*h->nprocs+1,4*(2*h->nprocs+1),'V','V',0);
    exercise<B,T>(h,20,5,'V','V',0);
    exercise<B,T>(h,4,1,'V','V',0,false);exercise<B,T>(h,1,4,'V','V',0,false);
    exercise<B,T>(h,0,5,'N','N',0);
}
template<DdlaBackend B>void run(int pr,int pc) {
    for(char major:{'R','C'}) {
        DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,B));
        if constexpr(B==DdlaBackend::GPU)if(transport!="default")
            DDLA_CHECK(ddlaSetCommunication(h,transport=="ccl"?DdlaCommunication::CCL:
                transport=="staged"?DdlaCommunication::HostStagedMPI:DdlaCommunication::MPI));
        DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,pr,pc,major));
        shape_policy<B,float>(h);shape_policy<B,double>(h);
        shape_policy<B,std::complex<float>>(h);shape_policy<B,std::complex<double>>(h);
        aliases<B>(h);qr_policy_mismatch<B>(h);cases<B,float>(h);cases<B,double>(h);cases<B,std::complex<float>>(h);cases<B,std::complex<double>>(h);
        if(!h->myid)std::cout<<"SVD_SHAPE_PASS backend="<<int(B)<<" grid="<<pr<<'x'<<pc<<" major="<<major<<" no_performance_claim=1"<<std::endl;
        DDLA_CHECK(ddlaDestroy(h));
    }
}
}
int main(int argc,char** argv) {
    ddla_test_mpi_init(&argc,&argv);
    try {
        int ranks=1;MPI_Comm_size(MPI_COMM_WORLD,&ranks);int pr=1,pc=ranks;bool gpu=false;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--gpu")gpu=true;
            else if(arg=="--quick")quick=true;
            else if(arg=="--cpu")gpu=false;
            else if(arg=="--transport"&&i+1<argc) {
                transport=argv[++i];
                if(transport!="default"&&transport!="mpi"&&transport!="staged"&&transport!="ccl")
                    throw std::invalid_argument("invalid transport");
            }
            else if(arg=="--grid"&&i+1<argc) {
                const std::string grid=argv[++i];const auto split=grid.find('x');pr=std::stoi(grid.substr(0,split));pc=std::stoi(grid.substr(split+1));
            } else throw std::invalid_argument("test_pgesvd_shape: invalid argument");
        }
        if(pr<1||pc<1||pr*pc!=ranks)throw std::invalid_argument("grid must match launched ranks");
        // This fixture intentionally validates the opt-in QR math paths.
        setenv("DDLA_SVD_SHAPE","qr",1);
        if(gpu) {
#if DDLA_HAS_GPU
            run<DdlaBackend::GPU>(pr,pc);
#else
            throw std::invalid_argument("GPU unavailable");
#endif
        } else run<DdlaBackend::CPU>(pr,pc);
        MPI_Finalize();return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
