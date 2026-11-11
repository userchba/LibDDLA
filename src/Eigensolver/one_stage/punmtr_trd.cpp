#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "ddla_connector.h"
#include "punmtr_trd_impl.h"
#include "punmtr_impl.h"
#include "reflector_build_T.h"
#include "Eigensolver/pvector_redist.h"
#include "Eigensolver/pheevd_alias.h"
#include "BLAS/blas_impl.h"
#include "check_agree.h"
#include "plan_util.h"
#include "helpers.h"
#include "instantiate.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <climits>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ddla { namespace spectral_impl {
UnmtrTrdPlan punmtr_trd_geometry(const DdlaHandle_t& h,DdlaBackend backend,char side,char uplo,char trans,
    int m,int n,const int* a,const int* c) {
    UnmtrTrdPlan p;p.backend=backend;p.side=side;p.uplo=uplo;p.trans=trans;
    p.m=m;p.n=n;p.nq=side=='L'?m:n;
    ddlaGetGridDims(h,p.pr,p.pc);ddlaGetGridCoords(h,p.mr,p.mc);
    std::copy_n(a,DDLA_DLEN_,p.a.begin());std::copy_n(c,DDLA_DLEN_,p.c.begin());return p;
}
template<class T>UnmtrTrdPlan punmtr_trd_check(const DdlaHandle_t& h,DdlaBackend backend,char side,char uplo,char trans,
    int m,int n,const T* A,const int* a,const T* tau,T* C,const int* c) {
    auto p=detail::agree_check(h,"punmtr stored reflectors",[&](detail::Agree& agreement) {
        agreement.mode("punmtr interface",1);
        if(!h||ddlaGetBackend(h)!=backend)throw std::invalid_argument("punmtr: null handle or backend mismatch");
        const char s=static_cast<char>(std::toupper(static_cast<unsigned char>(side)));
        const char u=static_cast<char>(std::toupper(static_cast<unsigned char>(uplo)));
        char t=static_cast<char>(std::toupper(static_cast<unsigned char>(trans)));
        constexpr bool real=std::is_same<T,typename real_type<T>::type>::value;
        if((s!='L'&&s!='R')||(u!='U'&&u!='L')||(t!='N'&&t!='C'&&!(real&&t=='T')))
            throw std::invalid_argument("punmtr: invalid side/uplo/trans");
        if(t=='T')t='C';
        check_desc(a,h);check_desc(c,h);
        const int nq=s=='L'?m:n;
        const int nb=a[DDLA_NB_];
        if(m<0||n<0||m>c[DDLA_M_]||n>c[DDLA_N_]||nq>a[DDLA_M_]||nq>a[DDLA_N_]||nq>INT_MAX/2||
           a[DDLA_MB_]!=nb||a[DDLA_RSRC_]!=0||a[DDLA_CSRC_]!=0||
           (s=='L'?(c[DDLA_MB_]!=nb||c[DDLA_RSRC_]!=0):(c[DDLA_NB_]!=nb||c[DDLA_CSRC_]!=0)))
            throw std::invalid_argument("punmtr: incompatible extents, blocks or reflector ownership");
        auto result=punmtr_trd_geometry(h,backend,s,u,t,m,n,a,c);
        if(!m||!n||nq<2)return result;
        const int ar=num_loc(nq,nb,result.mr,0,result.pr),ac=num_loc(nq,nb,result.mc,0,result.pc);
        const int cr=num_loc(m,c[DDLA_MB_],result.mr,c[DDLA_RSRC_],result.pr);
        const int cc=num_loc(n,c[DDLA_NB_],result.mc,c[DDLA_CSRC_],result.pc);
        const auto count=static_cast<std::size_t>(nb)*std::max(1,s=='L'?cc:cr);
        if(static_cast<std::size_t>(nb)*nb>static_cast<std::size_t>(INT_MAX/2)||count>static_cast<std::size_t>(INT_MAX/2))
            throw std::overflow_error("punmtr: panel collective count exceeds int");
        if(!tau||(ar&&ac&&!A)||(cr&&cc&&!C))throw std::invalid_argument("punmtr: missing active data");
        const auto ab=detail::reflector_active_span<T>(ar,ac,a[DDLA_LLD_]);
        const auto cb=detail::reflector_active_span<T>(cr,cc,c[DDLA_LLD_]);
        if(detail::memory_overlaps(A,ab,C,cb)||detail::memory_overlaps(tau,std::size_t(nq-1)*sizeof(T),C,cb))
            throw std::invalid_argument("punmtr: output overlaps stored reflectors or tau");
        return result;
    });
    int sig[24]={int(backend),int(p.side),int(p.uplo),int(p.trans),m,n,p.pr,p.pc,int(h->major),p.nq};
    detail::descriptor_signature(sig+10,a);detail::descriptor_signature(sig+17,c);
    detail::agree_signature(h,"punmtr stored reflectors",sig);return p;
}
template<class T>void punmtr_trd_workspace_size(const DdlaHandle_t& h,const UnmtrTrdPlan& p,std::size_t* bytes) {
    *bytes=0;if(!p.m||!p.n||p.nq<2)return;
    const bool left=p.side=='L',lower=p.uplo=='L',transpose=left==lower;
    const int nb=p.a[DDLA_NB_];const std::size_t cap=nb;
    const std::size_t lr=std::max(1,num_loc(p.nq,nb,left?p.mr:p.mc,0,left?p.pr:p.pc));
    const std::size_t lv=std::max(1,num_loc(p.nq,nb,lower?p.mc:p.mr,0,lower?p.pc:p.pr));
    const std::size_t axis=std::max(1,left?num_loc(p.n,p.c[DDLA_NB_],p.mc,p.c[DDLA_CSRC_],p.pc):
        num_loc(p.m,p.c[DDLA_MB_],p.mr,p.c[DDLA_RSRC_],p.pr));
    std::size_t transfer=0;
    if(transpose)ptranspose_vectors_workspace_size<T>(h,p.nq,nb,nb,&transfer);
    *bytes=detail::padded_bytes<T>(lr*cap)+3*detail::padded_bytes<T>(cap*cap)+
        2*detail::padded_bytes<T>(cap*axis)+(transpose?detail::padded_bytes<T>(lv*cap):0)+transfer;
}
namespace {
#if DDLA_GPU_VENDOR
template <class T>
__global__ void anchors_kernel(T* v, int ld, int width, int first, int nb, int row, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y;
    if (i < ld && j < width) {
        const int g = (i / nb * rows + row) * nb + i % nb, anchor = first + j - 1;
        if (g >= anchor)
            v[i + static_cast<std::size_t>(j) * ld] = g == anchor ? T(1) : T(0);
    }
}

#endif
} // namespace

template<DdlaBackend B,class T>void punmtr_trd_run(const DdlaHandle_t& h,const UnmtrTrdPlan& p,
    const T* A,const T* tau,T* C,void* backend_work,std::size_t backend_bytes) {
    if(!p.m||!p.n||p.nq<2)return;
    const bool left=p.side=='L',transpose=left==(p.uplo=='L');
    const int nq=p.nq,nb=p.a[DDLA_NB_],ld=p.a[DDLA_LLD_],ldc=p.c[DDLA_LLD_];
    const bool lower=p.uplo=='L';
    const int logical_pr=lower?p.pc:p.pr,logical_pc=lower?p.pr:p.pc;
    const int logical_mr=lower?p.mc:p.mr,logical_mc=lower?p.mr:p.mc;
    const int lr=std::max(1,num_loc(nq,nb,left?p.mr:p.mc,0,left?p.pr:p.pc));
    const int lv=std::max(1,num_loc(nq,nb,logical_mr,0,logical_pr));
    const int cr=num_loc(p.m,p.c[DDLA_MB_],p.mr,p.c[DDLA_RSRC_],p.pr);
    const int cc=num_loc(p.n,p.c[DDLA_NB_],p.mc,p.c[DDLA_CSRC_],p.pc);
    const int axis=std::max(1,left?cc:cr);
    char* const base=static_cast<char*>(backend_work);char* next=base;
    T* v=detail::take<T>(next,std::size_t(lr)*nb);
    T* g=detail::take<T>(next,std::size_t(nb)*nb);
    T* gp=detail::take<T>(next,std::size_t(nb)*nb);
    T* t=detail::take<T>(next,std::size_t(nb)*nb);
    T* w=detail::take<T>(next,std::size_t(nb)*axis);
    T* wr=detail::take<T>(next,std::size_t(nb)*axis);
    T* stored=transpose?detail::take<T>(next,std::size_t(lv)*nb):v;
    assert(static_cast<std::size_t>(next-base)<=backend_bytes);
    void* transfer=next;const auto transfer_bytes=backend_bytes-static_cast<std::size_t>(next-base);
    const bool forward=left==(p.trans=='N');
    for(int first=forward?1:std::max(1,(nq-1)/nb*nb);first<nq;) {
        const int width=std::min(nb-first%nb,nq-first);
        const int owner=first/nb%logical_pc;
        if(logical_mc==owner) {
            DDLA_CHECK(ddlaMemset(h,stored,0,std::size_t(lv)*width*sizeof(T)));
            for(int j=first;j<first+width;++j) {
                const int count=num_loc(j,nb,logical_mr,0,logical_pr);
                if(!count)continue;
                const T* src=A+detail::triangle_offset(lower,ld,0,indxg2l(j,nb,logical_pc));
                T* dst=stored+std::size_t(j-first)*lv;
                if(lower)
                    blas_impl::geam_run<B,T>(h,'C','C',count,1,T(1),src,ld,T(0),src,ld,dst,lv);
                else DDLA_CHECK(ddlaMemcpy(h,dst,src,std::size_t(count)*sizeof(T)));
            }
            if constexpr(B==DdlaBackend::CPU) {
                for(int j=0;j<width;++j)for(int i=0;i<lv;++i) {
                    const int global=indxl2g(i,nb,logical_mr,0,logical_pr),anchor=first+j-1;
                    if(global>=anchor)stored[i+std::size_t(j)*lv]=global==anchor?T(1):T(0);
                }
            } else {
#if DDLA_GPU_VENDOR
                using D=detail::DeviceScalarT<T>;
                anchors_kernel<D><<<dim3((lv+255)/256,width),256,0,static_cast<gpuStream_t>(h->stream)>>>(
                    reinterpret_cast<D*>(stored),lv,width,first,nb,logical_mr,logical_pr);
                DDLA_CHECK(ddlaGetLastError(h));
#endif
            }
            const int active=num_loc(first+width-1,nb,logical_mr,0,logical_pr);
            DDLA_CHECK(ddlaMemset(h,gp,0,std::size_t(width)*width*sizeof(T)));
            if(active)blas_impl::gemm_run<B,T>(h,'C','N',width,width,active,T(1),stored,lv,stored,lv,T(0),gp,width);
            commAllReduce<B>(h,(lower?CommScope::Row:CommScope::Col),gp,g,width*width,cclSum);
            detail::reflector_build_T<B,T>(h,width,g,tau+first-1,t,width,false);
        }
        commBcast<B>(h,(lower?CommScope::Col:CommScope::Row),stored,std::size_t(lv)*width,owner);
        commBcast<B>(h,(lower?CommScope::Col:CommScope::Row),t,std::size_t(width)*width,owner);
        if(transpose) {
            const auto transfer_plan=ptranspose_vectors_geometry(h,B,first+width-1,width,nb,lv,lr,false,left);
            ptranspose_vectors_run<B,T>(h,transfer_plan,stored,v,transfer,transfer_bytes);
        }
        const int active=num_loc(first+width-1,nb,left?p.mr:p.mc,0,left?p.pr:p.pc);

        detail::punmtr_run<B,T>(h,p.side,'U',p.trans=='N'?'C':'N',left?active:cr,left?cc:active,width,lr,width,ldc,left?width:std::max(1,cr),(p.side=='L'?h->nprows_:h->npcols_)>1,v,t,C,w,wr);
        if(forward)first+=width;
        else {if(first==1)break;first=std::max(1,first-nb);}
    }
}
} // namespace spectral_impl

template<DdlaBackend B,class T>void punmtr(const DdlaHandle_t& h,char side,char uplo,char trans,int m,int n,
    const T* A,const int* a,const T* tau,T* C,const int* c) {
    const auto p=spectral_impl::punmtr_trd_check<T>(h,B,side,uplo,trans,m,n,A,a,tau,C,c);
    if(!m||!n||p.nq<2)return;
    std::size_t bytes=0;spectral_impl::punmtr_trd_workspace_size<T>(h,p,&bytes);
    vector<char> work(h);
    detail::agree_check(h,"punmtr stored allocation",[&](detail::Agree&){work.resize(bytes);});
    spectral_impl::punmtr_trd_run<B,T>(h,p,A,tau,C,work.data(),bytes);
}
#define DDLA_UNMTR_TRD(B,T) \
 template void punmtr<B,T>(const DdlaHandle_t&,char,char,char,int,int,const T*,const int*,const T*,T*,const int*); \
 template void spectral_impl::punmtr_trd_run<B,T>(const DdlaHandle_t&,const spectral_impl::UnmtrTrdPlan&,const T*,const T*,T*,void*,std::size_t)
DDLA_FOR_EACH_BACKEND_TYPE(DDLA_UNMTR_TRD)
#define DDLA_UNMTR_TRD_PLAN(T) \
 template spectral_impl::UnmtrTrdPlan spectral_impl::punmtr_trd_check<T>(const DdlaHandle_t&,DdlaBackend,char,char,char,int,int,const T*,const int*,const T*,T*,const int*); \
 template void spectral_impl::punmtr_trd_workspace_size<T>(const DdlaHandle_t&,const spectral_impl::UnmtrTrdPlan&,std::size_t*)
DDLA_FOR_EACH_TYPE(DDLA_UNMTR_TRD_PLAN)
#undef DDLA_UNMTR_TRD_PLAN
#undef DDLA_UNMTR_TRD
} // namespace ddla
