#include "api_grid_test_common.h"
#include "workspace_test_common.h"
#include "ddla_stream.h"
#include "PBLAS/pgeadd_impl.h"
#include "PBLAS/pgemm_impl.h"
#include "PBLAS/pgemmt_impl.h"
#include "PBLAS/prankk_impl.h"
#include "PBLAS/ptrtrs_impl.h"
#include "PBLAS/ptrmm_impl.h"
#include "PBLAS/ptrtrm_impl.h"
#include "PBLAS/phemm_impl.h"
#include "PBLAS/pswap_impl.h"
#include "PBLAS/plapiv_impl.h"
#include "PBLAS/pdam_impl.h"
#include "PBLAS/pblas_workspace.h"
#include <ddla/ddla_vector.h>

#include <array>
#include <cstring>

using namespace api_grid_test;
namespace {
template <typename S> bool no_scratch(const S& s)
{return s.ddla_alloc==0 && s.ddla_free==0 && s.pinned_alloc==0 && s.raw_gpu_alloc==0 && s.host_new==0 && s.vendor_query==0 && s.resource_setup==0;}
template <typename S> bool pure(const S& s)
{return no_scratch(s) && s.copies==0 && s.launches==0 && s.syncs==0 && s.collectives==0;}
template <typename T> T entry(int i,int j,int salt)
{
    using R=typename real_type<T>::type;
    const R real=(i==j?R(30):R(0))+R(i+1)/R(71)+R(j+1)/R(113)+R(salt)/R(19);
    if constexpr(std::is_arithmetic_v<T>)return real;
    else return T(real,R(i-j)/R(127));
}
// Exactly the queried bytes of both kinds, surrounded by guards. The same
// addresses serve every run of a case.
class GuardedScratch {
    static constexpr std::size_t guard=256;
    DdlaHandle_t h_;
    std::size_t backend_bytes_,host_bytes_;
    ddla::vector<char> backend_;
    std::vector<char> host_;
public:
    GuardedScratch(DdlaHandle_t h,std::size_t backend_bytes,std::size_t host_bytes)
        :h_(h),backend_bytes_(backend_bytes),host_bytes_(host_bytes),
         backend_(h,backend_bytes+2*guard),host_(host_bytes+2*guard,char(0x6b))
    { DDLA_CHECK(ddlaMemset(h,backend_.data(),0x6b,backend_.size())); }
    void* backend(){return backend_.data()+guard;}
    void* host(){return host_.data()+guard;}
    void verify()
    {
        std::array<char,2*guard> device{};
        DDLA_CHECK(ddlaMemcpy(h_,device.data(),backend_.data(),guard));
        DDLA_CHECK(ddlaMemcpy(h_,device.data()+guard,backend_.data()+guard+backend_bytes_,guard));
        check_ddla_sync(h_);bool okay=true;
        for(std::size_t i=0;i<guard;++i)okay=okay && host_[i]==char(0x6b)
            && host_[guard+host_bytes_+i]==char(0x6b)
            && device[i]==char(0x6b) && device[guard+i]==char(0x6b);
        require_close(h_,"PBLAS queried workspace guards",okay?0:1,0);
    }
};
template <typename T> struct Matrices {
    DdlaHandle_t h;int d[DDLA_DLEN_];std::size_t count;
    DeviceBuffer<T> a,b,c;std::vector<T> ha,hb,hc;
    static std::size_t size(DdlaHandle_t h,int* d,int extent,int block)
    {DDLA_CHECK(ddlaDescInit(d,h,extent,extent,block,block,0,0));d[DDLA_LLD_]+=3+h->myid%3;return local_size(h,d);}
    explicit Matrices(DdlaHandle_t handle,int extent=16,int block=2):h(handle),count(size(h,d,extent,block)),a(h,count),b(h,count),c(h,count){}
    void reset(int salt)
    {
        ha=make_local<T>(h,d,[&](int i,int j){return entry<T>(i,j,salt);});
        hb=make_local<T>(h,d,[&](int i,int j){return entry<T>(i,j,salt+3);});
        hc=make_local<T>(h,d,[&](int i,int j){return entry<T>(i,j,salt+7);});
        for(int j=0;j<ddla_test::n_loc(h,d);++j)for(int i=ddla_test::m_loc(h,d);i<d[DDLA_LLD_];++i)
            ha[i+std::size_t(j)*d[DDLA_LLD_]]=hb[i+std::size_t(j)*d[DDLA_LLD_]]=hc[i+std::size_t(j)*d[DDLA_LLD_]]=T(-937);
        upload(h,a.ptr,ha);upload(h,b.ptr,hb);upload(h,c.ptr,hc);
    }
    void compare(const char* name,const std::vector<T>& ea,const std::vector<T>& eb,const std::vector<T>& ec)
    {
        const auto aa=download(h,a.ptr,count),bb=download(h,b.ptr,count),cc=download(h,c.ptr,count);
        double error=0;
        for(std::size_t i=0;i<count;++i) {
            error=std::max({error,workspace_test::relative_error(aa[i],ea[i]),
                workspace_test::relative_error(bb[i],eb[i]),workspace_test::relative_error(cc[i],ec[i])});
        }
        const double tolerance=sizeof(typename real_type<T>::type)==4?3e-5:2e-12;
        require_close(h,name,error,tolerance);
    }
};
// query(&backend_bytes, &host_bytes) is the size function of a case, run(backend,
// backend_bytes, host, host_bytes) its run and gold() the public entry it must
// match.  The query is pure (no allocation, copy, launch, sync or
// collective) and the run on the queried scratch allocates nothing.
template <typename T,typename Check,typename Query,typename Run,typename Gold>
void exercise(Matrices<T>& x,const char* name,Check check,Query query,Run run,Gold gold)
{
    x.reset(0);check();MPI_Barrier(x.h->comm);
    std::size_t backend_bytes=0,host_bytes=0;
    bool one=true;
    if(x.h->myid==0){workspace_test::Scope scope(true);query(&backend_bytes,&host_bytes);one=pure(scope.finish());}
    MPI_Barrier(x.h->comm);require_close(x.h,"PBLAS one-rank pure query",one?0:1,0);
    workspace_test::Scope query_scope(true);query(&backend_bytes,&host_bytes);const auto qs=query_scope.finish();
    require_close(x.h,"PBLAS all-rank pure query",pure(qs)?0:1,0);
    GuardedScratch scratch(x.h,backend_bytes,host_bytes);
    for(int pass=0;pass<2;++pass) {
        x.reset(pass);gold();check_ddla_sync(x.h);
        const auto ea=download(x.h,x.a.ptr,x.count),eb=download(x.h,x.b.ptr,x.count),ec=download(x.h,x.c.ptr,x.count);
        x.reset(pass);
        workspace_test::Scope scope;run(scratch.backend(),backend_bytes,scratch.host(),host_bytes);const auto snapshot=scope.finish();
        require_close(x.h,"PBLAS run on queried scratch has no managed scratch/query/setup",no_scratch(snapshot)?0:1,0);
        check_ddla_sync(x.h);x.compare(name,ea,eb,ec);scratch.verify();
    }
}

template <DdlaBackend B,typename T> void cases(DdlaHandle_t h)
{
    Matrices<T> x(h);const int n=11,k=7,nrhs=15;const T alpha=T(0.7),beta=T(0.3);
    for(auto algorithm:{pgemm_impl::Algorithm::AB,pgemm_impl::Algorithm::AC,pgemm_impl::Algorithm::BC}) {
        int opts_panel_width=1;const auto opts=pgemm_impl::pgemm_check<B>(h,'N','N',n,n,k,false,x.d,x.d,x.d,algorithm,opts_panel_width);
        exercise(x,"PBLAS GEMM branch reuse",[]{},
            [&](std::size_t* bb,std::size_t* hb){pgemm_impl::pgemm_workspace_size<T>(h,opts,opts_panel_width,false,'N','N',n,n,k,true,x.d,x.d,x.d,bb,hb);},
            [&](void* w,std::size_t wb,void* hw,std::size_t hwb){pgemm_impl::pgemm_run<B,T>(h,opts,opts_panel_width,'N','N',n,n,k,alpha,x.a.ptr,x.d,x.b.ptr,x.d,beta,x.c.ptr,x.d,w,wb,hw,hwb);},
            [&]{pgemm_impl::run<B,T>(h,'N','N',n,n,k,alpha,x.a.ptr,x.d,x.b.ptr,x.d,beta,x.c.ptr,x.d,algorithm);});
    }
    for(char trans:{'N','C'})for(bool alias:{false,true}) {
        exercise(x,"PBLAS GEADD transpose/alias reuse",[&]{pgeadd_impl::pgeadd_check<B>(h,trans,trans,n,n,x.d,1,1,x.d,1,1,x.d,1,1);},
            [&](std::size_t* bb,std::size_t* hb){pgeadd_impl::pgeadd_workspace_size<T>(h,trans,trans,true,true,n,n,x.d,bb,hb);},
            [&](void* w,std::size_t wb,void* hw,std::size_t hwb){pgeadd_impl::pgeadd_run<B,T>(h,trans,trans,n,n,alpha,x.a.ptr,1,1,x.d,beta,x.b.ptr,1,1,x.d,alias?x.a.ptr:x.c.ptr,1,1,x.d,w,wb,hw,hwb);},
            [&]{pgeadd<B,T>(h,trans,trans,n,n,alpha,x.a.ptr,1,1,x.d,beta,x.b.ptr,1,1,x.d,alias?x.a.ptr:x.c.ptr,1,1,x.d);});
    }
    for(char side:{'L','R'})for(char trans:{'N','C'}) {
        const int m=side=='L'?n:nrhs,cols=side=='L'?nrhs:n;
        exercise(x,"PBLAS triangular solve reuse",[&]{ptrtrs_impl::ptrtrs_check<B>(h,side,'L',trans,'N',m,cols,x.d,x.d);},
            [&](std::size_t* bb,std::size_t* hb){ptrtrs_impl::ptrtrs_workspace_size<T>(h,side,trans,m,cols,x.d,x.d,bb,hb);},
            [&](void* w,std::size_t wb,void* hw,std::size_t hwb){ptrtrs_impl::ptrtrs_run<B,T>(h,side,'L',trans,'N',m,cols,x.a.ptr,x.d,x.b.ptr,x.d,w,wb,hw,hwb);},
            [&]{ptrtrs<B,T>(h,side,'L',trans,'N',m,cols,x.a.ptr,x.d,x.b.ptr,x.d);});
        exercise(x,"PBLAS triangular multiply reuse",[&]{ptrmm_impl::ptrmm_check<B>(h,side,'U',trans,'N',m,cols,x.d,x.d);},
            [&](std::size_t* bb,std::size_t* hb){ptrmm_impl::ptrmm_workspace_size<T>(h,side,trans,m,cols,false,x.d,x.d,bb,hb);},
            [&](void* w,std::size_t wb,void* hw,std::size_t hwb){ptrmm_impl::ptrmm_run<B,T>(h,side,'U',trans,'N',m,cols,alpha,x.a.ptr,x.d,x.b.ptr,x.d,w,wb,hw,hwb);},
            [&]{ptrmm<B,T>(h,side,'U',trans,'N',m,cols,alpha,x.a.ptr,x.d,x.b.ptr,x.d);});
    }
    for(char ua:{'U','L'})for(char ub:{'U','L'})
        exercise(x,"PBLAS triangular product reuse",[&]{ptrtrm_impl::ptrtrm_check<B,T>(h,ua,'N',ub,'N',n,alpha,x.a.ptr,x.d,x.b.ptr,x.d,beta,x.c.ptr,x.d);},
            [&](std::size_t* bb,std::size_t* hb){ptrtrm_impl::ptrtrm_workspace_size<T>(h,'N','N',n,x.d,x.d,x.d,bb,hb);},
            [&](void* w,std::size_t wb,void* hw,std::size_t hwb){ptrtrm_impl::ptrtrm_run<B,T>(h,ua,'N',ub,'N',n,alpha,x.a.ptr,x.d,x.b.ptr,x.d,beta,x.c.ptr,x.d,w,wb,hw,hwb);},
            [&]{ptrtrm<B,T>(h,ua,'N',ub,'N',n,alpha,x.a.ptr,x.d,x.b.ptr,x.d,beta,x.c.ptr,x.d);});
    for(bool hermitian:{false,true})for(char side:{'L','R'})
        exercise(x,"PBLAS symmetric/Hermitian multiply reuse",[&]{phemm_impl::phemm_check<B,T>(h,side,'L',n,nrhs,alpha,x.a.ptr,x.d,x.b.ptr,x.d,beta,x.c.ptr,x.d,hermitian);},
            [&](std::size_t* bb,std::size_t* hb){phemm_impl::phemm_workspace_size<T>(h,side,n,nrhs,false,x.d,x.d,x.d,hermitian,bb,hb);},
            [&](void* w,std::size_t wb,void* hw,std::size_t hwb){phemm_impl::phemm_run<B,T>(h,side,'L',n,nrhs,alpha,x.a.ptr,x.d,x.b.ptr,x.d,beta,x.c.ptr,x.d,hermitian,w,wb,hw,hwb);},
            [&]{if(hermitian)phemm<B,T>(h,side,'L',n,nrhs,alpha,x.a.ptr,x.d,x.b.ptr,x.d,beta,x.c.ptr,x.d);else psymm<B,T>(h,side,'L',n,nrhs,alpha,x.a.ptr,x.d,x.b.ptr,x.d,beta,x.c.ptr,x.d);});
    for(bool hermitian:{false,true})for(bool rank2:{false,true})for(bool general:{false,true}) {
        if(general && rank2)continue;
        const char adj=hermitian?'C':'T';
        const char kind=general?'G':hermitian?'H':'S';
        const bool product=general
            ? pgemmt_impl::check<B,T>(h,'U','N',adj,n,k,alpha,x.a.ptr,x.d,x.b.ptr,x.d,beta,x.c.ptr,x.d)
            : prankk_impl::check<B,T>(h,kind,'U','N',adj,n,k,alpha,x.a.ptr,x.d,
                rank2?x.b.ptr:x.a.ptr,x.d,beta,x.c.ptr,x.d,rank2);
        exercise(x,"PBLAS triangular/rank update reuse",[]{},
            [&](std::size_t* bb,std::size_t* hb){if(general)pgemmt_impl::workspace_size<T>(h,'U','N',adj,product,n,k,x.d,x.d,x.d,bb,hb);
                else prankk_impl::workspace_size<T>(h,kind,'U','N',adj,product,rank2,n,k,x.d,x.d,x.d,bb,hb);},
            [&](void* w,std::size_t wb,void* hw,std::size_t hwb){if(general)pgemmt_impl::run<B,T>(h,'U','N',adj,product,n,k,alpha,x.a.ptr,x.d,x.b.ptr,x.d,beta,x.c.ptr,x.d,w,wb,hw,hwb);
                else prankk_impl::run<B,T>(h,kind,'U','N',adj,product,rank2,n,k,alpha,x.a.ptr,x.d,rank2?x.b.ptr:x.a.ptr,x.d,beta,x.c.ptr,x.d,w,wb,hw,hwb);},
            [&]{if(general){pgemmt<B,T>(h,'U','N',adj,n,k,alpha,x.a.ptr,x.d,x.b.ptr,x.d,beta,x.c.ptr,x.d);return;}
                // pherk/pher2k exist for complex types only; a real Hermitian update is the symmetric one.
                if constexpr(!std::is_arithmetic_v<T>)if(hermitian){
                    if(rank2)pher2k<B,T>(h,'U','N',n,k,alpha,x.a.ptr,x.d,x.b.ptr,x.d,std::real(beta),x.c.ptr,x.d);
                    else pherk<B,T>(h,'U','N',n,k,std::real(alpha),x.a.ptr,x.d,std::real(beta),x.c.ptr,x.d);
                    return;}
                if(rank2)psyr2k<B,T>(h,'U','N',n,k,alpha,x.a.ptr,x.d,x.b.ptr,x.d,beta,x.c.ptr,x.d);
                else psyrk<B,T>(h,'U','N',n,k,alpha,x.a.ptr,x.d,beta,x.c.ptr,x.d);});
    }
    const bool empty_product=pgemmt_impl::check<B,T>(h,'U','N','N',1,0,T(0),
        nullptr,x.d,nullptr,x.d,T(1),nullptr,x.d);
    require_close(h,"GEMMT empty product accepts null output",empty_product ? 1 : 0,0);
    bool rejected=false;
    try {
        prankk_impl::check<B,T>(h,'H','U','N','C',1,0,T(0),
            nullptr,x.d,nullptr,x.d,T(1),nullptr,x.d);
    } catch (const std::invalid_argument&) { rejected=true; }
    require_close(h,"Hermitian diagonal update requires output",rejected ? 0 : 1,0);
    for(bool column:{false,true}) {
        const int inca=column?1:x.d[DDLA_M_],len=column?n:nrhs,ia=column?1:2,ja=column?2:1,ib=column?1:6,jb=column?6:1;
        exercise(x,"PBLAS row/column swap reuse",[&]{pswap_impl::pswap_check<B>(h,len,ia,ja,x.d,inca,ib,jb,x.d,inca);},
            [&](std::size_t* bb,std::size_t* hb){*hb=0;pswap_impl::pswap_workspace_size<T>(h,x.d,bb);},
            [&](void* w,std::size_t wb,void*,std::size_t){pswap_impl::pswap_run<B,T>(h,len,x.a.ptr,ia,ja,x.d,inca,x.b.ptr,ib,jb,x.d,inca,w,wb);},
            [&]{pswap<B,T>(h,len,x.a.ptr,ia,ja,x.d,inca,x.b.ptr,ib,jb,x.d,inca);});
    }
    std::vector<int> piv(ddla_test::m_loc(h,x.d));
    for(std::size_t i=0;i<piv.size();++i){const int global=indx_l2g_r(x.d,h,int(i));piv[i]=global==0?n:global+1;}
    for(char rc:{'R','C'})
        exercise(x,"PBLAS host permutation reuse",[&]{plapiv_impl::plapiv_check<B>(h,'F',rc,'C',n,nrhs,x.d,x.d);},
            [&](std::size_t* bb,std::size_t* hb){*hb=0;plapiv_impl::plapiv_workspace_size<T>(h,nrhs,x.d,bb);},
            [&](void* w,std::size_t wb,void*,std::size_t){plapiv_impl::plapiv_run<B,T>(h,'F',rc,n,nrhs,x.a.ptr,x.d,piv.data(),x.d,w,wb);},
            [&]{plapiv<B,T>(h,'F',rc,'C',n,nrhs,x.a.ptr,x.d,piv.data(),x.d,nullptr);});
    // Scratch-free APIs deliberately have no size-query stage.
    x.reset(0);const int order=pdam_impl::pdam_check<B>(h,n,x.d);
    workspace_test::Scope pdam_scope;pdam_impl::pdam_run<B,T,T>(h,order,T(2),x.a.ptr,x.d);
    const auto pdam_snapshot=pdam_scope.finish();require_close(h,"PDAM check/run only",no_scratch(pdam_snapshot)?0:1,0);check_ddla_sync(h);
    const auto got=download(h,x.a.ptr,x.count);auto expected=x.ha;
    for(int j=0;j<ddla_test::n_loc(h,x.d);++j)for(int i=0;i<ddla_test::m_loc(h,x.d);++i)
        if(indx_l2g_r(x.d,h,i)==indx_l2g_c(x.d,h,j) && indx_l2g_r(x.d,h,i)<n)expected[i+std::size_t(j)*x.d[DDLA_LLD_]]+=T(2);
    require_close(h,"PDAM independent diagonal/padding reference",got==expected?0:1,0);
    int zero_opts_panel_width=1;const auto zero_opts=pgemm_impl::pgemm_check<B>(h,'N','N',n,n,0,true,x.d,x.d,x.d,pgemm_impl::Algorithm::AB,zero_opts_panel_width);
    exercise(x,"PBLAS zero-product no-read path",[]{},
        [&](std::size_t* bb,std::size_t* hb){pgemm_impl::pgemm_workspace_size<T>(h,zero_opts,zero_opts_panel_width,true,'N','N',n,n,0,false,x.d,x.d,x.d,bb,hb);},
        [&](void* w,std::size_t wb,void* hw,std::size_t hwb){pgemm_impl::pgemm_run<B,T,T,T>(h,zero_opts,zero_opts_panel_width,'N','N',n,n,0,T(0),nullptr,x.d,nullptr,x.d,T(0),x.c.ptr,x.d,w,wb,hw,hwb);},
        [&]{pgemm<B,T>(h,'N','N',n,n,0,T(0),nullptr,x.d,nullptr,x.d,T(0),x.c.ptr,x.d);});
    int empty_panel_width=1;const auto empty=pgemm_impl::pgemm_check<B>(h,'N','N',0,n,k,false,x.d,x.d,x.d,pgemm_impl::Algorithm::AB,empty_panel_width);
    exercise(x,"PBLAS empty global output",[]{},
        [&](std::size_t* bb,std::size_t* hb){pgemm_impl::pgemm_workspace_size<T>(h,empty,empty_panel_width,false,'N','N',0,n,k,true,x.d,x.d,x.d,bb,hb);},
        [&](void* w,std::size_t wb,void* hw,std::size_t hwb){pgemm_impl::pgemm_run<B,T,T,T>(h,empty,empty_panel_width,'N','N',0,n,k,alpha,nullptr,x.d,nullptr,x.d,beta,nullptr,x.d,w,wb,hw,hwb);},
        [&]{pgemm<B,T>(h,'N','N',0,n,k,alpha,nullptr,x.d,nullptr,x.d,beta,nullptr,x.d);});
    int tiny_panel_width=1;const auto tiny=pgemm_impl::pgemm_check<B>(h,'N','N',1,1,1,false,x.d,x.d,x.d,pgemm_impl::Algorithm::AB,tiny_panel_width);
    exercise(x,"PBLAS empty local owners",[]{},
        [&](std::size_t* bb,std::size_t* hb){pgemm_impl::pgemm_workspace_size<T>(h,tiny,tiny_panel_width,false,'N','N',1,1,1,true,x.d,x.d,x.d,bb,hb);},
        [&](void* w,std::size_t wb,void* hw,std::size_t hwb){pgemm_impl::pgemm_run<B,T>(h,tiny,tiny_panel_width,'N','N',1,1,1,alpha,x.a.ptr,x.d,x.b.ptr,x.d,beta,x.c.ptr,x.d,w,wb,hw,hwb);},
        [&]{pgemm<B,T>(h,'N','N',1,1,1,alpha,x.a.ptr,x.d,x.b.ptr,x.d,beta,x.c.ptr,x.d);});
}

template <DdlaBackend B,typename R> void mixed_cases(DdlaHandle_t h)
{
    using T=std::complex<R>;Matrices<T> x(h);x.reset(0);
    std::vector<R> real(x.count);for(std::size_t i=0;i<real.size();++i)real[i]=x.ha[i].real();
    DeviceBuffer<R> input(h,real.size());upload(h,input.ptr,real);
    const int n=11,k=7;const T alpha(R(0.7),R(0.2)),beta(R(0.3),R(-0.1));
    for(auto algorithm:{pgemm_impl::Algorithm::AB,pgemm_impl::Algorithm::AC,pgemm_impl::Algorithm::BC}) {
        int opts_panel_width=1;const auto opts=pgemm_impl::pgemm_check<B>(h,'N','N',n,n,k,false,x.d,x.d,x.d,algorithm,opts_panel_width);
        exercise(x,"PBLAS real-by-complex reuse",[]{},
            [&](std::size_t* bb,std::size_t* hb){pgemm_impl::pgemm_workspace_size<T,R,T>(h,opts,opts_panel_width,false,'N','N',n,n,k,true,x.d,x.d,x.d,bb,hb);},
            [&](void* w,std::size_t wb,void* hw,std::size_t hwb){pgemm_impl::pgemm_run<B,T,R,T>(h,opts,opts_panel_width,'N','N',n,n,k,alpha,input.ptr,x.d,x.b.ptr,x.d,beta,x.c.ptr,x.d,w,wb,hw,hwb);},
            [&]{pgemm_impl::run<B,T,R,T>(h,'N','N',n,n,k,alpha,input.ptr,x.d,x.b.ptr,x.d,beta,x.c.ptr,x.d,algorithm);});
        exercise(x,"PBLAS complex-by-real reuse",[]{},
            [&](std::size_t* bb,std::size_t* hb){pgemm_impl::pgemm_workspace_size<T,T,R>(h,opts,opts_panel_width,false,'N','N',n,n,k,true,x.d,x.d,x.d,bb,hb);},
            [&](void* w,std::size_t wb,void* hw,std::size_t hwb){pgemm_impl::pgemm_run<B,T,T,R>(h,opts,opts_panel_width,'N','N',n,n,k,alpha,x.a.ptr,x.d,input.ptr,x.d,beta,x.c.ptr,x.d,w,wb,hw,hwb);},
            [&]{pgemm_impl::run<B,T,T,R>(h,'N','N',n,n,k,alpha,x.a.ptr,x.d,input.ptr,x.d,beta,x.c.ptr,x.d,algorithm);});
    }
}

// One scratch, the larger of two families' sizes, serves a complex pgeadd
// (host schedule records) and a float ptrtrm (host pointer lists) in turn.
template <DdlaBackend B> void shared_scratch(DdlaHandle_t h)
{
    using T=std::complex<double>;Matrices<T> z(h);Matrices<float> f(h);z.reset(0);f.reset(0);
    std::size_t geadd_backend=0,geadd_host=0,trtrm_backend=0,trtrm_host=0;
    pgeadd_impl::pgeadd_workspace_size<T>(h,'C','C',true,true,11,11,z.d,&geadd_backend,&geadd_host);
    ptrtrm_impl::ptrtrm_workspace_size<float>(h,'N','N',11,f.d,f.d,f.d,&trtrm_backend,&trtrm_host);
    const std::size_t backend_bytes=std::max(geadd_backend,trtrm_backend),host_bytes=std::max(geadd_host,trtrm_host);
    GuardedScratch scratch(h,backend_bytes,host_bytes);
    for(int pass=0;pass<3;++pass) {
        if(pass%2==0) {
            z.reset(pass);pgeadd<B,T>(h,'C','C',11,11,T(0.7),z.a.ptr,1,1,z.d,T(0.3),z.b.ptr,1,1,z.d,z.c.ptr,1,1,z.d);check_ddla_sync(h);
            const auto ea=download(h,z.a.ptr,z.count),eb=download(h,z.b.ptr,z.count),ec=download(h,z.c.ptr,z.count);z.reset(pass);
            workspace_test::Scope scope;pgeadd_impl::pgeadd_run<B,T>(h,'C','C',11,11,T(0.7),z.a.ptr,1,1,z.d,T(0.3),z.b.ptr,1,1,z.d,z.c.ptr,1,1,z.d,
                scratch.backend(),backend_bytes,scratch.host(),host_bytes);
            const auto snap=scope.finish();require_close(h,"shared scratch complex upload has no scratch",no_scratch(snap)?0:1,0);check_ddla_sync(h);z.compare("shared scratch complex metadata reuse",ea,eb,ec);
        } else {
            f.reset(pass);ptrtrm<B,float>(h,'U','N','L','N',11,0.7f,f.a.ptr,f.d,f.b.ptr,f.d,0.3f,f.c.ptr,f.d);check_ddla_sync(h);
            const auto ea=download(h,f.a.ptr,f.count),eb=download(h,f.b.ptr,f.count),ec=download(h,f.c.ptr,f.count);f.reset(pass);
            workspace_test::Scope scope;ptrtrm_impl::ptrtrm_run<B,float>(h,'U','N','L','N',11,0.7f,f.a.ptr,f.d,f.b.ptr,f.d,0.3f,f.c.ptr,f.d,
                scratch.backend(),backend_bytes,scratch.host(),host_bytes);
            const auto snap=scope.finish();require_close(h,"shared scratch pointer upload has no scratch",no_scratch(snap)?0:1,0);check_ddla_sync(h);f.compare("shared scratch float pointer metadata reuse",ea,eb,ec);
        }
        scratch.verify();
    }
}

template <DdlaBackend B> void geadd_count_checks(DdlaHandle_t h)
{
    // Four large tiles describe count overflow without allocating a matrix.
    // On every multi-rank grid at least one off-diagonal tile moves remotely.
    constexpr int n=100000,nb=50000;int d[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(d,h,n,n,nb,nb,0,0));
    pgeadd_impl::pgeadd_check<B>(h,'T','T',n,n,d,1,1,d,1,1,d,1,1,false,false);
    pgeadd_impl::pgeadd_check<B>(h,'T','N',n,n,d,1,1,d,1,1,d,1,1,false,true);
    pgeadd_impl::pgeadd_check<B>(h,'N','T',n,n,d,1,1,d,1,1,d,1,1,true,false);
    workspace_test::Scope scope(true);
    std::size_t backend_bytes=1,host_bytes=1;
    pgeadd_impl::pgeadd_workspace_size<double>(h,'T','T',false,false,n,n,d,&backend_bytes,&host_bytes);
    const auto snapshot=scope.finish();
    require_close(h,"GEADD unused large inputs need no workspace",
        pure(snapshot) && host_bytes==0 && backend_bytes==0?0:1,0);
    if(h->nprocs>1) {
        bool rejected=false;
        try{pgeadd_impl::pgeadd_check<B>(h,'T','N',n,n,d,1,1,d,1,1,d,1,1,true,false);}
        catch(const std::exception&){rejected=true;}
        require_close(h,"GEADD active large input rejects MPI count overflow",rejected?0:1,0);
    }
}

template<class T>T shifted_triangle(char uplo,int i,int j,int pass)
{
    using R=typename real_type<T>::type;if(uplo=='L'?i<j:i>j)return T(0);
    if(i==j)return T(R(3)+R(i+pass)/R(23));
    const R re=R((i*7+j*3+pass*5)%17-8)/R(97);
    if constexpr(std::is_arithmetic_v<T>)return re;
    else return T(re,R((i*3+j*5+pass)%11-5)/R(131));
}
template<class T>T shifted_solution(int i,int j,int pass)
{
    using R=typename real_type<T>::type;const R re=R((i*5+j*7+pass*3)%23-11)/R(31);
    if constexpr(std::is_arithmetic_v<T>)return re;
    else return T(re,R((i+j*3+pass)%13-6)/R(37));
}
template<DdlaBackend B,class T>void shifted_triangular_cases(DdlaHandle_t h)
{
    using R=typename real_type<T>::type;constexpr int order=11,nrhs=4,nb=3;const T sentinel=T(-947);
    const int sources[2][2]={{h->nprows_-1,h->npcols_-1},{0,h->npcols_-1}};
    const int source_cases=h->nprows_==h->npcols_&&h->nprows_>1?2:1;
    for(int source=0;source<source_cases;++source)for(char side:{'L','R'})for(char uplo:{'L','U'})for(char trans:{'N','T','C'}) {
        const int m=side=='L'?order:nrhs,n=side=='L'?nrhs:order;
        int da[DDLA_DLEN_],db[DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(da,h,order+2,order+2,nb,nb,sources[source][0],sources[source][1]));
        DDLA_CHECK(ddlaDescInit(db,h,m+2,n+2,nb,nb,sources[source][0],sources[source][1]));
        da[DDLA_LLD_]+=4+h->myid%2;db[DDLA_LLD_]+=5+h->myid%3;
        DeviceBuffer<T>A(h,local_size(h,da)),rhs(h,local_size(h,db));
        ptrtrs_impl::ptrtrs_check<B>(h,side,uplo,trans,'N',m,n,da,db);
        MPI_Barrier(h->comm);bool one=true;
        if(!h->myid){workspace_test::Scope scope(true);std::size_t bb=0,hb=0;ptrtrs_impl::ptrtrs_workspace_size<T>(h,side,trans,m,n,da,db,&bb,&hb);one=pure(scope.finish());}
        MPI_Barrier(h->comm);require_close(h,"shifted PTRTRS one-rank pure query",one?0:1,0);
        std::size_t backend_bytes=0,host_bytes=0;
        workspace_test::Scope query_scope(true);ptrtrs_impl::ptrtrs_workspace_size<T>(h,side,trans,m,n,da,db,&backend_bytes,&host_bytes);const auto query=query_scope.finish();
        require_close(h,"shifted PTRTRS pure query",pure(query)?0:1,0);
        if(h->nprows_==h->npcols_&&da[DDLA_RSRC_]!=da[DDLA_CSRC_]&&trans!='N') {
            require_close(h,"shifted square PTRTRS uses the whole-slab route",host_bytes==0?0:1,0);
            std::size_t panel_backend=0,panel_host=0;
            pblas::panel_workspace_size<T>(h,side=='L',trans,side=='L'?nb:order,
                side=='L'?order:nb,da,da,&panel_backend,&panel_host);
            require_close(h,"shifted square pgeadd panel query avoids a block schedule",panel_host==0?0:1,0);
        }
        GuardedScratch workspace(h,backend_bytes,host_bytes);
        const std::string label=std::string("shifted PTRTRS analytic ")+side+uplo+trans+" scalar_bytes="+std::to_string(sizeof(T))+" sources="+std::to_string(da[DDLA_RSRC_])+","+std::to_string(da[DDLA_CSRC_]);
        for(int pass=0;pass<2;++pass) {
            auto a=make_local<T>(h,da,[&](int i,int j){if(i>=order||j>=order)return sentinel;
                if(uplo=='L'?i<j:i>j)return T(std::numeric_limits<R>::quiet_NaN());return shifted_triangle<T>(uplo,i,j,pass);});
            auto op=[&](int i,int j){T v=trans=='N'?shifted_triangle<T>(uplo,i,j,pass):shifted_triangle<T>(uplo,j,i,pass);
                if constexpr(!std::is_arithmetic_v<T>)if(trans=='C')v=std::conj(v);return v;};
            auto b=make_local<T>(h,db,[&](int i,int j){if(i>=m||j>=n)return sentinel;T total=0;
                for(int k=0;k<order;++k)total+=side=='L'?op(i,k)*shifted_solution<T>(k,j,pass):shifted_solution<T>(i,k,pass)*op(k,j);return total;});
            for(int j=0;j<ddla_test::n_loc(h,da);++j)for(int i=ddla_test::m_loc(h,da);i<da[DDLA_LLD_];++i)a[i+std::size_t(j)*da[DDLA_LLD_]]=sentinel;
            for(int j=0;j<ddla_test::n_loc(h,db);++j)for(int i=ddla_test::m_loc(h,db);i<db[DDLA_LLD_];++i)b[i+std::size_t(j)*db[DDLA_LLD_]]=sentinel;
            auto verify=[&] {
                const auto aa=download(h,A.ptr,a.size()),out=download(h,rhs.ptr,b.size());double error=0,padding=0;
                for(int j=0;j<ddla_test::n_loc(h,db);++j)for(int i=0;i<db[DDLA_LLD_];++i) {
                    const int gi=i<ddla_test::m_loc(h,db)?indx_l2g_r(db,h,i):m,gj=indx_l2g_c(db,h,j);const auto at=i+std::size_t(j)*db[DDLA_LLD_];
                    if(gi<m&&gj<n)error=std::max(error,workspace_test::relative_error(out[at],shifted_solution<T>(gi,gj,pass)));
                    else if(out[at]!=b[at])padding=1;
                }
                require_close(h,label,error,sizeof(R)==4?2e-5:2e-12);
                require_close(h,"shifted PTRTRS parent and LLD padding",padding,0);
                require_close(h,"shifted PTRTRS immutable triangle and parent",a.empty()?0:std::memcmp(aa.data(),a.data(),a.size()*sizeof(T))?1:0,0);
            };
            upload(h,A.ptr,a);upload(h,rhs.ptr,b);check_ddla_sync(h);
            workspace_test::Scope run_scope;ptrtrs_impl::ptrtrs_run<B,T>(h,side,uplo,trans,'N',m,n,A.ptr,da,rhs.ptr,db,workspace.backend(),backend_bytes,workspace.host(),host_bytes);const auto run=run_scope.finish();
            require_close(h,"shifted PTRTRS run has no allocation/query/setup",no_scratch(run)?0:1,0);verify();workspace.verify();
            upload(h,A.ptr,a);upload(h,rhs.ptr,b);ptrtrs<B,T>(h,side,uplo,trans,'N',m,n,A.ptr,da,rhs.ptr,db);verify();
        }
    }
}

// Exercise exact scratch and repeated no-allocation execution across both
// new group boundaries without enlarging every existing small PBLAS case.
template <DdlaBackend B,typename T> void panel_boundaries(DdlaHandle_t h)
{
    Matrices<T> x(h,1040,32);const T alpha=T(.7),beta=T(.3);
    for(bool general:{false,true}) {
        const int n=general ? 1031 : 17,k=general ? 3 : 1031;
        exercise(x,"PBLAS grouped triangular workspace boundary",[]{},
            [&](std::size_t* bb,std::size_t* hb){if(general)pgemmt_impl::workspace_size<T>(h,'U','N','T',true,n,k,x.d,x.d,x.d,bb,hb);
                else prankk_impl::workspace_size<T>(h,'S','U','N','T',true,false,n,k,x.d,x.d,x.d,bb,hb);},
            [&](void* w,std::size_t wb,void* hw,std::size_t hwb){if(general)pgemmt_impl::run<B,T>(h,'U','N','T',true,n,k,alpha,x.a.ptr,x.d,x.a.ptr,x.d,beta,x.c.ptr,x.d,w,wb,hw,hwb);
                else prankk_impl::run<B,T>(h,'S','U','N','T',true,false,n,k,alpha,x.a.ptr,x.d,x.a.ptr,x.d,beta,x.c.ptr,x.d,w,wb,hw,hwb);},
            [&]{if(general)pgemmt<B,T>(h,'U','N','T',n,k,alpha,x.a.ptr,x.d,x.a.ptr,x.d,beta,x.c.ptr,x.d);
                else psyrk<B,T>(h,'U','N',n,k,alpha,x.a.ptr,x.d,beta,x.c.ptr,x.d);});
    }
    for(int n:{1024,1031})for(char side:{'L','R'})
        exercise(x,"PBLAS grouped multiply workspace boundary",[]{},
            [&](std::size_t* bb,std::size_t* hb){phemm_impl::phemm_workspace_size<T>(h,side,n,n,false,x.d,x.d,x.d,true,bb,hb);},
            [&](void* w,std::size_t wb,void* hw,std::size_t hwb){phemm_impl::phemm_run<B,T>(h,side,'U',n,n,alpha,x.a.ptr,x.d,x.b.ptr,x.d,beta,x.c.ptr,x.d,true,w,wb,hw,hwb);},
            [&]{phemm<B,T>(h,side,'U',n,n,alpha,x.a.ptr,x.d,x.b.ptr,x.d,beta,x.c.ptr,x.d);});
}

template <DdlaBackend B> void run(MPI_Comm comm,int pr,int pc,char major)
{
    DdlaHandle_t h=nullptr;ddlaInit(h,B);ddlaSet(h,comm,pr,pc,major);
    geadd_count_checks<B>(h);
    panel_boundaries<B,float>(h);panel_boundaries<B,double>(h);
    panel_boundaries<B,std::complex<float>>(h);panel_boundaries<B,std::complex<double>>(h);
    cases<B,float>(h);cases<B,double>(h);cases<B,std::complex<float>>(h);cases<B,std::complex<double>>(h);
    mixed_cases<B,float>(h);mixed_cases<B,double>(h);shared_scratch<B>(h);
    shifted_triangular_cases<B,float>(h);shifted_triangular_cases<B,double>(h);
    shifted_triangular_cases<B,std::complex<float>>(h);shifted_triangular_cases<B,std::complex<double>>(h);
    if(h->myid==0)std::cout<<"workspace PBLAS PASS backend="<<int(B)<<" grid="<<pr<<"x"<<pc<<" major="<<major<<std::endl;
    ddlaDestroy(h);
}
}
int main(int argc,char** argv)
{
    MPI_Init(&argc,&argv);int size;MPI_Comm_size(MPI_COMM_WORLD,&size);
    TestOptions opts;std::string error;
    if(!parse_options(argc,argv,size,opts,error)){std::cerr<<error<<std::endl;MPI_Abort(MPI_COMM_WORLD,2);}
    for(auto grid:opts.grids)for(char major:{'R','C'}) {
        run<DdlaBackend::CPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#if DDLA_HAS_GPU
        run<DdlaBackend::GPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#endif
    }
    MPI_Finalize();return 0;
}
