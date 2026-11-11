#include "api_grid_test_common.h"
#include "ddla_stream.h"
#include "workspace_test_common.h"
#include "SVD/svd_workspace.h"
#include "Eigensolver/eig_workspace.h"
#include "Eigensolver/eig_cpu_kernels.h"
#include "Eigensolver/pvector_redist.h"

#include <array>
#include <cstring>
#include <limits>
#include <type_traits>

using namespace api_grid_test;
namespace {
using namespace ddla::spectral_impl;

template<class T> bool same(const std::vector<T>& a,const std::vector<T>& b)
{return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);}
template<class T> T value(int i,int j)
{
    using R=typename real_type<T>::type;
    const R re=R(((i+3)*17+(j+7)*31+(i*j+1)*13)%101-50)/R(101);
    if constexpr(std::is_same<T,R>::value)return re;
    else return T(re,R(((i+2)*11+(j+1)*7)%23-11)/R(29));
}
template<class T>T hermitian(int i,int j)
{
    if(i==j)return T(3+i);
    if(i>j)return value<T>(i,j);
    if constexpr(std::is_same<T,typename real_type<T>::type>::value)return value<T>(j,i);
    else return std::conj(value<T>(j,i));
}

// Exactly the sized bytes of both kinds between guards; one allocation serves
// every run of a case.
class GuardedScratch {
    static constexpr std::size_t guard=256;
    DdlaHandle_t h_;std::size_t backend_bytes_,host_bytes_;ddla::vector<char> backend_;std::vector<char> host_;
public:
    GuardedScratch(DdlaHandle_t h,std::size_t backend_bytes,std::size_t host_bytes)
        :h_(h),backend_bytes_(backend_bytes),host_bytes_(host_bytes),
         backend_(h,backend_bytes+2*guard),host_(host_bytes+2*guard,char(0x5a))
    {DDLA_CHECK(ddlaMemset(h,backend_.data(),0x5a,backend_.size()));}
    void* backend(){return backend_.data()+guard;}
    void* host(){return host_.data()+guard;}
    void verify()
    {
        std::array<char,2*guard> data{};
        DDLA_CHECK(ddlaMemcpy(h_,data.data(),backend_.data(),guard));
        DDLA_CHECK(ddlaMemcpy(h_,data.data()+guard,backend_.data()+guard+backend_bytes_,guard));
        DDLA_CHECK(ddlaSynchronize(h_));bool good=true;
        for(std::size_t i=0;i<guard;++i)good=good&&host_[i]==char(0x5a)&&
            host_[guard+host_bytes_+i]==char(0x5a)&&data[i]==char(0x5a)&&data[guard+i]==char(0x5a);
        require_close(h_,"spectral exact-size workspace guards",good?0:1,0);
    }
};

template<class S>bool no_allocation(const S& s)
{
    return s.ddla_alloc==0&&s.ddla_free==0&&s.pinned_alloc==0&&s.raw_gpu_alloc==0&&
        s.host_new==0&&s.vendor_query==0&&s.resource_setup==0;
}
// GPU messages staged in the handle's host buffers (tunnel builds, and the
// all-to-all of builds without CCL; comm_traits.h) grow them on first use:
// there the first run of a case may allocate.
template<DdlaBackend B>bool may_grow_staging(int pass)
{
#if defined(DDLA_USE_GPU_CPU_TUNNEL) || !defined(DDLA_USE_CCL)
    return B==DdlaBackend::GPU&&pass==0;
#else
    (void)pass;return false;
#endif
}
template<class F>void measured_run(DdlaHandle_t h,F&& function,bool may_allocate=false)
{
    workspace_test::Scope scope;function();const auto snapshot=scope.finish();
    require_close(h,"spectral run performs no managed allocation or vendor query",may_allocate||no_allocation(snapshot)?0:1,0);
}
// function writes this rank's byte counts; rank 0 alone first proves the query local.
template<class F> void local_query(DdlaHandle_t h,F&& function)
{
    MPI_Barrier(h->comm);bool pure=true;
    if(h->myid==0) {
        workspace_test::Scope scope(true);function();const auto s=scope.finish();
        pure=no_allocation(s)&&s.copies==0&&s.launches==0&&s.syncs==0&&s.collectives==0;
    }
    MPI_Barrier(h->comm);
    require_close(h,"spectral one-rank query performs no allocation or execution",pure?0:1,0);
    function();
}

template<DdlaBackend B,class T>
void brd_case(DdlaHandle_t h,int m,int n)
{
    using R=typename real_type<T>::type;const int k=std::min(m,n);
    int da[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(da,h,m+3,n+2,4,4,0,0));da[DDLA_LLD_]+=3;
    auto original=make_local<T>(h,da,[](int i,int j){return value<T>(i,j);});
    DeviceBuffer<T> A(h,original.size()),reference(h,original.size());upload(h,reference.ptr,original);
    std::vector<R> rd(std::max(k,1),R(-13)),re(std::max(k-1,1),R(-13));
    std::vector<T> rq(std::max(k,1),T(-13)),rp(rq),q(rq),p(rq);
    int info=0;pgebrd<B,T>(h,m,n,reference.ptr,da,rd.data(),re.data(),rq.data(),rp.data(),info);
    const auto expected=download(h,reference.ptr,original.size());
    std::vector<R>d(rd.size(),R(-13)),e(re.size(),R(-13));
    const auto plan=pgebrd_check(h,B,m,n,A.ptr,da,d.data(),e.data(),q.data(),p.data());
    std::size_t bytes=0;local_query(h,[&]{pgebrd_workspace_size<T>(h,plan,&bytes);});
    GuardedScratch work(h,bytes,0);
    for(int pass=0;pass<2;++pass) {
        upload(h,A.ptr,original);
        measured_run(h,[&]{pgebrd_run<B,T>(h,plan,A.ptr,da,d.data(),e.data(),q.data(),p.data(),info,work.backend(),bytes);},
            may_grow_staging<B>(pass));
        require_close(h,"borrowed pgebrd info",info,0);
        require_close(h,"borrowed pgebrd factors and padding",same(download(h,A.ptr,original.size()),expected)?0:1,0);
        require_close(h,"borrowed pgebrd scalar outputs",same(d,rd)&&same(e,re)&&same(q,rq)&&same(p,rp)?0:1,0);
        work.verify();
    }
    for(char vect:{'Q','P'})for(char trans:{'N','C'}) {
        const int nq=vect=='Q'?m:n,reflector_k=vect=='Q'?n:m;const T* tau=vect=='Q'?q.data():p.data();
        int dc[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(dc,h,nq+2,nq+3,4,4,0,vect=='P'?0:h->npcols_-1));dc[DDLA_LLD_]+=2;
        auto input=make_local<T>(h,dc,[](int i,int j){return i==j?T(1):T(0);});
        DeviceBuffer<T>C(h,input.size()),Cr(h,input.size());upload(h,Cr.ptr,input);
        punmbr<B,T>(h,vect,vect=='Q'?'L':'R',trans,nq,nq,reflector_k,A.ptr,da,tau,Cr.ptr,dc,info);
        const auto expect=download(h,Cr.ptr,input.size());
        const auto cp=punmbr_check(h,B,std::is_same<T,R>::value,vect,vect=='Q'?'L':'R',trans,nq,nq,reflector_k,A.ptr,da,tau,C.ptr,dc);
        std::size_t cb=0,ch=0;local_query(h,[&]{punmbr_workspace_size<T>(h,cp,&cb,&ch);});GuardedScratch cw(h,cb,ch);
        for(int pass=0;pass<2;++pass) {
            upload(h,C.ptr,input);
            measured_run(h,[&]{punmbr_run<B,T>(h,cp,A.ptr,da,tau,C.ptr,dc,info,cw.backend(),cb,cw.host(),ch);},may_grow_staging<B>(pass));
            require_close(h,"borrowed punmbr result and padding",same(download(h,C.ptr,input.size()),expect)?0:1,0);cw.verify();
        }
    }
}

template<DdlaBackend B,class T>
void hermitian_case(DdlaHandle_t h,int n,char uplo)
{
    int da[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(da,h,n+3,n+2,4,4,0,0));da[DDLA_LLD_]+=3;
    const T guard=T(-733);
    auto original=make_local<T>(h,da,[&](int i,int j){
        return i<n&&j<n&&(uplo=='U'?i<=j:i>=j)?hermitian<T>(i,j):guard;});
    auto expected=original;
    for(int j=0;j<ddla_test::n_loc(h,da);++j)for(int i=0;i<ddla_test::m_loc(h,da);++i) {
        const int r=indx_l2g_r(da,h,i),c=indx_l2g_c(da,h,j);
        if(r<n&&c<n)expected[i+std::size_t(j)*da[DDLA_LLD_]]=hermitian<T>(r,c);
    }
    DeviceBuffer<T>A(h,original.size());
    const auto p=pherm_complete_check(h,B,uplo,n,A.ptr,da);
    std::size_t bytes=0,host_bytes=0;local_query(h,[&]{pherm_complete_workspace_size<T>(h,p,0,&bytes,&host_bytes);});
    GuardedScratch work(h,bytes,host_bytes);
    for(int pass=0;pass<2;++pass) {
        upload(h,A.ptr,original);
        measured_run(h,[&]{pherm_complete_run<B,T>(h,p,A.ptr,nullptr,0,work.backend(),bytes,work.host(),host_bytes);},
            may_grow_staging<B>(pass));
        require_close(h,"borrowed Hermitian completion exact values and padding",same(download(h,A.ptr,original.size()),expected)?0:1,0);work.verify();
    }
    if(n) {
        std::size_t pack,local,unpack,diag,send_total,recv_total;
        pherm_complete_geometry(p,pack,local,unpack,diag,send_total,recv_total);
        const auto cells=send_total+recv_total;
        DeviceBuffer<T>scratch(h,cells+2);std::vector<T>poison(cells+2,T(-29));upload(h,scratch.ptr,poison);
        std::size_t lent=0,lent_host=0;local_query(h,[&]{pherm_complete_workspace_size<T>(h,p,cells,&lent,&lent_host);});
        GuardedScratch bw(h,lent,lent_host);
        for(int pass=0;pass<2;++pass) {
            upload(h,A.ptr,original);
            measured_run(h,[&]{pherm_complete_run<B,T>(h,p,A.ptr,scratch.ptr+1,cells,bw.backend(),lent,bw.host(),lent_host);});
            require_close(h,"borrowed completion output scratch result",same(download(h,A.ptr,original.size()),expected)?0:1,0);
            const auto after=download(h,scratch.ptr,cells+2);
            require_close(h,"borrowed completion output scratch guards",after.front()==T(-29)&&after.back()==T(-29)?0:1,0);bw.verify();
        }
    }
    if(!n)return;
    constexpr int nbw=2;const int steps=(n-1)/nbw;
    DeviceBuffer<T>tmat(h,std::size_t(nbw)*nbw*std::max(steps,1));int info=0;
    phe2hb<B,T>(h,n,A.ptr,da,nbw,tmat.ptr,info);
    int dz[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(dz,h,n+2,n+3,4,4,0,h->npcols_-1));dz[DDLA_LLD_]+=2;
    const auto identity=make_local<T>(h,dz,[&](int i,int j){return i<n&&j<n?T(i==j):guard;});
    DeviceBuffer<T>Z(h,identity.size()),Zr(h,identity.size());
    for(char trans:{'N','C'}) {
        upload(h,Zr.ptr,identity);punmtr_he2hb<B,T>(h,trans,n,nbw,A.ptr,da,tmat.ptr,Zr.ptr,dz,info);
        const auto expect=download(h,Zr.ptr,identity.size());
        const auto up=punmtr_he2hb_check(h,B,trans,n,nbw,A.ptr,da,tmat.ptr,Z.ptr,dz);
        std::size_t ub=0;local_query(h,[&]{punmtr_he2hb_workspace_size<T>(h,up,&ub);});GuardedScratch uw(h,ub,0);
        for(int pass=0;pass<2;++pass) {
            upload(h,Z.ptr,identity);
            measured_run(h,[&]{punmtr_he2hb_run<B,T>(h,up,A.ptr,da,tmat.ptr,Z.ptr,dz,info,uw.backend(),ub);},may_grow_staging<B>(pass));
            require_close(h,"borrowed band back-transform and padding",same(download(h,Z.ptr,identity.size()),expect)?0:1,0);uw.verify();
        }
    }
}

template<DdlaBackend B,class T>
void vector_case(DdlaHandle_t h,int n,bool reverse,bool padded=true)
{
    auto vector_value=[](int i,int j) {
        if constexpr(std::is_same<T,typename real_type<T>::type>::value)return T(1+i+31*j);
        else return T(1+i+31*j,2+3*i-7*j);
    };
    const int nb=4,nvc=3;const int spr=reverse?h->npcols_:h->nprows_,spc=reverse?h->nprows_:h->npcols_;
    const int sr=reverse?h->mypcol_:h->myprow_,sc=reverse?h->myprow_:h->mypcol_;
    const int rows=num_loc(n,nb,sr,0,spr),cols=num_loc(n,nb,sc,0,spc);
    // In particular, peers in the SAME process row have different source
    // strides. NaN padding must never participate in their reduction.
    const int ls=std::max(rows,1)+(padded?2+3*h->mypcol_:0),ld=std::max(cols,1)+3+h->myid;
    const T guard=T(-911-h->myid);
    using R=typename real_type<T>::type;
    std::vector<T>input(std::size_t(ls)*nvc,T(std::numeric_limits<R>::quiet_NaN()));
    std::vector<T>initial(std::size_t(ld)*nvc,guard),expected(initial);
    for(int j=0;j<nvc;++j){for(int i=0;i<rows;++i)input[i+std::size_t(j)*ls]=vector_value(indxl2g(i,nb,sr,0,spr),j);
        for(int i=0;i<cols;++i)expected[i+std::size_t(j)*ld]=vector_value(indxl2g(i,nb,sc,0,spc),j);}
    DeviceBuffer<T>src(h,input.size()),dst(h,initial.size());upload(h,src.ptr,input);
    const auto p=ptranspose_vectors_check(h,B,n,nvc,nb,ls,ld,false,reverse,src.ptr,dst.ptr);
    std::size_t bytes=0;local_query(h,[&]{ptranspose_vectors_workspace_size<T>(h,n,nvc,nb,&bytes);});GuardedScratch work(h,bytes,0);
    for(int pass=0;pass<2;++pass){upload(h,dst.ptr,initial);
        measured_run(h,[&]{ptranspose_vectors_run<B,T>(h,p,src.ptr,dst.ptr,work.backend(),bytes);},may_grow_staging<B>(pass));
        require_close(h,"borrowed vector transpose exact values and padding",same(download(h,dst.ptr,initial.size()),expected)?0:1,0);work.verify();}
    if(!reverse) {
        const auto rp=ptranspose_vectors_check(h,B,n,nvc,nb,ls,ld,true,false,src.ptr,dst.ptr);
        std::size_t rb=0;local_query(h,[&]{preduce_add_vectors_workspace_size<T>(h,n,nvc,nb,ls,&rb);});
        GuardedScratch rw(h,rb,0);auto sum_expected=initial;
        for(int j=0;j<nvc;++j)for(int i=0;i<cols;++i)
            sum_expected[i+std::size_t(j)*ld]+=T(h->npcols_)*vector_value(indxl2g(i,nb,sc,0,spc),j);
        for(int pass=0;pass<2;++pass) {
            upload(h,dst.ptr,initial);
            measured_run(h,[&]{preduce_add_vectors_run<B,T>(h,rp,src.ptr,dst.ptr,rw.backend(),rb);},may_grow_staging<B>(pass));
            require_close(h,"borrowed vector reduce-add with rank-varying strides and NaN source padding",
                same(download(h,dst.ptr,initial.size()),sum_expected)?0:1,0);rw.verify();
            require_close(h,"borrowed vector reduce-add preserves source including padding",
                same(download(h,src.ptr,input.size()),input)?0:1,0);
        }
    }

}

template<class F>void rejected_check(DdlaHandle_t h,const char* label,F&& function)
{
    int caught=0;
    try{function();}catch(const std::exception&){caught=1;}
    require_close(h,label,caught?0:1,0);
}

template<DdlaBackend B>void checker_cases(DdlaHandle_t h)
{
    int da[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(da,h,9,9,2,2,0,0));
    // Both LLD and vector strides are rank-local and may legitimately differ.
    da[DDLA_LLD_]+=h->myid+2;
    DeviceBuffer<double>A(h,local_size(h,da)),C(h,local_size(h,da)),tau_device(h,64);
    std::vector<double>original(A.count,-713),d(9),e(9),tau(9,0);
    upload(h,A.ptr,original);upload(h,C.ptr,original);
    const int faulty=h->nprocs-1;
    if constexpr(B==DdlaBackend::CPU) {
        // A prior handle on a subset must not change participation in the
        // world checker or its independently configured worker budget.
        if(h->myid==0) {
            DdlaHandle_t previous=nullptr;ddlaInit(previous,DdlaBackend::CPU);
            ddlaSet(previous,MPI_COMM_SELF,'R');ddlaDestroy(previous);
        }
    }
    const auto hp=pherm_complete_check(h,B,'U',7,A.ptr,da);
    require_close(h,"completion checker freezes a valid local thread budget",hp.threads>=1?0:1,0);
    (void)pgebrd_check(h,B,7,5,A.ptr,da,d.data(),e.data(),tau.data(),tau.data());
    (void)punmbr_check(h,B,true,'Q','L','N',7,7,5,A.ptr,da,tau.data(),C.ptr,da);
    (void)punmtr_he2hb_check(h,B,'N',7,2,A.ptr,da,tau_device.ptr,C.ptr,da);
    (void)ptranspose_vectors_check(h,B,7,3,2,16+h->myid,16+h->myid,false,false,A.ptr,C.ptr);
    rejected_check(h,"BRD rank-local invalid descriptor is rejected everywhere",[&] {
        (void)pgebrd_check(h,B,7,5,A.ptr,h->myid==faulty?nullptr:da,d.data(),e.data(),tau.data(),tau.data());
    });
    rejected_check(h,"BRD rank-local missing scalar output is rejected everywhere",[&] {
        (void)pgebrd_check(h,B,7,5,A.ptr,da,h->myid==faulty?nullptr:d.data(),e.data(),tau.data(),tau.data());
    });
    rejected_check(h,"UNMBR rank-local invalid flag is rejected everywhere",[&] {
        (void)punmbr_check(h,B,true,'Q','L',h->myid==faulty?'X':'N',7,7,5,A.ptr,da,tau.data(),C.ptr,da);
    });
    // Stage5 argument validation is local. Peers with valid flags must
    // return their valid plans rather than inherit another rank's exception.
    int band_caught=0;
    UnmtrHe2hbPlan band_plan{};
    try {
        band_plan=punmtr_he2hb_check(h,B,h->myid==faulty?'X':'N',7,2,A.ptr,da,tau_device.ptr,C.ptr,da);
    } catch(const std::exception&) {
        band_caught=1;
    }
    const bool band_local_result=band_caught==(h->myid==faulty?1:0)&&
        (h->myid==faulty||(band_plan.backend==B&&band_plan.trans=='N'&&band_plan.n==7&&band_plan.nbw==2));
    require_close(h,"band back-transform rejects only the locally invalid flag",band_local_result?0:1,0);
    rejected_check(h,"completion rank-local invalid flag is rejected everywhere",[&] {
        (void)pherm_complete_check(h,B,h->myid==faulty?'X':'U',7,A.ptr,da);
    });
    rejected_check(h,"vector rank-local negative extent is rejected everywhere",[&] {
        (void)ptranspose_vectors_check(h,B,h->myid==faulty?-1:7,3,2,16,16,false,false,A.ptr,C.ptr);
    });
    if(h->nprocs>1) {
        const bool fault=h->myid==faulty;
        rejected_check(h,"BRD valid-but-different extents are rejected",[&] {
            (void)pgebrd_check(h,B,7,fault?6:5,A.ptr,da,d.data(),e.data(),tau.data(),tau.data());
        });
        rejected_check(h,"BRD empty branches still agree metadata",[&] {
            (void)pgebrd_check(h,B,0,fault?1:0,nullptr,da,nullptr,nullptr,nullptr,nullptr);
        });
        int different[DDLA_DLEN_];std::copy_n(da,DDLA_DLEN_,different);if(fault)++different[DDLA_N_];
        rejected_check(h,"BRD valid-but-different parent descriptors are rejected",[&] {
            (void)pgebrd_check(h,B,7,5,A.ptr,different,d.data(),e.data(),tau.data(),tau.data());
        });
        rejected_check(h,"UNMBR valid-but-different transform flags are rejected",[&] {
            (void)punmbr_check(h,B,true,'Q','L',fault?'C':'N',7,7,5,A.ptr,da,tau.data(),C.ptr,da);
        });
        // Consistent collective execution arguments are the caller's duty.
        // Check each valid local width here, but never execute these mismatched plans.
        const int local_width=fault?1:2;
        int width_caught=0;
        UnmtrHe2hbPlan width_plan{};
        try {
            width_plan=punmtr_he2hb_check(h,B,'N',7,local_width,A.ptr,da,tau_device.ptr,C.ptr,da);
        } catch(const std::exception&) {
            width_caught=1;
        }
        const bool local_width_valid=!width_caught&&width_plan.backend==B&&width_plan.trans=='N'&&
            width_plan.n==7&&width_plan.nbw==local_width&&width_plan.nsteps==6/local_width;
        require_close(h,"band back-transform validates each local width without agreement",local_width_valid?0:1,0);
        rejected_check(h,"completion valid-but-different triangles are rejected",[&] {
            (void)pherm_complete_check(h,B,fault?'L':'U',7,A.ptr,da);
        });
        rejected_check(h,"completion empty branches still agree metadata",[&] {
            (void)pherm_complete_check(h,B,'U',fault?1:0,A.ptr,da);
        });
        rejected_check(h,"vector valid-but-different schedules are rejected",[&] {
            (void)ptranspose_vectors_check(h,B,7,3,2,16,16,fault,false,A.ptr,C.ptr);
        });
        rejected_check(h,"vector empty branches still agree metadata",[&] {
            (void)ptranspose_vectors_check(h,B,0,fault?1:0,2,16,16,false,false,nullptr,nullptr);
        });
    }
    require_close(h,"all checker failures leave the input matrix untouched",same(download(h,A.ptr,original.size()),original)?0:1,0);
    require_close(h,"all checker failures leave the output matrix untouched",same(download(h,C.ptr,original.size()),original)?0:1,0);
}

// Optional controls live solely in the diagnostic preload. Resolve them
// before the measured region; production LibDDLA exposes no fault hook.
struct CopyFaultApi {
    using Arm=void(*)(const void*,uint64_t);
    using Count=uint64_t(*)();
    Arm arm=nullptr;Count faults=nullptr;
    CopyFaultApi() {
        void* a=dlsym(RTLD_DEFAULT,"workspace_trace_fail_memcpy_after_upload");
        void* c=dlsym(RTLD_DEFAULT,"workspace_trace_memcpy_faults");
        static_assert(sizeof(a)==sizeof(arm)&&sizeof(c)==sizeof(faults),"dlsym function pointer ABI");
        std::memcpy(&arm,&a,sizeof(arm));std::memcpy(&faults,&c,sizeof(faults));
    }
};

template<DdlaBackend B>void metadata_failure_case(DdlaHandle_t h)
{
    if constexpr(B==DdlaBackend::GPU) {
        if(!workspace_test::available()) {
            if(!h->myid)std::cout<<"GPU metadata upload failure: not tested without diagnostic preload"<<std::endl;
            return;
        }
        CopyFaultApi api;
        require_close(h,"metadata upload one-shot fault controls are available",api.arm&&api.faults?0:1,0);
        constexpr int n=13;
        int desc[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(desc,h,n,n,4,4,0,0));
        auto input=make_local<double>(h,desc,[](int i,int j){return hermitian<double>(i,j);});
        DeviceBuffer<double>A(h,input.size());upload(h,A.ptr,input);
        const auto p=pherm_complete_check(h,B,'U',n,A.ptr,desc);
        std::size_t backend_bytes=0,host_bytes=0;
        local_query(h,[&]{pherm_complete_workspace_size<double>(h,p,0,&backend_bytes,&host_bytes);});
        ddla::vector<char> backend(h,backend_bytes);std::vector<char> host(host_bytes);
        std::size_t pack,local,unpack,diag,send_total,recv_total;
        pherm_complete_geometry(p,pack,local,unpack,diag,send_total,recv_total);
        const auto blocks=pack+local+unpack+diag;
        require_close(h,"fault rank owns upload metadata",h->myid==0&&blocks==0?1:0,0);
        // The block descriptors lead the host scratch (eig_workspace.h).
        if(!h->myid)api.arm(host.data(),blocks*sizeof(HermitianBlock));
        int caught=0;bool original_error=true;
        workspace_test::Scope scope;
        try{pherm_complete_run<B,double>(h,p,A.ptr,nullptr,0,backend.data(),backend_bytes,host.data(),host_bytes);}
        catch(const std::exception& error) {
            caught=1;
            // Rank zero must retain the original synthetic DDLA error;
            // peers receive the existing coordinated preparation failure.
            if(!h->myid)original_error=std::strstr(error.what(),"ddla error 5")!=nullptr;
        }
        const auto snapshot=scope.finish();
        const auto fired=api.faults();
        if(!h->myid)api.arm(nullptr,0);
        require_close(h,"metadata upload failure reaches every rank",caught?0:1,0);
        require_close(h,"metadata upload fault fires exactly once",fired==(!h->myid?1u:0u)?0:1,0);
        require_close(h,"metadata upload failure preserves the original exception",original_error?0:1,0);
        require_close(h,"metadata upload failure allocates no managed scratch",
            snapshot.ddla_alloc==0&&snapshot.ddla_free==0&&snapshot.pinned_alloc==0&&snapshot.raw_gpu_alloc==0&&
            snapshot.vendor_query==0&&snapshot.resource_setup==0?0:1,0);
    }
}

template<DdlaBackend B>void run(MPI_Comm comm,int pr,int pc,char major)
{
    DdlaHandle_t h=nullptr;ddlaInit(h,B);ddlaSet(h,comm,pr,pc,major);
    checker_cases<B>(h);
    metadata_failure_case<B>(h);
#define CASES(T) \
    brd_case<B,T>(h,13,9);brd_case<B,T>(h,9,13);brd_case<B,T>(h,1,1);brd_case<B,T>(h,0,0); \
    hermitian_case<B,T>(h,13,'U');hermitian_case<B,T>(h,13,'L');hermitian_case<B,T>(h,1,'U');hermitian_case<B,T>(h,0,'L'); \
    vector_case<B,T>(h,13,false);vector_case<B,T>(h,13,true);vector_case<B,T>(h,1,false);vector_case<B,T>(h,0,true); \
    vector_case<B,T>(h,13,false,false);vector_case<B,T>(h,1,false,false)
    CASES(float);CASES(double);CASES(std::complex<float>);CASES(std::complex<double>);
#undef CASES
    if(!h->myid)std::cout<<"workspace spectral PASS backend="<<int(B)<<" grid="<<pr<<"x"<<pc<<" major="<<major<<" allocation_trace="<<(workspace_test::available()?"enabled":"numerical-only")<<std::endl;
    ddlaDestroy(h);
}
}
int main(int argc,char** argv)
{
    ddla_test_mpi_init(&argc,&argv);int size=1;MPI_Comm_size(MPI_COMM_WORLD,&size);TestOptions options;std::string error;
    if(!parse_options(argc,argv,size,options,error)){std::cerr<<error<<std::endl;MPI_Abort(MPI_COMM_WORLD,2);}
    for(auto grid:options.grids)for(char major:{'R','C'}) {
        run<DdlaBackend::CPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#if DDLA_HAS_GPU
        run<DdlaBackend::GPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#endif
    }
    MPI_Finalize();return 0;
}
