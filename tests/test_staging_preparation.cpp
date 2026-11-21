// Focused cold-reservation and allocator-failure tests. GPU runs require the
// existing libpblas_alloc_failure.so preload, extended with native MPI traces.
#include "api_grid_test_common.h"
#include "matrix_test_common.h"
#include "PBLAS/pgemm_impl.h"
#include "PBLAS/pblas_workspace.h"
#include "PBLAS/pgeadd_impl.h"
#include "PBLAS/ptrtrs_impl.h"
#include "Eigensolver/hegst_impl.h"
#include "Eigensolver/phegvd_workspace.h"
#include "Eigensolver/chebyshev_impl.h"
#include "Eigensolver/pvector_redist.h"
#include "Eigensolver/one_stage/punmtr_trd_impl.h"
#include "SVD/pgesvd_workspace.h"
#include "QR/pgeqrf_impl.h"
#include "matrix_transfer.h"
#include "row_share.h"
#include "generalized_eigen_test_common.h"
#include "workspace_test_common.h"
#include "check_agree.h"
#include "ddla_stream.h"
#include "staging_preparation_trace_api.h"
#include "allocation_attribution_probe_api.h"
#include <dlfcn.h>
#include <cstring>
#include <array>
#include <iostream>
#include <functional>
using namespace ddla;
using namespace api_grid_test;
namespace {
void require_all(bool okay,const char* message)
{
    int bad=!okay;MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    if(bad)throw std::runtime_error(message);
}
template<class F> F symbol(const char* name)
{
    void* address=dlsym(RTLD_DEFAULT,name);F result=nullptr;
    static_assert(sizeof(result)==sizeof(address),"dlsym ABI");std::memcpy(&result,&address,sizeof(result));
    if(!result)throw std::runtime_error(std::string("staging test preload symbol missing: ")+name);
    return result;
}
struct Trace {
    int (*prewarm)()=nullptr;
    void (*arm)(std::size_t,unsigned)=nullptr;
    void (*disarm)()=nullptr;
    unsigned (*fired)()=nullptr;
    std::size_t (*failed_bytes)()=nullptr;
    void (*begin)(void*,void*)=nullptr;
    void (*end)(StagingPreparationTrace*)=nullptr;
    void (*dump)(int,const char*)=nullptr;
    void (*allocation_dump)(int,const char*)=nullptr;
    void (*slot_getter)(void*,void(*)(void*,void**,void**))=nullptr;
    explicit Trace(bool enabled) {
        if(!enabled)return;
        prewarm=symbol<decltype(prewarm)>("pblas_alloc_failure_prewarm");
        arm=symbol<decltype(arm)>("pblas_alloc_failure_arm_host_bytes");
        disarm=symbol<decltype(disarm)>("pblas_alloc_failure_disarm");
        fired=symbol<decltype(fired)>("pblas_alloc_failure_fired");
        failed_bytes=symbol<decltype(failed_bytes)>("pblas_alloc_failure_bytes");
        begin=symbol<decltype(begin)>("staging_preparation_trace_begin");
        end=symbol<decltype(end)>("staging_preparation_trace_end");
        dump=symbol<decltype(dump)>("staging_allocation_probe_dump");
        allocation_dump=symbol<decltype(allocation_dump)>("staging_preparation_allocation_dump");
        slot_getter=symbol<decltype(slot_getter)>("staging_preparation_trace_set_slot_getter");
        require_all(prewarm()==1,"allocator observer prewarm failed");
    }
    void refresh_slots(DdlaHandle_t h) const {
        slot_getter(h,[](void* context,void** a,void** b){
            const auto handle=static_cast<DdlaHandle_t>(context);
            const auto slots=handle->host_buffers<std::byte>(0,0);*a=slots[0];*b=slots[1];
        });
    }
};
DdlaCommunication transport=DdlaCommunication::HostStagedMPI;
char requested_major='R';
DdlaHandle_t create(DdlaBackend backend,int pr,int pc)
{
    DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,backend));
    if(backend==DdlaBackend::GPU)DDLA_CHECK(ddlaSetCommunication(h,transport));
    DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,pr,pc,requested_major));return h;
}
void cold(DdlaHandle_t h)
{
    const auto slots=h->host_buffers<std::byte>(0,0);
    require_all(!slots[0]&&!slots[1],"fresh handle staging cache was not cold");
}
std::array<std::byte*,2> reserve(DdlaHandle_t h,const std::array<std::size_t,2>& sizes)
{
    std::array<std::byte*,2> slots{};
    detail::agree_local(h,"test explicit staging reservation",[&]{slots=h->host_buffers<std::byte>(sizes[0],sizes[1]);});
    return slots;
}
void check_trace(DdlaHandle_t h,const Trace& trace,const std::array<std::size_t,2>& sizes,const char* operation)
{
    StagingPreparationTrace seen{};trace.end(&seen);
    if(seen.host_allocations)trace.allocation_dump(h->myid,operation);
    require_all(!seen.host_allocations&&seen.slot_bytes[0]==sizes[0]&&seen.slot_bytes[1]==sizes[1]
        &&(!(sizes[0]||sizes[1])||seen.messages),"reserved raw run allocated or staging bound differs from observed native payload high water");
    std::cout<<"STAGING_COLD_RESERVED operation="<<operation<<" rank="<<h->myid
        <<" requested="<<sizes[0]<<','<<sizes[1]<<" observed="<<seen.slot_bytes[0]<<','<<seen.slot_bytes[1]
        <<" host_allocations="<<seen.host_allocations<<" native_messages="<<seen.messages<<std::endl;
}
using Z=std::complex<double>;
Z entry(int i,int j){return {0.3+0.017*i-0.023*j,-0.2+0.011*i+0.019*j};}
void descriptor(DdlaHandle_t h,int* d,int m,int n,int mb,int nb)
{
    DDLA_CHECK(ddlaDescInit(d,h,m,n,mb,nb,0,0,std::max(1,num_loc(m,mb,h->myprow_,0,h->nprows_))+3));
}

template<DdlaBackend Backend>
void pgeadd_raw(int pr,int pc,const Trace& trace,char op,bool aligned,bool alias,bool two=false)
{
    auto h=create(Backend,pr,pc);
    {
        int da[DDLA_DLEN_],dc[DDLA_DLEN_];
        descriptor(h,da,29,31,aligned?3:2,aligned?3:3);
        descriptor(h,dc,29,31,aligned?3:4,aligned?3:5);
        if(alias)std::copy(dc,dc+DDLA_DLEN_,da);
        const int m=13,n=11,ia=aligned?1:2,ja=aligned?1:3,ic=aligned?1:3,jc=aligned?1:2;
        const auto input=make_local<Z>(h,da,[](int i,int j){return entry(i,j);});
        const auto initial=make_local<Z>(h,dc,[&](int i,int j){return alias?entry(i,j):Z(-7);});
        DeviceBuffer<Z> a(h,std::max(std::size_t(1),input.size())),c(h,std::max(std::size_t(1),initial.size()));
        upload(h,a.ptr,input);upload(h,c.ptr,initial);
        const Z* ap=alias?c.ptr:a.ptr;
        const char ob=two?(aligned?op:op=='N'?'C':'N'):'N';
        const Z alpha=two?Z(2):Z(1),beta=two?Z(-.5):Z(0);
        std::size_t bb=0,hb=0;pgeadd_impl::pgeadd_workspace_size<Z>(h,op,ob,true,two,m,n,dc,&bb,&hb,da,two?da:dc,true);
        ddla::vector<char> work(h,bb);std::vector<char> host(hb);
        const auto sizes=pgeadd_impl::pgeadd_staging_bytes(h,op,ob,true,two,m,n,da,ia,ja,two?da:dc,two?ia:1,two?ja:1,dc,ic,jc,sizeof(Z));
        if(h->nprocs==1)require_all(!sizes[0]&&!sizes[1],"one-rank PGEMM staging must be zero for every domain");
        if constexpr(Backend==DdlaBackend::GPU) {
            cold(h);
            if(h->communication()!=DdlaCommunication::HostStagedMPI)require_all(!sizes[0]&&!sizes[1],"nonstaged query must be zero");
        }else require_all(!sizes[0]&&!sizes[1],"CPU staging query must be zero");
        const auto slots=reserve(h,sizes);DDLA_CHECK(ddlaSynchronize(h));
        if constexpr(Backend==DdlaBackend::GPU)trace.begin(slots[0],slots[1]);
        pgeadd_impl::pgeadd_run<Backend,Z>(h,op,ob,m,n,alpha,ap,ia,ja,da,beta,two?ap:nullptr,two?ia:1,two?ja:1,two?da:dc,
            c.ptr,ic,jc,dc,work.data(),bb,host.data(),hb);
        if constexpr(Backend==DdlaBackend::GPU)check_trace(h,trace,sizes,"pgeadd");
        const auto result=download(h,c.ptr,initial.size());double error=0;
        for(int j=0;j<ddla_test::n_loc(h,dc);++j)for(int i=0;i<dc[DDLA_LLD_];++i) {
            const auto at=i+std::size_t(j)*dc[DDLA_LLD_];Z expected=initial[at];
            if(i<ddla_test::m_loc(h,dc)) {
                const int gi=indx_l2g_r(dc,h,i),gj=indx_l2g_c(dc,h,j);
                if(gi>=ic-1&&gi<ic-1+m&&gj>=jc-1&&gj<jc-1+n) {
                    const int r=gi-ic+1,col=gj-jc+1;
                    expected=op=='N'?entry(ia-1+r,ja-1+col):entry(ia-1+col,ja-1+r);
                    if(op=='C')expected=std::conj(expected);
                    if(two) {
                        auto other=ob=='N'?entry(ia-1+r,ja-1+col):entry(ia-1+col,ja-1+r);
                        if(ob=='C')other=std::conj(other);
                        expected=alpha*expected+beta*other;
                    }
                }
            }
            error=std::max(error,double(std::abs(result[at]-expected)));
        }
        require_close(h,"cold reserved PGEADD geometry/alias",error,1e-12);
    }
    DDLA_CHECK(ddlaDestroy(h));
}

template<DdlaBackend Backend,class TA,class TB>
void pgemm_raw(int pr,int pc,const Trace& trace,char ta,char tb,pgemm_impl::Algorithm requested)
{
    auto h=create(Backend,pr,pc);
    {
        constexpr int m=37,n=41,k=53,nb=7;
        Matrix<TA>a(h,ta=='N'?m:k,ta=='N'?k:m,nb,nb,0,0,1,false);
        Matrix<TB>b(h,tb=='N'?k:n,tb=='N'?n:k,nb,nb,0,0,2,false);
        Matrix<Z>c(h,m,n,nb,nb,0,0,3,false);
        const Z alpha(.7,-.2),beta(-.3,.15);int panel=21;
        auto algorithm=pgemm_impl::pgemm_check<Backend>(h,ta,tb,m,n,k,false,a.d,b.d,c.d,requested,panel);
        std::size_t bb=0,hb=0;pgemm_impl::pgemm_workspace_size<Z,TA,TB>(h,algorithm,panel,false,ta,tb,m,n,k,true,a.d,b.d,c.d,&bb,&hb);
        ddla::vector<char> work(h,bb);std::vector<char> host(hb);
        const auto sizes=pgemm_impl::pgemm_staging_bytes(h,algorithm,true,ta,tb,m,n,k,true,a.d,b.d,c.d,sizeof(TA),sizeof(TB),sizeof(Z));
        if constexpr(Backend==DdlaBackend::GPU) {
            cold(h);
            if(h->communication()!=DdlaCommunication::HostStagedMPI)require_all(!sizes[0]&&!sizes[1],"nonstaged PGEMM query must be zero");
        }else require_all(!sizes[0]&&!sizes[1],"CPU PGEMM staging query must be zero");
        const auto slots=reserve(h,sizes);DDLA_CHECK(ddlaSynchronize(h));
        if constexpr(Backend==DdlaBackend::GPU)trace.begin(slots[0],slots[1]);
        pgemm_impl::pgemm_run<Backend,Z,TA,TB>(h,algorithm,panel,ta,tb,m,n,k,alpha,a.data,a.d,b.data,b.d,beta,c.data,c.d,
            work.data(),bb,host.data(),hb);
        if constexpr(Backend==DdlaBackend::GPU)check_trace(h,trace,sizes,"pgemm");
        const auto result=c.download();double error=0;
        for(std::size_t p=0;p<result.size();++p) {
            const int i=p%c.d[DDLA_LLD_],j=p/c.d[DDLA_LLD_];Z expected=c.host[p];
            if(i<c.nr&&j<c.nc) {
                const int gi=indxl2g(i,nb,c.r,0,pr),gj=indxl2g(j,nb,c.c,0,pc);
                if(gi<m&&gj<n) {
                    std::complex<long double> sum=0;
                    for(int l=0;l<k;++l) {
                        std::complex<long double> av=wide(::value<TA>(ta=='N'?gi:l,ta=='N'?l:gi,1));
                        std::complex<long double> bv=wide(::value<TB>(tb=='N'?l:gj,tb=='N'?gj:l,2));
                        if(ta=='C')av=std::conj(av);if(tb=='C')bv=std::conj(bv);sum+=av*bv;
                    }
                    expected=alpha*Z(sum.real(),sum.imag())+beta*::value<Z>(gi,gj,3);
                }
            }
            error=std::max(error,double(std::abs(result[p]-expected)));
        }
        require_close(h,"reserved PGEMM AB/AC/BC",error,1e-11);
    }
    DDLA_CHECK(ddlaDestroy(h));
}

template<class T> T factor_entry(char uplo,int i,int j,int n)
{
    if(i==j)return T(generalized_test::diagonal(i,n));
    if(uplo=='L' && i==j+1)return generalized_test::subdiagonal<T>(i);
    if(uplo=='U' && j==i+1)return generalized_test::conjugate(generalized_test::subdiagonal<T>(j));
    return T(0);
}
template<class T> T poisoned_diagonal(T x)
{
    if constexpr(std::is_arithmetic_v<T>)return x;
    else return T(x.real(),std::numeric_limits<typename real_type<T>::type>::quiet_NaN());
}
template<class T>
std::vector<T> congruence_input(DdlaHandle_t h,const int* d,char uplo,int n,bool factor)
{
    const T sentinel=T(8157),nan=T(std::numeric_limits<typename real_type<T>::type>::quiet_NaN());
    return make_local<T>(h,d,[&](int i,int j) {
        if(i>=n || j>=n)return sentinel;
        if(!(uplo=='L'?i>=j:i<=j))return nan;
        const T x=factor?factor_entry<T>(uplo,i,j,n):generalized_test::a<T>(i,j,n);
        return !factor && i==j?poisoned_diagonal(x):x;
    });
}
template<class T>
void verify_congruence(DdlaHandle_t h,int n,const int* d,const std::vector<T>& result,
                      const std::vector<T>& input,const std::vector<T>& factor,
                      const std::vector<T>& fixed)
{
    const double eps=std::numeric_limits<typename real_type<T>::type>::epsilon();
    bool guards=factor.size()==fixed.size() && !std::memcmp(factor.data(),fixed.data(),factor.size()*sizeof(T));
    double error=0,imaginary=0;
    for(std::size_t at=0;at<result.size();++at) {
        const int i=int(at%d[DDLA_LLD_]),j=int(at/d[DDLA_LLD_]);
        const int gi=indx_l2g_r(d,h,i),gj=indx_l2g_c(d,h,j);
        if(i>=ddla_test::m_loc(h,d) || j>=ddla_test::n_loc(h,d) || gi>=n || gj>=n) {
            guards=guards && !std::memcmp(&result[at],&input[at],sizeof(T));
            continue;
        }
        error=std::max(error,workspace_test::relative_error(result[at],generalized_test::standard<T>(gi,gj,n))/
            (eps*std::max(1,n)));
        if constexpr(!std::is_arithmetic_v<T>)
            if(gi==gj)imaginary=std::max(imaginary,workspace_test::finite_error(double(std::abs(result[at].imag()))));
    }
    require_close(h,"reserved PHEGST analytic complete output (scaled)",error,400);
    require_all(guards && imaginary==0,"reserved PHEGST factor/padding/real-diagonal guard failed");
}
template<DdlaBackend Backend,class T>
void phegst_raw(int pr,int pc,const Trace& trace,char uplo,int n,int block=3)
{
    auto h=create(Backend,pr,pc);
    {
        int da[DDLA_DLEN_],df[DDLA_DLEN_];
        descriptor(h,da,n+2,n+3,block,block);descriptor(h,df,n+3,n+2,block,block);
        auto input=congruence_input<T>(h,da,uplo,n,false),fixed=congruence_input<T>(h,df,uplo,n,true);
        DeviceBuffer<T>a(h,std::max(std::size_t(1),input.size())),f(h,std::max(std::size_t(1),fixed.size()));
        input.resize(a.count,T(8157));fixed.resize(f.count,T(8157));
        upload(h,a.ptr,input);upload(h,f.ptr,fixed);
        const auto p=hegst_impl::check<T>(h,Backend,uplo,n,a.ptr,da,f.ptr,df);
        const auto w=hegst_impl::workspace_size<T>(h,p);
        if(n && n<=block)require_all(!w.backend_bytes&&!w.host_bytes&&!w.staging_bytes[0]&&!w.staging_bytes[1],
            "single-block PHEGST needs zero arenas/cache on every rank");
        ddla::vector<char> work(h,w.backend_bytes);std::vector<char> host(w.host_bytes);
        cold(h);
        if(Backend==DdlaBackend::CPU || transport!=DdlaCommunication::HostStagedMPI)
            require_all(!w.staging_bytes[0]&&!w.staging_bytes[1],"nonstaged PHEGST query must be zero");
        const auto slots=reserve(h,w.staging_bytes);DDLA_CHECK(ddlaSynchronize(h));
        if constexpr(Backend==DdlaBackend::GPU){trace.refresh_slots(h);trace.begin(slots[0],slots[1]);}
        hegst_impl::run<Backend,T>(h,p,a.ptr,f.ptr,work.data(),w.backend_bytes,host.data(),w.host_bytes);
        if constexpr(Backend==DdlaBackend::GPU) {
            if(n && n<=block) {
                StagingPreparationTrace seen{};trace.end(&seen);
                if(seen.host_allocations)trace.allocation_dump(h->myid,"phegst-leaf");
                require_all(!seen.host_allocations&&!seen.slot_bytes[0]&&!seen.slot_bytes[1]&&!seen.messages&&!seen.native_calls,
                    "single-block PHEGST raw run allocated or communicated");
                std::cout<<"PHEGST_LEAF_COLD_RESERVED rank="<<h->myid<<" n="<<n<<" block="<<block
                    <<" uplo="<<uplo<<" type_bytes="<<sizeof(T)
                    <<" requested=0,0 observed=0,0 host_allocations=0 native_messages=0 native_calls=0"<<std::endl;
            } else check_trace(h,trace,w.staging_bytes,uplo=='L'?"phegst-L":"phegst-U");
        }
        verify_congruence(h,n,da,download(h,a.ptr,a.count),input,download(h,f.ptr,f.count),fixed);
        if(!h->myid)std::cout<<"PHEGST_STAGING_CASE_PASS scalar_bytes="<<sizeof(T)
            <<" complex="<<(!std::is_arithmetic_v<T>)<<" uplo="<<uplo<<" n="<<n<<" block="<<block<<'\n';
    }
    DDLA_CHECK(ddlaDestroy(h));
}

template<DdlaBackend Backend>
void ptrtrs_raw(int pr,int pc,const Trace& trace,char side,char uplo,char trans)
{
    auto h=create(Backend,pr,pc);
    {
        constexpr int order=17,block=3;
        const int m=side=='L'?order:11,n=side=='L'?11:order;
        int da[DDLA_DLEN_],db[DDLA_DLEN_];
        descriptor(h,da,order+2,order+3,block,block);descriptor(h,db,m+2,n+3,block,block);
        auto fixed=congruence_input<Z>(h,da,uplo,order,true);
        const auto op=[&](int i,int j) {
            Z value=trans=='N'?factor_entry<Z>(uplo,i,j,order):factor_entry<Z>(uplo,j,i,order);
            return trans=='C'?std::conj(value):value;
        };
        auto input=make_local<Z>(h,db,[&](int i,int j) {
            if(i>=m || j>=n)return Z(8157);
            Z value=0;
            for(int k=0;k<order;++k)value+=side=='L'?op(i,k)*entry(k,j):entry(i,k)*op(k,j);
            return value;
        });
        DeviceBuffer<Z>a(h,std::max(std::size_t(1),fixed.size())),b(h,std::max(std::size_t(1),input.size()));
        fixed.resize(a.count,Z(8157));input.resize(b.count,Z(8157));upload(h,a.ptr,fixed);upload(h,b.ptr,input);
        std::size_t bb=0,hb=0;ptrtrs_impl::ptrtrs_workspace_size<Z>(h,side,trans,m,n,da,db,&bb,&hb);
        ddla::vector<char> work(h,bb);std::vector<char> host(hb);
        const auto sizes=ptrtrs_impl::ptrtrs_staging_bytes(h,side,uplo,trans,m,n,da,db,sizeof(Z));
        cold(h);
        if(Backend==DdlaBackend::CPU || transport!=DdlaCommunication::HostStagedMPI)
            require_all(!sizes[0]&&!sizes[1],"nonstaged PTRTRS query must be zero");
        const auto slots=reserve(h,sizes);DDLA_CHECK(ddlaSynchronize(h));
        if constexpr(Backend==DdlaBackend::GPU)trace.begin(slots[0],slots[1]);
        ptrtrs_impl::ptrtrs_run<Backend,Z>(h,side,uplo,trans,'N',m,n,a.ptr,da,b.ptr,db,work.data(),bb,host.data(),hb);
        if constexpr(Backend==DdlaBackend::GPU)check_trace(h,trace,sizes,"ptrtrs");
        const auto result=download(h,b.ptr,b.count),factor=download(h,a.ptr,a.count);
        bool guards=!std::memcmp(factor.data(),fixed.data(),fixed.size()*sizeof(Z));double error=0;
        for(std::size_t at=0;at<result.size();++at) {
            const int i=int(at%db[DDLA_LLD_]),j=int(at/db[DDLA_LLD_]);
            const int gi=indx_l2g_r(db,h,i),gj=indx_l2g_c(db,h,j);
            if(i>=ddla_test::m_loc(h,db) || j>=ddla_test::n_loc(h,db) || gi>=m || gj>=n) {
                guards=guards && !std::memcmp(&result[at],&input[at],sizeof(Z));continue;
            }
            error=std::max(error,workspace_test::relative_error(result[at],entry(gi,gj)));
        }
        require_close(h,"cold reserved PTRTRS side/uplo/trans",error,1e-12);
        require_all(guards,"reserved PTRTRS factor/padding guard failed");
        if(!h->myid)std::cout<<"PTRTRS_STAGING_CASE_PASS side="<<side<<" uplo="<<uplo<<" trans="<<trans<<'\n';
    }
    DDLA_CHECK(ddlaDestroy(h));
}
#if DDLA_HAS_GPU
void phegst_public_fault(int pr,int pc,const Trace& trace,char uplo,int slot)
{
    auto h=create(DdlaBackend::GPU,pr,pc);
    {
        constexpr int block=3,parent=43;
        int da[DDLA_DLEN_],df[DDLA_DLEN_];
        descriptor(h,da,parent,parent+1,block,block);descriptor(h,df,parent+1,parent,block,block);
        DeviceBuffer<Z>a(h,std::max(std::size_t(1),local_size(h,da))),f(h,std::max(std::size_t(1),local_size(h,df)));
        int n=0,target_rank=h->nprocs;
        hegst_impl::Workspace w;
        for(int candidate=13;candidate<=35;++candidate) {
            const auto p=hegst_impl::check<Z>(h,DdlaBackend::GPU,uplo,candidate,a.ptr,da,f.ptr,df);
            w=hegst_impl::workspace_size<Z>(h,p);
            const auto bytes=w.staging_bytes[slot];
            int owner=bytes && bytes%256?h->myid:h->nprocs;
            MPI_Allreduce(&owner,&target_rank,1,MPI_INT,MPI_MIN,h->comm);
            if(target_rank<h->nprocs){n=candidate;break;}
        }
        require_all(n>0,"no unambiguous PHEGST staged-slot fault geometry found");
        auto input=congruence_input<Z>(h,da,uplo,n,false),fixed=congruence_input<Z>(h,df,uplo,n,true);
        input.resize(a.count,Z(8157));fixed.resize(f.count,Z(8157));upload(h,a.ptr,input);upload(h,f.ptr,fixed);
        cold(h);
        const auto target=w.staging_bytes[slot];
        unsigned long long fault_bytes=h->myid==target_rank?target:0;
        MPI_Bcast(&fault_bytes,1,MPI_UNSIGNED_LONG_LONG,target_rank,h->comm);
        const unsigned ordinal=slot==1 && w.staging_bytes[0]==target?2:1;
        trace.arm(h->myid==target_rank?target:0,ordinal);
        bool threw=false;
        try {phegst<Z>(h,uplo,n,a.ptr,da,f.ptr,df);}catch(const std::exception&){threw=true;}
        trace.disarm();
        require_all(threw,"PHEGST staging allocation failure was not agreed before math on every rank");
        require_all(trace.fired()==unsigned(h->myid==target_rank) &&
            (!trace.fired() || trace.failed_bytes()==target),"native PHEGST exact-size staging fault did not fire once on target rank");
        const auto after=download(h,a.ptr,a.count),factor=download(h,f.ptr,f.count);
        require_all(!std::memcmp(after.data(),input.data(),a.count*sizeof(Z)) &&
            !std::memcmp(factor.data(),fixed.data(),f.count*sizeof(Z)),"PHEGST staging failure changed A or fixed factor");
        phegst<Z>(h,uplo,n,a.ptr,da,f.ptr,df);
        verify_congruence(h,n,da,download(h,a.ptr,a.count),input,download(h,f.ptr,f.count),fixed);
        if(!h->myid)std::cout<<"STAGING_PUBLIC_FAULT_PASS operation=phegst uplo="<<uplo<<" slot="<<slot
            <<" target_rank="<<target_rank<<" observed_bytes="<<fault_bytes
            <<" all_rank_throw=1 output_unchanged=1 same_handle_recovery=1\n";
    }
    DDLA_CHECK(ddlaDestroy(h));
}
void public_fault(int pr,int pc,const Trace& trace,bool gemm,int slot)
{
    auto h=create(DdlaBackend::GPU,pr,pc);
    {
        int da[DDLA_DLEN_],dc[DDLA_DLEN_];
        descriptor(h,da,29,31,2,3);descriptor(h,dc,29,31,4,5);
        const auto input=make_local<Z>(h,da,[](int i,int j){return entry(i,j);});
        const auto initial=make_local<Z>(h,dc,[](int,int){return Z(-7);});
        DeviceBuffer<Z>a(h,std::max(std::size_t(1),input.size())),c(h,std::max(std::size_t(1),initial.size()));
        upload(h,a.ptr,input);upload(h,c.ptr,initial);
        Matrix<Z>ga(h,37,53,7,7,0,0,1,false),gb(h,53,41,7,7,0,0,2,false),gc(h,37,41,7,7,0,0,3,false);
        int region[2]={13,11};
        if(!gemm) {
            if(h->myid==0) {
                bool found=false;
                for(int rows=13;rows<=21&&!found;++rows)for(int cols=11;cols<=21&&!found;++cols) {
                    const auto candidate=pgeadd_impl::pgeadd_staging_bytes(h,'C','N',true,false,rows,cols,
                        da,2,3,dc,1,1,dc,3,2,sizeof(Z));
                    if(candidate[slot] && candidate[slot]%256) {region[0]=rows;region[1]=cols;found=true;}
                }
                if(!found)region[0]=0;
            }
            MPI_Bcast(region,2,MPI_INT,0,MPI_COMM_WORLD);
            require_all(region[0]>0,"no unambiguous staged-slot allocation fault geometry found");
        }
        std::array<std::size_t,2> sizes;
        if(gemm) {
            int panel=21;
            const auto alg=pgemm_impl::pgemm_check<DdlaBackend::GPU>(h,'N','N',37,41,53,false,ga.d,gb.d,gc.d,pgemm_impl::Algorithm::Auto,panel);
            sizes=pgemm_impl::pgemm_staging_bytes(h,alg,true,'N','N',37,41,53,false,ga.d,gb.d,gc.d,sizeof(Z),sizeof(Z),sizeof(Z));
        }else sizes=pgeadd_impl::pgeadd_staging_bytes(h,'C','N',true,false,region[0],region[1],da,2,3,dc,1,1,dc,3,2,sizeof(Z));
        cold(h);
        // Metadata arenas are256-byte multiples. A nonmultiple exact size
        // therefore identifies this cold std::byte staging slot, not metadata.
        const auto target=sizes[slot];
        unsigned ordinal=slot==1&&sizes[0]==sizes[1]?2:1;
        require_all(h->myid!=0||(target>0&&target%256!=0),"fault shape must identify a unique nonmetadata staged-slot allocation size");
        trace.arm(h->myid==0?target:0,ordinal);
        bool threw=false;
        try {
            if(gemm)pgemm<Z>(h,'N','N',37,41,53,Z(1),ga.data,ga.d,gb.data,gb.d,Z(0),gc.data,gc.d);
            else pgeadd<Z>(h,'C','N',region[0],region[1],Z(1),a.ptr,2,3,da,Z(0),nullptr,1,1,dc,c.ptr,3,2,dc);
        }catch(const std::exception&){threw=true;}
        trace.disarm();
        require_all(threw,"staged-slot allocation failure was not agreed before math on every rank");
        require_all(trace.fired()==unsigned(h->myid==0)&&(!trace.fired()||trace.failed_bytes()==target),"exact staged-slot native operator-new fault did not fire once on target rank");
        const auto unchanged=gemm?gc.download()==gc.host:download(h,c.ptr,initial.size())==initial;
        require_all(unchanged,"staged allocation preparation failure modified output");
        // Retry on the same handle, including a partially grown two-slot cache.
        if(gemm)pgemm<Z>(h,'N','N',37,41,53,Z(1),ga.data,ga.d,gb.data,gb.d,Z(0),gc.data,gc.d);
        else pgeadd<Z>(h,'C','N',region[0],region[1],Z(1),a.ptr,2,3,da,Z(0),nullptr,1,1,dc,c.ptr,3,2,dc);
        DDLA_CHECK(ddlaSynchronize(h));
        double recovery_error=0;
        if(gemm) {
            const auto result=gc.download();
            for(std::size_t p=0;p<result.size();++p) {
                const int i=p%gc.d[DDLA_LLD_],j=p/gc.d[DDLA_LLD_];Z expected=gc.host[p];
                if(i<gc.nr&&j<gc.nc) {
                    const int gi=indxl2g(i,7,gc.r,0,pr),gj=indxl2g(j,7,gc.c,0,pc);
                    if(gi<37&&gj<41) {
                        expected=Z(0);
                        for(int l=0;l<53;++l)expected+=::value<Z>(gi,l,1)*::value<Z>(l,gj,2);
                    }
                }
                recovery_error=std::max(recovery_error,double(std::abs(result[p]-expected)));
            }
        }else {
            const auto result=download(h,c.ptr,initial.size());
            for(int j=0;j<ddla_test::n_loc(h,dc);++j)for(int i=0;i<dc[DDLA_LLD_];++i) {
                const auto p=i+std::size_t(j)*dc[DDLA_LLD_];Z expected=initial[p];
                if(i<ddla_test::m_loc(h,dc)) {
                    const int gi=indx_l2g_r(dc,h,i),gj=indx_l2g_c(dc,h,j);
                    if(gi>=2&&gi<2+region[0]&&gj>=1&&gj<1+region[1])expected=std::conj(entry(1+gj-1,2+gi-2));
                }
                recovery_error=std::max(recovery_error,double(std::abs(result[p]-expected)));
            }
        }
        require_close(h,"same-handle staged allocation recovery",recovery_error,1e-11);
        if(!h->myid)std::cout<<"STAGING_PUBLIC_FAULT_PASS operation="<<(gemm?"pgemm":"pgeadd")<<" slot="<<slot
            <<" observed_bytes="<<target<<" all_rank_throw=1 output_unchanged=1 same_handle_recovery=1\n";
    }
    DDLA_CHECK(ddlaDestroy(h));
}
#endif
#include "spectral_staging_cases.h"
#ifndef DDLA_STAGING_ONE_STAGE_DIAGNOSTIC_ONLY
#include "composite_staging_cases.h"
#endif
}
int main(int argc,char** argv)
{
    ddla_test_mpi_init(&argc,&argv);
    try {
        int size=1;MPI_Comm_size(MPI_COMM_WORLD,&size);int pr=1,pc=size;bool gpu=false,phegst_only=false,spectral_only=false,composite_only=false,one_stage_only=false;
        for(int i=1;i<argc;++i) {
            std::string arg=argv[i];if(arg=="--gpu")gpu=true;
            else if(arg=="--phegst-only")phegst_only=true;
            else if(arg=="--spectral-only")spectral_only=true;
            else if(arg=="--composite-only")composite_only=true;
            else if(arg=="--one-stage-only")one_stage_only=true;
            else if(arg=="--major"&&i+1<argc) {
                const std::string major=argv[++i];
                if(major!="R"&&major!="C")throw std::invalid_argument("--major requires R or C");
                requested_major=major[0];
            }
            else if(arg=="--transport"&&i+1<argc) {
                const std::string mode=argv[++i];
                if(mode=="staged")transport=DdlaCommunication::HostStagedMPI;
                else if(mode=="ccl")transport=DdlaCommunication::CCL;
                else if(mode=="mpi")transport=DdlaCommunication::MPI;
                else throw std::invalid_argument("invalid transport");
            }
            else if(arg=="--grid"&&i+1<argc){std::string grid=argv[++i];auto x=grid.find('x');pr=std::stoi(grid.substr(0,x));pc=std::stoi(grid.substr(x+1));}
            else throw std::invalid_argument("test_staging_preparation: invalid argument");
        }
        require_all(pr*pc==size,"grid must match launched ranks");
        require_all(int(phegst_only)+int(spectral_only)+int(composite_only)+int(one_stage_only)<=1,"focused family flags are mutually exclusive");
        setenv("DDLA_PGEMM_KPANEL","21",1);
        Trace trace(gpu);
        auto run=[&](auto backend) {
            constexpr auto B=decltype(backend)::value;
            if(!phegst_only&&!spectral_only&&!composite_only&&!one_stage_only) {
                for(char op:{'N','T','C'}) {
                    pgeadd_raw<B>(pr,pc,trace,op,false,false);
                    pgeadd_raw<B>(pr,pc,trace,op,true,false);
                    pgeadd_raw<B>(pr,pc,trace,op,false,false,true);
                    pgeadd_raw<B>(pr,pc,trace,op,true,false,true);
                }
                pgeadd_raw<B>(pr,pc,trace,'C',false,true);
                for(char ta:{'N','T','C'})for(char tb:{'N','T','C'})for(auto alg:{pgemm_impl::Algorithm::AB,pgemm_impl::Algorithm::AC,pgemm_impl::Algorithm::BC})
                    pgemm_raw<B,Z,Z>(pr,pc,trace,ta,tb,alg);
                pgemm_raw<B,double,Z>(pr,pc,trace,'N','C',pgemm_impl::Algorithm::AB);
                pgemm_raw<B,Z,double>(pr,pc,trace,'C','N',pgemm_impl::Algorithm::BC);
            }
            if(!spectral_only&&!composite_only&&!one_stage_only) {
            for(char side:{'L','R'})for(char uplo:{'L','U'})for(char trans:{'N','T','C'})
                ptrtrs_raw<B>(pr,pc,trace,side,uplo,trans);
            for(char uplo:{'L','U'})for(int n:{0,1,17}) {
                phegst_raw<B,float>(pr,pc,trace,uplo,n);
                phegst_raw<B,double>(pr,pc,trace,uplo,n);
                phegst_raw<B,std::complex<float>>(pr,pc,trace,uplo,n);
                phegst_raw<B,Z>(pr,pc,trace,uplo,n);
            }
            for(char uplo:{'L','U'})phegst_raw<B,Z>(pr,pc,trace,uplo,65);
            for(char uplo:{'L','U'})for(int block:{17,64}) {
                phegst_raw<B,float>(pr,pc,trace,uplo,17,block);
                phegst_raw<B,double>(pr,pc,trace,uplo,17,block);
                phegst_raw<B,std::complex<float>>(pr,pc,trace,uplo,17,block);
                phegst_raw<B,Z>(pr,pc,trace,uplo,17,block);
            }
            }
        };
        if(!gpu&&(spectral_only||composite_only||one_stage_only))throw std::invalid_argument("spectral/composite diagnostics require --gpu");
        if(!gpu)run(std::integral_constant<DdlaBackend,DdlaBackend::CPU>{});
#if DDLA_HAS_GPU
        else {
            run(std::integral_constant<DdlaBackend,DdlaBackend::GPU>{});
            if(one_stage_only){
#define ONE_STAGE_TYPE(T) for(int n:{7,1})for(char uplo:{'U','L'})for(char job:{'N','V'})spectral_one_stage<T>(pr,pc,trace,n,job,uplo,true)
                ONE_STAGE_TYPE(float);ONE_STAGE_TYPE(double);ONE_STAGE_TYPE(std::complex<float>);ONE_STAGE_TYPE(Z);
#undef ONE_STAGE_TYPE
            }
            if(!phegst_only&&!composite_only&&!one_stage_only)run_spectral_staging(pr,pc,trace);
            #ifndef DDLA_STAGING_ONE_STAGE_DIAGNOSTIC_ONLY
            if(!phegst_only&&!spectral_only&&!one_stage_only)composite_staging::run(pr,pc,trace);
#endif
            if(size>1&&transport==DdlaCommunication::HostStagedMPI) {
                if(!phegst_only&&!composite_only&&!one_stage_only) {
                    for(int slot:{0,1})chebyshev_public_staging_fault(pr,pc,trace,slot);
                    for(bool wide:{false,true})for(int slot:{0,1})svd_public_staging_fault(pr,pc,trace,wide,slot);
                }
                if(!spectral_only&&!phegst_only&&!composite_only&&!one_stage_only) {
                    public_fault(pr,pc,trace,false,0);public_fault(pr,pc,trace,false,1);public_fault(pr,pc,trace,true,0);
                }
                if(!spectral_only&&!composite_only&&!one_stage_only)for(char uplo:{'L','U'})for(int slot:{0,1})phegst_public_fault(pr,pc,trace,uplo,slot);
            }
        }
#else
        else throw std::runtime_error("GPU build required");
#endif
        int rank=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);if(!rank)std::cout<<"STAGING_PREPARATION_PASS major="<<requested_major<<" spectral_only="<<spectral_only<<" composite_only="<<composite_only<<" one_stage_only="<<one_stage_only<<" no_global_cold_allocation_claim=1\n";
        MPI_Finalize();return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);}
    return 1;
}
