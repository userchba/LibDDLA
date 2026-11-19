#include "generalized_eigen_test_common.h"
#include "workspace_test_common.h"
#include "Eigensolver/phegvd_workspace.h"
#include "ddla_stream.h"
#include "mpi_datatype.h"
#include "staging_preparation_trace_api.h"
#include <dlfcn.h>
#include <array>
#include <cstring>
#include <memory>
#include <optional>
using namespace api_grid_test;
namespace {
DdlaCommunication requested_transport=DdlaCommunication::HostStagedMPI;
char requested_major='R';
struct PayloadObserver {
    void(*begin)(void*,void*)=nullptr;
    void(*end)(StagingPreparationTrace*)=nullptr;
    PayloadObserver(){
        auto a=dlsym(RTLD_DEFAULT,"staging_preparation_trace_begin");
        auto b=dlsym(RTLD_DEFAULT,"staging_preparation_trace_end");
        std::memcpy(&begin,&a,sizeof(begin));std::memcpy(&end,&b,sizeof(end));
        if(!begin||!end){begin=nullptr;end=nullptr;}
        if(begin){
            auto a=dlsym(RTLD_DEFAULT,"pblas_alloc_failure_prewarm");int(*prewarm)()=nullptr;
            std::memcpy(&prewarm,&a,sizeof(prewarm));
            if(!prewarm||prewarm()!=1)throw std::runtime_error("PHEGVD payload observer prewarm failed");
        }
    }
};
struct Method {
    bool present=std::getenv("DDLA_EIGEN_METHOD")!=nullptr;
    std::string saved=present?std::getenv("DDLA_EIGEN_METHOD"):"";
    ~Method() {
        if (present) setenv("DDLA_EIGEN_METHOD",saved.c_str(),1);
        else unsetenv("DDLA_EIGEN_METHOD");
    }
};
bool no_alloc(const workspace_test::Snapshot& s) {
    return !s.ddla_alloc && !s.ddla_free && !s.pinned_alloc && !s.raw_gpu_alloc &&
        !s.host_new && !s.vendor_query && !s.resource_setup;
}
bool pure(const workspace_test::Snapshot& s) {
    // Vendor queries are explicitly allowed in sizing, never in execution.
    return !s.ddla_alloc && !s.ddla_free && !s.pinned_alloc && !s.raw_gpu_alloc &&
        !s.host_new && !s.resource_setup && !s.copies && !s.launches && !s.syncs && !s.collectives;
}
class Arena {
    static constexpr std::size_t guard=256;
    DdlaHandle_t h_;
    std::size_t bb_,hb_;
    ddla::vector<char> backend_;
    std::vector<char> host_;
public:
    Arena(const DdlaHandle_t& h,std::size_t b,std::size_t s)
        :h_(h),bb_(b),hb_(s),backend_(h,b+2*guard),host_(s+2*guard,char(0xa7)) {
        DDLA_CHECK(ddlaMemset(h,backend_.data(),0xa7,backend_.size()));
    }
    char* backend() { return backend_.data()+guard; }
    char* host() { return host_.data()+guard; }
    void verify() {
        std::array<unsigned char,2*guard> bytes{};
        DDLA_CHECK(ddlaMemcpy(h_,bytes.data(),backend_.data(),guard));
        DDLA_CHECK(ddlaMemcpy(h_,bytes.data()+guard,backend()+bb_,guard));
        DDLA_CHECK(ddlaSynchronize(h_));
        bool okay=true;
        for (std::size_t i=0;i<guard;++i)
            okay=okay && bytes[i]==0xa7 && bytes[guard+i]==0xa7 &&
                static_cast<unsigned char>(host_[i])==0xa7 &&
                static_cast<unsigned char>(host()[hb_+i])==0xa7;
        require_close(h_,"phegvd exact maximum workspace guards",okay?0:1,0);
    }
};
template<class T>
bool input_padding(const DdlaHandle_t& h,const int* d,int n,const std::vector<T>& actual,
    const std::vector<T>& before,const int* dz=nullptr,std::size_t output_offset=0,int nev=0,int output_rows=-1) {
    if (actual.size()!=before.size()) return false;
    const int zr=dz?num_loc(output_rows<0?n:output_rows,dz[DDLA_MB_],h->myprow_,dz[DDLA_RSRC_],h->nprows_):0;
    const int zc=dz?num_loc(nev,dz[DDLA_NB_],h->mypcol_,dz[DDLA_CSRC_],h->npcols_):0;
    for (std::size_t at=0;at<actual.size();++at) {
        const int i=int(at%d[DDLA_LLD_]),j=int(at/d[DDLA_LLD_]);
        const bool active=i<ddla_test::m_loc(h,d) && j<ddla_test::n_loc(h,d) &&
            indx_l2g_r(d,h,i)<n && indx_l2g_c(d,h,j)<n;
        bool output=false;
        if (dz && at>=output_offset) {
            const std::size_t relative=at-output_offset;
            output=relative/std::size_t(dz[DDLA_LLD_])<std::size_t(zc) &&
                relative%std::size_t(dz[DDLA_LLD_])<std::size_t(zr);
        }
        if (!active && !output && std::memcmp(&actual[at],&before[at],sizeof(T))) return false;
    }
    return true;
}
template<DdlaBackend B,class T>
void exercise_on(const DdlaHandle_t& h,int n,int nev,char uplo,int alias,const char* method,char job='V') {
    using namespace spectral_impl;
    using R=typename real_type<T>::type;
    Method restore;
    setenv("DDLA_EIGEN_METHOD",method,1);
    if (!h->myid)
        std::cout<<"PHEGVD_WORKSPACE_CASE n="<<n<<" nev="<<nev<<" job="<<job
            <<" uplo="<<uplo<<" alias="<<alias<<" method="<<method
            <<" scalar_bytes="<<sizeof(T)<<" backend="<<int(B)<<std::endl;
    constexpr int nb=4;
    int da[DDLA_DLEN_],db[DDLA_DLEN_],dz[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(da,h,n+2,n+3,nb,nb,0,0)); da[DDLA_LLD_]+=3;
    DDLA_CHECK(ddlaDescInit(db,h,n+3,n+2,nb,nb,0,0)); db[DDLA_LLD_]+=2;
    DDLA_CHECK(ddlaDescInit(dz,h,n+2,nev+3,nb,nb,0,h->npcols_-1)); dz[DDLA_LLD_]+=4;
    if (alias) std::copy_n(da,DDLA_DLEN_,dz);
    const bool vectors=job=='V' && nev>0;
    const T sentinel=T(9876);
    const auto initialA=make_local<T>(h,da,[&](int,int){return sentinel;});
    const auto initialB=make_local<T>(h,db,[&](int,int){return sentinel;});
    DeviceBuffer<T> A(h,std::max(std::size_t(1),initialA.size())+8);
    DeviceBuffer<T> C(h,std::max(std::size_t(1),initialB.size()));
    DeviceBuffer<T> Z(h,std::max(std::size_t(1),local_size(h,dz)));
    std::vector<R> w(n+4,R(-9876));
    // Initialize complete allocated spans, including extra alias guards and
    // descriptor padding, before any checker or native workspace query.
    upload(h,A.ptr,std::vector<T>(A.count,sentinel));
    upload(h,C.ptr,std::vector<T>(C.count,sentinel));
    upload(h,Z.ptr,std::vector<T>(Z.count,sentinel));
    T* output=alias?A.ptr+(alias==2?1:0):Z.ptr;
    auto p=phegvd_check<B,T>(h,job,uplo,n,nev,A.ptr,da,C.ptr,db,w.data()+2,
        vectors?output:nullptr,vectors?dz:nullptr);
    PheevdDriverPlan eig;
    std::unique_ptr<Phb2stTopology> topology;
    if (n) {
        p.complete=pherm_complete_check(h,B,uplo,n,A.ptr,da);
        const auto route=detail::pheevd_select<B,T>(h,vectors?'V':'N',n,nev,da,vectors?dz:nullptr);
        require_close(h,"composite preserves single-rank automatic local route",
            h->nprocs==1 && !std::strcmp(method,"auto") && route!=detail::PheevdRoute::Local?1:0,0);
        if (route==detail::PheevdRoute::TwoStage) topology=phb2st_topology_prepare(h);
        eig=pheevd_driver_check<B,T>(h,route,vectors?'V':'N',uplo,n,nev,A.ptr,da,w.data()+2,
            vectors?output:nullptr,vectors?dz:nullptr,topology.get());
    }
    PheevdDriverResources<T> resources(h,eig);
    if (resources.local_values.size())
        DDLA_CHECK(ddlaMemset(h,resources.local_values.data(),0xa7,
            resources.local_values.size()*sizeof(R)));
    if (resources.local_info.size())
        DDLA_CHECK(ddlaMemset(h,resources.local_info.data(),0xa7,
            resources.local_info.size()*sizeof(int)));
    DDLA_CHECK(ddlaSynchronize(h));
    struct QueryState {
        std::vector<T> a,b,z;
        std::vector<R> eigenvalues,device_values;
        std::vector<int> device_info;
    };
    const auto snapshot=[&] {
        QueryState state;
        state.a=download(h,A.ptr,A.count);
        state.b=download(h,C.ptr,C.count);
        state.z=download(h,Z.ptr,Z.count);
        state.eigenvalues=w;
        if (resources.local_values.size())
            state.device_values=download(h,resources.local_values.data(),resources.local_values.size());
        if (resources.local_info.size())
            state.device_info=download(h,resources.local_info.data(),resources.local_info.size());
        return state;
    };
    const auto same_bytes=[](const auto& before,const auto& after) {
        return before.size()==after.size() && (before.empty() ||
            !std::memcmp(before.data(),after.data(),before.size()*sizeof(before[0])));
    };
    const auto unchanged=[&](const QueryState& before,const QueryState& after) {
        return same_bytes(before.a,after.a) && same_bytes(before.b,after.b) &&
            same_bytes(before.z,after.z) && same_bytes(before.eigenvalues,after.eigenvalues) &&
            same_bytes(before.device_values,after.device_values) &&
            same_bytes(before.device_info,after.device_info);
    };
    const auto root_before=snapshot();
    MPI_Barrier(h->comm);
    bool query_pure=true;
    if (!h->myid) {
        workspace_test::Scope scope(true);
        (void)phegvd_workspace_size<T>(h,p,eig,resources,A.ptr,C.ptr,w.data()+2);
        query_pure=pure(scope.finish());
    }
    MPI_Barrier(h->comm);
    const auto root_after=snapshot();
    require_close(h,"composite root-only query has no allocation, communication or matrix work",query_pure?0:1,0);
    require_close(h,"root-only query preserves complete matrix/value/info bytes",
        unchanged(root_before,root_after)?0:1,0);
    const auto all_before=snapshot();
    workspace_test::Scope all_scope(true);
    const auto q=phegvd_workspace_size<T>(h,p,eig,resources,A.ptr,C.ptr,w.data()+2);
    const auto all_trace=all_scope.finish();
    const auto all_after=snapshot();
    require_close(h,"composite all-rank query has no allocation, communication or matrix work",pure(all_trace)?0:1,0);
    require_close(h,"all-rank query preserves complete matrix/value/info bytes",
        unchanged(all_before,all_after)?0:1,0);
    require_close(h,"composite bytes are maximum, not sum",
        q.backend_bytes==std::max({q.factor_solve_backend,q.congruence_backend,q.eigen.backend_bytes}) &&
        q.host_bytes==std::max({q.factor_solve_host,q.congruence_host,q.eigen.host_bytes})?0:1,0);
    // Staging is queried independently of vendor arena queries and on root
    // alone as well as every rank. No matrix, native setup or MPI is permitted.
    PayloadObserver payload;
    bool staging_pure=true;
    MPI_Barrier(h->comm);
    if(!h->myid){
        workspace_test::Scope scope(true);
        if(payload.begin)payload.begin(nullptr,nullptr);
        (void)phegvd_staging_bytes<T>(h,p,eig,resources);
        StagingPreparationTrace native{};if(payload.end)payload.end(&native);
        const auto seen=scope.finish();staging_pure=pure(seen)&&!seen.vendor_query&&!native.host_allocations&&!native.native_calls;
    }
    MPI_Barrier(h->comm);
    require_close(h,"PHEGVD root-only staging query is pure",staging_pure?0:1,0);
    workspace_test::Scope staging_scope(true);
    if(payload.begin)payload.begin(nullptr,nullptr);
    const auto staging=phegvd_staging_bytes<T>(h,p,eig,resources);
    StagingPreparationTrace staging_native{};if(payload.end)payload.end(&staging_native);
    const auto staging_seen=staging_scope.finish();
    require_close(h,"PHEGVD all-rank staging query is pure",pure(staging_seen)&&!staging_seen.vendor_query&&!staging_native.host_allocations&&!staging_native.native_calls?0:1,0);
    const auto cold=h->host_buffers<std::byte>(0,0);
    require_close(h,"PHEGVD first prepared run uses a cold handle",!cold[0]&&!cold[1]?0:1,0);
    phegvd_prepare_staging(h,staging);
    const auto slots=h->host_buffers<std::byte>(0,0);
    Arena arena(h,q.backend_bytes,q.host_bytes);
    for (int pass=0;pass<2;++pass) {
        const T poison=T(std::numeric_limits<R>::quiet_NaN());
        auto aa=make_local<T>(h,da,[&](int i,int j) {
            if (i>=n || j>=n) return sentinel;
            return (uplo=='U'?i<=j:i>=j)?generalized_test::a<T>(i,j,n,pass!=0):poison;
        });
        aa.resize(A.count,sentinel);
        auto bb=make_local<T>(h,db,[&](int i,int j) {
            if (i>=n || j>=n) return sentinel;
            return (uplo=='U'?i<=j:i>=j)?generalized_test::b<T>(i,j,n):poison;
        });
        // Empty column owners still have the one-element device placeholder.
        // Compare exactly the allocated span, as already done for A above.
        bb.resize(C.count,sentinel);
        upload(h,A.ptr,aa); upload(h,C.ptr,bb);
        upload(h,Z.ptr,std::vector<T>(Z.count,sentinel));
        std::fill(w.begin(),w.end(),R(-9876));
        int info=-1;
        DDLA_CHECK(ddlaSynchronize(h));
        workspace_test::Scope scope;
        if(payload.begin)payload.begin(slots[0],slots[1]);
        phegvd_run<B,T>(h,p,eig,resources,q,A.ptr,C.ptr,w.data()+2,vectors?output:nullptr,info,
            arena.backend(),q.backend_bytes,arena.host(),q.host_bytes);
        StagingPreparationTrace observed{};
        if(payload.end)payload.end(&observed);
        const auto trace=scope.finish();
        if(payload.end && (observed.host_allocations || observed.slot_bytes[0]!=staging[0] || observed.slot_bytes[1]!=staging[1]))
            std::cerr<<"PHEGVD_STAGING_MISMATCH rank="<<h->myid<<" method="<<method<<" pass="<<pass
                <<" requested="<<staging[0]<<','<<staging[1]<<" observed="<<observed.slot_bytes[0]<<','<<observed.slot_bytes[1]
                <<" host_new="<<observed.host_allocations<<" native_calls="<<observed.native_calls<<std::endl;
        require_close(h,"composite cold/repeat run has no algorithm allocation/query/setup",no_alloc(trace)&&!observed.host_allocations?0:1,0);
        if(payload.end)require_close(h,"PHEGVD complete staging bound attained on successful input",
            observed.slot_bytes[0]==staging[0]&&observed.slot_bytes[1]==staging[1]?0:1,0);
        if(!h->myid)std::cout<<"PHEGVD_COLD_RESERVED method="<<method<<" major="<<h->major<<" pass="<<pass
            <<" requested="<<staging[0]<<','<<staging[1]<<" payload_observer="<<bool(payload.end)<<" host_new=0"<<std::endl;
        require_close(h,"composite info",info,0);
        require_close(h,"composite eigenvalue guards",w[0]==R(-9876) && w[1]==R(-9876) &&
            w[n+2]==R(-9876) && w[n+3]==R(-9876)?0:1,0);
        const auto afterA=download(h,A.ptr,A.count),afterB=download(h,C.ptr,C.count);
        require_close(h,"composite parent/input padding",
            input_padding(h,da,n,afterA,aa,alias&&vectors?dz:nullptr,alias==2?1:0,vectors?nev:0) &&
            input_padding(h,db,n,afterB,bb)?0:1,0);
        std::vector<T> global(std::size_t(n)*(vectors?nev:0),T(0));
        if (vectors) {
            const std::size_t count=alias?A.count-std::size_t(alias==2?1:0):Z.count;
            const auto out=download(h,output,count);
            const int nr=num_loc(n,nb,h->myprow_,0,h->nprows_);
            const int nc=num_loc(nev,nb,h->mypcol_,dz[DDLA_CSRC_],h->npcols_);
            for (int j=0;j<nc;++j) for (int i=0;i<nr;++i) {
                const int gi=indx_l2g_r(dz,h,i),gj=indx_l2g_c(dz,h,j);
                global[gi+std::size_t(gj)*n]=out[i+std::size_t(j)*dz[DDLA_LLD_]];
            }
            if (n && nev)
                MPI_Allreduce(MPI_IN_PLACE,global.data(),int(global.size()),detail::mpi_datatype<T>(),MPI_SUM,h->comm);
            if (!alias) {
                const std::vector<T> before(Z.count,sentinel);
                require_close(h,"composite Z padding",input_padding(h,dz,0,out,before,dz,0,nev,n)?0:1,0);
            }
        }
        generalized_test::verify<T>(h,n,vectors?nev:0,w.data()+2,global,pass!=0);
        arena.verify();
    }
    resources.release();
    if (topology) topology->release();
    if (!h->myid)
        std::cout<<"PHEGVD_WORKSPACE n="<<n<<" nev="<<nev<<" scalar_bytes="<<sizeof(T)
            <<" backend="<<int(B)<<" method="<<method<<" route="<<int(eig.route)
            <<" factor_backend="<<q.factor_solve_backend<<" congruence_backend="<<q.congruence_backend
            <<" eigen_backend="<<q.eigen.backend_bytes<<" maximum_backend="<<q.backend_bytes
            <<" factor_host="<<q.factor_solve_host<<" congruence_host="<<q.congruence_host
            <<" eigen_host="<<q.eigen.host_bytes<<" maximum_host="<<q.host_bytes
            <<" staging_required="<<staging[0]<<','<<staging[1]
            <<" cache_capacity="<<h->host_buffer_capacities()[0]<<','<<h->host_buffer_capacities()[1]
            <<" local_query_resource_backend="<<resources.retained_backend_bytes()<<std::endl;
}
template<DdlaBackend B,class T>
void exercise(const DdlaHandle_t& parent,int n,int nev,char uplo,int alias,const char* method,char job='V')
{
    DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,B));
    if constexpr(B==DdlaBackend::GPU)DDLA_CHECK(ddlaSetCommunication(h,requested_transport));
    DDLA_CHECK(ddlaSet(h,parent->comm,parent->nprows_,parent->npcols_,requested_major));
    exercise_on<B,T>(h,n,nev,uplo,alias,method,job);
    DDLA_CHECK(ddlaDestroy(h));
}

template<DdlaBackend B,class T>
void public_failure_order_on(const DdlaHandle_t& h) {
    using R=typename real_type<T>::type;
    Method restore;
    setenv("DDLA_EIGEN_METHOD","invalid-must-not-be-read",1);
    constexpr int n=7,nb=4;
    int d[DDLA_DLEN_]; DDLA_CHECK(ddlaDescInit(d,h,n,n,nb,nb,0,0));
    auto aa=make_local<T>(h,d,[](int i,int j){return generalized_test::a<T>(i,j,n);});
    auto bb=make_local<T>(h,d,[](int i,int j){return T(i==j?(i==2?-1:1):0);});
    DeviceBuffer<T> A(h,std::max(std::size_t(1),aa.size()));
    DeviceBuffer<T> C(h,std::max(std::size_t(1),bb.size()));
    DeviceBuffer<T> Z(h,std::max(std::size_t(1),aa.size()));
    aa.resize(A.count,T(9876));
    bb.resize(C.count,T(9876));
    upload(h,A.ptr,aa); upload(h,C.ptr,bb);
    std::vector<R> w(n,R(-9876)); int info=-1;
    phegvd<T>(h,'V','U',n,n,A.ptr,d,C.ptr,d,w.data(),Z.ptr,d,info);
    const auto after=download(h,A.ptr,A.count);
    require_close(h,"public B failure precedes invalid method and leaves A/w unchanged",
        info==n+3 && !std::memcmp(after.data(),aa.data(),aa.size()*sizeof(T)) && w==std::vector<R>(n,R(-9876))?0:1,0);
    phegvd<T>(h,'V','L',0,0,A.ptr,d,C.ptr,d,nullptr,nullptr,nullptr,info);
    require_close(h,"public zero order precedes invalid method",info,0);
}
template<DdlaBackend B,class T>
void public_failure_order(const DdlaHandle_t& parent)
{
    DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,B));
    if constexpr(B==DdlaBackend::GPU)DDLA_CHECK(ddlaSetCommunication(h,requested_transport));
    DDLA_CHECK(ddlaSet(h,parent->comm,parent->nprows_,parent->npcols_,requested_major));
    public_failure_order_on<B,T>(h);
    DDLA_CHECK(ddlaDestroy(h));
}
template<DdlaBackend B,class T>
void cases(const DdlaHandle_t& h) {
    exercise<B,T>(h,0,0,'U',0,"auto");
    exercise<B,T>(h,1,1,'L',0,"auto");
    for (const char* method:{"auto","one_stage","two_stage"}) for (char uplo:{'U','L'}) {
        exercise<B,T>(h,17,0,uplo,0,method);
        exercise<B,T>(h,17,5,uplo,0,method);
        exercise<B,T>(h,17,5,uplo,1,method);
        exercise<B,T>(h,17,5,uplo,2,method);
        exercise<B,T>(h,17,17,uplo,0,method,'N');
    }
    public_failure_order<B,T>(h);
}
void body(const DdlaHandle_t& h,const Shape&) {
    constexpr auto B=ddla_test::execution_backend;
    cases<B,float>(h); cases<B,double>(h);
    cases<B,std::complex<float>>(h); cases<B,std::complex<double>>(h);
}
} // namespace
int main(int argc,char** argv)
{
    // Preserve the shared grid parser; consume only this fixture's transport
    // and rank-major controls before MPI/handle creation.
    int kept=1;
    for(int i=1;i<argc;++i){
        const std::string arg=argv[i];
        if(arg=="--transport"&&i+1<argc){
            const std::string value=argv[++i];
            if(value=="staged")requested_transport=DdlaCommunication::HostStagedMPI;
            else if(value=="ccl")requested_transport=DdlaCommunication::CCL;
            else if(value=="mpi")requested_transport=DdlaCommunication::MPI;
            else throw std::invalid_argument("--transport requires staged, ccl or mpi");
        }else if(arg=="--major"&&i+1<argc){
            const std::string value=argv[++i];
            if(value!="R"&&value!="C")throw std::invalid_argument("--major requires R or C");
            requested_major=value[0];
        }else argv[kept++]=argv[i];
    }
    argc=kept;
    return run_grid_test(argc,argv,"test_workspace_phegvd",body);
}
