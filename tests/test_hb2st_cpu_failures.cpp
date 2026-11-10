// Private allocation-error regressions. No library hooks: interpose the same
// C++ allocator used by test_phb2st_no_records, plus HostArray's POSIX allocator.
// Run with a timeout: a pre-fix rank-local failure strands the other ranks.
#include "api_grid_test_common.h"
#include "Eigensolver/two_stage/distributed_history.h"
#include "Eigensolver/eig_cpu_kernels.h"
#include "Eigensolver/two_stage/hb2st_host_workspace.h"
#include "Eigensolver/two_stage/hb2st_sink.h"
#include "Eigensolver/two_stage/phb2st_impl.h"
#include <ddla/ddla_vector.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <limits>
#include <new>
#include <thread>

namespace allocation_failure {
enum Kind { cpp_new=1, aligned=2 };
std::atomic<int> armed{0}, hits{0}, skip{0};
int selected_kind=0;
std::size_t selected_bytes=0;
bool background=false;
std::thread::id main_thread;

bool reject(int kind,std::size_t bytes) noexcept {
    if(!armed.load(std::memory_order_acquire) || kind!=selected_kind ||
       (selected_bytes && bytes!=selected_bytes) ||
       (background && std::this_thread::get_id()==main_thread)) return false;
    if(skip.fetch_sub(1,std::memory_order_relaxed)>0) return false;
    int yes=1;
    if(!armed.compare_exchange_strong(yes,0,std::memory_order_acq_rel)) return false;
    hits.fetch_add(1,std::memory_order_relaxed);
    return true;
}
void enable(bool victim,int kind,std::size_t bytes,int skipped=0,bool bg=false) {
    hits.store(0);selected_kind=kind;selected_bytes=bytes;skip=skipped;background=bg;
    armed.store(victim?1:0,std::memory_order_release);
}
void disable() {armed.store(0,std::memory_order_release);}
}
void* operator new(std::size_t bytes) {
    if(allocation_failure::reject(allocation_failure::cpp_new,bytes)) throw std::bad_alloc();
    void* p=std::malloc(bytes?bytes:1);if(!p) throw std::bad_alloc();return p;
}
void* operator new[](std::size_t n) {return ::operator new(n);}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete[](void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
void operator delete[](void* p,std::size_t) noexcept {std::free(p);}
extern "C" int posix_memalign(void** out,std::size_t alignment,std::size_t bytes) noexcept(noexcept(::posix_memalign(out,alignment,bytes))) {
    if(allocation_failure::reject(allocation_failure::aligned,bytes)) return ENOMEM;
    if(alignment<sizeof(void*) || (alignment&(alignment-1))) return EINVAL;
    if(bytes>std::numeric_limits<std::size_t>::max()-(alignment-1)) return ENOMEM;
    const std::size_t rounded=((std::max<std::size_t>(bytes,1)+alignment-1)/alignment)*alignment;
    void* p=std::aligned_alloc(alignment,rounded);if(!p) return ENOMEM;*out=p;return 0;
}
void* operator new(std::size_t n,std::align_val_t alignment) {
    void* p=nullptr;if(posix_memalign(&p,static_cast<std::size_t>(alignment),n)) throw std::bad_alloc();return p;
}
void* operator new[](std::size_t n,std::align_val_t a) {return ::operator new(n,a);}
void operator delete(void* p,std::align_val_t) noexcept {std::free(p);}
void operator delete[](void* p,std::align_val_t) noexcept {std::free(p);}
void operator delete(void* p,std::size_t,std::align_val_t) noexcept {std::free(p);}
void operator delete[](void* p,std::size_t,std::align_val_t) noexcept {std::free(p);}

namespace {
void require(bool ok,const char* what) {if(!ok) throw std::runtime_error(what);}
void check_failure(int caught,const char* label,int rank,int size) {
    allocation_failure::disable();
    int all=0,local=allocation_failure::hits.load(),hits=0;
    MPI_Allreduce(&caught,&all,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(&local,&hits,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    require(hits==1,"allocation injection did not hit exactly one rank");
    require(all==size,"allocation failure did not reach every rank");
    if(rank==0) std::printf("CPU_FAILURE %s caught=%d injected=%d PASS\n",label,all,hits);
}

template<class T> void replay_failures(int rank,int size) {
    using R=typename ddla::real_type<T>::type;
    constexpr int n=1025,kd=8;
    const int ncs=(size>=3 && rank==size-1)?0:32;
    ddla::detail::DistributedHistory<T> history;
    history.prepare(n,kd,MPI_COMM_WORLD);
    require(history.chunks()>=2,"test needs multiple history chunks");
    std::fill_n(history.cosines.get(),history.local_count,R(1));
    std::fill_n(history.sines.get(),history.local_count,T(0));
    std::vector<R> y(static_cast<std::size_t>(n)*32,R(1));
    std::vector<T> phases(n-1,T(1)),send(static_cast<std::size_t>(n)*std::max(ncs,1));
    std::vector<int> cols(32);for(int j=0;j<32;++j) cols[j]=j;
    const int sd=0,victim=size>1?1:0;
    struct Case {const char* name;int kind;std::size_t bytes;int skip;};
    // The convenience adapter allocates all of its scratch during preparation.
    // Tile, record and wave buffers cannot allocate during apply.
    const Case cases[]={{"replay workspace preparation",allocation_failure::cpp_new,0,0}};
    for(const auto& c:cases) {
        allocation_failure::enable(rank==victim,c.kind,c.bytes,c.skip);
        int caught=0;
        try {
            ddla::detail::cpu_replay_pack_distributed<T>(y.data(),y.size(),n,cols.data(),ncs,n,history,MPI_COMM_WORLD,
                std::is_same<T,R>::value?nullptr:phases.data(),3,1,&sd,send.data(),1);
        } catch(const std::exception&) {caught=1;}
        check_failure(caught,c.name,rank,size);
    }
    // A subsequent normal call on the same communicator/history must work.
    ddla::detail::cpu_replay_pack_distributed<T>(y.data(),y.size(),n,cols.data(),ncs,n,history,MPI_COMM_WORLD,
        std::is_same<T,R>::value?nullptr:phases.data(),3,1,&sd,send.data(),1);
    for(int i=0;i<n*ncs;++i) require(send[i]==T(1),"identity replay recovery differs");
}

// The prepared planner allocates no scratch. Corrupt only the record stream
// it is handed so the real ordering/capacity check takes the same
// cooperative producer/consumer unwind path without a production hook.
class ShiftedSink final:public ddla::Hb2stSink<double> {
    ddla::Hb2stSink<double>& sink_;bool shifted_;
public:
    ShiftedSink(ddla::Hb2stSink<double>& sink,bool shifted):sink_(sink),shifted_(shifted){}
    bool window_ring_allowed()const override{return sink_.window_ring_allowed();}
    long long stream_chunk()const override{return sink_.stream_chunk();}
    bool elastic_ring()const override{return sink_.elastic_ring();}
    bool drains_in_stage2()const override{return sink_.drains_in_stage2();}
    void start_stager_run(const ddla::hb2st_rot<double>* records,long long cap,const std::atomic<long long>* emitted,
        const ddla::StreamWindow* window,const std::atomic<int>* aborted,int chase_threads,bool roles)override{
        sink_.start_stager_run(records+(shifted_?1:0),cap-(shifted_?1:0),emitted,window,aborted,chase_threads,roles);}
    int extra_roles()const override{return sink_.extra_roles();}
    void run_role(int role)override{sink_.run_role(role);}
    void mark_chase_done()override{sink_.mark_chase_done();}
    std::atomic<long long>*fallback_emitted()override{return sink_.fallback_emitted();}
    void drain_stage2_run()override{sink_.drain_stage2_run();}
    void cancel_run()override{sink_.cancel_run();}
    const std::exception_ptr&consumer_error()const override{return sink_.consumer_error();}
    long long scheduled_rotations()const override{return sink_.scheduled_rotations();}
};

void sink_failure(const ddla::DdlaHandle_t& h,bool team,int groups,int rank,int size) {
    constexpr int n=1200,kd=8;
    setenv("DDLA_EIG_CHASE_TEAM",team?"1":"0",1);
    ddla::detail::set_phb2st_test_partitions(groups);
    auto topology=ddla::spectral_impl::phb2st_topology_prepare(h);
    const auto sp=ddla::detail::hb2st_accum_host_check(h,n,kd,ddla::detail::host_omp_threads(MPI_COMM_WORLD));
    int desc[9];
    require(ddla::ddlaDescInit(desc,h,n,n,8,8,0,0)==ddla::ddlaStatus_t::DDLA_STATUS_SUCCESS,"descriptor");
    auto a=api_grid_test::make_local<double>(h,desc,[](int i,int j) {
        return i==j?4.0:(i<j && j-i<=kd?0.03125*((i+j)%7-3):0.0);
    });
    if(a.empty()) a.resize(1);
    std::vector<double> d(n),e(n),phases(n-1);
    // The prepared chase feeding the CPU sink: size -> allocate once -> run.
    ddla::spectral_impl::Phb2stSinkPolicy policy;policy.chunk=sp.block_records;policy.max_roles=std::max(2,sp.threads);policy.roles_in_team=true;
    const auto p=ddla::spectral_impl::phb2st_check<double>(h,ddla::DdlaBackend::CPU,n,kd,a.data(),desc,d.data(),e.data(),phases.data(),phases.size(),
        *topology,ddla::spectral_impl::Phb2stOutput::Sink,policy);
    std::size_t sink_bytes=0,backend_bytes=0,host_bytes=0;
    ddla::detail::hb2st_accum_host_workspace_size<double>(h,sp,&sink_bytes);ddla::spectral_impl::phb2st_workspace_size<double>(p,&backend_bytes,&host_bytes);
    ddla::vector<char> sink_work(h,sink_bytes),backend_work(h,backend_bytes);std::vector<char> host_work(host_bytes);
    auto sink=ddla::detail::hb2st_accum_host_prepare<double>(h,sp);ddla::spectral_impl::Phb2stResources<double> resources(h,p);
    auto chase=[&](ddla::Hb2stSink<double>& consumer){
        std::size_t count=0;sink->reset_run(sink_work.data(),sink_bytes);
        try{ddla::spectral_impl::phb2st_run<ddla::DdlaBackend::CPU,double>(h,p,resources,a.data(),d.data(),e.data(),phases.data(),count,&consumer,
            backend_work.data(),backend_bytes,host_work.data(),host_bytes);}
        catch(...){sink->cancel_run();throw;}
    };
    ShiftedSink shifted(*sink,rank==0);int caught=0;
    try{chase(shifted);}
    catch(const std::exception&){caught=1;}
    int all=0,local_error=sink->consumer_error()?1:0,errors=0;
    MPI_Allreduce(&caught,&all,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);MPI_Allreduce(&local_error,&errors,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    require(all==size,"sink ordering failure did not reach every rank");require(errors==1,"sink ordering fault did not reach exactly one real planner");
    if(rank==0)std::printf("CPU_FAILURE %s caught=%d planner_errors=%d PASS\n",groups?"spatial planner":team?"team planner":"fallback planner",all,errors);
    // The same prepared worker pool and scratch must recover after every
    // producer/consumer has joined, including the failed planner.
    chase(*sink);
    require(!sink->consumer_error(),"sink did not recover after the ordering fault");
    for(auto value:d)require(std::isfinite(value),"recovered sink diagonal is not finite");for(int i=0;i<n-1;++i)require(std::isfinite(e[i]),"recovered sink off-diagonal is not finite");
    resources.release();topology->release();
    ddla::detail::set_phb2st_test_partitions(0);
}
}
int main(int argc,char** argv) {
    int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int rank=0,size=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    allocation_failure::main_thread=std::this_thread::get_id();
    try {
        require(provided>=MPI_THREAD_FUNNELED,"MPI thread support");
        setenv("DDLA_EIG_REPLAY_WAVE","1",1);
        setenv("DDLA_EIG_THREADS","3",1);
        setenv("DDLA_EIG_CHASE_THREADS","1",1);
        setenv("DDLA_EIG_CHASE_TEAM_THREADS","1",1);
        setenv("DDLA_EIG_CPU_ACCUM","1",1);
        setenv("DDLA_EIG_HOST_ACCUM_BLOCK","4096",1);
        replay_failures<float>(rank,size);replay_failures<double>(rank,size);
        replay_failures<std::complex<float>>(rank,size);replay_failures<std::complex<double>>(rank,size);
        ddla::DdlaHandle_t h=nullptr;
        require(ddla::ddlaInit(h,ddla::DdlaBackend::CPU)==ddla::ddlaStatus_t::DDLA_STATUS_SUCCESS,"CPU handle");
        require(ddla::ddlaSet(h,MPI_COMM_WORLD,1,size)==ddla::ddlaStatus_t::DDLA_STATUS_SUCCESS,"grid");
        sink_failure(h,false,0,rank,size);sink_failure(h,true,0,rank,size);
        if(size>1) sink_failure(h,true,2,rank,size);
        ddla::ddlaDestroy(h);
        if(rank==0) std::puts("test_hb2st_cpu_failures passed");
    } catch(const std::exception& e) {
        allocation_failure::disable();std::fprintf(stderr,"rank %d: %s\n",rank,e.what());MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Finalize();return 0;
}
