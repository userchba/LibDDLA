// The private bounded-history path must reproduce the public recorded chase
// exactly, including chunks split through a recording step and empty owners.
#include "api_grid_test_common.h"
#include "Eigensolver/eig_params.h"
#include "Eigensolver/eig_cpu_kernels.h"
#include "Eigensolver/two_stage/phb2st_impl.h"
#include <cstring>
#include <numeric>

namespace {
using namespace api_grid_test;
void require(bool condition, const char* what) {
    if (!condition) throw std::runtime_error(what);
}
template<class T> bool bits(const std::vector<T>& a,const std::vector<T>& b) {
    return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);
}
template<class T> T value(int i,int j,int kd) {
    using R=typename ddla::real_type<T>::type;
    if(i==j) return T(R(4)+R(i%13)/R(8));
    if(i>j || j-i>kd) return T(-913);
    const R x=R(((i+1)*7+(j+1)*13)%19-9)/R(16);
    if constexpr(std::is_same_v<T,R>) return x;
    else return T(x,R(((i+1)*11+(j+1)*3)%17-8)/R(32));
}
template<ddla::DdlaBackend Backend,class T> void run(const ddla::DdlaHandle_t& h,int n,int kd,bool team) {
    using R=typename ddla::real_type<T>::type;
    setenv("DDLA_EIG_CHASE_TEAM",team?"1":"0",1);
    int desc[9];
    require(ddla::ddlaDescInit(desc,h,n,n,3,3,0,0)==ddla::ddlaStatus_t::DDLA_STATUS_SUCCESS,"descriptor");
    desc[ddla::DDLA_LLD_]+=3;
    auto a=make_local<T>(h,desc,[&](int i,int j){return value<T>(i,j,kd);});
    if(a.empty()) a.resize(1,T(-719));
    const auto original=a;
    DeviceBuffer<T> buffer(h,a.size()); upload(h,buffer.ptr,a); check_ddla_sync(h);
    std::vector<R> d(n),e(n),dref(n),eref(n);
    std::vector<T> ph,phref;
    std::vector<ddla::hb2st_rot<T>> ref;
    int info=-1;
    const int ranks=ddla::ddlaGetSize(h);
    if(ranks>1 && n>2 && kd>ddla::eig_distributed_band_limit(n,ranks)) {
        ddla::detail::DistributedHistory<T> rejected_history;
        bool rejected=false;
        try {
            ddla::phb2st_node<Backend,T>(h,n,kd,buffer.ptr,desc,d.data(),e.data(),
                rejected_history,ph,true,info);
        } catch(const std::invalid_argument& error) {
            rejected=std::strstr(error.what(),"band is too wide")!=nullptr;
        }
        require(rejected,"distributed chase accepted a replicated wide band");
        return;
    }
    // Independent serial oracle. Complete records exist only in this test.
    ddla::spectral_impl::Phb2stHostPlan serial;
    serial.n=n; serial.band=kd; serial.records=true;
    std::vector<T> band(std::size_t(kd+1)*n,T(0));
    for(int j=0;j<n;++j)for(int i=std::max(0,j-kd);i<=j;++i)
        band[std::size_t(j)*(kd+1)+kd+i-j]=value<T>(i,j,kd);
    ddla::detail::ChaseCursor cursor(n,kd);
    while(!cursor.done())cursor.next();
    ref.resize(std::size_t(cursor.base));
    phref.resize(std::is_same_v<T,R>?0:std::size_t(n-1));
    std::size_t bytes=0,count=0;
    ddla::spectral_impl::phb2st_host_workspace_size<T>(serial,&bytes);
    std::vector<char> scratch(bytes);
    ddla::spectral_impl::phb2st_host_run<T>(serial,nullptr,band.data(),dref.data(),eref.data(),
        ref.data(),phref.data(),count,nullptr,nullptr,nullptr,scratch.data(),bytes);
    phref.resize(count);
    setenv("DDLA_EIG_CHASE_TEAM",team?"1":"0",1);
    ddla::detail::DistributedHistory<T> stream;
    ddla::phb2st_node<Backend,T>(h,n,kd,buffer.ptr,desc,d.data(),e.data(),stream,ph,true,info);
    require(info==0 && bits(d,dref) && bits(e,eref) && bits(ph,phref),"distributed d/e/phase mismatch");
    require(bits(download(h,buffer.ptr,a.size()),original),"input or padding changed");
    require(stream.ready,"full retained history exists");
    const auto& hist=stream;
    require(hist.total==static_cast<long long>(ref.size()),"record count mismatch");
    std::vector<ddla::hb2st_rot<T>> chunk,compact;
    for(long long q=hist.chunks();q-->0;) {
        hist.fetch(q,ddla::ddlaGetCommunicator(h),chunk);
        hist.fetch(q,ddla::ddlaGetCommunicator(h),compact,nullptr,true);
        require(compact.size()==chunk.size(),"compact history size mismatch");
        for(std::size_t i=0;i<chunk.size();++i) {
            const auto& expected=ref[static_cast<std::size_t>(q*hist.chunk_size)+i];
            require(chunk[i].j0==expected.j0 && std::memcmp(&chunk[i].c,&expected.c,sizeof(R))==0 &&
                    std::memcmp(&chunk[i].s,&expected.s,sizeof(T))==0,"record fields differ");
            require(compact[i].j0==expected.j0 && std::memcmp(&compact[i].c,&expected.c,sizeof(R))==0 &&
                    std::memcmp(&compact[i].s,&expected.s,sizeof(T))==0,"compact record fields differ");
        }
    }
    // One rank has no output columns, but must still fetch every chunk.
    const int rank=ddla::ddlaGetRank(h);
    const int ncs=rank%3==2 ? 0 : 19;
    std::vector<R> y(static_cast<std::size_t>(n)*std::max(ncs,1));
    for(std::size_t i=0;i<y.size();++i) y[i]=R(static_cast<int>(i%37)-18)/R(32);
    std::vector<int> cols(ncs); std::iota(cols.begin(),cols.end(),0);
    std::vector<T> actual(static_cast<std::size_t>(n)*ncs),expected(actual.size());
    int sd=0;
    const T* phase=ph.empty()?nullptr:ph.data();
    // These are generic RHS columns: the allocated slab has ncs columns,
    // which can exceed n for the small cases. Pass its actual element bound.
    if(n==1){
        bool rejected=ncs==0;
        if(ncs)try{ddla::detail::cpu_replay_pack<T>(y.data(),y.size()-1,n,cols.data(),ncs,n,ref.data(),ref.size(),phase,3,1,&sd,expected.data(),2);}catch(const std::invalid_argument& e){rejected=std::strstr(e.what(),"input column exceeds supplied slab")!=nullptr;}
        require(rejected,"short generic RHS slab accepted by local replay");
        rejected=false;
        try{ddla::detail::cpu_replay_pack_distributed<T>(y.data(),y.size()-1,n,cols.data(),ncs,n,hist,ddla::ddlaGetCommunicator(h),phase,3,1,&sd,actual.data(),2);}catch(const std::exception& e){rejected=ncs==0||std::strstr(e.what(),"input column exceeds supplied slab")!=nullptr;}
        require(rejected,"short generic RHS slab not rejected collectively");
    }
    ddla::detail::cpu_replay_pack<T>(y.data(),y.size(),n,cols.data(),ncs,n,ref.data(),ref.size(),phase,3,1,&sd,expected.data(),2);
    ddla::detail::cpu_replay_pack_distributed<T>(y.data(),y.size(),n,cols.data(),ncs,n,hist,ddla::ddlaGetCommunicator(h),phase,3,1,&sd,actual.data(),2);
    require(bits(actual,expected),"chunk replay differs from full replay");
    unsigned long long local=hist.local_count,sum=0;
    MPI_Allreduce(&local,&sum,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,ddla::ddlaGetCommunicator(h));
    require(sum==ref.size(),"retained history is not uniquely distributed");
    stream.clear();
    if(rank==0) std::printf("DISTRIBUTED_HISTORY n=%d kd=%d type_bytes=%zu complex=%d team=%d PASS\n",n,kd,sizeof(T),std::is_same_v<T,R>?0:1,team?1:0);
}
template<ddla::DdlaBackend Backend,class T> void cases(const ddla::DdlaHandle_t& h) {
    for(bool team:{false,true})
        for(auto p:{std::pair<int,int>{1,1},{2,1},{3,2},{5,6},{37,4},{769,8}})
            run<Backend,T>(h,p.first,p.second,team);
}
template<ddla::DdlaBackend Backend> void run_checks(const ddla::DdlaHandle_t& h) {
    cases<Backend,float>(h); cases<Backend,double>(h);
    cases<Backend,std::complex<float>>(h); cases<Backend,std::complex<double>>(h);
    run<Backend,double>(h,1453,8,true);
    for(int groups:{2,3}) {
        ddla::detail::set_phb2st_test_partitions(groups);
        for(auto shape:{std::pair<int,int>{1,1},{3,2},{5,6},{37,4},{769,8}}) {
            run<Backend,float>(h,shape.first,shape.second,true);
            run<Backend,double>(h,shape.first,shape.second,true);
            run<Backend,std::complex<float>>(h,shape.first,shape.second,true);
            run<Backend,std::complex<double>>(h,shape.first,shape.second,true);
        }
    }
    ddla::detail::set_phb2st_test_partitions(0);
}

}
int main(int argc,char** argv) {
    int provided=0;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int rank=0,size=0; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    try {
        require(provided>=MPI_THREAD_FUNNELED,"MPI thread support");
        api_grid_test::TestOptions opts; std::string error;
        require(api_grid_test::parse_options(argc,argv,size,opts,error),error.c_str());
        auto backend=ddla::default_backend_v;
        if(const char* arg=std::getenv("DDLA_TEST_BACKEND")) {
            const std::string name(arg);
            if(name=="CPU" || name=="cpu") backend=ddla::DdlaBackend::CPU;
            else if(name=="GPU" || name=="gpu") backend=ddla::DdlaBackend::GPU;
            else throw std::runtime_error("invalid DDLA_TEST_BACKEND");
        }
        for(auto grid:opts.grids) {
            ddla::DdlaHandle_t h=nullptr;
            require(ddla::ddlaInit(h,backend)==ddla::ddlaStatus_t::DDLA_STATUS_SUCCESS,"handle");
            require(ddla::ddlaSet(h,MPI_COMM_WORLD,grid.first,grid.second)==ddla::ddlaStatus_t::DDLA_STATUS_SUCCESS,"grid");
            switch(backend) {
            case ddla::DdlaBackend::CPU: run_checks<ddla::DdlaBackend::CPU>(h); break;
#if DDLA_HAS_GPU
            case ddla::DdlaBackend::GPU: run_checks<ddla::DdlaBackend::GPU>(h); break;
#endif
            default: throw std::runtime_error("backend unavailable");
            }
            ddla::ddlaDestroy(h);
        }
        if(rank==0) std::puts("test_phb2st_distributed passed");
    } catch(const std::exception& e) {
        std::fprintf(stderr,"test_phb2st_distributed rank=%d failed: %s\n",rank,e.what()); MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Finalize(); return 0;
}
