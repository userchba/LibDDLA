// One executable exercises handle-local transport selection and the same math
// across transports. Device-pointer MPI is opt-in, never inferred from a build.
#include "comm_test_common.h"
#include "api_grid_test_common.h"
#include "eig_test_common.h"
#include "Eigensolver/one_stage/phetrd_impl.h"
#include <string>

using namespace ddla;
namespace {
using Mode=DdlaCommunication;
using Status=ddlaStatus_t;
using Complex=std::complex<double>;
void require(DdlaHandle_t h,bool good,const char* label) {
    const MPI_Comm comm=h->comm==MPI_COMM_NULL?MPI_COMM_WORLD:h->comm;
    int bad=!good,any=0;MPI_CHECK(MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,comm));
    if(any)throw std::runtime_error(label);
}
void expect_status(DdlaHandle_t h,Status got,Status want,const char* label) { require(h,got==want,label); }
void resources(DdlaHandle_t h) {
    const bool ccl=h->backend==DdlaBackend::GPU&&ddlaGetCommunication(h)==Mode::CCL;
    require(h,ccl?(h->nccl_comm&&h->nccl_row_comm&&h->nccl_col_comm):
        (!h->nccl_comm&&!h->nccl_row_comm&&!h->nccl_col_comm),"CCL slots follow selected mode");
}

struct Math {
    DdlaHandle_t h;
    static constexpr int n=17,nb=4;
    int d[DDLA_DLEN_];
    std::vector<Complex> ah,bh,ch;
    vector<Complex> a,b,c;
    explicit Math(DdlaHandle_t handle):h(handle),a(h),b(h),c(h) {
        DDLA_CHECK(ddlaDescInit(d,h,n,n,nb,nb,0,0));d[DDLA_LLD_]+=2;
        ah=api_grid_test::make_local<Complex>(h,d,[](int i,int j){return api_grid_test::general_value(i,j,1);});
        bh=api_grid_test::make_local<Complex>(h,d,[](int i,int j){return api_grid_test::general_value(i,j,2);});
        ch=api_grid_test::make_local<Complex>(h,d,[](int i,int j){return api_grid_test::general_value(i,j,3);});
        a.resize(ah.size());b.resize(bh.size());c.resize(ch.size());
    }
    void products() {
        const Complex alpha(.8,-.2),beta(-.3,.1);
        for(char trans:{'N','C'}) {
            api_grid_test::upload(h,a.data(),ah);api_grid_test::upload(h,b.data(),bh);api_grid_test::upload(h,c.data(),ch);
            pgemm<Complex>(h,trans,trans,n,n,n,alpha,a.data(),d,b.data(),d,beta,c.data(),d);
            const auto out=api_grid_test::download(h,c.data(),ch.size());
            const double error=api_grid_test::local_max_error<Complex>(h,d,out,[&](int i,int j) {
                Complex ref=beta*api_grid_test::general_value(i,j,3);
                for(int k=0;k<n;++k)ref+=alpha*
                    api_grid_test::op_value(trans,n,n,i,k,api_grid_test::general_value,1)*
                    api_grid_test::op_value(trans,n,n,k,j,api_grid_test::general_value,2);
                return ref;
            });
            api_grid_test::require_close(h,"transport PGEMM independent oracle",error,2e-10);
            require(h,api_grid_test::download(h,a.data(),ah.size())==ah && api_grid_test::download(h,b.data(),bh.size())==bh,
                    "PGEMM keeps inputs read-only");
            bool padding=true;
            for(int j=0;j<ddla_test::n_loc(h,d);++j)for(int i=ddla_test::m_loc(h,d);i<d[DDLA_LLD_];++i)
                padding=padding&&out[i+std::size_t(j)*d[DDLA_LLD_]]==ch[i+std::size_t(j)*d[DDLA_LLD_]];
            require(h,padding,"PGEMM output padding");
        }
        auto herm=api_grid_test::make_local<Complex>(h,d,[](int i,int j){return api_grid_test::hpd_value(i,j,n);});
        api_grid_test::upload(h,a.data(),herm);api_grid_test::upload(h,b.data(),bh);api_grid_test::upload(h,c.data(),ch);
        phemm<Complex>(h,'L','U',n,n,alpha,a.data(),d,b.data(),d,beta,c.data(),d);
        const auto out=api_grid_test::download(h,c.data(),ch.size());
        const double error=api_grid_test::local_max_error<Complex>(h,d,out,[&](int i,int j) {
            Complex ref=beta*api_grid_test::general_value(i,j,3);
            for(int k=0;k<n;++k)ref+=alpha*api_grid_test::hpd_value(i,k,n)*api_grid_test::general_value(k,j,2);
            return ref;
        });
        api_grid_test::require_close(h,"transport PHEMM independent oracle",error,2e-10);
    }
    void eigensolvers() {
        auto input=api_grid_test::make_local<Complex>(h,d,[](int i,int j){return api_grid_test::hpd_value(i,j,n);});
        std::vector<double> w(n),reference;
        for(int method:{1,2}) {
            api_grid_test::upload(h,a.data(),input);api_grid_test::upload(h,c.data(),ch);
            int info=-1;
            if(method==1)pheevd_one_stage<Complex>(h,'V','U',n,a.data(),d,w.data(),c.data(),d,info);
            else pheevd_two_stage<Complex>(h,'V','U',n,a.data(),d,w.data(),c.data(),d,info);
            require(h,info==0,"transport eigensolver INFO");
            bool sorted=true;for(int i=0;i<n;++i)sorted=sorted&&std::isfinite(w[i])&&(!i||w[i]>=w[i-1]);
            require(h,sorted,"transport eigensolver sorted finite eigenvalues");
            api_grid_test::upload(h,a.data(),input);
            const double residual=eig_test::residual_metric<Complex>(h,a.data(),d,c.data(),d,w.data(),n);
            const double orth=eig_test::orthogonality_metric<Complex>(h,c.data(),d,n);
            api_grid_test::require_close(h,"transport eigensolver residual",residual,50);
            api_grid_test::require_close(h,"transport eigensolver orthogonality",orth,50);
            if(reference.empty())reference=w;
            else {double error=0;for(int i=0;i<n;++i)error=std::max(error,std::abs(w[i]-reference[i]));
                api_grid_test::require_close(h,"one-/two-stage eigenvalue agreement",error,2e-10);}
        }
    }
};

template<DdlaBackend B>void grouped_directional(DdlaHandle_t h) {
    // This is a directional gather, not a symmetric blocking Send/Recv cycle.
    // MPI grouping remains a no-op: symmetric exchanges use commSendrecv.
    const int n=131073,root=h->nprocs-1;
    vector<double> send(h,n),receive(h,std::size_t(n)*h->nprocs);
    std::vector<double> source(n,double(h->myid+1));
    DDLA_CHECK(ddlaMemcpy(h,send.data(),source.data(),n*sizeof(double)));
    commGroupStart<B>(h);
    if(h->myid==root) {
        for(int p=0;p<h->nprocs;++p) {
            if(p==root)DDLA_CHECK(ddlaMemcpy(h,receive.data()+std::size_t(p)*n,send.data(),n*sizeof(double)));
            else commRecv<B>(h,CommScope::Grid,receive.data()+std::size_t(p)*n,n,p);
        }
    } else commSend<B>(h,CommScope::Grid,send.data(),n,root);
    commGroupEnd<B>(h);
    bool good=true;
    if(h->myid==root) {
        const auto got=comm_test::download(h,receive);
        for(int p=0;p<h->nprocs;++p)for(int i=0;i<n;++i)good=good&&got[std::size_t(p)*n+i]==p+1;
    }
    require(h,good,"grouped directional gather above eager limit");
}

template<DdlaBackend B>void primitives(DdlaHandle_t h) {
    for(auto scope:{CommScope::Grid,CommScope::Row,CommScope::Col})for(int count:{0,37}) {
        comm_test::messages<B,float>(h,scope,count);comm_test::messages<B,double>(h,scope,count);
        comm_test::messages<B,std::complex<float>>(h,scope,count);comm_test::messages<B,Complex>(h,scope,count);
    }
    // 1 MiB is deliberately larger than the site's MPI eager limits.
    comm_test::messages<B,double>(h,CommScope::Grid,131073);
    grouped_directional<B>(h);
}

template<DdlaBackend B>void switch_to(DdlaHandle_t h,Mode next) {
    vector<unsigned char> pending(h,8*1024*1024);
#if DDLA_HAS_GPU
    if constexpr(B==DdlaBackend::GPU) {
        const std::size_t third=pending.size()/3;
        void* queues[]={h->stream,h->stream_data,h->level_queue()};
        for(int i=0;i<3;++i)GPU_CHECK(gpuMemsetAsync(pending.data()+i*third,0x3c,
            i==2?pending.size()-2*third:third,static_cast<gpuStream_t>(queues[i])));
    } else
#endif
    {DDLA_CHECK(ddlaMemset(h,pending.data(),0x3c,pending.size()));}
    DDLA_CHECK(ddlaSetCommunication(h,next));
    require(h,ddlaGetCommunication(h)==next,"switch commits selected mode");resources(h);
#if DDLA_HAS_GPU
    if constexpr(B==DdlaBackend::GPU) {
        require(h,gpuStreamQuery(static_cast<gpuStream_t>(h->stream))==gpuSuccess &&
            gpuStreamQuery(static_cast<gpuStream_t>(h->stream_data))==gpuSuccess &&
            gpuStreamQuery(static_cast<gpuStream_t>(h->stream_level))==gpuSuccess,
            "transport switch drains handle queues");
    }
#endif
    const auto got=comm_test::download(h,pending);
    require(h,std::all_of(got.begin(),got.end(),[](unsigned char x){return x==0x3c;}),"pending work survives transport switch");
}

// A checked reduction plan cannot borrow an arena sized for another transport.
template<DdlaBackend B>void stale_reduction_plan(DdlaHandle_t h) {
#if DDLA_HAS_GPU && defined(DDLA_USE_CCL)
    if constexpr(B==DdlaBackend::GPU) {
        DDLA_CHECK(ddlaSetCommunication(h,Mode::HostStagedMPI));
        constexpr int n=17,guard=256;int d[DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(d,h,n,n,4,4,0,0));
        const auto input=api_grid_test::make_local<Complex>(h,d,[](int i,int j){return api_grid_test::hpd_value(i,j,n);});
        vector<Complex> a(h,input.size()),tau(h,n-1);
        const std::vector<Complex> tau_before(n-1,Complex(-91,13));
        std::vector<double> de(2*n,-971);
        api_grid_test::upload(h,a.data(),input);api_grid_test::upload(h,tau.data(),tau_before);
        const auto plan=spectral_impl::phetrd_check<Complex>(h,B,'U',n,a.data(),d,de.data(),de.data()+n,tau.data());
        std::size_t bb=0,hb=0;spectral_impl::phetrd_workspace_size<Complex>(h,plan,&bb,&hb);
        vector<unsigned char> work(h,bb+2*guard);std::vector<unsigned char> host(hb+2*guard,0x79);
        DDLA_CHECK(ddlaMemset(h,work.data(),0x79,work.size()));
        spectral_impl::HetrdResources old_resources(h,plan);
        DDLA_CHECK(ddlaSetCommunication(h,Mode::CCL));
        bool query_rejected=false;
        try {std::size_t x=0,y=0;spectral_impl::phetrd_workspace_size<Complex>(h,plan,&x,&y);}
        catch(const std::invalid_argument&){query_rejected=true;}
        require(h,query_rejected,"stale phetrd query rejects changed transport");
        bool run_rejected=false;int info=-1;
        try {spectral_impl::phetrd_run<B,Complex>(h,plan,old_resources,a.data(),tau.data(),de.data(),info,
            work.data()+guard,bb,host.data()+guard,hb);}
        catch(const std::invalid_argument&){run_rejected=true;}
        require(h,run_rejected,"stale phetrd run collectively rejects short transport workspace");
        const auto untouched=comm_test::download(h,work);
        require(h,std::all_of(untouched.begin(),untouched.end(),[](unsigned char x){return x==0x79;})&&
            std::all_of(host.begin(),host.end(),[](unsigned char x){return x==0x79;}),
            "stale phetrd rejection touches neither workspace body nor canaries");
        require(h,comm_test::download(h,a)==input&&comm_test::download(h,tau)==tau_before&&
            std::all_of(de.begin(),de.end(),[](double x){return x==-971;}),"stale phetrd rejection preserves inputs and outputs");
        const auto fresh=spectral_impl::phetrd_check<Complex>(h,B,'U',n,a.data(),d,de.data(),de.data()+n,tau.data());
        std::size_t cb=0,ch=0;spectral_impl::phetrd_workspace_size<Complex>(h,fresh,&cb,&ch);
        require(h,cb==bb+guard&&ch==hb,"fresh CCL reduction query includes exact extra flag slot");
        vector<unsigned char> fresh_work(h,cb+2*guard);std::vector<unsigned char> fresh_host(ch+2*guard,0x79);
        DDLA_CHECK(ddlaMemset(h,fresh_work.data(),0x79,fresh_work.size()));
        spectral_impl::HetrdResources fresh_resources(h,fresh);
        spectral_impl::phetrd_run<B,Complex>(h,fresh,fresh_resources,a.data(),tau.data(),de.data(),info,
            fresh_work.data()+guard,cb,fresh_host.data()+guard,ch);
        const auto out=comm_test::download(h,fresh_work);bool intact=true;
        for(int i=0;i<guard;++i)intact=intact&&out[i]==0x79&&out[guard+cb+i]==0x79&&
            fresh_host[i]==0x79&&fresh_host[guard+ch+i]==0x79;
        require(h,info==0&&intact&&std::all_of(de.begin(),de.end(),[](double x){return std::isfinite(x);}),
            "fresh CCL reduction succeeds with exact guarded workspace");
        DDLA_CHECK(ddlaSetCommunication(h,Mode::HostStagedMPI));
        if(!h->myid)std::puts("RUNTIME_PHETRD_STALE_PLAN_PASS");
    }
#else
    (void)h;
#endif
}

template<DdlaBackend B>void run(MPI_Comm comm,int pr,int pc,bool device_mpi) {
    DdlaHandle_t h=nullptr,other=nullptr;
    DDLA_CHECK(ddlaInit(h,B));
    const Mode initial=ddlaGetCommunication(h);
    require(h,ddlaCommunicationAvailable(h,initial),"default communication mode is available");
    if constexpr(B==DdlaBackend::CPU)require(h,initial==Mode::MPI,"CPU default MPI");
    else {
#if defined(DDLA_USE_CCL)
        require(h,initial==Mode::CCL,"GPU default uses available CCL");
#else
        require(h,initial==Mode::MPI,"GPU default uses MPI without CCL");
#endif
    }
    require(h,!h->nccl_comm&&!h->nccl_row_comm&&!h->nccl_col_comm,"pregrid choice creates no CCL resources");
    expect_status(h,ddlaSetCommunication(h,static_cast<Mode>(-9)),Status::DDLA_STATUS_INVALID_VALUE,"pregrid invalid mode");
    require(h,ddlaGetCommunication(h)==initial,"pregrid rejection preserves mode");
    if constexpr(B==DdlaBackend::GPU) {
        int local_rank=0,P=0;MPI_Comm_rank(comm,&local_rank);MPI_Comm_size(comm,&P);
        if(P>1) {
            DDLA_CHECK(ddlaSetCommunication(h,local_rank?Mode::MPI:Mode::HostStagedMPI));
            bool caught=false;try{DDLA_CHECK(ddlaSet(h,comm,pr,pc));}catch(const std::exception&){caught=true;}
            require(h,caught&&!h->initialized,"ddlaSet rejects mismatched pregrid modes collectively");
        }
    }
    const Mode pregrid=B==DdlaBackend::CPU?Mode::MPI:Mode::HostStagedMPI;
    DDLA_CHECK(ddlaSetCommunication(h,pregrid));
    require(h,ddlaGetCommunication(h)==pregrid,"pregrid mode selection");
    DDLA_CHECK(ddlaSet(h,comm,pr,pc));
    require(h,ddlaGetCommunication(h)==pregrid,"ddlaSet retains selected mode");resources(h);
    DDLA_CHECK(ddlaInit(other,DdlaBackend::CPU));DDLA_CHECK(ddlaSet(other,comm,pr,pc));
    const auto old=ddlaGetCommunication(h);
    expect_status(h,ddlaSetCommunication(h,static_cast<Mode>(-9)),Status::DDLA_STATUS_INVALID_VALUE,"invalid mode status");
    require(h,ddlaGetCommunication(h)==old,"invalid mode preserves state");
    require(h,!ddlaCommunicationAvailable(h,static_cast<Mode>(-9)),"invalid mode unavailable");
    if(h->nprocs>1) {
        const Mode mismatch=h->myid?Mode::MPI:Mode::HostStagedMPI;
        expect_status(h,ddlaSetCommunication(h,mismatch),Status::DDLA_STATUS_INVALID_VALUE,"rank-mismatched transport rejected collectively");
        require(h,ddlaGetCommunication(h)==old,"rank mismatch preserves state");
        expect_status(h,ddlaSetCommunication(h,h->myid?old:static_cast<Mode>(std::numeric_limits<int>::min())),
            Status::DDLA_STATUS_INVALID_VALUE,"rank-local INT_MIN choice rejected collectively");
        require(h,ddlaGetCommunication(h)==old,"rank-local invalid choice preserves state");
    }
    for(Mode mode:{Mode::MPI,Mode::HostStagedMPI,Mode::CCL})if(!ddlaCommunicationAvailable(h,mode)) {
        expect_status(h,ddlaSetCommunication(h,mode),Status::DDLA_STATUS_BACKEND_UNAVAILABLE,"unsupported mode status");
        require(h,ddlaGetCommunication(h)==old,"unsupported mode preserves state");
    }
    commGroupStart<B>(h);
    const auto active=ddlaSetCommunication(h,old);
    commGroupEnd<B>(h);
    expect_status(h,active,Status::DDLA_STATUS_INVALID_VALUE,"setter rejects active communication group");
    require(h,ddlaGetCommunication(h)==old,"active group rejection preserves state");
    {
    Math math(h);
    comm_test::lazy_growth<B>(h); // fresh staged handle, before message cache exists
    stale_reduction_plan<B>(h);
    std::vector<Mode> modes;
    if constexpr(B==DdlaBackend::CPU)modes.push_back(Mode::MPI);
    else {
        modes.push_back(Mode::HostStagedMPI);
        if(ddlaCommunicationAvailable(h,Mode::CCL))modes.push_back(Mode::CCL);
        if(device_mpi)modes.push_back(Mode::MPI);
        else {
            // Selection is testable even when this MPI cannot exchange device pointers.
            switch_to<B>(h,Mode::MPI);
            switch_to<B>(h,Mode::HostStagedMPI);
            if(!h->myid)std::cout<<"GPU_DIRECT_MPI_TRAFFIC_SKIPPED explicit_flag_required=1\n";
        }
    }
    for(int cycle=0;cycle<2;++cycle) {
        for(Mode mode:modes) {
            if(mode!=ddlaGetCommunication(h))switch_to<B>(h,mode);
            else DDLA_CHECK(ddlaSetCommunication(h,mode));
            const void* before[]={h->nccl_comm,h->nccl_row_comm,h->nccl_col_comm};
            workspace_test::Scope same_mode;
            DDLA_CHECK(ddlaSetCommunication(h,mode)); // same mode is legal and collective
            const auto trace=same_mode.finish();
            require(h,!trace.ddla_alloc&&!trace.ddla_free&&!trace.pinned_alloc&&!trace.raw_gpu_alloc&&
                !trace.vendor_query&&!trace.resource_setup&&!trace.syncs,"same-mode setter has no resource/queue work");
            require(h,before[0]==h->nccl_comm&&before[1]==h->nccl_row_comm&&before[2]==h->nccl_col_comm,
                "same-mode setter retains CCL resources");
            resources(h);
            if(cycle==0) {
                commGroupStart<B>(h);const auto grouped_status=ddlaSetCommunication(h,mode);commGroupEnd<B>(h);
                expect_status(h,grouped_status,Status::DDLA_STATUS_INVALID_VALUE,"active group rejects setter in each transport");
            }
            require(h,ddlaGetCommunication(other)==Mode::MPI,"another handle keeps its transport");
            comm_test::messages<DdlaBackend::CPU,double>(other,CommScope::Grid,37);
            if(!h->myid)std::cout<<"RUNTIME_COMM_MODE backend="<<int(B)<<" mode="<<int(mode)<<" cycle="<<cycle
                <<" grid="<<pr<<'x'<<pc<<std::endl;
            primitives<B>(h);math.products();
            if(cycle==0)math.eigensolvers();
        }
        std::reverse(modes.begin(),modes.end());
    }
    } // matrix buffers are released while their handle is still alive
    const Mode final_mode=ddlaGetCommunication(h);
    DDLA_CHECK(ddlaSet(h,comm,pr,pc));
    require(h,ddlaGetCommunication(h)==final_mode,"repeated ddlaSet preserves communication selection");resources(h);
    comm_test::messages<B,double>(h,CommScope::Grid,37);
    // Also destroy after a nondefault selection, and in opposite handle order.
    DDLA_CHECK(ddlaDestroy(other));DDLA_CHECK(ddlaDestroy(h));
}
}
int main(int argc,char**argv) {
    ddla_test_mpi_init(&argc,&argv);
    try {
        bool cpu_only=false,device_mpi=false;int size=0;MPI_Comm_size(MPI_COMM_WORLD,&size);
        int pr=1,pc=size;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--cpu-only")cpu_only=true;
            else if(arg=="--gpu-aware-mpi")device_mpi=true;
            else if(arg=="--grid" && i+1<argc) {
                if(!api_grid_test::parse_grid_spec(argv[++i],pr,pc)||pr*pc!=size)throw std::invalid_argument("grid must cover all ranks");
            } else throw std::invalid_argument("usage: test_runtime_communication [--grid RxC] [--cpu-only] [--gpu-aware-mpi]");
        }
        DdlaHandle_t null=nullptr;
        if(ddlaSetCommunication(null,Mode::MPI)!=Status::DDLA_STATUS_INVALID_HANDLE ||
           ddlaCommunicationAvailable(null,Mode::MPI))throw std::runtime_error("null communication contract");
        bool caught=false;try{(void)ddlaGetCommunication(null);}catch(const std::runtime_error&){caught=true;}
        if(!caught)throw std::runtime_error("null communication getter must throw");
        // Reverse ranks to expose assumptions about world rank vs handle rank.
        int rank=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm reversed;
        MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD,0,size-rank,&reversed));
        run<DdlaBackend::CPU>(reversed,pr,pc,device_mpi);
#if DDLA_HAS_GPU
        if(!cpu_only)run<DdlaBackend::GPU>(reversed,pr,pc,device_mpi);
#else
        (void)cpu_only;(void)device_mpi;
#endif
        MPI_CHECK(MPI_Comm_free(&reversed));
        if(!rank)std::puts("RUNTIME_COMMUNICATION_PASS");MPI_Finalize();return 0;
    } catch(const std::exception& e) {
        std::cerr<<"RUNTIME_COMMUNICATION_FAILURE "<<e.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);return 1;
    }
}
