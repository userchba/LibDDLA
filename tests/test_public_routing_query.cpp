#include <ddla/ddla.h>
#include "Eigensolver/two_stage/pheevd_impl.h"
#include "ddla_stream.h"
#include "dc_threads.h"
#include <mpi.h>
#include <algorithm>
#include <vector>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#if defined(__linux__)
#include <sched.h>
#endif

namespace {
void require_all(bool condition,const char* message) {
    const int local=condition?1:0;int global=0;
    MPI_Allreduce(&local,&global,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    if(!global)throw std::runtime_error(message);
}
int allowed_cpus() {
#if defined(__linux__) && defined(CPU_COUNT)
    cpu_set_t mask;CPU_ZERO(&mask);
    if(sched_getaffinity(0,sizeof(mask),&mask)==0&&CPU_COUNT(&mask)>0)return CPU_COUNT(&mask);
#endif
    return std::max(1,int(std::thread::hardware_concurrency()));
}
// Metadata checks borrow non-null scalar addresses: no matrix values are
// inspected or solve resources allocated during pheevd_check.
ddla::spectral_impl::PheevdPlan checked_plan(ddla::DdlaHandle_t h,int n,int nb,int band) {
    const std::string width=std::to_string(band);
    setenv("DDLA_EIG_NBW",width.c_str(),1);
    int d[ddla::DDLA_DLEN_];ddla::ddlaDescInit(d,h,n,n,nb,nb,0,0);
    double a=0,w=0;
    auto topology=ddla::spectral_impl::phb2st_topology_prepare(h);
    auto plan=ddla::spectral_impl::pheevd_check<double>(h,ddla::DdlaBackend::CPU,'N','U',n,&a,d,&w,nullptr,nullptr,*topology,'A',nullptr,ddla::DdlaBackend::CPU,n);
    topology->release();
    return plan;
}
void warm_handle(MPI_Comm comm,int expected) {
    ddla::DdlaHandle_t previous=nullptr;
    ddla::ddlaInit(previous,ddla::DdlaBackend::CPU);ddla::ddlaSet(previous,comm,'R');
    if(previous->host_threads!=expected)throw std::runtime_error("prior handle budget");
    ddla::ddlaDestroy(previous);
}
void check_workers(ddla::DdlaHandle_t h,int cpus) {
    bool okay=true;
    if(h->host_threads<=16) {
        ddla::detail::DcHostTeam team(h->host_threads);
        std::vector<std::thread::id> identities(h->host_threads);
        std::vector<int> masks(h->host_threads,0);
#if defined(__linux__) && defined(CPU_COUNT)
        cpu_set_t caller;CPU_ZERO(&caller);
        if(sched_getaffinity(0,sizeof(caller),&caller)!=0)throw std::runtime_error("read worker control affinity");
#endif
        team.run_roles(h->host_threads,[&](int role){
            identities[role]=std::this_thread::get_id();
#if defined(__linux__) && defined(CPU_COUNT)
            cpu_set_t current;CPU_ZERO(&current);
            masks[role]=sched_getaffinity(0,sizeof(current),&current)==0&&CPU_EQUAL(&caller,&current);
#else
            masks[role]=allowed_cpus()==cpus;
#endif
        });
        okay=team.size()==h->host_threads;
        for(int role=0;role<h->host_threads;++role) {
            okay=okay&&identities[role]!=std::thread::id{}&&masks[role];
            for(int previous=0;previous<role;++previous)okay=okay&&identities[role]!=identities[previous];
        }
    }
    (void)cpus;
    require_all(okay,"distinct configured host workers and exact inherited affinity");
}
int affinity_workers(MPI_Comm node) {
#if defined(__linux__) && defined(CPU_COUNT)
    cpu_set_t mask;CPU_ZERO(&mask);
    if(sched_getaffinity(0,sizeof(mask),&mask)!=0)throw std::runtime_error("read affinity");
    int mine[CPU_SETSIZE]{},sharing[CPU_SETSIZE]{};
    for(int c=0;c<CPU_SETSIZE;++c)mine[c]=CPU_ISSET(c,&mask)?1:0;
    MPI_Allreduce(mine,sharing,CPU_SETSIZE,MPI_INT,MPI_SUM,node);
    double share=0;
    for(int c=0;c<CPU_SETSIZE;++c)if(mine[c])share+=1.0/sharing[c];
    return std::max(1,std::min(512,int(share+1e-9)));
#else
    int ranks=1;MPI_Comm_size(node,&ranks);
    return std::max(1,std::min(512,allowed_cpus()/ranks));
#endif
}
struct AffinityRestore {
#if defined(__linux__) && defined(CPU_COUNT)
    cpu_set_t original{};bool changed=false;
    AffinityRestore(){if(sched_getaffinity(0,sizeof(original),&original)!=0)throw std::runtime_error("save affinity");}
    ~AffinityRestore(){if(changed)sched_setaffinity(0,sizeof(original),&original);}
    int bind(MPI_Comm node,const std::string& kind,int rank,int ranks) {
        // Peers' union stays within the scheduler allocation, even when
        // the launcher initially gives each rank a disjoint binding.
        int mine[CPU_SETSIZE]{},all[CPU_SETSIZE]{};
        for(int c=0;c<CPU_SETSIZE;++c)mine[c]=CPU_ISSET(c,&original)?1:0;
        MPI_Allreduce(mine,all,CPU_SETSIZE,MPI_INT,MPI_MAX,node);
        std::vector<int> cpus;
        for(int c=0;c<CPU_SETSIZE;++c)if(all[c])cpus.push_back(c);
        cpu_set_t next;CPU_ZERO(&next);int expected=1;
        if(kind=="affinity_shared") {
            const int count=std::min(int(cpus.size()),10*ranks);
            for(int i=0;i<count;++i)CPU_SET(cpus[i],&next);
            expected=std::max(1,count/ranks);
        } else if(kind=="affinity_disjoint") {
            const int count=std::min(10,int(cpus.size())/ranks);
            if(count<2)throw std::runtime_error("disjoint-mask test needs at least two CPUs per rank");
            for(int i=0;i<count;++i)CPU_SET(cpus[rank*count+i],&next);
            expected=count;
        } else {
            if(int(cpus.size())<3*ranks)throw std::runtime_error("partial-mask test needs three CPUs per rank");
            for(int i=0;i<ranks;++i)CPU_SET(cpus[i],&next);
            CPU_SET(cpus[ranks+2*rank],&next);CPU_SET(cpus[ranks+2*rank+1],&next);
            expected=3;
        }
        if(sched_setaffinity(0,sizeof(next),&next)!=0)throw std::runtime_error("set controlled affinity");
        changed=true;
        cpu_set_t actual;CPU_ZERO(&actual);
        if(sched_getaffinity(0,sizeof(actual),&actual)!=0||!CPU_EQUAL(&actual,&next))throw std::runtime_error("controlled affinity not applied");
        return expected;
    }
#else
    int bind(MPI_Comm,const std::string&,int,int){throw std::runtime_error("controlled affinity needs Linux");}
#endif
};
void shape_checks() {
    struct Shape{int n,nb,bw,pr,pc;bool expected;};
    const Shape cases[]={{2047,64,64,2,4,false},{2048,64,64,1,1,true},
        {4096,64,64,1,8,true},{4096,64,64,8,1,true},{4096,64,64,6,8,true},
        {4096,64,64,8,8,true},{4096,64,64,8,9,false},
        {4096,128,64,2,4,false},{4096,64,32,2,4,false},{4096,0,64,2,4,false},
        {0,64,64,2,4,false},{-1,64,64,2,4,false},{INT_MAX,64,64,2,4,true},
        {4096,64,64,2,0,false}};
    bool good=true;
    for(const auto& c:cases)good=good&&ddla::detail::pheevd_cpu_householder_shape(c.n,c.nb,c.bw,c.pr,c.pc)==c.expected;
    for(int workers:{1,2,6,12,24,48,64})
        good=good&&ddla::detail::pheevd_cpu_householder_shape(5000,64,64,2,2,workers);
    for(int workers:{0,65})
        good=good&&!ddla::detail::pheevd_cpu_householder_shape(5000,64,64,2,2,workers);
    require_all(good,"public CPU shape boundaries");
}
}

int main(int argc,char** argv) {
    int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int rank=0,ranks=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    ddla::DdlaHandle_t h=nullptr;
    try {
        int pr=2,pc=4,n=4096,nb=64,nbw=64,required_nodes=1;std::string mode="cold";
        for(int i=1;i<argc;++i) {
            require_all(i+1<argc,"query option needs argument");const std::string key=argv[i],v=argv[++i];
            if(key=="--grid") {const auto x=v.find('x');if(x==std::string::npos)throw std::invalid_argument("grid RxC");pr=std::stoi(v.substr(0,x));pc=std::stoi(v.substr(x+1));}
            else if(key=="--mode")mode=v;
            else if(key=="--n")n=std::stoi(v);
            else if(key=="--nb")nb=std::stoi(v);
            else if(key=="--nbw")nbw=std::stoi(v);
            else if(key=="--require-nodes")required_nodes=std::stoi(v);
            else throw std::invalid_argument("unknown query option");
        }
        require_all(provided>=MPI_THREAD_FUNNELED&&pr>0&&pc>0&&static_cast<long long>(pr)*pc==ranks,"query geometry/thread contract");
        ::setenv("OMP_NUM_THREADS","1",1);::setenv("DDLA_EIG_THREADS","1",1);
        ::unsetenv("DDLA_EIG_CPU_ACCUM");::unsetenv("DDLA_EIG_CHASE_TEAM_THREADS");
        shape_checks();
        MPI_Comm node=MPI_COMM_NULL;MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node);
        int local=0,node_rank=0;MPI_Comm_size(node,&local);MPI_Comm_rank(node,&node_rank);
        const int leader=node_rank==0?1:0;int nodes=0;MPI_Allreduce(&leader,&nodes,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
        require_all(nodes==required_nodes,"query actual node topology differs");
        int expected_workers=1,warmed=0,legacy=0;
        AffinityRestore affinity;
        bool known=true;
        if(mode=="warm") {warm_handle(MPI_COMM_WORLD,1);warmed=1;}
        else if(mode=="half") {if(rank%2==0){warm_handle(MPI_COMM_SELF,1);warmed=1;}}
        else if(mode=="divergent") {
            const int previous=rank%2?2:1;setenv("DDLA_EIG_THREADS",previous==2?"2":"1",1);
            warm_handle(MPI_COMM_SELF,previous);warmed=1;setenv("DDLA_EIG_THREADS","1",1);
        } else if(mode=="budget_divergent") {
            expected_workers=rank==0?65:1;setenv("DDLA_EIG_THREADS",rank==0?"65":"1",1);
        } else if(mode=="thread2") {
            setenv("OMP_NUM_THREADS","3",1);setenv("DDLA_EIG_THREADS","2",1);expected_workers=2;
        } else if(mode=="omp_only") {
            setenv("OMP_NUM_THREADS","3",1);unsetenv("DDLA_EIG_THREADS");expected_workers=3;
        } else if(mode=="affinity"||mode=="affinity_half"||mode=="affinity_shared"||
                  mode=="affinity_disjoint"||mode=="affinity_partial") {
            unsetenv("OMP_NUM_THREADS");unsetenv("DDLA_EIG_THREADS");
            if(mode=="affinity_shared"||mode=="affinity_disjoint"||mode=="affinity_partial")
                expected_workers=affinity.bind(node,mode,node_rank,local);
            else expected_workers=affinity_workers(node);
            if(mode=="affinity_half"&&rank%2==0){warm_handle(MPI_COMM_SELF,std::min(512,allowed_cpus()));warmed=1;}
        } else if(mode.compare(0,6,"accum_")==0) {
            const std::string value=mode.substr(6);setenv("DDLA_EIG_CPU_ACCUM",value.c_str(),1);
            for(const char* control:{"1","on","force","0","off","replay"})legacy=legacy||value==control;
        } else if(mode=="team"||mode=="team_invalid"||mode=="team_empty") {
            const char* value=mode=="team"?"1":mode=="team_invalid"?"invalid_nonempty":"";
            setenv("DDLA_EIG_CHASE_TEAM_THREADS",value,1);legacy=*value!='\0';
        } else if(mode=="rank0_accum"||mode=="rank0_team") {
            if(rank==0){setenv(mode=="rank0_accum"?"DDLA_EIG_CPU_ACCUM":"DDLA_EIG_CHASE_TEAM_THREADS","off",1);legacy=1;}
        } else if(mode!="cold")known=false;
        require_all(known,"unknown query mode");
        DDLA_CHECK(ddla::ddlaInit(h,ddla::DdlaBackend::CPU));DDLA_CHECK(ddla::ddlaSet(h,MPI_COMM_WORLD,pr,pc,'R'));
        const int cpus=allowed_cpus();
        require_all(h->host_threads==expected_workers&&h->node_ranks==local,"handle affinity/override budget and cached node ranks");
        check_workers(h,cpus);
        const int local_expected=nodes==1&&!legacy&&ddla::detail::pheevd_cpu_householder_shape(n,nb,nbw,pr,pc,expected_workers)?1:0;
        int expected=0;MPI_Allreduce(&local_expected,&expected,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
        const auto first=checked_plan(h,n,nb,nbw);
        // A new subset handle reads current overrides independently. The
        // original handle and subsequent plans retain their configured budget.
        setenv("OMP_NUM_THREADS","4",1);setenv("DDLA_EIG_THREADS","3",1);
        const char* accum_env=std::getenv("DDLA_EIG_CPU_ACCUM");
        const char* team_env=std::getenv("DDLA_EIG_CHASE_TEAM_THREADS");
        const bool had_accum=accum_env!=nullptr,had_team=team_env!=nullptr;
        const std::string saved_accum=accum_env?accum_env:"",saved_team=team_env?team_env:"";
        unsetenv("DDLA_EIG_CPU_ACCUM");unsetenv("DDLA_EIG_CHASE_TEAM_THREADS");
        MPI_Comm subset=MPI_COMM_NULL;MPI_Comm_split(MPI_COMM_WORLD,rank%2,rank,&subset);
        ddla::DdlaHandle_t other=nullptr;ddla::ddlaInit(other,ddla::DdlaBackend::CPU);ddla::ddlaSet(other,subset,'R');
        const auto subset_plan=checked_plan(other,65,nb,nbw);
        MPI_Comm subset_node=MPI_COMM_NULL;MPI_Comm_split_type(subset,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&subset_node);
        int subset_local=0;MPI_Comm_size(subset_node,&subset_local);MPI_Comm_free(&subset_node);
        require_all(other->host_threads==3&&subset_plan.workers==3&&other->node_ranks==subset_local,
            "subset handle independent budget and cached node ranks");
        ddla::ddlaDestroy(other);MPI_Comm_free(&subset);
        if(had_accum)setenv("DDLA_EIG_CPU_ACCUM",saved_accum.c_str(),1);
        if(had_team)setenv("DDLA_EIG_CHASE_TEAM_THREADS",saved_team.c_str(),1);
        const auto repeat=checked_plan(h,n,nb,nbw);
        const bool selected=first.automatic_householder;
        const auto engine=expected?ddla::spectral_impl::PheevdEngine::Householder:ddla::spectral_impl::PheevdEngine::ReplayCpu;
        require_all(selected==bool(expected)&&repeat.automatic_householder==selected&&
            first.engine==engine&&repeat.engine==engine&&first.workers==expected_workers&&
            repeat.workers==expected_workers&&h->host_threads==expected_workers,
            "production checked engine, automatic policy and frozen handle budget");
        const int choice=selected?1:0;int lo=0,hi=0;MPI_Allreduce(&choice,&lo,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);MPI_Allreduce(&choice,&hi,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
        require_all(lo==hi,"production selection was not uniform");
        std::printf("PUBLIC_ROUTING_QUERY_RANK rank=%d mode=%s nodes=%d local=%d allowed=%d warmed=%d workers=%d selected=%d expected=%d status=PASS\n",rank,mode.c_str(),nodes,local,cpus,warmed,first.workers,choice,expected);
        DDLA_CHECK(ddla::ddlaSet(h,MPI_COMM_SELF,1,1));
        require_all(h->node_ranks==1&&h->host_threads==3,"reconfigured self topology and budget");
        DDLA_CHECK(ddla::ddlaSet(h,MPI_COMM_WORLD,pr,pc));
        require_all(h->node_ranks==local&&h->host_threads==3,"reconfigured world topology and budget");
        DDLA_CHECK(ddla::ddlaDestroy(h));MPI_Comm_free(&node);
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank==0)std::printf("PUBLIC_ROUTING_QUERY mode=%s n=%d nb=%d nbw=%d grid=%dx%d nodes=%d selected=%d ranks=%d shape_checks=23 status=PASS\n",mode.c_str(),n,nb,nbw,pr,pc,nodes,expected,ranks);
        MPI_Finalize();return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"rank %d: %s\n",rank,error.what());MPI_Abort(MPI_COMM_WORLD,1);}
    return 1;
}
