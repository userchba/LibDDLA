#include <ddla/ddla.h>
#include "Eigensolver/two_stage/pheevd_impl.h"
#include "Eigensolver/eig_cpu_kernels.h"
#include <mpi.h>
#include <algorithm>
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
int affinity_workers(int cpus,int ranks) {
#ifdef _OPENMP
    return std::max(1,std::min(512,cpus/std::max(1,ranks)));
#else
    (void)cpus;(void)ranks;return 1;
#endif
}
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
        DDLA_CHECK(ddla::ddlaInit(h,ddla::DdlaBackend::CPU));DDLA_CHECK(ddla::ddlaSet(h,MPI_COMM_WORLD,pr,pc,'R'));
        MPI_Comm node=MPI_COMM_NULL;MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node);
        int local=0,node_rank=0;MPI_Comm_size(node,&local);MPI_Comm_rank(node,&node_rank);MPI_Comm_free(&node);
        const int leader=node_rank==0?1:0;int nodes=0;MPI_Allreduce(&leader,&nodes,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
        require_all(nodes==required_nodes,"query actual node topology differs");
        const int cpus=allowed_cpus();int expected_workers=1,warmed=0,legacy=0;
#ifdef _OPENMP
        constexpr int openmp=1;
#else
        constexpr int openmp=0;
#endif
        bool known=true;
        if(mode=="warm") {expected_workers=ddla::detail::host_omp_threads(MPI_COMM_WORLD);warmed=1;require_all(expected_workers==1,"all-warm cache count");}
        else if(mode=="half") {if(rank%2==0){expected_workers=ddla::detail::host_omp_threads(MPI_COMM_SELF);warmed=1;}require_all(expected_workers==1,"half-warm SELF cache count");}
        else if(mode=="divergent") {
            expected_workers=rank%2?2:1;::setenv("DDLA_EIG_THREADS",expected_workers==2?"2":"1",1);
            const int got=ddla::detail::host_omp_threads(MPI_COMM_SELF);warmed=1;
            ::setenv("DDLA_EIG_THREADS","1",1);require_all(got==expected_workers,"divergent initial cache counts");
        } else if(mode=="thread2") {::setenv("OMP_NUM_THREADS","2",1);::setenv("DDLA_EIG_THREADS","2",1);expected_workers=2;}
        else if(mode=="affinity"||mode=="affinity_half") {
            ::unsetenv("OMP_NUM_THREADS");::unsetenv("DDLA_EIG_THREADS");expected_workers=affinity_workers(cpus,local);
            if(mode=="affinity_half"&&rank%2==0){expected_workers=affinity_workers(cpus,1);const int got=ddla::detail::host_omp_threads(MPI_COMM_SELF);warmed=1;if(got!=expected_workers)known=false;}
        } else if(mode.compare(0,6,"accum_")==0) {
            const std::string value=mode.substr(6);::setenv("DDLA_EIG_CPU_ACCUM",value.c_str(),1);
            for(const char* control:{"1","on","force","0","off","replay"})legacy=legacy||value==control;
        } else if(mode=="team"||mode=="team_invalid"||mode=="team_empty") {
            const char* value=mode=="team"?"1":mode=="team_invalid"?"invalid_nonempty":"";
            ::setenv("DDLA_EIG_CHASE_TEAM_THREADS",value,1);legacy=*value!='\0';
        } else if(mode=="rank0_accum"||mode=="rank0_team") {
            if(rank==0){::setenv(mode=="rank0_accum"?"DDLA_EIG_CPU_ACCUM":"DDLA_EIG_CHASE_TEAM_THREADS","off",1);legacy=1;}
        } else if(mode!="cold")known=false;
        require_all(known,"unknown mode or wrong affinity cache count");
        const int local_expected=nodes==1&&!legacy&&ddla::detail::pheevd_cpu_householder_shape(n,nb,nbw,pr,pc,expected_workers)?1:0;
        int expected=0;MPI_Allreduce(&local_expected,&expected,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
        const bool first=ddla::detail::pheevd_cpu_householder_selected(h,n,nb,nbw);
        const bool repeat=ddla::detail::pheevd_cpu_householder_selected(h,n,nb,nbw);
        const int cached=ddla::detail::host_omp_threads_for_node(local);
        const int sticky=ddla::detail::host_omp_threads_for_node(999999);
        require_all(first==bool(expected)&&repeat==first&&cached==expected_workers&&sticky==cached,"public selection/cache uniform result");
        const int choice=first?1:0;int lo=0,hi=0;MPI_Allreduce(&choice,&lo,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);MPI_Allreduce(&choice,&hi,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
        require_all(lo==hi,"public selection was not uniform");
        std::printf("PUBLIC_ROUTING_QUERY_RANK rank=%d mode=%s nodes=%d local=%d allowed=%d openmp=%d warmed=%d cached=%d selected=%d repeat=%d expected=%d status=PASS\n",rank,mode.c_str(),nodes,local,cpus,openmp,warmed,cached,choice,int(repeat),expected);
        DDLA_CHECK(ddla::ddlaDestroy(h));
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank==0)std::printf("PUBLIC_ROUTING_QUERY mode=%s n=%d nb=%d nbw=%d grid=%dx%d nodes=%d selected=%d ranks=%d shape_checks=23 status=PASS\n",mode.c_str(),n,nb,nbw,pr,pc,nodes,expected,ranks);
        MPI_Finalize();return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"rank %d: %s\n",rank,error.what());MPI_Abort(MPI_COMM_WORLD,1);}
    return 1;
}
