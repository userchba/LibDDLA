// Linux/ELF integration probe: interpose the actual LAPACK leaf entries and
// forward to the linked OpenBLAS. No production test hook or thread state.
#include <ddla/ddla.h>
#include "host_parallel.h"
#include "row_share.h"
#include "host_threads.h"
#include "Eigensolver/pstedc_impl.h"
#include <mpi.h>
#include <dlfcn.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#if defined(__linux__)
#include <sched.h>
#endif

extern "C" {
int omp_in_parallel(void) __attribute__((weak));
int omp_get_num_threads(void) __attribute__((weak));
void omp_set_dynamic(int) __attribute__((weak));
}

namespace {
template<class R> using Stedc = void (*)(const char*,const int*,R*,R*,R*,const int*,
                                        R*,const int*,int*,const int*,int*);
Stedc<float> real_sstedc=nullptr;
Stedc<double> real_dstedc=nullptr;
struct InjectedLeafFailure {};
struct Probe {
    std::atomic<bool> active{false},inject{false};
    std::atomic<int> calls{0},parallel_calls{0},blas_min{0},blas_max{0},omp_min{0},omp_max{0},outside{0};
#if defined(__linux__)
    cpu_set_t allowed{};
#endif
} probe;
void minimum(std::atomic<int>& x,int v) { int a=x.load();while(v<a && !x.compare_exchange_weak(a,v)) {} }
void maximum(std::atomic<int>& x,int v) { int a=x.load();while(v>a && !x.compare_exchange_weak(a,v)) {} }

void observe_leaf(const char* compz,const int* lwork) {
    if(!probe.active.load(std::memory_order_acquire) || *compz!='I' || *lwork<0) return;
    const int b=openblas_get_num_threads();
    const int t=omp_get_num_threads();
    probe.calls.fetch_add(1);
    probe.parallel_calls.fetch_add(omp_in_parallel()!=0);
    minimum(probe.blas_min,b);maximum(probe.blas_max,b);
    minimum(probe.omp_min,t);maximum(probe.omp_max,t);
#if defined(__linux__)
    cpu_set_t current;
    if(sched_getaffinity(0,sizeof(current),&current)!=0) probe.outside.fetch_add(1);
    else for(int c=0;c<CPU_SETSIZE;++c)
        if(CPU_ISSET(c,&current) && !CPU_ISSET(c,&probe.allowed)) {probe.outside.fetch_add(1);break;}
#endif
    // Throw before entering Fortran: the production C++ parallel helper must
    // join the other leaves and unwind its BLAS-thread guard.
    if(probe.inject.exchange(false)) throw InjectedLeafFailure{};
}
void require(bool ok,const char* message) {
    int pass=ok?1:0,all=0;
    MPI_Allreduce(&pass,&all,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    if(!all) throw std::runtime_error(message);
}
struct EnvGuard {
    std::string key,old;bool had;
    EnvGuard(const char* k,const char* v):key(k),had(std::getenv(k)!=nullptr) {
        if(had) old=std::getenv(k);
        if(setenv(k,v,1)!=0) throw std::runtime_error("setenv failed");
    }
    ~EnvGuard() {if(had)setenv(key.c_str(),old.c_str(),1);else unsetenv(key.c_str());}
};
std::string library_path(void* symbol) {
    Dl_info info{};
    if(!symbol || !dladdr(symbol,&info) || !info.dli_fname) return {};
    char* resolved=realpath(info.dli_fname,nullptr);
    if(!resolved) return {};
    std::string path(resolved);std::free(resolved);return path;
}
struct ExternalSymbol {
    void* address;
    std::string library;
};
ExternalSymbol external_symbol(const char* name,int rank) {
    // A non-PIE executable can export an undefined function with a nonzero
    // canonical PLT address. RTLD_DEFAULT can return that executable address,
    // even though calls are correctly relocated to the vendor library.
    // Skip the executable when identifying the actual external implementation.
    void* next=dlsym(RTLD_NEXT,name);
    void* global=dlsym(RTLD_DEFAULT,name);
    const auto next_path=library_path(next),global_path=library_path(global);
    std::printf("PSTEDC_THREAD_SYMBOL rank=%d name=%s next_address=%p next_path=%s "
                "default_address=%p default_path=%s\n",rank,name,next,
                next_path.empty()?"missing":next_path.c_str(),global,
                global_path.empty()?"missing":global_path.c_str());
    return {next,next_path};
}
int skip(int rank,const char* reason) {
    if(rank==0) std::printf("PSTEDC_THREAD_CONTROL available=0 backend=unsupported\n"
                           "PSTEDC_THREAD_TEST status=SKIP reason=%s\n",reason);
    return 77;
}

template<class R>
void run_case(const ddla::DdlaHandle_t& h,bool public_api,const char* kind,int threads,int blas) {
    int rank=0,P=1;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&P);
    const bool single=std::string(kind)=="singleleaf",unwind=std::string(kind)=="unwind";
    const int n=single?31:32*P*std::max(threads,2)+1;
    EnvGuard leaf("DDLA_EIG_DC_LEAF","64");
    // A single-rank CPU solve calls LAPACK once with the handle's BLAS
    // budget. Only the distributed path splits leaves across OpenMP workers.
    const bool local=P==1;
    int leaves=1,max_leaf=n;
    // A nontrivial multi-rank solve never uses a full-size replicated leaf.
    // The small case has one local leaf per rank on the two-rank test grid.
    const int leaf_limit=P>1&&n>1?std::min(64,n/2+n%2):64;
    while(!local && max_leaf>leaf_limit) {max_leaf=(max_leaf+1)/2;leaves*=2;}
    const int local_leaves=rank<leaves?(leaves-1-rank)/P+1:0;
    const bool parallel=!local && threads>1 && local_leaves>1;
    const int expected_calls=unwind && !parallel?std::min(local_leaves,1):local_leaves;
    const int expected_blas=local?threads:(parallel?1:blas);
    const ddla::detail::RowShare rows(n,P);
    const int nr=rows.count(rank),ld=std::max(nr,1)+3;
    const R sentinel=R(12345);
    std::vector<R> d(n),e(n,R(0)),w(n),Y(static_cast<std::size_t>(ld)*n+16,sentinel);
    for(int i=0;i<n;++i) {
        d[i]=R(2)+R((17*i)%23-11)/R(32);
        if(i+1<n)e[i]=R(0.13)+R((13*i)%17)/R(128);
    }
    const auto original_d=d,original_e=e;
    double norm=0;
    for(int i=0;i<n;++i)norm=std::max(norm,double(std::abs(d[i]))+double(std::abs(e[i]))+
                                             (i?double(std::abs(e[i-1])):0));
#if defined(__linux__)
    cpu_set_t before,after;
    require(sched_getaffinity(0,sizeof(before),&before)==0,"cannot read initial affinity");
    probe.allowed=before;
#endif
    require(openblas_get_num_threads()==blas,"BLAS count incorrect before solve");
    probe.calls=0;probe.parallel_calls=0;probe.outside=0;
    probe.blas_min=std::numeric_limits<int>::max();probe.blas_max=0;
    probe.omp_min=std::numeric_limits<int>::max();probe.omp_max=0;
    probe.inject=unwind;probe.active.store(true,std::memory_order_release);
    bool caught=false;
    try {
        if(public_api) {
            int info=0;ddla::pstedc<R>(h,'V',n,d.data(),e.data(),w.data(),Y.data(),ld,info);
            if(info!=0)throw std::runtime_error("pstedc returned nonzero info");
        } else ddla::pstedc(h,n,d.data(),e.data(),w.data(),Y.data(),ld);
    } catch(const InjectedLeafFailure&) {caught=true;}
    probe.active.store(false,std::memory_order_release);
    const int calls=probe.calls.load(),bmin=calls?probe.blas_min.load():0,bmax=probe.blas_max.load();
    const int omin=calls?probe.omp_min.load():0,omax=probe.omp_max.load();
    const int restored=openblas_get_num_threads();
    int affinity_before=0,affinity_after=0,affinity_equal=0;
#if defined(__linux__)
    require(sched_getaffinity(0,sizeof(after),&after)==0,"cannot read final affinity");
    affinity_before=CPU_COUNT(&before);affinity_after=CPU_COUNT(&after);affinity_equal=CPU_EQUAL(&before,&after);
#endif
    require(caught==unwind,"unexpected leaf exception outcome");
    if(calls!=expected_calls) {
        std::fprintf(stderr,"PSTEDC_THREAD_COVERAGE path=%s type=%s case=%s rank=%d ranks=%d "
                            "expected=%d observed=%d\n",public_api?"public":"internal",
                     std::is_same<R,float>::value?"s":"d",kind,rank,P,expected_calls,calls);
    }
    require(calls==expected_calls,"actual STEDC leaf interposition/coverage failed");
    require(!calls || (bmin==expected_blas && bmax==expected_blas),"wrong BLAS budget inside STEDC");
    require(!calls || (omin==(parallel?threads:1) && omax==(parallel?threads:1)),"wrong OpenMP leaf team");
    require(probe.parallel_calls.load()==(parallel?calls:0),"wrong parallel leaf coverage");
    require(restored==blas,"BLAS budget not restored after solve/unwind");
    require(affinity_equal && probe.outside.load()==0,"affinity changed or escaped allocation");
    double residual=0,orthogonal=0,padding_errors=0;
    if(!unwind) {
        bool finite=true;
        for(int j=0;j<n;++j) {
            finite=finite && std::isfinite(double(w[j])) && (!j || w[j]>=w[j-1]);
            for(int i=0;i<nr;++i)finite=finite && std::isfinite(double(Y[i+static_cast<std::size_t>(j)*ld]));
            for(int i=nr;i<ld;++i)padding_errors+=Y[i+static_cast<std::size_t>(j)*ld]!=sentinel;
        }
        for(std::size_t i=static_cast<std::size_t>(ld)*n;i<Y.size();++i)padding_errors+=Y[i]!=sentinel;
        require(finite && padding_errors==0,"nonfinite result or output padding changed");
        std::vector<double> full(static_cast<std::size_t>(n)*n,0);
        for(int j=0;j<n;++j)for(int i=0;i<nr;++i)
            full[(rows.lo(rank)+i)+static_cast<std::size_t>(j)*n]=Y[i+static_cast<std::size_t>(j)*ld];
        MPI_Allreduce(MPI_IN_PLACE,full.data(),n*n,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        if(rank==0) {
            for(int j=0;j<n;++j)for(int i=0;i<n;++i) {
                const auto at=[&](int r,int c){return full[r+static_cast<std::size_t>(c)*n];};
                double value=double(original_d[i])*at(i,j)-double(w[j])*at(i,j);
                if(i)value+=double(original_e[i-1])*at(i-1,j);
                if(i+1<n)value+=double(original_e[i])*at(i+1,j);
                residual=std::max(residual,std::abs(value));
                long double dot=0;
                for(int r=0;r<n;++r)dot+=static_cast<long double>(at(r,i))*at(r,j);
                orthogonal=std::max(orthogonal,double(std::abs(dot-(i==j?1.L:0.L))));
            }
            residual/=n*std::numeric_limits<R>::epsilon()*norm;
            orthogonal/=n*std::numeric_limits<R>::epsilon();
        }
        MPI_Bcast(&residual,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Bcast(&orthogonal,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        require(std::isfinite(residual) && std::isfinite(orthogonal) && residual<=50 && orthogonal<=50,
                "independent residual/orthogonality failed");
    }
    std::printf("PSTEDC_THREAD_CASE path=%s type=%s case=%s rank=%d ranks=%d omp_requested=%d blas_requested=%d "
                "expected_leaf_calls=%d expected_leaf_blas=%d leaf_calls=%d observed_leaf_blas_min=%d observed_leaf_blas_max=%d "
                "omp_leaf_calls=%d observed_omp_min=%d observed_omp_max=%d restored_blas=%d caught_exception=%d "
                "affinity_before=%d affinity_after=%d affinity_equal=%d leaf_affinity_outside=%d "
                "residual=%.9g orthogonality=%.9g padding_errors=%.0f status=PASS\n",
                public_api?"public":"borrowed",std::is_same<R,float>::value?"s":"d",kind,rank,P,threads,blas,
                expected_calls,expected_blas,calls,bmin,bmax,probe.parallel_calls.load(),omin,omax,restored,int(caught),
                affinity_before,affinity_after,affinity_equal,probe.outside.load(),residual,orthogonal,padding_errors);
    std::fflush(stdout);
}
} // namespace

extern "C" void sstedc_(const char* c,const int* n,float* d,float* e,float* z,const int* ldz,
                         float* work,const int* lwork,int* iwork,const int* liwork,int* info) {
    observe_leaf(c,lwork);real_sstedc(c,n,d,e,z,ldz,work,lwork,iwork,liwork,info);
}
extern "C" void dstedc_(const char* c,const int* n,double* d,double* e,double* z,const int* ldz,
                         double* work,const int* lwork,int* iwork,const int* liwork,int* info) {
    observe_leaf(c,lwork);real_dstedc(c,n,d,e,z,ldz,work,lwork,iwork,liwork,info);
}

int main(int argc,char** argv) {
    int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int rank=0,P=1;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&P);
    int result=0;
    try {
        int threads=4,blas=4;
        for(int i=1;i<argc;++i) {
            const std::string option(argv[i]);
            if((option=="--threads" || option=="--blas-threads") && i+1<argc) {
                const int value=std::stoi(argv[++i]);
                if(option=="--threads")threads=value;else blas=value;
            } else throw std::runtime_error("usage: test_pstedc_threads [--threads 1|4|8] [--blas-threads 4|8]");
        }
        require((threads==1 || threads==4 || threads==8) && (blas==4 || blas==8),"invalid test thread budget");
        require(provided>=MPI_THREAD_FUNNELED,"MPI_THREAD_FUNNELED unavailable");
        const auto ss=external_symbol("sstedc_",rank),ds=external_symbol("dstedc_",rank);
        const auto sg=external_symbol("sgemm_",rank),dg=external_symbol("dgemm_",rank);
        const auto bg=external_symbol("openblas_get_num_threads",rank),bs=external_symbol("openblas_set_num_threads",rank);
        const auto oi=external_symbol("omp_in_parallel",rank),on=external_symbol("omp_get_num_threads",rank);
        const auto od=external_symbol("omp_set_dynamic",rank);
        real_sstedc=reinterpret_cast<Stedc<float>>(ss.address);
        real_dstedc=reinterpret_cast<Stedc<double>>(ds.address);
        const auto& slib=ss.library;
        const auto& dlib=ds.library;
        const bool weak_control=openblas_get_num_threads && openblas_set_num_threads;
        const bool weak_omp=omp_in_parallel && omp_get_num_threads && omp_set_dynamic;
        const bool same_blas=ss.address && ds.address && sg.address && dg.address && bg.address && bs.address &&
                             !slib.empty() && slib==dlib && slib==sg.library && slib==dg.library &&
                             slib==bg.library && slib==bs.library;
        const bool same_omp=oi.address && on.address && od.address && !oi.library.empty() &&
                            oi.library==on.library && oi.library==od.library;
        int supported=weak_control && weak_omp && same_blas && same_omp;
#if !defined(__linux__)
        supported=0;
#endif
        std::printf("PSTEDC_THREAD_CAPABILITY rank=%d weak_control=%d weak_omp=%d same_blas=%d same_omp=%d supported=%d\n",
                    rank,int(weak_control),int(weak_omp),int(same_blas),int(same_omp),supported);
        std::fflush(stdout);
        MPI_Allreduce(MPI_IN_PLACE,&supported,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
        if(!supported || P>2)result=skip(rank,!supported?"requires_controlled_OpenBLAS_LAPACK_and_host_OpenMP":"use_one_or_two_MPI_ranks");
        else {
            if(rank==0) {
                std::printf("PSTEDC_THREAD_CONTROL available=1 backend=openblas\n"
                            "PSTEDC_THREAD_LIBRARY type=s path=%s\nPSTEDC_THREAD_LIBRARY type=d path=%s\n",slib.c_str(),dlib.c_str());
                std::fflush(stdout);
            }
            const auto nt=std::to_string(threads);
            EnvGuard omp("OMP_NUM_THREADS",nt.c_str()),eig("DDLA_EIG_THREADS",nt.c_str());
            omp_set_dynamic(0);
            ddla::detail::ScopedBlasThreads requested(blas);
            require(openblas_get_num_threads()==blas,"OpenBLAS cannot supply requested thread count");
            std::atomic<int> actual_team{0};
            ddla::detail::host_parallel_for(2*threads,1,threads,[&](long long,long long){maximum(actual_team,omp_get_num_threads());});
            int team_ok=actual_team.load()==threads;
            MPI_Allreduce(MPI_IN_PLACE,&team_ok,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
            if(!team_ok)result=skip(rank,"requested_DDLA_OpenMP_team_unavailable");
            else {
                ddla::DdlaHandle_t h=nullptr;
                require(ddla::ddlaInit(h,ddla::DdlaBackend::CPU)==ddla::ddlaStatus_t::DDLA_STATUS_SUCCESS,"CPU handle unavailable");
                require(ddla::ddlaSet(h,MPI_COMM_WORLD,P,1)==ddla::ddlaStatus_t::DDLA_STATUS_SUCCESS,"grid setup failed");
                for(bool public_api:{false,true})for(const char* kind:{"parallel","singleleaf","unwind"}) {
                    run_case<float>(h,public_api,kind,threads,blas);
                    run_case<double>(h,public_api,kind,threads,blas);
                }
                ddla::ddlaDestroy(h);
                std::printf("PSTEDC_THREAD_TEST rank=%d ranks=%d omp_requested=%d blas_requested=%d cores_per_rank=%d "
                            "slurm_job_id=%s status=PASS\n",rank,P,threads,blas,ddla::detail::host_allowed_cpus(),
                            std::getenv("SLURM_JOB_ID")?std::getenv("SLURM_JOB_ID"):"unset");
                std::fflush(stdout);
            }
        }
    } catch(const std::exception& e) {
        std::fprintf(stderr,"test_pstedc_threads rank=%d: %s\n",rank,e.what());MPI_Abort(MPI_COMM_WORLD,1);
    } catch(...) {std::fprintf(stderr,"test_pstedc_threads: unexpected exception\n");MPI_Abort(MPI_COMM_WORLD,1);}
    MPI_Finalize();return result;
}
