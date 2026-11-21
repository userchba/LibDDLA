// Linux/ELF native-entry observation, following test_pstedc_threads.cpp.
// No production hook. Reported BLAS settings are not actual worker counts.
#include "api_grid_test_common.h"
#include "host_threads.h"
#include "Eigensolver/tridiagonal_rows.h"
#include <dlfcn.h>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
using namespace api_grid_test;
namespace {
std::string transport = "default";
template<DdlaBackend B> void select_transport(DdlaHandle_t h) {
    if constexpr (B == DdlaBackend::GPU) {
        if (transport != "default") {
            const auto mode = transport == "ccl" ? DdlaCommunication::CCL :
                transport == "staged" ? DdlaCommunication::HostStagedMPI : DdlaCommunication::MPI;
            DDLA_CHECK(ddlaSetCommunication(h, mode));
            if (ddlaGetCommunication(h) != mode)
                throw std::runtime_error("requested GPU transport was not selected");
        }
    }
}
template<class R> using Dot=R(*)(const int*,const R*,const int*,const R*,const int*);
template<class R> using Lagts=void(*)(const int*,const int*,const R*,const R*,const R*,const R*,const int*,R*,R*,int*);
Dot<float> real_sdot=nullptr; Dot<double> real_ddot=nullptr;
Lagts<float> real_slagts=nullptr; Lagts<double> real_dlagts=nullptr;
struct InjectedDotFailure {};
struct Probe {
    std::atomic<bool> active{false},cpp{false},status{false};
    std::atomic<int> calls{0},current_min{0},current_max{0},native_min{0},native_max{0},status_injections{0};
} probe;
void minimum(std::atomic<int>& value,int x) { int old=value.load(); while(x<old && !value.compare_exchange_weak(old,x)) {} }
void maximum(std::atomic<int>& value,int x) { int old=value.load(); while(x>old && !value.compare_exchange_weak(old,x)) {} }
void observe_dot() {
    if (!probe.active.load(std::memory_order_acquire)) return;
    const int current=detail::ScopedBlasThreads::current(),native=openblas_get_num_threads();
    probe.calls.fetch_add(1);
    minimum(probe.current_min,current); maximum(probe.current_max,current);
    minimum(probe.native_min,native); maximum(probe.native_max,native);
    // Every participating rank has rows and throws at its first matched DOT,
    // before the following core Allreduce. Never inject on only one rank.
    if (probe.cpp.exchange(false)) throw InjectedDotFailure{};
}
bool inject_solve_status(int* info) {
    // Only the target owner calls native LAGTS; the existing next broadcast
    // carries this status to every rank. Wait until a matched DOT occurred so
    // the same case also observes the actual native budget before failure.
    if (probe.active.load(std::memory_order_acquire) && probe.calls.load()>0 && probe.status.exchange(false)) {
        ++probe.status_injections; *info=7; return true;
    }
    return false;
}
std::string library_path(void* symbol) {
    Dl_info info{};
    if (!symbol || !dladdr(symbol,&info) || !info.dli_fname) return {};
    char* resolved=realpath(info.dli_fname,nullptr);
    if (!resolved) return {};
    std::string path(resolved); std::free(resolved); return path;
}
struct Symbol {void* address=nullptr;std::string path;};
Symbol symbol(const char* name,int rank) {
    // RTLD_NEXT avoids a non-PIE executable's canonical PLT address, exactly
    // as the existing native STEDC-entry thread test does.
    void* address=dlsym(RTLD_NEXT,name); const auto path=library_path(address);
    std::printf("SELECTED_THREAD_SYMBOL rank=%d name=%s address=%p path=%s\n",rank,name,address,path.empty()?"missing":path.c_str());
    return {address,path};
}
struct Env {
    std::string key,old; bool had=false;
    Env(const char* name,const char* value):key(name),had(std::getenv(name)!=nullptr) {
        if (had) old=std::getenv(name); setenv(name,value,1);
    }
    ~Env() {if(had)setenv(key.c_str(),old.c_str(),1);else unsetenv(key.c_str());}
};
void require(DdlaHandle_t h,bool okay,const char* text) { require_close(h,text,okay?0:1,0); }

template<DdlaBackend B,class R> void run_case(DdlaHandle_t h,const char* kind,int external) {
    constexpr int n=17,k=3;
    const int nr=detail::RowShare(n,h->nprocs).count(h->myid),ldy=std::max(1,nr);
    const bool unwind=!std::strcmp(kind,"exception"),status_failure=!std::strcmp(kind,"status");
    require(h,nr>0,"thread observer requires every rank nonempty");
    const auto p=spectral_impl::selected_rows_policy<R>(h,'V',n,k,ldy);
    spectral_impl::PstedcVectorPlan unused;
    std::size_t bb=0,hb=0;
    spectral_impl::tridiagonal_rows_workspace_size<R>(h,p,unused,&bb,&hb);
    ddla::vector<char> backend(h,bb); std::vector<char> host(hb);
    DeviceBuffer<R> Y(h,std::size_t(ldy)*k); std::vector<R> initial(Y.count,R(-717)); upload(h,Y.ptr,initial);
    std::vector<R> d(n,R(2)),e(n,R(-1)),w(n,R(-313)); e[n-1]=0;
    probe.calls=0; probe.current_min=probe.native_min=std::numeric_limits<int>::max();
    probe.current_max=probe.native_max=probe.status_injections=0;
    probe.cpp=unwind; probe.status=status_failure;
    require(h,detail::ScopedBlasThreads::current()==external && openblas_get_num_threads()==external,
        "external BLAS setting differs from requested setup");
    probe.active.store(true,std::memory_order_release);
    bool caught_cpp=false,caught_status=false,caught_other=false;
    try {
        spectral_impl::tridiagonal_rows_run<B,R>(h,p,unused,nullptr,d.data(),e.data(),w.data(),Y.ptr,
            backend.data(),bb,host.data(),hb);
    } catch (const InjectedDotFailure&) {caught_cpp=true;}
      catch (const std::runtime_error& error) {
        caught_status=std::string(error.what()).find("code=7")!=std::string::npos &&
            std::string(error.what()).find("native=7")!=std::string::npos;
        caught_other=!caught_status;
      }
    probe.active.store(false,std::memory_order_release);
    const int calls=probe.calls.load(),restored=detail::ScopedBlasThreads::current();
    require(h,!caught_other && caught_cpp==unwind && caught_status==status_failure,"unexpected selected thread-test failure outcome");
    require(h,calls>0,"native DOT entry was not observed");
    require(h,probe.current_min.load()==h->host_threads && probe.current_max.load()==h->host_threads &&
        probe.native_min.load()==h->host_threads && probe.native_max.load()==h->host_threads,
        "native DOT entry did not receive the handle host budget");
    require(h,restored==external && openblas_get_num_threads()==external,"BLAS setting not restored after selected run/unwind");
    int injections=probe.status_injections.load(); MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,&injections,1,MPI_INT,MPI_SUM,h->comm));
    require(h,injections==(status_failure?1:0),"status injection did not use exactly one native owner and collective propagation");
    if (!unwind && !status_failure) {
        const auto local=download(h,Y.ptr,Y.count); bool finite=true;
        for (R x:w) finite=finite && std::isfinite(x);
        for (R x:local) finite=finite && std::isfinite(x);
        require(h,finite,"successful selected output is nonfinite");
    }
    std::printf("SELECTED_THREAD_CASE backend=%d type=%s case=%s rank=%d ranks=%d handle_budget=%d external=%d dot_calls=%d "
        "observed_current_min=%d observed_current_max=%d observed_native_min=%d observed_native_max=%d restored=%d "
        "caught_exception=%d caught_status=%d total_status_injections=%d actual_workers_measured=0 status=PASS\n",
        int(B),std::is_same<R,float>::value?"s":"d",kind,h->myid,h->nprocs,h->host_threads,external,calls,
        probe.current_min.load(),probe.current_max.load(),probe.native_min.load(),probe.native_max.load(),restored,
        int(caught_cpp),int(caught_status),injections);
    std::fflush(stdout);
}
template<DdlaBackend B> void backend_cases(int ranks,int requested,int external) {
    DdlaHandle_t h=nullptr; ddlaInit(h,B);
    select_transport<B>(h);
    ddlaSet(h,MPI_COMM_WORLD,ranks,1,'R');
    if (!h->myid) std::printf("SELECTED_THREAD_TRANSPORT backend=%d requested=%s actual=%d\n",
        int(B), transport.c_str(), int(ddlaGetCommunication(h)));
    require(h,h->host_threads==requested && requested!=external,"handle budget must freeze a different count from external BLAS");
    Env selected("DDLA_EIG_TRIDIAGONAL","selected");
    for (const char* kind:{"success","status","exception"}) {
        run_case<B,float>(h,kind,external); run_case<B,double>(h,kind,external);
    }
    ddlaDestroy(h);
}
}
extern "C" float sdot_(const int* n,const float* x,const int* ix,const float* y,const int* iy) {
    observe_dot(); return real_sdot(n,x,ix,y,iy);
}
extern "C" double ddot_(const int* n,const double* x,const int* ix,const double* y,const int* iy) {
    observe_dot(); return real_ddot(n,x,ix,y,iy);
}
extern "C" void slagts_(const int* job,const int* n,const float* a,const float* b,const float* c,const float* d,
    const int* piv,float* y,float* tolerance,int* info) {
    if (!inject_solve_status(info)) real_slagts(job,n,a,b,c,d,piv,y,tolerance,info);
}
extern "C" void dlagts_(const int* job,const int* n,const double* a,const double* b,const double* c,const double* d,
    const int* piv,double* y,double* tolerance,int* info) {
    if (!inject_solve_status(info)) real_dlagts(job,n,a,b,c,d,piv,y,tolerance,info);
}
int main(int argc,char** argv) {
    int provided=0; MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int rank=0,ranks=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    try {
        int requested=1,external=2;
        for (int i=1;i<argc;++i) {
            if (!std::strcmp(argv[i],"--threads") && i+1<argc) requested=std::stoi(argv[++i]);
            else if (!std::strcmp(argv[i],"--blas-threads") && i+1<argc) external=std::stoi(argv[++i]);
            else if (!std::strcmp(argv[i],"--transport") && i+1<argc) {
                transport=argv[++i];
                if (transport!="ccl" && transport!="staged" && transport!="mpi")
                    throw std::invalid_argument("--transport requires ccl|staged|mpi");
            } else throw std::invalid_argument("usage: test_selected_threads [--threads N] [--blas-threads M] [--transport ccl|staged|mpi], N != M");
        }
        if (provided<MPI_THREAD_FUNNELED || requested<1 || requested>8 || external<1 || external>8 ||
            requested==external || ranks>17) throw std::invalid_argument("invalid thread control or rank count");
        const auto sd=symbol("sdot_",rank),dd=symbol("ddot_",rank),sl=symbol("slagts_",rank),dl=symbol("dlagts_",rank);
        const auto get=symbol("openblas_get_num_threads",rank),set=symbol("openblas_set_num_threads",rank);
        real_sdot=reinterpret_cast<Dot<float>>(sd.address); real_ddot=reinterpret_cast<Dot<double>>(dd.address);
        real_slagts=reinterpret_cast<Lagts<float>>(sl.address); real_dlagts=reinterpret_cast<Lagts<double>>(dl.address);
        int supported=openblas_get_num_threads && openblas_set_num_threads && sd.address && dd.address && sl.address && dl.address &&
            get.address && set.address && !sd.path.empty() && sd.path==dd.path && sd.path==sl.path && sd.path==dl.path &&
            sd.path==get.path && sd.path==set.path;
#if !defined(__linux__)
        supported=0;
#endif
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,&supported,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD));
        if (!supported) {
            if (!rank) std::puts("SELECTED_THREAD_TEST status=SKIP reason=requires_one_controllable_ELF_OpenBLAS_provider");
            MPI_Finalize(); return 77;
        }
        const auto budget=std::to_string(requested);
        Env eig("DDLA_EIG_THREADS",budget.c_str());
        detail::ScopedBlasThreads outside(external);
        backend_cases<DdlaBackend::CPU>(ranks,requested,external);
#if DDLA_HAS_GPU
        backend_cases<DdlaBackend::GPU>(ranks,requested,external);
#endif
        if (!rank) std::puts("SELECTED_THREAD_TEST status=PASS actual_workers_measured=0");
    } catch (const std::exception& error) {
        std::fprintf(stderr,"selected thread test rank=%d: %s\n",rank,error.what()); MPI_Abort(MPI_COMM_WORLD,1);
    } catch (...) {std::fprintf(stderr,"selected thread test rank=%d unexpected exception\n",rank);MPI_Abort(MPI_COMM_WORLD,1);}
    MPI_Finalize(); return 0;
}
