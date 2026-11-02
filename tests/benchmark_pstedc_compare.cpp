// One method, one size, one MPI process launch: tridiagonal eigensolver comparison.
// Input restoration, warmups and validation are identical for every implementation.
// pstedc and row-output use the same solver; the former includes descriptor transfer.
// Optional references are enabled only by the task-local comparison build.
#include "api_grid_test_common.h"
#include "Eigensolver/pstedc_impl.h"
#include "Eigensolver/pstedc_workspace.h"
#include "row_share.h"
#include "mpi_datatype.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#ifdef DDLA_COMPARE_HAVE_ELPA
#include <elpa/elpa.h>
#endif

extern "C" {
void dsterf_(const int*,double*,double*,int*);
void ssterf_(const int*,float*,float*,int*);
#ifdef DDLA_COMPARE_HAVE_SCALAPACK
int Csys2blacs_handle(MPI_Comm);
void Cblacs_gridinit(int*,const char*,int,int);
void Cblacs_gridinfo(int,int*,int*,int*,int*);
void Cblacs_gridexit(int);
void pdstedc_(const char*,const int*,double*,double*,double*,const int*,const int*,const int*,double*,const int*,int*,const int*,int*);
void psstedc_(const char*,const int*,float*,float*,float*,const int*,const int*,const int*,float*,const int*,int*,const int*,int*);
#endif
#ifdef DDLA_COMPARE_ELPA_GPU_BRIDGE
void* ddla_compare_elpa_gpu_create(int,int,int,int,int,int,int,int,int*);
void ddla_compare_elpa_gpu_destroy(void*,int*);
void ddla_compare_elpa_gpu_d(void*,int,int,int,int,int,double*,double*,double*,int*);
void ddla_compare_elpa_gpu_s(void*,int,int,int,int,int,float*,float*,float*,int*);
#endif
}

using namespace api_grid_test;
namespace {
struct Options {
    std::string method="pstedc",kind="random",backend="cpu",type="double",workspace="reuse";
    int n=5000,nb=64,pr=2,pc=2,warmups=1,reps=3,check_max=6000;
};

double hash01(std::uint64_t i,int tag)
{
    std::uint64_t x=(i+1)*0x9E3779B97F4A7C15ull^std::uint64_t(tag)*0xC2B2AE3D27D4EB4Full;
    x^=x>>30;x*=0xBF58476D1CE4E5B9ull;x^=x>>27;x*=0x94D049BB133111EBull;x^=x>>31;
    return double(x>>11)/9007199254740992.0-0.5;
}

unsigned long long status_bytes(const char* key)
{
    std::ifstream in("/proc/self/status");std::string line;
    while(std::getline(in,line))if(line.compare(0,std::strlen(key),key)==0){
        std::istringstream value(line.substr(std::strlen(key)));unsigned long long kb=0;value>>kb;return kb*1024;
    }
    return 0;
}

// VmHWM is the kernel's process-lifetime high-water mark. The sampling window
// excludes validation; device figures are whole-device usage, not exact per-PID
// allocations. Dedicated GPUs and the recorded baseline are required to interpret it.
class MemorySample {
    ddla::DdlaHandle_t h_;std::atomic<bool> running_{false};std::thread sampler_;
    void sample() {
        rss_peak=std::max(rss_peak,status_bytes("VmRSS:"));
        if(gpu){std::size_t free=0,total=0;if(ddla::ddlaMemGetInfo(h_,&free,&total)==ddla::ddlaStatus_t::DDLA_STATUS_SUCCESS)
            gpu_peak=std::max(gpu_peak,static_cast<unsigned long long>(total-free));}
    }
public:
    unsigned long long rss_before=0,rss_peak=0,rss_hwm_before=0,rss_hwm_after=0,gpu_before=0,gpu_peak=0;
    bool gpu;
    explicit MemorySample(ddla::DdlaHandle_t h):h_(h),gpu(ddla::ddlaGetBackend(h)==ddla::DdlaBackend::GPU) {
        rss_before=status_bytes("VmRSS:");rss_hwm_before=status_bytes("VmHWM:");sample();gpu_before=gpu_peak;
        running_=true;sampler_=std::thread([this]{
            if(gpu)DDLA_CHECK(ddla::ddlaSetDevice(h_));
            while(running_){sample();std::this_thread::sleep_for(std::chrono::milliseconds(2));}
        });
    }
    void stop(){if(running_.exchange(false)){sampler_.join();sample();rss_hwm_after=status_bytes("VmHWM:");}}
    ~MemorySample(){stop();}
};

template<class R>void sterf(int n,R* d,R* e,int* info)
{if constexpr(std::is_same_v<R,double>)dsterf_(&n,d,e,info);else ssterf_(&n,d,e,info);}

#ifdef DDLA_COMPARE_HAVE_SCALAPACK
template<class R>void scalapack_call(int n,R* d,R* e,R* q,int* desc,R* work,int lwork,int* iwork,int liwork,int* info)
{
    const char compz='I';const int one=1;
    if constexpr(std::is_same_v<R,double>)pdstedc_(&compz,&n,d,e,q,&one,&one,desc,work,&lwork,iwork,&liwork,info);
    else psstedc_(&compz,&n,d,e,q,&one,&one,desc,work,&lwork,iwork,&liwork,info);
}
#endif

#ifdef DDLA_COMPARE_HAVE_ELPA
void elpa_check(int error,const char* where)
{if(error!=ELPA_OK)throw std::runtime_error(std::string(where)+": "+elpa_strerr(error));}
void elpa_option(elpa_t h,const char* name,int value)
{int error=0;elpa_set_integer(h,name,value,&error);elpa_check(error,name);}
template<class R>void elpa_call(elpa_t h,R* d,R* e,R* q,int* info)
{if constexpr(std::is_same_v<R,double>)elpa_solve_tridiagonal_d(h,d,e,q,info);else elpa_solve_tridiagonal_f(h,d,e,q,info);}
#endif

template<class R>
void validate(const Options& o,ddla::DdlaHandle_t h,const std::vector<R>& local,int ld,int nr,int nc,
              const int* desc,const ddla::detail::RowShare& rows,const std::vector<R>& d0,
              const std::vector<R>& e0,const std::vector<R>& w)
{
    const int n=o.n,rank=ddla::ddlaGetRank(h);const bool slab=o.method=="row-output";
    const double eps=std::numeric_limits<R>::epsilon();double normT=0,trace=0,squares=0;
    for(int i=0;i<n;++i){normT=std::max(normT,std::abs(double(d0[i]))+std::abs(double(e0[i]))+(i?std::abs(double(e0[i-1])):0));
        trace+=d0[i];squares+=double(d0[i])*d0[i]+2*double(e0[i])*e0[i];}
    int bad=0;double residual=0,norm_error=0,orthogonal=0,eigen_error=-1,eigen_replica=0;
    std::vector<R> root_w(w);MPI_Bcast(root_w.data(),n,ddla::detail::mpi_datatype<R>(),0,MPI_COMM_WORLD);
    double gottrace=0,gotsquares=0;
    for(int j=0;j<n;++j){bad+=!std::isfinite(w[j])||(j&&w[j]<w[j-1]);
        eigen_replica=std::max(eigen_replica,std::abs(double(w[j])-root_w[j]));gottrace+=w[j];gotsquares+=double(w[j])*w[j];}
    const int samples=std::min(n,16),width=16;
    std::vector<double> sampled(std::size_t(n)*samples,0),panel(std::size_t(n)*width);
    for(int j0=0;j0<n;j0+=width){const int jb=std::min(width,n-j0);std::fill(panel.begin(),panel.end(),0);
        for(int jl=0;jl<nc;++jl){const int j=slab?jl:ddla::indx_l2g_c(desc,h,jl);if(j<j0||j>=j0+jb)continue;
            for(int il=0;il<nr;++il){const int i=slab?rows.lo(rank)+il:ddla::indx_l2g_r(desc,h,il);
                if(i<n)panel[i+std::size_t(j-j0)*n]=local[il+std::size_t(jl)*ld];}}
        MPI_Allreduce(MPI_IN_PLACE,panel.data(),n*jb,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        for(int jj=0;jj<jb;++jj){const int j=j0+jj;const double* x=panel.data()+std::size_t(jj)*n;double norm=0;
            for(int i=0;i<n;++i){if(!std::isfinite(x[i])){++bad;continue;}norm+=x[i]*x[i];double r=(double(d0[i])-w[j])*x[i];
                if(i)r+=double(e0[i-1])*x[i-1];if(i+1<n)r+=double(e0[i])*x[i+1];residual=std::max(residual,std::abs(r));}
            norm_error=std::max(norm_error,std::abs(norm-1));
            for(int s=0;s<samples;++s)if(j==s*(n-1)/std::max(1,samples-1))std::copy_n(x,n,sampled.data()+std::size_t(s)*n);
        }
    }
    for(int a=0;a<samples;++a)for(int b=0;b<=a;++b){double dot=0;
        for(int i=0;i<n;++i)dot+=sampled[i+std::size_t(a)*n]*sampled[i+std::size_t(b)*n];
        orthogonal=std::max(orthogonal,std::abs(dot-(a==b?1:0)));}
    std::string eigen_reference="invariants";
    if(o.kind=="toeplitz"){eigen_reference="analytic";eigen_error=0;
        for(int j=0;j<n;++j)eigen_error=std::max(eigen_error,std::abs(double(w[j])-(2-2*std::cos((j+1)*std::acos(-1.0)/(n+1)))));}
    else if(n<=o.check_max){eigen_reference="lapack_sterf";eigen_error=0;
        if(!rank){auto ref=d0,off=e0;int info=0;sterf(n,ref.data(),off.data(),&info);if(info)++bad;
            for(int j=0;j<n;++j)eigen_error=std::max(eigen_error,std::abs(double(w[j])-ref[j]));}
        MPI_Bcast(&eigen_error,1,MPI_DOUBLE,0,MPI_COMM_WORLD);}
    const double rmetric=residual/(normT*eps*n),ometric=orthogonal/(eps*n),nmetric=norm_error/(eps*n);
    const double tmetric=std::abs(gottrace-trace)/(eps*n*std::max(1.0,normT));
    const double smetric=std::abs(gotsquares-squares)/(eps*n*std::max(1.0,normT*normT));
    if(!(rmetric<=50)||!(ometric<=50)||!(nmetric<=50)||!(tmetric<=100)||!(smetric<=100)||
       !(eigen_replica/(eps*std::max(1.0,normT))<=100)||(eigen_error>=0&&!(eigen_error/(eps*normT)<=200)))++bad;
    MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    if(!rank)std::cout<<"PSTEDC_CHECK method="<<o.method<<" workspace="<<o.workspace<<" n="<<n<<" residual="<<rmetric<<" all_column_norm="<<nmetric
        <<" sampled_ortho="<<ometric<<" trace="<<tmetric<<" spectral_square="<<smetric<<" eigen_reference="<<eigen_reference
        <<" eigenvalue_abs_error="<<eigen_error<<" eigenvalue_replica_abs_error="<<eigen_replica<<" status="<<(bad?"FAIL":"PASS")<<std::endl;
    if(bad)throw std::runtime_error("distributed eigenpair validation failed");
}

template<class R>void run(const Options& o,int rank,int P)
{
    const bool slab=o.method=="row-output",reference=o.method=="elpa"||o.method=="scalapack";
    const bool gpu=o.backend=="gpu";
    ddla::DdlaHandle_t h=nullptr;DDLA_CHECK(ddla::ddlaInit(h,gpu?ddla::DdlaBackend::GPU:ddla::DdlaBackend::CPU));
    DDLA_CHECK(ddla::ddlaSet(h,MPI_COMM_WORLD,o.pr,o.pc));
    // Start before allocating the output or any solver workspace.
    MemorySample memory(h);
    int desc[ddla::DDLA_DLEN_];DDLA_CHECK(ddla::ddlaDescInit(desc,h,o.n,o.n,o.nb,o.nb,0,0));
    const ddla::detail::RowShare rows(o.n,P);
    const int nr=slab?rows.count(rank):ddla_test::m_loc(h,desc),nc=slab?o.n:ddla_test::n_loc(h,desc);
    const int ld=slab?std::max(1,nr):desc[ddla::DDLA_LLD_];
    const std::size_t output_count=std::max<std::size_t>(1,std::size_t(ld)*nc);
    std::unique_ptr<DeviceBuffer<R>> device;
    std::vector<R> host_output;
    if(reference&&!gpu)host_output.resize(output_count);else device=std::make_unique<DeviceBuffer<R>>(h,output_count);
    R* output=device?device->ptr:host_output.data();
    std::vector<R> d0(o.n),e0(o.n,0),d(o.n),e(o.n),w(o.n);std::vector<double> times;
    for(int i=0;i<o.n;++i){d0[i]=R(o.kind=="toeplitz"?2:hash01(i,1));if(i+1<o.n)e0[i]=R(o.kind=="toeplitz"?-1:hash01(i,2));}
    std::size_t workspace_backend=0,workspace_host=0;bool workspace_known=false;
    ddla::spectral_impl::PstedcVectorPlan rows_plan{};
    if(slab){rows_plan=ddla::spectral_impl::pstedc_check<R>(h,o.n,d0.data(),e0.data(),w.data(),output,ld);
        ddla::spectral_impl::pstedc_workspace_size<R>(h,rows_plan,&workspace_backend,&workspace_host);workspace_known=true;}
    ddla::spectral_impl::PstedcPlan descriptor_plan{};
    if(o.method=="pstedc"){descriptor_plan=ddla::spectral_impl::pstedc_check<R>(h,'V',o.n,d0.data(),e0.data(),w.data(),output,desc);
        ddla::spectral_impl::pstedc_workspace_size<R>(h,descriptor_plan,&workspace_backend,&workspace_host);workspace_known=true;}
    std::unique_ptr<DeviceBuffer<unsigned char>> backend_arena;
    std::vector<std::max_align_t> host_arena;
    std::unique_ptr<ddla::spectral_impl::PstedcResources> rows_resources;
    if(!reference&&o.workspace=="reuse"){
        if(workspace_backend)backend_arena=std::make_unique<DeviceBuffer<unsigned char>>(h,workspace_backend);
        host_arena.resize((workspace_host+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t));
        rows_resources=std::make_unique<ddla::spectral_impl::PstedcResources>(h,slab?rows_plan:descriptor_plan.rows);
    }
    auto ddla_reuse=[&](auto backend_tag){
        constexpr ddla::DdlaBackend B=decltype(backend_tag)::value;
        void* backend_work=backend_arena?backend_arena->ptr:nullptr;
        void* host_work=host_arena.empty()?nullptr:host_arena.data();
        if(slab)ddla::spectral_impl::pstedc_run<B,R>(h,rows_plan,*rows_resources,d.data(),e.data(),w.data(),output,nullptr,
            backend_work,workspace_backend,host_work,workspace_host);
        else ddla::spectral_impl::pstedc_run<B,R>(h,descriptor_plan,d.data(),e.data(),w.data(),output,
            backend_work,workspace_backend,host_work,workspace_host,rows_resources.get());
    };
    int blacs=-1;std::vector<R> scawork;std::vector<int> scaiwork;
#ifdef DDLA_COMPARE_HAVE_SCALAPACK
    if(o.method=="scalapack"){
        blacs=Csys2blacs_handle(MPI_COMM_WORLD);Cblacs_gridinit(&blacs,"R",o.pr,o.pc);
        int pr=0,pc=0,ir=0,ic=0;Cblacs_gridinfo(blacs,&pr,&pc,&ir,&ic);
        if(pr!=o.pr||pc!=o.pc||ir!=rank/o.pc||ic!=rank%o.pc)throw std::runtime_error("BLACS grid disagrees with DDLA row-major grid");
        desc[ddla::DDLA_CTXT_]=blacs;R query=0;int iquery=0,info=0;d=d0;e=e0;
        scalapack_call(o.n,d.data(),e.data(),output,desc,&query,-1,&iquery,-1,&info);
        if(info||!(query>=1)||iquery<1)throw std::runtime_error("ScaLAPACK workspace query failed");
        scawork.resize(static_cast<std::size_t>(std::ceil(query)));scaiwork.resize(iquery);
        workspace_backend=scawork.size()*sizeof(R)+scaiwork.size()*sizeof(int);workspace_known=true;
    }
#else
    if(o.method=="scalapack")throw std::runtime_error("ScaLAPACK reference not enabled in this executable");
#endif
#ifdef DDLA_COMPARE_HAVE_ELPA
    elpa_t elpa=nullptr;std::unique_ptr<DeviceBuffer<R>> device_d,device_e;
    if(o.method=="elpa"){
        if(elpa_init(ELPA_API_VERSION)!=ELPA_OK)throw std::runtime_error("elpa_init failed");
        int error=0;
        if(gpu){
#ifdef DDLA_COMPARE_ELPA_GPU_BRIDGE
            int devices=0,local_rank=0;MPI_Comm local=MPI_COMM_NULL;
            DDLA_CHECK(ddla::ddlaGetDeviceCount(h,&devices));
            MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&local);
            MPI_Comm_rank(local,&local_rank);MPI_Comm_free(&local);
            if(devices<1)throw std::runtime_error("ELPA GPU reference has no visible device");
            elpa=static_cast<elpa_t>(ddla_compare_elpa_gpu_create(o.n,nr,nc,o.nb,rank/o.pc,rank%o.pc,
                int(MPI_Comm_c2f(MPI_COMM_WORLD)),local_rank%devices,&error));
            elpa_check(error,"ddla_compare_elpa_gpu_create");
            if(!elpa)throw std::runtime_error("ELPA GPU bridge returned a null object");
            device_d=std::make_unique<DeviceBuffer<R>>(h,o.n);device_e=std::make_unique<DeviceBuffer<R>>(h,o.n);
            workspace_backend=2*std::size_t(o.n)*sizeof(R);
#else
            throw std::runtime_error("ELPA public solve_tridiagonal is CPU-only; private GPU bridge not enabled");
#endif
        }else{
            elpa=elpa_allocate(&error);elpa_check(error,"elpa_allocate");
            elpa_option(elpa,"na",o.n);elpa_option(elpa,"nev",o.n);elpa_option(elpa,"local_nrows",nr);
            elpa_option(elpa,"local_ncols",nc);elpa_option(elpa,"nblk",o.nb);
            elpa_option(elpa,"mpi_comm_parent",int(MPI_Comm_c2f(MPI_COMM_WORLD)));
            elpa_option(elpa,"process_row",rank/o.pc);elpa_option(elpa,"process_col",rank%o.pc);
            elpa_check(elpa_setup(elpa),"elpa_setup");
            if(const char* threads=std::getenv("OMP_NUM_THREADS")){
                // The CPU reference links an OpenMP-enabled ELPA.
                elpa_option(elpa,"omp_threads",std::max(1,std::atoi(threads)));
            }
        }
    }
#else
    if(o.method=="elpa")throw std::runtime_error("ELPA reference not enabled in this executable");
#endif
    for(int rep=-o.warmups;rep<o.reps;++rep){d=d0;e=e0;check_ddla_sync(h);MPI_Barrier(MPI_COMM_WORLD);const double t0=MPI_Wtime();int info=0;
        if(!reference&&o.workspace=="reuse"){
            if(gpu){
#if DDLA_HAS_GPU
                ddla_reuse(std::integral_constant<ddla::DdlaBackend,ddla::DdlaBackend::GPU>{});
#else
                throw std::runtime_error("GPU workspace execution unavailable in this build");
#endif
            }else{
#if DDLA_HAS_CPU
                ddla_reuse(std::integral_constant<ddla::DdlaBackend,ddla::DdlaBackend::CPU>{});
#else
                throw std::runtime_error("CPU workspace execution unavailable in this build");
#endif
            }
        }
        else if(slab)ddla::pstedc<R>(h,o.n,d.data(),e.data(),w.data(),output,ld);
        else if(o.method=="pstedc"){
            ddla::pstedc<R>(h,'V',o.n,d.data(),e.data(),w.data(),output,desc,info);
        }
#ifdef DDLA_COMPARE_HAVE_SCALAPACK
        else if(o.method=="scalapack"){
            // P?LAED0 overwrites only diagonal leaf blocks. Include P?SYEVD's
            // required P?LASET(identity) preparation in this method's timer.
            std::fill(host_output.begin(),host_output.end(),R(0));
            for(int jl=0;jl<nc;++jl){const int j=ddla::indx_l2g_c(desc,h,jl),il=ddla::indx_g2l_r(desc,h,j);
                if(il>=0)host_output[il+std::size_t(jl)*ld]=R(1);}
            scalapack_call(o.n,d.data(),e.data(),output,desc,scawork.data(),int(scawork.size()),scaiwork.data(),int(scaiwork.size()),&info);w=d;
        }
#endif
#ifdef DDLA_COMPARE_HAVE_ELPA
        else if(o.method=="elpa"){
            if(!gpu)elpa_call(elpa,d.data(),e.data(),output,&info);
#ifdef DDLA_COMPARE_ELPA_GPU_BRIDGE
            else{upload(h,device_d->ptr,d);upload(h,device_e->ptr,e);check_ddla_sync(h);
                if constexpr(std::is_same_v<R,double>)ddla_compare_elpa_gpu_d(elpa,o.n,nr,nc,o.nb,1,device_d->ptr,device_e->ptr,output,&info);
                else ddla_compare_elpa_gpu_s(elpa,o.n,nr,nc,o.nb,1,device_d->ptr,device_e->ptr,output,&info);
                // The comparison starts and ends with replicated HOST tridiagonal/eigenvalue arrays.
                d=download(h,device_d->ptr,o.n);
            }
#endif
            w=d;
        }
#endif
        check_ddla_sync(h);const double dt=MPI_Wtime()-t0;double maximum=0;
        MPI_Allreduce(&dt,&maximum,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
        int badinfo=info!=0;MPI_Allreduce(MPI_IN_PLACE,&badinfo,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
        if(badinfo)throw std::runtime_error("solver returned nonzero info (local info="+std::to_string(info)+")");
        if(rep>=0){times.push_back(maximum);if(!rank)std::cout<<"PSTEDC_TIME method="<<o.method<<" workspace="<<o.workspace<<" backend="<<o.backend<<" n="<<o.n
            <<" kind="<<o.kind<<" type="<<o.type<<" rep="<<rep<<" seconds="<<std::setprecision(10)<<maximum<<std::endl;}
    }
    memory.stop();
    unsigned long long bytes[]={output_count*sizeof(R),workspace_backend,workspace_host,
        memory.rss_before,memory.rss_peak,memory.rss_hwm_before,memory.rss_hwm_after,memory.gpu_before,memory.gpu_peak};
    MPI_Allreduce(MPI_IN_PLACE,bytes,9,MPI_UNSIGNED_LONG_LONG,MPI_MAX,MPI_COMM_WORLD);
    auto sorted=times;std::sort(sorted.begin(),sorted.end());const double median=sorted.size()%2?sorted[sorted.size()/2]:0.5*(sorted[sorted.size()/2-1]+sorted[sorted.size()/2]);
    if(!rank)std::cout<<"PSTEDC_COMPARE method="<<o.method<<" workspace="<<o.workspace
        <<" allocation_policy="<<(o.method=="elpa"?"library_managed":o.method=="scalapack"||o.workspace=="reuse"?"caller_reused":"per_call")
        <<" backend="<<o.backend<<" n="<<o.n<<" kind="<<o.kind<<" type="<<o.type
        <<" grid="<<o.pr<<'x'<<o.pc<<" nb="<<o.nb<<" ranks="<<P<<" warmups="<<o.warmups<<" reps="<<o.reps
        <<std::setprecision(10)<<" min="<<sorted.front()<<" median="<<median<<" max="<<sorted.back()
        <<" output_bytes_max="<<bytes[0]<<" workspace_known="<<workspace_known<<" workspace_backend_bytes_max="<<bytes[1]<<" workspace_host_bytes_max="<<bytes[2]
        <<" rss_baseline_bytes_max="<<bytes[3]<<" rss_sampled_peak_bytes_max="<<bytes[4]<<" rss_hwm_baseline_bytes_max="<<bytes[5]<<" rss_hwm_after_bytes_max="<<bytes[6]
        <<" device_used_baseline_bytes_max="<<bytes[7]<<" device_used_sampled_peak_bytes_max="<<bytes[8]
        <<" device_memory_scope=whole_device sample_interval_ms=2 memory_window=before_output_through_solver"
        <<" scalapack_q_initialization="<<(o.method=="scalapack"?"timed_identity_before_each_call":"na")
        <<" elpa_interface="<<(o.method=="elpa"?(gpu?"private_device_solve_tridi":"public_host_solve_tridiagonal"):"na")
#ifdef DDLA_COMPARE_ELPA_VERSION
        <<" elpa_version="<<DDLA_COMPARE_ELPA_VERSION
#endif
        <<std::endl;
    const auto local=device?download(h,output,output_count):host_output;
    validate(o,h,local,ld,nr,nc,desc,rows,d0,e0,w);
#ifdef DDLA_COMPARE_HAVE_ELPA
    if(elpa){int error=0;
#ifdef DDLA_COMPARE_ELPA_GPU_BRIDGE
        if(gpu)ddla_compare_elpa_gpu_destroy(elpa,&error);else
#endif
        elpa_deallocate(elpa,&error);
        elpa_check(error,"elpa_deallocate");elpa_uninit(&error);}
    device_d.reset();device_e.reset();
#endif
#ifdef DDLA_COMPARE_HAVE_SCALAPACK
    if(blacs>=0)Cblacs_gridexit(blacs);
#endif
    rows_resources.reset();backend_arena.reset();device.reset();DDLA_CHECK(ddla::ddlaDestroy(h));
}
}

int main(int argc,char** argv)
{
    int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int rank=0,P=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&P);
    try{Options o;
        for(int i=1;i<argc;++i){const std::string a=argv[i];
            auto value=[&](){if(i+1>=argc)throw std::invalid_argument("missing value for "+a);return std::string(argv[++i]);};
            if(a=="--method")o.method=value();else if(a=="--kind")o.kind=value();else if(a=="--backend")o.backend=value();
            else if(a=="--type")o.type=value();else if(a=="--n")o.n=std::stoi(value());else if(a=="--nb")o.nb=std::stoi(value());
            else if(a=="--workspace")o.workspace=value();
            else if(a=="--grid"){if(!parse_grid_spec(value(),o.pr,o.pc))throw std::invalid_argument("invalid grid");}
            else if(a=="--warmups")o.warmups=std::stoi(value());else if(a=="--reps")o.reps=std::stoi(value());
            else if(a=="--check-max")o.check_max=std::stoi(value());else throw std::invalid_argument("unknown option "+a);
        }
        if(o.n<1||o.nb<1||o.warmups<1||o.reps<1||o.pr*o.pc!=P||
           (o.kind!="random"&&o.kind!="toeplitz")||(o.backend!="cpu"&&o.backend!="gpu")||
           (o.method!="row-output"&&o.method!="pstedc"&&o.method!="elpa"&&o.method!="scalapack")||
           (o.type!="double"&&o.type!="float")||(o.method=="scalapack"&&o.backend!="cpu"))throw std::invalid_argument("invalid comparison options");
        if(o.workspace!="reuse"&&o.workspace!="public")throw std::invalid_argument("invalid workspace mode");
        if(provided<MPI_THREAD_FUNNELED)throw std::runtime_error("MPI_THREAD_FUNNELED unavailable");
        if(!rank)std::cout<<"PSTEDC_PROVENANCE pid="<<getpid()<<" OMP_NUM_THREADS="<<(std::getenv("OMP_NUM_THREADS")?std::getenv("OMP_NUM_THREADS"):"unset")
            <<" OPENBLAS_NUM_THREADS="<<(std::getenv("OPENBLAS_NUM_THREADS")?std::getenv("OPENBLAS_NUM_THREADS"):"unset")<<std::endl;
        if(o.type=="float")run<float>(o,rank,P);else run<double>(o,rank,P);
    }catch(const std::exception& e){std::cerr<<"rank "<<rank<<": "<<e.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);}
    MPI_Finalize();return 0;
}
