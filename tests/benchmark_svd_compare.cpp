#include "mpi_test_init.h"
// End-to-end thin SVD benchmark. Input reset is excluded; workspace query,
// allocation, solve and completion are included for every implementation.
// Usage: benchmark_svd_compare cpu|gpu PRxPC M N NB REPEATS d|z
//                              ddla|lapack|scalapack|cusolver
// Each invocation measures one implementation, allowing same-job alternating
// baseline/candidate runs without loading two libraries in one process.
#include <ddla/ddla.h>
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "test_desc_helpers.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <sys/resource.h>

#if defined(DDLA_USE_CUDA)
#include <cuda_runtime.h>
#include <cusolverDn.h>
#endif

extern "C" {
void dgesdd_(const char*, const int*, const int*, double*, const int*, double*,
             double*, const int*, double*, const int*, double*, const int*, int*, int*);
void zgesdd_(const char*, const int*, const int*, std::complex<double>*, const int*, double*,
             std::complex<double>*, const int*, std::complex<double>*, const int*,
             std::complex<double>*, const int*, double*, int*, int*);
#if defined(DDLA_BENCH_SCALAPACK)
void Cblacs_get(int, int, int*);
void Cblacs_gridinit(int*, const char*, int, int);
void Cblacs_gridinfo(int, int*, int*, int*, int*);
void Cblacs_gridexit(int);
void pdgesvd_(const char*, const char*, const int*, const int*, double*, const int*,
              const int*, const int*, double*, double*, const int*, const int*,
              const int*, double*, const int*, const int*, const int*, double*, const int*, int*);
void pzgesvd_(const char*, const char*, const int*, const int*, std::complex<double>*,
              const int*, const int*, const int*, double*, std::complex<double>*,
              const int*, const int*, const int*, std::complex<double>*, const int*,
              const int*, const int*, std::complex<double>*, const int*, double*, int*);
#endif
}

using namespace ddla;

namespace {

constexpr int probes = 4;
template<class T> T conj_value(T x) { return x; }
template<class R> std::complex<R> conj_value(std::complex<R> x) { return std::conj(x); }
template<class T> MPI_Datatype mpi_type();
template<> MPI_Datatype mpi_type<double>() { return MPI_DOUBLE; }
template<> MPI_Datatype mpi_type<std::complex<double>>() { return MPI_C_DOUBLE_COMPLEX; }

double random_value(int i, int j, int tag)
{
    std::uint64_t x = std::uint64_t(i + 1) * 0x9e3779b97f4a7c15ull
                    ^ std::uint64_t(j + 1) * 0xc2b2ae3d27d4eb4full
                    ^ std::uint64_t(tag + 1) * 0x165667b19e3779f9ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    x ^= x >> 31;
    return double(x >> 11) / 9007199254740992.0 - 0.5;
}

template<class T> T entry(int i, int j, int tag)
{
    if constexpr (std::is_same<T,double>::value) return random_value(i,j,tag);
    else return T(random_value(i,j,tag), random_value(i,j,tag+1000));
}

template<class T> double norm(const std::vector<T>& x)
{
    long double ss = 0;
    for (const T& v : x) ss += std::norm(v);
    return std::sqrt(double(ss));
}

template<class T>
std::vector<T> apply(const DdlaHandle_t& h, const std::vector<T>& a, const int* d,
                     bool adj, const std::vector<T>& x)
{
    const int m = d[DDLA_M_], n = d[DDLA_N_], out = adj ? n : m, in = adj ? m : n;
    const int mr = ddla_test::m_loc(h,d), nc = ddla_test::n_loc(h,d);
    std::vector<T> y(std::size_t(out)*probes, T(0));
    for (int j=0; j<nc; ++j) {
        const int gj=indx_l2g_c(d,h,j);
        for (int i=0; i<mr; ++i) {
            const int gi=indx_l2g_r(d,h,i);
            const T v=a[i+std::size_t(j)*d[DDLA_LLD_]];
            for (int p=0; p<probes; ++p)
                if (adj) y[gj+std::size_t(p)*out] += conj_value(v)*x[gi+std::size_t(p)*in];
                else y[gi+std::size_t(p)*out] += v*x[gj+std::size_t(p)*in];
        }
    }
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,y.data(),int(y.size()),mpi_type<T>(),MPI_SUM,
                            ddlaGetCommunicator(h)));
    return y;
}

template<class T>
void check(const DdlaHandle_t& h, const std::vector<T>& a, const int* da,
           const std::vector<T>& u, const int* du, const std::vector<T>& vt, const int* dv,
           const std::vector<double>& s, double& residual, double& orth_u, double& orth_v)
{
    const int m=da[DDLA_M_],n=da[DDLA_N_],k=int(s.size());
    const double eps=std::numeric_limits<double>::epsilon();
    for (int i=0; i<k; ++i)
        if (!std::isfinite(s[i]) || s[i]<0 || (i && s[i]>s[i-1]))
            throw std::runtime_error("singular values are not finite, nonnegative and descending");
    auto root_s=s;
    MPI_CHECK(MPI_Bcast(root_s.data(),k,MPI_DOUBLE,0,ddlaGetCommunicator(h)));
    for (int i=0; i<k; ++i)
        if (std::abs(root_s[i]-s[i])>32*eps*std::max(1.0,s[0]))
            throw std::runtime_error("singular values disagree between ranks");
    std::vector<T> x(std::size_t(n)*probes),q(std::size_t(k)*probes);
    for (int p=0;p<probes;++p) {
        for (int i=0;i<n;++i) x[i+std::size_t(p)*n]=entry<T>(i,p,41);
        for (int i=0;i<k;++i) q[i+std::size_t(p)*k]=entry<T>(i,p,43);
    }
    const auto ax=apply(h,a,da,false,x);
    auto w=apply(h,vt,dv,false,x);
    for(int p=0;p<probes;++p) for(int i=0;i<k;++i) w[i+std::size_t(p)*k]*=s[i];
    auto diff=apply(h,u,du,false,w);
    for(std::size_t i=0;i<diff.size();++i)diff[i]-=ax[i];
    double a2=0;
    for(int j=0;j<ddla_test::n_loc(h,da);++j)
        for(int i=0;i<ddla_test::m_loc(h,da);++i)
            a2+=std::norm(a[i+std::size_t(j)*da[DDLA_LLD_]]);
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,&a2,1,MPI_DOUBLE,MPI_SUM,ddlaGetCommunicator(h)));
    residual=norm(diff)/(std::sqrt(a2)*norm(x)*std::max(m,n)*eps);
    diff=apply(h,u,du,true,apply(h,u,du,false,q));
    for(std::size_t i=0;i<diff.size();++i)diff[i]-=q[i];
    orth_u=norm(diff)/(norm(q)*m*eps);
    diff=apply(h,vt,dv,false,apply(h,vt,dv,true,q));
    for(std::size_t i=0;i<diff.size();++i)diff[i]-=q[i];
    orth_v=norm(diff)/(norm(q)*n*eps);
    if (!(residual<50 && orth_u<50 && orth_v<50))
        throw std::runtime_error("SVD reconstruction/orthogonality check failed");
}

void lapack(int m,int n,double* a,int lda,double* s,double* u,int ldu,double* vt,int ldv,
            std::size_t& workspace,int& info)
{
    const char job='S'; const int k=std::min(m,n); int lw=-1;
    double query=0;std::vector<int> iw(8*std::size_t(k));
    dgesdd_(&job,&m,&n,a,&lda,s,u,&ldu,vt,&ldv,&query,&lw,iw.data(),&info);
    if(info)return;
    if(!(query>=1 && query<=std::numeric_limits<int>::max())) throw std::runtime_error("invalid DGESDD workspace");
    lw=int(query);std::vector<double>w(lw);
    workspace=w.size()*sizeof(double)+iw.size()*sizeof(int);
    dgesdd_(&job,&m,&n,a,&lda,s,u,&ldu,vt,&ldv,w.data(),&lw,iw.data(),&info);
}

void lapack(int m,int n,std::complex<double>* a,int lda,double* s,std::complex<double>* u,
            int ldu,std::complex<double>* vt,int ldv,std::size_t& workspace,int& info)
{
    const char job='S';const int k=std::min(m,n);int lw=-1;
    std::complex<double>query{};std::vector<int>iw(8*std::size_t(k));
    // LAPACK ZGESDD JOBZ='S' real workspace bound, valid for either shape.
    std::vector<double>rw(std::size_t(k)*std::max(5LL*k+7,2LL*std::max(m,n)+2LL*k+1));
    zgesdd_(&job,&m,&n,a,&lda,s,u,&ldu,vt,&ldv,&query,&lw,rw.data(),iw.data(),&info);
    if(info)return;
    if(!(query.real()>=1 && query.real()<=std::numeric_limits<int>::max()))throw std::runtime_error("invalid ZGESDD workspace");
    lw=int(query.real());std::vector<std::complex<double>>w(lw);
    workspace=w.size()*sizeof(w[0])+rw.size()*sizeof(double)+iw.size()*sizeof(int);
    zgesdd_(&job,&m,&n,a,&lda,s,u,&ldu,vt,&ldv,w.data(),&lw,rw.data(),iw.data(),&info);
}

#if defined(DDLA_BENCH_SCALAPACK)
template<class T>
void scalapack(int m,int n,T* a,const int* da,double* s,T* u,const int* du,T* vt,
               const int* dv,std::size_t& workspace,int& info)
{
    const char job='V';const int one=1;int lw=-1;T query{};
    std::vector<double>rw;
    if constexpr(!std::is_same<T,double>::value)rw.resize(1+4*std::size_t(std::max(m,n)));
    auto call=[&](T* w){
        if constexpr(std::is_same<T,double>::value)
            pdgesvd_(&job,&job,&m,&n,a,&one,&one,da,s,u,&one,&one,du,vt,&one,&one,dv,w,&lw,&info);
        else pzgesvd_(&job,&job,&m,&n,a,&one,&one,da,s,u,&one,&one,du,vt,&one,&one,dv,w,&lw,rw.data(),&info);
    };
    call(&query);if(info)return;
    if(!(std::real(query)>=1 && std::real(query)<=std::numeric_limits<int>::max()))throw std::runtime_error("invalid ScaLAPACK workspace");
    lw=int(std::real(query));std::vector<T>w(lw);
    workspace=w.size()*sizeof(T)+rw.size()*sizeof(double);call(w.data());
}
#endif

#if defined(DDLA_USE_CUDA)
template<class T>
void cuda_svd(const DdlaHandle_t& h,int m,int n,T* a,int lda,double* s,T* u,int ldu,
              T* vt,int ldv,std::size_t& workspace,int& info)
{
    int lw=0;
    if constexpr(std::is_same<T,double>::value)
        SOLVER_CHECK(cusolverDnDgesvd_bufferSize(static_cast<gpusolverHandle_t>(h->solverH),m,n,&lw));
    else SOLVER_CHECK(cusolverDnZgesvd_bufferSize(static_cast<gpusolverHandle_t>(h->solverH),m,n,&lw));
    ddla::vector<T>w(h,lw);
    ddla::vector<double>ds(h,n),rw(h,std::max(1,n-1));
    ddla::vector<int>di(h,1);
    workspace=std::size_t(lw)*sizeof(T)+(std::size_t(n)+std::max(1,n-1))*sizeof(double)+sizeof(int);
    if constexpr(std::is_same<T,double>::value)
        SOLVER_CHECK(cusolverDnDgesvd(static_cast<gpusolverHandle_t>(h->solverH),'S','S',m,n,a,lda,ds.data(),u,ldu,vt,ldv,w.data(),lw,rw.data(),di.data()));
    else SOLVER_CHECK(cusolverDnZgesvd(static_cast<gpusolverHandle_t>(h->solverH),'S','S',m,n,reinterpret_cast<cuDoubleComplex*>(a),lda,
            ds.data(),reinterpret_cast<cuDoubleComplex*>(u),ldu,reinterpret_cast<cuDoubleComplex*>(vt),ldv,
            reinterpret_cast<cuDoubleComplex*>(w.data()),lw,rw.data(),di.data()));
    DDLA_CHECK(ddlaMemcpy(h,&info,di.data(),sizeof(int)));
    DDLA_CHECK(ddlaMemcpy(h,s,ds.data(),std::size_t(n)*sizeof(double)));
    DDLA_CHECK(ddlaSynchronize(h));
}

std::uint64_t pool_peak(const DdlaHandle_t& h,bool reset)
{
    if(ddlaGetBackend(h)!=DdlaBackend::GPU)return 0;
    int dev=0;GPU_CHECK(cudaGetDevice(&dev));cudaMemPool_t pool;
    GPU_CHECK(cudaDeviceGetDefaultMemPool(&pool,dev));std::uint64_t bytes=0;
    if(reset)GPU_CHECK(cudaMemPoolSetAttribute(pool,cudaMemPoolAttrUsedMemHigh,&bytes));
    GPU_CHECK(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrUsedMemHigh,&bytes));return bytes;
}
#else
std::uint64_t pool_peak(const DdlaHandle_t&,bool){return 0;}
#endif

template<class T>
void run(const DdlaHandle_t& h,int pr,int pc,int m,int n,int nb,int reps,const std::string& mode,
          const std::string& type,int blacs)
{
    const int rank=ddlaGetRank(h),k=std::min(m,n);
    const bool cpu=ddlaGetBackend(h)==DdlaBackend::CPU;
    MPI_Comm comm=ddlaGetCommunicator(h);
    int da[DDLA_DLEN_],du[DDLA_DLEN_],dv[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(da,h,m,n,nb,nb,0,0));
    DDLA_CHECK(ddlaDescInit(du,h,m,k,nb,nb,0,0));
    DDLA_CHECK(ddlaDescInit(dv,h,k,n,nb,nb,0,0));
    const auto count=[&](const int* d){return std::size_t(d[DDLA_LLD_])*std::max(1,ddla_test::n_loc(h,d));};
    const std::size_t na=count(da),nu=count(du),nv=count(dv);
    std::vector<T>a0(na,T(0));
    for(int j=0;j<ddla_test::n_loc(h,da);++j)for(int i=0;i<ddla_test::m_loc(h,da);++i)
        a0[i+std::size_t(j)*da[DDLA_LLD_]]=entry<T>(indx_l2g_r(da,h,i),indx_l2g_c(da,h,j),3);
    ddla::vector<T>a(h,na),u(h,nu),vt(h,nv);std::vector<double>s(k);
    int sa[DDLA_DLEN_],su[DDLA_DLEN_],sv[DDLA_DLEN_];
    std::copy_n(da,DDLA_DLEN_,sa);std::copy_n(du,DDLA_DLEN_,su);std::copy_n(dv,DDLA_DLEN_,sv);
    sa[DDLA_CTXT_]=su[DDLA_CTXT_]=sv[DDLA_CTXT_]=blacs;
    std::size_t ref_workspace=0;std::vector<double>times;
    std::uint64_t gpu_peak=0;
    for(int rep=-1;rep<reps;++rep){
        DDLA_CHECK(ddlaMemcpy(h,a.data(),a0.data(),na*sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));MPI_CHECK(MPI_Barrier(comm));pool_peak(h,true);
        const double begin=MPI_Wtime();int info=0;
        if(mode=="ddla"){
            if(cpu)pgesvd<DdlaBackend::CPU,T>(h,'V','V',m,n,a.data(),da,s.data(),u.data(),du,vt.data(),dv,info);
#if DDLA_HAS_GPU
            if(!cpu)pgesvd<DdlaBackend::GPU,T>(h,'V','V',m,n,a.data(),da,s.data(),u.data(),du,vt.data(),dv,info);
#endif
        }else if(mode=="lapack")lapack(m,n,a.data(),da[DDLA_LLD_],s.data(),u.data(),du[DDLA_LLD_],vt.data(),dv[DDLA_LLD_],ref_workspace,info);
#if defined(DDLA_BENCH_SCALAPACK)
        else if(mode=="scalapack")scalapack(m,n,a.data(),sa,s.data(),u.data(),su,vt.data(),sv,ref_workspace,info);
#endif
#if defined(DDLA_USE_CUDA)
        else if(mode=="cusolver")cuda_svd(h,m,n,a.data(),da[DDLA_LLD_],s.data(),u.data(),du[DDLA_LLD_],vt.data(),dv[DDLA_LLD_],ref_workspace,info);
#endif
        DDLA_CHECK(ddlaSynchronize(h));double elapsed=MPI_Wtime()-begin,max_time=0;
        MPI_CHECK(MPI_Allreduce(&elapsed,&max_time,1,MPI_DOUBLE,MPI_MAX,comm));
        int any_info=0,abs_info=std::abs(info);MPI_CHECK(MPI_Allreduce(&abs_info,&any_info,1,MPI_INT,MPI_MAX,comm));
        if(any_info)throw std::runtime_error("SVD returned nonzero INFO="+std::to_string(any_info));
        gpu_peak=std::max(gpu_peak,pool_peak(h,false));
        if(rep>=0){times.push_back(max_time);if(!rank)std::cout<<std::setprecision(12)<<"RAW mode="<<mode<<" backend="<<(cpu?"cpu":"gpu")<<" type="<<type<<" grid="<<pr<<'x'<<pc<<" m="<<m<<" n="<<n<<" nb="<<nb<<" rep="<<rep<<" seconds="<<max_time<<std::endl;}
    }
    // Capture memory before validation allocates host copies of the factors.
    // RSS is process-lifetime high-water, not isolated SVD workspace. The CUDA
    // pool high-water includes caller matrices plus asynchronous allocations;
    // it excludes vendor/context allocations made outside that memory pool.
    struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
    std::uint64_t rss=std::uint64_t(usage.ru_maxrss)*1024;
#if defined(__APPLE__)
    rss=std::uint64_t(usage.ru_maxrss);
#endif
    const std::uint64_t caller=(na+nu+nv)*sizeof(T);
    const std::uint64_t host_input=na*sizeof(T)+std::size_t(k)*sizeof(double);
    std::uint64_t local_mem[5]={caller,rss,gpu_peak,std::uint64_t(ref_workspace),host_input};
    std::vector<std::uint64_t>all_mem(rank==0?5*std::size_t(pr)*pc:0);
    MPI_CHECK(MPI_Gather(local_mem,5,MPI_UINT64_T,all_mem.data(),5,MPI_UINT64_T,0,comm));
    std::vector<T>hu(nu),hv(nv);DDLA_CHECK(ddlaMemcpy(h,hu.data(),u.data(),nu*sizeof(T)));
    DDLA_CHECK(ddlaMemcpy(h,hv.data(),vt.data(),nv*sizeof(T)));DDLA_CHECK(ddlaSynchronize(h));
    double residual=0,ou=0,ov=0;check(h,a0,da,hu,du,hv,dv,s,residual,ou,ov);
    if(!rank){
        for(int r=0;r<pr*pc;++r)std::cout<<"MEMORY mode="<<mode<<" rank="<<r<<" caller_backend_matrix_bytes="<<all_mem[5*r]<<" host_input_and_s_bytes="<<all_mem[5*r+4]<<" process_peak_rss_bytes="<<all_mem[5*r+1]<<" cuda_pool_peak_used_bytes="<<all_mem[5*r+2]<<" reference_workspace_bytes="<<all_mem[5*r+3]<<std::endl;
        const double mn=*std::min_element(times.begin(),times.end()),mx=*std::max_element(times.begin(),times.end());
        std::sort(times.begin(),times.end());double median=times[times.size()/2];if(times.size()%2==0)median=(median+times[times.size()/2-1])/2;
        std::cout<<std::setprecision(12)<<"RESULT mode="<<mode<<" backend="<<(cpu?"cpu":"gpu")<<" type="<<type<<" grid="<<pr<<'x'<<pc<<" m="<<m<<" n="<<n<<" nb="<<nb<<" reps="<<reps<<" median_s="<<median<<" min_s="<<mn<<" max_s="<<mx<<" residual_scaled="<<residual<<" orth_u_scaled="<<ou<<" orth_v_scaled="<<ov<<" s_max="<<s.front()<<" s_min="<<s.back()<<" check=PASS"<<std::endl;
        std::cout<<"SVD_COMPARE_PASS"<<std::endl;
    }
}

int positive(const char* s){std::size_t end=0;int v=std::stoi(s,&end);if(v<=0||end!=std::string(s).size())throw std::runtime_error("expected positive integer");return v;}
} // namespace

int main(int argc,char**argv)
{
    ddla_test_mpi_init(&argc,&argv);int rank=0,np=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&np);
    try{
        if(argc!=9)throw std::runtime_error("usage: benchmark_svd_compare cpu|gpu PRxPC M N NB REPEATS d|z ddla|lapack|scalapack|cusolver");
        const std::string backend=argv[1],grid=argv[2],type=argv[7],mode=argv[8];
        const auto sep=grid.find('x');if(sep==std::string::npos)throw std::runtime_error("grid must be PRxPC");
        const int pr=positive(grid.substr(0,sep).c_str()),pc=positive(grid.substr(sep+1).c_str());
        const int m=positive(argv[3]),n=positive(argv[4]),nb=positive(argv[5]),reps=positive(argv[6]);
        if(pr*pc!=np)throw std::runtime_error("grid size does not match MPI ranks");
        if(backend!="cpu"&&backend!="gpu")throw std::runtime_error("backend must be cpu or gpu");
        if(type!="d"&&type!="z")throw std::runtime_error("type must be d or z");
        if(mode!="ddla"&&mode!="lapack"&&mode!="scalapack"&&mode!="cusolver")throw std::runtime_error("unknown SVD mode");
        if(mode=="lapack"&&(backend!="cpu"||np!=1))throw std::runtime_error("lapack requires one CPU rank");
        if(mode=="scalapack"&&backend!="cpu")throw std::runtime_error("scalapack requires CPU buffers");
        if(mode=="cusolver"&&(backend!="gpu"||np!=1||m<n))throw std::runtime_error("cusolver gesvd requires one GPU rank and M>=N");
#if !defined(DDLA_BENCH_SCALAPACK)
        if(mode=="scalapack")throw std::runtime_error("ScaLAPACK comparison was not compiled");
#endif
#if !defined(DDLA_USE_CUDA)
        if(mode=="cusolver")throw std::runtime_error("cuSOLVER comparison requires a CUDA build");
#endif
        DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,backend=="cpu"?DdlaBackend::CPU:DdlaBackend::GPU));
        DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,pr,pc,'R'));int blacs=-1;
#if defined(DDLA_BENCH_SCALAPACK)
        if(mode=="scalapack"){
            Cblacs_get(-1,0,&blacs);Cblacs_gridinit(&blacs,"R",pr,pc);
            int rr,cc,ir,ic;Cblacs_gridinfo(blacs,&rr,&cc,&ir,&ic);
            if(rr!=pr||cc!=pc||ir!=ddla_test::myprow(h)||ic!=ddla_test::mypcol(h))throw std::runtime_error("BLACS/DDLA grid mapping mismatch");
        }
#endif
        if(!rank)std::cout<<"BENCHMARK warmup=1 timing=completed_max_rank_wall includes=workspace_query_allocation_free excludes=input_reset_and_validation memory_rss=process_lifetime_high_water memory_cuda_pool=caller_plus_async_allocations_not_total_device"<<std::endl;
        if(type=="d")run<double>(h,pr,pc,m,n,nb,reps,mode,type,blacs);
        else run<std::complex<double>>(h,pr,pc,m,n,nb,reps,mode,type,blacs);
#if defined(DDLA_BENCH_SCALAPACK)
        if(blacs>=0)Cblacs_gridexit(blacs);
#endif
        DDLA_CHECK(ddlaDestroy(h));MPI_Finalize();return 0;
    }catch(const std::exception& e){std::cerr<<"rank="<<rank<<" ERROR "<<e.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
