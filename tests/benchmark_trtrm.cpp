#include "mpi_test_init.h"
#include "trtrm_test_common.h"
#include <chrono>
#include <iomanip>
#if DDLA_HAS_GPU
#include "ddla_connector.h"
#endif
using namespace trtrm_test;

template <DdlaBackend Backend, typename F>
double timing(DdlaHandle_t h, F f, int reps)
{
    DDLA_CHECK(ddlaSynchronize(h));
    if constexpr (Backend == DdlaBackend::GPU) {
#if DDLA_HAS_GPU
        gpuEvent_t start, stop;
        GPU_CHECK(gpuEventCreate(&start)); GPU_CHECK(gpuEventCreate(&stop));
        const auto stream = reinterpret_cast<gpuStream_t>(ddlaGetStream(h));
        GPU_CHECK(gpuEventRecord(start, stream));
        for (int i = 0; i < reps; ++i) f();
        GPU_CHECK(gpuEventRecord(stop, stream)); GPU_CHECK(gpuEventSynchronize(stop));
        float ms = 0;
#if defined(DDLA_USE_CUDA)
        GPU_CHECK(cudaEventElapsedTime(&ms, start, stop));
#else
        GPU_CHECK(hipEventElapsedTime(&ms, start, stop));
#endif
        GPU_CHECK(gpuEventDestroy(start)); GPU_CHECK(gpuEventDestroy(stop));
        return ms / reps;
#endif
    }
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) f();
    DDLA_CHECK(ddlaSynchronize(h));
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/reps;
}

template <DdlaBackend Backend, typename T>
void bench(DdlaHandle_t h, int n, char uplo)
{
    const bool upper = uplo == 'U';
    const int ld = n;
    const std::size_t count = std::size_t(n) * n;
    std::vector<T> a(count, T(0)), b(count, T(0));
    for (int j=0; j<n; ++j) for (int i=0; i<n; ++i) if(upper ? i<=j : i>=j) {
        a[i+std::size_t(j)*ld] = data<T>(i,j,1);
        b[i+std::size_t(j)*ld] = data<T>(i,j,2);
    }
    Buffer<T> da(h,count), db(h,count), dc(h,count), dg(h,count);
    da.put(a); db.put(b);
    auto tri = [&] { trtrm<T>(h,uplo,'N',uplo,'N',n,T(1),da.p,ld,db.p,ld,T(0),dc.p,ld); };
    auto dense = [&] { gemm<T>(h,'N','N',n,n,n,T(1),da.p,ld,db.p,ld,T(0),dg.p,ld); };
    for (int warm=0; warm<3; ++warm) { tri(); dense(); }
    const auto got=dc.get(), ref=dg.get();
    long double e2=0, r2=0;
    for (int j=0;j<n;++j) for(int i=0;i<n;++i) if(upper ? i<=j : i>=j) {
        const auto p=i+std::size_t(j)*ld;
        const double e=std::abs(got[p]-ref[p]), r=std::abs(ref[p]);
        if (!std::isfinite(e) || !std::isfinite(r)) throw std::runtime_error("nonfinite benchmark result");
        e2+=e*e; r2+=r*r;
    }
    const double rel=std::sqrt(e2/std::max(1.0L,r2));
    if (rel > 64*std::numeric_limits<typename Scalar<T>::real>::epsilon()) throw std::runtime_error("benchmark correctness failed");
    const int reps = n <= 512 ? 50 : n <= 2048 ? 10 : 3;
    std::vector<double> t,g;
    for(int round=0;round<5;++round) {
        if(round%2) { g.push_back(timing<Backend>(h,dense,reps)); t.push_back(timing<Backend>(h,tri,reps)); }
        else { t.push_back(timing<Backend>(h,tri,reps)); g.push_back(timing<Backend>(h,dense,reps)); }
    }
    std::sort(t.begin(),t.end()); std::sort(g.begin(),g.end());
    const double factor = std::is_arithmetic_v<T> ? 2.0 : 8.0;
    const double tri_flops=factor*double(n)*(n+1.0)*(n+2.0)/6.0;
    std::cout<<std::setprecision(9)<<"RESULT,"<<uplo<<','<<Scalar<T>::name<<','<<n<<','<<t[2]<<','<<g[2]<<','<<g[2]/t[2]<<','<<rel<<','
             <<tri_flops/t[2]/1e6<<','<<factor*double(n)*n*n/g[2]/1e6<<','<<t.front()<<','<<t.back()<<','<<g.front()<<','<<g.back()<<std::endl;
}
int main(int argc,char** argv)
{
    ddla_test_mpi_init(&argc,&argv); int bad=0, np=0; MPI_Comm_size(MPI_COMM_WORLD,&np);
    try {
        if(np!=1) throw std::runtime_error("benchmark_trtrm requires one MPI process");
        constexpr auto Backend=default_backend_v;
        DdlaHandle_t h=nullptr; DDLA_CHECK(ddlaInit(h,Backend)); DDLA_CHECK(ddlaSet(h,MPI_COMM_SELF));
        std::cout<<"kind,uplo,type,n,trtrm_ms,gemm_ms,speedup,relative_error,trtrm_gflops,gemm_gflops,trtrm_min,trtrm_max,gemm_min,gemm_max"<<std::endl;
        std::string uplos="U";
        int arg=1;
        if(argc>arg && std::string(argv[arg])=="--uplo") {
            if(++arg>=argc) throw std::invalid_argument("--uplo requires U, L or both");
            const std::string value=argv[arg++];
            if(value=="U" || value=="u") uplos="U";
            else if(value=="L" || value=="l") uplos="L";
            else if(value=="both") uplos="UL";
            else throw std::invalid_argument("--uplo requires U, L or both");
        }
        const std::string types=argc>arg?argv[arg++]:"sdcz";
        std::vector<int> sizes={128,512,1024,2048,4096,8192};
        if(argc>arg) {sizes.clear(); for(int i=arg;i<argc;++i) {int n=std::stoi(argv[i]); if(n<=0) throw std::invalid_argument("positive n required"); sizes.push_back(n);} }
        for(char uplo:uplos) for(int n:sizes) {
            if(types.find('s')!=std::string::npos) bench<Backend,float>(h,n,uplo);
            if(types.find('d')!=std::string::npos) bench<Backend,double>(h,n,uplo);
            if(types.find('c')!=std::string::npos) bench<Backend,std::complex<float>>(h,n,uplo);
            if(types.find('z')!=std::string::npos) bench<Backend,std::complex<double>>(h,n,uplo);
        }
        DDLA_CHECK(ddlaDestroy(h));
    } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl; bad=1;}
    MPI_Finalize(); return bad;
}
