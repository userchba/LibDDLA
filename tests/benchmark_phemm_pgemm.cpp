#include <ddla/ddla.h>
#include "PBLAS/pgemm_impl.h"
#include "mpi_test_init.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace ddla;
namespace {
template <typename T> T scalar(double re,double im=0) {
    if constexpr(std::is_arithmetic_v<T>) return T(re);
    else return T(re,im);
}
template <typename T> std::complex<double> wide(T x) {
    if constexpr(std::is_arithmetic_v<T>) return {double(x),0};
    else return {double(x.real()),double(x.imag())};
}
template <typename T> T sample(int i,int j,int salt) {
    const std::uint64_t a=(std::uint64_t(i)+1)*104729+(std::uint64_t(j)+3)*13007+salt*8191;
    return scalar<T>(double(int((a^(a>>7))%997)-498)/499.,
                     double(int(((a*65537)^(a>>3))%991)-495)/497.);
}
template <typename T> T hermitian(int i,int j) {
    const auto x=wide(sample<T>(std::min(i,j),std::max(i,j),1));
    return scalar<T>(x.real(),i==j ? 0 : (i<j ? x.imag() : -x.imag()));
}
template <typename T> const char* type_name() {
    if constexpr(std::is_same_v<T,float>) return "s";
    else if constexpr(std::is_same_v<T,double>) return "d";
    else if constexpr(std::is_same_v<T,std::complex<float>>) return "c";
    else return "z";
}
const char* algorithm_name(pgemm_impl::Algorithm a) {
    return a==pgemm_impl::Algorithm::AB ? "AB" : a==pgemm_impl::Algorithm::AC ? "AC" : "BC";
}

template <typename T> struct Matrix {
    DdlaHandle_t h;
    int d[DDLA_DLEN_],nr,nc,pr,pc,r,c;
    ddla::vector<T> data;
    Matrix(DdlaHandle_t handle,int m,int n,int nb):h(handle),data(h) {
        ddlaGetGridDims(h,pr,pc);ddlaGetGridCoords(h,r,c);
        nr=num_loc(m,nb,r,0,pr);nc=num_loc(n,nb,c,0,pc);
        DDLA_CHECK(ddlaDescInit(d,h,m,n,nb,nb,0,0));
        data.resize(std::max(std::size_t(1),std::size_t(d[DDLA_LLD_])*nc));
    }
    int gi(int i) const {return indxl2g(i,d[DDLA_MB_],r,0,pr);}
    int gj(int j) const {return indxl2g(j,d[DDLA_NB_],c,0,pc);}
    template <typename F> void fill(F value) {
        std::vector<T> host(data.size(),T(0));
        for(int j=0;j<nc;++j) for(int i=0;i<nr;++i)
            host[i+std::size_t(j)*d[DDLA_LLD_]]=value(gi(i),gj(j));
        DDLA_CHECK(ddlaMemcpy(h,data.data(),host.data(),host.size()*sizeof(T)));
        DDLA_CHECK(ddlaDeviceSynchronize(h));
    }
    void reset(const Matrix& initial) {
        DDLA_CHECK(ddlaMemcpy(h,data.data(),initial.data.data(),data.size()*sizeof(T)));
    }
    std::vector<T> download() const {
        std::vector<T> host(data.size());
        DDLA_CHECK(ddlaMemcpy(h,host.data(),data.data(),host.size()*sizeof(T)));
        DDLA_CHECK(ddlaDeviceSynchronize(h));return host;
    }
};

template <typename T>
double compare(const Matrix<T>& ph,const Matrix<T>& pg,char side,bool oracle,T alpha,T beta) {
    const auto a=ph.download(),b=pg.download();long double local[2]={},global[2]={};
    int bad=0,any=0;
    const int q=side=='L' ? ph.d[DDLA_M_] : ph.d[DDLA_N_];
    for(int j=0;j<ph.nc;++j) for(int i=0;i<ph.nr;++i) {
        const auto p=i+std::size_t(j)*ph.d[DDLA_LLD_];
        auto ref=wide(b[p]);
        if(oracle) {
            const int x=ph.gi(i),y=ph.gj(j);ref=0;
            for(int k=0;k<q;++k)
                ref+=side=='L' ? wide(hermitian<T>(x,k))*wide(sample<T>(k,y,2))
                               : wide(sample<T>(x,k,2))*wide(hermitian<T>(k,y));
            ref=wide(alpha)*ref+wide(beta)*wide(sample<T>(x,y,3));
        }
        const double err=std::max(std::abs(wide(a[p])-ref),std::abs(wide(b[p])-ref));
        if(!std::isfinite(err) || !std::isfinite(std::abs(ref))) bad=1;
        else {local[0]+=static_cast<long double>(err)*err;local[1]+=std::norm(ref);}
    }
    MPI_Allreduce(local,global,2,MPI_LONG_DOUBLE,MPI_SUM,ddlaGetCommunicator(ph.h));
    MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(ph.h));
    const double error=std::sqrt(global[0]/std::max(1.L,global[1]));
    constexpr bool single=std::is_same_v<T,float> || std::is_same_v<T,std::complex<float>>;
    if(any || error>(single ? 5e-5 : 5e-12)) throw std::runtime_error("phemm/pgemm benchmark result mismatch");
    return error;
}

double median(std::vector<double> x) {
    std::sort(x.begin(),x.end());return (x[(x.size()-1)/2]+x[x.size()/2])/2;
}

template <DdlaBackend Backend,typename T>
void benchmark(DdlaHandle_t h,int m,int n,int nb,int reps,char side,char uplo,int ordinal) {
    const int q=side=='L' ? m : n;
    Matrix<T> full(h,q,q,nb),tri(h,q,q,nb),b(h,m,n,nb),initial(h,m,n,nb),ph(h,m,n,nb),pg(h,m,n,nb);
    full.fill([](int i,int j){return hermitian<T>(i,j);});
    tri.fill([&](int i,int j){
        const double nan=std::numeric_limits<double>::quiet_NaN();
        if(i==j) return scalar<T>(wide(hermitian<T>(i,j)).real(),nan);
        return (uplo=='U' ? i<j : i>j) ? hermitian<T>(i,j) : scalar<T>(nan,nan);
    });
    b.fill([](int i,int j){return sample<T>(i,j,2);});
    initial.fill([](int i,int j){return sample<T>(i,j,3);});
    const T alpha=scalar<T>(.7,-.2),beta=scalar<T>(-.3,.15);
    auto hemm=[&]{phemm<Backend,T>(h,side,uplo,m,n,alpha,tri.data.data(),tri.d,b.data.data(),b.d,beta,ph.data.data(),ph.d);};
    auto gemm=[&]{pgemm<Backend,T>(h,'N','N',m,n,q,alpha,
        side=='L' ? full.data.data() : b.data.data(),side=='L' ? full.d : b.d,
        side=='L' ? b.data.data() : full.data.data(),side=='L' ? b.d : full.d,beta,pg.data.data(),pg.d);};
    // Two untimed calls per routine; validate before any timing.
    for(int w=0;w<2;++w) {
        ph.reset(initial);pg.reset(initial);hemm();gemm();DDLA_CHECK(ddlaDeviceSynchronize(h));
    }
    const bool small=m<=64 && n<=64;
    double error=compare(ph,pg,side,small,alpha,beta);
    auto timed=[&](bool use_hemm) {
        (use_hemm ? ph : pg).reset(initial);
        DDLA_CHECK(ddlaDeviceSynchronize(h));MPI_Barrier(ddlaGetCommunicator(h));
        const double start=MPI_Wtime();
        if(use_hemm) hemm();else gemm();
        DDLA_CHECK(ddlaDeviceSynchronize(h));
        const double local=MPI_Wtime()-start;double elapsed=0;
        MPI_Allreduce(&local,&elapsed,1,MPI_DOUBLE,MPI_MAX,ddlaGetCommunicator(h));return elapsed;
    };
    int pr,pc;ddlaGetGridDims(h,pr,pc);
    std::vector<double> ht,gt;
    for(int rep=0;rep<reps;++rep) {
        const bool hemm_first=(rep+ordinal)%2;
        double a,btime;
        if(hemm_first){a=timed(true);btime=timed(false);}else{btime=timed(false);a=timed(true);}
        ht.push_back(a);gt.push_back(btime);
        if(ddlaGetRank(h)==0) std::cout<<"RAW,"<<int(Backend)<<','<<type_name<T>()<<','<<pr<<','<<pc<<','<<m<<','<<n<<','<<q<<','<<side<<','<<uplo<<','<<nb<<','<<rep<<','<<(hemm_first ? "phemm" : "pgemm")<<','<<std::setprecision(12)<<a<<','<<btime<<std::endl;
    }
    error=std::max(error,compare(ph,pg,side,small,alpha,beta));
    const auto algorithm=pgemm_impl::select('N','N',m,n,q,side=='L' ? full.d : b.d,side=='L' ? b.d : full.d,pg.d,pr,pc);
    const double hmed=median(ht),gmed=median(gt);
    const double flops=(std::is_arithmetic_v<T> ? 2. : 8.)*m*n*q;
    if(ddlaGetRank(h)==0) std::cout<<"RESULT,"<<int(Backend)<<','<<type_name<T>()<<','<<pr<<','<<pc<<','<<m<<','<<n<<','<<q<<','<<side<<','<<uplo<<','<<nb<<','<<reps<<','<<algorithm_name(algorithm)<<','<<std::setprecision(12)
        <<hmed<<','<<gmed<<','<<hmed/gmed<<','<<*std::min_element(ht.begin(),ht.end())<<','<<*std::max_element(ht.begin(),ht.end())
        <<','<<*std::min_element(gt.begin(),gt.end())<<','<<*std::max_element(gt.begin(),gt.end())<<','<<error<<','<<flops/hmed/1e9<<','<<flops/gmed/1e9<<std::endl;
}

template <DdlaBackend Backend>
void run(int pr,int pc,int nb,int reps,bool smoke) {
    DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,Backend));DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,pr,pc,'R'));
    const std::vector<std::pair<int,int>> shapes=smoke ? std::vector<std::pair<int,int>>{{31,23},{23,31}}
        : std::vector<std::pair<int,int>>{{1024,1024},{4096,4096},{8192,8192},{8192,1024},{1024,8192}};
    int ordinal=0;
    for(auto shape:shapes) for(char side:{'L','R'}) for(char uplo:{'U','L'}) {
        benchmark<Backend,float>(h,shape.first,shape.second,nb,reps,side,uplo,ordinal++);
        benchmark<Backend,double>(h,shape.first,shape.second,nb,reps,side,uplo,ordinal++);
        benchmark<Backend,std::complex<float>>(h,shape.first,shape.second,nb,reps,side,uplo,ordinal++);
        benchmark<Backend,std::complex<double>>(h,shape.first,shape.second,nb,reps,side,uplo,ordinal++);
    }
    if(ddlaGetRank(h)==0) std::cout<<"PHEMM_PGEMM_BENCH_PASS grid="<<pr<<'x'<<pc<<" cases="<<ordinal<<std::endl;
    DDLA_CHECK(ddlaDestroy(h));
}
} // namespace

int main(int argc,char** argv) {
    ddla_test_mpi_init(&argc,&argv);
    try {
        if(argc!=7) throw std::runtime_error("usage: benchmark_phemm_pgemm cpu|gpu pr pc nb repeats smoke|standard");
        const int pr=std::stoi(argv[2]),pc=std::stoi(argv[3]),nb=std::stoi(argv[4]),reps=std::stoi(argv[5]);
        int size;MPI_Comm_size(MPI_COMM_WORLD,&size);
        const std::string backend=argv[1],mode=argv[6];
        if(pr<=0 || pc<=0 || pr*pc!=size || nb<=0 || reps<=0 || (mode!="smoke" && mode!="standard")) throw std::runtime_error("invalid benchmark arguments");
        if(backend=="gpu") {
#if DDLA_HAS_GPU
            run<DdlaBackend::GPU>(pr,pc,nb,reps,mode=="smoke");
#else
            throw std::runtime_error("GPU backend unavailable");
#endif
        } else if(backend=="cpu") {
            run<DdlaBackend::CPU>(pr,pc,nb,reps,mode=="smoke");
        } else throw std::runtime_error("expected cpu or gpu");
        MPI_Finalize();return 0;
    } catch(const std::exception& e) {
        std::cerr<<"PHEMM_PGEMM_BENCH_FAILURE "<<e.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);return 1;
    }
}
