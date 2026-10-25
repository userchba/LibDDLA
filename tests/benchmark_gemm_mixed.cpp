#include <ddla/ddla.h>
#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using namespace ddla;
namespace {
void check(ddlaStatus_t s) { if(int(s)) throw std::runtime_error("DDLA runtime failure"); }
template <typename T> struct Buffer {
    DdlaHandle_t h; T* p=nullptr;
    Buffer(DdlaHandle_t handle,const std::vector<T>& v):h(handle) {
        check(ddlaMalloc(h,&p,v.size()*sizeof(T)));
        check(ddlaMemcpy(h,p,v.data(),v.size()*sizeof(T)));
    }
    ~Buffer() { ddlaFree(h,p); }
};
template <DdlaBackend BE,typename R>
void bench(DdlaHandle_t h,int m,int n,int k,char ta,char tb,int reps) {
    using Z=std::complex<R>;
    const int lda=ta=='N'?m:k,ldb=tb=='N'?k:n;
    const std::size_t na=std::size_t(m)*k,nb=std::size_t(k)*n,nc=std::size_t(m)*n;
    std::vector<R> a(na); std::vector<Z> ac(na),b(nb),c(nc),ref(nc);
    for (std::size_t p=0;p<na;++p) { a[p]=R(int((p*31+17)%97)-48)/R(53);ac[p]=Z(a[p],0); }
    for (std::size_t p=0;p<nb;++p) b[p]=Z(R(int((p*19+3)%89)-44)/R(47),R(int((p*23+5)%83)-41)/R(43));
    Buffer<R> da(h,a); Buffer<Z> dac(h,ac),db(h,b),dc(h,c),dr(h,ref);
    const Z alpha(1),beta(0);
    auto mixed=[&]{gemm<BE,R>(h,ta,tb,m,n,k,alpha,da.p,lda,db.p,ldb,beta,dc.p,m);};
    auto full=[&]{gemm<BE,Z>(h,ta,tb,m,n,k,alpha,dac.p,lda,db.p,ldb,beta,dr.p,m);};
    // beta=0 overwrites C, so each repetition has identical inputs. Host/device
    // input copies and preconversion of A for the complex baseline are untimed.
    // API wall time includes packing, combine, allocation, free and completion.
    check(ddlaSynchronize(h));
    for(int r=0;r<2;++r){mixed();full();check(ddlaSynchronize(h));}
    auto timed=[&](auto&& fn){check(ddlaSynchronize(h));auto start=std::chrono::steady_clock::now();fn();check(ddlaSynchronize(h));return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();};
    std::vector<double> tm,tf;
    for(int r=0;r<reps;++r) {
        if(r%2){tf.push_back(timed(full));tm.push_back(timed(mixed));}
        else{tm.push_back(timed(mixed));tf.push_back(timed(full));}
    }
    check(ddlaMemcpy(h,c.data(),dc.p,nc*sizeof(Z)));
    check(ddlaMemcpy(h,ref.data(),dr.p,nc*sizeof(Z)));check(ddlaSynchronize(h));
    long double err=0,scale=0;
    for(std::size_t p=0;p<nc;++p) {
        const double d=std::abs(c[p]-ref[p]);
        if(!std::isfinite(d)) throw std::runtime_error("non-finite benchmark result");
        err=std::max(err,(long double)d);scale=std::max(scale,(long double)std::abs(ref[p]));
    }
    const long double relative=err/std::max(1.L,scale);
    if(relative>32*std::numeric_limits<R>::epsilon()*std::sqrt((long double)k))
        throw std::runtime_error("benchmark correctness check failed");
    // Preserve every sample before sorting; emit after the timed work.
    for (int r=0;r<reps;++r)
        std::cerr<<std::setprecision(12)<<"sample,"<<(BE==DdlaBackend::CPU?"CPU":"GPU")<<','
            <<(sizeof(R)==8?"double":"float")<<','<<ta<<tb<<','<<m<<','<<n<<','<<k<<','
            <<r<<','<<tm[r]<<','<<tf[r]<<'\n';
    std::sort(tm.begin(),tm.end());std::sort(tf.begin(),tf.end());
    const double mixed_ms=tm[reps/2],complex_ms=tf[reps/2];
    std::cout<<(BE==DdlaBackend::CPU?"CPU":"GPU")<<','<<(sizeof(R)==8?"double":"float")<<','<<ta<<tb<<','<<m<<','<<n<<','<<k<<','
        <<mixed_ms<<','<<complex_ms<<','<<complex_ms/mixed_ms<<','<<(double)relative<<','
        <<tm.front()<<','<<tm.back()<<','<<tf.front()<<','<<tf.back()<<','<<reps<<','<<2<<'\n'<<std::flush;
}
template <DdlaBackend BE> void run(int argc,char**argv) {
    DdlaHandle_t h=nullptr;check(ddlaInit(h,BE));check(ddlaSet(h,MPI_COMM_WORLD));
    const int reps=argc>7?std::stoi(argv[7]):7;
    if(reps<1)throw std::invalid_argument("reps must be positive");
    if(argc>4) {
        const int m=std::stoi(argv[2]),n=std::stoi(argv[3]),k=std::stoi(argv[4]);
        const char ta=argc>5?argv[5][0]:'N',tb=argc>6?argv[6][0]:'N';
        if(m<1||n<1||k<1)throw std::invalid_argument("dimensions must be positive");
        bench<BE,float>(h,m,n,k,ta,tb,reps);bench<BE,double>(h,m,n,k,ta,tb,reps);
    } else for(int n:{64,256,1024,2048,4096}) {
        bench<BE,float>(h,n,n,n,'N','N',reps);bench<BE,double>(h,n,n,n,'N','N',reps);
    }
    check(ddlaDestroy(h));
}
}
int main(int argc,char**argv) {
    MPI_Init(&argc,&argv);int fail=0;
    try {
        int size=0;MPI_Comm_size(MPI_COMM_WORLD,&size);
        if(size!=1)throw std::invalid_argument("benchmark requires one MPI process");
        const std::string mode=argc>1?argv[1]:"gpu";
        std::cout<<std::setprecision(9)<<"backend,precision,trans,m,n,k,mixed_ms,complex_ms,speedup,max_relative_error,mixed_min_ms,mixed_max_ms,complex_min_ms,complex_max_ms,repeats,warmups\n";
#if DDLA_HAS_CPU
        if(mode=="cpu")run<DdlaBackend::CPU>(argc,argv);else
#endif
#if DDLA_HAS_GPU
        if(mode=="gpu")run<DdlaBackend::GPU>(argc,argv);else
#endif
        throw std::invalid_argument("requested backend unavailable");
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';fail=1;}
    MPI_Finalize();return fail;
}
