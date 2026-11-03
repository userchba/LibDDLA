#include "pgemm_test_common.h"
#include <iomanip>

const char* name(Algorithm a) {
    return a==Algorithm::AB?"AB":a==Algorithm::AC?"AC":a==Algorithm::BC?"BC":"auto";
}

template <DdlaBackend Backend, typename T>
void benchmark(DdlaHandle_t h,int m,int n,int k,char ta,char tb,int nb,int reps) {
    Matrix<T> a(h,ta=='N'?m:k,ta=='N'?k:m,nb,nb,0,0,1,false);
    Matrix<T> b(h,tb=='N'?k:n,tb=='N'?n:k,nb,nb,0,0,2,false);
    Matrix<T> c(h,m,n,nb,nb,0,0,3,false);
    const T alpha=scalar<T>(0.7,-0.2),beta=scalar<T>(-0.3,0.15);
    int pr,pc; ddlaGetGridDims(h,pr,pc);
    const auto selected=pgemm_impl::select(ta,tb,m,n,k,a.d,b.d,c.d,pr,pc);
    std::vector<T> reference;
    // AB is an additional cross-check here. The small regression suite uses
    // an independent dense oracle for every branch and scalar type.
    for(auto alg : {Algorithm::AB,Algorithm::AC,Algorithm::BC,Algorithm::Auto}) {
        c.reset();
        pgemm_impl::run<Backend>(h,ta,tb,m,n,k,alpha,a.data,a.d,b.data,b.d,beta,c.data,c.d,alg);
        auto out=c.download();
        if(reference.empty()) reference=out;
        double error=0;
        for(std::size_t i=0;i<out.size();++i) {
            const double e=std::abs(wide(out[i])-wide(reference[i]))/(1+std::abs(wide(reference[i])));
            if(!std::isfinite(e)) throw std::runtime_error("nonfinite benchmark result");
            error=std::max(error,e);
        }
        double worst=0;
        MPI_Allreduce(&error,&worst,1,MPI_DOUBLE,MPI_MAX,ddlaGetCommunicator(h));
        if(worst>2e-10) throw std::runtime_error("benchmark branch disagrees with AB");
        std::vector<double> times;
        // Untimed warm-up above; C restored before every measured call.
        for(int rep=0;rep<reps;++rep) {
            c.reset(); DDLA_CHECK(ddlaSynchronize(h)); MPI_Barrier(ddlaGetCommunicator(h));
            const double t0=MPI_Wtime();
            pgemm_impl::run<Backend>(h,ta,tb,m,n,k,alpha,a.data,a.d,b.data,b.d,beta,c.data,c.d,alg);
            DDLA_CHECK(ddlaSynchronize(h));
            const double local=MPI_Wtime()-t0; double elapsed=0;
            MPI_Allreduce(&local,&elapsed,1,MPI_DOUBLE,MPI_MAX,ddlaGetCommunicator(h));
            times.push_back(elapsed);
        }
        std::sort(times.begin(),times.end());
        if(ddlaGetRank(h)==0) std::cout << "RESULT," << int(Backend) << ','
            << (std::is_same_v<T,double>?"d":"z") << ',' << pr << ',' << pc << ',' << m << ',' << n << ',' << k
            << ',' << ta << tb << ',' << nb << ',' << name(alg) << ',' << name(selected)
            << ',' << std::setprecision(10) << times[times.size()/2] << ',' << worst << std::endl;
    }
}

template <DdlaBackend Backend>
void run(int pr,int pc,int m,int n,int k,char ta,char tb,int nb,int reps,char type) {
    DdlaHandle_t h=nullptr; DDLA_CHECK(ddlaInit(h,Backend)); DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,pr,pc));
    if(type=='d') benchmark<Backend,double>(h,m,n,k,ta,tb,nb,reps);
    else benchmark<Backend,std::complex<double>>(h,m,n,k,ta,tb,nb,reps);
    DDLA_CHECK(ddlaDestroy(h));
}

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);
    try {
        if(argc!=12) throw std::runtime_error("usage: benchmark_pgemm_branches cpu|gpu pr pc m n k ta tb nb repeats d|z");
        const int pr=std::stoi(argv[2]),pc=std::stoi(argv[3]),m=std::stoi(argv[4]),n=std::stoi(argv[5]),k=std::stoi(argv[6]);
        const char ta=argv[7][0],tb=argv[8][0],type=argv[11][0];
        const int nb=std::stoi(argv[9]),reps=std::stoi(argv[10]);
        int np;MPI_Comm_size(MPI_COMM_WORLD,&np);
        if(pr<1 || pc<1 || pr*pc!=np || m<1 || n<1 || k<1 || nb<1 || reps<1 ||
           std::string("NTC").find(ta)==std::string::npos || std::string("NTC").find(tb)==std::string::npos ||
           (type!='d' && type!='z')) throw std::runtime_error("invalid benchmark argument");
        if(std::string(argv[1])=="cpu") {
            run<DdlaBackend::CPU>(pr,pc,m,n,k,ta,tb,nb,reps,type);
        } else if(std::string(argv[1])=="gpu") {
#if DDLA_HAS_GPU
            run<DdlaBackend::GPU>(pr,pc,m,n,k,ta,tb,nb,reps,type);
#else
            throw std::runtime_error("GPU backend unavailable");
#endif
        } else throw std::runtime_error("invalid backend");
        MPI_Finalize();return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);return 1; }
}
