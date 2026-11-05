#include "ddla_connector.h"
#include "pblas_structured_test_common.h"
#include "ddla_stream.h"
#include <mpi.h>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#if defined(DDLA_USE_CUDA)
#include <cublas_v2.h>
#endif

#if defined(DDLA_BENCH_SCALAPACK)
extern "C" {
void Cblacs_get(int,int,int*);
void Cblacs_gridinit(int*,const char*,int,int);
void Cblacs_gridinfo(int,int*,int*,int*,int*);
void Cblacs_gridexit(int);
void descinit_(int*,const int*,const int*,const int*,const int*,const int*,const int*,const int*,const int*,int*);
#define DECLARE_PSYMM(P,T) void P(const char*,const char*,const int*,const int*,const T*,const T*,const int*,const int*,const int*,const T*,const int*,const int*,const int*,const T*,T*,const int*,const int*,const int*)
#define DECLARE_PSYRK(P,T,S) void P(const char*,const char*,const int*,const int*,const S*,const T*,const int*,const int*,const int*,const S*,T*,const int*,const int*,const int*)
#define DECLARE_PSYR2K(P,T,S) void P(const char*,const char*,const int*,const int*,const T*,const T*,const int*,const int*,const int*,const T*,const int*,const int*,const int*,const S*,T*,const int*,const int*,const int*)
DECLARE_PSYMM(pdsymm_,double);
DECLARE_PSYMM(pzsymm_,std::complex<double>);
DECLARE_PSYRK(pdsyrk_,double,double);
DECLARE_PSYRK(pzsyrk_,std::complex<double>,std::complex<double>);
DECLARE_PSYRK(pzherk_,std::complex<double>,double);
DECLARE_PSYR2K(pdsyr2k_,double,double);
DECLARE_PSYR2K(pzsyr2k_,std::complex<double>,std::complex<double>);
DECLARE_PSYR2K(pzher2k_,std::complex<double>,double);
#undef DECLARE_PSYMM
#undef DECLARE_PSYRK
#undef DECLARE_PSYR2K
}
#endif

using namespace ddla;
using namespace structured_test;
namespace {
struct BlacsGrid {
    int context=-1;
    BlacsGrid(DdlaHandle_t h,bool enabled) {
#if defined(DDLA_BENCH_SCALAPACK)
        if(enabled) {
            int pr,pc,r,c;ddlaGetGridDims(h,pr,pc);ddlaGetGridCoords(h,r,c);
            Cblacs_get(-1,0,&context);Cblacs_gridinit(&context,"R",pr,pc);
            int br,bc,mr,mc;Cblacs_gridinfo(context,&br,&bc,&mr,&mc);
            if(pr!=br || pc!=bc || r!=mr || c!=mc)throw std::runtime_error("DDLA/BLACS grid mismatch");
        }
#else
        (void)h;(void)enabled;
#endif
    }
    ~BlacsGrid() {
#if defined(DDLA_BENCH_SCALAPACK)
        if(context>=0)Cblacs_gridexit(context);
#endif
    }
};
#if defined(DDLA_BENCH_SCALAPACK)
template <typename T>
void scalapack(Operation op,int n,int k,T alpha,const Matrix<T>& a,const Matrix<T>& b,T beta,
               T* c,const int* da,const int* db,const int* dc) {
    const int one=1;
    const char side='L',uplo='U',trans='N';
    if constexpr(std::is_same_v<T,double>) {
        if(op==Operation::Symm)pdsymm_(&side,&uplo,&n,&n,&alpha,a.host.data(),&one,&one,da,b.host.data(),&one,&one,db,&beta,c,&one,&one,dc);
        else if(op==Operation::Syrk)pdsyrk_(&uplo,&trans,&n,&k,&alpha,a.host.data(),&one,&one,da,&beta,c,&one,&one,dc);
        else if(op==Operation::Syr2k)pdsyr2k_(&uplo,&trans,&n,&k,&alpha,a.host.data(),&one,&one,da,b.host.data(),&one,&one,db,&beta,c,&one,&one,dc);
        else throw std::runtime_error("real Hermitian benchmark unsupported");
    } else {
        const double arho=alpha.real(),brho=beta.real();
        if(op==Operation::Symm)pzsymm_(&side,&uplo,&n,&n,&alpha,a.host.data(),&one,&one,da,b.host.data(),&one,&one,db,&beta,c,&one,&one,dc);
        else if(op==Operation::Syrk)pzsyrk_(&uplo,&trans,&n,&k,&alpha,a.host.data(),&one,&one,da,&beta,c,&one,&one,dc);
        else if(op==Operation::Syr2k)pzsyr2k_(&uplo,&trans,&n,&k,&alpha,a.host.data(),&one,&one,da,b.host.data(),&one,&one,db,&beta,c,&one,&one,dc);
        else if(op==Operation::Herk)pzherk_(&uplo,&trans,&n,&k,&arho,a.host.data(),&one,&one,da,&brho,c,&one,&one,dc);
        else pzher2k_(&uplo,&trans,&n,&k,&alpha,a.host.data(),&one,&one,da,b.host.data(),&one,&one,db,&brho,c,&one,&one,dc);
    }
}
#endif

#if defined(DDLA_USE_CUDA)
struct Cublas {
    cublasHandle_t h=nullptr;
    Cublas(DdlaHandle_t handle,bool enabled) {
        if(enabled) {
            if(cublasCreate(&h)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cublasCreate failed");
            if(cublasSetStream(h,static_cast<gpuStream_t>(handle->stream))!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cublasSetStream failed");
        }
    }
    ~Cublas(){if(h)cublasDestroy(h);}
};
template <typename T>
void cublas_call(cublasHandle_t handle,Operation op,int n,int k,T alpha,
    const Matrix<T>& a,const Matrix<T>& b,T beta,Matrix<T>& c) {
    const int lda=a.d[DDLA_LLD_],ldb=b.d[DDLA_LLD_],ldc=c.d[DDLA_LLD_];
    cublasStatus_t status=CUBLAS_STATUS_NOT_SUPPORTED;
    if constexpr(std::is_same_v<T,double>) {
        if(op==Operation::Symm)status=cublasDsymm(handle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,n,n,&alpha,a.data.data(),lda,b.data.data(),ldb,&beta,c.data.data(),ldc);
        else if(op==Operation::Syrk)status=cublasDsyrk(handle,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,n,k,&alpha,a.data.data(),lda,&beta,c.data.data(),ldc);
        else if(op==Operation::Syr2k)status=cublasDsyr2k(handle,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,n,k,&alpha,a.data.data(),lda,b.data.data(),ldb,&beta,c.data.data(),ldc);
    } else {
        static_assert(sizeof(T)==sizeof(cuDoubleComplex));
        const cuDoubleComplex ca=make_cuDoubleComplex(alpha.real(),alpha.imag()),cb=make_cuDoubleComplex(beta.real(),beta.imag());
        const auto ap=reinterpret_cast<const cuDoubleComplex*>(a.data.data()),bp=reinterpret_cast<const cuDoubleComplex*>(b.data.data());
        auto cp=reinterpret_cast<cuDoubleComplex*>(c.data.data());const double ra=alpha.real(),rb=beta.real();
        if(op==Operation::Symm)status=cublasZsymm(handle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,n,n,&ca,ap,lda,bp,ldb,&cb,cp,ldc);
        else if(op==Operation::Syrk)status=cublasZsyrk(handle,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,n,k,&ca,ap,lda,&cb,cp,ldc);
        else if(op==Operation::Syr2k)status=cublasZsyr2k(handle,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,n,k,&ca,ap,lda,bp,ldb,&cb,cp,ldc);
        else if(op==Operation::Herk)status=cublasZherk(handle,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,n,k,&ra,ap,lda,&rb,cp,ldc);
        else status=cublasZher2k(handle,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,n,k,&ca,ap,lda,bp,ldb,&rb,cp,ldc);
    }
    if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS operation failed: "+std::to_string(int(status)));
}
#endif

struct Error {double absolute=0,relative=0;};
template <typename T>
Error compare(Operation op,const Matrix<T>& c,const std::vector<T>& output,const std::vector<T>& reference) {
    long double local[2]={0,0},global[2];double maximum=0,global_max;int bad=0,any;
    for(int j=0;j<c.nc;++j)for(int i=0;i<c.nr;++i) {
        const auto p=i+std::size_t(j)*c.d[DDLA_LLD_];
        if(op==Operation::Symm || c.gi(i)<=c.gj(j)) {
            const double e=std::abs(wide(output[p])-wide(reference[p]));
            if(!std::isfinite(e) || !std::isfinite(std::abs(wide(reference[p]))))bad=1;
            else {maximum=std::max(maximum,e);local[0]+=static_cast<long double>(e)*e;local[1]+=std::norm(wide(reference[p]));}
            if(hermitian(op) && c.gi(i)==c.gj(j) && wide(output[p]).imag()!=0)bad=1;
        }else if(std::memcmp(&output[p],&c.host[p],sizeof(T)))bad=1;
    }
    MPI_Allreduce(local,global,2,MPI_LONG_DOUBLE,MPI_SUM,ddlaGetCommunicator(c.h));
    MPI_Allreduce(&maximum,&global_max,1,MPI_DOUBLE,MPI_MAX,ddlaGetCommunicator(c.h));
    MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(c.h));
    const double relative=std::sqrt(global[0]/std::max(1.L,global[1]));
    if(any || relative>1e-12)throw std::runtime_error("benchmark vendor accuracy/triangle check failed, relative="+std::to_string(relative));
    return {global_max,relative};
}

template <typename T>
Error sample_check(Operation op,const Matrix<T>& c,int n,int k,T alpha,T beta) {
    const auto output=c.download();double maxabs=0,maxrel=0;int bad=0,any;
    // Selected deterministic entries on every rank make the multi-GPU result
    // check independent of all DDLA distributed multiplication routines.
    for(int jj=0;jj<std::min(8,c.nc);++jj)for(int ii=0;ii<std::min(8,c.nr);++ii) {
        const int i=ii*(c.nr-1)/std::max(1,std::min(8,c.nr)-1),j=jj*(c.nc-1)/std::max(1,std::min(8,c.nc)-1);
        const int x=c.gi(i),y=c.gj(j);if(op!=Operation::Symm && x>y)continue;
        const auto ref=reference<T>(op,'L','U','N',x,y,n,n,k,alpha,beta);
        const auto p=i+std::size_t(j)*c.d[DDLA_LLD_];
        const double e=std::abs(wide(output[p])-ref.first),r=e/(1+ref.second);
        if(!std::isfinite(r))bad=1;else{maxabs=std::max(maxabs,e);maxrel=std::max(maxrel,r);}
    }
    double local[]={maxabs,maxrel},global[2];
    MPI_Allreduce(local,global,2,MPI_DOUBLE,MPI_MAX,ddlaGetCommunicator(c.h));
    MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(c.h));
    if(any || global[1]>1e-12)throw std::runtime_error("benchmark independent sampled oracle failed");
    return {global[0],global[1]};
}

double median(std::vector<double> v){std::sort(v.begin(),v.end());return (v[(v.size()-1)/2]+v[v.size()/2])/2;}

template <DdlaBackend Backend,typename T>
void benchmark(DdlaHandle_t h,Operation op,int n,int nb,int repetitions,int warmup,int context,
#if defined(DDLA_USE_CUDA)
    cublasHandle_t cuda_handle,
#endif
    int ordinal) {
    const int k=n;const bool herm=hermitian(op);int pr,pc;ddlaGetGridDims(h,pr,pc);
    const bool cuda_reference=Backend==DdlaBackend::GPU && pr*pc==1;
    const bool cpu_reference=Backend==DdlaBackend::CPU;
    const bool timed_reference=cpu_reference || cuda_reference;
    const char* backend=Backend==DdlaBackend::CPU ? "cpu" : "gpu";
    const char* reference_name=cpu_reference ? "scalapack" : cuda_reference ? "cublas" : "none";
    const double nan=std::numeric_limits<double>::quiet_NaN();
    const T alpha=scalar<T>(.7,op==Operation::Herk ? 0 : -.2),beta=scalar<T>(-.3,herm ? 0 : .15);
    Matrix<T> a(h,n,k,nb),b(h,n,k,nb),c(h,n,n,nb),vendor(h,n,n,nb);
    a.fill(n,k,[&](int i,int j){return op==Operation::Symm && i>j ? scalar<T>(nan,nan) : value<T>(i,j,1);});
    b.fill(n,k,[](int i,int j){return value<T>(i,j,2);});
    c.fill(n,n,[&](int i,int j){
        if(op!=Operation::Symm && i>j)return scalar<T>(-789.5,456.25);
        return herm && i==j ? scalar<T>(wide(value<T>(i,j,3)).real(),nan) : value<T>(i,j,3);
    });vendor.host=c.host;
    a.upload();b.upload();c.upload();vendor.upload();DDLA_CHECK(ddlaDeviceSynchronize(h));
    std::vector<T> cpu_output=c.host;
#if defined(DDLA_BENCH_SCALAPACK)
    int sca[9],scb[9],scc[9];
    if(context>=0) {
        const int zero=0;int info;
        for(auto pair:{std::make_pair(a.d,sca),std::make_pair(b.d,scb),std::make_pair(c.d,scc)}) {
            const int* d=pair.first;
            descinit_(pair.second,&d[DDLA_M_],&d[DDLA_N_],&nb,&nb,&zero,&zero,&context,&d[DDLA_LLD_],&info);
            if(info)throw std::runtime_error("ScaLAPACK descriptor failed");
        }
    }
#endif
    auto call_ddla=[&]{invoke<Backend,T>(h,op,'L','U','N',n,n,k,alpha,a.data.data(),a.d,b.data.data(),b.d,beta,c.data.data(),c.d);};
    auto call_vendor=[&] {
        if constexpr(Backend==DdlaBackend::CPU) {
#if defined(DDLA_BENCH_SCALAPACK)
            scalapack(op,n,k,alpha,a,b,beta,cpu_output.data(),sca,scb,scc);
#else
            throw std::runtime_error("CPU benchmark requires DDLA_BENCH_SCALAPACK");
#endif
        }else {
#if defined(DDLA_USE_CUDA)
            cublas_call(cuda_handle,op,n,k,alpha,a,b,beta,vendor);
#else
            throw std::runtime_error("CUDA reference unavailable");
#endif
        }
    };
    auto reset=[&](bool ddla) {
        if(ddla)c.upload();else if(cpu_reference)cpu_output=c.host;else vendor.upload();
    };
    for(int w=0;w<std::max(1,warmup);++w) {
        reset(true);call_ddla();
        if(timed_reference){reset(false);call_vendor();}
        DDLA_CHECK(ddlaDeviceSynchronize(h));
    }
    Error error=sample_check(op,c,n,k,alpha,beta);
    const char* accuracy_reference="sampled_scalar";
    if(timed_reference) {
        const auto ref=cpu_reference ? cpu_output : vendor.download();
        const auto e=compare(op,c,c.download(),ref);error.absolute=std::max(error.absolute,e.absolute);error.relative=std::max(error.relative,e.relative);
        accuracy_reference=reference_name;
    }
#if defined(DDLA_BENCH_SCALAPACK)
    else if(context>=0) {
        cpu_output=c.host;scalapack(op,n,k,alpha,a,b,beta,cpu_output.data(),sca,scb,scc);
        const auto e=compare(op,c,c.download(),cpu_output);error.absolute=std::max(error.absolute,e.absolute);error.relative=std::max(error.relative,e.relative);
        accuracy_reference="scalapack";
    }
#endif
    auto timed=[&](bool ddla) {
        reset(ddla);DDLA_CHECK(ddlaDeviceSynchronize(h));MPI_Barrier(ddlaGetCommunicator(h));
        const double start=MPI_Wtime();if(ddla)call_ddla();else call_vendor();
        DDLA_CHECK(ddlaDeviceSynchronize(h));const double local=MPI_Wtime()-start;double elapsed;
        MPI_Allreduce(&local,&elapsed,1,MPI_DOUBLE,MPI_MAX,ddlaGetCommunicator(h));return elapsed;
    };
    std::vector<double> dt,vt;
    const double factor=std::is_arithmetic_v<T> ? 1. : 4.;
    const double flops=factor*(op==Operation::Symm ? 2.*n*n*k : (two_inputs(op) ? 2. : 1.)*double(n)*(n+1)*k);
    auto print_raw=[&](const char* implementation,int rep,int position,double elapsed) {
        if(ddlaGetRank(h)==0)std::cout<<"RAW,"<<backend<<','<<name(op)<<','<<type_name<T>()<<','<<n<<','<<k<<','<<nb<<','<<pr<<','<<pc<<','<<warmup<<','<<repetitions<<','<<rep<<','<<position<<','<<implementation<<','<<std::setprecision(12)<<elapsed<<','<<flops/elapsed/1e9<<','<<error.absolute<<','<<error.relative<<','<<accuracy_reference<<std::endl;
    };
    for(int rep=0;rep<repetitions;++rep) {
        if(timed_reference && (rep+ordinal)%2) {
            const double v=timed(false);vt.push_back(v);print_raw(reference_name,rep,0,v);
            const double d=timed(true);dt.push_back(d);print_raw("ddla",rep,1,d);
        }else {
            const double d=timed(true);dt.push_back(d);print_raw("ddla",rep,0,d);
            if(timed_reference){const double v=timed(false);vt.push_back(v);print_raw(reference_name,rep,1,v);}
        }
    }
    sample_check(op,c,n,k,alpha,beta);
    if(timed_reference)compare(op,c,c.download(),cpu_reference ? cpu_output : vendor.download());
    if(ddlaGetRank(h)==0) {
        std::cout<<"RESULT,"<<backend<<','<<name(op)<<','<<type_name<T>()<<','<<n<<','<<k<<','<<nb<<','<<pr<<','<<pc<<','<<warmup<<','<<repetitions<<','<<std::setprecision(12)<<median(dt)<<','<<*std::min_element(dt.begin(),dt.end())<<','<<*std::max_element(dt.begin(),dt.end())<<','<<reference_name<<',';
        if(timed_reference)std::cout<<median(vt)<<','<<*std::min_element(vt.begin(),vt.end())<<','<<*std::max_element(vt.begin(),vt.end())<<','<<median(vt)/median(dt);
        else std::cout<<"NA,NA,NA,NA";
        std::cout<<','<<error.absolute<<','<<error.relative<<','<<accuracy_reference<<std::endl;
    }
}

template <DdlaBackend Backend>
void run(int pr,int pc,int nb,int repetitions,int warmup,const std::vector<int>& sizes) {
    DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,Backend));DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,pr,pc,'R'));
    {
        BlacsGrid blacs(h,Backend==DdlaBackend::CPU || pr*pc>1);
#if defined(DDLA_USE_CUDA)
        Cublas cuda(h,Backend==DdlaBackend::GPU && pr*pc==1);
#endif
        int ordinal=0;
        if(ddlaGetRank(h)==0) {
            std::cout<<"# RAW: backend,operation,type,n,k,nb,pr,pc,warmup,repeats,repeat,order,implementation,seconds,gflops,max_abs_error,max_rel_error,accuracy_reference\n";
            std::cout<<"# RESULT: backend,operation,type,n,k,nb,pr,pc,warmup,repeats,ddla_median_s,ddla_min_s,ddla_max_s,reference,reference_median_s,reference_min_s,reference_max_s,speedup,max_abs_error,max_rel_error,accuracy_reference\n";
            std::cout<<"# Timing: public call including allocation and communication; initialized inputs resident; output reset excluded; completion synchronized; MPI maximum wall time. Flags L,U,N; m=n=k.\n";
#if defined(DDLA_USE_CUDA)
            if constexpr(Backend==DdlaBackend::GPU){int version=0;cublasHandle_t probe=nullptr;cublasCreate(&probe);cublasGetVersion(probe,&version);cublasDestroy(probe);std::cout<<"# cuBLAS version="<<version<<'\n';}
#endif
        }
        for(int n:sizes) {
            for(auto op:{Operation::Symm,Operation::Syrk,Operation::Syr2k})benchmark<Backend,double>(h,op,n,nb,repetitions,warmup,blacs.context,
#if defined(DDLA_USE_CUDA)
                cuda.h,
#endif
                ordinal++);
            for(auto op:{Operation::Symm,Operation::Syrk,Operation::Syr2k,Operation::Herk,Operation::Her2k})benchmark<Backend,std::complex<double>>(h,op,n,nb,repetitions,warmup,blacs.context,
#if defined(DDLA_USE_CUDA)
                cuda.h,
#endif
                ordinal++);
        }
        if(ddlaGetRank(h)==0)std::cout<<"STRUCTURED_PBLAS_BENCH_PASS cases="<<ordinal<<" grid="<<pr<<'x'<<pc<<std::endl;
    }
    DDLA_CHECK(ddlaDestroy(h));
}
}
int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);
    try {
        if(argc<6 || argc>8)throw std::runtime_error("usage: benchmark_pblas_structured cpu|gpu PR PC NB REPEATS [SIZES_CSV=1024,2048,4096] [WARMUP=2]");
        int size;MPI_Comm_size(MPI_COMM_WORLD,&size);
        const std::string backend=argv[1];const int pr=std::stoi(argv[2]),pc=std::stoi(argv[3]),nb=std::stoi(argv[4]),reps=std::stoi(argv[5]),warmup=argc>7 ? std::stoi(argv[7]) : 2;
        std::vector<int> sizes;std::stringstream ss(argc>6 ? argv[6] : "1024,2048,4096");std::string part;
        while(std::getline(ss,part,',')) {std::size_t used=0;const int n=std::stoi(part,&used);if(used!=part.size() || n<=0)throw std::runtime_error("invalid size");sizes.push_back(n);}
        if(pr<=0 || pc<=0 || std::int64_t(pr)*pc!=size || nb<=0 || reps<=0 || warmup<1 || sizes.empty())throw std::runtime_error("invalid benchmark arguments");
        if(backend=="cpu") {
            run<DdlaBackend::CPU>(pr,pc,nb,reps,warmup,sizes);
        }else if(backend=="gpu") {
#if DDLA_HAS_GPU && defined(DDLA_USE_CUDA)
            run<DdlaBackend::GPU>(pr,pc,nb,reps,warmup,sizes);
#else
            throw std::runtime_error("CUDA GPU unavailable");
#endif
        }else throw std::runtime_error("expected cpu or gpu");
        MPI_Finalize();return 0;
    }catch(const std::exception& e){std::cerr<<"STRUCTURED_PBLAS_BENCH_FAILURE "<<e.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
