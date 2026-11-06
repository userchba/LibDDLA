#include "pblas_structured_test_common.h"
#include "BLAS/hemmt_impl.h"
#include "PBLAS/pgemmt_impl.h"
#include "PBLAS/phemm_impl.h"
#include <iomanip>
#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>
using namespace ddla;
using structured_test::Matrix;
using structured_test::value;
namespace {
template <typename Call> double measure(DdlaHandle_t h,Call call,int reps) {
    const char* setting=std::getenv("DDLA_BENCH_WARMUPS");
    const int warmups=setting?std::stoi(setting):2;
    if(warmups<0)throw std::invalid_argument("negative DDLA_BENCH_WARMUPS");
    for(int i=0;i<warmups;++i)call();
    DDLA_CHECK(ddlaSynchronize(h));
    std::vector<double> samples;
    for(int i=0;i<reps;++i) {
        MPI_Barrier(ddlaGetCommunicator(h));double start=MPI_Wtime();call();DDLA_CHECK(ddlaSynchronize(h));
        double local=MPI_Wtime()-start,maximum=0;MPI_Allreduce(&local,&maximum,1,MPI_DOUBLE,MPI_MAX,ddlaGetCommunicator(h));samples.push_back(maximum);
    }
    std::sort(samples.begin(),samples.end());
    if(const char* path=std::getenv("DDLA_BENCH_SAMPLES")) {
        int rank;MPI_Comm_rank(ddlaGetCommunicator(h),&rank);
        if(!rank){std::ofstream file(path,std::ios::app);for(double t:samples)file<<std::setprecision(17)<<t<<',';file<<'\n';}
    }
    return samples[samples.size()/2];
}
template <DdlaBackend Backend,typename R,typename TA,typename TB>
void run_case(DdlaHandle_t h,const std::string& op,int n,char side,int reps) {
    using T=std::complex<R>;const bool local=op[0]!='p';
    const bool structured=op=="hemm"||op=="hemmt"||op=="phemm"||op=="phemmt";
    const bool triangle=op=="gemmt"||op=="pgemmt"||op=="hemmt"||op=="phemmt";
    const int batch=op=="gemmBatched"?4:1,nb=128;
    Matrix<TA> a(h,n,n*batch,nb);Matrix<TB> b(h,n,n*batch,nb);
    Matrix<T> promoted(h,n,n*batch,nb),c(h,n,n*batch,nb);
    a.fill(n,n*batch,[&](int i,int j){return value<TA>(i,j,1);});
    b.fill(n,n*batch,[&](int i,int j){return value<TB>(i,j,2);});
    if constexpr(std::is_arithmetic_v<TA>)
        for(std::size_t i=0;i<a.host.size();++i)promoted.host[i]=T(a.host[i]);
    else
        for(std::size_t i=0;i<b.host.size();++i)promoted.host[i]=T(b.host[i]);
    const T* ca=[&]() ->const T* {if constexpr(std::is_same_v<T,TA>)return a.data.data();else return promoted.data.data();}();
    const T* cb=[&]() ->const T* {if constexpr(std::is_same_v<T,TB>)return b.data.data();else return promoted.data.data();}();
    c.fill(n,n*batch,[&](int i,int j){return value<T>(i,j,3);});
    a.upload();b.upload();promoted.upload();c.upload();
    std::vector<const TA*> ha(batch);std::vector<const TB*> hb(batch);
    std::vector<const T*> hca(batch),hcb(batch);std::vector<T*> hc(batch);
    vector<const TA*> da(h,batch);vector<const TB*> db(h,batch);vector<const T*> dca(h,batch),dcb(h,batch);vector<T*> dc(h,batch);
    if(op=="gemmBatched") {
        const std::size_t stride=std::size_t(n)*n;
        for(int q=0;q<batch;++q){ha[q]=a.data.data()+q*stride;hb[q]=b.data.data()+q*stride;hca[q]=ca+q*stride;hcb[q]=cb+q*stride;hc[q]=c.data.data()+q*stride;}
        DDLA_CHECK(ddlaMemcpy(h,da.data(),ha.data(),batch*sizeof(TA*)));DDLA_CHECK(ddlaMemcpy(h,db.data(),hb.data(),batch*sizeof(TB*)));
        DDLA_CHECK(ddlaMemcpy(h,dca.data(),hca.data(),batch*sizeof(T*)));DDLA_CHECK(ddlaMemcpy(h,dcb.data(),hcb.data(),batch*sizeof(T*)));DDLA_CHECK(ddlaMemcpy(h,dc.data(),hc.data(),batch*sizeof(T*)));
    }
    const T alpha(.75,.125),beta(0);
    auto mixed=[&] {
        if(op=="gemmt")gemmt<Backend,R>(h,'L','N','N',n,n,alpha,a.data.data(),n,b.data.data(),n,beta,c.data.data(),n);
        else if(op=="pgemmt")pgemmt<Backend,R>(h,'L','N','N',n,n,alpha,a.data.data(),a.d,b.data.data(),b.d,beta,c.data.data(),c.d);
        else if(op=="hemm")hemm<Backend,R>(h,side,'L',n,n,alpha,a.data.data(),n,b.data.data(),n,beta,c.data.data(),n);
        else if(op=="hemmt")hemmt<Backend,R>(h,side,'L','L',n,alpha,a.data.data(),n,b.data.data(),n,beta,c.data.data(),n);
        else if(op=="phemm")phemm<Backend,R>(h,side,'L',n,n,alpha,a.data.data(),a.d,b.data.data(),b.d,beta,c.data.data(),c.d);
        else if(op=="phemmt")phemmt<Backend,R>(h,side,'L','L',n,alpha,a.data.data(),a.d,b.data.data(),b.d,beta,c.data.data(),c.d);
        else gemmBatched<Backend,R>(h,'N','N',n,n,n,alpha,da.data(),n,db.data(),n,beta,dc.data(),n,batch);
    };
    auto complex=[&] {
        if(op=="gemmt")gemmt<Backend,T>(h,'L','N','N',n,n,alpha,ca,n,cb,n,beta,c.data.data(),n);
        else if(op=="pgemmt")pgemmt<Backend,T>(h,'L','N','N',n,n,alpha,ca,promoted.d,cb,promoted.d,beta,c.data.data(),c.d);
        else if(op=="hemm")hemm<Backend,T>(h,side,'L',n,n,alpha,ca,n,cb,n,beta,c.data.data(),n);
        else if(op=="hemmt")hemmt<Backend,T>(h,side,'L','L',n,alpha,ca,n,cb,n,beta,c.data.data(),n);
        else if(op=="phemm")phemm<Backend,T>(h,side,'L',n,n,alpha,ca,promoted.d,cb,promoted.d,beta,c.data.data(),c.d);
        else if(op=="phemmt")phemmt<Backend,T>(h,side,'L','L',n,alpha,ca,promoted.d,cb,promoted.d,beta,c.data.data(),c.d);
        else gemmBatched<Backend,T>(h,'N','N',n,n,n,alpha,dca.data(),n,dcb.data(),n,beta,dc.data(),n,batch);
    };
    std::size_t mixed_bytes=0,complex_bytes=0,host=0;
    if(op=="gemmt"){blas_impl::gemmt_workspace_size<T,TA,TB>(h,'N','N',n,n,&mixed_bytes);blas_impl::gemmt_workspace_size<T>(h,'N','N',n,n,&complex_bytes);}
    else if(op=="pgemmt"){pgemmt_impl::workspace_size<T,TA,TB>(h,'L','N','N',true,n,n,a.d,b.d,c.d,&mixed_bytes,&host);pgemmt_impl::workspace_size<T>(h,'L','N','N',true,n,n,promoted.d,promoted.d,c.d,&complex_bytes,&host);}
    else if(op=="hemmt"){blas_impl::hemmt_workspace_size<T,TA,TB>(h,n,true,&mixed_bytes,n,side,'L');blas_impl::hemmt_workspace_size<T>(h,n,true,&complex_bytes);}
    else if(op=="hemm")blas_impl::hemmt_workspace_size<T,TA,TB>(h,n,true,&mixed_bytes,n,side,'F');
    else if(op=="phemm"||op=="phemmt"){phemm_impl::phemm_workspace_size<T,TA,TB>(h,side,n,n,false,a.d,b.d,c.d,true,&mixed_bytes,&host,triangle?'L':'F');phemm_impl::phemm_workspace_size<T>(h,side,n,n,false,promoted.d,promoted.d,c.d,true,&complex_bytes,&host,triangle?'L':'F');}
    else blas_impl::mixed_gemmBatched_workspace_size<R>(h,n,n,n,std::is_same_v<T,TA>,batch,&mixed_bytes);
    // Conversion is outside timing. Alternate measurement order for a stable
    // comparison of public calls, including their own scratch allocation.
    const double c1=measure(h,complex,reps),m1=measure(h,mixed,reps),m2=measure(h,mixed,reps),c2=measure(h,complex,reps);
    const double mt=(m1+m2)/2,ct=(c1+c2)/2;
    mixed();DDLA_CHECK(ddlaMemcpy(h,c.host.data(),c.data.data(),c.host.size()*sizeof(T)));
    DDLA_CHECK(ddlaSynchronize(h));complex();const auto complex_result=c.download();
    double error=0,scale=0;
    for(int j=0;j<c.nc;++j)for(int i=0;i<c.nr;++i) {
        if(c.gi(i)>=n||c.gj(j)>=n*batch||(triangle&&c.gi(i)<c.gj(j)))continue;
        const auto p=i+std::size_t(j)*c.d[DDLA_LLD_];
        const double e=std::abs(c.host[p]-complex_result[p]);
        if(!std::isfinite(e))throw std::runtime_error("nonfinite benchmark result");
        error=std::max(error,e);scale=std::max(scale,double(std::abs(complex_result[p])));
    }
    double rel=error/(1+scale),global=0;MPI_Allreduce(&rel,&global,1,MPI_DOUBLE,MPI_MAX,ddlaGetCommunicator(h));
    if(global>(std::is_same_v<R,float>?3e-5:3e-12))throw std::runtime_error("benchmark numerical mismatch");
    std::size_t maxmixed=0,maxcomplex=0;MPI_Allreduce(&mixed_bytes,&maxmixed,1,MPI_UNSIGNED_LONG,MPI_MAX,ddlaGetCommunicator(h));MPI_Allreduce(&complex_bytes,&maxcomplex,1,MPI_UNSIGNED_LONG,MPI_MAX,ddlaGetCommunicator(h));
    int rank,pr,pc;MPI_Comm_rank(ddlaGetCommunicator(h),&rank);ddlaGetGridDims(h,pr,pc);
    if(!rank)std::cout<<op<<','<<(std::is_arithmetic_v<TA>?"RC":"CR")<<','<<(std::is_same_v<R,float>?"float":"double")<<','<<side<<','<<n<<','<<pr<<'x'<<pc<<','<<batch<<','<<std::setprecision(9)<<mt<<','<<ct<<','<<ct/mt<<','<<maxmixed<<','<<maxcomplex<<','<<global<<std::endl;
}
template <DdlaBackend Backend,typename R> void run(int n,int pr,int pc,int reps) {
    DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,Backend));DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,pr,pc));
    const bool local=pr*pc==1;
    std::vector<std::string> ops=local?std::vector<std::string>{"gemmt","hemm","hemmt","gemmBatched"}:std::vector<std::string>{"pgemmt","phemm","phemmt"};
    const char* selected=std::getenv("DDLA_BENCH_OP");
    if(selected)ops.erase(std::remove_if(ops.begin(),ops.end(),[&](const std::string& op){return op!=selected;}),ops.end());
    if(ops.empty())throw std::invalid_argument("DDLA_BENCH_OP does not match the selected process grid");
    for(const auto& op:ops)for(char side:{'L','R'}) {
        if(side=='R'&&(op=="gemmt"||op=="pgemmt"||op=="gemmBatched"))continue;
        run_case<Backend,R,R,std::complex<R>>(h,op,n,side,reps);run_case<Backend,R,std::complex<R>,R>(h,op,n,side,reps);
    }
    DDLA_CHECK(ddlaDestroy(h));
}
}
int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);
    try {
        const std::string backend=argc>1?argv[1]:"cpu";const int n=argc>2?std::stoi(argv[2]):1024;
        int size,rank;MPI_Comm_size(MPI_COMM_WORLD,&size);MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        const int pr=argc>3?std::stoi(argv[3]):1,pc=argc>4?std::stoi(argv[4]):size,reps=argc>5?std::stoi(argv[5]):5;
        if(pr*pc!=size||n<1||reps<1)throw std::invalid_argument("benchmark_mixed_products backend n pr pc reps [float]");
        const bool single=argc>6&&std::string(argv[6])=="float";
        if(!rank)std::cout<<"operation,domain,precision,side,n,grid,batch,mixed_seconds,complex_seconds,speedup,mixed_ddla_scratch_bytes,complex_ddla_scratch_bytes,error\n";
        if(backend=="cpu"){if(single)run<DdlaBackend::CPU,float>(n,pr,pc,reps);else run<DdlaBackend::CPU,double>(n,pr,pc,reps);}
#if DDLA_HAS_GPU
        else {if(single)run<DdlaBackend::GPU,float>(n,pr,pc,reps);else run<DdlaBackend::GPU,double>(n,pr,pc,reps);}
#endif
        MPI_Finalize();return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
