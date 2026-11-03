#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "BLAS/gemmt_impl.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <cstdlib>
#include <type_traits>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace ddla;

template<class T> T value(double r,double i) {if constexpr(std::is_arithmetic_v<T>)return T(r);else return T(r,i);}
template<DdlaBackend B,class T>
bool test_case(DdlaHandle_t h,int n,int k,char ta,char tb,char u,T beta) {
    std::size_t bytes=0;blas_impl::gemmt_workspace_size<T>(h,ta,tb,n,k,&bytes);
    if(!bytes)throw std::runtime_error("batched GEMMT product requires scratch");
    const int ar=ta=='N'?n:k,ac=ta=='N'?k:n,br=tb=='N'?k:n,bc=tb=='N'?n:k;
    const int lda=ar+3,ldb=br+5,ldc=n+7;
    std::vector<T> ha(std::size_t(lda)*(ac+2),T(111)),hb(std::size_t(ldb)*(bc+2),T(222)),seed(std::size_t(ldc)*(n+3),T(333));
    for(int j=0;j<ac;++j)for(int i=0;i<ar;++i)ha[i+std::size_t(j)*lda]=value<T>(((i*17+j*7)%23-11)/32.,((i*3+j*11)%17-8)/32.);
    for(int j=0;j<bc;++j)for(int i=0;i<br;++i)hb[i+std::size_t(j)*ldb]=value<T>(((i*13+j*5)%19-9)/32.,((i*7+j*3)%13-6)/32.);
    const T nan=value<T>(std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN());
    for(int j=0;j<n;++j)for(int i=0;i<n;++i)seed[i+std::size_t(j)*ldc]=(u=='U'?i<=j:i>=j)&&beta!=T(0)?value<T>((i%7-j%5)/16.,(i%3+j%7+1)/32.):nan;
    vector<T>a(h,ha.size()),b(h,hb.size()),c(h,seed.size()),reference(h,seed.size());
    DDLA_CHECK(ddlaMemcpy(h,a.data(),ha.data(),ha.size()*sizeof(T)));DDLA_CHECK(ddlaMemcpy(h,b.data(),hb.data(),hb.size()*sizeof(T)));
    const T alpha=value<T>(.75,.125);
    DDLA_CHECK(ddlaMemcpy(h,reference.data(),seed.data(),seed.size()*sizeof(T)));
    gemm<B,T>(h,ta,tb,n,n,k,alpha,a.data(),lda,b.data(),ldb,beta,reference.data(),ldc);
    std::vector<T> expected(seed.size()),actual(seed.size());DDLA_CHECK(ddlaMemcpy(h,expected.data(),reference.data(),expected.size()*sizeof(T)));DDLA_CHECK(ddlaSynchronize(h));
    double scale=1;for(int j=0;j<n;++j)for(int i=0;i<n;++i)if(u=='U'?i<=j:i>=j)scale=std::max(scale,double(std::abs(expected[i+std::size_t(j)*ldc])));
    auto check=[&] {
        DDLA_CHECK(ddlaMemcpy(h,actual.data(),c.data(),actual.size()*sizeof(T)));DDLA_CHECK(ddlaSynchronize(h));double error=0;
        for(std::size_t p=0;p<actual.size();++p) {
            const int i=int(p%ldc),j=int(p/ldc);const bool selected=i<n&&j<n&&(u=='U'?i<=j:i>=j);
            if(selected) {const double delta=std::abs(actual[p]-expected[p]);if(!std::isfinite(delta))throw std::runtime_error("nonfinite batched GEMMT error");error=std::max(error,delta/scale);}
            else if(std::memcmp(&actual[p],&seed[p],sizeof(T)))throw std::runtime_error("GEMMT opposite triangle/padding changed");
        }
        using R=typename real_type<T>::type;if(error>64*std::numeric_limits<R>::epsilon())throw std::runtime_error("batched GEMMT tolerance");
    };
    vector<char> work(h,bytes+512);std::array<char,256> guard;guard.fill(char(0x5a));
    DDLA_CHECK(ddlaMemcpy(h,work.data(),guard.data(),guard.size()));DDLA_CHECK(ddlaMemcpy(h,work.data()+256+bytes,guard.data(),guard.size()));
    // First use leaves scratch uninitialized; repeated use rebuilds all tables.
    for(int repeat=0;repeat<2;++repeat) {
        DDLA_CHECK(ddlaMemcpy(h,c.data(),seed.data(),seed.size()*sizeof(T)));
        blas_impl::gemmt_run<B,T>(h,u,ta,tb,n,k,alpha,a.data(),lda,b.data(),ldb,beta,c.data(),ldc,work.data()+256,bytes);check();
    }
    DDLA_CHECK(ddlaMemcpy(h,c.data(),seed.data(),seed.size()*sizeof(T)));
    gemmt<B,T>(h,u,ta,tb,n,k,alpha,a.data(),lda,b.data(),ldb,beta,c.data(),ldc);check();
    std::array<char,256> first,last;DDLA_CHECK(ddlaMemcpy(h,first.data(),work.data(),256));DDLA_CHECK(ddlaMemcpy(h,last.data(),work.data()+256+bytes,256));
    std::vector<T> aa(ha.size()),bb(hb.size());DDLA_CHECK(ddlaMemcpy(h,aa.data(),a.data(),aa.size()*sizeof(T)));DDLA_CHECK(ddlaMemcpy(h,bb.data(),b.data(),bb.size()*sizeof(T)));DDLA_CHECK(ddlaSynchronize(h));
    if(first!=guard||last!=guard)throw std::runtime_error("GEMMT workspace bounds");
    if(std::memcmp(aa.data(),ha.data(),aa.size()*sizeof(T))||std::memcmp(bb.data(),hb.data(),bb.size()*sizeof(T)))throw std::runtime_error("GEMMT input modified");
    return true;
}
template<DdlaBackend B,class T> int scalar_cases(DdlaHandle_t h) {
    int cases=0;
    for(int n:{1,127,128,129,255,256,257}) {
        cases+=test_case<B,T>(h,n,19,'N','N','U',T(0));
        cases+=test_case<B,T>(h,n,19,'N','N','L',value<T>(.25,-.125));
    }
    for(char ta:{'N','T','C'})for(char tb:{'N','T','C'}) {
        cases+=test_case<B,T>(h,129,17,ta,tb,'U',T(0));
        cases+=test_case<B,T>(h,257,17,ta,tb,'L',value<T>(.25,-.125));
    }
    cases+=test_case<B,T>(h,512,513,'N','N','U',T(0));
    cases+=test_case<B,T>(h,513,513,'C','T','L',value<T>(.25,-.125));
    if(std::getenv("DDLA_GEMMT_LARGE_TEST")) {
        cases+=test_case<B,T>(h,4096,513,'N','N','U',T(0));
        cases+=test_case<B,T>(h,4097,513,'C','T','L',value<T>(.25,-.125));
    }
    return cases;
}
template<DdlaBackend B> int run() {
    DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,B));DDLA_CHECK(ddlaSet(h,MPI_COMM_SELF,1,1));
    std::size_t empty=1;
    blas_impl::gemmt_workspace_size<double>(h,'N','N',0,1,&empty);
    if(empty)throw std::runtime_error("empty GEMMT requested scratch");
    blas_impl::gemmt_workspace_size<double>(h,'C','T',1,0,&empty);
    if(empty)throw std::runtime_error("zero-k GEMMT requested scratch");
    const int cases=scalar_cases<B,float>(h)+scalar_cases<B,double>(h)+scalar_cases<B,std::complex<float>>(h)+scalar_cases<B,std::complex<double>>(h);
    DDLA_CHECK(ddlaDestroy(h));return cases;
}
int main(int argc,char**argv) {
    MPI_Init(&argc,&argv);
    try {
        int cases=run<DdlaBackend::CPU>();
#if DDLA_HAS_GPU
        cases+=run<DdlaBackend::GPU>();
#endif
        if(std::getenv("DDLA_GEMMT_BATCHED_REQUIRED")&&cases==0)throw std::runtime_error("optimized GEMMT path was not exercised");
        int rank=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);if(rank==0)std::cout<<"test_gemmt_batched passed batched_cases="<<cases<<'\n';MPI_Finalize();return 0;
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
