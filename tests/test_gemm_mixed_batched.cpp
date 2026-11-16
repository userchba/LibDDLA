#include "mpi_test_init.h"
#include "pblas_structured_test_common.h"
#include "workspace_test_common.h"
#include "BLAS/blas_impl.h"
#include <array>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include "mixed_products_test_common.h"
using namespace ddla;
namespace {
// One stored column remains valid for any lda>=m. This exceeds the range in
// which complex row pairs can be passed to real BLAS with a doubled lda.
// Reuse the shared mixed fixture's oracle conversions and collective checks.
template <typename R> void oversized_lda(DdlaHandle_t h,bool zero_beta,int& cases) {
    using T=std::complex<R>;
    constexpr int count=3,stride=3;
    const int lda=std::numeric_limits<int>::max()/2+1;
    const T sentinel(77,66),alpha(.75,.125),beta=zero_beta?T(0):T(.25,-.125);
    std::array<T,count*stride> a,c;
    std::array<R,count*stride> b;
    a.fill(sentinel);b.fill(R(88));c.fill(sentinel);
    std::array<const T*,count> ap;
    std::array<const R*,count> bp;
    std::array<T*,count> cp;
    for(int q=0;q<count;++q) {
        a[q*stride+1]=scalar<T>(.25*(q+1),-.125*(q+1));
        b[q*stride+1]=R(.5+.25*q);
        c[q*stride+1]=zero_beta?T(NAN,NAN):scalar<T>(.125,.0625);
        ap[q]=a.data()+q*stride+1;bp[q]=b.data()+q*stride+1;cp[q]=c.data()+q*stride+1;
    }
    const auto before_a=a;const auto before_b=b;
    gemmBatched<R>(h,'N','N',1,1,1,alpha,ap.data(),lda,bp.data(),1,beta,cp.data(),1,count);
    bool okay=a==before_a && b==before_b;double worst=0;
    for(int q=0;q<count;++q) {
        Wide reference=wide(alpha)*wide(a[q*stride+1])*wide(b[q*stride+1]);
        if(!zero_beta)reference+=wide(beta)*wide(scalar<T>(.125,.0625));
        const double error=double(std::abs(wide(c[q*stride+1])-reference)/(1+std::abs(reference)));
        okay=okay && std::isfinite(error) && c[q*stride]==sentinel && c[q*stride+2]==sentinel;
        worst=std::max(worst,error);
    }
    require(h,okay && worst<(std::is_same_v<R,float>?3e-5:3e-12),"mixed CPU batch oversized lda result/input/padding guard");
    ++cases;
}

template <DdlaBackend Backend, typename R, typename TA, typename TB>
void types(DdlaHandle_t h, bool large, int& cases) {
    for (char ta : {'N','T','C'}) for (char tb : {'N','T','C'})
        batch_case<Backend,R,TA,TB>(h,19,23,17,35,ta,tb,0,cases);
    for (int mode=1; mode<5; ++mode)
        batch_case<Backend,R,TA,TB>(h,7,5,mode==2?0:13,3,'C','C',mode,cases);
    batch_case<Backend,R,TA,TB>(h,33,29,257,3,'C','N',1,cases);
    batch_case<Backend,R,TA,TB>(h,7,513,17,2,'C','C',0,cases);
    batch_case<Backend,R,TA,TB>(h,129,97,521,40,'C','N',0,cases);
    batch_case<Backend,R,TA,TB>(h,97,129,1031,40,'N','C',1,cases);
    batch_case<Backend,R,TA,TB>(h,3,2,4,0,'N','N',0,cases);
    if (large) {
        batch_case<Backend,R,TA,TB>(h,257,265,269,3,'C','N',1,cases);
        batch_case<Backend,R,TA,TB>(h,1031,65,513,3,'N','C',0,cases);
        batch_case<Backend,R,TA,TB>(h,1031,1031,65,1,'C','T',1,cases);
    }
}
template <DdlaBackend Backend> void run(bool large) {
    DdlaHandle_t h=nullptr;
    DDLA_CHECK(ddlaInit(h,Backend));
    DDLA_CHECK(ddlaSet(h,MPI_COMM_SELF,1,1));
    using T=std::complex<double>;
    const double* const* a=nullptr;
    const T* const* b=nullptr;
    gemmBatched<double>(h,'N','N',7,7,7,T(0),a,7,b,7,T(1),nullptr,7,3);
    int rejected=0;
    try {gemmBatched<double>(h,'N','N',7,7,7,T(0),a,7,b,7,T(1),nullptr,7,-1);}
    catch (const std::invalid_argument&) {rejected=1;}
    require(h,rejected,"negative batch count accepted");
    int cases=0;
    types<Backend,float,float,std::complex<float>>(h,large,cases);
    types<Backend,float,std::complex<float>,float>(h,large,cases);
    types<Backend,double,double,std::complex<double>>(h,large,cases);
    types<Backend,double,std::complex<double>,double>(h,large,cases);
    if constexpr(Backend==DdlaBackend::CPU) {
        for(bool zero_beta:{false,true}) {
            oversized_lda<float>(h,zero_beta,cases);
            oversized_lda<double>(h,zero_beta,cases);
        }
    }
    DDLA_CHECK(ddlaDestroy(h));
    std::cout << "MIXED_BATCHED_PASS backend=" << int(Backend)
              << " cases=" << cases << " large=" << large << std::endl;
}
}
int main(int argc,char** argv) {
    ddla_test_mpi_init(&argc,&argv);
    try {
        const std::string mode=argc>1?argv[1]:"all";
        const bool large=std::getenv("DDLA_MIXED_LARGE_TEST")!=nullptr;
        if(mode!="gpu")run<DdlaBackend::CPU>(large);
#if DDLA_HAS_GPU
        if(mode!="cpu")run<DdlaBackend::GPU>(large);
#endif
        MPI_Finalize();return 0;
    } catch(const std::exception& e) {
        std::cerr << "MIXED_BATCHED_FAILURE " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD,1);return 1;
    }
}
