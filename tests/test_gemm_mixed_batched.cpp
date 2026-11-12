#include "mpi_test_init.h"
#include "pblas_structured_test_common.h"
#include "workspace_test_common.h"
#include "BLAS/blas_impl.h"
#include <array>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <string>
#include "mixed_products_test_common.h"
using namespace ddla;
namespace {
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
