// This client deliberately includes installed/public headers only. In particular
// it does not include backend_api.h or inherit private vendor definitions.
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>
#include <ddla/ddla_vector.h>
#include <mpi.h>
#if defined(DDLA_USE_CPU) || defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP) || defined(DDLA_USE_CCL) || defined(DDLA_USE_GPU_CPU_TUNNEL)
#error "public client inherited private backend definitions"
#endif
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
void require(bool value,const char* message)
{if(!value)throw std::runtime_error(message);}
template<class F>void rejects(F&& function,const char* message)
{bool caught=false;try{function();}catch(const std::exception&){caught=true;}require(caught,message);}
template<class F>void rejects_argument(F&& function,const char* fragment)
{
    bool caught=false;
    try{function();}catch(const std::invalid_argument& error){caught=std::string(error.what()).find(fragment)!=std::string::npos;}
    require(caught,fragment);
}
template<class T>void upload(const ddla::DdlaHandle_t& h,T* target,const std::vector<T>& values)
{ddla::DDLA_CHECK(ddla::ddlaMemcpy(h,target,values.data(),values.size()*sizeof(T)));}
template<class T>std::vector<T> download(const ddla::DdlaHandle_t& h,const T* source,std::size_t count)
{std::vector<T> values(count);ddla::DDLA_CHECK(ddla::ddlaMemcpy(h,values.data(),source,count*sizeof(T)));ddla::DDLA_CHECK(ddla::ddlaSynchronize(h));return values;}
template<class T>void equal(const std::vector<T>& a,const std::vector<T>& b,const char* message)
{
    require(a.size()==b.size(),message);
    using R=typename ddla::real_type<T>::type;
    for(std::size_t i=0;i<a.size();++i)
        require(std::abs(a[i]-b[i])<=R(200)*std::numeric_limits<R>::epsilon()*std::max(R(1),R(std::abs(b[i]))),message);
}
template<class T>void scalar_calls(const ddla::DdlaHandle_t& h)
{
    const std::vector<T> a={T(1),T(2),T(3),T(4)},identity={T(1),T(0),T(0),T(1)};
    ddla::vector<T>A(h,4),B(h,4),C(h,4);upload(h,A.data(),a);upload(h,B.data(),identity);
    // Scalar deduction and explicit scalar selection share the same handle.
    ddla::gemm(h,'N','N',2,2,2,T(1),A.data(),2,B.data(),2,T(0),C.data(),2);
    equal(download(h,C.data(),4),a,"deduced GEMM dispatch");
    ddla::scal<T>(h,4,T(2),C.data(),1);
    equal(download(h,C.data(),4),std::vector<T>{T(2),T(4),T(6),T(8)},"explicit scalar SCAL dispatch");
    ddla::gemm<T>(h,'N','N',0,2,2,T(1),nullptr,1,nullptr,2,T(0),nullptr,1);
    rejects([&]{ddla::gemm<T>(h,'X','N',2,2,2,T(1),A.data(),2,B.data(),2,T(0),C.data(),2);},"invalid flag was accepted");
    rejects([&]{ddla::gemm<T>(nullptr,'N','N',0,0,0,T(1),nullptr,1,nullptr,1,T(0),nullptr,1);},"null handle was accepted for an empty GEMM");
}
template<class R>void mixed_calls(const ddla::DdlaHandle_t& h)
{
    using Z=std::complex<R>;
    const std::vector<R> real={R(1),R(2),R(3),R(4)},diagonal={R(2),R(3)};
    const std::vector<Z> identity={Z(1),Z(0),Z(0),Z(1)},complex={Z(1,1),Z(2,-1),Z(3,2),Z(4,-2)};
    ddla::vector<R>A(h,4),D(h,2);ddla::vector<Z>B(h,4),C(h,4),E(h,4);
    upload(h,A.data(),real);upload(h,D.data(),diagonal);upload(h,B.data(),identity);
    ddla::gemm<R>(h,'N','N',2,2,2,Z(1),A.data(),2,B.data(),2,Z(0),C.data(),2);
    equal(download(h,C.data(),4),std::vector<Z>{Z(1),Z(2),Z(3),Z(4)},"explicit real x complex overload");
    upload(h,B.data(),complex);upload(h,A.data(),std::vector<R>{R(1),R(0),R(0),R(1)});
    ddla::gemm(h,'N','N',2,2,2,Z(1),B.data(),2,A.data(),2,Z(0),C.data(),2);
    equal(download(h,C.data(),4),complex,"deduced complex x real overload");
    // This overload has a second inferred scalar parameter and enable_if.
    ddla::scal<Z>(h,2,2,R(.5),C.data(),2);
    auto half=complex;for(auto& x:half)x*=R(.5);
    equal(download(h,C.data(),4),half,"real coefficient complex SCAL overload");
    ddla::dgmm<Z,R>(h,'L',2,2,B.data(),2,D.data(),1,E.data(),2);
    auto expected=complex;for(int j=0;j<2;++j)for(int i=0;i<2;++i)expected[i+2*j]*=diagonal[i];
    equal(download(h,E.data(),4),expected,"two-scalar DGMM overload");
}
void cholesky_calls(const ddla::DdlaHandle_t& h)
{
    int d[ddla::DDLA_DLEN_];ddla::DDLA_CHECK(ddla::ddlaDescInit(d,h,2,2,2,2,0,0));
    int info=-1;
    ddla::ppotrf<double>(h,'L',0,nullptr,1,1,d,info);
    require(info==0,"zero-size Cholesky INFO changed");
    ddla::vector<double>A(h,4);upload(h,A.data(),std::vector<double>{4.,0.,0.,9.});
    ddla::ppotrf(h,'L',2,A.data(),1,1,d,info);
    require(info==0,"Cholesky dispatch INFO changed");
    equal(download(h,A.data(),4),std::vector<double>{2.,0.,0.,3.},"Cholesky factorization");
    upload(h,A.data(),std::vector<double>{-4.,0.,0.,9.});
    ddla::ppotrf<double>(h,'L',1,A.data(),1,1,d,info);
    require(info==1,"nonpositive Cholesky pivot must fail");
    // The failed pivot's stored value is vendor-dependent; INFO reports failure.
    const auto failed=download(h,A.data(),4);
    equal(std::vector<double>(failed.begin()+1,failed.end()),std::vector<double>{0.,0.,9.},"failed factor preserves the parent region");
}
void cpu_only_operation_rejections(const ddla::DdlaHandle_t& cpu)
{
    rejects_argument([&]{ddla::gemmVbatched<double>(cpu,'N','N',nullptr,nullptr,nullptr,1.,nullptr,nullptr,nullptr,nullptr,0.,nullptr,nullptr,0);},"only the GPU backend is implemented");
    rejects_argument([&]{ddla::gemmVbatched<double>(nullptr,'N','N',nullptr,nullptr,nullptr,1.,nullptr,nullptr,nullptr,nullptr,0.,nullptr,nullptr,0);},"invalid DDLA handle");
    rejects_argument([&]{ddla::gemmVbatched2s<double>(cpu,'N','N',nullptr,nullptr,nullptr,1.,nullptr,nullptr,nullptr,nullptr,0.,nullptr,nullptr,
        'N','N',nullptr,nullptr,nullptr,1.,nullptr,nullptr,nullptr,0.,nullptr,nullptr,false,0,nullptr,0);},"only the GPU backend is implemented");
    rejects_argument([&]{ddla::gemmVbatched2s<double>(nullptr,'N','N',nullptr,nullptr,nullptr,1.,nullptr,nullptr,nullptr,nullptr,0.,nullptr,nullptr,
        'N','N',nullptr,nullptr,nullptr,1.,nullptr,nullptr,nullptr,0.,nullptr,nullptr,false,0,nullptr,0);},"invalid DDLA handle");
}
void exercise(const ddla::DdlaHandle_t& h)
{
    scalar_calls<float>(h);scalar_calls<double>(h);
    scalar_calls<std::complex<float>>(h);scalar_calls<std::complex<double>>(h);
    mixed_calls<float>(h);mixed_calls<double>(h);cholesky_calls(h);
}
}
int main(int argc,char** argv)
{
    int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    if(provided<MPI_THREAD_FUNNELED)MPI_Abort(MPI_COMM_WORLD,2);
    ddla::DdlaHandle_t cpu=nullptr,gpu=nullptr;int failed=0;
    try {
        bool cpu_only=false;
        for(int i=1;i<argc;++i){if(std::string(argv[i])=="--cpu-only")cpu_only=true;else throw std::runtime_error("unknown client option");}
        ddla::DDLA_CHECK(ddla::ddlaInit(cpu,ddla::DdlaBackend::CPU));
        ddla::DDLA_CHECK(ddla::ddlaSet(cpu,MPI_COMM_SELF,1,1));
        exercise(cpu);cpu_only_operation_rejections(cpu);
        using Scale=void(*)(const ddla::DdlaHandle_t&,int,const double&,double*,int);
        Scale scalar_entry=&ddla::scal<double>;
        double x=3.;scalar_entry(cpu,1,2.,&x,1);require(x==6.,"public scalar function pointer dispatch");
        if(!cpu_only&&ddla::ddlaBackendAvailable(ddla::DdlaBackend::GPU)) {
            ddla::DDLA_CHECK(ddla::ddlaInit(gpu,ddla::DdlaBackend::GPU));
            ddla::DDLA_CHECK(ddla::ddlaSet(gpu,MPI_COMM_SELF,1,1));exercise(gpu);
            // CPU selection must still work after using a GPU handle in this process.
            scalar_entry(cpu,1,.5,&x,1);require(x==3.,"CPU handle dispatch after GPU execution");
        }
        int rank=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        if(rank==0)std::cout<<"public handle dispatch PASS cpu=1 gpu="<<(gpu?1:0)<<" public_headers_only=1\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';failed=1;}
    if(gpu)ddla::ddlaDestroy(gpu);if(cpu)ddla::ddlaDestroy(cpu);
    MPI_Allreduce(MPI_IN_PLACE,&failed,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);MPI_Finalize();return failed;
}
