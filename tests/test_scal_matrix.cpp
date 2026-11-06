#include "pblas_structured_test_common.h"
#include "workspace_test_common.h"
#include "BLAS/blas_impl.h"
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
using namespace ddla;
using structured_test::scalar;
using structured_test::wide;
namespace {
void require(DdlaHandle_t h,bool ok,const char* message) {
    int bad=ok?0:1;
    MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h));
    if(bad)throw std::runtime_error(message);
}
void no_workspace(DdlaHandle_t h,const workspace_test::Snapshot& s) {
    require(h,!(s.ddla_alloc||s.ddla_free||s.host_new||s.pinned_alloc||s.raw_gpu_alloc||s.vendor_query||s.resource_setup),
        "scaling allocated workspace or queried resources");
}
template <typename T> bool close(T actual,T expected) {
    using R=typename real_type<T>::type;
    const double error=std::abs(wide(actual)-wide(expected))/(1+std::abs(wide(expected)));
    return std::isfinite(error)&&error<(std::is_same_v<R,float>?3e-5:3e-12);
}
template <DdlaBackend B,typename T>
void local(DdlaHandle_t h,int m,int n,int count,T alpha,bool batched,int& cases) {
    const int lda=std::max(1,m+3);
    const std::size_t stride=std::size_t(lda)*(n+2);
    std::vector<T> seed(stride*count,scalar<T>(77,66));
    const double nan=std::numeric_limits<double>::quiet_NaN();
    for(int q=0;q<count;++q)for(int j=0;j<n;++j)for(int i=0;i<m;++i)
        seed[q*stride+i+std::size_t(j)*lda]=alpha==T(0)?scalar<T>(nan,nan):structured_test::value<T>(i,j,q+1);
    vector<T> data(h,seed.size());std::vector<T*> pointers(count);
    for(int q=0;q<count;++q)pointers[q]=data.data()+q*stride;
    vector<T*> arrays(h,count);
    DDLA_CHECK(ddlaMemcpy(h,arrays.data(),pointers.data(),count*sizeof(T*)));
    auto reset=[&]{DDLA_CHECK(ddlaMemcpy(h,data.data(),seed.data(),seed.size()*sizeof(T)));};
    auto call=[&]{
        if(batched)scalBatched<B,T>(h,m,n,alpha,arrays.data(),lda,count);
        else scal<B,T>(h,m,n,alpha,data.data(),lda);
    };
    reset();call();DDLA_CHECK(ddlaSynchronize(h));reset();
    {workspace_test::Scope scope;call();no_workspace(h,scope.finish());}
    std::vector<T> out(seed.size());DDLA_CHECK(ddlaMemcpy(h,out.data(),data.data(),out.size()*sizeof(T)));DDLA_CHECK(ddlaSynchronize(h));
    bool ok=true;
    for(int q=0;q<count;++q)for(std::size_t p=0;p<stride;++p) {
        const int i=p%lda,j=p/lda;
        if(i<m&&j<n) {
            const T expected=alpha==T(0)?T(0):alpha*seed[q*stride+p];
            ok=ok&&close(out[q*stride+p],expected);
        } else ok=ok&&!std::memcmp(&out[q*stride+p],&seed[q*stride+p],sizeof(T));
    }
    require(h,ok,"matrix scaling changed a guard or produced a wrong value");++cases;
}
template <DdlaBackend B,typename T>
void distributed(DdlaHandle_t h,int m,int n,T alpha,int& cases) {
    int pr,pc;ddlaGetGridDims(h,pr,pc);
    structured_test::Matrix<T> a(h,m,n,4,pr-1,pc-1,true);
    const double nan=std::numeric_limits<double>::quiet_NaN();
    a.fill(m,n,[&](int i,int j){return alpha==T(0)?scalar<T>(nan,nan):structured_test::value<T>(i,j,3);});
    const int rows=num_loc(m,4,a.r,pr-1,pr),cols=num_loc(n,4,a.c,pc-1,pc);
    T* input=rows&&cols?a.data.data():nullptr;
    auto call=[&]{pscal<B,T>(h,m,n,alpha,input,a.d);};
    a.upload();call();DDLA_CHECK(ddlaSynchronize(h));a.upload();
    {workspace_test::Scope scope;call();no_workspace(h,scope.finish());}
    const auto out=a.download();bool ok=true;
    for(std::size_t p=0;p<out.size();++p) {
        const int i=p%a.d[DDLA_LLD_],j=p/a.d[DDLA_LLD_];
        if(i<a.nr&&j<a.nc&&a.gi(i)<m&&a.gj(j)<n) {
            const T expected=alpha==T(0)?T(0):alpha*a.host[p];ok=ok&&close(out[p],expected);
        } else ok=ok&&!std::memcmp(&out[p],&a.host[p],sizeof(T));
    }
    require(h,ok,"distributed scaling changed padding or an entry outside the region");++cases;
}
// Initcheck must see stores, with no prior read of freshly allocated data.
template <DdlaBackend B,typename T>
void zero_uninitialized(DdlaHandle_t h,bool batched,int& cases) {
    const int m=17,n=13,count=batched?3:1;
    const std::size_t stride=std::size_t(m)*n;
    vector<T> data(h,stride*count);
    std::vector<T*> pointers(count);
    for(int q=0;q<count;++q)pointers[q]=data.data()+q*stride;
    vector<T*> arrays(h,count);
    DDLA_CHECK(ddlaMemcpy(h,arrays.data(),pointers.data(),count*sizeof(T*)));
    if(batched)scalBatched<B,T>(h,m,n,T(0),arrays.data(),m,count);
    else scal<B,T>(h,m,n,T(0),data.data(),m);
    std::vector<T> out(stride*count);
    DDLA_CHECK(ddlaMemcpy(h,out.data(),data.data(),out.size()*sizeof(T)));
    DDLA_CHECK(ddlaSynchronize(h));
    require(h,std::all_of(out.begin(),out.end(),[](T x){return x==T(0);}),
        "zero scaling did not initialize the output");
    ++cases;
}
template <DdlaBackend B,typename T> void types(DdlaHandle_t h,int& cases) {
    for(T alpha:{T(0),T(1),scalar<T>(-.75,.25)}) {
        local<B,T>(h,17,11,1,alpha,false,cases);
        local<B,T>(h,17,11,35,alpha,true,cases);
        distributed<B,T>(h,17,11,alpha,cases);
        distributed<B,T>(h,1,2,alpha,cases);
    }
    zero_uninitialized<B,T>(h,false,cases);
    zero_uninitialized<B,T>(h,true,cases);
    distributed<B,T>(h,0,2,T(0),cases);
    local<B,T>(h,1,1,65537,scalar<T>(.5,.25),true,cases);
    scal<B,T>(h,0,3,T(0),nullptr,1);
    scalBatched<B,T>(h,2,3,T(0),nullptr,2,0);
    // The existing vector overload and stride contract remain unchanged.
    std::vector<T> seed(15,scalar<T>(9,8));for(int i=0;i<7;++i)seed[2*i]=scalar<T>(i+1,-i);
    vector<T> x(h,seed.size());DDLA_CHECK(ddlaMemcpy(h,x.data(),seed.data(),seed.size()*sizeof(T)));
    const T alpha=scalar<T>(.75,.125);scal<B,T>(h,7,alpha,x.data(),2);
    std::vector<T> out(seed.size());DDLA_CHECK(ddlaMemcpy(h,out.data(),x.data(),out.size()*sizeof(T)));DDLA_CHECK(ddlaSynchronize(h));
    bool ok=true;for(int i=0;i<15;++i)ok=ok&&close(out[i],i<14&&i%2==0?alpha*seed[i]:seed[i]);
    require(h,ok,"vector scal overload or stride regressed");++cases;
}
template <DdlaBackend B> void run(int pr,int pc) {
    DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,B));DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,pr,pc));
    int cases=0;types<B,float>(h,cases);types<B,double>(h,cases);
    types<B,std::complex<float>>(h,cases);types<B,std::complex<double>>(h,cases);
    bool rejected=false;try{scalBatched<B,double>(h,2,3,1.,nullptr,1,1);}catch(const std::exception&){rejected=true;}
    require(h,rejected,"invalid batched leading dimension accepted");
    structured_test::Matrix<double> a(h,3,3,2);
    int rank;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    rejected=false;try{pscal<B,double>(h,rank==0?4:3,3,1.,a.data.data(),a.d);}catch(const std::exception&){rejected=true;}
    require(h,rejected,"invalid pscal region did not reject collectively");
    if(pr*pc>1) {
        rejected=false;
        try{pscal<B,double>(h,rank==0?2:3,3,1.,a.data.data(),a.d);}
        catch(const std::exception&){rejected=true;}
        require(h,rejected,"inconsistent pscal regions did not reject collectively");
    }
    DDLA_CHECK(ddlaDestroy(h));if(!rank)std::cout<<"MATRIX_SCAL_PASS backend="<<int(B)<<" grid="<<pr<<'x'<<pc<<" cases="<<cases<<std::endl;
}
}
int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);
    try {
        const std::string mode=argc>1?argv[1]:"cpu";int size;MPI_Comm_size(MPI_COMM_WORLD,&size);
        const int pr=argc>2?std::stoi(argv[2]):1,pc=argc>3?std::stoi(argv[3]):size;
        if(pr*pc!=size||pr<1||pc<1)throw std::invalid_argument("invalid process grid");
        if(mode!="gpu")run<DdlaBackend::CPU>(pr,pc);
#if DDLA_HAS_GPU
        if(mode!="cpu")run<DdlaBackend::GPU>(pr,pc);
#endif
        MPI_Finalize();return 0;
    }catch(const std::exception& e){std::cerr<<"MATRIX_SCAL_FAILURE "<<e.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
