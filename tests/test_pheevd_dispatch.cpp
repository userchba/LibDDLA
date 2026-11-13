// Routing-only endpoint interposition. No matrix storage is read or allocated:
// large synthetic descriptors exercise public dispatch without a large solve.
#include "api_grid_test_common.h"
#include "backend_api.h" // Private backend endpoints are deliberately intercepted below.
#include "Eigensolver/eig_params.h"
#include "Eigensolver/pheevd_impl.h"
#include "host_threads.h"
#include "ddla_stream.h"
#include "instantiate.h"
#include <cstring>
using namespace api_grid_test;
namespace {
int called=0,calls=0,numbers[4]{};
const void* pointers[6]{};
template<class T>void record(int method,const DdlaHandle_t& h,char job,char uplo,int n,int nev,T* a,const int* da,
                            typename real_type<T>::type* w,T* z,const int* dz,int& info) {
    ++calls;called=method;numbers[0]=job;numbers[1]=uplo;numbers[2]=n;numbers[3]=nev;
    pointers[0]=h;pointers[1]=a;pointers[2]=da;pointers[3]=w;pointers[4]=z;pointers[5]=dz;info=17;
}
}
namespace ddla {
#define STUB(B,T) \
    template<> void pheevd_one_stage<B,T>(const DdlaHandle_t& h,char job,char uplo,int n,int nev,T* a,const int* da,typename real_type<T>::type* w,T* z,const int* dz,int& info) {record(1,h,job,uplo,n,nev,a,da,w,z,dz,info);} \
    template<> void pheevd_two_stage<B,T>(const DdlaHandle_t& h,char job,char uplo,int n,int nev,T* a,const int* da,typename real_type<T>::type* w,T* z,const int* dz,int& info) {record(2,h,job,uplo,n,nev,a,da,w,z,dz,info);}
DDLA_FOR_EACH_BACKEND_TYPE(STUB)
#undef STUB
namespace detail {
#define LOCAL_STUB(B,T) template<> void pheevd_local<B,T>(const DdlaHandle_t& h,char job,char uplo,int n,int nev,T* a,const int* da,typename real_type<T>::type* w,T* z,const int* dz,int& info) {record(3,h,job,uplo,n,nev,a,da,w,z,dz,info);}
DDLA_FOR_EACH_BACKEND_TYPE(LOCAL_STUB)
#undef LOCAL_STUB
}
}
namespace {
template<DdlaBackend B,class T>void test(DdlaHandle_t h,bool device) {
    using R=typename real_type<T>::type;
    const int original_workers=h->host_threads,original_node=h->node_ranks;
    const int workers=B==DdlaBackend::CPU?detail::pheevd_method_cpu_threads:1;
    detail::ScopedBlasThreads blas(std::max(1,workers));
    h->host_threads=workers;
    T a{},z{};R w{};int d[DDLA_DLEN_];int info=-1;
    const auto invoke=[&](char job,char uplo,int n,int expected,bool reject=false,int block=64,int nev=-1) {
        DDLA_CHECK(ddlaDescInit(d,h,std::max(1,n),std::max(1,n),block,block,0,0));
        T* out=job=='V'?&z:nullptr;const int* dz=job=='V'?d:nullptr;
        calls=called=0;info=-1;bool caught=false;
        try {pheevd<T>(h,job,uplo,n,nev<0?n:nev,&a,d,&w,out,dz,info);}catch(const std::exception&){caught=true;}
        require_close(h,"public dispatcher collective rejection",caught==reject?0:1,0);
        if(reject){require_close(h,"rejection before algorithm",calls,0);return;}
        const char* mode=std::getenv("DDLA_EIGEN_METHOD");
        if(h->nprocs==1 && (!mode || !*mode || std::strcmp(mode,"auto")==0)) expected=3;
        bool same=calls==1&&called==expected&&info==17&&numbers[0]==job&&numbers[1]==uplo&&numbers[2]==n&&numbers[3]==(nev<0?n:nev);
        const void* wanted[6]={h,&a,d,&w,out,dz};
        for(int i=0;i<6;++i)same=same&&pointers[i]==wanted[i];
        require_close(h,"public dispatcher selected endpoint and forwarded arguments",same?0:1,0);
    };
    for(char job:{'N','V'})for(char uplo:{'U','L'})for(int n:{0,1,511,512,2048,4096,8192,16384,32768}) {
        unsetenv("DDLA_EIG_NBW");unsetenv("DDLA_EIG_CPU_ACCUM");unsetenv("DDLA_EIG_CHASE_TEAM_THREADS");unsetenv("DDLA_EIG_PROFILE");
        const bool shape=detail::pheevd_one_stage_shape<B,T>(job,n,64,h->nprows_,h->npcols_,h->major);
        const bool local=shape&&h->node_ranks==h->nprocs&&(B==DdlaBackend::CPU||device)&&workers>0&&
            detail::ScopedBlasThreads::current()==workers;
        int prefer=local?1:0;MPI_Allreduce(MPI_IN_PLACE,&prefer,1,MPI_INT,MPI_MIN,h->comm);
        for(const char* mode:{static_cast<const char*>(nullptr),"","auto"}) {
            if(mode)setenv("DDLA_EIGEN_METHOD",mode,1);else unsetenv("DDLA_EIGEN_METHOD");
            invoke(job,uplo,n,prefer?1:2);
        }
        setenv("DDLA_EIGEN_METHOD","one_stage",1);setenv("DDLA_EIG_NBW","3",1);invoke(job,uplo,n,1);
        setenv("DDLA_EIGEN_METHOD","two_stage",1);invoke(job,uplo,n,2);
        setenv("DDLA_EIGEN_METHOD","auto",1);setenv("DDLA_EIG_NBW","64",1);invoke(job,uplo,n,2);
        unsetenv("DDLA_EIG_NBW");setenv("DDLA_EIG_CPU_ACCUM","off",1);invoke(job,uplo,n,2);unsetenv("DDLA_EIG_CPU_ACCUM");
        if(h->myid==0)setenv("DDLA_EIG_CHASE_TEAM_THREADS","1",1);
        invoke(job,uplo,n,2);unsetenv("DDLA_EIG_CHASE_TEAM_THREADS");
        if(h->myid==0)h->node_ranks=0;invoke(job,uplo,n,2);h->node_ranks=original_node;
        if(h->myid==0)h->host_threads=workers+1;
        invoke(job,uplo,n,2);h->host_threads=workers;
        {
            detail::ScopedBlasThreads changed(h->myid==0?workers+1:std::max(1,workers));
            invoke(job,uplo,n,2);
        }
        setenv("DDLA_EIG_PROFILE","1",1);invoke(job,uplo,n,2);unsetenv("DDLA_EIG_PROFILE");
        if(shape&&h->nprocs>1)invoke(job,uplo,n,0,true,h->myid==0?32:64);
        if(h->nprocs>1) {
            setenv("DDLA_EIGEN_METHOD",h->myid?"two_stage":"one_stage",1);invoke(job,uplo,n,0,true);
            setenv("DDLA_EIGEN_METHOD",h->myid?"auto":"invalid",1);invoke(job,uplo,n,0,true);
        }
    }
    unsetenv("DDLA_EIGEN_METHOD");
    invoke('V','U',8192,2,false,64,5);
    setenv("DDLA_EIGEN_METHOD","one_stage",1);
    invoke('V','U',8192,1,false,64,5);
    setenv("DDLA_EIGEN_METHOD","two_stage",1);
    invoke('V','U',8192,2,false,64,5);
    h->host_threads=original_workers;h->node_ranks=original_node;
    unsetenv("DDLA_EIGEN_METHOD");
}
template<DdlaBackend B>void backend(std::pair<int,int> grid) {
    DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,B));DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,grid.first,grid.second));
    bool device=false;
#if defined(DDLA_USE_CUDA) && defined(DDLA_USE_CCL)
    if constexpr(B==DdlaBackend::GPU) {
        int major=0,minor=0;GPU_CHECK(gpuDeviceGetAttribute(&major,gpuDevAttrComputeCapabilityMajor,h->local_device));
        GPU_CHECK(gpuDeviceGetAttribute(&minor,gpuDevAttrComputeCapabilityMinor,h->local_device));device=ddlaGetCommunication(h)==DdlaCommunication::CCL && major==7&&minor==0;
    }
#endif
    test<B,float>(h,device);test<B,double>(h,device);test<B,std::complex<float>>(h,device);test<B,std::complex<double>>(h,device);
    DDLA_CHECK(ddlaDestroy(h));
}
}
int main(int argc,char**argv) {
    ddla_test_mpi_init(&argc,&argv);
    try {
        int P=0;MPI_Comm_size(MPI_COMM_WORLD,&P);TestOptions o;std::string error;
        if(!parse_options(argc,argv,P,o,error))throw std::runtime_error(error);
        for(auto grid:o.grids) {
            backend<DdlaBackend::CPU>(grid);
#if DDLA_HAS_GPU
            backend<DdlaBackend::GPU>(grid);
#endif
        }
        std::puts("PHEEVD_PUBLIC_DISPATCH_PASS");MPI_Finalize();return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"PHEEVD_PUBLIC_DISPATCH_FAIL %s\n",e.what());MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
