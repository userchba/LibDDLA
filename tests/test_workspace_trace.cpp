#include <ddla/ddla.h>
#include "lapack_impl.h"
#include "workspace_test_common.h"
#include <mpi.h>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <new>
#include <stdexcept>
#include <thread>
#if defined(DDLA_USE_CUDA)
#include <cuda_runtime_api.h>
#include <curand.h>
#if defined(DDLA_USE_CCL)
#include <nccl.h>
#endif
#endif

static void require(bool good,const char* message)
{if(!good)throw std::runtime_error(message);}

static void controls(ddla::DdlaBackend backend)
{
    const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
    require(std::isinf(workspace_test::finite_error(nan)) &&
            std::isinf(workspace_test::relative_error(nan,0.0)) &&
            std::isinf(workspace_test::relative_error(inf,inf)),"non-finite numerical errors must fail");
    using namespace ddla;
    DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,backend));
    DDLA_CHECK(ddlaSet(h,MPI_COMM_SELF,1,1,'R'));
    void* memory=nullptr;
    {
        workspace_test::Scope scope;
        DDLA_CHECK(ddlaMalloc(h,&memory,64));DDLA_CHECK(ddlaFree(h,memory));
        const auto count=scope.finish();
        require(count.ddla_alloc>0 && count.ddla_free>0,"DDLA allocation controls not detected");
    }
    {
        workspace_test::Scope scope;
        void* p=::operator new(37);
        *static_cast<volatile unsigned char*>(p)=17;
        ::operator delete(p);
        const auto count=scope.finish();
        require(count.host_new>0,"host new control not detected");
    }
    DDLA_CHECK(ddlaMalloc(h,&memory,64));
    int value=19;
    {
        workspace_test::Scope scope;
        DDLA_CHECK(ddlaMemcpy(h,memory,&value,sizeof(value)));
        const auto count=scope.finish();
        require(count.copies>0,"copy control not detected");
    }
    DDLA_CHECK(ddlaSynchronize(h));DDLA_CHECK(ddlaFree(h,memory));
    {
        int result=0;workspace_test::Scope scope;
        MPI_Allreduce(&value,&result,1,MPI_INT,MPI_SUM,MPI_COMM_SELF);
        const auto count=scope.finish();
        require(result==value && count.collectives>0,"MPI communication control not detected");
    }
    {
        workspace_test::Scope scope(true);
        const int error=MPI_Barrier(MPI_COMM_SELF);
        const auto count=scope.finish();
        require(error==MPI_ERR_OTHER && count.collectives>0,"query MPI fail-fast control not detected");
    }
    {
        MPI_Comm prepared=MPI_COMM_NULL;
        workspace_test::Scope scope;
        require(MPI_Comm_dup(MPI_COMM_SELF,&prepared)==MPI_SUCCESS,"communicator setup control failed");
        const auto count=scope.finish();
        require(count.resource_setup>0,"communicator setup control not detected");
        MPI_Comm_free(&prepared);
        workspace_test::Scope query(true);
        const int error=MPI_Comm_dup(MPI_COMM_SELF,&prepared);
        const auto query_count=query.finish();
        require(error==MPI_ERR_OTHER && query_count.resource_setup>0,"query resource fail-fast control not detected");
    }
    {
        MPI_Datatype prepared=MPI_DATATYPE_NULL;
        workspace_test::Scope scope;
        require(MPI_Type_contiguous(2,MPI_DOUBLE,&prepared)==MPI_SUCCESS,"datatype setup control failed");
        require(MPI_Type_commit(&prepared)==MPI_SUCCESS,"datatype commit control failed");
        const auto count=scope.finish();
        require(count.resource_setup>0,"datatype setup control not detected");
        MPI_Type_free(&prepared);
        workspace_test::Scope query(true);
        const int error=MPI_Type_contiguous(2,MPI_DOUBLE,&prepared);
        const auto query_count=query.finish();
        require(error==MPI_ERR_OTHER && query_count.resource_setup>0,"query datatype fail-fast control not detected");
    }
    {
        workspace_test::Scope scope;
        std::thread worker([]{});worker.join();
        const auto count=scope.finish();
        require(count.host_new>0,"worker creation allocation control not detected");
    }
    double *a=nullptr,*tau=nullptr;
    DDLA_CHECK(ddlaMalloc(h,&a,6*sizeof(double)));
    DDLA_CHECK(ddlaMalloc(h,&tau,2*sizeof(double)));
    {
        std::size_t bytes=0;
        workspace_test::Scope scope;
        lapack_impl::geqrf_workspace_size<double>(h,'U',3,2,a,3,tau,&bytes);
        const auto count=scope.finish();
        require(bytes>0 && count.vendor_query>0,"vendor query control not detected");
    }
    DDLA_CHECK(ddlaFree(h,tau));DDLA_CHECK(ddlaFree(h,a));
#if defined(DDLA_USE_CUDA)
    if(backend==DdlaBackend::GPU) {
        workspace_test::Scope scope;
        void* p=nullptr;
        require(cudaMalloc(&p,31)==cudaSuccess,"raw allocation control failed");
        require(cudaFree(p)==cudaSuccess,"raw free control failed");
        const auto count=scope.finish();
        require(count.raw_gpu_alloc>0,"raw CUDA allocation control not detected");
        cudaEvent_t event=nullptr;
        workspace_test::Scope setup;
        const auto created=cudaEventCreateWithFlags(&event,cudaEventDisableTiming);
        const auto setup_count=setup.finish();
        require(created==cudaSuccess && setup_count.resource_setup>0,"CUDA event setup control not detected");
        require(cudaEventDestroy(event)==cudaSuccess,"CUDA event teardown control failed");
        workspace_test::Scope query_resource(true);
        const auto blocked=cudaEventCreateWithFlags(&event,cudaEventDisableTiming);
        const auto blocked_count=query_resource.finish();
        require(blocked==cudaErrorNotPermitted && blocked_count.resource_setup>0,"query CUDA resource fail-fast control not detected");
        curandGenerator_t generator=nullptr;
        workspace_test::Scope rng_setup;
        const auto rng_created=curandCreateGenerator(&generator,CURAND_RNG_PSEUDO_DEFAULT);
        const auto rng_count=rng_setup.finish();
        require(rng_created==CURAND_STATUS_SUCCESS && rng_count.resource_setup>0,"GPU RNG setup control not detected");
        require(curandDestroyGenerator(generator)==CURAND_STATUS_SUCCESS,"GPU RNG teardown control failed");
        workspace_test::Scope query_rng(true);
        const auto rng_blocked=curandCreateGenerator(&generator,CURAND_RNG_PSEUDO_DEFAULT);
        const auto rng_blocked_count=query_rng.finish();
        require(rng_blocked==CURAND_STATUS_INTERNAL_ERROR && rng_blocked_count.resource_setup>0,"query RNG resource fail-fast control not detected");
#if defined(DDLA_USE_CCL)
        using Group=ncclResult_t(*)();
        void* start_symbol=dlsym(RTLD_DEFAULT,"ncclGroupStart");
        void* end_symbol=dlsym(RTLD_DEFAULT,"ncclGroupEnd");
        Group start=nullptr,end=nullptr;
        std::memcpy(&start,&start_symbol,sizeof(start));std::memcpy(&end,&end_symbol,sizeof(end));
        require(start && end,"NCCL control symbols missing");
        workspace_test::Scope query_scope(true);
        const auto error=start();const auto query_count=query_scope.finish();
        if(error==ncclSuccess)(void)end();
        require(error==ncclInvalidUsage && query_count.collectives>0,"query NCCL fail-fast control not detected");
#endif
    }
#endif
    DDLA_CHECK(ddlaDestroy(h));
}

int main(int argc,char** argv)
{
    MPI_Init(&argc,&argv);
    try {
        if(!workspace_test::available()) {
            std::cout<<"workspace tracing unavailable; diagnostic controls skipped"<<std::endl;
            MPI_Finalize();return 77;
        }
        controls(ddla::DdlaBackend::CPU);
#if DDLA_HAS_GPU
        controls(ddla::DdlaBackend::GPU);
#endif
        std::cout<<"workspace diagnostic negative controls PASS"<<std::endl;
        MPI_Finalize();return 0;
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);return 1;
    }
}
