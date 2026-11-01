// Diagnostic preload, built separately from LibDDLA:
// mpicxx -std=c++17 -fPIC -shared -O2 -I<configured-include> -Iinclude \
//   tests/workspace_trace.cpp -o libworkspace_trace.so -ldl
// Add the matching CUDA definition/include paths for the CUDA profile.
#include "workspace_trace_api.h"
#include <ddla/ddla_handle_t.h>

#include <atomic>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <execinfo.h>
#include <new>
#include <unistd.h>

#if defined(DDLA_USE_CUDA)
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <curand.h>
#if defined(DDLA_USE_CCL)
#include <nccl.h>
#endif
#endif

namespace {
enum Counter { DdlaAlloc,DdlaFree,PinnedAlloc,RawGpuAlloc,HostNew,VendorQuery,Copies,Launches,Syncs,Collectives,ResourceSetup,CounterCount };
std::atomic<uint64_t> counters[CounterCount]{};
std::atomic<bool> enabled{false},query_mode{false};
std::atomic<const void*> failing_copy_source{nullptr};
std::atomic<uint64_t> failing_copy_bytes{0},copy_faults{0};
thread_local unsigned resolving=0;
char executable[4096]={};
const char* own_path=nullptr;

struct ResolutionGuard {ResolutionGuard(){++resolving;}~ResolutionGuard(){--resolving;}};
const char* basename(const char* path)
{if(!path)return "";const char* p=std::strrchr(path,'/');return p?p+1:path;}

template <typename F> F next(const char* name)
{
    ResolutionGuard guard;
    void* address=dlsym(RTLD_NEXT,name);F fn=nullptr;
    static_assert(sizeof(fn)==sizeof(address),"dlsym function pointer ABI");
    std::memcpy(&fn,&address,sizeof(fn));return fn;
}
template <typename F> F next_own(F self)
{
    ResolutionGuard guard;
    static_assert(sizeof(self)==sizeof(void*),"dladdr function pointer ABI");
    void* address=nullptr;std::memcpy(&address,&self,sizeof(address));
    Dl_info info{};
    if(!dladdr(address,&info) || !info.dli_sname)return nullptr;
    return next<F>(info.dli_sname);
}

// Stop at the first non-allocator owner. Looking farther up through a vendor
// frame would incorrectly charge its private allocation to the DDLA caller.
bool managed_caller()
{
    if(resolving)return false;
    ResolutionGuard guard;
    void* frames[32];const int count=backtrace(frames,32);
    for(int i=1;i<count;++i) {
        Dl_info info{};
        if(!dladdr(frames[i],&info) || !info.dli_fname)continue;
        if(own_path && std::strcmp(info.dli_fname,own_path)==0)continue;
        const char* file=basename(info.dli_fname);
        if(std::strstr(file,"libstdc++") || std::strstr(file,"libc++") ||
           std::strstr(file,"libgcc_s") || std::strstr(file,"libunwind") ||
           std::strncmp(file,"libc.so",7)==0 || std::strncmp(file,"ld-linux",8)==0)continue;
        if(std::strstr(file,"libddla.") || std::strstr(file,"libddla-") ||
           (executable[0] && std::strcmp(file,basename(executable))==0))return true;
        return false;
    }
    return false;
}
void count(Counter which,bool managed=false)
{
    if(enabled.load(std::memory_order_relaxed) && !resolving && (!managed || managed_caller()))
        counters[which].fetch_add(1,std::memory_order_relaxed);
}
bool collective()
{
    if(!enabled.load(std::memory_order_relaxed) || resolving)return false;
    counters[Collectives].fetch_add(1,std::memory_order_relaxed);
    return query_mode.load(std::memory_order_relaxed);
}
bool resource()
{
    if(!enabled.load(std::memory_order_relaxed) || resolving || !managed_caller())return false;
    counters[ResourceSetup].fetch_add(1,std::memory_order_relaxed);
    return query_mode.load(std::memory_order_relaxed);
}
void* allocate(std::size_t size,std::size_t alignment=0)
{
    count(HostNew,true);
    for(;;) {
        void* result=nullptr;
        if(alignment) {
            if(posix_memalign(&result,alignment,size?size:1)!=0)result=nullptr;
        } else result=std::malloc(size?size:1);
        if(result)return result;
        const auto handler=std::get_new_handler();
        if(!handler)throw std::bad_alloc();
        handler();
    }
}
void prewarm();
}

void* operator new(std::size_t n){return allocate(n);}
void* operator new[](std::size_t n){return allocate(n);}
void* operator new(std::size_t n,const std::nothrow_t&) noexcept{try{return allocate(n);}catch(...){return nullptr;}}
void* operator new[](std::size_t n,const std::nothrow_t&) noexcept{try{return allocate(n);}catch(...){return nullptr;}}
void* operator new(std::size_t n,std::align_val_t a){return allocate(n,std::size_t(a));}
void* operator new[](std::size_t n,std::align_val_t a){return allocate(n,std::size_t(a));}
void* operator new(std::size_t n,std::align_val_t a,const std::nothrow_t&) noexcept{try{return allocate(n,std::size_t(a));}catch(...){return nullptr;}}
void* operator new[](std::size_t n,std::align_val_t a,const std::nothrow_t&) noexcept{try{return allocate(n,std::size_t(a));}catch(...){return nullptr;}}
void operator delete(void* p) noexcept{std::free(p);}
void operator delete[](void* p) noexcept{std::free(p);}
void operator delete(void* p,std::size_t) noexcept{std::free(p);}
void operator delete[](void* p,std::size_t) noexcept{std::free(p);}
void operator delete(void* p,std::align_val_t) noexcept{std::free(p);}
void operator delete[](void* p,std::align_val_t) noexcept{std::free(p);}
void operator delete(void* p,std::size_t,std::align_val_t) noexcept{std::free(p);}
void operator delete[](void* p,std::size_t,std::align_val_t) noexcept{std::free(p);}
void operator delete(void* p,const std::nothrow_t&) noexcept{std::free(p);}
void operator delete[](void* p,const std::nothrow_t&) noexcept{std::free(p);}
void operator delete(void* p,std::align_val_t,const std::nothrow_t&) noexcept{std::free(p);}
void operator delete[](void* p,std::align_val_t,const std::nothrow_t&) noexcept{std::free(p);}

// Exact C++ declarations come from the public header. Symbol names are obtained
// from these wrappers rather than guessed from a compiler's mangling rules.
namespace ddla {
#define DDLA_TRACE(Name,CounterName,Params,Args) \
    using Name##Fn=ddlaStatus_t(*) Params; \
    static Name##Fn real_##Name(){static Name##Fn f=next_own<Name##Fn>(&Name);return f;} \
    ddlaStatus_t Name Params {count(CounterName);auto f=real_##Name();return f?f Args:ddlaStatus_t::DDLA_STATUS_INTERNAL_ERROR;}
DDLA_TRACE(ddlaMalloc,DdlaAlloc,(const DdlaHandle_t& h,void** p,std::size_t n),(h,p,n))
DDLA_TRACE(ddlaFree,DdlaFree,(const DdlaHandle_t& h,void* p),(h,p))
DDLA_TRACE(ddlaMallocHost,PinnedAlloc,(const DdlaHandle_t& h,void** p,std::size_t n),(h,p,n))
DDLA_TRACE(ddlaFreeHost,DdlaFree,(const DdlaHandle_t& h,void* p),(h,p))
using ddlaMemcpyFn=ddlaStatus_t(*)(const DdlaHandle_t&,void*,const void*,std::size_t);
static ddlaMemcpyFn real_ddlaMemcpy(){static ddlaMemcpyFn f=next_own<ddlaMemcpyFn>(&ddlaMemcpy);return f;}
ddlaStatus_t ddlaMemcpy(const DdlaHandle_t& h,void* dst,const void* src,std::size_t n)
{
    count(Copies);auto f=real_ddlaMemcpy();
    const auto result=f?f(h,dst,src,n):ddlaStatus_t::DDLA_STATUS_INTERNAL_ERROR;
    auto expected=src;
    if(result==ddlaStatus_t::DDLA_STATUS_SUCCESS && src!=nullptr &&
       n==failing_copy_bytes.load(std::memory_order_relaxed) &&
       failing_copy_source.compare_exchange_strong(expected,nullptr,std::memory_order_acq_rel)) {
        copy_faults.fetch_add(1,std::memory_order_relaxed);
        return ddlaStatus_t::DDLA_STATUS_INTERNAL_ERROR;
    }
    return result;
}
DDLA_TRACE(ddlaMemcpy2D,Copies,(const DdlaHandle_t& h,void* dst,std::size_t dp,const void* src,std::size_t sp,std::size_t w,std::size_t ht),(h,dst,dp,src,sp,w,ht))
DDLA_TRACE(ddlaMemset,Copies,(const DdlaHandle_t& h,void* dst,int value,std::size_t n),(h,dst,value,n))
DDLA_TRACE(ddlaSynchronize,Syncs,(const DdlaHandle_t& h),(h))
DDLA_TRACE(ddlaDeviceSynchronize,Syncs,(const DdlaHandle_t& h),(h))
DDLA_TRACE(ddlaEventSynchronize,Syncs,(const DdlaHandle_t& h,ddlaEvent_t event),(h,event))
#undef DDLA_TRACE
}

// Queries must not enter a collective, even when only one rank asks for its
// lengths. Return an ordinary MPI error before entering PMPI so the negative
// control fails immediately instead of hanging at an unmatched collective.
#define TRACE_MPI(Name,Params,Args) extern "C" int MPI_##Name Params \
{if(collective())return MPI_ERR_OTHER;return PMPI_##Name Args;}
TRACE_MPI(Allreduce,(const void* s,void* r,int n,MPI_Datatype t,MPI_Op op,MPI_Comm c),(s,r,n,t,op,c))
TRACE_MPI(Reduce,(const void* s,void* r,int n,MPI_Datatype t,MPI_Op op,int root,MPI_Comm c),(s,r,n,t,op,root,c))
TRACE_MPI(Bcast,(void* b,int n,MPI_Datatype t,int root,MPI_Comm c),(b,n,t,root,c))
TRACE_MPI(Barrier,(MPI_Comm c),(c))
TRACE_MPI(Allgather,(const void* s,int sn,MPI_Datatype st,void* r,int rn,MPI_Datatype rt,MPI_Comm c),(s,sn,st,r,rn,rt,c))
TRACE_MPI(Allgatherv,(const void* s,int sn,MPI_Datatype st,void* r,const int* rn,const int* d,MPI_Datatype rt,MPI_Comm c),(s,sn,st,r,rn,d,rt,c))
TRACE_MPI(Alltoall,(const void* s,int sn,MPI_Datatype st,void* r,int rn,MPI_Datatype rt,MPI_Comm c),(s,sn,st,r,rn,rt,c))
TRACE_MPI(Alltoallv,(const void* s,const int* sn,const int* sd,MPI_Datatype st,void* r,const int* rn,const int* rd,MPI_Datatype rt,MPI_Comm c),(s,sn,sd,st,r,rn,rd,rt,c))
TRACE_MPI(Gather,(const void* s,int sn,MPI_Datatype st,void* r,int rn,MPI_Datatype rt,int root,MPI_Comm c),(s,sn,st,r,rn,rt,root,c))
TRACE_MPI(Gatherv,(const void* s,int sn,MPI_Datatype st,void* r,const int* rn,const int* d,MPI_Datatype rt,int root,MPI_Comm c),(s,sn,st,r,rn,d,rt,root,c))
TRACE_MPI(Scatter,(const void* s,int sn,MPI_Datatype st,void* r,int rn,MPI_Datatype rt,int root,MPI_Comm c),(s,sn,st,r,rn,rt,root,c))
TRACE_MPI(Scatterv,(const void* s,const int* sn,const int* d,MPI_Datatype st,void* r,int rn,MPI_Datatype rt,int root,MPI_Comm c),(s,sn,d,st,r,rn,rt,root,c))
TRACE_MPI(Reduce_scatter,(const void* s,void* r,const int* n,MPI_Datatype t,MPI_Op op,MPI_Comm c),(s,r,n,t,op,c))
TRACE_MPI(Scan,(const void* s,void* r,int n,MPI_Datatype t,MPI_Op op,MPI_Comm c),(s,r,n,t,op,c))
TRACE_MPI(Exscan,(const void* s,void* r,int n,MPI_Datatype t,MPI_Op op,MPI_Comm c),(s,r,n,t,op,c))
#define TRACE_MPI_RESOURCE(Name,Params,Args) extern "C" int MPI_##Name Params \
{if(resource() || collective())return MPI_ERR_OTHER;return PMPI_##Name Args;}
TRACE_MPI_RESOURCE(Comm_dup,(MPI_Comm c,MPI_Comm* out),(c,out))
TRACE_MPI_RESOURCE(Comm_split,(MPI_Comm c,int color,int key,MPI_Comm* out),(c,color,key,out))
TRACE_MPI_RESOURCE(Comm_split_type,(MPI_Comm c,int type,int key,MPI_Info info,MPI_Comm* out),(c,type,key,info,out))
TRACE_MPI_RESOURCE(Comm_free,(MPI_Comm* c),(c))
TRACE_MPI_RESOURCE(Win_allocate_shared,(MPI_Aint n,int disp,MPI_Info info,MPI_Comm c,void* base,MPI_Win* w),(n,disp,info,c,base,w))
TRACE_MPI_RESOURCE(Win_allocate,(MPI_Aint n,int disp,MPI_Info info,MPI_Comm c,void* base,MPI_Win* w),(n,disp,info,c,base,w))
TRACE_MPI_RESOURCE(Win_create,(void* base,MPI_Aint n,int disp,MPI_Info info,MPI_Comm c,MPI_Win* w),(base,n,disp,info,c,w))
TRACE_MPI_RESOURCE(Win_free,(MPI_Win* w),(w))
#undef TRACE_MPI_RESOURCE
#define TRACE_MPI_LOCAL_RESOURCE(Name,Params,Args) extern "C" int MPI_##Name Params \
{if(resource())return MPI_ERR_OTHER;return PMPI_##Name Args;}
TRACE_MPI_LOCAL_RESOURCE(Type_contiguous,(int n,MPI_Datatype oldtype,MPI_Datatype* out),(n,oldtype,out))
TRACE_MPI_LOCAL_RESOURCE(Type_vector,(int n,int block,int stride,MPI_Datatype oldtype,MPI_Datatype* out),(n,block,stride,oldtype,out))
TRACE_MPI_LOCAL_RESOURCE(Type_create_resized,(MPI_Datatype oldtype,MPI_Aint lb,MPI_Aint extent,MPI_Datatype* out),(oldtype,lb,extent,out))
TRACE_MPI_LOCAL_RESOURCE(Type_commit,(MPI_Datatype* type),(type))
TRACE_MPI_LOCAL_RESOURCE(Type_free,(MPI_Datatype* type),(type))
#undef TRACE_MPI_LOCAL_RESOURCE
// Point-to-point operations are included in the communication counter too;
// an accidental query send/receive must fail before waiting for another rank.
TRACE_MPI(Send,(const void* b,int n,MPI_Datatype t,int peer,int tag,MPI_Comm c),(b,n,t,peer,tag,c))
TRACE_MPI(Recv,(void* b,int n,MPI_Datatype t,int peer,int tag,MPI_Comm c,MPI_Status* status),(b,n,t,peer,tag,c,status))
TRACE_MPI(Sendrecv,(const void* s,int sn,MPI_Datatype st,int dest,int stag,void* r,int rn,MPI_Datatype rt,int src,int rtag,MPI_Comm c,MPI_Status* status),(s,sn,st,dest,stag,r,rn,rt,src,rtag,c,status))
TRACE_MPI(Isend,(const void* b,int n,MPI_Datatype t,int peer,int tag,MPI_Comm c,MPI_Request* request),(b,n,t,peer,tag,c,request))
TRACE_MPI(Irecv,(void* b,int n,MPI_Datatype t,int peer,int tag,MPI_Comm c,MPI_Request* request),(b,n,t,peer,tag,c,request))
TRACE_MPI(Waitall,(int n,MPI_Request* requests,MPI_Status* statuses),(n,requests,statuses))
#undef TRACE_MPI

#define TRACE_QR(Name,T) \
    using Name##Fn=void(*)(const int*,const int*,T*,const int*,T*,T*,const int*,int*); \
    static Name##Fn real_##Name(){static Name##Fn f=next<Name##Fn>(#Name);return f;} \
    extern "C" void Name(const int* m,const int* n,T* a,const int* ld,T* tau,T* work,const int* lw,int* info) \
    {if(lw && *lw==-1)count(VendorQuery,true);auto f=real_##Name();if(f)f(m,n,a,ld,tau,work,lw,info);else if(info)*info=-1000;}
TRACE_QR(sgeqrf_,float)
TRACE_QR(dgeqrf_,double)
TRACE_QR(cgeqrf_,std::complex<float>)
TRACE_QR(zgeqrf_,std::complex<double>)
TRACE_QR(sgeqlf_,float)
TRACE_QR(dgeqlf_,double)
TRACE_QR(cgeqlf_,std::complex<float>)
TRACE_QR(zgeqlf_,std::complex<double>)
#undef TRACE_QR

#if defined(DDLA_USE_CUDA)
#define TRACE_CUDA(Name,CounterName,Managed,Params,Args) \
    using Name##Fn=cudaError_t(*) Params; \
    static Name##Fn real_##Name(){static Name##Fn f=next<Name##Fn>(#Name);return f;} \
    extern "C" cudaError_t Name Params {count(CounterName,Managed);auto f=real_##Name();return f?f Args:cudaErrorUnknown;}
TRACE_CUDA(cudaMalloc,RawGpuAlloc,true,(void** p,std::size_t n),(p,n))
TRACE_CUDA(cudaMallocAsync,RawGpuAlloc,true,(void** p,std::size_t n,cudaStream_t s),(p,n,s))
TRACE_CUDA(cudaMallocHost,PinnedAlloc,true,(void** p,std::size_t n),(p,n))
TRACE_CUDA(cudaHostAlloc,PinnedAlloc,true,(void** p,std::size_t n,unsigned flags),(p,n,flags))
#define TRACE_CUDA_RESOURCE(Name,Params,Args) \
    using Name##Fn=cudaError_t(*) Params; \
    static Name##Fn real_##Name(){static Name##Fn f=next<Name##Fn>(#Name);return f;} \
    extern "C" cudaError_t Name Params {if(resource())return cudaErrorNotPermitted;auto f=real_##Name();return f?f Args:cudaErrorUnknown;}
TRACE_CUDA_RESOURCE(cudaEventCreate,(cudaEvent_t* e),(e))
TRACE_CUDA_RESOURCE(cudaEventCreateWithFlags,(cudaEvent_t* e,unsigned flags),(e,flags))
TRACE_CUDA_RESOURCE(cudaEventDestroy,(cudaEvent_t e),(e))
TRACE_CUDA_RESOURCE(cudaStreamCreate,(cudaStream_t* s),(s))
TRACE_CUDA_RESOURCE(cudaStreamCreateWithFlags,(cudaStream_t* s,unsigned flags),(s,flags))
TRACE_CUDA_RESOURCE(cudaStreamCreateWithPriority,(cudaStream_t* s,unsigned flags,int priority),(s,flags,priority))
TRACE_CUDA_RESOURCE(cudaStreamDestroy,(cudaStream_t s),(s))
#undef TRACE_CUDA_RESOURCE
#define TRACE_CURAND_RESOURCE(Name,Params,Args) \
    using Name##Fn=curandStatus_t(*) Params; \
    static Name##Fn real_##Name(){static Name##Fn f=next<Name##Fn>(#Name);return f;} \
    extern "C" curandStatus_t Name Params {if(resource())return CURAND_STATUS_INTERNAL_ERROR;auto f=real_##Name();return f?f Args:CURAND_STATUS_INTERNAL_ERROR;}
TRACE_CURAND_RESOURCE(curandCreateGenerator,(curandGenerator_t* g,curandRngType_t type),(g,type))
TRACE_CURAND_RESOURCE(curandDestroyGenerator,(curandGenerator_t g),(g))
#undef TRACE_CURAND_RESOURCE
// Runtime frees are preserved, with DDLA's public free carrying the managed
// free counter. Vendor pool management is outside the scratch contract.
using CudaFreeFn=cudaError_t(*)(void*);
static CudaFreeFn real_cudaFree(){static auto f=next<CudaFreeFn>("cudaFree");return f;}
static CudaFreeFn real_cudaFreeHost(){static auto f=next<CudaFreeFn>("cudaFreeHost");return f;}
using CudaFreeAsyncFn=cudaError_t(*)(void*,cudaStream_t);
static CudaFreeAsyncFn real_cudaFreeAsync(){static auto f=next<CudaFreeAsyncFn>("cudaFreeAsync");return f;}
extern "C" cudaError_t cudaFree(void* p){auto f=real_cudaFree();return f?f(p):cudaErrorUnknown;}
extern "C" cudaError_t cudaFreeHost(void* p){auto f=real_cudaFreeHost();return f?f(p):cudaErrorUnknown;}
extern "C" cudaError_t cudaFreeAsync(void* p,cudaStream_t s){auto f=real_cudaFreeAsync();return f?f(p,s):cudaErrorUnknown;}
TRACE_CUDA(cudaMemcpy,Copies,true,(void* d,const void* s,std::size_t n,cudaMemcpyKind k),(d,s,n,k))
TRACE_CUDA(cudaMemcpyAsync,Copies,true,(void* d,const void* s,std::size_t n,cudaMemcpyKind k,cudaStream_t st),(d,s,n,k,st))
TRACE_CUDA(cudaMemcpy2D,Copies,true,(void* d,std::size_t dp,const void* s,std::size_t sp,std::size_t w,std::size_t h,cudaMemcpyKind k),(d,dp,s,sp,w,h,k))
TRACE_CUDA(cudaMemcpy2DAsync,Copies,true,(void* d,std::size_t dp,const void* s,std::size_t sp,std::size_t w,std::size_t h,cudaMemcpyKind k,cudaStream_t st),(d,dp,s,sp,w,h,k,st))
TRACE_CUDA(cudaMemset,Copies,true,(void* d,int value,std::size_t n),(d,value,n))
TRACE_CUDA(cudaMemsetAsync,Copies,true,(void* d,int value,std::size_t n,cudaStream_t st),(d,value,n,st))
TRACE_CUDA(cudaLaunchKernel,Launches,false,(const void* fn,dim3 grid,dim3 block,void** args,std::size_t shared,cudaStream_t st),(fn,grid,block,args,shared,st))
TRACE_CUDA(cudaStreamSynchronize,Syncs,true,(cudaStream_t st),(st))
TRACE_CUDA(cudaDeviceSynchronize,Syncs,true,(void),())
TRACE_CUDA(cudaEventSynchronize,Syncs,true,(cudaEvent_t event),(event))
#undef TRACE_CUDA

#define TRACE_SOLVER(Name,Params,Args) \
    using Name##Fn=cusolverStatus_t(*) Params; \
    static Name##Fn real_##Name(){static Name##Fn f=next<Name##Fn>(#Name);return f;} \
    extern "C" cusolverStatus_t Name Params {count(VendorQuery,true);auto f=real_##Name();return f?f Args:CUSOLVER_STATUS_INTERNAL_ERROR;}
#define TRACE_SOLVER_TYPE(P,T) \
 TRACE_SOLVER(cusolverDn##P##getrf_bufferSize,(cusolverDnHandle_t h,int m,int n,T* a,int ld,int* lw),(h,m,n,a,ld,lw)) \
 TRACE_SOLVER(cusolverDn##P##geqrf_bufferSize,(cusolverDnHandle_t h,int m,int n,T* a,int ld,int* lw),(h,m,n,a,ld,lw)) \
 TRACE_SOLVER(cusolverDn##P##potrf_bufferSize,(cusolverDnHandle_t h,cublasFillMode_t u,int n,T* a,int ld,int* lw),(h,u,n,a,ld,lw))
TRACE_SOLVER_TYPE(S,float)
TRACE_SOLVER_TYPE(D,double)
TRACE_SOLVER_TYPE(C,cuComplex)
TRACE_SOLVER_TYPE(Z,cuDoubleComplex)
#undef TRACE_SOLVER_TYPE
#undef TRACE_SOLVER

#if defined(DDLA_USE_CCL)
// Native communication must fail before enqueueing when a single rank is
// probing query purity, just like the PMPI diagnostic path above.
#define TRACE_NCCL(Name,Setup,Params,Args) \
    using Name##Fn=ncclResult_t(*) Params; \
    static Name##Fn real_##Name(){static Name##Fn f=next<Name##Fn>(#Name);return f;} \
    extern "C" ncclResult_t Name Params { \
        if(Setup)count(ResourceSetup); \
        if(collective())return ncclInvalidUsage; \
        auto f=real_##Name();return f?f Args:ncclInternalError;}
TRACE_NCCL(ncclAllReduce,false,(const void* s,void* r,std::size_t n,ncclDataType_t t,ncclRedOp_t op,ncclComm_t c,cudaStream_t st),(s,r,n,t,op,c,st))
TRACE_NCCL(ncclBcast,false,(void* b,std::size_t n,ncclDataType_t t,int root,ncclComm_t c,cudaStream_t st),(b,n,t,root,c,st))
TRACE_NCCL(ncclBroadcast,false,(const void* s,void* r,std::size_t n,ncclDataType_t t,int root,ncclComm_t c,cudaStream_t st),(s,r,n,t,root,c,st))
TRACE_NCCL(ncclReduce,false,(const void* s,void* r,std::size_t n,ncclDataType_t t,ncclRedOp_t op,int root,ncclComm_t c,cudaStream_t st),(s,r,n,t,op,root,c,st))
TRACE_NCCL(ncclAllGather,false,(const void* s,void* r,std::size_t n,ncclDataType_t t,ncclComm_t c,cudaStream_t st),(s,r,n,t,c,st))
TRACE_NCCL(ncclReduceScatter,false,(const void* s,void* r,std::size_t n,ncclDataType_t t,ncclRedOp_t op,ncclComm_t c,cudaStream_t st),(s,r,n,t,op,c,st))
TRACE_NCCL(ncclSend,false,(const void* b,std::size_t n,ncclDataType_t t,int peer,ncclComm_t c,cudaStream_t st),(b,n,t,peer,c,st))
TRACE_NCCL(ncclRecv,false,(void* b,std::size_t n,ncclDataType_t t,int peer,ncclComm_t c,cudaStream_t st),(b,n,t,peer,c,st))
TRACE_NCCL(ncclGroupStart,false,(void),())
TRACE_NCCL(ncclGroupEnd,false,(void),())
TRACE_NCCL(ncclCommInitRank,true,(ncclComm_t* c,int n,ncclUniqueId id,int rank),(c,n,id,rank))
TRACE_NCCL(ncclCommInitAll,true,(ncclComm_t* c,int n,const int* devices),(c,n,devices))
#undef TRACE_NCCL
#endif
#endif

namespace {
void prewarm()
{
    ResolutionGuard guard;
    const auto n=readlink("/proc/self/exe",executable,sizeof(executable)-1);
    if(n>=0)executable[n]='\0';
    Dl_info info{};
    if(dladdr(reinterpret_cast<void*>(&workspace_trace_begin),&info))own_path=info.dli_fname;
    void* frames[32];(void)backtrace(frames,32);
    (void)ddla::real_ddlaMalloc();(void)ddla::real_ddlaFree();
    (void)ddla::real_ddlaMallocHost();(void)ddla::real_ddlaFreeHost();
    (void)ddla::real_ddlaMemcpy();(void)ddla::real_ddlaMemcpy2D();
    (void)ddla::real_ddlaMemset();(void)ddla::real_ddlaSynchronize();
    (void)ddla::real_ddlaDeviceSynchronize();(void)ddla::real_ddlaEventSynchronize();
    (void)real_sgeqrf_();(void)real_dgeqrf_();(void)real_cgeqrf_();(void)real_zgeqrf_();
    (void)real_sgeqlf_();(void)real_dgeqlf_();(void)real_cgeqlf_();(void)real_zgeqlf_();
#if defined(DDLA_USE_CUDA)
#define WARM(Name) (void)real_##Name()
    WARM(cudaMalloc);WARM(cudaMallocAsync);WARM(cudaMallocHost);WARM(cudaHostAlloc);
    WARM(cudaEventCreate);WARM(cudaEventCreateWithFlags);WARM(cudaEventDestroy);
    WARM(cudaStreamCreate);WARM(cudaStreamCreateWithFlags);WARM(cudaStreamCreateWithPriority);WARM(cudaStreamDestroy);
    WARM(curandCreateGenerator);WARM(curandDestroyGenerator);
    WARM(cudaFree);WARM(cudaFreeAsync);WARM(cudaFreeHost);
    WARM(cudaMemcpy);WARM(cudaMemcpyAsync);WARM(cudaMemcpy2D);WARM(cudaMemcpy2DAsync);
    WARM(cudaMemset);WARM(cudaMemsetAsync);WARM(cudaLaunchKernel);
    WARM(cudaStreamSynchronize);WARM(cudaDeviceSynchronize);WARM(cudaEventSynchronize);
#define WARM_SOLVER(P) WARM(cusolverDn##P##getrf_bufferSize);WARM(cusolverDn##P##geqrf_bufferSize);WARM(cusolverDn##P##potrf_bufferSize)
    WARM_SOLVER(S);WARM_SOLVER(D);WARM_SOLVER(C);WARM_SOLVER(Z);
#undef WARM_SOLVER
#if defined(DDLA_USE_CCL)
    WARM(ncclAllReduce);WARM(ncclBcast);WARM(ncclBroadcast);WARM(ncclReduce);
    WARM(ncclAllGather);WARM(ncclReduceScatter);WARM(ncclSend);WARM(ncclRecv);
    WARM(ncclGroupStart);WARM(ncclGroupEnd);WARM(ncclCommInitRank);WARM(ncclCommInitAll);
#endif
#undef WARM
#endif
}
}

extern "C" void workspace_trace_begin(int query)
{
    enabled.store(false,std::memory_order_release);
    prewarm();
    for(auto& counter:counters)counter.store(0,std::memory_order_relaxed);
    query_mode.store(query!=0,std::memory_order_relaxed);
    enabled.store(true,std::memory_order_release);
}
extern "C" void workspace_trace_end(workspace_trace_snapshot* result)
{
    enabled.store(false,std::memory_order_release);
    if(!result)return;
    result->ddla_alloc=counters[DdlaAlloc].load(std::memory_order_relaxed);
    result->ddla_free=counters[DdlaFree].load(std::memory_order_relaxed);
    result->pinned_alloc=counters[PinnedAlloc].load(std::memory_order_relaxed);
    result->raw_gpu_alloc=counters[RawGpuAlloc].load(std::memory_order_relaxed);
    result->host_new=counters[HostNew].load(std::memory_order_relaxed);
    result->vendor_query=counters[VendorQuery].load(std::memory_order_relaxed);
    result->copies=counters[Copies].load(std::memory_order_relaxed);
    result->launches=counters[Launches].load(std::memory_order_relaxed);
    result->syncs=counters[Syncs].load(std::memory_order_relaxed);
    result->collectives=counters[Collectives].load(std::memory_order_relaxed);
    result->resource_setup=counters[ResourceSetup].load(std::memory_order_relaxed);
}

extern "C" void workspace_trace_fail_memcpy_after_upload(const void* source,uint64_t bytes)
{
    failing_copy_source.store(nullptr,std::memory_order_release);
    copy_faults.store(0,std::memory_order_relaxed);
    failing_copy_bytes.store(bytes,std::memory_order_relaxed);
    failing_copy_source.store(source,std::memory_order_release);
}
extern "C" uint64_t workspace_trace_memcpy_faults(void)
{return copy_faults.load(std::memory_order_relaxed);}
