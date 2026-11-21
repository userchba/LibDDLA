// Test-only vendor boundary. Preload after the two diagnostic interposers.
// It executes the REAL cuBLAS Dgemm and checks no numerical result itself.
#include "allocation_attribution_probe_api.h"
#include <cublas_v2.h>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <dlfcn.h>
#include <vector>
namespace {
using Dgemm=decltype(&cublasDgemm_v2);
std::atomic<std::size_t> controlled_bytes{0};
std::atomic<unsigned> allocations{0},forwards{0};
Dgemm real_dgemm()
{
    static Dgemm result=[] {
        void* symbol=dlsym(RTLD_NEXT,"cublasDgemm_v2");Dgemm value=nullptr;
        static_assert(sizeof(value)==sizeof(symbol),"dlsym ABI");
        std::memcpy(&value,&symbol,sizeof(value));return value;
    }();
    return result;
}
__attribute__((noinline)) void vendor_allocation(std::size_t bytes)
{
    std::vector<std::byte> payload;payload.resize(bytes);
    if(bytes){payload.front()=std::byte{0x35};payload.back()=std::byte{0x71};}
    asm volatile("" : : "g"(payload.data()),"g"(payload.size()) : "memory");
    allocations.fetch_add(1);
}
}
extern "C" int staging_vendor_probe_prepare(){return real_dgemm()?1:0;}
extern "C" void staging_vendor_probe_arm(std::size_t bytes)
{allocations.store(0);forwards.store(0);controlled_bytes.store(bytes);}
extern "C" unsigned staging_vendor_probe_allocations(){return allocations.load();}
extern "C" unsigned staging_vendor_probe_forwards(){return forwards.load();}
extern "C" cublasStatus_t CUBLASWINAPI cublasDgemm_v2(cublasHandle_t handle,
    cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,const double* alpha,
    const double* a,int lda,const double* b,int ldb,const double* beta,double* c,int ldc)
{
    const auto bytes=controlled_bytes.exchange(0);
    if(bytes)vendor_allocation(bytes);
    const auto next=real_dgemm();if(!next)return CUBLAS_STATUS_INTERNAL_ERROR;
    forwards.fetch_add(1);
    return next(handle,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}
