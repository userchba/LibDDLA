// Standalone CUDA 11.8 query-purity/equivalence probe; no DDLA linkage needed.
// nvcc -std=c++17 tests/test_vendor_workspace_query.cpp -lcusolver -lcublas -ldl -o query_probe
#include "workspace_test_common.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cuComplex.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <utility>

static void gpu(cudaError_t s) { if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
static void solver(cusolverStatus_t s) { if (s != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("cuSOLVER failure"); }
#define QUERY_WRAPPERS(T, PREFIX) \
static int qr(cusolverDnHandle_t h,int m,int n,T* a,int lda) { int l=0; solver(cusolverDn##PREFIX##geqrf_bufferSize(h,m,n,a,lda,&l)); return l; } \
static int lu(cusolverDnHandle_t h,int m,int n,T* a,int lda) { int l=0; solver(cusolverDn##PREFIX##getrf_bufferSize(h,m,n,a,lda,&l)); return l; } \
static int chol(cusolverDnHandle_t h,int n,T* a,int lda) { int l=0; solver(cusolverDn##PREFIX##potrf_bufferSize(h,CUBLAS_FILL_MODE_LOWER,n,a,lda,&l)); return l; }
QUERY_WRAPPERS(float,S)
QUERY_WRAPPERS(double,D)
QUERY_WRAPPERS(cuComplex,C)
QUERY_WRAPPERS(cuDoubleComplex,Z)
#undef QUERY_WRAPPERS

template <typename T> void check(cusolverDnHandle_t h, cudaStream_t stream, int m, int n)
{
    const int lda = m + 7;
    const auto bytes = std::size_t(lda) * n * sizeof(T);
    T *original=nullptr, *compact=nullptr, *tiny=nullptr;
    gpu(cudaMalloc(reinterpret_cast<void**>(&original), bytes));
    gpu(cudaMalloc(reinterpret_cast<void**>(&compact), std::size_t(m)*n*sizeof(T)));
    gpu(cudaMalloc(reinterpret_cast<void**>(&tiny), sizeof(T)));
    gpu(cudaMemsetAsync(original, 0xA5, bytes, stream));
    gpu(cudaMemsetAsync(compact, 0x5A, std::size_t(m)*n*sizeof(T), stream));
    gpu(cudaMemsetAsync(tiny, 0x39, sizeof(T), stream));
    gpu(cudaStreamSynchronize(stream));
    std::vector<unsigned char> before(bytes), after(bytes);
    gpu(cudaMemcpy(before.data(), original, bytes, cudaMemcpyDeviceToHost));
    for (int repeat=0; repeat<3; ++repeat) {
        // Probe CUDA 11.8 behavior for nonnull query pointers independently
        // of future scratch size/alignment. tiny is a valid one-scalar
        // allocation; original+1 remains T-aligned but loses base alignment.
        // This is observed vendor behavior, not a guarantee for future CUDA.
        workspace_test::Scope query(true);
        const int q0=qr(h,m,n,original,m), q1=qr(h,m,n,compact,m);
        const int qt=qr(h,m,n,tiny,m), qo=qr(h,m,n,original+1,m);
        const int l0=lu(h,m,n,original,m), l1=lu(h,m,n,compact,m);
        const int lt=lu(h,m,n,tiny,m), lo=lu(h,m,n,original+1,m);
        const int c0=chol(h,std::min(m,n),original,m), c1=chol(h,std::min(m,n),compact,m);
        const int ct=chol(h,std::min(m,n),tiny,m), co=chol(h,std::min(m,n),original+1,m);
        const int padded=qr(h,m,n,original,lda);
        const auto snapshot=query.finish();
        if (snapshot.ddla_alloc || snapshot.ddla_free || snapshot.raw_gpu_alloc || snapshot.host_new ||
            snapshot.copies || snapshot.launches || snapshot.syncs || snapshot.collectives)
            throw std::runtime_error("vendor workspace query allocated managed storage or executed work");
        if (workspace_test::available() && snapshot.vendor_query==0)
            throw std::runtime_error("vendor workspace query diagnostic negative control failed");
        if (q0!=q1 || q0!=qt || q0!=qo || l0!=l1 || l0!=lt || l0!=lo ||
            c0!=c1 || c0!=ct || c0!=co)
            throw std::runtime_error("pointer-size/alignment-dependent workspace query");
        // Record padded and compact queries separately; lda-dependent sizing
        // is legal and must remain represented by the actual query schedule.
        std::printf("sizeof(T)=%zu m=%d n=%d QR=%d LU=%d POTRF=%d padded_QR=%d repeat=%d\n",
            sizeof(T),m,n,q0,l0,c0,padded,repeat);
    }
    gpu(cudaStreamSynchronize(stream));
    gpu(cudaMemcpy(after.data(), original, bytes, cudaMemcpyDeviceToHost));
    if (before!=after) throw std::runtime_error("workspace query modified input");
    unsigned char tiny_after[sizeof(T)];
    gpu(cudaMemcpy(tiny_after, tiny, sizeof(T), cudaMemcpyDeviceToHost));
    for (auto byte : tiny_after)
        if (byte!=0x39) throw std::runtime_error("workspace query modified one-scalar query allocation");
    gpu(cudaFree(tiny));
    gpu(cudaFree(compact)); gpu(cudaFree(original));
}
int main()
{
    try {
        cudaStream_t stream{}; cusolverDnHandle_t handle{};
        gpu(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        solver(cusolverDnCreate(&handle)); solver(cusolverDnSetStream(handle,stream));
        for (auto shape : {std::pair<int,int>{257,31}, {31,257}, {128,128}, {1,1}}) {
            check<float>(handle,stream,shape.first,shape.second);
            check<double>(handle,stream,shape.first,shape.second);
            check<cuComplex>(handle,stream,shape.first,shape.second);
            check<cuDoubleComplex>(handle,stream,shape.first,shape.second);
        }
        solver(cusolverDnDestroy(handle)); gpu(cudaStreamDestroy(stream));
        std::puts("vendor workspace query purity/equivalence PASS");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
