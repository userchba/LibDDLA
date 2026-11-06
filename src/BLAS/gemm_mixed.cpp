#include "blas_impl.h"
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "plan_util.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include "gemm_mixed_cpu.h"

namespace ddla {
namespace {
char mixed_trans(char c)
{
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (c != 'N' && c != 'T' && c != 'C')
        throw std::invalid_argument("gemm(real,complex): invalid transpose flag");
    return c;
}

// Bound temporary storage, including for tall matrices. Each complex column
// becomes two real columns: A [Br Bi] = [Re(AB) Im(AB)]. No zero imaginary A
// is materialized. This follows BLIS's mixed-domain use of real GEMM, without
// requiring BLIS or changing the configured BLAS provider.
constexpr int mixed_panel = 512;

#if DDLA_HAS_GPU
// std::complex<R> arrays are specified to expose interleaved real/imaginary R
// elements through reinterpret_cast<R*>. Kernels use that scalar view, avoiding
// a std::complex/thrust::complex alias or by-value type pun.
template <typename R>
__device__ void mixed_store(R* C, std::size_t p, R re, R im,
                            R ar, R ai, R br, R bi)
{
    R cr = ar*re - ai*im, ci = ar*im + ai*re;
    if (br != R(0) || bi != R(0)) {
        const R oldr = C[2*p], oldi = C[2*p+1];
        cr += br*oldr - bi*oldi;
        ci += br*oldi + bi*oldr;
    }
    C[2*p] = cr;
    C[2*p+1] = ci;
}

template <typename R, bool Batched = false, bool Complex = true>
__global__ void mixed_scale(int m, int n, R br, R bi, R* C, int ldc, void* const* array = nullptr)
{
    if constexpr (Batched) C=static_cast<R*>(array[blockIdx.z]);
    for (std::size_t p = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
         p < std::size_t(m)*n; p += std::size_t(blockDim.x)*gridDim.x)
        if constexpr (Complex)
            mixed_store(C, p%m + (p/m)*ldc, R(0), R(0), R(0), R(0), br, bi);
        else {
            const std::size_t index=p%m+(p/m)*ldc;
            C[index]=br==R(0) ? R(0) : br*C[index];
        }
}

template <typename R, bool Batched = false>
__global__ void mixed_pack(char tb, int k, int nj, int j0, const R* B, int ldb, R* P, const void* const* array = nullptr, std::size_t stride = 0)
{
    if constexpr (Batched) { B=static_cast<const R*>(array[blockIdx.z]); P+=blockIdx.z*stride; }
    for (std::size_t p = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
         p < std::size_t(k)*nj; p += std::size_t(blockDim.x)*gridDim.x) {
        const std::size_t l = p%k, j = p/k;
        const std::size_t b = tb == 'N' ? l + (j0+j)*ldb : j0+j + l*ldb;
        P[l + 2*j*k] = B[2*b];
        P[l + (2*j+1)*k] = tb == 'C' ? -B[2*b+1] : B[2*b+1];
    }
}

template <typename R, bool Batched = false>
__global__ void mixed_combine(int m, int nj, const R* D, R ar, R ai, R br, R bi,
                              R* C, int ldc, void* const* array = nullptr, int j0 = 0, std::size_t stride = 0)
{
    if constexpr (Batched) { C=static_cast<R*>(array[blockIdx.z])+2*std::size_t(j0)*ldc; D+=blockIdx.z*stride; }
    for (std::size_t p = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
         p < std::size_t(m)*nj; p += std::size_t(blockDim.x)*gridDim.x) {
        const std::size_t i = p%m, j = p/m;
        mixed_store(C, i+j*ldc, D[i+2*j*m], D[i+(2*j+1)*m], ar, ai, br, bi);
    }
}

__device__ float mixed_fma(float a, float b, float c) { return fmaf(a,b,c); }
__device__ double mixed_fma(double a, double b, double c) { return fma(a,b,c); }

// MAGMA-style shared tiles + register microtiles. Each A load updates both
// real and imaginary accumulators, using two real FMAs per complex output.
// Transpose/conjugation are specialized outside the reduction loop. Tail
// loads are zero-filled, and every thread reaches both tile barriers.
// Each block walks k serially, so its load latency must be hidden inside
// the block: the next k tile is loaded into registers while the current one
// is computed. Small 32x16 tiles give enough blocks to fill the GPU at these
// sizes (64x32 tiles without the prefetch were 2.4-3.6x slower on a V100).
template <typename R, bool TA, bool TB, bool CONJ, bool Batched = false,
          bool TransposeC = false, int BM = 32, int BN = 16>
__global__ void mixed_tiled(int m, int n, int k, const R* A, int lda,
                            const R* B, int ldb, R ar, R ai, R br, R bi,
                            R* C, int ldc, const void* const* aa = nullptr, const void* const* ba = nullptr,
                            void* const* ca = nullptr)
{
    if constexpr (Batched) {
        A=static_cast<const R*>(aa[blockIdx.z]); B=static_cast<const R*>(ba[blockIdx.z]);
        C=static_cast<R*>(ca[blockIdx.z]);
    }
    constexpr int BK=16, TX=16, TY=8, NT=TX*TY;
    constexpr int MI=BM/TX, MJ=BN/TY, NA=BM*BK/NT, NB=BN*BK/NT;
    static_assert(BM*BK%NT==0 && BN*BK%NT==0, "every thread loads whole tiles");
    __shared__ R as[BK][BM+1], bs_r[BN][BK+1], bs_i[BN][BK+1];
    const int tx=threadIdx.x, ty=threadIdx.y, tid=ty*TX+tx;
    R real[MJ][MI] = {}, imag[MJ][MI] = {}, ra[NA], rr[NB], ri[NB];
    const std::size_t row = std::size_t(blockIdx.x)*BM;
    const std::size_t col = std::size_t(blockIdx.y)*BN;
    auto fetch = [&](std::size_t l0) {
        #pragma unroll
        for (int s=0; s<NA; ++s) {
            // Coalesce in the physical input's first dimension.
            const int p=tid+s*NT, i=TA ? p/BK : p%BM, l=TA ? p%BK : p/BM;
            ra[s] = row+i<std::size_t(m) && l0+l<std::size_t(k)
                ? A[TA ? l0+l+(row+i)*lda : row+i+(l0+l)*lda] : R(0);
        }
        #pragma unroll
        for (int s=0; s<NB; ++s) {
            const int p=tid+s*NT, j=TB ? p%BN : p/BK, l=TB ? p/BN : p%BK;
            R xr=0, xi=0;
            if (col+j<std::size_t(n) && l0+l<std::size_t(k)) {
                const auto q=TB ? col+j+(l0+l)*ldb : l0+l+(col+j)*ldb;
                xr=B[2*q]; xi=B[2*q+1];
            }
            rr[s]=xr; ri[s]=CONJ ? -xi : xi;
        }
    };
    fetch(0);
    for (std::size_t l0=0; l0<std::size_t(k); l0+=BK) {
        #pragma unroll
        for (int s=0; s<NA; ++s) {
            const int p=tid+s*NT;
            as[TA ? p%BK : p/BM][TA ? p/BK : p%BM]=ra[s];
        }
        #pragma unroll
        for (int s=0; s<NB; ++s) {
            const int p=tid+s*NT, j=TB ? p%BN : p/BK, l=TB ? p/BN : p%BK;
            bs_r[j][l]=rr[s]; bs_i[j][l]=ri[s];
        }
        __syncthreads();
        if (l0+BK<std::size_t(k)) fetch(l0+BK);
        #pragma unroll
        for (int l=0; l<BK; ++l) {
            R a[MI];
            #pragma unroll
            for (int i=0; i<MI; ++i) a[i]=as[l][tx+i*TX];
            #pragma unroll
            for (int j=0; j<MJ; ++j) {
                const R xr=bs_r[ty+j*TY][l], xi=bs_i[ty+j*TY][l];
                #pragma unroll
                for (int i=0; i<MI; ++i) {
                    real[j][i]=mixed_fma(a[i],xr,real[j][i]);
                    imag[j][i]=mixed_fma(a[i],xi,imag[j][i]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j=0; j<MJ; ++j)
        #pragma unroll
        for (int i=0; i<MI; ++i)
            if (row+tx+i*TX<std::size_t(m) && col+ty+j*TY<std::size_t(n))
                mixed_store(C, TransposeC ? col+ty+j*TY+(row+tx+i*TX)*ldc : row+tx+i*TX+(col+ty+j*TY)*ldc,
                             real[j][i], imag[j][i], ar, ai, br, bi);
}

template <typename R, bool ComplexLeft>
__global__ void mixed_batch_pointers(const void* const* aa,const void* const* ba,int ldb,
    char tb,int j0,R* data,std::size_t packed,std::size_t stride,
    const R** ap,const R** bp,R** cp,int count)
{
    const int q=blockIdx.x*blockDim.x+threadIdx.x;
    if (q>=count) return;
    R* p=data+q*stride;
    if constexpr (ComplexLeft) {
        ap[q]=p;
        bp[q]=static_cast<const R*>(ba[q])+(tb=='N'?std::size_t(j0)*ldb:std::size_t(j0));
    } else { ap[q]=static_cast<const R*>(aa[q]); bp[q]=p; }
    cp[q]=p+packed;
}

unsigned mixed_blocks(std::size_t n)
{
    return static_cast<unsigned>(std::min<std::size_t>((n-1)/256+1,65535));
}

template <typename R>
void mixed_gpu(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
               std::complex<R> alpha, const R* A, int lda,
               const std::complex<R>* B, int ldb, std::complex<R> beta,
               std::complex<R>* C, int ldc, bool product, void* scratch, [[maybe_unused]] std::size_t bytes)
{
    const gpuStream_t stream=static_cast<gpuStream_t>(h->stream);
    const R ar=alpha.real(), ai=alpha.imag(), br=beta.real(), bi=beta.imag();
    const R* b=reinterpret_cast<const R*>(B);
    R* c=reinterpret_cast<R*>(C);
    if (!product) {
        mixed_scale<<<mixed_blocks(std::size_t(m)*n),256,0,stream>>>(m,n,br,bi,c,ldc);
        GPU_CHECK(gpuGetLastError());
        return;
    }
    // The fused path avoids packing/allocations for launch-bound products.
    // The grid-y bound is part of dispatch, not a limit on the public API.
    if (m <= 256 && n <= 256 && k <= 256) {
        const dim3 grid((m-1)/32+1,(n-1)/16+1), block(16,8);
#define MIXED_LAUNCH(TA,TB,CJ) mixed_tiled<R,TA,TB,CJ><<<grid,block,0,stream>>>(m,n,k,A,lda,b,ldb,ar,ai,br,bi,c,ldc)
        if (ta=='N') {
            if (tb=='N') { MIXED_LAUNCH(false,false,false); }
            else if (tb=='T') { MIXED_LAUNCH(false,true,false); }
            else { MIXED_LAUNCH(false,true,true); }
        } else {
            if (tb=='N') { MIXED_LAUNCH(true,false,false); }
            else if (tb=='T') { MIXED_LAUNCH(true,true,false); }
            else { MIXED_LAUNCH(true,true,true); }
        }
#undef MIXED_LAUNCH
        GPU_CHECK(gpuGetLastError());
        return;
    }
    const int nb=std::min(n,mixed_panel);
    // [ packed panel of B (k x 2nb) | real product (m x 2nb) ], one region; the caller's storage remains live
    // through the queued operations.
    char* w=static_cast<char*>(scratch);
    R* const work=detail::take<R>(w,std::size_t(2)*k*nb+std::size_t(2)*m*nb);
    assert(static_cast<std::size_t>(w-static_cast<char*>(scratch))<=bytes);
    R* result=work+std::size_t(2)*k*nb;
    for (int j0=0; j0<n;) {
        const int nj=std::min(nb,n-j0);
        mixed_pack<<<mixed_blocks(std::size_t(k)*nj),256,0,stream>>>(tb,k,nj,j0,b,ldb,work);
        GPU_CHECK(gpuGetLastError());
        blas_impl::gemm_run<DdlaBackend::GPU,R>(h,ta,'N',m,2*nj,k,R(1),A,lda,work,k,R(0),result,m);
        mixed_combine<<<mixed_blocks(std::size_t(m)*nj),256,0,stream>>>(
            m,nj,result,ar,ai,br,bi,c+std::size_t(2)*j0*ldc,ldc);
        GPU_CHECK(gpuGetLastError());
        j0+=nj;
    }
}

// Complex-by-real products pack [Re op(A); Im op(A)] into one real 2m x k
// matrix (alpha and the ta=='C' conjugation folded in) and drive one real
// GEMM per column panel; D = P2 * op(B) has the real block on rows [0,m) and
// the imaginary block on [m,2m).
template <typename R, bool Batched = false>
__global__ void mixed_pack_cr(char ta, int m, int k, const R* A, int lda,
                              R ar, R ai, R* P2, const void* const* array = nullptr, std::size_t stride = 0)
{
    if constexpr (Batched) { A=static_cast<const R*>(array[blockIdx.z]); P2+=blockIdx.z*stride; }
    for (std::size_t p = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
         p < std::size_t(m)*k; p += std::size_t(blockDim.x)*gridDim.x) {
        const std::size_t i = p%std::size_t(m), l = p/std::size_t(m);
        const std::size_t a = ta == 'N' ? i + l*std::size_t(lda) : l + i*std::size_t(lda);
        const R br = A[2*a];
        const R bi = ta == 'C' ? -A[2*a+1] : A[2*a+1];
        P2[i + l*std::size_t(2*m)] = ar*br - ai*bi;
        P2[m + i + l*std::size_t(2*m)] = ar*bi + ai*br;
    }
}

template <typename R, bool Batched = false>
__global__ void mixed_combine_cr(int m, int nj, const R* D, R br, R bi,
                                 R* C, int ldc, void* const* array = nullptr, int j0 = 0, std::size_t stride = 0)
{
    if constexpr (Batched) { C=static_cast<R*>(array[blockIdx.z])+2*std::size_t(j0)*ldc; D+=blockIdx.z*stride; }
    for (std::size_t p = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
         p < std::size_t(m)*nj; p += std::size_t(blockDim.x)*gridDim.x) {
        const std::size_t i = p%std::size_t(m), j = p/std::size_t(m);
        mixed_store(C, i + j*std::size_t(ldc),
                    D[i + j*std::size_t(2*m)], D[m + i + j*std::size_t(2*m)],
                    R(1), R(0), br, bi);
    }
}

template <typename R>
void mixed_gpu_cr(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
                  std::complex<R> alpha, const std::complex<R>* A, int lda,
                  const R* B, int ldb, std::complex<R> beta,
                  std::complex<R>* C, int ldc, bool product, void* scratch, [[maybe_unused]] std::size_t bytes)
{
    const gpuStream_t stream=static_cast<gpuStream_t>(h->stream);
    const R ar=alpha.real(), ai=alpha.imag(), br=beta.real(), bi=beta.imag();
    const R* const a=reinterpret_cast<const R*>(A);
    R* const c=reinterpret_cast<R*>(C);
    if (!product) {
        mixed_scale<<<mixed_blocks(std::size_t(m)*n),256,0,stream>>>(m,n,br,bi,c,ldc);
        GPU_CHECK(gpuGetLastError());
        return;
    }
    const int nb=std::min(n,mixed_panel);
    const std::size_t pack_elems=2*std::size_t(m)*k, res_elems=2*std::size_t(m)*nb;
    // [ packed P2 (2m x k) | real product D (2m x nb) ], one region; the caller's storage remains live through the
    // queued operations.
    char* w=static_cast<char*>(scratch);
    R* const p2=detail::take<R>(w,pack_elems+res_elems);
    assert(static_cast<std::size_t>(w-static_cast<char*>(scratch))<=bytes);
    R* const d=p2+pack_elems;
    mixed_pack_cr<<<mixed_blocks(std::size_t(m)*k),256,0,stream>>>(
        ta,m,k,a,lda,ar,ai,p2);
    GPU_CHECK(gpuGetLastError());
    for (int j0=0; j0<n;) {
        const int nj=std::min(nb,n-j0);
        const R* const bj = B + (tb == 'N' ? std::size_t(j0)*ldb : std::size_t(j0));
        blas_impl::gemm_run<DdlaBackend::GPU,R>(h,'N',tb,2*m,nj,k,R(1),p2,2*m,bj,ldb,R(0),d,2*m);
        mixed_combine_cr<<<mixed_blocks(std::size_t(m)*nj),256,0,stream>>>(
            m,nj,d,br,bi,c+std::size_t(2)*j0*ldc,ldc);
        GPU_CHECK(gpuGetLastError());
        j0+=nj;
    }
}
#endif
} // namespace

namespace blas_impl {
// Expose the existing matrix scale kernel to the SCAL interfaces as well as
// GEMM. The complex branch is unchanged, including beta-zero no-read stores.
template <typename T>
void matrix_scal_gpu_run(const DdlaHandle_t& h, int m, int n, const T& alpha,
                         T* a, int lda, T* const* arrays, int count)
{
#if DDLA_HAS_GPU
    if (m<=0 || n<=0 || count<=0 || alpha==T(1)) return;
    using R=typename real_type<T>::type;
    constexpr bool complex=!std::is_arithmetic_v<T>;
    R re,im;
    if constexpr (complex) {re=alpha.real();im=alpha.imag();}
    else {re=alpha;im=R(0);}
    const auto stream=static_cast<gpuStream_t>(h->stream);
    if (!arrays) {
        mixed_scale<R,false,complex><<<mixed_blocks(std::size_t(m)*n),256,0,stream>>>(
            m,n,re,im,reinterpret_cast<R*>(a),lda);
        GPU_CHECK(gpuGetLastError());
        return;
    }
    for (int first=0; first<count;) {
        const int batch=std::min(count-first,65535);
        const dim3 grid(mixed_blocks(std::size_t(m)*n),1,batch);
        mixed_scale<R,true,complex><<<grid,256,0,stream>>>(m,n,re,im,nullptr,lda,
            reinterpret_cast<void* const*>(arrays+first));
        GPU_CHECK(gpuGetLastError());
        first+=batch;
    }
#endif
}

template <typename R>
void mixed_gemmBatched_workspace_size(const DdlaHandle_t& h,int m,int n,int k,
    bool complex_left,int count,std::size_t* bytes)
{
    *bytes=0;
    if (m<=0 || n<=0 || k<=0 || count<=0) return;
    if (h->backend==DdlaBackend::GPU && m<=256 && n<=256) return;
    int depth=k;
    std::size_t parallel=0;
    if(h->backend==DdlaBackend::CPU && m>=1024 && n>=1024 && count<=std::max(1,h->host_threads/2))
        detail::mixed_gemm_cpu_workspace_size<R>(h,m,n,k,complex_left,&parallel);
    const int cap=mixed_gemmBatched_capacity<R>(h->backend==DdlaBackend::CPU?h->host_threads:0,m,n,k,complex_left,count,depth,parallel);
    if(depth==0){*bytes=parallel;return;}
    const std::size_t panel=std::min(n,mixed_panel);
    const std::size_t packed=2*(complex_left ? std::size_t(m)*depth : std::size_t(depth)*panel);
    *bytes=detail::padded_bytes<R>((packed+2*std::size_t(m)*panel)*cap)
        +3*detail::padded_bytes<R*>(cap);
}

template <typename R, bool ComplexLeft>
void mixed_gemmBatched_gpu_run(const DdlaHandle_t& h,char ta,char tb,int m,int n,int k,
    const std::complex<R>& alpha,const void* const* aa,int lda,const void* const* ba,int ldb,
    const std::complex<R>& beta,std::complex<R>* const* ca,int ldc,int count,void* scratch,std::size_t bytes)
{
#if DDLA_HAS_GPU
    const auto stream=static_cast<gpuStream_t>(h->stream);
    auto cc=reinterpret_cast<void* const*>(ca);
    const R ar=alpha.real(),ai=alpha.imag(),br=beta.real(),bi=beta.imag();
    if (alpha==R(0) || k==0) {
        matrix_scal_gpu_run<std::complex<R>>(h,m,n,beta,nullptr,ldc,ca,count);
        return;
    }
    if (m<=256 && n<=256) {
        // For complex-by-real, compute the transposed product B^T*A^T and
        // transpose the stores. Conjugation still belongs to complex A.
        const int rows=ComplexLeft?n:m,cols=ComplexLeft?m:n;
        const bool at=ComplexLeft?tb=='N':ta!='N';
        const bool bt=ComplexLeft?ta=='N':tb!='N';
        const bool cj=ComplexLeft?ta=='C':tb=='C';
        const auto real=ComplexLeft?ba:aa,complex=ComplexLeft?aa:ba;
        for (int first=0;first<count;) {
            const int batch=std::min(count-first,65535);
            // Long reductions benefit from more output accumulators per
            // thread once the batch supplies enough independent blocks.
            const bool wide=k>=512 && std::size_t(batch)*((rows-1)/64+1)*((cols-1)/32+1)>=256;
            const dim3 grid((rows-1)/(wide?64:32)+1,(cols-1)/(wide?32:16)+1,batch),block(16,8);
#define BATCH_LAUNCH(A,B,C) do { \
    if (wide) mixed_tiled<R,A,B,C,true,ComplexLeft,64,32><<<grid,block,0,stream>>>(rows,cols,k,nullptr,ComplexLeft?ldb:lda,nullptr,ComplexLeft?lda:ldb,ar,ai,br,bi,nullptr,ldc,real+first,complex+first,cc+first); \
    else mixed_tiled<R,A,B,C,true,ComplexLeft><<<grid,block,0,stream>>>(rows,cols,k,nullptr,ComplexLeft?ldb:lda,nullptr,ComplexLeft?lda:ldb,ar,ai,br,bi,nullptr,ldc,real+first,complex+first,cc+first); \
} while (false)
            if (!at) {
                if (!bt) {if(cj){BATCH_LAUNCH(false,false,true);}else{BATCH_LAUNCH(false,false,false);}}
                else {if(cj){BATCH_LAUNCH(false,true,true);}else{BATCH_LAUNCH(false,true,false);}}
            } else {
                if (!bt) {if(cj){BATCH_LAUNCH(true,false,true);}else{BATCH_LAUNCH(true,false,false);}}
                else {if(cj){BATCH_LAUNCH(true,true,true);}else{BATCH_LAUNCH(true,true,false);}}
            }
#undef BATCH_LAUNCH
            GPU_CHECK(gpuGetLastError());first+=batch;
        }
        return;
    }
    int depth=k;
    const int panel=std::min(n,mixed_panel),cap=mixed_gemmBatched_capacity<R>(0,m,n,k,ComplexLeft,count,depth);
    const std::size_t packed=2*(ComplexLeft?std::size_t(m)*k:std::size_t(k)*panel);
    const std::size_t stride=packed+2*std::size_t(m)*panel;
    char* cursor=static_cast<char*>(scratch);
    R* data=detail::take<R>(cursor,stride*cap);
    const R** ap=detail::take<const R*>(cursor,cap);
    const R** bp=detail::take<const R*>(cursor,cap);
    R** cp=detail::take<R*>(cursor,cap);
    assert(std::size_t(cursor-static_cast<char*>(scratch))<=bytes);
    for (int first=0;first<count;) {
        const int batch=std::min(cap,count-first);
        if constexpr (ComplexLeft) {
            const dim3 grid(mixed_blocks(std::size_t(m)*k),1,batch);
            mixed_pack_cr<R,true><<<grid,256,0,stream>>>(ta,m,k,nullptr,lda,ar,ai,data,aa+first,stride);
            GPU_CHECK(gpuGetLastError());
        }
        for (int j0=0;j0<n;) {
            const int nj=std::min(panel,n-j0);
            if constexpr (!ComplexLeft) {
                const dim3 grid(mixed_blocks(std::size_t(k)*nj),1,batch);
                mixed_pack<R,true><<<grid,256,0,stream>>>(tb,k,nj,j0,nullptr,ldb,data,ba+first,stride);
                GPU_CHECK(gpuGetLastError());
            }
            mixed_batch_pointers<R,ComplexLeft><<<(batch+255)/256,256,0,stream>>>(
                aa+first,ba+first,ldb,tb,j0,data,packed,stride,ap,bp,cp,batch);
            GPU_CHECK(gpuGetLastError());
            gemmBatched_run<DdlaBackend::GPU,R>(h,ComplexLeft?'N':ta,ComplexLeft?tb:'N',
                ComplexLeft?2*m:m,ComplexLeft?nj:2*nj,k,R(1),ap,ComplexLeft?2*m:lda,
                bp,ComplexLeft?ldb:k,R(0),cp,ComplexLeft?2*m:m,batch);
            const dim3 grid(mixed_blocks(std::size_t(m)*nj),1,batch);
            if constexpr (ComplexLeft)
                mixed_combine_cr<R,true><<<grid,256,0,stream>>>(m,nj,data+packed,br,bi,nullptr,ldc,cc+first,j0,stride);
            else
                mixed_combine<R,true><<<grid,256,0,stream>>>(m,nj,data+packed,ar,ai,br,bi,nullptr,ldc,cc+first,j0,stride);
            GPU_CHECK(gpuGetLastError());j0+=nj;
        }
        first+=batch;
    }
#endif
}
} // namespace blas_impl

namespace blas_impl {
// CPU handles: the worker slots of gemm_mixed_cpu.cpp.  GPU handles: one region [ packed operand | real product ] --
// real-by-complex [ B panel (k x 2nb) | m x 2nb ], complex-by-real [ P2 (2m x k) | 2m x nb ], nb = min(n, 512) --
// and nothing for the fused small real-by-complex kernel (m, n, k <= 256).
template <typename R>
void mixed_gemm_workspace_size(const DdlaHandle_t& h, int m, int n, int k, bool complex_left,
                               std::size_t* backend_bytes)
{
    *backend_bytes = 0;
    if (m <= 0 || n <= 0 || k <= 0) return;
    if (h->backend == DdlaBackend::CPU) {
        detail::mixed_gemm_cpu_workspace_size<R>(h, m, n, k, complex_left, backend_bytes);
        return;
    }
    if (!complex_left && m <= 256 && n <= 256 && k <= 256) return;
    const std::size_t nb = std::min(n, mixed_panel);
    const std::size_t pack = complex_left ? 2 * std::size_t(m) * k : 2 * std::size_t(k) * nb;
    *backend_bytes = detail::padded_bytes<R>(pack + 2 * std::size_t(m) * nb);
}
} // namespace blas_impl

namespace blas_impl {
template <DdlaBackend Backend, typename R>
void gemm_check(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
          const std::complex<R>& alpha, const R* A, int lda,
          const std::complex<R>* B, int ldb, const std::complex<R>& beta,
          std::complex<R>* C, int ldc)
{
    static_assert(std::is_same_v<R,float> || std::is_same_v<R,double>);
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU);
    if (!h) throw std::runtime_error("gemm(real,complex): null handle");
    if (ddlaGetBackend(h) != Backend)
        throw std::runtime_error("gemm(real,complex): template backend does not match handle backend");
    ta=mixed_trans(ta); tb=mixed_trans(tb);
    if (m<0 || n<0 || k<0 || lda<std::max(1,ta=='N'?m:k) ||
        ldb<std::max(1,tb=='N'?k:n) || ldc<std::max(1,m))
        throw std::invalid_argument("gemm(real,complex): invalid dimensions or leading dimension");
    if (m==0 || n==0) return;
    const bool product=k!=0 && alpha!=R(0);
    if (!product && beta==R(1)) return;
    if (!C || (product && (!A || !B)))
        throw std::invalid_argument("gemm(real,complex): null matrix pointer");
    if (ta=='C') ta='T';
}
template <DdlaBackend Backend, typename R>
void gemm_run(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
          const std::complex<R>& alpha, const R* A, int lda,
          const std::complex<R>* B, int ldb, const std::complex<R>& beta,
          std::complex<R>* C, int ldc, void* backend_work, std::size_t backend_bytes)
{
    ta=upper(ta); tb=upper(tb);
    if (m==0 || n==0) return;
    const bool product=k!=0 && alpha!=R(0);
    if (!product && beta==R(1)) return;
    if (ta=='C') ta='T';
    if constexpr (Backend==DdlaBackend::CPU) {
        detail::mixed_gemm_cpu(h,ta,tb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,product,backend_work,backend_bytes);
    } else {
#if DDLA_HAS_GPU
        mixed_gpu(h,ta,tb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,product,backend_work,backend_bytes);
#endif
    }
}
} // namespace blas_impl
template <DdlaBackend Backend, typename R>
void gemm(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
          const std::complex<R>& alpha, const R* A, int lda,
          const std::complex<R>* B, int ldb, const std::complex<R>& beta,
          std::complex<R>* C, int ldc)
{
    blas_impl::gemm_check<Backend,R>(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    std::size_t backend_bytes = 0;
    if (alpha != R(0)) blas_impl::mixed_gemm_workspace_size<R>(h, m, n, k, false, &backend_bytes);
    if constexpr (Backend==DdlaBackend::CPU) {
        // The caller thread's cached scratch, reused across calls (gemm_mixed_cpu.h).
        char* const work=detail::mixed_gemm_cpu_cached_workspace(backend_bytes);
        blas_impl::gemm_run<Backend,R>(h,ta,tb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,work,backend_bytes);
    } else {
        vector<char> work(h, backend_bytes);
        blas_impl::gemm_run<Backend,R>(h,ta,tb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,work.data(),backend_bytes);
    }
}
#define DDLA_MIXED_PHASES(BACKEND,R) \
    template void blas_impl::gemm_check<BACKEND,R>(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k, const std::complex<R>& alpha, const R* A, int lda, const std::complex<R>* B, int ldb, const std::complex<R>& beta, std::complex<R>* C, int ldc); \
    template void blas_impl::gemm_run<BACKEND,R>(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k, const std::complex<R>& alpha, const R* A, int lda, const std::complex<R>* B, int ldb, const std::complex<R>& beta, std::complex<R>* C, int ldc, void* backend_work, std::size_t backend_bytes);
DDLA_MIXED_PHASES(DdlaBackend::CPU,float)
DDLA_MIXED_PHASES(DdlaBackend::CPU,double)
#if DDLA_HAS_GPU
DDLA_MIXED_PHASES(DdlaBackend::GPU,float)
DDLA_MIXED_PHASES(DdlaBackend::GPU,double)
#endif
#undef DDLA_MIXED_PHASES


#define MIXED_INST(B,R) template void gemm<B,R>(const DdlaHandle_t&,char,char,int,int,int, \
    const std::complex<R>&,const R*,int,const std::complex<R>*,int, \
    const std::complex<R>&,std::complex<R>*,int)
MIXED_INST(DdlaBackend::CPU,float);
MIXED_INST(DdlaBackend::CPU,double);
#if DDLA_HAS_GPU
MIXED_INST(DdlaBackend::GPU,float);
MIXED_INST(DdlaBackend::GPU,double);
#endif
#undef MIXED_INST

namespace blas_impl {
template <DdlaBackend Backend, typename R>
void gemm_check(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
          const std::complex<R>& alpha, const std::complex<R>* A, int lda,
          const R* B, int ldb, const std::complex<R>& beta,
          std::complex<R>* C, int ldc)
{
    static_assert(std::is_same_v<R,float> || std::is_same_v<R,double>);
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU);
    if (!h) throw std::runtime_error("gemm(complex,real): null handle");
    if (ddlaGetBackend(h) != Backend)
        throw std::runtime_error("gemm(complex,real): template backend does not match handle backend");
    ta=mixed_trans(ta); tb=mixed_trans(tb);
    if (m<0 || n<0 || k<0 || lda<std::max(1,ta=='N'?m:k) ||
        ldb<std::max(1,tb=='N'?k:n) || ldc<std::max(1,m))
        throw std::invalid_argument("gemm(complex,real): invalid dimensions or leading dimension");
    if (m==0 || n==0) return;
    const bool product=k!=0 && alpha!=R(0);
    if (!product && beta==R(1)) return;
    if (!C || (product && (!A || !B)))
        throw std::invalid_argument("gemm(complex,real): null matrix pointer");
    if (tb=='C') tb='T';
}
template <DdlaBackend Backend, typename R>
void gemm_run(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
          const std::complex<R>& alpha, const std::complex<R>* A, int lda,
          const R* B, int ldb, const std::complex<R>& beta,
          std::complex<R>* C, int ldc, void* backend_work, std::size_t backend_bytes)
{
    ta=upper(ta); tb=upper(tb);
    if (m==0 || n==0) return;
    const bool product=k!=0 && alpha!=R(0);
    if (!product && beta==R(1)) return;
    if (tb=='C') tb='T';
    if constexpr (Backend==DdlaBackend::CPU) {
        detail::mixed_gemm_cpu(h,ta,tb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,product,backend_work,backend_bytes);
    } else {
#if DDLA_HAS_GPU
        mixed_gpu_cr(h,ta,tb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,product,backend_work,backend_bytes);
#endif
    }
}
} // namespace blas_impl
template <DdlaBackend Backend, typename R>
void gemm(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
          const std::complex<R>& alpha, const std::complex<R>* A, int lda,
          const R* B, int ldb, const std::complex<R>& beta,
          std::complex<R>* C, int ldc)
{
    blas_impl::gemm_check<Backend,R>(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    std::size_t backend_bytes = 0;
    if (alpha != R(0)) blas_impl::mixed_gemm_workspace_size<R>(h, m, n, k, true, &backend_bytes);
    if constexpr (Backend==DdlaBackend::CPU) {
        // The caller thread's cached scratch, reused across calls (gemm_mixed_cpu.h).
        char* const work=detail::mixed_gemm_cpu_cached_workspace(backend_bytes);
        blas_impl::gemm_run<Backend,R>(h,ta,tb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,work,backend_bytes);
    } else {
        vector<char> work(h, backend_bytes);
        blas_impl::gemm_run<Backend,R>(h,ta,tb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,work.data(),backend_bytes);
    }
}
#define DDLA_MIXED_PHASES(BACKEND,R) \
    template void blas_impl::gemm_check<BACKEND,R>(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k, const std::complex<R>& alpha, const std::complex<R>* A, int lda, const R* B, int ldb, const std::complex<R>& beta, std::complex<R>* C, int ldc); \
    template void blas_impl::gemm_run<BACKEND,R>(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k, const std::complex<R>& alpha, const std::complex<R>* A, int lda, const R* B, int ldb, const std::complex<R>& beta, std::complex<R>* C, int ldc, void* backend_work, std::size_t backend_bytes);
DDLA_MIXED_PHASES(DdlaBackend::CPU,float)
DDLA_MIXED_PHASES(DdlaBackend::CPU,double)
#if DDLA_HAS_GPU
DDLA_MIXED_PHASES(DdlaBackend::GPU,float)
DDLA_MIXED_PHASES(DdlaBackend::GPU,double)
#endif
#undef DDLA_MIXED_PHASES


#define MIXED_CR_INST(B,R) template void gemm<B,R>(const DdlaHandle_t&,char,char,int,int,int, \
    const std::complex<R>&,const std::complex<R>*,int,const R*,int, \
    const std::complex<R>&,std::complex<R>*,int)
MIXED_CR_INST(DdlaBackend::CPU,float);
MIXED_CR_INST(DdlaBackend::CPU,double);
#if DDLA_HAS_GPU
MIXED_CR_INST(DdlaBackend::GPU,float);
MIXED_CR_INST(DdlaBackend::GPU,double);
#endif
#undef MIXED_CR_INST

template void blas_impl::mixed_gemm_workspace_size<float>(const DdlaHandle_t&,int,int,int,bool,std::size_t*);
template void blas_impl::mixed_gemm_workspace_size<double>(const DdlaHandle_t&,int,int,int,bool,std::size_t*);
template <DdlaBackend Backend,typename R>
void gemmBatched(const DdlaHandle_t& h,char ta,char tb,int m,int n,int k,
    const std::complex<R>& alpha,const R* const* a,int lda,const std::complex<R>* const* b,int ldb,
    const std::complex<R>& beta,std::complex<R>* const* c,int ldc,int count)
{
    using T=std::complex<R>;
    blas_impl::gemmBatched_check<Backend,T>(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,count);
    if (m==0 || n==0 || count==0 || ((alpha==T(0) || k==0) && beta==T(1))) return;
    std::size_t bytes=0;
    if (alpha!=T(0)) blas_impl::mixed_gemmBatched_workspace_size<R>(h,m,n,k,false,count,&bytes);
    vector<char> work(h,bytes);
    blas_impl::gemmBatched_run<Backend,T>(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,count,work.data(),bytes);
}

template <DdlaBackend Backend,typename R>
void gemmBatched(const DdlaHandle_t& h,char ta,char tb,int m,int n,int k,
    const std::complex<R>& alpha,const std::complex<R>* const* a,int lda,const R* const* b,int ldb,
    const std::complex<R>& beta,std::complex<R>* const* c,int ldc,int count)
{
    using T=std::complex<R>;
    blas_impl::gemmBatched_check<Backend,T>(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,count);
    if (m==0 || n==0 || count==0 || ((alpha==T(0) || k==0) && beta==T(1))) return;
    std::size_t bytes=0;
    if (alpha!=T(0)) blas_impl::mixed_gemmBatched_workspace_size<R>(h,m,n,k,true,count,&bytes);
    vector<char> work(h,bytes);
    blas_impl::gemmBatched_run<Backend,T>(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,count,work.data(),bytes);
}

template void gemmBatched<DdlaBackend::CPU,float>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<float>&,const float* const*,int,const std::complex<float>* const*,int,const std::complex<float>&,std::complex<float>* const*,int,int);
template void gemmBatched<DdlaBackend::CPU,float>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<float>&,const std::complex<float>* const*,int,const float* const*,int,const std::complex<float>&,std::complex<float>* const*,int,int);
template void gemmBatched<DdlaBackend::CPU,double>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<double>&,const double* const*,int,const std::complex<double>* const*,int,const std::complex<double>&,std::complex<double>* const*,int,int);
template void gemmBatched<DdlaBackend::CPU,double>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<double>&,const std::complex<double>* const*,int,const double* const*,int,const std::complex<double>&,std::complex<double>* const*,int,int);
#if DDLA_HAS_GPU
template void gemmBatched<DdlaBackend::GPU,float>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<float>&,const float* const*,int,const std::complex<float>* const*,int,const std::complex<float>&,std::complex<float>* const*,int,int);
template void gemmBatched<DdlaBackend::GPU,float>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<float>&,const std::complex<float>* const*,int,const float* const*,int,const std::complex<float>&,std::complex<float>* const*,int,int);
template void blas_impl::mixed_gemmBatched_gpu_run<float,false>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<float>&,const void* const*,int,const void* const*,int,const std::complex<float>&,std::complex<float>* const*,int,int,void*,std::size_t);
template void blas_impl::mixed_gemmBatched_gpu_run<float,true>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<float>&,const void* const*,int,const void* const*,int,const std::complex<float>&,std::complex<float>* const*,int,int,void*,std::size_t);
template void gemmBatched<DdlaBackend::GPU,double>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<double>&,const double* const*,int,const std::complex<double>* const*,int,const std::complex<double>&,std::complex<double>* const*,int,int);
template void gemmBatched<DdlaBackend::GPU,double>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<double>&,const std::complex<double>* const*,int,const double* const*,int,const std::complex<double>&,std::complex<double>* const*,int,int);
template void blas_impl::mixed_gemmBatched_gpu_run<double,false>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<double>&,const void* const*,int,const void* const*,int,const std::complex<double>&,std::complex<double>* const*,int,int,void*,std::size_t);
template void blas_impl::mixed_gemmBatched_gpu_run<double,true>(const DdlaHandle_t&,char,char,int,int,int,const std::complex<double>&,const void* const*,int,const void* const*,int,const std::complex<double>&,std::complex<double>* const*,int,int,void*,std::size_t);
#endif
template void blas_impl::mixed_gemmBatched_workspace_size<float>(const DdlaHandle_t&,int,int,int,bool,int,std::size_t*);
template void blas_impl::mixed_gemmBatched_workspace_size<double>(const DdlaHandle_t&,int,int,int,bool,int,std::size_t*);
#if DDLA_HAS_GPU
#define MATRIX_SCAL_GPU(T) template void blas_impl::matrix_scal_gpu_run<T>(const DdlaHandle_t&,int,int,const T&,T*,int,T* const*,int)
MATRIX_SCAL_GPU(float);
MATRIX_SCAL_GPU(double);
MATRIX_SCAL_GPU(std::complex<float>);
MATRIX_SCAL_GPU(std::complex<double>);
#undef MATRIX_SCAL_GPU
#endif
} // namespace ddla
