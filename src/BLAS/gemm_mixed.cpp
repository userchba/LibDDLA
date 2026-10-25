#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>
#include "ddla_connector.h"
#include "ddla_stream_impl.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

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
template <typename R>
std::size_t mixed_workspace(int m, int k, int nb)
{
    const std::size_t rows = std::size_t(m) + k;
    if (rows > std::numeric_limits<std::size_t>::max() / (2 * sizeof(R) * nb))
        throw std::overflow_error("gemm(real,complex): workspace size overflow");
    return 2 * rows * nb;
}

#if DDLA_HAS_CPU
template <typename R>
void mixed_cpu(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
               std::complex<R> alpha, const R* A, int lda,
               const std::complex<R>* B, int ldb, std::complex<R> beta,
               std::complex<R>* C, int ldc, bool product)
{
    if (!product) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                auto& c = C[i + std::size_t(j)*ldc];
                c = beta == R(0) ? std::complex<R>(0) : beta*c;
            }
        return;
    }
    const int nb = std::min(n, mixed_panel);
    std::vector<R> work(mixed_workspace<R>(m, k, nb));
    R* const packed = work.data();
    R* const result = packed + std::size_t(2)*k*nb;
    for (int j0 = 0; j0 < n;) {
        const int nj = std::min(nb, n-j0);
        for (int j = 0; j < nj; ++j)
            for (int l = 0; l < k; ++l) {
                const auto b = tb == 'N' ? B[l + std::size_t(j0+j)*ldb]
                                        : B[j0+j + std::size_t(l)*ldb];
                packed[l + std::size_t(2*j)*k] = b.real();
                packed[l + std::size_t(2*j+1)*k] = tb == 'C' ? -b.imag() : b.imag();
            }
        gemm<DdlaBackend::CPU, R>(h, ta, 'N', m, 2*nj, k,
                                   R(1), A, lda, packed, k, R(0), result, m);
        for (int j = 0; j < nj; ++j)
            for (int i = 0; i < m; ++i) {
                auto& c = C[i + std::size_t(j0+j)*ldc];
                const auto z = alpha * std::complex<R>(result[i + std::size_t(2*j)*m],
                                                       result[i + std::size_t(2*j+1)*m]);
                c = beta == R(0) ? z : z + beta*c;
            }
        j0 += nj;
    }
}
#endif

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

template <typename R>
__global__ void mixed_scale(int m, int n, R br, R bi, R* C, int ldc)
{
    for (std::size_t p = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
         p < std::size_t(m)*n; p += std::size_t(blockDim.x)*gridDim.x)
        mixed_store(C, p%m + (p/m)*ldc, R(0), R(0), R(0), R(0), br, bi);
}

template <typename R>
__global__ void mixed_pack(char tb, int k, int nj, int j0, const R* B, int ldb, R* P)
{
    for (std::size_t p = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
         p < std::size_t(k)*nj; p += std::size_t(blockDim.x)*gridDim.x) {
        const std::size_t l = p%k, j = p/k;
        const std::size_t b = tb == 'N' ? l + (j0+j)*ldb : j0+j + l*ldb;
        P[l + 2*j*k] = B[2*b];
        P[l + (2*j+1)*k] = tb == 'C' ? -B[2*b+1] : B[2*b+1];
    }
}

template <typename R>
__global__ void mixed_combine(int m, int nj, const R* D, R ar, R ai, R br, R bi,
                              R* C, int ldc)
{
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
template <typename R, bool TA, bool TB, bool CONJ>
__global__ void mixed_tiled(int m, int n, int k, const R* A, int lda,
                            const R* B, int ldb, R ar, R ai, R br, R bi,
                            R* C, int ldc)
{
    constexpr int BM=32, BN=16, BK=16, TX=16, TY=8, NT=TX*TY;
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
                mixed_store(C, row+tx+i*TX+(col+ty+j*TY)*ldc,
                             real[j][i], imag[j][i], ar, ai, br, bi);
}

unsigned mixed_blocks(std::size_t n)
{
    return static_cast<unsigned>(std::min<std::size_t>((n-1)/256+1,65535));
}

template <typename R>
void mixed_gpu(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
               std::complex<R> alpha, const R* A, int lda,
               const std::complex<R>* B, int ldb, std::complex<R> beta,
               std::complex<R>* C, int ldc, bool product)
{
    const gpuStream_t stream=h->stream;
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
    R* work=nullptr;
    GPU_CHECK(gpuMallocAsync(reinterpret_cast<void**>(&work),
                             mixed_workspace<R>(m,k,nb)*sizeof(R),stream));
    // One owner and one release on success or failure; frees are ordered after
    // the last queued use. The catch preserves the original launch/BLAS error.
    try {
        R* result=work+std::size_t(2)*k*nb;
        for (int j0=0; j0<n;) {
            const int nj=std::min(nb,n-j0);
            mixed_pack<<<mixed_blocks(std::size_t(k)*nj),256,0,stream>>>(tb,k,nj,j0,b,ldb,work);
            GPU_CHECK(gpuGetLastError());
            gemm<DdlaBackend::GPU,R>(h,ta,'N',m,2*nj,k,R(1),A,lda,work,k,R(0),result,m);
            mixed_combine<<<mixed_blocks(std::size_t(m)*nj),256,0,stream>>>(
                m,nj,result,ar,ai,br,bi,c+std::size_t(2)*j0*ldc,ldc);
            GPU_CHECK(gpuGetLastError());
            j0+=nj;
        }
    } catch (...) {
        (void)gpuFreeAsync(work,stream);
        throw;
    }
    GPU_CHECK(gpuFreeAsync(work,stream));
}
#endif
} // namespace

template <DdlaBackend Backend, typename R>
void gemm(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
          const std::complex<R>& alpha, const R* A, int lda,
          const std::complex<R>* B, int ldb, const std::complex<R>& beta,
          std::complex<R>* C, int ldc)
{
    static_assert(std::is_same_v<R,float> || std::is_same_v<R,double>);
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU);
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
    if constexpr (Backend==DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        mixed_cpu(h,ta,tb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,product);
#endif
    } else {
#if DDLA_HAS_GPU
        mixed_gpu(h,ta,tb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,product);
#endif
    }
}

#define MIXED_INST(B,R) template void gemm<B,R>(const DdlaHandle_t&,char,char,int,int,int, \
    const std::complex<R>&,const R*,int,const std::complex<R>*,int, \
    const std::complex<R>&,std::complex<R>*,int)
#if DDLA_HAS_CPU
MIXED_INST(DdlaBackend::CPU,float);
MIXED_INST(DdlaBackend::CPU,double);
#endif
#if DDLA_HAS_GPU
MIXED_INST(DdlaBackend::GPU,float);
MIXED_INST(DdlaBackend::GPU,double);
#endif
#undef MIXED_INST
} // namespace ddla
