#include "blas_impl.h"
#include "plan_util.h"
#include <ddla/ddla.h>
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "helpers.h"
#include "require_backend.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ddla {
namespace {

// Columns of C per CPU product panel; the panel is n x min(n, trtrm_panel_width).
constexpr int trtrm_panel_width = 256;

#if DDLA_GPU_VENDOR
// Retain these two-triangular-input kernels: the measured packed vendor
// TRMM composition was 1.5-4.9x slower for n=1024..4096 on V100.
// std::complex arrays may be accessed as their interleaved real components.
// Do not dereference a thrust::complex pointer: its alignment can be stricter
// than std::complex's, even though both have the same size and representation.
template <typename T> struct Global {
    __device__ static T load(const void* p, std::size_t i) { return static_cast<const T*>(p)[i]; }
    __device__ static void store(void* p, std::size_t i, T x) { static_cast<T*>(p)[i] = x; }
};
template <typename R> struct Global<thrust::complex<R>> {
    __device__ static thrust::complex<R> load(const void* p, std::size_t i) {
        const R* v = static_cast<const R*>(p) + 2*i;
        return {v[0], v[1]};
    }
    __device__ static void store(void* p, std::size_t i, thrust::complex<R> x) {
        R* v = static_cast<R*>(p) + 2*i;
        v[0] = x.real(); v[1] = x.imag();
    }
};

__device__ float madd(float a, float b, float c) { return fmaf(a, b, c); }
__device__ double madd(double a, double b, double c) { return fma(a, b, c); }
template <typename R>
__device__ thrust::complex<R> madd(thrust::complex<R> a, thrust::complex<R> b,
                                 thrust::complex<R> c)
{
    return {madd(-a.imag(), b.imag(), madd(a.real(), b.real(), c.real())),
            madd(a.imag(), b.real(), madd(a.real(), b.imag(), c.imag()))};
}

// thrust::conj has no real-scalar overloads; real transposes never conjugate.
__device__ float conj_if(int c, float v) { (void)c; return v; }
__device__ double conj_if(int c, double v) { (void)c; return v; }
template <typename R>
__device__ thrust::complex<R> conj_if(int c, thrust::complex<R> v)
{
    return c ? thrust::conj(v) : v;
}

// op(X)(r,c) for a triangular operand stored in one triangle: zero outside
// the triangle, index-swapped under a transpose flag, conjugated under 'C'.
// Check both effective coordinates before swapping the address for a transpose.
template <bool Upper, typename T>
__device__ T load_tri(const void* m, int ld, int r, int c, int n,
                      int trans, int conj)
{
    if (r >= n || c >= n || (Upper ? r > c : c > r)) return T(0);
    const std::size_t index = trans ? c + std::size_t(r) * ld : r + std::size_t(c) * ld;
    const T v = Global<T>::load(m, index);
    return conj_if(conj, v);
}

template <bool Upper, typename T, bool ReadC>
__global__ void scale_triangle(int n, T beta, void* c, int ldc)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (Upper ? (i <= j && j < n) : (j <= i && i < n)) {
        const std::size_t p = i + std::size_t(j) * ldc;
        if constexpr (ReadC) Global<T>::store(c, p, beta * Global<T>::load(c, p));
        else Global<T>::store(c, p, T(0));
    }
}

// Full-matrix variant of the alpha == 0 scaling path for mixed-triangle
// products, whose result is a full C.
template <typename T, bool ReadC>
__global__ void scale_full(int n, T beta, void* c, int ldc)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        const std::size_t p = i + std::size_t(j) * ldc;
        if constexpr (ReadC) Global<T>::store(c, p, beta * Global<T>::load(c, p));
        else Global<T>::store(c, p, T(0));
    }
}

// Each block owns one selected triangular output tile, with a register tile per thread.
// Shared panels are column-major; adjacent x lanes read rows. Prefetch keeps
// the next panel in registers while the current shared panel is multiplied.
// The opposite input triangle is never loaded, including diagonal panels.
template <bool Upper, typename T, bool ReadC, int Tile, int K, int BX, int BY, bool Prefetch>
__global__ void trtrm_triangle(int n, T alpha, const void* __restrict__ a, int lda,
                          const void* __restrict__ b, int ldb,
                          T beta, void* __restrict__ c, int ldc)
{
    constexpr int Threads = BX * BY, RM = Tile / BX, RN = Tile / BY, Loads = (Tile * K + Threads - 1) / Threads;
    static_assert(Tile % BX == 0 && Tile % BY == 0, "register tile must divide the output tile");
    static_assert(!Prefetch || (Tile * K) % Threads == 0, "prefetch panels must fill complete load groups");
    const unsigned long long tile = blockIdx.x;
    int col = int((sqrt(8.0 * double(tile) + 1.0) - 1.0) * 0.5);
    while (static_cast<unsigned long long>(col + 1) * (col + 2) / 2 <= tile) ++col;
    while (static_cast<unsigned long long>(col) * (col + 1) / 2 > tile) --col;
    const int row = int(tile - static_cast<unsigned long long>(col) * (col + 1) / 2);
    // Reverse both block indices to enumerate lower tiles by output column.
    const int last = (n - 1) / Tile;
    const int row0 = (Upper ? row : last - row) * Tile;
    const int col0 = (Upper ? col : last - col) * Tile;
    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * BX + tx;
    __shared__ T sa[K][Tile];
    __shared__ T sb[Tile][K + 1];
    T acc[RN][RM] = {};
    T next_a[Loads], next_b[Loads];
    const int kbegin = Upper ? row0 : col0;
    const int kend = min(n, (Upper ? col0 : row0) + Tile);
    // The first panel must be in shared memory before entering the pipeline.
    if constexpr (Prefetch) {
        #pragma unroll
        for (int q = tid; q < Tile * K; q += Threads) {
            const int r = q % Tile, k = q / Tile;
            const int i = row0 + r, kk = kbegin + k;
            sa[k][r] = (Upper ? (i <= kk && kk < n) : (kk <= i && i < n)) ? Global<T>::load(a, i + std::size_t(kk) * lda) : T(0);
            const int br = q % K, bc = q / K;
            const int bk = kbegin + br, j = col0 + bc;
            sb[bc][br] = (Upper ? (bk <= j && j < n) : (j <= bk && bk < n)) ? Global<T>::load(b, bk + std::size_t(j) * ldb) : T(0);
        }
    }
    for (int k0 = kbegin; k0 < kend; k0 += K) {
        if constexpr (!Prefetch) {
            #pragma unroll
            for (int q = tid; q < Tile * K; q += Threads) {
                const int r = q % Tile, k = q / Tile;
                const int i = row0 + r, kk = k0 + k;
                sa[k][r] = (Upper ? (i <= kk && kk < n) : (kk <= i && i < n)) ? Global<T>::load(a, i + std::size_t(kk) * lda) : T(0);
                const int br = q % K, bc = q / K;
                const int bk = k0 + br, j = col0 + bc;
                sb[bc][br] = (Upper ? (bk <= j && j < n) : (j <= bk && bk < n)) ? Global<T>::load(b, bk + std::size_t(j) * ldb) : T(0);
            }
        }
        __syncthreads();
        if constexpr (Prefetch) {
            if (k0 + K < kend) {
                #pragma unroll
                for (int l = 0; l < Loads; ++l) {
                    const int q = tid + l * Threads;
                    const int r = q % Tile, k = q / Tile;
                    const int i = row0 + r, kk = k0 + K + k;
                    next_a[l] = (Upper ? (i <= kk && kk < n) : (kk <= i && i < n)) ? Global<T>::load(a, i + std::size_t(kk) * lda) : T(0);
                    const int br = q % K, bc = q / K;
                    const int bk = k0 + K + br, j = col0 + bc;
                    next_b[l] = (Upper ? (bk <= j && j < n) : (j <= bk && bk < n)) ? Global<T>::load(b, bk + std::size_t(j) * ldb) : T(0);
                }
            }
        }
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            T ar[RM], br[RN];
            #pragma unroll
            for (int r = 0; r < RM; ++r) ar[r] = sa[k][tx + r * BX];
            #pragma unroll
            for (int s = 0; s < RN; ++s) br[s] = sb[ty + s * BY][k];
            #pragma unroll
            for (int s = 0; s < RN; ++s)
                #pragma unroll
                for (int r = 0; r < RM; ++r) acc[s][r] = madd(ar[r], br[s], acc[s][r]);
        }
        __syncthreads();
        if constexpr (Prefetch) {
            if (k0 + K < kend) {
                #pragma unroll
                for (int l = 0; l < Loads; ++l) {
                    const int q = tid + l * Threads;
                    sa[q / Tile][q % Tile] = next_a[l];
                    sb[q / K][q % K] = next_b[l];
                }
            }
        }
    }
    #pragma unroll
    for (int s = 0; s < RN; ++s) {
        const int j = col0 + ty + s * BY;
        #pragma unroll
        for (int r = 0; r < RM; ++r) {
            const int i = row0 + tx + r * BX;
            if (Upper ? (i <= j && j < n) : (j <= i && i < n)) {
                const std::size_t p = i + std::size_t(j) * ldc;
                T value = alpha * acc[s][r];
                if constexpr (ReadC) value += beta * Global<T>::load(c, p);
                Global<T>::store(c, p, value);
            }
        }
    }
}
template <bool Upper, typename D, int Tile, int K, int BX, int BY, bool Prefetch>
void launch(gpuStream_t stream, int n, D al, const void* a, int lda, const void* b, int ldb, D be, void* c, int ldc) {
    const std::uint64_t nt = (n - 1) / Tile + 1;
    const std::uint64_t blocks = nt * (nt + 1) / 2;
    if (be == D(0)) trtrm_triangle<Upper,D,false,Tile,K,BX,BY,Prefetch><<<unsigned(blocks),dim3(BX,BY),0,stream>>>(n,al,a,lda,b,ldb,be,c,ldc);
    else trtrm_triangle<Upper,D,true,Tile,K,BX,BY,Prefetch><<<unsigned(blocks),dim3(BX,BY),0,stream>>>(n,al,a,lda,b,ldb,be,c,ldc);
}
// General mixed-triangle / transposed product.  UpA/UpB are the EFFECTIVE
// triangles of op(A)/op(B) (storage triangle xor transpose flag); the result
// is triangular when they match and a full matrix when they differ.  The
// transpose and conjugate flags only remap the load address / value, so they
// stay runtime arguments.  Triangular results reuse the triangular tile
// enumeration and k pairing of trtrm_triangle; mixed results run a full tile
// grid where every tile sums the k corner that both triangle masks admit:
// U x L sums k in [max(row0, col0), n), L x U sums k in [0, min ends).
template <bool UpA, bool UpB, typename T, bool ReadC, int Tile, int K, int BX, int BY, bool Prefetch>
__global__ void trtrm_general(int n, T alpha, const void* __restrict__ a, int lda, int ta, int ca,
                              const void* __restrict__ b, int ldb, int tb, int cb,
                              T beta, void* __restrict__ c, int ldc)
{
    constexpr bool Tri = UpA == UpB;
    constexpr int Threads = BX * BY, RM = Tile / BX, RN = Tile / BY, Loads = (Tile * K + Threads - 1) / Threads;
    static_assert(Tile % BX == 0 && Tile % BY == 0, "register tile must divide the output tile");
    static_assert(!Prefetch || (Tile * K) % Threads == 0, "prefetch panels must fill complete load groups");
    const unsigned long long tile = blockIdx.x;
    const int last = (n - 1) / Tile;
    int row0, col0;
    if constexpr (Tri) {
        int col = int((sqrt(8.0 * double(tile) + 1.0) - 1.0) * 0.5);
        while (static_cast<unsigned long long>(col + 1) * (col + 2) / 2 <= tile) ++col;
        while (static_cast<unsigned long long>(col) * (col + 1) / 2 > tile) --col;
        const int row = int(tile - static_cast<unsigned long long>(col) * (col + 1) / 2);
        // Reverse both block indices to enumerate lower tiles by output column.
        row0 = (UpA ? row : last - row) * Tile;
        col0 = (UpA ? col : last - col) * Tile;
    } else {
        const int nt = last + 1;
        row0 = int(tile / nt) * Tile;
        col0 = int(tile % nt) * Tile;
    }
    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * BX + tx;
    __shared__ T sa[K][Tile];
    __shared__ T sb[Tile][K + 1];
    T acc[RN][RM] = {};
    T next_a[Loads], next_b[Loads];
    int kbegin, kend;
    if constexpr (Tri) {
        kbegin = UpA ? row0 : col0;
        kend = min(n, (UpA ? col0 : row0) + Tile);
    } else if constexpr (UpA) {
        kbegin = max(row0, col0);
        kend = n;
    } else {
        kbegin = 0;
        kend = min(min(n, row0 + Tile), col0 + Tile);
    }
    // The first panel must be in shared memory before entering the pipeline.
    if constexpr (Prefetch) {
        #pragma unroll
        for (int q = tid; q < Tile * K; q += Threads) {
            const int r = q % Tile, k = q / Tile;
            const int i = row0 + r, kk = kbegin + k;
            sa[k][r] = load_tri<UpA,T>(a, lda, i, kk, n, ta, ca);
            const int br = q % K, bc = q / K;
            const int bk = kbegin + br, j = col0 + bc;
            sb[bc][br] = load_tri<UpB,T>(b, ldb, bk, j, n, tb, cb);
        }
    }
    for (int k0 = kbegin; k0 < kend; k0 += K) {
        if constexpr (!Prefetch) {
            #pragma unroll
            for (int q = tid; q < Tile * K; q += Threads) {
                const int r = q % Tile, k = q / Tile;
                const int i = row0 + r, kk = k0 + k;
                sa[k][r] = load_tri<UpA,T>(a, lda, i, kk, n, ta, ca);
                const int br = q % K, bc = q / K;
                const int bk = k0 + br, j = col0 + bc;
                sb[bc][br] = load_tri<UpB,T>(b, ldb, bk, j, n, tb, cb);
            }
        }
        __syncthreads();
        if constexpr (Prefetch) {
            if (k0 + K < kend) {
                #pragma unroll
                for (int l = 0; l < Loads; ++l) {
                    const int q = tid + l * Threads;
                    const int r = q % Tile, k = q / Tile;
                    const int i = row0 + r, kk = k0 + K + k;
                    next_a[l] = load_tri<UpA,T>(a, lda, i, kk, n, ta, ca);
                    const int br = q % K, bc = q / K;
                    const int bk = k0 + K + br, j = col0 + bc;
                    next_b[l] = load_tri<UpB,T>(b, ldb, bk, j, n, tb, cb);
                }
            }
        }
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            T ar[RM], br[RN];
            #pragma unroll
            for (int r = 0; r < RM; ++r) ar[r] = sa[k][tx + r * BX];
            #pragma unroll
            for (int s = 0; s < RN; ++s) br[s] = sb[ty + s * BY][k];
            #pragma unroll
            for (int s = 0; s < RN; ++s)
                #pragma unroll
                for (int r = 0; r < RM; ++r) acc[s][r] = madd(ar[r], br[s], acc[s][r]);
        }
        __syncthreads();
        if constexpr (Prefetch) {
            if (k0 + K < kend) {
                #pragma unroll
                for (int l = 0; l < Loads; ++l) {
                    const int q = tid + l * Threads;
                    sa[q / Tile][q % Tile] = next_a[l];
                    sb[q / K][q % K] = next_b[l];
                }
            }
        }
    }
    #pragma unroll
    for (int s = 0; s < RN; ++s) {
        const int j = col0 + ty + s * BY;
        #pragma unroll
        for (int r = 0; r < RM; ++r) {
            const int i = row0 + tx + r * BX;
            if constexpr (Tri) {
                if (UpA ? (i <= j && j < n) : (j <= i && i < n)) {
                    const std::size_t p = i + std::size_t(j) * ldc;
                    T value = alpha * acc[s][r];
                    if constexpr (ReadC) value += beta * Global<T>::load(c, p);
                    Global<T>::store(c, p, value);
                }
            } else {
                if (i < n && j < n) {
                    const std::size_t p = i + std::size_t(j) * ldc;
                    T value = alpha * acc[s][r];
                    if constexpr (ReadC) value += beta * Global<T>::load(c, p);
                    Global<T>::store(c, p, value);
                }
            }
        }
    }
}

template <bool UpA, bool UpB, typename D, int Tile, int K, int BX, int BY, bool Prefetch>
void launch_general(gpuStream_t stream, int n, D al, const void* a, int lda, int ta, int ca,
                    const void* b, int ldb, int tb, int cb, D be, void* c, int ldc) {
    const std::uint64_t nt = (n - 1) / Tile + 1;
    const std::uint64_t blocks = UpA == UpB ? nt * (nt + 1) / 2 : nt * nt;
    if (be == D(0)) trtrm_general<UpA,UpB,D,false,Tile,K,BX,BY,Prefetch><<<unsigned(blocks),dim3(BX,BY),0,stream>>>(n,al,a,lda,ta,ca,b,ldb,tb,cb,be,c,ldc);
    else trtrm_general<UpA,UpB,D,true,Tile,K,BX,BY,Prefetch><<<unsigned(blocks),dim3(BX,BY),0,stream>>>(n,al,a,lda,ta,ca,b,ldb,tb,cb,be,c,ldc);
}
#endif

// OpenBLAS TRMM outperformed the scalar host loops. The packed operand reads
// only its stored triangle; one panel of the caller's scratch serves all
// column panels.
template <typename T>
void trtrm_cpu(const DdlaHandle_t& h, char ua, char xa, char ub, char xb,
               int n, const T& alpha, const T* a, int lda, const T* b, int ldb,
               const T& beta, T* c, int ldc, void* work, [[maybe_unused]] std::size_t bytes)
{
    const bool ta = xa != 'N', tb = xb != 'N';
    const bool upper_a = (ua == 'U') != ta, upper_b = (ub == 'U') != tb;
    const bool triangular = upper_a == upper_b;
    const bool read_c = beta != T(0);
    // [ product panel (n x min(n, 256)) ], only for alpha != 0.
    char* w = static_cast<char*>(work);
    T* const product = alpha == T(0) ? nullptr
        : detail::take<T>(w, std::size_t(n) * std::min(n, trtrm_panel_width));
    assert(static_cast<std::size_t>(w - static_cast<char*>(work)) <= bytes);
    const int width = alpha == T(0) ? n : std::min(n, trtrm_panel_width);
    for (int col0 = 0; col0 < n; col0 += width) {
        const int cols = std::min(width, n - col0);
        const int row0 = triangular && !upper_a ? col0 : 0;
        const int rows = triangular ? (upper_a ? col0 + cols : n - row0) : n;
        T* const out = c + row0 + std::size_t(col0)*ldc;
        if (product) {
            for (int j = 0; j < cols; ++j) {
                for (int i = 0; i < rows; ++i) {
                    const int r = row0 + i, s = col0 + j;
                    T value(0);
                    if (upper_b ? r <= s : r >= s) {
                        value = b[tb ? s + std::size_t(r)*ldb : r + std::size_t(s)*ldb];
                        if constexpr (!std::is_arithmetic_v<T>)
                            if (xb == 'C') value = std::conj(value);
                    }
                    product[i + std::size_t(j)*rows] = value;
                }
            }
            blas_impl::trmm_run<DdlaBackend::CPU, T>(h, 'L', ua, xa, 'N', rows, cols, alpha,
                                      a + row0 + std::size_t(row0)*lda, lda,
                                      product, rows);
        }
        if (!triangular && !product && !read_c) {
            // Preserve the concurrent poisoned-C regression fix explicitly.
            for (int j = 0; j < cols; ++j)
                DDLA_CHECK(ddlaMemset(h, out + std::size_t(j)*ldc, 0,
                                      std::size_t(rows)*sizeof(T)));
        } else if (!triangular) {
            blas_impl::geam_run<DdlaBackend::CPU, T>(h, 'N', 'N', rows, cols,
                                      product ? T(1) : T(0), product ? product : out,
                                      product ? rows : ldc, beta, out, ldc, out, ldc);
        } else {
            for (int j = 0; j < cols; ++j) {
                const int begin = upper_a ? 0 : std::max(0, col0 + j - row0);
                const int end = upper_a ? std::min(rows, col0 + j + 1 - row0) : rows;
                for (int i = begin; i < end; ++i) {
                    T value = product ? product[i + std::size_t(j)*rows] : T(0);
                    if (read_c) {
                        const T scaled = beta * out[i + std::size_t(j)*ldc];
                        value = product ? value + scaled : scaled;
                    }
                    out[i + std::size_t(j)*ldc] = value;
                }
            }
        }
    }
}


template <bool Upper, DdlaBackend Backend, typename T>
void trtrm_impl(const DdlaHandle_t& handle, int n, const T& alpha,
           const T* A, int lda, const T* B, int ldb,
           const T& beta, T* C, int ldc, [[maybe_unused]] void* work, [[maybe_unused]] std::size_t bytes)
{
    if (n == 0 || (alpha == T(0) && beta == T(1))) return;
    if constexpr (Backend == DdlaBackend::CPU) {
        trtrm_cpu(handle, Upper ? 'U' : 'L', 'N', Upper ? 'U' : 'L', 'N',
                   n, alpha, A, lda, B, ldb, beta, C, ldc, work, bytes);
    } else {
#if DDLA_GPU_VENDOR
        using D = detail::DeviceScalarT<T>;
        using R = typename detail::DeviceScalar<T>::real;
        static_assert(sizeof(D) == sizeof(T) && alignof(T) >= alignof(R) && sizeof(T) % alignof(R) == 0,
                      "complex component storage mismatch");
        static_assert(std::is_trivially_copyable<T>::value && std::is_trivially_copyable<D>::value,
                      "complex buffer bridge requires trivially copyable scalars");
        void* c = C;
        const D al = detail::to_device_scalar(alpha), be = detail::to_device_scalar(beta);
        if (alpha == T(0)) {
            const dim3 threads(16, 16), blocks((n - 1) / 16 + 1, (n - 1) / 16 + 1);
            if (beta == T(0)) scale_triangle<Upper, D, false><<<blocks, threads, 0, static_cast<gpuStream_t>(handle->stream)>>>(n, be, c, ldc);
            else scale_triangle<Upper, D, true><<<blocks, threads, 0, static_cast<gpuStream_t>(handle->stream)>>>(n, be, c, ldc);
        } else {
            // V100 measurements favor more small tiles before the GPU is full;
            // complex<double> needs the smaller register tile even at large n.
            if (n <= 128 && !std::is_same_v<T, float>)
                launch<Upper,D,16,8,16,16,false>(static_cast<gpuStream_t>(handle->stream),n,al,A,lda,B,ldb,be,C,ldc);
            else if (n <= 1024 || std::is_same_v<T, std::complex<double>> ||
                     (std::is_same_v<T, std::complex<float>> && n <= 2048))
                launch<Upper,D,32,8,16,8,true>(static_cast<gpuStream_t>(handle->stream),n,al,A,lda,B,ldb,be,C,ldc);
            else if constexpr (std::is_arithmetic_v<T>)
                launch<Upper,D,64,16,16,8,true>(static_cast<gpuStream_t>(handle->stream),n,al,A,lda,B,ldb,be,C,ldc);
            else
                launch<Upper,D,64,16,16,16,true>(static_cast<gpuStream_t>(handle->stream),n,al,A,lda,B,ldb,be,C,ldc);
        }
        DDLA_CHECK(ddlaGetLastError(handle));
#endif
    }
}

template <bool UpA, bool UpB, DdlaBackend Backend, typename T>
void trtrm_general_impl(const DdlaHandle_t& handle, int n, const T& alpha,
                        const T* A, int lda, bool ta, bool ca,
                        const T* B, int ldb, bool tb, bool cb,
                        const T& beta, T* C, int ldc, [[maybe_unused]] void* work,
                        [[maybe_unused]] std::size_t bytes)
{
    if constexpr (Backend == DdlaBackend::CPU) {
        trtrm_cpu(handle, (UpA != ta) ? 'U' : 'L', ca ? 'C' : (ta ? 'T' : 'N'),
                   (UpB != tb) ? 'U' : 'L', cb ? 'C' : (tb ? 'T' : 'N'),
                   n, alpha, A, lda, B, ldb, beta, C, ldc, work, bytes);
    } else {
#if DDLA_GPU_VENDOR
        using D = detail::DeviceScalarT<T>;
        using R = typename detail::DeviceScalar<T>::real;
        static_assert(sizeof(D) == sizeof(T) && alignof(T) >= alignof(R) && sizeof(T) % alignof(R) == 0,
                      "complex component storage mismatch");
        static_assert(std::is_trivially_copyable<T>::value && std::is_trivially_copyable<D>::value,
                      "complex buffer bridge requires trivially copyable scalars");
        void* c = C;
        const D al = detail::to_device_scalar(alpha), be = detail::to_device_scalar(beta);
        const int ta_ = ta, ca_ = ca, tb_ = tb, cb_ = cb;
        if (alpha == T(0)) {
            if constexpr (UpA == UpB) {
                const dim3 threads(16, 16), blocks((n - 1) / 16 + 1, (n - 1) / 16 + 1);
                if (beta == T(0)) scale_triangle<UpA, D, false><<<blocks, threads, 0, static_cast<gpuStream_t>(handle->stream)>>>(n, be, c, ldc);
                else scale_triangle<UpA, D, true><<<blocks, threads, 0, static_cast<gpuStream_t>(handle->stream)>>>(n, be, c, ldc);
            } else {
                const dim3 threads(16, 16), blocks((n - 1) / 16 + 1, (n - 1) / 16 + 1);
                if (beta == T(0)) scale_full<D, false><<<blocks, threads, 0, static_cast<gpuStream_t>(handle->stream)>>>(n, be, c, ldc);
                else scale_full<D, true><<<blocks, threads, 0, static_cast<gpuStream_t>(handle->stream)>>>(n, be, c, ldc);
            }
        } else {
            // Same tile ladder as the same-triangle kernel: mixed products
            // already expose n*n tiles, so the small-n threshold keeps its
            // shape and larger tiles still win once the GPU is full.
            if (n <= 128 && !std::is_same_v<T, float>)
                launch_general<UpA,UpB,D,16,8,16,16,false>(static_cast<gpuStream_t>(handle->stream),n,al,A,lda,ta_,ca_,B,ldb,tb_,cb_,be,C,ldc);
            else if (n <= 1024 || std::is_same_v<T, std::complex<double>> ||
                     (std::is_same_v<T, std::complex<float>> && n <= 2048))
                launch_general<UpA,UpB,D,32,8,16,8,true>(static_cast<gpuStream_t>(handle->stream),n,al,A,lda,ta_,ca_,B,ldb,tb_,cb_,be,C,ldc);
            else if constexpr (std::is_arithmetic_v<T>)
                launch_general<UpA,UpB,D,64,16,16,8,true>(static_cast<gpuStream_t>(handle->stream),n,al,A,lda,ta_,ca_,B,ldb,tb_,cb_,be,C,ldc);
            else
                launch_general<UpA,UpB,D,64,16,16,16,true>(static_cast<gpuStream_t>(handle->stream),n,al,A,lda,ta_,ca_,B,ldb,tb_,cb_,be,C,ldc);
        }
        DDLA_CHECK(ddlaGetLastError(handle));
#endif
    }
}

} // namespace

namespace blas_impl {
// [ CPU product panel (n x min(n, 256)) ]; the GPU kernels need no scratch.
template <typename T>
void trtrm_workspace_size(const DdlaHandle_t& handle, int n, std::size_t* backend_bytes)
{
    *backend_bytes = 0;
    if (handle->backend == DdlaBackend::CPU && n > 0)
        *backend_bytes = detail::padded_bytes<T>(std::size_t(n) * std::min(n, trtrm_panel_width));
}

template <DdlaBackend Backend, typename T>
void trtrm_dimensions_check(int n, const T& alpha, const T* A, int lda, const T* B, int ldb,
                            const T& beta, T* C, int ldc, bool triangular)
{
    if (n < 0 || lda < std::max(1, n) || ldb < std::max(1, n) || ldc < std::max(1, n))
        throw std::invalid_argument("trtrm: invalid order or leading dimension");
    if (n == 0 || (alpha == T(0) && beta == T(1))) return;
    if (!C) throw std::invalid_argument("trtrm: null C");
    if (alpha != T(0)) {
        if (!A || !B) throw std::invalid_argument("trtrm: null input");
        if (detail::matrix_overlaps(A, lda, n, n, C, ldc, n, n) ||
            detail::matrix_overlaps(B, ldb, n, n, C, ldc, n, n))
            throw std::invalid_argument("trtrm: C must not overlap A or B");
    }
    if constexpr (Backend == DdlaBackend::GPU) {
        if (n > 0 && alpha != T(0)) {
            const int tile = n <= 128 && !std::is_same_v<T,float> ? 16 :
                n <= 1024 || std::is_same_v<T,std::complex<double>> ||
                (std::is_same_v<T,std::complex<float>> && n <= 2048) ? 32 : 64;
            const auto nt = static_cast<std::uint64_t>((n - 1) / tile + 1);
            const auto blocks = triangular ? nt*(nt+1)/2 : nt*nt;
            if (blocks > 2147483647ULL) throw std::invalid_argument("trtrm: GPU grid too large");
        }
    }
}

template <DdlaBackend Backend, typename T>
void trtrm_check(const DdlaHandle_t& handle, char uplo_a, char transa, char uplo_b, char transb,
           int n, const T& alpha, const T* A, int lda, const T* B, int ldb,
           const T& beta, T* C, int ldc)
{
    detail::require_backend<Backend>(handle, "trtrm");
    const char ua = static_cast<char>(std::toupper(static_cast<unsigned char>(uplo_a)));
    const char ub = static_cast<char>(std::toupper(static_cast<unsigned char>(uplo_b)));
    const char xa = static_cast<char>(std::toupper(static_cast<unsigned char>(transa)));
    const char xb = static_cast<char>(std::toupper(static_cast<unsigned char>(transb)));
    if (ua != 'U' && ua != 'L') throw std::invalid_argument("trtrm: uplo_a must be U or L");
    if (ub != 'U' && ub != 'L') throw std::invalid_argument("trtrm: uplo_b must be U or L");
    if (xa != 'N' && xa != 'T' && xa != 'C') throw std::invalid_argument("trtrm: transa must be N, T or C");
    if (xb != 'N' && xb != 'T' && xb != 'C') throw std::invalid_argument("trtrm: transb must be N, T or C");
    const bool triangular = ((ua=='U') != (xa!='N')) == ((ub=='U') != (xb!='N'));
    trtrm_dimensions_check<Backend,T>(n,alpha,A,lda,B,ldb,beta,C,ldc,triangular);
}
template <DdlaBackend Backend, typename T>
void trtrm_run(const DdlaHandle_t& handle, char uplo_a, char transa, char uplo_b, char transb,
           int n, const T& alpha, const T* A, int lda, const T* B, int ldb,
           const T& beta, T* C, int ldc, void* backend_work, std::size_t backend_bytes)
{
    if (n==0 || (alpha==T(0) && beta==T(1))) return;
    const char ua=upper(uplo_a),ub=upper(uplo_b),xa=upper(transa),xb=upper(transb);
    const bool ta = xa != 'N', tb = xb != 'N';
    const bool eua = (ua == 'U') != ta, eub = (ub == 'U') != tb;
    if (!ta && !tb && eua == eub) {
        if (eua) trtrm_impl<true, Backend>(handle, n, alpha, A, lda, B, ldb, beta, C, ldc, backend_work, backend_bytes);
        else trtrm_impl<false, Backend>(handle, n, alpha, A, lda, B, ldb, beta, C, ldc, backend_work, backend_bytes);
        return;
    }
    const bool ca = xa == 'C', cb = xb == 'C';
    if (eua) {
        if (eub) trtrm_general_impl<true, true, Backend>(handle, n, alpha, A, lda, ta, ca, B, ldb, tb, cb, beta, C, ldc, backend_work, backend_bytes);
        else trtrm_general_impl<true, false, Backend>(handle, n, alpha, A, lda, ta, ca, B, ldb, tb, cb, beta, C, ldc, backend_work, backend_bytes);
    } else {
        if (eub) trtrm_general_impl<false, true, Backend>(handle, n, alpha, A, lda, ta, ca, B, ldb, tb, cb, beta, C, ldc, backend_work, backend_bytes);
        else trtrm_general_impl<false, false, Backend>(handle, n, alpha, A, lda, ta, ca, B, ldb, tb, cb, beta, C, ldc, backend_work, backend_bytes);
    }
}
} // namespace blas_impl
template <DdlaBackend Backend, typename T>
void trtrm(const DdlaHandle_t& handle, char uplo_a, char transa, char uplo_b, char transb,
           int n, const T& alpha, const T* A, int lda, const T* B, int ldb,
           const T& beta, T* C, int ldc)
{
    blas_impl::trtrm_check<Backend,T>(handle,uplo_a,transa,uplo_b,transb,n,alpha,A,lda,B,ldb,beta,C,ldc);
    std::size_t backend_bytes=0;
    if (alpha!=T(0)) blas_impl::trtrm_workspace_size<T>(handle,n,&backend_bytes);
    vector<char> work(handle,backend_bytes);
    blas_impl::trtrm_run<Backend,T>(handle,uplo_a,transa,uplo_b,transb,n,alpha,A,lda,B,ldb,beta,C,ldc,work.data(),backend_bytes);
}
#define INSTANTIATE_TRTRM(B, T) \
    template void trtrm<B, T>(const DdlaHandle_t&, char, char, char, char, int, const T&, const T*, int, const T*, int, const T&, T*, int)
#define INSTANTIATE_TRTRM_TYPES(B) \
    INSTANTIATE_TRTRM(B, float); \
    INSTANTIATE_TRTRM(B, double); \
    INSTANTIATE_TRTRM(B, std::complex<float>); \
    INSTANTIATE_TRTRM(B, std::complex<double>)
INSTANTIATE_TRTRM_TYPES(DdlaBackend::CPU);
#if DDLA_HAS_GPU
INSTANTIATE_TRTRM_TYPES(DdlaBackend::GPU);
#endif
#undef INSTANTIATE_TRTRM_TYPES
#undef INSTANTIATE_TRTRM
#define DDLA_TRTRM_PHASES(B,T) \
 template void blas_impl::trtrm_check<B,T>(const DdlaHandle_t&,char,char,char,char,int,const T&,const T*,int,const T*,int,const T&,T*,int); \
 template void blas_impl::trtrm_run<B,T>(const DdlaHandle_t&,char,char,char,char,int,const T&,const T*,int,const T*,int,const T&,T*,int,void*,std::size_t);
DDLA_TRTRM_PHASES(DdlaBackend::CPU,float)
DDLA_TRTRM_PHASES(DdlaBackend::CPU,double)
DDLA_TRTRM_PHASES(DdlaBackend::CPU,std::complex<float>)
DDLA_TRTRM_PHASES(DdlaBackend::CPU,std::complex<double>)
#if DDLA_HAS_GPU
DDLA_TRTRM_PHASES(DdlaBackend::GPU,float)
DDLA_TRTRM_PHASES(DdlaBackend::GPU,double)
DDLA_TRTRM_PHASES(DdlaBackend::GPU,std::complex<float>)
DDLA_TRTRM_PHASES(DdlaBackend::GPU,std::complex<double>)
#endif
#undef DDLA_TRTRM_PHASES
template void blas_impl::trtrm_workspace_size<float>(const DdlaHandle_t&,int,std::size_t*);
template void blas_impl::trtrm_workspace_size<double>(const DdlaHandle_t&,int,std::size_t*);
template void blas_impl::trtrm_workspace_size<std::complex<float>>(const DdlaHandle_t&,int,std::size_t*);
template void blas_impl::trtrm_workspace_size<std::complex<double>>(const DdlaHandle_t&,int,std::size_t*);
} // namespace ddla
