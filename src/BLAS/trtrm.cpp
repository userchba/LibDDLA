#include <ddla/ddla.h>
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "helpers.h"
#include "require_backend.h"
#include <algorithm>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ddla {
namespace {

// Address ranges include the column padding: rejecting overlap is conservative.
template <typename T>
bool overlaps(const T* a, int lda, const T* c, int ldc, int n)
{
    const auto x = reinterpret_cast<std::uintptr_t>(a);
    const auto y = reinterpret_cast<std::uintptr_t>(c);
    const std::size_t ax = (std::size_t(n - 1) * lda + n) * sizeof(T);
    const std::size_t cy = (std::size_t(n - 1) * ldc + n) * sizeof(T);
    return x <= y ? y - x < ax : x - y < cy;
}

#if DDLA_GPU_VENDOR
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
    if (blocks > 2147483647ULL) throw std::invalid_argument("trtrm: GPU grid too large");
    if (be == D(0)) trtrm_triangle<Upper,D,false,Tile,K,BX,BY,Prefetch><<<unsigned(blocks),dim3(BX,BY),0,stream>>>(n,al,a,lda,b,ldb,be,c,ldc);
    else trtrm_triangle<Upper,D,true,Tile,K,BX,BY,Prefetch><<<unsigned(blocks),dim3(BX,BY),0,stream>>>(n,al,a,lda,b,ldb,be,c,ldc);
}
#endif

template <bool Upper, DdlaBackend Backend, typename T>
void trtrm_impl(const DdlaHandle_t& handle, int n, const T& alpha,
           const T* A, int lda, const T* B, int ldb,
           const T& beta, T* C, int ldc)
{
    if (n < 0 || lda < std::max(1, n) || ldb < std::max(1, n) || ldc < std::max(1, n))
        throw std::invalid_argument("trtrm: invalid order or leading dimension");
    if (n == 0 || (alpha == T(0) && beta == T(1))) return;
    if (!C) throw std::invalid_argument("trtrm: null C");
    if (alpha != T(0)) {
        if (!A || !B) throw std::invalid_argument("trtrm: null input");
        if (overlaps(A, lda, C, ldc, n) || overlaps(B, ldb, C, ldc, n))
            throw std::invalid_argument("trtrm: C must not overlap A or B");
    }
    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        // j-k-i ordering reuses B(k,j) and traverses A and C contiguously.
        for (int j = 0; j < n; ++j) {
            T* cj = C + std::size_t(j) * ldc;
            for (int i = Upper ? 0 : j; i < (Upper ? j + 1 : n); ++i) cj[i] = beta == T(0) ? T(0) : beta * cj[i];
            if (alpha == T(0)) continue;
            for (int k = Upper ? 0 : j; k < (Upper ? j + 1 : n); ++k) {
                const T bk = alpha * B[k + std::size_t(j) * ldb];
                const T* ak = A + std::size_t(k) * lda;
                for (int i = Upper ? 0 : k; i < (Upper ? k + 1 : n); ++i) cj[i] += ak[i] * bk;
            }
        }
#endif
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

} // namespace

template <DdlaBackend Backend, typename T>
void trtrm(const DdlaHandle_t& handle, char uplo, int n, const T& alpha,
           const T* A, int lda, const T* B, int ldb,
           const T& beta, T* C, int ldc)
{
    detail::require_backend<Backend>(handle, "trtrm");
    if (uplo == 'U' || uplo == 'u')
        trtrm_impl<true, Backend>(handle, n, alpha, A, lda, B, ldb, beta, C, ldc);
    else if (uplo == 'L' || uplo == 'l')
        trtrm_impl<false, Backend>(handle, n, alpha, A, lda, B, ldb, beta, C, ldc);
    else
        throw std::invalid_argument("trtrm: uplo must be U or L");
}

#define INSTANTIATE_TRTRM(B, T) \
    template void trtrm<B, T>(const DdlaHandle_t&, char, int, const T&, const T*, int, const T*, int, const T&, T*, int)
#define INSTANTIATE_TRTRM_TYPES(B) \
    INSTANTIATE_TRTRM(B, float); \
    INSTANTIATE_TRTRM(B, double); \
    INSTANTIATE_TRTRM(B, std::complex<float>); \
    INSTANTIATE_TRTRM(B, std::complex<double>)
#if DDLA_HAS_CPU
INSTANTIATE_TRTRM_TYPES(DdlaBackend::CPU);
#endif
#if DDLA_HAS_GPU
INSTANTIATE_TRTRM_TYPES(DdlaBackend::GPU);
#endif
#undef INSTANTIATE_TRTRM_TYPES
#undef INSTANTIATE_TRTRM
} // namespace ddla
