#include "ddla_stream.h"
#include "gpu/gpu.h"
#include "helpers.h"
#include "hetrf_native.h"
#include "ldlt_gpu_max.h"
#include <algorithm>
#include <type_traits>

namespace ddla::ldlt_impl
{
namespace
{
template <class T> __device__ auto real_part(T x)
{
    if constexpr (std::is_arithmetic_v<T>)
        return x;
    else
        return x.real();
}
template <class T> __device__ auto real_load(const T* a, std::size_t i)
{
    if constexpr (std::is_arithmetic_v<T>)
        return a[i];
    else
        return reinterpret_cast<const typename T::value_type*>(a)[2 * i];
}

template <class T> __global__ void diagonal_factor(bool upper, int n, T* a, int lda, int* info)
{
    using R = decltype(real_part(T{}));
    __shared__ R pivot;
    if (!threadIdx.x)
        *info = 0;
    __syncthreads();
    for (int s = 0; s < n; ++s)
    {
        const int j = upper ? n - 1 - s : s;
        if (!threadIdx.x)
        {
            pivot = real_load(a, j + std::size_t(j) * lda);
            a[j + std::size_t(j) * lda] = T(pivot);
            if (pivot == R(0))
                *info = j + 1;
        }
        __syncthreads();
        if (pivot == R(0))
            return;
        const int first = upper ? 0 : j + 1, count = upper ? j : n - j - 1;
        for (int i = threadIdx.x; i < count; i += blockDim.x)
            a[first + i + std::size_t(j) * lda] /= pivot;
        __syncthreads();
        for (int p = threadIdx.x; p < count * count; p += blockDim.x)
        {
            const int i = first + p % count, k = first + p / count;
            if (upper ? i > k : i < k)
                continue;
            const T change =
                (a[i + std::size_t(j) * lda] * detail::device_conj(a[k + std::size_t(j) * lda])) *
                pivot;
            const auto index = i + std::size_t(k) * lda;
            if (i == k)
                a[index] = T(real_load(a, index) - real_part(change));
            else
                a[index] -= change;
        }
        __syncthreads();
    }
}

template <class T>
__global__ void panel_division(int rows, int cols, const T* d, int ldd, const T* w, int ldw, T* l,
                               int ldl)
{
    for (std::size_t p = blockIdx.x * blockDim.x + threadIdx.x; p < std::size_t(rows) * cols;
         p += blockDim.x * gridDim.x)
    {
        const int i = p % rows, j = p / rows;
        l[i + std::size_t(j) * ldl] =
            w[i + std::size_t(j) * ldw] / real_load(d, j + std::size_t(j) * ldd);
    }
}

template <class T>
__global__ void column_summary(T* column, int first, int count, int diagonal, PivotSummary* result)
{
    __shared__ double values[256];
    __shared__ int indices[256];
    gpu_detail::max_abs_column(first, first + count, diagonal, column, 1, values, indices);
    if (!threadIdx.x)
    {
        const auto value = real_load(column, diagonal);
        column[diagonal] = T(value);
        *result = {double(fabs(value)), fmax(0.0, values[0]), indices[0], 0};
    }
}

template <class T>
__global__ void normalize_pivots(int first, int count, int current, int neighbor, int step,
                                 int swap_a, int swap_b, const T* c0, const T* c1, T* l0, T* l1,
                                 T* w0, T* w1)
{
    const int current_source = current == swap_a ? swap_b : current == swap_b ? swap_a : current;
    const int neighbor_source = neighbor == swap_a   ? swap_b
                                : neighbor == swap_b ? swap_a
                                                     : neighbor;
    const auto d0 = real_load(c0, current_source);
    T d11{}, d22{}, inverse{};
    if (step == 2)
    {
        const T off = c0[neighbor_source];
        d11 = real_load(c1, neighbor_source) / off;
        d22 = d0 / detail::device_conj(off);
        inverse = (1 / (real_part(d11 * d22) - 1)) / off;
    }
    for (int i = first + blockIdx.x * blockDim.x + threadIdx.x; i < first + count;
         i += blockDim.x * gridDim.x)
    {
        const int source = i == swap_a ? swap_b : i == swap_b ? swap_a : i;
        const T x = c0[source];
        w0[i] = x;
        if (step == 1)
            l0[i] = i == current ? T(d0) : (d0 == 0 ? x : x / d0);
        else
        {
            const T y = c1[source];
            w1[i] = y;
            if (i == current || i == neighbor)
            {
                l0[i] = i == current ? T(d0) : x;
                l1[i] = i == neighbor ? T(real_load(c1, neighbor_source)) : y;
            }
            else
            {
                l0[i] = detail::device_conj(inverse) * (d11 * x - y);
                l1[i] = inverse * (d22 * y - x);
            }
        }
    }
}

template <bool Subtract, class T>
__global__ void stored_copy(bool upper, int n, int rows, int cols, int mb, int nb, int pr, int pc,
                            int mr, int mc, int rs, int cs, const T* src, int lds, T* dst, int ldd,
                            int row0, int col0)
{
    for (std::size_t p = blockIdx.x * blockDim.x + threadIdx.x; p < std::size_t(rows) * cols;
         p += blockDim.x * gridDim.x)
    {
        const int i = p % rows, j = p / rows, r = row0 + i, c = col0 + j;
        const int gi = ((r / mb) * pr + (mr - rs + pr) % pr) * mb + r % mb;
        const int gj = ((c / nb) * pc + (mc - cs + pc) % pc) * nb + c % nb;
        if (gi >= n || gj >= n || (upper ? gi > gj : gi < gj))
            continue;
        const auto source = i + std::size_t(j) * lds, target = r + std::size_t(c) * ldd;
        if constexpr (Subtract)
        {
            if (gi == gj)
                dst[target] = T(real_load(dst, target) - real_load(src, source));
            else
                dst[target] -= src[source];
        }
        else
            dst[target] = gi == gj ? T(real_load(src, source)) : src[source];
    }
}

int blocks(std::size_t elements)
{
    return int(std::min<std::size_t>((elements + 255) / 256, 65535));
}
} // namespace

template <class T>
void summarize_column(const DdlaHandle_t& h, T* column, int first, int count, int diagonal,
                      PivotSummary* output)
{
    using D = detail::DeviceScalarT<T>;
    static_assert(sizeof(D) == sizeof(T) && sizeof(T) % alignof(D) == 0);
    column_summary<<<1, 256, 0, static_cast<gpuStream_t>(h->stream)>>>(
        reinterpret_cast<D*>(column), first, count, diagonal, output);
    DDLA_CHECK(ddlaGetLastError(h));
}
template <class T>
void nopiv_diagonal(const DdlaHandle_t& h, char u, int n, T* a, int lda, int* info)
{
    using D = detail::DeviceScalarT<T>;
    static_assert(sizeof(D) == sizeof(T) && sizeof(T) % alignof(D) == 0);
    diagonal_factor<<<1, 256, 0, static_cast<gpuStream_t>(h->stream)>>>(
        u == 'U', n, reinterpret_cast<D*>(a), lda, info);
    DDLA_CHECK(ddlaGetLastError(h));
}
template <class T>
void divide_panel(const DdlaHandle_t& h, int rows, int cols, const T* d, int ldd, const T* w,
                  int ldw, T* l, int ldl)
{
    if (!rows || !cols)
        return;
    using D = detail::DeviceScalarT<T>;
    static_assert(sizeof(D) == sizeof(T) && sizeof(T) % alignof(D) == 0);
    panel_division<<<blocks(std::size_t(rows) * cols), 256, 0,
                     static_cast<gpuStream_t>(h->stream)>>>(
        rows, cols, reinterpret_cast<const D*>(d), ldd, reinterpret_cast<const D*>(w), ldw,
        reinterpret_cast<D*>(l), ldl);
    DDLA_CHECK(ddlaGetLastError(h));
}
template <class T>
void pivot_columns(const DdlaHandle_t& h, int first, int count, int current, int neighbor, int step,
                   int swap_a, int swap_b, const T* c0, const T* c1, T* l0, T* l1, T* w0, T* w1)
{
    if (!count)
        return;
    using D = detail::DeviceScalarT<T>;
    static_assert(sizeof(D) == sizeof(T) && sizeof(T) % alignof(D) == 0);
    normalize_pivots<<<blocks(count), 256, 0, static_cast<gpuStream_t>(h->stream)>>>(
        first, count, current, neighbor, step, swap_a, swap_b, reinterpret_cast<const D*>(c0),
        reinterpret_cast<const D*>(c1), reinterpret_cast<D*>(l0), reinterpret_cast<D*>(l1),
        reinterpret_cast<D*>(w0), reinterpret_cast<D*>(w1));
    DDLA_CHECK(ddlaGetLastError(h));
}
template <class T>
void copy_stored(const DdlaHandle_t& h, char u, int n, const int* d, const T* src, int lds, T* dst,
                 int ldd, int row0, int col0, int rows, int cols, bool subtract)
{
    if (rows < 0)
        rows = num_loc(n, d[DDLA_MB_], h->myprow_, d[DDLA_RSRC_], h->nprows_);
    if (cols < 0)
        cols = num_loc(n, d[DDLA_NB_], h->mypcol_, d[DDLA_CSRC_], h->npcols_);
    if (!rows || !cols)
        return;
    using D = detail::DeviceScalarT<T>;
    static_assert(sizeof(D) == sizeof(T) && sizeof(T) % alignof(D) == 0);
    const auto stream = static_cast<gpuStream_t>(h->stream);
    const auto grid = blocks(std::size_t(rows) * cols);
    if (subtract)
        stored_copy<true, D><<<grid, 256, 0, stream>>>(
            u == 'U', n, rows, cols, d[DDLA_MB_], d[DDLA_NB_], h->nprows_, h->npcols_, h->myprow_,
            h->mypcol_, d[DDLA_RSRC_], d[DDLA_CSRC_], reinterpret_cast<const D*>(src), lds,
            reinterpret_cast<D*>(dst), ldd, row0, col0);
    else
        stored_copy<false, D><<<grid, 256, 0, stream>>>(
            u == 'U', n, rows, cols, d[DDLA_MB_], d[DDLA_NB_], h->nprows_, h->npcols_, h->myprow_,
            h->mypcol_, d[DDLA_RSRC_], d[DDLA_CSRC_], reinterpret_cast<const D*>(src), lds,
            reinterpret_cast<D*>(dst), ldd, row0, col0);
    DDLA_CHECK(ddlaGetLastError(h));
}

#define INSTANCE(T)                                                                                \
    template void summarize_column<T>(const DdlaHandle_t&, T*, int, int, int, PivotSummary*);      \
    template void nopiv_diagonal<T>(const DdlaHandle_t&, char, int, T*, int, int*);                \
    template void divide_panel<T>(const DdlaHandle_t&, int, int, const T*, int, const T*, int, T*, \
                                  int);                                                            \
    template void pivot_columns<T>(const DdlaHandle_t&, int, int, int, int, int, int, int,         \
                                   const T*, const T*, T*, T*, T*, T*);                            \
    template void copy_stored<T>(const DdlaHandle_t&, char, int, const int*, const T*, int, T*,    \
                                 int, int, int, int, int, bool);
INSTANCE(float)
INSTANCE(double)
INSTANCE(std::complex<float>)
INSTANCE(std::complex<double>)
#undef INSTANCE
} // namespace ddla::ldlt_impl
