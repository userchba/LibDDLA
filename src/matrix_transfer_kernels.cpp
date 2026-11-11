#include "ddla_connector.h"
#include "matrix_transfer.h"
#include "ddla_desc.h"
#include "ddla_stream.h"
#include "instantiate.h"
#include "host_parallel.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace ddla::matrix_impl {

#if DDLA_HAS_GPU
// Integer storage makes every scalar copy bit-preserving, including complex
// values and NaN payloads, without device std::complex arithmetic.
template <int N>
struct Bits {
    std::uint32_t word[N];
};

// Blocks per launch: their descriptors travel in the kernel argument block.
constexpr int kBatch = 32;

template <int N>
struct Batch {
    Block<Bits<N>> block[kBatch];
};

// Zero-based global index of local index l (ScaLAPACK INDXL2G); shift is the
// process's block offset (p - src + np) % np. Every partial sum is at most the
// global index, which fits the descriptor's int extent, so 32-bit arithmetic is
// exact and avoids a 64-bit division per element.
__device__ int global_index(int l, int b, int shift, int np)
{
    const int q = l / b;
    return (q * np + shift) * b + (l - q * b);
}

// Grid z picks the block, grid y strides over its columns, grid x over its rows.
template <int N>
__global__ void map_kernel(bool scatter, Batch<N> batch, int i, int j, int m, int n, int mb, int nb, int rsrc, int csrc, int nprows, int npcols, Bits<N>* window, int ldw)
{
    const Block<Bits<N>>& b = batch.block[blockIdx.z];
    const Peer& p = b.p;
    const int rshift = (p.prow - rsrc + nprows) % nprows;
    const int cshift = (p.pcol - csrc + npcols) % npcols;
    for (std::size_t c = blockIdx.y; c < std::size_t(p.cols); c += gridDim.y) {
        // A peer owning every column (row) of the window holds them in window order.
        const int wc = p.cols == n ? int(c)
                                     : global_index(p.col0 + int(c), nb, cshift, npcols) - j;
        Bits<N>* const block = b.data + c * b.ld;
        Bits<N>* const column = window + std::size_t(wc) * ldw;
        for (std::size_t r = blockIdx.x * blockDim.x + threadIdx.x; r < std::size_t(p.rows);
             r += gridDim.x * blockDim.x) {
            const int wr = p.rows == m ? int(r)
                                         : global_index(p.row0 + int(r), mb, rshift, nprows) - i;
            if (scatter) block[r] = column[wr];
            else column[wr] = block[r];
        }
    }
}

#endif

namespace {
// Keep the serial loop separate from OpenMP dispatch. Inlining it
// into the larger driver slowed small and one-row transfers in measurements.
template <typename T>
__attribute__((noinline))
void map_serial(const DdlaHandle_t& h, bool scatter, const Block<T>* blocks, int count,
                const int* d, int ia, int ja, int m, int n, T* window, int ldw)
{
    // A fully owned axis is already in window order, as in the GPU mapper.
    for (int k = 0; k < count; ++k) {
        const Block<T>& b = blocks[k];
        if (!b.p.rows || !b.p.cols) continue;
        for (int c = 0; c < b.p.cols; ++c) {
            const int wc = b.p.cols == n ? c
                : indxl2g(b.p.col0 + c, d[DDLA_NB_], b.p.pcol,
                          d[DDLA_CSRC_], h->npcols_) - ja;
            T* const column = b.data + std::size_t(c) * b.ld;
            T* const wcolumn = window + std::size_t(wc) * ldw;
            if (b.p.rows == m) {
                if (scatter) std::memcpy(column, wcolumn, m * sizeof(T));
                else std::memcpy(wcolumn, column, m * sizeof(T));
                continue;
            }
            for (int r = 0; r < b.p.rows;) {
                const int lr = b.p.row0 + r;
                const int rows = std::min(b.p.rows - r, d[DDLA_MB_] - lr % d[DDLA_MB_]);
                const int wr = indxl2g(lr, d[DDLA_MB_], b.p.prow,
                                      d[DDLA_RSRC_], h->nprows_) - ia;
                T* const w = wcolumn + wr;
                if (scatter) std::memcpy(column + r, w, rows * sizeof(T));
                else std::memcpy(w, column + r, rows * sizeof(T));
                r += rows;
            }
        }
    }
}
} // namespace

template <DdlaBackend B, typename T>
void map_blocks(const DdlaHandle_t& h, bool scatter, const Block<T>* blocks, int count,
             const int* d, int ia, int ja, int m, int n, T* window, int ldw)
{
    if constexpr (B == DdlaBackend::CPU) {
        const auto serial = [&] {
            map_serial(h, scatter, blocks, count, d, ia, ja, m, n, window, ldw);
        };
        if (h->host_threads <= 1) {
            serial();
            return;
        }
        long long elements = 0;
        for (int k = 0; k < count; ++k)
            elements += static_cast<long long>(blocks[k].p.rows) * blocks[k].p.cols;
        constexpr long long grain = (1 << 20) / sizeof(T);
        if (elements < 8 * grain) {
            serial();
            return;
        }
        // Keep all peer pieces of a window column on one worker, avoiding
        // false sharing between adjacent process-row blocks. A thin window
        // instead uses disjoint row slabs, including the one-column case.
        const int threads = static_cast<int>(std::min<long long>(h->host_threads, elements / grain));
        const long long chunks = std::max(4LL * threads, (elements + grain - 1) / grain);
        const bool columns = n >= threads;
        const int extent = columns ? n : m;
        const long long step = std::max(1LL, (extent + chunks - 1) / chunks);
        detail::host_parallel_for(extent, step, threads, [&](long long lo, long long hi) {
            for (int k = 0; k < count; ++k) {
                const Block<T>& b = blocks[k];
                if (!b.p.rows || !b.p.cols) continue;
                const int first = columns ? 0 : std::max(0,
                    num_loc(ia + int(lo), d[DDLA_MB_], b.p.prow, d[DDLA_RSRC_], h->nprows_) - b.p.row0);
                const int end = columns ? b.p.rows : std::min(b.p.rows,
                    num_loc(ia + int(hi), d[DDLA_MB_], b.p.prow, d[DDLA_RSRC_], h->nprows_) - b.p.row0);
                if (first >= end) continue;
                const int c0 = columns ? std::max(0,
                    num_loc(ja + int(lo), d[DDLA_NB_], b.p.pcol, d[DDLA_CSRC_], h->npcols_) - b.p.col0) : 0;
                const int c1 = columns ? std::min(b.p.cols,
                    num_loc(ja + int(hi), d[DDLA_NB_], b.p.pcol, d[DDLA_CSRC_], h->npcols_) - b.p.col0) : b.p.cols;
                if (c0 >= c1) continue;
                const Block<T> part{b.data + first + std::size_t(c0) * b.ld, b.ld,
                    {b.p.row0 + first, b.p.col0 + c0, end - first, c1 - c0,
                     b.p.prow, b.p.pcol}};
                map_serial(h, scatter, &part, 1, d, ia, ja, m, n, window, ldw);
            }
        });
    }
#if DDLA_HAS_GPU
    else {
        constexpr int N = sizeof(T) / sizeof(std::uint32_t);
        static_assert(sizeof(Bits<N>) == sizeof(T));
        Batch<N> batch{};
        int size = 0, rows = 0, cols = 0;  // the largest block of a batch sizes its grid
        auto launch = [&] {
            const dim3 grid(std::min(65535, (rows - 1) / 256 + 1), std::min(65535, cols), size);
            map_kernel<<<grid, 256, 0, static_cast<gpuStream_t>(h->stream)>>>(scatter, batch, ia, ja, m, n, d[DDLA_MB_], d[DDLA_NB_], d[DDLA_RSRC_], d[DDLA_CSRC_], h->nprows_, h->npcols_, reinterpret_cast<Bits<N>*>(window),
                                                    ldw);
            DDLA_CHECK(ddlaGetLastError(h));
            size = rows = cols = 0;
        };
        for (int k = 0; k < count; ++k) {
            const Block<T>& b = blocks[k];
            if (!b.p.rows || !b.p.cols) continue;
            batch.block[size++] = {reinterpret_cast<Bits<N>*>(b.data), b.ld, b.p};
            rows = std::max(rows, b.p.rows);
            cols = std::max(cols, b.p.cols);
            if (size == kBatch) launch();
        }
        if (size) launch();
    }
#endif
}

#define INST(B, T) \
    template void map_blocks<B, T>(const DdlaHandle_t&, bool, const Block<T>*, int, const int*, int, int, int, int, T*, int)
DDLA_FOR_EACH_BACKEND_TYPE(INST)
#undef INST

#if DDLA_HAS_GPU
template <typename rt, bool complex_out>
__global__ void convert_rows_kernel(const rt* src, int lds, rt* dst, int ldd,
                                        int rows, int cols, bool transpose)
{
    const std::size_t count = std::size_t(rows) * cols;
    for (std::size_t p = blockIdx.x * blockDim.x + threadIdx.x; p < count;
         p += std::size_t(blockDim.x) * gridDim.x) {
        const int r = p % rows, c = p / rows;
        const rt x = src ? (transpose ? src[c + std::size_t(r) * lds]
                                     : src[r + std::size_t(c) * lds]) : rt(0);
        const std::size_t q = r + std::size_t(c) * ldd;
        if constexpr (complex_out) { dst[2*q] = x; dst[2*q+1] = rt(0); }
        else dst[q] = x;
    }
}

template <typename T>
void convert_rows_gpu(const DdlaHandle_t& h, const typename real_type<T>::type* src,
                      int lds, T* dst, int ldd, int rows, int cols, bool transpose)
{
    using R=typename real_type<T>::type;
    if (!rows || !cols) return;
    const unsigned blocks=unsigned(std::min<std::size_t>(65535,(std::size_t(rows)*cols+255)/256));
    convert_rows_kernel<R,!std::is_same<T,R>::value><<<blocks,256,0,static_cast<gpuStream_t>(h->stream)>>>(
        src,lds,reinterpret_cast<R*>(dst),ldd,rows,cols,transpose);
    DDLA_CHECK(ddlaGetLastError(h));
}
#define INST_ROWS(T) template void convert_rows_gpu<T>(const DdlaHandle_t&,const real_type<T>::type*,int,T*,int,int,int,bool)
DDLA_FOR_EACH_TYPE(INST_ROWS)
#undef INST_ROWS

#endif

} // namespace ddla::matrix_impl
