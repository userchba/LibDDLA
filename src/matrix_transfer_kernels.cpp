#include "ddla_connector.h"
#include "matrix_transfer.h"
#include "ddla_desc.h"
#include "ddla_stream.h"

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace ddla::matrix_impl {

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

template <typename T>
void map_gpu(const DdlaHandle_t& h, bool scatter, const Block<T>* blocks, int count,
             const int* desc, int ia, int ja, int m, int n, T* window, int ldw)
{
    constexpr int N = sizeof(T) / sizeof(std::uint32_t);
    static_assert(sizeof(Bits<N>) == sizeof(T));
    Batch<N> batch{};
    int size = 0, rows = 0, cols = 0;  // the largest block of a batch sizes its grid
    auto launch = [&] {
        const dim3 grid(std::min(65535, (rows - 1) / 256 + 1), std::min(65535, cols), size);
        map_kernel<<<grid, 256, 0, static_cast<gpuStream_t>(h->stream)>>>(scatter, batch, ia, ja, m, n, desc[DDLA_MB_], desc[DDLA_NB_], desc[DDLA_RSRC_], desc[DDLA_CSRC_], h->nprows_, h->npcols_, reinterpret_cast<Bits<N>*>(window),
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

#define INST(T) \
    template void map_gpu<T>(const DdlaHandle_t&, bool, const Block<T>*, int, const int*, int, int, int, int, T*, int)
INST(float);
INST(double);
INST(std::complex<float>);
INST(std::complex<double>);
#undef INST

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
INST_ROWS(float);
INST_ROWS(double);
INST_ROWS(std::complex<float>);
INST_ROWS(std::complex<double>);
#undef INST_ROWS

} // namespace ddla::matrix_impl
