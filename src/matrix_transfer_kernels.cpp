#include "ddla_connector.h"
#include "matrix_transfer.h"
#include "ddla_stream.h"

#include <algorithm>
#include <cstdint>

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
__global__ void map_kernel(bool scatter, Batch<N> batch, Region g, Bits<N>* window, int ldw)
{
    const Block<Bits<N>>& b = batch.block[blockIdx.z];
    const Peer& p = b.p;
    const int rshift = (p.prow - g.rsrc + g.nprows) % g.nprows;
    const int cshift = (p.pcol - g.csrc + g.npcols) % g.npcols;
    for (std::size_t c = blockIdx.y; c < std::size_t(p.cols); c += gridDim.y) {
        // A peer owning every column (row) of the window holds them in window order.
        const int wc = p.cols == g.n ? int(c)
                                     : global_index(p.col0 + int(c), g.nb, cshift, g.npcols) - g.j;
        Bits<N>* const block = b.data + c * b.ld;
        Bits<N>* const column = window + std::size_t(wc) * ldw;
        for (std::size_t r = blockIdx.x * blockDim.x + threadIdx.x; r < std::size_t(p.rows);
             r += gridDim.x * blockDim.x) {
            const int wr = p.rows == g.m ? int(r)
                                         : global_index(p.row0 + int(r), g.mb, rshift, g.nprows) - g.i;
            if (scatter) block[r] = column[wr];
            else column[wr] = block[r];
        }
    }
}

template <typename T>
void map_gpu(const DdlaHandle_t& h, bool scatter, const Block<T>* blocks, int count,
             const Region& g, T* window, int ldw)
{
    constexpr int N = sizeof(T) / sizeof(std::uint32_t);
    static_assert(sizeof(Bits<N>) == sizeof(T));
    Batch<N> batch{};
    int size = 0, rows = 0, cols = 0;  // the largest block of a batch sizes its grid
    auto launch = [&] {
        const dim3 grid(std::min(65535, (rows - 1) / 256 + 1), std::min(65535, cols), size);
        map_kernel<<<grid, 256, 0, static_cast<gpuStream_t>(h->stream)>>>(scatter, batch, g, reinterpret_cast<Bits<N>*>(window),
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
    template void map_gpu<T>(const DdlaHandle_t&, bool, const Block<T>*, int, const Region&, T*, int)
INST(float);
INST(double);
INST(std::complex<float>);
INST(std::complex<double>);
#undef INST

} // namespace ddla::matrix_impl
