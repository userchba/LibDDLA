#include "memory_tri_kernels.h"
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace ddla::detail {
namespace {

// Bit storage avoids numerical conversions and the stronger alignment that
// device complex types can require. The dispatcher checks word alignment.
template <class Word, int Count = 1> struct Bits {
    Word word[Count];
};

// The common fill path needs only one value and one output pointer. Keeping
// its launch arguments small matters for repeated tiny panel cleanups.
template <class Storage>
__global__ void fill_triangle(bool upper, int rows, int cols, void* dst,
                              std::size_t pitch, Storage value)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || col >= cols || (upper ? row > col : row < col))
        return;
    auto* output = reinterpret_cast<Storage*>(static_cast<char*>(dst) + std::size_t(col) * pitch);
    output[row] = value;
}

template <bool Copy, bool WholeElement, class Storage>
__device__ __forceinline__ void triangle_element(bool upper, bool diagonal, std::size_t index,
                                                 std::size_t col, std::size_t units, void* dst,
                                                 std::size_t dpitch, const void* src,
                                                 std::size_t spitch, Storage fill)
{
    const std::size_t row = WholeElement ? index : index / units;
    if (upper ? (row > col || (!diagonal && row == col))
              : (row < col || (!diagonal && row == col)))
        return;
    auto* output = reinterpret_cast<Storage*>(static_cast<char*>(dst) + col * dpitch);
    if constexpr (Copy) {
        const auto* input = reinterpret_cast<const Storage*>(
            static_cast<const char*>(src) + col * spitch);
        output[index] = input[index];
    } else {
        output[index] = fill;
    }
}

template <bool Copy, bool WholeElement, bool GridStride, class Storage>
__global__ void memory_tri(bool upper, bool diagonal, std::size_t row_units, int cols,
                           std::size_t units, void* dst, std::size_t dpitch, const void* src,
                           std::size_t spitch, Storage fill)
{
    const std::size_t col0 = blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t row0 = blockIdx.x * blockDim.x + threadIdx.x;
    if constexpr (!GridStride) {
        if (col0 < std::size_t(cols) && row0 < row_units)
            triangle_element<Copy, WholeElement>(upper, diagonal, row0, col0, units,
                                                  dst, dpitch, src, spitch, fill);
    } else {
        for (std::size_t col = col0; col < std::size_t(cols);
             col += std::size_t(gridDim.y) * blockDim.y)
            for (std::size_t row = row0; row < row_units;
                 row += std::size_t(gridDim.x) * blockDim.x)
                triangle_element<Copy, WholeElement>(upper, diagonal, row, col, units,
                                                      dst, dpitch, src, spitch, fill);
    }
}

template <bool Copy, bool WholeElement, class Storage>
void launch_grid(void* stream, dim3 grid, dim3 block, bool upper, bool diagonal,
                 std::size_t row_units, int cols, std::size_t units, void* dst,
                 std::size_t dpitch, const void* src, std::size_t spitch, Storage fill)
{
    if (std::size_t(grid.x) * block.x >= row_units &&
        std::size_t(grid.y) * block.y >= std::size_t(cols))
        memory_tri<Copy, WholeElement, false><<<grid, block, 0, static_cast<gpuStream_t>(stream)>>>(
            upper, diagonal, row_units, cols, units, dst, dpitch, src, spitch, fill);
    else
        memory_tri<Copy, WholeElement, true><<<grid, block, 0, static_cast<gpuStream_t>(stream)>>>(
            upper, diagonal, row_units, cols, units, dst, dpitch, src, spitch, fill);
}

template <bool Copy, class Storage>
void launch(void* stream, bool upper, bool diagonal, int rows, int cols, void* dst,
            std::size_t dpitch, const void* src, std::size_t spitch,
            std::size_t element_bytes, int value)
{
    const std::size_t units = element_bytes / sizeof(Storage);
    const std::size_t row_units = std::size_t(rows) * units;
    const dim3 block(32, 8);
    const dim3 grid(unsigned(std::min<std::size_t>((row_units - 1) / 32 + 1, 65535)),
                    unsigned(std::min<std::size_t>((std::size_t(cols) - 1) / 8 + 1, 65535)));
    Storage fill;
    std::memset(&fill, value, sizeof(fill));
    if constexpr (!Copy) {
        if (units == 1 && row_units <= std::size_t(grid.x) * block.x &&
            std::size_t(cols) <= std::size_t(grid.y) * block.y) {
            // A strict triangle is an inclusive triangle in a shifted view.
            // The shift excludes the original diagonal without extra masks.
            auto* output = static_cast<char*>(dst);
            if (!diagonal) {
                if (upper) { output += dpitch; --cols; }
                else { output += sizeof(Storage); --rows; }
            }
            const dim3 tile = rows <= 128 && cols <= 128 ? dim3(16, 16) : block;
            const dim3 blocks((rows - 1) / tile.x + 1, (cols - 1) / tile.y + 1);
            fill_triangle<<<blocks, tile, 0, static_cast<gpuStream_t>(stream)>>>(
                upper, rows, cols, output, dpitch, fill);
            return;
        }
    }
    if (units == 1)
        launch_grid<Copy, true>(stream, grid, block, upper, diagonal, row_units, cols, units,
                                 dst, dpitch, src, spitch, fill);
    else
        launch_grid<Copy, false>(stream, grid, block, upper, diagonal, row_units, cols, units,
                                  dst, dpitch, src, spitch, fill);
}

template <bool Copy>
void dispatch(void* stream, bool upper, bool diagonal, int rows, int cols, void* dst,
              std::size_t dpitch, const void* src, std::size_t spitch,
              std::size_t element_bytes, int value)
{
    std::uintptr_t alignment = reinterpret_cast<std::uintptr_t>(dst) | dpitch | element_bytes;
    if constexpr (Copy)
        alignment |= reinterpret_cast<std::uintptr_t>(src) | spitch;
    if (element_bytes == 16 && alignment % 8 == 0)
        launch<Copy, Bits<std::uint64_t, 2>>(stream, upper, diagonal, rows, cols, dst, dpitch,
                                            src, spitch, element_bytes, value);
    else if (alignment % 8 == 0)
        launch<Copy, Bits<std::uint64_t>>(stream, upper, diagonal, rows, cols, dst, dpitch,
                                        src, spitch, element_bytes, value);
    else if (alignment % 4 == 0)
        launch<Copy, Bits<std::uint32_t>>(stream, upper, diagonal, rows, cols, dst, dpitch,
                                        src, spitch, element_bytes, value);
    else if (alignment % 2 == 0)
        launch<Copy, Bits<std::uint16_t>>(stream, upper, diagonal, rows, cols, dst, dpitch,
                                        src, spitch, element_bytes, value);
    else
        launch<Copy, Bits<std::uint8_t>>(stream, upper, diagonal, rows, cols, dst, dpitch,
                                       src, spitch, element_bytes, value);
}
}

gpuError_t memory_tri_gpu(void* stream, bool copy, bool upper, bool diagonal, int rows, int cols,
                         void* dst, std::size_t dpitch, const void* src, std::size_t spitch,
                         std::size_t element_bytes, int value)
{
    if (copy)
        dispatch<true>(stream, upper, diagonal, rows, cols, dst, dpitch, src, spitch,
                       element_bytes, value);
    else
        dispatch<false>(stream, upper, diagonal, rows, cols, dst, dpitch, src, spitch,
                        element_bytes, value);
    return gpuGetLastError();
}
}
