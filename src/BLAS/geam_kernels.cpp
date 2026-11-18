#include "blas_impl.h"
#include "helpers.h"
#include "instantiate.h"

namespace ddla::blas_impl {
namespace {

template <typename T>
__global__ void transpose_copy(const T* source, T* target, int rows, int cols,
                              int lds, int ldt, bool conjugate)
{
    const std::size_t row_tiles = (std::size_t(rows) + 15) / 16;
    const std::size_t tiles = row_tiles * ((std::size_t(cols) + 15) / 16);
    for (std::size_t tile = blockIdx.x; tile < tiles; tile += gridDim.x)
        detail::transpose_tile(source, target, rows, cols, lds, ldt,
            int(tile % row_tiles) * 16, int(tile / row_tiles) * 16, conjugate);
}

} // namespace

template <typename T>
void geam_transpose_gpu(const DdlaHandle_t& h, bool conjugate, int m, int n,
                        const T* source, int lds, T* target, int ldt, void* queue)
{
    using DT = detail::DeviceScalarT<T>;
    static_assert(sizeof(DT) == sizeof(T) && sizeof(T) % alignof(DT) == 0);
    // Nominal std::complex alignment can be weaker than the device type.
    // Check actual device addresses; element strides preserve this alignment.
    if (reinterpret_cast<std::uintptr_t>(source) % alignof(DT) ||
        reinterpret_cast<std::uintptr_t>(target) % alignof(DT))
        throw std::invalid_argument("geam: misaligned device copy buffer");
    const std::size_t tiles = ((std::size_t(n) + 15) / 16)
                            * ((std::size_t(m) + 15) / 16);
    transpose_copy<DT><<<dim3(unsigned(std::min(tiles, std::size_t(65535)))),
        dim3(16, 16), 0, static_cast<gpuStream_t>(queue)>>>(
            reinterpret_cast<const DT*>(source), reinterpret_cast<DT*>(target),
            n, m, lds, ldt, conjugate);
    DDLA_CHECK(ddlaGetLastError(h));
}

#define INSTANTIATE(T) template void geam_transpose_gpu<T>(const DdlaHandle_t&, bool, \
    int, int, const T*, int, T*, int, void*)
DDLA_FOR_EACH_TYPE(INSTANTIATE)
#undef INSTANTIATE

} // namespace ddla::blas_impl
