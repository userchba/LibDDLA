#ifndef DDLA_MEMORY_TRI_KERNELS_H
#define DDLA_MEMORY_TRI_KERNELS_H

#include "ddla_connector.h"
#include <cstddef>

namespace ddla::detail {
gpuError_t memory_tri_gpu(void* stream, bool copy, bool upper, bool diagonal, int rows, int cols,
                         void* dst, std::size_t dpitch, const void* src, std::size_t spitch,
                         std::size_t element_bytes, int value);
}

#endif
