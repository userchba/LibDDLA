// Shared reduction extracted from the native MAGMA-style HETRF panel.
// Copyright (c) 2009-2023, The University of Tennessee.
// BSD-3-Clause terms and disclaimer are retained in MAGMA-LICENSE.
#ifndef DDLA_LDLT_GPU_MAX_H
#define DDLA_LDLT_GPU_MAX_H
#include "helpers.h"
#include <type_traits>
namespace ddla::ldlt_impl::gpu_detail
{
template <class T>
__device__ void max_abs_column(int first, int last, int skip, const T* x, int inc, double* values,
                               int* indices)
{
    double value = -1;
    int index = last;
    for (int i = first + threadIdx.x; i < last; i += blockDim.x)
    {
        if (i == skip)
            continue;
        const T z = x[std::size_t(i) * inc];
        double next;
        if constexpr (std::is_arithmetic_v<T>)
            next = fabs(z);
        else
            next = fabs(z.real()) + fabs(z.imag());
        if (next > value || (next == value && i < index))
        {
            value = next;
            index = i;
        }
    }
    values[threadIdx.x] = value;
    indices[threadIdx.x] = index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2)
    {
        if (threadIdx.x < stride)
        {
            const double other = values[threadIdx.x + stride];
            const int oi = indices[threadIdx.x + stride];
            if (other > values[threadIdx.x] ||
                (other == values[threadIdx.x] && oi < indices[threadIdx.x]))
            {
                values[threadIdx.x] = other;
                indices[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
}

} // namespace ddla::ldlt_impl::gpu_detail
#endif
