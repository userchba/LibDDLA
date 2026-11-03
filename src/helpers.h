#ifndef DDLA_HELPERS_H
#define DDLA_HELPERS_H

#include <complex>
#include <cstddef>
#include <cstdint>

namespace ddla::detail {

// Half-open address ranges. Compare distances instead of forming end addresses,
// so unrelated allocations and device addresses need not be dereferenced.
inline bool memory_overlaps(const void* a, std::size_t a_bytes,
                            const void* b, std::size_t b_bytes) noexcept
{
    if (!a || !b || !a_bytes || !b_bytes) return false;
    const auto ap = reinterpret_cast<std::uintptr_t>(a);
    const auto bp = reinterpret_cast<std::uintptr_t>(b);
    return ap <= bp ? bp - ap < a_bytes : ap - bp < b_bytes;
}

// Keep counts in elements: multiplying a large count by sizeof(T) could overflow.
template <typename T>
inline bool array_overlaps(const T* a, std::size_t a_count,
                           const T* b, std::size_t b_count) noexcept
{
    if (!a || !b || !a_count || !b_count) return false;
    const auto ap = reinterpret_cast<std::uintptr_t>(a);
    const auto bp = reinterpret_cast<std::uintptr_t>(b);
    return ap <= bp ? (bp - ap) / sizeof(T) < a_count : (ap - bp) / sizeof(T) < b_count;
}

// Column-major active spans include inter-column padding, but exclude padding
// after the last active column. Dimensions and leading dimensions are checked
// by the caller before this predicate is used.
template <typename T>
inline bool matrix_overlaps(const T* a, int lda, int a_rows, int a_cols,
                            const T* b, int ldb, int b_rows, int b_cols) noexcept
{
    if (!a_rows || !a_cols || !b_rows || !b_cols) return false;
    return array_overlaps(a, std::size_t(a_cols - 1) * lda + a_rows,
                          b, std::size_t(b_cols - 1) * ldb + b_rows);
}

// std::conj(real) returns a complex value; these overloads preserve the type.
template <typename T>
inline T host_conj(T value, bool = true) noexcept { return value; }

template <typename R>
inline std::complex<R> host_conj(std::complex<R> value, bool conjugate = true) noexcept
{
    return conjugate ? std::conj(value) : value;
}

} // namespace ddla::detail

#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
#include <thrust/complex.h>

namespace ddla::detail {

// Kernel arithmetic uses thrust::complex; the public API keeps std::complex.
// This maps types only: it does not establish pointer alignment or permit
// dereferencing a reinterpreted host scalar. Construct by-value scalars below.
template <typename T>
struct DeviceScalar {
    using type = T;
    using real = T;
};

template <typename R>
struct DeviceScalar<std::complex<R>> {
    using type = thrust::complex<R>;
    using real = R;
};

template <typename T>
using DeviceScalarT = typename DeviceScalar<T>::type;

template <typename T>
inline T to_device_scalar(const T& value) { return value; }

template <typename R>
inline thrust::complex<R> to_device_scalar(const std::complex<R>& value)
{
    return {value.real(), value.imag()};
}

#if defined(__CUDACC__) || defined(__HIP__)
template <typename T>
__device__ __forceinline__ T device_conj(const T& value, bool = true) { return value; }

template <typename R>
__device__ __forceinline__ thrust::complex<R> device_conj(const thrust::complex<R>& value, bool conjugate = true)
{
    return conjugate ? thrust::conj(value) : value;
}
#endif

} // namespace ddla::detail
#endif // DDLA_USE_CUDA || DDLA_USE_HIP

#endif // DDLA_HELPERS_H
