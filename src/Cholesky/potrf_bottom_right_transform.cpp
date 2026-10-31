#include "ddla_connector.h"
#include "potrf_bottom_right_transform.h"
#include "ddla_stream.h"

#include <cassert>
#include <cstddef>
#include <complex>

#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
#include <thrust/complex.h>
#endif

namespace ddla::detail {

#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)

namespace {

template <typename T>
struct DeviceScalar {
    using type = T;
};

template <>
struct DeviceScalar<std::complex<float>> {
    using type = thrust::complex<float>;
};

template <>
struct DeviceScalar<std::complex<double>> {
    using type = thrust::complex<double>;
};

template <typename T>
using DeviceScalarT = typename DeviceScalar<T>::type;

template <typename T>
__global__ void reverse_upper_to_lower_kernel(
    int n, const T* A, int lda, T* B, int ldb)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int column = blockIdx.y * blockDim.y + threadIdx.y;
    if(row >= n || column >= n || row < column){
        return;
    }

    const int source_row = n - 1 - row;
    const int source_column = n - 1 - column;
    B[row + static_cast<std::size_t>(column) * ldb] =
        A[source_row + static_cast<std::size_t>(source_column) * lda];
}

template <typename T>
__global__ void reverse_lower_to_upper_kernel(
    int n, const T* B, int ldb, T* A, int lda)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int column = blockIdx.y * blockDim.y + threadIdx.y;
    if(row >= n || column >= n || row > column){
        return;
    }

    const int source_row = n - 1 - row;
    const int source_column = n - 1 - column;
    A[row + static_cast<std::size_t>(column) * lda] =
        B[source_row + static_cast<std::size_t>(source_column) * ldb];
}

constexpr int kTileSize = 16;

} // namespace

#endif // DDLA_USE_CUDA || DDLA_USE_HIP

template <DdlaBackend Backend, typename T>
void reverse_upper_to_lower(
    const DdlaHandle_t& handle, int n, const T* A, int lda,
    T* B, int ldb)
{
    assert(n >= 0);
    assert(lda >= (n > 0 ? n : 1));
    assert(ldb >= (n > 0 ? n : 1));
    if(n == 0){
        return;
    }
    assert(A != nullptr);
    assert(B != nullptr);

    if constexpr (Backend == DdlaBackend::CPU){
        // B(r, c) = A(n-1-r, n-1-c) for r >= c, the host form of reverse_upper_to_lower_kernel
        for(int column = 0; column < n; ++column){
            for(int row = column; row < n; ++row){
                B[row + static_cast<std::size_t>(column) * ldb] =
                    A[(n - 1 - row) + static_cast<std::size_t>(n - 1 - column) * lda];
            }
        }
    }else{
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
        const gpuStream_t stream = static_cast<gpuStream_t>(handle->stream);
        using DeviceT = DeviceScalarT<T>;
        static_assert(sizeof(DeviceT) == sizeof(T));
        const dim3 threads(kTileSize, kTileSize);
        const dim3 blocks(
            (n + kTileSize - 1) / kTileSize,
            (n + kTileSize - 1) / kTileSize);
        reverse_upper_to_lower_kernel<DeviceT><<<blocks, threads, 0, stream>>>(
            n, reinterpret_cast<const DeviceT*>(A), lda,
            reinterpret_cast<DeviceT*>(B), ldb);
        DDLA_CHECK(ddlaGetLastError(handle));
#endif
    }
}

template <DdlaBackend Backend, typename T>
void reverse_lower_to_upper(
    const DdlaHandle_t& handle, int n, const T* B, int ldb,
    T* A, int lda)
{
    assert(n >= 0);
    assert(lda >= (n > 0 ? n : 1));
    assert(ldb >= (n > 0 ? n : 1));
    if(n == 0){
        return;
    }
    assert(A != nullptr);
    assert(B != nullptr);

    if constexpr (Backend == DdlaBackend::CPU){
        // A(r, c) = B(n-1-r, n-1-c) for r <= c, the host form of reverse_lower_to_upper_kernel
        for(int column = 0; column < n; ++column){
            for(int row = 0; row <= column; ++row){
                A[row + static_cast<std::size_t>(column) * lda] =
                    B[(n - 1 - row) + static_cast<std::size_t>(n - 1 - column) * ldb];
            }
        }
    }else{
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
        const gpuStream_t stream = static_cast<gpuStream_t>(handle->stream);
        using DeviceT = DeviceScalarT<T>;
        static_assert(sizeof(DeviceT) == sizeof(T));
        const dim3 threads(kTileSize, kTileSize);
        const dim3 blocks(
            (n + kTileSize - 1) / kTileSize,
            (n + kTileSize - 1) / kTileSize);
        reverse_lower_to_upper_kernel<DeviceT><<<blocks, threads, 0, stream>>>(
            n, reinterpret_cast<const DeviceT*>(B), ldb,
            reinterpret_cast<DeviceT*>(A), lda);
        DDLA_CHECK(ddlaGetLastError(handle));
#endif
    }
}

#define INSTANTIATE_REVERSE(BACKEND, TYPE)                                                              \
    template void reverse_upper_to_lower<BACKEND, TYPE>(const DdlaHandle_t&, int, const TYPE*, int, TYPE*, int); \
    template void reverse_lower_to_upper<BACKEND, TYPE>(const DdlaHandle_t&, int, const TYPE*, int, TYPE*, int)

#if DDLA_HAS_CPU
INSTANTIATE_REVERSE(DdlaBackend::CPU, float);
INSTANTIATE_REVERSE(DdlaBackend::CPU, double);
INSTANTIATE_REVERSE(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_REVERSE(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_REVERSE(DdlaBackend::GPU, float);
INSTANTIATE_REVERSE(DdlaBackend::GPU, double);
INSTANTIATE_REVERSE(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_REVERSE(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_REVERSE

} // namespace ddla::detail
