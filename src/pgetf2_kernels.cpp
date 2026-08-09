#include "pgetf2_kernels.h"

#include <climits>
#include <complex>
#include <type_traits>

namespace ddla::detail {

namespace {

constexpr int kPivotThreads = 256;
constexpr int kUpdateThreads = 256;

template <typename T>
struct alignas(sizeof(T) * 2) DeviceComplex {
    T real;
    T imag;

    __host__ __device__ DeviceComplex& operator-=(const DeviceComplex& other)
    {
        real -= other.real;
        imag -= other.imag;
        return *this;
    }
};

template <typename T>
__host__ __device__ DeviceComplex<T> operator*(const DeviceComplex<T>& lhs,
                                                const DeviceComplex<T>& rhs)
{
    return {
        lhs.real * rhs.real - lhs.imag * rhs.imag,
        lhs.real * rhs.imag + lhs.imag * rhs.real
    };
}

template <typename T>
struct DeviceScalar {
    using type = T;
};

template <>
struct DeviceScalar<std::complex<float>> {
    using type = DeviceComplex<float>;
};

template <>
struct DeviceScalar<std::complex<double>> {
    using type = DeviceComplex<double>;
};

template <typename T>
using DeviceScalarT = typename DeviceScalar<T>::type;

template <typename DeviceT>
struct PivotCandidate {
    double metric;
    int local_index;
    // ROCm may use a vector store for complex<float>; keep the value aligned.
    alignas(16) DeviceT value;
};

template <typename T>
__device__ __forceinline__ double component_abs(T value)
{
    return static_cast<double>(value < T(0) ? -value : value);
}

template <typename T>
__device__ __forceinline__ double pivot_metric(T value)
{
    return component_abs(value);
}

template <typename T>
__device__ __forceinline__ double pivot_metric(DeviceComplex<T> value)
{
    return component_abs(value.real) + component_abs(value.imag);
}

__device__ __forceinline__ bool better_candidate(
    double candidate_metric, int candidate_index,
    double current_metric, int current_index)
{
    return candidate_metric > current_metric
        || (candidate_metric == current_metric
            && candidate_index < current_index);
}

template <typename DeviceT>
__global__ void find_local_pivot_kernel(const DeviceT* column, int length,
                                        PivotCandidate<DeviceT>* result)
{
    __shared__ double metrics[kPivotThreads];
    __shared__ int indices[kPivotThreads];

    const int tid = threadIdx.x;
    double best_metric = -1.0;
    int best_index = INT_MAX;
    for(int index = tid; index < length; index += blockDim.x){
        const DeviceT value = column[index];
        const double metric = pivot_metric(value);
        if(better_candidate(metric, index, best_metric, best_index)){
            best_metric = metric;
            best_index = index;
        }
    }
    metrics[tid] = best_metric;
    indices[tid] = best_index;
    __syncthreads();

    for(int stride = kPivotThreads / 2; stride > 0; stride /= 2){
        if(tid < stride
           && better_candidate(metrics[tid + stride], indices[tid + stride],
                               metrics[tid], indices[tid])){
            metrics[tid] = metrics[tid + stride];
            indices[tid] = indices[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0){
        const int selected = indices[0];
        result->metric = metrics[0];
        result->local_index = selected;
        result->value = selected >= 0 && selected < length
                      ? column[selected] : DeviceT{};
    }
    return;
}

template <typename DeviceT>
__global__ void scale_update_kernel(int length_row, int length_col,
                                    DeviceT inverse_pivot,
                                    DeviceT* column,
                                    const DeviceT* pivot_row,
                                    DeviceT* trailing, int lld)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= length_row){
        return;
    }

    const DeviceT multiplier = column[row] * inverse_pivot;
    column[row] = multiplier;
    for(int col = 0; col < length_col; ++col){
        trailing[row + static_cast<size_t>(col) * lld] -= multiplier * pivot_row[col];
    }
    return;
}

// Scalar-type dispatch for host<->device scalar conversion. C++11 has no
// `if constexpr`: the std::complex<T> overloads split/join real and imag
// parts, the plain T overloads pass through; overload resolution (with
// partial ordering picking the more specialized overload) selects at compile
// time.
template <typename T>
DeviceComplex<T> to_device_scalar(const std::complex<T>& value)
{
    return {value.real(), value.imag()};
}

template <typename T>
T to_device_scalar(const T& value)
{
    return value;
}

// to_host_scalar keeps the original single-template shape (callers pass the
// scalar type explicitly, e.g. to_host_scalar<T>), so the complex-vs-real
// split goes through a tag-dispatch helper instead of overloads.
template <typename T>
T to_host_scalar_impl(const DeviceScalarT<T>& value, std::true_type /* complex */)
{
    return T(value.real, value.imag);
}

template <typename T>
T to_host_scalar_impl(const DeviceScalarT<T>& value, std::false_type /* real */)
{
    return value;
}

template <typename T>
T to_host_scalar(const DeviceScalarT<T>& value)
{
    typedef typename std::conditional<
        std::is_same<T, std::complex<float>>::value
        || std::is_same<T, std::complex<double>>::value,
        std::true_type, std::false_type>::type tag;
    return to_host_scalar_impl(value, tag());
}

} // namespace

template <typename T>
std::size_t pgetf2_pivot_workspace_size()
{
    using DeviceT = DeviceScalarT<T>;
    static_assert(sizeof(DeviceT) == sizeof(T));
    return sizeof(PivotCandidate<DeviceT>);
}

template <typename T>
void pgetf2_find_local_pivot(const T* d_column, int length,
                             void* d_workspace, runtimeStream_t stream,
                             double& metric, int& local_index, T& value)
{
    using DeviceT = DeviceScalarT<T>;
    static_assert(sizeof(DeviceT) == sizeof(T));

    auto* d_result = static_cast<PivotCandidate<DeviceT>*>(d_workspace);
    find_local_pivot_kernel<<<1, kPivotThreads, 0, stream>>>(
        reinterpret_cast<const DeviceT*>(d_column), length, d_result);
    RUNTIME_CHECK(runtimeGetLastError());

    PivotCandidate<DeviceT> result{};
    RUNTIME_CHECK(runtimeMemcpyAsync(&result, d_result, sizeof(result),
                                   runtimeMemcpyDeviceToHost, stream));
    RUNTIME_CHECK(runtimeStreamSynchronize(stream));

    metric = result.metric;
    local_index = result.local_index;
    value = to_host_scalar<T>(result.value);
    return;
}

template <typename T>
void pgetf2_scale_update(int length_row, int length_col,
                         const T& inverse_pivot,
                         T* d_column, const T* d_pivot_row,
                         T* d_trailing, int lld,
                         runtimeStream_t stream)
{
    if(length_row <= 0){
        return;
    }

    using DeviceT = DeviceScalarT<T>;
    static_assert(sizeof(DeviceT) == sizeof(T));
    const int blocks = (length_row + kUpdateThreads - 1) / kUpdateThreads;
    scale_update_kernel<<<blocks, kUpdateThreads, 0, stream>>>(
        length_row, length_col, to_device_scalar(inverse_pivot),
        reinterpret_cast<DeviceT*>(d_column),
        reinterpret_cast<const DeviceT*>(d_pivot_row),
        reinterpret_cast<DeviceT*>(d_trailing), lld);
    RUNTIME_CHECK(runtimeGetLastError());
    return;
}

template std::size_t pgetf2_pivot_workspace_size<float>();
template std::size_t pgetf2_pivot_workspace_size<double>();
template std::size_t pgetf2_pivot_workspace_size<std::complex<float>>();
template std::size_t pgetf2_pivot_workspace_size<std::complex<double>>();

template void pgetf2_find_local_pivot<float>(
    const float*, int, void*, runtimeStream_t, double&, int&, float&);
template void pgetf2_find_local_pivot<double>(
    const double*, int, void*, runtimeStream_t, double&, int&, double&);
template void pgetf2_find_local_pivot<std::complex<float>>(
    const std::complex<float>*, int, void*, runtimeStream_t,
    double&, int&, std::complex<float>&);
template void pgetf2_find_local_pivot<std::complex<double>>(
    const std::complex<double>*, int, void*, runtimeStream_t,
    double&, int&, std::complex<double>&);

template void pgetf2_scale_update<float>(
    int, int, const float&, float*, const float*, float*, int, runtimeStream_t);
template void pgetf2_scale_update<double>(
    int, int, const double&, double*, const double*, double*, int, runtimeStream_t);
template void pgetf2_scale_update<std::complex<float>>(
    int, int, const std::complex<float>&, std::complex<float>*,
    const std::complex<float>*, std::complex<float>*, int, runtimeStream_t);
template void pgetf2_scale_update<std::complex<double>>(
    int, int, const std::complex<double>&, std::complex<double>*,
    const std::complex<double>*, std::complex<double>*, int, runtimeStream_t);

} // namespace ddla::detail
