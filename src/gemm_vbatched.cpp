#include <ddla/gemmVbatched.h>
#include "ddla_stream_impl.h"

#include "vbatched/gemm_vbatched_compat.h"

#include <array>
#include <complex>
#include <string>
#include <stdexcept>
#include <type_traits>

// Device allocations are sufficiently aligned for every scalar type below,
// while equal element sizes keep each matrix element aligned when the storage
// is viewed as the private thrust complex type.
static_assert(sizeof(std::complex<float>) == sizeof(DdlaFloatComplex));
static_assert(std::is_trivially_copyable<std::complex<float>>::value);
static_assert(std::is_trivially_copyable<DdlaFloatComplex>::value);
static_assert(std::is_standard_layout<std::complex<float>>::value);
static_assert(std::is_standard_layout<DdlaFloatComplex>::value);
static_assert(alignof(DdlaFloatComplex) >= alignof(std::complex<float>));
static_assert(
    sizeof(std::complex<float>) % alignof(DdlaFloatComplex) == 0);
static_assert(sizeof(std::complex<double>) == sizeof(DdlaDoubleComplex));
static_assert(std::is_trivially_copyable<std::complex<double>>::value);
static_assert(std::is_trivially_copyable<DdlaDoubleComplex>::value);
static_assert(std::is_standard_layout<std::complex<double>>::value);
static_assert(std::is_standard_layout<DdlaDoubleComplex>::value);
static_assert(alignof(DdlaDoubleComplex) >= alignof(std::complex<double>));
static_assert(
    sizeof(std::complex<double>) % alignof(DdlaDoubleComplex) == 0);

#define DECLARE_VBATCHED_CORE(prefix, scalar_type)                              \
extern "C" void ddla_internal_##prefix##gemm_vbatched_core(                    \
    ddla::deblasOperation_t, ddla::deblasOperation_t, int, int, int,           \
    int*, int*, int*, scalar_type,                                              \
    const scalar_type* const*, int, int, int*,                                  \
    const scalar_type* const*, int, int, int*,                                  \
    scalar_type, scalar_type**, int, int, int*,                                 \
    int, ddla::runtimeStream_t);                                                \
extern "C" void ddla_internal_##prefix##gemm_vbatched_core_2s(                 \
    ddla::deblasOperation_t, ddla::deblasOperation_t, int, int, int,           \
    int*, int*, int*, scalar_type,                                              \
    const scalar_type* const*, int, int, int*,                                  \
    const scalar_type* const*, int, int, int*,                                  \
    scalar_type, scalar_type**, int, int, int*,                                 \
    ddla::deblasOperation_t, ddla::deblasOperation_t, int, int, int,           \
    int*, int*, int*, scalar_type,                                              \
    const scalar_type* const*, int, int, int*,                                  \
    int, int, int*, scalar_type, scalar_type**,                                 \
    int, int, int*, bool, int,                                                  \
    const int*, ddla::runtimeStream_t)

DECLARE_VBATCHED_CORE(s, float);
DECLARE_VBATCHED_CORE(d, double);
DECLARE_VBATCHED_CORE(c, DdlaFloatComplex);
DECLARE_VBATCHED_CORE(z, DdlaDoubleComplex);

#undef DECLARE_VBATCHED_CORE

namespace
{

struct MaxDimensions
{
    int m;
    int n;
    int k;
};

__global__ void validate_and_max_dimensions(
    ddla::deblasOperation_t transA, ddla::deblasOperation_t transB,
    const int* m, const int* n, const int* k,
    const int* lda, const int* ldb, const int* ldc,
    int batch_count, int* summary)
{
    const int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= batch_count)
        return;

    const int mb = m[batch];
    const int nb = n[batch];
    const int kb = k[batch];
    if (mb < 0 || nb < 0 || kb < 0)
    {
        atomicCAS(summary + 3, 0, batch + 1);
        return;
    }

    const int lda_required =
        transA == ddla::DEBLAS_OP_N ? (mb > 1 ? mb : 1)
                                    : (kb > 1 ? kb : 1);
    const int ldb_required =
        transB == ddla::DEBLAS_OP_N ? (kb > 1 ? kb : 1)
                                    : (nb > 1 ? nb : 1);
    const int ldc_required = mb > 1 ? mb : 1;
    if (lda[batch] < lda_required || ldb[batch] < ldb_required
        || ldc[batch] < ldc_required)
    {
        atomicCAS(summary + 3, 0, batch + 1);
        return;
    }

    atomicMax(summary + 0, mb);
    atomicMax(summary + 1, nb);
    atomicMax(summary + 2, kb);
}

bool valid_operation(ddla::deblasOperation_t operation)
{
    return operation == ddla::DEBLAS_OP_N || operation == ddla::DEBLAS_OP_T
        || operation == ddla::DEBLAS_OP_C;
}

MaxDimensions validate_dimensions(
    ddla::deblasOperation_t transA, ddla::deblasOperation_t transB,
    int* d_m, int* d_n, int* d_k,
    int* d_lda, int* d_ldb, int* d_ldc,
    int batch_count, ddla::runtimeStream_t stream)
{
    if (!valid_operation(transA) || !valid_operation(transB))
        throw std::invalid_argument("gemmVbatched: invalid transpose operation");
    if (d_m == nullptr || d_n == nullptr || d_k == nullptr
        || d_lda == nullptr || d_ldb == nullptr || d_ldc == nullptr)
        throw std::invalid_argument("gemmVbatched: null dimension array");

    int* d_summary = nullptr;
    ddla::RUNTIME_CHECK(
        ddla::runtimeMallocAsync(&d_summary, 4 * sizeof(int), stream));
    ddla::RUNTIME_CHECK(
        ddla::runtimeMemsetAsync(d_summary, 0, 4 * sizeof(int), stream));

    constexpr int block_size = 256;
    const int grid_size = (batch_count + block_size - 1) / block_size;
    DDLA_LAUNCH_KERNEL(
        (validate_and_max_dimensions), dim3(grid_size), dim3(block_size), 0,
        stream, transA, transB, d_m, d_n, d_k, d_lda, d_ldb, d_ldc,
        batch_count, d_summary);

    std::array<int, 4> summary{};
    ddla::RUNTIME_CHECK(ddla::runtimeMemcpyAsync(
        summary.data(), d_summary, 4 * sizeof(int),
        ddla::runtimeMemcpyDeviceToHost, stream));
    ddla::RUNTIME_CHECK(ddla::runtimeStreamSynchronize(stream));
    ddla::RUNTIME_CHECK(ddla::runtimeFreeAsync(d_summary, stream));

    if (summary[3] != 0)
        throw std::invalid_argument(
            "gemmVbatched: invalid dimensions at batch "
            + std::to_string(summary[3] - 1));
    return {summary[0], summary[1], summary[2]};
}

// Scalar-type overload dispatch for the vbatched core launch. C++11 has
// no `if constexpr`, so the former is_same_v<> chains become plain
// overloads on the scalar type: overload resolution picks the right
// precision core at compile time.
void launch_core_float(
    ddla::deblasOperation_t transA, ddla::deblasOperation_t transB,
    const MaxDimensions& maximum,
    int* d_m, int* d_n, int* d_k,
    float alpha, const float* const* d_A_array, int* d_lda,
    const float* const* d_B_array, int* d_ldb,
    float beta, float** d_C_array, int* d_ldc,
    int batch_count, ddla::runtimeStream_t stream)
{
    ddla_internal_sgemm_vbatched_core(
        transA, transB, maximum.m, maximum.n, maximum.k,
        d_m, d_n, d_k, alpha, d_A_array, 0, 0, d_lda,
        d_B_array, 0, 0, d_ldb, beta, d_C_array, 0, 0, d_ldc,
        batch_count, stream);
}

void launch_core_double(
    ddla::deblasOperation_t transA, ddla::deblasOperation_t transB,
    const MaxDimensions& maximum,
    int* d_m, int* d_n, int* d_k,
    double alpha, const double* const* d_A_array, int* d_lda,
    const double* const* d_B_array, int* d_ldb,
    double beta, double** d_C_array, int* d_ldc,
    int batch_count, ddla::runtimeStream_t stream)
{
    ddla_internal_dgemm_vbatched_core(
        transA, transB, maximum.m, maximum.n, maximum.k,
        d_m, d_n, d_k, alpha, d_A_array, 0, 0, d_lda,
        d_B_array, 0, 0, d_ldb, beta, d_C_array, 0, 0, d_ldc,
        batch_count, stream);
}

void launch_core_complex_float(
    ddla::deblasOperation_t transA, ddla::deblasOperation_t transB,
    const MaxDimensions& maximum,
    int* d_m, int* d_n, int* d_k,
    complex_float alpha, const complex_float* const* d_A_array, int* d_lda,
    const complex_float* const* d_B_array, int* d_ldb,
    complex_float beta, complex_float** d_C_array, int* d_ldc,
    int batch_count, ddla::runtimeStream_t stream)
{
    const DdlaFloatComplex alpha_internal =
        DDLA_C_MAKE(alpha.real(), alpha.imag());
    const DdlaFloatComplex beta_internal =
        DDLA_C_MAKE(beta.real(), beta.imag());
    ddla_internal_cgemm_vbatched_core(
        transA, transB, maximum.m, maximum.n, maximum.k,
        d_m, d_n, d_k, alpha_internal,
        reinterpret_cast<const DdlaFloatComplex* const*>(d_A_array),
        0, 0, d_lda,
        reinterpret_cast<const DdlaFloatComplex* const*>(d_B_array),
        0, 0, d_ldb, beta_internal,
        reinterpret_cast<DdlaFloatComplex**>(d_C_array),
        0, 0, d_ldc, batch_count, stream);
}

void launch_core_complex_double(
    ddla::deblasOperation_t transA, ddla::deblasOperation_t transB,
    const MaxDimensions& maximum,
    int* d_m, int* d_n, int* d_k,
    complex_double alpha, const complex_double* const* d_A_array, int* d_lda,
    const complex_double* const* d_B_array, int* d_ldb,
    complex_double beta, complex_double** d_C_array, int* d_ldc,
    int batch_count, ddla::runtimeStream_t stream)
{
    const DdlaDoubleComplex alpha_internal =
        DDLA_Z_MAKE(alpha.real(), alpha.imag());
    const DdlaDoubleComplex beta_internal =
        DDLA_Z_MAKE(beta.real(), beta.imag());
    ddla_internal_zgemm_vbatched_core(
        transA, transB, maximum.m, maximum.n, maximum.k,
        d_m, d_n, d_k, alpha_internal,
        reinterpret_cast<const DdlaDoubleComplex* const*>(d_A_array),
        0, 0, d_lda,
        reinterpret_cast<const DdlaDoubleComplex* const*>(d_B_array),
        0, 0, d_ldb, beta_internal,
        reinterpret_cast<DdlaDoubleComplex**>(d_C_array),
        0, 0, d_ldc, batch_count, stream);
}


// Two-stage (gemmVbatched2s) scalar-type overload dispatch, same
// rationale as launch_core above.
void launch_core_2s_float(
    ddla::deblasOperation_t transA_0, ddla::deblasOperation_t transB_0,
    const MaxDimensions& maximum_0,
    int* d_m_0, int* d_n_0, int* d_k_0,
    float alpha_0, const float* const* d_A_array_0, int* d_lda_0,
    const float* const* d_B_array_0, int* d_ldb_0,
    float beta_0, float** d_C_array_0, int* d_ldc_0,
    ddla::deblasOperation_t transA_1, ddla::deblasOperation_t transB_1,
    const MaxDimensions& maximum_1,
    int* d_m_1, int* d_n_1, int* d_k_1,
    float alpha_1, const float* const* d_AB_array_1,
    int* d_lda_1, int* d_ldb_1,
    float beta_1, float** d_C_array_1, int* d_ldc_1,
    bool C0_left, int batch_count, const int* segment_sizes,
    ddla::runtimeStream_t stream)
{
    ddla_internal_sgemm_vbatched_core_2s(
        transA_0, transB_0, maximum_0.m, maximum_0.n, maximum_0.k,
        d_m_0, d_n_0, d_k_0, alpha_0,
        d_A_array_0, 0, 0, d_lda_0, d_B_array_0, 0, 0, d_ldb_0,
        beta_0, d_C_array_0, 0, 0, d_ldc_0,
        transA_1, transB_1, maximum_1.m, maximum_1.n, maximum_1.k,
        d_m_1, d_n_1, d_k_1, alpha_1,
        d_AB_array_1, 0, 0, d_lda_1, 0, 0, d_ldb_1,
        beta_1, d_C_array_1, 0, 0, d_ldc_1, C0_left,
        batch_count, segment_sizes, stream);
}

void launch_core_2s_double(
    ddla::deblasOperation_t transA_0, ddla::deblasOperation_t transB_0,
    const MaxDimensions& maximum_0,
    int* d_m_0, int* d_n_0, int* d_k_0,
    double alpha_0, const double* const* d_A_array_0, int* d_lda_0,
    const double* const* d_B_array_0, int* d_ldb_0,
    double beta_0, double** d_C_array_0, int* d_ldc_0,
    ddla::deblasOperation_t transA_1, ddla::deblasOperation_t transB_1,
    const MaxDimensions& maximum_1,
    int* d_m_1, int* d_n_1, int* d_k_1,
    double alpha_1, const double* const* d_AB_array_1,
    int* d_lda_1, int* d_ldb_1,
    double beta_1, double** d_C_array_1, int* d_ldc_1,
    bool C0_left, int batch_count, const int* segment_sizes,
    ddla::runtimeStream_t stream)
{
    ddla_internal_dgemm_vbatched_core_2s(
        transA_0, transB_0, maximum_0.m, maximum_0.n, maximum_0.k,
        d_m_0, d_n_0, d_k_0, alpha_0,
        d_A_array_0, 0, 0, d_lda_0, d_B_array_0, 0, 0, d_ldb_0,
        beta_0, d_C_array_0, 0, 0, d_ldc_0,
        transA_1, transB_1, maximum_1.m, maximum_1.n, maximum_1.k,
        d_m_1, d_n_1, d_k_1, alpha_1,
        d_AB_array_1, 0, 0, d_lda_1, 0, 0, d_ldb_1,
        beta_1, d_C_array_1, 0, 0, d_ldc_1, C0_left,
        batch_count, segment_sizes, stream);
}

void launch_core_2s_complex_float(
    ddla::deblasOperation_t transA_0, ddla::deblasOperation_t transB_0,
    const MaxDimensions& maximum_0,
    int* d_m_0, int* d_n_0, int* d_k_0,
    complex_float alpha_0, const complex_float* const* d_A_array_0, int* d_lda_0,
    const complex_float* const* d_B_array_0, int* d_ldb_0,
    complex_float beta_0, complex_float** d_C_array_0, int* d_ldc_0,
    ddla::deblasOperation_t transA_1, ddla::deblasOperation_t transB_1,
    const MaxDimensions& maximum_1,
    int* d_m_1, int* d_n_1, int* d_k_1,
    complex_float alpha_1, const complex_float* const* d_AB_array_1,
    int* d_lda_1, int* d_ldb_1,
    complex_float beta_1, complex_float** d_C_array_1, int* d_ldc_1,
    bool C0_left, int batch_count, const int* segment_sizes,
    ddla::runtimeStream_t stream)
{
    ddla_internal_cgemm_vbatched_core_2s(
        transA_0, transB_0, maximum_0.m, maximum_0.n, maximum_0.k,
        d_m_0, d_n_0, d_k_0,
        DDLA_C_MAKE(alpha_0.real(), alpha_0.imag()),
        reinterpret_cast<const DdlaFloatComplex* const*>(d_A_array_0),
        0, 0, d_lda_0,
        reinterpret_cast<const DdlaFloatComplex* const*>(d_B_array_0),
        0, 0, d_ldb_0,
        DDLA_C_MAKE(beta_0.real(), beta_0.imag()),
        reinterpret_cast<DdlaFloatComplex**>(d_C_array_0),
        0, 0, d_ldc_0,
        transA_1, transB_1, maximum_1.m, maximum_1.n, maximum_1.k,
        d_m_1, d_n_1, d_k_1,
        DDLA_C_MAKE(alpha_1.real(), alpha_1.imag()),
        reinterpret_cast<const DdlaFloatComplex* const*>(d_AB_array_1),
        0, 0, d_lda_1, 0, 0, d_ldb_1,
        DDLA_C_MAKE(beta_1.real(), beta_1.imag()),
        reinterpret_cast<DdlaFloatComplex**>(d_C_array_1),
        0, 0, d_ldc_1, C0_left, batch_count, segment_sizes, stream);
}

void launch_core_2s_complex_double(
    ddla::deblasOperation_t transA_0, ddla::deblasOperation_t transB_0,
    const MaxDimensions& maximum_0,
    int* d_m_0, int* d_n_0, int* d_k_0,
    complex_double alpha_0, const complex_double* const* d_A_array_0, int* d_lda_0,
    const complex_double* const* d_B_array_0, int* d_ldb_0,
    complex_double beta_0, complex_double** d_C_array_0, int* d_ldc_0,
    ddla::deblasOperation_t transA_1, ddla::deblasOperation_t transB_1,
    const MaxDimensions& maximum_1,
    int* d_m_1, int* d_n_1, int* d_k_1,
    complex_double alpha_1, const complex_double* const* d_AB_array_1,
    int* d_lda_1, int* d_ldb_1,
    complex_double beta_1, complex_double** d_C_array_1, int* d_ldc_1,
    bool C0_left, int batch_count, const int* segment_sizes,
    ddla::runtimeStream_t stream)
{
    ddla_internal_zgemm_vbatched_core_2s(
        transA_0, transB_0, maximum_0.m, maximum_0.n, maximum_0.k,
        d_m_0, d_n_0, d_k_0,
        DDLA_Z_MAKE(alpha_0.real(), alpha_0.imag()),
        reinterpret_cast<const DdlaDoubleComplex* const*>(d_A_array_0),
        0, 0, d_lda_0,
        reinterpret_cast<const DdlaDoubleComplex* const*>(d_B_array_0),
        0, 0, d_ldb_0,
        DDLA_Z_MAKE(beta_0.real(), beta_0.imag()),
        reinterpret_cast<DdlaDoubleComplex**>(d_C_array_0),
        0, 0, d_ldc_0,
        transA_1, transB_1, maximum_1.m, maximum_1.n, maximum_1.k,
        d_m_1, d_n_1, d_k_1,
        DDLA_Z_MAKE(alpha_1.real(), alpha_1.imag()),
        reinterpret_cast<const DdlaDoubleComplex* const*>(d_AB_array_1),
        0, 0, d_lda_1, 0, 0, d_ldb_1,
        DDLA_Z_MAKE(beta_1.real(), beta_1.imag()),
        reinterpret_cast<DdlaDoubleComplex**>(d_C_array_1),
        0, 0, d_ldc_1, C0_left, batch_count, segment_sizes, stream);
}


} // namespace

namespace ddla
{

template <typename T>
void gemmVbatched(
    deblasOperation_t transA, deblasOperation_t transB,
    int* d_m, int* d_n, int* d_k,
    T alpha, const T* const* d_A_array, int* d_lda,
    const T* const* d_B_array, int* d_ldb,
    T beta, T** d_C_array, int* d_ldc,
    int batch_count, const DdlaHandle_t& handle)
{
    static_assert(
        std::is_same<T, float>::value || std::is_same<T, double>::value
        || std::is_same<T, std::complex<float>>::value
        || std::is_same<T, std::complex<double>>::value,
        "gemmVbatched supports float, double, and their complex types");

    if (batch_count < 0)
        throw std::invalid_argument("gemmVbatched: negative batch count");
    if (batch_count == 0)
        return;
    if (handle == nullptr || handle->stream == nullptr)
        throw std::invalid_argument("gemmVbatched: invalid DDLA handle");
    if (d_A_array == nullptr || d_B_array == nullptr || d_C_array == nullptr)
        throw std::invalid_argument("gemmVbatched: null matrix pointer array");

    const MaxDimensions maximum = validate_dimensions(
        transA, transB, d_m, d_n, d_k, d_lda, d_ldb, d_ldc,
        batch_count, handle->stream);
    launch_core(
        transA, transB, maximum, d_m, d_n, d_k,
        alpha, d_A_array, d_lda, d_B_array, d_ldb,
        beta, d_C_array, d_ldc, batch_count, handle->stream);
    RUNTIME_CHECK(runtimeGetLastError());
}

template <typename T>
void gemmVbatched2s(
    deblasOperation_t transA_0, deblasOperation_t transB_0,
    int* d_m_0, int* d_n_0, int* d_k_0,
    T alpha_0, const T* const* d_A_array_0, int* d_lda_0,
    const T* const* d_B_array_0, int* d_ldb_0,
    T beta_0, T** d_C_array_0, int* d_ldc_0,
    deblasOperation_t transA_1, deblasOperation_t transB_1,
    int* d_m_1, int* d_n_1, int* d_k_1,
    T alpha_1, const T* const* d_AB_array_1,
    int* d_lda_1, int* d_ldb_1,
    T beta_1, T** d_C_array_1, int* d_ldc_1,
    bool C0_left, int batch_count,
    const int* segment_sizes, int segment_count,
    const DdlaHandle_t& handle)
{
    if (batch_count < 0)
        throw std::invalid_argument("gemmVbatched2s: negative batch count");
    if (batch_count == 0)
        return;
    if (beta_0 != T{})
        throw std::invalid_argument("gemmVbatched2s: stage-zero beta must be zero");
    if (segment_sizes == nullptr || segment_count <= 0)
        throw std::invalid_argument("gemmVbatched2s: missing segment sizes");

    long long segment_total = 0;
    for (int segment = 0; segment < segment_count; ++segment)
    {
        if (segment_sizes[segment] <= 0)
            throw std::invalid_argument(
                "gemmVbatched2s: segment sizes must be positive");
        segment_total += segment_sizes[segment];
    }
    if (segment_total != batch_count)
        throw std::invalid_argument(
            "gemmVbatched2s: segment total does not match batch count");
    if (handle == nullptr || handle->stream == nullptr)
        throw std::invalid_argument("gemmVbatched2s: invalid DDLA handle");
    if (d_A_array_0 == nullptr || d_B_array_0 == nullptr
        || d_C_array_0 == nullptr || d_AB_array_1 == nullptr
        || d_C_array_1 == nullptr)
        throw std::invalid_argument("gemmVbatched2s: null matrix pointer array");

    const MaxDimensions maximum_0 = validate_dimensions(
        transA_0, transB_0, d_m_0, d_n_0, d_k_0,
        d_lda_0, d_ldb_0, d_ldc_0, batch_count, handle->stream);
    const MaxDimensions maximum_1 = validate_dimensions(
        transA_1, transB_1, d_m_1, d_n_1, d_k_1,
        d_lda_1, d_ldb_1, d_ldc_1, batch_count, handle->stream);

    launch_core_2s(
        transA_0, transB_0, maximum_0,
        d_m_0, d_n_0, d_k_0, alpha_0,
        d_A_array_0, d_lda_0, d_B_array_0, d_ldb_0,
        beta_0, d_C_array_0, d_ldc_0,
        transA_1, transB_1, maximum_1,
        d_m_1, d_n_1, d_k_1, alpha_1,
        d_AB_array_1, d_lda_1, d_ldb_1,
        beta_1, d_C_array_1, d_ldc_1,
        C0_left, batch_count, segment_sizes, handle->stream);
    RUNTIME_CHECK(runtimeGetLastError());
}

#define INSTANTIATE_VBATCHED(type)                                              \
template void gemmVbatched<type>(                                              \
    deblasOperation_t, deblasOperation_t, int*, int*, int*, type,              \
    const type* const*, int*, const type* const*, int*, type, type**, int*,     \
    int, const DdlaHandle_t&);                                                  \
template void gemmVbatched2s<type>(                                            \
    deblasOperation_t, deblasOperation_t, int*, int*, int*, type,              \
    const type* const*, int*, const type* const*, int*, type, type**, int*,     \
    deblasOperation_t, deblasOperation_t, int*, int*, int*, type,              \
    const type* const*, int*, int*, type, type**, int*, bool, int,             \
    const int*, int, const DdlaHandle_t&)

INSTANTIATE_VBATCHED(float);
INSTANTIATE_VBATCHED(double);
INSTANTIATE_VBATCHED(std::complex<float>);
INSTANTIATE_VBATCHED(std::complex<double>);

#undef INSTANTIATE_VBATCHED

} // namespace ddla
