#include "ddla_connector.h"
#include <ddla/ddla_handle_t.h>
#include <ddla/ddla.h>
#include <functional>

#include "mpi_test_init.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{

template <typename T>
T value(int index)
{
    if constexpr (
        std::is_same_v<T, std::complex<float>>
        || std::is_same_v<T, std::complex<double>>)
    {
        using Real = typename T::value_type;
        return T(
            Real((index % 13) - 6) / Real(7),
            Real((index % 9) - 4) / Real(11));
    }
    else
    {
        return T((index % 13) - 6) / T(7);
    }
}

template <typename T>
T conjugate_if_needed(T input, bool conjugate)
{
    if constexpr (
        std::is_same_v<T, std::complex<float>>
        || std::is_same_v<T, std::complex<double>>)
        return conjugate ? std::conj(input) : input;
    else
        return input;
}

template <typename T>
T matrix_element(
    const std::vector<T>& matrix, int leading_dimension,
    int row, int column, char operation)
{
    if (operation == 'N' || operation == 'n')
        return matrix[row + column * leading_dimension];
    return conjugate_if_needed(
        matrix[column + row * leading_dimension],
        operation == 'C' || operation == 'c');
}

template <typename T>
void reference_gemm(
    char transA, char transB,
    int m, int n, int k, T alpha,
    const std::vector<T>& A, int lda,
    const std::vector<T>& B, int ldb,
    T beta, std::vector<T>& C, int ldc)
{
    for (int column = 0; column < n; ++column)
    {
        for (int row = 0; row < m; ++row)
        {
            T sum{};
            for (int inner = 0; inner < k; ++inner)
            {
                sum += matrix_element(A, lda, row, inner, transA)
                    * matrix_element(B, ldb, inner, column, transB);
            }
            C[row + column * ldc] =
                alpha * sum + beta * C[row + column * ldc];
        }
    }
}

template <typename T>
double difference(T lhs, T rhs)
{
    return static_cast<double>(std::abs(lhs - rhs));
}

template <typename T>
class DeviceArray
{
  public:
    DeviceArray() = default;

    DeviceArray(const ddla::DdlaHandle_t& handle, const std::vector<T>& host)
        : handle_(handle)
    {
        if (host.empty())
            return;
        ddla::DDLA_CHECK(ddla::ddlaMalloc(handle_, &pointer_, host.size() * sizeof(T)));
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(
            handle_, pointer_, host.data(), host.size() * sizeof(T)));
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle_));
    }

    DeviceArray(const ddla::DdlaHandle_t& handle, std::size_t count, const T& initial)
        : DeviceArray(handle, std::vector<T>(count, initial))
    {
    }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    DeviceArray(DeviceArray&& other) noexcept
        : handle_(other.handle_),
          pointer_(std::exchange(other.pointer_, nullptr))
    {
    }

    DeviceArray& operator=(DeviceArray&& other) noexcept
    {
        if (this != &other)
        {
            release();
            handle_ = other.handle_;
            pointer_ = std::exchange(other.pointer_, nullptr);
        }
        return *this;
    }

    ~DeviceArray()
    {
        release();
    }

    T* data() const
    {
        return pointer_;
    }

    std::vector<T> download(std::size_t count) const
    {
        std::vector<T> host(count);
        if (count != 0)
        {
            ddla::DDLA_CHECK(ddla::ddlaMemcpy(
                handle_, host.data(), pointer_, count * sizeof(T)));
            ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle_));
        }
        return host;
    }

  private:
    void release()
    {
        if (pointer_ != nullptr)
            ddla::DDLA_CHECK(ddla::ddlaFree(handle_, pointer_));
        pointer_ = nullptr;
    }

    ddla::DdlaHandle_t handle_ = nullptr;
    T* pointer_ = nullptr;
};

template <typename T>
class DeviceMatrixBatch
{
  public:
    DeviceMatrixBatch(const ddla::DdlaHandle_t& handle,
                      const std::vector<std::vector<T>>& matrices)
        : handle_(handle)
    {
        pointers_.resize(matrices.size(), nullptr);
        for (std::size_t i = 0; i < matrices.size(); ++i)
        {
            ddla::DDLA_CHECK(ddla::ddlaMalloc(
                handle_, &pointers_[i], matrices[i].size() * sizeof(T)));
            ddla::DDLA_CHECK(ddla::ddlaMemcpy(
                handle_, pointers_[i], matrices[i].data(),
                matrices[i].size() * sizeof(T)));
        }
        if (!pointers_.empty())
        {
            ddla::DDLA_CHECK(ddla::ddlaMalloc(
                handle_, &device_pointers_, pointers_.size() * sizeof(T*)));
            ddla::DDLA_CHECK(ddla::ddlaMemcpy(
                handle_, device_pointers_, pointers_.data(),
                pointers_.size() * sizeof(T*)));
        }
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle_));
    }

    DeviceMatrixBatch(const DeviceMatrixBatch&) = delete;
    DeviceMatrixBatch& operator=(const DeviceMatrixBatch&) = delete;

    ~DeviceMatrixBatch()
    {
        ddla::DDLA_CHECK(ddla::ddlaFree(handle_, device_pointers_));
        for (T* pointer : pointers_)
            ddla::DDLA_CHECK(ddla::ddlaFree(handle_, pointer));
    }

    T** data() const
    {
        return device_pointers_;
    }

    std::vector<std::vector<T>> download(
        const std::vector<std::size_t>& sizes) const
    {
        std::vector<std::vector<T>> result(sizes.size());
        for (std::size_t i = 0; i < sizes.size(); ++i)
        {
            result[i].resize(sizes[i]);
            ddla::DDLA_CHECK(ddla::ddlaMemcpy(
                handle_, result[i].data(), pointers_[i], sizes[i] * sizeof(T)));
        }
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle_));
        return result;
    }

  private:
    ddla::DdlaHandle_t handle_ = nullptr;
    std::vector<T*> pointers_;
    T** device_pointers_ = nullptr;
};

int storage_rows(char operation, int m, int k)
{
    return (operation == 'N' || operation == 'n') ? m : k;
}

int storage_columns(char operation, int m, int k)
{
    return (operation == 'N' || operation == 'n') ? k : m;
}

template <typename T>
std::vector<T> make_matrix(int leading_dimension, int columns, int seed)
{
    std::vector<T> matrix(
        static_cast<std::size_t>(leading_dimension) * columns);
    for (std::size_t i = 0; i < matrix.size(); ++i)
        matrix[i] = value<T>(seed + static_cast<int>(i));
    return matrix;
}

template <typename T>
double run_standard_case(
    char transA, char transB,
    const ddla::DdlaHandle_t& handle)
{
    const std::vector<int> m{2, 4, 3};
    const std::vector<int> n{3, 2, 5};
    const std::vector<int> k{4, 3, 2};
    std::vector<int> lda(3), ldb(3), ldc(3);
    std::vector<std::vector<T>> A(3), B(3), C(3);
    std::vector<std::size_t> c_sizes(3);

    for (int batch = 0; batch < 3; ++batch)
    {
        lda[batch] = storage_rows(transA, m[batch], k[batch]) + 1;
        ldb[batch] = storage_rows(transB, k[batch], n[batch]) + 2;
        ldc[batch] = m[batch] + 1;
        A[batch] = make_matrix<T>(
            lda[batch], storage_columns(transA, m[batch], k[batch]),
            101 * batch + 1);
        B[batch] = make_matrix<T>(
            ldb[batch], storage_columns(transB, k[batch], n[batch]),
            101 * batch + 29);
        C[batch] = make_matrix<T>(ldc[batch], n[batch], 101 * batch + 53);
        c_sizes[batch] = C[batch].size();
    }

    const T alpha = value<T>(19);
    DeviceArray<int> d_m(handle, m), d_n(handle, n), d_k(handle, k);
    DeviceArray<int> d_lda(handle, lda), d_ldb(handle, ldb), d_ldc(handle, ldc);
    DeviceMatrixBatch<T> d_A(handle, A), d_B(handle, B);
    double maximum_error = 0.0;
    const std::array<T, 3> betas{T{}, T{1}, value<T>(23)};
    for (const T beta : betas)
    {
        auto expected = C;
        for (int batch = 0; batch < 3; ++batch)
        {
            reference_gemm(
                transA, transB, m[batch], n[batch], k[batch],
                alpha, A[batch], lda[batch], B[batch], ldb[batch],
                beta, expected[batch], ldc[batch]);
        }

        DeviceMatrixBatch<T> d_C(handle, C);
        ddla::gemmVbatched(
            handle, transA, transB, d_m.data(), d_n.data(), d_k.data(),
            alpha, const_cast<const T* const*>(d_A.data()), d_lda.data(),
            const_cast<const T* const*>(d_B.data()), d_ldb.data(),
            beta, d_C.data(), d_ldc.data(), 3);
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));

        const auto actual = d_C.download(c_sizes);
        for (int batch = 0; batch < 3; ++batch)
        {
            for (int column = 0; column < n[batch]; ++column)
            {
                for (int row = 0; row < m[batch]; ++row)
                {
                    maximum_error = std::max(
                        maximum_error,
                        difference(
                            actual[batch][row + column * ldc[batch]],
                            expected[batch][row + column * ldc[batch]]));
                }
            }
        }
    }
    return maximum_error;
}

template <typename T>
double run_two_stage_case(bool temporary_on_left, const ddla::DdlaHandle_t& handle)
{
    constexpr int batch_count = 4;
    const std::vector<int> segments{2, 2};
    const std::vector<int> m0{2, 3, 2, 5};
    const std::vector<int> n0{3, 2, 4, 2};
    const std::vector<int> k0{4, 2, 3, 2};
    const std::vector<int> lda0 = m0;
    const std::vector<int> ldb0 = k0;
    const std::vector<int> ldc0 = m0;

    std::vector<int> m1(batch_count), n1(batch_count), k1(batch_count);
    std::vector<int> lda1(batch_count), ldb1(batch_count), ldc1(batch_count);
    std::vector<std::vector<T>> A0(batch_count), B0(batch_count);
    std::vector<std::vector<T>> AB1(batch_count), output(batch_count);
    std::vector<std::vector<T>> expected(batch_count);
    std::vector<std::size_t> output_sizes(batch_count);

    for (int batch = 0; batch < batch_count; ++batch)
    {
        A0[batch] = make_matrix<T>(lda0[batch], k0[batch], 71 * batch + 1);
        B0[batch] = make_matrix<T>(ldb0[batch], n0[batch], 71 * batch + 19);
        if (temporary_on_left)
        {
            m1[batch] = m0[batch];
            n1[batch] = batch == 0 ? 2 : (batch == 1 ? 3 : 1);
            k1[batch] = n0[batch];
        }
        else
        {
            m1[batch] = batch == 0 ? 3 : (batch == 1 ? 2 : 4);
            n1[batch] = n0[batch];
            k1[batch] = m0[batch];
        }
        lda1[batch] = m1[batch];
        ldb1[batch] = k1[batch];
        ldc1[batch] = m1[batch] + 1;
        AB1[batch] = temporary_on_left
            ? make_matrix<T>(ldb1[batch], n1[batch], 71 * batch + 37)
            : make_matrix<T>(lda1[batch], k1[batch], 71 * batch + 37);
        output[batch] =
            make_matrix<T>(ldc1[batch], n1[batch], 71 * batch + 53);
        expected[batch] = output[batch];
        output_sizes[batch] = output[batch].size();
    }

    const T alpha0 = value<T>(17);
    const T alpha1 = value<T>(21);
    const T beta1 = value<T>(25);
    for (int batch = 0; batch < batch_count; ++batch)
    {
        std::vector<T> temporary(
            static_cast<std::size_t>(ldc0[batch]) * n0[batch], T{});
        reference_gemm(
            'N', 'N',
            m0[batch], n0[batch], k0[batch], alpha0,
            A0[batch], lda0[batch], B0[batch], ldb0[batch],
            T{}, temporary, ldc0[batch]);
        if (temporary_on_left)
        {
            reference_gemm(
                'N', 'N',
                m1[batch], n1[batch], k1[batch], alpha1,
                temporary, lda1[batch], AB1[batch], ldb1[batch],
                beta1, expected[batch], ldc1[batch]);
        }
        else
        {
            reference_gemm(
                'N', 'N',
                m1[batch], n1[batch], k1[batch], alpha1,
                AB1[batch], lda1[batch], temporary, ldb1[batch],
                beta1, expected[batch], ldc1[batch]);
        }
    }

    // Match LibRI: each segment packs its matrices independently into one buffer.
    // Segment offsets differ (0,6 versus 0,8), so reusing the pointer-table prefix
    // corrupts the second segment even though both segments fit in the buffer.
    DeviceArray<T> temporary_storage(handle, 18, T{});
    std::vector<T*> temporary_pointers{
        temporary_storage.data(), temporary_storage.data() + 6,
        temporary_storage.data(), temporary_storage.data() + 8};
    DeviceArray<T*> d_tmp(handle, temporary_pointers);

    DeviceArray<int> d_m0(handle, m0), d_n0(handle, n0), d_k0(handle, k0);
    DeviceArray<int> d_lda0(handle, lda0), d_ldb0(handle, ldb0), d_ldc0(handle, ldc0);
    DeviceArray<int> d_m1(handle, m1), d_n1(handle, n1), d_k1(handle, k1);
    DeviceArray<int> d_lda1(handle, lda1), d_ldb1(handle, ldb1), d_ldc1(handle, ldc1);
    DeviceMatrixBatch<T> d_A0(handle, A0);
    DeviceMatrixBatch<T> d_B0(handle, B0);
    DeviceMatrixBatch<T> d_AB1(handle, AB1), d_output(handle, output);

    auto invoke = [&](T beta0, const std::vector<int>& test_segments) {
        ddla::gemmVbatched2s(
            handle, 'N', 'N',
            d_m0.data(), d_n0.data(), d_k0.data(),
            alpha0, const_cast<const T* const*>(d_A0.data()), d_lda0.data(),
            const_cast<const T* const*>(d_B0.data()), d_ldb0.data(),
            beta0, d_tmp.data(), d_ldc0.data(),
            'N', 'N',
            d_m1.data(), d_n1.data(), d_k1.data(),
            alpha1, const_cast<const T* const*>(d_AB1.data()),
            d_lda1.data(), d_ldb1.data(), beta1,
            d_output.data(), d_ldc1.data(), temporary_on_left,
            batch_count, test_segments.data(),
            static_cast<int>(test_segments.size()));
    };

    invoke(T{}, segments);
    ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));

    if constexpr (std::is_same_v<T, double>)
    {
        for (const auto& invalid : std::vector<std::vector<int>>{{0, 4}, {-1, 5}, {1, 1}})
        {
            bool rejected = false;
            try
            {
                invoke(T{}, invalid);
            }
            catch (const std::invalid_argument&)
            {
                rejected = true;
            }
            if (!rejected)
                throw std::runtime_error("invalid two-stage segments were accepted");
        }
        bool rejected_beta = false;
        try
        {
            invoke(T(1), segments);
        }
        catch (const std::invalid_argument&)
        {
            rejected_beta = true;
        }
        if (!rejected_beta)
            throw std::runtime_error("nonzero stage-zero beta was accepted");
    }

    const auto actual = d_output.download(output_sizes);
    double maximum_error = 0.0;
    for (int batch = 0; batch < batch_count; ++batch)
    {
        for (int column = 0; column < n1[batch]; ++column)
        {
            for (int row = 0; row < m1[batch]; ++row)
            {
                maximum_error = std::max(
                    maximum_error,
                    difference(
                        actual[batch][row + column * ldc1[batch]],
                        expected[batch][row + column * ldc1[batch]]));
            }
        }
    }
    return maximum_error;
}

// One two-stage segment larger than the grid's z limit (65535): it must be
// launched in chunks, stage 0 of each chunk before its stage 1.  Each entry is
// a 1x1x1 product, C1 = alpha1 * (A0*B0) * AB1 + beta1 * C1.
double run_large_segment_case(const ddla::DdlaHandle_t& handle)
{
    const int batch = 70000;
    std::vector<double> data(std::size_t(5) * batch);
    for (int i = 0; i < batch; ++i) {
        data[i] = double(i % 7 + 1);            // A0
        data[batch + i] = 2.0;                  // B0
        data[2 * std::size_t(batch) + i] = 0.0; // C0 (temporary)
        data[3 * std::size_t(batch) + i] = 3.0; // AB1
        data[4 * std::size_t(batch) + i] = 1.0; // C1
    }
    DeviceArray<double> d_data(handle, data);
    std::vector<double*> a0(batch), b0(batch), c0(batch), ab1(batch), c1(batch);
    for (int i = 0; i < batch; ++i) {
        a0[i] = d_data.data() + i;
        b0[i] = d_data.data() + batch + i;
        c0[i] = d_data.data() + 2 * std::size_t(batch) + i;
        ab1[i] = d_data.data() + 3 * std::size_t(batch) + i;
        c1[i] = d_data.data() + 4 * std::size_t(batch) + i;
    }
    DeviceArray<double*> d_a0(handle, a0), d_b0(handle, b0), d_c0(handle, c0), d_ab1(handle, ab1), d_c1(handle, c1);
    DeviceArray<int> d_ones(handle, std::vector<int>(batch, 1));
    const std::vector<int> segments{batch};
    const double alpha1 = 1.0, beta1 = 0.5;
    ddla::gemmVbatched2s<ddla::DdlaBackend::GPU, double>(
        handle, 'N', 'N', d_ones.data(), d_ones.data(), d_ones.data(),
        1.0, const_cast<const double* const*>(d_a0.data()), d_ones.data(),
        const_cast<const double* const*>(d_b0.data()), d_ones.data(),
        0.0, d_c0.data(), d_ones.data(),
        'N', 'N', d_ones.data(), d_ones.data(), d_ones.data(),
        alpha1, const_cast<const double* const*>(d_ab1.data()), d_ones.data(), d_ones.data(),
        beta1, d_c1.data(), d_ones.data(), true, batch, segments.data(), 1);
    ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
    std::vector<double> out(batch);
    ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, out.data(), d_data.data() + 4 * std::size_t(batch),
                                      sizeof(double) * batch));
    ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
    double error = 0.0;
    for (int i = 0; i < batch; ++i)
        error = std::max(error, std::abs(out[i] - (alpha1 * 6.0 * double(i % 7 + 1) + beta1)));
    return error;
}

template <typename T>
double run_type(const ddla::DdlaHandle_t& handle, int& case_count)
{
    const std::vector<std::pair<char, char>>
        operations{
            {'N', 'N'},
            {'N', 'T'},
            {'T', 'N'},
            {'T', 'T'},
            {'C', 'N'},
            {'N', 'C'},
            {'C', 'C'},
            {'n', 't'}};

    const char* type_name = [] {
        if constexpr (std::is_same_v<T, float>)
            return "float";
        else if constexpr (std::is_same_v<T, double>)
            return "double";
        else if constexpr (std::is_same_v<T, std::complex<float>>)
            return "complex_float";
        else
            return "complex_double";
    }();

    double maximum_error = 0.0;
    int operation_index = 0;
    for (const auto& [transA, transB] : operations)
    {
        const double error = run_standard_case<T>(transA, transB, handle);
        std::cout << "DDLA_GEMM_VBATCHED_CASE type=" << type_name
                  << " operation_index=" << operation_index++
                  << " error=" << error << '\n';
        maximum_error = std::max(maximum_error, error);
        ++case_count;
    }
    const double left_error = run_two_stage_case<T>(true, handle);
    std::cout << "DDLA_GEMM_VBATCHED_2S_CASE type=" << type_name
              << " temporary=left error=" << left_error << '\n';
    maximum_error = std::max(maximum_error, left_error);
    ++case_count;
    const double right_error = run_two_stage_case<T>(false, handle);
    std::cout << "DDLA_GEMM_VBATCHED_2S_CASE type=" << type_name
              << " temporary=right error=" << right_error << '\n';
    maximum_error = std::max(maximum_error, right_error);
    ++case_count;
    if constexpr (std::is_same_v<T, double>) {
        const double large_error = run_large_segment_case(handle);
        std::cout << "DDLA_GEMM_VBATCHED_2S_CASE type=" << type_name
                  << " segment=70000 error=" << large_error << '\n';
        maximum_error = std::max(maximum_error, large_error);
        ++case_count;
    }

    // An empty batch is a no-op whatever the pointers are.
    ddla::gemmVbatched<ddla::DdlaBackend::GPU, T>(
        handle, 'N', 'N',
        nullptr, nullptr, nullptr, T{}, nullptr, nullptr,
        nullptr, nullptr, T{}, nullptr, nullptr, 0);
    ddla::gemmVbatched2s<ddla::DdlaBackend::GPU, T>(
        handle, 'N', 'N',
        nullptr, nullptr, nullptr, T{}, nullptr, nullptr,
        nullptr, nullptr, T{}, nullptr, nullptr,
        'N', 'N',
        nullptr, nullptr, nullptr, T{}, nullptr, nullptr, nullptr,
        T{}, nullptr, nullptr, false, 0, nullptr, 0);
    ++case_count;

    // A null handle is rejected, even for an empty batch; so is a handle of the other backend and the CPU backend.
    auto rejects = [](const std::function<void()>& call) {
        try { call(); }
        catch (const std::invalid_argument&) { return true; }
        return false;
    };
    if (!rejects([&] {
            ddla::gemmVbatched<ddla::DdlaBackend::GPU, T>(
                nullptr, 'N', 'N', nullptr, nullptr, nullptr, T{}, nullptr, nullptr,
                nullptr, nullptr, T{}, nullptr, nullptr, 0);
        }))
        throw std::runtime_error("a null handle was accepted by gemmVbatched");
    {
        ddla::DdlaHandle_t cpu = nullptr;
        ddla::ddlaInit(cpu, ddla::DdlaBackend::CPU);
        ddla::ddlaSet(cpu, MPI_COMM_WORLD);
        const bool mismatch = rejects([&] {
            ddla::gemmVbatched<ddla::DdlaBackend::GPU, T>(
                cpu, 'N', 'N', nullptr, nullptr, nullptr, T{}, nullptr, nullptr,
                nullptr, nullptr, T{}, nullptr, nullptr, 0);
        });
        const bool cpu_backend = rejects([&] {
            ddla::gemmVbatched<ddla::DdlaBackend::CPU, T>(
                cpu, 'N', 'N', nullptr, nullptr, nullptr, T{}, nullptr, nullptr,
                nullptr, nullptr, T{}, nullptr, nullptr, 0);
        });
        const bool mismatch_2s = rejects([&] {
            ddla::gemmVbatched2s<ddla::DdlaBackend::GPU, T>(
                cpu, 'N', 'N', nullptr, nullptr, nullptr, T{}, nullptr, nullptr,
                nullptr, nullptr, T{}, nullptr, nullptr,
                'N', 'N', nullptr, nullptr, nullptr, T{}, nullptr, nullptr, nullptr,
                T{}, nullptr, nullptr, false, 0, nullptr, 0);
        });
        ddla::ddlaDestroy(cpu);
        if (!mismatch || !cpu_backend || !mismatch_2s)
            throw std::runtime_error("gemmVbatched accepted a CPU handle or the CPU backend");
    }
    case_count += 1;
    return maximum_error;
}

} // namespace

int main(int argc, char** argv)
{
    ddla_test_mpi_init(&argc, &argv);
    int exit_code = 0;
    try
    {
        ddla::DdlaHandle_t handle = nullptr;
        ddla::ddlaInit(handle);
        ddla::ddlaSet(handle, MPI_COMM_WORLD);

        int case_count = 0;
        const double float_error = std::max(
            run_type<float>(handle, case_count),
            run_type<std::complex<float>>(handle, case_count));
        const double double_error = std::max(
            run_type<double>(handle, case_count),
            run_type<std::complex<double>>(handle, case_count));
        ddla::ddlaDestroy(handle);

        if (float_error > 1.0e-5 || double_error > 1.0e-12)
        {
            std::cerr << "DDLA_GEMM_VBATCHED_FAIL cases=" << case_count
                      << " float_error=" << float_error
                      << " double_error=" << double_error << '\n';
            exit_code = 2;
        }
        else
        {
            std::cout << "DDLA_GEMM_VBATCHED_PASS cases=" << case_count
                      << " float_error=" << float_error
                      << " double_error=" << double_error << '\n';
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "DDLA_GEMM_VBATCHED_FAIL exception="
                  << error.what() << '\n';
        exit_code = 3;
    }
    MPI_Finalize();
    return exit_code;
}
