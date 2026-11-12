#include "backend_api.h" // Internal-only wrong-backend implementation checks.
#include <cctype>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "mpi_test_init.h"

#include <ddla/ddla.h>

using namespace ddla;

namespace {

template <typename T>
T value(int i, int j, int salt)
{
    return T(((i * 17 + j * 13 + salt * 7) % 19) - 9);
}

template <>
std::complex<float> value(int i, int j, int salt)
{
    return {static_cast<float>(((i * 17 + j * 13 + salt * 7) % 19) - 9),
            static_cast<float>(((i * 11 + j * 23 + salt * 5) % 17) - 8)};
}

template <>
std::complex<double> value(int i, int j, int salt)
{
    return {static_cast<double>(((i * 17 + j * 13 + salt * 7) % 19) - 9),
            static_cast<double>(((i * 11 + j * 23 + salt * 5) % 17) - 8)};
}

template <typename T>
T conjugate(T x)
{
    return x;
}

template <>
std::complex<float> conjugate(std::complex<float> x)
{
    return std::conj(x);
}

template <>
std::complex<double> conjugate(std::complex<double> x)
{
    return std::conj(x);
}

template <typename T>
T matrix_op(const std::vector<T>& matrix, int ld, char trans, int i, int j)
{
    if (trans == 'N') {
        return matrix[i + j * ld];
    }
    const T x = matrix[j + i * ld];
    return trans == 'C' ? conjugate(x) : x;
}

template <DdlaBackend Backend, typename T>
int check_case(const DdlaHandle_t& handle, char transa, char transb, bool use_empty_template_arguments, bool lowercase = false)
{
    const int m = 7;
    const int n = 5;
    const int k = 9;
    const int rows_a = transa == 'N' ? m : k;
    const int cols_a = transa == 'N' ? k : m;
    const int rows_b = transb == 'N' ? k : n;
    const int cols_b = transb == 'N' ? n : k;
    const int lda = rows_a;
    const int ldb = rows_b;
    const int ldc = m;

    std::vector<T> h_a(static_cast<size_t>(rows_a) * cols_a);
    std::vector<T> h_b(static_cast<size_t>(rows_b) * cols_b);
    std::vector<T> h_c(static_cast<size_t>(m) * n);
    for (int j = 0; j < cols_a; ++j)
        for (int i = 0; i < rows_a; ++i)
            h_a[i + j * lda] = value<T>(i, j, 1);
    for (int j = 0; j < cols_b; ++j)
        for (int i = 0; i < rows_b; ++i)
            h_b[i + j * ldb] = value<T>(i, j, 2);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            h_c[i + j * ldc] = value<T>(i, j, 3);

    std::vector<T> reference = h_c;
    T alpha = T(0.75);
    T beta = T(-0.25);
    if constexpr (std::is_same_v<T, std::complex<float>> ||
                  std::is_same_v<T, std::complex<double>>) {
        alpha = T(0.75, -0.125);
        beta = T(-0.25, 0.375);
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            T sum = T(0);
            for (int l = 0; l < k; ++l) {
                sum += matrix_op(h_a, lda, transa, i, l) *
                       matrix_op(h_b, ldb, transb, l, j);
            }
            reference[i + j * ldc] = alpha * sum + beta * reference[i + j * ldc];
        }
    }

    T* a = nullptr;
    T* b = nullptr;
    T* c = nullptr;
    int status = 0;
    status |= static_cast<int>(ddlaMalloc(handle, reinterpret_cast<void**>(&a), h_a.size() * sizeof(T)));
    status |= static_cast<int>(ddlaMalloc(handle, reinterpret_cast<void**>(&b), h_b.size() * sizeof(T)));
    status |= static_cast<int>(ddlaMalloc(handle, reinterpret_cast<void**>(&c), h_c.size() * sizeof(T)));
    status |= static_cast<int>(ddlaMemcpy(handle, a, h_a.data(), h_a.size() * sizeof(T)));
    status |= static_cast<int>(ddlaMemcpy(handle, b, h_b.data(), h_b.size() * sizeof(T)));
    status |= static_cast<int>(ddlaMemcpy(handle, c, h_c.data(), h_c.size() * sizeof(T)));
    status |= static_cast<int>(ddlaSynchronize(handle));
    if (status != 0) return 1;

    // Fortran LSAME is case-insensitive; both backends must agree on that.
    const char fa = lowercase ? static_cast<char>(std::tolower(static_cast<unsigned char>(transa))) : transa;
    const char fb = lowercase ? static_cast<char>(std::tolower(static_cast<unsigned char>(transb))) : transb;
    if (use_empty_template_arguments) {
        gemm<>(handle, fa, fb, m, n, k,
               alpha, a, lda, b, ldb, beta, c, ldc);
    } else {
        gemm(handle, fa, fb, m, n, k,
                      alpha, a, lda, b, ldb, beta, c, ldc);
    }
    status |= static_cast<int>(ddlaSynchronize(handle));
    status |= static_cast<int>(ddlaMemcpy(handle, h_c.data(), c, h_c.size() * sizeof(T)));
    status |= static_cast<int>(ddlaSynchronize(handle));
    status |= static_cast<int>(ddlaFree(handle, a));
    status |= static_cast<int>(ddlaFree(handle, b));
    status |= static_cast<int>(ddlaFree(handle, c));
    if (status != 0) return 1;

    double local_error = 0.0;
    double local_scale = 0.0;
    for (size_t i = 0; i < h_c.size(); ++i) {
        local_error = std::max(local_error, static_cast<double>(std::abs(h_c[i] - reference[i])));
        local_scale = std::max(local_scale, static_cast<double>(std::abs(reference[i])));
    }
    double global_error = 0.0;
    double global_scale = 0.0;
    MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_scale, &global_scale, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const bool single = std::is_same_v<T, float> ||
                        std::is_same_v<T, std::complex<float>>;
    const double tolerance = single ? 2e-4 : 2e-11;
    return global_error <= tolerance * std::max(1.0, global_scale) ? 0 : 1;
}

template <DdlaBackend Backend, typename T>
int run_type(const DdlaHandle_t& handle)
{
    int failures = 0;
    for (char transa : {'N', 'T', 'C'})
        for (char transb : {'N', 'T', 'C'})
            failures += check_case<Backend, T>(handle, transa, transb, false);
    for (char transa : {'N', 'T', 'C'})
        failures += check_case<Backend, T>(handle, transa, 'N', false, true);
    {
        // An invalid flag must be rejected, not run as a conjugate transpose.
        T dummy = T(0);
        bool caught = false;
        try { gemm(handle, 'X', 'N', 1, 1, 1, T(1), &dummy, 1, &dummy, 1, T(0), &dummy, 1); }
        catch (const std::invalid_argument&) { caught = true; }
        if (!caught) ++failures;
    }
    // Empty template arguments deduce the scalar type on either handle.
    failures += check_case<Backend, T>(handle, 'N', 'N', true);
    return failures;
}

template <DdlaBackend Backend>
int run_backend()
{
    DdlaHandle_t handle = nullptr;
    ddlaInit(handle, Backend);
    ddlaSet(handle, MPI_COMM_WORLD);
    int failures = 0;
    failures += run_type<Backend, float>(handle);
    failures += run_type<Backend, double>(handle);
    failures += run_type<Backend, std::complex<float>>(handle);
    failures += run_type<Backend, std::complex<double>>(handle);
    ddlaDestroy(handle);
    return failures;
}

#if DDLA_HAS_GPU
int check_backend_mismatch()
{
    DdlaHandle_t cpu = nullptr;
    DdlaHandle_t gpu = nullptr;
    ddlaInit(cpu, DdlaBackend::CPU);
    ddlaInit(gpu, DdlaBackend::GPU);
    ddlaSet(cpu, MPI_COMM_WORLD);
    ddlaSet(gpu, MPI_COMM_WORLD);
    float data = 1.0f;
    bool cpu_rejected_gpu = false;
    bool gpu_rejected_cpu = false;
    try {
        gemm<DdlaBackend::CPU>(gpu, 'N', 'N', 1, 1, 1,
                               data, &data, 1, &data, 1, data, &data, 1);
    } catch (const std::runtime_error& e) {
        cpu_rejected_gpu = std::string(e.what()).find("does not match") != std::string::npos;
    }
    try {
        gemm<DdlaBackend::GPU>(cpu, 'N', 'N', 1, 1, 1,
                               data, &data, 1, &data, 1, data, &data, 1);
    } catch (const std::runtime_error& e) {
        gpu_rejected_cpu = std::string(e.what()).find("does not match") != std::string::npos;
    }
    ddlaDestroy(cpu);
    ddlaDestroy(gpu);
    return cpu_rejected_gpu && gpu_rejected_cpu ? 0 : 1;
}
#endif

} // namespace

int main(int argc, char** argv)
{
    ddla_test_mpi_init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#if DDLA_HAS_GPU
    static_assert(default_backend_v == DdlaBackend::GPU,
                  "GPU-capable builds must default to GPU");
#else
    static_assert(default_backend_v == DdlaBackend::CPU,
                  "CPU-only builds must default to CPU");
#endif

    int failures = 0;
    failures += run_backend<DdlaBackend::CPU>();
#if DDLA_HAS_GPU
    failures += run_backend<DdlaBackend::GPU>();
#endif
#if DDLA_HAS_GPU
    failures += check_backend_mismatch();
#endif

    int global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << (global_failures == 0 ? "test_backend_gemm passed" :
                                             "test_backend_gemm failed")
                  << std::endl;
    }
    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}
