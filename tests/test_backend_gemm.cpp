#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <mpi.h>

#include <ddla/ddla.h>
#include <ddla/gemm.h>

using namespace ddla;

namespace {

// C++11 has no `if constexpr`: complex-valued alpha/beta are assigned only
// for complex T; the tag overload keeps the two-argument T(...) constructor
// from ever being instantiated for real T.
template <typename T>
void set_complex_alpha_beta_impl(T& alpha, T& beta,
                                 double ar, double ai, double br, double bi,
                                 std::true_type)
{
    alpha = T(ar, ai);
    beta = T(br, bi);
}

template <typename T>
void set_complex_alpha_beta_impl(T&, T&, double, double, double, double,
                                 std::false_type)
{
}

template <typename T>
void set_complex_alpha_beta(T& alpha, T& beta,
                            double ar, double ai, double br, double bi)
{
    set_complex_alpha_beta_impl(
        alpha, beta, ar, ai, br, bi,
        std::integral_constant<bool,
            std::is_same<T, std::complex<float>>::value
            || std::is_same<T, std::complex<double>>::value>());
}

template <typename T>
T value(int i, int j, int salt)
{
    return T(((i * 17 + j * 13 + salt * 7) % 19) - 9);
}template <>
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
int check_case(const DdlaHandle_t& handle, char transa, char transb, bool use_default)
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
    set_complex_alpha_beta(alpha, beta, 0.75, -0.125, -0.25, 0.375);
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
    status |= ddla_malloc(reinterpret_cast<void**>(&a), h_a.size() * sizeof(T), handle);
    status |= ddla_malloc(reinterpret_cast<void**>(&b), h_b.size() * sizeof(T), handle);
    status |= ddla_malloc(reinterpret_cast<void**>(&c), h_c.size() * sizeof(T), handle);
    status |= ddla_memcpy(a, h_a.data(), h_a.size() * sizeof(T),
                          DdlaMemoryCopyKind::HostToDevice, handle);
    status |= ddla_memcpy(b, h_b.data(), h_b.size() * sizeof(T),
                          DdlaMemoryCopyKind::HostToDevice, handle);
    status |= ddla_memcpy(c, h_c.data(), h_c.size() * sizeof(T),
                          DdlaMemoryCopyKind::HostToDevice, handle);
    status |= ddla_synchronize(handle);
    if (status != 0) return 1;

    if (use_default) {
        gemm<>(handle, transa, transb, m, n, k,
               alpha, a, lda, b, ldb, beta, c, ldc);
    } else {
        gemm<Backend>(handle, transa, transb, m, n, k,
                      alpha, a, lda, b, ldb, beta, c, ldc);
    }
    status |= ddla_synchronize(handle);
    status |= ddla_memcpy(h_c.data(), c, h_c.size() * sizeof(T),
                          DdlaMemoryCopyKind::DeviceToHost, handle);
    status |= ddla_synchronize(handle);
    status |= ddla_free(a, handle);
    status |= ddla_free(b, handle);
    status |= ddla_free(c, handle);
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
    const bool single = std::is_same<T, float>::value ||
                        std::is_same<T, std::complex<float>>::value;
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
    if (Backend == default_backend_v)
        failures += check_case<Backend, T>(handle, 'N', 'N', true);
    return failures;
}

template <DdlaBackend Backend>
int run_backend()
{
    DdlaHandle_t handle = nullptr;
    ddla_init(handle, Backend);
    ddla_set(handle, MPI_COMM_WORLD);
    int failures = 0;
    failures += run_type<Backend, float>(handle);
    failures += run_type<Backend, double>(handle);
    failures += run_type<Backend, std::complex<float>>(handle);
    failures += run_type<Backend, std::complex<double>>(handle);
    ddla_destroy(handle);
    return failures;
}

#if DDLA_HAS_CPU && DDLA_HAS_GPU
int check_backend_mismatch()
{
    DdlaHandle_t cpu = nullptr;
    DdlaHandle_t gpu = nullptr;
    ddla_init(cpu, DdlaBackend::CPU);
    ddla_init(gpu, DdlaBackend::GPU);
    ddla_set(cpu, MPI_COMM_WORLD);
    ddla_set(gpu, MPI_COMM_WORLD);
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
    ddla_destroy(cpu);
    ddla_destroy(gpu);
    return cpu_rejected_gpu && gpu_rejected_cpu ? 0 : 1;
}
#endif

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
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
#if DDLA_HAS_CPU
    failures += run_backend<DdlaBackend::CPU>();
#endif
#if DDLA_HAS_GPU
    failures += run_backend<DdlaBackend::GPU>();
#endif
#if DDLA_HAS_CPU && DDLA_HAS_GPU
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
