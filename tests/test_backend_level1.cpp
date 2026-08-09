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
#include <ddla/axpy.h>
#include <ddla/iamax.h>
#include <ddla/geru.h>

using namespace ddla;

namespace {

template <typename T>
T value(int i, int salt)
{
    return T(((i * 17 + salt * 7) % 19) - 9);
}

template <>
std::complex<float> value(int i, int salt)
{
    return {static_cast<float>(((i * 17 + salt * 7) % 19) - 9),
            static_cast<float>(((i * 11 + salt * 5) % 17) - 8)};
}

template <>
std::complex<double> value(int i, int salt)
{
    return {static_cast<double>(((i * 17 + salt * 7) % 19) - 9),
            static_cast<double>(((i * 11 + salt * 5) % 17) - 8)};
}

template <typename T>
bool close(const T& a, const T& b)
{
    using Real = decltype(std::abs(std::declval<T>()));
    const Real tol = (std::is_same<T, float>::value ||
                      std::is_same<T, std::complex<float>>::value) ? 1e-4 : 1e-10;
    return std::abs(a - b) <= tol * (Real(1) + std::abs(b));
}

template <typename T>
T abs_magnitude(const T& x)
{
    return std::abs(x);
}

template <>
std::complex<float> abs_magnitude(const std::complex<float>& x)
{
    return std::abs(x.real()) + std::abs(x.imag());
}

template <>
std::complex<double> abs_magnitude(const std::complex<double>& x)
{
    return std::abs(x.real()) + std::abs(x.imag());
}

template <DdlaBackend Backend, typename T>
int check_axpy()
{
    DdlaHandle_t handle = nullptr;
    ddla_init(handle, Backend);
    ddla_set(handle);

    const int n = 17;
    const int incx = 1;
    const int incy = 2;
    const T alpha = value<T>(3, 1);
    std::vector<T> x(n * incx);
    std::vector<T> y(n * incy);
    std::vector<T> y_ref(n * incy);
    for (int i = 0; i < n; ++i) {
        x[i * incx] = value<T>(i, 2);
        y[i * incy] = value<T>(i, 3);
        y_ref[i * incy] = y[i * incy] + alpha * x[i * incx];
    }

    T* d_x = nullptr;
    T* d_y = nullptr;
    ddla_malloc(reinterpret_cast<void**>(&d_x), x.size() * sizeof(T), handle);
    ddla_malloc(reinterpret_cast<void**>(&d_y), y.size() * sizeof(T), handle);
    ddla_memcpy(d_x, x.data(), x.size() * sizeof(T),
                DdlaMemoryCopyKind::HostToDevice, handle);
    ddla_memcpy(d_y, y.data(), y.size() * sizeof(T),
                DdlaMemoryCopyKind::HostToDevice, handle);
    ddla::axpy<Backend>(handle, n, alpha, d_x, incx, d_y, incy);
    ddla_synchronize(handle);
    ddla_memcpy(y.data(), d_y, y.size() * sizeof(T),
                DdlaMemoryCopyKind::DeviceToHost, handle);
    ddla_synchronize(handle);
    ddla_free(d_x, handle);
    ddla_free(d_y, handle);
    ddla_destroy(handle);

    for (int i = 0; i < n; ++i) {
        if (!close(y[i * incy], y_ref[i * incy])) return 1;
    }
    return 0;
}

template <DdlaBackend Backend, typename T>
int check_iamax()
{
    DdlaHandle_t handle = nullptr;
    ddla_init(handle, Backend);
    ddla_set(handle);

    const int n = 23;
    const int incx = 1;
    std::vector<T> x(n * incx);
    for (int i = 0; i < n; ++i) {
        x[i * incx] = value<T>(i, 4);
    }
    x[13 * incx] = T(100);
    const int expected = 14; // 1-based, matching cuBLAS/hipBLAS iamax

    T* d_x = nullptr;
    ddla_malloc(reinterpret_cast<void**>(&d_x), x.size() * sizeof(T), handle);
    ddla_memcpy(d_x, x.data(), x.size() * sizeof(T),
                DdlaMemoryCopyKind::HostToDevice, handle);
    int result = -1;
    ddla::iamax<Backend>(handle, n, d_x, incx, result);
    ddla_synchronize(handle);
    ddla_free(d_x, handle);
    ddla_destroy(handle);

    return result == expected ? 0 : 1;
}

template <DdlaBackend Backend, typename T>
int check_geru()
{
    DdlaHandle_t handle = nullptr;
    ddla_init(handle, Backend);
    ddla_set(handle);

    const int m = 9;
    const int n = 7;
    const int incx = 2;
    const int incy = 1;
    const T alpha = value<T>(5, 5);
    std::vector<T> x(m * incx);
    std::vector<T> y(n * incy);
    std::vector<T> A(static_cast<size_t>(m) * n);
    std::vector<T> A_ref(static_cast<size_t>(m) * n);
    for (int i = 0; i < m; ++i) x[i * incx] = value<T>(i, 6);
    for (int j = 0; j < n; ++j) y[j * incy] = value<T>(j, 7);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            A[i + static_cast<size_t>(j) * m] = value<T>(i + j, 8);
            A_ref[i + static_cast<size_t>(j) * m] =
                A[i + static_cast<size_t>(j) * m] + alpha * x[i * incx] * y[j * incy];
        }
    }

    T* d_x = nullptr;
    T* d_y = nullptr;
    T* d_A = nullptr;
    ddla_malloc(reinterpret_cast<void**>(&d_x), x.size() * sizeof(T), handle);
    ddla_malloc(reinterpret_cast<void**>(&d_y), y.size() * sizeof(T), handle);
    ddla_malloc(reinterpret_cast<void**>(&d_A), A.size() * sizeof(T), handle);
    ddla_memcpy(d_x, x.data(), x.size() * sizeof(T),
                DdlaMemoryCopyKind::HostToDevice, handle);
    ddla_memcpy(d_y, y.data(), y.size() * sizeof(T),
                DdlaMemoryCopyKind::HostToDevice, handle);
    ddla_memcpy(d_A, A.data(), A.size() * sizeof(T),
                DdlaMemoryCopyKind::HostToDevice, handle);
    ddla::geru<Backend>(handle, m, n, alpha, d_x, incx, d_y, incy, d_A, m);
    ddla_synchronize(handle);
    ddla_memcpy(A.data(), d_A, A.size() * sizeof(T),
                DdlaMemoryCopyKind::DeviceToHost, handle);
    ddla_synchronize(handle);
    ddla_free(d_x, handle);
    ddla_free(d_y, handle);
    ddla_free(d_A, handle);
    ddla_destroy(handle);

    for (size_t k = 0; k < A.size(); ++k) {
        if (!close(A[k], A_ref[k])) return 1;
    }
    return 0;
}

template <DdlaBackend Backend>
int run_backend()
{
    int rc = 0;
    rc |= check_axpy<Backend, float>();
    rc |= check_axpy<Backend, double>();
    rc |= check_axpy<Backend, std::complex<float>>();
    rc |= check_axpy<Backend, std::complex<double>>();
    rc |= check_iamax<Backend, float>();
    rc |= check_iamax<Backend, double>();
    rc |= check_iamax<Backend, std::complex<float>>();
    rc |= check_iamax<Backend, std::complex<double>>();
    rc |= check_geru<Backend, float>();
    rc |= check_geru<Backend, double>();
    rc |= check_geru<Backend, std::complex<float>>();
    rc |= check_geru<Backend, std::complex<double>>();
    return rc;
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::cout << "=== Backend Level-1 (axpy/iamax/geru) Test ===" << std::endl;
    }

    int failures = 0;
#if DDLA_HAS_CPU
    if (rank == 0) std::cout << "  running CPU backend" << std::endl;
    failures += run_backend<DdlaBackend::CPU>();
#endif
#if DDLA_HAS_GPU
    if (rank == 0) std::cout << "  running GPU backend" << std::endl;
    failures += run_backend<DdlaBackend::GPU>();
#endif

    int global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << (global_failures == 0 ? "ALL LEVEL-1 TESTS PASSED" :
                                             "LEVEL-1 TEST FAILURES")
                  << std::endl;
    }
    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}
