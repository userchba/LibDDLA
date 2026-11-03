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
    const Real tol = (std::is_same_v<T, float> ||
                      std::is_same_v<T, std::complex<float>>) ? 1e-4 : 1e-10;
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
    ddlaInit(handle, Backend);
    ddlaSet(handle);

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
    ddlaMalloc(handle, reinterpret_cast<void**>(&d_x), x.size() * sizeof(T));
    ddlaMalloc(handle, reinterpret_cast<void**>(&d_y), y.size() * sizeof(T));
    ddlaMemcpy(handle, d_x, x.data(), x.size() * sizeof(T));
    ddlaMemcpy(handle, d_y, y.data(), y.size() * sizeof(T));
    ddla::axpy<Backend>(handle, n, alpha, d_x, incx, d_y, incy);
    ddlaSynchronize(handle);
    ddlaMemcpy(handle, y.data(), d_y, y.size() * sizeof(T));
    ddlaSynchronize(handle);
    ddlaFree(handle, d_x);
    ddlaFree(handle, d_y);
    ddlaDestroy(handle);

    for (int i = 0; i < n; ++i) {
        if (!close(y[i * incy], y_ref[i * incy])) return 1;
    }
    return 0;
}

template <DdlaBackend Backend, typename T>
int check_iamax()
{
    DdlaHandle_t handle = nullptr;
    ddlaInit(handle, Backend);
    ddlaSet(handle);

    const int n = 23;
    const int incx = 1;
    std::vector<T> x(n * incx);
    for (int i = 0; i < n; ++i) {
        x[i * incx] = value<T>(i, 4);
    }
    x[13 * incx] = T(100);
    const int expected = 14; // 1-based, matching cuBLAS/hipBLAS iamax

    T* d_x = nullptr;
    ddlaMalloc(handle, reinterpret_cast<void**>(&d_x), x.size() * sizeof(T));
    ddlaMemcpy(handle, d_x, x.data(), x.size() * sizeof(T));
    int result = -1;
    ddla::iamax<Backend>(handle, n, d_x, incx, result);
    ddlaSynchronize(handle);
    ddlaFree(handle, d_x);
    ddlaDestroy(handle);

    return result == expected ? 0 : 1;
}

template <DdlaBackend Backend, typename T>
int check_geru()
{
    DdlaHandle_t handle = nullptr;
    ddlaInit(handle, Backend);
    ddlaSet(handle);

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
    ddlaMalloc(handle, reinterpret_cast<void**>(&d_x), x.size() * sizeof(T));
    ddlaMalloc(handle, reinterpret_cast<void**>(&d_y), y.size() * sizeof(T));
    ddlaMalloc(handle, reinterpret_cast<void**>(&d_A), A.size() * sizeof(T));
    ddlaMemcpy(handle, d_x, x.data(), x.size() * sizeof(T));
    ddlaMemcpy(handle, d_y, y.data(), y.size() * sizeof(T));
    ddlaMemcpy(handle, d_A, A.data(), A.size() * sizeof(T));
    ddla::geru<Backend>(handle, m, n, alpha, d_x, incx, d_y, incy, d_A, m);
    ddlaSynchronize(handle);
    ddlaMemcpy(handle, A.data(), d_A, A.size() * sizeof(T));
    ddlaSynchronize(handle);
    ddlaFree(handle, d_x);
    ddlaFree(handle, d_y);
    ddlaFree(handle, d_A);
    ddlaDestroy(handle);

    for (size_t k = 0; k < A.size(); ++k) {
        if (!close(A[k], A_ref[k])) return 1;
    }
    return 0;
}

// x := alpha * x with a stride: the strided elements are scaled, everything in between is untouched.
template <DdlaBackend Backend, typename T>
int check_scal()
{
    DdlaHandle_t handle = nullptr;
    ddlaInit(handle, Backend);
    ddlaSet(handle);

    const int n = 17;
    const int incx = 3;
    const T alpha = value<T>(3, 1);
    std::vector<T> x(static_cast<size_t>(n) * incx);
    for (size_t k = 0; k < x.size(); ++k) x[k] = value<T>(static_cast<int>(k), 2);
    std::vector<T> x_ref = x;
    for (int i = 0; i < n; ++i) x_ref[static_cast<size_t>(i) * incx] = alpha * x[static_cast<size_t>(i) * incx];

    T* d_x = nullptr;
    ddlaMalloc(handle, reinterpret_cast<void**>(&d_x), x.size() * sizeof(T));
    ddlaMemcpy(handle, d_x, x.data(), x.size() * sizeof(T));
    ddla::scal<Backend>(handle, n, alpha, d_x, incx);
    ddlaSynchronize(handle);
    ddlaMemcpy(handle, x.data(), d_x, x.size() * sizeof(T));
    ddlaSynchronize(handle);
    ddlaFree(handle, d_x);
    ddlaDestroy(handle);

    for (size_t k = 0; k < x.size(); ++k) {
        if (!close(x[k], x_ref[k])) return 1;
    }
    return 0;
}

// Distinct, exactly representable value per element, so a swap that moves the wrong element cannot go unnoticed.
template <typename T>
T unique_value(int k, int salt)
{
    return T(static_cast<float>(k + 1 + 1000 * salt));
}

// x <-> y with the given strides.  Logical element i of a vector with a negative stride sits at (n - 1 - i) * |inc|
// (the BLAS convention), so (incx, incy) = (2, -3) pairs x[0] with y[(n - 1) * 3].
template <DdlaBackend Backend, typename T>
int check_swap(int incx, int incy)
{
    DdlaHandle_t handle = nullptr;
    ddlaInit(handle, Backend);
    ddlaSet(handle);

    const int n = 13;
    const int ax = std::abs(incx);
    const int ay = std::abs(incy);
    std::vector<T> x(static_cast<size_t>(n) * ax);
    std::vector<T> y(static_cast<size_t>(n) * ay);
    for (size_t k = 0; k < x.size(); ++k) x[k] = unique_value<T>(static_cast<int>(k), 1);
    for (size_t k = 0; k < y.size(); ++k) y[k] = unique_value<T>(static_cast<int>(k), 2);
    std::vector<T> x_ref = x;
    std::vector<T> y_ref = y;
    for (int i = 0; i < n; ++i) {
        const size_t ix = static_cast<size_t>(incx > 0 ? i : n - 1 - i) * ax;
        const size_t iy = static_cast<size_t>(incy > 0 ? i : n - 1 - i) * ay;
        std::swap(x_ref[ix], y_ref[iy]);
    }

    T* d_x = nullptr;
    T* d_y = nullptr;
    ddlaMalloc(handle, reinterpret_cast<void**>(&d_x), x.size() * sizeof(T));
    ddlaMalloc(handle, reinterpret_cast<void**>(&d_y), y.size() * sizeof(T));
    ddlaMemcpy(handle, d_x, x.data(), x.size() * sizeof(T));
    ddlaMemcpy(handle, d_y, y.data(), y.size() * sizeof(T));
    ddla::swap<Backend>(handle, n, d_x, incx, d_y, incy);
    ddlaSynchronize(handle);
    ddlaMemcpy(handle, x.data(), d_x, x.size() * sizeof(T));
    ddlaMemcpy(handle, y.data(), d_y, y.size() * sizeof(T));
    ddlaSynchronize(handle);
    ddlaFree(handle, d_x);
    ddlaFree(handle, d_y);
    ddlaDestroy(handle);

    for (size_t k = 0; k < x.size(); ++k) {
        if (!(x[k] == x_ref[k])) return 1;
    }
    for (size_t k = 0; k < y.size(); ++k) {
        if (!(y[k] == y_ref[k])) return 1;
    }
    return 0;
}

template <DdlaBackend Backend, typename T>
int check_swap_strides()
{
    int rc = 0;
    rc |= check_swap<Backend, T>(1, 1);
    rc |= check_swap<Backend, T>(2, 3);
    rc |= check_swap<Backend, T>(2, -3);
    rc |= check_swap<Backend, T>(-1, -2);
    return rc;
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
    rc |= check_scal<Backend, float>();
    rc |= check_scal<Backend, double>();
    rc |= check_scal<Backend, std::complex<float>>();
    rc |= check_scal<Backend, std::complex<double>>();
    rc |= check_swap_strides<Backend, float>();
    rc |= check_swap_strides<Backend, double>();
    rc |= check_swap_strides<Backend, std::complex<float>>();
    rc |= check_swap_strides<Backend, std::complex<double>>();
    return rc;
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::cout << "=== Backend Level-1 (axpy/iamax/geru/swap) Test ===" << std::endl;
    }

    int failures = 0;
    if (rank == 0) std::cout << "  running CPU backend" << std::endl;
    failures += run_backend<DdlaBackend::CPU>();
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
