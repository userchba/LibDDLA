#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>

#include "mpi_test_init.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

using namespace ddla;

template <typename T>
T sample(int i, bool right)
{
    using R = typename real_type<T>::type;
    const R re = R((i % 13) - 6) / R(8);
    if constexpr (std::is_arithmetic_v<T>) return right ? -re : re;
    else return T(right ? -re : re, R((i % 7) - 3) / R(right ? 16 : 4));
}

template <DdlaBackend Backend, typename T>
int check_type(DdlaHandle_t h)
{
    using R = typename real_type<T>::type;
    using W = std::complex<long double>;
    int bad = 0;
    for (int n : {0, 1, 7, 129}) for (int ix : {1, 2, -1, -2}) for (int iy : {1, 3, -1, -3}) {
        const int ax = std::abs(ix), ay = std::abs(iy);
        const int px = std::max(1, n)*ax, py = std::max(1, n)*ay;
        // Prefix guards make the old negative-stride bug observable without
        // allowing its backwards accesses to leave the allocation.
        std::vector<T> x(2*px + 3, sample<T>(37, false));
        std::vector<T> y(2*py + 3, sample<T>(43, true));
        W expected(0);
        long double scale = 0;
        // Fill physical storage from logical order independently of the
        // implementation's BLAS negative-increment starting-index rule.
        for (int i = 0; i < n; ++i) {
            const T a = sample<T>(i, false), b = sample<T>(i + 3, true);
            x[px + (ix > 0 ? i : n - 1 - i)*ax] = a;
            y[py + (iy > 0 ? i : n - 1 - i)*ay] = b;
            const W term = std::conj(W(a)) * W(b);
            expected += term;
            scale += std::abs(term);
        }
        vector<T> dx(h, x.size()), dy(h, y.size()), result(h, 1);
        DDLA_CHECK(ddlaMemcpy(h, dx.data(), x.data(), x.size()*sizeof(T)));
        DDLA_CHECK(ddlaMemcpy(h, dy.data(), y.data(), y.size()*sizeof(T)));
        dotc<Backend, T>(h, n, dx.data() + px, ix, dy.data() + py, iy, result.data());
        T got;
        DDLA_CHECK(ddlaMemcpy(h, &got, result.data(), sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
        const auto error = std::abs(W(got) - expected);
        if (!std::isfinite(error) || error > 64*std::numeric_limits<R>::epsilon()*std::max(1.0L, scale)) {
            std::cerr << "dotc mismatch n=" << n << " incx=" << ix << " incy=" << iy << '\n';
            ++bad;
        }
    }
    return bad;
}

template <DdlaBackend Backend>
int check_backend()
{
    DdlaHandle_t h = nullptr;
    DDLA_CHECK(ddlaInit(h, Backend));
    DDLA_CHECK(ddlaSet(h, MPI_COMM_WORLD));
    int bad = check_type<Backend, float>(h) + check_type<Backend, double>(h)
            + check_type<Backend, std::complex<float>>(h) + check_type<Backend, std::complex<double>>(h);
    DDLA_CHECK(ddlaDestroy(h));
    std::cout << (Backend == DdlaBackend::CPU ? "CPU" : "GPU") << " dotc: 256 stride/empty cases, failures=" << bad << '\n';
    return bad;
}

int main(int argc, char** argv)
{
    ddla_test_mpi_init(&argc, &argv);
    int bad = 0;
    try {
        const char* only = std::getenv("DDLA_TEST_ONLY");
        if (!only || std::string(only) != "gpu") bad += check_backend<DdlaBackend::CPU>();
#if DDLA_HAS_GPU
        if (!only || std::string(only) != "cpu") bad += check_backend<DdlaBackend::GPU>();
#endif
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; ++bad; }
    int total = 0;
    MPI_Allreduce(&bad, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    std::cout << "DOTC_CHECKS_FAILED=" << total << std::endl;
    MPI_Finalize();
    return total != 0;
}
