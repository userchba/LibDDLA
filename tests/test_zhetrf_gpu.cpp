#include "api_grid_test_common.h"
#include <ddla/ddla_vector.h>
#include <dlfcn.h>
#include <chrono>
#include <complex>
#include <cstring>
#include <iomanip>
#include <limits>

using namespace ddla;
using Z = std::complex<double>;
extern "C" void zhetrs_(const char*, const int*, const int*, const Z*, const int*, const int*, Z*,
                        const int*, int*);

Z entry(int i, int j, int n, int pattern, double scale)
{
    if (pattern == 2)
        return Z(0);
    if (pattern == 3)
        return i == j && i != n / 2 ? Z((i & 1) ? -2 : 2) * scale : Z(0);
    if (pattern == 1)
    {
        if (i == j)
            return i == n / 2 && (n & 1) ? Z(3) * scale : Z(0);
        if (j != n - 1 - i)
            return Z(0);
        return (i < j ? Z(1 + .01 * i, .3) : std::conj(Z(1 + .01 * j, .3))) * scale;
    }
    if (i == j)
        return Z((i & 1 ? -1 : 1) * (2 + .1 * (i % 7))) * scale;
    const int lo = std::min(i, j), hi = std::max(i, j);
    const Z value(.07 * (((lo + 3) * 17 + (hi + 1) * 13) % 17 - 8),
                  .05 * (((lo + 1) * 7 + (hi + 2) * 11) % 13 - 6));
    return (i < j ? value : std::conj(value)) * scale;
}

void test(DdlaHandle_t h, int n, char uplo, int pattern, double scale, bool imaginary_nan,
          bool expect_native, int repetitions = 0)
{
    const int ld = std::max(1, n) + 7;
    const Z poison(913, -417), nan(std::numeric_limits<double>::quiet_NaN(), -919);
    std::vector<Z> input(std::size_t(ld) * (n + 2), poison);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
        {
            input[i + std::size_t(j) * ld] =
                (uplo == 'L' ? i >= j : i <= j) ? entry(i, j, n, pattern, scale) : nan;
            if (i == j && imaginary_nan)
                input[i + std::size_t(j) * ld].imag(nan.real());
        }
    ddla::vector<Z> device(h, input.size());
    std::vector<int> piv(std::max(1, n));
    using Reset = void (*)();
    using Count = long long (*)();
    auto reset = reinterpret_cast<Reset>(dlsym(RTLD_DEFAULT, "ddla_zhetrf_trace_reset"));
    auto count = reinterpret_cast<Count>(dlsym(RTLD_DEFAULT, "ddla_zhetrf_trace_count"));
    if (!reset || !count)
        throw std::runtime_error("Required CPU-LAPACK execution hook missing");
    using Bytes = unsigned long long (*)();
    auto host_bytes =
        reinterpret_cast<Bytes>(dlsym(RTLD_DEFAULT, "ddla_zhetrf_trace_host_largest"));
    auto device_bytes =
        reinterpret_cast<Bytes>(dlsym(RTLD_DEFAULT, "ddla_zhetrf_trace_device_largest"));
    auto device_total =
        reinterpret_cast<Bytes>(dlsym(RTLD_DEFAULT, "ddla_zhetrf_trace_device_total"));
    std::vector<double> times;
    unsigned long long largest_host = 0, largest_device = 0, total_device = 0;
    int info = 0;
    for (int run = 0; run < std::max(1, repetitions + 1); ++run)
    {
        DDLA_CHECK(ddlaMemcpy(h, device.data(), input.data(), input.size() * sizeof(Z)));
        DDLA_CHECK(ddlaSynchronize(h));
        reset();
        const auto start = std::chrono::steady_clock::now();
        hetrf<DdlaBackend::GPU, Z>(h, uplo, n, device.data(), ld, piv.data(), info);
        DDLA_CHECK(ddlaSynchronize(h));
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const auto calls = count();
        if (host_bytes && device_bytes && device_total)
        {
            largest_host = host_bytes();
            largest_device = device_bytes();
            total_device = device_total();
        }
        if (n && ((expect_native && calls != 0) || (!expect_native && calls == 0)))
            throw std::runtime_error("CPU-LAPACK execution proof failed");
        if (run > 0)
            times.push_back(elapsed);
    }
    std::vector<Z> factors(input.size());
    DDLA_CHECK(ddlaMemcpy(h, factors.data(), device.data(), factors.size() * sizeof(Z)));
    DDLA_CHECK(ddlaSynchronize(h));
    for (int j = 0; j < n + 2; ++j)
        for (int i = 0; i < ld; ++i)
            if (!(i < n && j < n && (uplo == 'L' ? i >= j : i <= j)) &&
                std::memcmp(&input[i + std::size_t(j) * ld], &factors[i + std::size_t(j) * ld],
                            sizeof(Z)))
                throw std::runtime_error("Unselected triangle or padding was modified");
    if (pattern == 2 || pattern == 3)
    {
        const int expected = !n ? 0 : pattern == 2 ? (uplo == 'U' ? n : 1) : n / 2 + 1;
        if (info != expected)
            throw std::runtime_error("Incorrect singular INFO");
    }
    else
    {
        if (info != 0)
            throw std::runtime_error("Unexpected factorization INFO");
        for (int step = 0; step < n; ++step)
        {
            const int i = uplo == 'L' ? step : n - step - 1;
            const long long v = piv[i];
            if (!v || std::abs(v) > n)
                throw std::runtime_error("Invalid pivot index");
            if (v < 0)
            {
                const int other = uplo == 'L' ? i + 1 : i - 1;
                if (other < 0 || other >= n || piv[other] != v)
                    throw std::runtime_error("Invalid pivot pair");
                ++step;
            }
        }
        const int nrhs = 2, ldb = std::max(n, 1);
        std::vector<Z> rhs(std::size_t(ldb) * nrhs, Z(0));
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < n; ++i)
                for (int k = 0; k < n; ++k)
                    rhs[i + std::size_t(j) * ldb] +=
                        entry(i, k, n, pattern, scale) * Z(1 + .01 * (k % 13), .07 * (j - k % 5));
        const auto original_rhs = rhs;
        if (n)
            zhetrs_(&uplo, &n, &nrhs, factors.data(), &ld, piv.data(), rhs.data(), &ldb, &info);
        double error = 0, backward = 0, denom = 0;
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < n; ++i)
            {
                error = std::max(error, std::abs(rhs[i + std::size_t(j) * ldb] -
                                                 Z(1 + .01 * (i % 13), .07 * (j - i % 5))));
                Z residual = original_rhs[i + std::size_t(j) * ldb];
                for (int k = 0; k < n; ++k)
                    residual -= entry(i, k, n, pattern, scale) * rhs[k + std::size_t(j) * ldb];
                backward = std::max(backward, std::abs(residual));
                denom = std::max(denom, std::abs(original_rhs[i + std::size_t(j) * ldb]));
            }
        if (info || !std::isfinite(error) || error > 5e-10 * std::max(n, 1) ||
            !std::isfinite(backward) || backward / std::max(denom, 1e-300) > 1e-12 * std::max(n, 1))
        {
            std::cerr << "BAD n=" << n << " u=" << uplo << " pattern=" << pattern
                      << " error=" << error << " backward=" << backward / denom << std::endl;
            throw std::runtime_error("Independent LAPACK solve/residual check failed");
        }
    }
    if (!ddlaGetRank(h))
    {
        std::cout << "ZHETRF_PASS n=" << n << " uplo=" << uplo << " pattern=" << pattern
                  << " scale=" << scale << " native=" << expect_native << " info=" << info;
        if (!times.empty())
        {
            std::sort(times.begin(), times.end());
            std::cout << " median_s=" << std::setprecision(9) << times[times.size() / 2]
                      << " largest_host_new_bytes=" << largest_host
                      << " largest_device_request_bytes=" << largest_device
                      << " total_device_requests_bytes=" << total_device;
        }
        std::cout << std::endl;
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int ranks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try
    {
        DdlaHandle_t h = nullptr;
        DDLA_CHECK(ddlaInit(h, DdlaBackend::GPU));
        DDLA_CHECK(ddlaSet(h, MPI_COMM_WORLD, 1, ranks));
        const char* expected = std::getenv("DDLA_EXPECT_NATIVE");
        const bool native = !expected || std::strcmp(expected, "0") != 0;
        if (argc > 1)
        {
            const int n = std::atoi(argv[1]);
            for (char uplo : {'L', 'U'})
                test(h, n, uplo, argc > 2 ? std::atoi(argv[2]) : 0, 1, false, native,
                     argc > 3 ? std::atoi(argv[3]) : 3);
        }
        else
        {
            for (int n : {0, 1, 2, 3, 17, 63, 127, 128, 129, 255, 257, 601})
                for (char uplo : {'L', 'U'})
                    for (int pattern : {0, 1})
                        test(h, n, uplo, pattern, 1, n % 2, native);
            for (int n : {1, 17, 129})
                for (char uplo : {'L', 'U'})
                    for (int pattern : {2, 3})
                        test(h, n, uplo, pattern, 1, true, native);
            for (char uplo : {'L', 'U'})
                for (double scale : {1e-100, 1e100, 1e-300, 1e300})
                    test(h, 129, uplo, 0, scale, true, native);
            // Reuse another DDLA handle's stream, then restore both owners.
            DdlaHandle_t alternate = nullptr;
            DDLA_CHECK(ddlaInit(alternate, DdlaBackend::GPU));
            DDLA_CHECK(ddlaSet(alternate, MPI_COMM_WORLD, 1, ranks));
            void* previous = ddlaGetStream(h);
            DDLA_CHECK(ddlaSetStream(h, ddlaGetStream(alternate)));
            test(h, 129, 'L', 0, 1, true, native);
            test(h, 1100, 'U', 1, 1, false, native);
            for (char uplo : {'L', 'U'})
            {
                test(h, 2305, uplo, 0, 1, true, native);
                test(h, 2305, uplo, 1, 1, true, native);
                test(h, 2305, uplo, 3, 1, true, native);
            }
            DDLA_CHECK(ddlaSetStream(h, previous));
            DDLA_CHECK(ddlaDestroy(alternate));
        }
        DDLA_CHECK(ddlaDestroy(h));
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return 0;
}
