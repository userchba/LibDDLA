#include "LDLT/ldlt_impl.h"
#include "api_grid_test_common.h"
#include <chrono>
#include <cstring>
#include <ddla/ddla_vector.h>
#include <dlfcn.h>
#include <iomanip>
#include <limits>
using namespace ddla;
using namespace api_grid_test;
namespace
{
struct Options
{
    int pr = 1, pc = 1, n = -1, repeats = 0, pattern = -1, operation = -1;
    std::string scalar = "z";
};
template <class T> T number(double x, double y = 0)
{
    if constexpr (std::is_arithmetic_v<T>)
        return T(x);
    else
        return T(x, y);
}
double random_value(int i, int j, unsigned salt)
{
    std::uint64_t x = std::uint64_t(i + 1) * 0x9e3779b97f4a7c15ULL +
                      std::uint64_t(j + 3) * 0xbf58476d1ce4e5b9ULL + salt;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return double((x ^ (x >> 31)) % 2001) / 1000 - 1;
}
template <class T> T entry(int i, int j, int n, int pattern)
{
    if (pattern == 4)
        return T(0);
    if (pattern == 6)
    {
        const int middle = n / 2;
        if (i == j)
            return i == middle || i == middle + 1 ? T(0) : T(i & 1 ? -2 : 2);
        return (i == middle && j == middle + 1) || (j == middle && i == middle + 1)
                   ? number<T>(1, i < j ? .2 : -.2)
                   : T(0);
    }
    if (pattern == 5)
        return i == j && i != n / 2 ? T(i & 1 ? -2 : 2) : T(0);
    if (pattern == 0)
    {
        if (i == j)
            return T((i & 1 ? -1 : 1) * (4 + .03 * (i % 7)));
    }
    else
    {
        const int other = n - 1 - i;
        if (i == j)
        {
            if (i == other)
                return T(2);
            if (pattern == 1)
                return T(0);
            return T((pattern == 2 ? i < other : i > other) ? .01 : 4);
        }
        if (j == other)
            return number<T>(1 + .01 * std::min(i, j), i < j ? .2 : -.2);
    }
    const int lo = std::min(i, j), hi = std::max(i, j);
    return number<T>(.02 * random_value(lo, hi, 17) / std::max(1, n),
                     .02 * random_value(lo, hi, 71) / std::max(1, n) * (i < j ? 1 : -1));
}
struct Trace
{
    using Reset = void (*)();
    using Count = long long (*)();
    using Bytes = unsigned long long (*)();
    Reset reset;
    Count count;
    Bytes host, largest, total, peak, live;
    Trace()
        : reset(reinterpret_cast<Reset>(dlsym(RTLD_DEFAULT, "ddla_zhetrf_trace_reset"))),
          count(reinterpret_cast<Count>(dlsym(RTLD_DEFAULT, "ddla_zhetrf_trace_count"))),
          host(reinterpret_cast<Bytes>(dlsym(RTLD_DEFAULT, "ddla_zhetrf_trace_host_largest"))),
          largest(reinterpret_cast<Bytes>(dlsym(RTLD_DEFAULT, "ddla_zhetrf_trace_device_largest"))),
          total(reinterpret_cast<Bytes>(dlsym(RTLD_DEFAULT, "ddla_zhetrf_trace_device_total")))
    {
        peak = reinterpret_cast<Bytes>(dlsym(RTLD_DEFAULT, "ddla_zhetrf_trace_device_peak"));
        live = reinterpret_cast<Bytes>(dlsym(RTLD_DEFAULT, "ddla_zhetrf_trace_device_live"));
        if (!reset || !count)
            throw std::runtime_error("CPU execution counter not preloaded");
    }
};
template <class T>
void test(DdlaHandle_t h, DdlaHandle_t cpu, const Trace& trace, int operation, int n, char u,
          int pattern, int mb, int nb, bool native, int repeats, double scale = 1)
{
    const bool distributed = operation != 0, pivot = operation == 1;
    int d[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(d, h, n + 3, n + 4, mb, nb, h->nprows_ - 1, h->npcols_ - 1));
    d[DDLA_LLD_] += 3;
    const int ld = distributed ? d[DDLA_LLD_] : std::max(1, n) + 5;
    const int cols = distributed ? ddla_test::n_loc(h, d) : n + 2;
    const int local_rows = distributed ? ddla_test::m_loc(h, d) : n;
    const T poison = number<T>(131, -719), nan = number<T>(NAN, NAN);
    std::vector<T> input(std::size_t(ld) * cols, poison);
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < local_rows; ++i)
        {
            const int gi = distributed ? indx_l2g_r(d, h, i) : i,
                      gj = distributed ? indx_l2g_c(d, h, j) : j;
            if (gi < n && gj < n)
            {
                T value = (u == 'L' ? gi >= gj : gi <= gj) ? entry<T>(gi, gj, n, pattern) * T(scale)
                                                           : nan;
                if constexpr (!std::is_arithmetic_v<T>)
                    if (gi == gj)
                        value.imag(NAN);
                input[i + std::size_t(j) * ld] = value;
            }
        }
    vector<T> a(h, input.size());
    std::vector<int> piv(std::max(1, n));
    std::vector<double> times;
    unsigned long long biggest = 0, total = 0, host = 0, peak = 0;
    int info = 0;
    for (int run = 0; run < std::max(1, repeats + 1); ++run)
    {
        DDLA_CHECK(ddlaMemcpy(h, a.data(), input.data(), input.size() * sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
        trace.reset();
        const auto start = std::chrono::steady_clock::now();
        if (operation == 0)
            hetrf_nopiv<DdlaBackend::GPU, T>(h, u, n, a.data(), ld, info);
        else if (pivot)
            phetrf<DdlaBackend::GPU, T>(h, u, n, a.data(), d, piv.data(), info);
        else
            phetrf_nopiv<DdlaBackend::GPU, T>(h, u, n, a.data(), d, info);
        DDLA_CHECK(ddlaSynchronize(h));
        double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const auto calls = trace.count();
        if (trace.peak)
            peak = std::max(peak, trace.peak());
        if (trace.live && trace.live())
            throw std::runtime_error("Factorization leaked DDLA device workspace");
        if (native && calls)
            throw std::runtime_error("Native factorization called CPU factor/BLAS");
        if (!native && n > 1 && (pivot || pattern == 0) && (!distributed || h->myid == 0) &&
            calls == 0)
            throw std::runtime_error("Baseline negative-control counter did not fire");
        if (trace.host && trace.largest && trace.total)
        {
            biggest = std::max(biggest, trace.largest());
            total = std::max(total, trace.total());
            host = std::max(host, trace.host());
        }
        double maximum_time = 0;
        MPI_CHECK(MPI_Allreduce(&seconds, &maximum_time, 1, MPI_DOUBLE, MPI_MAX, h->comm));
        if (run > 0)
            times.push_back(maximum_time);
    }
    if (native && n >= 512)
    {
        const auto global = std::size_t(n) * n * sizeof(T);
        if (host >= global || (distributed && h->nprocs > 1 && biggest >= global))
            throw std::runtime_error("Native factor retained a full global matrix allocation");
    }
    const int expected = (pattern == 6 && !pivot            ? (u == 'L' ? n / 2 + 1 : n / 2 + 2)
                          : pattern == 1 && !pivot && n > 1 ? (u == 'L' ? 1 : n)
                          : pattern == 4                    ? (u == 'L' ? 1 : n)
                          : pattern == 5                    ? n / 2 + 1
                                                            : 0);
    if (info != (n ? expected : 0))
    {
        std::cerr << "INFO op=" << operation << " n=" << n << " u=" << u << " pattern=" << pattern
                  << " actual=" << info << " expected=" << expected << std::endl;
        throw std::runtime_error("Incorrect INFO");
    }
    std::vector<T> factors(input.size());
    DDLA_CHECK(ddlaMemcpy(h, factors.data(), a.data(), factors.size() * sizeof(T)));
    DDLA_CHECK(ddlaSynchronize(h));
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < ld; ++i)
        {
            const int gi = distributed ? indx_l2g_r(d, h, i) : i,
                      gj = distributed ? indx_l2g_c(d, h, j) : j;
            const bool active =
                i < local_rows && gi < n && gj < n && (u == 'L' ? gi >= gj : gi <= gj);
            const auto p = i + std::size_t(j) * ld;
            if (!active && std::memcmp(&input[p], &factors[p], sizeof(T)))
                throw std::runtime_error("Factor modified unused triangle/parent/padding");
        }
    double error = 0, residual = 0;
    if (n && !info)
    {
        vector<T> gathered(h, distributed && h->myid == 0 ? std::size_t(n) * n : 0);
        if (distributed)
            gather<DdlaBackend::GPU, T>(h, n, n, a.data(), 1, 1, d, 0, gathered.data(), n);
        if (!distributed || h->myid == 0)
        {
            std::vector<T> dense(std::size_t(n) * n);
            if (distributed)
            {
                DDLA_CHECK(ddlaMemcpy(h, dense.data(), gathered.data(), dense.size() * sizeof(T)));
                DDLA_CHECK(ddlaSynchronize(h));
            }
            else
                for (int j = 0; j < n; ++j)
                    std::copy_n(factors.data() + std::size_t(j) * ld, n,
                                dense.data() + std::size_t(j) * n);
            std::vector<T> rhs(n), original(n);
            for (int i = 0; i < n; ++i)
                for (int k = 0; k < n; ++k)
                    rhs[i] += (entry<T>(i, k, n, pattern) * T(scale)) *
                              number<T>(1 + .01 * (k % 7), .05 * (k % 3));
            original = rhs;
            int solve_info = 0;
            if (pivot)
                ldlt_impl::solve(std::true_type{}, u, n, 1, dense.data(), n, piv.data(), rhs.data(),
                                 n, solve_info);
            else
                hetrs_nopiv<DdlaBackend::CPU, T>(cpu, u, n, 1, dense.data(), n, rhs.data(), n,
                                                 solve_info);
            double denominator = 0;
            for (int i = 0; i < n; ++i)
            {
                error = std::max(
                    error, double(std::abs(rhs[i] - number<T>(1 + .01 * (i % 7), .05 * (i % 3)))));
                T r = original[i];
                for (int k = 0; k < n; ++k)
                    r -= (entry<T>(i, k, n, pattern) * T(scale)) * rhs[k];
                residual = std::max(residual, double(std::abs(r)));
                denominator = std::max(denominator, double(std::abs(original[i])));
            }
            residual /= std::max(denominator, 1e-300);
            const double epsilon = std::numeric_limits<typename real_type<T>::type>::epsilon();
            if (solve_info || !std::isfinite(error) || !std::isfinite(residual) ||
                error > 500 * epsilon * std::max(n, 1) || residual > 50 * epsilon * std::max(n, 1))
            {
                std::cerr << "BAD op=" << operation << " n=" << n << " u=" << u
                          << " pattern=" << pattern << " error=" << error
                          << " residual=" << residual << std::endl;
                throw std::runtime_error("Independent solve/residual failed");
            }
        }
    }
    if (repeats)
    {
        std::sort(times.begin(), times.end());
        for (int rank = 0; rank < h->nprocs; ++rank)
        {
            MPI_CHECK(MPI_Barrier(h->comm));
            if (rank == h->myid)
            {
                const int mr = num_loc(n, mb, h->myprow_, d[DDLA_RSRC_], h->nprows_);
                const int nc = num_loc(n, nb, h->mypcol_, d[DDLA_CSRC_], h->npcols_);
                std::size_t matrix_workspace = 0;
                if (distributed)
                {
                    if (!native)
                        matrix_workspace = h->myid == 0 ? std::size_t(n) * n * sizeof(T) : 0;
                    else if (pivot && n > 1 &&
                             !((h->nprocs == 1 || (n <= mb && n <= nb)) &&
                               std::is_same_v<T, std::complex<double>>))
                        matrix_workspace = std::size_t(std::max(mr, 1)) * nc * sizeof(T);
                }
                std::cout << std::setprecision(9) << "LDLT_BENCH op=" << operation << " n=" << n
                          << " uplo=" << u << " grid=" << h->nprows_ << "x" << h->npcols_
                          << " rank=" << rank << " native=" << native
                          << " median_s=" << times[times.size() / 2] << " host_largest=" << host
                          << " device_largest=" << biggest << " device_total=" << total
                          << " device_peak=" << peak << " expected_F_bytes=" << matrix_workspace
                          << " local_rows=" << mr << " local_cols=" << nc << " input_bytes="
                          << (std::size_t(d[DDLA_LLD_]) * ddla_test::n_loc(h, d) * sizeof(T))
                          << " error=" << error << " residual=" << residual << std::endl;
            }
        }
    }
    if (!h->myid && !repeats)
        std::cout << "LDLT_NATIVE_PASS op=" << operation << " bytes=" << sizeof(T)
                  << " real_bytes=" << sizeof(typename real_type<T>::type)
                  << " complex=" << !std::is_arithmetic_v<T> << " n=" << n << " uplo=" << u
                  << " pattern=" << pattern << " error=" << error << " residual=" << residual
                  << std::endl;
}
template <class T> void cases(DdlaHandle_t h, DdlaHandle_t cpu, const Trace& trace, bool native)
{
    for (int n : {0, 1, 2, 17, 65, 129})
        for (char u : {'L', 'U'})
        {
            for (int op : {0, 2})
                test<T>(h, cpu, trace, op, n, u, 0, 3, 4, native, 0);
            for (int pattern : {0, 1, 2, 3})
                test<T>(h, cpu, trace, 1, n, u, pattern, 3, 4, native, 0);
        }
    for (int n : {255, 256, 257})
        for (char u : {'L', 'U'})
            for (int pattern : {0, 5, 6})
                test<T>(h, cpu, trace, 0, n, u, pattern, 3, 4, native, 0);
    for (int n : {17, 65, 129})
        for (char u : {'L', 'U'})
            for (int op : {0, 1, 2})
                for (int pattern : {4, 5, 6})
                    test<T>(h, cpu, trace, op, n, u, pattern, 3, 4, native, 0);
    for (char u : {'L', 'U'})
        for (int op : {0, 1, 2})
            for (double scale : {sizeof(typename real_type<T>::type) == 4 ? 1e-20 : 1e-100,
                                 sizeof(typename real_type<T>::type) == 4 ? 1e20 : 1e100})
                test<T>(h, cpu, trace, op, 65, u, op == 1 ? 2 : 0, 3, 4, native, 0, scale);
    for (char u : {'L', 'U'})
        for (int op : {0, 2})
            test<T>(h, cpu, trace, op, 65, u, 1, 3, 4, native, 0);
}

void invalid_calls(DdlaHandle_t h)
{
    using T = std::complex<double>;
    int d[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(d, h, 17, 17, 3, 4, h->nprows_ - 1, h->npcols_ - 1));
    vector<T> a(h, std::size_t(d[DDLA_LLD_]) * ddla_test::n_loc(h, d));
    int piv[17], info = 0;
    auto rejects = [&](auto call)
    {
        int caught = 0;
        try
        {
            call();
        }
        catch (const std::invalid_argument&)
        {
            caught = 1;
        }
        int count = 0;
        MPI_Allreduce(&caught, &count, 1, MPI_INT, MPI_SUM, h->comm);
        if (count != h->nprocs)
            throw std::runtime_error("Invalid metadata did not fail collectively");
    };
    rejects([&] { phetrf<DdlaBackend::GPU, T>(h, 'L', -1, a.data(), d, piv, info); });
    rejects(
        [&]
        {
            phetrf<DdlaBackend::GPU, T>(h, 'L', 17, h->myid == 0 ? nullptr : a.data(), d, piv,
                                        info);
        });
    rejects(
        [&]
        {
            phetrf<DdlaBackend::GPU, T>(h, 'L', 17, a.data(), d, h->myid == 0 ? nullptr : piv,
                                        info);
        });
    if (h->nprocs > 1)
    {
        rejects(
            [&]
            {
                phetrf<DdlaBackend::GPU, T>(h, h->myid == 0 ? 'L' : 'U', 17, a.data(), d, piv,
                                            info);
            });
        rejects(
            [&]
            {
                phetrf_nopiv<DdlaBackend::GPU, T>(h, 'L', h->myid == 0 ? 16 : 17, a.data(), d,
                                                  info);
            });
    }
    if (!h->myid)
        std::cout << "LDLT_COLLECTIVE_VALIDATION_PASS" << std::endl;
}
} // namespace
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    try
    {
        Options o;
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "--grid" && i + 1 < argc)
            {
                const std::string g = argv[++i];
                const auto x = g.find('x');
                o.pr = std::stoi(g.substr(0, x));
                o.pc = std::stoi(g.substr(x + 1));
            }
            else if (arg == "--n" && i + 1 < argc)
                o.n = std::stoi(argv[++i]);
            else if (arg == "--op" && i + 1 < argc)
                o.operation = std::stoi(argv[++i]);
            else if (arg == "--pattern" && i + 1 < argc)
                o.pattern = std::stoi(argv[++i]);
            else if (arg == "--scalar" && i + 1 < argc)
                o.scalar = argv[++i];
            else if (arg == "--repeats" && i + 1 < argc)
                o.repeats = std::stoi(argv[++i]);
            else
                throw std::runtime_error("Unknown native LDLT test option");
        }
        DdlaHandle_t h = nullptr, cpu = nullptr;
        DDLA_CHECK(ddlaInit(h, DdlaBackend::GPU));
        DDLA_CHECK(ddlaSet(h, MPI_COMM_WORLD, o.pr, o.pc));
        DDLA_CHECK(ddlaInit(cpu, DdlaBackend::CPU));
        DDLA_CHECK(ddlaSet(cpu, MPI_COMM_WORLD, o.pr, o.pc));
        const char* setting = std::getenv("DDLA_EXPECT_NATIVE");
        const bool native = !setting || std::strcmp(setting, "0");
        Trace trace;
        if (o.n >= 0)
        {
            auto run = [&](auto value)
            {
                using T = decltype(value);
                for (int op : {0, 1, 2})
                    if (o.operation < 0 || o.operation == op)
                        for (char u : {'L', 'U'})
                            test<T>(h, cpu, trace, op, o.n, u,
                                    o.pattern >= 0 ? o.pattern
                                    : op == 1      ? 1
                                                   : 0,
                                    64, 64, native, o.repeats);
            };
            if (o.scalar == "d")
                run(double{});
            else if (o.scalar == "s")
                run(float{});
            else if (o.scalar == "c")
                run(std::complex<float>{});
            else if (o.scalar == "z")
                run(std::complex<double>{});
            else
                throw std::runtime_error("Unknown scalar type");
        }
        else
        {
            cases<float>(h, cpu, trace, native);
            cases<double>(h, cpu, trace, native);
            cases<std::complex<float>>(h, cpu, trace, native);
            cases<std::complex<double>>(h, cpu, trace, native);
            invalid_calls(h);
            DdlaHandle_t alternate = nullptr;
            DDLA_CHECK(ddlaInit(alternate, DdlaBackend::GPU));
            DDLA_CHECK(ddlaSet(alternate, MPI_COMM_WORLD, o.pr, o.pc));
            void* old = ddlaGetStream(h);
            DDLA_CHECK(ddlaSetStream(h, ddlaGetStream(alternate)));
            for (int op : {0, 1, 2})
                test<std::complex<double>>(h, cpu, trace, op, 129, 'U', op == 1 ? 2 : 0, 16, 24,
                                           native, 0);
            DDLA_CHECK(ddlaSetStream(h, old));
            DDLA_CHECK(ddlaDestroy(alternate));
        }
        DDLA_CHECK(ddlaDestroy(h));
        DDLA_CHECK(ddlaDestroy(cpu));
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return 0;
}
