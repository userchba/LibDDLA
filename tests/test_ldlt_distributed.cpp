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
    Reset reset = reinterpret_cast<Reset>(dlsym(RTLD_DEFAULT, "ddla_zhetrf_trace_reset"));
    Count count = reinterpret_cast<Count>(dlsym(RTLD_DEFAULT, "ddla_zhetrf_trace_count"));
    Bytes peak = reinterpret_cast<Bytes>(dlsym(RTLD_DEFAULT, "ddla_zhetrf_trace_device_peak"));
    Bytes largest =
        reinterpret_cast<Bytes>(dlsym(RTLD_DEFAULT, "ddla_zhetrf_trace_device_largest"));
    Bytes host = reinterpret_cast<Bytes>(dlsym(RTLD_DEFAULT, "ddla_zhetrf_trace_host_largest"));
};

template <DdlaBackend Backend, class T>
void test(DdlaHandle_t h, DdlaHandle_t cpu, const Trace &trace, int op, int n, int nrhs, char u,
          int pattern, int mb, int nb, bool shifted, int repeats, double scale = 1)
{
    const bool solve = op >= 3, pivot = op == 1 || op == 3;
    const T guard = number<T>(131, -719), nan = number<T>(NAN, NAN);
    int da[DDLA_DLEN_], db[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(da, h, n + 3, n + 4, mb, nb, h->nprows_ - 1, h->npcols_ - 1));
    da[DDLA_LLD_] += 3;
    DDLA_CHECK(ddlaDescInit(db, h, n + 4, nrhs + 5, shifted ? 5 : mb, shifted ? 7 : nb,
                            shifted ? 0 : da[DDLA_RSRC_], shifted ? 0 : da[DDLA_CSRC_]));
    db[DDLA_LLD_] += 4;
    const int ar = ddla_test::m_loc(h, da), ac = ddla_test::n_loc(h, da);
    const int br = ddla_test::m_loc(h, db), bc = ddla_test::n_loc(h, db);
    std::vector<T> input(std::size_t(da[DDLA_LLD_]) * ac, guard);
    for (int j = 0; j < ac; ++j)
        for (int i = 0; i < ar; ++i)
        {
            const int gi = indx_l2g_r(da, h, i), gj = indx_l2g_c(da, h, j);
            if (gi < n && gj < n)
            {
                T value = (u == 'L' ? gi >= gj : gi <= gj) ? entry<T>(gi, gj, n, pattern) * T(scale)
                                                           : nan;
                if constexpr (!std::is_arithmetic_v<T>)
                    if (gi == gj)
                        value.imag(NAN);
                input[i + std::size_t(j) * da[DDLA_LLD_]] = value;
            }
        }
    vector<T> a(h, input.size()), b(h, std::size_t(db[DDLA_LLD_]) * bc);
    std::vector<T> rhs(b.size(), guard);
    std::vector<int> piv(std::max(1, n)), piv_before;
    DDLA_CHECK(ddlaMemcpy(h, a.data(), input.data(), input.size() * sizeof(T)));
    int factor_info = 0;
    if (solve && n && nrhs)
    {
        std::vector<T> dense, original, full_rhs;
        if (!h->myid)
        {
            original.resize(std::size_t(n) * n);
            dense.resize(original.size());
            full_rhs.resize(std::size_t(n) * nrhs);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    original[i + std::size_t(j) * n] = entry<T>(i, j, n, pattern) * T(scale);
            dense = original;
            if (pivot)
                ldlt_impl::factor(std::true_type{}, u, n, dense.data(), n, piv.data(), factor_info);
            else
                ldlt_impl::factor_nopiv<true>(cpu, u, n, dense.data(), n, factor_info);
            std::vector<T> exact(std::size_t(n) * nrhs);
            for (int j = 0; j < nrhs; ++j)
                for (int i = 0; i < n; ++i)
                    exact[i + std::size_t(j) * n] = number<T>(1 + .01 * (i % 7), .05 * (j % 3));
            blas_impl::gemm_run<DdlaBackend::CPU, T>(cpu, 'N', 'N', n, nrhs, n, T(1),
                                                     original.data(), n, exact.data(), n, T(0),
                                                     full_rhs.data(), n);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    if (u == 'L' ? i < j : i > j)
                        dense[i + std::size_t(j) * n] = nan;
        }
        MPI_CHECK(MPI_Bcast(&factor_info, 1, MPI_INT, 0, h->comm));
        if (pivot)
            MPI_CHECK(MPI_Bcast(piv.data(), n, MPI_INT, 0, h->comm));
        { // Oracle transfer buffers retire before the measured API call.
            vector<T> global_a(h, h->myid ? 0 : std::size_t(n) * n),
                global_b(h, h->myid ? 0 : std::size_t(n) * nrhs);
            if (!h->myid)
            {
                DDLA_CHECK(ddlaMemcpy(h, global_a.data(), dense.data(), dense.size() * sizeof(T)));
                DDLA_CHECK(
                    ddlaMemcpy(h, global_b.data(), full_rhs.data(), full_rhs.size() * sizeof(T)));
            }
            DDLA_CHECK(ddlaMemcpy(h, b.data(), rhs.data(), rhs.size() * sizeof(T)));
            scatter<Backend, T>(h, n, n, a.data(), 1, 1, da, 0, global_a.data(), n);
            scatter<Backend, T>(h, n, nrhs, b.data(), 1, 1, db, 0, global_b.data(), n);
        }
        DDLA_CHECK(ddlaMemcpy(h, input.data(), a.data(), input.size() * sizeof(T)));
        DDLA_CHECK(ddlaMemcpy(h, rhs.data(), b.data(), rhs.size() * sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
        piv_before = piv;
    }
    const bool native =
        !std::getenv("DDLA_EXPECT_NATIVE") || std::strcmp(std::getenv("DDLA_EXPECT_NATIVE"), "0");
    std::vector<double> times;
    unsigned long long peak = 0, largest = 0, host = 0;
    int info = 0;
    for (int run = 0; run < std::max(1, repeats + 1); ++run)
    {
        DDLA_CHECK(ddlaMemcpy(h, a.data(), input.data(), input.size() * sizeof(T)));
        if (solve)
            DDLA_CHECK(ddlaMemcpy(h, b.data(), rhs.data(), rhs.size() * sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
        MPI_CHECK(MPI_Barrier(h->comm));
        if (trace.reset)
            trace.reset();
        const auto start = std::chrono::steady_clock::now();
        if (op == 1)
            phetrf<Backend, T>(h, u, n, a.data(), da, piv.data(), info);
        if (op == 2)
            phetrf_nopiv<Backend, T>(h, u, n, a.data(), da, info);
        if (op == 3)
            phetrs<Backend, T>(h, u, n, nrhs, a.data(), da, piv.data(), b.data(), db, info);
        if (op == 4)
            phetrs_nopiv<Backend, T>(h, u, n, nrhs, a.data(), da, b.data(), db, info);
        DDLA_CHECK(ddlaSynchronize(h));
        double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const long long calls = trace.count ? trace.count() : 0;
        if (trace.peak)
            peak = std::max(peak, trace.peak());
        if (trace.largest)
            largest = std::max(largest, trace.largest());
        if (trace.host)
            host = std::max(host, trace.host());
        if constexpr (Backend == DdlaBackend::GPU)
        {
            if (native && calls)
                throw std::runtime_error("GPU API called CPU factor/solve/BLAS");
            if (!native && solve && n > 1 && nrhs && factor_info == 0 && !h->myid && trace.count &&
                calls == 0)
                throw std::runtime_error("Old solve counter did not fire");
        }
        if (native && n >= 512 && h->nprocs > 1 &&
            largest >= std::size_t(n) * std::max(n, nrhs) * sizeof(T))
            throw std::runtime_error("Replicated/root dense backend workspace");
        double maximum = 0;
        MPI_CHECK(MPI_Allreduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, h->comm));
        if (run)
            times.push_back(maximum);
    }
    const int expected = (!n || (solve && !nrhs)) ? 0
                         : solve                  ? (pivot         ? 0
                                                     : factor_info ? (pattern == 4 ? 1 : n / 2 + 1)
                                                                   : 0)
                         : pattern == 6 && !pivot ? (u == 'L' ? n / 2 + 1 : n / 2 + 2)
                         : pattern == 1 && !pivot && n > 1 ? (u == 'L' ? 1 : n)
                         : pattern == 4                    ? (u == 'L' ? 1 : n)
                         : pattern == 5                    ? n / 2 + 1
                                                           : 0;
    if (info != expected)
    {
        std::cerr << "INFO op=" << op << " n=" << n << " actual=" << info
                  << " expected=" << expected << std::endl;
        throw std::runtime_error("Wrong INFO");
    }
    std::vector<T> result(input.size());
    DDLA_CHECK(ddlaMemcpy(h, result.data(), a.data(), result.size() * sizeof(T)));
    DDLA_CHECK(ddlaSynchronize(h));
    if (solve)
    {
        if (std::memcmp(result.data(), input.data(), input.size() * sizeof(T)))
            throw std::runtime_error("Solve modified factors");
        if (pivot && piv != piv_before && n && nrhs)
            throw std::runtime_error("Solve modified IPIV");
    }
    else
        for (int j = 0; j < ac; ++j)
            for (int i = 0; i < da[DDLA_LLD_]; ++i)
            {
                const int gi = indx_l2g_r(da, h, i), gj = indx_l2g_c(da, h, j);
                if (i < ar && gi < n && gj < n && (u == 'L' ? gi >= gj : gi <= gj))
                    continue;
                if (std::memcmp(&result[i + std::size_t(j) * da[DDLA_LLD_]],
                                &input[i + std::size_t(j) * da[DDLA_LLD_]], sizeof(T)))
                    throw std::runtime_error("Factor modified unused storage");
            }
    double error = 0;
    if (solve)
    {
        std::vector<T> solved(rhs.size());
        DDLA_CHECK(ddlaMemcpy(h, solved.data(), b.data(), solved.size() * sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
        for (int j = 0; j < bc; ++j)
            for (int i = 0; i < db[DDLA_LLD_]; ++i)
            {
                const int gi = indx_l2g_r(db, h, i), gj = indx_l2g_c(db, h, j);
                const auto pos = i + std::size_t(j) * db[DDLA_LLD_];
                if (info || i >= br || gi >= n || gj >= nrhs)
                {
                    if (std::memcmp(&solved[pos], &rhs[pos], sizeof(T)))
                        throw std::runtime_error("Solve changed guards/failed RHS");
                }
                else
                    error =
                        std::max(error, double(std::abs(solved[pos] - number<T>(1 + .01 * (gi % 7),
                                                                                .05 * (gj % 3)))));
            }
        double maximum = 0;
        MPI_CHECK(MPI_Allreduce(&error, &maximum, 1, MPI_DOUBLE, MPI_MAX, h->comm));
        error = maximum;
    }
    else if (n && !info)
    {
        vector<T> global(h, h->myid ? 0 : std::size_t(n) * n);
        gather<Backend, T>(h, n, n, a.data(), 1, 1, da, 0, global.data(), n);
        if (!h->myid)
        {
            std::vector<T> dense(std::size_t(n) * n), right(n);
            DDLA_CHECK(ddlaMemcpy(h, dense.data(), global.data(), dense.size() * sizeof(T)));
            DDLA_CHECK(ddlaSynchronize(h));
            for (int i = 0; i < n; ++i)
                for (int k = 0; k < n; ++k)
                    right[i] += entry<T>(i, k, n, pattern) * T(scale) *
                                number<T>(1 + .01 * (k % 7), .05 * (k % 3));
            int status = 0;
            if (pivot)
                ldlt_impl::solve(std::true_type{}, u, n, 1, dense.data(), n, piv.data(),
                                 right.data(), n, status);
            else
                ldlt_impl::solve_nopiv<true>(cpu, u, n, 1, dense.data(), n, right.data(), n,
                                             status);
            if (status)
                throw std::runtime_error("Independent solve rejected factors");
            for (int i = 0; i < n; ++i)
                error = std::max(error, double(std::abs(right[i] - number<T>(1 + .01 * (i % 7),
                                                                             .05 * (i % 3)))));
        }
    }
    if (!std::isfinite(error) ||
        error > 300 * std::max(n, 1) * std::numeric_limits<typename real_type<T>::type>::epsilon())
    {
        std::cerr << "ERROR backend=" << int(Backend) << " op=" << op << " n=" << n
                  << " rhs=" << nrhs << " u=" << u << " pattern=" << pattern << " error=" << error
                  << std::endl;
        throw std::runtime_error("Solution mismatch");
    }
    if (repeats)
    {
        std::sort(times.begin(), times.end());
        for (int rank = 0; rank < h->nprocs; ++rank)
        {
            MPI_CHECK(MPI_Barrier(h->comm));
            if (rank == h->myid)
                std::cout
                    << std::setprecision(9) << "LDLT_SHARED_BENCH backend=" << int(Backend)
                    << " op=" << op << " n=" << n << " nrhs=" << nrhs << " uplo=" << u << " mode="
                    << (std::getenv("DDLA_LDLT_PANEL") ? std::getenv("DDLA_LDLT_PANEL") : "default")
                    << " grid=" << h->nprows_ << "x" << h->npcols_ << " rank=" << rank
                    << " median_s=" << times[times.size() / 2] << " peak=" << peak
                    << " largest=" << largest << " host_largest=" << host
                    << " local_rows=" << num_loc(n, mb, h->myprow_, da[DDLA_RSRC_], h->nprows_)
                    << " local_cols=" << num_loc(n, nb, h->mypcol_, da[DDLA_CSRC_], h->npcols_)
                    << " error=" << error << std::endl;
        }
    }
    else if (!h->myid)
        std::cout << "LDLT_SHARED_PASS backend=" << int(Backend) << " op=" << op
                  << " bytes=" << sizeof(T) << " n=" << n << " rhs=" << nrhs << " u=" << u
                  << " pattern=" << pattern << " shifted=" << shifted << " error=" << error
                  << std::endl;
}
template <DdlaBackend B> void invalid_calls(DdlaHandle_t h)
{
    using T = std::complex<double>;
    int da[DDLA_DLEN_], db[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(da, h, 17, 17, 3, 4, 0, 0));
    DDLA_CHECK(ddlaDescInit(db, h, 17, 3, 3, 2, 0, 0));
    vector<T> a(h, std::size_t(da[DDLA_LLD_]) * ddla_test::n_loc(h, da));
    vector<T> b(h, std::size_t(db[DDLA_LLD_]) * ddla_test::n_loc(h, db));
    std::vector<int> piv(17, 0);
    int info = 0;
    auto rejects = [&](auto call, bool mismatch = false)
    {
        int caught = 0, total = 0;
        try
        {
            call();
        }
        catch (const std::invalid_argument &)
        {
            if (mismatch)
                throw;
            caught = 1;
        }
        catch (const std::runtime_error &e)
        {
            if (!mismatch || std::string(e.what()).find(
                                 "DDLA_LDLT_PANEL differs between the ranks") == std::string::npos)
                throw;
            caught = 1;
        }
        MPI_CHECK(MPI_Allreduce(&caught, &total, 1, MPI_INT, MPI_SUM, h->comm));
        if (total != h->nprocs)
            throw std::runtime_error("Invalid call did not fail on every rank");
    };
    rejects([&] { phetrs<B, T>(h, 'L', 17, 3, a.data(), da, piv.data(), b.data(), db, info); });
    const char *setting = std::getenv("DDLA_LDLT_PANEL");
    const bool existed = setting != nullptr;
    const std::string previous = setting ? setting : "";
    setenv("DDLA_LDLT_PANEL", "invalid", 1);
    rejects([&] { phetrf<B, T>(h, 'L', 17, a.data(), da, piv.data(), info); });
    if (h->nprocs > 1)
    {
        setenv("DDLA_LDLT_PANEL", h->myid ? "inplace" : "gather", 1);
        rejects([&] { phetrf<B, T>(h, 'L', 17, a.data(), da, piv.data(), info); }, true);
    }
    if (existed)
        setenv("DDLA_LDLT_PANEL", previous.c_str(), 1);
    else
        unsetenv("DDLA_LDLT_PANEL");
}
} // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    try
    {
        int world = 1;
        MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world));
        int pr = int(std::sqrt(double(world)));
        while (world % pr)
            --pr;
        int pc = world / pr, n = -1, nrhs = 3, op = -1, repeats = 0, pattern = -1;
        std::string backend = "all", scalar = "z";
        for (int i = 1; i < argc; ++i)
        {
            std::string s = argv[i];
            if (i + 1 >= argc)
                throw std::runtime_error("Missing argument");
            if (s == "--grid")
            {
                std::string g = argv[++i];
                auto x = g.find('x');
                pr = std::stoi(g.substr(0, x));
                pc = std::stoi(g.substr(x + 1));
            }
            else if (s == "--n")
                n = std::stoi(argv[++i]);
            else if (s == "--nrhs")
                nrhs = std::stoi(argv[++i]);
            else if (s == "--op")
                op = std::stoi(argv[++i]);
            else if (s == "--repeats")
                repeats = std::stoi(argv[++i]);
            else if (s == "--backend")
                backend = argv[++i];
            else if (s == "--scalar")
                scalar = argv[++i];
            else if (s == "--pattern")
                pattern = std::stoi(argv[++i]);
            else
                throw std::runtime_error("Unknown option");
        }
        Trace trace;
        auto run = [&](auto btag)
        {
            constexpr auto B = decltype(btag)::value;
            DdlaHandle_t h = nullptr, cpu = nullptr;
            DDLA_CHECK(ddlaInit(h, B));
            DDLA_CHECK(ddlaSet(h, MPI_COMM_WORLD, pr, pc));
            DDLA_CHECK(ddlaInit(cpu, DdlaBackend::CPU));
            DDLA_CHECK(ddlaSet(cpu, MPI_COMM_WORLD, pr, pc));
            auto typed = [&](auto tag)
            {
                using T = decltype(tag);
                if (n >= 0)
                {
                    for (int method : {1, 2, 3, 4})
                        if (op < 0 || op == method)
                            for (char u : {'L', 'U'})
                                test<B, T>(h, cpu, trace, method, n, nrhs, u,
                                           pattern < 0 ? (method == 1 || method == 3 ? 1 : 0)
                                                       : pattern,
                                           64, 64, false, repeats);
                }
                else
                {
                    for (int size : {0, 1, 2, 17, 65, 129})
                        for (char u : {'L', 'U'})
                            for (int method : {1, 2, 3, 4})
                                test<B, T>(h, cpu, trace, method, size, 3, u,
                                           method == 1 || method == 3 ? 2 : 0, 3, 4, true, 0);
                    for (char u : {'L', 'U'})
                        for (int layout : {0, 1})
                        {
                            for (int pat : {0, 1, 2, 3, 6})
                                test<B, T>(h, cpu, trace, 3, 127, 17, u, pat, 64, 64, layout, 0);
                            for (int pat : {0, 4, 5, 6})
                                test<B, T>(h, cpu, trace, 4, 127, 17, u, pat, 64, 64, layout, 0);
                            test<B, T>(h, cpu, trace, 1, 127, 3, u, 1, 64, 64, layout, 0);
                            test<B, T>(h, cpu, trace, 1, 129, 3, u, 3, 16, 24, layout, 0);
                            test<B, T>(h, cpu, trace, 2, 129, 3, u, 0, 16, 24, layout, 0);
                        }
                    for (char u : {'L', 'U'})
                    {
                        for (int method : {3, 4})
                            test<B, T>(h, cpu, trace, method, 65, 0, u, 0, 3, 4, true, 0);
                        for (int method : {1, 2})
                            for (int pat : {4, 5, 6})
                                test<B, T>(h, cpu, trace, method, 129, 3, u, pat, 16, 24, true, 0);
                    }
                    for (char u : {'L', 'U'})
                        for (int method : {1, 2, 3, 4})
                            for (double scale :
                                 {sizeof(typename real_type<T>::type) == 4 ? 1e-20 : 1e-100,
                                  sizeof(typename real_type<T>::type) == 4 ? 1e20 : 1e100})
                                test<B, T>(h, cpu, trace, method, 65, 3, u,
                                           method == 1 || method == 3 ? 1 : 0, 16, 24, true, 0,
                                           scale);
                }
            };
            if (n < 0 || scalar == "s")
                typed(float{});
            if (n < 0 || scalar == "d")
                typed(double{});
            if (n < 0 || scalar == "c")
                typed(std::complex<float>{});
            if (n < 0 || scalar == "z")
                typed(std::complex<double>{});
            if (n < 0)
            {
                invalid_calls<B>(h);
                if constexpr (B == DdlaBackend::GPU)
                {
                    DdlaHandle_t alternate = nullptr;
                    DDLA_CHECK(ddlaInit(alternate, B));
                    DDLA_CHECK(ddlaSet(alternate, MPI_COMM_WORLD, pr, pc));
                    auto previous = ddlaGetStream(h);
                    DDLA_CHECK(ddlaSetStream(h, ddlaGetStream(alternate)));
                    for (int method : {1, 2, 3, 4})
                        test<B, std::complex<double>>(h, cpu, trace, method, 65, 3, 'U',
                                                      method == 1 || method == 3 ? 1 : 0, 3, 4,
                                                      true, 0);
                    DDLA_CHECK(ddlaSetStream(h, previous));
                    DDLA_CHECK(ddlaDestroy(alternate));
                }
            }
            DDLA_CHECK(ddlaDestroy(h));
            DDLA_CHECK(ddlaDestroy(cpu));
        };
        if (backend == "all" || backend == "cpu")
            run(std::integral_constant<DdlaBackend, DdlaBackend::CPU>{});
#if DDLA_HAS_GPU
        if (backend == "all" || backend == "gpu")
            run(std::integral_constant<DdlaBackend, DdlaBackend::GPU>{});
#endif
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return 0;
}
