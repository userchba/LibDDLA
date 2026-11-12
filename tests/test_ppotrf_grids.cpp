#include "api_grid_test_common.h"
#include "Cholesky/cholesky_workspace.h"
#include <array>
#include <cstring>
#include <type_traits>

using namespace api_grid_test;

namespace {
template <class T> T scalar(double re, double im = 0)
{
    if constexpr (std::is_arithmetic_v<T>) return T(re);
    else return T(re, im);
}
template <class T> T conjugate(T x) { return ddla::detail::host_conj(x); }
template <class T> T factor(int i, int j)
{
    return i == j ? scalar<T>(2 + .01 * i) : scalar<T>(.003 * (1 + (i + j) % 5), .002 * (i - j));
}
template <class T> T matrix(int i, int j)
{
    T result = 0;
    for (int k = 0; k <= std::min(i, j); ++k) result += factor<T>(i, k) * conjugate(factor<T>(j, k));
    return result;
}

template <DdlaBackend Backend, class T>
void check(DdlaHandle_t h, int n, int nb, int rs, int cs, char uplo)
{
    int d[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(d, h, n + 3, n + 3, nb, nb, rs, cs));
    d[DDLA_LLD_] += 3;
    const int rows = ddla_test::m_loc(h, d), cols = ddla_test::n_loc(h, d), ld = d[DDLA_LLD_];
    const T sentinel = scalar<T>(-731, 19);
    const T poison = scalar<T>(std::numeric_limits<double>::quiet_NaN());
    std::vector<T> input(local_size(h, d), sentinel);
    for (int j = 0; j < cols; ++j) for (int i = 0; i < rows; ++i) {
        const int gi = indx_l2g_r(d, h, i), gj = indx_l2g_c(d, h, j);
        if (gi < n && gj < n)
            input[i + std::size_t(j) * ld] = (uplo == 'L' ? gi >= gj : gi <= gj) ? matrix<T>(gi, gj) : poison;
    }
    DeviceBuffer<T> a(h, input.size());
    upload(h, a.ptr, input);
    std::size_t bb = 0, hb = 0;
    cholesky_impl::ppotrf_workspace_size<T>(h, uplo, n, a.ptr, d, false, -1, &bb, &hb);
    constexpr std::size_t guard = 256;
    ddla::vector<char> work(h, bb + 2 * guard);
    std::vector<char> host(hb + 2 * guard, char(0x57));
    DDLA_CHECK(ddlaMemset(h, work.data(), 0x57, work.size()));
    for (int repeat = 0; repeat < 2; ++repeat) {
        upload(h, a.ptr, input);
        int info = -1;
        bool negative;
        if (repeat == 0) negative = ppotrf<Backend,T>(h, uplo, n, a.ptr, 1, 1, d, info);
        else negative = cholesky_impl::ppotrf_run<Backend,T>(h, uplo, n, a.ptr, 1, 1, d, info, false, -1,
            work.data() + guard, bb, host.data() + guard, hb);
        require_close(h, "ppotrf grid INFO", info == 0 && !negative ? 0 : 1, 0);
        const auto output = download(h, a.ptr, input.size());
        double error = 0;
        bool preserved = true;
        for (int j = 0; j < cols; ++j) for (int i = 0; i < ld; ++i) {
            const int gi = i < rows ? indx_l2g_r(d, h, i) : n + 3;
            const int gj = indx_l2g_c(d, h, j);
            const auto at = i + std::size_t(j) * ld;
            if (gi < n && gj < n && (uplo == 'L' ? gi >= gj : gi <= gj)) {
                const T want = uplo == 'L' ? factor<T>(gi, gj) : conjugate(factor<T>(gj, gi));
                const double diff = std::abs(output[at] - want) / (1 + std::abs(want));
                if (!std::isfinite(diff)) error = std::numeric_limits<double>::infinity();
                else error = std::max(error, diff);
            } else preserved = preserved && std::memcmp(&input[at], &output[at], sizeof(T)) == 0;
        }
        require_close(h, "ppotrf analytic factor", error, std::is_same_v<typename real_type<T>::type,float> ? 2e-5 : 3e-12);
        require_close(h, "ppotrf opposite triangle/parent/padding", preserved ? 0 : 1, 0);
        std::array<char, 2 * guard> edge{};
        DDLA_CHECK(ddlaMemcpy(h, edge.data(), work.data(), guard));
        DDLA_CHECK(ddlaMemcpy(h, edge.data() + guard, work.data() + guard + bb, guard));
        check_ddla_sync(h);
        bool intact = true;
        for (std::size_t i = 0; i < guard; ++i)
            intact = intact && edge[i] == char(0x57) && edge[guard + i] == char(0x57)
                && host[i] == char(0x57) && host[guard + hb + i] == char(0x57);
        require_close(h, "ppotrf exact workspace guards", intact ? 0 : 1, 0);
    }
    // A failure after the first panel must report a global, identical INFO.
    auto bad = input;
    for (int j = 0; j < cols; ++j) for (int i = 0; i < rows; ++i) {
        const int gi = indx_l2g_r(d, h, i), gj = indx_l2g_c(d, h, j);
        if (gi < n && gj < n) bad[i + std::size_t(j) * ld] = gi == gj ? T(gi == n - 1 ? -1 : 4) : T(0);
    }
    upload(h, a.ptr, bad);
    int info = 0;
    ppotrf<Backend,T>(h, uplo, n, a.ptr, 1, 1, d, info);
    require_close(h, "ppotrf global failed pivot", info == n ? 0 : 1, 0);
}

template <DdlaBackend Backend, class T> void cases(DdlaHandle_t h, int pr, int pc)
{
    for (int rs = 0; rs < pr; ++rs) for (int cs = 0; cs < pc; ++cs) for (char uplo : {'L', 'U'}) {
        check<Backend,T>(h, 17, 3, rs, cs, uplo);
        check<Backend,T>(h, 38, 8, rs, cs, uplo);
    }
    for (char uplo : {'L', 'U'}) {
        check<Backend,T>(h, 1, 3, pr - 1, pc - 1, uplo);
        check<Backend,T>(h, 2, 3, pr - 1, 0, uplo);
        check<Backend,T>(h, 129, 16, pr - 1, pc - 1, uplo);
    }
}
}

int main(int argc, char** argv)
{
    ddla_test_mpi_init(&argc, &argv);
    int size = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    for (int pr = 1; pr <= size; ++pr) if (size % pr == 0) for (char major : {'R', 'C'}) {
        const int pc = size / pr;
        DdlaHandle_t h = nullptr;
        DDLA_CHECK(ddlaInit(h));
        DDLA_CHECK(ddlaSet(h, MPI_COMM_WORLD, pr, pc, major));
        constexpr auto backend = default_backend_v;
        cases<backend,float>(h, pr, pc);
        cases<backend,double>(h, pr, pc);
        cases<backend,std::complex<float>>(h, pr, pc);
        cases<backend,std::complex<double>>(h, pr, pc);
        if (rank == 0) std::cout << "ppotrf grids PASS " << pr << "x" << pc << " order=" << major << std::endl;
        DDLA_CHECK(ddlaDestroy(h));
    }
    MPI_Finalize();
}
