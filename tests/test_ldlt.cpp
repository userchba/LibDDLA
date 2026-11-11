#include "ddla_stream.h"
#include "api_grid_test_common.h"
#include <ddla/ddla_vector.h>
#include <cstring>
using namespace api_grid_test;
namespace {
template <class T> T scalar(double r, double i = 0) {
    if constexpr (std::is_arithmetic_v<T>)
        return T(r);
    else
        return T(r, i);
}
template <bool H, class T> T entry(int i, int j, bool pair) {
    if (pair)
        return T(i != j);
    if (i == j)
        return T((i % 2 ? -1 : 1) * (4 + i));
    const int lo = std::min(i, j), hi = std::max(i, j);
    return scalar<T>(.03 * (1 + lo + hi), .02 * (lo + 1) * (H && i > j ? -1 : 1));
}
template <DdlaBackend B, bool H, bool Pivot, class T>
void test(DdlaHandle_t h, bool distributed, int n, bool pair) {
    int da[DDLA_DLEN_], db[DDLA_DLEN_];
    ddlaDescInit(da, h, n + 3, n + 4, 3, 4, h->nprows_ - 1, h->npcols_ - 1);
    da[DDLA_LLD_] += 2;
    ddlaDescInit(db, h, n + 2, 5, 2, 3, 0, 0);
    db[DDLA_LLD_] += 3;
    for (char u : {'U', 'L'}) {
        const int lda = distributed ? da[DDLA_LLD_] : std::max(1, n) + 3;
        const int ldb = distributed ? db[DDLA_LLD_] : std::max(1, n) + 4;
        const int ac = distributed ? ddla_test::n_loc(h, da) : n + 1;
        const int bc = distributed ? ddla_test::n_loc(h, db) : 3;
        std::vector<T> a(std::max(1, lda * ac), T(131)), b(std::max(1, ldb * bc), T(139));
        for (int j = 0; j < ac; ++j)
            for (int i = 0; i < lda; ++i) {
                const int gi = distributed ? indx_l2g_r(da, h, i) : i,
                          gj = distributed ? indx_l2g_c(da, h, j) : j;
                if (gi < n && gj < n && (!distributed || i < ddla_test::m_loc(h, da)))
                    a[i + std::size_t(j) * lda] = (u == 'L' ? gi >= gj : gi <= gj)
                                                      ? entry<H, T>(gi, gj, pair)
                                                      : scalar<T>(NAN, NAN);
            }
        for (int j = 0; j < bc; ++j)
            for (int i = 0; i < ldb; ++i) {
                const int gi = distributed ? indx_l2g_r(db, h, i) : i,
                          gj = distributed ? indx_l2g_c(db, h, j) : j;
                if (gi < n && gj < 2 && (!distributed || i < ddla_test::m_loc(h, db))) {
                    T value = 0;
                    for (int k = 0; k < n; ++k)
                        value += entry<H, T>(gi, k, pair) * scalar<T>(1 + .1 * k, .05 * gj);
                    b[i + std::size_t(j) * ldb] = value;
                }
            }
        const auto olda = a, oldb = b;
        DeviceBuffer<T> A(h, a.size()), RHS(h, b.size());
        upload(h, A.ptr, a);
        upload(h, RHS.ptr, b);
        std::vector<int> piv(std::max(1, n), 0);
        int info = -1;
        if (!distributed) {
            if constexpr (H == false) {
                if constexpr (Pivot == false)
                    sytrf_nopiv<B, T>(h, u, n, A.ptr, lda, info);
                if constexpr (Pivot == true)
                    sytrf<B, T>(h, u, n, A.ptr, lda, piv.data(), info);
            }
            if constexpr (H == true) {
                if constexpr (Pivot == false)
                    hetrf_nopiv<B, T>(h, u, n, A.ptr, lda, info);
                if constexpr (Pivot == true)
                    hetrf<B, T>(h, u, n, A.ptr, lda, piv.data(), info);
            }
        } else {
            if constexpr (H == false) {
                if constexpr (Pivot == false)
                    psytrf_nopiv<B, T>(h, u, n, A.ptr, da, info);
                if constexpr (Pivot == true)
                    psytrf<B, T>(h, u, n, A.ptr, da, piv.data(), info);
            }
            if constexpr (H == true) {
                if constexpr (Pivot == false)
                    phetrf_nopiv<B, T>(h, u, n, A.ptr, da, info);
                if constexpr (Pivot == true)
                    phetrf<B, T>(h, u, n, A.ptr, da, piv.data(), info);
            }
        }
        require_close(h, "factor info", info, 0);
        const auto fact = download(h, A.ptr, a.size());
        bool guarded = true;
        for (int j = 0; j < ac; ++j)
            for (int i = 0; i < lda; ++i) {
                const int gi = distributed ? indx_l2g_r(da, h, i) : i,
                          gj = distributed ? indx_l2g_c(da, h, j) : j;
                const bool active = gi < n && gj < n &&
                                    (!distributed || i < ddla_test::m_loc(h, da)) &&
                                    (u == 'L' ? gi >= gj : gi <= gj);
                if (!active)
                    guarded = guarded && !std::memcmp(&fact[i + std::size_t(j) * lda],
                                                      &olda[i + std::size_t(j) * lda], sizeof(T));
            }
        require_close(h, "factor opposite triangle and padding", guarded ? 0 : 1, 0);
        if constexpr (Pivot)
            if (pair)
                require_close(h, "2x2 pivot produced", piv[0] < 0 && piv[1] < 0 ? 0 : 1, 0);
        if (!distributed) {
            if constexpr (H == false) {
                if constexpr (Pivot == false)
                    sytrs_nopiv<B, T>(h, u, n, 2, A.ptr, lda, RHS.ptr, ldb, info);
                if constexpr (Pivot == true)
                    sytrs<B, T>(h, u, n, 2, A.ptr, lda, piv.data(), RHS.ptr, ldb, info);
            }
            if constexpr (H == true) {
                if constexpr (Pivot == false)
                    hetrs_nopiv<B, T>(h, u, n, 2, A.ptr, lda, RHS.ptr, ldb, info);
                if constexpr (Pivot == true)
                    hetrs<B, T>(h, u, n, 2, A.ptr, lda, piv.data(), RHS.ptr, ldb, info);
            }
        } else {
            if constexpr (H == false) {
                if constexpr (Pivot == false)
                    psytrs_nopiv<B, T>(h, u, n, 2, A.ptr, da, RHS.ptr, db, info);
                if constexpr (Pivot == true)
                    psytrs<B, T>(h, u, n, 2, A.ptr, da, piv.data(), RHS.ptr, db, info);
            }
            if constexpr (H == true) {
                if constexpr (Pivot == false)
                    phetrs_nopiv<B, T>(h, u, n, 2, A.ptr, da, RHS.ptr, db, info);
                if constexpr (Pivot == true)
                    phetrs<B, T>(h, u, n, 2, A.ptr, da, piv.data(), RHS.ptr, db, info);
            }
        }
        require_close(h, "solve info", info, 0);
        const auto result = download(h, RHS.ptr, b.size());
        double error = 0;
        guarded = true;
        for (int j = 0; j < bc; ++j)
            for (int i = 0; i < ldb; ++i) {
                const int gi = distributed ? indx_l2g_r(db, h, i) : i,
                          gj = distributed ? indx_l2g_c(db, h, j) : j;
                if (gi < n && gj < 2 && (!distributed || i < ddla_test::m_loc(h, db)))
                    error = std::max(error, double(std::abs(result[i + std::size_t(j) * ldb] -
                                                            scalar<T>(1 + .1 * gi, .05 * gj))));
                else
                    guarded = guarded && !std::memcmp(&result[i + std::size_t(j) * ldb],
                                                      &oldb[i + std::size_t(j) * ldb], sizeof(T));
            }
        require_close(h, "solution against independently constructed RHS", error,
                      200 * n * std::numeric_limits<typename real_type<T>::type>::epsilon());
        require_close(h, "solve RHS padding", guarded ? 0 : 1, 0);
        require_close(
            h, "solve immutable factors",
            std::memcmp(fact.data(), download(h, A.ptr, a.size()).data(), fact.size() * sizeof(T))
                ? 1
                : 0,
            0); // NaN guards checked bytewise below
    }
}
template <DdlaBackend B, class T> void cases(DdlaHandle_t h) {
    for (bool dist : {false, true}) {
        test<B, false, false, T>(h, dist, 17, false);
        test<B, true, false, T>(h, dist, 17, false);
        test<B, false, true, T>(h, dist, 17, false);
        test<B, true, true, T>(h, dist, 17, false);
        test<B, false, true, T>(h, dist, 2, true);
        test<B, true, true, T>(h, dist, 2, true);
    }
    DeviceBuffer<T> a(h, 4);
    upload(h, a.ptr, std::vector<T>{T(0), T(1), T(1), T(0)});
    int info = 0;
    sytrf_nopiv<B, T>(h, 'L', 2, a.ptr, 2, info);
    require_close(h, "no pivot zero breakdown", info, 1);
    sytrf_nopiv<B, T>(h, 'L', 0, nullptr, 1, info);
    require_close(h, "empty factor", info, 0);
    int piv[2];
    upload(h, a.ptr, std::vector<T>(4, T(0)));
    hetrf<B, T>(h, 'U', 2, a.ptr, 2, piv, info);
    require_close(h, "singular pivoted factor", info > 0 ? 0 : 1, 0);
}
template <DdlaBackend B> void run(int pr, int pc) {
    DdlaHandle_t h = nullptr;
    DDLA_CHECK(ddlaInit(h, B));
    DDLA_CHECK(ddlaSet(h, MPI_COMM_WORLD, pr, pc));
    cases<B, float>(h);
    cases<B, double>(h);
    cases<B, std::complex<float>>(h);
    cases<B, std::complex<double>>(h);
    DDLA_CHECK(ddlaDestroy(h));
}
} // namespace
int main(int argc, char **argv) {
    ddla_test_mpi_init(&argc, &argv);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    TestOptions options;
    std::string error;
    if (!parse_options(argc, argv, size, options, error))
        MPI_Abort(MPI_COMM_WORLD, 2);
    for (auto grid : options.grids) {
        run<DdlaBackend::CPU>(grid.first, grid.second);
#if DDLA_HAS_GPU
        run<DdlaBackend::GPU>(grid.first, grid.second);
#endif
    }
    MPI_Finalize();
    return 0;
}
