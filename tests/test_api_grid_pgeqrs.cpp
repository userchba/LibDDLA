#include "api_grid_test_common.h"
#include <ddla/ddla_vector.h>
#include "../src/helpers.h"
#include <array>
#include <cstring>
#include <type_traits>

using namespace api_grid_test;

namespace {

template <typename T>
T scalar(double re, double im = 0)
{
    using R = typename real_type<T>::type;
    if constexpr (std::is_same<T, R>::value) return T(re);
    else return T(R(re), R(im));
}

template <typename T>
T coefficient(int i, int j, int n)
{
    if (i < n) return i == j ? scalar<T>(4 + i % 3, 0.4) : T(0);
    return scalar<T>(0.1 * std::sin(i + 2 * j + 1), 0.1 * std::cos(2 * i - j));
}

template <typename T>
T solution(int i, int j)
{
    return scalar<T>(0.3 * std::sin(i + j + 1), 0.2 * std::cos(2 * i + j));
}

template <typename T>
T residual(int i, int rhs, int m, int n, bool noisy)
{
    if (!noisy) return T(0);
    if (i >= n) return scalar<T>(0.05 * std::cos(i + rhs), 0.03 * std::sin(i - rhs));
    T sum = 0;
    for (int row = n; row < m; ++row)
        sum += detail::host_conj(coefficient<T>(row, i, n)) * residual<T>(row, rhs, m, n, true);
    return -sum / detail::host_conj(coefficient<T>(i, i, n));
}

template <DdlaBackend Backend, typename T>
void run_case(DdlaHandle_t h, int m, int n, int nrhs, int nb, bool shifted)
{
    int pr, pc, row, col;
    ddlaGetGridDims(h, pr, pc);
    ddlaGetGridCoords(h, row, col);
    const int rsrc = shifted ? pr - 1 : 0;
    const int csrc = shifted ? pc - 1 : 0;
    int da[9], db[9];
    DDLA_CHECK(ddlaDescInit(da, h, m + 3, n + 4, nb, nb, rsrc, csrc,
                            num_loc(m + 3, nb, row, rsrc, pr) + 3));
    DDLA_CHECK(ddlaDescInit(db, h, m + 3, nrhs + 4, nb, nb + 1, rsrc, 0,
                            num_loc(m + 3, nb, row, rsrc, pr) + 4));
    const T guard = scalar<T>(-731, 29);
    const std::size_t asize = local_size(h, da), bsize = local_size(h, db);
    std::vector<T> ah(asize + 5, guard), bh(bsize + 5, guard);
    for (int j = 0; j < ddla_test::n_loc(h, da); ++j)
        for (int i = 0; i < ddla_test::m_loc(h, da); ++i) {
            const int gi = indx_l2g_r(da, h, i), gj = indx_l2g_c(da, h, j);
            if (gi < m && gj < n) ah[i + std::size_t(j) * da[DDLA_LLD_]] = coefficient<T>(gi, gj, n);
        }
    ddla::vector<T> a(h, ah.size()), b(h, bh.size());
    upload(h, a.data(), ah);
    std::vector<T> tau(std::size_t(ddla_test::n_loc(h, da)) + 3, guard);
    int info = -1;
    pgeqrf<Backend, T>(h, m, n, a.data(), da, tau.data(), info);
    require_close(h, "QR status", info == 0 ? 0 : 1, 0);
    const auto factored = download(h, a.data(), ah.size());
    const auto saved_tau = tau;
    const double tolerance = std::is_same<typename real_type<T>::type, float>::value ? 4e-4 : 2e-11;

    for (bool noisy : {false, true}) {
        std::fill(bh.begin(), bh.end(), guard);
        for (int j = 0; j < ddla_test::n_loc(h, db); ++j)
            for (int i = 0; i < ddla_test::m_loc(h, db); ++i) {
                const int gi = indx_l2g_r(db, h, i), gj = indx_l2g_c(db, h, j);
                if (gi >= m || gj >= nrhs) continue;
                T value = residual<T>(gi, gj, m, n, noisy);
                for (int k = 0; k < n; ++k)
                    value += coefficient<T>(gi, k, n) * solution<T>(k, gj);
                bh[i + std::size_t(j) * db[DDLA_LLD_]] = value;
            }
        upload(h, b.data(), bh);
        pgeqrs<Backend, T>(h, m, n, nrhs, a.data(), da, tau.data(), b.data(), db, info);
        const auto got = download(h, b.data(), bh.size());
        double error = 0, tail_norm = 0, expected_norm = 0;
        for (std::size_t offset = 0; offset < got.size(); ++offset) {
            const int j = int(offset / db[DDLA_LLD_]), i = int(offset % db[DDLA_LLD_]);
            const bool logical = offset < bsize && i < ddla_test::m_loc(h, db);
            const int gi = logical ? indx_l2g_r(db, h, i) : m;
            const int gj = logical ? indx_l2g_c(db, h, j) : nrhs;
            if (gi < n && gj < nrhs) {
                const double difference = std::abs(got[offset] - solution<T>(gi, gj));
                error = std::isfinite(difference) ? std::max(error, difference) : INFINITY;
            } else if (gi < m && gj < nrhs) {
                tail_norm += std::norm(got[offset]);
            } else if (got[offset] != bh[offset]) error = INFINITY;
        }
        if (ddlaGetRank(h) == 0)
            for (int j = 0; j < nrhs; ++j)
                for (int i = 0; i < m; ++i) expected_norm += std::norm(residual<T>(i, j, m, n, noisy));
        double norms[2] = {tail_norm, expected_norm};
        MPI_Allreduce(MPI_IN_PLACE, norms, 2, MPI_DOUBLE, MPI_SUM, ddlaGetCommunicator(h));
        require_close(h, "QR least-squares solution and B guards", error, tolerance);
        require_close(h, "QR transformed residual norm", std::abs(norms[0] - norms[1]), tolerance);
        const auto after = download(h, a.data(), ah.size());
        require_close(h, "QR inputs remain read-only", info == 0 &&
                      std::memcmp(after.data(), factored.data(), after.size() * sizeof(T)) == 0 &&
                      tau == saved_tau ? 0 : 1, 0);
    }

    auto rejects = [&](auto invoke, const char* name) {
        bool threw = false;
        try { invoke(); } catch (const std::exception&) { threw = true; }
        require_close(h, name, threw ? 0 : 1, 0);
    };
    rejects([&] { pgeqrs<Backend, T>(h, n - 1, n, nrhs, a.data(), da, tau.data(), b.data(), db, info); }, "reject underdetermined dimensions");
    rejects([&] { pgeqrs<Backend, T>(h, m, n, -1, a.data(), da, tau.data(), b.data(), db, info); }, "reject negative nrhs");
    rejects([&] { pgeqrs<Backend, T>(h, m, n, nrhs, a.data(), da, tau.data(), a.data(), da, info); }, "reject overlapping A and B");
    rejects([&] { pgeqrs<Backend, T>(h, m, n, nrhs, a.data(), da,
                 ddlaGetRank(h) == 0 ? nullptr : tau.data(), b.data(), db, info); }, "reject a rank-local null tau");
    if (ddlaGetSize(h) > 1)
        rejects([&] { pgeqrs<Backend, T>(h, m, n, ddlaGetRank(h) == 0 ? 0 : nrhs,
                     a.data(), da, tau.data(), b.data(), db, info); }, "reject inconsistent metadata before quick return");
    int bad[9];
    std::copy(db, db + 9, bad);
    bad[DDLA_MB_] = nb + 1;
    rejects([&] { pgeqrs<Backend, T>(h, m, n, nrhs, a.data(), da, tau.data(), b.data(), bad, info); }, "reject incompatible row distribution");
    for (const auto dimensions : {std::array<int, 3>{m, n, 0}, {m, 0, nrhs}, {0, 0, 0}}) {
        pgeqrs<Backend, T>(h, dimensions[0], dimensions[1], dimensions[2], nullptr, da, nullptr, nullptr, db, info);
        require_close(h, "empty solve status", info == 0 ? 0 : 1, 0);
    }
}

template <DdlaBackend Backend, typename T>
void cases(DdlaHandle_t h)
{
    run_case<Backend, T>(h, 1, 1, 1, 8, false);
    run_case<Backend, T>(h, 13, 5, 3, 3, true);
    run_case<Backend, T>(h, 33, 19, 9, 4, true);
    run_case<Backend, T>(h, 64, 64, 2, 8, false);
}

template <DdlaBackend Backend>
void backend_cases(DdlaHandle_t h)
{
    cases<Backend, float>(h);
    cases<Backend, double>(h);
    cases<Backend, std::complex<float>>(h);
    cases<Backend, std::complex<double>>(h);
}

void body(DdlaHandle_t h, const Shape&)
{
    if (ddlaGetBackend(h) == DdlaBackend::CPU) backend_cases<DdlaBackend::CPU>(h);
#if DDLA_HAS_GPU
    else {
        backend_cases<DdlaBackend::GPU>(h);
        int pr, pc;
        ddlaGetGridDims(h, pr, pc);
        DdlaHandle_t cpu = nullptr;
        ddlaInit(cpu, DdlaBackend::CPU);
        ddlaSet(cpu, ddlaGetCommunicator(h), pr, pc);
        backend_cases<DdlaBackend::CPU>(cpu);
        ddlaDestroy(cpu);
    }
#endif
}

} // namespace

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pgeqrs", body);
}
