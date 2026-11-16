#include "api_grid_test_common.h"
#include <ddla/ddla_vector.h>
#include <cstring>
#include <type_traits>
#if DDLA_HAS_GPU
#include "gpu/gpu.h"
#endif

using namespace api_grid_test;
namespace {
template<class T> T scalar(double re, double im = 0)
{
    if constexpr (std::is_floating_point_v<T>) return T(re);
    else return T(typename T::value_type(re), typename T::value_type(im));
}
template<class T> T value(int i, int j)
{
    const int row = (i + 7) % 41;
    return scalar<T>(row == j ? 5 : .002 * (row - j), .001 * (row + j));
}
template<class T> T expected(int i, int j) { return scalar<T>(.03 * (i + 1), .02 * (j + 1)); }
template<class T> T op_value(char trans, int i, int j)
{
    const T result = trans == 'N' ? value<T>(i, j) : value<T>(j, i);
    if constexpr (!std::is_floating_point_v<T>) return trans == 'C' ? std::conj(result) : result;
    else return result;
}
template<class T> void cases(const DdlaHandle_t& h)
{
    constexpr int n = 41, nrhs = 3, nb = 4, guard = 11, poison = -731;
    int pr, pc; ddlaGetGridDims(h, pr, pc);
    int da[DDLA_DLEN_]; DDLA_CHECK(ddlaDescInit(da, h, n, n, nb, nb, pr - 1, pc - 1));
    da[DDLA_LLD_] += 3;
    const auto input = make_local<T>(h, da, value<T>);
    DeviceBuffer<T> a(h, input.size());
    const int local_rows = ddla_test::m_loc(h, da);
    std::vector<int> initial(local_rows + 2 * guard, poison);
    ddla::vector<int> pivots(h, initial.size());
    for (const char* policy : {"inplace", "gather"}) {
        setenv("DDLA_LU_PANEL", policy, 1);
        upload(h, a.ptr, input);
        DDLA_CHECK(ddlaMemcpy(h, pivots.data(), initial.data(), initial.size() * sizeof(int)));
        int info = -1;
        pgetrf<T>(h, n, n, a.ptr, 1, 1, da, pivots.data() + guard, info);
        require_close(h, "backend LU info", info, 0);
        const auto factored_pivots = download(h, pivots.data(), pivots.size());
        bool good = true;
        for (int i = 0; i < guard; ++i)
            good = good && factored_pivots[i] == poison && factored_pivots[guard + local_rows + i] == poison;
        for (int i = 0; i < local_rows; ++i)
            good = good && factored_pivots[guard + i] >= indx_l2g_r(da, h, i) + 1 && factored_pivots[guard + i] <= n;
        require_close(h, "backend pivot values and guards", good ? 0 : 1, 0);
        for (char side : {'L', 'R'}) for (char trans : {'N', 'T', 'C'}) {
            const int rows = side == 'L' ? n : nrhs, cols = side == 'L' ? nrhs : n;
            int db[DDLA_DLEN_]; DDLA_CHECK(ddlaDescInit(db, h, rows, cols, nb, nb, pr - 1, pc - 1));
            auto rhs = make_local<T>(h, db, [&](int i, int j) {
                T sum{};
                for (int k = 0; k < n; ++k)
                    sum += side == 'L' ? op_value<T>(trans, i, k) * expected<T>(k, j)
                                       : expected<T>(i, k) * op_value<T>(trans, k, j);
                return sum;
            });
            DeviceBuffer<T> b(h, rhs.size()); upload(h, b.ptr, rhs);
            // No host pivot upload or reconstruction between factorization and solves.
            pgetrs<T>(h, side, trans, n, nrhs, a.ptr, da, pivots.data() + guard, b.ptr, db);
            const auto result = download(h, b.ptr, rhs.size());
            const auto want = make_local<T>(h, db, expected<T>);
            double error = 0;
            for (std::size_t i = 0; i < result.size(); ++i) error = std::max(error, double(std::abs(result[i] - want[i])));
            const bool single = std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>;
            require_close(h, "backend pivot chained solve", error, single ? 2e-4 : 2e-11);
        }
        const auto after = download(h, pivots.data(), pivots.size());
        require_close(h, "solves preserve backend pivots and guards", after == factored_pivots ? 0 : 1, 0);
    }
}
void check(const DdlaHandle_t& h, const Shape&)
{
    const char* env = std::getenv("DDLA_LU_PANEL");
    const bool had_policy = env != nullptr;
    const std::string saved = env ? env : "";
#if DDLA_HAS_GPU
    if (ddlaGetBackend(h) == DdlaBackend::GPU) {
        gpuStream_t stream = nullptr;
        GPU_CHECK(gpuStreamCreateWithFlags(&stream, gpuStreamNonBlocking));
        DDLA_CHECK(ddlaSetStream(h, stream)); // Ownership transfers to the handle.
    }
#endif
    cases<float>(h); cases<double>(h); cases<std::complex<float>>(h); cases<std::complex<double>>(h);
    if (had_policy) setenv("DDLA_LU_PANEL", saved.c_str(), 1);
    else unsetenv("DDLA_LU_PANEL");
}
}
int main(int argc, char** argv) { return run_grid_test(argc, argv, "test_lu_backend_pivots", check); }
