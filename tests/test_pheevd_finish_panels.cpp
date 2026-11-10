// Test the private finish operation against an independent dense product.
// Small forced output panels cross ownership/block boundaries and exercise
// receive sizes larger than row slabs without an expensive eigensolve.
#include "api_grid_test_common.h"
#include "Eigensolver/two_stage/hb2st_finish_panel.h"

#include <cstring>
#include <stdexcept>
#include <type_traits>

using namespace api_grid_test;

namespace {

template <typename T> std::complex<double> wide(T v) { return {double(v), 0}; }
template <typename R> std::complex<double> wide(std::complex<R> v) { return {double(v.real()), double(v.imag())}; }
template <typename T> T value(int i, int j)
{
    using rt = typename ddla::real_type<T>::type;
    const rt a = rt((i * 13 + j * 7) % 23 - 11) / rt(23);
    if constexpr (std::is_same<T, rt>::value) return a;
    else return T(a, rt((i * 5 + j * 11) % 19 - 9) / rt(19));
}
template <typename rt> rt yvalue(int i, int j) { return rt((i * 17 + j * 3) % 29 - 14) / rt(29); }

void check_widths()
{
    for (int block : {1, 3, 8}) for (int p : {1, 2, 3, 6})
        for (int w = 1; w <= 3 * block * p; ++w) {
            int actual_max = 0;
            for (int j0 = 0; j0 < block * p; ++j0)
                for (int owner = 0; owner < p; ++owner) {
                    int count = 0;
                    for (int j = j0; j < j0 + w; ++j) count += j / block % p == owner;
                    actual_max = std::max(actual_max, count);
                }
            if (static_cast<std::size_t>(actual_max) != ddla::detail::hb2st_finish_max_owned(w, block, p))
                throw std::runtime_error("panel ownership capacity is not exact");
        }
    for (int n : {0, 2, 67, 10000, 1000000}) for (int nr : {1, 2, 3}) for (int nc : {1, 2, 4})
        for (bool cpu : {false, true}) for (std::size_t ts : {std::size_t(4), std::size_t(8), std::size_t(16)}) {
            const std::size_t rs = ts == 16 ? 8 : ts;
            const int w = ddla::detail::hb2st_finish_panel_width(n, nr, nc, 64, 64, ts, rs, cpu, 512);
            if (w < 1 || (n > 0 && w > n)) throw std::runtime_error("invalid panel width");
            const auto bytes = ddla::detail::hb2st_finish_panel_bytes(n, nr, nc, 64, 64, w, ts, rs, cpu, 512);
            if (w > 1 && bytes > (std::size_t(256) << 20)) throw std::runtime_error("panel exceeds byte target");
        }
}

template <ddla::DdlaBackend Backend, typename T>
void check_product(const ddla::DdlaHandle_t& h, int n, int nb, int pw, bool shifted, bool with_phase)
{
    using rt = typename ddla::real_type<T>::type;
    int npr = 0, npc = 0;
    ddlaGetGridDims(h, npr, npc);
    const int rank = ddlaGetRank(h), P = npr * npc;
    const ddla::detail::RowShare rows(n, P);
    const int nrl = rows.count(rank), ldy = std::max(nrl, 1);
    int desc[ddla::DDLA_DLEN_];
    ddlaDescInit(desc, h, n + 5, n + 7, nb, nb, shifted ? npr - 1 : 0, shifted ? npc - 1 : 0);
    desc[ddla::DDLA_LLD_] += 7;
    const int lld = desc[ddla::DDLA_LLD_], nr = ddla_test::m_loc(h, desc), nc = ddla_test::n_loc(h, desc);
    const T sentinel = T(rt(12345));
    std::vector<T> q(static_cast<std::size_t>(std::max(nrl, 1)) * n, sentinel);
    std::vector<rt> y(static_cast<std::size_t>(ldy) * n, rt(6789));
    for (int j = 0; j < n; ++j) for (int i = 0; i < nrl; ++i) {
        q[i + static_cast<std::size_t>(j) * nrl] = value<T>(rows.lo(rank) + i, j);
        y[i + static_cast<std::size_t>(j) * ldy] = yvalue<rt>(rows.lo(rank) + i, j);
    }
    std::vector<T> phases;
    if constexpr (!std::is_same<T, rt>::value) if (with_phase) {
        phases.resize(std::max(n - 1, 0));
        for (int i = 0; i < n - 1; ++i) phases[i] = T(rt(std::cos(0.1 * i)), rt(std::sin(0.1 * i)));
    }
    std::vector<T> original(std::max(local_size(h, desc), std::size_t(1)), sentinel);
    for (int j = 0; j < nc; ++j) for (int i = 0; i < nr; ++i)
        if (ddla::indx_l2g_c(desc, h, j) < n && ddla::indx_l2g_r(desc, h, i) < n)
            original[i + static_cast<std::size_t>(j) * lld] = T(std::numeric_limits<rt>::quiet_NaN());
    DeviceBuffer<T> Q(h, q.size()), Z(h, original.size());
    DeviceBuffer<rt> Y(h, y.size());
    upload(h, Q.ptr, q); upload(h, Y.ptr, y); upload(h, Z.ptr, original);
    check_ddla_sync(h);
    // The finish panels are a host (CPU stage-4) engine; GPU handles finish
    // through hb2st_replay_gpu, which test_pheevd_gpu_replay covers.
    static_assert(Backend == ddla::DdlaBackend::CPU, "finish panels are host-only");
    {
        const std::function<void(T*, int, int)> pack = [&](T* out, int begin, int count) {
            if (nrl) std::memcpy(out, Q.ptr + static_cast<std::size_t>(begin) * nrl, static_cast<std::size_t>(nrl) * count * sizeof(T));
        };
        ddla::detail::hb2st_finish_rows_host<T>(h, n, Y.ptr, ldy, phases, Z.ptr, desc, pack, 1, pw);
    }
    check_ddla_sync(h);
    const auto result = download(h, Z.ptr, original.size());
    const auto q_after = download(h, Q.ptr, q.size());
    const auto y_after = download(h, Y.ptr, y.size());
    double error = 0, bad = 0, padding = 0;
    for (int j = 0; j < nc; ++j) {
        const int gj = ddla::indx_l2g_c(desc, h, j);
        for (int i = 0; i < lld; ++i) {
            const int gi = i < nr ? ddla::indx_l2g_r(desc, h, i) : n;
            const auto index = i + static_cast<std::size_t>(j) * lld;
            if (gi < n && gj < n) {
                std::complex<double> ref = 0;
                for (int k = 0; k < n; ++k) {
                    const auto phase = k > 0 && !phases.empty() ? wide(phases[k - 1]) : std::complex<double>(1, 0);
                    ref += wide(value<T>(gi, k)) * phase * double(yvalue<rt>(k, gj));
                }
                const auto got = wide(result[index]);
                if (!std::isfinite(got.real()) || !std::isfinite(got.imag())) ++bad;
                else error = std::max(error, std::abs(got - ref));
            } else padding += result[index] != sentinel;
        }
    }
    if (nc == 0) for (T v : result) padding += v != sentinel;
    const std::string tag = "finish n=" + std::to_string(n) + " panel=" + std::to_string(pw) + " bytes=" + std::to_string(sizeof(T)) +
                            " shifted=" + std::to_string(shifted) + " phase=" + std::to_string(with_phase);
    require_close(h, tag + " finite", bad, 0);
    require_close(h, tag + " padding", padding, 0);
    require_close(h, tag + " Q immutable", !q.empty() && std::memcmp(q.data(), q_after.data(), q.size() * sizeof(T)) != 0, 0);
    require_close(h, tag + " Y immutable", !y.empty() && std::memcmp(y.data(), y_after.data(), y.size() * sizeof(rt)) != 0, 0);
    require_close(h, tag + " dense product", error / (std::max(n, 1) * std::numeric_limits<rt>::epsilon()), 50);
}

template <ddla::DdlaBackend Backend> void run_checks(const ddla::DdlaHandle_t& h)
{
    for (bool shifted : {false, true}) for (int pw : {1, 3, 7, 19, 0})
        for (int n : {2, 37}) {
            check_product<Backend, float>(h, n, 4, pw, shifted, false);
            check_product<Backend, double>(h, n, 4, pw, shifted, false);
            check_product<Backend, std::complex<float>>(h, n, 4, pw, shifted, true);
            check_product<Backend, std::complex<double>>(h, n, 4, pw, shifted, true);
        }
    check_product<Backend, std::complex<double>>(h, 37, 4, 3, false, false);
    check_product<Backend, double>(h, 0, 4, 1, false, false);
}
} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    try {
        check_widths();
        int P = 0; MPI_Comm_size(MPI_COMM_WORLD, &P);
        TestOptions options; std::string error;
        if (!parse_options(argc, argv, P, options, error)) throw std::runtime_error(error);
        auto backend = ddla::default_backend_v;
        if (const char* arg = std::getenv("DDLA_TEST_BACKEND")) {
            const std::string value(arg);
            if (value == "CPU" || value == "cpu") backend = ddla::DdlaBackend::CPU;
            else if (value == "GPU" || value == "gpu") backend = ddla::DdlaBackend::GPU;
            else throw std::runtime_error("invalid DDLA_TEST_BACKEND");
        }
        for (const auto grid : options.grids) for (char major : {'R', 'C'}) {
            ddla::DdlaHandle_t h = nullptr;
            ddla::ddlaInit(h, backend); ddla::ddlaSet(h, MPI_COMM_WORLD, grid.first, grid.second, major);
            if (ddlaGetRank(h) == 0) std::cout << "finish grid " << grid_name(h) << " major=" << major << std::endl;
            switch (backend) {
            case ddla::DdlaBackend::CPU: run_checks<ddla::DdlaBackend::CPU>(h); break;
            case ddla::DdlaBackend::GPU:
                if (ddlaGetRank(h) == 0) std::cout << "skip: finish panels are host-only" << std::endl;
                break;
            default: throw std::runtime_error("backend unavailable");
            }
            ddla::ddlaDestroy(h);
        }
        int rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) std::cout << "test_pheevd_finish_panels passed" << std::endl;
    } catch (const std::exception& e) { std::cerr << e.what() << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); }
    MPI_Finalize();
    return 0;
}
