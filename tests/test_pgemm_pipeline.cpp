#include "mpi_test_init.h"
#include "pgemm_test_common.h"
#include "workspace_test_common.h"
#include "ddla_stream.h"
#include "comm_traits.h"
#include <ddla/ddla_vector.h>

#include <array>
#include <climits>
#include <string>

namespace {
using namespace ddla;

template<class T> std::complex<long double> extended(T x)
{
    if constexpr (std::is_arithmetic_v<T>) return {x, 0};
    else return {x.real(), x.imag()};
}

template<class T> auto op_entry(char op, int i, int j, int salt)
{
    auto x = extended(op == 'N' ? value<T>(i, j, salt) : value<T>(j, i, salt));
    return op == 'C' ? std::conj(x) : x;
}

void require_all(DdlaHandle_t h, bool okay, const char* message)
{
    int flag = okay ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &flag, 1, MPI_INT, MPI_MIN, h->comm);
    if (!flag) throw std::runtime_error(message);
}

bool no_owners(const workspace_test::Snapshot& s)
{
    return s.ddla_alloc == 0 && s.ddla_free == 0 && s.pinned_alloc == 0 &&
        s.raw_gpu_alloc == 0 && s.host_new == 0 && s.vendor_query == 0 && s.resource_setup == 0;
}

bool pure_query(const workspace_test::Snapshot& s)
{
    return no_owners(s) && s.copies == 0 && s.launches == 0 &&
        s.syncs == 0 && s.collectives == 0;
}

class Scratch {
    static constexpr std::size_t guard = 256;
    DdlaHandle_t h_;
    std::size_t bytes_, host_bytes_;
    ddla::vector<char> backend_;
    std::vector<char> host_;
public:
    Scratch(DdlaHandle_t h, std::size_t bytes, std::size_t host_bytes)
        : h_(h), bytes_(bytes), host_bytes_(host_bytes),
          backend_(h, bytes + 2 * guard), host_(host_bytes + 2 * guard, char(0x5b))
    {
        DDLA_CHECK(ddlaMemset(h, backend_.data(), 0x5b, backend_.size()));
    }
    void* backend() { return backend_.data() + guard; }
    void* host() { return host_.data() + guard; }
    void verify()
    {
        std::array<char, 2 * guard> ends{};
        DDLA_CHECK(ddlaMemcpy(h_, ends.data(), backend_.data(), guard));
        DDLA_CHECK(ddlaMemcpy(h_, ends.data() + guard, backend_.data() + guard + bytes_, guard));
        DDLA_CHECK(ddlaSynchronize(h_));
        bool okay = true;
        for (std::size_t i = 0; i < guard; ++i)
            okay = okay && ends[i] == char(0x5b) && ends[guard + i] == char(0x5b) &&
                host_[i] == char(0x5b) && host_[guard + host_bytes_ + i] == char(0x5b);
        require_all(h_, okay, "SUMMA exact workspace guard changed");
    }
};

template<class T>
void fill(Matrix<T>& x, int m, int n, int salt, bool poison)
{
    for (int j = 0; j < x.nc; ++j) for (int i = 0; i < x.nr; ++i) {
        const int gi = indxl2g(i, x.d[DDLA_MB_], x.r, x.d[DDLA_RSRC_], x.pr);
        const int gj = indxl2g(j, x.d[DDLA_NB_], x.c, x.d[DDLA_CSRC_], x.pc);
        if (gi < m && gj < n)
            x.host[i + std::size_t(j) * x.d[DDLA_LLD_]] = poison
                ? scalar<T>(std::numeric_limits<double>::quiet_NaN()) : value<T>(gi, gj, salt);
    }
    x.reset();
}

// One first run warms the existing vendor/transport caches outside tracing.
// Two changed-input runs then borrow exactly the queried storage. This test
// does not claim allocation-free cold vendor initialization or first staging.
template<DdlaBackend B, class T, class TA = T, class TB = T>
void exercise(DdlaHandle_t h, char ta, char tb, int m, int n, int k,
              int width, bool shifted, int mode, int& count, int kb = 3)
{
    using R = typename real_type<T>::type;
    const int mb = 4, nb = 5;
    const int ar = ta == 'N' ? m : k, ac = ta == 'N' ? k : m;
    const int br = tb == 'N' ? k : n, bc = tb == 'N' ? n : k;
    const int ars = shifted ? h->nprows_ - 1 : 0;
    const int acs = shifted ? h->npcols_ - 1 : 0;
    const int brs = shifted ? 0 : h->nprows_ - 1;
    const int bcs = shifted ? 0 : h->npcols_ - 1;
    const bool zero_alpha = mode >= 2, zero_beta = mode == 1 || mode == 3;
    const bool no_product = zero_alpha || k == 0;
    const T alpha = zero_alpha ? T(0) : scalar<T>(0.7, -0.2);
    const T beta = mode == 4 ? T(1) : zero_beta ? T(0) : scalar<T>(-0.3, 0.15);
    Matrix<TA> a(h, ar, ac, ta == 'N' ? mb : kb, ta == 'N' ? kb : mb, ars, acs, 1, no_product);
    Matrix<TB> b(h, br, bc, tb == 'N' ? kb : nb, tb == 'N' ? nb : kb, brs, bcs, 2, no_product);
    Matrix<T> c(h, m, n, mb, nb, 0, 0, 3, zero_beta);

    int environment_width = 0;
    const auto algorithm = pgemm_impl::pgemm_check<B>(h, ta, tb, m, n, k, zero_alpha,
        a.d, b.d, c.d, Algorithm::AB, environment_width);
    auto query = [&](int target, std::size_t& backend, std::size_t& host) {
        pgemm_impl::pgemm_workspace_size<T, TA, TB>(h, algorithm, target, zero_alpha,
            ta, tb, m, n, k, beta != T(0), a.d, b.d, c.d, &backend, &host);
    };
    std::size_t bytes = 0, host_bytes = 0, narrow_bytes = 0, narrow_host = 0;
    MPI_Barrier(h->comm);
    bool query_okay = true;
    if (!h->myid) {
        workspace_test::Scope scope(true);
        query(width, bytes, host_bytes);
        query_okay = pure_query(scope.finish());
    }
    MPI_Barrier(h->comm);
    require_all(h, query_okay, "SUMMA root-only query has effects");
    workspace_test::Scope scope(true);
    query(width, bytes, host_bytes);
    query(kb, narrow_bytes, narrow_host);
    const auto query_trace = scope.finish();
    require_all(h, pure_query(query_trace), "SUMMA all-rank query has effects");
    std::cout << "SUMMA_WORKSPACE rank=" << h->myid << " backend=" << int(B)
        << " scalar_bytes=" << sizeof(T) << " a_bytes=" << sizeof(TA) << " b_bytes=" << sizeof(TB)
        << " trans=" << ta << tb << " m=" << m << " n=" << n << " k=" << k
        << " target=" << width << " narrow_backend=" << narrow_bytes << " backend_bytes=" << bytes
        << " narrow_host=" << narrow_host << " host_bytes=" << host_bytes
        << " prepared_events_per_gpu_handle=7 event_runtime_bytes=not_queried" << std::endl;
    Scratch scratch(h, bytes, host_bytes);
    for (int pass = 0; pass < 3; ++pass) {
        const int as = 1 + 4 * pass, bs = 2 + 4 * pass, cs = 3 + 4 * pass;
        fill(a, ar, ac, as, no_product);
        fill(b, br, bc, bs, no_product);
        fill(c, m, n, cs, zero_beta);
        const TA* ap = no_product ? nullptr : a.data;
        const TB* bp = no_product ? nullptr : b.data;
        T* cp = mode == 4 ? nullptr : c.data;
        workspace_test::Snapshot trace{};
        if (pass) {
            workspace_test::Scope run_scope;
            pgemm_impl::pgemm_run<B, T, TA, TB>(h, algorithm, width, ta, tb, m, n, k,
                alpha, ap, a.d, bp, b.d, beta, cp, c.d,
                scratch.backend(), bytes, scratch.host(), host_bytes);
            trace = run_scope.finish();
            require_all(h, no_owners(trace), "SUMMA warm borrowed run allocated/queried/setup resources");
        } else {
            pgemm_impl::pgemm_run<B, T, TA, TB>(h, algorithm, width, ta, tb, m, n, k,
                alpha, ap, a.d, bp, b.d, beta, cp, c.d,
                scratch.backend(), bytes, scratch.host(), host_bytes);
        }
        const auto out = c.download();
        const auto aa = a.download();
        const auto bb = b.download();
        bool okay = !std::memcmp(aa.data(), a.host.data(), aa.size() * sizeof(TA)) &&
            !std::memcmp(bb.data(), b.host.data(), bb.size() * sizeof(TB));
        long double worst = 0;
        for (std::size_t p = 0; p < out.size(); ++p) {
            const int i = p % c.d[DDLA_LLD_], j = p / c.d[DDLA_LLD_];
            const int gi = indxl2g(i, mb, c.r, c.d[DDLA_RSRC_], c.pr);
            const int gj = indxl2g(j, nb, c.c, c.d[DDLA_CSRC_], c.pc);
            if (i < c.nr && j < c.nc && gi < m && gj < n) {
                std::complex<long double> reference = 0;
                long double bound = 0;
                if (!no_product) {
                    for (int l = 0; l < k; ++l) {
                        const auto av = op_entry<TA>(ta, gi, l, as);
                        const auto bv = op_entry<TB>(tb, l, gj, bs);
                        reference += av * bv;
                        bound += std::abs(av) * std::abs(bv);
                    }
                    reference *= extended(alpha);
                    bound *= std::abs(extended(alpha));
                }
                if (!zero_beta) {
                    const auto term = extended(beta) * extended(value<T>(gi, gj, cs));
                    reference += term;
                    bound += std::abs(term);
                }
                const long double error = std::abs(extended(out[p]) - reference);
                okay = okay && std::isfinite(error);
                worst = std::max(worst, error / std::max(1.L, bound));
            } else okay = okay && !std::memcmp(&out[p], &c.host[p], sizeof(T));
        }
        MPI_Allreduce(MPI_IN_PLACE, &worst, 1, MPI_LONG_DOUBLE, MPI_MAX, h->comm);
        if (worst > 96 * std::numeric_limits<R>::epsilon() && !h->myid)
            std::cerr << "SUMMA_NUMERICAL_FAILURE trans=" << ta << tb << " shape=" << m << ',' << n << ',' << k
                << " target=" << width << " mode=" << mode << " error=" << double(worst) << std::endl;
        require_all(h, okay && worst <= 96 * std::numeric_limits<R>::epsilon(),
            "SUMMA scalar oracle/input/parent guard failure");
        scratch.verify();
    }
    ++count;
}

template<DdlaBackend B, class T, class TA = T, class TB = T>
void types(DdlaHandle_t h, int& count)
{
    for (char ta : {'N', 'T', 'C'}) for (char tb : {'N', 'T', 'C'})
        exercise<B, T, TA, TB>(h, ta, tb, 17, 13, 29, 12, true, 0, count);
    // Requested 1024-wide NN panels at k<1024 must still respect the rank
    // share cap; 1 x P/P x 1 runs exercise this former full-panel corner.
    exercise<B, T, TA, TB>(h, 'N', 'N', 129, 133, 65, 1024, true, 0, count, 8);
    exercise<B, T, TA, TB>(h, 'C', 'C', 1, 2, 13, 12, true, 1, count);
    exercise<B, T, TA, TB>(h, 'T', 'N', 11, 7, 13, 3, false, 3, count);
    exercise<B, T, TA, TB>(h, 'T', 'C', 11, 7, 13, 12, false, 2, count);
    exercise<B, T, TA, TB>(h, 'C', 'N', 3, 4, 1, 12, true, 0, count);
    exercise<B, T, TA, TB>(h, 'N', 'C', 11, 7, 0, 12, true, 1, count);
    exercise<B, T, TA, TB>(h, 'C', 'C', 11, 7, 13, 12, true, 4, count);
    exercise<B, T, TA, TB>(h, 'N', 'T', 0, 7, 13, 12, false, 0, count);
    // nb=1 and an INT_MAX target previously overflowed stage ceiling and
    // reserved a target-sized panel even though only seven entries exist.
    exercise<B, T, TA, TB>(h, 'C', 'C', 3, 4, 7, INT_MAX, true, 0, count, 1);
}

template<DdlaBackend B>
void group_contract(DdlaHandle_t h)
{
    if constexpr (B == DdlaBackend::GPU) {
        if (h->communication() != DdlaCommunication::CCL) {
            // MPI/staging groups are host markers, not deferred GPU vendor
            // submissions. Keep products supported inside those groups.
            commGroupStart<B>(h, h->stream_data);
            int ignored_count = 0;
            exercise<B, double>(h, 'N', 'N', 17, 13, 29, 12, true, 0, ignored_count);
            commGroupEnd<B>(h, h->stream_data);
            return;
        }
        Matrix<double> a(h, 17, 29, 4, 3, 0, 0, 1, false);
        Matrix<double> b(h, 29, 13, 3, 5, 0, 0, 2, false);
        Matrix<double> c(h, 17, 13, 4, 5, 0, 0, 3, false);
        // A group may be rank-local. The owning checker must still reject
        // every rank before any panel events or copies are submitted.
        for (bool all_ranks : {false, true}) {
            const bool opened = all_ranks || h->myid == 0;
            if (opened) commGroupStart<B>(h, h->stream_data);
            int width = 0;
            const auto no_op = pgemm_impl::pgemm_check<B>(h, 'N', 'N', 17, 13, 29, true,
                a.d, b.d, c.d, Algorithm::AB, width);
            pgemm_impl::pgemm_run<B, double, double, double>(h, no_op, width, 'N', 'N', 17, 13, 29,
                0.0, nullptr, a.d, nullptr, b.d, 1.0, nullptr, c.d, nullptr, 0, nullptr, 0);
            bool caught = false;
            try {
                (void)pgemm_impl::pgemm_check<B>(h, 'N', 'N', 17, 13, 29, false,
                    a.d, b.d, c.d, Algorithm::AB, width);
            } catch (const std::invalid_argument&) { caught = true; }
            bool raw_caught = true;
            if (all_ranks) {
                raw_caught = false;
                try {
                    pgemm_impl::pgemm_run<B, double, double, double>(h, Algorithm::AB, 12, 'N', 'N', 17, 13, 29,
                        1.0, a.data, a.d, b.data, b.d, 0.0, c.data, c.d, nullptr, 0, nullptr, 0);
                } catch (const std::invalid_argument&) { raw_caught = true; }
            }
            if (opened) commGroupEnd<B>(h, h->stream_data);
            require_all(h, caught && raw_caught, "SUMMA active-group rejection/no-op contract failed");
            const auto out = c.download();
            require_all(h, !std::memcmp(out.data(), c.host.data(), out.size() * sizeof(double)),
                "SUMMA active-group rejection changed output");
        }
    }
}

template<DdlaBackend B>
void run(int rows, int columns, const std::string& transport)
{
    DdlaHandle_t h = nullptr;
    DDLA_CHECK(ddlaInit(h, B));
    if (transport != "default") {
        const auto mode = transport == "staged" ? DdlaCommunication::HostStagedMPI :
            transport == "ccl" ? DdlaCommunication::CCL : DdlaCommunication::MPI;
        DDLA_CHECK(ddlaSetCommunication(h, mode));
    }
    DDLA_CHECK(ddlaSet(h, MPI_COMM_WORLD, rows, columns));
    group_contract<B>(h);
    int count = 0;
    types<B, float>(h, count);
    types<B, double>(h, count);
    types<B, std::complex<float>>(h, count);
    types<B, std::complex<double>>(h, count);
    types<B, std::complex<float>, float, std::complex<float>>(h, count);
    types<B, std::complex<float>, std::complex<float>, float>(h, count);
    types<B, std::complex<double>, double, std::complex<double>>(h, count);
    types<B, std::complex<double>, std::complex<double>, double>(h, count);
    if (!h->myid) std::cout << "PGEMM_PIPELINE_PASS backend=" << int(B) << " grid=" << rows << 'x' << columns
        << " transport=" << transport << " cases=" << count
        << " trace=" << workspace_test::available() << " prewarm_native_and_transport=1 no_performance_claim=1" << std::endl;
    DDLA_CHECK(ddlaDestroy(h));
}
} // namespace

int main(int argc, char** argv)
{
    ddla_test_mpi_init(&argc, &argv);
    try {
        int ranks = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &ranks);
        bool gpu = false;
        int rows = 1, columns = ranks;
        std::string transport = "default";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--gpu") gpu = true;
            else if (arg == "--cpu") gpu = false;
            else if (arg == "--grid" && i + 2 < argc) { rows = std::stoi(argv[++i]); columns = std::stoi(argv[++i]); }
            else if (arg == "--transport" && i + 1 < argc) transport = argv[++i];
            else throw std::invalid_argument("usage: test_pgemm_pipeline [--cpu|--gpu] [--grid R C] [--transport default|mpi|staged|ccl]");
        }
        if (rows < 1 || columns < 1 || rows > ranks || columns != ranks / rows || rows * columns != ranks)
            throw std::invalid_argument("process grid does not match ranks");
        if (transport != "default" && transport != "mpi" && transport != "staged" && transport != "ccl")
            throw std::invalid_argument("unknown communication transport");
        if (gpu) {
#if DDLA_HAS_GPU
            run<DdlaBackend::GPU>(rows, columns, transport);
#else
            throw std::invalid_argument("GPU backend unavailable");
#endif
        } else run<DdlaBackend::CPU>(rows, columns, transport);
        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
