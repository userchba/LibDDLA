#include "api_grid_test_common.h"
#include "ddla_stream.h"
#include "mpi_datatype.h"
#include <cstring>
#include <climits>
#include <type_traits>
using namespace api_grid_test;
extern "C" {
void sormqr_(const char *, const char *, const int *, const int *, const int *, const float *,
             const int *, const float *, float *, const int *, float *, const int *, int *);
void sorgqr_(const int *, const int *, const int *, float *, const int *, const float *, float *,
             const int *, int *);
void dormqr_(const char *, const char *, const int *, const int *, const int *, const double *,
             const int *, const double *, double *, const int *, double *, const int *, int *);
void dorgqr_(const int *, const int *, const int *, double *, const int *, const double *, double *,
             const int *, int *);
void cunmqr_(const char *, const char *, const int *, const int *, const int *,
             const std::complex<float> *, const int *, const std::complex<float> *,
             std::complex<float> *, const int *, std::complex<float> *, const int *, int *);
void cungqr_(const int *, const int *, const int *, std::complex<float> *, const int *,
             const std::complex<float> *, std::complex<float> *, const int *, int *);
void zunmqr_(const char *, const char *, const int *, const int *, const int *,
             const std::complex<double> *, const int *, const std::complex<double> *,
             std::complex<double> *, const int *, std::complex<double> *, const int *, int *);
void zungqr_(const int *, const int *, const int *, std::complex<double> *, const int *,
             const std::complex<double> *, std::complex<double> *, const int *, int *);
}
namespace {
double finite_error(double value) { return std::isfinite(value) ? value : INFINITY; }
void reference(char side, char trans, int m, int n, int k, const float *a, int lda,
               const float *tau, float *c, int ldc) {
    int info = 0, lwork = -1;
    float query;
    sormqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, &query, &lwork, &info);
    if (info)
        throw std::runtime_error("reference query");
    lwork = std::max(1, int(std::real(query)));
    std::vector<float> work(lwork);
    sormqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work.data(), &lwork, &info);
    if (info)
        throw std::runtime_error("reference apply");
}
void reference_q(int m, int n, int k, float *a, int lda, const float *tau) {
    int info = 0, lwork = -1;
    float query;
    sorgqr_(&m, &n, &k, a, &lda, tau, &query, &lwork, &info);
    if (info)
        throw std::runtime_error("reference Q query");
    lwork = std::max(1, int(std::real(query)));
    std::vector<float> work(lwork);
    sorgqr_(&m, &n, &k, a, &lda, tau, work.data(), &lwork, &info);
    if (info)
        throw std::runtime_error("reference Q");
}
void reference(char side, char trans, int m, int n, int k, const double *a, int lda,
               const double *tau, double *c, int ldc) {
    int info = 0, lwork = -1;
    double query;
    dormqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, &query, &lwork, &info);
    if (info)
        throw std::runtime_error("reference query");
    lwork = std::max(1, int(std::real(query)));
    std::vector<double> work(lwork);
    dormqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work.data(), &lwork, &info);
    if (info)
        throw std::runtime_error("reference apply");
}
void reference_q(int m, int n, int k, double *a, int lda, const double *tau) {
    int info = 0, lwork = -1;
    double query;
    dorgqr_(&m, &n, &k, a, &lda, tau, &query, &lwork, &info);
    if (info)
        throw std::runtime_error("reference Q query");
    lwork = std::max(1, int(std::real(query)));
    std::vector<double> work(lwork);
    dorgqr_(&m, &n, &k, a, &lda, tau, work.data(), &lwork, &info);
    if (info)
        throw std::runtime_error("reference Q");
}
void reference(char side, char trans, int m, int n, int k, const std::complex<float> *a, int lda,
               const std::complex<float> *tau, std::complex<float> *c, int ldc) {
    int info = 0, lwork = -1;
    std::complex<float> query;
    cunmqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, &query, &lwork, &info);
    if (info)
        throw std::runtime_error("reference query");
    lwork = std::max(1, int(std::real(query)));
    std::vector<std::complex<float>> work(lwork);
    cunmqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work.data(), &lwork, &info);
    if (info)
        throw std::runtime_error("reference apply");
}
void reference_q(int m, int n, int k, std::complex<float> *a, int lda,
                 const std::complex<float> *tau) {
    int info = 0, lwork = -1;
    std::complex<float> query;
    cungqr_(&m, &n, &k, a, &lda, tau, &query, &lwork, &info);
    if (info)
        throw std::runtime_error("reference Q query");
    lwork = std::max(1, int(std::real(query)));
    std::vector<std::complex<float>> work(lwork);
    cungqr_(&m, &n, &k, a, &lda, tau, work.data(), &lwork, &info);
    if (info)
        throw std::runtime_error("reference Q");
}
void reference(char side, char trans, int m, int n, int k, const std::complex<double> *a, int lda,
               const std::complex<double> *tau, std::complex<double> *c, int ldc) {
    int info = 0, lwork = -1;
    std::complex<double> query;
    zunmqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, &query, &lwork, &info);
    if (info)
        throw std::runtime_error("reference query");
    lwork = std::max(1, int(std::real(query)));
    std::vector<std::complex<double>> work(lwork);
    zunmqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work.data(), &lwork, &info);
    if (info)
        throw std::runtime_error("reference apply");
}
void reference_q(int m, int n, int k, std::complex<double> *a, int lda,
                 const std::complex<double> *tau) {
    int info = 0, lwork = -1;
    std::complex<double> query;
    zungqr_(&m, &n, &k, a, &lda, tau, &query, &lwork, &info);
    if (info)
        throw std::runtime_error("reference Q query");
    lwork = std::max(1, int(std::real(query)));
    std::vector<std::complex<double>> work(lwork);
    zungqr_(&m, &n, &k, a, &lda, tau, work.data(), &lwork, &info);
    if (info)
        throw std::runtime_error("reference Q");
}
template <class T> T value(int i, int j) {
    if constexpr (std::is_arithmetic_v<T>)
        return T(std::sin(1 + i * .71 + j * .29) + (i == j ? 3 : 0));
    else
        return T(std::sin(1 + i * .71 + j * .29) + (i == j ? 3 : 0),
                 .2 * std::cos(i * .19 + j * .31));
}
template <DdlaBackend B, class T>
void test(DdlaHandle_t h, bool distributed, int order, int k, int qcols) {
    int da[DDLA_DLEN_];
    ddlaDescInit(da, h, order + 3, order + 3, 3, 4, h->nprows_ - 1, h->npcols_ - 1);
    da[DDLA_LLD_] += 2;
    const int lda = distributed ? da[DDLA_LLD_] : order + 3;
    auto a = distributed ? make_local<T>(h, da, [](int i, int j) { return value<T>(i, j); })
                         : std::vector<T>(std::size_t(lda) * (order + 3), T(137));
    if (!distributed)
        for (int j = 0; j < order; ++j)
            for (int i = 0; i < order; ++i)
                a[i + std::size_t(j) * lda] = value<T>(i, j);
    DeviceBuffer<T> A(h, a.size());
    upload(h, A.ptr, a);
    const int tcount =
        distributed ? num_loc(k, da[DDLA_NB_], h->mypcol_, da[DDLA_CSRC_], h->npcols_) : k;
    std::vector<T> tau(std::max(1, tcount), T(0));
    DeviceBuffer<T> dtau(h, std::max(1, k));
    int info = -1;
    if (k) {
        if (distributed)
            pgeqrf<T>(h, order, k, A.ptr, da, tau.data(), info);
        else {
            geqrf<T>(h, 'U', order, k, A.ptr, lda, dtau.ptr);
            tau = download(h, dtau.ptr, k);
        }
    }
    std::vector<T> full_tau(std::max(1, k), T(0)),
        full_a(std::size_t(order) * std::max(1, k), T(0));
    if (distributed) {
        for (int j = 0; j < k; ++j)
            if (indxg2p(j, da[DDLA_NB_], da[DDLA_CSRC_], h->npcols_) == h->mypcol_)
                full_tau[j] = tau[indxg2l(j, da[DDLA_NB_], h->npcols_)];
        MPI_Allreduce(MPI_IN_PLACE, full_tau.data(), k, ddla::detail::mpi_datatype<T>(), MPI_SUM,
                      h->row_comm);
        if (k) {
            DeviceBuffer<T> root(h, h->myid == 0 ? full_a.size() : 1);
            gather<T>(h, order, k, A.ptr, 1, 1, da, 0, root.ptr, order);
            if (!h->myid)
                full_a = download(h, root.ptr, full_a.size());
            MPI_Bcast(full_a.data(), int(full_a.size()), ddla::detail::mpi_datatype<T>(), 0,
                      h->comm);
        }
    } else {
        full_tau = tau;
        auto host = download(h, A.ptr, a.size());
        for (int j = 0; j < k; ++j)
            for (int i = 0; i < order; ++i)
                full_a[i + std::size_t(j) * order] = host[i + std::size_t(j) * lda];
    }
    // R, including unit-anchor slots, must be unread by the new routines.
    a = download(h, A.ptr, a.size());
    const int ac = distributed ? ddla_test::n_loc(h, da) : order + 3;
    for (int j = 0; j < ac; ++j)
        for (int i = 0; i < lda; ++i) {
            const int gi = distributed ? indx_l2g_r(da, h, i) : i,
                      gj = distributed ? indx_l2g_c(da, h, j) : j;
            if (gi < order && gj < k && gi <= gj && (!distributed || i < ddla_test::m_loc(h, da)))
                a[i + std::size_t(j) * lda] = T(NAN);
        }
    upload(h, A.ptr, a);
    for (char side : {'L', 'R'})
        for (char trans : std::string(std::is_arithmetic_v<T> ? "NTC" : "NC")) {
            const int m = side == 'L' ? order : 7, n = side == 'L' ? 7 : order;
            int dc[DDLA_DLEN_];
            ddlaDescInit(dc, h, m + 2, n + 3, 4, 5, 0, 0);
            dc[DDLA_LLD_] += 3;
            const int ldc = distributed ? dc[DDLA_LLD_] : m + 3;
            auto c = distributed
                         ? make_local<T>(h, dc, [](int i, int j) { return value<T>(i + 1, j + 2); })
                         : std::vector<T>(std::size_t(ldc) * (n + 2), T(139));
            if (!distributed)
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i)
                        c[i + std::size_t(j) * ldc] = value<T>(i + 1, j + 2);
            const auto before = c;
            DeviceBuffer<T> C(h, c.size());
            upload(h, C.ptr, c);
            std::vector<T> expected(std::size_t(m) * n);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    expected[i + std::size_t(j) * m] = value<T>(i + 1, j + 2);
            const char tr = std::is_arithmetic_v<T> && trans == 'C' ? 'T' : trans;
            reference(side, tr, m, n, k, full_a.data(), order, full_tau.data(), expected.data(), m);
            if (distributed)
                pormqr<T>(h, side, trans, m, n, k, A.ptr, da, tau.data(), C.ptr, dc, info);
            else
                ormqr<T>(h, side, trans, m, n, k, A.ptr, lda, dtau.ptr, C.ptr, ldc, info);
            require_close(h, "QR apply info", info, 0);
            auto got = download(h, C.ptr, c.size());
            double error = 0;
            bool guards = true;
            const int cols = distributed ? ddla_test::n_loc(h, dc) : n + 2;
            for (int j = 0; j < cols; ++j)
                for (int i = 0; i < ldc; ++i) {
                    const int gi = distributed ? indx_l2g_r(dc, h, i) : i,
                              gj = distributed ? indx_l2g_c(dc, h, j) : j;
                    const auto at = i + std::size_t(j) * ldc;
                    if (gi < m && gj < n && (!distributed || i < ddla_test::m_loc(h, dc)))
                        error =
                            std::max(error, finite_error(double(std::abs(
                                                got[at] - expected[gi + std::size_t(gj) * m]))));
                    else
                        guards = guards && !std::memcmp(&got[at], &before[at], sizeof(T));
                }
            require_close(h, "ORMQR against LAPACK", error,
                          500 * order *
                              std::numeric_limits<typename real_type<T>::type>::epsilon());
            require_close(h, "ORMQR parent/padding guards", guards ? 0 : 1, 0);
            require_close(
                h, "ORMQR readonly factor guards",
                std::memcmp(a.data(), download(h, A.ptr, a.size()).data(), a.size() * sizeof(T))
                    ? 1
                    : 0,
                0);
        }
    std::vector<T> expected(std::size_t(order) * qcols, T(0));
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < order; ++i)
            expected[i + std::size_t(j) * order] = full_a[i + std::size_t(j) * order];
    reference_q(order, qcols, k, expected.data(), order, full_tau.data());
    if (distributed)
        porgqr<T>(h, order, qcols, k, A.ptr, da, tau.data(), info);
    else
        orgqr<T>(h, order, qcols, k, A.ptr, lda, dtau.ptr, info);
    require_close(h, "ORGQR info", info, 0);
    auto got = download(h, A.ptr, a.size());
    double error = 0;
    bool guards = true;
    for (int j = 0; j < ac; ++j)
        for (int i = 0; i < lda; ++i) {
            const int gi = distributed ? indx_l2g_r(da, h, i) : i,
                      gj = distributed ? indx_l2g_c(da, h, j) : j;
            const auto at = i + std::size_t(j) * lda;
            if (gi < order && gj < qcols && (!distributed || i < ddla_test::m_loc(h, da)))
                error = std::max(error, finite_error(double(std::abs(
                                            got[at] - expected[gi + std::size_t(gj) * order]))));
            else
                guards = guards && !std::memcmp(&got[at], &a[at], sizeof(T));
        }
    require_close(h, "ORGQR against LAPACK", error,
                  500 * order * std::numeric_limits<typename real_type<T>::type>::epsilon());
    require_close(h, "ORGQR parent/padding guards", guards ? 0 : 1, 0);
}
template <DdlaBackend B, class T> void cases(DdlaHandle_t h) {
    for (bool distributed : {false, true}) {
        test<B, T>(h, distributed, 43, 37, 43);
        test<B, T>(h, distributed, 13, 7, 9);
        test<B, T>(h, distributed, 3, 1, 2);
        test<B, T>(h, distributed, 3, 0, 2);
    }
    int info = -1;
    ormqr<T>(h, 'L', 'N', 0, 3, 0, nullptr, 1, nullptr, nullptr, 1, info);
    require_close(h, "empty apply", info, 0);
    orgqr<T>(h, 0, 0, 0, nullptr, 1, nullptr, info);
    require_close(h, "empty generate", info, 0);
    bool caught = false;
    try {
        ormqr<T>(h, 'X', 'N', 0, 0, 0, nullptr, 1, nullptr, nullptr, 1, info);
    } catch (const std::invalid_argument &) {
        caught = true;
    }
    require_close(h, "invalid side", caught ? 0 : 1, 0);
    caught = false;
    try {
        ormqr<T>(h, 'L', 'N', 1, INT_MAX, 1, nullptr, 1, nullptr, nullptr, 1, info);
    } catch (const std::overflow_error &) {
        caught = true;
    }
    require_close(h, "product collective overflow rejected before data access", caught ? 0 : 1, 0);
}
template <DdlaBackend B> void run(int pr, int pc, char major) {
    DdlaHandle_t h = nullptr;
    DDLA_CHECK(ddlaInit(h, B));
    DDLA_CHECK(ddlaSet(h, MPI_COMM_WORLD, pr, pc, major));
    cases<B, float>(h);
    cases<B, double>(h);
    cases<B, std::complex<float>>(h);
    cases<B, std::complex<double>>(h);
    DDLA_CHECK(ddlaDestroy(h));
}
} // namespace
int main(int argc, char **argv) {
    ddla_test_mpi_init(&argc, &argv);
    int ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    TestOptions options;
    std::string error;
    if (!parse_options(argc, argv, ranks, options, error))
        MPI_Abort(MPI_COMM_WORLD, 2);
    for (auto grid : options.grids)
        for (char major : {'R', 'C'}) {
            run<DdlaBackend::CPU>(grid.first, grid.second, major);
#if DDLA_HAS_GPU
            run<DdlaBackend::GPU>(grid.first, grid.second, major);
#endif
        }
    MPI_Finalize();
    return 0;
}
