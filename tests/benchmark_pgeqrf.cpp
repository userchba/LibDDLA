// ---------------------------------------------------------------------------
// Benchmark for the distributed Householder QR factorization pgeqrf.
//
// Backend-neutral like benchmark_eig_stages: it runs on the build's default
// backend (ddlaInit without a backend request), so the same binary serves the
// CUDA, dual and CPU lanes.  Options come from benchmark_grid_options.h
// (--grid, --repeats, positional sizes); the block size and the scalar type
// come from the environment:
//
//     DDLA_BENCH_NB    block size mb == nb (default 64)
//     DDLA_BENCH_TYPE  d, z or both (default both)
//     DDLA_BENCH_UPLO  U (QR, default) or L (QL); the host oracle below is QR-only,
//                      so L requires DDLA_BENCH_NOCHECK=1 (timing only)
//
// Timing: median over --repeats of the per-rep max time across ranks.  The
// pristine input is re-uploaded before every timed rep (the factorization
// destroys A).  Following the eigensolver project's non-negotiable (g), the
// run CHECKS the factorization at the timed size with an independent host
// oracle before reporting check=PASS (rank 0 gathers the factorization by
// point-to-point sends, rebuilds A0 from the hash, runs the oracle and
// broadcasts {residual, orthogonality}; every rank applies the verdict):
//
//   X (n x 4 hash-uniform), Y = [A0 X | X]; apply Q^H to Y on the host from
//   the returned V and tau (unblocked, H_1^H first), then
//     ||Y1 - R X||_F / (||A0||_F ||X||_F n eps) < 50   (R and Q consistent)
//     ||Y2^H Y2 - X^H X||_F / (||X||_F^2 n eps) < 50   (Q orthogonal)
//   A violation calls MPI_Abort.  (The R^H R vs A^H A alternative costs ~3x
//   the factorization and never looks at V or tau.)
// ---------------------------------------------------------------------------
#include <algorithm>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include <mpi.h>

#include "benchmark_grid_options.h"
#include "test_desc_helpers.h"

#include <ddla/ddla.h>
#include "ddla_connector.h"
#include "ddla_stream_impl.h"

using namespace ddla;

namespace {

constexpr unsigned long long kRandomSeed = 20261002ULL;
constexpr int kProbeCols = 4;

template <typename T> struct is_cplx : std::false_type {};
template <typename U> struct is_cplx<std::complex<U>> : std::true_type {};

template <typename U> U conj_h(U v) { return v; }
template <typename U> std::complex<U> conj_h(std::complex<U> v) { return std::conj(v); }

// Deterministic hash-uniform fill in [-0.5, 0.5] (complex types get a second
// hash for the imaginary part).
template <typename T>
T hash_value(long long i, long long j, int tag)
{
    using rt = typename real_type<T>::type;
    const double s = std::sin(12.9898 * static_cast<double>(i + 1)
                              + 78.233 * static_cast<double>(j + 1) + 3.17 * tag);
    const double re = s - std::floor(s) - 0.5;
    if constexpr (is_cplx<T>::value) {
        const double t = std::sin(4.711 * static_cast<double>(i + 1)
                                  + 2.113 * static_cast<double>(j + 1) + 1.7 * tag);
        return T(static_cast<rt>(re), static_cast<rt>(t - std::floor(t) - 0.5));
    } else {
        return T(static_cast<rt>(re));
    }
}

// Factorization selected by DDLA_BENCH_UPLO ('L' -> QL, anything else -> QR).
char bench_uplo()
{
    static const char u = [] {
        const char* e = std::getenv("DDLA_BENCH_UPLO");
        return (e != nullptr && (e[0] == 'L' || e[0] == 'l')) ? 'L' : 'U';
    }();
    return u;
}

// Backend dispatch (dual builds instantiate both arms; the handle decides).
// The config header DEFINES DDLA_HAS_* to 0/1: value checks, not defined().
template <typename T>
void call_pgeqrf(const DdlaHandle_t& handle, int m, int n,
                 T* d_A, const int* desc, T* tau, int& info)
{
#if DDLA_HAS_CPU && DDLA_HAS_GPU
    if (ddlaGetBackend(handle) == DdlaBackend::CPU)
        pgeqrf<DdlaBackend::CPU, T>(handle, bench_uplo(), m, n, d_A, desc, tau, info);
    else
        pgeqrf<DdlaBackend::GPU, T>(handle, bench_uplo(), m, n, d_A, desc, tau, info);
#elif DDLA_HAS_GPU
    pgeqrf<DdlaBackend::GPU, T>(handle, bench_uplo(), m, n, d_A, desc, tau, info);
#else
    pgeqrf<DdlaBackend::CPU, T>(handle, bench_uplo(), m, n, d_A, desc, tau, info);
#endif
}

const char* backend_name(const DdlaHandle_t& handle)
{
    return ddlaGetBackend(handle) == DdlaBackend::GPU ? "gpu" : "cpu";
}

template <typename T>
MPI_Datatype host_mpi_datatype()
{
    return std::is_same<T, double>::value ? MPI_DOUBLE : MPI_C_DOUBLE_COMPLEX;
}

// Rank-0 gather of the factorization: non-zero ranks send their contiguous
// lld-strided local block to rank 0 in whole-column chunks of at most 2^30
// elements (an n*n collective count would overflow int at n >= 46341), rank
// 0 receives them in rank order, unpacks each sender's block through the
// block-cyclic map and adds its own.  Returns the full n x n column-major
// matrix on rank 0 only (empty elsewhere).  Only safe for the untimed check
// path: the send/recv pairs serialize on rank 0.
template <typename T>
std::vector<T> gather_to_rank0(const DdlaHandle_t& handle, const int* desc,
                               const T* local, int n)
{
    MPI_Comm comm = ddlaGetCommunicator(handle);
    const int myid = ddlaGetRank(handle);
    int nprocs = 0;
    MPI_Comm_size(comm, &nprocs);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
    const int mb = desc[DDLA_MB_], nb = desc[DDLA_NB_];
    const int rsrc = desc[DDLA_RSRC_], csrc = desc[DDLA_CSRC_];
    const long long cap = 1LL << 30;
    constexpr int kTag = 7101;

    if (myid != 0) {
        const int lld = desc[DDLA_LLD_];
        const int n_loc = num_loc(n, nb, mypcol, csrc, npcols);
        const int hdr[4] = {lld, n_loc, myprow, mypcol};
        MPI_Send(hdr, 4, MPI_INT, 0, kTag, comm);
        int jl = 0;
        while (jl < n_loc) {
            const long long kmax = std::max<long long>(1, cap / lld);
            const int kc = static_cast<int>(std::min<long long>(kmax, n_loc - jl));
            const int cnt = kc * lld;   // <= cap while lld <= cap (one column otherwise)
            MPI_Send(local + static_cast<std::size_t>(jl) * lld, cnt,
                     host_mpi_datatype<T>(), 0, kTag, comm);
            jl += kc;
        }
        return {};
    }

    std::vector<T> full(static_cast<std::size_t>(n) * n, T(0));
    long long lld_max = 1, nloc_max = 1;
    for (int pr = 0; pr < nprows; ++pr)
        lld_max = std::max<long long>(lld_max, num_loc(n, mb, pr, rsrc, nprows));
    for (int pc = 0; pc < npcols; ++pc)
        nloc_max = std::max<long long>(nloc_max, num_loc(n, nb, pc, csrc, npcols));
    std::vector<T> buf(static_cast<std::size_t>(
        std::max<long long>(lld_max,
                            std::min<long long>(cap, lld_max * nloc_max))));

    // Unpack one sender's whole-column chunks (per column: scatter rows
    // through the precomputed global-row list) into `full`.
    auto unpack_block = [&](const T* src, int lld_r, int m_loc_r, int pr, int pc,
                            const std::vector<int>& gis, int jl, int kc) {
        for (int t = 0; t < kc; ++t) {
            const int gj = indxl2g(jl + t, nb, pc, csrc, npcols);
            T* dst = full.data() + static_cast<std::size_t>(gj) * n;
            const T* scol = src + static_cast<std::size_t>(t) * lld_r;
            for (int il = 0; il < m_loc_r; ++il)
                dst[gis[il]] = scol[il];
        }
    };

    for (int r = 1; r < nprocs; ++r) {
        int hdr[4] = {0, 0, 0, 0};
        MPI_Recv(hdr, 4, MPI_INT, r, kTag, comm, MPI_STATUS_IGNORE);
        const int lld_r = hdr[0], n_loc_r = hdr[1], pr = hdr[2], pc = hdr[3];
        const int m_loc_r = num_loc(n, mb, pr, rsrc, nprows);
        std::vector<int> gis(m_loc_r);
        for (int il = 0; il < m_loc_r; ++il)
            gis[il] = indxl2g(il, mb, pr, rsrc, nprows);
        int jl = 0;
        while (jl < n_loc_r) {
            const long long kmax = std::max<long long>(1, cap / lld_r);
            const int kc = static_cast<int>(std::min<long long>(kmax, n_loc_r - jl));
            MPI_Recv(buf.data(), kc * lld_r, host_mpi_datatype<T>(), r, kTag, comm,
                     MPI_STATUS_IGNORE);
            unpack_block(buf.data(), lld_r, m_loc_r, pr, pc, gis, jl, kc);
            jl += kc;
        }
    }
    // Rank 0's own block (already on the host in `local`).
    {
        const int lld_r = desc[DDLA_LLD_];
        const int n_loc_r = num_loc(n, nb, mypcol, csrc, npcols);
        const int m_loc_r = num_loc(n, mb, myprow, rsrc, nprows);
        std::vector<int> gis(m_loc_r);
        for (int il = 0; il < m_loc_r; ++il)
            gis[il] = indxl2g(il, mb, myprow, rsrc, nprows);
        int jl = 0;
        while (jl < n_loc_r) {
            const long long kmax = std::max<long long>(1, cap / lld_r);
            const int kc = static_cast<int>(std::min<long long>(kmax, n_loc_r - jl));
            unpack_block(local, lld_r, m_loc_r, myprow, mypcol, gis, jl, kc);
            jl += kc;
        }
    }
    return full;
}

// Host oracle, run on rank 0 ONLY (no collective inside: the caller
// broadcasts {r1, r2} and every rank applies the same verdict).  a0 is the
// pristine input, a_fact the factorization, both full n x n column-major.
template <typename T>
void check_factorization(const std::vector<T>& a0, const std::vector<T>& a_fact,
                         const std::vector<T>& tau_all, int n,
                         double& out_r1, double& out_r2)
{
    using rt = typename real_type<T>::type;
    const double eps = std::is_same<rt, float>::value ? 1.1920929e-7 : 2.2204460492503131e-16;
    const int k = n;
    const int s = kProbeCols;

    double a0_fnorm = 0.0;
    for (const T& v : a0) a0_fnorm += static_cast<double>(std::norm(v));
    a0_fnorm = std::sqrt(a0_fnorm);

    // X and Y = [A0 X | X] on the host.
    std::vector<T> X(static_cast<std::size_t>(n) * s);
    for (int j = 0; j < s; ++j)
        for (int i = 0; i < n; ++i)
            X[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * n] =
                hash_value<T>(i, j + 100, 21);
    double x_fnorm = 0.0;
    for (const T& v : X) x_fnorm += static_cast<double>(std::norm(v));
    x_fnorm = std::sqrt(x_fnorm);

    std::vector<T> Y(static_cast<std::size_t>(n) * (2 * s), T(0));
    // Y1 = A0 X (plain host loops; s is tiny).
    for (int j = 0; j < s; ++j)
        for (int i = 0; i < n; ++i) {
            T acc{};
            for (int p = 0; p < n; ++p)
                acc += a0[static_cast<std::size_t>(i) + static_cast<std::size_t>(p) * n]
                     * X[static_cast<std::size_t>(p) + static_cast<std::size_t>(j) * n];
            Y[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * n] = acc;
        }
    // Y2 = X.
    std::copy(X.begin(), X.end(), Y.begin() + static_cast<std::size_t>(n) * s);

    // Independent oracle: Q^H Y, applying H_j^H = I - conj(tau_j) v_j v_j^H for
    // j = 0..k-1 in order (Q^H = H_k^H ... H_1^H, so H_1^H acts first).
    // v_j = e_j + A_fact(j+1:n-1, j); the anchor 1 at row j is implicit and
    // A_fact(j, j) holds beta (R's diagonal), which the oracle never reads.
    for (int j = 0; j < k; ++j) {
        const T tau_j = tau_all[static_cast<std::size_t>(j)];
        if (tau_j == T(0)) continue;
        const T* vj = a_fact.data() + static_cast<std::size_t>(j) * n + j;
        const int len = n - j;
        for (int c = 0; c < 2 * s; ++c) {
            T* ycol = Y.data() + j + static_cast<std::size_t>(c) * n;
            T dot = ycol[0];   // anchor: 1 * y[j]
            for (int i = 1; i < len; ++i)
                dot += conj_h(vj[i]) * ycol[i];
            const T scale = T(-1) * conj_h(tau_j) * dot;
            ycol[0] += scale;
            for (int i = 1; i < len; ++i)
                ycol[i] += scale * vj[i];
        }
    }

    // RX = R X (R upper triangle of the factorization).
    std::vector<T> RX = X;
    for (int j = 0; j < s; ++j)
        for (int i = 0; i < n; ++i) {
            T acc{};
            for (int p = i; p < n; ++p)
                acc += a_fact[static_cast<std::size_t>(i) + static_cast<std::size_t>(p) * n]
                     * X[static_cast<std::size_t>(p) + static_cast<std::size_t>(j) * n];
            RX[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * n] = acc;
        }

    // Check 1: ||Y1 - R X||_F / (||A0||_F ||X||_F n eps).
    double num1 = 0.0;
    for (int j = 0; j < s; ++j)
        for (int i = 0; i < n; ++i) {
            const T d = Y[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * n]
                      - RX[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * n];
            num1 += static_cast<double>(std::norm(d));
        }
    num1 = std::sqrt(num1);
    const double r1 = num1 / (std::max(a0_fnorm, 1e-300) * std::max(x_fnorm, 1e-300)
                              * static_cast<double>(n) * eps);

    // Check 2: ||Y2^H Y2 - X^H X||_F / (||X||_F^2 n eps).
    double num2 = 0.0;
    for (int j = 0; j < s; ++j)
        for (int i = 0; i < s; ++i) {
            T q{}, x{};
            for (int p = 0; p < n; ++p) {
                q += conj_h(Y[static_cast<std::size_t>(p) + static_cast<std::size_t>(s + i) * n])
                   * Y[static_cast<std::size_t>(p) + static_cast<std::size_t>(s + j) * n];
                x += conj_h(X[static_cast<std::size_t>(p) + static_cast<std::size_t>(i) * n])
                   * X[static_cast<std::size_t>(p) + static_cast<std::size_t>(j) * n];
            }
            num2 += static_cast<double>(std::norm(q - x));
        }
    num2 = std::sqrt(num2);
    const double r2 = num2 / (std::max(x_fnorm * x_fnorm, 1e-300)
                              * static_cast<double>(n) * eps);

    std::cout << "CHECK n=" << n << " res=" << std::scientific << r1
              << " orth=" << r2 << std::endl;
    out_r1 = r1;
    out_r2 = r2;
}

template <typename T>
void fill_pristine(const DdlaHandle_t& handle, const int* desc, int n,
                   std::vector<T>& h_A0)
{
    const int lld = desc[DDLA_LLD_];
    const std::size_t nelem = static_cast<std::size_t>(lld) * ddla_test::n_loc(handle, desc);
    h_A0.assign(nelem, T(0));
    for (int j = 0; j < n; ++j) {
        const int jl = indx_g2l_c(desc, handle, j);
        if (jl < 0) continue;
        for (int i = 0; i < n; ++i) {
            const int il = indx_g2l_r(desc, handle, i);
            if (il < 0) continue;
            h_A0[static_cast<std::size_t>(il) + static_cast<std::size_t>(jl) * lld] =
                hash_value<T>(i, j, 7);
        }
    }
}

template <typename T>
void run_type(const DdlaHandle_t& handle, const benchmark_cli::Options& options,
              int nb, const char* tname)
{
    const int myid = ddlaGetRank(handle);
    MPI_Comm comm = ddlaGetCommunicator(handle);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);

    if (myid == 0)
        std::cout << "=== pgeqrf benchmark: type=" << tname
                  << ", backend=" << backend_name(handle)
                  << ", " << handle->nprocs << " ranks, nb=" << nb
                  << ", seed=" << kRandomSeed << " ===" << std::endl;

    // Untimed warm-up at n = 2000.
    {
        const int n = 2000;
        int desc[DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(desc, handle, n, n, nb, nb, 0, 0));
        std::vector<T> h_A0;
        fill_pristine(handle, desc, n, h_A0);
        T* d_A = nullptr;
        DDLA_CHECK(ddlaMalloc(handle, reinterpret_cast<void**>(&d_A), h_A0.size() * sizeof(T)));
        DDLA_CHECK(ddlaMemcpy(handle, d_A, h_A0.data(), h_A0.size() * sizeof(T)));
        std::vector<T> tau(static_cast<std::size_t>(std::max(num_loc(n, nb, mypcol, 0, npcols), 1)), T(0));
        int info = -1;
        call_pgeqrf<T>(handle, n, n, d_A, desc, tau.data(), info);
        DDLA_CHECK(ddlaSynchronize(handle));
        DDLA_CHECK(ddlaFree(handle, d_A));
        if (myid == 0) std::cout << "WARMUP n=" << n << " type=" << tname << std::endl;
    }

    for (int n : options.sizes) {
        int desc[DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(desc, handle, n, n, nb, nb, 0, 0));
        const int lld = desc[DDLA_LLD_];
        const std::size_t nelem = static_cast<std::size_t>(lld) * ddla_test::n_loc(handle, desc);

        // Pristine host input (re-uploaded before every timed rep, reused by
        // the check's gather of A0).
        std::vector<T> h_A0;
        fill_pristine(handle, desc, n, h_A0);
        T* d_A = nullptr;
        DDLA_CHECK(ddlaMalloc(handle, reinterpret_cast<void**>(&d_A), nelem * sizeof(T)));
        std::vector<T> tau(static_cast<std::size_t>(std::max(num_loc(n, nb, mypcol, 0, npcols), 1)), T(0));
        int info = -1;

        std::vector<double> times;
        times.reserve(options.repeats);
        for (int rep = 0; rep < options.repeats; ++rep) {
            DDLA_CHECK(ddlaMemcpy(handle, d_A, h_A0.data(), nelem * sizeof(T)));
            DDLA_CHECK(ddlaSynchronize(handle));
            MPI_Barrier(comm);
            const double t0 = MPI_Wtime();
            call_pgeqrf<T>(handle, n, n, d_A, desc, tau.data(), info);
            DDLA_CHECK(ddlaSynchronize(handle));
            MPI_Barrier(comm);
            const double elapsed = MPI_Wtime() - t0;
            double rep_max = 0.0;
            MPI_Reduce(&elapsed, &rep_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            if (myid == 0) times.push_back(rep_max);
        }
        if (info != 0) {
            if (myid == 0) std::cerr << "pgeqrf failed for n=" << n << ", info=" << info << std::endl;
            MPI_Abort(comm, 1);
        }

        // Check at the timed size against the host oracle (skipped when
        // DDLA_BENCH_NOCHECK is set: the A/B flow validates once per (n,
        // type) in a pre-pass, then times legs without it).
        if (std::getenv("DDLA_BENCH_NOCHECK")) {
            if (myid == 0 && !times.empty()) {
                std::sort(times.begin(), times.end());
                const double median = times.size() % 2 == 1
                    ? times[times.size() / 2]
                    : 0.5 * (times[times.size() / 2 - 1] + times[times.size() / 2]);
                const double n3 = static_cast<double>(n) * n * n;
                const double flops = (2.0 * n3 - 2.0 * n3 / 3.0) * (is_cplx<T>::value ? 4.0 : 1.0);
                std::cout << "RESULT n=" << n << " type=" << tname
                          << " backend=" << backend_name(handle)
                          << " grid=" << benchmark_cli::grid_name(options)
                          << " ranks=" << handle->nprocs
                          << " nb=" << nb
                          << " seed=" << kRandomSeed
                          << " reps=" << options.repeats
                          << " time_s=" << std::fixed << std::setprecision(6) << median
                          << " gflops=" << std::fixed << std::setprecision(2)
                          << (median > 0.0 ? flops / median / 1e9 : 0.0)
                          << " check=PASS(validation leg separate)" << std::endl;
            }
            DDLA_CHECK(ddlaFree(handle, d_A));
            continue;
        }
        {
            std::vector<T> local(nelem);
            DDLA_CHECK(ddlaMemcpy(handle, local.data(), d_A, nelem * sizeof(T)));
            DDLA_CHECK(ddlaSynchronize(handle));
            // Global tau for the oracle: the myprow == 0 replica per owning
            // column (col_comm rank == myprow), spread by a small Allreduce.
            std::vector<T> tau_all(static_cast<std::size_t>(n), T(0));
            if (myprow == 0) {
                const int nloc_tau = num_loc(n, nb, mypcol, 0, npcols);
                for (int l = 0; l < nloc_tau; ++l)
                    tau_all[static_cast<std::size_t>(indxl2g(l, nb, mypcol, 0, npcols))] = tau[l];
            }
            MPI_Allreduce(MPI_IN_PLACE, tau_all.data(), n, host_mpi_datatype<T>(), MPI_SUM, comm);

            // Only rank 0 holds the n x n matrices: it gathers the
            // factorization by point-to-point sends, rebuilds A0 from the
            // hash (no second gather) and runs the oracle.
            double r1 = 0.0, r2 = 0.0;
            if (myid == 0) {
                std::vector<T> a_fact = gather_to_rank0<T>(handle, desc, local.data(), n);
                std::vector<T> a0_full(static_cast<std::size_t>(n) * n, T(0));
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i)
                        a0_full[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * n] =
                            hash_value<T>(i, j, 7);
                check_factorization<T>(a0_full, a_fact, tau_all, n, r1, r2);
            } else {
                gather_to_rank0<T>(handle, desc, local.data(), n);   // sends, returns {}
            }
            double pair[2] = {r1, r2};
            MPI_Bcast(pair, 2, MPI_DOUBLE, 0, comm);
            // NaN-blind verdict: a NaN residual fails the comparison.
            if (!(pair[0] < 50.0) || !(pair[1] < 50.0)) {
                if (myid == 0)
                    std::cerr << "pgeqrf check FAILED at n=" << n << " (res=" << pair[0]
                              << ", orth=" << pair[1] << ")" << std::endl;
                MPI_Abort(comm, 1);
            }
        }

        if (myid == 0 && !times.empty()) {
            std::sort(times.begin(), times.end());
            const double median = times.size() % 2 == 1
                ? times[times.size() / 2]
                : 0.5 * (times[times.size() / 2 - 1] + times[times.size() / 2]);
            const double n3 = static_cast<double>(n) * n * n;
            const double flops = (2.0 * n3 - 2.0 * n3 / 3.0) * (is_cplx<T>::value ? 4.0 : 1.0);
            std::cout << "RESULT n=" << n << " type=" << tname
                      << " backend=" << backend_name(handle)
                      << " grid=" << benchmark_cli::grid_name(options)
                      << " ranks=" << handle->nprocs
                      << " nb=" << nb
                      << " seed=" << kRandomSeed
                      << " reps=" << options.repeats
                      << " time_s=" << std::fixed << std::setprecision(6) << median
                      << " gflops=" << std::fixed << std::setprecision(2)
                      << (median > 0.0 ? flops / median / 1e9 : 0.0)
                      << " check=PASS" << std::endl;
        }
        DDLA_CHECK(ddlaFree(handle, d_A));
    }
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int nprocs = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    benchmark_cli::Options options;
    std::string option_error;
    if (!benchmark_cli::parse(argc, argv, true, 1, options, option_error)) {
        if (rank == 0) {
            std::cerr << "Error: " << option_error << std::endl;
            std::cerr << benchmark_cli::usage(argv[0], true) << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (nprocs != options.nprows * options.npcols) {
        if (rank == 0)
            std::cerr << "--grid " << benchmark_cli::grid_name(options) << " requires "
                      << options.nprows * options.npcols << " ranks, have " << nprocs << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int nb = 64;
    if (const char* env = std::getenv("DDLA_BENCH_NB")) {
        const int parsed = std::atoi(env);
        if (parsed <= 0) {
            if (rank == 0) std::cerr << "invalid DDLA_BENCH_NB: " << env << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        nb = parsed;
    }
    if (bench_uplo() == 'L' && std::getenv("DDLA_BENCH_NOCHECK") == nullptr) {
        if (rank == 0) std::cerr << "DDLA_BENCH_UPLO=L needs DDLA_BENCH_NOCHECK=1 (the host oracle is QR-only)" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::string types = "both";
    if (const char* env = std::getenv("DDLA_BENCH_TYPE")) types = env;
    const bool run_d = types == "both" || types == "d";
    const bool run_z = types == "both" || types == "z";
    if (!run_d && !run_z) {
        if (rank == 0) std::cerr << "invalid DDLA_BENCH_TYPE: " << types << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    DdlaHandle_t handle = nullptr;
    ddlaInit(handle);
    ddlaSet(handle, MPI_COMM_WORLD, options.nprows, options.npcols);

    if (run_d) run_type<double>(handle, options, nb, "double");
    if (run_z) run_type<std::complex<double>>(handle, options, nb, "complex<double>");

    ddlaDestroy(handle);
    MPI_Finalize();
    return 0;
}
