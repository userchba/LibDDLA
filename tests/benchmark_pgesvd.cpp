// ---------------------------------------------------------------------------
// Benchmark for the distributed SVD pgesvd and its stages.
//
// Backend-neutral like benchmark_pgeqrf: it runs on the build's default
// backend, so one binary serves the CUDA, dual and CPU lanes.  Options come
// from benchmark_grid_options.h (--grid, --repeats, positional sizes); the
// environment selects
//
//     DDLA_BENCH_NB    block size mb == nb (default 64)
//     DDLA_BENCH_TYPE  d, z or both (default both)
//
// Square n x n problems.  Per size and type, each timed separately on a fresh
// copy of the pristine input (every routine destroys A), median over
// --repeats of the per-rep max over ranks:
//
//   brd     pgebrd
//   bdsdc   independent rank-0 host LAPACK reference for the bidiagonal solve;
//           bdsdc_comm measures broadcasting its full factors. This reference
//           is not the distributed bidiagonal solver used by pgesvd.
//   svd_NN  pgesvd jobu = jobvt = 'N'   (pgebrd + dqds)
//   svd_VV  pgesvd jobu = jobvt = 'V'   (pgebrd + the dc bidiagonal solve +
//                                        both back-transforms)
//   unmbr   punmbr 'Q','L','N' plus 'P','R','C' on an n x n block
// svd_VV - brd - unmbr estimates the distributed bidiagonal stage.
//
// DDLA_BENCH_NOCHECK=1 skips the correctness check after the reps (profiling
// runs only: the default check keeps non-negotiable (g)).
//
// Following the eigensolver project's non-negotiable (g), every size is
// CHECKED at the timed size from the svd_VV output before check=PASS is
// printed, with distributed probes (X: n x 4 uniform):
//   ||A0 X - U diag(s) (VT X)||_F / (||A0||_F ||X||_F n eps) < 50
//   ||U^H (U Q) - Q||_F / (||Q||_F n eps) < 50,  same for VT^H
// A violation calls MPI_Abort.
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "mpi_test_init.h"

#include "benchmark_grid_options.h"
#include "test_desc_helpers.h"

#include <ddla/ddla.h>
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "mpi_datatype.h"

// Host ?BDSDC with COMPQ='I' through the Fortran ABI, for the bdsdc timing
// leg (the same prototypes pgesvd.cpp declares).
extern "C" {
void dbdsdc_(const char* uplo, const char* compq, const int* n, double* d, double* e,
             double* u, const int* ldu, double* vt, const int* ldvt, double* q, int* iq,
             double* work, int* iwork, int* info);
void sbdsdc_(const char* uplo, const char* compq, const int* n, float* d, float* e,
             float* u, const int* ldu, float* vt, const int* ldvt, float* q, int* iq,
             float* work, int* iwork, int* info);
}

using namespace ddla;

namespace {

// --- bdsdc timing leg helpers -----------------------------------------------

const char kCompqI = 'I';

void bdsdc_bench(char uplo, int n, float* d, float* e, float* u, float* vt, float* work,
                 int* iwork, int& info)
{
    sbdsdc_(&uplo, &kCompqI, &n, d, e, u, &n, vt, &n, nullptr, nullptr, work, iwork, &info);
}
void bdsdc_bench(char uplo, int n, double* d, double* e, double* u, double* vt, double* work,
                 int* iwork, int& info)
{
    dbdsdc_(&uplo, &kCompqI, &n, d, e, u, &n, vt, &n, nullptr, nullptr, work, iwork, &info);
}

// MPI_Bcast in chunks (bcast_large of pgesvd.cpp, retyped on the argument's
// element type via the first parameter).
template <typename rt>
void bcast_large_rt(const rt&, rt* p, std::size_t count, MPI_Comm comm)
{
    const std::size_t chunk = std::size_t(1) << 27;
    MPI_Datatype mt;
    if constexpr (std::is_same<rt, double>::value) mt = MPI_DOUBLE;
    else mt = MPI_FLOAT;
    for (std::size_t off = 0; off < count; off += chunk) {
        const int c = static_cast<int>(std::min(chunk, count - off));
        MPI_Bcast(p + off, c, mt, 0, comm);
    }
}

constexpr int kProbeCols = 4;

template <typename T> struct is_cplx : std::false_type {};
template <typename U> struct is_cplx<std::complex<U>> : std::true_type {};

template <typename U> U conj_h(U v) { return v; }
template <typename U> std::complex<U> conj_h(std::complex<U> v) { return std::conj(v); }

template <typename T> MPI_Datatype mpi_type();
template <> MPI_Datatype mpi_type<double>() { return MPI_DOUBLE; }
template <> MPI_Datatype mpi_type<std::complex<double>>() { return MPI_C_DOUBLE_COMPLEX; }

// splitmix64 of (i, j, tag) -> uniform in [-0.5, 0.5): full rank with spread
// singular values (a sin-of-linear-combination hash has a rank-2 core).
inline double hash01(long long i, long long j, int tag)
{
    std::uint64_t x = static_cast<std::uint64_t>(i) * 0x9E3779B97F4A7C15ull
                    ^ static_cast<std::uint64_t>(j) * 0xC2B2AE3D27D4EB4Full
                    ^ static_cast<std::uint64_t>(tag + 1) * 0x165667B19E3779F9ull;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27; x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    return static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0) - 0.5;
}

template <typename T>
T hash_value(long long i, long long j, int tag)
{
    if constexpr (is_cplx<T>::value) return T(hash01(i, j, tag), hash01(i, j, tag + 1000));
    else return T(hash01(i, j, tag));
}

const char* backend_name(const DdlaHandle_t& h)
{
    return ddlaGetBackend(h) == DdlaBackend::CPU ? "CPU" : "GPU";
}


template <typename T>
void call_pgebrd(const DdlaHandle_t& handle, int n, T* d_A, const int* desc,
                 std::vector<typename real_type<T>::type>& d, std::vector<typename real_type<T>::type>& e,
                 std::vector<T>& tq, std::vector<T>& tp)
{
    int info = 0;
    pgebrd<T>(handle, n, n, d_A, desc, d.data(), e.data(), tq.data(), tp.data(), info);
}

template <typename T>
int call_pgesvd(const DdlaHandle_t& handle, char job, int n, T* d_A, const int* desc,
                typename real_type<T>::type* s, T* d_U, T* d_VT)
{
    int info = 0;
    pgesvd<T>(handle, job, job, n, n, d_A, desc, s, d_U, desc, d_VT, desc, info);
    return info;
}

template <typename T>
void call_punmbr(const DdlaHandle_t& handle, char vect, char side, char trans, int n,
                 const T* d_A, const int* desc, const T* tau, T* d_C)
{
    int info = 0;
    punmbr<T>(handle, vect, side, trans, n, n, n, d_A, desc, tau, d_C, desc, info);
}

double median_of(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

template <typename T>
void run_type(const DdlaHandle_t& handle, const benchmark_cli::Options& options, int nb,
              const char* tname)
{
    using rt = typename real_type<T>::type;
    MPI_Comm comm = ddlaGetCommunicator(handle);
    const int myid = ddlaGetRank(handle);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
    const double eps = 2.2204460492503131e-16;

    for (const int n : options.sizes) {
        int desc[DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(desc, handle, n, n, nb, nb, 0, 0));
        const int mloc = ddla_test::m_loc(handle, desc);
        const int nloc = ddla_test::n_loc(handle, desc);
        const int lld = desc[DDLA_LLD_];
        const std::size_t nelem = static_cast<std::size_t>(lld) * std::max(nloc, 1);
        std::vector<int> grow(static_cast<std::size_t>(mloc)), gcol(static_cast<std::size_t>(nloc));
        for (int i = 0; i < mloc; ++i) grow[i] = indx_l2g_r(desc, handle, i);
        for (int j = 0; j < nloc; ++j) gcol[j] = indx_l2g_c(desc, handle, j);
        std::vector<T> h_A0(nelem, T(0));
        for (int j = 0; j < nloc; ++j)
            for (int i = 0; i < mloc; ++i)
                h_A0[i + static_cast<std::size_t>(j) * lld] = hash_value<T>(grow[i], gcol[j], 3);

        T *d_A = nullptr, *d_U = nullptr, *d_VT = nullptr;
        DDLA_CHECK(ddlaMalloc(handle, reinterpret_cast<void**>(&d_A), nelem * sizeof(T)));
        DDLA_CHECK(ddlaMalloc(handle, reinterpret_cast<void**>(&d_U), nelem * sizeof(T)));
        DDLA_CHECK(ddlaMalloc(handle, reinterpret_cast<void**>(&d_VT), nelem * sizeof(T)));
        auto refill = [&]() {
            DDLA_CHECK(ddlaMemcpy(handle, d_A, h_A0.data(), nelem * sizeof(T)));
            DDLA_CHECK(ddlaSynchronize(handle));
            MPI_Barrier(comm);
        };
        auto timed = [&](auto&& body) {
            refill();
            const double t0 = MPI_Wtime();
            body();
            DDLA_CHECK(ddlaSynchronize(handle));
            double t = MPI_Wtime() - t0, tmax = 0.0;
            MPI_Allreduce(&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, comm);
            return tmax;
        };

        std::vector<rt> d(n), e(std::max(1, n - 1)), s(n);
        std::vector<T> tq(n), tp(n);
        std::vector<double> t_brd, t_nn, t_vv, t_unmbr, t_bdsdc, t_bdc_comm;
        int info_vv = 0;
        // The leg's ?BDSDC always runs with UPLO='U' — the leg times the
        // solve cost by size, and every benchmark input here is square (the
        // real benchmark's uplo is 'U' too).
        const char uplo_c = 'U';
        // one untimed warm-up of the full driver, then the timed reps
        refill();
        (void)call_pgesvd<T>(handle, 'V', n, d_A, desc, s.data(), d_U, d_VT);
        for (int rep = 0; rep < options.repeats; ++rep) {
            t_brd.push_back(timed([&] { call_pgebrd<T>(handle, n, d_A, desc, d, e, tq, tp); }));
            // Independent host LAPACK reference: rank 0 times ?BDSDC alone (host, real
            // type — B is real for all four T), everyone times the broadcast
            // of s and the full k x k factors.  Run on the last rep (the
            // refill before it left pgebrd's current d/e in the vectors).
            if (rep == options.repeats - 1) {
                const std::size_t kk = static_cast<std::size_t>(n) * n;
                std::vector<rt> fu(kk, rt(0)), fvt(kk, rt(0)), work;   // every rank receives the broadcast
                std::vector<int> iwork;
                if (myid == 0) {
                    work.resize(3 * kk + 4 * static_cast<std::size_t>(n));
                    iwork.resize(8 * static_cast<std::size_t>(n));
                }
                std::copy(d.begin(), d.end(), s.begin());
                const double t0 = MPI_Wtime();
                if (myid == 0) {
                    int binfo = 0;
                    bdsdc_bench(uplo_c, n, s.data(), e.data(), fu.data(), fvt.data(),
                                work.data(), iwork.data(), binfo);
                    if (binfo != 0) { std::cerr << "bdsdc info=" << binfo << std::endl; MPI_Abort(comm, 1); }
                }
                const double t_solve = MPI_Wtime() - t0;
                MPI_Barrier(comm);
                const double t1 = MPI_Wtime();
                MPI_Bcast(s.data(), n, detail::mpi_datatype<rt>(), 0, comm);
                bcast_large_rt(rt{}, fu.data(), kk, comm);
                bcast_large_rt(rt{}, fvt.data(), kk, comm);
                const double t_comm = MPI_Wtime() - t1;
                double a = t_solve, amax = 0.0, cm = t_comm, cmax = 0.0;
                MPI_Allreduce(&a, &amax, 1, MPI_DOUBLE, MPI_MAX, comm);
                MPI_Allreduce(&cm, &cmax, 1, MPI_DOUBLE, MPI_MAX, comm);
                t_bdsdc.push_back(amax);
                t_bdc_comm.push_back(cmax);
                fu.clear(); fu.shrink_to_fit();
                fvt.clear(); fvt.shrink_to_fit();
                work.clear(); work.shrink_to_fit();
            }
            // Both back-transforms on the A pgebrd just factored (so no
            // refill), applied to a dense n x n C (A0's values).
            {
                DDLA_CHECK(ddlaMemcpy(handle, d_U, h_A0.data(), nelem * sizeof(T)));
                DDLA_CHECK(ddlaSynchronize(handle));
                MPI_Barrier(comm);
                const double t0 = MPI_Wtime();
                call_punmbr<T>(handle, 'Q', 'L', 'N', n, d_A, desc, tq.data(), d_U);
                call_punmbr<T>(handle, 'P', 'R', 'C', n, d_A, desc, tp.data(), d_U);
                DDLA_CHECK(ddlaSynchronize(handle));
                double t = MPI_Wtime() - t0, tmax = 0.0;
                MPI_Allreduce(&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, comm);
                t_unmbr.push_back(tmax);
            }
            t_nn.push_back(timed([&] { (void)call_pgesvd<T>(handle, 'N', n, d_A, desc, s.data(), d_U, d_VT); }));
            t_vv.push_back(timed([&] { info_vv = call_pgesvd<T>(handle, 'V', n, d_A, desc, s.data(), d_U, d_VT); }));
        }
        if (info_vv != 0) {
            if (myid == 0) std::cerr << "pgesvd info=" << info_vv << " at n=" << n << std::endl;
            MPI_Abort(comm, 1);
        }

        // ---- check at the timed size from the last svd_VV output -----------
        // DDLA_BENCH_NOCHECK=1 skips it (profiling runs only); the RESULT
        // line then reports check=NOCHECK instead of PASS.
        double res = 0.0, orth_u = 0.0, orth_v = 0.0;
        const bool do_check = std::getenv("DDLA_BENCH_NOCHECK") == nullptr;
        if (do_check) {
        std::vector<T> hU(nelem), hVT(nelem);
        DDLA_CHECK(ddlaMemcpy(handle, hU.data(), d_U, nelem * sizeof(T)));
        DDLA_CHECK(ddlaMemcpy(handle, hVT.data(), d_VT, nelem * sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(handle));
        const std::size_t np = static_cast<std::size_t>(n) * kProbeCols;
        std::vector<T> X(np), Q(np);
        for (int c = 0; c < kProbeCols; ++c)
            for (int i = 0; i < n; ++i) {
                X[i + static_cast<std::size_t>(c) * n] = hash_value<T>(i, c, 41);
                Q[i + static_cast<std::size_t>(c) * n] = hash_value<T>(i, c, 43);
            }
        // y = op(M) x for a distributed square M (local block hM) and a
        // replicated x; the result is replicated by one grid all-reduce.
        auto apply = [&](const std::vector<T>& hM, bool adjoint, const std::vector<T>& x) {
            std::vector<T> y(np, T(0));
            for (int c = 0; c < kProbeCols; ++c)
                for (int j = 0; j < nloc; ++j)
                    for (int i = 0; i < mloc; ++i) {
                        const T a = hM[i + static_cast<std::size_t>(j) * lld];
                        if (!adjoint) y[grow[i] + static_cast<std::size_t>(c) * n] += a * x[gcol[j] + static_cast<std::size_t>(c) * n];
                        else y[gcol[j] + static_cast<std::size_t>(c) * n] += conj_h(a) * x[grow[i] + static_cast<std::size_t>(c) * n];
                    }
            MPI_Allreduce(MPI_IN_PLACE, y.data(), static_cast<int>(np), mpi_type<T>(), MPI_SUM, comm);
            return y;
        };
        auto fnorm = [](const std::vector<T>& v) {
            double acc = 0.0;
            for (const T& x : v) acc += std::norm(x);
            return std::sqrt(acc);
        };
        double a0_f = 0.0;
        for (int j = 0; j < nloc; ++j)
            for (int i = 0; i < mloc; ++i) a0_f += std::norm(h_A0[i + static_cast<std::size_t>(j) * lld]);
        MPI_Allreduce(MPI_IN_PLACE, &a0_f, 1, MPI_DOUBLE, MPI_SUM, comm);
        a0_f = std::sqrt(a0_f);

        const auto ax = apply(h_A0, false, X);
        auto w = apply(hVT, false, X);
        for (int c = 0; c < kProbeCols; ++c)
            for (int i = 0; i < n; ++i) w[i + static_cast<std::size_t>(c) * n] *= T(s[i]);
        const auto usvx = apply(hU, false, w);
        std::vector<T> diff(np);
        for (std::size_t i = 0; i < np; ++i) diff[i] = ax[i] - usvx[i];
        res = fnorm(diff) / (a0_f * fnorm(X) * n * eps);
        const auto uq = apply(hU, false, Q);
        const auto uhuq = apply(hU, true, uq);
        const auto vq = apply(hVT, true, Q);          // V Q = VT^H Q
        const auto vhvq = apply(hVT, false, vq);
        for (std::size_t i = 0; i < np; ++i) { diff[i] = uhuq[i] - Q[i]; }
        orth_u = fnorm(diff) / (fnorm(Q) * n * eps);
        for (std::size_t i = 0; i < np; ++i) { diff[i] = vhvq[i] - Q[i]; }
        orth_v = fnorm(diff) / (fnorm(Q) * n * eps);
        if (!(res < 50.0 && orth_u < 50.0 && orth_v < 50.0)) {
            if (myid == 0)
                std::cerr << "CHECK FAILED n=" << n << " type=" << tname << " res=" << res
                          << " orth_u=" << orth_u << " orth_v=" << orth_v << std::endl;
            MPI_Abort(comm, 1);
        }
        }

        if (myid == 0) {
            const double brd = median_of(t_brd), nn = median_of(t_nn), vv = median_of(t_vv),
                         un = median_of(t_unmbr);
            std::cout << std::fixed << std::setprecision(3)
                      << "RESULT n=" << n << " type=" << tname << " backend=" << backend_name(handle)
                      << " grid=" << benchmark_cli::grid_name(options) << " nb=" << nb
                      << " reps=" << options.repeats
                      << " brd_s=" << brd << " svd_NN_s=" << nn << " svd_VV_s=" << vv
                      << " unmbr_s=" << un << " bdsqrV_est_s=" << std::max(0.0, vv - brd - un);
            if (!t_bdsdc.empty())
                std::cout << " bdsdc_s=" << median_of(t_bdsdc)
                          << " bdsdc_comm_s=" << median_of(t_bdc_comm);
            std::cout << std::scientific << std::setprecision(3) << " res=" << res << " orth_u=" << orth_u
                      << " orth_v=" << orth_v << " check=" << (do_check ? "PASS" : "NOCHECK")
                      << std::endl;
        }
        DDLA_CHECK(ddlaFree(handle, d_A));
        DDLA_CHECK(ddlaFree(handle, d_U));
        DDLA_CHECK(ddlaFree(handle, d_VT));
    }
}

} // namespace

int main(int argc, char** argv)
{
    ddla_test_mpi_init(&argc, &argv);
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
        nb = std::atoi(env);
        if (nb <= 0) {
            if (rank == 0) std::cerr << "invalid DDLA_BENCH_NB: " << env << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
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
