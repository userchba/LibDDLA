#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "api_grid_test_common.h"

// ---------------------------------------------------------------------------
// Isolation test for the distributed bidiagonalization pgebrd and the
// reflector apply punmbr (ScaLAPACK PxGEBRD / PxUNMBR semantics).
//
// For every geometry, both the upper (m >= n) and lower (m < n) reductions:
//   1. contract: d, e, tauq, taup bitwise identical on every rank; the input
//      outside the leading block (padded descriptors) untouched; info == 0;
//   2. against host LAPACK ?GEBRD on the same input: d, e, tauq, taup and
//      the reflector storage.  Golub-Kahan bidiagonalization is a Krylov
//      process, so blocked and unblocked orderings legitimately drift apart
//      by ~10-100 n*eps on random matrices; the bound is 1000 (a wrong
//      conjugation or offset is O(1), i.e. ~1e13 in these units);
//   3. reference-free: Q = ?UNGBR('Q'), P^H = ?UNGBR('P') from OUR output,
//      A = Q B P^H and Q, P unitary (bound 50);
//   4. punmbr on identity matrices against those host Q / P^H, all four
//      (vect, trans) combinations;
//   5. argument rejection: unsupported side, misaligned C.
// ---------------------------------------------------------------------------

#include "svd_test_common.h"

namespace {

template <typename T>
void call_pgebrd(const ddla::DdlaHandle_t& handle, int m, int n, T* d_A, const int* desc,
                 typename ddla::real_type<T>::type* d, typename ddla::real_type<T>::type* e,
                 T* tq, T* tp, int& info)
{
#if DDLA_HAS_GPU
    if (ddlaGetBackend(handle) == DdlaBackend::CPU)
        ddla::pgebrd<T>(handle, m, n, d_A, desc, d, e, tq, tp, info);
    else
        ddla::pgebrd<T>(handle, m, n, d_A, desc, d, e, tq, tp, info);
#elif DDLA_HAS_GPU
    ddla::pgebrd<T>(handle, m, n, d_A, desc, d, e, tq, tp, info);
#else
    ddla::pgebrd<T>(handle, m, n, d_A, desc, d, e, tq, tp, info);
#endif
}

// Fills.  FILL_TINY scales the hash fill to ~1e-305 (1e-36 for float): the
// column norms are below safmin/eps, so ?LARFG's rescale loop (knt > 0) runs
// and a naive sum of squares underflows to 0.  FILL_NEGZERO is a diagonal
// matrix whose off-diagonal entries are -0.0: every reflector is the trivial
// tau = 0 one (H = I), and the sign of zero must not leak into d, e or tau.
// FILL_HUGE scales it to ~1e300 (1e36 for float): the speculative GEMVs of the fused half-steps
// (A^H x, (V,X)^H x on the UNSCALED reflector tail) would overflow without the sigma guard.
enum { FILL_HASH = 0, FILL_TINY = 1, FILL_NEGZERO = 2, FILL_HUGE = 3 };
static bool scaled_fill(int fill) { return fill == FILL_TINY || fill == FILL_HUGE; }

template <typename T>
void run_case(const ddla::DdlaHandle_t& handle, const CaseGeom& g, const std::string& tag)
{
    using rt = typename ddla::real_type<T>::type;
    const double eps = std::is_same<rt, float>::value ? 1.1920929e-7 : 2.2204460492503131e-16;
    const int m = g.m, n = g.n, k = std::min(m, n);
    MPI_Comm comm = ddlaGetCommunicator(handle);
    const int myid = ddlaGetRank(handle);

    int desc[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(desc, handle, g.dm, g.dn, g.mb, g.nb, g.rsrc, g.csrc));
    auto value = [&](int i, int j) -> T {
        const T v = hash_value<T>(i, j, 5);
        if (g.fill == FILL_TINY) {
            const rt sc = std::is_same<rt, float>::value ? rt(1e-36) : rt(1e-305);
            return v * sc;
        }
        if (g.fill == FILL_HUGE) {
            const rt sc = std::is_same<rt, float>::value ? rt(1e36) : rt(1e300);
            return v * sc;
        }
        if (g.fill == FILL_NEGZERO) {
            if (i == j) return v;
            return T(rt(-0.0));
        }
        return v;
    };
    auto h_A = make_local<T>(handle, desc, value);
    DeviceBuffer<T> d_A(handle, h_A.size());
    upload(handle, d_A.ptr, h_A);
    check_ddla_sync(handle);

    std::vector<rt> d(static_cast<std::size_t>(std::max(k, 1)), rt(0));
    std::vector<rt> e(static_cast<std::size_t>(std::max(k - 1, 1)), rt(0));
    std::vector<T> tq(static_cast<std::size_t>(std::max(k, 1)), T(0));
    std::vector<T> tp(static_cast<std::size_t>(std::max(k, 1)), T(0));
    int info = -1;
    call_pgebrd<T>(handle, m, n, d_A.ptr, desc, d.data(), e.data(), tq.data(), tp.data(), info);
    check_ddla_sync(handle);
    int ginfo = -1;
    MPI_Allreduce(&info, &ginfo, 1, MPI_INT, MPI_MAX, comm);
    if (ginfo != 0) {
        if (myid == 0) std::cerr << "pgebrd info=" << ginfo << " for " << tag << std::endl;
        MPI_Abort(comm, 1);
    }

    // ---- 1. replication + outside-the-block contract ----------------------
    double repl = std::max(std::max(replication_err(handle, d), replication_err(handle, e)),
                           std::max(replication_err(handle, tq), replication_err(handle, tp)));
    require_close(handle, tag + " d/e/tau replication", repl, 0.0);

    const auto full = gather_full<T>(handle, d_A.ptr, desc, g.dm, g.dn);
    auto gat = [&](int i, int j) -> const T& {
        return full[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * g.dm];
    };
    double outside = 0.0;
    for (int j = 0; j < g.dn; ++j)
        for (int i = 0; i < g.dm; ++i)
            if (i >= m || j >= n) outside = std::max(outside, dabs(gat(i, j) - value(i, j)));
    require_close(handle, tag + " outside-leading-block", outside, 0.0);
    if (k == 0) {
        if (myid == 0)
            std::cout << "[grid " << grid_name(handle) << "] pgebrd " << tag << " ok (empty)" << std::endl;
        return;
    }

    // ---- 2. against host ?gebrd -------------------------------------------
    std::vector<T> a0(static_cast<std::size_t>(m) * n);
    // ||A0||_F with the fill's scale divided out first: summing squares of ~1e-305
    // entries underflows to 0 (the library's norms are robust, the test's must be too).
    const double fill_scale = (g.fill == FILL_TINY) ? (std::is_same<rt, float>::value ? 1e-36 : 1e-305)
                            : (g.fill == FILL_HUGE) ? (std::is_same<rt, float>::value ? 1e36 : 1e300) : 1.0;
    double a0_fnorm = 0.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            a0[i + static_cast<std::size_t>(j) * m] = value(i, j);
            a0_fnorm += std::norm(value(i, j) / rt(fill_scale));
        }
    a0_fnorm = std::sqrt(a0_fnorm) * fill_scale;
    {
        std::vector<T> ha = a0;
        std::vector<rt> hd(static_cast<std::size_t>(k)), he(static_cast<std::size_t>(std::max(k - 1, 1)));
        std::vector<T> htq(static_cast<std::size_t>(k)), htp(static_cast<std::size_t>(k));
        host_gebrd<T>(ha, m, n, hd, he, htq, htp);
        const double scale = std::max(m, n) * eps;
        double a_err = 0.0, a_err_band = 0.0, a_err_rest = 0.0, de_err = 0.0, tau_err = 0.0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                const double err = dabs(gat(i, j) - ha[i + static_cast<std::size_t>(j) * m]);
                a_err = std::max(a_err, err);
                // the bidiagonal band carries beta/e (scale of A0); everything else is the
                // reflector tails, which are scale-free O(1) numbers whatever A0's scale
                const bool band = (i == j) || (m >= n ? j == i + 1 : i == j + 1);
                (band ? a_err_band : a_err_rest) = std::max(band ? a_err_band : a_err_rest, err);
            }
        for (int i = 0; i < k; ++i) {
            de_err = std::max(de_err, dabs(d[i] - hd[i]));
            tau_err = std::max(tau_err, std::max(dabs(tq[i] - htq[i]), dabs(tp[i] - htp[i])));
        }
        for (int i = 0; i + 1 < k; ++i) de_err = std::max(de_err, dabs(e[i] - he[i]));
        const double anorm = scaled_fill(g.fill) ? a0_fnorm : std::max(a0_fnorm, 1.0);
        if (scaled_fill(g.fill)) {
            require_close(handle, tag + " A-band-vs-lapack", a_err_band / (anorm * scale), 1000.0);
            require_close(handle, tag + " A-tails-vs-lapack", a_err_rest / scale, 1000.0);
        } else {
            require_close(handle, tag + " A-vs-lapack", a_err / (anorm * scale), 1000.0);
        }
        require_close(handle, tag + " d/e-vs-lapack", de_err / (anorm * scale), 1000.0);
        require_close(handle, tag + " tau-vs-lapack", tau_err / scale, 1000.0);
    }

    if (g.tall) {
        if (myid == 0)
            std::cout << "[grid " << grid_name(handle) << "] pgebrd " << tag << " ok (tall-skinny)" << std::endl;
        return;
    }

    // ---- 3. reference-free: A = Q B P^H from our own reflectors ------------
    std::vector<T> hQ(static_cast<std::size_t>(m) * m, T(0));
    for (int j = 0; j < std::min(m, n); ++j)
        for (int i = 0; i < m; ++i) hQ[i + static_cast<std::size_t>(j) * m] = gat(i, j);
    host_ungbr<T>('Q', m, n, hQ, tq);
    std::vector<T> hPh(static_cast<std::size_t>(n) * n, T(0));
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < std::min(m, n); ++i) hPh[i + static_cast<std::size_t>(j) * n] = gat(i, j);
    host_ungbr<T>('P', n, m, hPh, tp);
    auto bval = [&](int i, int j) -> T {
        if (i == j && i < k) return T(d[i]);
        if (m >= n && j == i + 1 && i + 1 < k) return T(e[i]);
        if (m < n && i == j + 1 && j + 1 < k) return T(e[j]);
        return T(0);
    };
    double rec = 0.0, qorth = 0.0, porth = 0.0;
    {
        // QB (m x n) then (QB) P^H; B is bidiagonal so QB is cheap.
        std::vector<T> qb(static_cast<std::size_t>(m) * n, T(0));
        for (int j = 0; j < n; ++j)
            for (int p = std::max(0, j - 1); p <= std::min(j + 1, k - 1); ++p) {
                const T b = bval(p, j);
                if (b == T(0)) continue;
                for (int i = 0; i < m; ++i)
                    qb[i + static_cast<std::size_t>(j) * m] += hQ[i + static_cast<std::size_t>(p) * m] * b;
            }
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                T acc{};
                for (int p = 0; p < n; ++p)
                    acc += qb[i + static_cast<std::size_t>(p) * m] * hPh[p + static_cast<std::size_t>(j) * n];
                rec = std::max(rec, dabs(acc - a0[i + static_cast<std::size_t>(j) * m]));
            }
        for (int j = 0; j < m; ++j)
            for (int i = 0; i < m; ++i) {
                T acc{};
                for (int p = 0; p < m; ++p)
                    acc += conj_t(hQ[p + static_cast<std::size_t>(i) * m]) * hQ[p + static_cast<std::size_t>(j) * m];
                qorth = std::max(qorth, dabs(acc - T(i == j ? rt(1) : rt(0))));
            }
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                T acc{};
                for (int p = 0; p < n; ++p)
                    acc += hPh[i + static_cast<std::size_t>(p) * n] * conj_t(hPh[j + static_cast<std::size_t>(p) * n]);
                porth = std::max(porth, dabs(acc - T(i == j ? rt(1) : rt(0))));
            }
    }
    require_close(handle, tag + " A-QBP^H",
                  rec / ((scaled_fill(g.fill) ? a0_fnorm : std::max(a0_fnorm, 1.0)) * std::max(m, n) * eps),
                  50.0);
    require_close(handle, tag + " Q orth", qorth / (std::max(m, 1) * eps), 50.0);
    require_close(handle, tag + " P orth", porth / (std::max(n, 1) * eps), 50.0);

    // ---- 4. punmbr against the host operators -------------------------------
    check_punmbr<T>(handle, g, d_A.ptr, desc, tq, tp, hQ, hPh, tag);

    if (myid == 0)
        std::cout << "[grid " << grid_name(handle) << "] pgebrd " << tag << " ok"
                  << (g.dm != g.m || g.dn != g.n ? " (padded)" : "") << std::endl;
}

// Every rank must reject the call (a throw on some ranks only would hang the
// next collective, so the outcome is agreed before continuing).
template <typename Fn>
void require_throws(const ddla::DdlaHandle_t& handle, const std::string& name, Fn fn)
{
    int threw = 0;
    try { fn(); } catch (const std::exception&) { threw = 1; }
    int all = 0;
    MPI_Allreduce(&threw, &all, 1, MPI_INT, MPI_MIN, ddlaGetCommunicator(handle));
    require_close(handle, name + " rejected", all == 1 ? 0.0 : 1.0, 0.0);
}

template <typename T>
void check_rejection(const ddla::DdlaHandle_t& handle)
{
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    const int m = 12, n = 9;
    int descA[ddla::DDLA_DLEN_], descC[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, handle, m, n, 3, 3, 0, 0));
    DeviceBuffer<T> d_A(handle, local_size(handle, descA));
    DeviceBuffer<T> d_C(handle, 64 * 64);
    std::vector<T> tau(static_cast<std::size_t>(n), T(0));
    int info = 0;
    DDLA_CHECK(ddlaDescInit(descC, handle, m, m, 3, 3, 0, 0));
    require_throws(handle, "punmbr Q side=R", [&] {
        call_punmbr<T>(handle, 'Q', 'R', 'N', m, m, n, d_A.ptr, descA, tau.data(), d_C.ptr, descC, info);
    });
    require_throws(handle, "punmbr P side=L", [&] {
        call_punmbr<T>(handle, 'P', 'L', 'N', n, n, m, d_A.ptr, descA, tau.data(), d_C.ptr, descC, info);
    });
    if (nprows > 1) {
        DDLA_CHECK(ddlaDescInit(descC, handle, m, m, 2, 3, 0, 0));   // MB differs from A's
        require_throws(handle, "punmbr Q misaligned rows", [&] {
            call_punmbr<T>(handle, 'Q', 'L', 'N', m, m, n, d_A.ptr, descA, tau.data(), d_C.ptr, descC, info);
        });
    }
}

template <typename T>
void check_pgebrd(const ddla::DdlaHandle_t& handle)
{
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    const int pr = nprows - 1, pc = npcols - 1;
    const CaseGeom cases[] = {
        {271, 263, 257, 259, 0, 0, 273, 267}, // bounded panels, uneven blocks
        {263, 271, 259, 257, 0, 0, 267, 273}, // lower bidiagonal counterpart
        {23, 23, 4, 4, 0, 0, 23, 23},       // square
        {37, 17, 5, 3, 0, 0, 37, 17},       // tall, mb != nb
        {17, 37, 3, 5, 0, 0, 17, 37},       // wide (lower bidiagonal), mb != nb
        {21, 19, 3, 5, 0, 0, 21, 19},       // mb=3, nb=5
        {19, 21, 5, 2, 0, 0, 19, 21},       // wide, mb=5, nb=2
        {9, 5, 4, 6, 0, 0, 9, 5},           // n < nb
        {5, 9, 6, 4, 0, 0, 5, 9},           // m < mb (wide)
        {20, 20, 4, 4, pr, pc, 20, 20},     // rsrc = csrc = P-1
        {14, 22, 4, 4, pr, pc, 14, 22},     // wide, rsrc = csrc = P-1
        {21, 15, 4, 5, 0, 0, 27, 20},       // padded descriptor
        {15, 21, 5, 4, 0, 0, 20, 27},       // padded, wide
        {7, 9, 2, 1, 0, 0, 7, 9},           // nb = 1
        {9, 7, 1, 2, 0, 0, 9, 7},           // mb = 1
        {1, 5, 2, 2, 0, 0, 1, 5},           // m = 1
        {5, 1, 2, 2, 0, 0, 5, 1},           // n = 1
        {2, 2, 1, 1, 0, 0, 2, 2},           // tiny, one element per block
        {0, 5, 2, 2, 0, 0, 0, 5},           // m = 0
        {5, 0, 2, 2, 0, 0, 5, 0},           // n = 0
        {64, 48, 8, 6, 0, 0, 64, 48},       // multi-cycle
        {48, 64, 6, 8, 0, 0, 48, 64},       // multi-cycle, wide
        {23, 23, 4, 4, 0, 0, 23, 23, FILL_TINY},        // knt rescale, square
        {37, 17, 5, 3, 0, 0, 37, 17, FILL_TINY},        // knt rescale, tall
        {17, 37, 3, 5, 0, 0, 17, 37, FILL_TINY},        // knt rescale, wide
        {23, 23, 4, 4, 0, 0, 23, 23, FILL_HUGE},        // overflow guard, square
        {37, 17, 5, 3, 0, 0, 37, 17, FILL_HUGE},        // overflow guard, tall
        {17, 37, 3, 5, 0, 0, 17, 37, FILL_HUGE},        // overflow guard, wide
        {23, 23, 4, 4, 0, 0, 23, 23, FILL_NEGZERO},     // trivial reflectors, square
        {21, 13, 5, 3, 0, 0, 21, 13, FILL_NEGZERO},     // trivial reflectors, tall
        {13, 21, 3, 5, 0, 0, 13, 21, FILL_NEGZERO},     // trivial reflectors, wide
    };
    for (const CaseGeom& g : cases)
        run_case<T>(handle, g, std::string(type_tag<T>()) + " " + std::to_string(g.m) + "x" +
                                    std::to_string(g.n) +
                                    (g.fill == FILL_TINY ? " tiny" : g.fill == FILL_HUGE ? " huge" : g.fill == FILL_NEGZERO ? " negzero" : ""));
    check_rejection<T>(handle);
}

// m so large that EVERY rank owns more than 131072 local rows (the 1024 x 128 one-pass span of the
// element-wise device kernels): n = 9 keeps the host ?gebrd reference cheap; no Q (m x m) is formed.
template <typename T>
void check_pgebrd_tall(const ddla::DdlaHandle_t& handle)
{
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    const int m = (131072 + 64) * nprows + 5;
    CaseGeom g{m, 9, 64, 4, 0, 0, m, 9};
    g.tall = true;
    run_case<T>(handle, g, std::string(type_tag<T>()) + " " + std::to_string(m) + "x9");
}

static void check_body(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    (void)base;
    check_pgebrd<double>(handle);
    check_pgebrd<Complex>(handle);
    check_pgebrd<float>(handle);
    check_pgebrd<std::complex<float>>(handle);
    check_pgebrd_tall<double>(handle);
    check_pgebrd_tall<Complex>(handle);

    // Column-major process grid ordering.
    {
        int nprows = 0, npcols = 0;
        ddlaGetGridDims(handle, nprows, npcols);
        ddla::DdlaHandle_t h_cm = nullptr;
        ddla::ddlaInit(h_cm);
        ddla::ddlaSet(h_cm, MPI_COMM_WORLD, nprows, npcols, 'C');
        run_case<Complex>(h_cm, CaseGeom{23, 19, 4, 4, 0, 0, 23, 19}, "colmajor-grid tall");
        run_case<Complex>(h_cm, CaseGeom{19, 23, 4, 4, 0, 0, 19, 23}, "colmajor-grid wide");
        ddla::ddlaDestroy(h_cm);
    }

#if DDLA_HAS_GPU
    // Dual lane: the whole list again on a CPU handle.
    if (ddlaGetBackend(handle) == DdlaBackend::GPU && ddlaBackendAvailable(DdlaBackend::CPU)) {
        int nprows = 0, npcols = 0;
        ddlaGetGridDims(handle, nprows, npcols);
        ddla::DdlaHandle_t h_cpu = nullptr;
        ddla::ddlaInit(h_cpu, DdlaBackend::CPU);
        ddla::ddlaSet(h_cpu, MPI_COMM_WORLD, nprows, npcols, 'R');
        check_pgebrd<double>(h_cpu);
        check_pgebrd<Complex>(h_cpu);
        check_pgebrd<float>(h_cpu);
        check_pgebrd<std::complex<float>>(h_cpu);
        ddla::ddlaDestroy(h_cpu);
    }
#endif
}

} // namespace

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pgebrd", check_body);
}
