#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include "api_grid_test_common.h"

// ---------------------------------------------------------------------------
// Isolation test for the distributed Householder QL factorization, i.e.
// pgeqrf with uplo = 'L' (ScaLAPACK PxGEQLF semantics, A = Q L), plus the uplo
// dispatch itself (flag validation, case-insensitivity).
//
// The factored block is A(0:m, ja:ja+n) with ja = 0 (pgeqrf factors the leading
// block).  For every case:
//   1. bitwise contract: tau replicated identically across process rows,
//      sentinel-preserved tau entries of columns that carry no reflector,
//      input preserved outside the block (padded descriptors, columns left and
//      right of the panel, rows below m), info == 0;
//   2. against host LAPACK ?GEQLF on the same block: L, V, tau and the
//      per-column pivot (L diagonal) relative;
//   3. reference-free with Q = ?UNGQL(our V, tau): A = Q L and Q^H Q = I.
// The structured fill pins exactly-decided columns (alpha-only columns give
// tau == 0 with L keeping LAPACK's sign, a zero column, a complex alpha-only
// column with xnorm == 0).
//
// The fill is splitmix64-based, NOT the sin-hash of the older tests (whose
// off-diagonal core has rank 2 and makes entrywise LAPACK comparison
// ill-posed); a dominant pivot entry (+-5, complex phase for complex types)
// keeps every column's reflector well conditioned.
// ---------------------------------------------------------------------------

using namespace api_grid_test;

namespace {

extern "C" {
void sgeqlf_(const int*, const int*, float*, const int*, float*, float*, const int*, int*);
void dgeqlf_(const int*, const int*, double*, const int*, double*, double*, const int*, int*);
void cgeqlf_(const int*, const int*, std::complex<float>*, const int*, std::complex<float>*,
             std::complex<float>*, const int*, int*);
void zgeqlf_(const int*, const int*, std::complex<double>*, const int*, std::complex<double>*,
             std::complex<double>*, const int*, int*);
void sorgql_(const int*, const int*, const int*, float*, const int*, const float*, float*,
             const int*, int*);
void dorgql_(const int*, const int*, const int*, double*, const int*, const double*, double*,
             const int*, int*);
void cungql_(const int*, const int*, const int*, std::complex<float>*, const int*,
             const std::complex<float>*, std::complex<float>*, const int*, int*);
void zungql_(const int*, const int*, const int*, std::complex<double>*, const int*,
             const std::complex<double>*, std::complex<double>*, const int*, int*);
}

template <typename T> struct is_cplx : std::false_type {};
template <typename U> struct is_cplx<std::complex<U>> : std::true_type {};

template <typename U> U conj_t(U v) { return v; }
template <typename U> std::complex<U> conj_t(std::complex<U> v) { return std::conj(v); }

template <typename U>
double dabs(const U& v) { return static_cast<double>(std::abs(v)); }

// NaN-blind error accumulation guard: once e is NaN it stays NaN.
void acc_max(double& e, double v)
{
    if (std::isnan(v) || v > e) e = v;
}

std::string g_case_tag_suffix;

template <typename T> const char* type_tag() { return "?"; }
template <> const char* type_tag<float>() { return "s"; }
template <> const char* type_tag<double>() { return "d"; }
template <> const char* type_tag<std::complex<float>>() { return "c"; }
template <> const char* type_tag<std::complex<double>>() { return "z"; }

template <typename T> T tau_sentinel()
{
    if constexpr (is_cplx<T>::value) return T(-12345.0, 777.0);
    else return T(-12345.0);
}

// splitmix64 uniform in [-0.5, 0.5).
inline double unit_hash(std::uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0) - 0.5;
}

template <typename T>
T rnd_value(int i, int j, std::uint64_t salt)
{
    using rt = typename ddla::real_type<T>::type;
    const std::uint64_t key = (static_cast<std::uint64_t>(i) << 32) ^ static_cast<std::uint64_t>(j) ^ (salt << 56);
    if constexpr (is_cplx<T>::value)
        return T(static_cast<rt>(unit_hash(key)), static_cast<rt>(unit_hash(key ^ 0xA5A5A5A5ULL)));
    else
        return T(static_cast<rt>(unit_hash(key)));
}

enum { FILL_HASH = 0, FILL_STRUCTURED = 1, FILL_SUBNORMAL = 2, FILL_NEGZERO = 3, FILL_TINY0 = 4, FILL_HUGE = 5 };

struct CaseGeom {
    int m, n, ja;      // factored block A(0:m, ja:ja+n)
    int mb, nb;        // block sizes
    int rsrc, csrc;    // first owning process row/column
    int dm, dn;        // descriptor dims (>= ja+n, m)
    int fill;
};

inline const char* fill_tag(int fill)
{
    switch (fill) {
    case FILL_STRUCTURED: return " structured";
    case FILL_SUBNORMAL:  return " subnormal";
    case FILL_NEGZERO:    return " negzero";
    case FILL_TINY0:      return " tiny-zero-alpha";
    case FILL_HUGE:       return " huge";
    default:              return " hash";
    }
}

// Entry (i, j) of the global input.  Inside the block the pivot position
// (m - n + jj, j) of every reflector column is dominant; the structured fill
// pins three exactly-decided columns (n >= 3, m >= n):
//   jj = n-1: zero except alpha = -15 at its pivot (x == 0  -> tau 0, L keeps -15);
//   jj = n-2: zero above its pivot, complex alpha-only (real types: +1.5, tau 0);
//   jj = n-3: all zero (tau exactly 0).
// FILL_SUBNORMAL makes the FIRST column the factorization processes (jj = n-1, QL runs right to
// left) denormal, so its norm is below safmin (the xLARFG rescale path, distributed over the
// grid); FILL_NEGZERO puts alpha = -0.0 at that column's pivot with x nonzero; FILL_TINY0 puts
// alpha = 0 exactly under a tiny (normal-range, below safmin) x.
template <typename T>
T case_value(const CaseGeom& g, int i, int j)
{
    using rt = typename ddla::real_type<T>::type;
    const bool in_block = i < g.m && j >= g.ja && j < g.ja + g.n;
    if (!in_block) return rnd_value<T>(i, j, 5);
    const int jj = j - g.ja;
    const int piv = g.m - g.n + jj;
    if (g.fill == FILL_STRUCTURED && g.n >= 3 && g.m >= g.n) {
        if (jj == g.n - 1) return T(i == piv ? rt(-15) : rt(0));
        if (jj == g.n - 2) {
            if (i < piv) return T(0);
            if (i == piv) {
                if constexpr (is_cplx<T>::value) return T(rt(1.5), rt(-2.5));
                else return T(rt(1.5));
            }
            return rnd_value<T>(i, j, 6);   // L entry below the pivot
        }
        if (jj == g.n - 3) return T(0);
    }
    if (g.fill == FILL_SUBNORMAL && jj == g.n - 1) {
        const rt ts = std::is_same<rt, float>::value ? rt(1e-40) : rt(1e-310);
        const double s = 0.25 + (unit_hash(static_cast<std::uint64_t>(i) * 31 + 7) + 0.5);
        if constexpr (is_cplx<T>::value)
            return T(static_cast<rt>(ts * s), static_cast<rt>(ts * (0.25 + (unit_hash(static_cast<std::uint64_t>(i) * 17 + 3) + 0.5))));
        else return T(static_cast<rt>(ts * s));
    }
    if (g.fill == FILL_NEGZERO && jj == g.n - 1 && i == piv) return T(-rt(0));
    if (g.fill == FILL_HUGE && jj == g.n - 1) {
        // entries whose squares overflow (>= 1e154 (d) / 1e19 (s)): the plain x^H x of a norm would
        // be Inf, the scale-safe norm is finite
        const rt hs = std::is_same<rt, float>::value ? rt(1e30) : rt(1e200);
        const double s = 0.25 + (unit_hash(static_cast<std::uint64_t>(i) * 53 + 11) + 0.5);
        if constexpr (is_cplx<T>::value)
            return T(static_cast<rt>(hs * s), static_cast<rt>(hs * (0.25 + (unit_hash(static_cast<std::uint64_t>(i) * 29 + 13) + 0.5))));
        else return T(static_cast<rt>(hs * s));
    }
    if (g.fill == FILL_TINY0 && jj == g.n - 1) {
        if (i == piv) return T(0);
        const rt ts = std::is_same<rt, float>::value ? rt(1e-35) : rt(1e-300);
        const double s = 0.25 + (unit_hash(static_cast<std::uint64_t>(i) * 41 + 5) + 0.5);
        if constexpr (is_cplx<T>::value)
            return T(static_cast<rt>(ts * s), static_cast<rt>(ts * (0.25 + (unit_hash(static_cast<std::uint64_t>(i) * 23 + 9) + 0.5))));
        else return T(static_cast<rt>(ts * s));
    }
    if (i == piv) {
        if constexpr (is_cplx<T>::value) {
            const double phi = 6.28318530717958648 * (unit_hash(static_cast<std::uint64_t>(i) * 7919 + 13) + 0.5);
            return T(static_cast<rt>(5.0 * std::cos(phi)), static_cast<rt>(5.0 * std::sin(phi)));
        } else {
            return T(unit_hash(static_cast<std::uint64_t>(i) * 104729 + 3) > 0.0 ? rt(5) : rt(-5));
        }
    }
    return rnd_value<T>(i, j, 3) * rt(0.2);
}

template <typename T>
std::vector<T> gather_full(const ddla::DdlaHandle_t& handle, const T* d_A,
                           const int* desc, int rows, int cols)
{
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
    const int m_loc = ddla::num_loc(rows, desc[ddla::DDLA_MB_], myprow, desc[ddla::DDLA_RSRC_], nprows);
    const int n_loc = ddla::num_loc(cols, desc[ddla::DDLA_NB_], mypcol, desc[ddla::DDLA_CSRC_], npcols);
    const int lld = desc[ddla::DDLA_LLD_];
    const auto local = download<T>(handle, d_A, local_size(handle, desc));
    std::vector<T> full(static_cast<std::size_t>(rows) * cols, T(0));
    for (int j = 0; j < n_loc; ++j) {
        const int gj = ddla::indx_l2g_c(desc, handle, j);
        if (gj >= cols) continue;
        for (int i = 0; i < m_loc; ++i) {
            const int gi = ddla::indx_l2g_r(desc, handle, i);
            if (gi >= rows) continue;
            full[gi + static_cast<std::size_t>(gj) * rows] = local[i + static_cast<std::size_t>(j) * lld];
        }
    }
    MPI_Comm comm = ddlaGetCommunicator(handle);
    if constexpr (std::is_same<T, float>::value)
        MPI_Allreduce(MPI_IN_PLACE, full.data(), static_cast<int>(full.size()), MPI_FLOAT, MPI_SUM, comm);
    else if constexpr (std::is_same<T, double>::value)
        MPI_Allreduce(MPI_IN_PLACE, full.data(), static_cast<int>(full.size()), MPI_DOUBLE, MPI_SUM, comm);
    else if constexpr (std::is_same<T, std::complex<float>>::value)
        MPI_Allreduce(MPI_IN_PLACE, full.data(), static_cast<int>(full.size()), MPI_C_FLOAT_COMPLEX, MPI_SUM, comm);
    else
        MPI_Allreduce(MPI_IN_PLACE, full.data(), static_cast<int>(full.size()), MPI_C_DOUBLE_COMPLEX, MPI_SUM, comm);
    return full;
}

template <typename T>
void allreduce_sum(T* data, int count, MPI_Comm comm)
{
    if constexpr (std::is_same<T, float>::value)
        MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_FLOAT, MPI_SUM, comm);
    else if constexpr (std::is_same<T, double>::value)
        MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_DOUBLE, MPI_SUM, comm);
    else if constexpr (std::is_same<T, std::complex<float>>::value)
        MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_C_FLOAT_COMPLEX, MPI_SUM, comm);
    else
        MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_C_DOUBLE_COMPLEX, MPI_SUM, comm);
}

template <typename T>
void host_geqlf(std::vector<T>& a, int m, int n, std::vector<T>& tau)
{
    T wq[1];
    int lwork = -1, info = 0;
    auto call = [&](T* work, int lw) {
        if constexpr (std::is_same<T, float>::value) sgeqlf_(&m, &n, a.data(), &m, tau.data(), work, &lw, &info);
        else if constexpr (std::is_same<T, double>::value) dgeqlf_(&m, &n, a.data(), &m, tau.data(), work, &lw, &info);
        else if constexpr (std::is_same<T, std::complex<float>>::value) cgeqlf_(&m, &n, a.data(), &m, tau.data(), work, &lw, &info);
        else zgeqlf_(&m, &n, a.data(), &m, tau.data(), work, &lw, &info);
    };
    call(wq, lwork);
    lwork = std::max(1, static_cast<int>(std::real(wq[0])));
    std::vector<T> work(lwork);
    call(work.data(), lwork);
    if (info != 0) {
        std::cerr << "host ?geqlf failed, info=" << info << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Q (m x k) = the last k columns of H(k)...H(1) from the factored block's last
// k columns (V above the pivot, implicit unit pivot) and tau.
template <typename T>
std::vector<T> host_ungql(const std::vector<T>& lead, int m, int k, const std::vector<T>& tau)
{
    std::vector<T> q = lead;
    T wq[1];
    int lwork = -1, info = 0;
    auto call = [&](T* work, int lw) {
        if constexpr (std::is_same<T, float>::value) sorgql_(&m, &k, &k, q.data(), &m, tau.data(), work, &lw, &info);
        else if constexpr (std::is_same<T, double>::value) dorgql_(&m, &k, &k, q.data(), &m, tau.data(), work, &lw, &info);
        else if constexpr (std::is_same<T, std::complex<float>>::value) cungql_(&m, &k, &k, q.data(), &m, tau.data(), work, &lw, &info);
        else zungql_(&m, &k, &k, q.data(), &m, tau.data(), work, &lw, &info);
    };
    call(wq, lwork);
    lwork = std::max(1, static_cast<int>(std::real(wq[0])));
    std::vector<T> work(lwork);
    call(work.data(), lwork);
    if (info != 0) {
        std::cerr << "host ?ungql failed, info=" << info << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return q;
}

// Backend dispatch (DDLA_HAS_* are VALUE checks).
template <typename T>
void call_ql(const ddla::DdlaHandle_t& handle, const CaseGeom& g,
             T* d_A, const int* desc, T* tau, int& info)
{
    auto go = [&](auto backend_tag) {
        constexpr DdlaBackend B = decltype(backend_tag)::value;
        ddla::pgeqrf<B, T>(handle, 'L', g.m, g.n, d_A, desc, tau, info);
    };
#if DDLA_HAS_CPU && DDLA_HAS_GPU
    if (ddlaGetBackend(handle) == DdlaBackend::CPU) go(std::integral_constant<DdlaBackend, DdlaBackend::CPU>{});
    else go(std::integral_constant<DdlaBackend, DdlaBackend::GPU>{});
#elif DDLA_HAS_GPU
    go(std::integral_constant<DdlaBackend, DdlaBackend::GPU>{});
#else
    go(std::integral_constant<DdlaBackend, DdlaBackend::CPU>{});
#endif
}

template <typename T>
void run_case(const ddla::DdlaHandle_t& handle, const CaseGeom& g, const std::string& tag)
{
    // Debug/sanitizer knobs: DDLA_QL_TEST_FILTER runs only the cases whose tag contains the
    // substring (every rank sees the same value, so the collective sequence stays uniform);
    // DDLA_QL_TEST_NOPOISON skips the 64 MB allocator-poison leg (racecheck hangs on it).
    if (const char* flt = std::getenv("DDLA_QL_TEST_FILTER"))
        if (*flt != '\0' && tag.find(flt) == std::string::npos) return;
    using rt = typename ddla::real_type<T>::type;
    const double eps = std::is_same<rt, float>::value ? 1.1920929e-7 : 2.2204460492503131e-16;
    const int m = g.m, n = g.n, ja = g.ja, k = std::min(m, n);
    const int jfirst_g = ja + n - k;       // first reflector column (global)
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
    MPI_Comm comm = ddlaGetCommunicator(handle);
    const int myid = ddlaGetRank(handle);

    int desc[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(desc, handle, g.dm, g.dn, g.mb, g.nb, g.rsrc, g.csrc));

    // tau: EXACTLY LOCc(ja + n) long where the rank owns columns of the block's
    // range (ASAN catches overruns), a null pointer on ranks that own no
    // reflector column (the library must accept it and never write).
    const int nloc_tau = ddla::num_loc(ja + n, g.nb, mypcol, g.csrc, npcols);
    const int nown = ddla::num_loc(ja + n, g.nb, mypcol, g.csrc, npcols)
                   - ddla::num_loc(jfirst_g, g.nb, mypcol, g.csrc, npcols);
    std::vector<T> tau(static_cast<std::size_t>(std::max(nloc_tau, 1)), tau_sentinel<T>());
    T* const tau_arg = (k > 0 && nown == 0) ? nullptr : tau.data();

    auto h_A = make_local<T>(handle, desc, [&](int i, int j) { return case_value<T>(g, i, j); });
    DeviceBuffer<T> d_A(handle, h_A.size());
    upload(handle, d_A.ptr, h_A);
    check_ddla_sync(handle);

    int info = -1;
    call_ql<T>(handle, g, d_A.ptr, desc, tau_arg, info);
    check_ddla_sync(handle);
    const auto local1 = download<T>(handle, d_A.ptr, h_A.size());
    int global_info = -1;
    MPI_Allreduce(&info, &global_info, 1, MPI_INT, MPI_MAX, comm);
    if (global_info != 0) {
        if (myid == 0) std::cerr << "pgeqlf info=" << global_info << " for " << tag << std::endl;
        MPI_Abort(comm, 1);
    }

    // ---- 1a. tau sentinels for local columns that carry no reflector ---------
    double tau_sent_err = 0.0;
    for (int l = 0; l < nloc_tau; ++l) {
        const int gj = ddla::indxl2g(l, g.nb, mypcol, g.csrc, npcols);
        if (gj < jfirst_g || gj >= ja + n) acc_max(tau_sent_err, dabs(tau[l] - tau_sentinel<T>()));
    }

    // ---- 1b. tau replicated bitwise across process rows -----------------------
    const int ncol_tau = ja + n;
    std::vector<T> tau_row0(static_cast<std::size_t>(std::max(ncol_tau, 1)), T(0));
    if (myprow == 0)
        for (int l = 0; l < nloc_tau; ++l)
            tau_row0[ddla::indxl2g(l, g.nb, mypcol, g.csrc, npcols)] = tau[l];
    allreduce_sum<T>(tau_row0.data(), ncol_tau, comm);
    double tau_repl_err = 0.0;
    for (int l = 0; l < nloc_tau; ++l) {
        const int gj = ddla::indxl2g(l, g.nb, mypcol, g.csrc, npcols);
        if (gj >= jfirst_g && gj < ja + n)
            acc_max(tau_repl_err, dabs(tau[l] - tau_row0[static_cast<std::size_t>(gj)]));
    }

    const auto full = gather_full<T>(handle, d_A.ptr, desc, g.dm, g.dn);
    auto gat = [&](int i, int j) -> const T& {
        return full[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * g.dm];
    };

    // ---- 1c. everything outside the block equals the input ---------------------
    double outside_err = 0.0;
    for (int j = 0; j < g.dn; ++j)
        for (int i = 0; i < g.dm; ++i)
            if (i >= m || j < ja || j >= ja + n)
                acc_max(outside_err, dabs(gat(i, j) - case_value<T>(g, i, j)));

    // ---- 2. against host ?geqlf --------------------------------------------------
    if (m > 0 && n > 0) {
        std::vector<T> a0(static_cast<std::size_t>(m) * n, T(0));
        double a0_fnorm = 0.0;
        long double a0_acc = 0.0L;   // long double: the squares of the FILL_HUGE entries overflow a double
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                const T v = case_value<T>(g, i, ja + j);
                a0[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m] = v;
                a0_acc += std::norm(std::complex<long double>(std::real(v), std::imag(v)));
            }
        a0_fnorm = static_cast<double>(std::sqrt(a0_acc));
        auto fb = [&](int i, int j) -> const T& { return gat(i, ja + j); };

        std::vector<T> host_a = a0;
        std::vector<T> host_tau(static_cast<std::size_t>(k), T(0));
        host_geqlf(host_a, m, n, host_tau);
        auto ha = [&](int i, int j) -> const T& {
            return host_a[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m];
        };

        // A pivot of exactly -0.0 (FILL_NEGZERO): LAPACK (gfortran SIGN) takes it as negative, the vendor
        // geqrf of the GPU arm as +0, so on the GPU the sign of L(m-1, n-1) -- and with it the whole
        // reflector chain -- may differ from the host reference.  There the factorization must still be
        // valid (A = Q L and Q^H Q = I below) with |L(m-1, n-1)| = ||x||; the CPU arm keeps the strict
        // comparison.
        // (The in-place panel runs our own scalar path, LAPACK's sign included: it stays strict.)
        const char* panel_env = std::getenv("DDLA_QR_PANEL");
        const bool inplace_panel = panel_env != nullptr && std::strcmp(panel_env, "inplace") == 0;
        const bool vendor_sign = g.fill == FILL_NEGZERO && ddlaGetBackend(handle) == DdlaBackend::GPU && !inplace_panel;
        const double scale_l = std::max(a0_fnorm, 1.0) * std::max(m, n) * eps;
        const double scale_v = std::max(m, n) * eps;
        double l_err = 0.0, v_err = 0.0, tau_err = 0.0;
        for (int j = 0; j < n; ++j) {
            const int piv = m - n + j;            // may be negative (update-only columns)
            for (int i = std::max(piv, 0); i < m; ++i) acc_max(l_err, dabs(fb(i, j) - ha(i, j)));
            for (int i = 0; i < std::min(piv, m); ++i) acc_max(v_err, dabs(fb(i, j) - ha(i, j)));
        }
        for (int jj = 0; jj < k; ++jj)
            acc_max(tau_err, dabs(tau_row0[static_cast<std::size_t>(jfirst_g + jj)]
                                  - host_tau[static_cast<std::size_t>(jj)]));
        // per-column pivot (L diagonal), relative
        double ldiag_err = 0.0;
        for (int j = n - k; j < n; ++j) {
            const int piv = m - n + j;
            const double ref = dabs(ha(piv, j));
            if (ref != 0.0) acc_max(ldiag_err, dabs(fb(piv, j) - ha(piv, j)) / ref);
        }
        if (vendor_sign) {
            const double ref = dabs(ha(m - 1, n - 1));
            ldiag_err = std::fabs(dabs(fb(m - 1, n - 1)) - ref) / ref;   // |L(m-1, n-1)| only
            require_close(handle, tag + " |L pivot|-rel (-0.0 pivot, vendor sign)", ldiag_err,
                          20.0 * std::sqrt(static_cast<double>(std::max(m, 1))) * eps);
        } else {
            require_close(handle, tag + " L-diag-rel", ldiag_err,
                          20.0 * std::sqrt(static_cast<double>(std::max(m, 1))) * eps);
        }

        // ---- 3. reference-free: Q = ?ungql(V, tau), A = Q L, Q^H Q = I ---------
        std::vector<T> lead(static_cast<std::size_t>(m) * k, T(0));
        for (int jj = 0; jj < k; ++jj)
            for (int i = 0; i < m; ++i)
                lead[static_cast<std::size_t>(i) + static_cast<std::size_t>(jj) * m] = fb(i, n - k + jj);
        std::vector<T> tauv(static_cast<std::size_t>(k));
        for (int jj = 0; jj < k; ++jj) tauv[jj] = tau_row0[static_cast<std::size_t>(jfirst_g + jj)];
        const std::vector<T> q = host_ungql<T>(lead, m, k, tauv);
        auto qat = [&](int i, int j) -> const T& {
            return q[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m];
        };
        // L(r, j) = factored(r, j) for r >= m - n + j; A = Q_thin * L(m-k : m, :).
        double ql_err = 0.0, orth_err = 0.0;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                T acc{};
                for (int p = 0; p < k; ++p) {
                    const int r = m - k + p;
                    const T lp = (r >= m - n + j) ? fb(r, j) : T(0);
                    acc += qat(i, p) * lp;
                }
                acc_max(ql_err, dabs(acc - a0[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m]));
            }
        }
        for (int j = 0; j < k; ++j)
            for (int i = 0; i < k; ++i) {
                T acc{};
                for (int p = 0; p < m; ++p) acc += conj_t(qat(p, i)) * qat(p, j);
                acc -= T(i == j ? rt(1) : rt(0));
                acc_max(orth_err, dabs(acc));
            }
        const double ql_scaled = ql_err / (std::max(a0_fnorm, 1.0) * std::max(m, n) * eps);
        const double orth_scaled = orth_err / (std::max(m, 1) * eps);
        require_close(handle, tag + " A-QL", ql_scaled, 10.0);
        require_close(handle, tag + " orth", orth_scaled, 10.0);
        if (!vendor_sign) {
            require_close(handle, tag + " L-vs-lapack", l_err / scale_l, 10.0);
            require_close(handle, tag + " V-vs-lapack", v_err / scale_v, 10.0);
            require_close(handle, tag + " tau-vs-lapack", tau_err / scale_v, 10.0);
        }

        // ---- negzero fill: alpha = -0.0 with x nonzero must give a positive pivot, as gfortran
        // LAPACK produces (-SIGN keeps the sign of -0.0).
        if (g.fill == FILL_NEGZERO && !vendor_sign) {
            const T b0 = fb(m - 1, n - 1);
            if (!(std::real(b0) > rt(0))) {
                if (myid == 0) std::cerr << "negzero: L(m-1, n-1) not positive: " << b0 << std::endl;
                MPI_Abort(comm, 1);
            }
        }

        // ---- structured case: pin the exactly-decided outcomes ------------------
        if (g.fill == FILL_STRUCTURED && n >= 3 && m >= n) {
            double pin_err = 0.0;
            const int j1 = n - 1, j2 = n - 2, j3 = n - 3;
            acc_max(pin_err, dabs(fb(m - 1, j1) - T(rt(-15))));            // alpha-only keeps LAPACK's sign
            acc_max(pin_err, dabs(tau_row0[static_cast<std::size_t>(ja + j1)]));
            acc_max(pin_err, dabs(tau_row0[static_cast<std::size_t>(ja + j3)]));
            for (int i = 0; i < m; ++i) acc_max(pin_err, dabs(fb(i, j3))); // the zero column stays zero
            if constexpr (is_cplx<T>::value) {
                const T b = fb(m - 2, j2);
                acc_max(pin_err, dabs(std::imag(b)));
                if (!(std::real(b) < rt(0))) {
                    if (myid == 0) std::cerr << "structured: complex alpha-only beta not real negative: " << b << std::endl;
                    MPI_Abort(comm, 1);
                }
            } else {
                acc_max(pin_err, dabs(tau_row0[static_cast<std::size_t>(ja + j2)]));
                acc_max(pin_err, dabs(fb(m - 2, j2) - T(rt(1.5))));
            }
            require_close(handle, tag + " pinned", pin_err, 0.0);
        }
    }

    require_close(handle, tag + " tau-sentinel", tau_sent_err, 0.0);
    require_close(handle, tag + " tau-replication", tau_repl_err, 0.0);
    require_close(handle, tag + " outside-block", outside_err, 0.0);

    // ---- bitwise repeat: poison the allocator pool, refactor from the same input
    // and require identical bits in A and tau (any read of unwritten scratch shows
    // up as a mismatch).
    {
        if (std::getenv("DDLA_QL_TEST_NOPOISON") == nullptr) {
            static const std::vector<unsigned char> poison_bytes(std::size_t(64) << 20, 0xFF);
            void* poison = nullptr;
            check_ddla_malloc(&poison, poison_bytes.size(), handle);
            check_ddla_memcpy(poison, poison_bytes.data(), poison_bytes.size(),
                              handle);
            check_ddla_free(poison, handle);
        }
#if defined(__GLIBC__)
        mallopt(M_PERTURB, 0xA5);   // perturb every malloc of the repeat run (CPU workspaces)
#endif
        upload(handle, d_A.ptr, h_A);
        std::vector<T> tau2(static_cast<std::size_t>(std::max(nloc_tau, 1)), tau_sentinel<T>());
        T* const tau2_arg = (k > 0 && nown == 0) ? nullptr : tau2.data();
        int info2 = -1;
        call_ql<T>(handle, g, d_A.ptr, desc, tau2_arg, info2);
        check_ddla_sync(handle);
#if defined(__GLIBC__)
        mallopt(M_PERTURB, 0);
#endif
        int global_info2 = -1;
        MPI_Allreduce(&info2, &global_info2, 1, MPI_INT, MPI_MAX, comm);
        if (global_info2 != 0) {
            if (myid == 0) std::cerr << "pgeqlf repeat info=" << global_info2 << " for " << tag << std::endl;
            MPI_Abort(comm, 1);
        }
        const auto local2 = download<T>(handle, d_A.ptr, h_A.size());
        double mism = 0.0;
        for (std::size_t idx = 0; idx < local1.size(); ++idx)
            if (std::memcmp(&local1[idx], &local2[idx], sizeof(T)) != 0) ++mism;
        for (std::size_t idx = 0; idx < tau.size(); ++idx)
            if (std::memcmp(&tau[idx], &tau2[idx], sizeof(T)) != 0) ++mism;
        require_close(handle, tag + " bitwise-repeat", mism, 0.0);
    }

    if (myid == 0)
        std::cout << "[grid " << grid_name(handle) << "] pgeqlf " << tag
                  << " ok" << (g.dm != ja + n || g.dn != ja + n ? " (padded)" : "") << std::endl;
}

std::string geom_tag(const CaseGeom& g)
{
    std::string s = std::to_string(g.m) + "x" + std::to_string(g.n);
    if (g.ja != 0) s += "@" + std::to_string(g.ja);
    s += "(" + std::to_string(g.mb) + "," + std::to_string(g.nb) + ")";
    return s;
}

// Backend dispatch for the new calls (DDLA_HAS_* are VALUE checks).
template <typename Fn>
void on_backend(const ddla::DdlaHandle_t& handle, Fn fn)
{
#if DDLA_HAS_CPU && DDLA_HAS_GPU
    if (ddlaGetBackend(handle) == DdlaBackend::CPU) fn(std::integral_constant<DdlaBackend, DdlaBackend::CPU>{});
    else fn(std::integral_constant<DdlaBackend, DdlaBackend::GPU>{});
#elif DDLA_HAS_GPU
    fn(std::integral_constant<DdlaBackend, DdlaBackend::GPU>{});
#else
    fn(std::integral_constant<DdlaBackend, DdlaBackend::CPU>{});
#endif
}

template <typename T>
struct RunOut {
    std::vector<T> a, tau;
    bool threw = false;
};

// One public call on a fresh copy of the input: returns the local A, the tau array and whether the
// call threw.  The exception is swallowed on every rank alike (argument errors are raised before any
// collective), so the grid stays in step.
template <typename T, typename Call>
RunOut<T> run_call(const ddla::DdlaHandle_t& handle, const std::vector<T>& h_A, int ntau, Call call)
{
    RunOut<T> r;
    DeviceBuffer<T> d_A(handle, h_A.size());
    upload(handle, d_A.ptr, h_A);
    check_ddla_sync(handle);
    r.tau.assign(static_cast<std::size_t>(std::max(ntau, 1)), tau_sentinel<T>());
    int info = -1;
    try { call(d_A.ptr, r.tau.data(), info); } catch (const std::exception&) { r.threw = true; }
    check_ddla_sync(handle);
    r.a = download<T>(handle, d_A.ptr, h_A.size());
    return r;
}

template <typename T>
double bits_differ(const RunOut<T>& x, const RunOut<T>& y)
{
    double d = (x.a.size() != y.a.size() || x.tau.size() != y.tau.size()) ? 1.0 : 0.0;
    for (std::size_t i = 0; d == 0.0 && i < x.a.size(); ++i)
        if (std::memcmp(&x.a[i], &y.a[i], sizeof(T)) != 0) d += 1.0;
    for (std::size_t i = 0; i < x.tau.size() && i < y.tau.size(); ++i)
        if (std::memcmp(&x.tau[i], &y.tau[i], sizeof(T)) != 0) d += 1.0;
    return d;
}

// The uplo dispatch: invalid flags throw (before any collective), lowercase equals uppercase bit for bit,
// and 'U' really differs from 'L'.
template <typename T>
void check_uplo_dispatch(const ddla::DdlaHandle_t& handle)
{
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
    const std::string tag = std::string(type_tag<T>()) + " uplo-dispatch" + g_case_tag_suffix;
    const int geoms[][4] = {{37, 6, 5, 8}, {41, 8, 3, 8}, {23, 7, 4, 16}};   // m, n, mb, nb
    for (const auto& gm : geoms) {
        const int m = gm[0], n = gm[1], mb = gm[2], nb = gm[3];
        int desc[ddla::DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(desc, handle, m, n, mb, nb, 0, 0));
        const CaseGeom g{m, n, 0, mb, nb, 0, 0, m, n, FILL_HASH};
        const int ntau = ddla::num_loc(n, nb, mypcol, 0, npcols);
        auto h_A = make_local<T>(handle, desc, [&](int i, int j) { return case_value<T>(g, i, j); });
        auto full = [&](char u) {
            return run_call<T>(handle, h_A, ntau, [&](T* dA, T* tau, int& info) {
                on_backend(handle, [&](auto bt) {
                    constexpr DdlaBackend B = decltype(bt)::value;
                    ddla::pgeqrf<B, T>(handle, u, m, n, dA, desc, tau, info);
                });
            });
        };
        const std::string t2 = tag + " " + std::to_string(m) + "x" + std::to_string(n) + "(" + std::to_string(mb)
                             + "," + std::to_string(nb) + ")";
        double missing_throw = 0.0;
        for (const char bad : {'X', ' ', 'N', 'A'})
            if (!full(bad).threw) missing_throw += 1.0;
        require_close(handle, t2 + " invalid-uplo-throws", missing_throw, 0.0);
        const RunOut<T> qU = full('U'), qL = full('L');
        require_close(handle, t2 + " valid-uplo-does-not-throw", (qU.threw || qL.threw) ? 1.0 : 0.0, 0.0);
        require_close(handle, t2 + " lowercase-u-pgeqrf", bits_differ(qU, full('u')), 0.0);
        require_close(handle, t2 + " lowercase-l-pgeqrf", bits_differ(qL, full('l')), 0.0);
        // 'U' and 'L' must really differ (a dispatch that ignored uplo would pass the checks above).
        // Judged over the whole grid: ranks that own no column of this narrow matrix hold empty data,
        // where both results are trivially identical.
        double differs = bits_differ(qU, qL) > 0.0 ? 1.0 : 0.0;
        MPI_Allreduce(MPI_IN_PLACE, &differs, 1, MPI_DOUBLE, MPI_MAX, ddlaGetCommunicator(handle));
        require_close(handle, t2 + " U-differs-from-L", differs > 0.0 ? 0.0 : 1.0, 0.0);
    }
}

// 1x1 grid, pgeqrf('L'): narrow and wide panels.  The panel is one block column, so the panel widths come
// from the descriptor nb (2 .. 256 below); every case runs the full run_case checks (LAPACK comparison,
// A = Q L, orthogonality, bitwise repeat).
template <typename T>
void check_ql_1x1(const ddla::DdlaHandle_t& handle)
{
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    if (nprows != 1 || npcols != 1) return;   // descriptor widths like these only make sense on one process
    //           m    n  ja mb nb rsrc csrc  dm   dn  fill
    const CaseGeom cases1x1[] = {
        {37,  17, 0, 5, 3, 0, 0,  37,  17, FILL_HASH},
        {37,  17, 0, 5, 2, 0, 0,  37,  17, FILL_HASH},   // 9 narrow panels
        {13,  29, 0, 3, 4, 0, 0,  13,  29, FILL_HASH},   // m < n: columns without reflectors
        {19,  20, 0, 4, 5, 0, 0,  19,  20, FILL_HASH},   // n not a multiple of the width
        {17,  13, 0, 3, 4, 0, 0,  17,  13, FILL_STRUCTURED},
        {21,  15, 0, 4, 5, 0, 0,  27,  20, FILL_HASH},   // padded descriptor
        {900, 24, 0, 8, 32, 0, 0, 900, 24, FILL_HASH},   // one panel
        {700, 400, 0, 8, 8, 0, 0, 700, 400, FILL_HASH},  // 50 panels
        {700, 400, 0, 8, 128, 0, 0, 700, 400, FILL_HASH},   // wide panels, the leftmost one partial
        {513, 700, 0, 8, 256, 0, 0, 513, 700, FILL_HASH},   // m < n, 3 wide panels
    };
    const std::string base_tag = std::string(type_tag<T>()) + " ql-1x1 ";
    for (const CaseGeom& g : cases1x1)
        run_case<T>(handle, g, base_tag + geom_tag(g) + fill_tag(g.fill) + g_case_tag_suffix);
}

template <typename T>
void check_pgeqlf(const ddla::DdlaHandle_t& handle)
{
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    const int pr = nprows - 1, pc = npcols - 1;

    //           m    n   ja  mb  nb rsrc csrc  dm   dn  fill
    const CaseGeom cases[] = {
        {23,  23,  0, 4,  4,  0,  0,  23,  23, FILL_HASH},   // square
        {37,  17,  0, 5,  3,  0,  0,  37,  17, FILL_HASH},   // tall, mb != nb
        {13,  29,  0, 3,  4,  0,  0,  13,  29, FILL_HASH},   // wide: update-only columns, k = m
        {21,  19,  0, 3,  5,  0,  0,  21,  19, FILL_HASH},
        {18,  14,  0, 5,  2,  0,  0,  18,  14, FILL_HASH},
        {9,    5,  0, 4,  6,  0,  0,   9,   5, FILL_HASH},   // n < nb: a single panel
        {20,  20,  0, 4,  4,  pr, pc, 20,  20, FILL_HASH},   // rsrc = csrc = P-1
        {21,  15,  0, 4,  5,  0,  0,  27,  20, FILL_HASH},   // padded descriptor
        {7,    9,  0, 2,  1,  0,  0,   7,   9, FILL_HASH},   // nb = 1
        {1,    5,  0, 2,  2,  0,  0,   1,   5, FILL_HASH},   // m = 1 (rows live on one process row)
        {5,    1,  0, 2,  2,  0,  0,   5,   1, FILL_HASH},   // n = 1
        {0,    5,  0, 2,  2,  0,  0,   0,   5, FILL_HASH},   // m = 0
        {5,    0,  0, 2,  2,  0,  0,   5,   0, FILL_HASH},   // n = 0
        {64,  48,  0, 8,  6,  0,  0,  64,  48, FILL_HASH},   // multi-cycle
        {320, 288, 0, 16, 16, 0,  0, 320, 288, FILL_HASH},   // long multi-panel pipeline
        {257, 300, 0, 8,  32, 0,  0, 257, 300, FILL_HASH},   // wide nb = 32
        {17,  13,  0, 3,  4,  0,  0,  17,  13, FILL_STRUCTURED},
        {257, 193, 0, 32, 32, 0,  0, 257, 193, FILL_HASH},
        {4500, 24, 0, 8,  8,  0,  0, 4500, 24, FILL_HASH},   // long panel columns
        {4500, 24, 0, 8,  8,  0,  0, 4500, 24, FILL_SUBNORMAL}, // denormal x across every process row
        {17,  13,  0, 3,  4,  0,  0,  17,  13, FILL_SUBNORMAL}, // knt rescale path in the first processed column
        {17,  13,  0, 3,  4,  0,  0,  17,  13, FILL_NEGZERO},   // alpha = -0.0 with x nonzero
        {17,  13,  0, 3,  4,  0,  0,  17,  13, FILL_TINY0},     // alpha = 0 under a tiny x: rescale with beta = -||x||
        {17,  13,  0, 3,  4,  0,  0,  17,  13, FILL_HUGE},      // x whose squares overflow: the norm must stay finite
        {41,  37,  0, 3,  8,  pr, pc, 41,  37, FILL_HASH},   // mb < jb with nonzero sources
        {40,  30,  0, 7,  4,  pr, 0,  40,  30, FILL_HASH},   // mb > nb: window ends cut mb blocks
        {11,  11,  0, 4,  2,  0,  0,  11,  11, FILL_HASH},   // late (shrinking) windows on ranks with no rows
        {9,    9,  0, 1,  3,  0,  0,   9,   9, FILL_HASH},   // mb = 1: the root rotates every row block
        {700, 300, 0, 16, 32, 0,  0, 700, 300, FILL_HASH},
    };
    for (const CaseGeom& g : cases)
        run_case<T>(handle, g, std::string(type_tag<T>()) + " " + geom_tag(g) + fill_tag(g.fill)
                                   + g_case_tag_suffix);
    check_uplo_dispatch<T>(handle);
    check_ql_1x1<T>(handle);
}

static void check_body(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    // The case list is grid-independent (every case builds its own descriptors):
    // run it ONCE per grid.
    static int seen_nprows = -1, seen_npcols = -1;
    int nprows_g = 0, npcols_g = 0;
    ddlaGetGridDims(handle, nprows_g, npcols_g);
    if (nprows_g == seen_nprows && npcols_g == seen_npcols) return;
    seen_nprows = nprows_g;
    seen_npcols = npcols_g;
    (void)base;
    check_pgeqlf<double>(handle);
    check_pgeqlf<Complex>(handle);
    check_pgeqlf<float>(handle);
    check_pgeqlf<std::complex<float>>(handle);

    // The same case list with the in-place panel (DDLA_QR_PANEL=inplace; the default panel is the
    // gathered one).  Every rank sets and restores the same value, so the ranks always agree.
    {
        const char* old_panel = std::getenv("DDLA_QR_PANEL");
        const std::string old_panel_val = old_panel != nullptr ? old_panel : "";
        const std::string save_tag_suffix = g_case_tag_suffix;
        setenv("DDLA_QR_PANEL", "inplace", 1);
        g_case_tag_suffix = " [inplace]";
        check_pgeqlf<double>(handle);
        check_pgeqlf<Complex>(handle);
        check_pgeqlf<float>(handle);
        check_pgeqlf<std::complex<float>>(handle);
        g_case_tag_suffix = save_tag_suffix;
        if (old_panel == nullptr) unsetenv("DDLA_QR_PANEL");
        else setenv("DDLA_QR_PANEL", old_panel_val.c_str(), 1);
    }

    // One case on a column-major grid handle.
    {
        ddla::DdlaHandle_t h_cm = nullptr;
        ddla::ddlaInit(h_cm);
        ddla::ddlaSet(h_cm, MPI_COMM_WORLD, nprows_g, npcols_g, 'C');
        const CaseGeom cm{23, 23, 0, 4, 4, 0, 0, 23, 23, FILL_HASH};
        run_case<double>(h_cm, cm, "colmajor-grid");
        ddla::ddlaDestroy(h_cm);
    }

#if DDLA_HAS_CPU && DDLA_HAS_GPU
    // Dual lane: the whole case list again on a CPU handle.
    if (ddlaGetBackend(handle) == DdlaBackend::GPU && ddlaBackendAvailable(DdlaBackend::CPU)) {
        ddla::DdlaHandle_t h_cpu = nullptr;
        ddla::ddlaInit(h_cpu, DdlaBackend::CPU);
        ddla::ddlaSet(h_cpu, MPI_COMM_WORLD, nprows_g, npcols_g, 'R');
        const std::string save = g_case_tag_suffix;
        g_case_tag_suffix = " [cpu-arm]";
        check_pgeqlf<double>(h_cpu);
        check_pgeqlf<Complex>(h_cpu);
        check_pgeqlf<float>(h_cpu);
        check_pgeqlf<std::complex<float>>(h_cpu);
        setenv("DDLA_QR_PANEL", "inplace", 1);
        g_case_tag_suffix = " [cpu-arm inplace]";
        check_pgeqlf<double>(h_cpu);
        check_pgeqlf<Complex>(h_cpu);
        check_pgeqlf<float>(h_cpu);
        check_pgeqlf<std::complex<float>>(h_cpu);
        unsetenv("DDLA_QR_PANEL");
        g_case_tag_suffix = save;
        ddla::ddlaDestroy(h_cpu);
    }
#endif
}

} // namespace

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pgeqrf_ql", check_body);
}
