#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
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
// Isolation test for the distributed Householder QR factorization pgeqrf
// (ScaLAPACK PxGEQRF semantics).
//
// For every case the factorization is checked three ways:
//   1. bitwise contract: tau replicated identically across process rows,
//      sentinel-preserved tau entries for global columns >= k, input
//      preserved outside the leading block (padded descriptors), info == 0;
//   2. against host LAPACK ?GEQRF on the same input: R, V, tau and the
//      per-column R diagonal (relative, so rows dropped from a panel norm
//      cannot hide behind the absolute scale);
//   3. reference-free with Q = ?UNGQR(our V, tau): A = Q R and Q^H Q = I.
//
// The structured case pins exactly-decided columns (alpha-only columns give
// tau == 0 with R keeping LAPACK's sign -- not ELPA's tau = 2 convention; a
// zero column; a complex alpha-only column with xnorm == 0; a tiny column
// that exercises ZLARFG's knt rescale loop and the carried-norm re-norm).
// ---------------------------------------------------------------------------

using namespace api_grid_test;

namespace {

// ---------------------------------------------------------------------------
// Host LAPACK prototypes (OpenBLAS; each consumer declares its own).
// ---------------------------------------------------------------------------
extern "C" {
void sgeqrf_(const int*, const int*, float*, const int*, float*, float*, const int*, int*);
void dgeqrf_(const int*, const int*, double*, const int*, double*, double*, const int*, int*);
void cgeqrf_(const int*, const int*, std::complex<float>*, const int*, std::complex<float>*,
             std::complex<float>*, const int*, int*);
void zgeqrf_(const int*, const int*, std::complex<double>*, const int*, std::complex<double>*,
             std::complex<double>*, const int*, int*);
void sorgqr_(const int*, const int*, const int*, float*, const int*, const float*, float*,
             const int*, int*);
void dorgqr_(const int*, const int*, const int*, double*, const int*, const double*, double*,
             const int*, int*);
void cungqr_(const int*, const int*, const int*, std::complex<float>*, const int*,
             const std::complex<float>*, std::complex<float>*, const int*, int*);
void zungqr_(const int*, const int*, const int*, std::complex<double>*, const int*,
             const std::complex<double>*, std::complex<double>*, const int*, int*);
}

template <typename T> struct is_cplx : std::false_type {};
template <typename U> struct is_cplx<std::complex<U>> : std::true_type {};

// Host conjugation overloads (std::conj of a real argument returns complex).
template <typename U> U conj_t(U v) { return v; }
template <typename U> std::complex<U> conj_t(std::complex<U> v) { return std::conj(v); }

// Double-valued abs for error accumulation (std::max(double, float) would
// not deduce).
template <typename U>
double dabs(const U& v) { return static_cast<double>(std::abs(v)); }

// NaN-blind error accumulator: std::max silently drops a NaN (every
// comparison against NaN is false), which would turn a poisoned run into a
// clean pass.  Once e is NaN it stays NaN.
void acc_max(double& e, double v)
{
    if (std::isnan(v) || v > e) e = v;
}

// Case-tag qualifier, set to " [cpu-arm]" while the dual lane re-runs the
// whole case list on a CPU handle (empty otherwise).
std::string g_case_tag_suffix;

template <typename T> const char* type_tag() { return "?"; }
template <> const char* type_tag<float>() { return "s"; }
template <> const char* type_tag<double>() { return "d"; }
template <> const char* type_tag<std::complex<float>>() { return "c"; }
template <> const char* type_tag<std::complex<double>>() { return "z"; }

// Sentinel for untouched tau entries (global columns >= k).
template <typename T> T tau_sentinel()
{
    if constexpr (is_cplx<T>::value) return T(-12345.0, 777.0);
    else return T(-12345.0);
}

// Deterministic hash-uniform fill in [-0.5, 0.5] (complex types get a second
// hash for the imaginary part).
template <typename T>
T hash_value(int i, int j, int tag)
{
    using rt = typename ddla::real_type<T>::type;
    const double s = std::sin(12.9898 * (i + 1) + 78.233 * (j + 1) + 3.17 * tag);
    const double re = s - std::floor(s) - 0.5;
    if constexpr (is_cplx<T>::value) {
        const double t = std::sin(4.711 * (i + 1) + 2.113 * (j + 1) + 1.7 * tag);
        return T(static_cast<rt>(re), static_cast<rt>(t - std::floor(t) - 0.5));
    } else {
        return T(static_cast<rt>(re));
    }
}

// Fill for the structured case: hash-uniform off the special columns, +-3 on
// the diagonal, with the exactly-decided columns of the plan's table.
//   col 0: alpha=2 only -> tau=0, R=2;
//   col 1: alpha=-15 only -> tau=0, R stays -15 (LAPACK, not ELPA's tau=2);
//   col 2 and col nb+1: all zero -> tau exactly 0;
//   col 3: complex alpha only (xnorm == 0, Im alpha != 0 -> the general
//          ZLARFG path with an empty effective x; real types use a positive
//          alpha, the only alpha-only general case they have is none);
//   col 4: tiny alpha + tiny x -> the knt rescale and the re-norm.
template <typename T>
T structured_value(int i, int j, int nb)
{
    using rt = typename ddla::real_type<T>::type;
    if (j == 0) return T(i == 0 ? rt(2) : rt(0));
    if (j == 1) return T(i == 1 ? rt(-15) : rt(0));
    if (j == 2) return T(0);
    if (j == 3) {
        if (i == 3) {
            if constexpr (is_cplx<T>::value) return T(rt(1.5), rt(-2.5));
            else return T(rt(1.5));
        }
        return T(0);   // xnorm == 0 below the alpha
    }
    if (j == 4) {
        const rt ts = std::is_same<rt, float>::value ? rt(1e-33) : rt(1e-300);
        if (i < 4) return T(0);
        if (i == 4) return T(ts);
        const double s = std::sin(5.31 * (i + 1) + 1.7 * (j + 1));
        return T(static_cast<rt>(ts * (0.25 + (s - std::floor(s)))));
    }
    if (j == nb + 1) return T(0);
    if (i == j) {
        const double s = std::sin(12.9898 * (i + 1) + 78.233 * (j + 1));
        return T(s > 0.0 ? rt(3) : rt(-3));
    }
    return hash_value<T>(i, j, 9);
}

// Gather a distributed matrix's full descriptor block: every rank ends up with
// the whole thing (zero-fill + MPI_Allreduce).
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
            full[gi + static_cast<std::size_t>(gj) * rows] =
                local[i + static_cast<std::size_t>(j) * lld];
        }
    }
    MPI_Comm comm = ddlaGetCommunicator(handle);
    if constexpr (std::is_same<T, float>::value) {
        MPI_Allreduce(MPI_IN_PLACE, full.data(), static_cast<int>(full.size()),
                      MPI_FLOAT, MPI_SUM, comm);
    } else if constexpr (std::is_same<T, double>::value) {
        MPI_Allreduce(MPI_IN_PLACE, full.data(), static_cast<int>(full.size()),
                      MPI_DOUBLE, MPI_SUM, comm);
    } else if constexpr (std::is_same<T, std::complex<float>>::value) {
        MPI_Allreduce(MPI_IN_PLACE, full.data(), static_cast<int>(full.size()),
                      MPI_C_FLOAT_COMPLEX, MPI_SUM, comm);
    } else {
        MPI_Allreduce(MPI_IN_PLACE, full.data(), static_cast<int>(full.size()),
                      MPI_C_DOUBLE_COMPLEX, MPI_SUM, comm);
    }
    return full;
}

template <typename T>
void host_geqrf(std::vector<T>& a, int m, int n, std::vector<T>& tau)
{
    T work_query[1];
    int lwork = -1, info = 0;
    if constexpr (std::is_same<T, float>::value) {
        sgeqrf_(&m, &n, a.data(), &m, tau.data(), work_query, &lwork, &info);
    } else if constexpr (std::is_same<T, double>::value) {
        dgeqrf_(&m, &n, a.data(), &m, tau.data(), work_query, &lwork, &info);
    } else if constexpr (std::is_same<T, std::complex<float>>::value) {
        cgeqrf_(&m, &n, a.data(), &m, tau.data(), work_query, &lwork, &info);
    } else {
        zgeqrf_(&m, &n, a.data(), &m, tau.data(), work_query, &lwork, &info);
    }
    lwork = std::max(1, static_cast<int>(std::real(work_query[0])));
    std::vector<T> work(lwork);
    if constexpr (std::is_same<T, float>::value) {
        sgeqrf_(&m, &n, a.data(), &m, tau.data(), work.data(), &lwork, &info);
    } else if constexpr (std::is_same<T, double>::value) {
        dgeqrf_(&m, &n, a.data(), &m, tau.data(), work.data(), &lwork, &info);
    } else if constexpr (std::is_same<T, std::complex<float>>::value) {
        cgeqrf_(&m, &n, a.data(), &m, tau.data(), work.data(), &lwork, &info);
    } else {
        zgeqrf_(&m, &n, a.data(), &m, tau.data(), work.data(), &lwork, &info);
    }
    if (info != 0) {
        std::cerr << "host ?geqrf failed, info=" << info << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Q (m x k, column-major lda m) from the factorization output's first k
// columns (V below the diagonal, implicit unit diagonal) and tau.
template <typename T>
std::vector<T> host_ungqr(const std::vector<T>& a_fact, int m, int k, const std::vector<T>& tau)
{
    std::vector<T> q(a_fact.begin(), a_fact.begin() + static_cast<std::size_t>(m) * k);
    T work_query[1];
    int lwork = -1, info = 0;
    if constexpr (std::is_same<T, float>::value) {
        sorgqr_(&m, &k, &k, q.data(), &m, tau.data(), work_query, &lwork, &info);
    } else if constexpr (std::is_same<T, double>::value) {
        dorgqr_(&m, &k, &k, q.data(), &m, tau.data(), work_query, &lwork, &info);
    } else if constexpr (std::is_same<T, std::complex<float>>::value) {
        cungqr_(&m, &k, &k, q.data(), &m, tau.data(), work_query, &lwork, &info);
    } else {
        zungqr_(&m, &k, &k, q.data(), &m, tau.data(), work_query, &lwork, &info);
    }
    lwork = std::max(1, static_cast<int>(std::real(work_query[0])));
    std::vector<T> work(lwork);
    if constexpr (std::is_same<T, float>::value) {
        sorgqr_(&m, &k, &k, q.data(), &m, tau.data(), work.data(), &lwork, &info);
    } else if constexpr (std::is_same<T, double>::value) {
        dorgqr_(&m, &k, &k, q.data(), &m, tau.data(), work.data(), &lwork, &info);
    } else if constexpr (std::is_same<T, std::complex<float>>::value) {
        cungqr_(&m, &k, &k, q.data(), &m, tau.data(), work.data(), &lwork, &info);
    } else {
        zungqr_(&m, &k, &k, q.data(), &m, tau.data(), work.data(), &lwork, &info);
    }
    if (info != 0) {
        std::cerr << "host ?ungqr failed, info=" << info << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return q;
}

// Backend dispatch: in a dual build both instantiations exist and the handle
// decides; single-backend lanes instantiate only what they can.  The config
// header DEFINES DDLA_HAS_* to 0/1, so these must be VALUE checks, not
// defined() checks.
template <typename T>
void call_pgeqrf(const ddla::DdlaHandle_t& handle, int m, int n,
                 T* d_A, const int* desc, T* tau, int& info)
{
#if DDLA_HAS_CPU && DDLA_HAS_GPU
    if (ddlaGetBackend(handle) == DdlaBackend::CPU)
        ddla::pgeqrf<DdlaBackend::CPU, T>(handle, 'U', m, n, d_A, desc, tau, info);
    else
        ddla::pgeqrf<DdlaBackend::GPU, T>(handle, 'U', m, n, d_A, desc, tau, info);
#elif DDLA_HAS_GPU
    ddla::pgeqrf<DdlaBackend::GPU, T>(handle, 'U', m, n, d_A, desc, tau, info);
#else
    ddla::pgeqrf<DdlaBackend::CPU, T>(handle, 'U', m, n, d_A, desc, tau, info);
#endif
}

struct CaseGeom {
    int m, n;          // problem size
    int mb, nb;        // block sizes
    int rsrc, csrc;    // first owning process row/column
    int dm, dn;        // descriptor dims (>= m, n; equal except for padding)
    int fill;          // 0: hash-uniform, 1: structured
};

enum { FILL_HASH = 0, FILL_STRUCTURED = 1, FILL_SUBNORMAL = 2, FILL_NEGZERO = 3,
       FILL_MIXED_SCALE = 4, FILL_TINY0 = 5, FILL_HUGE = 6 };

inline const char* fill_tag(int fill)
{
    switch (fill) {
    case FILL_STRUCTURED:  return " structured";
    case FILL_SUBNORMAL:   return " subnormal";
    case FILL_NEGZERO:     return " negzero";
    case FILL_MIXED_SCALE: return " mixed-scale";
    case FILL_TINY0:       return " tiny-zero-alpha";
    case FILL_HUGE:        return " huge";
    default:               return " hash";
    }
}

// Row-band height of the mixed-scale fill: round_up(max(256, ceil(m/128)), 256) rows.  The panel
// norms used to run over chunks of exactly this height (round 4); the fill keeps the layout it
// was written against, a pure property of the test now.
inline int test_chunk(int m)
{
    long long c = (static_cast<long long>(m) + 127) / 128;   // ceil(m/128)
    if (c < 256) c = 256;
    return static_cast<int>(((c + 255) / 256) * 256);
}

// Fill for the hash cases: the +-5 diagonal makes the matrix nonsingular and,
// for the small cases where entry-wise R/V/tau uniqueness matters, strictly
// diagonally dominant (off-diagonal entries in +-0.05).  At larger m the
// off-diagonal row sums exceed 5, so dominance no longer holds -- but the
// hash off-diagonals keep exact rank deficiency a probability-zero event,
// and the structured case covers the exactly-decided pivots separately.
// (A pure sin-hash fill without the diagonal made the 23x23 case numerically
// singular at two pivots: both factorizations were valid QRs -- our A = Q R
// held to 1e-15 with orthogonal Q -- but R/V/tau are not unique at rank
// deficiency, so entry-wise comparison against host ?GEQRF is meaningless
// there.)
template <typename T>
T case_value(const CaseGeom& g, int i, int j)
{
    using rt = typename ddla::real_type<T>::type;
    if (g.fill == FILL_STRUCTURED && i < g.m && j < g.n) return structured_value<T>(i, j, g.nb);
    if (g.fill == FILL_SUBNORMAL && j == 0 && i < g.m) {
        // Denormal first column (|entries| ~ 1e-310 (d) / 1e-40 (s)) so the
        // panel hits |beta| < safmin and the knt rescale + carried-norm
        // re-norm path; everything else stays the dominant hash fill.
        const rt ts = std::is_same<rt, float>::value ? rt(1e-40) : rt(1e-310);
        if (i == 0) return T(ts);
        const double s = std::sin(5.31 * (i + 1) + 1.7);
        const double u = 0.25 + (s - std::floor(s));
        if constexpr (is_cplx<T>::value) {
            const double t = std::sin(4.711 * (i + 1) + 0.31);
            return T(static_cast<rt>(ts * u), static_cast<rt>(ts * (0.25 + (t - std::floor(t)))));
        }
        return T(static_cast<rt>(ts * u));
    }
    if (g.fill == FILL_NEGZERO && i == 0 && j == 0) return T(-rt(0));
    if (g.fill == FILL_HUGE && j == 0 && i < g.m) {
        // entries whose squares overflow (>= 1e154 (d) / 1e19 (s)): the plain x^H x of a norm would
        // be Inf, the scale-safe norm is finite
        const rt hs = std::is_same<rt, float>::value ? rt(1e30) : rt(1e200);
        const double s = std::sin(5.31 * (i + 1) + 1.7);
        const double u = 0.25 + (s - std::floor(s));
        if constexpr (is_cplx<T>::value) {
            const double t = std::sin(4.711 * (i + 1) + 0.31);
            return T(static_cast<rt>(hs * u), static_cast<rt>(hs * (0.25 + (t - std::floor(t)))));
        }
        return T(static_cast<rt>(hs * u));
    }
    if (g.fill == FILL_TINY0 && j == 0 && i < g.m) {
        // alpha EXACTLY zero under a tiny (normal-range, below safmin) x: the xLARFG rescale
        // path with beta = -||x|| coming from the norm of x alone.
        if (i == 0) return T(0);
        const rt ts = std::is_same<rt, float>::value ? rt(1e-35) : rt(1e-300);
        const double s = std::sin(5.31 * (i + 1) + 1.7);
        const double u = 0.25 + (s - std::floor(s));
        if constexpr (is_cplx<T>::value) {
            const double t = std::sin(4.711 * (i + 1) + 0.31);
            return T(static_cast<rt>(ts * u), static_cast<rt>(ts * (0.25 + (t - std::floor(t)))));
        }
        return T(static_cast<rt>(ts * u));
    }
    if (g.fill == FILL_MIXED_SCALE && i < g.m) {
        // Row-band pattern (bands of test_chunk(m) rows):
        // band 0 tiny (~1e-300 (d) / 1e-35 (s)), band 1 exactly zero, bands
        // >= 2 the dominant fill below.  Even columns therefore span bands
        // of wildly different exponents plus a zero band; ODD columns are
        // tiny + zero + tiny -- no O(1) band anywhere -- so a norm that let
        // a zero band win the maximum exponent would collapse their
        // norms (tau = 0 / beta = alpha instead of -||x||), which the
        // relative R-diagonal and tau checks catch.  Every tiny value's hash
        // carries the COLUMN index: identical tiny columns would be parallel,
        // the later ones reduced to rounding noise by the earlier
        // reflectors, and the entry-wise comparison against host LAPACK
        // ill-posed (beta sign flips at 1e-315, exactly the rank-deficiency
        // lesson of the pure hash fill).  An odd column's diagonal is boosted
        // 8x over the band value so its pivot alpha dominates the reflector
        // update noise while staying comparable to ||x|| -- tau off the
        // exactly-1 trivial value, and a wrong norm still produces a wrong
        // reflector instead of a harmless no-op.
        const int c = test_chunk(g.m);
        const int band = i / c;
        if (band == 1) return T(0);
        const rt ts = std::is_same<rt, float>::value ? rt(1e-35) : rt(1e-300);
        const double s = std::sin(5.31 * (i + 1) + 1.7 * (j + 1));
        const double u = 0.25 + (s - std::floor(s));
        if constexpr (is_cplx<T>::value) {
            const double t = std::sin(4.711 * (i + 1) + 2.113 * (j + 1));
            if (i == j && (j % 2) != 0)
                return T(static_cast<rt>(ts * 8 * u), static_cast<rt>(ts * (0.25 + (t - std::floor(t)))));
            if (band == 0 || (j % 2) != 0)
                return T(static_cast<rt>(ts * u), static_cast<rt>(ts * (0.25 + (t - std::floor(t)))));
        } else {
            if (i == j && (j % 2) != 0) return T(static_cast<rt>(ts * 8 * u));
            if (band == 0 || (j % 2) != 0) return T(static_cast<rt>(ts * u));
        }
        // even column, band >= 2: fall through to the dominant fill
    }
    if (i == j) {
        if constexpr (is_cplx<T>::value) {
            // 5 * e^{i phi} with phi hashed from the row index: the diagonal
            // keeps its dominance (modulus 5) but is genuinely complex, so a
            // missing conjugation on tau or the reflector shows up instead of
            // cancelling against real arithmetic.
            const double s = std::sin(7.31 * (i + 1) + 1.13 * (j + 1) + 5.19);
            const double phi = 6.28318530717958648 * (s - std::floor(s));
            return T(static_cast<rt>(5.0 * std::cos(phi)),
                     static_cast<rt>(5.0 * std::sin(phi)));
        }
        const double s = std::sin(7.31 * (i + 1) + 1.13 * (j + 1));
        return T(s > 0.0 ? rt(5) : rt(-5));
    }
    return hash_value<T>(i, j, 3) * rt(0.1);
}

// One full pgeqrf case at the given geometry.
template <typename T>
void run_case(const ddla::DdlaHandle_t& handle, const CaseGeom& g, const std::string& tag)
{
    using rt = typename ddla::real_type<T>::type;
    const double eps = std::is_same<rt, float>::value ? 1.1920929e-7 : 2.2204460492503131e-16;
    const int m = g.m, n = g.n, k = std::min(m, n);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
    MPI_Comm comm = ddlaGetCommunicator(handle);
    const int myid = ddlaGetRank(handle);

    int desc[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(desc, handle, g.dm, g.dn, g.mb, g.nb, g.rsrc, g.csrc));
    const int lld = desc[ddla::DDLA_LLD_];

    // tau: EXACTLY LOCc(n) long where the rank owns columns (ASAN catches
    // overruns), one dummy element on ranks that own none -- they must never
    // see a write (the sentinel/replication loops below cover them
    // vacuously since they only index local columns).
    const int nloc_tau = ddla::num_loc(n, g.nb, mypcol, g.csrc, npcols);
    std::vector<T> tau(static_cast<std::size_t>(std::max(nloc_tau, 1)), tau_sentinel<T>());

    auto h_A = make_local<T>(handle, desc, [&](int i, int j) { return case_value<T>(g, i, j); });
    DeviceBuffer<T> d_A(handle, h_A.size());
    upload(handle, d_A.ptr, h_A);
    check_ddla_sync(handle);

    int info = -1;
    call_pgeqrf<T>(handle, m, n, d_A.ptr, desc, tau.data(), info);
    check_ddla_sync(handle);
    const auto local1 = download<T>(handle, d_A.ptr, h_A.size());
    int global_info = -1;
    MPI_Allreduce(&info, &global_info, 1, MPI_INT, MPI_MAX, comm);
    if (global_info != 0) {
        if (myid == 0) std::cerr << "pgeqrf info=" << global_info << " for " << tag << std::endl;
        MPI_Abort(comm, 1);
    }

    // ---- 1a. tau sentinels for global columns >= k --------------------------
    double tau_sent_err = 0.0;
    for (int l = 0; l < nloc_tau; ++l) {
        const int gj = ddla::indxl2g(l, g.nb, mypcol, g.csrc, npcols);
        if (gj >= k) acc_max(tau_sent_err, dabs(tau[l] - tau_sentinel<T>()));
    }

    // ---- 1b. tau replicated bitwise across process rows ---------------------
    // Build the myprow == 0 replica per global column, then require exact
    // equality (identical host arithmetic on identical allgathered bits) on
    // every process row that owns the column.
    std::vector<T> tau_row0(static_cast<std::size_t>(std::max(n, 0)), T(0));
    if (myprow == 0) {
        for (int l = 0; l < nloc_tau; ++l)
            tau_row0[ddla::indxl2g(l, g.nb, mypcol, g.csrc, npcols)] = tau[l];
    }
    if constexpr (std::is_same<T, float>::value) {
        MPI_Allreduce(MPI_IN_PLACE, tau_row0.data(), n, MPI_FLOAT, MPI_SUM, comm);
    } else if constexpr (std::is_same<T, double>::value) {
        MPI_Allreduce(MPI_IN_PLACE, tau_row0.data(), n, MPI_DOUBLE, MPI_SUM, comm);
    } else if constexpr (std::is_same<T, std::complex<float>>::value) {
        MPI_Allreduce(MPI_IN_PLACE, tau_row0.data(), n, MPI_C_FLOAT_COMPLEX, MPI_SUM, comm);
    } else {
        MPI_Allreduce(MPI_IN_PLACE, tau_row0.data(), n, MPI_C_DOUBLE_COMPLEX, MPI_SUM, comm);
    }
    double tau_repl_err = 0.0;
    for (int l = 0; l < nloc_tau; ++l) {
        const int gj = ddla::indxl2g(l, g.nb, mypcol, g.csrc, npcols);
        acc_max(tau_repl_err, dabs(tau[l] - tau_row0[static_cast<std::size_t>(gj)]));
    }

    const auto full = gather_full<T>(handle, d_A.ptr, desc, g.dm, g.dn);
    auto gat = [&](int i, int j) -> const T& {
        return full[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * g.dm];
    };

    // Everything from here to the structured pins reads only replicated data
    // (the gathered full block, tau_row0) plus host references rebuilt from
    // the case geometry -- identical O(n^3)-class work on every rank, which
    // is the dominant share of the check phase at the large case sizes.  Run
    // it on rank 0 only and reduce one verdict; the local tau checks (1a/1b)
    // above stay per-rank.
    RootSection dense(handle);
    if (dense.root) {
    // ---- 1c. everything outside the leading block equals the input ----------
    double outside_err = 0.0;
    for (int j = 0; j < g.dn; ++j)
        for (int i = 0; i < g.dm; ++i)
            if (i >= m || j >= n)
                acc_max(outside_err,
                        dabs(gat(i, j) - case_value<T>(g, i, j)));
    dense.check(tag + " outside-leading-block", outside_err, 0.0);

    // ---- 2. against host ?geqrf ---------------------------------------------
    double r_err = 0.0, v_err = 0.0, tau_err = 0.0, a0_fnorm = 0.0;
    long double a0_acc = 0.0L;   // long double: the squares of the FILL_HUGE entries overflow a double
    std::vector<T> a0(static_cast<std::size_t>(m) * n, T(0));
    if (m > 0 && n > 0) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                a0[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m] =
                    case_value<T>(g, i, j);
                const T& av = a0[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m];
                a0_acc += std::norm(std::complex<long double>(std::real(av), std::imag(av)));
            }
        a0_fnorm = static_cast<double>(std::sqrt(a0_acc));

        std::vector<T> host_a = a0;
        std::vector<T> host_tau(static_cast<std::size_t>(k), T(0));
        if (k > 0) host_geqrf(host_a, m, n, host_tau);

        // A pivot of exactly -0.0 (FILL_NEGZERO): LAPACK (gfortran SIGN) takes it as negative, the vendor
        // geqrf of the GPU arm as +0, so on the GPU the sign of R(0,0) -- and with it the whole reflector
        // chain -- may differ from the host reference.  There the factorization must still be valid
        // (A = Q R and Q^H Q = I below) with |R(0,0)| = ||x||; the CPU arm keeps the strict comparison.
        // (The in-place panel runs our own scalar path, LAPACK's sign included: it stays strict.)
        const char* panel_env = std::getenv("DDLA_QR_PANEL");
        const bool inplace_panel = panel_env != nullptr && std::strcmp(panel_env, "inplace") == 0;
        const bool vendor_sign = g.fill == FILL_NEGZERO && ddlaGetBackend(handle) == DdlaBackend::GPU && !inplace_panel;
        const double scale_r = std::max(a0_fnorm, 1.0) * std::max(m, n) * eps;
        const double scale_v = std::max(m, n) * eps;
        for (int j = 0; j < n; ++j) {
            const int iend = std::min(j, m - 1);   // inclusive: upper triangle
            for (int i = 0; i <= iend; ++i)
                acc_max(r_err, dabs(gat(i, j) - host_a[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m]));
            for (int i = j + 1; i < m; ++i)
                if (j < k)
                    acc_max(v_err, dabs(gat(i, j) - host_a[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m]));
        }
        for (int j = 0; j < k; ++j)
            acc_max(tau_err, dabs(tau_row0[static_cast<std::size_t>(j)] - host_tau[static_cast<std::size_t>(j)]));

        // ---- per-column R diagonal, relative: rows dropped from a panel
        // norm (a lost block boundary in the gather) shift beta by their
        // share of ||x||, which the absolute r_err scale (10*max(m,n)*eps of
        // ||A||) can hide for float types at large m.  Exactly-zero pivots
        // (the structured case pins those separately) are skipped.
        double rdiag_err = 0.0;
        for (int j = 0; j < k && j < m; ++j) {
            const double ref =
                dabs(host_a[static_cast<std::size_t>(j) + static_cast<std::size_t>(j) * m]);
            if (ref != 0.0)
                acc_max(rdiag_err,
                        dabs(gat(j, j) - host_a[static_cast<std::size_t>(j) + static_cast<std::size_t>(j) * m]) / ref);
        }
        if (vendor_sign) {
            const double ref = dabs(host_a[0]);
            rdiag_err = std::fabs(dabs(gat(0, 0)) - ref) / ref;   // |R(0,0)| only
            dense.check(tag + " |R00|-rel (-0.0 pivot, vendor sign)", rdiag_err,
                          20.0 * std::sqrt(static_cast<double>(std::max(m, 1))) * eps);
        } else {
            dense.check(tag + " R-diag-rel", rdiag_err,
                          20.0 * std::sqrt(static_cast<double>(std::max(m, 1))) * eps);
        }

        // ---- subnormal fill: tau[0] and V(:,0) within 4 eps of host LAPACK
        // (relative), guarding the knt rescale + carried-norm re-norm path
        // end to end against the reference implementation.  Only pinned for
        // small m: two independently rounded robust norms of one vector
        // already drift by O(sqrt(m) * eps) relative, which at the long
        // columns exceeds 4 eps however correct both are (the
        // general R/V/tau checks below carry those cases).
        if (g.fill == FILL_SUBNORMAL && g.m <= 100) {
            double sub_err = dabs(tau_row0[0] - host_tau[0]) / std::max(1.0, dabs(host_tau[0]));
            for (int i = 1; i < m; ++i) {
                const T hv = host_a[static_cast<std::size_t>(i)];   // column 0, lda = m
                const T ov = gat(i, 0);
                const double den = dabs(hv) != 0.0 ? dabs(hv) : 1.0;
                acc_max(sub_err, dabs(ov - hv) / den);
            }
            dense.check(tag + " subnormal-4eps", sub_err / (4.0 * eps), 1.0);
        }
        // ---- negzero fill: alpha = -0.0 with x nonzero must give a positive
        // R(0,0), as gfortran LAPACK produces (-SIGN keeps the sign of -0.0).
        if (g.fill == FILL_NEGZERO && !vendor_sign) {
            const T b0 = gat(0, 0);
            if (!(std::real(b0) > rt(0))) {
                if (myid == 0)
                    std::cerr << "negzero col 0: R(0,0) not positive: " << b0 << std::endl;
                MPI_Abort(comm, 1);
            }
        }

        // ---- 3. reference-free with Q = ?ungqr(our V, tau) -------------------
        // Extract the leading m x k block with the gathered dm stride (the
        // descriptor may be padded, so full is NOT a dense m-column matrix).
        std::vector<T> lead(static_cast<std::size_t>(m) * k, T(0));
        for (int j = 0; j < k; ++j)
            for (int i = 0; i < m; ++i)
                lead[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m] = gat(i, j);
        const std::vector<T> q = host_ungqr<T>(
            lead, m, k, std::vector<T>(tau_row0.begin(), tau_row0.begin() + k));
        auto qat = [&](int i, int j) -> const T& {
            return q[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m];
        };
        // QR over the full n columns: R(i,j) = factored(i,j) for i <= j.
        double qr_err = 0.0, orth_err = 0.0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                T acc{};
                for (int p = 0; p < k; ++p) {
                    const T rp = (p <= j) ? gat(p, j) : T(0);
                    acc += qat(i, p) * rp;
                }
                acc_max(qr_err, dabs(acc - a0[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m]));
            }
        for (int j = 0; j < k; ++j)
            for (int i = 0; i < k; ++i) {
                T acc{};
                for (int p = 0; p < m; ++p)
                    acc += conj_t(qat(p, i)) * qat(p, j);
                acc -= T(i == j ? rt(1) : rt(0));
                acc_max(orth_err, dabs(acc));
            }
        const double qr_scaled = qr_err / (std::max(a0_fnorm, 1.0) * std::max(m, n) * eps);
        const double orth_scaled = orth_err / (std::max(m, 1) * eps);
        dense.check(tag + " A-QR", qr_scaled, 10.0);
        dense.check(tag + " orth", orth_scaled, 10.0);
        if (!vendor_sign) {
            dense.check(tag + " R-vs-lapack", r_err / scale_r, 10.0);
            dense.check(tag + " V-vs-lapack", v_err / scale_v, 10.0);
            dense.check(tag + " tau-vs-lapack", tau_err / scale_v, 10.0);
        }
    }

    // ---- structured case: pin the exactly-decided outcomes ------------------
    if (g.fill == FILL_STRUCTURED) {
        double pin_err = 0.0;
        acc_max(pin_err, dabs(gat(0, 0) - T(rt(2))));
        acc_max(pin_err, dabs(gat(1, 1) - T(rt(-15))));
        acc_max(pin_err, dabs(gat(2, 2)));
        acc_max(pin_err, dabs(gat(g.nb + 1, g.nb + 1)));
        acc_max(pin_err, dabs(tau_row0[0]));
        acc_max(pin_err, dabs(tau_row0[1]));
        acc_max(pin_err, dabs(tau_row0[2]));
        acc_max(pin_err, dabs(tau_row0[g.nb + 1]));
        if constexpr (is_cplx<T>::value) {
            // complex alpha-only column: beta real negative (general ZLARFG
            // path with xnorm == 0)
            const T b3 = gat(3, 3);
            acc_max(pin_err, dabs(std::imag(b3)));
            if (!(std::real(b3) < rt(0))) {
                if (myid == 0)
                    std::cerr << "structured col 3: beta not real negative: " << b3 << std::endl;
                MPI_Abort(comm, 1);
            }
        }
        {
            // tiny column: beta restored to a negative tiny
            const T b4 = gat(4, 4);
            const rt bound = std::is_same<rt, float>::value ? rt(1e-32) : rt(1e-299);
            if (!(std::real(b4) < rt(0) && dabs(b4) < static_cast<double>(bound))) {
                if (myid == 0)
                    std::cerr << "structured col 4: beta not negative tiny: " << b4 << std::endl;
                MPI_Abort(comm, 1);
            }
        }
        dense.check(tag + " pinned", pin_err, 0.0);
    }
    }
    dense.finish();

    require_close(handle, tag + " tau-sentinel", tau_sent_err, 0.0);
    require_close(handle, tag + " tau-replication", tau_repl_err, 0.0);

    // ---- bitwise repeat: poison the allocator pool, refactor again from the
    // same input and require identical bits in A and tau.  Any read of
    // unwritten scratch (the device-resident panel keeps no host mirrors)
    // surfaces as a mismatch, because the poisoned 0xFF fill differs from
    // whatever a stale/garbage read would produce run to run.
    {
        static const std::vector<unsigned char> poison_bytes(std::size_t(64) << 20, 0xFF);
        void* poison = nullptr;
        check_ddla_malloc(&poison, poison_bytes.size(), handle);
        check_ddla_memcpy(poison, poison_bytes.data(), poison_bytes.size(),
                          handle);
        check_ddla_free(poison, handle);

#if defined(__GLIBC__)
        // The 0xFF poison above cannot reach the CPU backend's host
        // workspaces: glibc recycles the freed run-1 allocations with run-1's
        // correct contents, so an unwritten-scratch read would compare
        // equal.  M_PERTURB makes every malloc of the repeat run return
        // perturb-filled memory instead (reset afterwards for the other
        // cases).  Two glibc caveats: the perturb byte is 0x5A (NOT the
        // requested value -- do not test for 0xA5 in memory), and the per-
        // thread tcache serves allocations <= 1032 B WITHOUT perturbation,
        // so the ctest ENVIRONMENT must set
        // GLIBC_TUNABLES=glibc.malloc.tcache_count=0 for this test.
        mallopt(M_PERTURB, 0xA5);
#endif

        upload(handle, d_A.ptr, h_A);
        std::vector<T> tau2(static_cast<std::size_t>(std::max(nloc_tau, 1)), tau_sentinel<T>());
        int info2 = -1;
        call_pgeqrf<T>(handle, m, n, d_A.ptr, desc, tau2.data(), info2);
        check_ddla_sync(handle);
#if defined(__GLIBC__)
        mallopt(M_PERTURB, 0);
#endif
        int global_info2 = -1;
        MPI_Allreduce(&info2, &global_info2, 1, MPI_INT, MPI_MAX, comm);
        if (global_info2 != 0) {
            if (myid == 0) std::cerr << "pgeqrf repeat info=" << global_info2 << " for " << tag << std::endl;
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
        std::cout << "[grid " << grid_name(handle) << "] pgeqrf " << tag << " ok"
                  << (g.dm != g.m || g.dn != g.n ? " (padded)" : "") << std::endl;
}

template <typename T>
void check_pgeqrf(const ddla::DdlaHandle_t& handle)
{
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    const int pr = nprows - 1, pc = npcols - 1;

    const CaseGeom cases[] = {
        {23, 23, 4, 4, 0, 0, 23, 23, FILL_HASH},          // square
        {37, 17, 5, 3, 0, 0, 37, 17, FILL_HASH},          // tall, mb != nb
        {13, 29, 3, 4, 0, 0, 13, 29, FILL_HASH},          // wide, columns beyond k updated
        {21, 19, 3, 5, 0, 0, 21, 19, FILL_HASH},          // mb=3, nb=5
        {18, 14, 5, 2, 0, 0, 18, 14, FILL_HASH},          // mb=5, nb=2
        {9, 5, 4, 6, 0, 0, 9, 5, FILL_HASH},              // n < nb
        {20, 20, 4, 4, pr, pc, 20, 20, FILL_HASH},        // rsrc = csrc = P-1
        {21, 15, 4, 5, 0, 0, 27, 20, FILL_HASH},          // padded descriptor
        {7, 9, 2, 1, 0, 0, 7, 9, FILL_HASH},              // nb = 1
        {1, 5, 2, 2, 0, 0, 1, 5, FILL_HASH},              // m = 1 (a process row owns no rows)
        {5, 1, 2, 2, 0, 0, 5, 1, FILL_HASH},              // n = 1
        {0, 5, 2, 2, 0, 0, 0, 5, FILL_HASH},              // m = 0
        {5, 0, 2, 2, 0, 0, 5, 0, FILL_HASH},              // n = 0
        {64, 48, 8, 6, 0, 0, 64, 48, FILL_HASH},          // multi-cycle
        {320, 288, 16, 16, 0, 0, 320, 288, FILL_HASH},    // long multi-panel pipeline
        {257, 300, 8, 32, 0, 0, 257, 300, FILL_HASH},     // wide nb=32 panels, m != n
        {17, 13, 3, 4, 0, 0, 17, 13, FILL_STRUCTURED},    // exactly-decided columns
        {17, 13, 3, 4, 0, 0, 17, 13, FILL_SUBNORMAL},     // denormal column: knt + carried-norm re-norm
        {17, 13, 3, 4, 0, 0, 17, 13, FILL_NEGZERO},       // alpha = -0.0 with x nonzero
        {17, 13, 3, 4, 0, 0, 17, 13, FILL_TINY0},         // alpha = 0 under a tiny x: rescale with beta = -||x||
        {17, 13, 3, 4, 0, 0, 17, 13, FILL_HUGE},          // x whose squares overflow: the norm must stay finite
        {257, 193, 32, 32, 0, 0, 257, 193, FILL_HASH},    // mb = nb = 32, m just past 2^8
        {4500, 24, 8, 8, 0, 0, 4500, 24, FILL_HASH},      // long panel columns, many block rows
        {4500, 24, 8, 8, 0, 0, 4500, 24, FILL_SUBNORMAL}, // denormal x across block rows
        {900, 24, 8, 32, 0, 0, 900, 24, FILL_MIXED_SCALE},      // one panel: tiny+zero+O(1) and tiny+zero+tiny columns
        {1000, 300, 8, 300, 0, 0, 1000, 300, FILL_HASH},      // nb=300: one very wide panel
        {512, 256, 16, 256, 0, 0, 512, 256, FILL_HASH},       // panel = 256 columns of 512 rows
        {512, 512, 16, 512, 0, 0, 512, 512, FILL_HASH},       // jb == n == m_w: the last column has no x and no columns to update
        {41, 37, 3, 8, pr, pc, 41, 37, FILL_HASH},        // mb < jb with nonzero sources
        {40, 30, 7, 4, pr, 0, 40, 30, FILL_HASH},         // mb > nb, panel starts cut mb blocks
        {11, 11, 4, 2, 0, 0, 11, 11, FILL_HASH},          // late panels on ranks with no rows
        {9, 9, 1, 3, 0, 0, 9, 9, FILL_HASH},              // mb = 1: the root rotates every column
    };

    for (const CaseGeom& g : cases)
        run_case<T>(handle, g, std::string(type_tag<T>()) + " " + std::to_string(g.m) + "x" +
                                   std::to_string(g.n) + fill_tag(g.fill) + g_case_tag_suffix);
}

// ---------------------------------------------------------------------------
// Wide panels on a 1x1 grid.  The panel is one block column of the descriptor, so a wide panel is a wide
// descriptor nb: several wide panels (the last one partial), m < n, and nb = 2048, which pins the
// qr_build_T tau-shared->global fallback on a z handle (2*2048*16 B = 64 KiB > every device's 48 KiB
// default cap).  Each case runs the full run_case checks.
// ---------------------------------------------------------------------------
template <typename T>
void check_pgeqrf_1x1(const ddla::DdlaHandle_t& handle)
{
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    if (nprows != 1 || npcols != 1) return;   // descriptor widths like these only make sense on one process

    const CaseGeom cases1x1[] = {
        {37, 17, 5, 3, 0, 0, 37, 17, FILL_HASH},      // m > n
        {13, 29, 3, 4, 0, 0, 13, 29, FILL_HASH},      // m < n (columns beyond k)
        {19, 20, 4, 5, 0, 0, 19, 20, FILL_HASH},      // n not a multiple of the width
        {17, 13, 3, 4, 0, 0, 17, 13, FILL_STRUCTURED},
        {17, 13, 3, 4, 0, 0, 17, 13, FILL_SUBNORMAL},
        {513, 1024, 8, 1024, 0, 0, 513, 1024, FILL_HASH},  // one panel of width 1024
        {513, 2048, 8, 2048, 0, 0, 513, 2048, FILL_HASH},  // jb = 2048: build_T smem fallback (z especially)
        {900, 24, 8, 32, 0, 0, 900, 24, FILL_MIXED_SCALE},
        {700, 400, 8, 128, 0, 0, 700, 400, FILL_HASH},     // 4 panels, the last one partial
        {513, 700, 8, 256, 0, 0, 513, 700, FILL_HASH},     // 3 panels, m < n columns beyond k
        {700, 400, 8, 8, 0, 0, 700, 400, FILL_HASH},       // 50 narrow panels
    };
    for (const CaseGeom& cg : cases1x1)
        run_case<T>(handle, cg, std::string(type_tag<T>()) + " " + std::to_string(cg.m) + "x" + std::to_string(cg.n)
                                    + "/nb" + std::to_string(cg.nb) + fill_tag(cg.fill) + g_case_tag_suffix);
}


// DDLA_QR_PANEL selects the panel variant: unset == "gather" bitwise, and "inplace" must really run the
// other panel (different rounding: the factored matrix differs bitwise from the gathered one).
template <typename T>
void check_panel_switch(const ddla::DdlaHandle_t& handle)
{
    const CaseGeom g{64, 48, 8, 6, 0, 0, 64, 48, FILL_HASH};
    auto factor = [&](const char* mode) {
        if (mode == nullptr) unsetenv("DDLA_QR_PANEL"); else setenv("DDLA_QR_PANEL", mode, 1);
        int desc[ddla::DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(desc, handle, g.dm, g.dn, g.mb, g.nb, g.rsrc, g.csrc));
        int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
        ddlaGetGridDims(handle, nprows, npcols);
        ddlaGetGridCoords(handle, myprow, mypcol);
        std::vector<T> tau(static_cast<std::size_t>(std::max(1, ddla::num_loc(g.n, g.nb, mypcol, g.csrc, npcols))), T(0));
        auto h_A = make_local<T>(handle, desc, [&](int i, int j) { return case_value<T>(g, i, j); });
        DeviceBuffer<T> d_A(handle, h_A.size());
        upload(handle, d_A.ptr, h_A);
        check_ddla_sync(handle);
        int info = -1;
        call_pgeqrf<T>(handle, g.m, g.n, d_A.ptr, desc, tau.data(), info);
        check_ddla_sync(handle);
        return download<T>(handle, d_A.ptr, h_A.size());
    };
    const char* old = std::getenv("DDLA_QR_PANEL");
    const std::string old_val = old != nullptr ? old : "";
    const auto a_def = factor(nullptr), a_gat = factor("gather"), a_in = factor("inplace");
    if (old == nullptr) unsetenv("DDLA_QR_PANEL"); else setenv("DDLA_QR_PANEL", old_val.c_str(), 1);
    double same_def = 0.0, ndiff = 0.0;
    for (std::size_t k = 0; k < a_def.size(); ++k) {
        if (std::memcmp(&a_def[k], &a_gat[k], sizeof(T)) != 0) same_def = 1.0;
        if (std::memcmp(&a_def[k], &a_in[k], sizeof(T)) != 0) ndiff += 1.0;
    }
    MPI_Comm comm = ddlaGetCommunicator(handle);
    MPI_Allreduce(MPI_IN_PLACE, &ndiff, 1, MPI_DOUBLE, MPI_SUM, comm);
    const std::string tag = std::string(type_tag<T>()) + " panel-switch" + g_case_tag_suffix;
    require_close(handle, tag + " default==gather", same_def, 0.0);
    require_close(handle, tag + " inplace-differs", ndiff > 0.0 ? 0.0 : 1.0, 0.0);
}

static void check_body(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    // The case list is grid-independent (every case builds its own
    // descriptors), so run it ONCE per grid: the default second shape would
    // double the runtime and time the sanitizers out.
    static int seen_nprows = -1, seen_npcols = -1;
    int nprows_g = 0, npcols_g = 0;
    ddlaGetGridDims(handle, nprows_g, npcols_g);
    if (nprows_g == seen_nprows && npcols_g == seen_npcols) return;
    seen_nprows = nprows_g;
    seen_npcols = npcols_g;
    (void)base;
    check_pgeqrf<double>(handle);
    check_pgeqrf<Complex>(handle);
    check_pgeqrf<float>(handle);
    check_pgeqrf<std::complex<float>>(handle);
    check_pgeqrf_1x1<double>(handle);
    check_pgeqrf_1x1<Complex>(handle);
    check_pgeqrf_1x1<float>(handle);
    check_pgeqrf_1x1<std::complex<float>>(handle);
    check_panel_switch<double>(handle);
    check_panel_switch<Complex>(handle);

    // The same case lists with the in-place panel (DDLA_QR_PANEL=inplace; the default panel is the
    // gathered one).  Every rank sets and restores the same value, so the ranks always agree.
    {
        const char* old_panel = std::getenv("DDLA_QR_PANEL");
        const std::string old_panel_val = old_panel != nullptr ? old_panel : "";
        const std::string save_tag_suffix = g_case_tag_suffix;
        setenv("DDLA_QR_PANEL", "inplace", 1);
        g_case_tag_suffix = " [inplace]";
        check_pgeqrf<double>(handle);
        check_pgeqrf<Complex>(handle);
        check_pgeqrf<float>(handle);
        check_pgeqrf<std::complex<float>>(handle);
        check_pgeqrf_1x1<double>(handle);
        check_pgeqrf_1x1<Complex>(handle);
        check_pgeqrf_1x1<float>(handle);
        check_pgeqrf_1x1<std::complex<float>>(handle);
        g_case_tag_suffix = save_tag_suffix;
        if (old_panel == nullptr) unsetenv("DDLA_QR_PANEL");
        else setenv("DDLA_QR_PANEL", old_panel_val.c_str(), 1);
    }

    // One case on a column-major grid handle (whatever the build's default
    // backend is).
    {
        int nprows = 0, npcols = 0;
        ddlaGetGridDims(handle, nprows, npcols);
        ddla::DdlaHandle_t h_cm = nullptr;
        ddla::ddlaInit(h_cm);
        ddla::ddlaSet(h_cm, MPI_COMM_WORLD, nprows, npcols, 'C');
        const CaseGeom cm{23, 23, 4, 4, 0, 0, 23, 23, FILL_HASH};
        run_case<double>(h_cm, cm, "colmajor-grid");
        ddla::ddlaDestroy(h_cm);
    }

#if DDLA_HAS_CPU && DDLA_HAS_GPU
    // Dual lane: the whole case list again on a CPU handle, so the CPU arm
    // gets real coverage in the dual configuration too.  The case tags are
    // backend-qualified so the two lanes' log lines cannot be confused (the
    // per-case outputs would otherwise be indistinguishable).
    if (ddlaGetBackend(handle) == DdlaBackend::GPU && ddlaBackendAvailable(DdlaBackend::CPU)) {
        int nprows = 0, npcols = 0;
        ddlaGetGridDims(handle, nprows, npcols);
        ddla::DdlaHandle_t h_cpu = nullptr;
        ddla::ddlaInit(h_cpu, DdlaBackend::CPU);
        ddla::ddlaSet(h_cpu, MPI_COMM_WORLD, nprows, npcols, 'R');
        const std::string save_tag_suffix = g_case_tag_suffix;
        g_case_tag_suffix = " [cpu-arm]";
        check_pgeqrf<double>(h_cpu);
        check_pgeqrf<Complex>(h_cpu);
        check_pgeqrf<float>(h_cpu);
        check_pgeqrf<std::complex<float>>(h_cpu);
        // ... and with the in-place panel on the CPU arm
        setenv("DDLA_QR_PANEL", "inplace", 1);
        g_case_tag_suffix = " [cpu-arm inplace]";
        check_pgeqrf<double>(h_cpu);
        check_pgeqrf<Complex>(h_cpu);
        check_pgeqrf<float>(h_cpu);
        check_pgeqrf<std::complex<float>>(h_cpu);
        unsetenv("DDLA_QR_PANEL");
        g_case_tag_suffix = save_tag_suffix;
        ddla::ddlaDestroy(h_cpu);
    }
#endif
}

} // namespace

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pgeqrf", check_body);
}
