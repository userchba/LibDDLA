// ---------------------------------------------------------------------------
// Host unit probe for the scale-safe norm helpers shared by pgebrd's CPU arm and the QR scalar
// path (src/pgeqrf_kernels.h): qr_norm_scale and qr_norm_pair_serial.  CPU-runnable on every
// backend (the GPU kernels share the same host/device functions).
//
// qr_norm_scale: its two factors must be NORMAL numbers (a subnormal factor would be flushed to
// zero by any FTZ setting) whose exponents add up to -e exactly.
//
// qr_norm_pair_serial: the two-pass norm (n = ||x||, ns = ||rsafmn x||) against a long double
// reference on zero, denormal, huge and mixed-exponent vectors, within 4 ulp (n of a subnormal norm: one
// denorm_min), and the NaN / Inf classification (NaN beats Inf, Inf beats everything finite).
//
// Registered as an MPI test because every test in this suite is an MPI program, but it needs no
// grid, communicator work, or GPU: every rank redundantly checks the same math and every rank
// must exit 0.
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <mpi.h>

#include "QR/pgeqrf_kernels.h"

namespace {

int g_failures = 0;

void fail(const std::string& what)
{
    std::printf("FAIL: %s\n", what.c_str());
    ++g_failures;
}

// %g-style formatting (std::to_string gives 6 fixed decimals, unreadable for 1e-300-scale values).
template <typename rt>
std::string g_fmt(rt v)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%g", static_cast<double>(v));
    return std::string(buf);
}

// Relative error of a against the long double reference r, in units of the type's epsilon.
template <typename rt>
long double rel_eps(rt a, long double r)
{
    if (r == 0.0L) return a == rt(0) ? 0.0L : 1e30L;
    return std::fabs((static_cast<long double>(a) - r) / r) / std::numeric_limits<rt>::epsilon();
}

template <typename rt, bool Cplx>
void set_entry(std::vector<rt>& x, int i, rt re, rt im)
{
    if constexpr (Cplx) { x[2 * i] = re; x[2 * i + 1] = im; }
    else { (void)im; x[i] = re; }
}

// ||x||_2 in long double (the exponent range of long double covers the squares of every
// double / float component, denormals included).
template <typename rt, bool Cplx>
long double ref_norm(const std::vector<rt>& x, int n)
{
    long double acc = 0.0L;
    for (int i = 0; i < n; ++i) {
        const long double re = x[Cplx ? 2 * i : i];
        acc += re * re;
        if constexpr (Cplx) {
            const long double im = x[2 * i + 1];
            acc += im * im;
        }
    }
    return std::sqrt(acc);
}

template <typename rt, bool Cplx>
void check_norm(const std::vector<rt>& x, int n, const char* what)
{
    rt nn = rt(0), ns = rt(0);
    if (n > 0)
        ddla::detail::qr_norm_pair_serial<rt, Cplx>(x.data(), n, nn, ns);
    const long double r = ref_norm<rt, Cplx>(x, n);
    // ns = ||rsafmn x|| = rsafmn * ||x|| (a power of two scaling), when it does not overflow
    const long double rs = std::ldexp(r, std::is_same<rt, float>::value ? 102 : 969);
    const bool ns_ok = rs > static_cast<long double>(std::numeric_limits<rt>::max())
                       || rel_eps<rt>(ns, rs) <= 4.0L;
    // n itself is a subnormal number when ||x|| is: it can only be right to within one denorm_min.
    const bool n_normal = r >= static_cast<long double>(std::numeric_limits<rt>::min());
    const bool n_ok = n_normal
        ? rel_eps<rt>(nn, r) <= 4.0L
        : std::fabs(static_cast<long double>(nn) - r) <= static_cast<long double>(std::numeric_limits<rt>::denorm_min());
    if (!n_ok || !ns_ok)
        fail(std::string(what) + " [n=" + g_fmt(nn) + " ns=" + g_fmt(ns)
             + " reference n=" + g_fmt(static_cast<rt>(r)) + "]");
}

// NaN / Inf entries decide the norm: both n and ns are NaN when any entry is NaN (even next to an
// Inf), +Inf when an Inf is the only non-finite entry.
template <typename rt, bool Cplx>
void check_special(int nan_at, int inf_at, const char* what)
{
    const int n = 8;
    std::vector<rt> x(std::size_t(Cplx ? 2 * n : n), rt(0));
    for (int i = 0; i < n; ++i) set_entry<rt, Cplx>(x, i, rt(1) + rt(0.25) * rt(i), rt(0.5));
    if (nan_at >= 0) set_entry<rt, Cplx>(x, nan_at, std::numeric_limits<rt>::quiet_NaN(), rt(1));
    if (inf_at >= 0)   // complex: the Inf sits in the imaginary part (alpha = 2 - Inf i); real: it is the entry itself
        set_entry<rt, Cplx>(x, inf_at, Cplx ? rt(2) : -std::numeric_limits<rt>::infinity(), -std::numeric_limits<rt>::infinity());
    rt nn = rt(0), ns = rt(0);
    ddla::detail::qr_norm_pair_serial<rt, Cplx>(x.data(), n, nn, ns);
    const bool want_nan = nan_at >= 0;
    const bool ok = want_nan ? (std::isnan(nn) && std::isnan(ns))
                             : (std::isinf(nn) && nn > rt(0) && std::isinf(ns) && ns > rt(0));
    if (!ok)
        fail(std::string(what) + " [n=" + g_fmt(nn) + " ns=" + g_fmt(ns) + "]");
}

template <typename rt, bool Cplx>
void run_norm_checks(const char* tname)
{
    const std::string pfx = std::string(tname) + (Cplx ? " cplx " : " real ");

    // ---- qr_norm_scale: both factors normal, exact powers of two, exponents summing to -e over
    // the whole frexp range of the type.
    {
        int emin = 0, emax = 0;
        std::frexp(std::numeric_limits<rt>::denorm_min(), &emin);
        std::frexp(std::numeric_limits<rt>::max(), &emax);
        for (int e = emin; e <= emax; ++e) {
            rt s1, s2;
            ddla::detail::qr_norm_scale(e, s1, s2);
            const bool normal = std::fpclassify(s1) == FP_NORMAL
                             && std::fpclassify(s2) == FP_NORMAL;
            const int i1 = std::ilogb(s1), i2 = std::ilogb(s2);
            const bool exact = i1 + i2 == -e
                && s1 == std::ldexp(rt(1), i1) && s2 == std::ldexp(rt(1), i2);
            if (!normal || !exact)
                fail(pfx + "scale e=" + std::to_string(e)
                     + " normal=" + std::to_string(normal)
                     + " exact=" + std::to_string(exact));
        }
    }

    const rt tiny = std::is_same<rt, float>::value ? rt(1e-40) : rt(1e-310);   // actual denormals
    const rt huge = std::is_same<rt, float>::value ? rt(1e30) : rt(1e300);

    // an all-zero column: both norms are exactly 0
    {
        std::vector<rt> x(std::size_t(Cplx ? 2 * 64 : 64), rt(0));
        check_norm<rt, Cplx>(x, 64, (pfx + "zeros").c_str());
    }
    // denormals: the sum of squares of the plain components would underflow to zero
    {
        const int n = 40;
        std::vector<rt> x(std::size_t(Cplx ? 2 * n : n), rt(0));
        for (int i = 0; i < n; ++i)
            set_entry<rt, Cplx>(x, i, tiny * rt(0.5 + (i % 7) / 11.0), tiny * rt(0.25));
        check_norm<rt, Cplx>(x, n, (pfx + "denormal").c_str());
    }
    // components near the overflow threshold of their squares
    {
        const int n = 40;
        std::vector<rt> x(std::size_t(Cplx ? 2 * n : n), rt(0));
        for (int i = 0; i < n; ++i)
            set_entry<rt, Cplx>(x, i, huge * rt(0.5 + (i % 5) / 7.0), huge * rt(0.125));
        check_norm<rt, Cplx>(x, n, (pfx + "huge").c_str());
    }
    // NaN / Inf classification
    check_special<rt, Cplx>(3, -1, (pfx + "nan").c_str());
    check_special<rt, Cplx>(-1, 5, (pfx + "inf").c_str());
    check_special<rt, Cplx>(2, 6, (pfx + "nan-beats-inf").c_str());
    // mixed exponents, generic rounding regime
    {
        const int n = 256;
        std::vector<rt> x(std::size_t(Cplx ? 2 * n : n), rt(0));
        for (int i = 0; i < n; ++i) {
            const rt v = i < 64 ? rt(1) + rt(0.25) * rt(i % 13)
                       : i < 128 ? rt(1e-8) * (rt(1) + rt(0.5) * rt(i % 7))
                       : i < 192 ? rt(1e-40) * (rt(1) + rt(0.125) * rt(i % 9))
                                 : rt(3.5e6) * (rt(1) + rt(0.0625) * rt(i % 11));
            set_entry<rt, Cplx>(x, i, v, rt(0.5) * v);
        }
        check_norm<rt, Cplx>(x, n, (pfx + "mixed-exponents").c_str());
    }
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    run_norm_checks<float, false>("float");
    run_norm_checks<double, false>("double");
    run_norm_checks<float, true>("complex<float>");
    run_norm_checks<double, true>("complex<double>");

    if (rank == 0) {
        if (g_failures == 0) std::printf("test_qr_norm_fold passed\n");
        else std::printf("test_qr_norm_fold FAILED: %d checks\n", g_failures);
    }
    MPI_Finalize();
    return g_failures == 0 ? 0 : 1;
}
