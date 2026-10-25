#ifndef DDLA_PGEQRF_KERNELS_H
#define DDLA_PGEQRF_KERNELS_H

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <type_traits>

#include "ddla_connector.h"

// ---------------------------------------------------------------------------
// Shared arithmetic of the QR panel code and of pgebrd: the exact LAPACK scalar path of one Householder
// reflector (xLARFG) and the scale-safe norm helpers around it.
//
// QR_HD functions compile for host AND device when this TU runs under
// nvcc/hipcc (__CUDACC__/__HIP__, per-function __host__ __device__), and as
// plain inline host functions in CPU-only and dual-build CXX TUs.  The
// arithmetic is an exact port of the LAPACK auxiliaries the scalar path
// needs (dlapy2.f / dlapy3.f / dladiv.f / dlarfg.f / zlarfg.f); port
// fidelity was proven against the .f sources (the gfortran -O2 scratch
// check of the device round), NOT bitwise against the prebuilt OpenBLAS
// dladiv_/sladiv_, whose .so carries FMA contraction (1-2 ulp differences
// in 0.5-9% of inputs are inherent to that comparison).
//
// The larfg knt rescale path keeps the carried-norm trick: safmin =
// DLAMCH('S')/DLAMCH('E') = 2^-969 (d) / 2^-102 (s) makes rsafmn a power of
// two, ns = ||rsafmn*x|| is derived from the same scaled sum as n
// (qr_norm_finish), and for IEEE s/d knt <= 1 the re-norm is exact.
//
// The gathered panel no longer uses any of this: it calls ddla::geqrf (src/geqrf.cpp).  The
// consumers are pgebrd (host and device scalar paths), the in-place panel (DDLA_QR_PANEL=inplace)
// and tests/test_qr_norm_fold.cpp.
// ---------------------------------------------------------------------------

namespace ddla {
namespace detail {

template <typename T> struct qr_rt_of { using type = T; };
template <typename U> struct qr_rt_of<std::complex<U>> { using type = U; };

#if defined(__CUDACC__) || defined(__HIP__)
#define DDLA_QR_HD __host__ __device__
#else
#define DDLA_QR_HD
#endif

// --- DLAMCH constants of the IEEE s/d models, as hex literals --------------
// Verified against OpenBLAS 0.3.29: dlamch('E') = eps/2 under
// round-to-nearest, dlamch('S') = min() (smallest normal), dlamch('O') =
// max() (overflow threshold).  Host static_asserts pin the literals.
template <typename rt> struct QrLamch;
template <>
struct QrLamch<float> {
    static constexpr float eps = 0x1p-24f;             // DLAMCH('E')  = eps/2
    static constexpr float un  = 0x1p-126f;            // DLAMCH('S')  = min()
    static constexpr float ov  = 0x1.fffffep127f;      // DLAMCH('O')  = max()
    static constexpr float be  = 0x1p49f;              // BS/(EPS*EPS), BS = 2
};
template <>
struct QrLamch<double> {
    static constexpr double eps = 0x1p-53;             // DLAMCH('E')  = eps/2
    static constexpr double un  = 0x1p-1022;           // DLAMCH('S')  = min()
    static constexpr double ov  = 0x1.fffffffffffffp1023;  // DLAMCH('O')
    static constexpr double be  = 0x1p107;             // BS/(EPS*EPS), BS = 2
};

constexpr double qr_pow2(int e) { double v = 1.0; for (int i = 0; i < e; ++i) v *= 2.0; return v; }
constexpr float  qr_pow2f(int e) { float v = 1.0f; for (int i = 0; i < e; ++i) v *= 2.0f; return v; }

static_assert(QrLamch<double>::eps == std::numeric_limits<double>::epsilon() * 0.5,
              "DLAMCH('E') must be eps/2");
static_assert(QrLamch<double>::un == std::numeric_limits<double>::min(),
              "DLAMCH('S') must be the smallest normal");
static_assert(QrLamch<double>::ov == std::numeric_limits<double>::max(),
              "DLAMCH('O') must be the overflow threshold");
static_assert(QrLamch<double>::be == qr_pow2(107), "BE = BS/EPS^2 for double");
static_assert(QrLamch<float>::eps == std::numeric_limits<float>::epsilon() * 0.5f,
              "DLAMCH('E') must be eps/2");
static_assert(QrLamch<float>::un == std::numeric_limits<float>::min(),
              "DLAMCH('S') must be the smallest normal");
static_assert(QrLamch<float>::ov == std::numeric_limits<float>::max(),
              "DLAMCH('O') must be the overflow threshold");
static_assert(QrLamch<float>::be == qr_pow2f(49), "BE = BS/EPS^2 for float");

// safmin = DLAMCH('S')/DLAMCH('E'), as dlamch-consuming LAPACK code computes
// it (= min()/(eps/2) = 2^-969 (d) / 2^-102 (s), per dlarfg.f).
template <typename rt>
DDLA_QR_HD constexpr rt qr_safmin() { return QrLamch<rt>::un / QrLamch<rt>::eps; }
static_assert(qr_safmin<double>() == std::numeric_limits<double>::min()
                                  / (std::numeric_limits<double>::epsilon() * 0.5),
              "safmin(double) mismatch");
static_assert(qr_safmin<float>() == std::numeric_limits<float>::min()
                                 / (std::numeric_limits<float>::epsilon() * 0.5f),
              "safmin(float) mismatch");

// Shift K with ||2^K x|| = rsafmn * ||x||: K = log2(rsafmn).
template <typename rt> struct QrNormShift;
template <> struct QrNormShift<float> { static constexpr int k = 102; };
template <> struct QrNormShift<double> { static constexpr int k = 969; };
static_assert(qr_pow2(QrNormShift<double>::k) == 1.0 / qr_safmin<double>(),
              "K(double) must satisfy 2^K = rsafmn");
static_assert(qr_pow2f(QrNormShift<float>::k) == 1.0f / qr_safmin<float>(),
              "K(float) must satisfy 2^K = rsafmn");

// --- math wrappers (unqualified names resolve to device intrinsics under
// nvcc/hipcc and to <cmath> on the host) ------------------------------------
template <typename rt> DDLA_QR_HD inline rt qr_abs(rt v) { return ::fabs(v); }
template <typename rt> DDLA_QR_HD inline rt qr_copysign(rt m, rt s) { return ::copysign(m, s); }
template <typename rt> DDLA_QR_HD inline bool qr_isnan(rt v) { return v != v; }
template <typename rt> DDLA_QR_HD inline bool qr_isinf(rt v)
{
    // |v| beyond the overflow threshold and not NaN: an infinity.  (The NaN
    // comparison is false, so inf > max() holds and NaN > max() does not.)
    return qr_abs(v) > QrLamch<rt>::ov;
}
template <typename rt> DDLA_QR_HD inline rt qr_sqrt(rt v) { return ::sqrt(v); }
template <typename rt> DDLA_QR_HD inline rt qr_frexp(rt v, int* e) { return ::frexp(v, e); }
template <typename rt> DDLA_QR_HD inline rt qr_ldexp(rt v, int e) { return ::ldexp(v, e); }

// +inf / quiet NaN without numeric_limits calls in device code: ov*2
// overflows to +inf, and inf*0 is a quiet NaN, both deterministic.
template <typename rt> DDLA_QR_HD inline rt qr_inf() { return QrLamch<rt>::ov * rt(2); }
template <typename rt> DDLA_QR_HD inline rt qr_nan() { return qr_inf<rt>() * rt(0); }

// --- LAPACK auxiliary ports (dlapy2.f / dlapy3.f) ---------------------------
// NaN is routed to the result by hand: every comparison against NaN is
// false, so the max/min below would silently drop a NaN component instead of
// propagating it into the norm.
template <typename rt>
DDLA_QR_HD inline rt qr_lapy2(rt x, rt y)
{
    const rt xabs = qr_abs(x), yabs = qr_abs(y);
    if (qr_isnan(xabs)) return xabs;
    if (qr_isnan(yabs)) return yabs;
    const rt w = xabs > yabs ? xabs : yabs;
    const rt z = xabs > yabs ? yabs : xabs;
    if (z == rt(0) || w > QrLamch<rt>::ov) return w;
    return w * qr_sqrt(rt(1) + (z / w) * (z / w));
}

template <typename rt>
DDLA_QR_HD inline rt qr_lapy3(rt x, rt y, rt z)
{
    const rt xabs = qr_abs(x), yabs = qr_abs(y), zabs = qr_abs(z);
    if (qr_isnan(xabs) || qr_isnan(yabs) || qr_isnan(zabs))
        return xabs + yabs + zabs;   // 0 + NaN = NaN: the NaN propagates
    rt w = xabs > yabs ? xabs : yabs;
    w = w > zabs ? w : zabs;
    if (w == rt(0) || w > QrLamch<rt>::ov) return xabs + yabs + zabs;
    return w * qr_sqrt((xabs / w) * (xabs / w) + (yabs / w) * (yabs / w)
                       + (zabs / w) * (zabs / w));
}

// --- DLADIV / DLADIV1 / DLADIV2 (dladiv.f, Baudin-Smith 2012) --------------
// Verbatim port, including the OV/UN/EPS rescale branches, so the complex
// quotient matches the host's OpenBLAS dladiv_ bitwise.
template <typename rt>
DDLA_QR_HD inline rt qr_dladiv2(rt a, rt b, rt c, rt d, rt r, rt t)
{
    if (r != rt(0)) {
        const rt br = b * r;
        if (br != rt(0)) return (a + br) * t;
        return a * t + (b * t) * r;
    }
    return (a + d * (b / c)) * t;
}

template <typename rt>
DDLA_QR_HD inline void qr_dladiv1(rt a, rt b, rt c, rt d, rt& p, rt& q)
{
    const rt r = d / c;
    const rt t = rt(1) / (c + d * r);
    p = qr_dladiv2(a, b, c, d, r, t);
    a = -a;
    q = qr_dladiv2(b, a, c, d, r, t);
}

template <typename rt>
DDLA_QR_HD inline void qr_dladiv(rt a, rt b, rt c, rt d, rt& p, rt& q)
{
    rt aa = a, bb = b, cc = c, dd = d;
    const rt bs = rt(2), half = rt(0.5), two = rt(2);
    rt ab = qr_abs(a) > qr_abs(b) ? qr_abs(a) : qr_abs(b);
    rt cd = qr_abs(c) > qr_abs(d) ? qr_abs(c) : qr_abs(d);
    rt s = rt(1);
    const rt ov = QrLamch<rt>::ov, un = QrLamch<rt>::un, eps = QrLamch<rt>::eps;
    const rt be = bs / (eps * eps);
    if (ab >= half * ov) { aa = half * aa; bb = half * bb; s = two * s; }
    if (cd >= half * ov) { cc = half * cc; dd = half * dd; s = half * s; }
    if (ab <= un * bs / eps) { aa = aa * be; bb = bb * be; s = s / be; }
    if (cd <= un * bs / eps) { cc = cc * be; dd = dd * be; s = s * be; }
    if (qr_abs(d) <= qr_abs(c)) {
        qr_dladiv1(aa, bb, cc, dd, p, q);
    } else {
        qr_dladiv1(bb, aa, dd, cc, p, q);
        q = -q;
    }
    p = p * s;
    q = q * s;
}

// --- robust local 2-norm ----------------------------------------------------
// Finish: from the sqrt of the fixed-order sum of squares of the components
// scaled by 2^-e (e = frexp exponent of the max component) produce the local
// robust norm n and the pre-scaled norm ns = ||rsafmn * x||.  ns is taken
// directly from s (NOT from n) so a denormal n cannot lose bits that the knt
// re-norm path would need.
template <typename rt>
DDLA_QR_HD inline void qr_norm_finish(rt s, int e, rt& n, rt& ns)
{
    n = qr_ldexp(s, e);
    ns = qr_ldexp(s, e + QrNormShift<rt>::k);
}

// Scale factors for the norm's fixed-order sum of squares: scl1*scl2 = 2^-e,
// split into one or two exact power-of-two factors that are themselves NORMAL
// numbers.  A single factor 2^-e overflows for a denormal max component
// (e.g. 2^1030 for double) and goes SUBNORMAL once e >= 127 (float) /
// e >= 1023 (double) -- and a subnormal factor would be flushed to zero by
// any FTZ/flush-to-zero setting, silently collapsing the norm.  The exponent
// is therefore clamped into the normal range and the remainder carried by the
// second factor: every intermediate stays representable and every product is
// exact, so the scaled squares (and with them the record bits) are identical
// whichever split is taken.
template <typename rt>
DDLA_QR_HD inline void qr_norm_scale(int e, rt& scl1, rt& scl2)
{
    int e1, e2;
    if constexpr (std::is_same<rt, float>::value) {
        e1 = e < -126 ? 126 : (e > 126 ? -126 : -e);
        e2 = -e - e1;
    } else {
        e1 = e < -1022 ? 1022 : (e > 1022 ? -1022 : -e);
        e2 = -e - e1;
    }
    scl1 = qr_ldexp(rt(1), e1);
    scl2 = e2 != 0 ? qr_ldexp(rt(1), e2) : rt(1);
}

// Serial two-pass version for the host arms (NaN-sticky max over the real
// components, then frexp/ldexp scaling, fixed-order sum of squares, sqrt).
template <typename rt, bool Cplx>
DDLA_QR_HD inline void qr_norm_pair_serial(const rt* x, int nx, rt& n, rt& ns)
{
    rt m = rt(0);
    bool has_nan = false, has_inf = false;
    for (int i = 0; i < nx; ++i) {
        const rt re = qr_abs(x[Cplx ? 2 * i : i]);
        if (qr_isnan(re)) has_nan = true;
        if (qr_isinf(re)) has_inf = true;
        if (re > m) m = re;
        if constexpr (Cplx) {
            const rt im = qr_abs(x[2 * i + 1]);
            if (qr_isnan(im)) has_nan = true;
            if (qr_isinf(im)) has_inf = true;
            if (im > m) m = im;
        }
    }
    if (has_nan) {
        n = qr_nan<rt>();
        ns = n;
        return;
    }
    if (has_inf) {
        n = qr_inf<rt>();
        ns = n;
        return;
    }
    if (m == rt(0)) {
        n = rt(0);
        ns = rt(0);
        return;
    }
    int e = 0;
    qr_frexp(m, &e);                        // m = f * 2^e, f in [0.5, 1)
    rt scl1, scl2;
    qr_norm_scale(e, scl1, scl2);
    rt acc = rt(0);
    for (int i = 0; i < nx; ++i) {
        const rt re = (x[Cplx ? 2 * i : i] * scl1) * scl2;
        acc += re * re;
        if constexpr (Cplx) {
            const rt im = (x[2 * i + 1] * scl1) * scl2;
            acc += im * im;
        }
    }
    qr_norm_finish(qr_sqrt(acc), e, n, ns);
}

// --- DLARFG/ZLARFG scalar path ---------------------------------------------
// Scalar state produced from the folded norms; identical bits on every
// consumer because the record (or the host norms) is identical.
template <typename rt, bool Cplx>
struct QrLarfg {
    rt ar, ai;        // (possibly rescaled) alpha
    rt xn;            // folded robust norm
    rt xns;           // folded pre-scaled norm ||rsafmn * x||
    rt safmin, rsafmn;
    bool trivial;     // xns == 0 (equivalently xnorm == 0) && (real || ai == 0): tau = 0, nothing written
    int knt;          // rescale iterations (<= 1 for IEEE s/d)
    rt beta;          // true beta (restored through the knt loop)
    rt tau_r, tau_i;  // tau; tau_i is meaningful only for Cplx
    rt vfac_r, vfac_i;// x scale: v = (x * rsafmn^knt) * vfac
};

template <typename rt, bool Cplx>
DDLA_QR_HD inline void qr_larfg_from_norms(rt ar, rt ai, rt xn, rt xns,
                                           QrLarfg<rt, Cplx>& s)
{
    s.xn = xn;
    s.xns = xns;
    s.ar = ar;
    s.ai = ai;
    s.safmin = qr_safmin<rt>();
    s.rsafmn = rt(1) / s.safmin;
    if (xns == rt(0) && s.ai == rt(0)) {
        // H = I: tau = 0, ALPHA (and x, V) left untouched, as in zlarfg.f.
        s.trivial = true;
        s.knt = 0;
        s.tau_r = rt(0);
        s.tau_i = rt(0);
        s.beta = s.ar;
        s.vfac_r = rt(0);
        s.vfac_i = rt(0);
        return;
    }
    s.trivial = false;
    rt beta;
    if constexpr (Cplx) beta = -qr_copysign(qr_lapy3(s.ar, s.ai, xn), s.ar);
    else                beta = -qr_copysign(qr_lapy2(s.ar, xn), s.ar);
    int knt = 0;
    if (qr_abs(beta) < s.safmin) {
        // XNORM/BETA/ALPHA may be inaccurate (underflow): rescale, then
        // recompute the norm from the carried ns and recompute beta.
        do {
            ++knt;
            beta *= s.rsafmn;
            s.ai *= s.rsafmn;
            s.ar *= s.rsafmn;
        } while (qr_abs(beta) < s.safmin && knt < 20);
        // New xnorm = ||rsafmn^knt x|| = rsafmn^(knt-1) * ||rsafmn x||; the
        // rescale is a power of two, so for knt = 1 the carried ns IS the
        // recomputed xnorm.
        rt xn2 = xns;
        for (int j = 1; j < knt; ++j) xn2 *= s.rsafmn;
        if constexpr (Cplx) beta = -qr_copysign(qr_lapy3(s.ar, s.ai, xn2), s.ar);
        else                beta = -qr_copysign(qr_lapy2(s.ar, xn2), s.ar);
    }
    s.knt = knt;
    // tau and vfac in the (possibly rescaled) domain, as in xlarfg.f.
    s.tau_r = (beta - s.ar) / beta;
    if constexpr (Cplx) s.tau_i = -s.ai / beta;
    if constexpr (Cplx) qr_dladiv(rt(1), rt(0), s.ar - beta, s.ai, s.vfac_r, s.vfac_i);
    else                s.vfac_r = rt(1) / (s.ar - beta);
    // Restore the true beta through the knt rescales, as dlarfg.f does.
    for (int j = 0; j < knt; ++j) beta *= s.safmin;
    s.beta = beta;
}

// --- block-cyclic row mapping (host/device copy of indxl2g) -----------------
// Global index of local index i on process coordinate p, matching
// ddla::indxl2g in ddla_desc.h bitwise.  Used by the gathered panel's row
// permutation kernels and the CPU arm's host loops.
DDLA_QR_HD inline int qr_l2g(int i, int nb, int p, int src, int np)
{
    return np * nb * (i / nb) + i % nb + ((np + p - src) % np) * nb;
}

// --- GPU launcher wrappers ---------------------------------------------------
// Defined in pgeqrf_kernels.cpp in GPU-vendor builds only.  They are declared in every build so the
// `if constexpr (Backend == GPU)` arms of the host code (pgeqr2_panel.cpp) name them without a
// preprocessor guard; a CPU-only build never instantiates those arms, so nothing refers to them.

// Local J A J reversal used by the vendor QL implementation. No distribution.
template <typename T>
void qr_reverse(const DdlaHandle_t& handle, int m, int n, const T* src, int ldsrc, T* dst, int lddst);

// Explicit native V from a panel window in LAPACK storage (natural order): entry
// (i, c) of v (lrp rows, ldv), g = global row of local row lr0 + i.  rev == 0 (QR,
// unit-lower V): 1 on the global diagonal (g == j0 + c), 0 above it, the window
// value (g - j0, c) below.  rev == 1 (QL, unit-upper V): 1 at g == j0 + m_w - jb + c,
// 0 below it, the window value (g - j0, c) above.  in_place: src aliases v and every
// entry reads only its own element (i, c).
template <typename T>
void qr_build_v(const DdlaHandle_t& handle, bool in_place, int rev, T* v, int ldv, const T* src, int ldsrc,
                int lrp, int jb, int j0, int m_w, int lr0, int mb, int myprow,
                int rsrc, int nprows);

// out := T^H for a jb x jb column-major T (ld jb): out(r, c) = conj(T(c, r)), written into a
// matrix with leading dimension ldout (the stage-1 panel stores T_L^H, the upper-triangular
// T of phe2hb, into its tmat slot).  Index-only, one block.
template <typename T>
void qr_adjoint_T(const DdlaHandle_t& handle, T* out, int ldout, const T* in, int jb);

// One T row per block from the reduced V^H V (only its upper triangle is read) and the
// device-resident tau.  backward == false (QR): T upper triangular, T(i,i) = tau_i,
// T(r,i) = -tau_i * sum_{p=r}^{i-1} T(r,p) * VV(p,i) (ZLARFT 'F').  backward == true (QL): T
// lower triangular, T(r,i) = -tau_i * sum_{p=i+1}^{r} T(r,p) * conj(VV(i,p)) (ZLARFT 'B').
// The full row (zeros in the other triangle included) is written; T has leading dimension ldt >= jb
// (the rows beyond jb of a padded T are not touched).  tau is
// preloaded into shared memory when twice the row fits under the device
// shared-memory limit, and read from global memory step by step otherwise
// (huge nb keeps the launch legal).
template <typename T>
void qr_build_T(const DdlaHandle_t& handle, int jb, const T* vv, const T* tau, T* tmat, int ldt, bool backward);

} // namespace detail
} // namespace ddla

#endif // DDLA_PGEQRF_KERNELS_H
