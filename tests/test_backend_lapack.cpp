// ---------------------------------------------------------------------------
// Backend-templated coverage of ddla::getrf, ddla::potrf, ddla::laswp, ddla::geqrf and ddla::larft -- the LAPACK-level wrappers that
// used to be raw cuSOLVER / hipSOLVER helpers in private headers (getrf.h, potrf.h, laswp.h) and now have a CPU arm
// (host LAPACK) and a GPU arm behind one `<Backend, T>` interface.
//
//   getrf  square / tall / wide shapes, padded leading dimension with the padding checked untouched, pivot range,
//          A = P*L*U rebuilt by hand and compared with the input, `info` for an exactly singular matrix;
//   potrf  'U' and 'L' on a Hermitian positive-definite matrix, A = U^H U / L L^H rebuilt by hand, the unreferenced
//          triangle checked bit for bit, `info` for a matrix that is not positive definite;
//   laswp  bit-exact against a hand-written LAPACK loop for incx = 1, 2, -1, -2 and sub-ranges k1..k2 (incx = 1 is
//          the vendor routine on CUDA, every other stride the row-by-row fallback), and incx = 1 followed by
//          incx = -1 restoring the matrix;
//   geqrf  uplo 'U' (QR) and 'L' (QL) on square / tall / wide shapes with a padded leading dimension: A = Q R / A = Q L
//          rebuilt from the stored reflectors by hand (Q formed explicitly and checked unitary), the padding untouched,
//          the real diagonal of complex R / L, and tau + the stored factors against host LAPACK ?geqrf / ?geqlf on
//          well-conditioned input; then single-column matrices that stress the reflector scalars (alpha = 0, alpha =
//          -0.0, x = 0, x = 0 with an imaginary alpha, subnormal / huge / mixed-scale x), compared with host LAPACK to
//          8 eps entrywise (bitwise for the exactly decided ones);
//   larft  direct 'F' and 'B' on reflectors with exact LAPACK ?larfg-style scalars (unitary H(i), some tau = 0): H(1)...H(k)
//          (H(k)...H(1)) multiplied out by hand equals I - V T V^H with the explicit V, T against the textbook recurrence
//          in double precision, the triangle shape (exact zeros in the other triangle, tau on the diagonal), the
//          implied entries of V (garbage planted in the diagonal and the unreferenced triangle) never read and V
//          bitwise unchanged, padded V and T untouched, k up to 300 (beyond one kernel block);
//   all    empty problems (info = 0) and the argument checks (null handle, backend mismatch, bad flag, null
//          pointers, leading dimension, negative sizes).
// Every pointer lives where the handle's backend does (host memory on a CPU handle, device memory on a GPU
// handle), `ipiv` and `info` included.  The reference arithmetic is written out by hand, not taken from the library.
//
// Follows the structure of test_backend_geam.cpp.  Debug aid:
//   DDLA_TEST_ONLY=cpu|gpu   run one backend alone
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <mpi.h>

#include <ddla/ddla.h>

using namespace ddla;

// Host LAPACK reference for the geqrf checks (the CPU BLAS library provides LAPACK: OpenBLAS and MKL do).
extern "C" {
void sgeqrf_(const int*, const int*, float*, const int*, float*, float*, const int*, int*);
void dgeqrf_(const int*, const int*, double*, const int*, double*, double*, const int*, int*);
void cgeqrf_(const int*, const int*, std::complex<float>*, const int*, std::complex<float>*, std::complex<float>*,
             const int*, int*);
void zgeqrf_(const int*, const int*, std::complex<double>*, const int*, std::complex<double>*, std::complex<double>*,
             const int*, int*);
void sgeqlf_(const int*, const int*, float*, const int*, float*, float*, const int*, int*);
void dgeqlf_(const int*, const int*, double*, const int*, double*, double*, const int*, int*);
void cgeqlf_(const int*, const int*, std::complex<float>*, const int*, std::complex<float>*, std::complex<float>*,
             const int*, int*);
void zgeqlf_(const int*, const int*, std::complex<double>*, const int*, std::complex<double>*, std::complex<double>*,
             const int*, int*);
}

namespace {

int g_rank = 0;

// -- deterministic data ------------------------------------------------------
struct Rng {
    std::uint64_t s;
    explicit Rng(std::uint64_t seed) : s(seed) {}
    std::uint64_t next()
    {
        s += 0x9E3779B97F4A7C15ull;
        std::uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uni() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0; }
};

template <typename T> struct Tr;
template <> struct Tr<float> {
    typedef float real;
    static float make(double a, double) { return static_cast<float>(a); }
    static float conjv(float x) { return x; }
    static double eps() { return 1.2e-7; }
    static const bool cplx = false;
};
template <> struct Tr<double> {
    typedef double real;
    static double make(double a, double) { return a; }
    static double conjv(double x) { return x; }
    static double eps() { return 2.3e-16; }
    static const bool cplx = false;
};
template <> struct Tr<std::complex<float>> {
    typedef float real;
    static std::complex<float> make(double a, double b) { return std::complex<float>(float(a), float(b)); }
    static std::complex<float> conjv(const std::complex<float>& x) { return std::conj(x); }
    static double eps() { return 1.2e-7; }
    static const bool cplx = true;
};
template <> struct Tr<std::complex<double>> {
    typedef double real;
    static std::complex<double> make(double a, double b) { return std::complex<double>(a, b); }
    static std::complex<double> conjv(const std::complex<double>& x) { return std::conj(x); }
    static double eps() { return 2.3e-16; }
    static const bool cplx = true;
};

template <typename T>
std::vector<T> random_vec(std::size_t n, std::uint64_t seed)
{
    Rng r(seed);
    std::vector<T> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double a = r.uni(), b = r.uni();
        v[i] = Tr<T>::make(a, b);
    }
    return v;
}

template <typename T>
bool same_bits(const T& a, const T& b)
{
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

// -- device (or host, on a CPU handle) buffers --------------------------------
template <typename T>
struct Buf {
    DdlaHandle_t h;
    T* p = nullptr;
    std::size_t n = 0;
    Buf(const DdlaHandle_t& handle, std::size_t count) : h(handle), n(count)
    {
        void* q = nullptr;
        if (ddlaMalloc(h, &q, std::max<std::size_t>(1, n) * sizeof(T)) != ddlaStatus_t::DDLA_STATUS_SUCCESS)
            throw std::runtime_error("test_backend_lapack: ddlaMalloc failed");
        p = static_cast<T*>(q);
    }
    ~Buf() { ddlaFree(h, p); }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    void up(const std::vector<T>& v)
    {
        if (!v.empty()) ddlaMemcpy(h, p, v.data(), v.size() * sizeof(T));
        ddlaSynchronize(h);
    }
    void down(std::vector<T>& v)
    {
        v.resize(n);
        if (n > 0) ddlaMemcpy(h, v.data(), p, n * sizeof(T));
        ddlaSynchronize(h);
    }
};

void report(const char* what, const std::string& detail)
{
    if (g_rank != 0) return;
    std::cerr << "  FAIL " << what << " " << detail << std::endl;
}

std::string dims(int m, int n, int extra = -1)
{
    std::string s = "m=" + std::to_string(m) + " n=" + std::to_string(n);
    if (extra >= 0) s += " " + std::to_string(extra);
    return s;
}

template <typename T>
T guard_value()
{
    return Tr<T>::make(777.0, -555.0);
}

// -- getrf --------------------------------------------------------------------------
// zero_col >= 0: that column of A is exactly zero, so U(zero_col, zero_col) is the first zero pivot.
template <DdlaBackend Backend, typename T>
int check_getrf(const DdlaHandle_t& h, int m, int n, int pad, int zero_col, std::uint64_t seed)
{
    const int lda = m + pad;
    const int k = std::min(m, n);
    std::vector<T> A0 = random_vec<T>(static_cast<std::size_t>(lda) * n, seed);
    for (int j = 0; j < n; ++j)
        for (int i = m; i < lda; ++i) A0[i + static_cast<std::size_t>(j) * lda] = guard_value<T>();
    if (zero_col >= 0)
        for (int i = 0; i < m; ++i) A0[i + static_cast<std::size_t>(zero_col) * lda] = Tr<T>::make(0.0, 0.0);

    Buf<T> dA(h, A0.size());
    Buf<int> dpiv(h, static_cast<std::size_t>(k)), dinfo(h, 1);
    dA.up(A0);
    dpiv.up(std::vector<int>(static_cast<std::size_t>(k), -1));
    dinfo.up(std::vector<int>(1, -77));

    getrf<Backend, T>(h, m, n, dA.p, lda, dpiv.p, dinfo.p);
    ddlaSynchronize(h);

    std::vector<T> F;
    std::vector<int> piv, info;
    dA.down(F);
    dpiv.down(piv);
    dinfo.down(info);

    int failures = 0;
    const int want_info = zero_col >= 0 ? zero_col + 1 : 0;
    if (info[0] != want_info) {
        report("getrf info", dims(m, n, zero_col) + " got " + std::to_string(info[0]) + " want " + std::to_string(want_info));
        ++failures;
    }
    for (int j = 0; j < n; ++j)
        for (int i = m; i < lda; ++i)
            if (!same_bits(F[i + static_cast<std::size_t>(j) * lda], guard_value<T>())) {
                report("getrf padding written", dims(m, n) + " at (" + std::to_string(i) + "," + std::to_string(j) + ")");
                return failures + 1;
            }
    if (zero_col >= 0) return failures;      // a zero pivot: the vendor routines may leave L undefined below it

    for (int i = 0; i < k; ++i)
        if (piv[i] < i + 1 || piv[i] > m) {
            report("getrf pivot range", dims(m, n) + " ipiv[" + std::to_string(i) + "]=" + std::to_string(piv[i]));
            return failures + 1;
        }

    // A = P * L * U: multiply the factors, then undo the interchanges from the last to the first.
    auto at = [&](int i, int j) { return F[i + static_cast<std::size_t>(j) * lda]; };
    std::vector<T> R(static_cast<std::size_t>(m) * n, Tr<T>::make(0.0, 0.0));
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            T s = Tr<T>::make(0.0, 0.0);
            const int tmax = std::min(std::min(i, j), k - 1);
            for (int t = 0; t <= tmax; ++t) s += (i == t ? Tr<T>::make(1.0, 0.0) : at(i, t)) * at(t, j);
            R[i + static_cast<std::size_t>(j) * m] = s;
        }
    for (int t = k - 1; t >= 0; --t) {
        const int p = piv[t] - 1;
        if (p != t)
            for (int j = 0; j < n; ++j) std::swap(R[t + static_cast<std::size_t>(j) * m], R[p + static_cast<std::size_t>(j) * m]);
    }
    double err = 0.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            err = std::max(err, static_cast<double>(std::abs(R[i + static_cast<std::size_t>(j) * m] -
                                                              A0[i + static_cast<std::size_t>(j) * lda])));
    const double tol = 200.0 * Tr<T>::eps() * std::max(m, n);
    if (!(err <= tol)) {
        report("getrf residual", dims(m, n) + " err " + std::to_string(err) + " tol " + std::to_string(tol));
        ++failures;
    }
    return failures;
}

// -- potrf --------------------------------------------------------------------------
// bad_k >= 0: the matrix is the identity except A(bad_k, bad_k) = -1, so the leading minor of order bad_k + 1 fails.
template <DdlaBackend Backend, typename T>
int check_potrf(const DdlaHandle_t& h, char uplo, int n, int pad, int bad_k, std::uint64_t seed)
{
    const int lda = n + pad;
    const std::size_t N = static_cast<std::size_t>(n);
    // Hermitian positive definite: A = B*B^H + n*I (diagonal made exactly real).
    std::vector<T> B = random_vec<T>(N * n, seed);
    std::vector<T> A(static_cast<std::size_t>(lda) * n, guard_value<T>());
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            T s = Tr<T>::make(0.0, 0.0);
            for (int t = 0; t < n; ++t) s += B[i + t * N] * Tr<T>::conjv(B[j + t * N]);
            if (i == j) s = Tr<T>::make(static_cast<double>(std::real(s)) + n, 0.0);
            A[i + static_cast<std::size_t>(j) * lda] = s;
        }
    if (bad_k >= 0)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                A[i + static_cast<std::size_t>(j) * lda] = Tr<T>::make(i == j ? (i == bad_k ? -1.0 : 1.0) : 0.0, 0.0);
    // The triangle potrf must not touch gets a sentinel that would wreck the result if it were read.
    std::vector<T> A0 = A;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            if ((uplo == 'U' && i > j) || (uplo == 'L' && i < j))
                A0[i + static_cast<std::size_t>(j) * lda] = Tr<T>::make(777.0 + i, -555.0 - j);

    Buf<T> dA(h, A0.size());
    Buf<int> dinfo(h, 1);
    dA.up(A0);
    dinfo.up(std::vector<int>(1, -77));
    potrf<Backend, T>(h, uplo, n, dA.p, lda, dinfo.p);
    ddlaSynchronize(h);
    std::vector<T> F;
    std::vector<int> info;
    dA.down(F);
    dinfo.down(info);

    int failures = 0;
    const int want_info = bad_k >= 0 ? bad_k + 1 : 0;
    if (info[0] != want_info) {
        report("potrf info", std::string("uplo=") + uplo + " " + dims(n, n, bad_k) + " got " + std::to_string(info[0]) +
                                 " want " + std::to_string(want_info));
        ++failures;
    }
    // Unreferenced triangle and padding: bit for bit what went in.
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < lda; ++i) {
            const bool referenced = i < n && ((uplo == 'U' && i <= j) || (uplo == 'L' && i >= j));
            if (referenced) continue;
            if (!same_bits(F[i + static_cast<std::size_t>(j) * lda], A0[i + static_cast<std::size_t>(j) * lda])) {
                report("potrf wrote outside the triangle", std::string("uplo=") + uplo + " " + dims(n, n) + " at (" +
                                                               std::to_string(i) + "," + std::to_string(j) + ")");
                return failures + 1;
            }
        }
    if (bad_k >= 0) return failures;

    double err = 0.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            if (!((uplo == 'U' && i <= j) || (uplo == 'L' && i >= j))) continue;
            T s = Tr<T>::make(0.0, 0.0);
            const int tmax = std::min(i, j);
            for (int t = 0; t <= tmax; ++t) {
                if (uplo == 'U')
                    s += Tr<T>::conjv(F[t + static_cast<std::size_t>(i) * lda]) * F[t + static_cast<std::size_t>(j) * lda];
                else
                    s += F[i + static_cast<std::size_t>(t) * lda] * Tr<T>::conjv(F[j + static_cast<std::size_t>(t) * lda]);
            }
            err = std::max(err, static_cast<double>(std::abs(s - A[i + static_cast<std::size_t>(j) * lda])));
        }
    const double tol = 200.0 * Tr<T>::eps() * n * n;
    if (!(err <= tol)) {
        report("potrf residual", std::string("uplo=") + uplo + " " + dims(n, n) + " err " + std::to_string(err) + " tol " +
                                     std::to_string(tol));
        ++failures;
    }
    return failures;
}

// -- laswp --------------------------------------------------------------------------
template <typename T>
void laswp_ref(std::vector<T>& A, int lda, int n, int k1, int k2, const std::vector<int>& ipiv, int incx)
{
    const int stride = std::abs(incx);
    const int first = incx > 0 ? k1 : k2;
    const int last = incx > 0 ? k2 : k1;
    const int step = incx > 0 ? 1 : -1;
    for (int i = first;; i += step) {
        const int p = ipiv[(k1 - 1) + static_cast<std::size_t>(i - k1) * stride];
        if (p != i)
            for (int j = 0; j < n; ++j)
                std::swap(A[(i - 1) + static_cast<std::size_t>(j) * lda], A[(p - 1) + static_cast<std::size_t>(j) * lda]);
        if (i == last) break;
    }
}

template <DdlaBackend Backend, typename T>
int check_laswp(const DdlaHandle_t& h, int m, int n, int pad, int k1, int k2, int incx, std::uint64_t seed)
{
    const int lda = m + pad;
    const int stride = std::abs(incx);
    std::vector<T> A0 = random_vec<T>(static_cast<std::size_t>(lda) * n, seed);
    for (int j = 0; j < n; ++j)
        for (int i = m; i < lda; ++i) A0[i + static_cast<std::size_t>(j) * lda] = guard_value<T>();
    Rng r(seed ^ 0xABCDEFull);
    std::vector<int> ipiv(static_cast<std::size_t>(k2) * stride);
    for (std::size_t i = 0; i < ipiv.size(); ++i) ipiv[i] = 1 + static_cast<int>(r.next() % static_cast<std::uint64_t>(m));

    std::vector<T> want = A0;
    laswp_ref(want, lda, n, k1, k2, ipiv, incx);

    Buf<T> dA(h, A0.size());
    Buf<int> dpiv(h, ipiv.size());
    dA.up(A0);
    dpiv.up(ipiv);
    laswp<Backend, T>(h, n, dA.p, lda, k1, k2, dpiv.p, incx);
    ddlaSynchronize(h);
    std::vector<T> got;
    dA.down(got);

    int failures = 0;
    for (std::size_t i = 0; i < got.size(); ++i)
        if (!same_bits(got[i], want[i])) {
            report("laswp", dims(m, n) + " k1=" + std::to_string(k1) + " k2=" + std::to_string(k2) + " incx=" +
                                std::to_string(incx) + " first mismatch at " + std::to_string(i));
            ++failures;
            break;
        }

    // incx and -incx are inverse permutations: applying both restores the matrix.
    if (incx == 1) {
        laswp<Backend, T>(h, n, dA.p, lda, k1, k2, dpiv.p, -1);
        ddlaSynchronize(h);
        dA.down(got);
        for (std::size_t i = 0; i < got.size(); ++i)
            if (!same_bits(got[i], A0[i])) {
                report("laswp round trip (incx=1 then -1)", dims(m, n) + " k1=" + std::to_string(k1) + " k2=" +
                                                                std::to_string(k2) + " first mismatch at " + std::to_string(i));
                ++failures;
                break;
            }
    }
    return failures;
}

// -- geqrf --------------------------------------------------------------------------
inline void lapack_geqrf(bool lower, int m, int n, float* A, int lda, float* tau, float* w, int lw, int* info)
{ if (lower) sgeqlf_(&m, &n, A, &lda, tau, w, &lw, info); else sgeqrf_(&m, &n, A, &lda, tau, w, &lw, info); }
inline void lapack_geqrf(bool lower, int m, int n, double* A, int lda, double* tau, double* w, int lw, int* info)
{ if (lower) dgeqlf_(&m, &n, A, &lda, tau, w, &lw, info); else dgeqrf_(&m, &n, A, &lda, tau, w, &lw, info); }
inline void lapack_geqrf(bool lower, int m, int n, std::complex<float>* A, int lda, std::complex<float>* tau,
                         std::complex<float>* w, int lw, int* info)
{ if (lower) cgeqlf_(&m, &n, A, &lda, tau, w, &lw, info); else cgeqrf_(&m, &n, A, &lda, tau, w, &lw, info); }
inline void lapack_geqrf(bool lower, int m, int n, std::complex<double>* A, int lda, std::complex<double>* tau,
                         std::complex<double>* w, int lw, int* info)
{ if (lower) zgeqlf_(&m, &n, A, &lda, tau, w, &lw, info); else zgeqrf_(&m, &n, A, &lda, tau, w, &lw, info); }

template <typename T>
void host_geqrf(bool lower, int m, int n, std::vector<T>& A, int lda, std::vector<T>& tau)
{
    tau.assign(static_cast<std::size_t>(std::max(1, std::min(m, n))), Tr<T>::make(0.0, 0.0));
    T q = Tr<T>::make(0.0, 0.0);
    int info = 0;
    lapack_geqrf(lower, m, n, A.data(), lda, tau.data(), &q, -1, &info);
    std::vector<T> work(static_cast<std::size_t>(std::max(1, static_cast<int>(std::real(q)))));
    lapack_geqrf(lower, m, n, A.data(), lda, tau.data(), work.data(), static_cast<int>(work.size()), &info);
}

// B := (I - tau v v^H) B for the m x ncols matrix B (leading dimension ldb).
template <typename T>
void apply_reflector(std::vector<T>& B, int ldb, int m, int ncols, const std::vector<T>& v, T tau)
{
    for (int j = 0; j < ncols; ++j) {
        T sum = Tr<T>::make(0.0, 0.0);
        for (int i = 0; i < m; ++i) sum += Tr<T>::conjv(v[i]) * B[i + static_cast<std::size_t>(j) * ldb];
        sum *= tau;
        for (int i = 0; i < m; ++i) B[i + static_cast<std::size_t>(j) * ldb] -= v[i] * sum;
    }
}

// Reflector i of the stored factorization F (leading dimension lda): QR unit entry at row i of column i; QL
// (?geqlf storage) unit entry at row m - k + i of column n - k + i, the vector above it.
template <typename T>
std::vector<T> stored_reflector(const std::vector<T>& F, int lda, int m, int n, bool lower, int i)
{
    const int k = std::min(m, n);
    std::vector<T> v(static_cast<std::size_t>(m), Tr<T>::make(0.0, 0.0));
    if (!lower) {
        v[i] = Tr<T>::make(1.0, 0.0);
        for (int r = i + 1; r < m; ++r) v[r] = F[r + static_cast<std::size_t>(i) * lda];
    } else {
        const int c = n - k + i, u = m - k + i;
        v[u] = Tr<T>::make(1.0, 0.0);
        for (int r = 0; r < u; ++r) v[r] = F[r + static_cast<std::size_t>(c) * lda];
    }
    return v;
}

// Reference-free check of a factorization F of A0 (both with leading dimension lda): A0 = Q R (QR) / Q L (QL) with
// Q built from the stored reflectors, Q unitary, a real diagonal.
template <typename T>
int check_geqrf_factors(const std::vector<T>& F, const std::vector<T>& tau, const std::vector<T>& A0, int lda, int m,
                        int n, bool lower, const std::string& tag)
{
    const int k = std::min(m, n);
    const double eps = Tr<T>::eps();
    double amax = 1.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) amax = std::max(amax, static_cast<double>(std::abs(A0[i + static_cast<std::size_t>(j) * lda])));
    int failures = 0;

    // the triangular factor (R: r <= c; L: r >= c + m - n) with the zeros the storage implies
    std::vector<T> B(static_cast<std::size_t>(m) * n, Tr<T>::make(0.0, 0.0));
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            if (lower ? (i >= j + (m - n)) : (i <= j)) B[i + static_cast<std::size_t>(j) * m] = F[i + static_cast<std::size_t>(j) * lda];
    // A = H(0)..H(k-1) R  /  A = H(k-1)..H(0) L : apply the reflectors to the factor in the order that builds Q
    std::vector<T> Q(static_cast<std::size_t>(m) * m, Tr<T>::make(0.0, 0.0));
    for (int i = 0; i < m; ++i) Q[i + static_cast<std::size_t>(i) * m] = Tr<T>::make(1.0, 0.0);
    for (int t = 0; t < k; ++t) {
        const int i = lower ? t : k - 1 - t;
        const std::vector<T> v = stored_reflector<T>(F, lda, m, n, lower, i);
        apply_reflector<T>(B, m, m, n, v, tau[i]);
        apply_reflector<T>(Q, m, m, m, v, tau[i]);
    }
    double err = 0.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            err = std::max(err, static_cast<double>(std::abs(B[i + static_cast<std::size_t>(j) * m] -
                                                              A0[i + static_cast<std::size_t>(j) * lda])));
    const double tol = 100.0 * eps * std::max(m, n) * amax;
    if (!(err <= tol)) {
        report("geqrf residual", tag + " err " + std::to_string(err) + " tol " + std::to_string(tol));
        ++failures;
    }
    double oerr = 0.0;   // Q^H Q = I
    for (int j = 0; j < m; ++j)
        for (int i = 0; i < m; ++i) {
            T sum = Tr<T>::make(0.0, 0.0);
            for (int r = 0; r < m; ++r) sum += Tr<T>::conjv(Q[r + static_cast<std::size_t>(i) * m]) * Q[r + static_cast<std::size_t>(j) * m];
            oerr = std::max(oerr, static_cast<double>(std::abs(sum - Tr<T>::make(i == j ? 1.0 : 0.0, 0.0))));
        }
    const double otol = 100.0 * eps * m;
    if (!(oerr <= otol)) {
        report("geqrf orthogonality", tag + " err " + std::to_string(oerr) + " tol " + std::to_string(otol));
        ++failures;
    }
    if (Tr<T>::cplx)
        for (int i = 0; i < k; ++i) {
            const T d = lower ? F[(m - k + i) + static_cast<std::size_t>(n - k + i) * lda] : F[i + static_cast<std::size_t>(i) * lda];
            if (std::imag(d) != 0) {
                report("geqrf diagonal not real", tag + " i=" + std::to_string(i));
                ++failures;
                break;
            }
        }
    return failures;
}

// uplo, shape, leading dimension padding; well-conditioned input (a unit-scale random matrix plus a diagonal weight on
// the factor's diagonal, so the factors are unique and comparable entrywise with host LAPACK).
template <DdlaBackend Backend, typename T>
int check_geqrf(const DdlaHandle_t& h, char uplo, int m, int n, int pad, std::uint64_t seed)
{
    const bool lower = (uplo == 'L');
    const int lda = m + pad;
    const int k = std::min(m, n);
    std::vector<T> A0 = random_vec<T>(static_cast<std::size_t>(lda) * n, seed);
    for (int j = 0; j < n; ++j) {
        for (int i = m; i < lda; ++i) A0[i + static_cast<std::size_t>(j) * lda] = guard_value<T>();
        for (int i = 0; i < m; ++i)
            if (lower ? (i - j == m - n) : (i == j)) A0[i + static_cast<std::size_t>(j) * lda] += Tr<T>::make(4.0, 0.0);
    }
    const std::string tag = std::string("uplo=") + uplo + " " + dims(m, n, pad);

    Buf<T> dA(h, A0.size()), dtau(h, static_cast<std::size_t>(k));
    dA.up(A0);
    dtau.up(std::vector<T>(static_cast<std::size_t>(k), guard_value<T>()));
    geqrf<Backend, T>(h, uplo, m, n, dA.p, lda, dtau.p);
    ddlaSynchronize(h);
    std::vector<T> F, tau;
    dA.down(F);
    dtau.down(tau);

    int failures = 0;
    for (int j = 0; j < n; ++j)
        for (int i = m; i < lda; ++i)
            if (!same_bits(F[i + static_cast<std::size_t>(j) * lda], guard_value<T>())) {
                report("geqrf padding written", tag + " at (" + std::to_string(i) + "," + std::to_string(j) + ")");
                return failures + 1;
            }
    failures += check_geqrf_factors<T>(F, tau, A0, lda, m, n, lower, tag);

    // against host LAPACK: same convention, so tau and every stored entry agree to rounding
    std::vector<T> L = A0, ltau;
    host_geqrf<T>(lower, m, n, L, lda, ltau);
    double derr = 0.0, terr = 0.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            derr = std::max(derr, static_cast<double>(std::abs(F[i + static_cast<std::size_t>(j) * lda] -
                                                                L[i + static_cast<std::size_t>(j) * lda])));
    for (int i = 0; i < k; ++i) terr = std::max(terr, static_cast<double>(std::abs(tau[i] - ltau[i])));
    const double ctol = 100.0 * Tr<T>::eps() * std::max(m, n);
    if (!(derr <= ctol) || !(terr <= ctol)) {
        report("geqrf vs LAPACK", tag + " factor err " + std::to_string(derr) + " tau err " + std::to_string(terr) +
                                      " tol " + std::to_string(ctol));
        ++failures;
    }
    return failures;
}

// A matrix built to stress the reflector scalars of one column; see check_geqrf_special.
enum SpecialKind { SP_ALPHA0, SP_ALPHA_NEGZERO, SP_X0, SP_X0_IMAG, SP_SUBNORMAL, SP_ALL_SUBNORMAL, SP_HUGE, SP_MIXED };

inline const char* special_name(int kind)
{
    static const char* names[] = {"alpha=0", "alpha=-0", "x=0", "x=0,Im(alpha)", "subnormal-x", "all-subnormal", "huge-x",
                                  "mixed-scale"};
    return names[kind];
}

// One tall matrix whose critical column holds alpha (the pivot row: row 0 for QR, row m - 1 for QL) and x (the rest of the
// column, a vector of length m - 1); the other column is O(1).  The critical reflector (its column and its tau) is compared
// with host LAPACK to 8 eps relative to the reference entry, plus two quanta of the smallest subnormal where an entry is
// subnormal (a result below the normal range cannot be more precise than that, and two correct implementations may
// round it one quantum apart); the exactly decided cases (x = 0 with a real alpha: tau = 0 and an untouched column) must
// match LAPACK bitwise.  The other column and its tau are O(1): 8 eps relative plus 64 eps absolute.
template <DdlaBackend Backend, typename T>
int check_geqrf_special(const DdlaHandle_t& h, char uplo, int kind)
{
    using R = typename Tr<T>::real;
    const bool lower = (uplo == 'L');
    const int m = 9, n = 2, k = 2;
    const R tiny = std::is_same<R, float>::value ? R(1e-40) : R(1e-310);
    const R huge = std::is_same<R, float>::value ? R(1e30) : R(1e300);
    const int cc = lower ? n - 1 : 0;   // the critical column: the first reflector LAPACK forms
    const int alpha_row = lower ? m - 1 : 0;
    std::vector<T> A0(static_cast<std::size_t>(m) * n);
    Rng r(4242 + kind);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) A0[i + static_cast<std::size_t>(j) * m] = Tr<T>::make(r.uni() + (i == j ? 3.0 : 0.0), r.uni());
    for (int i = 0; i < m; ++i) {
        const double u = 0.5 + 0.5 * r.uni(), w = 0.25 * r.uni();
        T val = Tr<T>::make(0.0, 0.0);
        const bool is_alpha = (i == alpha_row);
        switch (kind) {
        case SP_ALPHA0: val = is_alpha ? Tr<T>::make(0.0, 0.0) : Tr<T>::make(u, w); break;
        case SP_ALPHA_NEGZERO: val = is_alpha ? Tr<T>::make(-0.0, 0.0) : Tr<T>::make(u, w); break;
        case SP_X0: val = is_alpha ? Tr<T>::make(2.5, 0.0) : Tr<T>::make(0.0, 0.0); break;
        case SP_X0_IMAG: val = is_alpha ? Tr<T>::make(2.5, 1.5) : Tr<T>::make(0.0, 0.0); break;
        case SP_SUBNORMAL: val = is_alpha ? Tr<T>::make(1.25, 0.0) : Tr<T>::make(double(tiny) * u, double(tiny) * w); break;
        case SP_ALL_SUBNORMAL: val = Tr<T>::make(double(tiny) * (is_alpha ? 1.0 : u), double(tiny) * (is_alpha ? 0.0 : w)); break;
        case SP_HUGE: val = Tr<T>::make(double(huge) * (is_alpha ? 0.75 : u), double(huge) * (is_alpha ? 0.0 : w)); break;
        case SP_MIXED: val = Tr<T>::make((is_alpha ? 0.75 : u) * ((i % 3 == 0) ? double(tiny) : (i % 3 == 1) ? 1.0 : double(huge) * 1e-8),
                                         0.5 * w); break;
        }
        A0[i + static_cast<std::size_t>(cc) * m] = val;
    }

    Buf<T> dA(h, A0.size()), dtau(h, static_cast<std::size_t>(k));
    dA.up(A0);
    dtau.up(std::vector<T>(static_cast<std::size_t>(k), guard_value<T>()));
    geqrf<Backend, T>(h, uplo, m, n, dA.p, m, dtau.p);
    ddlaSynchronize(h);
    std::vector<T> F, tau;
    dA.down(F);
    dtau.down(tau);
    std::vector<T> L = A0, ltau;
    host_geqrf<T>(lower, m, n, L, m, ltau);

    const std::string tag = std::string("uplo=") + uplo + " " + special_name(kind);
    if (kind == SP_ALPHA_NEGZERO && Backend == DdlaBackend::GPU) {
        // A -0.0 pivot: LAPACK (gfortran SIGN) takes it as negative, the vendor geqrf as +0, so the sign of the
        // pivot -- and the whole reflector chain after it -- may differ.  The result must still be a valid
        // factorization of A0 with |pivot| = ||x|| (the reflector of the critical column is the first one formed).
        int bad = check_geqrf_factors<T>(F, tau, A0, m, m, n, lower, tag);
        const int prow = lower ? m - 1 : 0;
        const double got = std::abs(F[prow + static_cast<std::size_t>(cc) * m]);
        const double ref = std::abs(L[prow + static_cast<std::size_t>(cc) * m]);
        if (!(std::fabs(got - ref) <= 8.0 * Tr<T>::eps() * ref)) {
            report("geqrf -0.0 pivot magnitude", tag + " got " + std::to_string(got) + " want " + std::to_string(ref));
            ++bad;
        }
        return bad;
    }
    const bool exact = (kind == SP_X0) || (kind == SP_X0_IMAG && !Tr<T>::cplx);
    const double eps = Tr<T>::eps();
    const double quantum = 2.0 * static_cast<double>(std::numeric_limits<R>::denorm_min());
    const int crit_tau = lower ? k - 1 : 0;
    int failures = 0;
    auto close = [&](const T& a, const T& ref, double floor_abs, bool bitwise) {
        if (same_bits(a, ref)) return true;
        if (bitwise) return false;
        return static_cast<double>(std::abs(a - ref)) <= 8.0 * eps * static_cast<double>(std::abs(ref)) + floor_abs;
    };
    for (int j = 0; j < n; ++j) {
        const bool crit = (j == cc);
        for (int i = 0; i < m; ++i) {
            const std::size_t at = i + static_cast<std::size_t>(j) * m;
            if (!close(F[at], L[at], crit ? quantum : 64.0 * eps, crit && exact)) {
                report("geqrf special vs LAPACK", tag + " entry (" + std::to_string(i) + "," + std::to_string(j) + ") got " +
                                                      std::to_string(static_cast<double>(std::abs(F[at]))) + " want " +
                                                      std::to_string(static_cast<double>(std::abs(L[at]))));
                ++failures;
            }
        }
    }
    for (int i = 0; i < k; ++i) {
        const bool crit = (i == crit_tau);
        if (!close(tau[i], ltau[i], crit ? quantum : 64.0 * eps, crit && exact)) {
            report("geqrf special tau vs LAPACK", tag + " tau[" + std::to_string(i) + "] got " + std::to_string(static_cast<double>(std::real(tau[i]))) +
                                                      " want " + std::to_string(static_cast<double>(std::real(ltau[i]))));
            ++failures;
        }
    }
    return failures;
}

// -- larft ----------------------------------------------------------------------------
// V holds the stored trapezoid (strictly lower for 'F', strictly upper for 'B') in random numbers and a guard value
// everywhere else -- the implied unit entry, the other triangle, the padding rows -- so a read of an implied entry
// shows.  tau(i) is a ?larfg-style scalar (H(i) unitary), zero for every fifth reflector.
template <typename T>
std::complex<double> to_cd(const T& x) { return std::complex<double>(static_cast<double>(std::real(x)), static_cast<double>(std::imag(x))); }

template <DdlaBackend Backend, typename T>
int check_larft(const DdlaHandle_t& h, char direct, int n, int k, int padv, int padt, std::uint64_t seed)
{
    typedef std::complex<double> cd;
    const bool bwd = (direct == 'B');
    const int ldv = n + padv, ldt = k + padt;
    const std::size_t LV = static_cast<std::size_t>(ldv), LT = static_cast<std::size_t>(ldt);
    const std::string tag = std::string("direct=") + direct + " n=" + std::to_string(n) + " k=" + std::to_string(k) +
                            " padv=" + std::to_string(padv) + " padt=" + std::to_string(padt);
    auto unit_row = [&](int c) { return bwd ? n - k + c : c; };
    auto stored = [&](int i, int c) { return bwd ? i < unit_row(c) : i > unit_row(c); };

    std::vector<T> V0 = random_vec<T>(LV * k, seed);
    for (int c = 0; c < k; ++c)
        for (int i = 0; i < ldv; ++i)
            if (i >= n || !stored(i, c)) V0[i + c * LV] = guard_value<T>();
    // the explicit V of the reflectors, and tau
    std::vector<cd> X(static_cast<std::size_t>(n) * k, cd(0.0, 0.0));
    std::vector<T> tau(static_cast<std::size_t>(k));
    Rng rng(seed + 77);
    for (int c = 0; c < k; ++c) {
        double nrm2 = 1.0;
        X[unit_row(c) + static_cast<std::size_t>(c) * n] = cd(1.0, 0.0);
        for (int i = 0; i < n; ++i)
            if (stored(i, c)) {
                X[i + static_cast<std::size_t>(c) * n] = to_cd(V0[i + c * LV]);
                nrm2 += std::norm(X[i + static_cast<std::size_t>(c) * n]);
            }
        const double phi = 3.14159265358979 * rng.uni();
        const double u = 0.5 + 0.5 * rng.uni();
        double tr, ti = 0.0;
        if (Tr<T>::cplx) { tr = (1.0 + std::cos(phi)) / nrm2; ti = std::sin(phi) / nrm2; }   // |tau ||v||^2 - 1| = 1
        else tr = 2.0 * u / nrm2;                                                            // 1 - tau ||v||^2 in [-1, 1]
        if (c % 5 == 3) { tr = 0.0; ti = 0.0; }
        tau[c] = Tr<T>::make(tr, ti);
    }
    std::vector<T> T0(LT * k, guard_value<T>());

    Buf<T> dV(h, V0.size()), dtau(h, tau.size()), dT(h, T0.size());
    dV.up(V0);
    dtau.up(tau);
    dT.up(T0);
    larft<Backend, T>(h, direct, 'C', n, k, dV.p, ldv, dtau.p, dT.p, ldt);
    ddlaSynchronize(h);
    std::vector<T> V1, Tc;
    dV.down(V1);
    dT.down(Tc);

    int failures = 0;
    if (std::memcmp(V1.data(), V0.data(), V0.size() * sizeof(T)) != 0) {
        report("larft V modified", tag);
        ++failures;
    }
    const T zero = Tr<T>::make(0.0, 0.0);
    for (int c = 0; c < k; ++c) {
        for (int r = 0; r < k; ++r) {
            const T x = Tc[r + c * LT];
            const bool own = bwd ? r >= c : r <= c;
            if (!own && !(x == zero)) { report("larft other triangle not zero", tag); return failures + 1; }
        }
        if (!(Tc[c + c * LT] == tau[c])) { report("larft T(i,i) != tau(i)", tag + " i=" + std::to_string(c)); ++failures; }
        for (int r = k; r < ldt; ++r)
            if (!same_bits(Tc[r + c * LT], guard_value<T>())) { report("larft T padding written", tag); return failures + 1; }
    }

    // textbook recurrence in double: T(i,i) = tau(i); 'F': T(0:i-1,i) = -tau(i) T(0:i-1,0:i-1) (X(:,0:i-1)^H x(i));
    // 'B': T(i+1:k-1,i) = -tau(i) T(i+1:k-1,i+1:k-1) (X(:,i+1:k-1)^H x(i))
    std::vector<cd> R(static_cast<std::size_t>(k) * k, cd(0.0, 0.0));
    auto gram = [&](int a, int b) {   // x(a)^H x(b)
        cd s(0.0, 0.0);
        for (int i = 0; i < n; ++i) s += std::conj(X[i + static_cast<std::size_t>(a) * n]) * X[i + static_cast<std::size_t>(b) * n];
        return s;
    };
    for (int step = 0; step < k; ++step) {
        const int i = bwd ? k - 1 - step : step;
        const cd ti = to_cd(tau[i]);
        R[i + static_cast<std::size_t>(i) * k] = ti;
        if (ti == cd(0.0, 0.0)) continue;
        const int lo = bwd ? i + 1 : 0, hi = bwd ? k : i;   // the rows already built
        std::vector<cd> w(static_cast<std::size_t>(k), cd(0.0, 0.0));
        for (int a = lo; a < hi; ++a) w[a] = gram(a, i);
        for (int r = lo; r < hi; ++r) {
            cd s(0.0, 0.0);
            for (int a = lo; a < hi; ++a) s += R[r + static_cast<std::size_t>(a) * k] * w[a];
            R[r + static_cast<std::size_t>(i) * k] = -ti * s;
        }
    }
    double tmax = 1e-300;
    for (const cd& v : R) tmax = std::max(tmax, std::abs(v));
    double terr = 0.0;
    for (int c = 0; c < k; ++c)
        for (int r = 0; r < k; ++r) terr = std::max(terr, std::abs(to_cd(Tc[r + c * LT]) - R[r + static_cast<std::size_t>(c) * k]));
    const double eps = Tr<T>::eps();
    const double ttol = 16.0 * eps * (n + k) * tmax;
    if (!(terr <= ttol)) {
        report("larft T vs textbook", tag + " err " + std::to_string(terr) + " tol " + std::to_string(ttol));
        ++failures;
    }

    // the definition: H(1)...H(k) (F) / H(k)...H(1) (B) multiplied out equals I - V T V^H
    const std::size_t N = static_cast<std::size_t>(n);
    std::vector<cd> Q(N * N, cd(0.0, 0.0));
    for (std::size_t i = 0; i < N; ++i) Q[i + i * N] = cd(1.0, 0.0);
    std::vector<cd> y(N);
    for (int step = 0; step < k; ++step) {
        const int i = bwd ? k - 1 - step : step;
        const cd ti = to_cd(tau[i]);
        for (std::size_t r = 0; r < N; ++r) {   // Q := Q H(i) = Q - tau (Q v) v^H
            cd s(0.0, 0.0);
            for (std::size_t c = 0; c < N; ++c) s += Q[r + c * N] * X[c + static_cast<std::size_t>(i) * n];
            y[r] = ti * s;
        }
        for (std::size_t c = 0; c < N; ++c) {
            const cd vc = std::conj(X[c + static_cast<std::size_t>(i) * n]);
            for (std::size_t r = 0; r < N; ++r) Q[r + c * N] -= y[r] * vc;
        }
    }
    std::vector<cd> W(N * k, cd(0.0, 0.0));   // W = V T (T as computed)
    for (int b = 0; b < k; ++b)
        for (int a = 0; a < k; ++a) {
            const cd tab = to_cd(Tc[a + b * LT]);
            if (tab == cd(0.0, 0.0)) continue;
            for (std::size_t r = 0; r < N; ++r) W[r + static_cast<std::size_t>(b) * N] += X[r + static_cast<std::size_t>(a) * n] * tab;
        }
    double qerr = 0.0;
    for (std::size_t c = 0; c < N; ++c)
        for (std::size_t r = 0; r < N; ++r) {
            cd s(0.0, 0.0);
            for (int b = 0; b < k; ++b) s += W[r + static_cast<std::size_t>(b) * N] * std::conj(X[c + static_cast<std::size_t>(b) * n]);
            qerr = std::max(qerr, std::abs(Q[r + c * N] - (cd(r == c ? 1.0 : 0.0, 0.0) - s)));
        }
    const double qtol = 64.0 * eps * (n + k);
    if (!(qerr <= qtol)) {
        report("larft H != I - V T V^H", tag + " err " + std::to_string(qerr) + " tol " + std::to_string(qtol));
        ++failures;
    }
    return failures;
}

// -- empty problems ---------------------------------------------------------------------
template <DdlaBackend Backend, typename T>
int check_empty(const DdlaHandle_t& h)
{
    int failures = 0;
    Buf<T> dA(h, 16);
    Buf<int> dpiv(h, 4), dinfo(h, 1);
    auto info_after = [&](const char* what, const std::function<void()>& call) {
        dinfo.up(std::vector<int>(1, -77));
        call();
        ddlaSynchronize(h);
        std::vector<int> info;
        dinfo.down(info);
        if (info[0] != 0) {
            report(what, "info not set to 0 for an empty problem");
            ++failures;
        }
    };
    info_after("getrf m=0", [&] { getrf<Backend, T>(h, 0, 3, dA.p, 1, dpiv.p, dinfo.p); });
    info_after("getrf n=0", [&] { getrf<Backend, T>(h, 3, 0, dA.p, 3, dpiv.p, dinfo.p); });
    info_after("potrf n=0", [&] { potrf<Backend, T>(h, 'L', 0, dA.p, 1, dinfo.p); });
    // geqrf: nothing to factor, null pointers allowed.
    try {
        geqrf<Backend, T>(h, 'U', 0, 3, static_cast<T*>(nullptr), 1, static_cast<T*>(nullptr));
        geqrf<Backend, T>(h, 'L', 3, 0, static_cast<T*>(nullptr), 3, static_cast<T*>(nullptr));
    } catch (const std::exception& e) {
        report("geqrf empty", e.what());
        ++failures;
    }
    // larft: no reflectors, null pointers allowed.
    try {
        larft<Backend, T>(h, 'F', 'C', 5, 0, static_cast<T*>(nullptr), 5, static_cast<const T*>(nullptr), static_cast<T*>(nullptr), 1);
        larft<Backend, T>(h, 'B', 'C', 0, 0, static_cast<T*>(nullptr), 1, static_cast<const T*>(nullptr), static_cast<T*>(nullptr), 1);
    } catch (const std::exception& e) {
        report("larft empty", e.what());
        ++failures;
    }
    // laswp: nothing to do, nothing to touch (k1 > k2, n = 0), null pointers allowed.
    try {
        laswp<Backend, T>(h, 4, static_cast<T*>(nullptr), 4, 3, 2, nullptr, 1);
        laswp<Backend, T>(h, 0, static_cast<T*>(nullptr), 4, 1, 4, nullptr, 1);
    } catch (const std::exception& e) {
        report("laswp empty", e.what());
        ++failures;
    }
    return failures;
}

template <DdlaBackend Backend, typename T>
int run_type(const DdlaHandle_t& h)
{
    int failures = 0;
    struct Shape { int m, n; };
    const Shape shapes[] = {{1, 1}, {1, 7}, {7, 1}, {5, 5}, {37, 37}, {64, 64}, {50, 20}, {20, 50}, {130, 130}, {200, 150}};
    std::uint64_t seed = 1000;
    for (const Shape& s : shapes) {
        failures += check_getrf<Backend, T>(h, s.m, s.n, 0, -1, ++seed);
        failures += check_getrf<Backend, T>(h, s.m, s.n, 5, -1, ++seed);
    }
    failures += check_getrf<Backend, T>(h, 40, 40, 3, 17, ++seed);     // exactly singular: first zero pivot at 18
    failures += check_getrf<Backend, T>(h, 33, 33, 0, 0, ++seed);      // ... and at 1

    const int potrf_sizes[] = {1, 2, 5, 33, 64, 100};
    for (int n : potrf_sizes)
        for (char uplo : {'U', 'L'}) {
            failures += check_potrf<Backend, T>(h, uplo, n, 0, -1, ++seed);
            failures += check_potrf<Backend, T>(h, uplo, n, 4, -1, ++seed);
        }
    for (char uplo : {'U', 'L'}) {
        failures += check_potrf<Backend, T>(h, uplo, 8, 2, 3, ++seed);   // leading minor of order 4 fails
        failures += check_potrf<Backend, T>(h, uplo, 8, 0, 0, ++seed);   // ... order 1
    }

    struct Sw { int m, n, pad, k1, k2, incx; };
    const Sw sw[] = {
        {40, 33, 3, 1, 40, 1},  {40, 33, 0, 5, 31, 1},  {40, 33, 4, 1, 40, 2},  {40, 33, 0, 3, 29, 2},
        {40, 33, 2, 1, 40, -1}, {40, 33, 0, 5, 31, -2}, {40, 1, 0, 1, 40, 1},   {7, 5, 1, 2, 2, 1},
        {64, 130, 0, 1, 64, 1}, {64, 130, 3, 2, 63, -1},
    };
    for (const Sw& c : sw) failures += check_laswp<Backend, T>(h, c.m, c.n, c.pad, c.k1, c.k2, c.incx, ++seed);

    // geqrf: square / tall / wide, QR and QL, with and without padding (the largest shapes exceed one cuSOLVER panel)
    const Shape qshapes[] = {{1, 1}, {1, 6}, {6, 1}, {5, 5}, {37, 37}, {64, 20}, {20, 64}, {50, 20}, {130, 33}, {96, 96}};
    for (const Shape& sh : qshapes)
        for (char uplo : {'U', 'L'}) {
            failures += check_geqrf<Backend, T>(h, uplo, sh.m, sh.n, 0, ++seed);
            failures += check_geqrf<Backend, T>(h, uplo, sh.m, sh.n, 3, ++seed);
        }
    for (char uplo : {'U', 'L'})
        for (int kind = SP_ALPHA0; kind <= SP_MIXED; ++kind) failures += check_geqrf_special<Backend, T>(h, uplo, kind);

    // larft: a few reflectors up to more than one kernel block, both directions, padded and compact V / T
    struct LShape { int n, k; };
    const LShape lshapes[] = {{1, 1}, {2, 1}, {3, 3}, {8, 5}, {37, 32}, {37, 37}, {64, 33}, {130, 64}, {200, 128}, {330, 300}};
    int li = 0;
    for (const LShape& sh : lshapes)
        for (char d : {'F', 'B'}) {
            failures += check_larft<Backend, T>(h, d, sh.n, sh.k, (li & 1) ? 5 : 0, (li & 2) ? 3 : 0, ++seed);
            ++li;
        }

    failures += check_empty<Backend, T>(h);
    return failures;
}

template <DdlaBackend Backend>
int run_backend()
{
    DdlaHandle_t h = nullptr;
    ddlaInit(h, Backend);
    ddlaSet(h, MPI_COMM_WORLD);
    int failures = 0;
    failures += run_type<Backend, float>(h);
    failures += run_type<Backend, double>(h);
    failures += run_type<Backend, std::complex<float>>(h);
    failures += run_type<Backend, std::complex<double>>(h);
    ddlaDestroy(h);
    if (g_rank == 0)
        std::cout << "  " << (Backend == DdlaBackend::CPU ? "CPU" : "GPU") << " backend: "
                  << (failures == 0 ? "ok" : "FAILED") << std::endl;
    return failures;
}

// -- argument checks --------------------------------------------------------------------
template <DdlaBackend Backend>
int check_rejections(const DdlaHandle_t& h, const DdlaHandle_t& other)
{
    // Argument checks fire before any pointer is dereferenced, so host buffers are fine on either backend.
    double x[64] = {0};
    int ip[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    int info = 0;
    int failures = 0;
    auto expect_throw = [&](const std::function<void()>& fn, const char* what) {
        bool threw = false;
        try { fn(); }
        catch (const std::runtime_error& e) {
            threw = std::string(e.what()).find(what) != std::string::npos;
        }
        if (!threw) {
            if (g_rank == 0) std::cerr << "  FAIL rejection: expected \"" << what << "\"" << std::endl;
            ++failures;
        }
    };
    // getrf
    expect_throw([&] { getrf<Backend, double>(nullptr, 2, 2, x, 2, ip, &info); }, "null handle");
    expect_throw([&] { getrf<Backend, double>(h, -1, 2, x, 2, ip, &info); }, "negative");
    expect_throw([&] { getrf<Backend, double>(h, 2, 2, x, 2, ip, nullptr); }, "null info");
    expect_throw([&] { getrf<Backend, double>(h, 2, 2, nullptr, 2, ip, &info); }, "null A");
    expect_throw([&] { getrf<Backend, double>(h, 2, 2, x, 2, nullptr, &info); }, "null ipiv");
    expect_throw([&] { getrf<Backend, double>(h, 4, 2, x, 3, ip, &info); }, "lda too small");
    // potrf
    expect_throw([&] { potrf<Backend, double>(nullptr, 'L', 2, x, 2, &info); }, "null handle");
    expect_throw([&] { potrf<Backend, double>(h, 'X', 2, x, 2, &info); }, "invalid uplo");
    expect_throw([&] { potrf<Backend, double>(h, 'L', -1, x, 2, &info); }, "negative");
    expect_throw([&] { potrf<Backend, double>(h, 'L', 2, x, 2, nullptr); }, "null info");
    expect_throw([&] { potrf<Backend, double>(h, 'L', 2, nullptr, 2, &info); }, "null A");
    expect_throw([&] { potrf<Backend, double>(h, 'U', 4, x, 3, &info); }, "lda too small");
    // geqrf
    expect_throw([&] { geqrf<Backend, double>(nullptr, 'U', 2, 2, x, 2, x + 8); }, "null handle");
    expect_throw([&] { geqrf<Backend, double>(h, 'X', 2, 2, x, 2, x + 8); }, "uplo must be");
    expect_throw([&] { geqrf<Backend, double>(h, 'U', -1, 2, x, 2, x + 8); }, "negative");
    expect_throw([&] { geqrf<Backend, double>(h, 'L', 2, -1, x, 2, x + 8); }, "negative");
    expect_throw([&] { geqrf<Backend, double>(h, 'U', 2, 2, nullptr, 2, x + 8); }, "null A");
    expect_throw([&] { geqrf<Backend, double>(h, 'L', 2, 2, x, 2, nullptr); }, "null tau");
    expect_throw([&] { geqrf<Backend, double>(h, 'U', 4, 2, x, 3, x + 8); }, "lda too small");
    // larft
    const double* const ctau = x + 40;
    expect_throw([&] { larft<Backend, double>(nullptr, 'F', 'C', 4, 2, x, 4, ctau, x + 48, 2); }, "null handle");
    expect_throw([&] { larft<Backend, double>(h, 'X', 'C', 4, 2, x, 4, ctau, x + 48, 2); }, "direct must be");
    expect_throw([&] { larft<Backend, double>(h, 'F', 'R', 4, 2, x, 4, ctau, x + 48, 2); }, "storev = 'C'");
    expect_throw([&] { larft<Backend, double>(h, 'F', 'C', -1, 2, x, 4, ctau, x + 48, 2); }, "negative");
    expect_throw([&] { larft<Backend, double>(h, 'B', 'C', 4, -1, x, 4, ctau, x + 48, 2); }, "negative");
    expect_throw([&] { larft<Backend, double>(h, 'F', 'C', 2, 3, x, 4, ctau, x + 48, 3); }, "k must not exceed n");
    expect_throw([&] { larft<Backend, double>(h, 'F', 'C', 4, 2, nullptr, 4, ctau, x + 48, 2); }, "null V");
    expect_throw([&] { larft<Backend, double>(h, 'F', 'C', 4, 2, x, 4, nullptr, x + 48, 2); }, "null tau");
    expect_throw([&] { larft<Backend, double>(h, 'B', 'C', 4, 2, x, 4, ctau, nullptr, 2); }, "null T");
    expect_throw([&] { larft<Backend, double>(h, 'F', 'C', 4, 2, x, 3, ctau, x + 48, 2); }, "ldv too small");
    expect_throw([&] { larft<Backend, double>(h, 'F', 'C', 4, 2, x, 4, ctau, x + 48, 1); }, "ldt too small");
    // laswp
    expect_throw([&] { laswp<Backend, double>(nullptr, 2, x, 4, 1, 2, ip, 1); }, "null handle");
    expect_throw([&] { laswp<Backend, double>(h, 2, x, 4, 0, 2, ip, 1); }, "k1 must be");
    expect_throw([&] { laswp<Backend, double>(h, 2, x, 4, 1, 2, ip, 0); }, "incx must be nonzero");
    expect_throw([&] { laswp<Backend, double>(h, 2, nullptr, 4, 1, 2, ip, 1); }, "null A");
    expect_throw([&] { laswp<Backend, double>(h, 2, x, 4, 1, 2, nullptr, 1); }, "null ipiv");
    expect_throw([&] { laswp<Backend, double>(h, 2, x, 3, 1, 4, ip, 1); }, "lda too small");
    // swap
    expect_throw([&] { swap<Backend, double>(nullptr, 2, x, 1, x + 8, 1); }, "null handle");
    expect_throw([&] { swap<Backend, double>(h, 2, nullptr, 1, x + 8, 1); }, "null x or y");
    expect_throw([&] { swap<Backend, double>(h, 2, x, 0, x + 8, 1); }, "nonzero");
    expect_throw([&] { swap<Backend, double>(h, 2, x, 1, x + 8, 0); }, "nonzero");
    if (other != nullptr) {
        expect_throw([&] { getrf<Backend, double>(other, 2, 2, x, 2, ip, &info); }, "does not match");
        expect_throw([&] { potrf<Backend, double>(other, 'L', 2, x, 2, &info); }, "does not match");
        expect_throw([&] { laswp<Backend, double>(other, 2, x, 4, 1, 2, ip, 1); }, "does not match");
        expect_throw([&] { geqrf<Backend, double>(other, 'U', 2, 2, x, 2, x + 8); }, "does not match");
        expect_throw([&] { larft<Backend, double>(other, 'F', 'C', 4, 2, x, 4, ctau, x + 48, 2); }, "does not match");
        expect_throw([&] { swap<Backend, double>(other, 2, x, 1, x + 8, 1); }, "does not match");
    }
    return failures;
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);

    const char* only = std::getenv("DDLA_TEST_ONLY");
    const bool do_cpu = (!only || std::string(only) == "cpu");
    const bool do_gpu = (!only || std::string(only) == "gpu");
    int failures = 0;
#if DDLA_HAS_CPU
    if (do_cpu) failures += run_backend<DdlaBackend::CPU>();
#else
    (void)do_cpu;
#endif
#if DDLA_HAS_GPU
    if (do_gpu) failures += run_backend<DdlaBackend::GPU>();
#else
    (void)do_gpu;
#endif

    {   // argument checks: one handle of each available backend; in a dual build each is also handed the
        // other backend's handle to prove the mismatch is rejected.
        DdlaHandle_t cpu = nullptr, gpu = nullptr;
#if DDLA_HAS_CPU
        ddlaInit(cpu, DdlaBackend::CPU);
        ddlaSet(cpu, MPI_COMM_WORLD);
#endif
#if DDLA_HAS_GPU
        ddlaInit(gpu, DdlaBackend::GPU);
        ddlaSet(gpu, MPI_COMM_WORLD);
#endif
#if DDLA_HAS_CPU
        failures += check_rejections<DdlaBackend::CPU>(cpu, gpu);
#endif
#if DDLA_HAS_GPU
        failures += check_rejections<DdlaBackend::GPU>(gpu, cpu);
#endif
        if (cpu != nullptr) ddlaDestroy(cpu);
        if (gpu != nullptr) ddlaDestroy(gpu);
    }

    int global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (g_rank == 0) {
        if (global_failures == 0)
            std::cout << "test_backend_lapack passed" << std::endl;
        else
            std::cout << "test_backend_lapack FAILED: " << global_failures << " case(s)" << std::endl;
    }
    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}
