// ---------------------------------------------------------------------------
// Backend-templated coverage of ddla::geam:  C := alpha*op(A) + beta*op(B).
//
// geam replaced omatcopy (the scaled transpose-copy, now beta == 0) and has an
// OpenMP-threaded CPU arm, so this file checks:
//   * all nine op(A) x op(B) combinations, odd and tile-multiple shapes, padded
//     leading dimensions with the padding checked untouched (tile bounds),
//   * the single-operand forms the library uses (B == nullptr, beta == 0 with a
//     B that must be ignored, alpha == 1 copies),
//   * alpha == 0 and in-place C == A,
//   * the CPU arm's exact-copy contract (-0.0, Inf, NaN survive bit for bit),
//   * thread-count independence: the same large problem on a 1-thread and a
//     4-thread handle gives identical bits (CPU),
//   * the argument checks (null handle, backend mismatch, bad flag, null B,
//     leading dimensions).
// The reference arithmetic is written out by hand, not taken from the library.
//
// Follows the structure of test_backend_blas3.cpp.  Debug aids:
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
#include <vector>

#include <mpi.h>

#include <ddla/ddla.h>

#include "ddla_stream.h"   // DdlaStream::host_threads

using namespace ddla;

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
    static const bool cplx = false;
};
template <> struct Tr<double> {
    typedef double real;
    static double make(double a, double) { return a; }
    static double conjv(double x) { return x; }
    static const bool cplx = false;
};
template <> struct Tr<std::complex<float>> {
    typedef float real;
    static std::complex<float> make(double a, double b) { return std::complex<float>(float(a), float(b)); }
    static std::complex<float> conjv(const std::complex<float>& x) { return std::conj(x); }
    static const bool cplx = true;
};
template <> struct Tr<std::complex<double>> {
    typedef double real;
    static std::complex<double> make(double a, double b) { return std::complex<double>(a, b); }
    static std::complex<double> conjv(const std::complex<double>& x) { return std::conj(x); }
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
T opval(const std::vector<T>& X, int ld, char op, int i, int j)
{
    if (op == 'N') return X[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * ld];
    const T x = X[static_cast<std::size_t>(j) + static_cast<std::size_t>(i) * ld];
    return op == 'C' ? Tr<T>::conjv(x) : x;
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
            throw std::runtime_error("test_backend_geam: ddlaMalloc failed");
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

void report(const char* what, char ta, char tb, int m, int n, const std::string& detail)
{
    if (g_rank != 0) return;
    std::cerr << "  FAIL " << what << " ta=" << ta << " tb=" << tb << " m=" << m << " n=" << n
              << " " << detail << std::endl;
}

// -- one numerical case -------------------------------------------------------
// mode 0: general (both operands used)      mode 1: B == nullptr, beta == 0
// mode 2: beta == 0 with a non-null B that must be ignored
template <DdlaBackend Backend, typename T>
int check_case(const DdlaHandle_t& h, char ta, char tb, int m, int n, int mode, T alpha, T beta,
               int pada, int padb, int padc, std::uint64_t seed)
{
    typedef typename Tr<T>::real R;
    const int rowsA = ta == 'N' ? m : n, colsA = ta == 'N' ? n : m;
    const int rowsB = tb == 'N' ? m : n, colsB = tb == 'N' ? n : m;
    const int lda = rowsA + pada, ldb = rowsB + padb, ldc = m + padc;
    const std::vector<T> A = random_vec<T>(static_cast<std::size_t>(lda) * colsA, seed);
    const std::vector<T> B = random_vec<T>(static_cast<std::size_t>(ldb) * colsB, seed + 1);
    const T sentinel = Tr<T>::make(7.75, -3.5);
    const std::vector<T> C0(static_cast<std::size_t>(ldc) * n, sentinel);

    Buf<T> dA(h, A.size()), dB(h, B.size()), dC(h, C0.size());
    dA.up(A); dB.up(B); dC.up(C0);
    const bool use_b = (mode == 0);
    const T b_coef = use_b ? beta : Tr<T>::make(0.0, 0.0);
    const T* pB = (mode == 1) ? nullptr : dB.p;
    geam<Backend, T>(h, ta, tb, m, n, alpha, dA.p, lda, b_coef, pB, ldb, dC.p, ldc);
    std::vector<T> C;
    dC.down(C);

    const double eps = std::numeric_limits<R>::epsilon();
    int bad = 0;
    for (int j = 0; j < n && bad == 0; ++j) {
        for (int i = 0; i < ldc; ++i) {
            const T got = C[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * ldc];
            if (i >= m) {                                    // padding rows must be untouched
                if (std::memcmp(&got, &sentinel, sizeof(T)) != 0) {
                    report("padding overwritten", ta, tb, m, n, "at (" + std::to_string(i) + "," + std::to_string(j) + ")");
                    bad = 1;
                    break;
                }
                continue;
            }
            const T a = opval(A, lda, ta, i, j);
            T ref = alpha * a;
            double mag = std::abs(alpha) * std::abs(a);
            if (use_b) {
                const T b = opval(B, ldb, tb, i, j);
                ref = ref + beta * b;
                mag += std::abs(beta) * std::abs(b);
            }
            const double err = std::abs(got - ref);
            if (!(err <= 16.0 * eps * mag + 1e-30)) {
                report("value", ta, tb, m, n, "at (" + std::to_string(i) + "," + std::to_string(j) +
                       ") mode=" + std::to_string(mode) + " err=" + std::to_string(err));
                bad = 1;
                break;
            }
        }
    }
    return bad;
}

// alpha == 0: op(A) is not referenced (it holds NaN here), C := beta*op(B).  CPU only: the vendor
// routine computes 0*a and would propagate the NaN.
template <typename T>
int check_alpha_zero_cpu(const DdlaHandle_t& h, char ta, char tb, int m, int n)
{
    const int lda = (ta == 'N' ? m : n), ldb = (tb == 'N' ? m : n);
    typedef typename Tr<T>::real R;
    std::vector<T> A(static_cast<std::size_t>(lda) * (ta == 'N' ? n : m),
                     Tr<T>::make(std::numeric_limits<R>::quiet_NaN(), std::numeric_limits<R>::quiet_NaN()));
    const std::vector<T> B = random_vec<T>(static_cast<std::size_t>(ldb) * (tb == 'N' ? n : m), 91);
    const T beta = Tr<T>::make(-1.5, 0.25);
    std::vector<T> C(static_cast<std::size_t>(m) * n, Tr<T>::make(0, 0));
    Buf<T> dA(h, A.size()), dB(h, B.size()), dC(h, C.size());
    dA.up(A); dB.up(B); dC.up(C);
    geam<DdlaBackend::CPU, T>(h, ta, tb, m, n, Tr<T>::make(0, 0), dA.p, lda, beta, dB.p, ldb, dC.p, m);
    dC.down(C);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            const T ref = beta * opval(B, ldb, tb, i, j);
            if (!(std::abs(C[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m] - ref) <= 1e-6)) {
                report("alpha==0 read A", ta, tb, m, n, "");
                return 1;
            }
        }
    return 0;
}

// -- the exact-copy contract (CPU) --------------------------------------------
template <typename R>
R special_value(int k)
{
    switch (k) {
        case 0: return -R(0);
        case 1: return std::numeric_limits<R>::infinity();
        case 2: return -std::numeric_limits<R>::infinity();
        default: return std::numeric_limits<R>::quiet_NaN();
    }
}

template <typename T>
std::vector<T> special_vec(std::size_t n, std::uint64_t seed)
{
    typedef typename Tr<T>::real R;
    std::vector<T> v = random_vec<T>(n, seed);
    for (std::size_t e = 0; e < n; ++e) {
        R* parts = reinterpret_cast<R*>(&v[e]);        // set the exact bits: -0.0, Inf, NaN
        const int k1 = static_cast<int>(e % 11), k2 = static_cast<int>((e + 5) % 11);
        if (k1 < 4) parts[0] = special_value<R>(k1);
        if (Tr<T>::cplx && k2 < 4) parts[1] = special_value<R>(k2);
    }
    return v;
}

template <typename T>
int check_exact_copy_cpu(const DdlaHandle_t& h, char ta, int m, int n)
{
    const int rowsA = ta == 'N' ? m : n, colsA = ta == 'N' ? n : m;
    const int lda = rowsA + 2, ldc = m + 1;
    const std::vector<T> A = special_vec<T>(static_cast<std::size_t>(lda) * colsA, 5);
    std::vector<T> C(static_cast<std::size_t>(ldc) * n, Tr<T>::make(1, 1));
    Buf<T> dA(h, A.size()), dC(h, C.size());
    dA.up(A); dC.up(C);
    const T zero = Tr<T>::make(0, 0);
    geam<DdlaBackend::CPU, T>(h, ta, 'N', m, n, Tr<T>::make(1, 0), dA.p, lda, zero, nullptr, 1, dC.p, ldc);
    dC.down(C);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            const T x = ta == 'N' ? A[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * lda]
                                  : A[static_cast<std::size_t>(j) + static_cast<std::size_t>(i) * lda];
            const T want = ta == 'C' ? Tr<T>::conjv(x) : x;
            const T got = C[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * ldc];
            if (std::memcmp(&got, &want, sizeof(T)) != 0) {
                report("exact copy", ta, 'N', m, n, "at (" + std::to_string(i) + "," + std::to_string(j) + ")");
                return 1;
            }
        }
    return 0;
}

// -- in place -------------------------------------------------------------------
template <DdlaBackend Backend, typename T>
int check_inplace(const DdlaHandle_t& h, int m, int n, bool with_b)
{
    typedef typename Tr<T>::real R;
    const int ld = m + 3;
    const T alpha = Tr<T>::make(1.25, -0.5), beta = Tr<T>::make(-0.75, 0.5);
    const std::vector<T> A = random_vec<T>(static_cast<std::size_t>(ld) * n, 31);
    const std::vector<T> B = random_vec<T>(static_cast<std::size_t>(ld) * n, 32);
    Buf<T> dA(h, A.size()), dB(h, B.size());
    dA.up(A); dB.up(B);
    if (with_b)
        geam<Backend, T>(h, 'N', 'N', m, n, alpha, dA.p, ld, beta, dB.p, ld, dA.p, ld);
    else
        geam<Backend, T>(h, 'N', 'N', m, n, alpha, dA.p, ld, Tr<T>::make(0, 0), nullptr, 1, dA.p, ld);
    std::vector<T> C;
    dA.down(C);
    const double eps = std::numeric_limits<R>::epsilon();
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            const std::size_t k = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * ld;
            const T ref = with_b ? alpha * A[k] + beta * B[k] : alpha * A[k];
            const double mag = std::abs(alpha) * std::abs(A[k]) + (with_b ? std::abs(beta) * std::abs(B[k]) : 0.0);
            if (!(std::abs(C[k] - ref) <= 16.0 * eps * mag + 1e-30)) {
                report("in place", 'N', 'N', m, n, with_b ? "with B" : "alpha only");
                return 1;
            }
        }
    return 0;
}

template <DdlaBackend Backend, typename T>
int run_type(const DdlaHandle_t& h)
{
    int bad = 0;
    const char ops[3] = {'N', 'T', 'C'};
    const int shapes[][2] = {{37, 53}, {64, 64}, {32, 32}, {1, 40}, {40, 1}, {33, 31}, {65, 3}, {700, 500}, {20000, 7}};
    const T alpha = Tr<T>::make(0.7, -0.3), beta = Tr<T>::make(-1.3, 0.4);
    std::uint64_t seed = 100;

    for (const auto& sh : shapes) {
        const int m = sh[0], n = sh[1];
        for (char ta : ops) {
            for (char tb : ops) {
                const bool big = (m >= 700);   // the big cases span many tiles / strips: keep them light
                if (big && !(ta == tb || (ta == 'N' && tb == 'T'))) continue;
                bad += check_case<Backend, T>(h, ta, tb, m, n, 0, alpha, beta, 3, 2, 5, seed += 2);
            }
            // the single-operand (transpose-)copy forms: alpha general and alpha == 1
            bad += check_case<Backend, T>(h, ta, 'N', m, n, 1, alpha, beta, 2, 0, 3, seed += 2);
            bad += check_case<Backend, T>(h, ta, 'N', m, n, 1, Tr<T>::make(1, 0), beta, 4, 0, 1, seed += 2);
            bad += check_case<Backend, T>(h, ta, 'T', m, n, 2, alpha, beta, 1, 3, 2, seed += 2);
        }
    }
    // alpha == 0 (A zero-coefficient) with a finite A: both backends agree with the reference
    for (char tb : ops)
        bad += check_case<Backend, T>(h, 'N', tb, 37, 53, 0, Tr<T>::make(0, 0), beta, 1, 1, 1, seed += 2);

    bad += check_inplace<Backend, T>(h, 37, 53, true);
    bad += check_inplace<Backend, T>(h, 37, 53, false);

    // empty problems return before any pointer is looked at
    geam<Backend, T>(h, 'N', 'N', 0, 5, alpha, nullptr, 1, beta, nullptr, 1, nullptr, 1);
    geam<Backend, T>(h, 'T', 'C', 5, 0, alpha, nullptr, 1, beta, nullptr, 1, nullptr, 1);

#if DDLA_HAS_CPU
    if constexpr (Backend == DdlaBackend::CPU) {
        for (char ta : ops) {
            bad += check_exact_copy_cpu<T>(h, ta, 37, 53);
            bad += check_exact_copy_cpu<T>(h, ta, 64, 64);
            bad += check_exact_copy_cpu<T>(h, ta, 700, 500);
            bad += check_alpha_zero_cpu<T>(h, ta, 'N', 37, 53);
        }
        bad += check_alpha_zero_cpu<T>(h, 'T', 'C', 40, 33);
    }
#endif
    return bad;
}

template <DdlaBackend Backend>
int run_backend()
{
    DdlaHandle_t handle = nullptr;
    ddlaInit(handle, Backend);
    ddlaSet(handle, MPI_COMM_WORLD);
    int failures = 0;
    failures += run_type<Backend, float>(handle);
    failures += run_type<Backend, double>(handle);
    failures += run_type<Backend, std::complex<float>>(handle);
    failures += run_type<Backend, std::complex<double>>(handle);
    ddlaSynchronize(handle);
    int sum = 0;
    MPI_Allreduce(&failures, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (g_rank == 0)
        std::cerr << "  geam " << (Backend == DdlaBackend::CPU ? "CPU" : "GPU") << " CHECKS_FAILED=" << sum
                  << (sum == 0 ? "  (all numerical checks passed)" : "  *** some checks FAILED ***") << std::endl;
    ddlaDestroy(handle);
    return failures;
}

#if DDLA_HAS_CPU
// The same large problems on a 1-thread and a 4-thread handle give identical bits, and the handle really
// carries the thread budget it was asked for.
template <typename T>
int check_thread_invariance(const DdlaHandle_t& h1, const DdlaHandle_t& h4, int m, int n)
{
    int bad = 0;
    const char combos[][2] = {{'N', 'N'}, {'T', 'N'}, {'C', 'N'}, {'C', 'T'}, {'N', 'C'}};
    const T alpha = Tr<T>::make(0.7, -0.3), beta = Tr<T>::make(-1.3, 0.4);
    for (const auto& cb : combos) {
        const char ta = cb[0], tb = cb[1];
        const int lda = (ta == 'N' ? m : n) + 1, ldb = (tb == 'N' ? m : n) + 2, ldc = m + 1;
        const std::vector<T> A = random_vec<T>(static_cast<std::size_t>(lda) * (ta == 'N' ? n : m), 61);
        const std::vector<T> B = random_vec<T>(static_cast<std::size_t>(ldb) * (tb == 'N' ? n : m), 62);
        std::vector<T> out[2];
        const DdlaHandle_t* hs[2] = {&h1, &h4};
        for (int k = 0; k < 2; ++k) {
            Buf<T> dA(*hs[k], A.size()), dB(*hs[k], B.size()), dC(*hs[k], static_cast<std::size_t>(ldc) * n);
            dA.up(A); dB.up(B);
            std::vector<T> zero(static_cast<std::size_t>(ldc) * n, Tr<T>::make(0, 0));
            dC.up(zero);
            geam<DdlaBackend::CPU, T>(*hs[k], ta, tb, m, n, alpha, dA.p, lda, beta, dB.p, ldb, dC.p, ldc);
            dC.down(out[k]);
        }
        if (std::memcmp(out[0].data(), out[1].data(), out[0].size() * sizeof(T)) != 0) {
            report("threads 1 vs 4 differ", ta, tb, m, n, "");
            ++bad;
        }
    }
    return bad;
}

int check_threads()
{
    int bad = 0;
    setenv("DDLA_EIG_THREADS", "1", 1);
    DdlaHandle_t h1 = nullptr;
    ddlaInit(h1, DdlaBackend::CPU);
    ddlaSet(h1, MPI_COMM_WORLD);
    setenv("DDLA_EIG_THREADS", "4", 1);
    DdlaHandle_t h4 = nullptr;
    ddlaInit(h4, DdlaBackend::CPU);
    ddlaSet(h4, MPI_COMM_WORLD);
    unsetenv("DDLA_EIG_THREADS");
    if (h1->host_threads != 1 || h4->host_threads != 4) {
        if (g_rank == 0)
            std::cerr << "  FAIL host_threads: got " << h1->host_threads << " and " << h4->host_threads
                      << ", expected 1 and 4" << std::endl;
        ++bad;
    }
    const int sizes[][2] = {{700, 500}, {20000, 7}};   // tiles in both directions / strips split along a column
    for (const auto& sz : sizes) {
        bad += check_thread_invariance<float>(h1, h4, sz[0], sz[1]);
        bad += check_thread_invariance<double>(h1, h4, sz[0], sz[1]);
        bad += check_thread_invariance<std::complex<float>>(h1, h4, sz[0], sz[1]);
        bad += check_thread_invariance<std::complex<double>>(h1, h4, sz[0], sz[1]);
    }
    ddlaDestroy(h1);
    ddlaDestroy(h4);
    return bad;
}
#endif

// -- argument checks --------------------------------------------------------------
template <DdlaBackend Backend>
int check_rejections(const DdlaHandle_t& h, const DdlaHandle_t& other)
{
    double x[16] = {0};
    const double one = 1.0, zero = 0.0;
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
    expect_throw([&] { geam<Backend, double>(nullptr, 'N', 'N', 2, 2, one, x, 2, zero, nullptr, 1, x, 2); },
                 "null handle");
    expect_throw([&] { geam<Backend, double>(h, 'X', 'N', 2, 2, one, x, 2, zero, nullptr, 1, x, 2); },
                 "invalid trans");
    expect_throw([&] { geam<Backend, double>(h, 'N', 'Q', 2, 2, one, x, 2, zero, nullptr, 1, x, 2); },
                 "invalid trans");
    expect_throw([&] { geam<Backend, double>(h, 'N', 'N', 2, 2, one, x, 2, one, nullptr, 2, x + 8, 2); },
                 "null B");
    expect_throw([&] { geam<Backend, double>(h, 'N', 'N', 2, 2, one, nullptr, 2, zero, nullptr, 1, x, 2); },
                 "null A");
    expect_throw([&] { geam<Backend, double>(h, 'N', 'N', 4, 2, one, x, 3, zero, nullptr, 1, x + 8, 4); },
                 "lda too small");
    expect_throw([&] { geam<Backend, double>(h, 'T', 'N', 4, 2, one, x, 1, zero, nullptr, 1, x + 8, 4); },
                 "lda too small");
    expect_throw([&] { geam<Backend, double>(h, 'N', 'N', 4, 2, one, x, 4, one, x, 2, x + 8, 4); },
                 "ldb too small");
    expect_throw([&] { geam<Backend, double>(h, 'N', 'N', 4, 2, one, x, 4, zero, nullptr, 1, x + 8, 3); },
                 "ldc too small");
    if (other != nullptr)
        expect_throw([&] { geam<Backend, double>(other, 'N', 'N', 2, 2, one, x, 2, zero, nullptr, 1, x + 8, 2); },
                     "does not match");
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
    if (do_cpu) {
        failures += run_backend<DdlaBackend::CPU>();
        failures += check_threads();
    }
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
            std::cout << "test_backend_geam passed" << std::endl;
        else
            std::cout << "test_backend_geam FAILED: " << global_failures << " case(s)" << std::endl;
    }
    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}
