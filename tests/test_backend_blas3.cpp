// ---------------------------------------------------------------------------
// Backend-templated coverage for the BLAS-3 families that had NO test before
// the wrapper unification: trsm, syrk, trmm and gemmBatched.
//
// The other converted families are already covered numerically by
// test_api_grid_pvector_redist.cpp (gemv, hemv, trmv, nrm2, dotc, her2, herk,
// hemm, her2k); these four were the gap, so this file is their only evidence.
// TRMM checks its public in-place contract, including poisoned unused input
// entries and padding, independently of its QR/SVD/product callers.
//
// Each check builds a small matrix on the host, computes the expected result
// with an independent reference loop, runs the routine through the public
// templated entry, and compares.  The reference arithmetic is written out by
// hand rather than reusing library helpers, so a shared bug cannot hide.
//
// Follows the structure and tolerances of test_backend_gemm.cpp.
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <cstdio>

#include <mpi.h>

#include <ddla/ddla.h>

#include "ddla_stream.h"   // DdlaStream::host_threads (private)
#include "host_threads.h"       // detail::ScopedBlasThreads (private)

using namespace ddla;

namespace {

// Debug aids, inert unless their env vars are set.  They exist so a failure in
// this file can be bisected without editing it:
//   DDLA_TEST_TRACE=1        print per-case diagnostics to stderr
//   DDLA_TEST_ONLY=cpu|gpu   run one backend alone
//   DDLA_TEST_ONLY_FAMILY=x  run one family alone (trsm|trmm|syrk|gemmBatched)
inline bool trace_on()
{
    static const bool on = std::getenv("DDLA_TEST_TRACE") != nullptr;
    return on;
}

inline bool family_selected(const char* fam)
{
    const char* only = std::getenv("DDLA_TEST_ONLY_FAMILY");
    return !only || std::string(only) == fam;
}

inline void trace(const char* what, const char* a = "", const char* b = "",
                  const char* c = "", const char* d = "")
{
    if (!trace_on()) return;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) return;
    std::cerr << "  [" << what << "] " << a << " " << b << " " << c << " " << d
              << std::endl;
}

template <typename T>
T value(int i, int j, int salt)
{
    return T(((i * 17 + j * 13 + salt * 7) % 19) - 9);
}

template <>
std::complex<float> value(int i, int j, int salt)
{
    return {static_cast<float>(((i * 17 + j * 13 + salt * 7) % 19) - 9),
            static_cast<float>(((i * 11 + j * 23 + salt * 5) % 17) - 8)};
}

template <>
std::complex<double> value(int i, int j, int salt)
{
    return {static_cast<double>(((i * 17 + j * 13 + salt * 7) % 19) - 9),
            static_cast<double>(((i * 11 + j * 23 + salt * 5) % 17) - 8)};
}

template <typename T> T conj_v(T x) { return x; }
template <> std::complex<float> conj_v(std::complex<float> x) { return std::conj(x); }
template <> std::complex<double> conj_v(std::complex<double> x) { return std::conj(x); }

// Tolerances match test_backend_gemm.cpp.
template <typename T> constexpr double tol()
{
    constexpr bool single = std::is_same_v<T, float> ||
                            std::is_same_v<T, std::complex<float>>;
    return single ? 2e-4 : 2e-11;
}

template <typename T>
bool close_enough(const std::vector<T>& got, const std::vector<T>& want)
{
    double err = 0.0, scale = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        err = std::max(err, static_cast<double>(std::abs(got[i] - want[i])));
        scale = std::max(scale, static_cast<double>(std::abs(want[i])));
    }
    double gerr = 0.0, gscale = 0.0;
    MPI_Allreduce(&err, &gerr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&scale, &gscale, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return gerr <= tol<T>() * std::max(1.0, gscale);
}

// Upload a host vector, synchronously.  Small sizes only (this file is about
// flag semantics, not throughput).
// Upload exactly `count` elements.  Every caller passes the device buffer's
// own element count, so a host vector that is accidentally larger than its
// buffer cannot silently overrun it (that produced a heap abort inside
// ddlaFree when the sizes disagreed).
template <typename T>
bool upload(const DdlaHandle_t& handle, T* d, const std::vector<T>& h, std::size_t count)
{
    if (h.size() < count) throw std::runtime_error("upload: host vector too small");
    const ddlaStatus_t st = ddlaMemcpy(handle, d, h.data(), count * sizeof(T));
    const ddlaStatus_t sy = ddlaSynchronize(handle);
    return st == ddlaStatus_t::DDLA_STATUS_SUCCESS &&
           sy == ddlaStatus_t::DDLA_STATUS_SUCCESS;
}

template <typename T>
bool download(const DdlaHandle_t& handle, std::vector<T>& h, const T* d, std::size_t count)
{
    if (h.size() < count) throw std::runtime_error("download: host vector too small");
    if (ddlaMemcpy(handle, h.data(), d, count * sizeof(T)) != ddlaStatus_t::DDLA_STATUS_SUCCESS)
        return false;
    return ddlaSynchronize(handle) == ddlaStatus_t::DDLA_STATUS_SUCCESS;
}

template <typename T>
T* dev_alloc(const DdlaHandle_t& handle, std::size_t n)
{
    T* p = nullptr;
    const ddlaStatus_t st = ddlaMalloc(handle, reinterpret_cast<void**>(&p), n * sizeof(T));
    if (st != ddlaStatus_t::DDLA_STATUS_SUCCESS) {
        throw std::runtime_error("dev_alloc failed");
    }
    return p;
}

// A well-conditioned triangular matrix: strictly-diagonally-dominant, so the
// solve is stable and the reference is trustworthy.  off_diag scales the
// strictly-triangular part; diag_base keeps the diagonal away from zero.
template <typename T>
std::vector<T> make_triangular(int n, int ld, char uplo, char diag, int salt)
{
    std::vector<T> a(static_cast<std::size_t>(ld) * n, T(0));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const bool in_tri = (uplo == 'U') ? (i <= j) : (i >= j);
            if (!in_tri) continue;
            if (i == j) {
                if (diag == 'U') {
                    a[i + j * ld] = T(1);
                } else {
                    // real, comfortably non-zero diagonal
                    a[i + j * ld] = T(static_cast<double>(n) + 1.0 + (j % 3));
                }
            } else {
                a[i + j * ld] = value<T>(i, j, salt) * T(0.125);
            }
        }
    }
    return a;
}

// op(A)[i][j] for a triangular operand stored with the given uplo.
template <typename T>
T tri_op(const std::vector<T>& a, int ld, char uplo, char trans, char diag, int n,
         int i, int j)
{
    // element (i,j) of op(A); for the transpose/conjugate-transpose cases read
    // (j,i) and conjugate when 'C'
    int r = i, c = j;
    if (trans != 'N') { r = j; c = i; }
    const bool in_tri = (uplo == 'U') ? (r <= c) : (r >= c);
    if (!in_tri) return T(0);
    if (diag == 'U' && r == c) return T(1);
    const T x = a[r + c * ld];
    return trans == 'C' ? conj_v(x) : x;
}

// ---------------------------------------------------------------------------
// Shared scratch buffers: allocated once per (family, scalar type) and reused
// across every flag combination.  The stream-ordered allocation pool behind
// ddlaMallocAsync returns DDLA_STATUS_INTERNAL_ERROR once enough small
// allocations are in flight, so a per-case allocate/free cycle measures the
// allocator rather than the routine.
// ---------------------------------------------------------------------------
constexpr int kN = 6;      // order of the triangular operand
constexpr int kM = 6;      // rows of the rectangular operand (max)
constexpr int kNn = 6;     // cols of the rectangular operand (max)
constexpr int kLd = 6;

// Element counts for the gemmBatched case, defined ONCE for both the allocator
// and the filler.  Keeping these in a single place is what prevents the
// buffer-size mismatch that previously overran a device buffer.
constexpr int kBatchM = 5, kBatchN = 4, kBatchK = 3, kBatchCount = 3;
constexpr int kBatchSa = (kBatchM > kBatchK ? kBatchM : kBatchK) * kN;
constexpr int kBatchSb = (kBatchK > kBatchN ? kBatchK : kBatchN) * kN;
constexpr int kBatchSc = kBatchM * kBatchN;

// Total elements this test needs, laid out as one arena so there is exactly ONE
// device allocation per scalar type.  See the note above on why.
constexpr std::size_t kTriElems = static_cast<std::size_t>(kLd) * kN;
constexpr std::size_t kBElems = static_cast<std::size_t>(kLd) * kNn;
constexpr std::size_t kCElems = kBElems;
// each gemmBatched pointer array holds kBatchCount pointers
constexpr std::size_t kPtrBytes = static_cast<std::size_t>(kBatchCount) * sizeof(void*);
constexpr std::size_t kArenaElems =
    kTriElems + kBElems + kCElems +
    static_cast<std::size_t>(kBatchCount) * (kBatchSa + kBatchSb + kBatchSc) +
    3 * ((kPtrBytes + sizeof(std::complex<double>) - 1) / sizeof(std::complex<double>));

// A typed view over the shared byte arena.  All four scalar types share ONE
// device allocation, made before any work is queued and released afterwards.
template <typename T>
struct Scratch {
    T* tri = nullptr;
    T* b = nullptr;
    T* c = nullptr;
    T* batch_a[kBatchCount] = {};
    T* batch_b[kBatchCount] = {};
    T* batch_c[kBatchCount] = {};
    void* ptr_a = nullptr;   // device pointer arrays for gemmBatched
    void* ptr_b = nullptr;
    void* ptr_c = nullptr;
    std::size_t tri_n = kTriElems, b_n = kBElems, c_n = kCElems;

    // carve from a byte base; the caller guarantees it is large enough
    void carve(void* base)
    {
        T* p = reinterpret_cast<T*>(base);
        std::size_t off = 0;
        auto take = [&](std::size_t n) { T* q = p + off; off += n; return q; };
        tri = take(kTriElems);
        b = take(kBElems);
        c = take(kCElems);
        for (int i = 0; i < kBatchCount; ++i) {
            batch_a[i] = take(kBatchSa);
            batch_b[i] = take(kBatchSb);
            batch_c[i] = take(kBatchSc);
        }
        // pointer arrays: one batch-worth of T* each, carved off the end
        ptr_a = take(kPtrBytes / sizeof(T));
        ptr_b = take(kPtrBytes / sizeof(T));
        ptr_c = take(kPtrBytes / sizeof(T));
    }
};

// ---------------------------------------------------------------------------
// trsm: B := alpha * op(A)^{-1} B  (side='L')
//       B := alpha * B op(A)^{-1}  (side='R')
// trsm's A is declared non-const in the public signature (the vendor
// prototypes require it), so the caller must own a mutable A.
// ---------------------------------------------------------------------------
template <DdlaBackend Backend, typename T>
int check_trsm(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag,
               Scratch<T>& sc)
{
    const int m = (side == 'L') ? kN : 4;
    const int nn = (side == 'L') ? 4 : kN;

    const std::vector<T> a = make_triangular<T>(kN, kLd, uplo, diag, 1);
    std::vector<T> b(static_cast<std::size_t>(kLd) * nn);
    for (int j = 0; j < nn; ++j)
        for (int i = 0; i < m; ++i)
            b[i + j * kLd] = value<T>(i, j, 2);

    T alpha = T(0.75);
    if constexpr (std::is_same_v<T, std::complex<float>> ||
                  std::is_same_v<T, std::complex<double>>)
        alpha = T(0.75, -0.125);

    // Reference: invert op(A) explicitly on this tiny size, then multiply.
    // Multiplying rather than substituting keeps the reference independent of
    // the routine's own algorithm.
    std::vector<T> aop(static_cast<std::size_t>(kLd) * kN);
    for (int j = 0; j < kN; ++j)
        for (int i = 0; i < kN; ++i)
            aop[i + j * kLd] = tri_op<T>(a, kLd, uplo, trans, diag, kN, i, j);

    std::vector<T> inv(static_cast<std::size_t>(kLd) * kN, T(0));
    for (int i = 0; i < kN; ++i) inv[i + i * kLd] = T(1);
    std::vector<T> w = aop;
    for (int col = 0; col < kN; ++col) {
        const T piv = w[col + col * kLd];
        for (int j = 0; j < kN; ++j) { w[col + j * kLd] /= piv; inv[col + j * kLd] /= piv; }
        for (int r = 0; r < kN; ++r) {
            if (r == col) continue;
            const T f = w[r + col * kLd];
            if (f == T(0)) continue;
            for (int j = 0; j < kN; ++j) {
                w[r + j * kLd] -= f * w[col + j * kLd];
                inv[r + j * kLd] -= f * inv[col + j * kLd];
            }
        }
    }

    std::vector<T> want(static_cast<std::size_t>(kLd) * nn);
    for (int j = 0; j < nn; ++j) {
        for (int i = 0; i < m; ++i) {
            T sum = T(0);
            if (side == 'L') {
                for (int l = 0; l < kN; ++l) sum += inv[i + l * kLd] * b[l + j * kLd];
            } else {
                for (int l = 0; l < kN; ++l) sum += b[i + l * kLd] * inv[l + j * kLd];
            }
            want[i + j * kLd] = alpha * sum;
        }
    }

    // upload only the m x nn sub-blocks actually used
    std::vector<T> a_up(a.size(), T(0));
    for (int j = 0; j < kN; ++j)
        for (int i = 0; i < kN; ++i) a_up[i + j * kLd] = a[i + j * kLd];
    std::vector<T> b_up(sc.b_n, T(0));
    for (int j = 0; j < nn; ++j)
        for (int i = 0; i < m; ++i) b_up[i + j * kLd] = b[i + j * kLd];

    bool ok = upload(handle, sc.tri, a_up, sc.tri_n) &&
              upload(handle, sc.b, b_up, sc.b_n);
    if (ok) {
        trsm<Backend, T>(handle, side, uplo, trans, diag, m, nn, alpha,
                         sc.tri, kLd, sc.b, kLd);
        ok = ddlaSynchronize(handle) == ddlaStatus_t::DDLA_STATUS_SUCCESS;
    }
    std::vector<T> got(sc.b_n);
    if (ok) ok = download(handle, got, sc.b, sc.b_n);
    if (!ok) return 1;

    std::vector<T> want_full(sc.b_n, T(0));
    for (int j = 0; j < nn; ++j)
        for (int i = 0; i < m; ++i) want_full[i + j * kLd] = want[i + j * kLd];
    return close_enough<T>(got, want_full) ? 0 : 1;
}

// ---------------------------------------------------------------------------
// syrk (REAL ONLY): C := alpha*A*A^T + beta*C ('N')
//                    C := alpha*A^T*A + beta*C ('T')
// Only the uplo triangle of C is referenced; the other half is pre-zeroed so a
// library that wrote outside the triangle fails the comparison.
// ---------------------------------------------------------------------------
template <DdlaBackend Backend, typename T>
int check_syrk(const DdlaHandle_t& handle, char uplo, char trans, Scratch<T>& sc)
{
    const int k = 4;
    const int rows = (trans == 'N') ? kN : k;
    const int cols = (trans == 'N') ? k : kN;
    const int lda = kLd;

    std::vector<T> a(static_cast<std::size_t>(kLd) * kN, T(0));
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
            a[i + j * lda] = value<T>(i, j, 1);

    std::vector<T> c_seed(static_cast<std::size_t>(kLd) * kN, T(0));
    std::vector<T> c_in(static_cast<std::size_t>(kLd) * kN, T(0));
    for (int j = 0; j < kN; ++j) {
        for (int i = 0; i < kN; ++i) {
            const bool in_tri = (uplo == 'U') ? (i <= j) : (i >= j);
            if (in_tri) {
                c_in[i + j * kLd] = value<T>(i, j, 3);
                c_seed[i + j * kLd] = c_in[i + j * kLd];
            }
        }
    }

    const T alpha = T(0.75);
    const T beta = T(-0.25);

    std::vector<T> want(static_cast<std::size_t>(kLd) * kN, T(0));
    for (int j = 0; j < kN; ++j) {
        for (int i = 0; i < kN; ++i) {
            const bool in_tri = (uplo == 'U') ? (i <= j) : (i >= j);
            if (!in_tri) continue;
            T sum = T(0);
            for (int l = 0; l < k; ++l) {
                const T x = (trans == 'N') ? a[i + l * lda] : a[l + i * lda];
                const T y = (trans == 'N') ? a[j + l * lda] : a[l + j * lda];
                sum += x * y;
            }
            want[i + j * kLd] = alpha * sum + beta * c_in[i + j * kLd];
        }
    }

    bool ok = upload(handle, sc.tri, a, sc.tri_n) && upload(handle, sc.c, c_seed, sc.c_n);
    if (ok) {
        syrk<Backend, T>(handle, uplo, trans, kN, k, alpha, sc.tri, lda, beta, sc.c, kLd);
        ok = ddlaSynchronize(handle) == ddlaStatus_t::DDLA_STATUS_SUCCESS;
    }
    std::vector<T> got(sc.c_n);
    if (ok) ok = download(handle, got, sc.c, sc.c_n);
    if (!ok) return 1;
    return close_enough<T>(got, want) ? 0 : 1;
}

// ---------------------------------------------------------------------------
// trmm overwrites B; A and every B entry outside the logical matrix stay intact.
template <DdlaBackend Backend, typename T>
int check_trmm(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag,
               Scratch<T>& sc, bool zero_alpha = false)
{
    using R = typename real_type<T>::type;
    const T poison(std::numeric_limits<R>::quiet_NaN());
    const int m = (side == 'L') ? kN : 4;
    const int nn = (side == 'L') ? 4 : kN;
    std::vector<T> a = make_triangular<T>(kN, kLd, uplo, diag, 1);
    for (int j = 0; j < kN; ++j)
        for (int i = 0; i < kN; ++i)
            if ((uplo == 'U' ? i > j : i < j) || (diag == 'U' && i == j))
                a[i + j * kLd] = poison;
    std::vector<T> b_up(sc.b_n, T(-3.25));
    for (int j = 0; j < nn; ++j)
        for (int i = 0; i < m; ++i)
            b_up[i + j * kLd] = zero_alpha ? poison : value<T>(i, j, 2);
    T alpha = T(0.75);
    if constexpr (!std::is_arithmetic_v<T>) alpha = T(0.75, -0.125);
    if (zero_alpha) alpha = T(0);
    std::vector<T> want = b_up;
    for (int j = 0; j < nn; ++j) {
        for (int i = 0; i < m; ++i) {
            T sum(0);
            if (!zero_alpha) {
                if (side == 'L')
                    for (int l = 0; l < kN; ++l)
                        sum += tri_op<T>(a, kLd, uplo, trans, diag, kN, i, l) * b_up[l + j * kLd];
                else
                    for (int l = 0; l < kN; ++l)
                        sum += b_up[i + l * kLd] * tri_op<T>(a, kLd, uplo, trans, diag, kN, l, j);
            }
            want[i + j * kLd] = alpha * sum;
        }
    }
    std::vector<T> a_up(sc.tri_n, poison);
    std::copy(a.begin(), a.end(), a_up.begin());
    bool ok = upload(handle, sc.tri, a_up, sc.tri_n) && upload(handle, sc.b, b_up, sc.b_n);
    if (ok) {
        trmm<Backend, T>(handle, side, uplo, trans, diag, m, nn, alpha,
                         zero_alpha ? nullptr : sc.tri, kLd, sc.b, kLd);
        ok = ddlaSynchronize(handle) == ddlaStatus_t::DDLA_STATUS_SUCCESS;
    }
    std::vector<T> got(sc.b_n), after_a(sc.tri_n);
    if (ok) ok = download(handle, got, sc.b, sc.b_n) && download(handle, after_a, sc.tri, sc.tri_n);
    if (ok) ok = std::memcmp(after_a.data(), a_up.data(), sc.tri_n * sizeof(T)) == 0;
    for (std::size_t k = 0; ok && k < got.size(); ++k) {
        const bool active = k / kLd < static_cast<std::size_t>(nn) && k % kLd < static_cast<std::size_t>(m);
        if (active) ok = std::isfinite(static_cast<double>(std::abs(got[k])));
        else ok = std::memcmp(&got[k], &b_up[k], sizeof(T)) == 0;
    }
    // Every rank still enters the same comparison collectives on a guard failure.
    const bool numeric = close_enough<T>(got, want);
    return ok && numeric ? 0 : 1;
}

template <DdlaBackend Backend, typename T>
int check_trmm_edges(const DdlaHandle_t& h)
{
    trmm<Backend,T>(h,'l','u','n','n',0,2,T(1),nullptr,1,nullptr,1);
    trmm<Backend,T>(h,'L','U','N','N',3,0,T(1),nullptr,3,nullptr,3);
    T x(1);int failures = 0;
    auto reject = [&](auto call) {
        bool caught = false;
        try { call(); } catch (const std::exception&) { caught = true; }
        if (!caught) ++failures;
    };
    reject([&] { trmm<Backend,T>(h,'X','U','N','N',1,1,T(1),&x,1,&x,1); });
    reject([&] { trmm<Backend,T>(h,'L','X','N','N',1,1,T(1),&x,1,&x,1); });
    reject([&] { trmm<Backend,T>(h,'L','U','X','N',1,1,T(1),&x,1,&x,1); });
    reject([&] { trmm<Backend,T>(h,'L','U','N','X',1,1,T(1),&x,1,&x,1); });
    reject([&] { trmm<Backend,T>(h,'L','U','N','N',-1,1,T(1),&x,1,&x,1); });
    reject([&] { trmm<Backend,T>(h,'L','U','N','N',2,1,T(1),&x,1,&x,2); });
    reject([&] { trmm<Backend,T>(h,'L','U','N','N',2,1,T(1),&x,2,&x,1); });
    reject([&] { trmm<Backend,T>(h,'L','U','N','N',1,1,T(1),nullptr,1,&x,1); });
    reject([&] { trmm<Backend,T>(h,'L','U','N','N',1,1,T(1),&x,1,nullptr,1); });
    reject([&] { trmm<Backend,T>(nullptr,'L','U','N','N',0,0,T(1),nullptr,1,nullptr,1); });
    return failures;
}

// gemmBatched, driven the way the proven probe and src/ppotrf.cpp drive it:
// separate device buffers per matrix, and the ARRAY OF POINTERS copied into
// device memory (cuBLAS/hipBLAS read the array from there; the CPU arm
// dereferences the pointers itself).  Kept separate from the scratch-based
// checks because the vendor batched entry points have this extra requirement.
// ---------------------------------------------------------------------------
template <typename T>
T gb_value(int i, int j, int salt)
{
    return value<T>(i, j, salt);
}

template <DdlaBackend Backend, typename T>
int check_gemmBatched_standalone(const DdlaHandle_t& handle)
{
    const int m = 4, n = 3, k = 5, batch = 2;   // a single ('N','N') shape
    const int lda = m, ldb = k, ldc = m;

    std::vector<std::vector<T>> ha(batch), hb(batch), hc(batch), ref(batch);
    for (int b = 0; b < batch; ++b) {
        ha[b].assign(static_cast<std::size_t>(lda) * k, T(0));
        hb[b].assign(static_cast<std::size_t>(ldb) * n, T(0));
        hc[b].assign(static_cast<std::size_t>(ldc) * n, T(0));
        for (int j = 0; j < k; ++j)
            for (int i = 0; i < m; ++i) ha[b][i + j * lda] = gb_value<T>(i, j, b + 1);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < k; ++i) hb[b][i + j * ldb] = gb_value<T>(i, j, b + 5);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) hc[b][i + j * ldc] = gb_value<T>(i, j, b + 9);
    }
    T alpha = T(2), beta = T(3);
    if constexpr (std::is_same_v<T, std::complex<float>> ||
                  std::is_same_v<T, std::complex<double>>)
        { alpha = T(2, -0.5); beta = T(3, 0.25); }

    for (int b = 0; b < batch; ++b) {
        ref[b] = hc[b];
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                T sum = T(0);
                for (int l = 0; l < k; ++l) sum += ha[b][i + l * lda] * hb[b][l + j * ldb];
                ref[b][i + j * ldc] = alpha * sum + beta * ref[b][i + j * ldc];
            }
    }

    std::vector<T*> da(batch), db(batch), dc(batch);
    for (int b = 0; b < batch; ++b) {
        da[b] = dev_alloc<T>(handle, ha[b].size());
        db[b] = dev_alloc<T>(handle, hb[b].size());
        dc[b] = dev_alloc<T>(handle, hc[b].size());
        upload(handle, da[b], ha[b], ha[b].size());
        upload(handle, db[b], hb[b], hb[b].size());
        upload(handle, dc[b], hc[b], hc[b].size());
    }
    T** d_aptr = nullptr;
    T** d_bptr = nullptr;
    T** d_cptr = nullptr;
    ddlaMalloc(handle, reinterpret_cast<void**>(&d_aptr), batch * sizeof(T*));
    ddlaMalloc(handle, reinterpret_cast<void**>(&d_bptr), batch * sizeof(T*));
    ddlaMalloc(handle, reinterpret_cast<void**>(&d_cptr), batch * sizeof(T*));

    const T* const* A_ptr = da.data();
    const T* const* B_ptr = db.data();
    T* const* C_ptr = dc.data();
    bool ok = true;
    if constexpr (Backend == DdlaBackend::GPU) {
        ok = ddlaMemcpy(handle, d_aptr, da.data(), batch * sizeof(T*))
               == ddlaStatus_t::DDLA_STATUS_SUCCESS &&
             ddlaMemcpy(handle, d_bptr, db.data(), batch * sizeof(T*))
               == ddlaStatus_t::DDLA_STATUS_SUCCESS &&
             ddlaMemcpy(handle, d_cptr, dc.data(), batch * sizeof(T*))
               == ddlaStatus_t::DDLA_STATUS_SUCCESS &&
             ddlaSynchronize(handle) == ddlaStatus_t::DDLA_STATUS_SUCCESS;
        A_ptr = const_cast<const T* const*>(d_aptr);
        B_ptr = const_cast<const T* const*>(d_bptr);
        C_ptr = const_cast<T* const*>(d_cptr);
    }
    if (ok) {
        gemmBatched<Backend, T>(handle, 'N', 'N', m, n, k, alpha,
                                A_ptr, lda, B_ptr, ldb, beta, C_ptr, ldc, batch);
        ok = ddlaSynchronize(handle) == ddlaStatus_t::DDLA_STATUS_SUCCESS;
    }
    int bad = 0;
    for (int b = 0; b < batch && ok; ++b) {
        std::vector<T> got(hc[b].size());
        if (!download(handle, got, dc[b], got.size()) || !close_enough<T>(got, ref[b]))
            ++bad;
    }
    if (!ok) ++bad;

    for (int b = 0; b < batch; ++b) {
        ddlaFree(handle, da[b]);
        ddlaFree(handle, db[b]);
        ddlaFree(handle, dc[b]);
    }
    ddlaFree(handle, d_aptr);
    ddlaFree(handle, d_bptr);
    ddlaFree(handle, d_cptr);
    return bad;
}

#if DDLA_HAS_CPU
// ---------------------------------------------------------------------------
// gemmBatched on a CPU handle runs on the handle's host-thread budget (DdlaStream::host_threads): every OpenMP thread computes whole
// products while the BLAS is held to one thread, and the BLAS thread count is put back afterwards.  On a realistic batch (every op
// combination, padded leading dimensions, batch sizes 0 / 1 / fewer than the threads / many) this checks that a 1-thread and a 4-thread
// handle both match an independent reference, that the two results are BITWISE equal (each product is one single-threaded BLAS call
// either way), that C's padding is untouched, that the BLAS thread count after a call is the one before it, and that
// detail::ScopedBlasThreads pins it inside its scope.
// ---------------------------------------------------------------------------
template <typename T>
int check_gemmBatched_threads()
{
    int bad = 0;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto fail = [&](const char* what, char ta, char tb, int batch) {
        ++bad;
        if (rank == 0)
            std::cerr << "  FAIL gemmBatched threads: " << what << " ('" << ta << "','" << tb << "', batch " << batch << ")" << std::endl;
    };
    // the budget is read once, by ddlaSet
    auto make = [](const char* threads) {
        setenv("DDLA_EIG_THREADS", threads, 1);
        DdlaHandle_t h = nullptr;
        ddlaInit(h, DdlaBackend::CPU);
        ddlaSet(h, MPI_COMM_WORLD);
        return h;
    };
    DdlaHandle_t h1 = make("1");
    DdlaHandle_t h4 = make("4");
    unsetenv("DDLA_EIG_THREADS");
    if (h1->host_threads != 1 || h4->host_threads != 4) fail("host_threads of the two handles", '-', '-', 0);

    const int m = 9, n = 7, k = 11;
    T alpha = T(2), beta = T(3);
    if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)
        { alpha = T(2, -0.5); beta = T(3, 0.25); }

    struct Op { char ta, tb; };
    const Op ops[] = {{'N', 'N'}, {'N', 'C'}, {'C', 'N'}, {'T', 'T'}};
    for (const Op& op : ops) {
        for (int batch : {0, 1, 3, 37}) {
            const int a_rows = (op.ta == 'N') ? m : k, a_cols = (op.ta == 'N') ? k : m;
            const int b_rows = (op.tb == 'N') ? k : n, b_cols = (op.tb == 'N') ? n : k;
            const int lda = a_rows + 1, ldb = b_rows + 2, ldc = m + 3;   // padded: the padding must stay untouched
            const std::size_t asz = static_cast<std::size_t>(lda) * a_cols;
            const std::size_t bsz = static_cast<std::size_t>(ldb) * b_cols;
            const std::size_t csz = static_cast<std::size_t>(ldc) * n;
            std::vector<T> A(batch * asz), B(batch * bsz), C0(batch * csz);
            for (std::size_t e = 0; e < A.size(); ++e) A[e] = value<T>(static_cast<int>(e % 23), static_cast<int>(e / 23), 1);
            for (std::size_t e = 0; e < B.size(); ++e) B[e] = value<T>(static_cast<int>(e % 19), static_cast<int>(e / 19), 2);
            for (std::size_t e = 0; e < C0.size(); ++e) C0[e] = value<T>(static_cast<int>(e % 29), static_cast<int>(e / 29), 3);

            auto op_a = [&](const T* a, int i, int l) {
                return op.ta == 'N' ? a[i + l * lda] : (op.ta == 'T' ? a[l + i * lda] : conj_v(a[l + i * lda]));
            };
            auto op_b = [&](const T* b, int l, int j) {
                return op.tb == 'N' ? b[l + j * ldb] : (op.tb == 'T' ? b[j + l * ldb] : conj_v(b[j + l * ldb]));
            };
            std::vector<T> ref = C0;
            for (int p = 0; p < batch; ++p)
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i) {
                        T sum = T(0);
                        for (int l = 0; l < k; ++l) sum += op_a(&A[p * asz], i, l) * op_b(&B[p * bsz], l, j);
                        ref[p * csz + i + j * ldc] = alpha * sum + beta * C0[p * csz + i + j * ldc];
                    }

            std::vector<T> out[2] = {C0, C0};
            DdlaHandle_t hs[2] = {h1, h4};
            for (int w = 0; w < 2; ++w) {
                std::vector<const T*> pa(batch), pb(batch);
                std::vector<T*> pc(batch);
                for (int p = 0; p < batch; ++p) {
                    pa[p] = &A[p * asz];
                    pb[p] = &B[p * bsz];
                    pc[p] = &out[w][p * csz];
                }
                gemmBatched<DdlaBackend::CPU, T>(hs[w], op.ta, op.tb, m, n, k, alpha, pa.data(), lda,
                                                 pb.data(), ldb, beta, pc.data(), ldc, batch);
                if (!close_enough<T>(out[w], ref)) fail(w == 0 ? "1-thread result vs reference" : "4-thread result vs reference", op.ta, op.tb, batch);
            }
            if (out[0].size() != out[1].size() ||
                (!out[0].empty() && std::memcmp(out[0].data(), out[1].data(), out[0].size() * sizeof(T)) != 0))
                fail("1-thread and 4-thread results are not bitwise equal", op.ta, op.tb, batch);
        }
    }

    // The BLAS thread count survives a threaded call, and the guard pins it only inside its scope.  Done last: it changes the BLAS
    // setting (restored below), which the bitwise comparison above must not depend on.
    const int original = detail::ScopedBlasThreads::current();
    if (original > 0) {
        if (openblas_set_num_threads != nullptr) openblas_set_num_threads(3);
        else if (mkl_set_num_threads != nullptr) mkl_set_num_threads(3);
        const int before = detail::ScopedBlasThreads::current();
        const int batch = 16;
        std::vector<T> a(m * k, T(1)), b(k * n, T(1)), c(batch * m * n, T(0));
        std::vector<const T*> pa(batch, a.data()), pb(batch, b.data());
        std::vector<T*> pc(batch);
        for (int p = 0; p < batch; ++p) pc[p] = &c[p * m * n];
        gemmBatched<DdlaBackend::CPU, T>(h4, 'N', 'N', m, n, k, alpha, pa.data(), m, pb.data(), k, beta, pc.data(), m, batch);
        if (detail::ScopedBlasThreads::current() != before) fail("BLAS thread count not restored after gemmBatched", '-', '-', batch);
        {
            detail::ScopedBlasThreads pinned(1);
            if (detail::ScopedBlasThreads::current() != 1) fail("ScopedBlasThreads did not pin the BLAS to one thread", '-', '-', 0);
        }
        if (detail::ScopedBlasThreads::current() != before) fail("BLAS thread count not restored by ScopedBlasThreads", '-', '-', 0);
        if (openblas_set_num_threads != nullptr) openblas_set_num_threads(original);
        else if (mkl_set_num_threads != nullptr) mkl_set_num_threads(original);
    }
    ddlaDestroy(h1);
    ddlaDestroy(h4);
    return bad;
}
#endif // DDLA_HAS_CPU

template <DdlaBackend Backend, typename T>
int run_type(const DdlaHandle_t& handle, char* arena, std::size_t arena_bytes)
{
    int failures = 0;
    int f_trsm = 0, f_trmm = 0, f_syrk = 0, f_gemmB = 0;
    // This type's own slice: types must not overlap, or one type's uploads
    // clobber the previous type's queued work.
    if (kArenaElems * sizeof(T) > arena_bytes) {
        return 1;
    }
    Scratch<T> sc;
    sc.carve(arena);
    // device-resident pointer arrays for gemmBatched (see the note in
    // check_gemmBatched)
    T** d_aptr = reinterpret_cast<T**>(sc.ptr_a);
    T** d_bptr = reinterpret_cast<T**>(sc.ptr_b);
    T** d_cptr = reinterpret_cast<T**>(sc.ptr_c);
    for (char side : {'L', 'R'})
        for (char uplo : {'U', 'L'})
            for (char trans : {'N', 'T', 'C'})
                for (char diag : {'U', 'N'})
                    f_trsm += check_trsm<Backend, T>(handle, side, uplo, trans, diag, sc);
    for (char side : {'L', 'R'})
        for (char uplo : {'U', 'L'})
            for (char trans : {'N', 'T', 'C'})
                for (char diag : {'U', 'N'})
                    for (bool zero : {false, true})
                        f_trmm += check_trmm<Backend, T>(handle, side, uplo, trans, diag, sc, zero);
    f_trmm += check_trmm_edges<Backend,T>(handle);
    // syrk is real-only by design (complex callers use herk); the public syrk
    // only instantiates float/double, so requesting it with a complex T would
    // be a link error rather than a test failure.
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        for (char uplo : {'U', 'L'})
            for (char trans : {'N', 'T'})
                f_syrk += check_syrk<Backend, T>(handle, uplo, trans, sc);
    }
    f_gemmB += check_gemmBatched_standalone<Backend, T>(handle);
    failures = f_trsm + f_trmm + f_syrk + f_gemmB;
    return failures;
}

template <DdlaBackend Backend>
int run_backend()
{
    DdlaHandle_t handle = nullptr;
    ddlaInit(handle, Backend);
    ddlaSet(handle, MPI_COMM_WORLD);
    int failures = 0;
    // ONE allocation for all four scalar types: sized for the widest scalar
    // (complex<double>) so any typed view fits.  See the note on Scratch.
    // One allocation, split into four DISJOINT per-type slices.
    const std::size_t slice = kArenaElems * sizeof(std::complex<double>);
    const std::size_t arena_bytes = 4 * slice;
    void* arena = nullptr;
    if (ddlaMalloc(handle, &arena, arena_bytes) != ddlaStatus_t::DDLA_STATUS_SUCCESS) {
        ddlaDestroy(handle);
        return 1;
    }
    char* base = static_cast<char*>(arena);
    failures += run_type<Backend, float>(handle, base + 0 * slice, slice);
    failures += run_type<Backend, double>(handle, base + 1 * slice, slice);
    failures += run_type<Backend, std::complex<float>>(handle, base + 2 * slice, slice);
    failures += run_type<Backend, std::complex<double>>(handle, base + 3 * slice, slice);
    // Report the check result BEFORE teardown: an error surfaced by
    // gpuStreamDestroy (ddla_stream.h) is a teardown condition, and
    // conflating it with a failing check would misattribute the cause.
    ddlaSynchronize(handle);
    {
        int local_sum = 0;
        MPI_Allreduce(&failures, &local_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
            std::cerr << "  CHECKS_FAILED=" << local_sum
                      << (local_sum == 0 ? "  (all numerical checks passed)"
                                         : "  *** some checks FAILED ***")
                      << std::endl;
    }
    ddlaFree(handle, arena);
    ddlaSynchronize(handle);
    ddlaDestroy(handle);
    return failures;
}

#if DDLA_HAS_CPU && DDLA_HAS_GPU
// A backend mismatch and a null handle must both be rejected, per family.
int check_rejections()
{
    DdlaHandle_t cpu = nullptr;
    DdlaHandle_t gpu = nullptr;
    ddlaInit(cpu, DdlaBackend::CPU);
    ddlaInit(gpu, DdlaBackend::GPU);
    ddlaSet(cpu, MPI_COMM_WORLD);
    ddlaSet(gpu, MPI_COMM_WORLD);

    double x = 1.0;
    int failures = 0;
    auto expect_throw = [&](auto&& fn, const char* what) {
        bool threw = false;
        try { fn(); }
        catch (const std::runtime_error& e) {
            threw = std::string(e.what()).find(what) != std::string::npos;
        }
        if (!threw) ++failures;
    };
    expect_throw([&] { trsm<DdlaBackend::CPU>(gpu, 'L', 'U', 'N', 'N', 1, 1, x,
                                              &x, 1, &x, 1); }, "does not match");
    expect_throw([&] { trmm<DdlaBackend::CPU>(gpu, 'L', 'U', 'N', 'N', 1, 1, x,
                                              &x, 1, &x, 1); }, "does not match");
    expect_throw([&] { syrk<DdlaBackend::CPU>(gpu, 'U', 'N', 1, 1, x, &x, 1, x, &x, 1); },
                 "does not match");
    {
        const double* arr[1] = {&x};
        expect_throw([&] { gemmBatched<DdlaBackend::CPU>(gpu, 'N', 'N', 1, 1, 1, x,
                                                         arr, 1, arr, 1, x,
                                                         const_cast<double**>(arr), 1, 1); },
                     "does not match");
    }
    expect_throw([&] { trsm<DdlaBackend::CPU>(nullptr, 'L', 'U', 'N', 'N', 1, 1, x,
                                              &x, 1, &x, 1); }, "null handle");
    ddlaDestroy(cpu);
    ddlaDestroy(gpu);
    return failures;
}
#endif

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // DDLA_TEST_ONLY=cpu|gpu runs one backend alone, so a failure can be
    // attributed without the other backend's state in the process.
    const char* only = std::getenv("DDLA_TEST_ONLY");
    const bool do_cpu = (!only || std::string(only) == "cpu");
    const bool do_gpu = (!only || std::string(only) == "gpu");
    int failures = 0;
#if DDLA_HAS_CPU
    if (do_cpu) {
        failures += run_backend<DdlaBackend::CPU>();
        if (family_selected("gemmBatched")) {
            failures += check_gemmBatched_threads<float>();
            failures += check_gemmBatched_threads<double>();
            failures += check_gemmBatched_threads<std::complex<float>>();
            failures += check_gemmBatched_threads<std::complex<double>>();
        }
    }
#else
    (void)do_cpu;
#endif
#if DDLA_HAS_GPU
    if (do_gpu) failures += run_backend<DdlaBackend::GPU>();
#else
    (void)do_gpu;
#endif
#if DDLA_HAS_CPU && DDLA_HAS_GPU
    failures += check_rejections();
#endif

    int global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        if (global_failures == 0) {
            std::cout << "test_backend_blas3 passed" << std::endl;
        } else {
            std::cout << "test_backend_blas3 FAILED: " << global_failures
                      << " case(s)" << std::endl;
        }
    }
    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}
