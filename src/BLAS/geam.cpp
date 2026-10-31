#include "blas_impl.h"
// ---------------------------------------------------------------------------
// Unified backend-templated matrix add / transpose-copy:
//
//     C(m x n) := alpha * blas_impl::op(A) + beta * blas_impl::op(B),     op in {'N', 'T', 'C'}.
//
// Structure follows src/trmm.cpp and src/scal.cpp: one definition covering CPU
// and GPU, selected with `if constexpr` on the Backend template parameter, the
// arms guarded by the BUILD CAPABILITY macros (DDLA_HAS_CPU / DDLA_HAS_GPU),
// never by the per-TU vendor macro (a dual build defines DDLA_USE_CUDA in every
// TU, so a vendor-macro `#else` CPU arm would never compile there).
//
// The out-of-place scaled transpose-copy that used to be `omatcopy` is the
// special case beta == 0: `geam(h, 'T', 'N', m, n, 1, A, lda, 0, nullptr, 1, C, ldc)`.
//
// GPU arm: cuBLAS / hipBLAS geam.  With beta == 0 the call aliases B to A (the
// vendor routine needs a second operand), exactly what the old omatcopy did.
// CPU arm: tiles of C (32 x 32 when an operand is read transposed, strips of whole columns otherwise), run on the
// handle's host threads with OpenMP.
// Every element of C is computed on its own, so the result does not depend on
// the thread count; pure copies (alpha == 1, beta == 0) move the bits of A
// unchanged -- no multiply, so -0.0, Inf and NaN survive (cblas_?omatcopy,
// the old CPU arm, did not guarantee that).
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "blas_enums.h"
#include "ddla_connector.h"
#include "ddla_stream.h"

// ---------------------------------------------------------------------------
// GPU arm: the vendor calls.  Kept as free gpublasGeam overloads so the
// dispatch below is one BLAS_CHECK call per scalar type.  Private to this
// translation unit -- the public interface is the templated ddla::geam
// declared in <ddla/ddla.h>.
// ---------------------------------------------------------------------------
#if DDLA_HAS_GPU

namespace ddla {
namespace {

inline gpublasStatus_t gpublasGeam(
    gpublasHandle_t handle, gpublasOperation_t transA, gpublasOperation_t transB,
    int m, int n,
    const float& alpha, const float* A, int lda,
    const float& beta, const float* B, int ldb,
    float* C, int ldc)
{
#if DDLA_GPU_VENDOR
    return gpublasSgeam(handle, transA, transB, m, n, &alpha, A, lda, &beta, B, ldb, C, ldc);
#else
    throw std::runtime_error("geam: no GPU vendor selected at compile time");
#endif
}

inline gpublasStatus_t gpublasGeam(
    gpublasHandle_t handle, gpublasOperation_t transA, gpublasOperation_t transB,
    int m, int n,
    const double& alpha, const double* A, int lda,
    const double& beta, const double* B, int ldb,
    double* C, int ldc)
{
#if DDLA_GPU_VENDOR
    return gpublasDgeam(handle, transA, transB, m, n, &alpha, A, lda, &beta, B, ldb, C, ldc);
#else
    throw std::runtime_error("geam: no GPU vendor selected at compile time");
#endif
}

inline gpublasStatus_t gpublasGeam(
    gpublasHandle_t handle, gpublasOperation_t transA, gpublasOperation_t transB,
    int m, int n,
    const std::complex<float>& alpha, const std::complex<float>* A, int lda,
    const std::complex<float>& beta, const std::complex<float>* B, int ldb,
    std::complex<float>* C, int ldc)
{
#if DDLA_GPU_VENDOR
    return gpublasCgeam(handle, transA, transB, m, n,
                        reinterpret_cast<const gpublasComplex*>(&alpha),
                        reinterpret_cast<const gpublasComplex*>(A), lda,
                        reinterpret_cast<const gpublasComplex*>(&beta),
                        reinterpret_cast<const gpublasComplex*>(B), ldb,
                        reinterpret_cast<gpublasComplex*>(C), ldc);
#else
    throw std::runtime_error("geam: no GPU vendor selected at compile time");
#endif
}

inline gpublasStatus_t gpublasGeam(
    gpublasHandle_t handle, gpublasOperation_t transA, gpublasOperation_t transB,
    int m, int n,
    const std::complex<double>& alpha, const std::complex<double>* A, int lda,
    const std::complex<double>& beta, const std::complex<double>* B, int ldb,
    std::complex<double>* C, int ldc)
{
#if DDLA_GPU_VENDOR
    return gpublasZgeam(handle, transA, transB, m, n,
                        reinterpret_cast<const gpublasDoubleComplex*>(&alpha),
                        reinterpret_cast<const gpublasDoubleComplex*>(A), lda,
                        reinterpret_cast<const gpublasDoubleComplex*>(&beta),
                        reinterpret_cast<const gpublasDoubleComplex*>(B), ldb,
                        reinterpret_cast<gpublasDoubleComplex*>(C), ldc);
#else
    throw std::runtime_error("geam: no GPU vendor selected at compile time");
#endif
}

} // anonymous namespace
} // namespace ddla

#endif // DDLA_HAS_GPU

namespace ddla {

// ---------------------------------------------------------------------------
// CPU arm.
// ---------------------------------------------------------------------------
#if DDLA_HAS_CPU
namespace {

constexpr int kTile = 32;                        // transposed access: kTile x kTile tiles
constexpr int kStripElems = 1 << 14;             // plain access: strips of about this many elements
constexpr long long kParallelMinElems = 1 << 16; // below this, a parallel region costs more than it saves

// Conjugation is a no-op on real data, so 'C' behaves exactly like 'T'.
inline float  maybe_conj(float x, bool)   { return x; }
inline double maybe_conj(double x, bool)  { return x; }
template <typename R>
inline std::complex<R> maybe_conj(const std::complex<R>& x, bool c)
{
    return c ? std::conj(x) : x;
}

template <typename T>
struct GeamCpuArgs {
    bool ta, ca, tb, cb;     // operand transposed / conjugated
    bool has_a, has_b;       // alpha != 0, beta != 0: the operand is referenced
    bool a_one;              // alpha == 1: no multiply, bits of A unchanged
    T alpha, beta;
    const T* A; std::size_t lda;
    const T* B; std::size_t ldb;
    T* C; std::size_t ldc;
};

// Single operand (beta == 0): C(i,j) := alpha * blas_impl::op(A)(i,j), the copy / transpose-copy path.
template <typename T, bool TA, bool CA, bool ONE>
void copy_tile(const GeamCpuArgs<T>& g, int i0, int i1, int j0, int j1)
{
    const T* const A = g.A;
    T* const C = g.C;
    const std::size_t lda = g.lda, ldc = g.ldc;
    const T alpha = g.alpha;
    for (int j = j0; j < j1; ++j) {
        T* const c = C + static_cast<std::size_t>(j) * ldc;
        for (int i = i0; i < i1; ++i) {
            T a;
            if (TA) a = maybe_conj(A[static_cast<std::size_t>(j) + static_cast<std::size_t>(i) * lda], CA);
            else    a = A[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * lda];
            c[i] = ONE ? a : alpha * a;
        }
    }
}

template <typename T>
inline T load_op(const T* X, std::size_t ldx, bool trans, bool conj, int i, int j)
{
    return trans ? maybe_conj(X[static_cast<std::size_t>(j) + static_cast<std::size_t>(i) * ldx], conj)
                 : X[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * ldx];
}

// General case (two operands, or alpha == 0).
template <typename T>
void general_tile(const GeamCpuArgs<T>& g, int i0, int i1, int j0, int j1)
{
    const bool ta = g.ta, ca = g.ca, tb = g.tb, cb = g.cb, has_a = g.has_a, has_b = g.has_b;
    const T alpha = g.alpha, beta = g.beta;
    const T* const A = g.A;
    const T* const B = g.B;
    T* const C = g.C;
    const std::size_t lda = g.lda, ldb = g.ldb, ldc = g.ldc;
    for (int j = j0; j < j1; ++j) {
        T* const c = C + static_cast<std::size_t>(j) * ldc;
        for (int i = i0; i < i1; ++i) {
            if (has_a && has_b)
                c[i] = alpha * load_op(A, lda, ta, ca, i, j) + beta * load_op(B, ldb, tb, cb, i, j);
            else if (has_a)
                c[i] = alpha * load_op(A, lda, ta, ca, i, j);
            else if (has_b)
                c[i] = beta * load_op(B, ldb, tb, cb, i, j);
            else
                c[i] = T(0);
        }
    }
}

template <typename T>
void run_copy_tile(const GeamCpuArgs<T>& g, int i0, int i1, int j0, int j1)
{
    if (!g.ta) {
        if (g.a_one) copy_tile<T, false, false, true>(g, i0, i1, j0, j1);
        else         copy_tile<T, false, false, false>(g, i0, i1, j0, j1);
    } else if (!g.ca) {
        if (g.a_one) copy_tile<T, true, false, true>(g, i0, i1, j0, j1);
        else         copy_tile<T, true, false, false>(g, i0, i1, j0, j1);
    } else {
        if (g.a_one) copy_tile<T, true, true, true>(g, i0, i1, j0, j1);
        else         copy_tile<T, true, true, false>(g, i0, i1, j0, j1);
    }
}

template <typename T>
void cpu_geam(const GeamCpuArgs<T>& g, int m, int n, int nthreads)
{
    const bool single = g.has_a && !g.has_b;
    // An operand read transposed wants square-ish tiles (reads and writes both stay within a few cache lines per
    // tile row).  With no transposition the tile is a strip of whole columns -- contiguous in memory, so the
    // loops stream and vectorize -- of about kStripElems elements (taller than that: a piece of one column).
    const bool strided = g.ta || (g.has_b && g.tb);
    const int tm = strided ? kTile : std::min(m, kStripElems);
    const int tn = strided ? kTile : std::max(1, kStripElems / tm);
    const int tiles_i = (m + tm - 1) / tm;
    const int tiles_j = (n + tn - 1) / tn;
    const long long ntiles = static_cast<long long>(tiles_i) * tiles_j;
    auto tile = [&](long long t) {
        const int bi = static_cast<int>(t % tiles_i);
        const int bj = static_cast<int>(t / tiles_i);
        const int i0 = bi * tm, i1 = std::min(m, i0 + tm);
        const int j0 = bj * tn, j1 = std::min(n, j0 + tn);
        if (single) run_copy_tile(g, i0, i1, j0, j1);
        else        general_tile(g, i0, i1, j0, j1);
    };
    const int nt = (static_cast<long long>(m) * n < kParallelMinElems)
                       ? 1
                       : static_cast<int>(std::min<long long>(std::max(1, nthreads), ntiles));
    if (nt <= 1) {                       // no parallel region at all: a tiny call must stay cheap
        for (long long t = 0; t < ntiles; ++t) tile(t);
        return;
    }
    const int chunk = std::max(1, kStripElems / (tm * tn));   // tiles per scheduling step
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, chunk) num_threads(nt)
#endif
    for (long long t = 0; t < ntiles; ++t) tile(t);
}

} // anonymous namespace
#endif // DDLA_HAS_CPU

namespace {

inline const char* geam_backend_name(DdlaBackend backend)
{
    return backend == DdlaBackend::CPU ? "CPU" : "GPU";
}

// 'N'/'T'/'C' in either case; anything else is a caller bug, not a silent default.
inline char geam_op(char trans)
{
    switch (trans) {
        case 'N': case 'n': return 'N';
        case 'T': case 't': return 'T';
        case 'C': case 'c': return 'C';
        default: break;
    }
    throw std::runtime_error("geam: invalid trans flag (expected 'N', 'T' or 'C')");
}

} // anonymous namespace

namespace blas_impl {

template <DdlaBackend Backend, typename T>
void geam_check(const DdlaHandle_t& handle, char transa, char transb, int m, int n,
          const T& alpha, const T* A, int lda,
          const T& beta, const T* B, int ldb,
          T* C, int ldc)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU geam is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU geam is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("geam: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("geam: template backend ") + geam_backend_name(Backend) +
            " does not match handle backend " + geam_backend_name(actual));
    }
    const char ta = geam_op(transa);
    const char tb = geam_op(transb);
    if (m <= 0 || n <= 0) return;

    // Leading dimensions: blas_impl::op(A) is m x n, so A itself has m rows for 'N' and n rows otherwise.
    const bool zero_beta = (beta == T(0));
    if (A == nullptr) throw std::runtime_error("geam: null A");
    if (!zero_beta && B == nullptr) throw std::runtime_error("geam: null B with beta != 0");
    if (C == nullptr) throw std::runtime_error("geam: null C");
    if (lda < std::max(1, ta == 'N' ? m : n)) throw std::runtime_error("geam: lda too small");
    if (!zero_beta && ldb < std::max(1, tb == 'N' ? m : n)) throw std::runtime_error("geam: ldb too small");
    if (ldc < std::max(1, m)) throw std::runtime_error("geam: ldc too small");

}

template <DdlaBackend Backend, typename T>
void geam_run(const DdlaHandle_t& handle, char transa, char transb, int m, int n,
          const T& alpha, const T* A, int lda,
          const T& beta, const T* B, int ldb,
          T* C, int ldc)
{
    const char ta = upper(transa), tb = upper(transb);
    if (m <= 0 || n <= 0) return;
    const bool zero_beta = (beta == T(0));
    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        GeamCpuArgs<T> g;
        g.ta = (ta != 'N');
        g.ca = (ta == 'C');
        g.tb = (tb != 'N');
        g.cb = (tb == 'C');
        g.has_a = !(alpha == T(0));
        g.has_b = !zero_beta;
        g.a_one = (alpha == T(1));
        g.alpha = alpha;
        g.beta = beta;
        g.A = A; g.lda = static_cast<std::size_t>(lda);
        g.B = B; g.ldb = static_cast<std::size_t>(ldb);
        g.C = C; g.ldc = static_cast<std::size_t>(ldc);
        cpu_geam(g, m, n, handle->host_threads);
#endif
    } else {
#if DDLA_HAS_GPU
        const gpublasOperation_t opA = blas_impl::op(ta);
        if (zero_beta) {
            // The vendor routine always reads a second operand: alias it to A (beta == 0 makes its
            // contribution vanish), as the old omatcopy did.
            BLAS_CHECK(gpublasGeam(static_cast<gpublasHandle_t>(handle->blasH), opA, opA, m, n, alpha, A, lda, beta, A, lda, C, ldc));
        } else {
            BLAS_CHECK(gpublasGeam(static_cast<gpublasHandle_t>(handle->blasH), opA, blas_impl::op(tb), m, n,
                                   alpha, A, lda, beta, B, ldb, C, ldc));
        }
#endif
    }
}

} // namespace blas_impl

template <DdlaBackend Backend, typename T>
void geam(const DdlaHandle_t& handle, char transa, char transb, int m, int n,
          const T& alpha, const T* A, int lda,
          const T& beta, const T* B, int ldb,
          T* C, int ldc)
{
    blas_impl::geam_check<Backend, T>(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    blas_impl::geam_run<Backend, T>(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

#define INSTANTIATE_GEAM(BACKEND, TYPE)                                        \
    template void geam<BACKEND, TYPE>(                                         \
        const DdlaHandle_t&, char, char, int, int,                             \
        const TYPE&, const TYPE*, int, const TYPE&, const TYPE*, int, TYPE*, int)

#if DDLA_HAS_CPU
INSTANTIATE_GEAM(DdlaBackend::CPU, float);
INSTANTIATE_GEAM(DdlaBackend::CPU, double);
INSTANTIATE_GEAM(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_GEAM(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_GEAM(DdlaBackend::GPU, float);
INSTANTIATE_GEAM(DdlaBackend::GPU, double);
INSTANTIATE_GEAM(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_GEAM(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_GEAM


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::geam_check<BACKEND, TYPE>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, const TYPE& alpha, const TYPE* A, int lda, const TYPE& beta, const TYPE* B, int ldb, TYPE* C, int ldc); \
    template void blas_impl::geam_run<BACKEND, TYPE>(const DdlaHandle_t& handle, char transa, char transb, int m, int n, const TYPE& alpha, const TYPE* A, int lda, const TYPE& beta, const TYPE* B, int ldb, TYPE* C, int ldc);
#if DDLA_HAS_CPU
DDLA_INSTANTIATE_PHASES(DdlaBackend::CPU, float)
DDLA_INSTANTIATE_PHASES(DdlaBackend::CPU, double)
DDLA_INSTANTIATE_PHASES(DdlaBackend::CPU, std::complex<float>)
DDLA_INSTANTIATE_PHASES(DdlaBackend::CPU, std::complex<double>)
#endif
#if DDLA_HAS_GPU
DDLA_INSTANTIATE_PHASES(DdlaBackend::GPU, float)
DDLA_INSTANTIATE_PHASES(DdlaBackend::GPU, double)
DDLA_INSTANTIATE_PHASES(DdlaBackend::GPU, std::complex<float>)
DDLA_INSTANTIATE_PHASES(DdlaBackend::GPU, std::complex<double>)
#endif
#undef DDLA_INSTANTIATE_PHASES
} // namespace ddla
