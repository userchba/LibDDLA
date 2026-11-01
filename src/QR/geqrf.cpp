// ---------------------------------------------------------------------------
// Backend-templated QR / QL factorization of one matrix on one process (LAPACK ?geqrf / ?geqlf):
// A = Q * R for uplo = 'U', A = Q * L for uplo = 'L'.
//
// Structure follows src/getrf.cpp: one definition covering CPU (Fortran LAPACK ?geqrf / ?geqlf on host pointers) and
// GPU (cuSOLVER / hipSOLVER geqrf on device pointers, enqueued on the handle's stream), selected with `if constexpr`
// on the Backend template parameter, the arms guarded by the BUILD CAPABILITY macros (DDLA_HAS_CPU / DDLA_HAS_GPU).
// The vendor calls are private to this translation unit; the public interface is the templated ddla::geqrf declared in
// <ddla/ddla.h>.
//
// Three components (plan_util.h): geqrf_check; geqrf_workspace_size (backend bytes: the LAPACK work array on a CPU
// handle; on a GPU handle a device info slot, the QL path's reversed copy and the vendor workspace); geqrf_run on the
// caller's scratch, with the vendor lwork taken from the bytes it is given.
// cuSOLVER / hipSOLVER have no ?geqlf, so uplo = 'L' is run as a QR of the doubly reversed matrix
// A' = J A J (J = row / column reversal): A = Q L  <=>  A' = (J Q J) (J L J), J L J upper triangular, and the
// reflector i of the QR is the reflector k - 1 - i of the QL (k = min(m, n)), which is the order the scalars tau come
// back in.  The reversal and its inverse are index-only qr_reverse copies, so the factors are stored exactly as LAPACK ?geqlf stores them.
// The reversed copy A' and tau' are carved from the caller's scratch.
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <complex>
#include <cstddef>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <string>

#include "ddla_connector.h"
#include "ddla_stream.h"
#include "lapack_impl.h"
#include "pgeqrf_kernels.h"
#include "plan_util.h"

#if DDLA_HAS_CPU
// Host LAPACK ?geqrf / ?geqlf (Fortran ABI; the CPU BLAS library must provide LAPACK -- OpenBLAS and MKL do).
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
#endif

namespace ddla {

namespace {

inline const char* geqrf_backend_name(DdlaBackend backend)
{
    return backend == DdlaBackend::CPU ? "CPU" : "GPU";
}

#if DDLA_HAS_CPU
inline void lapack_geqrf(bool lower, int m, int n, float* A, int lda, float* tau, float* work, int lwork, int* info)
{
    if (lower) sgeqlf_(&m, &n, A, &lda, tau, work, &lwork, info);
    else sgeqrf_(&m, &n, A, &lda, tau, work, &lwork, info);
}
inline void lapack_geqrf(bool lower, int m, int n, double* A, int lda, double* tau, double* work, int lwork, int* info)
{
    if (lower) dgeqlf_(&m, &n, A, &lda, tau, work, &lwork, info);
    else dgeqrf_(&m, &n, A, &lda, tau, work, &lwork, info);
}
inline void lapack_geqrf(bool lower, int m, int n, std::complex<float>* A, int lda, std::complex<float>* tau,
                         std::complex<float>* work, int lwork, int* info)
{
    if (lower) cgeqlf_(&m, &n, A, &lda, tau, work, &lwork, info);
    else cgeqrf_(&m, &n, A, &lda, tau, work, &lwork, info);
}
inline void lapack_geqrf(bool lower, int m, int n, std::complex<double>* A, int lda, std::complex<double>* tau,
                         std::complex<double>* work, int lwork, int* info)
{
    if (lower) zgeqlf_(&m, &n, A, &lda, tau, work, &lwork, info);
    else zgeqrf_(&m, &n, A, &lda, tau, work, &lwork, info);
}

// The LAPACK workspace query (LAPACK returns the optimal size in work[0], as a real number); a pure host call.
template <typename T>
int cpu_geqrf_lwork(bool lower, int m, int n, T* A, int lda, T* tau)
{
    int info = 0;
    T opt{};
    lapack_geqrf(lower, m, n, A, lda, tau, &opt, -1, &info);
    if (info != 0) throw std::runtime_error("geqrf: LAPACK workspace query failed");
    const auto optimal = std::real(opt);
    if (!std::isfinite(optimal) || optimal < 0 || static_cast<long double>(optimal) > std::numeric_limits<int>::max())
        throw std::length_error("geqrf: LAPACK workspace length overflow");
    return std::max(1, static_cast<int>(optimal));
}

// Factor with the work array the caller's scratch holds.
template <typename T>
void cpu_geqrf(void* work, std::size_t work_bytes, bool lower, int m, int n, T* A, int lda, T* tau)
{
    const int lwork = static_cast<int>(std::min<std::size_t>(work_bytes / sizeof(T), std::numeric_limits<int>::max()));
    int info = 0;
    lapack_geqrf(lower, m, n, A, lda, tau, static_cast<T*>(work), lwork, &info);
    if (info != 0) throw std::runtime_error("geqrf: LAPACK reported an invalid argument");
}
#endif // DDLA_HAS_CPU

#if DDLA_HAS_GPU
#if DDLA_GPU_VENDOR

// Per scalar type: the buffer-size query and the factorization itself.
inline gpusolverStatus_t geqrf_buffer_size(gpusolverHandle_t h, int m, int n, float* A, int lda, int* lwork)
{
    return gpusolverSgeqrf_bufferSize(h, m, n, A, lda, lwork);
}
inline gpusolverStatus_t geqrf_buffer_size(gpusolverHandle_t h, int m, int n, double* A, int lda, int* lwork)
{
    return gpusolverDgeqrf_bufferSize(h, m, n, A, lda, lwork);
}
inline gpusolverStatus_t geqrf_buffer_size(gpusolverHandle_t h, int m, int n, std::complex<float>* A, int lda,
                                           int* lwork)
{
    return gpusolverCgeqrf_bufferSize(h, m, n, (gpusolverComplex*)A, lda, lwork);
}
inline gpusolverStatus_t geqrf_buffer_size(gpusolverHandle_t h, int m, int n, std::complex<double>* A, int lda,
                                           int* lwork)
{
    return gpusolverZgeqrf_bufferSize(h, m, n, (gpusolverDoubleComplex*)A, lda, lwork);
}

inline gpusolverStatus_t geqrf_call(gpusolverHandle_t h, int m, int n, float* A, int lda, float* tau, float* work,
                                    int lwork, int* info)
{
    return gpusolverSgeqrf(h, m, n, A, lda, tau, work, lwork, info);
}
inline gpusolverStatus_t geqrf_call(gpusolverHandle_t h, int m, int n, double* A, int lda, double* tau, double* work,
                                    int lwork, int* info)
{
    return gpusolverDgeqrf(h, m, n, A, lda, tau, work, lwork, info);
}
inline gpusolverStatus_t geqrf_call(gpusolverHandle_t h, int m, int n, std::complex<float>* A, int lda,
                                    std::complex<float>* tau, std::complex<float>* work, int lwork, int* info)
{
    return gpusolverCgeqrf(h, m, n, (gpusolverComplex*)A, lda, (gpusolverComplex*)tau, (gpusolverComplex*)work, lwork,
                           info);
}
inline gpusolverStatus_t geqrf_call(gpusolverHandle_t h, int m, int n, std::complex<double>* A, int lda,
                                    std::complex<double>* tau, std::complex<double>* work, int lwork, int* info)
{
    return gpusolverZgeqrf(h, m, n, (gpusolverDoubleComplex*)A, lda, (gpusolverDoubleComplex*)tau,
                           (gpusolverDoubleComplex*)work, lwork, info);
}

// A = Q R on the device.  The scratch is [ device info (padded to 256 bytes) | vendor workspace (the rest) ].  tau
// is zeroed first: cuSOLVER's single-precision panel kernel (geqr2_smem_domino_fast) reads tau before it writes it
// (compute-sanitizer initcheck reports it; the result does not depend on the value -- a 0xFF-poisoned tau gives
// bit-identical output), and tau is an output the caller may hand over uninitialized.
template <typename T>
void gpu_geqrf_qr(const DdlaHandle_t& handle, char* w, std::size_t bytes, int m, int n, T* A, int lda, T* tau)
{
    char* const base = w;
    int* const d_info = detail::take<int>(w, 1);
    T* const work = reinterpret_cast<T*>(w);
    assert(static_cast<std::size_t>(w - base) <= bytes);
    const std::size_t rest = bytes - static_cast<std::size_t>(w - base);
    const int lwork = static_cast<int>(std::min<std::size_t>(rest / sizeof(T), std::numeric_limits<int>::max()));

    GPU_CHECK(gpuMemsetAsync(tau, 0, static_cast<std::size_t>(std::min(m, n)) * sizeof(T), static_cast<gpuStream_t>(handle->stream)));
    SOLVER_CHECK(geqrf_call(static_cast<gpusolverHandle_t>(handle->solverH), m, n, A, lda, tau, work, lwork, d_info));
}

// A = Q L on the device through the QR of A' = J A J (see the header comment).  The scratch is
// [ A' (m x n, ld m) | tau' (k) | the QR's scratch ].
template <typename T>
void gpu_geqrf_ql(const DdlaHandle_t& handle, char* w, std::size_t bytes, int m, int n, T* A, int lda, T* tau)
{
    const int k = std::min(m, n);
    char* const base = w;
    T* const Ar = detail::take<T>(w, static_cast<std::size_t>(m) * n);
    T* const taur = detail::take<T>(w, static_cast<std::size_t>(k));
    assert(static_cast<std::size_t>(w - base) <= bytes);

    detail::qr_reverse<T>(handle, m, n, A, lda, Ar, m);
    gpu_geqrf_qr<T>(handle, w, bytes - static_cast<std::size_t>(w - base), m, n, Ar, m, taur);
    detail::qr_reverse<T>(handle, m, n, Ar, m, A, lda);
    // tau_L(k - 1 - i) = tau'(i): the same reversal on a k x 1 column
    detail::qr_reverse<T>(handle, k, 1, taur, k, tau, k);
}

#else // DDLA_GPU_VENDOR

template <typename T>
void gpu_geqrf_qr(const DdlaHandle_t&, char*, std::size_t, int, int, T*, int, T*)
{
    throw std::runtime_error("geqrf: no GPU vendor selected at compile time");
}
template <typename T>
void gpu_geqrf_ql(const DdlaHandle_t&, char*, std::size_t, int, int, T*, int, T*)
{
    throw std::runtime_error("geqrf: no GPU vendor selected at compile time");
}

#endif // DDLA_GPU_VENDOR
#endif // DDLA_HAS_GPU

} // anonymous namespace

namespace lapack_impl {

// Component 1.
template <DdlaBackend Backend>
char geqrf_check(const DdlaHandle_t& handle, char uplo, int m, int n, const void* A, int lda, const void* tau)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU geqrf is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU geqrf is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("geqrf: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("geqrf: template backend ") + geqrf_backend_name(Backend) +
            " does not match handle backend " + geqrf_backend_name(actual));
    }
    const char u = static_cast<char>(std::toupper(static_cast<unsigned char>(uplo)));
    if (u != 'U' && u != 'L') throw std::runtime_error("geqrf: uplo must be 'U' or 'L'");
    if (m < 0 || n < 0) throw std::runtime_error("geqrf: negative matrix size");

    const int k = std::min(m, n);
    if (k == 0) return u;   // nothing to factor: the pointers are not read
    if (A == nullptr) throw std::runtime_error("geqrf: null A");
    if (tau == nullptr) throw std::runtime_error("geqrf: null tau");
    if (lda < std::max(1, m)) throw std::runtime_error("geqrf: lda too small");
    return u;
}

// Component 2: the scratch of an m x n factorization (normalized uplo).  CPU handles: the ?geqrf / ?geqlf work array
// (LAPACK's workspace query, made with A, lda and tau).  GPU handles: [ device info | A' (m x n) | tau' (k), uplo 'L'
// only | vendor workspace ] -- the vendor query is made with A and the leading dimension the factorization uses (m for
// the reversed copy of the QL path).
template <typename T>
void geqrf_workspace_size(const DdlaHandle_t& handle, char uplo, int m, int n, T* A, int lda, T* tau,
                          std::size_t* backend_bytes)
{
    *backend_bytes = 0;
    if (std::min(m, n) <= 0) return;
    const bool lower = uplo == 'L';
    if (handle->backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        *backend_bytes = static_cast<std::size_t>(cpu_geqrf_lwork(lower, m, n, A, lda, tau)) * sizeof(T);
#endif
    } else {
#if DDLA_HAS_GPU && DDLA_GPU_VENDOR
        int lwork = 0;
        SOLVER_CHECK(geqrf_buffer_size(static_cast<gpusolverHandle_t>(handle->solverH), m, n, A, lower ? m : lda,
                                       &lwork));
        std::size_t bytes = detail::padded_bytes<int>(1);
        if (lower)
            bytes += detail::padded_bytes<T>(static_cast<std::size_t>(m) * n)
                   + detail::padded_bytes<T>(static_cast<std::size_t>(std::min(m, n)));
        *backend_bytes = bytes + static_cast<std::size_t>(lwork) * sizeof(T);
#endif
    }
    (void)lower; (void)A; (void)lda; (void)tau;
}

// Component 3.
template <DdlaBackend Backend, typename T>
void geqrf_run(const DdlaHandle_t& handle, char uplo, int m, int n, T* A, int lda, T* tau,
               [[maybe_unused]] void* backend_work, [[maybe_unused]] std::size_t backend_bytes)
{
    if (std::min(m, n) <= 0) return;   // nothing to factor

    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        cpu_geqrf(backend_work, backend_bytes, uplo == 'L', m, n, A, lda, tau);
#endif
    } else {
#if DDLA_HAS_GPU
        char* const w = static_cast<char*>(backend_work);
        if (uplo == 'L') gpu_geqrf_ql<T>(handle, w, backend_bytes, m, n, A, lda, tau);
        else gpu_geqrf_qr<T>(handle, w, backend_bytes, m, n, A, lda, tau);
#endif
    }
}

} // namespace lapack_impl

template <DdlaBackend Backend, typename T>
void geqrf(const DdlaHandle_t& handle, char uplo, int m, int n, T* A, int lda, T* tau)
{
    // 1. parameter check (also normalizes uplo).
    const char u = lapack_impl::geqrf_check<Backend>(handle, uplo, m, n, A, lda, tau);
    // No-op: nothing to factor.
    if (std::min(m, n) == 0) return;
    // 2. the work array / vendor scratch of this shape.
    std::size_t backend_bytes = 0;
    lapack_impl::geqrf_workspace_size<T>(handle, u, m, n, A, lda, tau, &backend_bytes);
    vector<char> work(handle, backend_bytes);
    // 3. the factorization.
    lapack_impl::geqrf_run<Backend, T>(handle, u, m, n, A, lda, tau, work.data(), backend_bytes);
}

#define INSTANTIATE_GEQRF(BACKEND, TYPE)                                     \
    template void geqrf<BACKEND, TYPE>(                                      \
        const DdlaHandle_t&, char, int, int, TYPE*, int, TYPE*);             \
    template void lapack_impl::geqrf_run<BACKEND, TYPE>(                     \
        const DdlaHandle_t&, char, int, int, TYPE*, int, TYPE*, void*, std::size_t)
#define INSTANTIATE_GEQRF_SIZE(TYPE)                                         \
    template void lapack_impl::geqrf_workspace_size<TYPE>(                   \
        const DdlaHandle_t&, char, int, int, TYPE*, int, TYPE*, std::size_t*)

#if DDLA_HAS_CPU
template char lapack_impl::geqrf_check<DdlaBackend::CPU>(
    const DdlaHandle_t&, char, int, int, const void*, int, const void*);
INSTANTIATE_GEQRF(DdlaBackend::CPU, float);
INSTANTIATE_GEQRF(DdlaBackend::CPU, double);
INSTANTIATE_GEQRF(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_GEQRF(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
template char lapack_impl::geqrf_check<DdlaBackend::GPU>(
    const DdlaHandle_t&, char, int, int, const void*, int, const void*);
INSTANTIATE_GEQRF(DdlaBackend::GPU, float);
INSTANTIATE_GEQRF(DdlaBackend::GPU, double);
INSTANTIATE_GEQRF(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_GEQRF(DdlaBackend::GPU, std::complex<double>);
#endif

INSTANTIATE_GEQRF_SIZE(float);
INSTANTIATE_GEQRF_SIZE(double);
INSTANTIATE_GEQRF_SIZE(std::complex<float>);
INSTANTIATE_GEQRF_SIZE(std::complex<double>);

#undef INSTANTIATE_GEQRF_SIZE
#undef INSTANTIATE_GEQRF

} // namespace ddla
