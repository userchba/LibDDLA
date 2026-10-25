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
// GPU arm: the vendor routine needs a scratch buffer and a device `info`.  Both are sized with the ?geqrf_bufferSize
// query, allocated stream-ordered on the solver's stream and freed there, so the call stays asynchronous: nothing is
// synchronized.  cuSOLVER / hipSOLVER have no ?geqlf, so uplo = 'L' is run as a QR of the doubly reversed matrix
// A' = J A J (J = row / column reversal): A = Q L  <=>  A' = (J Q J) (J L J), J L J upper triangular, and the
// reflector i of the QR is the reflector k - 1 - i of the QL (k = min(m, n)), which is the order the scalars tau come
// back in.  The reversal and its inverse are index-only qr_reverse copies, so the factors are stored exactly as LAPACK ?geqlf stores them.
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <algorithm>
#include <cctype>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "ddla_connector.h"
#include "ddla_stream_impl.h"
#include "pgeqrf_kernels.h"

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

template <typename T>
void cpu_geqrf(bool lower, int m, int n, T* A, int lda, T* tau)
{
    // Workspace query, then the factorization.  (LAPACK returns the optimal size in work[0], as a real number.)
    int info = 0;
    T opt{};
    lapack_geqrf(lower, m, n, A, lda, tau, &opt, -1, &info);
    if (info != 0) throw std::runtime_error("geqrf: LAPACK workspace query failed");
    const int lwork = std::max(1, static_cast<int>(std::real(opt)));
    std::vector<T> work(static_cast<std::size_t>(lwork));
    lapack_geqrf(lower, m, n, A, lda, tau, work.data(), lwork, &info);
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

// A = Q R on the device.  The vendor routine's scratch and its device `info` come out of ONE stream-ordered
// allocation (the info slot first, padded so the work array stays aligned).  tau is zeroed first: cuSOLVER's
// single-precision panel kernel (geqr2_smem_domino_fast) reads tau before it writes it (compute-sanitizer initcheck
// reports it; the result does not depend on the value -- a 0xFF-poisoned tau gives bit-identical output), and tau is
// an output the caller may hand over uninitialized.
template <typename T>
void gpu_geqrf_qr(gpusolverHandle_t h, int m, int n, T* A, int lda, T* tau)
{
    gpuStream_t stream;
    SOLVER_CHECK(gpusolverGetStream(h, &stream));

    int lwork = 0;
    SOLVER_CHECK(geqrf_buffer_size(h, m, n, A, lda, &lwork));

    constexpr std::size_t kInfoBytes = 256;
    char* buf = nullptr;
    GPU_CHECK(gpuMallocAsync(reinterpret_cast<void**>(&buf),
                             kInfoBytes + static_cast<std::size_t>(std::max(lwork, 0)) * sizeof(T), stream));
    int* d_info = reinterpret_cast<int*>(buf);
    T* work = reinterpret_cast<T*>(buf + kInfoBytes);

    GPU_CHECK(gpuMemsetAsync(tau, 0, static_cast<std::size_t>(std::min(m, n)) * sizeof(T), stream));
    const gpusolverStatus_t status = geqrf_call(h, m, n, A, lda, tau, work, lwork, d_info);

    GPU_CHECK(gpuFreeAsync(buf, stream));
    SOLVER_CHECK(status);
}

// A = Q L on the device through the QR of A' = J A J (see the header comment).
template <typename T>
void gpu_geqrf_ql(const DdlaHandle_t& handle, int m, int n, T* A, int lda, T* tau)
{
    gpuStream_t stream;
    SOLVER_CHECK(gpusolverGetStream(handle->solverH, &stream));
    const int k = std::min(m, n);

    // [ A' (m x n, ld m) | tau' (k) ], one stream-ordered allocation
    const std::size_t a_bytes = (static_cast<std::size_t>(m) * n * sizeof(T) + 255) / 256 * 256;
    char* buf = nullptr;
    GPU_CHECK(gpuMallocAsync(reinterpret_cast<void**>(&buf), a_bytes + static_cast<std::size_t>(k) * sizeof(T), stream));
    T* const Ar = reinterpret_cast<T*>(buf);
    T* const taur = reinterpret_cast<T*>(buf + a_bytes);

    detail::qr_reverse<T>(handle, m, n, A, lda, Ar, m);
    gpu_geqrf_qr<T>(handle->solverH, m, n, Ar, m, taur);
    detail::qr_reverse<T>(handle, m, n, Ar, m, A, lda);
    // tau_L(k - 1 - i) = tau'(i): the same reversal on a k x 1 column
    detail::qr_reverse<T>(handle, k, 1, taur, k, tau, k);

    GPU_CHECK(gpuFreeAsync(buf, stream));
}

#else // DDLA_GPU_VENDOR

template <typename T>
void gpu_geqrf_qr(gpusolverHandle_t, int, int, T*, int, T*)
{
    throw std::runtime_error("geqrf: no GPU vendor selected at compile time");
}
template <typename T>
void gpu_geqrf_ql(const DdlaHandle_t&, int, int, T*, int, T*)
{
    throw std::runtime_error("geqrf: no GPU vendor selected at compile time");
}

#endif // DDLA_GPU_VENDOR
#endif // DDLA_HAS_GPU

} // anonymous namespace

template <DdlaBackend Backend, typename T>
void geqrf(const DdlaHandle_t& handle, char uplo, int m, int n, T* A, int lda, T* tau)
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
    if (k == 0) return;   // nothing to factor
    if (A == nullptr) throw std::runtime_error("geqrf: null A");
    if (tau == nullptr) throw std::runtime_error("geqrf: null tau");
    if (lda < std::max(1, m)) throw std::runtime_error("geqrf: lda too small");

    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        cpu_geqrf(u == 'L', m, n, A, lda, tau);
#endif
    } else {
#if DDLA_HAS_GPU
        if (u == 'L') gpu_geqrf_ql<T>(handle, m, n, A, lda, tau);
        else gpu_geqrf_qr<T>(handle->solverH, m, n, A, lda, tau);
#endif
    }
}

#define INSTANTIATE_GEQRF(BACKEND, TYPE)                                     \
    template void geqrf<BACKEND, TYPE>(                                      \
        const DdlaHandle_t&, char, int, int, TYPE*, int, TYPE*)

#if DDLA_HAS_CPU
INSTANTIATE_GEQRF(DdlaBackend::CPU, float);
INSTANTIATE_GEQRF(DdlaBackend::CPU, double);
INSTANTIATE_GEQRF(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_GEQRF(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_GEQRF(DdlaBackend::GPU, float);
INSTANTIATE_GEQRF(DdlaBackend::GPU, double);
INSTANTIATE_GEQRF(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_GEQRF(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_GEQRF

} // namespace ddla
