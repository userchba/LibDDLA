// ---------------------------------------------------------------------------
// Backend-templated triangular factor of a block reflector (LAPACK ?larft, STOREV = 'C'): the compact-WY form of a product of
// k Householder reflectors H(i) = I - tau(i) v(i) v(i)^H,
//     direct = 'F':  H = H(1) H(2) ... H(k) = I - V T V^H,  T upper triangular,
//     direct = 'B':  H = H(k) ... H(2) H(1) = I - V T V^H,  T lower triangular,
// V being the n x k matrix of the reflector vectors in ?geqrf / ?geqlf storage (see ddla.h).
//
// Structure follows src/geqrf.cpp: one definition covering CPU (Fortran LAPACK ?larft on host pointers) and GPU (device
// pointers, enqueued on the handle's stream), selected with `if constexpr` on the Backend template parameter, the arms
// guarded by the BUILD CAPABILITY macros (DDLA_HAS_CPU / DDLA_HAS_GPU).  The public interface is the templated
// ddla::larft declared in <ddla/ddla.h>.
//
// GPU arm: the vendor libraries offer no usable ?larft -- cuSOLVER has cusolverDnXlarft only from CUDA 12.4 on, hipSOLVER has
// none -- so it is built the way MAGMA's magma_?larft_batched builds it: ONE BLAS-3 call forms the Gram matrix V^H V, and a
// small kernel runs the column recurrence
//     T(i,i) = tau(i),   T(0:i-1, i) = -tau(i) T(0:i-1, 0:i-1) (V^H V)(0:i-1, i)         ('F'; 'B' mirrors it)
// with the T row in shared memory.  Three steps, all stream-ordered on the handle's stream (nothing is synchronized):
//   1. the unit-triangular k x k head (F) / tail (B) of V is built explicitly into scratch -- V keeps LAPACK storage, with
//      R / L above / below the diagonal, so it cannot be fed to a BLAS call as it stands;
//   2. VV = V^H V = herk(head) + herk(the other n - k rows of V);
//   3. qr_build_T (pgeqrf_kernels.cpp): one block per row of T.
// The k x k scratch buffers are allocated and freed stream-ordered, like geqrf's vendor workspace.
//
// T is written completely: the triangle LAPACK owns and zeros in the other one (LAPACK itself leaves the other triangle
// alone); rows beyond k of a padded T (ldt > k) are not touched.
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <algorithm>
#include <cctype>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "ddla_connector.h"
#include "ddla_stream_impl.h"
#include "pgeqrf_kernels.h"

#if DDLA_HAS_CPU
// Host LAPACK ?larft (Fortran ABI; the CPU BLAS library must provide LAPACK -- OpenBLAS and MKL do).  V is not const: LAPACK
// releases before 3.10 overwrite the unit entries of V while they run and put the old values back.
extern "C" {
void slarft_(const char*, const char*, const int*, const int*, float*, const int*, const float*, float*, const int*);
void dlarft_(const char*, const char*, const int*, const int*, double*, const int*, const double*, double*, const int*);
void clarft_(const char*, const char*, const int*, const int*, std::complex<float>*, const int*,
             const std::complex<float>*, std::complex<float>*, const int*);
void zlarft_(const char*, const char*, const int*, const int*, std::complex<double>*, const int*,
             const std::complex<double>*, std::complex<double>*, const int*);
}
#endif

namespace ddla {

namespace {

inline const char* larft_backend_name(DdlaBackend backend)
{
    return backend == DdlaBackend::CPU ? "CPU" : "GPU";
}

#if DDLA_HAS_CPU
inline void lapack_larft(char direct, int n, int k, float* V, int ldv, const float* tau, float* t, int ldt)
{
    const char storev = 'C';
    slarft_(&direct, &storev, &n, &k, V, &ldv, tau, t, &ldt);
}
inline void lapack_larft(char direct, int n, int k, double* V, int ldv, const double* tau, double* t, int ldt)
{
    const char storev = 'C';
    dlarft_(&direct, &storev, &n, &k, V, &ldv, tau, t, &ldt);
}
inline void lapack_larft(char direct, int n, int k, std::complex<float>* V, int ldv, const std::complex<float>* tau,
                         std::complex<float>* t, int ldt)
{
    const char storev = 'C';
    clarft_(&direct, &storev, &n, &k, V, &ldv, tau, t, &ldt);
}
inline void lapack_larft(char direct, int n, int k, std::complex<double>* V, int ldv, const std::complex<double>* tau,
                         std::complex<double>* t, int ldt)
{
    const char storev = 'C';
    zlarft_(&direct, &storev, &n, &k, V, &ldv, tau, t, &ldt);
}

// LAPACK writes only the triangle T owns, so the other one is zeroed first (the GPU arm writes it as zeros too).
template <typename T>
void cpu_larft(char direct, int n, int k, T* V, int ldv, const T* tau, T* t, int ldt)
{
    for (int c = 0; c < k; ++c) std::fill(t + static_cast<std::size_t>(c) * ldt, t + static_cast<std::size_t>(c) * ldt + k, T(0));
    lapack_larft(direct, n, k, V, ldv, tau, t, ldt);
}
#endif // DDLA_HAS_CPU

#if DDLA_HAS_GPU
#if DDLA_GPU_VENDOR

// T of the block reflector V on the device (see the header comment).  `backward` selects 'B'.
template <typename T>
void gpu_larft(const DdlaHandle_t& handle, bool backward, int n, int k, T* V, int ldv, const T* tau, T* t, int ldt)
{
    using rt = typename real_type<T>::type;
    gpuStream_t stream = handle->stream;
    const int mb = std::max(n, 1);   // one process row: the block size does not enter the index map

    // [ head / tail of V (k x k) | V^H V (k x k) ], one stream-ordered allocation
    const std::size_t kk_bytes = static_cast<std::size_t>(k) * k * sizeof(T);
    const std::size_t head_bytes = (kk_bytes + 255) / 256 * 256;
    char* buf = nullptr;
    GPU_CHECK(gpuMallocAsync(reinterpret_cast<void**>(&buf), head_bytes + kk_bytes, stream));
    T* const head = reinterpret_cast<T*>(buf);
    T* const vv = reinterpret_cast<T*>(buf + head_bytes);

    // V^H V of the explicit V = [head; rest] (F) / [rest; tail] (B): the k x k unit-triangular block comes out of the
    // window in LAPACK storage, the other n - k rows are pure V entries.  Only the upper triangle of VV is formed and read.
    if (!backward) {
        detail::qr_build_v<T>(handle, false, 0, head, k, V, ldv, k, k, 0, n, 0, mb, 0, 0, 1);
        herk<DdlaBackend::GPU, T>(handle, 'U', 'C', k, k, rt(1), head, k, rt(0), vv, k);
        if (n > k) herk<DdlaBackend::GPU, T>(handle, 'U', 'C', k, n - k, rt(1), V + k, ldv, rt(1), vv, k);
    } else {
        detail::qr_build_v<T>(handle, false, 1, head, k, V, ldv, k, k, 0, n, n - k, mb, 0, 0, 1);
        if (n > k) {
            herk<DdlaBackend::GPU, T>(handle, 'U', 'C', k, n - k, rt(1), V, ldv, rt(0), vv, k);
            herk<DdlaBackend::GPU, T>(handle, 'U', 'C', k, k, rt(1), head, k, rt(1), vv, k);
        } else {
            herk<DdlaBackend::GPU, T>(handle, 'U', 'C', k, k, rt(1), head, k, rt(0), vv, k);
        }
    }
    detail::qr_build_T<T>(handle, k, vv, tau, t, ldt, backward);

    GPU_CHECK(gpuFreeAsync(buf, stream));
}

#else // DDLA_GPU_VENDOR

template <typename T>
void gpu_larft(const DdlaHandle_t&, bool, int, int, T*, int, const T*, T*, int)
{
    throw std::runtime_error("larft: no GPU vendor selected at compile time");
}

#endif // DDLA_GPU_VENDOR
#endif // DDLA_HAS_GPU

} // anonymous namespace

template <DdlaBackend Backend, typename T>
void larft(const DdlaHandle_t& handle, char direct, char storev, int n, int k, T* V, int ldv, const T* tau, T* Tmat,
           int ldt)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU larft is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU larft is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("larft: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("larft: template backend ") + larft_backend_name(Backend) +
            " does not match handle backend " + larft_backend_name(actual));
    }
    const char d = static_cast<char>(std::toupper(static_cast<unsigned char>(direct)));
    if (d != 'F' && d != 'B') throw std::runtime_error("larft: direct must be 'F' or 'B'");
    if (std::toupper(static_cast<unsigned char>(storev)) != 'C')
        throw std::runtime_error("larft: only storev = 'C' (columnwise) is supported");
    if (n < 0 || k < 0) throw std::runtime_error("larft: negative size");

    if (k == 0) return;   // nothing to build
    if (k > n) throw std::runtime_error("larft: k must not exceed n");
    if (V == nullptr) throw std::runtime_error("larft: null V");
    if (tau == nullptr) throw std::runtime_error("larft: null tau");
    if (Tmat == nullptr) throw std::runtime_error("larft: null T");
    if (ldv < std::max(1, n)) throw std::runtime_error("larft: ldv too small");
    if (ldt < k) throw std::runtime_error("larft: ldt too small");

    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        cpu_larft(d, n, k, V, ldv, tau, Tmat, ldt);
#endif
    } else {
#if DDLA_HAS_GPU
        gpu_larft<T>(handle, d == 'B', n, k, V, ldv, tau, Tmat, ldt);
#endif
    }
}

#define INSTANTIATE_LARFT(BACKEND, TYPE)                                     \
    template void larft<BACKEND, TYPE>(                                      \
        const DdlaHandle_t&, char, char, int, int, TYPE*, int, const TYPE*, TYPE*, int)

#if DDLA_HAS_CPU
INSTANTIATE_LARFT(DdlaBackend::CPU, float);
INSTANTIATE_LARFT(DdlaBackend::CPU, double);
INSTANTIATE_LARFT(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_LARFT(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_LARFT(DdlaBackend::GPU, float);
INSTANTIATE_LARFT(DdlaBackend::GPU, double);
INSTANTIATE_LARFT(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_LARFT(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_LARFT

} // namespace ddla
