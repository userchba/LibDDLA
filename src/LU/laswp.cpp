#include "BLAS/blas_impl.h"
// ---------------------------------------------------------------------------
// Backend-templated row interchanges (LAPACK ?laswp):  for i = k1..k2, swap row i of A with row ipiv(i).
//
// Dispatch follows the Backend template parameter. CPU code is always compiled;
// DDLA_HAS_GPU enables GPU code, and DDLA_USE_CUDA / DDLA_USE_HIP select its vendor.
//
// Three components (plan_util.h): laswp_check; laswp_workspace_size (host bytes of the pivot staging);
// laswp_run on the caller's scratch.
//
// GPU arm:
//   * cuSOLVER has a laswp (GPU_HAS_SOLVER_LASWP == 1): used when incx == 1, the only stride any caller uses.
//   * hipSOLVER has none, and cuSOLVER's stride semantics are not documented: every other case is a row-by-row
//     loop over blas_impl::swap_run<GPU, T>.  The pivots are copied to the host first (on the handle's stream, then synchronized),
//     because the swap sequence depends on them; the swaps themselves are enqueued on the handle's stream.  The host
//     copy lives in the caller's host scratch (laswp_workspace_size), so a call allocates nothing.
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "ddla_connector.h"
#include "ddla_stream.h"
#include "lapack_impl.h"

// Host LAPACK ?laswp (Fortran ABI; the CPU BLAS library must provide LAPACK -- OpenBLAS and MKL do).
extern "C" {
void slaswp_(const int*, float*, const int*, const int*, const int*, const int*, const int*);
void dlaswp_(const int*, double*, const int*, const int*, const int*, const int*, const int*);
void claswp_(const int*, std::complex<float>*, const int*, const int*, const int*, const int*, const int*);
void zlaswp_(const int*, std::complex<double>*, const int*, const int*, const int*, const int*, const int*);
}

namespace ddla {

namespace {

inline const char* laswp_backend_name(DdlaBackend backend)
{
    return backend == DdlaBackend::CPU ? "CPU" : "GPU";
}

inline void cpu_laswp(int n, float* A, int lda, int k1, int k2, const int* ipiv, int incx)
{
    slaswp_(&n, A, &lda, &k1, &k2, ipiv, &incx);
}

inline void cpu_laswp(int n, double* A, int lda, int k1, int k2, const int* ipiv, int incx)
{
    dlaswp_(&n, A, &lda, &k1, &k2, ipiv, &incx);
}

inline void cpu_laswp(int n, std::complex<float>* A, int lda, int k1, int k2, const int* ipiv, int incx)
{
    claswp_(&n, A, &lda, &k1, &k2, ipiv, &incx);
}

inline void cpu_laswp(int n, std::complex<double>* A, int lda, int k1, int k2, const int* ipiv, int incx)
{
    zlaswp_(&n, A, &lda, &k1, &k2, ipiv, &incx);
}

#if DDLA_HAS_GPU
#if DDLA_GPU_VENDOR && GPU_HAS_SOLVER_LASWP

inline gpusolverStatus_t solver_laswp(gpusolverHandle_t h, int n, float* A, int lda, int k1, int k2, const int* ipiv,
                                      int incx)
{
    return gpusolverSlaswp(h, n, A, lda, k1, k2, ipiv, incx);
}
inline gpusolverStatus_t solver_laswp(gpusolverHandle_t h, int n, double* A, int lda, int k1, int k2, const int* ipiv,
                                      int incx)
{
    return gpusolverDlaswp(h, n, A, lda, k1, k2, ipiv, incx);
}
inline gpusolverStatus_t solver_laswp(gpusolverHandle_t h, int n, std::complex<float>* A, int lda, int k1, int k2,
                                      const int* ipiv, int incx)
{
    return gpusolverClaswp(h, n, (gpusolverComplex*)A, lda, k1, k2, ipiv, incx);
}
inline gpusolverStatus_t solver_laswp(gpusolverHandle_t h, int n, std::complex<double>* A, int lda, int k1, int k2,
                                      const int* ipiv, int incx)
{
    return gpusolverZlaswp(h, n, (gpusolverDoubleComplex*)A, lda, k1, k2, ipiv, incx);
}

#endif // DDLA_GPU_VENDOR && GPU_HAS_SOLVER_LASWP

// Whether a GPU call goes through the row-by-row fallback (and so needs the host copy of the pivots): every case but
// the vendor's laswp with incx == 1.
inline bool gpu_laswp_uses_rows(int incx)
{
#if DDLA_GPU_VENDOR && GPU_HAS_SOLVER_LASWP
    return incx != 1;
#else
    (void)incx;
    return true;
#endif
}

// Row-by-row fallback, LAPACK semantics for any nonzero incx.  Pivot of row i is ipiv[k1 - 1 + (i - k1) * |incx|]
// (1-based row numbers); rows are processed k1..k2 for incx > 0 and k2..k1 for incx < 0 (so incx = -1 undoes incx = 1).
// The pivots are staged in the host buffer.
template <typename T>
void gpu_laswp_rows(const DdlaHandle_t& handle, int* piv, std::size_t piv_bytes, int n, T* A, int lda, int k1, int k2,
                    const int* ipiv, int incx)
{
    const std::size_t stride = static_cast<std::size_t>(incx < 0 ? -static_cast<long long>(incx) : incx);
    const std::size_t span = static_cast<std::size_t>(k2 - k1) * stride + 1;
    assert(span * sizeof(int) <= piv_bytes);
    (void)piv_bytes;
    DDLA_CHECK(ddlaMemcpy(handle, piv, ipiv + (k1 - 1), span * sizeof(int)));
    DDLA_CHECK(ddlaSynchronize(handle));

    const int first = incx > 0 ? k1 : k2;
    const int last = incx > 0 ? k2 : k1;
    const int step = incx > 0 ? 1 : -1;
    for (int i = first;; i += step) {
        const int p = piv[static_cast<std::size_t>(i - k1) * stride];
        if (p < 1) throw std::runtime_error("laswp: pivot row below 1");
        if (p != i)
            blas_impl::swap_run<DdlaBackend::GPU, T>(handle, n, A + (i - 1), lda, A + (p - 1), lda);
        if (i == last) break;
    }
}

template <typename T>
void gpu_laswp(const DdlaHandle_t& handle, void* host_work, std::size_t host_bytes, int n, T* A, int lda, int k1,
               int k2, const int* ipiv, int incx)
{
#if DDLA_GPU_VENDOR && GPU_HAS_SOLVER_LASWP
    if (incx == 1) {
        SOLVER_CHECK(solver_laswp(static_cast<gpusolverHandle_t>(handle->solverH), n, A, lda, k1, k2, ipiv, incx));
        return;
    }
#endif
#if DDLA_GPU_VENDOR
    gpu_laswp_rows(handle, static_cast<int*>(host_work), host_bytes, n, A, lda, k1, k2, ipiv, incx);
#else
    throw std::runtime_error("laswp: no GPU vendor selected at compile time");
#endif
}

#endif // DDLA_HAS_GPU

} // anonymous namespace

namespace lapack_impl {

// Component 1.
template <DdlaBackend Backend>
void laswp_check(const DdlaHandle_t& handle, int n, const void* A, int lda, int k1, int k2,
                 const int* ipiv, int incx)
{
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU laswp is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("laswp: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("laswp: template backend ") + laswp_backend_name(Backend) +
            " does not match handle backend " + laswp_backend_name(actual));
    }
    if (n <= 0 || k1 > k2) return;   // nothing to swap: the pointers are not read
    if (k1 < 1) throw std::runtime_error("laswp: k1 must be >= 1");
    if (incx == 0) throw std::runtime_error("laswp: incx must be nonzero");
    if (A == nullptr) throw std::runtime_error("laswp: null A");
    if (ipiv == nullptr) throw std::runtime_error("laswp: null ipiv");
    if (lda < std::max(1, k2)) throw std::runtime_error("laswp: lda too small");
}

// Component 2: the host copy of the pivots for the GPU row-by-row fallback, (k2 - k1) * |incx| + 1 entries.  Every
// other case needs none.
void laswp_workspace_size(const DdlaHandle_t& handle, int k1, int k2, int incx, std::size_t* host_bytes)
{
    *host_bytes = 0;
#if DDLA_HAS_GPU
    if (handle->backend == DdlaBackend::GPU && k1 <= k2 && incx != 0 && gpu_laswp_uses_rows(incx)) {
        const std::size_t stride = static_cast<std::size_t>(incx < 0 ? -static_cast<long long>(incx) : incx);
        *host_bytes = (static_cast<std::size_t>(k2 - k1) * stride + 1) * sizeof(int);
    }
#else
    (void)handle; (void)k1; (void)k2; (void)incx;
#endif
}

// Component 3.
template <DdlaBackend Backend, typename T>
void laswp_run(const DdlaHandle_t& handle, int n, T* A, int lda, int k1, int k2, const int* ipiv, int incx,
               [[maybe_unused]] void* host_work, [[maybe_unused]] std::size_t host_bytes)
{
    if (n <= 0 || k1 > k2) return;

    if constexpr (Backend == DdlaBackend::CPU) {
        cpu_laswp(n, A, lda, k1, k2, ipiv, incx);
    } else {
#if DDLA_HAS_GPU
        gpu_laswp(handle, host_work, host_bytes, n, A, lda, k1, k2, ipiv, incx);
#endif
    }
}

} // namespace lapack_impl

template <DdlaBackend Backend, typename T>
void laswp(const DdlaHandle_t& handle, int n, T* A, int lda, int k1, int k2, const int* ipiv, int incx)
{
    // 1. parameter check.
    lapack_impl::laswp_check<Backend>(handle, n, A, lda, k1, k2, ipiv, incx);
    // No-op: nothing to swap.
    if (n <= 0 || k1 > k2) return;
    // 2. the host staging of the pivots (none on a CPU handle).
    std::size_t host_bytes = 0;
    lapack_impl::laswp_workspace_size(handle, k1, k2, incx, &host_bytes);
    std::vector<char> pivots(host_bytes);
    // 3. the interchanges.
    lapack_impl::laswp_run<Backend, T>(handle, n, A, lda, k1, k2, ipiv, incx, pivots.data(), host_bytes);
}

#define INSTANTIATE_LASWP(BACKEND, TYPE)                                     \
    template void laswp<BACKEND, TYPE>(                                      \
        const DdlaHandle_t&, int, TYPE*, int, int, int, const int*, int);    \
    template void lapack_impl::laswp_run<BACKEND, TYPE>(                     \
        const DdlaHandle_t&, int, TYPE*, int, int, int, const int*, int, void*, std::size_t)

template void lapack_impl::laswp_check<DdlaBackend::CPU>(
    const DdlaHandle_t&, int, const void*, int, int, int, const int*, int);
INSTANTIATE_LASWP(DdlaBackend::CPU, float);
INSTANTIATE_LASWP(DdlaBackend::CPU, double);
INSTANTIATE_LASWP(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_LASWP(DdlaBackend::CPU, std::complex<double>);

#if DDLA_HAS_GPU
template void lapack_impl::laswp_check<DdlaBackend::GPU>(
    const DdlaHandle_t&, int, const void*, int, int, int, const int*, int);
INSTANTIATE_LASWP(DdlaBackend::GPU, float);
INSTANTIATE_LASWP(DdlaBackend::GPU, double);
INSTANTIATE_LASWP(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_LASWP(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_LASWP

} // namespace ddla
