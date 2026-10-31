// ---------------------------------------------------------------------------
// Backend-templated row interchanges (LAPACK ?laswp):  for i = k1..k2, swap row i of A with row ipiv(i).
//
// Structure follows src/getrf.cpp: one definition covering CPU (Fortran LAPACK ?laswp on host pointers) and GPU, selected
// with `if constexpr` on the Backend template parameter, the arms guarded by DDLA_HAS_CPU / DDLA_HAS_GPU.  The vendor
// calls are private to this translation unit; the public interface is the templated ddla::laswp declared in
// <ddla/ddla.h>.
//
// GPU arm:
//   * cuSOLVER has a laswp (GPU_HAS_SOLVER_LASWP == 1): used when incx == 1, the only stride any caller uses.
//   * hipSOLVER has none, and cuSOLVER's stride semantics are not documented: every other case is a row-by-row
//     loop over swap<GPU, T>.  The pivots are copied to the host first (on the handle's stream, then synchronized),
//     because the swap sequence depends on them; the swaps themselves are enqueued on the handle's stream.
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "ddla_connector.h"
#include "ddla_stream.h"

#if DDLA_HAS_CPU
// Host LAPACK ?laswp (Fortran ABI; the CPU BLAS library must provide LAPACK -- OpenBLAS and MKL do).
extern "C" {
void slaswp_(const int*, float*, const int*, const int*, const int*, const int*, const int*);
void dlaswp_(const int*, double*, const int*, const int*, const int*, const int*, const int*);
void claswp_(const int*, std::complex<float>*, const int*, const int*, const int*, const int*, const int*);
void zlaswp_(const int*, std::complex<double>*, const int*, const int*, const int*, const int*, const int*);
}
#endif

namespace ddla {

namespace {

inline const char* laswp_backend_name(DdlaBackend backend)
{
    return backend == DdlaBackend::CPU ? "CPU" : "GPU";
}

#if DDLA_HAS_CPU
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
#endif // DDLA_HAS_CPU

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

// Row-by-row fallback, LAPACK semantics for any nonzero incx.  Pivot of row i is ipiv[k1 - 1 + (i - k1) * |incx|]
// (1-based row numbers); rows are processed k1..k2 for incx > 0 and k2..k1 for incx < 0 (so incx = -1 undoes incx = 1).
template <typename T>
void gpu_laswp_rows(const DdlaHandle_t& handle, int n, T* A, int lda, int k1, int k2, const int* ipiv, int incx)
{
    const std::size_t stride = static_cast<std::size_t>(std::abs(incx));
    const std::size_t span = static_cast<std::size_t>(k2 - k1) * stride + 1;
    std::vector<int> piv(span);
    DDLA_CHECK(ddlaMemcpy(handle, piv.data(), ipiv + (k1 - 1), span * sizeof(int)));
    DDLA_CHECK(ddlaSynchronize(handle));

    const int first = incx > 0 ? k1 : k2;
    const int last = incx > 0 ? k2 : k1;
    const int step = incx > 0 ? 1 : -1;
    for (int i = first;; i += step) {
        const int p = piv[static_cast<std::size_t>(i - k1) * stride];
        if (p < 1) throw std::runtime_error("laswp: pivot row below 1");
        if (p != i)
            swap<DdlaBackend::GPU, T>(handle, n, A + (i - 1), lda, A + (p - 1), lda);
        if (i == last) break;
    }
}

template <typename T>
void gpu_laswp(const DdlaHandle_t& handle, int n, T* A, int lda, int k1, int k2, const int* ipiv, int incx)
{
#if DDLA_GPU_VENDOR && GPU_HAS_SOLVER_LASWP
    if (incx == 1) {
        SOLVER_CHECK(solver_laswp(static_cast<gpusolverHandle_t>(handle->solverH), n, A, lda, k1, k2, ipiv, incx));
        return;
    }
#endif
#if DDLA_GPU_VENDOR
    gpu_laswp_rows(handle, n, A, lda, k1, k2, ipiv, incx);
#else
    throw std::runtime_error("laswp: no GPU vendor selected at compile time");
#endif
}

#endif // DDLA_HAS_GPU

} // anonymous namespace

template <DdlaBackend Backend, typename T>
void laswp(const DdlaHandle_t& handle, int n, T* A, int lda, int k1, int k2, const int* ipiv, int incx)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU laswp is not available in this LibDDLA build");
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
    if (n <= 0 || k1 > k2) return;
    if (k1 < 1) throw std::runtime_error("laswp: k1 must be >= 1");
    if (incx == 0) throw std::runtime_error("laswp: incx must be nonzero");
    if (A == nullptr) throw std::runtime_error("laswp: null A");
    if (ipiv == nullptr) throw std::runtime_error("laswp: null ipiv");
    if (lda < std::max(1, k2)) throw std::runtime_error("laswp: lda too small");

    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        cpu_laswp(n, A, lda, k1, k2, ipiv, incx);
#endif
    } else {
#if DDLA_HAS_GPU
        gpu_laswp(handle, n, A, lda, k1, k2, ipiv, incx);
#endif
    }
}

#define INSTANTIATE_LASWP(BACKEND, TYPE)                                     \
    template void laswp<BACKEND, TYPE>(                                      \
        const DdlaHandle_t&, int, TYPE*, int, int, int, const int*, int)

#if DDLA_HAS_CPU
INSTANTIATE_LASWP(DdlaBackend::CPU, float);
INSTANTIATE_LASWP(DdlaBackend::CPU, double);
INSTANTIATE_LASWP(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_LASWP(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_LASWP(DdlaBackend::GPU, float);
INSTANTIATE_LASWP(DdlaBackend::GPU, double);
INSTANTIATE_LASWP(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_LASWP(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_LASWP

} // namespace ddla
