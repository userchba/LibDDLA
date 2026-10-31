// ---------------------------------------------------------------------------
// Backend-templated LU factorization with partial pivoting (LAPACK ?getrf):  A = P * L * U.
//
// Structure follows src/scal.cpp / src/geam.cpp: one definition covering CPU (Fortran LAPACK ?getrf on host pointers)
// and GPU (cuSOLVER / hipSOLVER getrf on device pointers, enqueued on the handle's stream), selected with
// `if constexpr` on the Backend template parameter, the arms guarded by the BUILD CAPABILITY macros
// (DDLA_HAS_CPU / DDLA_HAS_GPU).  The vendor calls are private to this translation unit; the public interface is the
// templated ddla::getrf declared in <ddla/ddla.h>.
//
// Three components (src/lapack_impl.h): getrf_check validates, getrf_plan sizes and allocates the vendor workspace,
// getrf_run factorizes; the public getrf is check -> plan -> run.
//
// GPU arm: the vendor routine needs a scratch buffer.  getrf_plan sizes it with the ?getrf_bufferSize query (the
// maximum over the planned shapes) and allocates it stream-ordered on the handle's stream -- the stream the solver
// handle is bound to -- and getrf_run re-queries the size of its own shape, so the call stays asynchronous: nothing
// is synchronized, `info` is valid once the caller synchronizes the stream.
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>

#include "ddla_connector.h"
#include "ddla_stream.h"
#include "lapack_impl.h"
#include "plan_util.h"

#if DDLA_HAS_CPU
// Host LAPACK ?getrf (Fortran ABI; the CPU BLAS library must provide LAPACK -- OpenBLAS and MKL do).
extern "C" {
void sgetrf_(const int*, const int*, float*, const int*, int*, int*);
void dgetrf_(const int*, const int*, double*, const int*, int*, int*);
void cgetrf_(const int*, const int*, std::complex<float>*, const int*, int*, int*);
void zgetrf_(const int*, const int*, std::complex<double>*, const int*, int*, int*);
}
#endif

namespace ddla {

namespace {

inline const char* getrf_backend_name(DdlaBackend backend)
{
    return backend == DdlaBackend::CPU ? "CPU" : "GPU";
}

#if DDLA_HAS_CPU
inline void cpu_getrf(int m, int n, float* A, int lda, int* ipiv, int* info)
{
    sgetrf_(&m, &n, A, &lda, ipiv, info);
}

inline void cpu_getrf(int m, int n, double* A, int lda, int* ipiv, int* info)
{
    dgetrf_(&m, &n, A, &lda, ipiv, info);
}

inline void cpu_getrf(int m, int n, std::complex<float>* A, int lda, int* ipiv, int* info)
{
    cgetrf_(&m, &n, A, &lda, ipiv, info);
}

inline void cpu_getrf(int m, int n, std::complex<double>* A, int lda, int* ipiv, int* info)
{
    zgetrf_(&m, &n, A, &lda, ipiv, info);
}
#endif // DDLA_HAS_CPU

#if DDLA_HAS_GPU
#if DDLA_GPU_VENDOR

// Per scalar type: the buffer-size query and the factorization itself.
inline gpusolverStatus_t getrf_buffer_size(gpusolverHandle_t h, int m, int n, float* A, int lda, int* lwork)
{
    return gpusolverSgetrf_bufferSize(h, m, n, A, lda, lwork);
}
inline gpusolverStatus_t getrf_buffer_size(gpusolverHandle_t h, int m, int n, double* A, int lda, int* lwork)
{
    return gpusolverDgetrf_bufferSize(h, m, n, A, lda, lwork);
}
inline gpusolverStatus_t getrf_buffer_size(gpusolverHandle_t h, int m, int n, std::complex<float>* A, int lda, int* lwork)
{
    return gpusolverCgetrf_bufferSize(h, m, n, (gpusolverComplex*)A, lda, lwork);
}
inline gpusolverStatus_t getrf_buffer_size(gpusolverHandle_t h, int m, int n, std::complex<double>* A, int lda, int* lwork)
{
    return gpusolverZgetrf_bufferSize(h, m, n, (gpusolverDoubleComplex*)A, lda, lwork);
}

inline gpusolverStatus_t getrf_call(gpusolverHandle_t h, int m, int n, float* A, int lda, float* work, int lwork,
                                    int* ipiv, int* info)
{
    return gpusolverSgetrf(h, m, n, A, lda, work, lwork, ipiv, info);
}
inline gpusolverStatus_t getrf_call(gpusolverHandle_t h, int m, int n, double* A, int lda, double* work, int lwork,
                                    int* ipiv, int* info)
{
    return gpusolverDgetrf(h, m, n, A, lda, work, lwork, ipiv, info);
}
inline gpusolverStatus_t getrf_call(gpusolverHandle_t h, int m, int n, std::complex<float>* A, int lda,
                                    std::complex<float>* work, int lwork, int* ipiv, int* info)
{
    return gpusolverCgetrf(h, m, n, (gpusolverComplex*)A, lda, (gpusolverComplex*)work, lwork, ipiv, info);
}
inline gpusolverStatus_t getrf_call(gpusolverHandle_t h, int m, int n, std::complex<double>* A, int lda,
                                    std::complex<double>* work, int lwork, int* ipiv, int* info)
{
    return gpusolverZgetrf(h, m, n, (gpusolverDoubleComplex*)A, lda, (gpusolverDoubleComplex*)work, lwork, ipiv, info);
}

// The factorization on the device: query the exact lwork of this shape (a host call), check that the planned
// workspace holds it, run.  The workspace lives in the workspace, on the stream the solver runs on.
template <typename T>
gpusolverStatus_t gpu_getrf(const DdlaHandle_t& handle, vector<T>& work, int m, int n, T* A, int lda,
                            int* ipiv, int* info)
{
    int lwork = 0;
    SOLVER_CHECK(getrf_buffer_size(static_cast<gpusolverHandle_t>(handle->solverH), m, n, A, lda, &lwork));
    assert(static_cast<std::size_t>(lwork) <= work.capacity());
    return getrf_call(static_cast<gpusolverHandle_t>(handle->solverH), m, n, A, lda, lwork > 0 ? work.data() : nullptr, lwork, ipiv, info);
}

#else // DDLA_GPU_VENDOR

template <typename T>
gpusolverStatus_t gpu_getrf(const DdlaHandle_t&, vector<T>&, int, int, T*, int, int*, int*)
{
    throw std::runtime_error("getrf: no GPU vendor selected at compile time");
}

#endif // DDLA_GPU_VENDOR
#endif // DDLA_HAS_GPU

} // anonymous namespace

namespace lapack_impl {

// Component 1.
template <DdlaBackend Backend>
void getrf_check(const DdlaHandle_t& handle, int m, int n, const void* A, int lda,
                 const int* ipiv, const int* info)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU getrf is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU getrf is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("getrf: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("getrf: template backend ") + getrf_backend_name(Backend) +
            " does not match handle backend " + getrf_backend_name(actual));
    }
    if (m < 0 || n < 0) throw std::runtime_error("getrf: negative matrix size");
    if (info == nullptr) throw std::runtime_error("getrf: null info");

    if (m == 0 || n == 0) return;   // nothing to factor: the pointers are not read
    if (A == nullptr) throw std::runtime_error("getrf: null A");
    if (ipiv == nullptr) throw std::runtime_error("getrf: null ipiv");
    if (lda < std::max(1, m)) throw std::runtime_error("getrf: lda too small");
}

// Component 2: the vendor workspace.  CPU handles (host LAPACK) need none.
template <typename T>
vector<T> getrf_plan(const DdlaHandle_t& handle, const Shape* shapes, std::size_t count, T* A, int lda)
{
    vector<T> work(handle);
#if DDLA_HAS_GPU && DDLA_GPU_VENDOR
    if (ddlaGetBackend(handle) == DdlaBackend::GPU) {
        int lwork = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const Shape& s = shapes[i];
            if (s.m <= 0 || s.n <= 0) continue;
            int need = 0;
            SOLVER_CHECK(getrf_buffer_size(static_cast<gpusolverHandle_t>(handle->solverH), s.m, s.n, A, lda, &need));
            lwork = std::max(lwork, need);
        }
        work.resize(static_cast<std::size_t>(lwork));
    }
#else
    (void)shapes; (void)count; (void)A; (void)lda;
#endif
    return work;
}

// Component 3.
template <DdlaBackend Backend, typename T>
void getrf_run(const DdlaHandle_t& handle, [[maybe_unused]] vector<T>& work, int m, int n, T* A, int lda,
               int* ipiv, int* info)
{
    if (m <= 0 || n <= 0) {
        // Nothing to factor: success, reported where the caller will look for it.
        if constexpr (Backend == DdlaBackend::CPU) {
            *info = 0;
        } else {
            DDLA_CHECK(ddlaMemset(handle, info, 0, sizeof(int)));
        }
        return;
    }

    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        cpu_getrf(m, n, A, lda, ipiv, info);
#endif
    } else {
#if DDLA_HAS_GPU
        SOLVER_CHECK(gpu_getrf(handle, work, m, n, A, lda, ipiv, info));
#endif
    }
}

} // namespace lapack_impl

template <DdlaBackend Backend, typename T>
void getrf(const DdlaHandle_t& handle, int m, int n, T* A, int lda, int* ipiv, int* info)
{
    // 1. parameter check.
    lapack_impl::getrf_check<Backend>(handle, m, n, A, lda, ipiv, info);
    // 2. the vendor workspace of this one shape (an empty shape plans nothing).
    vector<T> work = lapack_impl::getrf_plan<T>(handle, {{m, n}}, A, lda);
    // 3. the factorization (an empty problem only reports info = 0).
    lapack_impl::getrf_run<Backend, T>(handle, work, m, n, A, lda, ipiv, info);
}

#define INSTANTIATE_GETRF(BACKEND, TYPE)                                     \
    template void getrf<BACKEND, TYPE>(                                      \
        const DdlaHandle_t&, int, int, TYPE*, int, int*, int*);              \
    template void lapack_impl::getrf_run<BACKEND, TYPE>(                     \
        const DdlaHandle_t&, vector<TYPE>&, int, int, TYPE*, int, int*, int*)
#define INSTANTIATE_GETRF_PLAN(TYPE)                                         \
    template vector<TYPE> lapack_impl::getrf_plan<TYPE>(      \
        const DdlaHandle_t&, const lapack_impl::Shape*, std::size_t, TYPE*, int)

#if DDLA_HAS_CPU
template void lapack_impl::getrf_check<DdlaBackend::CPU>(
    const DdlaHandle_t&, int, int, const void*, int, const int*, const int*);
INSTANTIATE_GETRF(DdlaBackend::CPU, float);
INSTANTIATE_GETRF(DdlaBackend::CPU, double);
INSTANTIATE_GETRF(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_GETRF(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
template void lapack_impl::getrf_check<DdlaBackend::GPU>(
    const DdlaHandle_t&, int, int, const void*, int, const int*, const int*);
INSTANTIATE_GETRF(DdlaBackend::GPU, float);
INSTANTIATE_GETRF(DdlaBackend::GPU, double);
INSTANTIATE_GETRF(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_GETRF(DdlaBackend::GPU, std::complex<double>);
#endif

INSTANTIATE_GETRF_PLAN(float);
INSTANTIATE_GETRF_PLAN(double);
INSTANTIATE_GETRF_PLAN(std::complex<float>);
INSTANTIATE_GETRF_PLAN(std::complex<double>);

#undef INSTANTIATE_GETRF_PLAN
#undef INSTANTIATE_GETRF

} // namespace ddla
