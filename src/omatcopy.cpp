#include <ddla/geam.h>

#include <complex>
#include <stdexcept>
#include <string>

#include "ddla_stream_impl.h"

// ---------------------------------------------------------------------------
// ddla::omatcopy<Backend,T> -- the DdlaBackend-templated form of
// deblasOmatcopy, in the same shape as ddla::gemm / ddla::scal.
//
// Why this exists separately from deblasOmatcopy: every `deblas*` wrapper in
// include/ddla/ selects its implementation from the *translation unit's*
// vendor macros (`#if defined(DDLA_USE_CUDA) ... #elif defined(DDLA_USE_CPU)`),
// not from a backend template argument. In a dual (CPU+GPU) build a single
// object file is compiled with DDLA_USE_CUDA defined and DDLA_USE_CPU not
// defined, so deblasOmatcopy unconditionally resolves to cuBLAS geam there --
// calling it from an `if (Backend == CPU)` branch would hand host
// pointers to cuBLAS. This file breaks that coupling by binding the CPU path
// to the raw cblas_?omatcopy symbols directly (gated on the project-wide
// DDLA_HAS_CPU capability flag, not the per-TU DDLA_USE_CPU vendor flag),
// exactly as gemm.cpp does with sgemm_/dgemm_/cgemm_/zgemm_.
// ---------------------------------------------------------------------------

#if DDLA_HAS_CPU
extern "C" {
void cblas_somatcopy(int order, int trans, int rows, int cols,
                      float alpha, const float* a, int lda,
                      float* b, int ldb);
void cblas_domatcopy(int order, int trans, int rows, int cols,
                      double alpha, const double* a, int lda,
                      double* b, int ldb);
void cblas_comatcopy(int order, int trans, int rows, int cols,
                      const std::complex<float>* alpha, const std::complex<float>* a, int lda,
                      std::complex<float>* b, int ldb);
void cblas_zomatcopy(int order, int trans, int rows, int cols,
                      const std::complex<double>* alpha, const std::complex<double>* a, int lda,
                      std::complex<double>* b, int ldb);
}
#endif

namespace ddla {

#if DDLA_HAS_CPU
namespace {
constexpr int kCblasColMajorLocal  = 102;
constexpr int kCblasNoTransLocal   = 111;
constexpr int kCblasTransLocal     = 112;
constexpr int kCblasConjTransLocal = 113;

// Conjugation is a no-op on real data, so 'C' behaves exactly like 'T'.
inline int omatcopy_trans_real_local(char trans) {
    return (trans == 'N') ? kCblasNoTransLocal : kCblasTransLocal;
}
inline int omatcopy_trans_complex_local(char trans) {
    if (trans == 'N') return kCblasNoTransLocal;
    return (trans == 'C') ? kCblasConjTransLocal : kCblasTransLocal;
}
} // anonymous namespace

inline void cpu_omatcopy(char trans, int rows, int cols,
                         const float& alpha, const float* A, int lda, float* B, int ldb)
{
    cblas_somatcopy(kCblasColMajorLocal, omatcopy_trans_real_local(trans),
                    rows, cols, alpha, A, lda, B, ldb);
}

inline void cpu_omatcopy(char trans, int rows, int cols,
                         const double& alpha, const double* A, int lda, double* B, int ldb)
{
    cblas_domatcopy(kCblasColMajorLocal, omatcopy_trans_real_local(trans),
                    rows, cols, alpha, A, lda, B, ldb);
}

inline void cpu_omatcopy(char trans, int rows, int cols,
                         const std::complex<float>& alpha, const std::complex<float>* A, int lda,
                         std::complex<float>* B, int ldb)
{
    cblas_comatcopy(kCblasColMajorLocal, omatcopy_trans_complex_local(trans),
                    rows, cols, &alpha, A, lda, B, ldb);
}

inline void cpu_omatcopy(char trans, int rows, int cols,
                         const std::complex<double>& alpha, const std::complex<double>* A, int lda,
                         std::complex<double>* B, int ldb)
{
    cblas_zomatcopy(kCblasColMajorLocal, omatcopy_trans_complex_local(trans),
                    rows, cols, &alpha, A, lda, B, ldb);
}
#endif // DDLA_HAS_CPU

inline const char* omatcopy_backend_name(DdlaBackend backend)
{
    return backend == DdlaBackend::CPU ? "CPU" : "GPU";
}

template <DdlaBackend Backend, typename T>
void omatcopy(const DdlaHandle_t& handle, char trans, int rows, int cols,
              const T& alpha, const T* A, int lda, T* B, int ldb)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU omatcopy is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU omatcopy is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("omatcopy: null handle");
    }
    const DdlaBackend actual = ddla_get_backend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("omatcopy: template backend ") + omatcopy_backend_name(Backend) +
            " does not match handle backend " + omatcopy_backend_name(actual));
    }
    if (rows <= 0 || cols <= 0) return;

    if (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        cpu_omatcopy(trans, rows, cols, alpha, A, lda, B, ldb);
#endif
    } else {
#if DDLA_HAS_GPU
        const deblasOperation_t op = trans == 'N' ? DEBLAS_OP_N :
                                     trans == 'T' ? DEBLAS_OP_T : DEBLAS_OP_C;
        BLAS_CHECK(deblasOmatcopy(handle->blasH, op, rows, cols, alpha, A, lda, B, ldb));
#endif
    }
}

// C++11 has no `if constexpr`, and in a dual build both DDLA_HAS_CPU and
// DDLA_HAS_GPU are defined in every TU, so a plain runtime `if` would
// instantiate the GPU branch for CPU-backend templates too -- and
// runtimeMemcpy2DAsync<CPU> takes RuntimeTraits<CPU>::stream_t (int) while
// this TU's h->stream is a device stream. Dispatch on the backend tag so
// only the matching branch is instantiated.
namespace {
template <typename T>
void copy2D_impl(const DdlaHandle_t& handle, T* dst, int dst_ld,
                 const T* src, int src_ld, int rows, int cols,
                 std::integral_constant<DdlaBackend, DdlaBackend::CPU>)
{
#if DDLA_HAS_CPU
    cpu_omatcopy('N', rows, cols, (T)1.0, src, src_ld, dst, dst_ld);
#endif
}

template <typename T>
void copy2D_impl(const DdlaHandle_t& handle, T* dst, int dst_ld,
                 const T* src, int src_ld, int rows, int cols,
                 std::integral_constant<DdlaBackend, DdlaBackend::GPU>)
{
#if DDLA_HAS_GPU
    RUNTIME_CHECK(runtimeMemcpy2DAsync(
        dst, static_cast<std::size_t>(dst_ld) * sizeof(T),
        src, static_cast<std::size_t>(src_ld) * sizeof(T),
        static_cast<std::size_t>(rows) * sizeof(T),
        static_cast<std::size_t>(cols),
        runtimeMemcpyDeviceToDevice,
        handle->stream));
#endif
}
} // anonymous namespace

template <DdlaBackend Backend, typename T>
void copy2D(const DdlaHandle_t& handle, T* dst, int dst_ld,
            const T* src, int src_ld, int rows, int cols)
{
    if (rows <= 0 || cols <= 0) return;

    copy2D_impl(handle, dst, dst_ld, src, src_ld, rows, cols,
                std::integral_constant<DdlaBackend, Backend>());
}

#define INSTANTIATE_COPY2D(BACKEND, TYPE)                                    \
    template void copy2D<BACKEND, TYPE>(                                     \
        const DdlaHandle_t&, TYPE*, int, const TYPE*, int, int, int)

#if DDLA_HAS_CPU
INSTANTIATE_COPY2D(DdlaBackend::CPU, float);
INSTANTIATE_COPY2D(DdlaBackend::CPU, double);
INSTANTIATE_COPY2D(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_COPY2D(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_COPY2D(DdlaBackend::GPU, float);
INSTANTIATE_COPY2D(DdlaBackend::GPU, double);
INSTANTIATE_COPY2D(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_COPY2D(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_COPY2D

#define INSTANTIATE_OMATCOPY(BACKEND, TYPE)                                  \
    template void omatcopy<BACKEND, TYPE>(                                   \
        const DdlaHandle_t&, char, int, int,                                  \
        const TYPE&, const TYPE*, int, TYPE*, int)

#if DDLA_HAS_CPU
INSTANTIATE_OMATCOPY(DdlaBackend::CPU, float);
INSTANTIATE_OMATCOPY(DdlaBackend::CPU, double);
INSTANTIATE_OMATCOPY(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_OMATCOPY(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_OMATCOPY(DdlaBackend::GPU, float);
INSTANTIATE_OMATCOPY(DdlaBackend::GPU, double);
INSTANTIATE_OMATCOPY(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_OMATCOPY(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_OMATCOPY

} // namespace ddla
