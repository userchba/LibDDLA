#ifndef GEAM_H
#define GEAM_H

#include "ddla_connector.h"
#include "ddla_handle_t.h"

namespace ddla{

inline deblasStatus_t deblasGeam(
    deblasHandle_t handle, deblasOperation_t transA, deblasOperation_t transB,
    int m, int n,
    const float& alpha,
    const float* A, int lda,
    const float& beta,
    const float* B, int ldb,
    float* C, int ldc
    )
{
#if defined(DDLA_USE_CUDA)
    return cublasSgeam(
        handle, transA, transB,
        m, n,
        &alpha,
        A, lda,
        &beta,
        B, ldb,
        C, ldc
    );
#elif defined(DDLA_USE_HIP)
    return hipblasSgeam(
        handle, transA, transB,
        m, n,
        &alpha,
        A, lda,
        &beta,
        B, ldb,
        C, ldc
    );
#elif defined(DDLA_USE_CPU)
    (void)handle;
    bool is_na = (transA == 'N' || transA == 'n');
    bool is_nb = (transB == 'N' || transB == 'n');
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            float a_val = is_na ? A[i + j*lda] : A[j + i*lda];
            float b_val = is_nb ? B[i + j*ldb] : B[j + i*ldb];
            C[i + j*ldc] = alpha * a_val + beta * b_val;
        }
    }
    return DEBLAS_STATUS_SUCCESS;
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasGeam(
    deblasHandle_t handle, deblasOperation_t transA, deblasOperation_t transB,
    int m, int n,
    const double& alpha,
    const double* A, int lda,
    const double& beta,
    const double* B, int ldb,
    double* C, int ldc
    )
{
#if defined(DDLA_USE_CUDA)
    return cublasDgeam(
        handle, transA, transB,
        m, n,
        &alpha,
        A, lda,
        &beta,
        B, ldb,
        C, ldc
    );
#elif defined(DDLA_USE_HIP)
    return hipblasDgeam(
        handle, transA, transB,
        m, n,
        &alpha,
        A, lda,
        &beta,
        B, ldb,
        C, ldc
    );
#elif defined(DDLA_USE_CPU)
    (void)handle;
    bool is_na = (transA == 'N' || transA == 'n');
    bool is_nb = (transB == 'N' || transB == 'n');
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            double a_val = is_na ? A[i + j*lda] : A[j + i*lda];
            double b_val = is_nb ? B[i + j*ldb] : B[j + i*ldb];
            C[i + j*ldc] = alpha * a_val + beta * b_val;
        }
    }
    return DEBLAS_STATUS_SUCCESS;
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasGeam(
    deblasHandle_t handle, deblasOperation_t transA, deblasOperation_t transB,
    int m, int n,
    const std::complex<float>& alpha,
    const std::complex<float>* A, int lda,
    const std::complex<float>& beta,
    const std::complex<float>* B, int ldb,
    std::complex<float>* C, int ldc
    )
{
    #if defined(DDLA_USE_CUDA)
    return cublasCgeam(
        handle, transA, transB,
        m, n,
        (cuFloatComplex*)&alpha,
        (cuFloatComplex*)A, lda,
        (cuFloatComplex*)&beta,
        (cuFloatComplex*)B, ldb,
        (cuFloatComplex*)C, ldc
    );
    #elif defined(DDLA_USE_HIP)
    return hipblasCgeam(
        handle, transA, transB,
        m, n,
        (hipblasComplex*)&alpha,
        (hipblasComplex*)A, lda,
        (hipblasComplex*)&beta,
        (hipblasComplex*)B, ldb,
        (hipblasComplex*)C, ldc
    );
    #elif defined(DDLA_USE_CPU)
    (void)handle;
    bool is_na = (transA == 'N' || transA == 'n');
    bool is_ca = (transA == 'C' || transA == 'c');
    bool is_nb = (transB == 'N' || transB == 'n');
    bool is_cb = (transB == 'C' || transB == 'c');
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            std::complex<float> a_val = is_na ? A[i + j*lda] : A[j + i*lda];
            if (!is_na && is_ca) a_val = std::conj(a_val);
            std::complex<float> b_val = is_nb ? B[i + j*ldb] : B[j + i*ldb];
            if (!is_nb && is_cb) b_val = std::conj(b_val);
            C[i + j*ldc] = alpha * a_val + beta * b_val;
        }
    }
    return DEBLAS_STATUS_SUCCESS;
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif
}

inline deblasStatus_t deblasGeam(
    deblasHandle_t handle, deblasOperation_t transA, deblasOperation_t transB,
    int m, int n,
    const std::complex<double>& alpha,
    const std::complex<double>* A, int lda,
    const std::complex<double>& beta,
    const std::complex<double>* B, int ldb,
    std::complex<double>* C, int ldc
    )
{
#if defined(DDLA_USE_CUDA)
    return cublasZgeam(
        handle, transA, transB,
        m, n,
        (cuDoubleComplex*)&alpha,
        (cuDoubleComplex*)A, lda,
        (cuDoubleComplex*)&beta,
        (cuDoubleComplex*)B, ldb,
        (cuDoubleComplex*)C, ldc
    );
#elif defined(DDLA_USE_HIP)
    return hipblasZgeam(
        handle, transA, transB,
        m, n,
        (hipblasDoubleComplex*)&alpha,
        (hipblasDoubleComplex*)A, lda,
        (hipblasDoubleComplex*)&beta,
        (hipblasDoubleComplex*)B, ldb,
        (hipblasDoubleComplex*)C, ldc
    );
#elif defined(DDLA_USE_CPU)
    (void)handle;
    bool is_na = (transA == 'N' || transA == 'n');
    bool is_ca = (transA == 'C' || transA == 'c');
    bool is_nb = (transB == 'N' || transB == 'n');
    bool is_cb = (transB == 'C' || transB == 'c');
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            std::complex<double> a_val = is_na ? A[i + j*lda] : A[j + i*lda];
            if (!is_na && is_ca) a_val = std::conj(a_val);
            std::complex<double> b_val = is_nb ? B[i + j*ldb] : B[j + i*ldb];
            if (!is_nb && is_cb) b_val = std::conj(b_val);
            C[i + j*ldc] = alpha * a_val + beta * b_val;
        }
    }
    return DEBLAS_STATUS_SUCCESS;
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

// ---------------------------------------------------------------------------
// deblasOmatcopy: out-of-place scaled transpose-copy, C := alpha * op(A).
//
// This is the single-operand special case of geam that transport_block.cpp's
// packing code actually needs -- it called deblasGeam with beta=0 and B
// aliased to A only to get a vendor-portable transpose+copy primitive (no
// vendor BLAS exposes a standalone "scaled transpose copy" call). On CPU
// this now maps directly onto OpenBLAS's cblas_?omatcopy extension instead
// of a manual double loop; cuBLAS/hipBLAS have no separate single-operand
// primitive, so the CUDA/HIP branches forward to the existing deblasGeam
// idiom unchanged (beta=0, B=A).
//
// `rows`/`cols` describe A's shape *before* the transpose in `trans` is
// applied (matching cblas_?omatcopy's convention, and MKL's mkl_?omatcopy);
// the output C is cols x rows if `trans` is DEBLAS_OP_T/DEBLAS_OP_C, else
// rows x cols.
// ---------------------------------------------------------------------------
#if defined(DDLA_USE_CPU)
extern "C" {
// Raw cblas_?omatcopy symbols, declared directly rather than including
// <cblas.h> -- matches this project's existing convention in src/gemm.cpp
// of declaring vendor BLAS symbols by hand instead of depending on a vendor
// header. CBLAS_ORDER/CBLAS_TRANSPOSE are plain enums (CblasColMajor=102,
// CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113); passed as int here
// for the same header-free reason.
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

namespace detail {
constexpr int kCblasColMajor = 102;
constexpr int kCblasNoTrans = 111;
constexpr int kCblasTrans = 112;
constexpr int kCblasConjTrans = 113;

// Conjugation is a no-op on real data, so 'C' behaves exactly like 'T'.
inline int omatcopy_trans_real(deblasOperation_t trans) {
    return (trans == 'N' || trans == 'n') ? kCblasNoTrans : kCblasTrans;
}
inline int omatcopy_trans_complex(deblasOperation_t trans) {
    if (trans == 'N' || trans == 'n') return kCblasNoTrans;
    return (trans == 'C' || trans == 'c') ? kCblasConjTrans : kCblasTrans;
}
} // namespace detail
#endif // DDLA_USE_CPU

inline deblasStatus_t deblasOmatcopy(
    deblasHandle_t handle, deblasOperation_t trans, int rows, int cols,
    const float& alpha,
    const float* A, int lda,
    float* B, int ldb
    )
{
#if defined(DDLA_USE_CUDA)
    const float zero = 0.0f;
    return cublasSgeam(
        handle, trans, trans,
        (trans == DEBLAS_OP_N) ? rows : cols, (trans == DEBLAS_OP_N) ? cols : rows,
        &alpha, A, lda, &zero, A, lda, B, ldb
    );
#elif defined(DDLA_USE_HIP)
    const float zero = 0.0f;
    return hipblasSgeam(
        handle, trans, trans,
        (trans == DEBLAS_OP_N) ? rows : cols, (trans == DEBLAS_OP_N) ? cols : rows,
        &alpha, A, lda, &zero, A, lda, B, ldb
    );
#elif defined(DDLA_USE_CPU)
    (void)handle;
    cblas_somatcopy(detail::kCblasColMajor, detail::omatcopy_trans_real(trans),
                     rows, cols, alpha, A, lda, B, ldb);
    return DEBLAS_STATUS_SUCCESS;
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasOmatcopy(
    deblasHandle_t handle, deblasOperation_t trans, int rows, int cols,
    const double& alpha,
    const double* A, int lda,
    double* B, int ldb
    )
{
#if defined(DDLA_USE_CUDA)
    const double zero = 0.0;
    return cublasDgeam(
        handle, trans, trans,
        (trans == DEBLAS_OP_N) ? rows : cols, (trans == DEBLAS_OP_N) ? cols : rows,
        &alpha, A, lda, &zero, A, lda, B, ldb
    );
#elif defined(DDLA_USE_HIP)
    const double zero = 0.0;
    return hipblasDgeam(
        handle, trans, trans,
        (trans == DEBLAS_OP_N) ? rows : cols, (trans == DEBLAS_OP_N) ? cols : rows,
        &alpha, A, lda, &zero, A, lda, B, ldb
    );
#elif defined(DDLA_USE_CPU)
    (void)handle;
    cblas_domatcopy(detail::kCblasColMajor, detail::omatcopy_trans_real(trans),
                     rows, cols, alpha, A, lda, B, ldb);
    return DEBLAS_STATUS_SUCCESS;
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasOmatcopy(
    deblasHandle_t handle, deblasOperation_t trans, int rows, int cols,
    const std::complex<float>& alpha,
    const std::complex<float>* A, int lda,
    std::complex<float>* B, int ldb
    )
{
#if defined(DDLA_USE_CUDA)
    const std::complex<float> zero(0.0f, 0.0f);
    return cublasCgeam(
        handle, trans, trans,
        (trans == DEBLAS_OP_N) ? rows : cols, (trans == DEBLAS_OP_N) ? cols : rows,
        (cuFloatComplex*)&alpha, (cuFloatComplex*)A, lda,
        (cuFloatComplex*)&zero, (cuFloatComplex*)A, lda,
        (cuFloatComplex*)B, ldb
    );
#elif defined(DDLA_USE_HIP)
    const std::complex<float> zero(0.0f, 0.0f);
    return hipblasCgeam(
        handle, trans, trans,
        (trans == DEBLAS_OP_N) ? rows : cols, (trans == DEBLAS_OP_N) ? cols : rows,
        (hipblasComplex*)&alpha, (hipblasComplex*)A, lda,
        (hipblasComplex*)&zero, (hipblasComplex*)A, lda,
        (hipblasComplex*)B, ldb
    );
#elif defined(DDLA_USE_CPU)
    (void)handle;
    cblas_comatcopy(detail::kCblasColMajor, detail::omatcopy_trans_complex(trans),
                     rows, cols, &alpha, A, lda, B, ldb);
    return DEBLAS_STATUS_SUCCESS;
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline deblasStatus_t deblasOmatcopy(
    deblasHandle_t handle, deblasOperation_t trans, int rows, int cols,
    const std::complex<double>& alpha,
    const std::complex<double>* A, int lda,
    std::complex<double>* B, int ldb
    )
{
#if defined(DDLA_USE_CUDA)
    const std::complex<double> zero(0.0, 0.0);
    return cublasZgeam(
        handle, trans, trans,
        (trans == DEBLAS_OP_N) ? rows : cols, (trans == DEBLAS_OP_N) ? cols : rows,
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)A, lda,
        (cuDoubleComplex*)&zero, (cuDoubleComplex*)A, lda,
        (cuDoubleComplex*)B, ldb
    );
#elif defined(DDLA_USE_HIP)
    const std::complex<double> zero(0.0, 0.0);
    return hipblasZgeam(
        handle, trans, trans,
        (trans == DEBLAS_OP_N) ? rows : cols, (trans == DEBLAS_OP_N) ? cols : rows,
        (hipblasDoubleComplex*)&alpha, (hipblasDoubleComplex*)A, lda,
        (hipblasDoubleComplex*)&zero, (hipblasDoubleComplex*)A, lda,
        (hipblasDoubleComplex*)B, ldb
    );
#elif defined(DDLA_USE_CPU)
    (void)handle;
    cblas_zomatcopy(detail::kCblasColMajor, detail::omatcopy_trans_complex(trans),
                     rows, cols, &alpha, A, lda, B, ldb);
    return DEBLAS_STATUS_SUCCESS;
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

/**
 * @brief Backend-neutral out-of-place scaled transpose-copy: B := alpha * op(A).
 *
 * The DdlaBackend-templated counterpart of deblasOmatcopy above, in the same
 * shape as ddla::gemm / ddla::scal. Prefer this over calling deblasOmatcopy
 * directly from any code compiled once for both backends: the raw deblas*
 * wrappers pick their implementation from the translation unit's vendor
 * macros, so in a dual build deblasOmatcopy always resolves to cuBLAS geam
 * even on a CPU code path. This overload resolves on `Backend` instead.
 *
 * `rows`/`cols` describe A's shape *before* the transpose in `trans`.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void omatcopy(const DdlaHandle_t& handle, char trans, int rows, int cols,
              const T& alpha, const T* A, int lda, T* B, int ldb);

/**
 * @brief Backend-neutral strided 2D block copy: dst(rows x cols) := src(rows x cols).
 *
 * The non-transposing companion to omatcopy, and the unified replacement for
 * the hand-rolled `if (CPU) { memcpy loop } else {
 * runtimeMemcpy2DAsync }` pairs that used to sit in transport_block.cpp.
 *
 * CPU dispatches to cblas_?omatcopy with DEBLAS_OP_N; GPU dispatches to the
 * device 2D memcpy (runtimeMemcpy2DAsync, device-to-device) rather than to a
 * geam kernel, because a pure strided copy is bandwidth-bound and a memcpy
 * beats a scaling kernel for it.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void copy2D(const DdlaHandle_t& handle, T* dst, int dst_ld,
            const T* src, int src_ld, int rows, int cols);

}
#endif // GEAM_H
