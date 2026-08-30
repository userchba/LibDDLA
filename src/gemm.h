#ifndef GEMM_H
#define GEMM_H

// Public <ddla/ddla.h> must be included before any redeclaration: it
// owns the public default arguments (Backend = default_backend_v), and
// hipcc/clang rejects redeclarations that introduce defaults later.
#include <ddla/ddla.h>
#include "ddla_connector.h"
#include <ddla/ddla_handle_t.h>

namespace ddla{

/**
 * @brief The single unified local GEMM entry.
 *
 * Covers CPU (Fortran BLAS), and GPU (cuBLAS/hipBLAS) in one backend-templated
 * interface. CPU specializations consume host pointers; GPU specializations
 * consume device pointers.
 */
template <DdlaBackend Backend, typename T>
void gemm(
    const DdlaHandle_t& handle,
    char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const T& beta,
    T* C, int ldc);

} // namespace ddla

#endif // GEMM_H
