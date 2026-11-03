#ifndef DDLA_GEMMT_IMPL_H
#define DDLA_GEMMT_IMPL_H

#include <ddla/ddla.h>
#include <cstddef>

namespace ddla { namespace blas_impl {

// Updates only the selected triangle; no structure is assumed for partial
// products. The query sizes reusable caller-owned backend memory. The run
// neither allocates nor validates, and leaves the opposite triangle untouched.
// Product calls require the backend workspace returned by the query.
// Empty products and pure triangle scaling require no workspace.
// Backend scratch bytes only; this query performs no allocation.
template <typename T>
void gemmt_workspace_size(const DdlaHandle_t& h, char transa, char transb,
    int n, int k, std::size_t* backend_bytes);
template <DdlaBackend Backend, typename T>
void gemmt_run(const DdlaHandle_t& h,
    char uplo, char transa, char transb, int n, int k,
    const T& alpha, const T* A, int lda, const T* B, int ldb,
    const T& beta, T* C, int ldc, void* backend_work, std::size_t backend_bytes);

// Hermitian diagonal input uses only its real component, including when its
// imaginary storage has never been initialized.
template <DdlaBackend Backend, typename T>
void real_diagonal(const DdlaHandle_t& h, int n, T* C, int ldc);

}} // namespace ddla::gemmt_impl
#endif
