#ifndef DDLA_GEMMT_IMPL_H
#define DDLA_GEMMT_IMPL_H

#include "blas_impl.h"
#include <cstddef>

namespace ddla { namespace blas_impl {

// Updates only the selected triangle; no structure is assumed for partial
// products. The query sizes reusable caller-owned backend memory. The run
// neither allocates nor validates, and leaves the opposite triangle untouched.
// Product calls require the backend workspace returned by the query.
// Empty products and pure triangle scaling require no workspace.
// Backend scratch bytes only; this query performs no allocation.
// Copy full diagonal products from uniformly pitched temporary blocks into
// selected triangles, applying beta only to those output entries. targets is
// a backend pointer table, or null for consecutive diagonal blocks of C.
template <DdlaBackend Backend, typename T>
void gemmt_copy_run(const DdlaHandle_t& h, char uplo, int block_size, int full_blocks,
    int tail_width, const T& beta, const T* temporary, T* C, int ldc, T* const* targets,
    const int* widths = nullptr);

// Hermitian diagonal input uses only its real component, including when its
// imaginary storage has never been initialized.
template <DdlaBackend Backend, typename T>
void real_diagonal(const DdlaHandle_t& h, int n, T* C, int ldc);

}} // namespace ddla::gemmt_impl
#endif
