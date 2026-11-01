#ifndef DDLA_POTRF_BOTTOM_RIGHT_INTERNAL_H
#define DDLA_POTRF_BOTTOM_RIGHT_INTERNAL_H

#include <ddla/ddla_handle_t.h>

#include <cstddef>

#include "lapack_impl.h"

namespace ddla::detail {

// Factor one diagonal block in backend memory from the bottom-right. With
// uplo='U', compute A = U * U^H; with uplo='L', compute A = L^H * L. The
// selected triangle is reverse-packed into work, factored with the opposite
// standard POTRF triangle, and mapped back. work holds at least n*n elements.
// info_buffer and work are caller-owned and may be reused across diagonal blocks.
//
// This form validates its arguments (std::runtime_error: lapack_impl::
// potrf_bottom_right_block_check), allocates the ?potrf scratch of the block
// itself, then runs the overload below.
template <DdlaBackend Backend, typename T>
void potrf_bottom_right_block(
    char uplo, int n, T* A, int lda,
    int global_offset,
    T* work, int* info_buffer, int& info,
    const DdlaHandle_t& handle
);

// The same with the ?potrf scratch supplied: potrf_bytes of backend memory, at
// least what lapack_impl::potrf_workspace_size<T>(handle, <the opposite
// triangle: 'L' for uplo 'U', 'U' for uplo 'L'>, n, work, n, &bytes) reports
// for every block order that will be factored.  No validation, no allocation:
// a panel loop sizes once and calls this for every diagonal block.
template <DdlaBackend Backend, typename T>
void potrf_bottom_right_block(
    char uplo, int n, T* A, int lda,
    int global_offset,
    T* work, int* info_buffer, int& info,
    const DdlaHandle_t& handle,
    void* potrf_work, std::size_t potrf_bytes
);

} // namespace ddla::detail

#endif // DDLA_POTRF_BOTTOM_RIGHT_INTERNAL_H
