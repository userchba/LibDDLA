#ifndef DDLA_ROW_SWAP_H
#define DDLA_ROW_SWAP_H

// Public <ddla/ddla.h> first: owns DdlaBackend, default_backend_v, and every
// public default argument -- hipcc/clang rejects redeclarations that
// introduce defaults later.
#include <ddla/ddla.h>

namespace ddla {
namespace detail {

// Swap two row segments that live under one process-grid row distribution.
// row_a and row_b are GLOBAL 0-based row indices; col_a and col_b are LOCAL
// column starts into A and B with `count` elements each at the descriptor's
// leading dimension. A and B may alias as the same matrix (the pivot case:
// identical pointer, descriptor, row and column short-circuits to a no-op)
// or be two matrices that share the row owner set of one process column, so
// col_a/col_b/count agree on both sides of a cross-owner exchange.
// A same-owner pair swaps locally with one strided BLAS swap; otherwise the
// two owning process rows exchange through the caller's send and recv
// buffers (count elements each, distinct). Ranks owning neither row do
// nothing; no collective is called.
template <DdlaBackend Backend, typename T>
void swap_rows(const DdlaHandle_t& h,
               T* A, const int* descA, int row_a, int col_a,
               T* B, const int* descB, int row_b, int col_b,
               int count, T* send, T* recv);

} // namespace detail
} // namespace ddla

#endif // DDLA_ROW_SWAP_H
