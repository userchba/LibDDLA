#ifndef DDLA_PGETF2_KERNELS_H
#define DDLA_PGETF2_KERNELS_H

#include <cstddef>

#include "ddla_connector.h"

namespace ddla::detail {

constexpr int kLuPivotBatch = 32;
// Copy a bounded host pivot batch through kernel arguments; no host pointer is retained.
void pgetf2_store_pivots(const DdlaHandle_t& handle, int count, const int* host_pivots, int* pivots);
// Backend-to-backend prefix copy with conversion to descriptor-global row indices.
void pgetf2_copy_pivots(const DdlaHandle_t& handle, int count, const int* source, int* destination, int offset);

template <typename T>
std::size_t pgetf2_pivot_workspace_size();

// Returns a zero-based index relative to d_column. The call synchronizes
// stream once because the MPI pivot reduction consumes the host result.
template <typename T>
void pgetf2_find_local_pivot(const DdlaHandle_t& handle, const T* d_column, int length,
                             void* d_workspace,
                             double& metric, int& local_index, T& value);

template <typename T>
void pgetf2_scale_update(const DdlaHandle_t& handle, int length_row, int length_col,
                         const T& inverse_pivot,
                         T* d_column, const T* d_pivot_row,
                         T* d_trailing, int lld);

} // namespace ddla::detail

#endif // DDLA_PGETF2_KERNELS_H
