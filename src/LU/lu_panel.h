#ifndef DDLA_LU_PANEL_H
#define DDLA_LU_PANEL_H

#include "lu_internal.h"
#include <cstddef>
#include <vector>

namespace ddla::lu_impl {

// Read once per pgetf2 call and agreed across the handle's ranks; never
// called inside the inner-block loop.
bool use_gather(const DdlaHandle_t& handle);

template <typename T>
struct Workspace {
    explicit Workspace(const DdlaHandle_t& h) : handle(h) {}
    ~Workspace();
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    void reserve(bool root, int height, int width, int local_rows, int local_cols);
    DdlaHandle_t handle;
    T *window = nullptr, *packed = nullptr, *rows = nullptr;
    int* device_meta = nullptr;
    std::size_t window_cap = 0, packed_cap = 0, rows_cap = 0, meta_cap = 0;
    std::vector<int> meta;
};

// Entry validation is collective; internal panel calls do no grid validation.
void validate(const DdlaHandle_t&, int m, int n, const void* A, int ia, int ja,
              const int* desc, const int* ipiv, int width = -1);

template <DdlaBackend Backend, typename T>
void factor_inplace(const DdlaHandle_t&, int, int, int, T*, int, int,
                    const int*, int*, int&, int swap_ja);

template <DdlaBackend Backend, typename T>
void panel_gather(const DdlaHandle_t&, int m, int n, int width, T* A, int ia, int ja,
                  const int* desc, int* ipiv, int& info, Workspace<T>&, int swap_ja);

} // namespace ddla::lu_impl

namespace ddla {

// The blocked width factorization over the lu_impl::Workspace buffers. It
// reads DDLA_LU_PANEL once and dispatches each inner subpanel directly to
// lu_impl::panel_gather (gather) or lu_impl::factor_inplace (in place), with
// BLAS-3 trsm/gemm updates of the remaining panel columns in between.
// swap_ja retains the first selected column of the enclosing call so every
// subpanel also pivots the preceding L columns of that region; a plain
// pgetf2 call passes its own ja.
template <DdlaBackend Backend, typename T>
void pgetf2(const DdlaHandle_t&, int, int, int, T*, int, int, const int*, int*, int&,
            lu_impl::Workspace<T>&, int swap_ja);

} // namespace ddla
#endif
