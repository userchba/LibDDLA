#ifndef DDLA_CHOLESKY_WORKSPACE_H
#define DDLA_CHOLESKY_WORKSPACE_H

// The distributed Cholesky routines split into the three components of plan_util.h: <r>_check (collective
// validation), <r>_workspace_size (backend and host bytes; no allocation, no communication) and <r>_run on the
// caller's scratch (no validation, no allocation).  pposv runs ppotrf, then ppotrs, on one scratch.

#include <ddla/ddla.h>

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <stdexcept>

#include "ddla_desc.h"
#include "ddla_stream.h"
#include "lapack_impl.h"
#include "plan_util.h"
#include "PBLAS/pswap_impl.h"
#include "PBLAS/ptrtrs_impl.h"
#include "PBLAS/pblas_workspace.h"

namespace ddla { namespace cholesky_impl {

// Element counts of the panels of ppotrf (bottom = false) and ppotrf_bottom_right (bottom = true): count[0] the
// diagonal block (nb x nb), count[1] the panel copied out of A after the TRSM by the diagonal block (the row panel
// for uplo 'U', the column panel for 'L'), count[2] the other panel.  The row panel spans the local columns of the
// leading n x n block in ppotrf and its local rows in ppotrf_bottom_right; the column panel the opposite.
inline void panel_counts(const DdlaHandle_t& h, char uplo, int n, const int* d, bool bottom, std::size_t (&count)[3])
{
    const int nb = d[DDLA_MB_];
    const int ml = num_loc(n, nb, h->myprow_, d[DDLA_RSRC_], h->nprows_);
    const int nl = num_loc(n, nb, h->mypcol_, d[DDLA_CSRC_], h->npcols_);
    const std::size_t row = std::size_t(nb) * (bottom ? ml : nl), col = std::size_t(nb) * (bottom ? nl : ml);
    count[0] = std::size_t(nb) * nb;
    count[1] = uplo == 'U' ? row : col;
    count[2] = uplo == 'U' ? col : row;
}

// Length of each pointer list of the batched trailing update: one entry per pair of full local nb x nb tiles.
inline std::size_t batch_capacity(const DdlaHandle_t& h, int n, const int* d)
{
    const int nb = d[DDLA_MB_];
    const std::size_t batches = std::size_t(num_loc(n, nb, h->myprow_, d[DDLA_RSRC_], h->nprows_) / nb)
                              * std::size_t(num_loc(n, nb, h->mypcol_, d[DDLA_CSRC_], h->npcols_) / nb);
    if (batches > std::size_t(std::numeric_limits<int>::max())) throw std::length_error("Cholesky: batch count overflow");
    return batches;
}

// The ?potrf scratch of the diagonal blocks this rank factors: the largest vendor workspace over their orders
// (nothing on a CPU handle).  ppotrf factors a block in place (lda = LLD; with is_head the last block is one order
// smaller), ppotrf_bottom_right a reverse-packed copy on the opposite triangle (lda = the order).
template <class T>
std::size_t potrf_block_bytes(const DdlaHandle_t& h, char uplo, int n, T* A, const int* d, bool head, bool bottom)
{
    const int nb = d[DDLA_MB_];
    std::size_t bytes = 0;
    int previous = 0;
    for (int k = 0; k < n; k += nb) {
        const int width = std::min(nb, n - k), order = (!bottom && head && k + width == n) ? width - 1 : width;
        if (order <= 0 || order == previous || indxg2p(k, nb, d[DDLA_RSRC_], h->nprows_) != h->myprow_
            || indxg2p(k, nb, d[DDLA_CSRC_], h->npcols_) != h->mypcol_) continue;
        std::size_t block = 0;
        lapack_impl::potrf_workspace_size<T>(h, bottom ? (uplo == 'U' ? 'L' : 'U') : uplo, order, A,
                                             bottom ? order : d[DDLA_LLD_], &block);
        bytes = std::max(bytes, block);
        previous = order;
    }
    return bytes;
}

// ===========================================================================
// ppotrf
// ===========================================================================

template <DdlaBackend Backend>
void ppotrf_check(const DdlaHandle_t&, char, int, const void*, int, int, const int*, bool, int);

// The diagonal relay is a no-redistribution fast path only when both grid
// coordinates own the same global block sequence.
inline bool diagonal_relay(const DdlaHandle_t& h, const int* d)
{
    return h->nprows_ == h->npcols_ && d[DDLA_RSRC_] == d[DDLA_CSRC_];
}

template <class T>
void ppotrf_mirror_workspace_size(const DdlaHandle_t& h, char uplo, int n,
                                  const int* d, std::size_t* backend, std::size_t* host)
{
    *backend = *host = 0;
    if (diagonal_relay(h, d)) return;
    const bool lower = uplo == 'L';
    const int width = std::min(n, d[DDLA_MB_]);
    pblas::panel_workspace_size<T>(h, !lower, 'C', lower ? n : width,
        lower ? width : n, d, d, backend, host);
}

// Component 2.  Backend, in carving order: [ the left / right / target pointer lists of the batched update | the
// ?potrf info word | the diagonal block and the first panel (panel_counts), which share one region because the
// diagonal block is not read once that panel is written | the second panel ].  The ?potrf scratch (the rest of the
// buffer) overlays both panels, which are written only after the diagonal block is factored; the two pswap buffers
// of the head relocation (is_head, location != -1, n) overlay everything, since the swaps run first.  Host: the
// left / right / target lists, filled on the host and uploaded. General grids
// append PGEADD panel-redistribution scratch in both domains. n <= 0 needs nothing.
template <class T>
void ppotrf_workspace_size(const DdlaHandle_t& handle, char uplo, int n, T* A, const int* descA,
                           bool is_head, int location, std::size_t* backend_bytes, std::size_t* host_bytes)
{
    *backend_bytes = *host_bytes = 0;
    if (n <= 0) return;
    if (handle->nprocs==1 && !is_head) {
        lapack_impl::potrf_workspace_size<T>(handle,uplo,n,A,descA[DDLA_LLD_],backend_bytes);
        if (handle->backend==DdlaBackend::GPU) *backend_bytes+=detail::padded_bytes<int>(1);
        return;
    }
    std::size_t count[3];
    panel_counts(handle, uplo, n, descA, false, count);
    const std::size_t lists = 3 * detail::padded_bytes<T*>(batch_capacity(handle, n, descA));
    const std::size_t panels = std::max(detail::padded_bytes<T>(count[0]), detail::padded_bytes<T>(count[1]))
                             + detail::padded_bytes<T>(count[2]);
    std::size_t mirror_backend = 0, mirror_host = 0;
    ppotrf_mirror_workspace_size<T>(handle, uplo, n, descA, &mirror_backend, &mirror_host);
    *backend_bytes = lists + detail::padded_bytes<int>(1)
                   + std::max(panels + mirror_backend, potrf_block_bytes<T>(handle, uplo, n, A, descA, is_head, false));
    *host_bytes = lists + mirror_host;
    if (is_head && location != -1 && location != n) {
        std::size_t swap_bytes = 0;
        pswap_impl::pswap_workspace_size<T>(handle, descA, &swap_bytes);
        *backend_bytes = std::max(*backend_bytes, swap_bytes);
    }
}

// Component 3: returns whether the head correction negated the last pivot.  No validation, no allocation.
template <DdlaBackend Backend, class T>
bool ppotrf_run(const DdlaHandle_t& handle, char uplo, int n, T* A, int ia, int ja, const int* descA,
                int& info, bool is_head, int location,
                void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes);

// ===========================================================================
// ppotrf_bottom_right
// ===========================================================================

template <DdlaBackend Backend>
void ppotrf_bottom_right_check(const DdlaHandle_t&, char, int, const void*, const int*);

// Component 2.  Backend, in carving order: [ the left / right / target pointer lists | the ?potrf info word | the
// diagonal block (also the reverse-packed copy that is factored) and the first panel, sharing one region | the
// second panel ].  The ?potrf scratch (the rest of the buffer) overlays the second panel, which is written only
// after the diagonal block is factored.  Host: the left / right / target lists.  n <= 0 needs nothing.
template <class T>
void ppotrf_bottom_right_workspace_size(const DdlaHandle_t& handle, char uplo, int n, T* A, const int* descA,
                                        std::size_t* backend_bytes, std::size_t* host_bytes)
{
    *backend_bytes = *host_bytes = 0;
    if (n <= 0) return;
    if (handle->nprocs==1) {
        lapack_impl::potrf_bottom_right_workspace_size<T>(handle,uplo,n,A,backend_bytes);
        return;
    }
    std::size_t count[3];
    panel_counts(handle, uplo, n, descA, true, count);
    const std::size_t lists = 3 * detail::padded_bytes<T*>(batch_capacity(handle, n, descA));
    *backend_bytes = lists + detail::padded_bytes<int>(1)
                   + std::max(detail::padded_bytes<T>(count[0]), detail::padded_bytes<T>(count[1]))
                   + std::max(detail::padded_bytes<T>(count[2]),
                              potrf_block_bytes<T>(handle, uplo, n, A, descA, false, true));
    *host_bytes = lists;
}

// Component 3.  No validation, no allocation.
template <DdlaBackend Backend, class T>
void ppotrf_bottom_right_run(const DdlaHandle_t& handle, char uplo, int n, T* A, const int* descA, int& info,
                             void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes);

// ===========================================================================
// ppotrs
// ===========================================================================

template <DdlaBackend Backend>
void ppotrs_check(const DdlaHandle_t&, char, char, char, int, int, const void*, const int*, const void*, const int*,
                  int);

// Component 2.  The two triangular solves (op(A) = A and A^H) and the swaps of B that undo the head relocation
// (location != -1, n; before and after the solves) run one after the other on the whole scratch: the largest of
// the ptrtrs scratches of both transposes (backend and host) and the pswap buffers of B's descriptor (backend).
// n == 0 or nrhs == 0 needs nothing.
template <class T>
void ppotrs_workspace_size(const DdlaHandle_t& handle, char side, int n, int nrhs,
                           const int* descA, const int* descB, int location,
                           std::size_t* backend_bytes, std::size_t* host_bytes)
{
    *backend_bytes = *host_bytes = 0;
    if (n == 0 || nrhs == 0) return;
    const bool left = side == 'L';
    for (const char trans : {'N', 'C'}) {
        std::size_t solve_backend = 0, solve_host = 0;
        ptrtrs_impl::ptrtrs_workspace_size<T>(handle, side, trans, left ? n : nrhs, left ? nrhs : n, descA, descB,
                                              &solve_backend, &solve_host);
        *backend_bytes = std::max(*backend_bytes, solve_backend);
        *host_bytes = std::max(*host_bytes, solve_host);
    }
    if (location != -1 && location != n) {
        std::size_t swap_bytes = 0;
        pswap_impl::pswap_workspace_size<T>(handle, descB, &swap_bytes);
        *backend_bytes = std::max(*backend_bytes, swap_bytes);
    }
}

// Component 3.  No validation, no allocation.
template <DdlaBackend Backend, class T>
void ppotrs_run(const DdlaHandle_t& handle, char side, char uplo, char trans, int n, int nrhs,
                T* A, const int* descA, T* B, const int* descB, bool is_nega, int location,
                void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes);

// ===========================================================================
// pposv
// ===========================================================================

template <DdlaBackend Backend>
void pposv_check(const DdlaHandle_t&, char, char, char, int, int, const void*, int, int, const int*, const void*,
                 int, int, const int*, bool, int);

// Component 2: ppotrf and then ppotrs on the same scratch, so the larger of their scratches, kind by kind.
template <class T>
void pposv_workspace_size(const DdlaHandle_t& handle, char side, char uplo, int n, int nrhs, T* A,
                          const int* descA, const int* descB, bool is_head, int location,
                          std::size_t* backend_bytes, std::size_t* host_bytes)
{
    std::size_t solve_backend = 0, solve_host = 0;
    ppotrf_workspace_size<T>(handle, uplo, n, A, descA, is_head, location, backend_bytes, host_bytes);
    ppotrs_workspace_size<T>(handle, side, n, nrhs, descA, descB, location, &solve_backend, &solve_host);
    *backend_bytes = std::max(*backend_bytes, solve_backend);
    *host_bytes = std::max(*host_bytes, solve_host);
}

// Component 3.  No validation, no allocation.
template <DdlaBackend Backend, class T>
void pposv_run(const DdlaHandle_t& handle, char side, char uplo, char trans, int n, int nrhs,
               T* A, int ia, int ja, const int* descA, T* B, int ib, int jb, const int* descB,
               int& info, bool is_head, int location,
               void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes);

}} // namespace ddla::cholesky_impl

#endif // DDLA_CHOLESKY_WORKSPACE_H
