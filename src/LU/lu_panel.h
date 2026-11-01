#ifndef DDLA_LU_PANEL_H
#define DDLA_LU_PANEL_H

// ---------------------------------------------------------------------------
// The panel layer of the distributed LU: the two methods that factor one inner subpanel (in place on the distributed
// matrix, or gathered to one rank and factored by a local getrf), the scratch they work in, and the argument checks
// the LU entry points share.  Every routine of the family is split into the components of plan_util.h
// (<r>_check / <r>_workspace_size / <r>_run); the declarations of the routines themselves are in pgetrf_impl.h and
// pgetrs_impl.h.  Nothing here validates, reads the environment or allocates inside a run: the size side
// (inplace_workspace_size, gather_workspace_size) counts the bytes, the run side carves them and asserts that its
// calls fit.
// ---------------------------------------------------------------------------

#include "lu_internal.h"
#include "ddla_desc.h"

#include <algorithm>
#include <cstddef>

namespace ddla::lu_impl {

// ---------------------------------------------------------------------------
// The schedule.  Size and run both walk the panel / subpanel sequence of a factorization; the widths below are the
// one definition of that sequence, so the bytes a size function counts are the bytes a run carves.
// ---------------------------------------------------------------------------

// Inner subpanel width: at most 32 columns, at most half a descriptor block.
inline int inner_width(int nb)
{
    return std::min(32, std::max(1, nb / 2));
}

// Width of the inner subpanel that starts k columns into a width-wide panel whose first row is row0 (zero-based).
// The nb row-boundary term keeps every subpanel's diagonal block inside one process row.
inline int subpanel_width(int nb, int row0, int width, int k)
{
    return std::min({inner_width(nb), width - k, nb - (row0 + k) % nb});
}

// Width of pgetrf's panel at step k of an m x n factorization starting at (row0, col0), zero-based: independent
// offsets reach row and column block boundaries at different steps, and the diagonal block must fit both.
inline int panel_width(int nb, int row0, int col0, int m, int n, int k)
{
    return std::min({nb - (row0 + k) % nb, nb - (col0 + k) % nb, m - k, n - k});
}

// Local columns of this rank in the swap range [swap_ja - 1, ja - 1 + n) of a panel call (ja one-based).  The range
// end ja - 1 + n is the same for every subpanel of a call, so this count is fixed per pgetf2 / pgetrf call.
inline int swap_columns(const DdlaHandle_t& h, const int* desc, int ja, int n, int swap_ja)
{
    int nprows, npcols, myprow, mypcol;
    ddlaGetGridDims(h, nprows, npcols);
    ddlaGetGridCoords(h, myprow, mypcol);
    const int nb = desc[DDLA_MB_];
    return num_loc(ja - 1 + n, nb, mypcol, desc[DDLA_CSRC_], npcols)
         - num_loc(swap_ja - 1, nb, mypcol, desc[DDLA_CSRC_], npcols);
}

// ---------------------------------------------------------------------------
// The two panel methods.  Their backend scratch is one region (device memory on GPU handles, host memory on CPU
// handles) that both carve from its start: a gathered subpanel that turns out singular is replayed in place on the
// same region.  Only the gathered method needs host memory.
// ---------------------------------------------------------------------------

// Size of the in-place method for a swap range of swap_count local columns: backend bytes of [ temp | peer ] (the
// two row segments of a pivot swap, max(1, swap_count) entries each; temp also carries the row of U broadcast down
// the panel's process column) and, on a GPU handle, the pivot search's result slot.
template <typename T>
void inplace_workspace_size(const DdlaHandle_t& handle, int swap_count, std::size_t* backend_bytes);

// Size of the gathered method for the subpanel (m, n, width, ia, ja) whose swap range starts at swap_ja.  Backend:
// [ window (the subpanel on the root, m x width) | device_meta (the root's getrf info and pivots, width + 1) | the
// pool ], the pool holding in turn the transfer packing, the two swap row segments and the root's getrf workspace;
// or the in-place method's bytes if they are more (the replay).  Host: [ meta (the pivots broadcast to every rank,
// width + 1) | the root's peer list of the transfers ].  A is what the vendor query of the root's getrf is made with.
template <typename T>
void gather_workspace_size(const DdlaHandle_t& handle, int m, int n, int width, T* A, int ia, int ja,
                           const int* desc, int swap_ja, std::size_t* backend_bytes, std::size_t* host_bytes);

// Run side.  No validation, no environment reads, no allocation.
template <DdlaBackend Backend, typename T>
void factor_inplace(const DdlaHandle_t& handle, int m, int n, int width, T* A, int ia, int ja,
                    const int* desc, int* ipiv, int& info, int swap_ja,
                    void* backend_work, std::size_t backend_bytes);

template <DdlaBackend Backend, typename T>
void panel_gather(const DdlaHandle_t& handle, int m, int n, int width, T* A, int ia, int ja,
                  const int* desc, int* ipiv, int& info, int swap_ja,
                  void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes);

// ---------------------------------------------------------------------------
// Local argument checks.  They throw on this rank only; the callers run them inside detail::agree_check
// (check_agree.h), which turns a failure on any rank into a throw on every rank.
// ---------------------------------------------------------------------------

// The DDLA_LU_PANEL selector: true for 'gather', false for 'inplace' and for anything else (an invalid non-empty
// value is reported once on rank 0).  Local: the callers register the result with Agree::mode so that the ranks
// agree on it.
bool gather_mode(const DdlaHandle_t& handle);

// The region A(ia:ia+m-1, ja:ja+n-1) of a pgetf2 / pgetrf call: descriptor, offsets, dimensions, panel width
// (width < 0: the whole min(m, n) columns; width == -2: always rejected) and local storage.
// The entry check must validate desc with check_desc before calling this helper. Throws
// std::invalid_argument.  require_ipiv = false for callers that own the pivot vector (pgesv).
void check_region(const char* name, const DdlaHandle_t& handle, int m, int n, const void* A, int ia, int ja,
                  const int* desc, const int* ipiv, int width, bool require_ipiv);

// What every solve (pgetrs and pgesv families) relies on once the descriptors are valid: side and trans flags,
// non-negative sizes, the order and the right-hand sides inside the descriptors, square blocks of A and the row
// block size shared by A and B (the conditions of the triangular solves and pivoting a solve runs).  Throws
// std::invalid_argument.
void check_solve(const char* name, char side, char trans, int n, int nrhs, const int* descA, const int* descB);

} // namespace ddla::lu_impl

#endif
