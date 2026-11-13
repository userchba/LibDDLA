#ifndef DDLA_MATRIX_TRANSFER_H
#define DDLA_MATRIX_TRANSFER_H

#include <ddla/ddla.h>

#include <cstddef>

namespace ddla::matrix_impl {

// The part of the window one process owns: local rows row0 .. row0+rows-1 and
// local columns col0 .. col0+cols-1 of process (prow, pcol).
struct Peer {
    int row0, col0, rows, cols;
    int prow, pcol;
};

// A peer's part stored column-major at data with leading dimension ld.
template <typename T>
struct Block {
    T* data;
    int ld;
    Peer p;
};

template <DdlaBackend B, typename T>
void gather_check(const DdlaHandle_t& h, int m, int n, const T* A, int ia, int ja,
    const int* desc, int root, T* dst, int lddst);
template <DdlaBackend B, typename T>
void scatter_check(const DdlaHandle_t& h, int m, int n, T* A, int ia, int ja,
    const int* desc, int root, const T* src, int ldsrc);

// The scratch of gather_run / scatter_run for one window, on this rank.
// Backend: the packing buffer (its own block on a participant, the rest of the
// window on the root).  Host: the root's list of peer blocks (one Block per
// other participant).  Index math only: no communication, no data access.  A
// caller that moves many windows through one scratch sizes it for all of them:
// m * n entries and h->nprocs - 1 blocks bound every m x n window.
template <typename T>
void transfer_workspace_size(const DdlaHandle_t& h, int m, int n, int ia, int ja,
    const int* desc, int root, std::size_t* backend_bytes, std::size_t* host_bytes);

// Gather A(ia:ia+m-1, ja:ja+n-1) into dst on root, or scatter src on root into
// it, on the caller's scratch: no validation, no allocation, no grid
// collective.  All members of the selected communication scope participate.
// With group_root = false, scatter leaves grouping the root's sends to the
// caller.
template <DdlaBackend B, typename T>
void gather_run(const DdlaHandle_t& h, int m, int n, const T* A, int ia, int ja,
    const int* desc, int root, T* dst, int lddst,
    void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes);
template <DdlaBackend B, typename T>
void scatter_run(const DdlaHandle_t& h, int m, int n, T* A, int ia, int ja,
    const int* desc, int root, const T* src, int ldsrc, bool group_root,
    void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes);

// Local copy between distributed blocks and their global-order window.
// Uses the same mapping as gather/scatter; no communication or extra packing workspace.
// Blocks describe disjoint window entries and have separate storage, which
// must not overlap the window. CPU copies may use the handle's host threads.
template <DdlaBackend B, typename T>
void map_blocks(const DdlaHandle_t& h, bool scatter, const Block<T>* blocks, int count,
                const int* desc, int ia, int ja, int m, int n, T* window, int ldw);

// Convert a real contiguous RowShare(k,P) slab to a block-cyclic output.
// The k-by-k factor may be transposed, then clipped to the requested m-by-n
// block; logical extensions beyond the factor are zeroed.
// Source and active destination spans must be separate. Parent/LLD padding
// is preserved. Uses two reusable bounded panels, never a replicated factor.
// Collective over the grid, with local failures agreed before communication.
// scatter_rows is the one-shot entry: check, size, allocate, run.
//
// The scratch of scatter_rows_run: backend bytes of the converted panel, then
// the scatter's packing of one panel (each panel_rows x k entries, panel_rows
// = min(512, 32 MiB / (k entries), the largest share)); host bytes of the
// scatter's peer list (nprocs - 1 blocks).  One rank or k == 0 needs nothing.
template <typename T>
void scatter_rows_workspace_size(const DdlaHandle_t& h, int k,
    std::size_t* backend_bytes, std::size_t* host_bytes);
template <DdlaBackend B, typename T>
void scatter_rows_check(const DdlaHandle_t& h, int k,
    const typename real_type<T>::type* src, int lds, bool transpose,
    int m, int n, T* dst, const int* desc);
template <DdlaBackend B, typename T>
void scatter_rows_run(const DdlaHandle_t& h, int k,
    const typename real_type<T>::type* src, int lds, bool transpose,
    int m, int n, T* dst, const int* desc,
    void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes);

template <DdlaBackend B, typename T>
void scatter_rows(const DdlaHandle_t& h, int k,
                  const typename real_type<T>::type* src, int lds,
                  bool transpose, int m, int n, T* dst, const int* desc);

// Local real copy or real-to-complex conversion on either backend. rows and
// cols describe the output rectangle; transpose swaps the source dimensions.
// A null source zeroes the rectangle. Leading-dimension padding is preserved.
// Source and destination must be separate backend-memory spans with valid
// dimensions and pitches. No allocation, synchronization or communication.
template <DdlaBackend B, typename T>
void convert_rows(const DdlaHandle_t& h, const typename real_type<T>::type* src,
                  int lds, T* dst, int ldd, int rows, int cols, bool transpose);

} // namespace ddla::matrix_impl

#endif
