#ifndef DDLA_MATRIX_TRANSFER_H
#define DDLA_MATRIX_TRANSFER_H

#include <ddla/ddla.h>

#include <cstddef>
#include <vector>

namespace ddla::matrix_impl {

// The selected window, with its zero-based global origin (i, j), its size and
// the block-cyclic layout of the distributed matrix.
struct Region {
    int i, j, m, n;
    int mb, nb, rsrc, csrc;
    int nprows, npcols;
};

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

// Gather A(ia:ia+m-1, ja:ja+n-1) into dst on root, or scatter src on root into
// it. The caller owns the packing scratch: these entry points neither allocate
// nor run a grid collective. All members of the selected communication scope
// participate. With group_root = false, scatter leaves grouping the root's
// sends to the caller.
template <DdlaBackend B, typename T>
void gather(const DdlaHandle_t& h, int m, int n, const T* A, int ia, int ja,
            const int* desc, int root, T* dst, int lddst,
            T* packed, std::size_t capacity);

template <DdlaBackend B, typename T>
void scatter(const DdlaHandle_t& h, int m, int n, T* A, int ia, int ja,
             const int* desc, int root, const T* src, int ldsrc,
             T* packed, std::size_t capacity, bool group_root = true);

// The same, with the root's list of peer blocks in a caller-owned vector (the
// versions above build one per call).  A caller that moves many windows
// reserves `blocks` for the handle's rank count once (blocks.reserve(h->nprocs)),
// and then neither entry point allocates.
template <DdlaBackend B, typename T>
void gather(const DdlaHandle_t& h, int m, int n, const T* A, int ia, int ja,
            const int* desc, int root, T* dst, int lddst,
            T* packed, std::size_t capacity, std::vector<Block<T>>& blocks);

template <DdlaBackend B, typename T>
void scatter(const DdlaHandle_t& h, int m, int n, T* A, int ia, int ja,
             const int* desc, int root, const T* src, int ldsrc,
             T* packed, std::size_t capacity, std::vector<Block<T>>& blocks,
             bool group_root = true);

// GPU copy between count blocks and their entries of the window
// (matrix_transfer_kernels.cpp), one launch per batch of blocks. scatter
// copies window -> blocks. The blocks must not overlap each other or the window.
template <typename T>
void map_gpu(const DdlaHandle_t& h, bool scatter, const Block<T>* blocks, int count,
             const Region& g, T* window, int ldw);

} // namespace ddla::matrix_impl

#endif
