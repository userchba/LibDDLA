#ifndef DDLA_PVECTOR_REDIST_H
#define DDLA_PVECTOR_REDIST_H

// Public <ddla/ddla.h> first (hipcc default-argument rule, see src/ptran.h).
#include <ddla/ddla.h>

#include <cstddef>

namespace ddla {

// Library-internal components (plan_util.h).  ptranspose_vectors_check, or
// ptranspose_vectors_geometry for a call the library already validated,
// freezes the schedule that the runs take.
namespace spectral_impl {
struct VectorRedistPlan {
    DdlaBackend backend{};
    int n=0,nvc=0,nb=1,ld_src=1,ld_dst=1;
    int nprows=1,npcols=1,myprow=0,mypcol=0,nblks=0,lcm=1,max_round_blocks=0;
    bool accumulate=false,to_rows=false;
};
VectorRedistPlan ptranspose_vectors_geometry(const DdlaHandle_t&,DdlaBackend,int,int,int,int,int,bool,bool);
VectorRedistPlan ptranspose_vectors_check(const DdlaHandle_t&,DdlaBackend,int,int,int,int,int,bool,bool,const void*,const void*);

// Backend bytes of ptranspose_vectors_run for (n, nvc, nb) on this handle's
// grid, either direction: the per-round broadcast buffer of
// ptranspose_vectors_work_elems(handle, n, nvc, nb) entries.  Monotone in n
// and nvc, so the largest call's size serves every smaller one.
template<class T>
void ptranspose_vectors_workspace_size(const DdlaHandle_t& handle,int n,int nvc,int nb,std::size_t* backend_bytes);

// Backend bytes of preduce_add_vectors_run for (n, nvc, nb, ld_src): the
// row-summed panel (num_loc(n, nb, myprow, 0, nprows) x nvc, at least one
// entry), then one region that first holds the compact send pack of padded
// input (ld_src != the summed panel's leading dimension) and then the
// transpose's broadcast buffer (the pack dies after the reduction).
template<class T>
void preduce_add_vectors_workspace_size(const DdlaHandle_t& handle,int n,int nvc,int nb,int ld_src,
    std::size_t* backend_bytes);

// The transpose of the call p describes.  No validation, no allocation.  The
// broadcast buffer is the front of the scratch: a caller-supplied buffer of
// ptranspose_vectors_work_elems entries (backend_bytes = entries * sizeof(T))
// is enough.
template<DdlaBackend B,class T>
void ptranspose_vectors_run(const DdlaHandle_t&,const VectorRedistPlan& p,const T* src,T* dst,
    void* backend_work,std::size_t backend_bytes);
// The reduce-add of the call p describes (checked with accumulate = true,
// to_rows = false).  No validation, no allocation.
template<DdlaBackend B,class T>
void preduce_add_vectors_run(const DdlaHandle_t&,const VectorRedistPlan& p,const T* src,T* dst,
    void* backend_work,std::size_t backend_bytes);
} // namespace spectral_impl

/// Transpose an array of @p nvc vectors of global length @p n from a
/// ROW-distributed layout (block-cyclic with block size @p nb over the
/// nprows process rows; an identical copy is present in every process
/// column) to a COLUMN-distributed layout (block-cyclic with block size
/// @p nb over the npcols process columns, replicated across the process
/// rows).
///
/// Communication: lcm(nprows, npcols) rounds of broadcasts over the column
/// communicator.  In round @p n the active process column mod(n, npcols)
/// receives exactly the global blocks congruent to n (mod lcm), broadcast by
/// their owning process row mod(n, nprows); every other column skips the
/// round.  @p dst is fully overwritten (or augmented, when
/// @p accumulate_dst is true).  Local buffers are packed block-cyclic
/// arrays: element (local row l, vector j) lives at src[l + j*ld_src] with
/// global index indxl2g(l, nb, myprow, 0, nprows) for the row layout and
/// indxl2g(l, nb, mypcol, 0, npcols) for the column layout.
///
/// Set @p to_rows for the reverse (column-to-row) direction; the same
/// packing schedule and scratch bound apply, using row broadcasts.
///
/// Buffers live in backend memory: host for CPU handles, device for GPU
/// handles.  Every rank must call the routine collectively.
///
/// @p aux_work (optional, backend memory): the per-call scratch, of at least
/// ptranspose_vectors_work_elems(handle, n, nvc, nb) elements, passed as
/// @p aux_work_elems.  A caller that transposes many panels allocates one
/// buffer of the largest size and passes it to every call; with nullptr the
/// routine allocates and frees its own scratch.
template <DdlaBackend Backend, typename T>
void ptranspose_vectors(const DdlaHandle_t& handle,
                        const T* src, int ld_src,
                        T* dst, int ld_dst,
                        int n, int nvc, int nb,
                        bool accumulate_dst = false,
                        T* aux_work = nullptr, std::size_t aux_work_elems = 0,
                        bool to_rows = false);

/// Scratch elements ptranspose_vectors needs for (@p n, @p nvc, @p nb) on
/// this handle's grid (0 when there is nothing to transpose).  Monotone in
/// @p n and @p nvc, so the largest call's value covers every smaller one.
std::size_t ptranspose_vectors_work_elems(const DdlaHandle_t& handle,
                                          int n, int nvc, int nb);

/// Adjoint of ptranspose_vectors: the input @p src holds PER-PROCESS-COLUMN
/// partial copies of the vectors in the row-distributed layout (each process
/// column contributes its own partial over the process rows).  The partials
/// are summed across each process row (one row-scope all-reduce) and the
/// summed vectors are transposed into the column-distributed layout, ADDED
/// to @p dst (which the caller zeroes beforehand, or holds values to be
/// augmented).  Every block owned by this rank's process column is
/// augmented; blocks owned by other columns are untouched.
template <DdlaBackend Backend, typename T>
void preduce_add_vectors(const DdlaHandle_t& handle,
                         const T* src, int ld_src,
                         T* dst, int ld_dst,
                         int n, int nvc, int nb);

} // namespace ddla

#endif // DDLA_PVECTOR_REDIST_H
