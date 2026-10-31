#ifndef TRANSPORT_BLOCK_H
#define TRANSPORT_BLOCK_H

// Public <ddla/ddla.h> first: owns DdlaBackend,
// default_backend_v, and every public default argument -- hipcc/clang
// rejects redeclarations that introduce defaults later.
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "workspace.h"
#include <complex>
#include <vector>

namespace ddla{

// Unified CPU+GPU panel-transport primitive: extract an m x n panel of the
// (possibly transposed/conjugated) source matrix A, redistribute it via MPI
// (CPU backend) or NCCL/RCCL/host-staged-MPI (GPU backend, via CommTraits in
// src/comm_traits.h), and land the full result in block_A on every rank in
// the relevant row/column/grid group. Backend is templated (defaulting to
// default_backend_v) exactly like gemm<Backend,T>/pgemm<Backend,T> -- the
// same bracket-free call sites (ptrtrs.cpp, the API-grid tests) keep working
// unchanged, resolving Backend from the default in any GPU/dual build.
template <DdlaBackend Backend, typename T>
void transport_block(
    const DdlaHandle_t& handle,
    char sData, char trans,
    int m, int n,
    const T* A, int ia, int ja, const int* array_descA,
    T* block_A
);

// transport_block split into the three components every routine has (see
// ptrtrm_impl.h): library-internal callers plan once for the largest panel
// they will move and call transport_run for every panel, with no argument
// check and no allocation.
namespace transport_impl {

// One rectangular piece of the transposed non-square-grid redistribution.
struct RectBlock {
    int src_rank;
    int dst_rank;
    int count;
    int send_offset;
    int dst_offset;
    int src_loc;
    int len;
};

// What transport_plan allocates.  Only the transposed path on a non-square
// process grid stages anything: `send` packs the blocks this rank sends, and
// the three lists are the schedule of the current call (cleared and refilled
// by transport_run, reserved here).  A context is bound to one handle and
// serves every call whose extents stay within the planned bounds.
template <typename T>
struct TransportCtx {
    vector<T> send;
    std::vector<RectBlock> blocks, send_blocks, recv_blocks;

    explicit TransportCtx(DdlaHandle_t h) : send(h) {}
};

struct TransportOpts {
    char sData;
    char trans;
};

struct TransportWorkspaceSize {
    WorkspaceSize size;
    std::size_t send_count=0, block_count=0;
    std::size_t send_offset=0, blocks_offset=0, send_blocks_offset=0, recv_blocks_offset=0;
    std::size_t tunnel_offset=0, tunnel_bytes=0;
};

// Allocation-free requirements for every offset/owner of panels up to the
// supplied bounds. sData='B' reserves for both row and column panels.
template <typename T>
TransportWorkspaceSize transport_workspace_size(const DdlaHandle_t& handle,
    char sData,char trans,int m_max,int n_max,const int* array_descA);

template <DdlaBackend Backend,typename T>
void transport_run(const DdlaHandle_t& handle,const TransportWorkspaceSize& q,Workspace work,
    char sData,char trans,int m,int n,const T* A,int ia,int ja,const int* array_descA,T* block_A);

// Component 1: validate the arguments exactly like the public entry (a null
// or mismatched handle throws std::runtime_error, everything else
// std::invalid_argument).  Collective: every rank throws if any rank does.
// Callers handle the empty case (m == 0 or n == 0) before the check, as the
// public entry does.
template <DdlaBackend Backend>
TransportOpts transport_check(const DdlaHandle_t& handle, char sData, char trans,
                              int m, int n, const int* array_descA);

// Component 2: size and allocate the staging for panels of at most m_max x
// n_max (the extents transport_run will receive).  sData is 'R', 'C', or 'B'
// when the context serves both panel kinds with the same bounds.  No
// validation, no collectives.
template <typename T>
TransportCtx<T> transport_plan(const DdlaHandle_t& handle, char sData, char trans,
                               int m_max, int n_max, const int* array_descA);

// Grow an existing context to also serve panels of another kind or bound
// (reserve only grows): a caller that moves a transposed A panel and a
// transposed B panel through one context reserves for both.
template <typename T>
void transport_reserve(TransportCtx<T>& ctx, const DdlaHandle_t& handle, char sData, char trans,
                       int m_max, int n_max, const int* array_descA);

// Transitional owning-context execution. Its metadata stays within reserved
// capacity, but legacy tunnel staging may allocate. Use the Workspace overload
// for fully borrowed execution.
template <DdlaBackend Backend, typename T>
void transport_run(const DdlaHandle_t& handle, TransportCtx<T>& ctx,
                   char sData, char trans, int m, int n,
                   const T* A, int ia, int ja, const int* array_descA,
                   T* block_A);

} // namespace transport_impl

}

#endif // TRANSPORT_BLOCK_H
