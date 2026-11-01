#ifndef TRANSPORT_BLOCK_H
#define TRANSPORT_BLOCK_H

// Public <ddla/ddla.h> first: owns DdlaBackend,
// default_backend_v, and every public default argument -- hipcc/clang
// rejects redeclarations that introduce defaults later.
#include <ddla/ddla.h>
#include <complex>
#include <cstddef>

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

// transport_block split into the three components every routine has
// (plan_util.h): library-internal callers size the scratch once for the
// largest panel they will move and call transport_run for every panel, with
// no argument check and no allocation.
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

struct TransportOpts {
    char sData;
    char trans;
};

// Component 1: validate the arguments exactly like the public entry (a null
// or mismatched handle throws std::runtime_error, everything else
// std::invalid_argument).  Collective: every rank throws if any rank does.
// Callers handle the empty case (m == 0 or n == 0) before the check, as the
// public entry does.
template <DdlaBackend Backend>
TransportOpts transport_check(const DdlaHandle_t& handle, char sData, char trans,
                              int m, int n, const int* array_descA);

// Destination layout of a transposed panel.  A negative coordinate keeps the
// public primitive's crossed source ('R' panels land with CSRC on the process
// rows, 'C' panels with RSRC on the process columns); a triangular consumer
// can request its original row/column distribution instead, without changing
// the descriptor the source is read with.
struct TargetSources { int row = -1, col = -1; };

// Component 2: the scratch of panels of at most m_max x n_max (the extents
// transport_run will receive), wherever they start.  Only the transposed path
// on a non-square process grid stages anything: backend bytes of the send
// staging, host bytes of the schedule (every block that moves, the ones this
// rank sends, the ones it receives); so does a square grid whose target
// differs from the crossed source.  sData is 'R', 'C', or 'B' for scratch
// that serves both panel kinds with the same bounds.  No validation, no
// collectives.
template <typename T>
void transport_workspace_size(const DdlaHandle_t& handle, char sData, char trans,
                              int m_max, int n_max, const int* array_descA,
                              std::size_t* backend_bytes, std::size_t* host_bytes,
                              TargetSources target = {});

// Component 3: the transport.  No validation, no allocation.
template <DdlaBackend Backend, typename T>
void transport_run(const DdlaHandle_t& handle, char sData, char trans, int m, int n,
                   const T* A, int ia, int ja, const int* array_descA, T* block_A,
                   void* backend_work, std::size_t backend_bytes,
                   void* host_work, std::size_t host_bytes,
                   TargetSources target = {});

} // namespace transport_impl

}

#endif // TRANSPORT_BLOCK_H
