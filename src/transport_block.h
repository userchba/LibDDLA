#ifndef TRANSPORT_BLOCK_H
#define TRANSPORT_BLOCK_H

// Public <ddla/ddla.h> first: owns DdlaDesc, DdlaBackend,
// default_backend_v, and every public default argument -- hipcc/clang
// rejects redeclarations that introduce defaults later.
#include <ddla/ddla.h>
#include <complex>

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
    const char& sData, const char& trans,
    const int& m, const int& n,
    const T* d_A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    T* d_block_A
);

}

#endif // TRANSPORT_BLOCK_H
