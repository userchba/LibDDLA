#ifndef DDLA_PTRMM_IMPL_H
#define DDLA_PTRMM_IMPL_H

#include <ddla/ddla.h>

#include <cstddef>

namespace ddla { namespace ptrmm_impl {

// ptrmm split into the three components of plan_util.h: check, workspace
// size, run.

// The normalized flags, and whether there is nothing to do (m == 0 or n == 0).

// Component 1: the checks of the public entry.  A null or mismatched handle
// throws std::runtime_error, everything else std::invalid_argument.
// Collective: every rank throws if any rank does.
template <DdlaBackend Backend>
bool ptrmm_check(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag,
                      int m, int n, const int* descA, const int* descB);

// Component 2: the scratch for B (m x n) with the normalized side and trans
// of ptrmm_check.  Backend: the staged diagonal block, a block row/column of
// the original B, the off-diagonal panel of A and the buffer the panel is
// exchanged or transposed into, then the scratch of pgeadd for a transposed
// panel; host: that pgeadd's (direct panel broadcasts need no staging here,
// every transport is untransposed).  Nothing for alpha == 0.  No validation,
// no collectives.
template <typename T>
void ptrmm_workspace_size(const DdlaHandle_t& handle, char side, char trans, int m, int n,
                          bool alpha_is_zero, const int* descA, const int* descB,
                          std::size_t* backend_bytes, std::size_t* host_bytes);

// Component 3: B := alpha*op(A)*B or alpha*B*op(A) with the normalized flags.
// No validation, no allocation.
template <DdlaBackend Backend, typename T>
void ptrmm_run(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag,
               int m, int n, const T& alpha, const T* A, const int* descA, T* B, const int* descB,
               void* backend_work, std::size_t backend_bytes,
               void* host_work, std::size_t host_bytes);

}} // namespace ddla::ptrmm_impl

#endif // DDLA_PTRMM_IMPL_H
