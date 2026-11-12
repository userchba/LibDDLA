#ifndef DDLA_PTRTRS_IMPL_H
#define DDLA_PTRTRS_IMPL_H

#include <ddla/ddla.h>

#include <cstddef>

namespace ddla { namespace ptrtrs_impl {

// ptrtrs split into the three components of plan_util.h: check, workspace
// size, run.  The triangular solves of pgetrs/ppotrs size the scratch once and
// run twice (lower then upper factor), with no argument check and no
// allocation.

// Component 1: the checks of the public entry.  Collective: every rank throws
// if any rank does.
template <DdlaBackend Backend>
void ptrtrs_check(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag,
                  int m, int n, const int* array_descA, const int* array_descB);

// Component 2: the scratch of a solve of the leading m x n block of B with
// op(A) from the left or right side.  Backend: the staged diagonal block, the
// panel of A and the block row/column of B that every step gathers, then the
// send staging of the panel transport; host: the transport's schedule.
// trans == 'N' needs no transport scratch, any other value stages the
// transposed panel (on a non-square grid), so a size for trans != 'N' serves
// every side/uplo/trans/diag of that side and any order up to m x n.  No
// validation, no collectives.
template <typename T>
void ptrtrs_workspace_size(const DdlaHandle_t& handle, char side, char trans, int m, int n,
                           const int* array_descA, const int* array_descB,
                           std::size_t* backend_bytes, std::size_t* host_bytes);

// Component 3: B := op(A)^{-1} B (side 'L') or B := B op(A)^{-1} (side 'R').
// A is read-only. No validation, no allocation.
template <DdlaBackend Backend, typename T>
void ptrtrs_run(const DdlaHandle_t& handle,
                char side, char uplo, char trans, char diag, int m, int n,
                const T* A, const int* array_descA, T* B, const int* array_descB,
                void* backend_work, std::size_t backend_bytes,
                void* host_work, std::size_t host_bytes);

}} // namespace ddla::ptrtrs_impl

#endif // DDLA_PTRTRS_IMPL_H
