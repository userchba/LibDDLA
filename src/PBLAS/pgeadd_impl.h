#ifndef DDLA_PGEADD_IMPL_H
#define DDLA_PGEADD_IMPL_H

#include <ddla/ddla.h>

#include <cstddef>

namespace ddla { namespace pgeadd_impl {

// pgeadd split into the three components of plan_util.h: check, workspace
// size, run.  Library-internal callers (ptrmm, phemm, pgemm_sym, pgemm's
// stationary branches) size the scratch once for the largest panel they will
// add and call pgeadd_run for every panel, with no argument check and no
// allocation.

// The normalized transposition flags, and whether there is nothing to do.

// Component 1: validate every argument exactly like the public entry.  A
// null or mismatched handle throws std::runtime_error, everything else
// std::invalid_argument.  Collective: every rank throws if any rank does.
template <DdlaBackend Backend>
void pgeadd_check(const DdlaHandle_t& h, char transa, char transb, int m, int n,
                        const int* da, int ia, int ja,
                        const int* db, int ib, int jb,
                        const int* dc, int ic, int jc);

// Coefficient-aware validation used by the public wrapper. Unused inputs still
// satisfy the descriptor/region contract, but do not need MPI count capacity.
template <DdlaBackend Backend>
void pgeadd_check(const DdlaHandle_t& h, char transa, char transb, int m, int n,
                        const int* da, int ia, int ja,
                        const int* db, int ib, int jb,
                        const int* dc, int ic, int jc, bool use_a, bool use_b);

// Component 2: the scratch of calls of at most m x n with the block sizes of
// dc.  use_a / use_b say whether alpha / beta can be nonzero (a zero
// coefficient reads no operand and needs no buffer).  Backend: the staging of
// the redistribution and of the prepared operands, and on a GPU handle the
// device copy of the redistribution schedule; host: that schedule (only for
// calls that can take the redistribution route).  No validation, no
// collectives.
template <typename T>
void pgeadd_workspace_size(const DdlaHandle_t& h, char ta, char tb, bool use_a, bool use_b,
                           int m, int n, const int* dc,
                           std::size_t* backend_bytes, std::size_t* host_bytes);

// Component 3: C := alpha*op(A) + beta*op(B).  No validation, no allocation;
// ta/tb are the normalized flags of pgeadd_check.  The host scratch holds
// schedule records that are uploaded to the device, so it must be pageable
// memory (std::vector<char>): the uploads have consumed it when they return.
template <DdlaBackend Backend, typename T>
void pgeadd_run(const DdlaHandle_t& h, char ta, char tb, int m, int n,
                const T& alpha, const T* a, int ia, int ja, const int* da,
                const T& beta, const T* b, int ib, int jb, const int* db,
                T* c, int ic, int jc, const int* dc,
                void* backend_work, std::size_t backend_bytes,
                void* host_work, std::size_t host_bytes);

// The largest staging buffer of a call of at most m x n, in elements: the
// longest message such a call sends or receives, which a GPU_CPU_TUNNEL build
// stages in the handle's host buffers (callers grow them once, before a loop).
std::size_t pgeadd_staging_count(const DdlaHandle_t& h, char ta, char tb, bool use_a, bool use_b,
                                 int m, int n, const int* dc);

}} // namespace ddla::pgeadd_impl

#endif // DDLA_PGEADD_IMPL_H
