#ifndef DDLA_PSWAP_IMPL_H
#define DDLA_PSWAP_IMPL_H

#include <ddla/ddla.h>

#include <cstddef>

namespace ddla { namespace pswap_impl {

// pswap split into the three components of plan_util.h: check, workspace
// size, run.  Library-internal callers (ppotrf, ppotrs) size the scratch once
// for their descriptor and call pswap_run for every swap, with no argument
// check and no allocation.

// Component 1: the checks of the public entry.  Collective: every rank throws
// if any rank does.
template <DdlaBackend Backend>
void pswap_check(const DdlaHandle_t& handle, int N,
                 int ia, int ja, const int* array_descA, int inca,
                 int ib, int jb, const int* array_descB, int incb);

// Component 2: backend bytes of the two exchange buffers for swaps within
// matrices described by array_descA, each covering the longest local row or
// column of A: the row branch exchanges through a matched sendrecv pair, the
// column branch reuses the first.  No validation, no collectives.
template <typename T>
void pswap_workspace_size(const DdlaHandle_t& handle, const int* array_descA, std::size_t* backend_bytes);

// Component 3: the swap.  No validation, no allocation.  The exchange
// buffers are sized from array_descA, as pswap_workspace_size sizes them.
template <DdlaBackend Backend, typename T>
void pswap_run(const DdlaHandle_t& handle, int N,
               T* A, int ia, int ja, const int* array_descA, int inca,
               T* B, int ib, int jb, const int* array_descB, int incb,
               void* backend_work, std::size_t backend_bytes);

}} // namespace ddla::pswap_impl

#endif // DDLA_PSWAP_IMPL_H
