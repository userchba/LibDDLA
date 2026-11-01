#ifndef DDLA_PLAPIV_IMPL_H
#define DDLA_PLAPIV_IMPL_H

#include <ddla/ddla.h>

#include <cstddef>

namespace ddla { namespace plapiv_impl {

// plapiv split into the three components of plan_util.h: check, workspace
// size (the row/column exchange buffer), run.  Library-internal callers
// (pgetrs) size the scratch once and call plapiv_run, with no argument check
// and no allocation.

// Component 1: the checks of the public entry.  Collective: every rank throws
// if any rank does.
template <DdlaBackend Backend>
void plapiv_check(const DdlaHandle_t& handle, char direc, char rowcol, char pivroc,
                  int m, int n, const int* array_descA, const int* array_descIP);

// Component 2: backend bytes of the exchange buffer for pivoting segments of
// length n of the matrix described by array_descA: the longest segment this
// rank exchanges (a row of the leading n columns, or a column of the leading
// n rows).  No validation, no collectives.
template <typename T>
void plapiv_workspace_size(const DdlaHandle_t& handle, int n, const int* array_descA,
                           std::size_t* backend_bytes);

// Component 3: the pivoting.  No validation, no allocation.  `iwork` of the
// public routine is unused and has no counterpart here.
template <DdlaBackend Backend, typename T>
void plapiv_run(const DdlaHandle_t& handle, char direc, char rowcol,
                int m, int n, T* A, const int* array_descA,
                const int* ipiv, const int* array_descIP,
                void* backend_work, std::size_t backend_bytes);

}} // namespace ddla::plapiv_impl

#endif // DDLA_PLAPIV_IMPL_H
