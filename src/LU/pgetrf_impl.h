#ifndef DDLA_PGETRF_IMPL_H
#define DDLA_PGETRF_IMPL_H

#include <ddla/ddla.h>

#include <cstddef>

namespace ddla { namespace lu_impl {

// The distributed LU factorizations (pgetf2, pgetrf, pgetrf_nopiv, pgetrf_bpiv) split into the three components of
// plan_util.h:
//
//   1. <r>_check          -- parameter validation.  Collective (detail::agree_check): a bad argument on one rank
//                            throws on every rank (the failing rank its own exception, the others
//                            std::invalid_argument), and a DDLA_LU_PANEL that differs between the ranks throws
//                            std::runtime_error.  The check reads the environment and returns the agreed selector;
//                            nothing after it does.
//   2. <r>_workspace_size -- the scratch in bytes: backend memory (device on GPU handles, host on CPU handles) and,
//                            where the routine needs it, host memory.  No allocation, no communication.
//   3. <r>_run            -- the algorithm on the caller's scratch: no validation, no environment reads, no
//                            allocation.
//
// The public routine is check -> (no-op handling) -> workspace size -> allocate -> run.  pgetrf sizes the panel calls
// of its whole schedule and runs pgetf2_run on its own scratch for every panel.

// ===========================================================================
// pgetf2: blocked LU of a panel of `width` columns
// ===========================================================================

// Component 1: null / mismatched handle (std::runtime_error); then, in this order, the descriptor, the region
// A(ia:ia+m-1, ja:ja+n-1), the panel width (width < 0 is rejected), the local storage (null A / ipiv where this
// rank holds data) -- std::invalid_argument -- and the DDLA_LU_PANEL agreement.  Returns the agreed selector
// (true: gather).
template <DdlaBackend Backend>
bool pgetf2_check(const DdlaHandle_t& handle, int m, int n, int width, const void* A, int ia, int ja,
                  const int* desc, const int* ipiv);

// Component 2: the scratch of the panel call (m, n, width, ia, ja) whose swap range starts at column swap_ja, with
// the agreed selector `gather`.  Backend: one region for the row of U broadcast between the inner subpanels and the
// buffers of the panel method (lu_panel.h); host: the gathered method's pivot list and peer list (none in place).  A
// is what the vendor query of the gathered method's local getrf is made with.  width <= 0 needs nothing.
template <typename T>
void pgetf2_workspace_size(const DdlaHandle_t& handle, bool gather, int m, int n, int width, T* A, int ia, int ja,
                           const int* desc, int swap_ja, std::size_t* backend_bytes, std::size_t* host_bytes);

// Component 3: LU of the first `width` columns of A(ia:ia+m-1, ja:ja+n-1) (swaps cover columns swap_ja ..
// ja+n-1).  No validation, no environment reads, no allocation.
template <DdlaBackend Backend, typename T>
void pgetf2_run(const DdlaHandle_t& handle, bool gather, int m, int n, int width, T* A, int ia, int ja,
                const int* desc, int* ipiv, int& info, int swap_ja,
                void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes);

// ===========================================================================
// pgetrf: LU with partial pivoting of A(ia:ia+m-1, ja:ja+n-1)
// ===========================================================================

// Component 1: as pgetf2_check for the whole min(m, n) columns.  Returns the agreed DDLA_LU_PANEL selector.
template <DdlaBackend Backend>
bool pgetrf_check(const DdlaHandle_t& handle, int m, int n, const void* A, int ia, int ja,
                  const int* desc, const int* ipiv);

// Component 2: the scratch of an m x n factorization at (ia, ja).  Backend: the L panel and the diagonal block /
// U panel every step broadcasts, in one region with the panel calls' scratch; host: the panel calls'.  m == 0 or
// n == 0 needs nothing.
template <typename T>
void pgetrf_workspace_size(const DdlaHandle_t& handle, bool gather, int m, int n, T* A, int ia, int ja,
                           const int* desc, std::size_t* backend_bytes, std::size_t* host_bytes);

// Component 3: A = P L U.  No validation, no environment reads, no allocation.  Ends with a synchronize.
template <DdlaBackend Backend, typename T>
void pgetrf_run(const DdlaHandle_t& handle, bool gather, int m, int n, T* A, int ia, int ja,
                const int* desc, int* ipiv, int& info,
                void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes);

// ===========================================================================
// pgetrf_nopiv: LU without pivoting of the leading m x n block
// ===========================================================================

// Component 1: the checks of the public entry, collective: null / mismatched handle (std::runtime_error), the
// descriptor, then m x n inside the descriptor and square blocks (std::invalid_argument).  The sizes are not
// otherwise checked: m <= 0 or n <= 0 is an empty factorization.
template <DdlaBackend Backend>
void pgetrf_nopiv_check(const DdlaHandle_t& handle, int m, int n, const int* descA);

// Component 2: backend bytes of the diagonal block's info word, the diagonal block / L panel and the U panel.
template <typename T>
void pgetrf_nopiv_workspace_size(const DdlaHandle_t& handle, int m, int n, const int* descA,
                                 std::size_t* backend_bytes);

// Component 3: A = L U.  No validation, no allocation.  Ends with a synchronize.
template <DdlaBackend Backend, typename T>
void pgetrf_nopiv_run(const DdlaHandle_t& handle, int m, int n, T* A, const int* descA, int& info,
                      void* backend_work, std::size_t backend_bytes);

// ===========================================================================
// pgetrf_bpiv: block-partial-pivoting LU of the leading m x n block
// ===========================================================================

// Component 1: as pgetrf_nopiv_check.
template <DdlaBackend Backend>
void pgetrf_bpiv_check(const DdlaHandle_t& handle, int m, int n, const int* descA);

// Component 2: backend bytes of pgetrf_nopiv's buffers, whose region also holds the vendor workspace of the diagonal
// block's getrf; host bytes of laswp's pivot staging.  A and its leading dimension are what the vendor query is
// made with.
template <typename T>
void pgetrf_bpiv_workspace_size(const DdlaHandle_t& handle, int m, int n, T* A, const int* descA,
                                std::size_t* backend_bytes, std::size_t* host_bytes);

// Component 3: block LU; pivots is a backend array of 1-based block-local offsets.  No validation, no allocation.
// Ends with a synchronize.
template <DdlaBackend Backend, typename T>
void pgetrf_bpiv_run(const DdlaHandle_t& handle, int m, int n, T* A, const int* descA, int* pivots, int& info,
                     void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes);

}} // namespace ddla::lu_impl

#endif // DDLA_PGETRF_IMPL_H
