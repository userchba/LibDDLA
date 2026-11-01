#ifndef DDLA_PGETRS_IMPL_H
#define DDLA_PGETRS_IMPL_H

#include <ddla/ddla.h>

#include "pgetrf_impl.h"

#include <cstddef>

namespace ddla { namespace lu_impl {

// The distributed LU solves (pgetrs, pgetrs_nopiv, pgetrs_bpiv) and drivers (pgesv, pgesv_nopiv, pgesv_bpiv) split
// into the three components of plan_util.h (see pgetrf_impl.h):
//
//   1. <r>_check          -- parameter validation, collective (detail::agree_check): a bad argument on one rank
//                            throws on every rank.  A composite's check is its own: it validates what its run relies
//                            on, before either the factorization or the solve changes user data, and never calls the
//                            checks of the routines it runs.  The drivers pgesv also agree on DDLA_LU_PANEL.
//   2. <r>_workspace_size -- the scratch of the whole composite in bytes; no allocation, no communication.
//   3. <r>_run            -- the algorithm on the caller's scratch: no validation, no environment reads, no
//                            allocation.  Composites run the components of their callees (pgetrf_run, plapiv_run,
//                            ptrtrs_run, ...), never the public entries.
//
// The two triangular solves of a solve use the user's `trans`, so one ptrtrs size serves both.  A driver's
// factorization and solve run one after the other on the same scratch: it holds the larger of the two.

// ===========================================================================
// pgetrs
// ===========================================================================

// Component 1: descriptors of B and A, then the backend (null / mismatched handle: std::runtime_error), then
// std::invalid_argument for side / trans, negative n or nrhs, n x nrhs outside the descriptors, A without square
// blocks, and A and B with different row block sizes.
template <DdlaBackend Backend>
void pgetrs_check(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                  const int* descA, const int* descB);

// Component 2: n x nrhs, op(A) on the given side.  Backend: the scratch of plapiv and of the triangular solves, which
// run one after the other; host: the triangular solves'.  n == 0 or nrhs == 0 needs nothing.
template <typename T>
void pgetrs_workspace_size(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                           const int* descA, const int* descB, std::size_t* backend_bytes, std::size_t* host_bytes);

// Component 3: the permutation of B (plapiv_run, ipiv in host memory) and the two triangular solves (ptrtrs_run).
template <DdlaBackend Backend, typename T>
void pgetrs_run(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                T* A, const int* descA, const int* ipiv, T* B, const int* descB,
                void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes);

// ===========================================================================
// pgetrs_nopiv
// ===========================================================================

// Component 1: as pgetrs_check.
template <DdlaBackend Backend>
void pgetrs_nopiv_check(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                        const int* descA, const int* descB);

// Component 2: the triangular solves' scratch.
template <typename T>
void pgetrs_nopiv_workspace_size(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                                 const int* descA, const int* descB,
                                 std::size_t* backend_bytes, std::size_t* host_bytes);

template <DdlaBackend Backend, typename T>
void pgetrs_nopiv_run(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                      T* A, const int* descA, T* B, const int* descB,
                      void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes);

// ===========================================================================
// pgetrs_bpiv
// ===========================================================================

// Component 1: as pgetrs_check.
template <DdlaBackend Backend>
void pgetrs_bpiv_check(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                       const int* descA, const int* descB);

// Component 2: backend bytes of the triangular solves; host bytes of [ the host copy of the local block pivots
// (local rows of A's leading n x n block) | the pivots of one block (nb) | the triangular solves' ].
template <typename T>
void pgetrs_bpiv_workspace_size(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                                const int* descA, const int* descB,
                                std::size_t* backend_bytes, std::size_t* host_bytes);

// pivots is the backend array of pgetrf_bpiv.
template <DdlaBackend Backend, typename T>
void pgetrs_bpiv_run(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                     T* A, const int* descA, int* pivots, T* B, const int* descB,
                     void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes);

// ===========================================================================
// pgesv: pgetrf + pgetrs
// ===========================================================================

// Component 1: the checks of pgetrf (the region A(1:n, 1:n): bounds, square blocks, local storage; the pivots
// belong to the scratch) and of pgetrs, in the order of the former public sequence (descriptors of B and A,
// backend, n against A's descriptor, then the rest), all before anything is factored; and the DDLA_LU_PANEL
// agreement.  Returns the agreed selector (true: gather).
template <DdlaBackend Backend>
bool pgesv_check(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                 const void* A, const int* descA, const int* descB);

// Component 2: n x nrhs.  Backend: the larger of pgetrf's and pgetrs's; host: [ the pivots (local rows of A's
// leading n x n block) | the larger of pgetrf's and pgetrs's ].  A is what pgetrf's vendor query is made with.
// n == 0 needs nothing.
template <typename T>
void pgesv_workspace_size(const DdlaHandle_t& handle, bool gather, char side, char trans, int n, int nrhs,
                          T* A, const int* descA, const int* descB,
                          std::size_t* backend_bytes, std::size_t* host_bytes);

// Component 3: factor, then (info == 0 and nrhs > 0) solve.  info is pgetrf's: a nonzero info skips the solve and
// is reported by the public wrapper as std::runtime_error("pgesv: pgetrf returned info = <info>").
template <DdlaBackend Backend, typename T>
void pgesv_run(const DdlaHandle_t& handle, bool gather, char side, char trans, int n, int nrhs,
               T* A, const int* descA, T* B, const int* descB, int& info,
               void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes);

// ===========================================================================
// pgesv_nopiv
// ===========================================================================

// Component 1: descriptors of B and A, backend, then the checks of pgetrs_nopiv.
template <DdlaBackend Backend>
void pgesv_nopiv_check(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                       const int* descA, const int* descB);

// Component 2: backend bytes of the larger of pgetrf_nopiv's and pgetrs_nopiv's; host bytes of pgetrs_nopiv's.
template <typename T>
void pgesv_nopiv_workspace_size(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                                const int* descA, const int* descB,
                                std::size_t* backend_bytes, std::size_t* host_bytes);

template <DdlaBackend Backend, typename T>
void pgesv_nopiv_run(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                     T* A, const int* descA, T* B, const int* descB, int& info,
                     void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes);

// ===========================================================================
// pgesv_bpiv
// ===========================================================================

// Component 1: as pgesv_nopiv_check.
template <DdlaBackend Backend>
void pgesv_bpiv_check(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                      const int* descA, const int* descB);

// Component 2: backend bytes of [ the block pivots (max(1, local rows) entries) | the larger of pgetrf_bpiv's and
// pgetrs_bpiv's ]; host bytes of the larger of theirs.  A is what the vendor workspace query of the diagonal block's
// getrf is made with.
template <typename T>
void pgesv_bpiv_workspace_size(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                               T* A, const int* descA, const int* descB,
                               std::size_t* backend_bytes, std::size_t* host_bytes);

template <DdlaBackend Backend, typename T>
void pgesv_bpiv_run(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                    T* A, const int* descA, T* B, const int* descB, int& info,
                    void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes);

}} // namespace ddla::lu_impl

#endif // DDLA_PGETRS_IMPL_H
