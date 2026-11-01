#ifndef DDLA_LAPACK_IMPL_H
#define DDLA_LAPACK_IMPL_H

// Local (single-process) LAPACK routines split into the components of plan_util.h, as cuSOLVERMp splits its
// routines:
//   <r>_check          -- argument validation (std::runtime_error; local, no collectives),
//   <r>_workspace_size -- the scratch in bytes, only the kinds the routine needs; no allocation,
//   <r>_run            -- the factorization on the caller's scratch; no validation, no allocation.
// The caller allocates the bytes (ddla::vector<char> for backend memory, std::vector<char> for host memory).  A
// caller that runs several shapes on one buffer sizes every shape and keeps the largest: a run takes its vendor lwork
// from the bytes it is given, so a larger buffer serves a smaller shape.  getrf_nopiv needs no scratch: check + run.

#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>

#include <cstddef>

namespace ddla { namespace lapack_impl {

// ===========================================================================
// getrf
// ===========================================================================

// Component 1: null / mismatched handle (std::runtime_error), negative size, null info; then, for a non-empty
// problem only (m == 0 or n == 0 may carry null pointers), null A, null ipiv, lda < max(1, m).  Every failure is a
// std::runtime_error.  A is passed untyped: the check does not depend on the scalar type.
template <DdlaBackend Backend>
void getrf_check(const DdlaHandle_t& handle, int m, int n, const void* A, int lda,
                 const int* ipiv, const int* info);

// Component 2: backend bytes of an m x n factorization -- the vendor workspace on a GPU handle (queried with A and
// lda), nothing on a CPU handle.  An empty shape needs nothing.
template <typename T>
void getrf_workspace_size(const DdlaHandle_t& handle, int m, int n, T* A, int lda, std::size_t* backend_bytes);

// Component 3: A = P L U.  No validation, no allocation.  m <= 0 or n <= 0 only sets *info = 0 (a host store on a
// CPU handle, ddlaMemset on a GPU handle) and returns, so loop callers need not pre-filter.
template <DdlaBackend Backend, typename T>
void getrf_run(const DdlaHandle_t& handle, int m, int n, T* A, int lda, int* ipiv, int* info,
               void* backend_work, std::size_t backend_bytes);

// ===========================================================================
// potrf
// ===========================================================================

// Component 1: null / mismatched handle, invalid uplo ('U'/'L' in either case), negative order, null info; then,
// for n > 0, null A and lda < max(1, n).  Returns the normalized uplo ('U' or 'L') that the size and run take.
template <DdlaBackend Backend>
char potrf_check(const DdlaHandle_t& handle, char uplo, int n, const void* A, int lda, const int* info);

// Component 2: backend bytes of an order-n factorization on the normalized uplo -- the vendor workspace on a GPU
// handle (queried with A and lda), nothing on a CPU handle.
template <typename T>
void potrf_workspace_size(const DdlaHandle_t& handle, char uplo, int n, T* A, int lda, std::size_t* backend_bytes);

// Component 3: A = U^H U / L L^H on the normalized uplo.  n <= 0 only sets *info = 0, as getrf_run does.
template <DdlaBackend Backend, typename T>
void potrf_run(const DdlaHandle_t& handle, char uplo, int n, T* A, int lda, int* info,
               void* backend_work, std::size_t backend_bytes);

// ===========================================================================
// laswp
// ===========================================================================

// Component 1: null / mismatched handle; then, unless the call is empty (n <= 0 or k1 > k2), k1 < 1, incx == 0,
// null A, null ipiv, lda < max(1, k2).  Every failure is a std::runtime_error.
template <DdlaBackend Backend>
void laswp_check(const DdlaHandle_t& handle, int n, const void* A, int lda, int k1, int k2,
                 const int* ipiv, int incx);

// Component 2: host bytes of the pivot staging, (k2 - k1) * |incx| + 1 ints.  Nothing on a CPU handle, nor for the
// vendor's laswp (incx == 1) on a GPU handle that has one.
void laswp_workspace_size(const DdlaHandle_t& handle, int k1, int k2, int incx, std::size_t* host_bytes);

// Component 3: the row interchanges.  No validation, no allocation.  n <= 0 or k1 > k2 returns.
template <DdlaBackend Backend, typename T>
void laswp_run(const DdlaHandle_t& handle, int n, T* A, int lda, int k1, int k2, const int* ipiv, int incx,
               void* host_work, std::size_t host_bytes);

// ===========================================================================
// geqrf (QR for uplo 'U', QL for uplo 'L')
// ===========================================================================

// Component 1: null / mismatched handle, uplo not 'U'/'L' (either case), negative size; then, for a non-empty
// problem (min(m, n) > 0), null A, null tau, lda < max(1, m).  Returns the normalized uplo.
template <DdlaBackend Backend>
char geqrf_check(const DdlaHandle_t& handle, char uplo, int m, int n, const void* A, int lda, const void* tau);

// Component 2: backend bytes of an m x n factorization on the normalized uplo.  CPU handles: the ?geqrf / ?geqlf work
// array (LAPACK's workspace query, made with A, lda and tau).  GPU handles: a device info word, for uplo 'L' the
// reversed copy A' = J A J (cuSOLVER / hipSOLVER have no ?geqlf) and tau', and the vendor workspace (queried with A and
// the leading dimension of the matrix it factors: lda, or m for A').  min(m, n) <= 0 needs nothing.
template <typename T>
void geqrf_workspace_size(const DdlaHandle_t& handle, char uplo, int m, int n, T* A, int lda, T* tau,
                          std::size_t* backend_bytes);

// Component 3: A = Q R / A = Q L on the normalized uplo.  No validation, no allocation.  min(m, n) <= 0 returns.
template <DdlaBackend Backend, typename T>
void geqrf_run(const DdlaHandle_t& handle, char uplo, int m, int n, T* A, int lda, T* tau,
               void* backend_work, std::size_t backend_bytes);

// ===========================================================================
// larft (STOREV = 'C')
// ===========================================================================

// Component 1: null / mismatched handle, direct not 'F'/'B' (either case), storev not 'C', negative size; then,
// for k > 0, k > n, null V / tau / T, ldv < max(1, n), ldt < k.  Returns the normalized direct.
template <DdlaBackend Backend>
char larft_check(const DdlaHandle_t& handle, char direct, char storev, int n, int k,
                 const void* V, int ldv, const void* tau, const void* Tmat, int ldt);

// Component 2: backend bytes of an n x k reflector block -- on a GPU handle [ head / tail of V (k x k) | V^H V
// (k x k) ], nothing on a CPU handle.  k <= 0 needs nothing.
template <typename T>
void larft_workspace_size(const DdlaHandle_t& handle, int n, int k, std::size_t* backend_bytes);

// Component 3: T of the block reflector.  `direct` is the normalized 'F' / 'B'.  No validation, no allocation.
// k <= 0 returns.
template <DdlaBackend Backend, typename T>
void larft_run(const DdlaHandle_t& handle, char direct, int n, int k, T* V, int ldv, const T* tau, T* Tmat, int ldt,
               void* backend_work, std::size_t backend_bytes);

// ===========================================================================
// getrf_nopiv (private; src/LU/lu_internal.h)
// ===========================================================================
// No scratch (the register kernel and the BLAS calls it makes need none), so there is no size: check + run.

// Check: null / mismatched handle (std::runtime_error). The routine has no other validation.
template <DdlaBackend Backend>
void getrf_nopiv_check(const DdlaHandle_t& handle);

// Execution: A = L U without pivoting, info_buffer in backend memory. No validation, no allocation.
template <DdlaBackend Backend, typename T>
void getrf_nopiv_run(const DdlaHandle_t& handle, int m, int n, T* A, int lda, int* info_buffer);

// ===========================================================================
// potrf_bottom_right (single process; src/Cholesky/potrf_bottom_right*.cpp)
// ===========================================================================

// Component 1 of the public potrf_bottom_right: null / mismatched handle, uplo not exactly 'U' / 'L', negative order;
// then, for n > 0, null A and lda < n.  Every failure is a std::runtime_error.
template <DdlaBackend Backend>
void potrf_bottom_right_check(char uplo, int n, const void* A, int lda, const DdlaHandle_t& handle);

// Component 2: backend bytes of an order-n factorization: the reverse-packed diagonal block (nb x nb with
// nb = min(n, 128)), its ?potrf info word and the ?potrf scratch of that block (planned for the opposite triangle,
// queried with A).  n <= 0 needs nothing.
template <typename T>
void potrf_bottom_right_workspace_size(const DdlaHandle_t& handle, char uplo, int n, T* A, std::size_t* backend_bytes);

// Component 3: the factorization.  No validation, no allocation.  Ends with a synchronize, like the public routine.
template <DdlaBackend Backend, typename T>
void potrf_bottom_right_run(const DdlaHandle_t& handle, char uplo, int n, T* A, int lda, int& info,
                            void* backend_work, std::size_t backend_bytes);

// Component 1 of detail::potrf_bottom_right_block (src/Cholesky/potrf_bottom_right_internal.h): null / mismatched
// handle, uplo not exactly 'U' / 'L', negative order or global offset; then, for n > 0, null A / work /
// info_buffer and lda < n.  Every failure is a std::runtime_error.
template <DdlaBackend Backend>
void potrf_bottom_right_block_check(char uplo, int n, const void* A, int lda, int global_offset,
                                    const void* work, const void* info_buffer, const DdlaHandle_t& handle);

}} // namespace ddla::lapack_impl

#endif // DDLA_LAPACK_IMPL_H
