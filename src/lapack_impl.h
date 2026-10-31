#ifndef DDLA_LAPACK_IMPL_H
#define DDLA_LAPACK_IMPL_H

// ---------------------------------------------------------------------------
// The single-process LAPACK-level routines (getrf, potrf, laswp, geqrf, larft, the private getrf_nopiv and the
// bottom-right Cholesky) split into the three components every routine has (see PBLAS/ptrtrm_impl.h):
//
//   1. <r>_check -- parameter validation, exactly what the public entry rejects, in the same order, with the
//                   same exception types and messages.  LOCAL checks: no MPI, no collectives.
//   2. <r>_plan  -- size and allocate every temporary (device memory on GPU handles, host memory on CPU handles).
//   3. <r>_run   -- the algorithm: no validation, no environment reads, no allocation.
//
// The public routine composes 1 -> (empty / no-op handling) -> 2 -> 3.  Library-internal callers (the panel loops
// of pgetrf / ppotrf / pgeqrf ...) that already hold valid arguments call 2 once and 3 for every call:
//
//   auto work = lapack_impl::getrf_plan<T>(handle, {{m_max, w_max}}, A, lda);   // once
//   for (...) lapack_impl::getrf_run<Backend, T>(handle, work, m, w, A, lda, ipiv, info);
//
// Vendor workspace (cuSOLVER / hipSOLVER ?getrf / ?potrf / ?geqrf, the CPU ?geqrf work array) is sized from a LIST OF
// SHAPES.  Every plan takes the list either as a braced std::initializer_list or as a runtime array (pointer, count),
// for a caller that enumerates its panel schedule at run time; the initializer_list forms forward to the array forms,
// and a count of 0 yields an empty workspace without allocating.  Plan queries the vendor's bufferSize for every listed
// shape and allocates the maximum; run queries the bufferSize of its exact shape again (a pure host call), asserts it
// fits the planned capacity (assert) and hands the vendor exactly the lwork it queried.  So every shape a
// run receives must need no more than the planned maximum; pass the caller's real matrix pointer and leading
// dimension to plan (the query is made with them).  Backend workspace is bound to one handle (the memory is ordered on that
// handle's stream, which is also the stream the solver handle runs on) and is released when it goes out of scope.
// ---------------------------------------------------------------------------

#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace ddla { namespace lapack_impl {

// One m x n matrix handed to getrf / geqrf.
struct Shape {
    int m;
    int n;
};

// One call of larft: V is n x k.
struct LarftShape {
    int n;
    int k;
};

// One call of laswp: rows k1..k2 (1-based) with pivot stride incx.
struct LaswpShape {
    int k1;
    int k2;
    int incx;
};

// ===========================================================================
// getrf
// ===========================================================================

// Component 1: null / mismatched handle (std::runtime_error), negative size, null info; then, for a non-empty
// problem only (m == 0 or n == 0 may carry null pointers), null A, null ipiv, lda < max(1, m).  Every failure is a
// std::runtime_error.  A is passed untyped: the check does not depend on the scalar type.
template <DdlaBackend Backend>
void getrf_check(const DdlaHandle_t& handle, int m, int n, const void* A, int lda,
                 const int* ipiv, const int* info);

// Component 2: GPU vendor workspace (empty on CPU handles) for any of the listed shapes. A and lda
// are what the vendor's bufferSize query is made with.  Empty shapes (m <= 0 or n <= 0) need nothing; count == 0 plans
// nothing (shapes may then be null).  No validation.
template <typename T>
vector<T> getrf_plan(const DdlaHandle_t& handle, const Shape* shapes, std::size_t count, T* A, int lda);
template <typename T>
inline vector<T> getrf_plan(const DdlaHandle_t& handle, std::initializer_list<Shape> shapes, T* A, int lda)
{
    return getrf_plan<T>(handle, shapes.begin(), shapes.size(), A, lda);
}

// Component 3: A = P L U.  No validation, no allocation.  m <= 0 or n <= 0 only sets *info = 0 (a host store on a
// CPU handle, ddlaMemset on a GPU handle) and returns, so loop callers need not pre-filter.
template <DdlaBackend Backend, typename T>
void getrf_run(const DdlaHandle_t& handle, vector<T>& work, int m, int n, T* A, int lda,
               int* ipiv, int* info);

// ===========================================================================
// potrf
// ===========================================================================

// Component 1: null / mismatched handle, invalid uplo ('U'/'L' in either case), negative order, null info; then,
// for n > 0, null A and lda < max(1, n).  Returns the normalized uplo ('U' or 'L') that plan and run take.
template <DdlaBackend Backend>
char potrf_check(const DdlaHandle_t& handle, char uplo, int n, const void* A, int lda, const int* info);

// Component 2: GPU vendor workspace (empty on CPU handles) for any listed order, with normalized uplo
// and leading dimension lda.
// Orders <= 0 need nothing; count == 0 plans nothing (orders may then be null).
template <typename T>
vector<T> potrf_plan(const DdlaHandle_t& handle, char uplo, const int* orders, std::size_t count, T* A, int lda);
template <typename T>
inline vector<T> potrf_plan(const DdlaHandle_t& handle, char uplo, std::initializer_list<int> orders, T* A, int lda)
{
    return potrf_plan<T>(handle, uplo, orders.begin(), orders.size(), A, lda);
}

// Component 3: A = U^H U / L L^H on the normalized uplo.  n <= 0 only sets *info = 0, as getrf_run does.
template <DdlaBackend Backend, typename T>
void potrf_run(const DdlaHandle_t& handle, vector<T>& work, char uplo, int n, T* A, int lda, int* info);

// ===========================================================================
// laswp
// ===========================================================================

// Component 1: null / mismatched handle; then, unless the call is empty (n <= 0 or k1 > k2), k1 < 1, incx == 0,
// null A, null ipiv, lda < max(1, k2).  Every failure is a std::runtime_error.
template <DdlaBackend Backend>
void laswp_check(const DdlaHandle_t& handle, int n, const void* A, int lda, int k1, int k2,
                 const int* ipiv, int incx);

// Component 2: host pivot staging for the largest listed span (k2 - k1) * |incx| + 1. Needs nothing
// on a CPU handle, nor for the vendor's laswp (incx == 1) on a GPU handle that has one; count == 0 plans
// nothing (shapes may then be null).
std::vector<int> laswp_plan(const DdlaHandle_t& handle, const LaswpShape* shapes, std::size_t count);
inline std::vector<int> laswp_plan(const DdlaHandle_t& handle, std::initializer_list<LaswpShape> shapes)
{
    return laswp_plan(handle, shapes.begin(), shapes.size());
}

// Component 3: the row interchanges.  No validation, no allocation.  n <= 0 or k1 > k2 returns.
template <DdlaBackend Backend, typename T>
void laswp_run(const DdlaHandle_t& handle, std::vector<int>& pivots, int n, T* A, int lda, int k1, int k2,
               const int* ipiv, int incx);

// ===========================================================================
// geqrf (QR for uplo 'U', QL for uplo 'L')
// ===========================================================================

// What geqrf_plan allocates.  CPU handles: the ?geqrf / ?geqlf work array.  GPU handles: the vendor scratch, and for
// uplo 'L' the buffers of the reversed copy A' = J A J (cuSOLVER / hipSOLVER have no ?geqlf).
template <typename T>
struct GeqrfCtx {
    vector<T> work;     // CPU: LAPACK work array (elements)
    vector<char> qr;    // GPU: [ device info slot (256 bytes) | vendor workspace ] (bytes)
    vector<char> ql;    // GPU, uplo 'L': [ A' (m x n, ld m, padded to 256 bytes) | tau' (k) ] (bytes)

    explicit GeqrfCtx(DdlaHandle_t h) : work(h), qr(h), ql(h) {}
};

// Component 1: null / mismatched handle, uplo not 'U'/'L' (either case), negative size; then, for a non-empty
// problem (min(m, n) > 0), null A, null tau, lda < max(1, m).  Returns the normalized uplo.
template <DdlaBackend Backend>
char geqrf_check(const DdlaHandle_t& handle, char uplo, int m, int n, const void* A, int lda, const void* tau);

// Component 2: scratch for factorizations of any of the listed shapes (normalized uplo, leading dimension lda; A and
// tau are what the CPU workspace query is made with, A and lda what the vendor's bufferSize query is made with --
// for uplo 'L' on a GPU handle the query is made with the context's own reversed copy, as the factorization is).
// A context planned for 'U' cannot run 'L'.  Empty shapes (min(m, n) <= 0) need nothing; count == 0 plans nothing
// (shapes may then be null).
template <typename T>
GeqrfCtx<T> geqrf_plan(const DdlaHandle_t& handle, char uplo, const Shape* shapes, std::size_t count,
                       T* A, int lda, T* tau);
template <typename T>
inline GeqrfCtx<T> geqrf_plan(const DdlaHandle_t& handle, char uplo, std::initializer_list<Shape> shapes,
                              T* A, int lda, T* tau)
{
    return geqrf_plan<T>(handle, uplo, shapes.begin(), shapes.size(), A, lda, tau);
}

// Component 3: A = Q R / A = Q L on the normalized uplo.  No validation, no allocation.  min(m, n) <= 0 returns.
template <DdlaBackend Backend, typename T>
void geqrf_run(const DdlaHandle_t& handle, GeqrfCtx<T>& ctx, char uplo, int m, int n, T* A, int lda, T* tau);

// ===========================================================================
// larft (STOREV = 'C')
// ===========================================================================

// Component 1: null / mismatched handle, direct not 'F'/'B' (either case), storev not 'C', negative size; then,
// for k > 0, k > n, null V / tau / T, ldv < max(1, n), ldt < k.  Returns the normalized direct.
template <DdlaBackend Backend>
char larft_check(const DdlaHandle_t& handle, char direct, char storev, int n, int k,
                 const void* V, int ldv, const void* tau, const void* Tmat, int ldt);

// Component 2: GPU scratch [head / tail (k x k, padded to 256 bytes) | V^H V (k x k)], empty on CPU handles,
// for any of the listed shapes.  Empty shapes (k <= 0) need nothing; count == 0 plans nothing
// (shapes may then be null).
template <typename T>
vector<T> larft_plan(const DdlaHandle_t& handle, const LarftShape* shapes, std::size_t count);
template <typename T>
inline vector<T> larft_plan(const DdlaHandle_t& handle, std::initializer_list<LarftShape> shapes)
{
    return larft_plan<T>(handle, shapes.begin(), shapes.size());
}

// Component 3: T of the block reflector.  `direct` is the normalized 'F' / 'B'.  No validation, no allocation.
// k <= 0 returns.
template <DdlaBackend Backend, typename T>
void larft_run(const DdlaHandle_t& handle, vector<T>& work, char direct, int n, int k, T* V, int ldv,
               const T* tau, T* Tmat, int ldt);

// ===========================================================================
// getrf_nopiv (private; src/LU/lu_internal.h)
// ===========================================================================
// No temporary of its own (the register kernel and the BLAS calls it makes need no workspace from here), so there is
// no plan: check + run.

// Component 1: null / mismatched handle (std::runtime_error).  The routine has no other validation.
template <DdlaBackend Backend>
void getrf_nopiv_check(const DdlaHandle_t& handle);

// Component 3: A = L U without pivoting, info_buffer in backend memory.  No validation, no allocation.
template <DdlaBackend Backend, typename T>
void getrf_nopiv_run(const DdlaHandle_t& handle, int m, int n, T* A, int lda, int* info_buffer);

// ===========================================================================
// potrf_bottom_right (single process; src/Cholesky/potrf_bottom_right*.cpp)
// ===========================================================================

// What potrf_bottom_right_plan allocates: the reverse-packed diagonal block, its info word and the ?potrf workspace
// of that block.
template <typename T>
struct PotrfBottomRightCtx {
    vector<T> work;         // the reverse-packed diagonal block, nb x nb with nb = min(n, 128)
    vector<int> info_buffer;// the diagonal block's ?potrf info word, backend memory
    vector<T> potrf;      // the ?potrf of that block (planned for the opposite triangle)

    explicit PotrfBottomRightCtx(DdlaHandle_t h) : work(h), info_buffer(h), potrf(h) {}
};

// Component 1 of the public potrf_bottom_right: null / mismatched handle, uplo not exactly 'U' / 'L', negative order;
// then, for n > 0, null A and lda < n.  Every failure is a std::runtime_error.
template <DdlaBackend Backend>
void potrf_bottom_right_check(char uplo, int n, const void* A, int lda, const DdlaHandle_t& handle);

// Component 2: the buffers for an order-n factorization (n <= 0 allocates nothing).
template <typename T>
PotrfBottomRightCtx<T> potrf_bottom_right_plan(const DdlaHandle_t& handle, char uplo, int n);

// Component 3: the factorization.  No validation, no allocation.  Ends with a synchronize, like the public routine.
template <DdlaBackend Backend, typename T>
void potrf_bottom_right_run(const DdlaHandle_t& handle, PotrfBottomRightCtx<T>& ctx, char uplo, int n,
                            T* A, int lda, int& info);

// Component 1 of detail::potrf_bottom_right_block (src/Cholesky/potrf_bottom_right_internal.h): null / mismatched
// handle, uplo not exactly 'U' / 'L', negative order or global offset; then, for n > 0, null A / work /
// info_buffer and lda < n.  Every failure is a std::runtime_error.
template <DdlaBackend Backend>
void potrf_bottom_right_block_check(char uplo, int n, const void* A, int lda, int global_offset,
                                    const void* work, const void* info_buffer, const DdlaHandle_t& handle);

}} // namespace ddla::lapack_impl

#endif // DDLA_LAPACK_IMPL_H
