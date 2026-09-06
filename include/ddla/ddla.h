#ifndef DDLA_H
#define DDLA_H

#include <ddla/ddla_config.h>
#include "ddla_desc.h"
#include <cstdint>
#include <complex>
#include <stdexcept>
#include <string>

namespace ddla{

/// Check a ddlaStatus_t returned by a public API function.
///
/// Throws std::runtime_error with file:line context when @p status is not
/// DDLA_STATUS_SUCCESS, mirroring the private CHECK-family style.
inline void DDLA_CHECK(ddlaStatus_t status,
                       const char* file = __builtin_FILE(),
                       int line = __builtin_LINE())
{
    if (status != ddlaStatus_t::DDLA_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("ddla error ") +
            std::to_string(static_cast<int>(status)) + " at " + file + ":" +
            std::to_string(line));
    }
}

#if DDLA_HAS_GPU

/**
 * @brief Distributed triangular solve: B := op(A)^{-1} * B    (side='L')
 *                                     B := B * op(A)^{-1}    (side='R').
 *
 * Solves a triangular system on a distributed GPU matrix using block-cyclic
 * data distribution and MPI/NCCL/RCCL communication (per the DDLA_USE_CCL /
 * DDLA_USE_GPU_CPU_TUNNEL build configuration).  Corresponds to the
 * ScaLAPACK PZTRTRS / PDTRTRS routine.
 *
 * @tparam T  Scalar type (float, double, complex<float>, complex<double>).
 * @param side   'L' (left) or 'R' (right) -- which side of op(A) multiplies B.
 * @param uplo   'U' or 'L' -- which triangle of A is stored.
 * @param trans  'N' (no transpose), 'T' (transpose), 'C' (conjugate-transpose).
 * @param diag   'U' (unit diagonal) or 'N' (non-unit diagonal).
 * @param handle Handle owning the process grid and stream; every descriptor
 *               is interpreted against it.
 * @param m      Number of rows of B.
 * @param n      Number of columns of B.
 * @param d_A    Device pointer to distributed triangular matrix A.
 * @param descA  ScaLAPACK int[9] descriptor for A (must be square, mb == nb).
 * @param d_B    Device pointer to RHS / solution B.
 * @param descB  ScaLAPACK int[9] descriptor for B.
 *
 * The descriptors may describe matrices larger than the logical sub-matrix
 * (leading block anchored at global (0,0)); only the leading m x n block of B
 * and the corresponding leading block of A are referenced.
 */
template<typename T>
void ptrtrs(
    const DdlaHandle_t& handle,
    const char& side, const char& uplo, const char& trans, const char& diag,
    const int& m, const int& n,
    T* d_A, const int* descA,
    T* d_B, const int* descB
);

/**
 * @brief Apply a pivot permutation to a distributed matrix: swap rows or
 *        columns of A according to a column-cyclic pivot vector.
 *
 * Implements the pivoting applied after LU factorization (ScaLAPACK-style
 * PZLASWP).  For a permutation P = P(0)*...*P(m-2) where P(k) swaps row/col
 * k with row/col ipiv(k)-1, direc='F' applies pivots in ascending k order
 * (computing P^T*A for rows / A*P for columns) and direc='B' in descending
 * order (computing P*A for rows / A*P^T for columns).
 *
 * @tparam T   Scalar type.
 * @param direc   'F' -- forward pivoting order; 'B' -- backward.
 * @param rowcol  'R' -- pivot rows; 'C' -- pivot columns.
 * @param pivroc  'C' -- column-cyclic pivot distribution (only 'C').
 * @param m       Number of pivots (rows/columns of A to pivot).
 * @param n       For rowcol='R': number of columns in A; for rowcol='C':
 *                number of rows in A (the fixed segment length).
 * @param handle Handle owning the process grid and stream.
 * @param d_A     Device pointer to distributed matrix A.
 * @param descA   ScaLAPACK int[9] descriptor for A.
 * @param ipiv    Host array of pivot indices (1-based, length >= m).
 * @param descIP  ScaLAPACK int[9] descriptor for the pivot vector
 *                (same row distribution as A).
 * @param iwork   Workspace (unused, pass nullptr).
 *
 * The descriptor may describe a matrix larger than the logical sub-matrix
 * (leading block anchored at global (0,0)); only the leading m rows (rowcol
 * 'R') / m columns (rowcol 'C') and the leading n-column / n-row segment are
 * pivoted.
 */
template <typename T>
void plapiv(
    const DdlaHandle_t& handle,
    const char& direc, const char& rowcol, const char& pivroc,
    const int& m, const int& n,
    T* d_A, const int* descA,
    const int* ipiv, const int* descIP,
    int* iwork
);

/**
 * @brief Swap two rows or two columns between distributed matrices.
 *
 * Exchanges the segment of length N starting at (ia,ja) in A with the
 * segment starting at (ib,jb) in B.  inca=1 swaps columns, inca=m swaps rows.
 * Communication occurs only when the source and target rows/columns reside on
 * different processes.
 *
 * @tparam T        Scalar type.
 * @param handle Handle owning the process grid and stream.
 * @param N     Length of the segment to swap.
 * @param A     Device pointer to distributed matrix A.
 * @param ia    Starting global row index in A (1-based).
 * @param ja    Starting global column index in A (1-based).
 * @param descA ScaLAPACK int[9] descriptor for A.
 * @param inca  1 (swap columns) or m_A (swap rows).
 * @param B     Device pointer to distributed matrix B.
 * @param ib    Starting global row index in B (1-based).
 * @param jb    Starting global column index in B (1-based).
 * @param descB ScaLAPACK int[9] descriptor for B.
 * @param incb  Increment for B (must match inca when inca == 1).
 */
template <typename T>
void pswap(
    const DdlaHandle_t& handle,
    const int& N,
    T* A, int ia, int ja, const int* descA, const int& inca,
    T* B, int ib, int jb, const int* descB, const int& incb
);

/**
 * @brief Internal unblocked panel LU factorization for distributed matrices.
 *
 * Factors the panel starting at global column n_s with width nb_real within the
 * leading-block sub-matrix of size m-by-n (n_s + nb_real <= n).  This is the
 * inner kernel called by pgetrf to factor each diagonal block.  Outputs pivot
 * indices into ipiv (1-based).
 *
 * @tparam T   Scalar type.
 * @param m        Total rows of the logical sub-matrix (<= desc.m()).
 * @param n        Total columns of the logical sub-matrix (<= desc.n()).
 * @param nb_real  Actual width of this panel (<= nb).
 * @param d_A      Device pointer to matrix A (input/output).
 * @param n_s      Global starting column index of the panel.
 * @param handle   Handle owning the process grid and stream.
 * @param descA    ScaLAPACK int[9] descriptor for A.
 * @param ipiv     Host pivot array (output, 1-based).
 * @param info     0 on success, >0 if singular.
 */
template <typename T>
void pgetf2(
    const DdlaHandle_t& handle,
    const int& m, const int& n, const int& nb_real,
    T* d_A, const int& n_s, const int* descA,
    int* ipiv, // host
    int& info  // host
);

/**
 * @brief Alternative panel LU factorization (rank-revealing variant).
 *
 * Uses a slightly different communication pattern than pgetf2 for
 * pivot selection within a panel.
 *
 * @tparam T   Scalar type.
 * @param m        Total rows of the logical sub-matrix (<= desc.m()).
 * @param n        Total columns of the logical sub-matrix (<= desc.n()).
 * @param nb_real  Actual panel width.
 * @param d_A      Device pointer to matrix A.
 * @param n_start  Global starting column of the panel.
 * @param handle   Handle owning the process grid and stream.
 * @param descA    ScaLAPACK int[9] descriptor for A.
 * @param ipiv     Host pivot array (output).
 * @param info     0 on success.
 */
template <typename T>
void pgetf2_panel(
    const DdlaHandle_t& handle,
    const int& m, const int& n, const int& nb_real,
    T* d_A, const int& n_start, const int* descA,
    int* ipiv, // host
    int& info  // host
);

/**
 * @brief Distributed LU factorization with partial (row) pivoting:
 *        A = P * L * U.
 *
 * Factors a distributed m-by-n matrix panel by panel:
 *   1. Factor the diagonal block (pgetf2).
 *   2. Broadcast L and U factors along process rows / columns.
 *   3. Solve for the U panel (trsm).
 *   4. Update the trailing submatrix (gemm: C -= L*U).
 *
 * Requires square blocks (mb == nb).  Corresponds to ScaLAPACK
 * PZGETRF / PDGETRF.  The descriptor may describe a matrix larger than the
 * logical m-by-n sub-matrix (leading block anchored at global (0,0)); only
 * the leading m x n block is factored.
 *
 * @tparam T   Scalar type.
 * @param m        Number of rows of A (<= desc.m()).
 * @param n        Number of columns of A (<= desc.n()).
 * @param d_A      Device pointer to matrix A (input/output -- L+U factors).
 * @param handle   Handle owning the process grid and stream.
 * @param descA    ScaLAPACK int[9] descriptor for A (mb == nb required).
 * @param ipiv     Host pivot array (output, 1-based, length >= local rows).
 * @param info     0 on success, >0 if singular.
 */
template <typename T>
void pgetrf(
    const DdlaHandle_t& handle,
    const int& m, const int& n,
    T* d_A, const int* descA,
    int* ipiv, // host
    int& info  // host
);

/**
 * @brief Block LU factorization with partial pivoting within each block row.
 *
 * Computes A = P*L*U where pivoting is applied at the block level: within each
 * block column the diagonal block is factored with getrf (producing P1 A = L1 U1),
 * then the pivot is applied to the full rows (the already-factored columns as
 * well as the right panel), the U and L panels are computed via trsm, and the
 * trailing submatrix updated via gemm.  The output is a standard LU
 * factorization: each block's row swaps cover every column, including the L
 * part, so the factors can be used with any standard triangular solve.
 *
 * This is a right-looking block algorithm.  Corresponds to the block-wise
 * derivation in README.md (Experimental Routines).
 *
 * @tparam T   Scalar type.
 * @param m        Number of rows of A.
 * @param n        Number of columns of A.
 * @param d_A      Device pointer to matrix A (input/output -- L+U factors).
 * @param handle   Handle owning the process grid and stream.
 * @param descA    ScaLAPACK int[9] descriptor for A (mb == nb required).
 * @param d_ipiv   device pivot array (output, 1-based block-local offsets,
 *                 length >= local rows).
 * @param info     host info 0 on success, >0 if singular. 
 */
template <typename T>
void pgetrf_bpiv(const DdlaHandle_t& handle, const int& m, const int& n, T* d_A, const int* descA, int* d_ipiv, int& info);

/**
 * @brief Distributed LU factorization without pivoting.
 *
 * Factors a distributed m-by-n matrix A in-place as A = L * U without
 * pivoting.  L has implicit unit diagonal and is stored strictly below the
 * diagonal; U is stored on and above the diagonal.
 *
 * Uses a right-looking block algorithm: at each step the nb×nb diagonal
 * block is factored with the local getrf_nopiv, the right/lower panels are
 * computed via trsm, and the trailing submatrix is updated via gemm.  The
 * communication pattern mirrors pgetrf_bpiv, but no row pivots are applied.
 *
 * @tparam T   Scalar type.
 * @param m        Number of rows of A.
 * @param n        Number of columns of A.
 * @param d_A      Device pointer to matrix A (input/output -- L+U factors).
 * @param handle   Handle owning the process grid and stream.
 * @param descA    ScaLAPACK int[9] descriptor for A (mb == nb required).
 * @param info     host info: 0 on success, >0 if U(k,k) is exactly zero.
 */
template <typename T>
void pgetrf_nopiv(const DdlaHandle_t& handle, const int& m, const int& n, T* d_A, const int* descA, int& info);

/**
 * @brief Local LU factorization without pivoting.
 *
 * Factors a local (single-process, single-GPU) m-by-n matrix A in-place:
 * A = L * U.  L has implicit unit diagonal, stored strictly below diagonal.
 * U is stored on and above diagonal.
 *
 * Uses a right-looking block algorithm with block size nb=32:
 *   1. Panel factorization via custom getf2_nopiv_kernel.
 *   2. Solve for U panel via a triangular solve (trsm).
 *   3. Update trailing submatrix via gemm.
 *
 * @tparam T   Scalar type.
 * @param m        Number of rows of A.
 * @param n        Number of columns of A.
 * @param d_A      Device pointer to matrix A (input/output -- L+U factors).
 * @param lda      Leading dimension of A.
 * @param d_info      Device pointer to info (0 on success, k > 0 if U(k,k) is exactly zero).
 * @param ddla_handle DDLA handle (provides stream and BLAS handle).
 */
template <typename T>
void getrf_nopiv(int m, int n, T* d_A, int lda, int* d_info, const DdlaHandle_t& ddla_handle);

/**
 * @brief Distributed LU solve: solve op(A) * X = B (side='L') or
 *        X * op(A) = B (side='R') using the factors from pgetrf.
 *
 * Steps for side='L', trans='N': apply row pivots (plapiv), forward solve
 * L*Y=B (ptrtrs), backward solve U*X=Y (ptrtrs).  Other side/trans
 * combinations apply the trsm sequence in the mirrored order and apply the
 * pivot permutation on the solution side (rows for side='L', columns for
 * side='R').
 *
 * @tparam T   Scalar type.
 * @param side    'L' -- solve op(A)*X = B (B is n x nrhs);
 *                'R' -- solve X*op(A) = B (B is nrhs x n).
 * @param trans   'N', 'T' or 'C' -- operation applied to A.
 * @param n       Order of matrix A.
 * @param nrhs    Number of right-hand sides.
 * @param d_A     Device pointer to LU factors (from pgetrf).
 * @param handle  Handle owning the process grid and stream.
 * @param descA   ScaLAPACK int[9] descriptor for A.
 * @param ipiv    Host pivot array from pgetrf.
 * @param d_B     Device pointer to RHS / solution B (input/output).
 * @param descB   ScaLAPACK int[9] descriptor for B.
 */
template <typename T>
void pgetrs(
    const DdlaHandle_t& handle,
    const char& side, const char& trans, const int& n, const int& nrhs,
    T* d_A, const int* descA,
    const int* ipiv, // host
    T* d_B, const int* descB
);

/**
 * @brief Distributed LU solve without pivoting: solve op(A) * X = B
 *        (side='L') or X * op(A) = B (side='R') using the LU factors
 *        produced by pgetrf_nopiv.
 *
 * Because no pivoting is used, the solution is obtained by two triangular
 * solves in the mirrored order for side='R' / trans='T','C'.
 *
 * @tparam T   Scalar type.
 * @param side    'L' -- solve op(A)*X = B (B is n x nrhs);
 *                'R' -- solve X*op(A) = B (B is nrhs x n).
 * @param trans   'N', 'T' or 'C' -- operation applied to A.
 * @param n       Order of matrix A.
 * @param nrhs    Number of right-hand sides.
 * @param d_A     Device pointer to LU factors (from pgetrf_nopiv).
 * @param handle  Handle owning the process grid and stream.
 * @param descA   ScaLAPACK int[9] descriptor for A.
 * @param d_B     Device pointer to RHS / solution B (input/output).
 * @param descB   ScaLAPACK int[9] descriptor for B.
 */
template <typename T>
void pgetrs_nopiv(
    const DdlaHandle_t& handle,
    const char& side, const char& trans, const int& n, const int& nrhs,
    T* d_A, const int* descA,
    T* d_B, const int* descB
);

/**
 * @brief Distributed linear-system solver (driver): solve op(A) * X = B
 *        (side='L') or X * op(A) = B (side='R').
 *
 * Convenience wrapper: pgetrf (LU) + pgetrs (solve).  Corresponds to
 * ScaLAPACK PZGESV / PDGESV.
 *
 * @tparam T   Scalar type.
 * @param side    'L' -- solve op(A)*X = B (B is n x nrhs);
 *                'R' -- solve X*op(A) = B (B is nrhs x n).
 * @param trans   'N', 'T' or 'C' -- operation applied to A.
 * @param n       Order of square matrix A.
 * @param nrhs    Number of right-hand sides.
 * @param d_A     Device pointer to A (input: coefficient; output: LU factors).
 * @param handle  Handle owning the process grid and stream.
 * @param descA   ScaLAPACK int[9] descriptor for A.
 * @param d_B     Device pointer to RHS / solution B (input/output).
 * @param descB   ScaLAPACK int[9] descriptor for B.
 * @throws std::runtime_error if LU factorization fails (info != 0).
 */
template <typename T>
void pgesv(
    const DdlaHandle_t& handle,
    const char& side, const char& trans, const int& n, const int& nrhs,
    T* d_A, const int* descA,
    T* d_B, const int* descB
);

/**
 * @brief Distributed linear-system solver without pivoting (driver): solve
 *        op(A) * X = B (side='L') or X * op(A) = B (side='R').
 *
 * Convenience wrapper: pgetrf_nopiv (LU) + pgetrs_nopiv (solve).
 *
 * @tparam T   Scalar type.
 * @param side    'L' -- solve op(A)*X = B (B is n x nrhs);
 *                'R' -- solve X*op(A) = B (B is nrhs x n).
 * @param trans   'N', 'T' or 'C' -- operation applied to A.
 * @param n       Order of square matrix A.
 * @param nrhs    Number of right-hand sides.
 * @param d_A     Device pointer to A (input: coefficient; output: LU factors).
 * @param handle  Handle owning the process grid and stream.
 * @param descA   ScaLAPACK int[9] descriptor for A.
 * @param d_B     Device pointer to RHS / solution B (input/output).
 * @param descB   ScaLAPACK int[9] descriptor for B.
 * @throws std::runtime_error if LU factorization fails (info != 0).
 */
template <typename T>
void pgesv_nopiv(
    const DdlaHandle_t& handle,
    const char& side, const char& trans, const int& n, const int& nrhs,
    T* d_A, const int* descA,
    T* d_B, const int* descB
);

/**
 * @brief Distributed solve using the block LU factors from pgetrf_bpiv:
 *        solve op(A) * X = B (side='L') or X * op(A) = B (side='R').
 *
 * pgetrf_bpiv performs block-partial pivoting: each nb x nb diagonal block is
 * factored with a local getrf and its row permutation is applied to the full
 * rows (already-factored columns and right panel) with laswp in forward
 * order, yielding a standard A = P*L*U.  Its pivot array @p d_ipiv is a
 * device array holding 1-based offsets *within* each diagonal block (kept on
 * the owning process row).  The block permutations act on disjoint row sets,
 * so they commute and each block's swaps are local to one process row
 * (side='L') or process column (side='R'); pgetrs_bpiv applies them without
 * any data movement.
 *
 * @tparam T   Scalar type.
 * @param side    'L' -- solve op(A)*X = B (B is n x nrhs);
 *                'R' -- solve X*op(A) = B (B is nrhs x n).
 * @param trans   'N', 'T' or 'C' -- operation applied to A.
 * @param n       Order of matrix A.
 * @param nrhs    Number of right-hand sides.
 * @param d_A     Device pointer to LU factors (from pgetrf_bpiv).
 * @param handle  Handle owning the process grid and stream.
 * @param descA   ScaLAPACK int[9] descriptor for A.
 * @param d_ipiv  Device pivot array from pgetrf_bpiv (block-local, 1-based).
 * @param d_B     Device pointer to RHS / solution B (input/output).
 * @param descB   ScaLAPACK int[9] descriptor for B.
 */
template <typename T>
void pgetrs_bpiv(
    const DdlaHandle_t& handle,
    const char& side, const char& trans, const int& n, const int& nrhs,
    T* d_A, const int* descA,
    int* d_ipiv, // device
    T* d_B, const int* descB
);

/**
 * @brief Distributed linear-system solver using block-partial-pivoting LU
 *        (driver): solve op(A) * X = B (side='L') or X * op(A) = B
 *        (side='R').
 *
 * Convenience wrapper: pgetrf_bpiv (block LU with partial pivoting within
 * each block row) + pgetrs_bpiv (solve).
 *
 * @tparam T   Scalar type.
 * @param side    'L' -- solve op(A)*X = B (B is n x nrhs);
 *                'R' -- solve X*op(A) = B (B is nrhs x n).
 * @param trans   'N', 'T' or 'C' -- operation applied to A.
 * @param n       Order of square matrix A.
 * @param nrhs    Number of right-hand sides.
 * @param d_A     Device pointer to A (input: coefficient; output: LU factors).
 * @param handle  Handle owning the process grid and stream.
 * @param descA   ScaLAPACK int[9] descriptor for A.
 * @param d_B     Device pointer to RHS / solution B (input/output).
 * @param descB   ScaLAPACK int[9] descriptor for B.
 * @throws std::runtime_error if LU factorization fails (info != 0).
 */
template <typename T>
void pgesv_bpiv(
    const DdlaHandle_t& handle,
    const char& side, const char& trans, const int& n, const int& nrhs,
    T* d_A, const int* descA,
    T* d_B, const int* descB
);

#endif // DDLA_HAS_GPU

/**
 * @brief Distributed matrix-matrix multiplication:
 *        C := alpha * op(A) * op(B) + beta * C.
 *
 * Uses a 2D block-cyclic data distribution and broadcast of panel columns of
 * A and panel rows of B (AB-path) over MPI/NCCL/RCCL (per the DDLA_USE_CCL /
 * DDLA_USE_GPU_CPU_TUNNEL build configuration).  Supports all standard
 * transpose options:
 *   - 'N': op(X) = X
 *   - 'T': op(X) = X^T
 *   - 'C': op(X) = X^H  (conjugate-transpose)
 *
 * When transa or transb is not 'N', the descriptors must be
 * ScaLAPACK-compatible (e.g. for A^T, mb(C) == nb(A) and irsrc(C) == icsrc(A)).
 * The process grid may be rectangular.
 *
 * All three descriptors (@p descA, @p descB, @p descC) are interpreted
 * against the same @p handle (same backend and process grid).
 * CPU handles require host pointers; GPU handles require device pointers
 * allocated in the selected-accelerator memory space.  No implicit
 * migration between address spaces is performed.
 *
 * @tparam T    Scalar type.
 * @param handle   Handle owning the process grid, stream and backend.
 * @param transa   Operation applied to A ('N','T','C').
 * @param transb   Operation applied to B ('N','T','C').
 * @param m        Rows of op(A) and C.
 * @param n        Cols of op(B) and C.
 * @param k        Cols of op(A) / rows of op(B).
 * @param alpha    Scalar multiplier for A*B.
 * @param d_A      Pointer to distributed A (host for CPU, device for GPU).
 * @param descA    ScaLAPACK int[9] descriptor for A.
 * @param d_B      Pointer to distributed B (host for CPU, device for GPU).
 * @param descB    ScaLAPACK int[9] descriptor for B.
 * @param beta     Scalar multiplier for C.
 * @param d_C      Pointer to distributed C (input/output; host for CPU, device for GPU).
 * @param descC    ScaLAPACK int[9] descriptor for C.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pgemm(
    const DdlaHandle_t& handle,
    const char& transa, const char& transb,
    const int& m, const int& n, const int& k,
    const T& alpha,
    const T* d_A, const int* descA,
    const T* d_B, const int* descB,
    const T& beta,
    T* d_C, const int* descC
);

#if DDLA_HAS_GPU

/**
 * @brief Distributed matrix addition: C := alpha * op(A) + beta * op(B).
 *
 * Element-wise addition of two distributed matrices with optional transpose
 * operations.  Communication between processes is required when the data
 * distribution of op(A) differs from that of op(B) (e.g. one is transposed
 * and the other is not).
 *
 * @tparam T    Scalar type.
 * @param handle   Handle owning the process grid and stream.
 * @param transa   Operation for A ('N','T','C').
 * @param transb   Operation for B ('N','T','C').
 * @param m        Rows of C.
 * @param n        Cols of C.
 * @param alpha    Scalar multiplier for op(A).
 * @param d_A      Device pointer to distributed A.
 * @param descA    ScaLAPACK int[9] descriptor for A.
 * @param beta     Scalar multiplier for op(B).
 * @param d_B      Device pointer to distributed B.
 * @param descB    ScaLAPACK int[9] descriptor for B.
 * @param d_C      Device pointer to result C (output).
 * @param descC    ScaLAPACK int[9] descriptor for C.
 */
template <typename T>
void pgeadd(
    const DdlaHandle_t& handle,
    const char& transa, const char& transb,
    const int& m, const int& n,
    const T& alpha,
    const T* d_A, const int* descA,
    const T& beta,
    const T* d_B, const int* descB,
    T* d_C, const int* descC
);

/**
 * @brief Add a scalar to the diagonal of a distributed square matrix.
 *
 * For every global diagonal element A(i,i) with 0 <= i < n, add alpha.
 * Only the locally owned portion of the 2D block-cyclic distribution is
 * updated; no inter-process communication is required.  The descriptor may
 * describe a matrix larger than the logical leading-block sub-matrix: pass
 * n < 0 (the default) to add over the whole matrix, or a positive n to touch
 * only the leading n x n block.
 *
 * Supported combinations match LibRPA's DeviceConnector::pdam:
 *   (float,float), (double,double),
 *   (float, complex<float>), (complex<float>, complex<float>),
 *   (double, complex<double>), (complex<double>, complex<double>).
 *
 * @tparam T1  Scalar type of the value to add.
 * @tparam T2  Element type of the distributed matrix A.
 * @param handle Handle owning the process grid and stream.
 * @param alpha  Scalar to add to each diagonal element.
 * @param d_A    Device pointer to distributed matrix A (input/output).
 * @param descA  ScaLAPACK int[9] descriptor for A (may describe a matrix
 *               larger than the leading n x n sub-matrix touched).
 * @param n      Logical order of the leading sub-matrix (<= desc[DDLA_M_]);
 *               n < 0 means the whole matrix.  Default -1.
 */
template <typename T1, typename T2>
void pdam(const DdlaHandle_t& handle, const T1& alpha, T2* d_A, const int* descA, const int& n = -1);

/**
 * @brief Distributed Cholesky factorization.
 *
 * Factors a Hermitian positive-definite distributed matrix using the GPU
 * solver libraries.  Algorithm: factor diagonal block
 * (potrf), broadcast factor, solve off-diagonal (trsm), update trailing
 * submatrix via gemm/herk.  With uplo='L', computes A = L * L^H.
 * With uplo='U', computes A = U^H * U.
 *
 * @tparam T   Scalar type (float, double, complex<float>, complex<double>).
 * @param uplo     'L' or 'U' -- triangle of A to store and factor.
 * @param n        Order of A (<= desc.m(), desc.n()).
 * @param A        Device pointer to A (input: Hermitian pos-def; output: Cholesky factor).
 * @param ia       Reserved; must be 1 (1-based).  The factor operates on the
 *                 leading n x n sub-matrix anchored at global (0,0).
 * @param ja       Reserved; must be 1 (1-based).
 * @param handle   Handle owning the process grid and stream.
 * @param descA    ScaLAPACK int[9] descriptor for A (mb == nb required).
 * @param info     0 on success, >0 if not positive-definite.
 * @param is_head  Internal flag for multi-head Cholesky (default false).
 * @param location Internal row/col rearrangement index (default -1).
 * @return true if the last diagonal element needed a sign correction,
 *         false otherwise.
 *
 * The descriptor may describe a matrix larger than the logical n-by-n
 * sub-matrix (leading block anchored at global (0,0)); only the leading
 * n x n block is factored.
 */
template<typename T>
bool ppotrf(
    const DdlaHandle_t& handle,
    const char& uplo, const int& n,
    T* A, const int& ia, const int& ja, const int* descA,
    int& info, // host pointer
    bool is_head = false, int location = -1
);

/**
 * @brief Single-GPU Cholesky factorization from the bottom-right corner.
 *
 * With uplo='U', computes A = U * U^H and overwrites the upper triangle with
 * U.  With uplo='L', computes A = L^H * L and overwrites the lower triangle
 * with L.  In either mode the opposite triangle is not referenced or written.
 * These are bottom-right factorizations and therefore reverse the product
 * order used by the corresponding standard LAPACK POTRF convention.
 *
 * @tparam T      Scalar type (float, double, complex<float>, complex<double>).
 * @param uplo    'U' for A = U * U^H or 'L' for A = L^H * L.
 * @param n       Order of A.
 * @param d_A     Device pointer to A.
 * @param lda     Leading dimension of A.
 * @param info    0 on success; i > 0 identifies the failed reverse pivot.
 * @param handle  DDLA handle providing the GPU stream and BLAS handle.
 */
template <typename T>
void potrf_bottom_right(
    const char& uplo, const int& n, T* d_A, const int& lda,
    int& info, const DdlaHandle_t& handle
);

/**
 * @brief Distributed Cholesky factorization from the bottom-right corner.
 *
 * With uplo='U', computes A = U * U^H and overwrites the upper triangle with
 * U.  With uplo='L', computes A = L^H * L and overwrites the lower triangle
 * with L.  The matrix descriptor must use square blocks on a square process
 * grid; row and column source processes may differ.
 *
 * @tparam T            Scalar type (float, double, complex<float>,
 *                      complex<double>).
 * @param uplo          'U' for A = U * U^H or 'L' for A = L^H * L.
 * @param n             Order of A (<= desc.m(), desc.n()).
 * @param d_A           Device pointer to the local block-cyclic storage of A.
 * @param handle        Handle owning the process grid and stream.
 * @param descA         ScaLAPACK int[9] descriptor for the distributed matrix A.
 * @param info          0 on success; i > 0 identifies the failed global pivot.
 *
 * The descriptor may describe a matrix larger than the logical n-by-n
 * sub-matrix (leading block anchored at global (0,0)); only the leading
 * n x n block is factored.
 */
template <typename T>
void ppotrf_bottom_right(
    const DdlaHandle_t& handle,
    const char& uplo, const int& n, T* d_A,
    const int* descA, int& info
);

/**
 * @brief Distributed solve using Cholesky factorization: solve
 *        op(A) * X = B (side='L') or X * op(A) = B (side='R').
 *
 * Solves a Hermitian positive-definite system using the factor from
 * ppotrf.  For side='L', uplo='L' it applies L then L^H; the trsm order
 * is mirrored for side='R'.  Because A is Hermitian, op(A) == A for both
 * trans='N' and trans='C' (identical code path); trans='T' is not
 * supported.
 *
 * When `location` is a head-correction index (the same value passed to
 * ppotrf with is_head=true), ppotrs applies the matching permutation to B
 * -- rows for side='L', columns for side='R' -- around the solve and undoes
 * it afterward, so direct ppotrf + ppotrs users (and pposv) do not need to
 * permute B themselves.
 *
 * @tparam T   Scalar type.
 * @param side     'L' -- solve op(A)*X = B (B is n x nrhs);
 *                 'R' -- solve X*op(A) = B (B is nrhs x n).
 * @param uplo     'L' or 'U' -- triangle containing the Cholesky factor.
 * @param trans    'N' or 'C' (equivalent for Hermitian A).
 * @param n        Order of A.
 * @param nrhs     Number of right-hand sides.
 * @param d_A      Device pointer to Cholesky factor (from ppotrf).
 * @param handle   Handle owning the process grid and stream.
 * @param descA    ScaLAPACK int[9] descriptor for A.
 * @param d_B      Device pointer to RHS / solution B (input/output).
 * @param descB    ScaLAPACK int[9] descriptor for B.
 * @param is_nega  Diagonal sign-correction flag (from ppotrf return).
 * @param location Head-correction index forwarded from ppotrf; -1 (or == n)
 *                 means no B permutation.
 */
template <typename T>
void ppotrs(
    const DdlaHandle_t& handle,
    const char& side, const char& uplo, const char& trans,
    const int& n, const int& nrhs,
    T* d_A, const int* descA,
    T* d_B, const int* descB,
    bool is_nega = false, int location = -1
);

/**
 * @brief Distributed solver for Hermitian positive-definite systems
 *        (driver): solve op(A) * X = B (side='L') or X * op(A) = B
 *        (side='R') via Cholesky factorization.
 *
 * Convenience wrapper:  ppotrf + ppotrs.  Corresponds to ScaLAPACK PZPOSV.
 *
 * @tparam T   Scalar type.
 * @param side     'L' -- solve op(A)*X = B (B is n x nrhs);
 *                 'R' -- solve X*op(A) = B (B is nrhs x n).
 * @param uplo     'L' or 'U' -- triangle of A to store and factor.
 * @param trans    'N' or 'C' (equivalent for Hermitian A).
 * @param n        Order of A.
 * @param nrhs     Number of right-hand sides.
 * @param d_A      Device pointer to A (input: pos-def; output: Cholesky factor).
 * @param ia       Reserved; must be 1 (1-based).  Solves operate on the
 *                 leading n x n / n x nrhs sub-matrices anchored at (0,0).
 * @param ja       Reserved; must be 1 (1-based).
 * @param handle   Handle owning the process grid and stream.
 * @param descA    ScaLAPACK int[9] descriptor for A.
 * @param d_B      Device pointer to RHS / solution B (input/output).
 * @param ib       Reserved; must be 1 (1-based).
 * @param jb       Reserved; must be 1 (1-based).
 * @param descB    ScaLAPACK int[9] descriptor for B.
 * @param info     Output: 0 on success, >0 if not positive-definite.
 * @param is_head  Forwarded to ppotrf.
 * @param location Forwarded to ppotrf.
 */
template <typename T>
void pposv(
    const DdlaHandle_t& handle,
    const char& side, const char& uplo, const char& trans,
    const int & n, const int& nrhs,
    T* d_A, const int& ia, const int& ja, const int* descA,
    T* d_B, const int& ib, const int& jb, const int* descB,
    int& info, // host pointer
    bool is_head = false, int location = -1
);

#endif // DDLA_HAS_GPU

// ---------------------------------------------------------------------------
// Single-GPU / backend-neutral template interfaces.
//
// Declarations consolidated here from the per-routine headers, which moved
// to src/ (private). The definitions live in the corresponding src/*.cpp
// files as explicit instantiations exported from the shared library, or in
// the private headers for inline wrappers.
// ---------------------------------------------------------------------------

template <DdlaBackend Backend = default_backend_v, typename T>
void gemm(
    const DdlaHandle_t& handle,
    char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const T& beta,
    T* C, int ldc);

template <DdlaBackend Backend = default_backend_v, typename T>
void omatcopy(const DdlaHandle_t& handle, char trans, int rows, int cols,
              const T& alpha, const T* A, int lda, T* B, int ldb);

template <DdlaBackend Backend = default_backend_v, typename T>
void copy2D(const DdlaHandle_t& handle, T* dst, int dst_ld,
            const T* src, int src_ld, int rows, int cols);

template <typename T>
void gemmVbatched(
    char transA, char transB,
    int* d_m, int* d_n, int* d_k,
    T alpha,
    const T* const* d_A_array, int* d_lda,
    const T* const* d_B_array, int* d_ldb,
    T beta,
    T** d_C_array, int* d_ldc,
    int batch_count,
    const DdlaHandle_t& handle);

template <typename T>
void gemmVbatched2s(
    char transA_0, char transB_0,
    int* d_m_0, int* d_n_0, int* d_k_0,
    T alpha_0,
    const T* const* d_A_array_0, int* d_lda_0,
    const T* const* d_B_array_0, int* d_ldb_0,
    T beta_0,
    T** d_C_array_0, int* d_ldc_0,
    char transA_1, char transB_1,
    int* d_m_1, int* d_n_1, int* d_k_1,
    T alpha_1,
    const T* const* d_AB_array_1,
    int* d_lda_1, int* d_ldb_1,
    T beta_1,
    T** d_C_array_1, int* d_ldc_1,
    bool C0_left,
    int batch_count,
    const int* segment_sizes,
    int segment_count,
    const DdlaHandle_t& handle);

template <typename T>
void ptran(const DdlaHandle_t& handle,
           const T* d_A, const int* descA,
           T* d_AT, const int* descAT,
           bool conj = false);

template <DdlaBackend Backend = default_backend_v, typename T>
void transport_block(
    const DdlaHandle_t& handle,
    const char& sData, const char& trans,
    const int& m, const int& n,
    const T* d_A, const int& ia, const int& ja, const int* descA,
    T* d_block_A
);

template <DdlaBackend Backend = default_backend_v, typename T>
void random_generate(T* data, const int64_t& lengthOfData);

template <DdlaBackend Backend = default_backend_v, typename T>
void write_matrix(const T* A, const int& m, const int& n, const char* filename);

template <DdlaBackend Backend = default_backend_v, typename T>
void scal(const DdlaHandle_t& handle, int n, const T& alpha, T* x, int incx);

template <DdlaBackend Backend = default_backend_v, typename T>
void axpy(const DdlaHandle_t& handle, int n, const T& alpha,
          const T* x, int incx, T* y, int incy);

template <DdlaBackend Backend = default_backend_v, typename T>
void iamax(const DdlaHandle_t& handle, int n, const T* x, int incx, int& result);

template <DdlaBackend Backend = default_backend_v, typename T>
void geru(const DdlaHandle_t& handle, int m, int n, const T& alpha,
          const T* x, int incx, const T* y, int incy, T* A, int lda);


} // namespace ddla

#endif // DDLA_H
