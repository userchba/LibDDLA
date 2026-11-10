#ifndef DDLA_H
#define DDLA_H

#include <ddla/ddla_config.h>
#include <ddla/ddla_vector.h>
#include "ddla_desc.h"
#include <cstdint>
#include <complex>
#include <type_traits>
#include <stdexcept>
#include <string>

namespace ddla{

/**
 * @brief Scale rows or columns by a diagonal vector in backend memory.
 * Compute C = diag(x)*A for side='L', or C = A*diag(x) for side='R'.
 * A and C are column-major m-by-n matrices. x has m logical entries for L
 * and n for R. For incx>=0, entry i is x[i*incx]. For incx<0 it is
 * x[(length-1-i)*abs(incx)]. incx=0 repeats x[0].
 * A, x and C reside in the handle's memory domain. The four standard scalar
 * types are supported. Complex matrices also accept a real diagonal of the
 * matching precision. No conjugation is applied. A==C is supported when
 * lda==ldc. Other matrix overlap and output/diagonal overlap are rejected.
 * Empty outputs permit null pointers. Output padding and inputs are preserved
 * except for the explicitly requested in-place matrix update.
 * Execution is ordered on the handle stream and requires no DDLA workspace.
 */
template <DdlaBackend Backend = default_backend_v, typename T, typename U>
void dgmm(const DdlaHandle_t& handle, char side, int m, int n,
    const T* A, int lda, const U* x, int incx, T* C, int ldc);

/**
 * @brief Distributed diagonal scaling with the full diagonal on every rank.
 * Compute the leading m-by-n part of C using dgmm's side, type and stride
 * conventions. x is replicated in backend memory on every rank. Each replica
 * contains the same logical diagonal, with length m for L or n for R.
 * A and C use identical block sizes and source coordinates on the handle's
 * grid. Larger parent descriptors and different valid leading dimensions are
 * supported. Only the leading region is updated. In-place operation follows
 * dgmm's rules. Ranks without active entries permit null matrix/vector pointers.
 * Validation is collective and global metadata must agree. The caller supplies
 * matching diagonal values. Execution uses local block strips without matrix
 * communication, DDLA allocation or workspace and follows the handle stream.
 */
template <DdlaBackend Backend = default_backend_v, typename T, typename U>
void pdgmm(const DdlaHandle_t& handle, char side, int m, int n,
    const T* A, const int* descA, const U* x, int incx, T* C, const int* descC);



/// Real counterpart of a scalar type: `real_type<double>::type == double`,
/// `real_type<complex<double>>::type == double`.  Used for eigenvalue arrays
/// and residual norms of Hermitian problems.
template <typename T> struct real_type            { using type = T; };
template <typename U> struct real_type<std::complex<U>> { using type = U; };

// ---------------------------------------------------------------------------
// Distributed triangular solves, LU and Cholesky factorizations/solvers and their helpers (ptrtrs ... pdam, below).
// Like pgemm, every routine is templated on the backend and runs on both: `pgetrf<DdlaBackend::CPU>(handle, ...)` for a
// CPU handle, `pgetrf<DdlaBackend::GPU>(handle, ...)` for a GPU one.  The default is the build's default backend, so a
// plain `pgetrf(handle, ...)` deduces the scalar type as before.  The template backend must equal the handle's backend
// (anything else throws).  Pointers in the parameter documentation below refer to the handle's own
// memory: host memory for a CPU handle, device memory for a GPU handle.
// ---------------------------------------------------------------------------


/**
 * @brief Distributed triangular solve: B := op(A)^{-1} * B    (side='L')
 *                                     B := B * op(A)^{-1}    (side='R').
 *
 * Solves a triangular system on a distributed matrix using block-cyclic
 * data distribution and MPI (CPU handles) or MPI/NCCL/RCCL (GPU handles) communication (per the DDLA_USE_CCL /
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
 * @param A    Pointer in the selected backend to distributed triangular matrix A.
 * @param descA  ScaLAPACK int[9] descriptor for A (must be square, mb == nb).
 * @param B    Pointer in the selected backend to RHS / solution B.
 * @param descB  ScaLAPACK int[9] descriptor for B: same MB as A, and the
 *               solved dimension distributed like A's (side 'L': equal RSRC;
 *               side 'R': equal NB and CSRC).  Other layouts throw.
 *
 * The descriptors may describe matrices larger than the logical sub-matrix
 * (leading block anchored at global (0,0)); only the leading m x n block of B
 * and the corresponding leading block of A are referenced.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void ptrtrs(
    const DdlaHandle_t& handle,
    char side, char uplo, char trans, char diag,
    int m, int n,
    T* A, const int* descA,
    T* B, const int* descB
);

/**
 * @brief Distributed triangular multiply: B := alpha * op(A) * B (side='L')
 *                                      or B := alpha * B * op(A) (side='R').
 * @tparam Backend Must match the handle's backend (CPU or GPU).
 * @tparam T float, double, complex<float> or complex<double>.
 * @param handle Handle owning the process grid and execution stream.
 * @param side 'L' or 'R'.
 * @param uplo 'U' or 'L': the stored triangle of A.
 * @param trans 'N', 'T' or 'C': no transpose, transpose or conjugate transpose.
 * @param diag 'N' or 'U': stored or implicit unit diagonal.
 * @param m Number of rows of B.
 * @param n Number of columns of B.
 * @param alpha Scalar multiplier. Zero sets B to zero without reading A or B.
 * @param A Read-only triangular matrix; must not overlap B.
 * @param descA ScaLAPACK int[9] descriptor. A has square blocks.
 * @param B Matrix overwritten by the result, in the handle's backend memory.
 * @param descB ScaLAPACK int[9] descriptor. Its row block size (left) or column
 *              block size (right) must match A's block size. Source coordinates
 *              may differ; the other block size is unrestricted.
 *
 * Flags are case-insensitive. Only the leading m x n block of B and the
 * leading order x order block of A (order=m for left, n for right) participate;
 * larger descriptors and padded leading dimensions are supported. The unused
 * triangle and implicit unit diagonal do not contribute to the result.
 * Empty matrices leave B untouched. Invalid flags, dimensions or block sizes
 * throw std::invalid_argument, including in Release builds. All ranks in the
 * handle's grid must call with the same global arguments.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void ptrmm(
    const DdlaHandle_t& handle, char side, char uplo, char trans, char diag,
    int m, int n, const T& alpha,
    const T* A, const int* descA, T* B, const int* descB
);

/**
 * @brief Distributed Hermitian product: C := alpha*A*B + beta*C (side 'L'),
 *        or C := alpha*B*A + beta*C (side 'R'). Real types use symmetry.
 *
 * Only the selected triangle of A is read; the opposite triangle and the
 * imaginary diagonal components are never referenced. A and B are unchanged.
 * C is a full general matrix. Only the leading logical submatrices are used;
 * descriptor padding is untouched. A has order m (left) or n (right).
 *
 * A uses square blocks matching the multiplied dimension of B and C. B/C
 * block shapes must match; all three source coordinates may differ. C must
 * not overlap either referenced input. alpha == 0 reads no A/B; beta == 0
 * reads no old C. Empty output or alpha == 0, beta == 1 is a no-op and permits
 * null matrix pointers. Otherwise pointers are required where local entries
 * are referenced. Flags, dimensions and coefficient modes agree collectively.
 *
 * @tparam Backend CPU or GPU; must match the handle.
 * @tparam T float, double, complex<float> or complex<double>.
 * @param side 'L' or 'R' (case-insensitive).
 * @param uplo 'U' or 'L' (case-insensitive), the stored triangle of A.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void phemm(const DdlaHandle_t& handle, char side, char uplo, int m, int n,
    const T& alpha, const T* A, const int* descA,
    const T* B, const int* descB, const T& beta, T* C, const int* descC);
/** @brief Mixed real/complex product with matching float/double precision.
 * Flags, storage and zero coefficients follow the same-type operation.
 * Alpha, beta and C are complex. A real structured input is symmetric;
 * a complex Hermitian input ignores its imaginary diagonal.
 */
template <DdlaBackend Backend = default_backend_v, typename R>
void phemm(const DdlaHandle_t& handle, char side, char uplo, int m, int n,
    const std::complex<R>& alpha, const R* A, const int* descA,
    const std::complex<R>* B, const int* descB, const std::complex<R>& beta,
    std::complex<R>* C, const int* descC);
template <DdlaBackend Backend = default_backend_v, typename R>
void phemm(const DdlaHandle_t& handle, char side, char uplo, int m, int n,
    const std::complex<R>& alpha, const std::complex<R>* A, const int* descA,
    const R* B, const int* descB, const std::complex<R>& beta,
    std::complex<R>* C, const int* descC);


/**
 * @brief Hermitian product updating one triangle of a general square C.
 * C(uploC) := (alpha*A*B + beta*C)(uploC) for side L, or
 * (alpha*B*A + beta*C)(uploC) for side R. uploA selects the stored
 * Hermitian triangle of A. Its complex imaginary diagonal is ignored.
 * B and C are general matrices, and complex C diagonals are unrestricted.
 * Only the leading n-by-n regions are used. All descriptors use matching
 * square blocks. Source coordinates may differ and the grid may be rectangular.
 * Opposite C triangle, padding and both inputs remain untouched. alpha=0
 * reads no A/B, beta=0 reads no old C, and n=0 or alpha=0,beta=1 is a no-op.
 * Flags are case-insensitive. Four real/complex scalar types are supported.
 * All ranks call collectively and validation precedes matrix communication.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void phemmt(const DdlaHandle_t& handle,char side,char uploA,char uploC,int n,
    const T& alpha,const T* A,const int* descA,const T* B,const int* descB,
    const T& beta,T* C,const int* descC);
/** @brief Mixed real/complex product with matching float/double precision.
 * Flags, storage and zero coefficients follow the same-type operation.
 * Alpha, beta and C are complex. A real structured input is symmetric;
 * a complex Hermitian input ignores its imaginary diagonal.
 */
template <DdlaBackend Backend = default_backend_v, typename R>
void phemmt(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n,
    const std::complex<R>& alpha, const R* A, const int* descA,
    const std::complex<R>* B, const int* descB, const std::complex<R>& beta,
    std::complex<R>* C, const int* descC);
template <DdlaBackend Backend = default_backend_v, typename R>
void phemmt(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n,
    const std::complex<R>& alpha, const std::complex<R>* A, const int* descA,
    const R* B, const int* descB, const std::complex<R>& beta,
    std::complex<R>* C, const int* descC);


/**
 * @brief Distributed symmetric product, with the same layout and collective
 * contract as phemm. Complex symmetry uses transpose, not conjugate transpose;
 * the diagonal is complex. Only uplo of A is read; C is fully updated.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void psymm(const DdlaHandle_t& handle, char side, char uplo, int m, int n,
    const T& alpha, const T* A, const int* descA,
    const T* B, const int* descB, const T& beta, T* C, const int* descC);

/**
 * @brief Symmetric product updating only uploC of a general square C.
 * C(uploC) := (alpha*A*B + beta*C)(uploC) for side L, or
 * (alpha*B*A + beta*C)(uploC) for side R. Only uploA of symmetric A is read.
 * Complex A is reflected without conjugation and its full diagonal is used.
 * B and C are general n-by-n matrices. Opposite C entries and padding are
 * preserved. C must be disjoint from A/B when alpha is nonzero.
 * Flags, descriptors, zero coefficients and collective checks follow phemmt.
 * Only leading n-by-n regions are used. Backend and four scalar types follow
 * psymm. side accepts L/R and both triangles accept U/L, case-insensitively.
 */
template <DdlaBackend Backend = default_backend_v,typename T>
void psymmt(const DdlaHandle_t& handle,char side,char uploA,char uploC,int n,
    const T& alpha,const T* A,const int* descA,const T* B,const int* descB,
    const T& beta,T* C,const int* descC);

/**
 * @brief Distributed general matrix product updating one output triangle.
 *
 * Compute uplo='U' or 'L' of C := alpha*op(A)*op(B) + beta*C.
 * op(A) is n-by-k and op(B) is k-by-n. A and B are general matrices;
 * no symmetry or Hermitian structure is assumed. The opposite triangle,
 * entries outside the leading logical regions, and padding are untouched.
 *
 * All descriptors use the same square block size; source coordinates may
 * differ and the process grid may be rectangular. C must not overlap A/B.
 * All ranks call collectively with the same global arguments and descriptors
 * (except local leading dimensions). CPU/GPU pointers match the handle.
 *
 * beta=0 reads no old C; alpha=0 or k=0 reads no A/B. n=0 and
 * alpha=0,beta=1 are no-ops. Complex alpha, beta and diagonal entries are
 * unrestricted. Flags are case-insensitive; transa/transb are N/T/C.
 * Invalid arguments are rejected collectively before matrix communication.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pgemmt(const DdlaHandle_t& handle, char uplo,
    char transa, char transb, int n, int k,
    const T& alpha, const T* A, const int* descA,
    const T* B, const int* descB, const T& beta, T* C, const int* descC);
/** @brief Mixed real/complex product with matching float/double precision.
 * Flags, storage and zero coefficients follow the same-type operation.
 * Alpha, beta and C are complex. A real structured input is symmetric;
 * a complex Hermitian input ignores its imaginary diagonal.
 */
template <DdlaBackend Backend = default_backend_v, typename R>
void pgemmt(const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k,
    const std::complex<R>& alpha, const R* A, const int* descA,
    const std::complex<R>* B, const int* descB, const std::complex<R>& beta,
    std::complex<R>* C, const int* descC);
template <DdlaBackend Backend = default_backend_v, typename R>
void pgemmt(const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k,
    const std::complex<R>& alpha, const std::complex<R>* A, const int* descA,
    const R* B, const int* descB, const std::complex<R>& beta,
    std::complex<R>* C, const int* descC);


/**
 * @brief Distributed symmetric rank-k update of only uplo of C.
 * C := alpha*A*A^T + beta*C (N), or alpha*A^T*A + beta*C (T).
 * All descriptors use the same square block size; source coordinates may
 * differ. Leading n-by-k (N) or k-by-n (T) input and n-by-n output regions
 * are used. C must not overlap referenced inputs. Opposite triangle and
 * padding remain untouched. alpha==0 or k==0 reads no inputs; beta==0 reads
 * no old C. Flags are case-insensitive. Real types also accept C as T;
 * complex symmetric operations accept N/T only. Validation is collective.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void psyrk(const DdlaHandle_t& handle, char uplo, char trans, int n, int k,
    const T& alpha, const T* A, const int* descA,
    const T& beta, T* C, const int* descC);

/**
 * @brief Symmetric rank-2k update, with psyrk's layout/storage contract.
 * C := alpha*(A*B^T+B*A^T)+beta*C (N), or
 * alpha*(A^T*B+B^T*A)+beta*C (T). A and B have the same logical shape.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void psyr2k(const DdlaHandle_t& handle, char uplo, char trans, int n, int k,
    const T& alpha, const T* A, const int* descA, const T* B, const int* descB,
    const T& beta, T* C, const int* descC);

/**
 * @brief Hermitian rank-k update of only uplo of C (complex types).
 * C := alpha*A*A^H+beta*C (N), or alpha*A^H*A+beta*C (C).
 * Layout, non-overlap and inactive-input rules are those of psyrk. alpha
 * and beta are real; imaginary input diagonal components of C are ignored
 * and the resulting diagonal is real. The opposite triangle is untouched.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pherk(const DdlaHandle_t& handle, char uplo, char trans, int n, int k,
    const typename real_type<T>::type& alpha, const T* A, const int* descA,
    const typename real_type<T>::type& beta, T* C, const int* descC);

/**
 * @brief Hermitian rank-2k update, with pherk's layout/storage contract.
 * C := alpha*A*B^H+conj(alpha)*B*A^H+beta*C (N), or
 * alpha*A^H*B+conj(alpha)*B^H*A+beta*C (C). beta is real.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pher2k(const DdlaHandle_t& handle, char uplo, char trans, int n, int k,
    const T& alpha, const T* A, const int* descA, const T* B, const int* descB,
    const typename real_type<T>::type& beta, T* C, const int* descC);

/** @brief Local symmetric matrix product for real or complex scalars. */
template <DdlaBackend Backend = default_backend_v, typename T>
void symm(const DdlaHandle_t& handle, char side, char uplo, int m, int n,
    const T& alpha, const T* A, int lda, const T* B, int ldb,
    const T& beta, T* C, int ldc);

/** @brief Local symmetric rank-2k update for real or complex scalars. */
template <DdlaBackend Backend = default_backend_v, typename T>
void syr2k(const DdlaHandle_t& handle, char uplo, char trans, int n, int k,
    const T& alpha, const T* A, int lda, const T* B, int ldb,
    const T& beta, T* C, int ldc);


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
 * @param A     Pointer in the selected backend to distributed matrix A.
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
template <DdlaBackend Backend = default_backend_v, typename T>
void plapiv(
    const DdlaHandle_t& handle,
    char direc, char rowcol, char pivroc,
    int m, int n,
    T* A, const int* descA,
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
 * @param A     Pointer in the selected backend to distributed matrix A.
 * @param ia    Starting global row index in A (1-based).
 * @param ja    Starting global column index in A (1-based).
 * @param descA ScaLAPACK int[9] descriptor for A.
 * @param inca  1 (swap columns) or m_A (swap rows).
 * @param B     Pointer in the selected backend to distributed matrix B.
 * @param ib    Starting global row index in B (1-based).
 * @param jb    Starting global column index in B (1-based).
 * @param descB ScaLAPACK int[9] descriptor for B.
 * @param incb  Increment for B (must match inca when inca == 1).
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pswap(
    const DdlaHandle_t& handle,
    int N,
    T* A, int ia, int ja, const int* descA, int inca,
    T* B, int ib, int jb, const int* descB, int incb
);

/**
 * @brief Gather a distributed submatrix into a column-major buffer on one rank.
 *
 * Copies A(ia:ia+m-1,ja:ja+n-1), with one-based global ia/ja, to dst(0:m,0:n).
 * All ranks of the handle communicator call with consistent geometry and root.
 * root is a rank of that communicator, independent of process-grid ordering.
 * A and dst are device pointers for GPU handles and host pointers for CPU handles.
 * Only root uses dst/lddst; lddst >= m for a nonempty region. Ranks owning no
 * selected entries may pass null A. Non-root ranks may pass null dst.
 * Empty regions access no data. Source entries and destination padding are preserved.
 * GPU work is ordered on the handle stream; synchronize before host access.
 * Temporary packing storage is managed internally. Partial overlap is unsupported;
 * an exact local-view alias is allowed when root owns the complete selected region.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void gather(const DdlaHandle_t& handle, int m, int n,
            const T* A, int ia, int ja, const int* descA,
            int root, T* dst, int lddst);

/**
 * @brief Scatter a root column-major buffer into a distributed submatrix.
 *
 * Copies src(0:m,0:n) to A(ia:ia+m-1,ja:ja+n-1), with one-based global ia/ja.
 * Participation, backend memory, root numbering and alias rules follow gather.
 * Only root uses src/ldsrc; ldsrc >= m for a nonempty region. Other ranks may pass
 * null src. Only selected A entries are modified; padding and src are preserved.
 * Empty regions access no data. GPU work is ordered on the handle stream.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void scatter(const DdlaHandle_t& handle, int m, int n,
             T* A, int ia, int ja, const int* descA,
             int root, const T* src, int ldsrc);

/**
 * @brief Distributed LU factorization with partial (row) pivoting:
 *        A = P * L * U.
 *
 * Factors a distributed m-by-n matrix panel by panel:
 *   1. Factor the active panel.
 *   2. Broadcast L and U factors along process rows / columns.
 *   3. Solve for the U panel (trsm).
 *   4. Update the trailing submatrix (gemm: C -= L*U).
 *
 * Each block-column panel uses inner subpanels and BLAS-3 updates.
 * DDLA_LU_PANEL=inplace (default) factors each inner subpanel in the distributed
 * matrix; gather uses local getrf on its diagonal owner. Communication workspaces
 * are reused across panels. All ranks must agree on the selector. No factorization or
 * pivot bitwise equivalence between the methods is promised.
 *
 * Requires square blocks (mb == nb).  Corresponds to ScaLAPACK
 * PZGETRF / PDGETRF. ia,ja are one-based global indices; m,n are extents of
 * A(ia:ia+m-1,ja:ja+n-1). Independent, unaligned offsets are supported. Only
 * this region is changed, including its previously computed L columns when
 * later pivots are applied. Empty regions access no data. All handle ranks
 * must provide consistent region and descriptor metadata.
 *
 * @tparam T   Scalar type.
 * @param m        Number of selected rows.
 * @param n        Number of selected columns.
 * @param A      Pointer in the selected backend to matrix A (input/output -- L+U factors).
 * @param handle   Handle owning the process grid and stream.
 * @param ia      First selected global row (one-based).
 * @param ja      First selected global column (one-based).
 * @param descA    ScaLAPACK int[9] descriptor for A (mb == nb required).
 * @param ipiv     Host row-distributed pivots indexed by local descriptor row; values are
 *                absolute global one-based pivot rows. Other entries stay unchanged.
 * @param info     0 on success; >0 is the failing one-based position within the selected submatrix (early stop).
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pgetrf(
    const DdlaHandle_t& handle,
    int m, int n,
    T* A, int ia, int ja, const int* descA,
    int* ipiv, // host
    int& info  // host
);

/**
 * @brief Distributed LU factorization without pivoting: A = L * U.
 *
 * Right-looking block algorithm: at each step the nb x nb diagonal block is
 * factored with the local getrf_nopiv kernel, the right/lower panels are
 * computed via trsm, and the trailing submatrix is updated via gemm. The
 * communication pattern mirrors pgetrf_bpiv, but no row pivots are applied:
 * L has an implicit unit diagonal and is stored strictly below the diagonal,
 * U on and above it. Requires square blocks (mb == nb). Factors the whole
 * described matrix (no sub-matrix offsets). Corresponds to the no-pivot
 * variant of ScaLAPACK PxGETRF.
 *
 * @tparam T   Scalar type.
 * @param m        Number of rows of A.
 * @param n        Number of columns of A.
 * @param A      Pointer in the selected backend to matrix A (input/output -- L+U factors).
 * @param handle   Handle owning the process grid and stream.
 * @param descA    ScaLAPACK int[9] descriptor for A (mb == nb required).
 * @param info     0 on success; >0 if U(k,k) is exactly zero.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pgetrf_nopiv(
    const DdlaHandle_t& handle,
    int m, int n,
    T* A, const int* descA,
    int& info  // host
);

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
 * @param A     Pointer in the selected backend to LU factors (from pgetrf).
 * @param handle  Handle owning the process grid and stream.
 * @param descA   ScaLAPACK int[9] descriptor for A.
 * @param ipiv    Host pivot array from pgetrf.
 * @param B     Pointer in the selected backend to RHS / solution B (input/output).
 * @param descB   ScaLAPACK int[9] descriptor for B (same MB as A; rows for
 *                side 'L' / columns for side 'R' distributed like A's).
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pgetrs(
    const DdlaHandle_t& handle,
    char side, char trans, int n, int nrhs,
    T* A, const int* descA,
    const int* ipiv, // host
    T* B, const int* descB
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
 * @param A     Pointer in the selected backend to LU factors (from pgetrf_nopiv).
 * @param handle  Handle owning the process grid and stream.
 * @param descA   ScaLAPACK int[9] descriptor for A.
 * @param B     Pointer in the selected backend to RHS / solution B (input/output).
 * @param descB   ScaLAPACK int[9] descriptor for B.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pgetrs_nopiv(
    const DdlaHandle_t& handle,
    char side, char trans, int n, int nrhs,
    T* A, const int* descA,
    T* B, const int* descB
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
 * @param A     Pointer in the selected backend to A (input: coefficient; output: LU factors).
 * @param handle  Handle owning the process grid and stream.
 * @param descA   ScaLAPACK int[9] descriptor for A.
 * @param B     Pointer in the selected backend to RHS / solution B (input/output).
 * @param descB   ScaLAPACK int[9] descriptor for B.
 * @throws std::runtime_error if LU factorization fails (info != 0).
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pgesv(
    const DdlaHandle_t& handle,
    char side, char trans, int n, int nrhs,
    T* A, const int* descA,
    T* B, const int* descB
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
 * @param A     Pointer in the selected backend to A (input: coefficient; output: LU factors).
 * @param handle  Handle owning the process grid and stream.
 * @param descA   ScaLAPACK int[9] descriptor for A.
 * @param B     Pointer in the selected backend to RHS / solution B (input/output).
 * @param descB   ScaLAPACK int[9] descriptor for B.
 * @throws std::runtime_error if LU factorization fails (info != 0).
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pgesv_nopiv(
    const DdlaHandle_t& handle,
    char side, char trans, int n, int nrhs,
    T* A, const int* descA,
    T* B, const int* descB
);

/**
 * @brief Distributed block LU factorization with partial pivoting within
 *        each block row: A = P * L * U.
 *
 * Right-looking block algorithm. Each nb x nb diagonal block is factored with
 * the local getrf (partial pivoting inside the block, P1 A_block = L1 U1);
 * the block permutation is applied to the full rows -- the already-factored
 * columns as well as the right panel -- the U and L panels are computed via
 * trsm, and the trailing submatrix is updated via gemm. Every block's row
 * swaps cover every column, including the L part, so the output is a standard
 * LU factorization usable with any standard triangular solve.
 *
 * The pivot array @p pivots is a backend array (host for CPU, device for GPU)
 * holding 1-based offsets within each diagonal block, one entry per local row
 * of the block's process row; the block permutations act on disjoint row sets,
 * so pgetrs_bpiv applies them without any data movement. Requires square
 * blocks (mb == nb). Factors the whole described matrix (no sub-matrix
 * offsets).
 *
 * @tparam T   Scalar type.
 * @param m        Number of rows of A.
 * @param n        Number of columns of A.
 * @param A      Pointer in the selected backend to matrix A (input/output -- L+U factors).
 * @param handle   Handle owning the process grid and stream.
 * @param descA    ScaLAPACK int[9] descriptor for A (mb == nb required).
 * @param pivots   Pivot array in the selected backend (output, 1-based block-local offsets, length >= local rows).
 * @param info     0 on success; >0 if a diagonal block is singular.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pgetrf_bpiv(
    const DdlaHandle_t& handle,
    int m, int n,
    T* A, const int* descA,
    int* pivots, // host for CPU, device for GPU
    int& info  // host
);

/**
 * @brief Distributed solve using the block LU factors from pgetrf_bpiv:
 *        solve op(A) * X = B (side='L') or X * op(A) = B (side='R').
 *
 * pgetrf_bpiv performs block-partial pivoting: each nb x nb diagonal block is
 * factored with a local getrf and its row permutation is applied to the full
 * rows (already-factored columns and right panel) with laswp in forward
 * order, yielding a standard A = P*L*U.  Its pivot array @p pivots is a
 * backend array holding 1-based offsets *within* each diagonal block (kept on
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
 * @param A     Pointer in the selected backend to LU factors (from pgetrf_bpiv).
 * @param handle  Handle owning the process grid and stream.
 * @param descA   ScaLAPACK int[9] descriptor for A.
 * @param pivots  Pivot array in the selected backend from pgetrf_bpiv (block-local, 1-based).
 * @param B     Pointer in the selected backend to RHS / solution B (input/output).
 * @param descB   ScaLAPACK int[9] descriptor for B.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pgetrs_bpiv(
    const DdlaHandle_t& handle,
    char side, char trans, int n, int nrhs,
    T* A, const int* descA,
    int* pivots, // host for CPU, device for GPU
    T* B, const int* descB
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
 * @param A     Pointer in the selected backend to A (input: coefficient; output: LU factors).
 * @param handle  Handle owning the process grid and stream.
 * @param descA   ScaLAPACK int[9] descriptor for A.
 * @param B     Pointer in the selected backend to RHS / solution B (input/output).
 * @param descB   ScaLAPACK int[9] descriptor for B.
 * @throws std::runtime_error if LU factorization fails (info != 0).
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pgesv_bpiv(
    const DdlaHandle_t& handle,
    char side, char trans, int n, int nrhs,
    T* A, const int* descA,
    T* B, const int* descB
);


/**
 * @brief Distributed matrix-matrix multiplication:
 *        C := alpha * op(A) * op(B) + beta * C.
 *
 * Uses a 2D block-cyclic data distribution. Automatically selects AB (keep
 * C in place), BC (keep A in place), or AC (keep B in place) using estimated
 * communication volume. BC/AC communicate thin panels and reduce partial
 * products to C's owners; the output remains distributed, not replicated.
 * Communication uses MPI/NCCL/RCCL according to the build configuration.
 * Supports all standard transpose options:
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
 * @param A      Pointer to distributed A (host for CPU, device for GPU).
 * @param descA    ScaLAPACK int[9] descriptor for A.
 * @param B      Pointer to distributed B (host for CPU, device for GPU).
 * @param descB    ScaLAPACK int[9] descriptor for B.
 * @param beta     Scalar multiplier for C.
 * @param C      Pointer to distributed C (input/output; host for CPU, device for GPU).
 * @param descC    ScaLAPACK int[9] descriptor for C.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pgemm(
    const DdlaHandle_t& handle,
    char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const T* A, const int* descA,
    const T* B, const int* descB,
    const T& beta,
    T* C, const int* descC
);


/**
 * @brief Distributed Householder QR factorization, A = Q R (ScaLAPACK PxGEQRF).
 *
 * Factorizes the leading m x n block, k = min(m,n). R occupies the upper
 * trapezoid and the Householder vectors lie below its diagonal, with implicit
 * unit entries. Q = H(1)...H(k), where H(j) = I - tau(j) v(j) v(j)^H.
 * The complex diagonal of R is real.
 *
 * A uses backend memory. tau uses HOST memory, indexed by local column and
 * replicated across process rows, with storage for at least LOCc(n) entries.
 * Entries belonging to columns without reflectors are preserved. Descriptors
 * may describe larger parent matrices and use unequal row/column block sizes
 * and arbitrary valid process sources. Parent entries and padding are preserved.
 *
 * DDLA_QR_PANEL selects the gathered panel (gather, default) or the in-place
 * distributed panel (inplace). All ranks must select the same policy and
 * factorization. The gathered GPU panel follows the vendor QR sign convention.
 *
 * @tparam Backend CPU or GPU, matching the handle.
 * @tparam T float, double, complex<float> or complex<double>.
 * @param handle Handle owning the process grid and stream.
 * @param m Rows of the leading block, 0 <= m <= descA[M_].
 * @param n Columns of the leading block, 0 <= n <= descA[N_].
 * @param A Distributed matrix, overwritten with R and reflector vectors.
 * @param descA ScaLAPACK int[9] descriptor for A.
 * @param tau HOST reflector scalars, indexed by local column.
 * @param info HOST status, zero on success. Invalid arguments throw collectively.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pgeqrf(const DdlaHandle_t& handle, int m, int n,
            T* A, const int* descA, T* tau, int& info);

/**
 * @brief Solve min ||A_original X - B_original|| using pgeqrf factors.
 *
 * Requires m >= n >= 0, nrhs >= 0 and full column rank. A is the read-only
 * m-by-n QR factorization returned by pgeqrf. tau is read-only HOST
 * storage with the same local-column indexing and process-row replication
 * as pgeqrf's output. No explicit Q is formed.
 *
 * B's leading m-by-nrhs region is overwritten by Q^H B, then its first n
 * rows are replaced by the least-squares solution X. Rows n through m-1
 * retain the transformed residual. Parent entries and padding are preserved.
 * A needs square blocks. B must share A's row block size and row source,
 * while B's column block size and source may differ. A and B use backend
 * memory and B must be disjoint from A and tau. Descriptors may be larger
 * than the logical matrices. n=0 or nrhs=0 leaves B untouched and permits
 * null data pointers. info=0 on return. Invalid arguments throw collectively.
 * Rank deficiency is not detected, as in the underlying triangular solve.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pgeqrs(const DdlaHandle_t& handle, int m, int n, int nrhs,
            const T* A, const int* descA, const T* tau,
            T* B, const int* descB, int& info);

/**
 * @brief Compute a real bidiagonal SVD B = U diag(s) V^T by distributed divide and conquer.
 * @tparam T float or double. d, e and s are host vectors on every rank.
 * @param uplo 'U': upper bidiagonal; 'L': lower bidiagonal.
 * @param n Order of the bidiagonal matrix.
 * @param d Length n diagonal, destroyed on exit.
 * @param e Length max(0,n-1) off-diagonal, destroyed on exit.
 * @param s Length n singular values, descending and replicated on every rank.
 * @param U This rank's contiguous rows of all n left singular-vector columns.
 * @param V This rank's contiguous rows of all n right singular-vector columns,
 * returning V rather than V^T. Both factors are column-major in backend memory.
 * Communicator rank q among P ranks owns n/P+(q<n%P) rows starting at
 * q*(n/P)+min(q,n%P). For n>0, U and V must be nonnull on every rank.
 * @param ld Local leading dimension of both factors, at least max(1, local rows).
 * Padding and storage beyond the first n columns are preserved.
 * @param info Zero on success. Invalid arguments or numerical failure throw.
 * All ranks participate, including ranks without rows. Both factors remain
 * row-distributed throughout the solve and on return; no descriptor conversion
 * or full replicated factor is used. Scalar vectors and bounded panels may be replicated.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pbdsdc(const DdlaHandle_t& handle, char uplo, int n,
            T* d, T* e, T* s, T* U, T* V, int ld, int& info);

// Bidiagonalization and SVD interfaces used by the registered solver sources.
template <DdlaBackend Backend = default_backend_v, typename T>
void pgebrd(const DdlaHandle_t& handle, int m, int n,
            T* A, const int* descA,
            typename real_type<T>::type* d, typename real_type<T>::type* e,
            T* tauq, T* taup,
            int& info);

template <DdlaBackend Backend = default_backend_v, typename T>
void pgesvd(const DdlaHandle_t& handle,
            char jobu, char jobvt,
            int m, int n,
            T* A, const int* descA,
            typename real_type<T>::type* s,
            T* U, const int* descU,
            T* VT, const int* descVT,
            int& info);

template <DdlaBackend Backend = default_backend_v, typename T>
void punmbr(const DdlaHandle_t& handle,
            char vect, char side, char trans,
            int m, int n, int k,
            const T* A, const int* descA,
            const T* tau,
            T* C, const int* descC,
            int& info);

/**
 * @brief Distributed Householder QL factorization, A = Q L (ScaLAPACK PxGEQLF).
 *
 * Factorizes the leading m x n block, k = min(m,n), from right to left.
 * L occupies the lower trapezoid, whose diagonal has row index m-n+j for
 * column j. Reflector vectors lie above it, with implicit unit entries, and
 * Q = H(k)...H(1). The complex diagonal of L is real.
 * For m < n, the last k columns carry reflectors. tau uses their original
 * local-column indices, so storage for LOCc(n) entries is required.
 * Tau entries outside those columns are preserved.
 *
 * Backend memory, descriptor, parent-storage, collective validation and
 * DDLA_QR_PANEL conventions follow pgeqrf.
 *
 * @param handle Handle owning the process grid and stream.
 * @param m Rows of the leading block, 0 <= m <= descA[M_].
 * @param n Columns of the leading block, 0 <= n <= descA[N_].
 * @param A Distributed matrix, overwritten with L and reflector vectors.
 * @param descA ScaLAPACK int[9] descriptor for A.
 * @param tau HOST reflector scalars, indexed by local column.
 * @param info HOST status, zero on success. Invalid arguments throw collectively.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pgeqlf(const DdlaHandle_t& handle, int m, int n,
            T* A, const int* descA, T* tau, int& info);

/**
 * @brief Distributed addition: C(ic:ic+m-1,jc:jc+n-1) = alpha*op(A) + beta*op(B).
 *
 * Inputs and output may use different block sizes, source coordinates and
 * submatrix offsets. Aligned N regions with matching owners use local copies.
 * Aligned regions use whole-slab owner shifts for N, and whole-slab transpose
 * exchanges on square grids. Other layouts exchange rectangles split at both
 * input and output block boundaries. The whole-slab message path requires
 * m*n <= INT_MAX. All six origins are one-based global indices. ia/ja and
 * ib/jb locate stored inputs before N/T/C is applied, so transposed inputs
 * have n stored rows and m stored columns.
 * A zero coefficient leaves that input unread. Only the requested output
 * region is modified. C may alias an input; input values are consumed before
 * output is written.
 *
 * @tparam T       Scalar type.
 * @param handle   Handle owning the process grid and stream.
 * @param transa   Operation for A ('N','T','C').
 * @param transb   Operation for B ('N','T','C').
 * @param m        Rows of the output region.
 * @param n        Columns of the output region.
 * @param alpha    Multiplier for op(A).
 * @param A        Distributed input A in the selected backend.
 * @param ia       Starting global row in stored A (one-based).
 * @param ja       Starting global column in stored A (one-based).
 * @param descA    ScaLAPACK int[9] descriptor for A.
 * @param beta     Multiplier for op(B).
 * @param B        Distributed input B in the selected backend.
 * @param ib       Starting global row in stored B (one-based).
 * @param jb       Starting global column in stored B (one-based).
 * @param descB    ScaLAPACK int[9] descriptor for B.
 * @param C        Distributed output C in the selected backend.
 * @param ic       Starting global row in C (one-based).
 * @param jc       Starting global column in C (one-based).
 * @param descC    ScaLAPACK int[9] descriptor for C.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pgeadd(
    const DdlaHandle_t& handle, char transa, char transb,
    int m, int n,
    const T& alpha, const T* A, int ia, int ja, const int* descA,
    const T& beta, const T* B, int ib, int jb, const int* descB,
    T* C, int ic, int jc, const int* descC
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
 * @param A    Pointer in the selected backend to distributed matrix A (input/output).
 * @param descA  ScaLAPACK int[9] descriptor for A (may describe a matrix
 *               larger than the leading n x n sub-matrix touched).
 * @param n      Logical order of the leading sub-matrix (<= desc[DDLA_M_]);
 *               n < 0 means the whole matrix.  Default -1.
 */
template <DdlaBackend Backend = default_backend_v, typename T1, typename T2>
void pdam(const DdlaHandle_t& handle, const T1& alpha, T2* A, const int* descA, int n = -1);

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
 * @param A        Pointer in the selected backend to A (input: Hermitian pos-def; output: Cholesky factor).
 * @param ia       Reserved; must be 1 (1-based).  The factor operates on the
 *                 leading n x n sub-matrix anchored at global (0,0).
 * @param ja       Reserved; must be 1 (1-based).
 * @param handle   Handle owning the process grid and stream.
 * @param descA    ScaLAPACK int[9] descriptor for A (mb == nb required).
 *                 The process grid must be square and RSRC == CSRC; other
 *                 layouts throw std::invalid_argument (pposv inherits this).
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
template <DdlaBackend Backend = default_backend_v, typename T>
bool ppotrf(
    const DdlaHandle_t& handle,
    char uplo, int n,
    T* A, int ia, int ja, const int* descA,
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
 * @param A     Pointer in the selected backend to A.
 * @param lda     Leading dimension of A.
 * @param info    0 on success; i > 0 identifies the failed reverse pivot.
 * @param handle  DDLA handle providing the GPU stream and BLAS handle.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void potrf_bottom_right(
    char uplo, int n, T* A, int lda,
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
 * @param A           Pointer in the selected backend to the local block-cyclic storage of A.
 * @param handle        Handle owning the process grid and stream.
 * @param descA         ScaLAPACK int[9] descriptor for the distributed matrix A.
 * @param info          0 on success; i > 0 identifies the failed global pivot.
 *
 * The descriptor may describe a matrix larger than the logical n-by-n
 * sub-matrix (leading block anchored at global (0,0)); only the leading
 * n x n block is factored.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void ppotrf_bottom_right(
    const DdlaHandle_t& handle,
    char uplo, int n, T* A,
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
 * @param A      Pointer in the selected backend to Cholesky factor (from ppotrf).
 * @param handle   Handle owning the process grid and stream.
 * @param descA    ScaLAPACK int[9] descriptor for A.
 * @param B      Pointer in the selected backend to RHS / solution B (input/output).
 * @param descB    ScaLAPACK int[9] descriptor for B: same MB as A, and B's
 *                 rows (side 'L') / columns (side 'R') distributed like A's
 *                 (equal RSRC, resp. equal NB and CSRC).
 * @param is_nega  Diagonal sign-correction flag (from ppotrf return).  The
 *                 factor in A is left unchanged, so it can be reused.
 * @param location Head-correction index forwarded from ppotrf; -1 (or == n)
 *                 means no B permutation.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void ppotrs(
    const DdlaHandle_t& handle,
    char side, char uplo, char trans,
    int n, int nrhs,
    T* A, const int* descA,
    T* B, const int* descB,
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
 * @param A      Pointer in the selected backend to A (input: pos-def; output: Cholesky factor).
 * @param ia       Reserved; must be 1 (1-based).  Solves operate on the
 *                 leading n x n / n x nrhs sub-matrices anchored at (0,0).
 * @param ja       Reserved; must be 1 (1-based).
 * @param handle   Handle owning the process grid and stream.
 * @param descA    ScaLAPACK int[9] descriptor for A.
 * @param B      Pointer in the selected backend to RHS / solution B (input/output).
 * @param ib       Reserved; must be 1 (1-based).
 * @param jb       Reserved; must be 1 (1-based).
 * @param descB    ScaLAPACK int[9] descriptor for B.
 * @param info     Output: 0 on success, >0 if not positive-definite.
 * @param is_head  Forwarded to ppotrf.
 * @param location Forwarded to ppotrf.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pposv(
    const DdlaHandle_t& handle,
    char side, char uplo, char trans,
    int n, int nrhs,
    T* A, int ia, int ja, const int* descA,
    T* B, int ib, int jb, const int* descB,
    int& info, // host pointer
    bool is_head = false, int location = -1
);


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

/**
 * @brief Single-process real-by-complex GEMM:
 * C := alpha * op(A) * op(B) + beta * C.
 *
 * Column-major matrices; R is float or double, B/C and alpha/beta have
 * type std::complex<R>. Flags accept N/T/C (case insensitive); C on real A
 * is T. Leading dimensions are in elements of each matrix's own type.
 * Inputs must not overlap C. CPU pointers are host pointers; GPU pointers
 * are device pointers and work is ordered on the handle's BLAS stream.
 * No communication is performed. The backend must match the handle.
 *
 * Zero m/n returns without accessing matrices. If alpha == 0 or k == 0,
 * A/B are not accessed; beta == 0 never reads C. Invalid flags, dimensions,
 * leading dimensions, required null pointers or handles throw.
 * CPU and large GPU products use bounded column panels of real GEMM;
 * small GPU products use a fused tiled kernel with no temporary matrices.
 */
template <DdlaBackend Backend = default_backend_v, typename R>
void gemm(
    const DdlaHandle_t& handle,
    char transa, char transb,
    int m, int n, int k,
    const std::complex<R>& alpha,
    const R* A, int lda,
    const std::complex<R>* B, int ldb,
    const std::complex<R>& beta,
    std::complex<R>* C, int ldc);

/**
 * @brief Hermitian rank-k update: C := alpha*A*A**H + beta*C  ('N')  or
 *        C := alpha*A**H*A + beta*C.
 *
 * Same unified backend-templated form as @ref gemm: the CPU instantiation
 * calls Fortran xHERK (xSYRK for the real scalar types), the GPU instantiation
 * calls cuBLAS/hipBLAS.  CPU consumes HOST pointers, GPU DEVICE pointers.
 *
 * @p alpha and @p beta are the REAL type of @p T (the xHERK convention).  For
 * the real scalar types this is xSYRK, and a conjugate-transpose request is
 * normalised to a plain transpose.
 *
 * @tparam Backend DdlaBackend::CPU or DdlaBackend::GPU; must match the handle.
 * @tparam T       float, double, complex<float> or complex<double>.
 * @param handle Handle whose backend must equal @p Backend.
 * @param uplo   'U' or 'L' -- which triangle of C is updated.
 * @param trans  'N' or 'C' (also accepts 'T').
 * @param n      Order of C.
 * @param k      Inner dimension (columns of A when @p trans is 'N').
 * @param alpha  Real scaling factor for A*A**H.
 * @param A      Input matrix.
 * @param lda    Leading dimension of A.
 * @param beta   Real scaling factor for C.
 * @param C      In/out matrix; only the @p uplo triangle is referenced.
 * @param ldc    Leading dimension of C.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void herk(
    const DdlaHandle_t& handle,
    char uplo, char trans,
    int n, int k,
    const typename real_type<T>::type& alpha,
    const T* A, int lda,
    const typename real_type<T>::type& beta,
    T* C, int ldc);

/**
 * @brief Hermitian rank-2k update: C := alpha*A*B**H + conj(alpha)*B*A**H +
 *        beta*C  ('N')  or  C := alpha*A**H*B + conj(alpha)*B**H*A + beta*C.
 *
 * Same unified backend-templated form as @ref gemm: the CPU instantiation
 * calls Fortran xHER2K (xSYR2K for the real scalar types), the GPU
 * instantiation calls cuBLAS/hipBLAS.  CPU consumes HOST pointers, GPU DEVICE
 * pointers.
 *
 * Following the xHER2K convention (and cuBLAS v2), @p alpha is COMPLEX while
 * @p beta is REAL.  For the real scalar types this is xSYR2K, where both
 * scalars are real and a conjugate-transpose request is normalised to a plain
 * transpose.
 *
 * @tparam Backend DdlaBackend::CPU or DdlaBackend::GPU; must match the handle.
 * @tparam T       float, double, complex<float> or complex<double>.
 * @param handle Handle whose backend must equal @p Backend.
 * @param uplo   'U' or 'L' -- which triangle of C is updated.
 * @param trans  'N' or 'C' (also accepts 'T').
 * @param n      Order of C.
 * @param k      Inner dimension (columns of A and B when @p trans is 'N').
 * @param alpha  Complex scaling factor for A*B**H.
 * @param A      First input matrix.
 * @param lda    Leading dimension of A.
 * @param B      Second input matrix.
 * @param ldb    Leading dimension of B.
 * @param beta   Real scaling factor for C.
 * @param C      In/out matrix; only the @p uplo triangle is referenced.
 * @param ldc    Leading dimension of C.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void her2k(
    const DdlaHandle_t& handle,
    char uplo, char trans,
    int n, int k,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const typename real_type<T>::type& beta,
    T* C, int ldc);

/**
 * @brief Hermitian matrix-vector product: y := alpha*A*x + beta*y.
 *
 * Same unified backend-templated form as @ref gemm: the CPU instantiation
 * calls Fortran xHEMV (xSYMV for the real scalar types), the GPU instantiation
 * calls cuBLAS/hipBLAS.  CPU consumes HOST pointers, GPU DEVICE pointers.
 *
 * @tparam Backend DdlaBackend::CPU or DdlaBackend::GPU; must match the handle.
 * @tparam T       float, double, complex<float> or complex<double>.
 * @param handle Handle whose backend must equal @p Backend.
 * @param uplo   'U' or 'L' -- which triangle of A is referenced.
 * @param n      Order of A and length of x and y.
 * @param alpha  Scaling factor for A*x.
 * @param A      Hermitian input matrix.
 * @param lda    Leading dimension of A.
 * @param x      Input vector.
 * @param incx   Stride between elements of x.
 * @param beta   Scaling factor for y.
 * @param y      In/out vector.
 * @param incy   Stride between elements of y.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void hemv(
    const DdlaHandle_t& handle,
    char uplo,
    int n,
    const T& alpha,
    const T* A, int lda,
    const T* x, int incx,
    const T& beta,
    T* y, int incy);

/**
 * @brief Hermitian matrix-matrix product: C := alpha*A*B + beta*C  ('L')  or
 *        C := alpha*B*A + beta*C  ('R').
 *
 * Same unified backend-templated form as @ref gemm: the CPU instantiation
 * calls Fortran xHEMM (xSYMM for the real scalar types), the GPU instantiation
 * calls cuBLAS/hipBLAS.  CPU consumes HOST pointers, GPU DEVICE pointers.
 *
 * @tparam Backend DdlaBackend::CPU or DdlaBackend::GPU; must match the handle.
 * @tparam T       float, double, complex<float> or complex<double>.
 * @param handle Handle whose backend must equal @p Backend.
 * @param side   'L' (A multiplies B on the left) or 'R' (on the right).
 * @param uplo   'U' or 'L' -- which triangle of A is referenced.
 * @param m      Number of rows of C.
 * @param n      Number of columns of C.
 * @param alpha  Scaling factor for the product.
 * @param A      Hermitian input matrix.
 * @param lda    Leading dimension of A.
 * @param B      Input matrix.
 * @param ldb    Leading dimension of B.
 * @param beta   Scaling factor for C.
 * @param C      In/out matrix.
 * @param ldc    Leading dimension of C.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void hemm(
    const DdlaHandle_t& handle,
    char side, char uplo,
    int m, int n,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const T& beta,
    T* C, int ldc);
/** @brief Mixed real/complex product with matching float/double precision.
 * Flags, storage and zero coefficients follow the same-type operation.
 * Alpha, beta and C are complex. A real structured input is symmetric;
 * a complex Hermitian input ignores its imaginary diagonal.
 */
template <DdlaBackend Backend = default_backend_v, typename R>
void hemm(const DdlaHandle_t& handle, char side, char uplo, int m, int n,
    const std::complex<R>& alpha, const R* A, int lda,
    const std::complex<R>* B, int ldb, const std::complex<R>& beta,
    std::complex<R>* C, int ldc);
template <DdlaBackend Backend = default_backend_v, typename R>
void hemm(const DdlaHandle_t& handle, char side, char uplo, int m, int n,
    const std::complex<R>& alpha, const std::complex<R>* A, int lda,
    const R* B, int ldb, const std::complex<R>& beta,
    std::complex<R>* C, int ldc);


/**
 * @brief Local HEMM with only uploC of a general n-by-n C updated.
 * Same input/output triangle and coefficient semantics as phemmt. A/B/C
 * use column-major backend memory with leading dimensions at least max(1,n).
 * Real scalar types use symmetric A. C must be disjoint from A/B. The call
 * performs no rank communication. Temporary storage holds one bounded input
 * panel and diagonal GEMMT scratch. Private check/query/run components allow reuse.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void hemmt(const DdlaHandle_t& handle,char side,char uploA,char uploC,int n,
    const T& alpha,const T* A,int lda,const T* B,int ldb,const T& beta,T* C,int ldc);
/** @brief Mixed real/complex product with matching float/double precision.
 * Flags, storage and zero coefficients follow the same-type operation.
 * Alpha, beta and C are complex. A real structured input is symmetric;
 * a complex Hermitian input ignores its imaginary diagonal.
 */
template <DdlaBackend Backend = default_backend_v, typename R>
void hemmt(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n,
    const std::complex<R>& alpha, const R* A, int lda,
    const std::complex<R>* B, int ldb, const std::complex<R>& beta,
    std::complex<R>* C, int ldc);
template <DdlaBackend Backend = default_backend_v, typename R>
void hemmt(const DdlaHandle_t& handle, char side, char uploA, char uploC, int n,
    const std::complex<R>& alpha, const std::complex<R>* A, int lda,
    const R* B, int ldb, const std::complex<R>& beta,
    std::complex<R>* C, int ldc);


/**
 * @brief Local SYMM updating only uploC of a general n-by-n C.
 * Same symmetric input, selected output and coefficient semantics as psymmt.
 * A/B/C use column-major backend memory, with lda/ldb/ldc >= max(1,n).
 * Complex A keeps its diagonal and reflects uploA without conjugation.
 * alpha=0 reads no A/B, beta=0 reads no selected C, and n=0 or
 * alpha=0,beta=1 permits null data pointers. Inputs and opposite C are preserved.
 * The call uses the shared bounded HEMMT panel workspace and performs no MPI
 * communication. Backend and four scalar types follow symm.
 */
template <DdlaBackend Backend = default_backend_v,typename T>
void symmt(const DdlaHandle_t& handle,char side,char uploA,char uploC,int n,
    const T& alpha,const T* A,int lda,const T* B,int ldb,const T& beta,T* C,int ldc);

/**
 * @brief Hermitian rank-2 update: A := alpha*x*y**H + conj(alpha)*y*x**H + A.
 *
 * Same unified backend-templated form as @ref gemm: the CPU instantiation
 * calls Fortran xHER2 (xSYR2 for the real scalar types), the GPU instantiation
 * calls cuBLAS/hipBLAS.  CPU consumes HOST pointers, GPU DEVICE pointers.
 *
 * There is no @p beta and no transpose flag: the update is applied in place to
 * @p A, which is therefore the LAST argument (matching xHER2 / cuBLAS v2).
 *
 * @tparam Backend DdlaBackend::CPU or DdlaBackend::GPU; must match the handle.
 * @tparam T       float, double, complex<float> or complex<double>.
 * @param handle Handle whose backend must equal @p Backend.
 * @param uplo   'U' or 'L' -- which triangle of A is updated.
 * @param n      Order of A and length of x and y.
 * @param alpha  Scaling factor for x*y**H.
 * @param x      First input vector.
 * @param incx   Stride between elements of x.
 * @param y      Second input vector.
 * @param incy   Stride between elements of y.
 * @param A      In/out Hermitian matrix; only the @p uplo triangle is updated.
 * @param lda    Leading dimension of A.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void her2(
    const DdlaHandle_t& handle,
    char uplo,
    int n,
    const T& alpha,
    const T* x, int incx,
    const T* y, int incy,
    T* A, int lda);

/**
 * @brief Matrix-vector product: y := alpha*op(A)*x + beta*y.
 *
 * Same unified backend-templated form as @ref gemm: the CPU instantiation
 * calls Fortran xGEMV, the GPU instantiation calls cuBLAS/hipBLAS.  CPU
 * consumes HOST pointers, GPU DEVICE pointers.
 *
 * @tparam Backend DdlaBackend::CPU or DdlaBackend::GPU; must match the handle.
 * @tparam T       float, double, complex<float> or complex<double>.
 * @param handle Handle whose backend must equal @p Backend.
 * @param trans  'N' (y := alpha*A*x + beta*y), 'T' or 'C' (y :=
 *               alpha*A**T*x / alpha*A**H*x + beta*y).
 * @param m      Number of rows of A (and length of y under 'N').
 * @param n      Number of columns of A (and length of x).
 * @param alpha  Scaling factor for op(A)*x.
 * @param A      Input matrix.
 * @param lda    Leading dimension of A.
 * @param x      Input vector of length n ('N') or m (otherwise).
 * @param incx   Stride between elements of x.
 * @param beta   Scaling factor for y.
 * @param y      In/out vector of length m ('N') or n (otherwise).
 * @param incy   Stride between elements of y.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void gemv(
    const DdlaHandle_t& handle,
    char trans, int m, int n,
    const T& alpha,
    const T* A, int lda,
    const T* x, int incx,
    const T& beta,
    T* y, int incy);

/**
 * @brief Triangular matrix-vector product: x := op(A)*x, IN PLACE.
 *
 * Same unified backend-templated form as @ref gemm: the CPU instantiation
 * calls Fortran xTRMV, the GPU instantiation calls cuBLAS/hipBLAS.  CPU
 * consumes HOST pointers, GPU DEVICE pointers.
 *
 * @p x is modified in place, so a caller that needs the original vector must
 * keep its own copy (or re-upload it) before comparing results.
 *
 * @tparam Backend DdlaBackend::CPU or DdlaBackend::GPU; must match the handle.
 * @tparam T       float, double, complex<float> or complex<double>.
 * @param handle Handle whose backend must equal @p Backend.
 * @param uplo   'U' or 'L' -- which triangle of A is referenced.
 * @param trans  'N' (x := A*x), 'T' or 'C' (x := A**T*x / A**H*x).
 * @param diag   'U' (unit diagonal, A's diagonal not referenced) or 'N'.
 * @param n      Order of A and length of x.
 * @param A      Input triangular matrix.
 * @param lda    Leading dimension of A.
 * @param x      In/out vector, overwritten with op(A)*x.
 * @param incx   Stride between elements of x.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void trmv(
    const DdlaHandle_t& handle,
    char uplo, char trans, char diag,
    int n,
    const T* A, int lda,
    T* x, int incx);

/**
 * @brief Euclidean norm: result := ||x||_2.
 *
 * Same unified backend-templated form as @ref gemm: the CPU instantiation
 * calls Fortran xNRM2/xCNRM2/xZNRM2, the GPU instantiation calls
 * cuBLAS/hipBLAS.  The result is written THROUGH THE POINTER @p result, not
 * returned by value, because the two backends disagree on the pointer's
 * address space:
 *   - GPU handles: @p result is DEVICE memory.  The GPU instantiation forces
 *     the vendor handle into DEVICE pointer mode for the call (and restores
 *     the previous mode afterwards); passing a host pointer there is undefined.
 *   - CPU handles: @p result is HOST memory.
 *
 * @tparam Backend DdlaBackend::CPU or DdlaBackend::GPU; must match the handle.
 * @tparam T       float, double, complex<float> or complex<double>.
 * @param handle Handle whose backend must equal @p Backend.
 * @param n      Length of x.
 * @param x      Input vector.
 * @param incx   Stride between elements of x.
 * @param result Scalar result; DEVICE memory for GPU handles, HOST memory for
 *               CPU handles.  Always the REAL type of @p T.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void nrm2(
    const DdlaHandle_t& handle,
    int n,
    const T* x, int incx,
    typename real_type<T>::type* result);

/**
 * @brief Conjugate dot product: result := x**H * y (plain dot for real types).
 *
 * Same unified backend-templated form as @ref gemm: the CPU instantiation
 * calls Fortran xDOT for the real scalar types (the complex ones accumulate in
 * the host arm, since the Fortran ABI for complex-valued function returns is
 * compiler-dependent), the GPU instantiation calls cuBLAS/hipBLAS.  The result
 * is written THROUGH THE POINTER @p result, not returned by value, because the
 * two backends disagree on the pointer's address space:
 *   - GPU handles: @p result is DEVICE memory.  The GPU instantiation forces
 *     the vendor handle into DEVICE pointer mode for the call (and restores
 *     the previous mode afterwards); passing a host pointer there is undefined.
 *   - CPU handles: @p result is HOST memory.
 *
 * @tparam Backend DdlaBackend::CPU or DdlaBackend::GPU; must match the handle.
 * @tparam T       float, double, complex<float> or complex<double>.
 * @param handle Handle whose backend must equal @p Backend.
 * @param n      Length of x and y.
 * @param x      First input vector, conjugated in the product.
 * @param incx   Stride between elements of x.
 * @param y      Second input vector.
 * @param incy   Stride between elements of y.
 * @param result Scalar result; DEVICE memory for GPU handles, HOST memory for
 *               CPU handles.  Same scalar type as @p x and @p y.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void dotc(
    const DdlaHandle_t& handle,
    int n,
    const T* x, int incx,
    const T* y, int incy,
    T* result);

/**
 * @brief Triangular solve with multiple right-hand sides:
 *        side='L' -> B := alpha*op(A)**-1*B  (A is m-by-m),
 *        side='R' -> B := alpha*B*op(A)**-1  (A is n-by-n).
 *
 * Same unified backend-templated form as @ref gemm: the CPU instantiation
 * calls Fortran xTRSM, the GPU instantiation calls cuBLAS/hipBLAS.  CPU
 * consumes HOST pointers, GPU DEVICE pointers.
 *
 * @p A and @p B are non-const: the vendor prototypes require non-const
 * pointers even though A is only read.  @p B is read and written in place.
 *
 * @tparam Backend DdlaBackend::CPU or DdlaBackend::GPU; must match the handle.
 * @tparam T       float, double, complex<float> or complex<double>.
 * @param handle Handle whose backend must equal @p Backend.
 * @param side   'L' or 'R' -- whether op(A) multiplies B on the left or right.
 * @param uplo   'U' or 'L' -- which triangle of A holds the factor.
 * @param trans  'N', 'T' or 'C' -- op(A) = A, A**T or A**H.
 * @param diag   'U' (unit diagonal) or 'N' (non-unit).
 * @param m      Rows of B (and order of A when @p side is 'L').
 * @param n      Columns of B (and order of A when @p side is 'R').
 * @param alpha  Scaling factor applied to the solution.
 * @param A      Triangular factor; only the @p uplo triangle is referenced.
 * @param lda    Leading dimension of A.
 * @param B      Right-hand sides (in) / solution (out).
 * @param ldb    Leading dimension of B.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void trsm(
    const DdlaHandle_t& handle,
    char side, char uplo, char trans, char diag,
    int m, int n,
    const T& alpha,
    T* A, int lda,
    T* B, int ldb);

/**
 * @brief Local symmetric rank-k update: C := alpha*A*A^T + beta*C (N),
 *        or alpha*A^T*A + beta*C (T). Only uplo of C is updated.
 * @tparam T float, double, complex<float> or complex<double>.
 * @param trans N/T for complex inputs; real inputs also accept C as T.
 * Complex symmetry uses no conjugation. Use herk for a Hermitian update.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void syrk(
    const DdlaHandle_t& handle,
    char uplo, char trans,
    int n, int k,
    const T& alpha,
    const T* A, int lda,
    const T& beta,
    T* C, int ldc);

/**
 * @brief In-place triangular matrix multiply:
 *        side='L' -> B := alpha*op(A)*B  (A m-by-m),
 *        side='R' -> B := alpha*B*op(A)  (A n-by-n).
 *
 * The CPU arm calls Fortran xTRMM. The GPU arm calls cuBLAS/hipBLAS with
 * B as both input and output. CPU pointers are host memory; GPU pointers
 * are device memory. A must not overlap B. Leading-dimension padding is
 * untouched. Empty matrices leave B untouched; alpha=0 zeros the logical
 * matrix without reading A or the old B, and A may then be null.
 * Flags are case-insensitive; invalid flags or dimensions throw.
 *
 * @tparam Backend DdlaBackend::CPU or DdlaBackend::GPU; must match the handle.
 * @tparam T float, double, complex<float> or complex<double>.
 * @param handle Handle whose backend must equal Backend.
 * @param side 'L' or 'R': side on which op(A) multiplies B.
 * @param uplo 'U' or 'L': referenced triangle of A.
 * @param trans 'N', 'T' or 'C': op(A) = A, A**T or A**H.
 * @param diag 'U' (implicit unit diagonal) or 'N' (stored diagonal).
 * @param m Rows of B.
 * @param n Columns of B.
 * @param alpha Scaling factor.
 * @param A Read-only triangular factor; unused triangle/implicit diagonal not read.
 * @param lda Leading dimension of A, at least max(1,side=='L'?m:n).
 * @param B Input matrix overwritten by the product.
 * @param ldb Leading dimension of B, at least max(1,m).
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void trmm(
    const DdlaHandle_t& handle,
    char side, char uplo, char trans, char diag,
    int m, int n,
    const T& alpha,
    const T* A, int lda,
    T* B, int ldb);

/**
 * @brief Single-process triangular-by-triangular multiply with per-operand
 *        triangle and transpose flags:
 *        C := alpha*op(A)*op(B) + beta*C, op(X) in {X, X**T, X**H}.
 *
 * Each operand reads only the triangle of its stored matrix named by uplo_a
 * / uplo_b; a transpose flag flips that operand's EFFECTIVE triangle.
 * Matching effective triangles give a triangular result: upper x upper
 * writes the upper triangle of C, lower x lower the lower triangle, and in
 * both cases the opposite triangle of C is preserved.  An upper x lower or
 * lower x upper product fills a FULL C, and beta then applies to all of it.
 * transa/transb accept N/T/C (case insensitive); C conjugates complex
 * operands and equals T for real ones.
 *
 * Matrices are column-major n-by-n with explicit (non-unit) diagonals.
 * Unused input triangles and leading-dimension padding are never accessed.
 * A and B may coincide; C must not overlap either input, including padding.
 * CPU handles take host pointers; GPU handles take device pointers and enqueue
 * work on the handle stream. No process-grid collectives are performed.
 *
 * With alpha=0, A and B are not referenced and may be null. With beta=0,
 * the written region of C need not be initialized. n=0, or alpha=0 and beta=1,
 * is a no-op and all matrix pointers may be null. Leading dimensions must
 * still be >= max(1,n). Invalid flags, dimensions, pointers or overlap throw
 * std::invalid_argument; a null/mismatched handle throws std::runtime_error.
 *
 * @tparam Backend CPU or GPU; must match the handle.
 * @tparam T float, double, std::complex<float> or std::complex<double>.
 *
 * @param uplo_a 'U' or 'L' (case insensitive): triangle of stored A read.
 * @param transa 'N', 'T' or 'C' (case insensitive): op applied to A.
 * @param uplo_b 'U' or 'L' (case insensitive): triangle of stored B read.
 * @param transb 'N', 'T' or 'C' (case insensitive): op applied to B.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void trtrm(const DdlaHandle_t& handle, char uplo_a, char transa, char uplo_b, char transb,
           int n, const T& alpha, const T* A, int lda, const T* B, int ldb,
           const T& beta, T* C, int ldc);

/**
 * @brief Batched matrix multiply over arrays of pointers: for each
 *        i in [0, batchCount),
 *        C[i] := alpha*op(A[i])*op(B[i]) + beta*C[i].
 *
 * Same unified backend-templated form as @ref gemm: the CPU instantiation
 * loops over Fortran xGEMM, the GPU instantiation calls cuBLAS/hipBLAS
 * xgemmBatched.  Everything lives where the handle's memory does: on a CPU
 * handle the matrices AND the arrays of pointers are host memory, on a GPU
 * handle both are device memory (the vendor routine reads the pointer arrays
 * on the device).
 *
 * CPU threading: a CPU handle runs the loop on its host-thread budget (the
 * same one geam uses, fixed by ddlaSet), every OpenMP thread computing whole
 * products, so each core handles one matrix multiplication at a time.  The
 * BLAS library's own thread count (OpenBLAS / MKL) is held at one for the
 * duration of the loop and restored after it.  With one thread or a single
 * product the loop is serial and the BLAS keeps its threads.  The C[i] must
 * not overlap (the cuBLAS contract as well); the result does not depend on
 * the thread count.
 *
 * @tparam Backend DdlaBackend::CPU or DdlaBackend::GPU; must match the handle.
 * @tparam T       float, double, complex<float> or complex<double>.
 * @param handle     Handle whose backend must equal @p Backend.
 * @param transa     'N', 'T' or 'C' -- op(A[i]).
 * @param transb     'N', 'T' or 'C' -- op(B[i]).
 * @param m          Rows of op(A[i]) and C[i].
 * @param n          Columns of op(B[i]) and C[i].
 * @param k          Shared inner dimension.
 * @param alpha      Scaling factor for the product.
 * @param Aarray     Array of @p batchCount pointers to the left operands.
 * @param lda        Leading dimension of every A[i].
 * @param Barray     Array of @p batchCount pointers to the right operands.
 * @param ldb        Leading dimension of every B[i].
 * @param beta       Scaling factor for C.
 * @param Carray     Array of @p batchCount pointers to the in/out results.
 * @param ldc        Leading dimension of every C[i].
 * @param batchCount Number of matrix triples.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void gemmBatched(
    const DdlaHandle_t& handle,
    char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const T* const Aarray[], int lda,
    const T* const Barray[], int ldb,
    const T& beta,
    T* const Carray[], int ldc,
    int batchCount);
/** @brief Mixed real/complex GEMM over backend-resident pointer arrays.
 * One input is real and the other complex, with matching float/double
 * precision and complex alpha, beta and C. Flags and zero coefficients
 * follow GEMM. Output matrices must not overlap inputs or each other.
 */
template <DdlaBackend Backend = default_backend_v, typename R>
void gemmBatched(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k,
    const std::complex<R>& alpha, const R* const Aarray[], int lda,
    const std::complex<R>* const Barray[], int ldb, const std::complex<R>& beta,
    std::complex<R>* const Carray[], int ldc, int batchCount);
template <DdlaBackend Backend = default_backend_v, typename R>
void gemmBatched(const DdlaHandle_t& handle, char transa, char transb, int m, int n, int k,
    const std::complex<R>& alpha, const std::complex<R>* const Aarray[], int lda,
    const R* const Barray[], int ldb, const std::complex<R>& beta,
    std::complex<R>* const Carray[], int ldc, int batchCount);


/**
 * @brief Backend-neutral matrix add / transpose-copy: C := alpha * op(A) + beta * op(B).
 *
 * op(X) is X, X^T or X^H for transX = 'N', 'T', 'C' (either case).  C is m x n (column-major, leading
 * dimension ldc); op(A) and op(B) are m x n, so A is m x n for 'N' and n x m otherwise, and likewise B.
 * Same call shape as ddla::gemm: the handle, then the LAPACK character flags.  The template backend must
 * equal the handle's.
 *
 * - Scaled transpose-copy (what omatcopy was): beta = 0, B = nullptr,
 *   `geam<Backend, T>(h, 'T', 'N', m, n, alpha, A, lda, T(0), nullptr, 1, C, ldc)`.
 * - beta == 0 means B is not read (and may be nullptr).  On the CPU backend alpha == 0 likewise leaves A
 *   unread.  The GPU backend is cuBLAS / hipBLAS geam, which reads the operands it is given (0 * NaN is NaN
 *   there) and, for beta == 0, aliases B to A.
 * - alpha == 1 with beta == 0 on a CPU handle moves the bits of A unchanged (-0.0, Inf and NaN survive).
 * - C may be A (or B) when that operand is 'N' and the leading dimensions agree.
 * - A CPU handle runs the loop on OpenMP threads: OMP_NUM_THREADS, else the CPUs this process may use divided
 *   by the ranks of its node; DDLA_EIG_THREADS overrides.  The result does not depend on the thread count.
 * - Throws std::runtime_error for a null handle, a template backend that differs from the handle's, an
 *   invalid flag, a null A or C, a null B with beta != 0, or a leading dimension smaller than the row count of
 *   its matrix.  m <= 0 or n <= 0 returns without touching memory.
 *
 * @param handle  Handle whose backend must equal @p Backend.
 * @param transa  'N', 'T' or 'C' -- op(A).
 * @param transb  'N', 'T' or 'C' -- op(B); ignored when beta == 0.
 * @param m       Rows of op(A), op(B) and C.
 * @param n       Columns of op(A), op(B) and C.
 * @param alpha   Scaling factor for op(A).
 * @param A       First operand (never null).
 * @param lda     Leading dimension of A.
 * @param beta    Scaling factor for op(B).
 * @param B       Second operand; may be nullptr when beta == 0.
 * @param ldb     Leading dimension of B (not looked at when beta == 0).
 * @param C       Result.
 * @param ldc     Leading dimension of C.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void geam(const DdlaHandle_t& handle, char transa, char transb, int m, int n,
          const T& alpha, const T* A, int lda,
          const T& beta, const T* B, int ldb,
          T* C, int ldc);

/**
 * @brief Execute a batch of GEMMs whose dimensions and leading dimensions may differ.
 *
 * Handle-first like every other backend-templated routine.  The kernels are device code, so only the GPU backend does
 * the work: `gemmVbatched<DdlaBackend::CPU, T>` throws std::invalid_argument, as does a null handle or a handle of the
 * other backend.  Dimension, leading-dimension and matrix-pointer arrays are device resident.  The operation is
 * enqueued on the handle's stream.  A zero batch count is a no-op.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void gemmVbatched(
    const DdlaHandle_t& handle,
    char transA, char transB,
    int* d_m, int* d_n, int* d_k,
    T alpha,
    const T* const* d_A_array, int* d_lda,
    const T* const* d_B_array, int* d_ldb,
    T beta,
    T** d_C_array, int* d_ldc,
    int batch_count);

/**
 * @brief Two dependent variable-size GEMM stages that share a reusable temporary.
 *
 * Stage zero writes through the full batch-count temporary pointer array d_C_array_0.
 * Both stages use the corresponding pointer entries for each host-resident segment in segment_sizes.
 * Entries in different segments may alias to reuse temporary device storage.
 * Stage-zero beta must be zero, every segment must be positive, and the segment total must
 * equal batch_count. Backend rules as for gemmVbatched.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void gemmVbatched2s(
    const DdlaHandle_t& handle,
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
    int segment_count);


/**
 * @brief Fill a buffer with independent uniform random values in [0, 1) on a CPU handle and
 * (0, 1] on a GPU handle (the vendor generators exclude 0 and include 1).
 *
 * `data` lives where the handle's backend does: host memory for a CPU handle, device memory for a GPU handle.  The CPU
 * arm uses std::mt19937_64 seeded from the system clock; the GPU arm uses the vendor RNG (curand on CUDA, hiprand on
 * HIP) on the handle's stream and returns once the values are written.  The two are not bit-for-bit reproducible
 * against each other.  Complex types are filled with 2N independent real draws (real part first, then imaginary part).
 *
 * A non-positive `lengthOfData` is a safe no-op; `data` may be nullptr then.
 *
 * Supported T: float, double, std::complex<float>, std::complex<double>.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void random_generate(const DdlaHandle_t& handle, T* data, const int64_t& lengthOfData);

template <DdlaBackend Backend = default_backend_v, typename T>
void write_matrix(const T* A, int m, int n, const char* filename);

template <DdlaBackend Backend = default_backend_v, typename T>
void scal(const DdlaHandle_t& handle, int n, const T& alpha, T* x, int incx);
/** @brief Scale an m-by-n matrix region in place, preserving its padding.
 * Alpha zero overwrites active entries without reading their old values.
 * Unlike the vector overload, lda is a column-major leading dimension.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void scal(const DdlaHandle_t& handle, int m, int n, const T& alpha, T* A, int lda);

/** @brief Collective matrix-region scaling in a block-cyclic descriptor.
 * Scales the leading m-by-n region, with no redistribution or workspace.
 * Alpha zero overwrites active entries without reading them.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pscal(const DdlaHandle_t& handle, int m, int n, const T& alpha,
           T* A, const int* descA);

/** @brief Scale a batch of nonoverlapping m-by-n matrix regions in place.
 * Pointer arrays reside in the handle backend. Padding is preserved, alpha
 * zero never reads old entries, and the operation requires no workspace.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void scalBatched(const DdlaHandle_t& handle, int m, int n, const T& alpha,
                 T* const Aarray[], int lda, int batchCount);
/** @brief Real-coefficient scaling of complex matrix regions, componentwise.
 * Precision must match. Zero alpha never reads old entries; padding is kept.
 * T is the matrix type; Alpha is inferred, so scal<Backend,T> needs no cast.
 */
template <DdlaBackend Backend = default_backend_v,typename T,typename Alpha>
std::enable_if_t<std::is_same_v<T,std::complex<Alpha>>,void>
scal(const DdlaHandle_t& h,int m,int n,const Alpha& alpha,T* A,int lda);
template <DdlaBackend Backend = default_backend_v,typename T,typename Alpha>
std::enable_if_t<std::is_same_v<T,std::complex<Alpha>>,void>
pscal(const DdlaHandle_t& h,int m,int n,const Alpha& alpha,T* A,const int* descA);
template <DdlaBackend Backend = default_backend_v,typename T,typename Alpha>
std::enable_if_t<std::is_same_v<T,std::complex<Alpha>>,void>
scalBatched(const DdlaHandle_t& h,int m,int n,const Alpha& alpha,T* const Aarray[],int lda,int count);




template <DdlaBackend Backend = default_backend_v, typename T>
void axpy(const DdlaHandle_t& handle, int n, const T& alpha,
          const T* x, int incx, T* y, int incy);

template <DdlaBackend Backend = default_backend_v, typename T>
void iamax(const DdlaHandle_t& handle, int n, const T* x, int incx, int& result);

/**
 * @brief Backend-templated BLAS Level-1 swap: x <-> y.
 *
 * `n` elements of x and y with strides incx / incy (a negative stride walks the vector backwards, the BLAS
 * convention); both strides must be nonzero and x, y must not overlap.  CPU: ?swap on host pointers.  GPU: cuBLAS /
 * hipBLAS swap on device pointers, enqueued on the handle's stream.  n <= 0 returns.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void swap(const DdlaHandle_t& handle, int n, T* x, int incx, T* y, int incy);

template <DdlaBackend Backend = default_backend_v, typename T>
void geru(const DdlaHandle_t& handle, int m, int n, const T& alpha,
          const T* x, int incx, const T* y, int incy, T* A, int lda);

/**
 * @brief Backend-templated conjugate rank-1 update: A := alpha * x * y^H + A.
 *
 * This is the routine LAPACK's xLARF uses to apply a Householder reflector
 * from the left:  A := A - tau * v * w^H.  The update touches the leading
 * m x n submatrix of A only.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void gerc(const DdlaHandle_t& handle, int m, int n, const T& alpha,
          const T* x, int incx, const T* y, int incy, T* A, int lda);

/**
 * @brief Backend-templated LU factorization with partial pivoting, A = P * L * U (LAPACK ?getrf).
 *
 * A is m x n column-major with lda >= max(1, m); on exit it holds L (unit lower triangular, diagonal not stored) and
 * U.  `ipiv` has min(m, n) entries: row i was interchanged with row ipiv[i] (1-based, LAPACK convention).  `info` is
 * 0 on success and k > 0 when U(k, k) is exactly zero (the factorization is complete, but U is singular); invalid
 * arguments throw instead of reporting a negative value.
 *
 * Every pointer lives where the backend does: host memory for a CPU handle, device memory for a GPU handle -- `ipiv`
 * and `info` included.  A GPU call is asynchronous on the handle's stream, so `info` is valid once the stream has been
 * synchronized.  m <= 0 or n <= 0 sets *info = 0 and returns.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void getrf(const DdlaHandle_t& handle, int m, int n, T* A, int lda, int* ipiv, int* info);

/**
 * @brief Backend-templated Cholesky factorization of a Hermitian (symmetric) positive-definite matrix (LAPACK ?potrf):
 * A = U^H * U for uplo = 'U', A = L * L^H for uplo = 'L'.
 *
 * A is n x n column-major with lda >= max(1, n).  Only the `uplo` triangle is read and overwritten; the other triangle
 * is left untouched.  `info` is 0 on success and k > 0 when the leading minor of order k is not positive definite.
 * Pointer placement, asynchrony and the n <= 0 case are as for getrf.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void potrf(const DdlaHandle_t& handle, char uplo, int n, T* A, int lda, int* info);

/**
 * @brief Backend-templated row interchanges (LAPACK ?laswp): for i = k1..k2 (1-based), swap row i of the n-column
 * matrix A with row ipiv[k1 - 1 + (i - k1) * |incx|].
 *
 * Rows run k1..k2 for incx > 0 and k2..k1 for incx < 0, so incx = -1 undoes incx = 1 for the same pivots.  `ipiv`
 * (1-based row numbers, as ?getrf produces them) must hold at least k2 * |incx| entries; it lives with A (device memory
 * on a GPU handle).  lda >= max(1, k2).  n <= 0 or k1 > k2 returns.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void laswp(const DdlaHandle_t& handle, int n, T* A, int lda, int k1, int k2, const int* ipiv, int incx);

/**
 * @brief Backend-templated QR or QL factorization of one matrix (LAPACK ?geqrf / ?geqlf): A = Q * R for uplo = 'U',
 * A = Q * L for uplo = 'L'.
 *
 * A is m x n column-major with lda >= max(1, m); k = min(m, n) Householder reflectors H(i) = I - tau(i) v(i) v(i)^H are
 * produced and Q = H(1) ... H(k) (QR) or H(k) ... H(1) (QL).  The factors are stored exactly as LAPACK stores them.
 * 'U' (?geqrf): R is on and above the diagonal, v(i) has its unit entry at row i and lies below it.  'L' (?geqlf): L is
 * on and below the diagonal of the last k columns (the bottom n x n block for m >= n), v(i) has its unit entry at row
 * m - k + i of column n - k + i and lies above it; tau(i) belongs to that column.  `tau` has k entries.  For complex T
 * the diagonal of R / L is real.  The reflector scalars follow LAPACK ?larfg (beta = -sign(Re alpha) ||(alpha, x)||),
 * so tau = 0 and an untouched column when x = 0 and alpha is real.  One difference on the GPU: the vendor routine takes a
 * pivot alpha of exactly -0.0 as +0 (LAPACK takes it as negative), so for such a pivot beta has the opposite sign and the
 * reflectors after it differ from LAPACK's; the factorization is valid either way.
 *
 * Every pointer lives where the backend does: host memory for a CPU handle, device memory for a GPU handle.  A GPU call
 * is asynchronous on the handle's stream (the vendor scratch is allocated and freed stream-ordered).  The GPU
 * implementation has no ?geqlf to call: 'L' factors the row- and column-reversed copy J A J with the vendor geqrf.
 * m <= 0 or n <= 0 returns; invalid arguments (null handle, backend mismatch, uplo other than 'U'/'u'/'L'/'l',
 * negative sizes, lda < max(1, m), null A or tau) throw.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void geqrf(const DdlaHandle_t& handle, char uplo, int m, int n, T* A, int lda, T* tau);

/**
 * @brief Backend-templated triangular factor of a block reflector (LAPACK ?larft, STOREV = 'C'): the T of the compact-WY
 * form H = I - V T V^H of k Householder reflectors H(i) = I - tau(i) v(i) v(i)^H.
 *
 * V is n x k column-major with ldv >= max(1, n), tau holds k scalars, T is k x k with ldt >= k, and k <= n.
 * direct = 'F': H = H(1) H(2) ... H(k), T is upper triangular; v(i) has its unit entry at row i and lies below it, as
 * ?geqrf stores it.  direct = 'B': H = H(k) ... H(2) H(1), T is lower triangular; v(i) has its unit entry at row n - k + i
 * and lies above it, as ?geqlf stores it.  Only the strictly lower (F) / upper (B) trapezoid of V is read: the unit
 * entries and the zeros are implied, so the output of geqrf can be passed as it is (R / L stays in V's other triangle).
 * T(i,i) = tau(i).  T is written completely: LAPACK leaves the triangle T does not own alone, here it is zeroed, so T
 * can be shipped or transposed as a block; the rows of a padded T (ldt > k) are not touched.  Only storev = 'C'
 * (columnwise, the geqrf layout) is supported.  V is not modified (the CPU LAPACK may touch its unit entries while it
 * runs and restores them).
 *
 * Every pointer lives where the backend does: host memory for a CPU handle, device memory for a GPU handle, tau
 * included.  The CPU instantiation calls LAPACK ?larft.  The GPU one needs no vendor routine (cuSOLVER has
 * cusolverDnXlarft only from CUDA 12.4, hipSOLVER none): it forms V^H V with herk on an explicit unit-triangular head /
 * tail of V and runs the column recurrence of T in one small kernel, asynchronously on the handle's stream (a k x k
 * scratch pair is allocated and freed stream-ordered).  k <= 0 returns; invalid arguments (null handle, backend
 * mismatch, direct other than 'F'/'B', storev other than 'C', negative sizes, k > n, ldv < max(1, n), ldt < k, null
 * V / tau / T) throw.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void larft(const DdlaHandle_t& handle, char direct, char storev, int n, int k, T* V, int ldv, const T* tau, T* Tmat,
           int ldt);

// Interfaces for the registered reflector and mixed-domain operations.
template <DdlaBackend Backend = default_backend_v, typename R>
void pgemm(const DdlaHandle_t& handle, char transa, char transb,
           int m, int n, int k, const std::complex<R>& alpha,
           const R* A, const int* descA,
           const std::complex<R>* B, const int* descB,
           const std::complex<R>& beta, std::complex<R>* C, const int* descC);

template <DdlaBackend Backend = default_backend_v, typename R>
void pgemm(const DdlaHandle_t& handle, char transa, char transb,
           int m, int n, int k, const std::complex<R>& alpha,
           const std::complex<R>* A, const int* descA,
           const R* B, const int* descB,
           const std::complex<R>& beta, std::complex<R>* C, const int* descC);

/**
 * @brief Reduce a full distributed Hermitian matrix to band form.
 * A has square source-zero blocks. nbw divides the block size. Both triangles
 * of A are workspace; upper band entries and implicit-unit reflector tails
 * remain in A. tmat is backend storage replicated across ranks, with
 * ((n-1)/nbw) slots of nbw-by-nbw compact reflector factors (zero for n=0).
 * Supports all four scalar types and owns its internal workspace.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void phe2hb(const DdlaHandle_t& handle, int n, T* A, const int* descA,
            int nbw, T* tmat, int& info);

/**
 * @brief Apply the full-to-band transform produced by phe2hb to distributed Z.
 * trans='N' applies Q; 'C' applies Q^H and maps band eigenvectors back to the
 * original matrix. 'T' is accepted as an alias of 'C', including complex types.
 * A and tmat are read-only backend storage from phe2hb. Z's leading n-by-n
 * region is overwritten and must be disjoint from the factors. Its row block
 * size and source match A; its column distribution may differ. A has square
 * source-zero blocks and nbw divides its block size. Parent data and padding
 * outside the active region are preserved. No panels means identity.
 * GPU execution combines adjacent panel pairs in bounded device workspace.
 * The operation owns internal workspace; info is zero on success.
 * Argument checks are local; callers provide consistent arguments on all ranks.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void punmtr_he2hb(
    const DdlaHandle_t& handle,
    char trans,
    int n, int nbw,
    const T* A, const int* descA,
    const T* tmat,
    T* Z, const int* descZ,
    int& info
);

template <DdlaBackend Backend = default_backend_v, typename T>
void pherm_complete(
    const DdlaHandle_t& handle,
    char uplo, int n,
    T* A, const int* descA
);

template <DdlaBackend Backend = default_backend_v, typename T>
void phetrd(const DdlaHandle_t& handle, char uplo, int n, T* A, const int* descA,
    typename real_type<T>::type* d, typename real_type<T>::type* e, T* tau, int& info);

template <DdlaBackend Backend = default_backend_v, typename T>
void punmtr(const DdlaHandle_t& handle, char side, char uplo, char trans, int m, int n,
    const T* A, const int* descA, const T* tau, T* C, const int* descC);

template <DdlaBackend Backend = default_backend_v, typename T>
void punmtr(const DdlaHandle_t& handle, char side, char uplo, char trans,
    int m, int n, int k, const T* V, int ldv, const T* Tmat, int ldt,
    T* C, const int* descC);

template <DdlaBackend Backend = default_backend_v, typename R>
void gemm(
    const DdlaHandle_t& handle,
    char transa, char transb,
    int m, int n, int k,
    const std::complex<R>& alpha,
    const std::complex<R>* A, int lda,
    const R* B, int ldb,
    const std::complex<R>& beta,
    std::complex<R>* C, int ldc);

/**
 * @brief Diagonalize a replicated real symmetric tridiagonal matrix collectively.
 * @tparam T float or double. d, e and w are host vectors on every rank.
 * @param jobz 'N': eigenvalues only; 'V': eigenvalues and row-distributed vectors.
 * @param diag_in Length n diagonal, destroyed on exit.
 * @param e_in Length max(0,n-1) off-diagonal, destroyed on exit.
 * @param w Length n eigenvalues in ascending order, replicated on every rank.
 * @param z This rank's contiguous rows of all n eigenvector columns, column-major
 * in backend memory. For communicator rank q among P ranks, rows start at
 * q*(n/P)+min(q,n%P), with n/P+(q<n%P) rows. Column j corresponds to w[j].
 * May be null on ranks owning no rows. Ignored for 'N'.
 * @param ldz Local leading dimension, at least max(1, local row count) for 'V'.
 * Padding and storage beyond the first n columns are preserved. Ignored for 'N'.
 * @param info Zero on success. Invalid arguments or numerical failure throw.
 * All ranks participate, including ranks without rows. Eigenvectors remain
 * row-distributed throughout the divide-and-conquer tree and on return.
 * There is no descriptor-output overload or implicit 2D redistribution.
 */
template <typename T>
void pstedc(
    const DdlaHandle_t& handle,
    char jobz, int n,
    typename real_type<T>::type* diag_in,
    typename real_type<T>::type* e_in,
    typename real_type<T>::type* w,
    typename real_type<T>::type* z, int ldz,
    int& info
);

// Hermitian eigensolver interfaces; shared by CPU and GPU backends.
template <typename T>
struct hb2st_rot {
    int j0;
    typename real_type<T>::type c;
    T s;
};

/**
 * @brief Distributed Hermitian eigensolver with runtime method selection.
 *
 * DDLA_EIGEN_METHOD=one_stage selects pheevd_one_stage; two_stage selects
 * pheevd_two_stage (the default when unset or empty). The value is read on
 * every call and must agree across the process grid. Invalid values or
 * inconsistent selections throw collectively before either algorithm starts.
 * Both methods read only UPLO, destroy A, and keep A/Z distributed; w is a
 * host array on every rank. See the explicit entries for method requirements.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pheevd(
    const DdlaHandle_t& handle,
    char jobz, char uplo, int n,
    T* A, const int* descA,
    typename real_type<T>::type* w,
    T* Z, const int* descZ,
    int& info
);

/**
 * @brief Two-stage Hermitian eigensolver: full to band to tridiagonal.
 *
 * Same arguments as pheevd. Ignores DDLA_EIGEN_METHOD. All dense matrices
 * remain distributed; the requested band width is reduced on multiple ranks
 * when needed to bound per-rank storage. Reads only UPLO and destroys A.
 * Automatic Givens/Householder and band selection retains conservative
 * fallbacks outside measured profiles. On one CUDA SM70 node using NCCL, eight ranks
 * in a row-major 2x4 grid, one host worker/rank and block64, full-vector double solves
 * from4096 use Householder/band64; complex-double defaults from16384.
 * Stage2 uses CPU below16384 and GPU from16384. DDLA_EIG_NBW is respected;
 * explicit band64 also permits complex-double from8192, while explicit band16
 * selects Householder only for double from32768. Other widths retain Givens.
 * Eigenvalues-only, other devices/grids/thread budgets and float types retain
 * their existing GPU policy. Both algorithms keep quadratic storage distributed.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pheevd_two_stage(const DdlaHandle_t& handle, char jobz, char uplo, int n,
    T* A, const int* descA, typename real_type<T>::type* w,
    T* Z, const int* descZ, int& info);


/**
 * @brief Direct one-stage Hermitian eigensolver using Householder tridiagonalization.
 *
 * Reads the input triangle selected by UPLO, then uses both halves of A as
 * workspace for full-matrix GEMV/GEMM. A is destroyed in both triangles;
 * padding outside the active leading block remains unchanged unless it
 * overlaps Z.
 * w is a host array on every rank; Z remains distributed in descZ. Supports
 * N/V, U/L and all four scalar types. Requires square,
 * source-zero blocks for A and matching output row distribution. The direct
 * reduction buffers 32 rank-two updates; eigenvectors use blocked compact WY.
 * This explicit entry ignores DDLA_EIGEN_METHOD.
 *
 * On affected CUDA/cuBLAS versions, large direct reductions temporarily
 * disable the default process-wide cuBLASLt heuristic cache. An explicit
 * CUBLASLT_HEURISTICS_CACHE_CAPACITY setting is respected. Capacity is
 * restored on return; discarded cache entries are not restored. Coordinate
 * concurrent changes to this vendor process-wide setting.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pheevd_one_stage(const DdlaHandle_t& handle, char jobz, char uplo, int n,
    T* A, const int* descA, typename real_type<T>::type* w,
    T* Z, const int* descZ, int& info);

/**
 * @brief Single-process general matrix product updating one output triangle.
 *
 * Compute uplo='U' or 'L' of C := alpha*op(A)*op(B) + beta*C.
 * op(A) is n-by-k and op(B) is k-by-n. A and B are general matrices;
 * no symmetry or Hermitian structure is assumed. The opposite triangle and
 * padding are untouched. C uses ordinary column-major, not packed, storage.
 *
 * Matrices reside in the handle's backend memory. lda/ldb/ldc are column
 * strides in elements, with conventional BLAS minimum leading dimensions.
 * C must not overlap A/B. No rank communication is performed. Products
 * use batched GEMM with temporary diagonal blocks and pointer tables;
 * private workspace-size/run components allow callers to reuse that memory.
 * GPU work is ordered on the handle's stream without extra synchronization.
 *
 * beta=0 reads no old C; alpha=0 or k=0 reads no A/B. n=0 and
 * alpha=0,beta=1 are no-ops. Complex alpha, beta and diagonal entries are
 * unrestricted. Flags are case-insensitive; transa/transb are N/T/C.
 * The four standard real/complex scalar types are supported.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void gemmt(const DdlaHandle_t& handle, char uplo,
    char transa, char transb, int n, int k,
    const T& alpha, const T* A, int lda, const T* B, int ldb,
    const T& beta, T* C, int ldc);
/** @brief Mixed real/complex product with matching float/double precision.
 * Flags, storage and zero coefficients follow the same-type operation.
 * Alpha, beta and C are complex. A real structured input is symmetric;
 * a complex Hermitian input ignores its imaginary diagonal.
 */
template <DdlaBackend Backend = default_backend_v, typename R>
void gemmt(const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k,
    const std::complex<R>& alpha, const R* A, int lda,
    const std::complex<R>* B, int ldb, const std::complex<R>& beta,
    std::complex<R>* C, int ldc);
template <DdlaBackend Backend = default_backend_v, typename R>
void gemmt(const DdlaHandle_t& handle, char uplo, char transa, char transb, int n, int k,
    const std::complex<R>& alpha, const std::complex<R>* A, int lda,
    const R* B, int ldb, const std::complex<R>& beta,
    std::complex<R>* C, int ldc);


/**
 * @brief Distributed C := alpha*op(A)*op(B) + beta*C, aligned with trtrm.
 *
 * uplo_a/uplo_b name the stored triangles; transa/transb are N/T/C and flip
 * the effective triangle for T/C. C conjugates complex entries and equals T
 * for real types. Flags are case insensitive. Matching effective triangles
 * update only that triangle of C, preserving the opposite half; mixed
 * effective triangles update all of C. Diagonals are explicit (non-unit).
 *
 * All descriptors use matching square blocks and source coordinates. Larger
 * descriptors and padded leading dimensions are supported; only the leading
 * n-by-n region participates. C must not overlap either active input. With
 * alpha=0, A/B are not referenced and may be null; beta=0 does not read old C.
 * n=0 or alpha=0,beta=1 is a no-op. Empty ranks may pass null local buffers.
 * All grid ranks must participate with matching global operation arguments.
 *
 * Transposed panels are redistributed with bounded panel workspace, without
 * replicating full matrices. NN retains the batched triangular sweep. This
 * four-flag interface is the only public ptrtrm interface.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void ptrtrm(const DdlaHandle_t& handle, char uplo_a, char transa, char uplo_b, char transb,
    int n, const T& alpha, const T* A, const int* descA, const T* B, const int* descB,
    const T& beta, T* C, const int* descC);

/** LDL factorization/solve family. Pivoted variants use LAPACK Bunch-Kaufman
 * storage and 1-based host IPIV (replicated length n for distributed calls).
 * _nopiv uses only 1x1 pivots and info>0 reports a zero pivot. Both triangles
 * are supported, only the selected triangle is read/written. SY is transpose
 * symmetric, HE is Hermitian (real HE agrees with SY). Solve overwrites B.
 * info is host memory. Local hetrf<GPU, std::complex<double>> and all four
 * scalar types of hetrf_nopiv<GPU>, phetrf<GPU>, phetrf_nopiv<GPU>, phetrs<GPU>
 * and phetrs_nopiv<GPU> keep matrix arithmetic on the GPU. CPU and GPU HE
 * distributed calls keep quadratic matrix storage 2D block-cyclic.
 * DDLA_LDLT_PANEL=gather (default) caches at most 64 columns on rank 0 and
 * uses one extra distributed n-by-n matrix for pivoted phetrf. Setting it to
 * inplace updates only the caller's selected triangle without that matrix.
 * Both modes replicate bounded-width panel vectors, not a full dense matrix.
 * No-pivot phetrf always updates the selected triangle in place.
 * HE distributed solves compose permutations, unit triangular solves and
 * block-diagonal scaling. Const input factors are copied to distributed
 * storage only when pivot conversion or descriptor normalization requires it;
 * RHS temporaries are also distributed. IPIV and INFO remain host data.
 * Other GPU LDL variants stage arithmetic through host LAPACK. Real
 * distributed SY calls use the distributed HE path (real SY == HE); complex
 * distributed SY factors/solves still gather leading blocks to rank 0 with
 * quadratic root storage. Local GPU hetrs/hetrs_nopiv also retain host staging.
 * Flags/global metadata must agree across ranks.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void sytrf(const DdlaHandle_t& h, char uplo, int n, T* A, int lda, int* ipiv, int& info);
template <DdlaBackend Backend = default_backend_v, typename T>
void sytrs(const DdlaHandle_t& h, char uplo, int n, int nrhs, const T* A, int lda, const int* ipiv, T* B, int ldb, int& info);
template <DdlaBackend Backend = default_backend_v, typename T>
void sytrf_nopiv(const DdlaHandle_t& h, char uplo, int n, T* A, int lda, int& info);
template <DdlaBackend Backend = default_backend_v, typename T>
void sytrs_nopiv(const DdlaHandle_t& h, char uplo, int n, int nrhs, const T* A, int lda, T* B, int ldb, int& info);
template <DdlaBackend Backend = default_backend_v, typename T>
void hetrf(const DdlaHandle_t& h, char uplo, int n, T* A, int lda, int* ipiv, int& info);
template <DdlaBackend Backend = default_backend_v, typename T>
void hetrs(const DdlaHandle_t& h, char uplo, int n, int nrhs, const T* A, int lda, const int* ipiv, T* B, int ldb, int& info);
template <DdlaBackend Backend = default_backend_v, typename T>
void hetrf_nopiv(const DdlaHandle_t& h, char uplo, int n, T* A, int lda, int& info);
template <DdlaBackend Backend = default_backend_v, typename T>
void hetrs_nopiv(const DdlaHandle_t& h, char uplo, int n, int nrhs, const T* A, int lda, T* B, int ldb, int& info);
template <DdlaBackend Backend = default_backend_v, typename T>
void psytrf(const DdlaHandle_t& h, char uplo, int n, T* A, const int* descA, int* ipiv, int& info);
template <DdlaBackend Backend = default_backend_v, typename T>
void psytrs(const DdlaHandle_t& h, char uplo, int n, int nrhs, const T* A, const int* descA, const int* ipiv, T* B, const int* descB, int& info);
template <DdlaBackend Backend = default_backend_v, typename T>
void psytrf_nopiv(const DdlaHandle_t& h, char uplo, int n, T* A, const int* descA, int& info);
template <DdlaBackend Backend = default_backend_v, typename T>
void psytrs_nopiv(const DdlaHandle_t& h, char uplo, int n, int nrhs, const T* A, const int* descA, T* B, const int* descB, int& info);
template <DdlaBackend Backend = default_backend_v, typename T>
void phetrf(const DdlaHandle_t& h, char uplo, int n, T* A, const int* descA, int* ipiv, int& info);
template <DdlaBackend Backend = default_backend_v, typename T>
void phetrs(const DdlaHandle_t& h, char uplo, int n, int nrhs, const T* A, const int* descA, const int* ipiv, T* B, const int* descB, int& info);
template <DdlaBackend Backend = default_backend_v, typename T>
void phetrf_nopiv(const DdlaHandle_t& h, char uplo, int n, T* A, const int* descA, int& info);
template <DdlaBackend Backend = default_backend_v, typename T>
void phetrs_nopiv(const DdlaHandle_t& h, char uplo, int n, int nrhs, const T* A, const int* descA, T* B, const int* descB, int& info);


/** Apply Q from geqrf('U') or pgeqrf, with Q=H(1)...H(k).
 * side L/R and trans N/C are supported; real T is also accepted.
 * The ORMQR name includes complex unitary arithmetic (LAPACK UNMQR).
 * Local A and tau use backend memory. Distributed tau is HOST local-column
 * storage returned by pgeqrf, replicated across process rows. A and tau are
 * read-only and disjoint from C. Only reflector tails below R are read.
 * Distributed descriptors may have different blocks and source coordinates.
 * Only the leading m by n of C is modified. info is zero on success and
 * invalid arguments throw (collectively for distributed interfaces).
 * Distributed scratch includes replicated nq by min(k,32) panels per rank.
 */
template<DdlaBackend Backend=default_backend_v,typename T>
void ormqr(const DdlaHandle_t& h,char side,char trans,int m,int n,int k,
           const T* A,int lda,const T* tau,T* C,int ldc,int& info);
template<DdlaBackend Backend=default_backend_v,typename T>
void pormqr(const DdlaHandle_t& h,char side,char trans,int m,int n,int k,
            const T* A,const int* descA,const T* tau,T* C,const int* descC,int& info);
/** Replace the leading m by n of QR factors A by the first n columns of Q.
 * Requires m>=n>=k>=0. k=0 generates the corresponding identity columns.
 * The ORGQR name includes complex arithmetic (LAPACK UNGQR). tau follows
 * ORMQR/PORMQR memory and ownership conventions and must be disjoint from A.
 * Parent entries and leading-dimension padding are preserved. Reflector tails
 * are saved before A is overwritten. info is zero on success.
 */
template<DdlaBackend Backend=default_backend_v,typename T>
void orgqr(const DdlaHandle_t& h,int m,int n,int k,T* A,int lda,const T* tau,int& info);
template<DdlaBackend Backend=default_backend_v,typename T>
void porgqr(const DdlaHandle_t& h,int m,int n,int k,T* A,const int* descA,const T* tau,int& info);

} // namespace ddla

#endif // DDLA_H
