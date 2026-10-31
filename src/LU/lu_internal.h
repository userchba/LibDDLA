#ifndef DDLA_LU_INTERNAL_H
#define DDLA_LU_INTERNAL_H

#include <ddla/ddla.h>

// Private LU entry points. pgetf2 is the panel factorization the LU drivers
// are built on; getrf_nopiv is the local (single-process) kernel. The
// distributed factorization and solve families (pgetrf, pgetrf_nopiv,
// pgetrf_bpiv, pgetrs*, pgesv*) are public API declared in <ddla/ddla.h>.
namespace ddla {

/**
 * @brief Blocked LU width factorization of a selected submatrix.
 *
 * ia,ja are one-based global indices. m,n are the dimensions of the selected
 * region A(ia:ia+m-1,ja:ja+n-1); width leading columns are factored. The panel
 * must fit one descriptor block column; it may span block rows. Swaps affect only the selected
 * region. ipiv uses local descriptor row indices and absolute one-based global
 * pivot rows; entries outside the panel are untouched. Positive info is the
 * one-based failing position in the selected region. Empty panels access no data.
 *
 * DDLA_LU_PANEL selects the method, agreed once per call by all handle ranks:
 * 'inplace' (default) factors every inner subpanel on the distributed A,
 * 'gather' gathers each subpanel to one rank and factors it with a local
 * getrf. Inner subpanels of at most min(32, nb/2) columns are followed by
 * BLAS-3 trsm/gemm updates of the remaining panel columns; a panel no wider
 * than one subpanel degenerates to the plain per-column factorization.
 * pgetrf uses this schedule for both selectors and retains earlier L columns
 * inside its original selected region when applying later panel pivots.
 *
 * pgetf2 is check -> workspace_size -> run (lu_workspace.h): the check validates
 * collectively and freezes DDLA_LU_PANEL, the query returns local lengths, and
 * the run borrows caller storage without checks, policy reads or allocation.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pgetf2(const DdlaHandle_t& handle, int m, int n, int width,
            T* A, int ia, int ja, const int* descA, int* ipiv, int& info);

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
 * @param A      Pointer in the selected backend to matrix A (input/output -- L+U factors).
 * @param lda      Leading dimension of A.
 * @param info_buffer      Pointer in the selected backend to info (0 on success, k > 0 if U(k,k) is exactly zero).
 * @param ddla_handle DDLA handle (provides stream and BLAS handle).
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void getrf_nopiv(int m, int n, T* A, int lda, int* info_buffer, const DdlaHandle_t& ddla_handle);

} // namespace ddla

#endif // DDLA_LU_INTERNAL_H
