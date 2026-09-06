#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <cassert>
#include <vector>
#include <algorithm>
#include "ddla_connector.h"
#include "ddla_stream_impl.h"
#include "require_gpu.h"
#include "gemm.h"
#include "trsm.h"
#include "getrf.h"
#include "laswp.h"
#include "comm_traits.h"

namespace ddla {

/**
 * @brief Block LU factorization with partial pivoting within each block row.
 *
 * Right-looking block algorithm: at each step only the nb×nb diagonal block
 * is factored with getrf (local, single-GPU).  Pivoting is "block-partial" --
 * row swaps are confined to the diagonal block and then applied to the full
 * rows (the already-factored columns as well as the right panel) before the
 * trailing-submatrix update.  The output is a standard LU factorization
 * A = P*L*U: every block's row swaps are applied to the whole row, including
 * the L part in the already-factored columns.
 *
 * For each panel step k (global column n_s):
 *   1. Panel LU:     P1 * A11 = L1 * U1          (getrf on local nb×nb block)
 *   2. Apply pivot:  swap rows in right panel and in the factored columns
 *                    [0, n_s) of the current block rows   (desolverLaswp)
 *   3. Compute U12:  U12 = L1^{-1} * B           (trsm LEFT/LOWER/UNIT)
 *   4. Compute L21:  L21 = C * U1^{-1}           (trsm RIGHT/UPPER/NON-UNIT)
 *   5. Update trail: D <- D - L21 * U12          (gemm)
 */
template<typename T>
void pgetrf_bpiv(
    const DdlaHandle_t& handle, const int& m, const int& n,
    T* d_A, const int* array_descA,
    int* d_ipiv, // device, 1-based global row indices
    int& info  // host
)
{
    check_desc(array_descA, handle);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);


    assert(m <= array_descA[DDLA_M_]&& n <= array_descA[DDLA_N_]);
    DdlaHandle_t ddla_handle = handle;
    detail::require_gpu_backend(ddla_handle, "pgetrf_bpiv");



    int nb = array_descA[DDLA_MB_];
    assert(array_descA[DDLA_MB_] == array_descA[DDLA_NB_]);
    int lld = array_descA[DDLA_LLD_];

    int m_loc = num_loc(m, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
    int n_loc = num_loc(n, array_descA[DDLA_NB_], mypcol, array_descA[DDLA_CSRC_], npcols);

    runtimeStream_t stream = ddla_handle->stream;
    deblasHandle_t blasH = ddla_handle->blasH;
    desolverHandle_t solverH = ddla_handle->solverH;

    int nb_real;
    int mm_row_start = 0;  // local row start of trailing submatrix
    int mm_col_start = 0;  // local col start of trailing submatrix

    // Temp buffers
    T* d_temp_block;
    RUNTIME_CHECK(runtimeMallocAsync(&d_temp_block, sizeof(T) * nb * nb, stream));
    T* d_temp_L;
    RUNTIME_CHECK(runtimeMallocAsync(&d_temp_L, sizeof(T) * m_loc * nb, stream));
    T* d_temp_U;
    RUNTIME_CHECK(runtimeMallocAsync(&d_temp_U, sizeof(T) * nb * n_loc, stream));

    int* d_info = nullptr;
    RUNTIME_CHECK(runtimeMallocAsync(&d_info, sizeof(int), stream));

    info = 0;

    for (int n_s = 0; n_s < std::min(m, n); n_s += nb) {
        nb_real = std::min(nb, std::min(m, n) - n_s);

        int i_loc = indx_g2l_r(array_descA, handle, n_s);
        int j_loc = indx_g2l_c(array_descA, handle, n_s);

        int owner_row = indxg2p(n_s, nb, array_descA[DDLA_RSRC_], nprows);
        int owner_col = indxg2p(n_s, nb, array_descA[DDLA_CSRC_], npcols);

        // ================================================================
        // Step 1: Panel LU -- factor the nb×nb diagonal block in place
        // ================================================================
        // The diagonal block A11 lives entirely on process (owner_row, owner_col).
        // Call getrf directly on the local device buffer.
        

        if (myprow == owner_row && mypcol == owner_col) {
            SOLVER_CHECK(desolverGetrf(solverH, nb_real, nb_real, d_A + i_loc + j_loc * lld, lld, d_ipiv + mm_row_start, d_info));
            RUNTIME_CHECK(runtimeMemcpyAsync(&info, d_info, sizeof(int), runtimeMemcpyDeviceToHost, stream));
            RUNTIME_CHECK(runtimeStreamSynchronize(stream));
        }

        MPI_CHECK(MPI_Bcast(&info, 1, MPI_INT, ddla_handle->rc_to_rank(owner_row, owner_col), ddla_handle->comm));
        if (info != 0) {
            info += n_s;
            break;
        }
        if(n_s + nb_real == n){
            // Last block: no right panel, but for a standard A = P*L*U the
            // row swaps must still be applied to the already-factored columns
            // [0, n_s).  getrf wrote these pivots only on (owner_row,
            // owner_col) -- the loop breaks before step 2's row broadcast --
            // so broadcast them first.
            const int l_cols = num_loc(n_s, nb, mypcol, array_descA[DDLA_CSRC_], npcols);
            if(myprow == owner_row){
                commBcast(ddla_handle, CommScope::Row, d_ipiv + mm_row_start, (std::size_t)nb_real, owner_col);
                if(l_cols > 0){
#ifdef DDLA_USE_CUDA
                    SOLVER_CHECK(desolverLaswp(
                        solverH, l_cols,
                        d_A + mm_row_start, lld,
                        1, nb_real,   // 1-based local row range
                        d_ipiv + mm_row_start, 1
                    ));
#endif
#ifdef DDLA_USE_HIP
                    BLAS_CHECK(deblasLaswp(
                        blasH, l_cols,
                        d_A + mm_row_start, lld,
                        1, nb_real,   // 1-based local row range
                        d_ipiv + mm_row_start, 1
                    ));
#endif
                }
            }
            break;
        }

        // ================================================================
        // Step 2: Apply pivot to the right panel  B <- P1 * B
        // ================================================================
        // Instead of using pswap (which involves many pairwise point-to-point
        // communications), we:
        //   1. broadcast the pivot to the row of the owner process.
        //   2. Call desolverLaswp on device to apply all swaps at once.
        //
        // Only processes in the same row as owner_row own the pivoted rows,
        // so only they need to apply the swaps.
        // ================================================================
        // Step 3: Extract/broadcast the factored diagonal block (L1+U1)
        // ================================================================
        if (myprow == owner_row && mypcol == owner_col) {
            RUNTIME_CHECK(runtimeMemcpy2DAsync(
                d_temp_block, nb_real * sizeof(T),
                d_A + i_loc + j_loc * lld, lld * sizeof(T),
                nb_real * sizeof(T), nb_real,
                runtimeMemcpyDeviceToDevice, stream
            ));
        }
        int right_panel_col_start = (j_loc >= 0) ? (j_loc + nb_real) : mm_col_start;
        if (myprow == owner_row) {
            commBcast(ddla_handle, CommScope::Row, d_temp_block, (std::size_t)nb_real * nb_real, owner_col);
            commBcast(ddla_handle, CommScope::Row, d_ipiv + mm_row_start, (std::size_t)nb_real, owner_col);
            // Apply the row swaps to the already-factored columns [0, n_s) of
            // this block's rows, so the factorization is a standard
            // A = P*L*U (the L part is permuted too).  These local columns
            // hold no data of the current trailing submatrix, so this is a
            // purely local laswp.
            const int l_cols = num_loc(n_s, nb, mypcol, array_descA[DDLA_CSRC_], npcols);
            if(l_cols > 0){
#ifdef DDLA_USE_CUDA
                SOLVER_CHECK(desolverLaswp(
                    solverH, l_cols,
                    d_A + mm_row_start, lld,
                    1, nb_real,   // 1-based local row range
                    d_ipiv + mm_row_start, 1
                ));
#endif
#ifdef DDLA_USE_HIP
                BLAS_CHECK(deblasLaswp(
                    blasH, l_cols,
                    d_A + mm_row_start, lld,
                    1, nb_real,   // 1-based local row range
                    d_ipiv + mm_row_start, 1
                ));
#endif
            }
            // Apply row swaps to the local right panel columns [n_s+nb_real, n).
            // The pivoted rows in the local matrix start at i_loc.
            // Number of local columns in the right panel: those from n_s+nb_real to n-1.
            if (n_loc > right_panel_col_start) {
                // d_right_panel points to column n_s+nb_real (the start of right panel).
                // When j_loc >= 0, right panel starts at local column j_loc + nb_real.
                // When j_loc < 0, all local columns are in the right panel, starting at 0.
                
                T* d_right_panel = d_A + right_panel_col_start * lld + mm_row_start;
#ifdef DDLA_USE_CUDA
                SOLVER_CHECK(desolverLaswp(
                    solverH, n_loc - right_panel_col_start,
                    d_right_panel, lld,
                    1, nb_real,   // 1-based local row range
                    d_ipiv + mm_row_start, 1
                ));
#endif
#ifdef DDLA_USE_HIP
                BLAS_CHECK(deblasLaswp(
                    blasH, n_loc - right_panel_col_start,
                    d_right_panel, lld,
                    1, nb_real,   // 1-based local row range
                    d_ipiv + mm_row_start, 1
                ));

#endif
                // ================================================================
                // Step 4: U12 = L1^{-1} * B  (trsm LEFT LOWER UNIT)
                // ================================================================
                BLAS_CHECK(deblasTrsm(
                    blasH,
                    DEBLAS_SIDE_LEFT, DEBLAS_FILL_MODE_LOWER,
                    DEBLAS_OP_N, DEBLAS_DIAG_UNIT,
                    nb_real, n_loc - right_panel_col_start, T(1.0),
                    d_temp_block, nb_real,
                    d_right_panel, lld
                ));
                RUNTIME_CHECK(runtimeMemcpy2DAsync(
                    d_temp_U, nb_real * sizeof(T),
                    d_right_panel, lld * sizeof(T),
                    nb_real * sizeof(T), n_loc - right_panel_col_start,
                    runtimeMemcpyDeviceToDevice, stream
                ));
            }
            
        }
        // ================================================================
        // Step 5: L21 = C * U1^{-1}  (trsm RIGHT UPPER NON-UNIT)
        // ================================================================
        int left_panel_row_start = (i_loc >= 0) ? (i_loc + nb_real) : mm_row_start;
        if(mypcol == owner_col){
            commBcast(ddla_handle, CommScope::Col, d_temp_block, (std::size_t)nb_real * nb_real, owner_row);
            if(m_loc > left_panel_row_start){
                T* d_left_panel = d_A + mm_col_start * lld + left_panel_row_start;
                BLAS_CHECK(deblasTrsm(
                    blasH,
                    DEBLAS_SIDE_RIGHT, DEBLAS_FILL_MODE_UPPER,
                    DEBLAS_OP_N, DEBLAS_DIAG_NON_UNIT,
                    m_loc - left_panel_row_start, nb_real, T(1.0),
                    d_temp_block, nb_real,
                    d_left_panel, lld
                ));
                RUNTIME_CHECK(runtimeMemcpy2DAsync(
                    d_temp_L, (m_loc - left_panel_row_start) * sizeof(T),
                    d_left_panel, lld * sizeof(T),
                    (m_loc - left_panel_row_start) * sizeof(T), nb_real,
                    runtimeMemcpyDeviceToDevice, stream
                ));
            }
        }
        if(n_loc > right_panel_col_start) {
            commBcast(ddla_handle, CommScope::Col, d_temp_U, (std::size_t)nb_real * (n_loc - right_panel_col_start), owner_row);
        }
        if(m_loc > left_panel_row_start){
            commBcast(ddla_handle, CommScope::Row, d_temp_L, (std::size_t)(m_loc - left_panel_row_start) * nb_real, owner_col);
        }

        // ================================================================
        // Step 6: Schur-complement update  D <- D - L21 * U12
        // ================================================================
        int trailing_m = m_loc - left_panel_row_start;
        int trailing_n = n_loc - right_panel_col_start;

        if(trailing_m > 0 && trailing_n > 0){
            gemm<DdlaBackend::GPU, T>(ddla_handle, 'N', 'N',
                trailing_m, trailing_n, nb_real,
                T(-1.0),
                d_temp_L, trailing_m,
                d_temp_U, nb_real,
                T(1.0),
                d_A + left_panel_row_start + right_panel_col_start * lld, lld);
        }

        RUNTIME_CHECK(runtimeStreamSynchronize(stream));

        // Advance local pointers for next panel
        if (i_loc >= 0)
            mm_row_start += nb;
        if (j_loc >= 0)
            mm_col_start += nb;
    }

    RUNTIME_CHECK(runtimeFreeAsync(d_temp_block, stream));
    RUNTIME_CHECK(runtimeFreeAsync(d_temp_L, stream));
    RUNTIME_CHECK(runtimeFreeAsync(d_temp_U, stream));
    RUNTIME_CHECK(runtimeFreeAsync(d_info, stream));
    RUNTIME_CHECK(runtimeStreamSynchronize(stream));
}

// Explicit instantiations
template void pgetrf_bpiv<float>(
    const DdlaHandle_t&, const int& m, const int& n,
    float* d_A, const int* array_descA,
    int* ipiv, int& info
);
template void pgetrf_bpiv<double>(
    const DdlaHandle_t&, const int& m, const int& n,
    double* d_A, const int* array_descA,
    int* ipiv, int& info
);
template void pgetrf_bpiv<std::complex<float>>(
    const DdlaHandle_t&, const int& m, const int& n,
    std::complex<float>* d_A, const int* array_descA,
    int* ipiv, int& info
);
template void pgetrf_bpiv<std::complex<double>>(
    const DdlaHandle_t&, const int& m, const int& n,
    std::complex<double>* d_A, const int* array_descA,
    int* ipiv, int& info
);

} // namespace ddla
