#include <ddla/ddla.h>
#include "lu_internal.h"
#include "ddla_desc.h"
#include <cassert>
#include <vector>
#include <algorithm>
#include "ddla_connector.h"
#include "ddla_stream_impl.h"
#include "require_backend.h"
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
 *                    [0, n_s) of the current block rows   (laswp)
 *   3. Compute U12:  U12 = L1^{-1} * B           (trsm LEFT/LOWER/UNIT)
 *   4. Compute L21:  L21 = C * U1^{-1}           (trsm RIGHT/UPPER/NON-UNIT)
 *   5. Update trail: D <- D - L21 * U12          (gemm)
 */
template <DdlaBackend Backend, typename T>
void pgetrf_bpiv(
    const DdlaHandle_t& handle, int m, int n,
    T* A, const int* array_descA,
    int* pivots, // host for CPU, device for GPU; 1-based global row indices
    int& info  // host
)
{
    check_desc(array_descA, handle);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);


    assert(m <= array_descA[DDLA_M_]&& n <= array_descA[DDLA_N_]);
    DdlaHandle_t ddla_handle = handle;
    detail::require_backend<Backend>(ddla_handle, "pgetrf_bpiv");



    int nb = array_descA[DDLA_MB_];
    assert(array_descA[DDLA_MB_] == array_descA[DDLA_NB_]);
    int lld = array_descA[DDLA_LLD_];

    int m_loc = num_loc(m, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
    int n_loc = num_loc(n, array_descA[DDLA_NB_], mypcol, array_descA[DDLA_CSRC_], npcols);

    int nb_real;
    int mm_row_start = 0;  // local row start of trailing submatrix
    int mm_col_start = 0;  // local col start of trailing submatrix

    // Temp buffers
    T* temp_block;
    DDLA_CHECK(ddlaMalloc(ddla_handle, &temp_block, sizeof(T) * nb * nb));
    T* temp_L;
    DDLA_CHECK(ddlaMalloc(ddla_handle, &temp_L, sizeof(T) * m_loc * nb));
    T* temp_U;
    DDLA_CHECK(ddlaMalloc(ddla_handle, &temp_U, sizeof(T) * nb * n_loc));

    int* info_buffer = nullptr;
    DDLA_CHECK(ddlaMalloc(ddla_handle, &info_buffer, sizeof(int)));

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
        // Call getrf directly on the local backend buffer.
        

        if (myprow == owner_row && mypcol == owner_col) {
            getrf<Backend, T>(ddla_handle, nb_real, nb_real, A + i_loc + j_loc * lld, lld, pivots + mm_row_start, info_buffer);
            DDLA_CHECK(ddlaMemcpy(ddla_handle, &info, info_buffer, sizeof(int)));
            DDLA_CHECK(ddlaSynchronize(ddla_handle));
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
                commBcast<Backend>(ddla_handle, CommScope::Row, pivots + mm_row_start, (std::size_t)nb_real, owner_col);
                if(l_cols > 0){
                    laswp<Backend, T>(
                        ddla_handle, l_cols,
                        A + mm_row_start, lld,
                        1, nb_real,   // 1-based local row range
                        pivots + mm_row_start, 1
                    );
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
        //   2. Call laswp in the selected backend to apply all swaps at once.
        //
        // Only processes in the same row as owner_row own the pivoted rows,
        // so only they need to apply the swaps.
        // ================================================================
        // Step 3: Extract/broadcast the factored diagonal block (L1+U1)
        // ================================================================
        if (myprow == owner_row && mypcol == owner_col) {
            DDLA_CHECK(ddlaMemcpy2D(ddla_handle, temp_block, nb_real * sizeof(T), A + i_loc + j_loc * lld,
                                    lld * sizeof(T), nb_real * sizeof(T), nb_real));
        }
        int right_panel_col_start = (j_loc >= 0) ? (j_loc + nb_real) : mm_col_start;
        if (myprow == owner_row) {
            commBcast<Backend>(ddla_handle, CommScope::Row, temp_block, (std::size_t)nb_real * nb_real, owner_col);
            commBcast<Backend>(ddla_handle, CommScope::Row, pivots + mm_row_start, (std::size_t)nb_real, owner_col);
            // Apply the row swaps to the already-factored columns [0, n_s) of
            // this block's rows, so the factorization is a standard
            // A = P*L*U (the L part is permuted too).  These local columns
            // hold no data of the current trailing submatrix, so this is a
            // purely local laswp.
            const int l_cols = num_loc(n_s, nb, mypcol, array_descA[DDLA_CSRC_], npcols);
            if(l_cols > 0){
                laswp<Backend, T>(
                    ddla_handle, l_cols,
                    A + mm_row_start, lld,
                    1, nb_real,   // 1-based local row range
                    pivots + mm_row_start, 1
                );
            }
            // Apply row swaps to the local right panel columns [n_s+nb_real, n).
            // The pivoted rows in the local matrix start at i_loc.
            // Number of local columns in the right panel: those from n_s+nb_real to n-1.
            if (n_loc > right_panel_col_start) {
                // right_panel points to column n_s+nb_real (the start of right panel).
                // When j_loc >= 0, right panel starts at local column j_loc + nb_real.
                // When j_loc < 0, all local columns are in the right panel, starting at 0.
                
                T* right_panel = A + right_panel_col_start * lld + mm_row_start;
                laswp<Backend, T>(
                    ddla_handle, n_loc - right_panel_col_start,
                    right_panel, lld,
                    1, nb_real,   // 1-based local row range
                    pivots + mm_row_start, 1
                );
                // ================================================================
                // Step 4: U12 = L1^{-1} * B  (trsm LEFT LOWER UNIT)
                // ================================================================
                trsm<Backend, T>(
                    ddla_handle,
                    'L', 'L',
                    'N', 'U',
                    nb_real, n_loc - right_panel_col_start, T(1.0),
                    temp_block, nb_real,
                    right_panel, lld
                );
                DDLA_CHECK(ddlaMemcpy2D(ddla_handle, temp_U, nb_real * sizeof(T), right_panel, lld * sizeof(T),
                                        nb_real * sizeof(T), n_loc - right_panel_col_start));
            }
            
        }
        // ================================================================
        // Step 5: L21 = C * U1^{-1}  (trsm RIGHT UPPER NON-UNIT)
        // ================================================================
        int left_panel_row_start = (i_loc >= 0) ? (i_loc + nb_real) : mm_row_start;
        if(mypcol == owner_col){
            commBcast<Backend>(ddla_handle, CommScope::Col, temp_block, (std::size_t)nb_real * nb_real, owner_row);
            if(m_loc > left_panel_row_start){
                T* left_panel = A + mm_col_start * lld + left_panel_row_start;
                trsm<Backend, T>(
                    ddla_handle,
                    'R', 'U',
                    'N', 'N',
                    m_loc - left_panel_row_start, nb_real, T(1.0),
                    temp_block, nb_real,
                    left_panel, lld
                );
                DDLA_CHECK(ddlaMemcpy2D(ddla_handle, temp_L, (m_loc - left_panel_row_start) * sizeof(T),
                                        left_panel, lld * sizeof(T), (m_loc - left_panel_row_start) * sizeof(T),
                                        nb_real));
            }
        }
        if(n_loc > right_panel_col_start) {
            commBcast<Backend>(ddla_handle, CommScope::Col, temp_U, (std::size_t)nb_real * (n_loc - right_panel_col_start), owner_row);
        }
        if(m_loc > left_panel_row_start){
            commBcast<Backend>(ddla_handle, CommScope::Row, temp_L, (std::size_t)(m_loc - left_panel_row_start) * nb_real, owner_col);
        }

        // ================================================================
        // Step 6: Schur-complement update  D <- D - L21 * U12
        // ================================================================
        int trailing_m = m_loc - left_panel_row_start;
        int trailing_n = n_loc - right_panel_col_start;

        if(trailing_m > 0 && trailing_n > 0){
            gemm<Backend, T>(ddla_handle, 'N', 'N',
                trailing_m, trailing_n, nb_real,
                T(-1.0),
                temp_L, trailing_m,
                temp_U, nb_real,
                T(1.0),
                A + left_panel_row_start + right_panel_col_start * lld, lld);
        }

        DDLA_CHECK(ddlaSynchronize(ddla_handle));

        // Advance local pointers for next panel
        if (i_loc >= 0)
            mm_row_start += nb;
        if (j_loc >= 0)
            mm_col_start += nb;
    }

    DDLA_CHECK(ddlaFree(ddla_handle, temp_block));
    DDLA_CHECK(ddlaFree(ddla_handle, temp_L));
    DDLA_CHECK(ddlaFree(ddla_handle, temp_U));
    DDLA_CHECK(ddlaFree(ddla_handle, info_buffer));
    DDLA_CHECK(ddlaSynchronize(ddla_handle));
}

// Explicit instantiations
#define INSTANTIATE_PGETRF_BPIV(BACKEND, TYPE)                                   \
    template void pgetrf_bpiv<BACKEND, TYPE>(const DdlaHandle_t&, int m, int n, \
    TYPE* A, const int* array_descA, \
    int* ipiv, int& info)

#if DDLA_HAS_CPU
INSTANTIATE_PGETRF_BPIV(DdlaBackend::CPU, float);
INSTANTIATE_PGETRF_BPIV(DdlaBackend::CPU, double);
INSTANTIATE_PGETRF_BPIV(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PGETRF_BPIV(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_PGETRF_BPIV(DdlaBackend::GPU, float);
INSTANTIATE_PGETRF_BPIV(DdlaBackend::GPU, double);
INSTANTIATE_PGETRF_BPIV(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PGETRF_BPIV(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_PGETRF_BPIV


} // namespace ddla
