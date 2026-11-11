#include "BLAS/blas_impl.h"
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "lu_internal.h"
#include "ddla_desc.h"
#include <cassert>
#include <cstddef>
#include <vector>
#include <algorithm>
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "require_backend.h"
#include "comm_traits.h"
#include "check_agree.h"
#include "plan_util.h"
#include "lapack_impl.h"
#include "pgetrf_impl.h"
#include "instantiate.h"
#include <complex>
#include <stdexcept>
#include <string>

namespace ddla {

namespace lu_impl {

// ---------------------------------------------------------------------------
// The block LU behind pgetrf_bpiv and pgetrf_nopiv.  Both are the same right-looking schedule; `pivot` selects
// whether the diagonal block is factored with getrf (its row swaps then applied to the whole block row) or with
// getrf_nopiv.  Without pivoting nothing of the pivot path is sized, allocated or communicated.
// ---------------------------------------------------------------------------

// Component 1: null / mismatched handle (std::runtime_error) after the descriptor, then the two former asserts as
// std::invalid_argument.  Sizes are otherwise unchecked: m <= 0 or n <= 0 is an empty factorization.
template <DdlaBackend Backend>
void pgetrf_block_check(const DdlaHandle_t& handle, const char* name, int m, int n, const int* array_descA)
{
    detail::agree_check(handle, name, [&](detail::Agree&) {
        check_desc(array_descA, handle);
        detail::require_backend<Backend>(handle, name);
        if (m > array_descA[DDLA_M_] || n > array_descA[DDLA_N_])
            throw std::invalid_argument(std::string(name) + ": m x n exceeds the descriptor of A");
        if (array_descA[DDLA_MB_] != array_descA[DDLA_NB_])
            throw std::invalid_argument(std::string(name) + ": A needs square blocks");
    });
}

// Component 2.  Backend: [ the diagonal block's info word | the diagonal block (nb x nb), which the L panel (local
// rows x nb) reuses after the step's last trsm | the U panel (nb x local columns) ].  With pivoting the vendor
// workspace of the diagonal block's getrf shares the region of the last two: the factorization is synchronized
// before the block is copied out, and every step ends with a synchronize.  The diagonal blocks of the schedule are
// nb x nb and, for the last step, tail x tail; the vendor query is made only on the ranks that hold such a block
// (with A and its leading dimension).  Host: laswp's pivot staging for the widest block (pivoting only).
template <typename T>
void pgetrf_block_workspace_size(const DdlaHandle_t& handle, bool pivot, int m, int n, T* A,
                                 const int* array_descA, std::size_t* backend_bytes, std::size_t* host_bytes)
{
    *backend_bytes = *host_bytes = 0;
    if (m <= 0 || n <= 0) return;
    const int nb = array_descA[DDLA_MB_];
    const int m_loc = num_loc(m, nb, handle->myprow_, array_descA[DDLA_RSRC_], handle->nprows_);
    const int n_loc = num_loc(n, array_descA[DDLA_NB_], handle->mypcol_, array_descA[DDLA_CSRC_], handle->npcols_);
    std::size_t getrf_bytes = 0;
    if (pivot) {
        int previous_width = 0;
        for (int k = 0; k < std::min(m, n); k += nb) {
            const int width = std::min(nb, std::min(m, n) - k);
            if (indxg2p(k, nb, array_descA[DDLA_RSRC_], handle->nprows_) != handle->myprow_ ||
                indxg2p(k, nb, array_descA[DDLA_CSRC_], handle->npcols_) != handle->mypcol_ || width == previous_width)
                continue;
            std::size_t bytes = 0;
            lapack_impl::getrf_workspace_size<T>(handle, width, width, A, array_descA[DDLA_LLD_], &bytes);
            getrf_bytes = std::max(getrf_bytes, bytes);
            previous_width = width;
        }
        lapack_impl::laswp_workspace_size(handle, 1, std::min(nb, std::min(m, n)), 1, host_bytes);
    }
    *backend_bytes = detail::padded_bytes<int>(1)
                   + std::max(detail::padded_bytes<T>(std::size_t(std::max(nb, m_loc)) * nb)
                              + detail::padded_bytes<T>(std::size_t(nb) * n_loc), getrf_bytes);
}

// Component 3.  For each panel step (global column n_s):
//   1) factor the diagonal block: P1 * A11 = L1 * U1 (getrf), or A11 = L1 * U1 (getrf_nopiv)
//   2) broadcast the factored block along its process row and column
//   3) pivoting only: apply the block's row swaps to the already-factored columns and to the right panel (laswp)
//   4) U12 = L1^{-1} * B   (trsm LEFT/LOWER/UNIT)
//   5) L21 = C * U1^{-1}   (trsm RIGHT/UPPER/NON-UNIT)
//   6) broadcast the U panel down the column and the L panel across the row; D <- D - L21 * U12 (gemm)
template <DdlaBackend Backend, typename T>
void pgetrf_block_run(const DdlaHandle_t& handle, bool pivot, int m, int n, T* A,
                      const int* array_descA, int* pivots, // host for CPU, device for GPU; unused without pivoting
                      int& info,  // host
                      void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes)
{
    try {
        info = 0;
        if (m <= 0 || n <= 0) return;

        int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
        ddlaGetGridDims(handle, nprows, npcols);
        ddlaGetGridCoords(handle, myprow, mypcol);

        const int nb = array_descA[DDLA_MB_];
        const int lld = array_descA[DDLA_LLD_];

        const int m_loc = num_loc(m, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
        const int n_loc = num_loc(n, array_descA[DDLA_NB_], mypcol, array_descA[DDLA_CSRC_], npcols);

        int mm_row_start = 0;  // local row start of trailing submatrix
        int mm_col_start = 0;  // local col start of trailing submatrix

        // The scratch, in the order pgetrf_block_workspace_size counts it; the getrf's vendor workspace is the
        // region after the info word.
        char* w = static_cast<char*>(backend_work);
        int* const info_buffer = detail::take<int>(w, 1);
        char* const getrf_work = w;
        const std::size_t getrf_bytes = backend_bytes - std::size_t(getrf_work - static_cast<char*>(backend_work));
        T* const temp_block = detail::take<T>(w, std::size_t(std::max(nb, m_loc)) * nb);
        T* const temp_L = temp_block;
        T* const temp_U = detail::take<T>(w, std::size_t(nb) * n_loc);
        assert(std::size_t(w - static_cast<char*>(backend_work)) <= backend_bytes);

        for (int n_s = 0; n_s < std::min(m, n); n_s += nb) {
            const int nb_real = std::min(nb, std::min(m, n) - n_s);

            const int i_loc = indx_g2l_r(array_descA, handle, n_s);
            const int j_loc = indx_g2l_c(array_descA, handle, n_s);

            const int owner_row = indxg2p(n_s, nb, array_descA[DDLA_RSRC_], nprows);
            const int owner_col = indxg2p(n_s, nb, array_descA[DDLA_CSRC_], npcols);
            const bool on_row = myprow == owner_row, on_col = mypcol == owner_col;

            // The block's row swaps on `columns` local columns starting at `first` (the owner process row only).
            // The pivoted rows are local to that process row, so this is a purely local laswp.
            auto swap_rows = [&](T* first, int columns) {
                if (columns > 0)
                    lapack_impl::laswp_run<Backend, T>(handle, columns, first, lld, 1, nb_real,  // 1-based local rows
                                                       pivots + mm_row_start, 1, host_work, host_bytes);
            };
            // Local columns of the already-factored part [0, n_s): for a standard A = P*L*U the swaps reach them too.
            const int l_cols = pivot ? num_loc(n_s, nb, mypcol, array_descA[DDLA_CSRC_], npcols) : 0;

            // ================================================================
            // Step 1: factor the diagonal block in place
            // ================================================================
            // A11 lives entirely on process (owner_row, owner_col): a local factorization of the backend buffer.
            if (on_row && on_col) {
                T* const diagonal = A + i_loc + std::size_t(j_loc) * lld;
                if (pivot)
                    lapack_impl::getrf_run<Backend, T>(handle, nb_real, nb_real, diagonal, lld,
                                                       pivots + mm_row_start, info_buffer, getrf_work, getrf_bytes);
                else
                    lapack_impl::getrf_nopiv_run<Backend, T>(handle, nb_real, nb_real, diagonal, lld, info_buffer);
                DDLA_CHECK(ddlaMemcpy(handle, &info, info_buffer, sizeof(int)));
                DDLA_CHECK(ddlaSynchronize(handle));
            }

            MPI_CHECK(MPI_Bcast(&info, 1, MPI_INT, handle->rc_to_rank(owner_row, owner_col), handle->comm));
            if (info != 0) {
                // The local factorization returns a 1-based index within the diagonal block; shift it to the
                // global position.
                info += n_s;
                break;
            }
            if (n_s + nb_real == n && m <= n) {
                // Last block column with no rows below it: nothing left to solve.  (When m > n the rows below still
                // need L21 = A21 * U1^{-1}, so the general path below runs; its right panel and update are empty.)
                // The row swaps must still reach the already-factored columns; getrf wrote the pivots only on
                // (owner_row, owner_col), so broadcast them first.
                if (pivot && on_row) {
                    commBcast<Backend>(handle, CommScope::Row, pivots + mm_row_start, (std::size_t)nb_real, owner_col);
                    swap_rows(A + mm_row_start, l_cols);
                }
                break;
            }

            // ================================================================
            // Step 2: extract/broadcast the factored diagonal block (L1+U1)
            // ================================================================
            if (on_row && on_col) {
                DDLA_CHECK(ddlaMemcpy2D(handle, temp_block, nb_real * sizeof(T), A + i_loc + std::size_t(j_loc) * lld,
                                        lld * sizeof(T), nb_real * sizeof(T), nb_real));
            }
            if (on_row) {
                commBcast<Backend>(handle, CommScope::Row, temp_block, (std::size_t)nb_real * nb_real, owner_col);
            }
            if (on_col) {
                commBcast<Backend>(handle, CommScope::Col, temp_block, (std::size_t)nb_real * nb_real, owner_row);
            }

            // ================================================================
            // Steps 3 and 4: B <- P1 * B (pivoting), U12 = L1^{-1} * B  (trsm LEFT/LOWER/UNIT)
            // ================================================================
            // Only the owner process row holds the pivoted rows: one broadcast of the pivots along it and one
            // laswp per panel instead of pairwise row exchanges.
            const int right_panel_col_start = (j_loc >= 0) ? (j_loc + nb_real) : mm_col_start;
            if (on_row) {
                if (pivot) {
                    commBcast<Backend>(handle, CommScope::Row, pivots + mm_row_start, (std::size_t)nb_real, owner_col);
                    swap_rows(A + mm_row_start, l_cols);
                }
                if (n_loc > right_panel_col_start) {
                    T* right_panel = A + std::size_t(right_panel_col_start) * lld + mm_row_start;
                    if (pivot) swap_rows(right_panel, n_loc - right_panel_col_start);
                    blas_impl::trsm_run<Backend, T>(
                        handle,
                        'L', 'L',
                        'N', 'U',
                        nb_real, n_loc - right_panel_col_start, T(1.0),
                        temp_block, nb_real,
                        right_panel, lld
                    );
                    DDLA_CHECK(ddlaMemcpy2D(handle, temp_U, nb_real * sizeof(T), right_panel, lld * sizeof(T),
                                            nb_real * sizeof(T), n_loc - right_panel_col_start));
                }
            }

            // ================================================================
            // Step 5: L21 = C * U1^{-1}  (trsm RIGHT/UPPER/NON-UNIT)
            // ================================================================
            const int left_panel_row_start = (i_loc >= 0) ? (i_loc + nb_real) : mm_row_start;
            if (on_col) {
                if (m_loc > left_panel_row_start) {
                    T* left_panel = A + std::size_t(mm_col_start) * lld + left_panel_row_start;
                    blas_impl::trsm_run<Backend, T>(
                        handle,
                        'R', 'U',
                        'N', 'N',
                        m_loc - left_panel_row_start, nb_real, T(1.0),
                        temp_block, nb_real,
                        left_panel, lld
                    );
                    DDLA_CHECK(ddlaMemcpy2D(handle, temp_L, (m_loc - left_panel_row_start) * sizeof(T),
                                            left_panel, lld * sizeof(T), (m_loc - left_panel_row_start) * sizeof(T),
                                            nb_real));
                }
            }

            // ================================================================
            // Step 6: broadcast the U panel down the column and the L panel across the row;
            //         Schur-complement update  D <- D - L21 * U12
            // ================================================================
            if (n_loc > right_panel_col_start) {
                commBcast<Backend>(handle, CommScope::Col, temp_U, (std::size_t)nb_real * (n_loc - right_panel_col_start), owner_row);
            }
            if (m_loc > left_panel_row_start) {
                commBcast<Backend>(handle, CommScope::Row, temp_L, (std::size_t)(m_loc - left_panel_row_start) * nb_real, owner_col);
            }

            const int trailing_m = m_loc - left_panel_row_start;
            const int trailing_n = n_loc - right_panel_col_start;

            if (trailing_m > 0 && trailing_n > 0) {
                blas_impl::gemm_run<Backend, T>(handle, 'N', 'N',
                    trailing_m, trailing_n, nb_real,
                    T(-1.0),
                    temp_L, trailing_m,
                    temp_U, nb_real,
                    T(1.0),
                    A + left_panel_row_start + std::size_t(right_panel_col_start) * lld, lld);
            }

            DDLA_CHECK(ddlaSynchronize(handle));

            // Advance local pointers for next panel
            if (i_loc >= 0)
                mm_row_start += nb;
            if (j_loc >= 0)
                mm_col_start += nb;
        }

        DDLA_CHECK(ddlaSynchronize(handle));
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(handle) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(handle); } catch (...) {}
        throw;
    }
}

// pgetrf_bpiv's three components: the block LU with pivoting.
template <DdlaBackend Backend>
void pgetrf_bpiv_check(const DdlaHandle_t& handle, int m, int n, const int* array_descA)
{
    pgetrf_block_check<Backend>(handle, "pgetrf_bpiv", m, n, array_descA);
}

template <typename T>
void pgetrf_bpiv_workspace_size(const DdlaHandle_t& handle, int m, int n, T* A, const int* array_descA,
                                std::size_t* backend_bytes, std::size_t* host_bytes)
{
    pgetrf_block_workspace_size<T>(handle, true, m, n, A, array_descA, backend_bytes, host_bytes);
}

template <DdlaBackend Backend, typename T>
void pgetrf_bpiv_run(const DdlaHandle_t& handle, int m, int n, T* A,
                     const int* array_descA, int* pivots, // host for CPU, device for GPU; 1-based global row indices
                     int& info,  // host
                     void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes)
{
    pgetrf_block_run<Backend, T>(handle, true, m, n, A, array_descA, pivots, info,
                                 backend_work, backend_bytes, host_work, host_bytes);
}

} // namespace lu_impl

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
    // 1. parameter check (collective).
    lu_impl::pgetrf_bpiv_check<Backend>(handle, m, n, array_descA);
    // No-op: an empty factorization.
    info = 0;
    if (m <= 0 || n <= 0) return;
    // 2. the scratch.
    std::size_t backend_bytes = 0, host_bytes = 0;
    lu_impl::pgetrf_bpiv_workspace_size<T>(handle, m, n, A, array_descA, &backend_bytes, &host_bytes);
    vector<char> backend_work(handle, backend_bytes);
    std::vector<char> host_work(host_bytes);
    // 3. the factorization.
    lu_impl::pgetrf_bpiv_run<Backend, T>(handle, m, n, A, array_descA, pivots, info,
                                         backend_work.data(), backend_bytes, host_work.data(), host_bytes);
}

// Explicit instantiations
#define INSTANTIATE_PGETRF_BPIV(BACKEND, TYPE)                                   \
    template void pgetrf_bpiv<BACKEND, TYPE>(const DdlaHandle_t&, int m, int n, \
    TYPE* A, const int* array_descA, \
    int* ipiv, int& info); \
    template void lu_impl::pgetrf_bpiv_run<BACKEND, TYPE>(const DdlaHandle_t&, int, int, TYPE*, const int*, int*, \
    int&, void*, std::size_t, void*, std::size_t); \
    template void lu_impl::pgetrf_block_run<BACKEND, TYPE>(const DdlaHandle_t&, bool, int, int, TYPE*, const int*, \
    int*, int&, void*, std::size_t, void*, std::size_t)

#define INSTANTIATE_PGETRF_BPIV_SIZE(TYPE) \
    template void lu_impl::pgetrf_bpiv_workspace_size<TYPE>(const DdlaHandle_t&, int, int, TYPE*, const int*, \
    std::size_t*, std::size_t*); \
    template void lu_impl::pgetrf_block_workspace_size<TYPE>(const DdlaHandle_t&, bool, int, int, TYPE*, const int*, \
    std::size_t*, std::size_t*)

#define INSTANTIATE_PGETRF_BPIV_CHECK(BACKEND) \
    template void lu_impl::pgetrf_bpiv_check<BACKEND>(const DdlaHandle_t&, int, int, const int*); \
    template void lu_impl::pgetrf_block_check<BACKEND>(const DdlaHandle_t&, const char*, int, int, const int*)


DDLA_FOR_EACH_BACKEND(INSTANTIATE_PGETRF_BPIV_CHECK)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_PGETRF_BPIV)
DDLA_FOR_EACH_TYPE(INSTANTIATE_PGETRF_BPIV_SIZE)

#undef INSTANTIATE_PGETRF_BPIV_SIZE
#undef INSTANTIATE_PGETRF_BPIV_CHECK
#undef INSTANTIATE_PGETRF_BPIV

} // namespace ddla
