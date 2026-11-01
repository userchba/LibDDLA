#include "BLAS/blas_impl.h"
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "lu_internal.h"
#include "ddla_desc.h"
#include <cassert>
#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "require_backend.h"
#include "comm_traits.h"
#include "check_agree.h"
#include "plan_util.h"
#include "lapack_impl.h"
#include "pgetrf_impl.h"

namespace ddla {

namespace lu_impl {

// Component 1: null / mismatched handle (std::runtime_error) after the descriptor, then the two former asserts as
// std::invalid_argument.  Sizes are otherwise unchecked: m <= 0 or n <= 0 is an empty factorization.
template <DdlaBackend Backend>
void pgetrf_nopiv_check(const DdlaHandle_t& handle, int m, int n, const int* array_descA)
{
    detail::agree_check(handle, "pgetrf_nopiv", [&](detail::Agree&) {
        check_desc(array_descA, handle);
        detail::require_backend<Backend>(handle, "pgetrf_nopiv");
        if (m > array_descA[DDLA_M_] || n > array_descA[DDLA_N_])
            throw std::invalid_argument("pgetrf_nopiv: m x n exceeds the descriptor of A");
        if (array_descA[DDLA_MB_] != array_descA[DDLA_NB_])
            throw std::invalid_argument("pgetrf_nopiv: A needs square blocks");
    });
}

// Component 2: [ the diagonal block's info word | the diagonal block (nb x nb), which the L panel (local rows x nb)
// reuses after the step's last trsm | the U panel (nb x local columns) ].
template <typename T>
void pgetrf_nopiv_workspace_size(const DdlaHandle_t& handle, int m, int n, const int* array_descA,
                                 std::size_t* backend_bytes)
{
    *backend_bytes = 0;
    if (m <= 0 || n <= 0) return;
    const int nb = array_descA[DDLA_MB_];
    const int m_loc = num_loc(m, nb, handle->myprow_, array_descA[DDLA_RSRC_], handle->nprows_);
    const int n_loc = num_loc(n, array_descA[DDLA_NB_], handle->mypcol_, array_descA[DDLA_CSRC_], handle->npcols_);
    *backend_bytes = detail::padded_bytes<int>(1) + detail::padded_bytes<T>(std::size_t(std::max(nb, m_loc)) * nb)
                   + detail::padded_bytes<T>(std::size_t(nb) * n_loc);
}

// Component 3.
template <DdlaBackend Backend, typename T>
void pgetrf_nopiv_run(const DdlaHandle_t& handle, int m, int n, T* A, const int* array_descA, int& info,
                      void* backend_work, [[maybe_unused]] std::size_t backend_bytes)
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

        int nb_real;
        int mm_row_start = 0;  // local row start of trailing submatrix
        int mm_col_start = 0;  // local col start of trailing submatrix

        // The scratch, in the order pgetrf_nopiv_workspace_size counts it.
        char* w = static_cast<char*>(backend_work);
        int* const info_buffer = detail::take<int>(w, 1);
        T* const temp_block = detail::take<T>(w, std::size_t(std::max(nb, m_loc)) * nb);
        T* const temp_L = temp_block;
        T* const temp_U = detail::take<T>(w, std::size_t(nb) * n_loc);
        assert(std::size_t(w - static_cast<char*>(backend_work)) <= backend_bytes);

        for (int n_s = 0; n_s < std::min(m, n); n_s += nb) {
            nb_real = std::min(nb, std::min(m, n) - n_s);

            int i_loc = indx_g2l_r(array_descA, handle, n_s);
            int j_loc = indx_g2l_c(array_descA, handle, n_s);

            int owner_row = indxg2p(n_s, nb, array_descA[DDLA_RSRC_], nprows);
            int owner_col = indxg2p(n_s, nb, array_descA[DDLA_CSRC_], npcols);

            // ================================================================
            // Step 1: Panel LU -- factor the nb*nb diagonal block in place
            // ================================================================
            // The diagonal block A11 lives entirely on process (owner_row, owner_col).
            // Call the local getrf_nopiv directly on the local backend buffer.
            int local_info = 0;
            if (myprow == owner_row && mypcol == owner_col) {
                lapack_impl::getrf_nopiv_run<Backend, T>(handle, nb_real, nb_real, A + i_loc + j_loc * lld, lld, info_buffer);
                DDLA_CHECK(ddlaMemcpy(handle, &local_info, info_buffer, sizeof(int)));
                DDLA_CHECK(ddlaSynchronize(handle));
            }

            MPI_CHECK(MPI_Bcast(&local_info, 1, MPI_INT, handle->rc_to_rank(owner_row, owner_col), handle->comm));
            if (local_info != 0) {
                // getrf_nopiv returns a 1-based index within the diagonal block;
                // shift it to the global position.
                info = local_info + n_s;
                break;
            }
            if (n_s + nb_real == n)
                break;

            // ================================================================
            // Step 2: Extract/broadcast the factored diagonal block (L1+U1)
            // ================================================================
            if (myprow == owner_row && mypcol == owner_col) {
                DDLA_CHECK(ddlaMemcpy2D(handle, temp_block, nb_real * sizeof(T), A + i_loc + j_loc * lld,
                                        lld * sizeof(T), nb_real * sizeof(T), nb_real));
            }
            if (myprow == owner_row) {
                commBcast<Backend>(handle, CommScope::Row, temp_block, (std::size_t)nb_real * nb_real, owner_col);
            }
            if (mypcol == owner_col) {
                commBcast<Backend>(handle, CommScope::Col, temp_block, (std::size_t)nb_real * nb_real, owner_row);
            }

            // ================================================================
            // Step 3: Compute U12 = L1^{-1} * B  (trsm LEFT/LOWER/UNIT)
            // ================================================================
            int right_panel_col_start = (j_loc >= 0) ? (j_loc + nb_real) : mm_col_start;
            if (myprow == owner_row) {
                if (n_loc > right_panel_col_start) {
                    T* right_panel = A + right_panel_col_start * lld + mm_row_start;
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
            // Step 4: Compute L21 = C * U1^{-1}  (trsm RIGHT/UPPER/NON-UNIT)
            // ================================================================
            int left_panel_row_start = (i_loc >= 0) ? (i_loc + nb_real) : mm_row_start;
            if (mypcol == owner_col) {
                if (m_loc > left_panel_row_start) {
                    T* left_panel = A + mm_col_start * lld + left_panel_row_start;
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
            // Step 5: Broadcast U panel down column, L panel across row
            // ================================================================
            if (n_loc > right_panel_col_start) {
                commBcast<Backend>(handle, CommScope::Col, temp_U, (std::size_t)nb_real * (n_loc - right_panel_col_start), owner_row);
            }
            if (m_loc > left_panel_row_start) {
                commBcast<Backend>(handle, CommScope::Row, temp_L, (std::size_t)(m_loc - left_panel_row_start) * nb_real, owner_col);
            }

            // ================================================================
            // Step 6: Schur-complement update  D <- D - L21 * U12
            // ================================================================
            int trailing_m = m_loc - left_panel_row_start;
            int trailing_n = n_loc - right_panel_col_start;

            if (trailing_m > 0 && trailing_n > 0) {
                blas_impl::gemm_run<Backend, T>(handle, 'N', 'N',
                    trailing_m, trailing_n, nb_real,
                    T(-1.0),
                    temp_L, trailing_m,
                    temp_U, nb_real,
                    T(1.0),
                    A + left_panel_row_start + right_panel_col_start * lld, lld);
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

} // namespace lu_impl

/**
 * @brief Distributed LU factorization without pivoting.
 *
 * Right-looking block algorithm: at each step the nb*nb diagonal block is
 * factored with the local getrf_nopiv (single-GPU, no pivoting).  The U and
 * L panels are then computed via trsm and the trailing submatrix is updated
 * via gemm, exactly as in pgetrf_bpiv but without any row swaps.
 *
 * For each panel step k (global column n_s):
 *   1) Factor diagonal block: A11 = L1 * U1          (getrf_nopiv)
 *   2) Extract/broadcast the factored diagonal block (L1+U1)
 *   3) Compute U12:  U12 = L1^{-1} * B             (trsm LEFT/LOWER/UNIT)
 *   4) Compute L21:  L21 = C * U1^{-1}             (trsm RIGHT/UPPER/NON-UNIT)
 *   5) Broadcast U panel down column, L panel across row
 *   6) Update trailing: D <- D - L21 * U12         (gemm)
 */
template <DdlaBackend Backend, typename T>
void pgetrf_nopiv(
    const DdlaHandle_t& handle, int m, int n,
    T* A, const int* array_descA,
    int& info  // host
)
{
    // 1. parameter check (collective).
    lu_impl::pgetrf_nopiv_check<Backend>(handle, m, n, array_descA);
    // No-op: an empty factorization.
    info = 0;
    if (m <= 0 || n <= 0) return;
    // 2. the scratch.
    std::size_t backend_bytes = 0;
    lu_impl::pgetrf_nopiv_workspace_size<T>(handle, m, n, array_descA, &backend_bytes);
    vector<char> backend_work(handle, backend_bytes);
    // 3. the factorization.
    lu_impl::pgetrf_nopiv_run<Backend, T>(handle, m, n, A, array_descA, info, backend_work.data(), backend_bytes);
}

// Explicit instantiations
#define INSTANTIATE_PGETRF_NOPIV(BACKEND, TYPE)                                   \
    template void pgetrf_nopiv<BACKEND, TYPE>(const DdlaHandle_t&, int m, int n, \
    TYPE* A, const int* array_descA, \
    int& info); \
    template void lu_impl::pgetrf_nopiv_run<BACKEND, TYPE>(const DdlaHandle_t&, int, int, TYPE*, const int*, int&, \
    void*, std::size_t)

#define INSTANTIATE_PGETRF_NOPIV_SIZE(TYPE) \
    template void lu_impl::pgetrf_nopiv_workspace_size<TYPE>(const DdlaHandle_t&, int, int, const int*, std::size_t*)

#define INSTANTIATE_PGETRF_NOPIV_CHECK(BACKEND) \
    template void lu_impl::pgetrf_nopiv_check<BACKEND>(const DdlaHandle_t&, int, int, const int*)


#if DDLA_HAS_CPU
INSTANTIATE_PGETRF_NOPIV_CHECK(DdlaBackend::CPU);
INSTANTIATE_PGETRF_NOPIV(DdlaBackend::CPU, float);
INSTANTIATE_PGETRF_NOPIV(DdlaBackend::CPU, double);
INSTANTIATE_PGETRF_NOPIV(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PGETRF_NOPIV(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_PGETRF_NOPIV_CHECK(DdlaBackend::GPU);
INSTANTIATE_PGETRF_NOPIV(DdlaBackend::GPU, float);
INSTANTIATE_PGETRF_NOPIV(DdlaBackend::GPU, double);
INSTANTIATE_PGETRF_NOPIV(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PGETRF_NOPIV(DdlaBackend::GPU, std::complex<double>);
#endif

INSTANTIATE_PGETRF_NOPIV_SIZE(float);
INSTANTIATE_PGETRF_NOPIV_SIZE(double);
INSTANTIATE_PGETRF_NOPIV_SIZE(std::complex<float>);
INSTANTIATE_PGETRF_NOPIV_SIZE(std::complex<double>);

#undef INSTANTIATE_PGETRF_NOPIV_SIZE
#undef INSTANTIATE_PGETRF_NOPIV_CHECK
#undef INSTANTIATE_PGETRF_NOPIV

} // namespace ddla
