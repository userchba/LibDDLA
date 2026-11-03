#include "BLAS/blas_impl.h"
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "ddla_connector.h"
#include "ddla_desc.h"
#include <cassert>
#include <cstddef>
#include <vector>
#include <complex>
#include "ddla_stream.h"
#include "require_backend.h"
#include "check_agree.h"
#include "plan_util.h"
#include "lu_panel.h"
#include "pgetrs_impl.h"
#include "PBLAS/ptrtrs_impl.h"

namespace ddla {

namespace lu_impl {

// Component 1.
template <DdlaBackend Backend>
void pgetrs_bpiv_check(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                       const int* array_descA, const int* array_descB)
{
    detail::agree_check(handle, "pgetrs_bpiv", [&](detail::Agree&) {
        check_desc(array_descB, handle);
        check_desc(array_descA, handle);
        detail::require_backend<Backend>(handle, "pgetrs_bpiv");
        check_solve("pgetrs_bpiv", side, trans, n, nrhs, array_descA, array_descB);
    });
}

// Component 2: host bytes of [ the host copy of the local block pivots (local rows of A's leading n x n block) | the
// pivots of one block (nb) | the triangular solves' ]; backend bytes of the triangular solves'.  One size serves
// both solves (both use `trans`).
template <typename T>
void pgetrs_bpiv_workspace_size(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                                const int* array_descA, const int* array_descB,
                                std::size_t* backend_bytes, std::size_t* host_bytes)
{
    *backend_bytes = *host_bytes = 0;
    if (n == 0 || nrhs == 0) return;
    const int m_loc_A = num_loc(n, array_descA[DDLA_MB_], handle->myprow_, array_descA[DDLA_RSRC_], handle->nprows_);
    std::size_t solve_host = 0;
    ptrtrs_impl::ptrtrs_workspace_size<T>(handle, side, trans, side == 'L' ? n : nrhs, side == 'L' ? nrhs : n,
                                          array_descA, array_descB, backend_bytes, &solve_host);
    *host_bytes = detail::padded_bytes<int>(std::size_t(m_loc_A)) + detail::padded_bytes<int>(array_descA[DDLA_MB_])
                + solve_host;
}

// Component 3.
template<DdlaBackend Backend,class T>
void pgetrs_bpiv_run(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                     T* A, const int* array_descA,
                     int* pivots, // host for CPU, device for GPU
                     T* B, const int* array_descB,
                     void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes)
{
    try {
        if (n == 0 || nrhs == 0) return;

        int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
        ddlaGetGridDims(handle, nprows, npcols);
        ddlaGetGridCoords(handle, myprow, mypcol);

        const int nb = array_descA[DDLA_MB_];
        const int b_rows = (side == 'L') ? n : nrhs;
        const int b_cols = (side == 'L') ? nrhs : n;
        // Logical local extents of the leading-block sub-matrices.
        const int m_loc_A = num_loc(n, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
        const int n_loc_B = num_loc(b_cols, array_descB[DDLA_NB_], mypcol, array_descB[DDLA_CSRC_], npcols);
        const int m_loc_B = num_loc(b_rows, array_descB[DDLA_MB_], myprow, array_descB[DDLA_RSRC_], nprows);
        // The scratch, in the order pgetrs_bpiv_workspace_size counts it; the triangular solves get the rest.
        char* hw = static_cast<char*>(host_work);
        int* const h_ipiv = detail::take<int>(hw, std::size_t(m_loc_A));
        int* const piv = detail::take<int>(hw, std::size_t(nb));
        const std::size_t host_used = std::size_t(hw - static_cast<char*>(host_work));
        assert(host_used <= host_bytes);

        // Copy the local block pivots to host.  Valid entries live only on the
        // (owner_row, owner_col) process of each block (pgetrf_bpiv broadcasts
        // them along the process row for every block except the last, which
        // breaks before the broadcast).  apply_pivots below re-broadcasts each
        // block's pivots over the full grid from that single owner, so every
        // process gets correct values regardless of block.
        DDLA_CHECK(ddlaMemcpy(handle, h_ipiv, pivots, std::size_t(m_loc_A) * sizeof(int)));
        DDLA_CHECK(ddlaSynchronize(handle));

        // Apply the block-partial pivot permutation to B: rows of B for side='L',
        // columns of B for side='R'.  Each block's swaps are local to the owning
        // process row (columns) so no data movement is required.  `forward`
        // applies each block's swaps in the same order laswp applied them,
        // `backward` in reverse order.  Note the direction composes differently
        // for rows and columns: forward rows compute P^T*X (left multiply),
        // backward rows compute P*X, forward columns compute X*P (right
        // multiply), backward columns compute X*P^T, where A = P*L*U.
        //
        // Valid block pivots live only on the (owner_row, owner_col) process:
        // pgetrf_bpiv broadcasts them along the process row for every block
        // *except* the last (the loop breaks before the broadcast), and the
        // processes of the owning process column need them for a side='R' solve.
        // A single broadcast over the full grid from rc_to_rank(owner_row,
        // owner_col) gives every process the block pivots unconditionally.
        auto apply_pivots = [&](bool columns, bool forward){
            const int lldB = array_descB[DDLA_LLD_];
            for(int n_s = 0; n_s < n; n_s += nb){
                const int nb_real = std::min(nb, n - n_s);
                const int owner_row = indxg2p(n_s, nb, array_descA[DDLA_RSRC_], nprows);
                const int owner_col = indxg2p(n_s, nb, array_descA[DDLA_CSRC_], npcols);
                if(myprow == owner_row && mypcol == owner_col){
                    const int mm_row_start =
                        num_loc(n_s, nb, myprow, array_descA[DDLA_RSRC_], nprows);
                    for(int i = 0; i < nb_real; ++i)
                        piv[i] = h_ipiv[mm_row_start + i];
                }
                MPI_CHECK(MPI_Bcast(piv, nb_real, MPI_INT,
                                    handle->rc_to_rank(owner_row, owner_col), handle->comm));
                const int begin = forward ? 1 : nb_real;
                const int end   = forward ? nb_real + 1 : 0;
                const int step  = forward ? 1 : -1;
                for(int i = begin; i != end; i += step){
                    const int t = piv[i - 1] - 1; // 0-based target inside the block
                    if(t == i - 1)
                        continue;
                    if(columns){
                        if(mypcol == owner_col){
                            const int j1 = indx_g2l_c(array_descB, handle, n_s + i - 1);
                            const int j2 = indx_g2l_c(array_descB, handle, n_s + t);
                            blas_impl::swap_run<Backend, T>(handle,
                                                      m_loc_B, B + j1 * lldB, 1,
                                                      B + j2 * lldB, 1);
                        }
                    }else{
                        if(myprow == owner_row){
                            const int i1 = indx_g2l_r(array_descB, handle, n_s + i - 1);
                            const int i2 = indx_g2l_r(array_descB, handle, n_s + t);
                            blas_impl::swap_run<Backend, T>(handle,
                                                      n_loc_B, B + i1, lldB,
                                                      B + i2, lldB);
                        }
                    }
                }
            }
        };

        if(side == 'L'){
            if(trans == 'N'){
                // A = Q^T * L * U => X = U^-1 * L^-1 * (Q * B)
                apply_pivots(/*columns=*/false, /*forward=*/true);
                ptrtrs_impl::ptrtrs_run<Backend, T>(handle, 'L', 'L', 'N', 'U', b_rows, b_cols, A, array_descA, B, array_descB,
                                                    backend_work, backend_bytes, hw, host_bytes - host_used);
                ptrtrs_impl::ptrtrs_run<Backend, T>(handle, 'L', 'U', 'N', 'N', b_rows, b_cols, A, array_descA, B, array_descB,
                                                    backend_work, backend_bytes, hw, host_bytes - host_used);
            }else{
                // op(A)^T = U^T * L^T * Q => X = Q^T * L^-T * U^-T * B
                ptrtrs_impl::ptrtrs_run<Backend, T>(handle, 'L', 'U', trans, 'N', b_rows, b_cols, A, array_descA, B, array_descB,
                                                    backend_work, backend_bytes, hw, host_bytes - host_used);
                ptrtrs_impl::ptrtrs_run<Backend, T>(handle, 'L', 'L', trans, 'U', b_rows, b_cols, A, array_descA, B, array_descB,
                                                    backend_work, backend_bytes, hw, host_bytes - host_used);
                apply_pivots(/*columns=*/false, /*forward=*/false);
            }
        }else{
            if(trans == 'N'){
                // X * P * L * U = B => X = B * U^-1 * L^-1 * P^T
                ptrtrs_impl::ptrtrs_run<Backend, T>(handle, 'R', 'U', 'N', 'N', b_rows, b_cols, A, array_descA, B, array_descB,
                                                    backend_work, backend_bytes, hw, host_bytes - host_used);
                ptrtrs_impl::ptrtrs_run<Backend, T>(handle, 'R', 'L', 'N', 'U', b_rows, b_cols, A, array_descA, B, array_descB,
                                                    backend_work, backend_bytes, hw, host_bytes - host_used);
                // Columns compose opposite to rows: column swaps applied in the
                // laswp order right-multiply to Z*P, not Z*P^T; apply backward.
                apply_pivots(/*columns=*/true, /*forward=*/false);
            }else{
                // X * U^T * L^T * P^T = B => X = B * P * L^-T * U^-T
                // Column swaps in the laswp order give B*P; apply forward.
                apply_pivots(/*columns=*/true, /*forward=*/true);
                ptrtrs_impl::ptrtrs_run<Backend, T>(handle, 'R', 'L', trans, 'U', b_rows, b_cols, A, array_descA, B, array_descB,
                                                    backend_work, backend_bytes, hw, host_bytes - host_used);
                ptrtrs_impl::ptrtrs_run<Backend, T>(handle, 'R', 'U', trans, 'N', b_rows, b_cols, A, array_descA, B, array_descB,
                                                    backend_work, backend_bytes, hw, host_bytes - host_used);
            }
        }
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(handle) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(handle); } catch (...) {}
        throw;
    }
}

} // namespace lu_impl

/**
 * @brief Distributed solve using the block LU factors from pgetrf_bpiv.
 *
 * pgetrf_bpiv factors each nb x nb diagonal block with a local getrf and
 * applies each block's row permutation (block-partial pivoting) to the right
 * panel with laswp in forward order.  The stored pivots entries are 1-based
 * offsets *within* each diagonal block, kept in backend memory on the owning
 * process row only.  The block permutations act on disjoint row sets, so they
 * commute and each block's swaps are local to one process row (side='L') or
 * process column (side='R').
 *
 * For a side='L', trans='N' solve the permutation Q (the composite of the
 * block swaps in the same forward order laswp applied them) must be applied
 * to B before the two triangular solves; the remaining side/trans
 * combinations mirror the trsm order and apply Q^T (backward block-local
 * order) on the opposite side.
 *
 * @tparam T   Scalar type.
 * @param side    'L' -- solve op(A)*X = B (B is n x nrhs);
 *                'R' -- solve X*op(A) = B (B is nrhs x n).
 * @param trans   'N', 'T' or 'C' -- operation applied to A.
 * @param n       Order of matrix A.
 * @param nrhs    Number of right-hand sides.
 * @param A     Pointer in the selected backend to LU factors (from pgetrf_bpiv).
 * @param array_descA  ScaLAPACK int[9] descriptor for A.
 * @param pivots  Pivot array in the selected backend from pgetrf_bpiv (block-local, 1-based).
 * @param B     Pointer in the selected backend to RHS / solution B (input/output).
 * @param array_descB  ScaLAPACK int[9] descriptor for B.
 */
template <DdlaBackend Backend, typename T>
void pgetrs_bpiv(
    const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
    T* A, const int* array_descA,
    int* pivots, // host for CPU, device for GPU
    T* B, const int* array_descB
)
{
    // 1. parameter check (collective).
    lu_impl::pgetrs_bpiv_check<Backend>(handle, side, trans, n, nrhs, array_descA, array_descB);
    // No-op: nothing to solve for (B is left as it is).
    if (n == 0 || nrhs == 0) return;
    // 2. the host pivots and the scratch of the triangular solves.
    std::size_t backend_bytes = 0, host_bytes = 0;
    lu_impl::pgetrs_bpiv_workspace_size<T>(handle, side, trans, n, nrhs, array_descA, array_descB,
                                           &backend_bytes, &host_bytes);
    vector<char> backend_work(handle, backend_bytes);
    std::vector<char> host_work(host_bytes);
    // 3. the solve.
    lu_impl::pgetrs_bpiv_run<Backend, T>(handle, side, trans, n, nrhs, A, array_descA, pivots, B, array_descB,
                                         backend_work.data(), backend_bytes, host_work.data(), host_bytes);
}

#define INSTANTIATE_PGETRS_BPIV(BACKEND, TYPE)                                   \
    template void pgetrs_bpiv<BACKEND, TYPE>(const DdlaHandle_t&, char side, char trans, int n, int nrhs, \
    TYPE* A, const int* array_descA, \
    int* pivots, \
    TYPE* B, const int* array_descB); \
    template void lu_impl::pgetrs_bpiv_run<BACKEND, TYPE>(const DdlaHandle_t&, char, char, int, int, TYPE*, \
    const int*, int*, TYPE*, const int*, void*, std::size_t, void*, std::size_t)

#define INSTANTIATE_PGETRS_BPIV_SIZE(TYPE) \
    template void lu_impl::pgetrs_bpiv_workspace_size<TYPE>(const DdlaHandle_t&, char, char, int, int, const int*, \
    const int*, std::size_t*, std::size_t*)

#define INSTANTIATE_PGETRS_BPIV_CHECK(BACKEND) \
    template void lu_impl::pgetrs_bpiv_check<BACKEND>(const DdlaHandle_t&, char, char, int, int, const int*, \
    const int*)


INSTANTIATE_PGETRS_BPIV_CHECK(DdlaBackend::CPU);
INSTANTIATE_PGETRS_BPIV(DdlaBackend::CPU, float);
INSTANTIATE_PGETRS_BPIV(DdlaBackend::CPU, double);
INSTANTIATE_PGETRS_BPIV(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PGETRS_BPIV(DdlaBackend::CPU, std::complex<double>);

#if DDLA_HAS_GPU
INSTANTIATE_PGETRS_BPIV_CHECK(DdlaBackend::GPU);
INSTANTIATE_PGETRS_BPIV(DdlaBackend::GPU, float);
INSTANTIATE_PGETRS_BPIV(DdlaBackend::GPU, double);
INSTANTIATE_PGETRS_BPIV(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PGETRS_BPIV(DdlaBackend::GPU, std::complex<double>);
#endif

INSTANTIATE_PGETRS_BPIV_SIZE(float);
INSTANTIATE_PGETRS_BPIV_SIZE(double);
INSTANTIATE_PGETRS_BPIV_SIZE(std::complex<float>);
INSTANTIATE_PGETRS_BPIV_SIZE(std::complex<double>);

#undef INSTANTIATE_PGETRS_BPIV_SIZE
#undef INSTANTIATE_PGETRS_BPIV_CHECK
#undef INSTANTIATE_PGETRS_BPIV

} // namespace ddla
