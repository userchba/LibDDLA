#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <cassert>
#include <vector>
#include "swap.h"
#include "ddla_stream_impl.h"
#include "require_gpu.h"

namespace ddla {

/**
 * @brief Distributed solve using the block LU factors from pgetrf_bpiv.
 *
 * pgetrf_bpiv factors each nb x nb diagonal block with a local getrf and
 * applies each block's row permutation (block-partial pivoting) to the right
 * panel with laswp in forward order.  The stored d_ipiv entries are 1-based
 * offsets *within* each diagonal block, kept in device memory on the owning
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
 * @param d_A     Device pointer to LU factors (from pgetrf_bpiv).
 * @param array_descA  ScaLAPACK int[9] descriptor for A.
 * @param d_ipiv  Device pivot array from pgetrf_bpiv (block-local, 1-based).
 * @param d_B     Device pointer to RHS / solution B (input/output).
 * @param array_descB  ScaLAPACK int[9] descriptor for B.
 */
template<typename T>
void pgetrs_bpiv(
    const DdlaHandle_t& handle, const char& side, const char& trans, const int& n, const int& nrhs,
    T* d_A, const int* array_descA,
    int* d_ipiv, // device
    T* d_B, const int* array_descB
)
{
    check_desc(array_descB, handle);
    check_desc(array_descA, handle);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);


    DdlaHandle_t ddla_handle = handle;
    detail::require_gpu_backend(ddla_handle, "pgetrs_bpiv");
    assert(side == 'L' || side == 'R');
    assert(trans == 'N' || trans == 'T' || trans == 'C');
    assert(array_descA[DDLA_MB_] == array_descA[DDLA_NB_]);
    if(side == 'L'){
        assert(n <= array_descA[DDLA_M_] && n <= array_descA[DDLA_N_]);
        assert(n <= array_descB[DDLA_M_]);
    }else{
        assert(n <= array_descA[DDLA_M_] && n <= array_descA[DDLA_N_]);
        assert(n <= array_descB[DDLA_N_]);
    }
    const int nb = array_descA[DDLA_MB_];
    const int b_rows = (side == 'L') ? n : nrhs;
    const int b_cols = (side == 'L') ? nrhs : n;
    // Logical local extents of the leading-block sub-matrices.
    const int m_loc_A = num_loc(n, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
    const int n_loc_B = num_loc(b_cols, array_descB[DDLA_NB_], mypcol, array_descB[DDLA_CSRC_], npcols);
    const int m_loc_B = num_loc(b_rows, array_descB[DDLA_MB_], myprow, array_descB[DDLA_RSRC_], nprows);

    // Copy the local block pivots to host.  Valid entries live only on the
    // (owner_row, owner_col) process of each block (pgetrf_bpiv broadcasts
    // them along the process row for every block except the last, which
    // breaks before the broadcast).  apply_pivots below re-broadcasts each
    // block's pivots over the full grid from that single owner, so every
    // process gets correct values regardless of block.
    std::vector<int> h_ipiv(m_loc_A);
    RUNTIME_CHECK(runtimeMemcpyAsync(h_ipiv.data(), d_ipiv,
                                     h_ipiv.size() * sizeof(int),
                                     runtimeMemcpyDeviceToHost, ddla_handle->stream));
    RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));

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
        std::vector<int> piv(nb);
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
            MPI_CHECK(MPI_Bcast(piv.data(), nb_real, MPI_INT,
                                ddla_handle->rc_to_rank(owner_row, owner_col), ddla_handle->comm));
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
                        BLAS_CHECK(deblasSwap(ddla_handle->blasH,
                                              m_loc_B, d_B + j1 * lldB, 1,
                                              d_B + j2 * lldB, 1));
                    }
                }else{
                    if(myprow == owner_row){
                        const int i1 = indx_g2l_r(array_descB, handle, n_s + i - 1);
                        const int i2 = indx_g2l_r(array_descB, handle, n_s + t);
                        BLAS_CHECK(deblasSwap(ddla_handle->blasH,
                                              n_loc_B, d_B + i1, lldB,
                                              d_B + i2, lldB));
                    }
                }
            }
        }
    };

    if(side == 'L'){
        if(trans == 'N'){
            // A = Q^T * L * U => X = U^-1 * L^-1 * (Q * B)
            apply_pivots(/*columns=*/false, /*forward=*/true);
            ptrtrs(handle, 'L', 'L', 'N', 'U', b_rows, b_cols, d_A, array_descA, d_B, array_descB);
            ptrtrs(handle, 'L', 'U', 'N', 'N', b_rows, b_cols, d_A, array_descA, d_B, array_descB);
        }else{
            // op(A)^T = U^T * L^T * Q => X = Q^T * L^-T * U^-T * B
            ptrtrs(handle, 'L', 'U', trans, 'N', b_rows, b_cols, d_A, array_descA, d_B, array_descB);
            ptrtrs(handle, 'L', 'L', trans, 'U', b_rows, b_cols, d_A, array_descA, d_B, array_descB);
            apply_pivots(/*columns=*/false, /*forward=*/false);
        }
    }else{
        if(trans == 'N'){
            // X * P * L * U = B => X = B * U^-1 * L^-1 * P^T
            ptrtrs(handle, 'R', 'U', 'N', 'N', b_rows, b_cols, d_A, array_descA, d_B, array_descB);
            ptrtrs(handle, 'R', 'L', 'N', 'U', b_rows, b_cols, d_A, array_descA, d_B, array_descB);
            // Columns compose opposite to rows: column swaps applied in the
            // laswp order right-multiply to Z*P, not Z*P^T; apply backward.
            apply_pivots(/*columns=*/true, /*forward=*/false);
        }else{
            // X * U^T * L^T * P^T = B => X = B * P * L^-T * U^-T
            // Column swaps in the laswp order give B*P; apply forward.
            apply_pivots(/*columns=*/true, /*forward=*/true);
            ptrtrs(handle, 'R', 'L', trans, 'U', b_rows, b_cols, d_A, array_descA, d_B, array_descB);
            ptrtrs(handle, 'R', 'U', trans, 'N', b_rows, b_cols, d_A, array_descA, d_B, array_descB);
        }
    }
}

template void pgetrs_bpiv<float>(
    const DdlaHandle_t&, const char& side, const char& trans, const int& n, const int& nrhs,
    float* d_A, const int* array_descA,
    int* d_ipiv,
    float* d_B, const int* array_descB
);
template void pgetrs_bpiv<double>(
    const DdlaHandle_t&, const char& side, const char& trans, const int& n, const int& nrhs,
    double* d_A, const int* array_descA,
    int* d_ipiv,
    double* d_B, const int* array_descB
);
template void pgetrs_bpiv<std::complex<float>>(
    const DdlaHandle_t&, const char& side, const char& trans, const int& n, const int& nrhs,
    std::complex<float>* d_A, const int* array_descA,
    int* d_ipiv,
    std::complex<float>* d_B, const int* array_descB
);
template void pgetrs_bpiv<std::complex<double>>(
    const DdlaHandle_t&, const char& side, const char& trans, const int& n, const int& nrhs,
    std::complex<double>* d_A, const int* array_descA,
    int* d_ipiv,
    std::complex<double>* d_B, const int* array_descB
);

} // namespace ddla
