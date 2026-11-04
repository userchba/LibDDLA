#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "ddla_desc.h"
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <vector>
#include "BLAS/blas_impl.h"
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "require_backend.h"
#include "pblas_workspace.h"
#include "comm_traits.h"
#include "check_agree.h"
#include "plan_util.h"
#include "ptrtrs_impl.h"
namespace ddla{

namespace ptrtrs_impl {

// Component 1.
template <DdlaBackend Backend>
void ptrtrs_check(const DdlaHandle_t& handle, char side, char uplo, char trans, char diag,
                  int m, int n, const int* array_descA, const int* array_descB)
{
    detail::agree_check(handle, "ptrtrs", [&](detail::Agree&) {
        check_desc(array_descB, handle);
        check_desc(array_descA, handle);
        detail::require_backend<Backend>(handle, "ptrtrs");

        if (array_descA[DDLA_MB_] != array_descA[DDLA_NB_])
            throw std::invalid_argument("ptrtrs: A needs square blocks");
        if (array_descA[DDLA_MB_] != array_descB[DDLA_MB_])
            throw std::invalid_argument("ptrtrs: A and B need the same row block size");
        if (side != 'L' && side != 'R')
            throw std::invalid_argument("ptrtrs: side must be 'L' or 'R'");
        if (uplo != 'L' && uplo != 'U')
            throw std::invalid_argument("ptrtrs: uplo must be 'L' or 'U'");
        if (diag != 'U' && diag != 'N')
            throw std::invalid_argument("ptrtrs: diag must be 'U' or 'N'");
        if (trans != 'N' && trans != 'T' && trans != 'C')
            throw std::invalid_argument("ptrtrs: trans must be 'N', 'T' or 'C'");
        // The descriptor may describe a matrix larger than the logical
        // sub-matrix (leading block anchored at global (0,0)).
        const int n_solve = (side == 'L') ? m : n;
        if (n_solve > array_descA[DDLA_M_] || n_solve > array_descA[DDLA_N_] ||
            m > array_descB[DDLA_M_] || n > array_descB[DDLA_N_])
            throw std::invalid_argument("ptrtrs: m x n exceeds the descriptors");
    });
}

namespace {
// Element counts of ptrtrs's own buffers, in carving order: the diagonal
// block (nb x nb) | the panel of A (a block column of the local rows or a
// block row of the local columns) | the block row/column of B.
void scratch_counts(const DdlaHandle_t& handle, char side, int m, int n,
                    const int* array_descA, const int* array_descB, std::size_t (&count)[3])
{
    const int nb = array_descA[DDLA_MB_], order = side == 'L' ? m : n;
    const int ar = num_loc(order, array_descA[DDLA_MB_], handle->myprow_, array_descA[DDLA_RSRC_], handle->nprows_);
    const int ac = num_loc(order, array_descA[DDLA_NB_], handle->mypcol_, array_descA[DDLA_CSRC_], handle->npcols_);
    const int br = num_loc(m, array_descB[DDLA_MB_], handle->myprow_, array_descB[DDLA_RSRC_], handle->nprows_);
    const int bc = num_loc(n, array_descB[DDLA_NB_], handle->mypcol_, array_descB[DDLA_CSRC_], handle->npcols_);
    count[0] = std::size_t(nb) * nb;
    count[1] = std::size_t(std::max(ar, ac)) * nb;
    count[2] = std::size_t(nb) * std::max(br, bc);
}
} // namespace

// Component 2.  Backend [ diagonal block | panel of A | block of B
// (scratch_counts) | the panel transport's send staging ], host [ the panel
// transport's schedule ].  The A panel of a transposed solve is a row panel
// (side 'L', nb x n_solve) or a column panel (side 'R', n_solve x nb)
// gathered through pgeadd and row/column broadcasts with reusable redistribution scratch.
template <typename T>
void ptrtrs_workspace_size(const DdlaHandle_t& handle, char side, char trans, int m, int n,
                           const int* array_descA, const int* array_descB,
                           std::size_t* backend_bytes, std::size_t* host_bytes)
{
    *backend_bytes = *host_bytes = 0;
    if (m == 0 || n == 0) return;
    const int nb = array_descA[DDLA_MB_], order = side == 'L' ? m : n;
    std::size_t count[3], transport_bytes = 0;
    scratch_counts(handle, side, m, n, array_descA, array_descB, count);
    const bool left = side=='L';
    const bool stored_column = left == (trans=='N');
    pblas::panel_workspace_size<T>(handle,left,trans,
        stored_column?order:nb,stored_column?nb:order,array_descA,array_descA,
        &transport_bytes,host_bytes);
    for (const std::size_t c : count) *backend_bytes += detail::padded_bytes<T>(c);
    *backend_bytes += transport_bytes;
}

// Component 3.
template <DdlaBackend Backend, typename T>
void ptrtrs_run(const DdlaHandle_t& handle,
                char side, char uplo, char trans, char diag, int m, int n,
                T* A, const int* array_descA, T* B, const int* array_descB,
                void* backend_work, [[maybe_unused]] std::size_t backend_bytes,
                void* host_work, std::size_t host_bytes)
{
    try {
        if(m==0 || n==0)return;
        int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
        ddlaGetGridDims(handle, nprows, npcols);
        ddlaGetGridCoords(handle, myprow, mypcol);

        DdlaHandle_t ddla_handle = handle;
        int nb = array_descA[DDLA_MB_];
        int lldA = array_descA[DDLA_LLD_];
        int lldB = array_descB[DDLA_LLD_];


        // Order of the triangular system: op(A) is n_solve x n_solve, matching the
        // rows of B for side='L' (B is m x n) and the columns of B for side='R'.
        const int n_solve = (side == 'L') ? m : n;
        // The descriptor may describe a matrix larger than the logical sub-matrix
        // (leading-block, anchored at global (0,0)); all local extents below are
        // derived from the logical dims via num_loc so nothing beyond the leading
        // block is ever touched.
        const int m_loc_A = num_loc(n_solve, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
        const int n_loc_A = num_loc(n_solve, array_descA[DDLA_NB_], mypcol, array_descA[DDLA_CSRC_], npcols);
        const int m_loc_B = num_loc(m, array_descB[DDLA_MB_], myprow, array_descB[DDLA_RSRC_], nprows);
        const int n_loc_B = num_loc(n, array_descB[DDLA_NB_], mypcol, array_descB[DDLA_CSRC_], npcols);

        // The scratch, in the order ptrtrs_workspace_size counts it; the rest is
        // the panel transport's.
        std::size_t count[3];
        scratch_counts(handle, side, m, n, array_descA, array_descB, count);
        char* w = static_cast<char*>(backend_work);
        T* const block_diag = detail::take<T>(w, count[0]);
        T* const block_A = detail::take<T>(w, count[1]);
        T* const block_B = detail::take<T>(w, count[2]);
        const std::size_t used = static_cast<std::size_t>(w - static_cast<char*>(backend_work));
        assert(used <= backend_bytes);
        void* const transport_work = w;
        const std::size_t transport_bytes = backend_bytes - used;

        int owner_row, owner_col;
        int mm_row_start, mm_col_start, mm_row_step, mm_col_step;
        int64_t A_offset, B_offset;

        if(side == 'L'){
            // Left solve: B := op(A)^{-1} * B. Block forward/backward substitution:
            // at each diagonal block, trsm on the owner row of B, broadcast the
            // solved block row, then gemm-update the remaining rows of B.
            const bool solve_backward = (uplo == 'U' && trans == 'N') || (uplo == 'L' && trans != 'N');
            int n_s_start,n_s_end,n_s_step;
            if(solve_backward){
                n_s_start = n_solve % nb == 0 ? n_solve - nb : n_solve - n_solve % nb;
                n_s_end = -nb;
                n_s_step = -nb;
            }else{
                n_s_start = 0;
                n_s_end = n_solve % nb == 0 ? n_solve : n_solve - n_solve % nb + nb;
                n_s_step = nb;
            }
            for(int n_s = n_s_start; n_s != n_s_end; n_s += n_s_step){
                int nb_real = std::min(nb, n_solve - n_s);

                mm_row_start = num_loc(n_s, nb, myprow, array_descA[DDLA_RSRC_], nprows);
                mm_col_start = num_loc(n_s, nb, mypcol, array_descA[DDLA_CSRC_], npcols);

                owner_row = indxg2p(n_s, nb, array_descA[DDLA_RSRC_], nprows);
                owner_col = indxg2p(n_s, nb, array_descA[DDLA_CSRC_], npcols);

                if(myprow == owner_row)
                    mm_row_step = nb_real;
                else
                    mm_row_step = 0;
                if(mypcol == owner_col)
                    mm_col_step = nb_real;
                else
                    mm_col_step = 0;

                if(myprow == owner_row && mypcol == owner_col){
                    DDLA_CHECK(ddlaMemcpy2D(ddla_handle, block_diag, nb_real * sizeof(T),
                                            A + mm_row_start + mm_col_start * lldA, lldA * sizeof(T),
                                            nb_real * sizeof(T), nb_real));
                }
                DDLA_CHECK(ddlaSynchronize(ddla_handle));
                // 广播当前块行
                if(myprow == owner_row){
                    commBcast<Backend>(ddla_handle, CommScope::Row, block_diag, (std::size_t)nb_real * nb_real, owner_col);
                    blas_impl::trsm_run<Backend, T>(ddla_handle, 'L', uplo, trans, diag,
                        nb_real, n_loc_B, (T)1.0,
                        block_diag, nb_real,
                        B + mm_row_start, lldB
                    );
                }
                pblas::broadcast_panel<Backend, T>(handle, false, nb_real, n, B, n_s, 0, array_descB, block_B);
                int length_block_A;
                int g_m, g_n;
                int g_ia, g_ja;
                if(trans != 'N'){
                    g_m = nb_real;
                    g_ia = n_s;
                    if(uplo == 'L'){
                        A_offset = mm_row_start;
                        length_block_A = mm_col_start;
                        g_n = n_s;
                        g_ja = 0;
                    }else{
                        // U^H solve: gather the row panel to the right of the diagonal.
                        A_offset = mm_row_start + (mm_col_start + mm_col_step) * array_descA[DDLA_LLD_];
                        length_block_A = n_loc_A - mm_col_start - mm_col_step;
                        g_n = n_solve - n_s - nb_real;
                        g_ja = n_s + nb_real;
                    }
                }else{
                    g_ja = n_s;
                    g_n = nb_real;
                    if(uplo == 'L'){
                        length_block_A = m_loc_A - mm_row_start - mm_row_step;
                        A_offset = mm_row_start + mm_row_step + mm_col_start * array_descA[DDLA_LLD_];
                        g_m = n_solve - n_s - nb_real;
                        g_ia = n_s + nb_real;
                    }else{
                        // U solve: gather the column panel above the diagonal.
                        length_block_A = mm_row_start;
                        A_offset = mm_col_start * array_descA[DDLA_LLD_];
                        g_m = n_s;
                        g_ia = 0;
                    }
                }
                pblas::panel_run<Backend, T>(handle, true, trans, g_m, g_n, A, g_ia, g_ja, array_descA,
                    array_descA, block_A, transport_work, transport_bytes, host_work, host_bytes);
                if(solve_backward){
                    length_block_A = mm_row_start;
                    B_offset = 0;
                }else{
                    length_block_A = m_loc_A - mm_row_start - mm_row_step;
                    B_offset = mm_row_start + mm_row_step;
                }
                DDLA_CHECK(ddlaSynchronize(ddla_handle));
                if(length_block_A > 0){
                    blas_impl::gemm_run<Backend, T>(ddla_handle, 'N', 'N',
                        length_block_A, n_loc_B, nb_real,
                        (T)-1.0,
                        block_A, length_block_A,
                        block_B, nb_real,
                        (T)1.0,
                        B + B_offset, lldB);
                }
                DDLA_CHECK(ddlaSynchronize(ddla_handle));
            }
        }else{
            // Right solve: X := B * op(A)^{-1}, i.e. X * op(A) = B. Column-mirror of
            // the left solve: trsm on the owner column of B, broadcast the solved
            // block column, then gemm-update the remaining columns of B with the
            // row panel of A on the far side of the diagonal block.
            const bool solve_backward = (uplo == 'L' && trans == 'N') || (uplo == 'U' && trans != 'N');
            const bool far_right = !solve_backward;
            int n_s_start,n_s_end,n_s_step;
            if(solve_backward){
                n_s_start = n_solve % nb == 0 ? n_solve - nb : n_solve - n_solve % nb;
                n_s_end = -nb;
                n_s_step = -nb;
            }else{
                n_s_start = 0;
                n_s_end = n_solve % nb == 0 ? n_solve : n_solve - n_solve % nb + nb;
                n_s_step = nb;
            }
            for(int n_s = n_s_start; n_s != n_s_end; n_s += n_s_step){
                int nb_real = std::min(nb, n_solve - n_s);

                mm_row_start = num_loc(n_s, nb, myprow, array_descA[DDLA_RSRC_], nprows);
                mm_col_start = num_loc(n_s, nb, mypcol, array_descA[DDLA_CSRC_], npcols);

                owner_row = indxg2p(n_s, nb, array_descA[DDLA_RSRC_], nprows);
                owner_col = indxg2p(n_s, nb, array_descA[DDLA_CSRC_], npcols);

                if(myprow == owner_row)
                    mm_row_step = nb_real;
                else
                    mm_row_step = 0;
                if(mypcol == owner_col)
                    mm_col_step = nb_real;
                else
                    mm_col_step = 0;

                if(myprow == owner_row && mypcol == owner_col){
                    DDLA_CHECK(ddlaMemcpy2D(ddla_handle, block_diag, nb_real * sizeof(T),
                                            A + mm_row_start + mm_col_start * lldA, lldA * sizeof(T),
                                            nb_real * sizeof(T), nb_real));
                }
                DDLA_CHECK(ddlaSynchronize(ddla_handle));
                // broadcast the diagonal block within the column, then solve the
                // block column of B on the owner column's processes
                if(mypcol == owner_col){
                    commBcast<Backend>(ddla_handle, CommScope::Col, block_diag, (std::size_t)nb_real * nb_real, owner_row);
                    blas_impl::trsm_run<Backend, T>(ddla_handle, 'R', uplo, trans, diag,
                        m_loc_B, nb_real, (T)1.0,
                        block_diag, nb_real,
                        B + mm_col_start * lldB, lldB
                    );
                }
                // gather the solved block column of B (local part: m_loc x nb_real)
                pblas::broadcast_panel<Backend, T>(handle, true, m, nb_real, B, 0, n_s, array_descB, block_B);
                // gather the far-side panel of A: for trans='N' the row panel of A
                // on the far side of the diagonal block; for trans!='N' the column
                // panel of A on the far side, transposed during the gather.
                int length_block_A;
                int g_m, g_n;
                int g_ia, g_ja;
                if(trans == 'N'){
                    g_m = nb_real;
                    g_ia = n_s;
                    if(far_right){
                        g_n = n_solve - n_s - nb_real;
                        g_ja = n_s + nb_real;
                        A_offset = mm_row_start + (mm_col_start + mm_col_step) * array_descA[DDLA_LLD_];
                        length_block_A = n_loc_A - mm_col_start - mm_col_step;
                        B_offset = (mm_col_start + mm_col_step) * lldB;
                    }else{
                        g_n = n_s;
                        g_ja = 0;
                        A_offset = mm_row_start;
                        length_block_A = mm_col_start;
                        B_offset = 0;
                    }
                }else{
                    g_n = nb_real;
                    g_ja = n_s;
                    if(far_right){
                        g_m = n_solve - n_s - nb_real;
                        g_ia = n_s + nb_real;
                        length_block_A = n_loc_A - mm_col_start - mm_col_step;
                        B_offset = (mm_col_start + mm_col_step) * lldB;
                    }else{
                        g_m = n_s;
                        g_ia = 0;
                        length_block_A = mm_col_start;
                        B_offset = 0;
                    }
                }
                pblas::panel_run<Backend, T>(handle, false, trans, g_m, g_n, A, g_ia, g_ja, array_descA,
                    array_descA, block_A, transport_work, transport_bytes, host_work, host_bytes);
                DDLA_CHECK(ddlaSynchronize(ddla_handle));
                if(length_block_A > 0 && m_loc_B > 0){
                    blas_impl::gemm_run<Backend, T>(ddla_handle, 'N', 'N',
                        m_loc_B, length_block_A, nb_real,
                        (T)-1.0,
                        block_B, m_loc_B,
                        block_A, nb_real,
                        (T)1.0,
                        B + B_offset, lldB);
                }
                DDLA_CHECK(ddlaSynchronize(ddla_handle));
            }
        }
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(handle) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(handle); } catch (...) {}
        throw;
    }
}

} // namespace ptrtrs_impl

template <DdlaBackend Backend, typename T>
void ptrtrs(
    const DdlaHandle_t& handle, char side, char uplo, char trans, char diag,
    int m, int n,
    T* A, const int* array_descA,
    T* B, const int* array_descB
)
{
    ptrtrs_impl::ptrtrs_check<Backend>(handle, side, uplo, trans, diag, m, n, array_descA, array_descB);
    std::size_t backend_bytes = 0, host_bytes = 0;
    ptrtrs_impl::ptrtrs_workspace_size<T>(handle, side, trans, m, n, array_descA, array_descB,
                                          &backend_bytes, &host_bytes);
    vector<char> backend_work(handle, backend_bytes);
    std::vector<char> host_work(host_bytes);
    ptrtrs_impl::ptrtrs_run<Backend, T>(handle, side, uplo, trans, diag, m, n, A, array_descA, B, array_descB,
                                        backend_work.data(), backend_bytes, host_work.data(), host_bytes);
}


#define INSTANTIATE_PTRTRS(BACKEND, TYPE)                                   \
    template void ptrtrs<BACKEND, TYPE>(const DdlaHandle_t&, char side, char uplo, char trans, char diag, \
    int m, int n, \
    TYPE* A, const int* array_descA, \
    TYPE* B, const int* array_descB); \
    template void ptrtrs_impl::ptrtrs_run<BACKEND, TYPE>(const DdlaHandle_t&, \
    char side, char uplo, char trans, char diag, int m, int n, \
    TYPE* A, const int* array_descA, TYPE* B, const int* array_descB, \
    void*, std::size_t, void*, std::size_t)

#define INSTANTIATE_PTRTRS_CHECK(BACKEND) \
    template void ptrtrs_impl::ptrtrs_check<BACKEND>(const DdlaHandle_t&, char, char, char, char, \
    int, int, const int*, const int*)

#define INSTANTIATE_PTRTRS_SIZE(TYPE) \
    template void ptrtrs_impl::ptrtrs_workspace_size<TYPE>(const DdlaHandle_t&, char, char, \
    int, int, const int*, const int*, std::size_t*, std::size_t*)

INSTANTIATE_PTRTRS_CHECK(DdlaBackend::CPU);
INSTANTIATE_PTRTRS(DdlaBackend::CPU, float);
INSTANTIATE_PTRTRS(DdlaBackend::CPU, double);
INSTANTIATE_PTRTRS(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PTRTRS(DdlaBackend::CPU, std::complex<double>);

#if DDLA_HAS_GPU
INSTANTIATE_PTRTRS_CHECK(DdlaBackend::GPU);
INSTANTIATE_PTRTRS(DdlaBackend::GPU, float);
INSTANTIATE_PTRTRS(DdlaBackend::GPU, double);
INSTANTIATE_PTRTRS(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PTRTRS(DdlaBackend::GPU, std::complex<double>);
#endif

INSTANTIATE_PTRTRS_SIZE(float);
INSTANTIATE_PTRTRS_SIZE(double);
INSTANTIATE_PTRTRS_SIZE(std::complex<float>);
INSTANTIATE_PTRTRS_SIZE(std::complex<double>);

#undef INSTANTIATE_PTRTRS_SIZE
#undef INSTANTIATE_PTRTRS_CHECK
#undef INSTANTIATE_PTRTRS



}
