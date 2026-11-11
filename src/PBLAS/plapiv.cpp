#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "ddla_desc.h"
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include "BLAS/blas_impl.h"
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "require_backend.h"
#include "comm_traits.h"
#include "check_agree.h"
#include "plan_util.h"
#include "plapiv_impl.h"
#include "instantiate.h"

namespace ddla{

namespace plapiv_impl {

// Component 1.
template <DdlaBackend Backend>
void plapiv_check(const DdlaHandle_t& handle, char direc, char rowcol, char pivroc,
                  int m, int n, const int* array_descA, const int* array_descIP)
{
    detail::agree_check(handle, "plapiv", [&](detail::Agree&) {
        check_desc(array_descIP, handle);
        check_desc(array_descA, handle);
        detail::require_backend<Backend>(handle, "plapiv");

        if (direc != 'F' && direc != 'B')
            throw std::invalid_argument("plapiv: direc must be 'F' or 'B'");
        if (rowcol != 'R' && rowcol != 'C')
            throw std::invalid_argument("plapiv: rowcol must be 'R' or 'C'");
        if (pivroc != 'C')
            throw std::invalid_argument("plapiv: pivroc must be 'C'");
        if (rowcol == 'R') {
            if (m > array_descA[DDLA_M_] || n > array_descA[DDLA_N_])
                throw std::invalid_argument("plapiv: m x n exceeds the descriptor of A");
        } else {
            if (m > array_descA[DDLA_N_] || n > array_descA[DDLA_M_])
                throw std::invalid_argument("plapiv: m x n exceeds the descriptor of A");
        }
    });
}

// Component 2: one buffer of max(length_rowcol_R, length_rowcol_C)
// elements, the logical local extents of the leading-block sub-matrix:
// rowcol='R' pivots rows over the leading n columns, rowcol='C' pivots columns
// over the leading n rows (n is the fixed segment length, see ddla.h).
template <typename T>
void plapiv_workspace_size(const DdlaHandle_t& handle, int n, const int* array_descA,
                           std::size_t* backend_bytes)
{
    const int length_rowcol_R = num_loc(n, array_descA[DDLA_NB_], handle->mypcol_, array_descA[DDLA_CSRC_], handle->npcols_);
    const int length_rowcol_C = num_loc(n, array_descA[DDLA_MB_], handle->myprow_, array_descA[DDLA_RSRC_], handle->nprows_);
    *backend_bytes = std::size_t(std::max(length_rowcol_R, length_rowcol_C)) * sizeof(T);
}

// Component 3.
template <DdlaBackend Backend, typename T>
void plapiv_run(const DdlaHandle_t& handle, char direc, char rowcol,
                int m, int n, T* A, const int* array_descA,
                const int* ipiv, const int* array_descIP,
                void* backend_work, [[maybe_unused]] std::size_t backend_bytes)
{
    try {
        int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
        ddlaGetGridDims(handle, nprows, npcols);
        ddlaGetGridCoords(handle, myprow, mypcol);

        DdlaHandle_t ddla_handle = handle;
        int mb = array_descA[DDLA_MB_];
        int nb = array_descA[DDLA_NB_];
        int lldA = array_descA[DDLA_LLD_];
        const int length_rowcol_R = num_loc(n, nb, mypcol, array_descA[DDLA_CSRC_], npcols);
        const int length_rowcol_C = num_loc(n, mb, myprow, array_descA[DDLA_RSRC_], nprows);
        assert(std::size_t(std::max(length_rowcol_R, length_rowcol_C)) * sizeof(T) <= backend_bytes);

        T* const temp_A_target = static_cast<T*>(backend_work);

        // 初始化 NCCL
        MPI_Comm col_comm = ddla_handle->col_comm;
        MPI_Comm row_comm = ddla_handle->row_comm;
        int i_loc;
        int owner_row;
        int target_row;
        int target_i_global = 0;
        int target_i_loc;
        // Pivot i (global index) is stored in the local ipiv entry of global row i
        // on the owning process row (column-cyclic pivot distribution, pivroc='C'),
        // described by array_descIP, and broadcast within the column communicator.
        // direc='F' applies pivots in ascending i (computing P^T*B for rows /
        // B*P for columns); direc='B' in descending i (computing P*Z for rows /
        // Z*P^T for columns).  rowcol='R' swaps rows i and target in the matrix
        // A (array_descA); rowcol='C' swaps columns i and target.  The matrix
        // and the pivot vector are assumed to share the same row distribution
        // (same mb/irsrc/nprows), so a global row index maps to the same local
        // offset in both array_descA and dIP.
        const int begin = (direc=='F') ? 0 : m-2;
        const int end   = (direc=='F') ? m-1 : -1;
        const int step  = (direc=='F') ? 1 : -1;
        for(int i=begin; i!=end; i+=step){
            i_loc = indx_g2l_r(array_descIP, handle, i);
            owner_row = indxg2p(i, array_descIP[DDLA_MB_], array_descIP[DDLA_RSRC_], nprows);
            if(i_loc>=0){
                target_i_global = ipiv[i_loc] - 1;
            }
            MPI_CHECK(MPI_Bcast(&target_i_global, 1, MPI_INT, owner_row, col_comm));
            if(target_i_global == i)
                continue;
            if(rowcol=='C'){
                // Swap columns i and target of the matrix.  Column i lives on
                // process column owner_col_i and column target on owner_col_t; the
                // two columns are exchanged across the owning process columns of
                // the same process row (row communicator), segment length m_loc.
                const int i_col = indx_g2l_c(array_descA, handle, i);
                const int target_i_col = indx_g2l_c(array_descA, handle, target_i_global);
                const int owner_col_i = indxg2p(i, nb, array_descA[DDLA_CSRC_], npcols);
                const int owner_col_t = indxg2p(target_i_global, nb, array_descA[DDLA_CSRC_], npcols);
                const int length_v = length_rowcol_C;
                if(owner_col_i == owner_col_t){
                    if(mypcol == owner_col_i)
                        blas_impl::swap_run<Backend, T>(ddla_handle, length_v, A + i_col * lldA, 1, A + target_i_col * lldA, 1);
                }else{
                    if(mypcol == owner_col_t){
                        DDLA_CHECK(ddlaMemcpy2D(ddla_handle, temp_A_target, 1 * sizeof(T), A + target_i_col * lldA,
                                                1 * sizeof(T), 1 * sizeof(T), length_v));
                        commSend<Backend>(ddla_handle, CommScope::Row, temp_A_target, (std::size_t)length_v, owner_col_i);
                        commRecv<Backend>(ddla_handle, CommScope::Row, temp_A_target, (std::size_t)length_v, owner_col_i);
                        blas_impl::swap_run<Backend, T>(ddla_handle, length_v, A + target_i_col * lldA, 1, temp_A_target, 1);
                    }else if(mypcol == owner_col_i){
                        commRecv<Backend>(ddla_handle, CommScope::Row, temp_A_target, (std::size_t)length_v, owner_col_t);
                        blas_impl::swap_run<Backend, T>(ddla_handle, length_v, A + i_col * lldA, 1, temp_A_target, 1);
                        commSend<Backend>(ddla_handle, CommScope::Row, temp_A_target, (std::size_t)length_v, owner_col_t);
                    }
                }
            }else{
                // Swap rows i and target of the matrix.  Row i lives on process
                // row owner_row and row target on target_row; the two rows are
                // exchanged across the owning process rows of the same process
                // column (column communicator), segment length n_loc.
                const int i_locA = indx_g2l_r(array_descA, handle, i);
                target_row = indxg2p(target_i_global, array_descA[DDLA_MB_], array_descA[DDLA_RSRC_], nprows);
                target_i_loc = indx_g2l_r(array_descA, handle, target_i_global);
                if(target_row==owner_row){
                    if(myprow==owner_row)
                        blas_impl::swap_run<Backend, T>(ddla_handle, length_rowcol_R, A + i_locA, lldA, A + target_i_loc, lldA);
                }else{
                    if(myprow==target_row){
                        DDLA_CHECK(ddlaMemcpy2D(ddla_handle, temp_A_target, 1 * sizeof(T), A + target_i_loc,
                                                lldA * sizeof(T), 1 * sizeof(T), length_rowcol_R));
                        commSend<Backend>(ddla_handle, CommScope::Col, temp_A_target, (std::size_t)length_rowcol_R, owner_row);
                        commRecv<Backend>(ddla_handle, CommScope::Col, temp_A_target, (std::size_t)length_rowcol_R, owner_row);
                        blas_impl::swap_run<Backend, T>(ddla_handle, length_rowcol_R, A + target_i_loc, lldA, temp_A_target, 1);
                    }else if(myprow==owner_row){
                        commRecv<Backend>(ddla_handle, CommScope::Col, temp_A_target, (std::size_t)length_rowcol_R, target_row);
                        blas_impl::swap_run<Backend, T>(ddla_handle, length_rowcol_R, A + i_locA, lldA, temp_A_target, 1);
                        commSend<Backend>(ddla_handle, CommScope::Col, temp_A_target, (std::size_t)length_rowcol_R, target_row);
                    }
                }
            }
        }

        DDLA_CHECK(ddlaSynchronize(ddla_handle));
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(handle) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(handle); } catch (...) {}
        throw;
    }
}

} // namespace plapiv_impl

template <DdlaBackend Backend, typename T>
void plapiv(
    const DdlaHandle_t& handle, char direc, char rowcol, char pivroc,
    int m, int n,
    T* A,const int* array_descA,
    const int* ipiv, const int* array_descIP,
    int* iwork
)
{
    (void)iwork;
    plapiv_impl::plapiv_check<Backend>(handle, direc, rowcol, pivroc, m, n, array_descA, array_descIP);
    std::size_t backend_bytes = 0;
    plapiv_impl::plapiv_workspace_size<T>(handle, n, array_descA, &backend_bytes);
    vector<char> work(handle, backend_bytes);
    plapiv_impl::plapiv_run<Backend, T>(handle, direc, rowcol, m, n, A, array_descA, ipiv, array_descIP,
                                        work.data(), backend_bytes);
}

#define INSTANTIATE_PLAPIV(BACKEND, TYPE)                                   \
    template void plapiv<BACKEND, TYPE>(const DdlaHandle_t&, char direc, char rowcol, char pivroc, \
    int m, int n, \
    TYPE* A,const int* array_descA, \
    const int* ipiv, const int* array_descIP, \
    int* iwork); \
    template void plapiv_impl::plapiv_run<BACKEND, TYPE>(const DdlaHandle_t&, \
    char direc, char rowcol, int m, int n, \
    TYPE* A, const int* array_descA, const int* ipiv, const int* array_descIP, void*, std::size_t)

#define INSTANTIATE_PLAPIV_CHECK(BACKEND) \
    template void plapiv_impl::plapiv_check<BACKEND>(const DdlaHandle_t&, char, char, char, \
    int, int, const int*, const int*)

#define INSTANTIATE_PLAPIV_SIZE(TYPE) \
    template void plapiv_impl::plapiv_workspace_size<TYPE>(const DdlaHandle_t&, int, const int*, std::size_t*)

DDLA_FOR_EACH_BACKEND(INSTANTIATE_PLAPIV_CHECK)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_PLAPIV)
DDLA_FOR_EACH_TYPE(INSTANTIATE_PLAPIV_SIZE)

#undef INSTANTIATE_PLAPIV_SIZE
#undef INSTANTIATE_PLAPIV_CHECK
#undef INSTANTIATE_PLAPIV



}
