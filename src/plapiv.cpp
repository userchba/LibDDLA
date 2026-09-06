#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <cassert>
#include "ddla_connector.h"
#include "ddla_stream_impl.h"
#include "require_gpu.h"
#include "swap.h"
#include "comm_traits.h"

namespace ddla{

template <typename T>
void plapiv(
    const DdlaHandle_t& handle, const char& direc, const char& rowcol, const char& pivroc,
    const int& m, const int& n,
    T* d_A,const int* array_descA,
    const int* ipiv, const int* array_descIP,
    int* iwork
)
{
    check_desc(array_descIP, handle);
    check_desc(array_descA, handle);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);


    DdlaHandle_t ddla_handle = handle;
    detail::require_gpu_backend(ddla_handle, "plapiv");

    assert(direc=='F'||direc=='B');
    assert(rowcol=='R'||rowcol=='C');
    assert(pivroc=='C');
    if(rowcol=='R'){
        assert(m<=array_descA[DDLA_M_]);
        assert(n<=array_descA[DDLA_N_]);
    }else{
        assert(m<=array_descA[DDLA_N_]);
        assert(n<=array_descA[DDLA_M_]);
    }
    (void)iwork;


    int mb = array_descA[DDLA_MB_];
    int nb = array_descA[DDLA_NB_];
    int lldA = array_descA[DDLA_LLD_];
    // Logical local extents of the leading-block sub-matrix: rowcol='R'
    // pivots rows over the leading n columns, rowcol='C' pivots columns over
    // the leading n rows (n is the fixed segment length, see ddla.h).
    const int length_rowcol_R = num_loc(n, nb, mypcol, array_descA[DDLA_CSRC_], npcols);
    const int length_rowcol_C = num_loc(n, mb, myprow, array_descA[DDLA_RSRC_], nprows);

    T*temp_A_target;
    RUNTIME_CHECK(runtimeMallocAsync(&temp_A_target, sizeof(T)*std::max(length_rowcol_R, length_rowcol_C), ddla_handle->stream));

    runtimeStream_t stream = ddla_handle->stream;
    deblasHandle_t blasH = ddla_handle->blasH;

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
    // d_A (array_descA); rowcol='C' swaps columns i and target.  The matrix
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
                    BLAS_CHECK(deblasSwap(blasH, length_v, d_A + i_col * lldA, 1, d_A + target_i_col * lldA, 1));
            }else{
                if(mypcol == owner_col_t){
                    RUNTIME_CHECK(runtimeMemcpy2DAsync(
                        temp_A_target, 1 * sizeof(T),
                        d_A + target_i_col * lldA, 1 * sizeof(T),
                        1 * sizeof(T), length_v,
                        runtimeMemcpyDeviceToDevice, stream
                    ));
                    commSend(ddla_handle, CommScope::Row, temp_A_target, (std::size_t)length_v, owner_col_i);
                    commRecv(ddla_handle, CommScope::Row, temp_A_target, (std::size_t)length_v, owner_col_i);
                    BLAS_CHECK(deblasSwap(blasH, length_v, d_A + target_i_col * lldA, 1, temp_A_target, 1));
                }else if(mypcol == owner_col_i){
                    commRecv(ddla_handle, CommScope::Row, temp_A_target, (std::size_t)length_v, owner_col_t);
                    BLAS_CHECK(deblasSwap(blasH, length_v, d_A + i_col * lldA, 1, temp_A_target, 1));
                    commSend(ddla_handle, CommScope::Row, temp_A_target, (std::size_t)length_v, owner_col_t);
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
                    BLAS_CHECK(deblasSwap(blasH, length_rowcol_R, d_A + i_locA, lldA, d_A + target_i_loc, lldA));
            }else{
                if(myprow==target_row){
                    RUNTIME_CHECK(runtimeMemcpy2DAsync(
                        temp_A_target, 1 * sizeof(T),
                        d_A + target_i_loc, lldA * sizeof(T),
                        1 * sizeof(T), length_rowcol_R,
                        runtimeMemcpyDeviceToDevice, stream
                    ));
                    commSend(ddla_handle, CommScope::Col, temp_A_target, (std::size_t)length_rowcol_R, owner_row);
                    commRecv(ddla_handle, CommScope::Col, temp_A_target, (std::size_t)length_rowcol_R, owner_row);
                    BLAS_CHECK(deblasSwap(blasH, length_rowcol_R, d_A + target_i_loc, lldA, temp_A_target, 1));
                }else if(myprow==owner_row){
                    commRecv(ddla_handle, CommScope::Col, temp_A_target, (std::size_t)length_rowcol_R, target_row);
                    BLAS_CHECK(deblasSwap(blasH, length_rowcol_R, d_A + i_locA, lldA, temp_A_target, 1));
                    commSend(ddla_handle, CommScope::Col, temp_A_target, (std::size_t)length_rowcol_R, target_row);
                }
            }
        }
    }

    RUNTIME_CHECK(runtimeFreeAsync(temp_A_target, stream));
    RUNTIME_CHECK(runtimeStreamSynchronize(stream));

}

template void plapiv<std::complex<double>>(
    const DdlaHandle_t&, const char& direc, const char& rowcol, const char& pivroc,
    const int& m, const int& n,
    std::complex<double>* d_A,const int* array_descA,
    const int* ipiv, const int* array_descIP,
    int* iwork
);

template void plapiv<std::complex<float>>(
    const DdlaHandle_t&, const char& direc, const char& rowcol, const char& pivroc,
    const int& m, const int& n,
    std::complex<float>* d_A,const int* array_descA,
    const int* ipiv, const int* array_descIP,
    int* iwork
);

template void plapiv<float>(
    const DdlaHandle_t&, const char& direc, const char& rowcol, const char& pivroc,
    const int& m, const int& n,
    float* d_A,const int* array_descA,
    const int* ipiv, const int* array_descIP,
    int* iwork
);

template void plapiv<double>(
    const DdlaHandle_t&, const char& direc, const char& rowcol, const char& pivroc,
    const int& m, const int& n,
    double* d_A,const int* array_descA,
    const int* ipiv, const int* array_descIP,
    int* iwork
);


}
