#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <cassert>
#include "ddla_connector.h"
#include "ddla_stream_impl.h"
#include "require_gpu.h"
#include "trsm.h"
#include "transport_block.h"
#include "comm_traits.h"
#include "gemm.h"
namespace ddla{


template<typename T>
void ptrtrs(
    const DdlaHandle_t& handle, const char& side, const char& uplo, const char& trans, const char& diag,
    const int& m, const int& n,
    T* d_A, const int* array_descA,
    T* d_B, const int* array_descB
)
{
    check_desc(array_descB, handle);
    check_desc(array_descA, handle);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);


    DdlaHandle_t ddla_handle = handle;
    detail::require_gpu_backend(ddla_handle, "ptrtrs");
    
    assert(array_descA[DDLA_MB_]==array_descA[DDLA_NB_]);
    assert(array_descA[DDLA_MB_]==array_descB[DDLA_MB_]);
    assert(side=='L'||side=='R');
    assert(uplo=='L'||uplo=='U');
    assert(diag=='U'||diag=='N');
    assert(trans=='N'||trans=='T'||trans=='C');
    int nb = array_descA[DDLA_MB_];
    int lldA = array_descA[DDLA_LLD_];
    int lldB = array_descB[DDLA_LLD_];


    runtimeStream_t stream=ddla_handle->stream;
    deblasHandle_t blasH=ddla_handle->blasH;

    deblasFillMode_t uplo_device = (uplo == 'U') ? DEBLAS_FILL_MODE_UPPER : DEBLAS_FILL_MODE_LOWER;
    deblasDiagType_t diag_device = (diag == 'U') ? DEBLAS_DIAG_UNIT : DEBLAS_DIAG_NON_UNIT;
    deblasOperation_t trans_device;
    if(trans == 'N'){
        trans_device = DEBLAS_OP_N;
    }else if(trans == 'T'){
        trans_device = DEBLAS_OP_T;
    }else{
        trans_device = DEBLAS_OP_C;
    }
    deblasSideMode_t side_device = (side == 'L') ? DEBLAS_SIDE_LEFT : DEBLAS_SIDE_RIGHT;
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
    if(side=='L'){
        assert(m <= array_descA[DDLA_M_] && m <= array_descA[DDLA_N_]);
        assert(m <= array_descB[DDLA_M_] && n <= array_descB[DDLA_N_]);
    }else{
        assert(n <= array_descA[DDLA_M_] && n <= array_descA[DDLA_N_]);
        assert(m <= array_descB[DDLA_M_] && n <= array_descB[DDLA_N_]);
    }
    
    T* d_block_diag,*d_block_A,*d_block_B;
    RUNTIME_CHECK(runtimeMallocAsync(&d_block_diag, nb * nb * sizeof(T), stream));
    // side='L' stages an nb x n_loc block row of B; side='R' an m_loc x nb block column.
    RUNTIME_CHECK(runtimeMallocAsync(&d_block_B, nb * std::max(m_loc_B, n_loc_B) * sizeof(T), stream));
    RUNTIME_CHECK(runtimeMallocAsync(&d_block_A, std::max(m_loc_A, n_loc_A) * nb * sizeof(T), stream));

    int owner_row, owner_col;
    int mm_row_start, mm_col_start, mm_row_step, mm_col_step;
    int64_t A_offset, B_offset;

    if(side == 'L'){
        // Left solve: B := op(A)^{-1} * B. Block forward/backward substitution:
        // at each diagonal block, trsm on the owner row of B, broadcast the
        // solved block row, then gemm-update the remaining rows of B.
        const bool solve_backward = (uplo == 'U' && trans == 'N') || (uplo == 'L' && trans != 'N');
        const char panel_direction = (trans == 'N') ? 'C' : 'R';
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
                RUNTIME_CHECK(runtimeMemcpy2DAsync(
                    d_block_diag, nb_real * sizeof(T),
                    d_A + mm_row_start + mm_col_start * lldA, lldA * sizeof(T),
                    nb_real * sizeof(T), nb_real,
                    runtimeMemcpyDeviceToDevice, stream
                ));
            }
            RUNTIME_CHECK(runtimeStreamSynchronize(stream));
            // 广播当前块行
            if(myprow == owner_row){
                commBcast(ddla_handle, CommScope::Row, d_block_diag, (std::size_t)nb_real * nb_real, owner_col);
                BLAS_CHECK(deblasTrsm(
                    blasH, side_device, uplo_device, trans_device, diag_device,
                    nb_real, n_loc_B, 1.0,
                    d_block_diag, nb_real,
                    d_B + mm_row_start, lldB
                ));
            }
            transport_block(handle, 
                'R', 'N', 
                nb_real, n,
                d_B, n_s, 0, array_descB,
                d_block_B
            );
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
            transport_block(handle, 
                panel_direction, trans,
                g_m, g_n,
                d_A, g_ia, g_ja, array_descA,
                d_block_A
            );
            if(solve_backward){
                length_block_A = mm_row_start;
                B_offset = 0;
            }else{
                length_block_A = m_loc_A - mm_row_start - mm_row_step;
                B_offset = mm_row_start + mm_row_step;
            }
            RUNTIME_CHECK(runtimeStreamSynchronize(stream));
            if(length_block_A > 0){
                gemm<DdlaBackend::GPU, T>(ddla_handle, trans, 'N',
                    length_block_A, n_loc_B, nb_real,
                    (T)-1.0,
                    d_block_A, trans == 'N' ? length_block_A : nb_real,
                    d_block_B, nb_real,
                    (T)1.0,
                    d_B + B_offset, lldB);
            }
            RUNTIME_CHECK(runtimeStreamSynchronize(stream));
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
                RUNTIME_CHECK(runtimeMemcpy2DAsync(
                    d_block_diag, nb_real * sizeof(T),
                    d_A + mm_row_start + mm_col_start * lldA, lldA * sizeof(T),
                    nb_real * sizeof(T), nb_real,
                    runtimeMemcpyDeviceToDevice, stream
                ));
            }
            RUNTIME_CHECK(runtimeStreamSynchronize(stream));
            // broadcast the diagonal block within the column, then solve the
            // block column of B on the owner column's processes
            if(mypcol == owner_col){
                commBcast(ddla_handle, CommScope::Col, d_block_diag, (std::size_t)nb_real * nb_real, owner_row);
                BLAS_CHECK(deblasTrsm(
                    blasH, DEBLAS_SIDE_RIGHT, uplo_device, trans_device, diag_device,
                    m_loc_B, nb_real, 1.0,
                    d_block_diag, nb_real,
                    d_B + mm_col_start * lldB, lldB
                ));
            }
            // gather the solved block column of B (local part: m_loc x nb_real)
            transport_block(handle, 
                'C', 'N',
                m, nb_real,
                d_B, 0, n_s, array_descB,
                d_block_B
            );
            // gather the far-side panel of A: for trans='N' the row panel of A
            // on the far side of the diagonal block; for trans!='N' the column
            // panel of A on the far side, transposed during the gather.
            int length_block_A;
            int g_m, g_n;
            int g_ia, g_ja;
            char panel_dir;
            if(trans == 'N'){
                panel_dir = 'R';
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
                panel_dir = 'C';
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
            transport_block(handle, 
                panel_dir, trans,
                g_m, g_n,
                d_A, g_ia, g_ja, array_descA,
                d_block_A
            );
            RUNTIME_CHECK(runtimeStreamSynchronize(stream));
            if(length_block_A > 0 && m_loc_B > 0){
                gemm<DdlaBackend::GPU, T>(ddla_handle, 'N', 'N',
                    m_loc_B, length_block_A, nb_real,
                    (T)-1.0,
                    d_block_B, m_loc_B,
                    d_block_A, nb_real,
                    (T)1.0,
                    d_B + B_offset, lldB);
            }
            RUNTIME_CHECK(runtimeStreamSynchronize(stream));
        }
    }
    RUNTIME_CHECK(runtimeFreeAsync(d_block_A,stream));
    RUNTIME_CHECK(runtimeFreeAsync(d_block_B,stream));
    RUNTIME_CHECK(runtimeFreeAsync(d_block_diag,stream));
}


template void ptrtrs<float>
(
    const DdlaHandle_t&, const char& side, const char& uplo, const char& trans, const char& diag,
    const int& m, const int& n,
    float* d_A, const int* array_descA,
    float* d_B, const int* array_descB
);

template void ptrtrs<double>
(
    const DdlaHandle_t&, const char& side, const char& uplo, const char& trans, const char& diag,
    const int& m, const int& n,
    double* d_A, const int* array_descA,
    double* d_B, const int* array_descB
);

template void ptrtrs<std::complex<float>>
(
    const DdlaHandle_t&, const char& side, const char& uplo, const char& trans, const char& diag,
    const int& m, const int& n,
    std::complex<float>* d_A, const int* array_descA,
    std::complex<float>* d_B, const int* array_descB
);

template void ptrtrs<std::complex<double>>
(
    const DdlaHandle_t&, const char& side, const char& uplo, const char& trans, const char& diag,
    const int& m, const int& n,
    std::complex<double>* d_A, const int* array_descA,
    std::complex<double>* d_B, const int* array_descB
);


}
