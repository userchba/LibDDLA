#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <cassert>
#include <vector>
#include "ddla_connector.h"
#include "ddla_stream_impl.h"
#include "require_gpu.h"
#include "gemm.h"
#include "trsm.h"
#include "comm_traits.h"

namespace ddla{

template<typename T>
void pgetrf(
    const DdlaHandle_t& handle, const int& m, const int& n,
    T* d_A, const int* array_descA,
    int* ipiv, // host
    int& info  // host
)
{
    check_desc(array_descA, handle);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);


    DdlaHandle_t ddla_handle = handle;
    detail::require_gpu_backend(ddla_handle, "pgetrf");
    assert(m <= array_descA[DDLA_M_] && n <= array_descA[DDLA_N_]);


    int nb = array_descA[DDLA_MB_];
    int nb_real;
    assert(array_descA[DDLA_MB_]==array_descA[DDLA_NB_]);
    int lld = array_descA[DDLA_LLD_];

    int m_loc = num_loc(m, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
    int n_loc = num_loc(n, array_descA[DDLA_NB_], mypcol, array_descA[DDLA_CSRC_], npcols);

    runtimeStream_t stream=ddla_handle->stream;
    deblasHandle_t blasH=ddla_handle->blasH;

    int mm_row_start = 0;
    int mm_col_start = 0;
    int i_loc,j_loc;
    int owner_row,owner_col;

    T *d_temp_block;
    RUNTIME_CHECK(runtimeMallocAsync(&d_temp_block, sizeof(T)*nb*nb, stream));

    T *d_temp_L;
    RUNTIME_CHECK(runtimeMallocAsync(&d_temp_L, sizeof(T)*m_loc*nb, stream));

    T *d_temp_U;
    RUNTIME_CHECK(runtimeMallocAsync(&d_temp_U, sizeof(T)*nb*n_loc, stream));

    info = 0;

    for(int n_s=0;n_s<std::min(m,n);n_s+=nb){
        nb_real = std::min(nb, std::min(m,n)-n_s);

        i_loc = indx_g2l_r(array_descA, handle, n_s);
        j_loc = indx_g2l_c(array_descA, handle, n_s);

        owner_row = indxg2p(n_s, nb, array_descA[DDLA_RSRC_], nprows);
        owner_col = indxg2p(n_s, nb, array_descA[DDLA_CSRC_], npcols);


        // start pgetf2

        pgetf2_panel(handle, 
            m, n, nb_real,
            d_A, n_s, array_descA,
            ipiv, info
        );
        // finish pgetf2
        if(info != 0){
            break;
        }
        // update trailing matrix
        if(i_loc>=0)
            mm_row_start +=nb; // update row start   
        if(j_loc>=0){
            mm_col_start+=nb;
            if(i_loc>=0){
                RUNTIME_CHECK(runtimeMemcpy2DAsync(
                    d_temp_block, nb_real * sizeof(T),
                    d_A + i_loc + j_loc * lld, lld * sizeof(T),
                    nb_real * sizeof(T), nb_real,
                    runtimeMemcpyDeviceToDevice, stream
                ));
                
            }
            if(mm_row_start<m_loc){
                RUNTIME_CHECK(runtimeMemcpy2DAsync(
                    d_temp_L, (m_loc-mm_row_start) * sizeof(T),
                    d_A + mm_row_start + j_loc * lld, lld * sizeof(T),
                    (m_loc - mm_row_start) * sizeof(T), nb_real,
                    runtimeMemcpyDeviceToDevice, stream
                ));
            }
        }
        if(mm_row_start<m_loc){
            commBcast(ddla_handle, CommScope::Row, d_temp_L, (std::size_t)(m_loc - mm_row_start) * nb_real, owner_col);
        }
        // broadcast block column
        if(i_loc>=0){
            commBcast(ddla_handle, CommScope::Row, d_temp_block, (std::size_t)nb_real * nb_real, owner_col);
            if(mm_col_start<n_loc){
                BLAS_CHECK(deblasTrsm(
                    blasH, DEBLAS_SIDE_LEFT, DEBLAS_FILL_MODE_LOWER, DEBLAS_OP_N, DEBLAS_DIAG_UNIT,
                    nb_real, n_loc - mm_col_start, 1.0,
                    d_temp_block, nb_real,
                    d_A + i_loc + mm_col_start * lld, lld)
                );
                RUNTIME_CHECK(runtimeMemcpy2DAsync(
                    d_temp_U, nb_real * sizeof(T),
                    d_A + i_loc + mm_col_start * lld, lld * sizeof(T),
                    nb_real * sizeof(T), n_loc - mm_col_start,
                    runtimeMemcpyDeviceToDevice, stream
                ));
            }   
        }
        if(mm_col_start<n_loc){
            commBcast(ddla_handle, CommScope::Col, d_temp_U, (std::size_t)nb_real * (n_loc - mm_col_start), owner_row);
        }
        if(mm_row_start<m_loc&&mm_col_start<n_loc){
            gemm<DdlaBackend::GPU, T>(ddla_handle, 'N', 'N',
                m_loc - mm_row_start, n_loc - mm_col_start, nb_real,
                -1.0,
                d_temp_L, m_loc - mm_row_start,
                d_temp_U, nb_real,
                1.0,
                d_A + mm_row_start + mm_col_start * lld, lld);
        }
    }
    RUNTIME_CHECK(runtimeFreeAsync(d_temp_block, stream));
    RUNTIME_CHECK(runtimeFreeAsync(d_temp_L, stream));
    RUNTIME_CHECK(runtimeFreeAsync(d_temp_U, stream));
    RUNTIME_CHECK(runtimeStreamSynchronize(stream));

}

template void pgetrf<float>(
    const DdlaHandle_t&, const int& m, const int& n,
    float* d_A, const int* array_descA,
    int* ipiv, // host
    int& info  // host
);

template void pgetrf<double>(
    const DdlaHandle_t&, const int& m, const int& n,
    double* d_A, const int* array_descA,
    int* ipiv, // host
    int& info  // host
);

template void pgetrf<std::complex<float>>(
    const DdlaHandle_t&, const int& m, const int& n,
    std::complex<float>* d_A, const int* array_descA,
    int* ipiv, // host
    int& info  // host
);

template void pgetrf<std::complex<double>>(
    const DdlaHandle_t&, const int& m, const int& n,
    std::complex<double>* d_A, const int* array_descA,
    int* ipiv, // host
    int& info  // host
);

}
