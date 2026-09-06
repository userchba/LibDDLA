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

template <typename T>

void pgetf2_panel(
    const DdlaHandle_t& handle, const int& m, const int& n, const int& nb_real,
    T* d_A, const int& n_start, const int* array_descA,
    int* ipiv, // host
    int& info  // host
)
{
    check_desc(array_descA, handle);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);


    assert(m <= array_descA[DDLA_M_] && n <= array_descA[DDLA_N_]);
    DdlaHandle_t ddla_handle = handle;
    detail::require_gpu_backend(ddla_handle, "pgetf2_panel");



    int nb = array_descA[DDLA_MB_];
    assert(array_descA[DDLA_MB_]==array_descA[DDLA_NB_]);
    int lld = array_descA[DDLA_LLD_];

    int m_loc = num_loc(m, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
    int n_loc = num_loc(n, array_descA[DDLA_NB_], mypcol, array_descA[DDLA_CSRC_], npcols);

    const int panel = std::min(32, nb/2>0?nb/2:1);
    int panel_real;

    runtimeStream_t stream=ddla_handle->stream;
    deblasHandle_t blasH=ddla_handle->blasH;

    int mm_row_start = num_loc(n_start, nb, myprow, array_descA[DDLA_RSRC_], nprows);
    int mm_col_start = num_loc(n_start, nb, mypcol, array_descA[DDLA_CSRC_], npcols);
    int i_loc,j_loc;
    int owner_row;

    T *d_temp_U;
    RUNTIME_CHECK(runtimeMallocAsync(&d_temp_U, sizeof(T)*nb_real*panel, stream));

    info = 0;

    int j_s = indx_g2l_c(array_descA, handle, n_start);
    

    for(int n_s=n_start;n_s<n_start+nb_real;n_s+=panel){
        panel_real = std::min(panel, nb_real+n_start-n_s);

        i_loc = indx_g2l_r(array_descA, handle, n_s);
        j_loc = indx_g2l_c(array_descA, handle, n_s);

        owner_row = indxg2p(n_s, nb, array_descA[DDLA_RSRC_], nprows);
        // start pgetf2
        pgetf2(handle, 
            m, n, panel_real,
            d_A, n_s, array_descA,
            ipiv, info
        );
        if(info != 0){
            RUNTIME_CHECK(runtimeFreeAsync(d_temp_U, stream));
            return;
        }
        // finish pgetf2
        // update trailing matrix
        if(i_loc>=0)
            mm_row_start +=panel; // update row start   
        mm_col_start+=panel;

        // broadcast block column
        if(mm_col_start<j_s + nb_real && j_loc>=0){
            if(i_loc>=0){
                BLAS_CHECK(deblasTrsm(
                    blasH, DEBLAS_SIDE_LEFT, DEBLAS_FILL_MODE_LOWER, DEBLAS_OP_N, DEBLAS_DIAG_UNIT,
                    panel_real, j_s + nb_real - mm_col_start, 1.0,
                    d_A + i_loc + j_loc * lld, lld,
                    d_A + i_loc + mm_col_start * lld, lld)
                );
                RUNTIME_CHECK(runtimeMemcpy2DAsync(
                    d_temp_U, panel_real * sizeof(T),
                    d_A + i_loc + mm_col_start * lld, lld * sizeof(T),
                    panel_real * sizeof(T), j_s + nb_real - mm_col_start,
                    runtimeMemcpyDeviceToDevice, stream
                ));
            } 
            commBcast(ddla_handle, CommScope::Col, d_temp_U, (std::size_t)panel_real * (j_s + nb_real - mm_col_start), owner_row);
        }
        if(mm_row_start<m_loc && mm_col_start<j_s + nb_real && j_loc>=0){
            gemm<DdlaBackend::GPU, T>(ddla_handle, 'N', 'N',
                m_loc - mm_row_start, j_s + nb_real - mm_col_start, panel_real,
                -1.0,
                d_A + mm_row_start + j_loc * lld, lld,
                d_temp_U, panel_real,
                1.0,
                d_A + mm_row_start + mm_col_start * lld, lld
            );
        }
    }
    info = 0;
    RUNTIME_CHECK(runtimeFreeAsync(d_temp_U, stream));

}

template void pgetf2_panel<float>(
    const DdlaHandle_t&, const int& m, const int& n, const int& nb_real,
    float* d_A, const int& n_start, const int* array_descA,
    int* ipiv, // host
    int& info  // host
);

template void pgetf2_panel<double>(
    const DdlaHandle_t&, const int& m, const int& n, const int& nb_real,
    double* d_A, const int& n_start, const int* array_descA,
    int* ipiv, // host
    int& info  // host
);

template void pgetf2_panel<std::complex<float>>(
    const DdlaHandle_t&, const int& m, const int& n, const int& nb_real,
    std::complex<float>* d_A, const int& n_start, const int* array_descA,
    int* ipiv, // host
    int& info  // host
);

template void pgetf2_panel<std::complex<double>>(
    const DdlaHandle_t&, const int& m, const int& n, const int& nb_real,
    std::complex<double>* d_A, const int& n_start, const int* array_descA,
    int* ipiv, // host
    int& info  // host
);

}
