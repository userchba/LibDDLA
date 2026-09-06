#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <cassert>
#include "ddla_connector.h"
#include "ddla_stream_impl.h"
#include "require_gpu.h"
#include <vector>
#include "transport_block.h"
#include "geam.h"
#include "comm_traits.h"

namespace ddla{

template <typename T>
void pgeadd(
    const DdlaHandle_t& handle, const char& transa, const char& transb,
    const int& m, const int& n,
    const T& alpha,
    const T* d_A, const int* array_descA,
    const T& beta,
    const T* d_B, const int* array_descB,
    T* d_C, const int* array_descC
)
{
    check_desc(array_descA, handle);
    check_desc(array_descB, handle);
    check_desc(array_descC, handle);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);


    DdlaHandle_t ddla_handle = handle;
    detail::require_gpu_backend(ddla_handle, "pgeadd");

    if(transa != 'N' || transb != 'N')
    {
        if(nprows != npcols){
            throw std::runtime_error("the trans multiplication is now implemented for mxn(m!=n) mpi grid");
        }
    }
    {
        int mbA, nbA, mbB, nbB, mbC, nbC;
        mbC = array_descC[DDLA_MB_];
        nbC = array_descC[DDLA_NB_];
        if(transa == 'N'){
            mbA = array_descA[DDLA_MB_];
            nbA = array_descA[DDLA_NB_];
        }else{
            mbA = array_descA[DDLA_NB_];
            nbA = array_descA[DDLA_MB_];
        }

        if(transb == 'N'){
            mbB = array_descB[DDLA_MB_];
            nbB = array_descB[DDLA_NB_];
        }else{
            mbB = array_descB[DDLA_NB_];
            nbB = array_descB[DDLA_MB_];
        }
        assert(mbA == mbB && mbA == mbC);
        assert(nbA == nbB && nbA == nbC);
    }


    int m_loc_C = num_loc(m, array_descC[DDLA_MB_], myprow, array_descC[DDLA_RSRC_], nprows);
    int n_loc_C = num_loc(n, array_descC[DDLA_NB_], mypcol, array_descC[DDLA_CSRC_], npcols);

    runtimeStream_t stream = ddla_handle->stream;

    deblasOperation_t opA = (transa == 'N') ? DEBLAS_OP_N :
                            (transa == 'T') ? DEBLAS_OP_T : DEBLAS_OP_C;
    deblasOperation_t opB = (transb == 'N') ? DEBLAS_OP_N :
                            (transb == 'T') ? DEBLAS_OP_T : DEBLAS_OP_C;

    if(transa == 'N' && transb == 'N'){
        BLAS_CHECK(deblasGeam(
            ddla_handle->blasH, opA, opB,
            m_loc_C, n_loc_C,
            alpha, 
            d_A, array_descA[DDLA_LLD_],
            beta,
            d_B, array_descB[DDLA_LLD_],
            d_C, array_descC[DDLA_LLD_]
        ));
        
    }else if (transa != 'N' && transb != 'N'){
        if(myprow == mypcol){
            BLAS_CHECK(deblasGeam(
                ddla_handle->blasH, opA, opB,
                m_loc_C, n_loc_C,
                alpha, 
                d_A, array_descA[DDLA_LLD_],
                beta,
                d_B, array_descB[DDLA_LLD_],
                d_C, array_descC[DDLA_LLD_]
            ));
        }else{
            T* d_temp;
            RUNTIME_CHECK(runtimeMallocAsync((void**)&d_temp, m_loc_C * n_loc_C * sizeof(T), stream));
            BLAS_CHECK(deblasGeam(
                ddla_handle->blasH, opA, opB,
                n_loc_C, m_loc_C,
                alpha,
                d_A, array_descA[DDLA_LLD_],
                beta,
                d_B, array_descB[DDLA_LLD_],
                d_temp, n_loc_C
            ));
            int trans_rank = ddla_handle->rc_to_rank(mypcol, myprow);
            commGroupStart(ddla_handle);
            if(myprow > mypcol){
                commSend(ddla_handle, CommScope::Grid, d_temp, (std::size_t)m_loc_C * n_loc_C, trans_rank);
                commRecv(ddla_handle, CommScope::Grid, d_C, (std::size_t)m_loc_C * n_loc_C, trans_rank);
            }else{
                commRecv(ddla_handle, CommScope::Grid, d_C, (std::size_t)m_loc_C * n_loc_C, trans_rank);
                commSend(ddla_handle, CommScope::Grid, d_temp, (std::size_t)m_loc_C * n_loc_C, trans_rank);
            }
            commGroupEnd(ddla_handle);
            RUNTIME_CHECK(runtimeFreeAsync(d_temp, stream));
        }
    }else{
        if(myprow == mypcol){
            BLAS_CHECK(deblasGeam(
                ddla_handle->blasH, opA, opB,
                m_loc_C, n_loc_C,
                alpha, 
                d_A, array_descA[DDLA_LLD_],
                beta,
                d_B, array_descB[DDLA_LLD_],
                d_C, array_descC[DDLA_LLD_]
            ));
        }else{
            T* d_temp;
            RUNTIME_CHECK(runtimeMallocAsync((void**)&d_temp, m_loc_C * n_loc_C * sizeof(T), stream));
            const T* d_comm = transa != 'N' ? d_A : d_B;
            const T* d_nt = transa == 'N' ? d_A : d_B;
            const T& trans_scale = transa != 'N' ? alpha : beta;
            const T& nontrans_scale = transa == 'N' ? alpha : beta;
            deblasOperation_t op_trans = transa != 'N' ? opA : opB;
            int trans_rank = ddla_handle->rc_to_rank(mypcol, myprow);
            commGroupStart(ddla_handle);
            if(myprow > mypcol){
                commSend(ddla_handle, CommScope::Grid, d_comm, (std::size_t)m_loc_C * n_loc_C, trans_rank);
                commRecv(ddla_handle, CommScope::Grid, d_temp, (std::size_t)m_loc_C * n_loc_C, trans_rank);
            }else{
                commRecv(ddla_handle, CommScope::Grid, d_temp, (std::size_t)m_loc_C * n_loc_C, trans_rank);
                commSend(ddla_handle, CommScope::Grid, d_comm, (std::size_t)m_loc_C * n_loc_C, trans_rank);
            }
            commGroupEnd(ddla_handle);
            BLAS_CHECK(deblasGeam(
                ddla_handle->blasH, op_trans, DEBLAS_OP_N,
                m_loc_C, n_loc_C,
                trans_scale,
                d_temp, n_loc_C,
                nontrans_scale,
                d_nt, m_loc_C,
                d_C, m_loc_C
            ));
            RUNTIME_CHECK(runtimeFreeAsync(d_temp, stream));
        }
    }

    return;
    
}

template void pgeadd<float>(
    const DdlaHandle_t&, const char& transa, const char& transb,
    const int& m, const int& n,
    const float& alpha,
    const float* d_A, const int* array_descA,
    const float& beta,
    const float* d_B, const int* array_descB,
    float* d_C, const int* array_descC
);

template void pgeadd<double>(
    const DdlaHandle_t&, const char& transa, const char& transb,
    const int& m, const int& n,
    const double& alpha,
    const double* d_A, const int* array_descA,
    const double& beta,
    const double* d_B, const int* array_descB,
    double* d_C, const int* array_descC
);

template void pgeadd<std::complex<float>>(
    const DdlaHandle_t&, const char& transa, const char& transb,
    const int& m, const int& n,
    const std::complex<float>& alpha,
    const std::complex<float>* d_A, const int* array_descA,
    const std::complex<float>& beta,
    const std::complex<float>* d_B, const int* array_descB,
    std::complex<float>* d_C, const int* array_descC
);

template void pgeadd<std::complex<double>>(
    const DdlaHandle_t&, const char& transa, const char& transb,
    const int& m, const int& n,
    const std::complex<double>& alpha,
    const std::complex<double>* d_A, const int* array_descA,
    const std::complex<double>& beta,
    const std::complex<double>* d_B, const int* array_descB,
    std::complex<double>* d_C, const int* array_descC
);


}
