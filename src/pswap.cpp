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
void pswap(
    const DdlaHandle_t& handle, const int& N, 
    T* A, int ia, int ja, const int* array_descA, const int& inca,
    T* B, int ib, int jb, const int* array_descB, const int& incb
)
{
    check_desc(array_descA, handle);
    check_desc(array_descB, handle);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);

    ia--;
    ja--;
    ib--;
    jb--;
    detail::require_gpu_backend(handle, "pswap");

    assert(inca == 1 || inca == array_descA[DDLA_M_]);
    assert(incb == 1 || incb == array_descB[DDLA_M_]);
    if(inca == 1)
        assert(inca == incb);
    const int ia_loc = num_loc(ia, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
    const int ja_loc = num_loc(ja, array_descA[DDLA_NB_], mypcol, array_descA[DDLA_CSRC_], npcols);
    const int ib_loc = num_loc(ib, array_descB[DDLA_MB_], myprow, array_descB[DDLA_RSRC_], nprows);
    const int jb_loc = num_loc(jb, array_descB[DDLA_NB_], mypcol, array_descB[DDLA_CSRC_], npcols);
    
    const size_t a_offset = ia_loc + ja_loc * array_descA[DDLA_LLD_];
    const size_t b_offset = ib_loc + jb_loc * array_descB[DDLA_LLD_];

    T* temp_swap;
    RUNTIME_CHECK(runtimeMallocAsync((void**)&temp_swap, sizeof(T) * std::max(
        num_loc(array_descA[DDLA_M_], array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows),
        num_loc(array_descA[DDLA_N_], array_descA[DDLA_NB_], mypcol, array_descA[DDLA_CSRC_], npcols)), handle->stream));
    if(inca == 1){
        const int Na_loc = num_loc(N + ia, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
        const int Nb_loc = num_loc(N + ib, array_descB[DDLA_MB_], myprow, array_descB[DDLA_RSRC_], nprows);
        const int owner_col_A = indxg2p (ja, array_descA[DDLA_NB_], array_descA[DDLA_CSRC_], npcols);
        const int owner_col_B = indxg2p (jb, array_descB[DDLA_NB_], array_descB[DDLA_CSRC_], npcols);
        int length_v = Na_loc - ia_loc;
        assert(length_v == Nb_loc - ib_loc);
        if(length_v > 0){
            if(owner_col_A == owner_col_B){
                if(mypcol == owner_col_A)
                    BLAS_CHECK(deblasSwap(handle->blasH, length_v, A + a_offset, 1, B + b_offset, 1));
            }else{
                if(mypcol == owner_col_A){
                    RUNTIME_CHECK(runtimeMemcpy2DAsync(
                        temp_swap, sizeof(T), 
                        A + a_offset, sizeof(T),
                        sizeof(T), length_v,
                        runtimeMemcpyDeviceToDevice, handle->stream
                    ));
                    commSend(handle, CommScope::Row, temp_swap, (std::size_t)length_v, owner_col_B);
                    commRecv(handle, CommScope::Row, temp_swap, (std::size_t)length_v, owner_col_B);
                    BLAS_CHECK(deblasSwap(handle->blasH, length_v, temp_swap, 1, A + a_offset, 1));
                }else if(mypcol == owner_col_B){
                    commRecv(handle, CommScope::Row, temp_swap, (std::size_t)length_v, owner_col_A);
                    BLAS_CHECK(deblasSwap(handle->blasH, length_v, temp_swap, 1, B + b_offset, 1));
                    commSend(handle, CommScope::Row, temp_swap, (std::size_t)length_v, owner_col_A);
                }
            }
        }
    }else{
        const int Na_loc = num_loc(N + ja, array_descA[DDLA_NB_], mypcol, array_descA[DDLA_CSRC_], npcols);
        const int Nb_loc = num_loc(N + jb, array_descB[DDLA_NB_], mypcol, array_descB[DDLA_CSRC_], npcols);
        const int owner_row_A = indxg2p (ia, array_descA[DDLA_MB_], array_descA[DDLA_RSRC_], nprows);
        const int owner_row_B = indxg2p (ib, array_descB[DDLA_MB_], array_descB[DDLA_RSRC_], nprows);
        int length_v = Na_loc - ja_loc;
        assert(length_v == Nb_loc - jb_loc);
        if(length_v > 0){
            if(owner_row_A == owner_row_B){
                if(myprow == owner_row_A)
                    BLAS_CHECK(deblasSwap(handle->blasH, length_v, A + a_offset, array_descA[DDLA_LLD_], B + b_offset, array_descB[DDLA_LLD_]));
            }else{
                if(myprow == owner_row_A){
                    RUNTIME_CHECK(runtimeMemcpy2DAsync(
                        temp_swap, sizeof(T), 
                        A + a_offset, array_descA[DDLA_LLD_] * sizeof(T),
                        sizeof(T), length_v,
                        runtimeMemcpyDeviceToDevice, handle->stream
                    ));
                    commSend(handle, CommScope::Col, temp_swap, (std::size_t)length_v, owner_row_B);
                    commRecv(handle, CommScope::Col, temp_swap, (std::size_t)length_v, owner_row_B);
                    BLAS_CHECK(deblasSwap(handle->blasH, length_v, temp_swap, 1, A + a_offset, array_descA[DDLA_LLD_]));
                }else if(myprow == owner_row_B){
                    commRecv(handle, CommScope::Col, temp_swap, (std::size_t)length_v, owner_row_A);
                    BLAS_CHECK(deblasSwap(handle->blasH, length_v, temp_swap, 1, B + b_offset, array_descB[DDLA_LLD_]));
                    commSend(handle, CommScope::Col, temp_swap, (std::size_t)length_v, owner_row_A);
                }
            }
        }
    }
    RUNTIME_CHECK(runtimeFreeAsync(temp_swap, handle->stream));
}

template void pswap<std::complex<double>>(
    const DdlaHandle_t&, const int& N, 
    std::complex<double>* A, int ia, int ja, const int* array_descA, const int& inca,
    std::complex<double>* B, int ib, int jb, const int* array_descB, const int& incb
);

template void pswap<std::complex<float>>(
    const DdlaHandle_t&, const int& N, 
    std::complex<float>* A, int ia, int ja, const int* array_descA, const int& inca,
    std::complex<float>* B, int ib, int jb, const int* array_descB, const int& incb
);

template void pswap<float>(
    const DdlaHandle_t&, const int& N, 
    float* A, int ia, int ja, const int* array_descA, const int& inca,
    float* B, int ib, int jb, const int* array_descB, const int& incb
);

template void pswap<double>(
    const DdlaHandle_t&, const int& N, 
    double* A, int ia, int ja, const int* array_descA, const int& inca,
    double* B, int ib, int jb, const int* array_descB, const int& incb
);
} // DDLA
