#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <cassert>
#include "ddla_connector.h"
#include "ddla_stream_impl.h"
#include "require_backend.h"
#include "comm_traits.h"
#include "row_swap.h"
namespace ddla{

namespace detail {

// The one row-segment exchange behind pswap's row branch and the LU panel's
// pivot application; see row_swap.h for the contract.
template <DdlaBackend Backend, typename T>
void swap_rows(const DdlaHandle_t& h,
               T* A, const int* descA, int row_a, int col_a,
               T* B, const int* descB, int row_b, int col_b,
               int count, T* send, T* recv)
{
    if (count <= 0) return;
    if (A == B && descA == descB && row_a == row_b && col_a == col_b) return;
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(h, nprows, npcols);
    ddlaGetGridCoords(h, myprow, mypcol);
    const int owner_a = indxg2p(row_a, descA[DDLA_MB_], descA[DDLA_RSRC_], nprows);
    const int owner_b = indxg2p(row_b, descB[DDLA_MB_], descB[DDLA_RSRC_], nprows);
    if (owner_a == owner_b) {
        if (myprow != owner_a) return;
        const int la = num_loc(row_a, descA[DDLA_MB_], myprow, descA[DDLA_RSRC_], nprows);
        const int lb = num_loc(row_b, descB[DDLA_MB_], myprow, descB[DDLA_RSRC_], nprows);
        swap<Backend, T>(h, count,
                         A + std::size_t(la) + std::size_t(col_a) * descA[DDLA_LLD_], descA[DDLA_LLD_],
                         B + std::size_t(lb) + std::size_t(col_b) * descB[DDLA_LLD_], descB[DDLA_LLD_]);
        return;
    }
    if (myprow == owner_a) {
        T* row = A + std::size_t(num_loc(row_a, descA[DDLA_MB_], myprow, descA[DDLA_RSRC_], nprows))
               + std::size_t(col_a) * descA[DDLA_LLD_];
        DDLA_CHECK(ddlaMemcpy2D(h, send, sizeof(T), row,
                                std::size_t(descA[DDLA_LLD_]) * sizeof(T), sizeof(T), count));
        commSendrecv<Backend>(h, CommScope::Col, send, recv, std::size_t(count), owner_b);
        DDLA_CHECK(ddlaMemcpy2D(h, row, std::size_t(descA[DDLA_LLD_]) * sizeof(T),
                                recv, sizeof(T), sizeof(T), count));
    } else if (myprow == owner_b) {
        T* row = B + std::size_t(num_loc(row_b, descB[DDLA_MB_], myprow, descB[DDLA_RSRC_], nprows))
               + std::size_t(col_b) * descB[DDLA_LLD_];
        DDLA_CHECK(ddlaMemcpy2D(h, send, sizeof(T), row,
                                std::size_t(descB[DDLA_LLD_]) * sizeof(T), sizeof(T), count));
        commSendrecv<Backend>(h, CommScope::Col, send, recv, std::size_t(count), owner_a);
        DDLA_CHECK(ddlaMemcpy2D(h, row, std::size_t(descB[DDLA_LLD_]) * sizeof(T),
                                recv, sizeof(T), sizeof(T), count));
    }
}

} // namespace detail

template <DdlaBackend Backend, typename T>
void pswap(
    const DdlaHandle_t& handle, int N,
    T* A, int ia, int ja, const int* array_descA, int inca,
    T* B, int ib, int jb, const int* array_descB, int incb
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
    detail::require_backend<Backend>(handle, "pswap");

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

    // The row branch exchanges through a matched sendrecv pair, so it needs
    // two distinct count-sized buffers; the column branch reuses the first.
    const std::size_t temp_len = std::size_t(std::max(
        num_loc(array_descA[DDLA_M_], array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows),
        num_loc(array_descA[DDLA_N_], array_descA[DDLA_NB_], mypcol, array_descA[DDLA_CSRC_], npcols)));
    T* temp_swap;
    DDLA_CHECK(ddlaMalloc(handle, (void**)&temp_swap, sizeof(T) * temp_len * 2));
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
                    swap<Backend, T>(handle, length_v, A + a_offset, 1, B + b_offset, 1);
            }else{
                if(mypcol == owner_col_A){
                    DDLA_CHECK(ddlaMemcpy2D(handle, temp_swap, sizeof(T), A + a_offset, sizeof(T), sizeof(T),
                                            length_v));
                    commSend<Backend>(handle, CommScope::Row, temp_swap, (std::size_t)length_v, owner_col_B);
                    commRecv<Backend>(handle, CommScope::Row, temp_swap, (std::size_t)length_v, owner_col_B);
                    swap<Backend, T>(handle, length_v, temp_swap, 1, A + a_offset, 1);
                }else if(mypcol == owner_col_B){
                    commRecv<Backend>(handle, CommScope::Row, temp_swap, (std::size_t)length_v, owner_col_A);
                    swap<Backend, T>(handle, length_v, temp_swap, 1, B + b_offset, 1);
                    commSend<Backend>(handle, CommScope::Row, temp_swap, (std::size_t)length_v, owner_col_A);
                }
            }
        }
    }else{
        const int Na_loc = num_loc(N + ja, array_descA[DDLA_NB_], mypcol, array_descA[DDLA_CSRC_], npcols);
        const int Nb_loc = num_loc(N + jb, array_descB[DDLA_NB_], mypcol, array_descB[DDLA_CSRC_], npcols);
        int length_v = Na_loc - ja_loc;
        assert(length_v == Nb_loc - jb_loc);
        if(length_v > 0){
            detail::swap_rows<Backend, T>(handle, A, array_descA, ia, ja_loc,
                                          B, array_descB, ib, jb_loc,
                                          length_v, temp_swap, temp_swap + temp_len);
        }
    }
    DDLA_CHECK(ddlaFree(handle, temp_swap));
}

#define INSTANTIATE_PSWAP(BACKEND, TYPE)                                   \
    template void pswap<BACKEND, TYPE>(const DdlaHandle_t&, int N, \
    TYPE* A, int ia, int ja, const int* array_descA, int inca, \
    TYPE* B, int ib, int jb, const int* array_descB, int incb)

#define INSTANTIATE_SWAP_ROWS(BACKEND, TYPE) \
    template void detail::swap_rows<BACKEND, TYPE>(const DdlaHandle_t&, \
    TYPE*, const int*, int, int, TYPE*, const int*, int, int, int, TYPE*, TYPE*)

#if DDLA_HAS_CPU
INSTANTIATE_PSWAP(DdlaBackend::CPU, float);
INSTANTIATE_PSWAP(DdlaBackend::CPU, double);
INSTANTIATE_PSWAP(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PSWAP(DdlaBackend::CPU, std::complex<double>);
INSTANTIATE_SWAP_ROWS(DdlaBackend::CPU, float);
INSTANTIATE_SWAP_ROWS(DdlaBackend::CPU, double);
INSTANTIATE_SWAP_ROWS(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_SWAP_ROWS(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_PSWAP(DdlaBackend::GPU, float);
INSTANTIATE_PSWAP(DdlaBackend::GPU, double);
INSTANTIATE_PSWAP(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PSWAP(DdlaBackend::GPU, std::complex<double>);
INSTANTIATE_SWAP_ROWS(DdlaBackend::GPU, float);
INSTANTIATE_SWAP_ROWS(DdlaBackend::GPU, double);
INSTANTIATE_SWAP_ROWS(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_SWAP_ROWS(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_PSWAP
#undef INSTANTIATE_SWAP_ROWS

} // DDLA
