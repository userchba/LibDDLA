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
#include "row_swap.h"
#include "pswap_impl.h"
#include "plan_util.h"
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
        blas_impl::swap_run<Backend, T>(h, count,
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

namespace pswap_impl {

// Component 1.  The segments have to pair up entry for entry on every process
// row (or column): the local length of A's range and of B's range must agree.
template <DdlaBackend Backend>
void pswap_check(const DdlaHandle_t& handle, int N,
                 int ia, int ja, const int* array_descA, int inca,
                 int ib, int jb, const int* array_descB, int incb)
{
    detail::agree_check(handle, "pswap", [&](detail::Agree&) {
        check_desc(array_descA, handle);
        check_desc(array_descB, handle);
        detail::require_backend<Backend>(handle, "pswap");

        if (inca != 1 && inca != array_descA[DDLA_M_])
            throw std::invalid_argument("pswap: inca must be 1 or the row count of A");
        if (incb != 1 && incb != array_descB[DDLA_M_])
            throw std::invalid_argument("pswap: incb must be 1 or the row count of B");
        if (inca == 1 && incb != inca)
            throw std::invalid_argument("pswap: inca and incb must select the same direction");

        int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
        ddlaGetGridDims(handle, nprows, npcols);
        ddlaGetGridCoords(handle, myprow, mypcol);
        --ia; --ja; --ib; --jb;
        int length_a, length_b;
        if (inca == 1) {
            length_a = num_loc(N + ia, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows)
                     - num_loc(ia, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
            length_b = num_loc(N + ib, array_descB[DDLA_MB_], myprow, array_descB[DDLA_RSRC_], nprows)
                     - num_loc(ib, array_descB[DDLA_MB_], myprow, array_descB[DDLA_RSRC_], nprows);
        } else {
            length_a = num_loc(N + ja, array_descA[DDLA_NB_], mypcol, array_descA[DDLA_CSRC_], npcols)
                     - num_loc(ja, array_descA[DDLA_NB_], mypcol, array_descA[DDLA_CSRC_], npcols);
            length_b = num_loc(N + jb, array_descB[DDLA_NB_], mypcol, array_descB[DDLA_CSRC_], npcols)
                     - num_loc(jb, array_descB[DDLA_NB_], mypcol, array_descB[DDLA_CSRC_], npcols);
        }
        if (length_a != length_b)
            throw std::invalid_argument("pswap: the segments of A and B are distributed differently");
    });
}

namespace {
// The length of one exchange buffer: the longest local row or column of the
// matrices described by array_descA.
std::size_t exchange_length(const DdlaHandle_t& handle, const int* array_descA)
{
    return std::size_t(std::max(
        num_loc(array_descA[DDLA_M_], array_descA[DDLA_MB_], handle->myprow_, array_descA[DDLA_RSRC_], handle->nprows_),
        num_loc(array_descA[DDLA_N_], array_descA[DDLA_NB_], handle->mypcol_, array_descA[DDLA_CSRC_], handle->npcols_)));
}
} // namespace

// Component 2: [ send | receive ], exchange_length elements each.
template <typename T>
void pswap_workspace_size(const DdlaHandle_t& handle, const int* array_descA, std::size_t* backend_bytes)
{
    *backend_bytes = 2 * detail::padded_bytes<T>(exchange_length(handle, array_descA));
}

// Component 3.
template <DdlaBackend Backend, typename T>
void pswap_run(const DdlaHandle_t& handle, int N,
               T* A, int ia, int ja, const int* array_descA, int inca,
               T* B, int ib, int jb, const int* array_descB, int incb,
               void* backend_work, [[maybe_unused]] std::size_t backend_bytes)
{
    try {
        (void)incb;
        int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
        ddlaGetGridDims(handle, nprows, npcols);
        ddlaGetGridCoords(handle, myprow, mypcol);

        ia--;
        ja--;
        ib--;
        jb--;
        const int ia_loc = num_loc(ia, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
        const int ja_loc = num_loc(ja, array_descA[DDLA_NB_], mypcol, array_descA[DDLA_CSRC_], npcols);
        const int ib_loc = num_loc(ib, array_descB[DDLA_MB_], myprow, array_descB[DDLA_RSRC_], nprows);
        const int jb_loc = num_loc(jb, array_descB[DDLA_NB_], mypcol, array_descB[DDLA_CSRC_], npcols);

        const size_t a_offset = ia_loc + ja_loc * array_descA[DDLA_LLD_];
        const size_t b_offset = ib_loc + jb_loc * array_descB[DDLA_LLD_];

        // The exchange buffers, in the order pswap_workspace_size counts them.
        const std::size_t temp_len = exchange_length(handle, array_descA);
        char* w = static_cast<char*>(backend_work);
        T* const temp_swap = detail::take<T>(w, temp_len);
        T* const temp_recv = detail::take<T>(w, temp_len);
        assert(std::size_t(w - static_cast<char*>(backend_work)) <= backend_bytes);
        if(inca == 1){
            const int Na_loc = num_loc(N + ia, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
            const int owner_col_A = indxg2p (ja, array_descA[DDLA_NB_], array_descA[DDLA_CSRC_], npcols);
            const int owner_col_B = indxg2p (jb, array_descB[DDLA_NB_], array_descB[DDLA_CSRC_], npcols);
            int length_v = Na_loc - ia_loc;
            assert(std::size_t(length_v) <= temp_len || length_v <= 0);
            if(length_v > 0){
                if(owner_col_A == owner_col_B){
                    if(mypcol == owner_col_A)
                        blas_impl::swap_run<Backend, T>(handle, length_v, A + a_offset, 1, B + b_offset, 1);
                }else{
                    if(mypcol == owner_col_A){
                        DDLA_CHECK(ddlaMemcpy2D(handle, temp_swap, sizeof(T), A + a_offset, sizeof(T), sizeof(T),
                                                length_v));
                        commSend<Backend>(handle, CommScope::Row, temp_swap, (std::size_t)length_v, owner_col_B);
                        commRecv<Backend>(handle, CommScope::Row, temp_swap, (std::size_t)length_v, owner_col_B);
                        blas_impl::swap_run<Backend, T>(handle, length_v, temp_swap, 1, A + a_offset, 1);
                    }else if(mypcol == owner_col_B){
                        commRecv<Backend>(handle, CommScope::Row, temp_swap, (std::size_t)length_v, owner_col_A);
                        blas_impl::swap_run<Backend, T>(handle, length_v, temp_swap, 1, B + b_offset, 1);
                        commSend<Backend>(handle, CommScope::Row, temp_swap, (std::size_t)length_v, owner_col_A);
                    }
                }
            }
        }else{
            const int Na_loc = num_loc(N + ja, array_descA[DDLA_NB_], mypcol, array_descA[DDLA_CSRC_], npcols);
            int length_v = Na_loc - ja_loc;
            assert(std::size_t(length_v) <= temp_len || length_v <= 0);
            if(length_v > 0){
                detail::swap_rows<Backend, T>(handle, A, array_descA, ia, ja_loc,
                                              B, array_descB, ib, jb_loc,
                                              length_v, temp_swap, temp_recv);
            }
        }
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(handle) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(handle); } catch (...) {}
        throw;
    }
}

} // namespace pswap_impl

template <DdlaBackend Backend, typename T>
void pswap(
    const DdlaHandle_t& handle, int N,
    T* A, int ia, int ja, const int* array_descA, int inca,
    T* B, int ib, int jb, const int* array_descB, int incb
)
{
    pswap_impl::pswap_check<Backend>(handle, N, ia, ja, array_descA, inca,
                                     ib, jb, array_descB, incb);
    std::size_t backend_bytes = 0;
    pswap_impl::pswap_workspace_size<T>(handle, array_descA, &backend_bytes);
    vector<char> work(handle, backend_bytes);
    pswap_impl::pswap_run<Backend, T>(handle, N, A, ia, ja, array_descA, inca, B, ib, jb, array_descB, incb,
                                      work.data(), backend_bytes);
}

#define INSTANTIATE_PSWAP(BACKEND, TYPE)                                   \
    template void pswap<BACKEND, TYPE>(const DdlaHandle_t&, int N, \
    TYPE* A, int ia, int ja, const int* array_descA, int inca, \
    TYPE* B, int ib, int jb, const int* array_descB, int incb); \
    template void pswap_impl::pswap_run<BACKEND, TYPE>(const DdlaHandle_t&, int N, \
    TYPE* A, int ia, int ja, const int* array_descA, int inca, \
    TYPE* B, int ib, int jb, const int* array_descB, int incb, void*, std::size_t)

#define INSTANTIATE_PSWAP_CHECK(BACKEND) \
    template void pswap_impl::pswap_check<BACKEND>(const DdlaHandle_t&, int N, \
    int ia, int ja, const int* array_descA, int inca, \
    int ib, int jb, const int* array_descB, int incb)

#define INSTANTIATE_PSWAP_SIZE(TYPE) \
    template void pswap_impl::pswap_workspace_size<TYPE>(const DdlaHandle_t&, const int*, std::size_t*)

#define INSTANTIATE_SWAP_ROWS(BACKEND, TYPE) \
    template void detail::swap_rows<BACKEND, TYPE>(const DdlaHandle_t&, \
    TYPE*, const int*, int, int, TYPE*, const int*, int, int, int, TYPE*, TYPE*)

INSTANTIATE_PSWAP_CHECK(DdlaBackend::CPU);
INSTANTIATE_PSWAP(DdlaBackend::CPU, float);
INSTANTIATE_PSWAP(DdlaBackend::CPU, double);
INSTANTIATE_PSWAP(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PSWAP(DdlaBackend::CPU, std::complex<double>);
INSTANTIATE_SWAP_ROWS(DdlaBackend::CPU, float);
INSTANTIATE_SWAP_ROWS(DdlaBackend::CPU, double);
INSTANTIATE_SWAP_ROWS(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_SWAP_ROWS(DdlaBackend::CPU, std::complex<double>);

#if DDLA_HAS_GPU
INSTANTIATE_PSWAP_CHECK(DdlaBackend::GPU);
INSTANTIATE_PSWAP(DdlaBackend::GPU, float);
INSTANTIATE_PSWAP(DdlaBackend::GPU, double);
INSTANTIATE_PSWAP(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PSWAP(DdlaBackend::GPU, std::complex<double>);
INSTANTIATE_SWAP_ROWS(DdlaBackend::GPU, float);
INSTANTIATE_SWAP_ROWS(DdlaBackend::GPU, double);
INSTANTIATE_SWAP_ROWS(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_SWAP_ROWS(DdlaBackend::GPU, std::complex<double>);
#endif

INSTANTIATE_PSWAP_SIZE(float);
INSTANTIATE_PSWAP_SIZE(double);
INSTANTIATE_PSWAP_SIZE(std::complex<float>);
INSTANTIATE_PSWAP_SIZE(std::complex<double>);

#undef INSTANTIATE_PSWAP_SIZE
#undef INSTANTIATE_PSWAP_CHECK
#undef INSTANTIATE_PSWAP
#undef INSTANTIATE_SWAP_ROWS

} // DDLA
