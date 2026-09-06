#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <cassert>
#include <complex>
#include "ddla_connector.h"
#include "ddla_stream_impl.h"
#include "require_gpu.h"
#include "comm_traits.h"
#include "swap.h"

#include "pgetf2_kernels.h"

namespace ddla{

namespace {

struct MaxLoc {
    double value;
    int index;
};

template <typename T>
struct PivotBroadcast {
    int max_row = 0;
    int max_prow = 0;
    T max_value = T{};
};

} // namespace

// Factors the leading-block panel starting at global column n_s of width
// nb_real (<= nb), for the logical m-by-n sub-matrix anchored at global (0,0)
// of a possibly-larger descriptor.  All local extents are derived from the
// logical m/n (num_loc), so the panel kernel never touches rows/columns beyond
// the leading block.
template <typename T>
void pgetf2(
    const DdlaHandle_t& handle, const int& m, const int& n, const int& nb_real,
    T* d_A, const int& n_s, const int* array_descA,
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
    detail::require_gpu_backend(ddla_handle, "pgetf2");

    MPI_Comm row_comm = ddla_handle->row_comm;
    MPI_Comm col_comm = ddla_handle->col_comm;


    int m_loc = num_loc(m, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
    int n_loc = num_loc(n, array_descA[DDLA_NB_], mypcol, array_descA[DDLA_CSRC_], npcols);
    int lld = array_descA[DDLA_LLD_];
    int nb = array_descA[DDLA_NB_];

    runtimeStream_t stream=ddla_handle->stream;
    deblasHandle_t blasH=ddla_handle->blasH;
    
    int max_row;
    int max_prow;
    T max_value{};

    T *d_temp;
    const size_t row_buffer_elems = static_cast<size_t>(n_loc > 0 ? n_loc : 1);
    RUNTIME_CHECK(runtimeMallocAsync(&d_temp, sizeof(T)*row_buffer_elems, stream));

    T *d_temp_peer = nullptr;
    RUNTIME_CHECK(runtimeMallocAsync(&d_temp_peer, sizeof(T)*row_buffer_elems, stream));

    void* d_pivot_workspace = nullptr;
    RUNTIME_CHECK(runtimeMallocAsync(
        &d_pivot_workspace, detail::pgetf2_pivot_workspace_size<T>(), stream));
    

    int i_loc = indx_g2l_r(array_descA, handle, n_s);
    int j_loc = indx_g2l_c(array_descA, handle, n_s);

    int owner_row = indxg2p(n_s, nb, array_descA[DDLA_RSRC_], nprows);
    int owner_col = indxg2p(n_s, nb, array_descA[DDLA_CSRC_], npcols);

    int mm_row_start = num_loc(n_s, nb, myprow, array_descA[DDLA_RSRC_], nprows);
    int mm_col_start = num_loc(n_s, nb, mypcol, array_descA[DDLA_CSRC_], npcols);

    info = 0;
    for(int i_tf2 = 0; i_tf2 < nb_real; i_tf2++){
        // find max_rows and value
        int i_panel, j_panel;
        if(i_loc >= 0)
            i_panel = i_loc + i_tf2;
        else
            i_panel = mm_row_start;
        if(j_loc >= 0)
            j_panel = j_loc + i_tf2;
        else
            j_panel = mm_col_start;
        if(j_loc >= 0){
            MaxLoc local_max{-1.0, -1};
            MaxLoc global_max{-1.0, -1};
            T local_max_value{};
            if(i_panel<m_loc){
                int local_max_index = 0;
                detail::pgetf2_find_local_pivot(
                    d_A + j_panel * lld + i_panel, m_loc - i_panel,
                    d_pivot_workspace, stream,
                    local_max.value, local_max_index, local_max_value);
                const int local_row = i_panel + local_max_index;
                local_max.index = indx_l2g_r(array_descA, handle, local_row);
            }
            MPI_CHECK(MPI_Allreduce(&local_max, &global_max, 1,
                                    MPI_DOUBLE_INT, MPI_MAXLOC, col_comm));
            max_row = global_max.index;
            max_prow = indxg2p(max_row, nb, array_descA[DDLA_RSRC_], nprows);
            if(myprow == max_prow){
                max_value = local_max_value;
            }
            MPI_CHECK(MPI_Bcast(&max_value, static_cast<int>(sizeof(T)), MPI_BYTE,
                                max_prow, col_comm));
        }

        PivotBroadcast<T> pivot;
        if(mypcol == owner_col){
            pivot.max_row = max_row;
            pivot.max_prow = max_prow;
            pivot.max_value = max_value;
        }
        MPI_CHECK(MPI_Bcast(&pivot, static_cast<int>(sizeof(pivot)), MPI_BYTE, owner_col, row_comm));
        max_row = pivot.max_row;
        max_prow = pivot.max_prow;
        max_value = pivot.max_value;

        int max_loc_row = indx_g2l_r(array_descA, handle, max_row);
        if(myprow == owner_row){
            ipiv[i_panel] = max_row + 1; // 1-based index like fortran
        }
        // exchange rows
        if(owner_row == max_prow){
            if(myprow == owner_row && max_loc_row != i_panel)
                BLAS_CHECK(deblasSwap(
                    blasH, n_loc,
                    d_A + i_panel, lld,
                    d_A + max_loc_row, lld
                ));
        }else{
            if(myprow == owner_row){
                RUNTIME_CHECK(runtimeMemcpy2DAsync(
                    d_temp, sizeof(T),
                    d_A + i_panel, lld * sizeof(T),
                    sizeof(T), n_loc,
                    runtimeMemcpyDeviceToDevice, stream
                ));
                commGroupStart(ddla_handle);
                commSend(ddla_handle, CommScope::Col, d_temp, (std::size_t)n_loc, max_prow);
                commRecv(ddla_handle, CommScope::Col, d_temp_peer, (std::size_t)n_loc, max_prow);
                commGroupEnd(ddla_handle);
                RUNTIME_CHECK(runtimeMemcpy2DAsync(
                    d_A + i_panel, lld * sizeof(T),
                    d_temp_peer, sizeof(T),
                    sizeof(T), n_loc,
                    runtimeMemcpyDeviceToDevice, stream
                ));
            }else if(myprow == max_prow){
                RUNTIME_CHECK(runtimeMemcpy2DAsync(
                    d_temp, sizeof(T),
                    d_A + max_loc_row, lld * sizeof(T),
                    sizeof(T), n_loc,
                    runtimeMemcpyDeviceToDevice, stream
                ));
                commGroupStart(ddla_handle);
                commSend(ddla_handle, CommScope::Col, d_temp, (std::size_t)n_loc, owner_row);
                commRecv(ddla_handle, CommScope::Col, d_temp_peer, (std::size_t)n_loc, owner_row);
                commGroupEnd(ddla_handle);
                RUNTIME_CHECK(runtimeMemcpy2DAsync(
                    d_A + max_loc_row, lld * sizeof(T),
                    d_temp_peer, sizeof(T),
                    sizeof(T), n_loc,
                    runtimeMemcpyDeviceToDevice, stream
                ));
            }
        }
        if(std::abs(max_value) == 0.0){
            info = n_s+i_tf2+1;
            break;
        }
        if(j_loc>=0){
            const T inverse_pivot = T(1.0) / max_value;
            int64_t a_off;
            int length_row;
            
            if(i_loc>=0){
                a_off = (i_panel + 1) + j_panel * lld;
                length_row = m_loc - (i_panel + 1);
            }else{
                a_off = mm_row_start + j_panel * lld;
                length_row = m_loc - mm_row_start;
            }
            int length_col = nb_real - i_tf2 - 1;
            if(myprow == owner_row && length_col>0){
                RUNTIME_CHECK(runtimeMemcpy2DAsync(
                    d_temp, 1 * sizeof(T),
                    d_A + i_panel + (j_panel + 1) * lld, lld * sizeof(T),
                    1*sizeof(T), length_col,
                    runtimeMemcpyDeviceToDevice, stream
                ));
            }
            if(length_col>0)
                commBcast(ddla_handle, CommScope::Col, d_temp, (std::size_t)length_col, owner_row);
            T* d_trailing = length_col > 0 ? d_A + a_off + lld : d_A + a_off;
            detail::pgetf2_scale_update(
                length_row, length_col, inverse_pivot,
                d_A + a_off, d_temp, d_trailing, lld, stream);
        }
    }
    RUNTIME_CHECK(runtimeFreeAsync(d_temp, stream));
    RUNTIME_CHECK(runtimeFreeAsync(d_temp_peer, stream));
    RUNTIME_CHECK(runtimeFreeAsync(d_pivot_workspace, stream));
    if(info != 0){
        RUNTIME_CHECK(runtimeStreamSynchronize(stream));
    }
}

template void pgetf2<float>(
    const DdlaHandle_t&, const int& m, const int& n, const int& nb_real,
    float* d_A, const int& n_s, const int* array_descA,
    int* ipiv, // host
    int& info  // host
);

template void pgetf2<double>(
    const DdlaHandle_t&, const int& m, const int& n, const int& nb_real,
    double* d_A, const int& n_s, const int* array_descA,
    int* ipiv, // host
    int& info  // host
);

template void pgetf2<std::complex<float>>(
    const DdlaHandle_t&, const int& m, const int& n, const int& nb_real,
    std::complex<float>* d_A, const int& n_s, const int* array_descA,
    int* ipiv, // host
    int& info  // host
);

template void pgetf2<std::complex<double>>(
    const DdlaHandle_t&, const int& m, const int& n, const int& nb_real,
    std::complex<double>* d_A, const int& n_s, const int* array_descA,
    int* ipiv, // host
    int& info  // host
);

} // namespace ddla
