#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <cassert>
#include <cstddef>
#include "ddla_connector.h"
#include "ddla_stream_impl.h"
#include "require_gpu.h"
#include <vector>
#include <type_traits>
#include <cmath>
#include <algorithm>
#include "trsm.h"
#include "potrf.h"
#include "gemmBatched.h"
#include "herk.h"
#include "gemm.h"
#include "comm_traits.h"

namespace ddla{

template<typename T>
bool ppotrf(
    const DdlaHandle_t& handle, const char& uplo, const int& n,
    T* A, const int& ia, const int& ja, const int* array_descA,
    int& info, // host pointer
    bool is_head, int location
)
{
    check_desc(array_descA, handle);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);


    bool is_nega = false;
    assert(uplo == 'L' || uplo == 'U');
    assert(array_descA[DDLA_MB_] == array_descA[DDLA_NB_]);
    assert(n > 0);
    assert(n <= array_descA[DDLA_M_] && n <= array_descA[DDLA_N_]);
    // The factorization operates on the leading n x n sub-matrix anchored at
    // global (0,0); ia/ja are reserved and must be 1 (1-based).
    assert(ia == 1 && ja == 1);
    DdlaHandle_t ddla_handle = handle;
    detail::require_gpu_backend(ddla_handle, "ppotrf");
    if(is_head)
    if(location != -1 && location != n){
        // Symmetric permutation swapping global row/column `location` with
        // the last index `n`: row swap (inca == m(), full row) then column
        // swap (inca == 1, full column). A is a fully-populated (both
        // triangles) Hermitian array, so both swaps touching every row/
        // column entry keeps the matrix consistently Hermitian afterward --
        // this is not a packed-triangle representation.
        pswap(handle, 
            n,
            A, location, 1, array_descA, array_descA[DDLA_M_],
            A, n, 1, array_descA, array_descA[DDLA_M_]
        );
        pswap(handle, 
            // Was: A, 1, location, array_descA, 1 as the second operand --
            // swapping column `location` with itself, a no-op that left the
            // column swap half of the permutation never applied.
            n,
            A, 1, location, array_descA, 1,
            A, 1, n, array_descA, 1
        );
    }

    int nb = array_descA[DDLA_MB_];
    int lldA = array_descA[DDLA_LLD_];

    // Logical local extents of the leading-block n x n sub-matrix; the
    // descriptor may describe a larger matrix, so all local sizes below are
    // derived from the logical n via num_loc.
    const int m_loc_A = num_loc(n, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
    const int n_loc_A = num_loc(n, array_descA[DDLA_NB_], mypcol, array_descA[DDLA_CSRC_], npcols);

    runtimeStream_t stream=ddla_handle->stream;
    deblasHandle_t blasH=ddla_handle->blasH;
    desolverHandle_t solverH=ddla_handle->solverH;

    deblasFillMode_t uplo_device = (uplo == 'U') ? DEBLAS_FILL_MODE_UPPER : DEBLAS_FILL_MODE_LOWER;
    deblasDiagType_t diag_device = DEBLAS_DIAG_NON_UNIT;
    deblasOperation_t trans_device = DEBLAS_OP_C;
    deblasSideMode_t side_device = (uplo == 'U') ?DEBLAS_SIDE_LEFT : DEBLAS_SIDE_RIGHT;

    T* d_block_diag = nullptr;
    T* d_block_row = nullptr;
    T* d_block_col = nullptr;
    RUNTIME_CHECK(runtimeMallocAsync((void**)&d_block_diag,
                             static_cast<std::size_t>(nb) * nb * sizeof(T), stream));
    RUNTIME_CHECK(runtimeMallocAsync((void**)&d_block_row,
                             static_cast<std::size_t>(nb) * n_loc_A * sizeof(T), stream));
    RUNTIME_CHECK(runtimeMallocAsync((void**)&d_block_col,
                             static_cast<std::size_t>(nb) * m_loc_A * sizeof(T), stream));
    int *d_info = nullptr;
    RUNTIME_CHECK(runtimeMallocAsync((void**)&d_info, sizeof(int), stream));

    int owner_row, owner_col;
    int mm_row_start, mm_col_start;
    int nb_real;

    int num_row_block = m_loc_A / nb;
    int num_col_block = n_loc_A / nb;
    int batchCount = num_row_block * num_col_block;

    T** d_A_array = nullptr;
    T** d_B_array = nullptr;
    T** d_C_array = nullptr;
    std::vector<T*> h_A_array(batchCount), h_B_array(batchCount), h_C_array(batchCount);

    const std::size_t pointer_buffer_bytes = static_cast<std::size_t>(batchCount) * sizeof(T*);
    RUNTIME_CHECK(runtimeMallocAsync((void**)&d_A_array, pointer_buffer_bytes, stream));
    RUNTIME_CHECK(runtimeMallocAsync((void**)&d_B_array, pointer_buffer_bytes, stream));
    RUNTIME_CHECK(runtimeMallocAsync((void**)&d_C_array, pointer_buffer_bytes, stream));
    auto cleanup_device_buffers = [&]()
    {
        RUNTIME_CHECK(runtimeFreeAsync(d_A_array, stream));
        RUNTIME_CHECK(runtimeFreeAsync(d_B_array, stream));
        RUNTIME_CHECK(runtimeFreeAsync(d_C_array, stream));
        RUNTIME_CHECK(runtimeFreeAsync(d_block_diag, stream));
        RUNTIME_CHECK(runtimeFreeAsync(d_block_row, stream));
        RUNTIME_CHECK(runtimeFreeAsync(d_block_col, stream));
        RUNTIME_CHECK(runtimeFreeAsync(d_info, stream));
        RUNTIME_CHECK(runtimeStreamSynchronize(stream));
    };
    int h_info;
    int i_batch_count, row_s, col_s, row_remain, col_remain, length_row, length_col;
    for(int n_s = 0; n_s < n; n_s += nb)
    {
        nb_real = std::min(nb, n - n_s);
        mm_row_start = num_loc(n_s, nb, myprow, array_descA[DDLA_RSRC_], nprows);
        mm_col_start = num_loc(n_s, nb, mypcol, array_descA[DDLA_CSRC_], npcols);

        owner_row = indxg2p(n_s, nb, array_descA[DDLA_RSRC_], nprows);
        owner_col = indxg2p(n_s, nb, array_descA[DDLA_CSRC_], npcols);

        if(myprow == owner_row && mypcol == owner_col)
        {
            if(n_s + nb_real == n && is_head){
                if(nb_real > 1){
                    SOLVER_CHECK(desolverPotrf(solverH, uplo_device, nb_real - 1, A + mm_row_start + mm_col_start * lldA, lldA, d_info));
                    if(uplo == 'L'){
                        BLAS_CHECK(deblasTrsm(
                            blasH, side_device, uplo_device, trans_device, diag_device,
                            1, nb_real - 1, (T)1.0, 
                            A + mm_row_start + mm_col_start * lldA, lldA,
                            A + mm_row_start + nb_real - 1 + mm_col_start * lldA, lldA
                        ));
                        BLAS_CHECK(deblasHerk(
                            blasH, uplo_device, DEBLAS_OP_N,
                            1, nb_real - 1,
                            -1.0, A + mm_row_start + nb_real - 1 + mm_col_start * lldA, lldA,
                            1.0, A + mm_row_start + nb_real - 1 + (mm_col_start + nb_real - 1) * lldA, lldA
                        ));
                    }else{
                        BLAS_CHECK(deblasTrsm(
                            blasH, side_device, uplo_device, trans_device, diag_device,
                            nb_real - 1, 1, (T)1.0, 
                            A + mm_row_start + mm_col_start * lldA, lldA,
                            A + mm_row_start + (mm_col_start + nb_real - 1) * lldA, lldA
                        ));
                        BLAS_CHECK(deblasHerk(
                            blasH, uplo_device, DEBLAS_OP_C,
                            1, nb_real - 1,
                            -1.0, A + mm_row_start + (mm_col_start + nb_real - 1) * lldA, lldA,
                            1.0, A + mm_row_start + nb_real - 1 + (mm_col_start + nb_real - 1) * lldA, lldA
                        ));
                    }
                }
                T last_value;
                RUNTIME_CHECK(runtimeMemcpyAsync(&last_value, A + mm_row_start + nb_real - 1 + (mm_col_start + nb_real - 1) * lldA, sizeof(T), runtimeMemcpyDeviceToHost, stream));
                // The D2H copy above is async; the host reads last_value below,
                // so synchronize the stream before consuming it (no sync exists
                // between the enqueued copy and this read otherwise).
                RUNTIME_CHECK(runtimeStreamSynchronize(stream));
                is_nega = false;
                if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>){
                    if(last_value < 0){
                        is_nega = true;
                        last_value = -last_value;
                    }
                }else if constexpr (std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>>){
                    if(last_value.real() < 0){
                        is_nega = true;
                        last_value = -last_value;
                    }                
                }else{
                    throw std::runtime_error("unsupported template type\n");
                }
                last_value = std::sqrt(last_value);
                RUNTIME_CHECK(runtimeMemcpyAsync(A + mm_row_start + nb_real - 1 + (mm_col_start + nb_real - 1) * lldA, &last_value, sizeof(T), runtimeMemcpyHostToDevice, stream));
            }else
                SOLVER_CHECK(desolverPotrf(solverH, uplo_device, nb_real, A + mm_row_start + mm_col_start * lldA, lldA, d_info));
            RUNTIME_CHECK(runtimeStreamSynchronize(stream));
            RUNTIME_CHECK(runtimeMemcpy(&info, d_info, sizeof(int), runtimeMemcpyDeviceToHost));
            RUNTIME_CHECK(runtimeMemcpy2DAsync(
                d_block_diag, nb_real * sizeof(T),
                A + mm_row_start + mm_col_start * lldA, lldA * sizeof(T),
                nb_real * sizeof(T), nb_real,
                runtimeMemcpyDeviceToDevice, stream
            ));
        }
        if(n_s + nb_real == n)
            MPI_CHECK(MPI_Bcast(&is_nega, 1, MPI_CXX_BOOL, ddla_handle->rc_to_rank(owner_row, owner_col), ddla_handle->comm));
        MPI_CHECK(MPI_Bcast(&info, 1, MPI_INT, ddla_handle->rc_to_rank(owner_row, owner_col), ddla_handle->comm));
        if(info != 0){
            info = info + n_s;
            cleanup_device_buffers();
            return false;
        }
        if(uplo == 'L'){
        if(myprow == owner_row)
            mm_row_start += nb_real;
        length_row = m_loc_A - mm_row_start;
        if(mypcol == owner_col){
            commBcast(ddla_handle, CommScope::Col, d_block_diag, (std::size_t)nb_real * nb_real, owner_row);
            if(length_row > 0){
                BLAS_CHECK(deblasTrsm(
                    blasH, side_device, uplo_device, trans_device, diag_device,
                    length_row, nb_real, (T)1.0, 
                    d_block_diag, nb_real,
                    A + mm_row_start + mm_col_start * lldA, lldA
                ));
                RUNTIME_CHECK(runtimeMemcpy2DAsync(
                    d_block_col, length_row * sizeof(T),
                    A + mm_row_start + mm_col_start * lldA, lldA * sizeof(T),
                    length_row * sizeof(T), nb_real,
                    runtimeMemcpyDeviceToDevice, stream
                ));
            }
        }
        if(mypcol == owner_col)
            mm_col_start += nb_real;
        length_col = n_loc_A - mm_col_start;
        if(length_row > 0){
            commBcast(ddla_handle, CommScope::Row, d_block_col, (std::size_t)length_row * nb_real, owner_col);
        }
        if(myprow == mypcol){
            if(length_col > 0)
                RUNTIME_CHECK(runtimeMemcpyAsync(d_block_row, d_block_col, length_col * nb_real * sizeof(T), runtimeMemcpyDeviceToDevice, stream));
        }
        if(length_col > 0){
            commBcast(ddla_handle, CommScope::Col, d_block_row, (std::size_t)nb_real * length_col, mypcol);
        }
        if(myprow == mypcol){
            if(length_row > 0)
                BLAS_CHECK(deblasHerk(
                    blasH, uplo_device, DEBLAS_OP_N,
                    length_row, nb_real,
                    -1.0, d_block_col, length_row,
                    1.0, A + mm_row_start + mm_col_start * lldA, lldA
                ));
        }else{
            // the first approach in which the unused block will be polluted
            // if(length_row > 0 && length_col > 0)
            //     gemm<DdlaBackend::GPU, T>(
            //         ddla_handle, 'N', 'T',
            //         length_row, length_col, nb_real,
            //         (T)-1.0,
            //         d_block_col, length_row,
            //         d_block_row, length_col,
            //         (T)1.0,
            //         A + mm_row_start + mm_col_start * lldA, lldA
            //     );
            // the second method is to use the batched gemm which will not pollute the unused block
            if(length_row <= 0 || length_col <= 0)
                continue;
            num_col_block = length_col / nb;
            num_row_block = length_row / nb;
            if(length_col % nb !=0)
                num_col_block++;
            if(length_row % nb !=0)
                num_row_block++;
            
            i_batch_count = 0;
            
            col_s = nb;
            row_remain = length_row % nb;
            col_remain = length_col % nb;
            row_s = nb + row_remain;
            if(row_remain != 0){
                int g_row_s = indx_l2g_r(array_descA, handle, m_loc_A - row_remain);
                int g_col_s;
                int length_col_real =  length_col;
                do{
                    length_col_real -= nb;
                    g_col_s = indx_l2g_c(array_descA, handle, mm_col_start + length_col_real);
                }while(g_row_s < g_col_s);
                length_col_real += nb;
                if(length_col_real > 0)
                    gemm<DdlaBackend::GPU, T>(ddla_handle, 'N', 'C',
                        row_remain, length_col_real, nb_real, (T)-1.0,
                        d_block_col + length_row - row_remain, length_row,
                        d_block_row, length_col,
                        (T)1.0, A + mm_row_start + mm_col_start * lldA + (length_row - row_remain), lldA
                    );
            }
            if(col_remain != 0){
                int g_col_s = indx_l2g_c(array_descA, handle, n_loc_A - col_remain);
                int g_row_s;
                int length_row_real = length_row + nb;
                do{
                    length_row_real -= nb;
                    g_row_s = indx_l2g_r(array_descA, handle, mm_row_start + length_row - length_row_real);
                }while(g_row_s < g_col_s);
                if(length_row_real > 0)
                    gemm<DdlaBackend::GPU, T>(ddla_handle, 'N', 'C',
                        length_row_real, col_remain, nb, (T)-1.0,
                        d_block_col + length_row - length_row_real, length_row,
                        d_block_row + length_col - col_remain, length_col,
                        (T)1.0, A + mm_row_start + mm_col_start * lldA + (length_row - length_row_real) + (length_col - col_remain) * lldA, lldA
                    );
            }
            for(;row_s <= num_row_block * nb; row_s += nb){
                int g_row_s = indx_l2g_r(array_descA, handle, m_loc_A - row_s);
                int g_col_s;
                col_s = col_remain;
                do{
                    col_s += nb;
                    g_col_s = indx_l2g_c(array_descA, handle, n_loc_A - col_s);
                }while(g_row_s < g_col_s);
                for(; col_s <= num_col_block * nb; col_s += nb){
                    h_A_array[i_batch_count] = d_block_col + length_row - row_s;
                    h_B_array[i_batch_count] = d_block_row + length_col - col_s;
                    h_C_array[i_batch_count] = A + m_loc_A - row_s + (n_loc_A - col_s) * lldA;
                    i_batch_count++;
                }
            }
            if(i_batch_count == 0) continue;
            RUNTIME_CHECK(runtimeMemcpyAsync(d_A_array, h_A_array.data(), i_batch_count * sizeof(T*), runtimeMemcpyHostToDevice, stream));
            RUNTIME_CHECK(runtimeMemcpyAsync(d_B_array, h_B_array.data(), i_batch_count * sizeof(T*), runtimeMemcpyHostToDevice, stream));
            RUNTIME_CHECK(runtimeMemcpyAsync(d_C_array, h_C_array.data(), i_batch_count * sizeof(T*), runtimeMemcpyHostToDevice, stream));
            BLAS_CHECK(deblasGemmBatched(
                blasH, DEBLAS_OP_N, DEBLAS_OP_C,
                nb, nb, nb_real, -1.0,
                d_A_array, length_row,
                d_B_array, length_col,
                1.0, d_C_array, lldA,
                i_batch_count
            ));

            
            RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
        }
        }else{
            if(mypcol == owner_col)
                mm_col_start += nb_real;
            length_col = n_loc_A - mm_col_start;
            if(myprow == owner_row){
                commBcast(ddla_handle, CommScope::Row, d_block_diag, (std::size_t)nb_real * nb_real, owner_col);
                if(length_col > 0){
                    BLAS_CHECK(deblasTrsm(
                        blasH, side_device, uplo_device, trans_device, diag_device,
                        nb_real, length_col, (T)1.0,
                        d_block_diag, nb_real,
                        A + mm_row_start + mm_col_start * lldA, lldA
                    ));
                    RUNTIME_CHECK(runtimeMemcpy2DAsync(
                        d_block_row, nb_real * sizeof(T),
                        A + mm_row_start + mm_col_start * lldA, lldA * sizeof(T),
                        nb_real * sizeof(T), length_col,
                        runtimeMemcpyDeviceToDevice, stream
                    ));
                }
            }
            if(myprow == owner_row)
                mm_row_start += nb_real;
            length_row = m_loc_A - mm_row_start;
            if(length_col > 0){
                commBcast(ddla_handle, CommScope::Col, d_block_row, (std::size_t)nb_real * length_col, owner_row);
            }
            if(myprow == mypcol){
                if(length_row > 0)
                    RUNTIME_CHECK(runtimeMemcpyAsync(d_block_col, d_block_row, nb_real * length_row * sizeof(T), runtimeMemcpyDeviceToDevice, stream));
            }
            if(length_row > 0){
                commBcast(ddla_handle, CommScope::Row, d_block_col, (std::size_t)nb_real * length_row, myprow);
            }
            if(myprow == mypcol){
                if(length_col > 0)
                    BLAS_CHECK(deblasHerk(
                        blasH, uplo_device, DEBLAS_OP_C,
                        length_col, nb_real,
                        -1.0, d_block_row, nb_real,
                        1.0, A + mm_row_start + mm_col_start * lldA, lldA
                    ));
            }else{
                if(length_row <= 0 || length_col <= 0)
                    continue;

                const int row_full = length_row / nb * nb;
                const int col_full = length_col / nb * nb;
                const int row_remain = length_row - row_full;
                const int col_remain = length_col - col_full;

                if(row_remain != 0){
                    const int row_offset = row_full;
                    const int row_loc = mm_row_start + row_offset;
                    const int g_row = indx_l2g_r(array_descA, handle, row_loc);
                    for(int col_offset = 0; col_offset < length_col; col_offset += nb){
                        const int col_loc = mm_col_start + col_offset;
                        const int col_len = std::min(nb, length_col - col_offset);
                        const int g_col = indx_l2g_c(array_descA, handle, col_loc);
                        if(g_row >= g_col)
                            continue;
                        gemm<DdlaBackend::GPU, T>(ddla_handle, 'C', 'N',
                            row_remain, col_len, nb_real,
                            (T)-1.0,
                            d_block_col + row_offset * nb_real, nb_real,
                            d_block_row + col_offset * nb_real, nb_real,
                            (T)1.0,
                            A + row_loc + col_loc * lldA, lldA
                        );
                    }
                }

                if(col_remain != 0){
                    const int col_offset = col_full;
                    const int col_loc = mm_col_start + col_offset;
                    const int g_col = indx_l2g_c(array_descA, handle, col_loc);
                    for(int row_offset = 0; row_offset < row_full; row_offset += nb){
                        const int row_loc = mm_row_start + row_offset;
                        const int g_row = indx_l2g_r(array_descA, handle, row_loc);
                        if(g_row >= g_col)
                            continue;
                        gemm<DdlaBackend::GPU, T>(ddla_handle, 'C', 'N',
                            nb, col_remain, nb_real,
                            (T)-1.0,
                            d_block_col + row_offset * nb_real, nb_real,
                            d_block_row + col_offset * nb_real, nb_real,
                            (T)1.0,
                            A + row_loc + col_loc * lldA, lldA
                        );
                    }
                }

                i_batch_count = 0;
                for(int row_offset = 0; row_offset < row_full; row_offset += nb){
                    const int row_loc = mm_row_start + row_offset;
                    const int g_row = indx_l2g_r(array_descA, handle, row_loc);
                    for(int col_offset = 0; col_offset < col_full; col_offset += nb){
                        const int col_loc = mm_col_start + col_offset;
                        const int g_col = indx_l2g_c(array_descA, handle, col_loc);
                        if(g_row >= g_col)
                            continue;
                        h_A_array[i_batch_count] = d_block_col + row_offset * nb_real;
                        h_B_array[i_batch_count] = d_block_row + col_offset * nb_real;
                        h_C_array[i_batch_count] = A + row_loc + col_loc * lldA;
                        i_batch_count++;
                    }
                }
                if(i_batch_count > 0){
                    RUNTIME_CHECK(runtimeMemcpyAsync(d_A_array, h_A_array.data(), i_batch_count * sizeof(T*), runtimeMemcpyHostToDevice, stream));
                    RUNTIME_CHECK(runtimeMemcpyAsync(d_B_array, h_B_array.data(), i_batch_count * sizeof(T*), runtimeMemcpyHostToDevice, stream));
                    RUNTIME_CHECK(runtimeMemcpyAsync(d_C_array, h_C_array.data(), i_batch_count * sizeof(T*), runtimeMemcpyHostToDevice, stream));
                    BLAS_CHECK(deblasGemmBatched(
                        blasH, DEBLAS_OP_C, DEBLAS_OP_N,
                        nb, nb, nb_real, -1.0,
                        d_A_array, nb_real,
                        d_B_array, nb_real,
                        1.0, d_C_array, lldA,
                        i_batch_count
                    ));
                }
            }
        }
        RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
    }
    cleanup_device_buffers();
    return is_nega;

}

template bool ppotrf<float>(
    const DdlaHandle_t&, const char& uplo, const int& n,
    float* A, const int& ia, const int& ja, const int* array_descA,
    int& info, // host pointer
    bool is_head, int location
);

template bool ppotrf<double>(
    const DdlaHandle_t&, const char& uplo, const int& n,
    double* A, const int& ia, const int& ja, const int* array_descA,
    int& info, // host pointer
    bool is_head, int location
);

template bool ppotrf<std::complex<float>>(
    const DdlaHandle_t&, const char& uplo, const int& n,
    std::complex<float>* A, const int& ia, const int& ja, const int* array_descA,
    int& info, // host pointer
    bool is_head, int location
);

template bool ppotrf<std::complex<double>>(
    const DdlaHandle_t&, const char& uplo, const int& n,
    std::complex<double>* A, const int& ia, const int& ja, const int* array_descA,
    int& info, // host pointer
    bool is_head, int location
);


}
