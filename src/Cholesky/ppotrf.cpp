#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <cassert>
#include <cstddef>
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "require_backend.h"
#include <vector>
#include <type_traits>
#include <cmath>
#include <algorithm>
#include "comm_traits.h"

namespace ddla{

template <DdlaBackend Backend, typename T>
bool ppotrf(
    const DdlaHandle_t& handle, char uplo, int n,
    T* A, int ia, int ja, const int* array_descA,
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
    detail::require_backend<Backend>(ddla_handle, "ppotrf");
    if(is_head)
    if(location != -1 && location != n){
        // Symmetric permutation swapping global row/column `location` with
        // the last index `n`: row swap (inca == m(), full row) then column
        // swap (inca == 1, full column). A is a fully-populated (both
        // triangles) Hermitian array, so both swaps touching every row/
        // column entry keeps the matrix consistently Hermitian afterward --
        // this is not a packed-triangle representation.
        pswap<Backend>(handle, 
            n,
            A, location, 1, array_descA, array_descA[DDLA_M_],
            A, n, 1, array_descA, array_descA[DDLA_M_]
        );
        pswap<Backend>(handle, 
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

    // The unified trsm/herk/gemmBatched entries take LAPACK-style characters, so
    // the flags the old GPUBLAS_* enum locals encoded are passed through directly.
    // The side follows the factorized triangle (uplo='U' -> 'L', 'L' -> 'R'),
    // exactly as the removed side_device local did.
    const char side_char = (uplo == 'U') ? 'L' : 'R';

    T* block_diag = nullptr;
    T* block_row = nullptr;
    T* block_col = nullptr;
    DDLA_CHECK(ddlaMalloc(ddla_handle, (void**)&block_diag, static_cast<std::size_t>(nb) * nb * sizeof(T)));
    DDLA_CHECK(ddlaMalloc(ddla_handle, (void**)&block_row, static_cast<std::size_t>(nb) * n_loc_A * sizeof(T)));
    DDLA_CHECK(ddlaMalloc(ddla_handle, (void**)&block_col, static_cast<std::size_t>(nb) * m_loc_A * sizeof(T)));
    int *info_buffer = nullptr;
    DDLA_CHECK(ddlaMalloc(ddla_handle, (void**)&info_buffer, sizeof(int)));
    // potrf is skipped for a one-element head block (is_head, nb_real == 1), yet info is read back below
    DDLA_CHECK(ddlaMemset(ddla_handle, info_buffer, 0, sizeof(int)));

    int owner_row, owner_col;
    int mm_row_start, mm_col_start;
    int nb_real;

    int num_row_block = m_loc_A / nb;
    int num_col_block = n_loc_A / nb;
    int batchCount = num_row_block * num_col_block;

    T** A_array = nullptr;
    T** B_array = nullptr;
    T** C_array = nullptr;
    std::vector<T*> h_A_array(batchCount), h_B_array(batchCount), h_C_array(batchCount);

    const std::size_t pointer_buffer_bytes = static_cast<std::size_t>(batchCount) * sizeof(T*);
    DDLA_CHECK(ddlaMalloc(ddla_handle, (void**)&A_array, pointer_buffer_bytes));
    DDLA_CHECK(ddlaMalloc(ddla_handle, (void**)&B_array, pointer_buffer_bytes));
    DDLA_CHECK(ddlaMalloc(ddla_handle, (void**)&C_array, pointer_buffer_bytes));
    auto cleanup_device_buffers = [&]()
    {
        DDLA_CHECK(ddlaFree(ddla_handle, A_array));
        DDLA_CHECK(ddlaFree(ddla_handle, B_array));
        DDLA_CHECK(ddlaFree(ddla_handle, C_array));
        DDLA_CHECK(ddlaFree(ddla_handle, block_diag));
        DDLA_CHECK(ddlaFree(ddla_handle, block_row));
        DDLA_CHECK(ddlaFree(ddla_handle, block_col));
        DDLA_CHECK(ddlaFree(ddla_handle, info_buffer));
        DDLA_CHECK(ddlaSynchronize(ddla_handle));
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
                    potrf<Backend, T>(ddla_handle, uplo, nb_real - 1, A + mm_row_start + mm_col_start * lldA, lldA, info_buffer);
                    if(uplo == 'L'){
                        trsm<Backend, T>(
                            ddla_handle, side_char, uplo, 'C', 'N',
                            1, nb_real - 1, (T)1.0, 
                            A + mm_row_start + mm_col_start * lldA, lldA,
                            A + mm_row_start + nb_real - 1 + mm_col_start * lldA, lldA
                        );
                        herk<Backend, T>(
                            ddla_handle, uplo, 'N',
                            1, nb_real - 1,
                            -1.0, A + mm_row_start + nb_real - 1 + mm_col_start * lldA, lldA,
                            1.0, A + mm_row_start + nb_real - 1 + (mm_col_start + nb_real - 1) * lldA, lldA
                        );
                    }else{
                        trsm<Backend, T>(
                            ddla_handle, side_char, uplo, 'C', 'N',
                            nb_real - 1, 1, (T)1.0, 
                            A + mm_row_start + mm_col_start * lldA, lldA,
                            A + mm_row_start + (mm_col_start + nb_real - 1) * lldA, lldA
                        );
                        herk<Backend, T>(
                            ddla_handle, uplo, 'C',
                            1, nb_real - 1,
                            -1.0, A + mm_row_start + (mm_col_start + nb_real - 1) * lldA, lldA,
                            1.0, A + mm_row_start + nb_real - 1 + (mm_col_start + nb_real - 1) * lldA, lldA
                        );
                    }
                }
                T last_value;
                DDLA_CHECK(ddlaMemcpy(ddla_handle, &last_value,
                                      A + mm_row_start + nb_real - 1 + (mm_col_start + nb_real - 1) * lldA,
                                      sizeof(T)));
                // The D2H copy above is async; the host reads last_value below,
                // so synchronize the stream before consuming it (no sync exists
                // between the enqueued copy and this read otherwise).
                DDLA_CHECK(ddlaSynchronize(ddla_handle));
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
                DDLA_CHECK(ddlaMemcpy(ddla_handle,
                                      A + mm_row_start + nb_real - 1 + (mm_col_start + nb_real - 1) * lldA,
                                      &last_value, sizeof(T)));
            }else
                potrf<Backend, T>(ddla_handle, uplo, nb_real, A + mm_row_start + mm_col_start * lldA, lldA, info_buffer);
            DDLA_CHECK(ddlaSynchronize(ddla_handle));
            DDLA_CHECK(ddlaMemcpy(ddla_handle, &info, info_buffer, sizeof(int)));
            DDLA_CHECK(ddlaSynchronize(ddla_handle));
            DDLA_CHECK(ddlaMemcpy2D(ddla_handle, block_diag, nb_real * sizeof(T),
                                    A + mm_row_start + mm_col_start * lldA, lldA * sizeof(T), nb_real * sizeof(T),
                                    nb_real));
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
            commBcast<Backend>(ddla_handle, CommScope::Col, block_diag, (std::size_t)nb_real * nb_real, owner_row);
            if(length_row > 0){
                trsm<Backend, T>(
                    ddla_handle, side_char, uplo, 'C', 'N',
                    length_row, nb_real, (T)1.0, 
                    block_diag, nb_real,
                    A + mm_row_start + mm_col_start * lldA, lldA
                );
                DDLA_CHECK(ddlaMemcpy2D(ddla_handle, block_col, length_row * sizeof(T),
                                        A + mm_row_start + mm_col_start * lldA, lldA * sizeof(T),
                                        length_row * sizeof(T), nb_real));
            }
        }
        if(mypcol == owner_col)
            mm_col_start += nb_real;
        length_col = n_loc_A - mm_col_start;
        if(length_row > 0){
            commBcast<Backend>(ddla_handle, CommScope::Row, block_col, (std::size_t)length_row * nb_real, owner_col);
        }
        if(myprow == mypcol){
            if(length_col > 0)
                DDLA_CHECK(ddlaMemcpy(ddla_handle, block_row, block_col, length_col * nb_real * sizeof(T)));
        }
        if(length_col > 0){
            commBcast<Backend>(ddla_handle, CommScope::Col, block_row, (std::size_t)nb_real * length_col, mypcol);
        }
        if(myprow == mypcol){
            if(length_row > 0)
                herk<Backend, T>(
                    ddla_handle, uplo, 'N',
                    length_row, nb_real,
                    -1.0, block_col, length_row,
                    1.0, A + mm_row_start + mm_col_start * lldA, lldA
                );
        }else{
            // the first approach in which the unused block will be polluted
            // if(length_row > 0 && length_col > 0)
            //     gemm<Backend, T>(
            //         ddla_handle, 'N', 'T',
            //         length_row, length_col, nb_real,
            //         (T)-1.0,
            //         block_col, length_row,
            //         block_row, length_col,
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
                    gemm<Backend, T>(ddla_handle, 'N', 'C',
                        row_remain, length_col_real, nb_real, (T)-1.0,
                        block_col + length_row - row_remain, length_row,
                        block_row, length_col,
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
                    gemm<Backend, T>(ddla_handle, 'N', 'C',
                        length_row_real, col_remain, nb, (T)-1.0,
                        block_col + length_row - length_row_real, length_row,
                        block_row + length_col - col_remain, length_col,
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
                    h_A_array[i_batch_count] = block_col + length_row - row_s;
                    h_B_array[i_batch_count] = block_row + length_col - col_s;
                    h_C_array[i_batch_count] = A + m_loc_A - row_s + (n_loc_A - col_s) * lldA;
                    i_batch_count++;
                }
            }
            if(i_batch_count == 0) continue;
            DDLA_CHECK(ddlaMemcpy(ddla_handle, A_array, h_A_array.data(), i_batch_count * sizeof(T*)));
            DDLA_CHECK(ddlaMemcpy(ddla_handle, B_array, h_B_array.data(), i_batch_count * sizeof(T*)));
            DDLA_CHECK(ddlaMemcpy(ddla_handle, C_array, h_C_array.data(), i_batch_count * sizeof(T*)));
            gemmBatched<Backend, T>(
                ddla_handle, 'N', 'C',
                nb, nb, nb_real, (T)-1.0,
                A_array, length_row,
                B_array, length_col,
                (T)1.0, C_array, lldA,
                i_batch_count
            );

            
            DDLA_CHECK(ddlaSynchronize(ddla_handle));
        }
        }else{
            if(mypcol == owner_col)
                mm_col_start += nb_real;
            length_col = n_loc_A - mm_col_start;
            if(myprow == owner_row){
                commBcast<Backend>(ddla_handle, CommScope::Row, block_diag, (std::size_t)nb_real * nb_real, owner_col);
                if(length_col > 0){
                    trsm<Backend, T>(
                        ddla_handle, side_char, uplo, 'C', 'N',
                        nb_real, length_col, (T)1.0,
                        block_diag, nb_real,
                        A + mm_row_start + mm_col_start * lldA, lldA
                    );
                    DDLA_CHECK(ddlaMemcpy2D(ddla_handle, block_row, nb_real * sizeof(T),
                                            A + mm_row_start + mm_col_start * lldA, lldA * sizeof(T),
                                            nb_real * sizeof(T), length_col));
                }
            }
            if(myprow == owner_row)
                mm_row_start += nb_real;
            length_row = m_loc_A - mm_row_start;
            if(length_col > 0){
                commBcast<Backend>(ddla_handle, CommScope::Col, block_row, (std::size_t)nb_real * length_col, owner_row);
            }
            if(myprow == mypcol){
                if(length_row > 0)
                    DDLA_CHECK(ddlaMemcpy(ddla_handle, block_col, block_row, nb_real * length_row * sizeof(T)));
            }
            if(length_row > 0){
                commBcast<Backend>(ddla_handle, CommScope::Row, block_col, (std::size_t)nb_real * length_row, myprow);
            }
            if(myprow == mypcol){
                if(length_col > 0)
                    herk<Backend, T>(
                        ddla_handle, uplo, 'C',
                        length_col, nb_real,
                        -1.0, block_row, nb_real,
                        1.0, A + mm_row_start + mm_col_start * lldA, lldA
                    );
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
                        gemm<Backend, T>(ddla_handle, 'C', 'N',
                            row_remain, col_len, nb_real,
                            (T)-1.0,
                            block_col + row_offset * nb_real, nb_real,
                            block_row + col_offset * nb_real, nb_real,
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
                        gemm<Backend, T>(ddla_handle, 'C', 'N',
                            nb, col_remain, nb_real,
                            (T)-1.0,
                            block_col + row_offset * nb_real, nb_real,
                            block_row + col_offset * nb_real, nb_real,
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
                        h_A_array[i_batch_count] = block_col + row_offset * nb_real;
                        h_B_array[i_batch_count] = block_row + col_offset * nb_real;
                        h_C_array[i_batch_count] = A + row_loc + col_loc * lldA;
                        i_batch_count++;
                    }
                }
                if(i_batch_count > 0){
                    DDLA_CHECK(ddlaMemcpy(ddla_handle, A_array, h_A_array.data(), i_batch_count * sizeof(T*)));
                    DDLA_CHECK(ddlaMemcpy(ddla_handle, B_array, h_B_array.data(), i_batch_count * sizeof(T*)));
                    DDLA_CHECK(ddlaMemcpy(ddla_handle, C_array, h_C_array.data(), i_batch_count * sizeof(T*)));
                    gemmBatched<Backend, T>(
                        ddla_handle, 'C', 'N',
                        nb, nb, nb_real, (T)-1.0,
                        A_array, nb_real,
                        B_array, nb_real,
                        (T)1.0, C_array, lldA,
                        i_batch_count
                    );
                }
            }
        }
        DDLA_CHECK(ddlaSynchronize(ddla_handle));
    }
    cleanup_device_buffers();
    return is_nega;

}

#define INSTANTIATE_PPOTRF(BACKEND, TYPE)                                   \
    template bool ppotrf<BACKEND, TYPE>(const DdlaHandle_t&, char uplo, int n, \
    TYPE* A, int ia, int ja, const int* array_descA, \
    int& info, \
    bool is_head, int location)

#if DDLA_HAS_CPU
INSTANTIATE_PPOTRF(DdlaBackend::CPU, float);
INSTANTIATE_PPOTRF(DdlaBackend::CPU, double);
INSTANTIATE_PPOTRF(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PPOTRF(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_PPOTRF(DdlaBackend::GPU, float);
INSTANTIATE_PPOTRF(DdlaBackend::GPU, double);
INSTANTIATE_PPOTRF(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PPOTRF(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_PPOTRF



}
