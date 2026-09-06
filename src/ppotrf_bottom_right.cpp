#include "potrf_bottom_right_internal.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <complex>
#include <type_traits>
#include <vector>

#include <ddla/ddla.h>
#include "ddla_desc.h"
#include "ddla_connector.h"
#include "ddla_stream_impl.h"
#include "require_gpu.h"
#include "comm_traits.h"
#include "gemmBatched.h"
#include "herk.h"
#include "syrk.h"
#include "trsm.h"

namespace ddla{

namespace {

template <typename T>
void update_diagonal_tile(
    const deblasHandle_t blas_handle,
    const char uplo,
    const int nb, const int panel_width,
    const T* d_panel, const int ld_panel,
    T* d_tile, const int ldd
)
{
    const bool upper = uplo == 'U';
    const deblasFillMode_t fill_mode = upper
        ? DEBLAS_FILL_MODE_UPPER
        : DEBLAS_FILL_MODE_LOWER;
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>){
        const T minus_one = T(-1);
        const T one = T(1);
        BLAS_CHECK(deblasSyrk(
            blas_handle,
            fill_mode, upper ? DEBLAS_OP_N : DEBLAS_OP_T,
            nb, panel_width,
            minus_one, d_panel, ld_panel,
            one, d_tile, ldd
        ));
    }else{
        using Real = std::conditional_t<
            std::is_same_v<T, std::complex<float>>, float, double>;
        const Real minus_one = Real(-1);
        const Real one = Real(1);
        BLAS_CHECK(deblasHerk(
            blas_handle,
            fill_mode, upper ? DEBLAS_OP_N : DEBLAS_OP_C,
            nb, panel_width,
            minus_one, d_panel, ld_panel,
            one, d_tile, ldd
        ));
    }
}

} // namespace

template <typename T>
void ppotrf_bottom_right(
    const DdlaHandle_t& handle, const char& uplo, const int& n,
    T* d_A, const int* descA, int& info
)
{
    check_desc(descA, handle);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);


    assert(uplo == 'U' || uplo == 'L');
    assert(n >= 0);
    assert(descA[DDLA_DTYPE_] == DDLA_BLOCK_CYCLIC_2D);
    assert(n <= descA[DDLA_M_] && n <= descA[DDLA_N_]);
    assert(descA[DDLA_MB_] == descA[DDLA_NB_]);
    assert(nprows == npcols);

    detail::require_gpu_backend(handle, "ppotrf_bottom_right");
    assert(handle != nullptr);

    info = 0;
    if(n == 0){
        return;
    }

    const int nb = descA[DDLA_NB_];
    assert(nb > 0);

    const int nprocs_dim = nprows;
    const int lld = descA[DDLA_LLD_];

    const runtimeStream_t stream = handle->stream;
    const deblasHandle_t blas_handle = handle->blasH;

    const int max_local_rows = num_loc(
        n, nb, myprow, descA[DDLA_RSRC_], nprocs_dim);
    const int max_local_cols = num_loc(
        n, nb, mypcol, descA[DDLA_CSRC_], nprocs_dim);
    assert(d_A != nullptr || max_local_rows == 0 || max_local_cols == 0);

    T* d_diag = nullptr;
    T* d_row_panel = nullptr;
    T* d_col_panel = nullptr;
    int* d_info = nullptr;
    T** d_left_array = nullptr;
    T** d_right_array = nullptr;
    T** d_target_array = nullptr;

    const int max_row_blocks = max_local_rows / nb;
    const int max_col_blocks = max_local_cols / nb;
    const std::size_t max_batch_count =
        static_cast<std::size_t>(max_row_blocks) * max_col_blocks;

    RUNTIME_CHECK(runtimeMallocAsync(
        reinterpret_cast<void**>(&d_diag),
        static_cast<std::size_t>(nb) * nb * sizeof(T), stream));
    RUNTIME_CHECK(runtimeMallocAsync(
        reinterpret_cast<void**>(&d_row_panel),
        static_cast<std::size_t>(max_local_rows) * nb * sizeof(T), stream));
    RUNTIME_CHECK(runtimeMallocAsync(
        reinterpret_cast<void**>(&d_col_panel),
        static_cast<std::size_t>(max_local_cols) * nb * sizeof(T), stream));
    RUNTIME_CHECK(runtimeMallocAsync(
        reinterpret_cast<void**>(&d_info), sizeof(int), stream));

    const std::size_t pointer_buffer_bytes =
        max_batch_count * sizeof(T*);
    RUNTIME_CHECK(runtimeMallocAsync(
        reinterpret_cast<void**>(&d_left_array), pointer_buffer_bytes, stream));
    RUNTIME_CHECK(runtimeMallocAsync(
        reinterpret_cast<void**>(&d_right_array), pointer_buffer_bytes, stream));
    RUNTIME_CHECK(runtimeMallocAsync(
        reinterpret_cast<void**>(&d_target_array), pointer_buffer_bytes, stream));

    std::vector<T*> h_left_array(max_batch_count);
    std::vector<T*> h_right_array(max_batch_count);
    std::vector<T*> h_target_array(max_batch_count);

    auto cleanup = [&]()
    {
        RUNTIME_CHECK(runtimeFreeAsync(d_left_array, stream));
        RUNTIME_CHECK(runtimeFreeAsync(d_right_array, stream));
        RUNTIME_CHECK(runtimeFreeAsync(d_target_array, stream));
        RUNTIME_CHECK(runtimeFreeAsync(d_diag, stream));
        RUNTIME_CHECK(runtimeFreeAsync(d_row_panel, stream));
        RUNTIME_CHECK(runtimeFreeAsync(d_col_panel, stream));
        RUNTIME_CHECK(runtimeFreeAsync(d_info, stream));
        RUNTIME_CHECK(runtimeStreamSynchronize(stream));
    };

    const int last_block_start = ((n - 1) / nb) * nb;
    // block_start is always a block boundary.  The first diagonal block may be
    // partial, but every leading tile updated below is therefore nb-by-nb.
    for(int block_start = last_block_start;
        block_start >= 0;
        block_start -= nb){
        const int block_width = std::min(nb, n - block_start);
        const int owner_row = indxg2p(
            block_start, nb, descA[DDLA_RSRC_], nprocs_dim);
        const int owner_col = indxg2p(
            block_start, nb, descA[DDLA_CSRC_], nprocs_dim);

        const int local_row_prefix = num_loc(
            block_start, nb, myprow, descA[DDLA_RSRC_], nprocs_dim);
        const int local_col_prefix = num_loc(
            block_start, nb, mypcol, descA[DDLA_CSRC_], nprocs_dim);
        assert(local_row_prefix % nb == 0);
        assert(local_col_prefix % nb == 0);

        int block_info = 0;
        if(myprow == owner_row && mypcol == owner_col){
            T* const d_local_diag = d_A + local_row_prefix
                                  + static_cast<std::size_t>(local_col_prefix) * lld;
            detail::potrf_bottom_right_block(
                uplo, block_width, d_local_diag, lld,
                block_start, d_diag, d_info, block_info, handle
            );
        }

        MPI_CHECK(MPI_Bcast(
            &block_info, 1, MPI_INT,
            handle->rc_to_rank(owner_row, owner_col), handle->comm));
        info = block_info;
        if(info != 0){
            cleanup();
            return;
        }

        if(myprow == owner_row && mypcol == owner_col){
            const T* const d_local_diag = d_A + local_row_prefix
                                        + static_cast<std::size_t>(local_col_prefix) * lld;
            RUNTIME_CHECK(runtimeMemcpy2DAsync(
                d_diag, static_cast<std::size_t>(block_width) * sizeof(T),
                d_local_diag, static_cast<std::size_t>(lld) * sizeof(T),
                static_cast<std::size_t>(block_width) * sizeof(T), block_width,
                runtimeMemcpyDeviceToDevice, stream
            ));
        }

        if(uplo == 'U' && mypcol == owner_col){
            commBcast(handle, CommScope::Col, d_diag, static_cast<std::size_t>(block_width) * block_width, owner_row);
        }

        if(uplo == 'L' && myprow == owner_row){
            commBcast(handle, CommScope::Row, d_diag, static_cast<std::size_t>(block_width) * block_width, owner_col);
        }

        if(uplo == 'U' && local_row_prefix > 0){
            if(mypcol == owner_col){
                const T one = T(1);
                T* const d_local_panel = d_A
                                       + static_cast<std::size_t>(local_col_prefix) * lld;
                BLAS_CHECK(deblasTrsm(
                    blas_handle,
                    DEBLAS_SIDE_RIGHT,
                    DEBLAS_FILL_MODE_UPPER,
                    DEBLAS_OP_C,
                    DEBLAS_DIAG_NON_UNIT,
                    local_row_prefix, block_width,
                    one,
                    d_diag, block_width,
                    d_local_panel, lld
                ));
                RUNTIME_CHECK(runtimeMemcpy2DAsync(
                    d_row_panel,
                    static_cast<std::size_t>(local_row_prefix) * sizeof(T),
                    d_local_panel,
                    static_cast<std::size_t>(lld) * sizeof(T),
                    static_cast<std::size_t>(local_row_prefix) * sizeof(T),
                    block_width,
                    runtimeMemcpyDeviceToDevice, stream
                ));
            }

            commBcast(handle, CommScope::Row, d_row_panel, static_cast<std::size_t>(local_row_prefix) * block_width, owner_col);
        }

        if(uplo == 'U'){
            // A row tile owned by relay_row under irsrc is the same global
            // tile that this process column owns under icsrc.
            const int relay_row =
                (mypcol + descA[DDLA_RSRC_] - descA[DDLA_CSRC_] + nprocs_dim)
                % nprocs_dim;
            if(local_col_prefix > 0){
                if(myprow == relay_row){
                    const int relay_row_count = num_loc(
                        block_start, nb, relay_row,
                        descA[DDLA_RSRC_], nprocs_dim);
                    assert(relay_row_count == local_col_prefix);
                    RUNTIME_CHECK(runtimeMemcpyAsync(
                        d_col_panel, d_row_panel,
                        static_cast<std::size_t>(local_col_prefix)
                            * block_width * sizeof(T),
                        runtimeMemcpyDeviceToDevice, stream
                    ));
                }

                commBcast(handle, CommScope::Col, d_col_panel, static_cast<std::size_t>(local_col_prefix) * block_width, relay_row);
            }
        }else{
            if(local_col_prefix > 0){
                if(myprow == owner_row){
                    const T one = T(1);
                    T* const d_local_panel = d_A + local_row_prefix;
                    const deblasOperation_t trans =
                        std::is_same_v<T, float>
                            || std::is_same_v<T, double>
                        ? DEBLAS_OP_T
                        : DEBLAS_OP_C;
                    BLAS_CHECK(deblasTrsm(
                        blas_handle,
                        DEBLAS_SIDE_LEFT,
                        DEBLAS_FILL_MODE_LOWER,
                        trans,
                        DEBLAS_DIAG_NON_UNIT,
                        block_width, local_col_prefix,
                        one,
                        d_diag, block_width,
                        d_local_panel, lld
                    ));
                    RUNTIME_CHECK(runtimeMemcpy2DAsync(
                        d_col_panel,
                        static_cast<std::size_t>(block_width) * sizeof(T),
                        d_local_panel,
                        static_cast<std::size_t>(lld) * sizeof(T),
                        static_cast<std::size_t>(block_width) * sizeof(T),
                        local_col_prefix,
                        runtimeMemcpyDeviceToDevice, stream
                    ));
                }

                commBcast(handle, CommScope::Col, d_col_panel, static_cast<std::size_t>(local_col_prefix) * block_width, owner_row);
            }

            // A column tile owned by relay_col under icsrc is the same global
            // tile that this process row owns under irsrc.
            const int relay_col =
                (myprow + descA[DDLA_CSRC_] - descA[DDLA_RSRC_] + nprocs_dim)
                % nprocs_dim;
            if(local_row_prefix > 0){
                if(mypcol == relay_col){
                    const int relay_col_count = num_loc(
                        block_start, nb, relay_col,
                        descA[DDLA_CSRC_], nprocs_dim);
                    assert(relay_col_count == local_row_prefix);
                    RUNTIME_CHECK(runtimeMemcpyAsync(
                        d_row_panel, d_col_panel,
                        static_cast<std::size_t>(local_row_prefix)
                            * block_width * sizeof(T),
                        runtimeMemcpyDeviceToDevice, stream
                    ));
                }

                commBcast(handle, CommScope::Row, d_row_panel, static_cast<std::size_t>(local_row_prefix) * block_width, relay_col);
            }
        }

        int batch_count = 0;
        for(int local_col = 0;
            local_col < local_col_prefix;
            local_col += nb){
            const int global_col = indxl2g(
                local_col, nb, mypcol, descA[DDLA_CSRC_], nprocs_dim);
            for(int local_row = 0;
                local_row < local_row_prefix;
                local_row += nb){
                const int global_row = indxl2g(
                    local_row, nb, myprow, descA[DDLA_RSRC_], nprocs_dim);
                if((uplo == 'U' && global_row > global_col)
                   || (uplo == 'L' && global_row < global_col)){
                    continue;
                }

                T* const d_target = d_A + local_row
                                  + static_cast<std::size_t>(local_col) * lld;
                T* const d_left = uplo == 'U'
                    ? d_row_panel + local_row
                    : d_row_panel
                        + static_cast<std::size_t>(local_row) * block_width;
                T* const d_right = uplo == 'U'
                    ? d_col_panel + local_col
                    : d_col_panel
                        + static_cast<std::size_t>(local_col) * block_width;
                const int ld_left = uplo == 'U'
                    ? local_row_prefix
                    : block_width;
                const int ld_right = uplo == 'U'
                    ? local_col_prefix
                    : block_width;
                if(global_row == global_col){
                    update_diagonal_tile(
                        blas_handle,
                        uplo,
                        nb, block_width,
                        d_left, ld_left,
                        d_target, lld
                    );
                    continue;
                }

                assert(static_cast<std::size_t>(batch_count)
                       < max_batch_count);
                h_left_array[batch_count] = d_left;
                h_right_array[batch_count] = d_right;
                h_target_array[batch_count] = d_target;
                ++batch_count;
            }
        }

        if(batch_count > 0){
            const std::size_t active_pointer_bytes =
                static_cast<std::size_t>(batch_count) * sizeof(T*);
            RUNTIME_CHECK(runtimeMemcpyAsync(
                d_left_array, h_left_array.data(), active_pointer_bytes,
                runtimeMemcpyHostToDevice, stream));
            RUNTIME_CHECK(runtimeMemcpyAsync(
                d_right_array, h_right_array.data(), active_pointer_bytes,
                runtimeMemcpyHostToDevice, stream));
            RUNTIME_CHECK(runtimeMemcpyAsync(
                d_target_array, h_target_array.data(), active_pointer_bytes,
                runtimeMemcpyHostToDevice, stream));

            const T minus_one = T(-1);
            const T one = T(1);
            BLAS_CHECK(deblasGemmBatched(
                blas_handle,
                uplo == 'U' ? DEBLAS_OP_N : DEBLAS_OP_C,
                uplo == 'U' ? DEBLAS_OP_C : DEBLAS_OP_N,
                nb, nb, block_width,
                minus_one,
                d_left_array,
                uplo == 'U' ? local_row_prefix : block_width,
                d_right_array,
                uplo == 'U' ? local_col_prefix : block_width,
                one,
                d_target_array, lld,
                batch_count
            ));
        }
    }

    cleanup();
}

template void ppotrf_bottom_right<float>(
    const DdlaHandle_t&, const char&, const int&, float*, const int*, int&);
template void ppotrf_bottom_right<double>(
    const DdlaHandle_t&, const char&, const int&, double*, const int*, int&);
template void ppotrf_bottom_right<std::complex<float>>(
    const DdlaHandle_t&, const char&, const int&, std::complex<float>*, const int*, int&);
template void ppotrf_bottom_right<std::complex<double>>(
    const DdlaHandle_t&, const char&, const int&, std::complex<double>*, const int*, int&);

} // namespace ddla
