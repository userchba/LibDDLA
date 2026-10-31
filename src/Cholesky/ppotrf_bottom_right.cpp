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
#include "ddla_stream.h"
#include "require_backend.h"
#include "comm_traits.h"

namespace ddla{

namespace {

template <DdlaBackend Backend, typename T>
void update_diagonal_tile(
    const DdlaHandle_t& handle,
    const char uplo,
    const int nb, const int panel_width,
    const T* panel, const int ld_panel,
    T* tile, const int ldd
)
{
    const bool upper = uplo == 'U';
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>){
        const T minus_one = T(-1);
        const T one = T(1);
        syrk<Backend, T>(
            handle,
            uplo, upper ? 'N' : 'T',
            nb, panel_width,
            minus_one, panel, ld_panel,
            one, tile, ldd
        );
    }else{
        using Real = std::conditional_t<
            std::is_same_v<T, std::complex<float>>, float, double>;
        const Real minus_one = Real(-1);
        const Real one = Real(1);
        herk<Backend, T>(
            handle,
            uplo, upper ? 'N' : 'C',
            nb, panel_width,
            minus_one, panel, ld_panel,
            one, tile, ldd
        );
    }
}

} // namespace

template <DdlaBackend Backend, typename T>
void ppotrf_bottom_right(
    const DdlaHandle_t& handle, char uplo, int n,
    T* A, const int* descA, int& info
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

    detail::require_backend<Backend>(handle, "ppotrf_bottom_right");
    assert(handle != nullptr);

    info = 0;
    if(n == 0){
        return;
    }

    const int nb = descA[DDLA_NB_];
    assert(nb > 0);

    const int nprocs_dim = nprows;
    const int lld = descA[DDLA_LLD_];


    const int max_local_rows = num_loc(
        n, nb, myprow, descA[DDLA_RSRC_], nprocs_dim);
    const int max_local_cols = num_loc(
        n, nb, mypcol, descA[DDLA_CSRC_], nprocs_dim);
    assert(A != nullptr || max_local_rows == 0 || max_local_cols == 0);

    T* diag = nullptr;
    T* row_panel = nullptr;
    T* col_panel = nullptr;
    int* info_buffer = nullptr;
    T** left_array = nullptr;
    T** right_array = nullptr;
    T** target_array = nullptr;

    const int max_row_blocks = max_local_rows / nb;
    const int max_col_blocks = max_local_cols / nb;
    const std::size_t max_batch_count =
        static_cast<std::size_t>(max_row_blocks) * max_col_blocks;

    DDLA_CHECK(ddlaMalloc(handle, reinterpret_cast<void**>(&diag), static_cast<std::size_t>(nb) * nb * sizeof(T)));
    DDLA_CHECK(ddlaMalloc(handle, reinterpret_cast<void**>(&row_panel),
                          static_cast<std::size_t>(max_local_rows) * nb * sizeof(T)));
    DDLA_CHECK(ddlaMalloc(handle, reinterpret_cast<void**>(&col_panel),
                          static_cast<std::size_t>(max_local_cols) * nb * sizeof(T)));
    DDLA_CHECK(ddlaMalloc(handle, reinterpret_cast<void**>(&info_buffer), sizeof(int)));

    const std::size_t pointer_buffer_bytes =
        max_batch_count * sizeof(T*);
    DDLA_CHECK(ddlaMalloc(handle, reinterpret_cast<void**>(&left_array), pointer_buffer_bytes));
    DDLA_CHECK(ddlaMalloc(handle, reinterpret_cast<void**>(&right_array), pointer_buffer_bytes));
    DDLA_CHECK(ddlaMalloc(handle, reinterpret_cast<void**>(&target_array), pointer_buffer_bytes));

    std::vector<T*> h_left_array(max_batch_count);
    std::vector<T*> h_right_array(max_batch_count);
    std::vector<T*> h_target_array(max_batch_count);

    auto cleanup = [&]()
    {
        DDLA_CHECK(ddlaFree(handle, left_array));
        DDLA_CHECK(ddlaFree(handle, right_array));
        DDLA_CHECK(ddlaFree(handle, target_array));
        DDLA_CHECK(ddlaFree(handle, diag));
        DDLA_CHECK(ddlaFree(handle, row_panel));
        DDLA_CHECK(ddlaFree(handle, col_panel));
        DDLA_CHECK(ddlaFree(handle, info_buffer));
        DDLA_CHECK(ddlaSynchronize(handle));
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
            T* const local_diag = A + local_row_prefix
                                  + static_cast<std::size_t>(local_col_prefix) * lld;
            detail::potrf_bottom_right_block<Backend>(
                uplo, block_width, local_diag, lld,
                block_start, diag, info_buffer, block_info, handle
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
            const T* const local_diag = A + local_row_prefix
                                        + static_cast<std::size_t>(local_col_prefix) * lld;
            DDLA_CHECK(ddlaMemcpy2D(handle, diag, static_cast<std::size_t>(block_width) * sizeof(T), local_diag,
                                    static_cast<std::size_t>(lld) * sizeof(T),
                                    static_cast<std::size_t>(block_width) * sizeof(T), block_width));
        }

        if(uplo == 'U' && mypcol == owner_col){
            commBcast<Backend>(handle, CommScope::Col, diag, static_cast<std::size_t>(block_width) * block_width, owner_row);
        }

        if(uplo == 'L' && myprow == owner_row){
            commBcast<Backend>(handle, CommScope::Row, diag, static_cast<std::size_t>(block_width) * block_width, owner_col);
        }

        if(uplo == 'U' && local_row_prefix > 0){
            if(mypcol == owner_col){
                const T one = T(1);
                T* const local_panel = A
                                       + static_cast<std::size_t>(local_col_prefix) * lld;
                trsm<Backend, T>(
                    handle,
                    'R',
                    'U',
                    'C',
                    'N',
                    local_row_prefix, block_width,
                    one,
                    diag, block_width,
                    local_panel, lld
                );
                DDLA_CHECK(ddlaMemcpy2D(handle, row_panel, static_cast<std::size_t>(local_row_prefix) * sizeof(T),
                                        local_panel, static_cast<std::size_t>(lld) * sizeof(T),
                                        static_cast<std::size_t>(local_row_prefix) * sizeof(T), block_width));
            }

            commBcast<Backend>(handle, CommScope::Row, row_panel, static_cast<std::size_t>(local_row_prefix) * block_width, owner_col);
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
                    DDLA_CHECK(ddlaMemcpy(handle, col_panel, row_panel,
                                          static_cast<std::size_t>(local_col_prefix) * block_width * sizeof(T)));
                }

                commBcast<Backend>(handle, CommScope::Col, col_panel, static_cast<std::size_t>(local_col_prefix) * block_width, relay_row);
            }
        }else{
            if(local_col_prefix > 0){
                if(myprow == owner_row){
                    const T one = T(1);
                    T* const local_panel = A + local_row_prefix;
                    const char trans =
                        std::is_same_v<T, float>
                            || std::is_same_v<T, double>
                        ? 'T'
                        : 'C';
                    trsm<Backend, T>(
                        handle,
                        'L',
                        'L',
                        trans,
                        'N',
                        block_width, local_col_prefix,
                        one,
                        diag, block_width,
                        local_panel, lld
                    );
                    DDLA_CHECK(ddlaMemcpy2D(handle, col_panel, static_cast<std::size_t>(block_width) * sizeof(T),
                                            local_panel, static_cast<std::size_t>(lld) * sizeof(T),
                                            static_cast<std::size_t>(block_width) * sizeof(T), local_col_prefix));
                }

                commBcast<Backend>(handle, CommScope::Col, col_panel, static_cast<std::size_t>(local_col_prefix) * block_width, owner_row);
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
                    DDLA_CHECK(ddlaMemcpy(handle, row_panel, col_panel,
                                          static_cast<std::size_t>(local_row_prefix) * block_width * sizeof(T)));
                }

                commBcast<Backend>(handle, CommScope::Row, row_panel, static_cast<std::size_t>(local_row_prefix) * block_width, relay_col);
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

                T* const target = A + local_row
                                  + static_cast<std::size_t>(local_col) * lld;
                T* const left = uplo == 'U'
                    ? row_panel + local_row
                    : row_panel
                        + static_cast<std::size_t>(local_row) * block_width;
                T* const right = uplo == 'U'
                    ? col_panel + local_col
                    : col_panel
                        + static_cast<std::size_t>(local_col) * block_width;
                const int ld_left = uplo == 'U'
                    ? local_row_prefix
                    : block_width;
                const int ld_right = uplo == 'U'
                    ? local_col_prefix
                    : block_width;
                if(global_row == global_col){
                    update_diagonal_tile<Backend, T>(
                        handle,
                        uplo,
                        nb, block_width,
                        left, ld_left,
                        target, lld
                    );
                    continue;
                }

                assert(static_cast<std::size_t>(batch_count)
                       < max_batch_count);
                h_left_array[batch_count] = left;
                h_right_array[batch_count] = right;
                h_target_array[batch_count] = target;
                ++batch_count;
            }
        }

        if(batch_count > 0){
            const std::size_t active_pointer_bytes =
                static_cast<std::size_t>(batch_count) * sizeof(T*);
            DDLA_CHECK(ddlaMemcpy(handle, left_array, h_left_array.data(), active_pointer_bytes));
            DDLA_CHECK(ddlaMemcpy(handle, right_array, h_right_array.data(), active_pointer_bytes));
            DDLA_CHECK(ddlaMemcpy(handle, target_array, h_target_array.data(), active_pointer_bytes));

            const T minus_one = T(-1);
            const T one = T(1);
            gemmBatched<Backend, T>(
                handle,
                uplo == 'U' ? 'N' : 'C',
                uplo == 'U' ? 'C' : 'N',
                nb, nb, block_width,
                minus_one,
                left_array,
                uplo == 'U' ? local_row_prefix : block_width,
                right_array,
                uplo == 'U' ? local_col_prefix : block_width,
                one,
                target_array, lld,
                batch_count
            );
        }
    }

    cleanup();
}

#define INSTANTIATE_PPOTRF_BOTTOM_RIGHT(BACKEND, TYPE)                                   \
    template void ppotrf_bottom_right<BACKEND, TYPE>(const DdlaHandle_t&, char, int, TYPE*, const int*, int&)

#if DDLA_HAS_CPU
INSTANTIATE_PPOTRF_BOTTOM_RIGHT(DdlaBackend::CPU, float);
INSTANTIATE_PPOTRF_BOTTOM_RIGHT(DdlaBackend::CPU, double);
INSTANTIATE_PPOTRF_BOTTOM_RIGHT(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PPOTRF_BOTTOM_RIGHT(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_PPOTRF_BOTTOM_RIGHT(DdlaBackend::GPU, float);
INSTANTIATE_PPOTRF_BOTTOM_RIGHT(DdlaBackend::GPU, double);
INSTANTIATE_PPOTRF_BOTTOM_RIGHT(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PPOTRF_BOTTOM_RIGHT(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_PPOTRF_BOTTOM_RIGHT


} // namespace ddla
