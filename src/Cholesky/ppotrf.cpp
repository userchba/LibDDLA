#include "cholesky_workspace.h"
#include "BLAS/blas_impl.h"
#include "check_agree.h"
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "ddla_desc.h"
#include <cassert>
#include <cstddef>
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "plan_util.h"
#include "require_backend.h"
#include <vector>
#include <type_traits>
#include <cmath>
#include <algorithm>
#include "comm_traits.h"
#include "instantiate.h"

namespace ddla{

namespace cholesky_impl {
template<DdlaBackend Backend>
void ppotrf_check(const DdlaHandle_t& h,char uplo,int n,const void* A,int ia,int ja,const int* d,bool head,int location)
{
    detail::agree_check(h,"ppotrf",[&](detail::Agree&){check_desc(d,h);detail::require_backend<Backend>(h,"ppotrf");
        if((uplo!='L' && uplo!='U') || n<0 || n>d[DDLA_M_] || n>d[DDLA_N_] || d[DDLA_MB_]!=d[DDLA_NB_] || ia!=1 || ja!=1)
            throw std::invalid_argument("ppotrf: invalid triangle, dimensions, blocks or offsets");
        if(head && location!=-1 && (location<1 || location>n))throw std::invalid_argument("ppotrf: invalid head location");
        const int ml=num_loc(n,d[DDLA_MB_],h->myprow_,d[DDLA_RSRC_],h->nprows_),nl=num_loc(n,d[DDLA_NB_],h->mypcol_,d[DDLA_CSRC_],h->npcols_);
        if(n && ml && nl && !A)throw std::invalid_argument("ppotrf: null local matrix");});
}
template<DdlaBackend Backend,class T>
bool ppotrf_run(
    const DdlaHandle_t& handle, char uplo, int n,
    T* A, int ia, int ja, const int* array_descA,
    int& info, // host pointer
    bool is_head, int location,
    void* backend_work, std::size_t backend_bytes, void* host_work, [[maybe_unused]] std::size_t host_bytes
)
{
    if(n==0){info=0;return false;}
    if (handle->nprocs==1 && !is_head) {
        char* work=static_cast<char*>(backend_work);
        int* status=&info;
        std::size_t used=0;
        if constexpr (Backend==DdlaBackend::GPU) {
            status=detail::take<int>(work,1);
            used=detail::padded_bytes<int>(1);
        }
        lapack_impl::potrf_run<Backend,T>(handle,uplo,n,A,array_descA[DDLA_LLD_],status,
            work,backend_bytes-used);
        if constexpr (Backend==DdlaBackend::GPU) {
            DDLA_CHECK(ddlaMemcpy(handle,&info,status,sizeof(int)));
            DDLA_CHECK(ddlaSynchronize(handle));
        }
        return false;
    }
    bool is_nega=false;
    int nprows=handle->nprows_,npcols=handle->npcols_,myprow=handle->myprow_,mypcol=handle->mypcol_;
    DdlaHandle_t ddla_handle=handle;
    if(is_head)
    if(location != -1 && location != n){
        // Symmetric permutation swapping global row/column `location` with
        // the last index `n`: row swap (inca == m(), full row) then column
        // swap (inca == 1, full column). A is a fully-populated (both
        // triangles) Hermitian array, so both swaps touching every row/
        // column entry keeps the matrix consistently Hermitian afterward --
        // this is not a packed-triangle representation.
        pswap_impl::pswap_run<Backend,T>(handle,
            n,
            A, location, 1, array_descA, array_descA[DDLA_M_],
            A, n, 1, array_descA, array_descA[DDLA_M_],
            backend_work, backend_bytes
        );
        pswap_impl::pswap_run<Backend,T>(handle,
            // Was: A, 1, location, array_descA, 1 as the second operand --
            // swapping column `location` with itself, a no-op that left the
            // column swap half of the permutation never applied.
            n,
            A, 1, location, array_descA, 1,
            A, 1, n, array_descA, 1,
            backend_work, backend_bytes
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

    // The scratch, in the order ppotrf_workspace_size counts it (cholesky_workspace.h).  The ?potrf scratch is the
    // rest of the buffer after the info word, over the panels; the diagonal block and the first panel are carved from
    // the same place.
    const std::size_t batch_count = batch_capacity(handle, n, array_descA);
    std::size_t count[3];
    panel_counts(handle, uplo, n, array_descA, false, count);
    char* const base = static_cast<char*>(backend_work);
    char* w = base;
    T** const A_array = detail::take<T*>(w, batch_count);
    T** const B_array = detail::take<T*>(w, batch_count);
    T** const C_array = detail::take<T*>(w, batch_count);
    int* const info_buffer = detail::take<int>(w, 1);
    void* const solver = w;
    const std::size_t solver_bytes = backend_bytes - static_cast<std::size_t>(w - base);
    char* front = w;
    T* const block_diag = detail::take<T>(front, count[0]);
    T* const first = detail::take<T>(w, count[1]);
    w = std::max(w, front);
    T* const second = detail::take<T>(w, count[2]);
    const bool relay = diagonal_relay(handle, array_descA);
    void* const mirror_work = w;
    std::size_t mirror_bytes = 0, mirror_host_bytes = 0;
    ppotrf_mirror_workspace_size<T>(handle, uplo, n, array_descA, &mirror_bytes, &mirror_host_bytes);
    w += mirror_bytes;
    assert(static_cast<std::size_t>(w - base) <= backend_bytes);
    T* const block_row = uplo == 'U' ? first : second;
    T* const block_col = uplo == 'U' ? second : first;
    char* hw = static_cast<char*>(host_work);
    T** const h_A_array = detail::take<T*>(hw, batch_count);
    T** const h_B_array = detail::take<T*>(hw, batch_count);
    T** const h_C_array = detail::take<T*>(hw, batch_count);
    void* const mirror_host = hw;
    assert(static_cast<std::size_t>(hw - static_cast<char*>(host_work)) + mirror_host_bytes <= host_bytes);
    DDLA_CHECK(ddlaMemset(handle,info_buffer,0,sizeof(int)));
    int owner_row,owner_col,mm_row_start,mm_col_start,nb_real;
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
                    lapack_impl::potrf_run<Backend,T>(ddla_handle, uplo, nb_real - 1, A + mm_row_start + std::size_t(mm_col_start) * lldA, lldA, info_buffer, solver, solver_bytes);
                    if(uplo == 'L'){
                        blas_impl::trsm_run<Backend, T>(
                            ddla_handle, side_char, uplo, 'C', 'N',
                            1, nb_real - 1, (T)1.0,
                            A + mm_row_start + std::size_t(mm_col_start) * lldA, lldA,
                            A + mm_row_start + nb_real - 1 + std::size_t(mm_col_start) * lldA, lldA
                        );
                        blas_impl::herk_run<Backend, T>(
                            ddla_handle, uplo, 'N',
                            1, nb_real - 1,
                            -1.0, A + mm_row_start + nb_real - 1 + std::size_t(mm_col_start) * lldA, lldA,
                            1.0, A + mm_row_start + nb_real - 1 + std::size_t(mm_col_start + nb_real - 1) * lldA, lldA
                        );
                    }else{
                        blas_impl::trsm_run<Backend, T>(
                            ddla_handle, side_char, uplo, 'C', 'N',
                            nb_real - 1, 1, (T)1.0,
                            A + mm_row_start + std::size_t(mm_col_start) * lldA, lldA,
                            A + mm_row_start + std::size_t(mm_col_start + nb_real - 1) * lldA, lldA
                        );
                        blas_impl::herk_run<Backend, T>(
                            ddla_handle, uplo, 'C',
                            1, nb_real - 1,
                            -1.0, A + mm_row_start + std::size_t(mm_col_start + nb_real - 1) * lldA, lldA,
                            1.0, A + mm_row_start + nb_real - 1 + std::size_t(mm_col_start + nb_real - 1) * lldA, lldA
                        );
                    }
                }
                T last_value;
                DDLA_CHECK(ddlaMemcpy(ddla_handle, &last_value,
                                      A + mm_row_start + nb_real - 1 + std::size_t(mm_col_start + nb_real - 1) * lldA,
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
                                      A + mm_row_start + nb_real - 1 + std::size_t(mm_col_start + nb_real - 1) * lldA,
                                      &last_value, sizeof(T)));
            }else
                lapack_impl::potrf_run<Backend,T>(ddla_handle, uplo, nb_real, A + mm_row_start + std::size_t(mm_col_start) * lldA, lldA, info_buffer, solver, solver_bytes);
            DDLA_CHECK(ddlaSynchronize(ddla_handle));
            DDLA_CHECK(ddlaMemcpy(ddla_handle, &info, info_buffer, sizeof(int)));
            DDLA_CHECK(ddlaSynchronize(ddla_handle));
            DDLA_CHECK(ddlaMemcpy2D(ddla_handle, block_diag, nb_real * sizeof(T),
                                    A + mm_row_start + std::size_t(mm_col_start) * lldA, lldA * sizeof(T), nb_real * sizeof(T),
                                    nb_real));
        }
        if(n_s + nb_real == n)
            MPI_CHECK(MPI_Bcast(&is_nega, 1, MPI_CXX_BOOL, ddla_handle->rc_to_rank(owner_row, owner_col), ddla_handle->comm));
        MPI_CHECK(MPI_Bcast(&info, 1, MPI_INT, ddla_handle->rc_to_rank(owner_row, owner_col), ddla_handle->comm));
        if(info != 0){
            info = info + n_s;
            DDLA_CHECK(ddlaSynchronize(handle));
            return false;
        }
        // ---- the panel ----
        // The off-diagonal part of the block's column ('L', held by the block's process column) or of its row ('U',
        // held by its process row), solved against the diagonal block.  Both triangles run the same steps with rows
        // and columns exchanged: `along` is the scope the panel's ranks share, `across` the other one.
        const bool lower = uplo == 'L';
        const bool holds_panel = lower ? mypcol == owner_col : myprow == owner_row;
        const CommScope along = lower ? CommScope::Col : CommScope::Row;
        const CommScope across = lower ? CommScope::Row : CommScope::Col;
        // The panel starts after the block in its own direction.
        if(lower ? myprow == owner_row : mypcol == owner_col)
            (lower ? mm_row_start : mm_col_start) += nb_real;
        const int length_panel = lower ? m_loc_A - mm_row_start : n_loc_A - mm_col_start;
        if(holds_panel){
            commBcast<Backend>(ddla_handle, along, block_diag, (std::size_t)nb_real * nb_real, lower ? owner_row : owner_col);
            if(length_panel > 0){
                // 'L': length x nb_real below the block; 'U': nb_real x length to its right.
                const int panel_rows = lower ? length_panel : nb_real, panel_cols = lower ? nb_real : length_panel;
                T* const panel = A + mm_row_start + std::size_t(mm_col_start) * lldA;
                blas_impl::trsm_run<Backend, T>(
                    ddla_handle, side_char, uplo, 'C', 'N',
                    panel_rows, panel_cols, (T)1.0,
                    block_diag, nb_real,
                    panel, lldA
                );
                // packed copy: its leading dimension is its row count
                DDLA_CHECK(ddlaMemcpy2D(ddla_handle, first, panel_rows * sizeof(T),
                                        panel, lldA * sizeof(T),
                                        panel_rows * sizeof(T), panel_cols));
            }
        }
        // The trailing matrix starts after the block in the other direction too.
        if(lower ? mypcol == owner_col : myprow == owner_row)
            (lower ? mm_col_start : mm_row_start) += nb_real;
        const int length_row = m_loc_A - mm_row_start;
        const int length_col = n_loc_A - mm_col_start;
        // Spread the solved panel across the grid. On matching square layouts,
        // diagonal ranks can relay the same packed data in the other direction.
        const int length_mirror = lower ? length_col : length_row;
        if(length_panel > 0){
            commBcast<Backend>(ddla_handle, across, first, (std::size_t)length_panel * nb_real, lower ? owner_col : owner_row);
        }
        if (relay) {
            if(myprow == mypcol && length_mirror > 0)
                DDLA_CHECK(ddlaMemcpy(ddla_handle, second, first, length_mirror * nb_real * sizeof(T)));
            if(length_mirror > 0)
                commBcast<Backend>(ddla_handle, along, second, (std::size_t)nb_real * length_mirror, lower ? mypcol : myprow);
        } else {
            // Transpose the solved panel into the other grid direction, then
            // broadcast it. PGEADD handles rectangular grids, shifted sources
            // and either rank order; only a block-wide panel is redistributed.
            const int end = n_s + nb_real;
            pblas::panel_run<Backend,T>(handle, !lower, 'C',
                lower ? n - end : nb_real, lower ? nb_real : n - end,
                A, lower ? end : n_s, lower ? n_s : end, array_descA, array_descA,
                second, mirror_work, mirror_bytes, mirror_host, mirror_host_bytes);
        }

        // ---- the trailing update ----
        if (!relay) {
            // Global diagonal tiles need HERK even when their process row and
            // column differ. Strictly selected tiles use GEMM; never touch the
            // opposite triangle or the parent matrix outside the logical n.
            int batches = 0;
            const int ld_left = std::max(1, length_row);
            const int ld_right = nb_real;
            for (int col = 0; col < length_col; col += nb) {
                const int gc = indx_l2g_c(array_descA, handle, mm_col_start + col);
                const int cols = std::min(nb, length_col - col);
                for (int row = 0; row < length_row; row += nb) {
                    const int gr = indx_l2g_r(array_descA, handle, mm_row_start + row);
                    const int rows = std::min(nb, length_row - row);
                    if (lower ? gr < gc : gr > gc) continue;
                    T* const target = A + mm_row_start + row + std::size_t(mm_col_start + col) * lldA;
                    T* const left = block_col + row;
                    T* const right = block_row + std::size_t(col) * nb_real;
                    if (gr == gc) {
                        T* const panel = lower ? first + row : first + std::size_t(col) * nb_real;
                        blas_impl::herk_run<Backend,T>(handle, uplo, lower ? 'N' : 'C',
                            rows, nb_real, -1.0, panel, lower ? length_row : nb_real, 1.0, target, lldA);
                    } else if (rows == nb && cols == nb) {
                        h_A_array[batches] = left;
                        h_B_array[batches] = right;
                        h_C_array[batches++] = target;
                    } else {
                        blas_impl::gemm_run<Backend,T>(handle, 'N', 'N', rows, cols, nb_real,
                            T(-1), left, ld_left, right, ld_right, T(1), target, lldA);
                    }
                }
            }
            if (batches > 0) {
                DDLA_CHECK(ddlaMemcpy(handle, A_array, h_A_array, batches * sizeof(T*)));
                DDLA_CHECK(ddlaMemcpy(handle, B_array, h_B_array, batches * sizeof(T*)));
                DDLA_CHECK(ddlaMemcpy(handle, C_array, h_C_array, batches * sizeof(T*)));
                blas_impl::gemmBatched_run<Backend,T>(handle, 'N', 'N', nb, nb, nb_real,
                    T(-1), A_array, ld_left, B_array, ld_right, T(1), C_array, lldA, batches);
            }
        } else if(myprow == mypcol){
            // A diagonal process holds diagonal blocks: one herk on the stored triangle.
            if(length_panel > 0)
                blas_impl::herk_run<Backend, T>(
                    ddla_handle, uplo, lower ? 'N' : 'C',
                    length_panel, nb_real,
                    -1.0, first, lower ? length_panel : nb_real,
                    1.0, A + mm_row_start + std::size_t(mm_col_start) * lldA, lldA
                );
        }else{
            // An off-diagonal process holds whole nb x nb tiles that lie strictly inside or strictly outside the
            // stored triangle.  One gemm over the whole trailing part would write the unused tiles, so only the
            // tiles of the stored triangle are updated: the full ones in one batched gemm, the ones of a short
            // last block row or column in one gemm each (the tiles kept there are contiguous).
            if(length_row <= 0 || length_col <= 0)
                continue;

            // C(rows, cols) -= X(rows) * X(cols)^H ('L') or X(rows)^H * X(cols) ('U'); the row factor is cut from
            // block_col and the column factor from block_row, at these offsets and leading dimensions.
            const char op_rows = lower ? 'N' : 'C', op_cols = lower ? 'C' : 'N';
            const int ld_rows = lower ? length_row : nb_real, ld_cols = lower ? length_col : nb_real;
            auto rows_at = [&](int offset){ return block_col + (lower ? std::size_t(offset) : std::size_t(offset) * nb_real); };
            auto cols_at = [&](int offset){ return block_row + (lower ? std::size_t(offset) : std::size_t(offset) * nb_real); };
            auto tile_at = [&](int row_offset, int col_offset){
                return A + mm_row_start + row_offset + std::size_t(mm_col_start + col_offset) * lldA; };
            auto global_row = [&](int offset){ return indx_l2g_r(array_descA, handle, mm_row_start + offset); };
            auto global_col = [&](int offset){ return indx_l2g_c(array_descA, handle, mm_col_start + offset); };
            // Tiles follow the global order, so the stored ones of a tile row are a prefix ('L') or a suffix ('U')
            // of the tile columns, and the other way round for a tile column.
            auto stored = [&](int g_row, int g_col){ return lower ? g_row > g_col : g_row < g_col; };

            const int row_full = length_row / nb * nb, row_remain = length_row - row_full;
            const int col_full = length_col / nb * nb, col_remain = length_col - col_full;

            if(row_remain != 0){
                // the short last tile row against every tile column
                const int g_row = global_row(row_full);
                int split = 0;   // first column offset on the other side of the diagonal
                while(split < length_col && global_col(split) < g_row) split += nb;
                split = std::min(split, length_col);
                const int col_offset = lower ? 0 : split, cols = lower ? split : length_col - split;
                if(cols > 0)
                    blas_impl::gemm_run<Backend, T>(ddla_handle, op_rows, op_cols,
                        row_remain, cols, nb_real, (T)-1.0,
                        rows_at(row_full), ld_rows,
                        cols_at(col_offset), ld_cols,
                        (T)1.0, tile_at(row_full, col_offset), lldA
                    );
            }
            if(col_remain != 0){
                // the short last tile column against the full tile rows
                const int g_col = global_col(col_full);
                int split = 0;   // first row offset on the other side of the diagonal
                while(split < row_full && global_row(split) < g_col) split += nb;
                const int row_offset = lower ? split : 0, rows = lower ? row_full - split : split;
                if(rows > 0)
                    blas_impl::gemm_run<Backend, T>(ddla_handle, op_rows, op_cols,
                        rows, col_remain, nb_real, (T)-1.0,
                        rows_at(row_offset), ld_rows,
                        cols_at(col_full), ld_cols,
                        (T)1.0, tile_at(row_offset, col_full), lldA
                    );
            }

            int i_batch_count = 0;
            for(int row_offset = 0; row_offset < row_full; row_offset += nb){
                const int g_row = global_row(row_offset);
                for(int col_offset = 0; col_offset < col_full; col_offset += nb){
                    if(!stored(g_row, global_col(col_offset)))
                        continue;
                    h_A_array[i_batch_count] = rows_at(row_offset);
                    h_B_array[i_batch_count] = cols_at(col_offset);
                    h_C_array[i_batch_count] = tile_at(row_offset, col_offset);
                    i_batch_count++;
                }
            }
            if(i_batch_count > 0){
                DDLA_CHECK(ddlaMemcpy(ddla_handle, A_array, h_A_array, i_batch_count * sizeof(T*)));
                DDLA_CHECK(ddlaMemcpy(ddla_handle, B_array, h_B_array, i_batch_count * sizeof(T*)));
                DDLA_CHECK(ddlaMemcpy(ddla_handle, C_array, h_C_array, i_batch_count * sizeof(T*)));
                blas_impl::gemmBatched_run<Backend, T>(
                    ddla_handle, op_rows, op_cols,
                    nb, nb, nb_real, (T)-1.0,
                    A_array, ld_rows,
                    B_array, ld_cols,
                    (T)1.0, C_array, lldA,
                    i_batch_count
                );
            }
        }
        DDLA_CHECK(ddlaSynchronize(ddla_handle));
    }
    DDLA_CHECK(ddlaSynchronize(handle));
    return is_nega;
}
} // namespace cholesky_impl
template <DdlaBackend Backend, typename T>
bool ppotrf(
    const DdlaHandle_t& handle, char uplo, int n,
    T* A, int ia, int ja, const int* array_descA,
    int& info, // host pointer
    bool is_head, int location
)
{
    cholesky_impl::ppotrf_check<Backend>(handle,uplo,n,A,ia,ja,array_descA,is_head,location);
    info=0;if(n==0)return false;
    std::size_t backend_bytes=0,host_bytes=0;
    cholesky_impl::ppotrf_workspace_size<T>(handle,uplo,n,A,array_descA,is_head,location,&backend_bytes,&host_bytes);
    vector<char> backend_work(handle,backend_bytes);
    std::vector<char> host_work(host_bytes);
    return cholesky_impl::ppotrf_run<Backend,T>(handle,uplo,n,A,ia,ja,array_descA,info,is_head,location,
                                                backend_work.data(),backend_bytes,host_work.data(),host_bytes);
}

#define INSTANTIATE_PPOTRF(BACKEND, TYPE)                                   \
    template bool ppotrf<BACKEND, TYPE>(const DdlaHandle_t&, char uplo, int n, \
    TYPE* A, int ia, int ja, const int* array_descA, \
    int& info, \
    bool is_head, int location)

DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_PPOTRF)

#undef INSTANTIATE_PPOTRF



#define DDLA_CHOLESKY_RUN(BACKEND,TYPE) template bool cholesky_impl::ppotrf_run<BACKEND,TYPE>( const DdlaHandle_t& handle, char uplo, int n, TYPE* A, int ia, int ja, const int* array_descA, int& info, bool is_head, int location, void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes );
template void cholesky_impl::ppotrf_check<DdlaBackend::CPU>(const DdlaHandle_t&,char,int,const void*,int,int,const int*,bool,int);
DDLA_CHOLESKY_RUN(DdlaBackend::CPU,float)
DDLA_CHOLESKY_RUN(DdlaBackend::CPU,double)
DDLA_CHOLESKY_RUN(DdlaBackend::CPU,std::complex<float>)
DDLA_CHOLESKY_RUN(DdlaBackend::CPU,std::complex<double>)
#if DDLA_HAS_GPU
template void cholesky_impl::ppotrf_check<DdlaBackend::GPU>(const DdlaHandle_t&,char,int,const void*,int,int,const int*,bool,int);
DDLA_CHOLESKY_RUN(DdlaBackend::GPU,float)
DDLA_CHOLESKY_RUN(DdlaBackend::GPU,double)
DDLA_CHOLESKY_RUN(DdlaBackend::GPU,std::complex<float>)
DDLA_CHOLESKY_RUN(DdlaBackend::GPU,std::complex<double>)
#endif
#undef DDLA_CHOLESKY_RUN


} // namespace ddla
