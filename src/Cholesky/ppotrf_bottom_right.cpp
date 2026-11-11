#include "cholesky_workspace.h"
#include "BLAS/blas_impl.h"
#include "check_agree.h"
#include "potrf_bottom_right_internal.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <complex>
#include <type_traits>
#include <vector>

#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "ddla_desc.h"
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "plan_util.h"
#include "require_backend.h"
#include "comm_traits.h"
#include "instantiate.h"

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
        blas_impl::syrk_run<Backend, T>(
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
        blas_impl::herk_run<Backend, T>(
            handle,
            uplo, upper ? 'N' : 'C',
            nb, panel_width,
            minus_one, panel, ld_panel,
            one, tile, ldd
        );
    }
}

} // namespace

namespace cholesky_impl {
template<DdlaBackend Backend>
void ppotrf_bottom_right_check(const DdlaHandle_t& h,char uplo,int n,const void* A,const int* d)
{
    detail::agree_check(h,"ppotrf_bottom_right",[&](detail::Agree&){check_desc(d,h);detail::require_backend<Backend>(h,"ppotrf_bottom_right");
        if((uplo!='L' && uplo!='U') || n<0 || n>d[DDLA_M_] || n>d[DDLA_N_] || d[DDLA_MB_]!=d[DDLA_NB_] || h->nprows_!=h->npcols_)
            throw std::invalid_argument("ppotrf_bottom_right: invalid triangle, dimensions, blocks or grid");
        const int ml=num_loc(n,d[DDLA_MB_],h->myprow_,d[DDLA_RSRC_],h->nprows_),nl=num_loc(n,d[DDLA_NB_],h->mypcol_,d[DDLA_CSRC_],h->npcols_);
        if(n && ml && nl && !A)throw std::invalid_argument("ppotrf_bottom_right: null local matrix");});
}
template<DdlaBackend Backend,class T>
void ppotrf_bottom_right_run(
    const DdlaHandle_t& handle, char uplo, int n,
    T* A, const int* descA, int& info,
    void* backend_work, std::size_t backend_bytes, void* host_work, [[maybe_unused]] std::size_t host_bytes
)
{
    if(n==0){info=0;return;}
    int nprows=handle->nprows_,npcols=handle->npcols_,myprow=handle->myprow_,mypcol=handle->mypcol_;
    info=0;
    const int nb = descA[DDLA_NB_];

    const int nprocs_dim = nprows;
    const int lld = descA[DDLA_LLD_];


    const int max_local_rows = num_loc(
        n, nb, myprow, descA[DDLA_RSRC_], nprocs_dim);
    const int max_local_cols = num_loc(
        n, nb, mypcol, descA[DDLA_CSRC_], nprocs_dim);
    assert(A != nullptr || max_local_rows == 0 || max_local_cols == 0);

    // The scratch, in the order ppotrf_bottom_right_workspace_size counts it (cholesky_workspace.h).  The diagonal
    // block and the first panel are carved from the same place; the ?potrf scratch is the rest of the buffer after
    // them, over the second panel.
    const std::size_t max_batch_count = batch_capacity(handle, n, descA);
    std::size_t count[3];
    panel_counts(handle, uplo, n, descA, true, count);
    char* const base = static_cast<char*>(backend_work);
    char* w = base;
    T** const left_array = detail::take<T*>(w, max_batch_count);
    T** const right_array = detail::take<T*>(w, max_batch_count);
    T** const target_array = detail::take<T*>(w, max_batch_count);
    int* const info_buffer = detail::take<int>(w, 1);
    char* front = w;
    T* const diag = detail::take<T>(front, count[0]);
    T* const first = detail::take<T>(w, count[1]);
    w = std::max(w, front);
    void* const solver = w;
    const std::size_t solver_bytes = backend_bytes - static_cast<std::size_t>(w - base);
    T* const second = detail::take<T>(w, count[2]);
    assert(static_cast<std::size_t>(w - base) <= backend_bytes);
    T* const row_panel = uplo == 'U' ? first : second;
    T* const col_panel = uplo == 'U' ? second : first;
    char* hw = static_cast<char*>(host_work);
    T** const h_left_array = detail::take<T*>(hw, max_batch_count);
    T** const h_right_array = detail::take<T*>(hw, max_batch_count);
    T** const h_target_array = detail::take<T*>(hw, max_batch_count);
    assert(static_cast<std::size_t>(hw - static_cast<char*>(host_work)) <= host_bytes);
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
                block_start, diag, info_buffer, block_info, handle, solver, solver_bytes
            );
        }

        MPI_CHECK(MPI_Bcast(
            &block_info, 1, MPI_INT,
            handle->rc_to_rank(owner_row, owner_col), handle->comm));
        info = block_info;
        if(info != 0){
            DDLA_CHECK(ddlaSynchronize(handle));
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
                blas_impl::trsm_run<Backend, T>(
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
                    blas_impl::trsm_run<Backend, T>(
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
            DDLA_CHECK(ddlaMemcpy(handle, left_array, h_left_array, active_pointer_bytes));
            DDLA_CHECK(ddlaMemcpy(handle, right_array, h_right_array, active_pointer_bytes));
            DDLA_CHECK(ddlaMemcpy(handle, target_array, h_target_array, active_pointer_bytes));

            const T minus_one = T(-1);
            const T one = T(1);
            blas_impl::gemmBatched_run<Backend, T>(
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
        DDLA_CHECK(ddlaSynchronize(handle));
    }

    DDLA_CHECK(ddlaSynchronize(handle));
}
} // namespace cholesky_impl
template <DdlaBackend Backend, typename T>
void ppotrf_bottom_right(
    const DdlaHandle_t& handle, char uplo, int n,
    T* A, const int* descA, int& info
)
{
    cholesky_impl::ppotrf_bottom_right_check<Backend>(handle,uplo,n,A,descA);
    info=0;if(n==0)return;
    std::size_t backend_bytes=0,host_bytes=0;
    cholesky_impl::ppotrf_bottom_right_workspace_size<T>(handle,uplo,n,A,descA,&backend_bytes,&host_bytes);
    vector<char> backend_work(handle,backend_bytes);
    std::vector<char> host_work(host_bytes);
    cholesky_impl::ppotrf_bottom_right_run<Backend,T>(handle,uplo,n,A,descA,info,
                                                      backend_work.data(),backend_bytes,host_work.data(),host_bytes);
}

#define INSTANTIATE_PPOTRF_BOTTOM_RIGHT(BACKEND, TYPE)                                   \
    template void ppotrf_bottom_right<BACKEND, TYPE>(const DdlaHandle_t&, char, int, TYPE*, const int*, int&)

DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_PPOTRF_BOTTOM_RIGHT)

#undef INSTANTIATE_PPOTRF_BOTTOM_RIGHT



#define DDLA_CHOLESKY_RUN(BACKEND,TYPE) template void cholesky_impl::ppotrf_bottom_right_run<BACKEND,TYPE>( const DdlaHandle_t& handle, char uplo, int n, TYPE* A, const int* descA, int& info, void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes );
template void cholesky_impl::ppotrf_bottom_right_check<DdlaBackend::CPU>(const DdlaHandle_t&,char,int,const void*,const int*);
DDLA_CHOLESKY_RUN(DdlaBackend::CPU,float)
DDLA_CHOLESKY_RUN(DdlaBackend::CPU,double)
DDLA_CHOLESKY_RUN(DdlaBackend::CPU,std::complex<float>)
DDLA_CHOLESKY_RUN(DdlaBackend::CPU,std::complex<double>)
#if DDLA_HAS_GPU
template void cholesky_impl::ppotrf_bottom_right_check<DdlaBackend::GPU>(const DdlaHandle_t&,char,int,const void*,const int*);
DDLA_CHOLESKY_RUN(DdlaBackend::GPU,float)
DDLA_CHOLESKY_RUN(DdlaBackend::GPU,double)
DDLA_CHOLESKY_RUN(DdlaBackend::GPU,std::complex<float>)
DDLA_CHOLESKY_RUN(DdlaBackend::GPU,std::complex<double>)
#endif
#undef DDLA_CHOLESKY_RUN
} // namespace ddla
