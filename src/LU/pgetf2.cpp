#include "BLAS/blas_impl.h"
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "ddla_desc.h"
#include <cassert>
#include <algorithm>
#include <complex>
#include <cstddef>
#include <vector>
#include <array>
#include <cstring>
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "require_backend.h"
#include "comm_traits.h"
#include "lu_panel.h"
#include "pgetrf_impl.h"
#include "row_swap.h"
#include "check_agree.h"
#include "plan_util.h"

#include "pgetf2_kernels.h"
#include "instantiate.h"

namespace ddla{

namespace {

struct MaxLoc {
    double value;
    int index;
};

// The pivot metric of ?amax: |x| for real types, |re| + |im| for complex ones (what the vendor ?amax compares).
inline double pivot_metric(float v) { return std::abs(v); }
inline double pivot_metric(double v) { return std::abs(v); }
inline double pivot_metric(const std::complex<float>& v) { return std::abs(v.real()) + std::abs(v.imag()); }
inline double pivot_metric(const std::complex<double>& v) { return std::abs(v.real()) + std::abs(v.imag()); }

// The two per-column steps of the panel factorization.  A GPU handle runs the fused kernels of pgetf2_kernels.cpp (one
// launch and one host round trip for the pivot search, one launch for the scale and the rank-1 update): replacing them by
// iamax + scal + geru costs two extra synchronisations and launches per column, +11..24 % on a pgesv (measured 2026-10-20).
// A CPU handle has no launch latency to save and calls the BLAS routines directly.

// Local pivot of one column: metric (|x| or |re|+|im|), zero-based index into the column, and the element.
template <DdlaBackend Backend, typename T>
void find_local_pivot(const DdlaHandle_t& handle, const T* column, int length, void* d_workspace,
                      double& metric, int& index, T& value)
{
    if constexpr (Backend == DdlaBackend::GPU) {
        detail::pgetf2_find_local_pivot(handle, column, length, d_workspace, metric, index, value);
    } else {
        (void)d_workspace;
        int one_based = 0;
        blas_impl::iamax_run<Backend, T>(handle, length, column, 1, one_based);
        index = one_based - 1;
        DDLA_CHECK(ddlaMemcpy(handle, &value, column + index, sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(handle));
        metric = pivot_metric(value);
    }
}

// column := column * inverse_pivot, then trailing -= column * pivot_row^T for the length_col columns to the right.
template <DdlaBackend Backend, typename T>
void scale_update(const DdlaHandle_t& handle, int length_row, int length_col, const T& inverse_pivot,
                  T* column, const T* pivot_row, T* trailing, int lld)
{
    if constexpr (Backend == DdlaBackend::GPU) {
        detail::pgetf2_scale_update(handle, length_row, length_col, inverse_pivot, column, pivot_row, trailing, lld);
    } else {
        if (length_row <= 0) return;
        blas_impl::scal_run<Backend, T>(handle, length_row, inverse_pivot, column, 1);
        if (length_col > 0) {
            blas_impl::geru_run<Backend, T>(handle, length_row, length_col, T(-1), column, 1, pivot_row, 1, trailing, lld);
        }
    }
}

} // namespace

// Size of the in-place method (lu_panel.h): [ temp | peer ], then the pivot search's result slot on a GPU handle.
template <typename T>
void lu_impl::inplace_workspace_size(const DdlaHandle_t& h, int swap_count, std::size_t* backend_bytes)
{
    *backend_bytes = 2*detail::padded_bytes<T>(std::size_t(std::max(1,swap_count)));
#if DDLA_HAS_GPU
    if (h->backend == DdlaBackend::GPU) *backend_bytes += detail::pgetf2_pivot_workspace_size<T>();
#else
    (void)h;
#endif
}

template <DdlaBackend Backend, typename T>
void lu_impl::factor_inplace(const DdlaHandle_t& h, int m, int n, int width, T* A,
                             int ia, int ja, const int* desc, int* ipiv, int& info, int swap_ja,
                             void* backend_work, [[maybe_unused]] std::size_t backend_bytes)
{
    info = 0;
    if (width == 0) return;
    int nprows, npcols, myprow, mypcol;
    ddlaGetGridDims(h, nprows, npcols);
    ddlaGetGridCoords(h, myprow, mypcol);
    const int nb = desc[DDLA_MB_], ld = desc[DDLA_LLD_];
    const int row0 = ia-1, col0 = ja-1;
    assert(width <= std::min({m,n,nb-col0%nb}));
    const int row_end = num_loc(row0+m, nb, myprow, desc[DDLA_RSRC_], nprows);
    const int swap_begin = num_loc(swap_ja-1, nb, mypcol, desc[DDLA_CSRC_], npcols);
    const int swap_end = num_loc(col0+n, nb, mypcol, desc[DDLA_CSRC_], npcols);
    const int swap_count = swap_end-swap_begin;
    const int jl = indx_g2l_c(desc,h,col0);
    const int owner_col = indxg2p(col0,nb,desc[DDLA_CSRC_],npcols);
    // The scratch, in the order inplace_workspace_size counts it: the swap row segments (temp also carries the row
    // of U broadcast below, at most width - 1 <= swap_count entries on the ranks of the panel's process column),
    // then the pivot search's result slot, the rest.
    char* w = static_cast<char*>(backend_work);
    T* const temp = detail::take<T>(w, std::size_t(std::max(1,swap_count)));
    T* const peer = detail::take<T>(w, std::size_t(std::max(1,swap_count)));
    void* const pivot_workspace = w;
    assert(std::size_t(w-static_cast<char*>(backend_work)) <= backend_bytes);
    assert(jl < 0 || width <= std::max(1,swap_count));
    for (int k=0; k<width; ++k) {
        const int owner_row = indxg2p(row0+k,nb,desc[DDLA_RSRC_],nprows);
        const int i = num_loc(row0+k,nb,myprow,desc[DDLA_RSRC_],nprows);
        const int j = jl >= 0 ? jl+k : 0;
        int max_row = 0;
        T max_value{};
        std::array<unsigned char, sizeof(int) + sizeof(T)> pivot_message{};
        if (jl >= 0) {
            MaxLoc local{-1.0,-1}, global{-1.0,-1};
            T local_value{};
            if (i < row_end) {
                int index = 0;
                find_local_pivot<Backend,T>(h,A+i+std::size_t(j)*ld,row_end-i,
                    pivot_workspace,local.value,index,local_value);
                local.index = indx_l2g_r(desc,h,i+index);
            }
            MPI_CHECK(MPI_Allreduce(&local,&global,1,MPI_DOUBLE_INT,MPI_MAXLOC,h->col_comm));
            max_row = global.index;
            const int max_prow = indxg2p(global.index,nb,desc[DDLA_RSRC_],nprows);
            if (myprow == max_prow) max_value = local_value;
            MPI_CHECK(MPI_Bcast(&max_value,static_cast<int>(sizeof(T)),MPI_BYTE,max_prow,h->col_comm));
            std::memcpy(pivot_message.data(), &max_row, sizeof(max_row));
            std::memcpy(pivot_message.data() + sizeof(max_row), &max_value, sizeof(max_value));
        }
        MPI_CHECK(MPI_Bcast(pivot_message.data(),static_cast<int>(pivot_message.size()),MPI_BYTE,owner_col,h->row_comm));
        std::memcpy(&max_row, pivot_message.data(), sizeof(max_row));
        std::memcpy(&max_value, pivot_message.data() + sizeof(max_row), sizeof(max_value));
        if (myprow == owner_row) ipiv[i] = max_row+1;
        if (swap_count > 0)
            detail::swap_rows<Backend,T>(h, A, desc, row0+k, swap_begin,
                                         A, desc, max_row, swap_begin,
                                         swap_count, temp, peer);
        if (std::abs(max_value) == 0.0) { info=k+1; break; }
        if (jl >= 0) {
            const int columns = width-k-1;
            if (myprow == owner_row && columns > 0)
                DDLA_CHECK(ddlaMemcpy2D(h,temp,sizeof(T),A+i+std::size_t(j+1)*ld,
                    std::size_t(ld)*sizeof(T),sizeof(T),columns));
            if (columns > 0) commBcast<Backend>(h,CommScope::Col,temp,std::size_t(columns),owner_row);
            const int row_after = myprow == owner_row ? i+1 : i;
            if (row_after < row_end) {
                T* column = A+row_after+std::size_t(j)*ld;
                scale_update<Backend,T>(h,row_end-row_after,columns,T(1.0)/max_value,
                    column,temp,columns > 0 ? column+ld : column,ld);
            }
        }
    }
    if (info != 0) DDLA_CHECK(ddlaSynchronize(h));
}

namespace lu_impl {

// Component 1: the checks of the public entry.  Collective (detail::agree_check): a failure on any rank throws on
// every rank, and the DDLA_LU_PANEL selector must be the same on every rank.  Returns the agreed selector.
template <DdlaBackend Backend>
bool pgetf2_check(const DdlaHandle_t& handle, int m, int n, int width, const void* A, int ia, int ja,
                  const int* desc, const int* ipiv)
{
    return detail::agree_check(handle, "pgetf2", [&](detail::Agree& agree) -> bool {
        detail::require_backend<Backend>(handle, "pgetf2");
        check_desc(desc, handle);
        check_region("pgetf2", handle, m, n, A, ia, ja, desc, ipiv, width < 0 ? -2 : width, true);
        const bool gather = gather_mode(handle);
        agree.mode("DDLA_LU_PANEL", gather ? 1 : 0);
        return gather;
    });
}

// Component 2: backend bytes of one region that the row of U broadcast between the inner subpanels (width x
// inner_width entries) and the subpanels' method -- the in-place method, or the gathered one of every subpanel --
// use in turn: a subpanel's method returns before the row of U is copied out, and the update that reads it is
// enqueued before the next method starts.  Host bytes of the gathered method.
template <typename T>
void pgetf2_workspace_size(const DdlaHandle_t& handle, bool gather, int m, int n, int width, T* A, int ia, int ja,
                           const int* desc, int swap_ja, std::size_t* backend_bytes, std::size_t* host_bytes)
{
    *backend_bytes = *host_bytes = 0;
    if (width <= 0) return;
    const int nb = desc[DDLA_MB_];
    std::size_t method = 0;
    if (!gather) inplace_workspace_size<T>(handle, swap_columns(handle, desc, ja, n, swap_ja), &method);
    for (int k = 0; gather && k < width; ) {
        const int b = subpanel_width(nb, ia-1, width, k);
        std::size_t backend = 0, host = 0;
        gather_workspace_size<T>(handle, m-k, n-k, b, A, ia+k, ja+k, desc, swap_ja, &backend, &host);
        method = std::max(method, backend);
        *host_bytes = std::max(*host_bytes, host);
        k += b;
    }
    *backend_bytes = std::max(detail::padded_bytes<T>(std::size_t(width)*inner_width(nb)), method);
}

// Component 3: the blocked width factorization.  It dispatches each inner subpanel to panel_gather (gather) or
// factor_inplace (in place) according to the agreed selector, with BLAS-3 trsm/bcast/gemm updates of the remaining
// panel columns in between.  The subpanel width min(32, nb/2) bounds the root's serial work in the gather mode; a
// panel no wider than that degenerates to a single subpanel with no inner update.  The nb row-boundary term keeps
// every subpanel's diagonal block inside one process row so the trsm below holds it on a single rank.  swap_ja
// retains the first selected column of the enclosing call: a driver that factors later block columns still pivots
// the preceding L columns.  No validation, no environment reads, no allocation.
template <DdlaBackend Backend, typename T>
void pgetf2_run(const DdlaHandle_t& handle, bool gather, int m, int n, int width, T* A, int ia, int ja,
                const int* desc, int* ipiv, int& info, int swap_ja,
                void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes)
{
    try {
        info = 0;
        if (width == 0) return;
        int nprows, npcols, myprow, mypcol;
        ddlaGetGridDims(handle, nprows, npcols);
        ddlaGetGridCoords(handle, myprow, mypcol);
        const int nb = desc[DDLA_MB_], ld = desc[DDLA_LLD_];
        const int row0 = ia-1, col0 = ja-1;
        const int row_end = num_loc(row0+m, nb, myprow, desc[DDLA_RSRC_], nprows);
        const int col_end = num_loc(col0+width, nb, mypcol, desc[DDLA_CSRC_], npcols);
        const int panel_col = indx_g2l_c(desc, handle, col0);
        // The row of U starts the region the subpanels' method also starts from (pgetf2_workspace_size).
        assert(std::size_t(width)*inner_width(nb)*sizeof(T) <= backend_bytes);
        T* const temp_U = static_cast<T*>(backend_work);
        for (int k = 0; k < width; ) {
            const int b = subpanel_width(nb, row0, width, k);
            const int r = row0+k, c = col0+k;
            const int il = indx_g2l_r(desc, handle, r), jl = indx_g2l_c(desc, handle, c);
            if (gather)
                panel_gather<Backend,T>(handle, m-k, n-k, b, A, ia+k, ja+k, desc, ipiv, info, swap_ja,
                                        backend_work, backend_bytes, host_work, host_bytes);
            else
                factor_inplace<Backend,T>(handle, m-k, n-k, b, A, ia+k, ja+k, desc, ipiv, info, swap_ja,
                                          backend_work, backend_bytes);
            if (info != 0) {
                if (info > 0) info += k;
                return;
            }
            const int row_after = num_loc(r+b, nb, myprow, desc[DDLA_RSRC_], nprows);
            const int col_after = num_loc(c+b, nb, mypcol, desc[DDLA_CSRC_], npcols);
            if (panel_col >= 0 && col_after < col_end) {
                const int pr = indxg2p(r, nb, desc[DDLA_RSRC_], nprows);
                if (il >= 0) {
                    blas_impl::trsm_run<Backend,T>(handle, 'L', 'L', 'N', 'U', b, col_end-col_after, T(1),
                        A+il+std::size_t(jl)*ld, ld, A+il+std::size_t(col_after)*ld, ld);
                    DDLA_CHECK(ddlaMemcpy2D(handle, temp_U, std::size_t(b)*sizeof(T),
                        A+il+std::size_t(col_after)*ld, std::size_t(ld)*sizeof(T),
                        std::size_t(b)*sizeof(T), col_end-col_after));
                }
                commBcast<Backend>(handle, CommScope::Col, temp_U, std::size_t(b)*(col_end-col_after), pr);
            }
            if (panel_col >= 0 && row_after < row_end && col_after < col_end)
                blas_impl::gemm_run<Backend,T>(handle, 'N', 'N', row_end-row_after, col_end-col_after, b,
                    -1.0, A+row_after+std::size_t(jl)*ld, ld, temp_U, b, 1.0,
                    A+row_after+std::size_t(col_after)*ld, ld);
            k += b;
        }
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(handle) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(handle); } catch (...) {}
        throw;
    }
}

} // namespace lu_impl

// The public (private-header) entry: check -> (empty return) -> workspace size -> allocate -> run.
template <DdlaBackend Backend, typename T>
void pgetf2(const DdlaHandle_t& handle, int m, int n, int width, T* A, int ia, int ja,
            const int* desc, int* ipiv, int& info)
{
    // 1. parameter check (collective; returns the agreed DDLA_LU_PANEL selector).
    const bool gather = lu_impl::pgetf2_check<Backend>(handle, m, n, width, A, ia, ja, desc, ipiv);
    // No-op: an empty panel accesses no data.
    info = 0;
    if (width == 0) return;
    // 2. the scratch of this panel call.
    std::size_t backend_bytes = 0, host_bytes = 0;
    lu_impl::pgetf2_workspace_size<T>(handle, gather, m, n, width, A, ia, ja, desc, ja, &backend_bytes, &host_bytes);
    vector<char> backend_work(handle, backend_bytes);
    std::vector<char> host_work(host_bytes);
    // 3. the factorization.
    lu_impl::pgetf2_run<Backend,T>(handle, gather, m, n, width, A, ia, ja, desc, ipiv, info, ja,
                                   backend_work.data(), backend_bytes, host_work.data(), host_bytes);
}

#define INSTANTIATE_PGETF2(BACKEND, TYPE) \
    template void pgetf2<BACKEND,TYPE>(const DdlaHandle_t&, int, int, int, TYPE*, int, int, const int*, int*, int&); \
    template void lu_impl::pgetf2_run<BACKEND,TYPE>(const DdlaHandle_t&, bool, int, int, int, TYPE*, int, int, \
        const int*, int*, int&, int, void*, std::size_t, void*, std::size_t); \
    template void lu_impl::factor_inplace<BACKEND,TYPE>(const DdlaHandle_t&, int, int, int, TYPE*, int, int, \
        const int*, int*, int&, int, void*, std::size_t)

#define INSTANTIATE_PGETF2_SIZE(TYPE) \
    template void lu_impl::pgetf2_workspace_size<TYPE>(const DdlaHandle_t&, bool, int, int, int, TYPE*, int, int, \
        const int*, int, std::size_t*, std::size_t*); \
    template void lu_impl::inplace_workspace_size<TYPE>(const DdlaHandle_t&, int, std::size_t*)

#define INSTANTIATE_PGETF2_CHECK(BACKEND) \
    template bool lu_impl::pgetf2_check<BACKEND>(const DdlaHandle_t&, int, int, int, const void*, int, int, \
        const int*, const int*)

DDLA_FOR_EACH_BACKEND(INSTANTIATE_PGETF2_CHECK)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_PGETF2)
DDLA_FOR_EACH_TYPE(INSTANTIATE_PGETF2_SIZE)

#undef INSTANTIATE_PGETF2_SIZE
#undef INSTANTIATE_PGETF2_CHECK
#undef INSTANTIATE_PGETF2

} // namespace ddla
