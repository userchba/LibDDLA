#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <cassert>
#include <algorithm>
#include <complex>
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "require_backend.h"
#include "comm_traits.h"
#include "lu_panel.h"
#include "row_swap.h"

#include "pgetf2_kernels.h"

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
        iamax<Backend, T>(handle, length, column, 1, one_based);
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
        scal<Backend, T>(handle, length_row, inverse_pivot, column, 1);
        if (length_col > 0) {
            geru<Backend, T>(handle, length_row, length_col, T(-1), column, 1, pivot_row, 1, trailing, lld);
        }
    }
}

template <DdlaBackend Backend, typename T>
std::size_t pivot_workspace_bytes()
{
    if constexpr (Backend == DdlaBackend::GPU) return detail::pgetf2_pivot_workspace_size<T>();
    else return 0;
}

template <typename T>
struct PivotBroadcast {
    int max_row = 0;
    int max_prow = 0;
    T max_value = T{};
};

} // namespace

template <DdlaBackend Backend, typename T>
void lu_impl::factor_inplace(const DdlaHandle_t& h, int m, int n, int width, T* A,
                             int ia, int ja, const int* desc, int* ipiv, int& info, int swap_ja)
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
    T *temp = nullptr, *peer = nullptr;
    DDLA_CHECK(ddlaMalloc(h, &temp, sizeof(T)*std::size_t(std::max(1,swap_count))));
    DDLA_CHECK(ddlaMalloc(h, &peer, sizeof(T)*std::size_t(std::max(1,swap_count))));
    void* pivot_workspace = nullptr;
    DDLA_CHECK(ddlaMalloc(h, &pivot_workspace, pivot_workspace_bytes<Backend,T>()));
    for (int k=0; k<width; ++k) {
        const int owner_row = indxg2p(row0+k,nb,desc[DDLA_RSRC_],nprows);
        const int i = num_loc(row0+k,nb,myprow,desc[DDLA_RSRC_],nprows);
        const int j = jl >= 0 ? jl+k : 0;
        PivotBroadcast<T> pivot;
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
            pivot.max_row = global.index;
            pivot.max_prow = indxg2p(global.index,nb,desc[DDLA_RSRC_],nprows);
            if (myprow == pivot.max_prow) pivot.max_value = local_value;
            MPI_CHECK(MPI_Bcast(&pivot.max_value,static_cast<int>(sizeof(T)),MPI_BYTE,pivot.max_prow,h->col_comm));
        }
        MPI_CHECK(MPI_Bcast(&pivot,static_cast<int>(sizeof(pivot)),MPI_BYTE,owner_col,h->row_comm));
        if (myprow == owner_row) ipiv[i] = pivot.max_row+1;
        if (swap_count > 0)
            detail::swap_rows<Backend,T>(h, A, desc, row0+k, swap_begin,
                                         A, desc, pivot.max_row, swap_begin,
                                         swap_count, temp, peer);
        if (std::abs(pivot.max_value) == 0.0) { info=k+1; break; }
        if (jl >= 0) {
            const int columns = width-k-1;
            if (myprow == owner_row && columns > 0)
                DDLA_CHECK(ddlaMemcpy2D(h,temp,sizeof(T),A+i+std::size_t(j+1)*ld,
                    std::size_t(ld)*sizeof(T),sizeof(T),columns));
            if (columns > 0) commBcast<Backend>(h,CommScope::Col,temp,std::size_t(columns),owner_row);
            const int row_after = myprow == owner_row ? i+1 : i;
            if (row_after < row_end) {
                T* column = A+row_after+std::size_t(j)*ld;
                scale_update<Backend,T>(h,row_end-row_after,columns,T(1.0)/pivot.max_value,
                    column,temp,columns > 0 ? column+ld : column,ld);
            }
        }
    }
    DDLA_CHECK(ddlaFree(h,temp));
    DDLA_CHECK(ddlaFree(h,peer));
    DDLA_CHECK(ddlaFree(h,pivot_workspace));
    if (info != 0) DDLA_CHECK(ddlaSynchronize(h));
}

// The blocked width factorization; it selects the panel method itself. DDLA_LU_PANEL=gather factors every inner
// subpanel on one root rank (panel_gather) and lets the trsm/bcast/gemm updates below distribute the rest of the
// panel, DDLA_LU_PANEL=inplace (default) factors the same subpanels on the distributed A (factor_inplace). The
// subpanel width min(32, nb/2) bounds the root's serial work in the gather mode; a panel no wider than that
// degenerates to a single subpanel with no inner update. The nb row-boundary term keeps every subpanel's diagonal
// block inside one process row so the trsm below holds it on a single rank. swap_ja retains the first selected
// column of the enclosing call: a driver that factors later block columns still pivots the preceding L columns.
template <DdlaBackend Backend, typename T>
void pgetf2(const DdlaHandle_t& handle, int m, int n, int width, T* A, int ia, int ja,
            const int* desc, int* ipiv, int& info, lu_impl::Workspace<T>& workspace, int swap_ja)
{
    info = 0;
    if (width == 0) return;
    int nprows, npcols, myprow, mypcol;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
    const int nb = desc[DDLA_MB_], ld = desc[DDLA_LLD_];
    const int row0 = ia-1, col0 = ja-1;
    const bool gather = lu_impl::use_gather(handle);
    const int inner = std::min(32, std::max(1, nb/2));
    const int row_end = num_loc(row0+m, nb, myprow, desc[DDLA_RSRC_], nprows);
    const int col_end = num_loc(col0+width, nb, mypcol, desc[DDLA_CSRC_], npcols);
    const int panel_col = indx_g2l_c(desc, handle, col0);
    T* temp_U = nullptr;
    DDLA_CHECK(ddlaMalloc(handle, &temp_U, sizeof(T)*std::size_t(width)*inner));
    for (int k = 0; k < width; ) {
        const int b = std::min({inner, width-k, nb-(row0+k)%nb});
        const int r = row0+k, c = col0+k;
        const int il = indx_g2l_r(desc, handle, r), jl = indx_g2l_c(desc, handle, c);
        if (gather)
            lu_impl::panel_gather<Backend,T>(handle, m-k, n-k, b, A, ia+k, ja+k,
                                             desc, ipiv, info, workspace, swap_ja);
        else
            lu_impl::factor_inplace<Backend,T>(handle, m-k, n-k, b, A, ia+k, ja+k,
                                               desc, ipiv, info, swap_ja);
        if (info != 0) {
            if (info > 0) info += k;
            DDLA_CHECK(ddlaFree(handle, temp_U));
            return;
        }
        const int row_after = num_loc(r+b, nb, myprow, desc[DDLA_RSRC_], nprows);
        const int col_after = num_loc(c+b, nb, mypcol, desc[DDLA_CSRC_], npcols);
        if (panel_col >= 0 && col_after < col_end) {
            const int pr = indxg2p(r, nb, desc[DDLA_RSRC_], nprows);
            if (il >= 0) {
                trsm<Backend,T>(handle, 'L', 'L', 'N', 'U', b, col_end-col_after, T(1),
                    A+il+std::size_t(jl)*ld, ld, A+il+std::size_t(col_after)*ld, ld);
                DDLA_CHECK(ddlaMemcpy2D(handle, temp_U, std::size_t(b)*sizeof(T),
                    A+il+std::size_t(col_after)*ld, std::size_t(ld)*sizeof(T),
                    std::size_t(b)*sizeof(T), col_end-col_after));
            }
            commBcast<Backend>(handle, CommScope::Col, temp_U, std::size_t(b)*(col_end-col_after), pr);
        }
        if (panel_col >= 0 && row_after < row_end && col_after < col_end)
            gemm<Backend,T>(handle, 'N', 'N', row_end-row_after, col_end-col_after, b,
                -1.0, A+row_after+std::size_t(jl)*ld, ld, temp_U, b, 1.0,
                A+row_after+std::size_t(col_after)*ld, ld);
        k += b;
    }
    DDLA_CHECK(ddlaFree(handle, temp_U));
}

template <DdlaBackend Backend, typename T>
void pgetf2(const DdlaHandle_t& handle, int m, int n, int width, T* A, int ia, int ja,
            const int* desc, int* ipiv, int& info)
{
    detail::require_backend<Backend>(handle,"pgetf2");
    lu_impl::validate(handle,m,n,A,ia,ja,desc,ipiv,width < 0 ? -2 : width);
    lu_impl::Workspace<T> workspace(handle);
    pgetf2<Backend,T>(handle,m,n,width,A,ia,ja,desc,ipiv,info,workspace,ja);
}

#define INSTANTIATE_PGETF2(BACKEND, TYPE) \
    template void pgetf2<BACKEND,TYPE>(const DdlaHandle_t&, int, int, int, TYPE*, int, int, const int*, int*, int&); \
    template void pgetf2<BACKEND,TYPE>(const DdlaHandle_t&, int, int, int, TYPE*, int, int, const int*, int*, int&, lu_impl::Workspace<TYPE>&, int)

#define INSTANTIATE_PRIVATE(B,T) template void lu_impl::factor_inplace<B,T>(const DdlaHandle_t&, int, int, int, T*, int, int, const int*, int*, int&, int)
#if DDLA_HAS_CPU
INSTANTIATE_PRIVATE(DdlaBackend::CPU, float);
INSTANTIATE_PRIVATE(DdlaBackend::CPU, double);
INSTANTIATE_PRIVATE(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PRIVATE(DdlaBackend::CPU, std::complex<double>);
#endif
#if DDLA_HAS_GPU
INSTANTIATE_PRIVATE(DdlaBackend::GPU, float);
INSTANTIATE_PRIVATE(DdlaBackend::GPU, double);
INSTANTIATE_PRIVATE(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PRIVATE(DdlaBackend::GPU, std::complex<double>);
#endif
#undef INSTANTIATE_PRIVATE

#if DDLA_HAS_CPU
INSTANTIATE_PGETF2(DdlaBackend::CPU, float);
INSTANTIATE_PGETF2(DdlaBackend::CPU, double);
INSTANTIATE_PGETF2(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PGETF2(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_PGETF2(DdlaBackend::GPU, float);
INSTANTIATE_PGETF2(DdlaBackend::GPU, double);
INSTANTIATE_PGETF2(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PGETF2(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_PGETF2


} // namespace ddla
