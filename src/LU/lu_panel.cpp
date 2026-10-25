#include "lu_panel.h"
#include "ddla_desc.h"
#include "ddla_stream_impl.h"
#include "comm_traits.h"
#include "row_swap.h"
#include "matrix_transfer.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ddla::lu_impl {

bool use_gather(const DdlaHandle_t& h)
{
    const char* value = std::getenv("DDLA_LU_PANEL");
    int selected = value && std::strcmp(value, "gather") == 0 ? 1 : 0;
    if (value && *value && std::strcmp(value, "gather") && std::strcmp(value, "inplace")) {
        if (h->myid == 0)
            std::fprintf(stderr, "LU: ignoring DDLA_LU_PANEL=%s (need 'gather' or 'inplace')\n", value);
    }
    if (h->nprocs > 1) {
        int lo = selected, hi = selected;
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &lo, 1, MPI_INT, MPI_MIN, h->comm));
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &hi, 1, MPI_INT, MPI_MAX, h->comm));
        if (lo != hi) throw std::runtime_error("DDLA_LU_PANEL differs between the ranks");
    }
    return selected != 0;
}

void validate(const DdlaHandle_t& h, int m, int n, const void* A, int ia, int ja,
              const int* desc, const int* ipiv, int width)
{
    check_desc(desc,h);
    int nprows,npcols,myprow,mypcol;
    ddlaGetGridDims(h,nprows,npcols);
    ddlaGetGridCoords(h,myprow,mypcol);
    int bad = m < 0 || n < 0 || ia < 1 || ja < 1;
    const long long r0 = static_cast<long long>(ia)-1, c0 = static_cast<long long>(ja)-1;
    bad |= r0 > desc[DDLA_M_] || c0 > desc[DDLA_N_] ||
           r0+m > desc[DDLA_M_] || c0+n > desc[DDLA_N_];
    bad |= desc[DDLA_MB_] != desc[DDLA_NB_];
    const int steps = width < 0 ? std::min(m,n) : width;
    bad |= width < -1 || steps < 0 || steps > std::min(m,n);
    if (!bad && width >= 0)
        bad |= width > desc[DDLA_NB_]-int(c0)%desc[DDLA_NB_];
    if (!bad && steps > 0) {
        const int nr = num_loc(int(r0)+m,desc[DDLA_MB_],myprow,desc[DDLA_RSRC_],nprows)
                     -num_loc(int(r0),desc[DDLA_MB_],myprow,desc[DDLA_RSRC_],nprows);
        const int nc = num_loc(int(c0)+n,desc[DDLA_NB_],mypcol,desc[DDLA_CSRC_],npcols)
                     -num_loc(int(c0),desc[DDLA_NB_],mypcol,desc[DDLA_CSRC_],npcols);
        const int npiv = num_loc(int(r0)+steps,desc[DDLA_MB_],myprow,desc[DDLA_RSRC_],nprows)
                       -num_loc(int(r0),desc[DDLA_MB_],myprow,desc[DDLA_RSRC_],nprows);
        bad |= (nr > 0 && nc > 0 && A == nullptr) || (npiv > 0 && ipiv == nullptr);
    }
    if (h->nprocs > 1)
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,h->comm));
    if (bad) throw std::invalid_argument("LU: invalid submatrix offsets, dimensions, panel width, or local storage");
}

namespace {
template <typename T>
void grow(const DdlaHandle_t& h, T*& p, std::size_t& capacity, std::size_t count)
{
    if (count <= capacity) return;
    DDLA_CHECK(ddlaFree(h, p));
    p = nullptr;
    DDLA_CHECK(ddlaMalloc(h, &p, count * sizeof(T)));
    capacity = count;
}

} // namespace

template <typename T>
Workspace<T>::~Workspace()
{
    DDLA_CHECK(ddlaFree(handle, window));
    DDLA_CHECK(ddlaFree(handle, packed));
    DDLA_CHECK(ddlaFree(handle, rows));
    DDLA_CHECK(ddlaFree(handle, device_meta));
}

template <typename T>
void Workspace<T>::reserve(bool root, int height, int width, int local_rows, int local_cols)
{
    if (root) {
        grow(handle, window, window_cap, std::size_t(height) * width);
        grow(handle, device_meta, meta_cap, std::size_t(width) + 1);
    }
    grow(handle, packed, packed_cap, std::size_t(root ? height : local_rows) * width);
    grow(handle, rows, rows_cap, std::size_t(2) * std::max(1, local_cols));
    meta.resize(std::size_t(width) + 1);
}

template <DdlaBackend Backend, typename T>
void panel_gather(const DdlaHandle_t& h, int m, int n, int width, T* A, int ia, int ja,
                  const int* desc, int* ipiv, int& info, Workspace<T>& ws, int swap_ja)
{
    info = 0;
    if (width == 0) return;
    int nprows, npcols, myprow, mypcol;
    ddlaGetGridDims(h,nprows,npcols);
    ddlaGetGridCoords(h,myprow,mypcol);
    const int nb = desc[DDLA_MB_], rs = desc[DDLA_RSRC_], cs = desc[DDLA_CSRC_];
    const int row0 = ia-1, col0 = ja-1;
    assert(width <= std::min({m,n,nb-col0%nb}));
    const int pr = indxg2p(row0,nb,rs,nprows), pc = indxg2p(col0,nb,cs,npcols);
    const bool owner = mypcol == pc, root = owner && myprow == pr;
    const int local_rows = num_loc(row0+m,nb,myprow,rs,nprows)-num_loc(row0,nb,myprow,rs,nprows);
    const int first_col = num_loc(swap_ja-1,nb,mypcol,cs,npcols);
    const int last_col = num_loc(col0+n,nb,mypcol,cs,npcols);
    const int local_cols = last_col-first_col;
    const int lc = num_loc(col0,nb,mypcol,cs,npcols);
    ws.reserve(root,m,width,owner ? local_rows : 0,local_cols);
    const int root_rank = h->rc_to_rank(pr,pc);
    matrix_impl::gather<Backend,T>(h,m,width,A,ia,ja,desc,root_rank,ws.window,m,ws.packed,ws.packed_cap);
    if (root) {
        getrf<Backend,T>(h,m,width,ws.window,m,ws.device_meta+1,ws.device_meta);
        DDLA_CHECK(ddlaMemcpy(h,ws.meta.data(),ws.device_meta,(std::size_t(width)+1)*sizeof(int)));
        DDLA_CHECK(ddlaSynchronize(h));
        if (ws.meta[0] == 0)
            for (int k=1; k<=width; ++k) ws.meta[k] += row0;
    }
    MPI_CHECK(MPI_Bcast(ws.meta.data(),width+1,MPI_INT,root_rank,h->comm));
    if (ws.meta[0] < 0) throw std::runtime_error("LU gathered panel: local getrf rejected its arguments");
    if (ws.meta[0] > 0) {
        // The distributed input and user pivots are untouched. Replay the
        // original early-stop algorithm on exactly this inner subpanel.
        factor_inplace<Backend,T>(h,m,n,width,A,ia,ja,desc,ipiv,info,swap_ja);
        return;
    }
    const int left = owner ? lc-first_col : local_cols;
    const int right = owner ? lc+width : last_col;
    T* recv = ws.rows+std::max(1,local_cols);
    for (int k=0; k<width; ++k) {
        const int r = row0+k, pivot = ws.meta[k+1]-1;
        const int il = indx_g2l_r(desc,h,r);
        if (il >= 0) ipiv[il] = pivot+1;
        detail::swap_rows<Backend,T>(h, A, desc, r, first_col, A, desc, pivot, first_col,
                                     left, ws.rows, recv);
        detail::swap_rows<Backend,T>(h, A, desc, r, right, A, desc, pivot, right,
                                     last_col-right, ws.rows, recv);
    }
    matrix_impl::scatter<Backend,T>(h,m,width,A,ia,ja,desc,root_rank,ws.window,m,ws.packed,ws.packed_cap);
}

#define INST(B,T) template void panel_gather<B,T>(const DdlaHandle_t&, int, int, int, T*, int, int, const int*, int*, int&, Workspace<T>&, int)
#define INST_TYPES(B) INST(B,float); INST(B,double); INST(B,std::complex<float>); INST(B,std::complex<double>)
#if DDLA_HAS_CPU
INST_TYPES(DdlaBackend::CPU);
#endif
#if DDLA_HAS_GPU
INST_TYPES(DdlaBackend::GPU);
#endif
template struct Workspace<float>;
template struct Workspace<double>;
template struct Workspace<std::complex<float>>;
template struct Workspace<std::complex<double>>;
#undef INST_TYPES
#undef INST
} // namespace ddla::lu_impl
