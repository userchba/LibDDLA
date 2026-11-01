#include "lu_panel.h"
#include "ddla_connector.h"
#include "ddla_desc.h"
#include "ddla_stream.h"
#include "lapack_impl.h"
#include "row_swap.h"
#include "matrix_transfer.h"
#include "plan_util.h"

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ddla::lu_impl {

// ---------------------------------------------------------------------------
// Component 1 helpers: local argument checks.
// ---------------------------------------------------------------------------

bool gather_mode(const DdlaHandle_t& h)
{
    const char* value = std::getenv("DDLA_LU_PANEL");
    if (value && *value && std::strcmp(value, "gather") && std::strcmp(value, "inplace")) {
        if (h->myid == 0)
            std::fprintf(stderr, "LU: ignoring DDLA_LU_PANEL=%s (need 'gather' or 'inplace')\n", value);
    }
    return value && std::strcmp(value, "gather") == 0;
}

void check_region(const char* name, const DdlaHandle_t& h, int m, int n, const void* A, int ia, int ja,
                  const int* desc, const int* ipiv, int width, bool require_ipiv)
{
    int nprows,npcols,myprow,mypcol;
    ddlaGetGridDims(h,nprows,npcols);
    ddlaGetGridCoords(h,myprow,mypcol);
    int bad = m < 0 || n < 0 || ia < 1 || ja < 1;
    if (ia > 0 && ja > 0) {
        ia--;
        ja--;
        bad |= ia > desc[DDLA_M_] || ja > desc[DDLA_N_] ||
               static_cast<long long>(ia)+m > desc[DDLA_M_] ||
               static_cast<long long>(ja)+n > desc[DDLA_N_];
    }
    bad |= desc[DDLA_MB_] != desc[DDLA_NB_];
    const int steps = width < 0 ? std::min(m,n) : width;
    bad |= width < -1 || steps < 0 || steps > std::min(m,n);
    if (!bad && width >= 0)
        bad |= width > desc[DDLA_NB_]-ja%desc[DDLA_NB_];
    if (!bad && steps > 0) {
        const int nr = num_loc(ia+m,desc[DDLA_MB_],myprow,desc[DDLA_RSRC_],nprows)
                     -num_loc(ia,desc[DDLA_MB_],myprow,desc[DDLA_RSRC_],nprows);
        const int nc = num_loc(ja+n,desc[DDLA_NB_],mypcol,desc[DDLA_CSRC_],npcols)
                     -num_loc(ja,desc[DDLA_NB_],mypcol,desc[DDLA_CSRC_],npcols);
        const int npiv = num_loc(ia+steps,desc[DDLA_MB_],myprow,desc[DDLA_RSRC_],nprows)
                       -num_loc(ia,desc[DDLA_MB_],myprow,desc[DDLA_RSRC_],nprows);
        bad |= (nr > 0 && nc > 0 && A == nullptr) || (require_ipiv && npiv > 0 && ipiv == nullptr);
    }
    if (bad)
        throw std::invalid_argument(std::string(name) +
                                    ": invalid submatrix offsets, dimensions, panel width, or local storage");
}

void check_solve(const char* name, char side, char trans, int n, int nrhs, const int* descA, const int* descB)
{
    const std::string who(name);
    if (side != 'L' && side != 'R')
        throw std::invalid_argument(who + ": side must be 'L' or 'R'");
    if (trans != 'N' && trans != 'T' && trans != 'C')
        throw std::invalid_argument(who + ": trans must be 'N', 'T' or 'C'");
    if (n < 0 || nrhs < 0)
        throw std::invalid_argument(who + ": negative dimension");
    // side='L': B is n x nrhs, solves op(A)*X = B; side='R': B is nrhs x n, solves X*op(A) = B.  The descriptors
    // may describe matrices larger than the logical sub-matrix (leading block); only the leading n columns of A
    // and the corresponding leading block of B are referenced.
    const int b_rows = side == 'L' ? n : nrhs;
    const int b_cols = side == 'L' ? nrhs : n;
    if (n > descA[DDLA_M_] || n > descA[DDLA_N_] || b_rows > descB[DDLA_M_] || b_cols > descB[DDLA_N_])
        throw std::invalid_argument(who + ": n x nrhs exceeds the descriptors");
    if (descA[DDLA_MB_] != descA[DDLA_NB_])
        throw std::invalid_argument(who + ": A needs square blocks");
    if (descA[DDLA_MB_] != descB[DDLA_MB_])
        throw std::invalid_argument(who + ": A and B need the same row block size");
}

// ---------------------------------------------------------------------------
// Component 2: the gathered method's scratch for one subpanel.
// ---------------------------------------------------------------------------
template <typename T>
void gather_workspace_size(const DdlaHandle_t& h, int m, int n, int width, T* A, int ia, int ja,
                           const int* desc, int swap_ja, std::size_t* backend_bytes, std::size_t* host_bytes)
{
    ia--;
    ja--;
    int nprows, npcols, myprow, mypcol;
    ddlaGetGridDims(h,nprows,npcols);
    ddlaGetGridCoords(h,myprow,mypcol);
    const int nb = desc[DDLA_MB_], cs = desc[DDLA_CSRC_];
    const int pr = indxg2p(ia,nb,desc[DDLA_RSRC_],nprows);
    const int pc = indxg2p(ja,nb,cs,npcols);
    const int root_rank = h->rc_to_rank(pr,pc);
    const bool root = myprow == pr && mypcol == pc;
    const int local_cols = num_loc(ja+n,nb,mypcol,cs,npcols)
                         - num_loc(swap_ja-1,nb,mypcol,cs,npcols);
    const std::size_t w = std::size_t(width);
    // The pool: the packing of the gather / scatter (and the root's peer list, host), the two swap row segments,
    // and the vendor workspace of the root's local getrf of the window (m x width, leading dimension m; a CPU
    // handle's LAPACK needs none).
    std::size_t packed = 0, peers = 0, solver = 0, replay = 0;
    matrix_impl::transfer_workspace_size<T>(h,m,width,ia+1,ja+1,desc,root_rank,&packed,&peers);
    if (root) lapack_impl::getrf_workspace_size<T>(h,m,width,A,m,&solver);
    const std::size_t rows = 2*std::size_t(std::max(1,local_cols))*sizeof(T);
    inplace_workspace_size<T>(h,local_cols,&replay);
    *backend_bytes = std::max(detail::padded_bytes<T>(root ? std::size_t(m)*w : 0)
                              + detail::padded_bytes<int>(root ? w+1 : 0) + std::max({packed,rows,solver}),
                              replay);
    *host_bytes = detail::padded_bytes<int>(w+1) + peers;
}

// ---------------------------------------------------------------------------
// Component 3: the gathered subpanel.
// ---------------------------------------------------------------------------
template <DdlaBackend Backend, typename T>
void panel_gather(const DdlaHandle_t& h, int m, int n, int width, T* A, int ia, int ja,
                  const int* desc, int* ipiv, int& info, int swap_ja,
                  void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes)
{
    info = 0;
    if (width == 0) return;
    ia--;
    ja--;
    int nprows, npcols, myprow, mypcol;
    ddlaGetGridDims(h,nprows,npcols);
    ddlaGetGridCoords(h,myprow,mypcol);
    const int nb = desc[DDLA_MB_], cs = desc[DDLA_CSRC_];
    const int pr = indxg2p(ia,nb,desc[DDLA_RSRC_],nprows);
    const int pc = indxg2p(ja,nb,cs,npcols);
    const int root_rank = h->rc_to_rank(pr,pc);
    const bool owner = mypcol == pc;
    const bool root = owner && myprow == pr;
    const int first_col = num_loc(swap_ja-1,nb,mypcol,cs,npcols);
    const int last_col = num_loc(ja+n,nb,mypcol,cs,npcols);
    const int local_cols = last_col-first_col;
    const int lc = num_loc(ja,nb,mypcol,cs,npcols);
    assert(width <= std::min({m,n,nb-ja%nb}));
    const std::size_t w = std::size_t(width);
    // The scratch, in the order gather_workspace_size counts it; the pool is the rest of the region.
    char* pool = static_cast<char*>(backend_work);
    T* const window = detail::take<T>(pool, root ? std::size_t(m)*w : 0);
    int* const device_meta = detail::take<int>(pool, root ? w+1 : 0);
    const std::size_t used = std::size_t(pool-static_cast<char*>(backend_work));
    assert(used <= backend_bytes);
    const std::size_t pool_bytes = backend_bytes-used;
    char* peers = static_cast<char*>(host_work);
    int* const meta = detail::take<int>(peers, w+1);
    const std::size_t host_used = std::size_t(peers-static_cast<char*>(host_work));
    assert(host_used <= host_bytes);
    const std::size_t peer_bytes = host_bytes-host_used;
    matrix_impl::gather_run<Backend,T>(h,m,width,A,ia+1,ja+1,desc,root_rank,window,m,
                                       pool,pool_bytes,peers,peer_bytes);
    if (root) {
        lapack_impl::getrf_run<Backend,T>(h,m,width,window,m,device_meta+1,device_meta,pool,pool_bytes);
        DDLA_CHECK(ddlaMemcpy(h,meta,device_meta,(w+1)*sizeof(int)));
        DDLA_CHECK(ddlaSynchronize(h));
        if (meta[0] == 0)
            for (int k=1; k<=width; ++k) meta[k] += ia;
    }
    MPI_CHECK(MPI_Bcast(meta,width+1,MPI_INT,root_rank,h->comm));
    if (meta[0] < 0) throw std::runtime_error("LU gathered panel: local getrf rejected its arguments");
    if (meta[0] > 0) {
        // The distributed input and user pivots are untouched. Replay the
        // original early-stop algorithm on exactly this inner subpanel.
        factor_inplace<Backend,T>(h,m,n,width,A,ia+1,ja+1,desc,ipiv,info,swap_ja,backend_work,backend_bytes);
        return;
    }
    const int left = owner ? lc-first_col : local_cols;
    const int right = owner ? lc+width : last_col;
    assert(std::size_t(2)*std::size_t(std::max(1,local_cols))*sizeof(T) <= pool_bytes);
    T* const rows = reinterpret_cast<T*>(pool);
    T* const recv = rows+std::max(1,local_cols);
    for (int k=0; k<width; ++k) {
        const int r = ia+k, pivot = meta[k+1]-1;
        const int il = indx_g2l_r(desc,h,r);
        if (il >= 0) ipiv[il] = pivot+1;
        detail::swap_rows<Backend,T>(h, A, desc, r, first_col, A, desc, pivot, first_col,
                                     left, rows, recv);
        detail::swap_rows<Backend,T>(h, A, desc, r, right, A, desc, pivot, right,
                                     last_col-right, rows, recv);
    }
    matrix_impl::scatter_run<Backend,T>(h,m,width,A,ia+1,ja+1,desc,root_rank,window,m,true,
                                        pool,pool_bytes,peers,peer_bytes);
}

#define INST(B,T) template void panel_gather<B,T>(const DdlaHandle_t&, int, int, int, T*, int, int, const int*, \
    int*, int&, int, void*, std::size_t, void*, std::size_t)
#define INST_TYPES(B) INST(B,float); INST(B,double); INST(B,std::complex<float>); INST(B,std::complex<double>)
#if DDLA_HAS_CPU
INST_TYPES(DdlaBackend::CPU);
#endif
#if DDLA_HAS_GPU
INST_TYPES(DdlaBackend::GPU);
#endif
#define INST_SIZE(T) template void gather_workspace_size<T>(const DdlaHandle_t&, int, int, int, T*, int, int, \
    const int*, int, std::size_t*, std::size_t*)
INST_SIZE(float);
INST_SIZE(double);
INST_SIZE(std::complex<float>);
INST_SIZE(std::complex<double>);
#undef INST_SIZE
#undef INST_TYPES
#undef INST
} // namespace ddla::lu_impl
