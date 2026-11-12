#include "matrix_transfer.h"
#include "ddla_desc.h"
#include "ddla_stream.h"
#include "comm_traits.h"
#include "check_agree.h"
#include "plan_util.h"
#include "row_share.h"
#include "instantiate.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace ddla::matrix_impl {
#if DDLA_HAS_GPU
// GPU implementation detail; callers use convert_rows<B,T> from the header.
template <typename T>
void convert_rows_gpu(const DdlaHandle_t& h, const typename real_type<T>::type* src,
                      int lds, T* dst, int ldd, int rows, int cols, bool transpose);
#endif
namespace {

// rows * cols elements, rejecting a size whose byte count overflows size_t.
std::size_t checked_count(int rows, int cols, std::size_t scalar_bytes)
{
    const std::size_t r = rows, c = cols;
    if (c && r > std::numeric_limits<std::size_t>::max() / c / scalar_bytes)
        throw std::length_error("matrix transfer: storage size overflow");
    return r * c;
}

std::size_t block_size(const Peer& p)
{
    return std::size_t(p.rows) * p.cols;
}

// A one-block-column transfer involves only the root's process column.
// Otherwise every rank participates. Origins are zero-based here.
bool transfer_scope(const DdlaHandle_t& h, int n, int ja, const int* d, int root,
                    int& root_peer, int& root_col, CommScope& scope, int& rank, int& peers)
{
    int root_row;
    h->rank_to_rc(root, root_row, root_col);
    const int nb = d[DDLA_NB_];
    const bool column = (h->npcols_ == 1 || ja / nb == (ja + n - 1) / nb)
                        && root_col == indxg2p(ja, nb, d[DDLA_CSRC_], h->npcols_);
    scope = column ? CommScope::Col : CommScope::Grid;
    rank = column ? h->myprow_ : h->myid;
    peers = column ? h->nprows_ : h->nprocs;
    root_peer = column ? root_row : root;
    return !column || h->mypcol_ == root_col;
}

Peer transfer_peer(const DdlaHandle_t& h, int m, int n, int ia, int ja, const int* d,
                   CommScope scope, int root_col, int rank)
{
    int pr = rank, pc = root_col;
    if (scope != CommScope::Col) h->rank_to_rc(rank, pr, pc);
    const int r0 = num_loc(ia, d[DDLA_MB_], pr, d[DDLA_RSRC_], h->nprows_);
    const int c0 = num_loc(ja, d[DDLA_NB_], pc, d[DDLA_CSRC_], h->npcols_);
    const int r1 = num_loc(ia + m, d[DDLA_MB_], pr, d[DDLA_RSRC_], h->nprows_);
    const int c1 = num_loc(ja + n, d[DDLA_NB_], pc, d[DDLA_CSRC_], h->npcols_);
    return {r0, c0, r1 - r0, c1 - c0, pr, pc};
}

// The root stores the other participants' blocks back to back in its pack.
template <typename T>
int transfer_peer_blocks(const DdlaHandle_t& h, int m, int n, int ia, int ja, const int* d,
                         CommScope scope, int root_col, int root_peer, int peers,
                         T* packed, Block<T>* blocks)
{
    int count = 0;
    std::size_t offset = 0;
    for (int k = 0; k < peers; ++k) {
        if (k == root_peer) continue;
        const Peer p = transfer_peer(h, m, n, ia, ja, d, scope, root_col, k);
        ::new (static_cast<void*>(blocks + count++)) Block<T>{packed ? packed + offset : nullptr, p.rows, p};
        offset += block_size(p);
    }
    return count;
}

// The MPI paths of commSend/commRecv take an int count, so long messages go
// in pieces. An empty message sends nothing.
template <DdlaBackend B, typename T>
void send(const DdlaHandle_t& h, CommScope scope, const T* data, std::size_t count, int peer)
{
    constexpr std::size_t chunk = INT_MAX / sizeof(T);
    for (std::size_t k = 0; k < count; k += chunk)
        commSend<B>(h, scope, data + k, std::min(chunk, count - k), peer);
}

template <DdlaBackend B, typename T>
void recv(const DdlaHandle_t& h, CommScope scope, T* data, std::size_t count, int peer)
{
    constexpr std::size_t chunk = INT_MAX / sizeof(T);
    for (std::size_t k = 0; k < count; k += chunk)
        commRecv<B>(h, scope, data + k, std::min(chunk, count - k), peer);
}

// GPU handles take device pointers only.
template <DdlaBackend B>
void require_device([[maybe_unused]] const void* p, [[maybe_unused]] const char* what)
{
#if DDLA_HAS_GPU
    if constexpr (B == DdlaBackend::GPU) {
        if (gpuPointerMemType(p) != GpuMemType::Device) throw std::invalid_argument(what);
    }
#endif
}

// Collective argument check (component 1 of gather/scatter): each rank checks
// what it can see, then all ranks throw if any rank failed
// (detail::agree_check).  Whatever a rank finds wrong is reported as
// std::invalid_argument prefixed with the routine name.  Returns the packing
// scratch this rank needs.
template <DdlaBackend B, typename T>
std::size_t validate(const DdlaHandle_t& h, int m, int n, const T* A, int ia, int ja,
                     const int* d, int root, const T* buffer, int ldb, const char* name)
{
    return detail::agree_check(h, name, [&](detail::Agree&) -> std::size_t {
        if (!h || h->comm == MPI_COMM_NULL)
            throw std::invalid_argument(std::string(name) + ": handle has no communicator");
        try {
            std::size_t count = 0;
            if (ddlaGetBackend(h) != B) throw std::invalid_argument("backend mismatch");
            check_desc(d, h);
            if (m < 0 || n < 0 || ia < 1 || ja < 1
                || std::int64_t(ia) - 1 + m > d[DDLA_M_] || std::int64_t(ja) - 1 + n > d[DDLA_N_])
                throw std::invalid_argument("invalid submatrix bounds (ia/ja are one-based)");
            if (root < 0 || root >= h->nprocs) throw std::invalid_argument("invalid root rank");
            checked_count(m, n, sizeof(T));

            if (m && n) {
                if (h->myid == root) {
                    if (buffer) require_device<B>(buffer, "root storage must be device memory");
                    if (!buffer || ldb < m)
                        throw std::invalid_argument("invalid root buffer or leading dimension");
                    checked_count(ldb, n, sizeof(T));
                }
                ia--;
                ja--;
                int root_peer, root_col, rank, peers;
                CommScope scope;
                const bool active = transfer_scope(h, n, ja, d, root, root_peer, root_col, scope, rank, peers);
                if (active) {
                    const Peer self = transfer_peer(h, m, n, ia, ja, d, scope, root_col, rank);
                    if (block_size(self)) {
                        checked_count(d[DDLA_LLD_], self.col0 + self.cols, sizeof(T));
                        if (!A) throw std::invalid_argument("null local matrix data");
                        require_device<B>(A, "local matrix data must be device memory");
                    }
                    count = rank == root_peer ? std::size_t(m) * n - block_size(self) : block_size(self);
                }
            }
            return count;
        } catch (const std::exception& e) {
            throw std::invalid_argument(std::string(name) + ": " + e.what());
        }
    });
}

} // namespace

// Component 2: backend bytes of the packing buffer (the packing buffers), host
// bytes of the root's peer list (peers - 1 blocks).  An empty window needs
// nothing.
template <typename T>
void transfer_workspace_size(const DdlaHandle_t& h, int m, int n, int ia, int ja,
                             const int* desc, int root, std::size_t* backend_bytes,
                             std::size_t* host_bytes)
{
    *backend_bytes = *host_bytes = 0;
    if (!m || !n) return;
    ia--;
    ja--;
    int root_peer, root_col, rank, peers;
    CommScope scope;
    if (!transfer_scope(h, n, ja, desc, root, root_peer, root_col, scope, rank, peers)) return;
    const Peer self = transfer_peer(h, m, n, ia, ja, desc, scope, root_col, rank);
    *backend_bytes = (rank == root_peer ? std::size_t(m) * n - block_size(self) : block_size(self)) * sizeof(T);
    if (rank == root_peer) *host_bytes = std::size_t(peers - 1) * sizeof(Block<T>);
}

template <DdlaBackend B,typename T>
void gather_check(const DdlaHandle_t& h,int m,int n,const T* A,int ia,int ja,
    const int* desc,int root,T* dst,int lddst)
{ (void)validate<B>(h,m,n,A,ia,ja,desc,root,dst,lddst,"gather"); }

template <DdlaBackend B,typename T>
void scatter_check(const DdlaHandle_t& h,int m,int n,T* A,int ia,int ja,
    const int* desc,int root,const T* src,int ldsrc)
{ (void)validate<B>(h,m,n,A,ia,ja,desc,root,src,ldsrc,"scatter"); }

// Component 3.  The scratch is the packing buffer (backend_work) and the
// root's peer list (host_work).
template <DdlaBackend B, typename T>
void gather_run(const DdlaHandle_t& h, int m, int n, const T* A, int ia, int ja,
                const int* d, int root, T* dst, int lddst,
                void* backend_work, [[maybe_unused]] std::size_t backend_bytes,
                void* host_work, [[maybe_unused]] std::size_t host_bytes)
{
    if (!m || !n) return;
    ia--;
    ja--;
    int root_peer, root_col, rank, peers;
    CommScope scope;
    if (!transfer_scope(h, n, ja, d, root, root_peer, root_col, scope, rank, peers)) return;
    const Peer self = transfer_peer(h, m, n, ia, ja, d, scope, root_col, rank);
    assert((rank == root_peer ? std::size_t(m) * n - block_size(self) : block_size(self)) * sizeof(T) <= backend_bytes);
    T* const packed = static_cast<T*>(backend_work);
    const Block<T> own{block_size(self) ? const_cast<T*>(A) + self.row0 + std::size_t(self.col0) * d[DDLA_LLD_] : nullptr, d[DDLA_LLD_], self};

    // A participant packs its block and sends it to the root.
    if (rank != root_peer) {
        if (!block_size(own.p)) return;
        const std::size_t width = own.p.rows * sizeof(T);
        DDLA_CHECK(ddlaMemcpy2D(h, packed, width, own.data, own.ld * sizeof(T), width, own.p.cols));
        commGroupStart<B>(h);
        send<B>(h, scope, packed, block_size(own.p), root_peer);
        commGroupEnd<B>(h);
        return;
    }

    // Nothing moves when dst is exactly the root's local view of the window.
    if (own.data == dst && own.ld == lddst && own.p.rows == m && own.p.cols == n) return;

    // The root's own block is copied before the receives, while the peers
    // are still packing; the received blocks follow in one batch.
    map_blocks<B>(h, false, &own, 1, d, ia, ja, m, n, dst, lddst);
    assert(std::size_t(peers - 1) * sizeof(Block<T>) <= host_bytes);
    Block<T>* const blocks = static_cast<Block<T>*>(host_work);
    const int count = transfer_peer_blocks(h, m, n, ia, ja, d, scope, root_col, root_peer, peers, packed, blocks);
    commGroupStart<B>(h);
    for (int k = 0; k < count; ++k) {
        const Block<T>& b = blocks[k];
        recv<B>(h, scope, b.data, block_size(b.p), scope == CommScope::Col ? b.p.prow : h->rc_to_rank(b.p.prow, b.p.pcol));
    }
    commGroupEnd<B>(h);
    map_blocks<B>(h, false, blocks, count, d, ia, ja, m, n, dst, lddst);
}

// Component 3, with the scratch of gather_run.
template <DdlaBackend B, typename T>
void scatter_run(const DdlaHandle_t& h, int m, int n, T* A, int ia, int ja,
                 const int* d, int root, const T* src, int ldsrc, bool group_root,
                 void* backend_work, [[maybe_unused]] std::size_t backend_bytes,
                 void* host_work, [[maybe_unused]] std::size_t host_bytes)
{
    if (!m || !n) return;
    ia--;
    ja--;
    int root_peer, root_col, rank, peers;
    CommScope scope;
    if (!transfer_scope(h, n, ja, d, root, root_peer, root_col, scope, rank, peers)) return;
    const Peer self = transfer_peer(h, m, n, ia, ja, d, scope, root_col, rank);
    assert((rank == root_peer ? std::size_t(m) * n - block_size(self) : block_size(self)) * sizeof(T) <= backend_bytes);
    T* const packed = static_cast<T*>(backend_work);
    const Block<T> own{block_size(self) ? A + self.row0 + std::size_t(self.col0) * d[DDLA_LLD_] : nullptr, d[DDLA_LLD_], self};

    // A participant receives its packed block from the root and unpacks it.
    if (rank != root_peer) {
        if (!block_size(own.p)) return;
        commGroupStart<B>(h);
        recv<B>(h, scope, packed, block_size(own.p), root_peer);
        commGroupEnd<B>(h);
        const std::size_t width = own.p.rows * sizeof(T);
        DDLA_CHECK(ddlaMemcpy2D(h, own.data, own.ld * sizeof(T), packed, width, width, own.p.cols));
        return;
    }

    // Nothing moves when src is exactly the root's local view of the window.
    if (own.data == src && own.ld == ldsrc && own.p.rows == m && own.p.cols == n) return;

    T* const window = const_cast<T*>(src);  // only read: map copies window -> blocks
    assert(std::size_t(peers - 1) * sizeof(Block<T>) <= host_bytes);
    Block<T>* const blocks = static_cast<Block<T>*>(host_work);
    const int count = transfer_peer_blocks(h, m, n, ia, ja, d, scope, root_col, root_peer, peers, packed, blocks);
    map_blocks<B>(h, true, blocks, count, d, ia, ja, m, n, window, ldsrc);
    map_blocks<B>(h, true, &own, 1, d, ia, ja, m, n, window, ldsrc);
    if (group_root) commGroupStart<B>(h);
    for (int k = 0; k < count; ++k) {
        const Block<T>& b = blocks[k];
        send<B>(h, scope, b.data, block_size(b.p), scope == CommScope::Col ? b.p.prow : h->rc_to_rank(b.p.prow, b.p.pcol));
    }
    if (group_root) commGroupEnd<B>(h);
}

template <DdlaBackend B, typename T>
void convert_rows(const DdlaHandle_t& h, const typename real_type<T>::type* src,
                  int lds, T* dst, int ldd, int rows, int cols, bool transpose)
{
    if (!rows || !cols) return;
    if constexpr (B==DdlaBackend::CPU) {
        for (int c=0;c<cols;++c) for (int r=0;r<rows;++r)
            dst[r+std::size_t(c)*ldd]=src?T(transpose?src[c+std::size_t(r)*lds]
                                                  :src[r+std::size_t(c)*lds]):T(0);
    } else {
#if DDLA_HAS_GPU
        convert_rows_gpu<T>(h,src,lds,dst,ldd,rows,cols,transpose);
#endif
    }
}

namespace {
// Rows of one scatter_rows panel (k > 0, more than one rank): bounds both the
// conversion and the packing even for very large orders.
template <typename T>
int rows_panel(const DdlaHandle_t& h, int k)
{
    constexpr std::size_t panel_bytes=std::size_t(32)<<20;
    const detail::RowShare share(k,h->nprocs);
    return std::min({share.count(0),512,int(std::max<std::size_t>(1,panel_bytes/(std::size_t(k)*sizeof(T))))});
}
}

// Component 2: [ converted panel | the scatter's packing ] in backend memory,
// the scatter's peer list in host memory.  Every panel window has at most
// panel_rows x k entries and nprocs - 1 peers (transfer_workspace_size).
template <typename T>
void scatter_rows_workspace_size(const DdlaHandle_t& h,int k,std::size_t* backend_bytes,std::size_t* host_bytes)
{
    *backend_bytes=*host_bytes=0;
    if (h->nprocs<=1 || k==0) return;
    const std::size_t panel=std::size_t(rows_panel<T>(h,k))*k;
    *backend_bytes=detail::padded_bytes<T>(panel)+panel*sizeof(T);
    *host_bytes=std::size_t(h->nprocs-1)*sizeof(Block<T>);
}

template <DdlaBackend B,typename T>
void scatter_rows_check(const DdlaHandle_t& h,int k,const typename real_type<T>::type* src,
    int lds,bool transpose,int m,int n,T* dst,const int* desc)
{
    ddla::detail::agree_check(h,"scatter_rows",[&](ddla::detail::Agree& agreement) {
        if (!h) throw std::invalid_argument("scatter_rows: null handle");
        if (ddlaGetBackend(h)!=B || k<0 || m<k || n<k)
            throw std::invalid_argument("scatter_rows: incompatible backend or extents");
        check_desc(desc,h);
        if (m>desc[DDLA_M_] || n>desc[DDLA_N_])
            throw std::invalid_argument("scatter_rows: output exceeds descriptor");
        const detail::RowShare share(k,h->nprocs);
        if (lds<std::max(1,share.count(h->myid)) || (k && share.count(h->myid)>0 && !src))
            throw std::invalid_argument("scatter_rows: missing source rows");
        const int mloc=num_loc(m,desc[DDLA_MB_],h->myprow_,desc[DDLA_RSRC_],h->nprows_);
        const int nloc=num_loc(n,desc[DDLA_NB_],h->mypcol_,desc[DDLA_CSRC_],h->npcols_);
        if (mloc>0 && nloc>0 && !dst)throw std::invalid_argument("scatter_rows: missing output");
        agreement.mode("row-slab order",k);agreement.mode("row-slab output rows",m);
        agreement.mode("row-slab output columns",n);agreement.mode("row-slab transpose",transpose);
        checked_count(desc[DDLA_LLD_],nloc,sizeof(T));
        checked_count(lds,k,sizeof(typename real_type<T>::type));
    });
}

// Component 3.  Carves the converted panel and hands the rest of the scratch
// to scatter_run.
template <DdlaBackend B,typename T>
void scatter_rows_run(const DdlaHandle_t& h,int k,const typename real_type<T>::type* src,int lds,
    bool transpose,int m,int n,T* dst,const int* desc,
    void* backend_work,std::size_t backend_bytes,void* host_work,std::size_t host_bytes)
{
    try {
        const int panel_rows=h->nprocs>1 && k>0 ? rows_panel<T>(h,k) : 0;
        const std::size_t panel=std::size_t(panel_rows)*k;
        char* w=static_cast<char*>(backend_work);
        T* const converted=detail::take<T>(w,panel);
        const std::size_t used=w-static_cast<char*>(backend_work);
        assert(used<=backend_bytes);
        const int mloc=num_loc(m,desc[DDLA_MB_],h->myprow_,desc[DDLA_RSRC_],h->nprows_);
        const int nloc=num_loc(n,desc[DDLA_NB_],h->mypcol_,desc[DDLA_CSRC_],h->npcols_);
        const int factor_rows=num_loc(k,desc[DDLA_MB_],h->myprow_,desc[DDLA_RSRC_],h->nprows_);
        const int factor_cols=num_loc(k,desc[DDLA_NB_],h->mypcol_,desc[DDLA_CSRC_],h->npcols_);
        // Operational failure agreement only: all argument checking is complete.
        detail::agree_check(h,"scatter_rows preparation",[&](detail::Agree&) {
            if (mloc>factor_rows && nloc>0)
                convert_rows<B,T>(h,nullptr,1,dst+factor_rows,desc[DDLA_LLD_],mloc-factor_rows,nloc,false);
            if (nloc>factor_cols && factor_rows>0)
                convert_rows<B,T>(h,nullptr,1,dst+std::size_t(factor_cols)*desc[DDLA_LLD_],
                                  desc[DDLA_LLD_],factor_rows,nloc-factor_cols,false);
            if (h->nprocs==1)convert_rows<B,T>(h,src,lds,dst,desc[DDLA_LLD_],k,k,transpose);
        });
        const detail::RowShare share(k,h->nprocs);
        if (h->nprocs>1) for(int root=0;root<h->nprocs;++root) {
            for(int first=0;first<share.count(root);first+=panel_rows) {
                const int rows=std::min(panel_rows,share.count(root)-first);
                const int nr=transpose?k:rows,nc=transpose?rows:k;
                if constexpr(B==DdlaBackend::CPU) {
                    if(h->myid==root)convert_rows<B,T>(h,src+first,lds,converted,nr,nr,nc,transpose);
                } else {
                    ddla::detail::agree_check(h,"scatter_rows conversion",[&](ddla::detail::Agree&) {
                        if(h->myid==root)convert_rows<B,T>(h,src+first,lds,converted,nr,nr,nc,transpose);
                    });
                }
                const int global=share.lo(root)+first+1;
                scatter_run<B,T>(h,nr,nc,dst,transpose?1:global,transpose?global:1,desc,
                                root,converted,nr,true,w,backend_bytes-used,host_work,host_bytes);
            }
        }
        ddla::detail::agree_check(h,"scatter_rows completion",[&](ddla::detail::Agree&) {
            DDLA_CHECK(ddlaSynchronize(h));
        });
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(h) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(h); } catch (...) {}
        throw;
    }
}

template <DdlaBackend B, typename T>
void scatter_rows(const DdlaHandle_t& h,int k,const typename real_type<T>::type* src,
                  int lds,bool transpose,int m,int n,T* dst,const int* desc)
{
    scatter_rows_check<B,T>(h,k,src,lds,transpose,m,n,dst,desc);
    std::size_t backend_bytes=0,host_bytes=0;
    scatter_rows_workspace_size<T>(h,k,&backend_bytes,&host_bytes);
    vector<char> backend_work(h);
    std::vector<char> host_work;
    // Keep allocation failures coordinated before any output or transfer.
    detail::agree_check(h,"scatter_rows workspace",[&](detail::Agree&){
        backend_work.resize(backend_bytes);host_work.resize(host_bytes);});
    scatter_rows_run<B,T>(h,k,src,lds,transpose,m,n,dst,desc,
                          backend_work.data(),backend_bytes,host_work.data(),host_bytes);
}

#define INST_ROWS(B,T) \
 template void convert_rows<B,T>(const DdlaHandle_t&,const real_type<T>::type*,int,T*,int,int,int,bool); \
 template void scatter_rows<B,T>(const DdlaHandle_t&,int,const real_type<T>::type*,int,bool,int,int,T*,const int*); \
 template void scatter_rows_check<B,T>(const DdlaHandle_t&,int,const real_type<T>::type*,int,bool,int,int,T*,const int*); \
 template void scatter_rows_run<B,T>(const DdlaHandle_t&,int,const real_type<T>::type*,int,bool,int,int,T*,const int*,void*,std::size_t,void*,std::size_t)
DDLA_FOR_EACH_BACKEND_TYPE(INST_ROWS)
#undef INST_ROWS

#define INST_ROW_SIZE(T) template void scatter_rows_workspace_size<T>(const DdlaHandle_t&,int,std::size_t*,std::size_t*)
DDLA_FOR_EACH_TYPE(INST_ROW_SIZE)
#undef INST_ROW_SIZE

} // namespace ddla::matrix_impl

namespace ddla {

template <DdlaBackend B, typename T>
void gather(const DdlaHandle_t& h, int m, int n, const T* A, int ia, int ja,
            const int* d, int root, T* dst, int lddst)
{
    // 1. check (collective).
    matrix_impl::gather_check<B,T>(h,m,n,A,ia,ja,d,root,dst,lddst);
    // 2. the packing buffer and the root's peer list.
    std::size_t backend_bytes=0,host_bytes=0;
    matrix_impl::transfer_workspace_size<T>(h,m,n,ia,ja,d,root,&backend_bytes,&host_bytes);
    vector<char> backend_work(h,backend_bytes);
    std::vector<char> host_work(host_bytes);
    // 3. the transfer.
    matrix_impl::gather_run<B,T>(h,m,n,A,ia,ja,d,root,dst,lddst,
                                 backend_work.data(),backend_bytes,host_work.data(),host_bytes);
}

template <DdlaBackend B, typename T>
void scatter(const DdlaHandle_t& h, int m, int n, T* A, int ia, int ja,
             const int* d, int root, const T* src, int ldsrc)
{
    // 1. check (collective).
    matrix_impl::scatter_check<B,T>(h,m,n,A,ia,ja,d,root,src,ldsrc);
    // 2. the packing buffer and the root's peer list.
    std::size_t backend_bytes=0,host_bytes=0;
    matrix_impl::transfer_workspace_size<T>(h,m,n,ia,ja,d,root,&backend_bytes,&host_bytes);
    vector<char> backend_work(h,backend_bytes);
    std::vector<char> host_work(host_bytes);
    // 3. the transfer.
    matrix_impl::scatter_run<B,T>(h,m,n,A,ia,ja,d,root,src,ldsrc,true,
                                  backend_work.data(),backend_bytes,host_work.data(),host_bytes);
}

#define INST(B, T) \
    template void matrix_impl::gather_check<B,T>(const DdlaHandle_t&,int,int,const T*,int,int,const int*,int,T*,int); \
    template void matrix_impl::scatter_check<B,T>(const DdlaHandle_t&,int,int,T*,int,int,const int*,int,const T*,int); \
    template void matrix_impl::gather_run<B,T>(const DdlaHandle_t&,int,int,const T*,int,int,const int*,int,T*,int, \
                                               void*,std::size_t,void*,std::size_t); \
    template void matrix_impl::scatter_run<B,T>(const DdlaHandle_t&,int,int,T*,int,int,const int*,int,const T*,int,bool, \
                                                void*,std::size_t,void*,std::size_t); \
    template void gather<B, T>(const DdlaHandle_t&, int, int, const T*, int, int, \
                               const int*, int, T*, int); \
    template void scatter<B, T>(const DdlaHandle_t&, int, int, T*, int, int, \
                                const int*, int, const T*, int)
DDLA_FOR_EACH_BACKEND_TYPE(INST)
#undef INST

#define INST_TRANSFER_SIZE(T) template void matrix_impl::transfer_workspace_size<T>(const DdlaHandle_t&,int,int,int,int,const int*,int,std::size_t*,std::size_t*)
DDLA_FOR_EACH_TYPE(INST_TRANSFER_SIZE)
#undef INST_TRANSFER_SIZE

} // namespace ddla
