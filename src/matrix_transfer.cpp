#include "matrix_transfer.h"
#include "ddla_desc.h"
#include "ddla_stream.h"
#include "comm_traits.h"
#include "check_agree.h"
#include "plan_util.h"

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

// Who takes part in one transfer. A window inside one block column whose
// process column holds the root involves only that column (Col scope, where
// ranks are process rows); any other window involves the whole grid. Either
// way the participants own the whole window between them.
struct Transfer {
    Region g;
    int root, root_row, root_col;
    bool column;      // only the root's process column takes part
    bool active;      // this rank takes part
    CommScope scope;
    int rank;         // this rank within scope
    int peers;        // number of ranks within scope
    Peer self;        // the part of the window this rank owns

    Transfer(const DdlaHandle_t& h, int m, int n, int ia, int ja, const int* d, int target)
        : g{ia - 1, ja - 1, m, n, d[DDLA_MB_], d[DDLA_NB_], d[DDLA_RSRC_], d[DDLA_CSRC_],
            h->nprows_, h->npcols_},
          root(target)
    {
        h->rank_to_rc(root, root_row, root_col);
        const bool one_block_column = g.j / g.nb == (g.j + g.n - 1) / g.nb;
        column = (g.npcols == 1 || one_block_column)
                 && root_col == indxg2p(g.j, g.nb, g.csrc, g.npcols);
        active = !column || h->mypcol_ == root_col;
        scope = column ? CommScope::Col : CommScope::Grid;
        rank = column ? h->myprow_ : h->myid;
        peers = column ? g.nprows : h->nprocs;
        self = peer(h, rank);
    }

    int root_peer() const { return column ? root_row : root; }
    bool is_root() const { return rank == root_peer(); }

    Peer peer(const DdlaHandle_t& h, int k) const
    {
        int pr = k, pc = root_col;
        if (!column) h->rank_to_rc(k, pr, pc);
        const int r0 = num_loc(g.i, g.mb, pr, g.rsrc, g.nprows);
        const int c0 = num_loc(g.j, g.nb, pc, g.csrc, g.npcols);
        const int r1 = num_loc(g.i + g.m, g.mb, pr, g.rsrc, g.nprows);
        const int c1 = num_loc(g.j + g.n, g.nb, pc, g.csrc, g.npcols);
        return {r0, c0, r1 - r0, c1 - c0, pr, pc};
    }

    // The rank within scope of the process holding p.
    int rank_of(const DdlaHandle_t& h, const Peer& p) const
    {
        return column ? p.prow : h->rc_to_rank(p.prow, p.pcol);
    }

    // This rank's block inside its local storage A (null data when it owns none).
    template <typename T>
    Block<T> own_block(T* A, int ld) const
    {
        return {block_size(self) ? A + self.row0 + std::size_t(self.col0) * ld : nullptr, ld, self};
    }

    // The other participants' blocks, back to back in the root's packing
    // buffer; returns their number (peers - 1).
    template <typename T>
    int peer_blocks(const DdlaHandle_t& h, T* packed, Block<T>* blocks) const
    {
        int count = 0;
        std::size_t offset = 0;
        for (int k = 0; k < peers; ++k) {
            if (k == root_peer()) continue;
            const Peer p = peer(h, k);
            ::new (static_cast<void*>(blocks + count++)) Block<T>{packed ? packed + offset : nullptr, p.rows, p};
            offset += block_size(p);
        }
        return count;
    }

    // Packing scratch this rank needs: its own block on a participant, the
    // rest of the window on the root.
    std::size_t scratch() const
    {
        if (!active) return 0;
        return is_root() ? std::size_t(g.m) * g.n - block_size(self) : block_size(self);
    }
};

// Copies between blocks and their entries of the root's window: scatter copies
// window -> blocks, gather blocks -> window. Local rows inside one distribution
// block are contiguous in the window too, so on the host each such run of a
// column is one memcpy, column by column (tile-sized 2D copies touch many
// columns at once and measured slower). On the GPU a copy call costs more than
// the data it moves, so one kernel maps whole blocks.
template <DdlaBackend B, typename T>
void map(const DdlaHandle_t& h, bool scatter, const Block<T>* blocks, int count,
         const Region& g, T* window, int ldw)
{
    if constexpr (B == DdlaBackend::GPU) {
        map_gpu(h, scatter, blocks, count, g, window, ldw);
    } else {
        for (int k = 0; k < count; ++k) {
            const Block<T>& b = blocks[k];
            if (!b.p.rows || !b.p.cols) continue;
            for (int c = 0; c < b.p.cols; ++c) {
                const int wc = indxl2g(b.p.col0 + c, g.nb, b.p.pcol, g.csrc, g.npcols) - g.j;
                T* const column = b.data + std::size_t(c) * b.ld;
                T* const wcolumn = window + std::size_t(wc) * ldw;
                for (int r = 0; r < b.p.rows;) {
                    const int lr = b.p.row0 + r;
                    const int rows = std::min(b.p.rows - r, g.mb - lr % g.mb);
                    T* const w = wcolumn + indxl2g(lr, g.mb, b.p.prow, g.rsrc, g.nprows) - g.i;
                    if (scatter) std::memcpy(column + r, w, rows * sizeof(T));
                    else std::memcpy(w, column + r, rows * sizeof(T));
                    r += rows;
                }
            }
        }
    }
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
#if DDLA_GPU_VENDOR
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
                const Transfer t(h, m, n, ia, ja, d, root);
                if (t.active && block_size(t.self)) {
                    checked_count(d[DDLA_LLD_], t.self.col0 + t.self.cols, sizeof(T));
                    if (!A) throw std::invalid_argument("null local matrix data");
                    require_device<B>(A, "local matrix data must be device memory");
                }
                count = t.scratch();
            }
            return count;
        } catch (const std::exception& e) {
            throw std::invalid_argument(std::string(name) + ": " + e.what());
        }
    });
}

} // namespace

// Component 2: backend bytes of the packing buffer (Transfer::scratch()), host
// bytes of the root's peer list (peers - 1 blocks).  An empty window needs
// nothing.
template <typename T>
void transfer_workspace_size(const DdlaHandle_t& h, int m, int n, int ia, int ja,
                             const int* desc, int root, std::size_t* backend_bytes,
                             std::size_t* host_bytes)
{
    *backend_bytes = *host_bytes = 0;
    if (!m || !n) return;
    const Transfer t(h, m, n, ia, ja, desc, root);
    *backend_bytes = t.scratch() * sizeof(T);
    if (t.active && t.is_root()) *host_bytes = std::size_t(t.peers - 1) * sizeof(Block<T>);
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
    const Transfer t(h, m, n, ia, ja, d, root);
    if (!t.active) return;
    assert(t.scratch() * sizeof(T) <= backend_bytes);
    T* const packed = static_cast<T*>(backend_work);
    const Block<T> own = t.own_block(const_cast<T*>(A), d[DDLA_LLD_]);

    // A participant packs its block and sends it to the root.
    if (!t.is_root()) {
        if (!block_size(own.p)) return;
        const std::size_t width = own.p.rows * sizeof(T);
        DDLA_CHECK(ddlaMemcpy2D(h, packed, width, own.data, own.ld * sizeof(T), width, own.p.cols));
        commGroupStart<B>(h);
        send<B>(h, t.scope, packed, block_size(own.p), t.root_peer());
        commGroupEnd<B>(h);
        return;
    }

    // Nothing moves when dst is exactly the root's local view of the window.
    if (own.data == dst && own.ld == lddst && own.p.rows == m && own.p.cols == n) return;

    // The root's own block is copied before the receives, while the peers
    // are still packing; the received blocks follow in one batch.
    map<B>(h, false, &own, 1, t.g, dst, lddst);
    assert(std::size_t(t.peers - 1) * sizeof(Block<T>) <= host_bytes);
    Block<T>* const blocks = static_cast<Block<T>*>(host_work);
    const int count = t.peer_blocks(h, packed, blocks);
    commGroupStart<B>(h);
    for (int k = 0; k < count; ++k) {
        const Block<T>& b = blocks[k];
        recv<B>(h, t.scope, b.data, block_size(b.p), t.rank_of(h, b.p));
    }
    commGroupEnd<B>(h);
    map<B>(h, false, blocks, count, t.g, dst, lddst);
}

// Component 3, with the scratch of gather_run.
template <DdlaBackend B, typename T>
void scatter_run(const DdlaHandle_t& h, int m, int n, T* A, int ia, int ja,
                 const int* d, int root, const T* src, int ldsrc, bool group_root,
                 void* backend_work, [[maybe_unused]] std::size_t backend_bytes,
                 void* host_work, [[maybe_unused]] std::size_t host_bytes)
{
    if (!m || !n) return;
    const Transfer t(h, m, n, ia, ja, d, root);
    if (!t.active) return;
    assert(t.scratch() * sizeof(T) <= backend_bytes);
    T* const packed = static_cast<T*>(backend_work);
    const Block<T> own = t.own_block(A, d[DDLA_LLD_]);

    // A participant receives its packed block from the root and unpacks it.
    if (!t.is_root()) {
        if (!block_size(own.p)) return;
        commGroupStart<B>(h);
        recv<B>(h, t.scope, packed, block_size(own.p), t.root_peer());
        commGroupEnd<B>(h);
        const std::size_t width = own.p.rows * sizeof(T);
        DDLA_CHECK(ddlaMemcpy2D(h, own.data, own.ld * sizeof(T), packed, width, width, own.p.cols));
        return;
    }

    // Nothing moves when src is exactly the root's local view of the window.
    if (own.data == src && own.ld == ldsrc && own.p.rows == m && own.p.cols == n) return;

    T* const window = const_cast<T*>(src);  // only read: map copies window -> blocks
    assert(std::size_t(t.peers - 1) * sizeof(Block<T>) <= host_bytes);
    Block<T>* const blocks = static_cast<Block<T>*>(host_work);
    const int count = t.peer_blocks(h, packed, blocks);
    map<B>(h, true, blocks, count, t.g, window, ldsrc);
    map<B>(h, true, &own, 1, t.g, window, ldsrc);
    if (group_root) commGroupStart<B>(h);
    for (int k = 0; k < count; ++k) {
        const Block<T>& b = blocks[k];
        send<B>(h, t.scope, b.data, block_size(b.p), t.rank_of(h, b.p));
    }
    if (group_root) commGroupEnd<B>(h);
}

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
#define TYPES(B) \
    INST(B, float); \
    INST(B, double); \
    INST(B, std::complex<float>); \
    INST(B, std::complex<double>)
#if DDLA_HAS_CPU
TYPES(DdlaBackend::CPU);
#endif
#if DDLA_HAS_GPU
TYPES(DdlaBackend::GPU);
#endif
#undef TYPES
#undef INST

#define INST_TRANSFER_SIZE(T) template void matrix_impl::transfer_workspace_size<T>(const DdlaHandle_t&,int,int,int,int,const int*,int,std::size_t*,std::size_t*)
INST_TRANSFER_SIZE(float); INST_TRANSFER_SIZE(double); INST_TRANSFER_SIZE(std::complex<float>); INST_TRANSFER_SIZE(std::complex<double>);
#undef INST_TRANSFER_SIZE

} // namespace ddla
