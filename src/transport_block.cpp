#include <ddla/ddla.h>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>
#include <ddla/ddla_connector.h>
#include "ddla_stream_impl.h"
#include "comm_traits.h"
#include <ddla/geam.h>

namespace ddla {

inline const char* transport_block_backend_name(DdlaBackend backend)
{
    return backend == DdlaBackend::CPU ? "CPU" : "GPU";
}

// ---------------------------------------------------------------------------
// Unified transport_block<Backend,T>: merges the formerly GPU-only
// transport_block.cpp (NCCL/RCCL/host-tunnel communication, deblasOmatcopy
// transpose-copy, runtime* device memcpy/alloc) and the formerly CPU-only
// transport_panel_cpu (plain blocking MPI, manual transpose+conjugate loops,
// std::malloc/memcpy) hand-duplicated inside pgemm.cpp. The two were already
// branch-for-branch identical control flow (same split on trans=='N' vs
// transposed, same split on square vs non-square process grid, same index
// math via num_loc/indxg2p/indxg2l/rc_to_rank) -- only a few leaf operations
// genuinely differ, and none of them needs compile-time branch dispatch here:
// each is a single call into a unified primitive that owns the split itself:
//   copy_block     -> ddla::copy2D<Backend,T>       (geam.h / omatcopy.cpp)
//   pack_transpose -> ddla::omatcopy<Backend,T>     (geam.h / omatcopy.cpp)
//   alloc_buf/free_buf -> runtimeMalloc/runtimeFree<Backend> (ddla_connector.h)
// All collective communication (broadcast/send/recv/group) goes through the
// CommTraits<Backend>-based comm* primitives in comm_traits.h, which already
// select the right MPI/NCCL/RCCL/tunnel mechanism per backend.
// ---------------------------------------------------------------------------
template <DdlaBackend Backend, typename T>
void transport_block(
    const char& sData, const char& trans,
    const int& m, const int& n,
    const T* d_A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    T* d_block_A)
{
    if (m == 0 || n == 0) return;

    DdlaHandle_t h = array_descA.ddla_handle();
    if (h == nullptr) {
        throw std::runtime_error("transport_block: null handle");
    }
    const DdlaBackend actual_backend = ddla_get_backend(h);
    if (actual_backend != Backend) {
        throw std::runtime_error(
            std::string("transport_block: template backend ") +
            transport_block_backend_name(Backend) +
            " does not match descriptor handle backend " +
            transport_block_backend_name(actual_backend));
    }

    assert(sData == 'C' || sData == 'R');
    assert(trans == 'N' || trans == 'T' || trans == 'C');

    const DdlaDesc& d = array_descA;
    const int myrank = h->myid;

    // ---- divergent leaf operations: the only genuine CPU/GPU differences --
    // Plain column-contiguous block copy: dst[j*dst_pitch + i] <- src[i + j*src_ld].
    auto copy_block = [&](T* dst, int dst_pitch, const T* src, int rows, int cols, int src_ld) {
        ddla::copy2D<Backend, T>(h, dst, dst_pitch, src, src_ld, rows, cols);
    };
    // Transposed (and, for 'C', conjugated) pack:
    // dst[j + i*dst_stride] <- op(src)[i + j*src_ld], for i in [0,rows), j in [0,cols).
    auto pack_transpose = [&](T* dst, int dst_stride, const T* src, int rows, int cols, int src_ld) {
        const char op = (trans == 'C') ? 'C' : 'T';
        ddla::omatcopy<Backend, T>(h, op, rows, cols, (T)1.0, src, src_ld, dst, dst_stride);
    };
    auto alloc_buf = [&](std::size_t count) -> T* {
        if (count == 0) return nullptr;
        T* p = nullptr;
        RUNTIME_CHECK<Backend>(runtimeMalloc<Backend>(
            reinterpret_cast<void**>(&p), count * sizeof(T)));
        return p;
    };
    auto free_buf = [&](T* p) {
        RUNTIME_CHECK<Backend>(runtimeFree<Backend>(p));
    };

    // ---- index math: identical across both formerly-separate implementations
    int i_loc = num_loc(ia, d.mb(), d.myprow(), d.irsrc(), d.nprows());
    int j_loc = num_loc(ja, d.nb(), d.mypcol(), d.icsrc(), d.npcols());
    int m_loc = num_loc(ia + m, d.mb(), d.myprow(), d.irsrc(), d.nprows());
    int n_loc = num_loc(ja + n, d.nb(), d.mypcol(), d.icsrc(), d.npcols());
    int owner_row = indxg2p(ia, d.mb(), d.irsrc(), d.nprows());
    int owner_col = indxg2p(ja, d.nb(), d.icsrc(), d.npcols());

    if (trans == 'N') {
        if (sData == 'R' && n_loc > j_loc) {
            int cols = n_loc - j_loc;
            if (d.myprow() == owner_row)
                copy_block(d_block_A, m, d_A + i_loc + (std::size_t)j_loc * d.lld(), m, cols, d.lld());
            commBcast<Backend>(h, CommScope::Col, d_block_A, (std::size_t)m * cols, owner_row);
        } else if (sData == 'C' && m_loc > i_loc) {
            int rows = m_loc - i_loc;
            if (d.mypcol() == owner_col)
                copy_block(d_block_A, rows, d_A + i_loc + (std::size_t)j_loc * d.lld(), rows, n, d.lld());
            commBcast<Backend>(h, CommScope::Row, d_block_A, (std::size_t)rows * n, owner_col);
        }
    } else if (d.nprows() == d.npcols()) {
        // ---- square-grid transposed path -----------------------------------
        int trans_j_loc = num_loc(ja, d.nb(), d.myprow(), d.icsrc(), d.nprows());
        int trans_n_loc = num_loc(ja + n, d.nb(), d.myprow(), d.icsrc(), d.nprows());
        int trans_i_loc = num_loc(ia, d.mb(), d.mypcol(), d.irsrc(), d.npcols());
        int trans_m_loc = num_loc(ia + m, d.mb(), d.mypcol(), d.irsrc(), d.npcols());

        if (sData == 'R') {
            if (n_loc > j_loc) {
                int cols = n_loc - j_loc;
                if (d.myprow() == owner_row) {
                    copy_block(d_block_A, m, d_A + i_loc + (std::size_t)j_loc * d.lld(), m, cols, d.lld());
                    if (d.myprow() != d.mypcol())
                        commSend<Backend>(h, CommScope::Col, d_block_A, (std::size_t)m * cols, d.mypcol());
                } else if (d.myprow() == d.mypcol()) {
                    commRecv<Backend>(h, CommScope::Col, d_block_A, (std::size_t)m * cols, owner_row);
                }
            }
            if (trans_n_loc > trans_j_loc) {
                int tcols = trans_n_loc - trans_j_loc;
                commBcast<Backend>(h, CommScope::Row, d_block_A, (std::size_t)tcols * m, d.myprow());
            }
        } else { // sData == 'C'
            if (m_loc > i_loc) {
                int rows = m_loc - i_loc;
                if (d.mypcol() == owner_col) {
                    pack_transpose(d_block_A, n, d_A + i_loc + (std::size_t)j_loc * d.lld(), rows, n, d.lld());
                    if (d.myprow() != d.mypcol())
                        commSend<Backend>(h, CommScope::Row, d_block_A, (std::size_t)rows * n, d.myprow());
                } else if (d.myprow() == d.mypcol()) {
                    commRecv<Backend>(h, CommScope::Row, d_block_A, (std::size_t)rows * n, owner_col);
                }
            }
            if (trans_m_loc > trans_i_loc) {
                int trows = trans_m_loc - trans_i_loc;
                commBcast<Backend>(h, CommScope::Col, d_block_A, (std::size_t)trows * n, d.mypcol());
            }
        }
    } else {
        // ---- non-square-grid transposed path: rectangular redistribution --
        struct RectBlock {
            int src_rank;
            int dst_rank;
            int count;
            int send_offset;
            int dst_offset;
            int src_loc;
            int len;
        };

        auto block_end = [](int g, int block_size, int end) -> int {
            return std::min(end, (g / block_size + 1) * block_size);
        };

        if (sData == 'R') {
            const int target_j_loc = num_loc(ja, d.nb(), d.myprow(), d.icsrc(), d.nprows());
            const int target_n_loc = num_loc(ja + n, d.nb(), d.myprow(), d.icsrc(), d.nprows());
            const int target_cols = target_n_loc - target_j_loc;
            std::vector<RectBlock> blocks, send_blocks, recv_blocks;
            int send_total = 0;

            for (int g = ja; g < ja + n; ) {
                const int len = block_end(g, d.nb(), ja + n) - g;
                const int src_col = indxg2p(g, d.nb(), d.icsrc(), d.npcols());
                const int dst_row = indxg2p(g, d.nb(), d.icsrc(), d.nprows());
                const int src_rank = h->rc_to_rank(owner_row, src_col);
                const int dst_rank = h->rc_to_rank(dst_row, 0);
                const int dst_col_loc = num_loc(g, d.nb(), dst_row, d.icsrc(), d.nprows())
                                       - num_loc(ja, d.nb(), dst_row, d.icsrc(), d.nprows());
                RectBlock block{src_rank, dst_rank, m * len, -1, dst_col_loc * m,
                                indxg2l(g, d.nb(), d.npcols()), len};

                if (src_rank == dst_rank) {
                    if (myrank == src_rank)
                        copy_block(d_block_A + block.dst_offset, m,
                                   d_A + i_loc + (std::size_t)block.src_loc * d.lld(), m, block.len, d.lld());
                } else {
                    blocks.push_back(block);
                    if (myrank == src_rank) {
                        block.send_offset = send_total;
                        send_total += block.count;
                        send_blocks.push_back(block);
                    }
                    if (myrank == dst_rank) {
                        recv_blocks.push_back(block);
                    }
                }
                g += len;
            }

            T* d_sendbuf = (send_total > 0) ? alloc_buf((std::size_t)send_total) : nullptr;
            for (const auto& block : send_blocks)
                copy_block(d_sendbuf + block.send_offset, m,
                           d_A + i_loc + (std::size_t)block.src_loc * d.lld(), m, block.len, d.lld());

            if (!send_blocks.empty() || !recv_blocks.empty()) {
                commGroupStart<Backend>(h);
                for (const auto& block : blocks) {
                    if (myrank == block.src_rank) {
                        auto it = std::find_if(send_blocks.begin(), send_blocks.end(), [&](const RectBlock& item) {
                            return item.dst_rank == block.dst_rank && item.dst_offset == block.dst_offset;
                        });
                        assert(it != send_blocks.end());
                        commSend<Backend>(h, CommScope::Grid, d_sendbuf + it->send_offset,
                                          (std::size_t)block.count, block.dst_rank);
                    } else if (myrank == block.dst_rank) {
                        commRecv<Backend>(h, CommScope::Grid, d_block_A + block.dst_offset,
                                          (std::size_t)block.count, block.src_rank);
                    }
                }
                commGroupEnd<Backend>(h);
            }

            if (target_cols > 0)
                commBcast<Backend>(h, CommScope::Row, d_block_A, (std::size_t)m * target_cols, 0);
            free_buf(d_sendbuf);
        } else { // sData == 'C'
            const int target_i_loc = num_loc(ia, d.mb(), d.mypcol(), d.irsrc(), d.npcols());
            const int target_m_loc = num_loc(ia + m, d.mb(), d.mypcol(), d.irsrc(), d.npcols());
            const int target_cols = target_m_loc - target_i_loc;
            std::vector<RectBlock> blocks, send_blocks, recv_blocks;
            int send_total = 0;

            for (int g = ia; g < ia + m; ) {
                const int len = block_end(g, d.mb(), ia + m) - g;
                const int src_row = indxg2p(g, d.mb(), d.irsrc(), d.nprows());
                const int dst_col = indxg2p(g, d.mb(), d.irsrc(), d.npcols());
                const int src_rank = h->rc_to_rank(src_row, owner_col);
                const int dst_rank = h->rc_to_rank(0, dst_col);
                const int dst_col_loc = num_loc(g, d.mb(), dst_col, d.irsrc(), d.npcols())
                                       - num_loc(ia, d.mb(), dst_col, d.irsrc(), d.npcols());
                RectBlock block{src_rank, dst_rank, n * len, -1, dst_col_loc * n,
                                indxg2l(g, d.mb(), d.nprows()), len};

                if (src_rank == dst_rank) {
                    if (myrank == src_rank)
                        pack_transpose(d_block_A + block.dst_offset, n,
                                       d_A + (std::size_t)block.src_loc + (std::size_t)j_loc * d.lld(),
                                       block.len, n, d.lld());
                } else {
                    blocks.push_back(block);
                    if (myrank == src_rank) {
                        block.send_offset = send_total;
                        send_total += block.count;
                        send_blocks.push_back(block);
                    }
                    if (myrank == dst_rank) {
                        recv_blocks.push_back(block);
                    }
                }
                g += len;
            }

            T* d_sendbuf = (send_total > 0) ? alloc_buf((std::size_t)send_total) : nullptr;
            for (const auto& block : send_blocks)
                pack_transpose(d_sendbuf + block.send_offset, n,
                               d_A + (std::size_t)block.src_loc + (std::size_t)j_loc * d.lld(),
                               block.len, n, d.lld());

            if (!send_blocks.empty() || !recv_blocks.empty()) {
                commGroupStart<Backend>(h);
                for (const auto& block : blocks) {
                    if (myrank == block.src_rank) {
                        auto it = std::find_if(send_blocks.begin(), send_blocks.end(), [&](const RectBlock& item) {
                            return item.dst_rank == block.dst_rank && item.dst_offset == block.dst_offset;
                        });
                        assert(it != send_blocks.end());
                        commSend<Backend>(h, CommScope::Grid, d_sendbuf + it->send_offset,
                                          (std::size_t)block.count, block.dst_rank);
                    } else if (myrank == block.dst_rank) {
                        commRecv<Backend>(h, CommScope::Grid, d_block_A + block.dst_offset,
                                          (std::size_t)block.count, block.src_rank);
                    }
                }
                commGroupEnd<Backend>(h);
            }

            if (target_cols > 0)
                commBcast<Backend>(h, CommScope::Col, d_block_A, (std::size_t)n * target_cols, 0);
            free_buf(d_sendbuf);
        }
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------
#define INSTANTIATE_TRANSPORT_BLOCK(BACKEND, TYPE)                          \
    template void transport_block<BACKEND, TYPE>(                           \
        const char&, const char&, const int&, const int&,                  \
        const TYPE*, const int&, const int&, const DdlaDesc&, TYPE*)

#if DDLA_HAS_CPU
INSTANTIATE_TRANSPORT_BLOCK(DdlaBackend::CPU, float);
INSTANTIATE_TRANSPORT_BLOCK(DdlaBackend::CPU, double);
INSTANTIATE_TRANSPORT_BLOCK(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_TRANSPORT_BLOCK(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_TRANSPORT_BLOCK(DdlaBackend::GPU, float);
INSTANTIATE_TRANSPORT_BLOCK(DdlaBackend::GPU, double);
INSTANTIATE_TRANSPORT_BLOCK(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_TRANSPORT_BLOCK(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_TRANSPORT_BLOCK

} // namespace ddla
