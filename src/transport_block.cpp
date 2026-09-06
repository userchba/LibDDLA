#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>
#include "ddla_connector.h"
#include "ddla_stream_impl.h"
#include "comm_traits.h"
#include "geam.h"

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
// genuinely differ, and none of them needs an `if constexpr` here any more:
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
    const DdlaHandle_t& handle, const char& sData, const char& trans,
    const int& m, const int& n,
    const T* d_A, const int& ia, const int& ja, const int* array_descA,
    T* d_block_A)
{
    if (m == 0 || n == 0) return;

    DdlaHandle_t h = handle;
    if (h == nullptr) {
        throw std::runtime_error("transport_block: null handle");
    }
    const DdlaBackend actual_backend = ddlaGetBackend(h);
    if (actual_backend != Backend) {
        throw std::runtime_error(
            std::string("transport_block: template backend ") +
            transport_block_backend_name(Backend) +
            " does not match descriptor handle backend " +
            transport_block_backend_name(actual_backend));
    }

    assert(sData == 'C' || sData == 'R');
    assert(trans == 'N' || trans == 'T' || trans == 'C');

    check_desc(array_descA, handle);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
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
    int i_loc = num_loc(ia, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
    int j_loc = num_loc(ja, array_descA[DDLA_NB_], mypcol, array_descA[DDLA_CSRC_], npcols);
    int m_loc = num_loc(ia + m, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
    int n_loc = num_loc(ja + n, array_descA[DDLA_NB_], mypcol, array_descA[DDLA_CSRC_], npcols);
    int owner_row = indxg2p(ia, array_descA[DDLA_MB_], array_descA[DDLA_RSRC_], nprows);
    int owner_col = indxg2p(ja, array_descA[DDLA_NB_], array_descA[DDLA_CSRC_], npcols);

    if (trans == 'N') {
        if (sData == 'R' && n_loc > j_loc) {
            int cols = n_loc - j_loc;
            if (myprow == owner_row)
                copy_block(d_block_A, m, d_A + i_loc + (std::size_t)j_loc * array_descA[DDLA_LLD_], m, cols, array_descA[DDLA_LLD_]);
            commBcast<Backend>(h, CommScope::Col, d_block_A, (std::size_t)m * cols, owner_row);
        } else if (sData == 'C' && m_loc > i_loc) {
            int rows = m_loc - i_loc;
            if (mypcol == owner_col)
                copy_block(d_block_A, rows, d_A + i_loc + (std::size_t)j_loc * array_descA[DDLA_LLD_], rows, n, array_descA[DDLA_LLD_]);
            commBcast<Backend>(h, CommScope::Row, d_block_A, (std::size_t)rows * n, owner_col);
        }
    } else if (nprows == npcols) {
        // ---- square-grid transposed path -----------------------------------
        int trans_j_loc = num_loc(ja, array_descA[DDLA_NB_], myprow, array_descA[DDLA_CSRC_], nprows);
        int trans_n_loc = num_loc(ja + n, array_descA[DDLA_NB_], myprow, array_descA[DDLA_CSRC_], nprows);
        int trans_i_loc = num_loc(ia, array_descA[DDLA_MB_], mypcol, array_descA[DDLA_RSRC_], npcols);
        int trans_m_loc = num_loc(ia + m, array_descA[DDLA_MB_], mypcol, array_descA[DDLA_RSRC_], npcols);

        if (sData == 'R') {
            if (n_loc > j_loc) {
                int cols = n_loc - j_loc;
                if (myprow == owner_row) {
                    copy_block(d_block_A, m, d_A + i_loc + (std::size_t)j_loc * array_descA[DDLA_LLD_], m, cols, array_descA[DDLA_LLD_]);
                    if (myprow != mypcol)
                        commSend<Backend>(h, CommScope::Col, d_block_A, (std::size_t)m * cols, mypcol);
                } else if (myprow == mypcol) {
                    commRecv<Backend>(h, CommScope::Col, d_block_A, (std::size_t)m * cols, owner_row);
                }
            }
            if (trans_n_loc > trans_j_loc) {
                int tcols = trans_n_loc - trans_j_loc;
                commBcast<Backend>(h, CommScope::Row, d_block_A, (std::size_t)tcols * m, myprow);
            }
        } else { // sData == 'C'
            if (m_loc > i_loc) {
                int rows = m_loc - i_loc;
                if (mypcol == owner_col) {
                    pack_transpose(d_block_A, n, d_A + i_loc + (std::size_t)j_loc * array_descA[DDLA_LLD_], rows, n, array_descA[DDLA_LLD_]);
                    if (myprow != mypcol)
                        commSend<Backend>(h, CommScope::Row, d_block_A, (std::size_t)rows * n, myprow);
                } else if (myprow == mypcol) {
                    commRecv<Backend>(h, CommScope::Row, d_block_A, (std::size_t)rows * n, owner_col);
                }
            }
            if (trans_m_loc > trans_i_loc) {
                int trows = trans_m_loc - trans_i_loc;
                commBcast<Backend>(h, CommScope::Col, d_block_A, (std::size_t)trows * n, mypcol);
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
            const int target_j_loc = num_loc(ja, array_descA[DDLA_NB_], myprow, array_descA[DDLA_CSRC_], nprows);
            const int target_n_loc = num_loc(ja + n, array_descA[DDLA_NB_], myprow, array_descA[DDLA_CSRC_], nprows);
            const int target_cols = target_n_loc - target_j_loc;
            std::vector<RectBlock> blocks, send_blocks, recv_blocks;
            int send_total = 0;

            for (int g = ja; g < ja + n; ) {
                const int len = block_end(g, array_descA[DDLA_NB_], ja + n) - g;
                const int src_col = indxg2p(g, array_descA[DDLA_NB_], array_descA[DDLA_CSRC_], npcols);
                const int dst_row = indxg2p(g, array_descA[DDLA_NB_], array_descA[DDLA_CSRC_], nprows);
                const int src_rank = h->rc_to_rank(owner_row, src_col);
                const int dst_rank = h->rc_to_rank(dst_row, 0);
                const int dst_col_loc = num_loc(g, array_descA[DDLA_NB_], dst_row, array_descA[DDLA_CSRC_], nprows)
                                       - num_loc(ja, array_descA[DDLA_NB_], dst_row, array_descA[DDLA_CSRC_], nprows);
                RectBlock block{src_rank, dst_rank, m * len, -1, dst_col_loc * m,
                                indxg2l(g, array_descA[DDLA_NB_], npcols), len};

                if (src_rank == dst_rank) {
                    if (myrank == src_rank)
                        copy_block(d_block_A + block.dst_offset, m,
                                   d_A + i_loc + (std::size_t)block.src_loc * array_descA[DDLA_LLD_], m, block.len, array_descA[DDLA_LLD_]);
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
                           d_A + i_loc + (std::size_t)block.src_loc * array_descA[DDLA_LLD_], m, block.len, array_descA[DDLA_LLD_]);

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
            const int target_i_loc = num_loc(ia, array_descA[DDLA_MB_], mypcol, array_descA[DDLA_RSRC_], npcols);
            const int target_m_loc = num_loc(ia + m, array_descA[DDLA_MB_], mypcol, array_descA[DDLA_RSRC_], npcols);
            const int target_cols = target_m_loc - target_i_loc;
            std::vector<RectBlock> blocks, send_blocks, recv_blocks;
            int send_total = 0;

            for (int g = ia; g < ia + m; ) {
                const int len = block_end(g, array_descA[DDLA_MB_], ia + m) - g;
                const int src_row = indxg2p(g, array_descA[DDLA_MB_], array_descA[DDLA_RSRC_], nprows);
                const int dst_col = indxg2p(g, array_descA[DDLA_MB_], array_descA[DDLA_RSRC_], npcols);
                const int src_rank = h->rc_to_rank(src_row, owner_col);
                const int dst_rank = h->rc_to_rank(0, dst_col);
                const int dst_col_loc = num_loc(g, array_descA[DDLA_MB_], dst_col, array_descA[DDLA_RSRC_], npcols)
                                       - num_loc(ia, array_descA[DDLA_MB_], dst_col, array_descA[DDLA_RSRC_], npcols);
                RectBlock block{src_rank, dst_rank, n * len, -1, dst_col_loc * n,
                                indxg2l(g, array_descA[DDLA_MB_], nprows), len};

                if (src_rank == dst_rank) {
                    if (myrank == src_rank)
                        pack_transpose(d_block_A + block.dst_offset, n,
                                       d_A + (std::size_t)block.src_loc + (std::size_t)j_loc * array_descA[DDLA_LLD_],
                                       block.len, n, array_descA[DDLA_LLD_]);
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
                               d_A + (std::size_t)block.src_loc + (std::size_t)j_loc * array_descA[DDLA_LLD_],
                               block.len, n, array_descA[DDLA_LLD_]);

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
    template void transport_block<BACKEND, TYPE>(                           const DdlaHandle_t&, \
        const char&, const char&, const int&, const int&,                  \
        const TYPE*, const int&, const int&, const int*, TYPE*)

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
