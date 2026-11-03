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
#include "ddla_stream.h"
#include "comm_traits.h"
#include "check_agree.h"
#include "plan_util.h"
#include "transport_block.h"
#include "BLAS/blas_impl.h"

namespace ddla {

inline const char* transport_block_backend_name(DdlaBackend backend)
{
    return backend == DdlaBackend::CPU ? "CPU" : "GPU";
}

namespace transport_impl {

// Component 1.
template <DdlaBackend Backend>
void transport_check(const DdlaHandle_t& handle, char sData, char trans,
                              int m, int n, const int* array_descA)
{
    detail::agree_check(handle, "transport_block", [&](detail::Agree&) {
        if (handle == nullptr) {
            throw std::runtime_error("transport_block: null handle");
        }
        const DdlaBackend actual_backend = ddlaGetBackend(handle);
        if (actual_backend != Backend) {
            throw std::runtime_error(
                std::string("transport_block: template backend ") +
                transport_block_backend_name(Backend) +
                " does not match descriptor handle backend " +
                transport_block_backend_name(actual_backend));
        }
        if (sData != 'C' && sData != 'R')
            throw std::invalid_argument("transport_block: sData must be 'C' or 'R'");
        if (trans != 'N' && trans != 'T' && trans != 'C')
            throw std::invalid_argument("transport_block: trans must be 'N', 'T' or 'C'");
        if (m < 0 || n < 0)
            throw std::invalid_argument("transport_block: negative panel extent");
        check_desc(array_descA, handle);
    });
}

// Component 2.  Only the transposed path on a non-square process grid stages
// anything.  There a rank sends at most the entries of the panel's source
// rows/columns it owns -- m x locmax(n, NB, npcols) entries for a row panel,
// n x locmax(m, MB, nprows) for a column panel, wherever the panel starts --
// and the schedule holds one block per distinct (MB|NB)-block of the panel.
// Backend: the send staging.  Host: the three block lists (all blocks that
// move, the ones this rank sends, the ones it receives), in carving order.
template <typename T>
void transport_workspace_size(const DdlaHandle_t& handle, char sData, char trans,
                              int m_max, int n_max, const int* array_descA,
                              std::size_t* backend_bytes, std::size_t* host_bytes,
                              int target_row, int target_col)
{
    *backend_bytes = *host_bytes = 0;
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    if (trans == 'N') return;
    // A square grid redistributes only when its target differs from the crossed source.
    const bool row_moves = nprows != npcols || (target_row >= 0 && target_row != array_descA[DDLA_CSRC_]);
    const bool col_moves = nprows != npcols || (target_col >= 0 && target_col != array_descA[DDLA_RSRC_]);

    std::size_t send = 0;
    int blocks = 0;
    if (sData != 'C' && row_moves) {
        send = std::max(send, std::size_t(m_max) * detail::locmax(n_max, array_descA[DDLA_NB_], npcols));
        blocks = std::max(blocks, detail::tilesmax(n_max, array_descA[DDLA_NB_]));
    }
    if (sData != 'R' && col_moves) {
        send = std::max(send, std::size_t(n_max) * detail::locmax(m_max, array_descA[DDLA_MB_], nprows));
        blocks = std::max(blocks, detail::tilesmax(m_max, array_descA[DDLA_MB_]));
    }
    *backend_bytes = send * sizeof(T);
    *host_bytes = 3 * detail::padded_bytes<RectBlock>(blocks);
}

// ---------------------------------------------------------------------------
// Unified transport_block<Backend,T>: merges the formerly GPU-only
// transport_block.cpp (NCCL/RCCL/host-tunnel communication, the cuBLAS geam
// transpose-copy, runtime* device memcpy/alloc) and the formerly CPU-only
// transport_panel_cpu (plain blocking MPI, manual transpose+conjugate loops,
// std::malloc/memcpy) hand-duplicated inside pgemm.cpp. The two were already
// branch-for-branch identical control flow (same split on trans=='N' vs
// transposed, same split on square vs non-square process grid, same index
// math via num_loc/indxg2p/indxg2l/rc_to_rank) -- only a few leaf operations
// genuinely differ, and none of them needs an `if constexpr` here any more:
// each is a single call into a unified primitive that owns the split itself:
//   copy_block     -> ddlaMemcpy2D                  (ddla_handle_t.h)
//   pack_transpose -> ddla::geam<Backend,T>         (geam.cpp)
// The send staging and the schedule of the non-square transposed path are
// not allocated per call: they come from the caller's scratch
// (transport_workspace_size).
// All collective communication (broadcast/send/recv/group) goes through the
// CommTraits<Backend>-based comm* primitives in comm_traits.h, which already
// select the right MPI/NCCL/RCCL/tunnel mechanism per backend.
// ---------------------------------------------------------------------------
// Component 3.
template <DdlaBackend Backend, typename T>
void transport_run(
    const DdlaHandle_t& handle, char sData, char trans,
    int m, int n,
    const T* A, int ia, int ja, const int* array_descA,
    T* block_A,
    void* backend_work, [[maybe_unused]] std::size_t backend_bytes,
    void* host_work, [[maybe_unused]] std::size_t host_bytes,
    int target_row, int target_col)
{
    try {
        if (m == 0 || n == 0) return;

        DdlaHandle_t h = handle;
        int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
        ddlaGetGridDims(handle, nprows, npcols);
        ddlaGetGridCoords(handle, myprow, mypcol);
        const int myrank = h->myid;

        // ---- divergent leaf operations: the only genuine CPU/GPU differences --
        // Plain column-contiguous block copy: dst[j*dst_pitch + i] <- src[i + j*src_ld].
        auto copy_block = [&](T* dst, int dst_pitch, const T* src, int rows, int cols, int src_ld) {
            if (rows <= 0 || cols <= 0) return;
            DDLA_CHECK(ddlaMemcpy2D(h, dst, static_cast<std::size_t>(dst_pitch) * sizeof(T), src,
                                    static_cast<std::size_t>(src_ld) * sizeof(T),
                                    static_cast<std::size_t>(rows) * sizeof(T), static_cast<std::size_t>(cols)));
        };
        // Transposed (and, for 'C', conjugated) pack:
        // dst[j + i*dst_stride] <- op(src)[i + j*src_ld], for i in [0,rows), j in [0,cols).
        auto pack_transpose = [&](T* dst, int dst_stride, const T* src, int rows, int cols, int src_ld) {
            const char op = (trans == 'C') ? 'C' : 'T';
            blas_impl::geam_run<Backend,T>(h,op,'N',cols,rows,T(1),src,src_ld,T(0),nullptr,1,dst,dst_stride);
        };
        // ---- index math: identical across both formerly-separate implementations
        int i_loc = num_loc(ia, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
        int j_loc = num_loc(ja, array_descA[DDLA_NB_], mypcol, array_descA[DDLA_CSRC_], npcols);
        int m_loc = num_loc(ia + m, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
        int n_loc = num_loc(ja + n, array_descA[DDLA_NB_], mypcol, array_descA[DDLA_CSRC_], npcols);
        int owner_row = indxg2p(ia, array_descA[DDLA_MB_], array_descA[DDLA_RSRC_], nprows);
        int owner_col = indxg2p(ja, array_descA[DDLA_NB_], array_descA[DDLA_CSRC_], npcols);
        target_row = target_row < 0 ? array_descA[DDLA_CSRC_] : target_row;
        target_col = target_col < 0 ? array_descA[DDLA_RSRC_] : target_col;

        if (trans == 'N') {
            if (sData == 'R' && n_loc > j_loc) {
                int cols = n_loc - j_loc;
                if (myprow == owner_row)
                    copy_block(block_A, m, A + i_loc + (std::size_t)j_loc * array_descA[DDLA_LLD_], m, cols, array_descA[DDLA_LLD_]);
                commBcast<Backend>(h, CommScope::Col, block_A, (std::size_t)m * cols, owner_row);
            } else if (sData == 'C' && m_loc > i_loc) {
                int rows = m_loc - i_loc;
                if (mypcol == owner_col)
                    copy_block(block_A, rows, A + i_loc + (std::size_t)j_loc * array_descA[DDLA_LLD_], rows, n, array_descA[DDLA_LLD_]);
                commBcast<Backend>(h, CommScope::Row, block_A, (std::size_t)rows * n, owner_col);
            }
        } else if (nprows == npcols &&
                   (sData == 'R' ? target_row == array_descA[DDLA_CSRC_] : target_col == array_descA[DDLA_RSRC_])) {
            // ---- square-grid transposed path -----------------------------------
            int trans_j_loc = num_loc(ja, array_descA[DDLA_NB_], myprow, array_descA[DDLA_CSRC_], nprows);
            int trans_n_loc = num_loc(ja + n, array_descA[DDLA_NB_], myprow, array_descA[DDLA_CSRC_], nprows);
            int trans_i_loc = num_loc(ia, array_descA[DDLA_MB_], mypcol, array_descA[DDLA_RSRC_], npcols);
            int trans_m_loc = num_loc(ia + m, array_descA[DDLA_MB_], mypcol, array_descA[DDLA_RSRC_], npcols);

            if (sData == 'R') {
                if (n_loc > j_loc) {
                    int cols = n_loc - j_loc;
                    if (myprow == owner_row) {
                        copy_block(block_A, m, A + i_loc + (std::size_t)j_loc * array_descA[DDLA_LLD_], m, cols, array_descA[DDLA_LLD_]);
                        if (myprow != mypcol)
                            commSend<Backend>(h, CommScope::Col, block_A, (std::size_t)m * cols, mypcol);
                    } else if (myprow == mypcol) {
                        commRecv<Backend>(h, CommScope::Col, block_A, (std::size_t)m * cols, owner_row);
                    }
                }
                if (trans_n_loc > trans_j_loc) {
                    int tcols = trans_n_loc - trans_j_loc;
                    commBcast<Backend>(h, CommScope::Row, block_A, (std::size_t)tcols * m, myprow);
                }
            } else { // sData == 'C'
                if (m_loc > i_loc) {
                    int rows = m_loc - i_loc;
                    if (mypcol == owner_col) {
                        pack_transpose(block_A, n, A + i_loc + (std::size_t)j_loc * array_descA[DDLA_LLD_], rows, n, array_descA[DDLA_LLD_]);
                        if (myprow != mypcol)
                            commSend<Backend>(h, CommScope::Row, block_A, (std::size_t)rows * n, myprow);
                    } else if (myprow == mypcol) {
                        commRecv<Backend>(h, CommScope::Row, block_A, (std::size_t)rows * n, owner_col);
                    }
                }
                if (trans_m_loc > trans_i_loc) {
                    int trows = trans_m_loc - trans_i_loc;
                    commBcast<Backend>(h, CommScope::Col, block_A, (std::size_t)trows * n, mypcol);
                }
            }
        } else {
            // ---- general transposed redistribution, including remapped squares --
            auto block_end = [](int g, int block_size, int end) -> int {
                return std::min(end, (g / block_size + 1) * block_size);
            };
            // The scratch: the send staging, and the schedule of this call -- all
            // blocks that move, the ones this rank sends, the ones it receives.
            T* const sendbuf = static_cast<T*>(backend_work);
            const int tiles = sData == 'R' ? detail::tilesmax(n, array_descA[DDLA_NB_])
                                           : detail::tilesmax(m, array_descA[DDLA_MB_]);
            char* w = static_cast<char*>(host_work);
            RectBlock* const blocks = detail::take<RectBlock>(w, tiles);
            RectBlock* const send_blocks = detail::take<RectBlock>(w, tiles);
            RectBlock* const recv_blocks = detail::take<RectBlock>(w, tiles);
            assert(std::size_t(w - static_cast<char*>(host_work)) <= host_bytes);

            if (sData == 'R') {
                const int target_j_loc = num_loc(ja, array_descA[DDLA_NB_], myprow, target_row, nprows);
                const int target_n_loc = num_loc(ja + n, array_descA[DDLA_NB_], myprow, target_row, nprows);
                const int target_cols = target_n_loc - target_j_loc;
                std::size_t block_count = 0, send_count = 0, recv_count = 0;
                int send_total = 0;

                for (int g = ja; g < ja + n; ) {
                    const int len = block_end(g, array_descA[DDLA_NB_], ja + n) - g;
                    const int src_col = indxg2p(g, array_descA[DDLA_NB_], array_descA[DDLA_CSRC_], npcols);
                    const int dst_row = indxg2p(g, array_descA[DDLA_NB_], target_row, nprows);
                    const int src_rank = h->rc_to_rank(owner_row, src_col);
                    const int dst_rank = h->rc_to_rank(dst_row, 0);
                    const int dst_col_loc = num_loc(g, array_descA[DDLA_NB_], dst_row, target_row, nprows)
                                           - num_loc(ja, array_descA[DDLA_NB_], dst_row, target_row, nprows);
                    RectBlock block{src_rank, dst_rank, m * len, -1, dst_col_loc * m,
                                    indxg2l(g, array_descA[DDLA_NB_], npcols), len};

                    if (src_rank == dst_rank) {
                        if (myrank == src_rank)
                            copy_block(block_A + block.dst_offset, m,
                                       A + i_loc + (std::size_t)block.src_loc * array_descA[DDLA_LLD_], m, block.len, array_descA[DDLA_LLD_]);
                    } else {
                        blocks[block_count++] = block;
                        if (myrank == src_rank) {
                            block.send_offset = send_total;
                            send_total += block.count;
                            send_blocks[send_count++] = block;
                        }
                        if (myrank == dst_rank) {
                            recv_blocks[recv_count++] = block;
                        }
                    }
                    g += len;
                }

                assert(std::size_t(send_total) * sizeof(T) <= backend_bytes);
                for (std::size_t b = 0; b < send_count; ++b) {
                    const RectBlock& block = send_blocks[b];
                    copy_block(sendbuf + block.send_offset, m,
                               A + i_loc + (std::size_t)block.src_loc * array_descA[DDLA_LLD_], m, block.len, array_descA[DDLA_LLD_]);
                }

                if (send_count || recv_count) {
                    commGroupStart<Backend>(h);
                    for (std::size_t b = 0; b < block_count; ++b) {
                        const RectBlock& block = blocks[b];
                        if (myrank == block.src_rank) {
                            auto it = std::find_if(send_blocks, send_blocks + send_count, [&](const RectBlock& item) {
                                return item.dst_rank == block.dst_rank && item.dst_offset == block.dst_offset;
                            });
                            assert(it != send_blocks + send_count);
                            commSend<Backend>(h, CommScope::Grid, sendbuf + it->send_offset,
                                              (std::size_t)block.count, block.dst_rank);
                        } else if (myrank == block.dst_rank) {
                            commRecv<Backend>(h, CommScope::Grid, block_A + block.dst_offset,
                                              (std::size_t)block.count, block.src_rank);
                        }
                    }
                    commGroupEnd<Backend>(h);
                }

                if (target_cols > 0)
                    commBcast<Backend>(h, CommScope::Row, block_A, (std::size_t)m * target_cols, 0);
            } else { // sData == 'C'
                const int target_i_loc = num_loc(ia, array_descA[DDLA_MB_], mypcol, target_col, npcols);
                const int target_m_loc = num_loc(ia + m, array_descA[DDLA_MB_], mypcol, target_col, npcols);
                const int target_cols = target_m_loc - target_i_loc;
                std::size_t block_count = 0, send_count = 0, recv_count = 0;
                int send_total = 0;

                for (int g = ia; g < ia + m; ) {
                    const int len = block_end(g, array_descA[DDLA_MB_], ia + m) - g;
                    const int src_row = indxg2p(g, array_descA[DDLA_MB_], array_descA[DDLA_RSRC_], nprows);
                    const int dst_col = indxg2p(g, array_descA[DDLA_MB_], target_col, npcols);
                    const int src_rank = h->rc_to_rank(src_row, owner_col);
                    const int dst_rank = h->rc_to_rank(0, dst_col);
                    const int dst_col_loc = num_loc(g, array_descA[DDLA_MB_], dst_col, target_col, npcols)
                                           - num_loc(ia, array_descA[DDLA_MB_], dst_col, target_col, npcols);
                    RectBlock block{src_rank, dst_rank, n * len, -1, dst_col_loc * n,
                                    indxg2l(g, array_descA[DDLA_MB_], nprows), len};

                    if (src_rank == dst_rank) {
                        if (myrank == src_rank)
                            pack_transpose(block_A + block.dst_offset, n,
                                           A + (std::size_t)block.src_loc + (std::size_t)j_loc * array_descA[DDLA_LLD_],
                                           block.len, n, array_descA[DDLA_LLD_]);
                    } else {
                        blocks[block_count++] = block;
                        if (myrank == src_rank) {
                            block.send_offset = send_total;
                            send_total += block.count;
                            send_blocks[send_count++] = block;
                        }
                        if (myrank == dst_rank) {
                            recv_blocks[recv_count++] = block;
                        }
                    }
                    g += len;
                }

                assert(std::size_t(send_total) * sizeof(T) <= backend_bytes);
                for (std::size_t b = 0; b < send_count; ++b) {
                    const RectBlock& block = send_blocks[b];
                    pack_transpose(sendbuf + block.send_offset, n,
                                   A + (std::size_t)block.src_loc + (std::size_t)j_loc * array_descA[DDLA_LLD_],
                                   block.len, n, array_descA[DDLA_LLD_]);
                }

                if (send_count || recv_count) {
                    commGroupStart<Backend>(h);
                    for (std::size_t b = 0; b < block_count; ++b) {
                        const RectBlock& block = blocks[b];
                        if (myrank == block.src_rank) {
                            auto it = std::find_if(send_blocks, send_blocks + send_count, [&](const RectBlock& item) {
                                return item.dst_rank == block.dst_rank && item.dst_offset == block.dst_offset;
                            });
                            assert(it != send_blocks + send_count);
                            commSend<Backend>(h, CommScope::Grid, sendbuf + it->send_offset,
                                              (std::size_t)block.count, block.dst_rank);
                        } else if (myrank == block.dst_rank) {
                            commRecv<Backend>(h, CommScope::Grid, block_A + block.dst_offset,
                                              (std::size_t)block.count, block.src_rank);
                        }
                    }
                    commGroupEnd<Backend>(h);
                }

                if (target_cols > 0)
                    commBcast<Backend>(h, CommScope::Col, block_A, (std::size_t)n * target_cols, 0);
            }
        }
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(handle) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(handle); } catch (...) {}
        throw;
    }
}

} // namespace transport_impl

// The public entry: 1. check -> 2. size -> 3. run.  The empty panel returns
// before the check, as it always has.
template <DdlaBackend Backend, typename T>
void transport_block(
    const DdlaHandle_t& handle, char sData, char trans,
    int m, int n,
    const T* A, int ia, int ja, const int* array_descA,
    T* block_A)
{
    if (m == 0 || n == 0) return;
    transport_impl::transport_check<Backend>(handle, sData, trans, m, n, array_descA);
    std::size_t backend_bytes = 0, host_bytes = 0;
    transport_impl::transport_workspace_size<T>(handle, sData, trans, m, n, array_descA,
                                                &backend_bytes, &host_bytes);
    vector<char> backend_work(handle, backend_bytes);
    std::vector<char> host_work(host_bytes);
    transport_impl::transport_run<Backend, T>(handle, sData, trans, m, n,
                                              A, ia, ja, array_descA, block_A,
                                              backend_work.data(), backend_bytes,
                                              host_work.data(), host_bytes);
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------
#define INSTANTIATE_TRANSPORT_BLOCK(BACKEND, TYPE)                          \
    template void transport_block<BACKEND, TYPE>(const DdlaHandle_t&,       \
        char, char, int, int,                                               \
        const TYPE*, int, int, const int*, TYPE*);                          \
    template void transport_impl::transport_run<BACKEND, TYPE>(             \
        const DdlaHandle_t&, char, char, int, int, const TYPE*, int, int,   \
        const int*, TYPE*, void*, std::size_t, void*, std::size_t, int, int)

#define INSTANTIATE_TRANSPORT_CHECK(BACKEND)                                \
    template void                                  \
    transport_impl::transport_check<BACKEND>(const DdlaHandle_t&,           \
        char, char, int, int, const int*)

#define INSTANTIATE_TRANSPORT_SIZE(TYPE)                                    \
    template void transport_impl::transport_workspace_size<TYPE>(           \
        const DdlaHandle_t&, char, char, int, int, const int*,              \
        std::size_t*, std::size_t*, int, int)

INSTANTIATE_TRANSPORT_CHECK(DdlaBackend::CPU);
INSTANTIATE_TRANSPORT_BLOCK(DdlaBackend::CPU, float);
INSTANTIATE_TRANSPORT_BLOCK(DdlaBackend::CPU, double);
INSTANTIATE_TRANSPORT_BLOCK(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_TRANSPORT_BLOCK(DdlaBackend::CPU, std::complex<double>);

#if DDLA_HAS_GPU
INSTANTIATE_TRANSPORT_CHECK(DdlaBackend::GPU);
INSTANTIATE_TRANSPORT_BLOCK(DdlaBackend::GPU, float);
INSTANTIATE_TRANSPORT_BLOCK(DdlaBackend::GPU, double);
INSTANTIATE_TRANSPORT_BLOCK(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_TRANSPORT_BLOCK(DdlaBackend::GPU, std::complex<double>);
#endif

INSTANTIATE_TRANSPORT_SIZE(float);
INSTANTIATE_TRANSPORT_SIZE(double);
INSTANTIATE_TRANSPORT_SIZE(std::complex<float>);
INSTANTIATE_TRANSPORT_SIZE(std::complex<double>);

#undef INSTANTIATE_TRANSPORT_SIZE
#undef INSTANTIATE_TRANSPORT_CHECK
#undef INSTANTIATE_TRANSPORT_BLOCK

} // namespace ddla
