#ifndef DDLA_COMM_TRAITS_H
#define DDLA_COMM_TRAITS_H

#include "ddla_connector.h"
#include "ddla_stream.h"
#include "mpi_datatype.h"
#include <vector>
#include <cstddef>
#include <algorithm>
#include <climits>
#include <type_traits>

// ---------------------------------------------------------------------------
// CommTraits<DdlaBackend Backend>: a DdlaBackend-templated abstraction over
// LibDDLA's communication layer, replacing the old per-file
// `#ifdef DDLA_USE_CCL ncclComm_t X = handle->nccl_X; #else MPI_Comm X = handle->X; #endif`
// aliasing idiom with real functions. Lives directly in namespace ddla (not nested under
// ddla::detail) -- this header is already private (src/, not include/ddla/),
// so an additional detail nesting inside an already-internal header would be
// redundant.
// ---------------------------------------------------------------------------
namespace ddla {

// ---------------------------------------------------------------------------
// MPI_Allreduce maps each scalar type to its MPI_Op reduction datatype.
// The raw NCCL/RCCL and host-staged GPU_CPU_TUNNEL send/recv/bcast/allreduce
// primitives are inlined directly in the comm* wrappers below (no separate
// ccl* helper functions remain -- formerly include/ddla/ddla_comm.h).
// ---------------------------------------------------------------------------

inline int MPI_Allreduce(const float* sendbuff, float* recvbuff, int count, MPI_Op op, MPI_Comm comm)
{
    return MPI_Allreduce(sendbuff, recvbuff, count, MPI_FLOAT, op, comm);
}

inline int MPI_Allreduce(const std::complex<float>* sendbuff, std::complex<float>* recvbuff, int count, MPI_Op op, MPI_Comm comm)
{
    return MPI_Allreduce(sendbuff, recvbuff, count * 2, MPI_FLOAT, op, comm);
}

inline int MPI_Allreduce(const double* sendbuff, double* recvbuff, int count, MPI_Op op, MPI_Comm comm)
{
    return MPI_Allreduce(sendbuff, recvbuff, count, MPI_DOUBLE, op, comm);
}

inline int MPI_Allreduce(const std::complex<double>* sendbuff, std::complex<double>* recvbuff, int count, MPI_Op op, MPI_Comm comm)
{
    return MPI_Allreduce(sendbuff, recvbuff, count, MPI_DOUBLE_COMPLEX, op, comm);
}

// In dual builds (GPU vendor + DDLA_HAS_CPU) a single TU can instantiate both
// the CCL and the CPU path of commAllReduce; there cclOp is ncclRedOp_t and
// must be translated before it reaches MPI.
inline MPI_Op to_mpi_op(MPI_Op op) { return op; }

#ifdef DDLA_USE_CCL
// This must NOT throw: it is called from inside a collective, so a rank that
// threw would skip the MPI_Allreduce while the others entered it, and the job
// would hang instead of reporting the error.  An unsupported reduction falls
// back to MPI_OP_NULL, which MPI rejects with an ordinary error return on
// every rank -- a collective failure rather than a deadlock.  The library only
// ever produces the four standard reductions.
inline MPI_Op to_mpi_op(gpucclRedOp_t op)
{
    if (op == gpucclSum)  return MPI_SUM;
    if (op == gpucclProd) return MPI_PROD;
    if (op == gpucclMin)  return MPI_MIN;
    if (op == gpucclMax)  return MPI_MAX;
    return MPI_OP_NULL;
}
#endif

// ---------------------------------------------------------------------------
// Host staging for the GPU_CPU_TUNNEL path.  These return a typed pointer
// into a handle-owned, grow-on-demand byte buffer (see DdlaStream), so the
// tunnel collectives below avoid allocating a fresh std::vector per call.
// Each comm* wrapper is synchronous (blocking MPI plus a stream sync where
// an async host<->device copy follows), so a buffer is never live across two
// wrappers and reuse is safe.
//
// Caveats:
//  - The buffers grow to the largest size seen and never shrink (a
//    high-water mark), which is the intended allocation-amortization tradeoff.
//  - The buffers are owned per DdlaHandle_t and are therefore NOT safe for
//    concurrent host-side use of the same handle from multiple threads;
//    LibDDLA handles are single-threaded per rank today.
// ---------------------------------------------------------------------------
namespace detail {

template <typename T>
inline T* tunnel_staging(DdlaHandle_t h, int which, std::size_t count)
{
    std::vector<std::byte>& buf = (which == 0) ? h->tunnel_host_staging_a
                                               : h->tunnel_host_staging_b;
    const std::size_t need = (count ? count : 1) * sizeof(T);
    if (buf.size() < need) {
        buf.resize(need);
    }
    return reinterpret_cast<T*>(buf.data());
}

} // namespace detail

enum class CommScope { Grid, Row, Col };

template <DdlaBackend Backend> struct CommTraits;   // primary: intentionally undefined

template <>
struct CommTraits<DdlaBackend::CPU> {
    using comm_t = MPI_Comm;
    static comm_t comm(const DdlaHandle_t& h, CommScope s) {
        switch (s) {
            case CommScope::Row: return h->row_comm;
            case CommScope::Col: return h->col_comm;
            default:             return h->comm;
        }
    }
    static void group_start(const DdlaHandle_t&) {}
    static void group_end  (const DdlaHandle_t&) {}
};

#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
// Only *defined* in a TU that actually compiled in a GPU vendor.
template <>
struct CommTraits<DdlaBackend::GPU> {
#if defined(DDLA_USE_GPU_CPU_TUNNEL)
    // Tunnel wins if both DDLA_USE_CCL and DDLA_USE_GPU_CPU_TUNNEL are set
    // (matches this project's existing precedence elsewhere, e.g.
    // transport_block.cpp's `#if DDLA_USE_CCL && !DDLA_USE_GPU_CPU_TUNNEL`
    // group-batch guard).
    using comm_t = MPI_Comm;
    static comm_t comm(const DdlaHandle_t& h, CommScope s) {
        switch (s) {
            case CommScope::Row: return h->row_comm;
            case CommScope::Col: return h->col_comm;
            default:             return h->comm;
        }
    }
    static void group_start(const DdlaHandle_t&) {}
    static void group_end  (const DdlaHandle_t&) {}
#elif defined(DDLA_USE_CCL)
    using comm_t = gpucclComm_t;
    static comm_t comm(const DdlaHandle_t& h, CommScope s) {
        switch (s) {
            case CommScope::Row: return static_cast<gpucclComm_t>(h->nccl_row_comm);
            case CommScope::Col: return static_cast<gpucclComm_t>(h->nccl_col_comm);
            default:             return static_cast<gpucclComm_t>(h->nccl_comm);
        }
    }
    static void group_start(const DdlaHandle_t&) { CCL_CHECK(gpucclGroupStart()); }
    static void group_end  (const DdlaHandle_t&) { CCL_CHECK(gpucclGroupEnd());   }
#else
    // "Dead" configuration: GPU vendor compiled in, but neither DDLA_USE_CCL
    // nor DDLA_USE_GPU_CPU_TUNNEL set. Preserves today's direct-MPI-on-
    // device-pointer behavior verbatim (the inlined `#else` branches of the
    // comm* wrappers below) -- unsupported/untested, but unchanged from
    // before this refactor, so no regression.
    using comm_t = MPI_Comm;
    static comm_t comm(const DdlaHandle_t& h, CommScope s) {
        switch (s) {
            case CommScope::Row: return h->row_comm;
            case CommScope::Col: return h->col_comm;
            default:             return h->comm;
        }
    }
    static void group_start(const DdlaHandle_t&) {}
    static void group_end  (const DdlaHandle_t&) {}
#endif
};
#endif // defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)

// ---------------------------------------------------------------------------
// comm* function templates. Backend defaults to detail::local_backend_v (the
// per-TU constant already used by the runtime* family in ddla_connector.h),
// so GPU-only callers can invoke these bracket-free, e.g.
// `commBcast(h, CommScope::Row, buf, count, root)`, exactly like they already
// call `ddlaMalloc(...)` bracket-free today.
//
// Non-tunnel GPU calls select, by the very same macro that picked
// CommTraits<GPU>::comm_t's type above, between the inlined NCCL/RCCL
// implementation (DDLA_USE_CCL) and the "dead config" direct-MPI-on-device-
// pointer implementation (neither CCL nor tunnel) -- comm_traits.h does not
// need to distinguish those two cases itself beyond the #if/#elif/#else
// branches below.
//
// The tunnel case stages through handle-owned host scratch buffers (see
// detail::tunnel_staging below) -- callers never see or manage a host
// pointer; the D2H copy, blocking MPI call, and H2D copy are inlined directly
// in each wrapper below. Because the tunnel H2D copy is asynchronous, an extra
// ddlaSynchronize is added after it so the copy completes before the
// scratch buffer is reused by the next wrapper (in commBcast/commRecv/
// commAllReduce). commSend needs no such extra sync: MPI_Send is a blocking
// call, so the host buffer is safe to reuse the instant it returns.
// ---------------------------------------------------------------------------

template <DdlaBackend Backend = detail::local_backend_v>
inline void commGroupStart(const DdlaHandle_t& h) { CommTraits<Backend>::group_start(h); }

template <DdlaBackend Backend = detail::local_backend_v>
inline void commGroupEnd(const DdlaHandle_t& h) { CommTraits<Backend>::group_end(h); }

// Host broadcasts may use a prepared MPI datatype for strided storage.
// GPU collectives require contiguous buffers and use the typed overload below.
template <DdlaBackend Backend = detail::local_backend_v, typename T>
inline void commBcast(const DdlaHandle_t& h, CommScope scope, T* buf, int count,
                      MPI_Datatype datatype, int root)
{
    static_assert(Backend == DdlaBackend::CPU, "MPI datatypes require a CPU broadcast");
    MPI_CHECK(MPI_Bcast(buf, count, datatype, root, CommTraits<Backend>::comm(h, scope)));
}

template <DdlaBackend Backend = detail::local_backend_v, typename T>
inline void commBcast(const DdlaHandle_t& h, CommScope scope, T* buf, std::size_t count, int root)
{
    if constexpr (Backend == DdlaBackend::CPU) {
        commBcast<Backend>(h, scope, buf, static_cast<int>(count), detail::mpi_datatype<T>(), root);
    } else {
#if defined(DDLA_USE_GPU_CPU_TUNNEL)
        T* host = detail::tunnel_staging<T>(h, 0, count);
        const auto comm = CommTraits<Backend>::comm(h, scope);
        int rank;
        MPI_CHECK(MPI_Comm_rank(comm, &rank));
        // Receive buffers need not contain initialized values before Bcast.
        if (rank == root) DDLA_CHECK(ddlaMemcpy(h, host, buf, count * sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
        MPI_CHECK(MPI_Bcast(host, (int)count, detail::mpi_datatype<T>(), root, comm));
        DDLA_CHECK(ddlaMemcpy(h, buf, host, count * sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
#elif defined(DDLA_USE_CCL)
        CCL_CHECK(gpucclBcast(buf, count * sizeof(T), gpucclInt8, root, CommTraits<Backend>::comm(h, scope), static_cast<gpuStream_t>(h->stream)));
#else
        DDLA_CHECK(ddlaSynchronize(h));
        MPI_CHECK(MPI_Bcast(buf, (int)count, detail::mpi_datatype<T>(), root, CommTraits<Backend>::comm(h, scope)));
#endif
    }
}

template <DdlaBackend Backend = detail::local_backend_v, typename T>
inline void commSend(const DdlaHandle_t& h, CommScope scope, const T* buf, std::size_t count, int peer)
{
    if constexpr (Backend == DdlaBackend::CPU) {
        MPI_CHECK(MPI_Send(buf, (int)count, detail::mpi_datatype<T>(), peer, 0,
                           CommTraits<Backend>::comm(h, scope)));
    } else {
#if defined(DDLA_USE_GPU_CPU_TUNNEL)
        T* host = detail::tunnel_staging<T>(h, 0, count);
        DDLA_CHECK(ddlaMemcpy(h, host, buf, count * sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
        MPI_CHECK(MPI_Send(host, (int)count, detail::mpi_datatype<T>(), peer, 0, CommTraits<Backend>::comm(h, scope)));
#elif defined(DDLA_USE_CCL)
        CCL_CHECK(gpucclSend(buf, count * sizeof(T), gpucclInt8, peer, CommTraits<Backend>::comm(h, scope), static_cast<gpuStream_t>(h->stream)));
#else
        DDLA_CHECK(ddlaSynchronize(h));
        MPI_CHECK(MPI_Send(buf, (int)count, detail::mpi_datatype<T>(), peer, 0, CommTraits<Backend>::comm(h, scope)));
#endif
    }
}

template <DdlaBackend Backend = detail::local_backend_v, typename T>
inline void commRecv(const DdlaHandle_t& h, CommScope scope, T* buf, std::size_t count, int peer)
{
    if constexpr (Backend == DdlaBackend::CPU) {
        MPI_CHECK(MPI_Recv(buf, (int)count, detail::mpi_datatype<T>(), peer, 0,
                           CommTraits<Backend>::comm(h, scope), MPI_STATUS_IGNORE));
    } else {
#if defined(DDLA_USE_GPU_CPU_TUNNEL)
        T* host = detail::tunnel_staging<T>(h, 0, count);
        DDLA_CHECK(ddlaSynchronize(h));
        MPI_CHECK(MPI_Recv(host, (int)count, detail::mpi_datatype<T>(), peer, 0, CommTraits<Backend>::comm(h, scope), MPI_STATUS_IGNORE));
        DDLA_CHECK(ddlaMemcpy(h, buf, host, count * sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
#elif defined(DDLA_USE_CCL)
        CCL_CHECK(gpucclRecv(buf, count * sizeof(T), gpucclInt8, peer, CommTraits<Backend>::comm(h, scope), static_cast<gpuStream_t>(h->stream)));
#else
        DDLA_CHECK(ddlaSynchronize(h));
        MPI_CHECK(MPI_Recv(buf, (int)count, detail::mpi_datatype<T>(), peer, 0, CommTraits<Backend>::comm(h, scope), MPI_STATUS_IGNORE));
#endif
    }
}

// A local transpose can send to one rank and receive a different-sized
// region from another. MPI_Sendrecv also handles cycles longer than two.
template <DdlaBackend Backend = detail::local_backend_v, typename T>
inline void commSendrecv(const DdlaHandle_t& h, CommScope scope,
    const T* sbuf, std::size_t scount, int dest, T* rbuf, std::size_t rcount, int source)
{
    if constexpr (Backend == DdlaBackend::CPU) {
        MPI_CHECK(MPI_Sendrecv(sbuf, (int)scount, detail::mpi_datatype<T>(), dest, 0,
                               rbuf, (int)rcount, detail::mpi_datatype<T>(), source, 0,
                               CommTraits<Backend>::comm(h, scope), MPI_STATUS_IGNORE));
    } else {
#if defined(DDLA_USE_GPU_CPU_TUNNEL)
        T* hs = detail::tunnel_staging<T>(h, 0, scount);
        T* hr = detail::tunnel_staging<T>(h, 1, rcount);
        if (scount) DDLA_CHECK(ddlaMemcpy(h, hs, sbuf, scount * sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
        MPI_CHECK(MPI_Sendrecv(hs, (int)scount, detail::mpi_datatype<T>(), dest, 0,
                               hr, (int)rcount, detail::mpi_datatype<T>(), source, 0,
                               CommTraits<Backend>::comm(h, scope), MPI_STATUS_IGNORE));
        if (rcount) DDLA_CHECK(ddlaMemcpy(h, rbuf, hr, rcount * sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
#elif defined(DDLA_USE_CCL)
        CommTraits<Backend>::group_start(h);
        if (scount) commSend<Backend>(h, scope, sbuf, scount, dest);
        if (rcount) commRecv<Backend>(h, scope, rbuf, rcount, source);
        CommTraits<Backend>::group_end(h);
#else
        DDLA_CHECK(ddlaSynchronize(h));
        MPI_CHECK(MPI_Sendrecv(sbuf, (int)scount, detail::mpi_datatype<T>(), dest, 0,
                               rbuf, (int)rcount, detail::mpi_datatype<T>(), source, 0,
                               CommTraits<Backend>::comm(h, scope), MPI_STATUS_IGNORE));
#endif
    }
}

// Simultaneous exchange with ONE peer: both ranks call this with the same count, each sends sbuf and receives rbuf.  The CCL path
// issues the pair inside a group; every MPI path uses MPI_Sendrecv, because two ranks that both enter a blocking MPI_Send first
// (what commGroupStart/commSend/commRecv/commGroupEnd degrades to on a CPU handle) hang once the message exceeds the eager limit.
template <DdlaBackend Backend = detail::local_backend_v, typename T>
inline void commSendrecv(const DdlaHandle_t& h, CommScope scope, const T* sbuf, T* rbuf, std::size_t count, int peer)
{
    if constexpr (Backend == DdlaBackend::CPU) {
        MPI_CHECK(MPI_Sendrecv(sbuf, (int)count, detail::mpi_datatype<T>(), peer, 0,
                               rbuf, (int)count, detail::mpi_datatype<T>(), peer, 0,
                               CommTraits<Backend>::comm(h, scope), MPI_STATUS_IGNORE));
    } else {
#if defined(DDLA_USE_GPU_CPU_TUNNEL)
        T* hs = detail::tunnel_staging<T>(h, 0, count);
        T* hr = detail::tunnel_staging<T>(h, 1, count);
        DDLA_CHECK(ddlaMemcpy(h, hs, sbuf, count * sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
        MPI_CHECK(MPI_Sendrecv(hs, (int)count, detail::mpi_datatype<T>(), peer, 0,
                               hr, (int)count, detail::mpi_datatype<T>(), peer, 0,
                               CommTraits<Backend>::comm(h, scope), MPI_STATUS_IGNORE));
        DDLA_CHECK(ddlaMemcpy(h, rbuf, hr, count * sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
#elif defined(DDLA_USE_CCL)
        CommTraits<Backend>::group_start(h);
        commSend<Backend>(h, scope, sbuf, count, peer);
        commRecv<Backend>(h, scope, rbuf, count, peer);
        CommTraits<Backend>::group_end(h);
#else
        DDLA_CHECK(ddlaSynchronize(h));
        MPI_CHECK(MPI_Sendrecv(sbuf, (int)count, detail::mpi_datatype<T>(), peer, 0,
                               rbuf, (int)count, detail::mpi_datatype<T>(), peer, 0,
                               CommTraits<Backend>::comm(h, scope), MPI_STATUS_IGNORE));
#endif
    }
}

template <DdlaBackend Backend = detail::local_backend_v, typename T>
inline void commAllReduce(const DdlaHandle_t& h, CommScope scope,
                          const T* sbuf, T* rbuf, int count, cclOp op)
{
    if constexpr (Backend == DdlaBackend::CPU) {
        MPI_CHECK(MPI_Allreduce(sbuf, rbuf, count, to_mpi_op(op),
                                CommTraits<Backend>::comm(h, scope)));
    } else {
#if defined(DDLA_USE_GPU_CPU_TUNNEL)
        T* hs = detail::tunnel_staging<T>(h, 0, count);
        T* hr = detail::tunnel_staging<T>(h, 1, count);
        DDLA_CHECK(ddlaMemcpy(h, hs, sbuf, count * sizeof(T)));
        // MPI_Allreduce overwrites hr; no copy of the undefined output is needed.
        DDLA_CHECK(ddlaSynchronize(h));
        // In TUNNEL+CCL builds cclOp is ncclRedOp_t and must be translated
        // before it reaches MPI (identity for MPI_Op-only builds).
        MPI_CHECK(MPI_Allreduce(hs, hr, count, to_mpi_op(op), CommTraits<Backend>::comm(h, scope)));
        DDLA_CHECK(ddlaMemcpy(h, rbuf, hr, count * sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
#elif defined(DDLA_USE_CCL)
        if constexpr (std::is_same_v<T, float>) {
            CCL_CHECK(gpucclAllReduce(sbuf, rbuf, count, gpucclFloat32, op, CommTraits<Backend>::comm(h, scope), static_cast<gpuStream_t>(h->stream)));
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            CCL_CHECK(gpucclAllReduce(sbuf, rbuf, count * 2, gpucclFloat32, op, CommTraits<Backend>::comm(h, scope), static_cast<gpuStream_t>(h->stream)));
        } else if constexpr (std::is_same_v<T, double>) {
            CCL_CHECK(gpucclAllReduce(sbuf, rbuf, count, gpucclFloat64, op, CommTraits<Backend>::comm(h, scope), static_cast<gpuStream_t>(h->stream)));
        } else {
            CCL_CHECK(gpucclAllReduce(sbuf, rbuf, count * 2, gpucclFloat64, op, CommTraits<Backend>::comm(h, scope), static_cast<gpuStream_t>(h->stream)));
        }
#else
        DDLA_CHECK(ddlaSynchronize(h));
        MPI_CHECK(MPI_Allreduce(sbuf, rbuf, count, op, CommTraits<Backend>::comm(h, scope)));
#endif
    }
}

// Sum to one owner. The root may pass its send buffer as the receive buffer
// (an in-place reduction). Only the root's receive buffer is defined
// afterwards. Unlike all-reduce this does not replicate C.
template <DdlaBackend Backend = detail::local_backend_v, typename T>
inline void commReduceSum(const DdlaHandle_t& h, CommScope scope,
                          const T* sbuf, T* rbuf, std::size_t count, int root)
{
    const int rank = scope == CommScope::Row ? h->mypcol_ :
                     scope == CommScope::Col ? h->myprow_ : h->myid;
    // MPI spells an in-place root contribution MPI_IN_PLACE; NCCL and the
    // host-staged path accept sbuf == rbuf as they are.
    const bool in_place = rank == root && sbuf == rbuf;
    // Bound MPI element counts and complex NCCL real-component counts.
    for (std::size_t off = 0; off < count; ) {
        const int len = static_cast<int>(std::min(count - off, std::size_t(INT_MAX / 2)));
        if constexpr (Backend == DdlaBackend::CPU) {
            MPI_CHECK(::MPI_Reduce(in_place ? MPI_IN_PLACE : sbuf + off, rbuf + off, len,
                                  detail::mpi_datatype<T>(), MPI_SUM, root,
                                  CommTraits<Backend>::comm(h, scope)));
        } else {
#if defined(DDLA_USE_GPU_CPU_TUNNEL)
            T* hs = detail::tunnel_staging<T>(h, 0, len);
            T* hr = detail::tunnel_staging<T>(h, 1, len);
            DDLA_CHECK(ddlaMemcpy(h, hs, sbuf + off, sizeof(T) * len));
            DDLA_CHECK(ddlaSynchronize(h));
            MPI_CHECK(::MPI_Reduce(hs, hr, len, detail::mpi_datatype<T>(), MPI_SUM,
                                  root, CommTraits<Backend>::comm(h, scope)));
            if (rank == root) {
                DDLA_CHECK(ddlaMemcpy(h, rbuf + off, hr, sizeof(T) * len));
                DDLA_CHECK(ddlaSynchronize(h));
            }
#elif defined(DDLA_USE_CCL)
            constexpr bool single = std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>;
            constexpr int components = std::is_same_v<T, float> || std::is_same_v<T, double> ? 1 : 2;
            CCL_CHECK(gpucclReduce(sbuf + off, rbuf + off, std::size_t(len) * components,
                                  single ? gpucclFloat32 : gpucclFloat64, gpucclSum,
                                  root, CommTraits<Backend>::comm(h, scope), static_cast<gpuStream_t>(h->stream)));
#else
            DDLA_CHECK(ddlaSynchronize(h));
            MPI_CHECK(::MPI_Reduce(in_place ? MPI_IN_PLACE : sbuf + off, rbuf + off, len,
                                  detail::mpi_datatype<T>(), MPI_SUM, root,
                                  CommTraits<Backend>::comm(h, scope)));
#endif
        }
        off += len;
    }
}

template <DdlaBackend Backend = detail::local_backend_v, typename T>
inline void commAlltoallv(const DdlaHandle_t& h, CommScope scope, int nprocs,
                          const T* sendbuf, const int* sendcounts, const int* sdispls,
                          T* recvbuf, const int* recvcounts, const int* rdispls)
{
    if constexpr (Backend == DdlaBackend::CPU) {
        MPI_CHECK(MPI_Alltoallv(sendbuf, sendcounts, sdispls, detail::mpi_datatype<T>(),
                                recvbuf, recvcounts, rdispls, detail::mpi_datatype<T>(),
                                CommTraits<Backend>::comm(h, scope)));
    } else {
#if defined(DDLA_USE_CCL) && !defined(DDLA_USE_GPU_CPU_TUNNEL)
        // One message per peer, aggregating every block destined for/arriving
        // from that peer -- correct because send_blocks/recv_blocks are laid
        // out contiguously per peer (matches the sort-by-peer-rank ordering
        // callers already use to build sendcounts/sdispls/recvcounts/rdispls).
        CommTraits<Backend>::group_start(h);
        for (int p = 0; p < nprocs; ++p) {
            if (sendcounts[p] > 0)
                CCL_CHECK(gpucclSend(sendbuf + sdispls[p], sendcounts[p] * sizeof(T), gpucclInt8, p,
                                     CommTraits<Backend>::comm(h, scope), static_cast<gpuStream_t>(h->stream)));
            if (recvcounts[p] > 0)
                CCL_CHECK(gpucclRecv(recvbuf + rdispls[p], recvcounts[p] * sizeof(T), gpucclInt8, p,
                                     CommTraits<Backend>::comm(h, scope), static_cast<gpuStream_t>(h->stream)));
        }
        CommTraits<Backend>::group_end(h);
#else
        // Tunnel (and "dead config" without CCL): host-staged single
        // MPI_Alltoallv collective.
        int send_total = nprocs > 0 ? sdispls[nprocs - 1] + sendcounts[nprocs - 1] : 0;
        int recv_total = nprocs > 0 ? rdispls[nprocs - 1] + recvcounts[nprocs - 1] : 0;
        T* h_send = detail::tunnel_staging<T>(h, 0, static_cast<std::size_t>(send_total));
        T* h_recv = detail::tunnel_staging<T>(h, 1, static_cast<std::size_t>(recv_total));
        if (send_total > 0) {
            DDLA_CHECK(ddlaMemcpy(h, h_send, sendbuf, sizeof(T) * send_total));
            DDLA_CHECK(ddlaSynchronize(h));
        }
        MPI_CHECK(MPI_Alltoallv(h_send, sendcounts, sdispls, detail::mpi_datatype<T>(),
                                h_recv, recvcounts, rdispls, detail::mpi_datatype<T>(),
                                CommTraits<Backend>::comm(h, scope)));
        if (recv_total > 0) {
            DDLA_CHECK(ddlaMemcpy(h, recvbuf, h_recv, sizeof(T) * recv_total));
            DDLA_CHECK(ddlaSynchronize(h));
        }
#endif
    }
}

} // namespace ddla

#endif // DDLA_COMM_TRAITS_H
