#ifndef DDLA_COMM_TRAITS_H
#define DDLA_COMM_TRAITS_H

#include "ddla_connector.h"
#include "ddla_stream.h"
#include "mpi_communicator.h"
#include "mpi_datatype.h"
#include <algorithm>
#include <climits>
#include <type_traits>

namespace ddla {

// CPU template calls are also used for host metadata on GPU handles. They
// always use host MPI; only GPU-buffer calls use the handle's selected strategy.
template <DdlaBackend Backend>
inline const auto& communicator(const DdlaHandle_t& h)
{
    if constexpr (Backend == DdlaBackend::CPU) return host_mpi_communicator();
    else return *h->communicator;
}

inline MPI_Op to_mpi_op(MPI_Op op) { return op; }
#ifdef DDLA_USE_CCL
// Unknown reductions must reach the collective as invalid, rather than
// throwing locally while the other ranks enter a blocking reduction.
inline MPI_Op to_mpi_op(gpucclRedOp_t op)
{
    if (op == gpucclSum) return MPI_SUM;
    if (op == gpucclProd) return MPI_PROD;
    if (op == gpucclMin) return MPI_MIN;
    if (op == gpucclMax) return MPI_MAX;
    return MPI_OP_NULL;
}
#endif

template <DdlaBackend Backend = detail::local_backend_v>
inline void commGroupStart(const DdlaHandle_t& h) { communicator<Backend>(h).group_start(h); }

template <DdlaBackend Backend = detail::local_backend_v>
inline void commGroupEnd(const DdlaHandle_t& h) { communicator<Backend>(h).group_end(h); }

// Only CPU broadcasts accept derived MPI datatypes, including strided panels.
template <DdlaBackend Backend = detail::local_backend_v, typename T>
inline void commBcast(const DdlaHandle_t& h, CommScope scope, T* buf, int count,
                      MPI_Datatype datatype, int root)
{
    static_assert(Backend == DdlaBackend::CPU, "MPI datatypes require a CPU broadcast");
    communicator<Backend>(h).bcast(h, scope, buf, count, datatype, 0, root);
}

template <DdlaBackend Backend = detail::local_backend_v, typename T>
inline void commBcast(const DdlaHandle_t& h, CommScope scope, T* buf, std::size_t count, int root)
{
    communicator<Backend>(h).bcast(h, scope, buf, count, detail::mpi_datatype<T>(), sizeof(T), root);
}

template <DdlaBackend Backend = detail::local_backend_v, typename T>
inline void commSend(const DdlaHandle_t& h, CommScope scope, const T* buf, std::size_t count, int peer)
{
    communicator<Backend>(h).send(h, scope, buf, count, detail::mpi_datatype<T>(), sizeof(T), peer);
}

template <DdlaBackend Backend = detail::local_backend_v, typename T>
inline void commRecv(const DdlaHandle_t& h, CommScope scope, T* buf, std::size_t count, int peer)
{
    communicator<Backend>(h).recv(h, scope, buf, count, detail::mpi_datatype<T>(), sizeof(T), peer);
}

// MPI uses Sendrecv for cycles above the eager limit; CCL implements the
// simultaneous exchange with grouped sends/receives in its derived class.
template <DdlaBackend Backend = detail::local_backend_v, typename T>
inline void commSendrecv(const DdlaHandle_t& h, CommScope scope,
                         const T* send, std::size_t send_count, int dest,
                         T* recv, std::size_t recv_count, int source)
{
    communicator<Backend>(h).sendrecv(h, scope, send, send_count, dest, recv, recv_count, source,
                                      detail::mpi_datatype<T>(), sizeof(T));
}

template <DdlaBackend Backend = detail::local_backend_v, typename T>
inline void commSendrecv(const DdlaHandle_t& h, CommScope scope,
                         const T* send, T* recv, std::size_t count, int peer)
{
    commSendrecv<Backend>(h, scope, send, count, peer, recv, count, peer);
}

template <DdlaBackend Backend = detail::local_backend_v, typename T>
inline void commAllReduce(const DdlaHandle_t& h, CommScope scope,
                          const T* send, T* recv, int count, cclOp op)
{
    // Retain the existing complex-float componentwise MPI reduction and the
    // complex-double MPI datatype. CCL maps complex-double to real components.
    if constexpr (std::is_same_v<T, std::complex<float>>)
        communicator<Backend>(h).allreduce(h, scope, send, recv, count * 2,
                                            MPI_FLOAT, sizeof(float), to_mpi_op(op));
    else {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                      std::is_same_v<T, std::complex<double>>, "unsupported allreduce scalar type");
        const auto type = std::is_same_v<T, std::complex<double>> ? MPI_DOUBLE_COMPLEX : detail::mpi_datatype<T>();
        communicator<Backend>(h).allreduce(h, scope, send, recv, count, type, sizeof(T), to_mpi_op(op));
    }
}

template <DdlaBackend Backend = detail::local_backend_v, typename T>
inline void commReduceSum(const DdlaHandle_t& h, CommScope scope,
                          const T* send, T* recv, std::size_t count, int root)
{
    // Bound MPI counts and CCL real-component counts without allocating a
    // global reduction buffer. Only the root's output is defined afterwards.
    for (std::size_t offset = 0; offset < count; ) {
        const int length = static_cast<int>(std::min(count - offset, std::size_t(INT_MAX / 2)));
        communicator<Backend>(h).reduce_sum(h, scope, send + offset, recv + offset, length,
                                             detail::mpi_datatype<T>(), sizeof(T), root);
        offset += length;
    }
}

template <DdlaBackend Backend = detail::local_backend_v, typename T>
inline void commAllGather(const DdlaHandle_t& h, CommScope scope,
                          const T* send, T* recv, std::size_t count)
{
    communicator<Backend>(h).allgather(h, scope, send, recv, count, detail::mpi_datatype<T>(), sizeof(T));
}

template <DdlaBackend Backend = detail::local_backend_v, typename T>
inline void commAlltoallv(const DdlaHandle_t& h, CommScope scope, int nprocs,
                          const T* send, const int* send_counts, const int* send_displs,
                          T* recv, const int* recv_counts, const int* recv_displs)
{
    communicator<Backend>(h).alltoallv(h, scope, nprocs, send, send_counts, send_displs,
                                      recv, recv_counts, recv_displs, detail::mpi_datatype<T>(), sizeof(T));
}

} // namespace ddla
#endif
