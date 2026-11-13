#ifndef DDLA_MPI_COMMUNICATOR_H
#define DDLA_MPI_COMMUNICATOR_H

#include "ddla_stream.h"
#include "ddla_connector.h"

namespace ddla {

// Keep the host-MPI implementation visible to typed CPU calls. Its final
// class can be inlined even though GPU transports use the virtual interface.
template <DdlaBackend Backend>
class MpiCommunicator final : public Communicator {
    static void synchronize(const DdlaHandle_t& h)
    {
        if constexpr (Backend == DdlaBackend::GPU) DDLA_CHECK(ddlaSynchronize(h));
    }
public:
    DdlaCommunication mode() const noexcept override { return DdlaCommunication::MPI; }

    void bcast(const DdlaHandle_t& h, CommScope scope, void* buffer, std::size_t count,
               MPI_Datatype type, std::size_t, int root) const override
    {
        synchronize(h);
        MPI_CHECK(MPI_Bcast(buffer, static_cast<int>(count), type, root, mpi_comm(h, scope)));
    }
    void send(const DdlaHandle_t& h, CommScope scope, const void* buffer, std::size_t count,
              MPI_Datatype type, std::size_t, int peer) const override
    {
        synchronize(h);
        MPI_CHECK(MPI_Send(buffer, static_cast<int>(count), type, peer, 0, mpi_comm(h, scope)));
    }
    void recv(const DdlaHandle_t& h, CommScope scope, void* buffer, std::size_t count,
              MPI_Datatype type, std::size_t, int peer) const override
    {
        synchronize(h);
        MPI_CHECK(MPI_Recv(buffer, static_cast<int>(count), type, peer, 0,
                           mpi_comm(h, scope), MPI_STATUS_IGNORE));
    }
    void sendrecv(const DdlaHandle_t& h, CommScope scope,
                  const void* send, std::size_t send_count, int dest,
                  void* recv, std::size_t recv_count, int source,
                  MPI_Datatype type, std::size_t) const override
    {
        synchronize(h);
        MPI_CHECK(MPI_Sendrecv(send, static_cast<int>(send_count), type, dest, 0,
                               recv, static_cast<int>(recv_count), type, source, 0,
                               mpi_comm(h, scope), MPI_STATUS_IGNORE));
    }
    void allreduce(const DdlaHandle_t& h, CommScope scope, const void* send, void* recv,
                   int count, MPI_Datatype type, std::size_t, MPI_Op op) const override
    {
        synchronize(h);
        MPI_CHECK(MPI_Allreduce(send == recv ? MPI_IN_PLACE : send, recv,
                                count, type, op, mpi_comm(h, scope)));
    }
    void reduce_sum(const DdlaHandle_t& h, CommScope scope, const void* send, void* recv,
                    int count, MPI_Datatype type, std::size_t, int root) const override
    {
        synchronize(h);
        const int rank = scope == CommScope::Row ? h->mypcol_ :
                         scope == CommScope::Col ? h->myprow_ : h->myid;
        MPI_CHECK(MPI_Reduce(rank == root && send == recv ? MPI_IN_PLACE : send,
                             recv, count, type, MPI_SUM, root, mpi_comm(h, scope)));
    }
    void allgather(const DdlaHandle_t& h, CommScope scope, const void* send, void* recv,
                   std::size_t count, MPI_Datatype type, std::size_t) const override
    {
        synchronize(h);
        MPI_CHECK(MPI_Allgather(send, static_cast<int>(count), type,
                                recv, static_cast<int>(count), type, mpi_comm(h, scope)));
    }
    void alltoallv(const DdlaHandle_t& h, CommScope scope, int,
                   const void* send, const int* send_counts, const int* send_displs,
                   void* recv, const int* recv_counts, const int* recv_displs,
                   MPI_Datatype type, std::size_t) const override
    {
        synchronize(h);
        MPI_CHECK(MPI_Alltoallv(send, send_counts, send_displs, type,
                                recv, recv_counts, recv_displs, type, mpi_comm(h, scope)));
    }
};

inline const MpiCommunicator<DdlaBackend::CPU>& host_mpi_communicator() noexcept
{
    static const MpiCommunicator<DdlaBackend::CPU> mpi;
    return mpi;
}

} // namespace ddla
#endif
