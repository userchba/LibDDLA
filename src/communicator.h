#ifndef DDLA_COMMUNICATOR_H
#define DDLA_COMMUNICATOR_H

#include <ddla/ddla_handle_t.h>
#include <cstddef>

namespace ddla {

enum class CommScope { Grid, Row, Col };

// Private runtime interface. Scalar templates adapt to these typed-buffer
// operations in comm_traits.h; implementations share the handle's resources.
// Instances are immutable and shared, so selecting a transport allocates no
// strategy object and ordinary calls retain the existing workspace policy.
class Communicator {
protected:
    // Strategies are borrowed shared instances, never deleted through this
    // interface. A trivial destructor avoids exit-handler allocation.
    ~Communicator() = default;
public:
    virtual DdlaCommunication mode() const noexcept = 0;

    static const Communicator* get(DdlaBackend backend, DdlaCommunication mode) noexcept;
    static MPI_Comm mpi_comm(const DdlaHandle_t& h, CommScope scope) noexcept;

    virtual void initialize(const DdlaHandle_t& h) const {}
    virtual void release(const DdlaHandle_t& h, bool collective) const {}
    virtual void group_start(const DdlaHandle_t& h) const;
    virtual void group_end(const DdlaHandle_t& h) const;

    virtual void bcast(const DdlaHandle_t& h, CommScope scope, void* buffer,
                       std::size_t count, MPI_Datatype type, std::size_t width, int root) const = 0;
    virtual void send(const DdlaHandle_t& h, CommScope scope, const void* buffer,
                      std::size_t count, MPI_Datatype type, std::size_t width, int peer) const = 0;
    virtual void recv(const DdlaHandle_t& h, CommScope scope, void* buffer,
                      std::size_t count, MPI_Datatype type, std::size_t width, int peer) const = 0;
    virtual void sendrecv(const DdlaHandle_t& h, CommScope scope,
                          const void* send, std::size_t send_count, int dest,
                          void* recv, std::size_t recv_count, int source,
                          MPI_Datatype type, std::size_t width) const = 0;
    virtual void allreduce(const DdlaHandle_t& h, CommScope scope,
                           const void* send, void* recv, int count,
                           MPI_Datatype type, std::size_t width, MPI_Op op) const = 0;
    virtual void reduce_sum(const DdlaHandle_t& h, CommScope scope,
                            const void* send, void* recv, int count,
                            MPI_Datatype type, std::size_t width, int root) const = 0;
    virtual void allgather(const DdlaHandle_t& h, CommScope scope,
                           const void* send, void* recv, std::size_t count,
                           MPI_Datatype type, std::size_t width) const = 0;
    virtual void alltoallv(const DdlaHandle_t& h, CommScope scope, int nprocs,
                           const void* send, const int* send_counts, const int* send_displs,
                           void* recv, const int* recv_counts, const int* recv_displs,
                           MPI_Datatype type, std::size_t width) const = 0;
};

} // namespace ddla
#endif
