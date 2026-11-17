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
// A null optional queue retains compute-queue behavior; a nonnull queue selects
// that queue explicitly for device work. CPU metadata calls ignore it.
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
    virtual void group_start(const DdlaHandle_t& h, void* queue = nullptr) const;
    virtual void group_end(const DdlaHandle_t& h, void* queue = nullptr) const;

    virtual void bcast(const DdlaHandle_t& h, CommScope scope, void* buffer,
                       std::size_t count, MPI_Datatype type, std::size_t width, int root, void* queue = nullptr) const = 0;
    virtual void send(const DdlaHandle_t& h, CommScope scope, const void* buffer,
                      std::size_t count, MPI_Datatype type, std::size_t width, int peer, void* queue = nullptr) const = 0;
    virtual void recv(const DdlaHandle_t& h, CommScope scope, void* buffer,
                      std::size_t count, MPI_Datatype type, std::size_t width, int peer, void* queue = nullptr) const = 0;
    virtual void sendrecv(const DdlaHandle_t& h, CommScope scope,
                          const void* send, std::size_t send_count, int dest,
                          void* recv, std::size_t recv_count, int source,
                          MPI_Datatype type, std::size_t width, void* queue = nullptr) const = 0;
    virtual void allreduce(const DdlaHandle_t& h, CommScope scope,
                           const void* send, void* recv, int count,
                           MPI_Datatype type, std::size_t width, MPI_Op op, void* queue = nullptr) const = 0;
    virtual void reduce_sum(const DdlaHandle_t& h, CommScope scope,
                            const void* send, void* recv, int count,
                            MPI_Datatype type, std::size_t width, int root, void* queue = nullptr) const = 0;
    virtual void allgather(const DdlaHandle_t& h, CommScope scope,
                           const void* send, void* recv, std::size_t count,
                           MPI_Datatype type, std::size_t width, void* queue = nullptr) const = 0;
    virtual void alltoallv(const DdlaHandle_t& h, CommScope scope, int nprocs,
                           const void* send, const int* send_counts, const int* send_displs,
                           void* recv, const int* recv_counts, const int* recv_displs,
                           MPI_Datatype type, std::size_t width, void* queue = nullptr) const = 0;
};

} // namespace ddla
#endif
