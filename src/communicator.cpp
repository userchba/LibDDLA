#include "communicator.h"
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "mpi_communicator.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ddla {
namespace {

#if DDLA_HAS_GPU
std::size_t message_bytes(std::size_t count, std::size_t width)
{
    if (width && count > std::numeric_limits<std::size_t>::max() / width)
        throw std::length_error("ddla: host communication buffer size overflow");
    return count * width;
}

class HostStagedMpiCommunicator final : public Communicator {
public:
    DdlaCommunication mode() const noexcept override { return DdlaCommunication::HostStagedMPI; }

    void bcast(const DdlaHandle_t& h, CommScope scope, void* buffer, std::size_t count,
               MPI_Datatype type, std::size_t width, int root) const override
    {
        const auto bytes = message_bytes(count, width);
        auto* host = h->host_buffers<std::byte>(bytes)[0];
        const auto comm = mpi_comm(h, scope);
        int rank = 0;
        MPI_CHECK(MPI_Comm_rank(comm, &rank));
        if (rank == root) DDLA_CHECK(ddlaMemcpy(h, host, buffer, bytes));
        DDLA_CHECK(ddlaSynchronize(h));
        MPI_CHECK(MPI_Bcast(host, static_cast<int>(count), type, root, comm));
        DDLA_CHECK(ddlaMemcpy(h, buffer, host, bytes));
        DDLA_CHECK(ddlaSynchronize(h));
    }
    void send(const DdlaHandle_t& h, CommScope scope, const void* buffer, std::size_t count,
              MPI_Datatype type, std::size_t width, int peer) const override
    {
        const auto bytes = message_bytes(count, width);
        auto* host = h->host_buffers<std::byte>(bytes)[0];
        DDLA_CHECK(ddlaMemcpy(h, host, buffer, bytes));
        DDLA_CHECK(ddlaSynchronize(h));
        MPI_CHECK(MPI_Send(host, static_cast<int>(count), type, peer, 0, mpi_comm(h, scope)));
    }
    void recv(const DdlaHandle_t& h, CommScope scope, void* buffer, std::size_t count,
              MPI_Datatype type, std::size_t width, int peer) const override
    {
        const auto bytes = message_bytes(count, width);
        auto* host = h->host_buffers<std::byte>(bytes)[0];
        DDLA_CHECK(ddlaSynchronize(h));
        MPI_CHECK(MPI_Recv(host, static_cast<int>(count), type, peer, 0,
                           mpi_comm(h, scope), MPI_STATUS_IGNORE));
        DDLA_CHECK(ddlaMemcpy(h, buffer, host, bytes));
        DDLA_CHECK(ddlaSynchronize(h));
    }
    void sendrecv(const DdlaHandle_t& h, CommScope scope,
                  const void* send, std::size_t send_count, int dest,
                  void* recv, std::size_t recv_count, int source,
                  MPI_Datatype type, std::size_t width) const override
    {
        const auto send_bytes = message_bytes(send_count, width);
        const auto recv_bytes = message_bytes(recv_count, width);
        auto [hs, hr] = h->host_buffers<std::byte>(send_bytes, recv_bytes);
        if (send_count) DDLA_CHECK(ddlaMemcpy(h, hs, send, send_bytes));
        DDLA_CHECK(ddlaSynchronize(h));
        MPI_CHECK(MPI_Sendrecv(hs, static_cast<int>(send_count), type, dest, 0,
                               hr, static_cast<int>(recv_count), type, source, 0,
                               mpi_comm(h, scope), MPI_STATUS_IGNORE));
        if (recv_count) DDLA_CHECK(ddlaMemcpy(h, recv, hr, recv_bytes));
        DDLA_CHECK(ddlaSynchronize(h));
    }
    void allreduce(const DdlaHandle_t& h, CommScope scope, const void* send, void* recv,
                   int count, MPI_Datatype type, std::size_t width, MPI_Op op) const override
    {
        const auto bytes = message_bytes(count, width);
        auto [hs, hr] = h->host_buffers<std::byte>(bytes, bytes);
        DDLA_CHECK(ddlaMemcpy(h, hs, send, bytes));
        DDLA_CHECK(ddlaSynchronize(h));
        MPI_CHECK(MPI_Allreduce(hs, hr, count, type, op, mpi_comm(h, scope)));
        DDLA_CHECK(ddlaMemcpy(h, recv, hr, bytes));
        DDLA_CHECK(ddlaSynchronize(h));
    }
    void reduce_sum(const DdlaHandle_t& h, CommScope scope, const void* send, void* recv,
                    int count, MPI_Datatype type, std::size_t width, int root) const override
    {
        const auto bytes = message_bytes(count, width);
        auto [hs, hr] = h->host_buffers<std::byte>(bytes, bytes);
        DDLA_CHECK(ddlaMemcpy(h, hs, send, bytes));
        DDLA_CHECK(ddlaSynchronize(h));
        MPI_CHECK(MPI_Reduce(hs, hr, count, type, MPI_SUM, root, mpi_comm(h, scope)));
        const int rank = scope == CommScope::Row ? h->mypcol_ :
                         scope == CommScope::Col ? h->myprow_ : h->myid;
        if (rank == root) {
            DDLA_CHECK(ddlaMemcpy(h, recv, hr, bytes));
            DDLA_CHECK(ddlaSynchronize(h));
        }
    }
    void allgather(const DdlaHandle_t& h, CommScope scope, const void* send, void* recv,
                   std::size_t count, MPI_Datatype type, std::size_t width) const override
    {
        const auto comm = mpi_comm(h, scope);
        int size = 0;
        MPI_CHECK(MPI_Comm_size(comm, &size));
        const auto bytes = message_bytes(count, width);
        const auto total = message_bytes(bytes, size);
        auto [hs, hr] = h->host_buffers<std::byte>(bytes, total);
        DDLA_CHECK(ddlaMemcpy(h, hs, send, bytes));
        DDLA_CHECK(ddlaSynchronize(h));
        MPI_CHECK(MPI_Allgather(hs, static_cast<int>(count), type,
                                hr, static_cast<int>(count), type, comm));
        DDLA_CHECK(ddlaMemcpy(h, recv, hr, total));
        DDLA_CHECK(ddlaSynchronize(h));
    }
    void alltoallv(const DdlaHandle_t& h, CommScope scope, int nprocs,
                   const void* send, const int* send_counts, const int* send_displs,
                   void* recv, const int* recv_counts, const int* recv_displs,
                   MPI_Datatype type, std::size_t width) const override
    {
        // Callers pack contiguous regions in peer-rank order.
        const int send_total = nprocs ? send_displs[nprocs - 1] + send_counts[nprocs - 1] : 0;
        const int recv_total = nprocs ? recv_displs[nprocs - 1] + recv_counts[nprocs - 1] : 0;
        const auto send_bytes = message_bytes(send_total, width);
        const auto recv_bytes = message_bytes(recv_total, width);
        auto [hs, hr] = h->host_buffers<std::byte>(send_bytes, recv_bytes);
        if (send_total) {
            DDLA_CHECK(ddlaMemcpy(h, hs, send, send_bytes));
            DDLA_CHECK(ddlaSynchronize(h));
        }
        MPI_CHECK(MPI_Alltoallv(hs, send_counts, send_displs, type,
                                hr, recv_counts, recv_displs, type, mpi_comm(h, scope)));
        if (recv_total) {
            DDLA_CHECK(ddlaMemcpy(h, recv, hr, recv_bytes));
            DDLA_CHECK(ddlaSynchronize(h));
        }
    }
};

#ifdef DDLA_USE_CCL
bool create_comm(void*& opaque, MPI_Comm group)
{
    int rank = 0, size = 0;
    MPI_CHECK(MPI_Comm_rank(group, &rank));
    MPI_CHECK(MPI_Comm_size(group, &size));
    gpucclUniqueId id;
    int failed = rank == 0 && gpucclGetUniqueId(&id) != gpucclSuccess;
    MPI_CHECK(MPI_Bcast(&failed, 1, MPI_INT, 0, group));
    if (failed) return false;
    MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, group));
    gpucclComm_t native = nullptr;
    failed = gpucclCommInitRank(&native, size, id, rank) != gpucclSuccess;
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, group));
    if (failed) {
        if (native) (void)gpucclCommAbort(native);
        return false;
    }
    opaque = native;
    return true;
}

bool release_comm(void*& opaque, bool abort = false)
{
    if (!opaque) return true;
    const auto native = static_cast<gpucclComm_t>(opaque);
    const auto status = abort ? gpucclCommAbort(native) : gpucclCommDestroy(native);
    opaque = nullptr;
    return status == gpucclSuccess;
}

class CclCommunicator final : public Communicator {
    static gpucclComm_t native(const DdlaHandle_t& h, CommScope scope)
    {
        switch (scope) {
        case CommScope::Row: return static_cast<gpucclComm_t>(h->nccl_row_comm);
        case CommScope::Col: return static_cast<gpucclComm_t>(h->nccl_col_comm);
        default: return static_cast<gpucclComm_t>(h->nccl_comm);
        }
    }
    static gpucclRedOp_t reduction(MPI_Op op)
    {
        if (op == MPI_SUM) return gpucclSum;
        if (op == MPI_PROD) return gpucclProd;
        if (op == MPI_MIN) return gpucclMin;
        if (op == MPI_MAX) return gpucclMax;
        return static_cast<gpucclRedOp_t>(-1);
    }
    static auto reduction_type(MPI_Datatype type, std::size_t& count)
    {
        if (type == MPI_FLOAT) return gpucclFloat32;
        if (type == MPI_DOUBLE) return gpucclFloat64;
        count *= 2;
        if (type == MPI_CXX_FLOAT_COMPLEX) return gpucclFloat32;
        return gpucclFloat64;
    }
public:
    DdlaCommunication mode() const noexcept override { return DdlaCommunication::CCL; }
    void initialize(const DdlaHandle_t& h) const override
    {
        if (h->nccl_comm) return;
        void* temporary[3] = {};
        MPI_Comm groups[] = {h->comm, h->row_comm, h->col_comm};
        for (int i = 0; i < 3; ++i) {
            int failed = !create_comm(temporary[i], groups[i]);
            MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, h->comm));
            if (failed) {
                for (auto& resource : temporary) (void)release_comm(resource, true);
                throw std::runtime_error("ddla: CCL communicator initialization failed");
            }
        }
        h->nccl_comm = temporary[0];
        h->nccl_row_comm = temporary[1];
        h->nccl_col_comm = temporary[2];
    }
    void release(const DdlaHandle_t& h, bool collective) const override
    {
        if (!h->nccl_comm && !h->nccl_row_comm && !h->nccl_col_comm) return;
        int failed = !release_comm(h->nccl_comm);
        failed = !release_comm(h->nccl_row_comm) || failed;
        failed = !release_comm(h->nccl_col_comm) || failed;
        if (collective) MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, h->comm));
        if (failed) throw std::runtime_error("ddla: CCL communicator cleanup failed");
    }
    void group_start(const DdlaHandle_t& h) const override
    {
        CCL_CHECK(gpucclGroupStart());
        Communicator::group_start(h);
    }
    void group_end(const DdlaHandle_t& h) const override
    {
        Communicator::group_end(h);
        CCL_CHECK(gpucclGroupEnd());
    }
    void bcast(const DdlaHandle_t& h, CommScope scope, void* buffer, std::size_t count,
               MPI_Datatype, std::size_t width, int root) const override
    {
        CCL_CHECK(gpucclBcast(buffer, count * width, gpucclInt8, root,
                              native(h, scope), static_cast<gpuStream_t>(h->stream)));
    }
    void send(const DdlaHandle_t& h, CommScope scope, const void* buffer, std::size_t count,
              MPI_Datatype, std::size_t width, int peer) const override
    {
        CCL_CHECK(gpucclSend(buffer, count * width, gpucclInt8, peer,
                             native(h, scope), static_cast<gpuStream_t>(h->stream)));
    }
    void recv(const DdlaHandle_t& h, CommScope scope, void* buffer, std::size_t count,
              MPI_Datatype, std::size_t width, int peer) const override
    {
        CCL_CHECK(gpucclRecv(buffer, count * width, gpucclInt8, peer,
                             native(h, scope), static_cast<gpuStream_t>(h->stream)));
    }
    void sendrecv(const DdlaHandle_t& h, CommScope scope,
                  const void* send, std::size_t send_count, int dest,
                  void* recv, std::size_t recv_count, int source,
                  MPI_Datatype type, std::size_t width) const override
    {
        group_start(h);
        if (send_count) this->send(h, scope, send, send_count, type, width, dest);
        if (recv_count) this->recv(h, scope, recv, recv_count, type, width, source);
        group_end(h);
    }
    void allreduce(const DdlaHandle_t& h, CommScope scope, const void* send, void* recv,
                   int count, MPI_Datatype type, std::size_t, MPI_Op op) const override
    {
        std::size_t components = count;
        const auto datatype = reduction_type(type, components);
        CCL_CHECK(gpucclAllReduce(send, recv, components, datatype, reduction(op),
                                  native(h, scope), static_cast<gpuStream_t>(h->stream)));
    }
    void reduce_sum(const DdlaHandle_t& h, CommScope scope, const void* send, void* recv,
                    int count, MPI_Datatype type, std::size_t, int root) const override
    {
        std::size_t components = count;
        const auto datatype = reduction_type(type, components);
        CCL_CHECK(gpucclReduce(send, recv, components, datatype, gpucclSum, root,
                               native(h, scope), static_cast<gpuStream_t>(h->stream)));
    }
    void allgather(const DdlaHandle_t& h, CommScope scope, const void* send, void* recv,
                   std::size_t count, MPI_Datatype, std::size_t width) const override
    {
        CCL_CHECK(gpucclAllGather(send, recv, count * width, gpucclInt8,
                                  native(h, scope), static_cast<gpuStream_t>(h->stream)));
    }
    void alltoallv(const DdlaHandle_t& h, CommScope scope, int nprocs,
                   const void* send, const int* send_counts, const int* send_displs,
                   void* recv, const int* recv_counts, const int* recv_displs,
                   MPI_Datatype type, std::size_t width) const override
    {
        group_start(h);
        for (int p = 0; p < nprocs; ++p) {
            if (send_counts[p] > 0)
                this->send(h, scope, static_cast<const std::byte*>(send) + std::size_t(send_displs[p]) * width,
                           send_counts[p], type, width, p);
            if (recv_counts[p] > 0)
                this->recv(h, scope, static_cast<std::byte*>(recv) + std::size_t(recv_displs[p]) * width,
                           recv_counts[p], type, width, p);
        }
        group_end(h);
    }
};
#endif
#endif

} // namespace

const Communicator* Communicator::get(DdlaBackend backend, DdlaCommunication mode) noexcept
{
    if (backend == DdlaBackend::CPU)
        return mode == DdlaCommunication::MPI ? &host_mpi_communicator() : nullptr;
#if DDLA_HAS_GPU
    if (backend == DdlaBackend::GPU) {
        static const MpiCommunicator<DdlaBackend::GPU> mpi;
        static const HostStagedMpiCommunicator staged;
        switch (mode) {
        case DdlaCommunication::MPI: return &mpi;
        case DdlaCommunication::HostStagedMPI: return &staged;
#ifdef DDLA_USE_CCL
        case DdlaCommunication::CCL: {
            static const CclCommunicator ccl;
            return &ccl;
        }
#endif
        default: return nullptr;
        }
    }
#else
    (void)mode;
#endif
    return nullptr;
}

} // namespace ddla
