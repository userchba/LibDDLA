#ifndef DDLA_STREAM_H
#define DDLA_STREAM_H

#include <ddla/ddla_handle_t.h>
#include "communicator.h"
#include <mpi.h>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ddla {

// One layout for every backend and communication mode. Only ddla_stream.cpp
// manages compute resources; communicator.cpp manages CCL resources. CPU
// handles leave every opaque slot null.
class DdlaStream {
public:
    // -- per-handle device selection (replaces static local_device) ---------
    int local_device = -1;
    // Ranks of this communicator in the same shared-memory domain, cached
    // by ddlaSet alongside the host budget. Used by collective method policy.
    int node_ranks = 0;

    // -- MPI communicators --------------------------------------------------
    MPI_Comm comm      = MPI_COMM_NULL;   // duplicated from user, owned
    MPI_Comm row_comm  = MPI_COMM_NULL;
    MPI_Comm col_comm  = MPI_COMM_NULL;
    int myid  = -1, nprocs = 0;
    int myprow_ = -1, nprows_ = 0, mypcol_ = -1, npcols_ = 0;

    void* nccl_comm     = nullptr;
    void* nccl_row_comm = nullptr;
    void* nccl_col_comm = nullptr;

    void* stream = nullptr;
    void* stream_data = nullptr;
    // Private LEVEL queue of the eigensolver back-transform (Hb2stAccum), see level_queue().
    void* stream_level = nullptr;
    void* solverH = nullptr;
    void* blasH   = nullptr;

    // Pack these small flags together before the aligned backend/strategy
    // fields, retaining the handle's previous size on the supported ABI.
    char major = 'R';
    bool initialized = false;
    bool destroyed   = false;

    // -- backend identity (set during ddlaInit) -----------------------------
    // Placeholder until ddlaInit overwrites it; never consulted before that.
    DdlaBackend backend = DdlaBackend::CPU;
    const Communicator* communicator = Communicator::get(DdlaBackend::CPU, DdlaCommunication::MPI);
    DdlaCommunication communication() const noexcept { return communicator->mode(); }
    // Internal groups may be rank-local. Transport changes are forbidden until
    // they close; a handle is never used concurrently by application threads.
    int communication_group_depth = 0;

    // -- host threading ----------------------------------------------------
    // Host-loop and eigensolver budget, frozen collectively by ddlaSet for
    // this communicator, affinity and environment. Reconfigure the handle
    // to adopt changed bindings or overrides. Checked plans copy this value.
    int host_threads = 1;

    // Blocking tunnel communication reuses these two per-handle host buffers.
    // Send and receive storage must be disjoint for sendrecv/all-to-all/reduce.
    // Communication calls grow requested slots on demand and retain their
    // high-water sizes; algorithms do not pre-size this cache. Finish GPU
    // copies before another communication call reuses them. Like the handle's
    // streams, this storage is not for concurrent host-thread use.
    template <typename T>
    std::array<T*, 2> host_buffers(std::size_t send_count, std::size_t recv_count = 0)
    {
        const std::array<std::size_t, 2> counts{send_count, recv_count};
        for (const auto count : counts)
            if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
                throw std::length_error("ddla: host communication buffer size overflow");
        std::array<T*, 2> result{};
        for (std::size_t i = 0; i < counts.size(); ++i) {
            auto& buffer = host_buffers_[i];
            const std::size_t bytes = counts[i] * sizeof(T);
            if (buffer.size() < bytes) buffer.resize(bytes);
            // operator new supplies alignment for the supported BLAS scalars.
            // Empty slots may be null; MPI and ddlaMemcpy ignore zero counts.
            result[i] = buffer.empty() ? nullptr : reinterpret_cast<T*>(buffer.data());
        }
        return result;
    }

    void* level_queue();
    int getLocalDevice(MPI_Comm local_comm) const;
    void set_local_device(int dev);
    void init(MPI_Comm comm_group, char major);
    void init(int nprows, int npcols, MPI_Comm comm_group, char major);
    void clean();
    void select_communication(DdlaCommunication mode);
    ~DdlaStream();
    void check_memory();
    void rank_to_rc(int rank, int& row, int& col) const
    {
        if (major == 'R') { row = rank / npcols_; col = rank % npcols_; }
        else if (major == 'C') { row = rank % nprows_; col = rank / nprows_; }
        else throw std::runtime_error("major should be 'R' or 'C'\n");
    }
    int rc_to_rank(int row, int col) const
    {
        if (major == 'R') return row * npcols_ + col;
        if (major == 'C') return col * nprows_ + row;
        throw std::runtime_error("major should be 'R' or 'C'\n");
    }
private:
    std::vector<std::byte> host_buffers_[2];
};

// These topology/group operations need the completed handle layout.
inline MPI_Comm Communicator::mpi_comm(const DdlaHandle_t& h, CommScope scope) noexcept
{
    switch (scope) {
    case CommScope::Row: return h->row_comm;
    case CommScope::Col: return h->col_comm;
    default: return h->comm;
    }
}

inline void Communicator::group_start(const DdlaHandle_t& h) const
{
    ++h->communication_group_depth;
}

inline void Communicator::group_end(const DdlaHandle_t& h) const
{
    if (h->communication_group_depth <= 0)
        throw std::runtime_error("ddla: communication group is not active");
    --h->communication_group_depth;
}

} // namespace ddla

#endif // DDLA_STREAM_H
