#ifndef DDLA_STREAM_H
#define DDLA_STREAM_H

#include <ddla/ddla_handle_t.h>
#include <mpi.h>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ddla {

// One layout for every backend and communication mode. Only ddla_stream.cpp
// creates/destroys native resources; CPU handles leave every opaque slot null.
class DdlaStream {
public:
    // -- per-handle device selection (replaces static local_device) ---------
    int local_device = -1;

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

    char major = 'R';

    // -- backend identity (set during ddlaInit) -----------------------------
    // Placeholder until ddlaInit overwrites it; never consulted before that.
    DdlaBackend backend = DdlaBackend::CPU;

    // -- host threading (CPU handles) ---------------------------------------
    // Threads the handle's own host loops (geam) may use.  Set by ddlaSet (host_threads.h): that call is
    // collective over the handle's communicator, which the budget needs.  1 until then.
    int host_threads = 1;

    // -- lifecycle guards ---------------------------------------------------
    bool initialized = false;
    bool destroyed   = false;

    // -- host staging for the GPU_CPU_TUNNEL path ---------------------------
    // Grow-on-demand host buffers reused across comm* calls instead of
    // allocating a fresh std::vector per collective.  Two buffers are needed
    // because commAlltoallv stages both send and receive data simultaneously.
    // ::operator new guarantees alignment suitable for any fundamental type, so
    // reinterpret_cast<T*> is safe for T in
    // {float, double, std::complex<float>, std::complex<double>}.
    std::vector<std::byte> tunnel_host_staging_a;
    std::vector<std::byte> tunnel_host_staging_b;
    // Borrowed only for the lifetime of a workspace execution scope. Nested
    // scopes restore the previous binding; these pointers are never freed here.
    std::byte* tunnel_borrowed_a = nullptr;
    std::byte* tunnel_borrowed_b = nullptr;
    std::size_t tunnel_borrowed_a_bytes = 0;
    std::size_t tunnel_borrowed_b_bytes = 0;
    bool tunnel_borrowed_active = false;

    void* level_queue();
    int getLocalDevice(MPI_Comm local_comm) const;
    void set_local_device(int dev);
    void init(MPI_Comm comm_group, char major);
    void init(int nprows, int npcols, MPI_Comm comm_group, char major);
    void clean();
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
};

} // namespace ddla

#endif // DDLA_STREAM_H
