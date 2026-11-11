#ifndef DDLA_EIG_CPU_KERNELS_H
#define DDLA_EIG_CPU_KERNELS_H

// Host-only OpenMP kernels of the CPU eigensolver path (stage 4 so far).  This translation unit is
// compiled by the plain host C++ compiler in EVERY build (never by nvcc/hipcc), so OpenMP pragmas
// work in dual and GPU builds too; callers in nvcc-compiled files only see these declarations.
//
// Threading contract: kernels copy the host budget frozen at ddlaSet -- OMP_NUM_THREADS
// when the user sets it (an OpenMP-built BLAS follows the same variable), otherwise each CPU
// in this rank's affinity mask is shared among the node ranks allowed to run on it.
// DDLA_EIG_THREADS (>= 1) overrides both.  A build without OpenMP runs every kernel serially.

#include <mpi.h>
#include "host_parallel.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <type_traits>
#include <memory>
#include <new>

#include <ddla/ddla.h>

namespace ddla {
namespace detail {

/// Runs f(0), ..., f(nthreads-1) concurrently, f(w) on thread w of an OpenMP team (so the threads carry the
/// binding OMP_PROC_BIND/OMP_PLACES give them; a std::thread spawned after an OpenMP region would inherit
/// the pinned master's single-core mask).  Returns false WITHOUT running anything when OpenMP is not
/// available or the runtime cannot supply nthreads threads; the caller then uses its own threads.  f must
/// not throw (the team members' bodies catch everything).
bool host_run_team(int nthreads,void* context,void (*function)(void*,int));
template<class F>bool host_run_team(int nthreads,F&& function)
{
    using Callable=typename std::remove_reference<F>::type;
    return host_run_team(nthreads,const_cast<void*>(static_cast<const void*>(std::addressof(function))),
        [](void* context,int role){(*static_cast<Callable*>(context))(role);});
}
bool host_run_team(int nthreads, const std::function<void(int)>& f);

/// Host array that is NOT touched on allocation: zero(nthreads) first-touches it across the threads (and so
/// across the sockets) of a rank, where std::vector would fault every page on the allocating thread's socket.
template <typename T>
class HostArray {
public:
    HostArray() = default;
    HostArray(const HostArray&) = delete;
    HostArray& operator=(const HostArray&) = delete;
    ~HostArray() { std::free(p_); }
    void reset(std::size_t n)
    {
        std::free(p_);
        p_ = nullptr;
        n_ = n;
        if (n == 0) return;
        void* q = nullptr;
        if (posix_memalign(&q, 64, n * sizeof(T)) != 0) throw std::bad_alloc();
        p_ = static_cast<T*>(q);
    }
    void zero(int nthreads)
    {
        T* const p = p_;
        host_parallel_for(static_cast<long long>(n_ * sizeof(T) / 4096 + 1), 256, nthreads,
                          [p, this](long long a, long long b) {
                              const std::size_t lo = static_cast<std::size_t>(a) * 4096;
                              const std::size_t hi = std::min(n_ * sizeof(T), static_cast<std::size_t>(b) * 4096);
                              if (lo < hi) std::memset(reinterpret_cast<unsigned char*>(p) + lo, 0, hi - lo);
                          });
    }
    T* data() { return p_; }
    const T* data() const { return p_; }
    std::size_t size() const { return n_; }

private:
    T* p_ = nullptr;
    std::size_t n_ = 0;
};

/// Stage-4 CPU back-transform: for each of my ncs columns of the replicated tridiagonal eigenvector
/// matrix h_Y (real, column-major, ld ldy, column cols[k]) apply the phases (complex only: row i+1 *=
/// phases[i]) and then the recorded rotations in REVERSE stream order, and write the n rows of the
/// result straight into the Col-alltoallv send buffer: row r lands at sendbuf[sd[pr] + li*ncs + k] with
/// pr = (r/mb)%npr and li = mb*(r/(mb*npr)) + r%mb.  Bitwise identical to the serial reference loop
/// for every thread count and tile width (each column sees the same operations in the same order).
/// h_Y holds y_count reals (a generic RHS slab may have any number of columns): every selected column
/// cols[k]*ldy + n must lie inside it.
template <typename T>
void cpu_replay_pack(const typename real_type<T>::type* h_Y, std::size_t y_count, int ldy, const int* cols, int ncs, int n,
                     const hb2st_rot<T>* rots, std::size_t nrots, const T* phases, int mb, int npr,
                     const int* sd, T* sendbuf, int nthreads);

template <typename T> class DistributedHistory;
/// Bounded reverse replay from cyclic history chunks; collective over comm.
template <typename T>
void cpu_replay_pack_distributed(const typename real_type<T>::type* h_Y, std::size_t y_count, int ldy, const int* cols,
                                 int ncs, int n, const DistributedHistory<T>& history, MPI_Comm comm, const T* phases,
                                 int mb, int npr, const int* sd, T* sendbuf, int nthreads);

/// Test hook: whether the most recent cpu_replay_pack call on this process used the wave-ordered replay (false =
/// plain order: not a chase stream, DDLA_EIG_REPLAY_WAVE=0, or too few columns).
bool cpu_replay_last_used_waves();

// ---- node-shared wave schedule ---------------------------------------------------------------------------
// The replay regroups the rotation stream into dominoes (~0.49 n^2 records, 24 B each in double: 4.7 GB at n = 20000)
// and every rank used to hold its own copy of both the raw stream and the regrouped one -- 40 ranks on a node: ~400 GB.
// The regrouped stream and the wave tables are identical on every rank of a node, so ONE copy per node suffices: the
// leader plans the schedule into shared memory, all ranks of the node regroup their share of the records into one
// shared array, and the replay reads that array (read-only).

/// An MPI shared-memory window (node-local: MPI_Win_allocate_shared over the node communicator).
struct NodeWindow {
    MPI_Win win = MPI_WIN_NULL;
    void* base = nullptr;
};

/// The node-shared wave schedule of one rotation stream: views into two windows.  Freed by release() (COLLECTIVE over
/// the node communicator).  A moved-from or default object is empty (ok == false).
template <typename T>
struct WaveShared {
    bool ok = false;
    std::size_t nw = 0, nrots = 0;
    const int* gjlo = nullptr;
    const int* gcnt = nullptr;
    const std::int64_t* woff = nullptr;
    const hb2st_rot<T>* wrec = nullptr;
    NodeWindow tab, rec;

    /// Frees both windows (COLLECTIVE over the node communicator they were allocated on) and empties the object.
    void release()
    {
        if (tab.win != MPI_WIN_NULL) MPI_Win_free(&tab.win);
        if (rec.win != MPI_WIN_NULL) MPI_Win_free(&rec.win);
        tab.base = rec.base = nullptr;
        ok = false;
        gjlo = gcnt = nullptr;
        woff = nullptr;
        wrec = nullptr;
    }
};

/// Builds the node-shared schedule of the stream @p rots (nrots records; the SAME memory on every rank of the node
/// -- typically the chase's shared window).  COLLECTIVE over @p node (the node communicator): every rank calls it with
/// the same n and nrots.  Returns false -- on every rank alike, leaving @p out empty -- when the stream cannot use the
/// wave order (not a chase stream, DDLA_EIG_REPLAY_WAVE=0, trivial size) or a window could not be allocated; the
/// caller then keeps a private copy of the stream.  On success the plan buffers and nothing else are freed; the raw
/// stream is no longer needed afterwards.  nthreads threads of each rank take part in the regrouping.
template <typename T>
bool cpu_build_node_wave_schedule(MPI_Comm node, int n, const hb2st_rot<T>* rots, std::size_t nrots,
                                  int nthreads, WaveShared<T>& out);

/// cpu_replay_pack over a node-shared schedule (no raw stream needed).  Same arguments and the same bitwise result.
template <typename T>
void cpu_replay_pack_shared(const typename real_type<T>::type* h_Y, std::size_t y_count, int ldy, const int* cols,
                            int ncs, int n, const WaveShared<T>& ws, const T* phases, int mb, int npr, const int* sd,
                            T* sendbuf, int nthreads);

/// Stage-4 CPU scatter: the Col-alltoallv receive buffer holds, for each source process row pr, my
/// m_loc_rows local Z rows for the columns [share_lo[pr], share_lo[pr+1]) of my column set, row after
/// row (w = share_lo[pr+1]-share_lo[pr] contiguous values per row, block at offset rd[pr]).  Write
/// column k (my_cols order) into Z[li + col_lj[k]*lld].
template <typename T>
void cpu_scatter_z(const T* recvbuf, const int* rd, const int* share_lo, int npr, int nlc,
                   const int* col_lj, int m_loc_rows, T* Z, int lld, int nthreads);
// Scatter requires no scratch: validation and execution are the only phases.
template<class T>void cpu_scatter_z_check(const T*,const int*,const int*,int,int,const int*,int,T*,int,int);
template<class T>void cpu_scatter_z_run(const T*,const int*,const int*,int,int,const int*,int,T*,int,int);

} // namespace detail
} // namespace ddla

#endif // DDLA_EIG_CPU_KERNELS_H
