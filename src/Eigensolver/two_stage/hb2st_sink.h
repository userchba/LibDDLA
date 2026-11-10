#ifndef DDLA_HB2ST_SINK_H
#define DDLA_HB2ST_SINK_H

// Public <ddla/ddla.h> first (hipcc default-argument rule, see src/ptran.h).
#include <ddla/ddla.h>

#include <atomic>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ddla {

// ---------------------------------------------------------------------------
// The stage-4 design of pheevd, for BOTH backends: the stage-2 chase's rotation stream is consumed WHILE the chase runs --
// every rank accumulates its row share of Q_band = G_1 ... G_m from the identity (forward, stream order, no communication) --
// and stage 4 is then one finish: W = Q_band * (D * Y), with the row-slab tridiagonal eigenvectors Y broadcast slab by slab,
// and one Grid all-to-all into the 2D block-cyclic Z.  The sink is an internal interface with two implementations:
//   * Hb2stAccumHost<T>  (hb2st_sink_host.cpp) CPU: a planner thread + OpenMP-team workers accumulating cache-blocked tiles
//                                              inside the chase's own team of threads (see pheevd_check for the policy).
// The CPU backend keeps the classic replay (the node-shared wave-ordered stream) as the fallback pheevd picks when the
// accumulation would not pay (few threads per rank, many ranks per node, ...).
// Each implementation binds the scratch of a run (Q_band and its consumer's tables) with its own reset_run and has its own
// finish_rows_run; the interface below is what the chase drives between the two.
// ---------------------------------------------------------------------------

/// Shape of the chase's record window as the consumer sees it.  ring == 0: the plain window (stream position g is slot g).
/// ring > 0: the first @c ring slots are a RING (position g < switch_pos lives in slot g % ring); with an ELASTIC window the chase
/// abandons the ring at stream position switch_pos -- the first time it would otherwise have had to wait for a consumer -- and
/// writes every later position g at slot ext_offset + (g - switch_pos) (a plain, ever-growing area behind the ring).  A blocking
/// ring never switches (switch_pos is null).  @c consumed is this consumer's watermark: it publishes how many records it has
/// copied out.
struct StreamWindow {
    long long ring = 0;
    long long ext_offset = 0;
    const std::atomic<long long>* switch_pos = nullptr;
    std::atomic<long long>* consumed = nullptr;
};

template <typename T>
class Hb2stSink {
public:
    using rt = typename real_type<T>::type;

    virtual ~Hb2stSink() = default;

    /// A ring window is possible only when the consumer reads whole records chunk by chunk, and the ring is sized in
    /// stream_chunk() records.
    virtual bool window_ring_allowed() const = 0;
    virtual long long stream_chunk() const = 0;
    /// true: the chase may leave the ring when it would have to wait (GPU: the backlog buys stage-3 overlap);
    /// false: a BLOCKING ring -- the chase waits for the consumer (CPU: the accumulation is the bottleneck, waiting is free).
    virtual bool elastic_ring() const { return true; }

    /// Start consuming, on the scratch the implementation's reset_run bound.  @p rots is the record store (the node-team
    /// window or the caller's stable local buffer; it must stay valid until drain_stage2_run() or cancel_run() returned),
    /// @p emitted the chase's published record count, @p aborted (may be null) the chase team's abort flag.  @p win
    /// (optional, copied) describes a ring window.  @p chase_threads is the number of threads this rank's chase team runs
    /// (0: this rank runs no member); @p roles_in_team: the caller runs run_role(0 .. extra_roles()-1) as extra threads of the
    /// chase team's parallel region (false: the sink runs its roles on its own threads).
    virtual void start_stager_run(const hb2st_rot<T>* rots, long long rots_capacity, const std::atomic<long long>* emitted,
                                  const StreamWindow* win, const std::atomic<int>* aborted, int chase_threads,
                                  bool roles_in_team) = 0;

    /// Threads the sink wants as extra members of the chase team's region (valid after start_stager_run); 0 = it owns its
    /// threads.
    virtual int extra_roles() const { return 0; }
    /// Body of extra role @p role (0 <= role < extra_roles()), run on a thread of the chase team's region; returns when the
    /// stream is consumed (or the sink failed).  Never throws.
    virtual void run_role(int role) { (void)role; }

    /// The chase has published its last record (it returned, or failed): lets the consumer drain and exit.
    virtual void mark_chase_done() = 0;
    /// Published-record counter for a chase that has no shared window (serial / threaded fallback).
    virtual std::atomic<long long>* fallback_emitted() = 0;
    /// Application-thread completion boundary after mark_chase_done(): joins the sink's threads and completes its background /
    /// device work while Q_band is retained for stage 4.  Rethrows the consumer's error.  Exceptions must enter the caller's
    /// collective failure decision before more MPI.
    virtual void drain_stage2_run() = 0;
    /// Abort path: marks the chase done, stops and joins the consumer, drains outstanding device work.  Does not throw.
    virtual void cancel_run() = 0;
    /// true: everything is consumed when the chase returns (the CPU sink: the roles ran inside the chase region).
    virtual bool drains_in_stage2() const { return false; }

    /// The consumer's exception (invariant violation / device error), if any.
    virtual const std::exception_ptr& consumer_error() const = 0;
    /// Records the schedule expects (== the stream length a completed chase publishes).
    virtual long long scheduled_rotations() const = 0;
};

} // namespace ddla

#endif // DDLA_HB2ST_SINK_H
