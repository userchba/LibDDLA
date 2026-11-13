// ---------------------------------------------------------------------------
// STAGE-2 PARALLELISM: the threaded chase is back, behind a knob (2026-09-25).
// The serial chase is now the per-bulge FUSED form (host_chase below); the
// threaded chase host_chase_par still runs the phase-major order and must stay
// bitwise identical to it (.claude_jobs/eig2/chase_fused_check.cpp checks the
// fused serial against the frozen phase-major reference, chase_par_check.cpp
// the threaded chase against the serial).
//
// The chase's active bulges are provably independent WITHIN a phase: a
// debug-only occupancy stamp (compile with -DDLA_EIG_CHASE_STAMP) records every
// band and dc/wc access tagged by (step, phase, bulge) and finds ZERO
// within-phase collisions at every (n, kd) tried.  host_chase_par below splits
// each phase's bulge index across a thread team and is bitwise identical to
// the serial chase at T = 2,3,4,6,8 over n = 64..3000 and both scalar families
// (.claude_jobs/eig2/chase_par_check.cpp).
//
// Two measured lessons from its first life shape the current form:
//
//  1. It pays only at large n and small T: a step holds tens of microseconds
//     of work against ~7 barriers, so T=2 wins (1.87x at n=10000/kd=64) while
//     T>=6 loses.  It was first removed because the benchmark allocation gave
//     a 4-rank job 6 CPUs; the pipeline is now measured at 2 cpus per rank,
//     which is exactly the configuration T=2 needs.
//  2. The threaded body must be a SEPARATE function: as the `nt == 1` arm of
//     one function it cost the serial default 46% (2.17 s -> 3.17 s at
//     n=5000), because the per-phase slice/barrier closures stopped the
//     optimizer hoisting state out of the innermost loop.
//
// So the serial host_chase stays the default and byte-identical (now in its
// per-bulge fused form); host_chase_par runs only when DDLA_EIG_CHASE_THREADS
// >= 2, and the knob is read per call (the bitwise gate flips it with setenv
// between runs).
//
// The stamp is kept: it is the tool that answers the parallelism question, it
// compiles to nothing when the flag is absent, and it is what any future
// band-reduction work should start from.  Measurements and method:
// .claude_jobs/eig2/PERF_FINDINGS.md, section "P2".
// ---------------------------------------------------------------------------
// Stage 2 of the two-stage Hermitian eigensolver: band -> tridiagonal.
// Numerically a faithful re-implementation of the LAPACK ?SBTRD/?HBTRD
// (UPLO='U') Givens chase -- validated step-for-step against the reference
// implementation -- extended to RECORD every Q-side rotation so stage 4
// (punmtr_hb2st) can replay the transform on the eigenvector rows.
//
// Rotation convention (recorded stream):
//   real:    [c  s; -s  c]      x' = c*x + s*y ; y' = c*y - s*x
//   complex: [c  s; -conj(s) c] x' = c*x + s*y ; y' = c*y - conj(s)*x
// The rotation with index j0 mixes coordinate rows (j0-1, j0).
//
// Band storage (matching LAPACK's upper-band array):
//   band[col*(kd+1) + row] = A(col + row - kd, col),  row in [0, kd],
// where kd = nbw is the semi-bandwidth (row = kd is the diagonal).
// phe2hb's output stores the band in the upper triangle of the 2D
// block-cyclic A (entry A(i, j) for j-i in [0, nbw]), so row = kd + i - j.
//
// Distribution contract (v2, 2026-09-27): the band is gathered with one
// Grid-scope all-reduce; the chase then runs ONCE PER NODE, split over the
// node's ranks through a shared-memory window (host_chase_team below --
// bit-identical to the serial chase for any member count).  Only with one
// rank per node (or DDLA_EIG_CHASE_TEAM=0, or a window allocation failure)
// does it run redundantly on every rank (host_chase, or host_chase_par under
// DDLA_EIG_CHASE_THREADS >= 2).  The chase is O(n^2*nbw) host work -- the
// per-bulge fused form below; measurements in
// .claude_jobs/eig2/PERF_FINDINGS.md.
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>
#include "phb2st_impl.h"
#include "check_agree.h"

#include "ddla_connector.h"
#include "comm_traits.h"
#include "ddla_stream.h"
#include "ddla_desc.h"
#include "Eigensolver/eig_cpu_kernels.h"
#include "Eigensolver/eig_params.h"
#include "hb2st_sink.h"
#include "instantiate.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <complex>
#include <cstring>
#include <limits>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#include <sys/resource.h>   // getrusage (team profile, POSIX)

#ifdef DDLA_EIG_CHASE_STAMP
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <unordered_map>
#include <utility>
#endif

namespace ddla {

namespace {

template <typename T>
constexpr bool is_real_v = std::is_same<T, typename real_type<T>::type>::value;

#ifdef DDLA_EIG_CHASE_STAMP
// ---------------------------------------------------------------------------
// Debug-only occupancy stamp (compile with -DDDLA_EIG_CHASE_STAMP).
//
// The parallel chase splits each step's `nr` active bulges across threads.  That
// is only sound if two bulges never touch the same band element within one
// phase -- if they do, no barrier placement can fix it and the phase must be
// serialized instead.  Rather than reason that out (the index arithmetic has
// boundary special cases like `nrt = nr-1` and the `j2+l > n` guards), this
// records which bulge last touched each element and reports every collision
// within a (step, phase).  Same method as dag_rots.cpp used for the replay
// levels: derive the schedule from the code, not from the code's description.
//
// Detection is online: a map from element key to the bulge that touched it, with
// the map cleared at each phase boundary.  An element touched twice in a phase
// by two DIFFERENT bulges is recorded once as a collision.  Band elements are
// keyed (col + 8) * 1024 + row; the bias keeps the small negative columns the
// boundary guards can produce distinct.  dc/wc live in a separate key space.
// Compiled out entirely when the flag is absent.
enum ChaseStampPhase {
    SP_GENERATE = 0, SP_APPLY_RIGHT, SP_ANNIHILATE, SP_ZLAR2V,
    SP_UNCONJ, SP_APPLY_LEFT, SP_RECORD, SP_CREATE, SP_NPHASE
};
inline const char* chase_phase_name(int p)
{
    static const char* names[SP_NPHASE] = {
        "generate", "apply_right", "annihilate", "zlar2v",
        "unconj", "apply_left", "record", "create"
    };
    return (p >= 0 && p < SP_NPHASE) ? names[p] : "?";
}
struct ChaseStamp {
    // Within a phase: which bulge last touched each element, cleared per phase so
    // a hit by two different bulges is a within-phase conflict -- unfixable by
    // barriers, and the reason to check this at all.
    std::unordered_map<long, int> owner;
    long collisions = 0;
    std::set<std::pair<int, int>> colliding;      // (phase, |t1-t2|)

    // Across phases: the last bulge to touch each element, NOT cleared per
    // phase, so a later phase's access to an element last touched by another
    // bulge in an earlier phase is a cross-phase dependency.  `phase_pair`
    // accumulates ((phase_a, phase_b), |t_a-t_b|) for exactly those, which is
    // the barrier set: a barrier is needed between any two phases that appear
    // here with a nonzero gap.
    std::unordered_map<long, std::pair<int, int>> last;   // key -> (phase, bulge)
    std::set<std::pair<std::pair<int, int>, int>> phase_pair;

    int step = -1;
    int phase = SP_GENERATE;
    int bulge = -1;
    bool enabled = false;
    long touched = 0;
    long cross = 0;

    void phase_begin(int p) { phase = p; owner.clear(); }
    void rec(long key)
    {
        ++touched;
        auto it = owner.find(key);
        if (it == owner.end()) owner.emplace(key, bulge);
        else if (it->second != bulge) {
            ++collisions;
            colliding.insert({phase, std::abs(it->second - bulge)});
        }
        auto lt = last.find(key);
        if (lt == last.end()) {
            last.emplace(key, std::pair<int, int>{phase, bulge});
        } else {
            if (lt->second.first != phase && lt->second.second != bulge) {
                ++cross;
                phase_pair.insert({{lt->second.first, phase},
                                   std::abs(lt->second.second - bulge)});
            }
            lt->second = {phase, bulge};
        }
    }
};
inline ChaseStamp& chase_stamp() { static ChaseStamp s; return s; }
inline long chase_key_band(int row, int col)
{
    return (static_cast<long>(col) + 8) * 1024L + static_cast<long>(row);
}
inline long chase_key_work(int idx) { return (1L << 24) + static_cast<long>(idx); }

#define CHASE_STAMP_ON()        (::ddla::chase_stamp().enabled)
#define CHASE_STAMP_PHASE(p)    do { ::ddla::chase_stamp().phase_begin(p); } while (0)
#define CHASE_STAMP_BULGE(t)    do { ::ddla::chase_stamp().bulge = (t); } while (0)
#define CHASE_STAMP_ENTER()     do { ::ddla::chase_stamp().enabled = true; } while (0)
#define CHASE_STAMP_BAND(row, col) \
    do { if (::ddla::chase_stamp().enabled) \
             ::ddla::chase_stamp().rec(::ddla::chase_key_band((row), (col))); } while (0)
#define CHASE_STAMP_WORK(idx) \
    do { if (::ddla::chase_stamp().enabled) \
             ::ddla::chase_stamp().rec(::ddla::chase_key_work(idx)); } while (0)
#else
#define CHASE_STAMP_ON()        false
#define CHASE_STAMP_PHASE(p)    do {} while (0)
#define CHASE_STAMP_BULGE(t)    do {} while (0)
#define CHASE_STAMP_ENTER()     do {} while (0)
#define CHASE_STAMP_BAND(row, col) do {} while (0)
#define CHASE_STAMP_WORK(idx)      do {} while (0)
#endif

// ---------------------------------------------------------------------------
// Thread team for the host chase.
//
// The chase is a sequence of (step, phase) regions; within one phase the active
// bulges are independent -- verified, not assumed: the occupancy stamp above
// finds ZERO within-phase collisions at every size and kd tried, so splitting
// the bulge index across threads cannot change any element's value.  What does
// need ordering is the boundaries BETWEEN phases, where the stamp finds
// conflicts (bulge t's phase Q reads what bulge t' != t wrote in phase P).  The
// barrier positions below come from that conflict set, not from intuition: a
// barrier is needed exactly at the boundaries that separate a conflicting pair.
//
// Slices are contiguous so that a thread's bulges stay in stream order, which is
// what keeps the recorded rotation stream identical to the serial one.
// ---------------------------------------------------------------------------
// DLARTG (real): (c, s, r) with [c s; -s c] (f, g) = (r, 0).
template <typename T>
void givens_gen(T f, T g, typename real_type<T>::type& c, T& s, T& r)
{
    if constexpr (is_real_v<T>) {
        if (g == T(0)) { c = T(1); s = T(0); r = f; return; }
        if (f == T(0)) { c = T(0); s = T(1); r = g; return; }
        r = T(std::copysign(std::sqrt(std::fabs(f) * std::fabs(f)
                                      + std::fabs(g) * std::fabs(g)), f));
        c = f / r;
        s = g / r;
    } else {
        using rt = typename real_type<T>::type;
        if (g == T(0)) { c = rt(1); s = T(0); r = f; return; }
        if (f == T(0)) {
            c = rt(0);
            r = T(std::abs(g), rt(0));
            s = std::conj(g) / std::abs(g);
            return;
        }
        const rt ft = std::abs(f);
        const rt tt = std::sqrt(ft * ft + std::abs(g) * std::abs(g));
        c = ft / tt;
        s = (f / ft) * std::conj(g) / tt;
        r = c * f + s * g;
    }
}

// ZLAR2V on one bulge: the two-sided rotation (c, s) of the 2 x 2 diagonal
// block whose band entries are the diagonals x, y and the subdiagonal z.
// Every chase driver below applies it in its innermost loop (hence the forced
// inlining); the inputs are read before any is written.
template <typename T>
__attribute__((always_inline)) inline void lar2v_pair(typename real_type<T>::type c, T sv, T& x, T& y, T& z)
{
    using rt = typename real_type<T>::type;
    if constexpr (is_real_v<T>) {
        const rt xi = x;
        const rt yi = y;
        const rt zi = z;
        const rt t1 = sv * zi;
        const rt t2 = c * zi;
        const rt t3 = t2 - sv * xi;
        const rt t4 = t2 + sv * yi;
        const rt t5 = c * xi + t1;
        const rt t6 = c * yi - t1;
        x = c * t5 + sv * t4;
        y = c * t6 - sv * t3;
        z = c * t4 - sv * t5;
    } else {
        const rt xi = x.real();
        const rt yi = y.real();
        const T zi = z;
        const rt t1r = sv.real() * zi.real() - sv.imag() * zi.imag();
        const rt t1i = sv.real() * zi.imag() + sv.imag() * zi.real();
        const T t2 = c * zi;
        const T t3 = t2 - std::conj(sv) * xi;
        const T t4 = std::conj(t2) + sv * yi;
        const rt t5 = c * xi + t1r;
        const rt t6 = c * yi - t1r;
        x = T(c * t5 + (sv.real() * t4.real()
                        + sv.imag() * t4.imag()), rt(0));
        y = T(c * t6 - (sv.real() * t3.real()
                        - sv.imag() * t3.imag()), rt(0));
        z = c * t3 + std::conj(sv) * T(t6, t1i);
    }
}

// One DROT/ZROT scalar pair: x' = c*x + s*y ; y' = c*y - conj(s)*x
// (conj(s) == s for the real types).
template <typename T>
void rot_pair(typename real_type<T>::type c, T s, T& x, T& y)
{
    const T t = x;
    x = c * x + s * y;
    if constexpr (is_real_v<T>) {
        y = c * y - s * t;
    } else {
        y = c * y - std::conj(s) * t;
    }
}

// Post-chase extraction, shared by the serial and threaded chases: d and e
// from the finished band; for complex T also make the off-diagonal real and
// collect the n-1 phase factors that make it so (they become row scalings of
// the eigenvectors).  Runs once per chase, outside every hot loop, so sharing
// it costs neither path anything.  Plain (unstamped) accessor: the serial
// chase stamps only up to the point where it prints the stamp report, which
// is before this code runs.  Returns the number of phases written (n-1 for
// complex T, none for real T).
template <typename T>
std::size_t host_chase_tail(T* ab, int kd, int n,
                            typename real_type<T>::type* diagonal,
                            typename real_type<T>::type* e_off,
                            T* phases)
{
    using rt = typename real_type<T>::type;
    const int kd1 = kd + 1;
    auto B = [ab, kd1](int row, int col) -> T& {
        return ab[static_cast<std::size_t>(col) * kd1 + row];
    };
    for (int i = 0; i < n; ++i) {
        if constexpr (is_real_v<T>) {
            diagonal[i] = B(kd, i);
        } else {
            diagonal[i] = B(kd, i).real();
        }
    }
    if constexpr (is_real_v<T>) {
        for (int i = 0; i < n - 1; ++i)
            e_off[i] = B(kd - 1, i + 1);
    } else {
        for (int i = 0; i < n - 1; ++i) {
            T t = B(kd - 1, i + 1);
            const rt abst = std::abs(t);
            B(kd - 1, i + 1) = T(abst, rt(0));
            e_off[i] = abst;
            t = (abst != rt(0)) ? t / abst : T(1);
            if (i < n - 2)
                B(kd - 1, i + 2) =
                    B(kd - 1, i + 2) * t;
            phases[i] = std::conj(t);
        }
    }
    return is_real_v<T> ? 0 : static_cast<std::size_t>(std::max(0, n - 1));
}

// The redundant host chase.  band holds (kd+1)*n entries laid out as
// band[col*(kd+1) + row] (destroyed).  Produces d, e, the rotation stream (in
// application order) and, for complex T, the n-1 phase factors that make
// the off-diagonal real.
//
// The chase is the reference ?SBTRD cascade regrouped PER BULGE.  Within one
// (i, k) step every existing bulge runs its phases back to back -- generate,
// apply-right, ZLAR2V, unconj (complex), apply-left, record, create -- walking
// the bulges from the highest position down; then comes the single annihilate
// rotation, then the late phases of the bulge it just created.  This is
// value-identical to the phase-major order the reference loops use: within a
// step the bulges occupy disjoint column ranges [idx-1, idx+kd-1] (the next
// bulge up starts at idx+kd), so no element is written twice, and the only
// cross-bulge edge -- a bulge's create writes B(0, idx+kd), which the next
// bulge up reads in its generate -- is satisfied by the descending order.
// Verified bit-for-bit against the frozen phase-major chase over the full
// case grid of .claude_jobs/eig2/chase_fused_check.cpp (n 3..70 x kd 1..14,
// dense kd >= n-1, the (2kd-1)(kd+1) +/- 1 edges, n=3000 at kd 16/32/64,
// n=10000 at kd 16/32), all four scalar types.  The per-bulge order is what
// makes the chase cache-resident: each bulge's band columns stay in cache
// across its phases instead of being re-fetched once per phase (~kd sweeps
// of the whole bulge working set in the phase-major form).
//
// @p progress, when not null, receives a release-store of the record count
// after every step -- the publication contract for the Q_band overlap
// consumer (hb2st_sink.h): the consumer's acquire-paired polls may then read
// exactly the records below that watermark while the chase continues.  Sound
// because the reserve below makes the record buffer stable for the whole
// chase (the stream never exceeds n^2/2 + 64 records), so the concurrent
// element reads race nothing still being written.  Records within a step are
// written into their final slot indices (the bulges run in descending order),
// so the stream stays byte-identical to the phase-major form and the
// watermark is published only after every slot of the step is written.  The
// classic path passes nullptr and pays one predictable branch per step.
// KeepRecords=false is the eigenvalue-only fallback: no rotation records or
// publication watermark are produced, while the band arithmetic is unchanged.
//
// Band storage is COLUMN-major, matching LAPACK's own band array:
//   band[col*(kd+1) + row] = A(col + row - kd, col),  row in [0, kd].
// The chase's per-bulge phases walk `row` at a fixed column, so this layout
// makes them contiguous -- the row-major form (row*n + col) strides by n
// elements and costs ~1.4x on the reduction with identical arithmetic
// (measured).
// Compact spatial storage is used only by the multi-node path. Offsets
// remain global in the numerical kernel; inaccessible columns are never read.
template <typename T>
struct SpatialChase {
    using rt = typename real_type<T>::type;
    int offset = 0, width = 0, core_lo = 0, core_hi = 0;
    int valid_lo = 0, valid_hi = 0, n = 0, kd = 0;
    // the halo exchange before every step and the tail (d, e, phases; returns
    // the phase count) after the chase
    void* context=nullptr;
    void (*before_run)(void*,T*,rt*,T*)=nullptr;
    std::size_t (*finish_run)(void*,T*,rt*,rt*,T*)=nullptr;
    void before_step(T* b,rt* d,T* w){before_run(context,b,d,w);}
    std::size_t finish_output(T* b,rt* d,rt* e,T* p){return finish_run(context,b,d,e,p);}
    bool accepts(int idx) const {
        return core_lo < core_hi && std::max(0, idx-1) >= valid_lo &&
               std::min(n, idx+kd+1) <= valid_hi;
    }
    bool owns(int idx) const { return idx >= core_lo && idx < core_hi; }
};

template <typename T>
struct BoundedChaseOutput {
    long long capacity = 0;
    void* context=nullptr;void (*publish_run)(void*,long long,const hb2st_rot<T>*,bool)=nullptr;
    void emit(long long end,const hb2st_rot<T>* r,bool last){publish_run(context,end,r,last);}
};

// Record storage is caller-owned. Production callers publish bounded rings;
// serial test oracles may supply an array for the complete stream.
template<class T>inline void chase_record_store(hb2st_rot<T>* records,std::size_t at,const hb2st_rot<T>& value)
{::new(static_cast<void*>(records+at))hb2st_rot<T>(value);}
// dc/wc: the rotation cosines and sines of every column of the band view
// (column_count + 2 each); phase_count receives the tail's phase count.
template <typename T, bool KeepRecords = true, bool Bounded = false, bool Spatial = false>
void host_chase(T* band, int n, int kd,
                typename real_type<T>::type* diagonal,
                typename real_type<T>::type* e_off,
                hb2st_rot<T>* rots,
                T* phases, std::size_t& phase_count,
                std::atomic<long long>* progress,
                BoundedChaseOutput<T>* output,
                SpatialChase<T>* spatial,
                typename real_type<T>::type* dc, T* wc)
{
    using rt = typename real_type<T>::type;
    const int kd1 = kd + 1;
    const int kdm1 = kd - 1;
    const int kdn = std::min(n - 1, kd);
    T* ab = band;

    // the one place the band layout is expressed; every access below goes
    // through it so the layout can change without touching the algorithm --
    // and, under the debug flag, the one place the occupancy stamp can see
    // every band access.  Every access in this routine is read-modify-write
    // (rot_pair, ZLARGV and ZLAR2V all read the element they overwrite), so
    // recording accesses and flagging collisions is exact: there are no
    // read-only accesses to produce false positives.
    const int column_offset = Spatial ? spatial->offset : 0;
    const int column_count = Spatial ? spatial->width : n;
    auto B = [ab, kd1, column_offset](int row, int col) -> T& {
#ifdef DDLA_EIG_CHASE_STAMP
        CHASE_STAMP_BAND(row, col);
#endif
        return ab[static_cast<std::size_t>(col-column_offset) * kd1 + row];
    };

    phase_count = 0;
    long long bounded_base = 0, recorded = 0;
    if constexpr (Bounded) {
        if (!output || output->capacity <= 0) throw std::logic_error("bounded chase: missing output");
    }

    // the diagonal must be real for the complex types (phe2hb guarantees it
    // for its input; enforce defensively on the band view)
    if constexpr (!is_real_v<T>) {
        for (int col = column_offset; col < column_offset+column_count; ++col)
            B(kd, col) = T(B(kd, col).real(), rt(0));
    }

    std::fill_n(dc, static_cast<std::size_t>(column_count) + 2, rt(0));
    std::fill_n(wc, static_cast<std::size_t>(column_count) + 2, T(0));

    if (kd > 1) {
        int nr_old = 0;
        int j1 = kdn + 2;   // 1-based, exactly as in the reference loops
        int j2 = 1;

        for (int i = 1; i <= n - 2; ++i) {
            for (int k = kdn + 1; k >= 2; --k) {
                if constexpr (Spatial) spatial->before_step(ab,dc,wc);
                CHASE_STAMP_ENTER();
                // ---- per-step bookkeeping, pure integer arithmetic ----
                // j1/j2 advance into their top-phase values; the annihilate
                // then shifts j1 back unconditionally when k > 2 (even where
                // its rotation is skipped at the band's right edge), and the
                // bottom exit shifts j2 back after the record positions.
                j1 += kdn;
                j2 += kdn;
                const bool ann = (k > 2);
                const int j1t = j1, j2t = j2;
                const int j1m = j1t - (ann ? kdn + 1 : 0);
                const int nr_new = nr_old + (ann ? 1 : 0);
                const bool exit_ = (j2t + kdn > n);
                const int j2m = j2t - (exit_ ? kdn + 1 : 0);
                // record slots this step fills: positions j = j1m-1 + t*kd1,
                // valid while j < j2t and j < n (j >= 1 always holds because
                // j1m-1 = i+k-2 >= 1).  Kept general to match the reference
                // record loop exactly.
                int nrec = 0;
                long long base = 0;
                if constexpr (KeepRecords) {
                    const int t_span = (j2t > j1m - 1)
                                           ? (j2t - (j1m - 1) + kd1 - 1) / kd1 : 0;
                    const int t_n = (n > j1m - 1)
                                        ? (n - (j1m - 1) + kd1 - 1) / kd1 : 0;
                    nrec = std::min(t_span, t_n);
                    if constexpr (Bounded) base = bounded_base;
                    else {
                        base = recorded;
                        recorded += nrec;
                    }
                }

                // Late phases of one bulge at record position idx (its j0):
                // ZLAR2V, unconj (complex), apply-left, record, create.
                // @p is_last marks the rightmost active bulge, whose
                // apply-left stops at lend = min(kd-1, n-j2t) exactly as both
                // reference branch forms do; @p slot is the bulge's record
                // slot, or -1 when this step records no such position.
                auto late_phases = [&](int idx, bool is_last, long long slot) {
                    // ---- ZLAR2V: both sides, diagonal blocks ----
                    CHASE_STAMP_PHASE(::ddla::SP_ZLAR2V);
                    {
                        const int colx = idx - 1;
                        const int coly = idx;
                        lar2v_pair<T>(dc[idx-column_offset], wc[idx-column_offset], B(kd, colx), B(kd, coly), B(kd - 1, coly));
                    }
                    // ---- make the sine Hermitian before the left update ----
                    if constexpr (!is_real_v<T>) {
                        CHASE_STAMP_PHASE(::ddla::SP_UNCONJ);
                        CHASE_STAMP_WORK(idx);
                        wc[idx-column_offset] = std::conj(wc[idx-column_offset]);
                    }
                    // ---- apply from the left ----------------------------
                    // Both reference branch forms (?LARTV's per-l nrt guard
                    // and ?ROT's lend/trailing split) give every bulge the
                    // pairs B(kd-1-lp, idx+lp) / B(kd-lp, idx+lp), with the
                    // rightmost bulge stopping at lp <= n-j2t; the col < n
                    // check is the reference's defensive break.
                    CHASE_STAMP_PHASE(::ddla::SP_APPLY_LEFT);
                    {
                        const int lmax = is_last ? std::min(kdm1, n - j2t) : kdm1;
                        // Only band entries change in this loop. Capture the
                        // coefficients after the complex sine conjugation.
                        const rt c = dc[idx-column_offset];
                        const T sv = wc[idx-column_offset];
                        for (int lp = 1; lp <= lmax; ++lp) {
                            const int col = idx + lp;
                            if (col >= n) break;
                            rot_pair<T>(c, sv,
                                        B(kd - 1 - lp, col), B(kd - lp, col));
                        }
                    }
                    // ---- record the Q rotation ---------------------------
                    if constexpr (KeepRecords) {
                        if (slot >= 0 && (!Spatial || spatial->owns(idx))) {
                            hb2st_rot<T> rec{};  // value-init: zero the padding after j0
                            rec.j0 = idx;
                            rec.c = dc[idx-column_offset];
                            if constexpr (is_real_v<T>) rec.s = wc[idx-column_offset];
                            else                        rec.s = std::conj(wc[idx-column_offset]);
                            chase_record_store(rots,static_cast<std::size_t>(Bounded?slot%output->capacity:slot),rec);
                        }
                    }
                    // ---- create the next bulge entries outside the band --
                    // runs with the post-exit j2m, exactly as the reference's
                    // create range does after its bottom exit
                    if (idx < j2m && idx + kd < n) {
                        wc[idx + kd-column_offset] = wc[idx-column_offset] * B(0, idx + kd);
                        B(0, idx + kd) = dc[idx-column_offset] * B(0, idx + kd);
                    }
                };

                // ---- the existing bulges, highest position first --------
                // Old bulge s (s in [0, nr_old)) sits at idx = j1t-1+s*kd1
                // and is mid-phase bulge t = s + (ann ? 1 : 0): the annihilate
                // takes mid slot 0.  Descending order satisfies the one
                // cross-bulge edge (bulge s's create writes B(0, idx+kd),
                // which bulge s+1's generate reads) and keeps every bulge's
                // band columns in cache across its phases.
                for (int s = nr_old - 1; s >= 0; --s) {
                    const int t = s + (ann ? 1 : 0);
                    const int idx = j1t - 1 + s * kd1;
                    if constexpr (Spatial) { if (!spatial->accepts(idx)) continue; }
                    const int colx = idx - 1;
                    CHASE_STAMP_BULGE(t);
                    // ---- generate: zero WORK against the band's top row ---
                    CHASE_STAMP_PHASE(::ddla::SP_GENERATE);
                    CHASE_STAMP_WORK(idx);
                    {
                        const T f = B(0, colx);
                        const T g = wc[idx-column_offset];
                        if constexpr (is_real_v<T>) {
                            if (g == T(0)) {
                                dc[idx-column_offset] = rt(1);
                                wc[idx-column_offset] = T(0);
                            } else if (f == T(0)) {
                                dc[idx-column_offset] = rt(0);
                                wc[idx-column_offset] = T(1);
                                B(0, colx) = g;
                            } else if (std::fabs(f) > std::fabs(g)) {
                                const T tt = g / f;
                                const T ttt = T(std::sqrt(rt(1) + tt * tt));
                                dc[idx-column_offset] = rt(1) / ttt;
                                wc[idx-column_offset] = tt * dc[idx-column_offset];
                                B(0, colx) = f * ttt;
                            } else {
                                const T tt = f / g;
                                const T ttt = T(std::sqrt(rt(1) + tt * tt));
                                wc[idx-column_offset] = rt(1) / ttt;
                                dc[idx-column_offset] = tt * wc[idx-column_offset];
                                B(0, colx) = g * ttt;
                            }
                        } else {
                            if (g == T(0)) {
                                dc[idx-column_offset] = rt(1);
                                wc[idx-column_offset] = T(0);
                                B(0, colx) = f;
                            } else if (f == T(0)) {
                                dc[idx-column_offset] = rt(0);
                                wc[idx-column_offset] = std::conj(g) / std::abs(g);
                                B(0, colx) = T(std::abs(g), rt(0));
                            } else {
                                const rt f2 = f.real() * f.real() + f.imag() * f.imag();
                                const rt g2 = g.real() * g.real() + g.imag() * g.imag();
                                const rt f2s = std::sqrt(rt(1) + g2 / f2);
                                const T r = f2s * f;
                                dc[idx-column_offset] = rt(1) / f2s;
                                wc[idx-column_offset] = (r / (f2 + g2)) * std::conj(g);
                                B(0, colx) = r;
                            }
                        }
                    }
                    // ---- apply from the right -----------------------------
                    // The reference's two branch orders (?LARTV's l-major
                    // loops and ?ROT's bulge-major loops) do the same pairs
                    // B(l, idx-1) / B(l-1, idx); per bulge the two band
                    // columns are contiguous, so walk them through restrict
                    // pointers and let GCC vectorize (baseline SSE2, no FMA,
                    // so the vectorized form stays bit-identical).
                    CHASE_STAMP_PHASE(::ddla::SP_APPLY_RIGHT);
                    {
                        T* const __restrict px =
                            ab + static_cast<std::size_t>(colx-column_offset) * kd1;
                        T* const __restrict py =
                            ab + static_cast<std::size_t>(colx+1-column_offset) * kd1;
                        // Keep the invariant coefficients in registers; the
                        // band update cannot modify the chase scratch arrays.
                        const rt c = dc[idx-column_offset];
                        const T sv = wc[idx-column_offset];
                        for (int l = 1; l < kd; ++l)
                            rot_pair<T>(c, sv, px[l], py[l - 1]);
                    }
                    late_phases(idx, s == nr_old - 1,
                                t < nrec ? base + t : -1);
                }

                // ---- the annihilate rotation -----------------------------
                if (ann) {
                    CHASE_STAMP_PHASE(::ddla::SP_ANNIHILATE);
                    // a single rotation over a contiguous range: not worth
                    // splitting, so every thread executes it identically
                    if (k <= n - i + 1 && (!Spatial || spatial->accepts(i+k-2))) {
                        rt c;
                        T sv, r;
                        givens_gen<T>(B(kd - k + 2, i + k - 3),
                                      B(kd - k + 1, i + k - 2), c, sv, r);
                        dc[i + k - 2-column_offset] = c;
                        wc[i + k - 2-column_offset] = sv;
                        B(kd - k + 2, i + k - 3) = r;
                        for (int l = 0; l < k - 3; ++l)
                            rot_pair<T>(c, sv,
                                        B(kd - k + 3 + l, i + k - 3),
                                        B(kd - k + 2 + l, i + k - 2));
                    }
                    // (the nr += 1 / j1 shift the reference does here is
                    // already accounted for in nr_new / j1m above)
                }

                // ---- late phases of the bulge the annihilate created ----
                // its record position is j1m-1 = i+k-2.  Only the annihilate
                // creates it: for k == 2 the mid slot 0 IS old bulge 0 (same
                // idx, j1m == j1t) and the loop above already ran its late
                // phases -- a second call here would apply that bulge twice.
                // The nr_new > 0 guard is the reference's `if (nr > 0)`: at
                // the chase tail the bottom exits can drive nr negative, and
                // then the phantom bulge (whose annihilate rotation was
                // skipped, idx = i+k-2 possibly >= n) must not be touched.
                if (ann && nr_new > 0 && (!Spatial || spatial->accepts(j1m-1)))
                    late_phases(j1m - 1, nr_new == 1, nrec > 0 ? base : -1);

                // record positions beyond the active bulges, if the step's
                // span ever outruns nr (it does not at any reachable state --
                // both change at the same rate): fill them from dc/wc to
                // match the reference record loop exactly
                if constexpr (KeepRecords) {
                    for (int t = nr_new; t < nrec; ++t) {
                        const int idx = j1m - 1 + t * kd1;
                        if constexpr (Spatial) { if (!spatial->owns(idx)) continue; }
                        hb2st_rot<T> rec{};  // value-init: zero the padding after j0
                        rec.j0 = idx;
                        rec.c = dc[idx-column_offset];
                        if constexpr (is_real_v<T>) rec.s = wc[idx-column_offset];
                        else                        rec.s = std::conj(wc[idx-column_offset]);
                        chase_record_store(rots,static_cast<std::size_t>(Bounded?(base+t)%output->capacity:base+t),rec);
                    }
                    if constexpr (Bounded) {
                        bounded_base = base + nrec;
                        output->emit(bounded_base, rots, i == n-2 && k == 2);
                    }
                    // publish this step's records to the overlap consumer
                    // (release pairs with its acquire polls; see header comment)
                    if (progress != nullptr)
                        progress->store(base + nrec, std::memory_order_release);
                }

                nr_old = nr_new - (exit_ ? 1 : 0);
                j1 = j1m;
                j2 = j2m;
            }
        }

    }   // if (kd > 1)

#ifdef DDLA_EIG_CHASE_STAMP
    {
        ChaseStamp& st = chase_stamp();
        std::printf("STAMP n=%d kd=%d touched=%ld collisions=%ld cross=%ld\n",
                    n, kd, st.touched, st.collisions, st.cross);
        if (st.colliding.empty()) {
            std::printf("STAMP no within-phase collisions\n");
        } else {
            for (const auto& pr : st.colliding)
                std::printf("STAMP WITHIN-PHASE COLLISION %s gap=%d\n",
                            chase_phase_name(pr.first), pr.second);
        }
        if (st.phase_pair.empty()) {
            std::printf("STAMP no cross-phase conflicts\n");
        } else {
            for (const auto& pr : st.phase_pair)
                std::printf("STAMP cross %-12s -> %-12s gap=%d\n",
                            chase_phase_name(pr.first.first),
                            chase_phase_name(pr.first.second), pr.second);
        }
        st.enabled = false;
        st.owner.clear();
        st.last.clear();
        std::fflush(stdout);
    }
#endif

    if constexpr (Spatial) phase_count = spatial->finish_output(ab,diagonal,e_off,phases);
    else phase_count = host_chase_tail<T>(ab, kd, n, diagonal, e_off, phases);
}

// ---------------------------------------------------------------------------
// The threaded chase (DDLA_EIG_CHASE_THREADS >= 2).  A SEPARATE function,
// deliberately not an `nt == 1` arm of host_chase: with both in one function
// the per-phase slicing cost the serial default 46% (2.17 -> 3.17 s at
// n=5000, measured), so the serial body above stays its own function and this
// copy carries the parallel machinery.  It still runs the phase-major order:
// the fused serial chase above must remain bitwise identical to it.
//
// Structure (independence proven by the occupancy stamp, identity proven by
// the bitwise gate .claude_jobs/eig2/chase_par_check.cpp):
//   * a phase's active bulges touch disjoint band/dc/wc elements, so every
//     splittable phase partitions the BULGE INDEX t of [0, nr) and each
//     thread works a contiguous t-range;
//   * the unsplittable phases -- the single annihilate rotation and the
//     trailing partial bulge of apply_left -- run on thread 0 only:
//     rot_pair reads the element it overwrites, so redundant execution by
//     several threads corrupts the result even though every thread computes
//     the same value;
//   * one barrier between adjacent phases (the stamp's cross-phase conflict
//     set covers every adjacent pair that shares state);
//   * the loop bookkeeping (j1, j2, nr) is a pure function of the loop
//     indices, so every thread maintains a redundant copy in lockstep --
//     only the arrays are shared;
//   * record partitions the bulge index (NOT j) into per-thread buffers and
//     thread 0 concatenates them in thread order, which IS bulge order, so
//     the rotation stream and the sink's publication contract are exactly
//     the serial chase's;
//   * the serial code's early return inside the create phase is hoisted to
//     a t_ret that every thread computes identically, so the team terminates
//     in lockstep.
// ---------------------------------------------------------------------------

// x86 pause keeps a spinning barrier member from stealing decode bandwidth
// from the SMT sibling doing real work; harmless (empty) elsewhere
inline void chase_cpu_relax()
{
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

// cycle counter for the team profile (TSC; the cluster is x86 -- the
// fallback keeps non-x86 builds compiling but reports zero cycles)
inline unsigned long long chase_cycles()
{
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_ia32_rdtsc();
#else
    return 0;
#endif
}

// Sense-reversal spin barrier.  Spinning (not a condvar) is the design: a
// step's phases are tens of microseconds apart and a condvar wakeup costs
// 2-5 us against a ~100 ns spin wait when the team owns its cores.  The
// yield fallback after ~1024 pauses is the livelock guard -- a member that
// lost its core must not hold the others hostage forever (the original
// threaded chase hung 4 ranks x 6 threads on 6 CPUs before this existed).
struct ChaseBarrier {
    explicit ChaseBarrier(int nt) : nt_(nt) {}
    // false only on the abort path: a member failed, callers must unwind
    // without touching shared state again
    bool wait()
    {
        const unsigned e = epoch_.load(std::memory_order_relaxed);
        if (arrive_.fetch_add(1, std::memory_order_acq_rel) + 1 == nt_) {
            arrive_.store(0, std::memory_order_relaxed);
            epoch_.store(e + 1, std::memory_order_release);
            return !aborted();
        }
        int spins = 0;
        while (epoch_.load(std::memory_order_acquire) == e) {
            if (++spins % 1024 == 0) {
                if (aborted()) return false;
                std::this_thread::yield();
            } else {
                chase_cpu_relax();
            }
        }
        return true;
    }
    void abort() { aborted_.store(true, std::memory_order_release); }
    bool aborted() const { return aborted_.load(std::memory_order_acquire); }

private:
    const int nt_;
    std::atomic<int> arrive_{0};
    std::atomic<unsigned> epoch_{0};
    std::atomic<bool> aborted_{false};
};

// DDLA_EIG_CHASE_THREADS: opt-in thread count for the chase (default 1 =
// the serial host_chase).  Read on EVERY call -- the bitwise gate
// (chase_par_check.cpp) flips it with setenv between runs, so caching the
// value is not sound.
inline int chase_thread_count()
{
    const char* s = std::getenv("DDLA_EIG_CHASE_THREADS");
    if (s == nullptr || *s == '\0') return 1;
    const long v = std::strtol(s, nullptr, 10);
    if (v <= 1) return 1;
    return static_cast<int>(std::min<long>(v, 64));
}

// dc/wc: n+2 each; with records, pbuf_counts (nt) and pbuf_records (nt
// buffers of pbuf_stride >= n+1 records) hold each thread's step records.
template <typename T, bool KeepRecords = true>
void host_chase_par(T* band, int n, int kd,
                    typename real_type<T>::type* diagonal,
                    typename real_type<T>::type* e_off,
                    hb2st_rot<T>* rots,
                    T* phases, std::size_t& phase_count,
                    int nt,
                    std::atomic<long long>* progress,
                    detail::DcHostTeam* prepared_team,
                    typename real_type<T>::type* dc, T* wc,
                    std::size_t* pbuf_counts, hb2st_rot<T>* pbuf_records, std::size_t pbuf_stride)
{
    using rt = typename real_type<T>::type;
    const int kd1 = kd + 1;
    const int kdm1 = kd - 1;
    const int kdn = std::min(n - 1, kd);
    T* ab = band;
    // plain (unstamped) accessor: the occupancy stamp is a serial-run
    // instrument and is not thread-safe -- the parallel design was validated
    // by stamping the SERIAL chase, not this one
    auto B = [ab, kd1](int row, int col) -> T& {
        return ab[static_cast<std::size_t>(col) * kd1 + row];
    };

    phase_count = 0;
    std::size_t recorded = 0; // appended by thread 0 after the phase barrier

    // the diagonal must be real for the complex types (phe2hb guarantees it
    // for its input; enforce defensively on the band view)
    if constexpr (!is_real_v<T>) {
        for (int col = 0; col < n; ++col)
            B(kd, col) = T(B(kd, col).real(), rt(0));
    }

    // shared rotation arrays, written under the phase barriers below
    std::fill_n(dc,std::size_t(n)+2,rt(0));std::fill_n(wc,std::size_t(n)+2,T(0));

    if (kd <= 1) {
        // no chase at this bandwidth (same guard as the serial body)
        phase_count = host_chase_tail<T>(ab, kd, n, diagonal, e_off, phases);
        return;
    }

    ChaseBarrier bar(nt);
    // per-thread record buffers, sized to the hard per-step bound (a step's
    // valid records have distinct j in [1, n)), so nothing in the step loop
    // allocates -- or throws

    // the step body every thread runs; tid 0 is the caller.  Returns true
    // when the serial chase would have returned early from inside the
    // create phase (leaving d/e/phases untouched).
    auto body = [&](int tid) -> bool {
        hb2st_rot<T>* const mybuf = KeepRecords ? pbuf_records + std::size_t(tid) * pbuf_stride : nullptr;
        // contiguous bulge-index range [tlo, thi) of [0, count) for tid
        const auto slice = [nt, tid](int count, int& tlo, int& thi) {
            tlo = static_cast<int>(static_cast<long long>(count) * tid / nt);
            thi = static_cast<int>(static_cast<long long>(count) * (tid + 1) / nt);
        };
        int nr = 0;
        int j1 = kdn + 2;   // 1-based, exactly as in the serial loops
        int j2 = 1;

        for (int i = 1; i <= n - 2; ++i) {
            for (int k = kdn + 1; k >= 2; --k) {
                j1 += kdn;
                j2 += kdn;

                if (nr > 0) {
                    // ---- ZLARGV: zero WORK against the band's top row ----
                    {
                        int tlo, thi;
                        slice(nr, tlo, thi);
                        for (int t = tlo; t < thi; ++t) {
                            const int col = j1 - 2 + t * kd1;
                            const int idx = j1 - 1 + t * kd1;
                            const T f = B(0, col);
                            const T g = wc[idx];
                            if constexpr (is_real_v<T>) {
                                if (g == T(0)) {
                                    dc[idx] = rt(1);
                                    wc[idx] = T(0);
                                } else if (f == T(0)) {
                                    dc[idx] = rt(0);
                                    wc[idx] = T(1);
                                    B(0, col) = g;
                                } else if (std::fabs(f) > std::fabs(g)) {
                                    const T tt = g / f;
                                    const T ttt = T(std::sqrt(rt(1) + tt * tt));
                                    dc[idx] = rt(1) / ttt;
                                    wc[idx] = tt * dc[idx];
                                    B(0, col) = f * ttt;
                                } else {
                                    const T tt = f / g;
                                    const T ttt = T(std::sqrt(rt(1) + tt * tt));
                                    wc[idx] = rt(1) / ttt;
                                    dc[idx] = tt * wc[idx];
                                    B(0, col) = g * ttt;
                                }
                            } else {
                                if (g == T(0)) {
                                    dc[idx] = rt(1);
                                    wc[idx] = T(0);
                                    B(0, col) = f;
                                } else if (f == T(0)) {
                                    dc[idx] = rt(0);
                                    wc[idx] = std::conj(g) / std::abs(g);
                                    B(0, col) = T(std::abs(g), rt(0));
                                } else {
                                    const rt f2 = f.real() * f.real() + f.imag() * f.imag();
                                    const rt g2 = g.real() * g.real() + g.imag() * g.imag();
                                    const rt f2s = std::sqrt(rt(1) + g2 / f2);
                                    const T r = f2s * f;
                                    dc[idx] = rt(1) / f2s;
                                    wc[idx] = (r / (f2 + g2)) * std::conj(g);
                                    B(0, col) = r;
                                }
                            }
                        }
                    }
                    if (!bar.wait()) return false;
                    // ---- apply from the right ---------------------------
                    if (nr >= 2 * kd - 1) {
                        for (int l = 1; l < kd; ++l) {
                            int tlo, thi;
                            slice(nr, tlo, thi);
                            for (int t = tlo; t < thi; ++t) {
                                const int colx = j1 - 2 + t * kd1;
                                const int coly = j1 - 1 + t * kd1;
                                const int idx = j1 - 1 + t * kd1;
                                rot_pair<T>(dc[idx], wc[idx],
                                            B(l, colx), B(l - 1, coly));
                            }
                        }
                    } else {
                        int tlo, thi;
                        slice(nr, tlo, thi);
                        for (int t = tlo; t < thi; ++t) {
                            const int jinc = j1 - 1 + t * kd1;
                            const int idx = jinc;
                            for (int l = 0; l < kdm1; ++l)
                                rot_pair<T>(dc[idx], wc[idx],
                                            B(1 + l, jinc - 1), B(l, jinc));
                        }
                    }
                }
                if (!bar.wait()) return false;
                if (k > 2) {
                    // ---- the single annihilate rotation: thread 0 only ----
                    // (one rotation over a contiguous range, and rot_pair
                    // reads the element it overwrites, so redundant execution
                    // by several threads corrupts the result)
                    if (tid == 0 && k <= n - i + 1) {
                        rt c;
                        T sv, r;
                        givens_gen<T>(B(kd - k + 2, i + k - 3),
                                      B(kd - k + 1, i + k - 2), c, sv, r);
                        dc[i + k - 2] = c;
                        wc[i + k - 2] = sv;
                        B(kd - k + 2, i + k - 3) = r;
                        for (int l = 0; l < k - 3; ++l)
                            rot_pair<T>(c, sv,
                                        B(kd - k + 3 + l, i + k - 3),
                                        B(kd - k + 2 + l, i + k - 2));
                    }
                    // bookkeeping is redundant state: every thread applies it
                    nr += 1;
                    j1 -= kdn + 1;
                }
                if (!bar.wait()) return false;
                if (nr > 0) {
                    // ---- ZLAR2V + (complex) unconj: split by bulge ----
                    // zlar2v writes only band elements and unconj only
                    // wc[idx], both disjoint per bulge, so the two loops
                    // share one phase segment without a barrier between them
                    {
                        int tlo, thi;
                        slice(nr, tlo, thi);
                        for (int t = tlo; t < thi; ++t) {
                            const int colx = j1 - 2 + t * kd1;
                            const int coly = j1 - 1 + t * kd1;
                            const int idx = j1 - 1 + t * kd1;
                            lar2v_pair<T>(dc[idx], wc[idx], B(kd, colx), B(kd, coly), B(kd - 1, coly));
                        }
                        if constexpr (!is_real_v<T>) {
                            for (int t = tlo; t < thi; ++t) {
                                const int idx = j1 - 1 + t * kd1;
                                wc[idx] = std::conj(wc[idx]);
                            }
                        }
                    }
                    if (!bar.wait()) return false;
                    // ---- apply from the left: split by bulge ------------
                    if (2 * kd - 1 < nr) {
                        for (int l = 1; l < kd; ++l) {
                            const int nrt = (j2 + l > n) ? nr - 1 : nr;
                            int tlo, thi;
                            slice(nrt, tlo, thi);
                            for (int t = tlo; t < thi; ++t) {
                                const int col = j1 + l - 1 + t * kd1;
                                const int idx = j1 - 1 + t * kd1;
                                rot_pair<T>(dc[idx], wc[idx],
                                            B(kd - l - 1, col), B(kd - l, col));
                            }
                        }
                    } else {
                        const int j1end = j1 - 1 + kd1 * (nr - 2);
                        int tlo, thi;
                        slice(nr - 1, tlo, thi);
                        if (j1end >= j1 - 1)
                            for (int t = tlo; t < thi; ++t) {
                                const int jin = j1 - 1 + t * kd1;
                                const int idx = jin;
                                for (int l = 0; l < kd - 1; ++l) {
                                    const int col = jin + 1 + l;
                                    if (col >= n) break;
                                    rot_pair<T>(dc[idx], wc[idx],
                                                B(kd - 2 - l, col),
                                                B(kd - 1 - l, col));
                                }
                            }
                        // the trailing partial bulge: thread 0 only (one
                        // bulge, and rot_pair reads what it overwrites)
                        if (tid == 0) {
                            const int lend = std::min(kdm1, n - j2);
                            const int last = j1end + kd1;
                            if (lend > 0) {
                                const int idx = last;
                                for (int l = 0; l < lend; ++l) {
                                    const int col = last + 1 + l;
                                    if (col >= n) break;
                                    rot_pair<T>(dc[idx], wc[idx],
                                                B(kd - 2 - l, col),
                                                B(kd - 1 - l, col));
                                }
                            }
                        }
                    }
                }
                if (!bar.wait()) return false;
                // ---- record: partition the BULGE INDEX, not j -----------
                // this step's records are j = j1-1+t*kd1 < j2 -- the same
                // bulge indexing the phases above use, so splitting by t and
                // concatenating in thread order (ascending t) reproduces the
                // serial stream exactly
                if constexpr (KeepRecords) {
                    const int nrec =
                        (j2 > j1 - 1) ? (j2 - (j1 - 1) + kd1 - 1) / kd1 : 0;
                    int tlo, thi;
                    slice(nrec, tlo, thi);
                    std::size_t mine = 0;
                    for (int t = tlo; t < thi; ++t) {
                        const int j = j1 - 1 + t * kd1;
                        if (j - 1 < 0 || j >= n) continue;
                        hb2st_rot<T> rec{};  // value-init: zero the padding after j0
                        rec.j0 = j;
                        rec.c = dc[j];
                        if constexpr (is_real_v<T>) rec.s = wc[j];
                        else                        rec.s = std::conj(wc[j]);
                        chase_record_store(mybuf, mine++, rec);
                    }
                    pbuf_counts[std::size_t(tid)]=mine;
                }
                // Retain the phase barrier even without records, so every
                // thread finishes the left update before the create phase.
                if (!bar.wait()) return false;
                if constexpr (KeepRecords) {
                    if (tid == 0) {
                        for (int q = 0; q < nt; ++q) {
                            const auto* first=pbuf_records+std::size_t(q)*pbuf_stride;
                            const auto count=pbuf_counts[std::size_t(q)];
                            std::uninitialized_copy(first,first+count,rots+recorded);
                            recorded+=count;
                        }
                        // publish this step's records (release pairs with the
                        // consumer's acquire polls; see host_chase's header)
                        if (progress != nullptr)
                            progress->store(static_cast<long long>(recorded),
                                            std::memory_order_release);
                    }
                }
                if (j2 + kdn > n) {
                    nr -= 1;
                    j2 -= kdn + 1;
                }
                // ---- create: split by bulge, the serial early return is
                // hoisted to t_ret so the team terminates in lockstep ----
                {
                    const int njc =
                        (j2 > j1 - 1) ? (j2 - (j1 - 1) + kd1 - 1) / kd1 : 0;
                    int t_ret = njc;
                    for (int t = 0; t < njc; ++t)
                        if (j1 - 1 + t * kd1 >= j2) { t_ret = t; break; }
                    int tlo, thi;
                    slice(t_ret, tlo, thi);
                    for (int t = tlo; t < thi; ++t) {
                        const int j = j1 - 1 + t * kd1;
                        if (j + kd < n) {
                            wc[j + kd] = wc[j] * B(0, j + kd);
                            B(0, j + kd) = dc[j] * B(0, j + kd);
                        }
                    }
                    if (t_ret < njc) return true;
                }
                if (!bar.wait()) return false;
            }
        }
        return false;
    };

    bool early=false;
    if(prepared_team){prepared_team->run_roles(nt,[&](int role){try{const bool done=body(role);if(role==0)early=done;}catch(...){bar.abort();throw;}});}
    else {
    std::vector<std::thread> team;
    team.reserve(static_cast<std::size_t>(nt - 1));
    try {
        for (int w = 1; w < nt; ++w)
            team.emplace_back([&body, &bar, w] {
                try {
                    body(w);
                } catch (...) {
                    // nothing in the step loop can allocate; this is the
                    // belt-and-braces path -- unblock the others and exit
                    bar.abort();
                }
            });
    } catch (...) {
        // spawn failed mid-team: release the members already running, then
        // surface the spawn error (it outranks any generic abort message)
        bar.abort();
        for (auto& th : team) th.join();
        throw;
    }
    try {
        early = body(0);
    } catch (...) {
        bar.abort();
        for (auto& th : team) th.join();
        throw;
    }
    for (auto& th : team) th.join();
    }
    if (bar.aborted())
        throw std::runtime_error(
            "phb2st threaded chase: a team member aborted unexpectedly "
            "(DDLA_EIG_CHASE_THREADS=" + std::to_string(nt) + ")");
    if (!early)
        phase_count = host_chase_tail<T>(ab, kd, n, diagonal, e_off, phases);
}

// ---------------------------------------------------------------------------
// The node-team chase: the fused serial chase split over the RANKS of one
// node through a shared-memory window (2026-09-27).  Replaces the v1
// contract's redundant per-rank chase: with R ranks on a node, R cores
// computed identical results while R-1 sat idle.
//
// Why a step can be split (verified with an index-only replica of the fused
// chase over n = 40..2100, kd = 3..32, including sizes above the many-bulge
// threshold): within one step the bulges' access sets are disjoint except
// for ONE element per neighbouring pair -- B(0, idx+kd), which bulge s's
// create overwrites and bulge s+1's generate reads.  (The same create also
// writes wc[idx+kd], but that slot's next reader lives in the NEXT step,
// ordered by the end-of-step barrier -- not a within-step edge.)  Contiguous
// bulge ranges per member, highest-first, therefore need exactly one
// ordering rule at each range boundary:
//
//   pass 1 (generate through record) is fully parallel: its reads AND
//   writes stay inside the member's own bulge columns, all below the next
//   range's lowest column;
//   pass 2 (the creates) runs after the member publishes flag[m] = "pass 1
//   done"; its TOP create overwrites B(0, colx of the next range's lowest
//   bulge) and spins until that member's flag says the reads are done.
//   The waits point strictly upward, so they cannot cycle.  (The first
//   bring-up race was NOT the wc slot -- it was member 0's tail running
//   concurrently with the other ranges; the tail is ordered by a targeted
//   wait, see below.)
//
// Every element therefore sees the same operations in the same order as the
// serial chase -- d, e, the rotation stream and the phases are bit-identical
// for ANY member count (.claude_jobs/eig2/chase_team_check.cpp memcmps team
// vs DDLA_EIG_CHASE_TEAM=0 on every scalar type, grid and size below).
// Records go into their fixed slots base + t (disjoint across members), so
// the stream order is unchanged and the sink's publication contract holds:
// member 0 publishes the watermark after the step barrier.
//
// The window (MPI_Win_allocate_shared on the MPI_COMM_TYPE_SHARED split of
// the handle's communicator) holds the band, dc/wc, the whole record stream
// (n^2/2 + 64 slots -- the per-rank 1.1 GB vectors disappear), the d/e/phase
// outputs and the control block.  With several nodes, every node runs its own
// team on the identical all-reduced band; only the per-node replication
// remains, and nodes need no communication during the chase.  Team members:
// one per rank (its main thread); DDLA_EIG_CHASE_TEAM_THREADS=2 adds a second
// thread per rank.  The serial host_chase stays the fallback (single-member
// node, DDLA_EIG_CHASE_TEAM=0, window allocation failure); host_chase_par is
// untouched and still serves DDLA_EIG_CHASE_THREADS on the fallback path.
// ---------------------------------------------------------------------------

static_assert(std::atomic<long long>::is_always_lock_free,
              "node-team chase: shared atomics must be lock-free");
static_assert(std::atomic<int>::is_always_lock_free,
              "node-team chase: shared atomics must be lock-free");
static_assert(std::atomic<unsigned>::is_always_lock_free,
              "node-team chase: shared atomics must be lock-free");

// One hot atomic per cache-line PAIR (128 B): the adjacent-line prefetcher
// pulls lines in pairs, so 64-B spacing still shares a fetch.  Every member
// writes arrive/epoch each barrier, member 0 stores progress each step while
// each rank's consumer polls it, and each member's flag is spun on by the
// member below -- false sharing between these measured as ~5.5-8 us of
// per-step synchronization against 2-5 us of work per member (the team-vs-
// serial gap at n=5000, where parallelism cannot pay for it).
template <typename A>
struct ChaseTeamPad {
    A v{};
    char pad[128 - sizeof(A)];   // natural 8-B alignment: over-aligned
                                 // placement-new is not portable to nvcc
};
static_assert(sizeof(ChaseTeamPad<std::atomic<long long>>) == 128,
              "node-team chase: padded slot must span the full cache-line pair");
static_assert(sizeof(ChaseTeamPad<std::atomic<int>>) == 128,
              "node-team chase: padded slot must span the full cache-line pair");
static_assert(alignof(ChaseTeamPad<std::atomic<long long>>) == 8,
              "node-team chase: pad must keep natural alignment");
using ChaseTeamFlag = ChaseTeamPad<std::atomic<long long>>;

// Control block at the start of the shared window.  The leader placement-news
// it before the setup barrier; the cold atomics (phase_count, aborted -- read
// only on the abort path and once per chase) share the last line.  The
// window base is page-aligned and off_ctl is 0, so each padded slot sits on
// its own 128-B boundary.
struct ChaseTeamControl {
    ChaseTeamPad<std::atomic<long long>> progress;    // published record watermark
    ChaseTeamPad<std::atomic<int>> arrive;            // team barrier (sense reversal)
    ChaseTeamPad<std::atomic<unsigned>> epoch;
    ChaseTeamPad<std::atomic<int>> aborted;
    ChaseTeamPad<std::atomic<long long>> phase_count; // complex phases in ph_arr
    ChaseTeamPad<std::atomic<long long>> switch_pos;  // elastic ring: stream position at which the chase left the ring (LLONG_MAX: still in it)
};

inline std::size_t chase_team_align(std::size_t x, std::size_t a)
{
    return (x + a - 1) / a * a;
}

// The per-call shared window: node communicator, allocation, and typed
// pointers into the leader's segment (valid on every rank of the node).
template <typename T>
struct ChaseTeam {
    MPI_Comm node_comm = MPI_COMM_NULL;
    MPI_Win win = MPI_WIN_NULL;
    int myid = -1;               // global rank (TEAMPROF lines)
    double tsc_ghz = 0.0;        // TSC calibration for the profile
    int node_rank = 0, node_size = 1;
    int tpr = 1;                 // threads per rank (DDLA_EIG_CHASE_TEAM_THREADS)
    int participants = 1;        // ranks of the node that run chase members (<= node_size)
    std::function<void(long long)> handoff; // original application thread only
    void* handoff_context=nullptr;void (*handoff_run)(void*,long long)=nullptr;
    bool has_handoff() const noexcept{return handoff_run||bool(handoff);}
    void publish(long long end) const {if(handoff_run)handoff_run(handoff_context,end);else handoff(end);}
    bool profile=false,prefault=false;
    long long batch_records = detail::DistributedHistory<T>::chunk_size;
    bool is_member = true;       // node_rank < participants
    int members = 1;             // participants * tpr
    ChaseTeamControl* ctl = nullptr;
    ChaseTeamFlag* flag = nullptr;            // [members], stride 128 B, init -1
    T* band = nullptr;                        // (kd+1)*n, column-major
    typename real_type<T>::type* dc = nullptr;   // n+2
    T* wc = nullptr;                          // n+2
    hb2st_rot<T>* rots = nullptr;             // n^2/2 + 64 slots
    typename real_type<T>::type* diag_array = nullptr;   // n
    typename real_type<T>::type* e_arr = nullptr;   // n (slot n-1 unused)
    T* ph_arr = nullptr;                      // n
    long long capacity = 0;                   // record slots
    unsigned long long* guard = nullptr;      // ring only: sentinel words right after the ring; a slot index past the ring would clobber them
    long long ring_records = 0;               // > 0: the record area is a RING of this many slots (stream position g lives in slot g % ring_records)
    ChaseTeamFlag* consumed = nullptr;        // [node_size] stager watermarks (records consumed), stride 128 B; only with ring consumers
    bool elastic = false;                     // ring with consumers that is ABANDONED (switch_pos) instead of waited for
    long long ext_offset = 0;                 // elastic: record slot of the first position written after the switch
    int ring_consumers = 0;                   // > 0: that many ranks (node_rank 0..) read the ring, the chase waits for the slowest;
                                              // 0: nobody reads the stream (eigenvalues only) and the producers never wait

    ChaseTeam() = default;
    ChaseTeam(const ChaseTeam&) = delete;
    ChaseTeam& operator=(const ChaseTeam&) = delete;
};

// DDLA_EIG_CHASE_TEAM: the node-team chase is the default; =0 restores the
// replicated per-rank chase (A/B baseline).  Read on every call -- the
// bitwise gate flips it with setenv between runs.
inline bool chase_team_enabled()
{
    const char* s = std::getenv("DDLA_EIG_CHASE_TEAM");
    if (s == nullptr || *s == '\0') return true;
    return std::strtol(s, nullptr, 10) != 0;
}

// DDLA_EIG_CHASE_TEAM_THREADS: team members per rank (cap 8).  Default: 1 on GPU handles (2+ measured a
// LOSS there at every size, e.g. 5.17 vs 2.17 s at n=10000/kd=16: the extra member competes with the
// rank's sink consumer thread for the 2 CPUs per rank); on CPU handles min(threads per rank, 4 / ranks
// per node), i.e. a team of about 4 members -- the measured sweet spot (chase 3.85 -> 2.06 s at 4 members,
// n=5000/kd=64, 6 cpus; the lockstep steps hold tens of microseconds of work, so 40 members LOSE: 15.8 s
// vs 11 s serial at n=10000).  An explicit value always wins.  Read on every call.
inline int chase_team_threads_per_rank(int cpu_default = 1)
{
    const char* s = std::getenv("DDLA_EIG_CHASE_TEAM_THREADS");
    if (s == nullptr || *s == '\0') return cpu_default;
    const long v = std::strtol(s, nullptr, 10);
    if (v <= 1) return 1;
    return static_cast<int>(std::min<long>(v, 8));
}

// DDLA_EIG_CHASE_TEAM_PREFAULT=1: before the chase, each process writes one
// byte (value-preserving) per 4 KB page of the record area and band/dc/wc,
// so every page it will touch is mapped in its own page table.  Shared-
// memory first-touch faults cost more than private anonymous faults, and in
// a lockstep team one member's fault stalls all four; this moves the faults
// into a separately-timed prologue (TEAMPROF prefault_ms; the chase itself
// is chase_ms).  Measured 2026-09-28: it removes every in-chase fault and
// cuts the member loop 42-53%, but the prologue costs ~1.15 s at n=10000 --
// a diagnostic, not the fix.  Read on every call.
inline bool chase_team_prefault_enabled()
{
    const char* s = std::getenv("DDLA_EIG_CHASE_TEAM_PREFAULT");
    return s != nullptr && *s != '\0' && std::strtol(s, nullptr, 10) != 0;
}

// A record ring is followed by kChaseRingGuardWords sentinel words (the window area behind it -- d/e/phases -- is rewritten at the end of
// the chase, so a slot index past the ring would otherwise corrupt nothing visible and go unnoticed): chase_team_run checks them.
constexpr std::size_t kChaseRingGuardWords = 512;
constexpr unsigned long long kChaseRingGuardPattern = 0xA5C3E1F00F1E3C5Aull;

// Eigenvalue-only runs (jobz = 'N') read none of the rotation stream, so their chase window need not hold it: the record area
// is a ring of DDLA_EIG_CHASE_RING_RECORDS slots (default 8 x 262144 = 6.3 MB x 8 = ~50 MB of real doubles instead of
// n^2/2 x 24 B, i.e. 19 GB at n=40000), written round-robin.  Runs that keep or consume the stream never ask for a ring.
// A ring must hold several steps (a step records up to n/(kd+1)+2 slots): never below 4 steps.  0 disables the ring (A/B).
inline long long chase_ring_records(int n, int kd)
{
    const long long maxt = static_cast<long long>(n) / (kd + 1) + 2;
    long long rec = 8LL * 262144;
    if (const char* s = std::getenv("DDLA_EIG_CHASE_RING_RECORDS")) {
        if (*s != '\0') {
            const long long v = std::strtoll(s, nullptr, 10);
            if (v <= 0) return 0;
            rec = v;
        }
    }
    return std::max(rec, 4 * maxt);
}

// Split the handle's communicator into node groups, allocate the shared
// window (the node leader allocates, all ranks query). reset_run stages
// the gathered band after preparation.  Collective over handle->comm and node_comm.
// Returns false when the team must not run (single-member node, kd <= 1,
// window allocation failed) -- a node-uniform decision, so the node's ranks
// stay consistent; the caller falls back to the replicated chase.
template <typename T>
bool chase_team_setup(const DdlaHandle_t& handle,
                      const spectral_impl::Phb2stTeamPlan& plan, ChaseTeam<T>& tm)
{
    const int n = plan.n, kd = plan.band;
    MPI_Comm split = MPI_COMM_NULL;
    if (MPI_Comm_split_type(handle->comm, MPI_COMM_TYPE_SHARED, 0,
                            MPI_INFO_NULL, &split) != MPI_SUCCESS)
        return false;
    tm.node_comm = split;
    // the allocation-failure fallback only works if a failed
    // MPI_Win_allocate_shared RETURNS instead of aborting the job (the
    // window is n^2/2 * 24 B -- ~19 GB at n=40000)
    MPI_Comm_set_errhandler(tm.node_comm, MPI_ERRORS_RETURN);
    tm.myid = handle->myid;
    MPI_Comm_rank(split, &tm.node_rank);
    MPI_Comm_size(split, &tm.node_size);
    tm.tpr = plan.tpr;
    tm.participants = plan.participants;
    tm.profile = plan.profile;
    tm.prefault = plan.prefault;
    int mismatch = tm.node_rank != plan.node_rank || tm.node_size != plan.node_size;
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &mismatch, 1, MPI_INT, MPI_MAX, tm.node_comm));
    if (mismatch) {
        MPI_CHECK(MPI_Comm_free(&tm.node_comm));
        throw std::runtime_error("phb2st: prepared topology changed");
    }
    tm.is_member = tm.node_rank < tm.participants;
    tm.members = tm.participants * tm.tpr;
    if (tm.members <= 1 || kd <= 1) {
        MPI_Comm_free(&tm.node_comm);
        tm.node_comm = MPI_COMM_NULL;
        return false;
    }

    // the record area: the whole stream (n^2/2 + 64 slots) or, when the caller asked for a ring and it is smaller, the
    // ring.  The leader's value defines the layout for the whole node.
    long long ring_in[3] = {plan.ring_request, plan.consumers ? 1 : 0, plan.elastic ? 1 : 0};
    MPI_Bcast(ring_in, 3, MPI_LONG_LONG, 0, tm.node_comm);
    const long long ring = ring_in[0];
    const long long full_capacity = static_cast<long long>(n) * n / 2 + 64;
    tm.ring_records = (ring > 0 && ring < full_capacity) ? ring : 0;
    tm.ring_consumers = (tm.ring_records > 0 && ring_in[1] != 0) ? tm.node_size : 0;
    // an elastic ring needs its consumers' watermarks: without them it is just a plain window
    tm.elastic = tm.ring_consumers > 0 && ring_in[2] != 0;
    // elastic layout (records): [ring][guard][extension for the whole stream]; a plain/blocking ring is just the ring
    // (its guard follows the record area)
    const long long guard_records = (static_cast<long long>(kChaseRingGuardWords * sizeof(unsigned long long)) +
                                     static_cast<long long>(sizeof(hb2st_rot<T>)) - 1) / static_cast<long long>(sizeof(hb2st_rot<T>));
    tm.ext_offset = tm.elastic ? tm.ring_records + guard_records : 0;
    tm.capacity = tm.elastic ? tm.ext_offset + full_capacity : (tm.ring_records > 0 ? tm.ring_records : full_capacity);
    auto at = [](std::size_t off, std::size_t align) {
        return chase_team_align(off, align);
    };
    std::size_t off_ctl = 0;
    std::size_t off_flag = at(off_ctl + sizeof(ChaseTeamControl), 128);
    std::size_t off_cons = at(off_flag + sizeof(ChaseTeamFlag) * static_cast<std::size_t>(tm.members), 128);
    std::size_t off_band = at(off_cons + sizeof(ChaseTeamFlag) * static_cast<std::size_t>(tm.ring_consumers), 128);
    std::size_t off_dc = at(off_band + static_cast<std::size_t>(kd + 1) * static_cast<std::size_t>(n) * sizeof(T), 64);
    std::size_t off_wc = at(off_dc + (static_cast<std::size_t>(n) + 2) * sizeof(typename real_type<T>::type), 64);
    std::size_t off_rots = at(off_wc + (static_cast<std::size_t>(n) + 2) * sizeof(T), 64);
    std::size_t off_guard = at(off_rots + static_cast<std::size_t>(tm.capacity) * sizeof(hb2st_rot<T>), 64);
    std::size_t off_d = at(off_guard + (tm.ring_records > 0 && !tm.elastic ? kChaseRingGuardWords * sizeof(unsigned long long) : 0), 64);
    std::size_t off_e = at(off_d + static_cast<std::size_t>(n) * sizeof(typename real_type<T>::type), 64);
    std::size_t off_ph = at(off_e + static_cast<std::size_t>(n) * sizeof(typename real_type<T>::type), 64);
    std::size_t bytes = off_ph + static_cast<std::size_t>(n) * sizeof(T);
    // 127 bytes of slack put the control block on a 128-byte boundary
    // whatever the base alignment of the leader's segment
    const std::size_t allocation_bytes = bytes + 127;
    int myrc = MPI_SUCCESS;
    {
        void* mybase = nullptr;   // required out-arg; the shared_query below
                                  // resolves the leader's segment on every rank
        if (tm.node_rank == 0) {
            myrc = MPI_Win_allocate_shared(static_cast<MPI_Aint>(allocation_bytes), 8,
                                           MPI_INFO_NULL, tm.node_comm,
                                           &mybase, &tm.win);
        } else {
            myrc = MPI_Win_allocate_shared(0, 8, MPI_INFO_NULL, tm.node_comm,
                                           &mybase, &tm.win);
        }
    }
    int ok = (myrc == MPI_SUCCESS);
    MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, tm.node_comm);
    void* qbase = nullptr;
    if (ok) {
        MPI_Aint qsz = 0;
        int qdisp = 0;
        if (MPI_Win_shared_query(tm.win, 0, &qsz, &qdisp, &qbase) != MPI_SUCCESS ||
            qbase == nullptr ||
            static_cast<std::size_t>(qsz) < allocation_bytes)
            ok = 0;
    }
    // Direct cross-process atomics and the producer/consumer handoff require
    // one coherent shared mapping. A separate-model window cannot provide
    // that contract; every rank on this node chooses the same fallback
    // before any shared object is constructed or worker is started.
    if (ok) {
        int* model = nullptr;
        int present = 0;
        if (MPI_Win_get_attr(tm.win, MPI_WIN_MODEL, &model, &present) != MPI_SUCCESS ||
            !present || model == nullptr || *model != MPI_WIN_UNIFIED)
            ok = 0;
    }
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, tm.node_comm));
    if (!ok) {
        if (tm.win != MPI_WIN_NULL) MPI_CHECK(MPI_Win_free(&tm.win));
        tm.win = MPI_WIN_NULL;
        MPI_Comm_free(&tm.node_comm);
        tm.node_comm = MPI_COMM_NULL;
        return false;
    }

    char* p = static_cast<char*>(qbase) + (128 - reinterpret_cast<std::uintptr_t>(qbase) % 128) % 128;
    tm.ctl = reinterpret_cast<ChaseTeamControl*>(p + off_ctl);
    tm.flag = reinterpret_cast<ChaseTeamFlag*>(p + off_flag);
    tm.band = reinterpret_cast<T*>(p + off_band);
    tm.consumed = reinterpret_cast<ChaseTeamFlag*>(p + off_cons);
    tm.dc = reinterpret_cast<typename real_type<T>::type*>(p + off_dc);
    tm.wc = reinterpret_cast<T*>(p + off_wc);
    tm.rots = reinterpret_cast<hb2st_rot<T>*>(p + off_rots);
    tm.diag_array = reinterpret_cast<typename real_type<T>::type*>(p + off_d);
    tm.guard = tm.ring_records > 0 ? reinterpret_cast<unsigned long long*>(
                                         tm.elastic ? p + off_rots + static_cast<std::size_t>(tm.ring_records) * sizeof(hb2st_rot<T>) : p + off_guard)
                                   : nullptr;
    tm.e_arr = reinterpret_cast<typename real_type<T>::type*>(p + off_e);
    tm.ph_arr = reinterpret_cast<T*>(p + off_ph);

    if (tm.node_rank == 0) {
        // leader constructs the window's atomics over the raw MPI mapping
        // (reinterpret_cast alone leaves them formally unconstructed) and
        // then stages everything the members read before the setup barrier.
        // The other ranks begin their object lifetimes through the barrier:
        // constructing on their mapping view would clobber the leader's
        // values with the constructors' defaults.
        tm.ctl = new (p + off_ctl) ChaseTeamControl();
        tm.flag = reinterpret_cast<ChaseTeamFlag*>(p + off_flag);
        for (int q = 0; q < tm.members; ++q)
            new (tm.flag + q) ChaseTeamFlag();
        for (int q = 0; q < tm.ring_consumers; ++q)
            new (tm.consumed + q) ChaseTeamFlag();
        if (tm.guard != nullptr)
            for (std::size_t q = 0; q < kChaseRingGuardWords; ++q) tm.guard[q] = kChaseRingGuardPattern;
        for (int i = 0; i < n + 2; ++i) { tm.dc[i] = typename real_type<T>::type(0); tm.wc[i] = T(0); }
        tm.ctl->progress.v.store(0, std::memory_order_relaxed);
        tm.ctl->switch_pos.v.store(std::numeric_limits<long long>::max(), std::memory_order_relaxed);
        tm.ctl->phase_count.v.store(0, std::memory_order_relaxed);
        tm.ctl->arrive.v.store(0, std::memory_order_relaxed);
        tm.ctl->epoch.v.store(0, std::memory_order_relaxed);
        tm.ctl->aborted.v.store(0, std::memory_order_relaxed);
        for (int q = 0; q < tm.members; ++q)
            tm.flag[q].v.store(-1, std::memory_order_relaxed);
        for (int q = 0; q < tm.ring_consumers; ++q)
            tm.consumed[q].v.store(0, std::memory_order_relaxed);
    } else {
        tm.ctl = reinterpret_cast<ChaseTeamControl*>(p + off_ctl);
        tm.flag = reinterpret_cast<ChaseTeamFlag*>(p + off_flag);
    }
    MPI_Barrier(tm.node_comm);
    // DDLA_EIG_CHASE_TEAM_DEBUG_MAPS=1: print the window's /proc/self/maps
    // line -- identifies the backing (anonymous vs which file), which decides
    // where its page faults come from
    if (plan.debug_maps) {
        std::FILE* f = std::fopen("/proc/self/maps", "r");
        if (f != nullptr) {
            char line[512];
            const uintptr_t lo = reinterpret_cast<uintptr_t>(qbase);
            const uintptr_t hi = lo + bytes;
            while (std::fgets(line, sizeof(line), f) != nullptr) {
                unsigned long a = 0, b = 0;
                if (std::sscanf(line, "%lx-%lx", &a, &b) == 2 &&
                    lo >= a && lo < b) {
                    std::printf("TEAMPROF myid=%d maps %s", tm.myid, line);
                    break;
                }
            }
            std::fclose(f);
        }
    }
    MPI_Barrier(tm.node_comm);
    return true;
}

// Collective teardown (every rank of the node calls).
template <typename T>
void chase_team_teardown(ChaseTeam<T>& tm)
{
    if (tm.win != MPI_WIN_NULL) {
        MPI_Win_free(&tm.win);
        tm.win = MPI_WIN_NULL;
    }
    if (tm.node_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&tm.node_comm);
        tm.node_comm = MPI_COMM_NULL;
    }
}

// The chase as member @p m of @p v.members.  Runs the fused serial chase's
// exact per-bulge arithmetic (see host_chase for the phase semantics and the
// bit-identity argument); only the work division differs.  After the last
// step's barrier the leader (m == 0) runs host_chase_tail into the window's
// d/e/phase arrays; a closing barrier lets every member copy them out.
template <typename T>
void host_chase_team(const ChaseTeam<T>& v, int n, int kd, int m)
{
    using rt = typename real_type<T>::type;
    const int kd1 = kd + 1;
    const int kdm1 = kd - 1;
    const int kdn = std::min(n - 1, kd);
    T* ab = v.band;
    rt* dc = v.dc;
    T* wc = v.wc;
    hb2st_rot<T>* rots = v.rots;
    const long long ring_R = v.ring_records;   // > 0: record slots are reused round-robin
    const long long ring_maxt = static_cast<long long>(n) / (kd + 1) + 2;   // most records one step writes
    long long ring_min_cons = 0;               // cached minimum of the consumers' watermarks
    const int T_ = v.members;
    ChaseTeamControl* ctl = v.ctl;

    // plain accessor, exactly the serial chase's layout expression
    auto B = [ab, kd1](int row, int col) -> T& {
        return ab[static_cast<std::size_t>(col) * kd1 + row];
    };

    // sense-reversal barrier over the shared control block; same spin/yield
    // design as ChaseBarrier (the members share the node's cores)
    auto barrier = [&]() -> bool {
        const unsigned e = ctl->epoch.v.load(std::memory_order_relaxed);
        if (ctl->arrive.v.fetch_add(1, std::memory_order_acq_rel) + 1 == T_) {
            ctl->arrive.v.store(0, std::memory_order_relaxed);
            ctl->epoch.v.store(e + 1, std::memory_order_release);
            return ctl->aborted.v.load(std::memory_order_acquire) == 0;
        }
        int spins = 0;
        while (ctl->epoch.v.load(std::memory_order_acquire) == e) {
            if (++spins % 1024 == 0) {
                if (ctl->aborted.v.load(std::memory_order_acquire) != 0) return false;
                std::this_thread::yield();
            } else {
                chase_cpu_relax();
            }
        }
        return true;
    };

    long long handed = 0; // global record prefix copied at the last bounded handoff
    long long base = 0;    // this step's record base (redundant, pure arithmetic)
    long long step = 0;    // monotone step id the flags are checked against
    int nr_old = 0;
    int j1 = kdn + 2;
    int j2 = 1;

    // team profile (DDLA_EIG_PROFILE=1): per-member cycle accumulators and
    // thread rusage deltas; when the switch is off this costs one
    // predictable branch per measurement point and records nothing
    const bool prof = v.profile;
    unsigned long long c_pass1 = 0, c_wait = 0, c_pass2 = 0,
                       c_tail_wait = 0, c_tail = 0, c_barrier = 0;
    long long prof_steps = 0;
    struct rusage ru0 {}, ru1 {};
    if (prof) getrusage(RUSAGE_THREAD, &ru0);

    for (int i = 1; i <= n - 2; ++i) {
        for (int k = kdn + 1; k >= 2; --k) {
            // ---- per-step bookkeeping, identical to host_chase ----
            j1 += kdn;
            j2 += kdn;
            const bool ann = (k > 2);
            const int j1t = j1, j2t = j2;
            const int j1m = j1t - (ann ? kdn + 1 : 0);
            const int nr_new = nr_old + (ann ? 1 : 0);
            const bool exit_ = (j2t + kdn > n);
            const int j2m = j2t - (exit_ ? kdn + 1 : 0);
            const int t_span = (j2t > j1m - 1)
                                   ? (j2t - (j1m - 1) + kd1 - 1) / kd1 : 0;
            const int t_n = (n > j1m - 1)
                                ? (n - (j1m - 1) + kd1 - 1) / kd1 : 0;
            const int nrec = std::min(t_span, t_n);
            const long long stepbase = base;
            // record slot of step slot t: the stream position itself, or its position in the ring (a step never records more
            // than a quarter of the ring, so one wrap-around subtraction suffices); one division per step
            // An elastic ring is left for good at switch_pos (set by member 0 at the end of the step before it, ahead of the barrier every
            // member passed): from there on position g sits at ext_offset + (g - switch_pos) and never wraps.
            const long long sw = v.elastic ? ctl->switch_pos.v.load(std::memory_order_acquire) : std::numeric_limits<long long>::max();
            const bool in_ring = ring_R > 0 && stepbase < sw;
            const long long rbase = in_ring ? stepbase % ring_R : (ring_R > 0 ? v.ext_offset + (stepbase - sw) : stepbase);
            auto slot_of = [rbase, ring_R, in_ring](int t) -> long long {
                const long long s = rbase + t;
                return (in_ring && s >= ring_R) ? s - ring_R : s;
            };

            // generate: zero WORK against the band's top row (identical)
            auto do_generate = [&](long long s) {
                const int idx = j1t - 1 + static_cast<int>(s) * kd1;
                const int colx = idx - 1;
                const T f = B(0, colx);
                const T g = wc[idx];
                if constexpr (is_real_v<T>) {
                    if (g == T(0)) {
                        dc[idx] = rt(1);
                        wc[idx] = T(0);
                    } else if (f == T(0)) {
                        dc[idx] = rt(0);
                        wc[idx] = T(1);
                        B(0, colx) = g;
                    } else if (std::fabs(f) > std::fabs(g)) {
                        const T tt = g / f;
                        const T ttt = T(std::sqrt(rt(1) + tt * tt));
                        dc[idx] = rt(1) / ttt;
                        wc[idx] = tt * dc[idx];
                        B(0, colx) = f * ttt;
                    } else {
                        const T tt = f / g;
                        const T ttt = T(std::sqrt(rt(1) + tt * tt));
                        wc[idx] = rt(1) / ttt;
                        dc[idx] = tt * wc[idx];
                        B(0, colx) = g * ttt;
                    }
                } else {
                    if (g == T(0)) {
                        dc[idx] = rt(1);
                        wc[idx] = T(0);
                        B(0, colx) = f;
                    } else if (f == T(0)) {
                        dc[idx] = rt(0);
                        wc[idx] = std::conj(g) / std::abs(g);
                        B(0, colx) = T(std::abs(g), rt(0));
                    } else {
                        const rt f2 = f.real() * f.real() + f.imag() * f.imag();
                        const rt g2 = g.real() * g.real() + g.imag() * g.imag();
                        const rt f2s = std::sqrt(rt(1) + g2 / f2);
                        const T r = f2s * f;
                        dc[idx] = rt(1) / f2s;
                        wc[idx] = (r / (f2 + g2)) * std::conj(g);
                        B(0, colx) = r;
                    }
                }
            };

            // apply from the right (identical)
            auto do_apply_right = [&](long long s) {
                const int idx = j1t - 1 + static_cast<int>(s) * kd1;
                const int colx = idx - 1;
                T* const __restrict px =
                    ab + static_cast<std::size_t>(colx) * kd1;
                T* const __restrict py =
                    ab + static_cast<std::size_t>(colx + 1) * kd1;
                for (int l = 1; l < kd; ++l)
                    rot_pair<T>(dc[idx], wc[idx], px[l], py[l - 1]);
            };

            // Late read phases of one bulge -- identical arithmetic to the
            // serial chase's late_phases minus the create.  The create is
            // separate (do_create below): it overwrites B(0, idx+kd), which
            // the bulge one slot up reads in its generate, so the owner's
            // flag gates the top create of each range.  (The create's other
            // write, wc[idx+kd], is consumed in the next step, behind the
            // end-of-step barrier.)
            auto late_read = [&](int idx, bool is_last, long long slot) {
                // ---- ZLAR2V: both sides, diagonal blocks ----
                {
                    const int colx = idx - 1;
                    const int coly = idx;
                    lar2v_pair<T>(dc[idx], wc[idx], B(kd, colx), B(kd, coly), B(kd - 1, coly));
                }
                if constexpr (!is_real_v<T>) {
                    wc[idx] = std::conj(wc[idx]);
                }
                // ---- apply from the left ----
                {
                    const int lmax = is_last ? std::min(kdm1, n - j2t) : kdm1;
                    for (int lp = 1; lp <= lmax; ++lp) {
                        const int col = idx + lp;
                        if (col >= n) break;
                        rot_pair<T>(dc[idx], wc[idx],
                                    B(kd - 1 - lp, col), B(kd - lp, col));
                    }
                }
                // ---- record the Q rotation into its fixed slot ----
                if (slot >= 0) {
                    hb2st_rot<T> rec{};  // value-init: zero the padding after j0
                    rec.j0 = idx;
                    rec.c = dc[idx];
                    if constexpr (is_real_v<T>) rec.s = wc[idx];
                    else                        rec.s = std::conj(wc[idx]);
                    chase_record_store(rots,slot,rec);
                }
            };

            // The create half: overwrites B(0, idx+kd) -- read by the bulge
            // one slot up in its pass-1 generate -- and writes wc[idx+kd]
            // for the next step.  The boundary wait orders the overwrite
            // against the owning member's flag before it fires.
            auto do_create = [&](int idx, const int* wait_member,
                                 unsigned long long* wait_acc) {
                if (idx < j2m && idx + kd < n) {
                    if (wait_member != nullptr) {
                        const unsigned long long tw = wait_acc ? chase_cycles() : 0;
                        int spins = 0;
                        while (v.flag[*wait_member].v.load(std::memory_order_acquire) < step) {
                            if (++spins % 1024 == 0) {
                                if (ctl->aborted.v.load(std::memory_order_acquire) != 0) return;
                                std::this_thread::yield();
                            } else {
                                chase_cpu_relax();
                            }
                        }
                        if (wait_acc) *wait_acc += chase_cycles() - tw;
                    }
                    wc[idx + kd] = wc[idx] * B(0, idx + kd);
                    B(0, idx + kd) = dc[idx] * B(0, idx + kd);
                }
            };

            // ---- this member's contiguous bulge range, highest first ----
            const long long lo = static_cast<long long>(nr_old) * m / T_;
            const long long hi = static_cast<long long>(nr_old) * (m + 1) / T_;
            if (lo < hi) {
                const unsigned long long tp1 = prof ? chase_cycles() : 0;
                // ---- pass 1: everything but the creates ----
                // (pass 1 writes most of the band; what makes it parallel is
                // that every element it touches -- read or write -- lies in
                // the member's own bulge columns, which end one column below
                // the next range's lowest column.  The one cross-range
                // element, B(0, colx of the next range's lowest bulge), is
                // only ever WRITTEN by pass 2, after the neighbour's flag.)
                for (long long s = hi - 1; s >= lo; --s) {
                    const int t = static_cast<int>(s) + (ann ? 1 : 0);
                    const int idx = j1t - 1 + static_cast<int>(s) * kd1;
                    do_generate(s);
                    do_apply_right(s);
                    late_read(idx, s == nr_old - 1,
                              t < nrec ? slot_of(t) : -1);
                }
                // boundary rule: the flag goes up when the LOWEST bulge's
                // reads are done (generate through record).  The element at
                // stake is B(0, colx_lo): the member below's top create
                // overwrites it, and this member's generate must read the
                // pre-create value first.  (The create's other write,
                // wc[colx_lo + kd - 1], is consumed next step, behind the
                // end-of-step barrier.)
                if (prof) c_pass1 += chase_cycles() - tp1;
                v.flag[m].v.store(step, std::memory_order_release);
                const unsigned long long tp2 = prof ? chase_cycles() : 0;
                // ---- pass 2: the creates ----
                // create(s) overwrites what bulge s+1 read in pass 1; for
                // the range's top bulge that bulge belongs to the member
                // above, so wait for its flag.  Creates are otherwise
                // mutually independent (disjoint targets) and run descending
                // so each reads the previous create's B(0, .) product, as
                // the serial order does.
                for (long long s = hi - 1; s >= lo; --s) {
                    const int idx = j1t - 1 + static_cast<int>(s) * kd1;
                    int waitm = -1;
                    const int* waitp = nullptr;
                    if (s == hi - 1 && s + 1 < nr_old) {
                        // owner of bulge s+1 among the contiguous ranges:
                        // the first member q > m whose range ends above it
                        // (s+1 >= qlo_q holds by contiguity)
                        for (int q = m + 1; q < T_; ++q) {
                            if (s + 1 < static_cast<long long>(nr_old) * (q + 1) / T_) {
                                waitm = q;
                                break;
                            }
                        }
                        if (waitm < 0) waitm = T_ - 1;
                        waitp = &waitm;
                    }
                    do_create(idx, waitp, prof ? &c_wait : nullptr);
                }
                if (prof) c_pass2 += chase_cycles() - tp2;
            }

            // ---- member 0's tail is ordered by a targeted wait, not a
            // barrier.  Its only cross-member conflict is the new bulge's
            // create overwriting B(0, colx_0), which old bulge 0's pass-1
            // generate reads: when member 0 owns bulge 0 (nr_old >= members)
            // that generate ran in its own pass 1, and when member 0's range
            // is empty the wait below covers it.  The tail's other accesses
            // (columns idx_new-1 .. idx_new+kd-1) lie below old bulge 0's
            // lowest column idx_new+kd.  The end-of-step barrier still
            // fences the tail's writes (e.g. its wc product, consumed by the
            // next step's lowest generate) before anyone leaves the step.
            if (m == 0 && nr_old > 0 && hi == 0) {
                // member 0's range is empty: old bulge 0 belongs to the
                // first member with a non-empty range; wait for its pass-1
                // flag before the create can overwrite B(0, colx_0)
                int q = 0;
                while (q < T_ - 1 &&
                       static_cast<long long>(nr_old) * (q + 1) / T_ == 0)
                    ++q;
                const unsigned long long tw = prof ? chase_cycles() : 0;
                int spins = 0;
                while (v.flag[q].v.load(std::memory_order_acquire) < step) {
                    if (++spins % 1024 == 0) {
                        if (ctl->aborted.v.load(std::memory_order_acquire) != 0) return;
                        std::this_thread::yield();
                    } else {
                        chase_cpu_relax();
                    }
                }
                if (prof) c_tail_wait += chase_cycles() - tw;
            }

            // ---- member 0: annihilate, the new bulge's late phases, and
            // any phantom record fills -- exactly the serial position
            // between the old bulges and the end of the step ----
            if (m == 0) {
                const unsigned long long tt = prof ? chase_cycles() : 0;
                if (ann) {
                    if (k <= n - i + 1) {
                        rt c;
                        T sv, r;
                        givens_gen<T>(B(kd - k + 2, i + k - 3),
                                      B(kd - k + 1, i + k - 2), c, sv, r);
                        dc[i + k - 2] = c;
                        wc[i + k - 2] = sv;
                        B(kd - k + 2, i + k - 3) = r;
                        for (int l = 0; l < k - 3; ++l)
                            rot_pair<T>(c, sv,
                                        B(kd - k + 3 + l, i + k - 3),
                                        B(kd - k + 2 + l, i + k - 2));
                    }
                }
                if (ann && nr_new > 0) {
                    late_read(j1m - 1, nr_new == 1, nrec > 0 ? slot_of(0) : -1);
                    do_create(j1m - 1, nullptr, nullptr);
                }
                for (int t = nr_new; t < nrec; ++t) {
                    const int idx = j1m - 1 + t * kd1;
                    hb2st_rot<T> rec{};  // value-init: zero the padding after j0
                    rec.j0 = idx;
                    rec.c = dc[idx];
                    if constexpr (is_real_v<T>) rec.s = wc[idx];
                    else                        rec.s = std::conj(wc[idx]);
                    chase_record_store(rots,slot_of(t),rec);
                }
                if (prof) c_tail += chase_cycles() - tt;
            }

            // ---- end of step: barrier (the tail's writes must be visible
            // before any member starts the next step), then member 0
            // publishes the watermark ----
            // ring with consumers: the next step writes at most maxt records at stream positions
            // [stepbase + nrec, stepbase + nrec + maxt); each of them reuses the slot of the record R positions earlier, which the
            // slowest stager must have copied out.  Member 0 waits here, the others wait for it in the barrier below.  The wait is
            // over once every consumer has consumed `need` records; an abort or a failed stager (it publishes "infinity") ends it.
            // An ELASTIC ring does not wait: when the slowest stager is too far behind it leaves the ring (switch_pos = the next step's
            // first position) and the following steps write behind it; the unconsumed ring records stay where they are for the stagers.
            if (in_ring && v.ring_consumers > 0 && m == 0) {
                const long long need = stepbase + nrec + ring_maxt - ring_R;
                if (need > ring_min_cons) {
                    int spins = 0;
                    for (;;) {
                        long long mn = std::numeric_limits<long long>::max();
                        for (int q = 0; q < v.ring_consumers; ++q)
                            mn = std::min(mn, v.consumed[q].v.load(std::memory_order_acquire));
                        ring_min_cons = mn;
                        if (mn >= need) break;
                        if (v.elastic) {
                            ctl->switch_pos.v.store(stepbase + nrec, std::memory_order_release);
                            break;
                        }
                        if (ctl->aborted.v.load(std::memory_order_acquire) != 0) break;   // the barrier below reports it
                        if (++spins % 1024 == 0) std::this_thread::yield(); else chase_cpu_relax();
                    }
                }
            }
            const unsigned long long tb = prof ? chase_cycles() : 0;
            if (!barrier()) return;
            if (prof) {
                c_barrier += chase_cycles() - tb;
                ++prof_steps;
            }
            if (m == 0)
                ctl->progress.v.store(stepbase + nrec, std::memory_order_release);
            if (v.has_handoff() && (stepbase+nrec-handed >= v.batch_records || (i==n-2 && k==2))) {
                // Each rank's original application thread enters MPI. Extra
                // numerical members wait; no shared slot can be overwritten.
                if (m % v.tpr == 0) v.publish(stepbase+nrec);
                handed=stepbase+nrec;
                if (!barrier()) return;
            }

            nr_old = nr_new - (exit_ ? 1 : 0);
            j1 = j1m;
            j2 = j2m;
            ++step;
            base += nrec;
        }
    }

    // ---- tail: the leader extracts d/e/phases into the window; the closing
    // barrier lets every member copy them out (the caller does that, on the
    // rank's main thread) ----
    if (m == 0) {
        try {
            const long long pc = static_cast<long long>(
                host_chase_tail<T>(ab, kd, n, v.diag_array, v.e_arr, v.ph_arr));
            ctl->phase_count.v.store(pc, std::memory_order_release);
        } catch (...) {
            // release the team, then surface the failure on every member
            ctl->aborted.v.store(1, std::memory_order_release);
        }
    }
    (void)barrier();
    if (prof) {
        getrusage(RUSAGE_THREAD, &ru1);
        const double us = v.tsc_ghz > 0 ? 1.0 / (v.tsc_ghz * 1000.0) : 0.0; // cycles -> us
        std::printf("TEAMPROF myid=%d member=%d steps=%lld tsc_ghz=%.3f "
                    "pass1=%.1fus create_wait=%.1fus pass2=%.1fus "
                    "tail_wait=%.1fus tail=%.1fus barrier=%.1fus "
                    "th_minflt=%ld th_majflt=%ld th_nivcsw=%ld\n",
                    v.myid, m, prof_steps, v.tsc_ghz,
                    c_pass1 * us, c_wait * us, c_pass2 * us,
                    c_tail_wait * us, c_tail * us, c_barrier * us,
                    ru1.ru_minflt - ru0.ru_minflt,
                    ru1.ru_majflt - ru0.ru_majflt,
                    ru1.ru_nivcsw - ru0.ru_nivcsw);
    }
}

// Run the team on this node: the rank's main thread is member
// node_rank*tpr, DDLA_EIG_CHASE_TEAM_THREADS adds one more thread per rank.
// Throws when the team aborted.
template <typename T>
void chase_team_run(ChaseTeam<T>& tm,int n,int kd,Hb2stSink<T>* sink=nullptr,detail::DcHostTeam* prepared_team=nullptr)
{
    const int my_first = tm.node_rank * tm.tpr;
    const bool prof = tm.profile;
    struct rusage ru0 {}, ru1 {};
    if (prof) {
        // TSC calibration for the member lines
        const double t0 = MPI_Wtime();
        const unsigned long long c0 = chase_cycles();
        while (MPI_Wtime() - t0 < 0.002) chase_cpu_relax();
        tm.tsc_ghz = static_cast<double>(chase_cycles() - c0)
                     / ((MPI_Wtime() - t0) * 1e9);
    }
    const bool prefault = tm.prefault;
    const double tp0 = MPI_Wtime();
    double touch_s = 0.0;
    long pf_minflt = 0, pf_majflt = 0;
    if (prefault) {
        // each PROCESS touches every page it will fault on during the chase
        // (records, band, dc/wc), value-preserving so results are unchanged;
        // shared-memory first-touch faults cost more than private anonymous
        // ones and in a lockstep team one member's fault stalls all four
        struct rusage rp0 {}, rp1 {};
        if (prof) getrusage(RUSAGE_SELF, &rp0);
        auto touch = [](volatile char* p, std::size_t bytes) {
            for (std::size_t off = 0; off < bytes; off += 4096) {
                const char keep = p[off];
                p[off] = keep;
            }
        };
        touch(reinterpret_cast<volatile char*>(tm.rots),
              static_cast<std::size_t>(tm.capacity) * sizeof(hb2st_rot<T>));
        touch(reinterpret_cast<volatile char*>(tm.band),
              static_cast<std::size_t>(kd + 1) * static_cast<std::size_t>(n) * sizeof(T));
        touch(reinterpret_cast<volatile char*>(tm.dc),
              (static_cast<std::size_t>(n) + 2) * sizeof(typename real_type<T>::type));
        touch(reinterpret_cast<volatile char*>(tm.wc),
              (static_cast<std::size_t>(n) + 2) * sizeof(T));
        touch_s = MPI_Wtime() - tp0;
        if (prof) {
            getrusage(RUSAGE_SELF, &rp1);
            pf_minflt = rp1.ru_minflt - rp0.ru_minflt;
            pf_majflt = rp1.ru_majflt - rp0.ru_majflt;
        }
    }
    // No member may start the chase while another process is still in its
    // prefault: the touch is a non-atomic read-then-write, and member 0's
    // step-1 tail writes band/dc/wc/rots[0] bytes the other processes touch
    // too (a lost update).  The barrier also keeps prefault skew out of step
    // 1's barrier time.  Unconditional -- one intra-node barrier per call --
    // so a rank whose environment lacks the switch cannot hang the node.
    MPI_Barrier(tm.node_comm);
    const double tc0 = MPI_Wtime();
    if (prof) {
        if (prefault)
            std::printf("TEAMPROF myid=%d prefault_ms=%.1f prefault_wall_ms=%.1f "
                        "prefault_minflt=%ld prefault_majflt=%ld\n",
                        tm.myid, touch_s * 1e3, (tc0 - tp0) * 1e3,
                        pf_minflt, pf_majflt);
        // the rank-level rusage covers the chase only (after the prologue);
        // RUSAGE_SELF includes this process's member, helper and CUDA threads
        getrusage(RUSAGE_SELF, &ru0);
    }
    // Non-member ranks (CPU handles with more ranks per node than the team cap) run no chase member: they run the sink's roles (if
    // any) and then wait in the end-of-chase barrier below.  Members capture an exception instead of throwing past that barrier.
    // A CPU sink (extra_roles() > 0) runs its planner and accumulation workers as EXTRA THREADS OF THE SAME TEAM (thread w <
    // chase_threads: chase member, else sink role w - chase_threads), so they carry the binding the members get.
    std::exception_ptr member_error;
    const int chase_threads = tm.is_member ? tm.tpr : 0;
    const int extra_roles = sink != nullptr ? sink->extra_roles() : 0;
    const int team_threads = chase_threads + extra_roles;
    if (team_threads > 0) {
        // The extra members are OpenMP threads when the runtime can supply them (binding-safe: a std::thread
        // spawned after an OpenMP region inherits the pinned master's one-core mask and the lockstep team then
        // crawls -- 126 s instead of 4 s at n=5000 under OMP_PROC_BIND=close); otherwise plain threads.
        auto run_member = [&tm, n, kd, my_first, chase_threads, sink](int w) {
            if (w < chase_threads) {
                try {
                    host_chase_team<T>(tm, n, kd, my_first + w);
                } catch (...) {
                    // nothing in the step loop allocates; belt and braces
                    tm.ctl->aborted.v.store(1, std::memory_order_release);
                }
            } else {
                sink->run_role(w - chase_threads);   // never throws: a failure is recorded in the sink
            }
        };
        if (team_threads > 1 && detail::host_run_team(team_threads, run_member)) {
            // every member (this thread included) ran; an abort surfaces below
        } else if(prepared_team) {
            prepared_team->run_roles(team_threads,run_member);
        } else {
            std::vector<std::thread> ths;
            ths.reserve(static_cast<std::size_t>(std::max(team_threads - 1, 0)));
            for (int w = 1; w < team_threads; ++w)
                ths.emplace_back([&run_member, w] { run_member(w); });
            try {
                if (chase_threads > 0) host_chase_team<T>(tm, n, kd, my_first);
                else sink->run_role(0);
            } catch (...) {
                tm.ctl->aborted.v.store(1, std::memory_order_release);
                member_error = std::current_exception();
            }
            for (auto& th : ths) th.join();
        }
    }
    if (team_threads == 0 && tm.has_handoff()) {
        detail::ChaseCursor cursor(n,kd);
        long long handed=0;
        while (!cursor.done()) {
            cursor.next();
            if (cursor.base-handed >= tm.batch_records || cursor.done()) {
                tm.publish(cursor.base); handed=cursor.base;
            }
        }
    }
    if (tm.participants < tm.node_size) MPI_Barrier(tm.node_comm);   // identical condition on every rank of the node
    if (member_error) std::rethrow_exception(member_error);
    if (tm.ctl->aborted.v.load(std::memory_order_acquire) != 0)
        throw std::runtime_error(
            "phb2st node-team chase aborted (members=" +
            std::to_string(tm.members) + ")");
    if (tm.guard != nullptr) {
        // every member has left host_chase_team (its closing barrier): nothing writes the ring any more
        for (std::size_t q = 0; q < kChaseRingGuardWords; ++q)
            if (tm.guard[q] != kChaseRingGuardPattern)
                throw std::runtime_error("phb2st node-team chase: a record slot index ran past the ring (ring " +
                                         std::to_string(tm.ring_records) + " slots)");
    }
    if (prof && tm.node_rank == 0 && tm.ring_records > 0) {
        // where the chase left an elastic ring (-1: it never did: the stagers kept up for the whole stream)
        const long long swp = tm.ctl->switch_pos.v.load(std::memory_order_acquire);
        std::printf("TEAMPROF myid=%d window ring=%lld elastic=%d switched_at=%lld records=%lld\n", tm.myid, tm.ring_records,
                    tm.elastic ? 1 : 0, swp == std::numeric_limits<long long>::max() ? -1LL : swp,
                    tm.ctl->progress.v.load(std::memory_order_acquire));
    }
    if (prof && tm.is_member) {
        getrusage(RUSAGE_SELF, &ru1);
        std::printf("TEAMPROF myid=%d rank chase_ms=%.1f self_minflt=%ld "
                    "self_majflt=%ld self_nivcsw=%ld\n",
                    tm.myid, (MPI_Wtime() - tc0) * 1e3,
                    ru1.ru_minflt - ru0.ru_minflt,
                    ru1.ru_majflt - ru0.ru_majflt,
                    ru1.ru_nivcsw - ru0.ru_nivcsw);
    }
}

template<class T>class PreparedChaseTeam final:public spectral_impl::Phb2stTeamResources<T> {
    DdlaHandle_t h_;spectral_impl::Phb2stTeamPlan p_;ChaseTeam<T> team_;
    detail::DcHostTeam workers_;bool available_=false;
    bool wanted() const noexcept{return p_.enabled&&p_.participants*p_.tpr>1&&p_.band>1;}
public:
    PreparedChaseTeam(const DdlaHandle_t& h,const spectral_impl::Phb2stTeamPlan& p)
        :h_(h),p_(p),workers_(std::max(p.fallback_threads,p.tpr+p.sink_roles))
    {
        // The window chase_team_setup maps (records: the ring or the whole
        // stream; band, dc/wc, d/e/phases, flags and guard) must be one
        // MPI_Aint-addressable mapping.
        if(!wanted())return;using R=typename real_type<T>::type;
        const long double n=p.n,full=n*n/2+64,ring=static_cast<long double>(p.ring_request);
        const long double records=p.elastic?2*full+kChaseRingGuardWords:ring>0&&ring<full?ring:full;
        const long double bytes=records*sizeof(hb2st_rot<T>)+(p.band+1.0L)*n*sizeof(T)+(n+2)*(sizeof(R)+sizeof(T))+
            n*(2*sizeof(R)+sizeof(T))+kChaseRingGuardWords*sizeof(unsigned long long)+
            (static_cast<long double>(p.participants)*p.tpr+p.node_size+16)*128+127;
        if(bytes>static_cast<long double>(std::numeric_limits<MPI_Aint>::max()))
            throw std::overflow_error("phb2st: shared mapping exceeds MPI_Aint");
    }
    void prepare(){if(wanted())available_=chase_team_setup<T>(h_,p_,team_);}
    bool available() const noexcept override{return available_;}
    T* band_data() noexcept override{return available_?team_.band:nullptr;}
    detail::DcHostTeam& workers() noexcept override{return workers_;}
    void release() override{chase_team_teardown(team_);available_=false;}
    void reset_run(const T* band) override {
        assert(available_);MPI_CHECK(MPI_Barrier(team_.node_comm));
        if(team_.node_rank==0){const auto count=(std::size_t(p_.band)+1)*std::size_t(p_.n);
            std::memcpy(team_.band,band,count*sizeof(T));
            if constexpr(!is_real_v<T>)for(int col=0;col<p_.n;++col){auto& x=team_.band[std::size_t(col)*(p_.band+1)+p_.band];x=T(x.real(),0);}
            std::fill_n(team_.dc,std::size_t(p_.n)+2,typename real_type<T>::type(0));std::fill_n(team_.wc,std::size_t(p_.n)+2,T(0));
            team_.ctl->progress.v.store(0,std::memory_order_relaxed);team_.ctl->switch_pos.v.store(std::numeric_limits<long long>::max(),std::memory_order_relaxed);
            team_.ctl->phase_count.v.store(0,std::memory_order_relaxed);team_.ctl->arrive.v.store(0,std::memory_order_relaxed);
            team_.ctl->epoch.v.store(0,std::memory_order_relaxed);team_.ctl->aborted.v.store(0,std::memory_order_relaxed);
            for(int i=0;i<team_.members;++i)team_.flag[i].v.store(-1,std::memory_order_relaxed);
            for(int i=0;i<team_.ring_consumers;++i)team_.consumed[i].v.store(0,std::memory_order_relaxed);
            if(team_.guard)std::fill_n(team_.guard,kChaseRingGuardWords,kChaseRingGuardPattern);
        }MPI_CHECK(MPI_Barrier(team_.node_comm));
    }
    spectral_impl::Phb2stTeamStream<T> stream() noexcept override {
        spectral_impl::Phb2stTeamStream<T> result;if(!available_)return result;result.records=team_.rots;result.capacity=team_.capacity;
        result.emitted=&team_.ctl->progress.v;result.aborted=&team_.ctl->aborted.v;result.chase_threads=team_.is_member?team_.tpr:0;
        if(team_.ring_consumers){result.window.ring=team_.ring_records;result.window.ext_offset=team_.ext_offset;
            result.window.switch_pos=team_.elastic?&team_.ctl->switch_pos.v:nullptr;result.window.consumed=&team_.consumed[team_.node_rank].v;}
        return result;
    }
    void run(Hb2stSink<T>* sink,void* context,void (*handoff)(void*,long long),long long batch) override {
        assert(available_);team_.handoff_context=context;team_.handoff_run=handoff;team_.batch_records=batch;
        struct Retire{ChaseTeam<T>& t;~Retire(){t.handoff_context=nullptr;t.handoff_run=nullptr;}} retire{team_};
        chase_team_run(team_,p_.n,p_.band,sink,&workers_);
    }
    void outputs_run(typename real_type<T>::type* d,typename real_type<T>::type* e,T* phases,std::size_t& count,
        hb2st_rot<T>* records) override {
        assert(available_);std::memcpy(d,team_.diag_array,std::size_t(p_.n)*sizeof(*d));if(p_.n>1)std::memcpy(e,team_.e_arr,std::size_t(p_.n-1)*sizeof(*e));
        count=std::size_t(team_.ctl->phase_count.v.load(std::memory_order_acquire));
        for(std::size_t i=0;i<count;++i)::new(static_cast<void*>(phases+i))T(team_.ph_arr[i]);
        if(records){const auto n=std::size_t(team_.ctl->progress.v.load(std::memory_order_acquire));
            for(std::size_t i=0;i<n;++i)::new(static_cast<void*>(records+i))hb2st_rot<T>(team_.rots[i]);}
    }
};

} // namespace

#if DDLA_HAS_GPU
namespace {

// Fill the device band buffer with the entries of the 2D block-cyclic band
// this rank owns (zero elsewhere); a Grid-scope all-reduce completes it.
// The block-cyclic index math mirrors ddla_desc.h's host inlines.
template <typename T>
__global__ void band_extract_kernel(T* __restrict__ band, int n, int nbw,
                                    const T* __restrict__ a, int lld, int nb,
                                    int myprow, int mypcol, int npr, int npc)
{
    const std::size_t total = static_cast<std::size_t>(nbw + 1) * n;
    for (std::size_t p = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         p < total; p += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        // column-major band layout (matches host_chase's accessor)
        const int col = static_cast<int>(p / (nbw + 1));
        const int row = static_cast<int>(p % (nbw + 1));
        const int gi = col + row - nbw;
        if (gi < 0) continue;
        if ((col / nb) % npc != mypcol) continue;
        if ((gi / nb) % npr != myprow) continue;
        const int li = nb * (gi / (nb * npr)) + gi % nb;
        const int lj = nb * (col / (nb * npc)) + col % nb;
        band[p] = a[static_cast<std::size_t>(li) + static_cast<std::size_t>(lj) * lld];
    }
}

} // namespace
#endif

namespace spectral_impl {
template<class T>std::unique_ptr<Phb2stTeamResources<T>> phb2st_team_prepare(const DdlaHandle_t& h,const Phb2stTeamPlan& p)
{
    std::unique_ptr<PreparedChaseTeam<T>> resource;std::exception_ptr error;
    try{resource.reset(new PreparedChaseTeam<T>(h,p));}catch(...){error=std::current_exception();}
    detail::agree_failure(h->comm,error,"phb2st: worker preparation");
    resource->prepare();return resource;
}
#define DDLA_TEAM_CHASE(T) template std::unique_ptr<Phb2stTeamResources<T>> phb2st_team_prepare<T>(const DdlaHandle_t&,const Phb2stTeamPlan&)
DDLA_FOR_EACH_TYPE(DDLA_TEAM_CHASE)
#undef DDLA_TEAM_CHASE
// Scratch is carved in the order phb2st_host_workspace_size counts it.
template<class T>
void phb2st_host_run(const Phb2stHostPlan& p, detail::DcHostTeam* team, T* band,
    typename real_type<T>::type* d, typename real_type<T>::type* e,
    hb2st_rot<T>* records, T* phases, std::size_t& phase_count,
    std::atomic<long long>* progress, void* context,
    void (*publish)(void*,long long,const hb2st_rot<T>*,bool),
    void* host_work, std::size_t host_bytes)
{
    using R = typename real_type<T>::type;
    phase_count = 0;
    if (!p.n) return;
    assert(!p.bounded || p.threads == 1);

    char* cursor = static_cast<char*>(host_work);
    const std::size_t scratch = std::size_t(p.n) + 2;
    const std::size_t stride = std::size_t(p.n) + 1;
    R* dc = detail::take<R>(cursor, scratch);
    T* wc = detail::take<T>(cursor, scratch);
    std::size_t* counts = nullptr;
    hb2st_rot<T>* buffers = nullptr;
    if (p.records && p.threads > 1) {
        counts = detail::take<std::size_t>(cursor, std::size_t(p.threads));
        buffers = detail::take<hb2st_rot<T>>(cursor, std::size_t(p.threads)*stride);
    }
    assert(std::size_t(cursor-static_cast<char*>(host_work)) <= host_bytes);
    (void)host_bytes;

    if (p.threads > 1) {
        assert(team);
        if (p.records)
            host_chase_par<T,true>(band,p.n,p.band,d,e,records,phases,phase_count,
                p.threads,progress,team,dc,wc,counts,buffers,stride);
        else
            host_chase_par<T,false>(band,p.n,p.band,d,e,records,phases,phase_count,
                p.threads,progress,team,dc,wc,nullptr,nullptr,0);
        return;
    }
    if (!p.records) {
        host_chase<T,false>(band,p.n,p.band,d,e,records,phases,phase_count,
            progress,nullptr,nullptr,dc,wc);
        return;
    }
    if (p.bounded) {
        BoundedChaseOutput<T> output;
        output.capacity = p.record_capacity;
        output.context = context;
        output.publish_run = publish;
        host_chase<T,true,true>(band,p.n,p.band,d,e,records,phases,phase_count,
            progress,&output,nullptr,dc,wc);
        return;
    }
    host_chase<T,true>(band,p.n,p.band,d,e,records,phases,phase_count,
        progress,nullptr,nullptr,dc,wc);
}
#define DDLA_HOST_CHASE(T,R) template void phb2st_host_run<T>(const Phb2stHostPlan&,detail::DcHostTeam*,T*,R*,R*,hb2st_rot<T>*,T*,std::size_t&,std::atomic<long long>*,void*,void (*)(void*,long long,const hb2st_rot<T>*,bool),void*,std::size_t);
DDLA_HOST_CHASE(float,float)
DDLA_HOST_CHASE(double,double)
DDLA_HOST_CHASE(std::complex<float>,float)
DDLA_HOST_CHASE(std::complex<double>,double)
#undef DDLA_HOST_CHASE
Phb2stBandPlan phb2st_band_geometry(const DdlaHandle_t& h,DdlaBackend backend,int n,int band,const int* desc)
{
    Phb2stBandPlan p;p.backend=backend;p.n=n;p.band=band;p.nb=desc[DDLA_NB_];p.lld=desc[DDLA_LLD_];
    p.nprows=h->nprows_;p.npcols=h->npcols_;p.myprow=h->myprow_;p.mypcol=h->mypcol_;
    std::copy_n(desc,DDLA_DLEN_,p.descriptor.data());return p;
}
Phb2stBandPlan phb2st_band_check(const DdlaHandle_t& h,DdlaBackend backend,int n,int band,const void* A,const int* desc)
{
    auto p=detail::agree_check(h,"phb2st band",[&](detail::Agree& agreement) {
        agreement.mode("spectral operation",3123);
        if(!h||ddlaGetBackend(h)!=backend)throw std::invalid_argument("phb2st band: invalid handle/backend");
        if(h->nprows_<1||h->npcols_<1||n<0||band<1||band==INT_MAX||!desc||desc[DDLA_MB_]<1||desc[DDLA_NB_]<1||
            desc[DDLA_MB_]>INT_MAX/h->nprows_||desc[DDLA_NB_]>INT_MAX/h->npcols_)
            throw std::invalid_argument("phb2st band: invalid dimensions or block size");
        check_desc(desc,h);
        if(n>desc[DDLA_M_]||n>desc[DDLA_N_]||desc[DDLA_MB_]!=desc[DDLA_NB_]||desc[DDLA_RSRC_]!=0||desc[DDLA_CSRC_]!=0)
            throw std::invalid_argument("phb2st band: square source-zero descriptor required");
        const auto p=phb2st_band_geometry(h,backend,n,band,desc);
        if(n&&!A&&num_loc(n,p.nb,p.myprow,0,p.nprows)>0&&num_loc(n,p.nb,p.mypcol,0,p.npcols)>0)
            throw std::invalid_argument("phb2st band: missing active input");
        return p;
    });
    int signature[14]={3123,int(backend),n,band,h->nprows_,h->npcols_,int(h->major)};
    detail::descriptor_signature(signature+7,p.descriptor.data());detail::agree_signature(h,"phb2st band",signature);return p;
}
template<DdlaBackend Backend,class T>
void phb2st_band_run(const DdlaHandle_t& h,const Phb2stBandPlan& p,const T* A,T* host,
    void* backend_work,std::size_t backend_bytes)
{
    try {
        if(!p.n)return;
        const std::size_t elements=phb2st_band_count(p);char* w=static_cast<char*>(backend_work);
        T* band=detail::take<T>(w,elements);T* temporary=detail::take<T>(w,elements);
        assert(std::size_t(w-static_cast<char*>(backend_work))<=backend_bytes);(void)backend_bytes;
        const int n=p.n,nbw=p.band,nb=p.nb,lld=p.lld,npr=p.nprows,npc=p.npcols,myprow=p.myprow,mypcol=p.mypcol;
        if constexpr(Backend==DdlaBackend::CPU) {
            std::memset(band,0,elements*sizeof(T));
            for(int col=0;col<n;++col) {
                if((col/nb)%npc!=mypcol)continue;
                const int lj=nb*(col/(nb*npc))+col%nb;
                for(int gi=std::max(0,col-nbw);gi<=col;++gi) {
                    if((gi/nb)%npr!=myprow)continue;
                    const int li=nb*(gi/(nb*npr))+gi%nb,row=nbw+gi-col;
                    band[std::size_t(col)*(nbw+1)+row]=A[li+std::size_t(lj)*lld];
                }
            }
        } else {
#if DDLA_HAS_GPU
            DDLA_CHECK(ddlaMemset(h,band,0,elements*sizeof(T)));
            const unsigned blocks=static_cast<unsigned>(std::min(std::size_t(1024),(elements+255)/256));
            band_extract_kernel<T><<<blocks,256,0,static_cast<gpuStream_t>(h->stream)>>>(band,n,nbw,A,lld,nb,myprow,mypcol,npr,npc);
            DDLA_CHECK(ddlaGetLastError(h));DDLA_CHECK(ddlaSynchronize(h));
#endif
        }
        for(std::size_t offset=0;offset<elements;) {
            const int count=static_cast<int>(std::min(elements-offset,std::size_t(INT_MAX/2)));
            commAllReduce<Backend>(h,CommScope::Grid,band+offset,temporary+offset,count,cclSum);offset+=count;
        }
        if constexpr(Backend==DdlaBackend::CPU)std::memcpy(host,temporary,elements*sizeof(T));
        else {DDLA_CHECK(ddlaMemcpy(h,host,temporary,elements*sizeof(T)));DDLA_CHECK(ddlaSynchronize(h));}
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(h) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(h); } catch (...) {}
        throw;
    }
}
#define DDLA_BAND_RUN(B,T) template void phb2st_band_run<B,T>(const DdlaHandle_t&,const Phb2stBandPlan&,const T*,T*,void*,std::size_t)
DDLA_FOR_EACH_BACKEND_TYPE(DDLA_BAND_RUN)
#undef DDLA_BAND_RUN
} // namespace spectral_impl

#include "phb2st_spatial.h"

namespace spectral_impl {
Phb2stPlan phb2st_policy(const DdlaHandle_t& h,const Phb2stBandPlan& b,const Phb2stTopology& topology,
    Phb2stOutput output,Phb2stSinkPolicy sink)
{
    const auto& grid=*static_cast<const SpatialGrid*>(topology.state());Phb2stPlan p;
    detail::agree_check(h,"phb2st policy",[&](detail::Agree& agreement){agreement.mode("spectral operation",3143);
        if(grid.world!=h->comm||grid.rank!=h->myid||grid.ranks!=h->nprocs||int(output)<0||int(output)>int(Phb2stOutput::Sink)||
            (output==Phb2stOutput::Sink&&(!sink.ring_allowed||sink.chunk<1||sink.max_roles<0)))
            throw std::invalid_argument("phb2st: incompatible topology/output/sink policy");
        if(h->nprocs>1 && b.n>2 && b.band>eig_distributed_band_limit(b.n,h->nprocs))
            throw std::invalid_argument("phb2st: band is too wide for distributed storage");
    });
    p.input=b;p.output=output;p.sink=sink;p.spatial_enabled=topology.groups()>1;p.spatial=phb2st_spatial_geometry(h,b,topology,output==Phb2stOutput::History,output==Phb2stOutput::Sink,sink.chunk);
    detail::ChaseCursor cursor(b.n,b.band);while(!cursor.done())cursor.next();p.rotations=cursor.base;
    const auto maxt=static_cast<long long>(b.n)/(b.band+1)+2;const long long batch=detail::DistributedHistory<float>::chunk_size;
    p.local.n=b.n;p.local.band=b.band;p.local.records=output!=Phb2stOutput::None;
    p.local.bounded=output==Phb2stOutput::History||output==Phb2stOutput::Sink;
    p.local.record_capacity=output==Phb2stOutput::History?std::max(batch+maxt,4*maxt):0;
    if(output==Phb2stOutput::Sink){const auto chunks=std::max(std::size_t(8),chase_size_add(std::size_t(sink.chunk),std::size_t(2*maxt)+std::size_t(sink.chunk)-1)/std::size_t(sink.chunk));
        const auto cap=chase_size_mul(chunks,std::size_t(sink.chunk));if(cap>std::size_t(LLONG_MAX))throw std::overflow_error("phb2st: sink ring exceeds int64");
        p.local.record_capacity=static_cast<long long>(cap);}
    p.local.threads=p.local.bounded?1:chase_thread_count();
    auto& t=p.team;t.n=b.n;t.band=b.band;t.node_rank=topology.node_rank();t.node_size=topology.node_size();
    const int workers=h->host_threads;
    int default_threads=1;
    if(b.backend==DdlaBackend::CPU)
        default_threads=std::max(1,std::min(workers,4/std::max(1,t.node_size)));
    else if(b.n>=4096)
        default_threads=std::min(workers,2);
    t.tpr=chase_team_threads_per_rank(default_threads);t.participants=t.node_size;
    if(b.backend==DdlaBackend::CPU){int cap=0;if(const char* e=std::getenv("DDLA_EIG_CHASE_TEAM_RANKS"))cap=std::atoi(e);
        if(cap<=0)cap=std::max(1,4/t.tpr);t.participants=std::min(t.node_size,cap);}
    t.enabled=chase_team_enabled()&&!p.spatial_enabled;t.profile=eig_profile_enabled();t.prefault=chase_team_prefault_enabled();
    t.debug_maps=std::getenv("DDLA_EIG_CHASE_TEAM_DEBUG_MAPS")!=nullptr;t.fallback_threads=p.local.threads;t.sink_roles=sink.roles_in_team?sink.max_roles:0;
    t.ring_request=output==Phb2stOutput::None?chase_ring_records(b.n,b.band):p.local.bounded?p.local.record_capacity:0;
    t.consumers=output==Phb2stOutput::Sink;t.elastic=false;
    const int signature[]={3143,b.n,b.band,int(output),int(b.backend),p.spatial.groups,t.enabled?1:0,p.local.threads,
        sink.ring_allowed?1:0,sink.roles_in_team?1:0};detail::agree_signature(h,"phb2st policy",signature);
    long long minimum[2]={t.ring_request,sink.chunk},maximum[2]{};MPI_CHECK(MPI_Allreduce(minimum,maximum,2,MPI_LONG_LONG,MPI_MAX,h->comm));
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,minimum,2,MPI_LONG_LONG,MPI_MIN,h->comm));
    if(minimum[0]!=maximum[0]||minimum[1]!=maximum[1])throw std::invalid_argument("phb2st: ring policy differs between ranks");
    int node_values[2]={t.tpr,t.participants},node_min[2]{},node_max[2]{};
    MPI_CHECK(MPI_Allreduce(node_values,node_min,2,MPI_INT,MPI_MIN,grid.node));MPI_CHECK(MPI_Allreduce(node_values,node_max,2,MPI_INT,MPI_MAX,grid.node));
    int mismatch=node_min[0]!=node_max[0]||node_min[1]!=node_max[1];MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,&mismatch,1,MPI_INT,MPI_MAX,h->comm));
    if(mismatch)throw std::invalid_argument("phb2st: team policy differs within a node");
    return p;
}
template<class T>Phb2stPlan phb2st_check(const DdlaHandle_t& h,DdlaBackend backend,int n,int band,const T* A,const int* desc,
    typename real_type<T>::type* d,typename real_type<T>::type* e,T* phases,std::size_t phase_count,
    const Phb2stTopology& topology,Phb2stOutput output,Phb2stSinkPolicy sink)
{
    const auto b=phb2st_band_check(h,backend,n,band,A,desc);const auto p=phb2st_policy(h,b,topology,output,sink);
    detail::agree_check(h,"phb2st outputs",[&](detail::Agree&){
        if((n&&!d)||(n>1&&!e))throw std::invalid_argument("phb2st: missing tridiagonal output");
        if constexpr(!std::is_same<T,typename real_type<T>::type>::value)if(n>1&&(!phases||phase_count<std::size_t(n-1)))
            throw std::invalid_argument("phb2st: missing phase output");
    });return p;
}
template<class T>Phb2stResources<T>::Phb2stResources(const DdlaHandle_t& h,const Phb2stPlan& p,
    detail::DistributedHistory<T>* external_history)
{
    if(!p.input.n)return;
    if(!p.spatial_enabled)team_=phb2st_team_prepare<T>(h,p.team);
    if(p.output==Phb2stOutput::History){
        if(external_history){
            assert(external_history->ready&&external_history->n==p.input.n&&external_history->kd==p.input.band&&
                external_history->rank==p.spatial.rank&&external_history->ranks==p.spatial.ranks&&
                external_history->total==p.rotations);
            history_=external_history;
        }
        // prepare agrees on its failure; the team's mappings are then retired
        else try{owned_.prepare(p.input.n,p.input.band,h->comm);}catch(...){if(team_)team_->release();throw;}
    }
#if DDLA_HAS_GPU
    if(p.device_chase)try{device_setup(h,p);}catch(...){if(team_)team_->release();throw;}
#endif
}
template<DdlaBackend B,class T>void phb2st_run(const DdlaHandle_t& h,const Phb2stPlan& p,Phb2stResources<T>& resources,const T* A,
    typename real_type<T>::type* d,typename real_type<T>::type* e,T* phases,std::size_t& phase_count,
    Hb2stSink<T>* sink,void* backend_work,std::size_t backend_bytes,void* host_work,std::size_t host_bytes)
{
    phase_count=0;if(!p.input.n)return;const auto& b=p.input;const auto& grid=*static_cast<const SpatialGrid*>(p.spatial.topology);
    if(p.spatial_enabled){phb2st_spatial_run<B,T>(h,p.spatial,A,d,e,resources.history(),phases,phase_count,sink,
        backend_work,backend_bytes,host_work,host_bytes);return;}
    // The band, then one region: the gather's scratch, then the replicated
    // chase's record ring and scratch.
    char* const host_base=static_cast<char*>(host_work);char* hw=phb2st_band_align(host_base);
    const std::size_t band_count=phb2st_band_count(b);T* band=detail::take<T>(hw,band_count);
    const std::size_t rest=host_bytes-std::size_t(hw-host_base);
    phb2st_spatial_gather_run<B,T>(h,p.spatial,A,band,backend_work,backend_bytes,hw,rest);
    auto* team=resources.team();assert(team);
#if DDLA_HAS_GPU
    if(p.device_chase) {
        if constexpr(B==DdlaBackend::GPU) {
            T* reduced=band;hb2st_rot<T>* records=nullptr;long long capacity=0;
            if(team->available()) {
                team->reset_run(band);reduced=team->band_data();
                const auto stream=team->stream();records=stream.records;capacity=stream.capacity;
            } else {
                MPI_CHECK(MPI_Bcast(band,int(band_count),detail::mpi_datatype<T>(),0,h->comm));
                if(p.output==Phb2stOutput::History){records=detail::take<hb2st_rot<T>>(hw,std::size_t(p.local.record_capacity));capacity=p.local.record_capacity;}
            }
            resources.device_run(h,p,reduced,records,capacity);
            if(h->myid==0)phase_count=host_chase_tail<T>(reduced,b.band,b.n,d,e,phases);
            MPI_CHECK(MPI_Bcast(d,b.n,detail::mpi_datatype<typename real_type<T>::type>(),0,h->comm));
            MPI_CHECK(MPI_Bcast(e,std::max(0,b.n-1),detail::mpi_datatype<typename real_type<T>::type>(),0,h->comm));
            phase_count=std::is_same<T,typename real_type<T>::type>::value?0:std::size_t(std::max(0,b.n-1));
            if(phase_count)MPI_CHECK(MPI_Bcast(phases,int(phase_count),detail::mpi_datatype<T>(),0,h->comm));
            return;
        }
    }
#endif
    if(team->available()){
        team->reset_run(band);const auto stream=team->stream();
        if(sink){try{detail::agree_local(h->comm,"phb2st spatial allocation",[&]{sink->start_stager_run(stream.records,stream.capacity,stream.emitted,
            stream.window.ring?&stream.window:nullptr,stream.aborted,stream.chase_threads,p.sink.roles_in_team);});}catch(...){sink->cancel_run();throw;}}
        struct Handoff {const SpatialGrid& topology;const Phb2stTeamStream<T>& producer;detail::DistributedHistory<T>& history;long long handed=0;};
        Handoff handoff{grid,stream,resources.history()};
        void (*capture)(void*,long long)=[](void* pointer,long long end){auto& c=*static_cast<Handoff*>(pointer);MPI_CHECK(MPI_Barrier(c.topology.node));
            c.history.capture(c.handed,end,c.producer.records,c.producer.capacity);MPI_CHECK(MPI_Barrier(c.topology.node));c.handed=end;};
        try{team->run(sink,p.output==Phb2stOutput::History?&handoff:nullptr,p.output==Phb2stOutput::History?capture:nullptr,
            detail::DistributedHistory<T>::chunk_size);}
        catch(...){if(sink)sink->cancel_run();throw;}
        if(sink)sink->mark_chase_done();team->outputs_run(d,e,phases,phase_count,nullptr);
    }else{
        // the node leader gathered the whole band (one group): replicate it
        std::size_t bytes=band_count*sizeof(T);auto* data=reinterpret_cast<unsigned char*>(band);
        while(bytes){const auto count=int(std::min(bytes,std::size_t(INT_MAX)));MPI_CHECK(MPI_Bcast(data,count,MPI_BYTE,0,h->comm));data+=count;bytes-=count;}
        hb2st_rot<T>* local_records=p.local.bounded?detail::take<hb2st_rot<T>>(hw,std::size_t(p.local.record_capacity)):nullptr;
        const std::size_t local_bytes=host_bytes-std::size_t(hw-host_base);
        std::atomic<long long> emitted{0},consumed{0};std::atomic<int> aborted{0};StreamWindow window;window.ring=p.local.record_capacity;window.consumed=&consumed;
        if(sink){try{detail::agree_local(h->comm,"phb2st spatial allocation",[&]{sink->start_stager_run(local_records,p.local.record_capacity,&emitted,&window,&aborted,1,false);});}catch(...){sink->cancel_run();throw;}}
        struct Publish {const Phb2stPlan& p;detail::DistributedHistory<T>& history;std::atomic<long long>& emitted;std::atomic<long long>& consumed;
            std::atomic<int>& aborted;long long handed=0;};Publish publish{p,resources.history(),emitted,consumed,aborted};
        auto emit=[](void* pointer,long long end,const hb2st_rot<T>* ring,bool last){auto& c=*static_cast<Publish*>(pointer);
            if(c.p.output==Phb2stOutput::History){if(end-c.handed>=detail::DistributedHistory<T>::chunk_size||last){c.history.capture(c.handed,end,ring,c.p.local.record_capacity);c.handed=end;}}
            else if(c.p.output==Phb2stOutput::Sink){const auto maxt=static_cast<long long>(c.p.input.n)/(c.p.input.band+1)+2;c.emitted.store(end,std::memory_order_release);
                while(end+maxt-c.consumed.load(std::memory_order_acquire)>c.p.local.record_capacity){if(c.aborted.load(std::memory_order_acquire))throw std::runtime_error("phb2st: sink aborted");std::this_thread::yield();}}};
        try{phb2st_host_run<T>(p.local,&team->workers(),band,d,e,local_records,phases,phase_count,nullptr,&publish,emit,hw,local_bytes);}
        catch(...){aborted.store(1,std::memory_order_release);if(sink)sink->cancel_run();throw;}
        if(sink)sink->mark_chase_done();
    }
    if(sink){std::exception_ptr error;try{sink->drain_stage2_run();}catch(...){error=std::current_exception();}detail::agree_failure(h->comm,error,"phb2st: sink drain");}
}
#define DDLA_PHB_TYPE(T,R) template Phb2stPlan phb2st_check<T>(const DdlaHandle_t&,DdlaBackend,int,int,const T*,const int*,R*,R*,T*,std::size_t,const Phb2stTopology&,Phb2stOutput,Phb2stSinkPolicy);template Phb2stResources<T>::Phb2stResources(const DdlaHandle_t&,const Phb2stPlan&,detail::DistributedHistory<T>*);
DDLA_PHB_TYPE(float,float)
DDLA_PHB_TYPE(double,double)
DDLA_PHB_TYPE(std::complex<float>,float)
DDLA_PHB_TYPE(std::complex<double>,double)
#undef DDLA_PHB_TYPE
#define DDLA_PHB_RUN(B,T,R) template void phb2st_run<B,T>(const DdlaHandle_t&,const Phb2stPlan&,Phb2stResources<T>&,const T*,R*,R*,T*,std::size_t&,Hb2stSink<T>*,void*,std::size_t,void*,std::size_t);
DDLA_PHB_RUN(DdlaBackend::CPU,float,float)
DDLA_PHB_RUN(DdlaBackend::CPU,double,double)
DDLA_PHB_RUN(DdlaBackend::CPU,std::complex<float>,float)
DDLA_PHB_RUN(DdlaBackend::CPU,std::complex<double>,double)
#if DDLA_HAS_GPU
DDLA_PHB_RUN(DdlaBackend::GPU,float,float)
DDLA_PHB_RUN(DdlaBackend::GPU,double,double)
DDLA_PHB_RUN(DdlaBackend::GPU,std::complex<float>,float)
DDLA_PHB_RUN(DdlaBackend::GPU,std::complex<double>,double)
#endif
#undef DDLA_PHB_RUN

inline Phb2stBandPlan phb2st_owning_check(const DdlaHandle_t& h,DdlaBackend backend,int n,int band,
    const void* A,const int* descriptor,const void* diagonal,const void* off)
{
    const auto p=phb2st_band_check(h,backend,n,band,A,descriptor);
    detail::agree_check(h,"phb2st outputs",[&](detail::Agree&){
        if(n<1||!diagonal||(n>1&&!off))throw std::invalid_argument("phb2st: positive order and active outputs required");
    });return p;
}
} // namespace spectral_impl

// Private pheevd reduction. Eigenvalue-only calls retain no coefficients;
// vector calls retain compact coefficient chunks on unique global owners.
// Neither path builds a whole-node replay history.
template <DdlaBackend Backend, typename T>
void phb2st_node(
    const DdlaHandle_t& handle,
    int n, int nbw,
    T* A, const int* descA,
    typename real_type<T>::type* diagonal,
    typename real_type<T>::type* e_off,
    detail::DistributedHistory<T>& ns,
    std::vector<T>& phases,
    bool keep_stream,
    int& info)
{
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU phb2st_node is not available in this LibDDLA build");
    info=0;
    const auto input=spectral_impl::phb2st_owning_check(handle,Backend,n,nbw,A,descA,diagonal,e_off);
    ns.clear();
    auto topology=spectral_impl::phb2st_topology_prepare(handle);
    try{
        const auto output=keep_stream?spectral_impl::Phb2stOutput::History:spectral_impl::Phb2stOutput::None;
        const auto plan=spectral_impl::phb2st_policy(handle,input,*topology,output);
        std::size_t backend_bytes=0,host_bytes=0;vector<char> backend_work(handle);std::vector<char> host_work;
        detail::agree_local(handle->comm,"phb2st spatial allocation",[&]{
            spectral_impl::phb2st_workspace_size<T>(plan,&backend_bytes,&host_bytes);
            phases.resize(std::is_same<T,typename real_type<T>::type>::value?0:std::size_t(n-1));
            backend_work.resize(backend_bytes);host_work.resize(host_bytes);
        });
        // Retained output belongs to the caller's distributed history: the raw producer
        // fills that exact cyclic storage; no second history is prepared.
        if(keep_stream)ns.prepare(n,nbw,handle->comm);
        spectral_impl::Phb2stResources<T> resources(handle,plan,keep_stream?&ns:nullptr);
        std::size_t count=0;
        try{spectral_impl::phb2st_run<Backend,T>(handle,plan,resources,A,diagonal,e_off,phases.data(),count,nullptr,
            backend_work.data(),backend_bytes,host_work.data(),host_bytes);}
        catch(...){resources.release();throw;}
        phases.resize(count);resources.release();topology->release();
    }catch(...){topology->release();ns.clear();throw;}
}

#define INSTANTIATE_PHB2ST_NODE(BACKEND, TYPE)                                     \
    template void phb2st_node<BACKEND, TYPE>(                                      \
        const DdlaHandle_t&, int, int, TYPE*, const int*,            \
        typename real_type<TYPE>::type*, typename real_type<TYPE>::type*,          \
        detail::DistributedHistory<TYPE>&, std::vector<TYPE>&, bool, int&)

DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_PHB2ST_NODE)

#undef INSTANTIATE_PHB2ST_NODE

} // namespace ddla
