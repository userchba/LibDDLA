#ifndef DDLA_HB2ST_WAVE_SCHEDULE_H
#define DDLA_HB2ST_WAVE_SCHEDULE_H

// Host-only schedule used by the CPU accumulator and replay validation.
// Pure integer replay of the stage-2 chase's recording arithmetic: no matrix
// data, no device code, no vendor headers -- so test harnesses can exercise
// the exact schedule the sink consumes without a GPU.
//
// The chase records ~n^2/2 rotations whose (j0) sequence is a function of
// (n, kd) alone.  This header groups that sequence twice over:
//   * dependency levels (forward-greedy: a rotation's level is one past the
//     max level of the two columns it touches) -- the shipped accumulation
//     order, and
//   * capped domino waves plus window-disjoint launch batches -- the
//     wave-grouped order (DDLA_EIG_ACCUM=waves).
// A wave is a maximal run of records, taken across consecutive chase steps
// at the same slot index, whose j0 decreases by exactly 1 (the domino).
// Member p of a wave acts on the column pair (gjlo-p-1, gjlo-p), so a wave
// of gcnt members touches the consecutive columns [gjlo-gcnt, gjlo] inside
// its kernel window [gjlo-K, gjlo].  Rotations from different slots commute
// (slot column ranges sit kd1 = kd+1 apart), but a K-capped run continues
// into the next wave of the SAME slot at gjlo-K and the two waves share that
// boundary column -- chained waves must never run concurrently.  Launch
// batches therefore pack consecutive waves (table order) only while their
// windows stay pairwise disjoint, and batches are emitted in table order,
// which applies every shared-column pair in stream order.

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <vector>

namespace ddla {
namespace detail {

struct Hb2stSchedule {
    // dependency levels (0-based level l occupies records [lstart[l], lstart[l] + cnt_l[l]);
    // lstart has W+1 entries so lstart[W] is the stream length; end_l[l] is
    // the one-past record count at which level l+1's kernel may be enqueued)
    int W = 0;
    long long nrots = 0;
    std::vector<int> lstart;
    std::vector<int> cnt_l;
    std::vector<int> end_l;

    // wave grouping (capped domino runs)
    int K = 0;                  // wave-length cap: 16 real / 8 complex
    int nw = 0;
    std::vector<int> gjlo;      // per wave: j0 of the first member
    std::vector<int> gcnt;      // per wave: member count (<= K)
    std::vector<int> wend;      // per wave: one-past record count of its last member
    std::vector<int> midx;      // member p of wave w: record index midx[w*K + p]

    // window-disjoint launch batches over [0, nw)
    int nbatch = 0;
    std::vector<int> wbatch_lo;
    std::vector<int> wbatch_hi;
    std::vector<int> wbatch_gate;   // max wend over the batch (emission gate)
};

// Max waves per launch batch.  Only a cap: conflicts shrink the batch, and
// at kd=16 most batches close on a chained-wave clash well below it (the cap
// still binds at the tail: 512 cuts n=10000/kd=16 from 51796 to 19356
// launches, avg batch 151.6 waves).
constexpr int kHb2stWaveBatch = 512;

// Build the full schedule for an (n, kd) chase into @p sch.  @p is_real selects the wave
// cap; @p build_waves / @p build_levels select the tables (an unselected
// grouping keeps its empty tables; lstart always holds its W+1 entries).
// The tables and the builder's @p scratch keep their capacity from one call to
// the next.  Throws std::runtime_error if the wave partition fails to cover the
// stream exactly (an invariant of the grouping, checked here so every
// consumer -- including tests -- sees the same guarantee).
struct Hb2stScheduleScratch {
    std::vector<int> last,cnt,endv,run_j0,run_len,run_w,owner;
};
inline void build_hb2st_schedule_run(Hb2stSchedule& sch,Hb2stScheduleScratch& scratch,int n,int kd,bool is_real,
    bool build_waves=true,bool build_levels=true)
{
    sch.W=sch.nw=sch.nbatch=0;sch.nrots=0;
    sch.lstart.clear();sch.cnt_l.clear();sch.end_l.clear();sch.gjlo.clear();sch.gcnt.clear();sch.wend.clear();sch.midx.clear();
    sch.wbatch_lo.clear();sch.wbatch_hi.clear();sch.wbatch_gate.clear();
    const int kd1 = kd + 1;
    const int kdn = std::min(n - 1, kd);
    sch.K = is_real ? 16 : 8;

    if (kd <= 1) {
        // keep the level-table invariant (lstart holds the single prefix
        // entry) so consumers see a well-formed empty schedule
        sch.lstart.assign(1, 0);
        return;
    }

    // ---- one pass over the chase's record sequence ------------------------
    // last[] is the greedy level state; run_* is the per-slot wave-run state
    // (slot t spans [0, n/kd1 + 2)).
    auto& last=scratch.last;auto& cnt=scratch.cnt;auto& endv=scratch.endv;
    auto& run_j0=scratch.run_j0;auto& run_len=scratch.run_len;auto& run_w=scratch.run_w;
    const int maxt=n/kd1+2;
    if(build_levels){last.assign(n,0);cnt.assign(1,0);endv.assign(1,-1);}
    if(build_waves){run_j0.assign(maxt,0);run_len.assign(maxt,0);run_w.assign(maxt,0);}
    const int K = sch.K;
    int j1 = kdn + 2, j2 = 1;
    for (int i = 1; i <= n - 2; ++i) {
        for (int k = kdn + 1; k >= 2; --k) {
            j1 += kdn;
            j2 += kdn;
            if (k > 2)
                j1 -= kdn + 1;
            int t = 0;
            for (int j = j1 - 1; j < j2; j += kd1, ++t) {
                if (j - 1 < 0 || j >= n) continue;
                // ---- wave grouping (domino runs, capped at K) ----
                // a point continues slot t's open run iff its j0 is exactly
                // one below the run's expected next position; otherwise the
                // run ends and this point opens the next wave.  Skipped
                // steps leave a gap the check sees.
                if(build_waves) {
                if (run_len[static_cast<std::size_t>(t)] > 0 &&
                    j == run_j0[static_cast<std::size_t>(t)] -
                             run_len[static_cast<std::size_t>(t)] &&
                    run_len[static_cast<std::size_t>(t)] < K) {
                    const int w = run_w[static_cast<std::size_t>(t)];
                    sch.midx[static_cast<std::size_t>(w) * K +
                             run_len[static_cast<std::size_t>(t)]] =
                        static_cast<int>(sch.nrots);
                    ++run_len[static_cast<std::size_t>(t)];
                    sch.gcnt[static_cast<std::size_t>(w)] =
                        run_len[static_cast<std::size_t>(t)];
                    sch.wend[static_cast<std::size_t>(w)] =
                        static_cast<int>(sch.nrots) + 1;
                } else {
                    const int w = sch.nw++;
                    sch.gjlo.push_back(j);
                    sch.gcnt.push_back(1);
                    sch.wend.push_back(static_cast<int>(sch.nrots) + 1);
                    sch.midx.resize(static_cast<std::size_t>(sch.nw)*K,0);
                    sch.midx[static_cast<std::size_t>(w) * K] =
                        static_cast<int>(sch.nrots);
                    run_w[static_cast<std::size_t>(t)] = w;
                    run_len[static_cast<std::size_t>(t)] = 1;
                    run_j0[static_cast<std::size_t>(t)] = j;
                }
                }
                // ---- dependency level (forward-greedy) ----
                if(build_levels) {
                const int b = j, a = j - 1;
                const int lv =
                    1 + std::max(last[static_cast<std::size_t>(a)],
                                 last[static_cast<std::size_t>(b)]);
                last[static_cast<std::size_t>(a)] = lv;
                last[static_cast<std::size_t>(b)] = lv;
                if (lv > sch.W) {
                    sch.W = lv;
                    cnt.resize(static_cast<std::size_t>(sch.W) + 1, 0);
                    endv.resize(static_cast<std::size_t>(sch.W) + 1, -1);
                }
                ++cnt[static_cast<std::size_t>(lv)];
                // one-past the last record of level lv: the production count
                // at which lv's kernel may be enqueued (storing the record's
                // own index would gate one record early)
                endv[static_cast<std::size_t>(lv)] = static_cast<int>(sch.nrots)+1;
                }
                ++sch.nrots;
            }
            if (j2 + kdn > n)
                j2 -= kdn + 1;
        }
    }

    // the wave partition must cover the stream exactly
    if(build_waves) {
        long long wsum = 0;
        for (int w = 0; w < sch.nw; ++w)
            wsum += sch.gcnt[static_cast<std::size_t>(w)];
        if (wsum != sch.nrots)
            throw std::runtime_error(
                "Hb2stSchedule: wave partition covers " +
                std::to_string(wsum) + " of " + std::to_string(sch.nrots) +
                " records");
    }

    // ---- window-disjoint launch batches over the wave table ---------------
    // batch b collects consecutive waves while every window [gjlo-K, gjlo]
    // is disjoint from the others (overlap <=> |gjlo_a - gjlo_b| <= K);
    // chained waves share the boundary column gjlo-K and never co-batch.
    // owner[c] names the batch (by its first wave) whose window covers column
    // c, so the clash test is O(K) per wave instead of a scan over the batch
    // (which grew with kHb2stWaveBatch).  Windows are clamped to columns >= 0:
    // two windows with |X - g| <= K always share column min(X, g) >= 1, so
    // the clamped test sees every clash the unclamped one does.
    if(build_waves) {
        if(sch.nw) {
            sch.wbatch_lo.reserve(static_cast<std::size_t>(sch.nw / kHb2stWaveBatch +(sch.nw%kHb2stWaveBatch!=0)));
            sch.wbatch_hi.reserve(sch.wbatch_lo.size() + 1);
            sch.wbatch_gate.reserve(sch.wbatch_lo.size() + 1);
        }
        auto& owner=scratch.owner;owner.assign(static_cast<std::size_t>(n)+1,-1);
        int b0 = 0;
        while (b0 < sch.nw) {
            int b1 = b0 + 1;
            int gate = sch.wend[static_cast<std::size_t>(b0)];
            {
                const int X = sch.gjlo[static_cast<std::size_t>(b0)];
                for (int c = std::max(0, X - K); c <= X; ++c)
                    owner[static_cast<std::size_t>(c)] = b0;
            }
            while (b1 < sch.nw && (b1 - b0) < kHb2stWaveBatch) {
                const int X = sch.gjlo[static_cast<std::size_t>(b1)];
                bool clash = false;
                for (int c = std::max(0, X - K); c <= X; ++c) {
                    if (owner[static_cast<std::size_t>(c)] == b0) {
                        clash = true;
                        break;
                    }
                }
                if (clash) break;
                for (int c = std::max(0, X - K); c <= X; ++c)
                    owner[static_cast<std::size_t>(c)] = b0;
                gate = std::max(gate, sch.wend[static_cast<std::size_t>(b1)]);
                ++b1;
            }
            sch.wbatch_lo.push_back(b0);
            sch.wbatch_hi.push_back(b1);
            sch.wbatch_gate.push_back(gate);
            b0 = b1;
        }
        sch.nbatch = static_cast<int>(sch.wbatch_lo.size());
    }

    // exclusive prefix: level l (0-based) occupies [lstart[l], +cnt_l[l])
    sch.lstart.assign(static_cast<std::size_t>(sch.W) + 1, 0);
    for (int lv = 1; lv <= sch.W; ++lv)
        sch.lstart[static_cast<std::size_t>(lv)] =
            sch.lstart[static_cast<std::size_t>(lv) - 1] +
            cnt[static_cast<std::size_t>(lv)];
    // cnt/endv are indexed by level directly (slot 0 is the never-used
    // dummy); consumers are 0-based
    if(build_levels) {
        sch.cnt_l.assign(cnt.begin()+1,cnt.begin()+sch.W+1);
        sch.end_l.assign(endv.begin()+1,endv.begin()+sch.W+1);
    }
}
inline Hb2stSchedule build_hb2st_schedule(int n,int kd,bool is_real)
{
    Hb2stSchedule sch;Hb2stScheduleScratch scratch;
    build_hb2st_schedule_run(sch,scratch,n,kd,is_real);return sch;
}

// ---------------------------------------------------------------------------
// Streaming form of the wave schedule (2026-10-16).
//
// build_hb2st_schedule() materialises tables for the WHOLE stream: ~4.7 B per record per rank (3.7 GB at n=40000, on
// every rank, twice over while it is copied), although the consumer only ever needs the batches near the records it
// has staged.  Hb2stWaveGen replays the same recording arithmetic one chase step at a time and hands out the launch
// batches in table order, one at a time, with their own small tables.  The batch sequence -- waves, members, record
// indices, gates -- is IDENTICAL to the offline builder's (tests/.. wave_gen_check compares them for thousands of
// (n, kd, type) configurations), so the device sees the same kernels in the same order.  Memory is O(waves opened in a
// few dozen steps + n), independent of the stream length.
//
// Why a batch can be handed out early: a wave grows only by consecutive chase steps at one slot (its next member must
// sit exactly one column below the last one).  After the slot's chain breaks or reaches the cap the wave is final; a
// match after a gap of g steps would need g = 1 + m(kdn+1) with m >= 1 k==2 steps in between.  A batch is therefore
// handed out once the generator has run `lag_` (> 2 kdn + 6) steps past the step that opened its last wave, or at the end
// of the stream.  A (never expected) extension of a wave that was already handed out throws instead of corrupting the
// schedule.
//
// The waves not yet handed out and the sealed batches live in two rings
// whose capacity initialize() fixes from (n, kd): next() retains at most the
// oldest batch (<= kHb2stWaveBatch waves) plus the waves opened during its lag
// window (<= one per chase slot and step).  initialize() sizes every array, so
// re-initializing for the same (n, kd) allocates nothing, and neither does
// next() once the caller's Batch holds kHb2stWaveBatch waves of capacity.
// ---------------------------------------------------------------------------
class Hb2stWaveGen {
public:
    static constexpr int kMaxK = 16;

    struct Batch {
        int nw = 0;
        std::vector<int> gjlo;         // per wave: j0 of the first member
        std::vector<int> gcnt;         // per wave: member count (<= K)
        std::vector<long long> midx;   // member p of wave w: record index midx[w*K + p] (0 beyond gcnt)
        long long lowest = 0;          // record index of the first member of the first wave
        long long gate = 0;            // one past the last record of the batch's latest wave: emit once that many are staged
    };

    Hb2stWaveGen()=default;
    Hb2stWaveGen(int n,int kd,bool is_real){initialize(n,kd,is_real);}
    void initialize(int n,int kd,bool is_real) {
        n_=n;kd1_=kd+1;kdn_=std::min(n-1,kd);K_=is_real?16:8;
        finished_=false;j1_=0;j2_=0;k_=0;i_=1;step_=nrots_=lag_=0;
        wave_base_=nw_total_=0;max_buffered_=0;bid_=-1;bcount_=blast_open_=0;
        wave_head_=wave_count_=pending_head_=pending_count_=0;
        if (kd <= 1 || n < 3) { finished_ = true; return; }
        const int maxt = n / kd1_ + 2;
        run_j0_.assign(static_cast<std::size_t>(maxt), 0);
        run_len_.assign(static_cast<std::size_t>(maxt), 0);
        run_w_.assign(static_cast<std::size_t>(maxt), -1);
        owner_.assign(static_cast<std::size_t>(n) + 1, -1);
        j1_ = kdn_ + 2;
        j2_ = 1;
        k_ = kdn_ + 1;
        lag_ = 2LL * (kdn_ + 3) + 6;
        const std::size_t capacity = static_cast<std::size_t>(kHb2stWaveBatch) +
                                     static_cast<std::size_t>(lag_ + 1) * static_cast<std::size_t>(maxt);
        waves_.resize(capacity);
        pending_.resize(capacity);
    }

    int K() const { return K_; }

    /// Records generated so far (the stream length once next() has returned false).
    long long records() const { return nrots_; }
    long long waves() const { return nw_total_; }

    /// Largest number of waves the generator held at once (diagnostic: the memory bound).
    std::size_t max_buffered_waves() const { return max_buffered_; }

    /// The next launch batch in table order; false when the stream is exhausted.
    bool next(Batch& out)
    {
        for (;;) {
            if (pending_count_ > 0 && (finished_ || step_ >= pending_[pending_head_].last_open + lag_)) {
                extract(pending_[pending_head_], out);
                pending_head_ = (pending_head_ + 1) % pending_.size();
                --pending_count_;
                return true;
            }
            if (finished_) return false;
            if (!advance_step()) {
                if (bcount_ > 0) seal_building();
                bcount_ = 0;
                finished_ = true;
            }
        }
    }

    /// The number of records the chase will produce for (n, kd): the recording loop's arithmetic without any wave
    /// bookkeeping, O(1) per chase step (about n*kd steps).  Equal to build_hb2st_schedule(n, kd).nrots.
    static long long count_records(int n, int kd)
    {
        if (kd <= 1 || n < 3) return 0;
        const int kd1 = kd + 1;
        const int kdn = std::min(n - 1, kd);
        long long total = 0;
        long long j1 = kdn + 2, j2 = 1;
        for (int i = 1; i <= n - 2; ++i) {
            for (int k = kdn + 1; k >= 2; --k) {
                j1 += kdn;
                j2 += kdn;
                if (k > 2) j1 -= kdn + 1;
                // j = j1-1, j1-1+kd1, ... while j < j2; the chase records j with 1 <= j < n
                long long first = j1 - 1;
                if (first < 1) first += ((1 - first + kd1 - 1) / kd1) * kd1;
                const long long end = std::min<long long>(j2, n);
                if (end > first) total += (end - first + kd1 - 1) / kd1;
                if (j2 + kdn > n) j2 -= kdn + 1;
            }
        }
        return total;
    }

private:
    struct WaveRec {
        int gjlo = 0;
        int gcnt = 0;
        long long wend = 0;
        long long midx[kMaxK] = {};
    };
    struct Pending {
        long long count = 0;       // waves in the batch (they sit at the front of the wave ring in batch order)
        long long last_open = 0;   // step that opened the batch's last wave
    };

    int n_=0,kd1_=1,kdn_=0,K_=0;
    bool finished_ = false;
    int j1_ = 0, j2_ = 0, k_ = 0, i_ = 1;
    long long step_ = 0, nrots_ = 0, lag_ = 0;
    std::vector<int> run_j0_,run_len_;
    std::vector<long long> run_w_;     // absolute wave id of the slot's open run
    std::vector<WaveRec> waves_;       // ring of the waves not yet handed out; the oldest has id wave_base_
    std::size_t wave_head_ = 0, wave_count_ = 0;
    long long wave_base_ = 0, nw_total_ = 0;
    std::size_t max_buffered_ = 0;
    // batch composition (decided when a wave opens: it depends on gjlo only)
    std::vector<long long> owner_;
    long long bid_ = -1, bcount_ = 0, blast_open_ = 0;
    std::vector<Pending> pending_;     // ring of the sealed batches not yet handed out
    std::size_t pending_head_ = 0, pending_count_ = 0;

    WaveRec& wave_at(std::size_t i) { return waves_[(wave_head_ + i) % waves_.size()]; }

    void seal_building()
    {
        if (pending_count_ == pending_.size()) throw std::logic_error("Hb2stWaveGen: batch ring overflow");
        pending_[(pending_head_ + pending_count_) % pending_.size()] = Pending{bcount_, blast_open_};
        ++pending_count_;
    }

    void batch_add(int X)
    {
        const int lo = std::max(0, X - K_);
        bool start = (bcount_ == 0);
        if (!start) {
            if (bcount_ >= kHb2stWaveBatch) {
                start = true;
            } else {
                for (int c = lo; c <= X; ++c)
                    if (owner_[static_cast<std::size_t>(c)] == bid_) { start = true; break; }
            }
        }
        if (start) {
            if (bcount_ > 0) seal_building();
            ++bid_;
            bcount_ = 0;
        }
        for (int c = lo; c <= X; ++c) owner_[static_cast<std::size_t>(c)] = bid_;
        ++bcount_;
        blast_open_ = step_;
    }

    // one (i, k) step of the recording loop (the body of build_hb2st_schedule's double loop); false when none is left
    bool advance_step()
    {
        if (i_ > n_ - 2) return false;
        j1_ += kdn_;
        j2_ += kdn_;
        if (k_ > 2) j1_ -= kdn_ + 1;
        int t = 0;
        for (int j = j1_ - 1; j < j2_; j += kd1_, ++t) {
            if (j - 1 < 0 || j >= n_) continue;
            const std::size_t ts = static_cast<std::size_t>(t);
            const long long rec = nrots_;
            if (run_len_[ts] > 0 && j == run_j0_[ts] - run_len_[ts] && run_len_[ts] < K_) {
                const long long id = run_w_[ts];
                if (id < wave_base_)
                    throw std::logic_error("Hb2stWaveGen: a wave that was already handed out is extended (lag too short)");
                WaveRec& w = wave_at(static_cast<std::size_t>(id - wave_base_));
                w.midx[run_len_[ts]] = rec;
                ++run_len_[ts];
                w.gcnt = run_len_[ts];
                w.wend = rec + 1;
            } else {
                if (wave_count_ == waves_.size()) throw std::logic_error("Hb2stWaveGen: wave ring overflow");
                WaveRec& w = wave_at(wave_count_++);
                w = WaveRec();
                w.gjlo = j;
                w.gcnt = 1;
                w.wend = rec + 1;
                w.midx[0] = rec;
                run_w_[ts] = nw_total_++;
                run_len_[ts] = 1;
                run_j0_[ts] = j;
                batch_add(j);
                max_buffered_ = std::max(max_buffered_, wave_count_);
            }
            ++nrots_;
        }
        if (j2_ + kdn_ > n_) j2_ -= kdn_ + 1;
        ++step_;
        if (--k_ < 2) { k_ = kdn_ + 1; ++i_; }
        return true;
    }

    void extract(const Pending& pb, Batch& out)
    {
        out.nw = static_cast<int>(pb.count);
        out.gjlo.assign(static_cast<std::size_t>(pb.count), 0);
        out.gcnt.assign(static_cast<std::size_t>(pb.count), 0);
        out.midx.assign(static_cast<std::size_t>(pb.count) * K_, 0);
        out.gate = 0;
        for (long long w = 0; w < pb.count; ++w) {
            const WaveRec& r = wave_at(static_cast<std::size_t>(w));
            out.gjlo[static_cast<std::size_t>(w)] = r.gjlo;
            out.gcnt[static_cast<std::size_t>(w)] = r.gcnt;
            for (int p = 0; p < r.gcnt; ++p) out.midx[static_cast<std::size_t>(w) * K_ + p] = r.midx[p];
            out.gate = std::max(out.gate, r.wend);
        }
        out.lowest = out.midx[0];
        wave_head_ = (wave_head_ + static_cast<std::size_t>(pb.count)) % waves_.size();
        wave_count_ -= static_cast<std::size_t>(pb.count);
        wave_base_ += pb.count;
    }
};

} // namespace detail
} // namespace ddla

#endif // DDLA_HB2ST_WAVE_SCHEDULE_H
