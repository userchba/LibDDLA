// CPU implementation of the stage-4 sink (see hb2st_sink.h): Q_band is accumulated WHILE the stage-2 chase runs, the same
// design the GPU's Hb2stAccum uses, with the host's own cost structure.
//
//   * Q_band^T is stored per rank as TILES of kW lanes (256 bytes of a row: 32 real doubles, 16 complex doubles, ...), tile-major:
//     tile t holds Q_band rows row_lo + t*kW .. +kW-1, and inside the tile the n columns of Q_band are n rows of kW lanes
//     (complex: a real plane then an imaginary plane per row).  A rotation (j0, c, s) mixes the tile rows j0-1 and j0, every lane
//     independently:   a' = c*a + s*b ;  b' = c*b - conj(s)*a   with explicit rounding of the products and fused sums.
//   * The stream is cut into BLOCKS of consecutive chase steps (about DDLA_EIG_HOST_ACCUM_BLOCK records).  A planner thread
//     replays the chase's recording arithmetic (a pure function of (n, kd)) for the steps of a block, groups the records into
//     dominoes (one bulge's consecutive steps: adjacent row pairs, applied as a register-carried chain), assigns each domino a
//     dependency level inside the block and orders the dominoes into skewed cache tiles (bands of RB rows, one domino length of
//     skew per level, bands descending, level inside a band): a tile then touches only ~RB + skew*levels rows at a time and stays in
//     L2.  Any level assignment that respects the overlaps gives a valid order (the argument of eig_cpu_kernels.cpp); a record's
//     j0 is checked against the replay while it is copied into the block's coefficient arrays.
//   * The workers apply a published block to their own tiles in table order and count it done; the planner reuses a block slot when
//     every worker is through.  The records leave the chase window block by block (consumed watermark: the chase waits when the
//     ring is full -- on CPU the accumulation is the bottleneck, so waiting costs nothing and the window stays a few MB).
//   * The planner and the workers run as EXTRA THREADS OF THE CHASE TEAM'S OpenMP region (extra_roles/run_role), so they inherit the
//     binding the chase members get (OMP_PROC_BIND/OMP_PLACES); a std::thread started after an OpenMP region would inherit the pinned
//     master's one-core mask.  Stage 2 therefore returns with the stream fully consumed.
//   * finish_rows forms W = Q_band * (D*Y) in bounded output-column panels. Each panel is immediately redistributed
//     into the 2D block-cyclic Z; Q_band and Y remain immutable until all panels complete.
//
// Measured on p1 (n=10000, kd=64): the chase slows ~35-45% next to the accumulators but is hidden behind them, and the accumulation
// replaces the replay that ran after stage 3; pheevd_check decides where this pays.
//
// Q_band and the planner's block slots, level state and step generator live in the caller's buffer
// (hb2st_accum_host_workspace_size), bound by reset_run; only the threads and the synchronization are the sink's own.

#include "helpers.h"
#include "hb2st_sink.h"
#include "hb2st_finish_panel.h"
#include "hb2st_host_workspace.h"
#include "BLAS/blas_impl.h"
#include "check_agree.h"
#include "ddla_desc.h"
#include "ddla_connector.h"

#include "ddla_stream.h"   // the handle's fields (comm, myid, major)
#include "distributed_history.h"
#include "Eigensolver/eig_cpu_kernels.h"
#include "Eigensolver/eig_params.h"
#include "row_share.h"
#include "hb2st_wave_schedule.h"
#include "mpi_datatype.h"
#include "plan_util.h"
#include "instantiate.h"

#include <mpi.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace ddla {


namespace {

#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
#define HS_CLONES __attribute__((target_clones("arch=skylake-avx512", "arch=haswell", "default")))
#define HS_INLINE __attribute__((always_inline)) inline
#else
#define HS_CLONES
#define HS_INLINE inline
#endif

HS_INLINE float rfma(float a, float b, float c) { return __builtin_fmaf(a, b, c); }
HS_INLINE double rfma(double a, double b, double c) { return __builtin_fma(a, b, c); }

inline void cpu_relax()
{
#if defined(__x86_64__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}

double wall()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Q_band tiles and the finish panels start on a 64-byte boundary (the tile kernels' vectors, as posix_memalign gave them);
// a malloc'd buffer is only 16-aligned, so each size reserves 64 bytes of slack and the run rounds its cursor up.
constexpr std::size_t kTileAlignment = 64;
inline char* align_tiles(char* p)
{
    return reinterpret_cast<char*>((reinterpret_cast<std::uintptr_t>(p) + kTileAlignment - 1) / kTileAlignment * kTileAlignment);
}

long long env_ll(const char* name, long long dflt, long long lo)
{
    if (const char* s = std::getenv(name))
        if (*s != '\0') return std::max(lo, std::strtoll(s, nullptr, 10));
    return dflt;
}

// ---------------------------------------------------------------------------------------------------------------------
// Tile kernels.  X points at a tile: n rows of kW lanes (real) or n rows of [kW real | kW imaginary] lanes (complex).
// The dominoes of a block are stored in application order: domino o has gcnt[o] members on the row pairs
// (top-p-1, top-p), p = 0..cnt-1, coefficients at woff[o] + p; the carried row stays in registers.
// ---------------------------------------------------------------------------------------------------------------------
template <typename rt, int W>
HS_CLONES void apply_block_real(rt* __restrict X, int nw, const int* gjlo, const int* gcnt, const std::int32_t* woff, const rt* cs,
                                const rt* sn)
{
    for (int o = 0; o < nw; ++o) {
        const int cnt = gcnt[o], top = gjlo[o];
        const std::int32_t w0 = woff[o];
        rt x[W];
        const rt* src = X + static_cast<std::size_t>(top) * W;
        for (int k = 0; k < W; ++k) x[k] = src[k];
        for (int p = 0; p < cnt; ++p) {
            const rt c = cs[w0 + p], s = sn[w0 + p];
            rt* rb = X + static_cast<std::size_t>(top - p) * W;
            const rt* ra = rb - W;
#pragma GCC ivdep
            for (int k = 0; k < W; ++k) {
                const rt va = ra[k], vb = x[k];
                const rt na = rfma(c, va, s * vb);
                const rt nb = rfma(c, vb, -(s * va));
                rb[k] = nb;
                x[k] = na;
            }
        }
        rt* last = X + static_cast<std::size_t>(top - cnt) * W;
        for (int k = 0; k < W; ++k) last[k] = x[k];
    }
}

template <typename rt, int W>
HS_CLONES void apply_block_cplx(rt* __restrict X, int nw, const int* gjlo, const int* gcnt, const std::int32_t* woff, const rt* cs,
                                const rt* sn)
{
    constexpr std::size_t RS = 2 * static_cast<std::size_t>(W);
    for (int o = 0; o < nw; ++o) {
        const int cnt = gcnt[o], top = gjlo[o];
        const std::int32_t w0 = woff[o];
        rt xr[W], xi[W];
        const rt* src = X + static_cast<std::size_t>(top) * RS;
        for (int k = 0; k < W; ++k) { xr[k] = src[k]; xi[k] = src[W + k]; }
        for (int p = 0; p < cnt; ++p) {
            const rt c = cs[w0 + p];
            const rt sr = sn[2 * (w0 + p)], si = sn[2 * (w0 + p) + 1];
            rt* rb = X + static_cast<std::size_t>(top - p) * RS;
            const rt* ra = rb - RS;
#pragma GCC ivdep
            for (int k = 0; k < W; ++k) {
                const rt ar = ra[k], ai = ra[W + k], br = xr[k], bi = xi[k];
                // s*b = (sr*br - si*bi, sr*bi + si*br);  conj(s)*a = (sr*ar + si*ai, sr*ai - si*ar)
                const rt sb_r = rfma(sr, br, -(si * bi));
                const rt sb_i = rfma(sr, bi, si * br);
                const rt sa_r = rfma(sr, ar, si * ai);
                const rt sa_i = rfma(sr, ai, -(si * ar));
                const rt nar = rfma(c, ar, sb_r), nai = rfma(c, ai, sb_i);
                const rt nbr = rfma(c, br, -sa_r), nbi = rfma(c, bi, -sa_i);
                rb[k] = nbr;
                rb[W + k] = nbi;
                xr[k] = nar;
                xi[k] = nai;
            }
        }
        rt* last = X + static_cast<std::size_t>(top - cnt) * RS;
        for (int k = 0; k < W; ++k) { last[k] = xr[k]; last[W + k] = xi[k]; }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The chase's recording arithmetic, one step at a time (the loop of phb2st's recording and of Hb2stWaveGen::advance_step),
// grouping the records of the CURRENT BLOCK into dominoes: a slot's record joins the slot's open domino when its j0 is one below the
// domino's last and the domino is shorter than K.  Dominoes never span blocks (the runs are reset at a block start).
// ---------------------------------------------------------------------------------------------------------------------
struct StepGen {
    int n, kd1, kdn, K;
    int i = 1, k, j1, j2;
    long long nrots = 0;      // records generated so far (the next record's stream position)
    long long begin = 0;      // stream position of the block's first record
    std::size_t slots;
    int *run_j0, *run_len, *run_w;
    int *gj, *gc;             // dominoes of the block: j0 of the first member, member count
    std::int32_t *first, *last, *next;   // per-domino first/last and per-record next offsets, all relative to begin
    int nw = 0;               // dominoes of the block

    // Bytes the generator carves: run_j0, run_len, run_w (slots each), then gj, gc, first, last, next (records each).
    static std::size_t bytes(std::size_t slots, std::size_t records)
    {
        return 3 * detail::padded_bytes<int>(slots) + 2 * detail::padded_bytes<int>(records) +
               3 * detail::padded_bytes<std::int32_t>(records);
    }

    StepGen(int n_, int kd, int K_, std::size_t slots_, std::size_t records, char* cursor)
        : n(n_), kd1(kd + 1), kdn(std::min(n_ - 1, kd)), K(K_), k(kdn + 1), j1(kdn + 2), j2(1), slots(slots_)
    {
        run_j0 = detail::take<int>(cursor, slots); run_len = detail::take<int>(cursor, slots); run_w = detail::take<int>(cursor, slots);
        gj = detail::take<int>(cursor, records); gc = detail::take<int>(cursor, records);
        first = detail::take<std::int32_t>(cursor, records); last = detail::take<std::int32_t>(cursor, records);
        next = detail::take<std::int32_t>(cursor, records);
        std::fill(run_j0, run_j0 + slots, 0); std::fill(run_len, run_len + slots, 0); std::fill(run_w, run_w + slots, 0);
    }

    void start_block()
    {
        begin = nrots;
        nw = 0;
        std::fill(run_len, run_len + slots, 0);
    }
    long long block_records() const { return nrots - begin; }
    bool exhausted() const { return i > n - 2; }

    // one (i, k) step; false when the recording loop is over
    bool step()
    {
        if (i > n - 2) return false;
        j1 += kdn;
        j2 += kdn;
        if (k > 2) j1 -= kdn + 1;
        int t = 0;
        for (int j = j1 - 1; j < j2; j += kd1, ++t) {
            if (j - 1 < 0 || j >= n) continue;
            const std::size_t ts = static_cast<std::size_t>(t);
            const std::int32_t rel = static_cast<std::int32_t>(nrots - begin);
            next[rel] = -1;
            if (run_len[ts] > 0 && j == run_j0[ts] - run_len[ts] && run_len[ts] < K) {
                const int w = run_w[ts];
                next[last[w]] = rel; last[w] = rel;
                ++run_len[ts];
                gc[w] = run_len[ts];
            } else {
                const int w = nw++;
                gj[w] = j;
                gc[w] = 1;
                first[w] = rel; last[w] = rel;
                run_w[ts] = w;
                run_len[ts] = 1;
                run_j0[ts] = j;
            }
            ++nrots;
        }
        if (j2 + kdn > n) j2 -= kdn + 1;
        if (--k < 2) { k = kdn + 1; ++i; }
        return true;
    }
};

// The sort key of the skewed cache-tile order (build_block).
struct Key {
    long long bandrev;
    int level, top, w;
};

template <typename T>
class Hb2stAccumHost final : public detail::Hb2stAccumHostResources<T> {
public:
    using rt = typename real_type<T>::type;
    static constexpr bool kReal = std::is_same<T, rt>::value;
    static constexpr int kW = static_cast<int>(256 / sizeof(T));          // lanes per tile
    static constexpr std::size_t kRowReals = kReal ? static_cast<std::size_t>(kW) : 2 * static_cast<std::size_t>(kW);

    Hb2stAccumHost(const DdlaHandle_t&,const detail::Hb2stAccumHostPlan&);
    ~Hb2stAccumHost() override;
    void reset_run(void* backend_work,std::size_t backend_bytes) override;
    void start_stager_run(const hb2st_rot<T>*,long long,const std::atomic<long long>*,const StreamWindow*,const std::atomic<int>*,int,bool) override;
    void drain_stage2_run() override {join_roles();if(error_)std::rethrow_exception(error_);}
    void cancel_run() override {failed_.store(1,std::memory_order_release);chase_done_.store(1,std::memory_order_release);join_roles();}
    void finish_rows_run(const detail::Hb2stFinishHostPlan&,const rt*,const T*,T*,int&,void*,std::size_t) override;

    bool window_ring_allowed() const override { return true; }
    long long stream_chunk() const override { return block_recs_; }
    bool elastic_ring() const override { return false; }
    bool drains_in_stage2() const override { return true; }

    int extra_roles() const override { return 1 + nacc_; }
    void run_role(int role) override;
    void mark_chase_done() override { chase_done_.store(1, std::memory_order_release); }
    std::atomic<long long>* fallback_emitted() override { return &fb_emitted_; }
    const std::exception_ptr& consumer_error() const override { return error_; }
    long long scheduled_rotations() const override { return detail::Hb2stWaveGen::count_records(n_, kd_); }

private:
    struct Block {
        int nw = 0;
        long long begin = 0, end = 0;
        int* gjlo = nullptr;
        int* gcnt = nullptr;
        std::int32_t* woff = nullptr;
        rt* c = nullptr;                      // coefficients in application order; s: complex = (re, im) pairs
        rt* s = nullptr;
        std::atomic<int> done{0};             // workers through with this block
    };

    DdlaHandle_t h_;
    int n_, kd_, threads_;
    int npr_ = 1, npc_ = 1, mypr_ = 0, mypc_ = 0, myid_ = 0, P_ = 1;
    char major_ = 'C';
    int nrl_ = 0, row_lo_ = 0;
    long long block_recs_ = 524288;
    int blk_slots_ = 3, rb_ = 512;

    // the stream
    const hb2st_rot<T>* rots_ = nullptr;
    long long cap_ = 0;
    const std::atomic<long long>* emitted_ = nullptr;
    const std::atomic<int>* aborted_ = nullptr;
    StreamWindow win_;
    int chase_threads_ = 1, nacc_ = 1;
    std::atomic<int> chase_done_{0};
    std::atomic<long long> fb_emitted_{0};
    std::vector<std::thread> own_threads_;
    std::mutex pool_mutex_;std::condition_variable pool_work_,pool_done_;
    std::size_t pool_generation_=0;int pool_remaining_=0,pool_roles_=0;bool pool_stop_=false;
    void pool_worker(int role) noexcept;
    void join_roles();
    detail::Hb2stAccumHostPlan plan_;

    // accumulation state, carved from the reset_run buffer
    rt* qb_ = nullptr;
    std::size_t tile_elems_ = 0;                  // reals per tile
    int ntiles_ = 0;
    std::unique_ptr<Block[]> blocks_;
    std::atomic<long long> published_{0};
    std::atomic<int> planner_done_{0};
    std::atomic<int> failed_{0};
    std::mutex err_mu_;
    std::exception_ptr error_;
    int* last_ = nullptr;
    int* level_ = nullptr;
    Key* keys_ = nullptr;
    char* gen_work_ = nullptr;                    // the step generator's arrays (StepGen::bytes)
    double plan_s_ = 0.0, copy_s_ = 0.0, wait_s_ = 0.0;
    long long nblocks_ = 0;

    rt* tile(int t) { return qb_ + static_cast<std::size_t>(t) * tile_elems_; }
    long long slot_of(long long pos) const { return win_.ring > 0 ? pos % win_.ring : pos; }
    void fail(std::exception_ptr e);
    template <typename F>
    bool wait_for(F&& cond);
    void wait_records(long long end);
    void planner_body();
    void worker_body(int w);
    void build_block(StepGen& gen, Block& B);
    void apply_tile(int t, const Block& B);
    void init_tile(int t);
    void pack_columns(T* A, int lo, int cnt);
};

template<class T>Hb2stAccumHost<T>::Hb2stAccumHost(const DdlaHandle_t& h,const detail::Hb2stAccumHostPlan& p)
    :h_(h),n_(p.n),kd_(p.kd),threads_(p.threads),plan_(p)
{
    npr_=p.npr;npc_=p.npc;mypr_=p.mypr;mypc_=p.mypc;P_=p.P;myid_=p.rank;major_=p.major;row_lo_=p.row_lo;nrl_=p.nrl;
    block_recs_=p.block_records;blk_slots_=p.block_slots;rb_=p.rows_per_band;
    ntiles_=int((std::size_t(nrl_)+kW-1)/kW);tile_elems_=std::size_t(n_)*kRowReals;
    blocks_.reset(new Block[std::size_t(blk_slots_)]);own_threads_.reserve(std::size_t(std::max(2,threads_)));
    try{for(int role=0;role<std::max(2,threads_);++role)own_threads_.emplace_back([this,role]{pool_worker(role);});}
    catch(...){ {std::lock_guard<std::mutex> lock(pool_mutex_);pool_stop_=true;}pool_work_.notify_all();
        for(auto& t:own_threads_)if(t.joinable())t.join();throw;}
}
template<class T>Hb2stAccumHost<T>::~Hb2stAccumHost()
{
    failed_.store(1,std::memory_order_release);chase_done_.store(1,std::memory_order_release);
    {std::lock_guard<std::mutex> lock(pool_mutex_);pool_stop_=true;}pool_work_.notify_all();
    for(auto& t:own_threads_)if(t.joinable())t.join();
}
template<class T>void Hb2stAccumHost<T>::pool_worker(int role) noexcept
{
    std::size_t seen=0;std::unique_lock<std::mutex> lock(pool_mutex_);
    for(;;){pool_work_.wait(lock,[&]{return pool_stop_||seen!=pool_generation_;});if(pool_stop_)return;
        seen=pool_generation_;const bool active=role<pool_roles_;lock.unlock();if(active)run_role(role);lock.lock();
        if(--pool_remaining_==0)pool_done_.notify_one();}
}
template<class T>void Hb2stAccumHost<T>::reset_run(void* backend_work,std::size_t backend_bytes)
{
    join_roles();
    // this run's state, carved in the order hb2st_accum_host_workspace_size counts it
    const std::size_t rec=plan_.record_capacity,all=rec*std::size_t(blk_slots_);
    char* w=align_tiles(static_cast<char*>(backend_work));
    qb_=detail::take<rt>(w,std::size_t(ntiles_)*tile_elems_);
    int* gjlo=detail::take<int>(w,all);int* gcnt=detail::take<int>(w,all);std::int32_t* woff=detail::take<std::int32_t>(w,all);
    rt* c=detail::take<rt>(w,all);rt* s=detail::take<rt>(w,all*(kReal?1:2));
    for(int i=0;i<blk_slots_;++i){auto& b=blocks_[std::size_t(i)];const auto off=std::size_t(i)*rec;
        b.gjlo=gjlo+off;b.gcnt=gcnt+off;b.woff=woff+off;b.c=c+off;b.s=s+off*(kReal?1:2);
        b.nw=0;b.begin=b.end=0;b.done.store(0,std::memory_order_relaxed);
    }
    last_=detail::take<int>(w,std::size_t(n_)+1);level_=detail::take<int>(w,rec);keys_=detail::take<Key>(w,rec);
    gen_work_=w;w+=StepGen::bytes(plan_.slot_capacity,rec);
    assert(w-static_cast<char*>(backend_work)<=static_cast<std::ptrdiff_t>(backend_bytes));(void)backend_bytes;
    published_.store(0);planner_done_.store(0);failed_.store(0);chase_done_.store(0);fb_emitted_.store(0);
    error_=nullptr;std::fill(last_,last_+n_+1,0);plan_s_=copy_s_=wait_s_=0;nblocks_=0;
}
template<class T>void Hb2stAccumHost<T>::start_stager_run(const hb2st_rot<T>* rots,long long cap,
    const std::atomic<long long>* emitted,const StreamWindow* win,const std::atomic<int>* aborted,int chase_threads,bool roles_in_team)
{
    rots_=rots;cap_=cap;emitted_=emitted;aborted_=aborted;win_=win?*win:StreamWindow();
    if(win_.switch_pos)throw std::logic_error("Hb2stAccumHost: an elastic ring window is not supported");
    chase_threads_=std::max(0,chase_threads);nacc_=std::max(1,threads_-chase_threads_-1);
    if(!roles_in_team){std::lock_guard<std::mutex> lock(pool_mutex_);assert(!pool_remaining_);
        pool_roles_=extra_roles();pool_remaining_=int(own_threads_.size());++pool_generation_;pool_work_.notify_all();}
}

template <typename T>
void Hb2stAccumHost<T>::fail(std::exception_ptr e)
{
    {
        std::lock_guard<std::mutex> lk(err_mu_);
        if (!error_) error_ = e;
    }
    failed_.store(1, std::memory_order_release);
}

template <typename T>
template <typename F>
bool Hb2stAccumHost<T>::wait_for(F&& cond)
{
    int spins = 0;
    while (!cond()) {
        if (failed_.load(std::memory_order_acquire) != 0) return false;
        if (++spins < 512) cpu_relax();
        else std::this_thread::yield();
    }
    return true;
}

template <typename T>
void Hb2stAccumHost<T>::wait_records(long long end)
{
    int spins = 0;
    for (;;) {
        if (emitted_->load(std::memory_order_acquire) >= end) return;
        if (failed_.load(std::memory_order_acquire) != 0) throw std::runtime_error("Hb2stAccumHost: aborted");
        if (aborted_ != nullptr && aborted_->load(std::memory_order_acquire) != 0)
            throw std::runtime_error("Hb2stAccumHost: the chase aborted");
        if (chase_done_.load(std::memory_order_acquire) != 0 && emitted_->load(std::memory_order_acquire) < end)
            throw std::runtime_error("Hb2stAccumHost: the chase ended before the scheduled rotation stream (" +
                                     std::to_string(emitted_->load()) + " of at least " + std::to_string(end) + " records)");
        if (++spins < 512) cpu_relax();
        else std::this_thread::yield();
    }
}

template <typename T>
void Hb2stAccumHost<T>::run_role(int role)
{
    try {
        if (role == 0) planner_body();
        else worker_body(role - 1);
    } catch (...) {
        fail(std::current_exception());
    }
    if (role == 0) {
        // Only the planner reads rots_. Publish retirement after its body has
        // returned or unwound, including failure in StepGen/build_block. A
        // worker failure merely requests that exit: retiring in fail() would
        // let the producer overwrite records the planner is still copying.
        if (win_.consumed != nullptr)
            win_.consumed->store(std::numeric_limits<long long>::max(), std::memory_order_release);
        planner_done_.store(1, std::memory_order_release);
    }
}

template <typename T>
void Hb2stAccumHost<T>::init_tile(int t)
{
    rt* X = tile(t);
    std::memset(X, 0, tile_elems_ * sizeof(rt));
    for (int k = 0; k < kW; ++k) {
        const int i = t * kW + k;
        if (i >= nrl_) break;
        X[static_cast<std::size_t>(row_lo_ + i) * kRowReals + k] = rt(1);   // Q_band(row_lo+i, col row_lo+i) = 1
    }
}

template <typename T>
void Hb2stAccumHost<T>::apply_tile(int t, const Block& B)
{
    rt* X = tile(t);
    if constexpr (kReal)
        apply_block_real<rt, kW>(X, B.nw, B.gjlo, B.gcnt, B.woff, B.c, B.s);
    else
        apply_block_cplx<rt, kW>(X, B.nw, B.gjlo, B.gcnt, B.woff, B.c, B.s);
}

template <typename T>
void Hb2stAccumHost<T>::worker_body(int w)
{
    const int nth = nacc_;
    for (int t = w; t < ntiles_; t += nth) init_tile(t);
    for (long long b = 0;; ++b) {
        if (!wait_for([&] { return published_.load(std::memory_order_acquire) > b || planner_done_.load(std::memory_order_acquire) != 0; }))
            return;
        if (b >= published_.load(std::memory_order_acquire)) return;   // the planner is done and this block does not exist
        Block& B = blocks_[static_cast<std::size_t>(b % blk_slots_)];
        for (int t = w; t < ntiles_; t += nth) apply_tile(t, B);
        B.done.fetch_add(1, std::memory_order_acq_rel);
    }
}

// Levels, skewed order and the records of one block.  The block's dominoes are in `gen` (creation order = a topological order of the
// overlaps); the records must be in the window (the caller waited).
template <typename T>
void Hb2stAccumHost<T>::build_block(StepGen& gen, Block& B)
{
    const int nw0 = gen.nw;
    const int K = gen.K;
    B.begin = gen.begin;
    B.end = gen.nrots;
    if (nrl_ == 0 || nw0 == 0) {   // nothing to apply on this rank: the records are only consumed
        wait_records(gen.nrots);
        B.nw = 0;
        return;
    }
    // dependency levels inside the block
    int maxlevel = 0, maxcnt = 0;
    for (int w = 0; w < nw0; ++w) {
        const int top = gen.gj[static_cast<std::size_t>(w)], lo = top - gen.gc[static_cast<std::size_t>(w)];
        int lv = 0;
        for (int r = lo; r <= top; ++r) lv = std::max(lv, last_[static_cast<std::size_t>(r)]);
        ++lv;
        for (int r = lo; r <= top; ++r) last_[static_cast<std::size_t>(r)] = lv;
        level_[static_cast<std::size_t>(w)] = lv;
        maxlevel = std::max(maxlevel, lv);
        maxcnt = std::max(maxcnt, gen.gc[static_cast<std::size_t>(w)]);
    }
    for (int w = 0; w < nw0; ++w) {   // reset only what was touched
        const int top = gen.gj[static_cast<std::size_t>(w)], lo = top - gen.gc[static_cast<std::size_t>(w)];
        for (int r = lo; r <= top; ++r) last_[static_cast<std::size_t>(r)] = 0;
    }
    // skewed bands: u = top - skew*level (+ offset >= 0); a dependency X -> Z (Z later, overlapping rows) has level(Z) >= level(X)+1
    // and |top(Z) - top(X)| <= maxcnt < skew, so u(Z) < u(X): with bands DESCENDING and levels ascending inside a band the order is valid.
    const long long skew = maxcnt + 1;
    const long long off = skew * maxlevel;
    const long long nbands = (off + n_) / rb_ + 2;
    for (int w = 0; w < nw0; ++w) {
        const long long u = gen.gj[static_cast<std::size_t>(w)] - skew * level_[static_cast<std::size_t>(w)] + off;
        keys_[static_cast<std::size_t>(w)] = Key{nbands - 1 - u / rb_, level_[static_cast<std::size_t>(w)], gen.gj[static_cast<std::size_t>(w)], w};
    }
    std::sort(keys_, keys_ + nw0, [](const Key& a, const Key& b) {
        if (a.bandrev != b.bandrev) return a.bandrev < b.bandrev;
        if (a.level != b.level) return a.level < b.level;
        return a.top < b.top;
    });
    B.nw = nw0;
    wait_records(gen.nrots);   // the copy below reads the window
    const double t0 = wall();
    std::int32_t acc = 0;
    for (int o = 0; o < nw0; ++o) {
        const int w = keys_[static_cast<std::size_t>(o)].w;
        const int top = gen.gj[static_cast<std::size_t>(w)], cnt = gen.gc[static_cast<std::size_t>(w)];
        B.gjlo[static_cast<std::size_t>(o)] = top;
        B.gcnt[static_cast<std::size_t>(o)] = cnt;
        B.woff[static_cast<std::size_t>(o)] = acc;
        auto member = gen.first[w];
        for (int p = 0; p < cnt; ++p) {
            const long long pos = gen.begin + member; member = gen.next[member];
            const long long slot = slot_of(pos);
            if (slot < 0 || slot >= cap_) throw std::runtime_error("Hb2stAccumHost: a record position left the chase window");
            const hb2st_rot<T>& r = rots_[slot];
            if (r.j0 != top - p)
                throw std::runtime_error("Hb2stAccumHost: the rotation stream diverged from the chase's recording arithmetic (record " +
                                         std::to_string(pos) + ": j0 " + std::to_string(r.j0) + ", expected " + std::to_string(top - p) + ")");
            B.c[static_cast<std::size_t>(acc + p)] = r.c;
            if constexpr (kReal) {
                B.s[static_cast<std::size_t>(acc + p)] = r.s;
            } else {
                B.s[2 * static_cast<std::size_t>(acc + p)] = r.s.real();
                B.s[2 * static_cast<std::size_t>(acc + p) + 1] = r.s.imag();
            }
        }
        acc += cnt;
    }
    copy_s_ += wall() - t0;
}

template <typename T>
void Hb2stAccumHost<T>::planner_body()
{
    const int K = std::max(1, std::min(kd_, 64));
    StepGen gen(n_, kd_, K, plan_.slot_capacity, plan_.record_capacity, gen_work_);
    const int R = blk_slots_;
    long long b = 0;
    for (;; ++b) {
        Block& B = blocks_[static_cast<std::size_t>(b % R)];
        const double t0 = wall();
        gen.start_block();
        while (gen.block_records() < block_recs_ && gen.step()) {}
        if (gen.block_records() == 0) break;   // (a step can only add records, so this is the end of the stream)
        // the slot is free once every worker is through with the block that used it R blocks ago
        if (b >= R) {
            const double tw = wall();
            if (!wait_for([&] { return B.done.load(std::memory_order_acquire) >= nacc_; })) return;
            wait_s_ += wall() - tw;
        }
        plan_s_ += wall() - t0;
        build_block(gen, B);   // waits for the records, copies them
        B.done.store(0, std::memory_order_relaxed);
        if (win_.consumed != nullptr) win_.consumed->store(B.end, std::memory_order_release);   // the records below B.end are dead
        published_.store(b + 1, std::memory_order_release);
        ++nblocks_;
    }
}

// Waits for the sink's own threads (roles_in_team == false) to finish their roles; a no-op otherwise.
template<class T>void Hb2stAccumHost<T>::join_roles()
{
    std::unique_lock<std::mutex> lock(pool_mutex_);pool_done_.wait(lock,[&]{return pool_remaining_==0;});
}

// A(i, c') = Q_band(row_lo + i, lo + c'), i < nrl, c' < cnt (column-major, ld nrl): the GEMM's left operand for one slab
template <typename T>
void Hb2stAccumHost<T>::pack_columns(T* A, int lo, int cnt)
{
    detail::host_parallel_for(cnt, 8, threads_, [&](long long a, long long b) {
        for (long long cc = a; cc < b; ++cc) {
            const std::size_t col = static_cast<std::size_t>(lo) + static_cast<std::size_t>(cc);
            T* dst = A + static_cast<std::size_t>(cc) * nrl_;
            for (int t = 0; t < ntiles_; ++t) {
                const rt* row = tile(t) + col * kRowReals;
                const int lanes = std::min(kW, nrl_ - t * kW);
                if constexpr (kReal) {
                    for (int k = 0; k < lanes; ++k) dst[t * kW + k] = row[k];
                } else {
                    for (int k = 0; k < lanes; ++k) dst[t * kW + k] = T(row[k], row[kW + k]);
                }
            }
        }
    });
}

template<class T>void Hb2stAccumHost<T>::finish_rows_run(const detail::Hb2stFinishHostPlan& p,const rt* Y,const T* phases,T* Z,
    int& info,void* backend_work,std::size_t backend_bytes)
{
    info=0;drain_stage2_run();const double start=wall();
    detail::hb2st_finish_host_run<T>(h_,p,Y,phases,Z,this,
        [](void* self,T* a,int lo,int count){static_cast<Hb2stAccumHost*>(self)->pack_columns(a,lo,count);},backend_work,backend_bytes);
    if (plan_.profile)
        std::printf("HOSTSINK myid=%d nacc=%d threads=%d blocks=%lld tiles=%d | stage2: plan %.0f ms copy %.0f ms slot-wait %.0f ms | finish: %.0f ms\n",
                    myid_, nacc_, threads_, nblocks_, ntiles_, plan_s_ * 1e3, copy_s_ * 1e3, wait_s_ * 1e3, (wall() - start) * 1e3);
}

// Elements of the finish panels, in carving order: the product / receive panel W, the send panel, the phase-scaled Y panel,
// the packed Q_band columns A and the broadcast Y panel.
template<class T>std::array<std::size_t,5> finish_host_counts(const detail::Hb2stFinishHostPlan& p)
{
    using R=typename real_type<T>::type;const int* d=p.descriptor.data();std::array<std::size_t,5> c{};
    const std::size_t product=std::size_t(p.nrl)*std::size_t(p.panel);
    const std::size_t receive=std::size_t(num_loc(p.n,d[DDLA_MB_],p.mypr,d[DDLA_RSRC_],p.npr))*
        detail::hb2st_finish_max_owned(p.panel,d[DDLA_NB_],p.npc);
    if(p.P>1){c[0]=std::max({product,receive,std::size_t(1)});c[1]=std::max(product,std::size_t(1));
        c[4]=std::size_t(p.kp)*std::size_t(p.panel);}
    if constexpr(!std::is_same<T,R>::value)if(p.nrl)c[2]=std::size_t(p.kp)*std::size_t(p.panel);
    c[3]=std::size_t(p.nrl)*std::size_t(p.kp);return c;
}

} // namespace

namespace detail {

Hb2stAccumHostPlan hb2st_accum_host_check(const DdlaHandle_t& h,int n,int kd,int threads)
{
    auto p=agree_check(h,"Hb2stAccumHost",[&](Agree& agreement){agreement.mode("spectral operation",3142);
        if(!h||ddlaGetBackend(h)!=DdlaBackend::CPU||n<3||kd<=1||kd>=std::numeric_limits<int>::max()||threads<1)
            throw std::invalid_argument("Hb2stAccumHost: invalid handle, dimensions or worker budget");
        Hb2stAccumHostPlan r;r.n=n;r.kd=kd;r.threads=threads;r.npr=h->nprows_;r.npc=h->npcols_;r.mypr=h->myprow_;r.mypc=h->mypcol_;
        r.rank=h->myid;r.P=r.npr*r.npc;r.major=h->major;const RowShare rows(n,r.P);r.row_lo=rows.lo(r.rank);r.nrl=rows.count(r.rank);
        r.block_records=env_ll("DDLA_EIG_HOST_ACCUM_BLOCK",524288,4096);
        const auto slots=env_ll("DDLA_EIG_HOST_ACCUM_SLOTS",3,2),rb=env_ll("DDLA_EIG_HOST_ACCUM_RB",512,8);
        if(slots>std::numeric_limits<int>::max()||rb>std::numeric_limits<int>::max()||r.block_records>std::numeric_limits<std::int32_t>::max())
            throw std::overflow_error("Hb2stAccumHost: tuning parameter exceeds supported count");
        r.block_slots=int(slots);r.rows_per_band=int(rb);r.total=Hb2stWaveGen::count_records(n,kd);r.profile=eig_profile_enabled();
        // the largest block the planner cuts (whole chase steps until block_records) and the chase's slot count
        ChaseCursor cursor(n,kd);std::size_t block=0;
        while(!cursor.done()){block+=std::size_t(cursor.count());cursor.next();
            if(block>=std::size_t(r.block_records)||cursor.done()){r.record_capacity=std::max(r.record_capacity,block);block=0;}}
        r.slot_capacity=std::size_t(n)/(std::size_t(kd)+1)+2;
        if(r.record_capacity>std::size_t(std::numeric_limits<std::int32_t>::max()))
            throw std::overflow_error("Hb2stAccumHost: block member offsets exceed int32");
        return r;
    });
    const int signature[]={3142,n,kd,p.npr,p.npc,int(p.major),int(p.block_records),p.block_slots,p.rows_per_band,p.profile?1:0};
    agree_signature(h,"Hb2stAccumHost policy",signature);return p;
}
template<class T>void hb2st_accum_host_qband_workspace_size(const DdlaHandle_t&,const Hb2stAccumHostPlan& p,std::size_t* backend_bytes)
{
    using R=typename real_type<T>::type;constexpr std::size_t lanes=256/sizeof(T);
    const std::size_t tiles=(std::size_t(p.nrl)+lanes-1)/lanes;
    *backend_bytes=kTileAlignment+padded_bytes<R>(tiles*std::size_t(p.n)*(std::is_same<T,R>::value?lanes:2*lanes));
}
template<class T>void hb2st_accum_host_workspace_size(const DdlaHandle_t& h,const Hb2stAccumHostPlan& p,std::size_t* backend_bytes)
{
    using R=typename real_type<T>::type;const std::size_t rec=p.record_capacity,all=rec*std::size_t(p.block_slots);
    std::size_t bytes=0;hb2st_accum_host_qband_workspace_size<T>(h,p,&bytes);
    bytes+=3*padded_bytes<int>(all)+padded_bytes<R>(all)+padded_bytes<R>(all*(std::is_same<T,R>::value?1:2));
    bytes+=padded_bytes<int>(std::size_t(p.n)+1)+padded_bytes<int>(rec)+padded_bytes<Key>(rec);
    *backend_bytes=bytes+StepGen::bytes(p.slot_capacity,rec);
}
template<class T>std::unique_ptr<Hb2stAccumHostResources<T>> hb2st_accum_host_prepare(const DdlaHandle_t& h,const Hb2stAccumHostPlan& p)
{return std::unique_ptr<Hb2stAccumHostResources<T>>(new Hb2stAccumHost<T>(h,p));}
#define DDLA_HOST_ACCUM_INSTANTIATE(T) \
template void hb2st_accum_host_qband_workspace_size<T>(const DdlaHandle_t&,const Hb2stAccumHostPlan&,std::size_t*); \
template void hb2st_accum_host_workspace_size<T>(const DdlaHandle_t&,const Hb2stAccumHostPlan&,std::size_t*); \
template std::unique_ptr<Hb2stAccumHostResources<T>> hb2st_accum_host_prepare<T>(const DdlaHandle_t&,const Hb2stAccumHostPlan&)
DDLA_FOR_EACH_TYPE(DDLA_HOST_ACCUM_INSTANTIATE)
#undef DDLA_HOST_ACCUM_INSTANTIATE

template<class T>Hb2stFinishHostPlan hb2st_finish_host_geometry(const DdlaHandle_t& h,int n,int ldy,
    const int* desc,int threads,int panel_columns,int contraction_columns,bool phases)
{
    Hb2stFinishHostPlan p;p.n=n;p.ldy=ldy;p.npr=h->nprows_;p.npc=h->npcols_;p.mypr=h->myprow_;p.mypc=h->mypcol_;
    p.rank=h->myid;p.P=p.npr*p.npc;p.major=h->major;p.threads=std::max(1,threads);p.phases=phases;
    std::copy_n(desc,DDLA_DLEN_,p.descriptor.data());const RowShare rows(n,p.P);p.nrl=rows.count(p.rank);p.maxcnt=rows.count(0);
    const auto kcap=std::max(std::size_t(1),(std::size_t(64)<<20)/(std::size_t(std::max(p.maxcnt,1))*sizeof(T)));
    p.kp=static_cast<int>(std::min(std::size_t(std::max(1,contraction_columns)),std::min(std::size_t(std::max(p.maxcnt,1)),kcap)));
    p.panel=panel_columns>0?std::min(n,panel_columns):hb2st_finish_panel_width(n,p.npr,p.npc,desc[DDLA_MB_],desc[DDLA_NB_],
        sizeof(T),sizeof(typename real_type<T>::type),true,p.kp);return p;
}
template<class T>Hb2stFinishHostPlan hb2st_finish_host_check(const DdlaHandle_t& h,int n,
    const typename real_type<T>::type* Y,int ldy,const T* phases,std::size_t phase_count,T* Z,const int* desc,int threads,int panel_columns)
{
    using R=typename real_type<T>::type;
    auto p=agree_check(h,"hb2st CPU finish",[&](Agree& agreement){
        agreement.mode("spectral operation",3141);if(!h||ddlaGetBackend(h)!=DdlaBackend::CPU)throw std::invalid_argument("hb2st CPU finish: invalid backend");
        check_desc(desc,h);if(n<0||n>desc[DDLA_M_]||n>desc[DDLA_N_]||panel_columns<0||threads<1)
            throw std::invalid_argument("hb2st CPU finish: invalid dimensions or policy");
        const auto contraction=env_ll("DDLA_EIG_HOST_FINISH_PANEL",512,16);
        if(contraction>std::numeric_limits<int>::max())throw std::overflow_error("hb2st CPU finish: contraction panel exceeds int");
        auto plan=hb2st_finish_host_geometry<T>(h,n,ldy,desc,threads,panel_columns,static_cast<int>(contraction),phase_count>0);
        if(ldy!=std::max(plan.nrl,1)||(n&&plan.nrl&&!Y))throw std::invalid_argument("hb2st CPU finish: invalid compact Y slab");
        const int lr=num_loc(n,desc[DDLA_MB_],plan.mypr,desc[DDLA_RSRC_],plan.npr);
        const int lc=num_loc(n,desc[DDLA_NB_],plan.mypc,desc[DDLA_CSRC_],plan.npc);
        if(lr&&lc&&!Z)throw std::invalid_argument("hb2st CPU finish: missing output");
        if constexpr(!std::is_same<T,R>::value)if(phase_count&&(!phases||phase_count<std::size_t(std::max(0,n-1))))
            throw std::invalid_argument("hb2st CPU finish: missing phases");
        return plan;
    });
    const int signature[]={3141,n,p.npr,p.npc,int(p.major),p.kp,p.panel,p.phases?1:0,
        p.descriptor[DDLA_M_],p.descriptor[DDLA_N_],p.descriptor[DDLA_MB_],p.descriptor[DDLA_NB_],p.descriptor[DDLA_RSRC_],p.descriptor[DDLA_CSRC_]};
    agree_signature(h,"hb2st CPU finish",signature);
    const int lr=num_loc(n,p.descriptor[DDLA_MB_],p.mypr,p.descriptor[DDLA_RSRC_],p.npr);
    const int lc=num_loc(n,p.descriptor[DDLA_NB_],p.mypc,p.descriptor[DDLA_CSRC_],p.npc);
    const std::size_t zb=lc&&lr?(std::size_t(lc-1)*std::size_t(p.descriptor[DDLA_LLD_])+std::size_t(lr))*sizeof(T):0;
    const std::size_t yb=p.nrl?(std::size_t(std::max(0,n-1))*std::size_t(ldy)+std::size_t(p.nrl))*sizeof(R):0;
    int aliases=memory_overlaps(Z,zb,Y,yb);MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,&aliases,1,MPI_INT,MPI_MAX,h->comm));
    if(aliases)throw std::invalid_argument("hb2st CPU finish: output overlaps immutable Y");return p;
}
template<class T>void hb2st_finish_host_workspace_size(const DdlaHandle_t& h,const Hb2stFinishHostPlan& p,std::size_t* backend_bytes)
{
    using R=typename real_type<T>::type;*backend_bytes=0;if(!p.n)return;
    const auto c=finish_host_counts<T>(p);
    *backend_bytes=hb2st_finish_panel_map_bytes(p.n,p.npr,p.npc,p.mypr,p.rank,p.descriptor.data(),p.panel)+kTileAlignment+
        padded_bytes<T>(c[0])+padded_bytes<T>(c[1])+padded_bytes<T>(c[2])+padded_bytes<T>(c[3])+padded_bytes<R>(c[4]);
    if(p.input_2d.enabled&&p.P>1)*backend_bytes+=padded_bytes<R>(std::size_t(p.kp)*p.panel)+padded_bytes<matrix_impl::Block<R>>(h->nprocs-1);
}
template<class T>void hb2st_finish_host_run(const DdlaHandle_t& h,const Hb2stFinishHostPlan& p,
    const typename real_type<T>::type* Y,const T* phases,T* Z,void* context,void (*pack_columns)(void*,T*,int,int),
    void* backend_work,std::size_t backend_bytes)
{
    using rt=typename real_type<T>::type;constexpr bool real=std::is_same<T,rt>::value;if(!p.n)return;
    const int n=p.n,npr=p.npr,npc=p.npc,mypr=p.mypr,mypc=p.mypc,P=p.P,rank=p.rank,lld=p.descriptor[DDLA_LLD_];
    const int nrl=p.nrl,KP=p.kp,pw=p.panel,threads=p.threads,ldy=p.ldy;const bool direct=P==1;const RowShare rows(n,P);
    const bool scaled=p.phases&&phases;
    // carved in the order hb2st_finish_host_workspace_size counts it
    const auto counts=finish_host_counts<T>(p);char* w=static_cast<char*>(backend_work);
    auto map=hb2st_finish_panel_map(n,npr,npc,mypr,mypc,rank,p.major,p.descriptor.data(),pw,w);w=align_tiles(w);
    T* const work=take<T>(w,counts[0]);T* const send=take<T>(w,counts[1]);T* const py=take<T>(w,counts[2]);
    T* const A=take<T>(w,counts[3]);rt* const panel=take<rt>(w,counts[4]);
    void* gather_work=nullptr;void* gather_host=nullptr;std::size_t gather_bytes=0,gather_host_bytes=0;
    if(p.input_2d.enabled&&P>1){gather_bytes=padded_bytes<rt>(std::size_t(KP)*pw);gather_work=w;w+=gather_bytes;
        gather_host_bytes=padded_bytes<matrix_impl::Block<rt>>(P-1);gather_host=w;w+=gather_host_bytes;}
    assert(w-static_cast<char*>(backend_work)<=static_cast<std::ptrdiff_t>(backend_bytes));(void)backend_bytes;
    const MPI_Datatype dt_rt = mpi_datatype<rt>(), dt = mpi_datatype<T>();
    for (int j0 = 0; j0 < n; j0 += pw) {
        const int width = std::min(pw, n - j0);
        T* W = direct ? Z + static_cast<std::size_t>(j0) * lld : work;
        const int ldw = direct ? lld : std::max(nrl, 1);
        bool first = true;
        for (int r = 0; r < P; ++r) {
            const int count = rows.count(r);
            for (int k0 = 0; k0 < count; k0 += KP) {
                const int kp = std::min(KP, count - k0);
                const rt* src = nullptr;
                int lds = 0;
                if (P > 1) {
                    if(p.input_2d.enabled) {
                        matrix_impl::gather_run<DdlaBackend::CPU,rt>(h,kp,width,Y,rows.lo(r)+k0+1,j0+1,
                            p.input_2d.descriptor.data(),r,panel,kp,gather_work,gather_bytes,gather_host,gather_host_bytes);
                    } else if (r == rank) {
                        host_parallel_for(width, 8, threads, [&](long long a, long long b) {
                            for (long long j = a; j < b; ++j)
                                std::memcpy(panel + static_cast<std::size_t>(j) * kp,
                                            Y + static_cast<std::size_t>(j0 + j) * ldy + k0,
                                            static_cast<std::size_t>(kp) * sizeof(rt));
                        });
                    }
                    const std::size_t elements = static_cast<std::size_t>(kp) * width;
                    for (std::size_t off = 0; off < elements; off += (std::size_t(1) << 28))
                        if (MPI_Bcast(panel + off, static_cast<int>(std::min(elements - off, std::size_t(1) << 28)),
                                      dt_rt, r, h->comm) != MPI_SUCCESS)
                            throw std::runtime_error("hb2st finish: MPI_Bcast failed");
                    src = panel; lds = kp;
                } else {
                    lds=p.input_2d.enabled?p.input_2d.descriptor[DDLA_LLD_]:ldy;
                    src = Y + static_cast<std::size_t>(j0) * lds + k0;
                }
                if (nrl == 0) continue;
                const T* B = nullptr;
                int ldb = 0;
                if constexpr (real) { B = reinterpret_cast<const T*>(src); ldb = lds; }
                else {
                    const int row0 = rows.lo(r) + k0;
                    host_parallel_for(width, 16, threads, [&](long long a, long long b) {
                        for (long long j = a; j < b; ++j)
                            for (int i = 0; i < kp; ++i) {
                                const rt y = src[static_cast<std::size_t>(j) * lds + i];
                                py[static_cast<std::size_t>(j) * kp + i] =
                                    row0 + i >= 1 && scaled ?
                                        T(y * phases[static_cast<std::size_t>(row0 + i - 1)].real(),
                                          y * phases[static_cast<std::size_t>(row0 + i - 1)].imag()) : T(y, rt(0));
                            }
                    });
                    B = py; ldb = kp;
                }
                pack_columns(context, A, rows.lo(r) + k0, kp);
                blas_impl::gemm_run<DdlaBackend::CPU,T>(h, 'N', 'N', nrl, width, kp, T(1), A, nrl,
                                         B, ldb, first ? T(0) : T(1), W, ldw);
                first = false;
            }
        }
        if (direct) continue;
        map.columns(j0, width);
        if (nrl) {
            host_parallel_for(width, 8, threads, [&](long long a, long long b) {
                for (long long j = a; j < b; ++j) {
                    const int pc = map.col_owner(j0 + static_cast<int>(j));
                    for (int r = 0; r < nrl; ++r) {
                        const int pr = map.row_owner(rows.lo(rank) + r), q = map.rank_of(pr, pc);
                        send[static_cast<std::size_t>(map.sd[q]) + static_cast<std::size_t>(map.cs[j]) * map.nr[pr] + map.rslot[r]] =
                            W[static_cast<std::size_t>(j) * nrl + r];
                    }
                }
            });
        }
        if (MPI_Alltoallv(send, map.sc, map.sd, dt,
                          work, map.rc, map.rd, dt, h->comm) != MPI_SUCCESS)
            throw std::runtime_error("hb2st finish: MPI_Alltoallv failed");
        host_parallel_for(map.nlc, 4, threads, [&](long long a, long long b) {
            for (long long m = a; m < b; ++m) {
                T* dst = Z + static_cast<std::size_t>(map.lj[m]) * lld;
                for (int q = 0; q < P; ++q) {
                    const int count = map.rc_rows[q];
                    const T* src = work + static_cast<std::size_t>(map.rd[q]) + static_cast<std::size_t>(m) * count;
                    for (int k = 0; k < count; ++k) dst[map.li[map.rro[q] + k]] = src[k];
                }
            }
        });
    }
}

template <typename T>
void hb2st_finish_rows_host(const DdlaHandle_t& h,int n,const typename real_type<T>::type* Y,int ldy,
    const std::vector<T>& phases,T* Z,const int* desc,const std::function<void(T*,int,int)>& pack,int threads,int panel_columns)
{
    const auto p=hb2st_finish_host_check<T>(h,n,Y,ldy,phases.data(),phases.size(),Z,desc,threads,panel_columns);
    std::size_t backend_bytes=0;vector<char> backend_work(h);std::exception_ptr error;
    try{hb2st_finish_host_workspace_size<T>(h,p,&backend_bytes);backend_work.resize(backend_bytes);}catch(...){error=std::current_exception();}
    detail::agree_failure(h->comm,error,"hb2st CPU finish: preparation");
    hb2st_finish_host_run<T>(h,p,Y,phases.data(),Z,const_cast<void*>(static_cast<const void*>(&pack)),
        [](void* c,T* a,int lo,int count){(*static_cast<const std::function<void(T*,int,int)>*>(c))(a,lo,count);},
        backend_work.data(),backend_bytes);
}
#define DDLA_HOST_FINISH_INSTANTIATE(T,R) \
template Hb2stFinishHostPlan hb2st_finish_host_geometry<T>(const DdlaHandle_t&,int,int,const int*,int,int,int,bool); \
template Hb2stFinishHostPlan hb2st_finish_host_check<T>(const DdlaHandle_t&,int,const R*,int,const T*,std::size_t,T*,const int*,int,int); \
template void hb2st_finish_host_workspace_size<T>(const DdlaHandle_t&,const Hb2stFinishHostPlan&,std::size_t*); \
template void hb2st_finish_host_run<T>(const DdlaHandle_t&,const Hb2stFinishHostPlan&,const R*,const T*,T*,void*,void (*)(void*,T*,int,int),void*,std::size_t);
DDLA_HOST_FINISH_INSTANTIATE(float,float)
DDLA_HOST_FINISH_INSTANTIATE(double,double)
DDLA_HOST_FINISH_INSTANTIATE(std::complex<float>,float)
DDLA_HOST_FINISH_INSTANTIATE(std::complex<double>,double)
#undef DDLA_HOST_FINISH_INSTANTIATE

template void hb2st_finish_rows_host<float>(const DdlaHandle_t&, int, const float*, int, const std::vector<float>&, float*, const int*, const std::function<void(float*, int, int)>&, int, int);
template void hb2st_finish_rows_host<double>(const DdlaHandle_t&, int, const double*, int, const std::vector<double>&, double*, const int*, const std::function<void(double*, int, int)>&, int, int);
template void hb2st_finish_rows_host<std::complex<float>>(const DdlaHandle_t&, int, const float*, int, const std::vector<std::complex<float>>&, std::complex<float>*, const int*, const std::function<void(std::complex<float>*, int, int)>&, int, int);
template void hb2st_finish_rows_host<std::complex<double>>(const DdlaHandle_t&, int, const double*, int, const std::vector<std::complex<double>>&, std::complex<double>*, const int*, const std::function<void(std::complex<double>*, int, int)>&, int, int);

} // namespace detail


} // namespace ddla
