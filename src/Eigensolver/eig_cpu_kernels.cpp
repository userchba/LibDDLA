// Host-only OpenMP kernels of the CPU eigensolver path.  See eig_cpu_kernels.h.
//
// Numerics: this file is compiled with -ffp-contract=off (src/CMakeLists.txt) and the kernels spell the
// arithmetic exactly as the original serial loops did (every product rounded, then the add/sub), so
// the result is bit-for-bit the same on every CPU, at every tile width and thread count -- the AVX2 and
// AVX-512 function clones below only change how many columns one instruction handles, never the value.

#include "helpers.h"
#include "eig_cpu_kernels.h"
#include "ddla_connector.h"
#include "Eigensolver/two_stage/distributed_history.h"
#include "Eigensolver/two_stage/hb2st_cpu_replay_workspace.h"
#include "Eigensolver/two_stage/hb2st_cpu_replay_workspace.h"
#include "ddla_desc.h"
#include "eig_params.h"
#include "ddla_stream.h"
#include "Eigensolver/two_stage/eig_cpu_column_layout.h"
#include "row_share.h"
#include "mpi_datatype.h"
#include "plan_util.h"
#include <limits>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <new>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "host_threads.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
// Runtime-dispatched clones: the library is built without -march, so SSE2 would be the ceiling.
#define DDLA_CPU_CLONES __attribute__((target_clones("avx512f", "avx2", "default")))
#else
#define DDLA_CPU_CLONES
#endif

namespace ddla {
namespace detail {

namespace {

template <typename T>
constexpr bool k_real = std::is_same<T, typename real_type<T>::type>::value;

// ---- tile kernels --------------------------------------------------------------------------------
// X holds n rows of one W-wide column tile.  Real: row-major, row stride W.  Complex: split planes per
// row, [re(W) | im(W)], row stride 2W (so the loops are plain real vector code, no shuffles).

template <int W, typename rt>
DDLA_CPU_CLONES void rot_tile_real(rt* __restrict X, const hb2st_rot<rt>* rots, std::size_t nrots, int n)
{
    for (std::size_t q = nrots; q-- > 0;) {
        const hb2st_rot<rt>& r = rots[q];
        const int a = r.j0 - 1, b = r.j0;
        if (a < 0 || b >= n) continue;
        const rt c = r.c, s = r.s;
        rt* ra = X + static_cast<std::size_t>(a) * W;
        rt* rb = X + static_cast<std::size_t>(b) * W;
#pragma GCC ivdep
        for (int k = 0; k < W; ++k) {
            const rt xa = ra[k], xb = rb[k];
            ra[k] = c * xa - s * xb;
            rb[k] = s * xa + c * xb;
        }
    }
}

template <int W, typename rt>
DDLA_CPU_CLONES void rot_tile_cplx(rt* __restrict X, const hb2st_rot<std::complex<rt>>* rots, std::size_t nrots,
                                   int n)
{
    for (std::size_t q = nrots; q-- > 0;) {
        const hb2st_rot<std::complex<rt>>& r = rots[q];
        const int a = r.j0 - 1, b = r.j0;
        if (a < 0 || b >= n) continue;
        const rt c = r.c, sr = r.s.real(), si = r.s.imag();
        rt* ar = X + static_cast<std::size_t>(a) * (2 * W);
        rt* ai = ar + W;
        rt* br = X + static_cast<std::size_t>(b) * (2 * W);
        rt* bi = br + W;
#pragma GCC ivdep
        for (int k = 0; k < W; ++k) {
            const rt xar = ar[k], xai = ai[k], xbr = br[k], xbi = bi[k];
            // ra = c*xa - conj(s)*xb   (conj(s)*xb written out: the same products and the same
            // rounding as the serial std::complex expression)
            const rt t2r = sr * xbr + si * xbi;
            const rt t2i = sr * xbi - si * xbr;
            ar[k] = c * xar - t2r;
            ai[k] = c * xai - t2i;
            // rb = s*xa + c*xb
            const rt u1r = sr * xar - si * xai;
            const rt u1i = sr * xai + si * xar;
            br[k] = u1r + c * xbr;
            bi[k] = u1i + c * xbi;
        }
    }
}

template <typename T, int W>
void rot_tile(typename real_type<T>::type* X, const hb2st_rot<T>* rots, std::size_t nrots, int n)
{
    using rt = typename real_type<T>::type;
    if constexpr (k_real<T>) rot_tile_real<W, rt>(X, rots, nrots, n);
    else rot_tile_cplx<W, rt>(X, rots, nrots, n);
}

// ---- wave-ordered replay ------------------------------------------------------------------------
// The chase records its rotations step by step; one bulge's rotations in consecutive steps form a DOMINO: the
// same slot, j0 decreasing by exactly 1 (one row pair slides down the band).  Replaying a domino as a chain
// keeps the carried row in registers, and -- measured on the real stream, W=32, Xeon Gold 6138 -- the plain
// step-by-step order runs at ~1.1 G col-rot/s per thread while the wave order runs at 1.7-1.9 (bit-identical
// results: every row sees the same rotations in the same order, rotations of different slots commute).
// The schedule is rebuilt from (n, kd) alone by replaying the chase's integer record arithmetic (the same
// loops as hb2st_wave_schedule.h); kd is the first record's j0 (min(n-1, kd) for the very first step).  Every
// record's j0 is checked against that replay, so a stream that is not a chase stream falls back to the plain
// kernel instead of being reordered wrongly.

std::atomic<bool> g_last_replay_waves{false};

// A read-only view of a local or node-shared wave schedule.
template <typename T>
struct WaveView {
    const int* gjlo = nullptr;
    const std::int64_t* woff = nullptr;
    const int* gcnt = nullptr;
    const hb2st_rot<T>* wrec = nullptr;
    std::size_t nw = 0;
};

// The sort key of the cache-blocked wave order (plan_wave_schedule).
struct WaveKey { std::uint64_t k1, k2; int w; };

// One wave schedule, its arrays carved from scratch (take_wave_schedule): the tables gjlo / gcnt / woff in STORAGE
// order and rank (ranked: maps a stream-order wave number to its storage position), the planning scratch, and for a
// private schedule the placement of every record (wid / pid) and the regrouped records wrec.  capacity == 0: the wave
// order is off for this stream.
template <typename T>
struct WaveSchedule {
    bool ok = false, ranked = false;
    std::size_t capacity = 0, slots = 0, nw = 0;
    int *gjlo = nullptr, *gcnt = nullptr, *rank = nullptr;
    std::int64_t* woff = nullptr;
    int *run_j0 = nullptr, *run_len = nullptr, *run_w = nullptr;
    int *last = nullptr, *level = nullptr, *ord = nullptr, *gj2 = nullptr, *gc2 = nullptr;
    WaveKey* keys = nullptr;
    int* wid = nullptr;
    std::uint8_t* pid = nullptr;
    hb2st_rot<T>* wrec = nullptr;
    WaveView<T> view() const { return {gjlo, woff, gcnt, wrec, nw}; }
};

int tile_from_env();
CpuReplayPolicy replay_policy_for_threads(int threads)
{
    CpuReplayPolicy p;p.threads=std::max(1,threads);p.tile=tile_from_env();
    static const int mode=[] {const char* e=std::getenv("DDLA_EIG_REPLAY_WAVE");return e?std::atoi(e):1;}();
    static const int rb=[] {const char* e=std::getenv("DDLA_EIG_REPLAY_RB");const int v=e?std::atoi(e):512;return v>=8?v:512;}();
    static const int lam=[] {const char* e=std::getenv("DDLA_EIG_REPLAY_LAM");const int v=e?std::atoi(e):32;return v>=1?v:32;}();
    static const bool prof=[] {const char* e=std::getenv("DDLA_EIG_REPLAY_PROF");return e&&std::atoi(e)!=0;}();
    p.wave_mode=mode;p.row_block=rb;p.level_block=lam;p.profile=prof;p.stage_profile=eig_profile_enabled();
    const char* e=std::getenv("DDLA_EIG_REPLAY_WAVE");p.waves=!e||std::atoi(e)!=0;return p;
}

// The cache-blocked wave order (below) for this order and policy.
bool wave_tiled(int n, const CpuReplayPolicy& policy)
{
    return policy.wave_mode == 2 || (policy.wave_mode == 1 && n >= 4096);   // below ~1 MiB of tile the cache holds it anyway
}

// Bytes of one wave schedule of up to `capacity` records of an order-n chase with band kd (none when the wave order
// is off or cannot apply), in carving order: gjlo, gcnt, woff (capacity + 1) and rank; the chase slots run_j0, run_len
// and run_w (n / (kd + 1) + 2 each); with the tiled order last (n), level, keys, ord, gj2 and gc2; then, for a private
// schedule (records), wid, pid and the regrouped records.
template <typename T>
std::size_t wave_schedule_bytes(int n, int kd, std::size_t capacity, const CpuReplayPolicy& policy, bool records)
{
    if (!policy.waves || n < 4 || capacity == 0 || kd < 2) return 0;
    if (capacity > std::size_t(std::numeric_limits<int>::max())) throw std::overflow_error("CPU replay: wave index overflow");
    const std::size_t slots = std::size_t(n) / (std::size_t(kd) + 1) + 2;
    std::size_t bytes = 3 * padded_bytes<int>(capacity) + padded_bytes<std::int64_t>(capacity + 1) + 3 * padded_bytes<int>(slots);
    if (wave_tiled(n, policy))
        bytes += padded_bytes<int>(n) + padded_bytes<int>(capacity) + padded_bytes<WaveKey>(capacity) + 3 * padded_bytes<int>(capacity);
    if (records)
        bytes += padded_bytes<int>(capacity) + padded_bytes<std::uint8_t>(capacity) + padded_bytes<hb2st_rot<T>>(capacity);
    return bytes;
}

// The schedule's arrays, carved from cursor in the order wave_schedule_bytes counts them.
template <typename T>
WaveSchedule<T> take_wave_schedule(char*& cursor, int n, int kd, std::size_t capacity, const CpuReplayPolicy& policy,
                                   bool records)
{
    WaveSchedule<T> s;
    if (!wave_schedule_bytes<T>(n, kd, capacity, policy, records)) return s;
    s.capacity = capacity;
    s.slots = std::size_t(n) / (std::size_t(kd) + 1) + 2;
    s.gjlo = take<int>(cursor, capacity);
    s.gcnt = take<int>(cursor, capacity);
    s.woff = take<std::int64_t>(cursor, capacity + 1);
    s.rank = take<int>(cursor, capacity);
    s.run_j0 = take<int>(cursor, s.slots);
    s.run_len = take<int>(cursor, s.slots);
    s.run_w = take<int>(cursor, s.slots);
    if (wave_tiled(n, policy)) {
        s.last = take<int>(cursor, n);
        s.level = take<int>(cursor, capacity);
        s.keys = take<WaveKey>(cursor, capacity);
        s.ord = take<int>(cursor, capacity);
        s.gj2 = take<int>(cursor, capacity);
        s.gc2 = take<int>(cursor, capacity);
    }
    if (records) {
        s.wid = take<int>(cursor, capacity);
        s.pid = take<std::uint8_t>(cursor, capacity);
        s.wrec = take<hb2st_rot<T>>(cursor, capacity);
    }
    return s;
}

// The PLAN of the wave schedule: validates the stream against the chase's integer replay and computes the wave tables
// and, for every record, its wave and member number (wid / pid: caller-provided, nrots entries each).  The regrouped
// records are produced separately (regroup_records), so the plan can run on one rank of a node and the regrouping be
// shared out.  Returns false (no waves) for anything that is not a chase stream.  s's tables are in STORAGE order;
// s.rank (when s.ranked) maps a stream-order wave number to its storage position.
template <typename T>
bool plan_wave_schedule(int n, const hb2st_rot<T>* rots, std::size_t nrots, int* wid, std::uint8_t* pid,
                        WaveSchedule<T>& s, const CpuReplayPolicy& policy,
                        const ChaseCursor* start=nullptr, int start_offset=0)
{
    s.nw = 0;
    s.ranked = false;
    if (!policy.waves || nrots==0 || n<4) return false;
    const int kdn = start ? std::min(n-1,start->kd) : rots[0].j0;
    if (kdn < 2 || kdn > n - 1) return false;
    const int kd1 = start ? start->kd+1 : kdn+1;
    const int K = std::min(kdn, 64);   // wave-length cap (the natural run length is ~kd/2)
    const int maxt = n / kd1 + 2;
    // A stream whose band differs from the one the schedule was sized for keeps the plain order.
    if (nrots > s.capacity || static_cast<std::size_t>(maxt) > s.slots) return false;
    int* const gjlo = s.gjlo;
    int* const gcnt = s.gcnt;
    int* const run_j0 = s.run_j0;
    int* const run_len = s.run_len;
    int* const run_w = s.run_w;
    std::fill_n(run_j0, maxt, 0);
    std::fill_n(run_len, maxt, 0);
    std::fill_n(run_w, maxt, 0);
    std::size_t nw = 0;
    std::size_t q = 0;
    bool valid = true;
    ChaseCursor cursor=start ? *start : ChaseCursor(n,kdn);
    int offset=start_offset;
    while (!cursor.done() && valid && q<nrots) {
        const int nr=cursor.count();
        for(int t=offset;t<nr && q<nrots;++t) {
            const int j=cursor.first()+t*kd1;
            if (rots[q].j0!=j) { valid=false; break; }
                int w, p;
                if (run_len[static_cast<std::size_t>(t)] > 0 &&
                    j == run_j0[static_cast<std::size_t>(t)] - run_len[static_cast<std::size_t>(t)] &&
                    run_len[static_cast<std::size_t>(t)] < K) {
                    w = run_w[static_cast<std::size_t>(t)];
                    p = run_len[static_cast<std::size_t>(t)]++;
                    gcnt[static_cast<std::size_t>(w)] = run_len[static_cast<std::size_t>(t)];
                } else {
                    w = static_cast<int>(nw);
                    gjlo[nw] = j;
                    gcnt[nw] = 1;
                    ++nw;
                    run_w[static_cast<std::size_t>(t)] = w;
                    run_len[static_cast<std::size_t>(t)] = 1;
                    run_j0[static_cast<std::size_t>(t)] = j;
                    p = 0;
                }
                wid[q] = w;
                pid[q] = static_cast<std::uint8_t>(p);
                ++q;
        }
        offset=0;
        cursor.next();
    }
    if (!start && valid) {
        while (!cursor.done() && cursor.count()==0) cursor.next();
    }
    if (!valid || q != nrots || (!start && !cursor.done()))   // not a chase stream (or a different band): keep the plain order
        return false;

    // ---- cache-blocked wave ORDER (skewed tiles) -------------------------------------------------------
    // Every row of a tile is rotated by hundreds of waves; in stream order consecutive waves sit ~kd rows apart, so the
    // n-row tile is re-read from L3/DRAM about once per wave (the replay was DRAM-bound at large n).  Reorder the waves
    // (table order stays the dependency order) into skewed tiles:
    //   level(w)  = 1 + max over w's rows of the level of the last earlier wave touching the row   (longest path: two
    //               overlapping waves are ordered, and the later one has the larger level);
    //   u(w)      = top(w) - s*level(w),  s = longest wave + 1;  band(w) = floor(u / RB);  block(w) = (level-1) / LAM;
    //   forward order = blocks ascending, bands DESCENDING, level then top ascending inside a tile.
    // An edge X -> Z (Z after X, overlapping, so |top_Z - top_X| <= longest wave < s and level_Z > level_X) has
    // u_Z - u_X <= (s-1) - s < 0, i.e. band(Z) <= band(X) with block(Z) >= block(X): descending bands inside a block
    // and blocks ascending respect every dependency.  A tile touches ~RB + s*LAM rows (~260 KB at 256-byte rows), so
    // it stays in L2.  Measured (n=10000, kd=64, W=32, one tile per thread, Xeon Gold 6138): 2.2 G col-rot/s at one
    // thread (plain 1.1, waves 1.8) and 18 vs 8 (plain) / 11.4 (waves) G at 8 threads; bit-identical to the plain order.
    const int RB=policy.row_block,LAM=policy.level_block;
    // forward processing order (s.ord: indices into the wave table); none = table order
    if (wave_tiled(n, policy) && nw > 0) {
        int* const last = s.last;
        int* const level = s.level;
        WaveKey* const keys = s.keys;
        int* const ord = s.ord;
        assert(last && level && keys && ord);
        int maxcnt = 0;
        for (std::size_t w = 0; w < nw; ++w) maxcnt = std::max(maxcnt, gcnt[w]);
        const int skew = maxcnt + 1;
        std::fill_n(last, n, 0);
        std::fill_n(level, nw, 0);
        int maxlevel = 0;
        for (std::size_t w = 0; w < nw; ++w) {
            const int lo = gjlo[w] - gcnt[w], hi = gjlo[w];
            int lv = 0;
            for (int r = lo; r <= hi; ++r) lv = std::max(lv, last[static_cast<std::size_t>(r)]);
            ++lv;
            for (int r = lo; r <= hi; ++r) last[static_cast<std::size_t>(r)] = lv;
            level[w] = lv;
            maxlevel = std::max(maxlevel, lv);
        }
        const long long off = static_cast<long long>(skew) * maxlevel;
        const long long nbands = (off + n) / RB + 2;
        for (std::size_t w = 0; w < nw; ++w) {
            const long long u = static_cast<long long>(gjlo[w]) - static_cast<long long>(skew) * level[w] + off;
            keys[w].k1 = (static_cast<std::uint64_t>((level[w] - 1) / LAM) << 32) |
                         static_cast<std::uint64_t>(nbands - 1 - u / RB);
            keys[w].k2 = (static_cast<std::uint64_t>(level[w]) << 32) | static_cast<std::uint64_t>(gjlo[w]);
            keys[w].w = static_cast<int>(w);
        }
        std::sort(keys, keys + nw, [](const WaveKey& a, const WaveKey& b) {
            return a.k1 != b.k1 ? a.k1 < b.k1 : a.k2 < b.k2;
        });
        for (std::size_t i = 0; i < nw; ++i) ord[i] = keys[i].w;
        // storage position of every wave (the replay walks the storage arrays backwards)
        for (std::size_t i = 0; i < nw; ++i) s.rank[static_cast<std::size_t>(ord[i])] = static_cast<int>(i);
        for (std::size_t i = 0; i < nw; ++i) {
            s.gj2[i] = gjlo[static_cast<std::size_t>(ord[i])];
            s.gc2[i] = gcnt[static_cast<std::size_t>(ord[i])];
        }
        std::swap(s.gjlo, s.gj2);
        std::swap(s.gcnt, s.gc2);
        s.ranked = true;
    }
    s.woff[0] = 0;
    for (std::size_t w = 0; w < nw; ++w) s.woff[w + 1] = s.woff[w] + s.gcnt[w];
    s.nw = nw;
    return true;
}

// dst[woff[wave] + member] = rots[r] for the records [r0, r1): the regrouping, on nthreads threads.  Distinct records
// have distinct (wave, member) slots, so concurrent callers (the ranks of a node, each with its own r-range) never
// write the same element.
template <typename T>
void regroup_records(const hb2st_rot<T>* rots, std::size_t r0, std::size_t r1, const int* wid, const std::uint8_t* pid,
                     const int* rank, const std::int64_t* woff, hb2st_rot<T>* dst, int nthreads)
{
    if (r1 <= r0) return;
    host_parallel_for(static_cast<long long>(r1 - r0), 1 << 18, nthreads, [&](long long a, long long b) {
        for (long long i = a; i < b; ++i) {
            const std::size_t r = r0 + static_cast<std::size_t>(i);
            const int w = rank ? rank[wid[r]] : wid[r];
            dst[woff[w] + pid[r]] = rots[r];
        }
    });
}

// The private schedule of records [0, count) of a stream (or of one history chunk from its checkpoint): plan and
// regroup into s.wrec.  s.ok tells whether the replay can use it.
template <typename T>
void build_wave_schedule(WaveSchedule<T>& s, int n, const hb2st_rot<T>* records, std::size_t count, int threads,
                         const CpuReplayPolicy& policy, const ChaseCursor* start=nullptr, int offset=0)
{
    s.ok = false;
    if (!s.capacity) return;
    if (!plan_wave_schedule<T>(n, records, count, s.wid, s.pid, s, policy, start, offset)) return;
    regroup_records<T>(records, 0, count, s.wid, s.pid, s.ranked ? s.rank : nullptr, s.woff, s.wrec, threads);
    s.ok = true;
}

// Collective allocation of an MPI shared-memory window of `bytes` (the node leader allocates, everybody queries).  The
// outcome is agreed on every rank: on failure everything is released and false returned.
bool node_alloc_shared(MPI_Comm node, std::size_t bytes, NodeWindow& w)
{
    int nr = 0;
    MPI_Comm_rank(node, &nr);
    void* mybase = nullptr;
    MPI_Win win = MPI_WIN_NULL;
    const int rc = MPI_Win_allocate_shared(nr == 0 ? static_cast<MPI_Aint>(std::max<std::size_t>(bytes, 8)) : 0, 8,
                                           MPI_INFO_NULL, node, &mybase, &win);
    int ok = (rc == MPI_SUCCESS);
    MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, node);
    void* base = nullptr;
    if (ok) {
        MPI_Aint qsz = 0;
        int qdisp = 0;
        if (MPI_Win_shared_query(win, 0, &qsz, &qdisp, &base) != MPI_SUCCESS || base == nullptr ||
            static_cast<std::size_t>(qsz) < bytes)
            ok = 0;
    }
    MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, node);
    if (!ok) {
        if (win != MPI_WIN_NULL) MPI_Win_free(&win);
        return false;
    }
    w.win = win;
    w.base = base;
    return true;
}

// Wave tile kernels: waves in REVERSE table order, members p = cnt-1 .. 0 (rows ascending), the carried row in
// registers.  Same per-column arithmetic as rot_tile_real / rot_tile_cplx, so the tile is bitwise the plain
// kernel's.  Member p of a wave acts on rows (top-p-1, top-p).
template <int W, typename rt>
DDLA_CPU_CLONES void wave_tile_real(rt* __restrict X, const hb2st_rot<rt>* wrec, const std::int64_t* woff,
                                    const int* gjlo, const int* gcnt, std::size_t nw)
{
    for (std::size_t w = nw; w-- > 0;) {
        const int cnt = gcnt[w], top = gjlo[w];
        const hb2st_rot<rt>* rec = wrec + woff[w];
        rt x[W];
        const rt* src = X + static_cast<std::size_t>(top - cnt) * W;
        for (int k = 0; k < W; ++k) x[k] = src[k];
        for (int p = cnt - 1; p >= 0; --p) {
            const rt c = rec[p].c, s = rec[p].s;
            rt* rb = X + static_cast<std::size_t>(top - p) * W;
            rt* ra = rb - W;
#pragma GCC ivdep
            for (int k = 0; k < W; ++k) {
                const rt xa = x[k], xb = rb[k];
                ra[k] = c * xa - s * xb;
                x[k] = s * xa + c * xb;
            }
        }
        rt* last = X + static_cast<std::size_t>(top) * W;
        for (int k = 0; k < W; ++k) last[k] = x[k];
    }
}

template <int W, typename rt>
DDLA_CPU_CLONES void wave_tile_cplx(rt* __restrict X, const hb2st_rot<std::complex<rt>>* wrec,
                                    const std::int64_t* woff, const int* gjlo, const int* gcnt, std::size_t nw)
{
    for (std::size_t w = nw; w-- > 0;) {
        const int cnt = gcnt[w], top = gjlo[w];
        const hb2st_rot<std::complex<rt>>* rec = wrec + woff[w];
        rt xr[W], xi[W];
        const rt* src = X + static_cast<std::size_t>(top - cnt) * (2 * W);
        for (int k = 0; k < W; ++k) { xr[k] = src[k]; xi[k] = src[W + k]; }
        for (int p = cnt - 1; p >= 0; --p) {
            const rt c = rec[p].c, sr = rec[p].s.real(), si = rec[p].s.imag();
            rt* brr = X + static_cast<std::size_t>(top - p) * (2 * W);
            rt* bri = brr + W;
            rt* arr = brr - 2 * W;
            rt* ari = arr + W;
#pragma GCC ivdep
            for (int k = 0; k < W; ++k) {
                const rt xar = xr[k], xai = xi[k], xbr = brr[k], xbi = bri[k];
                const rt t2r = sr * xbr + si * xbi;
                const rt t2i = sr * xbi - si * xbr;
                arr[k] = c * xar - t2r;
                ari[k] = c * xai - t2i;
                const rt u1r = sr * xar - si * xai;
                const rt u1i = sr * xai + si * xar;
                xr[k] = u1r + c * xbr;
                xi[k] = u1i + c * xbi;
            }
        }
        rt* lr = X + static_cast<std::size_t>(top) * (2 * W);
        for (int k = 0; k < W; ++k) { lr[k] = xr[k]; lr[W + k] = xi[k]; }
    }
}

template <typename T, int W>
void wave_tile(typename real_type<T>::type* X, const WaveView<T>& ws)
{
    using rt = typename real_type<T>::type;
    if constexpr (k_real<T>)
        wave_tile_real<W, rt>(X, ws.wrec, ws.woff, ws.gjlo, ws.gcnt, ws.nw);
    else
        wave_tile_cplx<W, rt>(X, ws.wrec, ws.woff, ws.gjlo, ws.gcnt, ws.nw);
}

// ---- tile width --------------------------------------------------------------------------------
int tile_from_env()
{
    static const int v = [] {
        const char* e = std::getenv("DDLA_EIG_REPLAY_TILE");
        const int t = e ? std::atoi(e) : 0;
        return (t == 8 || t == 16 || t == 32 || t == 64) ? t : 0;
    }();
    return v;
}

// ---- tile gather / scatter ----------------------------------------------------------------------
template <typename T, int W>
void gather_tile(typename real_type<T>::type* X, const typename real_type<T>::type* h_Y, int ldy,
                 const int* cols, int wk, int n, const T* phases)
{
    using rt = typename real_type<T>::type;
    if constexpr (k_real<T>) {
        for (int k = 0; k < W; ++k) {
            if (k < wk) {
                const rt* col = h_Y + static_cast<std::size_t>(cols[k]) * ldy;
                for (int r = 0; r < n; ++r) X[static_cast<std::size_t>(r) * W + k] = col[r];
            } else {
                for (int r = 0; r < n; ++r) X[static_cast<std::size_t>(r) * W + k] = rt(0);
            }
        }
        (void)phases;
    } else {
        const std::size_t rs = 2 * W;
        for (int k = 0; k < W; ++k) {
            if (k < wk) {
                const rt* col = h_Y + static_cast<std::size_t>(cols[k]) * ldy;
                // row 0 is not scaled; row i+1 is multiplied by phases[i] exactly as the serial
                // `row[k] *= ph` on the complex value (v, 0) did: (v*pr - 0*pi, v*pi + 0*pr)
                X[k] = col[0];
                X[W + k] = rt(0);
                const rt zero = rt(0);
                for (int r = 1; r < n; ++r) {
                    const rt v = col[r], pr = phases[r - 1].real(), pi = phases[r - 1].imag();
                    X[static_cast<std::size_t>(r) * rs + k] = v * pr - zero * pi;
                    X[static_cast<std::size_t>(r) * rs + W + k] = v * pi + zero * pr;
                }
            } else {
                for (int r = 0; r < n; ++r) {
                    X[static_cast<std::size_t>(r) * rs + k] = rt(0);
                    X[static_cast<std::size_t>(r) * rs + W + k] = rt(0);
                }
            }
        }
    }
}

template <typename T, int W>
void pack_tile(const typename real_type<T>::type* X, int wk, int n, const std::size_t* rowoff, int k0, T* sendbuf)
{
    using rt = typename real_type<T>::type;
    for (int r = 0; r < n; ++r) {
        T* dst = sendbuf + rowoff[r] + k0;
        if constexpr (k_real<T>) {
            std::memcpy(dst, X + static_cast<std::size_t>(r) * W, static_cast<std::size_t>(wk) * sizeof(T));
        } else {
            const rt* re = X + static_cast<std::size_t>(r) * (2 * W);
            const rt* im = re + W;
            for (int k = 0; k < wk; ++k) dst[k] = T(re[k], im[k]);
        }
    }
}

} // namespace

bool cpu_replay_last_used_waves() { return g_last_replay_waves.load(std::memory_order_relaxed); }

bool host_run_team(int nthreads,void* context,void (*function)(void*,int))
{
#ifdef _OPENMP
    if (nthreads <= 1) return false;
    int got = 0;
#pragma omp parallel num_threads(nthreads)
    {
#pragma omp single
        got = omp_get_num_threads();
    }
    if (got != nthreads) return false;
#pragma omp parallel num_threads(nthreads)
    {
        function(context,omp_get_thread_num());
    }
    return true;
#else
    (void)nthreads;
    (void)context;(void)function;
    return false;
#endif
}

bool host_run_team(int nthreads,const std::function<void(int)>& function)
{
    return host_run_team(nthreads,const_cast<void*>(static_cast<const void*>(&function)),
        [](void* context,int role){(*static_cast<const std::function<void(int)>*>(context))(role);});
}

namespace {
// Local allocation or worker failures must be agreed before the next history
// fetch (and before the caller's redistribution). MPI stays on this thread.
template<class F>
void replay_collective_local(MPI_Comm comm, F&& work)
{
    std::exception_ptr failure;
    try { work(); } catch (...) { failure = std::current_exception(); }
    int ok = failure ? 0 : 1, all_ok = 0;
    MPI_CHECK(MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, comm));
    if (!all_ok) {
        if (failure) std::rethrow_exception(failure);
        throw std::runtime_error("distributed CPU replay failed on another rank");
    }
}

} // namespace

CpuReplayPolicy cpu_replay_policy_for_threads(int threads)
{return replay_policy_for_threads(threads);}
CpuReplayPolicy cpu_replay_policy(MPI_Comm comm)
{
    // All ranks enter the topology preflight, even if some previously warmed
    // the legacy worker cache and others did not.
    int ranks=1;MPI_Comm node=MPI_COMM_NULL;MPI_CHECK(MPI_Comm_split_type(comm,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node));
    MPI_CHECK(MPI_Comm_size(node,&ranks));MPI_CHECK(MPI_Comm_free(&node));return replay_policy_for_threads(host_omp_threads_for_node(ranks));
}

namespace {
// The tile slots start on a 64-byte boundary (the kernels' vector rows); the
// sizes reserve the up to 64 bytes this skips.
constexpr std::size_t tile_alignment=64;
inline char* align_tiles(char* p)
{return reinterpret_cast<char*>((reinterpret_cast<std::uintptr_t>(p)+tile_alignment-1)/tile_alignment*tile_alignment);}
template<class T>int replay_tile_geometry(int ncs,const CpuReplayPolicy& p)
{
    if(p.tile)return p.tile;int w=std::max(8,std::min(64,int(256/sizeof(T))));
    while(w>8&&(std::int64_t(ncs)+w-1)/w<2LL*p.threads)w/=2;return w;
}
// Reals between two tile slots of the pack: n x tile (two planes for complex), rounded up to 64 bytes.
template<class T>std::size_t pack_tile_stride(const CpuReplayPackPlan& p)
{return (std::size_t(std::max(p.n,0))*std::size_t(p.tile)*sizeof(T)+63)/64*64/sizeof(typename real_type<T>::type);}
// The pack's tile slots: one per thread, or one per tile with a distributed history (the tiles live across chunks).
template<class T>std::size_t pack_data_count(const CpuReplayPackPlan& p)
{
    std::size_t slots=1;
#ifdef _OPENMP
    slots=std::size_t(p.policy.threads);
#endif
    if(p.source==CpuReplaySource::Distributed)slots=std::size_t((std::int64_t(p.ncs)+p.tile-1)/p.tile);
    return std::max(std::size_t(1),pack_tile_stride<T>(p)*slots);
}
// The pack plans its own wave schedule: 16 or more columns of a stream that can be a chase stream.
inline bool pack_waves(const CpuReplayPackPlan& p)
{return p.source!=CpuReplaySource::Shared&&p.ncs>=16&&(p.source==CpuReplaySource::Distributed||p.kd<=p.n-1);}
template<class T,int W>void replay_pack_tiles(const CpuReplayPackPlan& p,const typename real_type<T>::type* Y,int ldy,const int* columns,
    const hb2st_rot<T>* records,std::size_t count,const T* phases,const std::size_t* rowoff,T* sendbuf,const WaveView<T>* wave,
    typename real_type<T>::type* data)
{
    using R=typename real_type<T>::type;
    const int tiles=int((std::int64_t(p.ncs)+W-1)/W);const std::size_t stride=pack_tile_stride<T>(p);
#ifdef _OPENMP
#pragma omp parallel num_threads(p.policy.threads)
#endif
    {
        int role=0;
#ifdef _OPENMP
        role=omp_get_thread_num();
#endif
        R* X=data+std::size_t(role)*stride;
#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
        for(int t=0;t<tiles;++t){const int first=t*W,width=std::min(W,p.ncs-first);gather_tile<T,W>(X,Y,ldy,columns+first,width,p.n,phases);if(wave)wave_tile<T,W>(X,*wave);else rot_tile<T,W>(X,records,count,p.n);pack_tile<T,W>(X,width,p.n,rowoff,first,sendbuf);}
    }
}
template<class T,int W>void replay_pack_distributed(const CpuReplayPackPlan& p,const typename real_type<T>::type* Y,int ldy,const int* columns,
    const DistributedHistory<T>& history,MPI_Comm comm,const T* phases,const std::size_t* rowoff,T* sendbuf,
    typename real_type<T>::type* data,hb2st_rot<T>* records,WaveSchedule<T>& schedule)
{
    using R=typename real_type<T>::type;
    const int tiles=int((std::int64_t(p.ncs)+W-1)/W);const auto stride=pack_tile_stride<T>(p);
    replay_collective_local(comm,[&]{host_parallel_for(tiles,1,p.policy.threads,[&](long long lo,long long hi){for(auto t=lo;t<hi;++t)gather_tile<T,W>(data+t*stride,Y,ldy,columns+t*W,std::min(W,p.ncs-int(t)*W),p.n,phases);});});
    for(long long chunk=history.chunks();chunk-->0;){DistributedHistoryHeader header;history.fetch(chunk,comm,records,&header);
        replay_collective_local(comm,[&]{if(p.ncs>=16)build_wave_schedule(schedule,p.n,records,header.count,p.policy.threads,p.policy,&header.cursor,header.offset);const auto view=schedule.view();
            host_parallel_for(tiles,1,p.policy.threads,[&](long long lo,long long hi){for(auto t=lo;t<hi;++t){R* x=data+t*stride;if(schedule.ok)wave_tile<T,W>(x,view);else rot_tile<T,W>(x,records,header.count,p.n);}});});
    }
    replay_collective_local(comm,[&]{host_parallel_for(tiles,1,p.policy.threads,[&](long long lo,long long hi){for(auto t=lo;t<hi;++t)pack_tile<T,W>(data+t*stride,std::min(W,p.ncs-int(t)*W),p.n,rowoff,int(t)*W,sendbuf);});});
}
}
template<class T>CpuReplayPackPlan cpu_replay_pack_geometry(int n,int ncs,int mb,int npr,int kd,std::size_t records,CpuReplaySource source,const CpuReplayPolicy& policy)
{return {n,ncs,mb,npr,replay_tile_geometry<T>(ncs,policy),kd,records,policy,source};}
template<class T>void cpu_replay_pack_check(const CpuReplayPackPlan& p,const typename real_type<T>::type* Y,std::size_t y_count,int ldy,const int* columns,
    const hb2st_rot<T>* records,std::size_t record_count,const WaveShared<T>* shared,const DistributedHistory<T>* history,const T* phases,const int* sd,T* sendbuf)
{
    if(p.n<0||p.ncs<0||p.mb<1||p.npr<1||p.policy.threads<1||!sd||(p.ncs&&!columns)||(p.ncs&&p.n&&(!Y||!sendbuf||ldy<p.n)))throw std::invalid_argument("CPU replay pack: invalid dimensions or pointers");
    if constexpr(!k_real<T>){if(p.ncs&&p.n>1&&!phases)throw std::invalid_argument("CPU replay pack: phases required");}
    for(int q=0;q<p.npr;++q)if(sd[q]<0)throw std::invalid_argument("CPU replay pack: negative displacement");for(int j=0;j<p.ncs;++j){
        if(columns[j]<0)throw std::invalid_argument("CPU replay pack: invalid input column");
        const std::size_t begin=std::size_t(columns[j])*std::size_t(std::max(ldy,0));
        if(begin>y_count||std::size_t(p.n)>y_count-begin)throw std::invalid_argument("CPU replay pack: input column exceeds supplied slab");
    }
    if(p.source==CpuReplaySource::Records&&(record_count!=p.records||(record_count&&!records)))throw std::invalid_argument("CPU replay pack: incompatible records");
    if(p.source==CpuReplaySource::Shared&&(!shared||!shared->ok||(shared->nw&&(!shared->gjlo||!shared->gcnt||!shared->woff||!shared->wrec))))throw std::invalid_argument("CPU replay pack: incompatible shared schedule");
    if(p.source==CpuReplaySource::Distributed&&(!history||!history->ready||history->n!=p.n||history->kd!=p.kd||p.records!=std::size_t(std::min(history->total,262144LL))))throw std::invalid_argument("CPU replay pack: incompatible history");
}
template<class T>void cpu_replay_pack_workspace_size(const CpuReplayPackPlan& p,std::size_t* host_bytes)
{
    using R=typename real_type<T>::type;*host_bytes=0;
    if(p.source!=CpuReplaySource::Distributed&&(p.n<=0||p.ncs<=0))return;
    std::size_t bytes=padded_bytes<std::size_t>(std::size_t(std::max(p.n,0)))+tile_alignment+padded_bytes<R>(pack_data_count<T>(p));
    if(p.source==CpuReplaySource::Distributed)bytes+=padded_bytes<hb2st_rot<T>>(p.records);
    if(pack_waves(p))bytes+=wave_schedule_bytes<T>(p.n,p.kd,p.records,p.policy,true);
    *host_bytes=bytes;
}
template<class T>void cpu_replay_pack_run(const CpuReplayPackPlan& p,const typename real_type<T>::type* Y,int ldy,const int* columns,
    const hb2st_rot<T>* records,const WaveShared<T>* shared,const DistributedHistory<T>* history,MPI_Comm comm,const T* phases,const int* sd,T* sendbuf,
    void* host_work,std::size_t host_bytes)
{
    using R=typename real_type<T>::type;
    if(p.source!=CpuReplaySource::Distributed&&(p.n<=0||p.ncs<=0))return;
    char* const base=static_cast<char*>(host_work);char* w=base;
    std::size_t* const rowoff=take<std::size_t>(w,std::size_t(std::max(p.n,0)));
    w=align_tiles(w);R* const data=take<R>(w,pack_data_count<T>(p));
    hb2st_rot<T>* const chunk=p.source==CpuReplaySource::Distributed?take<hb2st_rot<T>>(w,p.records):nullptr;
    WaveSchedule<T> schedule;if(pack_waves(p))schedule=take_wave_schedule<T>(w,p.n,p.kd,p.records,p.policy,true);
    assert(std::size_t(w-base)<=host_bytes);(void)host_bytes;
    for(int r=0;r<p.n;++r){const int pr=(r/p.mb)%p.npr,li=p.mb*(r/p.mb/p.npr)+r%p.mb;rowoff[r]=std::size_t(sd[pr])+std::size_t(li)*p.ncs;}
    if(p.source==CpuReplaySource::Distributed){
#define DIST(W) replay_pack_distributed<T,W>(p,Y,ldy,columns,*history,comm,phases,rowoff,sendbuf,data,chunk,schedule)
        switch(p.tile){case 8:DIST(8);break;case 16:DIST(16);break;case 32:DIST(32);break;default:DIST(64);break;}
#undef DIST
        return;
    }
    const auto begin=std::chrono::steady_clock::now();WaveView<T> view;const WaveView<T>* wave=nullptr;
    if(p.source==CpuReplaySource::Shared){view={shared->gjlo,shared->woff,shared->gcnt,shared->wrec,shared->nw};wave=&view;}
    else if(p.ncs>=16){build_wave_schedule(schedule,p.n,records,p.records,p.policy.threads,p.policy);if(schedule.ok){view=schedule.view();wave=&view;}}
    g_last_replay_waves.store(wave!=nullptr,std::memory_order_relaxed);const auto planned=std::chrono::steady_clock::now();
#define PACK(W) replay_pack_tiles<T,W>(p,Y,ldy,columns,records,p.records,phases,rowoff,sendbuf,wave,data)
    switch(p.tile){case 8:PACK(8);break;case 16:PACK(16);break;case 32:PACK(32);break;default:PACK(64);break;}
#undef PACK
    if(p.policy.profile&&p.source==CpuReplaySource::Records){const auto end=std::chrono::steady_clock::now();std::printf("REPLAY_PROF n=%d ncs=%d threads=%d waves=%d schedule=%.3f s replay=%.3f s\n",p.n,p.ncs,p.policy.threads,wave?1:0,std::chrono::duration<double>(planned-begin).count(),std::chrono::duration<double>(end-planned).count());std::fflush(stdout);}
}

template <typename T>
void cpu_replay_pack_distributed(const typename real_type<T>::type* Y,std::size_t y_count,int ldy,const int* cols,int ncs,int n,
    const DistributedHistory<T>& history,MPI_Comm comm,const T* phases,int mb,int npr,const int* sd,T* sendbuf,int nthreads)
{
    const auto p=cpu_replay_pack_geometry<T>(n,ncs,mb,npr,history.kd,std::size_t(std::min(history.total,history.chunk_size)),CpuReplaySource::Distributed,replay_policy_for_threads(nthreads));
    std::size_t host_bytes=0;std::vector<char> host_work;
    replay_collective_local(comm,[&]{cpu_replay_pack_check<T>(p,Y,y_count,ldy,cols,nullptr,0,nullptr,&history,phases,sd,sendbuf);cpu_replay_pack_workspace_size<T>(p,&host_bytes);host_work.resize(host_bytes);});
    cpu_replay_pack_run<T>(p,Y,ldy,cols,nullptr,nullptr,&history,comm,phases,sd,sendbuf,host_work.data(),host_bytes);
}
template <typename T>
void cpu_replay_pack(const typename real_type<T>::type* Y,std::size_t y_count,int ldy,const int* cols,int ncs,int n,
    const hb2st_rot<T>* rots,std::size_t nrots,const T* phases,int mb,int npr,const int* sd,T* sendbuf,int nthreads)
{
    if(ncs<=0||n<=0)return;if(!k_real<T>&&n>1&&!phases)throw std::runtime_error("cpu_replay_pack: complex replay needs the phases");
    const auto p=cpu_replay_pack_geometry<T>(n,ncs,mb,npr,nrots?rots[0].j0:1,nrots,CpuReplaySource::Records,replay_policy_for_threads(nthreads));
    cpu_replay_pack_check<T>(p,Y,y_count,ldy,cols,rots,nrots,nullptr,nullptr,phases,sd,sendbuf);
    std::size_t host_bytes=0;cpu_replay_pack_workspace_size<T>(p,&host_bytes);std::vector<char> host_work(host_bytes);
    cpu_replay_pack_run<T>(p,Y,ldy,cols,rots,nullptr,nullptr,MPI_COMM_NULL,phases,sd,sendbuf,host_work.data(),host_bytes);
}
template <typename T>
void cpu_replay_pack_shared(const typename real_type<T>::type* Y,std::size_t y_count,int ldy,const int* cols,int ncs,int n,
    const WaveShared<T>& shared,const T* phases,int mb,int npr,const int* sd,T* sendbuf,int nthreads)
{
    if(ncs<=0||n<=0)return;if(!shared.ok)throw std::logic_error("cpu_replay_pack_shared: empty schedule");
    if(!k_real<T>&&n>1&&!phases)throw std::runtime_error("cpu_replay_pack_shared: complex replay needs the phases");
    const auto p=cpu_replay_pack_geometry<T>(n,ncs,mb,npr,1,shared.nrots,CpuReplaySource::Shared,replay_policy_for_threads(nthreads));
    cpu_replay_pack_check<T>(p,Y,y_count,ldy,cols,nullptr,0,&shared,nullptr,phases,sd,sendbuf);
    std::size_t host_bytes=0;cpu_replay_pack_workspace_size<T>(p,&host_bytes);std::vector<char> host_work(host_bytes);
    cpu_replay_pack_run<T>(p,Y,ldy,cols,nullptr,&shared,nullptr,MPI_COMM_NULL,phases,sd,sendbuf,host_work.data(),host_bytes);
}

namespace {
// Whether the plan's stream gets a node-shared schedule, and the bytes of its windows: the record placement (wid,
// pid), the tables (woff | gjlo | gcnt | rank) and the regrouped records.
template<class T>bool node_wave_windows(const CpuNodeWavePlan& p,std::size_t* index,std::size_t* table,std::size_t* records)
{
    *index=*table=*records=0;
    if(p.kd>p.n-1||!wave_schedule_bytes<T>(p.n,p.kd,p.records,p.policy,false))return false;
    *index=p.records*(sizeof(int)+sizeof(std::uint8_t));*table=(p.records+1)*sizeof(std::int64_t)+p.records*3*sizeof(int);
    *records=p.records*sizeof(hb2st_rot<T>);return true;
}
}
template<class T>CpuNodeWavePlan cpu_node_wave_check(MPI_Comm node,int n,const hb2st_rot<T>* records,std::size_t record_count,const CpuReplayPolicy& policy)
{
    replay_collective_local(node,[&]{if(n<0||(record_count&&!records)||policy.threads<1)throw std::invalid_argument("CPU shared replay: invalid dimensions or record pointer");});
    CpuNodeWavePlan p{n,record_count?records[0].j0:1,policy.threads,record_count,policy};int signature[6]={p.n,p.kd,policy.wave_mode,policy.row_block,policy.level_block,policy.waves?1:0},minimum[6],maximum[6];
    MPI_CHECK(MPI_Allreduce(signature,minimum,6,MPI_INT,MPI_MIN,node));MPI_CHECK(MPI_Allreduce(signature,maximum,6,MPI_INT,MPI_MAX,node));
    for(int k=0;k<6;++k)if(minimum[k]!=maximum[k])throw std::invalid_argument("CPU shared replay: inconsistent schedule metadata");
    const unsigned long long count=record_count;unsigned long long lo=0,hi=0;MPI_CHECK(MPI_Allreduce(&count,&lo,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,node));MPI_CHECK(MPI_Allreduce(&count,&hi,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,node));if(lo!=hi)throw std::invalid_argument("CPU shared replay: inconsistent record counts");
    std::size_t index=0,table=0,bytes=0;node_wave_windows<T>(p,&index,&table,&bytes);
    const auto limit=std::size_t(std::numeric_limits<MPI_Aint>::max());if(index>limit||table>limit||bytes>limit)throw std::overflow_error("CPU replay: shared window size overflow");return p;
}
template<class T>void cpu_node_wave_workspace_size(const CpuNodeWavePlan& p,std::size_t* host_bytes)
{*host_bytes=p.kd>p.n-1?0:wave_schedule_bytes<T>(p.n,p.kd,p.records,p.policy,false);}
template<class T>CpuNodeWaveResources<T>::CpuNodeWaveResources(MPI_Comm comm,const CpuNodeWavePlan& p):node(comm)
{
    std::size_t index_bytes=0,table_bytes=0,record_bytes=0;if(!node_wave_windows<T>(p,&index_bytes,&table_bytes,&record_bytes))return;
    MPI_CHECK(MPI_Comm_set_errhandler(node,MPI_ERRORS_RETURN));
    if(!node_alloc_shared(node,index_bytes,index))throw std::bad_alloc();
    if(!node_alloc_shared(node,table_bytes,table)){release();throw std::bad_alloc();}
    if(!node_alloc_shared(node,record_bytes,records)){release();throw std::bad_alloc();}
}
template<class T>void CpuNodeWaveResources<T>::release_plan()
{if(index.win!=MPI_WIN_NULL)MPI_CHECK(MPI_Win_free(&index.win));index.base=nullptr;}
template<class T>void CpuNodeWaveResources<T>::release()
{release_plan();if(table.win!=MPI_WIN_NULL)MPI_CHECK(MPI_Win_free(&table.win));if(records.win!=MPI_WIN_NULL)MPI_CHECK(MPI_Win_free(&records.win));table.base=records.base=nullptr;}
template<class T>bool cpu_node_wave_fill_run(const CpuNodeWavePlan& p,CpuNodeWaveResources<T>& resources,const hb2st_rot<T>* rots,
    WaveShared<T>& out,void* host_work,std::size_t host_bytes)
{
    out={};if(p.kd>p.n-1)return false;
    char* w=static_cast<char*>(host_work);auto schedule=take_wave_schedule<T>(w,p.n,p.kd,p.records,p.policy,false);if(!schedule.capacity)return false;
    assert(std::size_t(w-static_cast<char*>(host_work))<=host_bytes);(void)host_bytes;
    int rank=0,ranks=1;MPI_CHECK(MPI_Comm_rank(resources.node,&rank));MPI_CHECK(MPI_Comm_size(resources.node,&ranks));
    auto* wid=static_cast<int*>(resources.index.base);auto* pid=reinterpret_cast<std::uint8_t*>(wid+p.records);
    auto* woff=static_cast<std::int64_t*>(resources.table.base);auto* gjlo=reinterpret_cast<int*>(woff+p.records+1);auto* gcnt=gjlo+p.records;auto* order=gcnt+p.records;
    long long header[3]={0,0,0};replay_collective_local(resources.node,[&]{if(rank==0){
        header[0]=plan_wave_schedule<T>(p.n,rots,p.records,wid,pid,schedule,p.policy)?1:0;
        header[1]=static_cast<long long>(schedule.nw);header[2]=schedule.ranked?1:0;
        if(header[0]){std::memcpy(woff,schedule.woff,(schedule.nw+1)*sizeof(*woff));std::memcpy(gjlo,schedule.gjlo,schedule.nw*sizeof(int));std::memcpy(gcnt,schedule.gcnt,schedule.nw*sizeof(int));if(header[2])std::memcpy(order,schedule.rank,schedule.nw*sizeof(int));}
    }});
    MPI_CHECK(MPI_Bcast(header,3,MPI_LONG_LONG,0,resources.node));if(!header[0])return false;MPI_CHECK(MPI_Barrier(resources.node));
    const auto first=p.records/std::size_t(ranks)*rank+std::min(std::size_t(rank),p.records%ranks),end=first+p.records/ranks+(std::size_t(rank)<p.records%ranks?1:0);
    replay_collective_local(resources.node,[&]{regroup_records<T>(rots,first,end,wid,pid,header[2]?order:nullptr,woff,static_cast<hb2st_rot<T>*>(resources.records.base),p.threads);});MPI_CHECK(MPI_Barrier(resources.node));
    out.ok=true;out.nw=std::size_t(header[1]);out.nrots=p.records;out.woff=woff;out.gjlo=gjlo;out.gcnt=gcnt;out.wrec=static_cast<const hb2st_rot<T>*>(resources.records.base);return true;
}
template <typename T>bool cpu_build_node_wave_schedule(MPI_Comm node,int n,const hb2st_rot<T>* rots,std::size_t nrots,int threads,WaveShared<T>& out)
{
    out={};const auto p=cpu_node_wave_check<T>(node,n,rots,nrots,replay_policy_for_threads(threads));if(!p.policy.waves||!nrots||n<4)return false;
    std::size_t host_bytes=0;std::vector<char> host_work;
    replay_collective_local(node,[&]{cpu_node_wave_workspace_size<T>(p,&host_bytes);host_work.resize(host_bytes);});if(!host_bytes)return false;
    std::optional<CpuNodeWaveResources<T>> prepared;try{prepared.emplace(node,p);}catch(const std::bad_alloc&){return false;}auto& resources=*prepared;
    try{if(!cpu_node_wave_fill_run<T>(p,resources,rots,out,host_work.data(),host_bytes)){resources.release();return false;}
        resources.release_plan();out.tab=resources.table;out.rec=resources.records;resources.table={};resources.records={};return true;
    }catch(...){resources.release();throw;}
}

template<class T>void cpu_scatter_z_check(const T* recv,const int* rd,const int* share,int npr,int nlc,const int* columns,int rows,T*Z,int ld,int threads)
{
    if(npr<1||nlc<0||rows<0||ld<std::max(rows,1)||threads<1||!rd||!share||(nlc&&(!columns||!recv))||(nlc&&rows&&!Z)||share[0]!=0||share[npr]!=nlc)throw std::invalid_argument("CPU scatter: invalid dimensions or pointers");
    for(int p=0;p<npr;++p)if(rd[p]<0||share[p]>share[p+1])throw std::invalid_argument("CPU scatter: invalid displacements");for(int j=0;j<nlc;++j)if(columns[j]<0)throw std::invalid_argument("CPU scatter: invalid column map");
}
template<class T>void cpu_scatter_z(const T* recv,const int* rd,const int* share,int npr,int nlc,const int* columns,int rows,T*Z,int ld,int threads)
{cpu_scatter_z_check(recv,rd,share,npr,nlc,columns,rows,Z,ld,threads);cpu_scatter_z_run(recv,rd,share,npr,nlc,columns,rows,Z,ld,threads);}
template <typename T>
void cpu_scatter_z_run(const T* recvbuf, const int* rd, const int* share_lo, int npr, int nlc,
                   const int* col_lj, int m_loc_rows, T* Z, int lld, int nthreads)
{
    (void)nlc;
    constexpr int KB = 16;   // columns per work item: 16 write streams, each contiguous in li
    int nitems=0;for(int pr=0;pr<npr;++pr)nitems+=((share_lo[pr+1]-share_lo[pr])/KB+((share_lo[pr+1]-share_lo[pr])%KB!=0));
    (void)nthreads;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(std::max(1, nthreads))
#endif
    for (int q = 0; q < nitems; ++q) {
        int pr=0,item=q;for(;pr<npr;++pr){const int count=((share_lo[pr+1]-share_lo[pr])/KB+((share_lo[pr+1]-share_lo[pr])%KB!=0));if(item<count)break;item-=count;}
        const int lo=share_lo[pr],w=share_lo[pr+1]-lo,k=lo+item*KB,kn=std::min(KB,w-item*KB);
        const T* src=recvbuf+rd[pr]+(k-lo);
        for (int li = 0; li < m_loc_rows; ++li) {
            const T* row = src + static_cast<std::size_t>(li) * w;
            for (int kk = 0; kk < kn; ++kk)
                Z[static_cast<std::size_t>(li) + static_cast<std::size_t>(col_lj[k + kk]) * lld] = row[kk];
        }
    }
}

namespace {
// The tiled block: ceil(ncs / tile) tiles of n x tile reals (two planes for complex), tile after tile.
template<class T>std::size_t replay_tile_stride(const Hb2stCpuReplayPlan& p)
{return std::size_t(p.n)*std::size_t(p.tile)*(k_real<T>?1:2);}
template<class T>std::size_t replay_block_count(const Hb2stCpuReplayPlan& p)
{return std::max(std::size_t(1),replay_tile_stride<T>(p)*std::size_t((std::int64_t(p.columns.ncs)+p.tile-1)/p.tile));}
// Entries of the panel buffers. Input: this rank's rows of every rank's panel columns (send) and all rows of my
// panel columns (receive). Output: all rows of my panel columns (send) and my rows of my process column's panel
// columns (receive).
inline void replay_input_counts(const Hb2stCpuReplayPlan& p,std::size_t* send,std::size_t* recv)
{
    const auto& m=p.columns;long long columns=0;for(int rank=0;rank<m.P;++rank)columns+=std::min(m.width,m.column_size(rank));
    *send=std::max(std::size_t(1),std::size_t(replay_count(static_cast<long long>(m.nrl)*columns)));
    *recv=std::max(std::size_t(1),std::size_t(replay_count(static_cast<long long>(p.n)*std::min(m.width,m.ncs))));
}
inline void replay_output_counts(const Hb2stCpuReplayPlan& p,std::size_t* send,std::size_t* recv)
{
    const auto& m=p.columns;
    *send=std::max(std::size_t(1),std::size_t(replay_count(static_cast<long long>(p.n)*std::min(m.width,m.ncs))));
    *recv=std::max(std::size_t(1),std::size_t(replay_count(static_cast<long long>(m.outrows)*std::min(m.width,num_loc(p.n,m.nb,m.pc,m.csrc,m.npc)))));
}
inline std::size_t replay_record_count(const Hb2stCpuReplayPlan& p){return std::size_t(std::min(p.total,262144LL));}
// The front of the buffer, the same for both runs: the layout and the tiled block.
template<class T>std::size_t replay_front_bytes(const Hb2stCpuReplayPlan& p)
{return cpu_replay_layout_bytes(p.columns)+tile_alignment+padded_bytes<typename real_type<T>::type>(replay_block_count<T>(p));}
template<class T>typename real_type<T>::type* replay_front(const Hb2stCpuReplayPlan& p,char*& cursor,CpuReplayLayout& map)
{
    map=cpu_replay_layout(p.columns,cursor);cursor=align_tiles(cursor);
    return take<typename real_type<T>::type>(cursor,replay_block_count<T>(p));
}

template<class T,int W>void replay_rows_input(const DdlaHandle_t& h,const Hb2stCpuReplayPlan& p,
    const typename real_type<T>::type* Y,const T* phases,void* work,std::size_t bytes)
{
    using R=typename real_type<T>::type;constexpr int planes=k_real<T>?1:2;
    char* const base=static_cast<char*>(work);char* w=base;CpuReplayLayout map;R* const data=replay_front<T>(p,w,map);
    char* const input_work=w;
    std::size_t nsend=0,nrecv=0;R* send=nullptr;R* recv=nullptr;
    if(!p.input_2d.enabled){replay_input_counts(p,&nsend,&nrecv);send=take<R>(w,nsend);recv=take<R>(w,nrecv);}
    assert(std::size_t(w-base)<=bytes);(void)bytes;
    const int n=p.n,ldy=p.ldy,threads=p.policy.threads;const std::size_t stride=replay_tile_stride<T>(p),count=replay_block_count<T>(p);
    host_parallel_for((count*sizeof(R)+4095)/4096,256,threads,[&](long long lo,long long hi){const auto begin=std::size_t(lo)*4096,end=std::min(count*sizeof(R),std::size_t(hi)*4096);if(begin<end)std::memset(reinterpret_cast<std::byte*>(data)+begin,0,end-begin);});
    // Each input value is put directly in its final replay tile. In complex
    // arithmetic preserve the original real/imaginary phase expressions,
    // including signed zeros; no BLAS or complex-expression reordering.
    auto put=[&](int col,int row,R value) {
        R* x=data+static_cast<std::size_t>(col/W)*stride+static_cast<std::size_t>(row)*W*planes+col%W;
        if constexpr(k_real<T>) *x=value;
        else if(row==0 || p.phase_count==0) {*x=value;x[W]=R(0);}
        else {
            const R pr=phases[row-1].real(),pi=phases[row-1].imag(),zero=R(0);
            *x=value*pr-zero*pi;x[W]=value*pi+zero*pr;
        }
    };
    if(p.input_2d.enabled) {
        eig_2d_input_cpu_run<R>(h,p.input_2d,Y,
            [&](int q){return map.column_count(q);},[&](int q,int k){return map.column(q)[k];},
            [&](int first,int columns,const R* panel,int ld){
                host_parallel_for(columns,4,threads,[&](long long lo,long long hi){
                    for(long long j=lo;j<hi;++j)for(int i=0;i<n;++i)put(first+int(j),i,panel[i+std::size_t(j)*ld]);
                });
            },input_work,bytes-std::size_t(input_work-base));
    } else if(map.P==1) {
        replay_collective_local(h->comm,[&] {
            host_parallel_for(map.ncs,4,threads,[&](long long lo,long long hi) {
                for(long long j=lo;j<hi;++j)
                    for(int i=0;i<n;++i) put(static_cast<int>(j),i,Y[i+static_cast<std::size_t>(j)*ldy]);
            });
        });
    } else {
        for(int first=0;first<map.maxcols;first+=map.width) {
            replay_collective_local(h->comm,[&] {
                map.input(first);
                for(int q=0;q<map.P;++q) {
                    const int count=map.count(q,first);const auto columns=map.column(q);
                    host_parallel_for(count,4,threads,[&](long long lo,long long hi) {
                        for(long long k=lo;k<hi;++k) {
                            const int j=columns[first+static_cast<int>(k)];
                            if(map.nrl) std::memcpy(send+map.sd[q]+static_cast<std::size_t>(k)*map.nrl,
                                Y+static_cast<std::size_t>(j)*ldy,static_cast<std::size_t>(map.nrl)*sizeof(R));
                        }
                    });
                }
            });
            MPI_CHECK(MPI_Alltoallv(send,map.sc,map.sd,mpi_datatype<R>(),
                recv,map.rc,map.rd,mpi_datatype<R>(),h->comm));
            replay_collective_local(h->comm,[&] {
                host_parallel_for(map.count(map.me,first),4,threads,[&](long long lo,long long hi) {
                    for(long long k=lo;k<hi;++k)
                        for(int q=0;q<map.P;++q) {
                            const int nr=map.rows.count(q),r0=map.rows.lo(q);
                            const R* src=recv+map.rd[q]+static_cast<std::size_t>(k)*nr;
                            for(int i=0;i<nr;++i) put(first+static_cast<int>(k),r0+i,src[i]);
                        }
                });
            });
        }
    }
}
template<class T,int W>void replay_rows_apply(const DdlaHandle_t& h,const Hb2stCpuReplayPlan& p,
    const DistributedHistory<T>& history,T* Z,void* work,std::size_t bytes)
{
    using R=typename real_type<T>::type;constexpr int planes=k_real<T>?1:2;
    char* const base=static_cast<char*>(work);char* w=base;CpuReplayLayout map;R* const data=replay_front<T>(p,w,map);
    // One region behind the block: a history chunk and its wave schedule, and after the chunk loop the output panel.
    char* chunk_end=w;const std::size_t record_count=replay_record_count(p);hb2st_rot<T>* const records=take<hb2st_rot<T>>(chunk_end,record_count);
    WaveSchedule<T> schedule;if(map.ncs>=16)schedule=take_wave_schedule<T>(chunk_end,p.n,p.kd,record_count,p.policy,true);
    char* output_end=w;std::size_t nsend=0,nrecv=0;replay_output_counts(p,&nsend,&nrecv);T* const send=take<T>(output_end,nsend);T* const recv=take<T>(output_end,nrecv);
    assert(std::size_t(std::max(chunk_end,output_end)-base)<=bytes);(void)bytes;
    const int n=p.n,threads=p.policy.threads,tiles=int((std::int64_t(map.ncs)+W-1)/W);const std::size_t stride=replay_tile_stride<T>(p);
    for(long long chunk=history.chunks();chunk-->0;) {
        DistributedHistoryHeader header;
        history.fetch(chunk,h->comm,records,&header);
        replay_collective_local(h->comm,[&] {
            if(map.ncs>=16)build_wave_schedule(schedule,n,records,header.count,threads,p.policy,&header.cursor,header.offset);
            const auto view=schedule.view();
            host_parallel_for(tiles,1,threads,[&](long long lo,long long hi) {
                for(long long t=lo;t<hi;++t) {
                    R* x=data+t*stride;
                    if(schedule.ok) wave_tile<T,W>(x,view);else rot_tile<T,W>(x,records,header.count,n);
                }
            });
        });
    }
    auto get=[&](int col,int row)->T {
        const R* x=data+static_cast<std::size_t>(col/W)*stride+static_cast<std::size_t>(row)*W*planes+col%W;
        if constexpr(k_real<T>) return *x;else return T(*x,x[W]);
    };
    if(map.npr==1) {
        const auto columns=map.column(map.me);
        replay_collective_local(h->comm,[&] {
            host_parallel_for(map.ncs,4,threads,[&](long long lo,long long hi) {
                for(long long k=lo;k<hi;++k) {
                    const int j=columns[k],lj=(j/map.nb/map.npc)*map.nb+j%map.nb;
                    for(int i=0;i<n;++i) Z[i+static_cast<std::size_t>(lj)*map.lld]=get(static_cast<int>(k),i);
                }
            });
        });
    } else {
        for(int first=0;first<map.maxoutcols;first+=map.width) {
            const int mine=map.output_count(map.me,first),start=map.output_first(map.me,first);
            replay_collective_local(h->comm,[&] {
                map.output(first);
                host_parallel_for(n,32,threads,[&](long long lo,long long hi) {
                    for(long long i=lo;i<hi;++i) {
                        T* dst=send+map.osd[map.row_owner[i]]+static_cast<std::size_t>(map.row_local[i])*mine;
                        for(int k=0;k<mine;++k) dst[k]=get(start+k,static_cast<int>(i));
                    }
                });
            });
            MPI_CHECK(MPI_Alltoallv(send,map.osc,map.osd,mpi_datatype<T>(),
                recv,map.orc,map.ord,mpi_datatype<T>(),h->col_comm));
            replay_collective_local(h->comm,[&] {
                for(int p=0;p<map.npr;++p) {
                    const int q=map.col_ranks[p],count=map.output_count(q,first),start=map.output_first(q,first);const auto columns=map.column(q);
                    host_parallel_for(count,4,threads,[&](long long lo,long long hi) {
                        for(long long k=lo;k<hi;++k) {
                            const int j=columns[start+static_cast<int>(k)],lj=(j/map.nb/map.npc)*map.nb+j%map.nb;
                            for(int i=0;i<map.outrows;++i)
                                Z[i+static_cast<std::size_t>(lj)*map.lld]=recv[map.ord[p]+static_cast<std::size_t>(i)*count+k];
                        }
                    });
                }
            });
        }
    }
}
} // namespace

template<class T>Hb2stCpuReplayPlan hb2st_cpu_replay_geometry(const DdlaHandle_t& h,int n,int kd,long long total,int ldy,
    std::size_t phase_count,const int* desc,const CpuReplayPolicy& policy,int panel)
{
    Hb2stCpuReplayPlan p;p.columns=cpu_replay_plan(h,n,desc,panel);p.n=n;p.kd=kd;p.total=total;p.ldy=ldy;p.phase_count=phase_count;p.policy=policy;p.tile=replay_tile_geometry<T>(p.columns.ncs,policy);return p;
}
template<class T>Hb2stCpuReplayPlan hb2st_cpu_replay_check(const DdlaHandle_t& h,int n,const typename real_type<T>::type* Y,int ldy,
    const DistributedHistory<T>& history,const T* phases,std::size_t phase_count,T* Z,const int* desc,int panel)
{
    using R=typename real_type<T>::type;if(!h||ddlaGetBackend(h)!=DdlaBackend::CPU)throw std::invalid_argument("CPU replay: CPU handle required");
    replay_collective_local(h->comm,[&]{check_desc(desc,h);if(n<0||n>desc[DDLA_M_]||n>desc[DDLA_N_]||panel<0)throw std::invalid_argument("CPU replay: invalid order or panel width");});
    int signature[6]={n,desc[DDLA_MB_],desc[DDLA_NB_],desc[DDLA_RSRC_],desc[DDLA_CSRC_],panel},minimum[6],maximum[6];
    MPI_CHECK(MPI_Allreduce(signature,minimum,6,MPI_INT,MPI_MIN,h->comm));MPI_CHECK(MPI_Allreduce(signature,maximum,6,MPI_INT,MPI_MAX,h->comm));
    for(int i=0;i<6;++i)if(minimum[i]!=maximum[i])throw std::invalid_argument("CPU replay: inconsistent shared metadata");
    const auto columns=cpu_replay_plan(h,n,desc,panel);
    replay_collective_local(h->comm,[&]{
        const std::size_t nc=num_loc(n,columns.nb,columns.pc,columns.csrc,columns.npc);
        if(ldy<std::max(columns.nrl,1)||(n>0&&columns.nrl>0&&!Y)||(columns.outrows>0&&nc>0&&!Z)||!history.ready||history.n!=n||history.rank!=columns.me||history.ranks!=columns.P)throw std::invalid_argument("CPU replay: incompatible input/history/output");
        if(phase_count&&!phases)throw std::invalid_argument("CPU replay: null phases");
        if constexpr(!k_real<T>)if(phase_count&&phase_count<std::size_t(std::max(n-1,0)))throw std::invalid_argument("CPU replay: incomplete phases");
        if(std::size_t(ldy)>std::numeric_limits<std::size_t>::max()/std::size_t(std::max(n,1))/sizeof(R)||
           std::size_t(columns.lld)>std::numeric_limits<std::size_t>::max()/std::max<std::size_t>(nc,1)/sizeof(T))throw std::overflow_error("CPU replay: input/output size overflow");
        if(Y&&Z&&n>0&&columns.outrows>0&&nc>0){
            const std::size_t ay=std::size_t(ldy)*std::size_t(n)*sizeof(R),bz=((nc-1)*std::size_t(columns.lld)+std::size_t(columns.outrows))*sizeof(T);
            if(memory_overlaps(Y,ay,Z,bz))throw std::invalid_argument("CPU replay: input/output overlap");
        }
    });
    const long long metadata[2]={history.kd,history.total};long long history_min[2],history_max[2];MPI_CHECK(MPI_Allreduce(metadata,history_min,2,MPI_LONG_LONG,MPI_MIN,h->comm));MPI_CHECK(MPI_Allreduce(metadata,history_max,2,MPI_LONG_LONG,MPI_MAX,h->comm));
    if(history_min[0]!=history_max[0]||history_min[1]!=history_max[1])throw std::invalid_argument("CPU replay: inconsistent history geometry");
    return hb2st_cpu_replay_geometry<T>(h,n,history.kd,history.total,ldy,phase_count,desc,cpu_replay_policy(h->comm),panel);
}
template<class T>void hb2st_cpu_replay_workspace_size(const DdlaHandle_t& h,const Hb2stCpuReplayPlan& p,std::size_t* backend_bytes)
{
    using R=typename real_type<T>::type;*backend_bytes=0;if(!p.n)return;
    std::size_t input_send=0,input_recv=0,output_send=0,output_recv=0;replay_input_counts(p,&input_send,&input_recv);replay_output_counts(p,&output_send,&output_recv);
    const std::size_t records=replay_record_count(p);std::size_t chunk=padded_bytes<hb2st_rot<T>>(records);
    if(p.columns.ncs>=16)chunk+=wave_schedule_bytes<T>(p.n,p.kd,records,p.policy,true);
    const auto input=p.input_2d.enabled?eig_2d_input_cpu_bytes<R>(h,p.input_2d):padded_bytes<R>(input_send)+padded_bytes<R>(input_recv);
    *backend_bytes=replay_front_bytes<T>(p)+std::max({input,chunk,
        padded_bytes<T>(output_send)+padded_bytes<T>(output_recv)});
}
template<class T>void hb2st_cpu_replay_input_workspace_size(const DdlaHandle_t& h,const Hb2stCpuReplayPlan& p,std::size_t* backend_bytes)
{
    using R=typename real_type<T>::type;*backend_bytes=0;if(!p.n)return;
    std::size_t input_send=0,input_recv=0;replay_input_counts(p,&input_send,&input_recv);
    *backend_bytes=replay_front_bytes<T>(p)+(p.input_2d.enabled?eig_2d_input_cpu_bytes<R>(h,p.input_2d):padded_bytes<R>(input_send)+padded_bytes<R>(input_recv));
}
template<class T>void hb2st_cpu_replay_input_run(const DdlaHandle_t& h,const Hb2stCpuReplayPlan& p,const typename real_type<T>::type* Y,
    const T* phases,void* backend_work,std::size_t backend_bytes)
{
    if(!p.n)return;
#define INPUT(W) replay_rows_input<T,W>(h,p,Y,phases,backend_work,backend_bytes)
    switch(p.tile){case 8:INPUT(8);break;case 16:INPUT(16);break;case 32:INPUT(32);break;default:INPUT(64);break;}
#undef INPUT
}
template<class T>void hb2st_cpu_replay_apply_run(const DdlaHandle_t& h,const Hb2stCpuReplayPlan& p,const DistributedHistory<T>& history,
    T* Z,int& info,void* backend_work,std::size_t backend_bytes)
{
    info=0;if(!p.n)return;
#define APPLY(W) replay_rows_apply<T,W>(h,p,history,Z,backend_work,backend_bytes)
    switch(p.tile){case 8:APPLY(8);break;case 16:APPLY(16);break;case 32:APPLY(32);break;default:APPLY(64);break;}
#undef APPLY
}
template<class T>void hb2st_cpu_replay_run(const DdlaHandle_t& h,const Hb2stCpuReplayPlan& p,const DistributedHistory<T>& history,
    const T* phases,const typename real_type<T>::type* Y,T* Z,int& info,void* backend_work,std::size_t backend_bytes)
{hb2st_cpu_replay_input_run<T>(h,p,Y,phases,backend_work,backend_bytes);hb2st_cpu_replay_apply_run<T>(h,p,history,Z,info,backend_work,backend_bytes);}
template<class T>void hb2st_replay_rows_cpu(const DdlaHandle_t& h,int n,typename real_type<T>::type*& Y,int ldy,
    const DistributedHistory<T>& history,const std::vector<T>& phases,T* Z,const int* desc,int& info,int panel)
{
    const auto p=hb2st_cpu_replay_check<T>(h,n,Y,ldy,history,phases.data(),phases.size(),Z,desc,panel);
    std::size_t backend_bytes=0;ddla::vector<char> backend_work(h);
    replay_collective_local(h->comm,[&]{hb2st_cpu_replay_workspace_size<T>(h,p,&backend_bytes);backend_work.resize(backend_bytes);});
    hb2st_cpu_replay_input_run<T>(h,p,Y,phases.data(),backend_work.data(),backend_bytes);
    // No later operation reads Y, and all input communication is complete.
    // The caller's pointer-reference guard sees null even if replay fails.
    replay_collective_local(h->comm,[&]{DDLA_CHECK(ddlaFree(h,Y));Y=nullptr;});
    hb2st_cpu_replay_apply_run<T>(h,p,history,Z,info,backend_work.data(),backend_bytes);
}

#define DDLA_INST_ROWS_CPU(T) \
    template Hb2stCpuReplayPlan hb2st_cpu_replay_geometry<T>(const DdlaHandle_t&,int,int,long long,int,std::size_t,const int*,const CpuReplayPolicy&,int); \
    template Hb2stCpuReplayPlan hb2st_cpu_replay_check<T>(const DdlaHandle_t&,int,const real_type<T>::type*,int,const DistributedHistory<T>&,const T*,std::size_t,T*,const int*,int); \
    template void hb2st_cpu_replay_workspace_size<T>(const DdlaHandle_t&,const Hb2stCpuReplayPlan&,std::size_t*); \
    template void hb2st_cpu_replay_input_workspace_size<T>(const DdlaHandle_t&,const Hb2stCpuReplayPlan&,std::size_t*); \
    template void hb2st_cpu_replay_input_run<T>(const DdlaHandle_t&,const Hb2stCpuReplayPlan&,const real_type<T>::type*,const T*,void*,std::size_t); \
    template void hb2st_cpu_replay_apply_run<T>(const DdlaHandle_t&,const Hb2stCpuReplayPlan&,const DistributedHistory<T>&,T*,int&,void*,std::size_t); \
    template void hb2st_cpu_replay_run<T>(const DdlaHandle_t&,const Hb2stCpuReplayPlan&,const DistributedHistory<T>&,const T*,const real_type<T>::type*,T*,int&,void*,std::size_t); \
    template void hb2st_replay_rows_cpu<T>(const DdlaHandle_t&,int,real_type<T>::type*&,int, \
        const DistributedHistory<T>&,const std::vector<T>&,T*,const int*,int&,int);
DDLA_INST_ROWS_CPU(float)
DDLA_INST_ROWS_CPU(double)
DDLA_INST_ROWS_CPU(std::complex<float>)
DDLA_INST_ROWS_CPU(std::complex<double>)
#undef DDLA_INST_ROWS_CPU

#define DDLA_INST_CPU_KERNELS(T)                                                                              \
    template CpuNodeWavePlan cpu_node_wave_check<T>(MPI_Comm,int,const hb2st_rot<T>*,std::size_t,const CpuReplayPolicy&); \
    template void cpu_node_wave_workspace_size<T>(const CpuNodeWavePlan&,std::size_t*); \
    template struct CpuNodeWaveResources<T>; \
    template bool cpu_node_wave_fill_run<T>(const CpuNodeWavePlan&,CpuNodeWaveResources<T>&,const hb2st_rot<T>*,WaveShared<T>&,void*,std::size_t); \
    template CpuReplayPackPlan cpu_replay_pack_geometry<T>(int,int,int,int,int,std::size_t,CpuReplaySource,const CpuReplayPolicy&); \
    template void cpu_replay_pack_check<T>(const CpuReplayPackPlan&,const real_type<T>::type*,std::size_t,int,const int*,const hb2st_rot<T>*,std::size_t,const WaveShared<T>*,const DistributedHistory<T>*,const T*,const int*,T*); \
    template void cpu_replay_pack_workspace_size<T>(const CpuReplayPackPlan&,std::size_t*); \
    template void cpu_replay_pack_run<T>(const CpuReplayPackPlan&,const real_type<T>::type*,int,const int*,const hb2st_rot<T>*,const WaveShared<T>*,const DistributedHistory<T>*,MPI_Comm,const T*,const int*,T*,void*,std::size_t); \
    template void cpu_replay_pack_distributed<T>(const real_type<T>::type*,std::size_t,int,const int*,int,int, \
        const DistributedHistory<T>&,MPI_Comm,const T*,int,int,const int*,T*,int); \
    template void cpu_replay_pack<T>(const real_type<T>::type*,std::size_t,int,const int*,int,int,                    \
                                     const hb2st_rot<T>*,std::size_t,const T*,int,int,const int*,T*,int);    \
    template void cpu_scatter_z_check<T>(const T*,const int*,const int*,int,int,const int*,int,T*,int,int); \
    template void cpu_scatter_z_run<T>(const T*,const int*,const int*,int,int,const int*,int,T*,int,int); \
    template void cpu_scatter_z<T>(const T*, const int*, const int*, int, int, const int*, int, T*, int, int);          \
    template void cpu_replay_pack_shared<T>(const real_type<T>::type*,std::size_t,int,const int*,int,int,              \
                                            const WaveShared<T>&,const T*,int,int,const int*,T*,int);    \
    template bool cpu_build_node_wave_schedule<T>(MPI_Comm, int, const hb2st_rot<T>*, std::size_t, int,       \
                                                  WaveShared<T>&);

DDLA_INST_CPU_KERNELS(float)
DDLA_INST_CPU_KERNELS(double)
DDLA_INST_CPU_KERNELS(std::complex<float>)
DDLA_INST_CPU_KERNELS(std::complex<double>)

} // namespace detail
} // namespace ddla
