#ifndef DDLA_EIG_PARAMS_H
#define DDLA_EIG_PARAMS_H

#include <mpi.h>
#include <algorithm>
#include <ddla/ddla_handle_t.h>
#include <complex>
#include <type_traits>

namespace ddla {

// ---------------------------------------------------------------------------
// Tuning parameters for the two-stage Hermitian eigensolver (pheevd).
//
// Defaults follow ELPA2 practice; every value can be overridden by an
// environment variable (same pattern as DDLA_PGEMM_KPANEL in pgemm.cpp):
//
//   DDLA_EIG_NBW            band block size nbw (stage 1/2); ALWAYS divides nblk
//                           (a stage-1 panel is one block column).  GPU default: 16
//                           (real and complex) whenever nblk is a positive multiple of
//                           16 (nblk >= 16), otherwise nblk; the two-stage driver can
//                           jointly choose Householder/band64 for its measured
//                           full-vector profiles.  CPU default: eig_cpu_band(nblk), the
//                           divisor of nblk nearest the measured optimum of 48-64 (64
//                           for nblk = 64, 128, 256, ...; 50 for nblk = 100; nblk itself
//                           when it is below 32 or has no usable divisor).
//                           An override must satisfy 1 <= nbw <= nblk and
//                           nblk % nbw == 0 (the rule phe2hb/punmtr_he2hb
//                           enforce); anything else throws
//                           std::invalid_argument. On a multi-rank solve,
//                           pheevd may choose a smaller divisor to keep the
//                           gathered band bounded by the rank count.
//   DDLA_EIG_B2F_BLOCKING   stage-5 (band -> full) CWY blocking factor.
//                           Default 3 (blocking_factor * nbw = cwy_blocking).
//   DDLA_EIG_STRIPE_WIDTH   stage-4 stripe width (rows of the transposed
//                           eigenvector workspace per process column).
//                           Default 1024.
//   DDLA_EIG_DC_LEAF        pstedc (stage 3) leaf size: subproblems are
//                           halved until the largest is <= this, then solved
//                           by host ?STEDC.  Default 1024 (128 for the CPU
//                           eigenvalues-only tree); values < 2 fall back to
//                           the default.  Read on rank 0, broadcast.
//   DDLA_EIG_DC_KEEP_MB     CPU eigenvalues-only pstedc: MiB per rank of
//                           secular-root deltas a merge may keep; larger
//                           merges recompute them in a second ?LAED4 pass
//                           (bit-identical).  Default 64; 0 = always
//                           recompute.  Read on rank 0, broadcast.
//   DDLA_EIG_DC_SPLIT       CPU merges only: 1 (default) = the merge GEMM
//                           splits by the children's row support (the
//                           dlaed2/dlaed3 trick, ~half the flops); 0 = one
//                           dense GEMM (the phase-1 port, kept for A/B).
//                           Summation order differs between the two, so the
//                           value is read on rank 0 and broadcast.
//   DDLA_EIG_THREADS        Host workers per handle, fixed by ddlaSet. A positive value overrides
//                           OMP_NUM_THREADS. Without a positive override, each allowed CPU contributes
//                           1 / the node-local ranks sharing it; floor the sum, minimum one.
//                           Environment/affinity changes take effect after another ddlaSet.
//   DDLA_EIG_REPLAY_TILE    CPU stage-4 replay column-tile width, 8|16|32|64.  Default: a 256-byte row
//                           segment (32 real double, 16 complex double, 64 float).
//   DDLA_EIG_REPLAY_WAVE    CPU stage-4 replay order.  0 = step by step (the recorded order); 1 (default) = regroup the
//                           recorded rotations into dominoes (one bulge's consecutive steps, adjacent row pairs) applied as
//                           register-carried chains and, for n >= 4096, ordered into skewed cache tiles (dependency
//                           levels, bands of rows skewed by one wave length per level, bands descending inside a level
//                           block) so a tile of ~RB + s*LAM rows stays in L2: ~2x faster than step by step at n=10000,
//                           more at larger n, bit-identical; 2 = skewed tiles at any n (tests).  The grouping is rebuilt
//                           from (n, band width) and every record is checked against it (a stream that is not a chase
//                           stream falls back to the plain order).
//   DDLA_EIG_REPLAY_RB      skewed-tile band height in rows (default 512, >= 8).
//   DDLA_EIG_REPLAY_LAM     levels per skewed-tile block (default 32).
//   DDLA_EIG_REPLAY_PROF    1 = print the replay's schedule and kernel times (CPU stage 4).
//   DDLA_EIG_CPU_ACCUM      CPU stage-4 engine: auto (default) | 1 | 0.  1/auto: accumulate Q_band during the chase (hb2st_sink_host.cpp,
//                           on cache-blocked host tiles), then bounded panels of Q_band * (D*Y); 0: replay rank-owned
//                           history chunks after stage 3 (the options above).  auto accumulates on ONE node when every rank has
//                           >= max(4, chase threads + 2) threads and n >= 2048, and replays otherwise.  Collective decision, read at
//                           every pheevd('V') call.
//   DDLA_EIG_HOST_ACCUM_BLOCK / _SLOTS / _RB
//                           CPU accumulation schedule: records per block (default 524288, >= 4096), blocks in flight (default 3),
//                           rows per skewed band (default 512).  Test knobs: Z does not depend on them (bitwise).
//   DDLA_EIG_HOST_FINISH_PANEL  rows of Y per broadcast/GEMM panel of the CPU finish (default 512); the buffers beside Q_band and W are
//                           panel-sized.  The GEMM's summation order follows it.
//   DDLA_EIG_PROFILE=1 also prints "STAGE4 engine=..." (the decision and its reason) and one HOSTSINK line per rank.
//   DDLA_EIG_CHASE_TEAM_THREADS
//                           stage-2 team members per rank (cap 8).  Default 1 on GPU handles; on CPU handles
//                           min(threads per rank, 4 / ranks per node), i.e. a team of about 4 members.
//   DDLA_EIG_CHASE_TEAM_RANKS
//                           CPU handles: at most this many ranks of a node run stage-2 chase members (default
//                           max(1, 4 / threads-per-rank)); the node's other ranks wait in an end-of-chase barrier and
//                           read the result from the shared window.  A team larger than ~4 members is slower than the
//                           serial chase (40 members: 17 s vs ~5 s at n=10000).  GPU handles: every rank is a member.
//   DDLA_EIG_PROFILE        when set to 1, stage 2/4 print per-phase timing
//                           breakdowns from rank 0 (gather / conversion /
//                           schedule / kernel / pack / alltoallv / scatter).
//                           Timers are compiled in but guarded on this flag,
//                           so the default path is untouched.  Measurement
//                           only -- never leave on in production.
// ---------------------------------------------------------------------------
struct EigParams {
    int nbw;            // band block size (divides nblk)
    int b2f_blocking;   // stage-5 CWY blocking factor
    int stripe_width;   // stage-4 stripe width
};

// CPU band width for block size nblk: a divisor of nblk (a stage-1 panel is one block
// column).  The band chase costs O(n^2 * band) on a few threads while stages 1 and 5 lose
// BLAS-3 efficiency below ~32, so the total is flat between 32 and 64 and rises on both
// sides (n=4000, 8 cores, Householder chase, real / complex seconds: band 16 5.5 / 10.7,
// 32 4.6 / 9.3, 48 4.3 / 9.0, 64 4.4 / 9.1, 100 4.8 / 9.6, 128 5.1 / 10.2; the Givens
// chase used outside the Householder profile gains more from a narrow band).  Preference:
// the largest divisor in [32, 64], then the smallest in (64, 128], then the largest in
// [16, 32), then nblk itself.
inline int eig_cpu_band(int nblk) noexcept
{
    if (nblk < 32) return std::max(1, nblk);
    int in_range = 0, above = 0, below = 0;
    for (int d = 1; d <= nblk / d; ++d) {
        if (nblk % d != 0) continue;
        for (const int c : {d, nblk / d}) {
            if (c >= 32 && c <= 64) in_range = std::max(in_range, c);
            else if (c > 64 && c <= 128) above = above ? std::min(above, c) : c;
            else if (c >= 16 && c < 32) below = std::max(below, c);
        }
    }
    return in_range ? in_range : above ? above : below ? below : nblk;
}

/// Select the eigensolver parameters for a problem of block size @p nblk.
/// @p on_gpu selects the GPU nbw rule (16 when nblk % 16 == 0, else nblk) vs the CPU rule
/// eig_cpu_band(nblk); the scalar type does not matter for nbw (@p is_complex is kept for
/// the callers).  Throws std::invalid_argument when DDLA_EIG_NBW is set to a value that
/// does not divide nblk.
EigParams eig_select_params(bool is_complex, int nblk, bool on_gpu);

// The band chase can hold the whole compact band on a rank. For a distributed
// problem, bound its width to a rank's share instead of allowing O(n^2)
// replicated storage. Scalar dimensions are fixed-size base cases.
inline int eig_distributed_band_limit(int n, int ranks) noexcept
{
    return (ranks>1 && n>0) ? std::max(1,(n-1)/ranks) : std::max(1,n);
}

// Stage 1 also requires the band width to divide the matrix block size. Keep
// the requested width when safe, otherwise choose the largest valid divisor.
inline int eig_distributed_bandwidth(int n,int block,int band,int ranks) noexcept
{
    if(ranks<=1 || n<=0) return band;
    const int limit=std::min(band,eig_distributed_band_limit(n,ranks));
    if(band<=limit)return band;
    int best=1;
    for(int d=1;d<=block/d;++d) if(block%d==0) {
        if(d<=limit)best=std::max(best,d);
        const int other=block/d;
        if(other<=limit)best=std::max(best,other);
    }
    return best;
}

/// Whether per-phase timing output is requested (DDLA_EIG_PROFILE=1).
/// Read from the environment on every call -- callers need not cache it.
bool eig_profile_enabled();

namespace detail {
// CPU one-stage profile: a rank whose BLAS threads equal its host budget (0 = a BLAS that
// cannot be queried).  The public dispatcher still requires every rank to agree.
inline bool pheevd_one_stage_cpu_workers(int host_threads, int blas_threads) noexcept
{
    return host_threads >= 1 && (blas_threads == 0 || blas_threads == host_threads);
}
template <DdlaBackend Backend, class T>
inline bool pheevd_one_stage_shape(char job, int n, int block, int pr, int pc,
                                   char major) noexcept
{
    if constexpr (Backend == DdlaBackend::CPU) {
        // All vectors of a problem below the Householder two-stage threshold (2048), any
        // block size and grid, row-major.  2026-11-15, one node, seconds one-stage against
        // two-stage: n=2000 on 40-96 cores (2 and 4 ranks) 0.32-0.50 against 0.79-1.02 real
        // and 0.43-0.85 against 1.35-1.78 complex; on 8 cores (2, 4 and 8 ranks) the
        // one-stage path wins through n=2000 and loses from n=2500 (real 1.0-2.0 against
        // 1.0-1.5, complex 2.5-5.3 against 2.5-3.2).  One rank keeps one-stage below 4096
        // (2026-11-18, n=4000: 2.12 against 2.72 real and 4.39 against 4.90 complex on 40
        // threads; 3.24 against 3.80 and 8.11 against 8.46 on 8).
        const int limit = pr == 1 && pc == 1 ? 4096 : 2048;
        return job == 'V' && major == 'R' && block >= 1 && pr >= 1 && pc >= 1 && n >= 1 && n < limit;
    } else {
        // Matched eight-V100 measurements, NB64, one host/BLAS worker per rank.
        // Retain two-stage outside the measured range and for close real-values cases.
        if (block != 64 || major != 'R' || n > 16384) return false;
        if (pr != 2 || pc != 4) return false;
        if constexpr (std::is_same_v<T,float>)
            return job == 'V' && n >= 4096;
        else if constexpr (std::is_same_v<T,double>)
            return job == 'V' && n >= 3072;
        else if constexpr (std::is_same_v<T,std::complex<float>>)
            return (job == 'N' && n >= 1536) || (job == 'V' && n >= 1024);
        else if constexpr (std::is_same_v<T,std::complex<double>>)
            return (job == 'N' || job == 'V') && n >= 1024;
        else
            return false;
    }
}
} // namespace detail

} // namespace ddla

#endif // DDLA_EIG_PARAMS_H
