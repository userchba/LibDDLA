#ifndef DDLA_EIG_PARAMS_H
#define DDLA_EIG_PARAMS_H

#include <mpi.h>
#include <algorithm>

namespace ddla {

// ---------------------------------------------------------------------------
// Tuning parameters for the two-stage Hermitian eigensolver (pheevd).
//
// Defaults follow ELPA2 practice; every value can be overridden by an
// environment variable (same pattern as DDLA_PGEMM_KPANEL in pgemm.cpp):
//
//   DDLA_EIG_NBW            band block size nbw (stage 1/2); ALWAYS divides nblk
//                           (a stage-1 panel is one block column).  Default: 16
//                           for GPU runs (real and complex) whenever nblk
//                           is a positive multiple of 16 (nblk >= 16);
//                           otherwise nblk.  Tuned for nblk = 64, 128, ...;
//                           a small nblk runs with a small nbw (slow stage 1).
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
//   DDLA_EIG_THREADS        CPU handles: host threads per rank for the library's own loops (stage-4 replay,
//                           stage-3 divide and conquer).  Default: OMP_NUM_THREADS when set, otherwise the CPUs
//                           the process may run on divided by the ranks of its node.  Read once, collectively.
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

/// Select the eigensolver parameters for a problem of block size @p nblk.
/// @p on_gpu selects the GPU nbw rule (16 when nblk % 16 == 0) vs nbw = nblk; the scalar
/// type no longer matters for nbw (@p is_complex is kept for the callers).  Throws
/// std::invalid_argument when DDLA_EIG_NBW is set to a value that does not divide nblk.
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

} // namespace ddla

#endif // DDLA_EIG_PARAMS_H
