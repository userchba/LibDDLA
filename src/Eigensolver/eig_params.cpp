#include "eig_params.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <mpi.h>


namespace ddla {

namespace {

int env_int(const char* name, int fallback)
{
    const char* s = std::getenv(name);
    if (s == nullptr || *s == '\0') return fallback;
    const int v = std::atoi(s);
    return (v > 0) ? v : fallback;
}

} // namespace

EigParams eig_select_params(bool is_complex, int nblk, bool on_gpu)
{
    EigParams p;

    // nbw divides nblk (a stage-1 panel is ONE block column, so its communication stays
    // on one process column; phe2hb / punmtr_he2hb throw otherwise).  The default is
    // tuned for production block sizes (nblk = 64, 128, ...): GPU runs pin nbw = 16
    // whenever nblk is a positive multiple of 16, everything else uses nbw = nblk.  A
    // small nblk therefore runs with a small nbw (correct, but stage 1 is slow there).
    // Measured on
    // the public pheevd API (MATRIX=elpa, 2x2 V100, 2 CPUs/rank, medians of
    // 3, ELPA 2025.01 anchored in the same job at the SAME nblk):
    //   * nbw = nblk lost at large blocks (job 3569836, n=10000): nblk=64
    //     8.14 s (ELPA 10.39), nblk=128 13.55 s (ELPA 12.54); the chase
    //     scales linearly with nbw (2.1 / 3.4 / 5.5 / 11.0 s at 16..128).
    //   * every (nblk, nbw) pair times like (nbw, nbw) (job 3570301), so the
    //     real choice at nblk >= 32 is nbw = 16 or 32:
    //       n=10000: 7.27 s (16) vs 7.08 s (32) -- within 3%;
    //       n=5000:  1.62 s (16) vs 1.81-1.85 s (32) -- 16 wins by ~11%.
    //     nblk=32 repeats it (job 3571670, three interleaved rounds, ELPA
    //     at nblk=32 10.21 / 2.44 s): n=10000 7.26 vs 7.11 s, n=5000 1.62
    //     vs 1.79 s.  Since stage 1 stopped allocating per panel (2026-10-02,
    //     job 3574563) nbw = 16 also wins at n=10000: 7.00 vs 7.05 s.
    //     16 is the default: it wins where the gap is large, and it is the
    //     faster chase once the stage-4 accumulation (which bounds stages
    //     2-4 at n=10000) stops hiding the chase.
    // GPU complex runs follow the same rule since 2026-10-07 (they used the
    // ELPA2 complex floor 32, i.e. 32 at nblk 16/32 and 64 at nblk 64): with
    // the wave accumulation the complex chase dominates, and it scales with
    // nbw (stage 2 ~2.5 s at nbw=32 vs ~1.3 s at 16, n=5000).  Job 3597609
    // (complex double, medians, counterbalanced rounds, CHECK on, ELPA
    // 2025.01 GPU in the same job at the same nblk), nblk 16 / 32 / 64:
    //   n=5000:  nbw=16 2.48 / 2.47 / 2.47 s, old default 3.45 / 3.49 /
    //            5.76 s (ELPA 4.24 / 4.18 / 4.88 -- nblk=64 lost to ELPA);
    //   n=10000: nbw=16 9.02 / 8.97 / 8.94 s, old default 13.77 / 12.32 /
    //            21.67 s (ELPA 18.56 / 18.57 / 21.04).
    // Stages 1 and 5 grow (n=10000: 1.41 -> 1.77 s and 0.62 -> 0.90 s) but
    // the chase halves (8.7 -> 4.9 s).  Shipped default confirmed in-situ
    // (job 3599012, BASE = pre-round library at nbw=32 vs NEW at the default
    // 16, nblk=32, means of 2 rounds): n=10000 12.37 -> 8.74 s (ELPA z
    // 18.57, job 3597609, same node) and n=5000 3.50 -> 2.42 s (ELPA 4.18).
    // CPU runs use nbw = nblk (the former ELPA2 floors, 64 real / 32 complex, rounded up to
    // a multiple of nblk, could exceed nblk and are gone with the multi-block panels).
    (void)is_complex;
    const int blk = (nblk > 0) ? nblk : 1;
    int nbw = (on_gpu && nblk >= 16 && nblk % 16 == 0) ? 16 : blk;
    // An env override must divide nblk (1 <= nbw <= nblk): anything else throws on every
    // rank, instead of silently falling back to the default.
    const char* nbw_s = std::getenv("DDLA_EIG_NBW");
    if (nbw_s != nullptr && *nbw_s != '\0') {
        char* end = nullptr;
        const long v = std::strtol(nbw_s, &end, 10);
        if (end == nbw_s || *end != '\0' || v < 1 || v > blk || blk % v != 0)
            throw std::invalid_argument(std::string("DDLA_EIG_NBW=") + nbw_s
                                        + ": nbw must divide the block size nblk = "
                                        + std::to_string(blk) + " (1 <= nbw <= nblk)");
        nbw = static_cast<int>(v);
    }
    p.nbw = nbw;

    p.b2f_blocking = env_int("DDLA_EIG_B2F_BLOCKING", 3);

    // Single stripe-width default for both backends for now; the CPU path of
    // stage 4 may want a cache-friendlier override once it is measured.
    p.stripe_width = env_int("DDLA_EIG_STRIPE_WIDTH", 1024);

    return p;
}

bool eig_profile_enabled()
{
    return env_int("DDLA_EIG_PROFILE", 0) == 1;
}

} // namespace ddla
