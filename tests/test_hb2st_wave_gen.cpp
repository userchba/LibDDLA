// ---------------------------------------------------------------------------
// Host-only test of the streaming wave schedule (src/hb2st_wave_schedule.h): Hb2stWaveGen must hand out exactly the launch batches
// build_hb2st_schedule builds from whole-stream tables -- the same waves, member record indices, gates and lowest records -- and
// count_records() must equal the offline stream length, for real and complex wave caps, over every small (n, kd) and the many-bulge
// thresholds.  The generator replaced the whole-stream host tables of the GPU accumulation sink; this test is what keeps the two
// builders from drifting apart.  No device, no matrix data; run under mpirun like every test (1 rank is enough).
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "Eigensolver/two_stage/hb2st_wave_schedule.h"

using namespace ddla::detail;

static long long g_configs = 0, g_failures = 0;

static bool check_config(int n, int kd, bool is_real)
{
    ++g_configs;
    const Hb2stSchedule sch = build_hb2st_schedule(n, kd, is_real);
    const int K = is_real ? 16 : 8;
    Hb2stWaveGen gen(n, kd, is_real);
    if (Hb2stWaveGen::count_records(n, kd) != sch.nrots) {
        std::printf("FAIL count_records n=%d kd=%d real=%d: %lld vs %lld\n", n, kd, static_cast<int>(is_real),
                    Hb2stWaveGen::count_records(n, kd), static_cast<long long>(sch.nrots));
        return false;
    }
    Hb2stWaveGen::Batch b;
    int nb = 0;
    long long waves = 0;
    while (gen.next(b)) {
        if (nb >= sch.nbatch) { std::printf("FAIL extra batch n=%d kd=%d real=%d\n", n, kd, static_cast<int>(is_real)); return false; }
        const int w0 = sch.wbatch_lo[static_cast<std::size_t>(nb)], w1 = sch.wbatch_hi[static_cast<std::size_t>(nb)];
        bool ok = (b.nw == w1 - w0) && (b.gate == sch.wbatch_gate[static_cast<std::size_t>(nb)]) &&
                  (b.lowest == static_cast<long long>(sch.midx[static_cast<std::size_t>(w0) * K]));
        for (int w = 0; ok && w < b.nw; ++w) {
            ok = ok && b.gjlo[static_cast<std::size_t>(w)] == sch.gjlo[static_cast<std::size_t>(w0 + w)] &&
                 b.gcnt[static_cast<std::size_t>(w)] == sch.gcnt[static_cast<std::size_t>(w0 + w)];
            for (int p = 0; ok && p < K; ++p)
                ok = ok && b.midx[static_cast<std::size_t>(w) * K + p] ==
                               static_cast<long long>(sch.midx[static_cast<std::size_t>(w0 + w) * K + p]);
        }
        if (!ok) {
            std::printf("FAIL batch %d n=%d kd=%d real=%d (waves %d vs %d, gate %lld vs %d)\n", nb, n, kd, static_cast<int>(is_real), b.nw,
                        w1 - w0, b.gate, sch.wbatch_gate[static_cast<std::size_t>(nb)]);
            return false;
        }
        waves += b.nw;
        ++nb;
    }
    if (nb != sch.nbatch || waves != sch.nw || gen.records() != sch.nrots) {
        std::printf("FAIL totals n=%d kd=%d real=%d: batches %d vs %d, waves %lld vs %d, records %lld vs %lld\n", n, kd, static_cast<int>(is_real), nb,
                    sch.nbatch, waves, sch.nw, gen.records(), static_cast<long long>(sch.nrots));
        return false;
    }
    return true;
}

int main()
{
    for (int n = 1; n <= 64; ++n)
        for (int kd = 1; kd <= 40; ++kd)
            for (int r = 0; r < 2; ++r)
                if (!check_config(n, kd, r == 1)) ++g_failures;
    // the many-bulge thresholds and production-like band widths
    const int ns[] = {97, 137, 256, 257, 560, 561, 575, 576, 577, 1000, 1500};
    const int kds[] = {2, 5, 8, 15, 16, 17, 32, 33, 64};
    for (int n : ns)
        for (int kd : kds)
            for (int r = 0; r < 2; ++r)
                if (!check_config(n, kd, r == 1)) ++g_failures;
    std::printf("test_hb2st_wave_gen: %lld configurations, %lld failures\n", g_configs, g_failures);
    if (g_failures != 0) { std::printf("test_hb2st_wave_gen FAILED\n"); return 1; }
    std::printf("test_hb2st_wave_gen passed\n");
    return 0;
}
