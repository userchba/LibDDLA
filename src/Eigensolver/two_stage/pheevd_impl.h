#ifndef DDLA_PHEEVD_IMPL_H
#define DDLA_PHEEVD_IMPL_H
#include "Eigensolver/eig_workspace.h"
#include "phe2hb_impl.h"
#include "phb2st_impl.h"
#include "Eigensolver/tridiagonal_rows.h"
#include "hb2st_host_workspace.h"
#include "hb2st_cpu_replay_workspace.h"
#if DDLA_HAS_GPU
#include "hb2st_replay_gpu.h"
#endif
#include "hb2st_householder_cpu.h"
#include "hb2st_householder_cpu_rows.h"
#include "Eigensolver/pheevd_alias.h"
#include "Eigensolver/eig_memprof.h"
#include "plan_util.h"
#include <cstddef>
#include <cstdint>
#include <optional>
namespace ddla { namespace detail {
// CPU Householder band chase: one shared-memory node, at most 128 ranks and 64 workers per
// rank, problems of at least 2048, any block size whose band (a divisor of nb, see
// eig_cpu_band) is at least 16.  The chase has no rank limit of its own; 65, 72, 96, 100
// and 128 ranks pass (2026-11-15, real and complex, bands 16-64), larger counts are
// untested.  Measured 2026-11-15 on 8 cores at n=4000, forced through the private entry:
// bands 16, 32, 48, 50, 64, 100 and 128 all pass and beat the Givens chase (stage 2
// 0.3-1.3 s against 4.3 s real, 0.6-2.5 s against 9.4 s complex).  Smaller bands and
// problems keep the Givens engine.
inline bool pheevd_cpu_householder_shape(int n,int nb,int nbw,int pr,int pc,int workers=1) noexcept
{
    const long long ranks=static_cast<long long>(pr)*pc;
    return n>=2048 && nbw>=16 && nb>=nbw && nb%nbw==0 && pr>=1 && pc>=1 &&
           ranks<=128 && workers>=1 && workers<=64;
}

// Conservative full-solve policy from the matched eight-V100 measurements.
// Return the chosen HH band, or zero to retain the existing Givens policy.
// Explicit widths are retained; default width changes only with the method.
template<class T>
inline int pheevd_gpu_householder_band(char job, int n, int nb, int band, bool explicit_band,
                                      int pr, int pc, char major, int workers, bool one_node,
                                      bool legacy_requested, bool tuned_device) noexcept
{
    if constexpr (!std::is_same<T, double>::value &&
                  !std::is_same<T, std::complex<double>>::value)
        return 0;
    if (job != 'V' || nb != 64 || pr != 2 || pc != 4 || major != 'R' || workers != 1 ||
        !one_node || legacy_requested || !tuned_device)
        return 0;
    if constexpr (std::is_same<T, double>::value)
    {
        if (n < 4096) return 0;
        if (!explicit_band || band == 64) return 64;
        return band == 16 && n >= 32768 ? 16 : 0;
    }
    else
    {
        if (explicit_band) return band == 64 && n >= 8192 ? 64 : 0;
        return n >= 16384 ? 64 : 0;
    }
}
inline DdlaBackend pheevd_gpu_householder_reduction(int n) noexcept
{
    return n >= 16384 ? DdlaBackend::GPU : DdlaBackend::CPU;
}

} }

namespace ddla { namespace detail {
// Private method interfaces. Backend selects matrix storage and execution;
// reduction_backend selects stage2 and defaults to the same backend. GPU
// matrices with CPU reduction retain the hybrid routes. Public dispatch is unchanged.
template<DdlaBackend Backend, class T>
void pheevd_givens(const DdlaHandle_t&, char, char, int, int, T*, const int*,
                  typename real_type<T>::type*, T*, const int*, int&,
                  DdlaBackend reduction_backend = Backend);
template<DdlaBackend Backend, class T>
void pheevd_householder(const DdlaHandle_t&, char, char, int, int, T*, const int*,
                       typename real_type<T>::type*, T*, const int*, int&,
                       DdlaBackend reduction_backend = Backend);
// Private stage4 comparison IDs: CPU WY=0, paired=1, FMA chain=2,
// FMA independent=3, real group4=4/group6=5; GPU WY=0, short reflectors=1.
template<DdlaBackend Backend, class T>
void pheevd_householder_stage4(const DdlaHandle_t&, char, char, int, int, T*, const int*,
                              typename real_type<T>::type*, T*, const int*, int&, int,
                              DdlaBackend reduction_backend = Backend);
} }

namespace ddla
{
namespace spectral_impl
{
enum class PheevdEngine
{
    ReplayCpu,
    ReplayGpu,
    AccumHost,
    Householder
};
struct PheevdPlan
{
    DdlaBackend backend{};
    char job = 'N', uplo = 'U';
    int n = 0, nev = 0, nb = 1, band = 1, steps = 0, ldy = 1, workers = 1;
    bool profile = false, automatic_householder = false;
    PheevdEngine engine = PheevdEngine::ReplayCpu;
    std::array<int, DDLA_DLEN_> source{}, destination{};
    std::size_t output_scratch = 0, replay_solve_bytes = 0, backup_bytes = 0, sink_bytes = 0,
                sink_front_bytes = 0;
    HermitianPlan complete{};
    Phe2hbPlan reduction{};
    Phb2stPlan chase{};
    PstedcVectorPlan vectors{};
    selected::Plan selected_plan{};
    PstedcPlan values{};
    UnmtrHe2hbPlan back{};
    detail::Hb2stAccumHostPlan accum{};
    detail::Hb2stFinishHostPlan finish{};
    detail::CpuReplayPolicy cpu_policy{};
    detail::Hb2stCpuReplayPlan cpu_replay{};
    detail::Hb2stHouseholderCpuPlan householder{};
    detail::HhRowsCpuPlan hh_rows{};
#if DDLA_HAS_GPU
    detail::Hb2stReplayGpuPlan gpu_replay{};
#endif
    detail::PheevdReflectorBackupPlan backup{};
};
template <class T>
PheevdPlan pheevd_check(const DdlaHandle_t &, DdlaBackend, char, char, int, T *, const int *,
                        typename real_type<T>::type *, T *, const int *, const Phb2stTopology &,
                        char method, const int *stage4, DdlaBackend reduction_backend, int nev);
// Both buffers begin with 64 bytes of alignment slack (std::vector and malloc'd
// memory is only 16-byte aligned); the buffers and regions are padded to
// 256-byte units, so every one starts on a 64-byte boundary.
constexpr std::size_t pheevd_alignment = 64;
inline char *pheevd_align(void *p)
{
    return reinterpret_cast<char *>((reinterpret_cast<std::uintptr_t>(p) + pheevd_alignment - 1) &
                                    ~std::uintptr_t(pheevd_alignment - 1));
}
// The scratch of pheevd_run on this rank (none for n == 0), in carving order.
// Host: d, e and the eigenvalues (n reals each) and the phases (n - 1 for
// complex T), live through the whole solve; then the host region.  Backend:
// stage 1's T factors (band x band per panel), live until stage 5; with the
// CPU sink the front of its buffer (its Q_band), live until stage 4 returned;
// then the backend region.  The stages take the regions in turn:
//  - the completion, then the reduction: their backend and host scratch;
//  - the chase: the Householder reduction's scratch (backend), or the rest of
//    the CPU sink's buffer (its planner tables, dead once the chase drained
//    them) followed by phb2st_run's scratch -- on a CPU handle its host
//    scratch goes to the backend region, as it has no backend scratch there;
//  - backend: the reflector backup (Z overlapping A), live until stage 5 read
//    it, then one region: eigenvalues only: pstedc_run's scratch; vectors: a
//    real row slab and its prepared solver scratch (DC: n columns; selected:
//    nev columns and linear host II scratch). GPU selected host upload rows
//    are retired before that arena is reused. Stage 4 consumes the backend
//    slab directly in its existing replay/Householder input path or contracts
//    row panels with Q_band. The final output retains its 2D descriptor.
//    Afterwards stage 5 reuses the region. Host: the largest scratch required
//    by stage 3 or stage 4.
template <class T>
void pheevd_workspace_size(const DdlaHandle_t &, const PheevdPlan &, T *valid_query_A,
                           std::size_t *backend_bytes, std::size_t *host_bytes);
// Bound to this checked plan and the handle/device/streams. Compatible
// repeats change matrix values only. A different shape, policy, descriptor or
// topology requires fresh preparation outside execution.
// release() is collective; the destructor only retires local owners.
template <class T> class PheevdResources
{
    std::optional<detail::EigMemProfile> owned_profile_;

  public:
    detail::EigMemProfile *profile = nullptr;
    bool external_profile = false;
    std::optional<Phb2stResources<T>> chase;
    std::optional<PstedcResources> vectors;
    std::unique_ptr<detail::Hb2stAccumHostResources<T>> accum;
    std::optional<detail::Hb2stHouseholderCpu<T>> householder;
#if DDLA_HAS_GPU
    std::optional<detail::Hb2stReplayGpuResources<T>> gpu_replay;
#endif
    PheevdResources(const DdlaHandle_t &, const PheevdPlan &, detail::EigMemProfile * = nullptr);
    PheevdResources(const PheevdResources &) = delete;
    PheevdResources &operator=(const PheevdResources &) = delete;
    void release()
    {
        if (chase)
            chase->release();
        if (householder)
            householder->release();
    }
};
// Pure two-slot payload high waters after topology/resources preparation.
// Direct MPI on host band/history/tridiagonal buffers consumes no GPU cache.
template<class T>std::array<std::size_t,2> pheevd_staging_bytes(const DdlaHandle_t&,const PheevdPlan&,const PheevdResources<T>&);
// The solve on the caller's scratch (pheevd_workspace_size bytes). No
// validation, no allocation.
template <DdlaBackend B, class T>
void pheevd_run(const DdlaHandle_t &, const PheevdPlan &, PheevdResources<T> &, T *A,
                typename real_type<T>::type *w, T *Z, int &info, void *backend_work,
                std::size_t backend_bytes, void *host_work, std::size_t host_bytes);
} // namespace spectral_impl
} // namespace ddla
#endif
