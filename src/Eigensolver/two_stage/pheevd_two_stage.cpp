// ---------------------------------------------------------------------------
// The two-stage Hermitian eigensolver driver (ELPA2-style):
//   pherm_complete (stored triangle -> full matrix) -> phe2hb (full -> band)
//   -> phb2st (band -> tridiagonal) -> pstedc (tridiagonal solve) ->
//   punmtr_hb2st (tridiagonal -> band back-transform) -> punmtr_he2hb('C')
//   (band -> full back-transform).
//
// jobz == 'N' stops after pstedc (the eigenvalues come from the distributed
// tridiagonal solve).  Only the uplo triangle of the input is read: stage 0
// rebuilds the full Hermitian matrix phe2hb requires from it, in place and
// distributed (one grid alltoallv, O(n^2/P) per rank) -- A is destroyed
// either way.
//
// Stage 3 computes row-distributed eigenvectors with pstedc. Stage 4 consumes
// those rows through bounded redistribution panels. Dense eigenvectors,
// merge workspaces and CPU Q_band storage remain distributed across ranks.
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>

#include "ddla_connector.h"
#include "comm_traits.h"
#include "ddla_desc.h"
#include "Eigensolver/eig_memprof.h"
#include "Eigensolver/eig_params.h"
#include "hb2st_sink.h"
#if DDLA_HAS_GPU
#include "hb2st_replay_gpu.h"
#endif
#include "hb2st_cpu_replay_workspace.h"
#include "hb2st_householder_cpu.h"
#include "hb2st_householder_cpu_rows.h"
#include "pheevd_impl.h"
#include "Eigensolver/pheevd_alias.h"
#include "Eigensolver/pherm_complete_workspace.h"
#include "phe2hb_impl.h"
#include "Eigensolver/eig_cpu_kernels.h"
#include "QR/pgeqrf_impl.h"
#include "check_agree.h"
#include "phb2st_impl.h"
#include "instantiate.h"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ddla
{

namespace detail
{
bool pheevd_cpu_householder_selected(const DdlaHandle_t &h, int n, int nb, int nbw)
{
    int pr = 0, pc = 0;
    ddlaGetGridDims(h, pr, pc);
    const long long ranks = static_cast<long long>(pr) * pc;
    const MPI_Comm comm = ddlaGetCommunicator(h);
    MPI_Comm node = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node));
    int local_size = 0;
    MPI_CHECK(MPI_Comm_size(node, &local_size));
    MPI_CHECK(MPI_Comm_free(&node));
    // Every rank enters the topology collective before the noncollective
    // cache query. A previous solve on a subset may have warmed only some
    // process caches; host_omp_threads(comm) is not safe at this boundary.
    const int workers = host_omp_threads_for_node(local_size);
    bool legacy_requested = false;
    if (const char *mode = std::getenv("DDLA_EIG_CPU_ACCUM"))
        for (const char *known : {"1", "on", "force", "0", "off", "replay"})
            legacy_requested = legacy_requested || std::strcmp(mode, known) == 0;
    if (const char *team = std::getenv("DDLA_EIG_CHASE_TEAM_THREADS"))
        legacy_requested = legacy_requested || *team != '\0';
    int eligible = (local_size == ranks && !legacy_requested &&
                    pheevd_cpu_householder_shape(n, nb, nbw, pr, pc, workers))
                       ? 1
                       : 0;
    int agreed = 0;
    MPI_CHECK(MPI_Allreduce(&eligible, &agreed, 1, MPI_INT, MPI_MIN, comm));
    return agreed != 0;
}
} // namespace detail

namespace spectral_impl
{
namespace
{
inline int pheevd_accum_mode()
{
    if (const char *e = std::getenv("DDLA_EIG_CPU_ACCUM"))
    {
        for (const char *key : {"1", "on", "force"})
            if (!std::strcmp(e, key))
                return 1;
        for (const char *key : {"0", "off", "replay"})
            if (!std::strcmp(e, key))
                return -1;
    }
    return 0;
}
// A region's bytes in whole 256-byte units, so that the region behind it stays aligned.
inline std::size_t pheevd_round(std::size_t bytes)
{
    return detail::padded_bytes<char>(bytes);
}
} // namespace
template <class T>
PheevdPlan pheevd_check(const DdlaHandle_t &h, DdlaBackend backend, char job, char uplo, int n, T *A,
                        const int *da, typename real_type<T>::type *eigenvalues, T *Z, const int *dz,
                        const Phb2stTopology &topology, char method, const int *stage4,
                        DdlaBackend reduction_backend)
{
    using R = typename real_type<T>::type;
    bool tuned_device = false;
    auto p = detail::agree_check(
        h, "pheevd",
        [&](detail::Agree &agreement)
        {
            agreement.mode("spectral operation", 3150);
            PheevdPlan p;
            if (!h || ddlaGetBackend(h) != backend || h->comm == MPI_COMM_NULL || h->nprocs < 1 ||
                h->nprows_ < 1 || h->npcols_ < 1 ||
                static_cast<long long>(h->nprows_) * h->npcols_ != h->nprocs || n < 0 ||
                (job != 'N' && job != 'V') || (uplo != 'U' && uplo != 'L') ||
                (method != 'A' && method != 'G' && method != 'H') ||
                (reduction_backend != DdlaBackend::CPU && reduction_backend != DdlaBackend::GPU) ||
                (backend == DdlaBackend::CPU && reduction_backend != DdlaBackend::CPU) ||
                (method == 'H' && backend==DdlaBackend::GPU && stage4 && (*stage4<0 || *stage4>1)))
                throw std::invalid_argument("pheevd: invalid handle/backend/order/options");
            const auto descriptor = [&](const int *d)
            {
                if (!d || d[DDLA_MB_] < 1 || d[DDLA_NB_] < 1 || d[DDLA_MB_] > INT_MAX / h->nprows_ ||
                    d[DDLA_NB_] > INT_MAX / h->npcols_)
                    throw std::invalid_argument("pheevd: descriptor arithmetic exceeds int");
                check_desc(d, h);
                if (n > d[DDLA_M_] || n > d[DDLA_N_])
                    throw std::invalid_argument("pheevd: order exceeds descriptor");
            };
            descriptor(da);
            if (da[DDLA_MB_] != da[DDLA_NB_] || da[DDLA_RSRC_] != 0 || da[DDLA_CSRC_] != 0)
                throw std::invalid_argument("pheevd: square source-zero A blocks required");
            if (job == 'V')
            {
                descriptor(dz);
                if (dz[DDLA_MB_] != dz[DDLA_NB_] || dz[DDLA_MB_] != da[DDLA_MB_] ||
                    dz[DDLA_RSRC_] != da[DDLA_RSRC_])
                    throw std::invalid_argument("pheevd: output row distribution must match A");
            }
            const auto active = [&](const int *d)
            {
                return num_loc(n, d[DDLA_MB_], h->myprow_, d[DDLA_RSRC_], h->nprows_) > 0 &&
                       num_loc(n, d[DDLA_NB_], h->mypcol_, d[DDLA_CSRC_], h->npcols_) > 0;
            };
            if ((n && !eigenvalues) || (active(da) && !A) || (job == 'V' && active(dz) && !Z))
                throw std::invalid_argument("pheevd: missing active input/output");
#if defined(DDLA_USE_CUDA) && defined(DDLA_USE_CCL) && !defined(DDLA_USE_GPU_CPU_TUNNEL)
            if (backend == DdlaBackend::GPU && method == 'A' && job == 'V' &&
                da[DDLA_MB_] == 64 && n >= 4096)
            {
                int major = 0, minor = 0;
                GPU_CHECK(gpuDeviceGetAttribute(&major, gpuDevAttrComputeCapabilityMajor, h->local_device));
                GPU_CHECK(gpuDeviceGetAttribute(&minor, gpuDevAttrComputeCapabilityMinor, h->local_device));
                tuned_device = major == 7 && minor == 0;
            }
#endif
            p.backend = backend;
            p.job = job;
            p.uplo = uplo;
            p.n = n;
            p.nb = da[DDLA_NB_];
            p.profile = eig_profile_enabled();
            std::copy_n(da, DDLA_DLEN_, p.source.data());
            std::copy_n(job == 'V' ? dz : da, DDLA_DLEN_, p.destination.data());
            if (n)
            {
                const auto parameters =
                    eig_select_params(!std::is_same<T, R>::value, p.nb, backend == DdlaBackend::GPU);
                p.band = eig_distributed_bandwidth(n, p.nb, parameters.nbw, h->nprocs);
                p.steps = (n - 1) / p.band;
            }
            return p;
        });
    int signature[27] = {3150,
                         int(backend),
                         int(job),
                         int(uplo),
                         n,
                         p.band,
                         h->nprows_,
                         h->npcols_,
                         int(h->major),
                         int(method),
                         stage4 ? *stage4 : -1,
                         p.profile ? 1 : 0};
    detail::descriptor_signature(signature + 12, p.source.data());
    detail::descriptor_signature(signature + 19, p.destination.data());
    signature[26] = int(reduction_backend);
    detail::agree_signature(h, "pheevd", signature);
    if (!n)
        return p;
    if (reduction_backend == DdlaBackend::GPU &&
        (p.band > 64 || topology.groups() != 1))
        throw std::invalid_argument("pheevd: GPU stage2 requires one node and band<=64");
    p.workers = detail::host_omp_threads_for_node(topology.node_size());
    const int mode = pheevd_accum_mode();
    bool legacy_requested = mode != 0;
    if (const char *e = std::getenv("DDLA_EIG_CHASE_TEAM_THREADS"))
        legacy_requested = legacy_requested || *e != '\0';
    int automatic = backend == DdlaBackend::CPU && method == 'A' && !legacy_requested &&
                    topology.groups() == 1 &&
                    detail::pheevd_cpu_householder_shape(n, p.nb, p.band, h->nprows_, h->npcols_, p.workers);
    if (backend == DdlaBackend::GPU && method == 'A')
    {
        const char* requested_band = std::getenv("DDLA_EIG_NBW");
        automatic = detail::pheevd_gpu_householder_band<T>(
            job, n, p.nb, p.band, requested_band && *requested_band,
            h->nprows_, h->npcols_, h->major, p.workers, topology.groups() == 1,
            legacy_requested, tuned_device);
    }
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &automatic, 1, MPI_INT, MPI_MIN, h->comm));
    p.automatic_householder = automatic != 0;
    if (p.automatic_householder && backend == DdlaBackend::GPU)
    {
        // MIN also preserves a narrower explicit width when other ranks use
        // the same initial default. Any ineligible rank keeps the old route.
        p.band = automatic;
        p.steps = (n - 1) / p.band;
        reduction_backend = detail::pheevd_gpu_householder_reduction(n);
    }
    const bool householder = method == 'H' || p.automatic_householder;
    p.complete = pherm_complete_check(h, backend, uplo, n, A, da);
    p.output_scratch = job == 'V' ? detail::output_scratch_elements(h, n, A, da, Z, dz) : 0;
    bool inplace = false;
    int upper = 0;
    if (householder)
    {
        if (topology.groups() != 1)
            throw std::invalid_argument("pheevd: Householder chase requires one shared-memory domain");
        p.engine = PheevdEngine::Householder;
        detail::Hb2stHouseholderCpuOptions options;
        if (reduction_backend == DdlaBackend::GPU)
            options.stage2 = detail::Hb2stHouseholderCpuOptions::Stage2::Gpu;
        if (p.automatic_householder && backend == DdlaBackend::CPU)
            options.stage4 = std::is_same<T, R>::value
                                 ? detail::Hb2stHouseholderCpuOptions::Stage4::GroupedReal4
                                 : detail::Hb2stHouseholderCpuOptions::Stage4::FmaIndependent;
        if (stage4)
            options.stage4 = static_cast<detail::Hb2stHouseholderCpuOptions::Stage4>(*stage4);
        p.householder = backend==DdlaBackend::CPU
            ? detail::hh_cpu_check(h, n, p.band, job == 'V', options)
            : detail::hh_cpu_geometry(h, n, p.band, job == 'V', options, p.workers);
        if(backend==DdlaBackend::GPU)
            p.chase.input=phb2st_band_geometry(h,backend,n,p.householder.band,da);
        upper = backend == DdlaBackend::CPU ? (p.automatic_householder ? 1024 : 2560) : 0;
        inplace = p.automatic_householder && backend == DdlaBackend::CPU
                      ? detail::phe2hb_cpu_qr_inplace(h, n, da, p.band)
                                          : qr_impl::qr_panel_inplace_env();
    }
    else if (backend == DdlaBackend::CPU)
    {
        int tpr = std::max(1, std::min(p.workers, 4 / std::max(1, topology.node_size())));
        if (const char *e = std::getenv("DDLA_EIG_CHASE_TEAM_THREADS"))
            tpr = std::max(1, std::min(8, std::atoi(e)));
        int use = job == 'V' && p.band > 1 && n >= 3 && mode >= 0 &&
                  (mode == 1 || (topology.groups() == 1 && p.workers >= std::max(4, tpr + 2) && n >= 2048));
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &use, 1, MPI_INT, MPI_MIN, h->comm));
        p.engine = use ? PheevdEngine::AccumHost : PheevdEngine::ReplayCpu;
        if (use)
            p.accum = detail::hb2st_accum_host_check(h, n, p.band, p.workers);
        if (!use && job == 'V')
            p.cpu_policy = detail::cpu_replay_policy_for_threads(p.workers);
        inplace = qr_impl::qr_panel_inplace_env();
    }
#if DDLA_HAS_GPU
    if (backend == DdlaBackend::GPU && !householder)
    {
        p.engine = PheevdEngine::ReplayGpu;
        inplace = qr_impl::qr_panel_inplace_env();
    }
#endif
    const int panel_policy[] = {inplace ? 1 : 0, int(p.engine), p.automatic_householder ? 1 : 0,
                                p.band, int(reduction_backend)};
    detail::agree_signature(h, "pheevd engine/panel policy", panel_policy);
    p.reduction = phe2hb_geometry(h, backend, n, da, p.band, upper, inplace);
    detail::agree_check(h, "pheevd panel counts",
                        [&](detail::Agree &)
                        {
                            const auto &r = p.reduction;
                            if (static_cast<long long>(r.ldv) * p.band > INT_MAX ||
                                static_cast<long long>(r.ldc) * p.band > INT_MAX ||
                                static_cast<long long>(p.band) * p.band > INT_MAX ||
                                r.extent > INT_MAX - (r.nb - 1))
                                throw std::overflow_error("pheevd: panel collective count exceeds int");
                        });
    if (p.engine != PheevdEngine::Householder)
    {
        Phb2stSinkPolicy sink;
        if (p.engine == PheevdEngine::AccumHost)
        {
            sink.chunk = p.accum.block_records;
            sink.max_roles = std::max(2, p.accum.threads);
            sink.roles_in_team = true;
        }
        p.chase = phb2st_policy(h, phb2st_band_geometry(h, backend, n, p.band, da), topology,
                                job == 'N'                            ? Phb2stOutput::None
                                : p.engine == PheevdEngine::AccumHost ? Phb2stOutput::Sink
                                                                      : Phb2stOutput::History,
                                sink);
        // Short streams need no storage beyond their complete record count;
        // a sink also keeps one step of the existing publication look-ahead.
        // Larger streams retain the original bounded-ring capacities.
        if (p.chase.local.bounded)
        {
            const auto lookahead =
                p.chase.output == Phb2stOutput::Sink ? static_cast<long long>(n) / (p.band + 1) + 2 : 0;
            p.chase.local.record_capacity =
                std::min(p.chase.local.record_capacity, std::max(1LL, p.chase.rotations + lookahead));
        }
        if (p.chase.output == Phb2stOutput::History && p.chase.spatial.ring_capacity)
            p.chase.spatial.ring_capacity =
                std::min(p.chase.spatial.ring_capacity, std::max(1LL, p.chase.rotations));
        if (reduction_backend == DdlaBackend::GPU)
        {
            p.chase.device_chase = true;
            p.chase.team.enabled = true;
            p.chase.team.participants = h->nprocs;
            p.chase.team.tpr = 1;
            p.chase.team.consumers = false;
            p.chase.team.elastic = false;
            p.chase.team.ring_request = p.chase.output == Phb2stOutput::History
                                           ? p.chase.local.record_capacity : 1;
        }
    }
    p.backup = detail::pheevd_reflector_backup_check<T>(h, n, A, da, Z, job == 'V' ? dz : nullptr);
    detail::pheevd_reflector_backup_workspace_size<T>(p.backup, &p.backup_bytes);
    if (p.engine == PheevdEngine::AccumHost)
    {
        std::size_t front = 0;
        detail::hb2st_accum_host_workspace_size<T>(h, p.accum, &p.sink_bytes);
        detail::hb2st_accum_host_qband_workspace_size<T>(h, p.accum, &front);
        p.sink_front_bytes = pheevd_round(front);
    }
    if (job == 'N')
    {
        p.values = pstedc_values_policy<R>(h, n);
        return p;
    }
    const detail::RowShare rows(n, h->nprocs);
    p.ldy = std::max(1, rows.count(h->myid));
    p.vectors = pstedc_policy<R>(h, n);
    p.vectors.ldy = p.ldy;
    // The composite owns Y separately from the replicated d/e/w arrays.
    p.vectors.borrow_q2 = rows.count(h->myid) > 0;
    const auto phases = std::is_same<T, R>::value ? 0 : std::size_t(n - 1);
    if (p.engine == PheevdEngine::Householder && backend==DdlaBackend::CPU)
        p.hh_rows = detail::hh_rows_cpu_geometry<T>(h, n, p.ldy, dz, phases,
                                                    p.householder.options.stage4 !=
                                                        detail::Hb2stHouseholderCpuOptions::Stage4::Wy,
                                                    p.householder.threads);
    else if (p.engine == PheevdEngine::AccumHost)
    {
        long long contraction = 512;
        if (const char *e = std::getenv("DDLA_EIG_HOST_FINISH_PANEL"))
            if (*e)
                contraction = std::max(16LL, std::strtoll(e, nullptr, 10));
        detail::agree_check(h, "pheevd finish policy",
                            [&](detail::Agree &)
                            {
                                if (contraction > INT_MAX)
                                    throw std::overflow_error("pheevd: finish panel exceeds int");
                            });
        p.finish = detail::hb2st_finish_host_geometry<T>(h, n, p.ldy, dz, p.workers, 0, int(contraction),
                                                         phases > 0);
        const int choice[] = {p.finish.kp, p.finish.panel};
        detail::agree_signature(h, "pheevd finish policy", choice);
    }
    else if (p.engine == PheevdEngine::ReplayCpu)
    {
        p.cpu_replay = detail::hb2st_cpu_replay_geometry<T>(h, n, p.band, p.chase.rotations, p.ldy, phases,
                                                            dz, p.cpu_policy);
        const int policy[] = {p.cpu_policy.tile,
                              p.cpu_policy.wave_mode,
                              p.cpu_policy.row_block,
                              p.cpu_policy.level_block,
                              p.cpu_policy.waves ? 1 : 0,
                              p.cpu_policy.profile ? 1 : 0,
                              p.cpu_policy.stage_profile ? 1 : 0};
        detail::agree_signature(h, "pheevd replay policy", policy);
    }
#if DDLA_HAS_GPU
    if (backend==DdlaBackend::GPU) {
        p.gpu_replay=detail::hb2st_replay_gpu_geometry<T>(h,n,p.band,p.chase.rotations,p.ldy,phases,dz);
        p.gpu_replay.householder=householder;
        if (householder) {
            p.gpu_replay.kd = p.householder.band;
            p.gpu_replay.householder_wy=p.householder.options.stage4==detail::Hb2stHouseholderCpuOptions::Stage4::Wy;
        }
    }
#endif
    if (p.engine == PheevdEngine::ReplayCpu)
    {
        std::size_t solver = 0, host = 0, input = 0;
        pstedc_workspace_size<R>(h, p.vectors, &solver, &host);
        detail::hb2st_cpu_replay_input_workspace_size<T>(h, p.cpu_replay, &input);
        p.replay_solve_bytes = std::max(pheevd_round(solver), pheevd_round(input));
    }
    const int *saved = p.backup.needed ? p.backup.saved.data() : da;
    p.back = punmtr_he2hb_geometry(h, backend, 'C', n, p.band, saved, dz);
    return p;
}

template <class T>
void pheevd_workspace_size(const DdlaHandle_t &h, const PheevdPlan &p, T *query_A, std::size_t *backend_bytes,
                           std::size_t *host_bytes)
{
    using R = typename real_type<T>::type;
    using detail::padded_bytes;
    *backend_bytes = *host_bytes = 0;
    if (!p.n)
        return;
    const bool cpu = p.backend == DdlaBackend::CPU;
    const std::size_t n = p.n;
    // region / host_region: the largest stage of each kind; engine: the CPU sink's Q_band front.
    std::size_t region = 0, host_region = 0, engine = 0, b = 0, hb = 0;
    const auto stage = [&](std::size_t backend, std::size_t host)
    {
        region = std::max(region, backend);
        host_region = std::max(host_region, host);
    };
    pherm_complete_workspace_size<T>(h, p.complete, cpu ? 0 : p.output_scratch / (sizeof(T) / sizeof(R)), &b,
                                     &hb);
    stage(b, hb);
    phe2hb_workspace_size<T>(h, p.reduction, query_A, query_A, &b, &hb);
    stage(b, hb);
    if (p.engine == PheevdEngine::Householder)
    {
        detail::hh_cpu_reduce_workspace_size<T>(h, p.householder.band, p.householder.chase_threads, &b);
        if(cpu)stage(b,0);
        else {
            std::size_t gather=0;phb2st_band_workspace_size<T>(p.chase.input,&gather);
            stage(gather,padded_bytes<T>(phb2st_band_count(p.chase.input))+b);
        }
    }
    else
    {
        phb2st_workspace_size<T>(p.chase, &b, &hb);
        if (cpu)
        {
            assert(!b);
            std::size_t sink_tail = 0;
            if (p.engine == PheevdEngine::AccumHost)
            {
                engine = p.sink_front_bytes;
                sink_tail = pheevd_round(p.sink_bytes) - engine;
            }
            stage(sink_tail + hb, 0);
        }
        else
            stage(b, hb);
    }
    std::size_t later = 0, later_host = 0;
    if (p.job == 'N')
        pstedc_workspace_size<R>(h, p.values, &later, &later_host);
    else
    {
        const std::size_t y = padded_bytes<R>(std::size_t(p.ldy) * p.n);
        std::size_t rows = 0;
        pstedc_workspace_size<R>(h, p.vectors, &rows, &later_host);
        rows = pheevd_round(rows);
        if (p.engine == PheevdEngine::ReplayCpu)
        {
            detail::hb2st_cpu_replay_workspace_size<T>(h, p.cpu_replay, &b);
            later = std::max(b, p.replay_solve_bytes + y);
        }
        else if (p.engine == PheevdEngine::AccumHost)
        {
            detail::hb2st_finish_host_workspace_size<T>(h, p.finish, &b);
            later = y + std::max(rows, b);
        }
        else if (p.engine == PheevdEngine::Householder && cpu)
        {
            detail::hh_rows_cpu_workspace_size<T>(h, p.hh_rows, &b);
            later = y + std::max(rows, b);
        }
#if DDLA_HAS_GPU
        if (!cpu)
        {
            detail::hb2st_replay_gpu_workspace_size<T>(h, p.gpu_replay, &b, &hb);
            later = y + std::max(rows, b);
            later_host = std::max(later_host, hb);
        }
#endif
        punmtr_he2hb_workspace_size<T>(h, p.back, &b);
        later = std::max(later, b);
    }
    stage(p.backup_bytes + later, later_host);
    *backend_bytes =
        pheevd_alignment + padded_bytes<T>(std::size_t(p.band) * p.band * p.steps) + engine + region;
    *host_bytes = pheevd_alignment + 3 * padded_bytes<R>(n) +
                  padded_bytes<T>(std::is_same<T, R>::value ? 0 : n - 1) + host_region;
}

template <class T>
PheevdResources<T>::PheevdResources(const DdlaHandle_t &h, const PheevdPlan &p,
                                    detail::EigMemProfile *external)
{
    if (!p.n)
        return;
    std::exception_ptr failure;
    try
    {
        if (p.profile)
        {
            if (external)
            {
                profile = external;
                external_profile = true;
            }
            else
            {
                owned_profile_.emplace(h, true, p.backend == DdlaBackend::GPU);
                profile = &*owned_profile_;
            }
        }
        if (p.job == 'V')
            vectors.emplace(h, p.vectors);
        if (p.engine == PheevdEngine::AccumHost)
            accum = detail::hb2st_accum_host_prepare<T>(h, p.accum);
#if DDLA_HAS_GPU
        if (p.backend==DdlaBackend::GPU && p.job == 'V')
            gpu_replay.emplace(h, p.gpu_replay);
#endif
    }
    catch (...)
    {
        failure = std::current_exception();
    }
    detail::agree_failure(h,failure,"pheevd: preparation");
    if (p.engine == PheevdEngine::Householder)
        householder.emplace(h, p.householder);
    else
        chase.emplace(h, p.chase);
}

template <DdlaBackend B, class T>
void pheevd_run(const DdlaHandle_t &h, const PheevdPlan &p, PheevdResources<T> &resources, T *A,
                typename real_type<T>::type *output, T *Z, int &info, void *backend_work,
                std::size_t backend_bytes, void *host_work, std::size_t host_bytes)
{
    try
    {
        using R = typename real_type<T>::type;
        using detail::take;
        info = 0;
        if (!p.n)
            return;
        auto *profile = resources.profile;
        if (profile && (!resources.external_profile || !profile->running()))
        {
            profile->begin_run();
            profile->phase("start");
        }
        const auto mark = [&](const char *name)
        {
            if (profile)
                profile->phase(name);
        };
        const auto note = [&](int code)
        {
            if (!info && code)
                info = code;
        };
        const std::size_t n = p.n;
        // The scalars, the T factors and the CPU sink's Q_band front in the order
        // pheevd_workspace_size counts them; the stages take turns on the regions behind.
        char *const host_base = static_cast<char *>(host_work);
        char *hw = pheevd_align(host_base);
        R *const d = take<R>(hw, n);
        R *const e = take<R>(hw, n);
        R *const values = take<R>(hw, n);
        T *const phases = take<T>(hw, std::is_same<T, R>::value ? 0 : n - 1);
        char *const base = static_cast<char *>(backend_work);
        char *w = pheevd_align(base);
        T *const tmat = take<T>(w, std::size_t(p.band) * p.band * p.steps);
        [[maybe_unused]] char *sink = nullptr;
        [[maybe_unused]] std::size_t sink_bytes = 0, sink_tail = 0;
        if (p.engine == PheevdEngine::AccumHost)
        {
            sink_bytes = p.sink_bytes;
            sink = w;
            w += p.sink_front_bytes;
            sink_tail = pheevd_round(sink_bytes) - p.sink_front_bytes;
        }
        assert(static_cast<std::size_t>(w - base) <= backend_bytes &&
               static_cast<std::size_t>(hw - host_base) <= host_bytes);
        char *const region = w;
        const std::size_t region_bytes = backend_bytes - static_cast<std::size_t>(w - base);
        char *const host_region = hw;
        const std::size_t host_region_bytes = host_bytes - static_cast<std::size_t>(hw - host_base);
        try
        {
            const std::size_t scratch_elements =
                B == DdlaBackend::CPU ? 0 : p.output_scratch / (sizeof(T) / sizeof(R));
            pherm_complete_run<B, T>(h, p.complete, A, scratch_elements ? Z : nullptr, scratch_elements,
                                     region, region_bytes, host_region, host_region_bytes);
            mark("sym");
            int stage = 0;
            phe2hb_run<B, T>(h, p.reduction, A, tmat, stage, region, region_bytes, host_region,
                             host_region_bytes);
            note(stage);
            mark("s1");
            std::size_t phase_count = std::is_same<T, R>::value ? 0 : n - 1;
            if (p.engine == PheevdEngine::Householder) {
                if constexpr(B==DdlaBackend::CPU)
                    resources.householder->reduce_run(A,p.source.data(),d,e,phases,region,region_bytes);
                else {
                    char* at=host_region;
                    T* band=take<T>(at,phb2st_band_count(p.chase.input));
                    phb2st_band_run<B,T>(h,p.chase.input,A,band,region,region_bytes);
                    resources.householder->reduce_run(nullptr,p.source.data(),d,e,phases,at,
                        host_region_bytes-std::size_t(at-host_region),band);
                }
            } else
            {
                Hb2stSink<T> *sink_consumer = nullptr;
                if (resources.accum)
                {
                    resources.accum->reset_run(sink, sink_bytes);
                    sink_consumer = resources.accum.get();
                }
                // A CPU handle's chase has host scratch only; it goes behind the sink's tail in the backend
                // region.
                if constexpr (B == DdlaBackend::CPU)
                    phb2st_run<B, T>(h, p.chase, *resources.chase, A, d, e, phases, phase_count,
                                     sink_consumer, nullptr, 0, region + sink_tail, region_bytes - sink_tail);
                else
                    phb2st_run<B, T>(h, p.chase, *resources.chase, A, d, e, phases, phase_count,
                                     sink_consumer, region, region_bytes, host_region, host_region_bytes);
            }
            mark("s2");
            // From here on: the reflector backup, then one region of each kind for stages 3-5.
            char *later = region;
            const std::size_t backup = p.backup_bytes;
            detail::pheevd_reflector_backup_save_run<T>(h, p.backup, A, later, backup);
            const T *const reflectors = p.backup.needed ? reinterpret_cast<const T *>(later) : A;
            later += backup;
            const std::size_t later_bytes = region_bytes - static_cast<std::size_t>(later - region);
            R *Y = nullptr;
            char *solve = later;
            std::size_t solve_bytes = later_bytes;
            if (p.job == 'V')
            {
                const std::size_t y = std::size_t(p.ldy) * p.n;
                if (p.engine == PheevdEngine::ReplayCpu)
                {
                    // Keep the row source behind both D&C scratch and replay input;
                    // replay apply can reuse it once every input panel was read.
                    solve_bytes = p.replay_solve_bytes;
                    char *at = later + solve_bytes;
                    Y = take<R>(at, y);
                }
                else
                {
                    Y = take<R>(solve, y);
                    solve_bytes = later_bytes - static_cast<std::size_t>(solve - later);
                }
                assert(resources.vectors);
                pstedc_run<B, R>(h, p.vectors, *resources.vectors, d, e, values, Y, nullptr, solve,
                                 solve_bytes, host_region, host_region_bytes);
            }
            else
                pstedc_run<B, R>(h, p.values, d, e, values, nullptr, later, later_bytes, host_region,
                                 host_region_bytes);
            if (resources.householder)
            {
                int finite = 1;
                const auto scale = resources.householder->eigenvalue_scale();
                for (std::size_t i = 0; i < n; ++i)
                {
                    values[i] *= scale;
                    if (!std::isfinite(values[i]))
                        finite = 0;
                }
                MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &finite, 1, MPI_INT, MPI_MIN, h->comm));
                if (!finite)
                    throw std::runtime_error("pheevd: CPU Householder eigenvalue rescaling is not finite");
            }
            std::copy(values, values + n, output);
            mark("s3");
            if (p.job == 'N')
            {
                mark("done");
                if (profile && !resources.external_profile)
                    profile->pause_run();
                return;
            }
            stage = 0;
            if (p.engine == PheevdEngine::Householder && B==DdlaBackend::CPU)
                detail::hh_rows_cpu_run<T>(h, p.hh_rows, Y, *resources.householder, phases, Z, stage, solve,
                                           solve_bytes);
            else if (p.engine == PheevdEngine::AccumHost)
                resources.accum->finish_rows_run(p.finish, Y, phases, Z, stage, solve, solve_bytes);
            else if (p.engine == PheevdEngine::ReplayCpu)
            {
                detail::hb2st_cpu_replay_input_run<T>(h, p.cpu_replay, Y, phases, later, solve_bytes);
                // The input run has read Y; the apply run reuses the whole region while
                // the tiled block the input run left at its front remains live.
                detail::hb2st_cpu_replay_apply_run<T>(h, p.cpu_replay, resources.chase->history(), Z, stage,
                                                      later, later_bytes);
            }
#if DDLA_HAS_GPU
            if constexpr (B == DdlaBackend::GPU)
            {
                const detail::DistributedHistory<T> empty_history{};
                detail::hb2st_replay_gpu_run<T>(h, p.gpu_replay, *resources.gpu_replay,
                                                resources.chase?resources.chase->history():empty_history,
                                                phases,Y,Z,stage,nullptr,solve,solve_bytes,host_region,host_region_bytes,
                                                resources.householder?&*resources.householder:nullptr);
            }
#endif
            note(stage);
            mark("s4");
            const int *descriptor = p.backup.needed ? p.backup.saved.data() : p.source.data();
            stage = 0;
            punmtr_he2hb_run<B, T>(h, p.back, reflectors, descriptor, tmat, Z, p.destination.data(),
                                   stage, later, later_bytes);
            note(stage);
            mark("s5");
            mark("done");
            if (profile && !resources.external_profile)
                profile->pause_run();
        }
        catch (...)
        {
            if (resources.accum)
                resources.accum->cancel_run();
            if (profile)
                profile->pause_run();
            throw;
        }
    }
    catch (...)
    {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(h) == DdlaBackend::GPU)
            try
            {
                (void)ddlaSynchronize(h);
            }
            catch (...)
            {
            }
        throw;
    }
}

#define DDLA_PHEEVD_PHASES(T, R)                                                                             \
    template PheevdPlan pheevd_check<T>(const DdlaHandle_t &, DdlaBackend, char, char, int, T *,             \
                                        const int *, R *, T *, const int *, const Phb2stTopology &, char,    \
                                        const int *, DdlaBackend);                                           \
    template void pheevd_workspace_size<T>(const DdlaHandle_t &, const PheevdPlan &, T *, std::size_t *,     \
                                           std::size_t *);                                                   \
    template PheevdResources<T>::PheevdResources(const DdlaHandle_t &, const PheevdPlan &,                   \
                                                 detail::EigMemProfile *);
DDLA_PHEEVD_PHASES(float, float)
DDLA_PHEEVD_PHASES(double, double)
DDLA_PHEEVD_PHASES(std::complex<float>, float)
DDLA_PHEEVD_PHASES(std::complex<double>, double)
#undef DDLA_PHEEVD_PHASES
#define DDLA_PHEEVD_RUN(B, T, R)                                                                             \
    template void pheevd_run<B, T>(const DdlaHandle_t &, const PheevdPlan &, PheevdResources<T> &, T *, R *, \
                                   T *, int &, void *, std::size_t, void *, std::size_t);
DDLA_PHEEVD_RUN(DdlaBackend::CPU, float, float)
DDLA_PHEEVD_RUN(DdlaBackend::CPU, double, double)
DDLA_PHEEVD_RUN(DdlaBackend::CPU, std::complex<float>, float)
DDLA_PHEEVD_RUN(DdlaBackend::CPU, std::complex<double>, double)
#if DDLA_HAS_GPU
DDLA_PHEEVD_RUN(DdlaBackend::GPU, float, float)
DDLA_PHEEVD_RUN(DdlaBackend::GPU, double, double)
DDLA_PHEEVD_RUN(DdlaBackend::GPU, std::complex<float>, float)
DDLA_PHEEVD_RUN(DdlaBackend::GPU, std::complex<double>, double)
#endif
#undef DDLA_PHEEVD_RUN
} // namespace spectral_impl

template <DdlaBackend Backend, typename T>
static void pheevd_two_stage_impl(const DdlaHandle_t &handle, char jobz, char uplo, int n, T *A,
                                  const int *descA, typename real_type<T>::type *w, T *Z, const int *descZ,
                                  int &info, char method, const int *private_stage4 = nullptr,
                                  DdlaBackend reduction_backend = DdlaBackend::CPU)
{
    info = 0;
    if (!handle)
        throw std::invalid_argument("pheevd: null handle");
    detail::agree_check(handle, "pheevd handle",
                        [&](detail::Agree &)
                        {
                            if (handle->comm == MPI_COMM_NULL || handle->nprocs < 1 || handle->nprows_ < 1 ||
                                handle->npcols_ < 1 ||
                                static_cast<long long>(handle->nprows_) * handle->npcols_ != handle->nprocs)
                                throw std::invalid_argument("pheevd: initialized process grid required");
                        });
    // The public-call baseline precedes topology, scratch and resource
    // preparation. Reusable raw callers provide their own prepared context.
    const bool profiling = eig_profile_enabled();
    std::optional<detail::EigMemProfile> prepared_profile;
    std::exception_ptr failure;
    try
    {
        prepared_profile.emplace(handle, profiling, Backend == DdlaBackend::GPU);
    }
    catch (...)
    {
        failure = std::current_exception();
    }
    detail::agree_failure(handle,failure,"pheevd: preparation");
    auto &profile = *prepared_profile;
    if (profiling)
        profile.phase("start");
    auto topology = spectral_impl::phb2st_topology_prepare(handle);
    spectral_impl::PheevdPlan plan;
    try
    {
        plan = spectral_impl::pheevd_check<T>(handle, Backend, jobz, uplo, n, A, descA, w, Z, descZ,
                                              *topology, method, private_stage4, reduction_backend);
    }
    catch (...)
    {
        topology->release();
        throw;
    }
    try
    {
        {
            std::size_t backend_bytes = 0, host_bytes = 0;
            vector<char> backend_work(handle);
            std::vector<char> host_work;
            try
            {
                spectral_impl::pheevd_workspace_size<T>(handle, plan, A, &backend_bytes, &host_bytes);
                backend_work.resize(backend_bytes);
                host_work.resize(host_bytes);
            }
            catch (...)
            {
                failure = std::current_exception();
            }
            detail::agree_failure(handle,failure,"pheevd: preparation");
            std::optional<spectral_impl::PheevdResources<T>> resources;
            try
            {
                resources.emplace(handle, plan, profiling ? &profile : nullptr);
            }
            catch (...)
            {
                failure = std::current_exception();
            }
            detail::agree_failure(handle,failure,"pheevd: preparation");
            if (profiling)
                profile.phase("prepare");
            try
            {
                spectral_impl::pheevd_run<Backend, T>(handle, plan, *resources, A, w, Z, info,
                                                      backend_work.data(), backend_bytes, host_work.data(),
                                                      host_bytes);
            }
            catch (...)
            {
                resources->release();
                throw;
            }
            resources->release();
        }
        topology->release();
    }
    catch (...)
    {
        topology->release();
        throw;
    }
    if (profiling)
    {
        profile.phase("free");
        profile.report_timing(handle->myid, n, plan.band);
        profile.report(handle->myid, n);
    }
}

template <DdlaBackend Backend, typename T>
void pheevd_two_stage(const DdlaHandle_t &handle, char jobz, char uplo, int n, T *A, const int *descA,
                      typename real_type<T>::type *w, T *Z, const int *descZ, int &info)
{
    pheevd_two_stage_impl<Backend, T>(handle, jobz, uplo, n, A, descA, w, Z, descZ, info, 'A');
}

namespace detail
{
template<DdlaBackend Backend, class T>
void pheevd_givens(const DdlaHandle_t& h, char job, char uplo, int n, T* A, const int* da,
                  typename real_type<T>::type* w, T* Z, const int* dz, int& info,
                  DdlaBackend reduction_backend)
{
    pheevd_two_stage_impl<Backend, T>(h, job, uplo, n, A, da, w, Z, dz, info,
                                     'G', nullptr, reduction_backend);
}
template<DdlaBackend Backend, class T>
void pheevd_householder(const DdlaHandle_t& h, char job, char uplo, int n, T* A, const int* da,
                       typename real_type<T>::type* w, T* Z, const int* dz, int& info,
                       DdlaBackend reduction_backend)
{
    pheevd_two_stage_impl<Backend, T>(h, job, uplo, n, A, da, w, Z, dz, info,
                                     'H', nullptr, reduction_backend);
}
template<DdlaBackend Backend, class T>
void pheevd_householder_stage4(const DdlaHandle_t& h, char job, char uplo, int n, T* A, const int* da,
                              typename real_type<T>::type* w, T* Z, const int* dz, int& info,
                              int stage4, DdlaBackend reduction_backend)
{
    pheevd_two_stage_impl<Backend, T>(h, job, uplo, n, A, da, w, Z, dz, info,
                                     'H', &stage4, reduction_backend);
}
#define INSTANTIATE_METHODS(B, T) \
    template void pheevd_givens<B, T>(const DdlaHandle_t&, char, char, int, T*, const int*, \
        real_type<T>::type*, T*, const int*, int&, DdlaBackend); \
    template void pheevd_householder<B, T>(const DdlaHandle_t&, char, char, int, T*, const int*, \
        real_type<T>::type*, T*, const int*, int&, DdlaBackend); \
    template void pheevd_householder_stage4<B, T>(const DdlaHandle_t&, char, char, int, T*, const int*, \
        real_type<T>::type*, T*, const int*, int&, int, DdlaBackend)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_METHODS)
#undef INSTANTIATE_METHODS
} // namespace detail

#define INSTANTIATE_PHEEVD_TWO_STAGE(BACKEND, TYPE)                                                          \
    template void pheevd_two_stage<BACKEND, TYPE>(const DdlaHandle_t &, char, char, int, TYPE *,             \
                                                  const int *, typename real_type<TYPE>::type *, TYPE *,     \
                                                  const int *, int &)

DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_PHEEVD_TWO_STAGE)

#undef INSTANTIATE_PHEEVD_TWO_STAGE

} // namespace ddla
