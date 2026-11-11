#ifndef DDLA_HB2ST_CPU_REPLAY_WORKSPACE_H
#define DDLA_HB2ST_CPU_REPLAY_WORKSPACE_H
// Components (plan_util.h) of the CPU stage-4 replay: the column-strip pack of
// a rotation stream, distributed rows replay and node-shared wave schedules.
#include "Eigensolver/eig_cpu_kernels.h"
#include "eig_cpu_column_layout.h"
#include "distributed_history.h"
#include "eig_2d_input.h"
#include <cstddef>
namespace ddla
{
namespace detail
{
// Resolve environment and the existing thread policy before check/size/run
// composition. Size functions and numerical kernels consume only this frozen POD.
struct CpuReplayPolicy
{
    int threads = 1, tile = 0, wave_mode = 1, row_block = 512, level_block = 32;
    bool waves = true, profile = false, stage_profile = false;
};
CpuReplayPolicy cpu_replay_policy(const DdlaHandle_t&);
// Composition already holding node topology can freeze the same environment
// policy with its resolved thread budget, without creating another communicator.
CpuReplayPolicy cpu_replay_policy_for_threads(int);
enum class CpuReplaySource
{
    Records,
    Shared,
    Distributed
};
struct CpuReplayPackPlan
{
    int n = 0, ncs = 0, mb = 1, npr = 1, tile = 8, kd = 1;
    std::size_t records = 0;
    CpuReplayPolicy policy{};
    CpuReplaySource source = CpuReplaySource::Records;
};
template <class T>
CpuReplayPackPlan cpu_replay_pack_geometry(int n, int ncs, int mb, int npr, int kd, std::size_t records,
                                           CpuReplaySource, const CpuReplayPolicy &);
// Y holds y_count reals: a generic RHS slab may have any number of columns,
// and every selected column columns[k]*ldy + n must lie inside it.
template <class T>
void cpu_replay_pack_check(const CpuReplayPackPlan &, const typename real_type<T>::type *Y,
                           std::size_t y_count, int ldy, const int *columns, const hb2st_rot<T> *records,
                           std::size_t record_count, const WaveShared<T> *, const DistributedHistory<T> *,
                           const T *phases, const int *sd, T *sendbuf);
// Host bytes of cpu_replay_pack_run (none for an empty local replay), in
// carving order: the send-buffer row offsets (n); the tile slots (one per
// thread, one per tile with a distributed history; n x tile each, two planes
// for complex) on a 64-byte boundary; one history chunk of records
// (distributed history); the wave schedule of the stream or of one chunk
// (16 or more columns; not for a node-shared schedule).
template <class T> void cpu_replay_pack_workspace_size(const CpuReplayPackPlan &, std::size_t *host_bytes);
// The source is p.source: records (p.records rotations), the node-shared
// schedule or the distributed history (collective over comm).
template <class T>
void cpu_replay_pack_run(const CpuReplayPackPlan &, const typename real_type<T>::type *Y, int ldy,
                         const int *columns, const hb2st_rot<T> *records, const WaveShared<T> *,
                         const DistributedHistory<T> *, MPI_Comm comm, const T *phases, const int *sd,
                         T *sendbuf, void *host_work, std::size_t host_bytes);

struct Hb2stCpuReplayPlan
{
    Eig2dInputPlan input_2d{};
    CpuReplayPlan columns{};
    CpuReplayPolicy policy{};
    int n = 0, kd = 1, ldy = 1, tile = 8;
    long long total = 0;
    std::size_t phase_count = 0;
};
template <class T>
Hb2stCpuReplayPlan hb2st_cpu_replay_geometry(const DdlaHandle_t &, int n, int kd, long long total, int ldy,
                                             std::size_t phase_count, const int *desc,
                                             const CpuReplayPolicy &, int panel = 64);
template <class T>
Hb2stCpuReplayPlan hb2st_cpu_replay_check(const DdlaHandle_t &, int n, const typename real_type<T>::type *Y,
                                          int ldy, const DistributedHistory<T> &, const T *phases,
                                          std::size_t phase_count, T *Z, const int *desc, int panel = 64);
// Backend bytes (CPU handle) of the replay's two runs, which share one buffer
// (none for n == 0), in carving order: the column layout; the tiled block
// (ceil(ncs / tile) tiles of n x tile, two planes for complex) on a 64-byte
// boundary, which the input run fills and the apply run consumes; then one
// region that holds, in turn, the input panel's send and receive buffers,
// one history chunk of records and its wave schedule (16 or more columns),
// and the output panel's send and receive buffers.
template <class T>
void hb2st_cpu_replay_workspace_size(const DdlaHandle_t &, const Hb2stCpuReplayPlan &,
                                     std::size_t *backend_bytes);
// The front of that buffer the input run uses: the layout, the tiled block and
// the input panel's buffers. A caller may keep Y in the same buffer behind
// these bytes; the apply run reuses them.
template <class T>
void hb2st_cpu_replay_input_workspace_size(const DdlaHandle_t &, const Hb2stCpuReplayPlan &,
                                           std::size_t *backend_bytes);
// Moves this rank's row slab Y (ld p.ldy; p.phase_count phases) into the
// tiled block. Every Y read is complete on return. Collective.
template <class T>
void hb2st_cpu_replay_input_run(const DdlaHandle_t &, const Hb2stCpuReplayPlan &,
                                const typename real_type<T>::type *Y, const T *phases, void *backend_work,
                                std::size_t backend_bytes);
// Replays the history on the tiled block that the input run left in the same
// buffer and writes Z. Collective.
template <class T>
void hb2st_cpu_replay_apply_run(const DdlaHandle_t &, const Hb2stCpuReplayPlan &,
                                const DistributedHistory<T> &, T *Z, int &info, void *backend_work,
                                std::size_t backend_bytes);
// The input run, then the apply run, on one buffer. Collective.
template <class T>
void hb2st_cpu_replay_run(const DdlaHandle_t &, const Hb2stCpuReplayPlan &, const DistributedHistory<T> &,
                          const T *phases, const typename real_type<T>::type *Y, T *Z, int &info,
                          void *backend_work, std::size_t backend_bytes);

// Genuine shared-window resources are prepared/released collectively outside
// fill/apply. The default distributed replay never prepares these windows.
struct CpuNodeWavePlan
{
    int n = 0, kd = 1, threads = 1;
    std::size_t records = 0;
    CpuReplayPolicy policy{};
};
template <class T>
CpuNodeWavePlan cpu_node_wave_check(MPI_Comm, int n, const hb2st_rot<T> *records, std::size_t record_count,
                                    const CpuReplayPolicy &);
// Host bytes of cpu_node_wave_fill_run (none when the stream cannot use the
// wave order): the leader's wave tables and planning scratch.
template <class T> void cpu_node_wave_workspace_size(const CpuNodeWavePlan &, std::size_t *host_bytes);
// The node windows of the plan's schedule (none when it cannot use the wave
// order): the record placement, the wave tables and the regrouped records.
template <class T> struct CpuNodeWaveResources
{
    MPI_Comm node = MPI_COMM_NULL;
    NodeWindow index{}, table{}, records{};
    CpuNodeWaveResources(MPI_Comm, const CpuNodeWavePlan &);
    CpuNodeWaveResources(const CpuNodeWaveResources &) = delete;
    CpuNodeWaveResources &operator=(const CpuNodeWaveResources &) = delete;
    void release_plan();
    void release();
};
// rots: the plan's p.records rotations. Collective over the resources' node.
template <class T>
bool cpu_node_wave_fill_run(const CpuNodeWavePlan &, CpuNodeWaveResources<T> &, const hb2st_rot<T> *rots,
                            WaveShared<T> &out, void *host_work, std::size_t host_bytes);
// Private consuming CPU stage 4. Yrows is a separate ddlaMalloc allocation on
// this handle. It is freed/null after row-to-tile redistribution; before then
// it remains caller-owned on failure. The full tiled transform is distributed
// over ranks, each history chunk is fetched once, and both redistributions use
// bounded panels. Descriptor sources/leading dimensions/padding are honored.
// All local failures are agreed before the next collective. panel_columns is
// a private test/benchmark control (0 selects 64 columns per replay rank).
template <class T>
void hb2st_replay_rows_cpu(const DdlaHandle_t &handle, int n, typename real_type<T>::type *&Yrows, int ldy,
                           const DistributedHistory<T> &history, const std::vector<T> &phases, T *Z,
                           const int *descZ, int &info, int panel_columns = 0);
} // namespace detail
} // namespace ddla
#endif
