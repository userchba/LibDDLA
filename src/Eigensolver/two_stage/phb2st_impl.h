#ifndef DDLA_PHB2ST_IMPL_H
#define DDLA_PHB2ST_IMPL_H

#include <ddla/ddla.h>
#include "plan_util.h"
#include "dc_threads.h"
#include "hb2st_sink.h"
#include "distributed_history.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ddla {
namespace spectral_impl {
struct Phb2stBandPlan {
    DdlaBackend backend{};int n=0,band=1,nb=1,lld=1;
    int nprows=1,npcols=1,myprow=0,mypcol=0;
    std::array<int,DDLA_DLEN_> descriptor{};
};
Phb2stBandPlan phb2st_band_geometry(const DdlaHandle_t&,DdlaBackend,int,int,const int*);
Phb2stBandPlan phb2st_band_check(const DdlaHandle_t&,DdlaBackend,int,int,const void*,const int*);
// Elements of the full band: (band+1) x n, column-major.
inline std::size_t phb2st_band_count(const Phb2stBandPlan& p)
{return (std::size_t(p.band)+1)*std::size_t(p.n);}
// A chase band starts on a 64-byte boundary of host scratch; a malloc'd
// buffer is only 16-aligned, so the sizes reserve phb2st_band_alignment
// bytes of slack and the runs round their cursor up.
constexpr std::size_t phb2st_band_alignment=64;
inline char* phb2st_band_align(char* p)
{
    const auto at=reinterpret_cast<std::uintptr_t>(p);
    return p+(phb2st_band_alignment-at%phb2st_band_alignment)%phb2st_band_alignment;
}
// Backend bytes of phb2st_band_run, in carving order: the band entries this
// rank owns (zero elsewhere) and the grid sum of them (band count each).
template<class T>void phb2st_band_workspace_size(const Phb2stBandPlan& p,std::size_t* backend_bytes)
{*backend_bytes=p.n?2*detail::padded_bytes<T>(phb2st_band_count(p)):0;}
// The full band on every rank, written to the host array band (band count
// elements). Collective over the grid.
template<DdlaBackend B,class T>void phb2st_band_run(const DdlaHandle_t&,const Phb2stBandPlan&,const T* A,T* band,
    void* backend_work,std::size_t backend_bytes);

// The replicated host chase; its band and numerical outputs are the caller's.
// The parent checker chooses this geometry after freezing its execution policy.
struct Phb2stHostPlan {int n=0,band=1,threads=1;bool records=false,bounded=false;long long record_capacity=0;};
// Host bytes of phb2st_host_run (none for n == 0), in carving order: the
// rotation cosines and sines of every column (n+2 each); with records on
// more than one thread, the per-thread record counts (threads) and record
// buffers (n+1 records per thread).
template<class T>void phb2st_host_workspace_size(const Phb2stHostPlan& p,std::size_t* host_bytes)
{
    *host_bytes=0;if(!p.n)return;const std::size_t scratch=std::size_t(p.n)+2;
    *host_bytes=detail::padded_bytes<typename real_type<T>::type>(scratch)+detail::padded_bytes<T>(scratch);
    if(p.records&&p.threads>1)*host_bytes+=detail::padded_bytes<std::size_t>(std::size_t(p.threads))+
        detail::padded_bytes<hb2st_rot<T>>(std::size_t(p.threads)*(std::size_t(p.n)+1));
}
// The chase of the (band+1) x n band (destroyed) into d, e and, for complex T,
// the phase_count phases. records: the ring of p.record_capacity records
// (bounded), the whole stream, or unused (no records). A bounded chase hands
// every step to publish(publish_context, end, ring, last).
template<class T>void phb2st_host_run(const Phb2stHostPlan&,detail::DcHostTeam*,T* band,
    typename real_type<T>::type* d,typename real_type<T>::type* e,hb2st_rot<T>* records,T* phases,std::size_t& phase_count,
    std::atomic<long long>* progress,void* publish_context,void (*publish)(void*,long long,const hb2st_rot<T>*,bool),
    void* host_work,std::size_t host_bytes);
struct Phb2stTeamPlan {
    int n=0,band=1,node_rank=0,node_size=1,tpr=1,participants=1,fallback_threads=1,sink_roles=0;
    long long ring_request=0;bool enabled=true,consumers=false,elastic=false,profile=false,prefault=false,debug_maps=false;
};
// The team window's record area: capacity slots (a ring of window.ring
// slots when that is nonzero), valid below the published watermark emitted.
template<class T>struct Phb2stTeamStream {
    hb2st_rot<T>* records=nullptr;long long capacity=0;const std::atomic<long long>* emitted=nullptr;std::atomic<int>* aborted=nullptr;
    StreamWindow window{};int chase_threads=0;
};
template<class T>class Phb2stTeamResources {
public:
    virtual ~Phb2stTeamResources()=default;
    virtual bool available() const noexcept=0;
    virtual detail::DcHostTeam& workers() noexcept=0;
    // Stages the (band+1) x n band (read on the node leader) for the next run.
    virtual void reset_run(const T* band)=0;
    virtual Phb2stTeamStream<T> stream() noexcept=0;
    virtual T* band_data() noexcept=0;
    virtual void run(Hb2stSink<T>*,void*,void (*)(void*,long long),long long)=0;
    // d (n), e (n-1), the phase_count phases and, when records is not null,
    // the whole stream.
    virtual void outputs_run(typename real_type<T>::type*,typename real_type<T>::type*,T* phases,std::size_t& phase_count,
        hb2st_rot<T>* records=nullptr)=0;
    // MPI mappings are retired explicitly and collectively outside execution.
    virtual void release()=0;
};
// Maps the plan's shared window (when the node team can run). Collective.
template<class T>std::unique_ptr<Phb2stTeamResources<T>> phb2st_team_prepare(const DdlaHandle_t&,const Phb2stTeamPlan&);

// Prepared topology contains MPI communicators and immutable rank routing.
// It never owns a numerical band or retained rotation history.
class Phb2stTopology {
public:
    virtual ~Phb2stTopology()=default;
    virtual int node_rank() const noexcept=0;
    virtual int node_size() const noexcept=0;
    virtual int groups() const noexcept=0;
    virtual const void* state() const noexcept=0;
    virtual void release()=0;
};
std::unique_ptr<Phb2stTopology> phb2st_topology_prepare(const DdlaHandle_t&);
// indices: the band entries of A this rank owns (it sends them all);
// received: the entries of its core it collects (node leaders only).
struct Phb2stSpatialPlan {
    Phb2stBandPlan input{};const void* topology=nullptr;
    int rank=0,ranks=1,node_rank=0,node_size=1,group=0,groups=1;
    int core_lo=0,core_hi=0,offset=0,width=0,halo_steps=1,halo=0;
    bool keep=false,sink=false;long long chunk=0,ring_capacity=0;
    std::size_t indices=0,received=0;
};
Phb2stSpatialPlan phb2st_spatial_geometry(const DdlaHandle_t&,const Phb2stBandPlan&,const Phb2stTopology&,bool,bool,long long);
// Elements of the compact band a rank keeps: width x (band+1), at least one.
inline std::size_t phb2st_spatial_band_count(const Phb2stSpatialPlan& p)
{return std::max(std::size_t(1),std::size_t(p.width)*(std::size_t(p.input.band)+1));}
// Bytes of phb2st_spatial_gather_run. Host, in carving order: the
// all-to-all counts and displacements (sc, sd, rc, rd: ranks each), the
// indices of the owned entries (p.indices), the send and receive buffers
// (max(1, p.indices) and max(1, p.received) scalars). Backend (GPU only):
// the indices and the extracted entries (max(1, p.indices) each).
template<class T>void phb2st_spatial_gather_workspace_size(const Phb2stSpatialPlan&,std::size_t* backend_bytes,
    std::size_t* host_bytes);
// Bytes of phb2st_spatial_run. Host, in carving order: the compact band
// (phb2st_spatial_band_count, on a 64-byte boundary), then one region that holds the gather's
// scratch and then, with several groups, the chase's: the record ring
// (ring_capacity), followed by one region that holds during the chase the
// halo exchange (send/receive ranges and counts: 8 lists of groups ints, the
// send and receive bytes; node leaders only), the rotation cosines and sines
// (width+2 each) and, with a ring, the record exchange (send and receive
// buffers of one chunk plus a step of records, 5 lists of ranks ints), and
// after the chase the tail (counts and displacements: ranks each, the core's
// and the whole two diagonals: max(1, 2 x core) and 2n scalars). Backend:
// the gather's.
template<class T>void phb2st_spatial_workspace_size(const Phb2stSpatialPlan&,std::size_t* backend_bytes,std::size_t* host_bytes);
// The node leader's compact band (phb2st_spatial_band_count elements) of the
// block-cyclic A. Collective over the handle's communicator.
template<DdlaBackend B,class T>void phb2st_spatial_gather_run(const DdlaHandle_t&,const Phb2stSpatialPlan&,const T* A,T* band,
    void* backend_work,std::size_t backend_bytes,void* host_work,std::size_t host_bytes);
// The chase split over the node leaders (p.groups > 1). A kept stream goes
// to history (prepared for the plan's geometry); phases: n-1 for complex T.
template<DdlaBackend B,class T>void phb2st_spatial_run(const DdlaHandle_t&,const Phb2stSpatialPlan&,const T* A,
    typename real_type<T>::type*,typename real_type<T>::type*,detail::DistributedHistory<T>& history,T* phases,
    std::size_t& phase_count,Hb2stSink<T>*,void* backend_work,std::size_t backend_bytes,void* host_work,std::size_t host_bytes);

enum class Phb2stOutput
{
    None,
    History,
    Sink
};
struct Phb2stSinkPolicy
{
    bool ring_allowed = true;
    long long chunk = 1;
    int max_roles = 0;
    bool roles_in_team = false;
};
struct Phb2stPlan
{
    Phb2stBandPlan input{};
    Phb2stSpatialPlan spatial{};
    Phb2stTeamPlan team{};
    Phb2stHostPlan local{};
    Phb2stOutput output = Phb2stOutput::None;
    Phb2stSinkPolicy sink{};
    long long rotations = 0;
    bool spatial_enabled = false;
    bool device_chase = false;
};
Phb2stPlan phb2st_policy(const DdlaHandle_t &, const Phb2stBandPlan &, const Phb2stTopology &, Phb2stOutput,
                         Phb2stSinkPolicy = {});
// phases has room for phase_count entries (n-1 for complex T).
template <class T>
Phb2stPlan phb2st_check(const DdlaHandle_t &, DdlaBackend, int, int, const T *, const int *,
                        typename real_type<T>::type *, typename real_type<T>::type *, T *phases,
                        std::size_t phase_count, const Phb2stTopology &, Phb2stOutput, Phb2stSinkPolicy = {});
// Bytes of phb2st_run. With several node groups: phb2st_spatial_workspace_size.
// Otherwise host, in carving order: the full band ((band+1) x n, on a 64-byte
// boundary), then one region reused by the band gather and by the bounded
// record ring and chase scratch.
// Backend bytes hold the gather workspace. The node team's window and retained
// history are resources (Phb2stResources), not scratch.
template <class T>
void phb2st_workspace_size(const Phb2stPlan &p, std::size_t *backend_bytes, std::size_t *host_bytes)
{
    *backend_bytes = *host_bytes = 0;
    const auto &b = p.input;
    if (!b.n)
        return;
    if (p.spatial_enabled)
    {
        phb2st_spatial_workspace_size<T>(p.spatial, backend_bytes, host_bytes);
        return;
    }
    std::size_t gather = 0, chase = 0;
    phb2st_spatial_gather_workspace_size<T>(p.spatial, backend_bytes, &gather);
    phb2st_host_workspace_size<T>(p.local, &chase);
    if (p.local.bounded)
        chase += detail::padded_bytes<hb2st_rot<T>>(std::size_t(p.local.record_capacity));
    *host_bytes =
        phb2st_band_alignment + detail::padded_bytes<T>(phb2st_band_count(b)) + std::max(gather, chase);
}
template <class T> class Phb2stResources
{
    std::unique_ptr<Phb2stTeamResources<T>> team_;
    detail::DistributedHistory<T> owned_{};
    detail::DistributedHistory<T> *history_ = &owned_;
#if DDLA_HAS_GPU
    int device_levels_=16,device_blocks_=1,device_tile_columns_=1,device_first_=0,device_last_=0,device_core_first_=0,device_core_last_=0,device_max_events_=0;
    std::unique_ptr<ddla::vector<T>> device_state_,device_output_,device_tiles_,device_packet_;
    std::unique_ptr<ddla::vector<int>> device_failure_;
    std::vector<T> host_device_state_,host_device_packet_;
    void device_setup(const DdlaHandle_t&,const Phb2stPlan&);
#endif

  public:
    // History output retains the compact cyclic history either in
    // external_history (prepared by the caller for the plan's geometry; never
    // released here) or in one this constructor prepares. Either stays live,
    // for the replay, while this object exists. Collective.
    Phb2stResources(const DdlaHandle_t &, const Phb2stPlan &,
                    detail::DistributedHistory<T> *external_history = nullptr);
    Phb2stResources(const Phb2stResources &) = delete;
    Phb2stResources &operator=(const Phb2stResources &) = delete;
#if DDLA_HAS_GPU
    // GPU formation/update only; reuse the native cyclic rotation history and replay.
    void device_run(const DdlaHandle_t&,const Phb2stPlan&,T* shared_band,hb2st_rot<T>* ring,long long capacity);
#endif
    detail::DistributedHistory<T> &history() noexcept
    {
        return *history_;
    }
    const detail::DistributedHistory<T> &history() const noexcept
    {
        return *history_;
    }
    Phb2stTeamResources<T> *team() noexcept
    {
        return team_.get();
    }
    void release()
    {
        if (team_)
            team_->release();
    }
};
// Stage 2 on the caller's scratch. phases: n-1 for complex T, phase_count of
// them written. sink consumes a sink output with its own prepared scratch.
template <DdlaBackend B, class T>
void phb2st_run(const DdlaHandle_t &, const Phb2stPlan &, Phb2stResources<T> &, const T *A,
                typename real_type<T>::type *d, typename real_type<T>::type *e, T *phases,
                std::size_t &phase_count, Hb2stSink<T> *sink, void *backend_work, std::size_t backend_bytes,
                void *host_work, std::size_t host_bytes);
} // namespace spectral_impl

namespace detail {
// Private test hook. Zero uses physical shared-memory nodes; a positive
// value partitions communicator ranks into that many consecutive groups.
// Every rank must set the same value before a collective reduction call.
int set_phb2st_test_partitions(int groups);
} // namespace detail
} // namespace ddla

namespace ddla
{
// Owning band reduction with uniquely distributed retained coefficients.
// keep_history=false retains no rotations. d/e/phases are replicated linear arrays.
template <DdlaBackend Backend, class T>
void phb2st_node(const DdlaHandle_t &, int n, int band, T *A, const int *descA,
                 typename real_type<T>::type *d, typename real_type<T>::type *e,
                 detail::DistributedHistory<T> &history, std::vector<T> &phases, bool keep_history,
                 int &info);
} // namespace ddla
#endif
