#ifndef DDLA_HB2ST_REPLAY_GPU_H
#define DDLA_HB2ST_REPLAY_GPU_H

#include <ddla/ddla.h>
#include "distributed_history.h"
#include "hb2st_replay_wave_schedule.h"
#include "eig_2d_input.h"
#include <cstddef>
#include <vector>

namespace ddla { namespace detail {
template<class T> class Hb2stHouseholderCpu;

// Optional private driver storage. main is caller-owned contiguous storage,
// disjoint from input and output. The common prefix has P+1 entries spanning
// [0,n]. A rank owns floor(main_elements/n) main columns and at most one
// additional spill column, held separately by the helper. If owns_input is
// false, input may alias output; conversion completes collectively before
// output writes, and the consumed input is nulled without freeing it.
template<class T>
struct Hb2stReplayGpuWorkspace {
    T* main=nullptr;
    std::size_t main_elements=0;
    const std::vector<int>* column_offsets=nullptr;
    bool owns_input=true;
};

// Frozen replay geometry. The compact column prefix (columns: P+1 entries),
// when supplied, remains immutable for the plan's lifetime; main is the
// caller's contiguous strip when borrow_main. Numeric buffers are supplied
// only to run.
struct Hb2stReplayGpuPlan {
    Eig2dInputPlan input_2d{};
    int n=0,kd=1,P=1,rank=0,npr=1,npc=1,pr=0,pc=0,nr=0,nc=0,clo=0;
    int mb=1,nb=1,rsrc=0,csrc=0,lld=1,ldy=1,outrows=0,outcols=0;
    int main_columns=0,spill_columns=0,stride=1,width=1,rcmax=1,record_capacity=0;
    long long total=0;
    bool direct=false,borrow_main=false,use_waves=true,phases=false,pipeline=false,householder=false,householder_wy=false;
    void* main=nullptr;const int* columns=nullptr;
};
// Trusted composite geometry: no numerical pointer inspection or collectives.
// Set plan.main after allocation when a separate main strip is borrowed.
template<class T>Hb2stReplayGpuPlan hb2st_replay_gpu_geometry(const DdlaHandle_t&,int,int,long long,int,
    std::size_t,const int*,int=0,int=0,bool=true,bool=false,std::size_t=0,const int* = nullptr);
// Compact parent sizes know their own deterministic partition before the
// caller P+1 prefix exists. Set plan.columns and plan.main once they exist.
template<class T>Hb2stReplayGpuPlan hb2st_replay_gpu_compact_geometry(const DdlaHandle_t& h,int n,int kd,
    long long total,int ldy,std::size_t phase_count,const int* desc,int panel_columns,int replay_records,
    bool use_waves,std::size_t main_elements,int first,int count)
{
    auto p=hb2st_replay_gpu_geometry<T>(h,n,kd,total,ldy,phase_count,desc,panel_columns,replay_records,use_waves);
    p.borrow_main=true;p.clo=first;p.nc=count;
    p.main_columns=n>0?static_cast<int>(std::min<std::size_t>(count,main_elements/n)):count;
    p.spill_columns=count-p.main_columns;p.stride=std::max(1,p.main_columns);return p;
}
// columns (P+1 entries) is the compact column prefix when main is borrowed.
template<class T>Hb2stReplayGpuPlan hb2st_replay_gpu_check(const DdlaHandle_t&,int n,
    const typename real_type<T>::type* Y,int ldy,const DistributedHistory<T>& history,const T* phases,std::size_t phase_count,
    const T* Z,const int* desc,int=0,int=0,bool=true,T* main=nullptr,std::size_t main_elements=0,const int* columns=nullptr);
// Bytes of hb2st_replay_gpu_run.  Backend, with history and columns on this
// rank: the replay's j0 / rows, cosines and sines (rcmax each), with waves the
// wave tops, member counts (rcmax each) and offsets (rcmax + 1); P > 1: the
// own strip (n x max(nc, 1), unless main is borrowed) and the spill column
// (n), the send and receive panels (n x width each) and eight device integer
// arrays (max(P, width) each); then the phases (n - 1) when the plan has them.
// Host: one history chunk of records (record_capacity); with history and
// columns: cosines and sines (rcmax each), the level state last (n), levels
// (rcmax), starts and cursors (rcmax + 1 each) and indices (rcmax); P > 1:
// sc, sd, rc, rd, column starts, row counts and row offsets (P each), rows per
// process row (npr), columns per process column (npc), slots (width), column
// displacements (P) and columns (width); then with waves the wave tables
// (hb2st_replay_waves_bytes for rcmax records).  The pinned upload slots are
// Hb2stReplayGpuResources'.
template<class T>void hb2st_replay_gpu_workspace_size(const DdlaHandle_t&,const Hb2stReplayGpuPlan&,std::size_t* backend_bytes,
    std::size_t* host_bytes);

// Two pinned upload slots and their copy events (pipelined plans only).
template<class T>class Hb2stReplayGpuResources {
    DdlaHandle_t h_;void* storage_=nullptr;ddlaEvent_t copied_[2]{};bool used_[2]{};
    std::size_t stride_=0,capacity_=0;bool pending_=false;
public:
    Hb2stReplayGpuResources(const DdlaHandle_t&,const Hb2stReplayGpuPlan&);
    ~Hb2stReplayGpuResources();
    Hb2stReplayGpuResources(const Hb2stReplayGpuResources&)=delete;
    Hb2stReplayGpuResources& operator=(const Hb2stReplayGpuResources&)=delete;
    void wait(int);void upload(int,int,void*,const void*,std::size_t);void record(int);void finish();
    void completed() noexcept{used_[0]=used_[1]=false;pending_=false;}
};
// Replays the history into the strip, then redistributes it into Z.  phases:
// the n - 1 phases when the plan has them.  input_consumed (may be null) is
// set once every rank has finished reading Y.
// A Householder plan instead supplies an empty rotation history and its host
// reduction state. The same conversion/output path applies bounded reflector
// packets (or the private WY reference); no complete history is replicated on a device.
template<class T>void hb2st_replay_gpu_run(const DdlaHandle_t&,const Hb2stReplayGpuPlan&,Hb2stReplayGpuResources<T>&,
    const DistributedHistory<T>& history,const T* phases,const typename real_type<T>::type* Y,T* Z,int& info,
    bool* input_consumed,void* backend_work,std::size_t backend_bytes,void* host_work,std::size_t host_bytes,
    Hb2stHouseholderCpu<T>* householder=nullptr);

// Private consuming stage-4 operation. Without workspace, Yrows must be a
// separate allocation made by ddlaMalloc on this handle. On successful input redistribution it is
// freed and set to nullptr; Z's descriptor padding is preserved. The history
// is sharded on the handle communicator and fetched one bounded chunk at a
// time. panel_columns and replay_records are private test controls; zero
// selects the normal bounded sizes. Every rank participates, including ranks
// with no eigenvector columns or output entries.
// use_waves=false retains the independent level-kernel test path.
// Host workspace and schedule failures are agreed before the next collective.
// If input redistribution has not completed, Yrows remains caller-owned;
// otherwise it is already nullptr, including when a later stage throws.
template <class T>
void hb2st_replay_rows_gpu(const DdlaHandle_t& handle, int n,
                          typename real_type<T>::type*& Yrows, int ldy,
                          const DistributedHistory<T>& history,
                          const std::vector<T>& phases,
                          T* Z, const int* descZ, int& info,
                          int panel_columns = 0, int replay_records = 0,
                          bool use_waves = true,
                          const Hb2stReplayGpuWorkspace<T>* workspace = nullptr);

} }
#endif
