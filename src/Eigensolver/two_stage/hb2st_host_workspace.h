#ifndef DDLA_HB2ST_HOST_WORKSPACE_H
#define DDLA_HB2ST_HOST_WORKSPACE_H
#include <ddla/ddla.h>
#include "hb2st_finish_panel.h"
#include "hb2st_sink.h"
#include "distributed_history.h"
#include "eig_2d_input.h"
#include <array>
#include <cstddef>
#include <memory>
#include <type_traits>

namespace ddla { namespace detail {
// Frozen CPU finish geometry. The callback packs a requested immutable Q_band
// column interval; its context is borrowed until the synchronous run returns.
struct Hb2stFinishHostPlan {
    Eig2dInputPlan input_2d{};
    int n=0,ldy=1,npr=1,npc=1,mypr=0,mypc=0,rank=0,P=1,nrl=0,maxcnt=0;
    int threads=1,kp=1,panel=1;char major='C';bool phases=false;
    std::array<int,DDLA_DLEN_> descriptor{};
};
template<class T>Hb2stFinishHostPlan hb2st_finish_host_geometry(const DdlaHandle_t&,int,int,const int*,int,
    int panel_columns=0,int contraction_columns=512,bool phases=false);
template<class T>Hb2stFinishHostPlan hb2st_finish_host_check(const DdlaHandle_t&,int n,
    const typename real_type<T>::type* Y,int ldy,const T* phases,std::size_t phase_count,T* Z,const int* desc,
    int threads,int panel_columns=0);
// Backend bytes (CPU handle) of hb2st_finish_host_run: the panel map
// (hb2st_finish_panel_map_bytes), 64 bytes of alignment slack, then the
// product / receive panel W and the send panel (P > 1), the phase-scaled Y
// panel (complex, a rank with rows), the packed Q_band columns A and the
// broadcast Y panel (P > 1).  n == 0 needs nothing.
template<class T>void hb2st_finish_host_workspace_size(const DdlaHandle_t&,const Hb2stFinishHostPlan&,std::size_t* backend_bytes);
// W = Q_band (D Y) in output-column panels, each redistributed into Z right
// away.  pack_columns(context, A, lo, count) writes the Q_band columns
// [lo, lo + count) of this rank's rows into A (column-major, ld nrl).
// phases: the n-1 phases when the plan has them.
template<class T>void hb2st_finish_host_run(const DdlaHandle_t&,const Hb2stFinishHostPlan&,
    const typename real_type<T>::type* Y,const T* phases,T* Z,void* context,void (*pack_columns)(void*,T*,int,int),
    void* backend_work,std::size_t backend_bytes);

struct Hb2stAccumHostPlan {
    int n=0,kd=1,threads=1,npr=1,npc=1,mypr=0,mypc=0,rank=0,P=1,nrl=0,row_lo=0;
    long long block_records=524288,total=0;int block_slots=3,rows_per_band=512;char major='C';bool profile=false;
    std::size_t record_capacity=0,slot_capacity=0;   // records of the largest block; chase slots
};
Hb2stAccumHostPlan hb2st_accum_host_check(const DdlaHandle_t&,int,int,int);
// Backend bytes (CPU handle) of a run of the CPU sink, ONE buffer from
// reset_run through finish_rows_run: 64 bytes of alignment slack and the
// Q_band tiles (live until finish_rows_run returned), then the planner's
// scratch (dead once drain_stage2_run returned): per block slot gjlo, gcnt,
// woff, cosines and sines (record_capacity each, sines twice for complex);
// the level state last (n+1), level and the sort keys (record_capacity each);
// the step generator's slot runs run_j0, run_len, run_w (slot_capacity each)
// and its domino lists gj, gc, first, last, next (record_capacity each).
template<class T>void hb2st_accum_host_workspace_size(const DdlaHandle_t&,const Hb2stAccumHostPlan&,std::size_t* backend_bytes);
// The front of that buffer which stays live until finish_rows_run returned
// (alignment slack and Q_band); a caller may overlay the rest after
// drain_stage2_run.
template<class T>void hb2st_accum_host_qband_workspace_size(const DdlaHandle_t&,const Hb2stAccumHostPlan&,std::size_t* backend_bytes);
// The CPU sink.  Its threads are created once, by hb2st_accum_host_prepare;
// Q_band and the planner's tables live in the buffer reset_run binds.
template<class T>class Hb2stAccumHostResources:public Hb2stSink<T> {
public:
    // Binds the hb2st_accum_host_workspace_size bytes for the next run.
    virtual void reset_run(void* backend_work,std::size_t backend_bytes)=0;
    // Stage 4 on the Q_band of this run: drains stage 2, then
    // hb2st_finish_host_run on backend_work (hb2st_finish_host_workspace_size
    // bytes, disjoint from the reset_run buffer's Q_band).
    virtual void finish_rows_run(const Hb2stFinishHostPlan&,const typename real_type<T>::type* Y,const T* phases,T* Z,int& info,
        void* backend_work,std::size_t backend_bytes)=0;
};
template<class T>std::unique_ptr<Hb2stAccumHostResources<T>> hb2st_accum_host_prepare(const DdlaHandle_t&,const Hb2stAccumHostPlan&);
} }
#endif
