#ifndef DDLA_HB2ST_REPLAY_WAVE_SCHEDULE_H
#define DDLA_HB2ST_REPLAY_WAVE_SCHEDULE_H

#include "plan_util.h"

#include <algorithm>
#include <cstddef>

namespace ddla { namespace detail {

// One bounded history interval, regrouped into the same domino runs used by
// Hb2stAccum and the CPU replay. Each member acts on (top-p-1, top-p).
// Forward table order preserves dependencies; reverse replay visits batches
// backward and members backward. Windows within one batch are disjoint.
// Cursor/record types are templates so this schedule can be checked without
// MPI, CUDA, or matrix data.  The tables live in caller scratch
// (hb2st_replay_waves_bytes, carved by hb2st_replay_waves): after build(),
// wave w has top[w], count[w] members members[offset[w] ..] and batch b is
// waves [batches[b], batches[b+1]), b < nbatches.
struct Hb2stReplayWaves {
    int *top=nullptr,*count=nullptr,*offset=nullptr,*members=nullptr,*batches=nullptr;
    int *run_top=nullptr,*run_count=nullptr,*run_wave=nullptr,*wave=nullptr,*position=nullptr,*owner=nullptr;
    int n=0,slots=0,capacity=0;   // owner: n entries, run_*: slots, per-record tables: capacity
    int waves=0,nbatches=0;

    template<class Record,class Cursor>
    bool build(int n_,int kd,const Record* records,int size,Cursor cursor,int first_offset,
               int skip,int cap)
    {
        waves=0; nbatches=0;
        if(size<0 || size>capacity || n_!=n) return false;
        if(size==0) { offset[0]=0; batches[0]=0; return true; }
        if(kd<=1 || cap<1 || cursor.n!=n || cursor.kd!=kd || first_offset<0 || skip<0) return false;
        const int limit=std::min(cap,std::min(n-1,kd));
        std::fill(run_count,run_count+slots,0);
        int t0=first_offset;
        while(skip>0 && !cursor.done()) {
            const int step=cursor.count();
            if(t0>step) return false;
            const int take=std::min(skip,step-t0);
            t0+=take; skip-=take;
            if(t0==step) { cursor.next(); t0=0; }
        }
        if(skip) return false;
        int q=0;
        while(q<size && !cursor.done()) {
            const int step=cursor.count();
            if(t0>step || step>slots) return false;
            for(int t=t0;t<step && q<size;++t,++q) {
                const long long j=cursor.first()+static_cast<long long>(t)*(kd+1LL);
                if(j<1 || j>=n || records[q].j0!=j) return false;
                int w=0,p=0;
                if(run_count[t]>0 && j==run_top[t]-run_count[t] && run_count[t]<limit) {
                    w=run_wave[t]; p=run_count[t]++;
                    count[w]=run_count[t];
                } else {
                    w=waves++;
                    top[w]=static_cast<int>(j); count[w]=1;
                    run_top[t]=static_cast<int>(j); run_count[t]=1; run_wave[t]=w;
                }
                wave[q]=w; position[q]=p;
            }
            cursor.next(); t0=0;
        }
        if(q!=size) return false;
        offset[0]=0;
        for(int w=0;w<waves;++w) offset[w+1]=offset[w]+count[w];
        if(offset[waves]!=size) return false;
        for(int r=0;r<size;++r) members[offset[wave[r]]+position[r]]=r;

        std::fill(owner,owner+n,-1);
        int first=0;
        while(first<waves) {
            batches[nbatches++]=first;
            int end=first;
            while(end<waves && end-first<512) {
                bool clash=false;
                for(int r=top[end]-count[end];r<=top[end];++r)
                    if(owner[r]==first) { clash=true; break; }
                if(clash) break;
                for(int r=top[end]-count[end];r<=top[end];++r) owner[r]=first;
                ++end;
            }
            first=end;
        }
        batches[nbatches]=waves;
        return true;
    }
};

// Bytes hb2st_replay_waves takes: top, count, members, wave, position
// (capacity each), offset and batches (capacity + 1 each), run_top, run_count,
// run_wave (one per chase slot) and owner (n).
inline std::size_t hb2st_replay_waves_bytes(int n,int kd,int capacity)
{
    const std::size_t records=static_cast<std::size_t>(capacity);
    const std::size_t slots=static_cast<std::size_t>(n/(static_cast<long long>(kd)+1)+2);
    return 5*padded_bytes<int>(records)+2*padded_bytes<int>(records+1)+3*padded_bytes<int>(slots)+
           padded_bytes<int>(static_cast<std::size_t>(n));
}

// Tables for intervals of at most capacity records, carved from cursor in the
// order hb2st_replay_waves_bytes counts them.
inline Hb2stReplayWaves hb2st_replay_waves(int n,int kd,int capacity,char*& cursor)
{
    Hb2stReplayWaves w;w.n=n;w.capacity=capacity;w.slots=static_cast<int>(n/(static_cast<long long>(kd)+1)+2);
    const std::size_t records=static_cast<std::size_t>(capacity);
    w.top=take<int>(cursor,records);w.count=take<int>(cursor,records);w.members=take<int>(cursor,records);
    w.wave=take<int>(cursor,records);w.position=take<int>(cursor,records);
    w.offset=take<int>(cursor,records+1);w.batches=take<int>(cursor,records+1);
    w.run_top=take<int>(cursor,w.slots);w.run_count=take<int>(cursor,w.slots);w.run_wave=take<int>(cursor,w.slots);
    w.owner=take<int>(cursor,static_cast<std::size_t>(n));
    return w;
}

} }
#endif
