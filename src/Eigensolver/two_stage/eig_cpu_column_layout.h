#ifndef DDLA_EIG_CPU_COLUMN_LAYOUT_H
#define DDLA_EIG_CPU_COLUMN_LAYOUT_H

// Common row-slab to full-height column-strip mapping. The plan is rank-local
// geometry; cpu_replay_layout carves the layout's lists from the caller's
// scratch (cpu_replay_layout_bytes of them) and fills them.
#include <ddla/ddla.h>
#include "ddla_stream.h"
#include "plan_util.h"
#include "row_share.h"
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ddla { namespace detail {
inline int replay_count(long long count)
{
    if(count<0||count>std::numeric_limits<int>::max())throw std::overflow_error("CPU replay: MPI count overflow");
    return static_cast<int>(count);
}
// offsets[q] = counts[0] + ... + counts[q-1] for the `ranks` entries; returns the total.
inline std::size_t replay_displacements(const int* counts,int* offsets,int ranks)
{
    long long total=0;for(int q=0;q<ranks;++q){offsets[q]=replay_count(total);total+=counts[q];}
    replay_count(total);return static_cast<std::size_t>(total);
}
struct CpuReplayPlan {
    int n=0,nev=0,P=1,me=0,npr=1,npc=1,pr=0,pc=0,mb=1,nb=1,rsrc=0,csrc=0,lld=1;
    int nrl=0,ncs=0,outrows=0,maxcols=0,maxoutcols=0,width=64;char major='R';
    int rank(int r,int c)const noexcept{return major=='R'?r*npc+c:c*npr+r;}
    int column_size(int q)const noexcept{
        const int r=major=='R'?q/npc:q%npr,c=major=='R'?q%npc:q/npr;
        const long long all=num_loc(nev,nb,c,csrc,npc);return int(all*(r+1)/npr-all*r/npr);
    }
};
inline CpuReplayPlan cpu_replay_plan(const DdlaHandle_t& h,int n,const int* d,int panel=64)
{
    CpuReplayPlan p;p.n=n;p.nev=std::min(n,d[DDLA_N_]);p.npr=h->nprows_;p.npc=h->npcols_;p.pr=h->myprow_;p.pc=h->mypcol_;p.P=h->nprocs;p.me=h->myid;p.major=h->major;
    p.mb=d[DDLA_MB_];p.nb=d[DDLA_NB_];p.rsrc=d[DDLA_RSRC_];p.csrc=d[DDLA_CSRC_];p.lld=d[DDLA_LLD_];
    p.nrl=RowShare(n,p.P).count(p.me);p.outrows=num_loc(n,p.mb,p.pr,p.rsrc,p.npr);
    for(int q=0;q<p.P;++q)p.maxcols=std::max(p.maxcols,p.column_size(q));
    for(int c=0;c<p.npc;++c)p.maxoutcols=std::max(p.maxoutcols,num_loc(p.nev,p.nb,c,p.csrc,p.npc));
    p.ncs=p.column_size(p.me);p.width=std::max(1,std::min(panel>0?panel:64,std::max(p.maxcols,1)));return p;
}
// Bytes cpu_replay_layout takes, in carving order: the column offsets (P + 1)
// and lists (n), the owner and local index of every row (n each), the row
// counts and column ranks of the process rows (npr each), the first share
// index of every rank (P), the input panel's counts and displacements (sc,
// sd, rc, rd: P each) and the output panel's (osc, osd, orc, ord: npr each).
inline std::size_t cpu_replay_layout_bytes(const CpuReplayPlan& p)
{
    const std::size_t n=std::max(p.n,0);
    return padded_bytes<int>(std::size_t(p.P)+1)+3*padded_bytes<int>(n)+2*padded_bytes<int>(p.npr)
        +5*padded_bytes<int>(p.P)+4*padded_bytes<int>(p.npr);
}
struct CpuReplayLayout:CpuReplayPlan {
    RowShare rows;
    // rank q replays the global columns column_data[column_offsets[q] .. column_offsets[q+1])
    int *column_offsets=nullptr,*column_data=nullptr;
    int *row_owner=nullptr,*row_local=nullptr,*row_counts=nullptr,*col_ranks=nullptr,*begins=nullptr;
    int *sc=nullptr,*sd=nullptr,*rc=nullptr,*rd=nullptr,*osc=nullptr,*osd=nullptr,*orc=nullptr,*ord=nullptr;
    const int* column(int q)const{return column_data+column_offsets[q];}
    int column_count(int q)const{return column_offsets[q+1]-column_offsets[q];}
    int count(int q,int first)const{return std::max(0,std::min(width,column_count(q)-first));}
    std::pair<std::size_t,std::size_t> input(int first){const int mine=count(me,first);for(int q=0;q<P;++q){sc[q]=replay_count(static_cast<long long>(nrl)*count(q,first));rc[q]=replay_count(static_cast<long long>(rows.count(q))*mine);}return {replay_displacements(sc,sd,P),replay_displacements(rc,rd,P)};}
    int output_first(int q,int first)const{return std::max(0,std::min(column_count(q),first-begins[q]));}
    int output_count(int q,int first)const{return output_first(q,first+width)-output_first(q,first);}
    std::pair<std::size_t,std::size_t> output(int first){const int mine=output_count(me,first);for(int p=0;p<npr;++p){osc[p]=replay_count(static_cast<long long>(row_counts[p])*mine);orc[p]=replay_count(static_cast<long long>(outrows)*output_count(col_ranks[p],first));}return {replay_displacements(osc,osd,npr),replay_displacements(orc,ord,npr)};}
};
// The layout of plan p, its lists carved from cursor (cpu_replay_layout_bytes).
inline CpuReplayLayout cpu_replay_layout(const CpuReplayPlan& p,char*& cursor)
{
    CpuReplayLayout m;static_cast<CpuReplayPlan&>(m)=p;m.rows=RowShare(p.n,p.P);
    const std::size_t n=std::max(p.n,0);
    m.column_offsets=take<int>(cursor,std::size_t(p.P)+1);m.column_data=take<int>(cursor,n);
    m.row_owner=take<int>(cursor,n);m.row_local=take<int>(cursor,n);m.row_counts=take<int>(cursor,p.npr);m.col_ranks=take<int>(cursor,p.npr);m.begins=take<int>(cursor,p.P);
    m.sc=take<int>(cursor,p.P);m.sd=take<int>(cursor,p.P);m.rc=take<int>(cursor,p.P);m.rd=take<int>(cursor,p.P);
    m.osc=take<int>(cursor,p.npr);m.osd=take<int>(cursor,p.npr);m.orc=take<int>(cursor,p.npr);m.ord=take<int>(cursor,p.npr);
    int offset=0;for(int rank=0;rank<p.P;++rank){
        const int row=p.major=='R'?rank/p.npc:rank%p.npr,col=p.major=='R'?rank%p.npc:rank/p.npr;
        const long long all=num_loc(p.nev,p.nb,col,p.csrc,p.npc);const int begin=int(all*row/p.npr),count=p.column_size(rank);
        m.column_offsets[rank]=offset;m.begins[rank]=begin;
        for(int j=0;j<count;++j)m.column_data[offset+j]=indxl2g(begin+j,p.nb,col,p.csrc,p.npc);
        offset+=count;
    }
    m.column_offsets[p.P]=offset;
    for(int row=0;row<p.npr;++row){m.col_ranks[row]=p.rank(row,p.pc);m.row_counts[row]=num_loc(p.n,p.mb,row,p.rsrc,p.npr);}
    for(int i=0;i<p.n;++i){m.row_owner[i]=(i/p.mb+p.rsrc)%p.npr;m.row_local[i]=(i/p.mb/p.npr)*p.mb+i%p.mb;}
    return m;
}
} }
#endif
