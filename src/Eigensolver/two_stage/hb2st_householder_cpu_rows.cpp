#include "helpers.h"
#include "hb2st_householder_cpu_rows.h"
#include "hb2st_householder_cpu.h"
#include "hb2st_householder_paired.h"
#include "eig_cpu_column_layout.h"
#include "Eigensolver/eig_cpu_kernels.h"
#include "ddla_desc.h"
#include "mpi_datatype.h"
#include "ddla_connector.h"
#include "plan_util.h"

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>

namespace ddla { namespace detail {
namespace {
// The column strips and panels start on a 64-byte boundary (the tile
// kernels' vector rows); the size reserves the up to 64 bytes this skips.
constexpr std::size_t rows_alignment=64;
inline char* align_rows(char* p)
{return reinterpret_cast<char*>((reinterpret_cast<std::uintptr_t>(p)+rows_alignment-1)/rows_alignment*rows_alignment);}

// Explicitly begin object lifetimes in the carved scratch before assignments
// or MPI writes, including std::complex in C++17. All four scalar types are
// trivially destructible, so releasing storage needs no destructor loop.
// Complex default construction also first-touches pages in parallel.
template<class T> T* start_rows(T* data,std::size_t count,int threads)
{
    static_assert(std::is_trivially_destructible<T>::value,"rows require trivial scalar destruction");
    host_parallel_for(static_cast<long long>(count),4096,threads,[&](long long lo,long long hi) {
        for(long long i=lo;i<hi;++i) ::new (static_cast<void*>(data+i)) T;
    });
    return data;
}
template<class T> class ColumnStorage {
    T* data_=nullptr;
    int n_=0,ncols_=0;
public:
    using value_type=T;
    static constexpr bool tiled=false;
    void bind(T* data,std::size_t count,int n,int ncols,int threads) {
        n_=n;ncols_=ncols;data_=start_rows(data,count,threads);
    }
    void put(int col,int row,T value) {data_[row+static_cast<std::size_t>(col)*n_]=value;}
    T get(int col,int row) {return data_[row+static_cast<std::size_t>(col)*n_];}
    void check(Hb2stHouseholderCpu<T>& state){state.apply_columns_check(data_,n_,ncols_);}
    void apply(Hb2stHouseholderCpu<T>& state,void* work,std::size_t bytes) {state.apply_columns_run(data_,n_,ncols_,work,bytes);}
};

// Source columns are column-major; destination rows hold adjacent tile lanes.
// Process only this source-row intersection, leaving all other rows and lanes
// untouched. The caller owns validation and joins the parallel row pass.
template<class T,int W>
void pack_rows_to_tiles(typename real_type<T>::type* tiles,int n,int first,int count,
                        int row0,const typename real_type<T>::type* source,int ld,
                        const T* phases,int lo,int hi)
{
    using R=typename real_type<T>::type;
    constexpr bool is_real=std::is_same<T,R>::value;
    const std::size_t stride=static_cast<std::size_t>(n)*W*hhpair::planes<T>;
    for(int begin=0;begin<count;) {
        const int col=first+begin,lane=col%W;
        const int active=std::min(W-lane,count-begin);
        for(int i=lo;i<hi;++i) {
            const int row=row0+i;
            R* dst=tiles+static_cast<std::size_t>(col/W)*stride+
                static_cast<std::size_t>(row)*W*hhpair::planes<T>+lane;
            const R* src=source+i+static_cast<std::size_t>(begin)*ld;
            if(row==0 || !phases) {
                for(int k=0;k<active;++k) {
                    dst[k]=src[static_cast<std::size_t>(k)*ld];
                    if constexpr(!is_real) dst[W+k]=R(0);
                }
            } else if constexpr(is_real) {
                const R phase=phases[row-1];
                for(int k=0;k<active;++k) dst[k]=src[static_cast<std::size_t>(k)*ld]*phase;
            } else {
                const R pr=phases[row-1].real(),pi=phases[row-1].imag(),zero=R(0);
                for(int k=0;k<active;++k) {
                    const R value=src[static_cast<std::size_t>(k)*ld];
                    dst[k]=value*pr-zero*pi;
                    dst[W+k]=value*pi+zero*pr;
                }
            }
        }
        begin+=active;
    }
}

template<class T,int W> class TileStorage {
    using R=typename real_type<T>::type;
    R* data_=nullptr;
    int n_=0,ncols_=0;
    std::size_t stride_=0;
    R* at(int col,int row) {
        return data_+static_cast<std::size_t>(col/W)*stride_+
            static_cast<std::size_t>(row)*W*hhpair::planes<T>+col%W;
    }
public:
    using value_type=R;
    static constexpr bool tiled=true;
    static constexpr int tile_width=W;
    void bind(R* data,std::size_t count,int n,int ncols,int threads) {
        n_=n;ncols_=ncols;
        stride_=static_cast<std::size_t>(n)*W*hhpair::planes<T>;
        data_=start_rows(data,count,threads);
    }
    void put(int col,int row,T value) {
        R* dst=at(col,row);
        *dst=hh::real_part(value);
        if constexpr(!std::is_same<T,R>::value)dst[W]=value.imag();
    }
    void pack_input(int first,int count,int row0,const R* source,int ld,
                    const T* phases,int lo,int hi) {
        pack_rows_to_tiles<T,W>(data_,n_,first,count,row0,source,ld,phases,lo,hi);
    }
    T get(int col,int row) {
        const R* src=at(col,row);
        if constexpr(std::is_same<T,R>::value)return *src;
        else return T(*src,src[W]);
    }
    void check(Hb2stHouseholderCpu<T>& state){state.apply_tiles_check(data_,W,ncols_);}
    void apply(Hb2stHouseholderCpu<T>& state,void*,std::size_t) {state.apply_tiles_run(data_,W,ncols_);}
};

// A local preparation or worker failure must be agreed before the next
// exchange. The application thread alone calls MPI.
template<class F>
void rows_collective_local(MPI_Comm comm, F&& work)
{
    std::exception_ptr failure;
    try { work(); } catch(...) { failure=std::current_exception(); }
    int ok=failure?0:1,all_ok=0;
    MPI_CHECK(MPI_Allreduce(&ok,&all_ok,1,MPI_INT,MPI_MIN,comm));
    if(!all_ok) {
        if(failure) std::rethrow_exception(failure);
        throw std::runtime_error("CPU Householder rows failed on another rank");
    }
}

// Elements of the column strips: n x ncs of T, or the split-real tiles (R) of
// a paired state; false when the tile layout overflows.
template<class T> bool rows_strip_count(const HhRowsCpuPlan& p,std::size_t& count)
{
    count=std::size_t(p.map.n)*std::size_t(p.map.ncs);
    const bool ok=!p.tile_width||hhpair::layout_elements(p.map.n,p.map.ncs,p.tile_width,hhpair::planes<T>,count);
    count=std::max(count,std::size_t(1));return ok;
}
// Entries of the panel buffers. Input: this rank's rows of every rank's panel
// columns (send) and all rows of my panel columns (receive). Output: all rows
// of my panel columns (send) and my rows of my process column's panel columns
// (receive).
inline void rows_input_counts(const CpuReplayPlan& m,std::size_t* send,std::size_t* recv)
{
    long long columns=0;for(int rank=0;rank<m.P;++rank)columns+=std::min(m.width,m.column_size(rank));
    *send=std::max(std::size_t(1),std::size_t(replay_count(static_cast<long long>(m.nrl)*columns)));
    *recv=std::max(std::size_t(1),std::size_t(replay_count(static_cast<long long>(m.n)*std::min(m.width,m.ncs))));
}
inline void rows_output_counts(const CpuReplayPlan& m,std::size_t* send,std::size_t* recv)
{
    *send=std::max(std::size_t(1),std::size_t(replay_count(static_cast<long long>(m.n)*std::min(m.width,m.ncs))));
    *recv=std::max(std::size_t(1),std::size_t(replay_count(static_cast<long long>(m.outrows)*std::min(m.width,num_loc(m.n,m.nb,m.pc,m.csrc,m.npc)))));
}
}

template<class T> HhRowsCpuPlan hh_rows_cpu_geometry(const DdlaHandle_t& h,int n,int ldy,const int* desc,std::size_t phase_count,
    bool paired,int threads)
{
    HhRowsCpuPlan p;p.map=cpu_replay_plan(h,n,desc,64);p.ldy=ldy;p.threads=threads;p.tile_width=paired?hhpair::pick_width<T>(p.map.ncs,threads):0;
    p.phase_count=phase_count;return p;
}
template<class T> HhRowsCpuPlan hh_rows_cpu_check(const DdlaHandle_t& h,int n,const typename real_type<T>::type* Y,int ldy,
    const Hb2stHouseholderCpu<T>& state,const T* phases,std::size_t phase_count,T* Z,const int* desc)
{
    using R=typename real_type<T>::type;
    if(!h||h->backend!=DdlaBackend::CPU)throw std::invalid_argument("CPU Householder rows: CPU handle required");
    HhRowsCpuPlan plan;
    rows_collective_local(h->comm,[&]{check_desc(desc,h);if(n<0||n>desc[DDLA_M_]||n>desc[DDLA_N_]||state.order()!=n)throw std::invalid_argument("CPU Householder rows: incompatible matrix order");});
    const int signature[5]={n,desc[DDLA_MB_],desc[DDLA_NB_],desc[DDLA_RSRC_],desc[DDLA_CSRC_]};int lo[5],hi[5];
    MPI_CHECK(MPI_Allreduce(signature,lo,5,MPI_INT,MPI_MIN,h->comm));MPI_CHECK(MPI_Allreduce(signature,hi,5,MPI_INT,MPI_MAX,h->comm));
    rows_collective_local(h->comm,[&]{
        for(int i=0;i<5;++i)if(lo[i]!=hi[i])throw std::invalid_argument("CPU Householder rows: inconsistent metadata");
        plan=hh_rows_cpu_geometry<T>(h,n,ldy,desc,phase_count,state.paired_replay(),state.threads());const auto& p=plan.map;
        const auto nc=std::size_t(num_loc(n,p.nb,p.pc,p.csrc,p.npc));
        if(ldy<std::max(p.nrl,1)||(n&&p.nrl&&!Y)||(p.outrows&&nc&&!Z)||(phase_count&&(!phases||phase_count<std::size_t(std::max(n-1,0)))))throw std::invalid_argument("CPU Householder rows: incompatible input/output/phases");
        constexpr auto limit=std::numeric_limits<std::size_t>::max();
        if(std::size_t(ldy)>limit/std::size_t(std::max(n,1))/sizeof(R)||std::size_t(p.lld)>limit/std::max<std::size_t>(nc,1)/sizeof(T)||
           std::size_t(n)>limit/std::size_t(std::max(p.ncs,1))/sizeof(T))throw std::overflow_error("CPU Householder rows: matrix size overflow");
        const auto ay=std::size_t(ldy)*std::size_t(n)*sizeof(R),bz=(nc?(nc-1)*std::size_t(p.lld)+std::size_t(p.outrows):0)*sizeof(T);
        if(Y&&Z&&n&&p.outrows&&nc){if(memory_overlaps(Y,ay,Z,bz))throw std::invalid_argument("CPU Householder rows: input/output overlap");}
    });return plan;
}
template<class T> void hh_rows_cpu_workspace_size(const DdlaHandle_t& h,const HhRowsCpuPlan& p,std::size_t* backend_bytes)
{
    using R=typename real_type<T>::type;const auto& m=p.map;*backend_bytes=0;if(!m.n)return;
    std::size_t strips=0;if(!rows_strip_count<T>(p,strips))throw std::overflow_error("CPU Householder tiled workspace overflow");
    std::size_t input=0,output=0,apply=0;
    if(p.input_2d.enabled)input=eig_2d_input_cpu_bytes<R>(h,p.input_2d);
    else if(m.P>1){std::size_t send=0,recv=0;rows_input_counts(m,&send,&recv);input=padded_bytes<R>(send)+padded_bytes<R>(recv);}
    if(m.npr>1){std::size_t send=0,recv=0;rows_output_counts(m,&send,&recv);output=padded_bytes<T>(send)+padded_bytes<T>(recv);}
    if(!p.tile_width)hh_cpu_apply_columns_workspace_size<T>(h,m.ncs,&apply);
    *backend_bytes=cpu_replay_layout_bytes(m)+rows_alignment+(p.tile_width?padded_bytes<R>(strips):padded_bytes<T>(strips))+std::max({input,output,apply});
}

template<bool Consume,class T,class Input>
void hh_rows_cpu_execute(const DdlaHandle_t& h,const HhRowsCpuPlan& plan,Input& Y,Hb2stHouseholderCpu<T>& state,
    const T* phases,T* Z,int& info,void* backend_work,std::size_t backend_bytes)
{
    using R=typename real_type<T>::type;constexpr bool is_real=std::is_same<T,R>::value;
    const int n=plan.map.n,ldy=plan.ldy,threads=plan.threads;info=0;
    if(!n){if constexpr(Consume)rows_collective_local(h->comm,[&]{DDLA_CHECK(ddlaFree(h,Y));Y=nullptr;});return;}
    if(!plan.phase_count)phases=nullptr;
    char* const base=static_cast<char*>(backend_work);char* front=base;
    CpuReplayLayout map=cpu_replay_layout(plan.map,front);front=align_rows(front);
    std::size_t strip_count=0;rows_strip_count<T>(plan,strip_count);
    // Both layouts consume rows directly into their final back-transform
    // storage; there is no full column-major/tiled conversion copy.
    auto run=[&](auto& columns) {
    // The panels and the scratch of apply_columns_run share the stage behind the strips.
    char* stage=front;auto* const strips=take<typename std::decay_t<decltype(columns)>::value_type>(stage,strip_count);
    assert(std::size_t(stage-base)<=backend_bytes);
    rows_collective_local(h->comm,[&] {
        columns.bind(strips,strip_count,n,map.ncs,threads);
    });
    [[maybe_unused]] auto put=[&](int col,int row,R value) {
        T x;
        if constexpr(is_real) x=row==0 || !phases?value:value*phases[row-1];
        else if(row==0 || !phases) x=T(value,R(0));
        else {
            const R pr=phases[row-1].real(),pi=phases[row-1].imag(),zero=R(0);
            x=T(value*pr-zero*pi,value*pi+zero*pr);
        }
        columns.put(col,row,x);
    };
    if(plan.input_2d.enabled) {
        eig_2d_input_cpu_run<R>(h,plan.input_2d,Y,
            [&](int q){return map.column_count(q);},[&](int q,int k){return map.column(q)[k];},
            [&](int first,int count,const R* panel,int ld){
                if constexpr(std::decay_t<decltype(columns)>::tiled)columns.pack_input(first,count,0,panel,ld,phases,0,n);
                else host_parallel_for(count,4,threads,[&](long long lo,long long hi){
                    for(long long j=lo;j<hi;++j)for(int i=0;i<n;++i)put(first+int(j),i,panel[i+std::size_t(j)*ld]);
                });
            },stage,backend_bytes-std::size_t(stage-base));
    } else if(map.P==1) {
        rows_collective_local(h->comm,[&] {
            if constexpr(std::decay_t<decltype(columns)>::tiled) {
                constexpr int width=std::decay_t<decltype(columns)>::tile_width;
                const int tiles=map.ncs/width+(map.ncs%width!=0);
                host_parallel_for(tiles,1,threads,[&](long long lo,long long hi) {
                    for(long long tile=lo;tile<hi;++tile) {
                        const int first=static_cast<int>(tile)*width,count=std::min(width,map.ncs-first);
                        columns.pack_input(first,count,0,Y+static_cast<std::size_t>(first)*ldy,ldy,phases,0,n);
                    }
                });
            } else {
            host_parallel_for(map.ncs,4,threads,[&](long long lo,long long hi) {
                for(long long j=lo;j<hi;++j)
                    for(int i=0;i<n;++i) put(static_cast<int>(j),i,Y[i+static_cast<std::size_t>(j)*ldy]);
            });
            }
        });
    } else {
        char* cursor=stage;std::size_t nsend=0,nrecv=0;rows_input_counts(plan.map,&nsend,&nrecv);
        R* const send=take<R>(cursor,nsend);R* const recv=take<R>(cursor,nrecv);
        assert(std::size_t(cursor-base)<=backend_bytes);
        rows_collective_local(h->comm,[&] {start_rows(send,nsend,threads);start_rows(recv,nrecv,threads);});
        for(int first=0;first<map.maxcols;first+=map.width) {
            rows_collective_local(h->comm,[&] {
                map.input(first);
                for(int q=0;q<map.P;++q) {
                    const int count=map.count(q,first);const auto columns=map.column(q);
                    host_parallel_for(count,4,threads,[&](long long lo,long long hi) {
                        for(long long k=lo;k<hi;++k) {
                            const int j=columns[first+static_cast<int>(k)];
                            if(map.nrl) std::memcpy(send+map.sd[q]+static_cast<std::size_t>(k)*map.nrl,
                                Y+static_cast<std::size_t>(j)*ldy,static_cast<std::size_t>(map.nrl)*sizeof(R));
                        }
                    });
                }
            });
            MPI_CHECK(MPI_Alltoallv(send,map.sc,map.sd,mpi_datatype<R>(),
                recv,map.rc,map.rd,mpi_datatype<R>(),h->comm));
            rows_collective_local(h->comm,[&] {
                if constexpr(std::decay_t<decltype(columns)>::tiled) {
                    const int mine=map.count(map.me,first);
                    if(mine>0) host_parallel_for(n,32,threads,[&](long long lo,long long hi) {
                        for(int q=0;q<map.P;++q) {
                            const int nr=map.rows.count(q),r0=map.rows.lo(q);
                            const int begin=std::max(0,static_cast<int>(lo)-r0);
                            const int end=std::min(nr,static_cast<int>(hi)-r0);
                            if(begin<end) columns.pack_input(first,mine,r0,recv+map.rd[q],nr,phases,begin,end);
                        }
                    });
                } else {
                host_parallel_for(map.count(map.me,first),4,threads,[&](long long lo,long long hi) {
                    for(long long k=lo;k<hi;++k)
                        for(int q=0;q<map.P;++q) {
                            const int nr=map.rows.count(q),r0=map.rows.lo(q);
                            const R* src=recv+map.rd[q]+static_cast<std::size_t>(k)*nr;
                            for(int i=0;i<nr;++i) put(first+static_cast<int>(k),r0+i,src[i]);
                        }
                });
                }
            });
        }
    }
    // hh_rows_cpu consumes Y at its original boundary; hh_rows_cpu_run
    // retains the caller input and performs no free.
    if constexpr(Consume)rows_collective_local(h->comm,[&]{DDLA_CHECK(ddlaFree(h,Y));Y=nullptr;});
    const std::size_t stage_bytes=backend_bytes-std::size_t(stage-base);
    if constexpr(Consume){try{columns.check(state);columns.apply(state,stage,stage_bytes);}catch(...){state.release();throw;}}
    else columns.apply(state,stage,stage_bytes);
    auto get=[&](int col,int row)->T {return columns.get(col,row);};
    if(map.npr==1) {
        const auto columns=map.column(map.me);
        rows_collective_local(h->comm,[&] {
            host_parallel_for(map.ncs,4,threads,[&](long long lo,long long hi) {
                for(long long k=lo;k<hi;++k) {
                    const int j=columns[k],lj=(j/map.nb/map.npc)*map.nb+j%map.nb;
                    for(int i=0;i<n;++i) Z[i+static_cast<std::size_t>(lj)*map.lld]=get(static_cast<int>(k),i);
                }
            });
        });
    } else {
        char* cursor=stage;std::size_t nsend=0,nrecv=0;rows_output_counts(plan.map,&nsend,&nrecv);
        T* const send=take<T>(cursor,nsend);T* const recv=take<T>(cursor,nrecv);
        assert(std::size_t(cursor-base)<=backend_bytes);
        rows_collective_local(h->comm,[&] {start_rows(send,nsend,threads);start_rows(recv,nrecv,threads);});
        for(int first=0;first<map.maxoutcols;first+=map.width) {
            const int mine=map.output_count(map.me,first),start=map.output_first(map.me,first);
            rows_collective_local(h->comm,[&] {
                map.output(first);
                host_parallel_for(n,32,threads,[&](long long lo,long long hi) {
                    for(long long i=lo;i<hi;++i) {
                        T* dst=send+map.osd[map.row_owner[i]]+static_cast<std::size_t>(map.row_local[i])*mine;
                        for(int k=0;k<mine;++k) dst[k]=get(start+k,static_cast<int>(i));
                    }
                });
            });
            MPI_CHECK(MPI_Alltoallv(send,map.osc,map.osd,mpi_datatype<T>(),
                recv,map.orc,map.ord,mpi_datatype<T>(),h->col_comm));
            rows_collective_local(h->comm,[&] {
                for(int p=0;p<map.npr;++p) {
                    const int q=map.col_ranks[p],count=map.output_count(q,first),start=map.output_first(q,first);const auto columns=map.column(q);
                    host_parallel_for(count,4,threads,[&](long long lo,long long hi) {
                        for(long long k=lo;k<hi;++k) {
                            const int j=columns[start+static_cast<int>(k)],lj=(j/map.nb/map.npc)*map.nb+j%map.nb;
                            for(int i=0;i<map.outrows;++i)
                                Z[i+static_cast<std::size_t>(lj)*map.lld]=recv[map.ord[p]+static_cast<std::size_t>(i)*count+k];
                        }
                    });
                }
            });
        }
    }
    };
    if(plan.tile_width) {
        switch(plan.tile_width) {
        case 8:{TileStorage<T,8> columns;run(columns);break;}
        case 16:{TileStorage<T,16> columns;run(columns);break;}
        case 32:{TileStorage<T,32> columns;run(columns);break;}
        default:{TileStorage<T,64> columns;run(columns);break;}
        }
    } else {
        ColumnStorage<T> columns;run(columns);
    }
}

template<class T>
void hh_rows_cpu_run(const DdlaHandle_t& h,const HhRowsCpuPlan& p,const typename real_type<T>::type* Y,Hb2stHouseholderCpu<T>& state,
    const T* phases,T* Z,int& info,void* backend_work,std::size_t backend_bytes)
{hh_rows_cpu_execute<false,T>(h,p,Y,state,phases,Z,info,backend_work,backend_bytes);}

template<class T>
void hh_rows_cpu(const DdlaHandle_t& h,int n,typename real_type<T>::type*& Y,int ldy,Hb2stHouseholderCpu<T>& state,
    const std::vector<T>& phases,T* Z,const int* desc,int& info)
{
    const auto p=hh_rows_cpu_check<T>(h,n,Y,ldy,state,phases.data(),phases.size(),Z,desc);
    std::size_t backend_bytes=0;ddla::vector<char> backend_work(h);
    rows_collective_local(h->comm,[&]{hh_rows_cpu_workspace_size<T>(h,p,&backend_bytes);backend_work.resize(backend_bytes);});
    hh_rows_cpu_execute<true,T>(h,p,Y,state,phases.data(),Z,info,backend_work.data(),backend_bytes);
}

#define DDLA_INST_HH_ROWS(T) \
    template void hh_rows_cpu<T>(const DdlaHandle_t&,int,real_type<T>::type*&,int, \
        Hb2stHouseholderCpu<T>&,const std::vector<T>&,T*,const int*,int&); \
    template HhRowsCpuPlan hh_rows_cpu_geometry<T>(const DdlaHandle_t&,int,int,const int*,std::size_t,bool,int); \
    template HhRowsCpuPlan hh_rows_cpu_check<T>(const DdlaHandle_t&,int,const real_type<T>::type*,int,const Hb2stHouseholderCpu<T>&,const T*,std::size_t,T*,const int*); \
    template void hh_rows_cpu_workspace_size<T>(const DdlaHandle_t&,const HhRowsCpuPlan&,std::size_t*); \
    template void hh_rows_cpu_run<T>(const DdlaHandle_t&,const HhRowsCpuPlan&,const real_type<T>::type*,Hb2stHouseholderCpu<T>&,const T*,T*,int&,void*,std::size_t);
DDLA_INST_HH_ROWS(float)
DDLA_INST_HH_ROWS(double)
DDLA_INST_HH_ROWS(std::complex<float>)
DDLA_INST_HH_ROWS(std::complex<double>)
#undef DDLA_INST_HH_ROWS
} }
