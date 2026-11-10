#include "ddla_connector.h"
// Included by phb2st.cpp inside namespace ddla. This private path retains
// one compact band core plus conservative temporal halos per node leader.
namespace detail {
namespace { std::atomic<int> phb2st_test_groups{0}; }
int set_phb2st_test_partitions(int groups) {
    if (groups < 0) throw std::invalid_argument("phb2st test partitions must be nonnegative");
    return phb2st_test_groups.exchange(groups);
}
}
namespace {
template<class Function> void spatial_allocate(MPI_Comm comm,Function&& function) {
    std::exception_ptr failure;
    try { function(); } catch(...) { failure=std::current_exception(); }
    int ok=failure?0:1,all=0; MPI_CHECK(MPI_Allreduce(&ok,&all,1,MPI_INT,MPI_MIN,comm));
    if(!all) { if(failure) std::rethrow_exception(failure); throw std::runtime_error("phb2st spatial allocation failed on another rank"); }
}
// Checked size arithmetic of the ring capacities.
inline std::size_t chase_size_add(std::size_t a,std::size_t b) {
    if(b>std::numeric_limits<std::size_t>::max()-a) throw std::length_error("phb2st: size overflow");
    return a+b;
}
inline std::size_t chase_size_mul(std::size_t a,std::size_t b) {
    if(a!=0 && b>std::numeric_limits<std::size_t>::max()/a) throw std::length_error("phb2st: size overflow");
    return a*b;
}

struct SpatialGrid {
    MPI_Comm world=MPI_COMM_NULL, node=MPI_COMM_NULL, leaders=MPI_COMM_NULL;
    int rank=0, ranks=1, local_rank=0, local_size=1, group=0, groups=1;
    std::vector<int> roots, coords;
    void setup(MPI_Comm comm,int prow,int pcol) {
        world=comm; MPI_CHECK(MPI_Comm_rank(world,&rank)); MPI_CHECK(MPI_Comm_size(world,&ranks));
        int forced=detail::phb2st_test_groups.load(),low=0,high=0;
        MPI_CHECK(MPI_Allreduce(&forced,&low,1,MPI_INT,MPI_MIN,world));
        MPI_CHECK(MPI_Allreduce(&forced,&high,1,MPI_INT,MPI_MAX,world));
        if(low!=high) throw std::invalid_argument("inconsistent phb2st test partitions");
        if(forced) MPI_CHECK(MPI_Comm_split(world,static_cast<int>(static_cast<long long>(rank)*std::min(forced,ranks)/ranks),rank,&node));
        else MPI_CHECK(MPI_Comm_split_type(world,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&node));
        MPI_CHECK(MPI_Comm_rank(node,&local_rank));MPI_CHECK(MPI_Comm_size(node,&local_size));
        int root=rank; MPI_CHECK(MPI_Bcast(&root,1,MPI_INT,0,node));
        spatial_allocate(world,[&]{roots.resize(ranks);}); MPI_CHECK(MPI_Allgather(&root,1,MPI_INT,roots.data(),1,MPI_INT,world));
        std::sort(roots.begin(),roots.end()); roots.erase(std::unique(roots.begin(),roots.end()),roots.end());
        groups=static_cast<int>(roots.size()); group=static_cast<int>(std::lower_bound(roots.begin(),roots.end(),root)-roots.begin());
        MPI_CHECK(MPI_Comm_split(world,local_rank==0?0:MPI_UNDEFINED,rank,&leaders));
        int c[2]={prow,pcol}; spatial_allocate(world,[&]{coords.resize(static_cast<std::size_t>(2)*ranks);});
        MPI_CHECK(MPI_Allgather(c,2,MPI_INT,coords.data(),2,MPI_INT,world));
    }
    void release() {
        if(leaders!=MPI_COMM_NULL) MPI_CHECK(MPI_Comm_free(&leaders));
        if(node!=MPI_COMM_NULL) MPI_CHECK(MPI_Comm_free(&node));
    }
    int lo(int q,int n) const { return static_cast<int>(static_cast<long long>(n)*q/groups); }
    int hi(int q,int n) const { return lo(q+1,n); }
    int owner(int col,int n) const {
        // Empty cores are possible in forced small-matrix tests.
        return static_cast<int>((static_cast<long long>(col+1)*groups-1)/n);
    }
};

inline int spatial_bytes(std::size_t elements,std::size_t size) {
    if(elements>static_cast<std::size_t>(std::numeric_limits<int>::max())/size)
        throw std::overflow_error("phb2st spatial transfer exceeds MPI count");
    return static_cast<int>(elements*size);
}
// displs[p] = counts[0] + ... + counts[p-1] for the `entries` counts; returns the total.
inline int spatial_displacements(const int* counts,int* displs,int entries) {
    long long total=0;
    for(int p=0;p<entries;++p) { displs[p]=static_cast<int>(total); total+=counts[p];
        if(total>std::numeric_limits<int>::max()) throw std::overflow_error("phb2st spatial displacement overflow"); }
    return static_cast<int>(total);
}

template<class Put>void spatial_visit(int rowrank,int colrank,int first,int last,int kd,int nb,int lld,int npr,int npc,int rsrc,int csrc,Put&& put){
    for(int col=first;col<last;++col){if((csrc+col/nb)%npc!=colrank)continue;const int lj=nb*(col/(nb*npc))+col%nb;
        for(int gi=std::max(0,col-kd);gi<=col;++gi){if((rsrc+gi/nb)%npr!=rowrank)continue;const int li=nb*(gi/(nb*npr))+gi%nb;
            put(col,kd+gi-col,static_cast<std::size_t>(lj)*lld+li);}}
}
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
template<class T> __global__ void spatial_extract_kernel(const T* A,const std::size_t* index,std::size_t count,T* values) {
    for(std::size_t p=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;p<count;p+=static_cast<std::size_t>(gridDim.x)*blockDim.x)
        values[p]=A[index[p]];
}
#endif

// Every rank sends the band entries of A it owns (indices: their local
// positions) to the leader of the group whose core holds them; the leaders
// assemble their compact band (zero outside the gathered core). receive
// holds receive_capacity scalars; device_indices/device_values (GPU) one
// slot per index.
template<DdlaBackend Backend,class T>
void gather_spatial_band(const DdlaHandle_t& handle,int n,int kd,const T* A,const int* desc,
    const SpatialGrid& grid,const SpatialChase<T>& domain,T* band,int* sc,int* sd,int* rc,int* rd,
    std::size_t* indices,T* send,T* receive,std::size_t receive_capacity,std::size_t* device_indices,T* device_values)
{
    int pr=0,pc=0,npr=0,npc=0;ddlaGetGridCoords(handle,pr,pc);ddlaGetGridDims(handle,npr,npc);
    const int nb=desc[DDLA_NB_],lld=desc[DDLA_LLD_],rsrc=desc[DDLA_RSRC_],csrc=desc[DDLA_CSRC_];
    auto visit=[&](int rowrank,int colrank,int first,int last,auto&& put){spatial_visit(rowrank,colrank,first,last,kd,nb,lld,npr,npc,rsrc,csrc,put);};
    std::size_t count=0;
    spatial_allocate(grid.world,[&] {
        std::fill_n(sc,grid.ranks,0);std::fill_n(rc,grid.ranks,0);
        for(int q=0;q<grid.groups;++q) {
            std::size_t owned=0;
            visit(pr,pc,grid.lo(q,n),grid.hi(q,n),[&](int,int,std::size_t index){indices[count++]=index;++owned;});
            sc[grid.roots[q]]=spatial_bytes(owned,sizeof(T));
        }
        spatial_displacements(sc,sd,grid.ranks);
        std::fill_n(band,std::max<std::size_t>(1,static_cast<std::size_t>(domain.width)*(kd+1)),T(0));
    });
    MPI_CHECK(MPI_Alltoall(sc,1,MPI_INT,rc,1,MPI_INT,grid.world));
    spatial_allocate(grid.world,[&] {
        const int bytes=spatial_displacements(rc,rd,grid.ranks);
        if(static_cast<std::size_t>(bytes)/sizeof(T)>receive_capacity) throw std::logic_error("spatial band receive capacity");
    });
    if constexpr(Backend==DdlaBackend::CPU) {
        for(std::size_t p=0;p<count;++p) send[p]=A[indices[p]];
    } else {
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
        spatial_allocate(grid.world,[&] {
        if(count) {
            DDLA_CHECK(ddlaMemcpy(handle,device_indices,indices,count*sizeof(std::size_t)));
            spatial_extract_kernel<T><<<std::min(1024,static_cast<int>((count+255)/256)),256,0,static_cast<gpuStream_t>(handle->stream)>>>(A,device_indices,count,device_values);
            DDLA_CHECK(ddlaGetLastError(handle));
            DDLA_CHECK(ddlaMemcpy(handle,send,device_values,count*sizeof(T)));
            DDLA_CHECK(ddlaSynchronize(handle));
        }
        });
#else
        (void)device_indices;(void)device_values;
#endif
    }
    MPI_CHECK(MPI_Alltoallv(send,sc,sd,MPI_BYTE,receive,rc,rd,MPI_BYTE,grid.world));
    if(grid.local_rank==0) {
        for(int source=0;source<grid.ranks;++source) {
            std::size_t p=static_cast<std::size_t>(rd[source])/sizeof(T);
            visit(grid.coords[2*source],grid.coords[2*source+1],domain.core_lo,domain.core_hi,
                  [&](int col,int row,std::size_t){band[static_cast<std::size_t>(col-domain.offset)*(kd+1)+row]=receive[p++];});
            if(p*sizeof(T)!=static_cast<std::size_t>(rd[source]+rc[source])) throw std::logic_error("spatial band receive mapping");
        }
    }
}

// The leaders' halo exchange before the chase steps and the tridiagonal tail
// after them. The halo lists and buffers (node leaders only) come from the
// chase cursor; the tail's from the tail cursor, which overlays them.
template<class T> struct SpatialHalo {
    using R=typename real_type<T>::type;
    const SpatialGrid& grid;
    SpatialChase<T>& domain;
    int n,kd,steps,halo;
    long long step=0,total_steps;
    int *send_begin=nullptr,*send_end=nullptr,*recv_begin=nullptr,*recv_end=nullptr,*sc=nullptr,*sd=nullptr,*rc=nullptr,*rd=nullptr;
    unsigned char *send=nullptr,*receive=nullptr;
    int *tail_counts=nullptr,*tail_displacements=nullptr;T *tail_local=nullptr,*tail_global=nullptr;
    SpatialHalo(const SpatialGrid& g,SpatialChase<T>& d,int n_,int kd_,char*& chase,char*& tail):grid(g),domain(d),n(n_),kd(kd_){
        const std::size_t lists=grid.local_rank==0?static_cast<std::size_t>(grid.groups):0;
        send_begin=detail::take<int>(chase,lists);send_end=detail::take<int>(chase,lists);
        recv_begin=detail::take<int>(chase,lists);recv_end=detail::take<int>(chase,lists);
        sc=detail::take<int>(chase,lists);sd=detail::take<int>(chase,lists);rc=detail::take<int>(chase,lists);rd=detail::take<int>(chase,lists);
        initialize(chase);
        tail_counts=detail::take<int>(tail,grid.ranks);tail_displacements=detail::take<int>(tail,grid.ranks);
        tail_local=detail::take<T>(tail,std::max(1,2*(domain.core_hi-domain.core_lo)));
        tail_global=detail::take<T>(tail,static_cast<std::size_t>(2)*n);
    }
    void initialize(char*& chase){
        const int smallest=n/grid.groups;
        steps=std::max(1,int(smallest/(4LL*(std::int64_t(kd)+1))));halo=steps*(kd+1);
        total_steps=static_cast<long long>(std::max(0,n-2))*std::min(n-1,kd);
        const int lo=grid.lo(grid.group,n),hi=grid.hi(grid.group,n);
        domain.n=n; domain.kd=kd;
        domain.core_lo=lo; domain.core_hi=grid.local_rank==0?hi:lo;
        domain.offset=grid.local_rank==0?std::max(0,lo-halo):lo;
        domain.width=grid.local_rank==0&&hi>lo?int(std::min<std::int64_t>(n,std::int64_t(hi)+halo))-domain.offset:0;
        if(grid.local_rank!=0) return;
        const std::size_t bytes=(std::size_t(kd)+2)*sizeof(T)+sizeof(R);
        for(int q=0;q<grid.groups;++q) {
            const int qlo=grid.lo(q,n),qhi=grid.hi(q,n);
            send_begin[q]=std::max(lo,std::max(0,qlo-halo)); send_end[q]=std::min(hi,int(std::min<std::int64_t>(n,std::int64_t(qhi)+halo)));
            recv_begin[q]=std::max(qlo,domain.offset); recv_end[q]=std::min(qhi,domain.offset+domain.width);
            if(q==grid.group || qlo==qhi || lo==hi) send_end[q]=send_begin[q],recv_end[q]=recv_begin[q];
            sc[q]=spatial_bytes(std::max(0,send_end[q]-send_begin[q]),bytes);
            rc[q]=spatial_bytes(std::max(0,recv_end[q]-recv_begin[q]),bytes);
        }
        send=detail::take<unsigned char>(chase,std::max(1,spatial_displacements(sc,sd,grid.groups)));
        receive=detail::take<unsigned char>(chase,std::max(1,spatial_displacements(rc,rd,grid.groups)));
    }
    void before(T* band,R* dc,T* wc) {
        if(grid.local_rank!=0) return;
        if(step%steps==0) {
            for(int q=0;q<grid.groups;++q) {
                std::size_t out=sd[q];
                for(int col=send_begin[q];col<send_end[q];++col) {
                    const std::size_t local=col-domain.offset;
                    std::memcpy(send+out,band+local*(kd+1),(kd+1)*sizeof(T)); out+=(kd+1)*sizeof(T);
                    std::memcpy(send+out,dc+local,sizeof(R)); out+=sizeof(R);
                    std::memcpy(send+out,wc+local,sizeof(T)); out+=sizeof(T);
                }
            }
            MPI_CHECK(MPI_Alltoallv(send,sc,sd,MPI_BYTE,receive,rc,rd,MPI_BYTE,grid.leaders));
            for(int q=0;q<grid.groups;++q) {
                std::size_t in=rd[q];
                for(int col=recv_begin[q];col<recv_end[q];++col) {
                    const std::size_t local=col-domain.offset;
                    std::memcpy(band+local*(kd+1),receive+in,(kd+1)*sizeof(T)); in+=(kd+1)*sizeof(T);
                    std::memcpy(dc+local,receive+in,sizeof(R)); in+=sizeof(R);
                    std::memcpy(wc+local,receive+in,sizeof(T)); in+=sizeof(T);
                }
            }
        }
        const int remain=static_cast<int>(std::min<long long>(steps-step%steps,total_steps-step));
        domain.valid_lo=std::max(0,domain.core_lo-remain*(kd+1));
        domain.valid_hi=int(std::min<std::int64_t>(n,std::int64_t(domain.core_hi)+std::int64_t(remain)*(kd+1)));
        ++step;
    }
    // Returns the number of phases written.
    std::size_t finish(T* band,R* d,R* e,T* phases) {
        // Only two diagonals are gathered. The original sequential phase
        // recurrence is unchanged; no floating reduction changes signed zero.
        int* counts=tail_counts;int* displs=tail_displacements;T* local=tail_local;T* tridiagonal=tail_global;
        std::fill_n(counts,grid.ranks,0);
        for(int q=0;q<grid.groups;++q) counts[grid.roots[q]]=spatial_bytes(2ULL*(grid.hi(q,n)-grid.lo(q,n)),sizeof(T));
        spatial_displacements(counts,displs,grid.ranks);
        for(int col=domain.core_lo;col<domain.core_hi;++col) {
            local[2*(col-domain.core_lo)]=band[static_cast<std::size_t>(col-domain.offset)*(kd+1)+kd-1];
            local[2*(col-domain.core_lo)+1]=band[static_cast<std::size_t>(col-domain.offset)*(kd+1)+kd];
        }
        MPI_CHECK(MPI_Allgatherv(local,counts[grid.rank],MPI_BYTE,tridiagonal,counts,displs,MPI_BYTE,grid.world));
        return host_chase_tail<T>(tridiagonal,1,n,d,e,phases);
    }
};

// The leaders' records of a bounded batch, either collected in stream order
// on every rank (sink or kept ring) or sent to the history's chunk owners.
template<class T> struct SpatialRecords {
    const SpatialGrid& grid;
    const SpatialChase<T>& domain;
    int n,kd;
    detail::ChaseCursor cursor;
    long long handed=0;
    std::size_t bound;   // records per exchange buffer: one history chunk plus a step
    hb2st_rot<T> *send=nullptr,*receive=nullptr;
    int *counts=nullptr,*displs=nullptr,*position=nullptr,*send_counts=nullptr,*send_displs=nullptr;
    SpatialRecords(const SpatialGrid& g,const SpatialChase<T>& d,int n_,int kd_,char*& w):grid(g),domain(d),n(n_),kd(kd_),cursor(n,kd){
        bound=detail::DistributedHistory<T>::chunk_size+static_cast<std::size_t>(n)/(kd+1)+2;
        spatial_bytes(bound,sizeof(hb2st_rot<T>));
        send=detail::take<hb2st_rot<T>>(w,bound);receive=detail::take<hb2st_rot<T>>(w,bound);
        counts=detail::take<int>(w,grid.ranks);displs=detail::take<int>(w,grid.ranks);position=detail::take<int>(w,grid.ranks);
        send_counts=detail::take<int>(w,grid.ranks);send_displs=detail::take<int>(w,grid.ranks);
    }
    // Enumerate only this leader's unique core records. Idle ranks advance
    // the integer cursor by steps, without walking all global coefficients.
    template<class Function> detail::ChaseCursor own_records(long long end,Function&& function) const {
        auto scan=cursor;
        while(scan.base<end && !scan.done()) {
            const int first=scan.first(),stride=kd+1,nr=scan.count();
            if(domain.core_lo<domain.core_hi) {
                const int begin=std::max(0,(domain.core_lo-first+stride-1)/stride);
                const int last=std::min(nr,std::max(0,(domain.core_hi-first+stride-1)/stride));
                for(int t=begin;t<last;++t) function(scan.base+t,first+t*stride);
            }
            scan.next();
        }
        if(scan.base!=end || end-handed>static_cast<long long>(bound)) throw std::logic_error("spatial record batch boundary");
        return scan;
    }
    void collect(long long end,hb2st_rot<T>* ring,long long capacity) {
        std::size_t filled=0;
        own_records(end,[&](long long g,int){send[filled++]=ring[g%capacity];});
        const int bytes=spatial_bytes(filled,sizeof(hb2st_rot<T>));
        MPI_CHECK(MPI_Allgather(&bytes,1,MPI_INT,counts,1,MPI_INT,grid.world));
        const int total=spatial_displacements(counts,displs,grid.ranks);
        if(total!=spatial_bytes(static_cast<std::size_t>(end-handed),sizeof(hb2st_rot<T>))) throw std::logic_error("spatial record ownership coverage");
        MPI_CHECK(MPI_Allgatherv(send,bytes,MPI_BYTE,receive,counts,displs,MPI_BYTE,grid.world));
        for(int p=0;p<grid.ranks;++p) position[p]=displs[p]/sizeof(hb2st_rot<T>);
        while(cursor.base<end && !cursor.done()) {
            for(int t=0;t<cursor.count();++t) {
                const int j=cursor.first()+t*(kd+1),owner=grid.roots[grid.owner(j,n)];
                const auto& record=receive[static_cast<std::size_t>(position[owner]++)];
                if(record.j0!=j) throw std::logic_error("spatial record order");
                ring[(cursor.base+t)%capacity]=record;
            }
            cursor.next();
        }
        handed=end;
    }
    template<class History>void distribute(long long end,const hb2st_rot<T>* ring,long long capacity,History& history) {
        std::fill_n(send_counts,grid.ranks,0);
        const auto finish=own_records(end,[&](long long g,int) {
            const int owner=static_cast<int>((g/history.chunk_size)%grid.ranks);
            send_counts[owner]+=static_cast<int>(sizeof(hb2st_rot<T>));
        });
        spatial_displacements(send_counts,send_displs,grid.ranks);
        for(int p=0;p<grid.ranks;++p) position[p]=send_displs[p]/sizeof(hb2st_rot<T>);
        own_records(end,[&](long long g,int) {
            const int owner=static_cast<int>((g/history.chunk_size)%grid.ranks);
            send[static_cast<std::size_t>(position[owner]++)]=ring[g%capacity];
        });
        MPI_CHECK(MPI_Alltoall(send_counts,1,MPI_INT,counts,1,MPI_INT,grid.world));
        const int received=spatial_displacements(counts,displs,grid.ranks);
        if(received>spatial_bytes(bound,sizeof(hb2st_rot<T>))) throw std::logic_error("spatial owner receive capacity");
        MPI_CHECK(MPI_Alltoallv(send,send_counts,send_displs,MPI_BYTE,
                               receive,counts,displs,MPI_BYTE,grid.world));
        for(int p=0;p<grid.ranks;++p) position[p]=displs[p]/sizeof(hb2st_rot<T>);
        // Only chunks owned by this rank need their global positions walked.
        // The others are skipped through the integer step count.
        while(cursor.base<end && !cursor.done()) {
            const long long step_end=cursor.base+cursor.count();
            for(long long g=cursor.base;g<step_end;) {
                const long long q=g/history.chunk_size;
                const long long chunk_end=std::min(step_end,(q+1)*history.chunk_size);
                if(q%grid.ranks==grid.rank) {
                    for(;g<chunk_end;++g) {
                        const int j=cursor.first()+static_cast<int>(g-cursor.base)*(kd+1);
                        const int source=grid.roots[grid.owner(j,n)];
                        const auto& record=receive[static_cast<std::size_t>(position[source]++)];
                        if(record.j0!=j) throw std::logic_error("spatial owner record order");
                        history.store_owned(g,record);
                    }
                } else g=chunk_end;
            }
            cursor.next();
        }
        if(cursor.base!=finish.base) throw std::logic_error("spatial owner cursor mismatch");
        for(int p=0;p<grid.ranks;++p)
            if(position[p]*sizeof(hb2st_rot<T>)!=static_cast<std::size_t>(displs[p]+counts[p]))
                throw std::logic_error("spatial owner receive coverage");
        handed=end;
    }
};
} // namespace

namespace spectral_impl {
namespace {
class PreparedSpatialTopology final:public Phb2stTopology {
    SpatialGrid grid_;
public:
    void prepare(const DdlaHandle_t& h){try{grid_.setup(h->comm,h->myprow_,h->mypcol_);}catch(...){grid_.release();throw;}}
    int node_rank() const noexcept override{return grid_.local_rank;}
    int node_size() const noexcept override{return grid_.local_size;}
    int groups() const noexcept override{return grid_.groups;}
    const void* state() const noexcept override{return &grid_;}
    void release() override{grid_.release();}
};
}
std::unique_ptr<Phb2stTopology> phb2st_topology_prepare(const DdlaHandle_t& h)
{
    std::unique_ptr<PreparedSpatialTopology> result;std::exception_ptr error;
    try{result.reset(new PreparedSpatialTopology);}catch(...){error=std::current_exception();}
    int failed=error?1:0;MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,&failed,1,MPI_INT,MPI_MAX,h->comm));
    if(failed){if(error)std::rethrow_exception(error);throw std::runtime_error("phb2st: topology preparation failed on another rank");}
    result->prepare(h);return result;
}
Phb2stSpatialPlan phb2st_spatial_geometry(const DdlaHandle_t&,const Phb2stBandPlan& b,const Phb2stTopology& topology,
    bool keep,bool sink,long long chunk)
{
    const auto& g=*static_cast<const SpatialGrid*>(topology.state());Phb2stSpatialPlan p;p.input=b;p.topology=topology.state();
    p.rank=g.rank;p.ranks=g.ranks;p.node_rank=g.local_rank;p.node_size=g.local_size;p.group=g.group;p.groups=g.groups;p.keep=keep;p.sink=sink;
    p.core_lo=g.lo(g.group,b.n);p.core_hi=g.local_rank==0?g.hi(g.group,b.n):p.core_lo;
    p.halo_steps=std::max(1,int((b.n/g.groups)/(4LL*(std::int64_t(b.band)+1))));p.halo=p.halo_steps*(b.band+1);
    p.offset=g.local_rank==0?std::max(0,p.core_lo-p.halo):p.core_lo;
    p.width=g.local_rank==0&&p.core_hi>p.core_lo?int(std::min<std::int64_t>(b.n,std::int64_t(p.core_hi)+p.halo))-p.offset:0;
    if(g.groups==1){p.core_lo=0;p.core_hi=g.local_rank==0?b.n:0;p.offset=0;p.width=p.core_hi;}
    if((keep||sink)&&b.band>1&&b.n>2){const auto maxt=std::size_t(b.n)/(std::size_t(b.band)+1)+2;
        const auto batch=std::size_t(detail::DistributedHistory<float>::chunk_size);p.chunk=sink?std::max(1LL,chunk):0;
        const auto capacity=sink?std::max(chase_size_mul(8,std::size_t(p.chunk)),
            chase_size_add(chase_size_add(std::size_t(p.chunk),batch),chase_size_mul(2,maxt))):chase_size_add(batch,maxt);
        if(capacity>std::size_t(LLONG_MAX))throw std::overflow_error("phb2st: spatial ring exceeds int64");p.ring_capacity=static_cast<long long>(capacity);}
    // The gather: every band entry this rank owns (all groups' columns),
    // and the entries of the core a leader assembles.
    spatial_visit(b.myprow,b.mypcol,0,b.n,b.band,b.nb,b.lld,b.nprows,b.npcols,0,0,[&](int,int,std::size_t){++p.indices;});
    for(int col=p.core_lo;col<p.core_hi;++col)p.received+=std::size_t(std::min(b.band,col))+1;
    return p;
}
template<class T>void phb2st_spatial_gather_workspace_size(const Phb2stSpatialPlan& p,std::size_t* backend_bytes,std::size_t* host_bytes)
{
    const std::size_t ranks=std::size_t(p.ranks),values=std::max(std::size_t(1),p.indices);
    spatial_bytes(p.indices,sizeof(T));spatial_bytes(p.received,sizeof(T));
    *host_bytes=4*detail::padded_bytes<int>(ranks)+detail::padded_bytes<std::size_t>(p.indices)+detail::padded_bytes<T>(values)+
        detail::padded_bytes<T>(std::max(std::size_t(1),p.received));
    *backend_bytes=p.input.backend==DdlaBackend::GPU?detail::padded_bytes<std::size_t>(values)+detail::padded_bytes<T>(values):0;
}
template<class T>void phb2st_spatial_workspace_size(const Phb2stSpatialPlan& p,std::size_t* backend_bytes,std::size_t* host_bytes)
{
    using R=typename real_type<T>::type;const auto& g=*static_cast<const SpatialGrid*>(p.topology);const auto& b=p.input;
    std::size_t gather=0,chase=0;phb2st_spatial_gather_workspace_size<T>(p,backend_bytes,&gather);
    if(p.groups>1){
        const std::size_t ranks=std::size_t(p.ranks),lists=p.node_rank==0?std::size_t(p.groups):0;
        std::size_t send_bytes=0,receive_bytes=0;const auto element=(std::size_t(b.band)+2)*sizeof(T)+sizeof(R);
        if(!p.node_rank)for(int rank=0;rank<p.groups;++rank){const int lo=g.lo(rank,b.n),hi=g.hi(rank,b.n);
            if(rank==p.group||hi==lo||p.core_lo==p.core_hi)continue;
            const auto out_begin=std::max(p.core_lo,int(std::max<std::int64_t>(0,std::int64_t(lo)-p.halo)));
            const auto out_end=std::min(p.core_hi,int(std::min<std::int64_t>(b.n,std::int64_t(hi)+p.halo)));
            const auto in_begin=std::max(lo,p.offset),in_end=std::min(hi,p.offset+p.width);
            send_bytes+=std::size_t(std::max(0,out_end-out_begin))*element;
            receive_bytes+=std::size_t(std::max(0,in_end-in_begin))*element;}
        spatial_bytes(send_bytes,1);spatial_bytes(receive_bytes,1);spatial_bytes(2*std::size_t(b.n),sizeof(T));
        std::size_t main=8*detail::padded_bytes<int>(lists)+detail::padded_bytes<R>(std::size_t(p.width)+2)+detail::padded_bytes<T>(std::size_t(p.width)+2);
        if(p.node_rank==0)main+=detail::padded_bytes<unsigned char>(std::max(std::size_t(1),send_bytes))+
            detail::padded_bytes<unsigned char>(std::max(std::size_t(1),receive_bytes));
        if(p.ring_capacity){const auto bound=std::size_t(detail::DistributedHistory<T>::chunk_size)+std::size_t(b.n)/(std::size_t(b.band)+1)+2;
            spatial_bytes(bound,sizeof(hb2st_rot<T>));main+=2*detail::padded_bytes<hb2st_rot<T>>(bound)+5*detail::padded_bytes<int>(ranks);}
        const std::size_t tail=2*detail::padded_bytes<int>(ranks)+detail::padded_bytes<T>(std::max(std::size_t(1),2*std::size_t(p.core_hi-p.core_lo)))+
            detail::padded_bytes<T>(2*std::size_t(b.n));
        chase=detail::padded_bytes<hb2st_rot<T>>(std::size_t(p.ring_capacity))+std::max(main,tail);
    }
    *host_bytes=phb2st_band_alignment+detail::padded_bytes<T>(phb2st_spatial_band_count(p))+std::max(gather,chase);
}
// The buffers in the order phb2st_spatial_gather_workspace_size counts them.
template<DdlaBackend B,class T>void phb2st_spatial_gather_run(const DdlaHandle_t& h,const Phb2stSpatialPlan& p,const T* A,T* band,
    void* backend_work,std::size_t backend_bytes,void* host_work,std::size_t host_bytes)
{
    try {
        const auto& g=*static_cast<const SpatialGrid*>(p.topology);const auto& b=p.input;
        SpatialChase<T> domain;domain.offset=p.offset;domain.width=p.width;domain.core_lo=p.core_lo;domain.core_hi=p.core_hi;
        const std::size_t ranks=std::size_t(p.ranks),values=std::max(std::size_t(1),p.indices),received=std::max(std::size_t(1),p.received);
        char* w=static_cast<char*>(host_work);
        int* sc=detail::take<int>(w,ranks);int* sd=detail::take<int>(w,ranks);int* rc=detail::take<int>(w,ranks);int* rd=detail::take<int>(w,ranks);
        std::size_t* indices=detail::take<std::size_t>(w,p.indices);T* send=detail::take<T>(w,values);T* receive=detail::take<T>(w,received);
        assert(std::size_t(w-static_cast<char*>(host_work))<=host_bytes);(void)host_bytes;
        std::size_t* device_indices=nullptr;T* device_values=nullptr;
        if constexpr(B==DdlaBackend::GPU){char* d=static_cast<char*>(backend_work);
            device_indices=detail::take<std::size_t>(d,values);device_values=detail::take<T>(d,values);
            assert(std::size_t(d-static_cast<char*>(backend_work))<=backend_bytes);}
        (void)backend_work;(void)backend_bytes;
        gather_spatial_band<B,T>(h,b.n,b.band,A,b.descriptor.data(),g,domain,band,sc,sd,rc,rd,indices,send,receive,received,
            device_indices,device_values);
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(h) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(h); } catch (...) {}
        throw;
    }
}
// The buffers in the order phb2st_spatial_workspace_size counts them.
template<DdlaBackend B,class T>void phb2st_spatial_run(const DdlaHandle_t& h,const Phb2stSpatialPlan& p,const T* A,
    typename real_type<T>::type* diagonal,typename real_type<T>::type* off,detail::DistributedHistory<T>& history,
    T* phases,std::size_t& phase_count,Hb2stSink<T>* sink,void* backend_work,std::size_t backend_bytes,void* host_work,std::size_t host_bytes)
{
    assert(p.groups>1);const auto& g=*static_cast<const SpatialGrid*>(p.topology);const auto& b=p.input;using R=typename real_type<T>::type;
    char* const host_base=static_cast<char*>(host_work);char* hw=phb2st_band_align(host_base);
    T* band=detail::take<T>(hw,phb2st_spatial_band_count(p));
    phb2st_spatial_gather_run<B,T>(h,p,A,band,backend_work,backend_bytes,hw,host_bytes-std::size_t(hw-host_base));
    // The chase reuses the gather's scratch: the ring, then the halo
    // exchange, dc/wc and the record exchange, with the tail overlaying them.
    hb2st_rot<T>* ring=detail::take<hb2st_rot<T>>(hw,std::size_t(p.ring_capacity));
    char* tail=hw;SpatialChase<T> domain;SpatialHalo<T> halo(g,domain,b.n,b.band,hw,tail);
    R* dc=detail::take<R>(hw,std::size_t(p.width)+2);T* wc=detail::take<T>(hw,std::size_t(p.width)+2);
    domain.context=&halo;domain.before_run=[](void* c,T* a,R* d,T* x){static_cast<SpatialHalo<T>*>(c)->before(a,d,x);};
    domain.finish_run=[](void* c,T* a,R* d,R* e,T* ph){return static_cast<SpatialHalo<T>*>(c)->finish(a,d,e,ph);};
    if(p.ring_capacity){
        SpatialRecords<T> records(g,domain,b.n,b.band,hw);
        assert(std::size_t(std::max(hw,tail)-host_base)<=host_bytes);
        std::atomic<long long> emitted{0},consumed{0};std::atomic<int> aborted{0};StreamWindow window;window.ring=p.ring_capacity;window.consumed=&consumed;
        if(sink){try{spatial_allocate(g.world,[&]{sink->start_stager_run(ring,p.ring_capacity,&emitted,&window,&aborted,1,false);});}
            catch(...){sink->cancel_run();throw;}}
        struct Publish {const Phb2stSpatialPlan& p;SpatialRecords<T>& record_exchange;detail::DistributedHistory<T>& history;
            hb2st_rot<T>* ring;Hb2stSink<T>* sink;std::atomic<long long>& emitted;std::atomic<long long>& consumed;long long handed=0;};
        Publish state{p,records,history,ring,sink,emitted,consumed};BoundedChaseOutput<T> output;output.capacity=p.ring_capacity;output.context=&state;
        output.publish_run=[](void* pointer,long long end,const hb2st_rot<T>*,bool last){auto& s=*static_cast<Publish*>(pointer);
            constexpr long long batch=detail::DistributedHistory<T>::chunk_size;const auto maxt=static_cast<long long>(s.p.input.n)/(s.p.input.band+1)+2;
            if(end-s.handed<batch&&!last)return;
            if(s.p.keep&&!s.sink)s.record_exchange.distribute(end,s.ring,s.p.ring_capacity,s.history);
            else{s.record_exchange.collect(end,s.ring,s.p.ring_capacity);if(s.p.keep)s.history.capture(s.handed,end,s.ring,s.p.ring_capacity);}
            if(s.sink){s.emitted.store(end,std::memory_order_release);while(end+batch+maxt-s.consumed.load(std::memory_order_acquire)>s.p.ring_capacity)std::this_thread::yield();}
            s.handed=end;
        };
        try{host_chase<T,true,true,true>(band,b.n,b.band,diagonal,off,ring,phases,phase_count,nullptr,&output,&domain,dc,wc);}
        catch(...){aborted.store(1,std::memory_order_release);if(sink)sink->cancel_run();throw;}
        if(sink){sink->mark_chase_done();std::exception_ptr error;try{sink->drain_stage2_run();}catch(...){error=std::current_exception();}
            int failed=error?1:0;MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,&failed,1,MPI_INT,MPI_MAX,g.world));
            if(failed){if(error)std::rethrow_exception(error);throw std::runtime_error("phb2st: spatial sink drain failed on another rank");}}
    }else{
        assert(std::size_t(std::max(hw,tail)-host_base)<=host_bytes);
        host_chase<T,false,false,true>(band,b.n,b.band,diagonal,off,ring,phases,phase_count,nullptr,nullptr,&domain,dc,wc);
    }
}
#define DDLA_SPATIAL_SIZE(T) template void phb2st_spatial_gather_workspace_size<T>(const Phb2stSpatialPlan&,std::size_t*,std::size_t*); \
 template void phb2st_spatial_workspace_size<T>(const Phb2stSpatialPlan&,std::size_t*,std::size_t*);
DDLA_SPATIAL_SIZE(float)
DDLA_SPATIAL_SIZE(double)
DDLA_SPATIAL_SIZE(std::complex<float>)
DDLA_SPATIAL_SIZE(std::complex<double>)
#undef DDLA_SPATIAL_SIZE
#define DDLA_SPATIAL_RUN(B,T,R) template void phb2st_spatial_gather_run<B,T>(const DdlaHandle_t&,const Phb2stSpatialPlan&,const T*,T*,void*,std::size_t,void*,std::size_t); \
 template void phb2st_spatial_run<B,T>(const DdlaHandle_t&,const Phb2stSpatialPlan&,const T*,R*,R*,detail::DistributedHistory<T>&,T*,std::size_t&,Hb2stSink<T>*,void*,std::size_t,void*,std::size_t);
DDLA_SPATIAL_RUN(DdlaBackend::CPU,float,float)
DDLA_SPATIAL_RUN(DdlaBackend::CPU,double,double)
DDLA_SPATIAL_RUN(DdlaBackend::CPU,std::complex<float>,float)
DDLA_SPATIAL_RUN(DdlaBackend::CPU,std::complex<double>,double)
#if DDLA_HAS_GPU
DDLA_SPATIAL_RUN(DdlaBackend::GPU,float,float)
DDLA_SPATIAL_RUN(DdlaBackend::GPU,double,double)
DDLA_SPATIAL_RUN(DdlaBackend::GPU,std::complex<float>,float)
DDLA_SPATIAL_RUN(DdlaBackend::GPU,std::complex<double>,double)
#endif
#undef DDLA_SPATIAL_RUN
} // namespace spectral_impl
