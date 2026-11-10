#include "api_grid_test_common.h"
#include "ddla_stream.h"
#include "workspace_test_common.h"
#include "Eigensolver/two_stage/distributed_history.h"
#include "Eigensolver/two_stage/phb2st_impl.h"
#include "dc_threads.h"
#include <array>
#include <cstdint>
#include <cstring>

using namespace api_grid_test;
namespace {
template<class S>bool clean(const S& s){return !s.ddla_alloc&&!s.ddla_free&&!s.pinned_alloc&&!s.raw_gpu_alloc&&!s.host_new&&!s.vendor_query&&!s.resource_setup;}
// Exactly the sized backend and host bytes between poisoned guards; one
// allocation serves every run of a case.
class Arena {
    static constexpr std::size_t guard=256;DdlaHandle_t h_;std::size_t backend_bytes_,host_bytes_;
    ddla::vector<char> backend_;std::vector<char> host_;
public:
    Arena(DdlaHandle_t h,std::size_t backend_bytes,std::size_t host_bytes):h_(h),backend_bytes_(backend_bytes),host_bytes_(host_bytes),
        backend_(h,backend_bytes+2*guard),host_(host_bytes+2*guard,char(0xa7))
    {DDLA_CHECK(ddlaMemset(h,backend_.data(),0xa7,backend_.size()));DDLA_CHECK(ddlaSynchronize(h));}
    void* backend(){return backend_.data()+guard;}
    char* host(){return host_.data()+guard;}
    void verify(){bool valid=true;std::array<char,2*guard> backend{};
        DDLA_CHECK(ddlaMemcpy(h_,backend.data(),backend_.data(),guard));
        DDLA_CHECK(ddlaMemcpy(h_,backend.data()+guard,backend_.data()+guard+backend_bytes_,guard));DDLA_CHECK(ddlaSynchronize(h_));
        for(std::size_t i=0;i<guard;++i)valid=valid&&host_[i]==char(0xa7)&&
            host_[guard+host_bytes_+i]==char(0xa7)&&backend[i]==char(0xa7)&&backend[guard+i]==char(0xa7);
        require_close(h_,"history/chase exact workspace bounds",valid?0:1,0);}
};
// Rotations of the chase of an order-n band of width kd, and the records one
// history chunk holds.
long long rotation_count(int n,int kd){detail::ChaseCursor cursor(n,kd);long long total=0;while(!cursor.done()){total+=cursor.count();cursor.next();}return total;}
template<class T>std::size_t chunk_records(long long total)
{return std::size_t(std::max(1LL,std::min(detail::DistributedHistory<T>::chunk_size,total)));}
template<class T>T sine(long long g,int repeat) {
    using R=typename real_type<T>::type;const auto x=g+repeat*19;
    if constexpr(std::is_same<T,R>::value)return R(x%17-8)/R(32);
    else return T(R(x%17-8)/R(32),R(x%13-6)/R(16));
}
template<class T>void history_case(DdlaHandle_t h,int n,int kd) {
    using R=typename real_type<T>::type;using Record=hb2st_rot<T>;using History=detail::DistributedHistory<T>;
    // prepare is the only allocation: the cyclic coefficient shard and its headers.
    History history;history.prepare(n,kd,h->comm);
    require_close(h,"history shards over the handle communicator",history.rank==h->myid&&history.ranks==h->nprocs?0:1,0);
    const long long maximum_step=static_cast<long long>(n)/(kd+1)+2;
    const auto capacity=std::max(1LL,History::chunk_size+maximum_step);
    // One chunk of records, sized up front: fetch then reuses its storage.
    const auto fetch_count=std::max(1LL,std::min(History::chunk_size,history.total));
    std::vector<Record> ring(static_cast<std::size_t>(capacity)),records(static_cast<std::size_t>(fetch_count));
    uint64_t local=history.local_count,total=0;MPI_Allreduce(&local,&total,1,MPI_UINT64_T,MPI_SUM,h->comm);
    require_close(h,"history coefficients have exactly one global owner",total==static_cast<uint64_t>(history.total)?0:1,0);
    for(int repeat=0;repeat<2;++repeat) {
        bool correct=true;workspace_test::Scope scope;
        detail::ChaseCursor cursor(n,kd);long long handed=0;
        while(!cursor.done()) {
            for(int j=0;j<cursor.count();++j) {
                const auto g=cursor.base+j;auto& r=ring[static_cast<std::size_t>(g%capacity)];
                r.j0=cursor.first()+j*(kd+1);r.c=R(r.j0);r.s=sine<T>(g,repeat);
            }
            const auto end=cursor.base+cursor.count();cursor.next();
            if(end-handed>=History::chunk_size||cursor.done()) {
                history.capture(handed,end,ring.data(),capacity);handed=end;
            }
        }
        for(bool compact:{false,true})for(long long chunk=history.chunks();chunk-->0;) {
            detail::DistributedHistoryHeader header;
            history.fetch(chunk,h->comm,records,&header,compact);
            const auto first=chunk*History::chunk_size;
            const int count=static_cast<int>(std::min(History::chunk_size,history.total-first));
            correct=correct&&header.cursor.base+header.offset==first&&header.count==count&&
                records.size()==static_cast<std::size_t>(count);
            for(int i=0;i<count;++i)correct=correct&&records[i].c==R(records[i].j0)&&records[i].s==sine<T>(first+i,repeat);
        }
        const auto trace=scope.finish();require_close(h,"history capture/fetch performs no allocation or resource setup",clean(trace)?0:1,0);
        require_close(h,"history cyclic coefficients, regenerated indices and repeat overwrite",correct?0:1,0);
    }
    if(!h->myid)std::cout<<"history memory n="<<n<<" kd="<<kd<<" chunks="<<history.chunks()<<" rank0_elements="<<history.local_count
        <<" rank0_bytes="<<history.local_count*(sizeof(R)+sizeof(T))+history.headers.size()*sizeof(detail::DistributedHistoryHeader)
        <<" producer_ring="<<capacity<<std::endl;
}
template<class T>T entry(int i,int j) {
    using R=typename real_type<T>::type;
    if constexpr(std::is_same<T,R>::value)return R(i*17-j*11)/R(32);
    else return T(R(i*17-j*11)/R(32),R(i*7+j*19)/R(64));
}
template<DdlaBackend B,class T>void extraction_case(DdlaHandle_t h,int n,int band) {
    int desc[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(desc,h,n+2,n+3,4,4,0,0));desc[DDLA_LLD_]+=2+h->myid;
    auto original=make_local<T>(h,desc,[](int i,int j){return entry<T>(i,j);});DeviceBuffer<T>A(h,original.size());upload(h,A.ptr,original);
    const auto p=spectral_impl::phb2st_band_check(h,B,n,band,A.ptr,desc);
    MPI_Barrier(h->comm);bool pure=true;
    if(!h->myid){workspace_test::Scope scope(true);std::size_t bytes=0;spectral_impl::phb2st_band_workspace_size<T>(p,&bytes);
        const auto trace=scope.finish();pure=clean(trace)&&!trace.copies&&!trace.launches&&!trace.syncs&&!trace.collectives;}
    MPI_Barrier(h->comm);require_close(h,"band extraction pure query",pure?0:1,0);
    std::size_t bytes=0;spectral_impl::phb2st_band_workspace_size<T>(p,&bytes);Arena arena(h,bytes,0);
    // The band is the caller's host array; guards around it catch writes past its count.
    const std::size_t count=spectral_impl::phb2st_band_count(p);std::vector<T> outside(count+16,T(-919));T* const result=outside.data()+8;
    for(int repeat=0;repeat<2;++repeat) {
        workspace_test::Scope scope;
        spectral_impl::phb2st_band_run<B,T>(h,p,A.ptr,result,arena.backend(),bytes);
        const auto trace=scope.finish();require_close(h,"band extraction raw no allocation/query/resource setup",clean(trace)?0:1,0);
        bool correct=count==(std::size_t(band)+1)*std::size_t(n);
        for(int j=0;j<n;++j)for(int row=0;row<=band;++row) {
            const int i=j+row-band;const T expected=i>=0?entry<T>(i,j):T(0);
            correct=correct&&result[row+std::size_t(j)*(band+1)]==expected;
        }
        for(std::size_t i=0;i<8;++i)correct=correct&&outside[i]==T(-919)&&outside[8+count+i]==T(-919);
        require_close(h,"band extraction values and host output guards",correct?0:1,0);arena.verify();
    }
    const auto after=download(h,A.ptr,original.size());
    require_close(h,"band extraction input and parent padding immutable",std::memcmp(original.data(),after.data(),original.size()*sizeof(T))?1:0,0);
}
void fixed_role_case(DdlaHandle_t h) {
    detail::DcHostTeam team(3);const auto caller=std::this_thread::get_id();std::array<std::thread::id,3> first{},next{};
    workspace_test::Scope scope;
    team.run_roles(3,[&](int role){first[role]=std::this_thread::get_id();});
    team.run_roles(3,[&](int role){next[role]=std::this_thread::get_id();});
    const auto trace=scope.finish();
    require_close(h,"prepared fixed roles perform no allocation/resource setup",clean(trace)?0:1,0);
    require_close(h,"MPI caller role zero and worker identities stay fixed",first==next&&first[0]==caller&&first[1]!=caller&&first[2]!=caller&&first[1]!=first[2]?0:1,0);
}
// The parallel fallback keeps its per-thread record counts and buffers in the
// caller's host scratch. Reuse poisoned bytes and compare its ordered records
// with the serial chase.
template<class T>void parallel_chase_case(DdlaHandle_t h) {
    using R=typename real_type<T>::type;constexpr int n=37,kd=4;
    spectral_impl::Phb2stHostPlan serial;serial.n=n;serial.band=kd;serial.records=true;
    auto parallel=serial;parallel.threads=3;
    std::size_t sb=0,pb=0;spectral_impl::phb2st_host_workspace_size<T>(serial,&sb);spectral_impl::phb2st_host_workspace_size<T>(parallel,&pb);
    Arena serial_arena(h,0,sb),parallel_arena(h,0,pb);detail::DcHostTeam team(3);
    std::vector<T> band(std::size_t(kd+1)*n),reference(band.size()),phases(n),reference_phases(n);
    std::vector<R>d(n),e(n),reference_d(n),reference_e(n);
    std::vector<hb2st_rot<T>> records(std::size_t(rotation_count(n,kd))),reference_records(records.size());
    for(int repeat=0;repeat<3;++repeat) {
        std::fill(band.begin(),band.end(),T(0));
        for(int col=0;col<n;++col)for(int row=std::max(0,col-kd);row<=col;++row){
            auto x=entry<T>(row+repeat,col);if(row==col)x=T(R(row+2)/R(3));
            band[std::size_t(col)*(kd+1)+kd+row-col]=x;
        }
        reference=band;std::size_t count=0,reference_count=0;
        spectral_impl::phb2st_host_run<T>(serial,nullptr,reference.data(),reference_d.data(),reference_e.data(),
            reference_records.data(),reference_phases.data(),reference_count,nullptr,nullptr,nullptr,serial_arena.host(),sb);
        std::memset(parallel_arena.host(),0x91,pb);
        workspace_test::Scope scope;
        spectral_impl::phb2st_host_run<T>(parallel,&team,band.data(),d.data(),e.data(),
            records.data(),phases.data(),count,nullptr,nullptr,nullptr,parallel_arena.host(),pb);
        const auto trace=scope.finish();require_close(h,"parallel chase reusable scratch has no growth",clean(trace)?0:1,0);
        bool same=count==reference_count&&!std::memcmp(d.data(),reference_d.data(),n*sizeof(R))&&
            !std::memcmp(e.data(),reference_e.data(),std::size_t(n-1)*sizeof(R))&&
            !std::memcmp(phases.data(),reference_phases.data(),count*sizeof(T));
        for(std::size_t i=0;i<records.size();++i)same=same&&records[i].j0==reference_records[i].j0&&
            !std::memcmp(&records[i].c,&reference_records[i].c,sizeof(R))&&!std::memcmp(&records[i].s,&reference_records[i].s,sizeof(T));
        for(int i=0;i<n;++i)same=same&&std::isfinite(d[i]);for(int i=0;i<n-1;++i)same=same&&std::isfinite(e[i]);
        require_close(h,"parallel chase preserves serial scalar values and record order",same?0:1,0);
        parallel_arena.verify();serial_arena.verify();
    }
}
template<class T>void chase_team_case(DdlaHandle_t h,int n,int kd,bool ring){
    using R=typename real_type<T>::type;MPI_Comm node=MPI_COMM_NULL;MPI_CHECK(MPI_Comm_split_type(h->comm,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node));
    int rank=0,size=1;MPI_CHECK(MPI_Comm_rank(node,&rank));MPI_CHECK(MPI_Comm_size(node,&size));
    spectral_impl::Phb2stTeamPlan p;p.n=n;p.band=kd;p.node_rank=rank;p.node_size=size;p.tpr=2;p.participants=std::min(2,size);p.fallback_threads=3;
    p.ring_request=ring?4*(static_cast<long long>(n)/(kd+1)+2):0;
    auto resource=spectral_impl::phb2st_team_prepare<T>(h,p);
    std::vector<T> band(std::size_t(kd+1)*n,T(0)),reference,phases(std::max(1,n-1)),expected_phases(phases.size());
    std::vector<R>d(n),e(std::max(1,n-1)),expected_d(n),expected_e(e.size());
    const auto total=rotation_count(n,kd);
    std::vector<hb2st_rot<T>> records(ring?0:std::size_t(total)),expected_records(static_cast<std::size_t>(total));
    spectral_impl::Phb2stHostPlan hp;hp.n=n;hp.band=kd;hp.records=true;
    std::size_t hb=0;spectral_impl::phb2st_host_workspace_size<T>(hp,&hb);Arena arena(h,0,hb);
    for(int repeat=0;repeat<2;++repeat){
        std::fill(band.begin(),band.end(),T(0));for(int col=0;col<n;++col)for(int row=std::max(0,col-kd);row<=col;++row){
            auto x=entry<T>(row+repeat,col);if(row==col)x=T(R(row+2)/R(3));band[std::size_t(col)*(kd+1)+kd+row-col]=x;}
        reference=band;std::size_t expected_count=0,count=0;
        spectral_impl::phb2st_host_run<T>(hp,nullptr,reference.data(),expected_d.data(),expected_e.data(),
            expected_records.data(),expected_phases.data(),expected_count,nullptr,nullptr,nullptr,arena.host(),hb);
        bool same=true;workspace_test::Scope scope;
        if(resource->available()){
            resource->reset_run(band.data());resource->run(nullptr,nullptr,nullptr,262144);
            resource->outputs_run(d.data(),e.data(),phases.data(),count,ring?nullptr:records.data());
            same=count==expected_count&&!std::memcmp(d.data(),expected_d.data(),std::size_t(n)*sizeof(R))&&
                !std::memcmp(e.data(),expected_e.data(),std::size_t(std::max(0,n-1))*sizeof(R))&&
                !std::memcmp(phases.data(),expected_phases.data(),count*sizeof(T));
            for(int i=0;i<n;++i)same=same&&std::isfinite(d[i]);for(int i=0;i<n-1;++i)same=same&&std::isfinite(e[i]);
            for(std::size_t i=0;i<count;++i)same=same&&std::isfinite(double(std::abs(phases[i])));
            if(!ring)for(std::size_t i=0;i<expected_records.size();++i)same=same&&records[i].j0==expected_records[i].j0&&records[i].c==expected_records[i].c&&records[i].s==expected_records[i].s;
        }
        const auto trace=scope.finish();require_close(h,"shared chase repeat no allocations/queries/resources",clean(trace)?0:1,0);
        require_close(h,"shared chase numerical identity to serial chase",same?0:1,0);
        arena.verify();
    }
    if(!h->myid){const auto stream=resource->stream();std::cout<<"shared chase memory n="<<n<<" ring="<<stream.window.ring<<" records="<<stream.capacity<<std::endl;}
    resource->release();MPI_CHECK(MPI_Comm_free(&node));
}
template<DdlaBackend B,class T>void spatial_case(DdlaHandle_t h,spectral_impl::Phb2stTopology& topology,int n,int kd,bool keep){
    using R=typename real_type<T>::type;int desc[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(desc,h,n+2,n+3,4,4,0,0));desc[DDLA_LLD_]+=h->myid+2;
    auto input=make_local<T>(h,desc,[](int,int){return T(-919);});DeviceBuffer<T>A(h,input.size());
    const auto band_plan=spectral_impl::phb2st_band_check(h,B,n,kd,A.ptr,desc);
    const auto p=spectral_impl::phb2st_spatial_geometry(h,band_plan,topology,keep,false,0);
    std::size_t backend_bytes=0,host_bytes=0;spectral_impl::phb2st_spatial_workspace_size<T>(p,&backend_bytes,&host_bytes);Arena arena(h,backend_bytes,host_bytes);
    // A kept stream goes to a history prepared for the plan's geometry (collective).
    detail::DistributedHistory<T> history;if(keep)history.prepare(n,kd,h->comm);
    spectral_impl::Phb2stHostPlan reference_plan;reference_plan.n=n;reference_plan.band=kd;reference_plan.records=keep;
    std::size_t reference_bytes=0;spectral_impl::phb2st_host_workspace_size<T>(reference_plan,&reference_bytes);Arena reference_work(h,0,reference_bytes);
    std::vector<T>band(std::size_t(kd+1)*n,T(0)),phases(std::max(1,n-1)),expected_phases(phases.size()),gathered(spectral_impl::phb2st_spatial_band_count(p));
    std::vector<R>d(n),e(std::max(1,n-1)),expected_d(n),expected_e(e.size());const auto total=rotation_count(n,kd);
    std::vector<hb2st_rot<T>>records(keep?std::size_t(total):0),fetched(chunk_records<T>(total));
    MPI_Barrier(h->comm);bool pure=true;if(!h->myid){workspace_test::Scope scope(true);std::size_t bb=0,hb=0;spectral_impl::phb2st_spatial_workspace_size<T>(p,&bb,&hb);
        const auto t=scope.finish();pure=clean(t)&&!t.copies&&!t.launches&&!t.syncs&&!t.collectives;}
    MPI_Barrier(h->comm);require_close(h,"compact spatial query pure and rank-local",pure?0:1,0);
    for(int repeat=0;repeat<2;++repeat){
        auto value=[&](int i,int j){return i==j?T(R(i+2)/R(3)):entry<T>(i+repeat,j);};
        input=make_local<T>(h,desc,value);upload(h,A.ptr,input);std::fill(band.begin(),band.end(),T(0));
        for(int col=0;col<n;++col)for(int row=std::max(0,col-kd);row<=col;++row)band[std::size_t(col)*(kd+1)+kd+row-col]=value(row,col);
        std::size_t count=0,expected_count=0;
        spectral_impl::phb2st_host_run<T>(reference_plan,nullptr,band.data(),expected_d.data(),expected_e.data(),
            records.data(),expected_phases.data(),expected_count,nullptr,nullptr,nullptr,reference_work.host(),reference_bytes);
        if(topology.groups()>1){workspace_test::Scope scope;
            spectral_impl::phb2st_spatial_run<B,T>(h,p,A.ptr,d.data(),e.data(),history,phases.data(),count,nullptr,arena.backend(),backend_bytes,arena.host(),host_bytes);
            const auto t=scope.finish();require_close(h,"compact spatial run no allocations/queries/resources",clean(t)?0:1,0);
            bool same=count==expected_count&&!std::memcmp(d.data(),expected_d.data(),std::size_t(n)*sizeof(R))&&
                !std::memcmp(e.data(),expected_e.data(),std::size_t(std::max(0,n-1))*sizeof(R))&&!std::memcmp(phases.data(),expected_phases.data(),count*sizeof(T));
            for(int i=0;i<n;++i)same=same&&std::isfinite(d[i]);for(int i=0;i<n-1;++i)same=same&&std::isfinite(e[i]);
            for(std::size_t i=0;i<count;++i)same=same&&std::isfinite(double(std::abs(phases[i])));
            if(keep)for(long long chunk=0;chunk<history.chunks();++chunk){history.fetch(chunk,h->comm,fetched.data());
                const auto first=chunk*history.chunk_size,stop=std::min(history.total,first+history.chunk_size);
                for(auto g=first;g<stop;++g){const auto& actual=fetched[std::size_t(g-first)];const auto& reference=records[std::size_t(g)];
                    same=same&&actual.j0==reference.j0&&actual.c==reference.c&&actual.s==reference.s;}}
            require_close(h,"compact spatial d/e/phases/history bitwise serial identity",same?0:1,0);
        }else{workspace_test::Scope scope;spectral_impl::phb2st_spatial_gather_run<B,T>(h,p,A.ptr,gathered.data(),arena.backend(),backend_bytes,arena.host(),host_bytes);const auto t=scope.finish();
            require_close(h,"one-node prepared band gather no allocation/query/resources",clean(t)?0:1,0);bool same=true;
            if(topology.node_rank()==0)for(int col=0;col<n;++col)for(int r=0;r<kd+1;++r){const int row=col+r-kd;
                const T expected=row>=0?value(row,col):T(0);same=same&&gathered[std::size_t(col)*(kd+1)+r]==expected;}
            require_close(h,"one-node prepared gather exact band",same?0:1,0);}
        const auto after=download(h,A.ptr,input.size());require_close(h,"spatial chase source/padding immutable",std::memcmp(after.data(),input.data(),input.size()*sizeof(T))?1:0,0);
        arena.verify();reference_work.verify();
    }
    // The gather alone runs in exactly its own (smaller) size.
    std::size_t gather_backend=0,gather_host=0;spectral_impl::phb2st_spatial_gather_workspace_size<T>(p,&gather_backend,&gather_host);
    if(n==17) {
        Arena gather_work(h,gather_backend,gather_host);workspace_test::Scope trace;
        spectral_impl::phb2st_spatial_gather_run<B,T>(h,p,A.ptr,gathered.data(),gather_work.backend(),gather_backend,gather_work.host(),gather_host);
        require_close(h,"exactly sized spatial gather has no workspace growth",clean(trace.finish())?0:1,0);gather_work.verify();
    }
    if(!h->myid)std::cout<<"spatial chase memory n="<<n<<" groups="<<p.groups<<" rank0_band="<<spectral_impl::phb2st_spatial_band_count(p)*sizeof(T)<<" ring="<<p.ring_capacity
        <<" gather="<<gather_host<<" reserved="<<host_bytes<<std::endl;
}
template<DdlaBackend B,class T>void main_chase_case(DdlaHandle_t h,spectral_impl::Phb2stTopology& topology,int n,spectral_impl::Phb2stOutput output,bool shared,bool external_history=false){
    using R=typename real_type<T>::type;const int kd=4;setenv("DDLA_EIG_CHASE_TEAM",shared?"1":"0",1);setenv("DDLA_EIG_CHASE_TEAM_THREADS","2",1);setenv("DDLA_EIG_CHASE_THREADS","3",1);
    int desc[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(desc,h,n+2,n+3,4,4,0,0));desc[DDLA_LLD_]+=h->myid+2;
    auto input=make_local<T>(h,desc,[](int,int){return T(-919);});DeviceBuffer<T>A(h,input.size());
    const auto total=rotation_count(n,kd);
    std::vector<R>d(n),e(std::max(1,n-1)),expected_d(n),expected_e(e.size());std::vector<T>phases(std::max(1,n-1)),expected_phases(phases.size()),band(std::size_t(kd+1)*n);
    std::vector<hb2st_rot<T>> expected_records(output==spectral_impl::Phb2stOutput::None?0:std::size_t(total)),fetched(chunk_records<T>(total));
    const auto p=spectral_impl::phb2st_check<T>(h,B,n,kd,A.ptr,desc,d.data(),e.data(),phases.data(),phases.size(),topology,output);
    MPI_Barrier(h->comm);bool pure=true;if(!h->myid){workspace_test::Scope scope(true);std::size_t bb=0,hb=0;spectral_impl::phb2st_workspace_size<T>(p,&bb,&hb);const auto t=scope.finish();
        pure=clean(t)&&!t.copies&&!t.launches&&!t.syncs&&!t.collectives;}
    MPI_Barrier(h->comm);require_close(h,"main chase pure rank-local workspace query",pure?0:1,0);
    std::size_t backend_bytes=0,host_bytes=0;spectral_impl::phb2st_workspace_size<T>(p,&backend_bytes,&host_bytes);Arena arena(h,backend_bytes,host_bytes);
    // A caller-prepared history is retained in place of one the resources prepare.
    detail::DistributedHistory<T> external;if(external_history)external.prepare(n,kd,h->comm);
    spectral_impl::Phb2stResources<T>resources(h,p,external_history?&external:nullptr);
    if(external_history)require_close(h,"resources retain the external history",&resources.history()==&external?0:1,0);
    std::vector<R>own_d(n),own_e(std::max(1,n-1));std::vector<T>own_phases;detail::DistributedHistory<T> own_history;
    spectral_impl::Phb2stHostPlan rp;rp.n=n;rp.band=kd;rp.records=output!=spectral_impl::Phb2stOutput::None;
    std::size_t reference_bytes=0;spectral_impl::phb2st_host_workspace_size<T>(rp,&reference_bytes);Arena reference(h,0,reference_bytes);
    setenv("DDLA_EIG_CHASE_TEAM",shared?"0":"1",1);setenv("DDLA_EIG_CHASE_THREADS","1",1);
    for(int repeat=0;repeat<2;++repeat){auto value=[&](int i,int j){return i==j?T(R(i+2)/R(3)):entry<T>(i+repeat,j);};
        input=make_local<T>(h,desc,value);upload(h,A.ptr,input);std::fill(band.begin(),band.end(),T(0));
        for(int j=0;j<n;++j)for(int i=std::max(0,j-kd);i<=j;++i)band[std::size_t(j)*(kd+1)+kd+i-j]=value(i,j);
        std::size_t ec=0,count=0;spectral_impl::phb2st_host_run<T>(rp,nullptr,band.data(),expected_d.data(),expected_e.data(),
            expected_records.data(),expected_phases.data(),ec,nullptr,nullptr,nullptr,reference.host(),reference_bytes);
        workspace_test::Scope scope;spectral_impl::phb2st_run<B,T>(h,p,resources,A.ptr,d.data(),e.data(),phases.data(),count,nullptr,
            arena.backend(),backend_bytes,arena.host(),host_bytes);
        const auto trace=scope.finish();require_close(h,"main chase repeat no allocations/queries/resources",clean(trace)?0:1,0);
        bool same=count==ec&&!std::memcmp(d.data(),expected_d.data(),std::size_t(n)*sizeof(R))&&
            !std::memcmp(e.data(),expected_e.data(),std::size_t(std::max(0,n-1))*sizeof(R))&&!std::memcmp(phases.data(),expected_phases.data(),count*sizeof(T));
        for(int i=0;i<n;++i)same=same&&std::isfinite(d[i]);for(int i=0;i<n-1;++i)same=same&&std::isfinite(e[i]);
        for(std::size_t i=0;i<count;++i)same=same&&std::isfinite(double(std::abs(phases[i])));
        if(output==spectral_impl::Phb2stOutput::History){const auto& retained=resources.history();for(long long chunk=0;chunk<retained.chunks();++chunk){
            retained.fetch(chunk,h->comm,fetched.data());const auto first=chunk*retained.chunk_size,stop=std::min(retained.total,first+retained.chunk_size);
            for(auto g=first;g<stop;++g){const auto& a=fetched[std::size_t(g-first)];const auto& b=expected_records[std::size_t(g)];same=same&&a.j0==b.j0&&a.c==b.c&&a.s==b.s;}}}
        require_close(h,"main chase frozen-policy numerical identity",same?0:1,0);arena.verify();reference.verify();
        if(n&&!external_history){int info=-1;
            phb2st_node<B,T>(h,n,kd,A.ptr,desc,own_d.data(),own_e.data(),own_history,own_phases,output==spectral_impl::Phb2stOutput::History,info);
            bool owner_same=info==0&&own_phases.size()==ec&&!std::memcmp(own_d.data(),expected_d.data(),std::size_t(n)*sizeof(R))&&
                !std::memcmp(own_e.data(),expected_e.data(),std::size_t(n-1)*sizeof(R))&&(!ec||!std::memcmp(own_phases.data(),expected_phases.data(),ec*sizeof(T)));
            if(output==spectral_impl::Phb2stOutput::History){owner_same=owner_same&&own_history.ready;
                for(long long chunk=0;chunk<own_history.chunks();++chunk){own_history.fetch(chunk,h->comm,fetched.data());
                    const auto first=chunk*own_history.chunk_size,stop=std::min(own_history.total,first+own_history.chunk_size);
                    for(auto g=first;g<stop;++g){const auto& a=fetched[std::size_t(g-first)];const auto& b=expected_records[std::size_t(g)];owner_same=owner_same&&a.j0==b.j0&&a.c==b.c&&a.s==b.s;}}}
            else owner_same=owner_same&&!own_history.ready;
            require_close(h,"owning distributed adapter preserve one-pass outputs and compact history",owner_same?0:1,0);
        }
        const auto after=download(h,A.ptr,input.size());require_close(h,"main chase input/padding immutable",std::memcmp(after.data(),input.data(),input.size()*sizeof(T))?1:0,0);
    }
    resources.release();own_history.clear();unsetenv("DDLA_EIG_CHASE_TEAM");unsetenv("DDLA_EIG_CHASE_TEAM_THREADS");unsetenv("DDLA_EIG_CHASE_THREADS");
}
template<DdlaBackend B>void run(MPI_Comm comm,int pr,int pc,char major) {
    DdlaHandle_t h=nullptr;ddlaInit(h,B);ddlaSet(h,comm,pr,pc,major);fixed_role_case(h);
#define TEAM(T) parallel_chase_case<T>(h);chase_team_case<T>(h,17,4,false);chase_team_case<T>(h,65,4,true)
    TEAM(float);TEAM(double);TEAM(std::complex<float>);TEAM(std::complex<double>);
#undef TEAM
    for(int forced:{0,2}){detail::set_phb2st_test_partitions(forced);auto topology=spectral_impl::phb2st_topology_prepare(h);
        if(!forced)for(auto output:{spectral_impl::Phb2stOutput::None,spectral_impl::Phb2stOutput::History})for(bool shared:{false,true}){
#define MAIN(T) main_chase_case<B,T>(h,*topology,0,output,shared);main_chase_case<B,T>(h,*topology,1,output,shared);main_chase_case<B,T>(h,*topology,17,output,shared);if(output==spectral_impl::Phb2stOutput::History)main_chase_case<B,T>(h,*topology,17,output,shared,true)
            MAIN(float);MAIN(double);MAIN(std::complex<float>);MAIN(std::complex<double>);
#undef MAIN
        }

#define SPATIAL(T) spatial_case<B,T>(h,*topology,1,1,false);spatial_case<B,T>(h,*topology,17,4,true);spatial_case<B,T>(h,*topology,773,4,true)
        SPATIAL(float);SPATIAL(double);SPATIAL(std::complex<float>);SPATIAL(std::complex<double>);
#undef SPATIAL
        topology->release();}detail::set_phb2st_test_partitions(0);
#define BAND(T) extraction_case<B,T>(h,17,4);extraction_case<B,T>(h,1,1);extraction_case<B,T>(h,0,4)
    BAND(float);BAND(double);BAND(std::complex<float>);BAND(std::complex<double>);
#undef BAND
#define CASES(T) history_case<T>(h,0,4);history_case<T>(h,1,1);history_case<T>(h,17,4);history_case<T>(h,1100,4);history_case<T>(h,1800,16)
    CASES(float);CASES(double);CASES(std::complex<float>);CASES(std::complex<double>);
#undef CASES
    if(!h->myid)std::cout<<"workspace history PASS backend="<<int(B)<<" grid="<<pr<<"x"<<pc<<" major="<<major
        <<" allocation_trace="<<(workspace_test::available()?"enabled":"numerical-only")<<std::endl;
    ddlaDestroy(h);
}
}
int main(int argc,char** argv) {
    int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);if(provided<MPI_THREAD_FUNNELED)MPI_Abort(MPI_COMM_WORLD,2);
    int size=1;MPI_Comm_size(MPI_COMM_WORLD,&size);TestOptions options;std::string error;
    if(!parse_options(argc,argv,size,options,error)){std::cerr<<error<<std::endl;MPI_Abort(MPI_COMM_WORLD,2);}
    for(auto grid:options.grids)for(char major:{'R','C'}) {
        run<DdlaBackend::CPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#if DDLA_HAS_GPU
        run<DdlaBackend::GPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#endif
    }
    MPI_Finalize();return 0;
}
