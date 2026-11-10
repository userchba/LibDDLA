#include "api_grid_test_common.h"
#include "ddla_stream.h"
#include "workspace_test_common.h"
#include "Eigensolver/two_stage/hb2st_host_workspace.h"
#include "Eigensolver/two_stage/phb2st_impl.h"
#include "dc_threads.h"
#include "grouped_stage5_oracle.h"
#include <array>
#include <cstring>
using namespace api_grid_test;
namespace {
template<class S>bool clean(const S& s){return !s.ddla_alloc&&!s.ddla_free&&!s.pinned_alloc&&!s.raw_gpu_alloc&&!s.host_new&&!s.vendor_query&&!s.resource_setup;}
// Exactly the sized bytes between poisoned guards (backend memory is host memory on the CPU handle).
struct Arena {
    static constexpr std::size_t guard=256;std::size_t bytes;ddla::vector<char> storage;
    Arena(DdlaHandle_t h,std::size_t n):bytes(n),storage(h,n+2*guard){std::memset(storage.data(),0xa7,storage.size());}
    char* data(){return storage.data()+guard;}
    void verify(DdlaHandle_t h){bool ok=true;for(std::size_t i=0;i<guard;++i)ok=ok&&storage.data()[i]==char(0xa7)&&storage.data()[guard+bytes+i]==char(0xa7);
        require_close(h,"host accumulator exact workspace bounds",ok?0:1,0);}
};
long long rotation_count(int n,int kd){detail::ChaseCursor cursor(n,kd);long long total=0;while(!cursor.done()){total+=cursor.count();cursor.next();}return total;}
template<class T>T sine(long long g,int repeat){using R=typename real_type<T>::type;
    if constexpr(std::is_same<T,R>::value)return R((g+repeat*7)%17-8)/R(64);
    else return T(R((g+repeat*7)%17-8)/R(128),R((g+repeat*11)%13-6)/R(128));}
template<class T>T phase(int i){using R=typename real_type<T>::type;
    if constexpr(std::is_same<T,R>::value)return T(1);else return T(std::cos(R(i+1)/R(19)),std::sin(R(i+1)/R(19)));}
template<class R>R yentry(int i,int j,int repeat){return R((i*7+j*11+repeat*13)%31-15)/R(32);}
template<class T>void accum_case(DdlaHandle_t h,int n,int kd){
    using R=typename real_type<T>::type;const auto p=detail::hb2st_accum_host_check(h,n,kd,4);
    std::size_t sink_bytes=0,qband=0;detail::hb2st_accum_host_workspace_size<T>(h,p,&sink_bytes);detail::hb2st_accum_host_qband_workspace_size<T>(h,p,&qband);
    int desc[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(desc,h,n+2,n+3,4,4,(h->nprows_>1?1:0),(h->npcols_>1?1:0)));desc[DDLA_LLD_]+=2+h->myid;
    const auto initial=make_local<T>(h,desc,[](int,int){return T(-919);});std::vector<T>Z(initial),phases(std::max(0,n-1));
    for(int i=0;i<n-1;++i)phases[i]=phase<T>(i);
    const int ldy=std::max(1,p.nrl);std::vector<R>Y(std::size_t(ldy)*n,R(-717));
    const auto f=detail::hb2st_finish_host_check<T>(h,n,Y.data(),ldy,phases.data(),phases.size(),Z.data(),desc,4,3);
    std::size_t finish_bytes=0;detail::hb2st_finish_host_workspace_size<T>(h,f,&finish_bytes);
    MPI_Barrier(h->comm);bool pure=true;if(!h->myid){workspace_test::Scope scope(true);std::size_t a=0,b=0,c=0;
        detail::hb2st_accum_host_workspace_size<T>(h,p,&a);detail::hb2st_accum_host_qband_workspace_size<T>(h,p,&b);detail::hb2st_finish_host_workspace_size<T>(h,f,&c);const auto t=scope.finish();
        pure=clean(t)&&!t.copies&&!t.launches&&!t.syncs&&!t.collectives;}
    MPI_Barrier(h->comm);require_close(h,"host accumulator rank-local pure queries",pure?0:1,0);
    // One buffer: the sink's, whose planner tail (dead after the drain) the finish overlays behind Q_band.
    const std::size_t total=std::max(sink_bytes,qband+finish_bytes);Arena arena(h,total);
    auto resource=detail::hb2st_accum_host_prepare<T>(h,p);detail::DcHostTeam roles(4);
    std::vector<hb2st_rot<T>>records(std::size_t(p.total));std::vector<T>Q(std::size_t(n)*n),expected(Q.size());
    for(int repeat=0;repeat<3;++repeat){
        Z=initial;for(int j=0;j<n;++j)for(int i=0;i<p.nrl;++i)Y[std::size_t(j)*ldy+i]=yentry<R>(p.row_lo+i,j,repeat);
        const auto oldY=Y;std::fill(Q.begin(),Q.end(),T(0));for(int i=0;i<n;++i)Q[std::size_t(i)*n+i]=T(1);
        detail::ChaseCursor cursor(n,kd);while(!cursor.done()){
            for(int k=0;k<cursor.count();++k){const auto g=cursor.base+k;auto& r=records[std::size_t(g)];r.j0=cursor.first()+k*(kd+1);r.s=sine<T>(g,repeat);
                r.c=std::sqrt(R(1)-R(std::norm(T(r.s))));
                for(int i=0;i<n;++i){const auto a=Q[std::size_t(r.j0-1)*n+i],b=Q[std::size_t(r.j0)*n+i];
                    Q[std::size_t(r.j0-1)*n+i]=r.c*a+r.s*b;
                    if constexpr(std::is_same<T,R>::value)Q[std::size_t(r.j0)*n+i]=r.c*b-r.s*a;
                    else Q[std::size_t(r.j0)*n+i]=r.c*b-std::conj(r.s)*a;}}
            cursor.next();}
        for(int j=0;j<n;++j)for(int i=0;i<n;++i){T v=0;for(int k=0;k<n;++k)v+=Q[std::size_t(k)*n+i]*(k?phase<T>(k-1):T(1))*yentry<R>(k,j,repeat);expected[std::size_t(j)*n+i]=v;}
        // A cancellation between compatible calls must quiesce before rebinding.
        if(repeat==1){std::atomic<long long> unavailable{0};resource->reset_run(arena.data(),sink_bytes);resource->start_stager_run(records.data(),p.total,&unavailable,nullptr,nullptr,1,false);resource->cancel_run();}
        std::atomic<long long>emitted{p.total};int info=-1;
        workspace_test::Scope scope;resource->reset_run(arena.data(),sink_bytes);
        resource->start_stager_run(records.data(),p.total,&emitted,nullptr,nullptr,1,repeat==2);
        if(repeat==2)roles.run_roles(resource->extra_roles(),[&](int role){resource->run_role(role);});
        resource->mark_chase_done();resource->drain_stage2_run();
        resource->finish_rows_run(f,Y.data(),phases.data(),Z.data(),info,arena.data()+qband,finish_bytes);const auto t=scope.finish();
        require_close(h,"host accumulate/reset/finish no allocation/query/resource setup",clean(t)?0:1,0);
        require_close(h,"host accumulate finish info",info?1:0,0);
        double error=0;bool padding=true;
        for(int j=0;j<ddla_test::n_loc(h,desc);++j)for(int i=0;i<desc[DDLA_LLD_];++i){
            const int gi=indx_l2g_r(desc,h,i),gj=indx_l2g_c(desc,h,j);const auto off=std::size_t(j)*desc[DDLA_LLD_]+i;
            if(gi<n&&gj<n&&i<ddla_test::m_loc(h,desc))error=std::max(error,workspace_test::relative_error(Z[off],expected[std::size_t(gj)*n+gi]));
            else padding=padding&&Z[off]==initial[off];}
        require_close(h,"host accumulate independent chronological reference",error,std::is_same<R,float>::value?3e-5:3e-13);
        require_close(h,"host accumulate immutable Y and output parent/padding",oldY==Y&&padding?0:1,0);arena.verify(h);
    }
    if(!h->myid)std::cout<<"host accumulation memory n="<<n<<" records="<<p.total<<" max_block="<<p.record_capacity<<" QB_bytes="<<qband
        <<" stage2_bytes="<<sink_bytes-qband<<" finish_bytes="<<finish_bytes<<" reserved="<<total<<std::endl;
}
template<class T>void prepared_bind_case(DdlaHandle_t h) {
    const auto p=detail::hb2st_accum_host_check(h,17,4,4);
    std::size_t bytes=0;detail::hb2st_accum_host_workspace_size<T>(h,p,&bytes);Arena work(h,bytes);
    auto resource=detail::hb2st_accum_host_prepare<T>(h,p);
    workspace_test::Scope scope;resource->reset_run(work.data(),bytes);resource->cancel_run();
    require_close(h,"prepared host accumulator binds caller bytes without growth",clean(scope.finish())?0:1,0);work.verify(h);
}
template<class T>void producer_case(DdlaHandle_t h,bool shared){
    using R=typename real_type<T>::type;constexpr int n=37,kd=4;
    auto topology=spectral_impl::phb2st_topology_prepare(h);const auto sp=detail::hb2st_accum_host_check(h,n,kd,4);
    std::size_t sink_bytes=0,qband=0;detail::hb2st_accum_host_workspace_size<T>(h,sp,&sink_bytes);detail::hb2st_accum_host_qband_workspace_size<T>(h,sp,&qband);
    int da[DDLA_DLEN_],dz[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(da,h,n+2,n+3,4,4,0,0));DDLA_CHECK(ddlaDescInit(dz,h,n+3,n+4,4,4,0,h->npcols_>1?1:0));
    da[DDLA_LLD_]+=2+h->myid;dz[DDLA_LLD_]+=3+h->myid;
    auto A=make_local<T>(h,da,[](int,int){return T(-919);});const auto initial=make_local<T>(h,dz,[](int,int){return T(-919);});auto Z=initial;
    const int ldy=std::max(1,sp.nrl);std::vector<R>Y(std::size_t(ldy)*n),d(n),e(n-1),expected_d(n),expected_e(n-1);
    std::vector<T>phases(n-1),reference_phases(n-1),band(std::size_t(kd+1)*n);
    std::vector<hb2st_rot<T>>records(static_cast<std::size_t>(rotation_count(n,kd)));
    setenv("DDLA_EIG_CHASE_TEAM",shared?"1":"0",1);setenv("DDLA_EIG_CHASE_TEAM_THREADS","2",1);
    spectral_impl::Phb2stSinkPolicy policy;policy.chunk=sp.block_records;policy.max_roles=std::max(2,sp.threads);policy.roles_in_team=true;
    const auto p=spectral_impl::phb2st_check<T>(h,DdlaBackend::CPU,n,kd,A.data(),da,d.data(),e.data(),phases.data(),phases.size(),*topology,spectral_impl::Phb2stOutput::Sink,policy);
    std::size_t chase_backend=0,chase_host=0;spectral_impl::phb2st_workspace_size<T>(p,&chase_backend,&chase_host);
    const auto f=detail::hb2st_finish_host_check<T>(h,n,Y.data(),ldy,phases.data(),phases.size(),Z.data(),dz,4,3);
    std::size_t finish_bytes=0;detail::hb2st_finish_host_workspace_size<T>(h,f,&finish_bytes);
    // One buffer: the sink's (live through the chase), then the chase's host and
    // backend scratch; after the drain the finish overlays everything behind Q_band.
    const std::size_t chase_at=sink_bytes,total=std::max(sink_bytes+chase_host+chase_backend,qband+finish_bytes);Arena arena(h,total);
    auto sink=detail::hb2st_accum_host_prepare<T>(h,sp);spectral_impl::Phb2stResources<T>resources(h,p);
    spectral_impl::Phb2stHostPlan reference_plan;reference_plan.n=n;reference_plan.band=kd;reference_plan.records=true;
    std::size_t reference_bytes=0;spectral_impl::phb2st_host_workspace_size<T>(reference_plan,&reference_bytes);Arena scratch(h,reference_bytes);
    std::vector<T>Q(std::size_t(n)*n),expected(Q.size());
    setenv("DDLA_EIG_CHASE_TEAM",shared?"0":"1",1);
    for(int repeat=0;repeat<2;++repeat){
        auto value=[&](int i,int j){using namespace grouped_stage5_oracle;return i==j?T(R(i+2)/R(3)):scalar<T>(hash(i+repeat,j,11),hash(i+repeat,j,13));};
        A=make_local<T>(h,da,value);const auto oldA=A;Z=initial;std::fill(band.begin(),band.end(),T(0));
        for(int j=0;j<n;++j)for(int i=std::max(0,j-kd);i<=j;++i)band[std::size_t(j)*(kd+1)+kd+i-j]=value(i,j);
        std::size_t pc=0,ec=0;spectral_impl::phb2st_host_run<T>(reference_plan,nullptr,band.data(),expected_d.data(),expected_e.data(),
            records.data(),reference_phases.data(),ec,nullptr,nullptr,nullptr,scratch.data(),reference_bytes);
        std::fill(Q.begin(),Q.end(),T(0));for(int i=0;i<n;++i)Q[std::size_t(i)*n+i]=T(1);
        for(const auto& r:records)for(int i=0;i<n;++i){const auto a=Q[std::size_t(r.j0-1)*n+i],b=Q[std::size_t(r.j0)*n+i];
            Q[std::size_t(r.j0-1)*n+i]=r.c*a+r.s*b;
            if constexpr(std::is_same<T,R>::value)Q[std::size_t(r.j0)*n+i]=r.c*b-r.s*a;else Q[std::size_t(r.j0)*n+i]=r.c*b-std::conj(r.s)*a;}
        for(int j=0;j<n;++j)for(int i=0;i<sp.nrl;++i)Y[std::size_t(j)*ldy+i]=yentry<R>(sp.row_lo+i,j,repeat);const auto oldY=Y;
        for(int j=0;j<n;++j)for(int i=0;i<n;++i){T v=0;for(int k=0;k<n;++k){const auto ph=!std::is_same<T,R>::value&&k?reference_phases[k-1]:T(1);
                v+=Q[std::size_t(k)*n+i]*ph*yentry<R>(k,j,repeat);}expected[std::size_t(j)*n+i]=v;}
        int info=-1;workspace_test::Scope scope;sink->reset_run(arena.data(),sink_bytes);
        spectral_impl::phb2st_run<DdlaBackend::CPU,T>(h,p,resources,A.data(),d.data(),e.data(),phases.data(),pc,sink.get(),
            arena.data()+chase_at+chase_host,chase_backend,arena.data()+chase_at,chase_host);
        sink->finish_rows_run(f,Y.data(),phases.data(),Z.data(),info,arena.data()+qband,finish_bytes);const auto trace=scope.finish();
        require_close(h,"prepared chase+live CPU sink+finish no allocation/query/resources",clean(trace)?0:1,0);
        bool same=pc==ec&&!std::memcmp(d.data(),expected_d.data(),n*sizeof(R))&&!std::memcmp(e.data(),expected_e.data(),(n-1)*sizeof(R))&&!std::memcmp(phases.data(),reference_phases.data(),pc*sizeof(T));
        for(auto x:d)same=same&&std::isfinite(x);for(auto x:e)same=same&&std::isfinite(x);
        require_close(h,"live producer/sink tridiagonal and phases",same&&info==0?0:1,0);
        double error=0;bool padding=true;for(int j=0;j<ddla_test::n_loc(h,dz);++j)for(int i=0;i<dz[DDLA_LLD_];++i){const int gi=indx_l2g_r(dz,h,i),gj=indx_l2g_c(dz,h,j);const auto at=std::size_t(j)*dz[DDLA_LLD_]+i;
            if(gi<n&&gj<n&&i<ddla_test::m_loc(h,dz))error=std::max(error,workspace_test::relative_error(Z[at],expected[std::size_t(gj)*n+gi]));else padding=padding&&Z[at]==initial[at];}
        require_close(h,"live producer/sink independent chronological finish",error,std::is_same<R,float>::value?4e-5:5e-13);
        require_close(h,"producer/sink immutable inputs and output parent padding",A==oldA&&Y==oldY&&padding?0:1,0);arena.verify(h);scratch.verify(h);
    }
    resources.release();topology->release();unsetenv("DDLA_EIG_CHASE_TEAM");unsetenv("DDLA_EIG_CHASE_TEAM_THREADS");
}
template<class T>void grouped_case(DdlaHandle_t h,int n,int nb,int b){
    using namespace grouped_stage5_oracle;const auto fixture=ql_fixture<T>(h,n,b,0);const auto Q=q_from_tau(fixture);
    int da[DDLA_DLEN_],dz[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(da,h,n+3,n+3,nb,nb,0,0));
    DDLA_CHECK(ddlaDescInit(dz,h,n+2,n+5,nb,2*nb,0,h->npcols_>1?1:0));da[DDLA_LLD_]+=h->myid+2;dz[DDLA_LLD_]+=h->myid+3;
    auto A=local_matrix<T>(h,da,[&](int i,int j){return i<n&&j<n?fixture.a[std::size_t(j)*n+i]:T(23);});
    const auto input=local_matrix<T>(h,dz,[&](int i,int j){return i<n&&j<n?input_z<T>(i,j):T(31);}),oldA=A;
    for(char trans:{'N','C','T'})for(int cap:{1,2,3}){
        auto Z=input;detail::GroupedStage5Options options;options.group_cap=cap;
        const auto p=detail::grouped_stage5_check<T>(h,trans,n,b,A.data(),da,fixture.t.data(),Z.data(),dz,options);
        MPI_Barrier(h->comm);bool pure=true;if(!h->myid){workspace_test::Scope scope(true);std::size_t bytes=0;detail::grouped_stage5_workspace_size<T>(h,p,&bytes);
            const auto t=scope.finish();pure=clean(t)&&!t.copies&&!t.launches&&!t.syncs&&!t.collectives;}
        MPI_Barrier(h->comm);require_close(h,"grouped stage5 pure rank-local query",pure?0:1,0);
        std::size_t bytes=0;detail::grouped_stage5_workspace_size<T>(h,p,&bytes);
        require_close(h,"stage5 exact single/multi-panel requested bytes",
            bytes==expected_workspace_bytes<T>(h,p)?0:1,0);
        Arena arena(h,bytes);
        const auto reference=expected(fixture,Q,trans);double scale=1;for(auto v:reference)scale=std::max(scale,workspace_test::finite_error(double(std::abs(v))));
        for(int repeat=0;repeat<2;++repeat){Z=input;int info=-1;workspace_test::Scope scope;
            detail::grouped_stage5_run<DdlaBackend::CPU,T>(h,p,A.data(),fixture.t.data(),Z.data(),info,arena.data(),bytes);const auto t=scope.finish();
            require_close(h,"grouped stage5 raw has no allocations/checks/queries/resources",clean(t)?0:1,0);require_close(h,"grouped stage5 raw info",info?1:0,0);
            double error=0;bool padding=true;for(int j=0;j<ddla_test::n_loc(h,dz);++j)for(int i=0;i<dz[DDLA_LLD_];++i){
                const int gi=indx_l2g_r(dz,h,i),gj=indx_l2g_c(dz,h,j);const auto off=std::size_t(j)*dz[DDLA_LLD_]+i;
                if(gi<n&&gj<n&&i<ddla_test::m_loc(h,dz))error=std::max(error,workspace_test::finite_error(double(std::abs(wide(Z[off])-reference[std::size_t(gj)*n+gi]))/scale));
                else padding=padding&&!std::memcmp(&Z[off],&input[off],sizeof(T));}
            require_close(h,"grouped stage5 dense elementary reference",error,50*std::max(n,1)*std::numeric_limits<Real<T>>::epsilon());
            require_close(h,"grouped stage5 readonly and parent padding",same(A,oldA)&&padding?0:1,0);arena.verify(h);
        }
    }
}
void run(MPI_Comm comm,int pr,int pc,char major){DdlaHandle_t h=nullptr;ddlaInit(h,DdlaBackend::CPU);ddlaSet(h,comm,pr,pc,major);
    setenv("DDLA_EIG_HOST_ACCUM_BLOCK","4096",1);setenv("DDLA_EIG_HOST_ACCUM_SLOTS","2",1);
#define CASE(T) prepared_bind_case<T>(h);accum_case<T>(h,3,2);accum_case<T>(h,17,4);accum_case<T>(h,145,4)
    CASE(float);CASE(double);CASE(std::complex<float>);CASE(std::complex<double>);
#undef CASE
#define PRODUCER(T) producer_case<T>(h,false);producer_case<T>(h,true)
    PRODUCER(float);PRODUCER(double);PRODUCER(std::complex<float>);PRODUCER(std::complex<double>);
#undef PRODUCER
#define STAGE5(T) grouped_case<T>(h,0,4,4);grouped_case<T>(h,1,4,4);grouped_case<T>(h,17,4,4);grouped_case<T>(h,22,4,2)
    STAGE5(float);STAGE5(double);STAGE5(std::complex<float>);STAGE5(std::complex<double>);
#undef STAGE5
    ddlaDestroy(h);}
}
int main(int argc,char** argv){int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);if(provided<MPI_THREAD_FUNNELED)MPI_Abort(MPI_COMM_WORLD,2);
    int size=1;MPI_Comm_size(MPI_COMM_WORLD,&size);TestOptions options;std::string error;
    if(!parse_options(argc,argv,size,options,error)){std::cerr<<error<<std::endl;MPI_Abort(MPI_COMM_WORLD,2);}
    for(auto grid:options.grids)for(char major:{'R','C'})run(MPI_COMM_WORLD,grid.first,grid.second,major);
    MPI_Finalize();return 0;}
