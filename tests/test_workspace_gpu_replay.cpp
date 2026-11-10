#include "api_grid_test_common.h"
#include "ddla_stream.h"
#include <helpers.h>
#include "workspace_test_common.h"
#include "Eigensolver/two_stage/hb2st_replay_gpu.h"
#include "Eigensolver/two_stage/hb2st_replay_wave_schedule.h"
#include "Eigensolver/two_stage/hb2st_wave_schedule.h"
#include <optional>
#include <dlfcn.h>
#include "row_share.h"
#include <array>
#include <cstring>
#include <limits>

using namespace api_grid_test;
namespace {
bool scratch_free(const workspace_test::Snapshot& s)
{return !s.ddla_alloc&&!s.ddla_free&&!s.pinned_alloc&&!s.raw_gpu_alloc&&!s.host_new&&!s.vendor_query&&!s.resource_setup;}
bool pure(const workspace_test::Snapshot& s)
{return scratch_free(s)&&!s.copies&&!s.launches&&!s.syncs&&!s.collectives;}
class Guarded {
    static constexpr std::size_t guard=256;
    DdlaHandle_t h_;std::size_t bb_,hb_;ddla::vector<char> backend_;std::vector<char> host_;
public:
    Guarded(DdlaHandle_t h,std::size_t bb,std::size_t hb):h_(h),bb_(bb),hb_(hb),backend_(h,bb+2*guard),host_(hb+2*guard,char(0x65)) {
        DDLA_CHECK(ddlaMemset(h,backend_.data(),0x65,backend_.size()));check_ddla_sync(h);
    }
    char* backend(){return backend_.data()+guard;}
    char* host(){return host_.data()+guard;}
    void verify(){std::array<unsigned char,2*guard> bytes{};
        DDLA_CHECK(ddlaMemcpy(h_,bytes.data(),backend_.data(),guard));
        DDLA_CHECK(ddlaMemcpy(h_,bytes.data()+guard,backend()+bb_,guard));check_ddla_sync(h_);
        bool ok=true;for(std::size_t i=0;i<guard;++i)ok=ok&&bytes[i]==0x65&&bytes[guard+i]==0x65&&
            static_cast<unsigned char>(host_[i])==0x65&&static_cast<unsigned char>(host()[hb_+i])==0x65;
        require_close(h_,"GPU replay exact arena canaries",ok?0:1,0);}
};
template<class T>std::complex<long double> wide(T v){return {static_cast<long double>(v),0};}
template<class R>std::complex<long double> wide(std::complex<R> v){return {v.real(),v.imag()};}
template<class R>R yvalue(int r,int c,int pass){return R((r*17+c*13+pass*3)%47-23)/R(47);}
template<class T>void fill_records(std::vector<hb2st_rot<T>>& records,int n,int kd,int pass)
{
    using R=typename real_type<T>::type;detail::ChaseCursor cursor(n,kd);std::size_t at=0;
    while(!cursor.done()) {
        for(int j=0;j<cursor.count();++j) {
            auto& r=records[at];r.j0=cursor.first()+j*(kd+1);
            const double angle=0.003*((static_cast<long long>(at)*7+pass*3)%101-50);r.c=R(std::cos(angle));const R magnitude=R(std::sin(angle));
            if constexpr(std::is_same_v<T,R>)r.s=magnitude;
            else {const double phi=0.013*((at*11)%71);r.s=T(magnitude*R(std::cos(phi)),magnitude*R(std::sin(phi)));}
            ++at;
        }
        cursor.next();
    }
    if(at!=records.size())throw std::runtime_error("GPU replay test cursor mismatch");
}
template<class T>void exercise(DdlaHandle_t h,int n,bool waves,bool compact,int kd=4)
{
    using R=typename real_type<T>::type;constexpr int nb=4;const int P=h->nprocs,rank=h->myid;
    const detail::RowShare rows(n,P);const int nr=rows.count(rank),ldy=std::max(1,nr)+3;
    int d[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(d,h,n+5,n+7,nb,nb,h->nprows_-1,h->npcols_-1));d[DDLA_LLD_]+=5+rank%2;
    detail::DistributedHistory<T> history;history.prepare(n,kd,h->comm);
    std::vector<hb2st_rot<T>> records(static_cast<std::size_t>(history.total));fill_records(records,n,kd,0);
    if(history.total)history.capture(0,history.total,records.data(),history.total);
    std::vector<T> phases;
    if constexpr(!std::is_same_v<T,R>)for(int r=1;r<n;++r)phases.emplace_back(R(std::cos(r*0.007)),R(std::sin(r*0.007)));
    DeviceBuffer<R> Y(h,std::size_t(ldy)*std::max(1,n));DeviceBuffer<T>Z(h,local_size(h,d));
    const T sentinel=T(139);std::vector<T> before(Z.count,sentinel);std::vector<R> input(Y.count);
    std::vector<int> prefix;int nc=rows.count(rank);
    if(compact&&P>1){prefix.resize(P+1);for(int r=0;r<P;++r)prefix[r]=rows.lo(r);prefix[P]=n;prefix[1]=0;nc=prefix[rank+1]-prefix[rank];}
    const int main_columns=!prefix.empty()&&nc>0?nc-1:0;
    DeviceBuffer<T> main(h,std::size_t(n)*main_columns);
    const auto p=detail::hb2st_replay_gpu_check<T>(h,n,Y.ptr,ldy,history,phases.data(),phases.size(),Z.ptr,d,7,17,waves,
        prefix.empty()?nullptr:main.ptr,std::size_t(n)*main_columns,prefix.empty()?nullptr:prefix.data());
    MPI_Barrier(h->comm);bool one=true;
    if(rank==0){workspace_test::Scope scope(true);std::size_t b=0,s=0;detail::hb2st_replay_gpu_workspace_size<T>(h,p,&b,&s);one=pure(scope.finish());}
    MPI_Barrier(h->comm);require_close(h,"GPU replay one-rank pure query",one?0:1,0);
    std::size_t bb=0,hb=0;workspace_test::Scope query_scope(true);detail::hb2st_replay_gpu_workspace_size<T>(h,p,&bb,&hb);const auto qs=query_scope.finish();
    require_close(h,"GPU replay all-rank pure query",pure(qs)?0:1,0);
    Guarded arena(h,bb,hb);auto* fixed_host=arena.host();auto* fixed_backend=arena.backend();detail::Hb2stReplayGpuResources<T> resources(h,p);
    for(int pass=0;pass<2;++pass) {
        fill_records(records,n,kd,pass);if(history.total)history.capture(0,history.total,records.data(),history.total);
        std::fill(input.begin(),input.end(),std::numeric_limits<R>::quiet_NaN());
        for(int c=0;c<n;++c)for(int r=0;r<nr;++r)input[r+std::size_t(c)*ldy]=yvalue<R>(rows.lo(rank)+r,c,pass);
        upload(h,Y.ptr,input);upload(h,Z.ptr,before);check_ddla_sync(h);R* original=Y.ptr;int info=-1;bool consumed=false;
        workspace_test::Scope scope;
        detail::hb2st_replay_gpu_run<T>(h,p,resources,history,phases.data(),Y.ptr,Z.ptr,info,&consumed,arena.backend(),bb,arena.host(),hb);
        const auto snapshot=scope.finish();require_close(h,"GPU replay run has no managed scratch/query/setup",scratch_free(snapshot)?0:1,0);
        require_close(h,"GPU replay raw preserves input ownership",Y.ptr==original&&consumed?0:1,0);require_close(h,"GPU replay info",info,0);
        const auto after=download(h,Z.ptr,before.size());const auto retained=download(h,Y.ptr,input.size());
        require_close(h,"GPU replay input bytes preserved",std::memcmp(retained.data(),input.data(),input.size()*sizeof(R))?1:0,0);
        double error=0,padding=0;const int zr=ddla_test::m_loc(h,d),zc=ddla_test::n_loc(h,d);
        for(int c=0;c<zc;++c)for(int r=0;r<d[DDLA_LLD_];++r) {
            const int gr=r<zr?indx_l2g_r(d,h,r):n,gc=indx_l2g_c(d,h,c);
            if(gr>=n||gc>=n)if(after[r+std::size_t(c)*d[DDLA_LLD_]]!=sentinel)padding=1;
        }
        for(int c=0;c<n;++c) {
            std::vector<std::complex<long double>> reference(n);
            for(int r=0;r<n;++r)reference[r]=wide(yvalue<R>(r,c,pass))*(r&&!phases.empty()?wide(phases[r-1]):std::complex<long double>(1,0));
            for(auto it=records.rbegin();it!=records.rend();++it) {
                const int b=it->j0,a=b-1;const auto x=reference[a],y=reference[b],s=wide(it->s);const long double cs=it->c;
                reference[a]=cs*x-std::conj(s)*y;reference[b]=s*x+cs*y;
            }
            for(int lc=0;lc<zc;++lc)if(indx_l2g_c(d,h,lc)==c)for(int lr=0;lr<zr;++lr) {
                const int r=indx_l2g_r(d,h,lr);if(r<n)error=std::max(error,workspace_test::finite_error(
                    static_cast<double>(std::abs(wide(after[lr+std::size_t(lc)*d[DDLA_LLD_]])-reference[r]))));
            }
        }
        require_close(h,"GPU replay output padding",padding,0);
        require_close(h,"GPU replay literal reverse reference",error/(std::max(1,n)*std::numeric_limits<R>::epsilon()),50);
        arena.verify();require_close(h,"GPU replay fixed arena reuse",arena.host()==fixed_host&&arena.backend()==fixed_backend?0:1,0);
    }
}

// Compare the streaming generator against the complete offline schedule.
// Reserve batch storage before entering the traced allocation-free interval.
void bounded_schedules(DdlaHandle_t h)
{
    for(int n:{1,2,17,83})for(int kd:{1,4,16})for(bool real:{false,true}) {
        const auto reference=detail::build_hb2st_schedule(n,kd,real);
        detail::Hb2stWaveGen gen(n,kd,real);detail::Hb2stWaveGen::Batch batch;
        batch.gjlo.reserve(512);batch.gcnt.reserve(512);batch.midx.reserve(512*gen.K());
        std::size_t wave=0,b=0;bool okay=true;
        for(;;){workspace_test::Scope scope;const bool next=gen.next(batch);auto s=scope.finish();okay=okay&&scratch_free(s);if(!next)break;
            okay=okay&&b<reference.wbatch_lo.size();
            if(!okay)break;
            okay=okay&&reference.wbatch_lo[b]==int(wave)&&reference.wbatch_hi[b]==int(wave)+batch.nw&&reference.wbatch_gate[b]==batch.gate;
            for(int i=0;i<batch.nw;++i,++wave){okay=okay&&batch.gjlo[i]==reference.gjlo[wave]&&batch.gcnt[i]==reference.gcnt[wave];
                for(int j=0;j<gen.K();++j)okay=okay&&batch.midx[std::size_t(i)*gen.K()+j]==reference.midx[wave*gen.K()+j];}
            ++b;
        }
        require_close(h,"streaming schedule equals offline tables without allocation",okay&&wave==std::size_t(reference.nw)&&b==reference.wbatch_lo.size()?0:1,0);
    }
}
void body(DdlaHandle_t h,const Shape&)
{
    bounded_schedules(h);
#if DDLA_HAS_GPU
    if(h->backend!=DdlaBackend::GPU)return;
    for(bool waves:{false,true})for(bool compact:{false,true}) {
        exercise<float>(h,37,waves,compact);exercise<double>(h,37,waves,compact);
        exercise<std::complex<float>>(h,37,waves,compact);exercise<std::complex<double>>(h,37,waves,compact);
    }
    exercise<std::complex<float>>(h,129,true,false,16);
    exercise<std::complex<double>>(h,129,true,false,16);
    exercise<double>(h,0,true,false);exercise<std::complex<double>>(h,1,true,false);
#else
    (void)h;
#endif
}
}
int main(int argc,char** argv)
{
    int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int P=0,rank=0;MPI_Comm_size(MPI_COMM_WORLD,&P);MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if(provided<MPI_THREAD_FUNNELED){if(!rank)std::cerr<<"MPI_THREAD_FUNNELED required\n";MPI_Abort(MPI_COMM_WORLD,1);}
    TestOptions options;std::string error;
    if(!parse_options(argc,argv,P,options,error)){if(!rank)std::cerr<<error<<"\n";MPI_Abort(MPI_COMM_WORLD,1);}
    for(auto grid:options.grids) {
        DdlaHandle_t h=nullptr;ddlaInit(h);ddlaSet(h,MPI_COMM_WORLD,grid.first,grid.second);
        body(h,options.shapes.front());check_ddla_sync(h);ddlaDestroy(h);
    }
    if(!rank)std::cout<<"test_workspace_gpu_replay passed\n";MPI_Finalize();return 0;
}
