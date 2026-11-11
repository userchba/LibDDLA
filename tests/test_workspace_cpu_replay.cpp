#include "api_grid_test_common.h"
#include "ddla_stream.h"
#include "Eigensolver/two_stage/hb2st_cpu_replay_workspace.h"
#include "workspace_test_common.h"
#include <array>
#include <cstring>
#include <type_traits>
using namespace api_grid_test;
using namespace ddla::detail;
static_assert(std::is_trivially_copyable_v<CpuReplayPolicy>);
static_assert(std::is_trivially_copyable_v<Hb2stCpuReplayPlan>);
namespace {
bool no_growth(const workspace_test::Snapshot& s){return !s.ddla_alloc&&!s.ddla_free&&!s.pinned_alloc&&!s.raw_gpu_alloc&&!s.host_new&&!s.vendor_query&&!s.resource_setup;}
bool pure(const workspace_test::Snapshot& s){return no_growth(s)&&!s.copies&&!s.launches&&!s.syncs&&!s.collectives;}
void clean(DdlaHandle_t h,const workspace_test::Snapshot& s,bool query=false){require_close(h,query?"CPU replay pure local query":"CPU replay allocation/resource counters",(query?pure(s):no_growth(s))?0:1,0);}
// Exactly sized scratch between poisoned guards (backend or host bytes: both
// are host memory on the CPU handle).
class Guarded {
    static constexpr std::size_t guard=256;DdlaHandle_t h_;std::size_t bytes_;ddla::vector<char> storage_;
public:
    Guarded(DdlaHandle_t h,std::size_t bytes):h_(h),bytes_(bytes),storage_(h,bytes+2*guard){std::memset(storage_.data(),0x69,storage_.size());}
    char* data(){return storage_.data()+guard;}
    void verify(){const char* w=storage_.data();bool okay=true;for(std::size_t i=0;i<guard;++i)okay=okay&&w[i]==0x69&&w[guard+bytes_+i]==0x69;require_close(h_,"CPU replay exact buffer guards",okay?0:1,0);}
};
template<class T>std::complex<long double> wide(T x){return {static_cast<long double>(x),0};}
template<class R>std::complex<long double> wide(std::complex<R> x){return {x.real(),x.imag()};}
template<class T>T scalar(double x,double y=0){if constexpr(std::is_arithmetic_v<T>)return T(x);else return T(x,y);}
constexpr int input_column_period=47;
template<class R>R value(int row,int column){return R((row*17+column*13)%input_column_period-23)/R(input_column_period);}
template<class T>std::vector<hb2st_rot<T>> records_for(int n,int kd){
    using R=typename real_type<T>::type;std::vector<hb2st_rot<T>> records;ChaseCursor cursor(n,kd);
    while(!cursor.done()){for(int t=0;t<cursor.count();++t){const auto g=records.size();const double angle=.003*(int((g*7)%101)-50);const R c=R(std::cos(angle)),s=R(std::sin(angle));T sine;if constexpr(std::is_arithmetic_v<T>)sine=s;else {const double phase=.013*((g*11)%71);sine=T(s*R(std::cos(phase)),s*R(std::sin(phase)));}records.push_back({cursor.first()+t*(kd+1),c,sine});}cursor.next();}return records;
}
template<class T>std::vector<T> phases_for(int n,bool enabled){std::vector<T> phases;if constexpr(!std::is_arithmetic_v<T>)if(enabled)for(int i=1;i<n;++i)phases.push_back(scalar<T>(std::cos(i*.007),std::sin(i*.007)));return phases;}
template<class T>std::vector<std::complex<long double>> reference(int n,int column,const std::vector<hb2st_rot<T>>& records,const std::vector<T>& phases){
    using R=typename real_type<T>::type;std::vector<std::complex<long double>> result(n);for(int row=0;row<n;++row)result[row]=wide(value<R>(row,column))*(row&&!phases.empty()?wide(phases[row-1]):std::complex<long double>(1,0));
    for(auto p=records.rbegin();p!=records.rend();++p){const int b=p->j0,a=b-1;if(a<0||b>=n)continue;const auto x=result[a],y=result[b],s=wide(p->s);result[a]=static_cast<long double>(p->c)*x-std::conj(s)*y;result[b]=s*x+static_cast<long double>(p->c)*y;}return result;
}
template<class T>void verify_output(DdlaHandle_t h,int n,const int*desc,const std::vector<T>& before,const std::vector<T>& output,const std::vector<hb2st_rot<T>>& records,const std::vector<T>& phases){
    using R=typename real_type<T>::type;double error=0;bool padding=true;const int rows=ddla_test::m_loc(h,desc),cols=ddla_test::n_loc(h,desc),ld=desc[DDLA_LLD_];
    // Inputs repeat every 47 columns; rotations and phases act only on rows.
    // Cache those independent reverse-order references while still comparing
    // every output element. This bounds the multi-chunk oracle's repeat work.
    std::vector<std::vector<std::complex<long double>>> references(std::min(n,input_column_period));
    for(int c=0;c<cols;++c){const int global=indx_l2g_c(desc,h,c);const std::vector<std::complex<long double>>* ref=nullptr;
        if(global<n){auto& cached=references[global%input_column_period];if(cached.empty())cached=reference(n,global,records,phases);ref=&cached;}
        for(int r=0;r<ld;++r){const auto at=r+std::size_t(c)*ld;const int row=r<rows?indx_l2g_r(desc,h,r):n;if(row<n&&global<n)error=std::max(error,workspace_test::finite_error(double(std::abs(wide(output[at])-(*ref)[row]))));else padding=padding&&!std::memcmp(&output[at],&before[at],sizeof(T));}}
    if(!cols)padding=output==before;require_close(h,"CPU replay independent reverse-order reference",error/(std::max(n,1)*std::numeric_limits<R>::epsilon()),60);require_close(h,"CPU replay parent/leading-dimension padding",padding?0:1,0);
}
template<class T>void rows_case(DdlaHandle_t h,int n,int kd,int panel,bool phases_enabled,bool require_multichunk=false){
    using R=typename real_type<T>::type;const auto records=records_for<T>(n,kd);const auto phases=phases_for<T>(n,phases_enabled);DistributedHistory<T> history;history.prepare(n,kd,h->comm);
    require_close(h,"CPU replay generated history record count",records.size()==std::size_t(history.total)?0:1,0);
    if(require_multichunk){
        require_close(h,"large CPU replay crosses a history chunk",history.chunks()>=2?0:1,0);
        require_close(h,"large CPU replay has records beyond the actual chunk",records.size()>std::size_t(history.chunk_size)?0:1,0);
        // n=800,kd=8 emits 279300 records. The second chunk starts five
        // records into a 22-record chase step, testing offset reconstruction.
        bool split=true;if(history.rank==1%history.ranks){const auto local=std::size_t(1/history.ranks);split=local<history.headers.size();if(split){const auto& header=history.headers[local];split=header.offset>0&&header.cursor.base+header.offset==history.chunk_size&&header.count==std::min(history.chunk_size,history.total-history.chunk_size);}}
        require_close(h,"large CPU replay chunk checkpoint splits a chase step",split?0:1,0);
    }
    if(!records.empty())history.capture(0,history.total,records.data(),history.total);
    int desc[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(desc,h,n+5,n+7,3,4,h->nprows_-1,h->npcols_-1));desc[DDLA_LLD_]+=5;const RowShare share(n,h->nprocs);const int ld=std::max(share.count(h->myid),1)+3;
    std::vector<R> input(std::size_t(ld)*std::max(n,1),std::numeric_limits<R>::quiet_NaN());for(int c=0;c<n;++c)for(int r=0;r<share.count(h->myid);++r)input[r+std::size_t(c)*ld]=value<R>(share.lo(h->myid)+r,c);
    const std::vector<T> before(std::max(std::size_t(1),local_size(h,desc)),scalar<T>(139));DeviceBuffer<R>Y(h,input.size());DeviceBuffer<T>Z(h,before.size());upload(h,Y.ptr,input);upload(h,Z.ptr,before);check_ddla_sync(h);
    const auto p=hb2st_cpu_replay_check<T>(h,n,Y.ptr,ld,history,phases.data(),phases.size(),Z.ptr,desc,panel);MPI_Barrier(h->comm);bool local=true;if(!h->myid){workspace_test::Scope trace(true);std::size_t bytes=0;hb2st_cpu_replay_workspace_size<T>(h,p,&bytes);local=pure(trace.finish());}MPI_Barrier(h->comm);require_close(h,"one-rank CPU replay query completes",local?0:1,0);
    workspace_test::Scope query(true);std::size_t bytes=0;hb2st_cpu_replay_workspace_size<T>(h,p,&bytes);clean(h,query.finish(),true);Guarded arena(h,bytes);R* original=Y.ptr;
    for(int pass=0;pass<2;++pass){upload(h,Y.ptr,input);upload(h,Z.ptr,before);check_ddla_sync(h);int info=-1;
        if(pass==0){
            // Neither run may depend on what the buffer held before.
            std::memset(arena.data(),0xa7,bytes);workspace_test::Scope gather;hb2st_cpu_replay_input_run<T>(h,p,Y.ptr,phases.data(),arena.data(),bytes);clean(h,gather.finish());
            const auto unchanged=download(h,Y.ptr,input.size());require_close(h,"CPU replay input phase preserves const Y",std::memcmp(unchanged.data(),input.data(),input.size()*sizeof(R))?1:0,0);
            // Both input reads and communication have joined. Corrupt Y before
            // apply; the tiled block the input run left in the buffer alone
            // must determine the answer, including complex phases.
            DDLA_CHECK(ddlaMemset(h,Y.ptr,0,input.size()*sizeof(R)));workspace_test::Scope apply;hb2st_cpu_replay_apply_run<T>(h,p,history,Z.ptr,info,arena.data(),bytes);clean(h,apply.finish());
        }else {workspace_test::Scope run;hb2st_cpu_replay_run<T>(h,p,history,phases.data(),Y.ptr,Z.ptr,info,arena.data(),bytes);clean(h,run.finish());const auto unchanged=download(h,Y.ptr,input.size());require_close(h,"combined CPU replay preserves const Y",std::memcmp(unchanged.data(),input.data(),input.size()*sizeof(R))?1:0,0);}
        require_close(h,"CPU replay retains input pointer",Y.ptr==original?0:1,0);require_close(h,"CPU replay info",info,0);verify_output(h,n,desc,before,download(h,Z.ptr,before.size()),records,phases);arena.verify();
    }
    // Y may live in the same buffer behind the input run's front, which the apply run then reuses.
    if(n){std::size_t full=0,front=0;workspace_test::Scope query(true);hb2st_cpu_replay_workspace_size<T>(h,p,&full);hb2st_cpu_replay_input_workspace_size<T>(h,p,&front);clean(h,query.finish(),true);
        const std::size_t total=std::max(full,front+padded_bytes<R>(input.size()));Guarded shared(h,total);
        for(int pass=0;pass<2;++pass){R* const slab=reinterpret_cast<R*>(shared.data()+front);std::memcpy(slab,input.data(),input.size()*sizeof(R));upload(h,Z.ptr,before);int info=-1;
            workspace_test::Scope gather;hb2st_cpu_replay_input_run<T>(h,p,slab,phases.data(),shared.data(),total);clean(h,gather.finish());require_close(h,"same-buffer input is live until redistribution completes",std::memcmp(slab,input.data(),input.size()*sizeof(R))?1:0,0);
            // Y's lifetime ends at this boundary. Applying the history may now
            // reuse its bytes without growing the caller's buffer.
            workspace_test::Scope apply;hb2st_cpu_replay_apply_run<T>(h,p,history,Z.ptr,info,shared.data(),total);clean(h,apply.finish());verify_output(h,n,desc,before,download(h,Z.ptr,before.size()),records,phases);require_close(h,"same-buffer CPU replay info",info,0);shared.verify();}
    }
}
template<class T>void pack_cases(DdlaHandle_t h){
    using R=typename real_type<T>::type;constexpr int n=97,ncs=32,kd=8;auto records=records_for<T>(n,kd);const auto phases=phases_for<T>(n,true);std::vector<R> input(std::size_t(n)*ncs);std::vector<int> columns(ncs);for(int j=0;j<ncs;++j){columns[j]=j;for(int i=0;i<n;++i)input[i+std::size_t(j)*n]=value<R>(i,j);}std::vector<T> output(std::size_t(n)*ncs);const int sd=0;
    for(int tile:{8,16,32,64})for(int mode:{0,1,2}){CpuReplayPolicy policy;policy.threads=2;policy.tile=tile;policy.waves=mode!=0;policy.wave_mode=mode;policy.row_block=32;policy.level_block=4;
        const auto p=cpu_replay_pack_geometry<T>(n,ncs,3,1,kd,records.size(),CpuReplaySource::Records,policy);workspace_test::Scope query(true);std::size_t bytes=0;cpu_replay_pack_workspace_size<T>(p,&bytes);clean(h,query.finish(),true);Guarded arena(h,bytes);
        for(int pass=0;pass<2;++pass){workspace_test::Scope run;cpu_replay_pack_run<T>(p,input.data(),n,columns.data(),records.data(),nullptr,nullptr,h->comm,std::is_arithmetic_v<T>?nullptr:phases.data(),&sd,output.data(),arena.data(),bytes);clean(h,run.finish());double error=0;for(int j=0;j<ncs;++j){const auto ref=reference(n,j,records,phases);for(int i=0;i<n;++i)error=std::max(error,workspace_test::finite_error(double(std::abs(wide(output[std::size_t(i)*ncs+j])-ref[i]))));}require_close(h,"plain/wave/skew/tile replay reference",error/(n*std::numeric_limits<R>::epsilon()),60);arena.verify();}
    }
    // A valid rotation stream that is not this chase schedule must retain
    // literal reverse order rather than incorrectly applying wave grouping.
    {auto irregular=records;++irregular[0].j0;CpuReplayPolicy policy;policy.threads=2;policy.tile=16;
        const auto p=cpu_replay_pack_geometry<T>(n,ncs,3,1,irregular[0].j0,irregular.size(),CpuReplaySource::Records,policy);std::size_t bytes=0;cpu_replay_pack_workspace_size<T>(p,&bytes);Guarded work(h,bytes);
        workspace_test::Scope run;cpu_replay_pack_run<T>(p,input.data(),n,columns.data(),irregular.data(),nullptr,nullptr,h->comm,std::is_arithmetic_v<T>?nullptr:phases.data(),&sd,output.data(),work.data(),bytes);clean(h,run.finish());
        require_close(h,"irregular stream keeps plain ordering",cpu_replay_last_used_waves()?1:0,0);double error=0;for(int j=0;j<ncs;++j){const auto ref=reference(n,j,irregular,phases);for(int i=0;i<n;++i)error=std::max(error,workspace_test::finite_error(double(std::abs(wide(output[std::size_t(i)*ncs+j])-ref[i]))));}require_close(h,"irregular reverse-order reference",error/(n*std::numeric_limits<R>::epsilon()),60);work.verify();}
    MPI_Comm node=MPI_COMM_NULL;MPI_CHECK(MPI_Comm_split_type(h->comm,MPI_COMM_TYPE_SHARED,h->myid,MPI_INFO_NULL,&node));CpuReplayPolicy policy;policy.threads=2;policy.tile=16;policy.wave_mode=2;const auto plan=cpu_node_wave_check<T>(node,n,records.data(),records.size(),policy);workspace_test::Scope query(true);std::size_t bytes=0;cpu_node_wave_workspace_size<T>(plan,&bytes);clean(h,query.finish(),true);Guarded arena(h,bytes);CpuNodeWaveResources<T> resources(node,plan);
    try{for(int pass=0;pass<2;++pass){WaveShared<T> view;workspace_test::Scope fill;const bool okay=cpu_node_wave_fill_run<T>(plan,resources,records.data(),view,arena.data(),bytes);clean(h,fill.finish());require_close(h,"prepared shared schedule accepts chase order",okay?0:1,0);
            const auto p=cpu_replay_pack_geometry<T>(n,ncs,3,1,kd,records.size(),CpuReplaySource::Shared,policy);std::size_t pack_bytes=0;cpu_replay_pack_workspace_size<T>(p,&pack_bytes);Guarded work(h,pack_bytes);workspace_test::Scope run;cpu_replay_pack_run<T>(p,input.data(),n,columns.data(),nullptr,&view,nullptr,h->comm,std::is_arithmetic_v<T>?nullptr:phases.data(),&sd,output.data(),work.data(),pack_bytes);clean(h,run.finish());double error=0;for(int j=0;j<ncs;++j){const auto ref=reference(n,j,records,phases);for(int i=0;i<n;++i)error=std::max(error,workspace_test::finite_error(double(std::abs(wide(output[std::size_t(i)*ncs+j])-ref[i]))));}require_close(h,"shared replay independent reference",error/(n*std::numeric_limits<R>::epsilon()),60);arena.verify();work.verify();}resources.release();}catch(...){resources.release();MPI_Comm_free(&node);throw;}MPI_CHECK(MPI_Comm_free(&node));
}
void column_layout_case(DdlaHandle_t h) {
    constexpr int n=37;int desc[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(desc,h,n,n,3,4,h->nprows_-1,h->npcols_-1));
    const auto p=cpu_replay_plan(h,n,desc,7);const std::size_t bytes=cpu_replay_layout_bytes(p);Guarded work(h,bytes);
    for(int repeat=0;repeat<3;++repeat) {
        std::memset(work.data(),0x97,bytes);workspace_test::Scope trace;char* cursor=work.data();const auto map=cpu_replay_layout(p,cursor);clean(h,trace.finish());
        bool valid=map.column_offsets[p.P]==n&&cursor==work.data()+bytes;
        for(int rank=0;rank<p.P;++rank){const int* const columns=map.column(rank);const int row=p.major=='R'?rank/p.npc:rank%p.npr,col=p.major=='R'?rank%p.npc:rank/p.npr;
            const long long all=num_loc(n,p.nb,col,p.csrc,p.npc);const int begin=int(all*row/p.npr);
            valid=valid&&map.column_count(rank)==p.column_size(rank);
            for(int j=0;j<map.column_count(rank);++j)valid=valid&&columns[j]==indxl2g(begin+j,p.nb,col,p.csrc,p.npc);
        }
        require_close(h,"CPU replay rebuilds flat column mapping in reused bytes",valid?0:1,0);work.verify();
    }
}
void scratch_free_scatter(DdlaHandle_t h){
    constexpr int rows=3,cols=35,ld=5;const int rd[2]={0,rows*17},share[3]={0,17,cols};std::vector<int> columns(cols);for(int j=0;j<cols;++j)columns[j]=j;std::vector<double> recv(rows*cols),checked(ld*cols,-91),borrowed=checked;
    for(int p=0;p<2;++p)for(int i=0;i<rows;++i)for(int j=share[p];j<share[p+1];++j)recv[rd[p]+i*(share[p+1]-share[p])+j-share[p]]=i+100*j;
    cpu_scatter_z_check(recv.data(),rd,share,2,cols,columns.data(),rows,borrowed.data(),ld,2);cpu_scatter_z(recv.data(),rd,share,2,cols,columns.data(),rows,checked.data(),ld,2);
    workspace_test::Scope trace;cpu_scatter_z_run(recv.data(),rd,share,2,cols,columns.data(),rows,borrowed.data(),ld,2);clean(h,trace.finish());require_close(h,"scratch-free scatter checked/run equivalence",checked==borrowed?0:1,0);
}
void run(MPI_Comm comm,int pr,int pc,char major){DdlaHandle_t h=nullptr;ddlaInit(h,DdlaBackend::CPU);ddlaSet(h,comm,pr,pc,major);column_layout_case(h);scratch_free_scatter(h);
#define CASE(T) rows_case<T>(h,37,4,3,true);rows_case<T>(h,3,2,1,false);rows_case<T>(h,1,1,64,false);rows_case<T>(h,0,1,64,false);pack_cases<T>(h)
    CASE(float);CASE(double);CASE(std::complex<float>);CASE(std::complex<double>);
#undef CASE
    rows_case<double>(h,800,8,64,true,true);if(!h->myid)std::cout<<"workspace CPU replay PASS grid="<<pr<<"x"<<pc<<" major="<<major<<std::endl;ddlaDestroy(h);}
}
int main(int argc,char**argv){ddla_test_mpi_init(&argc, &argv);int size;MPI_Comm_size(MPI_COMM_WORLD,&size);TestOptions options;std::string error;if(!parse_options(argc,argv,size,options,error)){std::cerr<<error<<std::endl;MPI_Abort(MPI_COMM_WORLD,2);}for(auto grid:options.grids)for(char major:{'R','C'}){
    run(MPI_COMM_WORLD,grid.first,grid.second,major);
}MPI_Finalize();return 0;}
