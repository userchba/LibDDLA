#include "api_grid_test_common.h"
#include "ddla_stream.h"
#include "Eigensolver/two_stage/hb2st_replay_gpu.h"
#include "Eigensolver/two_stage/hb2st_replay_storage.h"
#include "row_share.h"
#include "workspace_test_common.h"

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>

using namespace api_grid_test;
namespace {
template<class T> std::complex<long double> wide(T v) { return {static_cast<long double>(v),0}; }
template<class R> std::complex<long double> wide(std::complex<R> v) { return {static_cast<long double>(v.real()),static_cast<long double>(v.imag())}; }
template<class R> R yvalue(int r,int c) { return R((r*17+c*13)%47-23)/R(47); }

template<class T>
std::vector<ddla::hb2st_rot<T>> make_records(int n,int kd)
{
    using R=typename ddla::real_type<T>::type;
    std::vector<ddla::hb2st_rot<T>> result;
    const int kdn=std::min(n-1,kd);
    if(kd<=1 || n<3) return result;
    int j1=kdn+2,j2=1;
    for(int i=1;i<=n-2;++i) for(int k=kdn+1;k>=2;--k) {
        j1+=kdn; j2+=kdn;
        if(k>2) j1-=kd+1;
        for(int j=j1-1;j<j2;j+=kd+1) if(j>=1 && j<n) {
            const long long g=static_cast<long long>(result.size());
            const double angle=0.003*((g*7)%101-50);
            const R c=R(std::cos(angle)), mag=R(std::sin(angle));
            T s;
            if constexpr(std::is_same<T,R>::value) s=mag;
            else { const double phi=0.013*((g*11)%71); s=T(mag*R(std::cos(phi)),mag*R(std::sin(phi))); }
            ddla::hb2st_rot<T> record; record.j0=j; record.c=c; record.s=s;
            result.push_back(record);
        }
        if(j2+kdn>n) j2-=kd+1;
    }
    return result;
}

template<class T>
void check_replay(const ddla::DdlaHandle_t& h,int n,int kd,int panel,int replay,bool shifted,bool with_phases,bool waves=true,bool reuse=false,bool empty_owner=false)
{
    using R=typename ddla::real_type<T>::type;
    int npr=0,npc=0; ddlaGetGridDims(h,npr,npc);
    const int rank=ddlaGetRank(h),P=npr*npc;
    const ddla::detail::RowShare rows(n,P);
    const int nr=rows.count(rank),ldy=std::max(nr,1)+(reuse?0:3);
    const auto records=make_records<T>(n,kd);
    ddla::detail::DistributedHistory<T> history;
    history.prepare(n,kd,ddlaGetCommunicator(h));
    if(history.total!=static_cast<long long>(records.size())) throw std::runtime_error("test history count mismatch");
    if(!records.empty()) history.capture(0,history.total,records.data(),history.total);
    std::vector<T> phases;
    if constexpr(!std::is_same<T,R>::value) if(with_phases) {
        for(int r=1;r<n;++r) phases.emplace_back(R(std::cos(r*0.007)),R(std::sin(r*0.007)));
    }
    int desc[ddla::DDLA_DLEN_];
    ddlaDescInit(desc,h,n+(reuse?0:5),n+(reuse?0:7),4,4,shifted?npr-1:0,shifted?npc-1:0);
    if(!reuse) desc[ddla::DDLA_LLD_]+=5;
    const int lld=desc[ddla::DDLA_LLD_],zr=ddla_test::m_loc(h,desc),zc=ddla_test::n_loc(h,desc);
    const T sentinel=T(R(139));
    std::vector<T> before(std::max<std::size_t>(local_size(h,desc),1),sentinel);
    for(int c=0;c<zc;++c) for(int r=0;r<zr;++r)
        if(ddla::indx_l2g_r(desc,h,r)<n && ddla::indx_l2g_c(desc,h,c)<n)
            before[r+static_cast<std::size_t>(c)*lld]=T(std::numeric_limits<R>::quiet_NaN());
    std::vector<R> input(static_cast<std::size_t>(ldy)*std::max(n,1),std::numeric_limits<R>::quiet_NaN());
    for(int c=0;c<n;++c) for(int r=0;r<nr;++r) input[r+static_cast<std::size_t>(c)*ldy]=yvalue<R>(rows.lo(rank)+r,c);
    const bool use_storage=reuse && P>1;
    DeviceBuffer<R> Y(h,use_storage?1:input.size()); DeviceBuffer<T> Z(h,before.size());
    DeviceBuffer<T> A(h,use_storage?before.size():1);
    std::vector<T> original_A(use_storage?before.size():1,T(R(53)));
    if(use_storage) for(std::size_t i=0;i<original_A.size();++i) original_A[i]=T(R((i*7)%991)/R(991));
    upload(h,A.ptr,original_A); upload(h,Z.ptr,before); check_ddla_sync(h);
    int info=-1;
    if(use_storage) {
        ddla::detail::Hb2stReplayStorage<T> storage(h,n,A.ptr,desc,Z.ptr,desc);
        require_close(h,"reuse enabled for compact matrices",!storage.enabled(),0);
        storage.prepare();
        const bool fits=nr>0 && before.size()*sizeof(T)>=input.size()*sizeof(R);
        require_close(h,"reuse input borrowing capacity",(storage.input()==reinterpret_cast<R*>(Z.ptr))!=fits,0);
        require_close(h,"reuse input leading dimension",storage.ldy()!=ldy,0);
        const auto& offsets=storage.column_offsets();
        const int main=static_cast<int>(before.size()/n),count=offsets[rank+1]-offsets[rank];
        require_close(h,"reuse main plus at most one spill",count<main || count>main+1,0);
        // Simulate stage3 overwriting its lent A scratch; restore must recover
        // every original local A entry, not only the bytes replay touched.
        if(ddlaMemset(h,A.ptr,0,before.size()*sizeof(T))!=ddlaStatus_t::DDLA_STATUS_SUCCESS)
            throw std::runtime_error("reuse test scratch initialization failed");
        upload(h,storage.input(),input); check_ddla_sync(h);
        storage.replay(history,phases,info,panel,replay,waves);
        require_close(h,"reuse consumed input",storage.input()!=nullptr,0);
        storage.restore();
        const auto restored=download(h,A.ptr,original_A.size());
        require_close(h,"reuse restores A exactly",std::memcmp(restored.data(),original_A.data(),original_A.size()*sizeof(T))!=0,0);
    } else {
        upload(h,Y.ptr,input);
        // A nonempty replay with rank zero owning no eigenvector columns.
        // Input row ownership remains unchanged, so this rank still sends
        // input, fetches all history and receives its final block-cyclic Z.
        if(empty_owner && P>1) {
            std::vector<int> offsets(P+1);
            for(int q=0;q<P;++q) offsets[q]=rows.lo(q);
            offsets[P]=n; offsets[1]=0;
            const std::size_t count=static_cast<std::size_t>(n)*(offsets[rank+1]-offsets[rank]);
            DeviceBuffer<T> main(h,std::max<std::size_t>(count,1));
            ddla::detail::Hb2stReplayGpuWorkspace<T> workspace;
            workspace.main=main.ptr;workspace.main_elements=count;workspace.column_offsets=&offsets;
            ddla::detail::hb2st_replay_rows_gpu<T>(h,n,Y.ptr,ldy,history,phases,Z.ptr,desc,info,panel,replay,waves,&workspace);
        } else {
            ddla::detail::hb2st_replay_rows_gpu<T>(h,n,Y.ptr,ldy,history,phases,Z.ptr,desc,info,panel,replay,waves);
        }
    }
    check_ddla_sync(h);
    if(!use_storage) require_close(h,"replay consumed input",Y.ptr!=nullptr,0);
    require_close(h,"replay info",info,0);
    const auto after=download(h,Z.ptr,before.size());
    double bad=0,padding=0,error=0;
    for(int c=0;c<zc;++c) for(int r=0;r<lld;++r) {
        const int gr=r<zr?ddla::indx_l2g_r(desc,h,r):n;
        const int gc=ddla::indx_l2g_c(desc,h,c);
        const auto index=r+static_cast<std::size_t>(c)*lld;
        if(gr<n && gc<n) {
            const auto x=wide(after[index]);
            if(!std::isfinite(x.real()) || !std::isfinite(x.imag())) ++bad;
        } else padding+=after[index]!=sentinel;
    }
    if(zc==0) for(const auto& v:after) padding+=v!=sentinel;
    std::vector<int> probes;
    if(n<=64) for(int c=0;c<n;++c) probes.push_back(c);
    else probes={0,n/2,n-1};
    // Literal reverse row updates in long double are independent of the
    // GPU level permutation, packing, and row/column layouts.
    for(int c:probes) {
        std::vector<std::complex<long double>> ref(n);
        for(int r=0;r<n;++r) ref[r]=wide(yvalue<R>(r,c))*(r>0 && !phases.empty()?wide(phases[r-1]):std::complex<long double>(1,0));
        for(auto it=records.rbegin();it!=records.rend();++it) {
            const int b=it->j0,a=b-1;
            const auto x=ref[a],y=ref[b],s=wide(it->s);
            const long double cs=it->c;
            ref[a]=cs*x-std::conj(s)*y; ref[b]=s*x+cs*y;
        }
        for(int lc=0;lc<zc;++lc) if(ddla::indx_l2g_c(desc,h,lc)==c)
            for(int lr=0;lr<zr;++lr) {
                const int gr=ddla::indx_l2g_r(desc,h,lr);
                if(gr<n) error=std::max(error,static_cast<double>(std::abs(wide(after[lr+static_cast<std::size_t>(lc)*lld])-ref[gr])));
            }
    }
    const std::string tag="GPU replay n="+std::to_string(n)+" type="+std::to_string(sizeof(T))+" panel="+std::to_string(panel)+" records="+std::to_string(replay)+" shifted="+std::to_string(shifted)+" waves="+std::to_string(waves)+" reuse="+std::to_string(use_storage);
    require_close(h,tag+" finite",bad,0); require_close(h,tag+" padding",padding,0);
    require_close(h,tag+" dense reference",error/(std::max(n,1)*std::numeric_limits<R>::epsilon()),50);
}

void check_storage_eligibility(const ddla::DdlaHandle_t& h)
{
    int npr=0,npc=0; ddlaGetGridDims(h,npr,npc);
    const int P=npr*npc,rank=ddlaGetRank(h);
    constexpr int n=37;
    DeviceBuffer<double> A(h,4096),Z(h,4096);
    const std::vector<double> before(4096,139);
    upload(h,A.ptr,before); upload(h,Z.ptr,before); check_ddla_sync(h);
    for(int mode=0;mode<8;++mode) {
        int da[ddla::DDLA_DLEN_],dz[ddla::DDLA_DLEN_];
        const int order=mode==6?1:n;
        const int parent=order+(mode==2?5:0);
        ddlaDescInit(da,h,parent,parent,4,4,0,0);
        ddlaDescInit(dz,h,parent,parent,mode==3?8:4,mode==3?8:4,0,0);
        if(mode==1 || (mode==5 && rank==P-1)) { da[ddla::DDLA_LLD_]+=3; dz[ddla::DDLA_LLD_]+=3; }
        ddla::detail::Hb2stReplayStorage<double> storage(h,order,A.ptr,da,mode==4?A.ptr+1:mode==7?A.ptr:Z.ptr,dz);
        require_close(h,"reuse eligibility mode="+std::to_string(mode),storage.enabled()!=(mode==0 && P>1),0);
        if(!storage.enabled()) { storage.prepare(); storage.restore(); }
    }
    const auto afterA=download(h,A.ptr,before.size()),afterZ=download(h,Z.ptr,before.size());
    require_close(h,"reuse fallback leaves A unchanged",afterA!=before,0);
    require_close(h,"reuse fallback leaves Z unchanged",afterZ!=before,0);
}

// Repeated borrowed execution no longer allocates panel vectors. Fault the
// real post-consumption cosine upload through the diagnostic preload instead.
// Use one rank so a deliberately failing launch interval cannot strand peers.
void check_borrowed_copy_failure(const ddla::DdlaHandle_t& h)
{
    if(h->nprocs!=1)return;
    using Arm=void(*)(const void*,uint64_t);using Count=uint64_t(*)();Arm arm=nullptr;Count faults=nullptr;
    void* a=dlsym(RTLD_DEFAULT,"workspace_trace_fail_memcpy_after_upload");void* f=dlsym(RTLD_DEFAULT,"workspace_trace_memcpy_faults");
    std::memcpy(&arm,&a,sizeof(arm));std::memcpy(&faults,&f,sizeof(faults));
    if(!arm||!faults||!workspace_test::available()) {
        std::cout<<"GPU replay post-consumption copy fault: not tested without diagnostic preload"<<std::endl;return;
    }
    constexpr int n=37,kd=4;int d[ddla::DDLA_DLEN_];ddlaDescInit(d,h,n,n,4,4,0,0);
    const auto records=make_records<double>(n,kd);ddla::detail::DistributedHistory<double> history;
    history.prepare(n,kd,h->comm);history.capture(0,history.total,records.data(),history.total);
    DeviceBuffer<double>Y(h,n*n),Z(h,n*n);std::vector<double> input(n*n,0.25);upload(h,Y.ptr,input);check_ddla_sync(h);
    const auto p=ddla::detail::hb2st_replay_gpu_check<double>(h,n,Y.ptr,n,history,nullptr,0,Z.ptr,d,0,0,true);
    std::size_t bb=0,hb=0;ddla::detail::hb2st_replay_gpu_workspace_size<double>(h,p,&bb,&hb);
    ddla::vector<char> backend(h,bb);std::vector<char> host(hb);
    ddla::detail::Hb2stReplayGpuResources<double> resources(h,p);
    // The host history chunk precedes the cosine array in the carving order.
    arm(host.data()+ddla::detail::padded_bytes<ddla::hb2st_rot<double>>(p.record_capacity),static_cast<uint64_t>(p.rcmax)*sizeof(double));
    bool consumed=false,caught=false;double* original=Y.ptr;int info=0;
    workspace_test::Scope scope;
    try{ddla::detail::hb2st_replay_gpu_run<double>(h,p,resources,history,nullptr,Y.ptr,Z.ptr,info,&consumed,backend.data(),bb,host.data(),hb);}
    catch(const std::exception&){caught=true;}
    const auto snapshot=scope.finish();const auto fired=faults();arm(nullptr,0);
    // Pageable host scratch can be reused after a completed host-to-device copy.
    std::memset(host.data(),0x39,host.size());
    require_close(h,"borrowed replay post-consumption copy fault",caught&&consumed&&fired==1?0:1,0);
    require_close(h,"borrowed replay failure retains Y ownership",Y.ptr==original?0:1,0);
    require_close(h,"borrowed replay exceptional upload drains before arena reuse",snapshot.syncs>=2?0:1,0);
    const auto retained=download(h,Y.ptr,input.size());require_close(h,"borrowed replay failure keeps input bytes",
        std::memcmp(retained.data(),input.data(),input.size()*sizeof(double))?1:0,0);
}

void checks(const ddla::DdlaHandle_t& h)
{
    check_replay<double>(h,37,4,7,17,true,false,true,false,true);
    check_replay<std::complex<double>>(h,37,4,7,17,true,true,true,false,true);
    check_storage_eligibility(h);
    check_borrowed_copy_failure(h);
    check_replay<float>(h,37,4,7,17,false,false,true,true);
    check_replay<double>(h,37,4,7,17,false,false,true,true);
    check_replay<std::complex<float>>(h,37,4,7,17,false,true,true,true);
    check_replay<std::complex<double>>(h,37,4,7,17,false,true,true,true);
    check_replay<std::complex<double>>(h,37,4,1,17,true,true,false,true);
    check_replay<std::complex<double>>(h,83,16,7,257,true,true,true,true);
    check_replay<double>(h,800,16,0,0,false,false,true,true);
    check_replay<std::complex<double>>(h,800,16,0,0,false,true,true,true);
    for(bool shifted:{false,true}) for(int panel:{1,7,0}) {
        check_replay<float>(h,37,4,panel,17,shifted,false);
        check_replay<double>(h,37,4,panel,17,shifted,false);
        check_replay<std::complex<float>>(h,37,4,panel,17,shifted,true);
        check_replay<std::complex<double>>(h,37,4,panel,17,shifted,true);
    }
    for(int n:{0,1,2}) {
        check_replay<float>(h,n,1,1,0,true,false);
        check_replay<double>(h,n,1,1,0,true,false);
        check_replay<std::complex<float>>(h,n,1,1,0,true,true);
        check_replay<std::complex<double>>(h,n,1,1,0,true,true);
    }
    check_replay<std::complex<double>>(h,37,4,7,0,false,false);
    // Retain an independently scheduled GPU path as well as the literal
    // long-double reference; both paths see the same synthetic rotations.
    check_replay<float>(h,37,4,7,17,true,false,false);
    check_replay<double>(h,37,4,7,17,true,false,false);
    check_replay<std::complex<float>>(h,37,4,7,17,true,true,false);
    check_replay<std::complex<double>>(h,37,4,7,17,true,true,false);
    // Odd subchunk lengths split recording steps and domino runs at many
    // positions, including both K16 real and K8 complex wave boundaries.
    for(int records:{31,33,257}) {
        check_replay<double>(h,83,16,7,records,true,false);
        check_replay<std::complex<double>>(h,83,16,7,records,true,true);
    }
    // More than one retained-history chunk, including a chunk boundary that
    // cuts through a recording step and transfers ownership to another rank.
    check_replay<float>(h,800,16,0,0,false,false);
    check_replay<double>(h,800,16,0,0,false,false);
    check_replay<std::complex<float>>(h,800,16,0,0,false,true);
    check_replay<std::complex<double>>(h,800,16,0,0,false,true);
    check_replay<double>(h,800,16,0,0,false,false,false);
    check_replay<std::complex<double>>(h,800,16,0,0,false,true,false);
}
}
int main(int argc,char** argv)
{
    int provided=0; MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    try {
#if DDLA_HAS_GPU
        int P=0; MPI_Comm_size(MPI_COMM_WORLD,&P);
        TestOptions options; std::string error;
        if(!parse_options(argc,argv,P,options,error)) throw std::runtime_error(error);
        for(const auto grid:options.grids) for(char major:{'R','C'}) {
            ddla::DdlaHandle_t h=nullptr;
            ddla::ddlaInit(h,ddla::DdlaBackend::GPU); ddla::ddlaSet(h,MPI_COMM_WORLD,grid.first,grid.second,major);
            checks(h); ddla::ddlaDestroy(h);
        }
        int rank=0; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        if(rank==0) std::cout<<"test_pheevd_gpu_replay passed"<<std::endl;
#else
        throw std::runtime_error("test_pheevd_gpu_replay needs a GPU build");
#endif
    } catch(const std::exception& e) { std::cerr<<e.what()<<std::endl; MPI_Abort(MPI_COMM_WORLD,1); }
    MPI_Finalize(); return 0;
}
