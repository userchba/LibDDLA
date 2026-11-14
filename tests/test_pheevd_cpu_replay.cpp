// Consuming CPU replay: independent dense reference and input/output lifetime.
#include "api_grid_test_common.h"
#include "Eigensolver/two_stage/hb2st_cpu_replay_workspace.h"
#include "row_share.h"
#include <cstring>
#include <new>
#include <limits>
#include <thread>
#include <cerrno>
#include "allocation_probe_ddla.h"
#include "allocation_probe_caller.h"

extern "C" int ddla_test_external_allocations();

// Executable-only allocation injection. No production hooks. Only the main
// thread is armed. MPI may allocate on that same thread; host allocation
// wrappers therefore also check the owner before counting or injecting faults.
namespace allocation_probe {
thread_local bool enabled=false,fired=false,fail_after=false,resolving=false;
thread_local double** input=nullptr;
thread_local std::size_t before=0,after=0,fail_at=0;
void inspect() {
    if(!enabled) return;
    const bool consumed=*input==nullptr;
    std::size_t& count=consumed?after:before;++count;
    if(fail_at && consumed==fail_after && count==fail_at) {
        enabled=false;fired=true;throw std::bad_alloc();
    }
}
void begin(double** y,bool consumed,std::size_t index) {
    input=y;fail_after=consumed;fail_at=index;before=after=0;fired=false;enabled=true;
}
void inspect_host(void* caller) {
    if(!enabled || resolving) return;
    struct Guard {Guard(){resolving=true;} ~Guard(){resolving=false;}} guard;
    if(managed_caller(caller)) inspect();
}
void* allocate(std::size_t bytes,void* caller) {
    inspect_host(caller);void* p=std::malloc(bytes?bytes:1);if(!p) throw std::bad_alloc();return p;
}
int allocate_aligned(void** out,std::size_t alignment,std::size_t bytes,void* caller) noexcept {
    try {inspect_host(caller);} catch(const std::bad_alloc&) {return ENOMEM;}
    if(alignment<sizeof(void*) || (alignment&(alignment-1))) return EINVAL;
    if(bytes>std::numeric_limits<std::size_t>::max()-(alignment-1)) return ENOMEM;
    const std::size_t rounded=((std::max<std::size_t>(bytes,1)+alignment-1)/alignment)*alignment;
    void* p=std::aligned_alloc(alignment,rounded);if(!p) return ENOMEM;*out=p;return 0;
}
void* allocate(std::size_t bytes,std::align_val_t alignment,void* caller) {
    void* p=nullptr;if(allocate_aligned(&p,static_cast<std::size_t>(alignment),bytes,caller)) throw std::bad_alloc();return p;
}
}
// Preserve the original caller for every entry point, including array/aligned
// new; delegating to another interposing wrapper loses that boundary at -O0.
#define DDLA_ALLOCATION_CALLER __builtin_extract_return_addr(__builtin_return_address(0))
void* operator new(std::size_t n) {return allocation_probe::allocate(n,DDLA_ALLOCATION_CALLER);}
void* operator new[](std::size_t n) {return allocation_probe::allocate(n,DDLA_ALLOCATION_CALLER);}
extern "C" int posix_memalign(void** p,std::size_t a,std::size_t n) noexcept {
    return allocation_probe::allocate_aligned(p,a,n,DDLA_ALLOCATION_CALLER);
}
void* operator new(std::size_t n,std::align_val_t a) {return allocation_probe::allocate(n,a,DDLA_ALLOCATION_CALLER);}
void* operator new[](std::size_t n,std::align_val_t a) {return allocation_probe::allocate(n,a,DDLA_ALLOCATION_CALLER);}
#undef DDLA_ALLOCATION_CALLER
void operator delete(void* p) noexcept {std::free(p);}
void operator delete[](void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
void operator delete[](void* p,std::size_t) noexcept {std::free(p);}
void operator delete(void* p,std::align_val_t) noexcept {std::free(p);}
void operator delete[](void* p,std::align_val_t) noexcept {std::free(p);}
void operator delete(void* p,std::size_t,std::align_val_t) noexcept {std::free(p);}
void operator delete[](void* p,std::size_t,std::align_val_t) noexcept {std::free(p);}

using namespace api_grid_test;
namespace {
// Excluding vendor allocations must leave both counting and failure injection
// effective for actual scratch, before and after input consumption.
void check_allocation_probe(const ddla::DdlaHandle_t& h)
{
    using Allocate=void*(*)(std::size_t);
    using AlignedAllocate=void*(*)(std::size_t,std::align_val_t);
    Allocate volatile plain=static_cast<Allocate>(&::operator new);
    Allocate volatile array=static_cast<Allocate>(&::operator new[]);
    AlignedAllocate volatile aligned=static_cast<AlignedAllocate>(&::operator new);
    AlignedAllocate volatile aligned_array=static_cast<AlignedAllocate>(&::operator new[]);
    for(bool consumed:{false,true}) for(int kind=0;kind<6;++kind) for(bool fail:{false,true}) {
        double value=0; double* watched=consumed?nullptr:&value;
        allocation_probe::begin(&watched,consumed,fail?1:0);
        const bool external_ok=ddla_test_external_allocations()!=0;
        const bool external_clean=allocation_probe::before==0 && allocation_probe::after==0 && !allocation_probe::fired;
        void* p=nullptr; bool rejected=false;
        try {
            switch(kind) {
            case 0: p=plain(128); break;
            case 1: p=array(128); break;
            case 2: p=aligned(128,std::align_val_t(64)); break;
            case 3: p=aligned_array(128,std::align_val_t(64)); break;
            case 4: rejected=posix_memalign(&p,64,128)!=0; break;
            case 5: rejected=ddlaMalloc(h,&p,128)!=ddlaStatus_t::DDLA_STATUS_SUCCESS; break;
            }
        } catch(const std::bad_alloc&) {rejected=true;}
        allocation_probe::enabled=false;
        const auto count=consumed?allocation_probe::after:allocation_probe::before;
        const auto other=consumed?allocation_probe::before:allocation_probe::after;
        if(p) {
            if(kind==5) ddlaFree(h,p);
            else if(kind==1) ::operator delete[](p);
            else if(kind==2) ::operator delete(p,std::align_val_t(64));
            else if(kind==3) ::operator delete[](p,std::align_val_t(64));
            else if(kind==4) std::free(p);
            else ::operator delete(p);
        }
        if(!external_ok || !external_clean) throw std::runtime_error("allocation probe counted or failed a vendor allocation");
        if(count!=1 || other!=0 || rejected!=fail || allocation_probe::fired!=fail)
            throw std::runtime_error("allocation probe missed managed allocation or failure injection");
    }
}

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
void check_replay(const ddla::DdlaHandle_t& h,int n,int kd,int panel,bool shifted,bool with_phases)
{
    using R=typename ddla::real_type<T>::type;
    int npr=0,npc=0; ddlaGetGridDims(h,npr,npc);
    const int rank=ddlaGetRank(h),P=npr*npc;
    const ddla::detail::RowShare rows(n,P);
    const int nr=rows.count(rank),ldy=std::max(nr,1)+3;
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
    ddlaDescInit(desc,h,n+5,n+7,4,4,shifted?npr-1:0,shifted?npc-1:0);
    desc[ddla::DDLA_LLD_]+=5;
    const int lld=desc[ddla::DDLA_LLD_],zr=ddla_test::m_loc(h,desc),zc=ddla_test::n_loc(h,desc);
    const T sentinel=T(R(139));
    std::vector<T> before(std::max<std::size_t>(local_size(h,desc),1),sentinel);
    for(int c=0;c<zc;++c) for(int r=0;r<zr;++r)
        if(ddla::indx_l2g_r(desc,h,r)<n && ddla::indx_l2g_c(desc,h,c)<n)
            before[r+static_cast<std::size_t>(c)*lld]=T(std::numeric_limits<R>::quiet_NaN());
    std::vector<R> input(static_cast<std::size_t>(ldy)*std::max(n,1),std::numeric_limits<R>::quiet_NaN());
    for(int c=0;c<n;++c) for(int r=0;r<nr;++r) input[r+static_cast<std::size_t>(c)*ldy]=yvalue<R>(rows.lo(rank)+r,c);
    DeviceBuffer<R> Y(h,input.size()); DeviceBuffer<T> Z(h,before.size());
    upload(h,Y.ptr,input); upload(h,Z.ptr,before); check_ddla_sync(h);
    int info=-1;
    ddla::detail::hb2st_replay_rows_cpu<T>(h,n,Y.ptr,ldy,history,phases,Z.ptr,desc,info,panel);
    check_ddla_sync(h);
    require_close(h,"replay consumed input",Y.ptr!=nullptr,0);
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
    // The reference column owner is independent of both matrix layouts. This
    // checks every entry without repeating dense reference work on process rows.
    for(int c=0;c<n;++c) {
        std::vector<std::complex<long double>> ref(n);
        if(rank==c%P) {
            for(int r=0;r<n;++r) ref[r]=wide(yvalue<R>(r,c))*(r>0 && !phases.empty()?wide(phases[r-1]):std::complex<long double>(1,0));
            for(auto it=records.rbegin();it!=records.rend();++it) {
                const int b=it->j0,a=b-1;
                const auto x=ref[a],y=ref[b],s=wide(it->s);
                const long double cs=it->c;
                ref[a]=cs*x-std::conj(s)*y;ref[b]=s*x+cs*y;
            }
        }
        MPI_Bcast(ref.data(),static_cast<int>(ref.size()*sizeof(ref[0])),MPI_BYTE,c%P,ddlaGetCommunicator(h));
        for(int lc=0;lc<zc;++lc) if(ddla::indx_l2g_c(desc,h,lc)==c)
            for(int lr=0;lr<zr;++lr) {
                const int gr=ddla::indx_l2g_r(desc,h,lr);
                if(gr<n) error=std::max(error,static_cast<double>(std::abs(wide(after[lr+static_cast<std::size_t>(lc)*lld])-ref[gr])));
            }
    }
    const std::string tag="CPU replay n="+std::to_string(n)+" type="+std::to_string(sizeof(T))+" panel="+std::to_string(panel)+" shifted="+std::to_string(shifted);
    require_close(h,tag+" finite",bad,0); require_close(h,tag+" padding",padding,0);
    require_close(h,tag+" dense reference",error/(std::max(n,1)*std::numeric_limits<R>::epsilon()),50);
}

std::pair<std::size_t,std::size_t> allocation_case(const ddla::DdlaHandle_t& h,bool after_input,std::size_t fail_at)
{
    constexpr int n=37,kd=4;
    int npr=0,npc=0;ddlaGetGridDims(h,npr,npc);
    const int P=npr*npc,rank=ddlaGetRank(h),victim=P-1;
    const ddla::detail::RowShare rows(n,P);
    const int nr=rows.count(rank),ldy=std::max(nr,1)+3;
    const auto records=make_records<double>(n,kd);
    ddla::detail::DistributedHistory<double> history;
    history.prepare(n,kd,ddlaGetCommunicator(h));
    history.capture(0,history.total,records.data(),history.total);
    int desc[ddla::DDLA_DLEN_];ddlaDescInit(desc,h,n+5,n+7,4,4,npr-1,npc-1);desc[ddla::DDLA_LLD_]+=5;
    std::vector<double> input(static_cast<std::size_t>(ldy)*n,std::numeric_limits<double>::quiet_NaN());
    for(int c=0;c<n;++c) for(int r=0;r<nr;++r) input[r+static_cast<std::size_t>(c)*ldy]=yvalue<double>(rows.lo(rank)+r,c);
    std::vector<double> output(std::max<std::size_t>(local_size(h,desc),1),139),phases;
    DeviceBuffer<double> Y(h,input.size()),Z(h,output.size());upload(h,Y.ptr,input);upload(h,Z.ptr,output);
    double* original=Y.ptr;
    if(rank==victim) allocation_probe::begin(&Y.ptr,after_input,fail_at);
    int caught=0,info=-1;
    try {ddla::detail::hb2st_replay_rows_cpu<double>(h,n,Y.ptr,ldy,history,phases,Z.ptr,desc,info,3);}
    catch(const std::exception&) {caught=1;}
    allocation_probe::enabled=false;
    unsigned long long counts[2]={allocation_probe::before,allocation_probe::after};
    MPI_Bcast(counts,2,MPI_UNSIGNED_LONG_LONG,victim,ddlaGetCommunicator(h));
    int hits=rank==victim && allocation_probe::fired?1:0,all_hits=0,all_caught=0;
    MPI_Allreduce(&hits,&all_hits,1,MPI_INT,MPI_SUM,ddlaGetCommunicator(h));
    MPI_Allreduce(&caught,&all_caught,1,MPI_INT,MPI_SUM,ddlaGetCommunicator(h));
    if(fail_at) {
        require_close(h,"CPU failure injection",std::abs(all_hits-1),0);
        require_close(h,"CPU failure agreement",std::abs(all_caught-P),0);
        require_close(h,"CPU failure input ownership",after_input?(Y.ptr!=nullptr):(Y.ptr!=original),0);
        if(rank==0) std::printf("CPU_REPLAY_FAILURE after=%d index=%zu caught=%d PASS\n",after_input?1:0,fail_at,all_caught);
    } else {
        require_close(h,"CPU allocation probe success",caught,0);
        require_close(h,"CPU allocation probe consumed input",Y.ptr!=nullptr,0);
    }
    return {counts[0],counts[1]};
}

void validation_failure(const ddla::DdlaHandle_t& h)
{
    int npr=0,npc=0;ddlaGetGridDims(h,npr,npc);const int P=npr*npc,rank=ddlaGetRank(h);
    if(P==1) return;
    constexpr int n=3;
    ddla::detail::DistributedHistory<double> history;history.prepare(n,1,ddlaGetCommunicator(h));
    const ddla::detail::RowShare rows(n,P);const int ld=std::max(rows.count(rank),1);
    DeviceBuffer<double> Y(h,static_cast<std::size_t>(ld)*n);
    int desc[9];ddlaDescInit(desc,h,n,n,2,2,0,0);
    DeviceBuffer<double> Z(h,std::max<std::size_t>(local_size(h,desc),1));
    std::vector<double> phases;int info=-1,caught=0;
    try {ddla::detail::hb2st_replay_rows_cpu<double>(h,n,Y.ptr,ld,history,phases,Z.ptr,desc,info,rank==P-1?1:2);}
    catch(const std::exception&) {caught=1;}
    int all=0;MPI_Allreduce(&caught,&all,1,MPI_INT,MPI_SUM,ddlaGetCommunicator(h));
    require_close(h,"inconsistent panel rejected",std::abs(all-P),0);require_close(h,"invalid call retains input",Y.ptr==nullptr,0);
}

template<class T> void small_cases(const ddla::DdlaHandle_t& h)
{
    for(bool shifted:{false,true}) {
        for(int panel:{1,3,64}) check_replay<T>(h,37,4,panel,shifted,true);
        check_replay<T>(h,3,2,1,shifted,false);
    }
    check_replay<T>(h,0,1,1,true,false);
    check_replay<T>(h,1,1,1,true,false);
    check_replay<T>(h,2,1,1,false,true);
}
} // namespace
int main(int argc,char** argv)
{
    int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int rank=0,size=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    try {
        if(provided<MPI_THREAD_FUNNELED) throw std::runtime_error("MPI thread support");
        setenv("DDLA_EIG_THREADS","1",1);
        TestOptions options;std::string error;
        if(!parse_options(argc,argv,size,options,error)) throw std::runtime_error(error);
        for(const auto& grid:options.grids) {
            ddla::DdlaHandle_t h=nullptr;
            if(ddlaInit(h,DdlaBackend::CPU)!=ddlaStatus_t::DDLA_STATUS_SUCCESS ||
               ddlaSet(h,MPI_COMM_WORLD,grid.first,grid.second)!=ddlaStatus_t::DDLA_STATUS_SUCCESS)
                throw std::runtime_error("CPU handle/grid");
            check_allocation_probe(h);
            small_cases<float>(h);small_cases<double>(h);
            small_cases<std::complex<float>>(h);small_cases<std::complex<double>>(h);
            check_replay<double>(h,1025,8,64,true,true);
            const auto counts=allocation_case(h,false,0);
            if(counts.first==0)throw std::runtime_error("allocation probe missed replay preparation");
            if(counts.second!=0)throw std::runtime_error("replay allocated managed scratch after input consumption");
            allocation_case(h,false,1);if(counts.first>1)allocation_case(h,false,counts.first);
            // Raw apply has no managed allocation to fault after consumption.
            // A distinct post-consumption runtime-fault trigger is not supplied
            // by this test; do not manufacture an allocation for that purpose.
            validation_failure(h);
            ddlaDestroy(h);
        }
        if(rank==0) std::puts("test_pheevd_cpu_replay passed");
    } catch(const std::exception& e) {
        allocation_probe::enabled=false;std::fprintf(stderr,"rank %d: %s\n",rank,e.what());MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Finalize();return 0;
}
