#include "api_grid_test_common.h"
#include "ddla_stream.h"
#include "Eigensolver/pheevd_alias.h"
#include "Eigensolver/two_stage/hb2st_replay_storage.h"
#include "row_share.h"
#include "workspace_test_common.h"
#include <array>
#include <cstring>
#include <numeric>
#include <type_traits>

using namespace api_grid_test;
using namespace ddla::detail;
static_assert(std::is_trivially_copyable_v<PheevdReflectorBackupPlan>);
static_assert(std::is_trivially_copyable_v<Hb2stReplayStoragePlan>);
namespace {
bool no_growth(const workspace_test::Snapshot& s)
{return !s.ddla_alloc&&!s.ddla_free&&!s.pinned_alloc&&!s.raw_gpu_alloc&&!s.host_new&&!s.vendor_query&&!s.resource_setup;}
bool pure(const workspace_test::Snapshot& s)
{return no_growth(s)&&!s.copies&&!s.launches&&!s.syncs&&!s.collectives;}
void clean(DdlaHandle_t h,const workspace_test::Snapshot& s,bool query=false)
{require_close(h,query?"alias/storage pure query":"alias/storage raw allocation and resource counters",(query?pure(s):no_growth(s))?0:1,0);}
// Exactly the sized backend and host bytes between poisoned guards.
class Guarded {
    static constexpr std::size_t guard=256;
    DdlaHandle_t h_;std::size_t backend_bytes_,host_bytes_;ddla::vector<char> backend_;std::vector<char> host_;
public:
    Guarded(DdlaHandle_t h,std::size_t backend_bytes,std::size_t host_bytes):h_(h),backend_bytes_(backend_bytes),host_bytes_(host_bytes),
        backend_(h,backend_bytes+2*guard),host_(host_bytes+2*guard,char(0x69))
    {DDLA_CHECK(ddlaMemset(h,backend_.data(),0x69,backend_.size()));DDLA_CHECK(ddlaSynchronize(h));}
    void* backend(){return backend_.data()+guard;}
    char* host(){return host_.data()+guard;}
    void verify(){std::array<char,2*guard> bytes{};DDLA_CHECK(ddlaMemcpy(h_,bytes.data(),backend_.data(),guard));DDLA_CHECK(ddlaMemcpy(h_,bytes.data()+guard,backend_.data()+guard+backend_bytes_,guard));DDLA_CHECK(ddlaSynchronize(h_));
        bool okay=true;for(std::size_t i=0;i<guard;++i)okay=okay&&host_[i]==char(0x69)&&host_[guard+host_bytes_+i]==char(0x69)&&bytes[i]==char(0x69)&&bytes[guard+i]==char(0x69);require_close(h_,"alias/storage exact workspace guards",okay?0:1,0);}
};
template<class T>T scalar(double r,double i=0){if constexpr(std::is_arithmetic_v<T>)return T(r);else return T(r,i);}
template<class T>bool bits_equal(const std::vector<T>& a,const std::vector<T>& b)
{return a.size()==b.size()&&(a.empty()||!std::memcmp(a.data(),b.data(),a.size()*sizeof(T)));}

template<class T>void alias_case(DdlaHandle_t h,int n,char triangle)
{
    int desc[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(desc,h,n+5,n+7,3,4,h->nprows_-1,h->npcols_-1));desc[DDLA_LLD_]+=3;
    auto initial=make_local<T>(h,desc,[](int i,int j){return scalar<T>(101+i+100*j,.25+i+j);});initial.push_back(scalar<T>(-91));
    DeviceBuffer<T>A(h,initial.size()),Z(h,initial.size());upload(h,A.ptr,initial);upload(h,Z.ptr,initial);check_ddla_sync(h);
    // A single overlapping rank requires saved reflectors on every rank.
    const bool overlap=h->myprow_==h->nprows_-1&&h->mypcol_==h->npcols_-1;
    const auto p=pheevd_reflector_backup_check<T>(h,n,A.ptr,desc,overlap?A.ptr+(n>1?1:0):Z.ptr,desc,triangle);
    require_close(h,"reflector overlap globally frozen",p.needed?0:1,0);
    MPI_Barrier(h->comm);bool local=true;if(!h->myid){workspace_test::Scope trace(true);std::size_t bytes=0;pheevd_reflector_backup_workspace_size<T>(p,&bytes);local=pure(trace.finish());}MPI_Barrier(h->comm);require_close(h,"one-rank alias query completes",local?0:1,0);
    workspace_test::Scope trace(true);std::size_t bytes=0;pheevd_reflector_backup_workspace_size<T>(p,&bytes);clean(h,trace.finish(),true);Guarded arena(h,bytes,0);
    // The saved reflectors are at the front of the scratch (ld p.saved[DDLA_LLD_]).
    const std::size_t count=std::max<std::size_t>(1,std::size_t(p.saved[DDLA_LLD_])*std::size_t(p.columns));
    for(int pass=0;pass<2;++pass){
        DDLA_CHECK(ddlaMemset(h,arena.backend(),0x69,bytes));check_ddla_sync(h);
        workspace_test::Scope run;pheevd_reflector_backup_save_run<T>(h,p,A.ptr,arena.backend(),bytes);clean(h,run.finish());
        const auto saved=download(h,static_cast<const T*>(arena.backend()),count);bool okay=true;
        for(int c=0;c<p.columns;++c)for(int r=0;r<p.rows;++r){const int gr=indxl2g(r,desc[DDLA_MB_],p.pr,desc[DDLA_RSRC_],p.npr),gc=indxl2g(c,desc[DDLA_NB_],p.pc,desc[DDLA_CSRC_],p.npc);
            const bool selected=triangle=='A'||(triangle=='U'?gr<gc:gr>gc);const auto& got=saved[r+std::size_t(c)*p.saved[DDLA_LLD_]];const auto& original=initial[r+std::size_t(c)*desc[DDLA_LLD_]];
            if(selected)okay=okay&&std::memcmp(&got,&original,sizeof(T))==0;
            else {T untouched;std::memset(&untouched,0x69,sizeof(T));okay=okay&&std::memcmp(&got,&untouched,sizeof(T))==0;}
        }
        require_close(h,"reflector copy preserves full/strict triangle policy",okay?0:1,0);require_close(h,"reflector backup preserves input and padding",bits_equal(initial,download(h,A.ptr,initial.size()))?0:1,0);arena.verify();
    }
    const auto disjoint=pheevd_reflector_backup_check<T>(h,n,A.ptr,desc,Z.ptr,desc,triangle);require_close(h,"disjoint reflectors need no backup",disjoint.needed?1:0,0);
}

std::vector<int> reference_prefix(const Hb2stReplayStoragePlan& p)
{
    std::vector<std::size_t> capacity(p.P);std::vector<int> order(p.P),count(p.P),prefix(p.P+1);std::iota(order.begin(),order.end(),0);
    for(int rank=0;rank<p.P;++rank){const int row=p.major=='R'?rank/p.npc:rank%p.npr,col=p.major=='R'?rank%p.npc:rank/p.npr;capacity[rank]=std::size_t(num_loc(p.n,p.mb,row,p.rsrc,p.npr))*num_loc(p.n,p.nb,col,p.csrc,p.npc);count[rank]=int(capacity[rank]/p.n);}
    std::sort(order.begin(),order.end(),[&](int a,int b){const auto ar=capacity[a]%p.n,br=capacity[b]%p.n;return ar==br?a<b:ar>br;});
    const int extra=p.n-std::accumulate(count.begin(),count.end(),0);for(int i=0;i<extra;++i)++count[order[i]];for(int rank=0;rank<p.P;++rank)prefix[rank+1]=prefix[rank]+count[rank];return prefix;
}
void partition_cases(DdlaHandle_t h)
{
    // Exercise the trusted geometry arithmetic independently of GPU eligibility.
    for(int n:{1,5,17,37})for(int block:{1,3,8}){
        Hb2stReplayStoragePlan p;p.enabled=true;p.n=n;p.P=h->nprocs;p.npr=h->nprows_;p.npc=h->npcols_;p.major=h->major;p.mb=block;p.nb=block+1;p.rsrc=p.npr-1;p.csrc=p.npc-1;
        int assigned=0;for(int rank=0;rank<p.P;++rank){const int row=p.major=='R'?rank/p.npc:rank%p.npr,col=p.major=='R'?rank%p.npc:rank/p.npr;assigned+=num_loc(n,p.mb,row,p.rsrc,p.npr)*num_loc(n,p.nb,col,p.csrc,p.npc)/n;}p.spill=n-assigned;const auto expected=reference_prefix(p);
        workspace_test::Scope trace(true);bool okay=true;for(int rank=0;rank<=p.P;++rank)okay=okay&&hb2st_replay_storage_prefix(p,rank)==expected[rank];const auto snapshot=trace.finish();clean(h,snapshot,true);require_close(h,"compact partition keeps exact descending remainder/rank tie",okay?0:1,0);
    }
}
template<class T>void storage_case(DdlaHandle_t h,int n,int mb,int nb)
{
    int desc[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(desc,h,n,n,mb,nb,h->nprows_-1,h->npcols_-1));const auto count=std::max<std::size_t>(local_size(h,desc),1);
    std::vector<T> initial(count);for(std::size_t i=0;i<count;++i)initial[i]=scalar<T>(.25+i,.125+.5*i);
    DeviceBuffer<T>A(h,count),Z(h,count);upload(h,A.ptr,initial);upload(h,Z.ptr,initial);check_ddla_sync(h);
    const auto p=hb2st_replay_storage_check<T>(h,n,A.ptr,desc,Z.ptr,desc);int eligible=h->backend==DdlaBackend::GPU&&h->nprocs>1&&ddla_test::m_loc(h,desc)>0&&ddla_test::n_loc(h,desc)>0;MPI_Allreduce(MPI_IN_PLACE,&eligible,1,MPI_INT,MPI_MIN,h->comm);
    require_close(h,"compact replay storage eligibility",p.enabled==bool(eligible)?0:1,0);
    MPI_Barrier(h->comm);bool local=true;if(!h->myid){workspace_test::Scope trace(true);std::size_t bytes=0;hb2st_replay_storage_workspace_size<T>(h,p,&bytes);local=pure(trace.finish());}MPI_Barrier(h->comm);require_close(h,"one-rank replay storage query completes",local?0:1,0);
    workspace_test::Scope trace(true);std::size_t host_bytes=0;hb2st_replay_storage_workspace_size<T>(h,p,&host_bytes);clean(h,trace.finish(),true);if(!p.enabled)return;
    const auto expected=reference_prefix(p);require_close(h,"compact local partition metadata",p.first==expected[p.rank]&&p.count==expected[p.rank+1]-expected[p.rank]?0:1,0);
    const bool fits=p.real_capacity>=p.y_count&&RowShare(n,h->nprocs).count(h->myid)>0;require_close(h,"replay Y borrowing follows real capacity",p.borrow_y==fits?0:1,0);
    // The host scratch is [the column prefix (P + 1 ints) | the backup of A], carved in that order.
    Guarded arena(h,0,host_bytes);const auto before=download(h,A.ptr,count);const auto Zbefore=download(h,Z.ptr,count);
    const int* const prefix_at=reinterpret_cast<const int*>(arena.host());char* const backup_at=arena.host()+padded_bytes<int>(std::size_t(p.P)+1);
    for(int pass=0;pass<2;++pass){
        workspace_test::Scope save;hb2st_replay_storage_save_run<T>(h,p,A.ptr,arena.host(),host_bytes);clean(h,save.finish());bool prefix=true;for(int rank=0;rank<=p.P;++rank)prefix=prefix&&prefix_at[rank]==expected[rank];require_close(h,"replay prefix matches historical partition",prefix?0:1,0);
        require_close(h,"replay backup captures full compact A",std::memcmp(backup_at,before.data(),p.cells*sizeof(T))==0?0:1,0);
        DDLA_CHECK(ddlaMemset(h,A.ptr,0,p.cells*sizeof(T)));check_ddla_sync(h);workspace_test::Scope restore;hb2st_replay_storage_restore_run<T>(h,p,A.ptr,arena.host(),host_bytes);clean(h,restore.finish());
        // Immediately retire the backup bytes: restore_run must finish its
        // host-source read before returning.
        std::memset(backup_at,0xa7,p.cells*sizeof(T));require_close(h,"replay restores A exactly",bits_equal(before,download(h,A.ptr,count))?0:1,0);require_close(h,"save/restore preserve Z",bits_equal(Zbefore,download(h,Z.ptr,count))?0:1,0);arena.verify();
    }
    for(int mode=0;mode<6;++mode){int da[DDLA_DLEN_],dz[DDLA_DLEN_];std::copy(desc,desc+DDLA_DLEN_,da);std::copy(desc,desc+DDLA_DLEN_,dz);
        if(mode==0){da[DDLA_LLD_]+=2;dz[DDLA_LLD_]+=2;}if(mode==1){da[DDLA_M_]+=1;dz[DDLA_M_]+=1;}if(mode==2)dz[DDLA_NB_]+=1;
        if(mode==5&&h->myid==0){da[DDLA_LLD_]+=1;dz[DDLA_LLD_]+=1;}
        const auto rejected=hb2st_replay_storage_check<T>(h,n,A.ptr,da,mode==3?A.ptr+1:mode==4?A.ptr:Z.ptr,dz);require_close(h,"padded/parent/different/overlap replay fallback",rejected.enabled?1:0,0);
    }
}
template<DdlaBackend B>void run(MPI_Comm comm,int pr,int pc,char major)
{
    DdlaHandle_t h=nullptr;ddlaInit(h,B);ddlaSet(h,comm,pr,pc,major);partition_cases(h);
#define CASE(T) for(char triangle:{'A','U','L'}){alias_case<T>(h,19,triangle);alias_case<T>(h,1,triangle);}storage_case<T>(h,37,3,4);storage_case<T>(h,1,1,1)
    CASE(float);CASE(double);CASE(std::complex<float>);CASE(std::complex<double>);
#undef CASE
    if(!h->myid)std::cout<<"workspace alias/storage PASS backend="<<int(B)<<" grid="<<pr<<"x"<<pc<<" major="<<major<<std::endl;ddlaDestroy(h);
}
}
int main(int argc,char** argv)
{
    MPI_Init(&argc,&argv);int size;MPI_Comm_size(MPI_COMM_WORLD,&size);TestOptions options;std::string error;if(!parse_options(argc,argv,size,options,error)){std::cerr<<error<<std::endl;MPI_Abort(MPI_COMM_WORLD,2);}
    for(auto grid:options.grids)for(char major:{'R','C'}){
        run<DdlaBackend::CPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#if DDLA_HAS_GPU
        run<DdlaBackend::GPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#endif
    }MPI_Finalize();return 0;
}
