#include "helpers.h"
#include "hb2st_replay_storage.h"
#include "hb2st_replay_gpu.h"
#include "ddla_connector.h"
#include "ddla_desc.h"
#include "ddla_stream.h"
#include "row_share.h"
#include "comm_traits.h"
#include "plan_util.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ddla { namespace detail {
namespace {
void storage_agree(const DdlaHandle_t& h,const std::exception_ptr& failure)
{
    int failed=failure?1:0;
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,&failed,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h)));
    if(failed) {
        if(failure) std::rethrow_exception(failure);
        throw std::runtime_error("hb2st reuse: preparation failed on another rank");
    }
}
}

namespace {
std::size_t storage_capacity(const Hb2stReplayStoragePlan& p,int rank)
{
    const int row=p.major=='R'?rank/p.npc:rank%p.npr,col=p.major=='R'?rank%p.npc:rank/p.npr;
    return std::size_t(num_loc(p.n,p.mb,row,p.rsrc,p.npr))*std::size_t(num_loc(p.n,p.nb,col,p.csrc,p.npc));
}
int storage_columns(const Hb2stReplayStoragePlan& p,int rank)
{
    const auto capacity=storage_capacity(p,rank),remainder=capacity%p.n;int better=0;
    for(int other=0;other<p.P;++other){const auto r=storage_capacity(p,other)%p.n;if(r>remainder||(r==remainder&&other<rank))++better;}
    return int(capacity/p.n)+(better<p.spill?1:0);
}
}
int hb2st_replay_storage_prefix(const Hb2stReplayStoragePlan& p,int boundary)
{
    if(!p.enabled)return 0;
    int first=0;for(int rank=0;rank<boundary;++rank)first+=storage_columns(p,rank);return first;
}
template<class T>Hb2stReplayStoragePlan hb2st_replay_storage_check(const DdlaHandle_t& h,int n,T* A,const int* da,T* Z,const int* dz)
{
    using R=typename real_type<T>::type;if(!h)throw std::invalid_argument("hb2st reuse: null handle");
    Hb2stReplayStoragePlan p;p.n=n;p.P=h->nprocs;p.rank=h->myid;p.npr=h->nprows_;p.npc=h->npcols_;p.major=h->major;
    int eligible=h->backend==DdlaBackend::GPU&&p.P>1&&n>0&&A&&Z&&da&&dz;
    if(eligible){for(int k=0;k<DDLA_DLEN_;++k)eligible&=da[k]==dz[k];eligible&=da[DDLA_M_]==n&&da[DDLA_N_]==n&&da[DDLA_MB_]>0&&da[DDLA_NB_]>0;
        if(eligible){const int rows=num_loc(n,da[DDLA_MB_],h->myprow_,da[DDLA_RSRC_],p.npr),cols=num_loc(n,da[DDLA_NB_],h->mypcol_,da[DDLA_CSRC_],p.npc);
            eligible=rows>0&&cols>0&&da[DDLA_LLD_]==rows;const auto limit=std::numeric_limits<std::size_t>::max()/sizeof(T);
            if(eligible)eligible=std::size_t(rows)<=limit/std::size_t(cols);
            if(eligible){p.cells=std::size_t(rows)*cols;const auto bytes=p.cells*sizeof(T);eligible=!memory_overlaps(A,bytes,Z,bytes);}
        }
    }
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,&eligible,1,MPI_INT,MPI_MIN,h->comm));p.enabled=eligible!=0;if(!p.enabled)return p;
    std::exception_ptr failure;
    try{
        p.mb=da[DDLA_MB_];p.nb=da[DDLA_NB_];p.rsrc=da[DDLA_RSRC_];p.csrc=da[DDLA_CSRC_];
        int assigned=0;for(int rank=0;rank<p.P;++rank)assigned+=int(storage_capacity(p,rank)/n);p.spill=n-assigned;
        if(p.spill<0||p.spill>=p.P)throw std::logic_error("hb2st reuse: invalid compact column capacities");
        p.first=hb2st_replay_storage_prefix(p,p.rank);p.count=storage_columns(p,p.rank);
        p.real_capacity=p.cells*(sizeof(T)/sizeof(R));const RowShare rows(n,p.P);p.ldy=std::max(rows.count(p.rank),1);p.y_count=std::size_t(p.ldy)*std::size_t(n);
        p.borrow_y=rows.count(p.rank)>0&&p.real_capacity>=p.y_count;
        if(hb2st_replay_storage_prefix(p,p.P)!=n)throw std::logic_error("hb2st reuse: incomplete column partition");
    }catch(...){failure=std::current_exception();}storage_agree(h,failure);return p;
}
template<class T>void hb2st_replay_storage_workspace_size(const DdlaHandle_t&,const Hb2stReplayStoragePlan& p,std::size_t* host_bytes)
{
    *host_bytes=p.enabled?padded_bytes<int>(std::size_t(p.P)+1)+padded_bytes<T>(p.cells):0;
}
template<class T>void hb2st_replay_storage_save_run(const DdlaHandle_t& h,const Hb2stReplayStoragePlan& p,const T* A,
    void* host_work,std::size_t host_bytes)
{
    try {
        if(!p.enabled)return;
        // carved in the order hb2st_replay_storage_workspace_size counts it
        char* w=static_cast<char*>(host_work);int* columns=take<int>(w,std::size_t(p.P)+1);T* backup=take<T>(w,p.cells);
        assert(w-static_cast<char*>(host_work)<=static_cast<std::ptrdiff_t>(host_bytes));(void)host_bytes;
        int first=0;for(int rank=0;rank<p.P;++rank){columns[rank]=first;first+=storage_columns(p,rank);}columns[p.P]=first;
        std::exception_ptr failure;try{DDLA_CHECK(ddlaMemcpy(h,backup,A,p.cells*sizeof(T)));DDLA_CHECK(ddlaSynchronize(h));}catch(...){failure=std::current_exception();}storage_agree(h,failure);
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(h) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(h); } catch (...) {}
        throw;
    }
}
template<class T>void hb2st_replay_storage_restore_run(const DdlaHandle_t& h,const Hb2stReplayStoragePlan& p,T* A,
    void* host_work,std::size_t host_bytes)
{
    try {
        if(!p.enabled)return;
        char* w=static_cast<char*>(host_work);(void)take<int>(w,std::size_t(p.P)+1);const T* backup=take<T>(w,p.cells);
        assert(w-static_cast<char*>(host_work)<=static_cast<std::ptrdiff_t>(host_bytes));(void)host_bytes;
        std::exception_ptr failure;try{DDLA_CHECK(ddlaMemcpy(h,A,backup,p.cells*sizeof(T)));DDLA_CHECK(ddlaSynchronize(h));}catch(...){failure=std::current_exception();}storage_agree(h,failure);
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(h) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(h); } catch (...) {}
        throw;
    }
}
#define STORAGE_PHASE(T) \
 template Hb2stReplayStoragePlan hb2st_replay_storage_check<T>(const DdlaHandle_t&,int,T*,const int*,T*,const int*); \
 template void hb2st_replay_storage_workspace_size<T>(const DdlaHandle_t&,const Hb2stReplayStoragePlan&,std::size_t*); \
 template void hb2st_replay_storage_save_run<T>(const DdlaHandle_t&,const Hb2stReplayStoragePlan&,const T*,void*,std::size_t); \
 template void hb2st_replay_storage_restore_run<T>(const DdlaHandle_t&,const Hb2stReplayStoragePlan&,T*,void*,std::size_t)
STORAGE_PHASE(float);STORAGE_PHASE(double);STORAGE_PHASE(std::complex<float>);STORAGE_PHASE(std::complex<double>);
#undef STORAGE_PHASE

// Owning convenience wrapper: check, size, allocate and run in one object.
template<class T>
Hb2stReplayStorage<T>::Hb2stReplayStorage(const DdlaHandle_t& h,int n,T* A,const int* da,T* Z,const int* dz)
    :h_(h),n_(n),A_(A),Z_(Z),descA_(da),descZ_(dz)
{
    plan_=hb2st_replay_storage_check<T>(h,n,A,da,Z,dz);enabled_=plan_.enabled;P_=plan_.P;rank_=plan_.rank;cells_=plan_.cells;ldy_=plan_.ldy;
}

template<class T>
Hb2stReplayStorage<T>::~Hb2stReplayStorage() noexcept
{
    if(owns_Y_ && Y_) (void)ddlaFree(h_,Y_);
}

template<class T>
void Hb2stReplayStorage<T>::prepare()
{
    if(!enabled_||prepared_)return;std::exception_ptr failure;
    try{
        std::size_t host_bytes=0;hb2st_replay_storage_workspace_size<T>(h_,plan_,&host_bytes);
        host_work_.resize(host_bytes);columns_.resize(std::size_t(P_)+1);
        owns_Y_=!plan_.borrow_y;
        if(owns_Y_)DDLA_CHECK(ddlaMalloc(h_,reinterpret_cast<void**>(&Y_),plan_.y_count*sizeof(R)));else Y_=reinterpret_cast<R*>(Z_);
    }catch(...){failure=std::current_exception();}storage_agree(h_,failure);
    hb2st_replay_storage_save_run<T>(h_,plan_,A_,host_work_.data(),host_work_.size());
    // the prefix save_run wrote at the front of the host bytes
    std::copy_n(reinterpret_cast<const int*>(host_work_.data()),columns_.size(),columns_.begin());prepared_=true;
}

template<class T>
void Hb2stReplayStorage<T>::replay(const DistributedHistory<T>& history,const std::vector<T>& phases,int& info,
                                 int panel_columns,int replay_records,bool use_waves)
{
    std::exception_ptr failure;
    try { if(!prepared_ || restored_) throw std::logic_error("hb2st reuse: replay before prepare or after restore"); }
    catch(...) { failure=std::current_exception(); }
    storage_agree(h_,failure);
#if DDLA_HAS_GPU
    Hb2stReplayGpuWorkspace<T> workspace;
    workspace.main=A_; workspace.main_elements=cells_; workspace.column_offsets=&columns_; workspace.owns_input=owns_Y_;
    hb2st_replay_rows_gpu<T>(h_,n_,Y_,ldy_,history,phases,Z_,descZ_,info,panel_columns,replay_records,use_waves,&workspace);
#else
    (void)history; (void)phases; (void)info; (void)panel_columns; (void)replay_records; (void)use_waves;
    throw std::runtime_error("hb2st reuse: built without GPU support");
#endif
}

template<class T>
void Hb2stReplayStorage<T>::restore()
{
    if(!enabled_||restored_)return;std::exception_ptr failure;
    try{if(!prepared_||Y_)throw std::logic_error("hb2st reuse: restore before input consumption");}catch(...){failure=std::current_exception();}storage_agree(h_,failure);
    hb2st_replay_storage_restore_run<T>(h_,plan_,A_,host_work_.data(),host_work_.size());
    std::vector<char>().swap(host_work_);restored_=true;
}

template class Hb2stReplayStorage<float>;
template class Hb2stReplayStorage<double>;
template class Hb2stReplayStorage<std::complex<float>>;
template class Hb2stReplayStorage<std::complex<double>>;
} }
