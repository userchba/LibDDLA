#include "BLAS/blas_impl.h"
#include "punmtr_impl.h"
#include "ddla_stream.h"
#include "hb2st_householder_cpu.h"
#if DDLA_HAS_GPU
#include "hb2st_householder_gpu.h"
#endif
#include "hb2st_householder_kernels.h"
#include "hb2st_householder_paired.h"
#include "hb2st_householder_grouped_real.h"
#include "hb2st_householder_grouped_real_fast.h"
#include "Eigensolver/eig_cpu_kernels.h"
#include "ddla_connector.h"
#include "ddla_desc.h"
#include "mpi_datatype.h"
#include "comm_traits.h"
#include "plan_util.h"
#include "dc_threads.h"
#include "instantiate.h"
#include "check_agree.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

namespace ddla { namespace detail {
namespace {
using Index = hh::index_t;
constexpr int factor_width = 32;
constexpr int factor_batch = 32;
constexpr int paired_overlaps = factor_width/2;
constexpr std::size_t shared_alignment = 64;

std::size_t checked_bytes(Index count, std::size_t item) {
    std::size_t result=0;
    if(!hh::bytes(count,item,result))throw std::overflow_error("CPU Householder byte count overflow");
    return result;
}
Index checked_add(Index a,Index b) {
    Index result=0;if(!hh::add(a,b,result))throw std::overflow_error("CPU Householder offset overflow");return result;
}
Index checked_multiply(Index a,Index b) {
    Index result=0;if(!hh::multiply(a,b,result))throw std::overflow_error("CPU Householder size overflow");return result;
}
std::size_t align_shared(std::size_t bytes) {
    if(bytes>std::numeric_limits<std::size_t>::max()-(shared_alignment-1))
        throw std::overflow_error("CPU Householder alignment overflow");
    return (bytes+shared_alignment-1)/shared_alignment*shared_alignment;
}

// Numeric preparation can fail on one rank. All callers reach the agreement
// (detail::agree_local) before any following window/communication phase; only
// application threads execute MPI. Arbitrary MPI failures follow MPI_CHECK's
// abort contract.

struct SharedWindow {
    MPI_Win window=MPI_WIN_NULL;
    bool locked=false;
    void* local=nullptr;
    std::vector<void*> bases;
    std::vector<MPI_Aint> sizes;
    std::vector<Index> payloads,offsets,min_offsets,max_offsets;

    void metadata(int ranks) {
        const auto count=static_cast<std::size_t>(ranks);
        bases.resize(count);sizes.resize(count);payloads.resize(count);
        offsets.resize(count);min_offsets.resize(count);max_offsets.resize(count);
    }
    void allocate(MPI_Comm node,MPI_Comm world,std::size_t bytes,bool reject_model) {
        // MPI guarantees suitable storage for ordinary types, not a cache-line
        // aligned base. Reserve an interior alignment allowance independently
        // in every nonempty owner segment; zero-byte segments remain zero.
        std::size_t allocated=0;
        detail::agree_local(world,"CPU Householder preparation",[&] {
            if(bytes&&bytes>std::numeric_limits<std::size_t>::max()-(shared_alignment-1))
                throw std::overflow_error("CPU Householder alignment allowance overflow");
            allocated=bytes?bytes+(shared_alignment-1):0;
            if(allocated>static_cast<std::size_t>(std::numeric_limits<MPI_Aint>::max()))
                throw std::overflow_error("CPU Householder shared allocation exceeds MPI_Aint");
        });
        const Index payload=static_cast<Index>(bytes);
        MPI_CHECK(MPI_Allgather(&payload,1,MPI_UINT64_T,payloads.data(),1,MPI_UINT64_T,world));
        MPI_CHECK(MPI_Win_allocate_shared(static_cast<MPI_Aint>(allocated),1,MPI_INFO_NULL,node,&local,&window));
        MPI_CHECK(MPI_Win_set_errhandler(window,MPI_ERRORS_RETURN));
        detail::agree_local(world,"CPU Householder preparation",[&] {
            int flag=0;int* model=nullptr;
            MPI_CHECK(MPI_Win_get_attr(window,MPI_WIN_MODEL,&model,&flag));
            if(reject_model||!flag||!model||*model!=MPI_WIN_UNIFIED)
                throw std::runtime_error("CPU Householder requires MPI_WIN_UNIFIED");
            for(std::size_t q=0;q<bases.size();++q) {
                int displacement=0;void* raw=nullptr;MPI_Aint raw_size=0;
                MPI_CHECK(MPI_Win_shared_query(window,static_cast<int>(q),&raw_size,&displacement,&raw));
                if(displacement!=1||raw_size<0)
                    throw std::runtime_error("CPU Householder shared segment is invalid");
                offsets[q]=0;bases[q]=nullptr;sizes[q]=0;
                if(payloads[q]==0)continue;
                if(!raw)throw std::runtime_error("CPU Householder shared segment is missing");
                const auto address=reinterpret_cast<std::uintptr_t>(raw);
                const Index offset=(shared_alignment-address%shared_alignment)%shared_alignment;
                if(offset>static_cast<Index>(raw_size)||payloads[q]>static_cast<Index>(raw_size)-offset)
                    throw std::runtime_error("CPU Householder aligned shared payload is too short");
                offsets[q]=offset;
                bases[q]=static_cast<unsigned char*>(raw)+static_cast<std::size_t>(offset);
                sizes[q]=static_cast<MPI_Aint>(payloads[q]);
                if(reinterpret_cast<std::uintptr_t>(bases[q])%shared_alignment!=0)
                    throw std::runtime_error("CPU Householder interior alignment failed");
            }
        });
        // Different virtual mappings must not select different bytes of the
        // same physical owner segment. Check the offsets before constructing
        // any shared object or opening its passive access epoch.
        MPI_CHECK(MPI_Allreduce(offsets.data(),min_offsets.data(),static_cast<int>(offsets.size()),MPI_UINT64_T,MPI_MIN,world));
        MPI_CHECK(MPI_Allreduce(offsets.data(),max_offsets.data(),static_cast<int>(offsets.size()),MPI_UINT64_T,MPI_MAX,world));
        detail::agree_local(world,"CPU Householder preparation",[&] {
            for(std::size_t q=0;q<offsets.size();++q)
                if(min_offsets[q]!=max_offsets[q])
                    throw std::runtime_error("CPU Householder shared alignment offsets differ across ranks");
        });
        MPI_CHECK(MPI_Win_lock_all(0,window));locked=true;
    }
    void sync() {if(locked)MPI_CHECK(MPI_Win_sync(window));}
    void close() {
        if(window!=MPI_WIN_NULL) {
            if(locked){MPI_CHECK(MPI_Win_sync(window));MPI_CHECK(MPI_Win_unlock_all(window));locked=false;}
            MPI_CHECK(MPI_Win_free(&window));
        }
        local=nullptr;bases.clear();sizes.clear();payloads.clear();offsets.clear();min_offsets.clear();max_offsets.clear();
    }
};

struct alignas(shared_alignment) ChaseControl {
    std::atomic<std::uint64_t> arrived{0};
    std::atomic<std::uint64_t> epoch{0};
    std::atomic<int> abort{0};
};
static_assert(std::atomic<std::uint64_t>::is_always_lock_free,"shared chase counters must be lock free");
static_assert(std::atomic<int>::is_always_lock_free,"shared chase abort must be lock free");

void chase_barrier(ChaseControl* control,int ranks) noexcept {
    const std::uint64_t epoch=control->epoch.load(std::memory_order_relaxed);
    if(control->arrived.fetch_add(1,std::memory_order_acq_rel)==static_cast<std::uint64_t>(ranks-1)) {
        control->arrived.store(0,std::memory_order_relaxed);
        control->epoch.fetch_add(1,std::memory_order_release);
    } else {
        unsigned spins=0;
        while(control->epoch.load(std::memory_order_acquire)==epoch)
            if((++spins&255u)==0)std::this_thread::yield();
    }
}
}

Hb2stHouseholderCpuPlan hh_cpu_geometry(const DdlaHandle_t& h,int n,int band,bool keep,const Hb2stHouseholderCpuOptions& options,int threads)
{
    Hb2stHouseholderCpuPlan p;p.n=n;p.band=std::max(1,std::min(band,std::max(n-1,1)));p.rank=h->myid;p.ranks=h->nprocs;p.threads=threads;p.keep_history=keep;p.options=options;
    p.chase_threads=std::max(1,std::min(threads,8/std::max(1,p.ranks)));
    return p;
}
Hb2stHouseholderCpuPlan hh_cpu_check(const DdlaHandle_t& h,int n,int band,bool keep,const Hb2stHouseholderCpuOptions& options)
{
    if(!h)throw std::invalid_argument("CPU Householder: null handle");
    const auto world=h->comm;
    detail::agree_local(world,"CPU Householder preparation",[&]{
        using O=Hb2stHouseholderCpuOptions;
        if(h->backend!=DdlaBackend::CPU||n<0||band<1||h->nprocs<1||
            (options.stage2!=O::Stage2::Scalar&&options.stage2!=O::Stage2::Blas)||int(options.stage4)<0||int(options.stage4)>int(O::Stage4::GroupedReal6))
            throw std::invalid_argument("CPU Householder requires a CPU handle and valid dimensions");
        if(options.failure==O::Failure::Metadata&&(options.failure_rank<0||options.failure_rank==h->myid))throw std::bad_alloc();
    });
    const int input[5]={n,band,int(keep),int(options.stage2),int(options.stage4)};int lower[5]{},upper[5]{};
    MPI_CHECK(MPI_Allreduce(input,lower,5,MPI_INT,MPI_MIN,world));MPI_CHECK(MPI_Allreduce(input,upper,5,MPI_INT,MPI_MAX,world));
    detail::agree_local(world,"CPU Householder preparation",[&]{for(int i=0;i<5;++i)if(lower[i]!=upper[i])throw std::invalid_argument("CPU Householder dimensions differ across ranks");});
    return hh_cpu_geometry(h,n,band,keep,options,h->host_threads);
}
template<class T> void hh_cpu_reduce_workspace_size(const DdlaHandle_t&,int band,int chase_threads,std::size_t* backend_bytes)
{*backend_bytes=3*padded_bytes<T>(std::size_t(std::max(band,1))*std::max(chase_threads,1));}
template<class T> void hh_cpu_apply_columns_workspace_size(const DdlaHandle_t&,int columns,std::size_t* backend_bytes)
{*backend_bytes=padded_bytes<T>(std::size_t(factor_width)*std::size_t(std::max(columns,0)));}

// Unified CPU BLAS on zero-copy dense views of the packed lower band.
// A wrapper exception becomes a failed event; the caller publishes abort and
// drains every common level before any collective exception can escape.
template<class T>
bool hh_event_blas_cpu(const DdlaHandle_t& handle,hh::BandView<T> band,const hh::Event& event,
                       T* v,T* cross,T* w,typename real_type<T>::type& tau) noexcept {
    using R=typename real_type<T>::type;
    try {
        hh::DenseEventViews<T> view;
        if(!hh::dense_event_views(band,event,view)||!v||!cross||!w)return false;
        if(!hh::generate_reflector(band,event,v,tau))return false;
        if(tau==R(0))return true;
        blas_impl::hemv_run<DdlaBackend::CPU,T>(handle,'L',event.length,T(tau),view.diagonal,view.lda,v,1,T(0),w,1);
        T dot=0;
        for(int r=0;r<event.length;++r){if(!hh::finite(w[r]))return false;dot+=hh::adj(v[r])*w[r];}
        const R correction=-R(0.5)*tau*hh::real_part(dot);
        if(!hh::finite(correction))return false;
        for(int r=0;r<event.length;++r)w[r]+=correction*v[r];
        blas_impl::her2_run<DdlaBackend::CPU,T>(handle,'L',event.length,T(-1),v,1,w,1,view.diagonal,view.lda);
        if(view.left_columns) {
            blas_impl::gemv_run<DdlaBackend::CPU,T>(handle,'C',event.length,view.left_columns,T(1),view.left,view.lda,v,1,T(0),cross,1);
            blas_impl::gerc_run<DdlaBackend::CPU,T>(handle,event.length,view.left_columns,T(-tau),v,1,cross,1,view.left,view.lda);
        }
        if(view.right_rows) {
            blas_impl::gemv_run<DdlaBackend::CPU,T>(handle,'N',view.right_rows,event.length,T(1),view.right,view.lda,v,1,T(0),cross,1);
            blas_impl::gerc_run<DdlaBackend::CPU,T>(handle,view.right_rows,event.length,T(-tau),cross,1,v,1,view.right,view.lda);
        }
        // Retain the scalar path's finite-result rejection before publishing
        // this event. These scans read only the stored-lower write footprint.
        for(int c=0;c<event.length;++c)for(int r=c;r<event.length;++r)
            if(!hh::finite(view.diagonal[r+static_cast<std::size_t>(c)*view.lda]))return false;
        for(int c=0;c<view.left_columns;++c)for(int r=0;r<event.length;++r)
            if(!hh::finite(view.left[r+static_cast<std::size_t>(c)*view.lda]))return false;
        for(int c=0;c<event.length;++c)for(int r=0;r<view.right_rows;++r)
            if(!hh::finite(view.right[r+static_cast<std::size_t>(c)*view.lda]))return false;
        for(int r=event.start+1;r<event.start+event.length;++r)band.set(r,event.target,T(0));
        return true;
    }catch(...){return false;}
}

template<class T> struct Hb2stHouseholderCpu<T>::Impl {
    using R=typename Hb2stHouseholderCpu<T>::R;
    DdlaHandle_t handle=nullptr;
    MPI_Comm world=MPI_COMM_NULL,node=MPI_COMM_NULL;
    int n=0,b=1,rank=0,ranks=1,band_ld=1,sweeps=0,max_rows=1,threads=1,chase_threads=1;
    std::unique_ptr<DcHostTeam> chase_team;
    bool keep=false,reduced=false,released=false;
    Hb2stHouseholderCpuOptions options;
    SharedWindow band_window,history_window,factor_window;
    ChaseControl* control=nullptr;
    T* band=nullptr;
    std::size_t band_offset=0,history_offset=0;
    Index band_elements=0,local_elements=0,total_elements=0;
    std::vector<Index> sweep_panel0;
    std::vector<Index*> owner_prefix;
    std::vector<T*> owner_coeff;
    R scale=R(1);
#if DDLA_HAS_GPU
    int packet_levels=16,gpu_max_events=0,core_first=0,core_last=0,gpu_first=0,gpu_last=0;
    std::unique_ptr<ddla::vector<T>> gpu_band,gpu_packet;
    std::unique_ptr<ddla::vector<int>> gpu_failure;
    std::vector<T> host_packet;
#endif
    T* factors=nullptr;
    std::size_t factor_v_elements=0,factor_t_elements=0,factor_slot_elements=0,factor_capacity_elements=0;

    bool inject(Hb2stHouseholderCpuOptions::Failure point) const noexcept {
        return options.failure==point&&(options.failure_rank<0||options.failure_rank==rank);
    }
    void injected(Hb2stHouseholderCpuOptions::Failure point) const {
        if(inject(point))throw std::bad_alloc();
    }
    std::pair<std::size_t,std::size_t> band_share() const {
        const std::size_t size=static_cast<std::size_t>(band_elements);
        const std::size_t lo=size/static_cast<std::size_t>(ranks)*rank+
            std::min<std::size_t>(size%static_cast<std::size_t>(ranks),rank);
        const std::size_t count=size/static_cast<std::size_t>(ranks)+(static_cast<std::size_t>(rank)<size%static_cast<std::size_t>(ranks));
        return {lo,lo+count};
    }
    void setup(const DdlaHandle_t& h,const Hb2stHouseholderCpuPlan& p) {
        handle=h;world=h->comm;rank=p.rank;ranks=p.ranks;options=p.options;n=p.n;b=p.band;keep=p.keep_history;threads=p.threads;
        MPI_CHECK(MPI_Allreduce(&p.chase_threads,&chase_threads,1,MPI_INT,MPI_MIN,world));
        detail::agree_local(world,"CPU Householder preparation",[&]{if(chase_threads>1)chase_team=std::make_unique<DcHostTeam>(chase_threads);});
        detail::agree_local(world,"CPU Householder preparation",[&]{injected(Hb2stHouseholderCpuOptions::Failure::Metadata);});
        std::vector<Index> own_prefix;
        detail::agree_local(world,"CPU Householder preparation",[&] {
            band_ld=std::max(1,hh::band_rows(n,b));band_elements=checked_multiply(n,band_ld);
            band_offset=align_shared(sizeof(ChaseControl));checked_add(band_offset,checked_bytes(band_elements,sizeof(T)));
            sweeps=hh::sweep_count(n,b);
            sweep_panel0.assign(std::size_t(sweeps)+1,0);own_prefix.assign(std::size_t(sweeps)+1,0);
            owner_prefix.assign(std::size_t(ranks),nullptr);owner_coeff.assign(std::size_t(ranks),nullptr);
            for(int j=0;j<sweeps;++j){const int length=hh::sweep_length(n,b,j),panels=hh::panel_count(length,factor_width);Index own=0,all=0;
                if(!hh::owner_prefix(length,b,factor_width,ranks,sweep_panel0[j],rank,panels,own)||!hh::prefix_coeff(b,length,all))
                    throw std::overflow_error("CPU Householder history geometry overflow");
                own_prefix[j+1]=checked_add(own_prefix[j],own);sweep_panel0[j+1]=checked_add(sweep_panel0[j],panels);
                total_elements=checked_add(total_elements,all);
            }
            local_elements=keep?own_prefix.back():0;history_offset=align_shared(checked_bytes(Index(sweeps)+1,sizeof(Index)));
            checked_add(history_offset,checked_bytes(local_elements,sizeof(T)));
            // One fixed slot geometry for every back-transform keeps the
            // layout of the live shared factor mapping.
            max_rows=std::max(1,int(std::min<Index>(n,Index(b)+factor_width-1)));
            factor_v_elements=checked_bytes(checked_multiply(max_rows,factor_width),1);factor_t_elements=std::size_t(factor_width)*factor_width;
            const Index wy=checked_add(factor_v_elements,factor_t_elements),paired=checked_add(checked_multiply(b,factor_width),paired_overlaps);
            const Index grouped=checked_add(paired,hhgroup_real::gram_elements<T,6>(factor_width));
            factor_slot_elements=std::size_t(std::max({wy,paired,grouped}));
            checked_bytes(checked_multiply(factor_slot_elements,factor_batch),sizeof(T));
            band_window.metadata(ranks);history_window.metadata(ranks);factor_window.metadata(ranks);
        });
        MPI_CHECK(MPI_Comm_split_type(world,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&node));
        MPI_CHECK(MPI_Comm_set_errhandler(node,MPI_ERRORS_RETURN));
        detail::agree_local(world,"CPU Householder preparation",[&] {
            int node_size=0,node_rank=0;
            MPI_CHECK(MPI_Comm_size(node,&node_size));MPI_CHECK(MPI_Comm_rank(node,&node_rank));
            if(node_size!=ranks||node_rank!=rank)
                throw std::runtime_error("CPU Householder prototype requires one shared-memory node");
        });
        detail::agree_local(world,"CPU Householder preparation",[&] {injected(Hb2stHouseholderCpuOptions::Failure::Band);});
        const std::size_t band_bytes=rank==0?static_cast<std::size_t>(checked_add(band_offset,checked_bytes(band_elements,sizeof(T)))):0;
        band_window.allocate(node,world,band_bytes,options.reject_memory_model);
        detail::agree_local(world,"CPU Householder preparation",[&] {
            auto* base=static_cast<unsigned char*>(band_window.bases[0]);
            if(!base||band_window.sizes[0]<static_cast<MPI_Aint>(band_offset+checked_bytes(band_elements,sizeof(T))))
                throw std::runtime_error("CPU Householder band mapping is too short");
            control=reinterpret_cast<ChaseControl*>(base);
            band=reinterpret_cast<T*>(base+band_offset);
            if(rank==0)::new(static_cast<void*>(control)) ChaseControl();
            const auto range=band_share();
            for(std::size_t q=range.first;q<range.second;++q)::new(static_cast<void*>(band+q)) T();
        });
        band_window.sync();MPI_CHECK(MPI_Barrier(node));band_window.sync();
        if(keep) {
            detail::agree_local(world,"CPU Householder preparation",[&] {injected(Hb2stHouseholderCpuOptions::Failure::History);});
            const std::size_t bytes=static_cast<std::size_t>(checked_add(history_offset,checked_bytes(local_elements,sizeof(T))));
            history_window.allocate(node,world,bytes,false);
            detail::agree_local(world,"CPU Householder preparation",[&] {
                for(int owner=0;owner<ranks;++owner) {
                    auto* base=static_cast<unsigned char*>(history_window.bases[owner]);
                    if(!base||history_window.sizes[owner]<static_cast<MPI_Aint>(history_offset))
                        throw std::runtime_error("CPU Householder history mapping is too short");
                    owner_prefix[owner]=reinterpret_cast<Index*>(base);
                    owner_coeff[owner]=reinterpret_cast<T*>(base+history_offset);
                }
                for(int j=0;j<=sweeps;++j)::new(static_cast<void*>(owner_prefix[rank]+j)) Index(own_prefix[j]);
                for(Index q=0;q<local_elements;++q)::new(static_cast<void*>(owner_coeff[rank]+q)) T();
            });
            history_window.sync();MPI_CHECK(MPI_Barrier(node));history_window.sync();
            detail::agree_local(world,"CPU Householder preparation",[&] {
                for(int owner=0;owner<ranks;++owner) {
                    const Index count=owner_prefix[owner][sweeps];
                    const Index expected=checked_add(history_offset,checked_bytes(count,sizeof(T)));
                    if(expected>static_cast<Index>(history_window.sizes[owner]))
                        throw std::runtime_error("CPU Householder owner history exceeds mapping");
                }
            });
        }
#if DDLA_HAS_GPU
        detail::agree_local(world,"CPU Householder preparation",[&] {
            if(options.stage2!=Hb2stHouseholderCpuOptions::Stage2::Gpu)return;
            if(h->backend!=DdlaBackend::GPU||b>64)
                throw std::invalid_argument("GPU Householder stage2 requires a GPU handle and band<=64");
            packet_levels=ranks==1?64:16;
            core_first=int(std::int64_t(n)*rank/ranks);
            core_last=int(std::int64_t(n)*(rank+1)/ranks);
            // An event's column footprint has width <=2*b. After each level,
            // shrink the valid interval by2*b; a16-level halo protects the core.
            const int halo=2*b*packet_levels;
            gpu_first=std::max(0,core_first-halo);
            gpu_last=std::min(n,core_last+halo);
            gpu_max_events=(gpu_last-gpu_first)/(3*b-1)+2;
            gpu_failure=std::make_unique<ddla::vector<int>>(h,1);
            gpu_band=std::make_unique<ddla::vector<T>>(h,std::max<std::size_t>(1,std::size_t(gpu_last-gpu_first)*band_ld));
            if(keep) {
                const std::size_t slots=std::size_t(packet_levels)*gpu_max_events*b;
                gpu_packet=std::make_unique<ddla::vector<T>>(h,slots);
                host_packet.resize(slots);
                DDLA_CHECK(ddlaMemset(h,gpu_packet->data(),0,slots*sizeof(T)));
                DDLA_CHECK(ddlaSynchronize(h));
            }
        });
#endif
        if(keep&&sweeps)prepare_factor_window();
        (void)hhpair_fma::supported();
    }
    void prepare_factor_window() {
        const Index required=checked_multiply(factor_slot_elements,factor_batch);
        const std::size_t required_bytes=checked_bytes(required,sizeof(T));
        if(factor_window.window==MPI_WIN_NULL) {
            factor_window.allocate(node,world,rank==0?required_bytes:0,false);
            detail::agree_local(world,"CPU Householder preparation",[&] {
                auto* base=factor_window.bases[0];
                if(!base||factor_window.sizes[0]<static_cast<MPI_Aint>(required_bytes))
                    throw std::runtime_error("CPU Householder factor mapping is too short");
                factors=static_cast<T*>(base);
                for(int slot=rank;slot<factor_batch;slot+=ranks)
                    for(std::size_t q=0;q<factor_slot_elements;++q)
                        ::new(static_cast<void*>(factors+static_cast<std::size_t>(slot)*factor_slot_elements+q)) T();
                factor_capacity_elements=static_cast<std::size_t>(required);
            });
            factor_window.sync();MPI_CHECK(MPI_Barrier(node));factor_window.sync();
        }
        detail::agree_local(world,"CPU Householder preparation",[&] {
            if(!factors||required>factor_capacity_elements||factor_window.sizes[0]<static_cast<MPI_Aint>(required_bytes))
                throw std::runtime_error("CPU Householder factor capacity changed");
        });
    }
    const T* record(int sweep,int stage) const noexcept {
        if(!keep||sweep<0||sweep>=sweeps)return nullptr;
        int owner=0;Index relative=0,count=0;
        if(!hh::record_offset(hh::sweep_length(n,b,sweep),b,factor_width,ranks,sweep_panel0[sweep],stage,owner,relative,count))return nullptr;
        const Index begin=owner_prefix[owner][sweep],end=owner_prefix[owner][sweep+1];
        if(end<begin||relative>end-begin||count>end-begin-relative)return nullptr;
        return owner_coeff[owner]+begin+relative;
    }
    void close() {
        if(released)return;
        // Every own failure is agreed before reaching this retirement barrier.
        // The caller explicitly retires the state on normal and cooperative
        // exception paths. No destructor initiates collective MPI operations.
        if(world!=MPI_COMM_NULL)MPI_CHECK(MPI_Barrier(world));
#if DDLA_HAS_GPU
        gpu_failure.reset();gpu_packet.reset();gpu_band.reset();host_packet.clear();
#endif
        factor_window.close();history_window.close();band_window.close();
        if(node!=MPI_COMM_NULL)MPI_CHECK(MPI_Comm_free(&node));
        control=nullptr;band=nullptr;factors=nullptr;released=true;
    }
};

template<class T>
Hb2stHouseholderCpu<T>::Hb2stHouseholderCpu(const DdlaHandle_t& h,int n,int b,bool keep,const Hb2stHouseholderCpuOptions& options)
    :Hb2stHouseholderCpu(h,hh_cpu_check(h,n,b,keep,options)) {}
template<class T>
Hb2stHouseholderCpu<T>::Hb2stHouseholderCpu(const DdlaHandle_t& h,const Hb2stHouseholderCpuPlan& p) {
    detail::agree_local(h->comm,"CPU Householder preparation",[&]{impl_.reset(new Impl);});
    try{impl_->setup(h,p);}catch(...){impl_->close();throw;}
}

template<class T> Hb2stHouseholderCpu<T>::~Hb2stHouseholderCpu() = default;

template<class T>
void Hb2stHouseholderCpu<T>::reduce_run(const T* A,const int* desc,R* d,R* e,T* phases,void* backend_work,std::size_t backend_bytes,const T* packed_upper) {
    auto& s=*impl_;
    char* cursor=static_cast<char*>(backend_work);
    const std::size_t scratch_count=std::size_t(s.b)*s.chase_threads;
    T* const vectors=take<T>(cursor,scratch_count);
    T* const products=take<T>(cursor,scratch_count);
    T* const updates=take<T>(cursor,scratch_count);
    assert(std::size_t(cursor-static_cast<char*>(backend_work))<=backend_bytes);(void)backend_bytes;
    s.reduced=false;
    if(s.rank==0)s.control->abort.store(0,std::memory_order_release);
    const auto reset_range=s.band_share();for(std::size_t q=reset_range.first;q<reset_range.second;++q)s.band[q]=T(0);
    s.band_window.sync();MPI_CHECK(MPI_Barrier(s.node));s.band_window.sync();
    int nr=0,nc=0,pr=0,pc=0;ddlaGetGridDims(s.handle,nr,nc);ddlaGetGridCoords(s.handle,pr,pc);
    detail::agree_local(s.world,"CPU Householder preparation",[&] {
        const int mb=desc[DDLA_MB_],nb=desc[DDLA_NB_],rs=desc[DDLA_RSRC_],cs=desc[DDLA_CSRC_],ld=desc[DDLA_LLD_];
        for(int col=0;col<s.n;++col) {
            if((col/nb+cs)%nc!=pc)continue;
            const int lj=(col/nb/nc)*nb+col%nb;
            for(int row=std::max(0,col-s.b);row<=col;++row) {
                if((row/mb+rs)%nr!=pr)continue;
                const int li=(row/mb/nr)*mb+row%mb;
                const T value=packed_upper
                    ? packed_upper[std::size_t(col)*(s.b+1)+s.b+row-col]
                    : A[li+static_cast<std::size_t>(lj)*ld];
                // Stage 1 stores the upper band. Its conjugate is the
                // packed lower-band entry consumed by the Hermitian core.
                s.band[static_cast<std::size_t>(row)*s.band_ld+col-row]=row==col?T(hh::real_part(value)):hh::adj(value);
            }
        }
    });
    s.band_window.sync();MPI_CHECK(MPI_Barrier(s.node));s.band_window.sync();
    R local_max=0;int finite=1;
    const auto range=s.band_share();
    for(std::size_t q=range.first;q<range.second;++q) {
        if(!hh::finite(s.band[q]))finite=0;
        else local_max=std::max(local_max,hh::component_scale(s.band[q]));
    }
    int all_finite=0;MPI_CHECK(MPI_Allreduce(&finite,&all_finite,1,MPI_INT,MPI_MIN,s.world));
    detail::agree_local(s.world,"CPU Householder preparation",[&] {if(!all_finite)throw std::invalid_argument("CPU Householder band contains non-finite values");});
    MPI_CHECK(MPI_Allreduce(&local_max,&s.scale,1,mpi_datatype<R>(),MPI_MAX,s.world));
    if(s.scale==R(0))s.scale=R(1);
    for(std::size_t q=range.first;q<range.second;++q)s.band[q]=hh::divide_real(s.band[q],s.scale);
    s.band_window.sync();MPI_CHECK(MPI_Barrier(s.node));s.band_window.sync();
    hh::BandView<T> band{s.band,s.n,s.b,s.band_ld};
    if(s.n>=3&&s.b>=2) {
        const Index last=Index(3)*static_cast<Index>(s.n-3),inject_level=last/2;
#if DDLA_HAS_GPU
        if(s.options.stage2==Hb2stHouseholderCpuOptions::Stage2::Gpu) {
            detail::agree_local(s.world,"CPU Householder preparation",[&] {
                DDLA_CHECK(ddlaMemcpy(s.handle,s.gpu_band->data(),s.band+std::size_t(s.gpu_first)*s.band_ld,
                    std::size_t(s.gpu_last-s.gpu_first)*s.band_ld*sizeof(T)));
                DDLA_CHECK(ddlaMemset(s.handle,s.gpu_failure->data(),0,sizeof(int)));
                DDLA_CHECK(ddlaSynchronize(s.handle));
            });
            for(int first=0;first<=int(last);first+=s.packet_levels) {
                const int levels=std::min(s.packet_levels,int(last)+1-first);
                detail::agree_local(s.world,"CPU Householder preparation",[&] {
                    if(s.inject(Hb2stHouseholderCpuOptions::Failure::Reduce)&&
                       Index(first)<=inject_level&&inject_level<Index(first+levels))
                        throw std::runtime_error("GPU Householder injected reduction failure");
                    hh_gpu_reduce_levels<T>(s.handle,s.n,s.b,s.band_ld,first,levels,s.gpu_max_events,
                        s.gpu_first,s.gpu_last,s.gpu_band->data(),s.keep?s.gpu_packet->data():nullptr,s.gpu_failure->data());
                    if(s.keep) {
                        const std::size_t count=std::size_t(levels)*s.gpu_max_events*s.b;
                        DDLA_CHECK(ddlaMemcpy(s.handle,s.host_packet.data(),s.gpu_packet->data(),count*sizeof(T)));
                    }
                    // The device core stays resident between batches. Publish
                    // it to the compact host band only after the last level.
                    if(first+levels>int(last))
                        DDLA_CHECK(ddlaMemcpy(s.handle,s.band+std::size_t(s.core_first)*s.band_ld,
                            s.gpu_band->data()+std::size_t(s.core_first-s.gpu_first)*s.band_ld,
                            std::size_t(s.core_last-s.core_first)*s.band_ld*sizeof(T)));
                    int failure=0;
                    DDLA_CHECK(ddlaMemcpy(s.handle,&failure,s.gpu_failure->data(),sizeof(int)));
                    DDLA_CHECK(ddlaSynchronize(s.handle));
                    if(failure)throw std::runtime_error("GPU Householder chase produced non-finite values");
                    if(s.keep)for(int k=0;k<levels;++k) {
                        int lo=0,hi=-1;
                        if(!hh::local_level_bounds(s.n,s.b,first+k,s.gpu_first,s.gpu_last,2*s.b*k,lo,hi))continue;
                        for(int stage=lo;stage<=hi;++stage) {
                            const int sweep=first+k-3*stage;
                            hh::Event event;hh::event_geometry(s.n,s.b,stage,sweep,event);
                            if(event.start<s.core_first||event.start>=s.core_last)continue;
                            T* out=const_cast<T*>(s.record(sweep,stage));
                            if(!out)throw std::runtime_error("GPU Householder history offset is invalid");
                            const T* src=s.host_packet.data()+(std::size_t(k)*s.gpu_max_events+stage-lo)*s.b;
                            std::copy(src,src+event.length,out);
                        }
                    }
                });
                if(first+levels<=int(last)) {
                    // Each peer sends only its owned columns intersecting this
                    // rank's halo. No halo result is ever used as a send source.
                    // Grouped NCCL/RCCL stays on the handle stream; the existing
                    // Sendrecv primitive supplies the host-staged MPI fallback.
                    commGroupStart<DdlaBackend::GPU>(s.handle);
                    for(int peer=0;peer<s.ranks;++peer) {
                        if(peer==s.rank)continue;
                        const int peer_first=int(std::int64_t(s.n)*peer/s.ranks);
                        const int peer_last=int(std::int64_t(s.n)*(peer+1)/s.ranks);
                        const int halo=2*s.b*s.packet_levels;
                        const int send_first=std::max(s.core_first,std::max(0,peer_first-halo));
                        const int send_last=std::min(s.core_last,std::min(s.n,peer_last+halo));
                        const int recv_first=std::max(s.gpu_first,peer_first);
                        const int recv_last=std::min(s.gpu_last,peer_last);
                        const std::size_t send_count=std::size_t(std::max(0,send_last-send_first))*s.band_ld;
                        const std::size_t recv_count=std::size_t(std::max(0,recv_last-recv_first))*s.band_ld;
                        if(!send_count&&!recv_count)continue;
                        // A zero-count pointer still names allocated storage.
                        const T* send=s.gpu_band->data()+(send_count?std::size_t(send_first-s.gpu_first)*s.band_ld:0);
                        T* recv=s.gpu_band->data()+(recv_count?std::size_t(recv_first-s.gpu_first)*s.band_ld:0);
                        commSendrecv<DdlaBackend::GPU>(s.handle,CommScope::Grid,
                            send,send_count,peer,recv,recv_count,peer);
                    }
                    commGroupEnd<DdlaBackend::GPU>(s.handle);
                }
            }
            s.band_window.sync();MPI_CHECK(MPI_Barrier(s.node));s.band_window.sync();
            s.history_window.sync();MPI_CHECK(MPI_Barrier(s.node));s.history_window.sync();
        } else
#endif
        {
        const int participants=s.ranks*s.chase_threads;
        // Independent events at one level may run on ranks and local workers.
        // Each worker owns its reflector scratch and drains every common level.
        auto chase=[&](int worker) {
            const int member=s.rank*s.chase_threads+worker;
            T* const v=vectors+std::size_t(worker)*s.b;
            T* const av=products+std::size_t(worker)*s.b;
            T* const w=updates+std::size_t(worker)*s.b;
            for(Index level=0;level<=last;++level) {
                if(worker==0&&level==inject_level&&s.inject(Hb2stHouseholderCpuOptions::Failure::Reduce))
                    s.control->abort.store(1,std::memory_order_release);
                // The preceding level's exit barrier publishes its band and
                // history writes. Keep the initial barrier and the common
                // injection barrier so every rank sees a selected-rank abort
                // before attempting that level's events.
                if(level==0||level==inject_level)chase_barrier(s.control,participants);
                if(!s.control->abort.load(std::memory_order_acquire)) {
                    int lo=0,hi=-1;
                    if(hh::level_bounds(s.n,s.b,level,lo,hi))
                        for(std::int64_t stage=static_cast<std::int64_t>(lo)+member;stage<=hi;stage+=participants) {
                            if(s.control->abort.load(std::memory_order_acquire))break;
                            const int sweep=static_cast<int>(level-Index(3)*static_cast<Index>(stage));
                            hh::Event event;R tau=0;
                            const bool geometry=hh::event_geometry(s.n,s.b,static_cast<int>(stage),sweep,event);
                            const bool applied=geometry&&(s.options.stage2==Hb2stHouseholderCpuOptions::Stage2::Blas?
                                hh_event_blas_cpu<T>(s.handle,band,event,v,av,w,tau):
                                hh::reduce_event(band,event,v,av,w,tau));
                            if(!applied) {
                                s.control->abort.store(1,std::memory_order_release);break;
                            }
                            if(s.keep) {
                                const T* pointer=s.record(sweep,static_cast<int>(stage));
                                if(!pointer){s.control->abort.store(1,std::memory_order_release);break;}
                                T* out=const_cast<T*>(pointer);out[0]=T(tau);
                                for(int r=1;r<event.length;++r)out[r]=v[r];
                            }
                        }
                }
                chase_barrier(s.control,participants);
            }
        };
        // Parallel events each invoke small BLAS operations. A single BLAS
        // worker per event avoids nesting the rank's BLAS pool inside the team.
        ScopedBlasThreads serial_blas(1);
        if(s.chase_threads==1)chase(0);
        else if(!host_run_team(s.chase_threads,chase))s.chase_team->run_roles(s.chase_threads,chase);
        }
    } else if(s.inject(Hb2stHouseholderCpuOptions::Failure::Reduce))s.control->abort.store(1,std::memory_order_release);
    s.band_window.sync();s.history_window.sync();
    detail::agree_local(s.world,"CPU Householder preparation",[&] {
        if(s.control->abort.load(std::memory_order_acquire))throw std::runtime_error("CPU Householder chase failed; all levels drained");
        T previous=T(1);
        for(int i=0;i<s.n;++i) {
            d[i]=hh::real_part(band.get(i,i));
            e[i]=R(0);
            if(!hh::finite(d[i]))throw std::runtime_error("CPU Householder diagonal is not finite");
            if(i+1==s.n)continue;
            const T off=band.get(i+1,i);
            if constexpr(std::is_same<T,R>::value)e[i]=off;
            else {
                // Compute magnitude from scaled components and normalize
                // every phase product; subnormal complex division is avoided.
                const R magnitude=hh::component_scale(off);
                if(magnitude==R(0))e[i]=R(0);
                else {
                    const T scaled=hh::divide_real(off,magnitude);
                    e[i]=magnitude*std::hypot(scaled.real(),scaled.imag());
                }
                previous=hh::unit_phase(previous*hh::unit_phase(off));phases[i]=previous;
            }
            if(!hh::finite(d[i])||!hh::finite(e[i]))throw std::runtime_error("CPU Householder tridiagonal is not finite");
        }
        s.reduced=true;
    });
}

template<class T>
void Hb2stHouseholderCpu<T>::apply_columns_run(T* columns,int ld,int ncols,void* backend_work,std::size_t backend_bytes) {
    apply_wy_run(columns,ld,ncols,backend_work,backend_bytes,false);
}

template<class T>
void hh_gpu_apply_workspace_size(int band,int columns,bool wy,std::size_t* bytes) {
    const bool compact=!wy && hh_gpu_compact_band(band);
    const std::size_t slot=compact ? std::size_t(band)*factor_width :
        (std::size_t(band)+factor_width-1)*factor_width+std::size_t(factor_width)*factor_width;
    *bytes=padded_bytes<T>(slot*factor_batch);
    if(!compact)*bytes+=padded_bytes<T>(std::size_t(factor_width)*columns);
}
#if DDLA_HAS_GPU
template<class T>
void Hb2stHouseholderCpu<T>::apply_rows_gpu_run(T* rows,int ld,int ncols,void* work,std::size_t bytes) {
    apply_wy_run(rows,ld,ncols,work,bytes,true);
}
#endif

template<class T>
void Hb2stHouseholderCpu<T>::apply_wy_run(T* columns,int ld,int ncols,void* backend_work,std::size_t backend_bytes,bool device_rows) {
    auto& s=*impl_;
    char* cursor=static_cast<char*>(backend_work);
    const bool compact=device_rows && s.options.stage4!=Hb2stHouseholderCpuOptions::Stage4::Wy && hh_gpu_compact_band(s.b);
    const std::size_t gpu_slot=compact ? std::size_t(s.b)*factor_width : s.factor_v_elements+s.factor_t_elements;
    T* const device_factors=device_rows?take<T>(cursor,gpu_slot*factor_batch):nullptr;
    T* const work=compact?nullptr:take<T>(cursor,std::size_t(factor_width)*std::size_t(ncols));
    assert(std::size_t(cursor-static_cast<char*>(backend_work))<=backend_bytes);(void)backend_bytes;
    std::array<T,factor_width> tri_work;
    detail::agree_local(s.world,"CPU Householder preparation",[&]{s.injected(Hb2stHouseholderCpuOptions::Failure::Factors);});
    if(s.sweeps==0)return;
    int sweep=0,panel=0;
    while(sweep<s.sweeps) {
        std::array<hh::Panel,factor_batch> batch;
        int count=0;
        detail::agree_local(s.world,"CPU Householder preparation",[&] {
            while(count<factor_batch&&sweep<s.sweeps) {
                if(!hh::panel_geometry(s.n,s.b,factor_width,sweep,panel,batch[count]))
                    throw std::logic_error("CPU Householder replay panel geometry is invalid");
                ++count;++panel;
                if(panel==hh::panel_count(hh::sweep_length(s.n,s.b,sweep),factor_width)){++sweep;panel=0;}
            }
            for(int slot=s.rank;slot<count;slot+=s.ranks) {
                const auto& geometry=batch[slot];
                T* V=s.factors+static_cast<std::size_t>(slot)*s.factor_slot_elements;
                T* tri=V+s.factor_v_elements;
                int row0=0,rows=0;
                const auto packed_at=[&](int stage) noexcept {return s.record(geometry.sweep,stage);};
                if(compact) {
                    hhpair::PanelView<T> view;
                    if(!hhpair::prepare_panel(s.n,s.b,geometry.sweep,geometry.first_stage,
                        geometry.count,packed_at,V,s.b,V+std::size_t(s.b)*factor_width,view))
                        throw std::runtime_error("GPU Householder panel packing failed");
                } else {
                    if(!hh::form_wy(s.n,s.b,geometry.sweep,geometry.first_stage,geometry.count,
                        packed_at,V,s.max_rows,tri,factor_width,tri_work.data(),row0,rows)||
                        row0!=geometry.row0||rows!=geometry.rows)
                        throw std::runtime_error("CPU Householder WY formation failed");
                    if(device_rows)
                        for(std::size_t i=0;i<gpu_slot;++i)V[i]=hh::adj(V[i]);
                }
            }
            s.factor_window.sync();
        });
        s.factor_window.sync();
        detail::agree_local(s.world,"CPU Householder preparation",[&] {
            s.injected(Hb2stHouseholderCpuOptions::Failure::Apply);
            if(ncols==0)return;
            if(device_rows) {
                // Shared factors stay bounded by factor_batch; no complete
                // history is replicated on any device. This rank retires its
                // host reads before another rank can overwrite these slots.
                DDLA_CHECK(ddlaMemcpy2D(s.handle,device_factors,gpu_slot*sizeof(T),
                    s.factors,s.factor_slot_elements*sizeof(T),gpu_slot*sizeof(T),count));
            }
            for(int slot=0;slot<count;++slot) {
                const auto& geometry=batch[slot];
                const T* V=s.factors+static_cast<std::size_t>(slot)*s.factor_slot_elements;
                const T* tri=V+s.factor_v_elements;
                if(device_rows) {
#if DDLA_HAS_GPU
                    T* dv=device_factors+std::size_t(slot)*gpu_slot;
                    T* dt=compact?nullptr:dv+s.factor_v_elements;
                    // The device strip is X^T. Applying (I-V*T*V^H) X
                    // therefore applies its transpose on the right. PUNMTR
                    // receives conj(V), conj(T) and trans='C'.
                    if(compact) {
                        hh::Event first;
                        hh::event_geometry(s.n,s.b,geometry.first_stage,geometry.sweep,first);
                        hh_gpu_panel_run<T>(s.handle,columns,ld,ncols,s.n,s.b,first.start,geometry.count,dv);
                    } else {
                        punmtr_run<DdlaBackend::GPU,T>(s.handle,'R','U','C',ncols,geometry.rows,
                            geometry.count,s.max_rows,factor_width,ld,std::max(ncols,1),false,
                            dv,dt,columns+std::size_t(geometry.row0)*ld,work,nullptr);
                    }
#endif
                } else {
                    punmtr_run<DdlaBackend::CPU,T>(s.handle,'L','U','N',geometry.rows,ncols,
                        geometry.count,s.max_rows,factor_width,ld,factor_width,false,
                        V,tri,columns+geometry.row0,work,nullptr);
                }
            }
            if(device_rows) DDLA_CHECK(ddlaSynchronize(s.handle));
            // Every host/device reader completes before factor-slot reuse.
            s.factor_window.sync();
        });
        s.factor_window.sync();
    }
}

template<class T>
void Hb2stHouseholderCpu<T>::apply_tiles_run(R* tiles,int width,int ncols) {
    auto& s=*impl_;
    const int threads=s.threads;
    const Index record_capacity=Index(s.b)*factor_width;
    detail::agree_local(s.world,"CPU Householder preparation",[&]{s.injected(Hb2stHouseholderCpuOptions::Failure::Factors);});
    if(s.sweeps==0)return;
    int sweep=0,panel=0;
    while(sweep<s.sweeps) {
        std::array<hh::Panel,factor_batch> batch;
        int count=0;
        detail::agree_local(s.world,"CPU Householder preparation",[&] {
            while(count<factor_batch&&sweep<s.sweeps) {
                if(!hh::panel_geometry(s.n,s.b,factor_width,sweep,panel,batch[count]))
                    throw std::logic_error("CPU Householder paired panel geometry is invalid");
                ++count;++panel;
                if(panel==hh::panel_count(hh::sweep_length(s.n,s.b,sweep),factor_width)) {
                    ++sweep;panel=0;
                }
            }
            for(int slot=s.rank;slot<count;slot+=s.ranks) {
                const auto& geometry=batch[slot];
                T* records=s.factors+static_cast<std::size_t>(slot)*s.factor_slot_elements;
                T* overlap=records+record_capacity;
                const auto packed_at=[&](int stage) noexcept {return s.record(geometry.sweep,stage);};
                hhpair::PanelView<T> view;
                if(!hhpair::prepare_panel(s.n,s.b,geometry.sweep,geometry.first_stage,geometry.count,
                    packed_at,records,s.b,overlap,view)||view.first_row!=geometry.row0+geometry.count-1)
                    throw std::runtime_error("CPU Householder paired preparation failed");
                // This injection proves that a partially prepared packet
                // cannot reach readers on any rank.
                s.injected(Hb2stHouseholderCpuOptions::Failure::Packet);
                if constexpr(hhpair::is_real<T>) {
                    T* gram=overlap+paired_overlaps;
                    constexpr std::size_t capacity=hhgroup_real::gram_elements<T,6>(factor_width);
                    if(s.options.stage4==Hb2stHouseholderCpuOptions::Stage4::GroupedReal4) {
                        hhgroup_real::PanelView<T,4> grouped;
                        if(!hhgroup_real::prepare_groups<T,4>(view,gram,capacity,grouped))
                            throw std::runtime_error("CPU Householder group4 preparation failed");
                    } else if(s.options.stage4==Hb2stHouseholderCpuOptions::Stage4::GroupedReal6) {
                        hhgroup_real::PanelView<T,6> grouped;
                        if(!hhgroup_real::prepare_groups<T,6>(view,gram,capacity,grouped))
                            throw std::runtime_error("CPU Householder group6 preparation failed");
                    }
                }
            }
            s.factor_window.sync();
        });
        // Retain the owner's current publish barrier/Win_sync ordering.
        s.factor_window.sync();
        detail::agree_local(s.world,"CPU Householder preparation",[&] {
            s.injected(Hb2stHouseholderCpuOptions::Failure::Apply);
            if(ncols==0)return;
            auto run=[&](auto tag) {
                constexpr int W=decltype(tag)::value;
                const int nt=ncols/W+(ncols%W!=0);
                const std::size_t stride=static_cast<std::size_t>(s.n)*W*hhpair::planes<T>;
                std::array<hhpair::PanelView<T>,factor_batch> views;
                for(int slot=0;slot<count;++slot) {
                    const auto& geometry=batch[slot];
                    const T* records=s.factors+static_cast<std::size_t>(slot)*s.factor_slot_elements;
                    views[slot]={s.n,s.b,geometry.row0+geometry.count-1,
                        geometry.count,s.b,records,records+record_capacity};
                }
                // Views contain process-local mapped addresses and live
                // until the one joined worker pass has completely ended.
                auto execute=[&](const auto& local_views,auto kernel) {
                    host_parallel_for(nt,1,threads,[&](long long lo,long long hi) {
                        for(long long t=lo;t<hi;++t) {
                            R* tile=tiles+static_cast<std::size_t>(t)*stride;
                            const int active=std::min(W,ncols-static_cast<int>(t)*W);
                            for(int slot=0;slot<count;++slot) {
                                if(!kernel(tile,local_views[slot],active))
                                    throw std::runtime_error("CPU Householder tiled application failed");
                            }
                        }
                    });
                };
                auto grouped=[&](auto group_tag) {
                    constexpr int K=decltype(group_tag)::value;
                    std::array<hhgroup_real::PanelView<T,K>,factor_batch> local_views;
                    for(int slot=0;slot<count;++slot) {
                        auto& view=local_views[slot];
                        static_cast<hhpair::PanelView<T>&>(view)=views[slot];
                        if constexpr(hhpair::is_real<T>)
                            view.gram=views[slot].overlap+paired_overlaps;
                        view.gram_elements=hhgroup_real::gram_elements<T,K>(views[slot].count);
                    }
                    execute(local_views,[](R* tile,const auto& view,int active) {
                        return hhgroup_real_fast::apply_panel<T,W,K>(tile,view,active);
                    });
                };
                using Method=Hb2stHouseholderCpuOptions::Stage4;
                switch(s.options.stage4) {
                case Method::FmaChain:
                    execute(views,[](R* tile,const auto& view,int active) {
                        return hhpair_fma::apply_panel<T,W,hhpair_fma::Strategy::Chain>(tile,view,active);
                    });break;
                case Method::FmaIndependent:
                    execute(views,[](R* tile,const auto& view,int active) {
                        return hhpair_fma::apply_panel<T,W,hhpair_fma::Strategy::Independent>(tile,view,active);
                    });break;
                case Method::GroupedReal4:grouped(std::integral_constant<int,4>{});break;
                case Method::GroupedReal6:grouped(std::integral_constant<int,6>{});break;
                default:
                    execute(views,[](R* tile,const auto& view,int active) {
                        return hhpair::apply_panel<T,W>(tile,view,active);
                    });break;
                }
            };
            switch(width) {
            case 8:run(std::integral_constant<int,8>{});break;
            case 16:run(std::integral_constant<int,16>{});break;
            case 32:run(std::integral_constant<int,32>{});break;
            default:run(std::integral_constant<int,64>{});break;
            }
            // host_parallel_for joins before this reader retires.
            s.factor_window.sync();
        });
        // Retain the owner's current reader-drain barrier/Win_sync order.
        s.factor_window.sync();
    }
}

template<class T>
void Hb2stHouseholderCpu<T>::reduce_check(const T* A,const int* desc,R* d,R* e,T* phases,std::size_t phase_count)const {
    const auto& s=*impl_;
    detail::agree_local(s.world,"CPU Householder preparation",[&]{
        if(s.released||s.reduced)throw std::logic_error("CPU Householder reduce called in wrong state");
        check_desc(desc,s.handle);
        if(s.n>desc[DDLA_M_]||s.n>desc[DDLA_N_]||(s.n&&(!d||!e)))throw std::invalid_argument("CPU Householder input/output is invalid");
        if constexpr(!std::is_same<T,R>::value)if(phase_count<std::size_t(std::max(s.n-1,0))||(s.n>1&&!phases))throw std::invalid_argument("CPU Householder phases are too short");
        if(!A)for(int col=0;col<s.n;++col){if((col/desc[DDLA_NB_]+desc[DDLA_CSRC_])%s.handle->npcols_!=s.handle->mypcol_)continue;
            for(int row=std::max(0,col-s.b);row<=col;++row)if((row/desc[DDLA_MB_]+desc[DDLA_RSRC_])%s.handle->nprows_==s.handle->myprow_)throw std::invalid_argument("CPU Householder local input missing");
        }
    });
}
template<class T>
void Hb2stHouseholderCpu<T>::reduce(const T* A,const int* desc,R* d,R* e,std::vector<T>& phases){
    auto& s=*impl_;
    try{std::size_t backend_bytes=0;hh_cpu_reduce_workspace_size<T>(s.handle,s.b,s.chase_threads,&backend_bytes);ddla::vector<char> backend_work(s.handle);
        detail::agree_local(s.world,"CPU Householder preparation",[&]{if constexpr(std::is_same<T,R>::value)phases.clear();else phases.resize(std::size_t(std::max(s.n-1,0)));backend_work.resize(backend_bytes);});
        reduce_check(A,desc,d,e,phases.data(),phases.size());reduce_run(A,desc,d,e,phases.data(),backend_work.data(),backend_bytes);
    }catch(...){s.close();throw;}
}
template<class T>
void Hb2stHouseholderCpu<T>::apply_columns_check(T* columns,int ld,int count)const{
    const auto& s=*impl_;detail::agree_local(s.world,"CPU Householder preparation",[&]{
        if(s.released||!s.reduced||!s.keep)throw std::logic_error("CPU Householder apply called in wrong state");
        if(count<0||ld<std::max(s.n,1)||(count&&s.n&&!columns))throw std::invalid_argument("CPU Householder column strip is invalid");
    });
}
template<class T>
void Hb2stHouseholderCpu<T>::apply_columns(T* columns,int ld,int count){
    auto& s=*impl_;try{apply_columns_check(columns,ld,count);
        std::size_t backend_bytes=0;hh_cpu_apply_columns_workspace_size<T>(s.handle,count,&backend_bytes);
        ddla::vector<char> backend_work(s.handle);detail::agree_local(s.world,"CPU Householder preparation",[&]{backend_work.resize(backend_bytes);});
        apply_columns_run(columns,ld,count,backend_work.data(),backend_bytes);
    }catch(...){s.close();throw;}
}
template<class T>
void Hb2stHouseholderCpu<T>::apply_tiles_check(R* tiles,int width,int columns)const{
    const auto& s=*impl_;detail::agree_local(s.world,"CPU Householder preparation",[&]{
        if(s.released||!s.reduced||!s.keep)throw std::logic_error("CPU Householder paired apply called in wrong state");
        if(columns<0||(width!=8&&width!=16&&width!=32&&width!=64)||(columns&&s.n&&!tiles))throw std::invalid_argument("CPU Householder tile strip is invalid");
        std::size_t count=0;if(!hhpair::layout_elements(s.n,columns,width,hhpair::planes<T>,count))throw std::overflow_error("CPU Householder tile strip overflow");checked_bytes(count,sizeof(R));
    });
}
template<class T>
void Hb2stHouseholderCpu<T>::apply_tiles(R* tiles,int width,int columns){
    try{apply_tiles_check(tiles,width,columns);apply_tiles_run(tiles,width,columns);}catch(...){impl_->close();throw;}
}

template<class T> void Hb2stHouseholderCpu<T>::release(){if(impl_)impl_->close();}
template<class T> std::size_t Hb2stHouseholderCpu<T>::local_history_elements()const{return static_cast<std::size_t>(impl_->local_elements);}
template<class T> std::size_t Hb2stHouseholderCpu<T>::total_history_elements()const{return impl_->keep?static_cast<std::size_t>(impl_->total_elements):0;}
template<class T> std::size_t Hb2stHouseholderCpu<T>::local_history_bytes()const{return checked_bytes(impl_->local_elements,sizeof(T));}
template<class T> int Hb2stHouseholderCpu<T>::panel_width()const{return factor_width;}
template<class T> int Hb2stHouseholderCpu<T>::order()const{return impl_->n;}
template<class T> int Hb2stHouseholderCpu<T>::threads()const{return impl_->threads;}
template<class T> bool Hb2stHouseholderCpu<T>::paired_replay()const{return impl_->options.stage4!=Hb2stHouseholderCpuOptions::Stage4::Wy;}
template<class T> typename Hb2stHouseholderCpu<T>::R Hb2stHouseholderCpu<T>::eigenvalue_scale()const{return impl_->scale;}

#define INSTANTIATE(T) template void hh_cpu_reduce_workspace_size<T>(const DdlaHandle_t&,int,int,std::size_t*); template void hh_cpu_apply_columns_workspace_size<T>(const DdlaHandle_t&,int,std::size_t*); template class Hb2stHouseholderCpu<T>; template bool hh_event_blas_cpu<T>(const DdlaHandle_t&,hh::BandView<T>,const hh::Event&,T*,T*,T*,typename real_type<T>::type&) noexcept
DDLA_FOR_EACH_TYPE(INSTANTIATE)
#undef INSTANTIATE
template void hh_gpu_apply_workspace_size<float>(int,int,bool,std::size_t*);
template void hh_gpu_apply_workspace_size<double>(int,int,bool,std::size_t*);
template void hh_gpu_apply_workspace_size<std::complex<float>>(int,int,bool,std::size_t*);
template void hh_gpu_apply_workspace_size<std::complex<double>>(int,int,bool,std::size_t*);
} }
