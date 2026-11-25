#include "api_grid_test_common.h"
#include "workspace_test_common.h"
#include "Eigensolver/one_stage/one_stage_workspace.h"
#include "Eigensolver/two_stage/pheevd_impl.h"
#include "Eigensolver/pheevd_driver.h"
#include "mpi_datatype.h"
#include "Eigensolver/selected_tridiagonal_core.h"
#include "matrix_transfer.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <type_traits>
using namespace api_grid_test;
namespace {
std::string transport = "default";
template<DdlaBackend B> void select_transport(DdlaHandle_t h) {
    if constexpr (B == DdlaBackend::GPU) {
        if (transport != "default") {
            const auto mode = transport == "ccl" ? DdlaCommunication::CCL :
                transport == "staged" ? DdlaCommunication::HostStagedMPI : DdlaCommunication::MPI;
            DDLA_CHECK(ddlaSetCommunication(h, mode));
            if (ddlaGetCommunication(h) != mode)
                throw std::runtime_error("requested GPU transport was not selected");
        }
    }
}
template<class S> bool no_alloc(const S& s,bool warm=false) {
    return !s.ddla_alloc && !s.ddla_free && !s.pinned_alloc && !s.raw_gpu_alloc &&
        (warm || !s.host_new) && !s.resource_setup;
}
class Arena {
    static constexpr std::size_t fence=256;
    DdlaHandle_t h_; std::size_t bb_,hb_;
    ddla::vector<char> backend_; std::vector<char> host_;
public:
    Arena(DdlaHandle_t h,std::size_t b,std::size_t s):h_(h),bb_(b),hb_(s),backend_(h,b+2*fence),host_(s+2*fence,char(0xa7)) {
        DDLA_CHECK(ddlaMemset(h,backend_.data(),0xa7,backend_.size()));
    }
    char* backend() { return backend_.data()+fence; }
    char* host() { return host_.data()+fence; }
    void verify() {
        std::array<unsigned char,2*fence> guards{};
        DDLA_CHECK(ddlaMemcpy(h_,guards.data(),backend_.data(),fence));
        DDLA_CHECK(ddlaMemcpy(h_,guards.data()+fence,backend()+bb_,fence));
        DDLA_CHECK(ddlaSynchronize(h_));
        bool okay=true;
        for (std::size_t i=0;i<fence;++i) okay=okay && guards[i]==0xa7 && guards[fence+i]==0xa7 &&
            static_cast<unsigned char>(host_[i])==0xa7 && static_cast<unsigned char>(host()[hb_+i])==0xa7;
        require_close(h_,"selected exact arena guards",okay?0:1,0);
    }
};
template<class T> T conjugate(T x) { if constexpr(std::is_arithmetic<T>::value) return x; else return std::conj(x); }
template<class T> T phase(int i) { if constexpr(std::is_arithmetic<T>::value) return T(1); else return T(std::cos(.19*i),std::sin(.19*i)); }
// Independent small dense oracle with known spectrum, also used by the existing nev fixture.
template<class T> T entry(int i,int j,int n,bool repeated,int repeat) {
    double norm=0,weighted=0;
    if (!repeated) {
        // Integer power sums are exact in double for these bounded fixtures;
        // avoid an O(n) loop inside every entry of the larger oracle.
        norm=double(n)*(n+1)*(2*n+1)/6;
        const double triangular=double(n)*(n+1)/2;
        weighted=triangular*triangular;
    } else for (int k=0;k<n;++k) {
        const double x=k+1,lambda=k/3+1; norm+=x*x; weighted+=x*x*lambda;
    }
    const double li=repeated?i/3+1:i+1,lj=repeated?j/3+1:j+1;
    double value=(i==j?li+double(repeat)/16:0)+(i+1.)*(j+1.)/norm*(4*weighted/norm-2*li-2*lj);
    return T(value)*phase<T>(i)*conjugate(phase<T>(j));
}
template<class T> std::complex<double> wide(T x) { return {double(x),0}; }
template<class R> std::complex<double> wide(std::complex<R> x) { return {double(x.real()),double(x.imag())}; }
template<class T> bool same(const std::vector<T>& a,const std::vector<T>& b) {
    return a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*sizeof(T));
}

// This mode calls the maintained host core directly, before any DDLA handle
// (including a GPU handle) is created. n=2 also exercises empty row owners at P=4.
void scaling_require(MPI_Comm comm,const char* label,bool okay) {
    int local=okay?1:0,all=0;
    MPI_Allreduce(&local,&all,1,MPI_INT,MPI_MIN,comm);
    if (!all) {
        int rank=0; MPI_Comm_rank(comm,&rank);
        if (!rank) std::cerr<<"SELECTED_SCALING_FAIL "<<label<<std::endl;
        MPI_Abort(comm,1);
    }
}
// Distinct singleton blocks need no reorthogonalization. Each additional
// vector therefore contributes exactly one certificate dot reduction.
template<class R> void certificate_reductions(MPI_Comm comm) {
    namespace core=ddla::spectral_impl::selected;
    constexpr int n=32;
    int rank=0,ranks=1;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&ranks);
    int ldy=std::max(1,detail::RowShare(n,ranks).count(rank));
    std::vector<R> d(n),e(n-1,R(0)),w(n),Y(std::size_t(ldy)*n);
    for (int i=0;i<n;++i) d[i]=R(i+1);
    const auto plan=core::check<R>(comm,n,n,d.data(),e.data(),w.data(),Y.data(),ldy);
    std::vector<unsigned char> work(plan.bytes);
    auto status=core::run<R>(plan,d.data(),e.data(),w.data(),Y.data(),work.data());
    scaling_require(comm,"one batched certificate reduction for each later vector",
        status.code==core::Code::success && status.dot_reductions==n-1);
    if (!rank) std::cout<<"SELECTED_CERTIFICATE_REDUCTIONS nev="<<n<<" count="<<status.dot_reductions<<std::endl;
}
template<class R>
void scaling_case(MPI_Comm comm,bool offdiagonal,int sign,int alias,bool exact,bool baseline_negative) {
    namespace core=ddla::spectral_impl::selected;
    constexpr int n=2,nev=1,offset=3;
    const R tiny=std::numeric_limits<R>::denorm_min();
    scaling_require(comm,"IEEE gradual underflow required",std::numeric_limits<R>::is_iec559 &&
        tiny>R(0) && R(3)*tiny!=R(2)*tiny);
    std::vector<R> d(8,R(-919)),e(8,R(-818)),values(8,R(-313));
    d[offset]=R(1.5); d[offset+1]=R(sign*(exact || offdiagonal?4:3))*tiny;
    e[offset]=offdiagonal?R(sign*(exact?4:3))*tiny:R(0);
    R* w=alias==1?d.data()+offset:alias==2?e.data()+offset:values.data()+offset;
    int rank=0,ranks=1; MPI_Comm_rank(comm,&rank); MPI_Comm_size(comm,&ranks);
    const detail::RowShare rows(n,ranks);
    const int ldy=std::max(1,rows.count(rank))+2;
    std::vector<R> Y(std::size_t(ldy)+2*offset,R(-717));
    const auto d0=d,e0=e,v0=values,y0=Y;
    if (!rank) std::cout<<"SELECTED_SCALING_BEGIN scalar_bytes="<<sizeof(R)<<" ranks="<<ranks
        <<" source="<<(offdiagonal?"e":"d")<<" sign="<<sign<<" alias="<<alias
        <<" exact="<<exact<<" baseline_negative="<<baseline_negative<<std::endl;
    const auto plan=core::check<R>(comm,n,nev,d.data()+offset,e.data()+offset,w,Y.data()+offset,ldy);
    scaling_require(comm,"check preserves every caller byte",same(d,d0)&&same(e,e0)&&same(values,v0)&&same(Y,y0));
    // Core regions are 64-byte multiples; retain exact-size work plus canaries.
    const std::uint64_t fence=UINT64_C(0xa7a7a7a7a7a7a7a7);
    std::vector<std::uint64_t> work((plan.bytes+7)/8+16,fence);
    const auto status=core::run<R>(plan,d.data()+offset,e.data()+offset,w,Y.data()+offset,work.data()+8);
    int low=int(status.code),high=low;
    MPI_Allreduce(MPI_IN_PLACE,&low,1,MPI_INT,MPI_MIN,comm);
    MPI_Allreduce(MPI_IN_PLACE,&high,1,MPI_INT,MPI_MAX,comm);
    scaling_require(comm,"agreed code and exponent",low==high && status.scale_exponent==1);
    bool guards=true;
    for (int i=0;i<8;++i) guards=guards && work[i]==fence && work[work.size()-1-i]==fence;
    scaling_require(comm,"scratch guards",guards);
    if (baseline_negative) {
        // Independent original-input check: the split eigenvalue is exactly
        // 3*denorm_min, not a reference formed by repeating the core's scale.
        scaling_require(comm,"baseline accepts and changes nonzero original eigenvalue",
            status.code==core::Code::success && w[0]==R(4)*tiny &&
            w[0]!=d0[offset+1] && w[0]!=R(0) && w[1]==R(1.5));
    } else if (!exact) {
        scaling_require(comm,"loss refused before any caller write",status.code==core::Code::scaling_loss &&
            same(d,d0)&&same(e,e0)&&same(values,v0)&&same(Y,y0));
    } else {
        // For these split/negligibly coupled 2x2 controls, correctly rounded
        // all-n eigenvalues are {sign*4*denorm_min,1.5}; no rescaled oracle.
        scaling_require(comm,"reversible input succeeds with original all-n values",
            status.code==core::Code::success && w[0]==d0[offset+1] && w[1]==d0[offset]);
        const double tol=16*double(std::numeric_limits<R>::epsilon());
        bool vector_ok=true;
        for (int i=0;i<plan.rows;++i) {
            const R x=Y[offset+i];
            vector_ok=vector_ok && std::isfinite(x) &&
                (plan.row0+i==1?std::abs(std::abs(double(x))-1)<=tol:std::abs(double(x))<=tol);
        }
        scaling_require(comm,"known lowest eigenvector",vector_ok);
    }
    if (exact || baseline_negative) {
        // Only the selected w span and locally owned Y rows may change.
        auto expected_d=d0,expected_e=e0,expected_v=v0;
        R* expected=alias==1?expected_d.data()+offset:alias==2?expected_e.data()+offset:expected_v.data()+offset;
        std::copy_n(w,n,expected);
        auto expected_y=y0;
        std::copy_n(Y.data()+offset,plan.rows,expected_y.data()+offset);
        scaling_require(comm,"success input and output padding guards",same(d,expected_d)&&same(e,expected_e)&&
            same(values,expected_v)&&same(Y,expected_y));
    }
    if (!rank) std::cout<<"SELECTED_SCALING_CASE scalar_bytes="<<sizeof(R)<<" ranks="<<ranks
        <<" source="<<(offdiagonal?"e":"d")<<" sign="<<sign<<" alias="<<alias
        <<" exact="<<exact<<" baseline_negative="<<baseline_negative<<" code="<<int(status.code)<<std::endl;
}
template<class R> void scaling_cases(MPI_Comm comm,bool baseline_negative) {
    if (baseline_negative) scaling_case<R>(comm,false,1,0,false,true);
    else for (bool exact:{false,true}) for (bool offdiagonal:{false,true})
        for (int sign:{-1,1}) for (int alias:{0,1,2})
            scaling_case<R>(comm,offdiagonal,sign,alias,exact,false);
}

template<int Route,DdlaBackend B,class T>
void solve_case(DdlaHandle_t h,spectral_impl::Phb2stTopology& topology,int n,int nev,char uplo,
    const char* policy,bool parent,int alias,int engine=0,int stage4=0,bool repeated=false,
    DdlaBackend reduction=DdlaBackend::CPU,char job='V',int nb=4) {
    using R=typename real_type<T>::type;
    if (!h->myid) std::cout<<"SELECTED_BEGIN route="<<Route<<" backend="<<int(B)<<" scalar_bytes="<<sizeof(T)
        <<" n="<<n<<" nev="<<nev<<" job="<<job<<" uplo="<<uplo<<" policy="<<(policy?policy:"defaultdc")
        <<" engine="<<engine<<" stage4="<<stage4<<" reduction="<<int(reduction)<<" alias="<<alias
        <<" parent="<<parent<<" repeated="<<repeated<<std::endl;
    int da[DDLA_DLEN_],dz[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(da,h,n+2,n+3,nb,nb,0,0));
    DDLA_CHECK(ddlaDescInit(dz,h,n+(parent?2:0),parent?n+3:nev,nb,nb,0,parent?h->npcols_-1:0));
    da[DDLA_LLD_]+=3; dz[DDLA_LLD_]+=2;
    if (alias) std::copy_n(da,DDLA_DLEN_,dz);
    const std::size_t shift=alias==2?1:0;
    DeviceBuffer<T> A(h,local_size(h,da)+shift),Zstorage(h,local_size(h,dz));
    T* Z=alias?A.ptr+shift:Zstorage.ptr;
    std::vector<T> initA(A.count,T(-919)),initZ(Zstorage.count,T(-717));
    upload(h,A.ptr,initA); upload(h,Zstorage.ptr,initZ);
    std::vector<R> w(std::size_t(n)+8,R(-313));
    if (policy) setenv("DDLA_EIG_TRIDIAGONAL",policy,1); else unsetenv("DDLA_EIG_TRIDIAGONAL");
    setenv("DDLA_EIG_CPU_ACCUM",engine==1?"1":"0",1);
    setenv("DDLA_QR_PANEL","gather",1);
    auto p=[&] {
        if constexpr(Route==1) return spectral_impl::one_stage_check<T>(h,B,job,uplo,n,A.ptr,da,w.data()+4,
            nev?Z:nullptr,nev?dz:nullptr,nev);
        else return spectral_impl::pheevd_check<T>(h,B,job,uplo,n,A.ptr,da,w.data()+4,
            nev?Z:nullptr,nev?dz:nullptr,topology,engine>=2?'H':'G',engine>=2?&stage4:nullptr,reduction,nev);
    }();
    if (!h->myid) {
        if constexpr(Route==1) std::cout<<"SELECTED_CHECKED route=1 backend="<<int(p.backend)<<" n="<<p.n<<" nev="<<p.nev
            <<" nb="<<p.a[DDLA_NB_]<<" job="<<p.job<<" selected="<<bool(p.selected_plan.n)
            <<" host_budget="<<h->host_threads<<std::endl;
        else {
            const bool hh=p.engine==spectral_impl::PheevdEngine::Householder;
            std::cout<<"SELECTED_CHECKED route=2 backend="<<int(p.backend)<<" n="<<p.n<<" nev="<<p.nev<<" nb="<<p.nb
                <<" band="<<p.band<<" job="<<p.job<<" selected="<<bool(p.selected_plan.n)<<" engine="<<int(p.engine)
                <<" workers="<<p.workers<<" hh_active="<<hh
                <<" replay_cpu_active="<<(p.engine==spectral_impl::PheevdEngine::ReplayCpu)
                <<" accum_active="<<(p.engine==spectral_impl::PheevdEngine::AccumHost)<<" replay_threads="<<p.cpu_policy.threads<<" replay_tile="<<p.cpu_policy.tile
                <<" replay_waves="<<p.cpu_policy.waves<<" replay_wave_mode="<<p.cpu_policy.wave_mode
                <<" replay_row_block="<<p.cpu_policy.row_block<<" replay_level_block="<<p.cpu_policy.level_block
                <<" hh_stage2="<<(hh?int(p.householder.options.stage2):-1)<<" hh_stage4="<<(hh?int(p.householder.options.stage4):-1)
                <<" hh_threads="<<(hh?p.householder.threads:0)<<" hh_chase_threads="<<(hh?p.householder.chase_threads:0)
                <<" device_chase="<<p.chase.device_chase<<" spatial_chase="<<p.chase.spatial_enabled
                <<" stage1_backend="<<int(p.reduction.backend)<<" stage1_inplace="<<p.reduction.inplace
                <<" resolved_replay_tile="<<p.cpu_replay.tile<<" replay_local_columns="<<p.cpu_replay.columns.ncs
                <<" hh_row_tile="<<p.hh_rows.tile_width<<" hh_local_columns="<<p.hh_rows.map.ncs
                <<" finish_panel="<<p.finish.panel<<" finish_contraction="<<p.finish.kp;
#if DDLA_HAS_GPU
            if constexpr(B==DdlaBackend::GPU) std::cout<<" gpu_householder="<<p.gpu_replay.householder
                <<" gpu_householder_wy="<<p.gpu_replay.householder_wy<<" gpu_local_columns="<<p.gpu_replay.nc
                <<" gpu_replay_width="<<p.gpu_replay.width<<" gpu_replay_waves="<<p.gpu_replay.use_waves;
#endif
            std::cout<<std::endl;
        }
    }
    const bool selected=policy && !std::strcmp(policy,"selected") && job=='V' && nev>0 && nev<n;
    require_close(h,"selected policy eligibility",(p.selected_plan.n!=0)==selected?0:1,0);
    if (selected) {
        const detail::RowShare rows(n,h->nprocs);
        require_close(h,"selected pure geometry matches RowShare",
            p.selected_plan.rows==rows.count(h->myid) && p.selected_plan.row0==rows.lo(h->myid) &&
            p.selected_plan.ldy==std::max(1,rows.count(h->myid)) && p.selected_plan.nev==nev?0:1,0);
    }
    auto size=[&](std::size_t* b,std::size_t* s) {
        if constexpr(Route==1) spectral_impl::one_stage_workspace_size<T>(h,p,b,s);
        else spectral_impl::pheevd_workspace_size<T>(h,p,A.ptr,b,s);
    };
    auto unchanged=[&] {
        return same(download(h,A.ptr,A.count),initA) && same(download(h,Zstorage.ptr,Zstorage.count),initZ) &&
            std::all_of(w.begin(),w.end(),[](R x){return x==R(-313);});
    };
    // Full allocated spans are initialized before check/query; snapshots/downloads
    // are outside trace. Both root-only and all-rank queries must be pure.
    bool pure=true; std::uint64_t root_queries=0,all_queries=0;
    MPI_Barrier(h->comm);
    if (!h->myid) {
        workspace_test::Scope trace(true); std::size_t b=0,s=0; size(&b,&s); const auto t=trace.finish();
        pure=no_alloc(t) && !t.copies && !t.launches && !t.syncs && !t.collectives;
        root_queries=t.vendor_query;
    }
    MPI_Barrier(h->comm);
    require_close(h,"selected root-only query is pure",pure?0:1,0);
    require_close(h,"selected root-only query/check leaves every caller byte unchanged",unchanged()?0:1,0);
    std::size_t backend_bytes=0,host_bytes=0;
    {
        workspace_test::Scope trace(true); size(&backend_bytes,&host_bytes); const auto t=trace.finish();
        pure=no_alloc(t) && !t.copies && !t.launches && !t.syncs && !t.collectives;
        all_queries=t.vendor_query;
    }
    require_close(h,"selected all-rank query is pure",pure?0:1,0);
    require_close(h,"selected all-rank query leaves every caller byte unchanged",unchanged()?0:1,0);
    Arena arena(h,backend_bytes,host_bytes);
    auto resources=[&] {
        if constexpr(Route==1) return spectral_impl::OneStageResources(h,p);
        else return spectral_impl::PheevdResources<T>(h,p);
    }();
    if (selected) require_close(h,"selected route skips DC resource preparation",resources.vectors?1:0,0);
    // Every raw repeat must consume the frozen choice, not this invalid setting.
    setenv("DDLA_EIG_TRIDIAGONAL","invalid_after_query",1);
    setenv("DDLA_EIG_CPU_ACCUM",engine==1?"0":"1",1);
    setenv("DDLA_QR_PANEL","inplace",1);
    for (int repeat=0;repeat<2;++repeat) {
        auto original=make_local<T>(h,da,[&](int i,int j) {
            if (i>=n || j>=n) return T(-919);
            if (uplo=='U'?i>j:i<j) return T(std::numeric_limits<R>::quiet_NaN());
            return entry<T>(i,j,n,repeated,repeat);
        });
        original.resize(A.count,T(-919));
        upload(h,A.ptr,original); if (!alias) upload(h,Zstorage.ptr,initZ);
        std::fill(w.begin(),w.end(),R(-313)); int info=-1;
        workspace_test::Scope trace;
        if constexpr(Route==1) spectral_impl::one_stage_run<B,T>(h,p,resources,A.ptr,w.data()+4,
            nev?Z:nullptr,info,nullptr,arena.backend(),backend_bytes,arena.host(),host_bytes);
        else spectral_impl::pheevd_run<B,T>(h,p,resources,A.ptr,w.data()+4,nev?Z:nullptr,info,
            arena.backend(),backend_bytes,arena.host(),host_bytes);
        const auto t=trace.finish();
        const bool warm=B==DdlaBackend::GPU && ddlaGetCommunication(h)==DdlaCommunication::HostStagedMPI && repeat==0;
        require_close(h,"selected raw run has no algorithm allocation/query",no_alloc(t,warm) && !t.vendor_query?0:1,0);
        require_close(h,"selected raw info",info,0);
        const double eps=std::numeric_limits<R>::epsilon();
        double values=0;
        for (int j=0;j<n;++j) {
            const double expected=(repeated?j/3+1:j+1)+double(repeat)/16;
            values=std::max(values,workspace_test::finite_error(std::abs(double(w[4+j])-expected)/(eps*std::max(n,1))));
        }
        require_close(h,"selected all-n values match independent spectrum",values,100);
        bool wguard=true; for (int i=0;i<4;++i) wguard=wguard && w[i]==R(-313) && w[std::size_t(n)+4+i]==R(-313);
        require_close(h,"selected eigenvalue guards",wguard?0:1,0);
        const auto afterA=download(h,A.ptr,A.count);
        bool apad=true;
        for (int j=0;j<ddla_test::n_loc(h,da);++j) for (int i=0;i<da[DDLA_LLD_];++i) {
            const int gi=indx_l2g_r(da,h,i),gj=indx_l2g_c(da,h,j); const auto at=std::size_t(j)*da[DDLA_LLD_]+i;
            const bool aactive=i<ddla_test::m_loc(h,da) && gi<n && gj<n;
            bool zactive=false;
            if (alias && at>=shift) {
                const auto zat=at-shift; const int zi=int(zat%dz[DDLA_LLD_]),zj=int(zat/dz[DDLA_LLD_]);
                zactive=zj<ddla_test::n_loc(h,dz) && zi<ddla_test::m_loc(h,dz) &&
                    indx_l2g_r(dz,h,zi)<n && indx_l2g_c(dz,h,zj)<nev && job=='V';
            }
            if (!aactive && !zactive) apad=apad && !std::memcmp(&afterA[at],&original[at],sizeof(T));
        }
        require_close(h,"selected input parent and padding",apad?0:1,0);
        double residual=0,orthogonality=0;
        if (job=='V' && nev) {
            const auto result=download(h,Z,local_size(h,dz));
            std::vector<T> Q(std::size_t(n)*nev,T(0)); bool zpad=true;
            for (int j=0;j<ddla_test::n_loc(h,dz);++j) for (int i=0;i<dz[DDLA_LLD_];++i) {
                const int gi=indx_l2g_r(dz,h,i),gj=indx_l2g_c(dz,h,j); const auto at=std::size_t(j)*dz[DDLA_LLD_]+i;
                if (i<ddla_test::m_loc(h,dz) && gi<n && gj<nev) Q[std::size_t(gj)*n+gi]=result[at];
                else {
                    const bool insideA=alias && at+shift<original.size() &&
                        int((at+shift)%da[DDLA_LLD_])<ddla_test::m_loc(h,da) &&
                        indx_l2g_r(da,h,int((at+shift)%da[DDLA_LLD_]))<n &&
                        indx_l2g_c(da,h,int((at+shift)/da[DDLA_LLD_]))<n;
                    if (!insideA) { const T expected=alias?original[at+shift]:initZ[at]; zpad=zpad && !std::memcmp(&result[at],&expected,sizeof(T)); }
                }
            }
            require_close(h,"selected rectangular output and padding",zpad?0:1,0);
            // Bounded test-only oracle n<=257; production never gathers a basis.
            if (!Q.empty()) MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,Q.data(),int(Q.size()),detail::mpi_datatype<T>(),MPI_SUM,h->comm));
            for (int j=0;j<nev;++j) for (int i=0;i<n;++i) {
                std::complex<double> r=-double(w[4+j])*wide(Q[std::size_t(j)*n+i]);
                for (int k=0;k<n;++k) r+=wide(entry<T>(i,k,n,repeated,repeat))*wide(Q[std::size_t(j)*n+k]);
                residual=std::max(residual,workspace_test::finite_error(std::abs(r)/(eps*std::max(n,1)*std::max(n,1))));
            }
            for (int j=0;j<nev;++j) for (int k=0;k<nev;++k) {
                std::complex<double> dot=0;
                for (int i=0;i<n;++i) dot+=std::conj(wide(Q[std::size_t(j)*n+i]))*wide(Q[std::size_t(k)*n+i]);
                orthogonality=std::max(orthogonality,workspace_test::finite_error(std::abs(dot-std::complex<double>(j==k?1:0))/(eps*std::max(n,1))));
            }
            require_close(h,"selected dense residual",residual,100);
            require_close(h,"selected orthogonality",orthogonality,100);
        } else if (!alias) require_close(h,"selected values-only output unchanged",same(download(h,Zstorage.ptr,Zstorage.count),initZ)?0:1,0);
        arena.verify();
        if (!h->myid) std::cout<<"SELECTED_INTEGRATION route="<<Route<<" backend="<<int(B)<<" scalar_bytes="<<sizeof(T)
            <<" n="<<n<<" nev="<<nev<<" job="<<job<<" uplo="<<uplo<<" policy="<<(policy?policy:"defaultdc")
            <<" selected="<<selected<<" engine="<<engine<<" stage4="<<stage4<<" reduction="<<int(reduction)
            <<" alias="<<alias<<" parent="<<parent<<" repeated="<<repeated<<" repeat="<<repeat
            <<" backend_bytes="<<backend_bytes<<" host_bytes="<<host_bytes<<" values="<<values
            <<" residual="<<residual<<" orth="<<orthogonality<<" root_vendor_queries="<<root_queries<<" rank0_vendor_queries="<<all_queries<<" trace="<<workspace_test::available()<<std::endl;
    }
    if constexpr(Route==2) resources.release();
    unsetenv("DDLA_EIG_TRIDIAGONAL"); unsetenv("DDLA_EIG_CPU_ACCUM"); unsetenv("DDLA_QR_PANEL");
}

template<DdlaBackend B> void policy_errors(DdlaHandle_t h,spectral_impl::Phb2stTopology& topology) {
    int da[DDLA_DLEN_],dz[DDLA_DLEN_]; DDLA_CHECK(ddlaDescInit(da,h,17,17,4,4,0,0)); DDLA_CHECK(ddlaDescInit(dz,h,17,5,4,4,0,0));
    DeviceBuffer<double> A(h,local_size(h,da)),Z(h,local_size(h,dz)); double w[17]{};
    for (int route:{1,2}) for (int test:{0,1}) {
        if (test && h->nprocs==1) continue;
        setenv("DDLA_EIG_TRIDIAGONAL",test?(h->myid?"selected":"dc"):"invalid",1);
        bool caught=false;
        try {
            if (route==1) (void)spectral_impl::one_stage_check<double>(h,B,'V','U',17,A.ptr,da,w,Z.ptr,dz,5);
            else (void)spectral_impl::pheevd_check<double>(h,B,'V','U',17,A.ptr,da,w,Z.ptr,dz,topology,'G',nullptr,DdlaBackend::CPU,5);
        } catch (const std::exception&) { caught=true; }
        require_close(h,"selected invalid or inconsistent policy fails collectively",caught?0:1,0);
    }
    unsetenv("DDLA_EIG_TRIDIAGONAL");
}

// Direct compact panel coverage includes transpose, whose old callers retain
// their default k-wide arena. This test's prepared bound is exactly sufficient.
template<DdlaBackend B,class T> void compact_scatter(DdlaHandle_t h,bool transpose,int n=17,int k=3) {
    using R=typename real_type<T>::type;
    const detail::RowShare share(n,h->nprocs);
    const int nr=share.count(h->myid),ld=std::max(1,nr),m=transpose?k:n,cols=transpose?n:k;
    int desc[DDLA_DLEN_]; DDLA_CHECK(ddlaDescInit(desc,h,m+2,cols+2,4,4,0,0)); desc[DDLA_LLD_]+=2;
    std::vector<R> rows(std::size_t(ld)*k,R(-515));
    for (int j=0;j<k;++j) for (int i=0;i<nr;++i) rows[i+std::size_t(j)*ld]=R(share.lo(h->myid)+i+10*j);
    DeviceBuffer<R> Y(h,rows.size()); upload(h,Y.ptr,rows);
    DeviceBuffer<T> Z(h,local_size(h,desc)); std::vector<T> before(Z.count,T(-717)); upload(h,Z.ptr,before);
    std::size_t bb=0,hb=0,oldb=0,oldh=0;
    matrix_impl::scatter_rows_workspace_size<T>(h,n,&bb,&hb,k);
    matrix_impl::scatter_rows_workspace_size<T>(h,n,&oldb,&oldh);
    require_close(h,"compact transfer query removes unrequested columns",h->nprocs>1?(bb<oldb?0:1):(bb==oldb?0:1),0);
    Arena arena(h,bb,hb);
    workspace_test::Scope trace;
    matrix_impl::scatter_rows_run<B,T>(h,n,Y.ptr,ld,transpose,m,cols,Z.ptr,desc,
        arena.backend(),bb,arena.host(),hb,k);
    const auto t=trace.finish();
    const bool warm=B==DdlaBackend::GPU && ddlaGetCommunication(h)==DdlaCommunication::HostStagedMPI;
    require_close(h,"compact transfer uses borrowed arena",no_alloc(t,warm) && !t.vendor_query?0:1,0);
    const auto after=download(h,Z.ptr,Z.count); bool okay=true;
    for (int j=0;j<ddla_test::n_loc(h,desc);++j) for (int i=0;i<desc[DDLA_LLD_];++i) {
        const int gi=indx_l2g_r(desc,h,i),gj=indx_l2g_c(desc,h,j); const auto at=std::size_t(j)*desc[DDLA_LLD_]+i;
        const T expected=i<ddla_test::m_loc(h,desc) && gi<m && gj<cols?T(transpose?gj+10*gi:gi+10*gj):before[at];
        okay=okay && !std::memcmp(&after[at],&expected,sizeof(T));
    }
    require_close(h,"compact transpose/nontranspose transfer and padding",okay?0:1,0); arena.verify();
    require_close(h,"compact transfer source remains immutable",same(download(h,Y.ptr,Y.count),rows)?0:1,0);
    if (!h->myid) std::cout<<"SELECTED_TRANSFER backend="<<int(B)<<" scalar_bytes="<<sizeof(T)<<" transpose="<<transpose
        <<" n="<<n<<" nev="<<k<<" max_row_share="<<share.count(0)<<" row_panel_cap=512"
        <<" compact_bytes="<<bb<<" old_bytes="<<oldb<<std::endl;
}

template<DdlaBackend B> void query_contracts(DdlaHandle_t h,spectral_impl::Phb2stTopology& topology) {
    constexpr int n=257,k=5;
    const int ld=std::max(1,detail::RowShare(n,h->nprocs).count(h->myid));
    setenv("DDLA_EIG_TRIDIAGONAL","selected",1);
    const auto selected=spectral_impl::selected_rows_policy<double>(h,'V',n,k,ld);
    auto dc=spectral_impl::pstedc_policy<double>(h,n, 128); dc.ldy=ld;
    std::size_t b=0,s=0;
    {
        workspace_test::Scope trace(true); spectral_impl::tridiagonal_rows_workspace_size<double>(h,selected,dc,&b,&s);
        const auto t=trace.finish();
        require_close(h,"selected row query is arithmetic only",no_alloc(t)&&!t.vendor_query&&!t.copies&&!t.launches&&!t.syncs&&!t.collectives?0:1,0);
    }
    const auto linear=detail::padded_bytes<char>(selected.bytes),slab=detail::padded_bytes<double>(std::size_t(ld)*k);
    require_close(h,"selected domains contain only compact slab plus linear core",
        B==DdlaBackend::CPU?(b==linear&&s==0?0:1):(b==0&&s==slab+linear?0:1),0);
    // Cheby's projected solve asks for every q vector; the same private driver
    // must retain D&C even while the environment explicitly requests selected.
    int desc[DDLA_DLEN_]; DDLA_CHECK(ddlaDescInit(desc,h,17,17,4,4,0,0));
    DeviceBuffer<double> A(h,local_size(h,desc)),Z(h,local_size(h,desc)); double w[17]{};
    for (auto route:{detail::PheevdRoute::OneStage,detail::PheevdRoute::TwoStage}) {
        const auto full=spectral_impl::pheevd_driver_check<B,double>(h,route,'V','U',17,17,A.ptr,desc,w,Z.ptr,desc,&topology);
        const bool dc_only=full.one?!full.one->selected_plan.n:!full.two->selected_plan.n;
        require_close(h,"full projected driver remains D&C",dc_only?0:1,0);
    }
    // Exercise the selected failure boundary directly: a representable nonzero
    // lost to canonical scaling must fail collectively before any row consumer.
    const int nr=detail::RowShare(2,h->nprocs).count(h->myid),ldy=std::max(1,nr);
    const auto loss=spectral_impl::selected_rows_policy<double>(h,'V',2,1,ldy);
    double d[2]={std::numeric_limits<double>::max()/16,std::numeric_limits<double>::min()*8},e[1]={0},out[2]={-313,-313};
    const std::array<double,2> before{{d[0],d[1]}};
    DeviceBuffer<double> Y(h,std::size_t(ldy)); std::vector<double> untouched(Y.count,-717); upload(h,Y.ptr,untouched);
    spectral_impl::tridiagonal_rows_workspace_size<double>(h,loss,dc,&b,&s); Arena arena(h,b,s);
    bool caught=false;
    try { spectral_impl::tridiagonal_rows_run<B,double>(h,loss,dc,nullptr,d,e,out,Y.ptr,arena.backend(),b,arena.host(),s); }
    catch (const std::runtime_error&) { caught=true; }
    require_close(h,"selected numerical failure is collective before backtransform",caught?0:1,0);
    require_close(h,"selected failed upload/output stays untouched",
        same(download(h,Y.ptr,Y.count),untouched)&&out[0]==-313&&out[1]==-313&&d[0]==before[0]&&d[1]==before[1]?0:1,0);
    arena.verify(); unsetenv("DDLA_EIG_TRIDIAGONAL");
    if (!h->myid) std::cout<<"SELECTED_QUERY backend="<<int(B)<<" n="<<n<<" nev="<<k<<" slab_bytes="<<slab
        <<" linear_core_bytes="<<linear<<" zero_owner_lld_padding=explicit"<<std::endl;
}

template<DdlaBackend B> void generalized_failure_order(DdlaHandle_t h) {
    constexpr int n=7;
    int d[DDLA_DLEN_]; DDLA_CHECK(ddlaDescInit(d,h,n,n,4,4,0,0));
    DeviceBuffer<double> A(h,local_size(h,d)),C(h,local_size(h,d)),Z(h,local_size(h,d));
    auto aa=make_local<double>(h,d,[](int i,int j){return i==j?double(i+2):.01;});
    auto bb=make_local<double>(h,d,[](int i,int j){return i==j?(i==2?-1.:1.):0.;});
    aa.resize(A.count,-919); bb.resize(C.count,-919); std::vector<double> zz(Z.count,-717),w(n,-313);
    upload(h,A.ptr,aa); upload(h,C.ptr,bb); upload(h,Z.ptr,zz);
    setenv("DDLA_EIG_TRIDIAGONAL","invalid-before-factor-failure",1);
    int info=-1; phegvd<double>(h,'V','U',n,3,A.ptr,d,C.ptr,d,w.data(),Z.ptr,d,info);
    require_close(h,"generalized non-PD failure precedes selected preparation",
        info==n+3 && same(download(h,A.ptr,A.count),aa) && same(download(h,Z.ptr,Z.count),zz) &&
        std::all_of(w.begin(),w.end(),[](double x){return x==-313;})?0:1,0);
    phegvd<double>(h,'V','L',0,0,A.ptr,d,C.ptr,d,nullptr,nullptr,nullptr,info);
    require_close(h,"generalized empty return precedes selected policy",info,0);
    unsetenv("DDLA_EIG_TRIDIAGONAL");
}

template<DdlaBackend B,class T> void cases(DdlaHandle_t h,spectral_impl::Phb2stTopology& topology) {
    for (const char* policy:{static_cast<const char*>(nullptr),"selected"}) for (char uplo:{'U','L'}) {
#define BASIC(R) \
        for(int nev:{0,1,5,17}) solve_case<R,B,T>(h,topology,17,nev,uplo,policy,false,0); \
        solve_case<R,B,T>(h,topology,17,5,uplo,policy,true,0); \
        solve_case<R,B,T>(h,topology,17,5,uplo,policy,true,1); \
        solve_case<R,B,T>(h,topology,17,5,uplo,policy,true,2); \
        solve_case<R,B,T>(h,topology,17,5,uplo,policy,false,0,0,0,true); \
        solve_case<R,B,T>(h,topology,2,1,uplo,policy,false,0); \
        solve_case<R,B,T>(h,topology,0,0,uplo,policy,false,0)
        BASIC(1); BASIC(2);
#undef BASIC
    }
    for (const char* policy:{"dc","selected"}) {
        if constexpr(B==DdlaBackend::CPU) solve_case<2,B,T>(h,topology,17,5,'L',policy,true,0,1);
        if (topology.groups()==1) {
            const int methods=B==DdlaBackend::GPU?2:(std::is_arithmetic<T>::value?6:4);
            for (int stage=0;stage<methods;++stage) solve_case<2,B,T>(h,topology,17,5,'U',policy,true,0,2,stage);
#if DDLA_HAS_GPU
            if constexpr(B==DdlaBackend::GPU) {
                solve_case<2,B,T>(h,topology,17,5,'L',policy,true,0,0,0,false,DdlaBackend::GPU);
                for (int stage=0;stage<2;++stage) solve_case<2,B,T>(h,topology,17,5,'L',policy,true,0,2,stage,false,DdlaBackend::GPU);
            }
#endif
        }
    }
}
template<DdlaBackend B,class T> void large_cases(DdlaHandle_t h,spectral_impl::Phb2stTopology& topology) {
    // Bounded alternative to a2051 dense solve:129 vectors exercise complete
    // stage4 tiles plus tails;5003-row compact transfers exercise >1 row panel.
    for (const char* policy:{"dc","selected"}) {
        solve_case<1,B,T>(h,topology,257,129,'U',policy,false,0,0,0,false,DdlaBackend::CPU,'V',64);
        solve_case<2,B,T>(h,topology,257,129,'L',policy,false,0,0,0,false,DdlaBackend::CPU,'V',64);
        if constexpr(B==DdlaBackend::CPU)
            solve_case<2,B,T>(h,topology,257,129,'L',policy,false,0,1,0,false,DdlaBackend::CPU,'V',64);
        if (topology.groups()==1) {
            const int methods=B==DdlaBackend::GPU?2:(std::is_arithmetic<T>::value?6:4);
            for (int stage=0;stage<methods;++stage)
                solve_case<2,B,T>(h,topology,257,129,'U',policy,false,0,2,stage,false,DdlaBackend::CPU,'V',64);
#if DDLA_HAS_GPU
            if constexpr(B==DdlaBackend::GPU) {
                solve_case<2,B,T>(h,topology,257,129,'L',policy,false,0,0,0,false,DdlaBackend::GPU,'V',64);
                for (int stage=0;stage<2;++stage)
                    solve_case<2,B,T>(h,topology,257,129,'L',policy,false,0,2,stage,false,DdlaBackend::GPU,'V',64);
            }
#endif
        }
    }
    compact_scatter<B,T>(h,false,5003,17); compact_scatter<B,T>(h,true,5003,17);
}

template<DdlaBackend B> void run(MPI_Comm comm,int pr,int pc,char major,bool large) {
    DdlaHandle_t h=nullptr; ddlaInit(h,B);
    select_transport<B>(h);
    ddlaSet(h,comm,pr,pc,major);
    if (!h->myid) std::cout << "SELECTED_TRANSPORT backend=" << int(B)
        << " requested=" << transport << " actual=" << int(ddlaGetCommunication(h)) << std::endl;
    auto topology=spectral_impl::phb2st_topology_prepare(h);
    if (large) {
        large_cases<B,double>(h,*topology); large_cases<B,std::complex<double>>(h,*topology);
    } else {
    cases<B,float>(h,*topology); cases<B,double>(h,*topology);
    cases<B,std::complex<float>>(h,*topology); cases<B,std::complex<double>>(h,*topology);
    policy_errors<B>(h,*topology);
    query_contracts<B>(h,*topology);
    generalized_failure_order<B>(h);
    compact_scatter<B,double>(h,false); compact_scatter<B,double>(h,true);
    compact_scatter<B,std::complex<double>>(h,false); compact_scatter<B,std::complex<double>>(h,true);
    }
    topology->release(); ddlaDestroy(h);
}
}
int main(int argc,char** argv) {
    int provided=0; MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    if (provided<MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD,2);
    int ranks=1; MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    certificate_reductions<float>(MPI_COMM_WORLD);
    certificate_reductions<double>(MPI_COMM_WORLD);
    bool large=false,scaling_only=false,scaling_baseline_negative=false; int kept=1;
    for (int i=1;i<argc;++i) {
        if (!std::strcmp(argv[i],"--scaling-only")) scaling_only=true;
        else if (!std::strcmp(argv[i],"--scaling-baseline-negative")) scaling_baseline_negative=true;
        else if (!std::strcmp(argv[i],"--large")) large=true;
        else if (!std::strcmp(argv[i],"--transport")) {
            if (i+1>=argc) { std::cerr << "--transport requires ccl|staged|mpi" << std::endl; MPI_Abort(MPI_COMM_WORLD,2); }
            transport=argv[++i];
            if (transport!="ccl" && transport!="staged" && transport!="mpi") {
                std::cerr << "--transport requires ccl|staged|mpi" << std::endl; MPI_Abort(MPI_COMM_WORLD,2);
            }
        } else argv[kept++]=argv[i];
    }
    argc=kept;
    if (scaling_only || scaling_baseline_negative) {
        scaling_require(MPI_COMM_WORLD,"scaling mode requires 1/2/4 ranks and no extra options",
            (ranks==1 || ranks==2 || ranks==4) && argc==1 && !large && transport=="default" &&
            !(scaling_only && scaling_baseline_negative));
        scaling_cases<float>(MPI_COMM_WORLD,scaling_baseline_negative);
        scaling_cases<double>(MPI_COMM_WORLD,scaling_baseline_negative);
        int rank=0; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        if (!rank) std::cout<<(scaling_baseline_negative?"SELECTED_SCALING_BASELINE_LOSS_OBSERVED":
            "SELECTED_SCALING_PASS")<<std::endl;
        MPI_Finalize(); return 0;
    }
    TestOptions options; std::string error;
    if (!parse_options(argc,argv,ranks,options,error)) { std::cerr<<error<<std::endl; MPI_Abort(MPI_COMM_WORLD,2); }
    for (auto grid:options.grids) for (char major:{'R','C'}) {
        run<DdlaBackend::CPU>(MPI_COMM_WORLD,grid.first,grid.second,major,large);
#if DDLA_HAS_GPU
        run<DdlaBackend::GPU>(MPI_COMM_WORLD,grid.first,grid.second,major,large);
#endif
    }
    int rank=0; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if (!rank) {
        std::cout<<"SELECTED_EIGENSOLVER_MODE mode="<<(large?"large":"default")<<std::endl;
        std::cout<<"SELECTED_EIGENSOLVER_INTEGRATION_PASS"<<std::endl;
    }
    MPI_Finalize(); return 0;
}
