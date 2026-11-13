#include "api_grid_test_common.h"
#include "Eigensolver/two_stage/pheevd_impl.h"
#include "Eigensolver/two_stage/hb2st_householder_kernels.h"
#include "row_share.h"
#include "eig_test_common.h"
#include "mpi_datatype.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>

using namespace api_grid_test;
namespace {
namespace hh=ddla::detail::hh;
template<class T> T value(double re,double im) {
    using R=typename real_type<T>::type;
    if constexpr(std::is_same<T,R>::value)return T(re);
    else return T(R(re),R(im));
}
template<class T> std::complex<long double> wide(T x) {return {static_cast<long double>(x),0};}
template<class R> std::complex<long double> wide(std::complex<R> x) {return {x.real(),x.imag()};}
template<class T> T band_value(int i,int j,int b,typename real_type<T>::type magnitude) {
    if(std::abs(i-j)>b)return T(0);
    if(i<j)return hh::adj(band_value<T>(j,i,b,magnitude));
    const double re=i==j?0.25+0.01*(i%13):std::sin((i+1)*0.719+(j+1)*0.131);
    const double im=i==j?0:std::cos((i+1)*0.293-(j+1)*0.817);
    return magnitude*value<T>(re,im);
}

template<class T>
void check_band(const DdlaHandle_t& h,int n,int b,bool shifted,int range_kind) {
    using R=typename real_type<T>::type;
    const int P=ddlaGetSize(h),rank=ddlaGetRank(h);
    int npr=0,npc=0;ddlaGetGridDims(h,npr,npc);
    int desc[DDLA_DLEN_];ddlaDescInit(desc,h,n+3,n+5,4,4,shifted?npr-1:0,shifted?npc-1:0);
    desc[DDLA_LLD_]+=3;
    const int rows=ddla_test::m_loc(h,desc),cols=ddla_test::n_loc(h,desc),ld=desc[DDLA_LLD_];
    const T poison=value<T>(12345,-34567);
    std::vector<T> local(std::max<std::size_t>(1,local_size(h,desc)),poison);
    R magnitude=R(1);
    if(range_kind==1)magnitude=std::numeric_limits<R>::denorm_min()*R(16);
    if(range_kind==2)magnitude=std::numeric_limits<R>::max()/R(4096);
    const int effective=std::max(1,std::min(b,std::max(n-1,1)));
    for(int c=0;c<cols;++c)for(int r=0;r<rows;++r) {
        const int gi=indx_l2g_r(desc,h,r),gj=indx_l2g_c(desc,h,c);
        if(gi<n&&gj<n&&gi<=gj&&gj-gi<=effective) {
            T x=band_value<T>(gi,gj,effective,magnitude);
            if constexpr(!std::is_same<T,R>::value)
                if(gi==gj)x=T(x.real(),std::numeric_limits<R>::quiet_NaN());
            local[r+static_cast<std::size_t>(c)*ld]=x;
        }
    }
    const auto input=local;
    std::vector<R> d(std::max(n,1)),e(std::max(n,1));std::vector<T> phases;
    detail::Hb2stHouseholderCpu<T> state(h,n,b,true);
    try {
        state.reduce(local.data(),desc,d.data(),e.data(),phases);
        const R scale=state.eigenvalue_scale();
        require_close(h,"HH finite positive input scale",(!std::isfinite(scale)||scale<=R(0)),0);
        require_close(h,"HH preserves input band/poison/padding",std::memcmp(input.data(),local.data(),input.size()*sizeof(T))!=0,0);
        std::uint64_t own=state.local_history_elements(),total=0;
        MPI_Allreduce(&own,&total,1,MPI_UINT64_T,MPI_SUM,ddlaGetCommunicator(h));
        require_close(h,"HH exact distributed history ownership",total!=state.total_history_elements(),0);
        if constexpr(std::is_same<T,R>::value)require_close(h,"HH real signed tridiagonal has no phases",!phases.empty(),0);
        else require_close(h,"HH complex phase count",phases.size()!=static_cast<std::size_t>(std::max(n-1,0)),0);
        const detail::RowShare share(n,P);const int ncols=share.count(rank),first=share.lo(rank),qld=std::max(n,1)+3;
        const T sentinel=T(R(137));
        std::vector<T> strip(std::max<std::size_t>(1,static_cast<std::size_t>(qld)*ncols),sentinel);
        for(int c=0;c<ncols;++c)for(int r=0;r<n;++r) {
            T phase=T(1);if(r>0&&!phases.empty())phase=phases[r-1];
            strip[r+static_cast<std::size_t>(c)*qld]=r==first+c?phase:T(0);
        }
        state.apply_columns(strip.data(),qld,ncols);
        std::vector<T> q(static_cast<std::size_t>(n)*n,T(0));
        double padding=0,finite=0;
        for(int c=0;c<ncols;++c) {
            for(int r=0;r<n;++r) {
                const T x=strip[r+static_cast<std::size_t>(c)*qld];
                if(!hh::finite(x))++finite;
                q[r+static_cast<std::size_t>(first+c)*n]=x;
            }
            for(int r=n;r<qld;++r)padding+=strip[r+static_cast<std::size_t>(c)*qld]!=sentinel;
        }
        if(ncols==0)padding+=strip[0]!=sentinel;
        if(!q.empty())MPI_Allreduce(MPI_IN_PLACE,q.data(),n*n,ddla::detail::mpi_datatype<T>(),MPI_SUM,ddlaGetCommunicator(h));
        long double norm=0,residual=0,orthogonality=0;
        for(int c=0;c<n;++c)for(int r=0;r<n;++r) {
            const auto x=wide(hh::divide_real(band_value<T>(r,c,effective,magnitude),scale));norm+=std::norm(x);
            std::complex<long double> aq=0,qt=wide(q[r+static_cast<std::size_t>(c)*n])*static_cast<long double>(d[c]),dot=0;
            for(int k=0;k<n;++k) {
                aq+=wide(hh::divide_real(band_value<T>(r,k,effective,magnitude),scale))*wide(q[k+static_cast<std::size_t>(c)*n]);
                dot+=std::conj(wide(q[k+static_cast<std::size_t>(r)*n]))*wide(q[k+static_cast<std::size_t>(c)*n]);
            }
            if(c)qt+=wide(q[r+static_cast<std::size_t>(c-1)*n])*static_cast<long double>(e[c-1]);
            if(c+1<n)qt+=wide(q[r+static_cast<std::size_t>(c+1)*n])*static_cast<long double>(e[c]);
            residual+=std::norm(aq-qt);orthogonality+=std::norm(dot-(r==c?1.L:0.L));
        }
        const long double eps=std::numeric_limits<R>::epsilon(),nn=std::max(n,1);
        const double rm=static_cast<double>(std::sqrt(residual)/(nn*eps*std::max(std::sqrt(norm),1.L)));
        const double om=static_cast<double>(std::sqrt(orthogonality)/(nn*eps));
        const std::string tag="HH band n="+std::to_string(n)+" b="+std::to_string(b)+" bytes="+std::to_string(sizeof(T))+" range="+std::to_string(range_kind);
        require_close(h,tag+" finite",finite,0);require_close(h,tag+" padding",padding,0);
        require_close(h,tag+" independent normalized similarity",std::isfinite(rm)?rm:INFINITY,50);
        require_close(h,tag+" independent orthogonality",std::isfinite(om)?om:INFINITY,50);
        state.release();state.release();
    }catch(...){state.release();throw;}
}

template<class T>
void check_failures(const DdlaHandle_t& h) {
    using R=typename real_type<T>::type;using Options=detail::Hb2stHouseholderCpuOptions;
    const int n=33,rank=ddlaGetRank(h),P=ddlaGetSize(h);
    int desc[DDLA_DLEN_];ddlaDescInit(desc,h,n,n,4,4,0,0);
    auto matrix=make_local<T>(h,desc,[&](int i,int j){return band_value<T>(i,j,4,R(1));});
    for(auto failure:{Options::Failure::Metadata,Options::Failure::Band,Options::Failure::History,
                     Options::Failure::Factors,Options::Failure::Reduce,Options::Failure::Apply}) {
        Options options;options.failure=failure;options.failure_rank=std::min(1,P-1);
        int caught=0;
        try {
            detail::Hb2stHouseholderCpu<T> state(h,n,4,true,options);
            try {
                std::vector<R>d(n),e(n);std::vector<T>phases;
                state.reduce(matrix.data(),desc,d.data(),e.data(),phases);
                std::vector<T> strip(static_cast<std::size_t>(n)*2,T(0));
                state.apply_columns(strip.data(),n,rank%2?0:2);
                state.release();
            }catch(...){state.release();throw;}
        }catch(const std::exception&){caught=1;}
        require_close(h,"HH cooperative failure is caught everywhere",caught!=1,0);
    }
    Options model;model.reject_memory_model=rank==std::min(1,P-1);
    int caught=0;try {detail::Hb2stHouseholderCpu<T> state(h,n,4,true,model);state.release();}catch(const std::exception&){caught=1;}
    require_close(h,"HH unsupported memory model rejects collectively",caught!=1,0);
    detail::Hb2stHouseholderCpu<T> next(h,n,4,false);
    try {
        std::vector<R>d(n),e(n);std::vector<T>phases;next.reduce(matrix.data(),desc,d.data(),e.data(),phases);
        require_close(h,"HH eigenvalues-only has no retained history",next.local_history_elements()+next.total_history_elements(),0);
        next.release();
    }catch(...){next.release();throw;}
}

template<class T> T full_value(int i,int j) {
    if(i<j)return hh::adj(full_value<T>(j,i));
    return i==j?value<T>(3+0.02*i,0):value<T>(0.1*std::sin(i*0.61+j*0.37),0.1*std::cos(i*0.31-j*0.27));
}
template<class T>
void check_full(const DdlaHandle_t& h,int n,bool upper,bool alias) {
    using R=typename real_type<T>::type;
    const int P=ddlaGetSize(h);int npr=0,npc=0;ddlaGetGridDims(h,npr,npc);
    int descA[DDLA_DLEN_],descZ[DDLA_DLEN_];
    ddlaDescInit(descA,h,n+3,n+5,8,8,0,0);descA[DDLA_LLD_]+=3;
    ddlaDescInit(descZ,h,n+3,n+5,8,8,0,alias?0:npc-1);descZ[DDLA_LLD_]+=3;
    if(alias)std::copy(descA,descA+DDLA_DLEN_,descZ);
    const T sentinel=value<T>(139,-137);
    auto make_input=[&](const int* desc,bool poisoned) {
        std::vector<T> local(std::max<std::size_t>(1,local_size(h,desc)),sentinel);
        const int rows=ddla_test::m_loc(h,desc),cols=ddla_test::n_loc(h,desc),ld=desc[DDLA_LLD_];
        for(int c=0;c<cols;++c)for(int r=0;r<rows;++r) {
            const int i=indx_l2g_r(desc,h,r),j=indx_l2g_c(desc,h,c);
            if(i>=n||j>=n)continue;
            T x=full_value<T>(i,j);
            if(poisoned&&(upper?i>j:i<j))x=value<T>(1e5,-2e5);
            if constexpr(!std::is_same<T,R>::value)if(poisoned&&i==j)x=T(x.real(),R(1e5));
            local[r+static_cast<std::size_t>(c)*ld]=x;
        }
        return local;
    };
    const auto input=make_input(descA,true),full=make_input(descA,false);
    DeviceBuffer<T>A(h,input.size()),Z(h,std::max<std::size_t>(1,local_size(h,descZ))),reference(h,input.size());
    upload(h,A.ptr,input);upload(h,Z.ptr,std::vector<T>(Z.count,sentinel));upload(h,reference.ptr,input);
    std::vector<R>w(n),old(n),only(n);int info=-1;
    detail::pheevd_householder<DdlaBackend::CPU, T>(h,'V',upper?'U':'L',n,n,A.ptr,descA,w.data(),alias?A.ptr:Z.ptr,descZ,info);
    require_close(h,"HH private full driver info",info,0);
    T* output=alias?A.ptr:Z.ptr;
    const auto after=download(h,output,alias?input.size():Z.count);
    double padbad=0,finite=0;
    const int zr=ddla_test::m_loc(h,descZ),zc=ddla_test::n_loc(h,descZ),zld=descZ[DDLA_LLD_];
    for(int c=0;c<zc;++c)for(int r=0;r<zld;++r) {
        const int i=r<zr?indx_l2g_r(descZ,h,r):n,j=indx_l2g_c(descZ,h,c);
        const T x=after[r+static_cast<std::size_t>(c)*zld];
        if(i<n&&j<n){if(!hh::finite(x))++finite;}else padbad+=x!=sentinel;
    }
    require_close(h,"HH full output padding",padbad,0);require_close(h,"HH full finite vectors",finite,0);
    upload(h,reference.ptr,full);
    const double residual=eig_test::residual_metric<T>(h,reference.ptr,descA,output,descZ,w.data(),n);
    const double orthogonal=eig_test::orthogonality_metric<T>(h,output,descZ,n);
    require_close(h,"HH full independent residual",std::isfinite(residual)?residual:INFINITY,50);
    require_close(h,"HH full independent orthogonality",std::isfinite(orthogonal)?orthogonal:INFINITY,50);
    upload(h,reference.ptr,input);
    pheevd<T>(h,'N',upper?'U':'L',n,n,reference.ptr,descA,old.data(),nullptr,nullptr,info);
    require_close(h,"HH reference eigenvalues info",info,0);
    upload(h,reference.ptr,input);
    detail::pheevd_householder<DdlaBackend::CPU, T>(h,'N',upper?'U':'L',n,n,reference.ptr,descA,only.data(),nullptr,nullptr,info);
    require_close(h,"HH eigenvalues-only info",info,0);
    double error=0,norm=0;
    for(int j=0;j<n;++j) {
        if(!std::isfinite(w[j])||!std::isfinite(only[j])||!std::isfinite(old[j])||(j&&w[j]<w[j-1]))error=INFINITY;
        else error=std::max({error,std::abs(double(w[j]-old[j])),std::abs(double(only[j]-old[j]))});
        for(int i=0;i<n;++i)norm+=std::norm(full_value<T>(i,j));
    }
    require_close(h,"HH full eigenvalue agreement",error/(std::max(1,n)*std::numeric_limits<R>::epsilon()*std::sqrt(norm)),50);
    (void)P;
}
}

int main(int argc,char**argv) {
    int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int rank=0,size=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    try {
        if(provided<MPI_THREAD_FUNNELED)throw std::runtime_error("MPI thread support");
        TestOptions options;std::string error;if(!parse_options(argc,argv,size,options,error))throw std::runtime_error(error);
        for(const auto& grid:options.grids) {
            DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,DdlaBackend::CPU));DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,grid.first,grid.second));
#define HH_CASES(T) \
            for(auto shape:std::vector<std::pair<int,int>>{{0,1},{1,4},{2,4},{3,2},{17,4},{33,8},{65,13},{97,16},{129,8}}) \
                check_band<T>(h,shape.first,shape.second,shape.first%2,0); \
            check_band<T>(h,17,4,true,1);check_band<T>(h,17,4,true,2); \
            check_failures<T>(h); \
            check_full<T>(h,33,true,false);check_full<T>(h,33,false,true);
            HH_CASES(float)
            HH_CASES(double)
            HH_CASES(std::complex<float>)
            HH_CASES(std::complex<double>)
#undef HH_CASES
            DDLA_CHECK(ddlaDestroy(h));
        }
        if(rank==0)std::puts("test_hb2st_householder_cpu passed");
    }catch(const std::exception& e){std::fprintf(stderr,"rank %d: %s\n",rank,e.what());MPI_Abort(MPI_COMM_WORLD,1);}
    MPI_Finalize();return 0;
}
