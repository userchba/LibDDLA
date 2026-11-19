#include "generalized_eigen_test_common.h"
#include "workspace_test_common.h"
#include "Eigensolver/hegst_impl.h"
#include "ddla_stream.h"
#include "check_agree.h"
#include <array>
#include <cstring>

namespace {
using namespace ddla;
using namespace api_grid_test;

bool raw_no_alloc(const workspace_test::Snapshot& s)
{
    return !s.ddla_alloc&&!s.ddla_free&&!s.pinned_alloc&&!s.raw_gpu_alloc&&
        !s.host_new&&!s.vendor_query&&!s.resource_setup;
}
bool query_pure(const workspace_test::Snapshot& s)
{
    return raw_no_alloc(s)&&!s.copies&&!s.launches&&!s.syncs&&!s.collectives;
}
class RawArena {
    static constexpr std::size_t guard=256;
    DdlaHandle_t h_;
    std::size_t bb_,hb_;
    ddla::vector<char> backend_;
    std::vector<char> host_;
public:
    RawArena(DdlaHandle_t h,std::size_t b,std::size_t s):h_(h),bb_(b),hb_(s),
        backend_(h,b+2*guard),host_(s+2*guard,char(0xa7))
    {DDLA_CHECK(ddlaMemset(h,backend_.data(),0xa7,backend_.size()));}
    char* backend(){return backend_.data()+guard;}
    char* host(){return host_.data()+guard;}
    bool valid(){
        std::array<unsigned char,2*guard> bytes{};
        DDLA_CHECK(ddlaMemcpy(h_,bytes.data(),backend_.data(),guard));
        DDLA_CHECK(ddlaMemcpy(h_,bytes.data()+guard,backend()+bb_,guard));
        DDLA_CHECK(ddlaSynchronize(h_));bool okay=true;
        for(std::size_t i=0;i<guard;++i)okay=okay&&bytes[i]==0xa7&&bytes[guard+i]==0xa7&&
            static_cast<unsigned char>(host_[i])==0xa7&&static_cast<unsigned char>(host()[hb_+i])==0xa7;
        return okay;
    }
};

template<class T> T selected_diagonal(T value)
{
    if constexpr(std::is_arithmetic_v<T>) return value;
    else return T(value.real(),std::numeric_limits<typename real_type<T>::type>::quiet_NaN());
}

template<class T>
void exercise(const DdlaHandle_t& h,int n,int block,char uplo)
{
    using R=typename real_type<T>::type;
    require_close(h,"PHEGST raw allocation diagnostics preload present",workspace_test::available()?0:1,0);
    int da[DDLA_DLEN_],df[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(da,h,n+2,n+3,block,block,0,0));
    DDLA_CHECK(ddlaDescInit(df,h,n+3,n+2,block,block,0,0));
    da[DDLA_LLD_]+=3;
    df[DDLA_LLD_]+=2;
    const T sentinel=T(8157),nan=T(std::numeric_limits<R>::quiet_NaN());
    auto f=make_local<T>(h,df,[&](int i,int j) {
        if(i>=n || j>=n)return sentinel;
        return (uplo=='L'?i>=j:i<=j)?generalized_test::b<T>(i,j,n):nan;
    });
    DeviceBuffer<T> factor(h,std::max(std::size_t(1),f.size()));
    DeviceBuffer<T> a(h,std::max(std::size_t(1),local_size(h,da)));
    f.resize(factor.count,sentinel);
    upload(h,factor.ptr,f);
    int info=0;
    if(n)ppotrf<T>(h,uplo,n,factor.ptr,1,1,df,info);
    require_close(h,"public phegst factor once",info,0);
    f=download(h,factor.ptr,factor.count);
    for(int j=0;j<ddla_test::n_loc(h,df);++j)for(int i=0;i<ddla_test::m_loc(h,df);++i) {
        const int gi=indx_l2g_r(df,h,i),gj=indx_l2g_c(df,h,j);
        if(gi<n && gj<n && (uplo=='L'?gi<gj:gi>gj))
            f[i+std::size_t(j)*df[DDLA_LLD_]]=nan;
    }
    upload(h,factor.ptr,f);
    int row=0,column=0,rows=1,columns=1;
    ddlaGetGridCoords(h,row,column);
    ddlaGetGridDims(h,rows,columns);
    const bool active=num_loc(n,block,row,0,rows)>0 && num_loc(n,block,column,0,columns)>0;
    // Public and borrowed entry use the same checked dense core. No warmup
    // solve precedes raw observation; reserve the transport explicitly.
    upload(h,a.ptr,std::vector<T>(a.count,sentinel));
    const auto plan=hegst_impl::check<T>(h,ddlaGetBackend(h),uplo,n,active?a.ptr:nullptr,da,
                                        active?factor.ptr:nullptr,df);
    bool root_pure=true;
    MPI_Barrier(h->comm);
    if(!h->myid){workspace_test::Scope scope(true);(void)hegst_impl::workspace_size<T>(h,plan);
        root_pure=query_pure(scope.finish());}
    MPI_Barrier(h->comm);
    workspace_test::Scope query_scope(true);const auto q=hegst_impl::workspace_size<T>(h,plan);
    require_close(h,"dense congruence pure root/all-rank query",root_pure&&query_pure(query_scope.finish())?0:1,0);
    const bool leaf=n>0 && n<=block;
    if(leaf)require_close(h,"single-block congruence has zero arenas/cache on every rank",
        !q.backend_bytes&&!q.host_bytes&&!q.staging_bytes[0]&&!q.staging_bytes[1]?0:1,0);
    RawArena arena(h,q.backend_bytes,q.host_bytes);
    detail::agree_local(h,"phegst test transport reservation",[&]{
        if(q.staging_bytes[0]||q.staging_bytes[1])h->host_buffers<std::byte>(q.staging_bytes[0],q.staging_bytes[1]);
    });
    const auto raw_run=[&](const hegst_impl::Plan& selected){
        if(ddlaGetBackend(h)==DdlaBackend::CPU)
            hegst_impl::run<DdlaBackend::CPU,T>(h,selected,active?a.ptr:nullptr,active?factor.ptr:nullptr,
                arena.backend(),q.backend_bytes,arena.host(),q.host_bytes);
#if DDLA_HAS_GPU
        else hegst_impl::run<DdlaBackend::GPU,T>(h,selected,active?a.ptr:nullptr,active?factor.ptr:nullptr,
                arena.backend(),q.backend_bytes,arena.host(),q.host_bytes);
#endif
    };
    for(int pass=0;pass<2;++pass) {
        auto input=make_local<T>(h,da,[&](int i,int j) {
            if(i>=n || j>=n)return sentinel;
            if(!(uplo=='L'?i>=j:i<=j))return nan;
            T value=generalized_test::a<T>(i,j,n);
            if(pass)value=T(1.25)*value+T(.37)*generalized_test::b<T>(i,j,n);
            return i==j?selected_diagonal(value):value;
        });
        input.resize(a.count,sentinel);
        upload(h,a.ptr,input);
        const T* fixed_factor=active?factor.ptr:nullptr;
        DDLA_CHECK(ddlaSynchronize(h));
        workspace_test::Scope raw_scope;raw_run(plan);const auto raw_trace=raw_scope.finish();
        require_close(h,"dense congruence raw first/repeat has no allocation/query/setup",raw_no_alloc(raw_trace)?0:1,0);
        if(leaf)require_close(h,"single-block congruence has no communication on any rank",raw_trace.collectives?1:0,0);
        const auto actual=download(h,a.ptr,a.count);
        upload(h,a.ptr,input);
        phegst<T>(h,uplo,n,active?a.ptr:nullptr,da,fixed_factor,df);
        const auto owning=download(h,a.ptr,a.count),after=download(h,factor.ptr,factor.count);
        double owning_error=0;
        for(std::size_t at=0;at<actual.size();++at)
            owning_error=std::max(owning_error,workspace_test::relative_error(actual[at],owning[at]));
        require_close(h,"dense congruence owning/raw repeat agreement",owning_error,8*std::numeric_limits<R>::epsilon());
        require_close(h,"dense congruence exact arena guards",arena.valid()?0:1,0);
        bool guards=!std::memcmp(f.data(),after.data(),factor.count*sizeof(T));
        double error=0,imaginary=0;
        for(std::size_t at=0;at<actual.size();++at) {
            const int i=int(at%da[DDLA_LLD_]),j=int(at/da[DDLA_LLD_]);
            const int gi=indx_l2g_r(da,h,i),gj=indx_l2g_c(da,h,j);
            if(i>=ddla_test::m_loc(h,da) || j>=ddla_test::n_loc(h,da) || gi>=n || gj>=n) {
                guards=guards && !std::memcmp(&actual[at],&input[at],sizeof(T));
                continue;
            }
            T expected=generalized_test::standard<T>(gi,gj,n);
            if(pass)expected=T(1.25)*expected+T(gi==gj?.37:0.);
            error=std::max(error,workspace_test::relative_error(actual[at],expected)/
                (std::numeric_limits<R>::epsilon()*std::max(1,n)));
            if constexpr(!std::is_arithmetic_v<T>)
                if(gi==gj)imaginary=std::max(imaginary,double(std::abs(actual[at].imag())));
        }
        require_close(h,"public phegst analytic reduced matrix (scaled)",error,400);
        require_close(h,"public phegst const factor, parents and real diagonal",guards && imaginary==0?0:1,0);
    }
    if(n==17 && block==4 && uplo=='L') {
        const auto before_a=download(h,a.ptr,a.count),before_f=download(h,factor.ptr,factor.count);
        for(int mode=0;mode<3;++mode){
            auto stale=plan;
            if(mode==0)stale.transport=plan.transport==DdlaCommunication::MPI?DdlaCommunication::HostStagedMPI:DdlaCommunication::MPI;
            if(mode==1)++stale.completion.pr;
            if(mode==2)++stale.completion.threads;
            bool query_rejected=false,run_rejected=false;
            try{(void)hegst_impl::workspace_size<T>(h,stale);}catch(const std::exception&){query_rejected=true;}
            try{raw_run(stale);}catch(const std::exception&){run_rejected=true;}
            require_close(h,"dense congruence prepared transport/grid/host-budget snapshot guard",query_rejected&&run_rejected?0:1,0);
        }
        for(int mode=0;mode<3;++mode) {
            const bool root=ddlaGetRank(h)==0;
            const int count=mode==1 && root?(ddlaGetSize(h)>1?n+1:n+4):n;
            bool rejected=false;
            try {
                phegst<T>(h,mode==2 && root?'?':'L',count,
                    mode==0 && root?factor.ptr:a.ptr,da,factor.ptr,df);
            } catch(const std::exception&) {rejected=true;}
            require_close(h,"public phegst collective argument/alias rejection",rejected?0:1,0);
            const auto after_a=download(h,a.ptr,a.count),after_f=download(h,factor.ptr,factor.count);
            require_close(h,"public phegst rejected matrices unchanged",
                !std::memcmp(before_a.data(),after_a.data(),a.count*sizeof(T)) &&
                !std::memcmp(before_f.data(),after_f.data(),factor.count*sizeof(T))?0:1,0);
        }
    }
    if(!ddlaGetRank(h))std::cout<<"PHEGST_DENSE_RAW_PASS n="<<n<<" block="<<block<<" uplo="<<uplo
        <<" type_bytes="<<sizeof(T)<<" query_pure=1 raw_no_alloc=1 guards=1 repeats=2"
        <<" backend_bytes="<<q.backend_bytes<<" host_bytes="<<q.host_bytes<<" staging="<<q.staging_bytes[0]<<','<<q.staging_bytes[1]<<std::endl;
}
template<class T>void leaf_query(const DdlaHandle_t& h,char uplo)
{
    // Real parent allocations, including empty/rounded owners. Only the input
    // descriptor's unavoidable single-block ownership is quadratic; the query
    // must add zero workspace/cache even at a large block size.
    constexpr int n=1024;
    int da[DDLA_DLEN_],df[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(da,h,n+2,n+3,n,n,0,0));da[DDLA_LLD_]+=3;
    DDLA_CHECK(ddlaDescInit(df,h,n+3,n+2,n,n,0,0));df[DDLA_LLD_]+=2;
    DeviceBuffer<T> a(h,std::max(std::size_t(1),local_size(h,da))),f(h,std::max(std::size_t(1),local_size(h,df)));
    DDLA_CHECK(ddlaMemset(h,a.ptr,0xa7,a.count*sizeof(T)));
    DDLA_CHECK(ddlaMemset(h,f.ptr,0xa7,f.count*sizeof(T)));
    const auto p=hegst_impl::check<T>(h,ddlaGetBackend(h),uplo,n,a.ptr,da,f.ptr,df);
    DDLA_CHECK(ddlaSynchronize(h));
    for(bool root_only:{true,false}) {
        MPI_Barrier(h->comm);bool okay=true;
        if(!root_only || !h->myid) {
            workspace_test::Scope scope(true);
            const auto q=hegst_impl::workspace_size<T>(h,p);
            std::size_t bb=17,hb=19;
            hegst_impl::congruence_solve_workspace_size<T>(h,uplo,n,da,df,&bb,&hb);
            const auto staging=hegst_impl::congruence_staging_bytes<T>(h,p.completion,da,df);
            const auto trace=scope.finish();
            okay=query_pure(trace)&&!bb&&!hb&&!q.backend_bytes&&!q.host_bytes&&
                !q.staging_bytes[0]&&!q.staging_bytes[1]&&!staging[0]&&!staging[1];
        }
        MPI_Barrier(h->comm);
        require_close(h,"single-block 1024 pure root/all-rank zero query with valid owners",okay?0:1,0);
    }
    if(!h->myid)std::cout<<"PHEGST_LEAF_QUERY_PASS n="<<n<<" block="<<n<<" uplo="<<uplo
        <<" type_bytes="<<sizeof(T)<<" backend_bytes=0 host_bytes=0 staging=0,0 root_and_all=1 valid_input_owners=1"<<std::endl;
}
template<class T>void cases(const DdlaHandle_t& h)
{
    for(int n:{0,1,17})for(char uplo:{'L','U'})exercise<T>(h,n,4,uplo);
    for(char uplo:{'L','U'})exercise<T>(h,17,17,uplo);
    for(char uplo:{'L','U'}) {
        exercise<T>(h,1,17,uplo);
        exercise<T>(h,17,64,uplo);
        leaf_query<T>(h,uplo);
    }
}
}
int main(int argc,char** argv)
{
    ddla_test_mpi_init(&argc,&argv);
    try {
        int ranks=1;MPI_Comm_size(MPI_COMM_WORLD,&ranks);
        int rows=1,columns=ranks;char major='R';
        DdlaBackend backend=DdlaBackend::CPU;
        std::string transport="default";
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--cpu")backend=DdlaBackend::CPU;
            else if(arg=="--gpu")backend=DdlaBackend::GPU;
            else if(arg=="--grid" && i+1<argc) {
                if(!parse_grid_spec(argv[++i],rows,columns))throw std::invalid_argument("invalid grid");
            } else if(arg=="--major" && i+1<argc)major=argv[++i][0];
            else if(arg=="--transport" && i+1<argc)transport=argv[++i];
            else throw std::invalid_argument("usage: test_phegst [--cpu|--gpu] [--grid RxC] [--major R|C] [--transport default|mpi|staged|ccl]");
        }
        if(rows<1 || rows>ranks || ranks%rows || columns!=ranks/rows || (major!='R' && major!='C'))
            throw std::invalid_argument("invalid process grid");
        if(transport!="default" && transport!="mpi" && transport!="staged" && transport!="ccl")
            throw std::invalid_argument("invalid transport");
        if(backend==DdlaBackend::CPU && transport!="default" && transport!="mpi")
            throw std::invalid_argument("CPU handles require MPI transport");
        DdlaHandle_t h=nullptr;
        DDLA_CHECK(ddlaInit(h,backend));
        if(backend==DdlaBackend::GPU && transport!="default")
            DDLA_CHECK(ddlaSetCommunication(h,transport=="mpi"?DdlaCommunication::MPI:
                transport=="staged"?DdlaCommunication::HostStagedMPI:DdlaCommunication::CCL));
        DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,rows,columns,major));
        cases<float>(h);cases<double>(h);
        cases<std::complex<float>>(h);cases<std::complex<double>>(h);
        if(!ddlaGetRank(h))std::cout<<"PUBLIC_PHEGST_PASS backend="<<int(backend)
            <<" grid="<<rows<<'x'<<columns<<" major="<<major<<" transport="<<transport
            <<" factor_reuse=2 cases=48 leaf_queries=8"<<std::endl;
        DDLA_CHECK(ddlaDestroy(h));
        MPI_Finalize();return 0;
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<std::endl;
        MPI_Abort(MPI_COMM_WORLD,1);return 1;
    }
}
