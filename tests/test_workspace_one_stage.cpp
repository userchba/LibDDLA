#include "api_grid_test_common.h"
#include "ddla_stream.h"
#include "eig_test_common.h"
#include "workspace_test_common.h"
#include "Eigensolver/one_stage/one_stage_workspace.h"
#include <array>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

using namespace api_grid_test;
namespace {
bool no_scratch(const workspace_test::Snapshot& s,bool allow_host_growth=false)
{return !s.ddla_alloc&&!s.ddla_free&&!s.pinned_alloc&&!s.raw_gpu_alloc&&(allow_host_growth||!s.host_new)&&!s.vendor_query&&!s.resource_setup;}
bool pure(const workspace_test::Snapshot& s)
{return no_scratch(s)&&!s.copies&&!s.launches&&!s.syncs&&!s.collectives;}
// Exactly the sized backend and host bytes, between 256-byte canaries.
class Guarded {
    static constexpr std::size_t guard=256;
    DdlaHandle_t h_;std::size_t backend_bytes_,host_bytes_;
    ddla::vector<unsigned char> backend_;std::vector<unsigned char> host_;
public:
    Guarded(DdlaHandle_t h,std::size_t backend_bytes,std::size_t host_bytes)
        :h_(h),backend_bytes_(backend_bytes),host_bytes_(host_bytes),backend_(h,backend_bytes+2*guard),
         host_(host_bytes+2*guard,0x79)
    {
        DDLA_CHECK(ddlaMemset(h,backend_.data(),0x79,backend_.size()));
        check_ddla_sync(h);
    }
    void* backend(){return backend_.data()+guard;}
    void* host(){return host_.data()+guard;}
    void verify()
    {
        std::array<unsigned char,2*guard> device{};
        DDLA_CHECK(ddlaMemcpy(h_,device.data(),backend_.data(),guard));
        DDLA_CHECK(ddlaMemcpy(h_,device.data()+guard,backend_.data()+guard+backend_bytes_,guard));check_ddla_sync(h_);
        bool intact=true;
        for(std::size_t i=0;i<guard;++i)intact=intact&&host_[i]==0x79
            &&host_[guard+host_bytes_+i]==0x79&&device[i]==0x79&&device[guard+i]==0x79;
        require_close(h_,"one-stage exact workspace canaries",intact?0:1,0);
    }
};
template<class T>T entry(int i,int j,int salt,bool diagonal)
{
    using R=typename real_type<T>::type;
    if(i==j)return T(R(i+1)+R(salt)/R(8));
    if(diagonal)return T(0);
    const int lo=std::min(i,j),hi=std::max(i,j);
    const R re=R((lo*13+hi*7+salt)%17-8)/R(70);
    if constexpr(std::is_same_v<T,R>)return re;
    else {const R im=R((lo*5+hi*3+salt)%11-5)/R(90);return i<j?T(re,im):T(re,-im);}
}
template<DdlaBackend B,class T>
void exercise(DdlaHandle_t h,bool stored,char uplo,char job,int n,bool alias=false,bool tight=false,int nb=4)
{
    using R=typename real_type<T>::type;
    int da[DDLA_DLEN_],dz[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(da,h,tight?n:n+2,tight?n:n+3,nb,nb,0,0));
    DDLA_CHECK(ddlaDescInit(dz,h,tight?n:n+2,tight?n:n+3,nb,nb,0,0));
    if(!tight){da[DDLA_LLD_]+=3+h->myid%2;dz[DDLA_LLD_]+=4+h->myid%2;}
    DeviceBuffer<T> a(h,local_size(h,da)),z(h,local_size(h,dz));
    std::vector<R> eigenvalues(std::max(1,n),R(-999));
    const T sentinel=T(937);const R nan=std::numeric_limits<R>::quiet_NaN();
    auto input=[&](int salt,bool full) {
        auto values=make_local<T>(h,da,[&](int i,int j) {
            if(i>=n||j>=n)return sentinel;
            if(!full&&(uplo=='U'?i>j:i<j))return T(nan);
            if constexpr(!std::is_same_v<T,R>)if(!full&&i==j)return T(R(i+1)+R(salt)/R(8),nan);
            return entry<T>(i,j,salt,job=='N');
        });
        for(int j=0;j<ddla_test::n_loc(h,da);++j)for(int i=ddla_test::m_loc(h,da);i<da[DDLA_LLD_];++i)
            values[i+std::size_t(j)*da[DDLA_LLD_]]=sentinel;
        return values;
    };
    auto before=input(0,false);upload(h,a.ptr,before);
    std::vector<T> output(local_size(h,dz),sentinel);upload(h,z.ptr,output);
    T* target=alias?a.ptr:z.ptr;const int* target_desc=alias?da:dz;
    const auto p=spectral_impl::one_stage_check<T>(h,B,stored,job,uplo,n,a.ptr,da,eigenvalues.data(),
        job=='V'?target:nullptr,job=='V'?target_desc:nullptr);
    if(n&&job=='V'){
        const detail::RowShare rows(n,h->nprocs);
        require_close(h,"one-stage default uses tightly packed row D&C",
            p.vectors.n==n&&p.vectors.ldy==std::max(1,rows.count(h->myid))&&
            p.vectors.borrow_q2==(rows.count(h->myid)>0)?0:1,0);
    }
    MPI_Barrier(h->comm);bool one=true;
    if(h->myid==0){
        workspace_test::Scope scope(true);std::size_t backend=0,host=0;
        spectral_impl::one_stage_workspace_size<T>(h,p,&backend,&host);one=pure(scope.finish());
    }
    MPI_Barrier(h->comm);require_close(h,"one-stage one-rank query purity",one?0:1,0);
    std::size_t backend_bytes=0,host_bytes=0;
    workspace_test::Scope query_scope(true);spectral_impl::one_stage_workspace_size<T>(h,p,&backend_bytes,&host_bytes);
    const auto qs=query_scope.finish();
    require_close(h,"one-stage all-rank query purity",pure(qs)?0:1,0);
    const auto after_query=download(h,a.ptr,before.size());
    require_close(h,"one-stage check/query preserve input",!before.empty()&&std::memcmp(after_query.data(),before.data(),before.size()*sizeof(T))?1:0,0);
    Guarded owner(h,backend_bytes,host_bytes);spectral_impl::OneStageResources resources(h,p);
    for(int pass=0;pass<2;++pass) {
        before=input(pass,false);upload(h,a.ptr,before);upload(h,z.ptr,output);
        std::fill(eigenvalues.begin(),eigenvalues.end(),R(-999));check_ddla_sync(h);int info=-1;
        workspace_test::Scope scope;
        spectral_impl::one_stage_run<B,T>(h,p,resources,a.ptr,eigenvalues.data(),job=='V'?target:nullptr,info,nullptr,
            owner.backend(),backend_bytes,owner.host(),host_bytes);
        const auto snapshot=scope.finish();
        bool allow_host_growth=false;
#if defined(DDLA_USE_GPU_CPU_TUNNEL)
        allow_host_growth=B==DdlaBackend::GPU && pass==0;
#endif
        require_close(h,"one-stage run reuses algorithm scratch and warm communication storage",no_scratch(snapshot,allow_host_growth)?0:1,0);
        require_close(h,"one-stage info",info,0);check_ddla_sync(h);owner.verify();
        const auto result=download(h,job=='V'?target:a.ptr,job=='V'?local_size(h,target_desc):before.size());
        const auto changed_input=download(h,a.ptr,before.size());
        const int ar=num_loc(n,nb,h->myprow_,0,h->nprows_),ac=num_loc(n,nb,h->mypcol_,0,h->npcols_);
        double changed=0;
        for(int j=0;j<ddla_test::n_loc(h,da);++j)for(int i=0;i<da[DDLA_LLD_];++i) {
            const auto at=i+std::size_t(j)*da[DDLA_LLD_];
            const int gi=indxl2g(i,nb,h->myprow_,0,h->nprows_),gj=indxl2g(j,nb,h->mypcol_,0,h->npcols_);
            const bool padding=i>=ar||j>=ac;
            if(padding||(!(job=='V'&&alias)&&stored&&(uplo=='U'?gi>gj:gi<gj)))
                if(std::memcmp(&changed_input[at],&before[at],sizeof(T)))changed=1;
        }
        require_close(h,"one-stage preserves unused input/padding",changed,0);
        if(job=='V') {
            const int nr=num_loc(n,nb,h->myprow_,0,h->nprows_),nc=num_loc(n,nb,h->mypcol_,target_desc[DDLA_CSRC_],h->npcols_);
            double pad=0,finite=0;
            for(int j=0;j<ddla_test::n_loc(h,target_desc);++j)for(int i=0;i<target_desc[DDLA_LLD_];++i)
                if(i>=nr||j>=nc) {
                    const auto at=i+std::size_t(j)*target_desc[DDLA_LLD_];const T expected=alias?before[at]:sentinel;
                    if(std::memcmp(&result[at],&expected,sizeof(T)))pad=1;
                } else if(!std::isfinite(std::abs(result[i+std::size_t(j)*target_desc[DDLA_LLD_]])))finite=1;
            for(int i=0;i<n;++i)if(!std::isfinite(eigenvalues[i]))finite=1;
            require_close(h,"one-stage active output is finite",finite,0);
            require_close(h,"one-stage output padding",pad,0);
            if(alias) {
                // A/Z alias is packed into the separate padded descriptor for
                // the independent residual, without assuming equal local LLDs.
                DDLA_CHECK(ddlaMemcpy2D(h,z.ptr,std::size_t(dz[DDLA_LLD_])*sizeof(T),a.ptr,
                    std::size_t(da[DDLA_LLD_])*sizeof(T),std::size_t(nr)*sizeof(T),nc));
            }
            upload(h,a.ptr,input(pass,true));check_ddla_sync(h);
            if(n){require_close(h,"one-stage independent residual",workspace_test::finite_error(eig_test::residual_metric<T,B>(h,a.ptr,da,z.ptr,dz,eigenvalues.data(),n)),50);
                require_close(h,"one-stage independent orthogonality",workspace_test::finite_error(eig_test::orthogonality_metric<T,B>(h,z.ptr,dz,n)),50);}
        } else {
            double error=0;
            for(int i=0;i<n;++i)error=std::max(error,workspace_test::finite_error(double(std::abs(eigenvalues[i]-(R(i+1)+R(pass)/R(8))))/
                (std::max(1,n)*std::numeric_limits<R>::epsilon())));
            require_close(h,"one-stage diagonal eigenvalues",error,50);
        }
    }
}
template<DdlaBackend B,class T>void cases(DdlaHandle_t h)
{
    for(bool stored:{false,true})for(char uplo:{'U','L'}) {
        exercise<B,T>(h,stored,uplo,'V',9);
        exercise<B,T>(h,stored,uplo,'N',9);
    }
    for(bool stored:{false,true})exercise<B,T>(h,stored,'L','V',9,true);
    exercise<B,T>(h,true,'U','V',1);
}
#if defined(DDLA_GPUBLAS_HAS_HEURISTICS_CACHE) && DDLA_GPUBLAS_HAS_HEURISTICS_CACHE
void cache_policy(DdlaHandle_t h)
{
    if(h->backend!=DdlaBackend::GPU||std::getenv("CUBLASLT_HEURISTICS_CACHE_CAPACITY"))return;
    int version=0;BLAS_CHECK(gpublasGetVersion(static_cast<gpublasHandle_t>(h->blasH),&version));
    if(version<111103||version>=120304)return;
    int d[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(d,h,1024,1024,32,32,0,0));
    DeviceBuffer<double>a(h,local_size(h,d)),z(h,local_size(h,d));std::array<double,1024>w{};
    const auto p=spectral_impl::one_stage_check<double>(h,DdlaBackend::GPU,true,'V','U',1024,a.ptr,d,w.data(),z.ptr,d);
    spectral_impl::OneStageResources resources(h,p);
    std::size_t original=0;BLAS_CHECK(gpublasHeuristicsCacheGetCapacity(&original));
    struct Restore {std::size_t capacity;~Restore(){(void)gpublasHeuristicsCacheSetCapacity(capacity);}}restore{original};
    // Change capacity after resource preparation: each entry must observe this
    // value rather than restoring construction-time state on reusable calls.
    for(std::size_t capacity:{std::size_t(4096),std::size_t(8192)}) {
        BLAS_CHECK(gpublasHeuristicsCacheSetCapacity(capacity));
        workspace_test::Scope scope;resources.begin_reduction();std::size_t active=0;
        BLAS_CHECK(gpublasHeuristicsCacheGetCapacity(&active));resources.end_reduction();const auto snapshot=scope.finish();
        std::size_t after=0;BLAS_CHECK(gpublasHeuristicsCacheGetCapacity(&after));
        require_close(h,"one-stage cache policy has no scratch/setup",no_scratch(snapshot)?0:1,0);
        require_close(h,"one-stage cache observes/restores run-entry capacity",
            active==(capacity==8192?0:capacity)&&after==capacity?0:1,0);
    }
    BLAS_CHECK(gpublasHeuristicsCacheSetCapacity(8192));resources.begin_reduction();resources.cancel_reduction();
    std::size_t after=0;BLAS_CHECK(gpublasHeuristicsCacheGetCapacity(&after));
    require_close(h,"one-stage exceptional cache restore",after==8192?0:1,0);
}
#endif

template<DdlaBackend B>void run(MPI_Comm comm,int pr,int pc,char major)
{
    DdlaHandle_t h=nullptr;ddlaInit(h,B);ddlaSet(h,comm,pr,pc,major);
    cases<B,float>(h);cases<B,double>(h);cases<B,std::complex<float>>(h);cases<B,std::complex<double>>(h);
    exercise<B,double>(h,false,'U','V',37,false,true);
    exercise<B,std::complex<double>>(h,true,'L','V',137);
    for(char uplo:{'U','L'})for(bool stored:{false,true})
        exercise<B,std::complex<double>>(h,stored,uplo,'V',139,false,false,129);
    for(bool stored:{false,true})for(char job:{'N','V'})exercise<B,double>(h,stored,'U',job,0);
#if defined(DDLA_GPUBLAS_HAS_HEURISTICS_CACHE) && DDLA_GPUBLAS_HAS_HEURISTICS_CACHE
    if constexpr(B==DdlaBackend::GPU)cache_policy(h);
#endif
    if(h->myid==0)std::cout<<"workspace one-stage PASS backend="<<int(B)<<" grid="<<pr<<"x"<<pc<<" major="<<major<<std::endl;
    ddlaDestroy(h);
}
}
int main(int argc,char** argv)
{
    MPI_Init(&argc,&argv);int size;MPI_Comm_size(MPI_COMM_WORLD,&size);TestOptions options;std::string error;
    if(!parse_options(argc,argv,size,options,error)){std::cerr<<error<<std::endl;MPI_Abort(MPI_COMM_WORLD,2);}
    // Exercise D&C merges/resources on these small numerical problems.
    setenv("DDLA_EIG_DC_LEAF","4",1);
    for(auto grid:options.grids)for(char major:{'R','C'}) {
        run<DdlaBackend::CPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#if DDLA_HAS_GPU
        run<DdlaBackend::GPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#endif
    }
    MPI_Finalize();return 0;
}
