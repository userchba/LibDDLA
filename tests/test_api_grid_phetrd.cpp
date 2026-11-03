#include "api_grid_test_common.h"
#include "workspace_test_common.h"
#include "ddla_stream.h"
#include "Eigensolver/one_stage/phetrd_impl.h"
#include "Eigensolver/one_stage/punmtr_trd_impl.h"
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <type_traits>
using namespace api_grid_test;
namespace {
using cd=std::complex<double>;
template<class T>T scalar(double r,double i=0){if constexpr(std::is_arithmetic_v<T>)return T(r);else return T(r,i);}
template<class T>cd wide(T v){return {double(std::real(v)),double(std::imag(v))};}
template<class T>T entry(int i,int j){
    if(i==j)return T(2+.07*i);
    if(i>j){const auto v=entry<T>(j,i);if constexpr(std::is_arithmetic_v<T>)return v;else return std::conj(v);}
    return scalar<T>(((i*13+j*7)%17-8)/31.,((i*5+j*11)%19-9)/37.);
}
template<class T>T centry(int i,int j){return scalar<T>(((i*7+j*11)%23-11)/17.,((i*3+j*5)%13-6)/19.);}
bool clean(const workspace_test::Snapshot& s){return !s.ddla_alloc&&!s.ddla_free&&!s.pinned_alloc&&!s.raw_gpu_alloc&&!s.host_new&&!s.vendor_query&&!s.resource_setup;}
bool pure(const workspace_test::Snapshot& s){return clean(s)&&!s.copies&&!s.launches&&!s.syncs&&!s.collectives;}
class Guarded {
    DdlaHandle_t h_;std::size_t nb_,nh_;ddla::vector<unsigned char>b_;std::vector<unsigned char>host_;
public:
    Guarded(DdlaHandle_t h,std::size_t nb,std::size_t nh=0):h_(h),nb_(nb),nh_(nh),b_(h,nb+512),host_(nh+512,0x69){
        DDLA_CHECK(ddlaMemset(h,b_.data(),0x69,b_.size()));check_ddla_sync(h);
    }
    void* backend(){return b_.data()+256;}void* host(){return host_.data()+256;}
    void verify(){std::array<unsigned char,512>x{};DDLA_CHECK(ddlaMemcpy(h_,x.data(),b_.data(),256));
        DDLA_CHECK(ddlaMemcpy(h_,x.data()+256,b_.data()+256+nb_,256));check_ddla_sync(h_);
        bool ok=true;for(int i=0;i<256;++i)ok=ok&&x[i]==0x69&&x[i+256]==0x69&&host_[i]==0x69&&host_[256+nh_+i]==0x69;
        require_close(h_,"tridiagonal exact scratch guards",ok?0:1,0);
    }
};
template<class T>std::vector<T> gather_triangle(DdlaHandle_t h,const std::vector<T>& a,const int* d,int n,char uplo){
    std::vector<T> full(std::size_t(n)*n,T(0));
    for(int j=0;j<ddla_test::n_loc(h,d);++j)for(int i=0;i<ddla_test::m_loc(h,d);++i){
        int gi=indxl2g(i,d[DDLA_MB_],h->myprow_,0,h->nprows_),gj=indxl2g(j,d[DDLA_NB_],h->mypcol_,0,h->npcols_);
        if(gi<n&&gj<n&&(uplo=='U'?gi<=gj:gi>=gj))full[gi+std::size_t(gj)*n]=a[i+std::size_t(j)*d[DDLA_LLD_]];
    }
    if(n)MPI_Allreduce(MPI_IN_PLACE,full.data(),n*n,ddla::detail::mpi_datatype<T>(),MPI_SUM,h->comm);return full;
}
template<class T>std::vector<cd> form_q(int n,char uplo,const std::vector<T>& a,const std::vector<T>& tau){
    std::vector<cd> q(std::size_t(n)*n),v(n);for(int i=0;i<n;++i)q[i+std::size_t(i)*n]=1;
    // Independent scalar composition of H_j^H; no library application helper.
    for(int j=1;j<n;++j){
        std::fill(v.begin(),v.end(),cd(0));v[j-1]=1;
        for(int i=0;i<j-1;++i)v[i]=uplo=='U'?wide(a[i+std::size_t(j)*n]):std::conj(wide(a[j+std::size_t(i)*n]));
        for(int c=0;c<n;++c){cd dot=0;for(int i=0;i<n;++i)dot+=std::conj(v[i])*q[i+std::size_t(c)*n];
            for(int i=0;i<n;++i)q[i+std::size_t(c)*n]-=std::conj(wide(tau[j-1]))*v[i]*dot;}
    }return q;
}
template<DdlaBackend B,class T>void application(DdlaHandle_t h,char side,char uplo,char trans,int nq,int nb,
    const T* A,const int* da,const T* tau,const std::vector<cd>& q,bool raw){
    using R=typename real_type<T>::type;
    const int m=side=='L'?nq:5,n=side=='L'?5:nq;
    int dc[DDLA_DLEN_];ddlaDescInit(dc,h,m+2,n+3,side=='L'?nb:2,side=='L'?3:nb,
        side=='L'?0:h->nprows_-1,side=='L'?h->npcols_-1:0);dc[DDLA_LLD_]+=3+h->myid%2;
    const T sentinel=scalar<T>(-937,17);
    auto input=make_local<T>(h,dc,[&](int i,int j){return i<m&&j<n?centry<T>(i,j):sentinel;});
    for(int j=0;j<ddla_test::n_loc(h,dc);++j)for(int i=ddla_test::m_loc(h,dc);i<dc[DDLA_LLD_];++i)input[i+std::size_t(j)*dc[DDLA_LLD_]]=sentinel;
    DeviceBuffer<T>C(h,input.size());upload(h,C.ptr,input);
    std::vector<cd> expected(std::size_t(m)*n);
    auto op=[&](int i,int j){return trans=='N'?q[i+std::size_t(j)*nq]:std::conj(q[j+std::size_t(i)*nq]);};
    for(int j=0;j<n;++j)for(int i=0;i<m;++i)for(int k=0;k<nq;++k)
        expected[i+std::size_t(j)*m]+=side=='L'?op(i,k)*wide(centry<T>(k,j)):wide(centry<T>(i,k))*op(k,j);
    auto verify=[&]{auto result=download(h,C.ptr,input.size());double error=0,bad=0;
        for(int j=0;j<ddla_test::n_loc(h,dc);++j)for(int i=0;i<dc[DDLA_LLD_];++i){
            int gi=indxl2g(i,dc[DDLA_MB_],h->myprow_,dc[DDLA_RSRC_],h->nprows_),gj=indxl2g(j,dc[DDLA_NB_],h->mypcol_,dc[DDLA_CSRC_],h->npcols_);
            const auto at=i+std::size_t(j)*dc[DDLA_LLD_];
            if(i<ddla_test::m_loc(h,dc)&&gi<m&&gj<n){const cd e=expected[gi+std::size_t(gj)*m];error=std::max(error,workspace_test::finite_error(std::abs(wide(result[at])-e)/(1+std::abs(e))));}
            else if(std::memcmp(&result[at],&input[at],sizeof(T)))bad=1;
        }
        require_close(h,"stored punmtr independent dense Q",error,100*std::max(1,nq)*std::numeric_limits<R>::epsilon());
        require_close(h,"stored punmtr output parent/padding",bad,0);
    };
    punmtr<B,T>(h,side,uplo,trans,m,n,A,da,tau,C.ptr,dc);check_ddla_sync(h);verify();
    if(!raw)return;
    const auto p=spectral_impl::punmtr_trd_check<T>(h,B,side,uplo,trans,m,n,A,da,tau,C.ptr,dc);
    std::size_t bytes=0;workspace_test::Scope query(true);spectral_impl::punmtr_trd_workspace_size<T>(h,p,&bytes);
    require_close(h,"stored punmtr pure query",pure(query.finish())?0:1,0);
    Guarded scratch(h,bytes);
    spectral_impl::punmtr_trd_run<B,T>(h,p,A,tau,C.ptr,scratch.backend(),bytes);check_ddla_sync(h);
    for(int pass=0;pass<2;++pass){upload(h,C.ptr,input);check_ddla_sync(h);workspace_test::Scope scope;
        spectral_impl::punmtr_trd_run<B,T>(h,p,A,tau,C.ptr,scratch.backend(),bytes);
        require_close(h,"stored punmtr raw run allocation/query/setup",clean(scope.finish())?0:1,0);check_ddla_sync(h);verify();scratch.verify();}
}
template<DdlaBackend B,class T>void reduction(DdlaHandle_t h,int n,int nb,char uplo){
    using R=typename real_type<T>::type;const R nan=std::numeric_limits<R>::quiet_NaN();const T sentinel=scalar<T>(-739,23);
    int da[DDLA_DLEN_];ddlaDescInit(da,h,n+2,n+3,nb,nb,0,0);da[DDLA_LLD_]+=2+h->myid%2;
    auto input=make_local<T>(h,da,[&](int i,int j){if(i>=n||j>=n)return sentinel;
        if(uplo=='U'?i>j:i<j)return scalar<T>(nan,nan);
        if constexpr(!std::is_arithmetic_v<T>)if(i==j)return T(std::real(entry<T>(i,j)),nan);
        return entry<T>(i,j);});
    for(int j=0;j<ddla_test::n_loc(h,da);++j)for(int i=ddla_test::m_loc(h,da);i<da[DDLA_LLD_];++i)input[i+std::size_t(j)*da[DDLA_LLD_]]=sentinel;
    DeviceBuffer<T>A(h,input.size()),Tau(h,std::size_t(std::max(0,n-1))+2);
    std::vector<T> tau_before(Tau.count,sentinel);upload(h,Tau.ptr,tau_before);upload(h,A.ptr,input);
    std::vector<R>d(std::size_t(n)+2,R(-731)),e(std::size_t(std::max(0,n-1))+2,R(-733));int info=-1;
    phetrd<B,T>(h,uplo,n,n?A.ptr:nullptr,da,n?d.data()+1:nullptr,n>1?e.data()+1:nullptr,n>1?Tau.ptr+1:nullptr,info);
    require_close(h,"phetrd info",info,0);check_ddla_sync(h);
    auto after=download(h,A.ptr,input.size()),tau_guarded=download(h,Tau.ptr,Tau.count);
    require_close(h,"phetrd scalar guards",d.front()!=R(-731)||d.back()!=R(-731)||e.front()!=R(-733)||e.back()!=R(-733)||
        std::memcmp(&tau_guarded.front(),&sentinel,sizeof(T))||std::memcmp(&tau_guarded.back(),&sentinel,sizeof(T))?1:0,0);
    auto verify_storage=[&](const std::vector<T>& actual){double bad=0;
        for(int j=0;j<ddla_test::n_loc(h,da);++j)for(int i=0;i<da[DDLA_LLD_];++i){
            int gi=indxl2g(i,nb,h->myprow_,0,h->nprows_),gj=indxl2g(j,nb,h->mypcol_,0,h->npcols_);const auto at=i+std::size_t(j)*da[DDLA_LLD_];
            if(i>=ddla_test::m_loc(h,da)||gi>=n||gj>=n||(uplo=='U'?gi>gj:gi<gj))if(std::memcmp(&actual[at],&input[at],sizeof(T)))bad=1;
        }require_close(h,"phetrd preserves opposite triangle/parent/padding",bad,0);
    };verify_storage(after);
    auto full=gather_triangle(h,after,da,n,uplo);std::vector<T> tau(tau_guarded.begin()+1,tau_guarded.end()-1);
    auto q=form_q(n,uplo,full,tau);std::vector<cd> aq(std::size_t(n)*n);
    for(int j=0;j<n;++j)for(int i=0;i<n;++i)for(int k=0;k<n;++k)aq[i+std::size_t(j)*n]+=wide(entry<T>(i,k))*q[k+std::size_t(j)*n];
    double residual=0,orth=0;
    for(int j=0;j<n;++j)for(int i=0;i<n;++i){cd v=0,o=0;
        for(int k=0;k<n;++k){v+=std::conj(q[k+std::size_t(i)*n])*aq[k+std::size_t(j)*n];o+=std::conj(q[k+std::size_t(i)*n])*q[k+std::size_t(j)*n];}
        const double target=i==j?d[i+1]:std::abs(i-j)==1?e[std::min(i,j)+1]:0;
        residual=std::max(residual,workspace_test::finite_error(std::abs(v-target)/(1+std::abs(target))));
        orth=std::max(orth,workspace_test::finite_error(std::abs(o-cd(i==j))));
    }
    const double tol=200*std::max(1,n)*std::numeric_limits<R>::epsilon();
    require_close(h,"phetrd independent Q^H A Q tridiagonal",residual,tol);require_close(h,"phetrd independent Q orthogonality",orth,tol);
    std::vector<R>dmin=d,dmax=d,emin=e,emax=e;
    MPI_Allreduce(MPI_IN_PLACE,dmin.data(),int(d.size()),ddla::detail::mpi_datatype<R>(),MPI_MIN,h->comm);
    MPI_Allreduce(MPI_IN_PLACE,dmax.data(),int(d.size()),ddla::detail::mpi_datatype<R>(),MPI_MAX,h->comm);
    MPI_Allreduce(MPI_IN_PLACE,emin.data(),int(e.size()),ddla::detail::mpi_datatype<R>(),MPI_MIN,h->comm);
    MPI_Allreduce(MPI_IN_PLACE,emax.data(),int(e.size()),ddla::detail::mpi_datatype<R>(),MPI_MAX,h->comm);
    require_close(h,"phetrd replicated d/e",dmin==dmax&&emin==emax?0:1,0);
    const bool raw=n==9&&nb==4;
    if(raw){const auto p=spectral_impl::phetrd_check<T>(h,B,uplo,n,A.ptr,da,d.data()+1,e.data()+1,Tau.ptr+1);
        std::size_t bytes=0,host=0;bool one=true;MPI_Barrier(h->comm);
        if(!h->myid){workspace_test::Scope scope(true);spectral_impl::phetrd_workspace_size<T>(h,p,&bytes,&host);one=pure(scope.finish());}
        MPI_Barrier(h->comm);require_close(h,"phetrd one-rank local query",one?0:1,0);
        spectral_impl::phetrd_workspace_size<T>(h,p,&bytes,&host);Guarded scratch(h,bytes,host);spectral_impl::HetrdResources resources(h,p);
        std::vector<R> de(std::size_t(2)*n);upload(h,A.ptr,input);
        spectral_impl::phetrd_run<B,T>(h,p,resources,A.ptr,Tau.ptr+1,de.data(),info,scratch.backend(),bytes,scratch.host(),host);check_ddla_sync(h);
        for(int pass=0;pass<2;++pass){upload(h,A.ptr,input);check_ddla_sync(h);workspace_test::Scope scope;
            spectral_impl::phetrd_run<B,T>(h,p,resources,A.ptr,Tau.ptr+1,de.data(),info,scratch.backend(),bytes,scratch.host(),host);
            require_close(h,"phetrd raw run allocation/query/setup",clean(scope.finish())?0:1,0);
            double error=0;for(int j=0;j<n;++j)error=std::max(error,workspace_test::finite_error(std::abs(double(de[j]-d[j+1]))));
            for(int j=0;j<n-1;++j)error=std::max(error,workspace_test::finite_error(std::abs(double(de[n+j]-e[j+1]))));
            require_close(h,"phetrd raw/public agreement",error,tol);scratch.verify();verify_storage(download(h,A.ptr,input.size()));
        }
    }
    const auto before_apply=download(h,A.ptr,input.size()),before_tau=download(h,Tau.ptr,Tau.count);
    for(char side:{'L','R'})for(char trans:{'N','C'})application<B,T>(h,side,uplo,trans,n,nb,A.ptr,da,Tau.ptr+1,q,raw);
    if constexpr(std::is_arithmetic_v<T>)if(raw)application<B,T>(h,'R',uplo,'T',n,nb,A.ptr,da,Tau.ptr+1,q,false);
    auto final_a=download(h,A.ptr,input.size()),final_tau=download(h,Tau.ptr,Tau.count);
    require_close(h,"stored punmtr readonly A/tau",std::memcmp(before_apply.data(),final_a.data(),input.size()*sizeof(T))||
        std::memcmp(before_tau.data(),final_tau.data(),Tau.count*sizeof(T))?1:0,0);
}
template<DdlaBackend B>void invalid(DdlaHandle_t h){
    int da[DDLA_DLEN_],dc[DDLA_DLEN_];ddlaDescInit(da,h,11,11,3,3,0,0);ddlaDescInit(dc,h,11,11,3,3,0,0);
    DeviceBuffer<double>A(h,512),Tau(h,20),C(h,512);std::vector<double>d(11),e(10);int info=0;
    auto rejected=[&](auto&& f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}require_close(h,"tridiagonal API invalid metadata rejected collectively",caught?0:1,0);};
    rejected([&]{phetrd<B,double>(h,'X',9,A.ptr,da,d.data(),e.data(),Tau.ptr,info);});
    rejected([&]{phetrd<B,double>(h,'U',-1,A.ptr,da,d.data(),e.data(),Tau.ptr,info);});
    rejected([&]{phetrd<B,double>(h,'U',12,A.ptr,da,d.data(),e.data(),Tau.ptr,info);});
    rejected([&]{phetrd<B,double>(h,'U',9,!h->myid?nullptr:A.ptr,da,d.data(),e.data(),Tau.ptr,info);});
    rejected([&]{phetrd<B,double>(h,'U',9,A.ptr,da,!h->myid?nullptr:d.data(),e.data(),Tau.ptr,info);});
    rejected([&]{phetrd<B,double>(h,'U',9,A.ptr,da,d.data(),!h->myid?nullptr:e.data(),Tau.ptr,info);});
    rejected([&]{phetrd<B,double>(h,'U',9,A.ptr,da,d.data(),e.data(),!h->myid?nullptr:Tau.ptr,info);});
    auto apply=[&](char side,char uplo,char trans,int m,int n,const double* a,const double* tau,double* c,const int* desc){punmtr<B,double>(h,side,uplo,trans,m,n,a,da,tau,c,desc);};
    rejected([&]{apply('X','U','N',9,7,A.ptr,Tau.ptr,C.ptr,dc);});
    rejected([&]{apply('L','X','N',9,7,A.ptr,Tau.ptr,C.ptr,dc);});
    rejected([&]{apply('L','U','X',9,7,A.ptr,Tau.ptr,C.ptr,dc);});
    rejected([&]{apply('L','U','N',12,7,A.ptr,Tau.ptr,C.ptr,dc);});
    rejected([&]{apply('L','U','N',9,7,A.ptr,!h->myid?nullptr:Tau.ptr,C.ptr,dc);});
    rejected([&]{apply('L','U','N',9,7,A.ptr,Tau.ptr,A.ptr,dc);});
    int bad[DDLA_DLEN_];std::copy_n(dc,DDLA_DLEN_,bad);bad[DDLA_MB_]=2;
    rejected([&]{apply('L','U','N',9,7,A.ptr,Tau.ptr,C.ptr,bad);});
    if(h->nprocs>1){
        rejected([&]{phetrd<B,double>(h,'U',!h->myid?0:9,A.ptr,da,d.data(),e.data(),Tau.ptr,info);});
        rejected([&]{apply('L','U','N',!h->myid?0:9,7,A.ptr,Tau.ptr,C.ptr,dc);});
        rejected([&]{if(!h->myid)punmtr<B,double>(h,'L','U','N',0,0,0,nullptr,0,nullptr,0,nullptr,dc);
            else punmtr<B,double>(h,'L','U','N',0,0,nullptr,da,nullptr,nullptr,dc);});
    }
    punmtr<B,double>(h,'L','U','N',0,7,nullptr,da,nullptr,nullptr,dc);
    punmtr<B,double>(h,'R','L','C',7,0,nullptr,da,nullptr,nullptr,dc);
    DeviceBuffer<std::complex<double>>Z(h,128);
    rejected([&]{punmtr<B,std::complex<double>>(h,'L','U','T',0,0,nullptr,da,nullptr,Z.ptr,dc);});
#if DDLA_HAS_GPU
    constexpr auto other=B==DdlaBackend::CPU?DdlaBackend::GPU:DdlaBackend::CPU;
    rejected([&]{phetrd<other,double>(h,'U',0,nullptr,da,nullptr,nullptr,nullptr,info);});
#endif
}
template<DdlaBackend B,class T>void cases(DdlaHandle_t h){for(char uplo:{'U','L'})for(auto shape:{std::array<int,2>{9,4},{17,3},{5,16},{2,1},{1,3},{0,3}})reduction<B,T>(h,shape[0],shape[1],uplo);}

#if defined(DDLA_GPUBLAS_HAS_HEURISTICS_CACHE) && DDLA_GPUBLAS_HAS_HEURISTICS_CACHE
void check_cache_resources(DdlaHandle_t h) {
    if(std::getenv("CUBLASLT_HEURISTICS_CACHE_CAPACITY"))return;
    std::size_t saved=0,current=0;
    BLAS_CHECK(gpublasHeuristicsCacheGetCapacity(&saved));
    try {
        BLAS_CHECK(gpublasHeuristicsCacheSetCapacity(8192));
        spectral_impl::HetrdPlan p;p.backend=DdlaBackend::GPU;p.n=1024;
        int version=0;BLAS_CHECK(gpublasGetVersion(static_cast<gpublasHandle_t>(h->blasH),&version));
        const bool eligible=version>=111103&&version<120304;
        {
            spectral_impl::HetrdResources resources(h,p);resources.begin_reduction();
            BLAS_CHECK(gpublasHeuristicsCacheGetCapacity(&current));
            require_close(h,"reduction cache enters expected state",current==(eligible?0u:8192u)?0:1,0);
            resources.end_reduction();
        }
        BLAS_CHECK(gpublasHeuristicsCacheGetCapacity(&current));
        require_close(h,"reduction cache restored on success",current==8192?0:1,0);
        {
            spectral_impl::HetrdResources resources(h,p);resources.begin_reduction();
            // Leaving an active scope exercises cancellation/destructor cleanup.
        }
        BLAS_CHECK(gpublasHeuristicsCacheGetCapacity(&current));
        require_close(h,"reduction cache restored on cancellation",current==8192?0:1,0);
        if(!h->myid)std::cout<<"heuristics cache resources PASS eligible="<<eligible<<std::endl;
    } catch(...) {
        (void)gpublasHeuristicsCacheSetCapacity(saved);throw;
    }
    BLAS_CHECK(gpublasHeuristicsCacheSetCapacity(saved));
}
#endif

template<DdlaBackend B>void run(MPI_Comm comm,int pr,int pc,char major){
    DdlaHandle_t h=nullptr;ddlaInit(h,B);ddlaSet(h,comm,pr,pc,major);
#if defined(DDLA_GPUBLAS_HAS_HEURISTICS_CACHE) && DDLA_GPUBLAS_HAS_HEURISTICS_CACHE
    if constexpr(B==DdlaBackend::GPU)check_cache_resources(h);
#endif
    cases<B,float>(h);cases<B,double>(h);cases<B,std::complex<float>>(h);cases<B,std::complex<double>>(h);invalid<B>(h);
    if(!h->myid)std::cout<<"phetrd/stored-punmtr PASS backend="<<int(B)<<" grid="<<pr<<"x"<<pc<<" major="<<major<<std::endl;ddlaDestroy(h);
}
}
int main(int argc,char**argv){MPI_Init(&argc,&argv);int size=0;MPI_Comm_size(MPI_COMM_WORLD,&size);TestOptions options;std::string error;
    if(!parse_options(argc,argv,size,options,error)){std::cerr<<error<<std::endl;MPI_Abort(MPI_COMM_WORLD,2);}
    for(auto grid:options.grids)for(char major:{'R','C'}){
        run<DdlaBackend::CPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#if DDLA_HAS_GPU
        run<DdlaBackend::GPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#endif
    }MPI_Finalize();return 0;}
