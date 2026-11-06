#include "api_grid_test_common.h"
#include "ddla_stream.h"
#include "mpi_datatype.h"
#include "workspace_test_common.h"
#include "SVD/pgesvd_workspace.h"
#include <ddla/ddla_vector.h>
#include <array>
#include <cstring>
#include <optional>

using namespace api_grid_test;
namespace {
using namespace ddla::spectral_impl;
// Exactly the queried bytes of both kinds, surrounded by guards. The same
// addresses serve every execution in a case.
class GuardedScratch {
    static constexpr std::size_t guard=256;
    DdlaHandle_t h_;std::size_t backend_bytes_,host_bytes_;ddla::vector<char> backend_;std::vector<char> host_;
public:
    GuardedScratch(DdlaHandle_t h,std::size_t backend_bytes,std::size_t host_bytes)
        :h_(h),backend_bytes_(backend_bytes),host_bytes_(host_bytes),
         backend_(h,backend_bytes+2*guard),host_(host_bytes+2*guard,char(0x5a))
    {DDLA_CHECK(ddlaMemset(h,backend_.data(),0x5a,backend_.size()));}
    void* backend(){return backend_.data()+guard;}
    void* host(){return host_.data()+guard;}
    void verify() {
        std::array<char,2*guard> data{};
        DDLA_CHECK(ddlaMemcpy(h_,data.data(),backend_.data(),guard));
        DDLA_CHECK(ddlaMemcpy(h_,data.data()+guard,backend_.data()+guard+backend_bytes_,guard));DDLA_CHECK(ddlaSynchronize(h_));
        bool valid=true;for(std::size_t i=0;i<guard;++i)valid=valid&&host_[i]==char(0x5a)&&
            host_[guard+host_bytes_+i]==char(0x5a)&&data[i]==char(0x5a)&&data[guard+i]==char(0x5a);
        require_close(h_,"pgesvd exact workspace guards",valid?0:1,0);
    }
};
template<class S>bool no_allocation(const S& s){return !s.ddla_alloc&&!s.ddla_free&&!s.pinned_alloc&&!s.raw_gpu_alloc&&!s.host_new&&!s.vendor_query&&!s.resource_setup;}
// A tunnel build stages GPU messages in the handle's host buffers, which grow
// on first use (comm_traits.h): there the first run of a case may allocate.
template<DdlaBackend B,class S>bool run_allocates_nothing(const S& s,int pass)
{
#if defined(DDLA_USE_GPU_CPU_TUNNEL)
    if(B==DdlaBackend::GPU&&pass==0)return true;
#endif
    (void)pass;return no_allocation(s);
}
template<class F>void query(DdlaHandle_t h,F&& function) {
    MPI_Barrier(h->comm);bool pure=true;
    if(!h->myid){workspace_test::Scope scope(true);function();const auto s=scope.finish();
        pure=no_allocation(s)&&!s.copies&&!s.launches&&!s.syncs&&!s.collectives;}
    MPI_Barrier(h->comm);require_close(h,"pgesvd one-rank query purity",pure?0:1,0);function();
}
template<class T>T value(int i,int j) {
    using R=typename real_type<T>::type;const R re=R(((i+3)*17+(j+7)*31+(i*j+1)*13)%101-50)/R(101);
    if constexpr(std::is_same<T,R>::value)return re;
    else return T(re,R(((i+2)*11+(j+1)*7)%23-11)/R(29));
}
template<class T>T conjugate(T x){if constexpr(std::is_same<T,typename real_type<T>::type>::value)return x;else return std::conj(x);}
template<class T>std::vector<T> global(DdlaHandle_t h,const std::vector<T>& local,const int* desc,int m,int n) {
    std::vector<T> result(std::size_t(m)*n,T(0));
    const int rows=num_loc(desc[DDLA_M_],desc[DDLA_MB_],h->myprow_,desc[DDLA_RSRC_],h->nprows_);
    const int cols=num_loc(desc[DDLA_N_],desc[DDLA_NB_],h->mypcol_,desc[DDLA_CSRC_],h->npcols_);
    for(int j=0;j<cols;++j){const int gj=indxl2g(j,desc[DDLA_NB_],h->mypcol_,desc[DDLA_CSRC_],h->npcols_);if(gj>=n)continue;
        for(int i=0;i<rows;++i){const int gi=indxl2g(i,desc[DDLA_MB_],h->myprow_,desc[DDLA_RSRC_],h->nprows_);
            if(gi<m)result[gi+std::size_t(gj)*m]=local[i+std::size_t(j)*desc[DDLA_LLD_]];}}
    if(m&&n)MPI_Allreduce(MPI_IN_PLACE,result.data(),m*n,detail::mpi_datatype<T>(),MPI_SUM,h->comm);return result;
}
template<class T>bool padding(DdlaHandle_t h,const std::vector<T>& actual,const std::vector<T>& expected,const int* desc,int m,int n) {
    const int cols=num_loc(desc[DDLA_N_],desc[DDLA_NB_],h->mypcol_,desc[DDLA_CSRC_],h->npcols_);
    const int rows=num_loc(desc[DDLA_M_],desc[DDLA_MB_],h->myprow_,desc[DDLA_RSRC_],h->nprows_);
    for(int j=0;j<cols;++j){const int gj=indxl2g(j,desc[DDLA_NB_],h->mypcol_,desc[DDLA_CSRC_],h->npcols_);
        for(int i=0;i<desc[DDLA_LLD_];++i){const int gi=i<rows?indxl2g(i,desc[DDLA_MB_],h->myprow_,desc[DDLA_RSRC_],h->nprows_):desc[DDLA_M_];
            if((gi>=m||gj>=n)&&actual[i+std::size_t(j)*desc[DDLA_LLD_]]!=expected[i+std::size_t(j)*desc[DDLA_LLD_]])return false;}}
    return true;
}
template<DdlaBackend B,class T>void svd_case(DdlaHandle_t h,int m,int n,char ju,char jv) {
    using R=typename real_type<T>::type;const int k=std::min(m,n);const bool wantu=ju=='V',wantv=jv=='V';
    int da[DDLA_DLEN_],du[DDLA_DLEN_],dv[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(da,h,m+2,n+3,4,4,0,0));da[DDLA_LLD_]+=2+h->myid;
    DDLA_CHECK(ddlaDescInit(du,h,m+2,k+2,4,4,0,0));du[DDLA_LLD_]+=3+h->myid;
    DDLA_CHECK(ddlaDescInit(dv,h,k+2,n+2,4,4,0,0));dv[DDLA_LLD_]+=4+h->myid;
    auto original=make_local<T>(h,da,[](int i,int j){return value<T>(i,j);});
    auto initial_u=make_local<T>(h,du,[](int,int){return T(-717);});auto initial_v=make_local<T>(h,dv,[](int,int){return T(-818);});
    DeviceBuffer<T>A(h,original.size()),U(h,initial_u.size()),V(h,initial_v.size());std::vector<R> values(k);
    setenv("DDLA_SVD_DC_LEAF","4",1);
    if constexpr(B==DdlaBackend::CPU && std::is_same<T,double>::value) {
        if(m==13 && n==9 && wantu && wantv) {
            unsetenv("DDLA_SVD_DC_THREADS");
            const auto defaults=pgesvd_check<T>(h,B,ju,jv,m,n,A.ptr,da,values.data(),U.ptr,du,V.ptr,dv);
            require_close(h,"SVD D&C respects the bound rank worker budget",
                          std::abs(defaults.dc.threads-std::min(8,h->host_threads)),0);
        }
    }
    setenv("DDLA_SVD_DC_THREADS","3",1);
    const auto p=pgesvd_check<T>(h,B,ju,jv,m,n,A.ptr,da,values.data(),wantu?U.ptr:nullptr,wantu?du:nullptr,wantv?V.ptr:nullptr,wantv?dv:nullptr);
    std::size_t backend_bytes=0,host_bytes=0;query(h,[&]{pgesvd_workspace_size<T>(h,p,&backend_bytes,&host_bytes);});
    GuardedScratch scratch(h,backend_bytes,host_bytes);
    std::optional<detail::DcHostTeam> team;if(const int threads=pgesvd_team_threads(p))team.emplace(threads);
    // The query/run contract freezes the distributed solver's leaf policy.
    setenv("DDLA_SVD_DC_LEAF","128",1);
    std::vector<R> first_values;std::vector<T> first_a,first_u,first_v;
    for(int repeat=0;repeat<2;++repeat) {
        upload(h,A.ptr,original);upload(h,U.ptr,initial_u);upload(h,V.ptr,initial_v);int info=-1;
        workspace_test::Scope scope;
        pgesvd_run<B,T>(h,p,team?&*team:nullptr,A.ptr,values.data(),wantu?U.ptr:nullptr,wantv?V.ptr:nullptr,info,
            scratch.backend(),backend_bytes,scratch.host(),host_bytes);
        const auto counts=scope.finish();
        require_close(h,"pgesvd raw allocation/query/resource count",run_allocates_nothing<B>(counts,repeat)?0:1,0);
        require_close(h,"pgesvd info",info?1:0,0);
        auto a=download(h,A.ptr,original.size()),u=download(h,U.ptr,initial_u.size()),v=download(h,V.ptr,initial_v.size());
        require_close(h,"pgesvd input parent/padding",padding(h,a,original,da,m,n)?0:1,0);
        require_close(h,"pgesvd U parent/padding",padding(h,u,initial_u,du,wantu?m:0,wantu?k:0)?0:1,0);
        require_close(h,"pgesvd VT parent/padding",padding(h,v,initial_v,dv,wantv?k:0,wantv?n:0)?0:1,0);
        if(!repeat){first_values=values;first_a=a;first_u=u;first_v=v;}
        else require_close(h,"pgesvd same-scratch bitwise repeat",(k&&std::memcmp(values.data(),first_values.data(),k*sizeof(R)))||
            std::memcmp(a.data(),first_a.data(),a.size()*sizeof(T))||std::memcmp(u.data(),first_u.data(),u.size()*sizeof(T))||
            std::memcmp(v.data(),first_v.data(),v.size()*sizeof(T))?1:0,0);
        double norm_a=0,norm_s=0;for(int j=0;j<n;++j)for(int i=0;i<m;++i)norm_a+=std::norm(value<T>(i,j));
        bool sorted=true;for(int j=0;j<k;++j){norm_s+=double(values[j])*values[j];sorted=sorted&&std::isfinite(values[j])&&values[j]>=R(0)&&(j==0||values[j-1]>=values[j]);}
        const double tol=std::is_same<R,float>::value?8e-4:8e-11;
        require_close(h,"pgesvd singular-value norm",workspace_test::relative_error(norm_s,norm_a),tol);
        require_close(h,"pgesvd descending nonnegative values",sorted?0:1,0);
        auto gu=wantu?global(h,u,du,m,k):std::vector<T>{};auto gv=wantv?global(h,v,dv,k,n):std::vector<T>{};
        double orthogonal=0,residual=0;
        for(int j=0;j<k;++j)for(int i=0;i<k;++i) {
            T up=0,vp=0;if(wantu)for(int r=0;r<m;++r)up+=conjugate(gu[r+std::size_t(i)*m])*gu[r+std::size_t(j)*m];
            if(wantv)for(int c=0;c<n;++c)vp+=gv[i+std::size_t(c)*k]*conjugate(gv[j+std::size_t(c)*k]);
            if(wantu)orthogonal=std::max(orthogonal,workspace_test::finite_error(double(std::abs(up-T(i==j)))));
            if(wantv)orthogonal=std::max(orthogonal,workspace_test::finite_error(double(std::abs(vp-T(i==j)))));
        }
        if(wantu&&wantv)for(int j=0;j<n;++j)for(int i=0;i<m;++i) {
            T product=0;for(int l=0;l<k;++l)product+=gu[i+std::size_t(l)*m]*values[l]*gv[l+std::size_t(j)*k];
            residual=std::max(residual,workspace_test::finite_error(double(std::abs(product-value<T>(i,j)))));
        }
        require_close(h,"pgesvd output orthogonality",orthogonal,tol);require_close(h,"pgesvd reconstruction",residual,tol);scratch.verify();
    }
    unsetenv("DDLA_SVD_DC_LEAF");unsetenv("DDLA_SVD_DC_THREADS");
}
template<DdlaBackend B>void checker_cases(DdlaHandle_t h) {
    constexpr int m=13,n=9;int da[DDLA_DLEN_],du[DDLA_DLEN_],dv[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(da,h,m+2,n+2,4,4,0,0));DDLA_CHECK(ddlaDescInit(du,h,m+2,n+2,4,4,0,0));
    DDLA_CHECK(ddlaDescInit(dv,h,n+2,n+2,4,4,0,0));
    auto a=make_local<double>(h,da,[](int i,int j){return value<double>(i,j);});
    auto u=make_local<double>(h,du,[](int,int){return -718.;}),v=make_local<double>(h,dv,[](int,int){return -819.;});
    DeviceBuffer<double>A(h,a.size()),U(h,u.size()),V(h,v.size());upload(h,A.ptr,a);upload(h,U.ptr,u);upload(h,V.ptr,v);
    std::vector<double>w(n,-917.);
    const auto rejected=[&](auto&& call){bool caught=false;try{call();}catch(const std::exception&){caught=true;}
        require_close(h,"pgesvd invalid metadata rejected on every rank",caught?0:1,0);};
    int small[DDLA_DLEN_];std::copy_n(du,DDLA_DLEN_,small);if(!h->myid)small[DDLA_M_]=m-1;
    rejected([&]{(void)pgesvd_check<double>(h,B,'V','V',m,n,A.ptr,da,w.data(),U.ptr,small,V.ptr,dv);});
    rejected([&]{(void)pgesvd_check<double>(h,B,'V',!h->myid?'X':'V',m,n,A.ptr,da,w.data(),U.ptr,du,V.ptr,dv);});
    if(h->nprocs>1) {
        rejected([&]{(void)pgesvd_check<double>(h,B,'V','V',!h->myid?m-1:m,n,A.ptr,da,w.data(),U.ptr,du,V.ptr,dv);});
        rejected([&]{(void)pgesvd_check<double>(h,B,'N','N',!h->myid?0:m,n,A.ptr,da,w.data(),nullptr,nullptr,nullptr,nullptr);});
    }
    const auto after_a=download(h,A.ptr,a.size()),after_u=download(h,U.ptr,u.size()),after_v=download(h,V.ptr,v.size());
    const bool unchanged=!std::memcmp(a.data(),after_a.data(),a.size()*sizeof(double))&&
        !std::memcmp(u.data(),after_u.data(),u.size()*sizeof(double))&&!std::memcmp(v.data(),after_v.data(),v.size()*sizeof(double))&&
        std::all_of(w.begin(),w.end(),[](double value){return value==-917.;});
    require_close(h,"pgesvd checks preserve numerical inputs and outputs",unchanged?0:1,0);
}
template<DdlaBackend B>void run(MPI_Comm comm,int pr,int pc,char major) {
    DdlaHandle_t h=nullptr;ddlaInit(h,B);ddlaSet(h,comm,pr,pc,major);checker_cases<B>(h);
#define CASES(T) \
    svd_case<B,T>(h,13,9,'V','V');svd_case<B,T>(h,9,13,'V','V'); \
    svd_case<B,T>(h,13,9,'V','N');svd_case<B,T>(h,13,9,'N','V');svd_case<B,T>(h,13,9,'N','N'); \
    svd_case<B,T>(h,1,1,'V','V')
    CASES(float);CASES(double);CASES(std::complex<float>);CASES(std::complex<double>);
#undef CASES
    svd_case<B,double>(h,0,9,'N','N');
    if(!h->myid)std::cout<<"workspace pgesvd PASS backend="<<int(B)<<" grid="<<pr<<"x"<<pc<<" major="<<major
        <<" allocation_trace="<<(workspace_test::available()?"enabled":"numerical-only")<<std::endl;
    ddlaDestroy(h);
}
}
int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);int size=1;MPI_Comm_size(MPI_COMM_WORLD,&size);TestOptions options;std::string error;
    if(!parse_options(argc,argv,size,options,error)){std::cerr<<error<<std::endl;MPI_Abort(MPI_COMM_WORLD,2);}
    for(auto grid:options.grids)for(char major:{'R','C'}) {
        run<DdlaBackend::CPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#if DDLA_HAS_GPU
        run<DdlaBackend::GPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#endif
    }
    MPI_Finalize();return 0;
}
