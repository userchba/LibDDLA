#include "api_grid_test_common.h"
#include "eig_test_common.h"
#include "Eigensolver/two_stage/pheevd_impl.h"
#include <limits>
using namespace api_grid_test;
namespace {
template<class T> T entry(int i,int j) {
    using R=typename real_type<T>::type;
    if(i==j)return T(R(4)+R(i)/R(100));
    if constexpr(std::is_same<T,R>::value)return T(.1*std::sin(i+j));
    else return T(.1*std::sin(i+j),.1*std::sin(i-j));
}
template<DdlaBackend B,class T>void methods(DdlaHandle_t h) {
    using R=typename real_type<T>::type;
    constexpr int n=65;int d[DDLA_DLEN_];ddlaDescInit(d,h,n+3,n+5,8,8,0,0);d[DDLA_LLD_]+=3;
    const T sentinel=T(-917);std::vector<T> input(local_size(h,d),sentinel),blank(input);
    for(int j=0;j<ddla_test::n_loc(h,d);++j)for(int i=0;i<ddla_test::m_loc(h,d);++i) {
        int gi=indx_l2g_r(d,h,i),gj=indx_l2g_c(d,h,j);
        if(gi<n&&gj<n)input[i+std::size_t(j)*d[DDLA_LLD_]]=entry<T>(gi,gj);
    }
    DeviceBuffer<T>A(h,input.size()),Z(h,input.size());std::vector<R> reference(n),w(n),only(n);int info=-1;
    upload(h,A.ptr,input);pheevd_two_stage<B,T>(h,'N','U',n,A.ptr,d,reference.data(),nullptr,nullptr,info);
    require_close(h,"common methods reference info",info,0);
    auto topology=spectral_impl::phb2st_topology_prepare(h);
    for(char method:{'G','H'})for(DdlaBackend reduction:{DdlaBackend::CPU,DdlaBackend::GPU}) {
        if(B==DdlaBackend::CPU&&reduction==DdlaBackend::GPU)continue;
        const auto plan=spectral_impl::pheevd_check<T>(h,B,'V','U',n,A.ptr,d,w.data(),Z.ptr,d,*topology,method,nullptr,reduction);
        bool native=method=='H'?plan.householder.options.stage2==detail::Hb2stHouseholderCpuOptions::Stage2::Gpu:plan.chase.device_chase;
        require_close(h,"common method and reduction placement",
            (plan.engine==spectral_impl::PheevdEngine::Householder)==(method=='H')&&
            native==(reduction==DdlaBackend::GPU)&&!plan.automatic_householder?0:1,0);
        upload(h,A.ptr,input);upload(h,Z.ptr,blank);
        if(method=='G')detail::pheevd_givens<B,T>(h,'V','U',n,A.ptr,d,w.data(),Z.ptr,d,info,reduction);
        else detail::pheevd_householder<B,T>(h,'V','U',n,A.ptr,d,w.data(),Z.ptr,d,info,reduction);
        require_close(h,"common method vectors info",info,0);
        const auto output=download(h,Z.ptr,input.size());bool pad=true;
        for(int j=0;j<ddla_test::n_loc(h,d);++j)for(int i=0;i<d[DDLA_LLD_];++i) {
            const int gi=i<ddla_test::m_loc(h,d)?indx_l2g_r(d,h,i):n,gj=indx_l2g_c(d,h,j);
            if(gi>=n||gj>=n)pad=pad&&output[i+std::size_t(j)*d[DDLA_LLD_]]==sentinel;
        }
        require_close(h,"common method output padding",pad?0:1,0);
        upload(h,A.ptr,input);
        double residual=eig_test::residual_metric<T,B>(h,A.ptr,d,Z.ptr,d,w.data(),n);
        double orth=eig_test::orthogonality_metric<T,B>(h,Z.ptr,d,n);
        require_close(h,"common method residual",std::isfinite(residual)?residual:INFINITY,50);
        require_close(h,"common method orthogonality",std::isfinite(orth)?orth:INFINITY,50);
        upload(h,A.ptr,input);
        if(method=='G')detail::pheevd_givens<B,T>(h,'N','L',n,A.ptr,d,only.data(),nullptr,nullptr,info,reduction);
        else detail::pheevd_householder<B,T>(h,'N','L',n,A.ptr,d,only.data(),nullptr,nullptr,info,reduction);
        require_close(h,"common method values info",info,0);double error=0,norm=0;
        for(int j=0;j<n;++j) {
            if(!std::isfinite(w[j])||!std::isfinite(only[j])||!std::isfinite(reference[j])||(j&&w[j]<w[j-1]))error=INFINITY;
            else error=std::max({error,double(std::abs(w[j]-reference[j])),double(std::abs(only[j]-reference[j]))});
            for(int i=0;i<n;++i)norm+=std::norm(entry<T>(i,j));
        }
        require_close(h,"common method eigenvalues",error/(n*std::numeric_limits<R>::epsilon()*std::sqrt(norm)),50);
    }
    topology->release();
}
template<DdlaBackend B>void selectors(DdlaHandle_t h) {
    int d[DDLA_DLEN_];ddlaDescInit(d,h,0,0,8,8,0,0);int info=-1;
    detail::pheevd_givens<B,double>(h,'N','U',0,nullptr,d,nullptr,nullptr,nullptr,info);
    require_close(h,"common Givens empty call",info,0);
    detail::pheevd_householder<B,double>(h,'N','U',0,nullptr,d,nullptr,nullptr,nullptr,info);
    require_close(h,"common Householder empty call",info,0);
    auto topology=spectral_impl::phb2st_topology_prepare(h);
    auto rejected=[&](char method,DdlaBackend reduction) {
        bool caught=false;
        try{(void)spectral_impl::pheevd_check<double>(h,B,'N','U',0,nullptr,d,nullptr,nullptr,nullptr,*topology,method,nullptr,reduction);}
        catch(const std::exception&){caught=true;}
        require_close(h,"common method selector rejected collectively",caught?0:1,0);
    };
    rejected('X',B);
    if constexpr(B==DdlaBackend::CPU)rejected('G',DdlaBackend::GPU);
    if(ddlaGetSize(h)>1) {
        rejected(ddlaGetRank(h)?'G':'H',B);
        if constexpr(B==DdlaBackend::GPU)rejected('H',ddlaGetRank(h)?DdlaBackend::GPU:DdlaBackend::CPU);
    }
    if constexpr(B==DdlaBackend::CPU) {
        unsetenv("DDLA_EIG_CPU_ACCUM");unsetenv("DDLA_EIG_CHASE_TEAM_THREADS");setenv("DDLA_EIG_NBW","64",1);
        constexpr int n=2048;ddlaDescInit(d,h,n,n,64,64,0,0);DeviceBuffer<double>A(h,local_size(h,d));std::vector<double>w(n);
        auto automatic=spectral_impl::pheevd_check<double>(h,B,'N','U',n,A.ptr,d,w.data(),nullptr,nullptr,*topology);
        auto givens=spectral_impl::pheevd_check<double>(h,B,'N','U',n,A.ptr,d,w.data(),nullptr,nullptr,*topology,'G');
        require_close(h,"explicit Givens bypasses automatic Householder",
            automatic.automatic_householder&&!givens.automatic_householder&&givens.engine!=spectral_impl::PheevdEngine::Householder?0:1,0);
        unsetenv("DDLA_EIG_NBW");
    }
    topology->release();
}
template<DdlaBackend B>void backend(const std::pair<int,int>& grid) {
    DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,B));DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,grid.first,grid.second));
    unsetenv("DDLA_EIG_CPU_ACCUM");unsetenv("DDLA_EIG_CHASE_TEAM_THREADS");unsetenv("DDLA_EIG_NBW");
    methods<B,float>(h);methods<B,double>(h);methods<B,std::complex<float>>(h);methods<B,std::complex<double>>(h);
    selectors<B>(h);DDLA_CHECK(ddlaDestroy(h));
}
}
int main(int argc,char**argv) {
    ddla_test_mpi_init(&argc, &argv);
    try {
        int P=0;MPI_Comm_size(MPI_COMM_WORLD,&P);TestOptions options;std::string error;
        if(!parse_options(argc,argv,P,options,error))throw std::runtime_error(error);
        for(const auto& grid:options.grids) {
            backend<DdlaBackend::CPU>(grid);
#if DDLA_HAS_GPU
            backend<DdlaBackend::GPU>(grid);
#endif
        }
        std::puts("PHEEVD_METHOD_BACKENDS_PASS");MPI_Finalize();return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"METHOD_BACKENDS_FAIL %s\n",e.what());MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
