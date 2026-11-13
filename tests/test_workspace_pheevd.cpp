#include "api_grid_test_common.h"
#include "ddla_stream.h"
#include "workspace_test_common.h"
#include "Eigensolver/two_stage/pheevd_impl.h"
#include "Eigensolver/eig_params.h"
#include "mpi_datatype.h"
#include <array>
#include <cstdint>
#include <cstring>
using namespace api_grid_test;
namespace {
template<class S>bool no_alloc(const S& s,bool allow_host_growth=false){return !s.ddla_alloc&&!s.ddla_free&&!s.pinned_alloc&&!s.raw_gpu_alloc&&(allow_host_growth||!s.host_new)&&!s.resource_setup;}
class Arena {
    static constexpr std::size_t guard=256;
    DdlaHandle_t h_;std::size_t bb_,hb_;
    ddla::vector<char> backend_;std::vector<char> host_;
public:
    Arena(DdlaHandle_t h,std::size_t bb,std::size_t hb)
        :h_(h),bb_(bb),hb_(hb),backend_(h,bb+2*guard),host_(hb+2*guard,char(0xa7)) {
        DDLA_CHECK(ddlaMemset(h,backend_.data(),0xa7,backend_.size()));
    }
    char* backend(){return backend_.data()+guard;}
    char* host(){return host_.data()+guard;}
    void verify(){std::array<unsigned char,2*guard> bytes{};
        DDLA_CHECK(ddlaMemcpy(h_,bytes.data(),backend_.data(),guard));
        DDLA_CHECK(ddlaMemcpy(h_,bytes.data()+guard,backend()+bb_,guard));DDLA_CHECK(ddlaSynchronize(h_));
        bool ok=true;for(std::size_t i=0;i<guard;++i)ok=ok&&bytes[i]==0xa7&&bytes[guard+i]==0xa7&&
            static_cast<unsigned char>(host_[i])==0xa7&&static_cast<unsigned char>(host()[hb_+i])==0xa7;
        require_close(h_,"pheevd exact reusable arena guards",ok?0:1,0);}
};
template<class T>T conjv(T x){return x;}template<class R>std::complex<R>conjv(std::complex<R>x){return std::conj(x);}
template<class T>T matrix(int i,int j,int repeat){using R=typename real_type<T>::type;if(i>j)return conjv(matrix<T>(j,i,repeat));
    if(i==j)return T(R(4)+R(i+repeat)/R(17));
    if constexpr(std::is_same<T,R>::value)return R((i*7+j*11+repeat*3)%17-8)/R(127);
    else return T(R((i*7+j*11+repeat*3)%17-8)/R(127),R((i*13+j*3+repeat*5)%19-9)/R(131));}
template<class T>std::complex<double> wide(T x){return {double(x),0};}template<class R>std::complex<double>wide(std::complex<R>x){return {double(x.real()),double(x.imag())};}
template<class T>bool padding(DdlaHandle_t h,const int* d,int n,const std::vector<T>& actual,const std::vector<T>& before){
    for(int j=0;j<ddla_test::n_loc(h,d);++j)for(int i=0;i<d[DDLA_LLD_];++i){const int gi=indx_l2g_r(d,h,i),gj=indx_l2g_c(d,h,j);
        if(i>=ddla_test::m_loc(h,d)||gi>=n||gj>=n){const auto at=std::size_t(j)*d[DDLA_LLD_]+i;if(std::memcmp(&actual[at],&before[at],sizeof(T)))return false;}}return true;}
template<DdlaBackend B,class T>void solver_case(DdlaHandle_t h,spectral_impl::Phb2stTopology& topology,int n,char job,char uplo,
    bool compact,bool alias,int engine,bool profiling=false){
    using R=typename real_type<T>::type;int da[DDLA_DLEN_],dz[DDLA_DLEN_];const int parent=compact?0:3;
    DDLA_CHECK(ddlaDescInit(da,h,n+parent,n+parent,4,4,0,0));DDLA_CHECK(ddlaDescInit(dz,h,n+parent,n+parent,4,4,0,compact?0:(h->npcols_>1?1:0)));
    if(!compact){da[DDLA_LLD_]+=2+h->myid;dz[DDLA_LLD_]+=3+h->myid;}if(alias)std::copy_n(da,DDLA_DLEN_,dz);
    const auto initialA=make_local<T>(h,da,[](int,int){return T(-919);}),initialZ=make_local<T>(h,dz,[](int,int){return T(-717);});
    DeviceBuffer<T>A(h,initialA.size()),storageZ(h,initialZ.size());T* Z=alias?A.ptr:storageZ.ptr;
    std::vector<R>w(std::size_t(n)+8,R(-313));
    setenv("DDLA_EIG_PROFILE",profiling?"1":"0",1);setenv("DDLA_EIG_CPU_ACCUM",engine==1?"1":"0",1);setenv("DDLA_QR_PANEL","gather",1);
    const bool force=engine>=2;const int stage4=engine==3?0:1;
    const auto p=spectral_impl::pheevd_check<T>(h,B,job,uplo,n,A.ptr,da,w.data()+4,Z,dz,topology,force?'H':'A',force?&stage4:nullptr);
    if(n&&job=='V'){
        const detail::RowShare rows(n,h->nprocs);
        bool row_input=true;
        if(p.engine==spectral_impl::PheevdEngine::Householder)row_input=!p.hh_rows.input_2d.enabled;
        else if(p.engine==spectral_impl::PheevdEngine::AccumHost)row_input=!p.finish.input_2d.enabled;
        else if(p.engine==spectral_impl::PheevdEngine::ReplayCpu)row_input=!p.cpu_replay.input_2d.enabled;
#if DDLA_HAS_GPU
        if(p.engine==spectral_impl::PheevdEngine::ReplayGpu)row_input=!p.gpu_replay.input_2d.enabled;
#endif
        require_close(h,"two-stage default uses row D&C and direct row input",
            row_input&&p.vectors.n==n&&p.ldy==std::max(1,rows.count(h->myid))&&
            p.vectors.ldy==p.ldy&&p.vectors.borrow_q2==(rows.count(h->myid)>0)?0:1,0);
    }
    if(n>2 && h->nprocs>1)
        require_close(h,"pheevd distributed band is bounded and divides the block",
            p.band>ddla::eig_distributed_band_limit(n,h->nprocs) || da[DDLA_NB_]%p.band ? 1 : 0,0);
    if(!h->myid)std::cout<<"pheevd case n="<<n<<" scalar_bytes="<<sizeof(T)<<" backend="<<int(B)<<" job="<<job<<" uplo="<<uplo
        <<" padded="<<!compact<<" alias="<<alias<<" engine="<<int(p.engine)<<" profile="<<profiling<<std::endl;
    MPI_Barrier(h->comm);bool pure=true;std::uint64_t vendor=0;
    if(!h->myid){workspace_test::Scope scope(true);std::size_t b=0,s=0;spectral_impl::pheevd_workspace_size<T>(h,p,A.ptr,&b,&s);const auto t=scope.finish();vendor=t.vendor_query;
        pure=no_alloc(t)&&!t.copies&&!t.launches&&!t.syncs&&!t.collectives;}
    MPI_Barrier(h->comm);require_close(h,"pheevd query allocates/communicates/writes no data",pure?0:1,0);
    std::size_t backend_bytes=0,host_bytes=0;
    spectral_impl::pheevd_workspace_size<T>(h,p,A.ptr,&backend_bytes,&host_bytes);
    Arena arena(h,backend_bytes,host_bytes);
    spectral_impl::PheevdResources<T> resources(h,p);
    setenv("DDLA_EIG_PROFILE",profiling?"0":"1",1);setenv("DDLA_EIG_CPU_ACCUM",engine==1?"0":"1",1);setenv("DDLA_QR_PANEL","inplace",1);
    for(int repeat=0;repeat<2;++repeat){
        const auto poison=T(std::numeric_limits<R>::quiet_NaN());
        const auto original=make_local<T>(h,da,[&](int i,int j){if(i>=n||j>=n)return T(-919);
            return (uplo=='U'?i<=j:i>=j)?matrix<T>(i,j,repeat):poison;});
        upload(h,A.ptr,original);if(!alias)upload(h,Z,initialZ);std::fill(w.begin(),w.end(),R(-313));int info=-1;
        workspace_test::Scope scope;spectral_impl::pheevd_run<B,T>(h,p,resources,A.ptr,w.data()+4,Z,info,arena.backend(),backend_bytes,arena.host(),host_bytes);const auto t=scope.finish();
        const bool allow_host_growth=B==DdlaBackend::GPU &&
            ddlaGetCommunication(h)==DdlaCommunication::HostStagedMPI && repeat==0;
        require_close(h,"pheevd raw repeat reuses algorithm scratch and warm communication storage",no_alloc(t,allow_host_growth)&&!t.vendor_query?0:1,0);require_close(h,"pheevd raw info",info?1:0,0);
        bool sorted=true,guarded=true;for(int i=0;i<n;++i)sorted=sorted&&std::isfinite(w[4+i])&&(i==0||w[3+i]<=w[4+i]);
        for(int i=0;i<4;++i)guarded=guarded&&w[i]==R(-313)&&w[std::size_t(n)+4+i]==R(-313);
        require_close(h,"pheevd finite sorted values and scalar guards",sorted&&guarded?0:1,0);
        const auto afterA=download(h,A.ptr,initialA.size());require_close(h,"pheevd input parent/padding",padding(h,da,n,afterA,original)?0:1,0);
        if(job=='V'){
            const auto local=download(h,Z,initialZ.size());require_close(h,"pheevd output parent/padding",padding(h,dz,n,local,alias?original:initialZ)?0:1,0);
            std::vector<T>Q(std::size_t(n)*n,T(0));for(int j=0;j<ddla_test::n_loc(h,dz);++j)for(int i=0;i<ddla_test::m_loc(h,dz);++i){
                const int gi=indx_l2g_r(dz,h,i),gj=indx_l2g_c(dz,h,j);if(gi<n&&gj<n)Q[std::size_t(gj)*n+gi]=local[std::size_t(j)*dz[DDLA_LLD_]+i];}
            if(n)MPI_Allreduce(MPI_IN_PLACE,Q.data(),n*n,detail::mpi_datatype<T>(),MPI_SUM,h->comm);
            double residual=0,orthogonality=0;for(int j=0;j<n;++j)for(int i=0;i<n;++i){std::complex<double> v=0,dot=0;
                for(int k=0;k<n;++k){v+=wide(matrix<T>(i,k,repeat))*wide(Q[std::size_t(j)*n+k]);dot+=std::conj(wide(Q[std::size_t(i)*n+k]))*wide(Q[std::size_t(j)*n+k]);}
                residual=std::max(residual,workspace_test::finite_error(std::abs(v-double(w[4+j])*wide(Q[std::size_t(j)*n+i]))));
                orthogonality=std::max(orthogonality,workspace_test::finite_error(std::abs(dot-std::complex<double>(i==j?1:0))));}
            require_close(h,"pheevd independent Hermitian residual",residual,std::is_same<R,float>::value?4e-4:8e-11);
            require_close(h,"pheevd orthogonality",orthogonality,std::is_same<R,float>::value?4e-4:8e-11);
        }else if(!alias){const auto z=download(h,Z,initialZ.size());require_close(h,"pheevd values-only Z unchanged",std::memcmp(z.data(),initialZ.data(),z.size()*sizeof(T))?1:0,0);}
        arena.verify();
    }
    resources.release();if(resources.profile){resources.profile->report_timing(h->myid,n,p.band);resources.profile->report(h->myid,n);}
    if(!h->myid)std::cout<<"pheevd memory n="<<n<<" type_bytes="<<sizeof(T)<<" engine="<<int(p.engine)<<" layout=rows"
        <<" host="<<host_bytes<<" backend="<<backend_bytes<<" vendor_queries="<<vendor<<std::endl;
    unsetenv("DDLA_EIG_PROFILE");unsetenv("DDLA_EIG_CPU_ACCUM");unsetenv("DDLA_QR_PANEL");
}
template<DdlaBackend B>void run(MPI_Comm comm,int pr,int pc,char major){DdlaHandle_t h=nullptr;ddlaInit(h,B);ddlaSet(h,comm,pr,pc,major);auto topology=spectral_impl::phb2st_topology_prepare(h);
#define CASE(T) for(char uplo:{'U','L'}){solver_case<B,T>(h,*topology,17,'V',uplo,false,false,0);solver_case<B,T>(h,*topology,17,'V',uplo,true,false,0);solver_case<B,T>(h,*topology,17,'V',uplo,false,true,0);solver_case<B,T>(h,*topology,17,'N',uplo,false,false,0);}solver_case<B,T>(h,*topology,1,'V','L',true,false,0);solver_case<B,T>(h,*topology,0,'V','U',false,false,0)
    CASE(float);CASE(double);CASE(std::complex<float>);CASE(std::complex<double>);
#undef CASE
    if constexpr(B==DdlaBackend::CPU){
#define CPU(T) solver_case<B,T>(h,*topology,17,'V','U',false,false,1);if(topology->groups()==1){solver_case<B,T>(h,*topology,17,'V','L',false,true,2);solver_case<B,T>(h,*topology,17,'V','U',false,false,3);}
        CPU(float);CPU(double);CPU(std::complex<float>);CPU(std::complex<double>);
#undef CPU
    }
#if DDLA_HAS_GPU
    if constexpr(B==DdlaBackend::GPU) {
        if(topology->groups()==1) {
#define GPU_HH(T) solver_case<B,T>(h,*topology,17,'V','L',false,true,2);solver_case<B,T>(h,*topology,17,'V','U',false,false,3);solver_case<B,T>(h,*topology,1,'N','U',false,false,2)
            GPU_HH(float);GPU_HH(double);GPU_HH(std::complex<float>);GPU_HH(std::complex<double>);
#undef GPU_HH
        }
    }
#endif
    solver_case<B,double>(h,*topology,17,'V','U',false,false,0,true);topology->release();ddlaDestroy(h);}
}
int main(int argc,char** argv){int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);if(provided<MPI_THREAD_FUNNELED)MPI_Abort(MPI_COMM_WORLD,2);
    int size=1;MPI_Comm_size(MPI_COMM_WORLD,&size);TestOptions options;std::string error;if(!parse_options(argc,argv,size,options,error)){std::cerr<<error<<std::endl;MPI_Abort(MPI_COMM_WORLD,2);}
    for(auto grid:options.grids)for(char major:{'R','C'}){
        run<DdlaBackend::CPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#if DDLA_HAS_GPU
        run<DdlaBackend::GPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#endif
    }MPI_Finalize();return 0;}
