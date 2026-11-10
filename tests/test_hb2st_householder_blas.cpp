#include "api_grid_test_common.h"
#include "hh_blas_event_oracle.h"
#include "row_share.h"
#include "mpi_datatype.h"
#include <cstdio>
using namespace api_grid_test;
namespace {
namespace hh=ddla::detail::hh;
template<class T>std::complex<long double> wide(T x){return {static_cast<long double>(x),0};}
template<class R>std::complex<long double> wide(std::complex<R>x){return {x.real(),x.imag()};}
template<class T>T band_value(int i,int j){if(std::abs(i-j)>8)return T(0);if(i<j)return hh::adj(band_value<T>(j,i));return hh_blas_oracle::value<T>(i==j?2+0.01*i:std::sin((i+1)*0.719+(j+1)*0.131),i==j?0:std::cos((i+1)*0.293-(j+1)*0.817));}
template<class T>
void check_policy(const DdlaHandle_t& h,detail::Hb2stHouseholderCpuOptions::Stage2 mode) {
    using R=typename real_type<T>::type;const int n=65,P=ddlaGetSize(h),rank=ddlaGetRank(h);
    int desc[DDLA_DLEN_];ddlaDescInit(desc,h,n,n,4,4,0,0);
    auto matrix=make_local<T>(h,desc,[](int i,int j){return band_value<T>(i,j);});
    detail::Hb2stHouseholderCpuOptions options;options.stage2=mode;
    detail::Hb2stHouseholderCpu<T> state(h,n,8,true,options);
    try {
        std::vector<R>d(n),e(n);std::vector<T>phases;state.reduce(matrix.data(),desc,d.data(),e.data(),phases);
        const R scale=state.eigenvalue_scale();const detail::RowShare share(n,P);const int columns=share.count(rank),first=share.lo(rank);
        std::vector<T>local(std::max<std::size_t>(1,static_cast<std::size_t>(n)*columns),T(0));
        for(int j=0;j<columns;++j)local[first+j+static_cast<std::size_t>(j)*n]=(first+j&&!phases.empty())?phases[first+j-1]:T(1);
        state.apply_columns(local.data(),n,columns);
        std::vector<T>q(static_cast<std::size_t>(n)*n,T(0));
        for(int j=0;j<columns;++j)for(int i=0;i<n;++i)q[i+static_cast<std::size_t>(first+j)*n]=local[i+static_cast<std::size_t>(j)*n];
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,q.data(),n*n,detail::mpi_datatype<T>(),MPI_SUM,ddlaGetCommunicator(h)));
        long double residual=0,orthogonal=0,norm=0;
        for(int j=0;j<n;++j)for(int i=0;i<n;++i) {
            std::complex<long double>aq=0,dot=0,qt=wide(q[i+static_cast<std::size_t>(j)*n])*static_cast<long double>(d[j]);
            for(int k=0;k<n;++k){aq+=wide(hh::divide_real(band_value<T>(i,k),scale))*wide(q[k+static_cast<std::size_t>(j)*n]);dot+=std::conj(wide(q[k+static_cast<std::size_t>(i)*n]))*wide(q[k+static_cast<std::size_t>(j)*n]);}
            if(j)qt+=wide(q[i+static_cast<std::size_t>(j-1)*n])*static_cast<long double>(e[j-1]);
            if(j+1<n)qt+=wide(q[i+static_cast<std::size_t>(j+1)*n])*static_cast<long double>(e[j]);
            residual+=std::norm(aq-qt);orthogonal+=std::norm(dot-(i==j?1.L:0.L));norm+=std::norm(wide(hh::divide_real(band_value<T>(i,j),scale)));
        }
        const auto eps=std::numeric_limits<R>::epsilon();const double rm=std::sqrt(residual)/(n*eps*std::sqrt(norm)),om=std::sqrt(orthogonal)/(n*eps);
        require_close(h,"HH stage2 selected-policy similarity",std::isfinite(rm)?rm:INFINITY,50);
        require_close(h,"HH stage2 selected-policy orthogonality",std::isfinite(om)?om:INFINITY,50);
        state.release();
    }catch(...){state.release();throw;}
}
void check_mismatched_policy(const DdlaHandle_t& h) {
    using Options=detail::Hb2stHouseholderCpuOptions;Options options;
    if(ddlaGetSize(h)>1)options.stage2=ddlaGetRank(h)==0?Options::Stage2::Scalar:Options::Stage2::Blas;
    else options.stage2=static_cast<Options::Stage2>(7);
    int caught=0;try{detail::Hb2stHouseholderCpu<double>state(h,17,4,true,options);state.release();}catch(const std::exception&){caught=1;}
    require_close(h,"HH invalid/inconsistent stage2 policy rejected",caught!=1,0);
}
template<class T>void check_events(const DdlaHandle_t& h) {
    const auto counts=hh_blas_oracle::run<T>(h);
    require_close(h,"HH event mapping and scalar reference",counts.events==0,0);
    if(ddlaGetRank(h)==0)std::printf("HH_BLAS_EVENTS PASS bytes=%zu events=%llu mappings=%llu untouched=%llu scalar_bitwise_values=%llu error=%.9g\n",sizeof(T),(unsigned long long)counts.events,(unsigned long long)counts.mapping,(unsigned long long)counts.unchanged,(unsigned long long)counts.scalar_values,counts.error);
    check_policy<T>(h,detail::Hb2stHouseholderCpuOptions::Stage2::Scalar);
    check_policy<T>(h,detail::Hb2stHouseholderCpuOptions::Stage2::Blas);
}
}
int main(int argc,char**argv) {
    int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int rank=0,size=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    try {
        if(provided<MPI_THREAD_FUNNELED)throw std::runtime_error("MPI thread support");
        TestOptions options;std::string error;if(!parse_options(argc,argv,size,options,error))throw std::runtime_error(error);
        for(auto grid:options.grids) {
            DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,DdlaBackend::CPU));DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,grid.first,grid.second));
            check_events<float>(h);check_events<double>(h);check_events<std::complex<float>>(h);check_events<std::complex<double>>(h);
            check_mismatched_policy(h);DDLA_CHECK(ddlaDestroy(h));
        }
        if(rank==0)std::puts("test_hb2st_householder_blas passed");
    }catch(const std::exception& error){std::fprintf(stderr,"rank %d: %s\n",rank,error.what());MPI_Abort(MPI_COMM_WORLD,1);}
    MPI_Finalize();return 0;
}
