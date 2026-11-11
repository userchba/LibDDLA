#include "api_grid_test_common.h"
#include "ddla_stream.h"
#include "ddla_connector.h"
#include <cstdlib>
#include "Eigensolver/two_stage/pheevd_impl.h"
#include "Eigensolver/two_stage/hb2st_householder_kernels.h"
#include "eig_test_common.h"
#include <cstdio>
#include <limits>
#include <type_traits>
using namespace api_grid_test;
// Where the band reduction runs.  test_pheevd_device_stage2.cpp builds this
// same test with the reduction on the device.
#ifndef DDLA_TEST_REDUCTION_BACKEND
#define DDLA_TEST_REDUCTION_BACKEND DdlaBackend::CPU
#endif
namespace {
namespace hh=ddla::detail::hh;
template<class T> T value(double re,double im) {
    using R=typename real_type<T>::type;
    if constexpr(std::is_same<T,R>::value)return T(re);
    else return T(R(re),R(im));
}
template<class T> T full_value(int i,int j) {
    if(i<j)return hh::adj(full_value<T>(j,i));
    return i==j?value<T>(3+0.02*i,0):value<T>(0.1*std::sin(i*0.61+j*0.37),0.1*std::cos(i*0.31-j*0.27));
}
template<class T>
void check_full(const DdlaHandle_t& h,int n,bool upper,bool alias,int nb=8) {
    using R=typename real_type<T>::type;
    const int P=ddlaGetSize(h);int npr=0,npc=0;ddlaGetGridDims(h,npr,npc);
    int descA[DDLA_DLEN_],descZ[DDLA_DLEN_];
    ddlaDescInit(descA,h,n+3,n+5,nb,nb,0,0);descA[DDLA_LLD_]+=3;
    ddlaDescInit(descZ,h,n+3,n+5,nb,nb,0,alias?0:npc-1);descZ[DDLA_LLD_]+=3;
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
    detail::pheevd_householder<DdlaBackend::GPU, T>(h,'V',upper?'U':'L',n,A.ptr,descA,w.data(),alias?A.ptr:Z.ptr,descZ,info, DDLA_TEST_REDUCTION_BACKEND);
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
    const double residual=eig_test::residual_metric<T, DdlaBackend::GPU>(h,reference.ptr,descA,output,descZ,w.data(),n);
    const double orthogonal=eig_test::orthogonality_metric<T, DdlaBackend::GPU>(h,output,descZ,n);
    require_close(h,"HH full independent residual",std::isfinite(residual)?residual:INFINITY,50);
    require_close(h,"HH full independent orthogonality",std::isfinite(orthogonal)?orthogonal:INFINITY,50);
    upload(h,reference.ptr,input);
    pheevd_two_stage<DdlaBackend::GPU,T>(h,'N',upper?'U':'L',n,reference.ptr,descA,old.data(),nullptr,nullptr,info);
    require_close(h,"HH reference eigenvalues info",info,0);
    upload(h,reference.ptr,input);
    detail::pheevd_householder<DdlaBackend::GPU, T>(h,'N',upper?'U':'L',n,reference.ptr,descA,only.data(),nullptr,nullptr,info, DDLA_TEST_REDUCTION_BACKEND);
    require_close(h,"HH eigenvalues-only info",info,0);
    double error=0,norm=0;
    for(int j=0;j<n;++j) {
        if(!std::isfinite(w[j])||!std::isfinite(only[j])||!std::isfinite(old[j])||(j&&w[j]<w[j-1]))error=INFINITY;
        else error=std::max({error,std::abs(double(w[j]-old[j])),std::abs(double(only[j]-old[j]))});
        for(int i=0;i<n;++i)norm+=std::norm(full_value<T>(i,j));
    }
    require_close(h,"HH full eigenvalue agreement",error/(std::max(1,n)*std::numeric_limits<R>::epsilon()*std::sqrt(norm)),50);
    if(n==129) {
        // Independent BLAS/WY application of the identical saved transform.
        // Check the sliding kernel's orientation, order and conjugation.
        DeviceBuffer<T> wy(h,Z.count);
        upload(h,reference.ptr,input);upload(h,wy.ptr,std::vector<T>(wy.count,sentinel));
        std::vector<R> ww(n);
        detail::pheevd_householder_stage4<DdlaBackend::GPU, T>(h,'V',upper?'U':'L',n,
            reference.ptr,descA,ww.data(),wy.ptr,descZ,info,0, DDLA_TEST_REDUCTION_BACKEND);
        const auto expected=download(h,wy.ptr,wy.count);
        double difference=0,scale=1;
        for(int c=0;c<zc;++c)for(int r=0;r<zr;++r) {
            if(indx_l2g_r(descZ,h,r)>=n||indx_l2g_c(descZ,h,c)>=n)continue;
            const auto at=r+std::size_t(c)*zld;
            difference=std::max(difference,double(std::abs(after[at]-expected[at])));
            scale=std::max(scale,double(std::abs(expected[at])));
        }
        require_close(h,"GPU short reflectors versus BLAS/WY",difference/(n*std::numeric_limits<R>::epsilon()*scale),50);
    }
    (void)P;
}
}
int main(int argc,char** argv) {
    int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int rank=0,size=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    try {
        if(provided<MPI_THREAD_FUNNELED)throw std::runtime_error("MPI thread support");
        TestOptions options;std::string error;
        if(!parse_options(argc,argv,size,options,error))throw std::runtime_error(error);
        for(const auto& grid:options.grids) {
            DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,DdlaBackend::GPU));
            DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,grid.first,grid.second));
            int device=-1;GPU_CHECK(gpuGetDevice(&device));
            const char* visible=std::getenv("CUDA_VISIBLE_DEVICES");
            std::printf("HH_GPU_DEVICE rank=%d device=%d visible=%s\n",
                ddlaGetRank(h),device,visible?visible:"");
            if(!rank)std::printf("HH_GPU_GRID %d %d\n",grid.first,grid.second);
            if(!std::getenv("DDLA_HH_GPU_TEST_SMALL")) {
                unsetenv("DDLA_EIG_NBW");
#define FULL_CASE(T) for(int n:{1,2,3,4,5,17,33,129,257}){check_full<T>(h,n,true,false);check_full<T>(h,n,false,true);}
                FULL_CASE(float);FULL_CASE(double);FULL_CASE(std::complex<float>);FULL_CASE(std::complex<double>);
#undef FULL_CASE
            }
            for(int band:{16,32,64}) {
                if(std::getenv("DDLA_HH_GPU_TEST_SMALL") && band!=64)continue;
                const auto width=std::to_string(band);setenv("DDLA_EIG_NBW",width.c_str(),1);
#define BAND_CASE(T) check_full<T>(h,129,true,false,64);check_full<T>(h,129,false,true,64)
                BAND_CASE(float);BAND_CASE(double);BAND_CASE(std::complex<float>);BAND_CASE(std::complex<double>);
#undef BAND_CASE
            }
            unsetenv("DDLA_EIG_NBW");
            int da[DDLA_DLEN_];ddlaDescInit(da,h,0,0,8,8,0,0);int info=-1;
            detail::pheevd_householder<DdlaBackend::GPU, double>(h,'N','U',0,nullptr,da,nullptr,nullptr,nullptr,info, DDLA_TEST_REDUCTION_BACKEND);
            require_close(h,"HH GPU empty output",info,0);
            DDLA_CHECK(ddlaDestroy(h));
        }
        if(!rank)std::puts("HOUSEHOLDER_GPU_PASS");
        MPI_Finalize();return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"rank%d: %s\n",rank,e.what());MPI_Abort(MPI_COMM_WORLD,1);return 1;
    }
}
