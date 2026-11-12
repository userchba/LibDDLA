#include "api_grid_test_common.h"
#include "lapack_impl.h"
#include <limits>
#include <type_traits>
#include <cstring>
using namespace api_grid_test;
namespace {
template<class T>T entry(int i,int j) {
    using R=typename real_type<T>::type;
    if(i==j)return T(3+R(i)/10);
    const R re=R(((std::min(i,j)*7+std::max(i,j)*11)%17)-8)/20;
    if constexpr(std::is_same_v<T,R>)return re;
    else return T(re,R(i<j?1:-1)/9);
}
template<class T>std::complex<double> wide(T x) {
    if constexpr(std::is_same_v<T,typename real_type<T>::type>)return {double(x),0};
    else return {double(x.real()),double(x.imag())};
}
template<DdlaBackend B,class T>void cases(DdlaHandle_t h) {
    using R=typename real_type<T>::type;
    int rank=0;MPI_Comm_rank(ddlaGetCommunicator(h),&rank);
    // Different rank-local orders catch an accidental grid collective in a local interface.
    for(int size:{0,1,17,33,65})for(char uplo:{'U','L'})for(char job:{'N','V'}) {
        const int n=size?size+rank:0,ld=std::max(1,n+5);
        std::vector<T> initial(std::size_t(ld)*(n+2),T(12345));
        const R nan=std::numeric_limits<R>::quiet_NaN();
        for(int j=0;j<n;++j)for(int i=0;i<n;++i) {
            T x=(uplo=='U'?i<=j:i>=j)?entry<T>(i,j):T(nan);
            if constexpr(!std::is_same_v<T,R>)if(i==j)x=T(std::real(x),nan);
            initial[i+std::size_t(j)*ld]=x;
        }
        DeviceBuffer<T> a(h,initial.size());DeviceBuffer<R> values(h,n+2);DeviceBuffer<int> status(h,3);
        const std::vector<R> w0(n+2,R(6789));const std::vector<int> info0(3,9182);
        upload(h,a.ptr,initial);upload(h,values.ptr,w0);upload(h,status.ptr,info0);
        auto call=[&](bool borrowed) {
            if(borrowed) {
                int lwork=0,lrwork=0,liwork=0;std::size_t padding_elements=0;
                lapack_impl::heevd_workspace_size<T>(h,job,uplo,n,a.ptr,ld,values.ptr+1,lwork,lrwork,liwork,padding_elements);
                const auto after=download(h,a.ptr,initial.size());
                require_close(h,"local eigensolver workspace query preserves A",std::memcmp(after.data(),initial.data(),initial.size()*sizeof(T))?1:0,0);
                ddla::vector<T> work(h,std::max(1,lwork)),padding(h,padding_elements);
                std::vector<R> real_work(std::max(1,lrwork));std::vector<int> int_work(std::max(1,liwork));
                lapack_impl::heevd_run<B,T>(h,job,uplo,n,a.ptr,ld,values.ptr+1,work.data(),lwork,
                    real_work.data(),lrwork,int_work.data(),liwork,status.ptr+1,padding.data());
                ddlaSynchronize(h);
            } else if constexpr(std::is_same_v<T,R>)syevd<T>(h,job,uplo,n,n?a.ptr:nullptr,ld,n?values.ptr+1:nullptr,status.ptr+1);
            else heevd<T>(h,job,uplo,n,n?a.ptr:nullptr,ld,n?values.ptr+1:nullptr,status.ptr+1);
        };
        for(bool borrowed:{false,true}) {
            upload(h,a.ptr,initial);upload(h,values.ptr,w0);upload(h,status.ptr,info0);call(borrowed);
            const auto out=download(h,a.ptr,initial.size());const auto w=download(h,values.ptr,n+2);const auto info=download(h,status.ptr,3);
            bool okay=info[0]==9182&&info[1]==0&&info[2]==9182&&w.front()==R(6789)&&w.back()==R(6789);
            for(int j=0;j<n+2;++j)for(int i=0;i<ld;++i)
                if(i>=n||j>=n)okay=okay&&!std::memcmp(&out[i+std::size_t(j)*ld],&initial[i+std::size_t(j)*ld],sizeof(T));
            for(int i=0;i<n;++i)okay=okay&&std::isfinite(w[i+1])&&(!i||w[i+1]>=w[i]);
            if(job=='V')for(int j=0;j<n;++j)for(int i=0;i<n;++i)
                okay=okay&&std::isfinite(std::abs(out[i+std::size_t(j)*ld]));
            double residual=0,orth=0;
            if(job=='V')for(int j=0;j<n;++j)for(int i=0;i<n;++i) {
                std::complex<double> az=0,zz=0;
                for(int k=0;k<n;++k) {
                    az+=wide(entry<T>(i,k))*wide(out[k+std::size_t(j)*ld]);
                    zz+=std::conj(wide(out[k+std::size_t(i)*ld]))*wide(out[k+std::size_t(j)*ld]);
                }
                residual=std::max(residual,std::abs(az-wide(out[i+std::size_t(j)*ld])*double(w[j+1])));
                orth=std::max(orth,std::abs(zz-(i==j?1.:0.)));
            }
            if(!std::isfinite(residual)||!std::isfinite(orth))okay=false;
            if(!okay) {
                std::cout<<"LOCAL_EVD_GUARD_FAILURE n="<<n<<" lda="<<ld<<" scalar_bytes="<<sizeof(T)<<" job="<<job<<" uplo="<<uplo<<" borrowed="<<borrowed<<std::endl;
                int shown=0;
                for(int j=0;j<n+2&&shown<12;++j)for(int i=0;i<ld&&shown<12;++i)
                    if((i>=n||j>=n)&&std::memcmp(&out[i+std::size_t(j)*ld],&initial[i+std::size_t(j)*ld],sizeof(T))) {
                        std::cout<<"PADDING_CHANGED i="<<i<<" j="<<j<<" before="<<initial[i+std::size_t(j)*ld]<<" after="<<out[i+std::size_t(j)*ld]<<std::endl;
                        ++shown;
                    }
            }
            require_close(h,"local eigensolver info/finite/padding guards",okay?0:1,0);
            require_close(h,"local eigensolver residual",residual,std::is_same_v<R,float>?3e-4:2e-11);
            require_close(h,"local eigensolver orthogonality",orth,std::is_same_v<R,float>?3e-4:2e-11);
        }
    }
}
void body(const DdlaHandle_t& h,const Shape&) {
    if(ddlaGetBackend(h)==DdlaBackend::CPU) {
        cases<DdlaBackend::CPU,float>(h);cases<DdlaBackend::CPU,double>(h);
        cases<DdlaBackend::CPU,std::complex<float>>(h);cases<DdlaBackend::CPU,std::complex<double>>(h);
    }
#if DDLA_HAS_GPU
    else {
        cases<DdlaBackend::GPU,float>(h);cases<DdlaBackend::GPU,double>(h);
        cases<DdlaBackend::GPU,std::complex<float>>(h);cases<DdlaBackend::GPU,std::complex<double>>(h);
    }
#endif
}
}
int main(int argc,char**argv){return run_grid_test(argc,argv,"test_heevd",body);}
