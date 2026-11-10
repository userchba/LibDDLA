#include "api_grid_test_common.h"
#include "Eigensolver/two_stage/phb2st_impl.h"
#include "Eigensolver/two_stage/hb2st_householder_kernels.h"
#include "Eigensolver/eig_params.h"
#include <limits>
#include <cstdio>
using namespace api_grid_test;
namespace hh=ddla::detail::hh;
int extreme_values=0;
template<class T>T gv_value(int i,int j,int b,bool zero) {
    using R=typename real_type<T>::type;if(zero||j-i>b||i>j)return T(0);
    if constexpr(std::is_same<T,std::complex<double>>::value) {
        if(extreme_values) {
            const double scale=extreme_values==1?1e150:1e100;
            if(i==j)return T(scale);
            if(i==0&&j==1) {
                const double tiny=std::numeric_limits<double>::denorm_min();
                return extreme_values==1?T(0,1e-200):(extreme_values==2?T(4*tiny,7*tiny):T(0));
            }
            if(i==0&&j==2) {
                const double tiny=std::numeric_limits<double>::denorm_min();
                return extreme_values==3?T(4*tiny,7*tiny):T(scale,.2*scale);
            }
            return T(0);
        }
    }
    if(i==j)return T(3+.002*i);
    if constexpr(std::is_same<T,R>::value)return T(.13*std::sin(i*.61+j*.37));
    else return T(.13*std::sin(i*.61+j*.37),.11*std::cos(i*.31-j*.27));
}
template<class T>std::complex<long double> wide(T x) {
    using R=typename real_type<T>::type;
    if constexpr(std::is_same<T,R>::value)return {static_cast<long double>(x),0};
    else return {static_cast<long double>(x.real()),static_cast<long double>(x.imag())};
}
template<class T>void probe(DdlaHandle_t h,int n,int requested,bool zero=false) {
    using R=typename real_type<T>::type;using C=std::complex<long double>;using namespace ddla::spectral_impl;
    const int b=std::max(1,(ddlaGetSize(h)==1?requested:std::min(requested,ddla::eig_distributed_band_limit(n,ddlaGetSize(h)))));
    int desc[DDLA_DLEN_];ddlaDescInit(desc,h,n,n,64,64,0,0);desc[DDLA_LLD_]+=3;
    const auto input=make_local<T>(h,desc,[&](int i,int j){return gv_value<T>(i,j,b,zero);});
    DeviceBuffer<T>a(h,std::max<std::size_t>(1,input.size()));upload(h,a.ptr,input);
    std::vector<R>d(n),e(n);std::vector<T>phase(std::max(0,n-1));auto topology=phb2st_topology_prepare(h);
    auto plan=phb2st_check<T>(h,DdlaBackend::GPU,n,b,a.ptr,desc,d.data(),e.data(),phase.data(),phase.size(),*topology,Phb2stOutput::History);
    plan.device_chase=true;plan.team.enabled=true;plan.team.participants=ddlaGetSize(h);plan.team.tpr=1;
    plan.team.ring_request=plan.local.record_capacity;plan.team.consumers=false;
    Phb2stResources<T>resources(h,plan);std::size_t db=0,hb=0;phb2st_workspace_size<T>(plan,&db,&hb);
    ddla::vector<char>device(h,db);std::vector<char>host(hb);std::size_t phase_count=0;
    phb2st_run<DdlaBackend::GPU,T>(h,plan,resources,a.ptr,d.data(),e.data(),phase.data(),phase_count,nullptr,device.data(),db,host.data(),hb);
    require_close(h,"Givens input/padding unchanged",download(h,a.ptr,input.size())!=input?1:0,0);
    const int cols=n<=129?n:3;
    std::vector<T>x(std::size_t(n)*cols),tx(x.size());
    for(int i=0;i<n;++i)for(int j=0;j<cols;++j)
        x[std::size_t(i)*cols+j]=cols==n?T(i==j):T(R(std::sin(.13*i+.73*j)));
    for(int i=0;i<n;++i)for(int j=0;j<cols;++j) {
        auto at=std::size_t(i)*cols+j;tx[at]=d[i]*x[at];if(i)tx[at]+=e[i-1]*x[at-cols];if(i+1<n)tx[at]+=e[i]*x[at+cols];
    }
    auto ux=x,utx=tx;
    if constexpr(!std::is_same<T,R>::value)
        for(int i=1;i<n;++i)for(int j=0;j<cols;++j){ux[std::size_t(i)*cols+j]*=phase[i-1];utx[std::size_t(i)*cols+j]*=phase[i-1];}
    auto& history=resources.history();std::vector<hb2st_rot<T>>chunk;
    for(long long q=history.chunks();q-->0;) {
        history.fetch(q,ddlaGetCommunicator(h),chunk);
        for(auto it=chunk.rbegin();it!=chunk.rend();++it) {
            const int a=it->j0-1,b=it->j0;
            for(auto* out:{&ux,&utx})for(int j=0;j<cols;++j) {
                T& v=(*out)[std::size_t(a)*cols+j];T& w=(*out)[std::size_t(b)*cols+j];const T old=v;
                v=it->c*old-hh::adj(it->s)*w;w=it->s*old+it->c*w;
            }
        }
    }
    long double norm=0,error=0,orth=0;
    for(int i=0;i<n;++i)for(int j=0;j<cols;++j) {
        C z=0;for(int k=std::max(0,i-b);k<std::min(n,i+b+1);++k)
            z+=(i<=k?wide(gv_value<T>(i,k,b,zero)):std::conj(wide(gv_value<T>(k,i,b,zero))))*wide(ux[std::size_t(k)*cols+j]);
        norm+=std::norm(z);error+=std::norm(z-wide(utx[std::size_t(i)*cols+j]));
    }
    for(int j=0;j<cols;++j)for(int k=0;k<cols;++k) {
        C before=0,after=0;for(int i=0;i<n;++i){before+=std::conj(wide(x[std::size_t(i)*cols+j]))*wide(x[std::size_t(i)*cols+k]);after+=std::conj(wide(ux[std::size_t(i)*cols+j]))*wide(ux[std::size_t(i)*cols+k]);}
        orth=std::max(orth,std::abs(before-after)/std::max(1.L,std::abs(before)));
    }
    const double scale=std::max(1,n)*std::numeric_limits<R>::epsilon();
    const double resid=double(std::sqrt(error)/std::max(1.L,std::sqrt(norm)))/scale,unit=double(orth)/scale;
    require_close(h,"GPU Givens independent band similarity",std::isfinite(resid)?resid:INFINITY,50);
    require_close(h,"GPU Givens transform Gram agreement",std::isfinite(unit)?unit:INFINITY,50);
    unsigned long long count=history.local_count,total=0;
    MPI_Allreduce(&count,&total,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,ddlaGetCommunicator(h));
    require_close(h,"Givens unique distributed coefficient ownership",total==static_cast<unsigned long long>(history.total)?0:1,0);
    if(!ddlaGetRank(h))std::printf("GIVENS_BAND n=%d band=%d bytes=%zu residual=%.12g gram=%.12g status=PASS\n",n,b,sizeof(T),resid,unit);
    resources.release();topology->release();
}
void reject_unshared(DdlaHandle_t h) {
    using namespace ddla::spectral_impl;
    if(ddlaGetSize(h)==1)return;
    constexpr int n=65,b=4;int desc[DDLA_DLEN_];ddlaDescInit(desc,h,n,n,8,8,0,0);
    DeviceBuffer<double>A(h,std::max<std::size_t>(1,local_size(h,desc)));
    std::vector<double>d(n),e(n);auto topology=phb2st_topology_prepare(h);
    auto plan=phb2st_check<double>(h,DdlaBackend::GPU,n,b,A.ptr,desc,d.data(),e.data(),nullptr,0,*topology,Phb2stOutput::History);
    plan.device_chase=true;plan.team.enabled=false;
    bool rejected=false;
    try{Phb2stResources<double>resources(h,plan);resources.release();}
    catch(const std::invalid_argument& error){rejected=std::string(error.what()).find("requires a shared band and record ring")!=std::string::npos;}
    require_close(h,"GPU Givens rejects unavailable shared transport",rejected?0:1,0);
    topology->release();
}
int main(int argc,char**argv) {
    int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    try {
        int P=0;MPI_Comm_size(MPI_COMM_WORLD,&P);DdlaHandle_t h=nullptr;ddlaInit(h,DdlaBackend::GPU);ddlaSet(h,MPI_COMM_WORLD,P>=4?2:1,P/(P>=4?2:1));
        const bool phase_only=argc>1&&std::string(argv[1])=="--phase-range-only";
        if(!phase_only) {
        for(int b:{2,3,8,16,64})for(int n:{1,2,3,5,17,65,129}) {
            probe<float>(h,n,b);probe<double>(h,n,b);probe<std::complex<float>>(h,n,b);probe<std::complex<double>>(h,n,b);
        }
        for(int b:{16,64}){probe<double>(h,4099,b);probe<std::complex<double>>(h,4099,b);}
        probe<double>(h,129,16,true);probe<std::complex<double>>(h,129,64,true);
        }
        for(int mode:{1,2,3}) {
            extreme_values=mode;
            probe<std::complex<double>>(h,65,2);
        }
        extreme_values=0;
        reject_unshared(h);
        ddlaDestroy(h);std::puts("GIVENS_DEVICE_BAND_PASS");MPI_Finalize();return 0;
    }catch(const std::exception&e){std::fprintf(stderr,"GIVENS_BAND_FAIL %s\n",e.what());MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
