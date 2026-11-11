#include "api_grid_test_common.h"
#include "Eigensolver/two_stage/hb2st_householder_cpu.h"
#include "Eigensolver/two_stage/hb2st_householder_kernels.h"
#include <limits>
#include <cstdio>
using namespace api_grid_test;
template<class T>T probe_value(double re,double im) {
    using R=typename real_type<T>::type;
    if constexpr(std::is_same<T,R>::value)return T(re);
    else return T(R(re),R(im));
}
template<class T>std::complex<long double> wide_probe(T x) {
    using R=typename real_type<T>::type;
    if constexpr(std::is_same<T,R>::value)return {static_cast<long double>(x),0};
    else return {static_cast<long double>(x.real()),static_cast<long double>(x.imag())};
}
template<class T>void similarity(DdlaHandle_t h,int n,int b,const std::vector<T>& band,
    ddla::detail::Hb2stHouseholderCpu<T>& reducer,const std::vector<typename real_type<T>::type>& diag,
    const std::vector<typename real_type<T>::type>& off,const std::vector<T>& phases,const char* tag) {
    using R=typename real_type<T>::type;using C=std::complex<long double>;
    std::vector<T> q(std::size_t(n)*n,T(0));for(int j=0;j<n;++j)q[std::size_t(j)*n+j]=T(1);
    DeviceBuffer<T> device_q(h,std::max<std::size_t>(1,q.size()));upload(h,device_q.ptr,q);
    std::size_t bytes=0;ddla::detail::hh_gpu_apply_workspace_size<T>(b,n,false,&bytes);ddla::vector<char>work(h,bytes);
    reducer.apply_rows_gpu_run(device_q.ptr,std::max(n,1),n,work.data(),work.size());
    q=download(h,device_q.ptr,q.size());
    std::vector<C> u(q.size());
    for(int j=0;j<n;++j)for(int i=0;i<n;++i)u[std::size_t(j)*n+i]=wide_probe(q[std::size_t(i)*n+j])*
        (std::is_same<T,R>::value || j==0?C(1):wide_probe(phases[j-1]));
    long double norm2=0,error2=0,orth=0;const long double scale=reducer.eigenvalue_scale();
    auto a=[&](int i,int j) {
        if(std::abs(i-j)>b)return C(0);
        if(i<=j)return wide_probe(band[std::size_t(j)*(b+1)+b+i-j]);
        return std::conj(wide_probe(band[std::size_t(i)*(b+1)+b+j-i]));
    };
    for(int i=0;i<n;++i)for(int k=std::max(0,i-b);k<std::min(n,i+b+1);++k)norm2+=std::norm(a(i,k));
    for(int j=0;j<n;++j)for(int i=0;i<n;++i) {
        C z=0;
        for(int k=std::max(0,i-b);k<std::min(n,i+b+1);++k)z+=a(i,k)*u[std::size_t(j)*n+k];
        C expected=static_cast<long double>(diag[j])*u[std::size_t(j)*n+i];
        if(j)expected+=static_cast<long double>(off[j-1])*u[std::size_t(j-1)*n+i];
        if(j+1<n)expected+=static_cast<long double>(off[j])*u[std::size_t(j+1)*n+i];
        error2+=std::norm(z-scale*expected);
        C dot=0;for(int k=0;k<n;++k)dot+=std::conj(u[std::size_t(i)*n+k])*u[std::size_t(j)*n+k];
        orth=std::max(orth,std::abs(dot-C(i==j?1:0)));
    }
    const long double denom=std::max(1,n)*static_cast<long double>(std::numeric_limits<R>::epsilon());
    const double residual=double(std::sqrt(error2)/(denom*std::max(1.L,std::sqrt(norm2))));
    const double orthogonal=double(orth/denom);
    require_close(h,std::string(tag)+" direct band similarity",std::isfinite(residual)?residual:INFINITY,50);
    require_close(h,std::string(tag)+" direct transform orthogonality",std::isfinite(orthogonal)?orthogonal:INFINITY,50);
}
// Linear-memory independent similarity sketch crosses GPU ownership boundaries.
// For fixed X verify B*(U*D*X) == scale*U*D*(T*X), without materializing U.
template<class T>void sketch(DdlaHandle_t h,int n,int requested,bool zero) {
    using R=typename real_type<T>::type;using C=std::complex<long double>;
    using namespace ddla::detail;
    const int b=std::min(requested,std::max(1,n-1)),cols=3;
    int desc[DDLA_DLEN_];ddlaDescInit(desc,h,n,n,64,64,0,0);
    std::vector<T> band(std::size_t(b+1)*n,T(0));
    for(int j=0;j<n;++j)for(int i=std::max(0,j-b);i<=j;++i)
        band[std::size_t(j)*(b+1)+b+i-j]=zero?T(0):probe_value<T>(i==j?3+.002*i:.13*std::sin(i*.61+j*.37),i==j?0:.11*std::cos(i*.31-j*.27));
    Hb2stHouseholderCpuOptions options;options.stage2=Hb2stHouseholderCpuOptions::Stage2::Gpu;
    auto plan=hh_cpu_geometry(h,n,b,true,options,1);Hb2stHouseholderCpu<T> gpu(h,plan);
    std::size_t bytes=0;hh_cpu_reduce_workspace_size<T>(h,b,1,&bytes);std::vector<char>work(bytes);
    std::vector<R>d(n),e(n);std::vector<T>phases(std::max(0,n-1));
    gpu.reduce_run(nullptr,desc,d.data(),e.data(),phases.data(),work.data(),work.size(),band.data());
    std::vector<T>x(std::size_t(n)*cols),tx(x.size()),ux(x.size()),utx(x.size());
    for(int i=0;i<n;++i)for(int j=0;j<cols;++j)x[std::size_t(i)*cols+j]=probe_value<T>(std::sin(.13*i+.73*j),std::cos(.17*i-.27*j));
    for(int i=0;i<n;++i)for(int j=0;j<cols;++j) {
        auto at=std::size_t(i)*cols+j;tx[at]=d[i]*x[at];
        if(i)tx[at]+=e[i-1]*x[at-cols];
        if(i+1<n)tx[at]+=e[i]*x[at+cols];
        T phase=T(1);if constexpr(!std::is_same<T,R>::value)if(i)phase=phases[i-1];
        ux[at]=phase*x[at];utx[at]=phase*tx[at];
    }
    DeviceBuffer<T>a(h,ux.size()),c(h,utx.size());upload(h,a.ptr,ux);upload(h,c.ptr,utx);
    hh_gpu_apply_workspace_size<T>(b,cols,false,&bytes);ddla::vector<char>apply(h,bytes);
    gpu.apply_rows_gpu_run(a.ptr,cols,cols,apply.data(),apply.size());
    gpu.apply_rows_gpu_run(c.ptr,cols,cols,apply.data(),apply.size());
    ux=download(h,a.ptr,ux.size());utx=download(h,c.ptr,utx.size());
    long double norm=0,error=0;auto get=[&](int i,int j) {
        if(i<=j)return wide_probe(band[std::size_t(j)*(b+1)+b+i-j]);
        return std::conj(wide_probe(band[std::size_t(i)*(b+1)+b+j-i]));
    };
    for(int i=0;i<n;++i)for(int j=0;j<cols;++j) {
        C z=0;for(int k=std::max(0,i-b);k<std::min(n,i+b+1);++k)z+=get(i,k)*wide_probe(ux[std::size_t(k)*cols+j]);
        norm+=std::norm(z);error+=std::norm(z-static_cast<long double>(gpu.eigenvalue_scale())*wide_probe(utx[std::size_t(i)*cols+j]));
    }
    const double metric=double(std::sqrt(error)/(std::max(1.L,std::sqrt(norm))*n*std::numeric_limits<R>::epsilon()));
    require_close(h,"distributed band similarity sketch",std::isfinite(metric)?metric:INFINITY,50);
    if(!ddlaGetRank(h))std::printf("DISTRIBUTED_SKETCH n=%d band=%d bytes=%zu error=%.12g history_local=%zu\n",n,b,sizeof(T),metric,gpu.local_history_bytes());
    gpu.release();
}
template<class T>void failure_probe(DdlaHandle_t h) {
    using namespace ddla::detail;using R=typename real_type<T>::type;
    const int n=257,b=16;int desc[DDLA_DLEN_];ddlaDescInit(desc,h,n,n,64,64,0,0);
    Hb2stHouseholderCpuOptions options;options.stage2=Hb2stHouseholderCpuOptions::Stage2::Gpu;
    options.failure=Hb2stHouseholderCpuOptions::Failure::Reduce;options.failure_rank=ddlaGetSize(h)-1;
    auto plan=hh_cpu_geometry(h,n,b,true,options,1);Hb2stHouseholderCpu<T>gpu(h,plan);
    std::size_t bytes=0;hh_cpu_reduce_workspace_size<T>(h,b,1,&bytes);std::vector<char>work(bytes);
    std::vector<T>band(std::size_t(b+1)*n,T(1)),ph(n);std::vector<R>d(n),e(n);bool caught=false;
    try{gpu.reduce_run(nullptr,desc,d.data(),e.data(),ph.data(),work.data(),bytes,band.data());}
    catch(const std::exception&){caught=true;}
    require_close(h,"selected rank GPU stage2 failure agreement",caught?0:1,0);gpu.release();
}
template<class T>void probe(DdlaHandle_t h,int n,int requested,bool zero) {
    using R=typename real_type<T>::type;using namespace ddla::detail;
    const int b=std::min(requested,std::max(1,n-1));
    int desc[DDLA_DLEN_];ddlaDescInit(desc,h,n,n,64,64,0,0);
    std::vector<T> band(std::size_t(b+1)*n,T(0));
    for(int j=0;j<n;++j)for(int i=std::max(0,j-b);i<=j;++i) {
        const double re=i==j?3+0.02*i:0.13*std::sin(i*.61+j*.37);
        const double im=i==j?0:0.11*std::cos(i*.31-j*.27);
        band[std::size_t(j)*(b+1)+b+i-j]=zero?T(0):probe_value<T>(re,im);
    }
    const auto before=band;
    Hb2stHouseholderCpuOptions options;
    auto pc=hh_cpu_geometry(h,n,b,true,options,3);options.stage2=Hb2stHouseholderCpuOptions::Stage2::Gpu;
    auto pg=hh_cpu_geometry(h,n,b,true,options,3);
    Hb2stHouseholderCpu<T>cpu(h,pc),gpu(h,pg);
    std::size_t bytes=0;hh_cpu_reduce_workspace_size<T>(h,b,3,&bytes);std::vector<char>work(bytes);
    std::vector<R>dc(n),ec(n),dg(n),eg(n);std::vector<T>fc(std::max(0,n-1)),fg(std::max(0,n-1));
    cpu.reduce_run(nullptr,desc,dc.data(),ec.data(),fc.data(),work.data(),work.size(),band.data());
    gpu.reduce_run(nullptr,desc,dg.data(),eg.data(),fg.data(),work.data(),work.size(),band.data());
    double err=0;
    for(int i=0;i<n;++i) {
        if(!std::isfinite(dc[i])||!std::isfinite(dg[i])||!std::isfinite(ec[i])||!std::isfinite(eg[i]))err=INFINITY;
        else err=std::max({err,double(std::abs(dc[i]-dg[i])),double(std::abs(ec[i]-eg[i]))});
    }
    for(int i=0;i<n-1;++i) {
        if(!hh::finite(fc[i])||!hh::finite(fg[i]))err=INFINITY;
        else err=std::max(err,double(std::abs(fc[i]-fg[i])));
    }
    if(!std::isfinite(err))throw std::runtime_error("stage2 nonfinite d/e/phase");
    // Tridiagonal entries/phases depend on rounded reflector choices. Check
    // the actual Hermitian similarity and unitarity for each route instead.
    std::printf("RAW_TRIDIAGONAL_DIFFERENCE n=%d band=%d bytes=%zu scaled=%g\n",n,b,sizeof(T),err/(std::max(1,n)*std::numeric_limits<R>::epsilon()));
    similarity(h,n,b,band,cpu,dc,ec,fc,"CPU");
    similarity(h,n,b,band,gpu,dg,eg,fg,"GPU");
    require_close(h,"stage2 packed input unchanged",before!=band?1:0,0);
    require_close(h,"stage2 scaling agreement",std::abs(cpu.eigenvalue_scale()-gpu.eigenvalue_scale()),0);
    require_close(h,"stage2 history ownership",cpu.local_history_bytes()!=gpu.local_history_bytes()?1:0,0);
    gpu.release();cpu.release();
}
int main(int argc,char**argv) {
    ddla_test_mpi_init(&argc, &argv);
    try {
        int size=0;MPI_Comm_size(MPI_COMM_WORLD,&size);
        DdlaHandle_t h=nullptr;ddlaInit(h,DdlaBackend::GPU);ddlaSet(h,MPI_COMM_WORLD,size>=4?2:1,size/(size>=4?2:1));
        for(int b:{16,64})for(int n:{0,1,2,3,5,17,65,129,257}) {
            probe<float>(h,n,b,false);probe<double>(h,n,b,false);
            probe<std::complex<float>>(h,n,b,false);probe<std::complex<double>>(h,n,b,false);
        }
        probe<double>(h,129,16,true);probe<std::complex<double>>(h,129,64,true);
        for(int b:{16,64}) {
            sketch<double>(h,4099,b,false);sketch<std::complex<double>>(h,4099,b,false);
        }
        failure_probe<double>(h);
        ddlaDestroy(h);std::puts("GPU_CPU_STAGE2_PROBE_PASS");MPI_Finalize();return 0;
    }catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
