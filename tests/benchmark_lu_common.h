#ifndef DDLA_BENCHMARK_LU_COMMON_H
#define DDLA_BENCHMARK_LU_COMMON_H
#include "api_grid_test_common.h"
#include "LU/lu_internal.h"
#include "benchmark_grid_options.h"
#include "mpi_datatype.h"
#include <array>
#include <cstdint>
#include <iomanip>
#include <type_traits>
#if defined(DDLA_USE_CUDA)
#include <cuda_profiler_api.h>
#endif

namespace lu_bench {
using namespace api_grid_test;
struct Options {
    benchmark_cli::Options grid;
    int nb=128,warmups=2;
    std::string type="z",matrix="random",mode="both";
};
template<class T> T scalar(double r,double i=0) {
    if constexpr(std::is_same_v<T,double>) return r; else return T(r,i);
}
template<class T> T entry(int i,int j,int n,const std::string& kind) {
    const int r=kind=="permuted" ? (i+n/3+1)%n : i;
    std::uint64_t x=std::uint64_t(r+1)*0x9e3779b97f4a7c15ULL+std::uint64_t(j+1)*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;x^=x>>31;
    T value=scalar<T>(double(x&0xffffff)/16777216.0-0.5,double((x>>24)&0xffffff)/16777216.0-0.5);
    if(kind!="random") {value*=1e-4;if(r==j)value+=T(2);}
    return value;
}
template<class T> T probe(int j,int q) {
    return scalar<T>((double((j*17+q*11)%97)-48)/97.0,(double((j*7+q*13)%89)-44)/89.0);
}
template<class T> void sum(const DdlaHandle_t& h,std::vector<T>& x) {
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,x.data(),static_cast<int>(x.size()),ddla::detail::mpi_datatype<T>(),MPI_SUM,ddlaGetCommunicator(h)));
}
inline double quantile(std::vector<double> x,double q) {
    std::sort(x.begin(),x.end());return x[static_cast<std::size_t>(q*(x.size()-1))];
}
template<class T> std::array<double,3> check(const DdlaHandle_t& h,int n,int width,const int* desc,
        T* device_A,const std::vector<int>& piv,const std::vector<T>& reference,
        const Options& o,bool full,std::size_t count) {
    constexpr int nrhs=3;
    auto out=download(h,device_A,count);
    std::vector<int> gpiv(n,-1),perm(n);
    for(int k=0;k<width;++k){int r=indx_g2l_r(desc,h,k);if(r>=0)gpiv[k]=piv[r];}
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,gpiv.data(),n,MPI_INT,MPI_MAX,ddlaGetCommunicator(h)));
    for(int i=0;i<n;++i)perm[i]=i;
    for(int k=0;k<width;++k) {
        if(gpiv[k]<k+1 || gpiv[k]>n) MPI_Abort(ddlaGetCommunicator(h),11);
        std::swap(perm[k],perm[gpiv[k]-1]);
    }
    std::vector<T> ux(std::size_t(n)*nrhs,T(0)),lux(ux.size(),T(0));
    const int ml=ddla_test::m_loc(h,desc),nl=ddla_test::n_loc(h,desc),ld=desc[DDLA_LLD_];
    double outside=0;
    for(int c=0;c<nl;++c) {
        int gc=indx_l2g_c(desc,h,c);
        for(int r=0;r<ml;++r) {
            int gr=indx_l2g_r(desc,h,r);T a=out[r+std::size_t(c)*ld];
            if(!std::isfinite(double(std::abs(a)))) MPI_Abort(ddlaGetCommunicator(h),14);
            if(gc<width && gr<=gc)
                for(int q=0;q<nrhs;++q) ux[gr+q*n]+=a*probe<T>(gc,q);
            if(!full && gc>=width) outside=std::max(outside,double(std::abs(a-entry<T>(perm[gr],gc,n,o.matrix))));
        }
    }
    sum(h,ux);
    for(int c=0;c<nl;++c) {
        int gc=indx_l2g_c(desc,h,c);if(gc>=width)continue;
        for(int r=0;r<ml;++r) {
            int gr=indx_l2g_r(desc,h,r);if(gr<=gc)continue;
            T a=out[r+std::size_t(c)*ld];
            for(int q=0;q<nrhs;++q)lux[gr+q*n]+=a*ux[gc+q*n];
        }
    }
    sum(h,lux);
    double error=0,scale=1;
    for(int q=0;q<nrhs;++q)for(int i=0;i<n;++i) {
        if(i<width)lux[i+q*n]+=ux[i+q*n];
        if(!std::isfinite(double(std::abs(lux[i+q*n])))) MPI_Abort(ddlaGetCommunicator(h),14);
        error=std::max(error,double(std::abs(lux[i+q*n]-reference[perm[i]+q*n])));
        scale=std::max(scale,double(std::abs(reference[perm[i]+q*n])));
    }
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,&outside,1,MPI_DOUBLE,MPI_MAX,ddlaGetCommunicator(h)));
    double solve_error=0;
    if(full) {
        int db[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(db,h,n,nrhs,o.nb,o.nb,0,0));
        auto rhs=make_local<T>(h,db,[&](int i,int j){return reference[i+j*n];});
        DeviceBuffer<T> b(h,rhs.size());upload(h,b.ptr,rhs);
        ddla::pgetrs(h,'L','N',n,nrhs,device_A,desc,piv.data(),b.ptr,db);
        check_ddla_sync(h);auto sol=download(h,b.ptr,rhs.size());
        solve_error=local_max_error(h,db,sol,[](int i,int j){return probe<T>(i,j);});
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,&solve_error,1,MPI_DOUBLE,MPI_MAX,ddlaGetCommunicator(h)));
    }
    if(!std::isfinite(error) || error/scale>5e-9 || outside!=0 || !std::isfinite(solve_error) || solve_error>5e-9) {
        int rank;MPI_Comm_rank(ddlaGetCommunicator(h),&rank);
        if(rank==0)std::cerr<<"VALIDATION_FAIL n="<<n<<" residual="<<error/scale<<" outside="<<outside<<" solve="<<solve_error<<std::endl;
        MPI_Abort(ddlaGetCommunicator(h),12);
    }
    return {error/scale,outside,solve_error};
}
template<class T>void run_size(const DdlaHandle_t& h,int n,const Options& o,bool full,int rank) {
    constexpr int nrhs=3;
    int desc[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(desc,h,n,n,o.nb,o.nb,0,0));
    const int width=full?n:std::min(n,o.nb);
    auto initial=make_local<T>(h,desc,[&](int i,int j){return entry<T>(i,j,n,o.matrix);});
    DeviceBuffer<T> a(h,initial.size());
    std::vector<T> reference(std::size_t(n)*nrhs,T(0));
    for(int c=0;c<ddla_test::n_loc(h,desc);++c) {
        int gc=indx_l2g_c(desc,h,c);if(gc>=width)continue;
        for(int r=0;r<ddla_test::m_loc(h,desc);++r) {
            int gr=indx_l2g_r(desc,h,r);
            for(int q=0;q<nrhs;++q)reference[gr+q*n]+=initial[r+std::size_t(c)*desc[DDLA_LLD_]]*probe<T>(gc,q);
        }
    }
    sum(h,reference);
    std::vector<int> ipiv(std::max(1,ddla_test::m_loc(h,desc)),-1);
    std::array<std::vector<double>,2> samples;
    std::array<std::array<double,3>,2> errors{};
    std::string routine=full?"pgetrf":"pgetf2";
    for(int iteration=0;iteration<o.warmups+o.grid.repeats;++iteration) {
        for(int pass=0;pass<2;++pass) {
            int method=(iteration+pass)%2;
            const char* mode=method?"gather":"inplace";
            if(o.mode!="both" && o.mode!=mode)continue;
            setenv("DDLA_LU_PANEL",mode,1);
            upload(h,a.ptr,initial);std::fill(ipiv.begin(),ipiv.end(),-1);check_ddla_sync(h);
            MPI_CHECK(MPI_Barrier(ddlaGetCommunicator(h)));
            #if defined(DDLA_USE_CUDA)
            if(std::getenv("DDLA_BENCH_PROFILE")) GPU_CHECK(cudaProfilerStart());
#endif
            int info=-1;const double begin=MPI_Wtime();
            if(full)ddla::pgetrf(h,n,n,a.ptr, 1, 1,desc,ipiv.data(),info);
            else ddla::pgetf2(h,n,n,width,a.ptr, 1, 1,desc,ipiv.data(),info);
            check_ddla_sync(h);double elapsed=MPI_Wtime()-begin;
#if defined(DDLA_USE_CUDA)
            if(std::getenv("DDLA_BENCH_PROFILE")) GPU_CHECK(cudaProfilerStop());
#endif
            MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,&elapsed,1,MPI_DOUBLE,MPI_MAX,ddlaGetCommunicator(h)));
            int failed=info!=0;MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,&failed,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h)));
            if(failed){if(rank==0)std::cerr<<"LU info="<<info<<std::endl;MPI_Abort(ddlaGetCommunicator(h),13);}
            if(iteration>=o.warmups) {
                samples[method].push_back(elapsed);
                if(rank==0)std::cout<<std::setprecision(12)<<"RAW routine="<<routine<<" type="<<o.type<<" n="<<n<<" nb="<<o.nb
                    <<" grid="<<benchmark_cli::grid_name(o.grid)<<" matrix="<<o.matrix<<" method="<<mode
                    <<" rep="<<iteration-o.warmups<<" time_s="<<elapsed<<std::endl;
            }
            if(iteration==o.warmups+o.grid.repeats-1)
                errors[method]=check(h,n,width,desc,a.ptr,ipiv,reference,o,full,initial.size());
        }
    }
    if(rank==0)for(int method=0;method<2;++method)if(!samples[method].empty()) {
        auto& x=samples[method];
        std::cout<<std::setprecision(12)<<"RESULT routine="<<routine<<" type="<<o.type<<" n="<<n<<" nb="<<o.nb
            <<" grid="<<benchmark_cli::grid_name(o.grid)<<" matrix="<<o.matrix<<" method="<<(method?"gather":"inplace")
            <<" warmups="<<o.warmups<<" repeats="<<o.grid.repeats<<" median_time_s="<<quantile(x,.5)
            <<" min_s="<<quantile(x,0)<<" max_s="<<quantile(x,1)<<" q1_s="<<quantile(x,.25)<<" q3_s="<<quantile(x,.75)
            <<" factor_projection_error="<<errors[method][0]<<" outside_error="<<errors[method][1]<<" solve_error="<<errors[method][2]<<std::endl;
    }
}
inline int main(int argc,char**argv,bool full) {
    MPI_Init(&argc,&argv);int rank,size;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    Options o;std::vector<char*> remaining{argv[0]};std::string error;
    for(int i=1;i<argc;++i) {
        std::string arg=argv[i];
        if(arg=="--nb" || arg=="--type" || arg=="--matrix" || arg=="--mode" || arg=="--warmups") {
            if(i+1==argc){error="missing option value";break;}
            std::string v=argv[++i];
            if(arg=="--nb") {if(!benchmark_cli::parse_positive_int(v,o.nb))error="invalid nb";}
            else if(arg=="--warmups") {if(!parse_nonneg_int(v,o.warmups))error="invalid warmups";}
            else if(arg=="--type")o.type=v;else if(arg=="--matrix")o.matrix=v;else o.mode=v;
        } else remaining.push_back(argv[i]);
    }
    if(error.empty())benchmark_cli::parse(static_cast<int>(remaining.size()),remaining.data(),true,7,o.grid,error);
    if(o.type!="d" && o.type!="z")error="type must be d or z";
    if(o.matrix!="random" && o.matrix!="dominant" && o.matrix!="permuted")error="matrix must be random, dominant or permuted";
    if(o.mode!="both" && o.mode!="inplace" && o.mode!="gather")error="mode must be both, inplace or gather";
    if(o.grid.nprows*o.grid.npcols!=size)error="grid does not match MPI rank count";
    if(!error.empty()){if(rank==0)std::cerr<<error<<std::endl;MPI_Abort(MPI_COMM_WORLD,2);}
    DdlaHandle_t h=nullptr;ddlaInit(h);ddlaSet(h,MPI_COMM_WORLD,o.grid.nprows,o.grid.npcols);
    for(int n:o.grid.sizes)if(o.type=="d")run_size<double>(h,n,o,full,rank);else run_size<std::complex<double>>(h,n,o,full,rank);
    ddlaDestroy(h);MPI_Finalize();return 0;
}
}
#endif
