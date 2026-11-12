#include "test_backend.h"
#include "Eigensolver/pstedc_impl.h"
// Private row-baseline pstedc benchmark. One warmup, then max-over-rank wall
// times; inputs are restored before every call. Validation uses row slabs,
// O(n) boundary/norm vectors and sampled orthogonality, never a full replica.
// mpirun -n 4 ./benchmark_pstedc [--kind 0|1|5] [--reps R] [--check-max N] n ...
#include "api_grid_test_common.h"
#include "row_share.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

extern "C" void dsterf_(const int*,double*,double*,int*);
using namespace api_grid_test;
namespace {
double hash01(std::uint64_t i,int tag)
{
    std::uint64_t x=(i+1)*0x9E3779B97F4A7C15ull^std::uint64_t(tag)*0xC2B2AE3D27D4EB4Full;
    x^=x>>30;x*=0xBF58476D1CE4E5B9ull;x^=x>>27;x*=0x94D049BB133111EBull;x^=x>>31;
    return double(x>>11)/9007199254740992.0-0.5;
}
}
int main(int argc,char** argv)
{
    ddla_test_mpi_init(&argc,&argv);
    try {
        int rank=0,P=1;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&P);
        int kind=0,reps=3,check_max=6000;std::vector<int> sizes;
        for(int i=1;i<argc;++i){const std::string arg=argv[i];
            if(arg=="--kind"&&i+1<argc)kind=std::stoi(argv[++i]);
            else if(arg=="--reps"&&i+1<argc)reps=std::stoi(argv[++i]);
            else if(arg=="--check-max"&&i+1<argc)check_max=std::stoi(argv[++i]);
            else if(arg.rfind("--",0)==0)throw std::invalid_argument("unknown benchmark option: "+arg);
            else sizes.push_back(std::stoi(arg));
        }
        if(reps<1||(kind!=0&&kind!=1&&kind!=5))throw std::invalid_argument("invalid repetitions or matrix kind");
        if(sizes.empty())sizes={2000,5000,10000};
        ddla::DdlaHandle_t h=nullptr;ddla::ddlaInit(h, ddla_test::execution_backend);ddla::ddlaSet(h,MPI_COMM_WORLD,1,P);
        const double eps=std::numeric_limits<double>::epsilon();
        for(int n:sizes){
            if(n<1)throw std::invalid_argument("benchmark n must be positive");
            const ddla::detail::RowShare share(n,P);const int nr=share.count(rank),lo=share.lo(rank),ld=std::max(1,nr);
            DeviceBuffer<double> Y(h,std::size_t(ld)*n);
            std::vector<double> d0(n),e0(n,0),d,e,w(n),times;
            for(int i=0;i<n;++i){d0[i]=kind==5?2:kind==1?0.5*(i%4):hash01(i,1);
                if(i+1<n)e0[i]=kind==5?-1:kind==1?1e-3*hash01(i,3):hash01(i,2);}
            double normT=0;
            for(int i=0;i<n;++i)normT=std::max(normT,std::abs(d0[i])+std::abs(e0[i])+(i?std::abs(e0[i-1]):0));
            for(int rep=-1;rep<reps;++rep){
                d=d0;e=e0;check_ddla_sync(h);MPI_Barrier(MPI_COMM_WORLD);const double t0=MPI_Wtime();
                int info=0;ddla::pstedc<double>(h,'V',n,d.data(),e.data(),w.data(),Y.ptr,ld,info);
                check_ddla_sync(h);const double dt=MPI_Wtime()-t0;double maximum=0;
                MPI_Allreduce(&dt,&maximum,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
                if(info)throw std::runtime_error("pstedc returned nonzero info");
                if(rep>=0)times.push_back(maximum);
            }
            std::sort(times.begin(),times.end());
            const double median=times.size()%2?times[times.size()/2]:0.5*(times[times.size()/2-1]+times[times.size()/2]);
            const auto local=download<double>(h,Y.ptr,std::size_t(ld)*n);
            std::vector<double> first(n,0),last(n,0),before(n,0),after(n,0),norms(n,0);
            if(nr)for(int j=0;j<n;++j){first[j]=local[std::size_t(j)*ld];last[j]=local[nr-1+std::size_t(j)*ld];}
            const int left=nr&&lo>0?share.owner(lo-1):MPI_PROC_NULL;
            const int right=nr&&lo+nr<n?share.owner(lo+nr):MPI_PROC_NULL;
            MPI_Sendrecv(first.data(),n,MPI_DOUBLE,left,11,after.data(),n,MPI_DOUBLE,right,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            MPI_Sendrecv(last.data(),n,MPI_DOUBLE,right,12,before.data(),n,MPI_DOUBLE,left,12,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            double residual=0,orthogonal=0,eigen_error=0;int bad=0;
            for(int j=0;j<n;++j){bad+=!std::isfinite(w[j])||(j&&w[j]<w[j-1]);
                for(int r=0;r<nr;++r){const int i=lo+r;const auto at=r+std::size_t(j)*ld;const double x=local[at];
                    if(!std::isfinite(x)){++bad;continue;}
                    norms[j]+=x*x;double v=(d0[i]-w[j])*x;
                    if(i)v+=e0[i-1]*(r?local[at-1]:before[j]);
                    if(i+1<n)v+=e0[i]*(r+1<nr?local[at+1]:after[j]);
                    residual=std::max(residual,std::abs(v));
                }
            }
            MPI_Allreduce(MPI_IN_PLACE,norms.data(),n,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            double norm_error=0;for(double x:norms)norm_error=std::max(norm_error,std::abs(x-1));
            const int samples=std::min(n,16);std::vector<double> gram(samples*samples,0);
            for(int a=0;a<samples;++a)for(int b=0;b<samples;++b){const int ja=a*(n-1)/std::max(1,samples-1),jb=b*(n-1)/std::max(1,samples-1);
                for(int r=0;r<nr;++r)gram[a+samples*b]+=local[r+std::size_t(ja)*ld]*local[r+std::size_t(jb)*ld];}
            MPI_Allreduce(MPI_IN_PLACE,gram.data(),int(gram.size()),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            for(int a=0;a<samples;++a)for(int b=0;b<samples;++b)orthogonal=std::max(orthogonal,std::abs(gram[a+samples*b]-(a==b)));
            if(kind==5)for(int j=0;j<n;++j)eigen_error=std::max(eigen_error,std::abs(w[j]-(2-2*std::cos((j+1)*std::acos(-1.0)/(n+1)))));
            else if(n<=check_max){auto ref=d0,off=e0;int info=0;dsterf_(&n,ref.data(),off.data(),&info);
                if(info)throw std::runtime_error("reference STERF failed");
                for(int j=0;j<n;++j)eigen_error=std::max(eigen_error,std::abs(w[j]-ref[j]));}
            MPI_Allreduce(MPI_IN_PLACE,&residual,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
            const double scaled_residual=residual/(normT*eps*n),scaled_ortho=orthogonal/(eps*n);
            if(bad||scaled_residual>50||scaled_ortho>50||norm_error/(eps*n)>50||eigen_error/(eps*normT)>100)
                throw std::runtime_error("distributed eigenpair validation failed");
            const auto plan=ddla::spectral_impl::pstedc_check<double>(h,'V',n,d0.data(),e0.data(),w.data(),Y.ptr,ld);
            std::size_t backend=0,host=0;ddla::spectral_impl::pstedc_workspace_size<double>(h,plan,&backend,&host);
            unsigned long long bytes[3]={std::size_t(ld)*n*sizeof(double),backend,host};
            MPI_Allreduce(MPI_IN_PLACE,bytes,3,MPI_UNSIGNED_LONG_LONG,MPI_MAX,MPI_COMM_WORLD);
            if(!rank)std::cout<<"PSTEDC_BENCH n="<<n<<" P="<<P<<" kind="<<kind<<" reps="<<reps
                <<std::fixed<<std::setprecision(6)<<" min="<<times.front()<<" median="<<median
                <<std::scientific<<" residual="<<scaled_residual<<" sampled_ortho="<<scaled_ortho
                <<" output_bytes_max="<<bytes[0]<<" workspace_backend_bytes_max="<<bytes[1]
                <<" workspace_host_bytes_max="<<bytes[2]<<" ok"<<std::endl;
        }
        ddla::ddlaDestroy(h);
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);}
    MPI_Finalize();return 0;
}
