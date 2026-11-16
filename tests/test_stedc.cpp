#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "Eigensolver/stedc_impl.h"
#include "ddla_connector.h"
#include "mpi_test_init.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstring>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

extern "C" void dstedc_(const char*,const int*,double*,double*,double*,const int*,double*,const int*,int*,const int*,int*);

using namespace ddla;
namespace {
int rank_id = 0;
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template<class T> T phase(int i) {
    using R=typename real_type<T>::type;
    if constexpr (std::is_same_v<T,R>) return i%2 ? T(-1) : T(1);
    else {const T p[]={T(1),T(0,1),T(-1),T(0,-1)};return p[i%4];}
}
template<class T> std::complex<double> wide(T v) {
    if constexpr (std::is_same_v<T,typename real_type<T>::type>) return {double(v),0};
    else return {double(v.real()),double(v.imag())};
}
// Check the whole basis in O(n^2), including degenerate eigenspaces.
// Fixed sign vectors make failures reproducible without a random dependency.
double probe_sign(int j,int probe) {
    std::uint64_t bits=std::uint64_t(j)+0x9e3779b97f4a7c15ull*(probe+1);
    bits=(bits^(bits>>30))*0xbf58476d1ce4e5b9ull;
    bits=(bits^(bits>>27))*0x94d049bb133111ebull;
    return ((bits^(bits>>31))>>63)?1.:-1.;
}
template<class Accessor> double orthogonality_probes(int n,Accessor at) {
    double error=0;
    for(int probe=0;probe<2;++probe) {
        std::vector<std::complex<double>> u(n,0);
        for(int j=0;j<n;++j) {
            const double sign=probe_sign(j,probe);
            for(int i=0;i<n;++i)u[i]+=at(i,j)*sign;
        }
        for(int j=0;j<n;++j) {
            std::complex<double> value=0;
            for(int i=0;i<n;++i)value+=std::conj(at(i,j))*u[i];
            require(std::isfinite(value.real()) && std::isfinite(value.imag()),
                "STEDC nonfinite orthogonality probe");
            error=std::max(error,std::abs(value-probe_sign(j,probe)));
        }
    }
    return error;
}
template<class T> void one(DdlaHandle_t h,int n,char compz,int pattern,double scale=1,int padding=3) {
    using R=typename real_type<T>::type;
    const int ld=std::max(1,n)+padding;
    const T guard=T(-731);
    std::vector<R> d(n+2,R(-91)), e(n+1,R(-83));
    for (int i=0;i<n;++i) d[i+1]=R(scale*(pattern==1 ? 3 : 2));
    for (int i=0;i+1<n;++i) e[i+1]=R(scale*((pattern==1 && i%7==0) ? 0 : -1));
    const auto di=d, ei=e;
    std::vector<T> z(std::size_t(ld)*(n+1)+2,guard);
    for (int j=0;j<n;++j) for (int i=0;i<n;++i)
        z[1+i+std::size_t(j)*ld]=compz=='V' && i==(j+1)%n ? phase<T>(j) : T(0);
    const auto zi=z;
    ddla::vector<T> device(h,z.size());
    DDLA_CHECK(ddlaMemcpy(h,device.data(),z.data(),z.size()*sizeof(T)));
    const auto p=spectral_impl::stedc_plan<T>(h,compz,n,ld);
    if constexpr (std::is_same_v<T,std::complex<float>>) {
        if (ddlaGetBackend(h)==DdlaBackend::CPU && n==4096 && compz!='N' && p.lrwork>=16777216)
            require(p.lrwork >= (compz=='I'?33570817:67219457),
                "CSTEDC float RWORK query rounded below its integer minimum");
    }
    ddla::vector<char> bw(h,p.backend_bytes);
    std::vector<char> hw(p.host_bytes);
    spectral_impl::PstedcResources resources(h,p.rows);
    int info=-1;
    // Owning public and reusable private paths must have the same contract.
    for (int repeat=0;repeat<2;++repeat) {
        d=di;e=ei;
        DDLA_CHECK(ddlaMemcpy(h,device.data(),zi.data(),zi.size()*sizeof(T)));
        if (!repeat)
            stedc<T>(h,compz,n,n?d.data()+1:nullptr,n>1?e.data()+1:nullptr,
                compz=='N'?nullptr:device.data()+1,ld,info);
        else if (ddlaGetBackend(h)==DdlaBackend::CPU)
            spectral_impl::stedc_run<DdlaBackend::CPU,T>(h,p,resources,d.data()+1,e.data()+1,
                device.data()+1,ld,info,bw.data(),hw.data());
#if DDLA_HAS_GPU
        else spectral_impl::stedc_run<DdlaBackend::GPU,T>(h,p,resources,d.data()+1,e.data()+1,
                device.data()+1,ld,info,bw.data(),hw.data());
#endif
        require(info==0,"STEDC nonzero info");
#if DDLA_HAS_GPU
        // The private reusable call leaves scratch alive, so queue readiness
        // observes result completion without an owning destructor's async free.
        if (repeat && n && compz!='N' && ddlaGetBackend(h)==DdlaBackend::GPU)
            require(gpuStreamQuery(static_cast<gpuStream_t>(ddlaGetStream(h)))==gpuSuccess,
                "STEDC reusable output was not complete at return");
#endif
        DDLA_CHECK(ddlaMemcpy(h,z.data(),device.data(),z.size()*sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
        require(d.front()==di.front() && d.back()==di.back() &&
                e.front()==ei.front() && e.back()==ei.back(),"STEDC D/E guard overwritten");
        for (std::size_t k=0;k<z.size();++k) {
            const bool active=compz!='N' && k>=1 && k<1+std::size_t(ld)*n && (k-1)%ld<std::size_t(n);
            if (!active) require(!std::memcmp(&z[k],&zi[k],sizeof(T)),"STEDC Z guard/padding overwritten");
        }
        if (!n) continue;
        const double eps=std::numeric_limits<R>::epsilon();
        double residual=0,orth=0;
        auto q=[&](int i,int j) {
            if(compz=='V') return wide(z[1+(i+1)%n+std::size_t(j)*ld])*std::conj(wide(phase<T>(i)));
            return wide(z[1+i+std::size_t(j)*ld]);
        };
        for (int j=0;j<n;++j) {
            require(std::isfinite(d[j+1]),"STEDC nonfinite eigenvalue");
            if(j)require(d[j]<=d[j+1],"STEDC eigenvalues not sorted");
            if(pattern==0) {
                const double exact=2-2*std::cos((j+1)*std::acos(-1.)/(n+1));
                require(std::abs(double(d[j+1])/scale-exact)<80*eps*std::sqrt(double(std::max(1,n))),"STEDC analytical eigenvalue error");
            }
            if(compz!='N') {
                double norm=0;
                for (int i=0;i<n;++i) {
                    const auto entry=q(i,j);
                    require(std::isfinite(entry.real()) && std::isfinite(entry.imag()),
                        "STEDC nonfinite eigenvector");
                    norm+=std::norm(entry);
                    auto r=(double(di[i+1])/scale-double(d[j+1])/scale)*entry;
                    if(i)r+=double(ei[i])/scale*q(i-1,j);
                    if(i+1<n)r+=double(ei[i+1])/scale*q(i+1,j);
                    residual=std::max(residual,std::abs(r));
                }
                require(std::isfinite(norm),"STEDC nonfinite eigenvector norm");
                orth=std::max(orth,std::abs(norm-1));
            }
        }
        if(compz!='N') {
            const int samples=n<=64?n:33;
            for (int a=0;a<samples;++a) for(int b=0;b<samples;++b) {
                const int ja=a*(n-1)/std::max(1,samples-1),jb=b*(n-1)/std::max(1,samples-1);
                std::complex<double> dot=0;
                for(int i=0;i<n;++i)dot+=std::conj(q(i,ja))*q(i,jb);
                orth=std::max(orth,std::abs(dot-(a==b?1.:0.)));
            }
            if(n>64) {
                const double error=orthogonality_probes(n,[&](int i,int j){
                    return wide(z[1+i+std::size_t(j)*ld]);
                });
                require(error<100*eps*std::sqrt(double(n)),"STEDC full-basis orthogonality probe failed");
            }
            require(residual<40*eps*std::sqrt(double(std::max(1,n))),"STEDC tridiagonal residual too large");
            require(orth<40*eps*std::sqrt(double(std::max(1,n))),"STEDC orthogonality error");
        }
        if(rank_id==0 && repeat==0)std::cout<<"case bytes="<<sizeof(T)<<" n="<<n<<" compz="<<compz
            <<" pattern="<<pattern<<" scale="<<scale<<" residual="<<residual<<" orth="<<orth<<'\n';
    }
}
template<class T> void typed(DdlaHandle_t h,bool quick) {
    using R=typename real_type<T>::type;
    for (int n:{0,1,19+rank_id})for(char job:{'N','I','V'})one<T>(h,n,job,0);
    for(char job:{'N','I','V'})one<T>(h,41+rank_id,job,1);
    one<T>(h,37,'I',0,std::is_same_v<R,float>?1e-25:1e-200);
    one<T>(h,37,'V',0,std::is_same_v<R,float>?1e25:1e200);
    if(!quick) {
        one<T>(h,1025+rank_id,'I',0);
        one<T>(h,1073+rank_id,'I',0,1,0);
        one<T>(h,1031+rank_id,'V',1);
        if constexpr (std::is_same_v<T,std::complex<float>>) {
            one<T>(h,4096,'I',1);
            one<T>(h,4096,'V',1);
        }
    }
    if constexpr (std::is_same_v<T,float>) {
        if (ddlaGetBackend(h)==DdlaBackend::CPU) {
            const auto rounded=spectral_impl::stedc_plan<T>(h,'I',4096,4096);
            require(rounded.lwork<16777216 || rounded.lwork>=16793601,
                "SSTEDC float WORK query rounded below its integer minimum");
            const auto largest=spectral_impl::stedc_plan<T>(h,'I',46338,46338);
            require(largest.lwork<16777216 || largest.lwork>=2147395597,
                "SSTEDC large float WORK query was undersized");
            bool rejected=false;
            try {(void)spectral_impl::stedc_plan<T>(h,'I',46339,46339);}
            catch(const std::length_error&){rejected=true;}
            require(rejected,"SSTEDC overflowing LP64 query was not rejected");
        }
    }
    if constexpr (std::is_same_v<T,std::complex<float>>) {
        if (ddlaGetBackend(h)==DdlaBackend::CPU) {
            const auto largest=spectral_impl::stedc_plan<T>(h,'I',32767,32767);
            require(largest.lrwork<16777216 || largest.lrwork==std::numeric_limits<int>::max(),
                "CSTEDC float query rounded beyond LP64 despite a valid minimum");
            bool rejected=false;
            try {(void)spectral_impl::stedc_plan<T>(h,'I',32768,32768);}
            catch(const std::length_error&){rejected=true;}
            require(rejected,"CSTEDC overflowing LP64 query was not rejected");
        }
    }
    int info=713;bool threw=false;
    try {stedc<T>(h,'X',2,nullptr,nullptr,nullptr,1,info);}catch(const std::invalid_argument&){threw=true;}
    require(threw && info==713,"STEDC invalid COMPZ/info guard");
    threw=false;
    try {stedc<T>(h,'I',2,nullptr,nullptr,nullptr,1,info);}catch(const std::invalid_argument&){threw=true;}
    require(threw && info==713,"STEDC invalid storage/info guard");
}
void bench(DdlaHandle_t h,int n,int repeats) {
    using Clock=std::chrono::steady_clock;
    std::vector<double> d0(n),e0(std::max(1,n-1));
    // Deterministic dense-origin-like unreduced tridiagonal, not an already
    // diagonal or symmetry-deflated Toeplitz timing fixture.
    for(int i=0;i<n;++i)d0[i]=std::sin(.73*i)+.2*std::cos(.17*i);
    for(int i=0;i+1<n;++i)e0[i]=std::sqrt(double(n-i-1))*(1+.04*std::sin(.37*i));
    auto p=spectral_impl::stedc_plan<double>(h,'I',n,n);
    ddla::vector<char> bw(h,p.backend_bytes);
    std::vector<char> hw(p.host_bytes);
    ddla::vector<double> z(h,std::size_t(n)*n);
    spectral_impl::PstedcResources resources(h,p.rows);
    std::vector<double> results,reference;
    std::vector<double> raw_work(std::max(1,p.lwork));
    std::vector<int> raw_iwork(std::max(1,p.liwork));
    for(int mode=0;mode<(ddlaGetBackend(h)==DdlaBackend::CPU?3:2);++mode) {
        std::vector<double> times;
        for(int repeat=0;repeat<=repeats;++repeat) {
            auto d=d0,e=e0;int info=-1;
            DDLA_CHECK(ddlaSynchronize(h));
            const auto begin=Clock::now();
            if(mode==2) {
                const char job='I';
                dstedc_(&job,&n,d.data(),e.data(),z.data(),&n,raw_work.data(),&p.lwork,raw_iwork.data(),&p.liwork,&info);
            } else if(mode==1)stedc<double>(h,'I',n,d.data(),e.data(),z.data(),n,info);
            else if(ddlaGetBackend(h)==DdlaBackend::CPU)
                spectral_impl::stedc_run<DdlaBackend::CPU,double>(h,p,resources,d.data(),e.data(),z.data(),n,info,bw.data(),hw.data());
#if DDLA_HAS_GPU
            else spectral_impl::stedc_run<DdlaBackend::GPU,double>(h,p,resources,d.data(),e.data(),z.data(),n,info,bw.data(),hw.data());
#endif
            DDLA_CHECK(ddlaSynchronize(h));
            const double seconds=std::chrono::duration<double>(Clock::now()-begin).count();
            require(info==0,"STEDC benchmark nonzero info");
            if(!mode && !repeat)reference=d;
            else for(int i=0;i<n;++i)require(std::abs(d[i]-reference[i])<1e-10*std::max(1.,std::abs(reference[i])),"STEDC repeat eigenvalue mismatch");
            if(repeat)times.push_back(seconds);
        }
        std::sort(times.begin(),times.end());
        const double median=times.size()%2?times[times.size()/2]:(times[times.size()/2-1]+times[times.size()/2])/2;
        std::cout<<"BENCH n="<<n<<" backend="<<(ddlaGetBackend(h)==DdlaBackend::CPU?"CPU":"GPU")
            <<" mode="<<(mode==2?"lapack":mode==1?"owning":"reused")<<" median="<<median
            <<" min="<<times.front()<<" max="<<times.back()<<" backend_bytes="<<p.backend_bytes
            <<" host_bytes="<<p.host_bytes<<std::endl;
    }
    // Validate the timed result with every residual entry and sampled column
    // inner products, outside the measured interval. Avoid an O(n^3) host check.
    std::vector<double> q(std::size_t(n)*n);
    DDLA_CHECK(ddlaMemcpy(h,q.data(),z.data(),q.size()*sizeof(double)));
    DDLA_CHECK(ddlaSynchronize(h));
    double residual=0,orth=0,norm=0;
    for(int i=0;i<n;++i)norm=std::max(norm,std::abs(d0[i])+(i?std::abs(e0[i-1]):0)+(i+1<n?std::abs(e0[i]):0));
    for(int j=0;j<n;++j) {
        long double column_norm=0;
        for(int i=0;i<n;++i) {
            const double entry=q[i+std::size_t(j)*n];
            require(std::isfinite(entry),"STEDC benchmark nonfinite eigenvector");
            column_norm+=static_cast<long double>(entry)*entry;
            double v=(d0[i]-reference[j])*entry;
            if(i)v+=e0[i-1]*q[i-1+std::size_t(j)*n];
            if(i+1<n)v+=e0[i]*q[i+1+std::size_t(j)*n];
            require(std::isfinite(v),"STEDC benchmark nonfinite residual");
            residual=std::max(residual,std::abs(v)/norm);
        }
        orth=std::max(orth,std::abs(double(column_norm)-1));
    }
    for(int a=0;a<17;++a)for(int b=0;b<17;++b) {
        const int ja=a*(n-1)/16,jb=b*(n-1)/16;long double dot=0;
        for(int i=0;i<n;++i)dot+=static_cast<long double>(q[i+std::size_t(ja)*n])*q[i+std::size_t(jb)*n];
        orth=std::max(orth,std::abs(double(dot)-(a==b?1:0)));
    }
    auto expected=d0, off=e0;
    const char job='N';const int one=1;int info=0,iw=0;double dummy=0,work=0;
    dstedc_(&job,&n,expected.data(),off.data(),&dummy,&one,&work,&one,&iw,&one,&info);
    require(info==0,"STEDC LAPACK values-only oracle failed");
    double spectrum=0;
    for(int i=0;i<n;++i) {
        require(std::isfinite(expected[i]),"STEDC oracle nonfinite eigenvalue");
        spectrum=std::max(spectrum,std::abs(reference[i]-expected[i])/norm);
    }
    const double probe=orthogonality_probes(n,[&](int i,int j){
        return std::complex<double>(q[i+std::size_t(j)*n],0);
    });
    require(probe<1e-10,"STEDC benchmark full-basis orthogonality probe failed");
    require(spectrum<1e-10,"STEDC LAPACK spectrum mismatch");
    require(residual<1e-10 && orth<1e-10,"STEDC benchmark residual/orthogonality failed");
    std::cout<<"BENCH_CHECK n="<<n<<" residual="<<residual<<" orth="<<orth<<" spectrum="<<spectrum<<std::endl;
}
} // namespace

int main(int argc,char** argv) {
    ddla_test_mpi_init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank_id);
    bool gpu=false,benchmark=false,quick=false;
    int n=2048,repeats=3;
    for(int i=1;i<argc;++i) {
        std::string arg=argv[i];
        if(arg=="--gpu")gpu=true;
        else if(arg=="--bench"){benchmark=true;n=std::stoi(argv[++i]);}
        else if(arg=="--repeats")repeats=std::stoi(argv[++i]);
        else if(arg=="--quick")quick=true;
        else throw std::invalid_argument("unknown argument");
    }
    DdlaHandle_t h=nullptr;
    try {
        DDLA_CHECK(ddlaInit(h,gpu?DdlaBackend::GPU:DdlaBackend::CPU));
        DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD));
        if(benchmark)bench(h,n,repeats);
        else {
            typed<float>(h,quick);typed<double>(h,quick);
            typed<std::complex<float>>(h,quick);typed<std::complex<double>>(h,quick);
        }
        DDLA_CHECK(ddlaDestroy(h));
        std::cout<<"STEDC_PASS rank="<<rank_id<<std::endl;
        MPI_Finalize();
    }catch(const std::exception& ex){std::cerr<<"STEDC_FAIL rank="<<rank_id<<" "<<ex.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);}
}
