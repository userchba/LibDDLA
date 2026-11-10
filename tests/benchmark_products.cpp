#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "ddla_stream.h"
#include "host_threads.h"
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace ddla;
namespace {
using Wide=std::complex<long double>;

struct Options {
    std::string backend,api;
    char scalar;
    int n,nb=128,warmups=3,repeats=9;
    std::string cases="all";
};
int number(const char* text,const char* label,int minimum) {
    std::size_t used=0;const int value=std::stoi(text,&used);
    if(text[used]!='\0' || value<minimum) throw std::invalid_argument(std::string("invalid ")+label);
    return value;
}
template <typename T> T scalar(double re,double im=0) {
    if constexpr(std::is_arithmetic_v<T>) return T(re);
    else return T(re,im);
}
template <typename T> Wide wide(T x) {
    if constexpr(std::is_arithmetic_v<T>) return {x,0};
    else return {x.real(),x.imag()};
}
template <typename T> T conjugate(T x) {
    if constexpr(std::is_arithmetic_v<T>) return x;
    else return std::conj(x);
}
// Small dyadic inputs are deterministic from global indices. No random-number
// generation, matrix transfer, or validation is included in a timed interval.
template <typename T> T entry(int i,int j,int salt) {
    const auto r=(std::uint64_t(i)*17+std::uint64_t(j)*29+salt*7)%61;
    const auto q=(std::uint64_t(i)*31+std::uint64_t(j)*11+salt*13)%59;
    return scalar<T>((int(r)-30)/32.,(int(q)-29)/32.);
}
struct Case { bool symmetric;char a,b;const char* name; };

template <typename T> T input_value(const Case& c,bool first,int i,int j) {
    if(c.symmetric) {
        if(first) return entry<T>(i,j,1);
        const T x=entry<T>(j,i,1),weight=T((i%2 ? -1 : 1)*(i%7+1)/16.);
        return weight*conjugate(x);
    }
    const char triangle=first ? c.a : c.b;
    if(triangle=='U' ? i>j : i<j) return T(0);
    return entry<T>(i,j,first ? 2 : 3);
}

struct Layout {
    int d[DDLA_DLEN_],n,nb,pr,pc,r,c,nr,nc,ld;
    std::size_t elements;
    Layout(DdlaHandle_t h,const Options& o) : n(o.n),nb(o.nb) {
        ddlaGetGridDims(h,pr,pc);ddlaGetGridCoords(h,r,c);
        nr=num_loc(n,nb,r,0,pr);nc=num_loc(n,nb,c,0,pc);ld=std::max(1,nr)+2;
        DDLA_CHECK(ddlaDescInit(d,h,n,n,nb,nb,0,0,ld));
        elements=std::size_t(ld)*nc+8;
    }
    int row(int i) const {return indxl2g(i,nb,r,0,pr);}
    int col(int j) const {return indxl2g(j,nb,c,0,pc);}
    bool active(std::size_t p) const {return p<std::size_t(ld)*nc && p%ld<std::size_t(nr);}
};

std::vector<std::size_t> sample_positions(const Layout& l) {
    std::vector<std::size_t> positions;
    if(l.nr==0 || l.nc==0) return positions;
    const auto add=[&](int i,int j) {
        const auto p=std::size_t(i)+std::size_t(j)*l.ld;
        if(std::find(positions.begin(),positions.end(),p)==positions.end()) positions.push_back(p);
    };
    add(0,0);add(l.nr-1,0);add(0,l.nc-1);add(l.nr-1,l.nc-1);
    for(int s=0;s<64 && positions.size()<12;++s)
        add(int((std::uint64_t(s)*7919+std::uint64_t(l.r)*97+17)%l.nr),
            int((std::uint64_t(s)*3571+std::uint64_t(l.c)*53+29)%l.nc));
    return positions;
}

template <typename T> double upload_input(DdlaHandle_t h,const Layout& l,const Case& c,
                                         bool first,T* device) {
    const T guard=scalar<T>(-987.25,654.5);
    std::vector<T> host(l.elements,guard);double maximum=0;
    for(int j=0;j<l.nc;++j) for(int i=0;i<l.nr;++i) {
        const T value=input_value<T>(c,first,l.row(i),l.col(j));
        host[i+std::size_t(j)*l.ld]=value;maximum=std::max(maximum,double(std::abs(wide(value))));
    }
    DDLA_CHECK(ddlaMemcpy(h,device,host.data(),host.size()*sizeof(T)));
    DDLA_CHECK(ddlaSynchronize(h));
    double global=0;MPI_Allreduce(&maximum,&global,1,MPI_DOUBLE,MPI_MAX,ddlaGetCommunicator(h));
    return global;
}

template <typename T> void poison_output(DdlaHandle_t h,const Layout& l,T* output) {
    constexpr std::size_t chunk_size=1u<<18;
    std::vector<T> chunk(std::min(chunk_size,l.elements));
    const T guard=scalar<T>(-987.25,654.5);
    const T poison=scalar<T>(std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN());
    for(std::size_t first=0;first<l.elements;first+=chunk.size()) {
        const auto count=std::min(chunk.size(),l.elements-first);
        for(std::size_t q=0;q<count;++q) chunk[q]=l.active(first+q) ? poison : guard;
        DDLA_CHECK(ddlaMemcpy(h,output+first,chunk.data(),count*sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
    }
}

template <typename T> void check_inputs(DdlaHandle_t h,const Layout& l,const Case& c,
    const T* a,const T* b,const std::vector<std::size_t>& points) {
    int bad=0;
    for(std::size_t p:points) {
        T av,bv;DDLA_CHECK(ddlaMemcpy(h,&av,a+p,sizeof(T)));
        DDLA_CHECK(ddlaMemcpy(h,&bv,b+p,sizeof(T)));DDLA_CHECK(ddlaSynchronize(h));
        const int i=l.row(int(p%l.ld)),j=l.col(int(p/l.ld));
        if(av!=input_value<T>(c,true,i,j) || bv!=input_value<T>(c,false,i,j)) bad=1;
    }
    int any=0;MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h));
    if(any) throw std::runtime_error("benchmark sampled input immutability check failed");
}

struct Errors { long double absolute=0,normalized=0; };
void merge_errors(DdlaHandle_t h,Errors& e,int bad,const char* phase) {
    long double in[2]={e.absolute,e.normalized},out[2]={};int any=0;
    MPI_Allreduce(in,out,2,MPI_LONG_DOUBLE,MPI_MAX,ddlaGetCommunicator(h));
    MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h));
    e={out[0],out[1]};
    if(any) throw std::runtime_error(std::string("benchmark nonfinite output or padding change: ")+phase);
}

template <typename T> void sampled_oracle(DdlaHandle_t h,const Layout& l,const Case& c,
    const T* output,const std::vector<std::size_t>& points,Errors& e,bool selected=false) {
    int bad=0;
    for(std::size_t p:points) {
        T actual;DDLA_CHECK(ddlaMemcpy(h,&actual,output+p,sizeof(T)));DDLA_CHECK(ddlaSynchronize(h));
        const int i=l.row(int(p%l.ld)),j=l.col(int(p/l.ld));
        if(selected && c.symmetric && (c.a=='U' ? i>j : i<j)) continue;
        Wide sum=0;long double scale=0;
        for(int k=0;k<l.n;++k) {
            const Wide av=wide(input_value<T>(c,true,i,k)),bv=wide(input_value<T>(c,false,k,j));
            sum+=av*bv;scale+=std::abs(av)*std::abs(bv);
        }
        const long double delta=std::abs(wide(actual)-sum),relative=delta/std::max(1.L,scale);
        if(!std::isfinite(delta) || !std::isfinite(relative)) bad=1;
        else {e.absolute=std::max(e.absolute,delta);e.normalized=std::max(e.normalized,relative);}
    }
    merge_errors(h,e,bad,"independent sampled oracle");
}

template <typename T> std::vector<T> save_baseline(DdlaHandle_t h,const Layout& l,const T* output) {
    std::vector<T> baseline(l.elements);const T guard=scalar<T>(-987.25,654.5);
    DDLA_CHECK(ddlaMemcpy(h,baseline.data(),output,baseline.size()*sizeof(T)));DDLA_CHECK(ddlaSynchronize(h));
    int bad=0;
    for(std::size_t p=0;p<baseline.size();++p) {
        if(l.active(p)) {
            const Wide x=wide(baseline[p]);if(!std::isfinite(x.real()) || !std::isfinite(x.imag())) bad=1;
        } else if(std::memcmp(&baseline[p],&guard,sizeof(T))) bad=1;
    }
    Errors e;merge_errors(h,e,bad,"baseline");return baseline;
}

template <typename T> void compare_full(DdlaHandle_t h,const Layout& l,const T* output,
    const std::vector<T>& baseline,long double scale,Errors& e,const Case& c) {
    // Streaming the candidate limits host workspace to one reference local
    // matrix plus a small transfer buffer, even at production matrix sizes.
    constexpr std::size_t chunk_size=1u<<18;
    std::vector<T> chunk(std::min(chunk_size,l.elements));int bad=0;
    for(std::size_t first=0;first<l.elements;first+=chunk.size()) {
        const auto count=std::min(chunk.size(),l.elements-first);
        DDLA_CHECK(ddlaMemcpy(h,chunk.data(),output+first,count*sizeof(T)));DDLA_CHECK(ddlaSynchronize(h));
        for(std::size_t q=0;q<count;++q) {
            const auto p=first+q;
            const bool untouched=c.symmetric && (c.a=='U' ? l.row(int(p%l.ld))>l.col(int(p/l.ld)) : l.row(int(p%l.ld))<l.col(int(p/l.ld)));
            if(l.active(p) && !untouched) {
                const long double delta=std::abs(wide(chunk[q])-wide(baseline[p]));
                if(!std::isfinite(delta)) bad=1;
                else {e.absolute=std::max(e.absolute,delta);e.normalized=std::max(e.normalized,delta/scale);}
            } else {
                const T expected=l.active(p) ? scalar<T>(std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN()) : baseline[p];
                if(std::memcmp(&chunk[q],&expected,sizeof(T))) bad=1;
            }
        }
    }
    merge_errors(h,e,bad,"candidate versus complete baseline");
}

template <typename Fn> double time_call(DdlaHandle_t h,Fn&& call) {
    DDLA_CHECK(ddlaSynchronize(h));MPI_Barrier(ddlaGetCommunicator(h));
    const double start=MPI_Wtime();call();DDLA_CHECK(ddlaSynchronize(h));
    const double elapsed=MPI_Wtime()-start;double maximum=0;
    MPI_Allreduce(&elapsed,&maximum,1,MPI_DOUBLE,MPI_MAX,ddlaGetCommunicator(h));
    if(!std::isfinite(maximum) || maximum<=0) throw std::runtime_error("invalid benchmark duration");
    return maximum;
}
double median(std::vector<double> v) {
    std::sort(v.begin(),v.end());const auto mid=v.size()/2;
    return v.size()%2 ? v[mid] : (v[mid-1]+v[mid])*.5;
}

void emit(const Options& o,const Layout& l,const Case& c,const char* record,
    const char* function,const char* baseline,int threads,int rep,double seconds,
    double baseline_seconds,double minimum,double maximum,const Errors& errors) {
    const std::string structure=c.symmetric ? std::string(1,c.a) : std::string()+c.a+c.b;
    const long double n=o.n,flops=(o.scalar=='s' || o.scalar=='d' ? 2.L : 8.L)*n*n*n;
    std::cout<<record<<','<<function<<','<<baseline<<','<<c.name<<','<<structure<<','<<o.scalar<<','<<o.backend<<','<<o.api
        <<','<<o.n<<','<<o.n<<','<<o.nb<<','<<l.pr<<'x'<<l.pc<<','<<l.pr*l.pc<<','<<threads
        <<','<<o.warmups<<','<<o.repeats<<','<<rep<<','<<seconds<<','<<double(flops/seconds/1e9L)
        <<','<<double(errors.absolute)<<','<<double(errors.normalized)<<','<<baseline_seconds
        <<','<<baseline_seconds/seconds<<','<<minimum<<','<<maximum<<'\n';
}

template <DdlaBackend Backend,typename T>
void benchmark_case(DdlaHandle_t h,const Options& o,const Layout& l,const Case& c,
    ddla::vector<T>& a,ddla::vector<T>& b,ddla::vector<T>& output) {
    const bool distributed=o.api=="distributed";
    const double max_a=upload_input<T>(h,l,c,true,a.data()),max_b=upload_input<T>(h,l,c,false,b.data());
    const long double scale=std::max(1.L,(long double)o.n*max_a*max_b);
    poison_output<T>(h,l,output.data());
    const auto candidate=[&] {
        if(distributed) {
            if(c.symmetric) pgemmt<Backend,T>(h,c.a,'N','N',o.n,o.n,T(1),a.data(),l.d,b.data(),l.d,T(0),output.data(),l.d);
            else ptrtrm<Backend,T>(h,c.a,'N',c.b,'N',o.n,T(1),a.data(),l.d,b.data(),l.d,T(0),output.data(),l.d);
        } else {
            if(c.symmetric) gemmt<Backend,T>(h,c.a,'N','N',o.n,o.n,T(1),a.data(),l.ld,b.data(),l.ld,T(0),output.data(),l.ld);
            else trtrm<Backend,T>(h,c.a,'N',c.b,'N',o.n,T(1),a.data(),l.ld,b.data(),l.ld,T(0),output.data(),l.ld);
        }
    };
    const auto baseline=[&] {
        if(distributed) pgemm<Backend,T>(h,'N','N',o.n,o.n,o.n,T(1),a.data(),l.d,b.data(),l.d,T(0),output.data(),l.d);
        else gemm<Backend,T>(h,'N','N',o.n,o.n,o.n,T(1),a.data(),l.ld,b.data(),l.ld,T(0),output.data(),l.ld);
    };
    const auto points=sample_positions(l);Errors errors;
    baseline();DDLA_CHECK(ddlaSynchronize(h));
    const auto reference=save_baseline<T>(h,l,output.data());
    sampled_oracle<T>(h,l,c,output.data(),points,errors);
    poison_output<T>(h,l,output.data());
    candidate();DDLA_CHECK(ddlaSynchronize(h));
    compare_full<T>(h,l,output.data(),reference,scale,errors,c);
    sampled_oracle<T>(h,l,c,output.data(),points,errors,true);
    check_inputs<T>(h,l,c,a.data(),b.data(),points);
    using R=typename ddla::real_type<T>::type;
    const auto verify=[&] {
        if(errors.normalized>64*std::numeric_limits<R>::epsilon()) {
            if(ddlaGetRank(h)==0) std::cerr<<"BENCHMARK_VALIDATION_FAILURE case="<<c.name<<" scalar="<<o.scalar
                <<" abs="<<double(errors.absolute)<<" normalized="<<double(errors.normalized)<<std::endl;
            throw std::runtime_error("benchmark numerical tolerance exceeded");
        }
    };
    verify();
    for(int r=0;r<o.warmups;++r) {
        if(r%2) {baseline();candidate();} else {candidate();baseline();}
        DDLA_CHECK(ddlaSynchronize(h));
    }
    std::vector<double> candidate_times(o.repeats),baseline_times(o.repeats);
    for(int r=0;r<o.repeats;++r) {
        if(r%2) {baseline_times[r]=time_call(h,baseline);candidate_times[r]=time_call(h,candidate);}
        else {candidate_times[r]=time_call(h,candidate);baseline_times[r]=time_call(h,baseline);}
    }
    // Final independent calls validate repeated-use behavior outside timing.
    baseline();DDLA_CHECK(ddlaSynchronize(h));
    sampled_oracle<T>(h,l,c,output.data(),points,errors);
    poison_output<T>(h,l,output.data());
    candidate();DDLA_CHECK(ddlaSynchronize(h));
    compare_full<T>(h,l,output.data(),reference,scale,errors,c);
    sampled_oracle<T>(h,l,c,output.data(),points,errors,true);
    check_inputs<T>(h,l,c,a.data(),b.data(),points);verify();
    if(ddlaGetRank(h)==0) {
        const char* name=c.symmetric ? (distributed ? "pgemmt" : "gemmt") : (distributed ? "ptrtrm" : "trtrm");
        const char* base=distributed ? "pgemm" : "gemm";
        for(int r=0;r<o.repeats;++r) {
            emit(o,l,c,"sample",name,base,h->host_threads,r,candidate_times[r],baseline_times[r],candidate_times[r],candidate_times[r],errors);
            emit(o,l,c,"sample",base,base,h->host_threads,r,baseline_times[r],baseline_times[r],baseline_times[r],baseline_times[r],errors);
        }
        const auto candidate_bounds=std::minmax_element(candidate_times.begin(),candidate_times.end());
        const auto baseline_bounds=std::minmax_element(baseline_times.begin(),baseline_times.end());
        const double base_median=median(baseline_times);
        emit(o,l,c,"summary",name,base,h->host_threads,-1,median(candidate_times),base_median,*candidate_bounds.first,*candidate_bounds.second,errors);
        emit(o,l,c,"summary",base,base,h->host_threads,-1,base_median,base_median,*baseline_bounds.first,*baseline_bounds.second,errors);
        std::cout<<std::flush;
    }
}

template <DdlaBackend Backend,typename T> void run(const Options& o,int pr,int pc) {
    DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,Backend));DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,pr,pc));
    {
        // The same per-rank budget applies to DDLA's host loops and the host
        // BLAS; the helper restores any earlier BLAS setting on exit.
        detail::ScopedBlasThreads thread_budget(h->host_threads);
        const Layout l(h,o);
        ddla::vector<T> a(h,l.elements),b(h,l.elements),output(h,l.elements);
        for(const Case& c:std::vector<Case>{{true,'U','U',"symU"},{true,'L','L',"symL"},
                                           {false,'U','L',"triUL"},{false,'L','U',"triLU"}})
            if(o.cases=="all" || o.cases==(c.symmetric ? "sym" : "tri"))
                benchmark_case<Backend,T>(h,o,l,c,a,b,output);
    }
    DDLA_CHECK(ddlaDestroy(h));
}
template <DdlaBackend Backend> void dispatch(const Options& o,int pr,int pc) {
    switch(o.scalar) {
        case 's':run<Backend,float>(o,pr,pc);break;
        case 'd':run<Backend,double>(o,pr,pc);break;
        case 'c':run<Backend,std::complex<float>>(o,pr,pc);break;
        case 'z':run<Backend,std::complex<double>>(o,pr,pc);break;
        default:throw std::invalid_argument("scalar must be s, d, c, or z");
    }
}
} // namespace

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);
    try {
        if(argc<5 || argc>8) throw std::invalid_argument("usage: benchmark_products cpu|gpu local|distributed s|d|c|z n [nb=128] [warmups=3] [repeats=9]");
        Options o{argv[1],argv[2],argv[3][0],number(argv[4],"matrix order",1)};
        if(std::strlen(argv[3])!=1) throw std::invalid_argument("scalar must be one of s,d,c,z");
        if(o.api!="local" && o.api!="distributed") throw std::invalid_argument("API must be local or distributed");
        if(argc>5) o.nb=number(argv[5],"block size",1);
        if(argc>6) o.warmups=number(argv[6],"warmups",0);
        if(argc>7) o.repeats=number(argv[7],"repeats",1);
        if(const char* cases=std::getenv("DDLA_BENCH_CASES")) {
            o.cases=cases;
            if(o.cases!="all" && o.cases!="sym" && o.cases!="tri")
                throw std::invalid_argument("DDLA_BENCH_CASES must be all, sym, or tri");
        }
        if(const char* threads=std::getenv("DDLA_BENCH_THREADS")) {
            number(threads,"thread budget",1);setenv("OMP_NUM_THREADS",threads,1);setenv("DDLA_EIG_THREADS",threads,1);
        } else if(!std::getenv("OMP_NUM_THREADS")) setenv("OMP_NUM_THREADS","1",0);
        int size,rank;MPI_Comm_size(MPI_COMM_WORLD,&size);MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        if(o.api=="local" && size!=1) throw std::invalid_argument("local benchmark requires exactly one MPI rank");
        int pr=int(std::sqrt(double(size)));while(size%pr) --pr;int pc=size/pr;
        if(const char* grid=std::getenv("DDLA_BENCH_GRID")) {
            const std::string text=grid;const auto split=text.find('x');
            if(split==std::string::npos) throw std::invalid_argument("DDLA_BENCH_GRID must be rowsxcolumns");
            pr=number(text.substr(0,split).c_str(),"process rows",1);pc=number(text.substr(split+1).c_str(),"process columns",1);
            if((long long)pr*pc!=size) throw std::invalid_argument("process grid differs from MPI rank count");
        }
        if(rank==0) std::cout<<std::setprecision(17)
            <<"record,function,baseline,case,symmetry,scalar,backend,api,n,k,nb,grid,ranks,threads,warmups,repeats,rep,seconds,dense_equivalent_gflops,max_absolute_error,normalized_error,baseline_seconds,speedup,min_seconds,max_seconds\n";
        if(o.backend=="cpu") dispatch<DdlaBackend::CPU>(o,pr,pc);else
#if DDLA_HAS_GPU
        if(o.backend=="gpu") dispatch<DdlaBackend::GPU>(o,pr,pc);else
#endif
        throw std::invalid_argument("requested backend unavailable");
        MPI_Finalize();return 0;
    } catch(const std::exception& error) {
        std::cerr<<"BENCHMARK_PRODUCTS_ERROR "<<error.what()<<std::endl;
        MPI_Abort(MPI_COMM_WORLD,1);return 1;
    }
}
