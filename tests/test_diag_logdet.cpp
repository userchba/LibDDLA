#include "api_grid_test_common.h"
#include <cstring>
#include <type_traits>
using namespace api_grid_test;
namespace {
int failures = 0, cases = 0;
void check(bool okay, const char* message) {
    ++cases;
    if (!okay) { ++failures; std::cerr << message << std::endl; }
}
template<class T> T value(double x, double y=0) {
    using R=typename real_type<T>::type;
    if constexpr(std::is_same<T,std::complex<R>>::value) return {R(x),R(y)};
    else return R(x);
}
template<class T> std::complex<long double> wide(T x) {
    using R=typename real_type<T>::type;
    if constexpr(std::is_same<T,std::complex<R>>::value) return {x.real(),x.imag()};
    else return {x,0};
}
template<class T> std::complex<long double> reference(const std::vector<T>& d) {
    long double magnitude=0,phase=0;bool zero=false;
    for(auto x:d) {
        const auto z=wide(x);const auto r=std::hypot(z.real(),z.imag());
        if(r==0)zero=true;
        else {magnitude+=std::log(r);phase+=std::atan2(z.imag(),z.real());}
    }
    if(zero)return {-std::numeric_limits<long double>::infinity(),0};
    return {magnitude,std::remainder(phase,2*std::acos(-1.L))};
}
template<class R> bool matches(std::complex<R> actual,std::complex<long double> expected) {
    if(std::isinf(expected.real()))return std::isinf(actual.real()) && actual.real()<0 && actual.imag()==0;
    const long double tolerance=std::is_same<R,float>::value?2e-5L:3e-12L;
    return std::isfinite(actual.real()) && std::isfinite(actual.imag()) &&
        std::abs(actual.real()-expected.real())<tolerance*(1+std::abs(expected.real())) &&
        std::abs(std::remainder(actual.imag()-expected.imag(),2*std::acos(-1.L)))<tolerance;
}
template<class T> void exercise(DdlaHandle_t h,const std::vector<T>& diagonal,bool invalid=false) {
    using R=typename real_type<T>::type;
    const int n=int(diagonal.size()),lda=std::max(1,n)+3;
    const T nan=value<T>(std::numeric_limits<R>::quiet_NaN());
    std::vector<T> local(std::size_t(lda)*n,nan);
    for(int i=0;i<n;++i)local[i+std::size_t(i)*lda]=diagonal[i];
    DeviceBuffer<T> a(h,local.size());upload(h,a.ptr,local);
    const std::complex<R> sentinel(91,-13);auto result=sentinel;bool threw=false;
    try{diag_logdet<T>(h,n,n?a.ptr:nullptr,lda,result);}catch(const std::exception&){threw=true;}
    check(invalid?threw&&result==sentinel:!threw&&matches(result,reference(diagonal)),"local logdet oracle/error");
    const auto after=download(h,a.ptr,local.size());
    check(local.empty()||!std::memcmp(local.data(),after.data(),local.size()*sizeof(T)),"local input changed");

    // Parent padding, rectangular blocks and rotated source ranks; all other
    // entries (including parent diagonal outside n) remain poisoned.
    int pr,pc;ddlaGetGridDims(h,pr,pc);
    for(auto blocks:{std::pair<int,int>{1,1},{2,3},{7,4}}) {
        int desc[DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(desc,h,n+2,n+3,blocks.first,blocks.second,pr-1,pc-1));
        desc[DDLA_LLD_]+=2;
        auto host=make_local<T>(h,desc,[&](int i,int j){return i==j&&i<n?diagonal[i]:nan;});
        DeviceBuffer<T> distributed(h,host.size());upload(h,distributed.ptr,host);
        result=sentinel;threw=false;
        int row,col;ddlaGetGridCoords(h,row,col);
        const bool active=num_loc(n,blocks.first,row,pr-1,pr)>0 && num_loc(n,blocks.second,col,pc-1,pc)>0;
        try{pdiag_logdet<T>(h,n,active?distributed.ptr:nullptr,desc,result);}catch(const std::exception&){threw=true;}
        check(invalid?threw&&result==sentinel:!threw&&matches(result,reference(diagonal)),"distributed logdet oracle/error");
        const auto out=download(h,distributed.ptr,host.size());
        check(host.empty()||!std::memcmp(host.data(),out.data(),host.size()*sizeof(T)),"distributed input changed");
        if(!invalid) {
            // Different valid orders must fail collectively before data work.
            result=sentinel;threw=false;
            try{pdiag_logdet<T>(h,ddlaGetRank(h)==0?n+1:n,distributed.ptr,desc,result);}
            catch(const std::exception&){threw=true;}
            if(ddlaGetSize(h)>1)check(threw&&result==sentinel,"metadata mismatch not rejected");
        }
    }
}
template<class T> void typed(DdlaHandle_t h) {
    using R=typename real_type<T>::type;
    exercise<T>(h,{});
    exercise<T>(h,{value<T>(2),value<T>(3),value<T>(5)});
    exercise<T>(h,{value<T>(-2),value<T>(3),value<T>(5)});
    exercise<T>(h,{value<T>(-2),value<T>(-3),value<T>(5)});
    exercise<T>(h,{value<T>(0),value<T>(-3),value<T>(5)});
    exercise<T>(h,{value<T>(std::numeric_limits<R>::max(),std::numeric_limits<R>::max()),
                   value<T>(std::numeric_limits<R>::denorm_min(),-std::numeric_limits<R>::denorm_min())});
    if constexpr(std::is_same<T,std::complex<R>>::value) {
        const auto z=std::polar(R(1),R(.8)*std::acos(R(-1)));
        exercise<T>(h,{z,z,z});
    }
    exercise<T>(h,{value<T>(1),value<T>(std::numeric_limits<R>::infinity())},true);
    exercise<T>(h,{value<T>(1),value<T>(std::numeric_limits<R>::quiet_NaN())},true);
    std::vector<T> d(11);
    for(int i=0;i<11;++i)d[i]=value<T>(1+i%3);
    exercise<T>(h,d);
    // Activates the local OpenMP reduction, and tests exact sign parity.
    const int n=1025;std::vector<T> matrix(std::size_t(n)*n,T(0));
    for(int i=0;i<n;++i)matrix[i+std::size_t(i)*n]=value<T>(-1);
    DeviceBuffer<T> a(h,matrix.size());upload(h,a.ptr,matrix);
    std::complex<R> result;
    diag_logdet<T>(h,n,a.ptr,n,result);
    check(matches(result,{0,std::acos(-1.L)}),"large negative parity/parallel reduction");
    bool threw=false;result={91,-13};
    try{diag_logdet<T>(h,-1,a.ptr,n,result);}catch(const std::exception&){threw=true;}
    check(threw&&result==std::complex<R>(91,-13),"negative order/result guard");
    threw=false;
    try{diag_logdet<T>(h,std::numeric_limits<int>::max(),a.ptr,std::numeric_limits<int>::max(),result);}
    catch(const std::exception&){threw=true;}
    check(threw&&result==std::complex<R>(91,-13),"overflowing matrix span/result guard");
    // Valid single column: lda is never used to address a later column.
    diag_logdet<T>(h,1,a.ptr,std::numeric_limits<int>::max(),result);
    check(matches(result,{0,std::acos(-1.L)}),"large leading dimension");
}
}
int main(int argc,char**argv) {
    ddla_test_mpi_init(&argc,&argv);
    try {
        const bool gpu=argc>1&&std::string(argv[1])=="gpu";
        const std::string transport=argc>2?argv[2]:"default";
        int size;MPI_Comm_size(MPI_COMM_WORLD,&size);
        for(int pr=1;pr<=size;++pr)if(size%pr==0)for(char major:{'R','C'}) {
            DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,gpu?DdlaBackend::GPU:DdlaBackend::CPU));
            if(transport=="staged")DDLA_CHECK(ddlaSetCommunication(h,DdlaCommunication::HostStagedMPI));
            if(transport=="ccl")DDLA_CHECK(ddlaSetCommunication(h,DdlaCommunication::CCL));
            DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,pr,size/pr,major));
            typed<float>(h);typed<double>(h);typed<std::complex<float>>(h);typed<std::complex<double>>(h);
            DDLA_CHECK(ddlaDestroy(h));
        }
        int total=0;MPI_Allreduce(&failures,&total,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
        int rank;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        if(rank==0)std::cout<<"DIAG_LOGDET_"<<(total?"FAIL":"PASS")<<" backend="<<(gpu?"gpu":"cpu")
                           <<" cases_per_rank="<<cases<<" failures="<<total<<std::endl;
        MPI_Finalize();return total?1:0;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
