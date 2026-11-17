#include "api_grid_test_common.h"
#include "lapack_impl.h"
#include <array>
#include <chrono>
#include <cstring>
#include <type_traits>
using namespace api_grid_test;
namespace {
// A = S (I-2uu^T) diag(1,...,n) (I-2uu^T) S^H, u_i=1/sqrt(n).
// The exact spectrum is 1,...,n. S has exact unit entries, including imaginary
// phases for complex types. No vendor solve supplies the expected values.
// Zero matrices also check repeated spectra without choosing a preferred basis.
template<class T> T phase(int i) {
    using R=typename real_type<T>::type;
    if constexpr(std::is_same_v<T,R>) return i%2?T(-1):T(1);
    else { const T p[4]={T(1),T(0,1),T(-1),T(0,-1)}; return p[i%4]; }
}
template<class T> T adjoint(T x) {
    if constexpr(std::is_same_v<T,typename real_type<T>::type>) return x;
    else return std::conj(x);
}
template<class T> T entry(int i,int j,int n) {
    using R=typename real_type<T>::type;
    const R value=(i==j?R(i+1):R(0))+R(2)*R(n-i-j-1)/R(n);
    return T(value)*phase<T>(i)*adjoint(phase<T>(j));
}
template<class T> std::complex<double> wide(T x) {
    if constexpr(std::is_same_v<T,typename real_type<T>::type>) return {double(x),0};
    else return {double(x.real()),double(x.imag())};
}
template<class T> void one(DdlaHandle_t h,int n,char job,char uplo,bool benchmark,bool zero=false) {
    using R=typename real_type<T>::type;
    constexpr int guard=4;
    const int lda=n+5;
    const std::size_t matrix_elements=std::size_t(lda)*(n+2)+2*guard;
    std::vector<T> initial(matrix_elements,T(-919));
    const R nan=std::numeric_limits<R>::quiet_NaN();
    for(int j=0;j<n;++j) for(int i=0;i<n;++i)
        initial[guard+i+std::size_t(j)*lda]=(uplo=='U'?i<=j:i>=j)?(zero?T(0):entry<T>(i,j,n)):T(nan);
    DeviceBuffer<T> a(h,matrix_elements);
    DeviceBuffer<R> values(h,n+2*guard);
    DeviceBuffer<int> info(h,2*guard+1);
    const std::vector<R> original_values(n+2*guard,R(-313));
    const std::vector<int> original_info(2*guard+1,-717);
    upload(h,a.ptr,initial); upload(h,values.ptr,original_values); upload(h,info.ptr,original_info);
    int lwork=0,lrwork=0,liwork=0; std::size_t padding=0;
    lapack_impl::heevd_workspace_size<T>(h,job,uplo,n,a.ptr+guard,lda,values.ptr+guard,
        lwork,lrwork,liwork,padding);
    const auto query_matrix=download(h,a.ptr,matrix_elements);
    const auto query_values=download(h,values.ptr,values.count);
    const auto query_info=download(h,info.ptr,info.count);
    require_close(h,"fresh local EVD query preserves caller bytes",
        !std::memcmp(initial.data(),query_matrix.data(),initial.size()*sizeof(T)) &&
        !std::memcmp(original_values.data(),query_values.data(),original_values.size()*sizeof(R)) &&
        !std::memcmp(original_info.data(),query_info.data(),original_info.size()*sizeof(int))?0:1,0);
    // Only caller A/w/info are initialized. Public syevd/heevd owns its fresh
    // workspace allocation; this fixture never obtains or initializes it.
    std::array<double,5> seconds{};
    const int repeats=benchmark?6:1;
    for(int repeat=0;repeat<repeats;++repeat) {
        upload(h,a.ptr,initial); upload(h,values.ptr,original_values); upload(h,info.ptr,original_info);
        DDLA_CHECK(ddlaSynchronize(h));
        const auto begin=std::chrono::steady_clock::now();
        if constexpr(std::is_same_v<T,R>)
            syevd<T>(h,job,uplo,n,a.ptr+guard,lda,values.ptr+guard,info.ptr+guard);
        else heevd<T>(h,job,uplo,n,a.ptr+guard,lda,values.ptr+guard,info.ptr+guard);
        DDLA_CHECK(ddlaSynchronize(h));
        const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        if(benchmark && repeat) seconds[repeat-1]=elapsed;
        const auto out=download(h,a.ptr,matrix_elements);
        const auto w=download(h,values.ptr,values.count);
        const auto status=download(h,info.ptr,info.count);
        bool okay=status[guard]==0;
        for(std::size_t i=0;i<out.size();++i) {
            const bool active=i>=guard && i<std::size_t(guard)+std::size_t(lda)*n &&
                (i-guard)%lda<std::size_t(n);
            if(!active) okay=okay && !std::memcmp(&out[i],&initial[i],sizeof(T));
        }
        for(std::size_t i=0;i<w.size();++i)
            if(i<guard || i>=std::size_t(guard+n)) okay=okay && w[i]==original_values[i];
        for(std::size_t i=0;i<status.size();++i)
            if(i!=guard) okay=okay && status[i]==original_info[i];
        double spectrum=0,residual=0,orthogonality=0;
        for(int j=0;j<n;++j) {
            okay=okay && std::isfinite(w[guard+j]);
            const double error=std::abs(double(w[guard+j])-double(zero?0:j+1))/std::max(1,n);
            if(!std::isfinite(error)) okay=false;
            else spectrum=std::max(spectrum,error);
        }
        if(job=='V') {
            for(int j=0;j<n;++j) for(int i=0;i<n;++i)
                okay=okay && std::isfinite(std::abs(out[guard+i+std::size_t(j)*lda]));
            // Full independent O(n^3) checks are confined to bounded correctness
            // cases. Benchmark validation retains all values and every guard.
            if(!benchmark) for(int j=0;j<n;++j) for(int i=0;i<n;++i) {
                std::complex<double> az=0,gram=0;
                for(int k=0;k<n;++k) {
                    az+=wide(zero?T(0):entry<T>(i,k,n))*wide(out[guard+k+std::size_t(j)*lda]);
                    gram+=std::conj(wide(out[guard+k+std::size_t(i)*lda]))*wide(out[guard+k+std::size_t(j)*lda]);
                }
                const double r=std::abs(az-wide(out[guard+i+std::size_t(j)*lda])*double(w[guard+j]))/std::max(1,n);
                const double o=std::abs(gram-(i==j?1.:0.));
                if(!std::isfinite(r)||!std::isfinite(o)) okay=false;
                else {residual=std::max(residual,r);orthogonality=std::max(orthogonality,o);}
            }
        }
        require_close(h,"fresh local EVD info/finite/all padding guards",okay?0:1,0);
        const double tolerance=std::is_same_v<R,float>?3e-5:3e-12;
        require_close(h,"fresh local EVD analytic all-n spectrum",spectrum,tolerance);
        require_close(h,"fresh local EVD original residual",residual,tolerance);
        require_close(h,"fresh local EVD orthogonality",orthogonality,tolerance);
    }
    std::cout<<"HEEVD_FRESH_CASE scalar_bytes="<<sizeof(T)<<" real="<<std::is_same_v<T,R>
        <<" n="<<n<<" lda="<<lda<<" job="<<job<<" uplo="<<uplo<<" lwork="<<lwork
        <<" work_bytes="<<std::size_t(lwork)*sizeof(T)<<" padding_elements="<<padding
        <<" benchmark="<<benchmark<<" zero="<<zero<<std::endl;
    if(benchmark) {
        std::cout<<"HEEVD_FRESH_TIME scalar_bytes="<<sizeof(T)<<" real="<<std::is_same_v<T,R>
            <<" n="<<n<<" warmup=1 samples=";
        for(double t:seconds) std::cout<<t<<",";
        std::sort(seconds.begin(),seconds.end());
        std::cout<<" median="<<seconds[2]<<" validation=all_values_and_guards"<<std::endl;
    }
}
template<class T> void cases(DdlaHandle_t h,bool benchmark) {
    if(benchmark) one<T>(h,2048,'V','L',true);
    else {
        for(int n:{1,2,3,5,17,33,65}) for(char uplo:{'U','L'}) for(char job:{'N','V'})
            one<T>(h,n,job,uplo,false);
        for(int n:{3,5,17}) for(char uplo:{'U','L'}) for(char job:{'N','V'})
            one<T>(h,n,job,uplo,false,true);
    }
}
}
int main(int argc,char** argv) {
    int provided=0; MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    if(provided<MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD,2);
    int ranks=0; MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    bool gpu=false,benchmark=false; std::string scalar="all";
    for(int i=1;i<argc;++i) {
        if(!std::strcmp(argv[i],"--gpu")) gpu=true;
        else if(!std::strcmp(argv[i],"--benchmark")) benchmark=true;
        else if(!std::strcmp(argv[i],"--type") && i+1<argc) scalar=argv[++i];
        else {std::cerr<<"usage: test_heevd_fresh [--gpu] [--benchmark] [--type s|d|c|z|all]"<<std::endl;MPI_Abort(MPI_COMM_WORLD,2);}
    }
    if(ranks!=1 || (scalar!="all"&&scalar!="s"&&scalar!="d"&&scalar!="c"&&scalar!="z") || (benchmark&&!gpu))
        MPI_Abort(MPI_COMM_WORLD,2);
    try {
        DdlaHandle_t h=nullptr;
        ddlaInit(h,gpu?DdlaBackend::GPU:DdlaBackend::CPU); ddlaSet(h,MPI_COMM_WORLD,1,1,'R');
        if(scalar=="all"||scalar=="s") cases<float>(h,benchmark);
        if(scalar=="all"||scalar=="d") cases<double>(h,benchmark);
        if(scalar=="all"||scalar=="c") cases<std::complex<float>>(h,benchmark);
        if(scalar=="all"||scalar=="z") cases<std::complex<double>>(h,benchmark);
        ddlaDestroy(h);
        std::cout<<(benchmark?"HEEVD_FRESH_BENCHMARK_PASS":"HEEVD_FRESH_PASS")<<std::endl;
    } catch(const std::exception& error) {std::cerr<<error.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);}
    MPI_Finalize();return 0;
}
