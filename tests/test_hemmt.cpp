#include "pblas_structured_test_common.h"
#include "workspace_test_common.h"
#include "BLAS/hemmt_impl.h"
#include "PBLAS/phemm_impl.h"
#include <array>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
using namespace ddla;
using structured_test::Matrix;
using structured_test::scalar;
using structured_test::value;
namespace {
#ifdef DDLA_TEST_SYMMT
constexpr bool hermitian_structure=false;
constexpr const char* test_name="SYMMT";
#else
constexpr bool hermitian_structure=true;
constexpr const char* test_name="HEMMT";
#endif
template <typename T> using LocalProduct=void(*)(const DdlaHandle_t&,char,char,char,int,const T&,const T*,int,const T*,int,const T&,T*,int);
template <DdlaBackend Backend,typename T>
constexpr LocalProduct<T> local_product=hermitian_structure ? static_cast<LocalProduct<T>>(hemmt<Backend,T>) : symmt<Backend,T>;
template <typename T> using DistributedProduct=void(*)(const DdlaHandle_t&,char,char,char,int,const T&,const T*,const int*,const T*,const int*,const T&,T*,const int*);
template <DdlaBackend Backend,typename T>
constexpr DistributedProduct<T> distributed_product=hermitian_structure ? static_cast<DistributedProduct<T>>(phemmt<Backend,T>) : psymmt<Backend,T>;
using Wide=std::complex<long double>;
template <typename T> Wide wide(T z) {
    if constexpr(std::is_arithmetic_v<T>) return {z,0};
    else return {z.real(),z.imag()};
}
template <typename T> T conjugate(T z) {
    if constexpr(std::is_arithmetic_v<T> || !hermitian_structure) return z;
    else return std::conj(z);
}
template <typename T> T feature(int i,int axis,int salt) {
    return scalar<T>(((i*(3+salt)+axis*5)%17-8)/16.,((i*(5+salt)+axis*3)%13-6)/16.);
}
template <typename T> T input_a(int i,int j,bool analytic) {
    if(analytic) {
        T z=0;for(int a=0;a<2;++a)z+=feature<T>(i,a,1)*conjugate(feature<T>(j,a,1));
        return z;
    }
    const T z=value<T>(std::min(i,j),std::max(i,j),1);
    if(hermitian_structure && i==j)return scalar<T>(double(wide(z).real()));
    return i<j?z:conjugate(z);
}
template <typename T> T general(int i,int j,bool analytic) {
    if(!analytic)return value<T>(i,j,2);
    T z=0;for(int a=0;a<2;++a)z+=feature<T>(i,a,2)*feature<T>(j,a,3);
    return z;
}
void agree(DdlaHandle_t h,int bad,const std::string& message) {
    int all=0;MPI_Allreduce(&bad,&all,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h));
    if(all)throw std::runtime_error(message);
}
template <typename T> void reference(char side,int n,T alpha,bool analytic,Wide (&dot)[2][2]) {
    if(!analytic || alpha==T(0))return;
    for(int i=0;i<2;++i)for(int j=0;j<2;++j)for(int l=0;l<n;++l)
        dot[i][j]+=side=='L' ? wide(conjugate(feature<T>(l,i,1)))*wide(feature<T>(l,j,2))
                            : wide(feature<T>(l,i,3))*wide(feature<T>(l,j,1));
}
template <typename T> Wide expected(char side,int i,int j,int n,T alpha,T beta,bool analytic,const Wide (&dot)[2][2]) {
    Wide sum=0;
    if(alpha!=T(0)) {
        if(analytic)for(int a=0;a<2;++a)for(int b=0;b<2;++b)
            sum+=side=='L' ? wide(feature<T>(i,a,1))*dot[a][b]*wide(feature<T>(j,b,3))
                          : wide(feature<T>(i,a,2))*dot[a][b]*wide(conjugate(feature<T>(j,b,1)));
        else for(int l=0;l<n;++l)
            sum+=side=='L' ? wide(input_a<T>(i,l,false))*wide(general<T>(l,j,false))
                          : wide(general<T>(i,l,false))*wide(input_a<T>(l,j,false));
    }
    return wide(alpha)*sum+(beta==T(0)?Wide(0):wide(beta)*wide(value<T>(i,j,3)));
}
template <DdlaBackend Backend,typename T>
void run_case(DdlaHandle_t h,bool local,char side,char ua,char uc,int n,int nb,int mode,bool analytic,bool borrowed,int& cases) {
    int pr,pc;ddlaGetGridDims(h,pr,pc);
    T alpha=mode>=2?T(0):scalar<T>(.75,.125);
    T beta=mode==1||mode==3?T(0):mode==4?T(1):scalar<T>(.25,-.125);
    Matrix<T> a(h,n,n,nb,pr-1,pc-1,true),b(h,n,n,nb,0,pc-1,true),c(h,n,n,nb,pr/2,pc/2,true);
    const T nan=scalar<T>(std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN());
    a.fill(n,n,[&](int i,int j){
        if(ua=='U'?i>j:i<j)return nan;
        T z=input_a<T>(i,j,analytic);
        if constexpr(!std::is_arithmetic_v<T> && hermitian_structure)if(i==j)z.imag(std::numeric_limits<typename T::value_type>::quiet_NaN());
        return z;
    });
    b.fill(n,n,[&](int i,int j){return general<T>(i,j,analytic);});
    c.fill(n,n,[&](int i,int j){return (uc=='U'?i>j:i<j)||beta==T(0)?nan:value<T>(i,j,3);});
    a.upload();b.upload();c.upload();
    // Local entry points use one local matrix on each rank, with a self-grid handle.
    std::size_t bytes=0,host_bytes=0;
    {
        workspace_test::Scope scope(true);
        if(local)blas_impl::hemmt_workspace_size<T>(h,n,alpha!=T(0),&bytes);
        else phemm_impl::phemm_workspace_size<T>(h,side,n,n,alpha==T(0),a.d,b.d,c.d,hermitian_structure,&bytes,&host_bytes,uc);
        const auto stats=scope.finish();
        agree(h,stats.ddla_alloc||stats.ddla_free||stats.host_new||stats.pinned_alloc||stats.raw_gpu_alloc||stats.vendor_query||stats.resource_setup||stats.copies||stats.launches||stats.syncs||stats.collectives,
            std::string(test_name)+" workspace query performed work");
    }
    ddla::vector<char> work(h,borrowed?bytes+512:0);std::vector<char> host(borrowed?host_bytes+512:0,char(0x5a));
    std::array<char,256> guard;guard.fill(char(0x5a));
    if(borrowed){DDLA_CHECK(ddlaMemcpy(h,work.data(),guard.data(),256));DDLA_CHECK(ddlaMemcpy(h,work.data()+256+bytes,guard.data(),256));}
    const auto has=[&](const Matrix<T>& x){return num_loc(n,nb,x.r,x.d[DDLA_RSRC_],pr)>0&&num_loc(n,nb,x.c,x.d[DDLA_CSRC_],pc)>0;};
    Wide dot[2][2]={};reference<T>(side,n,alpha,analytic,dot);
    const int repeats=borrowed?3:1;
    for(int repeat=0;repeat<repeats;++repeat) {
        c.upload();
        if(borrowed) {
            workspace_test::Scope scope;
            if(local)blas_impl::hemmt_run<Backend,T>(h,side,ua,uc,n,alpha,a.data.data(),a.d[DDLA_LLD_],b.data.data(),b.d[DDLA_LLD_],beta,c.data.data(),c.d[DDLA_LLD_],work.data()+256,bytes,hermitian_structure);
            else phemm_impl::phemm_run<Backend,T>(h,side,ua,n,n,alpha,a.data.data(),a.d,b.data.data(),b.d,beta,c.data.data(),c.d,hermitian_structure,work.data()+256,bytes,host.data()+256,host_bytes,uc);
            const auto stats=scope.finish();
            bool cold_tunnel=false;
#if defined(DDLA_USE_GPU_CPU_TUNNEL)
            // The existing communication cache grows on its first larger transfer.
            // Validate that result too, then require allocation-free reuse twice.
            cold_tunnel=Backend==DdlaBackend::GPU && !local && repeat==0;
#endif
            agree(h,stats.ddla_alloc||stats.ddla_free||(!cold_tunnel&&stats.host_new)||stats.pinned_alloc||stats.raw_gpu_alloc||stats.vendor_query||stats.resource_setup,std::string(test_name)+" run allocated scratch or queried resources");
        } else {
            const auto flag=[&](char x){return mode==1?char(std::tolower(static_cast<unsigned char>(x))):x;};
            if(local)local_product<Backend,T>(h,flag(side),flag(ua),flag(uc),n,alpha,n&&alpha!=T(0)?a.data.data():nullptr,a.d[DDLA_LLD_],n&&alpha!=T(0)?b.data.data():nullptr,b.d[DDLA_LLD_],beta,n?c.data.data():nullptr,c.d[DDLA_LLD_]);
            else distributed_product<Backend,T>(h,flag(side),flag(ua),flag(uc),n,alpha,has(a)&&alpha!=T(0)?a.data.data():nullptr,a.d,has(b)&&alpha!=T(0)?b.data.data():nullptr,b.d,beta,has(c)?c.data.data():nullptr,c.d);
        }
        auto out=c.download();auto aa=a.download();auto bb=b.download();int bad=0;
        if(std::memcmp(aa.data(),a.host.data(),aa.size()*sizeof(T))||std::memcmp(bb.data(),b.host.data(),bb.size()*sizeof(T)))bad=1;
        for(std::size_t p=0;p<out.size();++p) {
            const int i=int(p%c.d[DDLA_LLD_]),j=int(p/c.d[DDLA_LLD_]);
            const int gi=c.gi(i),gj=c.gj(j);
            if(i<c.nr&&j<c.nc&&gi<n&&gj<n&&(uc=='U'?gi<=gj:gi>=gj)) {
                const Wide want=expected<T>(side,gi,gj,n,alpha,beta,analytic,dot);
                const long double error=std::abs(wide(out[p])-want);
                const long double scale=1+std::abs(want)+2*n*std::abs(wide(alpha));
                using R=typename real_type<T>::type;
                if(!std::isfinite(error)||error>64*std::numeric_limits<R>::epsilon()*scale) {
                    if(!bad)std::cerr<<test_name<<" mismatch local="<<local<<" type="<<sizeof(T)<<" flags="<<side<<ua<<uc<<" n="<<n<<" nb="<<nb<<" borrowed="<<borrowed<<" at="<<gi<<","<<gj<<" actual="<<double(wide(out[p]).real())<<","<<double(wide(out[p]).imag())<<" expected="<<double(want.real())<<","<<double(want.imag())<<std::endl;
                    bad=1;
                }
            } else if(std::memcmp(&out[p],&c.host[p],sizeof(T)))bad=1;
        }
        agree(h,bad,std::string(test_name)+" result, input or output guard check failed");
    }
    if(borrowed) {
        std::array<char,256> first,last;DDLA_CHECK(ddlaMemcpy(h,first.data(),work.data(),256));DDLA_CHECK(ddlaMemcpy(h,last.data(),work.data()+256+bytes,256));DDLA_CHECK(ddlaSynchronize(h));
        int bad=first!=guard||last!=guard;
        for(int i=0;i<256;++i)if(host[i]!=char(0x5a)||host[256+host_bytes+i]!=char(0x5a))bad=1;
        agree(h,bad,std::string(test_name)+" workspace guard changed");
    }
    ++cases;
}
template <DdlaBackend Backend,typename T> void scalar_cases(DdlaHandle_t h,bool local,bool large,int& cases) {
    for(char side:{'L','R'})for(char ua:{'U','L'})for(char uc:{'U','L'}) {
        for(int n:{0,1,3,17})for(int mode=0;mode<5;++mode)run_case<Backend,T>(h,local,side,ua,uc,n,4,mode,false,false,cases);
        run_case<Backend,T>(h,local,side,ua,uc,17,4,0,false,true,cases);
        if(large) {
            run_case<Backend,T>(h,local,side,ua,uc,513,129,1,true,true,cases);
            run_case<Backend,T>(h,local,side,ua,uc,1031,200,0,true,true,cases);
        }
    }
}
template <DdlaBackend Backend> void invalid_cases(DdlaHandle_t h) {
    using T=std::complex<double>;
    Matrix<T> a(h,7,7,4,0,0,true),b(h,7,7,4,0,0,true),c(h,7,7,4,0,0,true);
    const auto reject=[&](auto call) {int caught=0;try{call();}catch(const std::exception&){caught=1;}agree(h,!caught,std::string(test_name)+" invalid arguments accepted");};
    const auto call=[&](char side,char ua,char uc,int n,T* output) {
        distributed_product<Backend,T>(h,side,ua,uc,n,T(1),a.data.data(),a.d,b.data.data(),b.d,T(0),output,c.d);
    };
    reject([&]{call('X','U','U',7,c.data.data());});
    reject([&]{call('L','X','U',7,c.data.data());});
    reject([&]{call('L','U','F',7,c.data.data());});
    reject([&]{call('L','U','U',-1,c.data.data());});
    reject([&]{call('L','U','U',7,a.data.data());});
    int rank=0,size=0;MPI_Comm_rank(ddlaGetCommunicator(h),&rank);MPI_Comm_size(ddlaGetCommunicator(h),&size);
    if(size>1)reject([&]{call('L','U',rank==0?'U':'L',7,c.data.data());});
    distributed_product<Backend,T>(h,'L','U','L',7,T(0),nullptr,a.d,nullptr,b.d,T(1),nullptr,c.d);
    reject([&]{local_product<Backend,T>(h,'L','U','F',7,T(1),a.data.data(),a.d[DDLA_LLD_],b.data.data(),b.d[DDLA_LLD_],T(0),c.data.data(),c.d[DDLA_LLD_]);});
    local_product<Backend,T>(h,'L','U','U',7,T(0),nullptr,7,nullptr,7,T(1),nullptr,7);
    reject([&]{local_product<Backend,T>(h,'L','U','U',7,T(1),nullptr,7,b.data.data(),7,T(0),c.data.data(),7);});
    reject([&]{local_product<Backend,T>(h,'L','U','U',7,T(1),a.data.data(),7,nullptr,7,T(0),c.data.data(),7);});
    reject([&]{local_product<Backend,T>(h,'L','U','U',7,T(0),nullptr,7,nullptr,7,T(0),nullptr,7);});
    reject([&]{local_product<Backend,T>(h,'L','U','U',7,T(0),nullptr,6,nullptr,7,T(1),nullptr,7);});
    reject([&]{call('L','U','U',7,nullptr);});
}
template <DdlaBackend Backend> void run(int pr,int pc,bool large,char order) {
    DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,Backend));DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,pr,pc,order));
    int cases=0;
    invalid_cases<Backend>(h);
    scalar_cases<Backend,float>(h,false,large,cases);scalar_cases<Backend,double>(h,false,large,cases);
    scalar_cases<Backend,std::complex<float>>(h,false,large,cases);scalar_cases<Backend,std::complex<double>>(h,false,large,cases);
    DDLA_CHECK(ddlaDestroy(h));
    DDLA_CHECK(ddlaInit(h,Backend));DDLA_CHECK(ddlaSet(h,MPI_COMM_SELF,1,1));
    scalar_cases<Backend,float>(h,true,large,cases);scalar_cases<Backend,double>(h,true,large,cases);
    scalar_cases<Backend,std::complex<float>>(h,true,large,cases);scalar_cases<Backend,std::complex<double>>(h,true,large,cases);
    DDLA_CHECK(ddlaDestroy(h));int rank=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if(rank==0)std::cout<<test_name<<"_PASS backend="<<int(Backend)<<" grid="<<pr<<"x"<<pc<<" cases="<<cases<<" large="<<large<<std::endl;
}
}
int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);
    try {
        const std::string mode=argc>1?argv[1]:"all";int size=0;MPI_Comm_size(MPI_COMM_WORLD,&size);
        const int pr=argc>2?std::stoi(argv[2]):1,pc=argc>3?std::stoi(argv[3]):size;
        if(pr<=0||pc<=0||pr*pc!=size)throw std::runtime_error(std::string(test_name)+" invalid test grid");
        const bool large=std::getenv("DDLA_HEMMT_LARGE_TEST")!=nullptr;
        const char order=argc>4?argv[4][0]:'R';
        if(order!='R'&&order!='C')throw std::runtime_error("invalid grid order");
        if(mode!="gpu")run<DdlaBackend::CPU>(pr,pc,large,order);
#if DDLA_HAS_GPU
        if(mode!="cpu")run<DdlaBackend::GPU>(pr,pc,large,order);
#endif
        MPI_Finalize();return 0;
    } catch(const std::exception& e){std::cerr<<test_name<<"_TEST_FAILURE "<<e.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
