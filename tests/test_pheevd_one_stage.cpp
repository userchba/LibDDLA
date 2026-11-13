#include "api_grid_test_common.h"
#include "eig_test_common.h"
#include <limits>
#include <cstring>
#include <type_traits>
using namespace api_grid_test;
namespace {
template<class T>T entry(int i,int j,int kind) {
    using R=typename ddla::real_type<T>::type;
    if(kind==1)return i==j?T(R(i%3-1)):T(0);
    if(kind==2)return T(0);
    if(i==j)return T(R(0.1*(i%7-3)));
    const int lo=std::min(i,j),hi=std::max(i,j);
    const R r=kind==3?R(0):R(0.07*((lo*13+hi*7)%17-8));
    if constexpr(std::is_same_v<T,R>)return T(r);
    else {const R im=R(0.09*((lo*5+hi*3)%11-5));return i<j?T(r,im):T(r,-im);}
}
template<class T>void run(const ddla::DdlaHandle_t& h,int n,int nb,char uplo,int kind,bool alias,bool tight=false) {
    using R=typename ddla::real_type<T>::type;
    int da[ddla::DDLA_DLEN_],dz[ddla::DDLA_DLEN_];
    ddlaDescInit(da,h,tight?n:n+2,tight?n:n+3,nb,nb,0,0);ddlaDescInit(dz,h,tight?n:n+2,tight?n:n+3,nb,nb,0,0);
    if(!tight){da[ddla::DDLA_LLD_]+=3;dz[ddla::DDLA_LLD_]+=3;}
    DeviceBuffer<T>a(h,std::max<std::size_t>(1,local_size(h,da))),z(h,std::max<std::size_t>(1,local_size(h,dz)));
    const T sentinel=T(12345);const R nan=std::numeric_limits<R>::quiet_NaN();
    auto full=make_local<T>(h,da,[&](int i,int j){return i<n&&j<n?entry<T>(i,j,kind):sentinel;});
    auto input=make_local<T>(h,da,[&](int i,int j){
        if(i>=n||j>=n)return sentinel;
        if(uplo=='U'?i>j:i<j)return T(nan);
        if constexpr(!std::is_same_v<T,R>) {if(i==j)return T(R(std::real(entry<T>(i,j,kind))),nan);}
        return entry<T>(i,j,kind);
    });
    // make_local zeroes LLD padding; retain that exact before-image too.
    upload(h,a.ptr,input);std::vector<T> poison(local_size(h,dz),T(nan));upload(h,z.ptr,poison);
    std::vector<R>w(std::max(n,1)),wn(std::max(n,1));int info=0;
    auto check_input=[&] {
        const auto after=download<T>(h,a.ptr,input.size());
        int pr,pc,mr,mc;ddlaGetGridDims(h,pr,pc);ddlaGetGridCoords(h,mr,mc);
        double changed=0;
        for(int j=0;j<ddla_test::n_loc(h,da);++j)for(int i=0;i<da[ddla::DDLA_LLD_];++i) {
            const int gi=ddla::indxl2g(i,nb,mr,0,pr),gj=ddla::indxl2g(j,nb,mc,0,pc);
            if(i>=ddla_test::m_loc(h,da)||gi>=n||gj>=n) {
                const auto at=i+static_cast<std::size_t>(j)*da[ddla::DDLA_LLD_];
                if(std::memcmp(&after[at],&input[at],sizeof(T)))changed=1;
            }
        }
        require_close(h,"full-method input padding unchanged",changed,0);
    };
    ddla::pheevd_one_stage(h,'V',uplo,n,n,a.ptr,da,w.data(),alias?a.ptr:z.ptr,alias?da:dz,info);
    require_close(h,"one-stage info",info,0);
    check_ddla_sync(h);
    if(!alias)check_input();
    auto result=download<T>(h,alias?a.ptr:z.ptr,local_size(h,alias?da:dz));
    int pr,pc,mr,mc;ddlaGetGridDims(h,pr,pc);ddlaGetGridCoords(h,mr,mc);
    const int nr=ddla::num_loc(n,nb,mr,0,pr),nc=ddla::num_loc(n,nb,mc,0,pc),ld=dz[ddla::DDLA_LLD_];
    double finite_error=0,padding_error=0;
    for(int j=0;j<ddla_test::n_loc(h,dz);++j)for(int i=0;i<ld;++i) {
        const auto at=i+static_cast<std::size_t>(j)*ld;
        if(i<nr&&j<nc)finite_error=std::max(finite_error,std::isfinite(std::abs(result[at]))?0.:1.);
        else if(alias)padding_error=std::max(padding_error,double(std::abs(result[at]-input[at])));
        else padding_error=std::max(padding_error,std::isnan(std::abs(result[at]))?0.:1.);
    }
    require_close(h,"finite output",finite_error,0);require_close(h,"output padding",padding_error,0);
    if(alias)upload(h,z.ptr,result);
    upload(h,a.ptr,full);check_ddla_sync(h);
    if(kind!=2&&n) {
        require_close(h,"one-stage residual",eig_test::residual_metric<T>(h,a.ptr,da,z.ptr,dz,w.data(),n),50);
    }
    if(n)require_close(h,"one-stage orthogonality",eig_test::orthogonality_metric<T>(h,z.ptr,dz,n),50);
    upload(h,a.ptr,input);
    ddla::pheevd_one_stage(h,'N',uplo,n,n,a.ptr,da,wn.data(),static_cast<T*>(nullptr),nullptr,info);
    check_ddla_sync(h);check_input();
    double error=0;
    for(int i=0;i<n;++i) {
        if(!std::isfinite(w[i])||!std::isfinite(wn[i])||(i&&w[i]<w[i-1]))error=1e30;
        error=std::max(error,double(std::abs(w[i]-wn[i]))/(std::max(1,n)*std::numeric_limits<R>::epsilon()));
        if(kind==2)require_close(h,"zero eigenvalues",std::abs(w[i]),0);
    }
    require_close(h,"one-stage N/V eigenvalues",error,50);
}
template<class T>void cases(const ddla::DdlaHandle_t& h) {
    for(int n:{0,1,2,3,17,65,137,257})for(char uplo:{'U','L'})run<T>(h,n,16,uplo,0,false);
    for(int kind:{1,2,3})for(char uplo:{'U','L'})run<T>(h,33,16,uplo,kind,false);
    // Tight, separate output can be lent to the solver; aliasing retains the
    // owned-workspace path. Existing parent/LLD cases above must never borrow.
    for(char uplo:{'U','L'}) {
        for(int n:{0,1,137})run<T>(h,n,16,uplo,0,false,true);
        run<T>(h,65,16,uplo,0,true,true);
    }
    run<T>(h,65,16,'U',0,true);run<T>(h,65,16,'L',0,true);run<T>(h,41,1,'U',0,false);run<T>(h,67,64,'U',0,false);
    // Descriptor-aligned reflector panels: exact boundaries, a short first
    // panel, nb == 1, and workspace capacity above the former fixed 128.
    for(char uplo:{'U','L'}) {
        for(int n:{15,16,32})run<T>(h,n,16,uplo,0,false);
        run<T>(h,5,1,uplo,0,false);
        run<T>(h,139,129,uplo,0,false);
    }
}
void body(const ddla::DdlaHandle_t& h,const Shape&) {
    int ranks=0,rank=0;MPI_Comm_size(ddlaGetCommunicator(h),&ranks);MPI_Comm_rank(ddlaGetCommunicator(h),&rank);
    if(ranks>1){
        int d[ddla::DDLA_DLEN_];ddlaDescInit(d,h,1,1,1,1,0,0);
        DeviceBuffer<double>a(h,std::max<std::size_t>(1,local_size(h,d)));double w=0;int info=0;bool caught=false;
        try {
            ddla::pheevd_one_stage(h,'N',rank==0?'U':'L',1,1,a.ptr,d,&w,static_cast<double*>(nullptr),nullptr,info);
        }catch(const std::invalid_argument&){caught=true;}
        require_close(h,"input triangle mismatch rejected collectively",caught?0:1,0);
    }
    cases<float>(h);cases<double>(h);cases<std::complex<float>>(h);cases<std::complex<double>>(h);
}
}
int main(int argc,char**argv) {return run_grid_test(argc,argv,"test_pheevd_one_stage",body);}
