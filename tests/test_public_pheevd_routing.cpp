#include "api_grid_test_common.h"
#include "eig_test_common.h"
#include "Eigensolver/two_stage/pheevd_impl.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
using namespace api_grid_test;

namespace {
template<class T>T val(double re,double im) {
    using R=typename ddla::real_type<T>::type;
    if constexpr(std::is_same<T,R>::value)return T(re);
    else return T(R(re),R(im));
}
template<class T>T adj(T x){return x;}
template<class R>std::complex<R>adj(std::complex<R>x){return std::conj(x);}
template<class T>T input_value(int i,int j) {
    if(i<j)return adj(input_value<T>(j,i));
    return i==j?val<T>(3+0.02*i,0):val<T>(0.1*std::sin(i*0.61+j*0.37),0.1*std::cos(i*0.31-j*0.27));
}
template<class T>bool finite(T x){return std::isfinite(std::abs(x));}
template<class T>const char* scalar_name() {
    if constexpr(std::is_same<T,float>::value)return "s";
    else if constexpr(std::is_same<T,double>::value)return "d";
    else if constexpr(std::is_same<T,std::complex<float>>::value)return "c";
    else return "z";
}
template<class T>double padding_and_finite(const ddla::DdlaHandle_t& h,const int* desc,
                                          const std::vector<T>& data,int n,T sentinel,bool vectors) {
    const int rows=ddla_test::m_loc(h,desc),cols=ddla_test::n_loc(h,desc),ld=desc[ddla::DDLA_LLD_];
    double bad=0;
    for(std::size_t p=0;p<data.size();++p) {
        const int r=int(p%ld),c=int(p/ld);
        const bool active=r<rows&&c<cols&&ddla::indx_l2g_r(desc,h,r)<n&&ddla::indx_l2g_c(desc,h,c)<n;
        if(active){if(vectors&&!finite(data[p]))++bad;}
        else if(data[p]!=sentinel)++bad;
    }
    return bad;
}
template<class T>double residual(const ddla::DdlaHandle_t& h,const T* A,const int* da,
                                 const T* Z,const int* dz,const typename ddla::real_type<T>::type* w,int n) {
    using namespace ddla;using R=typename real_type<T>::type;
    DeviceBuffer<T> AZ(h,std::max<std::size_t>(1,local_size(h,dz)));
    pgemm<T>(h,'N','N',n,n,n,T(1),A,da,Z,dz,T(0),AZ.ptr,dz);
    int pr=0,pc=0,mr=0,mc=0;ddlaGetGridDims(h,pr,pc);ddlaGetGridCoords(h,mr,mc);
    const int rows=num_loc(n,dz[DDLA_MB_],mr,dz[DDLA_RSRC_],pr),cols=num_loc(n,dz[DDLA_NB_],mc,dz[DDLA_CSRC_],pc),ld=dz[DDLA_LLD_];
    for(int j=0;j<cols;++j)axpy<T>(h,rows,T(-w[indx_l2g_c(dz,h,j)]),Z+static_cast<std::size_t>(j)*ld,1,AZ.ptr+static_cast<std::size_t>(j)*ld,1);
    check_ddla_sync(h);
    const double a=eig_test::frobenius_norm(h,A,da,n),r=eig_test::frobenius_norm(h,AZ.ptr,dz,n);
    return a>0?r/(n*a*eig_test::eps_trait<R>::value):std::numeric_limits<double>::infinity();
}
template<class T>double orthogonal(const ddla::DdlaHandle_t& h,const T* Z,const int* dz,int n) {
    using namespace ddla;using R=typename real_type<T>::type;
    int dg[DDLA_DLEN_];ddlaDescInit(dg,h,n,n,dz[DDLA_MB_],dz[DDLA_NB_],dz[DDLA_RSRC_],dz[DDLA_CSRC_]);
    DeviceBuffer<T> G(h,std::max<std::size_t>(1,local_size(h,dg)));
    pgemm<T>(h,'C','N',n,n,n,T(1),Z,dz,Z,dz,T(0),G.ptr,dg);
    check_ddla_sync(h);
    int pr=0,pc=0,mr=0,mc=0;ddlaGetGridDims(h,pr,pc);ddlaGetGridCoords(h,mr,mc);
    const int rows=num_loc(n,dg[DDLA_MB_],mr,dg[DDLA_RSRC_],pr),cols=num_loc(n,dg[DDLA_NB_],mc,dg[DDLA_CSRC_],pc),ld=dg[DDLA_LLD_];
    std::vector<T> local(static_cast<std::size_t>(rows)*cols);
    if(rows>0&&cols>0)DDLA_CHECK(ddlaMemcpy2D(h,local.data(),static_cast<std::size_t>(rows)*sizeof(T),G.ptr,static_cast<std::size_t>(ld)*sizeof(T),static_cast<std::size_t>(rows)*sizeof(T),cols));
    check_ddla_sync(h);double err=0;int bad=0;
    for(int j=0;j<cols;++j)for(int i=0;i<rows;++i) {
        const T x=local[i+static_cast<std::size_t>(j)*rows];
        if(!finite(x)){bad=1;continue;}
        const T expect=indx_l2g_r(dg,h,i)==indx_l2g_c(dg,h,j)?T(1):T(0);
        err=std::max(err,double(std::abs(x-expect)));
    }
    int any=0;double maximum=0;MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h));MPI_Allreduce(&err,&maximum,1,MPI_DOUBLE,MPI_MAX,ddlaGetCommunicator(h));
    return any?std::numeric_limits<double>::infinity():maximum/(n*eig_test::eps_trait<R>::value);
}
template<class T>void solve(const ddla::DdlaHandle_t& h,int n,int nb,char uplo,bool alias,bool shared_parent) {
    using namespace ddla;using R=typename real_type<T>::type;
    int pr=0,pc=0;ddlaGetGridDims(h,pr,pc);const int rank=ddlaGetRank(h);
    if(n>std::numeric_limits<int>::max()-7||(shared_parent&&n>std::numeric_limits<int>::max()/2))throw std::invalid_argument("test parent extent overflow");
    if(shared_parent&&(alias||n%nb||n%(static_cast<long long>(nb)*pc)))throw std::invalid_argument("shared parent needs complete column-grid cycles and A!=Z");
    int da[DDLA_DLEN_],dz[DDLA_DLEN_];ddlaDescInit(da,h,shared_parent?n:n+3,shared_parent?2*n:n+5,nb,nb,0,0);da[DDLA_LLD_]+=3;
    ddlaDescInit(dz,h,shared_parent?n:n+3,shared_parent?n:n+5,nb,nb,0,alias||shared_parent?0:pc-1);dz[DDLA_LLD_]+=3;
    if(alias)std::copy(da,da+DDLA_DLEN_,dz);
    const T sentinel=val<T>(139,-137);const R nan=std::numeric_limits<R>::quiet_NaN();
    auto make=[&](bool poison) {
        std::vector<T> data(std::max<std::size_t>(1,local_size(h,da))+17,sentinel);
        const int rows=ddla_test::m_loc(h,da),cols=ddla_test::n_loc(h,da),ld=da[DDLA_LLD_];
        for(int j=0;j<cols;++j)for(int i=0;i<rows;++i) {
            const int gi=indx_l2g_r(da,h,i),gj=indx_l2g_c(da,h,j);if(gi>=n||gj>=n)continue;
            T value=input_value<T>(gi,gj);
            if(poison&&((uplo=='U'&&gi>gj)||(uplo=='L'&&gi<gj)))value=val<T>(nan,nan);
            if constexpr(!std::is_same<T,R>::value)if(poison&&gi==gj)value=T(value.real(),nan);
            data[i+static_cast<std::size_t>(j)*ld]=value;
        }
        return data;
    };
    const auto input=make(true),full=make(false);
    DeviceBuffer<T>A(h,input.size()),Z(h,shared_parent?1:std::max<std::size_t>(1,local_size(h,dz))),reference(h,input.size());
    int mr=0,mc=0;ddlaGetGridCoords(h,mr,mc);
    const int active_columns=num_loc(n,nb,mc,0,pc);
    const std::size_t z_offset=static_cast<std::size_t>(da[DDLA_LLD_])*active_columns;
    T* output=shared_parent?A.ptr+z_offset:alias?A.ptr:Z.ptr;
    upload(h,A.ptr,input);upload(h,Z.ptr,std::vector<T>(Z.count,sentinel));
    const R w_guard=R(139);std::vector<R>w(n+7,w_guard),wn(n+7,w_guard),wp(n+7,w_guard);
    auto guarded=[&](const std::vector<R>& values) {
        bool good=true;for(int j=0;j<3;++j)good=good&&values[j]==w_guard;
        for(std::size_t j=n+3;j<values.size();++j)good=good&&values[j]==w_guard;
        for(int j=0;j<n;++j)good=good&&std::isfinite(values[j+3])&&(!j||values[j+3]>=values[j+2]);
        return good;
    };
    auto invoke=[&](const char* phase,auto function) {
        MPI_Barrier(ddlaGetCommunicator(h));const double start=MPI_Wtime();function();
        const double elapsed=MPI_Wtime()-start;double maximum=0;MPI_Allreduce(&elapsed,&maximum,1,MPI_DOUBLE,MPI_MAX,ddlaGetCommunicator(h));
        require_close(h,std::string("whole-call clock ")+phase,std::isfinite(maximum)&&maximum>0?0:INFINITY,0);
        if(rank==0)std::printf("PUBLIC_ROUTING_CALL phase=%s type=%s n=%d grid=%dx%d uplo=%c alias=%d shared_parent=%d maxrank_seconds=%.9f clock=whole_call status=PASS\n",phase,scalar_name<T>(),n,pr,pc,uplo,int(alias),int(shared_parent),maximum);
    };
    int info=-1;
    invoke("public_V",[&]{pheevd<T>(h,'V',uplo,n,A.ptr,da,w.data()+3,output,dz,info);});
    require_close(h,"public V info",info,0);require_close(h,"public V w guards/finite/order",guarded(w)?0:INFINITY,0);
    const auto after_a=download(h,A.ptr,input.size());
    if(shared_parent) {
        const int rows=num_loc(n,nb,mr,0,pr),ld=da[DDLA_LLD_];double bad=0;
        for(std::size_t p=0;p<after_a.size();++p) {
            const std::size_t c=p/ld;const int r=int(p%ld);
            const bool a_active=r<rows&&c<static_cast<std::size_t>(active_columns);
            const bool z_active=r<rows&&c>=static_cast<std::size_t>(active_columns)&&c<static_cast<std::size_t>(2*active_columns);
            if(z_active){if(!finite(after_a[p]))++bad;}
            else if(!a_active&&after_a[p]!=sentinel)++bad;
        }
        require_close(h,"public disjoint active parent A/Z guards and finite vectors",bad,0);
    } else require_close(h,"public V A inactive parent/LLD padding",padding_and_finite(h,da,after_a,n,sentinel,alias),0);
    if(!alias&&!shared_parent){const auto after_z=download(h,Z.ptr,Z.count);require_close(h,"public V Z finite/padding",padding_and_finite(h,dz,after_z,n,sentinel,true),0);}
    upload(h,reference.ptr,full);
    const double r=residual(h,reference.ptr,da,output,dz,w.data()+3,n),o=orthogonal(h,output,dz,n),norm=eig_test::frobenius_norm(h,reference.ptr,da,n);
    require_close(h,"public independent residual",std::isfinite(r)?r:INFINITY,50);require_close(h,"public independent orthogonality",std::isfinite(o)?o:INFINITY,50);
    upload(h,reference.ptr,input);
    invoke("public_N",[&]{pheevd<T>(h,'N',uplo,n,reference.ptr,da,wn.data()+3,nullptr,nullptr,info);});
    require_close(h,"public N info",info,0);require_close(h,"public N w guards/finite/order",guarded(wn)?0:INFINITY,0);
    require_close(h,"public N A inactive parent/LLD padding",padding_and_finite(h,da,download(h,reference.ptr,input.size()),n,sentinel,false),0);
    upload(h,reference.ptr,input);
    constexpr int method=std::is_same<T,R>::value?4:3;
    invoke("private_N",[&]{detail::pheevd_householder_stage4<DdlaBackend::CPU, T>(h,'N',uplo,n,reference.ptr,da,wp.data()+3,nullptr,nullptr,info,method);});
    require_close(h,"private N info",info,0);require_close(h,"private N w guards/finite/order",guarded(wp)?0:INFINITY,0);
    require_close(h,"private N A inactive parent/LLD padding",padding_and_finite(h,da,download(h,reference.ptr,input.size()),n,sentinel,false),0);
    double en=0,ep=0;
    for(int j=0;j<n;++j){en=std::max(en,std::abs(double(w[j+3])-wn[j+3]));ep=std::max(ep,std::abs(double(w[j+3])-wp[j+3]));}
    const double denominator=n*std::numeric_limits<R>::epsilon()*norm;
    require_close(h,"positive eigenvalue normalization",std::isfinite(denominator)&&denominator>0?0:INFINITY,0);
    en/=denominator;ep/=denominator;
    require_close(h,"public N vs public V eigenvalues",std::isfinite(en)?en:INFINITY,50);require_close(h,"matching private HH N vs public V eigenvalues",std::isfinite(ep)?ep:INFINITY,50);
    if(rank==0)std::printf("PUBLIC_PHEEVD_CASE type=%s n=%d nb=%d grid=%dx%d uplo=%c alias=%d shared_parent=%d z_csrc=%d parent=%dx%d lld_pad=3 private_method=%d residual=%.9g orthogonality=%.9g public_N_agreement=%.9g private_N_agreement=%.9g finite=PASS padding=PASS w_guard=PASS status=PASS\n",scalar_name<T>(),n,nb,pr,pc,uplo,int(alias),int(shared_parent),dz[DDLA_CSRC_],shared_parent?0:3,shared_parent?n:5,method,r,o,en,ep);
}
}

int main(int argc,char** argv) {
    int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);int rank=0,ranks=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    ddla::DdlaHandle_t h=nullptr;
    try {
        int n=4097,nb=64,pr=2,pc=4;bool alias=false,shared_parent=false;char uplo='U';std::string type="d";
        for(int i=1;i<argc;++i){if(i+1>=argc)throw std::invalid_argument("option requires value");const std::string key=argv[i],v=argv[++i];
            if(key=="--n")n=std::stoi(v);else if(key=="--nb")nb=std::stoi(v);else if(key=="--type")type=v;
            else if(key=="--uplo"){if(v!="U"&&v!="L")throw std::invalid_argument("U/L");uplo=v[0];}
            else if(key=="--alias"){if(v!="0"&&v!="1")throw std::invalid_argument("alias0/1");alias=v=="1";}
            else if(key=="--shared-parent"){if(v!="0"&&v!="1")throw std::invalid_argument("shared-parent0/1");shared_parent=v=="1";}
            else if(key=="--grid"){const auto x=v.find('x');if(x==std::string::npos)throw std::invalid_argument("grid RxC");pr=std::stoi(v.substr(0,x));pc=std::stoi(v.substr(x+1));}
            else throw std::invalid_argument("unknown option");}
        if(provided<MPI_THREAD_FUNNELED||n<1||nb<1||pr<1||pc<1||static_cast<long long>(pr)*pc!=ranks)throw std::invalid_argument("full test geometry/thread contract");
        ::setenv("DDLA_EIG_PROFILE","1",1);
        DDLA_CHECK(ddla::ddlaInit(h,ddla::DdlaBackend::CPU));DDLA_CHECK(ddla::ddlaSet(h,MPI_COMM_WORLD,pr,pc,'R'));
        if(type=="s")solve<float>(h,n,nb,uplo,alias,shared_parent);else if(type=="d")solve<double>(h,n,nb,uplo,alias,shared_parent);
        else if(type=="c")solve<std::complex<float>>(h,n,nb,uplo,alias,shared_parent);else if(type=="z")solve<std::complex<double>>(h,n,nb,uplo,alias,shared_parent);else throw std::invalid_argument("scalar s/d/c/z");
        DDLA_CHECK(ddla::ddlaDestroy(h));if(rank==0)std::puts("test_public_pheevd_routing passed");MPI_Finalize();return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"rank %d: %s\n",rank,error.what());MPI_Abort(MPI_COMM_WORLD,1);}
    return 1;
}
