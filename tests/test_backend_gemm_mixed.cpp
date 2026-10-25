#include <ddla/ddla.h>
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using namespace ddla;
namespace {
void check(ddlaStatus_t s)
{
    if (static_cast<int>(s)!=0) throw std::runtime_error("DDLA runtime failure");
}
template <typename T> struct Buffer {
    DdlaHandle_t h; T* p=nullptr;
    Buffer(DdlaHandle_t handle,const std::vector<T>& v):h(handle) {
        check(ddlaMalloc(h,&p,v.size()*sizeof(T)));
        check(ddlaMemcpy(h,p,v.data(),v.size()*sizeof(T)));
    }
    ~Buffer() { ddlaFree(h,p); }
    std::vector<T> read(std::size_t n) {
        std::vector<T> v(n);
        check(ddlaMemcpy(h,v.data(),p,n*sizeof(T))); check(ddlaSynchronize(h));
        return v;
    }
};
template <typename R> R value(std::size_t p,int salt) {
    return R(int((p*31+salt*17)%97)-48)/R(53);
}
template <typename R> std::complex<long double> wide(std::complex<R> x) {
    return {x.real(),x.imag()};
}

template <DdlaBackend BE,typename R>
int one(DdlaHandle_t h,char ta,char tb,int m,int n,int k,
        std::complex<R> alpha,std::complex<R> beta,bool poison=false,bool implicit=false)
{
    using Z=std::complex<R>;
    const bool an=ta=='N'||ta=='n', bn=tb=='N'||tb=='n', bc=tb=='C'||tb=='c';
    const int lda=std::max(1,an?m:k)+3, ldb=std::max(1,bn?k:n)+5, ldc=std::max(1,m)+7;
    // Prefix/suffix guards and padded leading dimensions detect stray writes.
    constexpr int guard=3;
    const Z sentinel(271,-193);
    std::vector<R> a(2*guard+std::size_t(lda)*std::max(1,an?k:m),R(197));
    std::vector<Z> b(2*guard+std::size_t(ldb)*std::max(1,bn?n:k),sentinel);
    std::vector<Z> c(2*guard+std::size_t(ldc)*std::max(1,n),sentinel);
    for (int j=0;j<(an?k:m);++j) for (int i=0;i<(an?m:k);++i)
        a[guard+i+std::size_t(j)*lda]=value<R>(i+std::size_t(j)*lda,1);
    for (int j=0;j<(bn?n:k);++j) for (int i=0;i<(bn?k:n);++i)
        b[guard+i+std::size_t(j)*ldb]=Z(value<R>(i+std::size_t(j)*ldb,2),value<R>(i+std::size_t(j)*ldb,3));
    for (int j=0;j<n;++j) for (int i=0;i<m;++i)
        c[guard+i+std::size_t(j)*ldc]=poison?Z(std::numeric_limits<R>::quiet_NaN(),0):Z(value<R>(i+j*ldc,4),value<R>(i+j*ldc,5));
    Buffer<R> da(h,a); Buffer<Z> db(h,b),dc(h,c);
    const bool product=k>0 && alpha!=R(0);
    // Null A/B proves no-read alpha=0 and k=0 semantics independently of NaNs.
    const R* ap=product?da.p+guard:nullptr;
    const Z* bp=product?db.p+guard:nullptr;
    if (implicit) gemm<>(h,ta,tb,m,n,k,alpha,ap,lda,bp,ldb,beta,dc.p+guard,ldc);
    else gemm<BE,R>(h,ta,tb,m,n,k,alpha,ap,lda,bp,ldb,beta,dc.p+guard,ldc);
    const auto out=dc.read(c.size());
    int failures=0;
    for (std::size_t p=0;p<c.size();++p) {
        const bool active=p>=guard && p<std::size_t(guard)+std::size_t(ldc)*n && (p-guard)%ldc<std::size_t(m);
        if (!active) { if (out[p]!=c[p]) ++failures; continue; }
        const int i=(p-guard)%ldc,j=(p-guard)/ldc;
        std::complex<long double> sum=0;
        long double scale=0;
        if (product) for (int l=0;l<k;++l) {
            const long double av=a[guard+(an?i+std::size_t(l)*lda:l+std::size_t(i)*lda)];
            auto bv=wide(b[guard+(bn?l+std::size_t(j)*ldb:j+std::size_t(l)*ldb)]);
            if (bc) bv=std::conj(bv);
            sum+=av*bv; scale+=std::abs(av)*std::abs(bv);
        }
        auto ref=product?wide(alpha)*sum:std::complex<long double>(0);
        scale*=std::abs(wide(alpha));
        if (beta!=R(0)) { ref+=wide(beta)*wide(c[p]); scale+=std::abs(wide(beta)*wide(c[p])); }
        const long double err=std::abs(wide(out[p])-ref);
        const long double tol=32*std::numeric_limits<R>::epsilon()*std::max(1.L,scale);
        if (!std::isfinite(err)||err>tol) ++failures;
    }
    if (da.read(a.size())!=a || db.read(b.size())!=b) ++failures;
    if (failures) std::cerr<<"case failed "<<int(BE)<<" "<<sizeof(R)<<" "<<ta<<tb<<" "<<m<<","<<n<<","<<k<<" alpha="<<alpha<<" beta="<<beta<<" count="<<failures<<"\n";
    return failures?1:0;
}
template <DdlaBackend BE,typename R> int run_type(DdlaHandle_t h) {
    using Z=std::complex<R>; int fails=0,cases=0;
    const int shapes[][3]={{1,1,1},{7,5,9},{65,35,17},{129,67,33},{257,33,67},{17,513,19},{31,23,257}};
    for (char a: {'N','T','C','n','t','c'}) for (char b: {'N','T','C'})
        for (auto& s:shapes) { fails+=one<BE,R>(h,a,b,s[0],s[1],s[2],Z(.7,-.3),Z(-.2,.4));++cases; }
    for (int dim:{255,256,257}) {
        fails+=one<BE,R>(h,'N','N',dim,dim,17,Z(1),Z(0),true); ++cases;
    }
    for (auto& s:shapes) {
        fails+=one<BE,R>(h,'C','C',s[0],s[1],s[2],Z(1),Z(0),true);
        fails+=one<BE,R>(h,'T','N',s[0],s[1],s[2],Z(0),Z(0),true);
        fails+=one<BE,R>(h,'N','C',s[0],s[1],0,Z(1),Z(.2,-.3));
        fails+=one<BE,R>(h,'N','C',s[0],s[1],0,Z(std::numeric_limits<R>::quiet_NaN(),0),Z(0),true);
        fails+=one<BE,R>(h,'N','N',0,s[1],s[2],Z(1),Z(0));
        fails+=one<BE,R>(h,'N','N',s[0],0,s[2],Z(1),Z(0)); cases+=6;
    }
    if constexpr (BE==default_backend_v) { fails+=one<BE,R>(h,'n','c',9,11,5,Z(1),Z(0),true,true);++cases; }
    // Quick return permits null matrices; invalid metadata is still rejected.
    gemm<BE,R>(h,'N','N',2,3,0,Z(1),nullptr,2,nullptr,1,Z(1),nullptr,2);
    auto reject=[&](DdlaHandle_t handle,char a,char b,int m,int n,int k,int la,int lb,int lc) {
        try { gemm<BE,R>(handle,a,b,m,n,k,Z(1),nullptr,la,nullptr,lb,Z(0),nullptr,lc); }
        catch(const std::exception&) { return 0; } return 1;
    };
    fails+=reject(nullptr,'N','N',1,1,1,1,1,1);
    fails+=reject(h,'X','N',0,0,0,1,1,1);
    fails+=reject(h,'N','X',0,0,0,1,1,1);
    fails+=reject(h,'N','N',-1,1,1,1,1,1);
    fails+=reject(h,'N','N',1,-1,1,1,1,1);
    fails+=reject(h,'N','N',1,1,-1,1,1,1);
    fails+=reject(h,'N','N',2,1,1,1,1,2);
    fails+=reject(h,'N','N',1,1,2,1,1,1);
    fails+=reject(h,'N','N',2,1,1,2,1,1);
    fails+=reject(h,'N','N',1,1,1,1,1,1);
#if DDLA_HAS_CPU && DDLA_HAS_GPU
    constexpr auto other=BE==DdlaBackend::CPU?DdlaBackend::GPU:DdlaBackend::CPU;
    bool rejected=false;
    try { gemm<other,R>(h,'N','N',0,0,0,Z(1),nullptr,1,nullptr,1,Z(0),nullptr,1); }
    catch (const std::runtime_error&) { rejected=true; }
    if (!rejected) ++fails;
#endif
    std::cout<<"mixed GEMM backend="<<int(BE)<<" real_bytes="<<sizeof(R)<<" numerical_cases="<<cases<<" failures="<<fails<<"\n";
    return fails;
}
template <DdlaBackend BE> int run() {
    DdlaHandle_t h=nullptr; check(ddlaInit(h,BE)); check(ddlaSet(h,MPI_COMM_WORLD));
    const int fails=run_type<BE,float>(h)+run_type<BE,double>(h);
    check(ddlaDestroy(h)); return fails;
}
}
int main(int argc,char**argv) {
    MPI_Init(&argc,&argv); int failures=0;
    const std::string mode=argc>1?argv[1]:"all";
    try {
#if DDLA_HAS_CPU
        if(mode!="gpu") failures+=run<DdlaBackend::CPU>();
#endif
#if DDLA_HAS_GPU
        if(mode!="cpu") failures+=run<DdlaBackend::GPU>();
#endif
    } catch(const std::exception& e) { std::cerr<<e.what()<<"\n"; ++failures; }
    std::cout<<(failures?"FAIL":"PASS")<<" test_backend_gemm_mixed\n";
    MPI_Finalize(); return failures?1:0;
}
