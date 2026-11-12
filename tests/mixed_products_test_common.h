#ifndef DDLA_MIXED_PRODUCTS_TEST_COMMON_H
#define DDLA_MIXED_PRODUCTS_TEST_COMMON_H

// Helpers shared by test_gemm_mixed_batched and test_mixed_products: the structured operands, their
// extended-precision reference product and the mixed batched GEMM case both tests run.

#include "pblas_structured_test_common.h"
#include "workspace_test_common.h"
#include "BLAS/blas_impl.h"
#include <array>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace ddla;
using structured_test::scalar;
using Wide=std::complex<long double>;
namespace {
template <typename T> Wide wide(T x) {
    if constexpr (std::is_arithmetic_v<T>) return {x,0};
    else return {x.real(),x.imag()};
}
template <typename T> T feature(int i,int axis,int salt) {
    return scalar<T>(((i*(3+salt)+axis*5)%17-8)/16.,((i*(5+salt)+axis*3)%13-6)/16.);
}
template <typename T> T general(int i,int j,int salt) {
    T x=0;for(int a=0;a<2;++a)x+=feature<T>(i,a,salt)*feature<T>(j,a,salt+1);
    return x;
}
void require(DdlaHandle_t h,bool condition,const std::string& message) {
    int bad=condition?0:1,all=0;
    MPI_Allreduce(&bad,&all,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h));
    if(all)throw std::runtime_error(message);
}
void query_pure(DdlaHandle_t h,const workspace_test::Snapshot& s) {
    require(h,!(s.ddla_alloc||s.ddla_free||s.host_new||s.pinned_alloc||s.raw_gpu_alloc||s.vendor_query||s.resource_setup||s.copies||s.launches||s.syncs||s.collectives),"mixed query performed work");
}
void run_reused(DdlaHandle_t h,const workspace_test::Snapshot& s) {
    require(h,!(s.ddla_alloc||s.ddla_free||s.host_new||s.pinned_alloc||s.raw_gpu_alloc||s.vendor_query||s.resource_setup),"mixed private run allocated or queried resources");
}
// Independent rank-two oracle. Input values are exactly representable binary
// fractions, so long-double factors describe the stored matrices directly.
template <typename T> Wide left_factor(bool structured,char trans,int i,int axis,int salt) {
    if(structured)return wide(feature<T>(i,axis,1));
    auto x=wide(feature<T>(i,axis,salt+(trans=='N'?0:1)));
    return trans=='C'?std::conj(x):x;
}
template <typename T> Wide right_factor(bool structured,char trans,int j,int axis,int salt) {
    if(structured)return std::conj(wide(feature<T>(j,axis,1)));
    auto x=wide(feature<T>(j,axis,salt+(trans=='N'?1:0)));
    return trans=='C'?std::conj(x):x;
}
template <typename TA,typename TB>
void contraction(bool a_struct,bool b_struct,char ta,char tb,int k,int sa,int sb,Wide (&dot)[2][2]) {
    for(int a=0;a<2;++a)for(int b=0;b<2;++b)for(int l=0;l<k;++l)
        dot[a][b]+=right_factor<TA>(a_struct,ta,l,a,sa)*left_factor<TB>(b_struct,tb,l,b,sb);
}
template <typename TA,typename TB>
Wide product(bool a_struct,bool b_struct,char ta,char tb,int i,int j,int sa,int sb,const Wide (&dot)[2][2]) {
    Wide out=0;for(int a=0;a<2;++a)for(int b=0;b<2;++b)
        out+=left_factor<TA>(a_struct,ta,i,a,sa)*dot[a][b]*right_factor<TB>(b_struct,tb,j,b,sb);
    return out;
}

template <DdlaBackend Backend,typename R,typename TA,typename TB>
void batch_case(DdlaHandle_t h,int m,int n,int k,int count,char ta,char tb,int mode,int& cases)
{
    using T=std::complex<R>;
    const int ar=ta=='N'?m:k,ac=ta=='N'?k:m,br=tb=='N'?k:n,bc=tb=='N'?n:k;
    const int lda=std::max(1,ar+3),ldb=std::max(1,br+5),ldc=std::max(1,m+7);
    const std::size_t as=std::size_t(lda)*(ac+2),bs=std::size_t(ldb)*(bc+2),cs=std::size_t(ldc)*(n+3);
    std::vector<TA> ha(as*count,TA(99));std::vector<TB> hb(bs*count,TB(88));std::vector<T> seed(cs*count,T(77,66));
    const T alpha=mode>=2?T(0):T(.75,.125),beta=mode==1||mode==3?T(0):mode==4?T(1):T(.25,-.125);
    std::vector<const TA*> ap(count);std::vector<const TB*> bp(count);std::vector<T*> cp(count);
    for(int q=0;q<count;++q) {
        for(int j=0;j<ac;++j)for(int i=0;i<ar;++i)ha[q*as+i+std::size_t(j)*lda]=general<TA>(i,j,1+q%3);
        for(int j=0;j<bc;++j)for(int i=0;i<br;++i)hb[q*bs+i+std::size_t(j)*ldb]=general<TB>(i,j,5+q%3);
        for(int j=0;j<n;++j)for(int i=0;i<m;++i)seed[q*cs+i+std::size_t(j)*ldc]=beta==T(0)?T(NAN,NAN):general<T>(i,j,9);
    }
    vector<TA> a(h,ha.size());vector<TB> b(h,hb.size());vector<T> c(h,seed.size());
    DDLA_CHECK(ddlaMemcpy(h,a.data(),ha.data(),ha.size()*sizeof(TA)));DDLA_CHECK(ddlaMemcpy(h,b.data(),hb.data(),hb.size()*sizeof(TB)));
    for(int q=0;q<count;++q){ap[q]=a.data()+q*as;bp[q]=b.data()+q*bs;cp[q]=c.data()+q*cs;}
    vector<const TA*> da(h,count);vector<const TB*> db(h,count);vector<T*> dc(h,count);
    DDLA_CHECK(ddlaMemcpy(h,da.data(),ap.data(),count*sizeof(TA*)));DDLA_CHECK(ddlaMemcpy(h,db.data(),bp.data(),count*sizeof(TB*)));DDLA_CHECK(ddlaMemcpy(h,dc.data(),cp.data(),count*sizeof(T*)));
    auto check=[&] {
        std::vector<T> out(seed.size());DDLA_CHECK(ddlaMemcpy(h,out.data(),c.data(),out.size()*sizeof(T)));DDLA_CHECK(ddlaSynchronize(h));
        double worst=0;bool ok=true;
        for(int q=0;q<count;++q) {
            Wide dot[2][2]={};if(alpha!=T(0)&&k)contraction<TA,TB>(false,false,ta,tb,k,1+q%3,5+q%3,dot);
            for(std::size_t pos=0;pos<cs;++pos) {
                const int i=pos%ldc,j=pos/ldc;
                if(i>=m||j>=n){if(std::memcmp(&out[q*cs+pos],&seed[q*cs+pos],sizeof(T)))ok=false;continue;}
                Wide ref=0;if(alpha!=T(0)&&k)ref=wide(alpha)*product<TA,TB>(false,false,ta,tb,i,j,1+q%3,5+q%3,dot);
                if(beta!=T(0))ref+=wide(beta)*wide(general<T>(i,j,9));
                const double error=double(std::abs(wide(out[q*cs+pos])-ref)/(1+std::abs(ref)));
                if(!std::isfinite(error))ok=false;else worst=std::max(worst,error);
            }
        }
        require(h,ok&&worst<(std::is_same_v<R,float>?3e-5:3e-12),"mixed batch result/guard failure m="+std::to_string(m)+" n="+std::to_string(n)+" k="+std::to_string(k)+" error="+std::to_string(worst));
    };
    auto reset=[&]{DDLA_CHECK(ddlaMemcpy(h,c.data(),seed.data(),seed.size()*sizeof(T)));};
    reset();gemmBatched<R>(h,ta,tb,m,n,k,alpha,alpha==T(0)?nullptr:da.data(),lda,alpha==T(0)?nullptr:db.data(),ldb,beta,dc.data(),ldc,count);check();
    std::size_t bytes=0;{workspace_test::Scope scope(true);if(alpha!=T(0))blas_impl::mixed_gemmBatched_workspace_size<R>(h,m,n,k,std::is_same_v<TA,T>,count,&bytes);query_pure(h,scope.finish());}
    vector<char> work(h,bytes+512);DDLA_CHECK(ddlaMemset(h,work.data(),0x5a,bytes+512));
    for(int rep=0;rep<2;++rep){reset();workspace_test::Scope scope;blas_impl::gemmBatched_run<Backend,T>(h,ta,tb,m,n,k,alpha,da.data(),lda,db.data(),ldb,beta,dc.data(),ldc,count,work.data()+256,bytes);run_reused(h,scope.finish());check();}
    std::array<char,256> first,last,guard;guard.fill(char(0x5a));DDLA_CHECK(ddlaMemcpy(h,first.data(),work.data(),256));DDLA_CHECK(ddlaMemcpy(h,last.data(),work.data()+256+bytes,256));DDLA_CHECK(ddlaSynchronize(h));
    require(h,first==guard&&last==guard,"mixed batch scratch guard");
    std::vector<TA> aa(ha.size());std::vector<TB> bb(hb.size());DDLA_CHECK(ddlaMemcpy(h,aa.data(),a.data(),aa.size()*sizeof(TA)));DDLA_CHECK(ddlaMemcpy(h,bb.data(),b.data(),bb.size()*sizeof(TB)));DDLA_CHECK(ddlaSynchronize(h));
    require(h,!std::memcmp(aa.data(),ha.data(),aa.size()*sizeof(TA))&&!std::memcmp(bb.data(),hb.data(),bb.size()*sizeof(TB)),"mixed batch input modified");++cases;
}
} // namespace

#endif // DDLA_MIXED_PRODUCTS_TEST_COMMON_H
