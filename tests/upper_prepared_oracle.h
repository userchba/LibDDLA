#ifndef DDLA_UPPER_PREPARED_ORACLE_H
#define DDLA_UPPER_PREPARED_ORACLE_H
#include <ddla/ddla.h>
#include "Eigensolver/two_stage/phe2hb_impl.h"
#include "Eigensolver/pvector_redist.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>
namespace upper_prepared_oracle {
template<class T>T adj(T x){return x;}
template<class R>std::complex<R>adj(std::complex<R>x){return std::conj(x);}
template<class T>std::complex<long double>wide(T x){return {static_cast<long double>(x),0};}
template<class R>std::complex<long double>wide(std::complex<R>x){return {x.real(),x.imag()};}
template<class T>T value(int i,int j){using R=typename ddla::real_type<T>::type;if(i<j){return adj(value<T>(j,i));}const R r=R(i==j?2+.1*i:std::sin((i+1)*.719+(j+1)*.131));if constexpr(std::is_same<T,R>::value)return r;else return T(r,R(i==j?0:std::cos((i+1)*.293-(j+1)*.817)));}
// Nonseparable dense fixture keeps raw Householder reference comparisons
// away from the null targets of the prepared operation's trigonometric data.
template<class T>T full_value(int i,int j){using R=typename ddla::real_type<T>::type;if(i<j){return adj(full_value<T>(j,i));}std::uint64_t x=std::uint64_t(i+1)*0x9e3779b97f4a7c15ULL^std::uint64_t(j+1)*0xbf58476d1ce4e5b9ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;x^=x>>31;const R re=R(i==j?3+.1*i:(double(x&65535)/65536-.5)*.8);if constexpr(std::is_same<T,R>::value)return re;else return T(re,R(i==j?0:(double((x>>32)&65535)/65536-.5)*.8));}
template<class T>T vector(int i,int j,int seed){using R=typename ddla::real_type<T>::type;const R r=R(std::sin((i+1)*.37+(j+seed)*.13));if constexpr(std::is_same<T,R>::value)return r;else return T(r,R(std::cos((i+seed)*.23-(j+1)*.71)));}
struct Result{double av=0,update=0;std::size_t changed_padding=0,poison=0;};
template<class T>Result run(const ddla::DdlaHandle_t&h,int n,int nb,int tile,int parent,int pad){
 using R=typename ddla::real_type<T>::type;using C=std::complex<long double>;
 int pr=0,pc=0,mr=0,mc=0;ddla::ddlaGetGridDims(h,pr,pc);ddla::ddlaGetGridCoords(h,mr,mc);
 int desc[ddla::DDLA_DLEN_];DDLA_CHECK(ddla::ddlaDescInit(desc,h,n+parent,n+parent,nb,nb,0,0));desc[ddla::DDLA_LLD_]+=pad;
 const int lld=desc[ddla::DDLA_LLD_],nr=ddla::num_loc(n,nb,mr,0,pr),nc=ddla::num_loc(n,nb,mc,0,pc);
 const int ar=ddla::num_loc(n+parent,nb,mr,0,pr),ac=ddla::num_loc(n+parent,nb,mc,0,pc),k=3,ldr=std::max(1,nr)+2,ldc=std::max(1,nc)+3;
 auto global=[&](int i,int p,int np){return (i/nb*np+p)*nb+i%nb;};
 std::vector<T>a(std::max<std::size_t>(1,static_cast<std::size_t>(lld)*ac),T(731));
 for(int j=0;j<ac;++j)for(int i=0;i<ar;++i){int gi=global(i,mr,pr),gj=global(j,mc,pc);if(gi<n&&gj<n)a[i+std::size_t(j)*lld]=gi/tile>gj/tile?T(std::numeric_limits<R>::quiet_NaN()):value<T>(gi,gj);}
 const auto before=a;
 std::vector<T>vr(std::size_t(ldr)*2*k),vc(std::size_t(ldc)*2*k);
 for(int c=0;c<k;++c){for(int i=0;i<nr;++i){int g=global(i,mr,pr);vr[i+std::size_t(c)*ldr]=vector<T>(g,c,3);vr[i+std::size_t(c+k)*ldr]=vector<T>(g,c,7);}for(int j=0;j<nc;++j){int g=global(j,mc,pc);vc[j+std::size_t(c)*ldc]=vector<T>(g,c,7);vc[j+std::size_t(c+k)*ldc]=vector<T>(g,c,3);}}
 std::vector<T>rp(std::size_t(ldr)*k),ry(rp.size()),cp(std::size_t(ldc)*k),cy(cp.size());
 const auto auxn=ddla::ptranspose_vectors_work_elems(h,n,k,nb);std::vector<T>aux(std::max<std::size_t>(1,auxn));
 ddla::detail::phe2hb_upper_av_cpu<T>(h,n,a.data(),desc,vr.data(),ldr,vc.data()+std::size_t(k)*ldc,ldc,k,rp.data(),ry.data(),cp.data(),cy.data(),aux.data(),auxn,tile);
 Result result;long double maxy=1,err=0;
 for(int c=0;c<k;++c)for(int i=0;i<nr;++i){int gi=global(i,mr,pr);C expect=0;for(int j=0;j<n;++j){expect+=wide(value<T>(gi,j))*wide(vector<T>(j,c,3));}maxy=std::max(maxy,std::abs(expect));const auto d=std::abs(wide(ry[i+std::size_t(c)*ldr])-expect);if(!std::isfinite(d))err=std::numeric_limits<long double>::infinity();else err=std::max(err,d);}
 const long double denom=std::max(n,1)*std::numeric_limits<R>::epsilon();result.av=static_cast<double>(err/(denom*maxy));
 ddla::detail::phe2hb_upper_update_cpu<T>(h,n,a.data(),desc,vr.data(),ldr,vc.data(),ldc,2*k,tile);
 long double maxa=1;err=0;
 for(int j=0;j<ac;++j)for(int i=0;i<lld;++i){int gi=global(i,mr,pr),gj=global(j,mc,pc);auto at=i+std::size_t(j)*lld;const bool active=i<ar&&gi<n&&gj<n&&gi/tile<=gj/tile;
  if(active){C expect=wide(value<T>(gi,gj));for(int c=0;c<k;++c){expect-=wide(vector<T>(gi,c,3))*std::conj(wide(vector<T>(gj,c,7)))+wide(vector<T>(gi,c,7))*std::conj(wide(vector<T>(gj,c,3)));}maxa=std::max(maxa,std::abs(expect));const auto d=std::abs(wide(a[at])-expect);if(!std::isfinite(d))err=std::numeric_limits<long double>::infinity();else err=std::max(err,d);}
  else{result.changed_padding+=std::memcmp(&a[at],&before[at],sizeof(T))!=0;if(i<ar&&gi<n&&gj<n&&gi/tile>gj/tile)++result.poison;}
 }
 if(ac==0)result.changed_padding+=std::memcmp(a.data(),before.data(),sizeof(T))!=0;
 result.update=static_cast<double>(err/(denom*maxa));return result;
}
}
#endif
