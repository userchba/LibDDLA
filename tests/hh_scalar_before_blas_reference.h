#ifndef DDLA_HH_SCALAR_BEFORE_BLAS_REFERENCE_H
#define DDLA_HH_SCALAR_BEFORE_BLAS_REFERENCE_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace ddla { namespace detail { namespace hh_scalar_reference {

using index_t=std::uint64_t;
template<class T> struct Real {using type=T;};
template<class R> struct Real<std::complex<R>> {using type=R;};
template<class T> using real_t=typename Real<T>::type;
template<class T> inline T adj(T x) noexcept {return x;}
template<class R> inline std::complex<R> adj(std::complex<R> x) noexcept {return std::conj(x);}
template<class T> inline T real_part(T x) noexcept {return x;}
template<class R> inline R real_part(std::complex<R> x) noexcept {return x.real();}
template<class T> inline bool finite(T x) noexcept {return std::isfinite(x);}
template<class R> inline bool finite(std::complex<R> x) noexcept {return std::isfinite(x.real())&&std::isfinite(x.imag());}
template<class T> inline real_t<T> component_scale(T x) noexcept {return std::abs(x);}
template<class R> inline R component_scale(std::complex<R> x) noexcept {return std::max(std::abs(x.real()),std::abs(x.imag()));}
template<class T> inline T divide_real(T x,real_t<T> scale) noexcept {return x/scale;}
template<class R> inline std::complex<R> divide_real(std::complex<R> x,R scale) noexcept {return {x.real()/scale,x.imag()/scale};}
template<class T> inline T unit_phase(T x) noexcept {return x<T(0)?T(-1):T(1);}
template<class R> inline std::complex<R> unit_phase(std::complex<R> x) noexcept {
    const R scale=component_scale(x);
    if(scale==R(0))return {R(1),R(0)};
    const R re=x.real()/scale,im=x.imag()/scale,rho=std::hypot(re,im);
    return {re/rho,im/rho};
}

inline bool add(index_t a,index_t b,index_t& out) noexcept {
    if(a>std::numeric_limits<index_t>::max()-b)return false;
    out=a+b;return true;
}
inline bool multiply(index_t a,index_t b,index_t& out) noexcept {
    if(b&&a>std::numeric_limits<index_t>::max()/b)return false;
    out=a*b;return true;
}
inline bool bytes(index_t count,std::size_t element_bytes,std::size_t& out) noexcept {
    if(element_bytes&&count>std::numeric_limits<std::size_t>::max()/element_bytes)return false;
    out=static_cast<std::size_t>(count)*element_bytes;return true;
}
inline int band_rows(int n,int b) noexcept {
    if(n<=0||b<0)return 0;
    return static_cast<int>(std::min<index_t>(static_cast<index_t>(n),std::max<index_t>(1,index_t(2)*static_cast<index_t>(b))));
}

struct Event {int stage=0,sweep=0,start=0,length=0,target=0,left=0,stop=0;};
inline bool event_geometry(int n,int b,int i,int j,Event& out) noexcept {
    if(n<3||b<2||i<0||i>n-3||j<0)return false;
    index_t shift=0,s=0;
    if(!multiply(static_cast<index_t>(j),static_cast<index_t>(b),shift)||
       !add(static_cast<index_t>(i)+1,shift,s)||s>=static_cast<index_t>(n-1))return false;
    const index_t e=std::min<index_t>(n,s+static_cast<index_t>(b));
    out.stage=i;out.sweep=j;out.start=static_cast<int>(s);out.length=static_cast<int>(e-s);
    out.target=j?out.start-b:i;out.left=out.target;
    out.stop=static_cast<int>(std::min<index_t>(n,e+static_cast<index_t>(b)));
    return out.length>1&&out.target>=0;
}
// For a fixed level=3*i+j the active stages form one contiguous interval.
// Their starts differ by3*b-1, so each event has disjoint matrix-entry access.
inline bool level_bounds(int n,int b,index_t level,int& lo,int& hi) noexcept {
    if(n<3||b<2)return false;
    const index_t tail=static_cast<index_t>(n-3),last=3*tail;
    if(level>last)return false;
    index_t bl=0;
    if(!multiply(static_cast<index_t>(b),level,bl))return false;
    const index_t denominator=index_t(3)*static_cast<index_t>(b)-1;
    const index_t numerator=bl>tail?bl-tail:0;
    const index_t lower=numerator/denominator+(numerator%denominator!=0);
    const index_t upper=std::min(tail,level/3);
    if(lower>upper)return false;
    lo=static_cast<int>(lower);hi=static_cast<int>(upper);return true;
}
inline int sweep_count(int n,int b) noexcept {return n>=3&&b>=2?(n-3)/b+1:0;}
inline int sweep_length(int n,int b,int sweep) noexcept {
    if(sweep<0||sweep>=sweep_count(n,b))return 0;
    return static_cast<int>(static_cast<index_t>(n-2)-static_cast<index_t>(sweep)*static_cast<index_t>(b));
}
inline int panel_count(int length,int k) noexcept {return length>0&&k>0?length/k+(length%k!=0):0;}

// F(t)=sum_{r=0}^{t-1}min(b,r+2). A packed reflector stores tau-as-T,
// then its vector tail; its leading one is implicit, so it occupies ellT's.
inline bool prefix_coeff(int b,index_t t,index_t& out) noexcept {
    if(b<2)return false;
    const index_t m=std::min(t,static_cast<index_t>(b-1));
    index_t left=0,right=0;
    index_t x=m,y=m+3;
    if((x&1)==0)x/=2;else y/=2;
    if(!multiply(x,y,left)||!multiply(t-m,static_cast<index_t>(b),right))return false;
    return add(left,right,out);
}
inline bool panel_coeff(int length,int b,int k,int panel,index_t& out) noexcept {
    if(length<=0||b<2||k<=0||panel<0||panel>=panel_count(length,k))return false;
    const index_t start=static_cast<index_t>(panel)*static_cast<index_t>(k);
    const index_t stop=std::min<index_t>(length,start+static_cast<index_t>(k));
    index_t a=0,z=0;
    if(!prefix_coeff(b,start,a)||!prefix_coeff(b,stop,z))return false;
    out=z-a;return true;
}
// Sum coefficients on owner before end_panel. Only ramp panels and a final
// partial panel differ from k*b; the partial correction is applied once.
inline bool owner_prefix(int length,int b,int k,int ranks,index_t q0,int owner,int end_panel,index_t& out) noexcept {
    if(length<=0||b<2||k<=0||ranks<=0||owner<0||owner>=ranks||end_panel<0||end_panel>panel_count(length,k))return false;
    const index_t p=static_cast<index_t>(ranks),phase=q0%p;
    const index_t first=(static_cast<index_t>(owner)+p-phase)%p;
    const index_t stop=static_cast<index_t>(end_panel);
    const index_t count=stop<=first?0:1+(stop-1-first)/p;
    index_t full=0;
    if(!multiply(static_cast<index_t>(k),static_cast<index_t>(b),full)||!multiply(count,full,out))return false;
    const index_t ramp=(static_cast<index_t>(b-1)+static_cast<index_t>(k)-1)/static_cast<index_t>(k);
    const index_t total=static_cast<index_t>(panel_count(length,k));
    for(index_t t=first;t<std::min(stop,std::min(ramp,total));t+=p) {
        index_t actual=0;
        if(!panel_coeff(length,b,k,static_cast<int>(t),actual)||actual>full||out<full-actual)return false;
        out-=full-actual;
    }
    const index_t last=total-1;
    if(last>=ramp&&last<stop&&last>=first&&(last-first)%p==0) {
        index_t actual=0;
        if(!panel_coeff(length,b,k,static_cast<int>(last),actual)||actual>full||out<full-actual)return false;
        out-=full-actual;
    }
    return true;
}
inline bool record_offset(int length,int b,int k,int ranks,index_t q0,int stage,int& owner,index_t& offset,index_t& count) noexcept {
    if(length<=0||b<2||k<=0||ranks<=0||stage<0||stage>=length)return false;
    const index_t r=static_cast<index_t>(length-1-stage),panel=r/static_cast<index_t>(k);
    index_t q=0;
    if(!add(q0,panel,q))return false;
    owner=static_cast<int>(q%static_cast<index_t>(ranks));
    index_t base=0,a=0,z=0;
    if(!owner_prefix(length,b,k,ranks,q0,owner,static_cast<int>(panel),base)||
       !prefix_coeff(b,panel*static_cast<index_t>(k),a)||!prefix_coeff(b,r,z)||
       !add(base,z-a,offset))return false;
    count=std::min<index_t>(b,r+2);return true;
}

// The caller validates the allocation/stride once before workers start.
// The event kernel's geometry proof keeps every write inside this lower band.
template<class T> struct BandView {
    T* data=nullptr;int n=0,b=0,ld=0;
    T get(int i,int j) const noexcept {
        if(i<j)return adj(get(j,i));
        return i-j<ld?data[static_cast<std::size_t>(j)*ld+i-j]:T(0);
    }
    void set(int i,int j,T x) const noexcept {
        if(i<j){set(j,i,adj(x));return;}
        data[static_cast<std::size_t>(j)*ld+i-j]=x;
    }
};

// Input band must be normalized and finite. Scratchv/av/w each has at leastb
// elements and is private to the numerical worker. No allocation or MPI occurs.
// False means numerical failure; all workers must still follow the drain path.
template<class T> bool reduce_event(BandView<T> a,const Event& e,T* v,T* av,T* w,real_t<T>& tau) noexcept {
    using R=real_t<T>;
    const int ell=e.length,s=e.start,end=s+ell;
    R scale=0;
    for(int r=0;r<ell;++r){v[r]=a.get(s+r,e.target);if(!finite(v[r]))return false;scale=std::max(scale,component_scale(v[r]));}
    if(scale==R(0)){for(int r=0;r<ell;++r)v[r]=T(0);v[0]=T(1);tau=0;return true;}
    R norm2=0;
    for(int r=0;r<ell;++r){v[r]=divide_real(v[r],scale);norm2+=real_part(adj(v[r])*v[r]);}
    const R rho=std::sqrt(norm2);
    const T alpha=-unit_phase(v[0])*rho,u0=v[0]-alpha;
    if(!finite(rho)||!finite(u0)||component_scale(u0)==R(0))return false;
    v[0]=T(1);norm2=1;
    for(int r=1;r<ell;++r){v[r]/=u0;norm2+=real_part(adj(v[r])*v[r]);}
    tau=R(2)/norm2;
    if(!finite(tau))return false;
    for(int r=0;r<ell;++r){T value=0;for(int c=0;c<ell;++c)value+=a.get(s+r,s+c)*v[c];av[r]=value;if(!finite(value))return false;}
    T dot=0;
    for(int r=0;r<ell;++r){w[r]=tau*av[r];dot+=adj(v[r])*w[r];}
    const R correction=-R(0.5)*tau*real_part(dot);
    for(int r=0;r<ell;++r)w[r]+=correction*v[r];
    for(int c=0;c<ell;++c)for(int r=c;r<ell;++r){
        T value=a.get(s+r,s+c)-v[r]*adj(w[c])-w[r]*adj(v[c]);
        if(r==c)value=T(real_part(value));
        if(!finite(value)) { return false; }
        a.set(s+r,s+c,value);
    }
    for(int c=e.left;c<s;++c){
        T value=0;for(int r=0;r<ell;++r)value+=adj(v[r])*a.get(s+r,c);
        for(int r=0;r<ell;++r){const T updated=a.get(s+r,c)-tau*v[r]*value;if(!finite(updated))return false;a.set(s+r,c,updated);}
    }
    for(int r=end;r<e.stop;++r){
        T value=0;for(int c=0;c<ell;++c)value+=a.get(r,s+c)*v[c];
        for(int c=0;c<ell;++c){const T updated=a.get(r,s+c)-tau*value*adj(v[c]);if(!finite(updated))return false;a.set(r,s+c,updated);}
    }
    for(int r=s+1;r<end;++r)a.set(r,e.target,T(0));
    return true;
}

struct Panel {int sweep=0,first_stage=0,count=0,row0=0,rows=0;index_t coeffs=0;};
inline bool panel_geometry(int n,int b,int k,int sweep,int panel,Panel& out) noexcept {
    const int length=sweep_length(n,b,sweep);
    if(k<=0||panel<0||panel>=panel_count(length,k))return false;
    const index_t first_r=static_cast<index_t>(panel)*static_cast<index_t>(k);
    out.sweep=sweep;out.first_stage=length-1-static_cast<int>(first_r);
    out.count=static_cast<int>(std::min<index_t>(k,static_cast<index_t>(length)-first_r));
    Event low,high;
    if(!event_geometry(n,b,out.first_stage-out.count+1,sweep,low)||!event_geometry(n,b,out.first_stage,sweep,high)||
       !panel_coeff(length,b,k,panel,out.coeffs))return false;
    out.row0=low.start;out.rows=high.start+high.length-low.start;return true;
}

// packed_at(stage) returns [tau-as-T,v1,...] for this sweep. Callback must be
// noexcept and return a valid whole record or nullptr. Vandtri are column-major;
// only their active rows/columns are written. work hascountT's. The conventional
// forward-product recurrence receives reversed application events: this factor
// applies H_last*...*H_first to X. No dense square factor is formed.
template<class T,class PackedAt> bool form_wy(int n,int b,int sweep,int first_stage,int count,PackedAt packed_at,
    T* V,int ldv,T* tri,int ldt,T* work,int& row0,int& rows) noexcept {
    using R=real_t<T>;
    if(count<=0||first_stage<count-1||!V||!tri||!work)return false;
    Event low,high;
    if(!event_geometry(n,b,first_stage-count+1,sweep,low)||!event_geometry(n,b,first_stage,sweep,high))return false;
    row0=low.start;rows=high.start+high.length-row0;
    if(ldv<rows||ldt<count)return false;
    for(int c=0;c<count;++c){
        for(int r=0;r<rows;++r)V[static_cast<std::size_t>(c)*ldv+r]=T(0);
        for(int r=0;r<count;++r)tri[static_cast<std::size_t>(c)*ldt+r]=T(0);
    }
    for(int col=0;col<count;++col){
        const int stage=first_stage-count+1+col;
        Event event;
        if(!event_geometry(n,b,stage,sweep,event))return false;
        const T* packed=packed_at(stage);
        if(!packed||!finite(packed[0]))return false;
        const R tau=real_part(packed[0]);
        if(tau<R(0)||tau>R(2)||packed[0]!=T(tau))return false;
        T* vc=V+static_cast<std::size_t>(col)*ldv;
        vc[event.start-row0]=T(1);
        for(int r=1;r<event.length;++r){if(!finite(packed[r]))return false;vc[event.start-row0+r]=packed[r];}
        tri[static_cast<std::size_t>(col)*ldt+col]=T(tau);
        for(int prev=0;prev<col;++prev){
            T value=0;const T* vp=V+static_cast<std::size_t>(prev)*ldv;
            for(int r=0;r<rows;++r)value+=adj(vp[r])*vc[r];
            work[prev]=value;
        }
        for(int r=0;r<col;++r){
            T value=0;
            for(int k=r;k<col;++k)value+=tri[static_cast<std::size_t>(k)*ldt+r]*work[k];
            value=-tau*value;if(!finite(value))return false;
            tri[static_cast<std::size_t>(col)*ldt+r]=value;
        }
    }
    return true;
}

}}} // namespace ddla::detail::hh
#endif
