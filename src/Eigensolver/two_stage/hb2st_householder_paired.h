#ifndef DDLA_HB2ST_HOUSEHOLDER_PAIRED_H
#define DDLA_HB2ST_HOUSEHOLDER_PAIRED_H

#include "hb2st_householder_kernels.h"
#include <algorithm>
#include <cstddef>
#include <limits>
#include <type_traits>

#ifndef DDLA_HH_PAIR_ENABLE_CLONES
#define DDLA_HH_PAIR_ENABLE_CLONES 0
#endif
#if DDLA_HH_PAIR_ENABLE_CLONES && defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
#define DDLA_HH_PAIR_CLONES __attribute__((target_clones("avx512f", "avx2", "default")))
#else
#define DDLA_HH_PAIR_CLONES
#endif
#if defined(__GNUC__) || defined(__clang__)
#define DDLA_HH_PAIR_INLINE __attribute__((always_inline)) inline
#define DDLA_HH_PAIR_RESTRICT __restrict__
#else
#define DDLA_HH_PAIR_INLINE inline
#define DDLA_HH_PAIR_RESTRICT
#endif
#if defined(__GNUC__) && !defined(__clang__)
#define DDLA_HH_PAIR_IVDEP _Pragma("GCC ivdep")
#else
#define DDLA_HH_PAIR_IVDEP
#endif

namespace ddla { namespace detail { namespace hhpair {

template<class T> using real_t=hh::real_t<T>;
template<class T> constexpr bool is_real=std::is_same<T,real_t<T>>::value;
template<class T> constexpr int planes=is_real<T>?1:2;
template<class T> inline real_t<T> imag_part(T) noexcept {return 0;}
template<class R> inline R imag_part(std::complex<R> x) noexcept {return x.imag();}

template<class T> struct PanelView {
    int n=0,b=0,first_row=0,count=0,stride=0;
    const T* records=nullptr;  // [tau-as-T,tail], fixed stride; application order
    const T* overlap=nullptr;  // u^H*v, one coefficient per adjacent pair
};
inline bool layout_elements(int n,int ncols,int width,int components,std::size_t& out) noexcept {
    if(n<0||ncols<0||width<=0||components<1||components>2)return false;
    const std::size_t tiles=static_cast<std::size_t>(ncols/width)+(ncols%width!=0);
    hh::index_t value=0;
    if(!hh::multiply(static_cast<hh::index_t>(n),static_cast<hh::index_t>(width),value)||
       !hh::multiply(value,static_cast<hh::index_t>(components),value)||
       !hh::multiply(value,static_cast<hh::index_t>(tiles),value)||
       value>std::numeric_limits<std::size_t>::max())return false;
    out=static_cast<std::size_t>(value);return true;
}
template<class T> inline int pick_width(int ncols,int threads) noexcept {
    int width=256/(int(sizeof(real_t<T>))*planes<T>);
    while(width>8&&static_cast<long long>(ncols/width)+(ncols%width!=0)<2LL*std::max(1,threads))width/=2;
    return width;
}
template<class T> inline T coefficient(const T* record,int q) noexcept {return q==0?T(1):record[q];}
template<class T> inline bool valid_view(const PanelView<T>& p) noexcept {
    return p.n>=3&&p.b>=2&&p.count>0&&p.stride>=p.b&&p.first_row>=p.count&&p.first_row<=p.n-2&&
           p.records&&(p.count<2||p.overlap);
}

// Pack only a bounded panel. Source records use tau followed by the vector
// tail; no source padding is read. Zero-tau identities do not read tails.
// Shared publication belongs to the caller, after checking this return value.
template<class T,class PackedAt>
bool prepare_panel(int n,int b,int sweep,int first_stage,int count,PackedAt packed_at,
                   T* records,int stride,T* overlap,PanelView<T>& view) noexcept {
    if(count<=0||stride<b||!records||(count>1&&!overlap))return false;
    hh::Event first,last;
    if(!hh::event_geometry(n,b,first_stage,sweep,first)||
       !hh::event_geometry(n,b,first_stage-count+1,sweep,last))return false;
    for(int q=0;q<count;++q){
        hh::Event event;
        if(!hh::event_geometry(n,b,first_stage-q,sweep,event))return false;
        const T* src=packed_at(first_stage-q);
        if(!src||!hh::finite(src[0]))return false;
        const real_t<T> tau=hh::real_part(src[0]);
        if(src[0]!=T(tau)||tau<real_t<T>(0)||tau>real_t<T>(2))return false;
        T* dst=records+static_cast<std::size_t>(q)*stride;
        dst[0]=T(tau);
        for(int r=1;r<b;++r){
            T value=0;
            if(tau!=real_t<T>(0)&&r<event.length){value=src[r];if(!hh::finite(value))return false;}
            dst[r]=value;
        }
    }
    for(int q=0;q+1<count;q+=2){
        const T* v=records+static_cast<std::size_t>(q)*stride;
        const T* u=v+stride;
        T g=0;
        if(hh::real_part(v[0])!=real_t<T>(0)&&hh::real_part(u[0])!=real_t<T>(0)){
            const int start_v=first.start-q;
            const int lv=std::min(b,n-start_v),lu=std::min(b,n-start_v+1);
            for(int r=1;r<lu&&r<=lv;++r)g+=hh::adj(u[r])*coefficient(v,r-1);
        }
        if(!hh::finite(g))return false;
        overlap[q/2]=g;
    }
    view={n,b,first.start,count,stride,records,overlap};
    return valid_view(view);
}

// One tile holds n rows, each row [real(W),imag(W)] in complex arithmetic.
// Only lanes[0,active) are accessed, including the final partial column tile.
template<class T,int W>
DDLA_HH_PAIR_INLINE void apply_single(real_t<T>* DDLA_HH_PAIR_RESTRICT x,int start,int length,
                                     const T* record,int active) noexcept {
    using R=real_t<T>;
    const R tau=hh::real_part(record[0]);
    if(tau==R(0))return;
    alignas(64) R dr[W],di[W];
    DDLA_HH_PAIR_IVDEP
    for(int c=0;c<active;++c){dr[c]=0;if constexpr(!is_real<T>)di[c]=0;}
    for(int r=0;r<length;++r){
        const T value=coefficient(record,r);
        const R vr=hh::real_part(value),vi=imag_part(value);
        const R* row=x+static_cast<std::size_t>(start+r)*W*planes<T>;
        DDLA_HH_PAIR_IVDEP
        for(int c=0;c<active;++c){
            if constexpr(is_real<T>)dr[c]+=vr*row[c];
            else{const R xr=row[c],xi=row[W+c];dr[c]+=vr*xr+vi*xi;di[c]+=vr*xi-vi*xr;}
        }
    }
    DDLA_HH_PAIR_IVDEP
    for(int c=0;c<active;++c){dr[c]*=tau;if constexpr(!is_real<T>)di[c]*=tau;}
    for(int r=0;r<length;++r){
        const T value=coefficient(record,r);
        const R vr=hh::real_part(value),vi=imag_part(value);
        R* row=x+static_cast<std::size_t>(start+r)*W*planes<T>;
        DDLA_HH_PAIR_IVDEP
        for(int c=0;c<active;++c){
            if constexpr(is_real<T>)row[c]-=vr*dr[c];
            else{row[c]-=vr*dr[c]-vi*di[c];row[W+c]-=vr*di[c]+vi*dr[c];}
        }
    }
}

template<class T,int W>
DDLA_HH_PAIR_INLINE void apply_pair(real_t<T>* DDLA_HH_PAIR_RESTRICT x,int lower,int lu,int lv,
                                   const T* u,const T* v,T g,int active) noexcept {
    using R=real_t<T>;
    const R tv=hh::real_part(v[0]),tu=hh::real_part(u[0]);
    if(tv==R(0)){apply_single<T,W>(x,lower,lu,u,active);return;}
    if(tu==R(0)){apply_single<T,W>(x,lower+1,lv,v,active);return;}
    alignas(64) R ar[W],ai[W],br[W],bi[W];
    DDLA_HH_PAIR_IVDEP
    for(int c=0;c<active;++c){ar[c]=br[c]=0;if constexpr(!is_real<T>)ai[c]=bi[c]=0;}
    const int span=std::max(lu,lv+1);
    for(int r=0;r<span;++r){
        const T cv=r>=1&&r-1<lv?coefficient(v,r-1):T(0);
        const T cu=r<lu?coefficient(u,r):T(0);
        const R vr=hh::real_part(cv),vi=imag_part(cv),ur=hh::real_part(cu),ui=imag_part(cu);
        const R* row=x+static_cast<std::size_t>(lower+r)*W*planes<T>;
        DDLA_HH_PAIR_IVDEP
        for(int c=0;c<active;++c){
            if constexpr(is_real<T>){const R value=row[c];ar[c]+=vr*value;br[c]+=ur*value;}
            else{const R xr=row[c],xi=row[W+c];ar[c]+=vr*xr+vi*xi;ai[c]+=vr*xi-vi*xr;br[c]+=ur*xr+ui*xi;bi[c]+=ur*xi-ui*xr;}
        }
    }
    const R gr=hh::real_part(g),gi=imag_part(g);
    DDLA_HH_PAIR_IVDEP
    for(int c=0;c<active;++c){
        ar[c]*=tv;
        if constexpr(is_real<T>)br[c]=tu*(br[c]-gr*ar[c]);
        else{ai[c]*=tv;const R cr=gr*ar[c]-gi*ai[c],ci=gr*ai[c]+gi*ar[c];br[c]=tu*(br[c]-cr);bi[c]=tu*(bi[c]-ci);}
    }
    for(int r=0;r<span;++r){
        const T cv=r>=1&&r-1<lv?coefficient(v,r-1):T(0);
        const T cu=r<lu?coefficient(u,r):T(0);
        const R vr=hh::real_part(cv),vi=imag_part(cv),ur=hh::real_part(cu),ui=imag_part(cu);
        R* row=x+static_cast<std::size_t>(lower+r)*W*planes<T>;
        DDLA_HH_PAIR_IVDEP
        for(int c=0;c<active;++c){
            if constexpr(is_real<T>)row[c]-=vr*ar[c]+ur*br[c];
            else{row[c]-=vr*ar[c]-vi*ai[c]+ur*br[c]-ui*bi[c];row[W+c]-=vr*ai[c]+vi*ar[c]+ur*bi[c]+ui*br[c];}
        }
    }
}

// Clones are opt-in for a separate native assembly/performance gate. The
// portable definition is the reference and does not require an x86 target.
template<class T,int W>
DDLA_HH_PAIR_CLONES bool apply_panel(real_t<T>* tile,const PanelView<T>& panel,int active=W) noexcept {
    static_assert(W==8||W==16||W==32||W==64,"unsupported Householder tile width");
    if(active<0||active>W||panel.count<0)return false;
    if(active==0||panel.count==0)return true;
    if(!tile||!valid_view(panel))return false;
    for(int q=0;q<panel.count;q+=2){
        const int start=panel.first_row-q;
        const T* v=panel.records+static_cast<std::size_t>(q)*panel.stride;
        const int lv=std::min(panel.b,panel.n-start);
        if(q+1==panel.count)apply_single<T,W>(tile,start,lv,v,active);
        else{
            const T* u=v+panel.stride;
            const int lu=std::min(panel.b,panel.n-start+1);
            apply_pair<T,W>(tile,start-1,lu,lv,u,v,panel.overlap[q/2],active);
        }
    }
    return true;
}

}}} // namespace ddla::detail::hhpair
#undef DDLA_HH_PAIR_CLONES
#undef DDLA_HH_PAIR_INLINE
#undef DDLA_HH_PAIR_RESTRICT
#undef DDLA_HH_PAIR_IVDEP
#endif
