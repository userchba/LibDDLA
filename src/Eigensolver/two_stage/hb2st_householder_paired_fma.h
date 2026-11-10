#ifndef DDLA_HB2ST_HOUSEHOLDER_PAIRED_FMA_H
#define DDLA_HB2ST_HOUSEHOLDER_PAIRED_FMA_H

#include "hb2st_householder_paired.h"
#include <cmath>
#include <cstdint>

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
#include <immintrin.h>
#define DDLA_HH_FMA_X86 1
#define DDLA_HH_FMA_INLINE __attribute__((target("avx512f,fma"),always_inline)) inline
#if defined(__GNUC__) && !defined(__clang__)
#define DDLA_HH_FMA_ENTRY __attribute__((target("avx512f,fma"),noinline,noclone,used))
#else
#define DDLA_HH_FMA_ENTRY __attribute__((target("avx512f,fma"),noinline,used))
#endif
#else
#define DDLA_HH_FMA_X86 0
#define DDLA_HH_FMA_INLINE inline
#define DDLA_HH_FMA_ENTRY
#endif
#if defined(__GNUC__) || defined(__clang__)
#define DDLA_HH_FMA_RESTRICT __restrict__
#else
#define DDLA_HH_FMA_RESTRICT
#endif

namespace ddla { namespace detail { namespace hhpair_fma {
namespace strict = hhpair;
template<class T> using real_t = strict::real_t<T>;
template<class T> using PanelView = strict::PanelView<T>;
template<class T> constexpr bool is_real = strict::is_real<T>;
template<class T> constexpr int planes = strict::planes<T>;
using strict::prepare_panel;
using strict::layout_elements;
using strict::pick_width;

enum class Strategy { Chain, Independent };

inline bool supported() noexcept {
#if DDLA_HH_FMA_X86
    static const bool available = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma");
    return available;
#else
    return false;
#endif
}

// The array backend models AVX512 lane masks and explicit fused operations.
// It is a numerical model, never the production hot fallback.
// On x86 its common implementation has the same target as the vector backend;
// apply_model therefore checks the ISA before entering it.
template<class R> struct ModelOps {
    static constexpr int lanes=64/int(sizeof(R));
    struct V { R x[lanes]; };
    static DDLA_HH_FMA_INLINE V zero() noexcept { return V{}; }
    static DDLA_HH_FMA_INLINE V set(R x) noexcept {
        V v;
        for(int i=0;i<lanes;++i) { v.x[i]=x; }
        return v;
    }
    static DDLA_HH_FMA_INLINE V load(const R* x,std::uint32_t m) noexcept {
        V v{};
        for(int i=0;i<lanes;++i) {
            if(m&(std::uint32_t(1)<<i)) { v.x[i]=x[i]; }
        }
        return v;
    }
    static DDLA_HH_FMA_INLINE void store(R* x,std::uint32_t m,const V& v) noexcept {
        for(int i=0;i<lanes;++i) {
            if(m&(std::uint32_t(1)<<i)) { x[i]=v.x[i]; }
        }
    }
    static DDLA_HH_FMA_INLINE V add(const V& a,const V& b) noexcept {
        V v;
        for(int i=0;i<lanes;++i) { v.x[i]=a.x[i]+b.x[i]; }
        return v;
    }
    static DDLA_HH_FMA_INLINE V sub(const V& a,const V& b) noexcept {
        V v;
        for(int i=0;i<lanes;++i) { v.x[i]=a.x[i]-b.x[i]; }
        return v;
    }
    static DDLA_HH_FMA_INLINE V mul(const V& a,const V& b) noexcept {
        V v;
        for(int i=0;i<lanes;++i) { v.x[i]=a.x[i]*b.x[i]; }
        return v;
    }
    static DDLA_HH_FMA_INLINE V fmadd(const V& a,const V& b,const V& c) noexcept {
        V v;
        for(int i=0;i<lanes;++i) { v.x[i]=std::fma(a.x[i],b.x[i],c.x[i]); }
        return v;
    }
    static DDLA_HH_FMA_INLINE V fnmadd(const V& a,const V& b,const V& c) noexcept {
        V v;
        for(int i=0;i<lanes;++i) { v.x[i]=std::fma(-a.x[i],b.x[i],c.x[i]); }
        return v;
    }
};

#if DDLA_HH_FMA_X86
template<class R> struct Avx512Ops;
template<> struct Avx512Ops<float> {
    using V = __m512;
    static constexpr int lanes = 16;
    static DDLA_HH_FMA_INLINE V zero() noexcept { return _mm512_setzero_ps(); }
    static DDLA_HH_FMA_INLINE V set(float x) noexcept { return _mm512_set1_ps(x); }
    static DDLA_HH_FMA_INLINE V load(const float* x, std::uint32_t m) noexcept { return _mm512_maskz_loadu_ps(static_cast<__mmask16>(m),x); }
    static DDLA_HH_FMA_INLINE void store(float* x, std::uint32_t m, V v) noexcept { _mm512_mask_storeu_ps(x,static_cast<__mmask16>(m),v); }
    static DDLA_HH_FMA_INLINE V add(V a,V b) noexcept { return _mm512_add_ps(a,b); }
    static DDLA_HH_FMA_INLINE V sub(V a,V b) noexcept { return _mm512_sub_ps(a,b); }
    static DDLA_HH_FMA_INLINE V mul(V a,V b) noexcept { return _mm512_mul_ps(a,b); }
    static DDLA_HH_FMA_INLINE V fmadd(V a,V b,V c) noexcept { return _mm512_fmadd_ps(a,b,c); }
    static DDLA_HH_FMA_INLINE V fnmadd(V a,V b,V c) noexcept { return _mm512_fnmadd_ps(a,b,c); }
};
template<> struct Avx512Ops<double> {
    using V = __m512d;
    static constexpr int lanes = 8;
    static DDLA_HH_FMA_INLINE V zero() noexcept { return _mm512_setzero_pd(); }
    static DDLA_HH_FMA_INLINE V set(double x) noexcept { return _mm512_set1_pd(x); }
    static DDLA_HH_FMA_INLINE V load(const double* x, std::uint32_t m) noexcept { return _mm512_maskz_loadu_pd(static_cast<__mmask8>(m),x); }
    static DDLA_HH_FMA_INLINE void store(double* x, std::uint32_t m, V v) noexcept { _mm512_mask_storeu_pd(x,static_cast<__mmask8>(m),v); }
    static DDLA_HH_FMA_INLINE V add(V a,V b) noexcept { return _mm512_add_pd(a,b); }
    static DDLA_HH_FMA_INLINE V sub(V a,V b) noexcept { return _mm512_sub_pd(a,b); }
    static DDLA_HH_FMA_INLINE V mul(V a,V b) noexcept { return _mm512_mul_pd(a,b); }
    static DDLA_HH_FMA_INLINE V fmadd(V a,V b,V c) noexcept { return _mm512_fmadd_pd(a,b,c); }
    static DDLA_HH_FMA_INLINE V fnmadd(V a,V b,V c) noexcept { return _mm512_fnmadd_pd(a,b,c); }
};
#endif

template<class T,int W,class B>
DDLA_HH_FMA_INLINE void real_row(const real_t<T>* x,int lower,int r,int lu,int lv,
                                const T* u,const T* v,int c,std::uint32_t mask,
                                typename B::V& a,typename B::V& b) noexcept {
    using R = real_t<T>;
    const R vr = r>=1&&r-1<lv ? strict::coefficient(v,r-1) : R(0);
    const R ur = r<lu ? strict::coefficient(u,r) : R(0);
    const auto row = B::load(x+static_cast<std::size_t>(lower+r)*W+c,mask);
    a=B::fmadd(B::set(vr),row,a);
    b=B::fmadd(B::set(ur),row,b);
}

template<class T,int W,Strategy S,class B,bool Interior=false>
DDLA_HH_FMA_INLINE void complex_row(const real_t<T>* x,int lower,int r,int lu,int lv,
                                   const T* u,const T* v,int c,std::uint32_t mask,
                                   typename B::V& ar,typename B::V& ai,
                                   typename B::V& br,typename B::V& bi,
                                   typename B::V& a2,typename B::V& a3,
                                   typename B::V& b2,typename B::V& b3) noexcept {
    const T cv=Interior ? v[r-1] : (r>=1&&r-1<lv ? strict::coefficient(v,r-1) : T(0));
    const T cu=Interior ? u[r] : (r<lu ? strict::coefficient(u,r) : T(0));
    const auto vr=B::set(hh::real_part(cv)),vi=B::set(strict::imag_part(cv));
    const auto ur=B::set(hh::real_part(cu)),ui=B::set(strict::imag_part(cu));
    const auto* row=x+static_cast<std::size_t>(lower+r)*W*2+c;
    const auto xr=B::load(row,mask),xi=B::load(row+W,mask);
    if constexpr(S==Strategy::Independent) {
        ar=B::fmadd(vr,xr,ar); ai=B::fmadd(vi,xi,ai);
        a2=B::fmadd(vr,xi,a2); a3=B::fmadd(vi,xr,a3);
        br=B::fmadd(ur,xr,br); bi=B::fmadd(ui,xi,bi);
        b2=B::fmadd(ur,xi,b2); b3=B::fmadd(ui,xr,b3);
    } else {
        ar=B::fmadd(vr,xr,ar); ar=B::fmadd(vi,xi,ar);
        ai=B::fmadd(vr,xi,ai); ai=B::fnmadd(vi,xr,ai);
        br=B::fmadd(ur,xr,br); br=B::fmadd(ui,xi,br);
        bi=B::fmadd(ur,xi,bi); bi=B::fnmadd(ui,xr,bi);
    }
}

template<class T,int W,Strategy S,class B>
DDLA_HH_FMA_INLINE void one_record(real_t<T>* x,int start,int length,const T* record,int active) noexcept {
    using R=real_t<T>;
    using V=typename B::V;
    const R tau=hh::real_part(record[0]);
    if(tau==R(0)) return;
    for(int c=0;c<active;c+=B::lanes) {
        const int lanes=std::min(B::lanes,active-c);
        const std::uint32_t mask=(std::uint32_t(1)<<lanes)-1;
        V a=B::zero(),b=B::zero(),a2=B::zero(),a3=B::zero();
        if constexpr(is_real<T> && S==Strategy::Independent) {
            int r=0;
            for(;r+3<length;r+=4) {
                a=B::fmadd(B::set(strict::coefficient(record,r)),B::load(x+static_cast<std::size_t>(start+r)*W+c,mask),a);
                b=B::fmadd(B::set(strict::coefficient(record,r+1)),B::load(x+static_cast<std::size_t>(start+r+1)*W+c,mask),b);
                a2=B::fmadd(B::set(strict::coefficient(record,r+2)),B::load(x+static_cast<std::size_t>(start+r+2)*W+c,mask),a2);
                a3=B::fmadd(B::set(strict::coefficient(record,r+3)),B::load(x+static_cast<std::size_t>(start+r+3)*W+c,mask),a3);
            }
            for(;r<length;++r) a=B::fmadd(B::set(strict::coefficient(record,r)),B::load(x+static_cast<std::size_t>(start+r)*W+c,mask),a);
            a=B::add(B::add(a,b),B::add(a2,a3));
        } else {
            for(int r=0;r<length;++r) {
                const T v=strict::coefficient(record,r);
                const V vr=B::set(hh::real_part(v));
                const auto* row=x+static_cast<std::size_t>(start+r)*W*planes<T>+c;
                const V xr=B::load(row,mask);
                if constexpr(is_real<T>) a=B::fmadd(vr,xr,a);
                else {
                    const V vi=B::set(strict::imag_part(v)),xi=B::load(row+W,mask);
                    if constexpr(S==Strategy::Independent) {
                        a=B::fmadd(vr,xr,a); b=B::fmadd(vi,xi,b);
                        a2=B::fmadd(vr,xi,a2); a3=B::fmadd(vi,xr,a3);
                    } else {
                        a=B::fmadd(vr,xr,a); a=B::fmadd(vi,xi,a);
                        b=B::fmadd(vr,xi,b); b=B::fnmadd(vi,xr,b);
                    }
                }
            }
            if constexpr(!is_real<T> && S==Strategy::Independent) {
                a=B::add(a,b); b=B::sub(a2,a3);
            }
        }
        a=B::mul(B::set(tau),a);
        if constexpr(!is_real<T>) b=B::mul(B::set(tau),b);
        for(int r=0;r<length;++r) {
            const T v=strict::coefficient(record,r);
            const V vr=B::set(hh::real_part(v));
            auto* row=x+static_cast<std::size_t>(start+r)*W*planes<T>+c;
            V xr=B::load(row,mask);
            xr=B::fnmadd(vr,a,xr);
            if constexpr(is_real<T>) B::store(row,mask,xr);
            else {
                const V vi=B::set(strict::imag_part(v));
                xr=B::fmadd(vi,b,xr);
                V xi=B::load(row+W,mask);
                xi=B::fnmadd(vr,b,xi); xi=B::fnmadd(vi,a,xi);
                B::store(row,mask,xr); B::store(row+W,mask,xi);
            }
        }
    }
}

template<class T,int W,Strategy S,class B>
DDLA_HH_FMA_INLINE void two_records(real_t<T>* x,int lower,int lu,int lv,
                                   const T* u,const T* v,T g,int active) noexcept {
    using R=real_t<T>;
    using V=typename B::V;
    const R tv=hh::real_part(v[0]),tu=hh::real_part(u[0]);
    if(tv==R(0)) { one_record<T,W,S,B>(x,lower,lu,u,active); return; }
    if(tu==R(0)) { one_record<T,W,S,B>(x,lower+1,lv,v,active); return; }
    const int span=std::max(lu,lv+1);
    for(int c=0;c<active;c+=B::lanes) {
        const int lanes=std::min(B::lanes,active-c);
        const std::uint32_t mask=(std::uint32_t(1)<<lanes)-1;
        V ar=B::zero(),ai=B::zero(),br=B::zero(),bi=B::zero();
        V a2=B::zero(),a3=B::zero(),b2=B::zero(),b3=B::zero();
        if constexpr(is_real<T>) {
            int r=0;
            if constexpr(S==Strategy::Independent) {
                for(;r+3<span;r+=4) {
                    real_row<T,W,B>(x,lower,r,lu,lv,u,v,c,mask,ar,br);
                    real_row<T,W,B>(x,lower,r+1,lu,lv,u,v,c,mask,ai,bi);
                    real_row<T,W,B>(x,lower,r+2,lu,lv,u,v,c,mask,a2,b2);
                    real_row<T,W,B>(x,lower,r+3,lu,lv,u,v,c,mask,a3,b3);
                }
            }
            for(;r<span;++r) real_row<T,W,B>(x,lower,r,lu,lv,u,v,c,mask,ar,br);
            if constexpr(S==Strategy::Independent) {
                ar=B::add(B::add(ar,ai),B::add(a2,a3));
                br=B::add(B::add(br,bi),B::add(b2,b3));
            }
        } else {
            // The two leading rows contain implicit unit entries. Between
            // those anchors and the shorter tail, both stored coefficients
            // exist, so their bounds need not be tested for every SIMD row.
            int r=0;
            for(;r<std::min(span,2);++r)
                complex_row<T,W,S,B>(x,lower,r,lu,lv,u,v,c,mask,ar,ai,br,bi,a2,a3,b2,b3);
            const int common_end=std::min(lu,lv+1);
            for(;r<common_end;++r)
                complex_row<T,W,S,B,true>(x,lower,r,lu,lv,u,v,c,mask,ar,ai,br,bi,a2,a3,b2,b3);
            for(;r<span;++r)
                complex_row<T,W,S,B>(x,lower,r,lu,lv,u,v,c,mask,ar,ai,br,bi,a2,a3,b2,b3);
            if constexpr(S==Strategy::Independent) {
                ar=B::add(ar,ai); ai=B::sub(a2,a3);
                br=B::add(br,bi); bi=B::sub(b2,b3);
            }
        }
        ar=B::mul(B::set(tv),ar);
        if constexpr(!is_real<T>) ai=B::mul(B::set(tv),ai);
        const V gr=B::set(hh::real_part(g));
        br=B::fnmadd(gr,ar,br);
        if constexpr(!is_real<T>) {
            const V gi=B::set(strict::imag_part(g));
            br=B::fmadd(gi,ai,br);
            bi=B::fnmadd(gr,ai,bi); bi=B::fnmadd(gi,ar,bi);
            bi=B::mul(B::set(tu),bi);
        }
        br=B::mul(B::set(tu),br);
        for(int r=0;r<span;++r) {
            const T cv=r>=1&&r-1<lv ? strict::coefficient(v,r-1) : T(0);
            const T cu=r<lu ? strict::coefficient(u,r) : T(0);
            const V vr=B::set(hh::real_part(cv)),ur=B::set(hh::real_part(cu));
            auto* row=x+static_cast<std::size_t>(lower+r)*W*planes<T>+c;
            V xr=B::load(row,mask);
            xr=B::fnmadd(vr,ar,xr);
            if constexpr(is_real<T>) {
                xr=B::fnmadd(ur,br,xr); B::store(row,mask,xr);
            } else {
                const V vi=B::set(strict::imag_part(cv)),ui=B::set(strict::imag_part(cu));
                xr=B::fmadd(vi,ai,xr); xr=B::fnmadd(ur,br,xr); xr=B::fmadd(ui,bi,xr);
                V xi=B::load(row+W,mask);
                xi=B::fnmadd(vr,ai,xi); xi=B::fnmadd(vi,ar,xi);
                xi=B::fnmadd(ur,bi,xi); xi=B::fnmadd(ui,br,xi);
                B::store(row,mask,xr); B::store(row+W,mask,xi);
            }
        }
    }
}

template<class T,int W,Strategy S,class B>
DDLA_HH_FMA_ENTRY bool apply_impl(real_t<T>* DDLA_HH_FMA_RESTRICT tile,const PanelView<T>& panel,int active) noexcept {
    static_assert(W==8||W==16||W==32||W==64,"unsupported Householder tile width");
    if(active<0||active>W||panel.count<0) return false;
    if(active==0||panel.count==0) return true;
    if(!tile||!strict::valid_view(panel)) return false;
    for(int q=0;q<panel.count;q+=2) {
        const int start=panel.first_row-q;
        const T* v=panel.records+static_cast<std::size_t>(q)*panel.stride;
        const int lv=std::min(panel.b,panel.n-start);
        if(q+1==panel.count) one_record<T,W,S,B>(tile,start,lv,v,active);
        else {
            const T* u=v+panel.stride;
            const int lu=std::min(panel.b,panel.n-start+1);
            two_records<T,W,S,B>(tile,start-1,lu,lv,u,v,panel.overlap[q/2],active);
        }
    }
    return true;
}

template<class T,int W,Strategy S=Strategy::Independent>
bool apply_panel(real_t<T>* tile,const PanelView<T>& panel,int active=W) noexcept {
#if DDLA_HH_FMA_X86
    if(supported()) return apply_impl<T,W,S,Avx512Ops<real_t<T>>>(tile,panel,active);
#endif
    return strict::apply_panel<T,W>(tile,panel,active);
}

template<class T,int W,Strategy S=Strategy::Independent>
bool apply_model(real_t<T>* tile,const PanelView<T>& panel,int active=W) noexcept {
#if DDLA_HH_FMA_X86
    if(!supported()) return false;
#endif
    return apply_impl<T,W,S,ModelOps<real_t<T>>>(tile,panel,active);
}

}}}
#undef DDLA_HH_FMA_X86
#undef DDLA_HH_FMA_INLINE
#undef DDLA_HH_FMA_ENTRY
#undef DDLA_HH_FMA_RESTRICT
#endif
