#ifndef DDLA_HB2ST_HOUSEHOLDER_GROUPED_REAL_H
#define DDLA_HB2ST_HOUSEHOLDER_GROUPED_REAL_H

// Standalone experiment. The owning MPI state is deliberately absent.
#include "hb2st_householder_paired_fma.h"
#include <utility>

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
#define DDLA_HHG_X86 1
#define DDLA_HHG_INLINE __attribute__((target("avx512f,fma"),always_inline)) inline
#if defined(__GNUC__) && !defined(__clang__)
#define DDLA_HHG_ENTRY __attribute__((target("avx512f,fma"),noinline,noclone,used))
#else
#define DDLA_HHG_ENTRY __attribute__((target("avx512f,fma"),noinline,used))
#endif
#else
#define DDLA_HHG_X86 0
#define DDLA_HHG_INLINE inline
#define DDLA_HHG_ENTRY
#endif
#if defined(__GNUC__) || defined(__clang__)
#define DDLA_HHG_RESTRICT __restrict__
#else
#define DDLA_HHG_RESTRICT
#endif

namespace ddla { namespace detail { namespace hhgroup_real {
namespace strict=hhpair;
namespace fused=hhpair_fma;
template<class T> using real_t=strict::real_t<T>;
using strict::planes;
using strict::is_real;
using strict::layout_elements;
using strict::pick_width;
inline bool supported() noexcept { return fused::supported(); }
constexpr int maximum_records=32;

// The inherited strict view, records and adjacent-pair overlap are immutable.
// Only real panels with at most32 records use the additional Gram packet.
// Other panels, and all complex panels, dispatch to the strict kernel.
template<class T,int K> struct PanelView:strict::PanelView<T> {
    static_assert(K==4||K==6,"group size must be4 or6");
    const real_t<T>* gram=nullptr;
    std::size_t gram_elements=0;
};
template<class T,int K> constexpr std::size_t gram_elements(int count) noexcept {
    static_assert(K==4||K==6,"group size must be4 or6");
    if constexpr(!is_real<T>) return 0;
    return count>0&&count<=maximum_records?
        static_cast<std::size_t>((count+K-1)/K)*(K*(K-1)/2):0;
}

// Fixed group stride K*(K-1)/2, index j*(j-1)/2+i for j>i.
// Padded final-group slots are zero. Active zero-tau records never read tails.
// This CPU preparation runs once per panel, before publication; no allocation.
template<class T,int K>
bool prepare_groups(strict::PanelView<T> p,real_t<T>* gram,std::size_t capacity,
                    PanelView<T,K>& out) noexcept {
    using R=real_t<T>;
    out={};
    if(p.count<0||(p.count>0&&!strict::valid_view(p))) return false;
    const auto needed=gram_elements<T,K>(p.count);
    if(needed&&(!gram||capacity<needed)) return false;
    if constexpr(is_real<T>) {
        if(needed) {
            std::fill(gram,gram+needed,R(0));
            constexpr int stride=K*(K-1)/2;
            for(int q=0,group=0;q<p.count;q+=K,++group) {
                const int take=std::min(K,p.count-q);
                for(int j=0;j<take;++j) {
                    const T* vj=p.records+static_cast<std::size_t>(q+j)*p.stride;
                    if(!hh::finite(vj[0])||vj[0]<R(0)||vj[0]>R(2)) return false;
                    if(vj[0]==R(0)) continue;
                    const int sj=p.first_row-q-j,lj=std::min(p.b,p.n-sj);
                    for(int i=0;i<j;++i) {
                        const T* vi=p.records+static_cast<std::size_t>(q+i)*p.stride;
                        if(vi[0]==R(0)) continue;
                        const int si=p.first_row-q-i,li=std::min(p.b,p.n-si);
                        const int end=std::min(sj+lj,si+li);
                        R value=0;
                        for(int row=si;row<end;++row)
                            value+=strict::coefficient(vj,row-sj)*strict::coefficient(vi,row-si);
                        if(!hh::finite(value)) return false;
                        gram[static_cast<std::size_t>(group)*stride+j*(j-1)/2+i]=value;
                    }
                }
            }
        }
    }
    static_cast<strict::PanelView<T>&>(out)=p;
    out.gram=needed?gram:nullptr;out.gram_elements=needed;
    return true;
}

// Named fields keep the compile-time reflector accumulators independent;
// no row-dependent indexing into a vector-register array is required.
template<class B> struct Registers {
    using V=typename B::V;
    V a0,a1,a2,a3,a4,a5;
    template<int J> DDLA_HHG_INLINE V& get() noexcept {
        static_assert(J>=0&&J<6,"register index");
        if constexpr(J==0)return a0;
        else if constexpr(J==1)return a1;
        else if constexpr(J==2)return a2;
        else if constexpr(J==3)return a3;
        else if constexpr(J==4)return a4;
        else return a5;
    }
};
template<class R,int K> struct Group {
    const R* record[K];R tau[K];int offset[K],length[K],span;
};
template<int J,class R,int K>
DDLA_HHG_INLINE bool covers(const Group<R,K>& g,int row) noexcept {
    return g.tau[J]!=R(0)&&row>=g.offset[J]&&row-g.offset[J]<g.length[J];
}
template<int J,class R,int K,class B>
DDLA_HHG_INLINE void dot_row(const Group<R,K>& g,int row,typename B::V x,
                            Registers<B>& dots) noexcept {
    if(covers<J>(g,row)) dots.template get<J>()=B::fmadd(
        B::set(strict::coefficient(g.record[J],row-g.offset[J])),x,dots.template get<J>());
}
template<int J,int I,class B>
DDLA_HHG_INLINE void subtract_previous(Registers<B>& dots,
                                      const typename B::V coefficient) noexcept {
    dots.template get<J>()=B::fnmadd(coefficient,dots.template get<I>(),dots.template get<J>());
}
template<int J,class R,int K,class B,std::size_t...I>
DDLA_HHG_INLINE void solve_one(const Group<R,K>& g,const R* gram,
                               Registers<B>& dots,std::index_sequence<I...>) noexcept {
    (void)gram; // J=0 has an empty index pack and accesses no Gram entry.
    if(g.tau[J]!=R(0)) {
        // A previous identity is skipped even if its unused coupling is poisoned.
        ((g.tau[I]!=R(0)?(subtract_previous<J,int(I),B>(dots,B::set(gram[J*(J-1)/2+I])),void()):void()),...);
        dots.template get<J>()=B::mul(B::set(g.tau[J]),dots.template get<J>());
    } else dots.template get<J>()=B::zero();
}
template<int J,class R,int K,class B>
DDLA_HHG_INLINE void update_row(const Group<R,K>& g,int row,
                               Registers<B>& dots,typename B::V& x) noexcept {
    if(covers<J>(g,row)) x=B::fnmadd(B::set(strict::coefficient(g.record[J],row-g.offset[J])),
                                    dots.template get<J>(),x);
}

template<class R,int W,int K,class B,std::size_t...J>
DDLA_HHG_INLINE void apply_group(R* DDLA_HHG_RESTRICT tile,int lower,const Group<R,K>& g,
                                 const R* gram,int active,std::index_sequence<J...>) noexcept {
    using V=typename B::V;
    bool any=false;((any=any||g.tau[J]!=R(0)),...);
    if(!any)return;
    for(int c=0;c<active;c+=B::lanes) {
        const auto mask=(std::uint32_t(1)<<std::min(B::lanes,active-c))-1;
        Registers<B> dots;
        ((dots.template get<int(J)>()=B::zero()),...);
        // Every dot uses the original group input, before any tile store.
        for(int row=0;row<g.span;++row) {
            if(!(covers<int(J)>(g,row)||...))continue;
            const V x=B::load(tile+static_cast<std::size_t>(lower+row)*W+c,mask);
            (dot_row<int(J),R,K,B>(g,row,x,dots),...);
        }
        // Ordered triangular recurrence: d_j=tau_j*(r_j-sum_{i<j}G_ji*d_i).
        (solve_one<int(J),R,K,B>(g,gram,dots,std::make_index_sequence<J>{}),...);
        for(int row=0;row<g.span;++row) {
            if(!(covers<int(J)>(g,row)||...))continue;
            R* p=tile+static_cast<std::size_t>(lower+row)*W+c;
            V x=B::load(p,mask);
            (update_row<int(J),R,K,B>(g,row,dots,x),...);
            B::store(p,mask,x);
        }
    }
}

template<class R,int W,int K,class B>
DDLA_HHG_ENTRY bool apply_impl(R* DDLA_HHG_RESTRICT tile,const PanelView<R,K>& p,int active) noexcept {
    static_assert(W==8||W==16||W==32||W==64,"unsupported tile width");
    if(active<0||active>W||p.count<0)return false;
    if(active==0||p.count==0)return true;
    if(!tile||!strict::valid_view(p)||p.count>maximum_records||
       !p.gram||p.gram_elements<gram_elements<R,K>(p.count))return false;
    constexpr int stride=K*(K-1)/2;
    for(int q=0,group=0;q<p.count;q+=K,++group) {
        const int take=std::min(K,p.count-q),lower=p.first_row-q-take+1;
        Group<R,K> g{};
        for(int j=0;j<K;++j) {
            if(j<take) {
                g.record[j]=p.records+static_cast<std::size_t>(q+j)*p.stride;
                g.tau[j]=g.record[j][0];g.offset[j]=take-1-j;
                g.length[j]=std::min(p.b,p.n-(p.first_row-q-j));
                g.span=std::max(g.span,g.offset[j]+g.length[j]);
            }
        }
        apply_group<R,W,K,B>(tile,lower,g,p.gram+static_cast<std::size_t>(group)*stride,
                              active,std::make_index_sequence<K>{});
    }
    return true;
}
template<class T,int W,int K>
bool apply_panel(real_t<T>* tile,const PanelView<T,K>& p,int active=W) noexcept {
#if DDLA_HHG_X86
    if constexpr(is_real<T>) {
        if(p.count<=maximum_records&&supported())
            return apply_impl<T,W,K,fused::Avx512Ops<T>>(tile,p,active);
    }
#endif
    return strict::apply_panel<T,W>(tile,p,active);
}
template<class T,int W,int K>
bool apply_model(real_t<T>* tile,const PanelView<T,K>& p,int active=W) noexcept {
    if constexpr(is_real<T>) {
        if(p.count<=maximum_records) {
#if DDLA_HHG_X86
            if(!supported())return false;
#endif
            return apply_impl<T,W,K,fused::ModelOps<T>>(tile,p,active);
        }
    }
    return strict::apply_panel<T,W>(tile,p,active);
}
}}}
#undef DDLA_HHG_X86
#undef DDLA_HHG_INLINE
#undef DDLA_HHG_ENTRY
#undef DDLA_HHG_RESTRICT
#endif
