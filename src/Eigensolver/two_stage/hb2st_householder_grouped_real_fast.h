#ifndef DDLA_HB2ST_HOUSEHOLDER_GROUPED_REAL_FAST_H
#define DDLA_HB2ST_HOUSEHOLDER_GROUPED_REAL_FAST_H
#include "hb2st_householder_grouped_real.h"

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
#define DDLA_HHGF_X86 1
#define DDLA_HHGF_INLINE __attribute__((target("avx512f,fma"),always_inline)) inline
#if defined(__GNUC__) && !defined(__clang__)
#define DDLA_HHGF_ENTRY __attribute__((target("avx512f,fma"),noinline,noclone,used))
#else
#define DDLA_HHGF_ENTRY __attribute__((target("avx512f,fma"),noinline,used))
#endif
#else
#define DDLA_HHGF_X86 0
#define DDLA_HHGF_INLINE inline
#define DDLA_HHGF_ENTRY
#endif
#if defined(__GNUC__) || defined(__clang__)
#define DDLA_HHGF_RESTRICT __restrict__
#else
#define DDLA_HHGF_RESTRICT
#endif

namespace ddla { namespace detail { namespace hhgroup_real_fast {
namespace generic=hhgroup_real;
namespace strict=hhpair;
namespace fused=hhpair_fma;
using generic::real_t;using generic::is_real;using generic::planes;
using generic::PanelView;using generic::prepare_groups;using generic::gram_elements;
using generic::maximum_records;using generic::layout_elements;using generic::pick_width;
using generic::supported;

// Scalar metadata only. Requiring a full group leaves every partial group
// on the unchanged generic path. The open tail index0 is excluded from the
// interior: lo >= offset_j+1, hi <= offset_j+length_j for every j.
template<class R,int K>
inline bool interior(const generic::Group<R,K>& g,int take,int& lo,int& hi) noexcept {
    if(take!=K)return false;
    lo=0;hi=g.span;
    for(int j=0;j<K;++j) {
        if(g.tau[j]==R(0))return false;
        lo=std::max(lo,g.offset[j]+1);
        hi=std::min(hi,g.offset[j]+g.length[j]);
    }
    return lo<hi;
}

template<class R,int W,int K,class B,std::size_t...J>
DDLA_HHGF_INLINE void generic_dot(R* tile,int lower,const generic::Group<R,K>& g,int row,
                                  int c,std::uint32_t mask,generic::Registers<B>& dots,
                                  std::index_sequence<J...>) noexcept {
    if(!(generic::covers<int(J)>(g,row)||...))return;
    const auto x=B::load(tile+static_cast<std::size_t>(lower+row)*W+c,mask);
    (generic::dot_row<int(J),R,K,B>(g,row,x,dots),...);
}
template<int J,class R,int K,class B>
DDLA_HHGF_INLINE void tail_dot(const generic::Group<R,K>& g,int row,typename B::V x,
                               generic::Registers<B>& dots) noexcept {
    // Eligibility proves this is an actual tail coefficient for every J.
    dots.template get<J>()=B::fmadd(B::set(g.record[J][row-g.offset[J]]),x,dots.template get<J>());
}
template<class R,int W,int K,class B,std::size_t...J>
DDLA_HHGF_INLINE void generic_update(R* tile,int lower,const generic::Group<R,K>& g,int row,
                                     int c,std::uint32_t mask,generic::Registers<B>& dots,
                                     std::index_sequence<J...>) noexcept {
    if(!(generic::covers<int(J)>(g,row)||...))return;
    R* p=tile+static_cast<std::size_t>(lower+row)*W+c;
    auto x=B::load(p,mask);
    (generic::update_row<int(J),R,K,B>(g,row,dots,x),...);
    B::store(p,mask,x);
}
template<int J,class R,int K,class B>
DDLA_HHGF_INLINE void tail_update(const generic::Group<R,K>& g,int row,
                                  generic::Registers<B>& dots,typename B::V& x) noexcept {
    x=B::fnmadd(B::set(g.record[J][row-g.offset[J]]),dots.template get<J>(),x);
}

template<class R,int W,int K,class B,std::size_t...J>
DDLA_HHGF_INLINE void apply_group(R* DDLA_HHGF_RESTRICT tile,int lower,const generic::Group<R,K>& g,
                                 const R* gram,int active,int lo,int hi,std::index_sequence<J...> indices) noexcept {
    for(int c=0;c<active;c+=B::lanes) {
        const auto mask=(std::uint32_t(1)<<std::min(B::lanes,active-c))-1;
        generic::Registers<B> dots;
        ((dots.template get<int(J)>()=B::zero()),...);
        for(int row=0;row<lo;++row)generic_dot<R,W,K,B>(tile,lower,g,row,c,mask,dots,indices);
        for(int row=lo;row<hi;++row) {
            const auto x=B::load(tile+static_cast<std::size_t>(lower+row)*W+c,mask);
            (tail_dot<int(J),R,K,B>(g,row,x,dots),...);
        }
        for(int row=hi;row<g.span;++row)generic_dot<R,W,K,B>(tile,lower,g,row,c,mask,dots,indices);
        // Exactly the frozen recurrence and FMA order; no tile store so far.
        (generic::solve_one<int(J),R,K,B>(g,gram,dots,std::make_index_sequence<J>{}),...);
        for(int row=0;row<lo;++row)generic_update<R,W,K,B>(tile,lower,g,row,c,mask,dots,indices);
        for(int row=lo;row<hi;++row) {
            R* p=tile+static_cast<std::size_t>(lower+row)*W+c;
            auto x=B::load(p,mask);
            (tail_update<int(J),R,K,B>(g,row,dots,x),...);
            B::store(p,mask,x);
        }
        for(int row=hi;row<g.span;++row)generic_update<R,W,K,B>(tile,lower,g,row,c,mask,dots,indices);
    }
}

// Interleave two independent full double-vector chunks. Each chunk keeps
// the original row and reflector FMA order; only independent columns overlap.
template<int J,class R,int K,class B>
DDLA_HHGF_INLINE void tail_dot_pair(const generic::Group<R,K>& g,int row,
                                   typename B::V x0,typename B::V x1,
                                   generic::Registers<B>& dots0,generic::Registers<B>& dots1) noexcept {
    const auto coefficient=B::set(g.record[J][row-g.offset[J]]);
    dots0.template get<J>()=B::fmadd(coefficient,x0,dots0.template get<J>());
    dots1.template get<J>()=B::fmadd(coefficient,x1,dots1.template get<J>());
}
template<int J,class R,int K,class B>
DDLA_HHGF_INLINE void tail_update_pair(const generic::Group<R,K>& g,int row,
                                      generic::Registers<B>& dots0,generic::Registers<B>& dots1,
                                      typename B::V& x0,typename B::V& x1) noexcept {
    const auto coefficient=B::set(g.record[J][row-g.offset[J]]);
    x0=B::fnmadd(coefficient,dots0.template get<J>(),x0);
    x1=B::fnmadd(coefficient,dots1.template get<J>(),x1);
}
template<class R,int W,int K,class B,std::size_t...J>
DDLA_HHGF_INLINE void apply_group_interleaved(R* DDLA_HHGF_RESTRICT tile,int lower,
                                             const generic::Group<R,K>& g,const R* gram,
                                             int active,int lo,int hi,std::index_sequence<J...> indices) noexcept {
    static_assert(std::is_same<R,double>::value&&K==4&&B::lanes==8,"double group4 interleave only");
    constexpr std::uint32_t mask=(std::uint32_t(1)<<B::lanes)-1;
    int c=0;
    for(;c+2*B::lanes<=active;c+=2*B::lanes) {
        generic::Registers<B> dots0,dots1;
        ((dots0.template get<int(J)>()=B::zero()),...);
        ((dots1.template get<int(J)>()=B::zero()),...);
        for(int row=0;row<lo;++row) {
            generic_dot<R,W,K,B>(tile,lower,g,row,c,mask,dots0,indices);
            generic_dot<R,W,K,B>(tile,lower,g,row,c+B::lanes,mask,dots1,indices);
        }
        for(int row=lo;row<hi;++row) {
            const R* p=tile+static_cast<std::size_t>(lower+row)*W+c;
            const auto x0=B::load(p,mask),x1=B::load(p+B::lanes,mask);
            (tail_dot_pair<int(J),R,K,B>(g,row,x0,x1,dots0,dots1),...);
        }
        for(int row=hi;row<g.span;++row) {
            generic_dot<R,W,K,B>(tile,lower,g,row,c,mask,dots0,indices);
            generic_dot<R,W,K,B>(tile,lower,g,row,c+B::lanes,mask,dots1,indices);
        }
        (generic::solve_one<int(J),R,K,B>(g,gram,dots0,std::make_index_sequence<J>{}),...);
        (generic::solve_one<int(J),R,K,B>(g,gram,dots1,std::make_index_sequence<J>{}),...);
        for(int row=0;row<lo;++row) {
            generic_update<R,W,K,B>(tile,lower,g,row,c,mask,dots0,indices);
            generic_update<R,W,K,B>(tile,lower,g,row,c+B::lanes,mask,dots1,indices);
        }
        for(int row=lo;row<hi;++row) {
            R* p=tile+static_cast<std::size_t>(lower+row)*W+c;
            auto x0=B::load(p,mask),x1=B::load(p+B::lanes,mask);
            (tail_update_pair<int(J),R,K,B>(g,row,dots0,dots1,x0,x1),...);
            B::store(p,mask,x0);B::store(p+B::lanes,mask,x1);
        }
        for(int row=hi;row<g.span;++row) {
            generic_update<R,W,K,B>(tile,lower,g,row,c,mask,dots0,indices);
            generic_update<R,W,K,B>(tile,lower,g,row,c+B::lanes,mask,dots1,indices);
        }
    }
    // A remaining partial or single-vector column chunk uses the frozen path.
    if(c<active)apply_group<R,W,K,B>(tile+c,lower,g,gram,active-c,lo,hi,indices);
}

template<class R,int W,int K,class B>
DDLA_HHGF_ENTRY bool apply_impl(R* DDLA_HHGF_RESTRICT tile,const PanelView<R,K>& p,int active) noexcept {
    static_assert(W==8||W==16||W==32||W==64,"unsupported tile width");
    if(active<0||active>W||p.count<0)return false;
    if(active==0||p.count==0)return true;
    if(!tile||!strict::valid_view(p)||p.count>maximum_records||
       !p.gram||p.gram_elements<gram_elements<R,K>(p.count))return false;
    constexpr int stride=K*(K-1)/2;
    for(int q=0,group=0;q<p.count;q+=K,++group) {
        const int take=std::min(K,p.count-q),lower=p.first_row-q-take+1;
        generic::Group<R,K> g{};
        for(int j=0;j<K;++j) {
            if(j<take) {
                g.record[j]=p.records+static_cast<std::size_t>(q+j)*p.stride;
                g.tau[j]=g.record[j][0];g.offset[j]=take-1-j;
                g.length[j]=std::min(p.b,p.n-(p.first_row-q-j));
                g.span=std::max(g.span,g.offset[j]+g.length[j]);
            }
        }
        const R* gram=p.gram+static_cast<std::size_t>(group)*stride;
        int lo=0,hi=0;
        if(interior(g,take,lo,hi)) {
            if constexpr(std::is_same<R,double>::value&&K==4&&B::lanes==8) {
                bool uniform=true;
                for(int j=1;j<K;++j)uniform=uniform&&g.length[j]==g.length[0];
                if(uniform&&active>=2*B::lanes)
                    apply_group_interleaved<R,W,K,B>(tile,lower,g,gram,active,lo,hi,std::make_index_sequence<K>{});
                else apply_group<R,W,K,B>(tile,lower,g,gram,active,lo,hi,std::make_index_sequence<K>{});
            } else apply_group<R,W,K,B>(tile,lower,g,gram,active,lo,hi,std::make_index_sequence<K>{});
        } else
            generic::apply_group<R,W,K,B>(tile,lower,g,gram,active,std::make_index_sequence<K>{});
    }
    return true;
}
template<class T,int W,int K>
bool apply_panel(real_t<T>* tile,const PanelView<T,K>& p,int active=W) noexcept {
#if DDLA_HHGF_X86
    if constexpr(is_real<T>) {
        if(p.count<=maximum_records&&supported())
            return hhgroup_real_fast::apply_impl<T,W,K,fused::Avx512Ops<T>>(tile,p,active);
    }
#endif
    return strict::apply_panel<T,W>(tile,p,active);
}
template<class T,int W,int K>
bool apply_model(real_t<T>* tile,const PanelView<T,K>& p,int active=W) noexcept {
    if constexpr(is_real<T>) {
        if(p.count<=maximum_records) {
#if DDLA_HHGF_X86
            if(!supported())return false;
#endif
            return hhgroup_real_fast::apply_impl<T,W,K,fused::ModelOps<T>>(tile,p,active);
        }
    }
    return strict::apply_panel<T,W>(tile,p,active);
}
}}}
#undef DDLA_HHGF_X86
#undef DDLA_HHGF_INLINE
#undef DDLA_HHGF_ENTRY
#undef DDLA_HHGF_RESTRICT
#endif
