#ifndef DDLA_TEST_GROUPED_STAGE5_ORACLE_H
#define DDLA_TEST_GROUPED_STAGE5_ORACLE_H
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "Eigensolver/eig_workspace.h"
#include "Eigensolver/two_stage/punmtr_he2hb_grouped.h"
#include "Eigensolver/pheevd_alias.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace grouped_stage5_oracle {
using Wide = std::complex<long double>;
template<class T> using Real = typename ddla::real_type<T>::type;
template<class T> T conjugate_value(T x) { return x; }
template<class R> std::complex<R> conjugate_value(std::complex<R> x) { return std::conj(x); }
template<class T> Wide wide(T x) { return Wide(x,0); }
template<class R> Wide wide(std::complex<R> x) { return Wide(x.real(),x.imag()); }
template<class T> T scalar(long double re,long double im=0) {
    if constexpr(std::is_same<T,Real<T>>::value) { (void)im; return T(re); }
    else return T(Real<T>(re),Real<T>(im));
}
inline long double hash(int i,int j,int tag) {
    std::uint64_t x=std::uint64_t(i+17)*0x9e3779b185ebca87ULL ^ std::uint64_t(j+3)*0xc2b2ae3d27d4eb4fULL ^ std::uint64_t(tag+11)*0x165667b19e3779f9ULL;
    x^=x>>33; x*=0xff51afd7ed558ccdULL; x^=x>>33;
    return static_cast<long double>(x%1000003)/1000003-.5L;
}
template<class T> T input_z(int i,int j) {
    return j%5==0 ? T(0) : scalar<T>(hash(i,j,3),hash(i,j,7));
}
template<class T> T input_a(int i,int j) {
    if(i==j) return T(2+hash(i,j,19));
    return i<j ? scalar<T>(hash(i,j,11),hash(i,j,13)) : conjugate_value(input_a<T>(j,i));
}
inline int global(int local,int block,int rank,int source,int size) {
    return (local/block*size+(rank-source+size)%size)*block+local%block;
}
struct Grid { int pr,pc,mr,mc; };
inline Grid grid(const ddla::DdlaHandle_t& h) {
    Grid g{}; ddlaGetGridDims(h,g.pr,g.pc); ddlaGetGridCoords(h,g.mr,g.mc); return g;
}
inline std::size_t elements(const ddla::DdlaHandle_t& h,const int* d) {
    auto g=grid(h); const int cols=ddla::num_loc(d[ddla::DDLA_N_],d[ddla::DDLA_NB_],g.mc,d[ddla::DDLA_CSRC_],g.pc);
    return std::max<std::size_t>(1,std::size_t(d[ddla::DDLA_LLD_])*cols);
}
template<class T,class F> std::vector<T> local_matrix(const ddla::DdlaHandle_t& h,const int* d,F value) {
    auto g=grid(h);std::vector<T>a(elements(h,d),scalar<T>(std::numeric_limits<Real<T>>::quiet_NaN()));
    const int rows=ddla::num_loc(d[ddla::DDLA_M_],d[ddla::DDLA_MB_],g.mr,d[ddla::DDLA_RSRC_],g.pr);
    const int cols=ddla::num_loc(d[ddla::DDLA_N_],d[ddla::DDLA_NB_],g.mc,d[ddla::DDLA_CSRC_],g.pc);
    for(int c=0;c<cols;++c)for(int r=0;r<rows;++r)
        a[r+std::size_t(c)*d[ddla::DDLA_LLD_]]=value(global(r,d[ddla::DDLA_MB_],g.mr,d[ddla::DDLA_RSRC_],g.pr),global(c,d[ddla::DDLA_NB_],g.mc,d[ddla::DDLA_CSRC_],g.pc));
    return a;
}
template<class T> bool same(const std::vector<T>&a,const std::vector<T>&b) {
    return a.size()==b.size()&&std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0;
}
template<class T> struct Fixture {
    int n,b;std::vector<T>a,t,tau; // Dense small fixture only, never production workspace.
};
// Actual public CPU QL/LARFT, independently generated for each source panel.
// The expected elementary reflectors use the GEQLF taus rather than T's offdiagonal entries.
template<class T> Fixture<T> ql_fixture(const ddla::DdlaHandle_t&h,int n,int b,int pattern) {
    Fixture<T> f{n,b,std::vector<T>(std::max<std::size_t>(1,std::size_t(n)*n),scalar<T>(std::numeric_limits<Real<T>>::quiet_NaN())),
        std::vector<T>(std::max<std::size_t>(1,std::size_t(n? (n-1)/b:0)*b*b),T(0)),
        std::vector<T>(std::max<std::size_t>(1,std::size_t(n? (n-1)/b:0)*b),T(0))};
    for(int step=1;step<=(n?(n-1)/b:0);++step) {
        const int start=step*b,k=std::min(b,n-start),m=start+k-b;
        std::vector<T>w(std::size_t(m)*k),tau(k),tl(std::size_t(k)*k,T(0));
        for(int c=0;c<k;++c)for(int r=0;r<m;++r)
            w[r+std::size_t(c)*m]=(pattern==1||(pattern==2&&c%2==0))?T(0):scalar<T>(hash(r,start+c,31),hash(r,start+c,43));
        ddla::geqrf<ddla::DdlaBackend::CPU,T>(h,'L',m,k,w.data(),m,tau.data());
        ddla::larft<ddla::DdlaBackend::CPU,T>(h,'B','C',m,k,w.data(),m,tau.data(),tl.data(),k);
        for(int c=0;c<k;++c) {
            const int pivot=start+c-b;
            for(int r=0;r<pivot;++r) f.a[r+std::size_t(start+c)*n]=w[r+std::size_t(c)*m];
            f.tau[(step-1)*b+c]=conjugate_value(tau[c]);
            for(int r=0;r<k;++r) f.t[std::size_t(step-1)*b*b+r+std::size_t(c)*b]=conjugate_value(tl[c+std::size_t(r)*k]);
        }
        // Padding of a partial slot must not contribute to either operation.
        for(int c=0;c<b;++c)for(int r=0;r<b;++r)if(r>=k||c>=k)
            f.t[std::size_t(step-1)*b*b+r+std::size_t(c)*b]=scalar<T>(std::numeric_limits<Real<T>>::quiet_NaN());
    }
    return f;
}
template<class T> std::vector<Wide> q_from_tau(const Fixture<T>&f) {
    const int n=f.n,b=f.b,steps=n?(n-1)/b:0;std::vector<Wide>q(std::size_t(n)*n),x(n);
    for(int col=0;col<n;++col) {
        std::fill(x.begin(),x.end(),Wide(0));x[col]=1;
        for(int step=steps;step>=1;--step) {
            const int start=step*b,k=std::min(b,n-start);
            for(int c=k-1;c>=0;--c) {
                const int pivot=start+c-b;const Wide tau=wide(f.tau[(step-1)*b+c]);
                if(tau==Wide(0))continue;
                Wide dot=x[pivot];for(int r=0;r<pivot;++r)dot+=std::conj(wide(f.a[r+std::size_t(start+c)*n]))*x[r];
                for(int r=0;r<pivot;++r)x[r]-=tau*dot*wide(f.a[r+std::size_t(start+c)*n]);
                x[pivot]-=tau*dot;
            }
        }
        for(int r=0;r<n;++r)q[r+std::size_t(col)*n]=x[r];
    }
    return q;
}
template<class T> std::vector<Wide> expected(const Fixture<T>&f,const std::vector<Wide>&q,char trans) {
    const int n=f.n;std::vector<Wide>out(std::size_t(n)*n);
    for(int c=0;c<n;++c)for(int r=0;r<n;++r)for(int l=0;l<n;++l)
        out[r+std::size_t(c)*n]+=(trans=='N'?q[r+std::size_t(l)*n]:std::conj(q[l+std::size_t(r)*n]))*wide(input_z<T>(l,c));
    return out;
}
struct Result {double dense=0,sequential=0,guards=0;std::size_t variants=0;};
template<class T> Result prepared(const ddla::DdlaHandle_t&h,const Fixture<T>&f,const T*A,const int*da,int parent,int pad,int zblock,int zsource) {
    auto g=grid(h);const int n=f.n,b=f.b;int dz[ddla::DDLA_DLEN_];
    ddla::ddlaDescInit(dz,h,n+parent,n+parent,da[ddla::DDLA_MB_],zblock,0,zsource);
    dz[ddla::DDLA_LLD_]+=pad;
    const auto input=local_matrix<T>(h,dz,[&](int i,int j){return i<n&&j<n?input_z<T>(i,j):T(17);});
    const std::vector<T> a_before(A,A+elements(h,da)),t_before=f.t;
    const auto q=q_from_tau(f);Result result;
    const int rows=ddla::num_loc(n,dz[ddla::DDLA_MB_],g.mr,0,g.pr),cols=ddla::num_loc(n,dz[ddla::DDLA_NB_],g.mc,zsource,g.pc);
    const int parent_rows=ddla::num_loc(dz[ddla::DDLA_M_],dz[ddla::DDLA_MB_],g.mr,0,g.pr),parent_cols=ddla::num_loc(dz[ddla::DDLA_N_],dz[ddla::DDLA_NB_],g.mc,zsource,g.pc);
    for(char trans:{'N','C','T'}) {
        auto old=input;int info=0;
        ddla::punmtr_he2hb<ddla::DdlaBackend::CPU,T>(h,trans,n,b,A,da,f.t.data(),old.data(),dz,info);
        if(info)throw std::runtime_error("current stage5 info");
        const auto ref=expected(f,q,trans);long double scale=1;for(auto x:ref)scale=std::max(scale,std::abs(x));
        const long double denominator=std::max(1,n)*std::numeric_limits<Real<T>>::epsilon()*scale;
        for(int cap=1;cap<=3;++cap) {
            auto z=input;ddla::detail::GroupedStage5Options o;o.group_cap=cap;
            ddla::detail::punmtr_he2hb_grouped<ddla::DdlaBackend::CPU,T>(h,trans,n,b,A,da,f.t.data(),z.data(),dz,o,info);
            if(info)throw std::runtime_error("grouped stage5 info");
            for(int c=0;c<cols;++c)for(int r=0;r<rows;++r) {
                const int gi=global(r,dz[ddla::DDLA_MB_],g.mr,0,g.pr),gj=global(c,zblock,g.mc,zsource,g.pc);
                const auto at=r+std::size_t(c)*dz[ddla::DDLA_LLD_];
                const auto d=std::abs(wide(z[at])-ref[gi+std::size_t(gj)*n])/denominator;
                const auto e=std::abs(wide(z[at])-wide(old[at]))/denominator;
                result.dense=std::isfinite(d)?std::max(result.dense,double(d)):std::numeric_limits<double>::infinity();
                result.sequential=std::isfinite(e)?std::max(result.sequential,double(e)):std::numeric_limits<double>::infinity();
            }
            for(int c=0;c<parent_cols;++c)for(int r=0;r<dz[ddla::DDLA_LLD_];++r) {
                const int gi=r<parent_rows?global(r,dz[ddla::DDLA_MB_],g.mr,0,g.pr):n+parent;
                const int gj=global(c,zblock,g.mc,zsource,g.pc);const auto at=r+std::size_t(c)*dz[ddla::DDLA_LLD_];
                if(gi>=n||gj>=n)result.guards+=std::memcmp(&z[at],&input[at],sizeof(T))!=0;
            }
            if(n<=b)result.guards+=!same(z,input);
            result.guards+=std::memcmp(A,a_before.data(),a_before.size()*sizeof(T))!=0;
            result.guards+=!same(f.t,t_before);++result.variants;
        }
    }
    return result;
}
template<class T> Result direct(const ddla::DdlaHandle_t&h,int n,int nb,int b,int pattern,int parent,int pad) {
    auto f=ql_fixture<T>(h,n,b,pattern);int da[ddla::DDLA_DLEN_];ddla::ddlaDescInit(da,h,n+parent,n+parent,nb,nb,0,0);da[ddla::DDLA_LLD_]+=pad;
    auto a=local_matrix<T>(h,da,[&](int i,int j){return i<n&&j<n?f.a[i+std::size_t(j)*n]:T(23);});
    auto g=grid(h);return prepared(h,f,a.data(),da,parent,pad,pattern%2?1:2*nb,g.pc>1?1:0);
}
// Both leading matrices share one overallocated parent but have disjoint active
// spans. In the reverse layout Z's unused parent columns cover active A too.
template<class T> Result shared_parent(const ddla::DdlaHandle_t&h,int n,int nb,int b,int pad,bool reverse) {
    auto f=ql_fixture<T>(h,n,b,0);const auto g=grid(h);int da[ddla::DDLA_DLEN_],dz[ddla::DDLA_DLEN_];
    const int zs=reverse&&g.pc>1?1:0,zb=reverse?2*nb:nb;
    ddla::ddlaDescInit(da,h,n+(pad?3:0),2*n,nb,nb,0,0);
    ddla::ddlaDescInit(dz,h,n+(reverse&&pad?5:0),reverse?2*n:n,nb,zb,0,zs);
    da[ddla::DDLA_LLD_]+=pad;dz[ddla::DDLA_LLD_]+=pad?5:0;
    const int ar=ddla::num_loc(n,nb,g.mr,0,g.pr),ac=ddla::num_loc(n,nb,g.mc,0,g.pc);
    const int zr=ddla::num_loc(n,nb,g.mr,0,g.pr),zc=ddla::num_loc(n,zb,g.mc,zs,g.pc);
    const auto span=[](int rows,int cols,int ld){return rows&&cols?std::size_t(cols-1)*ld+rows:0;};
    const auto aspan=span(ar,ac,da[ddla::DDLA_LLD_]),zspan=span(zr,zc,dz[ddla::DDLA_LLD_]);
    const std::size_t guard=7,ao=guard+(reverse?zspan:0),zo=guard+(reverse?0:aspan);
    const auto capacity=ao+elements(h,da)+zo+elements(h,dz)+aspan+zspan+2*guard;
    f.t.resize(f.t.size()+elements(h,dz)+aspan+zspan+guard,scalar<T>(53,-11));
    std::vector<T>shared(capacity,scalar<T>(71,-13));T*A=shared.data()+ao;T*Z=shared.data()+zo;
    for(int c=0;c<ac;++c)for(int r=0;r<ar;++r)A[r+std::size_t(c)*da[ddla::DDLA_LLD_]]=f.a[global(r,nb,g.mr,0,g.pr)+std::size_t(global(c,nb,g.mc,0,g.pc))*n];
    for(int c=0;c<zc;++c)for(int r=0;r<zr;++r)Z[r+std::size_t(c)*dz[ddla::DDLA_LLD_]]=input_z<T>(global(r,nb,g.mr,0,g.pr),global(c,zb,g.mc,zs,g.pc));
    const auto initial=shared,t_before=f.t;const auto q=q_from_tau(f);Result result;
    auto protected_bytes=[&] {
        double bad=0;
        for(std::size_t p=0;p<shared.size();++p){
            const auto z=p>=zo?p-zo:std::numeric_limits<std::size_t>::max();
            const bool active=p>=zo&&z/std::size_t(dz[ddla::DDLA_LLD_])<std::size_t(zc)&&z%std::size_t(dz[ddla::DDLA_LLD_])<std::size_t(zr);
            if(!active)bad+=std::memcmp(&shared[p],&initial[p],sizeof(T))!=0;
        }
        return bad;
    };
    int target=-1;
    for(int rank=0;rank<g.pr*g.pc;++rank)if(ddla::num_loc(n,nb,rank/g.pc,0,g.pr)>0&&ddla::num_loc(n,nb,rank%g.pc,0,g.pc)>0&&ddla::num_loc(n,zb,rank%g.pc,zs,g.pc)>0){target=rank;break;}
    for(char trans:{'N','C','T'}) {
        auto old=local_matrix<T>(h,dz,[&](int i,int j){return i<n&&j<n?input_z<T>(i,j):T(31);});int info=0;
        ddla::punmtr_he2hb<ddla::DdlaBackend::CPU,T>(h,trans,n,b,A,da,f.t.data(),old.data(),dz,info);
        if(info)throw std::runtime_error("shared-parent current stage5 info");
        const auto dense=expected(f,q,trans);long double scale=1;for(auto v:dense)scale=std::max(scale,std::abs(v));
        const long double denominator=std::max(1,n)*std::numeric_limits<Real<T>>::epsilon()*scale;
        for(int cap=1;cap<=3;++cap) {
            std::copy(initial.begin(),initial.end(),shared.begin());
            ddla::detail::GroupedStage5Options o;o.group_cap=cap;
            { // Actual driver composition must skip an unnecessary backup.
                ddla::detail::PheevdReflectorBackup<T>backup(h,n,A,da,Z,dz);
                result.guards+=backup.active()||backup.data()!=A;
                ddla::detail::punmtr_he2hb_grouped<ddla::DdlaBackend::CPU,T>(h,trans,n,b,backup.data(),backup.descriptor(),f.t.data(),Z,dz,o,info);
            }
            if(info)throw std::runtime_error("shared-parent grouped stage5 info");
            for(int c=0;c<zc;++c)for(int r=0;r<zr;++r){
                const auto at=r+std::size_t(c)*dz[ddla::DDLA_LLD_];const int i=global(r,nb,g.mr,0,g.pr),j=global(c,zb,g.mc,zs,g.pc);
                const auto d=std::abs(wide(Z[at])-dense[i+std::size_t(j)*n])/denominator,e=std::abs(wide(Z[at])-wide(old[at]))/denominator;
                result.dense=std::isfinite(d)?std::max(result.dense,double(d)):std::numeric_limits<double>::infinity();
                result.sequential=std::isfinite(e)?std::max(result.sequential,double(e)):std::numeric_limits<double>::infinity();
            }
            result.guards+=protected_bytes()+!same(f.t,t_before);++result.variants;
            if(n>b&&target>=0)for(int alias=0;alias<4;++alias){
                std::copy(initial.begin(),initial.end(),shared.begin());const T*ap=A;T*zp=Z;
                if(ddlaGetRank(h)==target){
                    if(alias==0)zp=A;
                    if(alias==1)zp=A+std::size_t(ac-1)*da[ddla::DDLA_LLD_];
                    if(alias==2)ap=Z+std::size_t(zc-1)*dz[ddla::DDLA_LLD_];
                    if(alias==3)zp=f.t.data();
                }
                bool caught=false;info=0;
                try{ddla::detail::grouped_stage5_check<T>(h,trans,n,b,ap,da,f.t.data(),zp,dz,o);}catch(const std::exception&){caught=true;}
                result.guards+=(caught!=(ddlaGetRank(h)==target))||!same(shared,initial)||!same(f.t,t_before);
            }
        }
        std::copy(initial.begin(),initial.end(),shared.begin());
    }
    return result;
}
#if DDLA_HAS_GPU
// Small independent host reference; only the matrices used by GPU execution live on device.
template <class T>
Result prepared_gpu(
    const ddla::DdlaHandle_t& h,
    const Fixture<T>& f,
    const T* host_a,
    const int* da,
    int parent,
    int pad,
    int zblock,
    int zsource)
{
    const auto g = grid(h);
    const int n = f.n, b = f.b;
    int dz[ddla::DDLA_DLEN_];
    ddla::ddlaDescInit(dz, h, n + parent, n + parent, da[ddla::DDLA_MB_], zblock, 0, zsource);
    dz[ddla::DDLA_LLD_] += pad;
    const auto input = local_matrix<T>(h, dz, [&](int i, int j) {
        return i < n && j < n ? input_z<T>(i, j) : T(17);
    });
    const std::vector<T> a_before(host_a, host_a + elements(h, da));
    // Disjoint views of the same device parent also exercise active-span ownership.
    ddla::vector<T> parent_data(h, a_before.size() + 3 + input.size()), factor(h, f.t.size()),
        old(h, input.size());
    T* A = parent_data.data();
    T* Z = A + a_before.size() + 3;
    DDLA_CHECK(ddlaMemcpy(h, A, a_before.data(), a_before.size() * sizeof(T)));
    DDLA_CHECK(ddlaMemcpy(h, factor.data(), f.t.data(), f.t.size() * sizeof(T)));
    const auto q = q_from_tau(f);
    Result out;
    const int rows = ddla::num_loc(n, dz[ddla::DDLA_MB_], g.mr, 0, g.pr);
    const int cols = ddla::num_loc(n, dz[ddla::DDLA_NB_], g.mc, zsource, g.pc);
    for (char trans : {'N', 'C', 'T'}) {
        DDLA_CHECK(ddlaMemcpy(h, old.data(), input.data(), input.size() * sizeof(T)));
        int info = 0;
        const auto sequential=ddla::spectral_impl::punmtr_he2hb_check(
            h,ddla::DdlaBackend::GPU,trans,n,b,A,da,factor.data(),old.data(),dz);
        std::size_t bytes=0;
        ddla::spectral_impl::punmtr_he2hb_sequential_workspace_size<T>(h,sequential,&bytes);
        ddla::vector<char> work(h,bytes);
        ddla::spectral_impl::punmtr_he2hb_sequential_run<ddla::DdlaBackend::GPU,T>(
            h,sequential,A,da,factor.data(),old.data(),dz,info,work.data(),bytes);
        std::vector<T> reference(input.size());
        DDLA_CHECK(ddlaMemcpy(h, reference.data(), old.data(), reference.size() * sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
        const auto dense = expected(f, q, trans);
        long double scale = 1;
        for (auto value : dense)
            scale = std::max(scale, std::abs(value));
        const long double denominator =
            std::max(1, n) * std::numeric_limits<Real<T>>::epsilon() * scale;
        for (int cap = 1; cap <= 3; ++cap) {
            DDLA_CHECK(ddlaMemcpy(h, Z, input.data(), input.size() * sizeof(T)));
            ddla::detail::GroupedStage5Options options;
            options.group_cap = cap;
            ddla::detail::punmtr_he2hb_grouped<ddla::DdlaBackend::GPU, T>(
                h, trans, n, b, A, da, factor.data(), Z, dz, options, info);
            std::vector<T> result(input.size());
            DDLA_CHECK(ddlaMemcpy(h, result.data(), Z, result.size() * sizeof(T)));
            DDLA_CHECK(ddlaSynchronize(h));
            out.guards += info != 0;
            for (std::size_t at = 0; at < result.size(); ++at) {
                const int r = int(at % dz[ddla::DDLA_LLD_]), c = int(at / dz[ddla::DDLA_LLD_]);
                if (r < rows && c < cols) {
                    const int i = global(r, dz[ddla::DDLA_MB_], g.mr, 0, g.pr);
                    const int j = global(c, dz[ddla::DDLA_NB_], g.mc, zsource, g.pc);
                    const auto d =
                        std::abs(wide(result[at]) - dense[i + std::size_t(j) * n]) / denominator;
                    const auto e = std::abs(wide(result[at]) - wide(reference[at])) / denominator;
                    out.dense = std::isfinite(d) ? std::max(out.dense, double(d))
                                                 : std::numeric_limits<double>::infinity();
                    out.sequential = std::isfinite(e) ? std::max(out.sequential, double(e))
                                                      : std::numeric_limits<double>::infinity();
                } else
                    out.guards += std::memcmp(&result[at], &input[at], sizeof(T)) != 0;
            }
            ++out.variants;
        }
    }
    std::vector<T> a_after(a_before.size()), t_after(f.t.size());
    DDLA_CHECK(ddlaMemcpy(h, a_after.data(), A, a_after.size() * sizeof(T)));
    DDLA_CHECK(ddlaMemcpy(h, t_after.data(), factor.data(), t_after.size() * sizeof(T)));
    DDLA_CHECK(ddlaSynchronize(h));
    out.guards += !same(a_after, a_before) || !same(t_after, f.t);
    return out;
}

template <class T>
void device_cases(
    const ddla::DdlaHandle_t& h, const ddla::DdlaHandle_t& cpu)
{
    for (auto shape : {std::array<int, 3>{0, 4, 4},
                       {1, 4, 4},
                       {2, 1, 1},
                       {9, 4, 4},
                       {17, 4, 4},
                       {22, 4, 2},
                       {41, 8, 2}})
        for (int pattern : {0, 1, 2, 3}) {
            const int n = shape[0], nb = shape[1], band = shape[2];
            const auto g = grid(h);
            const auto f = ql_fixture<T>(cpu, n, band, pattern);
            int da[ddla::DDLA_DLEN_];
            ddla::ddlaDescInit(da, h, n + 5, n + 5, nb, nb, 0, 0);
            da[ddla::DDLA_LLD_] += 3;
            auto a = local_matrix<T>(h, da, [&](int i, int j) {
                return i < n && j < n ? f.a[i + std::size_t(j) * n] : T(23);
            });
            const auto result =
                prepared_gpu(h, f, a.data(), da, 5, 3, pattern % 2 ? 1 : 2 * nb, g.pc > 1 ? 1 : 0);
            api_grid_test::require_close(h, "GPU grouped independent elementary", result.dense, 50);
            api_grid_test::require_close(h, "GPU grouped sequential", result.sequential, 50);
            api_grid_test::require_close(
                h, "GPU grouped readonly/parent/padding", result.guards, 0);
        }
}
#endif
// Validation stays local: a malformed rank rejects its own call without MPI.
// Exercise the check entry point so valid peers do not launch numerical work.
template<class T> double validation(const ddla::DdlaHandle_t& h) {
    const int n=17,b=4;
    auto f=ql_fixture<T>(h,n,b,0);
    int da[ddla::DDLA_DLEN_],dz[ddla::DDLA_DLEN_];
    ddla::ddlaDescInit(da,h,n,n,b,b,0,0);
    ddla::ddlaDescInit(dz,h,n,n,b,b,0,0);
    auto a=local_matrix<T>(h,da,[&](int i,int j){return f.a[i+std::size_t(j)*n];});
    auto z=local_matrix<T>(h,dz,[](int i,int j){return input_z<T>(i,j);});
    const auto ab=a,tb=f.t,zb=z;
    const bool target=ddlaGetRank(h)==ddlaGetSize(h)-1;
    double bad=0;
    for(int mutation=0;mutation<11;++mutation) {
        int ad[ddla::DDLA_DLEN_],zd[ddla::DDLA_DLEN_];
        std::copy_n(da,ddla::DDLA_DLEN_,ad);
        std::copy_n(dz,ddla::DDLA_DLEN_,zd);
        ddla::detail::GroupedStage5Options o;
        char trans='C';int order=n,band=b;
        const int* input_desc=ad;
        T* output=z.data();const T* factors=f.t.data();
        if(target) {
            switch(mutation) {
            case 0:o.group_cap=0;break;
            case 1:o.group_cap=4;break;
            case 2:zd[ddla::DDLA_MB_]=2;break;
            case 3:output=a.data();break;
            case 4:factors=z.data();break;
            case 5:trans='X';break;
            case 6:order=-1;break;
            case 7:ad[ddla::DDLA_RSRC_]=1;break;
            case 8:zd[ddla::DDLA_RSRC_]=1;break;
            case 9:input_desc=nullptr;break;
            case 10:band=0;break;
            }
        }
        bool caught=false;
        try {ddla::detail::grouped_stage5_check<T>(h,trans,order,band,a.data(),input_desc,factors,output,zd,o);}
        catch(const std::exception&) {caught=true;}
        bad+=caught!=target;
        bad+=!same(a,ab)||!same(z,zb)||!same(f.t,tb);
    }
    return bad;
}
} // namespace grouped_stage5_oracle
#endif
