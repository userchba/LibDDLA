#include "mpi_test_init.h"
#include <ddla/ddla.h>
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using Wide = std::complex<long double>;
template<class T> Wide wide(T x) { return Wide(x); }
template<class R> Wide wide(std::complex<R> x) { return {x.real(),x.imag()}; }
template<class T> T value(int i) { return T((i%11-5)*0.0625); }
template<class T> T sample(int i) {
    if constexpr (std::is_same_v<T,float> || std::is_same_v<T,double>) return value<T>(i);
    else return T(value<typename T::value_type>(i),value<typename T::value_type>(i*3+1));
}
template<class T> Wide at(const std::vector<T>& a,int ld,char trans,int i,int j) {
    Wide z=wide(a[trans=='N'?i+std::size_t(j)*ld:j+std::size_t(i)*ld]);
    return trans=='C'?std::conj(z):z;
}
template<class A,class B,class Z>
void oracle(char ta,char tb,int m,int n,int k,Z alpha,const std::vector<A>& a,int lda,
    const std::vector<B>& b,int ldb,Z beta,std::vector<Z>& c,int ldc)
{
    for(int j=0;j<n;++j) for(int i=0;i<m;++i) {
        Wide sum=0;
        if(alpha!=Z(0)) for(int l=0;l<k;++l) sum+=at(a,lda,ta,i,l)*at(b,ldb,tb,l,j);
        Wide z=wide(alpha)*sum;
        if(beta!=Z(0)) z+=wide(beta)*wide(c[i+std::size_t(j)*ldc]);
        if constexpr(std::is_floating_point_v<Z>) c[i+std::size_t(j)*ldc]=Z(z.real());
        else c[i+std::size_t(j)*ldc]=Z(z.real(),z.imag());
    }
}
template<class T> class Buffer {
    ddla::DdlaHandle_t h_;
    ddla::vector<T> storage_;
public:
    Buffer(ddla::DdlaHandle_t h,const std::vector<T>& host):h_(h),storage_(h,host.size()) {
        if(!host.empty()) ddla::DDLA_CHECK(ddla::ddlaMemcpy(h_,storage_.data(),host.data(),host.size()*sizeof(T)));
        // Upload-source host vectors below may be temporary metadata tables.
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(h_));
    }
    T* data() { return storage_.data(); }
    void download(std::vector<T>& host) {
        if(!host.empty()) ddla::DDLA_CHECK(ddla::ddlaMemcpy(h_,host.data(),data(),host.size()*sizeof(T)));
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(h_));
    }
};
template<class T> void compare(const std::vector<T>& got,
    const std::vector<T>& ref) {
    using R=typename ddla::real_type<T>::type;
    const long double tol=std::is_same_v<R,float>?1e-5L:1e-12L;
    for(std::size_t i=0;i<got.size();++i)
        if(!std::isfinite(std::abs(got[i])) || std::abs(wide(got[i])-wide(ref[i]))>tol*(1+std::abs(wide(ref[i]))))
            throw std::runtime_error("mixed variable-batch oracle mismatch at element "+std::to_string(i));
}
template<class R,bool Left> void one_stage(ddla::DdlaHandle_t h,char ta,char tb,bool zero_alpha,bool zero_beta)
{
    using Z=std::complex<R>;
    using A=std::conditional_t<Left,Z,R>; using B=std::conditional_t<Left,R,Z>;
    constexpr int count=5;
    std::vector<int> m{3,7,0,5,2},n{4,2,3,0,3},k{5,0,2,4,1},la(count),lb(count),lc(count);
    std::vector<std::vector<A>> a(count); std::vector<std::vector<B>> b(count);
    std::vector<std::vector<Z>> c(count),ref(count);
    std::vector<std::unique_ptr<Buffer<A>>> da; std::vector<std::unique_ptr<Buffer<B>>> db;
    std::vector<std::unique_ptr<Buffer<Z>>> dc;
    std::vector<const A*> ap; std::vector<const B*> bp; std::vector<Z*> cp;
    const Z alpha=zero_alpha?Z(0):Z(R(.7),R(-.2));
    const Z beta=zero_beta?Z(0):Z(R(-.3),R(.1));
    for(int q=0;q<count;++q) {
        la[q]=std::max(1,ta=='N'?m[q]:k[q])+2; lb[q]=std::max(1,tb=='N'?k[q]:n[q])+1; lc[q]=std::max(1,m[q])+3;
        a[q].resize(la[q]*std::max(1,ta=='N'?k[q]:m[q]));
        b[q].resize(lb[q]*std::max(1,tb=='N'?n[q]:k[q])); c[q].resize(lc[q]*std::max(1,n[q]));
        for(std::size_t p=0;p<a[q].size();++p) a[q][p]=sample<A>(p+q);
        for(std::size_t p=0;p<b[q].size();++p) b[q][p]=sample<B>(p+q+4);
        for(std::size_t p=0;p<c[q].size();++p) c[q][p]=sample<Z>(p+q+2);
        if(zero_beta) for(int j=0;j<n[q];++j) for(int i=0;i<m[q];++i)
            c[q][i+j*lc[q]]=Z(std::numeric_limits<R>::quiet_NaN(),0);
        ref[q]=c[q]; oracle(ta,tb,m[q],n[q],k[q],alpha,a[q],la[q],b[q],lb[q],beta,ref[q],lc[q]);
        da.emplace_back(new Buffer<A>(h,a[q])); db.emplace_back(new Buffer<B>(h,b[q])); dc.emplace_back(new Buffer<Z>(h,c[q]));
        ap.push_back(zero_alpha || k[q]==0?nullptr:da.back()->data());
        bp.push_back(zero_alpha || k[q]==0?nullptr:db.back()->data()); cp.push_back(dc.back()->data());
    }
    Buffer<int> dm(h,m),dn(h,n),dk(h,k),dla(h,la),dlb(h,lb),dlc(h,lc);
    Buffer<const A*> dap(h,ap); Buffer<const B*> dbp(h,bp); Buffer<Z*> dcp(h,cp);
    ddla::gemmVbatched<R>(h,ta,tb,dm.data(),dn.data(),dk.data(),alpha,dap.data(),dla.data(),dbp.data(),dlb.data(),beta,dcp.data(),dlc.data(),count);
    for(int q=0;q<count;++q) { dc[q]->download(c[q]); compare(c[q],ref[q]); }
    // A bad LD in a late entry must be rejected before any matrix is written.
    la.back()=0; Buffer<int> invalid(h,la);
    bool rejected=false;
    try { ddla::gemmVbatched<R>(h,ta,tb,dm.data(),dn.data(),dk.data(),alpha,dap.data(),invalid.data(),dbp.data(),dlb.data(),beta,dcp.data(),dlc.data(),count); }
    catch(const std::invalid_argument&) { rejected=true; }
    if(!rejected) throw std::runtime_error("mixed variable-batch bad LD accepted");
    for(int q=0;q<count;++q) { dc[q]->download(c[q]); compare(c[q],ref[q]); }
    if(!zero_alpha) {
        ap.back()=nullptr; Buffer<const A*> invalid_pointer(h,ap); rejected=false;
        try { ddla::gemmVbatched<R>(h,ta,tb,dm.data(),dn.data(),dk.data(),alpha,invalid_pointer.data(),dla.data(),dbp.data(),dlb.data(),beta,dcp.data(),dlc.data(),count); }
        catch(const std::invalid_argument&) { rejected=true; }
        if(!rejected) throw std::runtime_error("mixed variable-batch null operand accepted");
        for(int q=0;q<count;++q) { dc[q]->download(c[q]); compare(c[q],ref[q]); }
    }
}

template<class R,int Pattern> void two_stage(ddla::DdlaHandle_t h,bool left)
{
    using Z=std::complex<R>;
    using A=std::conditional_t<Pattern==1 || Pattern==3 || Pattern==4,Z,R>;
    using B=std::conditional_t<Pattern==0 || Pattern==2 || Pattern==4,Z,R>;
    using T0=std::conditional_t<Pattern==5,R,Z>;
    using TX=std::conditional_t<Pattern==2 || Pattern==3 || Pattern==5,Z,R>;
    constexpr int count=5;
    std::vector<int> m0{2,4,2,4,0},n0{3,2,3,2,0},k0{5,3,5,3,0},outer{4,5,4,5,0};
    std::vector<int> la0(count),lb0(count),lc0(count),m1(count),n1(count),k1(count),la1(count),lb1(count),lc1(count);
    const int segments[]={2,3};
    std::vector<std::vector<A>> a(count); std::vector<std::vector<B>> b(count);
    std::vector<std::vector<TX>> ext(count); std::vector<std::vector<Z>> out(count),ref(count);
    std::vector<std::vector<T0>> tmp(count);
    std::vector<std::vector<T0>> initial_tmp(2);
    std::vector<std::unique_ptr<Buffer<A>>> da; std::vector<std::unique_ptr<Buffer<B>>> db;
    std::vector<std::unique_ptr<Buffer<TX>>> de; std::vector<std::unique_ptr<Buffer<Z>>> dc;
    std::vector<std::unique_ptr<Buffer<T0>>> dt;
    std::vector<const A*> ap; std::vector<const B*> bp; std::vector<const TX*> ep;
    std::vector<Z*> cp; std::vector<T0*> tp;
    const T0 a0=sample<T0>(17);
    const Z a1(R(.6),R(.2)),beta(R(-.2),R(.1));
    for(int q=0;q<count;++q) {
        // Stage zero is C/T: physical A is k0 x m0 and B is n0 x k0.
        la0[q]=std::max(1,k0[q])+2; lb0[q]=std::max(1,n0[q])+3; lc0[q]=std::max(1,m0[q])+4;
        m1[q]=left?n0[q]:outer[q]; n1[q]=left?outer[q]:m0[q]; k1[q]=left?m0[q]:n0[q];
        const int external_ld=std::max(1,left?k1[q]:m1[q])+2;
        la1[q]=left?lc0[q]:external_ld; lb1[q]=left?external_ld:lc0[q]; lc1[q]=std::max(1,m1[q])+3;
        a[q].resize(la0[q]*std::max(1,m0[q])); b[q].resize(lb0[q]*std::max(1,k0[q]));
        ext[q].resize(external_ld*std::max(1,left?n1[q]:k1[q]));
        out[q].resize(lc1[q]*std::max(1,n1[q])); tmp[q].resize(lc0[q]*std::max(1,n0[q]));
        for(std::size_t p=0;p<a[q].size();++p) a[q][p]=sample<A>(p+q);
        for(std::size_t p=0;p<b[q].size();++p) b[q][p]=sample<B>(p+2*q+1);
        for(std::size_t p=0;p<ext[q].size();++p) ext[q][p]=sample<TX>(p+3*q);
        for(std::size_t p=0;p<out[q].size();++p) out[q][p]=sample<Z>(p+1);
        ref[q]=out[q];
        oracle('C','T',m0[q],n0[q],k0[q],a0,a[q],la0[q],b[q],lb0[q],T0(0),tmp[q],lc0[q]);
        if(left) oracle('C','N',m1[q],n1[q],k1[q],a1,tmp[q],la1[q],ext[q],lb1[q],beta,ref[q],lc1[q]);
        else oracle('N','C',m1[q],n1[q],k1[q],a1,ext[q],la1[q],tmp[q],lb1[q],beta,ref[q],lc1[q]);
        da.emplace_back(new Buffer<A>(h,a[q])); db.emplace_back(new Buffer<B>(h,b[q])); de.emplace_back(new Buffer<TX>(h,ext[q])); dc.emplace_back(new Buffer<Z>(h,out[q]));
        if(q<2) {
            initial_tmp[q].resize(tmp[q].size());
            for(std::size_t p=0;p<initial_tmp[q].size();++p) initial_tmp[q][p]=sample<T0>(p+61);
            for(int j=0;j<n0[q];++j) for(int i=0;i<m0[q];++i)
                initial_tmp[q][i+std::size_t(j)*lc0[q]]=T0(std::numeric_limits<R>::quiet_NaN());
            dt.emplace_back(new Buffer<T0>(h,initial_tmp[q]));
        }
        ap.push_back(da.back()->data()); bp.push_back(db.back()->data()); ep.push_back(de.back()->data()); cp.push_back(dc.back()->data()); tp.push_back(dt[q%2]->data());
    }
    Buffer<int> dm0(h,m0),dn0(h,n0),dk0(h,k0),dla0(h,la0),dlb0(h,lb0),dlc0(h,lc0),dlc1(h,lc1);
    Buffer<const A*> dap(h,ap); Buffer<const B*> dbp(h,bp);
    Buffer<const TX*> dep(h,ep); Buffer<Z*> dcp(h,cp); Buffer<T0*> dtp(h,tp);
    auto run=[&](T0 beta0,const int* seg,int nseg,const std::vector<int>& test_m1,
        const std::vector<int>& test_n1,const std::vector<int>& test_k1,
        const std::vector<int>& test_la1,const std::vector<int>& test_lb1) {
        Buffer<int> dm1(h,test_m1),dn1(h,test_n1),dk1(h,test_k1),dla1(h,test_la1),dlb1(h,test_lb1);
        ddla::gemmVbatched2s<R>(h,'C','T',dm0.data(),dn0.data(),dk0.data(),a0,dap.data(),dla0.data(),dbp.data(),dlb0.data(),beta0,dtp.data(),dlc0.data(),
            left?'C':'N',left?'N':'C',dm1.data(),dn1.data(),dk1.data(),a1,dep.data(),dla1.data(),dlb1.data(),beta,dcp.data(),dlc1.data(),left,count,seg,nseg);
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(h));
    };
    auto require_unchanged=[&] {
        for(int q=0;q<count;++q) {
            auto got=out[q]; dc[q]->download(got);
            if(std::memcmp(got.data(),out[q].data(),got.size()*sizeof(Z)))
                throw std::runtime_error("invalid mixed two-stage call changed final output");
        }
        for(int q=0;q<2;++q) {
            auto got=initial_tmp[q]; dt[q]->download(got);
            if(std::memcmp(got.data(),initial_tmp[q].data(),got.size()*sizeof(T0)))
                throw std::runtime_error("invalid mixed two-stage call changed intermediate output");
        }
    };
    for(int invalid=0;invalid<5;++invalid) {
        bool rejected=false; const int bad[]={1,2};
        auto tm=m1,tn=n1,tk=k1,tla=la1,tlb=lb1;
        // Corrupt a late entry so partial execution would alter earlier C0/C1.
        // Shrinking the footprint or LD keeps mutant accesses within storage.
        if(invalid==2) { if(left) --tk[3]; else --tn[3]; }
        if(invalid==3) { if(left) --tm[3]; else --tk[3]; }
        if(invalid==4) { if(left) --tla[3]; else --tlb[3]; }
        try { run(invalid==1?T0(1):T0(0),invalid==0?bad:segments,2,tm,tn,tk,tla,tlb); }
        catch(const std::invalid_argument&) { rejected=true; }
        require_unchanged();
        if(!rejected) throw std::runtime_error("mixed two-stage invalid contract accepted, case "+std::to_string(invalid));
    }
    run(T0(0),segments,2,m1,n1,k1,la1,lb1);
    for(int q=0;q<count;++q) { dc[q]->download(out[q]); compare(out[q],ref[q]); }
}
template<class R> void suite(ddla::DdlaHandle_t h) {
    using Z=std::complex<R>;
    for(char ta: {'N','T','C'}) for(char tb: {'N','T','C'}) {
        one_stage<R,false>(h,ta,tb,false,false); one_stage<R,true>(h,ta,tb,false,false);
    }
    one_stage<R,false>(h,'C','C',true,true); one_stage<R,true>(h,'C','C',true,true);
    one_stage<R,false>(h,'N','N',false,true); one_stage<R,true>(h,'N','N',false,true);
    for(bool left: {false,true}) {
        two_stage<R,0>(h,left); two_stage<R,1>(h,left); two_stage<R,2>(h,left);
        two_stage<R,3>(h,left); two_stage<R,4>(h,left); two_stage<R,5>(h,left);
    }
    ddla::gemmVbatched<R>(h,'N','N',nullptr,nullptr,nullptr,std::complex<R>(1),
        static_cast<const R* const*>(nullptr),nullptr,static_cast<const std::complex<R>* const*>(nullptr),nullptr,
        std::complex<R>(0),nullptr,nullptr,0);
    Buffer<int> dims(h,std::vector<int>{2}),zero(h,std::vector<int>{0});
    Buffer<const R*> rp(h,std::vector<const R*>{nullptr});
    Buffer<const Z*> zp(h,std::vector<const Z*>{nullptr});
    Buffer<Z*> cp(h,std::vector<Z*>{nullptr});
    // Neither input nor output pointer entries may be dereferenced in these no-ops.
    ddla::gemmVbatched<R>(h,'N','C',dims.data(),dims.data(),zero.data(),Z(1),
        rp.data(),dims.data(),zp.data(),dims.data(),Z(1),cp.data(),dims.data(),1);
    ddla::gemmVbatched<R>(h,'C','N',dims.data(),dims.data(),dims.data(),Z(0),
        zp.data(),dims.data(),rp.data(),dims.data(),Z(1),cp.data(),dims.data(),1);
    ddla::DDLA_CHECK(ddla::ddlaSynchronize(h));
}

template<class R,bool Left> void benchmark(ddla::DdlaHandle_t h,int scale)
{
    using Z=std::complex<R>; using A=std::conditional_t<Left,Z,R>; using B=std::conditional_t<Left,R,Z>;
    constexpr int count=32,repeats=10;
    std::vector<int> m(count),n(count),k(count),la(count),lb(count),lc(count);
    std::vector<std::unique_ptr<Buffer<A>>> da; std::vector<std::unique_ptr<Buffer<B>>> db;
    std::vector<std::unique_ptr<Buffer<Z>>> promoted,dc,reference;
    std::vector<const A*> ap; std::vector<const B*> bp;
    std::vector<const Z*> zap,zbp; std::vector<Z*> cp,rp;
    std::size_t native_bytes=0,promotion_bytes=0;
    for(int q=0;q<count;++q) {
        m[q]=scale+(q%4)*3; n[q]=scale+(q%5)*2; k[q]=scale+(q%3)*7;
        la[q]=m[q]+1; lb[q]=k[q]+2; lc[q]=m[q]+3;
        std::vector<A> a(la[q]*k[q]); std::vector<B> b(lb[q]*n[q]);
        std::vector<Z> c(lc[q]*n[q]);
        for(std::size_t p=0;p<a.size();++p) a[p]=sample<A>(p+q);
        for(std::size_t p=0;p<b.size();++p) b[p]=sample<B>(p+q+5);
        da.emplace_back(new Buffer<A>(h,a)); db.emplace_back(new Buffer<B>(h,b));
        dc.emplace_back(new Buffer<Z>(h,c)); reference.emplace_back(new Buffer<Z>(h,c));
        ap.push_back(da.back()->data()); bp.push_back(db.back()->data());
        cp.push_back(dc.back()->data()); rp.push_back(reference.back()->data());
        std::vector<Z> packed(Left?b.size():a.size());
        for(std::size_t p=0;p<packed.size();++p) {
            if constexpr(Left) packed[p]=Z(b[p]); else packed[p]=Z(a[p]);
        }
        promoted.emplace_back(new Buffer<Z>(h,packed));
        if constexpr(Left) { zap.push_back(da.back()->data()); zbp.push_back(promoted.back()->data()); }
        else { zap.push_back(promoted.back()->data()); zbp.push_back(db.back()->data()); }
        native_bytes+=a.size()*sizeof(A)+b.size()*sizeof(B)+c.size()*sizeof(Z);
        promotion_bytes+=packed.size()*sizeof(Z);
    }
    Buffer<int> dm(h,m),dn(h,n),dk(h,k),dla(h,la),dlb(h,lb),dlc(h,lc);
    Buffer<const A*> dap(h,ap); Buffer<const B*> dbp(h,bp);
    Buffer<const Z*> dzap(h,zap),dzbp(h,zbp); Buffer<Z*> dcp(h,cp),drp(h,rp);
    auto native=[&] { ddla::gemmVbatched<R>(h,'N','N',dm.data(),dn.data(),dk.data(),Z(1),dap.data(),dla.data(),dbp.data(),dlb.data(),Z(0),dcp.data(),dlc.data(),count); };
    auto promoted_run=[&] { ddla::gemmVbatched<Z>(h,'N','N',dm.data(),dn.data(),dk.data(),Z(1),dzap.data(),dla.data(),dzbp.data(),dlb.data(),Z(0),drp.data(),dlc.data(),count); };
    native(); promoted_run(); ddla::DDLA_CHECK(ddla::ddlaSynchronize(h));
    auto time=[&](auto&& run) {
        const double start=MPI_Wtime();
        for(int repeat=0;repeat<repeats;++repeat) run();
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(h));
        return (MPI_Wtime()-start)/repeats;
    };
    const double native_seconds=time(native),promoted_seconds=time(promoted_run);
    for(int q=0;q<count;++q) {
        std::vector<Z> c(lc[q]*n[q]),ref(c.size());
        dc[q]->download(c); reference[q]->download(ref); compare(c,ref);
    }
    std::cout<<"mixed_vbatch_benchmark real_bytes="<<sizeof(R)<<" complex_left="<<Left
        <<" base_dimension="<<scale<<" batch="<<count<<" native_seconds="<<native_seconds
        <<" promoted_seconds="<<promoted_seconds<<" native_matrix_bytes="<<native_bytes
        <<" promoted_operand_bytes="<<promotion_bytes<<" promotion_excluded_from_timing=1\n";
}
}
int main(int argc,char** argv) {
    ddla_test_mpi_init(&argc,&argv); int failed=0; ddla::DdlaHandle_t h=nullptr;
    try {
        const bool gpu=argc>1 && std::string(argv[1])=="--gpu";
        ddla::DDLA_CHECK(ddla::ddlaInit(h,gpu?ddla::DdlaBackend::GPU:ddla::DdlaBackend::CPU));
        ddla::DDLA_CHECK(ddla::ddlaSet(h,MPI_COMM_SELF,1,1));
        suite<float>(h); suite<double>(h);
        if(gpu && argc>2 && std::string(argv[2])=="--benchmark")
            for(int scale: {24,96,256,384}) {
                benchmark<float,false>(h,scale); benchmark<float,true>(h,scale);
                benchmark<double,false>(h,scale); benchmark<double,true>(h,scale);
            }
        std::cout<<"mixed variable-batch independent oracle PASS backend="<<(gpu?"GPU":"CPU")<<'\n';
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; failed=1; }
    if(h) ddla::ddlaDestroy(h); MPI_Finalize(); return failed;
}
