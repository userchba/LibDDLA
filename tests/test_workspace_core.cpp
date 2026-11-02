#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "lapack_impl.h"
#include "plan_util.h"
#include "BLAS/blas_impl.h"
#include "BLAS/vbatched_impl.h"
#include "workspace_test_common.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
using namespace ddla;
void require(bool c, const char* message) { if (!c) throw std::runtime_error(message); }
void query_clean(const workspace_test::Snapshot& s)
{
    require(s.ddla_alloc==0 && s.ddla_free==0 && s.pinned_alloc==0 && s.raw_gpu_alloc==0 && s.host_new==0,
            "workspace query allocated or released managed storage");
    require(s.copies==0 && s.launches==0 && s.syncs==0 && s.collectives==0,
            "workspace query executed work or communication");
}
void run_clean(const workspace_test::Snapshot& s)
{
    require(s.ddla_alloc==0 && s.ddla_free==0 && s.pinned_alloc==0 && s.raw_gpu_alloc==0 && s.host_new==0,
            "workspace execution allocated or released managed storage");
    require(s.vendor_query==0 && s.resource_setup==0, "workspace execution repeated a vendor query or resource setup");
}
template<class T> T value(double r, double i=0) {
    if constexpr (std::is_arithmetic_v<T>) { (void)i; return T(r); }
    else return T(r,i);
}
template<class T> double tolerance() {
    using R=typename real_type<T>::type;
    return std::is_same_v<R,float> ? 3e-5 : 3e-12;
}
template<class T> void same(const std::vector<T>& a,const std::vector<T>& b)
{
    require(a.size()==b.size(),"comparison size");
    for(std::size_t i=0;i<a.size();++i)
        require(std::abs(a[i]-b[i])<=tolerance<T>()*(1+std::abs(a[i])),"checked/unchecked result differs");
}
template<class T> std::vector<T> read(DdlaHandle_t h,const ddla::vector<T>& a)
{
    std::vector<T> out(a.size());
    DDLA_CHECK(ddlaMemcpy(h,out.data(),a.data(),out.size()*sizeof(T)));
    DDLA_CHECK(ddlaSynchronize(h)); return out;
}
template<class T> void write(DdlaHandle_t h,ddla::vector<T>& a,const std::vector<T>& in)
{ DDLA_CHECK(ddlaMemcpy(h,a.data(),in.data(),in.size()*sizeof(T))); DDLA_CHECK(ddlaSynchronize(h)); }

// Exactly sized scratch of each kind with poisoned bytes before and after it.
struct Guarded {
    DdlaHandle_t h;
    std::size_t backend_bytes,host_bytes;
    void* backend=nullptr;
    void* host=nullptr;
    std::vector<unsigned char> host_storage;
    ddla::vector<unsigned char> device;
    static constexpr std::size_t guard=256;
    Guarded(DdlaHandle_t handle,std::size_t backend_size,std::size_t host_size)
        :h(handle),backend_bytes(backend_size),host_bytes(host_size),device(handle) {
        if(host_bytes) {host_storage.assign(host_bytes+2*guard,0xA7);host=host_storage.data()+guard;}
        if(backend_bytes) {
            device.resize(backend_bytes+2*guard);
            DDLA_CHECK(ddlaMemset(h,device.data(),0xA7,device.size())); DDLA_CHECK(ddlaSynchronize(h));
            backend=device.data()+guard;
        }
    }
    void check() {
        for(std::size_t i=0;i<host_storage.size();++i)
            if(i<guard || i>=guard+host_bytes) require(host_storage[i]==0xA7,"host workspace canary");
        const auto data=read(h,device);
        for(std::size_t i=0;i<data.size();++i)
            if(i<guard || i>=guard+backend_bytes) require(data[i]==0xA7,"backend workspace canary");
    }
};

template<DdlaBackend B,class T> void lapack(DdlaHandle_t h)
{
    constexpr int m=7,n=5,lda=10;
    std::vector<T> initial(lda*m,value<T>(-31));
    for(int j=0;j<m;++j) for(int i=0;i<m;++i)
        initial[i+std::size_t(j)*lda]=value<T>(i==j ? 9+i : .015*(i+j+1), .003*(i-j));
    ddla::vector<T> a(h,initial.size()),b(h,initial.size()),tau(h,m),tauref(h,m);
    ddla::vector<int> piv(h,m),pivref(h,m),info(h,1),inforef(h,1);
    write(h,a,initial); write(h,b,initial);
    // Several shapes on one buffer: size every shape and keep the largest.
    const int shapes[2][2]={{m,n},{4,3}};
    std::size_t lu_bytes=0;
    workspace_test::Scope lu_scope(true);
    for(const auto& sh:shapes) {
        std::size_t bytes=0;lapack_impl::getrf_workspace_size<T>(h,sh[0],sh[1],a.data(),lda,&bytes);lu_bytes=std::max(lu_bytes,bytes);
    }
    query_clean(lu_scope.finish()); same(initial,read(h,a));
    Guarded luwork(h,lu_bytes,0);
    for(const auto& sh:shapes) {
        write(h,a,initial);write(h,b,initial);
        const std::vector<int> pivot_poison(m,-19);write(h,piv,pivot_poison);write(h,pivref,pivot_poison);
        getrf<B,T>(h,sh[0],sh[1],b.data(),lda,pivref.data(),inforef.data());
        workspace_test::Scope run;
        lapack_impl::getrf_run<B,T>(h,sh[0],sh[1],a.data(),lda,piv.data(),info.data(),luwork.backend,lu_bytes);
        run_clean(run.finish());
        same(read(h,b),read(h,a));require(read(h,piv)==read(h,pivref),"LU pivots");
        require(read(h,info)==read(h,inforef),"LU info");luwork.check();
    }
    const int orders[2]={m,4};
    std::size_t chol_bytes=0;
    workspace_test::Scope chol_scope(true);
    for(int order:orders) {
        std::size_t bytes=0;lapack_impl::potrf_workspace_size<T>(h,'L',order,a.data(),lda,&bytes);chol_bytes=std::max(chol_bytes,bytes);
    }
    query_clean(chol_scope.finish());Guarded cholwork(h,chol_bytes,0);
    for(int order:orders) {
        write(h,a,initial);write(h,b,initial);
        potrf<B,T>(h,'L',order,b.data(),lda,inforef.data());
        workspace_test::Scope run;
        lapack_impl::potrf_run<B,T>(h,'L',order,a.data(),lda,info.data(),cholwork.backend,chol_bytes);
        run_clean(run.finish());same(read(h,b),read(h,a));require(read(h,info)==read(h,inforef),"Cholesky info");cholwork.check();
    }
    for(char uplo:{'U','L'}) {
        write(h,a,initial);
        const std::vector<T> tau_poison(m,value<T>(27));write(h,tau,tau_poison);
        std::size_t qr_bytes=0;
        workspace_test::Scope query_scope(true);
        for(const auto& sh:shapes) {
            std::size_t bytes=0;lapack_impl::geqrf_workspace_size<T>(h,uplo,sh[0],sh[1],a.data(),lda,tau.data(),&bytes);qr_bytes=std::max(qr_bytes,bytes);
        }
        query_clean(query_scope.finish());same(initial,read(h,a));same(tau_poison,read(h,tau));Guarded work(h,qr_bytes,0);
        for(const auto& sh:shapes) {
            write(h,a,initial);write(h,b,initial);write(h,tau,tau_poison);write(h,tauref,tau_poison);
            geqrf<B,T>(h,uplo,sh[0],sh[1],b.data(),lda,tauref.data());
            workspace_test::Scope run;
            lapack_impl::geqrf_run<B,T>(h,uplo,sh[0],sh[1],a.data(),lda,tau.data(),work.backend,qr_bytes);
            run_clean(run.finish());same(read(h,b),read(h,a));same(read(h,tauref),read(h,tau));work.check();
        }
    }
    for(char direct:{'F','B'}) {
        std::size_t bytes=0;
        workspace_test::Scope scope(true);lapack_impl::larft_workspace_size<T>(h,m,3,&bytes);query_clean(scope.finish());
        Guarded work(h,bytes,0);ddla::vector<T> t(h,15),tref(h,15);
        std::vector<T> tv(15,value<T>(-9)),ts(m,value<T>(.25));ts[1]=T(0);
        write(h,a,initial);write(h,b,initial);write(h,t,tv);write(h,tref,tv);write(h,tau,ts);
        larft<B,T>(h,direct,'C',m,3,b.data(),lda,tau.data(),tref.data(),5);
        workspace_test::Scope run;
        lapack_impl::larft_run<B,T>(h,direct,m,3,a.data(),lda,tau.data(),t.data(),5,work.backend,bytes);
        run_clean(run.finish());same(read(h,tref),read(h,t));work.check();
    }
    std::size_t swap_bytes=0;
    workspace_test::Scope swaps_scope(true);lapack_impl::laswp_workspace_size(h,2,4,-2,&swap_bytes);query_clean(swaps_scope.finish());
    Guarded swapwork(h,0,swap_bytes);ddla::vector<int> ipiv(h,8);std::vector<int> ps(8,1);ps[1]=4;ps[3]=2;ps[5]=3;write(h,ipiv,ps);
    write(h,a,initial);write(h,b,initial);laswp<B,T>(h,n,b.data(),lda,2,4,ipiv.data(),-2);
    workspace_test::Scope swap_run;
    lapack_impl::laswp_run<B,T>(h,n,a.data(),lda,2,4,ipiv.data(),-2,swapwork.host,swap_bytes);
    run_clean(swap_run.finish());same(read(h,b),read(h,a));swapwork.check();
    write(h,a,initial);write(h,b,initial);
    std::size_t bottom_bytes=0;
    workspace_test::Scope bottom_scope(true);lapack_impl::potrf_bottom_right_workspace_size<T>(h,'U',m,a.data(),&bottom_bytes);query_clean(bottom_scope.finish());
    Guarded bottom(h,bottom_bytes,0);int got=0,want=0;potrf_bottom_right<B,T>('U',m,b.data(),lda,want,h);
    workspace_test::Scope bottom_run;
    lapack_impl::potrf_bottom_right_run<B,T>(h,'U',m,a.data(),lda,got,bottom.backend,bottom_bytes);
    run_clean(bottom_run.finish());same(read(h,b),read(h,a));require(got==want,"bottom-right info");bottom.check();
    std::size_t qe=1,le=1;
    workspace_test::Scope empty_scope(true);
    lapack_impl::geqrf_workspace_size<T>(h,'L',0,0,static_cast<T*>(nullptr),1,static_cast<T*>(nullptr),&qe);
    lapack_impl::getrf_workspace_size<T>(h,0,0,static_cast<T*>(nullptr),1,&le);
    query_clean(empty_scope.finish());require(qe==0 && le==0,"empty local query");
}

template<DdlaBackend B,class T> void blas(DdlaHandle_t h)
{
    constexpr int n=9;
    std::vector<T> initial(n*n),other(n*n),out(n*n,value<T>(.3));
    for(int j=0;j<n;++j) for(int i=0;i<n;++i) {
        initial[i+j*n]=value<T>(.01*(1+i+j),.003*(i-j));other[i+j*n]=value<T>(.02*(1+2*i+j),.002*(i+j));
    }
    ddla::vector<T> a(h,n*n),b(h,n*n),c(h,n*n),ref(h,n*n);write(h,a,initial);write(h,b,other);write(h,c,out);write(h,ref,out);
    gemm<B,T>(h,'N','C',n,n,n,T(1),a.data(),n,b.data(),n,T(0),ref.data(),n);
    workspace_test::Scope run;blas_impl::gemm_run<B,T>(h,'N','C',n,n,n,T(1),a.data(),n,b.data(),n,T(0),c.data(),n);run_clean(run.finish());same(read(h,ref),read(h,c));
    std::size_t bytes=0;
    workspace_test::Scope tq(true);blas_impl::trtrm_workspace_size<T>(h,n,&bytes);query_clean(tq.finish());Guarded work(h,bytes,0);
    for(int repeat=0;repeat<2;++repeat) {
        write(h,c,out);write(h,ref,out);
        trtrm<B,T>(h,'U','N','L','C',n,T(1),a.data(),n,b.data(),n,T(.5),ref.data(),n);
        workspace_test::Scope trun;blas_impl::trtrm_run<B,T>(h,'U','N','L','C',n,T(1),a.data(),n,b.data(),n,T(.5),c.data(),n,work.backend,bytes);run_clean(trun.finish());
        same(read(h,ref),read(h,c));work.check();
    }
}

template<DdlaBackend B,class R> void mixed(DdlaHandle_t h)
{
    using C=std::complex<R>;constexpr int m=65,n=263,k=33;
    std::vector<R> ar(m*k,R(.125)),br(k*n,R(.25));
    std::vector<C> ac(m*k,C(.125,.03)),bc(k*n,C(.25,-.02)),out(m*n,C(.3,.4));
    ddla::vector<R> dra(h,ar.size()),drb(h,br.size());ddla::vector<C> dca(h,ac.size()),dcb(h,bc.size()),c(h,out.size()),ref(h,out.size());
    write(h,dra,ar);write(h,drb,br);write(h,dca,ac);write(h,dcb,bc);
    for(bool cr:{false,true}) {
        std::size_t bytes=0;
        workspace_test::Scope qs(true);blas_impl::mixed_gemm_workspace_size<R>(h,m,n,k,cr,&bytes);query_clean(qs.finish());Guarded work(h,bytes,0);
        for(int repeat=0;repeat<2;++repeat) {
            write(h,c,out);write(h,ref,out);
            if(cr) gemm<B,R>(h,'N','N',m,n,k,C(.7,.1),dca.data(),m,drb.data(),k,C(.2,-.1),ref.data(),m);
            else gemm<B,R>(h,'N','N',m,n,k,C(.7,.1),dra.data(),m,dcb.data(),k,C(.2,-.1),ref.data(),m);
            if constexpr(B==DdlaBackend::CPU) {
                // The checked convenience path retains its historical warm
                // caller-thread scratch cache, outside the numerical components.
                workspace_test::Scope cached;
                if(cr)gemm<B,R>(h,'N','N',m,n,k,C(.7,.1),dca.data(),m,drb.data(),k,C(.2,-.1),c.data(),m);
                else gemm<B,R>(h,'N','N',m,n,k,C(.7,.1),dra.data(),m,dcb.data(),k,C(.2,-.1),c.data(),m);
                run_clean(cached.finish());same(read(h,ref),read(h,c));write(h,c,out);
            }
            workspace_test::Scope run;
            if(cr) blas_impl::gemm_run<B,R>(h,'N','N',m,n,k,C(.7,.1),dca.data(),m,drb.data(),k,C(.2,-.1),c.data(),m,work.backend,bytes);
            else blas_impl::gemm_run<B,R>(h,'N','N',m,n,k,C(.7,.1),dra.data(),m,dcb.data(),k,C(.2,-.1),c.data(),m,work.backend,bytes);
            run_clean(run.finish());same(read(h,ref),read(h,c));work.check();
        }
    }
}
template<DdlaBackend B,class T> void vbatched(DdlaHandle_t h)
{
    if constexpr (B == DdlaBackend::GPU) {
        constexpr int batches=3, dim=2, elements=batches*dim*dim;
        std::vector<T> av(elements,value<T>(.25,.01)),bv(elements,value<T>(.5,-.02)),cv(elements,value<T>(-.75));
        ddla::vector<T> a(h,elements),b(h,elements),c(h,elements),ref(h,elements);
        write(h,a,av);write(h,b,bv);write(h,c,cv);write(h,ref,cv);
        ddla::vector<int> dims(h,batches);write(h,dims,std::vector<int>(batches,dim));
        std::vector<const T*> ap,bp;std::vector<T*> cp,rp;
        for(int i=0;i<batches;++i){ap.push_back(a.data()+i*dim*dim);bp.push_back(b.data()+i*dim*dim);cp.push_back(c.data()+i*dim*dim);rp.push_back(ref.data()+i*dim*dim);}
        ddla::vector<const T*> da(h,batches),db(h,batches);ddla::vector<T*> dc(h,batches),dr(h,batches);
        write(h,da,ap);write(h,db,bp);write(h,dc,cp);write(h,dr,rp);
        std::size_t bytes=0;
        workspace_test::Scope query(true);blas_impl::gemmVbatched_workspace_size(h,batches,&bytes);query_clean(query.finish());Guarded scratch(h,bytes,0);
        blas_impl::gemmVbatched_check<B,T>(h,'N','N',dims.data(),dims.data(),dims.data(),T(1),da.data(),dims.data(),db.data(),dims.data(),T(0),dc.data(),dims.data(),batches);
        workspace_test::Scope validation;
        int maximum_m, maximum_n, maximum_k;blas_impl::vbatched_dimensions_check(h,'N','N',dims.data(),dims.data(),dims.data(),dims.data(),dims.data(),dims.data(),batches,scratch.backend,bytes,maximum_m,maximum_n,maximum_k);
        run_clean(validation.finish());
        gemmVbatched<B,T>(h,'N','N',dims.data(),dims.data(),dims.data(),T(1),da.data(),dims.data(),db.data(),dims.data(),T(0),dr.data(),dims.data(),batches);
        workspace_test::Scope run;
        blas_impl::gemmVbatched_run<B,T>(h, maximum_m, maximum_n, maximum_k,'N','N',dims.data(),dims.data(),dims.data(),T(1),da.data(),dims.data(),db.data(),dims.data(),T(0),dc.data(),dims.data(),batches);
        run_clean(run.finish());same(read(h,ref),read(h,c));scratch.check();
    }
}
void layout(DdlaHandle_t h)
{
    if (workspace_test::available()) {
        // Negative control: diagnostics must detect a deliberately owning path.
        workspace_test::Scope allocation;
        ddla::vector<std::byte> probe(h,64);
        const auto allocated=allocation.finish();
        require(allocated.ddla_alloc>0,"workspace allocation diagnostic negative control failed");
        workspace_test::Scope release;
        probe.reset();
        require(release.finish().ddla_free>0,"workspace release diagnostic negative control failed");
    }
    // Carving: each take() advances by its padded_bytes, so buffers start 256-byte aligned relative to the base.
    alignas(256) char base[1024];
    char* w=base;
    int* const i=detail::take<int>(w,3);
    auto* const z=detail::take<std::complex<double>>(w,17);
    double* const d=detail::take<double>(w,0);
    require(reinterpret_cast<char*>(i)==base && reinterpret_cast<char*>(z)==base+256 && reinterpret_cast<char*>(d)==base+768 &&
            w==base+768 && detail::padded_bytes<int>(3)+detail::padded_bytes<std::complex<double>>(17)==768 &&
            detail::padded_bytes<double>(0)==0,"workspace carving");
}
template<DdlaBackend B> void backend()
{
    DdlaHandle_t h{};DDLA_CHECK(ddlaInit(h,B));DDLA_CHECK(ddlaSet(h,MPI_COMM_SELF));layout(h);
    lapack<B,float>(h);lapack<B,double>(h);lapack<B,std::complex<float>>(h);lapack<B,std::complex<double>>(h);
    blas<B,float>(h);blas<B,double>(h);blas<B,std::complex<float>>(h);blas<B,std::complex<double>>(h);mixed<B,float>(h);mixed<B,double>(h);
    vbatched<B,float>(h);vbatched<B,double>(h);vbatched<B,std::complex<float>>(h);vbatched<B,std::complex<double>>(h);
    DDLA_CHECK(ddlaDestroy(h));
}
}
int main(int argc,char** argv)
{
    MPI_Init(&argc,&argv);int status=0;
    try {
#if DDLA_HAS_CPU
        backend<DdlaBackend::CPU>();
#endif
#if DDLA_HAS_GPU
        backend<DdlaBackend::GPU>();
#endif
        std::cout<<"workspace core PASS\n";
    }catch(const std::exception& e){std::cerr<<"workspace core FAIL: "<<e.what()<<"\n";status=1;}
    MPI_Finalize();return status;
}
