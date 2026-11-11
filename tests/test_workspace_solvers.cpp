#include "api_grid_test_common.h"
#include "ddla_stream.h"
#include "mpi_datatype.h"
#include "workspace_test_common.h"
#include "Eigensolver/pstedc_impl.h"
#include "SVD/pbdsdc_impl.h"
#include "Eigensolver/two_stage/phe2hb_impl.h"
#include "dc_threads.h"
#include <atomic>

#include <array>
#include <cstring>
#include <limits>

using namespace api_grid_test;
namespace {
using namespace ddla::spectral_impl;
// Exactly the sized backend and host bytes, between 256-byte guards. The same
// addresses serve every run of a case.
class GuardedScratch {
    static constexpr std::size_t guard=256;
    DdlaHandle_t h_;std::size_t backend_bytes_,host_bytes_;ddla::vector<char> backend_;std::vector<char> host_;
public:
    GuardedScratch(DdlaHandle_t h,std::size_t backend_bytes,std::size_t host_bytes)
        :h_(h),backend_bytes_(backend_bytes),host_bytes_(host_bytes),
         backend_(h,backend_bytes+2*guard),host_(host_bytes+2*guard,char(0x5a))
    {DDLA_CHECK(ddlaMemset(h,backend_.data(),0x5a,backend_.size()));}
    void* backend(){return backend_.data()+guard;}
    void* host(){return host_.data()+guard;}
    void verify() {
        std::array<char,2*guard> device{};
        DDLA_CHECK(ddlaMemcpy(h_,device.data(),backend_.data(),guard));
        DDLA_CHECK(ddlaMemcpy(h_,device.data()+guard,backend_.data()+guard+backend_bytes_,guard));
        DDLA_CHECK(ddlaSynchronize(h_));bool good=true;
        for(std::size_t i=0;i<guard;++i)good=good&&host_[i]==char(0x5a)&&
            host_[guard+host_bytes_+i]==char(0x5a)&&device[i]==char(0x5a)&&device[guard+i]==char(0x5a);
        require_close(h_,"solver exact workspace guards",good?0:1,0);
    }
};
template<class S>bool no_allocation(const S& s) {
    return !s.ddla_alloc&&!s.ddla_free&&!s.pinned_alloc&&!s.raw_gpu_alloc&&!s.host_new&&!s.vendor_query&&!s.resource_setup;
}
// GPU messages staged in the handle's host buffers (tunnel builds, and the
// all-to-all of builds without CCL; comm_traits.h) grow them on first use:
// there the first run of a case may allocate.
template<DdlaBackend B>bool may_grow_staging(int pass) {
#if defined(DDLA_USE_GPU_CPU_TUNNEL) || !defined(DDLA_USE_CCL)
    return B==DdlaBackend::GPU&&pass==0;
#else
    (void)pass;return false;
#endif
}
template<class F>void measured_run(DdlaHandle_t h,F&& f,bool may_allocate=false) {
    workspace_test::Scope scope;f();const auto s=scope.finish();
    require_close(h,"solver raw execution has no managed allocations, queries or resource creation",may_allocate||no_allocation(s)?0:1,0);
}
// f writes this rank's byte counts. Rank 0 alone first proves the query local.
template<class F>void local_query(DdlaHandle_t h,F&& f) {
    MPI_Barrier(h->comm);bool pure=true;
    if(!h->myid) {
        workspace_test::Scope scope(true);f();const auto s=scope.finish();
        pure=no_allocation(s)&&!s.copies&&!s.launches&&!s.syncs&&!s.collectives;
    }
    MPI_Barrier(h->comm);require_close(h,"solver query is rank-local and output-pure",pure?0:1,0);
    f();
}

template<DdlaBackend B,class R>void tridiagonal_case(DdlaHandle_t h,int n,bool compact,bool lend_q) {
    const detail::RowShare share(n,h->nprocs);const int rows=share.count(h->myid),lo=share.lo(h->myid);
    const int ld=std::max(1,rows)+(compact?0:2+h->myid);
    const R poison=R(-919);
    std::vector<R> diagonal(n),off(std::max(1,n-1)),d(n),e(off.size()),w(n);
    for(int i=0;i<n;++i)diagonal[i]=R(2+i)/R(3);
    for(int i=0;i+1<n;++i)off[i]=R((i%3)+1)/R(13);
    const auto ysize=std::max<std::size_t>(1,std::size_t(ld)*n);
    const auto qsize=std::max<std::size_t>(1,std::size_t(std::max(1,rows))*n);
    DeviceBuffer<R> Y(h,ysize),Q(h,qsize);
    std::vector<R> original(ysize,poison);
    R* scratch=lend_q?Q.ptr:nullptr;
    const std::size_t scratch_count=lend_q?qsize:0;
    const auto p=pstedc_check<R>(h,n,d.data(),e.data(),w.data(),Y.ptr,ld,scratch,scratch_count);
    std::size_t backend_bytes=0,host_bytes=0;
    local_query(h,[&]{pstedc_workspace_size<R>(h,p,&backend_bytes,&host_bytes);});
    require_close(h,"compact output and caller factor storage are preserved in solver plan",
        (n&&rows&&((p.borrow_q!=lend_q)||(p.borrow_q2!=compact)))?1:0,0);
    GuardedScratch work(h,backend_bytes,host_bytes);PstedcResources resources(h,p);
    std::vector<R> first_values,first_vectors;
    for(int repeat=0;repeat<2;++repeat) {
        d=diagonal;e=off;upload(h,Y.ptr,original);
        measured_run(h,[&]{pstedc_run<B,R>(h,p,resources,d.data(),e.data(),w.data(),Y.ptr,scratch,
            work.backend(),backend_bytes,work.host(),host_bytes);},may_grow_staging<B>(repeat));
        auto local=download(h,Y.ptr,ysize);
        bool padding=true;for(int j=0;j<n;++j)for(int r=rows;r<ld;++r)padding=padding&&local[r+std::size_t(j)*ld]==poison;
        require_close(h,"tridiagonal output padding",padding?0:1,0);
        if(!repeat){first_values=w;first_vectors=local;}
        else {
            require_close(h,"tridiagonal same-scratch repeat values",n&&std::memcmp(w.data(),first_values.data(),n*sizeof(R))?1:0,0);
            require_close(h,"tridiagonal same-scratch repeat vectors",std::memcmp(local.data(),first_vectors.data(),local.size()*sizeof(R))?1:0,0);
        }
        // Replicate only this small test matrix to check T*Z=Z*diag(w) and Z^T*Z=I.
        std::vector<R> matrix(std::size_t(n)*n,R(0));
        for(int j=0;j<n;++j)for(int r=0;r<rows;++r)matrix[lo+r+std::size_t(j)*n]=local[r+std::size_t(j)*ld];
        if(n)MPI_Allreduce(MPI_IN_PLACE,matrix.data(),n*n,detail::mpi_datatype<R>(),MPI_SUM,h->comm);
        double residual=0,orthogonal=0;
        for(int j=0;j<n;++j)for(int i=0;i<n;++i) {
            R tz=diagonal[i]*matrix[i+std::size_t(j)*n];
            if(i)tz+=off[i-1]*matrix[i-1+std::size_t(j)*n];
            if(i+1<n)tz+=off[i]*matrix[i+1+std::size_t(j)*n];
            residual=std::max(residual,workspace_test::finite_error(double(std::abs(tz-w[j]*matrix[i+std::size_t(j)*n]))));
            R dot=0;for(int r=0;r<n;++r)dot+=matrix[r+std::size_t(i)*n]*matrix[r+std::size_t(j)*n];
            orthogonal=std::max(orthogonal,workspace_test::finite_error(double(std::abs(dot-R(i==j)))));
        }
        const double tol=std::is_same<R,float>::value?2e-4:2e-11;
        require_close(h,"tridiagonal residual",residual,tol);require_close(h,"tridiagonal orthogonality",orthogonal,tol);
        bool sorted=true;for(int i=0;i<n;++i)sorted=sorted&&std::isfinite(w[i])&&(i==0||w[i-1]<=w[i]);
        require_close(h,"tridiagonal ascending values",sorted?0:1,0);work.verify();
    }
}

template<DdlaBackend B,class R>void bidiagonal_case(DdlaHandle_t h,int n,char uplo,int pattern,bool padded) {
    const detail::RowShare share(n,h->nprocs);const int rows=share.count(h->myid),lo=share.lo(h->myid);
    const int ld=std::max(1,rows)+(padded?2+h->myid:0);
    const auto elements=std::max<std::size_t>(1,std::size_t(ld)*n);const R poison=R(-313);
    std::vector<R> diagonal(n),off(std::max(1,n-1),R(0)),d(n),e(off.size()),values(n),initial(elements,poison);
    for(int i=0;i<n;++i)diagonal[i]=pattern==2?R(0):pattern==1?R(2+i%3):R(2)+R(i)/R(17);
    if(!pattern)for(int i=0;i+1<n;++i)off[i]=R(1+i%4)/R(11);
    DeviceBuffer<R> U(h,elements),V(h,elements);
    const auto p=pbdsdc_check(h,B,uplo,n,d.data(),e.data(),values.data(),U.ptr,V.ptr,ld);
    std::size_t backend_bytes=0,host_bytes=0;
    local_query(h,[&]{pbdsdc_workspace_size<R>(h,p,&backend_bytes,&host_bytes);});
    GuardedScratch work(h,backend_bytes,host_bytes);detail::DcHostTeam team(p.threads);
    std::vector<R> first_values,first_U,first_V;
    for(int repeat=0;repeat<2;++repeat) {
        d=diagonal;e=off;upload(h,U.ptr,initial);upload(h,V.ptr,initial);int info=-1;
        measured_run(h,[&]{pbdsdc_run<B,R>(h,p,team,d.data(),e.data(),values.data(),U.ptr,V.ptr,info,
            work.backend(),backend_bytes,work.host(),host_bytes);},may_grow_staging<B>(repeat));
        require_close(h,"bidiagonal info",info?1:0,0);
        auto u=download(h,U.ptr,elements),v=download(h,V.ptr,elements);
        bool padding=true;for(int j=0;j<n;++j)for(int r=rows;r<ld;++r)
            padding=padding&&u[r+std::size_t(j)*ld]==poison&&v[r+std::size_t(j)*ld]==poison;
        require_close(h,"bidiagonal output padding",padding?0:1,0);
        if(!repeat){first_values=values;first_U=u;first_V=v;}
        else require_close(h,"bidiagonal same-scratch repeat",(n&&std::memcmp(values.data(),first_values.data(),n*sizeof(R)))||
            std::memcmp(u.data(),first_U.data(),elements*sizeof(R))||std::memcmp(v.data(),first_V.data(),elements*sizeof(R))?1:0,0);
        std::vector<R> gu(std::size_t(n)*n,R(0)),gv(gu);
        for(int j=0;j<n;++j)for(int r=0;r<rows;++r) {
            gu[lo+r+std::size_t(j)*n]=u[r+std::size_t(j)*ld];gv[lo+r+std::size_t(j)*n]=v[r+std::size_t(j)*ld];
        }
        if(n){MPI_Allreduce(MPI_IN_PLACE,gu.data(),n*n,detail::mpi_datatype<R>(),MPI_SUM,h->comm);
            MPI_Allreduce(MPI_IN_PLACE,gv.data(),n*n,detail::mpi_datatype<R>(),MPI_SUM,h->comm);}
        double residual=0,orthogonal=0;
        for(int j=0;j<n;++j)for(int i=0;i<n;++i) {
            R product=0,du=0,dv=0;
            for(int k=0;k<n;++k) {
                product+=gu[i+std::size_t(k)*n]*values[k]*gv[j+std::size_t(k)*n];
                du+=gu[k+std::size_t(i)*n]*gu[k+std::size_t(j)*n];
                dv+=gv[k+std::size_t(i)*n]*gv[k+std::size_t(j)*n];
            }
            const R expected=i==j?diagonal[i]:((uplo=='U'&&j==i+1)?off[i]:((uplo=='L'&&i==j+1)?off[j]:R(0)));
            residual=std::max(residual,workspace_test::finite_error(double(std::abs(product-expected))));
            orthogonal=std::max({orthogonal,workspace_test::finite_error(double(std::abs(du-R(i==j)))),workspace_test::finite_error(double(std::abs(dv-R(i==j))))});
        }
        const double tol=std::is_same<R,float>::value?6e-4:3e-11;
        require_close(h,"bidiagonal reconstruction",residual,tol);require_close(h,"bidiagonal orthogonality",orthogonal,tol);
        bool sorted=true;for(int i=0;i<n;++i)sorted=sorted&&std::isfinite(values[i])&&(i==0||values[i-1]>=values[i]);
        require_close(h,"bidiagonal descending values",sorted?0:1,0);
        if(pattern==1) {
            std::vector<int> order(n);for(int j=0;j<n;++j)order[j]=j;
            std::stable_sort(order.begin(),order.end(),[&](int a,int b){return diagonal[a]>diagonal[b];});
            bool stable=true;for(int j=0;j<n;++j)for(int r=0;r<n;++r)
                stable=stable&&gu[r+std::size_t(j)*n]==R(r==order[j])&&gv[r+std::size_t(j)*n]==R(r==order[j]);
            require_close(h,"bidiagonal stable tie ordering",stable?0:1,0);
        }
        work.verify();
    }
    if(!h->myid&&(pattern==1||n<=p.leaf)) {
        const auto actual=pattern==1&&n>p.leaf?3*std::size_t(share.count(0)):
            (n?2*std::size_t(n)*n+n:0);
        std::cout<<"bidiagonal retained leaf bytes actual_rank0="<<actual*sizeof(R)
            <<" reserved_rank0="<<pbdsdc_retained_bound(p)*sizeof(R)<<" n="<<n<<" leaf="<<p.leaf<<" ranks="<<p.ranks<<std::endl;
    }
}
void team_case(DdlaHandle_t h) {
    detail::DcHostTeam team(3);std::array<std::atomic<int>,41> touched{};
    for(auto& value:touched)value.store(0);
    measured_run(h,[&]{for(int repeat=0;repeat<3;++repeat)team.run(3,41,4,[&](int lo,int hi){
        for(int i=lo;i<hi;++i)touched[i].fetch_add(1,std::memory_order_relaxed);
    });});
    bool good=true;for(auto& value:touched)good=good&&value.load()==3;
    require_close(h,"prepared dynamic worker chunks",good?0:1,0);
    bool caught=false;try{team.run(3,41,1,[](int lo,int){if(lo==7)throw std::runtime_error("worker failure");});}
    catch(const std::runtime_error&){caught=true;}
    team.run(3,41,4,[&](int lo,int hi){for(int i=lo;i<hi;++i)touched[i].fetch_add(1,std::memory_order_relaxed);});
    for(auto& value:touched)good=good&&value.load()==4;
    require_close(h,"prepared workers join exceptions and remain reusable",caught&&good?0:1,0);
}

template<class T>T band_value(int i,int j) {
    using R=typename real_type<T>::type;
    if(i==j)return T(3+i);
    if(i>j) {
        if constexpr(std::is_same<T,R>::value)return T(R((i*17+j*11)%29-14)/R(31));
        else return T(R((i*17+j*11)%29-14)/R(31),R((i*13+j*19)%23-11)/R(37));
    }
    if constexpr(std::is_same<T,R>::value)return band_value<T>(j,i);else return std::conj(band_value<T>(j,i));
}
template<class T>T adjoint(T x){if constexpr(std::is_same<T,typename real_type<T>::type>::value)return x;else return std::conj(x);}
template<class T>std::vector<T> gather_square(DdlaHandle_t h,const std::vector<T>& local,const int* desc,int n) {
    std::vector<T> result(std::size_t(n)*n,T(0));
    const int rows=num_loc(n,desc[DDLA_MB_],h->myprow_,0,h->nprows_),cols=num_loc(n,desc[DDLA_NB_],h->mypcol_,0,h->npcols_);
    for(int j=0;j<cols;++j)for(int i=0;i<rows;++i) {
        const int gi=indxl2g(i,desc[DDLA_MB_],h->myprow_,0,h->nprows_),gj=indxl2g(j,desc[DDLA_NB_],h->mypcol_,0,h->npcols_);
        result[gi+std::size_t(gj)*n]=local[i+std::size_t(j)*desc[DDLA_LLD_]];
    }
    if(n)MPI_Allreduce(MPI_IN_PLACE,result.data(),n*n,detail::mpi_datatype<T>(),MPI_SUM,h->comm);return result;
}
template<DdlaBackend B,class T>void band_case(DdlaHandle_t h,int n,int upper_tile,bool inplace) {
    using R=typename real_type<T>::type;constexpr int nb=8,nbw=4;
    int desc[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(desc,h,n+3,n+2,nb,nb,0,0));desc[DDLA_LLD_]+=2+h->myid;
    auto original=make_local<T>(h,desc,[](int i,int j){return band_value<T>(i,j);});DeviceBuffer<T>A(h,original.size());
    const int steps=n?(n-1)/nbw:0;const auto count=std::size_t(steps)*nbw*nbw;
    std::vector<T> initial(std::max<std::size_t>(count,1)+16,T(-818));DeviceBuffer<T> factors(h,initial.size());
    setenv("DDLA_QR_PANEL",inplace?"inplace":"gather",1);
    const auto p=phe2hb_check(h,B,n,A.ptr,desc,nbw,factors.ptr+8,upper_tile,false);
    std::size_t backend_bytes=0,host_bytes=0;MPI_Barrier(h->comm);bool pure=true;
    if(!h->myid){workspace_test::Scope scope(true);phe2hb_workspace_size<T>(h,p,A.ptr,factors.ptr+8,&backend_bytes,&host_bytes);
        const auto trace=scope.finish();pure=!trace.ddla_alloc&&!trace.ddla_free&&!trace.pinned_alloc&&!trace.raw_gpu_alloc&&!trace.host_new&&
            !trace.copies&&!trace.launches&&!trace.syncs&&!trace.collectives&&!trace.resource_setup;}
    MPI_Barrier(h->comm);require_close(h,"band reduction one-rank query purity",pure?0:1,0);
    phe2hb_workspace_size<T>(h,p,A.ptr,factors.ptr+8,&backend_bytes,&host_bytes);GuardedScratch work(h,backend_bytes,host_bytes);
    setenv("DDLA_QR_PANEL",inplace?"gather":"inplace",1);
    std::vector<T> first_a,first_t;
    for(int repeat=0;repeat<2;++repeat) {
        upload(h,A.ptr,original);upload(h,factors.ptr,initial);int info=-1;
        measured_run(h,[&]{phe2hb_run<B,T>(h,p,A.ptr,factors.ptr+8,info,work.backend(),backend_bytes,work.host(),host_bytes);},
            may_grow_staging<B>(repeat));
        require_close(h,"band reduction info",info?1:0,0);
        auto a=download(h,A.ptr,original.size()),t=download(h,factors.ptr,initial.size());
        bool padding=true;
        const int rows=num_loc(desc[DDLA_M_],nb,h->myprow_,0,h->nprows_),cols=num_loc(desc[DDLA_N_],nb,h->mypcol_,0,h->npcols_);
        for(int j=0;j<cols;++j)for(int i=0;i<desc[DDLA_LLD_];++i) {
            const int gi=i<rows?indxl2g(i,nb,h->myprow_,0,h->nprows_):desc[DDLA_M_],gj=indxl2g(j,nb,h->mypcol_,0,h->npcols_);
            if(gi>=n||gj>=n)padding=padding&&a[i+std::size_t(j)*desc[DDLA_LLD_]]==original[i+std::size_t(j)*desc[DDLA_LLD_]];
        }
        for(std::size_t i=0;i<t.size();++i)if(i<8||i>=8+count)padding=padding&&t[i]==initial[i];
        require_close(h,"band reduction parent/padding and factor guards",padding?0:1,0);
        if(!repeat){first_a=a;first_t=t;}else require_close(h,"band same-scratch bitwise repeat",
            std::memcmp(a.data(),first_a.data(),a.size()*sizeof(T))||std::memcmp(t.data(),first_t.data(),t.size()*sizeof(T))?1:0,0);
        if(n) {
            int dz[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(dz,h,n,n,nb,nb,0,0));
            auto identity=make_local<T>(h,dz,[](int i,int j){return T(i==j);});DeviceBuffer<T>Z(h,identity.size());upload(h,Z.ptr,identity);
            punmtr_he2hb<B,T>(h,'N',n,nbw,A.ptr,desc,factors.ptr+8,Z.ptr,dz,info);
            const auto reduced=gather_square(h,a,desc,n),Q=gather_square(h,download(h,Z.ptr,identity.size()),dz,n);
            std::vector<T> band(std::size_t(n)*n,T(0)),tmp(band),reconstructed(band);
            for(int j=0;j<n;++j)for(int i=std::max(0,j-nbw);i<=j;++i) {
                band[i+std::size_t(j)*n]=reduced[i+std::size_t(j)*n];
                band[j+std::size_t(i)*n]=adjoint(reduced[i+std::size_t(j)*n]);
            }
            for(int j=0;j<n;++j)for(int k=0;k<n;++k)for(int i=0;i<n;++i)
                tmp[i+std::size_t(j)*n]+=band[i+std::size_t(k)*n]*Q[k+std::size_t(j)*n];
            for(int j=0;j<n;++j)for(int k=0;k<n;++k)for(int i=0;i<n;++i)
                reconstructed[i+std::size_t(j)*n]+=adjoint(Q[k+std::size_t(i)*n])*tmp[k+std::size_t(j)*n];
            double error=0;for(int j=0;j<n;++j)for(int i=0;i<n;++i)
                error=std::max(error,workspace_test::finite_error(double(std::abs(reconstructed[i+std::size_t(j)*n]-band_value<T>(i,j)))));
            require_close(h,"band independent reconstruction",error,std::is_same<R,float>::value?8e-4:8e-11);
        }
        work.verify();
    }
    unsetenv("DDLA_QR_PANEL");
}

template<DdlaBackend B>void run(MPI_Comm comm,int pr,int pc,char major) {
    DdlaHandle_t h=nullptr;ddlaInit(h,B);ddlaSet(h,comm,pr,pc,major);
    team_case(h);
#define BAND(T) band_case<B,T>(h,17,0,false);band_case<B,T>(h,17,0,true)
    BAND(float);BAND(double);BAND(std::complex<float>);BAND(std::complex<double>);
#undef BAND
    band_case<B,double>(h,1,0,false);band_case<B,std::complex<double>>(h,0,0,true);
    if constexpr(B==DdlaBackend::CPU){band_case<B,double>(h,17,8,false);band_case<B,std::complex<double>>(h,17,8,true);}
    // Force both a many-leaf tree and a single-leaf tree. Policy reads occur
    // before each measured region, and no child LAPACK query is needed.
    setenv("DDLA_EIG_DC_LEAF","4",1);
    tridiagonal_case<B,float>(h,19,false,false);tridiagonal_case<B,double>(h,19,false,false);
    tridiagonal_case<B,float>(h,19,true,true);tridiagonal_case<B,double>(h,19,true,true);
    tridiagonal_case<B,float>(h,1,false,false);tridiagonal_case<B,double>(h,0,true,false);
    unsetenv("DDLA_EIG_DC_LEAF");
    setenv("DDLA_SVD_DC_LEAF","4",1);setenv("DDLA_SVD_DC_THREADS","3",1);
    for(char uplo:{'U','L'}) {
        bidiagonal_case<B,float>(h,23,uplo,0,true);bidiagonal_case<B,double>(h,23,uplo,0,true);
        bidiagonal_case<B,float>(h,23,uplo,1,false);bidiagonal_case<B,double>(h,23,uplo,1,false);
        bidiagonal_case<B,double>(h,23,uplo,2,true);
    }
    bidiagonal_case<B,float>(h,1,'U',0,true);bidiagonal_case<B,double>(h,0,'L',0,true);
    if(major=='R')bidiagonal_case<B,double>(h,389,'U',0,false);
    unsetenv("DDLA_SVD_DC_LEAF");unsetenv("DDLA_SVD_DC_THREADS");
    if(!h->myid)std::cout<<"workspace solvers PASS backend="<<int(B)<<" grid="<<pr<<"x"<<pc<<" major="<<major
        <<" allocation_trace="<<(workspace_test::available()?"enabled":"numerical-only")<<std::endl;
    ddlaDestroy(h);
}
}
int main(int argc,char** argv) {
    ddla_test_mpi_init(&argc,&argv);int size=1;MPI_Comm_size(MPI_COMM_WORLD,&size);TestOptions options;std::string error;
    if(!parse_options(argc,argv,size,options,error)){std::cerr<<error<<std::endl;MPI_Abort(MPI_COMM_WORLD,2);}
    for(auto grid:options.grids)for(char major:{'R','C'}) {
        run<DdlaBackend::CPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#if DDLA_HAS_GPU
        run<DdlaBackend::GPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#endif
    }
    MPI_Finalize();return 0;
}
