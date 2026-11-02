#include "Eigensolver/pstedc_impl.h"
#include "api_grid_test_common.h"
#include "ddla_stream.h"
#include "Eigensolver/pstedc_workspace.h"
#include "workspace_test_common.h"
#include <array>
#include <optional>
#include <type_traits>
#include <vector>

using namespace api_grid_test;
using namespace ddla::spectral_impl;
namespace {
struct Env {
    const char* name;bool present;std::string saved;
    Env(const char* n,const char* value):name(n),present(std::getenv(n)!=nullptr),saved(present?std::getenv(n):""){setenv(name,value,1);}
    ~Env(){if(present)setenv(name,saved.c_str(),1);else unsetenv(name);}
};
bool no_growth(const workspace_test::Snapshot& s)
{
    return !s.ddla_alloc&&!s.ddla_free&&!s.pinned_alloc&&!s.raw_gpu_alloc&&!s.host_new&&!s.vendor_query&&!s.resource_setup;
}
bool pure(const workspace_test::Snapshot& s)
{return no_growth(s)&&!s.copies&&!s.launches&&!s.syncs&&!s.collectives;}

// Exactly the sized backend and host bytes, between 256-byte canaries.
class Guarded {
    static constexpr std::size_t guard=256;
    DdlaHandle_t h_;std::size_t backend_bytes_,host_bytes_;
    ddla::vector<unsigned char> backend_;std::vector<unsigned char> host_;
public:
    Guarded(DdlaHandle_t h,std::size_t backend_bytes,std::size_t host_bytes)
        :h_(h),backend_bytes_(backend_bytes),host_bytes_(host_bytes),backend_(h,backend_bytes+2*guard),
         host_(host_bytes+2*guard,0x69)
    {
        DDLA_CHECK(ddlaMemset(h,backend_.data(),0x69,backend_.size()));
        DDLA_CHECK(ddlaSynchronize(h));
    }
    void* backend(){return backend_.data()+guard;}
    void* host(){return host_.data()+guard;}
    void verify()
    {
        std::array<unsigned char,2*guard> device{};
        DDLA_CHECK(ddlaMemcpy(h_,device.data(),backend_.data(),guard));
        DDLA_CHECK(ddlaMemcpy(h_,device.data()+guard,backend_.data()+guard+backend_bytes_,guard));
        DDLA_CHECK(ddlaSynchronize(h_));bool okay=true;
        for(std::size_t i=0;i<guard;++i)okay=okay&&host_[i]==0x69&&host_[guard+host_bytes_+i]==0x69
            &&device[i]==0x69&&device[guard+i]==0x69;
        require_close(h_,"pstedc exact workspace guards",okay?0:1,0);
    }
};
template<class R>double tolerance(){return std::is_same_v<R,float>?8e-4:2e-10;}

template<DdlaBackend B,class R>
void solve_case(DdlaHandle_t h,char jobz,int n,bool split_e=false)
{
    const ddla::detail::RowShare share(n,h->nprocs);
    const int nr=share.count(h->myid),lo=share.lo(h->myid);
    const int ld=std::max(1,nr)+3;const R sentinel=R(-731);
    std::vector<R> diag(n+2,sentinel),off(n+2,sentinel),values(n+2,sentinel);
    for(int i=0;i<n;++i)diag[i+1]=R(2);
    for(int i=0;i+1<n;++i)off[i+1]=split_e&&i%7==0?R(0):R(-1);
    const auto original_d=diag,original_e=off;
    const std::vector<R> initial(std::size_t(ld)*n+2,sentinel);
    DeviceBuffer<R> z(h,initial.size()),reference(h,initial.size());upload(h,z.ptr,initial);upload(h,reference.ptr,initial);
    auto refd=diag,refe=off,refw=values;int info=0;
    pstedc<R>(h,jobz,n,refd.data()+1,refe.data()+1,refw.data()+1,jobz=='V'?reference.ptr+1:nullptr,ld,info);
    const auto expected=download(h,reference.ptr,initial.size());
    const auto plan=pstedc_check<R>(h,jobz,n,diag.data()+1,off.data()+1,values.data()+1,jobz=='V'?z.ptr+1:nullptr,ld);
    MPI_Barrier(h->comm);bool local=true;
    if(h->myid==0){
        workspace_test::Scope scope(true);std::size_t backend=0,host=0;
        pstedc_workspace_size<R>(h,plan,&backend,&host);local=pure(scope.finish());
    }
    MPI_Barrier(h->comm);require_close(h,"one-rank pstedc query is pure",local?0:1,0);
    std::size_t backend_bytes=0,host_bytes=0;
    workspace_test::Scope query_scope(true);pstedc_workspace_size<R>(h,plan,&backend_bytes,&host_bytes);
    const auto snapshot=query_scope.finish();
    require_close(h,"pstedc query has no work, resources or allocation",pure(snapshot)?0:1,0);
    require_close(h,"pstedc query preserves inputs",diag==original_d&&off==original_e?0:1,0);
    if(jobz=='N'&&h->backend==DdlaBackend::GPU)require_close(h,"values-only pstedc has no device workspace",double(backend_bytes),0);
    if(h->backend==DdlaBackend::CPU)require_close(h,"CPU pstedc keeps its host arrays in backend memory",double(host_bytes),0);
    Guarded arena(h,backend_bytes,host_bytes);std::optional<PstedcResources> rows;
    if(plan.mode==PstedcMode::rows)rows.emplace(h,plan.rows);
    for(int repeat=0;repeat<2;++repeat){
        diag=original_d;off=original_e;std::fill(values.begin(),values.end(),sentinel);upload(h,z.ptr,initial);
        workspace_test::Scope run_scope;
        pstedc_run<B,R>(h,plan,diag.data()+1,off.data()+1,values.data()+1,jobz=='V'?z.ptr+1:nullptr,
            arena.backend(),backend_bytes,arena.host(),host_bytes,rows?&*rows:nullptr);
        const auto run_snapshot=run_scope.finish();
        require_close(h,"pstedc borrowed run has no scratch growth, resource setup or size query",no_growth(run_snapshot)?0:1,0);
        double error=0;
        for(int j=0;j<n;++j)error=std::max(error,workspace_test::relative_error(values[j+1],refw[j+1]));
        require_close(h,"checked and borrowed pstedc eigenvalues",error,tolerance<R>());
        if(!split_e){
            error=0;for(int j=0;j<n;++j){const double want=2-2*std::cos((j+1)*std::acos(-1.0)/(n+1));error=std::max(error,workspace_test::finite_error(std::abs(double(values[j+1])-want)));}
            require_close(h,"pstedc analytic Toeplitz eigenvalues",error,tolerance<R>());
        }
        require_close(h,"pstedc host input/output guards",diag.front()==sentinel&&diag.back()==sentinel&&off.front()==sentinel&&off.back()==sentinel&&values.front()==sentinel&&values.back()==sentinel?0:1,0);
        const auto result=download(h,z.ptr,initial.size());
        std::vector<double> full(std::size_t(n)*n,0);
        if(jobz=='V'){
            for(int j=0;j<n;++j)for(int r=0;r<nr;++r)
                full[lo+r+std::size_t(j)*n]=result[1+r+std::size_t(j)*ld];
            MPI_Allreduce(MPI_IN_PLACE,full.data(),n*n,MPI_DOUBLE,MPI_SUM,h->comm);
        }
        if(jobz=='V'){
            double residual=0,norm_error=0,agreement=0,orthogonality=0;
            for(int j=0;j<n;++j){double norm=0;
                for(int i=0;i<n;++i){const double x=full[i+std::size_t(j)*n];norm+=x*x;
                    double r=2*x-double(values[j+1])*x;
                    if(i)r+=double(original_e[i])*full[i-1+std::size_t(j)*n];
                    if(i+1<n)r+=double(original_e[i+1])*full[i+1+std::size_t(j)*n];
                    residual=std::max(residual,workspace_test::finite_error(std::abs(r)));
                }
                norm_error=std::max(norm_error,workspace_test::finite_error(std::abs(norm-1)));
            }
            for(int j=0;j<n;++j)for(int k=0;k<j;++k){double dot=0;for(int i=0;i<n;++i)dot+=full[i+std::size_t(j)*n]*full[i+std::size_t(k)*n];orthogonality=std::max(orthogonality,workspace_test::finite_error(std::abs(dot)));}
            for(int j=0;j<n;++j)for(int r=0;r<nr;++r){const auto at=1+r+std::size_t(j)*ld;
                agreement=std::max(agreement,workspace_test::finite_error(double(std::abs(result[at]-expected[at]))));}
            require_close(h,"pstedc eigenvector orthogonality",orthogonality,tolerance<R>());
            require_close(h,"pstedc eigenvector residual",residual,tolerance<R>());
            require_close(h,"pstedc eigenvector norms",norm_error,tolerance<R>());
            require_close(h,"checked and borrowed pstedc vectors",agreement,tolerance<R>());
        }
        bool padding=result.front()==sentinel&&result.back()==sentinel;
        for(int col=0;col<n;++col)for(int row=nr;row<ld;++row)padding=padding&&result[1+std::size_t(col)*ld+row]==sentinel;
        if(jobz=='N')padding=padding&&result==initial;
        require_close(h,"pstedc output padding/unused z",padding?0:1,0);
        arena.verify();
    }
}

template<class R>void check_scaling(DdlaHandle_t h)
{
    PstedcPlan p;p.mode=PstedcMode::rows;p.rows.backend=h->backend;
    p.rows.leaf=64;p.rows.strip=16;p.rows.root_chunk=32;p.rows.threads=1;
    p.rows.borrow_q2=true;
    for(int n:{20000,40000}){
        std::size_t single=0;
        for(int ranks:{1,2,4}){
            p.rows.n=n;p.rows.ranks=ranks;p.rows.rank=0;
            p.rows.ldy=(n+ranks-1)/ranks;
            std::size_t backend=0,host=0;pstedc_workspace_size<R>(h,p,&backend,&host);
            const auto total=backend+host;
            if(ranks==1)single=total;
            require_close(h,"pstedc quadratic workspace is distributed",
                ranks==4&&total>=single/2?1:0,0);
            if(h->myid==0)std::cout<<"PSTEDC_WORKSPACE n="<<n<<" ranks="<<ranks
                <<" scalar="<<sizeof(R)<<" backend_bytes="<<backend<<" host_bytes="<<host<<std::endl;
        }
    }
}

template<DdlaBackend B,class R>void cases(DdlaHandle_t h)
{
    check_scaling<R>(h);
    {
        // A large override must not revert a multi-rank eigenvector solve to
        // a single n-by-n leaf (including dimensions smaller than P).
        Env leaves("DDLA_EIG_DC_LEAF","4096");
        Env strips("DDLA_EIG_DC_STRIP","4096");
        for(int n:{2,3,4,73,1024,1025}) {
            const auto plan=pstedc_policy<R>(h,n);
            const int leaves=pstedc_leaves(n,plan.leaf);
            const int largest=pstedc_leaf_geometry(n,leaves,leaves-1);
            require_close(h,"distributed pstedc has no full-matrix leaf",
                h->nprocs>1 && (leaves<2 || largest>=n) ? 1 : 0,0);
            require_close(h,"distributed pstedc uses bounded secular strips",
                h->nprocs>1 && n>2 && plan.strip>=n ? 1 : 0,0);
            require_close(h,"distributed pstedc uses bounded root chunks",
                h->nprocs>1 && plan.root_chunk>=n ? 1 : 0,0);
        }
    }
    Env threads("DDLA_EIG_THREADS","2");
    {
        // Ranks without eigenvector rows may provide no output allocation.
        std::vector<R> d(1,R(2)),e(1,R(0)),w(1);
        DeviceBuffer<R> local(h,h->myid==0?1:0);int info=0;
        pstedc<R>(h,'V',1,d.data(),e.data(),w.data(),h->myid==0?local.ptr:nullptr,1,info);
        require_close(h,"pstedc zero-row null output",info||w[0]!=R(2)?1:0,0);
    }
    for(const char* leaf:{"256","8"}){
        Env leaves("DDLA_EIG_DC_LEAF",leaf);
        for(const char* keep:{"0","64"}){Env cache("DDLA_EIG_DC_KEEP_MB",keep);solve_case<B,R>(h,'N',73);}
        for(const char* split:{"0","1","2"}){Env splitting("DDLA_EIG_DC_SPLIT",split);solve_case<B,R>(h,'V',73);}
        solve_case<B,R>(h,'V',73);
    }
    Env leaves("DDLA_EIG_DC_LEAF","8");
    solve_case<B,R>(h,'N',131);solve_case<B,R>(h,'N',33,true);
    solve_case<B,R>(h,'V',33,true);solve_case<B,R>(h,'V',1);
    solve_case<B,R>(h,'N',0);solve_case<B,R>(h,'V',0);solve_case<B,R>(h,'V',2);
}
template<DdlaBackend B>void run(MPI_Comm comm,int pr,int pc,char major)
{
    DdlaHandle_t h=nullptr;ddlaInit(h,B);ddlaSet(h,comm,pr,pc,major);
    cases<B,float>(h);cases<B,double>(h);
    if(h->myid==0)std::cout<<"workspace pstedc PASS backend="<<int(B)<<" grid="<<pr<<"x"<<pc<<" major="<<major<<std::endl;
    ddlaDestroy(h);
}
}
int main(int argc,char** argv)
{
    MPI_Init(&argc,&argv);int size;MPI_Comm_size(MPI_COMM_WORLD,&size);TestOptions options;std::string error;
    if(!parse_options(argc,argv,size,options,error)){std::cerr<<error<<std::endl;MPI_Abort(MPI_COMM_WORLD,2);}
    for(auto grid:options.grids)for(char major:{'R','C'}){
#if DDLA_HAS_CPU
        run<DdlaBackend::CPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#endif
#if DDLA_HAS_GPU
        run<DdlaBackend::GPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#endif
    }
    MPI_Finalize();return 0;
}
