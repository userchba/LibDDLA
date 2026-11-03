#include "api_grid_test_common.h"
#include "workspace_test_common.h"
#include "Eigensolver/pstedc_workspace.h"
#include "Eigensolver/pstedc_numeric.h"
#include "ddla_stream.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <type_traits>

using namespace api_grid_test;
using namespace ddla::spectral_impl;
namespace {
struct Env {
    const char* key;bool had;std::string old;
    Env(const char* name,const char* value):key(name),had(std::getenv(name)!=nullptr),old(had?std::getenv(name):""){setenv(key,value,1);}
    ~Env(){if(had)setenv(key,old.c_str(),1);else unsetenv(key);}
};
bool no_growth(const workspace_test::Snapshot& s)
{return !s.ddla_alloc&&!s.ddla_free&&!s.pinned_alloc&&!s.raw_gpu_alloc&&!s.host_new&&!s.vendor_query&&!s.resource_setup;}
bool pure(const workspace_test::Snapshot& s)
{return no_growth(s)&&!s.copies&&!s.launches&&!s.syncs&&!s.collectives;}

// Exact query sizes surrounded by canaries, including CPU's merged host region.
class Scratch {
    static constexpr int guard=256;
    DdlaHandle_t h_;std::size_t b_,hbytes_;
    DeviceBuffer<unsigned char> backend_;std::vector<unsigned char> host_;
public:
    Scratch(DdlaHandle_t h,std::size_t b,std::size_t hb):h_(h),b_(b),hbytes_(hb),backend_(h,b+2*guard),host_(hb+2*guard,0x69)
    {DDLA_CHECK(ddlaMemset(h,backend_.ptr,0x69,b+2*guard));DDLA_CHECK(ddlaSynchronize(h));}
    void* backend(){return backend_.ptr+guard;}
    void* host(){return host_.data()+guard;}
    void check(){const auto all=download(h_,backend_.ptr,b_+2*guard);bool okay=true;
        for(int i=0;i<guard;++i)okay=okay&&all[i]==0x69&&all[guard+b_+i]==0x69&&host_[i]==0x69&&host_[guard+hbytes_+i]==0x69;
        require_close(h_,"descriptor workspace exact-size guards",okay?0:1,0);}
};

template<DdlaBackend B,class R>
void solve(DdlaHandle_t h,int n,int mb,int nb,const char* kind,bool borrowed,R scale=R(1))
{
    const int rs=h->nprows_-1,cs=h->npcols_-1;
    int desc[9];DDLA_CHECK(ddlaDescInit(desc,h,n+3,n+5,mb,nb,rs,cs));desc[DDLA_LLD_]+=3;
    const int nr=num_loc(n,mb,h->myprow_,rs,h->nprows_),nc=num_loc(n,nb,h->mypcol_,cs,h->npcols_);
    const int parentcols=num_loc(n+5,nb,h->mypcol_,cs,h->npcols_),ld=desc[DDLA_LLD_];
    const R sentinel=R(-731);
    std::vector<R> d(n+2,sentinel),e(n+2,sentinel),w(n+2,sentinel);
    for(int i=0;i<n;++i){
        const std::string type=kind;
        d[i+1]=type=="zero"?R(0):type=="diagonal"?R(i%5):R(2)+R((17*i)%23-11)/R(32);
        if(type=="toeplitz")d[i+1]=R(2);
        if(i+1<n){e[i+1]=R((13*i)%17+1)/R(128);if(i%2)e[i+1]=-e[i+1];
            if(type=="zero"||type=="diagonal"||(type=="split"&&i%7==0))e[i+1]=R(0);
            if(type=="toeplitz")e[i+1]=R(-1);
            if(type=="cluster"){d[i+1]=R(1)+R(i%3)*std::numeric_limits<R>::epsilon();e[i+1]*=std::numeric_limits<R>::epsilon()*R(4);}}
        d[i+1]*=scale;if(i+1<n)e[i+1]*=scale;
    }
    const auto original_d=d,original_e=e;
    const std::vector<R> initial(std::size_t(ld)*parentcols+2,sentinel);
    DeviceBuffer<R> output(h,initial.size());upload(h,output.ptr,initial);
    const auto plan=pstedc_check<R>(h,'V',n,d.data()+1,e.data()+1,w.data()+1,nr&&nc?output.ptr+1:nullptr,desc);
    if(h->nprocs>1&&n>1)
        require_close(h,"pstedc nontrivial multi-rank solve never has a full-size leaf",pstedc_leaves(n,plan.rows.leaf)>1&&plan.rows.leaf<n?0:1,0);
    if(h->nprocs>1&&n>1){
        const int share=n/h->nprocs+(n%h->nprocs!=0);
        require_close(h,"secular strips fit a rank share",plan.rows.strip<=std::max(2,share)?0:1,0);
        require_close(h,"root chunks fit a rank share",plan.rows.root_chunk<=std::max(1,share)?0:1,0);
    }
    std::size_t bytes=0,hostbytes=0;
    workspace_test::Scope query(true);pstedc_workspace_size<R>(h,plan,&bytes,&hostbytes);
    const auto queried=query.finish();require_close(h,"descriptor workspace query pure",pure(queried)?0:1,0);
    if(B==DdlaBackend::CPU)require_close(h,"pstedc CPU host region belongs to backend arena",hostbytes,0);
    // The query is rank-local and must also be callable on rank zero alone.
    bool alone=true;if(h->myid==0){workspace_test::Scope one(true);std::size_t b=0,hb=0;pstedc_workspace_size<R>(h,plan,&b,&hb);alone=pure(one.finish())&&b==bytes&&hb==hostbytes;}
    require_close(h,"pstedc one-rank query",alone?0:1,0);
    Scratch scratch(h,bytes,hostbytes);
    PstedcResources resources(h,plan.rows);
    for(int repeat=0;repeat<(borrowed?2:1);++repeat){
        d=original_d;e=original_e;upload(h,output.ptr,initial);int info=0;
        if(borrowed){workspace_test::Scope run;
            pstedc_run<B,R>(h,plan,d.data()+1,e.data()+1,w.data()+1,nr&&nc?output.ptr+1:nullptr,scratch.backend(),bytes,scratch.host(),hostbytes,&resources);
            const auto snapshot=run.finish();require_close(h,"pstedc borrowed run has no scratch growth or query",no_growth(snapshot)?0:1,0);
        }else pstedc<R>(h,'V',n,d.data()+1,e.data()+1,w.data()+1,nr&&nc?output.ptr+1:nullptr,desc,info);
        require_close(h,"pstedc solver info",info,0);
        const auto local=download(h,output.ptr,initial.size());std::vector<double> full(std::size_t(n)*n,0);
        bool padding=local.front()==sentinel&&local.back()==sentinel;
        for(int c=0;c<parentcols;++c)for(int r=0;r<ld;++r){
            const bool active=r<nr&&c<nc;const R value=local[1+r+std::size_t(c)*ld];
            if(active){const int i=indxl2g(r,mb,h->myprow_,rs,h->nprows_),j=indxl2g(c,nb,h->mypcol_,cs,h->npcols_);full[i+std::size_t(j)*n]=value;}
            else padding=padding&&value==sentinel;
        }
        if(n)MPI_Allreduce(MPI_IN_PLACE,full.data(),n*n,MPI_DOUBLE,MPI_SUM,h->comm);
        require_close(h,"pstedc output padding and parent matrix",padding?0:1,0);
        require_close(h,"pstedc host guards",d.front()==sentinel&&d.back()==sentinel&&e.front()==sentinel&&e.back()==sentinel&&w.front()==sentinel&&w.back()==sentinel?0:1,0);
        double residual=0,orthogonality=0,norm=0;
        for(int i=0;i<n;++i)norm=std::max(norm,std::abs(double(original_d[i+1]))+(i?std::abs(double(original_e[i])):0)+(i+1<n?std::abs(double(original_e[i+1])):0));
        for(int j=0;j<n;++j)for(int i=0;i<n;++i){
            const auto at=[&](int r,int c){return full[r+std::size_t(c)*n];};
            double r=(double(original_d[i+1])-double(w[j+1]))*at(i,j);
            if(i)r+=double(original_e[i])*at(i-1,j);if(i+1<n)r+=double(original_e[i+1])*at(i+1,j);
            residual=std::max(residual,workspace_test::finite_error(std::abs(r)));
            long double dot=0;for(int k=0;k<n;++k)dot+=static_cast<long double>(at(k,i))*at(k,j);
            orthogonality=std::max(orthogonality,workspace_test::finite_error(double(std::abs(dot-(i==j?1.L:0.L)))));
        }
        const double eps=std::numeric_limits<R>::epsilon();
        if(norm>0)residual/=norm*std::max(1,n)*eps;
        orthogonality/=std::max(1,n)*eps;
        require_close(h,"pstedc independent tridiagonal residual scaled by n eps norm",residual,100);
        require_close(h,"pstedc independent orthogonality scaled by n eps",orthogonality,100);
        if(n){
            std::vector<R> rd(original_d.begin()+1,original_d.end()-1),re(original_e.begin()+1,original_e.end()-1),work(1);int iw=0,li=0;
            stedc('N',n,rd.data(),re.data(),static_cast<R*>(nullptr),1,work.data(),1,&iw,1,li);
            require_close(h,"independent host reference info",li,0);double error=0;
            for(int j=0;j<n;++j)error=std::max(error,workspace_test::finite_error(std::abs(double(w[j+1])-double(rd[j]))/(std::max(double(std::abs(scale)),std::abs(double(rd[j])))+std::numeric_limits<double>::min())));
            require_close(h,"pstedc eigenvalues match independent host LAPACK",error,eps*100*std::max(1,n));
        }
        scratch.check();
    }
    if(h->myid==0)std::cout<<"PSTEDC_CASE type="<<(std::is_same<R,float>::value?"s":"d")<<" n="<<n<<" mb="<<mb<<" nb="<<nb<<" case="<<kind
        <<" borrowed_run="<<borrowed<<" leaves="<<pstedc_leaves(n,plan.rows.leaf)<<" backend_bytes="<<bytes<<" host_bytes="<<hostbytes<<" PASS"<<std::endl;
}

template<DdlaBackend B,class R>void cases(DdlaHandle_t h)
{
    Env leaf("DDLA_EIG_DC_LEAF","8");Env threads("DDLA_EIG_THREADS","2");
    for(int n:{0,1,2,3,7,33})solve<B,R>(h,n,3,3,"random",true);
    solve<B,R>(h,2,16,16,"random",true);solve<B,R>(h,7,16,16,"random",true);
    solve<B,R>(h,73,3,5,"random",true);solve<B,R>(h,73,3,3,"split",true);
    solve<B,R>(h,33,3,3,"zero",true);solve<B,R>(h,33,3,3,"diagonal",true);
    solve<B,R>(h,73,3,3,"cluster",true);solve<B,R>(h,131,3,3,"toeplitz",false);
    solve<B,R>(h,257,3,3,"toeplitz",true);
    {Env wider_leaves("DDLA_EIG_DC_LEAF","64");solve<B,R>(h,36,9,9,"random",true);}
    solve<B,R>(h,33,3,3,"random",false,std::is_same<R,float>::value?R(1e25):R(1e200));
    solve<B,R>(h,33,3,5,"random",true,std::is_same<R,float>::value?R(1e-25):R(1e-200));
    // A large leaf override must not create replicated full-size matrices.
    {Env large("DDLA_EIG_DC_LEAF","20000");int desc[9];ddlaDescInit(desc,h,20000,20000,128,128,0,0);
        R dummy=0;auto plan=pstedc_check<R>(h,'V',20000,&dummy,&dummy,&dummy,&dummy,desc);std::size_t b=0,hb=0;pstedc_workspace_size<R>(h,plan,&b,&hb);
        require_close(h,"nontrivial multi-rank leaf smaller than matrix",h->nprocs==1||plan.rows.leaf<20000?0:1,0);
        if(h->myid==0)std::cout<<"PSTEDC_WORKSPACE n=20000 scalar="<<sizeof(R)<<" ranks="<<h->nprocs<<" backend_bytes="<<b<<" host_bytes="<<hb<<std::endl;}
    // The descriptor form also preserves the linear-memory values-only path.
    {std::vector<R> d(7,R(2)),e(7,R(-1)),w(7),ref=d,off=e,work(1);int iw=0,info=0;
        pstedc<R>(h,'N',7,d.data(),e.data(),w.data(),nullptr,static_cast<const int*>(nullptr),info);
        stedc('N',7,ref.data(),off.data(),static_cast<R*>(nullptr),1,work.data(),1,&iw,1,info);
        double error=0;for(int i=0;i<7;++i)error=std::max(error,workspace_test::finite_error(std::abs(double(w[i]-ref[i]))));
        require_close(h,"descriptor values-only reference",error,100*std::numeric_limits<R>::epsilon());}
    // Missing output or mismatched metadata on one rank must fail on all ranks.
    std::vector<R> d(7,R(2)),e(7,R(-1)),w(7);int desc[9];ddlaDescInit(desc,h,7,7,1,1,0,0);
    DeviceBuffer<R> output(h,std::size_t(std::max(1,num_loc(7,1,h->myprow_,0,h->nprows_)))*std::max(1,num_loc(7,1,h->mypcol_,0,h->npcols_)));
    bool caught=false;try{auto bad=std::array<int,9>{};std::copy_n(desc,9,bad.data());if(h->myid==0)bad[DDLA_LLD_]=0;
        (void)pstedc_check<R>(h,'V',7,d.data(),e.data(),w.data(),output.ptr,bad.data());}catch(const std::invalid_argument&){caught=true;}
    require_close(h,"pstedc one-rank invalid descriptor agreed",caught?0:1,0);
    if(h->nprocs>1){caught=false;try{(void)pstedc_check<R>(h,'V',h->myid==0?6:7,d.data(),e.data(),w.data(),output.ptr,desc);}catch(const std::invalid_argument&){caught=true;}
        require_close(h,"pstedc dimension disagreement rejected",caught?0:1,0);}
}
template<DdlaBackend B>void run(int pr,int pc,char major)
{
    DdlaHandle_t h=nullptr;ddlaInit(h,B);ddlaSet(h,MPI_COMM_WORLD,pr,pc,major);
    cases<B,float>(h);cases<B,double>(h);
    if(h->myid==0)std::cout<<"PSTEDC PASS backend="<<int(B)<<" grid="<<pr<<"x"<<pc<<" major="<<major<<std::endl;
    ddlaDestroy(h);
}
} // namespace
int main(int argc,char** argv)
{
    MPI_Init(&argc,&argv);int size=0;MPI_Comm_size(MPI_COMM_WORLD,&size);TestOptions options;std::string error;
    if(!parse_options(argc,argv,size,options,error)){std::cerr<<error<<std::endl;MPI_Abort(MPI_COMM_WORLD,2);}
    for(auto grid:options.grids)for(char major:{'R','C'}){
        run<DdlaBackend::CPU>(grid.first,grid.second,major);
#if DDLA_HAS_GPU
        run<DdlaBackend::GPU>(grid.first,grid.second,major);
#endif
    }
    MPI_Finalize();return 0;
}
