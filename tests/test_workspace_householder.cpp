#include "api_grid_test_common.h"
#include "Eigensolver/two_stage/hb2st_householder_cpu.h"
#include "Eigensolver/two_stage/hb2st_householder_cpu_rows.h"
#include "Eigensolver/two_stage/hb2st_householder_kernels.h"
#include "workspace_test_common.h"
#include <cstring>
#include <type_traits>
using namespace api_grid_test;
using namespace ddla::detail;
static_assert(std::is_trivially_copyable_v<Hb2stHouseholderCpuPlan>);
static_assert(std::is_trivially_copyable_v<HhRowsCpuPlan>);
namespace {
void clean(DdlaHandle_t h,const workspace_test::Snapshot& s,bool query=false){
    const bool okay=!s.ddla_alloc&&!s.ddla_free&&!s.pinned_alloc&&!s.raw_gpu_alloc&&!s.host_new&&!s.vendor_query&&!s.resource_setup
        &&(!query||(!s.copies&&!s.launches&&!s.syncs&&!s.collectives));
    require_close(h,query?"Householder pure local query":"Householder raw run has no allocation/query/resource setup",okay?0:1,0);
}
// Exactly sized backend scratch (host memory on the CPU handle) between poisoned guards.
class Guarded {
    DdlaHandle_t h_;std::size_t bytes_;ddla::vector<char> storage_;static constexpr std::size_t guard=256;
public:
    Guarded(DdlaHandle_t h,std::size_t bytes):h_(h),bytes_(bytes),storage_(h,bytes+2*guard){std::memset(storage_.data(),0x6b,storage_.size());}
    char* data(){return storage_.data()+guard;}
    void verify(){const char* w=storage_.data();bool okay=true;for(std::size_t i=0;i<guard;++i)okay=okay&&w[i]==0x6b&&w[guard+bytes_+i]==0x6b;require_close(h_,"Householder exact workspace guards",okay?0:1,0);}
};
template<class T>T scalar(double r,double i=0){if constexpr(std::is_arithmetic_v<T>)return T(r);else return T(r,i);}
template<class T>T matrix(int i,int j,int b){if(std::abs(i-j)>b)return T(0);if(i>j)return hh::adj(matrix<T>(j,i,b));return i==j?scalar<T>(2+.03*i):scalar<T>(.13*std::sin(1+i+3*j),.07*std::cos(2*i+j));}
template<class T>double tolerance(){return std::is_same_v<typename real_type<T>::type,float>?5e-5:8e-12;}
template<class T>void same(DdlaHandle_t h,const std::vector<T>& a,const std::vector<T>& b){double error=0;for(std::size_t i=0;i<a.size();++i)error=std::max(error,workspace_test::relative_error(b[i],a[i]));require_close(h,"checked/raw Householder results",error,tolerance<T>());}

template<class T>void case_run(DdlaHandle_t h,int n,Hb2stHouseholderCpuOptions::Stage4 method)
{
    using R=typename real_type<T>::type;const int b=std::min(4,std::max(n-1,1));
    int desc[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(desc,h,n+3,n+5,3,4,h->nprows_-1,h->npcols_-1));desc[DDLA_LLD_]+=3;
    auto input=make_local<T>(h,desc,[&](int i,int j){return i<n&&j<n&&i<=j&&j-i<=b?matrix<T>(i,j,b):scalar<T>(-731);});
    DeviceBuffer<T>A(h,input.size());upload(h,A.ptr,input);
    Hb2stHouseholderCpuOptions options;options.stage4=method;
    std::vector<R>d(std::max(n,1)),e(std::max(n,1)),rd=d,re=e;
    std::vector<T> phases(std::is_same_v<T,R>?0:std::size_t(std::max(n-1,0))),reference_phases;
    std::vector<T> ref_columns;
    const RowShare rows(n,h->nprocs);const int ncols=rows.count(h->myid),first=rows.lo(h->myid),ld=std::max(n,1)+3;
    const T sentinel=scalar<T>(-913);std::vector<T> columns(std::max(std::size_t(1),std::size_t(ld)*ncols),sentinel);
    {
        Hb2stHouseholderCpu<T> reference(h,n,b,true,options);
        try{reference.reduce(A.ptr,desc,rd.data(),re.data(),reference_phases);
            for(int j=0;j<ncols;++j)for(int i=0;i<n;++i)columns[i+std::size_t(j)*ld]=i==first+j?(i&& !reference_phases.empty()?reference_phases[i-1]:T(1)):T(0);
            ref_columns=columns;reference.apply_columns(ref_columns.data(),ld,ncols);reference.release();
        }catch(...){reference.release();throw;}
    }
    const auto p=hh_cpu_check(h,n,b,true,options);
    MPI_Barrier(h->comm);bool pure=true;
    if(!h->myid){workspace_test::Scope scope(true);std::size_t bytes=0;hh_cpu_reduce_workspace_size<T>(h,p.band,p.chase_threads,&bytes);const auto s=scope.finish();pure=!s.ddla_alloc&&!s.ddla_free&&!s.pinned_alloc&&!s.raw_gpu_alloc&&!s.host_new&&!s.vendor_query&&!s.copies&&!s.launches&&!s.syncs&&!s.collectives&&!s.resource_setup;}
    MPI_Barrier(h->comm);require_close(h,"one-rank Householder query completes",pure?0:1,0);
    std::size_t reduce_bytes=0,apply_bytes=0;
    workspace_test::Scope query(true);hh_cpu_reduce_workspace_size<T>(h,p.band,p.chase_threads,&reduce_bytes);hh_cpu_apply_columns_workspace_size<T>(h,ncols,&apply_bytes);clean(h,query.finish(),true);
    Guarded reduce_work(h,reduce_bytes);Hb2stHouseholderCpu<T> state(h,p);
    try{
        Guarded apply_work(h,apply_bytes);
        for(int pass=0;pass<2;++pass){
            workspace_test::Scope reduce_scope;state.reduce_run(A.ptr,desc,d.data(),e.data(),phases.data(),reduce_work.data(),reduce_bytes);clean(h,reduce_scope.finish());
            same(h,rd,d);same(h,re,e);same(h,reference_phases,phases);
            // Only the state's own storage and shared mappings survive stage2.
            // The reduction scratch is dead and may serve another scalar
            // family before the next raw reduce or any apply.
            std::memset(reduce_work.data(),0xa7,reduce_bytes);
            auto result=columns;workspace_test::Scope apply_scope;state.apply_columns_run(result.data(),ld,ncols,apply_work.data(),apply_bytes);clean(h,apply_scope.finish());same(h,ref_columns,result);
            reduce_work.verify();apply_work.verify();
        }
        same(h,input,download(h,A.ptr,input.size()));
        const int yld=std::max(rows.count(h->myid),1)+3;
        std::vector<R> Y(std::size_t(yld)*n,std::numeric_limits<R>::quiet_NaN());
        for(int j=0;j<n;++j)for(int i=0;i<rows.count(h->myid);++i)Y[i+std::size_t(j)*yld]=R(.01*(rows.lo(h->myid)+i)+.02*j);
        DeviceBuffer<R> yr(h,Y.size());upload(h,yr.ptr,Y);
        DeviceBuffer<T> zr(h,input.size()),z(h,input.size());std::vector<T> output(input.size(),sentinel);upload(h,zr.ptr,output);
        int info=0;hh_rows_cpu<T>(h,n,yr.ptr,yld,state,phases,zr.ptr,desc,info);const auto expected=download(h,zr.ptr,output.size());
        const auto rp=hh_rows_cpu_check<T>(h,n,Y.data(),yld,state,phases.data(),phases.size(),z.ptr,desc);
        std::size_t row_bytes=0;workspace_test::Scope row_query(true);hh_rows_cpu_workspace_size<T>(h,rp,&row_bytes);clean(h,row_query.finish(),true);Guarded row_work(h,row_bytes);
        const auto original=Y;
        for(int pass=0;pass<2;++pass){upload(h,z.ptr,output);workspace_test::Scope scope;hh_rows_cpu_run<T>(h,rp,Y.data(),state,phases.data(),z.ptr,info,row_work.data(),row_bytes);clean(h,scope.finish());
            same(h,expected,download(h,z.ptr,output.size()));require_close(h,"Householder rows run preserves input including NaN padding",Y.size()==original.size()&&(Y.empty()||std::memcmp(Y.data(),original.data(),Y.size()*sizeof(R))==0)?0:1,0);row_work.verify();}
        state.release();reduce_work.verify();
    }catch(...){state.release();throw;}
}
void run(MPI_Comm comm,int pr,int pc,char major){
    DdlaHandle_t h=nullptr;ddlaInit(h,DdlaBackend::CPU);ddlaSet(h,comm,pr,pc,major);
#define CASE(T) case_run<T>(h,17,Hb2stHouseholderCpuOptions::Stage4::Wy);case_run<T>(h,17,Hb2stHouseholderCpuOptions::Stage4::Paired);case_run<T>(h,1,Hb2stHouseholderCpuOptions::Stage4::Paired)
    CASE(float);CASE(double);CASE(std::complex<float>);CASE(std::complex<double>);
#undef CASE
    case_run<double>(h,17,Hb2stHouseholderCpuOptions::Stage4::GroupedReal4);case_run<std::complex<double>>(h,17,Hb2stHouseholderCpuOptions::Stage4::FmaIndependent);
    if(!h->myid)std::cout<<"workspace Householder PASS grid="<<pr<<"x"<<pc<<" major="<<major<<std::endl;ddlaDestroy(h);
}
}
int main(int argc,char** argv){ddla_test_mpi_init(&argc, &argv);int size;MPI_Comm_size(MPI_COMM_WORLD,&size);TestOptions opts;std::string error;if(!parse_options(argc,argv,size,opts,error)){std::cerr<<error<<std::endl;MPI_Abort(MPI_COMM_WORLD,2);}for(auto grid:opts.grids)for(char major:{'R','C'}){
    run(MPI_COMM_WORLD,grid.first,grid.second,major);
}MPI_Finalize();return 0;}
