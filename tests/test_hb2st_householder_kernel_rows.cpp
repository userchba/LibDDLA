// Consuming Householder row/column conversion and descriptor preservation.
#include "api_grid_test_common.h"
#include "Eigensolver/two_stage/hb2st_householder_cpu.h"
#include "Eigensolver/two_stage/hb2st_householder_cpu_rows.h"
#include "row_share.h"
#include <cstring>
#include <cstdio>
#include <new>
#include "allocation_probe_ddla.h"

namespace allocation_probe {
thread_local bool enabled=false,fired=false;
void inspect() {
    if(enabled) {enabled=false;fired=true;throw std::bad_alloc();}
}
}
void* operator new(std::size_t bytes) {
    allocation_probe::inspect();
    void* p=std::malloc(bytes?bytes:1);if(!p) throw std::bad_alloc();return p;
}
void* operator new[](std::size_t n) {return ::operator new(n);}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete[](void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
void operator delete[](void* p,std::size_t) noexcept {std::free(p);}

using namespace api_grid_test;
namespace {
template<class T> std::complex<long double> wide(T x) {return {static_cast<long double>(x),0};}
template<class R> std::complex<long double> wide(std::complex<R> x) {return {x.real(),x.imag()};}
template<class R> R yvalue(int row,int col) {return R((row*17+col*13)%47-23)/R(47);}

// A diagonal input generates identity band transforms. This isolates the
// materializer's row mapping, phases, lifetime and output packing from the
// separately tested Householder numerical back-transform.
template<class T>
void check_mapping(const DdlaHandle_t& h,int n,bool shifted,bool with_phases,
                   detail::Hb2stHouseholderCpuOptions::Stage4 method)
{
    using R=typename real_type<T>::type;
    int npr=0,npc=0;ddlaGetGridDims(h,npr,npc);
    const int rank=ddlaGetRank(h),P=npr*npc;
    const detail::RowShare rows(n,P);
    const int nr=rows.count(rank),ldy=std::max(nr,1)+3;
    int desc[DDLA_DLEN_];
    ddlaDescInit(desc,h,n+5,n+7,4,4,shifted?npr-1:0,shifted?npc-1:0);
    desc[DDLA_LLD_]+=5;
    const int lld=desc[DDLA_LLD_],zr=ddla_test::m_loc(h,desc),zc=ddla_test::n_loc(h,desc);
    const T sentinel=T(R(139));
    std::vector<T> matrix(std::max<std::size_t>(local_size(h,desc),1),sentinel);
    for(int c=0;c<zc;++c) for(int r=0;r<zr;++r) {
        const int gr=indx_l2g_r(desc,h,r),gc=indx_l2g_c(desc,h,c);
        if(gr<n && gc<n) matrix[r+static_cast<std::size_t>(c)*lld]=gr==gc?T(R(gr+1)):T(0);
    }
    const auto original=matrix;
    std::vector<R> d(std::max(n,1)),e(std::max(n,1));
    std::vector<T> phases;
    detail::Hb2stHouseholderCpuOptions options;options.stage4=method;
    detail::Hb2stHouseholderCpu<T> state(h,n,std::min(4,std::max(1,n-1)),true,options);
    try {
        state.reduce(matrix.data(),desc,d.data(),e.data(),phases);
        phases.clear();
        if(with_phases) for(int r=1;r<n;++r) {
            if constexpr(std::is_same<T,R>::value) phases.push_back(r%2?T(-1):T(1));
            else phases.emplace_back(R(std::cos(r*0.007)),R(std::sin(r*0.007)));
        }
        std::vector<R> input(static_cast<std::size_t>(ldy)*std::max(n,1),std::numeric_limits<R>::quiet_NaN());
        for(int c=0;c<n;++c) for(int r=0;r<nr;++r)
            input[r+static_cast<std::size_t>(c)*ldy]=yvalue<R>(rows.lo(rank)+r,c);
        std::vector<T> output(matrix.size(),sentinel);
        for(int c=0;c<zc;++c) for(int r=0;r<zr;++r)
            if(indx_l2g_r(desc,h,r)<n && indx_l2g_c(desc,h,c)<n)
                output[r+static_cast<std::size_t>(c)*lld]=T(std::numeric_limits<R>::quiet_NaN());
        DeviceBuffer<R> Y(h,input.size());DeviceBuffer<T> Z(h,output.size());
        upload(h,Y.ptr,input);upload(h,Z.ptr,output);check_ddla_sync(h);
        int info=-1;
        detail::hh_rows_cpu<T>(h,n,Y.ptr,ldy,state,phases,Z.ptr,desc,info);
        check_ddla_sync(h);
        require_close(h,"Householder rows consumed input",Y.ptr!=nullptr,0);
        require_close(h,"Householder rows info",info,0);
        const auto actual=download(h,Z.ptr,output.size());
        double error=0,padding=0,finite=0;
        for(int c=0;c<zc;++c) for(int r=0;r<lld;++r) {
            const int gr=r<zr?indx_l2g_r(desc,h,r):n,gc=indx_l2g_c(desc,h,c);
            const std::size_t at=r+static_cast<std::size_t>(c)*lld;
            if(gr<n && gc<n) {
                const auto got=wide(actual[at]);
                const auto expected=wide(yvalue<R>(gr,gc))*(gr>0 && !phases.empty()?wide(phases[gr-1]):std::complex<long double>(1,0));
                if(!std::isfinite(got.real()) || !std::isfinite(got.imag())) ++finite;
                error=std::max(error,static_cast<double>(std::abs(got-expected)));
            } else padding+=actual[at]!=sentinel;
        }
        if(zc==0) for(const auto& x:actual) padding+=x!=sentinel;
        double changed=std::memcmp(matrix.data(),original.data(),matrix.size()*sizeof(T))!=0;
        const std::string tag="Householder rows n="+std::to_string(n)+" type="+std::to_string(sizeof(T))+" shifted="+std::to_string(shifted)+" method="+std::to_string(int(method));
        require_close(h,tag+" finite",finite,0);require_close(h,tag+" padding",padding,0);
        require_close(h,tag+" input band preserved",changed,0);
        require_close(h,tag+" phase/map reference",error/std::numeric_limits<R>::epsilon(),20);
        state.release();
    } catch(...) {state.release();throw;}
}

void rejected_before_consumption(const DdlaHandle_t& h,bool allocation,bool overlap=false)
{
    constexpr int n=17;
    int npr=0,npc=0;ddlaGetGridDims(h,npr,npc);
    const int rank=ddlaGetRank(h),P=npr*npc,victim=P-1;
    const detail::RowShare rows(n,P);
    const int ld=std::max(rows.count(rank),1);
    int desc[DDLA_DLEN_];ddlaDescInit(desc,h,n,n,4,4,0,0);
    std::vector<double> matrix(std::max<std::size_t>(local_size(h,desc),1),0),phases;
    std::vector<double> d(n),e(n);
    detail::Hb2stHouseholderCpu<double> state(h,n,4,true);
    try {
        state.reduce(matrix.data(),desc,d.data(),e.data(),phases);
        DeviceBuffer<double> Y(h,static_cast<std::size_t>(ld)*n),Z(h,matrix.size());
        double* original=Y.ptr;
        if(rank==victim && allocation) {allocation_probe::fired=false;allocation_probe::enabled=true;}
        int caught=0,info=-1;
        try {detail::hh_rows_cpu<double>(h,allocation || overlap || rank!=victim?n:n-1,
            Y.ptr,ld,state,phases,overlap?Y.ptr:Z.ptr,desc,info);}
        catch(const std::exception&) {caught=1;}
        allocation_probe::enabled=false;
        int all_caught=0;
        MPI_Allreduce(&caught,&all_caught,1,MPI_INT,MPI_SUM,ddlaGetCommunicator(h));
        require_close(h,"Householder rows preparation agreement",std::abs(all_caught-P),0);
        require_close(h,"Householder rows retains unconsumed input",Y.ptr!=original,0);
        if(allocation) {
            int fired=rank==victim && allocation_probe::fired?1:0,hits=0;
            MPI_Allreduce(&fired,&hits,1,MPI_INT,MPI_SUM,ddlaGetCommunicator(h));
            require_close(h,"Householder rows injected allocation hit",std::abs(hits-1),0);
        }
        state.release();
    } catch(...) {allocation_probe::enabled=false;state.release();throw;}
}

void failed_after_consumption(const DdlaHandle_t& h,
                              detail::Hb2stHouseholderCpuOptions::Stage4 method)
{
    constexpr int n=17;
    int npr=0,npc=0;ddlaGetGridDims(h,npr,npc);
    const int rank=ddlaGetRank(h),P=npr*npc;
    const detail::RowShare rows(n,P);
    const int ld=std::max(rows.count(rank),1);
    int desc[DDLA_DLEN_];ddlaDescInit(desc,h,n,n,4,4,0,0);
    std::vector<double> matrix(std::max<std::size_t>(local_size(h,desc),1),0),phases;
    std::vector<double> d(n),e(n),input(static_cast<std::size_t>(ld)*n,0);
    detail::Hb2stHouseholderCpuOptions options;
    options.stage4=method;
    options.failure=detail::Hb2stHouseholderCpuOptions::Failure::Apply;options.failure_rank=P-1;
    detail::Hb2stHouseholderCpu<double> state(h,n,4,true,options);
    try {
        state.reduce(matrix.data(),desc,d.data(),e.data(),phases);
        DeviceBuffer<double> Y(h,input.size()),Z(h,matrix.size());upload(h,Y.ptr,input);
        int caught=0,info=-1;
        try {detail::hh_rows_cpu<double>(h,n,Y.ptr,ld,state,phases,Z.ptr,desc,info);}
        catch(const std::exception&) {caught=1;}
        int all=0;MPI_Allreduce(&caught,&all,1,MPI_INT,MPI_SUM,ddlaGetCommunicator(h));
        require_close(h,"Householder rows apply failure agreement",std::abs(all-P),0);
        require_close(h,"Householder rows consumed input on apply failure",Y.ptr!=nullptr,0);
        state.release();
    } catch(...) {state.release();throw;}
}
}

int main(int argc,char** argv)
{
    int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int rank=0,size=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    try {
        if(provided<MPI_THREAD_FUNNELED) throw std::runtime_error("MPI thread support");
        // Respect the allocation's thread policy; native validation must run
        // this actual entry with both one and multiple host workers.
        TestOptions options;std::string error;
        if(!parse_options(argc,argv,size,options,error)) throw std::runtime_error(error);
        for(const auto& grid:options.grids) {
            DdlaHandle_t h=nullptr;
            if(ddlaInit(h,DdlaBackend::CPU)!=ddlaStatus_t::DDLA_STATUS_SUCCESS ||
               ddlaSet(h,MPI_COMM_WORLD,grid.first,grid.second)!=ddlaStatus_t::DDLA_STATUS_SUCCESS)
                throw std::runtime_error("CPU handle/grid");
            using Method=detail::Hb2stHouseholderCpuOptions::Stage4;
            for(auto method:{Method::Wy,Method::Paired,Method::FmaChain,Method::FmaIndependent,
                             Method::GroupedReal4,Method::GroupedReal6}) {
#define DDLA_ROWS_CASES(T) \
            for(bool shifted:{false,true}) { \
                check_mapping<T>(h,17,shifted,true,method); \
                check_mapping<T>(h,2,shifted,false,method); \
            } \
            check_mapping<T>(h,0,true,false,method); \
            check_mapping<T>(h,1,true,false,method);
            DDLA_ROWS_CASES(float)
            DDLA_ROWS_CASES(double)
            DDLA_ROWS_CASES(std::complex<float>)
            DDLA_ROWS_CASES(std::complex<double>)
#undef DDLA_ROWS_CASES
            check_mapping<double>(h,389,true,false,method);
            check_mapping<std::complex<double>>(h,389,true,true,method);
            failed_after_consumption(h,method);
            if(rank==0)std::printf("HH_KERNEL_ROWS PASS method=%d scalar_cases=26 apply_failure=1\n",int(method));
            }
            rejected_before_consumption(h,false);rejected_before_consumption(h,true);
            rejected_before_consumption(h,false,true);
            ddlaDestroy(h);
        }
        if(rank==0) std::puts("test_hb2st_householder_kernel_rows passed");
    } catch(const std::exception& e) {
        allocation_probe::enabled=false;
        std::fprintf(stderr,"rank %d: %s\n",rank,e.what());MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Finalize();return 0;
}
