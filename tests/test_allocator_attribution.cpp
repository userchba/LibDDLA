// Native-only attribution controls. No mocked DDLA operation or vendor result.
#include "api_grid_test_common.h"
#include "workspace_test_common.h"
#include "allocation_attribution_probe_api.h"
#include "Eigensolver/chebyshev_impl.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <vector>
using namespace ddla;
using namespace api_grid_test;
namespace {
using Z=std::complex<double>;
constexpr auto GPU=DdlaBackend::GPU;
namespace cb=ddla::spectral_impl::chebyshev;
void require_all(bool good,const char* text)
{
    int bad=!good;MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    if(bad)throw std::runtime_error(text);
}
template<class F>F load(const char* name)
{
    auto address=dlsym(RTLD_DEFAULT,name);F value=nullptr;
    static_assert(sizeof(value)==sizeof(address),"dlsym ABI");std::memcpy(&value,&address,sizeof(value));
    if(!value)throw std::runtime_error(std::string("missing preload symbol: ")+name);
    return value;
}
const char* base(const char* s){const char* p=std::strrchr(s?s:"",'/');return p?p+1:(s?s:"");}
struct Fault {
    int(*prewarm)()=load<int(*)()>("pblas_alloc_failure_prewarm");
    void(*arm)(std::size_t,unsigned)=load<void(*)(std::size_t,unsigned)>("pblas_alloc_failure_arm_host_bytes");
    void(*disarm)()=load<void(*)()>("pblas_alloc_failure_disarm");
    unsigned(*fired)()=load<unsigned(*)()>("pblas_alloc_failure_fired");
    void(*end)(AllocationProbeSnapshot*)=load<void(*)(AllocationProbeSnapshot*)>("staging_allocation_probe_end");
    void(*dump)(int,const char*)=load<void(*)(int,const char*)>("staging_allocation_probe_dump");
    Fault(){require_all(prewarm()==1,"fault preload prewarm failed");}
};
struct Vendor {
    int(*prepare)()=load<int(*)()>("staging_vendor_probe_prepare");
    void(*arm)(std::size_t)=load<void(*)(std::size_t)>("staging_vendor_probe_arm");
    unsigned(*allocations)()=load<unsigned(*)()>("staging_vendor_probe_allocations");
    unsigned(*forwards)()=load<unsigned(*)()>("staging_vendor_probe_forwards");
    Vendor(){require_all(prepare()==1,"real cuBLAS boundary was not resolved");}
};
__attribute__((noinline)) void fixture_bytes(std::size_t n)
{
    std::vector<std::byte> bytes;bytes.resize(n);
    if(n){bytes.front()=std::byte{1};bytes.back()=std::byte{2};}
    asm volatile("" : : "g"(bytes.data()),"g"(bytes.size()) : "memory");
}
void bindings(int rank,bool workspace)
{
    // This call emits the same weak byte-vector helper as the failing ELF.
    // -rdynamic makes it observable/preemptible just as in that executable.
    fixture_bytes(129);
    Dl_info allocation{},helper{};
    auto a=dlsym(RTLD_DEFAULT,"_Znwm");
    auto v=dlsym(RTLD_DEFAULT,"_ZNSt6vectorISt4byteSaIS0_EE17_M_default_appendEm");
    const bool okay=a&&v&&dladdr(a,&allocation)&&dladdr(v,&helper);
    if(okay)std::fprintf(stderr,"ALLOCATOR_BINDING rank=%d operator_new=%s byte_helper=%s byte_symbol=%s\n",
        rank,allocation.dli_fname,helper.dli_fname,helper.dli_sname?helper.dli_sname:"");
    require_all(okay&&std::strcmp(base(helper.dli_fname),"test_allocator_attribution")==0&&
        std::strcmp(base(allocation.dli_fname),workspace?"libworkspace_trace.so":"libpblas_alloc_failure.so")==0,
        "required actual allocator/helper ELF binding was not reproduced");
}
DdlaHandle_t create(int pr,int pc,char major)
{
    DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,GPU));
    DDLA_CHECK(ddlaSetCommunication(h,DdlaCommunication::HostStagedMPI));
    DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,pr,pc,major));return h;
}
void descriptor(DdlaHandle_t h,int* d,int m,int n,int mb,int nb)
{
    DDLA_CHECK(ddlaDescInit(d,h,m,n,mb,nb,0,0,
        std::max(1,num_loc(m,mb,h->myprow_,0,h->nprows_))+3));
}
void public_cold_fault(int pr,int pc,char major,Fault& fault)
{
    auto h=create(pr,pc,major);
    {
        constexpr int nev=3,nex=2,parent=45;
        int da[DDLA_DLEN_],dz[DDLA_DLEN_];
        descriptor(h,da,parent,parent,3,3);descriptor(h,dz,parent,nev+2,2,1);
        DeviceBuffer<Z>a(h,local_size(h,da)),z(h,local_size(h,dz));
        std::vector<double>w(nev,-313.);
        int n=0,target_rank=h->nprocs;std::array<std::size_t,2> bytes{};
        for(int candidate=13;candidate<=39;++candidate){
            auto p=cb::check<Z>(h,GPU,'L',candidate,nev,nex,a.ptr,da,w.data(),z.ptr,dz,2e-11,80,12,false);
            cb::Resources<Z> r(h,p);cb::prepare<GPU>(h,p,r);bytes=cb::staging_bytes<Z>(h,p);
            const auto b=bytes[0];
            bool unique=b&&b%256&&b!=p.stats_elements*sizeof(double)&&b!=2*std::size_t(p.q)*sizeof(double)&&
                b!=std::size_t(p.q)*sizeof(Z)&&b!=std::size_t(std::max(1,p.columns))*sizeof(Z)&&
                b!=std::size_t(p.q)*sizeof(double);
            int owner=unique?h->myid:h->nprocs;
            MPI_Allreduce(&owner,&target_rank,1,MPI_INT,MPI_MIN,h->comm);
            if(target_rank<h->nprocs){n=candidate;break;}
        }
        require_all(n>0,"could not identify a cold DDLA staging allocation");
        auto initial=make_local<Z>(h,da,[&](int i,int j){return i<n&&j<n?(i>=j?Z(i==j?i+1:0):Z(NAN)):Z(8157);});
        auto initial_z=make_local<Z>(h,dz,[](int,int){return Z(717);});
        upload(h,a.ptr,initial);upload(h,z.ptr,initial_z);DDLA_CHECK(ddlaSynchronize(h));
        const auto slots=h->host_buffers<std::byte>(0,0);
        require_all(!slots[0]&&!slots[1],"attribution probe requires a truly cold cache");
        const bool target=h->myid==target_rank;
        fault.arm(target?bytes[0]:0,1);
        int iterations=-11,converged=-22,info=-33;bool threw=false;
        try{pheevd_chebyshev<Z>(h,'L',n,nev,nex,a.ptr,da,w.data(),z.ptr,dz,2e-11,80,12,false,iterations,converged,info);}
        catch(const std::exception&){threw=true;}
        AllocationProbeSnapshot seen{};fault.end(&seen);const auto fired=fault.fired();fault.disarm();
        std::fprintf(stderr,"ALLOCATION_ATTRIBUTION_COLD rank=%d target_rank=%d bytes=%zu,%zu threw=%d fired=%u iterations=%d converged=%d info=%d\n",
            h->myid,target_rank,bytes[0],bytes[1],threw,fired,iterations,converged,info);
        fault.dump(h->myid,"public_chebyshev_cold");
        require_all(threw&&fired==unsigned(target),"actual DDLA cold staging fault did not fire/agreed-return");
        require_all(!target||(seen.library_with_helper&&seen.legacy_rejected_library),
            "actual DDLA stack did not reproduce the executable-helper rejection");
        const auto after=download(h,a.ptr,initial.size()),after_z=download(h,z.ptr,initial_z.size());
        bool same=iterations==-11&&converged==-22&&info==-33&&
            !std::memcmp(after.data(),initial.data(),initial.size()*sizeof(Z))&&
            !std::memcmp(after_z.data(),initial_z.data(),initial_z.size()*sizeof(Z));
        for(auto x:w)same=same&&x==-313.;
        require_all(same,"cold reservation failure mutated matrix, output or status");
        // Same-handle retry also verifies that failed reservation did not poison resources.
        pheevd_chebyshev<Z>(h,'L',n,nev,nex,a.ptr,da,w.data(),z.ptr,dz,2e-11,80,12,false,iterations,converged,info);
        bool good=info==0&&converged==nev;
        for(int j=0;j<nev;++j)good=good&&std::isfinite(w[j])&&std::abs(w[j]-(j+1.))<2e-8;
        require_all(good,"public Chebyshev retry did not recover finite known eigenvalues");
        if(!h->myid)std::puts("ALLOCATION_ATTRIBUTION_DDLA_PASS cold=1 fault=1 unchanged=1 retry=1 old_filter_rejection=1");
    }
    DDLA_CHECK(ddlaDestroy(h));
}
void fixture_control(int rank,Fault& fault,std::size_t bytes)
{
    fault.arm(bytes,1);bool threw=false;
    try{fixture_bytes(bytes);}catch(const std::bad_alloc&){threw=true;}
    AllocationProbeSnapshot seen{};fault.end(&seen);const auto fired=fault.fired();fault.disarm();
    fault.dump(rank,"fixture_byte_vector");
    require_all(!threw&&!fired&&seen.fixture&&seen.library==0&&seen.remaining_ordinal==1,
        "fixture byte-vector allocation consumed the DDLA-only fault");
    if(!rank)std::puts("ALLOCATION_ATTRIBUTION_FIXTURE_PASS fired=0 remaining_ordinal=1");
}
void vendor_controls(int pr,int pc,char major,bool workspace,Fault* fault,Vendor& vendor)
{
    auto h=create(pr,pc,major);
    {
        DeviceBuffer<double>a(h,4),b(h,4),c(h,4);
        upload(h,a.ptr,std::vector<double>{1,2,3,4});upload(h,b.ptr,std::vector<double>{1,0,0,1});
        const auto gemm=[&]{ddla::gemm<double>(h,'N','N',2,2,2,1.,a.ptr,2,b.ptr,2,0.,c.ptr,2);};
        gemm();DDLA_CHECK(ddlaSynchronize(h)); // vendor and tracer first-use outside controls
        constexpr std::size_t bytes=12347;
        if(workspace){
            require_all(workspace_test::available(),"workspace preload missing");
            workspace_test::Scope fixture_scope;fixture_bytes(bytes);
            const auto fixture=fixture_scope.finish();
            require_all(fixture.host_new>0,"workspace fixture positive control was not counted");
            if(!h->myid)std::puts("WORKSPACE_ALLOCATION_ATTRIBUTION_FIXTURE_PASS host_new_positive=1");
        }else fixture_control(h->myid,*fault,bytes);
        bool threw=false;AllocationProbeSnapshot seen{};workspace_trace_snapshot calls{};
        if(workspace){
            vendor.arm(bytes);workspace_test::Scope scope;
            try{gemm();DDLA_CHECK(ddlaSynchronize(h));}catch(const std::exception&){threw=true;}
            calls=scope.finish();
        }else{
            fault->arm(bytes,1);vendor.arm(bytes);
            try{gemm();DDLA_CHECK(ddlaSynchronize(h));}catch(const std::exception&){threw=true;}
            fault->end(&seen);const auto fired=fault->fired();fault->disarm();
            fault->dump(h->myid,"real_ddla_gemm_vendor_vector");
            require_all(!fired&&seen.other&&seen.library==0&&seen.remaining_ordinal==1,
                "vendor byte-vector allocation consumed a DDLA-ancestor fault");
        }
        require_all(!threw&&vendor.allocations()==1&&vendor.forwards()==1,
            "controlled vendor allocation or real cuBLAS forwarding did not occur");
        const auto out=download(h,c.ptr,4);bool valid=true;
        for(int i=0;i<4;++i)valid=valid&&out[i]==i+1.;
        require_all(valid,"actual forwarded cuBLAS GEMM returned wrong output");
        if(workspace)require_all(calls.host_new==0,"workspace charged vendor allocation through the executable helper");
        if(!h->myid)std::puts(workspace?
            "WORKSPACE_ALLOCATION_ATTRIBUTION_VENDOR_PASS host_new=0 real_forward=1 correct_result=1":
            "ALLOCATION_ATTRIBUTION_VENDOR_PASS fired=0 remaining_ordinal=1 real_forward=1 correct_result=1");
    }
    DDLA_CHECK(ddlaDestroy(h));
}
}
int main(int argc,char** argv)
{
    ddla_test_mpi_init(&argc,&argv);int rank=0,size=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    int pr=1,pc=size;char major='R';bool workspace=false;int status=0;
    try{
        for(int i=1;i<argc;++i){
            if(std::strcmp(argv[i],"--grid")==0&&i+1<argc){
                if(std::sscanf(argv[++i],"%dx%d",&pr,&pc)!=2)throw std::runtime_error("invalid --grid RxC");
            }else if(std::strcmp(argv[i],"--major")==0&&i+1<argc)major=argv[++i][0];
            else if(std::strcmp(argv[i],"--workspace-only")==0)workspace=true;
            else throw std::runtime_error("usage: test_allocator_attribution [--grid RxC] [--major R|C] [--workspace-only]");
        }
        require_all(pr>0&&pc>0&&pr*pc==size&&(major=='R'||major=='C'),"invalid grid or major");
        bindings(rank,workspace);Vendor vendor;
        if(workspace){vendor_controls(pr,pc,major,true,nullptr,vendor);if(!rank)std::puts("WORKSPACE_ALLOCATION_ATTRIBUTION_PASS");}
        else{
            require_all(size>1,"DDLA cold staging attribution requires a multi-rank native run");
            Fault fault;public_cold_fault(pr,pc,major,fault);vendor_controls(pr,pc,major,false,&fault,vendor);
            if(!rank)std::puts("ALLOCATION_ATTRIBUTION_PROBE_PASS");
        }
    }catch(const std::exception& e){std::fprintf(stderr,"allocator attribution rank %d: %s\n",rank,e.what());status=1;}
    int global=0;MPI_Allreduce(&status,&global,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);MPI_Finalize();return global;
}
