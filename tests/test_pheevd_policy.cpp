#include "api_grid_test_common.h"
#include "ddla_stream.h"
#include "Eigensolver/eig_params.h"
#include "Eigensolver/two_stage/pheevd_impl.h"
#include <cstdio>
using namespace api_grid_test;
namespace {
template<class T>int choice(char job,int n,int nb,int band,bool explicit_band,int pr=2,int pc=4,
                            char major='R',int workers=1,bool shared=true,bool legacy=false,bool device=true) {
    return detail::pheevd_gpu_householder_band<T>(job,n,nb,band,explicit_band,pr,pc,major,workers,shared,legacy,device);
}
void local_policy() {
    auto check=[](bool ok){if(!ok)throw std::runtime_error("measured pheevd policy boundary");};
    for(int n:{4096,8192,16384,32768})check(choice<double>('V',n,64,16,false)==64);
    check(choice<double>('V',4095,64,16,false)==0);
    check(choice<double>('V',32767,64,16,true)==0);
    check(choice<double>('V',32768,64,16,true)==16);
    check(choice<double>('V',8192,64,64,true)==64);
    check(choice<double>('V',32768,64,32,true)==0);
    check(choice<std::complex<double>>('V',8192,64,16,false)==0);
    check(choice<std::complex<double>>('V',16384,64,16,false)==64);
    check(choice<std::complex<double>>('V',8192,64,64,true)==64);
    check(choice<std::complex<double>>('V',32768,64,16,true)==0);
    check(choice<float>('V',32768,64,16,false)==0);
    check(choice<std::complex<float>>('V',32768,64,16,false)==0);
    check(choice<double>('N',32768,64,16,false)==0);
    check(choice<double>('V',32768,32,16,false)==0);
    check(choice<double>('V',32768,64,16,false,1,8)==0);
    check(choice<double>('V',32768,64,16,false,2,4,'C')==0);
    check(choice<double>('V',32768,64,16,false,2,4,'R',2)==0);
    check(choice<double>('V',32768,64,16,false,2,4,'R',1,false)==0);
    check(choice<double>('V',32768,64,16,false,2,4,'R',1,true,true)==0);
    check(choice<double>('V',32768,64,16,false,2,4,'R',1,true,false,false)==0);
    check(detail::pheevd_gpu_householder_reduction(16383)==DdlaBackend::CPU);
    check(detail::pheevd_gpu_householder_reduction(16384)==DdlaBackend::GPU);
}
#if DDLA_HAS_GPU
template<class T>void checked_policy(DdlaHandle_t h,int n,const char* band,char job) {
    using R=typename real_type<T>::type;
    if(band)setenv("DDLA_EIG_NBW",band,1);else unsetenv("DDLA_EIG_NBW");
    int d[DDLA_DLEN_];ddlaDescInit(d,h,n,n,64,64,0,0);
    // The alias is valid; this test checks metadata without executing the solve.
    DeviceBuffer<T> A(h,local_size(h,d));std::vector<R>w(std::max(1,n));
    auto topology=spectral_impl::phb2st_topology_prepare(h);
    auto p=spectral_impl::pheevd_check<T>(h,DdlaBackend::GPU,job,'U',n,A.ptr,d,w.data(),A.ptr,d,*topology);
    int pr=0,pc=0;ddlaGetGridDims(h,pr,pc);
    bool tuned=false;
#if defined(DDLA_USE_CUDA) && defined(DDLA_USE_CCL) && !defined(DDLA_USE_GPU_CPU_TUNNEL)
    int major=0,minor=0;
    GPU_CHECK(gpuDeviceGetAttribute(&major,gpuDevAttrComputeCapabilityMajor,h->local_device));
    GPU_CHECK(gpuDeviceGetAttribute(&minor,gpuDevAttrComputeCapabilityMinor,h->local_device));
    tuned=major==7&&minor==0;
#endif
    const int initial=eig_distributed_bandwidth(n,64,band?std::atoi(band):16,ddlaGetSize(h));
    int expected=choice<T>(job,n,64,initial,band!=nullptr,pr,pc,h->major,p.workers,topology->groups()==1,false,tuned);
    MPI_Allreduce(MPI_IN_PLACE,&expected,1,MPI_INT,MPI_MIN,ddlaGetCommunicator(h));
    const bool hh=p.engine==spectral_impl::PheevdEngine::Householder;
    const bool gpu=hh&&p.householder.options.stage2==detail::Hb2stHouseholderCpuOptions::Stage2::Gpu;
    require_close(h,"automatic policy checked band/method/placement",
        hh==(expected!=0)&&p.band==(expected?expected:initial)&&
        (!hh||gpu==(detail::pheevd_gpu_householder_reduction(n)==DdlaBackend::GPU))?0:1,0);
    auto forced=spectral_impl::pheevd_check<T>(h,DdlaBackend::GPU,job,'U',n,A.ptr,d,w.data(),A.ptr,d,*topology,'G');
    require_close(h,"explicit Givens retains requested band and host reduction",
        !forced.automatic_householder&&forced.engine==spectral_impl::PheevdEngine::ReplayGpu&&
        !forced.chase.device_chase&&forced.band==initial?0:1,0);
    // A legacy request on one rank must disable the automatic choice everywhere.
    if(ddlaGetRank(h)==0)setenv("DDLA_EIG_CHASE_TEAM_THREADS","1",1);
    auto legacy=spectral_impl::pheevd_check<T>(h,DdlaBackend::GPU,job,'U',n,A.ptr,d,w.data(),A.ptr,d,*topology);
    require_close(h,"automatic policy respects collective legacy fallback",
        !legacy.automatic_householder&&legacy.engine==spectral_impl::PheevdEngine::ReplayGpu&&legacy.band==initial?0:1,0);
    unsetenv("DDLA_EIG_CHASE_TEAM_THREADS");topology->release();
}
#endif
}
int main(int argc,char**argv) {
    ddla_test_mpi_init(&argc, &argv);
    try {
        local_policy();
#if DDLA_HAS_GPU
        int P=0;MPI_Comm_size(MPI_COMM_WORLD,&P);int pr=P==8?2:(P>=4?2:1),pc=P/pr;
        DdlaHandle_t h=nullptr;ddlaInit(h,DdlaBackend::GPU);ddlaSet(h,MPI_COMM_WORLD,pr,pc);
        unsetenv("DDLA_EIG_CPU_ACCUM");unsetenv("DDLA_EIG_CHASE_TEAM_THREADS");
        for(int n:{4096,8192,16384,32768})checked_policy<double>(h,n,nullptr,'V');
        checked_policy<double>(h,32768,"16",'V');checked_policy<double>(h,8192,"16",'V');
        checked_policy<double>(h,8192,"64",'V');checked_policy<double>(h,32768,nullptr,'N');
        checked_policy<std::complex<double>>(h,8192,nullptr,'V');checked_policy<std::complex<double>>(h,16384,nullptr,'V');
        checked_policy<std::complex<double>>(h,8192,"64",'V');checked_policy<float>(h,16384,nullptr,'V');
        unsetenv("DDLA_EIG_NBW");ddlaDestroy(h);
#endif
        std::puts("PHEEVD_POLICY_PASS");MPI_Finalize();return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"PHEEVD_POLICY_FAIL %s\n",e.what());MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
