#include "mpi_test_init.h"
#include "host_parallel.h"
#include "ddla_connector.h"
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#if DDLA_HAS_GPU
#include "gpu/gpu.h"
#endif

using namespace ddla;
namespace {
void require(bool good,const char* what)
{
    if(!good) throw std::runtime_error(what);
}
void run(DdlaBackend backend)
{
    DdlaHandle_t h=nullptr;
    DDLA_CHECK(ddlaInit(h,backend));
    DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,1,1));
#if DDLA_HAS_GPU
    if(backend==DdlaBackend::GPU) {
        gpuStream_t alternate=nullptr;
        GPU_CHECK(gpuStreamCreateWithFlags(&alternate,gpuStreamNonBlocking));
        DDLA_CHECK(ddlaSetStream(h,alternate)); // The handle now owns it.
    }
#endif
    require(ddlaMemset(h,nullptr,7,0)==ddlaStatus_t::DDLA_STATUS_SUCCESS,"empty memset");
    require(ddlaMemset(h,nullptr,0,1)!=ddlaStatus_t::DDLA_STATUS_SUCCESS,"null active memset");
    const std::size_t sizes[]={0,1,7,63,256,4095,4096,4097,(1u<<20)-1,(1u<<20)+1,
        (2u<<20)-1,(2u<<20)+1,(8u<<20)+3,(32u<<20)+7};
    int cases=0;
    for(std::size_t bytes:sizes) for(std::size_t offset:{1u,7u,64u}) for(int value:{0,0xa5,0x1ff,-1}) {
        const std::size_t prefix=64+offset,total=prefix+bytes+64;
        std::vector<unsigned char> initial(total,0x3d),expected=initial,actual(total);
        std::memset(expected.data()+prefix,value,bytes);
        ddla::vector<unsigned char> data(h,total);
        DDLA_CHECK(ddlaMemcpy(h,data.data(),initial.data(),total));
        DDLA_CHECK(ddlaMemset(h,data.data()+prefix,value,bytes));
        DDLA_CHECK(ddlaMemcpy(h,actual.data(),data.data(),total));
        DDLA_CHECK(ddlaSynchronize(h));
        require(actual==expected,"memset byte value, alignment or guard mismatch");
        ++cases;
    }
    if(backend==DdlaBackend::CPU) {
        constexpr std::size_t half=8u<<20,guard=64;
        std::vector<unsigned char> buffer(2*half+2*guard,0x3d);
        detail::host_parallel_for(2,1,2,[&](long long first,long long last) {
            for(long long part=first;part<last;++part)
                DDLA_CHECK(ddlaMemset(h,buffer.data()+guard+std::size_t(part)*half,0xa5,half));
        });
        require(std::all_of(buffer.begin(),buffer.begin()+guard,[](unsigned char v){return v==0x3d;}),"nested prefix guard");
        require(std::all_of(buffer.begin()+guard,buffer.end()-guard,[](unsigned char v){return v==0xa5;}),"nested parallel memset");
        require(std::all_of(buffer.end()-guard,buffer.end(),[](unsigned char v){return v==0x3d;}),"nested suffix guard");
    }
    DDLA_CHECK(ddlaDestroy(h));
    std::printf("MEMSET_PASS backend=%s cases=%d\n",backend==DdlaBackend::CPU?"CPU":"GPU",cases);
}
}
int main(int argc,char** argv)
{
    ddla_test_mpi_init(&argc,&argv);
    int ranks=0;MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    if(ranks!=1){MPI_Finalize();return 77;}
    try {
        require(ddlaMemset(nullptr,nullptr,0,0)!=ddlaStatus_t::DDLA_STATUS_SUCCESS,"null handle");
        run(DdlaBackend::CPU);
#if DDLA_HAS_GPU
        run(DdlaBackend::GPU);
#endif
    } catch(const std::exception& e) {
        std::fprintf(stderr,"%s\n",e.what());MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Finalize();
}
