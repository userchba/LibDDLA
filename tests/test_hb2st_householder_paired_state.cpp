#include "api_grid_test_common.h"
#include "hh_paired_state_oracle.h"
#include <cstdio>

using namespace api_grid_test;
namespace {
template<class T> void run(const DdlaHandle_t& h) {
    const int n=129,b=8;
    int desc[DDLA_DLEN_];
    ddlaDescInit(desc,h,n,n,4,4,0,0);
    auto matrix=make_local<T>(h,desc,[&](int i,int j) {
        return hh_paired_state_oracle::value<T>(i,j,b);
    });
    for(auto stage2:{detail::Hb2stHouseholderCpuOptions::Stage2::Scalar,
                     detail::Hb2stHouseholderCpuOptions::Stage2::Blas}) {
        const auto x=hh_paired_state_oracle::run<T>(h,n,b,desc,matrix.data(),stage2);
        if(ddlaGetRank(h)==0)
            std::printf("HH_PAIRED_STATE PASS bytes=%zu stage2=%d applications=%zu inactive=%zu residual=%.9g orth=%.9g difference=%.9g\n",
                sizeof(T),static_cast<int>(stage2),x.applications,x.inactive,
                x.similarity,x.orthogonality,x.difference);
    }
}
void policy(const DdlaHandle_t& h) {
    using Options=detail::Hb2stHouseholderCpuOptions;
    Options options;
    if(ddlaGetSize(h)>1)
        options.stage4=ddlaGetRank(h)==0?Options::Stage4::Wy:Options::Stage4::Paired;
    else options.stage4=static_cast<Options::Stage4>(7);
    bool caught=false;
    try {
        detail::Hb2stHouseholderCpu<double> state(h,17,4,true,options);
        state.release();
    }catch(const std::exception&){caught=true;}
    require_close(h,"HH mismatched stage4 policy rejects",!caught,0);
}
}

int main(int argc,char** argv) {
    int provided=0;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int rank=0,P=0;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&P);
    try {
        if(provided<MPI_THREAD_FUNNELED)throw std::runtime_error("MPI thread support");
        TestOptions options;
        std::string error;
        if(!parse_options(argc,argv,P,options,error))throw std::runtime_error(error);
        for(auto grid:options.grids) {
            DdlaHandle_t h=nullptr;
            DDLA_CHECK(ddlaInit(h,DdlaBackend::CPU));
            DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,grid.first,grid.second));
            run<float>(h);run<double>(h);
            run<std::complex<float>>(h);run<std::complex<double>>(h);
            policy(h);
            DDLA_CHECK(ddlaDestroy(h));
        }
        if(rank==0)std::puts("test_hb2st_householder_paired_state passed");
    }catch(const std::exception& error) {
        std::fprintf(stderr,"rank %d: %s\n",rank,error.what());
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Finalize();
    return 0;
}
