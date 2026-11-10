#include "api_grid_test_common.h"
#include "hh_kernel_state_oracle.h"
#include "Eigensolver/two_stage/hb2st_householder_paired_fma.h"
#include <cstdio>

using namespace api_grid_test;
namespace {
using Options=detail::Hb2stHouseholderCpuOptions;
using Method=Options::Stage4;
constexpr Method methods[]={Method::Wy,Method::Paired,Method::FmaChain,
    Method::FmaIndependent,Method::GroupedReal4,Method::GroupedReal6};

template<class T> void run(const DdlaHandle_t& h,int n,int b) {
    int panels=0;
    for(int j=0;j<detail::hh::sweep_count(n,b);++j)
        panels+=detail::hh::panel_count(detail::hh::sweep_length(n,b,j),32);
    if(n==129&&panels<=32)throw std::runtime_error("kernel fixture must reuse a partial factor batch");
    int desc[DDLA_DLEN_];
    ddlaDescInit(desc,h,n,n,4,4,0,0);
    auto matrix=make_local<T>(h,desc,[&](int i,int j) {
        return hh_kernel_state_oracle::value<T>(i,j,b);
    });
    std::vector<unsigned char>strict,wy_tiles;
    for(auto method:methods) {
        const auto x=hh_kernel_state_oracle::run<T>(h,n,b,desc,matrix.data(),
            Options::Stage2::Blas,method);
        if(method==Method::Wy)wy_tiles=x.first_tiled_output;
        if(method==Method::Paired) {
            strict=x.first_tiled_output;
            require_close(h,"direct Wy tile call preserves strict contract",strict!=wy_tiles,0);
        }
        if(method!=Method::Wy&&method!=Method::Paired) {
            int native=detail::hhpair_fma::supported()?1:0,any_native=0;
            MPI_CHECK(MPI_Allreduce(&native,&any_native,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h)));
            constexpr bool real=detail::hhpair::is_real<T>;
            const bool grouped=method==Method::GroupedReal4||method==Method::GroupedReal6;
            if(!any_native||(grouped&&!real))
                require_close(h,"kernel fallback is bitwise strict",x.first_tiled_output!=strict,0);
            else if(n==129)
                // This nontrivial fixture must actually enter changed native
                // arithmetic. No particular changed bit/hash is prescribed.
                require_close(h,"native kernel routing changes some rounding",x.first_tiled_output==strict,0);
        }
        if(ddlaGetRank(h)==0)
            std::printf("HH_KERNEL_STATE PASS bytes=%zu planes=%d n=%d band=%d panels=%d method=%d applications=%zu inactive=%zu residual=%.9g orth=%.9g difference=%.9g\n",
                sizeof(T),detail::hhpair::planes<T>,n,b,panels,static_cast<int>(method),x.applications,x.inactive,
                x.similarity,x.orthogonality,x.difference);
    }
}

// Factors fails before factor allocation; Packet fails after an owner has
// written strict records; Apply fails before numeric callbacks. Every path
// must agree across ranks and allow safe release. Pre-publication failures
// leave all tiles untouched. Apply can update active entries on non-victims
// before the final agreement; their inactive entries must remain untouched.
template<class T> void failures(const DdlaHandle_t& h) {
    using R=typename real_type<T>::type;
    constexpr int n=65,b=8,width=8;
    int desc[DDLA_DLEN_];ddlaDescInit(desc,h,n,n,4,4,0,0);
    auto matrix=make_local<T>(h,desc,[&](int i,int j) {
        return hh_kernel_state_oracle::value<T>(i,j,b);
    });
    const int P=ddlaGetSize(h),rank=ddlaGetRank(h),ncols=n/P+(rank<n%P);
    std::size_t elements=0;
    if(!detail::hhpair::layout_elements(n,ncols,width,detail::hhpair::planes<T>,elements))
        throw std::runtime_error("failure tile geometry");
    for(auto method:methods) for(auto point:{Options::Failure::Factors,
        Options::Failure::Packet,Options::Failure::Apply}) {
        Options options;options.stage4=method;options.failure=point;
        // All tested grids have <=6 ranks; rank1 owns a first-batch slot.
        options.failure_rank=P>1?1:0;
        detail::Hb2stHouseholderCpu<T> state(h,n,b,true,options);
        std::vector<R>d(n),e(n);std::vector<T>phases;
        state.reduce(matrix.data(),desc,d.data(),e.data(),phases);
        std::vector<R> tiles(std::max<std::size_t>(1,elements),R(137));
        const auto before=tiles;
        bool caught=false;
        try {state.apply_tiles(tiles.data(),width,ncols);}
        catch(const std::exception&){caught=true;}
        require_close(h,"kernel preparation/apply failure agreed",!caught,0);
        bool changed_inactive=false;
        if(point!=Options::Failure::Apply||rank==options.failure_rank) {
            changed_inactive=std::memcmp(tiles.data(),before.data(),tiles.size()*sizeof(R))!=0;
        } else {
            const int nt=ncols/width+(ncols%width!=0);
            const std::size_t stride=static_cast<std::size_t>(n)*width*detail::hhpair::planes<T>;
            for(int t=0;t<nt;++t) {
                const int active=std::min(width,ncols-t*width);
                for(int row=0;row<n;++row)for(int plane=0;plane<detail::hhpair::planes<T>;++plane)
                    for(int column=active;column<width;++column) {
                        const std::size_t index=static_cast<std::size_t>(t)*stride+
                            static_cast<std::size_t>(row)*width*detail::hhpair::planes<T>+
                            static_cast<std::size_t>(plane)*width+column;
                        changed_inactive=changed_inactive||
                            std::memcmp(&tiles[index],&before[index],sizeof(R))!=0;
                    }
            }
            if(ncols==0)
                changed_inactive=std::memcmp(tiles.data(),before.data(),tiles.size()*sizeof(R))!=0;
        }
        require_close(h,"kernel failure preserves unconsumed/inactive tile entries",changed_inactive,0);
        state.release();state.release();
        if(rank==0)std::printf("HH_KERNEL_FAILURE PASS bytes=%zu method=%d failure=%d\n",
            sizeof(T),static_cast<int>(method),static_cast<int>(point));
    }
}

void policy(const DdlaHandle_t& h) {
    // Every pair of distinct valid methods must reject before allocation.
    if(ddlaGetSize(h)>1) {
        for(auto method:methods) {
            Options options;options.stage4=ddlaGetRank(h)==0?method:
                (method==Method::GroupedReal6?Method::Wy:Method(int(method)+1));
            bool caught=false;
            try {detail::Hb2stHouseholderCpu<double> state(h,17,4,true,options);state.release();}
            catch(const std::exception&){caught=true;}
            require_close(h,"mismatched valid kernel policy rejects",!caught,0);
        }
    }
    for(int invalid:{-1,6,7,127}) {
        Options options;options.stage4=static_cast<Method>(invalid);
        bool caught=false;
        try {detail::Hb2stHouseholderCpu<double> state(h,17,4,true,options);state.release();}
        catch(const std::exception&){caught=true;}
        require_close(h,"invalid kernel policy rejects",!caught,0);
        if(ddlaGetSize(h)>1) {
            options.stage4=ddlaGetRank(h)==ddlaGetSize(h)-1?static_cast<Method>(invalid):Method::Paired;
            caught=false;
            try {detail::Hb2stHouseholderCpu<double> state(h,17,4,true,options);state.release();}
            catch(const std::exception&){caught=true;}
            require_close(h,"one-rank invalid kernel policy rejects collectively",!caught,0);
        }
    }
}
}

int main(int argc,char** argv) {
    int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int rank=0,P=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&P);
    try {
        if(provided<MPI_THREAD_FUNNELED)throw std::runtime_error("MPI thread support");
        TestOptions options;std::string error;
        if(!parse_options(argc,argv,P,options,error))throw std::runtime_error(error);
        for(auto grid:options.grids) {
            DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,DdlaBackend::CPU));
            DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,grid.first,grid.second));
#define HH_CASES(T) \
            for(int n:{0,1,2})run<T>(h,n,1); \
            run<T>(h,129,8);failures<T>(h);
            HH_CASES(float)
            HH_CASES(double)
            HH_CASES(std::complex<float>)
            HH_CASES(std::complex<double>)
#undef HH_CASES
            policy(h);DDLA_CHECK(ddlaDestroy(h));
        }
        if(rank==0)std::puts("test_hb2st_householder_kernel_state passed");
    }catch(const std::exception& error) {
        std::fprintf(stderr,"rank %d: %s\n",rank,error.what());MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Finalize();return 0;
}
