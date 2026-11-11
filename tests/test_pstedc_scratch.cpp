// Borrowed stage-3 CPU/GPU scratch must match the owned path without consuming
// caller padding, touching rejected spans, or taking ownership of storage.
#include "api_grid_test_common.h"
#include "Eigensolver/pstedc_impl.h"
#include "row_share.h"

#include <cstdint>
#include <complex>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>

using namespace api_grid_test;
namespace {
struct EnvGuard {
    std::string key, old;
    bool had;
    EnvGuard(const char* k, const char* value) : key(k), had(std::getenv(k)!=nullptr) {
        if (had) old=std::getenv(k);
        if (setenv(k,value,1)!=0) throw std::runtime_error("setenv failed");
    }
    ~EnvGuard() { if (had) setenv(key.c_str(),old.c_str(),1); else unsetenv(key.c_str()); }
};

enum class Mode { Owned, Adequate, Short, Overlap, Unaligned };
const char* mode_name(Mode mode) {
    switch(mode) {
    case Mode::Owned:return "owned";
    case Mode::Adequate:return "adequate";
    case Mode::Short:return "short";
    case Mode::Overlap:return "overlap";
    case Mode::Unaligned:return "unaligned";
    }
    return "invalid";
}

template<class R, bool complex_backed=false>
void check_case(const ddla::DdlaHandle_t& h,int n,int kind,int padding) {
    int P=0; MPI_Comm_size(ddlaGetCommunicator(h),&P);
    const int rank=ddlaGetRank(h);
    const ddla::detail::RowShare rows(n,P);
    const int nr=rows.count(rank),ld=std::max(nr,1),ldy=ld+padding;
    const std::size_t slab=static_cast<std::size_t>(ld)*n;
    const std::size_t count=std::max<std::size_t>(static_cast<std::size_t>(ldy)*n,1);
    constexpr std::size_t guard=64;
    const std::size_t scratch_count=complex_backed ? ((slab+2*guard+1)/2)*2 : slab+2*guard;
    const R sentinel=R(12345);
    const double eps=std::numeric_limits<R>::epsilon();
    std::vector<R> d0(n),e0(n,R(0));
    double norm=1;
    for(int i=0;i<n;++i) {
        if(kind==2) {
            // Nonzero off-diagonals and diagonal jitter exercise a connected
            // merge across the CPU's 128-column strip on multi-rank grids.
            d0[i]=R(2)+R((i*17)%23-11)/R(4096);
            if(i+1<n) e0[i]=R(-1);
        } else {
            d0[i]=kind ? R((i%4)*0.5) : R(2)+R((i*17)%23-11)/R(32);
            if(i+1<n && !kind) e0[i]=R((i*13)%17-8)/R(32);
        }
    }
    for(int i=0;i<n;++i)
        norm=std::max(norm,std::abs(double(d0[i]))+std::abs(double(e0[i]))+(i?std::abs(double(e0[i-1])):0));
    std::vector<R> reference,wr;
    for(Mode mode:{Mode::Owned,Mode::Adequate,Mode::Short,Mode::Overlap,Mode::Unaligned}) {
        DeviceBuffer<R> Y(h,count+guard);
        std::unique_ptr<DeviceBuffer<R>> storage;
        std::vector<std::complex<R>> complex_storage;
        R* storage_ptr=nullptr;
        if constexpr (complex_backed) {
            // Use actual complex objects, not just an allocation of equal bytes.
            // C++17 guarantees interleaved real component array access.
            complex_storage.resize(scratch_count/2);
            storage_ptr=reinterpret_cast<R*>(complex_storage.data());
        } else {
            storage.reset(new DeviceBuffer<R>(h,scratch_count));
            storage_ptr=storage->ptr;
        }
        upload(h,Y.ptr,std::vector<R>(count+guard,sentinel));
        upload(h,storage_ptr,std::vector<R>(scratch_count,sentinel));
        std::vector<R> d(d0),e(e0),w(n,std::numeric_limits<R>::quiet_NaN());
        R* scratch=storage_ptr+guard;
        std::size_t available=slab;
        if(mode==Mode::Short) available=slab?slab-1:0;
        if(mode==Mode::Overlap) scratch=Y.ptr+(count>1?1:0);
        if(mode==Mode::Unaligned)
            scratch=reinterpret_cast<R*>(reinterpret_cast<unsigned char*>(scratch)+1);
        if(mode==Mode::Owned)
            ddla::pstedc(h,n,d.data(),e.data(),w.data(),Y.ptr,ldy);
        else
            ddla::pstedc(h,n,d.data(),e.data(),w.data(),Y.ptr,ldy,scratch,available);
        check_ddla_sync(h);
        const auto actual=download(h,Y.ptr,count+guard);
        const auto scratch_after=download(h,storage_ptr,scratch_count);
        const std::string tag="pstedc scratch n="+std::to_string(n)+" bytes="+std::to_string(sizeof(R))+
            " kind="+std::to_string(kind)+" padding="+std::to_string(padding)+" mode="+mode_name(mode)+
            " complex_backed="+std::to_string(complex_backed);
        double bad=0,padbad=0,guardbad=0,changed=0,diff=0,wdiff=0;
        for(std::size_t i=0;i<guard;++i) {
            guardbad+=scratch_after[i]!=sentinel;
            guardbad+=scratch_after[guard+slab+i]!=sentinel;
        }
        for(std::size_t i=slab+2*guard;i<scratch_count;++i) guardbad+=scratch_after[i]!=sentinel;
        for(std::size_t i=0;i<slab;++i) changed+=scratch_after[guard+i]!=sentinel;
        const bool borrowed=mode==Mode::Adequate && nr>0;
        require_close(h,tag+" scratch guards",guardbad,0);
        require_close(h,tag+" scratch ownership",borrowed?(changed!=double(slab)):(changed!=0),0);
        // Re-upload after return, then let its owner free it: this also checks
        // that a borrowed pointer remains usable and is not released by pstedc.
        upload(h,storage_ptr,std::vector<R>(scratch_count,sentinel));
        for(int j=0;j<n;++j) {
            if(!std::isfinite(double(w[j])) || (j && w[j]<w[j-1])) ++bad;
            for(int r=0;r<nr;++r) {
                const auto k=r+static_cast<std::size_t>(j)*ldy;
                if(!std::isfinite(double(actual[k]))) ++bad;
            }
            for(int r=nr;r<ldy;++r) padbad+=actual[r+static_cast<std::size_t>(j)*ldy]!=sentinel;
        }
        if(n==0) padbad+=actual[0]!=sentinel;
        for(std::size_t i=count;i<actual.size();++i) padbad+=actual[i]!=sentinel;
        if(mode==Mode::Owned) { reference=actual;wr=w; }
        else {
            for(int j=0;j<n;++j) {
                wdiff=std::max(wdiff,std::abs(double(w[j])-double(wr[j])));
                for(int r=0;r<nr;++r) {
                    const auto k=r+static_cast<std::size_t>(j)*ldy;
                    diff=std::max(diff,std::abs(double(actual[k])-double(reference[k])));
                }
            }
        }
        require_close(h,tag+" finite sorted",bad,0);
        require_close(h,tag+" output padding",padbad,0);
        require_close(h,tag+" eigenvalues",wdiff/(std::max(n,1)*eps*norm),10);
        require_close(h,tag+" vectors",diff/(std::max(n,1)*eps),10);
        if(mode!=Mode::Adequate || n==0) continue;
        std::vector<double> full(static_cast<std::size_t>(n)*n,0);
        for(int j=0;j<n;++j) for(int r=0;r<nr;++r)
            full[rows.lo(rank)+r+static_cast<std::size_t>(j)*n]=actual[r+static_cast<std::size_t>(j)*ldy];
        MPI_Allreduce(MPI_IN_PLACE,full.data(),n*n,MPI_DOUBLE,MPI_SUM,ddlaGetCommunicator(h));
        double residual=0,orthogonal=0;
        for(int j=0;j<n;++j) for(int i=0;i<n;++i) {
            double value=double(d0[i])*full[i+static_cast<std::size_t>(j)*n];
            if(i) value+=double(e0[i-1])*full[i-1+static_cast<std::size_t>(j)*n];
            if(i+1<n) value+=double(e0[i])*full[i+1+static_cast<std::size_t>(j)*n];
            residual=std::max(residual,std::abs(value-double(w[j])*full[i+static_cast<std::size_t>(j)*n]));
            double dot=0;
            for(int r=0;r<n;++r) dot+=full[r+static_cast<std::size_t>(j)*n]*full[r+static_cast<std::size_t>(i)*n];
            orthogonal=std::max(orthogonal,std::abs(dot-(i==j?1.0:0.0)));
        }
        require_close(h,tag+" independent residual",std::isfinite(residual)?residual/(n*eps*norm):INFINITY,50);
        require_close(h,tag+" independent orthogonality",std::isfinite(orthogonal)?orthogonal/(n*eps):INFINITY,50);
    }
}
template<class R> void run(const ddla::DdlaHandle_t& h) {
    EnvGuard leaf("DDLA_EIG_DC_LEAF","16");
    for(int padding:{0,3}) {
        for(int n:{0,1,2,37,67}) check_case<R>(h,n,0,padding);
        check_case<R>(h,67,1,padding);
        if(ddla::ddlaGetBackend(h)==ddla::DdlaBackend::CPU) {
            check_case<R>(h,257,2,padding);
            check_case<R,true>(h,257,2,padding);
            for(int n:{0,1,2,37,67}) check_case<R,true>(h,n,0,padding);
            check_case<R,true>(h,67,1,padding);
        }
    }
}
}

int main(int argc,char**argv) {
    ddla_test_mpi_init(&argc,&argv);
    try {
        int size=0;MPI_Comm_size(MPI_COMM_WORLD,&size);
        TestOptions options;std::string error;
        if(!parse_options(argc,argv,size,options,error))throw std::runtime_error(error);
        auto backend=ddla::default_backend_v;
        if(const char* name=std::getenv("DDLA_TEST_BACKEND")) {
            const std::string s(name);
            if(s=="CPU"||s=="cpu")backend=ddla::DdlaBackend::CPU;
            else if(s=="GPU"||s=="gpu")backend=ddla::DdlaBackend::GPU;
            else throw std::runtime_error("DDLA_TEST_BACKEND must be CPU or GPU");
        }
        for(auto grid:options.grids) {
            ddla::DdlaHandle_t h=nullptr;
            if(ddla::ddlaInit(h,backend)!=ddla::ddlaStatus_t::DDLA_STATUS_SUCCESS)
                throw std::runtime_error("backend unavailable");
            if(ddla::ddlaSet(h,MPI_COMM_WORLD,grid.first,grid.second)!=ddla::ddlaStatus_t::DDLA_STATUS_SUCCESS)
                throw std::runtime_error("grid initialization failed");
            run<float>(h);run<double>(h);check_ddla_sync(h);ddla::ddlaDestroy(h);
        }
        int rank=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        if(rank==0)std::puts("test_pstedc_scratch passed");
    } catch(const std::exception& e) {
        std::fprintf(stderr,"test_pstedc_scratch: %s\n",e.what());MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Finalize();return 0;
}
