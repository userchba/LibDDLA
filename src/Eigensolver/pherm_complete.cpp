// ---------------------------------------------------------------------------
// pherm_complete: build the full Hermitian matrix from ONE stored triangle.
//
// ScaLAPACK contract for uplo = 'U' / 'L': only that triangle of the leading
// n x n block is referenced (the opposite triangle and the imaginary part of
// the diagonal are ignored).  The two-stage eigensolver's first stage needs
// the FULL matrix (ELPA's bandred requires it too), so pheevd calls this
// first.  The pass is in place and distributed:
//
//   * every STRICTLY stored block (br, bc) is conjugate-transposed into the
//     opposite block (bc, br).  Source blocks and destination blocks are
//     disjoint sets, so no scratch copy of the matrix exists; the only
//     buffers are the send/receive packs, about n^2 / (2 P) elements each;
//   * blocks whose source and destination are the same rank are copied
//     locally; the rest travel in ONE grid alltoallv (same plan ordering as
//     ptran: blocks sorted by peer rank, then by source block origin, on both
//     sides);
//   * diagonal blocks always live on one rank: their stored triangle is
//     mirrored locally and the diagonal is made real.
//
// Memory and traffic are O(n^2 / P) per rank (the previous allreduce-based
// symmetrization held two n^2 buffers on every rank and reduced n^2 elements).
// Requires mb == nb and RSRC = CSRC = 0 (the eigensolver's standing contract).
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>

#include "ddla_connector.h"
#include "check_agree.h"
#include "helpers.h"
#include "comm_traits.h"
#include "ddla_desc.h"
#include "host_parallel.h"
#include "pherm_complete_workspace.h"
#include "eig_workspace.h"
#include "plan_util.h"

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <climits>
#include <complex>
#include <cstddef>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ddla {

namespace {

// One rectangular piece of work: dst(r, c), 0 <= r < rows, 0 <= c < cols.
//   transpose kernels: dst(r, c) = conj(src[src_off + c + r * src_ld])
//   copy kernels:      dst(r, c) =      src[src_off + r + c * src_ld]
using HermBlk=spectral_impl::HermitianBlock;
struct HermPlan {
    HermBlk *pack=nullptr,*local=nullptr,*unpack=nullptr,*diag=nullptr;
    std::size_t npack=0,nlocal=0,nunpack=0,ndiag=0;
    int *sendcounts=nullptr,*recvcounts=nullptr,*sdispls=nullptr,*rdispls=nullptr;
    std::size_t send_total=0,recv_total=0;
};

void pherm_agree_failure(const DdlaHandle_t& handle, const std::exception_ptr& failure)
{
    int failed = failure ? 1 : 0;
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX,
                            ddlaGetCommunicator(handle)));
    if (failed) {
        if (failure) std::rethrow_exception(failure);
        throw std::runtime_error("pherm_complete: preparation failed on another rank");
    }
}

int pherm_checked_count(long long count)
{
    if (count < 0 || count > INT_MAX)
        throw std::overflow_error("pherm_complete: MPI count exceeds int");
    return static_cast<int>(count);
}

// meta holds pack_count + local_count + unpack_count + diag_count descriptors, counts six
// arrays of Pr * Pc ints.
HermPlan pherm_build_plan(const DdlaHandle_t& handle,const spectral_impl::HermitianPlan& p,
    std::size_t pack_count, std::size_t local_count, std::size_t unpack_count, std::size_t diag_count, std::size_t send_total_count, std::size_t recv_total_count,HermBlk* meta,int* counts)
{
    const int Pr=p.pr,Pc=p.pc,myprow=p.mr,mypcol=p.mc,n=p.n,nb=p.nb,lld=p.lld;
    const int nprocs=Pr*Pc,nblk=(n-1)/nb+1;
    HermPlan plan;
    plan.pack=meta;plan.local=meta+pack_count;plan.unpack=meta+pack_count+local_count;
    plan.diag=meta+pack_count+local_count+unpack_count;
    plan.npack=pack_count;plan.nlocal=local_count;plan.nunpack=unpack_count;plan.ndiag=diag_count;
    plan.sendcounts=counts;plan.recvcounts=counts+nprocs;
    plan.sdispls=counts+2*nprocs;plan.rdispls=counts+3*nprocs;
    auto* scursor=counts+4*nprocs;auto* rcursor=counts+5*nprocs;
    std::fill_n(counts,6*nprocs,0);
    // Count once, then prefix the per-peer block slots. The second global
    // block traversal has the same stable peer/source-origin ordering as
    // the former sorted Item vectors, without allocation or sorting.
    for(int br=0;br<nblk;++br)for(int bc=0;bc<nblk;++bc) {
        if(p.upper?bc<=br:bc>=br)continue;
        const bool src=br%Pr==myprow&&bc%Pc==mypcol,dst=bc%Pr==myprow&&br%Pc==mypcol;
        if(src==dst)continue;
        const int peer=src?handle->rc_to_rank(bc%Pr,br%Pc):handle->rc_to_rank(br%Pr,bc%Pc);
        const int elements=std::min(nb,n-br*nb)*std::min(nb,n-bc*nb);
        if(src){plan.sendcounts[peer]+=elements;++scursor[peer];}
        else{plan.recvcounts[peer]+=elements;++rcursor[peer];}
    }
    int sb=0,rb=0,so=0,ro=0;
    for(int peer=0;peer<nprocs;++peer) {
        const int s=scursor[peer],r=rcursor[peer];scursor[peer]=sb;rcursor[peer]=rb;sb+=s;rb+=r;
        plan.sdispls[peer]=so;plan.rdispls[peer]=ro;so+=plan.sendcounts[peer];ro+=plan.recvcounts[peer];
    }
    // Reuse the counts as per-peer packing cursors, restore their message
    // lengths after all descriptors have been materialized.
    std::copy_n(plan.sdispls,nprocs,plan.sendcounts);
    std::copy_n(plan.rdispls,nprocs,plan.recvcounts);
    std::size_t local=0;
    for(int br=0;br<nblk;++br)for(int bc=0;bc<nblk;++bc) {
        if(p.upper?bc<=br:bc>=br)continue;
        const bool src=br%Pr==myprow&&bc%Pc==mypcol,dst=bc%Pr==myprow&&br%Pc==mypcol;
        if(!src&&!dst)continue;
        const int gr=br*nb,gc=bc*nb,bm=std::min(nb,n-gr),bn=std::min(nb,n-gc);
        const long long as=static_cast<long long>(indxg2l(gr,nb,Pr))+static_cast<long long>(indxg2l(gc,nb,Pc))*lld;
        const long long ad=static_cast<long long>(indxg2l(gc,nb,Pr))+static_cast<long long>(indxg2l(gr,nb,Pc))*lld;
        if(src&&dst)plan.local[local++]={as,ad,lld,lld,bn,bm};
        else if(src) {
            const int peer=handle->rc_to_rank(bc%Pr,br%Pc);
            plan.pack[scursor[peer]++]={as,plan.sendcounts[peer],lld,bn,bn,bm};
            plan.sendcounts[peer]+=bm*bn;
        } else {
            const int peer=handle->rc_to_rank(br%Pr,bc%Pc);
            plan.unpack[rcursor[peer]++]={plan.recvcounts[peer],ad,bn,lld,bn,bm};
            plan.recvcounts[peer]+=bm*bn;
        }
    }
    for(int peer=0;peer<nprocs;++peer) {
        plan.sendcounts[peer]-=plan.sdispls[peer];plan.recvcounts[peer]-=plan.rdispls[peer];
    }
    std::size_t diagonal=0;
    for(int b=0;b<nblk;++b)if(b%Pr==myprow&&b%Pc==mypcol) {
        const int global=b*nb,width=std::min(nb,n-global);
        const long long offset=static_cast<long long>(indxg2l(global,nb,Pr))+static_cast<long long>(indxg2l(global,nb,Pc))*lld;
        plan.diag[diagonal++]={offset,offset,lld,lld,width,width};
    }
    plan.send_total=send_total_count;plan.recv_total=recv_total_count;return plan;
}

// The GPU packs live in the caller's scratch when it holds both.
bool pherm_borrows_scratch(const spectral_impl::HermitianPlan& p,std::size_t send_total_count,std::size_t recv_total_count,
    std::size_t scratch_elements)
{
    return p.backend==DdlaBackend::GPU&&scratch_elements>=send_total_count&&scratch_elements-send_total_count>=recv_total_count;
}

// ----- host helpers (CPU backend) -------------------------------------------

#if DDLA_HAS_CPU

template <typename U> inline U pherm_conj(U v) { return v; }
inline std::complex<float> pherm_conj(std::complex<float> v) { return std::conj(v); }
inline std::complex<double> pherm_conj(std::complex<double> v) { return std::conj(v); }

template <typename U> inline void pherm_make_real(U&) {}
template <typename R> inline void pherm_make_real(std::complex<R>& v) { v = std::complex<R>(v.real(), R(0)); }

// f(i) for every block i of v; blocks write disjoint regions of the destination
template <typename F>
void pherm_for_blocks(const HermBlk* v, std::size_t count, int nth, const F& f)
{
    if (count == 0) return;
    detail::host_parallel_for(static_cast<long long>(count), 1, nth,
                              [&](long long a, long long b) {
                                  for (long long i = a; i < b; ++i) f(v[static_cast<std::size_t>(i)]);
                              });
}

template <typename T>
void pherm_exec_cpu(const DdlaHandle_t& handle,const HermPlan& p,bool upper,T* A,int nprocs,
    int nth,T* sendbuf,T* recvbuf)
{
    std::exception_ptr failure;
    auto xpose = [](const T* src, T* dst) {
        return [src, dst](const HermBlk& b) {
            for (int c = 0; c < b.cols; ++c)
                for (int r = 0; r < b.rows; ++r)
                    dst[b.dst_off + r + static_cast<long long>(c) * b.dst_ld] =
                        pherm_conj(src[b.src_off + c + static_cast<long long>(r) * b.src_ld]);
        };
    };
    try {
        pherm_for_blocks(p.pack, p.npack, nth, xpose(A, sendbuf));
        pherm_for_blocks(p.local, p.nlocal, nth, xpose(A, A));   // disjoint source and destination blocks
    } catch (...) { failure = std::current_exception(); }
    pherm_agree_failure(handle, failure);

    // grid-wide collective: every rank calls it, including ranks with nothing to exchange
    commAlltoallv<DdlaBackend::CPU>(handle, CommScope::Grid, nprocs,
                                    static_cast<const T*>(sendbuf), p.sendcounts, p.sdispls,
                                    recvbuf, p.recvcounts, p.rdispls);

    pherm_for_blocks(p.unpack, p.nunpack, nth, [recvbuf, A](const HermBlk& b) {
        for (int c = 0; c < b.cols; ++c)
            for (int r = 0; r < b.rows; ++r)
                A[b.dst_off + r + static_cast<long long>(c) * b.dst_ld] =
                    recvbuf[b.src_off + r + static_cast<long long>(c) * b.src_ld];
    });
    pherm_for_blocks(p.diag, p.ndiag, nth, [A, upper](const HermBlk& b) {
        T* a = A + b.src_off;
        for (int c = 0; c < b.cols; ++c) {
            pherm_make_real(a[c + static_cast<long long>(c) * b.src_ld]);
            // the stored triangle is read, the opposite one is written
            for (int r = upper ? c + 1 : 0; r < (upper ? b.rows : c); ++r)
                a[r + static_cast<long long>(c) * b.src_ld] =
                    pherm_conj(a[c + static_cast<long long>(r) * b.src_ld]);
        }
    });

}

#endif // DDLA_HAS_CPU

// ----- device kernels (GPU backend) -----------------------------------------

#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)

template <typename U>
__device__ __forceinline__ void pherm_dmake_real(U&) {}
template <typename R>
__device__ __forceinline__ void pherm_dmake_real(thrust::complex<R>& v) { v = thrust::complex<R>(v.real(), R(0)); }

// one thread block per HermBlk; src and dst may be the same array (the regions are disjoint)
template <typename U>
__global__ void pherm_xpose_kernel(const U* src, U* dst, const HermBlk* blks)
{
    const HermBlk b = blks[blockIdx.x];
    for (int c = threadIdx.y; c < b.cols; c += blockDim.y)
        for (int r = threadIdx.x; r < b.rows; r += blockDim.x)
            dst[b.dst_off + r + static_cast<long long>(c) * b.dst_ld] =
                detail::device_conj(src[b.src_off + c + static_cast<long long>(r) * b.src_ld]);
}

template <typename U>
__global__ void pherm_copy_kernel(const U* src, U* dst, const HermBlk* blks)
{
    const HermBlk b = blks[blockIdx.x];
    for (int c = threadIdx.y; c < b.cols; c += blockDim.y)
        for (int r = threadIdx.x; r < b.rows; r += blockDim.x)
            dst[b.dst_off + r + static_cast<long long>(c) * b.dst_ld] =
                src[b.src_off + r + static_cast<long long>(c) * b.src_ld];
}

template <typename U>
__global__ void pherm_diag_kernel(U* a_base, const HermBlk* blks, bool upper)
{
    const HermBlk b = blks[blockIdx.x];
    U* a = a_base + b.src_off;
    for (int c = threadIdx.y; c < b.cols; c += blockDim.y) {
        for (int r = threadIdx.x; r < b.rows; r += blockDim.x) {
            const long long lo = r + static_cast<long long>(c) * b.src_ld;
            if (r == c)
                pherm_dmake_real(a[lo]);
            else if (upper ? r > c : r < c)
                a[lo] = detail::device_conj(a[c + static_cast<long long>(r) * b.src_ld]);
        }
    }
}

template <typename T>
void pherm_exec_gpu(const DdlaHandle_t& handle,const HermPlan& p,bool upper,T* A,int nprocs,
    HermBlk* d_meta,HermBlk* meta,T* d_send,T* d_recv)
{
    using devT=detail::DeviceScalarT<T>;
    const gpuStream_t stream=static_cast<gpuStream_t>(handle->stream);
    const std::size_t o_pack=0,o_local=p.npack,o_unpack=o_local+p.nlocal,o_diag=o_unpack+p.nunpack;
    const std::size_t meta_count=o_diag+p.ndiag;
    std::exception_ptr failure;
    devT* a_dev = reinterpret_cast<devT*>(A);
    devT* send_dev = reinterpret_cast<devT*>(d_send);
    devT* recv_dev = reinterpret_cast<devT*>(d_recv);
    const dim3 block(32, 8, 1);

    try {
#if !defined(DDLA_USE_CCL) || defined(DDLA_USE_GPU_CPU_TUNNEL)
        // commAlltoallv must not grow a host buffer after peers enter MPI.
        detail::tunnel_staging<T>(handle, 0, p.send_total);
        detail::tunnel_staging<T>(handle, 1, p.recv_total);
#endif
        if(meta_count)
            DDLA_CHECK(ddlaMemcpy(handle,d_meta,meta,meta_count*sizeof(HermBlk)));
        if (p.npack) {
            pherm_xpose_kernel<devT><<<static_cast<unsigned>(p.npack), block, 0, stream>>>(
                a_dev, send_dev, d_meta + o_pack);
            DDLA_CHECK(ddlaGetLastError(handle));
        }
        if (p.nlocal) {
            pherm_xpose_kernel<devT><<<static_cast<unsigned>(p.nlocal), block, 0, stream>>>(
                a_dev, a_dev, d_meta + o_local);
            DDLA_CHECK(ddlaGetLastError(handle));
        }
    } catch (...) { failure = std::current_exception(); }
    pherm_agree_failure(handle, failure);

    // grid-wide collective: every rank calls it, including ranks with nothing to exchange
    commAlltoallv<DdlaBackend::GPU>(handle, CommScope::Grid, nprocs,
                                    static_cast<const T*>(d_send), p.sendcounts, p.sdispls,
                                    d_recv, p.recvcounts, p.rdispls);

    if (p.nunpack) {
        pherm_copy_kernel<devT><<<static_cast<unsigned>(p.nunpack), block, 0, stream>>>(
            recv_dev, a_dev, d_meta + o_unpack);
        DDLA_CHECK(ddlaGetLastError(handle));
    }
    if (p.ndiag) {
        pherm_diag_kernel<devT><<<static_cast<unsigned>(p.ndiag), block, 0, stream>>>(
            a_dev, d_meta + o_diag, upper);
        DDLA_CHECK(ddlaGetLastError(handle));
    }

    // Complete the metadata upload before caller-owned host scratch can be reused.
    DDLA_CHECK(ddlaSynchronize(handle));
}

#endif // DDLA_USE_CUDA || DDLA_USE_HIP

} // namespace

namespace spectral_impl {
HermitianPlan pherm_complete_check(const DdlaHandle_t& handle,DdlaBackend backend,char uplo,int n,const void* A,const int* descA)
{
    auto p=detail::agree_check(handle,"pherm_complete",[&](detail::Agree& agreement)->HermitianPlan {
        agreement.mode("spectral operation",3104);
    if(!handle)throw std::runtime_error("pherm_complete: null handle");
    if(ddlaGetBackend(handle)!=backend)throw std::runtime_error("pherm_complete: template backend does not match handle backend");
    if(uplo!='U'&&uplo!='L')throw std::runtime_error("pherm_complete: uplo must be 'U' or 'L'");
    check_desc(descA,handle);
    if(descA[DDLA_MB_]!=descA[DDLA_NB_])throw std::runtime_error("pherm_complete: requires square blocks (mb == nb)");
    if(descA[DDLA_RSRC_]!=0||descA[DDLA_CSRC_]!=0)throw std::runtime_error("pherm_complete: requires RSRC == CSRC == 0");
    if(n<0||n>descA[DDLA_M_]||n>descA[DDLA_N_])throw std::runtime_error("pherm_complete: n exceeds descriptor dimensions");
    HermitianPlan p;p.backend=backend;p.upper=uplo=='U';p.n=n;p.nb=descA[DDLA_NB_];p.lld=descA[DDLA_LLD_];
    ddlaGetGridDims(handle,p.pr,p.pc);ddlaGetGridCoords(handle,p.mr,p.mc);p.major=handle->major;
    if(n>0&&num_loc(n,p.nb,p.mr,0,p.pr)>0&&num_loc(n,p.nb,p.mc,0,p.pc)>0&&!A)
        throw std::invalid_argument("pherm_complete: null active input/output");
    return p;
    });
    int signature[13]={int(backend),int(uplo),n,handle->nprows_,handle->npcols_,int(handle->major)};
    detail::descriptor_signature(signature+6,descA);
    detail::agree_signature(handle,"pherm_complete",signature);
#if DDLA_HAS_CPU
    if(n>0&&backend==DdlaBackend::CPU) {
        // Every rank enters topology discovery, even if only some worker
        // caches were warmed by previous calls on a different communicator.
        MPI_Comm node=MPI_COMM_NULL;int ranks_on_node=1;
        MPI_CHECK(MPI_Comm_split_type(ddlaGetCommunicator(handle),MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node));
        MPI_CHECK(MPI_Comm_size(node,&ranks_on_node));MPI_CHECK(MPI_Comm_free(&node));
        p.threads=detail::host_omp_threads_for_node(ranks_on_node);
    }
#endif
    return p;
}
void pherm_complete_geometry(const HermitianPlan& p,std::size_t& pack_count, std::size_t& local_count, std::size_t& unpack_count, std::size_t& diag_count, std::size_t& send_total_count, std::size_t& recv_total_count)
{
    pack_count=local_count=unpack_count=diag_count=send_total_count=recv_total_count=0;if(p.n<1)return;
    const int blocks=(p.n-1)/p.nb+1;
    for(int br=0;br<blocks;++br)for(int bc=0;bc<blocks;++bc) {
        if(p.upper?bc<=br:bc>=br)continue;
        const bool src=br%p.pr==p.mr&&bc%p.pc==p.mc,dst=bc%p.pr==p.mr&&br%p.pc==p.mc;
        const auto elements=static_cast<std::size_t>(std::min(p.nb,p.n-br*p.nb))*std::min(p.nb,p.n-bc*p.nb);
        if(src&&dst)++local_count;
        else if(src){++pack_count;send_total_count+=elements;}
        else if(dst){++unpack_count;recv_total_count+=elements;}
    }
    for(int b=0;b<blocks;++b)if(b%p.pr==p.mr&&b%p.pc==p.mc)++diag_count;
    pherm_checked_count(static_cast<long long>(send_total_count));pherm_checked_count(static_cast<long long>(recv_total_count));
    return;
}
// Host: [ descriptors | six per-peer int arrays ].  Backend: [ device descriptors (GPU) | send pack | receive pack ],
// without the packs when they borrow the caller's scratch.
template<class T>
void pherm_complete_workspace_size(const DdlaHandle_t&,const HermitianPlan& p,std::size_t scratch_elements,
    std::size_t* backend_bytes,std::size_t* host_bytes)
{
    *backend_bytes=0;*host_bytes=0;
    if(p.n<1)return;
    std::size_t pack_count, local_count, unpack_count, diag_count, send_total_count, recv_total_count;
    pherm_complete_geometry(p,pack_count, local_count, unpack_count, diag_count, send_total_count, recv_total_count);
    const auto blocks=pack_count+local_count+unpack_count+diag_count;
    *host_bytes=detail::padded_bytes<HermitianBlock>(blocks)+detail::padded_bytes<int>(std::size_t(p.pr)*p.pc*6);
    if(p.backend==DdlaBackend::GPU)*backend_bytes=detail::padded_bytes<HermitianBlock>(blocks);
    if(!pherm_borrows_scratch(p,send_total_count,recv_total_count,scratch_elements))
        *backend_bytes+=detail::padded_bytes<T>(send_total_count)+detail::padded_bytes<T>(recv_total_count);
}

// The completion on the caller's scratch, carved in the order the size function counts it.  The packs borrow
// scratch exactly when the size function left them out.
template<DdlaBackend Backend,class T>
void pherm_complete_run(const DdlaHandle_t& handle,const HermitianPlan& p,T* A,T* scratch,std::size_t scratch_elements,
    void* backend_work,std::size_t backend_bytes,void* host_work,std::size_t host_bytes)
{
    if(p.n<1)return;
    std::size_t pack_count, local_count, unpack_count, diag_count, send_total_count, recv_total_count;
    pherm_complete_geometry(p,pack_count, local_count, unpack_count, diag_count, send_total_count, recv_total_count);
    const auto blocks=pack_count+local_count+unpack_count+diag_count;
    char* hw=static_cast<char*>(host_work);
    HermBlk* const meta=detail::take<HermBlk>(hw,blocks);
    int* const counts=detail::take<int>(hw,std::size_t(p.pr)*p.pc*6);
    assert(static_cast<std::size_t>(hw-static_cast<char*>(host_work))<=host_bytes);
    (void)host_bytes;
    const auto plan=pherm_build_plan(handle,p,pack_count, local_count, unpack_count, diag_count, send_total_count, recv_total_count,meta,counts);
    char* w=static_cast<char*>(backend_work);
    HermBlk* const device=Backend==DdlaBackend::GPU?detail::take<HermBlk>(w,blocks):nullptr;
    T *send=nullptr,*recv=nullptr;
    if(pherm_borrows_scratch(p,send_total_count,recv_total_count,scratch_elements)) {
        send=send_total_count?scratch:nullptr;recv=recv_total_count?scratch+send_total_count:nullptr;
    } else {
        send=detail::take<T>(w,send_total_count);recv=detail::take<T>(w,recv_total_count);
    }
    assert(static_cast<std::size_t>(w-static_cast<char*>(backend_work))<=backend_bytes);
    (void)backend_bytes;(void)device;
    if constexpr(Backend==DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        pherm_exec_cpu<T>(handle,plan,p.upper,A,p.pr*p.pc,p.threads,send,recv);
#endif
    } else {
#if defined(DDLA_USE_CUDA)||defined(DDLA_USE_HIP)
        pherm_exec_gpu<T>(handle,plan,p.upper,A,p.pr*p.pc,device,meta,send,recv);
#endif
    }
}
} // namespace spectral_impl

template<DdlaBackend Backend,class T>
void detail::pherm_complete_with_workspace(const DdlaHandle_t& handle,char uplo,int n,T* A,const int* descA,
    T* scratch,std::size_t scratch_elements)
{
    const auto p=spectral_impl::pherm_complete_check(handle,Backend,uplo,n,A,descA);
    const std::size_t capacity=scratch?scratch_elements:0;
    std::size_t backend_bytes=0,host_bytes=0;
    vector<char> backend_work(handle);std::vector<char> host_work;std::exception_ptr failure;
    try {
        spectral_impl::pherm_complete_workspace_size<T>(handle,p,capacity,&backend_bytes,&host_bytes);
        backend_work.resize(backend_bytes);host_work.resize(host_bytes);
    } catch(...){failure=std::current_exception();}
    if(n>0)pherm_agree_failure(handle,failure);
    else if(failure)std::rethrow_exception(failure);
    spectral_impl::pherm_complete_run<Backend,T>(handle,p,A,scratch,capacity,
        backend_work.data(),backend_bytes,host_work.data(),host_bytes);
}

template <DdlaBackend Backend, typename T>
void pherm_complete(const DdlaHandle_t& handle, char uplo, int n,
                    T* A, const int* descA)
{
    detail::pherm_complete_with_workspace<Backend, T>(handle, uplo, n, A, descA, nullptr, 0);
}

#define INSTANTIATE_PHERM_COMPLETE(BACKEND, TYPE)                                \
    template void pherm_complete<BACKEND, TYPE>(                                 \
        const DdlaHandle_t&, char, int, TYPE*, const int*);                       \
    template void detail::pherm_complete_with_workspace<BACKEND, TYPE>(          \
        const DdlaHandle_t&, char, int, TYPE*, const int*, TYPE*, std::size_t); \
    template void spectral_impl::pherm_complete_run<BACKEND,TYPE>(const DdlaHandle_t&, \
        const spectral_impl::HermitianPlan&,TYPE*,TYPE*,std::size_t,void*,std::size_t,void*,std::size_t)
#define INSTANTIATE_PHERM_COMPLETE_SIZE(TYPE) \
    template void spectral_impl::pherm_complete_workspace_size<TYPE>(const DdlaHandle_t&, \
        const spectral_impl::HermitianPlan&,std::size_t,std::size_t*,std::size_t*)

#if DDLA_HAS_CPU
INSTANTIATE_PHERM_COMPLETE(DdlaBackend::CPU, float);
INSTANTIATE_PHERM_COMPLETE(DdlaBackend::CPU, double);
INSTANTIATE_PHERM_COMPLETE(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PHERM_COMPLETE(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_PHERM_COMPLETE(DdlaBackend::GPU, float);
INSTANTIATE_PHERM_COMPLETE(DdlaBackend::GPU, double);
INSTANTIATE_PHERM_COMPLETE(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PHERM_COMPLETE(DdlaBackend::GPU, std::complex<double>);
#endif

INSTANTIATE_PHERM_COMPLETE_SIZE(float);
INSTANTIATE_PHERM_COMPLETE_SIZE(double);
INSTANTIATE_PHERM_COMPLETE_SIZE(std::complex<float>);
INSTANTIATE_PHERM_COMPLETE_SIZE(std::complex<double>);

#undef INSTANTIATE_PHERM_COMPLETE_SIZE
#undef INSTANTIATE_PHERM_COMPLETE

} // namespace ddla
