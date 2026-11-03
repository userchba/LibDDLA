#include "pvector_redist.h"

#include "ddla_connector.h"
#include "check_agree.h"
#include "comm_traits.h"
#include "ddla_stream.h"
#include "helpers.h"
#include "plan_util.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <limits>
#include "ddla_desc.h"
#include <string>
#include <vector>

namespace ddla {

namespace {

inline int pv_gcd(int a, int b)
{
    while (b != 0) { const int t = a % b; a = b; b = t; }
    return a;
}

// Number of blocks of the round starting at global block @p first (stride
// @p lcm) among @p nblks_tot blocks.
inline int round_block_count(int first, int nblks_tot, int lcm)
{
    if (first >= nblks_tot) return 0;
    return (nblks_tot - 1 - first) / lcm + 1;
}

} // namespace

namespace detail {

#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)

// Gather, for one round, the blocks global-index i = round_start + s*lcm of
// all nvc vectors from the row-distributed layout (local start
// (i/nprows)*nb) into the contiguous per-round buffer aux (slot s*nb inside
// vector region lc*aux_stride).  One thread block per (slot, vector) pair.
template <typename devT>
__global__ void pv_pack_round_kernel(
    const devT* __restrict__ src, devT* __restrict__ aux,
    int n, int nblks_comm, int nb, int nvc, int aux_stride,
    int round_start, int lcm, int nprows, int ld_src)
{
    const int s  = blockIdx.x % nblks_comm;
    const int lc = blockIdx.x / nblks_comm;
    const int i  = round_start + s * lcm;
    const int full = n - i * nb;
    const int nl = full < nb ? full : nb;
    const devT* svec = src + static_cast<std::size_t>(lc) * ld_src + static_cast<std::size_t>((i / nprows) * nb);
    devT* avec = aux + static_cast<std::size_t>(lc) * aux_stride + static_cast<std::size_t>(s * nb);
    for (int e = threadIdx.x; e < nl; e += blockDim.x) avec[e] = svec[e];
}

// Reverse of pv_pack_round_kernel into the column-distributed layout (local
// start (i/npcols)*nb), overwriting or accumulating.
template <typename devT>
__global__ void pv_unpack_round_kernel(
    const devT* __restrict__ aux, devT* __restrict__ dst,
    int n, int nblks_comm, int nb, int nvc, int aux_stride,
    int round_start, int lcm, int npcols, int ld_dst, bool accumulate)
{
    const int s  = blockIdx.x % nblks_comm;
    const int lc = blockIdx.x / nblks_comm;
    const int i  = round_start + s * lcm;
    const int full = n - i * nb;
    const int nl = full < nb ? full : nb;
    const devT* avec = aux + static_cast<std::size_t>(lc) * aux_stride + static_cast<std::size_t>(s * nb);
    devT* dvec = dst + static_cast<std::size_t>(lc) * ld_dst + static_cast<std::size_t>((i / npcols) * nb);
    for (int e = threadIdx.x; e < nl; e += blockDim.x) {
        dvec[e] = accumulate ? dvec[e] + avec[e] : avec[e];
    }
}

#endif // defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)

} // namespace detail

std::size_t ptranspose_vectors_work_elems(const DdlaHandle_t& handle,
                                          int n, int nvc, int nb)
{
    if (handle == nullptr)
        throw std::runtime_error("ptranspose_vectors_work_elems: null handle");
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    const int nblks_tot = (n + nb - 1) / nb;
    if (nblks_tot <= 0 || nvc <= 0) return 0;
    const int lcm = nprows / pv_gcd(nprows, npcols) * npcols;
    const int max_round_blks = (nblks_tot + lcm - 1) / lcm;
    return static_cast<std::size_t>(nb) * max_round_blks * nvc;
}

namespace spectral_impl {
VectorRedistPlan ptranspose_vectors_geometry(const DdlaHandle_t& handle,DdlaBackend backend,
    int n,int nvc,int nb,int ld_src,int ld_dst,bool accumulate,bool to_rows)
{
    VectorRedistPlan p;p.backend=backend;p.n=n;p.nvc=nvc;p.nb=nb;p.ld_src=ld_src;p.ld_dst=ld_dst;
    p.accumulate=accumulate;p.to_rows=to_rows;
    ddlaGetGridDims(handle,p.nprows,p.npcols);ddlaGetGridCoords(handle,p.myprow,p.mypcol);
    if(to_rows){std::swap(p.nprows,p.npcols);std::swap(p.myprow,p.mypcol);}
    p.nblks=n/nb+(n%nb!=0);
    const long long lcm=static_cast<long long>(p.nprows/pv_gcd(p.nprows,p.npcols))*p.npcols;
    p.lcm=static_cast<int>(lcm);
    p.max_round_blocks=p.nblks/p.lcm+(p.nblks%p.lcm!=0);
    return p;
}
VectorRedistPlan ptranspose_vectors_check(const DdlaHandle_t& handle,DdlaBackend backend,
    int n,int nvc,int nb,int ld_src,int ld_dst,bool accumulate,bool to_rows,const void* src,const void* dst)
{
    const auto p=detail::agree_check(handle,"ptranspose_vectors",[&](detail::Agree& agreement)->VectorRedistPlan {
        agreement.mode("spectral operation",3105);
    if(!handle)throw std::runtime_error("ptranspose_vectors: null handle");
    if(ddlaGetBackend(handle)!=backend)
        throw std::runtime_error("ptranspose_vectors: template backend does not match handle backend");
    if(n<0||nvc<0||nb<=0)throw std::invalid_argument("ptranspose_vectors: invalid dimensions");
    int pr=0,pc=0;ddlaGetGridDims(handle,pr,pc);
    if(static_cast<long long>(pr/pv_gcd(pr,pc))*pc>std::numeric_limits<int>::max())
        throw std::overflow_error("ptranspose_vectors: grid LCM overflow");
    const auto p=ptranspose_vectors_geometry(handle,backend,n,nvc,nb,ld_src,ld_dst,accumulate,to_rows);
    if(n&&nvc&&(ld_src<std::max(1,num_loc(n,nb,p.myprow,0,p.nprows))||
                  ld_dst<std::max(1,num_loc(n,nb,p.mypcol,0,p.npcols))))
        throw std::invalid_argument("ptranspose_vectors: invalid leading dimension");
    if(n&&nvc&&((num_loc(n,nb,p.myprow,0,p.nprows)>0&&!src)||
                  (num_loc(n,nb,p.mypcol,0,p.npcols)>0&&!dst)))
        throw std::invalid_argument("ptranspose_vectors: missing active input/output");
    return p;
    });
    const int signature[]={int(backend),n,nvc,nb,accumulate?1:0,to_rows?1:0,
        handle->nprows_,handle->npcols_,int(handle->major)};
    detail::agree_signature(handle,"ptranspose_vectors",signature);
    return p;
}

// The broadcast buffer, nb * max_round_blocks * nvc entries.
template<class T>
void ptranspose_vectors_workspace_size(const DdlaHandle_t& handle,int n,int nvc,int nb,std::size_t* backend_bytes)
{
    *backend_bytes=detail::padded_bytes<T>(ptranspose_vectors_work_elems(handle,n,nvc,nb));
}

// [ summed panel | send pack of padded input, then the transpose's broadcast buffer ].
template<class T>
void preduce_add_vectors_workspace_size(const DdlaHandle_t& handle,int n,int nvc,int nb,int ld_src,
    std::size_t* backend_bytes)
{
    *backend_bytes=0;
    if(n<=0||nvc<=0)return;
    int nprows=0,npcols=0,myprow=0,mypcol=0;
    ddlaGetGridDims(handle,nprows,npcols);ddlaGetGridCoords(handle,myprow,mypcol);
    // All members of one process row share the active row count, while
    // the caller's source stride/padding is allowed to differ by rank.
    const int sum_rows=num_loc(n,nb,myprow,0,nprows),sum_ld=std::max(sum_rows,1);
    const auto sum_storage=std::max<std::size_t>(static_cast<std::size_t>(sum_rows)*nvc,1);
    // The send pack dies after the reduction. Reuse it for the subsequent
    // transpose rounds instead of retaining a third simultaneous buffer.
    const auto packed_storage=ld_src==sum_ld?std::size_t(1):sum_storage;
    *backend_bytes=detail::padded_bytes<T>(sum_storage)+
        detail::padded_bytes<T>(std::max(packed_storage,ptranspose_vectors_work_elems(handle,n,nvc,nb)));
}

template <DdlaBackend Backend, typename T>
void ptranspose_vectors_run(const DdlaHandle_t& handle,const VectorRedistPlan& p,const T* src,T* dst,
    void* backend_work,std::size_t backend_bytes)
{
    static_assert(Backend!=DdlaBackend::GPU||DDLA_HAS_GPU,"GPU vector redistribution is not available");
    if(!p.n||!p.nvc)return;
    const int n=p.n,nvc=p.nvc,nb=p.nb,ld_src=p.ld_src,ld_dst=p.ld_dst;
    const int nprows=p.nprows,npcols=p.npcols,myprow=p.myprow,mypcol=p.mypcol;
    const int nblks_tot=p.nblks,lcm=p.lcm,max_round_blks=p.max_round_blocks;
    const bool accumulate_dst=p.accumulate;
    const CommScope scope=p.to_rows?CommScope::Row:CommScope::Col;
    // Per-round scratch buffer (backend memory via the handle): vector
    // regions are laid out contiguously with per-round stride nb*count, so a
    // single broadcast of count*nb*nvc elements covers the whole round.
    //
    // Zeroed once up front: the per-slot packing writes only nl elements when
    // the round's last global block is partial, but the broadcast always sends
    // the full nb-element slot.  The receiving unpack reads only nl elements,
    // so the padding is never consumed -- zeroing keeps the transmitted bytes
    // defined rather than leaving uninitialized memory on the wire.
    //
    // The caller's scratch replaces the per-call allocation: phe2hb
    // transposes two panels per step, and a cudaMallocAsync/cudaFreeAsync
    // pair per call is trimmed and re-mapped by the default pool at the next
    // synchronize (release threshold 0).  It is the only buffer: the front
    // of the scratch, with no padding after it.
    const int aux_stride_max = nb * max_round_blks;
    const std::size_t aux_elems = static_cast<std::size_t>(aux_stride_max) * nvc;
    T* const aux = static_cast<T*>(backend_work);
    assert(aux_elems * sizeof(T) <= backend_bytes);
    (void)backend_bytes;
    if constexpr (Backend == DdlaBackend::CPU) {
        std::memset(aux, 0, aux_elems * sizeof(T));
    } else {
        DDLA_CHECK(ddlaMemset(handle, aux, 0, aux_elems * sizeof(T)));
    }

#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
    using devT = detail::DeviceScalarT<T>;
    const devT* src_dev = reinterpret_cast<const devT*>(src);
    devT* aux_dev = reinterpret_cast<devT*>(aux);
    devT* dst_dev = reinterpret_cast<devT*>(dst);
    gpuStream_t stream = static_cast<gpuStream_t>(handle->stream);
    const int threads = 256;
#endif

    for (int r = 0; r < lcm; ++r) {
        const int ipt = r % npcols;   // active destination process column
        const int ips = r % nprows;   // source process row (broadcast root)
        if (mypcol != ipt) continue;

        const int count = round_block_count(r, nblks_tot, lcm);
        if (count == 0) continue;
        const int aux_stride = nb * count;
        const std::size_t bcast_elems = static_cast<std::size_t>(aux_stride) * nvc;

        if constexpr (Backend == DdlaBackend::CPU) {
            // Only the broadcast root owns global blocks i = r + s*lcm
            // (i mod nprows == ips); packing on other process rows would read
            // offsets beyond their local vector extent.
            if (myprow == ips) {
                for (int s = 0; s < count; ++s) {
                    const int i = r + s * lcm;
                    const int nl = std::min(n - i * nb, nb);
                    const int src_off = (i / nprows) * nb;
                    for (int lc = 0; lc < nvc; ++lc)
                        std::memcpy(aux + static_cast<std::size_t>(lc) * aux_stride + static_cast<std::size_t>(s * nb),
                                    src + static_cast<std::size_t>(lc) * ld_src + src_off,
                                    sizeof(T) * nl);
                }
            }
        } else {
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
            if (myprow == ips) {
                const dim3 grid_pack(static_cast<unsigned>(count * nvc), 1, 1);
                detail::pv_pack_round_kernel<devT><<<grid_pack, threads, 0, stream>>>(
                    src_dev, aux_dev, n, count, nb, nvc, aux_stride, r, lcm, nprows, ld_src);
                DDLA_CHECK(ddlaGetLastError(handle));
            }
#endif
        }

        commBcast<Backend>(handle, scope, aux, bcast_elems, ips);

        if constexpr (Backend == DdlaBackend::CPU) {
            for (int s = 0; s < count; ++s) {
                const int i = r + s * lcm;
                const int nl = std::min(n - i * nb, nb);
                const int dst_off = (i / npcols) * nb;
                for (int lc = 0; lc < nvc; ++lc) {
                    T* d = dst + static_cast<std::size_t>(lc) * ld_dst + dst_off;
                    const T* a = aux + static_cast<std::size_t>(lc) * aux_stride + static_cast<std::size_t>(s * nb);
                    if (accumulate_dst) {
                        for (int e = 0; e < nl; ++e) d[e] += a[e];
                    } else {
                        std::memcpy(d, a, sizeof(T) * nl);
                    }
                }
            }
        } else {
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
            const dim3 grid_unpack(static_cast<unsigned>(count * nvc), 1, 1);
            detail::pv_unpack_round_kernel<devT><<<grid_unpack, threads, 0, stream>>>(
                aux_dev, dst_dev, n, count, nb, nvc, aux_stride, r, lcm, npcols, ld_dst,
                accumulate_dst);
            DDLA_CHECK(ddlaGetLastError(handle));
#endif
        }
    }

}

template<DdlaBackend Backend,class T>
void preduce_add_vectors_run(const DdlaHandle_t& handle,const VectorRedistPlan& p,const T* src,T* dst,
    void* backend_work,std::size_t backend_bytes)
{
    if(!p.n||!p.nvc)return;
    const int sum_rows=num_loc(p.n,p.nb,p.myprow,0,p.nprows),sum_ld=std::max(sum_rows,1);
    const std::size_t sum_elements=static_cast<std::size_t>(sum_rows)*p.nvc;
    char* const base=static_cast<char*>(backend_work);
    char* w=base;
    T* const summed=detail::take<T>(w,std::max<std::size_t>(sum_elements,1));
    assert(static_cast<std::size_t>(w-base)<=backend_bytes);
    // Already compact input can enter the reduction directly. Only padded
    // input needs packing; zero-row peers retain a valid dummy send pointer.
    // The pack starts the region the transpose takes after the reduction.
    const T* send=src;
    if(p.ld_src!=sum_ld||sum_elements==0) {
        T* const packed=reinterpret_cast<T*>(w);
        if(sum_rows>0)
            DDLA_CHECK(ddlaMemcpy2D(handle,packed,std::size_t(sum_ld)*sizeof(T),
                src,std::size_t(p.ld_src)*sizeof(T),std::size_t(sum_rows)*sizeof(T),p.nvc));
        send=packed;
    }
    // Complex reductions expand each element to two reals in CCL. Keep
    // every individual count representable in both that path and MPI.
    constexpr auto max_count=std::size_t(std::numeric_limits<int>::max()/2);
    if(sum_elements==0)
        commAllReduce<Backend>(handle,CommScope::Row,send,summed,0,cclSum);
    for(std::size_t offset=0;offset<sum_elements;) {
        const auto count=std::min(max_count,sum_elements-offset);
        commAllReduce<Backend>(handle,CommScope::Row,send+offset,summed+offset,
            static_cast<int>(count),cclSum);
        offset+=count;
    }
    auto compact=p;compact.ld_src=sum_ld;
    ptranspose_vectors_run<Backend,T>(handle,compact,summed,dst,w,backend_bytes-static_cast<std::size_t>(w-base));
}
} // namespace spectral_impl

template <DdlaBackend Backend, typename T>
void ptranspose_vectors(const DdlaHandle_t& handle,const T* src,int ld_src,
                        T* dst,int ld_dst,int n,int nvc,int nb,bool accumulate_dst,
                        T* aux_work,std::size_t aux_work_elems,bool to_rows)
{
    const auto p=spectral_impl::ptranspose_vectors_check(handle,Backend,n,nvc,nb,ld_src,ld_dst,accumulate_dst,to_rows,src,dst);
    if(aux_work) {
        const std::size_t aux_elems=ptranspose_vectors_work_elems(handle,n,nvc,nb);
        if(aux_work_elems<aux_elems)
            throw std::runtime_error("ptranspose_vectors: aux_work holds "+std::to_string(aux_work_elems)+
                                     " elements, needs "+std::to_string(aux_elems));
        spectral_impl::ptranspose_vectors_run<Backend,T>(handle,p,src,dst,aux_work,aux_work_elems*sizeof(T));
    } else {
        std::size_t backend_bytes=0;
        spectral_impl::ptranspose_vectors_workspace_size<T>(handle,n,nvc,nb,&backend_bytes);
        vector<char> backend_work(handle,backend_bytes);
        spectral_impl::ptranspose_vectors_run<Backend,T>(handle,p,src,dst,backend_work.data(),backend_bytes);
    }
}

template <DdlaBackend Backend, typename T>
void preduce_add_vectors(
    const DdlaHandle_t& handle,
    const T* src, int ld_src,
    T* dst, int ld_dst,
    int n, int nvc, int nb)
{
    const auto p=spectral_impl::ptranspose_vectors_check(handle,Backend,n,nvc,nb,ld_src,ld_dst,true,false,src,dst);
    std::size_t backend_bytes=0;
    spectral_impl::preduce_add_vectors_workspace_size<T>(handle,n,nvc,nb,ld_src,&backend_bytes);
    vector<char> backend_work(handle,backend_bytes);
    spectral_impl::preduce_add_vectors_run<Backend,T>(handle,p,src,dst,backend_work.data(),backend_bytes);
}

// Explicit instantiations, four scalar types per available backend.
#define INSTANTIATE_PVECTOR_REDIST(BACKEND, TYPE)                                   \
    template void ptranspose_vectors<BACKEND, TYPE>(                                \
        const DdlaHandle_t&, const TYPE*, int, TYPE*, int, int, int, int, bool,     \
        TYPE*, std::size_t, bool);                                                        \
    template void preduce_add_vectors<BACKEND, TYPE>(                               \
        const DdlaHandle_t&, const TYPE*, int, TYPE*, int, int, int, int); \
    template void spectral_impl::ptranspose_vectors_run<BACKEND,TYPE>(const DdlaHandle_t&, \
        const spectral_impl::VectorRedistPlan&,const TYPE*,TYPE*,void*,std::size_t); \
    template void spectral_impl::preduce_add_vectors_run<BACKEND,TYPE>(const DdlaHandle_t&, \
        const spectral_impl::VectorRedistPlan&,const TYPE*,TYPE*,void*,std::size_t)
#define INSTANTIATE_PVECTOR_REDIST_SIZE(TYPE) \
    template void spectral_impl::ptranspose_vectors_workspace_size<TYPE>(const DdlaHandle_t&,int,int,int,std::size_t*); \
    template void spectral_impl::preduce_add_vectors_workspace_size<TYPE>(const DdlaHandle_t&,int,int,int,int,std::size_t*)

INSTANTIATE_PVECTOR_REDIST(DdlaBackend::CPU, float);
INSTANTIATE_PVECTOR_REDIST(DdlaBackend::CPU, double);
INSTANTIATE_PVECTOR_REDIST(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PVECTOR_REDIST(DdlaBackend::CPU, std::complex<double>);

#if DDLA_HAS_GPU
INSTANTIATE_PVECTOR_REDIST(DdlaBackend::GPU, float);
INSTANTIATE_PVECTOR_REDIST(DdlaBackend::GPU, double);
INSTANTIATE_PVECTOR_REDIST(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PVECTOR_REDIST(DdlaBackend::GPU, std::complex<double>);
#endif

INSTANTIATE_PVECTOR_REDIST_SIZE(float);
INSTANTIATE_PVECTOR_REDIST_SIZE(double);
INSTANTIATE_PVECTOR_REDIST_SIZE(std::complex<float>);
INSTANTIATE_PVECTOR_REDIST_SIZE(std::complex<double>);

#undef INSTANTIATE_PVECTOR_REDIST_SIZE
#undef INSTANTIATE_PVECTOR_REDIST

} // namespace ddla
