#include "hb2st_replay_gpu.h"
#include "hb2st_householder_cpu.h"
#include "hb2st_replay_wave_schedule.h"
#include "comm_traits.h"
#include "ddla_connector.h"
#include "ddla_desc.h"
#include "ddla_stream.h"
#include "row_share.h"
#include "helpers.h"
#include "check_agree.h"
#include "plan_util.h"
#include "instantiate.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ddla { namespace detail {
namespace {
#if DDLA_HAS_GPU
template <class D, class R> __device__ D widen(R v) { return D(v); }

template <class D, class R>
__global__ void initialize_direct(const R* Y, int ldy, D* Z, int ldz, int n, const D* phases)
{
    __shared__ D tile[32][33];
    const int r0=blockIdx.y*32, c0=blockIdx.x*32;
    const int tx=threadIdx.x, ty=threadIdx.y;
    for(int j=ty;j<32;j+=blockDim.y) {
        const int r=r0+tx, c=c0+j;
        if(r<n && c<n) {
            D v=widen<D>(Y[r+static_cast<std::size_t>(c)*ldy]);
            if(phases && r>0) v*=phases[r-1];
            tile[j][tx]=v;
        }
    }
    __syncthreads();
    for(int j=ty;j<32;j+=blockDim.y)
        if(r0+j<n && c0+tx<n)
            Z[static_cast<std::size_t>(r0+j)*ldz+c0+tx]=tile[tx][j];
}

template <class D>
__global__ void transpose_square(D* Z, int ld, int n)
{
    const int r0=blockIdx.y*32, c0=blockIdx.x*32;
    if(blockIdx.x<blockIdx.y) return;
    __shared__ D a[32][33];
    __shared__ D b[32][33];
    const int tx=threadIdx.x, ty=threadIdx.y;
    for(int j=ty;j<32;j+=blockDim.y) {
        if(r0+j<n && c0+tx<n) a[j][tx]=Z[static_cast<std::size_t>(r0+j)*ld+c0+tx];
        if(blockIdx.x!=blockIdx.y && c0+j<n && r0+tx<n)
            b[j][tx]=Z[static_cast<std::size_t>(c0+j)*ld+r0+tx];
    }
    __syncthreads();
    for(int j=ty;j<32;j+=blockDim.y) {
        if(blockIdx.x==blockIdx.y) {
            if(r0+j<n && c0+tx<n) Z[static_cast<std::size_t>(r0+j)*ld+c0+tx]=a[tx][j];
        } else {
            if(c0+j<n && r0+tx<n) Z[static_cast<std::size_t>(c0+j)*ld+r0+tx]=a[tx][j];
            if(r0+j<n && c0+tx<n) Z[static_cast<std::size_t>(r0+j)*ld+c0+tx]=b[tx][j];
        }
    }
}

template <class R>
__global__ void pack_input(const R* Y,int ldy,int rows,const int* starts,const int* counts,
                           const int* disp,R* send)
{
    const int q=blockIdx.y;
    const std::size_t count=counts[q];
    for(std::size_t p=blockIdx.x*blockDim.x+threadIdx.x;p<count;p+=blockDim.x*gridDim.x)
        send[disp[q]+p]=Y[p%rows+static_cast<std::size_t>(starts[q]+p/rows)*ldy];
}

template <class D,class R>
__global__ void unpack_input(const R* recv,const int* rows,const int* rowlo,const int* disp,
                             int count,int input_offset,int coloffset,D* strip,int ld,const D* phases)
{
    const int q=blockIdx.y, nr=rows[q];
    const std::size_t total=static_cast<std::size_t>(nr)*count;
    for(std::size_t p=blockIdx.x*blockDim.x+threadIdx.x;p<total;p+=blockDim.x*gridDim.x) {
        const int r=rowlo[q]+p%nr;
        D v=widen<D>(recv[disp[q]+static_cast<std::size_t>(input_offset)*nr+p]);
        if(phases && r>0) v*=phases[r-1];
        strip[static_cast<std::size_t>(r)*ld+coloffset+p/nr]=v;
    }
}

template <class D,class R>
__device__ __forceinline__ void reverse_pair(R c,D s,D& a,D& b)
{
    const D va=a,vb=b,cd=D(c);
    a=cd*va-device_conj(s)*vb;
    b=s*va+cd*vb;
}

template <class D,class R>
__global__ void replay_level(D* strip,int ld,int columns,const int* rows,const R* cs,const D* ss,int offset)
{
    const int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=columns) return;
    const int p=offset+blockIdx.y, b=rows[p], a=b-1;
    D va=strip[static_cast<std::size_t>(a)*ld+col];
    D vb=strip[static_cast<std::size_t>(b)*ld+col];
    reverse_pair(cs[p],ss[p],va,vb);
    strip[static_cast<std::size_t>(a)*ld+col]=va;
    strip[static_cast<std::size_t>(b)*ld+col]=vb;
}

template<class D,int K> struct ReplayMinBlocks {
    // A 16-rotation complex-double window needs more registers than the
    // real kernels. Two resident blocks avoid forcing that window to spill.
    static constexpr int value=std::is_same<D,DeviceScalarT<std::complex<double>>>::value&&K>8?2:
        std::is_same<D,float>::value?3:4;
};

template<class D,class R,int K>
__global__ void __launch_bounds__(256,(ReplayMinBlocks<D,K>::value))
replay_waves(D* strip,int ld,int columns,const int* top,const int* counts,const int* offsets,
             const int* members,const R* cs,const D* ss,int first)
{
    constexpr bool real=std::is_same<D,R>::value;
    __shared__ R sc[K],sr[K],si[real?1:K];
    const int w=first+blockIdx.y,len=counts[w],p=threadIdx.x;
    if(p<len) {
        const int record=members[offsets[w]+p];
        sc[p]=cs[record];
        if constexpr(real) sr[p]=ss[record];
        else { const D s=ss[record]; sr[p]=s.real(); si[p]=s.imag(); }
    }
    __syncthreads();
    const int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=columns) return;
    const int row0=top[w]-K;
    D window[K+1];
    // Compile-time indices keep the domino window in registers. Only the
    // active rows are loaded/stored, matching the schedule's disjoint test.
    #pragma unroll
    for(int j=0;j<=K;++j)
        if(j>=K-len) window[j]=strip[static_cast<std::size_t>(row0+j)*ld+col];
    #pragma unroll
    for(int j=K-1;j>=0;--j) if(j<len) {
        if constexpr(real) reverse_pair(sc[j],sr[j],window[K-j-1],window[K-j]);
        else reverse_pair(sc[j],D(sr[j],si[j]),window[K-j-1],window[K-j]);
    }
    #pragma unroll
    for(int j=0;j<=K;++j)
        if(j>=K-len) strip[static_cast<std::size_t>(row0+j)*ld+col]=window[j];
}

template <class D>
__global__ void pack_output(const D* strip,int ld,int n,int coloffset,int count,int first,
                            int mb,int nb,int npr,int npc,int rsrc,int csrc,bool rowmajor,
                            const int* ownedrows,const int* slots,int slot_offset,const int* disp,D* send)
{
    const std::size_t total=static_cast<std::size_t>(n)*count;
    for(std::size_t p=blockIdx.x*blockDim.x+threadIdx.x;p<total;p+=blockDim.x*gridDim.x) {
        const int r=p/count, c=p%count, gj=first+c;
        const int pr=(r/mb+rsrc)%npr, pc=(gj/nb+csrc)%npc;
        const int q=rowmajor?pr*npc+pc:pc*npr+pr;
        const int lr=(r/mb/npr)*mb+r%mb;
        send[disp[q]+lr+static_cast<std::size_t>(slots[slot_offset+c])*ownedrows[pr]]=
            strip[static_cast<std::size_t>(r)*ld+coloffset+c];
    }
}

template <class D>
__global__ void unpack_output(const D* recv,const int* counts,const int* disp,const int* column_disp,
                              const int* columns,int rows,int nb,int npc,D* Z,int lld)
{
    const int q=blockIdx.y;
    const std::size_t count=counts[q];
    for(std::size_t p=blockIdx.x*blockDim.x+threadIdx.x;p<count;p+=blockDim.x*gridDim.x) {
        const int gj=columns[column_disp[q]+p/rows];
        const int lj=(gj/nb/npc)*nb+gj%nb;
        Z[p%rows+static_cast<std::size_t>(lj)*lld]=recv[disp[q]+p];
    }
}

int checked_count(long long count)
{
    if(count<0 || count>INT_MAX) throw std::overflow_error("hb2st replay: MPI panel count exceeds int");
    return static_cast<int>(count);
}
void displacements(const int* count,int* disp,int P)
{
    long long total=0;
    for(int q=0;q<P;++q) { disp[q]=checked_count(total); total+=count[q]; }
    checked_count(total);
}
#endif
} // namespace

template<class T>Hb2stReplayGpuPlan hb2st_replay_gpu_geometry(const DdlaHandle_t& h,int n,int kd,long long total,
    int ldy,std::size_t phase_count,const int* desc,int panel_columns,int replay_records,bool use_waves,
    bool borrow_main,std::size_t main_elements,const int* columns)
{
    Hb2stReplayGpuPlan p;p.n=n;p.kd=kd;p.total=total;p.ldy=ldy;p.use_waves=use_waves;p.borrow_main=borrow_main;p.columns=columns;
    ddlaGetGridDims(h,p.npr,p.npc);ddlaGetGridCoords(h,p.pr,p.pc);p.P=h->nprocs;p.rank=h->myid;
    const RowShare split(n,p.P);p.nr=split.count(p.rank);p.clo=!columns?split.lo(p.rank):columns[p.rank];
    p.nc=!columns?split.count(p.rank):columns[p.rank+1]-p.clo;
    p.mb=desc[DDLA_MB_];p.nb=desc[DDLA_NB_];p.rsrc=desc[DDLA_RSRC_];p.csrc=desc[DDLA_CSRC_];p.lld=desc[DDLA_LLD_];
    p.outrows=num_loc(n,p.mb,p.pr,p.rsrc,p.npr);p.outcols=num_loc(n,p.nb,p.pc,p.csrc,p.npc);
    p.main_columns=borrow_main&&n>0?static_cast<int>(std::min<std::size_t>(std::max(p.nc,0),main_elements/n)):p.nc;
    p.spill_columns=p.nc-p.main_columns;p.direct=p.P==1;p.stride=p.direct?p.lld:std::max(p.main_columns,1);
    using R=typename real_type<T>::type;p.phases=!std::is_same<T,R>::value&&phase_count>0;
    if(!n)return p;
    const std::size_t target=std::size_t(64)<<20;
    p.width=std::min(n,static_cast<int>(std::max<std::size_t>(1,std::min<std::size_t>(512,target/(4*std::size_t(n)*sizeof(T))))));
    if(p.width>=p.nb)p.width=(p.width/p.nb)*p.nb;
    if(panel_columns>0)p.width=std::max(1,std::min(p.width,panel_columns));
    constexpr long long chunk=DistributedHistory<T>::chunk_size;
    p.record_capacity=static_cast<int>(std::min(chunk,std::max(0LL,total)));
    const int available=std::max(1,p.record_capacity);
    p.rcmax=replay_records>0?std::min(available,replay_records):available;p.pipeline=total>p.rcmax;
    return p;
}

template<class T>Hb2stReplayGpuPlan hb2st_replay_gpu_check(const DdlaHandle_t& h,int n,
    const typename real_type<T>::type* Y,int ldy,const DistributedHistory<T>& history,const T* phases,std::size_t phase_count,
    const T* Z,const int* desc,int panel_columns,int replay_records,bool use_waves,T* main,std::size_t main_elements,
    const int* columns)
{
    using R=typename real_type<T>::type;
    auto p=detail::agree_check(h,"hb2st replay",[&](detail::Agree&) {
        if(!h||ddlaGetBackend(h)!=DdlaBackend::GPU||!DDLA_HAS_GPU)throw std::invalid_argument("hb2st replay: GPU handle required");
        check_desc(desc,h);
        if(n<0||n>desc[DDLA_M_]||n>desc[DDLA_N_]||h->nprocs>65535)
            throw std::invalid_argument("hb2st replay: invalid order or grid");
        if(!history.ready||history.n!=n||history.kd<1||history.total<0||history.ranks!=h->nprocs||history.rank!=h->myid)
            throw std::invalid_argument("hb2st replay: incompatible history");
        const bool borrowed=columns!=nullptr;
        if(borrowed) {
            if(h->nprocs==1||columns[0]!=0||columns[h->nprocs]!=n)
                throw std::invalid_argument("hb2st replay: invalid compact prefix");
            for(int i=0;i<h->nprocs;++i)if(columns[i]<0||columns[i]>columns[i+1])
                throw std::invalid_argument("hb2st replay: invalid compact prefix");
        }
        auto result=hb2st_replay_gpu_geometry<T>(h,n,history.kd,history.total,ldy,phase_count,desc,panel_columns,replay_records,use_waves,borrowed,main_elements,columns);
        result.main=main;
        if(ldy<std::max(result.nr,1)||(n&&result.nr&&!Y)||(result.outrows&&result.outcols&&!Z)||
            result.spill_columns>1||(result.main_columns>0&&borrowed&&!main))
            throw std::invalid_argument("hb2st replay: incompatible slab/output/strip");
        if constexpr(!std::is_same<T,R>::value)if(phase_count&&(!phases||phase_count<std::size_t(std::max(0,n-1))))
            throw std::invalid_argument("hb2st replay: incomplete phases");
        const std::size_t ybytes=std::size_t(ldy)*std::size_t(n)*sizeof(R);
        const std::size_t zbytes=result.outcols&&result.outrows?(std::size_t(result.outcols-1)*std::size_t(result.lld)+std::size_t(result.outrows))*sizeof(T):0;
        if(!borrowed&&detail::memory_overlaps(Y,ybytes,Z,zbytes))throw std::invalid_argument("hb2st replay: input/output overlap needs a borrowed strip");
        const std::size_t main_bytes=main_elements*sizeof(T);
        if(borrowed&&(detail::memory_overlaps(main,main_bytes,Y,ybytes)||detail::memory_overlaps(main,main_bytes,Z,zbytes)))
            throw std::invalid_argument("hb2st replay: main strip overlaps input/output");
        return result;
    });
    const std::uint64_t total=static_cast<std::uint64_t>(p.total);
    int signature[19]={n,p.kd,p.P,p.npr,p.npc,p.use_waves?1:0,p.width,p.rcmax,p.borrow_main?1:0,p.phases?1:0,
        int(total&0x7fffffffU),int((total>>31)&0x7fffffffU)};
    detail::descriptor_signature(signature+12,desc);detail::agree_signature(h,"hb2st replay",signature);
    // A compact prefix controls the all-to-all schedule. Compare bounded
    // fixed stack chunks, excluding every rank-local stride/capacity field.
    if(p.borrow_main)for(std::size_t off=0;off<std::size_t(p.P)+1;off+=64) {
        int low[64],high[64];const int count=static_cast<int>(std::min(std::size_t(64),std::size_t(p.P)+1-off));
        MPI_CHECK(MPI_Allreduce(p.columns+off,low,count,MPI_INT,MPI_MIN,h->comm));
        MPI_CHECK(MPI_Allreduce(p.columns+off,high,count,MPI_INT,MPI_MAX,h->comm));
        for(int i=0;i<count;++i)if(low[i]!=high[i])throw std::invalid_argument("hb2st replay: column prefix differs between ranks");
    }
    return p;
}

template<class T>void hb2st_replay_gpu_workspace_size(const DdlaHandle_t& h,const Hb2stReplayGpuPlan& p,std::size_t* backend_bytes,
    std::size_t* host_bytes)
{
    using R=typename real_type<T>::type;*backend_bytes=*host_bytes=0;if(!p.n)return;
    const bool replay=p.total>0&&p.nc>0;const std::size_t rcmax=std::size_t(p.rcmax),width=std::size_t(p.width),n=std::size_t(p.n);
    std::size_t backend=0,host=padded_bytes<hb2st_rot<T>>(std::size_t(p.record_capacity));
    if(replay) {
        backend+=padded_bytes<int>(rcmax)+padded_bytes<R>(rcmax)+padded_bytes<T>(rcmax);
        if(p.use_waves)backend+=2*padded_bytes<int>(rcmax)+padded_bytes<int>(rcmax+1);
        host+=padded_bytes<R>(rcmax)+padded_bytes<T>(rcmax)+padded_bytes<int>(n)+2*padded_bytes<int>(rcmax)+2*padded_bytes<int>(rcmax+1);
    }
    if(!p.direct) {
        if(!p.borrow_main)backend+=padded_bytes<T>(n*std::size_t(std::max(p.nc,1)));
        if(p.spill_columns)backend+=padded_bytes<T>(n);
        backend+=2*padded_bytes<T>(n*width)+8*padded_bytes<int>(std::size_t(std::max(p.P,p.width)));
        host+=8*padded_bytes<int>(std::size_t(p.P))+padded_bytes<int>(std::size_t(p.npr))+padded_bytes<int>(std::size_t(p.npc))+
            2*padded_bytes<int>(width);
    }
    if(p.phases)backend+=padded_bytes<T>(std::size_t(std::max(0,p.n-1)));
    if(replay&&p.use_waves)host+=hb2st_replay_waves_bytes(p.n,p.kd,p.rcmax);
    if(p.input_2d.enabled&&!p.direct){std::size_t b=0,hb=0;eig_2d_input_workspace_size<R>(h,p.input_2d,&b,&hb);backend+=b;host+=hb;}
    if(p.householder && p.n>=3 && p.kd>=2) {
        std::size_t apply=0;hh_gpu_apply_workspace_size<T>(p.kd,p.nc,p.householder_wy,&apply);
        backend+=apply;
    }
    *backend_bytes=backend;*host_bytes=host;
}

// Two slots of rcmax cosines, sines and four integer tables (plus one int),
// each rounded up to max_align_t, when the replay runs in several intervals.
template<class T>Hb2stReplayGpuResources<T>::Hb2stReplayGpuResources(const DdlaHandle_t& h,const Hb2stReplayGpuPlan& p):h_(h)
{
    using R=typename real_type<T>::type;
    if(!(p.total>0&&p.nc>0&&p.pipeline))return;
    constexpr std::size_t align=alignof(std::max_align_t);
    capacity_=std::size_t(p.rcmax);stride_=(capacity_*(sizeof(R)+sizeof(T)+4*sizeof(int))+sizeof(int)+align-1)/align*align;
    try {DDLA_CHECK(ddlaMallocHost(h_,&storage_,2*stride_));for(auto& e:copied_)DDLA_CHECK(ddlaEventCreate(h_,&e));}
    catch(...) {for(auto e:copied_)if(e)try{(void)ddlaEventDestroy(h_,e);}catch(...){}if(storage_)try{(void)ddlaFreeHost(h_,storage_);}catch(...){}throw;}
}
template<class T>Hb2stReplayGpuResources<T>::~Hb2stReplayGpuResources()
{
    if(pending_)try{(void)ddlaSynchronize(h_);}catch(...){}
    for(auto e:copied_)if(e)try{(void)ddlaEventDestroy(h_,e);}catch(...){}
    if(storage_)try{(void)ddlaFreeHost(h_,storage_);}catch(...){}
}
template<class T>void Hb2stReplayGpuResources<T>::wait(int slot)
{if(used_[slot]){DDLA_CHECK(ddlaEventSynchronize(h_,copied_[slot]));used_[slot]=false;}}
template<class T>void Hb2stReplayGpuResources<T>::upload(int slot,int part,void* destination,const void* source,std::size_t bytes)
{
    if(!bytes)return;using R=typename real_type<T>::type;
    auto* p=static_cast<std::byte*>(storage_)+std::size_t(slot)*stride_;
    const auto off=part==0?0:part==1?capacity_*sizeof(R):capacity_*(sizeof(R)+sizeof(T))+std::size_t(part-2)*capacity_*sizeof(int);
    assert(off<=stride_&&bytes<=stride_-off);std::memcpy(p+off,source,bytes);pending_=true;DDLA_CHECK(ddlaMemcpy(h_,destination,p+off,bytes));
}
template<class T>void Hb2stReplayGpuResources<T>::record(int slot)
{DDLA_CHECK(ddlaEventRecord(h_,copied_[slot]));used_[slot]=true;}
template<class T>void Hb2stReplayGpuResources<T>::finish()
{if(pending_)DDLA_CHECK(ddlaSynchronize(h_));completed();}

template <class T>
void hb2st_replay_gpu_run(const DdlaHandle_t& h,const Hb2stReplayGpuPlan& p,Hb2stReplayGpuResources<T>& uploads,
                          const DistributedHistory<T>& history,const T* phases,const typename real_type<T>::type* Y,T* Z,int& info,
                          bool* input_consumed,void* backend_work,std::size_t backend_bytes,void* host_work,std::size_t host_bytes,
                          Hb2stHouseholderCpu<T>* householder)
{
    try {
        info=0;if(input_consumed)*input_consumed=false;
#if DDLA_HAS_GPU
        using R=typename real_type<T>::type;using D=DeviceScalarT<T>;
        if(!p.n){if(input_consumed)*input_consumed=true;return;}
        uploads.finish(); // Reuse after an exceptional run also retires old slot reads.
        const int n=p.n,npr=p.npr,npc=p.npc,pr=p.pr,pc=p.pc,P=p.P,rank=p.rank,nr=p.nr,nc=p.nc,clo=p.clo;
        const int mb=p.mb,nb=p.nb,rsrc=p.rsrc,csrc=p.csrc,lld=p.lld,ldy=p.ldy;
        const int outrows=p.outrows,main_columns=p.main_columns,spill_columns=p.spill_columns,stride=p.stride,width=p.width,rcmax=p.rcmax;
        const bool direct=p.direct,use_waves=p.use_waves,pipeline=p.pipeline;const RowShare split(n,P);
        const bool replay=history.total>0&&nc>0;
        const int wave_size=(std::is_same<T,R>::value ||
            (std::is_same<T,std::complex<double>>::value && p.kd>=16))?16:8;
        // carved in the order hb2st_replay_gpu_workspace_size counts it
        char* bw=static_cast<char*>(backend_work);char* hw=static_cast<char*>(host_work);
        hb2st_rot<T>* const records=take<hb2st_rot<T>>(hw,std::size_t(p.record_capacity));
        R* cosines=nullptr;T* sines=nullptr;int *last=nullptr,*levels=nullptr,*starts=nullptr,*cursor=nullptr,*indices=nullptr;
        int* dj=nullptr;R* dc=nullptr;D* ds=nullptr;int *dtop=nullptr,*dcount=nullptr,*doffset=nullptr;
        if(replay) {
            cosines=take<R>(hw,rcmax);sines=take<T>(hw,rcmax);last=take<int>(hw,n);levels=take<int>(hw,rcmax);
            starts=take<int>(hw,std::size_t(rcmax)+1);cursor=take<int>(hw,std::size_t(rcmax)+1);indices=take<int>(hw,rcmax);
            dj=take<int>(bw,rcmax);dc=take<R>(bw,rcmax);ds=reinterpret_cast<D*>(take<T>(bw,rcmax));
            if(use_waves){dtop=take<int>(bw,rcmax);dcount=take<int>(bw,rcmax);doffset=take<int>(bw,std::size_t(rcmax)+1);}
        }
        // host panel integers: sc, sd, rc, rd, column starts, row counts, row offsets, rows per process row,
        // columns per process column, slots, column displacements, columns
        int* panel[12]{};D* strip=direct?reinterpret_cast<D*>(Z):p.borrow_main?reinterpret_cast<D*>(p.main):nullptr;
        D* spill=nullptr;void* send=nullptr;void* recv=nullptr;int* di[8]{};
        if(!direct) {
            const int counts[12]={P,P,P,P,P,P,P,npr,npc,width,P,width};
            for(int i=0;i<12;++i)panel[i]=take<int>(hw,counts[i]);
            if(!p.borrow_main)strip=reinterpret_cast<D*>(take<T>(bw,std::size_t(n)*std::max(nc,1)));
            if(spill_columns)spill=reinterpret_cast<D*>(take<T>(bw,n));
            send=take<T>(bw,std::size_t(n)*width);recv=take<T>(bw,std::size_t(n)*width);
            for(int i=0;i<8;++i)di[i]=take<int>(bw,std::max(P,width));
        }
        int *d0=di[0],*d1=di[1],*d2=di[2],*d3=di[3],*d4=di[4],*d5=di[5],*d6=di[6],*d7=di[7];
        D* dph=p.phases?reinterpret_cast<D*>(take<T>(bw,std::max(0,n-1))):nullptr;
        Hb2stReplayWaves waves;if(use_waves&&replay)waves=hb2st_replay_waves(n,p.kd,rcmax,hw);
        void* input_work=bw;void* input_host=hw;std::size_t input_bytes=0,input_host_bytes=0;
        if(p.input_2d.enabled&&!direct){input_bytes=p.input_2d.backend_bytes;input_host_bytes=p.input_2d.host_bytes;bw+=input_bytes;hw+=input_host_bytes;}
        void* hh_work=bw;std::size_t hh_bytes=0;
        if(p.householder && p.n>=3 && p.kd>=2){hh_gpu_apply_workspace_size<T>(p.kd,p.nc,p.householder_wy,&hh_bytes);bw+=hh_bytes;}
        assert(bw-static_cast<char*>(backend_work)<=static_cast<std::ptrdiff_t>(backend_bytes));
        assert(hw-static_cast<char*>(host_work)<=static_cast<std::ptrdiff_t>(host_bytes));(void)backend_bytes;(void)host_bytes;
        auto column_lo=[&](int rank){return !p.columns?split.lo(rank):p.columns[rank];};
        auto column_hi=[&](int rank){return !p.columns?split.hi(rank):p.columns[rank+1];};
        if(dph)DDLA_CHECK(ddlaMemcpy(h,dph,phases,std::size_t(n-1)*sizeof(T)));
        auto copy_ints=[&](int* dst,const int* values,std::size_t count) {
            if(count) DDLA_CHECK(ddlaMemcpy(h,dst,values,count*sizeof(int)));
        };
        if(direct) {
            initialize_direct<D,R><<<dim3((n+31)/32,(n+31)/32),dim3(32,8),0,static_cast<gpuStream_t>(h->stream)>>>(Y,p.input_2d.enabled?p.input_2d.descriptor[DDLA_LLD_]:ldy,strip,stride,n,dph);
            DDLA_CHECK(ddlaGetLastError(h));
        } else if(p.input_2d.enabled) {
            const int zero=0;copy_ints(d4,&n,1);copy_ints(d5,&zero,1);copy_ints(d3,&zero,1);
            eig_2d_input_run<DdlaBackend::GPU,R>(h,p.input_2d,Y,
                [&](int q){return column_hi(q)-column_lo(q);},[&](int q,int k){return column_lo(q)+k;},
                [&](int first,int count,const R* values,int){
                    const int main_count=std::max(0,std::min(count,main_columns-first));
                    if(main_count)unpack_input<D,R><<<dim3(32,1),256,0,static_cast<gpuStream_t>(h->stream)>>>(
                        values,d4,d5,d3,main_count,0,first,strip,stride,dph);
                    if(count>main_count)unpack_input<D,R><<<dim3(32,1),256,0,static_cast<gpuStream_t>(h->stream)>>>(
                        values,d4,d5,d3,count-main_count,main_count,0,spill,1,dph);
                    DDLA_CHECK(ddlaGetLastError(h));
                },input_work,input_bytes,input_host,input_host_bytes);
        } else {
            int *sc=panel[0],*sd=panel[1],*rc=panel[2],*rd=panel[3],*beg=panel[4],*rows=panel[5],*lo=panel[6];
            for(int q=0;q<P;++q){rows[q]=split.count(q);lo[q]=split.lo(q);}
            copy_ints(d4,rows,P); copy_ints(d5,lo,P);
            for(int j0=0;j0<n;j0+=width) {
                const int end=std::min(n,j0+width);
                const int a=std::max(j0,clo),b=std::min(end,column_hi(rank)),count=std::max(0,b-a);
                for(int q=0;q<P;++q) {
                    beg[q]=std::max(j0,column_lo(q));
                    const int c=std::max(0,std::min(end,column_hi(q))-beg[q]);
                    sc[q]=checked_count(static_cast<long long>(nr)*c);
                    rc[q]=checked_count(static_cast<long long>(rows[q])*count);
                }
                displacements(sc,sd,P); displacements(rc,rd,P);
                copy_ints(d0,beg,P); copy_ints(d1,sc,P); copy_ints(d2,sd,P); copy_ints(d3,rd,P);
                pack_input<R><<<dim3(32,P),256,0,static_cast<gpuStream_t>(h->stream)>>>(Y,ldy,nr,d0,d1,d2,static_cast<R*>(send));
                DDLA_CHECK(ddlaGetLastError(h));
                commAlltoallv<DdlaBackend::GPU>(h,CommScope::Grid,P,static_cast<R*>(send),sc,sd,static_cast<R*>(recv),rc,rd);
                const int main_count=std::max(0,std::min(b,clo+main_columns)-a);
                if(main_count>0)
                    unpack_input<D,R><<<dim3(32,P),256,0,static_cast<gpuStream_t>(h->stream)>>>(static_cast<R*>(recv),d4,d5,d3,main_count,0,a-clo,strip,stride,dph);
                if(count>main_count)
                    unpack_input<D,R><<<dim3(32,P),256,0,static_cast<gpuStream_t>(h->stream)>>>(static_cast<R*>(recv),d4,d5,d3,1,main_count,0,spill,1,dph);
                DDLA_CHECK(ddlaGetLastError(h));
                DDLA_CHECK(ddlaSynchronize(h));
            }
        }
        DDLA_CHECK(ddlaSynchronize(h));
        // Every rank has finished reading Y before any rank may overwrite an
        // aliased output allocation. Borrowed Y remains caller-owned.
        MPI_CHECK(MPI_Barrier(h->comm));
        if(input_consumed)*input_consumed=true;

        if(householder && p.n>=3 && p.kd>=2) {
            // All ranks participate, including ranks with no column work.
            householder->apply_rows_gpu_run(reinterpret_cast<T*>(strip),stride,main_columns,hh_work,hh_bytes);
            // Compact borrowing can put one extra column in a separate strip.
            if(p.borrow_main)householder->apply_rows_gpu_run(reinterpret_cast<T*>(spill),1,spill_columns,hh_work,hh_bytes);
        }
        int upload_slot=0;
        for(long long q=history.chunks();q-->0;) {
            DistributedHistoryHeader checkpoint;
            const int fetched=static_cast<int>(std::min(DistributedHistory<T>::chunk_size,history.total-q*DistributedHistory<T>::chunk_size));
            history.fetch(q,h->comm,records,&checkpoint,true);
            for(int end=fetched;end>0;) {
                const int begin=std::max(0,end-rcmax),count=end-begin;
                std::exception_ptr failure;
                bool wave_order=false;int nstarts=0;
                try {
                    if(nc>0) {
                        if(pipeline) uploads.wait(upload_slot);
                        if(use_waves) wave_order=waves.build(n,history.kd,records+begin,count,
                            checkpoint.cursor,checkpoint.offset,begin,wave_size);
                        if(wave_order) {
                            for(int k=0;k<count;++k) {
                                cosines[k]=records[begin+k].c; sines[k]=records[begin+k].s;
                            }
                        } else {
                            std::fill(last,last+n,0); starts[0]=0; nstarts=1;
                            for(int k=count;k-->0;) {
                                const int b=records[begin+k].j0;
                                if(b<1 || b>=n) throw std::logic_error("hb2st replay: invalid rotation row");
                                const int lv=1+std::max(last[b-1],last[b]);
                                last[b-1]=last[b]=lv; levels[k]=lv;
                                while(nstarts<=lv) starts[nstarts++]=0;
                                ++starts[lv];
                            }
                            for(int lv=1;lv<nstarts;++lv) starts[lv]+=starts[lv-1];
                            std::copy(starts,starts+nstarts,cursor);
                            for(int k=0;k<count;++k) {
                                const int at=cursor[levels[k]-1]++;
                                indices[at]=records[begin+k].j0; cosines[at]=records[begin+k].c; sines[at]=records[begin+k].s;
                            }
                        }
                    }
                } catch(...) { failure=std::current_exception(); }
                // This interval has no GPU collectives. Agree preparation and
                // enqueue failures together before any next history broadcast;
                // empty column owners must join the same agreement.
                try {
                if(!failure && nc>0) {
                    auto upload=[&](int part,void* destination,const void* values,std::size_t bytes) {
                        if(pipeline) uploads.upload(upload_slot,part,destination,values,bytes);
                        else if(bytes)
                            DDLA_CHECK(ddlaMemcpy(h,destination,values,bytes));
                    };
                    upload(0,dc,cosines,std::size_t(count)*sizeof(R)); upload(1,ds,sines,std::size_t(count)*sizeof(T));
                    if(wave_order) {
                        upload(2,dj,waves.members,std::size_t(count)*sizeof(int)); upload(3,dtop,waves.top,std::size_t(waves.waves)*sizeof(int));
                        upload(4,dcount,waves.count,std::size_t(waves.waves)*sizeof(int)); upload(5,doffset,waves.offset,std::size_t(waves.waves+1)*sizeof(int));
                        if(pipeline) uploads.record(upload_slot);
                        for(int b=waves.nbatches;b>0;--b) {
                            const int first=waves.batches[b-1],len=waves.batches[b]-first;
                            for(int part=0;part<2;++part) {
                                const int columns=part?spill_columns:main_columns;
                                if(columns==0) continue;
                                D* matrix=part?spill:strip; const int ld=part?1:stride;
                                if(wave_size==16)
                                    replay_waves<D,R,16><<<dim3((columns+255)/256,len),256,0,static_cast<gpuStream_t>(h->stream)>>>(matrix,ld,columns,dtop,dcount,doffset,dj,dc,ds,first);
                                else
                                    replay_waves<D,R,8><<<dim3((columns+255)/256,len),256,0,static_cast<gpuStream_t>(h->stream)>>>(matrix,ld,columns,dtop,dcount,doffset,dj,dc,ds,first);
                            }
                        }
                    } else {
                        upload(2,dj,indices,std::size_t(count)*sizeof(int));
                        if(pipeline) uploads.record(upload_slot);
                        for(int lv=1;lv<nstarts;++lv) {
                            const int lo=starts[lv-1],len=starts[lv]-lo;
                            for(int k=0;k<len;k+=65535) for(int part=0;part<2;++part) {
                                const int columns=part?spill_columns:main_columns;
                                if(columns>0)
                                    replay_level<D,R><<<dim3((columns+127)/128,std::min(65535,len-k)),128,0,static_cast<gpuStream_t>(h->stream)>>>(part?spill:strip,part?1:stride,columns,dj,dc,ds,lo+k);
                            }
                        }
                    }
                    DDLA_CHECK(ddlaGetLastError(h));
                    // Single-interval calls retain the simple pageable path.
                    // Pipelined calls protect pinned host slots by copy events;
                    // stream ordering protects the reused device arrays.
                    if(!pipeline) DDLA_CHECK(ddlaSynchronize(h));
                }
                } catch(...) { failure=std::current_exception(); }
                detail::agree_failure(h,failure,"hb2st replay: preparation");
                upload_slot^=1;
                end=begin;
            }
        }
        std::exception_ptr replay_error;
        try { DDLA_CHECK(ddlaSynchronize(h)); }
        catch(...) { replay_error=std::current_exception(); }
        detail::agree_failure(h,replay_error,"hb2st replay: preparation");
        if(direct) {
            transpose_square<D><<<dim3((n+31)/32,(n+31)/32),dim3(32,8),0,static_cast<gpuStream_t>(h->stream)>>>(strip,stride,n);
            DDLA_CHECK(ddlaGetLastError(h));
        } else {
            int *sc=panel[0],*sd=panel[1],*rc=panel[2],*rd=panel[3],*nrpr=panel[7],*ncpc=panel[8];
            int *slots=panel[9],*column_disp=panel[10],*columns=panel[11];
            std::exception_ptr failure;
            for(int p=0;p<npr;++p)nrpr[p]=num_loc(n,mb,p,rsrc,npr);
            copy_ints(d4,nrpr,npr);
            for(int j0=0;j0<n;j0+=width) {
                const int end=std::min(n,j0+width),a=std::max(j0,clo),b=std::min(end,column_hi(rank));
                const int count=std::max(0,b-a);
                int ncolumns=0;
                failure=nullptr;
                try {
                    std::fill(ncpc,ncpc+npc,0);
                    for(int k=0;k<count;++k) slots[k]=ncpc[((a+k)/nb+csrc)%npc]++;
                    for(int q=0;q<P;++q) {
                        const int qr=h->major=='R'?q/npc:q%npr, qc=h->major=='R'?q%npc:q/npr;
                        sc[q]=checked_count(static_cast<long long>(nrpr[qr])*ncpc[qc]);
                        column_disp[q]=ncolumns;
                        for(int j=std::max(j0,column_lo(q));j<std::min(end,column_hi(q));++j)
                            if((j/nb+csrc)%npc==pc) columns[ncolumns++]=j;
                        rc[q]=checked_count(static_cast<long long>(outrows)*(ncolumns-column_disp[q]));
                    }
                    displacements(sc,sd,P); displacements(rc,rd,P);
                } catch(...) { failure=std::current_exception(); }
                detail::agree_failure(h,failure,"hb2st replay: preparation");
                copy_ints(d0,sd,P); copy_ints(d1,slots,count); copy_ints(d2,rc,P); copy_ints(d3,rd,P);
                copy_ints(d6,column_disp,P); copy_ints(d7,columns,ncolumns);
                const int main_count=std::max(0,std::min(b,clo+main_columns)-a);
                if(main_count>0) pack_output<D><<<static_cast<int>(std::min<long long>(1024,(n*static_cast<long long>(main_count)+255)/256)),256,0,static_cast<gpuStream_t>(h->stream)>>>(
                    strip,stride,n,a-clo,main_count,a,mb,nb,npr,npc,rsrc,csrc,h->major=='R',d4,d1,0,d0,static_cast<D*>(send));
                if(count>main_count) pack_output<D><<<static_cast<int>(std::min<long long>(1024,(n+255LL)/256)),256,0,static_cast<gpuStream_t>(h->stream)>>>(
                    spill,1,n,0,1,a+main_count,mb,nb,npr,npc,rsrc,csrc,h->major=='R',d4,d1,main_count,d0,static_cast<D*>(send));
                DDLA_CHECK(ddlaGetLastError(h));
                commAlltoallv<DdlaBackend::GPU>(h,CommScope::Grid,P,reinterpret_cast<T*>(send),sc,sd,reinterpret_cast<T*>(recv),rc,rd);
                unpack_output<D><<<dim3(32,P),256,0,static_cast<gpuStream_t>(h->stream)>>>(static_cast<D*>(recv),d2,d3,d6,d7,outrows,nb,npc,reinterpret_cast<D*>(Z),lld);
                DDLA_CHECK(ddlaGetLastError(h));
                DDLA_CHECK(ddlaSynchronize(h));
            }
        }
        DDLA_CHECK(ddlaSynchronize(h));uploads.completed();
#else
        (void)h;(void)p;(void)uploads;(void)history;(void)phases;(void)Y;(void)Z;
        (void)backend_work;(void)backend_bytes;(void)host_work;(void)host_bytes;
        throw std::runtime_error("hb2st replay: built without GPU support");
#endif
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(h) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(h); } catch (...) {}
        throw;
    }
}

template<class T>void hb2st_replay_rows_gpu(const DdlaHandle_t& h,int n,typename real_type<T>::type*& Y,int ldy,
    const DistributedHistory<T>& history,const std::vector<T>& phases,T* Z,const int* desc,int& info,int panel_columns,
    int replay_records,bool use_waves,const Hb2stReplayGpuWorkspace<T>* workspace)
{
    info=0;const int* columns=nullptr;
    if(workspace&&workspace->column_offsets&&!workspace->column_offsets->empty())columns=workspace->column_offsets->data();
    detail::agree_check(h,"hb2st replay",[&](detail::Agree&) {
        if(workspace&&!columns)throw std::invalid_argument("hb2st replay: missing compact prefix");
        if(columns&&workspace->column_offsets->size()!=std::size_t(h->nprocs)+1)throw std::invalid_argument("hb2st replay: invalid compact prefix");
    });
    const auto p=hb2st_replay_gpu_check<T>(h,n,Y,ldy,history,phases.data(),phases.size(),Z,desc,panel_columns,
        replay_records,use_waves,workspace?workspace->main:nullptr,workspace?workspace->main_elements:0,columns);
    if(p.borrow_main)detail::agree_check(h,"hb2st replay",[&](detail::Agree&) {
        if(!workspace->owns_input||!Y||!Z||!n||!p.outrows||!p.outcols)return;
        const std::size_t ay=std::size_t(ldy)*std::size_t(n)*sizeof(typename real_type<T>::type);
        const std::size_t bz=(std::size_t(p.outcols-1)*std::size_t(p.lld)+std::size_t(p.outrows))*sizeof(T);
        if(detail::memory_overlaps(Y,ay,Z,bz))throw std::invalid_argument("hb2st replay: owned input overlaps output");
    });
    std::size_t backend_bytes=0,host_bytes=0;vector<char> backend_work(h);std::vector<char> host_work;
    std::optional<Hb2stReplayGpuResources<T>> resources;std::exception_ptr failure;
    try{hb2st_replay_gpu_workspace_size<T>(h,p,&backend_bytes,&host_bytes);backend_work.resize(backend_bytes);host_work.resize(host_bytes);
        resources.emplace(h,p);}catch(...){failure=std::current_exception();}
    detail::agree_failure(h->comm,failure,"hb2st replay: workspace preparation");
    bool consumed=false;
    auto retire=[&] {if(consumed){if(Y&&(!workspace||workspace->owns_input))DDLA_CHECK(ddlaFree(h,Y));Y=nullptr;}};
    try{hb2st_replay_gpu_run<T>(h,p,*resources,history,phases.data(),Y,Z,info,&consumed,backend_work.data(),backend_bytes,
        host_work.data(),host_bytes);}
    catch(...){retire();throw;}
    retire();
}

#define DDLA_REPLAY_PHASE(T) \
 template Hb2stReplayGpuPlan hb2st_replay_gpu_geometry<T>(const DdlaHandle_t&,int,int,long long,int,std::size_t,const int*,int,int,bool,bool,std::size_t,const int*); \
 template Hb2stReplayGpuPlan hb2st_replay_gpu_check<T>(const DdlaHandle_t&,int,const typename real_type<T>::type*,int,const DistributedHistory<T>&,const T*,std::size_t,const T*,const int*,int,int,bool,T*,std::size_t,const int*); \
 template void hb2st_replay_gpu_workspace_size<T>(const DdlaHandle_t&,const Hb2stReplayGpuPlan&,std::size_t*,std::size_t*); \
 template class Hb2stReplayGpuResources<T>; \
 template void hb2st_replay_gpu_run<T>(const DdlaHandle_t&,const Hb2stReplayGpuPlan&,Hb2stReplayGpuResources<T>&,const DistributedHistory<T>&,const T*,const typename real_type<T>::type*,T*,int&,bool*,void*,std::size_t,void*,std::size_t,Hb2stHouseholderCpu<T>*)
DDLA_FOR_EACH_TYPE(DDLA_REPLAY_PHASE)
#undef DDLA_REPLAY_PHASE

#define DDLA_REPLAY_INSTANTIATE(T) template void hb2st_replay_rows_gpu<T>(const DdlaHandle_t&,int,typename real_type<T>::type*&,int,const DistributedHistory<T>&,const std::vector<T>&,T*,const int*,int&,int,int,bool,const Hb2stReplayGpuWorkspace<T>*)
DDLA_FOR_EACH_TYPE(DDLA_REPLAY_INSTANTIATE)
#undef DDLA_REPLAY_INSTANTIATE
} }
