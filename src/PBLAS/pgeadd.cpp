#include "ddla_connector.h"
#include <ddla/ddla.h>
#include "ddla_desc.h"
#include "ddla_stream.h"
#include "require_backend.h"
#include "comm_traits.h"
#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <vector>

#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
#include <thrust/complex.h>
#endif

namespace ddla { namespace {

struct CopyBlock {
    int rows, cols;
    std::size_t src, dst;
    int lds, ldd;
};

#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
template <typename T> struct DeviceScalar { using type = T; };
template <> struct DeviceScalar<std::complex<float>> { using type = thrust::complex<float>; };
template <> struct DeviceScalar<std::complex<double>> { using type = thrust::complex<double>; };
template <typename T> __device__ T conjugate(T x, bool) { return x; }
template <typename R> __device__ thrust::complex<R> conjugate(thrust::complex<R> x, bool c) {
    return c ? thrust::conj(x) : x;
}

// Packing and scattering share this copy/transpose kernel.
template <typename T> __global__ void copy_blocks(
    const T* src, T* dst, const CopyBlock* blocks, bool transpose, bool conj)
{
    const CopyBlock b = blocks[blockIdx.x];
    if (!transpose) {
        for (int j=threadIdx.y; j<b.cols; j+=blockDim.y)
            for (int i=threadIdx.x; i<b.rows; i+=blockDim.x)
                dst[b.dst+i+std::size_t(j)*b.ldd] = src[b.src+i+std::size_t(j)*b.lds];
        return;
    }
    constexpr int tile_size = 16;
    __shared__ T tile[tile_size][tile_size+1];
    for (int j0=0; j0<b.cols; j0+=tile_size) for (int i0=0; i0<b.rows; i0+=tile_size) {
        const int i=i0+threadIdx.x, j=j0+threadIdx.y;
        if (i<b.rows && j<b.cols) tile[threadIdx.y][threadIdx.x]=src[b.src+i+std::size_t(j)*b.lds];
        __syncthreads();
        const int r=j0+threadIdx.x, c=i0+threadIdx.y;
        if (r<b.cols && c<b.rows) dst[b.dst+r+std::size_t(c)*b.ldd]=conjugate(tile[threadIdx.x][threadIdx.y],conj);
        __syncthreads();
    }
}
#endif

template <DdlaBackend Backend, typename T>
void copy_batch(DdlaHandle_t h, char op, const T* src, T* dst, const std::vector<CopyBlock>& blocks)
{
    if (blocks.empty()) return;
    if constexpr (Backend == DdlaBackend::CPU) {
        for (const auto& b : blocks)
            geam<Backend>(h,op,'N',op=='N'?b.rows:b.cols,op=='N'?b.cols:b.rows,
                T(1),src+b.src,b.lds,T(0),static_cast<const T*>(nullptr),1,dst+b.dst,b.ldd);
    } else {
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
        vector<CopyBlock> device_blocks(h);
        device_blocks.resize(blocks.size());
        DDLA_CHECK(ddlaMemcpy(h,device_blocks.data(),blocks.data(),blocks.size()*sizeof(CopyBlock)));
        using DT = typename DeviceScalar<T>::type;
        copy_blocks<DT><<<dim3(blocks.size()),dim3(16,16),0,static_cast<gpuStream_t>(h->stream)>>>(
            reinterpret_cast<const DT*>(src),reinterpret_cast<DT*>(dst),device_blocks.data(),op!='N',op=='C');
        DDLA_CHECK(ddlaGetLastError(h));
#endif
    }
}

char operation(char op) {
    op=static_cast<char>(std::toupper(static_cast<unsigned char>(op)));
    if (op!='N' && op!='T' && op!='C') throw std::invalid_argument("pgeadd: expected N, T or C");
    return op;
}

void check_region(const int* d, int i, int j, int m, int n) {
    if (i<1 || j<1 || m<0 || n<0 || i-1>d[DDLA_M_] || j-1>d[DDLA_N_] ||
        m>d[DDLA_M_]-(i-1) || n>d[DDLA_N_]-(j-1))
        throw std::invalid_argument("pgeadd: submatrix outside descriptor");
}

// Equal block boundaries let each input tile map to exactly one output tile.
// N additionally preserves ownership, so it needs no communication.
void check_alignment(DdlaHandle_t h, char op, const int* a, int ia, int ja,
                     const int* c, int ic, int jc) {
    const bool trans=op!='N';
    const int mb=a[trans?DDLA_NB_:DDLA_MB_], nb=a[trans?DDLA_MB_:DDLA_NB_];
    if (mb!=c[DDLA_MB_] || nb!=c[DDLA_NB_] ||
        (trans?ja:ia)%mb!=ic%mb || (trans?ia:ja)%nb!=jc%nb)
        throw std::invalid_argument("pgeadd: blocks and submatrix offsets must align after N/T/C");
    if (!trans &&
        (indxg2p(ia,mb,a[DDLA_RSRC_],h->nprows_)!=indxg2p(ic,mb,c[DDLA_RSRC_],h->nprows_) ||
         indxg2p(ja,nb,a[DDLA_CSRC_],h->npcols_)!=indxg2p(jc,nb,c[DDLA_CSRC_],h->npcols_)))
        throw std::invalid_argument("pgeadd: N requires matching input and output block owners");
}

struct TransposePeers { int dest, source; };

template <typename T>
bool overlaps(const T* a, int lda, int ar, int ac, const T* b, int ldb, int br, int bc) {
    if (ar==0 || ac==0 || br==0 || bc==0) return false;
    // Include pitch gaps conservatively. Comparing address distances avoids
    // ordering unrelated pointers or overflowing the end of an address range.
    const auto ap=reinterpret_cast<std::uintptr_t>(a),bp=reinterpret_cast<std::uintptr_t>(b);
    const std::size_t aspan=std::size_t(ac-1)*lda+ar,bspan=std::size_t(bc-1)*ldb+br;
    return ap<=bp ? (bp-ap)/sizeof(T)<aspan : (ap-bp)/sizeof(T)<bspan;
}

// On a square grid the transposed local slab has exactly one destination.
TransposePeers transpose_peers(DdlaHandle_t h, const int* a, int ia, int ja,
                               const int* c, int ic, int jc) {
    const int p=h->nprows_;
    const long long dr=static_cast<long long>(ic)/c[DDLA_MB_]-ja/a[DDLA_NB_]+c[DDLA_RSRC_]-a[DDLA_CSRC_];
    const long long dc=static_cast<long long>(jc)/c[DDLA_NB_]-ia/a[DDLA_MB_]+c[DDLA_CSRC_]-a[DDLA_RSRC_];
    const auto wrap=[&](long long x) { const int y=x%p;return y<0?y+p:y; };
    return {h->rc_to_rank(wrap(h->mypcol_+dr),wrap(h->myprow_+dc)),
            h->rc_to_rank(wrap(h->mypcol_-dc),wrap(h->myprow_-dr))};
}

// The square-grid route sends each local slab as one message with an int count.
bool square_route(DdlaHandle_t h, int m, int n) {
    return h->nprows_==h->npcols_ && std::size_t(m)*n<=INT_MAX;
}

// Aligned operands already share block sizes and offsets with C, so equal
// origin owners make two stored regions share every owner and local position.
bool colocated(DdlaHandle_t h, const int* a, int ia, int ja, const int* b, int ib, int jb) {
    return indxg2p(ia,a[DDLA_MB_],a[DDLA_RSRC_],h->nprows_)==indxg2p(ib,b[DDLA_MB_],b[DDLA_RSRC_],h->nprows_) &&
           indxg2p(ja,a[DDLA_NB_],a[DDLA_CSRC_],h->npcols_)==indxg2p(jb,b[DDLA_NB_],b[DDLA_CSRC_],h->npcols_);
}

// This rank's part of the rows x cols stored region at zero-based (i,j).
template <typename T> struct Slab { const T* p; int ld, rows, cols; };
template <typename T>
Slab<T> slab(DdlaHandle_t h, const T* x, const int* d, int i, int j, int rows, int cols) {
    const int pr=h->nprows_,pc=h->npcols_;
    const int r0=num_loc(i,d[DDLA_MB_],h->myprow_,d[DDLA_RSRC_],pr);
    const int c0=num_loc(j,d[DDLA_NB_],h->mypcol_,d[DDLA_CSRC_],pc);
    const int lr=num_loc(i+rows,d[DDLA_MB_],h->myprow_,d[DDLA_RSRC_],pr)-r0;
    const int lc=num_loc(j+cols,d[DDLA_NB_],h->mypcol_,d[DDLA_CSRC_],pc)-c0;
    return {lr>0 && lc>0 ? x+r0+std::size_t(c0)*d[DDLA_LLD_] : x,d[DDLA_LLD_],lr,lc};
}

// A transposed input with its zero-based stored origin and coefficient.
template <typename T> struct Operand { const T* x; const int* d; int i, j; char op; T scale; };

// On a square grid, swapping the block axes maps a whole local slab to one
// owner. Origin/source shifts only rotate that map. A second operand must be
// co-located with the first: one geam pre-adds both slabs into one message.
template <DdlaBackend Backend, typename T>
void transpose_square(DdlaHandle_t h, int m, int n, const Operand<T>& a, const Operand<T>* b,
    T* c, const int* dc, int ic, int jc, int ldc)
{
    const int pr=h->nprows_,pc=h->npcols_,rank=h->myid;
    const Slab<T> sa=slab(h,a.x,a.d,a.i,a.j,n,m);
    const Slab<T> sb=b ? slab(h,b->x,b->d,b->i,b->j,n,m) : Slab<T>{nullptr,1,0,0};
    const int c_row0=num_loc(ic,dc[DDLA_MB_],h->myprow_,dc[DDLA_RSRC_],pr);
    const int c_col0=num_loc(jc,dc[DDLA_NB_],h->mypcol_,dc[DDLA_CSRC_],pc);
    const int mr=num_loc(ic+m,dc[DDLA_MB_],h->myprow_,dc[DDLA_RSRC_],pr)-c_row0;
    const int nc=num_loc(jc+n,dc[DDLA_NB_],h->mypcol_,dc[DDLA_CSRC_],pc)-c_col0;
    const bool snapshot_local=overlaps(sa.p,sa.ld,sa.rows,sa.cols,c,ldc,mr,nc) ||
        (b && overlaps(sb.p,sb.ld,sb.rows,sb.cols,c,ldc,mr,nc));
    const auto peers=transpose_peers(h,a.d,a.i,a.j,dc,ic,jc);
    const std::size_t scount=std::size_t(sa.rows)*sa.cols,rcount=std::size_t(mr)*nc;
    const auto combine=[&](T* dst, int rows, int cols, int ldd) {
        geam<Backend>(h,a.op,b?b->op:'N',rows,cols,a.scale,sa.p,sa.ld,
            b?b->scale:T(0),sb.p,sb.ld,dst,ldd);
    };
    if (peers.dest==rank) {
        if (!rcount) return;
        if (!snapshot_local) {
            combine(c,mr,nc,ldc);
            return;
        }
    }
    vector<T> send(h,scount);
    if (scount) combine(send.data(),sa.cols,sa.rows,sa.cols);
    if (peers.dest==rank) {
        DDLA_CHECK(ddlaMemcpy2D(h,c,std::size_t(ldc)*sizeof(T),send.data(),
            std::size_t(mr)*sizeof(T),std::size_t(mr)*sizeof(T),nc));
    } else {
        // Compact prepared operands receive directly; padded C needs one
        // contiguous receive followed by a pitched copy. Pack before
        // receiving so a same-base input/output alias is also safe.
        const bool compact=ldc==mr;
        vector<T> receive(h,compact?0:rcount);
        T* dst=compact?c:receive.data();
        commSendrecv<Backend>(h,CommScope::Grid,send.data(),scount,peers.dest,dst,rcount,peers.source);
        if (!compact && rcount)
            DDLA_CHECK(ddlaMemcpy2D(h,c,std::size_t(ldc)*sizeof(T),dst,
                std::size_t(mr)*sizeof(T),std::size_t(mr)*sizeof(T),nc));
    }
}

template <DdlaBackend Backend, typename T>
void transpose(DdlaHandle_t h, char op, int m, int n,
    const T* a, const int* da, int ia, int ja,
    T* c, const int* dc, int ic, int jc, int ldc)
{
    if (square_route(h,m,n)) {
        transpose_square<Backend,T>(h,m,n,Operand<T>{a,da,ia,ja,op,T(1)},nullptr,c,dc,ic,jc,ldc);
        return;
    }
    const int pr=h->nprows_,pc=h->npcols_,rank=h->myid,np=pr*pc;
    const int c_row0=num_loc(ic,dc[DDLA_MB_],h->myprow_,dc[DDLA_RSRC_],pr);
    const int c_col0=num_loc(jc,dc[DDLA_NB_],h->mypcol_,dc[DDLA_CSRC_],pc);
    const int ar0=num_loc(ia,da[DDLA_MB_],h->myprow_,da[DDLA_RSRC_],pr);
    const int ac0=num_loc(ja,da[DDLA_NB_],h->mypcol_,da[DDLA_CSRC_],pc);
    const int rows=num_loc(ia+n,da[DDLA_MB_],h->myprow_,da[DDLA_RSRC_],pr)-ar0;
    const int cols=num_loc(ja+m,da[DDLA_NB_],h->mypcol_,da[DDLA_CSRC_],pc)-ac0;
    const int mr=num_loc(ic+m,dc[DDLA_MB_],h->myprow_,dc[DDLA_RSRC_],pr)-c_row0;
    const int nc=num_loc(jc+n,dc[DDLA_NB_],h->mypcol_,dc[DDLA_CSRC_],pc)-c_col0;
    const T* input=rows>0 && cols>0 ? a+ar0+std::size_t(ac0)*da[DDLA_LLD_] : a;
    const bool snapshot_local=overlaps(input,da[DDLA_LLD_],rows,cols,c,ldc,mr,nc);
    struct Transfer { CopyBlock block; int peer; };
    std::vector<CopyBlock> local;
    std::vector<Transfer> sends,receives;
    for (int i=0; i<m; ) {
        const int nr=std::min(m-i,dc[DDLA_MB_]-(ic+i)%dc[DDLA_MB_]);
        for (int j=0; j<n; ) {
            const int nc=std::min(n-j,dc[DDLA_NB_]-(jc+j)%dc[DDLA_NB_]);
            const int ar=ia+j,ac=ja+i,cr=ic+i,cc=jc+j;
            const int src=h->rc_to_rank(indxg2p(ar,da[DDLA_MB_],da[DDLA_RSRC_],pr),
                                       indxg2p(ac,da[DDLA_NB_],da[DDLA_CSRC_],pc));
            const int dst=h->rc_to_rank(indxg2p(cr,dc[DDLA_MB_],dc[DDLA_RSRC_],pr),
                                       indxg2p(cc,dc[DDLA_NB_],dc[DDLA_CSRC_],pc));
            CopyBlock b{nc,nr,0,0,da[DDLA_LLD_],ldc};
            if (src==rank) b.src=indxg2l(ar,da[DDLA_MB_],pr)+std::size_t(indxg2l(ac,da[DDLA_NB_],pc))*b.lds;
            if (dst==rank) b.dst=indxg2l(cr,dc[DDLA_MB_],pr)-c_row0+
                std::size_t(indxg2l(cc,dc[DDLA_NB_],pc)-c_col0)*ldc;
            if (src==rank && dst==rank) local.push_back(b);
            else if (src==rank) sends.push_back({b,dst});
            else if (dst==rank) receives.push_back({b,src});
            j+=nc;
        }
        i+=nr;
    }
    // Stable ordering preserves the same global rectangle order within a peer.
    const auto peer_less=[](const Transfer& a,const Transfer& b) { return a.peer<b.peer; };
    std::stable_sort(sends.begin(),sends.end(),peer_less);
    std::stable_sort(receives.begin(),receives.end(),peer_less);
    std::vector<int> sc(np,0),rc(np,0),sd(np,0),rd(np,0);
    std::vector<CopyBlock> pack,scatter;
    std::size_t ns=0,nr=0;
    for (const auto& t:sends) {
        const auto& b=t.block;
        const std::size_t size=std::size_t(b.rows)*b.cols;
        if (size>INT_MAX-ns) throw std::overflow_error("pgeadd: MPI send count overflow");
        pack.push_back({b.rows,b.cols,b.src,ns,b.lds,b.cols});
        ns+=size; sc[t.peer]+=static_cast<int>(size);
    }
    for (const auto& t:receives) {
        const auto& b=t.block;
        const std::size_t size=std::size_t(b.rows)*b.cols;
        if (size>INT_MAX-nr) throw std::overflow_error("pgeadd: MPI receive count overflow");
        scatter.push_back({b.cols,b.rows,nr,b.dst,b.cols,ldc});
        nr+=size; rc[t.peer]+=static_cast<int>(size);
    }
    std::vector<CopyBlock> local_scatter;
    if (snapshot_local) {
        // Only local overlap needs a snapshot: every remote input is already
        // packed before C is touched. Appended local data is outside MPI counts.
        for (const auto& b:local) {
            const int rows=b.cols,cols=b.rows;
            pack.push_back({b.rows,b.cols,b.src,ns,b.lds,rows});
            local_scatter.push_back({rows,cols,ns,b.dst,rows,ldc});
            ns+=std::size_t(rows)*cols;
        }
    }
    for (int p=1;p<np;++p) { sd[p]=sd[p-1]+sc[p-1]; rd[p]=rd[p-1]+rc[p-1]; }
    vector<T> send(h),receive(h);
    send.resize(ns); receive.resize(nr);
    copy_batch<Backend>(h,op,a,send.data(),pack);
    if (snapshot_local) copy_batch<Backend>(h,'N',send.data(),c,local_scatter);
    else copy_batch<Backend>(h,op,a,c,local);
    // All ranks participate, including those with only local or zero data.
    commAlltoallv<Backend>(h,CommScope::Grid,np,send.data(),sc.data(),sd.data(),receive.data(),rc.data(),rd.data());
    copy_batch<Backend>(h,'N',receive.data(),c,scatter);
}

} // namespace

template <DdlaBackend Backend, typename T>
void pgeadd(const DdlaHandle_t& h, char transa, char transb, int m, int n,
    const T& alpha, const T* a, int ia, int ja, const int* da,
    const T& beta, const T* b, int ib, int jb, const int* db,
    T* c, int ic, int jc, const int* dc)
{
    detail::require_backend<Backend>(h,"pgeadd");
    const char ta=operation(transa),tb=operation(transb);
    check_desc(da,h); check_desc(db,h); check_desc(dc,h);
    check_region(da,ia,ja,ta=='N'?m:n,ta=='N'?n:m);
    check_region(db,ib,jb,tb=='N'?m:n,tb=='N'?n:m);
    check_region(dc,ic,jc,m,n);
    check_alignment(h,ta,da,ia-1,ja-1,dc,ic-1,jc-1);
    check_alignment(h,tb,db,ib-1,jb-1,dc,ic-1,jc-1);
    if (m==0 || n==0) return;
    const int pr=h->nprows_,pc=h->npcols_,r=h->myprow_,col=h->mypcol_;
    const int row0=num_loc(ic-1,dc[DDLA_MB_],r,dc[DDLA_RSRC_],pr);
    const int col0=num_loc(jc-1,dc[DDLA_NB_],col,dc[DDLA_CSRC_],pc);
    const int mr=num_loc(ic-1+m,dc[DDLA_MB_],r,dc[DDLA_RSRC_],pr)-row0;
    const int nc=num_loc(jc-1+n,dc[DDLA_NB_],col,dc[DDLA_CSRC_],pc)-col0;
    T* cv=mr>0 && nc>0 ? c+row0+std::size_t(col0)*dc[DDLA_LLD_] : c;
    // Unit transpose copies can write C directly. Local aliases snapshot
    // their input first; this never changes communicator participation.
    const bool copy_a=alpha==T(1) && beta==T(0);
    const bool copy_b=alpha==T(0) && beta==T(1);
    if (copy_a || copy_b) {
        const T* x=copy_a?a:b; const int* dx=copy_a?da:db;
        const int ix=copy_a?ia:ib,jx=copy_a?ja:jb;
        const char op=copy_a?ta:tb;
        if (op!='N') {
            transpose<Backend>(h,op,m,n,x,dx,ix-1,jx-1,cv,dc,ic-1,jc-1,dc[DDLA_LLD_]);
            return;
        }
    }
    const bool square=square_route(h,m,n);
    // Co-located transposed operands pre-add on the sender, so one message
    // delivers alpha*op(A)+beta*op(B) straight into C.
    if (square && ta!='N' && tb!='N' && alpha!=T(0) && beta!=T(0) &&
        colocated(h,da,ia-1,ja-1,db,ib-1,jb-1)) {
        const Operand<T> ob{b,db,ib-1,jb-1,tb,beta};
        transpose_square<Backend>(h,m,n,Operand<T>{a,da,ia-1,ja-1,ta,alpha},&ob,
            cv,dc,ic-1,jc-1,dc[DDLA_LLD_]);
        return;
    }
    vector<T> aa(h),bb(h);
    int lda=1,ldb=1;
    char opa=ta,opb=tb;
    auto add=[&](const T* av,const T* bv) {
        if (mr==0 || nc==0) return;
        if (alpha!=T(0))
            geam<Backend>(h,opa,opb,mr,nc,alpha,av,lda,beta,bv,ldb,cv,dc[DDLA_LLD_]);
        else if (beta!=T(0))
            geam<Backend>(h,opb,'N',mr,nc,beta,bv,ldb,T(0),static_cast<const T*>(nullptr),1,cv,dc[DDLA_LLD_]);
        else
            geam<Backend>(h,'N','N',mr,nc,T(0),cv,dc[DDLA_LLD_],T(0),static_cast<const T*>(nullptr),1,cv,dc[DDLA_LLD_]);
    };
    auto local=[&](const T* x,const int* dx,int ix,int jx,const T& scale,vector<T>& buf,int& ld)->const T* {
        if (scale==T(0) || mr==0 || nc==0) return nullptr;
        const int xr=num_loc(ix-1,dx[DDLA_MB_],r,dx[DDLA_RSRC_],pr);
        const int xc=num_loc(jx-1,dx[DDLA_NB_],col,dx[DDLA_CSRC_],pc);
        ld=dx[DDLA_LLD_];
        const T* input=x+xr+std::size_t(xc)*ld;
        if ((input==cv && ld==dc[DDLA_LLD_]) ||
            !overlaps(input,ld,mr,nc,cv,dc[DDLA_LLD_],mr,nc)) return input;
        // Shifted aliases must be consumed before either operand writes C.
        buf.resize(std::size_t(mr)*nc);
        DDLA_CHECK(ddlaMemcpy2D(h,buf.data(),std::size_t(mr)*sizeof(T),input,
            std::size_t(ld)*sizeof(T),std::size_t(mr)*sizeof(T),nc));
        ld=mr;
        return buf.data();
    };
    auto prepare=[&](const T* x,const int* dx,int ix,int jx,char& op,const T& scale,vector<T>& buf,int& ld)->const T* {
        if (op=='N') return local(x,dx,ix,jx,scale,buf,ld);
        if (scale==T(0)) return nullptr;
        if (square) {
            // The slab stays untransposed: add() applies op while reading it.
            const auto peers=transpose_peers(h,dx,ix-1,jx-1,dc,ic-1,jc-1);
            const Slab<T> s=slab(h,x,dx,ix-1,jx-1,n,m);
            const auto copy=[&](T* dst) {
                DDLA_CHECK(ddlaMemcpy2D(h,dst,std::size_t(s.rows)*sizeof(T),s.p,
                    std::size_t(s.ld)*sizeof(T),std::size_t(s.rows)*sizeof(T),s.cols));
            };
            if (peers.dest==h->myid) {
                if (mr==0 || nc==0) return nullptr;
                ld=s.ld;
                if (!overlaps(s.p,s.ld,s.rows,s.cols,cv,dc[DDLA_LLD_],mr,nc)) return s.p;
                // In-place geam only permits N operands, so overlap needs a copy.
                buf.resize(std::size_t(s.rows)*s.cols);
                copy(buf.data());
                ld=s.rows;
                return buf.data();
            }
            // A contiguous slab is sent from the input itself.
            const std::size_t scount=std::size_t(s.rows)*s.cols,rcount=std::size_t(mr)*nc;
            vector<T> packed(h);
            const T* send=s.p;
            if (scount && s.ld!=s.rows) {
                packed.resize(scount);
                copy(packed.data());
                send=packed.data();
            }
            buf.resize(rcount);
            commSendrecv<Backend>(h,CommScope::Grid,send,scount,peers.dest,buf.data(),rcount,peers.source);
            ld=std::max(1,nc);
            return buf.data();
        }
        ld=std::max(1,mr);
        buf.resize(std::size_t(mr)*nc);
        transpose<Backend>(h,op,m,n,x,dx,ix-1,jx-1,buf.data(),dc,ic-1,jc-1,ld);
        op='N';
        return buf.data();
    };
    const T* av=prepare(a,da,ia,ja,opa,alpha,aa,lda);
    const T* bv=prepare(b,db,ib,jb,opb,beta,bb,ldb);
    add(av,bv);
}

#define INSTANTIATE_PGEADD(B,T) \
    template void pgeadd<B,T>(const DdlaHandle_t&,char,char,int,int, \
        const T&,const T*,int,int,const int*,const T&,const T*,int,int,const int*,T*,int,int,const int*)
#if DDLA_HAS_CPU
INSTANTIATE_PGEADD(DdlaBackend::CPU,float);
INSTANTIATE_PGEADD(DdlaBackend::CPU,double);
INSTANTIATE_PGEADD(DdlaBackend::CPU,std::complex<float>);
INSTANTIATE_PGEADD(DdlaBackend::CPU,std::complex<double>);
#endif
#if DDLA_HAS_GPU
INSTANTIATE_PGEADD(DdlaBackend::GPU,float);
INSTANTIATE_PGEADD(DdlaBackend::GPU,double);
INSTANTIATE_PGEADD(DdlaBackend::GPU,std::complex<float>);
INSTANTIATE_PGEADD(DdlaBackend::GPU,std::complex<double>);
#endif
#undef INSTANTIATE_PGEADD
} // namespace ddla
