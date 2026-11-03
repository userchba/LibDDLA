#include "ddla_connector.h"
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "ddla_desc.h"
#include "ddla_stream.h"
#include "helpers.h"
#include "require_backend.h"
#include "comm_traits.h"
#include "check_agree.h"
#include "plan_util.h"
#include "pgeadd_impl.h"
#include "BLAS/blas_impl.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ddla { namespace pgeadd_impl { namespace {

// One rectangle copied (and optionally transposed) between a matrix and a
// packed buffer; packing and scattering share the copy kernel.
struct CopyBlock {
    int rows, cols;
    std::size_t src, dst;
    int lds, ldd;
};

// One rectangle of the redistribution path and the rank on its other end.
struct Transfer {
    CopyBlock block;
    int peer;
};

// Stable counting sort of the count transfers in v by peer, through the
// scratch list (std::stable_sort allocates its own buffer).
void sort_by_peer(Transfer* v, std::size_t count, Transfer* scratch, int* cursor, int np)
{
    std::fill(cursor,cursor+np+1,0);
    for (std::size_t i=0;i<count;++i) ++cursor[v[i].peer+1];
    for (int p=0;p<np;++p) cursor[p+1]+=cursor[p];
    for (std::size_t i=0;i<count;++i) scratch[cursor[v[i].peer]++]=v[i];
    std::copy(scratch,scratch+count,v);
}

#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
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
        if (r<b.cols && c<b.rows) dst[b.dst+r+std::size_t(c)*b.ldd]=detail::device_conj(tile[threadIdx.x][threadIdx.y],conj);
        __syncthreads();
    }
}
#endif

// Copy the count rectangles of blocks.  A GPU handle first uploads the list
// to device_blocks; the source list is pageable host memory, so the upload has
// read it when ddlaMemcpy returns and the list may be refilled right away.
template <DdlaBackend Backend, typename T>
void copy_batch(DdlaHandle_t h, char op, const T* src, T* dst, const CopyBlock* blocks, std::size_t count,
                [[maybe_unused]] CopyBlock* device_blocks)
{
    if (count==0) return;
    if constexpr (Backend == DdlaBackend::CPU) {
        for (std::size_t i=0;i<count;++i) {
            const CopyBlock& b=blocks[i];
            blas_impl::geam_run<Backend>(h,op,'N',op=='N'?b.rows:b.cols,op=='N'?b.cols:b.rows,
                T(1),src+b.src,b.lds,T(0),static_cast<const T*>(nullptr),1,dst+b.dst,b.ldd);
        }
    } else {
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
        DDLA_CHECK(ddlaMemcpy(h,device_blocks,blocks,count*sizeof(CopyBlock)));
        using DT = detail::DeviceScalarT<T>;
        copy_blocks<DT><<<dim3(count),dim3(16,16),0,static_cast<gpuStream_t>(h->stream)>>>(
            reinterpret_cast<const DT*>(src),reinterpret_cast<DT*>(dst),device_blocks,op!='N',op=='C');
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


// On a square grid the transposed local slab has exactly one destination.
void transpose_peers(DdlaHandle_t h, const int* a, int ia, int ja,
                     const int* c, int ic, int jc, int& dest, int& source) {
    const int p=h->nprows_;
    const long long dr=static_cast<long long>(ic)/c[DDLA_MB_]-ja/a[DDLA_NB_]+c[DDLA_RSRC_]-a[DDLA_CSRC_];
    const long long dc=static_cast<long long>(jc)/c[DDLA_NB_]-ia/a[DDLA_MB_]+c[DDLA_CSRC_]-a[DDLA_RSRC_];
    const auto wrap=[&](long long x) { const int y=x%p;return y<0?y+p:y; };
    dest=h->rc_to_rank(wrap(h->mypcol_+dr),wrap(h->myprow_+dc));
    source=h->rc_to_rank(wrap(h->mypcol_-dc),wrap(h->myprow_-dr));
}

// The square-grid route sends each local slab as one message with an int count.
bool square_route(DdlaHandle_t h, int m, int n) {
    return h->nprows_==h->npcols_ && std::size_t(m)*n<=INT_MAX;
}

// Element counts of the operand staging of calls of at most m x n, in
// carving order: send | receive | packed | aa | bb.  Every bound is taken from
// C's block sizes: the alignment check makes A's and B's blocks equal to them
// after N/T/C.  A transposed operand's stored region is n x m, its local part
// at most locmax(n,nb,pr) x locmax(m,mb,pc); an operand that is not transposed
// shares C's owners, so its local part is C's.  The buffers a path can use are
// counted as the maximum over the paths the flags allow (the choice between
// them also depends on pointer aliasing, which a size for a loop of calls
// cannot know).  Every count is monotone in m and n.
void staging_counts(const DdlaHandle_t& h, char ta, char tb, bool use_a, bool use_b,
                    int m, int n, const int* dc, std::size_t (&count)[5])
{
    const int pr=h->nprows_,pc=h->npcols_,mb=dc[DDLA_MB_],nb=dc[DDLA_NB_];
    const std::size_t cloc=std::size_t(detail::locmax(m,mb,pr))*detail::locmax(n,nb,pc);
    const std::size_t slab_t=std::size_t(detail::locmax(n,nb,pr))*detail::locmax(m,mb,pc);
    const bool a_t=use_a && ta!='N', b_t=use_b && tb!='N';
    const std::size_t slab_a=ta=='N'?cloc:slab_t, slab_b=tb=='N'?cloc:slab_t;
    std::size_t send=0,receive=0,packed=0;
    if (a_t) { send=std::max(send,slab_a); receive=cloc; packed=std::max(packed,slab_a); }
    if (b_t) { send=std::max(send,slab_b); receive=cloc; packed=std::max(packed,slab_b); }
    count[0]=send;
    count[1]=receive;
    count[2]=packed;
    count[3]=use_a ? (a_t?std::max(slab_a,cloc):cloc) : 0;
    count[4]=use_b ? (b_t?std::max(slab_b,cloc):cloc) : 0;
}

// Whether a call of at most m x n can take the redistribution route (a
// transposed operand on a non-square grid, or a slab too large for one MPI
// count), and its schedule bounds: each rank is the source of at most `src`
// A tiles (the ones it owns) and the destination of at most `dst` C tiles.
bool redistribution(const DdlaHandle_t& h, bool transposed, int m, int n, const int* dc,
                    std::size_t& src, std::size_t& dst)
{
    src=dst=0;
    if (!transposed || square_route(h,m,n)) return false;
    const int pr=h->nprows_,pc=h->npcols_;
    const std::size_t tm=detail::tilesmax(m,dc[DDLA_MB_]),tn=detail::tilesmax(n,dc[DDLA_NB_]);
    src=((tn+pr-1)/pr)*((tm+pc-1)/pc);
    dst=((tm+pr-1)/pr)*((tn+pc-1)/pc);
    return true;
}

// Host bytes of the redistribution schedule, in carving order: local |
// local_scatter (min(src, dst) each) | sends (src) | receives (dst) | ordered
// (the sort scratch) | pack (src) | scatter (dst) | sc | rc | sd | rd (np each)
// | cursor (np + 1).
std::size_t schedule_bytes(std::size_t src, std::size_t dst, int np)
{
    const std::size_t both=std::min(src,dst);
    return 2*detail::padded_bytes<CopyBlock>(both)+detail::padded_bytes<Transfer>(src)
         + detail::padded_bytes<Transfer>(dst)+detail::padded_bytes<Transfer>(std::max(src,dst))
         + detail::padded_bytes<CopyBlock>(src)+detail::padded_bytes<CopyBlock>(dst)
         + 4*detail::padded_bytes<int>(std::size_t(np))+detail::padded_bytes<int>(std::size_t(np)+1);
}

// Aligned operands already share block sizes and offsets with C, so equal
// origin owners make two stored regions share every owner and local position.
bool colocated(DdlaHandle_t h, const int* a, int ia, int ja, const int* b, int ib, int jb) {
    return indxg2p(ia,a[DDLA_MB_],a[DDLA_RSRC_],h->nprows_)==indxg2p(ib,b[DDLA_MB_],b[DDLA_RSRC_],h->nprows_) &&
           indxg2p(ja,a[DDLA_NB_],a[DDLA_CSRC_],h->npcols_)==indxg2p(jb,b[DDLA_NB_],b[DDLA_CSRC_],h->npcols_);
}

// This rank's part of the rows x cols stored region at zero-based (i,j).
template <typename T>
const T* slab(DdlaHandle_t h, const T* x, const int* d, int i, int j, int rows, int cols,
              int& ld, int& lr, int& lc) {
    const int pr=h->nprows_,pc=h->npcols_;
    const int r0=num_loc(i,d[DDLA_MB_],h->myprow_,d[DDLA_RSRC_],pr);
    const int c0=num_loc(j,d[DDLA_NB_],h->mypcol_,d[DDLA_CSRC_],pc);
    lr=num_loc(i+rows,d[DDLA_MB_],h->myprow_,d[DDLA_RSRC_],pr)-r0;
    lc=num_loc(j+cols,d[DDLA_NB_],h->mypcol_,d[DDLA_CSRC_],pc)-c0;
    ld=d[DDLA_LLD_];
    return lr>0 && lc>0 ? x+r0+std::size_t(c0)*ld : x;
}


// On a square grid, swapping the block axes maps a whole local slab to one
// owner. Origin/source shifts only rotate that map. A second operand must be
// co-located with the first: one geam pre-adds both slabs into one message.
// send / receive are the staging buffers of pgeadd_run's scratch.
template <DdlaBackend Backend, typename T>
void transpose_square(DdlaHandle_t h, T* send, T* receive, int m, int n,
    char opa, const T* a, const int* da, int ia, int ja, const T& alpha,
    bool use_b, char opb, const T* b, const int* db, int ib, int jb, const T& beta,
    T* c, const int* dc, int ic, int jc, int ldc)
{
    const int pr=h->nprows_,pc=h->npcols_,rank=h->myid;
    int lda, ar, ac, ldb=1, br=0, bc=0;
    const T* sa=slab(h,a,da,ia,ja,n,m,lda,ar,ac);
    const T* sb=use_b ? slab(h,b,db,ib,jb,n,m,ldb,br,bc) : nullptr;
    const int c_row0=num_loc(ic,dc[DDLA_MB_],h->myprow_,dc[DDLA_RSRC_],pr);
    const int c_col0=num_loc(jc,dc[DDLA_NB_],h->mypcol_,dc[DDLA_CSRC_],pc);
    const int mr=num_loc(ic+m,dc[DDLA_MB_],h->myprow_,dc[DDLA_RSRC_],pr)-c_row0;
    const int nc=num_loc(jc+n,dc[DDLA_NB_],h->mypcol_,dc[DDLA_CSRC_],pc)-c_col0;
    const bool snapshot_local=detail::matrix_overlaps(sa,lda,ar,ac,c,ldc,mr,nc) ||
        (use_b && detail::matrix_overlaps(sb,ldb,br,bc,c,ldc,mr,nc));
    int dest, source;
    transpose_peers(h,da,ia,ja,dc,ic,jc,dest,source);
    const std::size_t scount=std::size_t(ar)*ac,rcount=std::size_t(mr)*nc;
    const auto combine=[&](T* dst, int rows, int cols, int ldd) {
        blas_impl::geam_run<Backend>(h,opa,use_b?opb:'N',rows,cols,alpha,sa,lda,
            use_b?beta:T(0),sb,ldb,dst,ldd);
    };
    if (dest==rank) {
        if (!rcount) return;
        if (!snapshot_local) {
            combine(c,mr,nc,ldc);
            return;
        }
    }
    if (scount) combine(send,ac,ar,ac);
    if (dest==rank) {
        DDLA_CHECK(ddlaMemcpy2D(h,c,std::size_t(ldc)*sizeof(T),send,
            std::size_t(mr)*sizeof(T),std::size_t(mr)*sizeof(T),nc));
    } else {
        // Compact prepared operands receive directly; padded C needs one
        // contiguous receive followed by a pitched copy. Pack before
        // receiving so a same-base input/output alias is also safe.
        const bool compact=ldc==mr;
        T* dst=compact?c:receive;
        commSendrecv<Backend>(h,CommScope::Grid,send,scount,dest,dst,rcount,source);
        if (!compact && rcount)
            DDLA_CHECK(ddlaMemcpy2D(h,c,std::size_t(ldc)*sizeof(T),dst,
                std::size_t(mr)*sizeof(T),std::size_t(mr)*sizeof(T),nc));
    }
}

// The transposed copy op(A) -> C of one operand.  send / receive are the
// staging buffers and device_blocks the device schedule of pgeadd_run's
// backend scratch; the host schedule is carved from host (host_bytes).
template <DdlaBackend Backend, typename T>
void transpose(DdlaHandle_t h, T* send, T* receive, CopyBlock* device_blocks,
    char* host, [[maybe_unused]] std::size_t host_bytes, char op, int m, int n,
    const T* a, const int* da, int ia, int ja,
    T* c, const int* dc, int ic, int jc, int ldc)
{
    if (square_route(h,m,n)) {
        transpose_square<Backend,T>(h,send,receive,m,n,op,a,da,ia,ja,T(1),false,'N',nullptr,nullptr,0,0,T(0),c,dc,ic,jc,ldc);
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
    const bool snapshot_local=detail::matrix_overlaps(input,da[DDLA_LLD_],rows,cols,c,ldc,mr,nc);

    // The schedule lists, in the order schedule_bytes counts them.
    std::size_t src=0,dst=0;
    redistribution(h,true,m,n,dc,src,dst);
    const std::size_t both=std::min(src,dst);
    char* hw=host;
    CopyBlock* const local=detail::take<CopyBlock>(hw,both);
    CopyBlock* const local_scatter=detail::take<CopyBlock>(hw,both);
    Transfer* const sends=detail::take<Transfer>(hw,src);
    Transfer* const receives=detail::take<Transfer>(hw,dst);
    Transfer* const ordered=detail::take<Transfer>(hw,std::max(src,dst));
    CopyBlock* const pack=detail::take<CopyBlock>(hw,src);
    CopyBlock* const scatter=detail::take<CopyBlock>(hw,dst);
    int* const sc=detail::take<int>(hw,std::size_t(np));
    int* const rc=detail::take<int>(hw,std::size_t(np));
    int* const sd=detail::take<int>(hw,std::size_t(np));
    int* const rd=detail::take<int>(hw,std::size_t(np));
    int* const cursor=detail::take<int>(hw,std::size_t(np)+1);
    assert(std::size_t(hw-host)<=host_bytes);

    std::size_t nlocal=0,nsends=0,nreceives=0;
    for (int i=0; i<m; ) {
        const int nr=std::min(m-i,dc[DDLA_MB_]-(ic+i)%dc[DDLA_MB_]);
        for (int j=0; j<n; ) {
            const int nc=std::min(n-j,dc[DDLA_NB_]-(jc+j)%dc[DDLA_NB_]);
            const int ar=ia+j,ac=ja+i,cr=ic+i,cc=jc+j;
            const int src_rank=h->rc_to_rank(indxg2p(ar,da[DDLA_MB_],da[DDLA_RSRC_],pr),
                                            indxg2p(ac,da[DDLA_NB_],da[DDLA_CSRC_],pc));
            const int dst_rank=h->rc_to_rank(indxg2p(cr,dc[DDLA_MB_],dc[DDLA_RSRC_],pr),
                                            indxg2p(cc,dc[DDLA_NB_],dc[DDLA_CSRC_],pc));
            CopyBlock b{nc,nr,0,0,da[DDLA_LLD_],ldc};
            if (src_rank==rank) b.src=indxg2l(ar,da[DDLA_MB_],pr)+std::size_t(indxg2l(ac,da[DDLA_NB_],pc))*b.lds;
            if (dst_rank==rank) b.dst=indxg2l(cr,dc[DDLA_MB_],pr)-c_row0+
                std::size_t(indxg2l(cc,dc[DDLA_NB_],pc)-c_col0)*ldc;
            if (src_rank==rank && dst_rank==rank) local[nlocal++]=b;
            else if (src_rank==rank) sends[nsends++]={b,dst_rank};
            else if (dst_rank==rank) receives[nreceives++]={b,src_rank};
            j+=nc;
        }
        i+=nr;
    }
    assert(nlocal<=both && nsends<=src && nreceives<=dst);
    // Stable ordering preserves the same global rectangle order within a peer.
    sort_by_peer(sends,nsends,ordered,cursor,np);
    sort_by_peer(receives,nreceives,ordered,cursor,np);
    std::fill(sc,sc+np,0); std::fill(rc,rc+np,0); std::fill(sd,sd+np,0); std::fill(rd,rd+np,0);
    std::size_t npack=0,nscatter=0,nlocal_scatter=0;
    std::size_t ns=0,nr=0;
    for (std::size_t i=0;i<nsends;++i) {
        const Transfer& t=sends[i];
        const auto& b=t.block;
        const std::size_t size=std::size_t(b.rows)*b.cols;
        assert(size<=std::size_t(INT_MAX)-ns);
        pack[npack++]={b.rows,b.cols,b.src,ns,b.lds,b.cols};
        ns+=size; sc[t.peer]+=static_cast<int>(size);
    }
    for (std::size_t i=0;i<nreceives;++i) {
        const Transfer& t=receives[i];
        const auto& b=t.block;
        const std::size_t size=std::size_t(b.rows)*b.cols;
        assert(size<=std::size_t(INT_MAX)-nr);
        scatter[nscatter++]={b.cols,b.rows,nr,b.dst,b.cols,ldc};
        nr+=size; rc[t.peer]+=static_cast<int>(size);
    }
    if (snapshot_local) {
        // Only local overlap needs a snapshot: every remote input is already
        // packed before C is touched. Appended local data is outside MPI counts.
        for (std::size_t i=0;i<nlocal;++i) {
            const CopyBlock& b=local[i];
            const int rows=b.cols,cols=b.rows;
            pack[npack++]={b.rows,b.cols,b.src,ns,b.lds,rows};
            local_scatter[nlocal_scatter++]={rows,cols,ns,b.dst,rows,ldc};
            ns+=std::size_t(rows)*cols;
        }
    }
    assert(npack<=src);
    for (int p=1;p<np;++p) { sd[p]=sd[p-1]+sc[p-1]; rd[p]=rd[p-1]+rc[p-1]; }
    // The device copy of the lists: pack (src) | local (both) | scatter (dst).
    CopyBlock* const dpack=device_blocks;
    CopyBlock* const dlocal=dpack?dpack+src:nullptr;
    CopyBlock* const dscatter=dlocal?dlocal+both:nullptr;
    copy_batch<Backend>(h,op,a,send,pack,npack,dpack);
    if (snapshot_local) copy_batch<Backend>(h,'N',send,c,local_scatter,nlocal_scatter,dlocal);
    else copy_batch<Backend>(h,op,a,c,local,nlocal,dlocal);
    // All ranks participate, including those with only local or zero data.
    commAlltoallv<Backend>(h,CommScope::Grid,np,send,sc,sd,receive,rc,rd);
    copy_batch<Backend>(h,'N',receive,c,scatter,nscatter,dscatter);
}

} // namespace

// Component 1.
void check_mpi_counts(const DdlaHandle_t& h,char op,int m,int n,const int* da,int ia,int ja,
    const int* dc,int ic,int jc)
{
    if(op=='N' || m==0 || n==0 || square_route(h,m,n))return;
    std::size_t sent=0,received=0;
    for(int i=0;i<m;) {
        const int rows=std::min(m-i,dc[DDLA_MB_]-(ic+i)%dc[DDLA_MB_]);
        for(int j=0;j<n;) {
            const int cols=std::min(n-j,dc[DDLA_NB_]-(jc+j)%dc[DDLA_NB_]);
            const int source=h->rc_to_rank(indxg2p(ia+j,da[DDLA_MB_],da[DDLA_RSRC_],h->nprows_),
                indxg2p(ja+i,da[DDLA_NB_],da[DDLA_CSRC_],h->npcols_));
            const int target=h->rc_to_rank(indxg2p(ic+i,dc[DDLA_MB_],dc[DDLA_RSRC_],h->nprows_),
                indxg2p(jc+j,dc[DDLA_NB_],dc[DDLA_CSRC_],h->npcols_));
            const auto count=std::size_t(rows)*cols;
            if(source!=target && source==h->myid) {
                if(count>std::size_t(INT_MAX)-sent)throw std::overflow_error("pgeadd: MPI send count overflow");
                sent+=count;
            }
            if(source!=target && target==h->myid) {
                if(count>std::size_t(INT_MAX)-received)throw std::overflow_error("pgeadd: MPI receive count overflow");
                received+=count;
            }
            j+=cols;
        }
        i+=rows;
    }
}

template <DdlaBackend Backend>
void pgeadd_check(const DdlaHandle_t& h, char transa, char transb, int m, int n,
                        const int* da, int ia, int ja,
                        const int* db, int ib, int jb,
                        const int* dc, int ic, int jc, bool use_a, bool use_b)
{
    detail::agree_check(h, "pgeadd", [&](detail::Agree&) {
        detail::require_backend<Backend>(h,"pgeadd");
        transa=operation(transa);
        transb=operation(transb);
        check_desc(da,h); check_desc(db,h); check_desc(dc,h);
        check_region(da,ia,ja,transa=='N'?m:n,transa=='N'?n:m);
        check_region(db,ib,jb,transb=='N'?m:n,transb=='N'?n:m);
        check_region(dc,ic,jc,m,n);
        check_alignment(h,transa,da,ia-1,ja-1,dc,ic-1,jc-1);
        check_alignment(h,transb,db,ib-1,jb-1,dc,ic-1,jc-1);
        if(use_a)check_mpi_counts(h,transa,m,n,da,ia-1,ja-1,dc,ic-1,jc-1);
        if(use_b)check_mpi_counts(h,transb,m,n,db,ib-1,jb-1,dc,ic-1,jc-1);
    });
}

template <DdlaBackend Backend>
void pgeadd_check(const DdlaHandle_t& h, char transa, char transb, int m, int n,
                        const int* da, int ia, int ja,
                        const int* db, int ib, int jb,
                        const int* dc, int ic, int jc)
{
    pgeadd_check<Backend>(h,transa,transb,m,n,da,ia,ja,db,ib,jb,dc,ic,jc,true,true);
}

// Component 2.  Backend: [ send | receive | packed | aa | bb ]
// (staging_counts) and, on a GPU handle that can take the redistribution
// route, the device copy of its schedule [ pack (src) | local (min(src, dst))
// | scatter (dst) ]; host: the schedule itself (schedule_bytes).
template <typename T>
void pgeadd_workspace_size(const DdlaHandle_t& h, char ta, char tb, bool use_a, bool use_b,
                           int m, int n, const int* dc,
                           std::size_t* backend_bytes, std::size_t* host_bytes)
{
    *backend_bytes=0;
    *host_bytes=0;
    if (m==0 || n==0) return;
    std::size_t count[5];
    staging_counts(h,ta,tb,use_a,use_b,m,n,dc,count);
    std::size_t bytes=0;
    for (const std::size_t c : count) bytes+=detail::padded_bytes<T>(c);
    std::size_t src=0,dst=0;
    if (redistribution(h,(use_a && ta!='N') || (use_b && tb!='N'),m,n,dc,src,dst)) {
        if (h->backend==DdlaBackend::GPU)
            bytes+=detail::padded_bytes<CopyBlock>(src+std::min(src,dst)+dst);
        *host_bytes=schedule_bytes(src,dst,h->nprows_*h->npcols_);
    }
    *backend_bytes=bytes;
}

// Component 3.
template <DdlaBackend Backend, typename T>
void pgeadd_run(const DdlaHandle_t& h, char ta, char tb, int m, int n,
    const T& alpha, const T* a, int ia, int ja, const int* da,
    const T& beta, const T* b, int ib, int jb, const int* db,
    T* c, int ic, int jc, const int* dc,
    void* backend_work, [[maybe_unused]] std::size_t backend_bytes,
    void* host_work, std::size_t host_bytes)
{
    try {
        if (m==0 || n==0) return;
        // The scratch, in the order pgeadd_workspace_size counts it, for this call
        // (no larger than the call it was sized for).
        std::size_t count[5];
        staging_counts(h,ta,tb,alpha!=T(0),beta!=T(0),m,n,dc,count);
        char* w=static_cast<char*>(backend_work);
        T* const send=detail::take<T>(w,count[0]);
        T* const receive=detail::take<T>(w,count[1]);
        T* const packed=detail::take<T>(w,count[2]);
        T* const aa=detail::take<T>(w,count[3]);
        T* const bb=detail::take<T>(w,count[4]);
        CopyBlock* device_blocks=nullptr;
        std::size_t src=0,dst=0;
        if (Backend==DdlaBackend::GPU &&
            redistribution(h,(alpha!=T(0) && ta!='N') || (beta!=T(0) && tb!='N'),m,n,dc,src,dst))
            device_blocks=detail::take<CopyBlock>(w,src+std::min(src,dst)+dst);
        assert(std::size_t(w-static_cast<char*>(backend_work))<=backend_bytes);
        char* const host=static_cast<char*>(host_work);

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
                transpose<Backend>(h,send,receive,device_blocks,host,host_bytes,op,m,n,x,dx,ix-1,jx-1,
                    cv,dc,ic-1,jc-1,dc[DDLA_LLD_]);
                return;
            }
        }
        const bool square=square_route(h,m,n);
        // Co-located transposed operands pre-add on the sender, so one message
        // delivers alpha*op(A)+beta*op(B) straight into C.
        if (square && ta!='N' && tb!='N' && alpha!=T(0) && beta!=T(0) &&
            colocated(h,da,ia-1,ja-1,db,ib-1,jb-1)) {
            transpose_square<Backend>(h,send,receive,m,n,ta,a,da,ia-1,ja-1,alpha,true,tb,b,db,ib-1,jb-1,beta,
                cv,dc,ic-1,jc-1,dc[DDLA_LLD_]);
            return;
        }
        int lda=1,ldb=1;
        char opa=ta,opb=tb;
        auto add=[&](const T* av,const T* bv) {
            if (mr==0 || nc==0) return;
            if (alpha!=T(0))
                blas_impl::geam_run<Backend>(h,opa,opb,mr,nc,alpha,av,lda,beta,bv,ldb,cv,dc[DDLA_LLD_]);
            else if (beta!=T(0))
                blas_impl::geam_run<Backend>(h,opb,'N',mr,nc,beta,bv,ldb,T(0),static_cast<const T*>(nullptr),1,cv,dc[DDLA_LLD_]);
            else
                blas_impl::geam_run<Backend>(h,'N','N',mr,nc,T(0),cv,dc[DDLA_LLD_],T(0),static_cast<const T*>(nullptr),1,cv,dc[DDLA_LLD_]);
        };
        auto local=[&](const T* x,const int* dx,int ix,int jx,const T& scale,T* buf,int& ld)->const T* {
            if (scale==T(0) || mr==0 || nc==0) return nullptr;
            const int xr=num_loc(ix-1,dx[DDLA_MB_],r,dx[DDLA_RSRC_],pr);
            const int xc=num_loc(jx-1,dx[DDLA_NB_],col,dx[DDLA_CSRC_],pc);
            ld=dx[DDLA_LLD_];
            const T* input=x+xr+std::size_t(xc)*ld;
            if ((input==cv && ld==dc[DDLA_LLD_]) ||
                !detail::matrix_overlaps(input,ld,mr,nc,cv,dc[DDLA_LLD_],mr,nc)) return input;
            // Shifted aliases must be consumed before either operand writes C.
            DDLA_CHECK(ddlaMemcpy2D(h,buf,std::size_t(mr)*sizeof(T),input,
                std::size_t(ld)*sizeof(T),std::size_t(mr)*sizeof(T),nc));
            ld=mr;
            return buf;
        };
        auto prepare=[&](const T* x,const int* dx,int ix,int jx,char& op,const T& scale,T* buf,int& ld)->const T* {
            if (op=='N') return local(x,dx,ix,jx,scale,buf,ld);
            if (scale==T(0)) return nullptr;
            if (square) {
                // The slab stays untransposed: add() applies op while reading it.
                int dest, source;
                transpose_peers(h,dx,ix-1,jx-1,dc,ic-1,jc-1,dest,source);
                int slab_ld, slab_rows, slab_cols;
                const T* slab_input=slab(h,x,dx,ix-1,jx-1,n,m,slab_ld,slab_rows,slab_cols);
                const auto copy=[&](T* dst) {
                    DDLA_CHECK(ddlaMemcpy2D(h,dst,std::size_t(slab_rows)*sizeof(T),slab_input,
                        std::size_t(slab_ld)*sizeof(T),std::size_t(slab_rows)*sizeof(T),slab_cols));
                };
                if (dest==h->myid) {
                    if (mr==0 || nc==0) return nullptr;
                    ld=slab_ld;
                    if (!detail::matrix_overlaps(slab_input,slab_ld,slab_rows,slab_cols,cv,dc[DDLA_LLD_],mr,nc)) return slab_input;
                    // In-place geam only permits N operands, so overlap needs a copy.
                    copy(buf);
                    ld=slab_rows;
                    return buf;
                }
                // A contiguous slab is sent from the input itself.
                const std::size_t scount=std::size_t(slab_rows)*slab_cols,rcount=std::size_t(mr)*nc;
                const T* slab_send=slab_input;
                if (scount && slab_ld!=slab_rows) {
                    copy(packed);
                    slab_send=packed;
                }
                commSendrecv<Backend>(h,CommScope::Grid,slab_send,scount,dest,buf,rcount,source);
                ld=std::max(1,nc);
                return buf;
            }
            ld=std::max(1,mr);
            transpose<Backend>(h,send,receive,device_blocks,host,host_bytes,op,m,n,x,dx,ix-1,jx-1,
                buf,dc,ic-1,jc-1,ld);
            op='N';
            return buf;
        };
        const T* av=prepare(a,da,ia,ja,opa,alpha,aa,lda);
        const T* bv=prepare(b,db,ib,jb,opb,beta,bb,ldb);
        add(av,bv);
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(h) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(h); } catch (...) {}
        throw;
    }
}

} // namespace pgeadd_impl

template <DdlaBackend Backend, typename T>
void pgeadd(const DdlaHandle_t& h, char transa, char transb, int m, int n,
    const T& alpha, const T* a, int ia, int ja, const int* da,
    const T& beta, const T* b, int ib, int jb, const int* db,
    T* c, int ic, int jc, const int* dc)
{
    pgeadd_impl::pgeadd_check<Backend>(
        h,transa,transb,m,n,da,ia,ja,db,ib,jb,dc,ic,jc,alpha!=T(0),beta!=T(0));
    if (m==0 || n==0) return;
    transa=static_cast<char>(std::toupper(static_cast<unsigned char>(transa)));
    transb=static_cast<char>(std::toupper(static_cast<unsigned char>(transb)));
    std::size_t backend_bytes=0,host_bytes=0;
    pgeadd_impl::pgeadd_workspace_size<T>(h,transa,transb,alpha!=T(0),beta!=T(0),m,n,dc,
        &backend_bytes,&host_bytes);
    vector<char> backend_work(h,backend_bytes);
    std::vector<char> host_work(host_bytes);
    pgeadd_impl::pgeadd_run<Backend,T>(h,transa,transb,m,n,
        alpha,a,ia,ja,da,beta,b,ib,jb,db,c,ic,jc,dc,
        backend_work.data(),backend_bytes,host_work.data(),host_bytes);
}

#define INSTANTIATE_PGEADD(B,T) \
    template void pgeadd<B,T>(const DdlaHandle_t&,char,char,int,int, \
        const T&,const T*,int,int,const int*,const T&,const T*,int,int,const int*,T*,int,int,const int*); \
    template void pgeadd_impl::pgeadd_run<B,T>(const DdlaHandle_t&,char,char,int,int, \
        const T&,const T*,int,int,const int*,const T&,const T*,int,int,const int*,T*,int,int,const int*, \
        void*,std::size_t,void*,std::size_t)
#define INSTANTIATE_PGEADD_CHECK(B) \
    template void pgeadd_impl::pgeadd_check<B>(const DdlaHandle_t&,char,char,int,int, \
        const int*,int,int,const int*,int,int,const int*,int,int); \
    template void pgeadd_impl::pgeadd_check<B>(const DdlaHandle_t&,char,char,int,int, \
        const int*,int,int,const int*,int,int,const int*,int,int,bool,bool)
#define INSTANTIATE_PGEADD_SIZE(T) \
    template void pgeadd_impl::pgeadd_workspace_size<T>(const DdlaHandle_t&,char,char,bool,bool,int,int,const int*, \
        std::size_t*,std::size_t*)
INSTANTIATE_PGEADD_CHECK(DdlaBackend::CPU);
INSTANTIATE_PGEADD(DdlaBackend::CPU,float);
INSTANTIATE_PGEADD(DdlaBackend::CPU,double);
INSTANTIATE_PGEADD(DdlaBackend::CPU,std::complex<float>);
INSTANTIATE_PGEADD(DdlaBackend::CPU,std::complex<double>);
#if DDLA_HAS_GPU
INSTANTIATE_PGEADD_CHECK(DdlaBackend::GPU);
INSTANTIATE_PGEADD(DdlaBackend::GPU,float);
INSTANTIATE_PGEADD(DdlaBackend::GPU,double);
INSTANTIATE_PGEADD(DdlaBackend::GPU,std::complex<float>);
INSTANTIATE_PGEADD(DdlaBackend::GPU,std::complex<double>);
#endif
INSTANTIATE_PGEADD_SIZE(float);
INSTANTIATE_PGEADD_SIZE(double);
INSTANTIATE_PGEADD_SIZE(std::complex<float>);
INSTANTIATE_PGEADD_SIZE(std::complex<double>);
#undef INSTANTIATE_PGEADD_SIZE
#undef INSTANTIATE_PGEADD_CHECK
#undef INSTANTIATE_PGEADD
} // namespace ddla
