#include "ddla_connector.h"
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "phemm_impl.h"
#include "BLAS/hemmt_impl.h"
#include "pgemmt_panel_impl.h"
#include "pgeadd_impl.h"
#include "pblas_workspace.h"
#include "BLAS/blas_impl.h"
#include "ddla_desc.h"
#include "ddla_stream.h"
#include "helpers.h"
#include "comm_traits.h"
#include "check_agree.h"
#include "require_backend.h"
#include "plan_util.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ddla { namespace phemm_impl { namespace {

// The short panel dimension is replicated. Shift the long axis to C's
// source coordinate without changing its global block phase.
template <DdlaBackend Backend, typename T>
const T* shift_panel(const DdlaHandle_t& h, bool rows, int block, int first, int length,
                     int source, int target, int width, const T* input, T* work)
{
    if (source == target) return input;
    const int p = rows ? h->nprows_ : h->npcols_;
    const int r = rows ? h->myprow_ : h->mypcol_;
    const int send = num_loc(first+length,block,r,source,p)-num_loc(first,block,r,source,p);
    const int recv = num_loc(first+length,block,r,target,p)-num_loc(first,block,r,target,p);
    const int shift = (target-source+p)%p;
    commSendrecv<Backend>(h,rows ? CommScope::Col : CommScope::Row,
        input,std::size_t(send)*width,(r+shift)%p,
        work,std::size_t(recv)*width,(r-shift+p)%p);
    return work;
}

// Element counts of phemm's own buffers, in carving order: the panel of A
// in C's distribution (one block wide) | the broadcast diagonal block | the
// transported piece of A or B | the buffer the piece is shifted or
// transposed into | grouped B panel (GPU groups only).
template <typename T,typename TA=T,typename TB=T>
int scratch_counts(const DdlaHandle_t& h, bool left, int m, int n,
                    const int* da, const int* db, const int* dc, std::size_t (&count)[5], bool triangular=false)
{
    const int order=left ? m : n,nb=da[DDLA_MB_];
    const int pr=h->nprows_,pc=h->npcols_,r=h->myprow_,c=h->mypcol_;
    const int along=num_loc(order,nb,left ? r : c,dc[left ? DDLA_RSRC_ : DDLA_CSRC_],left ? pr : pc);
    const int source=num_loc(order,nb,left ? r : c,da[left ? DDLA_RSRC_ : DDLA_CSRC_],left ? pr : pc);
    const int free_b=num_loc(left ? n : m,db[left ? DDLA_NB_ : DDLA_MB_],left ? c : r,
                             db[left ? DDLA_CSRC_ : DDLA_RSRC_],left ? pc : pr);
    const int free_c=num_loc(left ? n : m,dc[left ? DDLA_NB_ : DDLA_MB_],left ? c : r,
                             dc[left ? DDLA_CSRC_ : DDLA_RSRC_],left ? pc : pr);
    // Wide complex panels lose to the original small updates at this size.
    const int kp=!std::is_arithmetic_v<T> && order<=1024 ? std::min(nb,order)
                                                       : pblas::contraction_width(h,nb,order,!std::is_same_v<T,TA> || !std::is_same_v<T,TB>);
    count[0]=std::size_t(kp)*std::max(1,along);
    count[4]=(kp>nb || triangular) ? std::size_t(kp)*std::max(1,free_c) : 0;
    count[1]=std::size_t(nb)*nb;
    count[2]=std::size_t(nb)*std::max({1,along,source,free_b,free_c});
    count[3]=count[2];
    return kp;
}

// The descriptor of the reflected panel pgeadd adds: the sweep's longest
// panel is `order` long and one block wide.
void reflected_panel(const DdlaHandle_t& h, bool left, int order, int nb, int* d)
{
    pblas::describe(d,h,left ? order : nb,left ? nb : order,nb,nb,0,0);
}

// Copy one strictly off-diagonal region. Its stored orientation is selected
// before transport/pgeadd: neither primitive ever receives a diagonal tile
// or a rectangle that crosses into A's unstored triangle.  panel is the raw
// window base and offset the window-local block position; piece / work are
// phemm's buffers, add_work / add_host the reflected panel's pgeadd scratch.
// When the panel's replicated axis is trivial -- side 'L' with one process
// column, side 'R' with one process row -- the panel's distribution is
// itself a descriptor, and the reflected region is ONE fused pgeadd
// transpose straight into it: the concentrated staging, the second transport
// and the final memcpy disappear (the write address equals the base the
// memcpy would have used).  Every other grid keeps the transport routes.
template <DdlaBackend Backend, typename T>
void off_diagonal(const DdlaHandle_t& h, T* panel, T* piece, T* work,
                  void* add_work, std::size_t add_bytes, void* add_host, std::size_t add_host_bytes,
                  bool left, bool direct, bool hermitian,
                  int k, int width, int first, int length, int along, int panel_ld,
                  int offset, int kw, int order,
                  const T* A, const int* da, const int* dc)
{
    if (length == 0) return;
    const int nb = da[DDLA_MB_];
    const int coord = left ? h->myprow_ : h->mypcol_;
    const int procs = left ? h->nprows_ : h->npcols_;
    const int target = dc[left ? DDLA_RSRC_ : DDLA_CSRC_];
    const int start = num_loc(first,nb,coord,target,procs);
    const int count = num_loc(first+length,nb,coord,target,procs)-start;
    T* const active = panel+(left ? std::size_t(offset)*along : std::size_t(offset));
    const T* input;
    if (direct) {
        pblas::broadcast_panel<Backend,T>(h, left, left ? length : width, left ? width : length, A,
            left ? first : k, left ? k : first, da, piece);
        input = shift_panel<Backend>(h,left,nb,first,length,
            da[left ? DDLA_RSRC_ : DDLA_CSRC_],target,width,piece,work);
    } else if ((left ? h->npcols_ : h->nprows_)==1) {
        // The descriptor spans the whole window, so the region's global block
        // phase matches the panel's and pgeadd's fused transpose lands each
        // rank's share exactly where the memcpy below would put it.
        int dp[DDLA_DLEN_];
        if (left)
            pblas::describe(dp,h,order,kw,nb,nb,dc[DDLA_RSRC_],dc[DDLA_CSRC_],panel_ld);
        else
            pblas::describe(dp,h,kw,order,nb,nb,dc[DDLA_RSRC_],dc[DDLA_CSRC_],panel_ld);
        pgeadd_impl::pgeadd_run<Backend,T>(h,hermitian ? 'C' : 'T','N',
            left ? length : width,left ? width : length,
            T(1),A,(left ? k : first)+1,(left ? first : k)+1,da,
            T(0),static_cast<const T*>(nullptr),1,1,dp,
            panel,(left ? first : offset)+1,(left ? offset : first)+1,dp,
            add_work,add_bytes,add_host,add_host_bytes);
        return;
    } else {
        pblas::panel_run<Backend,T>(h,left,hermitian?'C':'T',
            left?width:length,left?length:width,A,left?k:first,left?first:k,da,dc,piece,
            add_work,add_bytes,add_host,add_host_bytes);
        input = piece;
    }
    if (count > 0) {
        if (left)
            DDLA_CHECK(ddlaMemcpy2D(h,active+start,std::size_t(along)*sizeof(T),
                input,std::size_t(count)*sizeof(T),std::size_t(count)*sizeof(T),width));
        else
            DDLA_CHECK(ddlaMemcpy2D(h,active+std::size_t(start)*panel_ld,std::size_t(panel_ld)*sizeof(T),
                input,std::size_t(width)*sizeof(T),std::size_t(width)*sizeof(T),count));
    }
}

// Prepare one Hermitian/symmetric A panel and one general B panel in C's
// distribution. The triangular driver keeps B at a stable address for its
// batched pointer lists. The full-output driver may consume the transfer buffer.
template <DdlaBackend Backend, typename T,typename TA,typename TB>
const TB* prepare_panel(const DdlaHandle_t& h, bool left, char uplo, bool hermitian,
    bool pack_b, int m, int n, int first, int kw,
    const TA* A, const int* da, const TB* B, const int* db, const int* dc,
    TA* panel, TA* diagonal, T* piece, T* work, TB* b_panel,
    void* add_work, std::size_t add_bytes, void* host_work, std::size_t host_bytes)
{
    const int nb=da[DDLA_MB_],order=left ? m : n;
    const int pr=h->nprows_,pc=h->npcols_,r=h->myprow_,c=h->mypcol_;
    const int ml=num_loc(m,dc[DDLA_MB_],r,dc[DDLA_RSRC_],pr);
    const int nl=num_loc(n,dc[DDLA_NB_],c,dc[DDLA_CSRC_],pc);
    const int along=left ? ml : nl;
    const int panel_ld=left ? along : kw;
    const TB* b=b_panel;
    for (int k=first;k<first+kw;) {
        const int width=std::min(nb,first+kw-k),offset=k-first;
        TA* active=panel+(left ? std::size_t(offset)*along : std::size_t(offset));
        const int ar=indxg2p(k,nb,da[DDLA_RSRC_],pr),ac=indxg2p(k,nb,da[DDLA_CSRC_],pc);
        if (r==ar && c==ac) {
            const int i=num_loc(k,nb,r,da[DDLA_RSRC_],pr),j=num_loc(k,nb,c,da[DDLA_CSRC_],pc);
            blas_impl::diagonal_block_run<Backend,TA>(h,uplo=='U',hermitian,width,A+i+std::size_t(j)*da[DDLA_LLD_],da[DDLA_LLD_],diagonal,width);
        }
        const bool direct_before=left ? uplo=='U' : uplo=='L';
        bool grouped=false;
#if defined(DDLA_USE_CCL) && !defined(DDLA_USE_GPU_CPU_TUNNEL)
        if constexpr (Backend==DdlaBackend::GPU)
            grouped=da[left ? DDLA_RSRC_ : DDLA_CSRC_]==dc[left ? DDLA_RSRC_ : DDLA_CSRC_]
                && db[left ? DDLA_CSRC_ : DDLA_RSRC_]==dc[left ? DDLA_CSRC_ : DDLA_RSRC_]
                && db[left ? DDLA_NB_ : DDLA_MB_]==dc[left ? DDLA_NB_ : DDLA_MB_];
#endif
        const int direct_first=direct_before ? 0 : k+width;
        const int direct_length=direct_before ? k : order-direct_first;
        bool group_active=false;
        try {
            if (grouped) {
                const int reflected_first=direct_before ? k+width : 0;
                const int reflected_length=direct_before ? order-reflected_first : k;
                // Finish the dependent reflected transfer first. The
                // remaining direct, diagonal and B broadcasts use
                // distinct piece/diagonal/work buffers and can share
                // one NCCL/RCCL launch without forwarding dependencies.
                off_diagonal<Backend,TA>(h,panel,reinterpret_cast<TA*>(piece),reinterpret_cast<TA*>(work),add_work,add_bytes,host_work,host_bytes,
                    left,false,hermitian,k,width,reflected_first,reflected_length,along,panel_ld,
                    offset,kw,order,A,da,dc);
                commGroupStart<Backend>(h);group_active=true;
                if (direct_length>0)
                    pblas::broadcast_panel<Backend,TA>(h, left, left ? direct_length : width,
                        left ? width : direct_length, A, left ? direct_first : k,
                        left ? k : direct_first, da, reinterpret_cast<TA*>(piece));
            } else {
                for (bool before : {true,false}) {
                    const int row=before ? 0 : k+width,length=before ? k : order-row;
                    off_diagonal<Backend,TA>(h,panel,reinterpret_cast<TA*>(piece),reinterpret_cast<TA*>(work),add_work,add_bytes,host_work,host_bytes,
                        left,direct_before==before,hermitian,k,width,row,length,along,panel_ld,
                        offset,kw,order,A,da,dc);
                }
            }
            commBcast<Backend>(h,CommScope::Grid,diagonal,std::size_t(width)*width,h->rc_to_rank(ar,ac));
            pblas::broadcast_panel<Backend,TB>(h, !left, left ? width : m, left ? n : width, B,
                left ? k : 0, left ? 0 : k, db, reinterpret_cast<TB*>(grouped ? work : piece));
            if (grouped) {group_active=false;commGroupEnd<Backend>(h);}
        } catch (...) {
            if (group_active) try {commGroupEnd<Backend>(h);} catch (...) {}
            throw;
        }
        const int coord=left ? r : c,procs=left ? pr : pc,target=dc[left ? DDLA_RSRC_ : DDLA_CSRC_];
        if (grouped && direct_length>0) {
            const int start=num_loc(direct_first,nb,coord,target,procs);
            const int count=num_loc(direct_first+direct_length,nb,coord,target,procs)-start;
            if (count>0) {
                if (left) DDLA_CHECK(ddlaMemcpy2D(h,active+start,std::size_t(along)*sizeof(TA),
                    piece,std::size_t(count)*sizeof(TA),std::size_t(count)*sizeof(TA),width));
                else DDLA_CHECK(ddlaMemcpy2D(h,active+std::size_t(start)*panel_ld,std::size_t(panel_ld)*sizeof(TA),
                    piece,std::size_t(width)*sizeof(TA),std::size_t(width)*sizeof(TA),count));
            }
        }
        if (coord==indxg2p(k,nb,target,procs)) {
            const int start=num_loc(k,nb,coord,target,procs);
            DDLA_CHECK(ddlaMemcpy2D(h,active+(left ? std::size_t(start) : std::size_t(start)*panel_ld),
                std::size_t(panel_ld)*sizeof(TA),diagonal,std::size_t(width)*sizeof(TA),
                std::size_t(width)*sizeof(TA),width));
        }
        const TB* local_b=grouped ? reinterpret_cast<TB*>(work) : shift_panel<Backend>(h,!left,db[left ? DDLA_NB_ : DDLA_MB_],0,left ? n : m,
            db[left ? DDLA_CSRC_ : DDLA_RSRC_],dc[left ? DDLA_CSRC_ : DDLA_RSRC_],width,reinterpret_cast<TB*>(piece),reinterpret_cast<TB*>(work));
        if (kw>nb || pack_b) {
            if (left && nl>0)
                DDLA_CHECK(ddlaMemcpy2D(h,b_panel+offset,std::size_t(kw)*sizeof(TB),local_b,
                    std::size_t(width)*sizeof(TB),std::size_t(width)*sizeof(TB),nl));
            else if (!left && ml>0)
                DDLA_CHECK(ddlaMemcpy2D(h,b_panel+std::size_t(offset)*ml,std::size_t(ml)*sizeof(TB),
                    local_b,std::size_t(ml)*sizeof(TB),std::size_t(ml)*sizeof(TB),width));
        } else b=local_b;
        k+=width;
    }
    return b;
}

// Consume the common prepared panels with full local GEMM updates. Ranks
// without local output still participate in every panel transfer.
template <DdlaBackend Backend, typename T,typename TA,typename TB,typename Prepare>
void gemm_driver(const DdlaHandle_t& h, bool left, int m, int n, int k, int full,
    const T& alpha, const T& beta, T* C, int ldc, TA* panel, const Prepare& prepare,void* work,std::size_t bytes)
{
    for (int first=0;first<k;) {
        const int kw=std::min(full,k-first);
        const TB* b=prepare(first,kw);
        if (m>0 && n>0) {
            if (left) blas_impl::local_gemm_run<Backend,T,TA,TB>(h,'N','N',m,n,kw,alpha,panel,m,
                b,kw,first==0 ? beta : T(1),C,ldc,work,bytes);
            else blas_impl::local_gemm_run<Backend,T,TB,TA>(h,'N','N',m,n,kw,alpha,b,m,
                panel,kw,first==0 ? beta : T(1),C,ldc,work,bytes);
        }
        first+=kw;
    }
}

}} // namespace phemm_impl::<anonymous>

namespace phemm_impl {

template <DdlaBackend Backend, typename T,typename TA,typename TB>
bool phemm_check(const DdlaHandle_t& h, char side, char uplo, int m, int n,
    const T& alpha, const TA* A, const int* da, const TB* B, const int* db,
    const T& beta, const T* C, const int* dc, bool hermitian, char uploC)
{
    uploC=blas_impl::upper(uploC);
    const char* const routine=hermitian ? (uploC=='F' ? "phemm" : "phemmt")
                                      : (uploC=='F' ? "psymm" : "psymmt");
    return detail::agree_check(h,routine,[&](detail::Agree& agree) {
        detail::require_backend<Backend>(h,routine);
        check_desc(da,h); check_desc(db,h); check_desc(dc,h);
        side = static_cast<char>(std::toupper(static_cast<unsigned char>(side)));
        uplo = static_cast<char>(std::toupper(static_cast<unsigned char>(uplo)));
        if ((side!='L' && side!='R') || (uplo!='U' && uplo!='L'))
            throw std::invalid_argument(std::string(routine)+": expected side L/R and uplo U/L");
        if (uploC!='F' && uploC!='U' && uploC!='L')
            throw std::invalid_argument(std::string(routine)+": expected output F/U/L");
        if (uploC!='F' && (m!=n || dc[DDLA_MB_]!=dc[DDLA_NB_]))
            throw std::invalid_argument(std::string(routine)+": square output and square matching blocks required");
        const bool left = side=='L';
        const int order = left ? m : n, nb = da[DDLA_MB_];
        if (m<0 || n<0 || order>da[DDLA_M_] || order>da[DDLA_N_] ||
            m>db[DDLA_M_] || n>db[DDLA_N_] || m>dc[DDLA_M_] || n>dc[DDLA_N_])
            throw std::invalid_argument(std::string(routine)+": dimensions outside descriptors");
        if (da[DDLA_NB_]!=nb || db[left ? DDLA_MB_ : DDLA_NB_]!=nb ||
            db[DDLA_MB_]!=dc[DDLA_MB_] || db[DDLA_NB_]!=dc[DDLA_NB_])
            throw std::invalid_argument(std::string(routine)+": square A blocks must match B/C's multiplied blocks; B/C blocks must match");
        agree.mode("phemm/psymm structure/side/uplo",(hermitian ? 4 : 0)+(left ? 2 : 0)+(uplo=='U')+8*(uploC=='F' ? 0 : uploC=='U' ? 1 : 2));
        agree.mode("phemm m",m); agree.mode("phemm n",n);
        agree.mode("phemm coefficient modes",(alpha==T(0) ? 4 : 0)+(beta==T(0) ? 2 : 0)+(beta==T(1)));
        const bool noop = m==0 || n==0 || (alpha==T(0) && beta==T(1));
        if (!noop) {
            const int pr=h->nprows_,pc=h->npcols_,r=h->myprow_,c=h->mypcol_;
            const int mr=num_loc(m,dc[DDLA_MB_],r,dc[DDLA_RSRC_],pr);
            const int nc=num_loc(n,dc[DDLA_NB_],c,dc[DDLA_CSRC_],pc);
            const bool selected=mr>0 && nc>0 && (uploC=='F' || (uploC=='U'
                ? indxl2g(0,dc[DDLA_MB_],r,dc[DDLA_RSRC_],pr)<=indxl2g(nc-1,dc[DDLA_NB_],c,dc[DDLA_CSRC_],pc)
                : indxl2g(mr-1,dc[DDLA_MB_],r,dc[DDLA_RSRC_],pr)>=indxl2g(0,dc[DDLA_NB_],c,dc[DDLA_CSRC_],pc)));
            if (selected && !C) throw std::invalid_argument(std::string(routine)+": null C");
            if (alpha!=T(0)) {
                const int ar=num_loc(order,nb,r,da[DDLA_RSRC_],pr),ac=num_loc(order,nb,c,da[DDLA_CSRC_],pc);
                const int br=num_loc(m,db[DDLA_MB_],r,db[DDLA_RSRC_],pr);
                const int bc=num_loc(n,db[DDLA_NB_],c,db[DDLA_CSRC_],pc);
                const bool stored = ar>0 && ac>0 && (uplo=='U'
                    ? indxl2g(0,nb,r,da[DDLA_RSRC_],pr)<=indxl2g(ac-1,nb,c,da[DDLA_CSRC_],pc)
                    : indxl2g(ar-1,nb,r,da[DDLA_RSRC_],pr)>=indxl2g(0,nb,c,da[DDLA_CSRC_],pc));
                if ((stored && !A) || (br>0 && bc>0 && !B)) throw std::invalid_argument(std::string(routine)+": null input");
                if (detail::matrix_overlaps(A,da[DDLA_LLD_],ar,ac,C,dc[DDLA_LLD_],mr,nc) ||
                    detail::matrix_overlaps(B,db[DDLA_LLD_],br,bc,C,dc[DDLA_LLD_],mr,nc))
                    throw std::invalid_argument(std::string(routine)+": C must not overlap A or B");
            }
        }
        return noop;
    });
}

// Backend [ panel | diagonal | piece | work (scratch_counts) | the reflected
// panel's pgeadd ], host [ the reflected panel's pgeadd ].
template <typename T,typename TA,typename TB>
void phemm_workspace_size(const DdlaHandle_t& h, char side, int m, int n,
    bool alpha_is_zero, const int* da, const int* db, const int* dc, bool hermitian,
    std::size_t* backend_bytes, std::size_t* host_bytes, char uploC)
{
    *backend_bytes=*host_bytes=0;
    if (alpha_is_zero || m==0 || n==0) return;
    constexpr bool mixed=!std::is_same_v<T,TA> || !std::is_same_v<T,TB>;
    if (h->nprocs==1 && (uploC!='F' || mixed)) {
        blas_impl::hemmt_workspace_size<T,TA,TB>(h,side=='L'?m:n,true,backend_bytes,side=='L'?n:m,side,uploC);
        return;
    }
    // Native GPU SYMM/HEMM loses to packed GEMM on large padded matrices.
    if (!mixed && h->nprocs==1 && ((h->backend==DdlaBackend::CPU && !hermitian) ||
                        (h->backend==DdlaBackend::GPU && std::max(m,n)<=1024))) return;
    const bool left=side=='L';const int order=left ? m : n;
    std::size_t count[5],add_bytes=0;
    const int kp=scratch_counts<T,TA,TB>(h,left,m,n,da,db,dc,count,uploC!='F');
    int d[DDLA_DLEN_];
    // Each reflected pgeadd moves at most one descriptor block across the
    // full order, even when several blocks share the prepared panel. Its
    // scratch depends on that region, not on a square enclosing descriptor.
    reflected_panel(h,left,order,da[DDLA_MB_],d);
    if ((left ? h->npcols_ : h->nprows_)==1)
        pgeadd_impl::pgeadd_workspace_size<TA>(h,hermitian?'C':'T','N',true,false,
            left ? order : std::min(order,da[DDLA_MB_]),
            left ? std::min(order,da[DDLA_MB_]) : order,
            d,&add_bytes,host_bytes);
    else
        pblas::panel_workspace_size<TA>(h,left,hermitian?'C':'T',
            left?da[DDLA_MB_]:order,left?order:da[DDLA_MB_],da,dc,&add_bytes,host_bytes);
    *backend_bytes+=detail::padded_bytes<TA>(count[0])+detail::padded_bytes<TA>(count[1])
        +detail::padded_bytes<T>(count[2])+detail::padded_bytes<T>(count[3])+detail::padded_bytes<TB>(count[4]);
    *backend_bytes+=add_bytes;
    if (uploC!='F') {
        std::size_t batch=0,host=0,triangle=0;
        if (left) pgemmt_impl::panel::workspace_size<T,TA,TB>(h,uploC,m,order,kp,dc,&batch,&host,&triangle);
        else pgemmt_impl::panel::workspace_size<T,TB,TA>(h,uploC,m,order,kp,dc,&batch,&host,&triangle);
        *backend_bytes+=detail::padded_bytes<char>(batch)+detail::padded_bytes<char>(triangle);
        *host_bytes+=detail::padded_bytes<char>(host);
    }
    if (uploC=='F') {
        const int ml=num_loc(m,dc[DDLA_MB_],h->myprow_,dc[DDLA_RSRC_],h->nprows_);
        const int nl=num_loc(n,dc[DDLA_NB_],h->mypcol_,dc[DDLA_CSRC_],h->npcols_);
        std::size_t bytes=0;
        for (int width : {kp,order%kp}) if (width>0) {
            std::size_t current=0;
            if (left) blas_impl::local_gemm_workspace_size<T,TA,TB>(h,ml,nl,width,&current);
            else blas_impl::local_gemm_workspace_size<T,TB,TA>(h,ml,nl,width,&current);
            bytes=std::max(bytes,current);
        }
        *backend_bytes+=detail::padded_bytes<char>(bytes);
    }
}

template <DdlaBackend Backend, typename T,typename TA,typename TB>
void phemm_run(const DdlaHandle_t& h, char side, char uplo,
    int m, int n, const T& alpha, const TA* A, const int* da,
    const TB* B, const int* db, const T& beta, T* C, const int* dc, bool hermitian,
    void* backend_work, [[maybe_unused]] std::size_t backend_bytes, void* host_work, std::size_t host_bytes, char uploC)
{
    try {
        if (m==0 || n==0 || (alpha==T(0) && beta==T(1))) return;
        constexpr bool mixed=!std::is_same_v<T,TA> || !std::is_same_v<T,TB>;
        if (h->nprocs==1 && (uploC!='F' || mixed)) {
            blas_impl::hemmt_run<Backend,T,TA,TB>(h,side,uplo,uploC,side=='L'?m:n,alpha,A,da[DDLA_LLD_],B,db[DDLA_LLD_],
                beta,C,dc[DDLA_LLD_],backend_work,backend_bytes,hermitian,side=='L'?n:m);
            return;
        }
        if constexpr (!mixed) {
        if(h->nprocs==1 && alpha!=T(0) && ((Backend==DdlaBackend::CPU && !hermitian) ||
                                               (Backend==DdlaBackend::GPU && std::max(m,n)<=1024))) {
            if (hermitian)
                blas_impl::hemm_run<Backend,T>(h,side,uplo,m,n,alpha,A,da[DDLA_LLD_],B,db[DDLA_LLD_],beta,C,dc[DDLA_LLD_]);
            else
                blas_impl::symm_run<Backend,T>(h,side,uplo,m,n,alpha,A,da[DDLA_LLD_],B,db[DDLA_LLD_],beta,C,dc[DDLA_LLD_]);
            return;
        }
        }
        const bool left=side=='L';const int order=left ? m : n;
        const int pr=h->nprows_,pc=h->npcols_,r=h->myprow_,c=h->mypcol_;
        const int ml=num_loc(m,dc[DDLA_MB_],r,dc[DDLA_RSRC_],pr);
        const int nl=num_loc(n,dc[DDLA_NB_],c,dc[DDLA_CSRC_],pc),ldc=dc[DDLA_LLD_];
        if (alpha==T(0)) {
            if (uploC!='F') {
                pblas::triangle_panel_run<Backend,T,T,T>(h,uploC,false,m,0,T(0),nullptr,nullptr,beta,C,dc,nullptr,0);
                return;
            }

            blas_impl::matrix_scal_run<Backend,T>(h,ml,nl,beta,C,ldc);
            return;
        }
        // The scratch, in the order phemm_workspace_size counts it; the rest is
        // the reflected panel's pgeadd.
        std::size_t count[5];
        const int kp=scratch_counts<T,TA,TB>(h,left,m,n,da,db,dc,count,uploC!='F');
        std::size_t batch_bytes=0,batch_host_bytes=0,triangle_bytes=0;
        if (uploC!='F') {
            if (left) pgemmt_impl::panel::workspace_size<T,TA,TB>(h,uploC,m,order,kp,dc,&batch_bytes,&batch_host_bytes,&triangle_bytes);
            else pgemmt_impl::panel::workspace_size<T,TB,TA>(h,uploC,m,order,kp,dc,&batch_bytes,&batch_host_bytes,&triangle_bytes);
            backend_bytes-=detail::padded_bytes<char>(batch_bytes)+detail::padded_bytes<char>(triangle_bytes);
            host_bytes-=detail::padded_bytes<char>(batch_host_bytes);
        }
        char* batch_work=uploC!='F' ? static_cast<char*>(backend_work)+backend_bytes : nullptr;
        char* batch_host=batch_host_bytes ? static_cast<char*>(host_work)+host_bytes : nullptr;
        void* triangle_work=triangle_bytes ? batch_work+detail::padded_bytes<char>(batch_bytes) : nullptr;
        char* w=static_cast<char*>(backend_work);
        TA* const panel=detail::take<TA>(w,count[0]);
        TA* const diagonal=detail::take<TA>(w,count[1]);
        T* const piece=detail::take<T>(w,count[2]);
        T* const work=detail::take<T>(w,count[3]);
        TB* const b_panel=detail::take<TB>(w,count[4]);
        const std::size_t used=static_cast<std::size_t>(w-static_cast<char*>(backend_work));
        assert(used<=backend_bytes);
        // All temporaries use the handle's queue. Only scratch contains reflected
        // entries; A itself is never completed or written.
        std::size_t product_bytes=0;
        if (uploC=='F') {
            for (int width : {kp,order%kp}) if (width>0) {
                std::size_t current=0;
                if (left) blas_impl::local_gemm_workspace_size<T,TA,TB>(h,ml,nl,width,&current);
                else blas_impl::local_gemm_workspace_size<T,TB,TA>(h,ml,nl,width,&current);
                product_bytes=std::max(product_bytes,current);
            }
        }
        const auto prepare=[&](int first,int kw) {
            return prepare_panel<Backend,T,TA,TB>(h,left,uplo,hermitian,uploC!='F',m,n,first,kw,
                A,da,B,db,dc,panel,diagonal,piece,work,b_panel,w,backend_bytes-used-detail::padded_bytes<char>(product_bytes),
                host_work,host_bytes);
        };
        if (uploC=='F') {
            void* product_work=product_bytes ? static_cast<char*>(backend_work)+backend_bytes-detail::padded_bytes<char>(product_bytes) : nullptr;
            gemm_driver<Backend,T,TA,TB>(h,left,ml,nl,order,kp,alpha,beta,C,ldc,panel,prepare,product_work,product_bytes);
        } else if (left)
            pgemmt_impl::panel::run<Backend,T>(h,uploC,m,order,kp,alpha,beta,C,dc,prepare,
                panel,b_panel,batch_work,batch_bytes,batch_host,batch_host_bytes,triangle_work,triangle_bytes);
        else
            pgemmt_impl::panel::run<Backend,T>(h,uploC,m,order,kp,alpha,beta,C,dc,prepare,
                b_panel,panel,batch_work,batch_bytes,batch_host,batch_host_bytes,triangle_work,triangle_bytes);

    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(h) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(h); } catch (...) {}
        throw;
    }
}

} // namespace phemm_impl

template <DdlaBackend Backend, typename T>
void phemm(const DdlaHandle_t& h, char side, char uplo, int m, int n,
    const T& alpha, const T* A, const int* da, const T* B, const int* db,
    const T& beta, T* C, const int* dc)
{
    const bool noop=phemm_impl::phemm_check<Backend,T>(h,side,uplo,m,n,alpha,A,da,B,db,beta,C,dc);
    if (noop) return;
    side=static_cast<char>(std::toupper(static_cast<unsigned char>(side)));
    uplo=static_cast<char>(std::toupper(static_cast<unsigned char>(uplo)));
    std::size_t backend_bytes=0,host_bytes=0;
    phemm_impl::phemm_workspace_size<T>(h,side,m,n,alpha==T(0),da,db,dc,true,&backend_bytes,&host_bytes);
    vector<char> backend_work(h,backend_bytes);
    std::vector<char> host_work(host_bytes);
    phemm_impl::phemm_run<Backend,T>(h,side,uplo,m,n,alpha,A,da,B,db,beta,C,dc,true,
        backend_work.data(),backend_bytes,host_work.data(),host_bytes);
}

#define INSTANTIATE_PHEMM(B,T) \
 template void phemm<B,T>(const DdlaHandle_t&,char,char,int,int,const T&,const T*,const int*,const T*,const int*,const T&,T*,const int*); \
 template bool phemm_impl::phemm_check<B,T>(const DdlaHandle_t&,char,char,int,int,const T&,const T*,const int*,const T*,const int*,const T&,const T*,const int*,bool,char); \
 template void phemm_impl::phemm_run<B,T>(const DdlaHandle_t&,char,char,int,int,const T&,const T*,const int*,const T*,const int*,const T&,T*,const int*,bool,void*,std::size_t,void*,std::size_t,char)
#define INSTANTIATE_PHEMM_TYPES(B) \
 INSTANTIATE_PHEMM(B,float); INSTANTIATE_PHEMM(B,double); \
 INSTANTIATE_PHEMM(B,std::complex<float>); INSTANTIATE_PHEMM(B,std::complex<double>)
INSTANTIATE_PHEMM_TYPES(DdlaBackend::CPU);
#if DDLA_HAS_GPU
INSTANTIATE_PHEMM_TYPES(DdlaBackend::GPU);
#endif
#define INSTANTIATE_SIZE(T) template void phemm_impl::phemm_workspace_size<T>(const DdlaHandle_t&,char,int,int,bool,const int*,const int*,const int*,bool,std::size_t*,std::size_t*,char)
INSTANTIATE_SIZE(float); INSTANTIATE_SIZE(double);
INSTANTIATE_SIZE(std::complex<float>); INSTANTIATE_SIZE(std::complex<double>);
#undef INSTANTIATE_SIZE
#undef INSTANTIATE_PHEMM_TYPES
#undef INSTANTIATE_PHEMM
template <DdlaBackend Backend,typename R>
void phemm(const DdlaHandle_t& h,char side,char uplo,int m,int n,
    const std::complex<R>& alpha,const R* A,const int* da,const std::complex<R>* B,const int* db,
    const std::complex<R>& beta,std::complex<R>* C,const int* dc)
{
    using T=std::complex<R>;
    if (phemm_impl::phemm_check<Backend,T>(h,side,uplo,m,n,alpha,A,da,B,db,beta,C,dc)) return;
    side=blas_impl::upper(side);uplo=blas_impl::upper(uplo);
    std::size_t bytes=0,host_bytes=0;
    phemm_impl::phemm_workspace_size<T,R,std::complex<R>>(h,side,m,n,alpha==T(0),da,db,dc,true,&bytes,&host_bytes);
    vector<char> work(h,bytes);std::vector<char> host(host_bytes);
    phemm_impl::phemm_run<Backend,T>(h,side,uplo,m,n,alpha,A,da,B,db,beta,C,dc,true,
        work.data(),bytes,host.data(),host_bytes);
}

template <DdlaBackend Backend,typename R>
void phemm(const DdlaHandle_t& h,char side,char uplo,int m,int n,
    const std::complex<R>& alpha,const std::complex<R>* A,const int* da,const R* B,const int* db,
    const std::complex<R>& beta,std::complex<R>* C,const int* dc)
{
    using T=std::complex<R>;
    if (phemm_impl::phemm_check<Backend,T>(h,side,uplo,m,n,alpha,A,da,B,db,beta,C,dc)) return;
    side=blas_impl::upper(side);uplo=blas_impl::upper(uplo);
    std::size_t bytes=0,host_bytes=0;
    phemm_impl::phemm_workspace_size<T,std::complex<R>,R>(h,side,m,n,alpha==T(0),da,db,dc,true,&bytes,&host_bytes);
    vector<char> work(h,bytes);std::vector<char> host(host_bytes);
    phemm_impl::phemm_run<Backend,T>(h,side,uplo,m,n,alpha,A,da,B,db,beta,C,dc,true,
        work.data(),bytes,host.data(),host_bytes);
}

template void phemm<DdlaBackend::CPU,float>(const DdlaHandle_t&,char,char,int,int,const std::complex<float>&,const float*,const int*,const std::complex<float>*,const int*,const std::complex<float>&,std::complex<float>*,const int*);
template bool phemm_impl::phemm_check<DdlaBackend::CPU,std::complex<float>,float,std::complex<float>>(const DdlaHandle_t&,char,char,int,int,const std::complex<float>&,const float*,const int*,const std::complex<float>*,const int*,const std::complex<float>&,const std::complex<float>*,const int*,bool,char);
template void phemm_impl::phemm_run<DdlaBackend::CPU,std::complex<float>,float,std::complex<float>>(const DdlaHandle_t&,char,char,int,int,const std::complex<float>&,const float*,const int*,const std::complex<float>*,const int*,const std::complex<float>&,std::complex<float>*,const int*,bool,void*,std::size_t,void*,std::size_t,char);
template void phemm<DdlaBackend::CPU,float>(const DdlaHandle_t&,char,char,int,int,const std::complex<float>&,const std::complex<float>*,const int*,const float*,const int*,const std::complex<float>&,std::complex<float>*,const int*);
template bool phemm_impl::phemm_check<DdlaBackend::CPU,std::complex<float>,std::complex<float>,float>(const DdlaHandle_t&,char,char,int,int,const std::complex<float>&,const std::complex<float>*,const int*,const float*,const int*,const std::complex<float>&,const std::complex<float>*,const int*,bool,char);
template void phemm_impl::phemm_run<DdlaBackend::CPU,std::complex<float>,std::complex<float>,float>(const DdlaHandle_t&,char,char,int,int,const std::complex<float>&,const std::complex<float>*,const int*,const float*,const int*,const std::complex<float>&,std::complex<float>*,const int*,bool,void*,std::size_t,void*,std::size_t,char);
template void phemm<DdlaBackend::CPU,double>(const DdlaHandle_t&,char,char,int,int,const std::complex<double>&,const double*,const int*,const std::complex<double>*,const int*,const std::complex<double>&,std::complex<double>*,const int*);
template bool phemm_impl::phemm_check<DdlaBackend::CPU,std::complex<double>,double,std::complex<double>>(const DdlaHandle_t&,char,char,int,int,const std::complex<double>&,const double*,const int*,const std::complex<double>*,const int*,const std::complex<double>&,const std::complex<double>*,const int*,bool,char);
template void phemm_impl::phemm_run<DdlaBackend::CPU,std::complex<double>,double,std::complex<double>>(const DdlaHandle_t&,char,char,int,int,const std::complex<double>&,const double*,const int*,const std::complex<double>*,const int*,const std::complex<double>&,std::complex<double>*,const int*,bool,void*,std::size_t,void*,std::size_t,char);
template void phemm<DdlaBackend::CPU,double>(const DdlaHandle_t&,char,char,int,int,const std::complex<double>&,const std::complex<double>*,const int*,const double*,const int*,const std::complex<double>&,std::complex<double>*,const int*);
template bool phemm_impl::phemm_check<DdlaBackend::CPU,std::complex<double>,std::complex<double>,double>(const DdlaHandle_t&,char,char,int,int,const std::complex<double>&,const std::complex<double>*,const int*,const double*,const int*,const std::complex<double>&,const std::complex<double>*,const int*,bool,char);
template void phemm_impl::phemm_run<DdlaBackend::CPU,std::complex<double>,std::complex<double>,double>(const DdlaHandle_t&,char,char,int,int,const std::complex<double>&,const std::complex<double>*,const int*,const double*,const int*,const std::complex<double>&,std::complex<double>*,const int*,bool,void*,std::size_t,void*,std::size_t,char);
#if DDLA_HAS_GPU
template void phemm<DdlaBackend::GPU,float>(const DdlaHandle_t&,char,char,int,int,const std::complex<float>&,const float*,const int*,const std::complex<float>*,const int*,const std::complex<float>&,std::complex<float>*,const int*);
template bool phemm_impl::phemm_check<DdlaBackend::GPU,std::complex<float>,float,std::complex<float>>(const DdlaHandle_t&,char,char,int,int,const std::complex<float>&,const float*,const int*,const std::complex<float>*,const int*,const std::complex<float>&,const std::complex<float>*,const int*,bool,char);
template void phemm_impl::phemm_run<DdlaBackend::GPU,std::complex<float>,float,std::complex<float>>(const DdlaHandle_t&,char,char,int,int,const std::complex<float>&,const float*,const int*,const std::complex<float>*,const int*,const std::complex<float>&,std::complex<float>*,const int*,bool,void*,std::size_t,void*,std::size_t,char);
template void phemm<DdlaBackend::GPU,float>(const DdlaHandle_t&,char,char,int,int,const std::complex<float>&,const std::complex<float>*,const int*,const float*,const int*,const std::complex<float>&,std::complex<float>*,const int*);
template bool phemm_impl::phemm_check<DdlaBackend::GPU,std::complex<float>,std::complex<float>,float>(const DdlaHandle_t&,char,char,int,int,const std::complex<float>&,const std::complex<float>*,const int*,const float*,const int*,const std::complex<float>&,const std::complex<float>*,const int*,bool,char);
template void phemm_impl::phemm_run<DdlaBackend::GPU,std::complex<float>,std::complex<float>,float>(const DdlaHandle_t&,char,char,int,int,const std::complex<float>&,const std::complex<float>*,const int*,const float*,const int*,const std::complex<float>&,std::complex<float>*,const int*,bool,void*,std::size_t,void*,std::size_t,char);
template void phemm<DdlaBackend::GPU,double>(const DdlaHandle_t&,char,char,int,int,const std::complex<double>&,const double*,const int*,const std::complex<double>*,const int*,const std::complex<double>&,std::complex<double>*,const int*);
template bool phemm_impl::phemm_check<DdlaBackend::GPU,std::complex<double>,double,std::complex<double>>(const DdlaHandle_t&,char,char,int,int,const std::complex<double>&,const double*,const int*,const std::complex<double>*,const int*,const std::complex<double>&,const std::complex<double>*,const int*,bool,char);
template void phemm_impl::phemm_run<DdlaBackend::GPU,std::complex<double>,double,std::complex<double>>(const DdlaHandle_t&,char,char,int,int,const std::complex<double>&,const double*,const int*,const std::complex<double>*,const int*,const std::complex<double>&,std::complex<double>*,const int*,bool,void*,std::size_t,void*,std::size_t,char);
template void phemm<DdlaBackend::GPU,double>(const DdlaHandle_t&,char,char,int,int,const std::complex<double>&,const std::complex<double>*,const int*,const double*,const int*,const std::complex<double>&,std::complex<double>*,const int*);
template bool phemm_impl::phemm_check<DdlaBackend::GPU,std::complex<double>,std::complex<double>,double>(const DdlaHandle_t&,char,char,int,int,const std::complex<double>&,const std::complex<double>*,const int*,const double*,const int*,const std::complex<double>&,const std::complex<double>*,const int*,bool,char);
template void phemm_impl::phemm_run<DdlaBackend::GPU,std::complex<double>,std::complex<double>,double>(const DdlaHandle_t&,char,char,int,int,const std::complex<double>&,const std::complex<double>*,const int*,const double*,const int*,const std::complex<double>&,std::complex<double>*,const int*,bool,void*,std::size_t,void*,std::size_t,char);
#endif
template void phemm_impl::phemm_workspace_size<std::complex<float>,float,std::complex<float>>(const DdlaHandle_t&,char,int,int,bool,const int*,const int*,const int*,bool,std::size_t*,std::size_t*,char);
template void phemm_impl::phemm_workspace_size<std::complex<float>,std::complex<float>,float>(const DdlaHandle_t&,char,int,int,bool,const int*,const int*,const int*,bool,std::size_t*,std::size_t*,char);
template void phemm_impl::phemm_workspace_size<std::complex<double>,double,std::complex<double>>(const DdlaHandle_t&,char,int,int,bool,const int*,const int*,const int*,bool,std::size_t*,std::size_t*,char);
template void phemm_impl::phemm_workspace_size<std::complex<double>,std::complex<double>,double>(const DdlaHandle_t&,char,int,int,bool,const int*,const int*,const int*,bool,std::size_t*,std::size_t*,char);
} // namespace ddla
