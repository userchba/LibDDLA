#ifndef DDLA_PBLAS_WORKSPACE_H
#define DDLA_PBLAS_WORKSPACE_H

#include <ddla/ddla.h>
#include <ddla/ddla_desc.h>
#include "ddla_stream.h"
#include "comm_traits.h"
#include "pgeadd_impl.h"
#include "BLAS/blas_impl.h"
#include "BLAS/gemmt_impl.h"
#include "check_agree.h"
#include "ddla_desc.h"
#include "helpers.h"
#include "require_backend.h"

#include <algorithm>

namespace ddla { namespace pblas {

// Group local GPU updates without changing descriptor-block communication.
inline int contraction_width(const DdlaHandle_t& h,int nb,int k,bool mixed=false) noexcept
{
    int width=std::min(k,h->backend==DdlaBackend::GPU && k>=1024 ? std::max(nb,(1024/nb)*nb) : nb);
    if (mixed && h->backend==DdlaBackend::GPU && h->nprocs>1) {
        // Keep both operand panels within their block-rounded share of the
        // global matrix, including 1 x P grids and k near the GPU panel size.
        const int peers=std::max(h->nprows_,h->npcols_);
        width=std::min(width,std::max(nb,(k/peers/nb)*nb));
    }
    return width;
}

// Derived descriptors in a trusted algorithm need index geometry only. The
// public descriptor constructor remains responsible for validating user input.
inline void describe(int* d,DdlaHandle_t h,int m,int n,int mb,int nb,int rs,int cs,int lld=-1) noexcept
{
    d[DDLA_DTYPE_]=DDLA_BLOCK_CYCLIC_2D;d[DDLA_CTXT_]=0;
    d[DDLA_M_]=m;d[DDLA_N_]=n;d[DDLA_MB_]=mb;d[DDLA_NB_]=nb;
    d[DDLA_RSRC_]=rs;d[DDLA_CSRC_]=cs;
    d[DDLA_LLD_]=std::max(lld,std::max(1,num_loc(m,mb,h->myprow_,rs,h->nprows_)));
}

// Replicate a one-block-wide, untransposed panel. Packing is a pitched copy
// only on its owning process row/column. A compact panel already in place
// needs just the broadcast.
template <DdlaBackend Backend, typename T>
void broadcast_panel(const DdlaHandle_t& h, bool column_panel, int m, int n,
                     const T* input, int i, int j, const int* d, T* output)
{
    if (m == 0 || n == 0) return;
    const int row0 = num_loc(i, d[DDLA_MB_], h->myprow_, d[DDLA_RSRC_], h->nprows_);
    const int col0 = num_loc(j, d[DDLA_NB_], h->mypcol_, d[DDLA_CSRC_], h->npcols_);
    const int rows = num_loc(i + m, d[DDLA_MB_], h->myprow_, d[DDLA_RSRC_], h->nprows_) - row0;
    const int cols = num_loc(j + n, d[DDLA_NB_], h->mypcol_, d[DDLA_CSRC_], h->npcols_) - col0;
    const int root = column_panel
        ? indxg2p(j, d[DDLA_NB_], d[DDLA_CSRC_], h->npcols_)
        : indxg2p(i, d[DDLA_MB_], d[DDLA_RSRC_], h->nprows_);
    const int packed_rows = column_panel ? rows : m;
    const int packed_cols = column_panel ? n : cols;
    const int coordinate = column_panel ? h->mypcol_ : h->myprow_;
    if (packed_rows == 0 || packed_cols == 0) return;
    if (coordinate == root) {
        const T* source = input + row0 + std::size_t(col0) * d[DDLA_LLD_];
        if (source != output || d[DDLA_LLD_] != packed_rows)
            DDLA_CHECK(ddlaMemcpy2D(h, output, std::size_t(packed_rows) * sizeof(T),
                source, std::size_t(d[DDLA_LLD_]) * sizeof(T),
                std::size_t(packed_rows) * sizeof(T), packed_cols));
    }
    commBcast<Backend>(h, column_panel ? CommScope::Row : CommScope::Col,
        output, std::size_t(packed_rows) * packed_cols, root);
}

inline bool panel_local_layout(bool column_panel, const int* source, const int* target)
{
    const int block = column_panel ? DDLA_MB_ : DDLA_NB_;
    const int owner = column_panel ? DDLA_RSRC_ : DDLA_CSRC_;
    return source[block] == target[block] && source[owner] == target[owner];
}

// Algorithm panels start at descriptor-block boundaries. Their short dimension
// occupies one block. The compact view retains the target distribution along
// the long dimension and places the short dimension on one broadcast root.
inline void panel_descriptor(int* d, DdlaHandle_t h, bool column_panel, char op,
                             int m, int n, int i, int j,
                             const int* source, const int* target)
{
    const bool transposed = op != 'N';
    const int rows = transposed ? n : m;
    const int cols = transposed ? m : n;
    const int mb = column_panel ? target[DDLA_MB_] : source[transposed ? DDLA_NB_ : DDLA_MB_];
    const int nb = column_panel ? source[transposed ? DDLA_MB_ : DDLA_NB_] : target[DDLA_NB_];
    int rs = 0, cs = 0;
    if (column_panel) {
        rs = indxg2p(transposed ? j : i, mb, target[DDLA_RSRC_], h->nprows_);
        if (!transposed) cs = indxg2p(j, source[DDLA_NB_], source[DDLA_CSRC_], h->npcols_);
    } else {
        cs = indxg2p(transposed ? i : j, nb, target[DDLA_CSRC_], h->npcols_);
        if (!transposed) rs = indxg2p(i, source[DDLA_MB_], source[DDLA_RSRC_], h->nprows_);
    }
    describe(d, h, rows, cols, mb, nb, rs, cs);
}

template <typename T>
void panel_workspace_size(const DdlaHandle_t& h, bool column_panel, char op,
                          int m, int n, const int* source, const int* target,
                          std::size_t* backend_bytes, std::size_t* host_bytes)
{
    *backend_bytes = *host_bytes = 0;
    if (m == 0 || n == 0 || (op == 'N' && panel_local_layout(column_panel, source, target)))
        return;
    int d[DDLA_DLEN_];
    panel_descriptor(d, h, column_panel, op, m, n, 0, 0, source, target);
    const bool general = op == 'N'
        || source[op == 'N' ? DDLA_MB_ : DDLA_NB_] != d[DDLA_MB_]
        || source[op == 'N' ? DDLA_NB_ : DDLA_MB_] != d[DDLA_NB_];
    pgeadd_impl::pgeadd_workspace_size<T>(h, op, 'N', true, false,
        d[DDLA_M_], d[DDLA_N_], d, backend_bytes, host_bytes, source, d, general);
}

// Materialize op(input) into the target panel layout, then replicate its short
// dimension. All redistribution belongs to pgeadd. The N path keeps
// the direct pack/broadcast route and uses no extra workspace.
template <DdlaBackend Backend, typename T>
void panel_run(const DdlaHandle_t& h, bool column_panel, char op, int m, int n,
               const T* input, int i, int j, const int* source, const int* target,
               T* output, void* work, std::size_t bytes, void* host, std::size_t host_bytes)
{
    if (m == 0 || n == 0) return;
    if (op == 'N' && panel_local_layout(column_panel, source, target)) {
        broadcast_panel<Backend>(h, column_panel, m, n, input, i, j, source, output);
        return;
    }
    int d[DDLA_DLEN_];
    panel_descriptor(d, h, column_panel, op, m, n, i, j, source, target);
    pgeadd_impl::pgeadd_run<Backend,T>(h, op, 'N', d[DDLA_M_], d[DDLA_N_],
        T(1), input, i + 1, j + 1, source, T(0), nullptr, 1, 1, d,
        output, 1, 1, d, work, bytes, host, host_bytes);
    broadcast_panel<Backend>(h, column_panel, d[DDLA_M_], d[DDLA_N_], output, 0, 0, d, output);
}

// Validate general product geometry and distinguish collective entry points.
// The caller combines operation-specific checks in the same Agree instance.
template <DdlaBackend Backend,typename T,typename TA=T,typename TB=T>
bool product_check(const DdlaHandle_t& h, detail::Agree& agree, char uplo, char ta, char tb,
    int n, int k, const T& alpha, const TA* A, const int* da, const TB* B, const int* db,
    const T& beta, const T* C, const int* dc, int operation=0, bool force_output=false)
{
    detail::require_backend<Backend>(h,"pgemmt");
    check_desc(da,h); check_desc(db,h); check_desc(dc,h);
    uplo=blas_impl::upper(uplo); ta=blas_impl::upper(ta); tb=blas_impl::upper(tb);
    if ((uplo!='U' && uplo!='L') ||
        (ta!='N' && ta!='T' && ta!='C') || (tb!='N' && tb!='T' && tb!='C'))
        throw std::invalid_argument("pgemmt: expected uplo U/L and N/T/C operations");
    if (n<0 || k<0) throw std::invalid_argument("pgemmt: negative dimension");
    const int ar=ta=='N' ? n : k, ac=ta=='N' ? k : n;
    const int br=tb=='N' ? k : n, bc=tb=='N' ? n : k;
    if (ar>da[DDLA_M_] || ac>da[DDLA_N_] || br>db[DDLA_M_] || bc>db[DDLA_N_] ||
        n>dc[DDLA_M_] || n>dc[DDLA_N_])
        throw std::invalid_argument("pgemmt: dimensions outside descriptors");
    const int nb=dc[DDLA_MB_];
    if (dc[DDLA_NB_]!=nb || da[DDLA_MB_]!=nb || da[DDLA_NB_]!=nb ||
        db[DDLA_MB_]!=nb || db[DDLA_NB_]!=nb)
        throw std::invalid_argument("pgemmt: all descriptors must use the same square blocks");
    const bool product=alpha!=T(0) && k>0;
    // Flags and zero-coefficient modes control collective participation.
    const int flags=((uplo=='U')<<14)+(int(ta)<<7)+int(tb);
    agree.mode("pgemmt flags/coefficient modes",flags*64+operation*8+(product ? 4 : 0)+(beta==T(0) ? 2 : 0)+(beta==T(1)));
    agree.mode("pgemmt n",n); agree.mode("pgemmt k",k); agree.mode("pgemmt block size",nb);
    if (n>0 && (force_output || product || beta!=T(1))) {
        const int r=h->myprow_,c=h->mypcol_,pr=h->nprows_,pc=h->npcols_;
        const int mr=num_loc(n,nb,r,dc[DDLA_RSRC_],pr);
        const int nc=num_loc(n,nb,c,dc[DDLA_CSRC_],pc);
        const bool selected=mr>0 && nc>0 && (uplo=='U'
            ? indxl2g(0,nb,r,dc[DDLA_RSRC_],pr)<=indxl2g(nc-1,nb,c,dc[DDLA_CSRC_],pc)
            : indxl2g(mr-1,nb,r,dc[DDLA_RSRC_],pr)>=indxl2g(0,nb,c,dc[DDLA_CSRC_],pc));
        if (selected && !C) throw std::invalid_argument("pgemmt: null C");
        if (product) {
            const int alr=num_loc(ar,nb,r,da[DDLA_RSRC_],pr),alc=num_loc(ac,nb,c,da[DDLA_CSRC_],pc);
            const int blr=num_loc(br,nb,r,db[DDLA_RSRC_],pr),blc=num_loc(bc,nb,c,db[DDLA_CSRC_],pc);
            if ((alr>0 && alc>0 && !A) || (blr>0 && blc>0 && !B))
                throw std::invalid_argument("pgemmt: null input");
            if (detail::matrix_overlaps(A,da[DDLA_LLD_],alr,alc,C,dc[DDLA_LLD_],mr,nc) ||
                detail::matrix_overlaps(B,db[DDLA_LLD_],blr,blc,C,dc[DDLA_LLD_],mr,nc))
                throw std::invalid_argument("pgemmt: C must not overlap A or B");
        }
    }
    return product;
}

// Reusable contraction-panel storage and transfers for triangular products.
template <typename T,typename TA=T,typename TB=T>
void product_workspace_size(const DdlaHandle_t& h, char ta, char tb, bool product, bool rank2, int n, int k,
             const int* da, const int* db, const int* dc,
             std::size_t (&count)[4], std::size_t (&add)[2][2])
{
    count[0]=count[1]=count[2]=count[3]=0;
    add[0][0]=add[0][1]=add[1][0]=add[1][1]=0;
    if (n==0 || !product || h->nprocs==1) return;
    const int nb=dc[DDLA_MB_],kw=std::min(nb,k);
    const int pr=h->nprows_,pc=h->npcols_,r=h->myprow_,c=h->mypcol_;
    const int mr=num_loc(n,nb,r,dc[DDLA_RSRC_],pr),nc=num_loc(n,nb,c,dc[DDLA_CSRC_],pc);
    const int sa=num_loc(n,nb,r,da[DDLA_RSRC_],pr);
    const int sb=num_loc(n,nb,c,db[DDLA_CSRC_],pc);
    const int swapped_a=rank2 ? num_loc(n,nb,r,db[DDLA_RSRC_],pr) : 0;
    const int swapped_b=rank2 ? num_loc(n,nb,c,da[DDLA_CSRC_],pc) : 0;
    count[0]=std::size_t(std::max(1,mr))*pblas::contraction_width(h,nb,k,!std::is_same_v<T,TA> || !std::is_same_v<T,TB>);
    count[1]=std::size_t(std::max(1,nc))*pblas::contraction_width(h,nb,k,!std::is_same_v<T,TA> || !std::is_same_v<T,TB>);
    count[3]=h->backend==DdlaBackend::GPU && pblas::contraction_width(h,nb,k,!std::is_same_v<T,TA> || !std::is_same_v<T,TB>)>kw
        ? std::size_t(std::max(1,nc))*pblas::contraction_width(h,nb,k,!std::is_same_v<T,TA> || !std::is_same_v<T,TB>) : 0;
    count[2]=std::size_t(std::max({1,mr,nc,sa,sb,swapped_a,swapped_b}))*kw;
    if (ta!='N')
        pblas::panel_workspace_size<TA>(h,true,ta,kw,n,da,dc,&add[0][0],&add[0][1]);
    if (tb!='N')
        pblas::panel_workspace_size<TB>(h,false,tb,n,kw,db,dc,&add[1][0],&add[1][1]);
    if (rank2) {
        std::size_t backend=0,host=0;
        if (ta!='N') {
            pblas::panel_workspace_size<TA>(h,true,ta,kw,n,db,dc,&backend,&host);
            add[0][0]=std::max(add[0][0],backend);add[0][1]=std::max(add[0][1],host);
        }
        if (tb!='N') {
            pblas::panel_workspace_size<TB>(h,false,tb,n,kw,da,dc,&backend,&host);
            add[1][0]=std::max(add[1][0],backend);add[1][1]=std::max(add[1][1],host);
        }
    }
}

template <DdlaBackend Backend, typename T>
void product_block_run(const DdlaHandle_t& h, bool left, char trans,
           int n, int g, int width, const T* input, const int* di, const int* dc,
           T* output, T* work, void* add_work, std::size_t add_bytes, void* add_host, std::size_t add_host_bytes)
{
    const int nb=dc[DDLA_MB_];
    const int coord=left ? h->myprow_ : h->mypcol_;
    const int procs=left ? h->nprows_ : h->npcols_;
    const int target=dc[left ? DDLA_RSRC_ : DDLA_CSRC_];
    if (trans=='N') {
        const int source=di[left ? DDLA_RSRC_ : DDLA_CSRC_];
        T* const packed=source==target ? output : work;
        pblas::broadcast_panel<Backend,T>(h, left, left ? n : width, left ? width : n, input, left ? 0 : g,
            left ? g : 0, di, packed);
        if (source!=target) {
            const int send=num_loc(n,nb,coord,source,procs);
            const int recv=num_loc(n,nb,coord,target,procs);
            const int shift=(target-source+procs)%procs;
            commSendrecv<Backend>(h,left ? CommScope::Col : CommScope::Row,
                packed,std::size_t(send)*width,(coord+shift)%procs,
                output,std::size_t(recv)*width,(coord-shift+procs)%procs);
        }
        return;
    }
    pblas::panel_run<Backend,T>(h,left,trans,left?width:n,left?n:width,
        input,left?g:0,left?0:g,di,dc,output,add_work,add_bytes,add_host,add_host_bytes);
}

template <DdlaBackend Backend, typename T>
void product_panel_run(const DdlaHandle_t& h, bool left, char trans, int n, int g, int kw,
    const T* input, const int* di, const int* dc, T* output, T* work, T* b_tile,
    void* add_work, std::size_t add_bytes, void* add_host, std::size_t add_host_bytes)
{
    const int nb=dc[DDLA_MB_];
    const int load_rows=num_loc(n,nb,h->myprow_,dc[DDLA_RSRC_],h->nprows_);
    const int load_cols=num_loc(n,nb,h->mypcol_,dc[DDLA_CSRC_],h->npcols_);
#if defined(DDLA_USE_CCL)
    if constexpr (Backend==DdlaBackend::GPU) {
        // These N broadcasts have distinct buffers and no intermediate
        // receive-to-send dependency. One NCCL/RCCL group can enqueue
        // the whole contraction panel; transposed/remapped transports
        // retain their required ordering in the general path below.
        const int source=di[left ? DDLA_RSRC_ : DDLA_CSRC_];
        const int target=dc[left ? DDLA_RSRC_ : DDLA_CSRC_];
        if (h->communication==DdlaCommunication::CCL && kw>nb && trans=='N' && source==target) {
            commGroupStart<Backend>(h);
            bool active=true;
            try {
                for (int offset=0;offset<kw;) {
                    const int width=std::min(nb,kw-offset);
                    T* out=left ? output+std::size_t(offset)*load_rows : b_tile+std::size_t(offset)*load_cols;
                    product_block_run<Backend,T>(h,left,trans,n,g+offset,width,input,di,dc,out,work,
                        add_work,add_bytes,add_host,add_host_bytes);
                    offset+=width;
                }
                active=false;
                commGroupEnd<Backend>(h);
            } catch (...) {
                if (active) try {commGroupEnd<Backend>(h);} catch (...) {}
                throw;
            }
            if (!left && load_cols>0) for (int offset=0;offset<kw;) {
                const int width=std::min(nb,kw-offset);
                DDLA_CHECK(ddlaMemcpy2D(h,output+offset,std::size_t(kw)*sizeof(T),
                    b_tile+std::size_t(offset)*load_cols,std::size_t(width)*sizeof(T),
                    std::size_t(width)*sizeof(T),load_cols));
                offset+=width;
            }
            return;
        }
    }
#endif
    for (int offset=0;offset<kw;) {
        const int width=std::min(nb,kw-offset);
        T* target=left ? output+std::size_t(offset)*load_rows : (kw>nb ? b_tile : output);
        product_block_run<Backend,T>(h,left,trans,n,g+offset,width,input,di,dc,target,work,
            add_work,add_bytes,add_host,add_host_bytes);
        if (!left && kw>nb && load_cols>0)
            DDLA_CHECK(ddlaMemcpy2D(h,output+offset,std::size_t(kw)*sizeof(T),
                b_tile,std::size_t(width)*sizeof(T),std::size_t(width)*sizeof(T),load_cols));
        offset+=width;
    }
}

// Update one contraction panel, preserving the unselected triangle.
template <DdlaBackend Backend, typename T,typename TA=T,typename TB=T>
void triangle_panel_run(const DdlaHandle_t& h, char uplo, bool product, int n, int kw,
    const T& alpha, const TA* a, const TB* b, const T& beta, T* C, const int* dc,
    void* triangle_work, std::size_t triangle_bytes)
{
    const int nb=dc[DDLA_MB_],ldc=dc[DDLA_LLD_];
    const int r=h->myprow_,c=h->mypcol_,pr=h->nprows_,pc=h->npcols_;
    const int mr=num_loc(n,nb,r,dc[DDLA_RSRC_],pr);
    const bool upper_triangle=uplo=='U';
    for (int j=0;j<n;) {
        const int width=std::min(nb,n-j);
        if (c==indxg2p(j,nb,dc[DDLA_CSRC_],pc)) {
            const int jl=num_loc(j,nb,c,dc[DDLA_CSRC_],pc);
            const int start=upper_triangle ? 0 : num_loc(j+width,nb,r,dc[DDLA_RSRC_],pr);
            const int end=upper_triangle ? num_loc(j,nb,r,dc[DDLA_RSRC_],pr) : mr;
            if (end>start) {
                T* out=C+start+std::size_t(jl)*ldc;
                if (product)
                    blas_impl::local_gemm_run<Backend,T,TA,TB>(h,'N','N',end-start,width,kw,alpha,
                        a+start,mr,b+std::size_t(jl)*kw,kw,beta,out,ldc,triangle_work,triangle_bytes);
                else if (beta==T(0)) {
                    if constexpr (Backend==DdlaBackend::GPU)
                        blas_impl::matrix_scal_run<Backend,T>(h,end-start,width,beta,out,ldc);
                    else // Keep the faster CPU clears for small triangular panels.
                        for (int col=0;col<width;++col)
                            DDLA_CHECK(ddlaMemset(h,out+std::size_t(col)*ldc,0,
                                std::size_t(end-start)*sizeof(T)));
                } else if (beta!=T(1))
                    blas_impl::geam_run<Backend,T>(h,'N','N',end-start,width,beta,out,ldc,
                        T(0),static_cast<const T*>(nullptr),1,out,ldc);
            }
            if (r==indxg2p(j,nb,dc[DDLA_RSRC_],pr)) {
                const int il=num_loc(j,nb,r,dc[DDLA_RSRC_],pr);
                blas_impl::gemmt_run<Backend,T,TA,TB>(h,uplo,'N','N',width,kw,
                    product ? alpha : T(0),product ? a+il : nullptr,std::max(1,mr),
                    product ? b+std::size_t(jl)*kw : nullptr,std::max(1,kw),
                    beta,C+il+std::size_t(jl)*ldc,ldc,triangle_work,triangle_bytes);
            }
        }
        j+=width;
    }
}

}}
#endif
