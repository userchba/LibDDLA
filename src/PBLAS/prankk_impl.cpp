#include "helpers.h"
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "prankk_impl.h"
#include "pgeadd_impl.h"
#include "pblas_workspace.h"
#include "BLAS/blas_impl.h"
#include "BLAS/gemmt_impl.h"
#include "ddla_desc.h"
#include "ddla_stream.h"
#include "comm_traits.h"
#include "check_agree.h"
#include "require_backend.h"
#include "plan_util.h"
#include "instantiate.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <limits>
#include <type_traits>
#include <vector>

namespace ddla { namespace prankk_impl { namespace {

char upper(char c) { return static_cast<char>(std::toupper(static_cast<unsigned char>(c))); }

template <typename T> bool rank_batches(const DdlaHandle_t& h,int n)
{return h->backend!=DdlaBackend::GPU || !std::is_arithmetic_v<T> || n>1024;}

template <typename Visit>
void output_tiles(const DdlaHandle_t& h,char uplo,int n,
    const int* dc,const Visit& visit)
{
    const int nb=dc[DDLA_MB_],r=h->myprow_,c=h->mypcol_;
    const int pr=h->nprows_,pc=h->npcols_;
    const int rs=dc[DDLA_RSRC_],cs=dc[DDLA_CSRC_];
    const int mr=num_loc(n,nb,r,rs,pr),nc=num_loc(n,nb,c,cs,pc);

    for (int j=0;j<nc;) {
        const int gj=indxl2g(j,nb,c,cs,pc),nj=std::min(nb,n-gj);
        for (int i=0;i<mr;) {
            const int gi=indxl2g(i,nb,r,rs,pr),mi=std::min(nb,n-gi);
            if (uplo=='U' ? gi<gj : gi>gj)
                visit(int(mi<nb)*2+int(nj<nb),i,j,mi,nj);
            i+=mi;
        }
        j+=nj;
    }
}

template <typename T>
void scratch(const DdlaHandle_t& h,char ta,char tb,bool product,bool rank2,int n,int k,
    const int* da,const int* db,const int* dc,std::size_t (&count)[4],std::size_t (&add)[2][2],
    std::size_t& triangle_bytes)
{
    pblas::product_workspace_size<T>(h,ta,tb,product,rank2,n,k,da,db,dc,count,add);
    triangle_bytes=0;
    if (n>0 && product && h->nprocs>1 && !rank_batches<T>(h,n))
        blas_impl::gemmt_workspace_size<T>(h,'N','N',std::min(n,dc[DDLA_MB_]),
            std::min(k,pblas::contraction_width(h,dc[DDLA_MB_],k)),&triangle_bytes);
}

// The lists of the batched rank update: the off-diagonal output tiles this
// rank owns, and the gemmBatched pointer entries (every tile for each k pitch,
// full and tail, and each rank-2k term).
void rank_counts(const DdlaHandle_t& h, char uplo, char, char, bool, bool rank2, int n, int k, const int* dc,
                 std::size_t& tiles, std::size_t& pointers)
{
    tiles=0;
    output_tiles(h,uplo,n,dc,[&](int,int,int,int,int) {
        ++tiles;
    });
    const int full=pblas::contraction_width(h,dc[DDLA_MB_],k),tail=k%full;
    const int pitches=tail>0 && tail!=full ? 2 : 1;
    pointers=tiles*(rank2 ? 2 : 1)*pitches;
}

// Backend lists contain one pointer plane per full/tail pitch and term.
// Host storage contains local tile positions and pointer lists.
template <typename T>
void rank_bytes(const DdlaHandle_t& h, char uplo, char ta, char tb, bool product, bool rank2, int n, int k, const int* dc,
                const std::size_t (&panels)[4], std::size_t& backend_bytes, std::size_t& host_bytes)
{
    std::size_t tiles=0,pointers=0;
    rank_counts(h,uplo,ta,tb,product,rank2,n,k,dc,tiles,pointers);
    backend_bytes=rank2 ? detail::padded_bytes<T>(panels[0])+detail::padded_bytes<T>(panels[1]) : 0;
    if (h->backend==DdlaBackend::GPU)
        backend_bytes+=2*detail::padded_bytes<const T*>(pointers)+detail::padded_bytes<T*>(pointers);
    host_bytes=detail::padded_bytes<int>(2*tiles)
        +2*detail::padded_bytes<const T*>(pointers)+detail::padded_bytes<T*>(pointers);
}

// Independent output tiles can share one batched GEMM. Edge tiles form
// separate shape groups; diagonal tiles use rank-update BLAS instead.
// load(left, trans, g, kw, input, di, output) fills one k panel; a / b are
// the first term's panels (panels[] elements each, from scratch()), and the
// second term's panels and the lists are carved from w / hw (rank_bytes).
template <DdlaBackend Backend, typename T, typename Load>
void rank_run(const DdlaHandle_t& h, char symmetry, char uplo, char ta, char tb, bool product, bool rank2,
    int n, int k, const T& alpha, const T* A, const int* da,
    const T* B, const int* db, const T& beta, T* C, const int* dc,
    const Load& load, T* a, T* b, T* mirror_work, const std::size_t (&panels)[4],
    char* w, [[maybe_unused]] std::size_t backend_bytes, char* hw, [[maybe_unused]] std::size_t host_bytes)
{
    const int nb=dc[DDLA_MB_],r=h->myprow_,c=h->mypcol_;
    const int mr=num_loc(n,nb,r,dc[DDLA_RSRC_],h->nprows_);
    const int nc=num_loc(n,nb,c,dc[DDLA_CSRC_],h->npcols_);
    const int ldc=dc[DDLA_LLD_],terms=rank2 ? 2 : 1;
    const int full=pblas::contraction_width(h,nb,k),tail=k%full;
    const int pitches=tail>0 && tail!=full ? 2 : 1;

    // The lists, in the order rank_bytes counts them.
    std::size_t count_tiles=0,pointers=0;
    rank_counts(h,uplo,ta,tb,product,rank2,n,k,dc,count_tiles,pointers);
    char* const w0=w;
    char* const hw0=hw;
    T* const a2=rank2 ? detail::take<T>(w,panels[0]) : nullptr;
    T* const b2=rank2 ? detail::take<T>(w,panels[1]) : nullptr;
    const T** device_a=nullptr;
    const T** device_b=nullptr;
    T** device_c=nullptr;
    if constexpr (Backend==DdlaBackend::GPU) {
        device_a=detail::take<const T*>(w,pointers);
        device_b=detail::take<const T*>(w,pointers);
        device_c=detail::take<T*>(w,pointers);
    }
    int* const positions=count_tiles ? detail::take<int>(hw,2*count_tiles) : nullptr;
    const T** const host_a=pointers ? detail::take<const T*>(hw,pointers) : nullptr;
    const T** const host_b=pointers ? detail::take<const T*>(hw,pointers) : nullptr;
    T** const host_c=pointers ? detail::take<T*>(hw,pointers) : nullptr;
    assert(std::size_t(w-w0)<=backend_bytes && (host_bytes==0 || std::size_t(hw-hw0)<=host_bytes));

    std::size_t counts[4]={},first[5]={},next[4]={};
    int rows[4]={},columns[4]={};
    output_tiles(h,uplo,n,dc,[&](int group,int,int,int height,int width) {
        ++counts[group];rows[group]=height;columns[group]=width;
    });
    for (int group=0;group<4;++group) {
        first[group+1]=first[group]+counts[group];next[group]=first[group];
    }
    assert(first[4]==count_tiles);
    output_tiles(h,uplo,n,dc,[&](int group,int row,int column,int,int) {
        const auto index=next[group]++;
        positions[2*index]=row;positions[2*index+1]=column;
    });

    // Panel addresses stay fixed across k panels. C and every pointer table
    // are rebuilt on each run, permitting reuse with another output buffer.
    for (int pitch=0;pitch<pitches;++pitch) for (int term=0;term<terms;++term) {
        const int kw=pitch==0 ? full : tail;
        const T* pa=term==0 ? a : a2;
        const T* pb=term==0 ? b : b2;
        const std::size_t base=(pitch*terms+term)*count_tiles;
        for (std::size_t index=0;index<count_tiles;++index) {
            const int row=positions[2*index],column=positions[2*index+1];
            host_a[base+index]=pa+row;
            host_b[base+index]=pb+std::size_t(column)*kw;
            host_c[base+index]=C+row+std::size_t(column)*ldc;
        }
    }
    const T* const* ap=host_a;
    const T* const* bp=host_b;
    T* const* cp=host_c;
    if constexpr (Backend==DdlaBackend::GPU) {
        if (pointers>0) {
            DDLA_CHECK(ddlaMemcpy(h,device_a,host_a,pointers*sizeof(const T*)));
            DDLA_CHECK(ddlaMemcpy(h,device_b,host_b,pointers*sizeof(const T*)));
            DDLA_CHECK(ddlaMemcpy(h,device_c,host_c,pointers*sizeof(T*)));
            // The pageable host pointer lists may be released/reused after
            // run returns; complete their asynchronous uploads once here.
            DDLA_CHECK(ddlaSynchronize(h));
        }
        ap=device_a; bp=device_b; cp=device_c;
    }
    for (int g=0;g<k;) {
        const int kw=std::min(full,k-g);
        load(true,ta,g,kw,A,da,a);
        if (rank2) load(true,ta,g,kw,B,db,a2);
        if constexpr (Backend==DdlaBackend::GPU) {
            if (h->nprows_==h->npcols_) {
                // The left panels are already replicated across each row.
                // Exchange with the transposed grid peer once per wide panel
                // instead of transporting the same factors again block by block.
                const int p=h->nprows_;
                const int shift=(dc[DDLA_RSRC_]-dc[DDLA_CSRC_]+p)%p;
                const int peer=h->rc_to_rank((c+shift)%p,(r-shift+p)%p);
                for (int term=0;term<terms;++term) {
                    const T* source=rank2 && term==0 ? a2 : a;
                    T* output=term==0 ? b : b2;
                    const T* input=source;
                    if (peer!=h->myid) {
                        commSendrecv<Backend>(h,CommScope::Grid,source,std::size_t(mr)*kw,peer,
                            mirror_work,std::size_t(nc)*kw,peer);
                        input=mirror_work;
                    }
                    if (nc>0)
                        blas_impl::geam_run<Backend,T>(h,symmetry=='H' ? 'C' : 'T','N',kw,nc,
                            T(1),input,std::max(1,nc),T(0),nullptr,1,output,kw);
                }
            } else {
                load(false,tb,g,kw,B,db,b);
                if (rank2) load(false,tb,g,kw,A,da,b2);
            }
        } else {
            load(false,tb,g,kw,B,db,b);
            if (rank2) load(false,tb,g,kw,A,da,b2);
        }
        const T scale=g==0 ? beta : T(1);
        const int pitch=kw==full ? 0 : 1;
        for (int term=0;term<terms;++term) {
            T coefficient=alpha;
            if constexpr (!std::is_arithmetic_v<T>)
                if (term!=0 && symmetry=='H') coefficient=std::conj(alpha);
            const std::size_t base=(pitch*terms+term)*count_tiles;
            for (int group=0;group<4;++group) {
                const T output_beta=term==0 ? scale : T(1);
                for (std::size_t done=0;done<counts[group];) {
                    const int batch=static_cast<int>(std::min<std::size_t>(counts[group]-done,std::numeric_limits<int>::max()));
                    const std::size_t start=base+first[group]+done;
                    blas_impl::gemmBatched_run<Backend,T>(h,'N','N',rows[group],columns[group],kw,coefficient,
                        ap+start,std::max(1,mr),bp+start,kw,output_beta,
                        cp+start,ldc,batch);
                    done+=batch;
                }
            }
        }
        for (int j=0;j<n;) {
            const int width=std::min(nb,n-j);
            if (r==indxg2p(j,nb,dc[DDLA_RSRC_],h->nprows_) &&
                c==indxg2p(j,nb,dc[DDLA_CSRC_],h->npcols_)) {
                const int il=num_loc(j,nb,r,dc[DDLA_RSRC_],h->nprows_);
                const int jl=num_loc(j,nb,c,dc[DDLA_CSRC_],h->npcols_);
                T* out=C+il+std::size_t(jl)*ldc;
                // Small single-precision vendor rank-k kernels accumulate a
                // long K serially. Preserve short diagonal reductions while
                // the bulk off-diagonal GEMMs keep the wide panel.
                const int chunk=Backend==DdlaBackend::GPU && sizeof(typename real_type<T>::type)==4
                    ? std::min(kw,std::min(nb,128)) : kw;
                for (int offset=0;offset<kw;offset+=chunk) {
                const int diagonal_k=std::min(chunk,kw-offset);
                const T diagonal_beta=offset==0 ? scale : T(1);
                const T* diagonal_a=a+il+std::size_t(offset)*mr;
                const T* diagonal_a2=rank2 ? a2+il+std::size_t(offset)*mr : nullptr;
                if (symmetry=='H') {
                    if (rank2)
                        blas_impl::her2k_run<Backend,T>(h,uplo,'N',width,diagonal_k,alpha,diagonal_a,mr,
                            diagonal_a2,mr,std::real(diagonal_beta),out,ldc);
                    else
                        blas_impl::herk_run<Backend,T>(h,uplo,'N',width,diagonal_k,std::real(alpha),
                            diagonal_a,mr,std::real(diagonal_beta),out,ldc);
                } else {
                    if (rank2)
                        blas_impl::syr2k_run<Backend,T>(h,uplo,'N',width,diagonal_k,alpha,diagonal_a,mr,
                            diagonal_a2,mr,diagonal_beta,out,ldc);
                    else
                        blas_impl::syrk_run<Backend,T>(h,uplo,'N',width,diagonal_k,alpha,diagonal_a,mr,diagonal_beta,out,ldc);
                }
                }
            }
            j+=width;
        }
        g+=kw;
    }
}

}} // namespace prankk_impl::<anonymous>

namespace prankk_impl {

template <DdlaBackend Backend,typename T>
bool check(const DdlaHandle_t& h,char symmetry,char uplo,char ta,char tb,int n,int k,
    const T& alpha,const T* A,const int* da,const T* B,const int* db,const T& beta,
    const T* C,const int* dc,bool rank2)
{
    return detail::agree_check(h,"distributed rank update",[&](detail::Agree& agree) {
        symmetry=upper(symmetry);ta=upper(ta);tb=upper(tb);
        const char reflected=symmetry=='H' ? 'C' : 'T';
        if ((symmetry!='S' && symmetry!='H') || (ta!='N' && ta!=reflected) ||
            tb!=(ta=='N' ? reflected : 'N'))
            throw std::invalid_argument("distributed rank update: expected N/T for symmetric or N/C for Hermitian");
        if constexpr (!std::is_arithmetic_v<T>)
            if (symmetry=='H' && beta.imag()!=0)
                throw std::invalid_argument("distributed rank update: Hermitian beta must be real");
        const int operation=1+int(symmetry=='H')+2*int(rank2);
        return pblas::product_check<Backend,T>(h,agree,uplo,ta,tb,n,k,alpha,A,da,B,db,beta,C,dc,operation,symmetry=='H');
    });

}

template <typename T>
void workspace_size(const DdlaHandle_t& h, char symmetry, char uplo, char ta, char tb, bool product, bool rank2, int n, int k,
    const int* da, const int* db, const int* dc,
    std::size_t* backend_bytes, std::size_t* host_bytes)
{
    std::size_t count[4],add[2][2],triangle_bytes=0;
    scratch<T>(h,ta,tb,product,rank2,n,k,da,db,dc,count,add,triangle_bytes);
    *backend_bytes=triangle_bytes;
    *host_bytes=0;
    for (const std::size_t c : count) *backend_bytes+=detail::padded_bytes<T>(c);
    for (const auto& bytes : add) {
        *backend_bytes+=detail::padded_bytes<char>(bytes[0]);
        *host_bytes+=detail::padded_bytes<char>(bytes[1]);
    }
    if (n>0 && product && h->nprocs>1 && rank_batches<T>(h,n)) {
        std::size_t backend=0,host=0;
        rank_bytes<T>(h,uplo,ta,tb,product,rank2,n,k,dc,count,backend,host);
        *backend_bytes+=backend;
        *host_bytes+=host;
    }
}

template <DdlaBackend Backend, typename T>
void run(const DdlaHandle_t& h, char symmetry, char uplo, char ta, char tb, bool product, bool rank2,
    int n, int k, const T& alpha, const T* A, const int* da,
    const T* B, const int* db, const T& beta, T* C, const int* dc,
    void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes)
{
    std::size_t count[4],add[2][2],triangle_bytes=0;
    scratch<T>(h,ta,tb,product,rank2,n,k,da,db,dc,count,add,triangle_bytes);
    void* triangle_work=triangle_bytes ? static_cast<char*>(backend_work)+backend_bytes-triangle_bytes : nullptr;
    backend_bytes-=triangle_bytes;
    try {
        if(h->nprocs==1 && product) {
            // Small single-precision vendor rank kernels reduce long K
            // serially. Keep short contractions here as on distributed diagonals.
            int chunk=k;
            if constexpr (Backend==DdlaBackend::GPU && sizeof(typename real_type<T>::type)==4)
                if(n<=128)chunk=std::min(k,std::min(dc[DDLA_MB_],128));
            for(int first=0;first<k;first+=chunk) {
                const int width=std::min(chunk,k-first);
                const T* a=A+(ta=='N' ? std::size_t(first)*da[DDLA_LLD_] : first);
                const T* b=rank2 ? B+(ta=='N' ? std::size_t(first)*db[DDLA_LLD_] : first) : nullptr;
                const T scale=first==0 ? beta : T(1);
                if(symmetry=='H') {
                    if(rank2)blas_impl::her2k_run<Backend,T>(h,uplo,ta,n,width,alpha,a,da[DDLA_LLD_],b,db[DDLA_LLD_],std::real(scale),C,dc[DDLA_LLD_]);
                    else blas_impl::herk_run<Backend,T>(h,uplo,ta,n,width,std::real(alpha),a,da[DDLA_LLD_],std::real(scale),C,dc[DDLA_LLD_]);
                } else {
                    if(rank2)blas_impl::syr2k_run<Backend,T>(h,uplo,ta,n,width,alpha,a,da[DDLA_LLD_],b,db[DDLA_LLD_],scale,C,dc[DDLA_LLD_]);
                    else blas_impl::syrk_run<Backend,T>(h,uplo,ta,n,width,alpha,a,da[DDLA_LLD_],scale,C,dc[DDLA_LLD_]);
                }
            }
            return;
        }

        if (n==0 || (!product && beta==T(1) && symmetry!='H')) return;

        // The scratch, in the order workspace_size counts it.
        char* w=static_cast<char*>(backend_work);
        char* hw=static_cast<char*>(host_work);
        T* const a=detail::take<T>(w,count[0]);
        T* const b=detail::take<T>(w,count[1]);
        T* const work=detail::take<T>(w,count[2]);
        T* const b_tile=detail::take<T>(w,count[3]);
        char* add_work[2];
        char* add_host[2];
        for (int i=0;i<2;++i) {
            add_work[i]=detail::take<char>(w,add[i][0]);
            add_host[i]=detail::take<char>(hw,add[i][1]);
        }
        const std::size_t used=static_cast<std::size_t>(w-static_cast<char*>(backend_work));
        const std::size_t host_used=static_cast<std::size_t>(hw-static_cast<char*>(host_work));
        assert(used<=backend_bytes && host_used<=host_bytes);
        // One k panel of op(A) (left) or op(B) into output, through the exchange
        // buffer and the panel's pgeadd scratch.
        const auto load=[&](bool left,char trans,int g,int kw,const T* input,const int* di,T* output) {
            const int i=left ? 0 : 1;
            pblas::product_panel_run<Backend,T>(h,left,trans,n,g,kw,input,di,dc,output,work,b_tile,
                add_work[i],add[i][0],add_host[i],add[i][1]);
        };
        const bool batched=product && rank_batches<T>(h,n);
        if (batched) {
            rank_run<Backend,T>(h,symmetry,uplo,ta,tb,product,rank2,n,k,alpha,A,da,B,db,beta,C,dc,load,a,b,count[3] ? b_tile : work,count,
                                w,backend_bytes-used,hw,host_bytes-host_used);
            return;
        }
        const int nb=dc[DDLA_MB_],ldc=dc[DDLA_LLD_];
        const int r=h->myprow_,c=h->mypcol_,pr=h->nprows_,pc=h->npcols_;

        // A Hermitian input diagonal is real by definition; its unused imaginary
        // components must not enter beta*C, even when they contain NaNs.
        if (!batched && symmetry=='H' && beta!=T(0)) {
            for (int j=0;j<n;) {
                const int width=std::min(nb,n-j);
                if (r==indxg2p(j,nb,dc[DDLA_RSRC_],pr) && c==indxg2p(j,nb,dc[DDLA_CSRC_],pc)) {
                    const int il=num_loc(j,nb,r,dc[DDLA_RSRC_],pr),jl=num_loc(j,nb,c,dc[DDLA_CSRC_],pc);
                    blas_impl::real_diagonal<Backend,T>(h,width,C+il+std::size_t(jl)*ldc,ldc);
                }
                j+=width;
            }
        }

        // Each selected off-diagonal rectangle is one GEMM. Diagonal tiles use
        // the local triangular product, so no discarded output is multiplied.
        if (!batched) {
            for (int g=0;g<(product ? k : 1);) {
                const int kw=product ? std::min(pblas::contraction_width(h,nb,k),k-g) : 0;
                const int terms=rank2 && product ? 2 : 1;
                for (int term=0;term<terms;++term) {
                    T coefficient=alpha;
                    if constexpr (!std::is_arithmetic_v<T>)
                        if (term!=0 && symmetry=='H') coefficient=std::conj(alpha);
                    if (product) {
                        load(true,ta,g,kw,term==0 ? A : B,term==0 ? da : db,a);
                        load(false,tb,g,kw,term==0 ? B : A,term==0 ? db : da,b);
                    }
                    const T scale=g==0 && term==0 ? beta : T(1);
                    pblas::triangle_panel_run<Backend,T>(h,uplo,product,n,kw,coefficient,a,b,scale,
                        C,dc,triangle_work,triangle_bytes);
                }
                if (!product) break;
                g+=kw;
            }
        }

        if (symmetry=='H') {
            for (int j=0;j<n;) {
                const int width=std::min(nb,n-j);
                if (r==indxg2p(j,nb,dc[DDLA_RSRC_],pr) && c==indxg2p(j,nb,dc[DDLA_CSRC_],pc)) {
                    const int il=num_loc(j,nb,r,dc[DDLA_RSRC_],pr),jl=num_loc(j,nb,c,dc[DDLA_CSRC_],pc);
                    blas_impl::real_diagonal<Backend,T>(h,width,C+il+std::size_t(jl)*ldc,ldc);
                }
                j+=width;
            }
        }

    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(h) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(h); } catch (...) {}
        throw;
    }
}

} // namespace prankk_impl
} // namespace ddla

#define INSTANTIATE(B,T) \
 template bool ddla::prankk_impl::check<B,T>(const ddla::DdlaHandle_t&,char,char,char,char,int,int,const T&,const T*,const int*,const T*,const int*,const T&,const T*,const int*,bool); \
 template void ddla::prankk_impl::run<B,T>(const ddla::DdlaHandle_t&,char,char,char,char,bool,bool,int,int,const T&,const T*,const int*,const T*,const int*,const T&,T*,const int*,void*,std::size_t,void*,std::size_t)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE)
#define INSTANTIATE_SIZE(T) template void ddla::prankk_impl::workspace_size<T>(const ddla::DdlaHandle_t&,char,char,char,char,bool,bool,int,int,const int*,const int*,const int*,std::size_t*,std::size_t*)
DDLA_FOR_EACH_TYPE(INSTANTIATE_SIZE)
#undef INSTANTIATE_SIZE
#undef INSTANTIATE
