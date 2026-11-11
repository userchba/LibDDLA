#include "helpers.h"
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "ddla_desc.h"
#include "comm_traits.h"
#include "ptrtrm_impl.h"
#include "BLAS/blas_impl.h"
#include "BLAS/gemmt_impl.h"
#include "require_backend.h"
#include "check_agree.h"
#include "plan_util.h"
#include "pgeadd_impl.h"
#include "pblas_workspace.h"
#include "instantiate.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ddla {
namespace {

// Element counts of the broadcast panels of A / B (whole block rows/columns
// of nb x nb blocks).
std::size_t pack_a_count(int nrows,int nb) { return std::size_t(std::max(nrows,1))*nb*nb; }
std::size_t pack_b_count(int ncols,int nb) { return std::size_t(std::max(ncols,1))*nb*nb; }

// All triangular updates use the same staged pointer lists: count triples
// in the host lists ha / hb / hc, uploaded to stageA / stageB / stageC. Copies
// and GEMMs are ordered on the handle's queue before the next batch reuses
// the device lists; the host lists are pageable, so the uploads have read
// them when ddlaMemcpy returns and the next batch refills them right away.
template <DdlaBackend Backend, typename T>
void flush_batch(const DdlaHandle_t& h, const T** ha, const T** hb, T** hc, std::size_t& count,
                 const T** stageA, const T** stageB, T** stageC,
                 const T& alpha, int m, int n, int k, const T& beta,
                 int lda, int ldb, int ldc)
{
    try {
        if (count==0) return;
        DDLA_CHECK(ddlaMemcpy(h,stageA,ha,count*sizeof(const T*)));
        DDLA_CHECK(ddlaMemcpy(h,stageB,hb,count*sizeof(const T*)));
        DDLA_CHECK(ddlaMemcpy(h,stageC,hc,count*sizeof(T*)));
        blas_impl::gemmBatched_run<Backend,T>(h,'N','N',m,n,k,alpha,
            static_cast<const T* const*>(stageA),lda,
            static_cast<const T* const*>(stageB),ldb,beta,
            static_cast<T* const*>(stageC),ldc,static_cast<int>(count));
        count=0;
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(h) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(h); } catch (...) {}
        throw;
    }
}

// Used only for the zero-product path. Zero fills do not load old C, and a
// matching-triangle update never touches the opposite triangle or padding.
template <DdlaBackend Backend, typename T>
void scale_output(const DdlaHandle_t& h, char ua, char ub, int n,
                  const T& beta, T* C, const int* dc)
{
    if (n==0 || beta==T(1)) return;
    const int nb=dc[DDLA_MB_],r=h->myprow_,c=h->mypcol_,pr=h->nprows_,pc=h->npcols_;
    const int rs=dc[DDLA_RSRC_],cs=dc[DDLA_CSRC_],ldc=dc[DDLA_LLD_];
    const int rows=num_loc(n,nb,r,rs,pr),cols=num_loc(n,nb,c,cs,pc);
    for (int j=0;j<cols;++j) {
        const int global_j=indxl2g(j,nb,c,cs,pc);
        const int first=ua==ub && ua=='L' ? num_loc(global_j,nb,r,rs,pr) : 0;
        const int end=ua==ub && ua=='U' ? num_loc(global_j+1,nb,r,rs,pr) : rows;
        if (end<=first) continue;
        T* out=C+first+std::size_t(j)*ldc;
        if (beta==T(0)) DDLA_CHECK(ddlaMemset(h,out,0,std::size_t(end-first)*sizeof(T)));
        else blas_impl::geam_run<Backend,T>(h,'N','N',end-first,1,beta,out,ldc,T(0),
                             static_cast<const T*>(nullptr),1,out,ldc);
    }
}

template <typename T>
void ptrtrm_base_workspace_size(const DdlaHandle_t& handle, int n,
                           const int* descA, const int* descB, const int* descC,
                           std::size_t* backend_bytes, std::size_t* host_bytes)
{
    *backend_bytes = *host_bytes = 0;
    if (n == 0) return;
    const int pr=handle->nprows_,pc=handle->npcols_,r=handle->myprow_,c=handle->mypcol_;
    const int nb=descA[DDLA_MB_],rs=descA[DDLA_RSRC_],cs=descA[DDLA_CSRC_];
    const int lldA=descA[DDLA_LLD_],lldB=descB[DDLA_LLD_],lldC=descC[DDLA_LLD_];
    const int K=n>0?(n-1)/nb+1:0;
    const int shiftR=(r-rs+pr)%pr,shiftC=(c-cs+pc)%pc;
    const int nrows=shiftR<K?(K-1-shiftR)/pr+1:0;
    const int ncols=shiftC<K?(K-1-shiftC)/pc+1:0;
    const std::size_t maxbatch=std::size_t(nrows)*ncols+1;

    std::size_t local_bytes = 0;
    blas_impl::trtrm_workspace_size<T>(handle, nb, &local_bytes);
    *backend_bytes = detail::padded_bytes<T>(pack_a_count(nrows,nb)) + detail::padded_bytes<T>(pack_b_count(ncols,nb))
                   + 2 * detail::padded_bytes<const T*>(maxbatch) + detail::padded_bytes<T*>(maxbatch)
                   + local_bytes;
    *host_bytes = 2 * detail::padded_bytes<const T*>(maxbatch) + detail::padded_bytes<T*>(maxbatch)
                + detail::padded_bytes<std::size_t>(std::size_t(nrows))
                + detail::padded_bytes<std::size_t>(std::size_t(ncols));
}

// One k sweep for all four effective triangle pairs and N/T/C operands.
// Transpose is selected once by ptrtrm_run. The false specialization contains
// no pgeadd calls, descriptors, extra workspace query or transposed-panel views.
// Matching triangles retain the row/column/corner schedule; mixed triangles
// retain the prefix/prefix or suffix/suffix batches and their first-write beta.
template <bool UpperA, bool UpperB, bool Transpose, DdlaBackend Backend, typename T>
void ptrtrm_sweep(const DdlaHandle_t& handle, char ta, char tb, int n,
    const int* descA, const int* descB, const int* descC,
    const T& alpha, const T* A, const T* B, const T& beta, T* C,
    void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes)
{
    const int pr=handle->nprows_,pc=handle->npcols_,r=handle->myprow_,c=handle->mypcol_;
    const int nb=descA[DDLA_MB_],rs=descA[DDLA_RSRC_],cs=descA[DDLA_CSRC_];
    const int lldA=descA[DDLA_LLD_],lldB=descB[DDLA_LLD_],lldC=descC[DDLA_LLD_];
    const int K=(n-1)/nb+1;
    const int shiftR=(r-rs+pr)%pr,shiftC=(c-cs+pc)%pc;
    const int nrows=shiftR<K?(K-1-shiftR)/pr+1:0,ncols=shiftC<K?(K-1-shiftC)/pc+1:0;
    const std::size_t maxbatch=std::size_t(nrows)*ncols+1;
    auto width=[&](int g){return std::min(nb,n-g*nb);};
    std::size_t base_bytes=backend_bytes,base_host_bytes=host_bytes;
    if constexpr (Transpose)
        ptrtrm_base_workspace_size<T>(handle,n,descA,descB,descC,&base_bytes,&base_host_bytes);
    char* w=static_cast<char*>(backend_work);char* hw=static_cast<char*>(host_work);
    T* const packA=detail::take<T>(w,pack_a_count(nrows,nb));
    T* const packB=detail::take<T>(w,pack_b_count(ncols,nb));
    DDLA_CHECK(ddlaMemset(handle,packA,0,sizeof(T)));
    DDLA_CHECK(ddlaMemset(handle,packB,0,sizeof(T)));
    const T** const stageA=detail::take<const T*>(w,maxbatch);
    const T** const stageB=detail::take<const T*>(w,maxbatch);
    T** const stageC=detail::take<T*>(w,maxbatch);
    const T** const ha=detail::take<const T*>(hw,maxbatch);
    const T** const hb=detail::take<const T*>(hw,maxbatch);
    T** const hc=detail::take<T*>(hw,maxbatch);
    std::size_t* const offsA=detail::take<std::size_t>(hw,std::size_t(nrows));
    std::size_t* const offsB=detail::take<std::size_t>(hw,std::size_t(ncols));
    const std::size_t used=std::size_t(w-static_cast<char*>(backend_work));
    assert(used<=base_bytes);
    assert(std::size_t(hw-static_cast<char*>(host_work))<=base_host_bytes);
    void* const local=w;const std::size_t local_bytes=base_bytes-used;
    std::size_t staged=0;
    auto push=[&](const T* a,const T* b,T* out){
        assert(staged<maxbatch);ha[staged]=a;hb[staged]=b;hc[staged]=out;++staged;
    };
    auto flush=[&](int m,int nn,int k,const T& bta,int lda,int ldb,int ldc){
        flush_batch<Backend,T>(handle,ha,hb,hc,staged,stageA,stageB,stageC,alpha,m,nn,k,bta,lda,ldb,ldc);
    };
    const T one=T(1);
    for(int k=0;k<K;++k) {
        const int wk=width(k),k0=k*nb;
        const int ar=indxg2p(k0,nb,rs,pr),ac=indxg2p(k0,nb,cs,pc);
        std::size_t countA=0,countB=0;
        int panel_lda_a=wk;
        if constexpr (Transpose) {
            // pgeadd also moves blocks to their owners in op(A)/op(B).
            // A is a column-major local slab; B has the same layout as NN.
            const int fa=UpperA?0:k0,ma=UpperA?k0+wk:n-k0;
            const int fb=UpperB?k0:0,cols_b=UpperB?n-k0:k0+wk;
            int ap[DDLA_DLEN_],bp[DDLA_DLEN_];
            pblas::describe(ap,handle,ma,wk,nb,nb,indxg2p(fa,nb,rs,pr),ac);
            pblas::describe(bp,handle,wk,cols_b,nb,nb,ar,indxg2p(fb,nb,cs,pc),wk);
            void* add=static_cast<char*>(backend_work)+base_bytes;
            void* add_host=static_cast<char*>(host_work)+base_host_bytes;
            pgeadd_impl::pgeadd_run<Backend,T>(handle,ta,'N',ma,wk,T(1),A,
                (ta=='N'?fa:k0)+1,(ta=='N'?k0:fa)+1,descA,T(0),nullptr,1,1,ap,
                packA,1,1,ap,add,backend_bytes-base_bytes,add_host,host_bytes-base_host_bytes);
            pgeadd_impl::pgeadd_run<Backend,T>(handle,tb,'N',wk,cols_b,T(1),B,
                (tb=='N'?k0:fb)+1,(tb=='N'?fb:k0)+1,descB,T(0),nullptr,1,1,bp,
                packB,1,1,bp,add,backend_bytes-base_bytes,add_host,host_bytes-base_host_bytes);
            panel_lda_a=ap[DDLA_LLD_];
            for(int t=0;t<nrows;++t) {
                const int i=shiftR+t*pr;
                offsA[t]=(UpperA?i<=k:i>=k)?num_loc(i*nb-fa,nb,r,ap[DDLA_RSRC_],pr):0;
            }
            for(int t=0;t<ncols;++t) {
                const int j=shiftC+t*pc;
                offsB[t]=(UpperB?j>=k:j<=k)?std::size_t(wk)*num_loc(j*nb-fb,nb,c,bp[DDLA_CSRC_],pc):0;
            }
            if(r==ar && c==ac) {
                DDLA_CHECK(ddlaMemsetTri(handle, UpperA ? 'L' : 'U', wk, wk, packA+offsA[(k-shiftR)/pr], panel_lda_a, 0, false));
                DDLA_CHECK(ddlaMemsetTri(handle, UpperB ? 'L' : 'U', wk, wk, packB+offsB[(k-shiftC)/pc], wk, 0, false));
            }
            countA=std::size_t(num_loc(ma,nb,r,ap[DDLA_RSRC_],pr))*wk;
            countB=std::size_t(num_loc(cols_b,nb,c,bp[DDLA_CSRC_],pc))*wk;
        } else {
            // Preserve direct block copies and the unused-A-panel skip in NN.
            const bool need_packA=UpperA!=UpperB || (UpperA?k<K-1:k>0);
            if(need_packA) for(int t=0;t<nrows;++t) {
                const int i=shiftR+t*pr;offsA[t]=countA;
                if(UpperA?i<=k:i>=k) countA+=std::size_t(width(i))*wk;
            }
            for(int t=0;t<ncols;++t) {
                const int j=shiftC+t*pc;offsB[t]=countB;
                if(UpperB?j>=k:j<=k) countB+=std::size_t(wk)*width(j);
            }
            if(c==ac && need_packA) {
                const int qk=(k-shiftC)/pc;
                for(int t=0;t<nrows;++t) {
                    const int i=shiftR+t*pr;
                    if(!(UpperA?i<=k:i>=k)) continue;
                    T* out=packA+offsA[t];
                    DDLA_CHECK(ddlaMemcpy2D(handle,out,std::size_t(width(i))*sizeof(T),
                        A+std::size_t(t)*nb+std::size_t(qk)*nb*lldA,std::size_t(lldA)*sizeof(T),
                        std::size_t(width(i))*sizeof(T),wk));
                    if(i==k) DDLA_CHECK(ddlaMemsetTri(handle, UpperA ? 'L' : 'U', wk, wk, out, wk, 0, false));
                }
            }
            if(r==ar) {
                const int pk=(k-shiftR)/pr;
                for(int t=0;t<ncols;++t) {
                    const int j=shiftC+t*pc;
                    if(!(UpperB?j>=k:j<=k)) continue;
                    T* out=packB+offsB[t];
                    DDLA_CHECK(ddlaMemcpy2D(handle,out,std::size_t(wk)*sizeof(T),
                        B+std::size_t(pk)*nb+std::size_t(t)*nb*lldB,std::size_t(lldB)*sizeof(T),
                        std::size_t(wk)*sizeof(T),width(j)));
                    if(j==k) DDLA_CHECK(ddlaMemsetTri(handle, UpperB ? 'L' : 'U', wk, wk, out, wk, 0, false));
                }
            }
        }
        commBcast<Backend>(handle,CommScope::Row,packA,countA?countA:1,ac);
        commBcast<Backend>(handle,CommScope::Col,packB,countB?countB:1,ar);
        auto packed_lda_a=[&](int block_rows){if constexpr(Transpose)return panel_lda_a;else return block_rows;};
        if constexpr (UpperA==UpperB) {
            if(r==ar && c==ac) {
                const std::size_t p=std::size_t((k-shiftR)/pr)*nb,q=std::size_t((k-shiftC)/pc)*nb;
                const T* akk;const T* bkk;int lda,ldb;
                if constexpr(Transpose) {
                    akk=packA+offsA[(k-shiftR)/pr];bkk=packB+offsB[(k-shiftC)/pc];lda=panel_lda_a;ldb=wk;
                } else {akk=A+p+q*lldA;bkk=B+p+q*lldB;lda=lldA;ldb=lldB;}
                blas_impl::trtrm_run<Backend,T>(handle,UpperA?'U':'L','N',UpperB?'U':'L','N',wk,alpha,
                    akk,lda,bkk,ldb,beta,C+p+q*lldC,lldC,local,local_bytes);
            }
            auto row_b=[&](int t,std::size_t pk){
                if constexpr(Transpose)return packB+offsB[t];else return B+pk+std::size_t(t)*nb*lldB;
            };
            auto column_a=[&](int t,std::size_t qk){
                if constexpr(Transpose)return packA+offsA[t];else return A+std::size_t(t)*nb+qk*lldA;
            };
            // Row hook: block row k of C, first write (upper) / last write (lower).
            if (r == ar && ncols > 0 && (UpperA ? k < K-1 : k > 0)) {
                const std::size_t pk = std::size_t((k - shiftR) / pr) * nb;
                const T* akk = packA + offsA[(k - shiftR) / pr];
                const T bta = UpperA ? beta : one;
                for (int pass = 0; pass < 2; ++pass) {
                    for (int t = 0; t < ncols; ++t) {
                        const int j = shiftC + t * pc;
                        if (!(UpperA ? j > k : j < k)) continue;
                        if ((j == K - 1) != (pass == 1)) continue;
                        push(akk, row_b(t,pk), C + pk + std::size_t(t) * nb * lldC);
                    }
                    flush(wk, pass == 1 ? width(K - 1) : nb, wk,
                          bta, packed_lda_a(wk), Transpose ? wk : lldB, lldC);
                }
            }

            // Column hook: block column k of C, last write (upper) / first write (lower).
            if (c == ac && nrows > 0) {
                const std::size_t qk = std::size_t((k - shiftC) / pc) * nb;
                const T* bkk = packB + offsB[(k - shiftC) / pc];
                const T bta = UpperA ? one : beta;
                for (int pass = 0; pass < 2; ++pass) {
                    for (int t = 0; t < nrows; ++t) {
                        const int i = shiftR + t * pr;
                        if (!(UpperA ? i < k : i > k)) continue;
                        if ((i == K - 1) != (pass == 1)) continue;
                        push(column_a(t,qk), bkk, C + std::size_t(t) * nb + qk * lldC);
                    }
                    flush(pass == 1 ? width(K - 1) : nb, wk, wk,
                          bta, Transpose ? panel_lda_a : lldA, wk, lldC);
                }
            }

            // Off-diagonal corner: strictly inside the triangle interval of both
            // indices, so only the outer side can meet the partial last block.
            if (nrows > 0 && ncols > 0) {
                for (int pass = 0; pass < 2; ++pass) {
                    for (int t = 0; t < nrows; ++t) {
                        const int i = shiftR + t * pr;
                        if (!(UpperA ? i < k : i > k)) continue;
                        if ((i == K - 1) != (!UpperA && pass == 1)) continue;
                        for (int s = 0; s < ncols; ++s) {
                            const int j = shiftC + s * pc;
                            if (!(UpperA ? j > k : j < k)) continue;
                            if ((j == K - 1) != (UpperA && pass == 1)) continue;
                            push(packA + offsA[t], packB + offsB[s],
                                 C + std::size_t(t) * nb + std::size_t(s) * nb * lldC);
                        }
                    }
                    flush(UpperA ? nb : (pass == 1 ? width(K - 1) : nb),
                          UpperA ? (pass == 1 ? width(K - 1) : nb) : nb,
                          wk, one,
                          packed_lda_a(!UpperA && pass == 1 ? width(K - 1) : wk), wk, lldC);
                }
            }

        } else {
            // Full and partial last blocks must have separate lda/m/n values.
            // When beta==1, old and newly reached tiles can share one batch.
            const int passes=UpperA && beta!=T(1) ? 2 : 1;
            for (int tail_i=0;tail_i<2;++tail_i) for (int tail_j=0;tail_j<2;++tail_j) {
                const int mi=tail_i ? width(K-1) : nb,nj=tail_j ? width(K-1) : nb;
                for (int pass=0;pass<passes;++pass) {
                    for (int t=0;t<nrows;++t) {
                        const int i=shiftR+t*pr;
                        if (!(UpperA ? i<=k : i>=k) || ((i==K-1)!=(tail_i==1))) continue;
                        for (int s=0;s<ncols;++s) {
                            const int j=shiftC+s*pc;
                            if (!(UpperA ? j<=k : j>=k) || ((j==K-1)!=(tail_j==1))) continue;
                            const bool first=UpperA ? k==std::max(i,j) : k==0;
                            if (passes==2 && first!=(pass==1)) continue;
                            assert(staged<maxbatch);
                            ha[staged]=packA+offsA[t];
                            hb[staged]=packB+offsB[s];
                            hc[staged]=C+std::size_t(t)*nb+std::size_t(s)*nb*lldC;
                            ++staged;
                        }
                    }
                    const T bta=UpperA ? (passes==2 && pass==1 ? beta : T(1)) : (k==0 ? beta : T(1));
                    flush_batch<Backend,T>(handle,ha,hb,hc,staged,stageA,stageB,stageC,alpha,mi,nj,wk,bta,packed_lda_a(mi),wk,lldC);
                }
            }

        }
    }
}

template <bool Transpose,DdlaBackend Backend,typename T>
void dispatch_sweep(const DdlaHandle_t& h,char ua,char ub,char ta,char tb,int n,
    const int* da,const int* db,const int* dc,const T& alpha,const T* A,const T* B,
    const T& beta,T* C,void* w,std::size_t bytes,void* hw,std::size_t host_bytes)
{
    if(ua=='U') {
        if(ub=='U') ptrtrm_sweep<true,true,Transpose,Backend,T>(h,ta,tb,n,da,db,dc,alpha,A,B,beta,C,w,bytes,hw,host_bytes);
        else ptrtrm_sweep<true,false,Transpose,Backend,T>(h,ta,tb,n,da,db,dc,alpha,A,B,beta,C,w,bytes,hw,host_bytes);
    } else {
        if(ub=='U') ptrtrm_sweep<false,true,Transpose,Backend,T>(h,ta,tb,n,da,db,dc,alpha,A,B,beta,C,w,bytes,hw,host_bytes);
        else ptrtrm_sweep<false,false,Transpose,Backend,T>(h,ta,tb,n,da,db,dc,alpha,A,B,beta,C,w,bytes,hw,host_bytes);
    }
}

} // namespace

namespace ptrtrm_impl {

// Phase 1: validate the stored triangles, operations and distributed layout.

template <DdlaBackend Backend, typename T>
void ptrtrm_check(const DdlaHandle_t& h, char ua, char ta, char ub, char tb, int n,
    const T& alpha, const T* A, const int* da, const T* B, const int* db,
    const T& beta, const T* C, const int* dc)
{
    detail::agree_check(h,"ptrtrm",[&](detail::Agree& agree) {
        detail::require_backend<Backend>(h,"ptrtrm");
        check_desc(da,h); check_desc(db,h); check_desc(dc,h);
        ua=static_cast<char>(std::toupper(static_cast<unsigned char>(ua)));
        ub=static_cast<char>(std::toupper(static_cast<unsigned char>(ub)));
        ta=static_cast<char>(std::toupper(static_cast<unsigned char>(ta)));
        tb=static_cast<char>(std::toupper(static_cast<unsigned char>(tb)));
        if ((ta!='N' && ta!='T' && ta!='C') || (tb!='N' && tb!='T' && tb!='C'))
            throw std::invalid_argument("ptrtrm: transa/transb must be N, T or C");
        const char ea=ta=='N'?ua:(ua=='U'?'L':'U');
        const char eb=tb=='N'?ub:(ub=='U'?'L':'U');
        if ((ua!='U' && ua!='L') || (ub!='U' && ub!='L'))
            throw std::invalid_argument("ptrtrm: both input triangles must be U or L");
        if (n<0 || n>da[DDLA_M_] || n>da[DDLA_N_] || n>db[DDLA_M_] ||
            n>db[DDLA_N_] || n>dc[DDLA_M_] || n>dc[DDLA_N_])
            throw std::invalid_argument("ptrtrm: order outside descriptors");
        const int nb=da[DDLA_MB_],rs=da[DDLA_RSRC_],cs=da[DDLA_CSRC_];
        if (da[DDLA_NB_]!=nb || db[DDLA_MB_]!=nb || db[DDLA_NB_]!=nb ||
            dc[DDLA_MB_]!=nb || dc[DDLA_NB_]!=nb)
            throw std::invalid_argument("ptrtrm: A, B and C need matching square blocks");
        if (db[DDLA_RSRC_]!=rs || dc[DDLA_RSRC_]!=rs || db[DDLA_CSRC_]!=cs || dc[DDLA_CSRC_]!=cs)
            throw std::invalid_argument("ptrtrm: A, B and C need matching source coordinates");
        // One mode covers all 36 flag combinations without expanding Agree.
        const auto op_mode=[](char t){return t=='N'?0:t=='T'?1:2;};
        agree.mode("ptrtrm operand flags",(ua=='U'?2:0)+(ub=='U')+4*op_mode(ta)+12*op_mode(tb));
        agree.mode("ptrtrm order",n);
        agree.mode("ptrtrm coefficient modes",(alpha==T(0) ? 4 : 0)+(beta==T(0) ? 2 : 0)+(beta==T(1)));
        agree.mode("ptrtrm block size",nb);
        agree.mode("ptrtrm row source",rs);
        agree.mode("ptrtrm column source",cs);
        if (n==0 || (alpha==T(0) && beta==T(1))) return;
        const int r=h->myprow_,c=h->mypcol_,pr=h->nprows_,pc=h->npcols_;
        const int rows=num_loc(n,nb,r,rs,pr),cols=num_loc(n,nb,c,cs,pc);
        const auto stored=[&](char u) {
            if (rows==0 || cols==0) return false;
            return u=='U' ? indxl2g(0,nb,r,rs,pr)<=indxl2g(cols-1,nb,c,cs,pc)
                          : indxl2g(rows-1,nb,r,rs,pr)>=indxl2g(0,nb,c,cs,pc);
        };
        const bool output=ea==eb ? stored(ea) : rows>0 && cols>0;
        if (output && !C) throw std::invalid_argument("ptrtrm: null C");
        if (alpha!=T(0)) {
            if ((stored(ua) && !A) || (stored(ub) && !B))
                throw std::invalid_argument("ptrtrm: null input");
            if (detail::matrix_overlaps(A,da[DDLA_LLD_],rows,cols,C,dc[DDLA_LLD_],rows,cols) ||
                detail::matrix_overlaps(B,db[DDLA_LLD_],rows,cols,C,dc[DDLA_LLD_],rows,cols))
                throw std::invalid_argument("ptrtrm: C must not overlap A or B");
        }
        return;
    });
}

// Extra redistribution workspace is shared by both panel extractions.
template <typename T>
void ptrtrm_workspace_size(const DdlaHandle_t& h, char ta, char tb, int n,
    const int* da, const int* db, const int* dc, std::size_t* bytes, std::size_t* host)
{
    ptrtrm_base_workspace_size<T>(h,n,da,db,dc,bytes,host);
    if(n==0 || (ta=='N' && tb=='N')) return;
    std::size_t extra=0,extra_host=0;
    const int nb=da[DDLA_MB_];
    for(int operand=0;operand<2;++operand) {
        int d[DDLA_DLEN_];
        pblas::describe(d,h,operand?nb:n,operand?n:nb,nb,nb,0,0);
        std::size_t b=0,hb=0;
        pgeadd_impl::pgeadd_workspace_size<T>(h,operand?tb:ta,'N',true,false,
            d[DDLA_M_],d[DDLA_N_],d,&b,&hb);
        extra=std::max(extra,b);extra_host=std::max(extra_host,hb);
    }
    *bytes+=extra;*host+=extra_host;
}

// Phase 3: execute the normalized four-flag operation on reusable scratch.

template <DdlaBackend Backend, typename T>
void ptrtrm_run(const DdlaHandle_t& h, char ua, char ta, char ub, char tb, int n,
    const T& alpha, const T* A, const int* da, const T* B, const int* db,
    const T& beta, T* C, const int* dc, void* work, std::size_t bytes,
    void* host, std::size_t host_bytes)
{
    if(n==0 || (alpha==T(0) && beta==T(1))) return;
    const char ea=ta=='N'?ua:(ua=='U'?'L':'U'),eb=tb=='N'?ub:(ub=='U'?'L':'U');
    if(alpha==T(0)) scale_output<Backend,T>(h,ea,eb,n,beta,C,dc);
    else if(ta=='N' && tb=='N')
        dispatch_sweep<false,Backend,T>(h,ea,eb,ta,tb,n,da,db,dc,alpha,A,B,beta,C,work,bytes,host,host_bytes);
    else
        dispatch_sweep<true,Backend,T>(h,ea,eb,ta,tb,n,da,db,dc,alpha,A,B,beta,C,work,bytes,host,host_bytes);
}

}} // namespace ddla::ptrtrm_impl

namespace ddla {

template <DdlaBackend Backend, typename T>
void ptrtrm(const DdlaHandle_t& h, char ua, char ta, char ub, char tb, int n,
    const T& alpha, const T* A, const int* da, const T* B, const int* db,
    const T& beta, T* C, const int* dc)
{
    ptrtrm_impl::ptrtrm_check<Backend,T>(h,ua,ta,ub,tb,n,alpha,A,da,B,db,beta,C,dc);
    auto upper=[](char x){return static_cast<char>(std::toupper(static_cast<unsigned char>(x)));};
    ua=upper(ua);ub=upper(ub);ta=upper(ta);tb=upper(tb);
    if(n==0 || (alpha==T(0) && beta==T(1))) return;
    if(alpha==T(0)) {
        scale_output<Backend,T>(h,ta=='N'?ua:(ua=='U'?'L':'U'),tb=='N'?ub:(ub=='U'?'L':'U'),n,beta,C,dc);
        return;
    }
    std::size_t bytes=0,host_bytes=0;
    ptrtrm_impl::ptrtrm_workspace_size<T>(h,ta,tb,n,da,db,dc,&bytes,&host_bytes);
    vector<char> work(h,bytes);std::vector<char> host(host_bytes);
    ptrtrm_impl::ptrtrm_run<Backend,T>(h,ua,ta,ub,tb,n,alpha,A,da,B,db,beta,C,dc,
        work.data(),bytes,host.data(),host_bytes);
}

#define INSTANTIATE_PTRTRM(BACKEND, TYPE) \
    template void ptrtrm<BACKEND, TYPE>(const DdlaHandle_t&,char,char,char,char,int,const TYPE&,const TYPE*,const int*,const TYPE*,const int*,const TYPE&,TYPE*,const int*); \
    template void ptrtrm_impl::ptrtrm_check<BACKEND, TYPE>(const DdlaHandle_t&,char,char,char,char,int,const TYPE&,const TYPE*,const int*,const TYPE*,const int*,const TYPE&,const TYPE*,const int*); \
    template void ptrtrm_impl::ptrtrm_run<BACKEND, TYPE>(const DdlaHandle_t&,char,char,char,char,int,const TYPE&,const TYPE*,const int*,const TYPE*,const int*,const TYPE&,TYPE*,const int*,void*,std::size_t,void*,std::size_t)
#define INSTANTIATE_PTRTRM_SIZE(TYPE) \
    template void ptrtrm_impl::ptrtrm_workspace_size<TYPE>(const DdlaHandle_t&,char,char,int,const int*,const int*,const int*,std::size_t*,std::size_t*)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_PTRTRM)
DDLA_FOR_EACH_TYPE(INSTANTIATE_PTRTRM_SIZE)
#undef INSTANTIATE_PTRTRM_SIZE
#undef INSTANTIATE_PTRTRM

} // namespace ddla
