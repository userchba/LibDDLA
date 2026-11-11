// ---------------------------------------------------------------------------
// Stage 1 of the two-stage Hermitian eigensolver: full -> band (ELPA2
// bandred).  Panels are processed backward; each panel is factored by
// phe2hb_panel_run (reflectors stored above the band edge in A) and its
// effect on the rest of the leading block is applied with the blocked
// two-sided update
//
//     A := A - V U^H - U V^H,   U = (A V) T^H - 1/2 V T (V^H A V) T^H,
//
// where V holds the panel reflectors and the T factor comes from
// phe2hb_panel_run, which stores it (T = T_L^H of the panel's QL factorization, no T
// of its own is built here) into the replicated tmat slot (Qp = I - V T V^H = H_1 ... H_m).  The update
// region is the full leading [0, panel_start) x [0, panel_start) block: V's
// support stops at the panel's largest pivot row, but U carries A's full
// leading rows, so the U V^H term reaches every leading row and the V U^H
// term every leading column.  (Sizing the region by an lr_end-based bound
// instead would be wrong.)  A private CPU candidate keeps only coarse
// upper offdiagonal tiles and full diagonal tiles: its AV reconstructs both
// contributions from that storage, and its update skips unused lower tiles.
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>

#include "phe2hb_impl.h"
#include "BLAS/blas_impl.h"
#include "QR/pgeqrf_impl.h"
#include "check_agree.h"
#include "plan_util.h"
#include "Eigensolver/pvector_redist.h"
#include "ddla_connector.h"
#include "comm_traits.h"
#include "ddla_desc.h"
#include "instantiate.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <complex>
#include <exception>
#include <climits>
#include <new>
#include <type_traits>
#include <stdexcept>
#include <vector>

namespace ddla {

namespace spectral_impl {
template<class T>void phe2hb_upper_av_run(const DdlaHandle_t&,const Phe2hbPlan&,int,const T*,const T*,const T*,int,T*,T*,T*,T*,void*,std::size_t);
template<class T>void phe2hb_upper_update_run(const DdlaHandle_t&,const Phe2hbPlan&,int,T*,const T*,const T*,int);
Phe2hbPlan phe2hb_geometry(const DdlaHandle_t& h,DdlaBackend backend,int n,const int* desc,int nbw,int requested_upper_tile,bool inplace)
{
    Phe2hbPlan p;p.backend=backend;p.n=n;p.nbw=nbw;p.nb=desc[DDLA_NB_];p.lld=desc[DDLA_LLD_];p.inplace=inplace;
    std::copy_n(desc,DDLA_DLEN_,p.descriptor.data());p.nprows=h->nprows_;p.npcols=h->npcols_;p.myprow=h->myprow_;p.mypcol=h->mypcol_;
    p.steps=n?(n-1)/nbw:0;p.extent=p.steps*nbw;p.ldv=std::max(1,num_loc(p.extent,p.nb,p.myprow,0,p.nprows));p.ldc=std::max(1,num_loc(p.extent,p.nb,p.mypcol,0,p.npcols));
    const int tile=requested_upper_tile?std::max(p.nb,(requested_upper_tile/p.nb)*p.nb):0;p.upper_tile=tile&&p.extent>tile?tile:0;return p;
}
Phe2hbPlan phe2hb_check(const DdlaHandle_t& h,DdlaBackend backend,int n,const void* A,const int* desc,
    int nbw,const void* tmat,int requested_upper_tile,bool automatic_cpu_qr)
{
    auto p=detail::agree_check(h,"phe2hb",[&](detail::Agree& agreement) {
        agreement.mode("spectral operation",3122);Phe2hbPlan result;
        if(!h||ddlaGetBackend(h)!=backend)throw std::invalid_argument("phe2hb: invalid handle/backend");
        check_desc(desc,h);
        if(n<0||n>desc[DDLA_M_]||n>desc[DDLA_N_]||desc[DDLA_MB_]!=desc[DDLA_NB_]||
            nbw<1||nbw>desc[DDLA_NB_]||desc[DDLA_NB_]%nbw||requested_upper_tile<0||
            ((requested_upper_tile||automatic_cpu_qr)&&backend!=DdlaBackend::CPU))
            throw std::invalid_argument("phe2hb: invalid dimensions, block size or CPU policy");
        if(desc[DDLA_RSRC_]!=0||desc[DDLA_CSRC_]!=0)
            throw std::invalid_argument("phe2hb: source-zero block distribution required");
        result=phe2hb_geometry(h,backend,n,desc,nbw,requested_upper_tile);
        if(result.steps&&(!tmat||(!A&&num_loc(n,result.nb,result.myprow,0,result.nprows)>0&&num_loc(n,result.nb,result.mypcol,0,result.npcols)>0)))
            throw std::invalid_argument("phe2hb: missing active matrix or reflector factors");
        if(static_cast<long long>(result.ldv)*nbw>INT_MAX||static_cast<long long>(result.ldc)*nbw>INT_MAX||
            static_cast<long long>(nbw)*nbw>INT_MAX||result.extent>INT_MAX-(result.nb-1))
            throw std::overflow_error("phe2hb: vector collective count exceeds int");
        return result;
    });
    int signature[16]={3122,int(backend),n,nbw,requested_upper_tile,automatic_cpu_qr?1:0,h->nprows_,h->npcols_,int(h->major)};
    detail::descriptor_signature(signature+9,p.descriptor.data());detail::agree_signature(h,"phe2hb",signature);
    if(!p.steps)return p;
    if(automatic_cpu_qr)p.inplace=detail::phe2hb_cpu_qr_inplace(h,n,desc,nbw);
    else
    {
        p.inplace=qr_impl::qr_panel_inplace_env();const int mode[]={p.inplace?1:0};detail::agree_signature(h,"phe2hb QR policy",mode);
    }
    return p;
}

template<class T>void phe2hb_workspace_size(const DdlaHandle_t& h,const Phe2hbPlan& p,T* A,T* tau,
    std::size_t* backend_bytes,std::size_t* host_bytes)
{
    *backend_bytes=*host_bytes=0;if(!p.steps)return;
    const std::size_t ldv=p.ldv,ldc=p.ldc,nbw=p.nbw;
    std::size_t panel=0,transpose=0;
    phe2hb_panel_workspace_size<T>(h,p.descriptor.data(),p.n,p.nbw,p.inplace,A,tau,&panel,host_bytes);
    ptranspose_vectors_workspace_size<T>(h,p.extent,p.nbw,p.nb,&transpose);
    *backend_bytes=detail::padded_bytes<T>((ldv+(p.upper_tile>0?std::max(ldv,ldc):ldv))*nbw)
        +detail::padded_bytes<T>(ldc*2*nbw)+2*detail::padded_bytes<T>(ldv*nbw)
        +3*detail::padded_bytes<T>(nbw*nbw)+std::max(panel,transpose);
}

template<DdlaBackend Backend,class T>
void phe2hb_run(const DdlaHandle_t& handle,const Phe2hbPlan& policy,T* A,T* tmat,int& info,
    void* backend_work,std::size_t backend_bytes,void* host_work,std::size_t host_bytes)
{
    try {
        static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                      "GPU phe2hb is not available in this LibDDLA build");
        info=0;if(!policy.steps)return;
        const int n=policy.n,nbw=policy.nbw,nb=policy.nb,lld=policy.lld,nsteps=policy.steps;
        const int nprows=policy.nprows,npcols=policy.npcols,myprow=policy.myprow,mypcol=policy.mypcol;
        const int ld_v=policy.ldv,ld_c=policy.ldc,upper_tile=policy.upper_tile;
        const int* descA=policy.descriptor.data();
        // The buffers in the order phe2hb_workspace_size counts them; the panel's QL
        // factorization and the transposes take turns on the rest.
        char* const base=static_cast<char*>(backend_work);char* w=base;
        T* vmr=detail::take<T>(w,(std::size_t(ld_v)+(upper_tile>0?std::max(ld_v,ld_c):ld_v))*nbw);
        T* Vc=detail::take<T>(w,std::size_t(ld_c)*2*nbw);
        T* Yp=detail::take<T>(w,std::size_t(ld_v)*nbw);T* Y=detail::take<T>(w,std::size_t(ld_v)*nbw);
        T* vav=detail::take<T>(w,std::size_t(nbw)*nbw);T* Zbuf=detail::take<T>(w,std::size_t(nbw)*nbw);
        T* M=detail::take<T>(w,std::size_t(nbw)*nbw);
        assert(static_cast<std::size_t>(w-base)<=backend_bytes);
        void* const child=w;const std::size_t child_bytes=backend_bytes-static_cast<std::size_t>(w-base);
        const auto reduce=[&](CommScope scope,const T* src,T* dst,std::size_t count) {
            // Keep the complex CCL real-component count representable as int.
            for(std::size_t offset=0;offset<count;) {
                const int length=static_cast<int>(std::min(count-offset,std::size_t(INT_MAX/2)));
                commAllReduce<Backend>(handle,scope,src+offset,dst+offset,length,cclSum);offset+=length;
            }
        };
        for (int istep = nsteps; istep >= 1; --istep) {
            const int panel_start = istep * nbw;                     // 0-based
            const int n_cols = std::min(n, panel_start + nbw) - panel_start;
            // Update region: the full leading [0, panel_start)^2 block.  V's
            // support ends at the largest pivot row (panel_start + n_cols - nbw
            // - 1), but U = (A V) T^H - ... has A's full leading rows, so the
            // U V^H term reaches every leading row and the V U^H term every
            // leading column -- exactly the leading panel_start-square.
            const int l_rows = ddla::num_loc(panel_start, nb, myprow, 0, nprows);
            const int l_cols = ddla::num_loc(panel_start, nb, mypcol, 0, npcols);
            // vmr column blocks:  V in [0, n_cols),  U in [n_cols, 2*n_cols).
            // Vc  column blocks:  Uc in [0, n_cols), Vc in [n_cols, 2*n_cols).
            // The trailing GEMM then walks k = 0..2*n_cols-1 pairing column j of
            // both operands, so its two halves are exactly V*Uc^H and U*Vc^H.
            T* U = vmr + static_cast<std::size_t>(n_cols) * ld_v;
            T* Vcp = Vc + static_cast<std::size_t>(n_cols) * ld_c;
            T* Ucp = Vc;
            // This panel's T factor: the panel step stores it (T_L^H, stream-ordered) into its
            // tmat slot before step 5 and step 6 read it.
            T* const Tstep = tmat + static_cast<std::size_t>(istep - 1) * nbw * nbw;

            // ---- 1) panel factorization (also produces V in vmr and T in the tmat slot) ----
            phe2hb_panel_run<Backend,T>(handle,A,descA,panel_start,n_cols,nbw,policy.inplace,
                vmr,ld_v,Tstep,child,child_bytes,host_work,host_bytes);
            const auto trans=ptranspose_vectors_geometry(handle,Backend,panel_start,n_cols,nb,ld_v,ld_c,false,false);
            // ---- 2) (the T factor was produced by the panel step) ----------------

            // ---- 3) V to the column layout --------------------------------------
            ptranspose_vectors_run<Backend,T>(handle,trans,vmr,Vcp,child,child_bytes);

            // ---- 4) Y = A V: local partials, summed over the process row --------
            {
                if constexpr (Backend == DdlaBackend::CPU) {
                    if(upper_tile>0) {
                        // U is not live until Step6. Its private allocation also
                        // fits ld_c*n_cols for the temporary column result.
                        phe2hb_upper_av_run<T>(handle,policy,panel_start,A,vmr,Vcp,n_cols,Yp,Y,Ucp,U,child,child_bytes);
                    }
                }
                if (Backend != DdlaBackend::CPU || upper_tile == 0) {
                    DDLA_CHECK(ddlaMemset(handle,Yp,0,static_cast<std::size_t>(ld_v)*n_cols*sizeof(T)));
                    if(l_rows>0&&l_cols>0)
                        blas_impl::gemm_run<Backend,T>(handle,'N','N',l_rows,n_cols,l_cols,T(1),A,lld,Vcp,ld_c,T(0),Yp,ld_v);
                    reduce(CommScope::Row,Yp,Y,std::size_t(ld_v)*n_cols);
                }
            }

            // ---- 5) Z = V^H Y (Col-allreduce), M = T Z T^H in backend memory ------
            {
                DDLA_CHECK(ddlaMemset(handle, Zbuf, 0, static_cast<std::size_t>(nbw) * nbw * sizeof(T)));
                if (l_rows > 0) {
                    blas_impl::gemm_run<Backend, T>(handle, 'C', 'N', n_cols, n_cols, l_rows,
                                     T(1), vmr, ld_v, Y, ld_v, T(0), Zbuf, nbw);
                }
                reduce(CommScope::Col,Zbuf,vav,std::size_t(nbw)*nbw);
                if (l_rows > 0) {
                    // M = T Z T^H stays in backend memory as two GEMMs (the old host
                    // loop paid a D2H of Z + two n_cols^3 loops + an H2D of M,
                    // i.e. two host synchronizations per panel).  Same-node A/B
                    // (job 3581306, c2v100m2-1, one bench binary, three
                    // counterbalanced rounds, medians): stage 1 1.436 -> 1.417 s
                    // at n=10000 (nblk,nbw) = (16,16), 1.434 -> 1.420 s at
                    // (32,16), 1.145 -> 1.125 s at (32,32), and 0.557 -> 0.546 s
                    // at n=5000 (16,16) -- a consistent 11-20 ms (~1%) off
                    // stage 1, beyond the ~1-2 ms round spread.  Reusing Zbuf
                    // as the W = T Z scratch is ordered behind the allreduce that
                    // reads it in every build: CCL collectives are enqueued on
                    // static_cast<gpuStream_t>(handle->stream) (the GEMMs follow on the same stream), and
                    // the tunnel/direct-MPI paths synchronize inside
                    // commAllReduce.  Only the leading n_cols x n_cols block of
                    // M is written (beta = 0), and that is the only block step
                    // 6 reads.
                    blas_impl::gemm_run<Backend, T>(handle, 'N', 'N', n_cols, n_cols, n_cols,
                                     T(1), Tstep, nbw, vav, nbw, T(0), Zbuf, nbw);
                    blas_impl::gemm_run<Backend, T>(handle, 'N', 'C', n_cols, n_cols, n_cols,
                                     T(1), Zbuf, nbw, Tstep, nbw, T(0), M, nbw);
                }
            }

            // ---- 6) U = Y T^H - 1/2 V M  (row-distributed) ----------------------
            {
                if (l_rows > 0) {
                    blas_impl::gemm_run<Backend, T>(handle, 'N', 'C', l_rows, n_cols, n_cols,
                                     T(1), Y, ld_v, Tstep, nbw, T(0), U, ld_v);
                    blas_impl::gemm_run<Backend, T>(handle, 'N', 'N', l_rows, n_cols, n_cols,
                                     T(-0.5), vmr, ld_v, M, nbw, T(1), U, ld_v);
                }
                DDLA_CHECK(ddlaSynchronize(handle));
            }

            // ---- 7) U to the column layout --------------------------------------
            ptranspose_vectors_run<Backend,T>(handle,trans,U,Ucp,child,child_bytes);

            // ---- 8) update: A -= V U^H + U V^H over the leading block -----------
            // One GEMM of width 2*n_cols, with the two rank-1 terms packed side by
            // side: [V|U] (row-distributed) against [Uc|Vc] (column-distributed)
            // accumulates exactly V*Uc^H + U*Vc^H on the first 2*n_cols rows of
            // the k-dimension of each operand.  Same flops as two GEMMs, but A is
            // read and written once per panel instead of twice.  Same-node A/B at
            // n=20000 (4xV100, one job, interleaved rounds): 4.19/4.24 s fused vs
            // 4.38/4.33 s split for real, 6.94/6.98 s vs 7.48/7.49 s for complex
            // -- a consistent 4-7% on stage 1.
            {
                if constexpr (Backend == DdlaBackend::CPU) {
                    if(upper_tile>0) {
                        phe2hb_upper_update_run<T>(handle,policy,panel_start,A,vmr,Vc,2*n_cols);
                        continue;
                    }
                }
                if (l_rows > 0 && l_cols > 0) {
                    const int k2 = 2 * n_cols;
                    blas_impl::gemm_run<Backend, T>(handle, 'N', 'C', l_rows, l_cols, k2,
                                     T(-1), vmr, ld_v, Vc, ld_c, T(1), A, lld);
                }
                DDLA_CHECK(ddlaSynchronize(handle));
            }
        }

    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(handle) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(handle); } catch (...) {}
        throw;
    }
}

} // namespace spectral_impl

template<DdlaBackend Backend,class T>
static void phe2hb_impl(const DdlaHandle_t& h,int n,T* A,const int* desc,int nbw,T* tmat,int& info,
    int upper_tile,int panel_policy=-1)
{
    info=0;auto p=spectral_impl::phe2hb_check(h,Backend,n,A,desc,nbw,tmat,upper_tile,false);
    if(panel_policy>=0)p.inplace=panel_policy!=0;
    if(!p.steps)return;
    // size and allocation may fail on one rank only: agree before the collective run
    std::size_t backend_bytes=0,host_bytes=0;
    vector<char> backend_work(h);std::vector<char> host_work;std::exception_ptr failure;
    try{
        spectral_impl::phe2hb_workspace_size<T>(h,p,A,tmat,&backend_bytes,&host_bytes);
        backend_work.resize(backend_bytes);host_work.resize(host_bytes);
    }catch(...){failure=std::current_exception();}
    detail::agree_failure(h->comm,failure,"phe2hb: workspace preparation");
    spectral_impl::phe2hb_run<Backend,T>(h,p,A,tmat,info,backend_work.data(),backend_bytes,host_work.data(),host_bytes);
}

template <DdlaBackend Backend,typename T>
void phe2hb(const DdlaHandle_t& h,int n,T* A,const int* desc,int nbw,T* tmat,int& info)
{
    phe2hb_impl<Backend,T>(h,n,A,desc,nbw,tmat,info,0);
}

namespace spectral_impl {
template<class T>
void phe2hb_upper_av_run(const DdlaHandle_t& h,const Phe2hbPlan& p,int extent,const T* A,
    const T* Vr,const T* Vc,int nvec,T* Wr_partial,T* Wr,T* Wc_partial,T* Wc,
    void* work,std::size_t bytes)
{
    std::exception_ptr failure;
    const int nb=p.nb,lld=p.lld,pr=p.nprows,pc=p.npcols,mr=p.myprow,mc=p.mypcol,ldr=p.ldv,ldc=p.ldc,tile=p.upper_tile;
    try {
        DDLA_CHECK(ddlaMemset(h,Wr_partial,0,std::size_t(ldr)*nvec*sizeof(T)));
        DDLA_CHECK(ddlaMemset(h,Wc_partial,0,std::size_t(ldc)*nvec*sizeof(T)));
        for(int first=0;first<extent;) {
            const int edge=first+std::min(tile,extent-first);
            const int r0=num_loc(first,nb,mr,0,pr),r1=num_loc(edge,nb,mr,0,pr);
            const int c0=num_loc(first,nb,mc,0,pc),c1=num_loc(edge,nb,mc,0,pc);
            if(r1>0&&c1>c0)blas_impl::gemm_run<DdlaBackend::CPU,T>(h,'N','N',r1,nvec,c1-c0,T(1),
                A+std::size_t(c0)*lld,lld,Vc+c0,ldc,T(1),Wr_partial,ldr);
            if(r0>0&&c1>c0)blas_impl::gemm_run<DdlaBackend::CPU,T>(h,'C','N',c1-c0,nvec,r0,T(1),
                A+std::size_t(c0)*lld,lld,Vr,ldr,T(1),Wc_partial+c0,ldc);
            first=edge;
        }
    }catch(...){failure=std::current_exception();}
    detail::agree_failure(h,failure,"CPU upper band reduction");
    commAllReduce<DdlaBackend::CPU>(h,CommScope::Row,Wr_partial,Wr,ldr*nvec,cclSum);
    commAllReduce<DdlaBackend::CPU>(h,CommScope::Col,Wc_partial,Wc,ldc*nvec,cclSum);
    const auto reverse=ptranspose_vectors_geometry(h,DdlaBackend::CPU,extent,nvec,nb,ldc,ldr,true,true);
    ptranspose_vectors_run<DdlaBackend::CPU,T>(h,reverse,Wc,Wr,work,bytes);
}

template<class T>
void phe2hb_upper_update_run(const DdlaHandle_t& h,const Phe2hbPlan& p,int extent,T* A,
    const T* rows,const T* cols,int inner)
{
    std::exception_ptr failure;
    try {
        const int nb=p.nb,lld=p.lld,pr=p.nprows,pc=p.npcols,mr=p.myprow,mc=p.mypcol,tile=p.upper_tile;
        for(int first=0;first<extent;) {
            const int edge=first+std::min(tile,extent-first),nr=num_loc(edge,nb,mr,0,pr);
            const int c0=num_loc(first,nb,mc,0,pc),c1=num_loc(edge,nb,mc,0,pc);
            if(nr>0&&c1>c0)blas_impl::gemm_run<DdlaBackend::CPU,T>(h,'N','C',nr,c1-c0,inner,T(-1),rows,p.ldv,
                cols+c0,p.ldc,T(1),A+std::size_t(c0)*lld,lld);
            first=edge;
        }
        DDLA_CHECK(ddlaSynchronize(h));
    }catch(...){failure=std::current_exception();}
    detail::agree_failure(h,failure,"CPU upper band reduction");
}
} // namespace spectral_impl
namespace detail {
namespace {
spectral_impl::Phe2hbPlan upper_cpu_geometry(const DdlaHandle_t& h,const int* desc,int ldr,int ldc,int tile)
{
    spectral_impl::Phe2hbPlan p;p.backend=DdlaBackend::CPU;p.nb=desc[DDLA_NB_];p.lld=desc[DDLA_LLD_];p.ldv=ldr;p.ldc=ldc;p.upper_tile=tile;
    p.nprows=h->nprows_;p.npcols=h->npcols_;p.myprow=h->myprow_;p.mypcol=h->mypcol_;return p;
}
}
template<class T>
void phe2hb_upper_av_cpu(const DdlaHandle_t& h,int extent,const T* A,const int* desc,
    const T* Vr,int ldr,const T* Vc,int ldc,int nvec,T* Wr_partial,T* Wr,T* Wc_partial,T* Wc,
    T* aux,std::size_t aux_elements,int tile)
{
    std::exception_ptr failure;spectral_impl::Phe2hbPlan p;
    try {
        const int nb=desc[DDLA_NB_];
        if(nb<1||desc[DDLA_MB_]!=nb||desc[DDLA_RSRC_]!=0||desc[DDLA_CSRC_]!=0||tile<nb||tile%nb||extent<0||nvec<1||
            static_cast<long long>(ldr)*nvec>INT_MAX||static_cast<long long>(ldc)*nvec>INT_MAX||
            aux_elements<ptranspose_vectors_work_elems(h,extent,nvec,nb))
            throw std::invalid_argument("CPU upper AV metadata/workspace invalid");
        p=upper_cpu_geometry(h,desc,ldr,ldc,tile);
    }catch(...){failure=std::current_exception();}
    detail::agree_failure(h,failure,"CPU upper band reduction");
    spectral_impl::phe2hb_upper_av_run(h,p,extent,A,Vr,Vc,nvec,Wr_partial,Wr,Wc_partial,Wc,aux,aux_elements*sizeof(T));
}
template<class T>
void phe2hb_upper_update_cpu(const DdlaHandle_t& h,int extent,T* A,const int* desc,
    const T* rows,int ldr,const T* cols,int ldc,int inner,int tile)
{
    std::exception_ptr failure;spectral_impl::Phe2hbPlan p;
    try {
        const int nb=desc[DDLA_NB_];
        if(nb<1||desc[DDLA_MB_]!=nb||desc[DDLA_RSRC_]!=0||desc[DDLA_CSRC_]!=0||extent<0||tile<nb||tile%nb||inner<1)
            throw std::invalid_argument("CPU upper update rectangle metadata invalid");
        p=upper_cpu_geometry(h,desc,ldr,ldc,tile);
    }catch(...){failure=std::current_exception();}
    detail::agree_failure(h,failure,"CPU upper band reduction");spectral_impl::phe2hb_upper_update_run(h,p,extent,A,rows,cols,inner);
}

template<class T>
static bool phe2hb_upper_cpu_impl(const DdlaHandle_t& h,int n,T* A,const int* desc,int nbw,
                                  T* tmat,int& info,int requested_tile,bool auto_qr)
{
    if(!h)throw std::invalid_argument("CPU upper band reduction needs a handle");
    std::exception_ptr failure;
    try {
        check_desc(desc,h);
        if(ddlaGetBackend(h)!=DdlaBackend::CPU||n<0||n>desc[DDLA_M_]||n>desc[DDLA_N_]||
           desc[DDLA_MB_]!=desc[DDLA_NB_]||desc[DDLA_RSRC_]!=0||desc[DDLA_CSRC_]!=0||
           nbw<1||nbw>desc[DDLA_NB_]||desc[DDLA_NB_]%nbw||requested_tile<1||
           (n>0&&nbw>0&&(n-1)/nbw>0&&!tmat))
            throw std::invalid_argument("CPU upper band reduction requires source0 and valid metadata");
    }catch(...){failure=std::current_exception();}
    detail::agree_failure(h,failure,"CPU upper band reduction");
    const int sig[9]={n,nbw,desc[DDLA_M_],desc[DDLA_N_],desc[DDLA_MB_],desc[DDLA_NB_],
                      desc[DDLA_RSRC_],desc[DDLA_CSRC_],requested_tile};
    int lo[9]={},hi[9]={};
    MPI_CHECK(MPI_Allreduce(sig,lo,9,MPI_INT,MPI_MIN,ddlaGetCommunicator(h)));
    MPI_CHECK(MPI_Allreduce(sig,hi,9,MPI_INT,MPI_MAX,ddlaGetCommunicator(h)));
    for(int q=0;q<9;++q)if(lo[q]!=hi[q])
        throw std::invalid_argument("CPU upper band reduction metadata differs across ranks");
    const int panel_policy=auto_qr?(phe2hb_cpu_qr_inplace(h,n,desc,nbw)?1:0):-1;
    const int nb=desc[DDLA_NB_],tile=std::max(nb,(requested_tile/nb)*nb);
    const int max_extent=n>0?((n-1)/nbw)*nbw:0;
    if(max_extent<=tile) {
        if(auto_qr)phe2hb_impl<DdlaBackend::CPU,T>(h,n,A,desc,nbw,tmat,info,0,panel_policy);
        else phe2hb<DdlaBackend::CPU,T>(h,n,A,desc,nbw,tmat,info);
        return false;
    }
    failure=nullptr;
    try {
        int pr=0,pc=0,mr=0,mc=0;ddlaGetGridDims(h,pr,pc);ddlaGetGridCoords(h,mr,mc);
        const int lr=std::max(1,num_loc(max_extent,nb,mr,0,pr));
        const int lc=std::max(1,num_loc(max_extent,nb,mc,0,pc));
        if(static_cast<long long>(lr)*nbw>INT_MAX||static_cast<long long>(lc)*nbw>INT_MAX||
           static_cast<long long>(nbw)*nbw>INT_MAX||max_extent>INT_MAX-(nb-1))
            throw std::overflow_error("CPU upper band vector reduction count overflow");
        if(num_loc(n,nb,mr,0,pr)>0&&num_loc(n,nb,mc,0,pc)>0&&!A)
            throw std::invalid_argument("CPU upper band matrix missing");
    }catch(...){failure=std::current_exception();}
    detail::agree_failure(h,failure,"CPU upper band reduction");
    phe2hb_impl<DdlaBackend::CPU,T>(h,n,A,desc,nbw,tmat,info,tile,panel_policy);
    return true;
}

template<class T>
bool phe2hb_upper_cpu(const DdlaHandle_t& h,int n,T* A,const int* desc,int nbw,
                     T* tmat,int& info,int requested_tile)
{
    return phe2hb_upper_cpu_impl(h,n,A,desc,nbw,tmat,info,requested_tile,false);
}

template<class T>
bool phe2hb_cpu_auto_qr(const DdlaHandle_t& h,int n,T* A,const int* desc,int nbw,
                      T* tmat,int& info,int requested_tile)
{
    return phe2hb_upper_cpu_impl(h,n,A,desc,nbw,tmat,info,requested_tile,true);
}
#define DDLA_INST_UPPER(T) \
 template bool phe2hb_cpu_auto_qr<T>(const DdlaHandle_t&,int,T*,const int*,int,T*,int&,int); \
 template bool phe2hb_upper_cpu<T>(const DdlaHandle_t&,int,T*,const int*,int,T*,int&,int); \
 template void phe2hb_upper_av_cpu<T>(const DdlaHandle_t&,int,const T*,const int*,const T*,int,const T*,int,int,T*,T*,T*,T*,T*,std::size_t,int); \
 template void phe2hb_upper_update_cpu<T>(const DdlaHandle_t&,int,T*,const int*,const T*,int,const T*,int,int,int)
DDLA_FOR_EACH_TYPE(DDLA_INST_UPPER)
#undef DDLA_INST_UPPER
} // namespace detail

#define INSTANTIATE_PHE2HB_SIZE(TYPE) \
    template void spectral_impl::phe2hb_workspace_size<TYPE>(const DdlaHandle_t&,const spectral_impl::Phe2hbPlan&, \
        TYPE*,TYPE*,std::size_t*,std::size_t*)
#define INSTANTIATE_PHE2HB(BACKEND, TYPE)                                           \
    template void phe2hb<BACKEND, TYPE>(                                            \
        const DdlaHandle_t&, int, TYPE*, const int*, int, TYPE*, int&); \
    template void spectral_impl::phe2hb_run<BACKEND,TYPE>(const DdlaHandle_t&,const spectral_impl::Phe2hbPlan&, \
        TYPE*,TYPE*,int&,void*,std::size_t,void*,std::size_t)

DDLA_FOR_EACH_TYPE(INSTANTIATE_PHE2HB_SIZE)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_PHE2HB)

#undef INSTANTIATE_PHE2HB
#undef INSTANTIATE_PHE2HB_SIZE

} // namespace ddla
