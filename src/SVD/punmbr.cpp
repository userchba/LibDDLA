// ---------------------------------------------------------------------------
// Apply the orthogonal factors of pgebrd (ScaLAPACK PxUNMBR / PxORMBR
// semantics, LAPACK ?UNMBR conventions):
//   vect = 'Q', side = 'L':  C := op(Q) C,  Q = H(0)...H(nref-1) (columns of A)
//   vect = 'P', side = 'R':  C := C op(P),  P = G(0)...G(nref-1) (rows of A)
// These are the two combinations an SVD back-transform needs; the transposed
// layouts (Q from the right, P from the left) would need V redistributed to
// the other grid dimension and are rejected.
//
// Blocked ?LARFB, one panel of nbp reflectors at a time:
//   left :  W = V^H C (Col all-reduce),  C -= V op(T) W
//   right:  Z = C W (Row all-reduce),    C -= Z op(T) W^H
// Each panel has at most min(block size, nref, 256) reflectors and is clipped
// at descriptor-block boundaries. It lies on ONE process column (row),
// which packs the distributed vector pieces in a
// single call, restores the unit-trapezoid structure with one kernel and
// broadcasts the active suffix along its process row (column).  Entries before
// the first reflector anchor are zero and do not enter the Gram matrix or
// the update. T comes from V^H V (an all-reduce when needed) and the existing QR
// recurrence directly in backend storage. Bounded GPU panels stay on device.
// C must share A's distribution along the applied dimension: its rows
// for 'Q' (mb, rsrc), its columns for 'P' (nb, csrc).
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>

#include "ddla_connector.h"
#include "check_agree.h"
#include "BLAS/blas_impl.h"
#include "comm_traits.h"
#include "ddla_stream.h"
#include "ddla_desc.h"
#include "QR/pgeqrf_kernels.h"
#include "plan_util.h"
#include "reflector_build_T.h"
#include "punmtr_impl.h"
#include "svd_workspace.h"
#include "instantiate.h"
#include "require_backend.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ddla {

namespace spectral_impl {
PunmbrPlan punmbr_geometry(const DdlaHandle_t& h,DdlaBackend backend,char vect,char side,char trans,
    int m,int n,int k,const int* descA,const int* descC)
{
    PunmbrPlan p;p.backend=backend;p.vect=vect;p.side=side;p.trans=trans;p.m=m;p.n=n;p.k=k;
    p.nq=vect=='Q'?m:n;
    if(vect=='Q'){p.nref=p.nq>=k?k:p.nq-1;p.off=p.nq>=k?0:1;}
    else{p.nref=k<p.nq?k:p.nq-1;p.off=k<p.nq?0:1;}
    if(!m||!n||p.nref<=0)return p;
    p.mbA=descA[DDLA_MB_];p.nbA=descA[DDLA_NB_];p.rsrcA=descA[DDLA_RSRC_];p.csrcA=descA[DDLA_CSRC_];
    p.lldA=descA[DDLA_LLD_];p.lldC=descC[DDLA_LLD_];
    ddlaGetGridDims(h,p.nprows,p.npcols);ddlaGetGridCoords(h,p.myprow,p.mypcol);
    p.mlocC=num_loc(m,descC[DDLA_MB_],p.myprow,descC[DDLA_RSRC_],p.nprows);
    p.nlocC=num_loc(n,descC[DDLA_NB_],p.mypcol,descC[DDLA_CSRC_],p.npcols);
    p.nbp=std::max(1,std::min({vect=='Q'?p.nbA:p.mbA,p.nref,kSvdPanelWidth}));return p;
}
PunmbrPlan punmbr_check(const DdlaHandle_t& handle,DdlaBackend backend,bool is_real,
    char vect,char side,char trans,int m,int n,int k,const void* A,const int* descA,
    const void* tau,const void* C,const int* descC)
{
    const auto p=detail::agree_check(handle,"punmbr",[&](detail::Agree& agreement)->PunmbrPlan {
        agreement.mode("spectral operation",3102);
    detail::require_backend(handle,backend,"punmbr");
    const char v=static_cast<char>(std::toupper(static_cast<unsigned char>(vect)));
    const char s=static_cast<char>(std::toupper(static_cast<unsigned char>(side)));
    const char t=static_cast<char>(std::toupper(static_cast<unsigned char>(trans)));
    if(v!='Q'&&v!='P')throw std::runtime_error("punmbr: vect must be 'Q' or 'P'");
    if((v=='Q'&&s!='L')||(v=='P'&&s!='R'))
        throw std::runtime_error("punmbr: only vect='Q' with side='L' and vect='P' with side='R' are implemented");
    if(t!='N'&&t!='C'&&!(is_real&&t=='T'))
        throw std::runtime_error("punmbr: trans must be 'N' or 'C' ('T' for real types)");
    if(m<0||n<0||k<0)throw std::runtime_error("punmbr: m, n, k must be non-negative");
    check_desc(descA,handle);check_desc(descC,handle);
    if(m>descC[DDLA_M_]||n>descC[DDLA_N_])
        throw std::runtime_error("punmbr: m/n outside C's descriptor dimensions");
    const auto p=punmbr_geometry(handle,backend,v,s,t,m,n,k,descA,descC);
    if(v=='Q') {
        if(p.nq>descA[DDLA_M_]||std::min(p.nq,k)>descA[DDLA_N_])
            throw std::runtime_error("punmbr: A's descriptor is smaller than nq x min(nq, k)");
        if(descC[DDLA_MB_]!=descA[DDLA_MB_]||descC[DDLA_RSRC_]!=descA[DDLA_RSRC_])
            throw std::runtime_error("punmbr: vect='Q' needs C's rows distributed like A's (equal MB and RSRC)");
    } else {
        if(std::min(p.nq,k)>descA[DDLA_M_]||p.nq>descA[DDLA_N_])
            throw std::runtime_error("punmbr: A's descriptor is smaller than min(nq, k) x nq");
        if(descC[DDLA_NB_]!=descA[DDLA_NB_]||descC[DDLA_CSRC_]!=descA[DDLA_CSRC_])
            throw std::runtime_error("punmbr: vect='P' needs C's columns distributed like A's (equal NB and CSRC)");
    }
    if(!m||!n||p.nref<=0)return p;
    if(!tau)throw std::runtime_error("punmbr: null tau array");
    const int ar=num_loc(v=='Q'?p.nq:std::min(p.nq,k),p.mbA,p.myprow,p.rsrcA,p.nprows);
    const int ac=num_loc(v=='Q'?std::min(p.nq,k):p.nq,p.nbA,p.mypcol,p.csrcA,p.npcols);
    if(ar>0&&ac>0&&!A)throw std::invalid_argument("punmbr: null active reflector input");
    if(p.mlocC>0&&p.nlocC>0&&!C)throw std::invalid_argument("punmbr: null active output");
    return p;
    });
    int signature[25]={int(backend),is_real?1:0,int(p.vect),int(p.side),int(p.trans),m,n,k,
        handle->nprows_,handle->npcols_,int(handle->major)};
    detail::descriptor_signature(signature+11,descA);detail::descriptor_signature(signature+18,descC);
    detail::agree_signature(handle,"punmbr",signature);
    return p;
}

// Component 2 (svd_workspace.h).
template <typename T>
void punmbr_workspace_size(const DdlaHandle_t&, const PunmbrPlan& p, std::size_t* backend_bytes, std::size_t* host_bytes)
{
    *backend_bytes = 0;
    *host_bytes = 0;
    if (!p.m || !p.n || p.nref <= 0) return;
    const std::size_t nbp2 = static_cast<std::size_t>(p.nbp) * p.nbp;
    const bool left = p.vect == 'Q';
    const std::size_t panel = static_cast<std::size_t>(std::max(1, left ? p.mlocC : p.nlocC)) * p.nbp;
    const std::size_t product = static_cast<std::size_t>(p.nbp) * std::max(1, left ? p.nlocC : p.mlocC);
    const bool reduce_gram = (left ? p.nprows : p.npcols) > 1;
    *backend_bytes = (2 + int(reduce_gram)) * detail::padded_bytes<T>(nbp2)
                   + (p.backend == DdlaBackend::GPU ? detail::padded_bytes<T>(p.nref) : 0)
                   + detail::padded_bytes<T>(panel) + 2 * detail::padded_bytes<T>(product);
}

// Component 3.
template <DdlaBackend Backend, typename T>
void punmbr_run(const DdlaHandle_t& handle,const PunmbrPlan& p,const T* A,const int*,const T* tau,T* C,const int*,
    int& info,void* backend_work,[[maybe_unused]] std::size_t backend_bytes,
    [[maybe_unused]] void* host_work,[[maybe_unused]] std::size_t host_bytes)
{
    try {
        using rt=typename real_type<T>::type;
        static_assert(Backend!=DdlaBackend::GPU||DDLA_HAS_GPU,"GPU punmbr is not available");
        info=0;if(!p.m||!p.n||p.nref<=0)return;
        const char t=p.trans;
        const bool left=p.vect=='Q';
        const bool reduce_gram=(left?p.nprows:p.npcols)>1;
        [[maybe_unused]] const int nq=p.nq;const int nref=p.nref,off=p.off,nbp=p.nbp;
        const int mbA=p.mbA,nbA=p.nbA,rsrcA=p.rsrcA,csrcA=p.csrcA,lldA=p.lldA,lldC=p.lldC;
        const int nprows=p.nprows,npcols=p.npcols,myprow=p.myprow,mypcol=p.mypcol,mlocC=p.mlocC,nlocC=p.nlocC;
        const bool notran=t=='N';
        const std::size_t nbp2 = static_cast<std::size_t>(nbp) * nbp;
        // Panel order: Q C = Q_0 (Q_1 (... C)) and C P^H = (C P_L^H) ... P_0^H
        // apply the last panel first; Q^H C and C P apply the first panel first.
        const bool backward = left ? notran : !notran;
        // A panel must stay within one descriptor block, even when a large
        // descriptor block is split into several bounded panels.
        const int panel_block=left?nbA:mbA;
        const int panels_per_block=panel_block/nbp+(panel_block%nbp!=0);
        const int tail=nref%panel_block;
        const int panel_count=(nref/panel_block)*panels_per_block+tail/nbp+(tail%nbp!=0);

        char* const base = static_cast<char*>(backend_work);
        char* w = base;
        T* const G = detail::take<T>(w, nbp2);
        T* const Gr = reduce_gram ? detail::take<T>(w, nbp2) : G;
        T* const T_panel = detail::take<T>(w, nbp2);
        [[maybe_unused]] T* d_tau = nullptr;
        if constexpr (Backend == DdlaBackend::GPU) {
            d_tau = detail::take<T>(w, static_cast<std::size_t>(nref));
            DDLA_CHECK(ddlaMemcpy(handle, d_tau, tau, static_cast<std::size_t>(nref) * sizeof(T)));
        }

        // Packed active rows begin at local index first.  The global mapping must
        // retain that offset when restoring each reflector's unit entry.
        auto unit_trapezoid = [&](T* V, int ld, int rows, int jb, int u0, int first,
                                  int blk, int me, int src, int np) {
            if constexpr (Backend == DdlaBackend::GPU) {
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
                detail::qr_build_v<T>(handle, true, 0, V, ld, V, ld, rows, jb,
                                       u0, nq - u0, first, blk, me, src, np);
#endif
            } else {
                for (int jj = 0; jj < jb; ++jj)
                    for (int i = 0; i < rows; ++i) {
                        const int g = indxl2g(first + i, blk, me, src, np);
                        const int u = u0 + jj;
                        if (g < u) V[i + static_cast<std::size_t>(jj) * ld] = T(0);
                        else if (g == u) V[i + static_cast<std::size_t>(jj) * ld] = T(1);
                    }
            }
        };

        // Both backends build T directly from the reduced Gram matrix.
        // Bounded panels fit the GPU recurrence without host staging.
        auto panel_T = [&](const T* R, int rows, int ld, int jb, int j0, CommScope gram_scope) {
            DDLA_CHECK(ddlaMemset(handle, G, 0, static_cast<std::size_t>(jb) * jb * sizeof(T)));
            if (rows > 0)
                blas_impl::herk_run<Backend, T>(handle, 'U', 'C', jb, rows, rt(1), R, ld, rt(0), G, jb);
            if (reduce_gram)
                commAllReduce<Backend>(handle, gram_scope, G, Gr, jb * jb, cclSum);
            detail::reflector_build_T<Backend, T>(handle, jb, Gr,
                Backend == DdlaBackend::CPU ? tau + j0 : d_tau + j0, T_panel, jb, false);
        };

        const int local_length = left ? mlocC : nlocC;
        const int other_length = left ? nlocC : mlocC;
        T* const V = detail::take<T>(w, static_cast<std::size_t>(std::max(1, local_length)) * nbp);
        T* const W = detail::take<T>(w, static_cast<std::size_t>(nbp) * std::max(1, other_length));
        T* const Wr = detail::take<T>(w, static_cast<std::size_t>(nbp) * std::max(1, other_length));
        assert(static_cast<std::size_t>(w - base) <= backend_bytes);

        const int block = left ? mbA : nbA;
        const int me = left ? myprow : mypcol;
        const int source = left ? rsrcA : csrcA;
        const int ranks = left ? nprows : npcols;
        const CommScope broadcast_scope = left ? CommScope::Row : CommScope::Col;
        const CommScope gram_scope = left ? CommScope::Col : CommScope::Row;
        for (int panel = 0; panel < panel_count; ++panel) {
            const int index = backward ? panel_count - 1 - panel : panel;
            const int offset = (index % panels_per_block) * nbp;
            const int j0 = (index / panels_per_block) * panel_block + offset;
            const int jb = std::min({nbp, nref - j0, panel_block - offset});
            const int first = num_loc(j0 + off, block, me, source, ranks);
            const int active = local_length - first;
            const int ldv = std::max(1, active);
            const int owner = left ? indxg2p(j0, nbA, csrcA, npcols)
                                   : indxg2p(j0, mbA, rsrcA, nprows);
            if (active > 0) {
                if ((left ? mypcol : myprow) == owner) {
                    if (left) {
                        // Q's reflector columns already have column-major layout.
                        DDLA_CHECK(ddlaMemcpy2D(handle, V, static_cast<std::size_t>(ldv) * sizeof(T),
                            A + first + static_cast<std::size_t>(indxg2l(j0, nbA, npcols)) * lldA,
                            static_cast<std::size_t>(lldA) * sizeof(T),
                            static_cast<std::size_t>(active) * sizeof(T), jb));
                    } else {
                        // A stores conjugated P vectors in rows; unpack as columns.
                        blas_impl::geam_run<Backend, T>(handle, 'C', 'N', active, jb, T(1),
                            A + indxg2l(j0, mbA, nprows) + static_cast<std::size_t>(first) * lldA,
                            lldA, T(0), nullptr, 1, V, ldv);
                    }
                    unit_trapezoid(V, ldv, active, jb, j0 + off, first, block, me, source, ranks);
                }
                if ((left ? npcols : nprows) > 1)
                    commBcast<Backend>(handle, broadcast_scope, V, static_cast<std::size_t>(active) * jb, owner);
            }
            panel_T(V, active, ldv, jb, j0, gram_scope);
            T* active_C = C;
            const int rows = left ? active : mlocC;
            const int cols = left ? nlocC : active;
            if (rows > 0 && cols > 0)
                active_C += left ? first : static_cast<std::size_t>(first) * lldC;
            detail::punmtr_run<Backend, T>(handle, left ? 'L' : 'R', 'U', t,
                rows, cols, jb, ldv, jb, lldC, left ? jb : std::max(1, mlocC),
                reduce_gram, V, T_panel, active_C, W, Wr);
        }
        DDLA_CHECK(ddlaSynchronize(handle));
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(handle) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(handle); } catch (...) {}
        throw;
    }
}

} // namespace spectral_impl

template<DdlaBackend Backend,class T>
void punmbr(const DdlaHandle_t& handle,char vect,char side,char trans,int m,int n,int k,
    const T* A,const int* descA,const T* tau,T* C,const int* descC,int& info)
{
    info=0;
    const auto p=spectral_impl::punmbr_check(handle,Backend,std::is_same<T,typename real_type<T>::type>::value,
                                           vect,side,trans,m,n,k,A,descA,tau,C,descC);
    std::size_t backend_bytes=0,host_bytes=0;
    spectral_impl::punmbr_workspace_size<T>(handle,p,&backend_bytes,&host_bytes);
    vector<char> backend_work(handle,backend_bytes);
    std::vector<char> host_work(host_bytes);
    spectral_impl::punmbr_run<Backend,T>(handle,p,A,descA,tau,C,descC,info,
                                         backend_work.data(),backend_bytes,host_work.data(),host_bytes);
}

#define INSTANTIATE_PUNMBR(BACKEND, TYPE)                                          \
    template void punmbr<BACKEND, TYPE>(                                           \
        const DdlaHandle_t&, char, char, char,                \
        int, int, int, const TYPE*, const int*, const TYPE*,  \
        TYPE*, const int*, int&); \
    template void spectral_impl::punmbr_run<BACKEND,TYPE>(const DdlaHandle_t&, \
        const spectral_impl::PunmbrPlan&,const TYPE*,const int*,const TYPE*,TYPE*,const int*,int&, \
        void*,std::size_t,void*,std::size_t)
#define INSTANTIATE_PUNMBR_SIZE(TYPE) \
    template void spectral_impl::punmbr_workspace_size<TYPE>(const DdlaHandle_t&, \
        const spectral_impl::PunmbrPlan&,std::size_t*,std::size_t*)

DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_PUNMBR)
DDLA_FOR_EACH_TYPE(INSTANTIATE_PUNMBR_SIZE)

#undef INSTANTIATE_PUNMBR_SIZE
#undef INSTANTIATE_PUNMBR

} // namespace ddla
