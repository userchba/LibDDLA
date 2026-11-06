#ifndef DDLA_SVD_WORKSPACE_H
#define DDLA_SVD_WORKSPACE_H

// The components (plan_util.h) of pgebrd and punmbr, which pgesvd composes:
//   <r>_check          -- validation, agreed over the grid; returns the rank-local plan,
//   <r>_workspace_size -- the plan's scratch in bytes; no allocation, no communication,
//   <r>_run            -- the algorithm on the caller's scratch; no validation, no allocation.

#include <ddla/ddla.h>
#include <cstddef>

namespace ddla { namespace spectral_impl {

// Bound replicated panel/Gram storage independently of global matrix size.
inline constexpr int kSvdPanelWidth = 256;

// Checked rank-local geometry; none of these records owns storage.  nbp is
// pgebrd's panel width; ldv / ldy are the padded leading dimensions of its
// panel buffers VX / YU: every panel column holds its mloc / nloc vector rows
// and two padding rows ([ld] = tau, [ld+1] = beta), so ONE commBcast of
// (mloc / nloc) + 2 entries carries vector + scalars.
struct PgebrdPlan {
    DdlaBackend backend{};
    int m=0,n=0,k=0,mb=1,nb=1,rsrc=0,csrc=0,lld=1;
    int nprows=1,npcols=1,myprow=0,mypcol=0,mloc=0,nloc=0;
    int nbp=1,ldv=1,ldy=1;
};
PgebrdPlan pgebrd_geometry(const DdlaHandle_t&,DdlaBackend,int,int,const int*);
PgebrdPlan pgebrd_check(const DdlaHandle_t&,DdlaBackend,int,int,const void*,const int*,
                        const void*,const void*,const void*,const void*);

// Backend bytes of pgebrd_run on this rank (none for k == 0), in carving order: the
// panel buffers VX (ldv x 2 nbp) and YU (ldy x 2 nbp) and the all-reduce send and
// receive buffers; then on a CPU handle the row work vector (ldy) and the ?LARFG
// norm receive buffer (4 max(nprows, npcols) reals); reflector scalars use the
// panel padding described above. On a GPU handle sigma, the amax slots (nprows npcols), the amax
// partials and the device mirrors of d (k), e (max(1, k - 1)), tauq (k) and taup (k).
template<class T>
void pgebrd_workspace_size(const DdlaHandle_t&,const PgebrdPlan&,std::size_t* backend_bytes);
template<DdlaBackend B,class T>
void pgebrd_run(const DdlaHandle_t&,const PgebrdPlan&,T* A,const int* descA,
    typename real_type<T>::type* d,typename real_type<T>::type* e,T* tauq,T* taup,int& info,
    void* backend_work,std::size_t backend_bytes);

struct PunmbrPlan {
    DdlaBackend backend{};
    char vect='Q',side='L',trans='N';
    int m=0,n=0,k=0,nq=0,nref=0,off=0,nbp=1;
    int mbA=1,nbA=1,rsrcA=0,csrcA=0,lldA=1,lldC=1;
    int nprows=1,npcols=1,myprow=0,mypcol=0,mlocC=0,nlocC=0;
};
PunmbrPlan punmbr_geometry(const DdlaHandle_t&,DdlaBackend,char,char,char,int,int,int,const int*,const int*);
PunmbrPlan punmbr_check(const DdlaHandle_t&,DdlaBackend,bool,char,char,char,
    int,int,int,const void*,const int*,const void*,const void*,const int*);

// The scratch of punmbr_run on this rank (none for an empty C or no reflector), in
// carving order. Backend: the panel Gram matrix G and T (nbp x nbp each),
// plus a separate reduced Gram matrix only for a multi-rank Gram scope;
// on a GPU handle the device copy of tau (nref), then the reflector
// panel (local rows of C x nbp for vect 'Q', local columns for 'P') and the product
// with its reduction (nbp x local columns / rows of C each). No host staging:
// CPU builds T in backend host storage; bounded GPU panels build T on device.
template<class T>
void punmbr_workspace_size(const DdlaHandle_t&,const PunmbrPlan&,std::size_t* backend_bytes,std::size_t* host_bytes);
template<DdlaBackend B,class T>
void punmbr_run(const DdlaHandle_t&,const PunmbrPlan&,const T* A,const int* descA,const T* tau,
    T* C,const int* descC,int& info,
    void* backend_work,std::size_t backend_bytes,void* host_work,std::size_t host_bytes);

} }
#endif
