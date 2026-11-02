#ifndef DDLA_EIG_WORKSPACE_H
#define DDLA_EIG_WORKSPACE_H

#include <ddla/ddla.h>
#include <cstddef>

namespace ddla { namespace spectral_impl {
struct UnmtrHe2hbPlan {
    DdlaBackend backend{};
    char trans='N';
    int n=0,nbw=1,nb=1,lldA=1,lldZ=1,nsteps=0;
    int nprows=1,npcols=1,myprow=0,mypcol=0,zcols=0,ldv=1;
};
UnmtrHe2hbPlan punmtr_he2hb_geometry(const DdlaHandle_t&,DdlaBackend,char,int,int,const int*,const int*);
UnmtrHe2hbPlan punmtr_he2hb_check(const DdlaHandle_t&,DdlaBackend,char,int,int,const void*,const int*,const void*,const void*,const int*);

// Backend bytes of punmtr_he2hb_run: the gathered reflectors V (ldv x nbw),
// the product W = V^H Z (nbw x max(zcols, 1)) and its all-reduce W2 (same
// size).  No panel (nsteps == 0) needs nothing.
template<class T>
void punmtr_he2hb_workspace_size(const DdlaHandle_t&,const UnmtrHe2hbPlan& p,std::size_t* backend_bytes);
template<DdlaBackend B,class T>
void punmtr_he2hb_run(const DdlaHandle_t&,const UnmtrHe2hbPlan&,const T*,const int*,const T*,T*,const int*,int&,
    void* backend_work,std::size_t backend_bytes);

struct HermitianBlock {
    long long src_off=0,dst_off=0;
    int src_ld=1,dst_ld=1,rows=0,cols=0;
};
struct HermitianPlan {
    DdlaBackend backend{};bool upper=true;
    int n=0,nb=1,lld=1,pr=1,pc=1,mr=0,mc=0,threads=1;
    char major='R';
};
HermitianPlan pherm_complete_check(const DdlaHandle_t&,DdlaBackend,char,int,const void*,const int*);
// This rank's block counts and pack lengths (rank-local integer math).
void pherm_complete_geometry(const HermitianPlan&,std::size_t& pack,std::size_t& local,
    std::size_t& unpack,std::size_t& diag,std::size_t& send_total,std::size_t& recv_total);

// Scratch of pherm_complete_run.  Host bytes: the block descriptors
// (pack | local | unpack | diag, one HermitianBlock each) and six per-peer int
// arrays (counts, displacements, cursors).  Backend bytes: on a GPU handle the
// device copy of the descriptors, then the send and receive packs
// (send_total / recv_total entries) unless the caller's scratch holds both:
// scratch_elements >= send_total + recv_total on a GPU handle (pass 0 for no
// scratch; the run decides the same way from the same count).  n < 1 needs
// nothing.  The geometry pass may throw std::overflow_error.
template<class T>
void pherm_complete_workspace_size(const DdlaHandle_t&,const HermitianPlan& p,std::size_t scratch_elements,
    std::size_t* backend_bytes,std::size_t* host_bytes);
// The completion of A in place.  scratch / scratch_elements are the caller's
// optional T storage (disjoint from A; nullptr with 0 for none), the same
// count the size function was given.  No validation, no allocation.
template<DdlaBackend B,class T>
void pherm_complete_run(const DdlaHandle_t&,const HermitianPlan&,T* A,T* scratch,std::size_t scratch_elements,
    void* backend_work,std::size_t backend_bytes,void* host_work,std::size_t host_bytes);

} }
#endif
