#ifndef DDLA_PGESVD_WORKSPACE_H
#define DDLA_PGESVD_WORKSPACE_H
#include "svd_workspace.h"
#include "pbdsdc_impl.h"
#include <array>
#include <cstddef>

namespace ddla { namespace spectral_impl {
enum class PgesvdRoute { Direct, TallQr, WideQr };
struct PgesvdPlan {
    PgesvdRoute route=PgesvdRoute::Direct;
    // DDLA_SVD_SHAPE is read only by check: absent/empty/direct means Direct;
    // explicit qr permits the existing memory-eligible aspect>=4 routes.
    // Frozen QR geometry/query counts; size/run never read that environment
    // or query a vendor. Single-rank CPU keeps its native direct route.
    bool qr_inplace=false;
    std::size_t qr_backend_bytes=0,qr_host_bytes=0;
    std::array<int,DDLA_DLEN_> descF{},descR{};
    DdlaBackend backend{};char jobu='N',jobvt='N',uplo='U';
    int m=0,n=0,k=0,ranks=1,rank=0,row_ld=1;
    std::array<int,DDLA_DLEN_> descA{},descU{},descVT{};
    PgebrdPlan brd{};PunmbrPlan unm_u{},unm_v{};PbdsdcPlan dc{};
};
template<class T>PgesvdPlan pgesvd_check(const DdlaHandle_t&,DdlaBackend,char,char,int,int,
    const T*,const int*,const typename real_type<T>::type*,const T*,const int*,const T*,const int*);

// Threads of the host team for distributed D&C (dc.n > 0 for vectors);
// 0 means the values-only run takes no team.
inline int pgesvd_team_threads(const PgesvdPlan& p){return p.dc.n?p.dc.threads:0;}

// Shape routes use local QR tau, plus complete host tau only for the large
// back-transform, followed by max-phase host work. Tall backend work is
// max(QR, R+square-child, apply). Wide work overlays F+QR/adjoint, F+R+copyback,
// R+requested-square-factors+max(child,output-adjoint), and the final apply.
// Every internal descriptor is block-cyclic; conservative memory eligibility
// keeps small/imbalanced layouts on Direct. QR vendor counts are frozen in check.
//
// Direct scratch of pgesvd_run on this rank (none for k == 0), in carving order.
// Host: d (k reals), e (max(1, k - 1) reals), tauq and taup (k each), live through
// every stage; the stages share the rest. Host stage scratch: pbdsdc',
// scatter_rows' or punmbr's scratch for vectors, dqds work otherwise.
// Backend: pgebrd's or punmbr's scratch, or the row slabs U_s / V_s
// (row_ld x k reals each) followed by pbdsdc' or scatter_rows' scratch.
template<class T>
void pgesvd_workspace_size(const DdlaHandle_t&,const PgesvdPlan&,std::size_t* backend_bytes,std::size_t* host_bytes);
// Pure slot maxima, exact for fixed schedules; PBDSDC contributes its reachable
// data-dependent supremum. Owning preparation must reserve before raw writes.
template<class T>std::array<std::size_t,2> pgesvd_staging_bytes(const DdlaHandle_t&,const PgesvdPlan&);
template<DdlaBackend B,class T>
void pgesvd_run(const DdlaHandle_t&,const PgesvdPlan&,detail::DcHostTeam*,T* A,
    typename real_type<T>::type* s,T* U,T* VT,int& info,
    void* backend_work,std::size_t backend_bytes,void* host_work,std::size_t host_bytes);
} }
#endif
