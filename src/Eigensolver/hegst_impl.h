#ifndef DDLA_PRIVATE_HEGST_IMPL_H
#define DDLA_PRIVATE_HEGST_IMPL_H

#include <ddla/ddla.h>
#include "Eigensolver/eig_workspace.h"
#include <array>
#include <cstddef>

namespace ddla { namespace hegst_impl {
// Prepared handle identity and checked matrix spans. Geometry already belongs
// to the completion plan; do not duplicate its order/block/grid coordinates.
struct Plan {
    DdlaCommunication transport{};
    MPI_Comm comm=MPI_COMM_NULL;
    void* compute=nullptr;
    void* data=nullptr;
    int a[DDLA_DLEN_]{},factor[DDLA_DLEN_]{};
    spectral_impl::HermitianPlan completion{};
};
struct Workspace {
    std::size_t backend_bytes=0,host_bytes=0;
    // Payload requirements, separate from retained vector capacity/arenas.
    std::array<std::size_t,2> staging_bytes{};
};
template<class T>
Plan check(const DdlaHandle_t&,DdlaBackend,char,int,T*,const int*,const T*,const int*);
template<class T> Workspace workspace_size(const DdlaHandle_t&,const Plan&);

// Shared type-1 dense congruence, also used directly by PHEGVD. The solve-only
// query reads no completion/eigen plan and is safe before factoring B. Full
// sizing/staging compose initial completion and both solves with domain maxima;
// a final completion needs exactly the same scratch and payload geometry.
// Checked source-zero n<=descriptor-block inputs live on owner (0,0): local
// completion/TRSM uses zero backend/host arena and staging on every rank.
template<class T>
void congruence_solve_workspace_size(const DdlaHandle_t&,char uplo,int n,
    const int* a,const int* factor,std::size_t*,std::size_t*);
template<class T>
Workspace congruence_workspace_size(const DdlaHandle_t&,const spectral_impl::HermitianPlan&,
    const int* a,const int* factor);
template<class T>
std::array<std::size_t,2> congruence_staging_bytes(const DdlaHandle_t&,
    const spectral_impl::HermitianPlan&,const int* a,const int* factor);
// No validation, ownership, query, allocation or extra success retirement.
// B is read-only. Caller has checked plans and reserved staging beforehand.
template<DdlaBackend Backend,class T>
void congruence_run(const DdlaHandle_t&,const spectral_impl::HermitianPlan&,T* a,const int*,
    const T* factor,const int*,void*,std::size_t,void*,std::size_t);

// PHEGST adds prepared-handle guards, final canonical completion and retirement
// before borrowed scratch may be released/reused. Full leading A is written;
// only input UPLO is read. Factor/parent padding remain byte-for-byte const.
// Reserve workspace_size().staging_bytes under owning agreement before run.
template<DdlaBackend Backend,class T>
void run(const DdlaHandle_t&,const Plan&,T*,const T*,void*,std::size_t,void*,std::size_t);
} }
#endif
