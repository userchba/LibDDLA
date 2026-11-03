#ifndef DDLA_PHEEVD_ALIAS_H
#define DDLA_PHEEVD_ALIAS_H
#include "helpers.h"
#include <ddla/ddla.h>
#include "ddla_connector.h"
#include "ddla_desc.h"
#include "comm_traits.h"
#include "plan_util.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>

namespace ddla { namespace detail {
// Track the pointer by reference because stage 4 can consume and clear Y.
// Unwinding performs only local stream-ordered cleanup, never a collective.
template<class T>
class SolverBufferGuard {
    const DdlaHandle_t& handle_;
    T*& pointer_;
public:
    SolverBufferGuard(const DdlaHandle_t& handle, T*& pointer) noexcept
        : handle_(handle), pointer_(pointer) {}
    SolverBufferGuard(const SolverBufferGuard&) = delete;
    SolverBufferGuard& operator=(const SolverBufferGuard&) = delete;
    ~SolverBufferGuard() noexcept {
        if (pointer_) (void)ddlaFree(handle_, pointer_);
        pointer_ = nullptr;
    }
    void reset() {
        T* pointer = pointer_;
        pointer_ = nullptr;
        DDLA_CHECK(ddlaFree(handle_, pointer));
    }
};

// Stage 3 may overwrite only storage that stage 4 will replace with active
// eigenvectors. Parent descriptors and padded leading dimensions keep their
// normal allocation path. Compare spans without overflowing pointer ends.
template<class T>
std::size_t output_scratch_elements(const DdlaHandle_t& handle, int n,
                                   const T* A, const int* descA,
                                   T* Z, const int* descZ)
{
    using rt = typename real_type<T>::type;
    if (!Z || descZ[DDLA_M_] != n || descZ[DDLA_N_] != n) return 0;
    int pr=0, pc=0, mr=0, mc=0;
    ddlaGetGridDims(handle,pr,pc);
    ddlaGetGridCoords(handle,mr,mc);
    const int zr=num_loc(n,descZ[DDLA_MB_],mr,descZ[DDLA_RSRC_],pr);
    const int zc=num_loc(n,descZ[DDLA_NB_],mc,descZ[DDLA_CSRC_],pc);
    if (zr<=0 || zc<=0 || descZ[DDLA_LLD_]!=zr) return 0;
    const std::size_t limit=std::numeric_limits<std::size_t>::max()/sizeof(T);
    if (static_cast<std::size_t>(zr)>limit/static_cast<std::size_t>(zc)) return 0;
    const std::size_t zbytes=static_cast<std::size_t>(zr)*zc*sizeof(T);
    const int ac=num_loc(descA[DDLA_N_],descA[DDLA_NB_],mc,descA[DDLA_CSRC_],pc);
    if (ac>0 && static_cast<std::size_t>(descA[DDLA_LLD_])>limit/static_cast<std::size_t>(ac)) return 0;
    const std::size_t abytes=static_cast<std::size_t>(descA[DDLA_LLD_])*ac*sizeof(T);
    if (memory_overlaps(A, abytes, Z, zbytes)) return 0;
    return zbytes/sizeof(rt);
}

// Freeze global overlap in the checker; the local saved storage is then sized
// without touching matrices. Active spans include inter-column padding but exclude
// rows below the final active column, preserving the former overlap rule.
struct PheevdReflectorBackupPlan {
    bool needed=false;int n=0,rows=0,columns=0,source_ld=1;
    int npr=1,npc=1,pr=0,pc=0;char triangle='A';
    std::array<int,DDLA_DLEN_> saved{};
};
template<class T>inline std::size_t reflector_active_span(int rows,int columns,int ld)
{
    if(!rows||!columns)return 0;
    const std::size_t limit=std::numeric_limits<std::size_t>::max()/sizeof(T),gaps=std::size_t(columns-1);
    if(gaps&&std::size_t(ld)>(limit-std::size_t(rows))/gaps)throw std::length_error("pheevd: reflector span overflow");
    return (gaps*std::size_t(ld)+std::size_t(rows))*sizeof(T);
}
template<class F>inline void reflector_backup_collective(const DdlaHandle_t& h,F&& operation)
{
    std::exception_ptr failure;
    try{operation();}catch(...){failure=std::current_exception();}
    int ok=failure?0:1,all_ok=0;MPI_CHECK(MPI_Allreduce(&ok,&all_ok,1,MPI_INT,MPI_MIN,ddlaGetCommunicator(h)));
    if(!all_ok){if(failure)std::rethrow_exception(failure);throw std::runtime_error("pheevd: reflector backup failed on another rank");}
}
template<class T>PheevdReflectorBackupPlan pheevd_reflector_backup_check(const DdlaHandle_t& h,int n,
    const T* A,const int* da,const T* Z,const int* dz,char triangle='A')
{
    PheevdReflectorBackupPlan p;p.n=n;p.triangle=triangle;if(!dz||n<=0)return p;
    if(!h)throw std::invalid_argument("pheevd: reflector backup requires a handle");
    int overlap=0;
    reflector_backup_collective(h,[&]{
        check_desc(da,h);check_desc(dz,h);
        if(n>da[DDLA_M_]||n>da[DDLA_N_]||n>dz[DDLA_M_]||n>dz[DDLA_N_])throw std::invalid_argument("pheevd: reflector backup order exceeds descriptor");
        ddlaGetGridDims(h,p.npr,p.npc);ddlaGetGridCoords(h,p.pr,p.pc);
        p.rows=num_loc(n,da[DDLA_MB_],p.pr,da[DDLA_RSRC_],p.npr);p.columns=num_loc(n,da[DDLA_NB_],p.pc,da[DDLA_CSRC_],p.npc);p.source_ld=da[DDLA_LLD_];
        const auto ab=reflector_active_span<T>(p.rows,p.columns,p.source_ld);
        const auto zb=reflector_active_span<T>(num_loc(n,dz[DDLA_MB_],p.pr,dz[DDLA_RSRC_],p.npr),num_loc(n,dz[DDLA_NB_],p.pc,dz[DDLA_CSRC_],p.npc),dz[DDLA_LLD_]);
        if((ab&&!A)||(zb&&!Z))throw std::invalid_argument("pheevd: null active matrix");
        overlap=memory_overlaps(A,ab,Z,zb);
        std::copy(da,da+DDLA_DLEN_,p.saved.begin());p.saved[DDLA_M_]=p.saved[DDLA_N_]=n;p.saved[DDLA_LLD_]=std::max(p.rows,1);
    });
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,&overlap,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h)));p.needed=overlap!=0;return p;
}
// Backend bytes of pheevd_reflector_backup_save_run: the saved reflectors (saved LLD x
// local columns, at least one element).  Nothing when no backup is needed.
template<class T>void pheevd_reflector_backup_workspace_size(const PheevdReflectorBackupPlan& p,std::size_t* backend_bytes)
{*backend_bytes=p.needed?padded_bytes<T>(std::max(std::size_t(1),std::size_t(p.saved[DDLA_LLD_])*std::size_t(p.columns))):0;}
// Copies the active reflectors of A to the front of backend_work (ld p.saved[DDLA_LLD_]);
// they stay there for the caller. Collective.
template<class T>void pheevd_reflector_backup_save_run(const DdlaHandle_t& h,const PheevdReflectorBackupPlan& p,const T* A,
    void* backend_work,std::size_t backend_bytes)
{
    try {
        if(!p.needed)return;
        char* w=static_cast<char*>(backend_work);
        T* const saved=take<T>(w,std::max(std::size_t(1),std::size_t(p.saved[DDLA_LLD_])*std::size_t(p.columns)));
        assert(static_cast<std::size_t>(w-static_cast<char*>(backend_work))<=backend_bytes);(void)backend_bytes;
        reflector_backup_collective(h,[&]{
            const int ld=p.saved[DDLA_LLD_];
            if(p.triangle=='U'||p.triangle=='L')for(int c=0;c<p.columns;++c){
                const int global=indxl2g(c,p.saved[DDLA_NB_],p.pc,p.saved[DDLA_CSRC_],p.npc);
                const int first=p.triangle=='L'?num_loc(global+1,p.saved[DDLA_MB_],p.pr,p.saved[DDLA_RSRC_],p.npr):0;
                const int end=p.triangle=='U'?num_loc(global,p.saved[DDLA_MB_],p.pr,p.saved[DDLA_RSRC_],p.npr):p.rows;
                if(end>first)DDLA_CHECK(ddlaMemcpy(h,saved+first+std::size_t(c)*ld,A+first+std::size_t(c)*p.source_ld,std::size_t(end-first)*sizeof(T)));
            }else if(p.rows&&p.columns)DDLA_CHECK(ddlaMemcpy2D(h,saved,std::size_t(ld)*sizeof(T),A,std::size_t(p.source_ld)*sizeof(T),std::size_t(p.rows)*sizeof(T),p.columns));
            DDLA_CHECK(ddlaSynchronize(h));
        });
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(h) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(h); } catch (...) {}
        throw;
    }
}

// Owning convenience adapter. Composite drivers keep the POD plan, carve the
// saved reflectors from their own scratch and call save_run.
template<class T>class PheevdReflectorBackup {
    DdlaHandle_t h_;const T* original_;const int* original_desc_;
    PheevdReflectorBackupPlan plan_{};T* saved_=nullptr;
public:
    PheevdReflectorBackup(const DdlaHandle_t& h,int n,const T* A,const int* da,const T* Z,const int* dz,char triangle='A')
        :h_(h),original_(A),original_desc_(da)
    {
        plan_=pheevd_reflector_backup_check<T>(h,n,A,da,Z,dz,triangle);
        if(!plan_.needed)return;
        std::size_t bytes=0;pheevd_reflector_backup_workspace_size<T>(plan_,&bytes);
        try{
            reflector_backup_collective(h,[&]{DDLA_CHECK(ddlaMalloc(h,reinterpret_cast<void**>(&saved_),bytes));});
            pheevd_reflector_backup_save_run<T>(h,plan_,A,saved_,bytes);
        }catch(...){if(saved_)(void)ddlaFree(h_,saved_);saved_=nullptr;throw;}
    }
    PheevdReflectorBackup(const PheevdReflectorBackup&)=delete;PheevdReflectorBackup& operator=(const PheevdReflectorBackup&)=delete;
    ~PheevdReflectorBackup(){if(saved_)(void)ddlaFree(h_,saved_);}
    const T* data()const noexcept{return plan_.needed?saved_:original_;}
    const int* descriptor()const noexcept{return plan_.needed?plan_.saved.data():original_desc_;}
    bool active()const noexcept{return plan_.needed;}
};
} }
#endif
