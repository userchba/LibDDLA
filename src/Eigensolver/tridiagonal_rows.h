#ifndef DDLA_PRIVATE_TRIDIAGONAL_ROWS_H
#define DDLA_PRIVATE_TRIDIAGONAL_ROWS_H

// Experimental composition boundary for One/Two-stage row-vector solves.
// Public pstedc remains the full-spectrum D&C operation.
#include "Eigensolver/pstedc_impl.h"
#include "Eigensolver/selected_tridiagonal_core.h"
#include "plan_util.h"
#include "check_agree.h"
#include "host_threads.h"
#include <ddla/ddla_handle_t.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ddla { namespace spectral_impl {
// A nonzero returned plan selects the compact algorithm. No pointers exist yet:
// the enclosing checker has already validated their eventual input/output
// geometry and disjoint borrowed arena. Policy is read/agreed only here.
template<class R>
selected::Plan selected_rows_policy(const DdlaHandle_t& h,char job,int n,int nev,int ldy)
{
    const int requested=detail::agree_check(h,"pheevd tridiagonal policy",[&](detail::Agree&) {
        const char* env=std::getenv("DDLA_EIG_TRIDIAGONAL");
        if (!env || !*env || !std::strcmp(env,"dc")) return 0;
        if (!std::strcmp(env,"selected")) return 1;
        throw std::invalid_argument("pheevd: DDLA_EIG_TRIDIAGONAL must be dc or selected");
    });
    const int signature[]={requested};
    detail::agree_signature(h,"pheevd tridiagonal policy",signature);
    selected::Plan p;
    if (requested && job=='V' && nev>0 && nev<n)
        detail::agree_local(h,"pheevd selected geometry",[&] {
            p=selected::geometry<R>(h->comm,n,nev,h->myid,h->nprocs,ldy);
        });
    return p;
}

template<class R>
void tridiagonal_rows_workspace_size(const DdlaHandle_t& h,const selected::Plan& compact,
    const PstedcVectorPlan& dc,std::size_t* backend_bytes,std::size_t* host_bytes)
{
    if (!compact.n) {
        pstedc_workspace_size<R>(h,dc,backend_bytes,host_bytes);
        return;
    }
    *backend_bytes=*host_bytes=0;
    if (ddlaGetBackend(h)==DdlaBackend::CPU)
        *backend_bytes=detail::padded_bytes<char>(selected::workspace_bytes(compact));
    else
        *host_bytes=detail::padded_bytes<R>(std::size_t(compact.ldy)*compact.nev)+
            detail::padded_bytes<char>(selected::workspace_bytes(compact));
}

template<DdlaBackend B,class R>
void tridiagonal_rows_run(const DdlaHandle_t& h,const selected::Plan& compact,
    const PstedcVectorPlan& dc,PstedcResources* dc_resources,R* d,R* e,R* w,R* Y,
    void* backend_work,std::size_t backend_bytes,void* host_work,std::size_t host_bytes)
{
    if (!compact.n) {
        assert(dc_resources);
        pstedc_run<B,R>(h,dc,*dc_resources,d,e,w,Y,nullptr,backend_work,backend_bytes,host_work,host_bytes);
        return;
    }
    R* rows=Y;
    void* scratch=backend_work;
    if constexpr (B==DdlaBackend::GPU) {
        char* cursor=static_cast<char*>(host_work);
        rows=detail::take<R>(cursor,std::size_t(compact.ldy)*compact.nev);
        scratch=cursor;
        assert(host_bytes>=std::size_t(cursor-static_cast<char*>(host_work))+selected::workspace_bytes(compact));
    } else {
        assert(backend_bytes>=selected::workspace_bytes(compact));
    }
    selected::Status status;
    {
        // Host II executes on CPU even for GPU handles. Respect the handle's
        // prepared host budget, and restore the caller's process-global BLAS
        // setting on success, numerical status failure and C++ unwinding.
        detail::ScopedBlasThreads threads(h->host_threads);
        status=selected::run<R>(compact,d,e,w,rows,scratch);
    }
    if (status.code!=selected::Code::success) {
        // The core returns one agreed numerical status before any consumer.
        char message[192];
        std::snprintf(message,sizeof(message),"pheevd: selected tridiagonal failure code=%d vector=%d native=%d",
            int(status.code),status.eigenvector,status.native_info);
        throw std::runtime_error(message);
    }
    if constexpr (B==DdlaBackend::GPU) {
        detail::agree_check(h,"pheevd selected row upload",[&](detail::Agree&) {
            if (compact.rows && compact.nev) {
                DDLA_CHECK(ddlaMemcpy2D(h,Y,std::size_t(compact.ldy)*sizeof(R),rows,
                    std::size_t(compact.ldy)*sizeof(R),std::size_t(compact.rows)*sizeof(R),compact.nev));
            }
            // Retire every host-source read before the shared arena is reused.
            DDLA_CHECK(ddlaSynchronize(h));
        });
    }
}
} }
#endif
