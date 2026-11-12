#include "BLAS/blas_impl.h"
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "ddla_desc.h"
#include <cassert>
#include <algorithm>
#include <complex>
#include <cstddef>
#include <vector>
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "require_backend.h"
#include "comm_traits.h"
#include "check_agree.h"
#include "plan_util.h"
#include "lu_panel.h"
#include "lapack_impl.h"
#include "pgetrf_impl.h"
#include "instantiate.h"

namespace ddla{

namespace lu_impl {

// Component 1.
template <DdlaBackend Backend>
bool pgetrf_check(const DdlaHandle_t& handle, int m, int n, const void* A, int ia, int ja,
                  const int* desc, const int* ipiv)
{
    return detail::agree_check(handle, "pgetrf", [&](detail::Agree& agree) -> bool {
        detail::require_backend<Backend>(handle, "pgetrf");
        check_desc(desc, handle);
        check_region("pgetrf", handle, m, n, A, ia, ja, desc, ipiv, -1, true);
        const bool gather = gather_mode(handle);
        agree.mode("DDLA_LU_PANEL", gather ? 1 : 0);
        return gather;
    });
}

// Component 2.  The panel calls of the schedule are the ones pgetrf_run makes (same panel_width), sized once here: no
// panel allocates.  Backend: [ the L panel (local rows x nb) | the diagonal block (nb x nb), which the U panel
// (nb x local columns) reuses after the step's trsm ], a region that every panel call also uses from its start:
// the call returns before the step copies the factored panel out, and the previous step's update is enqueued before
// it.  Host: the panel calls'.
template <typename T>
void pgetrf_workspace_size(const DdlaHandle_t& handle, bool gather, int m, int n, T* A, int ia, int ja,
                           const int* desc, std::size_t* backend_bytes, std::size_t* host_bytes)
{
    *backend_bytes = *host_bytes = 0;
    if (m == 0 || n == 0) return;
    if (handle->nprocs == 1) {
        lapack_impl::getrf_workspace_size<T>(handle,m,n,A+(ia-1)+std::size_t(ja-1)*desc[DDLA_LLD_],
            desc[DDLA_LLD_],backend_bytes);
        *backend_bytes += detail::padded_bytes<int>(std::min(m,n));
        if (handle->backend == DdlaBackend::GPU)
            *backend_bytes += detail::padded_bytes<int>(1);
        return;
    }
    int nprows, npcols, myprow, mypcol;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
    const int nb = desc[DDLA_MB_];
    const int row0 = ia - 1, col0 = ja - 1;
    const int local_rows = num_loc(row0 + m, nb, myprow, desc[DDLA_RSRC_], nprows)
                         - num_loc(row0, nb, myprow, desc[DDLA_RSRC_], nprows);
    const int local_cols = num_loc(col0 + n, nb, mypcol, desc[DDLA_CSRC_], npcols)
                         - num_loc(col0, nb, mypcol, desc[DDLA_CSRC_], npcols);
    std::size_t backend = detail::padded_bytes<T>(std::size_t(local_rows)*nb)
                        + detail::padded_bytes<T>(std::size_t(nb)*std::max(nb, local_cols));
    for (int k = 0; k < std::min(m,n); ) {
        const int b = panel_width(nb, row0, col0, m, n, k);
        std::size_t panel_backend = 0, panel_host = 0;
        pgetf2_workspace_size<T>(handle, gather, m-k, n-k, b, A, ia+k, ja+k, desc, ja, &panel_backend, &panel_host);
        backend = std::max(backend, panel_backend);
        *host_bytes = std::max(*host_bytes, panel_host);
        k += b;
    }
    *backend_bytes = backend;
}

// Component 3.
template <DdlaBackend Backend, typename T>
void pgetrf_run(const DdlaHandle_t& handle, bool gather, int m, int n, T* A, int ia, int ja,
                const int* desc, int* ipiv, int& info,
                void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes)
{
    try {
        info = 0;
        if (m == 0 || n == 0) return;

        if (handle->nprocs == 1) {
            const int k=std::min(m,n);
            char* work=static_cast<char*>(backend_work);
            int* piv=detail::take<int>(work,k);
            int* status=&info;
            std::size_t used=detail::padded_bytes<int>(k);
            if constexpr (Backend == DdlaBackend::GPU) {
                status=detail::take<int>(work,1);
                used+=detail::padded_bytes<int>(1);
            }
            lapack_impl::getrf_run<Backend,T>(handle,m,n,A+(ia-1)+std::size_t(ja-1)*desc[DDLA_LLD_],
                desc[DDLA_LLD_],piv,status,work,backend_bytes-used);
            if constexpr (Backend == DdlaBackend::GPU) {
                DDLA_CHECK(ddlaMemcpy(handle,&info,status,sizeof(int)));
                DDLA_CHECK(ddlaSynchronize(handle));
            }
            // Preserve the distributed interface's untouched pivot suffix on
            // singular input, although the local vendor continues factoring.
            const int count=info>0?std::min(k,info):(info==0?k:0);
            DDLA_CHECK(ddlaMemcpy(handle,ipiv+ia-1,piv,std::size_t(count)*sizeof(int)));
            if constexpr (Backend == DdlaBackend::GPU) DDLA_CHECK(ddlaSynchronize(handle));
            for (int i=0;i<count;++i) ipiv[ia-1+i]+=ia-1;
            return;
        }
        int nprows, npcols, myprow, mypcol;
        ddlaGetGridDims(handle, nprows, npcols);
        ddlaGetGridCoords(handle, myprow, mypcol);
        const int nb = desc[DDLA_MB_], ld = desc[DDLA_LLD_];
        const int row0 = ia - 1, col0 = ja - 1;
        const int row_end = num_loc(row0 + m, nb, myprow, desc[DDLA_RSRC_], nprows);
        const int col_end = num_loc(col0 + n, nb, mypcol, desc[DDLA_CSRC_], npcols);
        const int local_rows = row_end - num_loc(row0, nb, myprow, desc[DDLA_RSRC_], nprows);
        const int local_cols = col_end - num_loc(col0, nb, mypcol, desc[DDLA_CSRC_], npcols);

        // The scratch, in the order pgetrf_workspace_size counts it; the panel calls use the whole region.
        char* w = static_cast<char*>(backend_work);
        T* const temp_L = detail::take<T>(w, std::size_t(local_rows)*nb);
        T* const temp_block = detail::take<T>(w, std::size_t(nb)*std::max(nb, local_cols));
        T* const temp_U = temp_block;
        assert(std::size_t(w - static_cast<char*>(backend_work)) <= backend_bytes);

        for (int k = 0; k < std::min(m,n); ) {
            const int r = row0 + k, c = col0 + k;
            // Independent offsets can reach row and column block boundaries at
            // different steps. Each diagonal block must fit both distributions.
            const int b = panel_width(nb, row0, col0, m, n, k);
            const int il = indx_g2l_r(desc, handle, r), jl = indx_g2l_c(desc, handle, c);
            const int pr = indxg2p(r, nb, desc[DDLA_RSRC_], nprows);
            const int pc = indxg2p(c, nb, desc[DDLA_CSRC_], npcols);

            pgetf2_run<Backend,T>(handle, gather, m-k, n-k, b, A, ia+k, ja+k, desc, ipiv, info, ja,
                                  backend_work, backend_bytes, host_work, host_bytes);
            if (info != 0) { if (info > 0) info += k; break; }

            const int row_after = num_loc(r+b, nb, myprow, desc[DDLA_RSRC_], nprows);
            const int col_after = num_loc(c+b, nb, mypcol, desc[DDLA_CSRC_], npcols);
            if (jl >= 0) {
                if (il >= 0)
                    DDLA_CHECK(ddlaMemcpy2D(handle, temp_block, std::size_t(b)*sizeof(T),
                        A+il+std::size_t(jl)*ld, std::size_t(ld)*sizeof(T), std::size_t(b)*sizeof(T), b));
                if (row_after < row_end)
                    DDLA_CHECK(ddlaMemcpy2D(handle, temp_L, std::size_t(row_end-row_after)*sizeof(T),
                        A+row_after+std::size_t(jl)*ld, std::size_t(ld)*sizeof(T),
                        std::size_t(row_end-row_after)*sizeof(T), b));
            }
            if (row_after < row_end)
                commBcast<Backend>(handle, CommScope::Row, temp_L, std::size_t(row_end-row_after)*b, pc);
            if (il >= 0) {
                commBcast<Backend>(handle, CommScope::Row, temp_block, std::size_t(b)*b, pc);
                if (col_after < col_end) {
                    blas_impl::trsm_run<Backend,T>(handle, 'L', 'L', 'N', 'U', b, col_end-col_after, T(1),
                        temp_block, b, A+il+std::size_t(col_after)*ld, ld);
                    DDLA_CHECK(ddlaMemcpy2D(handle, temp_U, std::size_t(b)*sizeof(T),
                        A+il+std::size_t(col_after)*ld, std::size_t(ld)*sizeof(T),
                        std::size_t(b)*sizeof(T), col_end-col_after));
                }
            }
            if (col_after < col_end)
                commBcast<Backend>(handle, CommScope::Col, temp_U, std::size_t(b)*(col_end-col_after), pr);
            if (row_after < row_end && col_after < col_end)
                blas_impl::gemm_run<Backend,T>(handle, 'N', 'N', row_end-row_after, col_end-col_after, b,
                    -1.0, temp_L, row_end-row_after, temp_U, b, 1.0,
                    A+row_after+std::size_t(col_after)*ld, ld);
            k += b;
        }
        DDLA_CHECK(ddlaSynchronize(handle));
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(handle) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(handle); } catch (...) {}
        throw;
    }
}

} // namespace lu_impl

template <DdlaBackend Backend, typename T>
void pgetrf(const DdlaHandle_t& handle, int m, int n, T* A, int ia, int ja,
             const int* desc, int* ipiv, int& info)
{
    // 1. parameter check (collective; returns the agreed DDLA_LU_PANEL selector).
    const bool gather = lu_impl::pgetrf_check<Backend>(handle, m, n, A, ia, ja, desc, ipiv);
    // No-op: an empty region accesses no data.
    info = 0;
    if (m == 0 || n == 0) return;
    // 2. the scratch of the whole factorization, panel calls included.
    std::size_t backend_bytes = 0, host_bytes = 0;
    lu_impl::pgetrf_workspace_size<T>(handle, gather, m, n, A, ia, ja, desc, &backend_bytes, &host_bytes);
    vector<char> backend_work(handle, backend_bytes);
    std::vector<char> host_work(host_bytes);
    // 3. the factorization.
    lu_impl::pgetrf_run<Backend,T>(handle, gather, m, n, A, ia, ja, desc, ipiv, info,
                                   backend_work.data(), backend_bytes, host_work.data(), host_bytes);
}

#define INSTANTIATE_PGETRF(BACKEND, TYPE)                                   \
    template void pgetrf<BACKEND, TYPE>(const DdlaHandle_t&, int m, int n, \
    TYPE* A, int ia, int ja, const int* array_descA, \
    int* ipiv, \
    int& info); \
    template void lu_impl::pgetrf_run<BACKEND, TYPE>(const DdlaHandle_t&, bool, int, int, TYPE*, int, int, \
    const int*, int*, int&, void*, std::size_t, void*, std::size_t)

#define INSTANTIATE_PGETRF_SIZE(TYPE) \
    template void lu_impl::pgetrf_workspace_size<TYPE>(const DdlaHandle_t&, bool, int, int, TYPE*, int, int, \
    const int*, std::size_t*, std::size_t*)

#define INSTANTIATE_PGETRF_CHECK(BACKEND) \
    template bool lu_impl::pgetrf_check<BACKEND>(const DdlaHandle_t&, int, int, const void*, int, int, \
    const int*, const int*)


DDLA_FOR_EACH_BACKEND(INSTANTIATE_PGETRF_CHECK)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_PGETRF)
DDLA_FOR_EACH_TYPE(INSTANTIATE_PGETRF_SIZE)

#undef INSTANTIATE_PGETRF_SIZE
#undef INSTANTIATE_PGETRF_CHECK
#undef INSTANTIATE_PGETRF

} // namespace ddla
