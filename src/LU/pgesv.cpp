#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "ddla_desc.h"
#include <algorithm>
#include <cassert>
#include <complex>
#include <cstddef>
#include <vector>
#include "ddla_connector.h"
#include <stdexcept>
#include <string>
#include "ddla_stream.h"
#include "require_backend.h"
#include "check_agree.h"
#include "plan_util.h"
#include "lu_panel.h"
#include "pgetrf_impl.h"
#include "pgetrs_impl.h"
#include "instantiate.h"

namespace ddla{

namespace lu_impl {

// Component 1.  The conditions of the former public sequence (pgetrf, then pgetrs), all checked up front.
template <DdlaBackend Backend>
bool pgesv_check(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                 const void* A, const int* array_descA, const int* array_descB)
{
    return detail::agree_check(handle, "pgesv", [&](detail::Agree& agree) -> bool {
        check_desc(array_descB, handle);
        check_desc(array_descA, handle);
        detail::require_backend<Backend>(handle, "pgesv");
        if (n > array_descA[DDLA_M_] || n > array_descA[DDLA_N_])
            throw std::invalid_argument("pgesv: n exceeds the descriptor of A");
        // pgetrf's region check of A(1:n, 1:n); the pivot vector belongs to the scratch.
        check_region("pgesv", handle, n, n, A, 1, 1, array_descA, nullptr, -1, false);
        check_solve("pgesv", side, trans, n, nrhs, array_descA, array_descB);
        const bool gather = gather_mode(handle);
        agree.mode("DDLA_LU_PANEL", gather ? 1 : 0);
        return gather;
    });
}

// Component 2: host bytes of [ the pivots (local rows of A's leading n x n block) | the rest ], backend bytes of
// the rest; the factorization and the solve run one after the other on the rest, so it is the larger of the two.
template <typename T>
void pgesv_workspace_size(const DdlaHandle_t& handle, bool gather, char side, char trans, int n, int nrhs,
                          T* A, const int* array_descA, const int* array_descB,
                          std::size_t* backend_bytes, std::size_t* host_bytes)
{
    *backend_bytes = *host_bytes = 0;
    if (n == 0) return;
    const int local_rows = num_loc(n, array_descA[DDLA_MB_], handle->myprow_, array_descA[DDLA_RSRC_],
                                   handle->nprows_);
    std::size_t factor_backend = 0, factor_host = 0, solve_backend = 0, solve_host = 0;
    pgetrf_workspace_size<T>(handle, gather, n, n, A, 1, 1, array_descA, &factor_backend, &factor_host);
    pgetrs_workspace_size<T>(handle, side, trans, n, nrhs, array_descA, array_descB, &solve_backend, &solve_host);
    *backend_bytes = std::max(factor_backend, solve_backend);
    *host_bytes = detail::padded_bytes<int>(std::size_t(local_rows)) + std::max(factor_host, solve_host);
}

// Component 3.
template <DdlaBackend Backend, typename T>
void pgesv_run(const DdlaHandle_t& handle, bool gather, char side, char trans, int n, int nrhs,
               T* A, const int* array_descA, T* B, const int* array_descB, int& info,
               void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes)
{
    try {
        info = 0;
        if (n == 0) return;
        // The scratch, in the order pgesv_workspace_size counts it.
        const int local_rows = num_loc(n, array_descA[DDLA_MB_], handle->myprow_, array_descA[DDLA_RSRC_],
                                       handle->nprows_);
        char* hw = static_cast<char*>(host_work);
        int* const ipiv = detail::take<int>(hw, std::size_t(local_rows));
        const std::size_t used = std::size_t(hw - static_cast<char*>(host_work));
        assert(used <= host_bytes);
        pgetrf_run<Backend, T>(handle, gather, n, n, A, 1, 1, array_descA, ipiv, info,
                               backend_work, backend_bytes, hw, host_bytes - used);
        if (info != 0 || nrhs == 0) return;
        pgetrs_run<Backend, T>(handle, side, trans, n, nrhs, A, array_descA, ipiv, B, array_descB,
                               backend_work, backend_bytes, hw, host_bytes - used);
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(handle) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(handle); } catch (...) {}
        throw;
    }
}

} // namespace lu_impl

template <DdlaBackend Backend, typename T>
void pgesv(
    const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
    T* A, const int* array_descA,
    T* B, const int* array_descB
)
{
    // 1. parameter check (collective; returns the agreed DDLA_LU_PANEL selector).
    const bool gather = lu_impl::pgesv_check<Backend>(handle, side, trans, n, nrhs, A, array_descA, array_descB);
    // No-op: an empty system.
    if (n == 0) return;
    // 2. the pivots and the scratch of the factorization and of the solve.
    std::size_t backend_bytes = 0, host_bytes = 0;
    lu_impl::pgesv_workspace_size<T>(handle, gather, side, trans, n, nrhs, A, array_descA, array_descB,
                                     &backend_bytes, &host_bytes);
    vector<char> backend_work(handle, backend_bytes);
    std::vector<char> host_work(host_bytes);
    // 3. factor, then solve.
    int info = 0;
    lu_impl::pgesv_run<Backend, T>(handle, gather, side, trans, n, nrhs, A, array_descA, B, array_descB, info,
                                   backend_work.data(), backend_bytes, host_work.data(), host_bytes);
    if(info !=0){
        throw std::runtime_error("pgesv: pgetrf returned info = " + std::to_string(info));
    }
}

#define INSTANTIATE_PGESV(BACKEND, TYPE)                                   \
    template void pgesv<BACKEND, TYPE>(const DdlaHandle_t&, char side, char trans, int n, int nrhs, \
    TYPE* A, const int* array_descA, \
    TYPE* B, const int* array_descB); \
    template void lu_impl::pgesv_run<BACKEND, TYPE>(const DdlaHandle_t&, bool, char, char, int, int, TYPE*, \
    const int*, TYPE*, const int*, int&, void*, std::size_t, void*, std::size_t)

#define INSTANTIATE_PGESV_SIZE(TYPE) \
    template void lu_impl::pgesv_workspace_size<TYPE>(const DdlaHandle_t&, bool, char, char, int, int, TYPE*, \
    const int*, const int*, std::size_t*, std::size_t*)

#define INSTANTIATE_PGESV_CHECK(BACKEND) \
    template bool lu_impl::pgesv_check<BACKEND>(const DdlaHandle_t&, char, char, int, int, const void*, \
    const int*, const int*)


DDLA_FOR_EACH_BACKEND(INSTANTIATE_PGESV_CHECK)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_PGESV)
DDLA_FOR_EACH_TYPE(INSTANTIATE_PGESV_SIZE)

#undef INSTANTIATE_PGESV_SIZE
#undef INSTANTIATE_PGESV_CHECK
#undef INSTANTIATE_PGESV

} // namespace ddla
