#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "lu_internal.h"
#include "ddla_desc.h"
#include <algorithm>
#include <cassert>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include "ddla_stream.h"
#include "require_backend.h"
#include "check_agree.h"
#include "plan_util.h"
#include "lu_panel.h"
#include "pgetrf_impl.h"
#include "pgetrs_impl.h"

namespace ddla {

namespace lu_impl {

// Component 1.  The conditions of the former public sequence (pgetrf_bpiv, then pgetrs_bpiv), checked up front.
template <DdlaBackend Backend>
void pgesv_bpiv_check(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                      const int* array_descA, const int* array_descB)
{
    detail::agree_check(handle, "pgesv_bpiv", [&](detail::Agree&) {
        check_desc(array_descB, handle);
        check_desc(array_descA, handle);
        detail::require_backend<Backend>(handle, "pgesv_bpiv");
        check_solve("pgesv_bpiv", side, trans, n, nrhs, array_descA, array_descB);
    });
}

// Component 2: backend bytes of [ the block pivots (max(1, local rows of A's leading n x n block)) | the rest ];
// the factorization and the solve run one after the other on the rest (and on the host bytes), so it is the larger
// of the two.
template <typename T>
void pgesv_bpiv_workspace_size(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                               T* A, const int* array_descA, const int* array_descB,
                               std::size_t* backend_bytes, std::size_t* host_bytes)
{
    *backend_bytes = *host_bytes = 0;
    if (n == 0) return;
    const int m_loc_A = num_loc(n, array_descA[DDLA_MB_], handle->myprow_, array_descA[DDLA_RSRC_], handle->nprows_);
    std::size_t factor_backend = 0, factor_host = 0, solve_backend = 0, solve_host = 0;
    pgetrf_bpiv_workspace_size<T>(handle, n, n, A, array_descA, &factor_backend, &factor_host);
    pgetrs_bpiv_workspace_size<T>(handle, side, trans, n, nrhs, array_descA, array_descB,
                                  &solve_backend, &solve_host);
    *backend_bytes = detail::padded_bytes<int>(std::size_t(std::max(1, m_loc_A)))
                   + std::max(factor_backend, solve_backend);
    *host_bytes = std::max(factor_host, solve_host);
}

// Component 3.
template <DdlaBackend Backend, typename T>
void pgesv_bpiv_run(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                    T* A, const int* array_descA, T* B, const int* array_descB, int& info,
                    void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes)
{
    try {
        info = 0;
        if (n == 0) return;
        // The scratch, in the order pgesv_bpiv_workspace_size counts it.
        const int m_loc_A = num_loc(n, array_descA[DDLA_MB_], handle->myprow_, array_descA[DDLA_RSRC_], handle->nprows_);
        char* w = static_cast<char*>(backend_work);
        int* const pivots = detail::take<int>(w, std::size_t(std::max(1, m_loc_A)));
        const std::size_t used = std::size_t(w - static_cast<char*>(backend_work));
        assert(used <= backend_bytes);
        pgetrf_bpiv_run<Backend, T>(handle, n, n, A, array_descA, pivots, info,
                                    w, backend_bytes - used, host_work, host_bytes);
        if (info != 0 || nrhs == 0) return;
        pgetrs_bpiv_run<Backend, T>(handle, side, trans, n, nrhs, A, array_descA, pivots, B, array_descB,
                                    w, backend_bytes - used, host_work, host_bytes);
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(handle) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(handle); } catch (...) {}
        throw;
    }
}

} // namespace lu_impl

/**
 * @brief Distributed linear-system solver using block-partial-pivoting LU
 *        (driver): solve op(A) * X = B (side='L') or X * op(A) = B
 *        (side='R').
 *
 * Convenience wrapper: pgetrf_bpiv (block LU with partial pivoting within
 * each block row) + pgetrs_bpiv (solve).  Corresponds to the same
 * ScaLAPACK-style interface as pgesv, but with block-partial pivoting.
 *
 * @tparam T   Scalar type.
 * @param side    'L' -- solve op(A)*X = B (B is n x nrhs);
 *                'R' -- solve X*op(A) = B (B is nrhs x n).
 * @param trans   'N', 'T' or 'C' -- operation applied to A.
 * @param n       Order of square matrix A.
 * @param nrhs    Number of right-hand sides.
 * @param A     Pointer in the selected backend to A (input: coefficient; output: LU factors).
 * @param array_descA  ScaLAPACK int[9] descriptor for A.
 * @param B     Pointer in the selected backend to RHS / solution B (input/output).
 * @param array_descB  ScaLAPACK int[9] descriptor for B.
 * @throws std::runtime_error if LU factorization fails (info != 0).
 */
template <DdlaBackend Backend, typename T>
void pgesv_bpiv(
    const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
    T* A, const int* array_descA,
    T* B, const int* array_descB
)
{
    // 1. parameter check (collective).
    lu_impl::pgesv_bpiv_check<Backend>(handle, side, trans, n, nrhs, array_descA, array_descB);
    // No-op: an empty system.
    if (n == 0) return;
    // 2. the block pivots and the scratch of the factorization and of the solve.
    std::size_t backend_bytes = 0, host_bytes = 0;
    lu_impl::pgesv_bpiv_workspace_size<T>(handle, side, trans, n, nrhs, A, array_descA, array_descB,
                                          &backend_bytes, &host_bytes);
    vector<char> backend_work(handle, backend_bytes);
    std::vector<char> host_work(host_bytes);
    // 3. factor, then solve.
    int info = 0;
    lu_impl::pgesv_bpiv_run<Backend, T>(handle, side, trans, n, nrhs, A, array_descA, B, array_descB, info,
                                        backend_work.data(), backend_bytes, host_work.data(), host_bytes);
    if(info != 0){
        throw std::runtime_error("pgesv_bpiv: pgetrf_bpiv returned info = " + std::to_string(info));
    }
}

#define INSTANTIATE_PGESV_BPIV(BACKEND, TYPE)                                   \
    template void pgesv_bpiv<BACKEND, TYPE>(const DdlaHandle_t&, char side, char trans, int n, int nrhs, \
    TYPE* A, const int* array_descA, \
    TYPE* B, const int* array_descB); \
    template void lu_impl::pgesv_bpiv_run<BACKEND, TYPE>(const DdlaHandle_t&, char, char, int, int, TYPE*, \
    const int*, TYPE*, const int*, int&, void*, std::size_t, void*, std::size_t)

#define INSTANTIATE_PGESV_BPIV_SIZE(TYPE) \
    template void lu_impl::pgesv_bpiv_workspace_size<TYPE>(const DdlaHandle_t&, char, char, int, int, TYPE*, \
    const int*, const int*, std::size_t*, std::size_t*)

#define INSTANTIATE_PGESV_BPIV_CHECK(BACKEND) \
    template void lu_impl::pgesv_bpiv_check<BACKEND>(const DdlaHandle_t&, char, char, int, int, const int*, \
    const int*)


#if DDLA_HAS_CPU
INSTANTIATE_PGESV_BPIV_CHECK(DdlaBackend::CPU);
INSTANTIATE_PGESV_BPIV(DdlaBackend::CPU, float);
INSTANTIATE_PGESV_BPIV(DdlaBackend::CPU, double);
INSTANTIATE_PGESV_BPIV(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PGESV_BPIV(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_PGESV_BPIV_CHECK(DdlaBackend::GPU);
INSTANTIATE_PGESV_BPIV(DdlaBackend::GPU, float);
INSTANTIATE_PGESV_BPIV(DdlaBackend::GPU, double);
INSTANTIATE_PGESV_BPIV(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PGESV_BPIV(DdlaBackend::GPU, std::complex<double>);
#endif

INSTANTIATE_PGESV_BPIV_SIZE(float);
INSTANTIATE_PGESV_BPIV_SIZE(double);
INSTANTIATE_PGESV_BPIV_SIZE(std::complex<float>);
INSTANTIATE_PGESV_BPIV_SIZE(std::complex<double>);

#undef INSTANTIATE_PGESV_BPIV_SIZE
#undef INSTANTIATE_PGESV_BPIV_CHECK
#undef INSTANTIATE_PGESV_BPIV

} // namespace ddla
