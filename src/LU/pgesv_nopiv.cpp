#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "lu_internal.h"
#include "ddla_desc.h"
#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include "ddla_stream.h"
#include "require_backend.h"
#include "check_agree.h"
#include "lu_panel.h"
#include "pgetrf_impl.h"
#include "pgetrs_impl.h"

namespace ddla {

namespace lu_impl {

// Component 1.  The conditions of the former public sequence (pgetrf_nopiv, then pgetrs_nopiv), checked up front.
template <DdlaBackend Backend>
void pgesv_nopiv_check(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                       const int* array_descA, const int* array_descB)
{
    detail::agree_check(handle, "pgesv_nopiv", [&](detail::Agree&) {
        check_desc(array_descB, handle);
        check_desc(array_descA, handle);
        detail::require_backend<Backend>(handle, "pgesv_nopiv");
        check_solve("pgesv_nopiv", side, trans, n, nrhs, array_descA, array_descB);
    });
}

// Component 2: the factorization and the solve run one after the other on the same bytes: the larger of the two.
template <typename T>
void pgesv_nopiv_workspace_size(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                                const int* array_descA, const int* array_descB,
                                std::size_t* backend_bytes, std::size_t* host_bytes)
{
    *backend_bytes = *host_bytes = 0;
    if (n == 0) return;
    std::size_t factor_bytes = 0, solve_bytes = 0;
    pgetrf_nopiv_workspace_size<T>(handle, n, n, array_descA, &factor_bytes);
    pgetrs_nopiv_workspace_size<T>(handle, side, trans, n, nrhs, array_descA, array_descB, &solve_bytes, host_bytes);
    *backend_bytes = std::max(factor_bytes, solve_bytes);
}

// Component 3.
template <DdlaBackend Backend, typename T>
void pgesv_nopiv_run(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                     T* A, const int* array_descA, T* B, const int* array_descB, int& info,
                     void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes)
{
    try {
        info = 0;
        if (n == 0) return;
        pgetrf_nopiv_run<Backend, T>(handle, n, n, A, array_descA, info, backend_work, backend_bytes);
        if (info != 0 || nrhs == 0) return;
        pgetrs_nopiv_run<Backend, T>(handle, side, trans, n, nrhs, A, array_descA, B, array_descB,
                                     backend_work, backend_bytes, host_work, host_bytes);
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(handle) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(handle); } catch (...) {}
        throw;
    }
}

} // namespace lu_impl

/**
 * @brief Distributed linear-system solver without pivoting (driver).
 *
 * Convenience wrapper: pgetrf_nopiv (LU) + pgetrs_nopiv (solve).
 * Solves A * X = B without pivoting.
 *
 * @tparam T   Scalar type.
 * @param n       Order of square matrix A.
 * @param nrhs    Number of right-hand sides.
 * @param A     Pointer in the selected backend to A (input: coefficient; output: LU factors).
 * @param array_descA  ScaLAPACK int[9] descriptor for A.
 * @param B     Pointer in the selected backend to RHS / solution B (input/output).
 * @param array_descB  ScaLAPACK int[9] descriptor for B.
 * @throws std::runtime_error if LU factorization fails (info != 0).
 */
template <DdlaBackend Backend, typename T>
void pgesv_nopiv(
    const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
    T* A, const int* array_descA,
    T* B, const int* array_descB
)
{
    // 1. parameter check (collective).
    lu_impl::pgesv_nopiv_check<Backend>(handle, side, trans, n, nrhs, array_descA, array_descB);
    // No-op: an empty system.
    if (n == 0) return;
    // 2. the scratch of the factorization and of the solve.
    std::size_t backend_bytes = 0, host_bytes = 0;
    lu_impl::pgesv_nopiv_workspace_size<T>(handle, side, trans, n, nrhs, array_descA, array_descB,
                                           &backend_bytes, &host_bytes);
    vector<char> backend_work(handle, backend_bytes);
    std::vector<char> host_work(host_bytes);
    // 3. factor, then solve.
    int info = 1;
    lu_impl::pgesv_nopiv_run<Backend, T>(handle, side, trans, n, nrhs, A, array_descA, B, array_descB, info,
                                         backend_work.data(), backend_bytes, host_work.data(), host_bytes);
    if (info != 0) {
        throw std::runtime_error("pgesv_nopiv: pgetrf_nopiv returned info = " + std::to_string(info));
    }
}

#define INSTANTIATE_PGESV_NOPIV(BACKEND, TYPE)                                   \
    template void pgesv_nopiv<BACKEND, TYPE>(const DdlaHandle_t&, char side, char trans, int n, int nrhs, \
    TYPE* A, const int* array_descA, \
    TYPE* B, const int* array_descB); \
    template void lu_impl::pgesv_nopiv_run<BACKEND, TYPE>(const DdlaHandle_t&, char, char, int, int, TYPE*, \
    const int*, TYPE*, const int*, int&, void*, std::size_t, void*, std::size_t)

#define INSTANTIATE_PGESV_NOPIV_SIZE(TYPE) \
    template void lu_impl::pgesv_nopiv_workspace_size<TYPE>(const DdlaHandle_t&, char, char, int, int, const int*, \
    const int*, std::size_t*, std::size_t*)

#define INSTANTIATE_PGESV_NOPIV_CHECK(BACKEND) \
    template void lu_impl::pgesv_nopiv_check<BACKEND>(const DdlaHandle_t&, char, char, int, int, const int*, \
    const int*)


#if DDLA_HAS_CPU
INSTANTIATE_PGESV_NOPIV_CHECK(DdlaBackend::CPU);
INSTANTIATE_PGESV_NOPIV(DdlaBackend::CPU, float);
INSTANTIATE_PGESV_NOPIV(DdlaBackend::CPU, double);
INSTANTIATE_PGESV_NOPIV(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PGESV_NOPIV(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_PGESV_NOPIV_CHECK(DdlaBackend::GPU);
INSTANTIATE_PGESV_NOPIV(DdlaBackend::GPU, float);
INSTANTIATE_PGESV_NOPIV(DdlaBackend::GPU, double);
INSTANTIATE_PGESV_NOPIV(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PGESV_NOPIV(DdlaBackend::GPU, std::complex<double>);
#endif

INSTANTIATE_PGESV_NOPIV_SIZE(float);
INSTANTIATE_PGESV_NOPIV_SIZE(double);
INSTANTIATE_PGESV_NOPIV_SIZE(std::complex<float>);
INSTANTIATE_PGESV_NOPIV_SIZE(std::complex<double>);

#undef INSTANTIATE_PGESV_NOPIV_SIZE
#undef INSTANTIATE_PGESV_NOPIV_CHECK
#undef INSTANTIATE_PGESV_NOPIV

} // namespace ddla
