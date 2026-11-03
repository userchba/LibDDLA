#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "lu_internal.h"
#include "ddla_desc.h"
#include <complex>
#include <cstddef>
#include <vector>
#include "ddla_stream.h"
#include "require_backend.h"
#include "check_agree.h"
#include "lu_panel.h"
#include "pgetrs_impl.h"
#include "PBLAS/ptrtrs_impl.h"

namespace ddla {

namespace lu_impl {

// Component 1.
template <DdlaBackend Backend>
void pgetrs_nopiv_check(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                        const int* array_descA, const int* array_descB)
{
    detail::agree_check(handle, "pgetrs_nopiv", [&](detail::Agree&) {
        check_desc(array_descB, handle);
        check_desc(array_descA, handle);
        detail::require_backend<Backend>(handle, "pgetrs_nopiv");
        check_solve("pgetrs_nopiv", side, trans, n, nrhs, array_descA, array_descB);
    });
}

// Component 2: the triangular solves' scratch; one size serves both (both use `trans`).
template <typename T>
void pgetrs_nopiv_workspace_size(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                                 const int* array_descA, const int* array_descB,
                                 std::size_t* backend_bytes, std::size_t* host_bytes)
{
    *backend_bytes = *host_bytes = 0;
    if (n == 0 || nrhs == 0) return;
    ptrtrs_impl::ptrtrs_workspace_size<T>(handle, side, trans, side == 'L' ? n : nrhs, side == 'L' ? nrhs : n,
                                          array_descA, array_descB, backend_bytes, host_bytes);
}

// Component 3.
template <DdlaBackend Backend, typename T>
void pgetrs_nopiv_run(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                      T* A, const int* array_descA, T* B, const int* array_descB,
                      void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes)
{
    try {
        if (n == 0 || nrhs == 0) return;
        // Leading-block sub-matrix: descriptors may be larger than the logical
        // sub-matrix (anchored at global (0,0)).
        const int b_rows = (side == 'L') ? n : nrhs;
        const int b_cols = (side == 'L') ? nrhs : n;

        if(side == 'L'){
            if(trans == 'N'){
                // A = L*U => X = U^-1 * L^-1 * B: solve L then U.
                ptrtrs_impl::ptrtrs_run<Backend, T>(handle, 'L', 'L', 'N', 'U', b_rows, b_cols,
                       A, array_descA, B, array_descB, backend_work, backend_bytes, host_work, host_bytes);
                ptrtrs_impl::ptrtrs_run<Backend, T>(handle, 'L', 'U', 'N', 'N', b_rows, b_cols,
                       A, array_descA, B, array_descB, backend_work, backend_bytes, host_work, host_bytes);
            }else{
                // A^T = U^T * L^T => X = L^-T * U^-T * B: solve U^T then L^T.
                ptrtrs_impl::ptrtrs_run<Backend, T>(handle, 'L', 'U', trans, 'N', b_rows, b_cols,
                       A, array_descA, B, array_descB, backend_work, backend_bytes, host_work, host_bytes);
                ptrtrs_impl::ptrtrs_run<Backend, T>(handle, 'L', 'L', trans, 'U', b_rows, b_cols,
                       A, array_descA, B, array_descB, backend_work, backend_bytes, host_work, host_bytes);
            }
        }else{
            if(trans == 'N'){
                // X * L * U = B => X = B * U^-1 * L^-1: solve U then L.
                ptrtrs_impl::ptrtrs_run<Backend, T>(handle, 'R', 'U', 'N', 'N', b_rows, b_cols,
                       A, array_descA, B, array_descB, backend_work, backend_bytes, host_work, host_bytes);
                ptrtrs_impl::ptrtrs_run<Backend, T>(handle, 'R', 'L', 'N', 'U', b_rows, b_cols,
                       A, array_descA, B, array_descB, backend_work, backend_bytes, host_work, host_bytes);
            }else{
                // X * U^T * L^T = B => X = B * L^-T * U^-T: solve L^T then U^T.
                ptrtrs_impl::ptrtrs_run<Backend, T>(handle, 'R', 'L', trans, 'U', b_rows, b_cols,
                       A, array_descA, B, array_descB, backend_work, backend_bytes, host_work, host_bytes);
                ptrtrs_impl::ptrtrs_run<Backend, T>(handle, 'R', 'U', trans, 'N', b_rows, b_cols,
                       A, array_descA, B, array_descB, backend_work, backend_bytes, host_work, host_bytes);
            }
        }
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(handle) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(handle); } catch (...) {}
        throw;
    }
}

} // namespace lu_impl

/**
 * @brief Distributed LU solve without pivoting.
 *
 * Solves A * X = B using the factors from pgetrf_nopiv.  Because no
 * pivoting was performed, the solve is simply two triangular solves:
 *   1) L * Y = B  (forward, lower triangular, unit diagonal)
 *   2) U * X = Y  (backward, upper triangular, non-unit diagonal)
 *
 * Only trans='N' is supported.
 *
 * @tparam T   Scalar type.
 * @param trans   'N' -- no transpose (only 'N' supported).
 * @param n       Order of matrix A.
 * @param nrhs    Number of right-hand sides.
 * @param A     Pointer in the selected backend to LU factors (from pgetrf_nopiv).
 * @param array_descA  ScaLAPACK int[9] descriptor for A.
 * @param B     Pointer in the selected backend to RHS / solution B (input/output).
 * @param array_descB  ScaLAPACK int[9] descriptor for B.
 */
template <DdlaBackend Backend, typename T>
void pgetrs_nopiv(
    const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
    T* A, const int* array_descA,
    T* B, const int* array_descB
)
{
    // 1. parameter check (collective).
    lu_impl::pgetrs_nopiv_check<Backend>(handle, side, trans, n, nrhs, array_descA, array_descB);
    // No-op: nothing to solve for (B is left as it is).
    if (n == 0 || nrhs == 0) return;
    // 2. the scratch of the triangular solves.
    std::size_t backend_bytes = 0, host_bytes = 0;
    lu_impl::pgetrs_nopiv_workspace_size<T>(handle, side, trans, n, nrhs, array_descA, array_descB,
                                            &backend_bytes, &host_bytes);
    vector<char> backend_work(handle, backend_bytes);
    std::vector<char> host_work(host_bytes);
    // 3. the solve.
    lu_impl::pgetrs_nopiv_run<Backend, T>(handle, side, trans, n, nrhs, A, array_descA, B, array_descB,
                                          backend_work.data(), backend_bytes, host_work.data(), host_bytes);
}

#define INSTANTIATE_PGETRS_NOPIV(BACKEND, TYPE)                                   \
    template void pgetrs_nopiv<BACKEND, TYPE>(const DdlaHandle_t&, char side, char trans, int n, int nrhs, \
    TYPE* A, const int* array_descA, \
    TYPE* B, const int* array_descB); \
    template void lu_impl::pgetrs_nopiv_run<BACKEND, TYPE>(const DdlaHandle_t&, char, char, int, int, TYPE*, \
    const int*, TYPE*, const int*, void*, std::size_t, void*, std::size_t)

#define INSTANTIATE_PGETRS_NOPIV_SIZE(TYPE) \
    template void lu_impl::pgetrs_nopiv_workspace_size<TYPE>(const DdlaHandle_t&, char, char, int, int, const int*, \
    const int*, std::size_t*, std::size_t*)

#define INSTANTIATE_PGETRS_NOPIV_CHECK(BACKEND) \
    template void lu_impl::pgetrs_nopiv_check<BACKEND>(const DdlaHandle_t&, char, char, int, int, const int*, \
    const int*)


INSTANTIATE_PGETRS_NOPIV_CHECK(DdlaBackend::CPU);
INSTANTIATE_PGETRS_NOPIV(DdlaBackend::CPU, float);
INSTANTIATE_PGETRS_NOPIV(DdlaBackend::CPU, double);
INSTANTIATE_PGETRS_NOPIV(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PGETRS_NOPIV(DdlaBackend::CPU, std::complex<double>);

#if DDLA_HAS_GPU
INSTANTIATE_PGETRS_NOPIV_CHECK(DdlaBackend::GPU);
INSTANTIATE_PGETRS_NOPIV(DdlaBackend::GPU, float);
INSTANTIATE_PGETRS_NOPIV(DdlaBackend::GPU, double);
INSTANTIATE_PGETRS_NOPIV(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PGETRS_NOPIV(DdlaBackend::GPU, std::complex<double>);
#endif

INSTANTIATE_PGETRS_NOPIV_SIZE(float);
INSTANTIATE_PGETRS_NOPIV_SIZE(double);
INSTANTIATE_PGETRS_NOPIV_SIZE(std::complex<float>);
INSTANTIATE_PGETRS_NOPIV_SIZE(std::complex<double>);

#undef INSTANTIATE_PGETRS_NOPIV_SIZE
#undef INSTANTIATE_PGETRS_NOPIV_CHECK
#undef INSTANTIATE_PGETRS_NOPIV

} // namespace ddla
