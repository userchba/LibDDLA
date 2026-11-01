#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "ddla_desc.h"
#include <algorithm>
#include <complex>
#include <cstddef>
#include <vector>
#include "ddla_stream.h"
#include "require_backend.h"
#include "check_agree.h"
#include "lu_panel.h"
#include "pgetrs_impl.h"
#include "PBLAS/plapiv_impl.h"
#include "PBLAS/ptrtrs_impl.h"

namespace ddla{

namespace lu_impl {

// Component 1.
template <DdlaBackend Backend>
void pgetrs_check(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                  const int* array_descA, const int* array_descB)
{
    detail::agree_check(handle, "pgetrs", [&](detail::Agree&) {
        check_desc(array_descB, handle);
        check_desc(array_descA, handle);
        detail::require_backend<Backend>(handle, "pgetrs");
        check_solve("pgetrs", side, trans, n, nrhs, array_descA, array_descB);
    });
}

// Component 2.  side='L': B is n x nrhs, solves op(A)*X = B; side='R': B is nrhs x n, solves X*op(A) = B.  The
// two triangular solves use `trans`, so one ptrtrs size serves both; plapiv's exchange buffer covers the nrhs-long
// segments of B.  The pivoting and the solves run one after the other on the same bytes.
template <typename T>
void pgetrs_workspace_size(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                           const int* array_descA, const int* array_descB,
                           std::size_t* backend_bytes, std::size_t* host_bytes)
{
    *backend_bytes = *host_bytes = 0;
    if (n == 0 || nrhs == 0) return;
    std::size_t solve_bytes = 0, pivot_bytes = 0;
    ptrtrs_impl::ptrtrs_workspace_size<T>(handle, side, trans, side == 'L' ? n : nrhs, side == 'L' ? nrhs : n,
                                          array_descA, array_descB, &solve_bytes, host_bytes);
    plapiv_impl::plapiv_workspace_size<T>(handle, nrhs, array_descB, &pivot_bytes);
    *backend_bytes = std::max(solve_bytes, pivot_bytes);
}

// Component 3.
template<DdlaBackend Backend,class T>
void pgetrs_run(const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
                T* A, const int* array_descA,
                const int* ipiv, // host
                T* B, const int* array_descB,
                void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes)
{
    try {
        if(n==0 || nrhs==0)return;

        // Descriptors may describe matrices larger than the logical sub-matrix
        // (leading-block); only the leading n columns of A and the corresponding
        // leading block of B are referenced.
        const int b_rows = (side == 'L') ? n : nrhs;
        const int b_cols = (side == 'L') ? nrhs : n;

        // Apply the pgetrf pivot permutation to B via plapiv.  With
        // A = P*L*U and P = P(0)*...*P(n-2), P(k) swapping rows k and
        // ipiv(k)-1, forward application computes P^T*B (rows) or B*P (columns),
        // backward application computes P*Z (rows) or Z*P^T (columns).  rowcol
        // selects rows of B (side='L') or columns of B (side='R'); direc selects
        // the application order.  The pivot vector follows array_descA's row
        // distribution, so array_descIP is dA.
        if(side == 'L'){
            if(trans == 'N'){
                // A = P*L*U => X = U^-1 * L^-1 * (P^T * B)
                plapiv_impl::plapiv_run<Backend, T>(handle, 'F', 'R', n, nrhs, B, array_descB, ipiv, array_descA,
                                                    backend_work, backend_bytes);
                ptrtrs_impl::ptrtrs_run<Backend, T>(handle, 'L', 'L', 'N', 'U', b_rows, b_cols, A, array_descA, B, array_descB,
                                                    backend_work, backend_bytes, host_work, host_bytes);
                ptrtrs_impl::ptrtrs_run<Backend, T>(handle, 'L', 'U', 'N', 'N', b_rows, b_cols, A, array_descA, B, array_descB,
                                                    backend_work, backend_bytes, host_work, host_bytes);
            }else{
                // op(A)^T = U^T * L^T * P^T => X = P * L^-T * U^-T * B
                ptrtrs_impl::ptrtrs_run<Backend, T>(handle, 'L', 'U', trans, 'N', b_rows, b_cols, A, array_descA, B, array_descB,
                                                    backend_work, backend_bytes, host_work, host_bytes);
                ptrtrs_impl::ptrtrs_run<Backend, T>(handle, 'L', 'L', trans, 'U', b_rows, b_cols, A, array_descA, B, array_descB,
                                                    backend_work, backend_bytes, host_work, host_bytes);
                plapiv_impl::plapiv_run<Backend, T>(handle, 'B', 'R', n, nrhs, B, array_descB, ipiv, array_descA,
                                                    backend_work, backend_bytes);
            }
        }else{
            if(trans == 'N'){
                // X * P * L * U = B => X = B * U^-1 * L^-1 * P^T
                ptrtrs_impl::ptrtrs_run<Backend, T>(handle, 'R', 'U', 'N', 'N', b_rows, b_cols, A, array_descA, B, array_descB,
                                                    backend_work, backend_bytes, host_work, host_bytes);
                ptrtrs_impl::ptrtrs_run<Backend, T>(handle, 'R', 'L', 'N', 'U', b_rows, b_cols, A, array_descA, B, array_descB,
                                                    backend_work, backend_bytes, host_work, host_bytes);
                plapiv_impl::plapiv_run<Backend, T>(handle, 'B', 'C', n, nrhs, B, array_descB, ipiv, array_descA,
                                                    backend_work, backend_bytes);
            }else{
                // X * U^T * L^T * P^T = B => X = B * P * L^-T * U^-T
                plapiv_impl::plapiv_run<Backend, T>(handle, 'F', 'C', n, nrhs, B, array_descB, ipiv, array_descA,
                                                    backend_work, backend_bytes);
                ptrtrs_impl::ptrtrs_run<Backend, T>(handle, 'R', 'L', trans, 'U', b_rows, b_cols, A, array_descA, B, array_descB,
                                                    backend_work, backend_bytes, host_work, host_bytes);
                ptrtrs_impl::ptrtrs_run<Backend, T>(handle, 'R', 'U', trans, 'N', b_rows, b_cols, A, array_descA, B, array_descB,
                                                    backend_work, backend_bytes, host_work, host_bytes);
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

template <DdlaBackend Backend, typename T>
void pgetrs(
    const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
    T* A, const int* array_descA,
    const int* ipiv, // host
    T* B, const int* array_descB
)
{
    // 1. parameter check (collective).
    lu_impl::pgetrs_check<Backend>(handle, side, trans, n, nrhs, array_descA, array_descB);
    // No-op: nothing to solve for (B is left as it is).
    if (n == 0 || nrhs == 0) return;
    // 2. the scratch of the pivoting and the triangular solves.
    std::size_t backend_bytes = 0, host_bytes = 0;
    lu_impl::pgetrs_workspace_size<T>(handle, side, trans, n, nrhs, array_descA, array_descB,
                                      &backend_bytes, &host_bytes);
    vector<char> backend_work(handle, backend_bytes);
    std::vector<char> host_work(host_bytes);
    // 3. the solve.
    lu_impl::pgetrs_run<Backend, T>(handle, side, trans, n, nrhs, A, array_descA, ipiv, B, array_descB,
                                    backend_work.data(), backend_bytes, host_work.data(), host_bytes);
}

#define INSTANTIATE_PGETRS(BACKEND, TYPE)                                   \
    template void pgetrs<BACKEND, TYPE>(const DdlaHandle_t&, char side, char trans, int n, int nrhs, \
    TYPE* A, const int* array_descA, \
    const int* ipiv, \
    TYPE* B, const int* array_descB); \
    template void lu_impl::pgetrs_run<BACKEND, TYPE>(const DdlaHandle_t&, char, char, int, int, TYPE*, const int*, \
    const int*, TYPE*, const int*, void*, std::size_t, void*, std::size_t)

#define INSTANTIATE_PGETRS_SIZE(TYPE) \
    template void lu_impl::pgetrs_workspace_size<TYPE>(const DdlaHandle_t&, char, char, int, int, const int*, \
    const int*, std::size_t*, std::size_t*)

#define INSTANTIATE_PGETRS_CHECK(BACKEND) \
    template void lu_impl::pgetrs_check<BACKEND>(const DdlaHandle_t&, char, char, int, int, const int*, const int*)


#if DDLA_HAS_CPU
INSTANTIATE_PGETRS_CHECK(DdlaBackend::CPU);
INSTANTIATE_PGETRS(DdlaBackend::CPU, float);
INSTANTIATE_PGETRS(DdlaBackend::CPU, double);
INSTANTIATE_PGETRS(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PGETRS(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_PGETRS_CHECK(DdlaBackend::GPU);
INSTANTIATE_PGETRS(DdlaBackend::GPU, float);
INSTANTIATE_PGETRS(DdlaBackend::GPU, double);
INSTANTIATE_PGETRS(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PGETRS(DdlaBackend::GPU, std::complex<double>);
#endif

INSTANTIATE_PGETRS_SIZE(float);
INSTANTIATE_PGETRS_SIZE(double);
INSTANTIATE_PGETRS_SIZE(std::complex<float>);
INSTANTIATE_PGETRS_SIZE(std::complex<double>);

#undef INSTANTIATE_PGETRS_SIZE
#undef INSTANTIATE_PGETRS_CHECK
#undef INSTANTIATE_PGETRS

} // namespace ddla
