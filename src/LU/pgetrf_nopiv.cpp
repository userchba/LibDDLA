#include "BLAS/blas_impl.h"
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "lu_internal.h"
#include "ddla_desc.h"
#include <cassert>
#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "require_backend.h"
#include "comm_traits.h"
#include "check_agree.h"
#include "plan_util.h"
#include "lapack_impl.h"
#include "pgetrf_impl.h"
#include "instantiate.h"

namespace ddla {

namespace lu_impl {

// pgetrf_nopiv's three components: the block LU of pgetrf_bpiv.cpp without pivoting.
template <DdlaBackend Backend>
void pgetrf_nopiv_check(const DdlaHandle_t& handle, int m, int n, const int* array_descA)
{
    pgetrf_block_check<Backend>(handle, "pgetrf_nopiv", m, n, array_descA);
}

// [ the diagonal block's info word | the diagonal block (nb x nb), which the L panel (local rows x nb) reuses after
// the step's last trsm | the U panel (nb x local columns) ]; no host scratch.
template <typename T>
void pgetrf_nopiv_workspace_size(const DdlaHandle_t& handle, int m, int n, const int* array_descA,
                                 std::size_t* backend_bytes)
{
    if (handle->nprocs == 1) {
        *backend_bytes=detail::padded_bytes<int>(1);
        return;
    }
    std::size_t host_bytes = 0;
    pgetrf_block_workspace_size<T>(handle, false, m, n, nullptr, array_descA, backend_bytes, &host_bytes);
}

template <DdlaBackend Backend, typename T>
void pgetrf_nopiv_run(const DdlaHandle_t& handle, int m, int n, T* A, const int* array_descA, int& info,
                      void* backend_work, std::size_t backend_bytes)
{
    if (handle->nprocs == 1) {
        info=0;
        if (m<=0 || n<=0) return;
        int* status=static_cast<int*>(backend_work);
        lapack_impl::getrf_nopiv_run<Backend,T>(handle,m,n,A,array_descA[DDLA_LLD_],status);
        DDLA_CHECK(ddlaMemcpy(handle,&info,status,sizeof(int)));
        DDLA_CHECK(ddlaSynchronize(handle));
        return;
    }
    pgetrf_block_run<Backend, T>(handle, false, m, n, A, array_descA, nullptr, info,
                                 backend_work, backend_bytes, nullptr, 0);
}

} // namespace lu_impl

/**
 * @brief Distributed LU factorization without pivoting.
 *
 * Right-looking block algorithm: at each step the nb*nb diagonal block is
 * factored with the local getrf_nopiv (single-GPU, no pivoting).  The U and
 * L panels are then computed via trsm and the trailing submatrix is updated
 * via gemm, exactly as in pgetrf_bpiv but without any row swaps.
 *
 * For each panel step k (global column n_s):
 *   1) Factor diagonal block: A11 = L1 * U1          (getrf_nopiv)
 *   2) Extract/broadcast the factored diagonal block (L1+U1)
 *   3) Compute U12:  U12 = L1^{-1} * B             (trsm LEFT/LOWER/UNIT)
 *   4) Compute L21:  L21 = C * U1^{-1}             (trsm RIGHT/UPPER/NON-UNIT)
 *   5) Broadcast U panel down column, L panel across row
 *   6) Update trailing: D <- D - L21 * U12         (gemm)
 */
template <DdlaBackend Backend, typename T>
void pgetrf_nopiv(
    const DdlaHandle_t& handle, int m, int n,
    T* A, const int* array_descA,
    int& info  // host
)
{
    // 1. parameter check (collective).
    lu_impl::pgetrf_nopiv_check<Backend>(handle, m, n, array_descA);
    // No-op: an empty factorization.
    info = 0;
    if (m <= 0 || n <= 0) return;
    // 2. the scratch.
    std::size_t backend_bytes = 0;
    lu_impl::pgetrf_nopiv_workspace_size<T>(handle, m, n, array_descA, &backend_bytes);
    vector<char> backend_work(handle, backend_bytes);
    // 3. the factorization.
    lu_impl::pgetrf_nopiv_run<Backend, T>(handle, m, n, A, array_descA, info, backend_work.data(), backend_bytes);
}

// Explicit instantiations
#define INSTANTIATE_PGETRF_NOPIV(BACKEND, TYPE)                                   \
    template void pgetrf_nopiv<BACKEND, TYPE>(const DdlaHandle_t&, int m, int n, \
    TYPE* A, const int* array_descA, \
    int& info); \
    template void lu_impl::pgetrf_nopiv_run<BACKEND, TYPE>(const DdlaHandle_t&, int, int, TYPE*, const int*, int&, \
    void*, std::size_t)

#define INSTANTIATE_PGETRF_NOPIV_SIZE(TYPE) \
    template void lu_impl::pgetrf_nopiv_workspace_size<TYPE>(const DdlaHandle_t&, int, int, const int*, std::size_t*)

#define INSTANTIATE_PGETRF_NOPIV_CHECK(BACKEND) \
    template void lu_impl::pgetrf_nopiv_check<BACKEND>(const DdlaHandle_t&, int, int, const int*)


DDLA_FOR_EACH_BACKEND(INSTANTIATE_PGETRF_NOPIV_CHECK)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_PGETRF_NOPIV)
DDLA_FOR_EACH_TYPE(INSTANTIATE_PGETRF_NOPIV_SIZE)

#undef INSTANTIATE_PGETRF_NOPIV_SIZE
#undef INSTANTIATE_PGETRF_NOPIV_CHECK
#undef INSTANTIATE_PGETRF_NOPIV

} // namespace ddla
