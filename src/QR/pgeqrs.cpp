#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>

#include "check_agree.h"
#include "ddla_desc.h"
#include "helpers.h"
#include "mpi_datatype.h"
#include "require_backend.h"
#include "PBLAS/ptrtrs_impl.h"
#include "SVD/svd_workspace.h"
#include "instantiate.h"

#include <algorithm>
#include <complex>
#include <vector>

namespace ddla {

template <DdlaBackend Backend, typename T>
void pgeqrs(const DdlaHandle_t& handle, int m, int n, int nrhs,
            const T* A, const int* descA, const T* tau,
            T* B, const int* descB, int& info)
{
    info = 0;
    detail::agree_check(handle, "pgeqrs", [&](detail::Agree&) {
        detail::require_backend<Backend>(handle, "pgeqrs");
        check_desc(descA, handle);
        check_desc(descB, handle);
        if (m < 0 || n < 0 || n > m || nrhs < 0)
            throw std::invalid_argument("pgeqrs: require m >= n >= 0 and nrhs >= 0");
        if (m > descA[DDLA_M_] || n > descA[DDLA_N_] ||
            m > descB[DDLA_M_] || nrhs > descB[DDLA_N_])
            throw std::invalid_argument("pgeqrs: dimensions exceed the descriptors");
        if (descA[DDLA_MB_] != descA[DDLA_NB_] ||
            descA[DDLA_MB_] != descB[DDLA_MB_] ||
            descA[DDLA_RSRC_] != descB[DDLA_RSRC_])
            throw std::invalid_argument("pgeqrs: A needs square blocks and B needs A's row distribution");
        if (n == 0 || nrhs == 0) return;

        const int ar = num_loc(m, descA[DDLA_MB_], handle->myprow_, descA[DDLA_RSRC_], handle->nprows_);
        const int ac = num_loc(n, descA[DDLA_NB_], handle->mypcol_, descA[DDLA_CSRC_], handle->npcols_);
        const int bc = num_loc(nrhs, descB[DDLA_NB_], handle->mypcol_, descB[DDLA_CSRC_], handle->npcols_);
        if ((ar && ac && !A) || (ac && !tau) || (ar && bc && !B))
            throw std::invalid_argument("pgeqrs: null active input or output");
        if (detail::matrix_overlaps(A, descA[DDLA_LLD_], ar, ac,
                                    B, descB[DDLA_LLD_], ar, bc) ||
            detail::matrix_overlaps(tau, std::max(1, ac), ac, 1,
                                    B, descB[DDLA_LLD_], ar, bc))
            throw std::invalid_argument("pgeqrs: B must be disjoint from A and tau");
    });
    int signature[17] = {m, n, nrhs};
    detail::descriptor_signature(signature + 3, descA);
    detail::descriptor_signature(signature + 10, descB);
    detail::agree_signature(handle, "pgeqrs", signature);
    if (n == 0 || nrhs == 0) return;

    // For m >= n the Q-left punmbr path uses QR's columns, unit diagonal,
    // and forward reflector product. Its tau input is replicated on the host.
    const auto apply = spectral_impl::punmbr_geometry(handle, Backend, 'Q', 'L', 'C',
                                                     m, nrhs, n, descA, descB);
    std::size_t apply_backend = 0, apply_host = 0, solve_backend = 0, solve_host = 0;
    spectral_impl::punmbr_workspace_size<T>(handle, apply, &apply_backend, &apply_host);
    ptrtrs_impl::ptrtrs_workspace_size<T>(handle, 'L', 'N', n, nrhs, descA, descB,
                                          &solve_backend, &solve_host);
    const std::size_t backend_bytes = std::max(apply_backend, solve_backend);
    const std::size_t host_bytes = std::max(apply_host, solve_host);
    vector<char> backend_work(handle);
    std::vector<char> host_work;
    std::vector<T> full_tau;
    detail::agree_check(handle, "pgeqrs workspace", [&](detail::Agree&) {
        backend_work.resize(backend_bytes);
        host_work.resize(host_bytes);
        full_tau.resize(n, T(0));
    });
    for (int j = 0; j < n; ++j) {
        if (indxg2p(j, descA[DDLA_NB_], descA[DDLA_CSRC_], handle->npcols_) == handle->mypcol_)
            full_tau[j] = tau[indxg2l(j, descA[DDLA_NB_], handle->npcols_)];
    }
    // Each process row owns one copy of every tau entry, split over its columns.
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, full_tau.data(), n, detail::mpi_datatype<T>(),
                             MPI_SUM, handle->row_comm));
    spectral_impl::punmbr_run<Backend, T>(handle, apply, A, descA, full_tau.data(), B, descB,
                                         info, backend_work.data(), backend_bytes,
                                         host_work.data(), host_bytes);
    // ptrtrs reads A, although its existing interface accepts a mutable pointer.
    ptrtrs_impl::ptrtrs_run<Backend, T>(handle, 'L', 'U', 'N', 'N', n, nrhs,
                                       const_cast<T*>(A), descA, B, descB,
                                       backend_work.data(), backend_bytes,
                                       host_work.data(), host_bytes);
}

#define INSTANTIATE_PGEQRS(BACKEND, TYPE) \
    template void pgeqrs<BACKEND, TYPE>(const DdlaHandle_t&, int, int, int, \
        const TYPE*, const int*, const TYPE*, TYPE*, const int*, int&)

DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_PGEQRS)
#undef INSTANTIATE_PGEQRS

} // namespace ddla
