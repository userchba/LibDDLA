#ifndef EIG_TEST_COMMON_H
#define EIG_TEST_COMMON_H

// ---------------------------------------------------------------------------
// Verification helpers for the two-stage Hermitian eigensolver tests.
//
// The manufactured-solution checks in api_grid_test_common.h do not apply to
// an eigensolver; the standard LAPACK-style metrics are used instead:
//
//   residual        ||A V - V Lambda||_F / (n ||A||_F eps)      accept < 50
//   orthogonality   ||V^H V - I||_F   / (n eps)                 accept < 50
//   eigenvalues     max|lambda_i - ref_i| / (||A||_F eps)        (reported)
//
// All helpers assume the descriptor describes exactly the n x n leading
// problem (mb == nb, anchored at (0,0)), which is how the eigensolver tests
// build their inputs.  Communication-free by construction: local errors are
// reduced with a single MPI max at the end.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <vector>

#include <mpi.h>

#include "api_grid_test_common.h"

namespace eig_test {

using Complex = std::complex<double>;

template <typename T> struct eps_trait;
template <> struct eps_trait<float>  { static constexpr double value = 1.19e-7; };
template <> struct eps_trait<double> { static constexpr double value = 2.22e-16; };

/// Distributed Frobenius norm of the leading n x n block described by @p desc.
///
/// The distributed buffer is downloaded once and the norm is accumulated on
/// the host; reading a device pointer directly (as an earlier revision did)
/// is wrong for every GPU backend.
template <typename T>
double frobenius_norm(const ddla::DdlaHandle_t& handle, const T* d_A, const int* desc, int n)
{
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
    const int m_loc = ddla::num_loc(n, desc[ddla::DDLA_MB_], myprow, desc[ddla::DDLA_RSRC_], nprows);
    const int n_loc = ddla::num_loc(n, desc[ddla::DDLA_NB_], mypcol, desc[ddla::DDLA_CSRC_], npcols);
    const int lld = desc[ddla::DDLA_LLD_];
    // GEMM initializes only the logical rectangle; descriptor padding may be
    // uninitialized. Download active entries without reading padding or the
    // dummy storage of ranks with an empty local extent.
    std::vector<T> local(static_cast<std::size_t>(m_loc) * n_loc);
    if (m_loc > 0 && n_loc > 0)
        ddla::DDLA_CHECK(ddla::ddlaMemcpy2D(handle, local.data(), static_cast<std::size_t>(m_loc) * sizeof(T),
                                          d_A, static_cast<std::size_t>(lld) * sizeof(T),
                                          static_cast<std::size_t>(m_loc) * sizeof(T), n_loc));
    api_grid_test::check_ddla_sync(handle);
    double local2 = 0.0;
    for (int j = 0; j < n_loc; ++j) {
        const int gj = ddla::indx_l2g_c(desc, handle, j);
        if (gj >= n) continue;
        for (int i = 0; i < m_loc; ++i) {
            const int gi = ddla::indx_l2g_r(desc, handle, i);
            if (gi >= n) continue;
            const double a = std::abs(local[i + static_cast<std::size_t>(j) * m_loc]);
            local2 += a * a;
        }
    }
    double global2 = 0.0;
    MPI_Allreduce(&local2, &global2, 1, MPI_DOUBLE, MPI_SUM, ddlaGetCommunicator(handle));
    return std::sqrt(global2);
}

/// Tolerance helper: the accepted bound `threshold * n * eps` for the
/// metrics below, in the units those metrics are expressed in.  `threshold`
/// defaults to the `< 50` convention documented at the top of this file.
template <typename T>
constexpr double eig_tolerance(int n, double threshold = 50.0)
{
    return threshold * static_cast<double>(n) * eps_trait<T>::value;
}

/// Residual metric  ||A V - V Lambda||_F / (n ||A||_F eps)  for jobz='V'.
/// Uses a workspace copy of Z; the input Z is left untouched.
template <typename T, ddla::DdlaBackend Backend = ddla::default_backend_v>
double residual_metric(const ddla::DdlaHandle_t& handle,
                       const T* d_A, const int* descA,
                       const T* d_Z, const int* descZ,
                       const typename ddla::real_type<T>::type* w, int n)
{
    using rt = typename ddla::real_type<T>::type;
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
    const int m_loc = ddla::num_loc(n, descZ[ddla::DDLA_MB_], myprow, descZ[ddla::DDLA_RSRC_], nprows);
    const int n_loc = ddla::num_loc(n, descZ[ddla::DDLA_NB_], mypcol, descZ[ddla::DDLA_CSRC_], npcols);
    const int lld = descZ[ddla::DDLA_LLD_];

    // AZ = A * Z (pgemm with the Z descriptor for both operands and output).
    api_grid_test::DeviceBuffer<T> d_AZ(handle, api_grid_test::local_size(handle, descZ));
    const T zero{};
    ddla::pgemm<Backend, T>(handle, 'N', 'N', n, n, n,
                   T(1.0), d_A, descA, d_Z, descZ, zero, d_AZ.ptr, descZ);
    // AZ -= Z * diag(w): subtract w[global column] * Z[:, j] from the
    // matching local column of AZ.
    for (int j = 0; j < n_loc; ++j) {
        const int gj = ddla::indx_l2g_c(descZ, handle, j);
        if (gj >= n) continue;
        const T alpha = T(-static_cast<rt>(w[gj]));
        ddla::axpy<Backend, T>(handle, m_loc, alpha,
                   d_Z + j * static_cast<std::ptrdiff_t>(lld), 1,
                   d_AZ.ptr + j * static_cast<std::ptrdiff_t>(lld), 1);
    }
    api_grid_test::check_ddla_sync(handle);

    const double norm_A = frobenius_norm(handle, d_A, descA, n);
    const double norm_R = frobenius_norm(handle, d_AZ.ptr, descZ, n);
    const double eps = eps_trait<rt>::value;
    return (norm_A > 0.0) ? norm_R / (static_cast<double>(n) * norm_A * eps) : 0.0;
}

/// Orthogonality metric  ||V^H V - I||_F / (n eps).  Builds G = Z^H Z in a
/// small distributed buffer and takes the max deviation from the identity.
template <typename T, ddla::DdlaBackend Backend = ddla::default_backend_v>
double orthogonality_metric(const ddla::DdlaHandle_t& handle,
                            const T* d_Z, const int* descZ, int n)
{
    using rt = typename ddla::real_type<T>::type;
    int descG[ddla::DDLA_DLEN_];
    ddlaDescInit(descG, handle, n, n,
                 descZ[ddla::DDLA_MB_], descZ[ddla::DDLA_NB_],
                 descZ[ddla::DDLA_RSRC_], descZ[ddla::DDLA_CSRC_]);
    api_grid_test::DeviceBuffer<T> d_G(handle, api_grid_test::local_size(handle, descG));
    const T zero{};
    ddla::pgemm<Backend, T>(handle, 'C', 'N', n, n, n,
                   T(1.0), d_Z, descZ, d_Z, descZ, zero, d_G.ptr, descG);
    api_grid_test::check_ddla_sync(handle);

    // Local max |G(i,j) - delta_ij| over the owned part of the leading block.
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
    const int m_loc = ddla::num_loc(n, descG[ddla::DDLA_MB_], myprow, descG[ddla::DDLA_RSRC_], nprows);
    const int n_loc = ddla::num_loc(n, descG[ddla::DDLA_NB_], mypcol, descG[ddla::DDLA_CSRC_], npcols);
    const int lld = descG[ddla::DDLA_LLD_];
    // The Gram matrix also has uninitialized dummy rows when m_loc == 0.
    std::vector<T> hostG(static_cast<std::size_t>(m_loc) * n_loc);
    if (m_loc > 0 && n_loc > 0)
        ddla::DDLA_CHECK(ddla::ddlaMemcpy2D(handle, hostG.data(), static_cast<std::size_t>(m_loc) * sizeof(T),
                                          d_G.ptr, static_cast<std::size_t>(lld) * sizeof(T),
                                          static_cast<std::size_t>(m_loc) * sizeof(T), n_loc));
    api_grid_test::check_ddla_sync(handle);
    double local_err = 0.0;
    for (int j = 0; j < n_loc; ++j) {
        const int gj = ddla::indx_l2g_c(descG, handle, j);
        if (gj >= n) continue;
        for (int i = 0; i < m_loc; ++i) {
            const int gi = ddla::indx_l2g_r(descG, handle, i);
            if (gi >= n) continue;
            const T expected = (gi == gj) ? T(1.0) : T(0.0);
            local_err = std::max(local_err,
                                 static_cast<double>(std::abs(hostG[i + static_cast<std::size_t>(j) * m_loc] - expected)));
        }
    }
    double global_err = 0.0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, ddlaGetCommunicator(handle));
    const double eps = eps_trait<rt>::value;
    return global_err / (static_cast<double>(n) * eps);
}

/// Eigenvalue agreement  max_i |w_i - w_ref_i| / (||A||_F eps); w is a
/// replicated host array so no communication is needed.
template <typename T>
double eigenvalue_metric(const ddla::DdlaHandle_t& handle,
                         const typename ddla::real_type<T>::type* w,
                         const typename ddla::real_type<T>::type* w_ref,
                         int n, double norm_A)
{
    using rt = typename ddla::real_type<T>::type;
    rt local_max = 0;
    for (int i = 0; i < n; ++i)
        local_max = std::max(local_max, static_cast<rt>(std::abs(w[i] - w_ref[i])));
    const double eps = eps_trait<rt>::value;
    return (norm_A > 0.0) ? static_cast<double>(local_max) / (norm_A * eps) : 0.0;
}

} // namespace eig_test

#endif // EIG_TEST_COMMON_H
