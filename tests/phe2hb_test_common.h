#ifndef DDLA_PHE2HB_TEST_COMMON_H
#define DDLA_PHE2HB_TEST_COMMON_H

// Helpers shared by test_api_grid_phe2hb and test_phe2hb_upper_cpu: the gather of a distributed matrix and the host
// reconstruction of the stage-1 reflectors and band.

#include <complex>
#include <cstring>
#include <type_traits>
#include <vector>

#include "api_grid_test_common.h"
#include "Eigensolver/eig_params.h"
#include "eig_test_common.h"

using namespace api_grid_test;

namespace {

template <typename U> U conj_v2(U v) { return v; }
template <typename U> std::complex<U> conj_v2(std::complex<U> v) { return std::conj(v); }

// Gather a distributed matrix's leading n x n block: every rank ends up with
// the full block (zero-fill + MPI_Allreduce).  The reduction datatype is
// selected from the scalar type rather than derived from sizeof(T), which
// integer-divided to 0 for single precision.
template <typename T>
std::vector<T> gather_full(const ddla::DdlaHandle_t& handle, const T* d_A,
                           const int* desc, int n)
{
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
    const int m_loc = ddla::num_loc(n, desc[ddla::DDLA_MB_], myprow, desc[ddla::DDLA_RSRC_], nprows);
    const int n_loc = ddla::num_loc(n, desc[ddla::DDLA_NB_], mypcol, desc[ddla::DDLA_CSRC_], npcols);
    const int lld = desc[ddla::DDLA_LLD_];
    const auto local = download<T>(handle, d_A, local_size(handle, desc));
    std::vector<T> full(static_cast<std::size_t>(n) * n, T(0));
    for (int j = 0; j < n_loc; ++j) {
        const int gj = ddla::indx_l2g_c(desc, handle, j);
        if (gj >= n) continue;
        for (int i = 0; i < m_loc; ++i) {
            const int gi = ddla::indx_l2g_r(desc, handle, i);
            if (gi >= n) continue;
            full[gi + static_cast<std::size_t>(gj) * n] =
                local[i + static_cast<std::size_t>(j) * lld];
        }
    }
    MPI_Comm comm = ddlaGetCommunicator(handle);
    if constexpr (std::is_same<T, float>::value) {
        MPI_Allreduce(MPI_IN_PLACE, full.data(), static_cast<int>(full.size()),
                      MPI_FLOAT, MPI_SUM, comm);
    } else if constexpr (std::is_same<T, double>::value) {
        MPI_Allreduce(MPI_IN_PLACE, full.data(), static_cast<int>(full.size()),
                      MPI_DOUBLE, MPI_SUM, comm);
    } else if constexpr (std::is_same<T, std::complex<float>>::value) {
        MPI_Allreduce(MPI_IN_PLACE, full.data(), static_cast<int>(full.size()),
                      MPI_C_FLOAT_COMPLEX, MPI_SUM, comm);
    } else {
        MPI_Allreduce(MPI_IN_PLACE, full.data(), static_cast<int>(full.size()),
                      MPI_C_DOUBLE_COMPLEX, MPI_SUM, comm);
    }
    return full;
}

// Host reconstruction of Q from the stored reflectors: apply them to the
// identity, innermost transform first.
template <typename T>
std::vector<T> build_host_Q(const std::vector<T>& Aout, const std::vector<T>& tmat,
                            int n, int nbw, int nsteps)
{
    auto idx = [n](int i, int j) { return i + static_cast<std::size_t>(j) * n; };
    std::vector<T> Q(static_cast<std::size_t>(n) * n, T(0)), x(n);
    for (int jcol = 0; jcol < n; ++jcol) {
        std::fill(x.begin(), x.end(), T(0));
        x[jcol] = T(1);
        for (int istep = nsteps; istep >= 1; --istep) {
            const int panel_start = istep * nbw;
            const int n_cols = std::min(n, panel_start + nbw) - panel_start;
            for (int lc = n_cols - 1; lc >= 0; --lc) {
                const int j = panel_start + lc;
                const int nrow = j - nbw;
                const T tau = tmat[static_cast<std::size_t>(lc) + lc * nbw
                                   + static_cast<std::size_t>(istep - 1) * nbw * nbw];
                if (tau == T(0)) continue;
                // v = stored vector + implicit anchor 1 at nrow
                T dot{};
                for (int i = 0; i < nrow; ++i) dot += conj_v2(Aout[idx(i, j)]) * x[i];
                dot += x[nrow];   // anchor
                for (int i = 0; i < nrow; ++i) x[i] -= tau * dot * Aout[idx(i, j)];
                x[nrow] -= tau * dot;
            }
        }
        for (int i = 0; i < n; ++i) Q[idx(i, jcol)] = x[i];
    }
    return Q;
}

// Dense host products for the small test sizes.
template <typename T>
std::vector<T> host_mul(const std::vector<T>& A, const std::vector<T>& B, int n)
{
    auto idx = [n](int i, int j) { return i + static_cast<std::size_t>(j) * n; };
    std::vector<T> C(static_cast<std::size_t>(n) * n, T(0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            T acc{};
            for (int k = 0; k < n; ++k) acc += A[idx(i, k)] * B[idx(k, j)];
            C[idx(i, j)] = acc;
        }
    return C;
}

/// C = A^H * B.
template <typename T>
std::vector<T> host_mul_conjA(const std::vector<T>& A, const std::vector<T>& B, int n)
{
    auto idx = [n](int i, int j) { return i + static_cast<std::size_t>(j) * n; };
    std::vector<T> C(static_cast<std::size_t>(n) * n, T(0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            T acc{};
            for (int k = 0; k < n; ++k) acc += conj_v2(A[idx(k, i)]) * B[idx(k, j)];
            C[idx(i, j)] = acc;
        }
    return C;
}

/// C = A * B^H.
template <typename T>
std::vector<T> host_mul_conjB(const std::vector<T>& A, const std::vector<T>& B, int n)
{
    auto idx = [n](int i, int j) { return i + static_cast<std::size_t>(j) * n; };
    std::vector<T> C(static_cast<std::size_t>(n) * n, T(0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            T acc{};
            for (int k = 0; k < n; ++k) acc += A[idx(i, k)] * conj_v2(B[idx(j, k)]);
            C[idx(i, j)] = acc;
        }
    return C;
}

/// Conjugate transpose of a dense host matrix.
template <typename T>
std::vector<T> host_adjoint(const std::vector<T>& A, int n)
{
    auto idx = [n](int i, int j) { return i + static_cast<std::size_t>(j) * n; };
    std::vector<T> AH(static_cast<std::size_t>(n) * n, T(0));
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) AH[idx(i, j)] = conj_v2(A[idx(j, i)]);
    return AH;
}

} // namespace

#endif // DDLA_PHE2HB_TEST_COMMON_H
