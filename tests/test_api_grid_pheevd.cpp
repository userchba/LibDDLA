#include "Eigensolver/pstedc_impl.h"
#include "row_share.h"
#include "mpi_datatype.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>
#include <cstdlib>
#include <cstdio>

#include "api_grid_test_common.h"
#include "Eigensolver/eig_params.h"
#include "eig_test_common.h"

// ---------------------------------------------------------------------------
// End-to-end test for the two-stage Hermitian eigensolver driver (pheevd).
//
// Per (grid, size, type): build a full Hermitian input, run pheevd, then
// check the standard LAPACK-style metrics from eig_test_common.h:
//   residual        ||A V - V Lambda||_F / (n ||A||_F eps)          < 50
//   orthogonality   ||V^H V - I||_F / (n eps)                      < 50
//   eigenvalues     max|w - w_ref| / (||A||_F eps)                 < 1e4
// The eigenvalue reference comes from the host ?SYEVD/?HEEVD on the
// gathered input (every rank computes it redundantly).  jobz='N' is
// checked for eigenvalue agreement only.  The matrix handed to pheevd holds
// ONLY the uplo triangle (the opposite strict triangle and, for complex
// types, the imaginary part of the diagonal are poison): the ScaLAPACK
// contract says they are not referenced.
//
// The eigenvalue bound is looser than the residual/orthogonality bound
// because the distributed tridiagonal solve plus the two back-transforms
// accumulate over ~n*nbw/2 rotations; observed values are O(1e2).
//
// check_pstedc_ldz additionally pins stage 3's output layout: pstedc with a
// leading dimension ldz > n must fill only the n x n block (padding rows
// untouched) and give the same answer as ldz == n.
// ---------------------------------------------------------------------------

using namespace api_grid_test;

extern "C" {
void ssyevd_(const char*, const char*, const int*, float*, const int*,
             float*, float*, const int*, int*, const int*, int*);
void dsyevd_(const char*, const char*, const int*, double*, const int*,
             double*, double*, const int*, int*, const int*, int*);
void cheevd_(const char*, const char*, const int*, std::complex<float>*, const int*,
             float*, std::complex<float>*, const int*, float*, const int*,
             int*, const int*, int*);
void zheevd_(const char*, const char*, const int*, std::complex<double>*, const int*,
             double*, std::complex<double>*, const int*, double*, const int*,
             int*, const int*, int*);
void dsterf_(const int*, double*, double*, int*);
}

namespace {

template <typename U> U conj_v2(U v) { return v; }
template <typename U> std::complex<U> conj_v2(std::complex<U> v) { return std::conj(v); }

template <typename T>
T herm_input(int i, int j)
{
    using rt = typename ddla::real_type<T>::type;
    const double re = 0.3 * ((i * 7 + j * 13) % 11) - 1.4 + 0.05 * ((i + j) % 3);
    const double im = 0.15 * ((i * 5 + j * 3) % 7) - 0.4;
    if (i == j) return T(static_cast<rt>(3.0 + 0.1 * i));
    if constexpr (std::is_same_v<T, rt>) {
        (void)im;
        if (i > j) return T(static_cast<rt>(re));
        return herm_input<T>(j, i);
    } else if (i > j) {
        return T(static_cast<rt>(re), static_cast<rt>(im));
    } else {
        return conj_v2(herm_input<T>(j, i));
    }
}

// Finite, entry-dependent poison for the triangle a uplo call must not read.
template <typename T>
T poison_value(int i, int j, double mag)
{
    using rt = typename ddla::real_type<T>::type;
    const double re = mag * (1.0 + ((i * 3 + j) % 5));
    if constexpr (std::is_same_v<T, rt>) return T(static_cast<rt>(re));
    else return T(static_cast<rt>(re), static_cast<rt>(-0.5 * re));
}

// The ScaLAPACK input of a uplo call: the stored triangle of the full Hermitian matrix, with the
// opposite strict triangle AND (complex) the imaginary part of the diagonal replaced by poison.  A
// solver that reads anything it must not read sees eigenvalue errors of order `mag`.  (2026-10-15:
// the previous uplo='L' leg zeroed the upper triangle and then rebuilt the reference's lower one
// FROM that zeroed triangle, so solver and reference both solved a diagonal matrix -- and the
// driver read the wrong triangle without anyone noticing.)
template <typename T>
std::vector<T> poisoned_input(const std::vector<T>& full, int n, bool upper, double mag)
{
    using rt = typename ddla::real_type<T>::type;
    std::vector<T> a(full);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const std::size_t k = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * n;
            if (upper ? i > j : i < j) {
                a[k] = poison_value<T>(i, j, mag);
            } else if (i == j) {
                if constexpr (!std::is_same_v<T, rt>)   // complex: real diagonal, poisoned imaginary part
                    a[k] = T(std::real(full[k]), std::imag(poison_value<T>(i, j, mag)));
            }
        }
    }
    return a;
}

template <typename T>
std::vector<typename ddla::real_type<T>::type> host_eigh(std::vector<T> a, int n)
{
    using rt = typename ddla::real_type<T>::type;
    // the ?SYEVD/?HEEVD workspaces are real/int arrays regardless of T
    std::vector<rt> w(n), rwork(1);
    std::vector<T> work(1);
    std::vector<int> iwork(1);
    int info = 0, lwork = -1, lrwork = -1, liwork = -1;
    if constexpr (std::is_same_v<T, float>) {
        ssyevd_("N", "U", &n, a.data(), &n, w.data(), work.data(), &lwork,
                iwork.data(), &liwork, &info);
    } else if constexpr (std::is_same_v<T, double>) {
        dsyevd_("N", "U", &n, a.data(), &n, w.data(), work.data(), &lwork,
                iwork.data(), &liwork, &info);
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        cheevd_("N", "U", &n, a.data(), &n, w.data(), work.data(), &lwork,
                rwork.data(), &lrwork, iwork.data(), &liwork, &info);
    } else {
        zheevd_("N", "U", &n, a.data(), &n, w.data(), work.data(), &lwork,
                rwork.data(), &lrwork, iwork.data(), &liwork, &info);
    }
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
        lwork = static_cast<int>(work[0]) + 64;
    else
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
        lwork = static_cast<int>(work[0]) + 64;
    else
        lwork = static_cast<int>(work[0].real()) + 64;
    liwork = std::max(iwork[0], 1);
    work.assign(static_cast<std::size_t>(lwork), rt(0));
    iwork.assign(static_cast<std::size_t>(liwork), 0);
    if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, double>) {
        lrwork = std::max(1, 3 * n - 2);
        rwork.assign(static_cast<std::size_t>(lrwork), rt(0));
    }
    if constexpr (std::is_same_v<T, float>) {
        ssyevd_("N", "U", &n, a.data(), &n, w.data(), work.data(), &lwork,
                iwork.data(), &liwork, &info);
    } else if constexpr (std::is_same_v<T, double>) {
        dsyevd_("N", "U", &n, a.data(), &n, w.data(), work.data(), &lwork,
                iwork.data(), &liwork, &info);
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        cheevd_("N", "U", &n, a.data(), &n, w.data(), work.data(), &lwork,
                rwork.data(), &lrwork, iwork.data(), &liwork, &info);
    } else {
        zheevd_("N", "U", &n, a.data(), &n, w.data(), work.data(), &lwork,
                rwork.data(), &lrwork, iwork.data(), &liwork, &info);
    }
    return w;
}

template <typename T>
void check_pheevd(const ddla::DdlaHandle_t& handle, int n, int nblk, bool upper)
{
    using rt = typename ddla::real_type<T>::type;
    const char uplo = upper ? 'U' : 'L';

    int descA[ddla::DDLA_DLEN_];
    ddlaDescInit(descA, handle, n, n, nblk, nblk, 0, 0);
    int descZ[ddla::DDLA_DLEN_];
    ddlaDescInit(descZ, handle, n, n, nblk, nblk, 0, 0);
    api_grid_test::DeviceBuffer<T> d_A(handle, local_size(handle, descA));
    api_grid_test::DeviceBuffer<T> d_Z(handle, local_size(handle, descZ));

    // The reference matrix is the full Hermitian one.  The matrix handed to pheevd holds the uplo
    // triangle only: the opposite strict triangle and the imaginary diagonal are POISON (ScaLAPACK
    // contract -- they must not be read).
    std::vector<T> Ah_full(static_cast<std::size_t>(n) * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            Ah_full[i + static_cast<std::size_t>(j) * n] = herm_input<T>(i, j);
    const std::vector<T> Ah = poisoned_input<T>(Ah_full, n, upper, 1.0e6);
    const std::vector<rt> ref_eigs = host_eigh<T>(Ah_full, n);
    double normA = 0.0;
    for (const T v : Ah_full) normA = std::max(normA, static_cast<double>(std::abs(v)));

    {
        std::vector<T> local(local_size(handle, descA), T(0));
        int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
        ddlaGetGridDims(handle, nprows, npcols);
        ddlaGetGridCoords(handle, myprow, mypcol);
        const int m_loc = ddla::num_loc(n, nblk, myprow, 0, nprows);
        const int n_loc = ddla::num_loc(n, nblk, mypcol, 0, npcols);
        const int lld = descA[ddla::DDLA_LLD_];
        for (int jl = 0; jl < n_loc; ++jl) {
            const int gj = ddla::indxl2g(jl, nblk, mypcol, 0, npcols);
            if (gj >= n) continue;
            for (int il = 0; il < m_loc; ++il) {
                const int gi = ddla::indxl2g(il, nblk, myprow, 0, nprows);
                if (gi >= n) continue;
                local[il + static_cast<std::size_t>(jl) * lld] =
                    Ah[gi + static_cast<std::size_t>(gj) * n];
            }
        }
        upload<T>(handle, d_A.ptr, local);
    }
    check_ddla_sync(handle);

    std::vector<rt> w(n);
    int info = 0;
    ddla::pheevd(handle, 'V', uplo, n, d_A.ptr, descA, w.data(),
                 d_Z.ptr, descZ, info);
    require_close(handle, "pheevd info", info, 0.5);

    // eigenvalues
    {
        double dev = 0.0;
        for (int i = 0; i < n; ++i)
            dev = std::max(dev, std::fabs(static_cast<double>(w[i] - ref_eigs[i])));
        const double metric = dev / (normA * eig_test::eps_trait<rt>::value);
        // The metric accumulates the backward error of both reduction stages
        // (phe2hb panels plus the chase); that grows like sqrt(n*nbw) eps
        // rather than n eps, so the bound is 8 n-eps units -- observed values
        // sit near 1 for double and a few for float.
        require_close(handle, "pheevd eigenvalues", metric / n, 8.0);
    }
    // residual + orthogonality: the residual metric consumes A, which the
    // driver destroyed -- rebuild it in the descA distribution first
    {
        std::vector<T> local(local_size(handle, descA), T(0));
        int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
        ddlaGetGridDims(handle, nprows, npcols);
        ddlaGetGridCoords(handle, myprow, mypcol);
        const int m_loc = ddla::num_loc(n, nblk, myprow, 0, nprows);
        const int n_loc = ddla::num_loc(n, nblk, mypcol, 0, npcols);
        const int lld = descA[ddla::DDLA_LLD_];
        for (int jl = 0; jl < n_loc; ++jl) {
            const int gj = ddla::indxl2g(jl, nblk, mypcol, 0, npcols);
            if (gj >= n) continue;
            for (int il = 0; il < m_loc; ++il) {
                const int gi = ddla::indxl2g(il, nblk, myprow, 0, nprows);
                if (gi >= n) continue;
                local[il + static_cast<std::size_t>(jl) * lld] =
                    Ah_full[gi + static_cast<std::size_t>(gj) * n];
            }
        }
        upload<T>(handle, d_A.ptr, local);
    }
    check_ddla_sync(handle);
    {
        const double resid = eig_test::residual_metric<T>(handle, d_A.ptr, descA,
                                                          d_Z.ptr, descZ, w.data(), n);
        require_close(handle, "pheevd residual", resid, 50.0);
        const double ortho = eig_test::orthogonality_metric<T>(handle, d_Z.ptr, descZ, n);
        require_close(handle, "pheevd orthogonality", ortho, 50.0);
    }
}

// splitmix64 of (i, tag) -> uniform in [-0.5, 0.5).
double hash01(int i, int tag)
{
    std::uint64_t x = static_cast<std::uint64_t>(i + 1) * 0x9E3779B97F4A7C15ull
                    ^ static_cast<std::uint64_t>(tag) * 0xC2B2AE3D27D4EB4Full;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27; x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    return static_cast<double>(x >> 11) / 9007199254740992.0 - 0.5;
}

// pstedc returns only this rank's rows; a padded local leading dimension
// must not overwrite the caller's padding.
// A sentinel in the padding rows, the tridiagonal residual of the strided
// result, agreement with the compact local-stride solve, and a z == nullptr 'N' call
// cover each of those.  kind 0: generic spectrum; kind 1: repeated diagonal
// values and tiny couplings, so ?LAED8 deflates through Givens rotations
// that mix the children's columns (the split-GEMM column classes).
template <typename rt>
void check_pstedc_ldz(const ddla::DdlaHandle_t& handle, int n, int kind)
{
    const rt kSentinel = rt(12345);
    const int pad = 7;
    int P=1;MPI_Comm_size(ddla::ddlaGetCommunicator(handle),&P);
    const ddla::detail::RowShare share(n,P);
    const int nr=share.count(ddla::ddlaGetRank(handle)),lo=share.lo(ddla::ddlaGetRank(handle));
    const int ld=std::max(1,nr);
    const double eps = static_cast<double>(std::numeric_limits<rt>::epsilon());
    std::vector<rt> d0(n), e0(n, rt(0));
    double normT = 0.0;
    for (int i = 0; i < n; ++i) {
        if (kind == 0) {
            d0[i] = static_cast<rt>(hash01(i, 1));
            e0[i] = static_cast<rt>(hash01(i, 2));
        } else {
            d0[i] = static_cast<rt>(0.5 * (i % 4));
            e0[i] = static_cast<rt>(1e-3 * hash01(i, 3));
        }
        normT = std::max(normT, std::fabs(static_cast<double>(d0[i])) +
                                 2.0 * std::fabs(static_cast<double>(e0[i])));
    }
    const std::string tag = "pstedc ldz kind" + std::to_string(kind);

    auto solve = [&](int ldz, std::vector<rt>& w, std::vector<rt>& z) {
        const std::size_t count = static_cast<std::size_t>(ldz) * n;
        DeviceBuffer<rt> d_z(handle, count);
        upload<rt>(handle, d_z.ptr, std::vector<rt>(count, kSentinel));
        std::vector<rt> dd(d0), ee(e0);
        w.assign(n, rt(0));
        int info = 0;
        ddla::pstedc<rt>(handle, 'V', n, dd.data(), ee.data(), w.data(),
                         d_z.ptr, ldz, info);
        check_ddla_sync(handle);
        require_close(handle, tag + " info", info, 0.5);
        z = download<rt>(handle, d_z.ptr, count);
    };
    std::vector<rt> wA, zA, wB, zB;
    solve(ld, wA, zA);
    solve(ld + pad, wB, zB);

    // padding rows untouched, bit for bit
    double bad = 0.0;
    for (int j = 0; j < n; ++j)
        for (int r = nr; r < ld + pad; ++r)
            if (zB[static_cast<std::size_t>(j) * (ld + pad) + r] != kSentinel) bad += 1.0;
    require_close(handle, tag + " padding rows overwritten", bad, 0.5);

    // the strided result matches the packed one
    double zdev = 0.0, wdev = 0.0;
    for (int j = 0; j < n; ++j) {
        wdev = std::max(wdev, std::fabs(static_cast<double>(wB[j] - wA[j])));
        for (int r = 0; r < nr; ++r)
            zdev = std::max(zdev, std::fabs(static_cast<double>(
                zB[static_cast<std::size_t>(j) * (ld + pad) + r] -
                zA[static_cast<std::size_t>(j) * ld + r])));
    }
    require_close(handle, tag + " padded vs compact local z", zdev / (n * eps), 10.0);
    require_close(handle, tag + " padded vs compact local w", wdev / (normT * eps * n), 10.0);

    std::vector<rt> full(std::size_t(n)*n,rt(0));
    for(int j=0;j<n;++j)for(int r=0;r<nr;++r)
        full[lo+r+std::size_t(j)*n]=zB[r+std::size_t(j)*(ld+pad)];
    MPI_Allreduce(MPI_IN_PLACE,full.data(),n*n,ddla::detail::mpi_datatype<rt>(),MPI_SUM,ddla::ddlaGetCommunicator(handle));
    // T z_j = w_j z_j on the strided result (a layout bug is O(1) here)
    double resid = 0.0;
    for (int j = 0; j < n; ++j) {
        const rt* y = full.data() + static_cast<std::size_t>(j) * n;
        for (int i = 0; i < n; ++i) {
            double t = static_cast<double>(d0[i]) * y[i];
            if (i > 0) t += static_cast<double>(e0[i - 1]) * y[i - 1];
            if (i + 1 < n) t += static_cast<double>(e0[i]) * y[i + 1];
            resid = std::max(resid, std::fabs(t - static_cast<double>(wB[j]) * y[i]));
        }
    }
    require_close(handle, tag + " residual", resid / (normT * eps * n), 50.0);
}

// jobz='N' (2026-10-11): on a CPU handle the eigenvalue-only tree (first/last rows
// of the eigenvectors, no n x n matrix); the same host tree on GPU handles. Gates:
//  * eigenvalue error against an independent reference (LAPACK ?STERF in double,
//    or the analytic Toeplitz eigenvalues) in units of eps*|T| -- no n factor, which
//    in float would admit 7e-4|T| at n = 598 -- must stay within max(25, 4 x the
//    'V' solve's own error on the same input);
//  * the result is BIT-IDENTICAL under every CPU split-GEMM mode and with the
//    root-delta cache on and off (the split reorder must not leak into the rows
//    path; the recompute pass must reproduce the first pass exactly);
//  * zero matrix: exactly zero.
// kinds: 0 generic; 1 near-ties (repeated diagonal values, tiny couplings: Givens
// rotations, mixed columns); 2 sparse couplings (e nonzero at 1 of 3 indices, so
// lower-level merges have K == 0 and feed the next z); 3 diagonal (e == 0);
// 4 zero matrix; 5 Toeplitz (2, -1): large K.
template <typename rt>
void check_pstedc_n(const ddla::DdlaHandle_t& handle, int n, int kind)
{
    const double eps = static_cast<double>(std::numeric_limits<rt>::epsilon());
    std::vector<rt> d0(n), e0(n, rt(0));
    for (int i = 0; i < n; ++i) {
        switch (kind) {
        case 0: d0[i] = static_cast<rt>(hash01(i, 1)); e0[i] = static_cast<rt>(hash01(i, 2)); break;
        case 1: d0[i] = static_cast<rt>(0.5 * (i % 4)); e0[i] = static_cast<rt>(1e-3 * hash01(i, 3)); break;
        case 2: d0[i] = static_cast<rt>(hash01(i, 4));
                e0[i] = (i % 3 == 0) ? static_cast<rt>(hash01(i, 5)) : rt(0); break;
        case 3: d0[i] = static_cast<rt>(hash01(i, 6)); break;
        case 4: break;
        default: d0[i] = rt(2); e0[i] = (i + 1 < n) ? rt(-1) : rt(0); break;
        }
    }
    double normT = 0.0;
    for (int i = 0; i < n; ++i)
        normT = std::max(normT, std::fabs(static_cast<double>(d0[i])) +
                                 (i > 0 ? std::fabs(static_cast<double>(e0[i - 1])) : 0.0) +
                                 std::fabs(static_cast<double>(e0[i])));
    const std::string tag = "pstedc N kind" + std::to_string(kind) + " n=" + std::to_string(n);

    std::vector<double> ref(n);
    if (kind == 5) {
        const long double pi = 3.14159265358979323846264338327950288L;
        for (int k = 1; k <= n; ++k)
            ref[k - 1] = static_cast<double>(2.0L - 2.0L * std::cos(k * pi / (n + 1)));
    } else {
        std::vector<double> dd(d0.begin(), d0.end()), ee(e0.begin(), e0.end());
        int info = 0;
        dsterf_(&n, dd.data(), ee.data(), &info);
        require_close(handle, tag + " reference ?sterf info", info, 0.5);
        ref = dd;
    }
    auto err_vs_ref = [&](const std::vector<rt>& w) {
        double m = 0.0;
        for (int i = 0; i < n; ++i)
            m = std::max(m, std::fabs(static_cast<double>(w[i]) - ref[i]));
        return normT > 0.0 ? m / (eps * normT) : m;
    };

    // the eigenvector solve on the same input: baseline error and a cross-check
    std::vector<rt> wV(n, rt(0));
    {
        int P=1;MPI_Comm_size(ddla::ddlaGetCommunicator(handle),&P);
        const ddla::detail::RowShare share(n,P);
        const int ld=std::max(1,share.count(ddla::ddlaGetRank(handle)));
        DeviceBuffer<rt> d_z(handle, static_cast<std::size_t>(ld) * n);
        std::vector<rt> dd(d0), ee(e0);
        int info = 0;
        ddla::pstedc<rt>(handle, 'V', n, dd.data(), ee.data(), wV.data(), d_z.ptr, ld, info);
        check_ddla_sync(handle);
        require_close(handle, tag + " 'V' info", info, 0.5);
    }
    const double errV = err_vs_ref(wV);

    std::vector<rt> w0;
    for (const char* split : {"1", "0", "2"}) {
        for (const bool keep0 : {false, true}) {
            setenv("DDLA_EIG_DC_SPLIT", split, 1);
            if (keep0) setenv("DDLA_EIG_DC_KEEP_MB", "0", 1);
            else unsetenv("DDLA_EIG_DC_KEEP_MB");
            std::vector<rt> dd(d0), ee(e0), wN(n, rt(0));
            int info = 0;
            ddla::pstedc<rt>(handle, 'N', n, dd.data(), ee.data(), wN.data(),
                             static_cast<rt*>(nullptr), n, info);
            require_close(handle, tag + " 'N' info", info, 0.5);
            if (w0.empty()) {
                w0 = wN;
                if (kind == 4) {
                    double nz = 0.0;
                    for (int i = 0; i < n; ++i) if (wN[i] != rt(0)) nz += 1.0;
                    require_close(handle, tag + " zero matrix: nonzero eigenvalues", nz, 0.5);
                } else {
                    require_close(handle, tag + " eigenvalue error [eps|T|] vs reference",
                                  err_vs_ref(wN), std::max(25.0, 4.0 * errV));
                }
            } else {
                double diff = 0.0;
                for (int i = 0; i < n; ++i) if (wN[i] != w0[i]) diff += 1.0;
                require_close(handle, tag + " bit-identical across split/keep (differing entries)",
                              diff, 0.5);
            }
        }
    }
    unsetenv("DDLA_EIG_DC_SPLIT");
    unsetenv("DDLA_EIG_DC_KEEP_MB");
}

// DDLA_EIG_NBW must divide nblk (1 <= nbw <= nblk): any other value makes pheevd throw on every
// rank, before any collective (it used to fall back to the default silently).
template <typename T>
void check_nbw_env_refused(const ddla::DdlaHandle_t& handle)
{
    const int n = 100, nb = 64;
    int desc[ddla::DDLA_DLEN_];
    ddlaDescInit(desc, handle, n, n, nb, nb, 0, 0);
    api_grid_test::DeviceBuffer<T> d_A(handle, local_size(handle, desc));
    std::vector<typename ddla::real_type<T>::type> w(n);
    // Band-width policy belongs to the explicit two-stage algorithm.
    for (const char* bad : {"48", "128", "0", "-4", "abc", "16x"}) {
        setenv("DDLA_EIG_NBW", bad, 1);
        double threw = 0.0;
        int info = 0;
        try {
            ddla::pheevd_two_stage(handle, 'N', 'U', n, d_A.ptr, desc, w.data(), static_cast<T*>(nullptr), desc, info);
        } catch (const std::exception&) { threw = 1.0; }
        require_close(handle, std::string("pheevd_two_stage refuses DDLA_EIG_NBW=") + bad, 1.0 - threw, 0.0);
    }
    unsetenv("DDLA_EIG_NBW");
}

template <typename T>
void check_body_t(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    using rt = typename ddla::real_type<T>::type;
    const ddla::EigParams params = ddla::eig_select_params(
        !std::is_same_v<T, typename ddla::real_type<T>::type>, base.nb, false);
    const int nbw = params.nbw;
    const int n = 2 * nbw + base.nb + 3;   // at least two panels
    check_pheevd<T>(handle, n, base.nb, true);
    // uplo='L': only the lower triangle is stored (the upper one is poison, see poisoned_input).
    // Stage 0 (pherm_complete) rebuilds the full matrix from the STORED triangle for both uplo
    // values; before 2026-10-15 'L' read the upper triangle and this leg could not see it.
    check_pheevd<T>(handle, n, base.nb, false);
    // The GPU default band width, at a size where the chase runs with more
    // than 2*nbw-1 active bulges: its vectorized ?LARTV branches only run
    // there (n > (2*nbw-1)*(nbw+1)), and a row off-by-one in one of them
    // broke every production-size solve while the small cases above passed.
    // Double precision only (the chase is one template for all four types).
    if constexpr (std::is_same<rt, double>::value) {
        if (ddlaGetBackend(handle) == ddla::DdlaBackend::GPU) {
            const int nbw_gpu = ddla::eig_select_params(
                !std::is_same<T, rt>::value, base.nb, true).nbw;
            const int n_mb = (2 * nbw_gpu - 1) * (nbw_gpu + 1) + base.nb + 7;
            check_pheevd<T>(handle, n_mb, base.nb, true);
        }
    }

    // nbw <= nblk, nbw | nblk (a stage-1 panel is one block column): the GPU default pins
    // nbw = 16 for every nblk that is a positive multiple of 16, everything else uses
    // nbw = nblk, and DDLA_EIG_NBW must divide nblk.  Cover one large-block case on the
    // default path (GPU picks nbw = 16; CPU picks nblk) and one with an explicit override
    // below the block size, both at a size where the nbw = 16 chase runs above the
    // many-bulge threshold -- that is the production configuration for nblk in
    // {32, 64, 128}.
    if constexpr (std::is_same<rt, double>::value) {
        const int nbw16 = 16;
        const int n_big = (2 * nbw16 - 1) * (nbw16 + 1) + 64 + 7;
        check_pheevd<T>(handle, n_big, 64, true);
        check_pheevd<T>(handle, n_big, 32, false);   // lower-stored at a size with a real chase
        setenv("DDLA_EIG_NBW", "16", 1);
        check_pheevd<T>(handle, n_big, 32, true);
        // production block size 64 with explicit bands below it, and block sizes that are not a
        // multiple of 16 (nbw = nblk by default): 24 even, 96 = 6*16 (GPU default 16), 65 odd
        setenv("DDLA_EIG_NBW", "32", 1);
        check_pheevd<T>(handle, n_big, 64, true);
        setenv("DDLA_EIG_NBW", "8", 1);
        check_pheevd<T>(handle, n_big, 64, false);
        unsetenv("DDLA_EIG_NBW");
        check_pheevd<T>(handle, 300, 24, true);
        check_pheevd<T>(handle, 300, 96, true);
        check_pheevd<T>(handle, 300, 65, false);
        // tiny block sizes stay valid (nbw = nblk): the chase at kd = 1, 2
        check_pheevd<T>(handle, 40, 1, true);
        check_pheevd<T>(handle, 41, 2, false);
        check_nbw_env_refused<T>(handle);
    }

    // Distributed divide-and-conquer merges on every lane (2026-10-06): the
    // default leaf 1024 keeps every test size inside one leaf ?STEDC, so the
    // merge machinery never ran in CI.  Leaf 32 at n_big forces a real tree
    // (the top merges cross the 32-roots-per-rank threshold on 4 ranks, so
    // both the split and the non-split merge paths run).  pstedc is the only
    // stage-3 solver, so there is no replicated-solver leg any more.  The
    // ldz > n legs run under each CPU merge-GEMM mode (the knob is ignored
    // by GPU handles).  Real types only: the DC solver is real-only.
    if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
        const int nbw16 = 16;
        const int n_dc = (2 * nbw16 - 1) * (nbw16 + 1) + 64 + 7;
        setenv("DDLA_EIG_DC_LEAF", "32", 1);
        check_pheevd<T>(handle, n_dc, 32, true);
        for (const char* split : {"1", "0", "2"}) {
            setenv("DDLA_EIG_DC_SPLIT", split, 1);
            check_pstedc_ldz<rt>(handle, n_dc, 0);
            check_pstedc_ldz<rt>(handle, n_dc, 1);
        }
        unsetenv("DDLA_EIG_DC_SPLIT");
        // eigenvalues only: every input kind through a real tree (leaf 32), then
        // the deepest trees on tiny n (leaf 2: blocks of size 1), then the
        // default leaves (CPU 'N' tree at leaf 128 vs the single-leaf 'V')
        for (int kind = 0; kind <= 5; ++kind) check_pstedc_n<rt>(handle, n_dc, kind);
        setenv("DDLA_EIG_DC_LEAF", "2", 1);
        for (const int nn : {1, 2, 3, 5, 33}) {
            check_pstedc_n<rt>(handle, nn, 0);
            check_pstedc_n<rt>(handle, nn, 1);
        }
        unsetenv("DDLA_EIG_DC_LEAF");
        check_pstedc_n<rt>(handle, n_dc, 0);
        check_pstedc_n<rt>(handle, n_dc, 2);
    }

    // jobz='N': eigenvalues only (Z untouched); reuse the pstedc-based check.  A
    // second size with leaf 32 makes the CPU eigenvalue-only tree actually merge
    // (the first size is a single leaf).
    auto check_pheevd_N = [&](int nn, bool upper) {
        std::vector<rt> w2(nn);
        int descA[ddla::DDLA_DLEN_];
        ddlaDescInit(descA, handle, nn, nn, base.nb, base.nb, 0, 0);
        api_grid_test::DeviceBuffer<T> d_A(handle, local_size(handle, descA));
        std::vector<T> local(local_size(handle, descA), T(0));
        int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
        ddlaGetGridDims(handle, nprows, npcols);
        ddlaGetGridCoords(handle, myprow, mypcol);
        const int m_loc = ddla::num_loc(nn, base.nb, myprow, 0, nprows);
        const int n_loc = ddla::num_loc(nn, base.nb, mypcol, 0, npcols);
        const int lld = descA[ddla::DDLA_LLD_];
        std::vector<T> Ah_full(static_cast<std::size_t>(nn) * nn);
        for (int j = 0; j < nn; ++j)
            for (int i = 0; i < nn; ++i)
                Ah_full[i + static_cast<std::size_t>(j) * nn] = herm_input<T>(i, j);
        const std::vector<T> Ah = poisoned_input<T>(Ah_full, nn, upper, 1.0e6);
        for (int jl = 0; jl < n_loc; ++jl) {
            const int gj = ddla::indxl2g(jl, base.nb, mypcol, 0, npcols);
            if (gj >= nn) continue;
            for (int il = 0; il < m_loc; ++il) {
                const int gi = ddla::indxl2g(il, base.nb, myprow, 0, nprows);
                if (gi >= nn) continue;
                local[il + static_cast<std::size_t>(jl) * lld] =
                    Ah[gi + static_cast<std::size_t>(gj) * nn];
            }
        }
        upload<T>(handle, d_A.ptr, local);
        int info = 0;
        ddla::pheevd(handle, 'N', upper ? 'U' : 'L', nn, d_A.ptr, descA, w2.data(),
                     static_cast<T*>(nullptr), descA, info);
        require_close(handle, "pheevd 'N' info", info, 0.5);
        const std::vector<rt> ref = host_eigh<T>(Ah_full, nn);
        double normA = 0.0;
        for (const T v : Ah_full) normA = std::max(normA, static_cast<double>(std::abs(v)));
        double dev = 0.0;
        for (int i = 0; i < nn; ++i)
            dev = std::max(dev, std::fabs(static_cast<double>(w2[i] - ref[i])));
        const double metric = dev / (normA * eig_test::eps_trait<rt>::value);
        // same two-stage backward-error bound as the jobz='V' check above
        require_close(handle, "pheevd 'N' eigenvalues", metric / nn, 8.0);
    };
    check_pheevd_N(n, true);
    check_pheevd_N(n, false);
    {
        const int nbw16 = 16;
        const int n_dcN = (2 * nbw16 - 1) * (nbw16 + 1) + 64 + 7;
        setenv("DDLA_EIG_DC_LEAF", "32", 1);
        check_pheevd_N(n_dcN, true);
        check_pheevd_N(n_dcN, false);
        unsetenv("DDLA_EIG_DC_LEAF");
    }
}

} // namespace

static void check_body(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    check_body_t<float>(handle, base);
    check_body_t<double>(handle, base);
    check_body_t<std::complex<float>>(handle, base);
    check_body_t<std::complex<double>>(handle, base);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pheevd", check_body);
}
