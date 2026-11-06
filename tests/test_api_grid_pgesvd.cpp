#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "api_grid_test_common.h"

// ---------------------------------------------------------------------------
// Regression test for the distributed SVD pgesvd (ScaLAPACK PxGESVD
// semantics: economy U (m x k), VT (k x n), s descending).
//
// Every case runs all four (jobu, jobvt) combinations, with vectors under
// distributed D&C using both the default and a small DDLA_SVD_DC_LEAF,
// and checks:
//   - info == 0, s bitwise identical on every rank, descending, >= 0;
//   - s against host LAPACK ?GESVD:  |s - s_ref| / (s_max max(m,n) eps);
//   - U^H U = I and VT VT^H = I (when computed);
//   - jobu = jobvt = 'V':  A = U diag(s) VT;
//     only U:   U U^H A = A and ||(U^H A)(i,:)|| = s_i;
//     only VT:  A VT^H VT = A and ||(A VT^H)(:,i)|| = s_i
//     (all reference-free, bound 50 in units of n*eps);
//   - padded descriptors keep everything outside the leading blocks.
// The fills cover generic spectra, rank deficiency (exact zero singular
// values), a zero matrix, repeated singular values and a graded matrix.
// ---------------------------------------------------------------------------

using namespace api_grid_test;

namespace {

extern "C" {
void sgesvd_(const char*, const char*, const int*, const int*, float*, const int*, float*,
             float*, const int*, float*, const int*, float*, const int*, int*);
void dgesvd_(const char*, const char*, const int*, const int*, double*, const int*, double*,
             double*, const int*, double*, const int*, double*, const int*, int*);
void cgesvd_(const char*, const char*, const int*, const int*, std::complex<float>*, const int*,
             float*, std::complex<float>*, const int*, std::complex<float>*, const int*,
             std::complex<float>*, const int*, float*, int*);
void zgesvd_(const char*, const char*, const int*, const int*, std::complex<double>*, const int*,
             double*, std::complex<double>*, const int*, std::complex<double>*, const int*,
             std::complex<double>*, const int*, double*, int*);
}

template <typename T> struct is_cplx : std::false_type {};
template <typename U> struct is_cplx<std::complex<U>> : std::true_type {};

template <typename U> U conj_t(U v) { return v; }
template <typename U> std::complex<U> conj_t(std::complex<U> v) { return std::conj(v); }

template <typename U>
double dabs(const U& v) { return static_cast<double>(std::abs(v)); }

template <typename T> const char* type_tag() { return "?"; }
template <> const char* type_tag<float>() { return "s"; }
template <> const char* type_tag<double>() { return "d"; }
template <> const char* type_tag<std::complex<float>>() { return "c"; }
template <> const char* type_tag<std::complex<double>>() { return "z"; }

template <typename T> MPI_Datatype mpi_type();
template <> MPI_Datatype mpi_type<float>() { return MPI_FLOAT; }
template <> MPI_Datatype mpi_type<double>() { return MPI_DOUBLE; }
template <> MPI_Datatype mpi_type<std::complex<float>>() { return MPI_C_FLOAT_COMPLEX; }
template <> MPI_Datatype mpi_type<std::complex<double>>() { return MPI_C_DOUBLE_COMPLEX; }

enum Fill { FILL_HASH, FILL_RANK2, FILL_ZERO, FILL_REPEATED, FILL_GRADED };

// splitmix64 of (i, j, tag) -> uniform in [-0.5, 0.5).  A sin-of-linear-
// combination "hash" is NOT random enough here: sin(a*i + b*j) is a rank-2
// matrix, and its fractional part left a 23 x 23 fill with numerical rank 19,
// where the last Golub-Kahan reflectors are pure rounding noise.
inline double hash01(int i, int j, int tag)
{
    std::uint64_t x = static_cast<std::uint64_t>(i) * 0x9E3779B97F4A7C15ull
                    ^ static_cast<std::uint64_t>(j) * 0xC2B2AE3D27D4EB4Full
                    ^ static_cast<std::uint64_t>(tag + 1) * 0x165667B19E3779F9ull;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27; x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    return static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0) - 0.5;
}

template <typename T>
T hash_value(int i, int j, int tag)
{
    using rt = typename ddla::real_type<T>::type;
    if constexpr (is_cplx<T>::value)
        return T(static_cast<rt>(hash01(i, j, tag)), static_cast<rt>(hash01(i, j, tag + 1000)));
    else
        return T(static_cast<rt>(hash01(i, j, tag)));
}

// Leading m x n block of a dense test matrix.
template <typename T>
T fill_value(Fill f, int m, int n, int i, int j)
{
    using rt = typename ddla::real_type<T>::type;
    switch (f) {
    case FILL_RANK2:        // two outer products: all but two singular values are 0
        return hash_value<T>(i, 0, 11) * hash_value<T>(0, j, 12)
             + hash_value<T>(i, 1, 13) * hash_value<T>(1, j, 14);
    case FILL_ZERO:
        return T(0);
    case FILL_REPEATED: {   // 2 x (leading block of a dense Householder matrix
                            // I - 2 w w^H / w^H w): every singular value is 2
        const int N = std::max(m, n);
        double ww = 0.0;
        for (int q = 0; q < N; ++q) ww += std::norm(hash_value<T>(q, 0, 21));
        const T h = T(i == j ? rt(1) : rt(0))
                  - T(static_cast<rt>(2.0 / ww)) * hash_value<T>(i, 0, 21) * conj_t(hash_value<T>(j, 0, 21));
        return T(rt(2)) * h;
    }
    case FILL_GRADED:       // columns scaled over eight decades
        return hash_value<T>(i, j, 15) * static_cast<rt>(std::pow(10.0, -8.0 * j / 40.0));
    case FILL_HASH:
    default:
        return hash_value<T>(i, j, 7);
    }
}

template <typename T>
std::vector<T> gather_full(const ddla::DdlaHandle_t& handle, const T* d_A,
                           const int* desc, int rows, int cols)
{
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
    const int m_loc = ddla::num_loc(rows, desc[ddla::DDLA_MB_], myprow, desc[ddla::DDLA_RSRC_], nprows);
    const int n_loc = ddla::num_loc(cols, desc[ddla::DDLA_NB_], mypcol, desc[ddla::DDLA_CSRC_], npcols);
    const int lld = desc[ddla::DDLA_LLD_];
    const auto local = download<T>(handle, d_A, local_size(handle, desc));
    std::vector<T> full(static_cast<std::size_t>(rows) * cols, T(0));
    for (int j = 0; j < n_loc; ++j) {
        const int gj = ddla::indx_l2g_c(desc, handle, j);
        for (int i = 0; i < m_loc; ++i) {
            const int gi = ddla::indx_l2g_r(desc, handle, i);
            full[gi + static_cast<std::size_t>(gj) * rows] = local[i + static_cast<std::size_t>(j) * lld];
        }
    }
    if (!full.empty())
        MPI_Allreduce(MPI_IN_PLACE, full.data(), static_cast<int>(full.size()), mpi_type<T>(),
                      MPI_SUM, ddlaGetCommunicator(handle));
    return full;
}

template <typename T>
std::vector<typename ddla::real_type<T>::type> host_singular_values(std::vector<T> a, int m, int n)
{
    using rt = typename ddla::real_type<T>::type;
    const int k = std::min(m, n);
    std::vector<rt> s(static_cast<std::size_t>(std::max(k, 1)));
    std::vector<rt> rwork(static_cast<std::size_t>(std::max(1, 5 * k)));
    const char job = 'N';
    const int lda = std::max(1, m), one = 1;
    T dummy[1];
    T q[1];
    int lwork = -1, info = 0;
    auto call = [&](T* work, int lw) {
        if constexpr (std::is_same<T, float>::value)
            sgesvd_(&job, &job, &m, &n, a.data(), &lda, s.data(), dummy, &one, dummy, &one, work, &lw, &info);
        else if constexpr (std::is_same<T, double>::value)
            dgesvd_(&job, &job, &m, &n, a.data(), &lda, s.data(), dummy, &one, dummy, &one, work, &lw, &info);
        else if constexpr (std::is_same<T, std::complex<float>>::value)
            cgesvd_(&job, &job, &m, &n, a.data(), &lda, s.data(), dummy, &one, dummy, &one, work, &lw,
                    rwork.data(), &info);
        else
            zgesvd_(&job, &job, &m, &n, a.data(), &lda, s.data(), dummy, &one, dummy, &one, work, &lw,
                    rwork.data(), &info);
    };
    call(q, lwork);
    lwork = std::max(1, static_cast<int>(std::real(q[0])));
    std::vector<T> work(static_cast<std::size_t>(lwork));
    call(work.data(), lwork);
    if (info != 0) {
        std::cerr << "host ?gesvd failed, info=" << info << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return s;
}

template <typename T>
void call_pgesvd(const ddla::DdlaHandle_t& handle, char jobu, char jobvt, int m, int n,
                 T* d_A, const int* descA, typename ddla::real_type<T>::type* s,
                 T* d_U, const int* descU, T* d_VT, const int* descVT, int& info)
{
#if DDLA_HAS_GPU
    if (ddlaGetBackend(handle) == DdlaBackend::CPU)
        ddla::pgesvd<DdlaBackend::CPU, T>(handle, jobu, jobvt, m, n, d_A, descA, s, d_U, descU,
                                          d_VT, descVT, info);
    else
        ddla::pgesvd<DdlaBackend::GPU, T>(handle, jobu, jobvt, m, n, d_A, descA, s, d_U, descU,
                                          d_VT, descVT, info);
#elif DDLA_HAS_GPU
    ddla::pgesvd<DdlaBackend::GPU, T>(handle, jobu, jobvt, m, n, d_A, descA, s, d_U, descU, d_VT,
                                      descVT, info);
#else
    ddla::pgesvd<DdlaBackend::CPU, T>(handle, jobu, jobvt, m, n, d_A, descA, s, d_U, descU, d_VT,
                                      descVT, info);
#endif
}

struct CaseGeom {
    int m, n;          // problem size
    int mb, nb;        // A's block sizes (U shares mb/rsrc, VT nb/csrc)
    int rsrc, csrc;
    int pad;           // extra rows/cols on every descriptor
    Fill fill;
};

template <typename T>
void run_case(const ddla::DdlaHandle_t& handle, const CaseGeom& g, char jobu, char jobvt,
              const std::string& tag)
{
    using rt = typename ddla::real_type<T>::type;
    const double eps = std::is_same<rt, float>::value ? 1.1920929e-7 : 2.2204460492503131e-16;
    const int m = g.m, n = g.n, k = std::min(m, n);
    const bool wu = (jobu == 'V'), wv = (jobvt == 'V');
    MPI_Comm comm = ddlaGetCommunicator(handle);
    const int myid = ddlaGetRank(handle);
    const int p = g.pad;

    int descA[ddla::DDLA_DLEN_], descU[ddla::DDLA_DLEN_], descVT[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, handle, m + p, n + p, g.mb, g.nb, g.rsrc, g.csrc));
    // U: rows like A's, columns with an unrelated block size; VT: columns
    // like A's, rows unrelated.
    const int ubn = 2 + (g.nb % 3), vbm = 3 + (g.mb % 2);
    DDLA_CHECK(ddlaDescInit(descU, handle, m + p, k + p, g.mb, ubn, g.rsrc, 0));
    DDLA_CHECK(ddlaDescInit(descVT, handle, k + p, n + p, vbm, g.nb, 0, g.csrc));

    auto value = [&](int i, int j) {
        if (i >= m || j >= n) return T(rt(3));                  // padding sentinel
        return fill_value<T>(g.fill, m, n, i, j);
    };
    const T sent = T(rt(-9));
    auto h_A = make_local<T>(handle, descA, value);
    auto h_U = make_local<T>(handle, descU, [&](int, int) { return sent; });
    auto h_VT = make_local<T>(handle, descVT, [&](int, int) { return sent; });
    DeviceBuffer<T> d_A(handle, h_A.size()), d_U(handle, h_U.size()), d_VT(handle, h_VT.size());
    upload(handle, d_A.ptr, h_A);
    upload(handle, d_U.ptr, h_U);
    upload(handle, d_VT.ptr, h_VT);
    check_ddla_sync(handle);

    std::vector<rt> s(static_cast<std::size_t>(std::max(k, 1)), rt(-1));
    int info = -1;
    call_pgesvd<T>(handle, jobu, jobvt, m, n, d_A.ptr, descA, s.data(), d_U.ptr, descU,
                   d_VT.ptr, descVT, info);
    check_ddla_sync(handle);
    int ginfo = -1;
    MPI_Allreduce(&info, &ginfo, 1, MPI_INT, MPI_MAX, comm);
    if (ginfo != 0) {
        if (myid == 0) std::cerr << "pgesvd info=" << ginfo << " for " << tag << std::endl;
        MPI_Abort(comm, 1);
    }

    // A outside its leading block untouched (A itself is destroyed inside).
    {
        const auto fa = gather_full<T>(handle, d_A.ptr, descA, m + p, n + p);
        double outside = 0.0;
        for (int j = 0; j < n + p; ++j)
            for (int i = 0; i < m + p; ++i)
                if (i >= m || j >= n)
                    outside = std::max(outside, dabs(fa[i + static_cast<std::size_t>(j) * (m + p)] - T(rt(3))));
        require_close(handle, tag + " A outside", outside, 0.0);
    }
    if (k == 0) {
        if (myid == 0)
            std::cout << "[grid " << grid_name(handle) << "] pgesvd " << tag << " ok (empty)" << std::endl;
        return;
    }

    // ---- singular values -----------------------------------------------------
    {
        std::vector<rt> ref(s);
        MPI_Bcast(ref.data(), static_cast<int>(ref.size()), mpi_type<rt>(), 0, comm);
        double repl = 0.0;
        for (int i = 0; i < k; ++i) repl = std::max(repl, dabs(s[i] - ref[i]));
        require_close(handle, tag + " s replication", repl, 0.0);
    }
    double order = 0.0;
    for (int i = 0; i < k; ++i) {
        if (s[i] < rt(0)) order = 1.0;
        if (i + 1 < k && s[i] < s[i + 1]) order = 1.0;
    }
    require_close(handle, tag + " s descending/nonnegative", order, 0.0);

    std::vector<T> a0(static_cast<std::size_t>(m) * n);
    double a0_fnorm = 0.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            a0[i + static_cast<std::size_t>(j) * m] = fill_value<T>(g.fill, m, n, i, j);
            a0_fnorm += std::norm(a0[i + static_cast<std::size_t>(j) * m]);
        }
    a0_fnorm = std::sqrt(a0_fnorm);
    const double anorm = std::max(a0_fnorm, 1.0);
    const double unit = std::max(m, n) * eps;
    {
        const auto sref = host_singular_values<T>(a0, m, n);
        double err = 0.0;
        for (int i = 0; i < k; ++i) err = std::max(err, dabs(s[i] - sref[i]));
        require_close(handle, tag + " s-vs-lapack",
                      err / (std::max(static_cast<double>(sref[0]), 1.0) * unit), 50.0);
    }

    std::vector<T> U, VT;
    if (wu) {
        const auto fu = gather_full<T>(handle, d_U.ptr, descU, m + p, k + p);
        double outside = 0.0;
        U.assign(static_cast<std::size_t>(m) * k, T(0));
        for (int j = 0; j < k + p; ++j)
            for (int i = 0; i < m + p; ++i) {
                const T v = fu[i + static_cast<std::size_t>(j) * (m + p)];
                if (i < m && j < k) U[i + static_cast<std::size_t>(j) * m] = v;
                else outside = std::max(outside, dabs(v - sent));
            }
        require_close(handle, tag + " U outside", outside, 0.0);
        double orth = 0.0;
        for (int j = 0; j < k; ++j)
            for (int i = 0; i < k; ++i) {
                T acc{};
                for (int q = 0; q < m; ++q)
                    acc += conj_t(U[q + static_cast<std::size_t>(i) * m]) * U[q + static_cast<std::size_t>(j) * m];
                orth = std::max(orth, dabs(acc - T(i == j ? rt(1) : rt(0))));
            }
        require_close(handle, tag + " U orth", orth / (std::max(m, 1) * eps), 50.0);
    }
    if (wv) {
        const auto fv = gather_full<T>(handle, d_VT.ptr, descVT, k + p, n + p);
        double outside = 0.0;
        VT.assign(static_cast<std::size_t>(k) * n, T(0));
        for (int j = 0; j < n + p; ++j)
            for (int i = 0; i < k + p; ++i) {
                const T v = fv[i + static_cast<std::size_t>(j) * (k + p)];
                if (i < k && j < n) VT[i + static_cast<std::size_t>(j) * k] = v;
                else outside = std::max(outside, dabs(v - sent));
            }
        require_close(handle, tag + " VT outside", outside, 0.0);
        double orth = 0.0;
        for (int j = 0; j < k; ++j)
            for (int i = 0; i < k; ++i) {
                T acc{};
                for (int q = 0; q < n; ++q)
                    acc += VT[i + static_cast<std::size_t>(q) * k] * conj_t(VT[j + static_cast<std::size_t>(q) * k]);
                orth = std::max(orth, dabs(acc - T(i == j ? rt(1) : rt(0))));
            }
        require_close(handle, tag + " VT orth", orth / (std::max(n, 1) * eps), 50.0);
    }
    if (!wu && !wv) {
        // VT/U were not referenced: their sentinels must survive untouched.
        const auto fu = gather_full<T>(handle, d_U.ptr, descU, m + p, k + p);
        const auto fv = gather_full<T>(handle, d_VT.ptr, descVT, k + p, n + p);
        double untouched = 0.0;
        for (const T& v : fu) untouched = std::max(untouched, dabs(v - sent));
        for (const T& v : fv) untouched = std::max(untouched, dabs(v - sent));
        require_close(handle, tag + " U/VT unreferenced", untouched, 0.0);
    }

    if (wu && wv) {
        double res = 0.0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                T acc{};
                for (int q = 0; q < k; ++q)
                    acc += U[i + static_cast<std::size_t>(q) * m] * T(s[q]) * VT[q + static_cast<std::size_t>(j) * k];
                res = std::max(res, dabs(acc - a0[i + static_cast<std::size_t>(j) * m]));
            }
        require_close(handle, tag + " A-USV^H", res / (anorm * unit), 50.0);
    } else if (wu) {
        // W = U^H A (k x n): ||W(i,:)|| = s_i and U W = A.
        std::vector<T> W(static_cast<std::size_t>(k) * n, T(0));
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < k; ++i) {
                T acc{};
                for (int q = 0; q < m; ++q)
                    acc += conj_t(U[q + static_cast<std::size_t>(i) * m]) * a0[q + static_cast<std::size_t>(j) * m];
                W[i + static_cast<std::size_t>(j) * k] = acc;
            }
        double rows = 0.0, proj = 0.0;
        for (int i = 0; i < k; ++i) {
            double nrm = 0.0;
            for (int j = 0; j < n; ++j) nrm += std::norm(W[i + static_cast<std::size_t>(j) * k]);
            rows = std::max(rows, std::abs(std::sqrt(nrm) - static_cast<double>(s[i])));
        }
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                T acc{};
                for (int q = 0; q < k; ++q)
                    acc += U[i + static_cast<std::size_t>(q) * m] * W[q + static_cast<std::size_t>(j) * k];
                proj = std::max(proj, dabs(acc - a0[i + static_cast<std::size_t>(j) * m]));
            }
        require_close(handle, tag + " |U^H A| rows = s", rows / (anorm * unit), 50.0);
        require_close(handle, tag + " U U^H A = A", proj / (anorm * unit), 50.0);
    } else if (wv) {
        // Z = A VT^H (m x k): ||Z(:,i)|| = s_i and Z VT = A.
        std::vector<T> Z(static_cast<std::size_t>(m) * k, T(0));
        for (int j = 0; j < k; ++j)
            for (int i = 0; i < m; ++i) {
                T acc{};
                for (int q = 0; q < n; ++q)
                    acc += a0[i + static_cast<std::size_t>(q) * m] * conj_t(VT[j + static_cast<std::size_t>(q) * k]);
                Z[i + static_cast<std::size_t>(j) * m] = acc;
            }
        double cols = 0.0, proj = 0.0;
        for (int j = 0; j < k; ++j) {
            double nrm = 0.0;
            for (int i = 0; i < m; ++i) nrm += std::norm(Z[i + static_cast<std::size_t>(j) * m]);
            cols = std::max(cols, std::abs(std::sqrt(nrm) - static_cast<double>(s[j])));
        }
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                T acc{};
                for (int q = 0; q < k; ++q)
                    acc += Z[i + static_cast<std::size_t>(q) * m] * VT[q + static_cast<std::size_t>(j) * k];
                proj = std::max(proj, dabs(acc - a0[i + static_cast<std::size_t>(j) * m]));
            }
        require_close(handle, tag + " |A VT^H| cols = s", cols / (anorm * unit), 50.0);
        require_close(handle, tag + " A VT^H VT = A", proj / (anorm * unit), 50.0);
    }

    if (myid == 0)
        std::cout << "[grid " << grid_name(handle) << "] pgesvd " << tag << " ok" << std::endl;
}

template <typename Fn>
void require_throws(const ddla::DdlaHandle_t& handle, const std::string& name, Fn fn)
{
    int threw = 0;
    try { fn(); } catch (const std::exception&) { threw = 1; }
    int all = 0;
    MPI_Allreduce(&threw, &all, 1, MPI_INT, MPI_MIN, ddlaGetCommunicator(handle));
    require_close(handle, name + " rejected", all == 1 ? 0.0 : 1.0, 0.0);
}

template <typename T>
void check_rejection(const ddla::DdlaHandle_t& handle)
{
    using rt = typename ddla::real_type<T>::type;
    const int m = 12, n = 9, k = 9;
    int descA[ddla::DDLA_DLEN_], descU[ddla::DDLA_DLEN_], descVT[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, handle, m, n, 3, 3, 0, 0));
    DDLA_CHECK(ddlaDescInit(descU, handle, m, k, 2, 3, 0, 0));      // MB differs from A's
    DDLA_CHECK(ddlaDescInit(descVT, handle, k, n, 3, 3, 0, 0));
    DeviceBuffer<T> d_A(handle, local_size(handle, descA));
    DeviceBuffer<T> d_U(handle, local_size(handle, descU));
    DeviceBuffer<T> d_VT(handle, local_size(handle, descVT));
    std::vector<rt> s(static_cast<std::size_t>(k));
    int info = 0;
    require_throws(handle, "pgesvd misaligned U", [&] {
        call_pgesvd<T>(handle, 'V', 'V', m, n, d_A.ptr, descA, s.data(), d_U.ptr, descU,
                       d_VT.ptr, descVT, info);
    });
    require_throws(handle, "pgesvd jobu='A'", [&] {
        call_pgesvd<T>(handle, 'A', 'N', m, n, d_A.ptr, descA, s.data(), d_U.ptr, descU,
                       d_VT.ptr, descVT, info);
    });
}

template <typename T>
void check_pgesvd(const ddla::DdlaHandle_t& handle)
{
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    const int pr = nprows - 1, pc = npcols - 1;
    const CaseGeom cases[] = {
        {23, 23, 4, 4, 0, 0, 0, FILL_HASH},          // square
        {37, 17, 5, 3, 0, 0, 0, FILL_HASH},          // tall, mb != nb
        {17, 37, 3, 5, 0, 0, 0, FILL_HASH},          // wide (lower bidiagonal)
        {20, 14, 4, 4, pr, pc, 0, FILL_HASH},        // rsrc = csrc = P-1
        {14, 20, 4, 4, pr, pc, 0, FILL_HASH},        // wide, rsrc = csrc = P-1
        {21, 15, 4, 5, 0, 0, 3, FILL_HASH},          // padded descriptors
        {15, 21, 5, 4, 0, 0, 3, FILL_HASH},          // padded, wide
        {9, 7, 1, 2, 0, 0, 0, FILL_HASH},            // mb = 1
        {7, 9, 2, 1, 0, 0, 0, FILL_HASH},            // nb = 1
        {1, 6, 2, 2, 0, 0, 0, FILL_HASH},            // m = 1
        {6, 1, 2, 2, 0, 0, 0, FILL_HASH},            // n = 1
        {1, 1, 1, 1, 0, 0, 0, FILL_HASH},            // 1 x 1
        {0, 4, 2, 2, 0, 0, 0, FILL_HASH},            // m = 0
        {4, 0, 2, 2, 0, 0, 0, FILL_HASH},            // n = 0
        {24, 18, 4, 3, 0, 0, 0, FILL_RANK2},         // rank deficient
        {18, 24, 3, 4, 0, 0, 0, FILL_RANK2},         // rank deficient, wide
        {16, 12, 4, 4, 0, 0, 0, FILL_ZERO},          // zero matrix
        {19, 16, 4, 4, 0, 0, 0, FILL_REPEATED},      // repeated singular values
        {16, 19, 4, 4, 0, 0, 0, FILL_REPEATED},      // repeated, wide
        {30, 40, 5, 5, 0, 0, 0, FILL_GRADED},        // graded
        {64, 48, 8, 6, 0, 0, 0, FILL_HASH},          // multi-cycle
        {48, 64, 6, 8, 0, 0, 0, FILL_HASH},          // multi-cycle, wide
        // k > 25 (?BDSDC's SMLSIZ): real divide-and-conquer levels, incl.
        // heavy deflation (rank 2, all-equal singular values)
        {150, 120, 16, 12, 0, 0, 0, FILL_HASH},
        {120, 150, 12, 16, 0, 0, 0, FILL_HASH},
        {80, 60, 8, 8, 0, 0, 0, FILL_RANK2},
        {60, 70, 8, 8, 0, 0, 0, FILL_REPEATED},
    };
    const char jobs[4][2] = {{'V', 'V'}, {'V', 'N'}, {'N', 'V'}, {'N', 'N'}};
    // Default and small leaves exercise both leaf solves and deep merges.
    // The values-only dqds job is independent of the leaf policy, so runs once.
    struct Leg { const char* name; const char* leaf; };
    const Leg legs[] = {{"dc", nullptr}, {"dc-leaf16", "16"}};
    for (const Leg& lg : legs) {
        const std::string leg(lg.name);
        if (lg.leaf != nullptr) setenv("DDLA_SVD_DC_LEAF", lg.leaf, 1);
        else unsetenv("DDLA_SVD_DC_LEAF");
        for (const CaseGeom& g : cases)
            for (const auto& jb : jobs) {
                if (jb[0] == 'N' && jb[1] == 'N' && leg != "dc") continue;
                run_case<T>(handle, g, jb[0], jb[1],
                            std::string(type_tag<T>()) + " " + std::to_string(g.m) + "x" +
                                std::to_string(g.n) + " fill" + std::to_string(static_cast<int>(g.fill)) +
                                " " + jb[0] + jb[1] + " " + leg);
            }
    }
    unsetenv("DDLA_SVD_DC_LEAF");
    check_rejection<T>(handle);
}

static void check_body(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    (void)base;
    check_pgesvd<double>(handle);
    check_pgesvd<Complex>(handle);
    check_pgesvd<float>(handle);
    check_pgesvd<std::complex<float>>(handle);

    {
        int nprows = 0, npcols = 0;
        ddlaGetGridDims(handle, nprows, npcols);
        ddla::DdlaHandle_t h_cm = nullptr;
        ddla::ddlaInit(h_cm);
        ddla::ddlaSet(h_cm, MPI_COMM_WORLD, nprows, npcols, 'C');
        run_case<Complex>(h_cm, CaseGeom{23, 19, 4, 4, 0, 0, 0, FILL_HASH}, 'V', 'V', "colmajor-grid tall");
        run_case<Complex>(h_cm, CaseGeom{19, 23, 4, 4, 0, 0, 0, FILL_HASH}, 'V', 'V', "colmajor-grid wide");
        ddla::ddlaDestroy(h_cm);
    }

#if DDLA_HAS_GPU
    if (ddlaGetBackend(handle) == DdlaBackend::GPU && ddlaBackendAvailable(DdlaBackend::CPU)) {
        int nprows = 0, npcols = 0;
        ddlaGetGridDims(handle, nprows, npcols);
        ddla::DdlaHandle_t h_cpu = nullptr;
        ddla::ddlaInit(h_cpu, DdlaBackend::CPU);
        ddla::ddlaSet(h_cpu, MPI_COMM_WORLD, nprows, npcols, 'R');
        check_pgesvd<double>(h_cpu);
        check_pgesvd<Complex>(h_cpu);
        check_pgesvd<float>(h_cpu);
        check_pgesvd<std::complex<float>>(h_cpu);
        ddla::ddlaDestroy(h_cpu);
    }
#endif
}

} // namespace

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pgesvd", check_body);
}
