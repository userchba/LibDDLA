#ifndef DDLA_SVD_TEST_COMMON_H
#define DDLA_SVD_TEST_COMMON_H

// Helpers shared by test_api_grid_pgebrd and test_api_grid_punmbr: the host LAPACK ?GEBRD / ?UNGBR reference, the
// deterministic fill, the gather of a distributed matrix and the punmbr-on-identity check.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "api_grid_test_common.h"

using namespace api_grid_test;

namespace {

extern "C" {
void sgebrd_(const int*, const int*, float*, const int*, float*, float*, float*, float*,
             float*, const int*, int*);
void dgebrd_(const int*, const int*, double*, const int*, double*, double*, double*, double*,
             double*, const int*, int*);
void cgebrd_(const int*, const int*, std::complex<float>*, const int*, float*, float*,
             std::complex<float>*, std::complex<float>*, std::complex<float>*, const int*, int*);
void zgebrd_(const int*, const int*, std::complex<double>*, const int*, double*, double*,
             std::complex<double>*, std::complex<double>*, std::complex<double>*, const int*,
             int*);
void sorgbr_(const char*, const int*, const int*, const int*, float*, const int*, const float*,
             float*, const int*, int*);
void dorgbr_(const char*, const int*, const int*, const int*, double*, const int*,
             const double*, double*, const int*, int*);
void cungbr_(const char*, const int*, const int*, const int*, std::complex<float>*, const int*,
             const std::complex<float>*, std::complex<float>*, const int*, int*);
void zungbr_(const char*, const int*, const int*, const int*, std::complex<double>*, const int*,
             const std::complex<double>*, std::complex<double>*, const int*, int*);
}

template <typename T> struct is_cplx : std::false_type {};
template <typename U> struct is_cplx<std::complex<U>> : std::true_type {};

template <typename U> U conj_t(U v) { return v; }
template <typename U> std::complex<U> conj_t(std::complex<U> v) { return std::conj(v); }

template <typename U>
double dabs(const U& v) { const double e=static_cast<double>(std::abs(v)); return std::isfinite(e)?e:std::numeric_limits<double>::infinity(); }

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

// Deterministic uniform fill in [-0.5, 0.5): splitmix64 of (i, j, tag).  The
// LAPACK comparison needs a full-rank matrix with spread singular values:
// once Golub-Kahan reaches a (numerical) null space or a cluster, the later
// reflectors are rounding noise and differ between any two codes.  That rules
// out a diagonally dominant fill (clustered singular values) and a
// sin(a*i + b*j) "hash" (a rank-2 core: it left a 23 x 23 fill with
// numerical rank 19).
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

// Every rank ends up with the descriptor's whole leading rows x cols block.
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

// max over all ranks of |x - x(rank 0)|, element-wise: 0 iff bitwise equal
// (for finite values).
template <typename T>
double replication_err(const ddla::DdlaHandle_t& handle, const std::vector<T>& x)
{
    std::vector<T> ref(x);
    if (!ref.empty())
        MPI_Bcast(ref.data(), static_cast<int>(ref.size()), mpi_type<T>(), 0,
                  ddlaGetCommunicator(handle));
    double err = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) err = std::max(err, dabs(x[i] - ref[i]));
    return err;
}

template <typename T>
void host_gebrd(std::vector<T>& a, int m, int n, std::vector<typename ddla::real_type<T>::type>& d,
                std::vector<typename ddla::real_type<T>::type>& e, std::vector<T>& tq,
                std::vector<T>& tp)
{
    const int lda = std::max(1, m);
    T q[1];
    int lwork = -1, info = 0;
    auto call = [&](T* work, int lw) {
        if constexpr (std::is_same<T, float>::value)
            sgebrd_(&m, &n, a.data(), &lda, d.data(), e.data(), tq.data(), tp.data(), work, &lw, &info);
        else if constexpr (std::is_same<T, double>::value)
            dgebrd_(&m, &n, a.data(), &lda, d.data(), e.data(), tq.data(), tp.data(), work, &lw, &info);
        else if constexpr (std::is_same<T, std::complex<float>>::value)
            cgebrd_(&m, &n, a.data(), &lda, d.data(), e.data(), tq.data(), tp.data(), work, &lw, &info);
        else
            zgebrd_(&m, &n, a.data(), &lda, d.data(), e.data(), tq.data(), tp.data(), work, &lw, &info);
    };
    call(q, lwork);
    lwork = std::max(1, static_cast<int>(std::real(q[0])));
    std::vector<T> work(static_cast<std::size_t>(lwork));
    call(work.data(), lwork);
    if (info != 0) {
        std::cerr << "host ?gebrd failed, info=" << info << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// ?UNGBR on a dense order x order buffer (column-major, ld = order) whose
// leading block holds pgebrd/?GEBRD reflector storage.
template <typename T>
void host_ungbr(char vect, int order, int kk, std::vector<T>& a, const std::vector<T>& tau)
{
    T q[1];
    int lwork = -1, info = 0;
    const int lda = std::max(1, order);
    auto call = [&](T* work, int lw) {
        if constexpr (std::is_same<T, float>::value)
            sorgbr_(&vect, &order, &order, &kk, a.data(), &lda, tau.data(), work, &lw, &info);
        else if constexpr (std::is_same<T, double>::value)
            dorgbr_(&vect, &order, &order, &kk, a.data(), &lda, tau.data(), work, &lw, &info);
        else if constexpr (std::is_same<T, std::complex<float>>::value)
            cungbr_(&vect, &order, &order, &kk, a.data(), &lda, tau.data(), work, &lw, &info);
        else
            zungbr_(&vect, &order, &order, &kk, a.data(), &lda, tau.data(), work, &lw, &info);
    };
    call(q, lwork);
    lwork = std::max(1, static_cast<int>(std::real(q[0])));
    std::vector<T> work(static_cast<std::size_t>(lwork));
    call(work.data(), lwork);
    if (info != 0) {
        std::cerr << "host ?ungbr failed, info=" << info << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

template <typename T>
void call_punmbr(const ddla::DdlaHandle_t& handle, char vect, char side, char trans, int m, int n,
                 int k, const T* d_A, const int* descA, const T* tau, T* d_C, const int* descC,
                 int& info)
{
#if DDLA_HAS_GPU
    if (ddlaGetBackend(handle) == DdlaBackend::CPU)
        ddla::punmbr<DdlaBackend::CPU, T>(handle, vect, side, trans, m, n, k, d_A, descA, tau,
                                          d_C, descC, info);
    else
        ddla::punmbr<DdlaBackend::GPU, T>(handle, vect, side, trans, m, n, k, d_A, descA, tau,
                                          d_C, descC, info);
#elif DDLA_HAS_GPU
    ddla::punmbr<DdlaBackend::GPU, T>(handle, vect, side, trans, m, n, k, d_A, descA, tau, d_C,
                                      descC, info);
#else
    ddla::punmbr<DdlaBackend::CPU, T>(handle, vect, side, trans, m, n, k, d_A, descA, tau, d_C,
                                      descC, info);
#endif
}

struct CaseGeom {
    int m, n;          // problem size
    int mb, nb;        // block sizes
    int rsrc, csrc;    // first owning process row/column
    int dm, dn;        // descriptor dims (>= m, n; larger means padded)
    int fill = 0;      // FILL_HASH / FILL_TINY / FILL_NEGZERO / FILL_HUGE
    bool tall = false; // tall-skinny: compare with host ?gebrd only (an m x m Q does not fit)
};

// punmbr on a distributed identity: the result must equal the host operator.
template <typename T>
void check_punmbr(const ddla::DdlaHandle_t& handle, const CaseGeom& g, const T* d_A,
                  const int* descA, const std::vector<T>& tq, const std::vector<T>& tp,
                  const std::vector<T>& hQ, const std::vector<T>& hPh, const std::string& tag)
{
    using rt = typename ddla::real_type<T>::type;
    const double eps = std::is_same<rt, float>::value ? 1.1920929e-7 : 2.2204460492503131e-16;
    const int m = g.m, n = g.n;
    const int pad = (g.dm != g.m || g.dn != g.n) ? 2 : 0;
    // vect = 'Q': C is m x m, rows distributed like A's.
    for (char trans : {'N', 'C'}) {
        int desc[ddla::DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(desc, handle, m + pad, m + pad, g.mb, g.nb, g.rsrc, 0));
        auto h_C = make_local<T>(handle, desc, [&](int i, int j) {
            if (i >= m || j >= m) return T(7);                 // sentinel outside the block
            return T(i == j ? rt(1) : rt(0));
        });
        DeviceBuffer<T> d_C(handle, h_C.size());
        upload(handle, d_C.ptr, h_C);
        int info = -1;
        call_punmbr<T>(handle, 'Q', 'L', trans, m, m, n, d_A, descA, tq.data(), d_C.ptr, desc, info);
        check_ddla_sync(handle);
        const auto full = gather_full<T>(handle, d_C.ptr, desc, m + pad, m + pad);
        double err = 0.0, sent = 0.0;
        for (int j = 0; j < m + pad; ++j)
            for (int i = 0; i < m + pad; ++i) {
                const T got = full[i + static_cast<std::size_t>(j) * (m + pad)];
                if (i >= m || j >= m) { sent = std::max(sent, dabs(got - T(7))); continue; }
                const T want = (trans == 'N') ? hQ[i + static_cast<std::size_t>(j) * m]
                                              : conj_t(hQ[j + static_cast<std::size_t>(i) * m]);
                err = std::max(err, dabs(got - want));
            }
        require_close(handle, tag + " punmbr Q" + trans, err / (std::max(m, 1) * eps), 50.0);
        require_close(handle, tag + " punmbr Q" + trans + " outside", sent, 0.0);
    }
    // vect = 'P': C is n x n, columns distributed like A's.
    for (char trans : {'N', 'C'}) {
        int desc[ddla::DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(desc, handle, n + pad, n + pad, g.mb, g.nb, 0, g.csrc));
        auto h_C = make_local<T>(handle, desc, [&](int i, int j) {
            if (i >= n || j >= n) return T(7);
            return T(i == j ? rt(1) : rt(0));
        });
        DeviceBuffer<T> d_C(handle, h_C.size());
        upload(handle, d_C.ptr, h_C);
        int info = -1;
        call_punmbr<T>(handle, 'P', 'R', trans, n, n, m, d_A, descA, tp.data(), d_C.ptr, desc, info);
        check_ddla_sync(handle);
        const auto full = gather_full<T>(handle, d_C.ptr, desc, n + pad, n + pad);
        double err = 0.0, sent = 0.0;
        for (int j = 0; j < n + pad; ++j)
            for (int i = 0; i < n + pad; ++i) {
                const T got = full[i + static_cast<std::size_t>(j) * (n + pad)];
                if (i >= n || j >= n) { sent = std::max(sent, dabs(got - T(7))); continue; }
                // C P = P = (P^H)^H;  C P^H = P^H.
                const T want = (trans == 'N') ? conj_t(hPh[j + static_cast<std::size_t>(i) * n])
                                              : hPh[i + static_cast<std::size_t>(j) * n];
                err = std::max(err, dabs(got - want));
            }
        require_close(handle, tag + " punmbr P" + trans, err / (std::max(n, 1) * eps), 50.0);
        require_close(handle, tag + " punmbr P" + trans + " outside", sent, 0.0);
    }
}

} // namespace

#endif // DDLA_SVD_TEST_COMMON_H
