#include "Eigensolver/two_stage/phb2st_impl.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "api_grid_test_common.h"
#include "Eigensolver/eig_params.h"
#include "eig_test_common.h"

// ---------------------------------------------------------------------------
// Isolation test for stage 2 (phb2st: band -> tridiagonal with recorded
// rotations).
//
// Pipeline per (grid, size, type): build a full Hermitian input, run
// phe2hb to band form, run phb2st, then verify on the host (every rank has
// the replicated d/e/rotations, so all checks are local and redundant):
//   1. spectrum: the eigenvalues of the produced tridiagonal (pstedc 'N')
//      match the eigenvalues of the gathered input (host ?SYEVD/?HEEVD) --
//      an end-to-end invariant since phe2hb's similarity preserves it;
//   2. reconstruction: building Q from the recorded rotation stream (the
//      verified forward column-mixing convention, plus the complex phases)
//      reconstructs the phe2hb output band, B = Q T Q^H -- the chase maps
//      the band to tridiagonal form, not the original dense input;
//   3. the stage-4 row-replay convention (phases, then the reverse-order
//      two-row rotations) reproduces Q*Y for a probe matrix.
//
// The reconstruction metric is the eig_test_common-style
// ||A_rec - A||_max / (||A||_max * n * eps) with the standard < 50 bound;
// the chase accumulates ~sqrt(n*nbw) eps of backward error, which sits
// orders of magnitude below the bound at the tested sizes.
// ---------------------------------------------------------------------------

using namespace api_grid_test;

extern "C" {
void ssyevd_(const char *, const char *, const int *, float *, const int *, float *, float *, const int *,
             int *, const int *, int *);
void dsyevd_(const char *, const char *, const int *, double *, const int *, double *, double *, const int *,
             int *, const int *, int *);
void cheevd_(const char *, const char *, const int *, std::complex<float> *, const int *, float *,
             std::complex<float> *, const int *, float *, const int *, int *, const int *, int *);
void zheevd_(const char *, const char *, const int *, std::complex<double> *, const int *, double *,
             std::complex<double> *, const int *, double *, const int *, int *, const int *, int *);
}

namespace {

template <typename U> U conj_v2(U v) { return v; }
template <typename U> std::complex<U> conj_v2(std::complex<U> v) { return std::conj(v); }

template <typename T> T herm_input(int i, int j) {
    using rt = typename ddla::real_type<T>::type;
    const double re = 0.3 * ((i * 7 + j * 13) % 11) - 1.4 + 0.05 * ((i + j) % 3);
    const double im = 0.15 * ((i * 5 + j * 3) % 7) - 0.4;
    if (i == j)
        return T(static_cast<rt>(3.0 + 0.1 * i));
    if constexpr (std::is_same_v<T, rt>) {
        (void)im;
        if (i > j)
            return T(static_cast<rt>(re));
        return herm_input<T>(j, i);
    } else if (i > j) {
        return T(static_cast<rt>(re), static_cast<rt>(im));
    } else {
        return conj_v2(herm_input<T>(j, i));
    }
}

// host eigenvalues (ascending) of a Hermitian n x n column-major matrix;
// every rank computes redundantly (no communication)
template <typename T> std::vector<typename ddla::real_type<T>::type> host_eigh(std::vector<T> a, int n) {
    using rt = typename ddla::real_type<T>::type;
    // ?SYEVD/?HEEVD workspaces: WORK is complex for the complex routines,
    // RWORK is real; the query returns their sizes in WORK(1)/IWORK(1).
    std::vector<rt> w(n), rwork(1);
    std::vector<T> work(1);
    std::vector<int> iwork(1);
    int info = 0, lwork = -1, lrwork = -1, liwork = -1;
    if constexpr (std::is_same_v<T, float>) {
        ssyevd_("N", "U", &n, a.data(), &n, w.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
    } else if constexpr (std::is_same_v<T, double>) {
        dsyevd_("N", "U", &n, a.data(), &n, w.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        cheevd_("N", "U", &n, a.data(), &n, w.data(), work.data(), &lwork, rwork.data(), &lrwork,
                iwork.data(), &liwork, &info);
    } else {
        zheevd_("N", "U", &n, a.data(), &n, w.data(), work.data(), &lwork, rwork.data(), &lrwork,
                iwork.data(), &liwork, &info);
    }
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
        lwork = static_cast<int>(work[0]) + 64;
    else
        lwork = static_cast<int>(work[0].real()) + 64;
    liwork = std::max(iwork[0], 1);
    work.assign(static_cast<std::size_t>(lwork), T(0));
    iwork.assign(static_cast<std::size_t>(liwork), 0);
    if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, double>) {
        lrwork = std::max(1, 3 * n - 2);
        rwork.assign(static_cast<std::size_t>(lrwork), rt(0));
    }
    if constexpr (std::is_same_v<T, float>) {
        ssyevd_("N", "U", &n, a.data(), &n, w.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
    } else if constexpr (std::is_same_v<T, double>) {
        dsyevd_("N", "U", &n, a.data(), &n, w.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        cheevd_("N", "U", &n, a.data(), &n, w.data(), work.data(), &lwork, rwork.data(), &lrwork,
                iwork.data(), &liwork, &info);
    } else {
        zheevd_("N", "U", &n, a.data(), &n, w.data(), work.data(), &lwork, rwork.data(), &lrwork,
                iwork.data(), &liwork, &info);
    }
    return w;
}

// Gather a distributed matrix's leading n x n block: every rank ends up
// with the full block (zero-fill + MPI_Allreduce).
template <typename T>
std::vector<T> gather_full(const ddla::DdlaHandle_t &handle, const T *d_A, const int *desc, int n) {
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
        if (gj >= n)
            continue;
        for (int i = 0; i < m_loc; ++i) {
            const int gi = ddla::indx_l2g_r(desc, handle, i);
            if (gi >= n)
                continue;
            full[gi + static_cast<std::size_t>(gj) * n] = local[i + static_cast<std::size_t>(j) * lld];
        }
    }
    MPI_Comm comm = ddlaGetCommunicator(handle);
    if constexpr (std::is_same_v<T, float>) {
        MPI_Allreduce(MPI_IN_PLACE, full.data(), n * n, MPI_FLOAT, MPI_SUM, comm);
    } else if constexpr (std::is_same_v<T, double>) {
        MPI_Allreduce(MPI_IN_PLACE, full.data(), n * n, MPI_DOUBLE, MPI_SUM, comm);
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        MPI_Allreduce(MPI_IN_PLACE, full.data(), n * n, MPI_C_FLOAT_COMPLEX, MPI_SUM, comm);
    } else {
        MPI_Allreduce(MPI_IN_PLACE, full.data(), n * n, MPI_C_DOUBLE_COMPLEX, MPI_SUM, comm);
    }
    return full;
}

template <typename T>
void check_phb2st(const ddla::DdlaHandle_t &handle, int n, int nblk, int nbw, const char *label) {
    using rt = typename ddla::real_type<T>::type;
    const bool is_cx = !std::is_same_v<T, rt>;
    const double eps = (std::is_same_v<rt, float>) ? 1.19e-7 : 2.22e-16;

    int descA[ddla::DDLA_DLEN_];
    ddlaDescInit(descA, handle, n, n, nblk, nblk, 0, 0);
    api_grid_test::DeviceBuffer<T> d_A(handle, local_size(handle, descA));

    const int ranks = ddlaGetSize(handle);
    if (ranks > 1 && n > 2 && nbw > ddla::eig_distributed_band_limit(n, ranks)) {
        std::vector<rt> diagonal(n), off(n);
        ddla::detail::DistributedHistory<T> history;
        std::vector<T> phases;
        int info = 0;
        bool caught = false;
        try {
            ddla::phb2st_node<ddla::default_backend_v, T>(handle, n, nbw, d_A.ptr, descA, diagonal.data(),
                                                          off.data(), history, phases, true, info);
        } catch (const std::invalid_argument &) {
            caught = true;
        }
        require_close(handle, "phb2st rejects a replicated wide band", caught ? 0 : 1, 0);
        return;
    }

    // full Hermitian input on the host, then scatter
    std::vector<T> Ah(static_cast<std::size_t>(n) * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            Ah[i + static_cast<std::size_t>(j) * n] = herm_input<T>(i, j);
    const std::vector<rt> ref_eigs = host_eigh<T>(Ah, n);
    double normA = 0.0;
    for (const T v : Ah)
        normA = std::max(normA, static_cast<double>(std::abs(v)));

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
            if (gj >= n)
                continue;
            for (int il = 0; il < m_loc; ++il) {
                const int gi = ddla::indxl2g(il, nblk, myprow, 0, nprows);
                if (gi >= n)
                    continue;
                local[il + static_cast<std::size_t>(jl) * lld] = Ah[gi + static_cast<std::size_t>(gj) * n];
            }
        }
        upload<T>(handle, d_A.ptr, local);
    }
    check_ddla_sync(handle);

    // stage 1: full -> band
    const int nsteps = (n - 1) / nbw;
    api_grid_test::DeviceBuffer<T> d_tmat(handle,
                                          static_cast<std::size_t>(nbw) * nbw * std::max(nsteps, 1) + 1);
    int info1 = 0;
    ddla::phe2hb(handle, n, d_A.ptr, descA, nbw, d_tmat.ptr, info1);
    require_close(handle, std::string(label) + " phe2hb info", info1, 0.5);

    // the phe2hb output band, gathered (the chase consumes exactly the
    // upper-band entries A(i,j), j - i <= nbw)
    const std::vector<T> band_full = gather_full<T>(handle, d_A.ptr, descA, n);

    // stage 2: band -> tridiagonal with recorded rotations
    std::vector<rt> d_eigs(n), e_off(n);
    ddla::detail::DistributedHistory<T> history;
    std::vector<T> phases;
    int info2 = 0;
    ddla::phb2st_node<ddla::default_backend_v, T>(handle, n, nbw, d_A.ptr, descA, d_eigs.data(), e_off.data(),
                                                  history, phases, true, info2);
    require_close(handle, std::string(label) + " phb2st info", info2, 0.5);

    // ---- check 1: spectrum ----------------------------------------------
    {
        std::vector<rt> dd(d_eigs), ee(e_off), teigs(n);
        int info3 = 0;
        ddla::pstedc<rt>(handle, 'N', n, dd.data(), ee.data(), teigs.data(), static_cast<rt *>(nullptr),
                         static_cast<const int *>(nullptr), info3);
        require_close(handle, std::string(label) + " pstedc info", info3, 0.5);
        double dev = 0.0;
        for (int i = 0; i < n; ++i)
            dev = std::max(dev, std::fabs(static_cast<double>(teigs[i] - ref_eigs[i])));
        const double metric = dev / (normA * eps);
        // The metric accumulates the backward error of BOTH reduction stages
        // (phe2hb's Householder panels plus this chase); that grows like
        // sqrt(n*nbw) eps rather than n eps, so the bound is set at 8 n-eps
        // units -- observed values sit near 1 for double and a few for float.
        require_close(handle, std::string(label) + " spectrum", metric / n, 8.0);
    }

    // ---- check 2: reconstruction B = Q T Q^H -----------------------------
    {
        // dense Hermitian view of the phe2hb output band: the upper-band
        // entries mirrored, zero beyond the band (what the chase consumed)
        std::vector<T> Bm(static_cast<std::size_t>(n) * n, T(0));
        for (int j = 0; j < n; ++j)
            for (int i = std::max(0, j - nbw); i <= j; ++i) {
                const T v = band_full[i + static_cast<std::size_t>(j) * n];
                if constexpr (std::is_same_v<T, rt>) {
                    Bm[i + static_cast<std::size_t>(j) * n] = v;
                    Bm[j + static_cast<std::size_t>(i) * n] = v;
                } else if (i == j) {
                    Bm[i + static_cast<std::size_t>(j) * n] = T(v.real(), rt(0));
                } else {
                    Bm[i + static_cast<std::size_t>(j) * n] = v;
                    Bm[j + static_cast<std::size_t>(i) * n] = std::conj(v);
                }
            }
        double normB = 0.0;
        for (const T v : Bm)
            normB = std::max(normB, static_cast<double>(std::abs(v)));

        std::vector<T> Q(static_cast<std::size_t>(n) * n, T(0));
        for (int i = 0; i < n; ++i)
            Q[i + static_cast<std::size_t>(i) * n] = T(1);
        std::vector<ddla::hb2st_rot<T>> chunk;
        for (long long q = 0; q < history.chunks(); ++q) {
            history.fetch(q, ddlaGetCommunicator(handle), chunk);
            for (const auto &rec : chunk) {
                const int a = rec.j0 - 1, b = rec.j0;
                if (a < 0 || b >= n)
                    continue;
                for (int r = 0; r < n; ++r) {
                    T &xa = Q[r + static_cast<std::size_t>(a) * n];
                    T &xb = Q[r + static_cast<std::size_t>(b) * n];
                    const T ta = xa;
                    xa = rec.c * ta + rec.s * xb;
                    if constexpr (std::is_same_v<T, rt>)
                        xb = rec.c * xb - rec.s * ta;
                    else
                        xb = rec.c * xb - std::conj(rec.s) * ta;
                }
            }
        }
        if constexpr (!std::is_same_v<T, rt>) {
            for (int i = 0; i < n - 1; ++i) {
                const T ph = phases[i];
                for (int r = 0; r < n; ++r)
                    Q[r + static_cast<std::size_t>(i + 1) * n] *= ph;
            }
        }
        std::vector<T> Tm(static_cast<std::size_t>(n) * n, T(0));
        for (int i = 0; i < n; ++i)
            Tm[i + static_cast<std::size_t>(i) * n] = T(d_eigs[i]);
        for (int i = 0; i + 1 < n; ++i) {
            Tm[(i + 1) + static_cast<std::size_t>(i) * n] = T(e_off[i]);
            if constexpr (std::is_same_v<T, rt>)
                Tm[i + static_cast<std::size_t>(i + 1) * n] = T(e_off[i]);
            else
                Tm[i + static_cast<std::size_t>(i + 1) * n] = std::conj(T(e_off[i]));
        }
        // A_rec = Q * T * Q^H, two passes
        std::vector<T> QT(static_cast<std::size_t>(n) * n, T(0));
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                T acc{};
                for (int k = 0; k < n; ++k)
                    acc += Q[i + static_cast<std::size_t>(k) * n] * Tm[k + static_cast<std::size_t>(j) * n];
                QT[i + static_cast<std::size_t>(j) * n] = acc;
            }
        double err = 0.0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                T acc{};
                for (int k = 0; k < n; ++k) {
                    T qh = Q[j + static_cast<std::size_t>(k) * n];
                    if constexpr (!std::is_same_v<T, rt>)
                        qh = std::conj(qh);
                    acc += QT[i + static_cast<std::size_t>(k) * n] * qh;
                }
                err = std::max(err,
                               static_cast<double>(std::abs(acc - Bm[i + static_cast<std::size_t>(j) * n])));
            }
        require_close(handle, std::string(label) + " reconstruction", err / (normB * n * eps), 50.0);
    }

    // ---- check 3: the stage-4 row-replay convention ----------------------
    {
        const int ncol = 3;
        // probe block Y (n x ncol, column-major with stride n): the rows are
        // the coordinates the rotations mix
        std::vector<T> Yv(static_cast<std::size_t>(n) * ncol);
        for (int c = 0; c < ncol; ++c)
            for (int i = 0; i < n; ++i)
                Yv[i + static_cast<std::size_t>(c) * n] =
                    T(static_cast<rt>(-1.0 + 0.01 * static_cast<double>((i + 13 * c) % 37)));
        // reference: X = Q * Y with the column-built Q (as in check 2)
        std::vector<T> X(static_cast<std::size_t>(n) * ncol, T(0));
        // build Q again (same as check 2)
        std::vector<T> Q(static_cast<std::size_t>(n) * n, T(0));
        for (int i = 0; i < n; ++i)
            Q[i + static_cast<std::size_t>(i) * n] = T(1);
        std::vector<ddla::hb2st_rot<T>> chunk;
        for (long long q = 0; q < history.chunks(); ++q) {
            history.fetch(q, ddlaGetCommunicator(handle), chunk);
            for (const auto &rec : chunk) {
                const int a = rec.j0 - 1, b = rec.j0;
                if (a < 0 || b >= n)
                    continue;
                for (int r = 0; r < n; ++r) {
                    T &xa = Q[r + static_cast<std::size_t>(a) * n];
                    T &xb = Q[r + static_cast<std::size_t>(b) * n];
                    const T ta = xa;
                    xa = rec.c * ta + rec.s * xb;
                    if constexpr (std::is_same_v<T, rt>)
                        xb = rec.c * xb - rec.s * ta;
                    else
                        xb = rec.c * xb - std::conj(rec.s) * ta;
                }
            }
        }
        if constexpr (!std::is_same_v<T, rt>) {
            // phases multiply Q's columns (Q = R * D); the replay applies the
            // same factors to the probe rows first, giving R * (D * Y)
            for (int i = 0; i < n - 1; ++i) {
                const T ph = phases[i];
                for (int r = 0; r < n; ++r)
                    Q[r + static_cast<std::size_t>(i + 1) * n] *= ph;
            }
        }
        for (int c = 0; c < ncol; ++c)
            for (int i = 0; i < n; ++i) {
                T acc{};
                for (int k = 0; k < n; ++k)
                    acc += Q[i + static_cast<std::size_t>(k) * n] * Yv[k + static_cast<std::size_t>(c) * n];
                X[i + static_cast<std::size_t>(c) * n] = acc;
            }
        // replay: phases first, then the reverse-order two-row rotations.
        // Each rotation mixes coordinate rows (a, b) across every probe column
        // (stride n), exactly what punmtr_hb2st applies to the eigenvector
        // rows.
        std::vector<T> X2(Yv);
        if constexpr (!std::is_same_v<T, rt>) {
            for (int i = 0; i < n - 1; ++i) {
                const T ph = phases[i];
                for (int c = 0; c < ncol; ++c)
                    X2[(i + 1) + static_cast<std::size_t>(c) * n] *= ph;
            }
        }
        for (long long q = history.chunks(); q-- > 0;) {
            history.fetch(q, ddlaGetCommunicator(handle), chunk);
            for (auto it = chunk.rbegin(); it != chunk.rend(); ++it) {
                const int a = it->j0 - 1, b = it->j0;
                if (a < 0 || b >= n)
                    continue;
                for (int c = 0; c < ncol; ++c) {
                    T &xa = X2[a + static_cast<std::size_t>(c) * n];
                    T &xb = X2[b + static_cast<std::size_t>(c) * n];
                    const T ta = xa;
                    if constexpr (std::is_same_v<T, rt>) {
                        xa = it->c * ta - it->s * xb;
                        xb = it->s * ta + it->c * xb;
                    } else {
                        xa = it->c * ta - std::conj(it->s) * xb;
                        xb = it->s * ta + it->c * xb;
                    }
                }
            }
        }
        double err = 0.0;
        for (std::size_t p = 0; p < X.size(); ++p)
            err = std::max(err, static_cast<double>(std::abs(X[p] - X2[p])));
        require_close(handle, std::string(label) + " row replay", err / (normA * n * eps), 50.0);
    }
}

template <typename T> void check_body_t(const ddla::DdlaHandle_t &handle, const Shape &base) {
    // a small case inside the regime the chase transcription was validated
    // against the reference implementation (n=12, kd=3)
    check_phb2st<T>(handle, std::max(12, 3 * ddlaGetSize(handle) + 1), 3, 3, "small");
    // many active bulges: the chase's vectorized apply-left branch (the
    // ?LARTV path) runs only while nr > 2*kd-1 bulges are active, i.e. for
    // n > ~2*kd^2, which no other case here reaches.  It carried a band-row
    // off-by-one that broke eig(T) at every such size (n >= 700 at kd=16,
    // n = 10000 at kd=64) while all the small cases passed.
    check_phb2st<T>(handle, 60, 3, 3, "many-bulges kd=3");
    check_phb2st<T>(handle, 150, 5, 5, "many-bulges kd=5");
    // production nbw (eig_select_params: divides nb) at a size with at least two panels
    {
        const ddla::EigParams params =
            ddla::eig_select_params(!std::is_same_v<T, typename ddla::real_type<T>::type>, base.nb, false);
        const int nbw = params.nbw;
        const int n = std::max(2 * nbw + base.nb + 3, ddlaGetSize(handle) * nbw + 1);
        check_phb2st<T>(handle, n, base.nb, nbw, "prod-nbw");
    }
    // a smaller band width and a degenerate n <= nbw case
    {
        const int nbw = base.nb;
        const int n = std::max(3 * nbw + 5, ddlaGetSize(handle) * nbw + 1);
        check_phb2st<T>(handle, n, base.nb, nbw, "nbw=nb");
    }
    check_phb2st<T>(handle, base.nb, base.nb, base.nb, "n=nb");
    // larger bands (the chase's kd = 16 .. 64 regime) need a block size they divide: nbw | nb
    check_phb2st<T>(handle, 200, 64, 16, "nb=64,nbw=16");
    check_phb2st<T>(handle, 330, 64, 64, "nb=64,nbw=nb=64");
    check_phb2st<T>(handle, 200, 32, 8, "nb=32,nbw=8");
    check_phb2st<T>(handle, 150, 12, 4, "nb=12,nbw=4");
}

} // namespace

static void check_body(const ddla::DdlaHandle_t &handle, const Shape &base) {
    check_body_t<double>(handle, base);
    check_body_t<Complex>(handle, base);
    // Single precision is compiled into the library for both backends, so it
    // is exercised here too.
    check_body_t<float>(handle, base);
    check_body_t<std::complex<float>>(handle, base);
}

int main(int argc, char **argv) { return run_grid_test(argc, argv, "test_api_grid_phb2st", check_body); }
