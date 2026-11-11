#include <cstring>
#include <type_traits>

#include "api_grid_test_common.h"
#include "Eigensolver/eig_params.h"
#include "eig_test_common.h"
#include "phe2hb_test_common.h"

// ---------------------------------------------------------------------------
// Isolation test for stage 1 (phe2hb) and stage 5 (punmtr_he2hb).
//
// Stage 1: reduce a full Hermitian matrix to banded form, reconstruct
// A = Q^H B Q on the host from the banded part and the stored reflectors, and
// compare against the input.
//
// Stage 5: run both transform directions on the identity and check them
// against the host-built Q and Q^H, then check P B P^H = A for each.
//
// Output contract exercised:
//   - upper triangle: band entries at rows [j-nbw, j] plus the reflector
//     vector at rows < j-nbw (anchor 1 at row j-nbw implicit);
//   - d_tmat: replicated nbw x nbw x nsteps slots, diagonal = raw tau.
//
// Tolerances follow the `50 * n * eps` convention documented in
// eig_test_common.h.  The previously fixed 1e-11 (5e-4 for float) sat ~10^5
// above the observed error, so a five-orders-of-magnitude regression would
// have stayed green; the observed errors are ~2-3 n eps.
// ---------------------------------------------------------------------------

using namespace api_grid_test;

namespace {


template <typename T>
T herm_input(int i, int j)
{
    using rt = typename ddla::real_type<T>::type;
    const double re = 0.3 * ((i * 7 + j * 13) % 11) - 1.4 + 0.05 * ((i + j) % 3);
    const double im = 0.15 * ((i * 5 + j * 3) % 7) - 0.4;
    if (i == j) return T(static_cast<rt>(3.0 + 0.1 * i));   // real, positive-ish diagonal
    if constexpr (std::is_same_v<T, rt>) {
        (void)im;
        if (i > j) return T(static_cast<rt>(re));
        return herm_input<T>(j, i);   // mirror: real input must be symmetric
    } else if (i > j) {
        return T(static_cast<rt>(re), static_cast<rt>(im));
    } else {
        return conj_v2(herm_input<T>(j, i));   // mirror: A(i,j) = conj(A(j,i))
    }
}

template <typename T>
double max_abs_diff(const std::vector<T>& X, const std::vector<T>& Y)
{
    double err = 0.0;
    for (std::size_t k = 0; k < X.size(); ++k)
        err = std::max(err, static_cast<double>(std::abs(X[k] - Y[k])));
    return err;
}

// One full stage-1 + stage-5 case at the given geometry.
template <typename T>
void run_case(const ddla::DdlaHandle_t& handle, int n, int nb, int nbw, const std::string& tag)
{
    using rt = typename ddla::real_type<T>::type;
    // Backward-error metric in the units eig_test_common.h documents:
    //   ||X - A||_max / (||A||_max * n * eps)   accept < 50
    // The previous fixed 1e-11 was compared against ||X - A||/(||A||_max * n)
    // only, i.e. it was ~1/(n*eps) times looser than this; a regression of
    // five orders of magnitude would have passed unnoticed.
    const double tol = 50.0;
    const double eps = eig_test::eps_trait<rt>::value;

    int desc[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(desc, handle, n, n, nb, nb, 0, 0));

    auto h_A = make_local<T>(handle, desc, [](int i, int j) { return herm_input<T>(i, j); });
    DeviceBuffer<T> d_A(handle, h_A.size());
    upload(handle, d_A.ptr, h_A);
    check_ddla_sync(handle);

    const int nsteps = (n - 1) / nbw;
    DeviceBuffer<T> d_tmat(handle, static_cast<std::size_t>(nbw) * nbw * std::max(nsteps, 1));
    {   // poison the T slots: phe2hb must overwrite every entry (zeros included)
        const std::vector<T> poison(static_cast<std::size_t>(nbw) * nbw * std::max(nsteps, 1), T(7));
        upload(handle, d_tmat.ptr, poison);
        check_ddla_sync(handle);
    }
    int info = 0;
    ddla::phe2hb(handle, n, d_A.ptr, desc, nbw, d_tmat.ptr, info);
    check_ddla_sync(handle);
    if (info != 0) {
        if (ddlaGetRank(handle) == 0) std::cerr << "phe2hb info=" << info << std::endl;
        MPI_Abort(ddlaGetCommunicator(handle), 1);
    }

    const auto Aout = gather_full<T>(handle, d_A.ptr, desc, n);
    const auto tmat = download<T>(handle, d_tmat.ptr,
                                  static_cast<std::size_t>(nbw) * nbw * std::max(nsteps, 1));

    // ---- the T slots: upper triangular, zero below the diagonal and outside the panel's
    // n_cols x n_cols block (the panel step stores T_L^H into the zeroed slot), real diagonal
    // magnitude <= 2 (tau = (beta - alpha)/beta has |tau - 1| <= 1, so |tau| <= 2) ----------
    {
        double stray = 0.0, diag_bad = 0.0;
        for (int is = 1; is <= nsteps; ++is) {
            const int ps = is * nbw;
            const int ncols = std::min(n, ps + nbw) - ps;
            const T* slot = tmat.data() + static_cast<std::size_t>(is - 1) * nbw * nbw;
            for (int c = 0; c < nbw; ++c)
                for (int r = 0; r < nbw; ++r) {
                    const double v = std::abs(slot[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * nbw]);
                    if (r > c || r >= ncols || c >= ncols) stray = std::max(stray, v);
                    else if (r == c) diag_bad = std::max(diag_bad, v > 2.0 + 1e-6 ? 1.0 : 0.0);
                }
        }
        require_close(handle, "phe2hb T slots zero outside the upper triangle " + tag, stray, 0.0);
        require_close(handle, "phe2hb T diagonal is a tau " + tag, diag_bad, 0.0);
    }

    // ---- host reconstruction: A_pre = Q^H B Q -----------------------------
    // Root-only: B, Q, A_in and the O(n^3) host products below read only the
    // replicated gather_full output, so computing them on every rank repeats
    // identical dense work P times -- the dominant share of this test's check
    // phase.  The stage-5 punmtr/gather calls stay collective on all ranks;
    // only their dense comparisons move inside the root-only verdicts.
    auto idx = [n](int i, int j) { return i + static_cast<std::size_t>(j) * n; };
    // Dense check inputs; filled on the root rank only (RootSection below):
    // every input reads only the replicated gather_full output, so computing
    // them on every rank repeats identical dense work P times -- the dominant
    // share of this test's check phase.
    std::vector<T> B, Q, A_in;
    double denom = 0.0;
    RootSection dense(handle);
    if (dense.root) {
    // B = banded part of A_out (upper band mirrored to a full Hermitian matrix).
    B.assign(static_cast<std::size_t>(n) * n, T(0));
    for (int j = 0; j < n; ++j)
        for (int i = std::max(0, j - nbw); i <= j; ++i)
            B[idx(i, j)] = Aout[idx(i, j)];
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < j; ++i)
            if (j - i <= nbw) B[idx(j, i)] = conj_v2(B[idx(i, j)]);

    Q = build_host_Q(Aout, tmat, n, nbw, nsteps);

    // The input matrix and the norm scale for all three checks below.
    A_in.assign(static_cast<std::size_t>(n) * n, T(0));
    double scale = 0.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            A_in[idx(i, j)] = herm_input<T>(i, j);
            scale = std::max(scale, static_cast<double>(std::abs(A_in[idx(i, j)])));
        }
    denom = std::max(scale, 1.0) * n * eps;

    // A_rec = Q^H * B * Q
    {
        const auto A_rec = host_mul<T>(host_mul_conjA<T>(Q, B, n), Q, n);
        dense.check("phe2hb reconstruction " + tag,
                      max_abs_diff(A_rec, A_in) / denom, tol);
    }
    }

    // ---- stage 5: transform the identity in both directions ---------------
    int descZ[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descZ, handle, n, n, nb, nb, 0, 0));
    auto h_I = make_local<T>(handle, descZ, [](int i, int j) { return T(i == j ? 1 : 0); });

    // trans='N' must give Q;  trans='C' must give Q^H.
    for (int pass = 0; pass < 2; ++pass) {
        const char trans = (pass == 0) ? 'N' : 'C';
        const bool adjoint = (trans != 'N');
        DeviceBuffer<T> d_Z(handle, h_I.size());
        upload(handle, d_Z.ptr, h_I);
        check_ddla_sync(handle);
        int info5 = 0;
        ddla::punmtr_he2hb(handle, trans, n, nbw, d_A.ptr, desc, d_tmat.ptr,
                           d_Z.ptr, descZ, info5);
        check_ddla_sync(handle);
        if (info5 != 0) {
            if (ddlaGetRank(handle) == 0)
                std::cerr << "punmtr_he2hb('" << trans << "') info=" << info5 << std::endl;
            MPI_Abort(ddlaGetCommunicator(handle), 1);
        }
        const auto P = gather_full<T>(handle, d_Z.ptr, descZ, n);

        // P should equal the host-built Q, or its adjoint (dense work is
        // root-only; Q and B exist only there).
        if (dense.root) {
        const std::vector<T> Qref = adjoint ? host_adjoint<T>(Q, n) : Q;
        dense.check(std::string("stage5 '") + trans + "' P = " + (adjoint ? "Q^H" : "Q"),
                      max_abs_diff(P, Qref) / denom, tol);

        // The factorization is A = Q^H B Q, so applying the stage-5 transform
        // to B recovers A in both directions, just from opposite sides: with
        // P = Q (trans='N') the identity is P^H B P = A, and with P = Q^H
        // (trans='C') it is P B P^H = A.  This is exactly the reduction the
        // eigensolver driver relies on, so the two cases are checked
        // separately rather than collapsed into one symmetric-looking form.
        const auto A_chk = adjoint ? host_mul_conjB<T>(host_mul<T>(P, B, n), P, n)
                                   : host_mul<T>(host_mul_conjA<T>(P, B, n), P, n);
        const char* expr = adjoint ? "P B P^H = A" : "P^H B P = A";
        dense.check(std::string("stage5 '") + trans + "' " + expr,
                      max_abs_diff(A_chk, A_in) / denom, tol);
        }
    }
    dense.finish();
}

// phe2hb and punmtr_he2hb must REFUSE an nbw that does not divide nb (nbw > nb, nb % nbw != 0,
// nbw = 0): every rank throws before any collective.
template <typename T>
void check_refusals(const ddla::DdlaHandle_t& handle, int nb)
{
    const int n = 4 * nb + 3;
    int desc[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(desc, handle, n, n, nb, nb, 0, 0));
    auto h_A = make_local<T>(handle, desc, [](int i, int j) { return herm_input<T>(i, j); });
    DeviceBuffer<T> d_A(handle, h_A.size());
    upload(handle, d_A.ptr, h_A);
    DeviceBuffer<T> d_tmat(handle, static_cast<std::size_t>(4) * nb * nb + 16);
    DeviceBuffer<T> d_Z(handle, h_A.size());
    upload(handle, d_Z.ptr, h_A);
    check_ddla_sync(handle);
    std::vector<int> bad = {2 * nb, nb + 1, 0};
    if (nb > 2 && nb % (nb - 1) != 0) bad.push_back(nb - 1);   // does not divide nb
    for (const int nbw : bad) {
        double threw1 = 0.0, threw5 = 0.0;
        int info = 0;
        try { ddla::phe2hb(handle, n, d_A.ptr, desc, nbw, d_tmat.ptr, info); }
        catch (const std::exception&) { threw1 = 1.0; }
        try { ddla::punmtr_he2hb(handle, 'C', n, nbw, d_A.ptr, desc, d_tmat.ptr, d_Z.ptr, desc, info); }
        catch (const std::exception&) { threw5 = 1.0; }
        check_ddla_sync(handle);
        // refused UP FRONT: neither A nor Z was touched (a check that only fired mid-call would
        // have factored some panels first)
        const auto a_after = download<T>(handle, d_A.ptr, h_A.size());
        const auto z_after = download<T>(handle, d_Z.ptr, h_A.size());
        double mism = 0.0;
        for (std::size_t i = 0; i < h_A.size(); ++i) {
            if (std::memcmp(&a_after[i], &h_A[i], sizeof(T)) != 0) mism += 1.0;
            if (std::memcmp(&z_after[i], &h_A[i], sizeof(T)) != 0) mism += 1.0;
        }
        const std::string tag = "sizeof=" + std::to_string(sizeof(T)) + " nb=" + std::to_string(nb) + " nbw=" + std::to_string(nbw);
        require_close(handle, "phe2hb refuses " + tag, 1.0 - threw1, 0.0);
        require_close(handle, "punmtr_he2hb refuses " + tag, 1.0 - threw5, 0.0);
        require_close(handle, "refusal leaves A and Z untouched " + tag, mism, 0.0);
    }
}

template <typename T>
void check_phe2hb(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    using rt = typename ddla::real_type<T>::type;
    const bool is_complex = !std::is_same<T, rt>::value;
    const int nb = base.nb;

    // The production nbw -- what the GPU pheevd driver would actually pass
    // (eig_select_params: 16 when nb is a multiple of 16, otherwise nbw = nb).
    {
        const ddla::EigParams params = ddla::eig_select_params(is_complex, nb, true);
        const int nbw = params.nbw;
        const int n = 2 * nbw + nb + 3;   // >= 2 panels
        run_case<T>(handle, n, nb, nbw, "prod-nbw");
    }

    // nbw always divides nb (a panel is ONE block column): nbw = nb, nb/2, nb/4 where they
    // are representable.  With the default shapes (nb = 3, 4) nb/4 degenerates below the
    // nbw >= 2 floor and is skipped here; nbw = 1 runs in the dedicated cases below.
    struct NbwCase { int den; const char* label; };
    const NbwCase nbw_cases[] = { {1, "nbw=nb"}, {2, "nbw=nb/2"}, {4, "nbw=nb/4"} };
    for (const NbwCase& c : nbw_cases) {
        if (nb % c.den != 0) continue;        // nb/2 needs even nb, nb/4 % 4
        const int nbw = nb / c.den;
        if (nbw < 2) continue;                // degenerate band width
        const int n = 3 * nbw + 5 * nb + 7;   // non-multiple sizes on purpose
        run_case<T>(handle, n, nb, nbw, c.label);
    }

    // nbw < nb at production-like ratios and sizes, independent of the base shape:
    // n spans several block columns, so the panel owner's process column changes only every
    // nb/nbw panels, plus a partial last block and a partial last panel.  nb = 64 / 96 / 65
    // are the block sizes the eigensolver is tuned for (65: odd, nbw = 5 and 13).
    struct DivGeom { int nb; int nbw; int blocks; const char* label; };
    const DivGeom div_geoms[] = {
        {8, 4, 5, "nb=8,nbw=4"}, {16, 4, 5, "nb=16,nbw=4"}, {16, 2, 5, "nb=16,nbw=2"},
        {12, 3, 5, "nb=12,nbw=3"},
        {64, 16, 3, "nb=64,nbw=16"}, {64, 64, 3, "nb=64,nbw=nb"}, {64, 8, 2, "nb=64,nbw=8"},
        {96, 32, 3, "nb=96,nbw=32"}, {65, 5, 3, "nb=65,nbw=5"}, {65, 13, 3, "nb=65,nbw=13"},
    };
    for (const DivGeom& g : div_geoms) {
        const int n = g.blocks * g.nb + 3 * g.nbw + 1;   // partial block, partial panel
        run_case<T>(handle, n, g.nb, g.nbw, g.label);
    }

    // The degenerate band widths: nbw = 1 (tridiagonal "band": one column per panel) and 2.
    run_case<T>(handle, 30, 1, 1, "nb=1,nbw=1");
    run_case<T>(handle, 29, 2, 2, "nb=2,nbw=2");
    run_case<T>(handle, 31, 4, 1, "nb=4,nbw=1");

    // the phb2st test's small case (n=12, nbw=3) -- cross-verification
    run_case<T>(handle, 12, 3, 3, "small-cross");
    // Degenerate geometries: the `nsteps` clamp branches the old test never
    // took (n <= nbw gives nsteps == 0, i.e. a no-op reduction).
    run_case<T>(handle, 1, nb, nb, "n=1");
    run_case<T>(handle, nb, nb, nb, "n=nb=nbw");
    run_case<T>(handle, nb - 1, nb, nb, "n<nbw");
    run_case<T>(handle, nb + nb / 2, nb, nb, "nbw<n<2nbw");

    check_refusals<T>(handle, nb);
}

static void check_body(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    check_phe2hb<double>(handle, base);
    check_phe2hb<Complex>(handle, base);
    // Single precision is compiled into the library for both backends, so it
    // is exercised here too (this requires gather_full's real element count).
    check_phe2hb<float>(handle, base);
    check_phe2hb<std::complex<float>>(handle, base);
}

} // namespace

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_phe2hb", check_body);
}
