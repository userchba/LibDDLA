// ---------------------------------------------------------------------------
// test_backend_solvers: the distributed LU / Cholesky / triangular-solve family and its helpers, called with an EXPLICIT
// backend on every backend this build contains (CPU, GPU, or both in a dual build) and all four scalar types.
//
// The api_grid tests call the routines bare, i.e. with the build's default backend, so in a dual build they only ever
// reach the GPU.  This test is what proves that a CPU handle takes the CPU arms (host BLAS/LAPACK, MPI communication)
// of the same sources, and that the backend named in the template must be the handle's.
// ---------------------------------------------------------------------------
#include "api_grid_test_common.h"
#include "LU/lu_internal.h"

#include <stdexcept>
#include <type_traits>

using namespace api_grid_test;

namespace {

template <typename T> struct is_cplx : std::false_type {};
template <typename U> struct is_cplx<std::complex<U>> : std::true_type {};

// The harness generators produce std::complex<double>; real types keep the real part (every generator is built so that
// the real part alone is a valid matrix: strictly dominant, symmetric positive definite, triangular, ...).
template <typename T>
T to_T(const Complex& z)
{
    if constexpr (is_cplx<T>::value) {
        using R = typename T::value_type;
        return T(static_cast<R>(z.real()), static_cast<R>(z.imag()));
    } else {
        return static_cast<T>(z.real());
    }
}

template <typename T> double tolerance() { return std::is_same<typename ddla::real_type<T>::type, float>::value ? 2e-3 : 5e-9; }

using GenFn = Complex (*)(int, int, int);

// B := op(A) X  (side 'L', B is n x nrhs)  or  X op(A)  (side 'R', B is nrhs x n), computed in T.
template <typename T>
std::vector<T> rhs_for(const DdlaHandle_t& h, const int* descB, int n, char side, char trans, GenFn gen, int tag)
{
    return make_local<T>(h, descB, [&](int i, int j) {
        T sum = T(0);
        for (int l = 0; l < n; ++l) {
            if (side == 'L') sum += to_T<T>(op_value(trans, n, n, i, l, gen, tag)) * to_T<T>(x_value(l, j));
            else             sum += to_T<T>(x_value(i, l)) * to_T<T>(op_value(trans, n, n, l, j, gen, tag));
        }
        return sum;
    });
}

template <typename T>
void check_x(const DdlaHandle_t& h, const int* descB, const T* d_B, size_t count, const std::string& name, double tol)
{
    auto out = download(h, d_B, count);
    const double err = local_max_error<T>(h, descB, out, [](int i, int j) { return to_T<T>(x_value(i, j)); });
    require_close(h, name, err, tol);
}

std::string tag_of(DdlaBackend b, const char* routine, const char* type, const std::string& what = "")
{
    return std::string(b == DdlaBackend::CPU ? "[CPU] " : "[GPU] ") + routine + "<" + type + ">" + (what.empty() ? "" : " " + what);
}

template <typename T> const char* type_name()
{
    if (std::is_same<T, float>::value) return "float";
    if (std::is_same<T, double>::value) return "double";
    if (std::is_same<T, std::complex<float>>::value) return "cfloat";
    return "cdouble";
}

// ---- LU, partial pivoting / no pivoting / block pivoting -------------------------------------------------------------
template <DdlaBackend B, typename T>
void check_lu(const DdlaHandle_t& h, int n, int nb, int nrhs, bool all_variants)
{
    int descA[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, h, n, n, nb, nb, 0, 0));
    const char* ty = type_name<T>();
    const double tol = tolerance<T>();

    struct Case { char side; char trans; };
    const Case cases[] = {{'L', 'N'}, {'L', 'C'}, {'R', 'N'}, {'R', 'T'}};

    for (GenFn gen : {GenFn(dominant_value), GenFn(cycled_value)}) {
        const bool pivoting = (gen == GenFn(cycled_value));
        for (const Case& c : cases) {
            if (!all_variants && !(c.side == 'L' && c.trans == 'N') && !(c.side == 'R' && c.trans == 'C')) continue;
            const int b_rows = (c.side == 'L') ? n : nrhs;
            const int b_cols = (c.side == 'L') ? nrhs : n;
            int descB[DDLA_DLEN_];
            DDLA_CHECK(ddlaDescInit(descB, h, b_rows, b_cols, nb, nb, 0, 0));
            const std::string what = std::string(pivoting ? "cycled " : "dominant ") + c.side + "," + c.trans;
            auto h_A = make_local<T>(h, descA, [&](int i, int j) { return to_T<T>(gen(i, j, n)); });
            auto h_B = rhs_for<T>(h, descB, n, c.side, c.trans, gen, n);

            auto fresh = [&](DeviceBuffer<T>& dA, DeviceBuffer<T>& dB) {
                upload(h, dA.ptr, h_A);
                upload(h, dB.ptr, h_B);
                check_ddla_sync(h);
            };

            {   // pgetrf + pgetrs
                DeviceBuffer<T> dA(h, h_A.size()), dB(h, h_B.size());
                fresh(dA, dB);
                std::vector<int> ipiv(ddla_test::m_loc(h, descA) + 1, 0);
                int info = -1;
                ddla::pgetrf<B>(h, n, n, dA.ptr, 1, 1, descA, ipiv.data(), info);
                if (info != 0) MPI_Abort(ddlaGetCommunicator(h), 1);
                ddla::pgetrs<B>(h, c.side, c.trans, n, nrhs, dA.ptr, descA, ipiv.data(), dB.ptr, descB);
                check_x<T>(h, descB, dB.ptr, h_B.size(), tag_of(B, "pgetrf+pgetrs", ty, what), tol);
            }
            {   // pgesv
                DeviceBuffer<T> dA(h, h_A.size()), dB(h, h_B.size());
                fresh(dA, dB);
                ddla::pgesv<B>(h, c.side, c.trans, n, nrhs, dA.ptr, descA, dB.ptr, descB);
                check_x<T>(h, descB, dB.ptr, h_B.size(), tag_of(B, "pgesv", ty, what), tol);
            }
            if (!pivoting) {   // pgetrf_nopiv + pgetrs_nopiv / pgesv_nopiv (a pivoting matrix would need row swaps)
                DeviceBuffer<T> dA(h, h_A.size()), dB(h, h_B.size());
                fresh(dA, dB);
                int info = -1;
                ddla::pgetrf_nopiv<B>(h, n, n, dA.ptr, descA, info);
                if (info != 0) MPI_Abort(ddlaGetCommunicator(h), 1);
                ddla::pgetrs_nopiv<B>(h, c.side, c.trans, n, nrhs, dA.ptr, descA, dB.ptr, descB);
                check_x<T>(h, descB, dB.ptr, h_B.size(), tag_of(B, "pgetrf_nopiv+pgetrs_nopiv", ty, what), tol);

                fresh(dA, dB);
                ddla::pgesv_nopiv<B>(h, c.side, c.trans, n, nrhs, dA.ptr, descA, dB.ptr, descB);
                check_x<T>(h, descB, dB.ptr, h_B.size(), tag_of(B, "pgesv_nopiv", ty, what), tol);
            }
            {   // block pivoting: pgetrf_bpiv + pgetrs_bpiv, and the driver
                DeviceBuffer<T> dA(h, h_A.size()), dB(h, h_B.size());
                fresh(dA, dB);
                DeviceBuffer<int> d_ipiv(h, ddla_test::m_loc(h, descA) + 1);
                int info = -1;
                ddla::pgetrf_bpiv<B>(h, n, n, dA.ptr, descA, d_ipiv.ptr, info);
                if (info != 0) MPI_Abort(ddlaGetCommunicator(h), 1);
                ddla::pgetrs_bpiv<B>(h, c.side, c.trans, n, nrhs, dA.ptr, descA, d_ipiv.ptr, dB.ptr, descB);
                check_x<T>(h, descB, dB.ptr, h_B.size(), tag_of(B, "pgetrf_bpiv+pgetrs_bpiv", ty, what), tol);

                fresh(dA, dB);
                ddla::pgesv_bpiv<B>(h, c.side, c.trans, n, nrhs, dA.ptr, descA, dB.ptr, descB);
                check_x<T>(h, descB, dB.ptr, h_B.size(), tag_of(B, "pgesv_bpiv", ty, what), tol);
            }
        }
    }
}

// A zero pivot / a singular column reports info = (1-based index of the first zero pivot).
template <DdlaBackend B, typename T>
void check_lu_info(const DdlaHandle_t& h, int n, int nb)
{
    int descA[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, h, n, n, nb, nb, 0, 0));
    const int zero_at = std::min(nb + 1, n - 1);
    auto h_A = make_local<T>(h, descA, [=](int i, int j) { return (i != j) ? T(0) : T(i == zero_at ? 0 : 1); });
    const char* ty = type_name<T>();
    {
        DeviceBuffer<T> dA(h, h_A.size());
        upload(h, dA.ptr, h_A);
        check_ddla_sync(h);
        std::vector<int> ipiv(ddla_test::m_loc(h, descA) + 1, -1);
        int info = -1;
        ddla::pgetrf<B>(h, n, n, dA.ptr, 1, 1, descA, ipiv.data(), info);
        require_close(h, tag_of(B, "pgetrf", ty, "singular info"), std::abs(info - (zero_at + 1)), 0.0);
    }
    {
        DeviceBuffer<T> dA(h, h_A.size());
        upload(h, dA.ptr, h_A);
        check_ddla_sync(h);
        int info = -1;
        ddla::pgetrf_nopiv<B>(h, n, n, dA.ptr, descA, info);
        require_close(h, tag_of(B, "pgetrf_nopiv", ty, "zero pivot info"), std::abs(info - (zero_at + 1)), 0.0);
    }
}

// hpd_value with the diagonal entry at head_idx negated (the head element of ppotrf's head correction).
Complex head_value(int i, int j, int n, int head_idx)
{
    if (i == j) {
        const double diag = 5.0 + 0.2 * n + 0.05 * i;
        return Complex(i == head_idx ? -diag : diag, 0.0);
    }
    return hpd_value(i, j, n);
}

// ---- Cholesky --------------------------------------------------------------------------------------------------------
template <DdlaBackend B, typename T>
void check_cholesky(const DdlaHandle_t& h, int n, int nb, int nrhs)
{
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(h, nprows, npcols);
    if (nprows != npcols) return;   // ppotrf / ppotrs need a square grid
    int descA[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, h, n, n, nb, nb, 0, 0));
    const char* ty = type_name<T>();
    const double tol = tolerance<T>();
    auto h_A = make_local<T>(h, descA, [=](int i, int j) { return to_T<T>(hpd_value(i, j, n)); });

    for (char side : {'L', 'R'}) {
        for (char uplo : {'L', 'U'}) {
            const int b_rows = (side == 'L') ? n : nrhs;
            const int b_cols = (side == 'L') ? nrhs : n;
            int descB[DDLA_DLEN_];
            DDLA_CHECK(ddlaDescInit(descB, h, b_rows, b_cols, nb, nb, 0, 0));
            auto h_B = rhs_for<T>(h, descB, n, side, 'N', GenFn(hpd_value), n);
            DeviceBuffer<T> dA(h, h_A.size()), dB(h, h_B.size());
            upload(h, dA.ptr, h_A);
            upload(h, dB.ptr, h_B);
            check_ddla_sync(h);
            int info = -1;
            const bool is_nega = ddla::ppotrf<B>(h, uplo, n, dA.ptr, 1, 1, descA, info);
            if (info != 0 || is_nega) MPI_Abort(ddlaGetCommunicator(h), 1);
            ddla::ppotrs<B>(h, side, uplo, 'N', n, nrhs, dA.ptr, descA, dB.ptr, descB, is_nega);
            check_x<T>(h, descB, dB.ptr, h_B.size(),
                       tag_of(B, "ppotrf+ppotrs", ty, std::string(1, side) + "," + uplo), tol);

            // pposv does both
            upload(h, dA.ptr, h_A);
            upload(h, dB.ptr, h_B);
            check_ddla_sync(h);
            info = -1;
            ddla::pposv<B>(h, side, uplo, 'N', n, nrhs, dA.ptr, 1, 1, descA, dB.ptr, 1, 1, descB, info);
            if (info != 0) MPI_Abort(ddlaGetCommunicator(h), 1);
            check_x<T>(h, descB, dB.ptr, h_B.size(),
                       tag_of(B, "pposv", ty, std::string(1, side) + "," + uplo), tol);
        }
    }

    {   // a one-element head block skips the block factorization yet still reports its info
        int tiny[DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(tiny, h, 1, 1, nb, nb, 0, 0));
        auto h_t = make_local<T>(h, tiny, [](int, int) { return T(4); });
        DeviceBuffer<T> dT(h, h_t.size());
        upload(h, dT.ptr, h_t);
        check_ddla_sync(h);
        int info = -1;
        const bool is_nega = ddla::ppotrf<B>(h, 'L', 1, dT.ptr, 1, 1, tiny, info, true, -1);
        require_close(h, tag_of(B, "ppotrf", ty, "n=1 head status"), std::abs(info) + (is_nega ? 1.0 : 0.0), 0.0);
        auto out = download(h, dT.ptr, h_t.size());
        require_close(h, tag_of(B, "ppotrf", ty, "n=1 head factor"),
                      local_max_error<T>(h, tiny, out, [](int, int) { return T(2); }), tol);
    }
    {   // not positive definite: info = 1
        auto h_n = make_local<T>(h, descA, [](int i, int j) { return (i != j) ? T(0) : T(i == 0 ? -1 : 2); });
        DeviceBuffer<T> dN(h, h_n.size());
        upload(h, dN.ptr, h_n);
        check_ddla_sync(h);
        int info = -1;
        const bool is_nega = ddla::ppotrf<B>(h, 'L', n, dN.ptr, 1, 1, descA, info);
        require_close(h, tag_of(B, "ppotrf", ty, "non-PD info"), std::abs(info - 1) + (is_nega ? 1.0 : 0.0), 0.0);
    }

    // bottom-right Cholesky: uplo 'U' gives A = U U^H, uplo 'L' gives A = L^H L.  Solve with the factor through ptrtrs.
    for (char uplo : {'U', 'L'}) {
        int descB[DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(descB, h, n, nrhs, nb, nb, 0, 0));
        auto h_B = rhs_for<T>(h, descB, n, 'L', 'N', GenFn(hpd_value), n);
        DeviceBuffer<T> dA(h, h_A.size()), dB(h, h_B.size());
        upload(h, dA.ptr, h_A);
        upload(h, dB.ptr, h_B);
        check_ddla_sync(h);
        int info = -1;
        ddla::ppotrf_bottom_right<B>(h, uplo, n, dA.ptr, descA, info);
        if (info != 0) MPI_Abort(ddlaGetCommunicator(h), 1);
        if (uplo == 'U') {   // U y = b, U^H x = y
            ddla::ptrtrs<B>(h, 'L', 'U', 'N', 'N', n, nrhs, dA.ptr, descA, dB.ptr, descB);
            ddla::ptrtrs<B>(h, 'L', 'U', 'C', 'N', n, nrhs, dA.ptr, descA, dB.ptr, descB);
        } else {             // L^H y = b, L x = y
            ddla::ptrtrs<B>(h, 'L', 'L', 'C', 'N', n, nrhs, dA.ptr, descA, dB.ptr, descB);
            ddla::ptrtrs<B>(h, 'L', 'L', 'N', 'N', n, nrhs, dA.ptr, descA, dB.ptr, descB);
        }
        check_x<T>(h, descB, dB.ptr, h_B.size(),
                   tag_of(B, "ppotrf_bottom_right", ty, std::string(1, uplo)), tol);
    }

    {   // head element at an interior position: ppotrf relocates it to the last index with pswap, ppotrs undoes it with pswap
        const int head_idx0 = std::max(0, n / 3 - 1);
        const int location = head_idx0 + 1;   // 1-based
        int descB[DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(descB, h, n, nrhs, nb, nb, 0, 0));
        auto h_H = make_local<T>(h, descA, [&](int i, int j) { return to_T<T>(head_value(i, j, n, head_idx0)); });
        auto h_B = make_local<T>(h, descB, [&](int i, int j) {
            T sum = T(0);
            for (int l = 0; l < n; ++l) sum += to_T<T>(head_value(i, l, n, head_idx0)) * to_T<T>(x_value(l, j));
            return sum;
        });
        DeviceBuffer<T> dH(h, h_H.size()), dB(h, h_B.size());
        upload(h, dH.ptr, h_H);
        upload(h, dB.ptr, h_B);
        check_ddla_sync(h);
        int info = -1;
        const bool is_nega = ddla::ppotrf<B>(h, 'L', n, dH.ptr, 1, 1, descA, info, true, location);
        if (info != 0) MPI_Abort(ddlaGetCommunicator(h), 1);
        require_close(h, tag_of(B, "ppotrf", ty, "head is_nega"), is_nega ? 0.0 : 1.0, 0.0);
        ddla::ppotrs<B>(h, 'L', 'L', 'N', n, nrhs, dH.ptr, descA, dB.ptr, descB, is_nega, location);
        check_x<T>(h, descB, dB.ptr, h_B.size(), tag_of(B, "ppotrf+ppotrs", ty, "head location"), tol);
    }
}

// ---- triangular solves -----------------------------------------------------------------------------------------------
Complex stored_tri(char uplo, int i, int j) { return (uplo == 'L') ? triangular_l_value(i, j) : triangular_l_value(j, i); }

Complex op_tri(char uplo, char trans, char diag, int i, int j)
{
    if (diag == 'U' && i == j) return Complex(1.0, 0.0);
    if (trans == 'N') return stored_tri(uplo, i, j);
    const Complex raw = stored_tri(uplo, j, i);
    return (trans == 'T') ? raw : std::conj(raw);
}

template <DdlaBackend B, typename T>
void check_ptrtrs(const DdlaHandle_t& h, int n, int nb, int nrhs)
{
    int descA[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, h, n, n, nb, nb, 0, 0));
    const char* ty = type_name<T>();
    for (char side : {'L', 'R'})
        for (char uplo : {'L', 'U'})
            for (char trans : {'N', 'T', 'C'})
                for (char diag : {'N', 'U'}) {
                    const int b_rows = (side == 'L') ? n : nrhs;
                    const int b_cols = (side == 'L') ? nrhs : n;
                    int descB[DDLA_DLEN_];
                    DDLA_CHECK(ddlaDescInit(descB, h, b_rows, b_cols, nb, nb, 0, 0));
                    auto h_A = make_local<T>(h, descA, [&](int i, int j) { return to_T<T>(stored_tri(uplo, i, j)); });
                    auto h_B = make_local<T>(h, descB, [&](int i, int j) {
                        T sum = T(0);
                        for (int l = 0; l < n; ++l) {
                            if (side == 'L') sum += to_T<T>(op_tri(uplo, trans, diag, i, l)) * to_T<T>(x_value(l, j));
                            else             sum += to_T<T>(x_value(i, l)) * to_T<T>(op_tri(uplo, trans, diag, l, j));
                        }
                        return sum;
                    });
                    DeviceBuffer<T> dA(h, h_A.size()), dB(h, h_B.size());
                    upload(h, dA.ptr, h_A);
                    upload(h, dB.ptr, h_B);
                    check_ddla_sync(h);
                    ddla::ptrtrs<B>(h, side, uplo, trans, diag, b_rows, b_cols, dA.ptr, descA, dB.ptr, descB);
                    check_x<T>(h, descB, dB.ptr, h_B.size(),
                               tag_of(B, "ptrtrs", ty, std::string(1, side) + "," + uplo + "," + trans + "," + diag),
                               2 * tolerance<T>());
                }
}

// ---- helpers: pgeadd, pdam ------------------------------------------------------------------------------------
template <DdlaBackend B, typename T>
void check_helpers(const DdlaHandle_t& h, const Shape& base)
{
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(h, nprows, npcols);
    const int nb = base.nb;
    const char* ty = type_name<T>();
    const double tol = 10 * tolerance<T>();
    const T alpha = to_T<T>(Complex(2.0, -0.5));
    const T beta = to_T<T>(Complex(-0.75, 1.25));

    auto geadd = [&](char ta, char tb, int m, int n) {
        const int a_rows = ta == 'N' ? m : n, a_cols = ta == 'N' ? n : m;
        const int b_rows = tb == 'N' ? m : n, b_cols = tb == 'N' ? n : m;
        int dA[DDLA_DLEN_], dB[DDLA_DLEN_], dC[DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(dA, h, a_rows, a_cols, nb, nb, 0, 0));
        DDLA_CHECK(ddlaDescInit(dB, h, b_rows, b_cols, nb, nb, 0, 0));
        DDLA_CHECK(ddlaDescInit(dC, h, m, n, nb, nb, 0, 0));
        auto hA = make_local<T>(h, dA, [](int i, int j) { return to_T<T>(general_value(i, j, 4)); });
        auto hB = make_local<T>(h, dB, [](int i, int j) { return to_T<T>(general_value(i, j, 5)); });
        std::vector<T> hC(local_size(h, dC), T(0));
        DeviceBuffer<T> bA(h, hA.size()), bB(h, hB.size()), bC(h, hC.size());
        upload(h, bA.ptr, hA);
        upload(h, bB.ptr, hB);
        upload(h, bC.ptr, hC);
        check_ddla_sync(h);
        ddla::pgeadd<B>(h, ta, tb, m, n, alpha, bA.ptr, 1, 1, dA, beta, bB.ptr, 1, 1, dB, bC.ptr, 1, 1, dC);
        auto out = download(h, bC.ptr, hC.size());
        const double err = local_max_error<T>(h, dC, out, [&](int i, int j) {
            return alpha * to_T<T>(op_value(ta, a_rows, a_cols, i, j, general_value, 4))
                 + beta * to_T<T>(op_value(tb, b_rows, b_cols, i, j, general_value, 5));
        });
        require_close(h, tag_of(B, "pgeadd", ty, std::string(1, ta) + "," + tb), err, tol);
    };
    geadd('N', 'N', round_up_for_grid(base.m, nb, nprows), round_up_for_grid(base.n, nb, npcols));
    if (nprows == npcols) {
        const int ns = round_up_for_grid(std::max(base.m, base.n), nb, nprows);
        geadd('C', 'N', ns, ns);
        geadd('N', 'T', ns, ns);
    }

    {   // pdam
        const int n = square_size(h, base);
        int desc[DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(desc, h, n, n, nb, nb, 0, 0));
        std::vector<T> hA(local_size(h, desc), T(0));
        DeviceBuffer<T> dAm(h, hA.size());
        upload(h, dAm.ptr, hA);
        check_ddla_sync(h);
        ddla::pdam<B>(h, alpha, dAm.ptr, desc);
        auto out = download(h, dAm.ptr, hA.size());
        require_close(h, tag_of(B, "pdam", ty),
                      local_max_error<T>(h, desc, out, [&](int i, int j) { return i == j ? alpha : T(0); }), tol);
    }

    for (bool conj : {false, true}) {   // pgeadd transpose-copy
        const int m = round_up_for_grid(base.m, nb, nprows);
        const int n = round_up_for_grid(base.n, nb, npcols);
        int dA[DDLA_DLEN_], dAT[DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(dA, h, m, n, nb, nb, 0, 0));
        DDLA_CHECK(ddlaDescInit(dAT, h, n, m, nb, nb, 0, 0));
        auto hA = make_local<T>(h, dA, [](int i, int j) { return to_T<T>(general_value(i, j, 6)); });
        std::vector<T> hAT(local_size(h, dAT), T(0));
        DeviceBuffer<T> bA(h, hA.size()), bAT(h, hAT.size());
        upload(h, bA.ptr, hA);
        upload(h, bAT.ptr, hAT);
        check_ddla_sync(h);
        ddla::pgeadd<B>(h, conj ? 'C' : 'T', 'N', n, m, T(1), bA.ptr, 1, 1, dA,
                        T(0), bAT.ptr, 1, 1, dAT, bAT.ptr, 1, 1, dAT);
        auto out = download(h, bAT.ptr, hAT.size());
        const double err = local_max_error<T>(h, dAT, out, [&](int i, int j) {
            const Complex v = general_value(j, i, 6);
            return to_T<T>(conj ? std::conj(v) : v);
        });
        require_close(h, tag_of(B, "pgeadd transpose", ty, conj ? "conj" : "plain"), err, tol);
    }
}

// Row exchanges of the partial-pivot panel move whole local rows between process rows; a row longer than the MPI eager
// limit used to hang a CPU handle (both partners entered a blocking send first).  n = 640 makes every row a few KB.
template <DdlaBackend B, typename T>
void check_long_rows(const DdlaHandle_t& h)
{
    const int n = 640, nb = 32, nrhs = 3;
    int descA[DDLA_DLEN_], descB[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, h, n, n, nb, nb, 0, 0));
    DDLA_CHECK(ddlaDescInit(descB, h, n, nrhs, nb, nb, 0, 0));
    auto h_A = make_local<T>(h, descA, [=](int i, int j) { return to_T<T>(cycled_value(i, j, n)); });
    auto h_B = rhs_for<T>(h, descB, n, 'L', 'N', GenFn(cycled_value), n);
    DeviceBuffer<T> dA(h, h_A.size()), dB(h, h_B.size());
    upload(h, dA.ptr, h_A);
    upload(h, dB.ptr, h_B);
    check_ddla_sync(h);
    ddla::pgesv<B>(h, 'L', 'N', n, nrhs, dA.ptr, descA, dB.ptr, descB);
    check_x<T>(h, descB, dB.ptr, h_B.size(), tag_of(B, "pgesv", type_name<T>(), "n=640 long pivot rows"),
               tolerance<T>());
}

template <DdlaBackend B, typename T>
void check_all_types(const DdlaHandle_t& h, const Shape& base, bool first_shape)
{
    const int nb = base.nb;
    const int n = square_size(h, base);
    const int nrhs = nrhs_size(base);
    // The solver variants are cheap at this size: every one runs for every type, but the 24-case ptrtrs sweep and the
    // four-way side/trans enumeration of the LU solves only on the first shape.
    check_lu<B, T>(h, n, nb, nrhs, first_shape);
    check_lu_info<B, T>(h, n, nb);
    check_cholesky<B, T>(h, n, nb, nrhs);
    if (first_shape) check_ptrtrs<B, T>(h, n, nb, nrhs_size(base, 5));
    check_helpers<B, T>(h, base);
}

template <DdlaBackend B>
void run_backend(const DdlaHandle_t& h, const std::vector<Shape>& shapes)
{
    bool first = true;
    for (const Shape& s : shapes) {
        check_all_types<B, float>(h, s, first);
        check_all_types<B, double>(h, s, first);
        check_all_types<B, std::complex<float>>(h, s, first);
        check_all_types<B, Complex>(h, s, first);
        first = false;
    }
    check_long_rows<B, double>(h);
    check_long_rows<B, Complex>(h);
}

// A routine called for the other backend than the handle's must throw (and must not start any communication).
template <DdlaBackend Wrong>
void check_rejected(const DdlaHandle_t& h, const char* name)
{
    const int n = 16, nb = 4;
    int desc[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(desc, h, n, n, nb, nb, 0, 0));
    std::vector<double> local(local_size(h, desc), 1.0);
    DeviceBuffer<double> d(h, local.size());
    int info = 0;
    std::vector<int> ipiv(ddla_test::m_loc(h, desc) + 1, 0);
    bool threw = false;
    try { ddla::pgetrf<Wrong>(h, n, n, d.ptr, 1, 1, desc, ipiv.data(), info); } catch (const std::runtime_error&) { threw = true; }
    bool threw2 = false;
    try { ddla::ptrtrs<Wrong>(h, 'L', 'L', 'N', 'N', n, n, d.ptr, desc, d.ptr, desc); } catch (const std::runtime_error&) { threw2 = true; }
    bool threw3 = false;
    try { ddla::pgeadd<Wrong>(h, 'T', 'N', n, n, 1.0, d.ptr, 1, 1, desc, 0.0, d.ptr, 1, 1, desc, d.ptr, 1, 1, desc); } catch (const std::runtime_error&) { threw3 = true; }
    require_close(h, std::string(name) + " wrong-backend template rejected",
                  (threw ? 0.0 : 1.0) + (threw2 ? 0.0 : 1.0) + (threw3 ? 0.0 : 1.0), 0.0);
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int nprocs = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    TestOptions options;
    std::string error;
    if (!parse_options(argc, argv, nprocs, options, error)) {
        if (rank == 0) std::cerr << "Error: " << error << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (auto grid : options.grids) {
#if DDLA_HAS_CPU
        {
            DdlaHandle_t h = nullptr;
            ddlaInit(h, DdlaBackend::CPU);
            ddlaSet(h, MPI_COMM_WORLD, grid.first, grid.second);
            if (rank == 0) std::cout << "=== CPU handle, grid " << grid_name(h) << " ===" << std::endl;
            run_backend<DdlaBackend::CPU>(h, options.shapes);
#if DDLA_HAS_GPU
            check_rejected<DdlaBackend::GPU>(h, "[CPU handle, <GPU>]");
#endif
            check_ddla_sync(h);
            ddlaDestroy(h);
        }
#endif
#if DDLA_HAS_GPU
        {
            DdlaHandle_t h = nullptr;
            ddlaInit(h, DdlaBackend::GPU);
            ddlaSet(h, MPI_COMM_WORLD, grid.first, grid.second);
            if (rank == 0) std::cout << "=== GPU handle, grid " << grid_name(h) << " ===" << std::endl;
            run_backend<DdlaBackend::GPU>(h, options.shapes);
#if DDLA_HAS_CPU
            check_rejected<DdlaBackend::CPU>(h, "[GPU handle, <CPU>]");
#endif
            check_ddla_sync(h);
            ddlaDestroy(h);
        }
#endif
    }

    if (rank == 0) std::cout << "test_backend_solvers passed" << std::endl;
    MPI_Finalize();
    return 0;
}
