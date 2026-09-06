#include "api_grid_test_common.h"

#include <type_traits>

using namespace api_grid_test;

// ---------------------------------------------------------------------------
// Sub-matrix (leading-block) regression harness.
//
// For each converted routine we run it twice:
//   1. "exact":  descriptor dims == logical dims (the existing behavior).
//   2. "padded": descriptor dims strictly larger than the logical dims, with
//      the padding region prefilled with a sentinel.
// We then require that the leading-block result of the padded run matches the
// exact run element-wise, and that the padding region is byte-identical to the
// sentinel (the routine must never touch data beyond the leading block).
// ---------------------------------------------------------------------------

namespace {

const Complex kSentinel(12345.0, -6789.0);

// A padded side length that guarantees at least one extra local row/column on
// every process (so the padding region is non-empty everywhere).
inline int padded_size(int n, int nb, int nprows, int npcols)
{
    return n + nb * nprows * npcols + nb;
}

// Gather the leading m x n block of a local buffer into a global m x n array.
inline std::vector<Complex> gather_leading(const ddla::DdlaHandle_t& handle,
                                           const int* desc,
                                           const std::vector<Complex>& local,
                                           int m, int n)
{
    std::vector<Complex> G(static_cast<size_t>(m) * n, Complex(0, 0));
    for(int jloc = 0; jloc < ddla_test::n_loc(handle, desc); ++jloc){
        const int j = indx_l2g_c(desc, handle, jloc);
        if(j >= n) continue;
        for(int iloc = 0; iloc < ddla_test::m_loc(handle, desc); ++iloc){
            const int i = indx_l2g_r(desc, handle, iloc);
            if(i >= m) continue;
            G[static_cast<size_t>(i) * n + j] =
                local[iloc + static_cast<size_t>(jloc) * desc[DDLA_LLD_]];
        }
    }
    return G;
}

inline double global_max_abs_diff(const std::vector<Complex>& A,
                                  const std::vector<Complex>& B)
{
    double err = 0.0;
    for(size_t i = 0; i < A.size(); ++i)
        err = std::max(err, std::abs(A[i] - B[i]));
    return err;
}

// Max deviation of every element outside the leading m x n block from sentinel.
inline double padding_max_err(const ddla::DdlaHandle_t& handle,
                              const int* desc,
                              const std::vector<Complex>& local,
                              int m, int n)
{
    double err = 0.0;
    for(int jloc = 0; jloc < ddla_test::n_loc(handle, desc); ++jloc){
        const int j = indx_l2g_c(desc, handle, jloc);
        for(int iloc = 0; iloc < ddla_test::m_loc(handle, desc); ++iloc){
            const int i = indx_l2g_r(desc, handle, iloc);
            if(i < m && j < n) continue;
            err = std::max(err, std::abs(
                local[iloc + static_cast<size_t>(jloc) * desc[DDLA_LLD_]] - kSentinel));
        }
    }
    return err;
}

// Build a padded local buffer: sentinel everywhere, then the leading m x n
// block filled by value(i, j).
template <typename Fn>
inline std::vector<Complex> make_padded_local(const ddla::DdlaHandle_t& handle,
                                              const int* desc,
                                              int m, int n, Fn value)
{
    std::vector<Complex> local(static_cast<size_t>(desc[DDLA_LLD_])
                               * ddla_test::n_loc(handle, desc), kSentinel);
    for(int jloc = 0; jloc < ddla_test::n_loc(handle, desc); ++jloc){
        const int j = indx_l2g_c(desc, handle, jloc);
        if(j >= n) continue;
        for(int iloc = 0; iloc < ddla_test::m_loc(handle, desc); ++iloc){
            const int i = indx_l2g_r(desc, handle, iloc);
            if(i >= m) continue;
            local[iloc + static_cast<size_t>(jloc) * desc[DDLA_LLD_]] = value(i, j);
        }
    }
    return local;
}

// Run a routine twice (exact vs padded) and compare.  `run` is a callable
// taking the descriptor and device buffers; it must already have been given the
// exact/padded descriptors and uploaded the corresponding host buffers.
// `gather_m`/`gather_n` are the leading-block dims to compare.
struct CaseResult {
    std::string name;
    double lead_err = 0.0;    // max |padded[lead] - exact[lead]|
    double pad_err = 0.0;     // max |padded[padding] - sentinel|
};

// Verify one output buffer: compare leading block against the exact run's
// gathered output and check the padding against the sentinel.
inline void check_leading_block(const ddla::DdlaHandle_t& handle,
                                const std::string& name,
                                const int* desc_exact,
                                const int* desc_padded,
                                const std::vector<Complex>& exact_local,
                                const std::vector<Complex>& padded_local,
                                int m, int n, double tol)
{
    const auto exact_G = gather_leading(handle, desc_exact, exact_local, m, n);
    const auto padded_G = gather_leading(handle, desc_padded, padded_local, m, n);
    const double lead_err = global_max_abs_diff(exact_G, padded_G);
    require_close(handle, name + " leading-block", lead_err, tol);

    const double pad_err = padding_max_err(handle, desc_padded, padded_local, m, n);
    require_close(handle, name + " padding-untouched", pad_err, 0.0);
}

// ---------------------------------------------------------------------------
// 1. ptrtrs: A is n x n triangular (padded to P), B is m x nrhs (padded).
// ---------------------------------------------------------------------------
void check_ptrtrs_submatrix(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    const int n = square_size(handle, base);
    const int nrhs = nrhs_size(base, 4);
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    const int P = padded_size(n, nb, nprows, npcols);

    for(char side : {'L', 'R'}){
        const int b_rows = (side == 'L') ? n : nrhs;
        const int b_cols = (side == 'L') ? nrhs : n;
        for(char uplo : {'L', 'U'}){
            for(char trans : {'N', 'T'}){
                const std::string name = std::string("ptrtrs-sub(") + side + "," + uplo + "," + trans + ")";

                int descA_ex[ddla::DDLA_DLEN_], descB_ex[ddla::DDLA_DLEN_];
                DDLA_CHECK(ddlaDescInit(descA_ex, handle, n, n, nb, nb, 0, 0));
                DDLA_CHECK(ddlaDescInit(descB_ex, handle, b_rows, b_cols, nb, nb, 0, 0));
                int descA_p[ddla::DDLA_DLEN_], descB_p[ddla::DDLA_DLEN_];
                DDLA_CHECK(ddlaDescInit(descA_p, handle, P, P, nb, nb, 0, 0));
                DDLA_CHECK(ddlaDescInit(descB_p, handle, P, P, nb, nb, 0, 0));

                auto triA = [&](int i, int j){
                    return (uplo == 'L') ? triangular_l_value(i, j)
                                         : triangular_l_value(j, i);
                };
                auto op_triA = [&](int i, int j){
                    if(trans == 'N') return triA(i, j);
                    const Complex raw = triA(j, i);
                    return (trans == 'T') ? raw : std::conj(raw);
                };
                auto rhsB = [&](int i, int j){
                    Complex sum(0, 0);
                    for(int l = 0; l < n; ++l){
                        if(side == 'L')
                            sum += op_triA(i, l) * x_value(l, j);
                        else
                            sum += x_value(i, l) * op_triA(l, j);
                    }
                    return sum;
                };

                // exact run
                std::vector<Complex> exact_out;
                {
                    auto h_A = make_local<Complex>(handle, descA_ex, triA);
                    auto h_B = make_local<Complex>(handle, descB_ex, rhsB);
                    DeviceBuffer<Complex> d_A(handle, h_A.size());
                    DeviceBuffer<Complex> d_B(handle, h_B.size());
                    upload(handle, d_A.ptr, h_A);
                    upload(handle, d_B.ptr, h_B);
                    check_ddla_sync(handle);
                    ddla::ptrtrs(handle, side, uplo, trans, 'N',
                                 b_rows, b_cols, d_A.ptr, descA_ex, d_B.ptr, descB_ex);
                    exact_out = download(handle, d_B.ptr, h_B.size());
                    // Verify exact run against the analytical solution.
                    const double err = local_max_error<Complex>(handle, descB_ex, exact_out, [](int i, int j){
                        return x_value(i, j);
                    });
                    require_close(handle, name + " exact", err, 2e-10);
                }

                // padded run
                {
                    auto h_A = make_padded_local(handle, descA_p, n, n, triA);
                    auto h_B = make_padded_local(handle, descB_p, b_rows, b_cols, rhsB);
                    DeviceBuffer<Complex> d_A(handle, h_A.size());
                    DeviceBuffer<Complex> d_B(handle, h_B.size());
                    upload(handle, d_A.ptr, h_A);
                    upload(handle, d_B.ptr, h_B);
                    check_ddla_sync(handle);
                    ddla::ptrtrs(handle, side, uplo, trans, 'N',
                                 b_rows, b_cols, d_A.ptr, descA_p, d_B.ptr, descB_p);
                    auto out = download(handle, d_B.ptr, h_B.size());
                    check_leading_block(handle, name,
                                        descB_ex, descB_p,
                                        exact_out, out,
                                        b_rows, b_cols, 2e-10);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2. pgetrf + pgetrs (with pivoting).
// ---------------------------------------------------------------------------
void check_pgetrf_pgetrs_submatrix(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    const int n = square_size(handle, base);
    const int nrhs = nrhs_size(base);
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    const int P = padded_size(n, nb, nprows, npcols);

    // --- pgetrf: A in-place LU factors ---
    {
        const std::string name = "pgetrf-sub";
        int descA_ex[ddla::DDLA_DLEN_], descA_p[ddla::DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(descA_ex, handle, n, n, nb, nb, 0, 0));
        DDLA_CHECK(ddlaDescInit(descA_p, handle, P, P, nb, nb, 0, 0));

        auto A = [&](int i, int j){ return dominant_value(i, j, n); };

        std::vector<Complex> exact_out;
        {
            auto h_A = make_local<Complex>(handle, descA_ex, A);
            DeviceBuffer<Complex> d_A(handle, h_A.size());
            upload(handle, d_A.ptr, h_A);
            check_ddla_sync(handle);
            std::vector<int> ipiv(ddla_test::m_loc(handle, descA_ex));
            int info = -1;
            ddla::pgetrf(handle, n, n, d_A.ptr, descA_ex, ipiv.data(), info);
            if(info != 0) MPI_Abort(ddlaGetCommunicator(handle), 1);
            exact_out = download(handle, d_A.ptr, h_A.size());
        }
        {
            auto h_A = make_padded_local(handle, descA_p, n, n, A);
            DeviceBuffer<Complex> d_A(handle, h_A.size());
            upload(handle, d_A.ptr, h_A);
            check_ddla_sync(handle);
            std::vector<int> ipiv(ddla_test::m_loc(handle, descA_p));
            int info = -1;
            ddla::pgetrf(handle, n, n, d_A.ptr, descA_p, ipiv.data(), info);
            if(info != 0) MPI_Abort(ddlaGetCommunicator(handle), 1);
            auto out = download(handle, d_A.ptr, h_A.size());
            check_leading_block(handle, name, descA_ex, descA_p,
                                exact_out, out, n, n, 1e-10);
        }
    }

    // --- pgetrs: LU solve (needs pgetrf factors + ipiv) ---
    for(char side : {'L', 'R'}){
        const int b_rows = (side == 'L') ? n : nrhs;
        const int b_cols = (side == 'L') ? nrhs : n;
        const std::string name = std::string("pgetrs-sub(") + side + ")";
        int descA_ex[ddla::DDLA_DLEN_], descA_p[ddla::DDLA_DLEN_];
        int descB_ex[ddla::DDLA_DLEN_], descB_p[ddla::DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(descA_ex, handle, n, n, nb, nb, 0, 0));
        DDLA_CHECK(ddlaDescInit(descB_ex, handle, b_rows, b_cols, nb, nb, 0, 0));
        DDLA_CHECK(ddlaDescInit(descA_p, handle, P, P, nb, nb, 0, 0));
        DDLA_CHECK(ddlaDescInit(descB_p, handle, P, P, nb, nb, 0, 0));

        auto A = [&](int i, int j){ return dominant_value(i, j, n); };
        auto B = [&](int i, int j){
            Complex sum(0, 0);
            for(int l = 0; l < n; ++l){
                if(side == 'L') sum += dominant_value(i, l, n) * x_value(l, j);
                else            sum += x_value(i, l) * dominant_value(l, j, n);
            }
            return sum;
        };

        std::vector<Complex> exact_out;
        {
            auto h_A = make_local<Complex>(handle, descA_ex, A);
            auto h_B = make_local<Complex>(handle, descB_ex, B);
            DeviceBuffer<Complex> d_A(handle, h_A.size());
            DeviceBuffer<Complex> d_B(handle, h_B.size());
            upload(handle, d_A.ptr, h_A);
            upload(handle, d_B.ptr, h_B);
            check_ddla_sync(handle);
            std::vector<int> ipiv(ddla_test::m_loc(handle, descA_ex));
            int info = -1;
            ddla::pgetrf(handle, n, n, d_A.ptr, descA_ex, ipiv.data(), info);
            if(info != 0) MPI_Abort(ddlaGetCommunicator(handle), 1);
            ddla::pgetrs(handle, side, 'N', n, nrhs, d_A.ptr, descA_ex, ipiv.data(), d_B.ptr, descB_ex);
            exact_out = download(handle, d_B.ptr, h_B.size());
        }
        {
            auto h_A = make_padded_local(handle, descA_p, n, n, A);
            auto h_B = make_padded_local(handle, descB_p, b_rows, b_cols, B);
            DeviceBuffer<Complex> d_A(handle, h_A.size());
            DeviceBuffer<Complex> d_B(handle, h_B.size());
            upload(handle, d_A.ptr, h_A);
            upload(handle, d_B.ptr, h_B);
            check_ddla_sync(handle);
            std::vector<int> ipiv(ddla_test::m_loc(handle, descA_p));
            int info = -1;
            ddla::pgetrf(handle, n, n, d_A.ptr, descA_p, ipiv.data(), info);
            if(info != 0) MPI_Abort(ddlaGetCommunicator(handle), 1);
            ddla::pgetrs(handle, side, 'N', n, nrhs, d_A.ptr, descA_p, ipiv.data(), d_B.ptr, descB_p);
            auto out = download(handle, d_B.ptr, h_B.size());
            check_leading_block(handle, name, descB_ex, descB_p,
                                exact_out, out, b_rows, b_cols, 5e-9);
        }
    }
}

// ---------------------------------------------------------------------------
// 3. ppotrf / ppotrs / pposv (Cholesky).  Complex only; square grid.
// ---------------------------------------------------------------------------
void check_ppotrf_pposv_submatrix(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    if(skip_non_square_grid(handle, "ppotrf-sub")) return;
    const int nb = base.nb;
    const int n = square_size(handle, base);
    const int nrhs = nrhs_size(base);
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    const int P = padded_size(n, nb, nprows, npcols);

    // --- ppotrf ---
    {
        const std::string name = "ppotrf-sub";
        int descA_ex[ddla::DDLA_DLEN_], descA_p[ddla::DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(descA_ex, handle, n, n, nb, nb, 0, 0));
        DDLA_CHECK(ddlaDescInit(descA_p, handle, P, P, nb, nb, 0, 0));
        auto A = [&](int i, int j){ return hpd_value(i, j, n); };

        std::vector<Complex> exact_out;
        {
            auto h_A = make_local<Complex>(handle, descA_ex, A);
            DeviceBuffer<Complex> d_A(handle, h_A.size());
            upload(handle, d_A.ptr, h_A);
            check_ddla_sync(handle);
            int info = -1;
            ddla::ppotrf(handle, 'L', n, d_A.ptr, 1, 1, descA_ex, info);
            if(info != 0) MPI_Abort(ddlaGetCommunicator(handle), 1);
            exact_out = download(handle, d_A.ptr, h_A.size());
        }
        {
            auto h_A = make_padded_local(handle, descA_p, n, n, A);
            DeviceBuffer<Complex> d_A(handle, h_A.size());
            upload(handle, d_A.ptr, h_A);
            check_ddla_sync(handle);
            int info = -1;
            ddla::ppotrf(handle, 'L', n, d_A.ptr, 1, 1, descA_p, info);
            if(info != 0) MPI_Abort(ddlaGetCommunicator(handle), 1);
            auto out = download(handle, d_A.ptr, h_A.size());
            check_leading_block(handle, name, descA_ex, descA_p,
                                exact_out, out, n, n, 1e-9);
        }
    }

    // --- pposv (covers ppotrf + ppotrs) ---
    {
        const std::string name = "pposv-sub";
        int descA_ex[ddla::DDLA_DLEN_], descA_p[ddla::DDLA_DLEN_];
        int descB_ex[ddla::DDLA_DLEN_], descB_p[ddla::DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(descA_ex, handle, n, n, nb, nb, 0, 0));
        DDLA_CHECK(ddlaDescInit(descB_ex, handle, n, nrhs, nb, nb, 0, 0));
        DDLA_CHECK(ddlaDescInit(descA_p, handle, P, P, nb, nb, 0, 0));
        DDLA_CHECK(ddlaDescInit(descB_p, handle, P, P, nb, nb, 0, 0));
        auto A = [&](int i, int j){ return hpd_value(i, j, n); };
        auto B = build_rhs(handle, descB_ex, n, hpd_value, n);

        std::vector<Complex> exact_out;
        {
            auto h_A = make_local<Complex>(handle, descA_ex, A);
            DeviceBuffer<Complex> d_A(handle, h_A.size());
            DeviceBuffer<Complex> d_B(handle, B.size());
            upload(handle, d_A.ptr, h_A);
            upload(handle, d_B.ptr, B);
            check_ddla_sync(handle);
            int info = -1;
            ddla::pposv(handle, 'L', 'L', 'N', n, nrhs, d_A.ptr, 1, 1, descA_ex,
                        d_B.ptr, 1, 1, descB_ex, info);
            if(info != 0) MPI_Abort(ddlaGetCommunicator(handle), 1);
            exact_out = download(handle, d_B.ptr, B.size());
        }
        {
            auto h_A = make_padded_local(handle, descA_p, n, n, A);
            auto h_B = make_padded_local(handle, descB_p, n, nrhs, [&](int i, int j){
                Complex sum(0, 0);
                for(int l = 0; l < n; ++l)
                    sum += hpd_value(i, l, n) * x_value(l, j);
                return sum;
            });
            DeviceBuffer<Complex> d_A(handle, h_A.size());
            DeviceBuffer<Complex> d_B(handle, h_B.size());
            upload(handle, d_A.ptr, h_A);
            upload(handle, d_B.ptr, h_B);
            check_ddla_sync(handle);
            int info = -1;
            ddla::pposv(handle, 'L', 'L', 'N', n, nrhs, d_A.ptr, 1, 1, descA_p,
                        d_B.ptr, 1, 1, descB_p, info);
            if(info != 0) MPI_Abort(ddlaGetCommunicator(handle), 1);
            auto out = download(handle, d_B.ptr, h_B.size());
            check_leading_block(handle, name, descB_ex, descB_p,
                                exact_out, out, n, nrhs, 5e-9);
        }
    }
}

// ---------------------------------------------------------------------------
// 4. ppotrf_bottom_right.
// ---------------------------------------------------------------------------
void check_ppotrf_br_submatrix(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    if(skip_non_square_grid(handle, "ppotrf_bottom_right-sub")) return;
    const int nb = base.nb;
    const int n = square_size(handle, base);
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    const int P = padded_size(n, nb, nprows, npcols);

    const std::string name = "ppotrf_bottom_right-sub";
    int descA_ex[ddla::DDLA_DLEN_], descA_p[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA_ex, handle, n, n, nb, nb, 0, 0));
    DDLA_CHECK(ddlaDescInit(descA_p, handle, P, P, nb, nb, 0, 0));
    // Hermitian positive-definite for uplo='U' bottom-right factorization.
    auto A = [&](int i, int j){ return hpd_value(i, j, n); };

    std::vector<Complex> exact_out;
    {
        auto h_A = make_local<Complex>(handle, descA_ex, A);
        DeviceBuffer<Complex> d_A(handle, h_A.size());
        upload(handle, d_A.ptr, h_A);
        check_ddla_sync(handle);
        int info = -1;
        ddla::ppotrf_bottom_right(handle, 'U', n, d_A.ptr, descA_ex, info);
        if(info != 0) MPI_Abort(ddlaGetCommunicator(handle), 1);
        exact_out = download(handle, d_A.ptr, h_A.size());
    }
    {
        auto h_A = make_padded_local(handle, descA_p, n, n, A);
        DeviceBuffer<Complex> d_A(handle, h_A.size());
        upload(handle, d_A.ptr, h_A);
        check_ddla_sync(handle);
        int info = -1;
        ddla::ppotrf_bottom_right(handle, 'U', n, d_A.ptr, descA_p, info);
        if(info != 0) MPI_Abort(ddlaGetCommunicator(handle), 1);
        auto out = download(handle, d_A.ptr, h_A.size());
        check_leading_block(handle, name, descA_ex, descA_p,
                            exact_out, out, n, n, 1e-9);
    }
}

// ---------------------------------------------------------------------------
// 5. plapiv: m pivots over an n-segment; padded descriptor.
// ---------------------------------------------------------------------------
void check_plapiv_submatrix(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    const int m = round_up_for_grid(base.m, nb, std::max(nprows, npcols));
    const int P = padded_size(m, nb, nprows, npcols);

    auto value = [](int i, int j){ return Complex(10.0 * i + j, -0.5 * i + 0.25 * j); };
    // Global pivot vector: swap (0, m-1), (1, m-2); rest identity.
    std::vector<int> g_ipiv(m);
    for(int k = 0; k < m; ++k){
        if(k == 0)      g_ipiv[k] = m;
        else if(k == 1) g_ipiv[k] = m - 1;
        else            g_ipiv[k] = k + 1;
    }

    for(char direc : {'F', 'B'}){
        for(char rowcol : {'R', 'C'}){
            const std::string name = std::string("plapiv-sub(") + direc + "," + rowcol + ")";
            const int seg = (rowcol == 'R') ? m : m; // square matrix, segment = m
            int desc_ex[ddla::DDLA_DLEN_], desc_p[ddla::DDLA_DLEN_];
            DDLA_CHECK(ddlaDescInit(desc_ex, handle, m, m, nb, nb, 0, 0));
            DDLA_CHECK(ddlaDescInit(desc_p, handle, P, P, nb, nb, 0, 0));

            std::vector<Complex> exact_out;
            {
                std::vector<int> ipiv(ddla_test::m_loc(handle, desc_ex), 0);
                for(int iloc = 0; iloc < ddla_test::m_loc(handle, desc_ex); ++iloc)
                    ipiv[iloc] = g_ipiv[indx_l2g_r(desc_ex, handle, iloc)];
                auto h_A = make_local<Complex>(handle, desc_ex, value);
                DeviceBuffer<Complex> d_A(handle, h_A.size());
                upload(handle, d_A.ptr, h_A);
                check_ddla_sync(handle);
                ddla::plapiv(handle, direc, rowcol, 'C', m, seg, d_A.ptr, desc_ex, ipiv.data(), desc_ex, nullptr);
                exact_out = download(handle, d_A.ptr, h_A.size());
            }
            {
                std::vector<int> ipiv(ddla_test::m_loc(handle, desc_p), 0);
                for(int iloc = 0; iloc < ddla_test::m_loc(handle, desc_p); ++iloc){
                    const int k = indx_l2g_r(desc_p, handle, iloc);
                    ipiv[iloc] = (k < m) ? g_ipiv[k] : k + 1;
                }
                auto h_A = make_padded_local(handle, desc_p, m, m, value);
                DeviceBuffer<Complex> d_A(handle, h_A.size());
                upload(handle, d_A.ptr, h_A);
                check_ddla_sync(handle);
                ddla::plapiv(handle, direc, rowcol, 'C', m, seg, d_A.ptr, desc_p, ipiv.data(), desc_p, nullptr);
                auto out = download(handle, d_A.ptr, h_A.size());
                check_leading_block(handle, name, desc_ex, desc_p,
                                    exact_out, out, m, m, 1e-12);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 6. pdam: add alpha to the leading n x n diagonal.
// ---------------------------------------------------------------------------
void check_pdam_submatrix(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    const int P = padded_size(base.m, nb, nprows, npcols);
    const int n = base.m;

    auto value = [](int i, int j){ return Complex(3.0 + 0.01 * i, -0.02 * j); };
    const Complex alpha(0.5, -0.25);

    int desc_p[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(desc_p, handle, P, P, nb, nb, 0, 0));

    // Reference: add alpha to the leading n x n diagonal only.
    auto expected = [&](int i, int j){
        Complex v = value(i, j);
        if(i == j && i < n) v += alpha;
        return v;
    };

    auto h_A = make_padded_local(handle, desc_p, P, P, value);  // whole buffer is "matrix"
    DeviceBuffer<Complex> d_A(handle, h_A.size());
    upload(handle, d_A.ptr, h_A);
    check_ddla_sync(handle);

    ddla::pdam(handle, alpha, d_A.ptr, desc_p, n);
    auto out = download(handle, d_A.ptr, h_A.size());

    // Leading n x n matches expected; padding (i>=n or j>=n) untouched.
    double err = 0.0;
    for(int jloc = 0; jloc < ddla_test::n_loc(handle, desc_p); ++jloc){
        const int j = indx_l2g_c(desc_p, handle, jloc);
        for(int iloc = 0; iloc < ddla_test::m_loc(handle, desc_p); ++iloc){
            const int i = indx_l2g_r(desc_p, handle, iloc);
            const Complex want = (i < P && j < P) ? expected(i, j) : kSentinel;
            err = std::max(err, std::abs(
                out[iloc + static_cast<size_t>(jloc) * desc_p[DDLA_LLD_]] - want));
        }
    }
    require_close(handle, "pdam-sub", err, 1e-12);
}

} // namespace

void check_submatrix(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    check_ptrtrs_submatrix(handle, base);
    check_pgetrf_pgetrs_submatrix(handle, base);
    check_ppotrf_pposv_submatrix(handle, base);
    check_ppotrf_br_submatrix(handle, base);
    check_plapiv_submatrix(handle, base);
    check_pdam_submatrix(handle, base);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_submatrix", check_submatrix);
}
