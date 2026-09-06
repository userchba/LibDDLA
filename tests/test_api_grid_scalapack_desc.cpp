/**
 * @file test_api_grid_scalapack_desc.cpp
 * @brief The ScaLAPACK int[9] descriptor contract.
 *
 * With the plain int[9] being the only descriptor spelling, this file pins
 * the three guarantees callers rely on:
 *   1. ddlaDescInit fills every slot the way DESCINIT does and treats LLD
 *      as a suggestion, storing MAX(lld, MAX(1, LOCr(M_A))): the tight bound
 *      by default, a padded value verbatim, an undersized one raised;
 *   2. a caller-padded LLD (over-allocated local buffer) is honoured by the
 *      routines, compared element-by-element against a tight run;
 *   3. malformed descriptors are rejected -- ddlaDescInit returns a nonzero
 *      status and the routines throw std::invalid_argument -- instead of
 *      being silently misinterpreted.
 *
 * The descriptors deliberately carry a nonsense value in desc[DDLA_CTXT_].
 * LibDDLA takes the process grid from the handle and must never read that
 * slot, so a bogus context must not perturb any result.
 */
#include "api_grid_test_common.h"

#include <algorithm>
#include <stdexcept>

using namespace api_grid_test;

namespace {

constexpr int kBogusContext = -12345;

/// A padded local buffer: value(i, j) on logically owned entries, laid out
/// with the given (larger) leading dimension.
template <typename T, typename Fn>
std::vector<T> make_local_lld(const ddla::DdlaHandle_t& handle, const int* desc,
                              int lld, Fn value)
{
    std::vector<T> local(static_cast<size_t>(lld)
                         * ddla_test::n_loc(handle, desc), T{});
    for(int jloc = 0; jloc < ddla_test::n_loc(handle, desc); ++jloc){
        const int j = indx_l2g_c(desc, handle, jloc);
        if(j >= desc[ddla::DDLA_N_]) continue;
        for(int iloc = 0; iloc < ddla_test::m_loc(handle, desc); ++iloc){
            const int i = indx_l2g_r(desc, handle, iloc);
            if(i >= desc[ddla::DDLA_M_]) continue;
            local[iloc + jloc * lld] = value(i, j);
        }
    }
    return local;
}

// Largest elementwise difference over the logically owned entries, allowing
// the two buffers to use different leading dimensions.
double diff_over_owned(const ddla::DdlaHandle_t& handle, const int* desc,
                       const std::vector<Complex>& lhs, int lhs_lld,
                       const std::vector<Complex>& rhs, int rhs_lld)
{
    double err = 0.0;
    for(int jloc = 0; jloc < ddla_test::n_loc(handle, desc); ++jloc){
        if(indx_l2g_c(desc, handle, jloc) >= desc[ddla::DDLA_N_]) continue;
        for(int iloc = 0; iloc < ddla_test::m_loc(handle, desc); ++iloc){
            if(indx_l2g_r(desc, handle, iloc) >= desc[ddla::DDLA_M_]) continue;
            const Complex a = lhs[iloc + jloc * lhs_lld];
            const Complex b = rhs[iloc + jloc * rhs_lld];
            err = std::max(err, static_cast<double>(std::abs(a - b)));
        }
    }
    return err;
}

// --------------------------------------------------------------------------
// ddlaDescInit reproduces DESCINIT, zeroes CTXT, and applies the LLD contract.
// --------------------------------------------------------------------------
void check_descinit_round_trip(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int n = square_size(handle, base);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);

    int desc[ddla::DDLA_DLEN_];
    std::fill(desc, desc + ddla::DDLA_DLEN_, 0);
    desc[ddla::DDLA_CTXT_] = kBogusContext;
    const ddlaStatus_t rc = ddlaDescInit(desc, handle,
                                         n, n, base.nb, base.nb, g_test_irsrc, g_test_icsrc);

    double err = 0.0;
    auto same = [&err](bool ok){ if(!ok) err = 1.0; };
    same(rc == ddlaStatus_t::DDLA_STATUS_SUCCESS);
    same(desc[ddla::DDLA_DTYPE_] == ddla::DDLA_BLOCK_CYCLIC_2D);
    same(desc[ddla::DDLA_M_] == n);
    same(desc[ddla::DDLA_N_] == n);
    same(desc[ddla::DDLA_MB_] == base.nb);
    same(desc[ddla::DDLA_NB_] == base.nb);
    same(desc[ddla::DDLA_RSRC_] == g_test_irsrc);
    same(desc[ddla::DDLA_CSRC_] == g_test_icsrc);
    const int expected_lld = std::max(1, ddla::num_loc(n, base.nb, myprow,
                                                       g_test_irsrc, nprows));
    same(desc[ddla::DDLA_LLD_] == expected_lld);
    // CTXT_ is zeroed so a filled descriptor can be copied or compared
    // wholesale; the pre-filled kBogusContext must be overwritten.
    same(desc[ddla::DDLA_CTXT_] == 0);

    require_close(handle, "ddlaDescInit round trip", err, 0.0);
}

// --------------------------------------------------------------------------
// ddlaDescInit rejects malformed input via its status.
// --------------------------------------------------------------------------
void check_descinit_rejects_bad_input(const ddla::DdlaHandle_t& handle,
                                      const Shape& base)
{
    const int n = square_size(handle, base);
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);

    int desc[ddla::DDLA_DLEN_];
    double err = 0.0;
    const int rank = ddlaGetRank(handle);
    auto expect_fail = [&](const char* what, ddlaStatus_t rc){
        if(rc == ddlaStatus_t::DDLA_STATUS_SUCCESS){
            err = 1.0;
            if(rank == 0)
                std::cerr << "FAIL: accepted " << what << std::endl;
        }
    };
    const ddla::DdlaHandle_t null_handle = nullptr;

    expect_fail("a null descriptor",
                ddlaDescInit(nullptr, handle, n, n, base.nb, base.nb, 0, 0));
    expect_fail("a null handle",
                ddlaDescInit(desc, null_handle, n, n, base.nb, base.nb, 0, 0));
    expect_fail("a negative M",
                ddlaDescInit(desc, handle, -n, n, base.nb, base.nb, 0, 0));
    expect_fail("a non-positive NB",
                ddlaDescInit(desc, handle, n, n, base.nb, 0, 0, 0));
    expect_fail("an out-of-range RSRC",
                ddlaDescInit(desc, handle, n, n, base.nb, base.nb, nprows, 0));
    expect_fail("an out-of-range CSRC",
                ddlaDescInit(desc, handle, n, n, base.nb, base.nb, 0, npcols));

    require_close(handle, "ddlaDescInit rejects bad input", err, 0.0);
}

// --------------------------------------------------------------------------
// The LLD argument is a suggestion, stored as DESCINIT's
// MAX(LLD, MAX(1, LOCr(M_A))) (TOOLS/descinit.f) but without its INFO = -9:
// a padded value is kept verbatim, an undersized one is raised to the tight
// bound rather than rejected, and the negative default derives that bound.
// So ddlaDescInit can never emit an LLD_ the reading side would refuse.
// --------------------------------------------------------------------------
void check_descinit_lld(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int n = square_size(handle, base);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
    const int lld_min = std::max(1, ddla::num_loc(n, base.nb, myprow,
                                                 g_test_irsrc, nprows));

    double err = 0.0;
    const int rank = ddlaGetRank(handle);
    int desc[ddla::DDLA_DLEN_];

    // Undersized: raised to the tight bound, not rejected -- lld is a
    // suggestion. (lld_min - 1 is 0 when lld_min is 1, still below the bound.)
    if (ddlaDescInit(desc, handle, n, n, base.nb, base.nb, g_test_irsrc, g_test_icsrc, lld_min - 1)
        != ddlaStatus_t::DDLA_STATUS_SUCCESS) {
        err = 1.0;
        if(rank == 0)
            std::cerr << "FAIL: rejected an undersized LLD suggestion"
                      << std::endl;
    }
    if (desc[ddla::DDLA_LLD_] != lld_min) {
        err = 1.0;
        if(rank == 0)
            std::cerr << "FAIL: undersized LLD not raised to MAX(1, LOCr) (got "
                      << desc[ddla::DDLA_LLD_] << ", wanted " << lld_min << ")"
                      << std::endl;
    }

    // Padded: stored verbatim -- exactly what was asked for.
    if (ddlaDescInit(desc, handle, n, n, base.nb, base.nb, g_test_irsrc, g_test_icsrc, lld_min + 3)
        != ddlaStatus_t::DDLA_STATUS_SUCCESS) {
        err = 1.0;
        if(rank == 0)
            std::cerr << "FAIL: rejected a padded LLD" << std::endl;
    }
    if (desc[ddla::DDLA_LLD_] != lld_min + 3) {
        err = 1.0;
        if(rank == 0)
            std::cerr << "FAIL: padded LLD not stored verbatim (got "
                      << desc[ddla::DDLA_LLD_] << ", wanted "
                      << lld_min + 3 << ")" << std::endl;
    }

    require_close(handle, "ddlaDescInit LLD contract", err, 0.0);
}

// --------------------------------------------------------------------------
// The routines reject malformed descriptors instead of misreading them.
// --------------------------------------------------------------------------
void check_routine_rejects_bad_desc(const ddla::DdlaHandle_t& handle,
                                    const Shape& base)
{
    if(skip_non_square_grid(handle, "pgemm rejects bad descriptors")) return;

    const int nb = base.nb;
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    const int m = round_up_for_grid(base.m, nb, nprows);
    const int n = round_up_for_grid(base.n, nb, npcols);
    const int k = std::max(base.k, nb * std::max(nprows, npcols) + 1);

    int descA[ddla::DDLA_DLEN_], descB[ddla::DDLA_DLEN_], descC[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, handle, k, m, nb, nb, g_test_irsrc, g_test_icsrc));
    DDLA_CHECK(ddlaDescInit(descB, handle, k, n, nb, nb, g_test_irsrc, g_test_icsrc));
    DDLA_CHECK(ddlaDescInit(descC, handle, m, n, nb, nb, g_test_irsrc, g_test_icsrc));

    auto h_A = make_local<Complex>(handle, descA, [](int i, int j){ return general_value(i, j, 1); });
    auto h_B = make_local<Complex>(handle, descB, [](int i, int j){ return general_value(i, j, 2); });
    auto h_C = make_local<Complex>(handle, descC, [](int i, int j){ return general_value(i, j, 3); });
    DeviceBuffer<Complex> d_A(handle, h_A.size());
    DeviceBuffer<Complex> d_B(handle, h_B.size());
    DeviceBuffer<Complex> d_C(handle, h_C.size());
    upload(handle, d_A.ptr, h_A);
    upload(handle, d_B.ptr, h_B);
    upload(handle, d_C.ptr, h_C);
    check_ddla_sync(handle);

    double err = 0.0;
    const int rank = ddlaGetRank(handle);
    auto must_throw = [&](const char* what, auto&& fn){
        try {
            fn();
            err = 1.0;
            if(rank == 0)
                std::cerr << "FAIL: accepted " << what << std::endl;
        } catch(const std::invalid_argument&) {
            // expected
        } catch(...) {
            err = 1.0;
            if(rank == 0)
                std::cerr << "FAIL: wrong exception for " << what << std::endl;
        }
    };

    int bad[ddla::DDLA_DLEN_];
    std::copy(descA, descA + ddla::DDLA_DLEN_, bad);
    bad[ddla::DDLA_DTYPE_] = 501;   // a non-dense descriptor type
    must_throw("a non-dense DTYPE_", [&]{
        ddla::pgemm<>(handle, 'N', 'N', m, n, k, Complex(1.0, 0.0),
                      d_A.ptr, bad, d_B.ptr, descB,
                      Complex(0.0, 0.0), d_C.ptr, descC);
    });

    if(descA[ddla::DDLA_LLD_] > 1){
        std::copy(descA, descA + ddla::DDLA_DLEN_, bad);
        bad[ddla::DDLA_LLD_] = descA[ddla::DDLA_LLD_] - 1;
        must_throw("an LLD_A below max(1, LOCr)", [&]{
            ddla::pgemm<>(handle, 'N', 'N', m, n, k, Complex(1.0, 0.0),
                          d_A.ptr, bad, d_B.ptr, descB,
                          Complex(0.0, 0.0), d_C.ptr, descC);
        });
    }

    require_close(handle, "pgemm rejects malformed descriptors", err, 0.0);
}

// --------------------------------------------------------------------------
// LLD_A larger than max(1, LOCr) is legal ScaLAPACK and must be honoured:
// every local buffer gets three extra rows of stride and a bogus CTXT_, and
// the padded run is compared against the tight run element by element.
// --------------------------------------------------------------------------
void check_padded_lld(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    if(skip_non_square_grid(handle, "pgemm padded LLD")) return;

    const int nb = base.nb;
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    const int m = round_up_for_grid(base.m, nb, nprows);
    const int n = round_up_for_grid(base.n, nb, npcols);
    const int k = std::max(base.k, nb * std::max(nprows, npcols) + 1);
    const Complex alpha(0.8, -0.2);
    const Complex beta(-0.3, 0.1);

    int descA[ddla::DDLA_DLEN_], descB[ddla::DDLA_DLEN_], descC[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, handle, m, k, nb, nb, g_test_irsrc, g_test_icsrc));
    DDLA_CHECK(ddlaDescInit(descB, handle, k, n, nb, nb, g_test_irsrc, g_test_icsrc));
    DDLA_CHECK(ddlaDescInit(descC, handle, m, n, nb, nb, g_test_irsrc, g_test_icsrc));

    auto valA = [](int i, int j){ return general_value(i, j, 1); };
    auto valB = [](int i, int j){ return general_value(i, j, 2); };
    auto valC = [](int i, int j){ return general_value(i, j, 3); };

    // Tight reference run.
    auto h_A = make_local<Complex>(handle, descA, valA);
    auto h_B = make_local<Complex>(handle, descB, valB);
    auto h_C = make_local<Complex>(handle, descC, valC);
    DeviceBuffer<Complex> d_A(handle, h_A.size());
    DeviceBuffer<Complex> d_B(handle, h_B.size());
    DeviceBuffer<Complex> d_C(handle, h_C.size());
    upload(handle, d_A.ptr, h_A);
    upload(handle, d_B.ptr, h_B);
    upload(handle, d_C.ptr, h_C);
    check_ddla_sync(handle);
    ddla::pgemm<>(handle, 'N', 'N', m, n, k, alpha,
                  d_A.ptr, descA, d_B.ptr, descB,
                  beta, d_C.ptr, descC);
    const auto tight = download(handle, d_C.ptr, h_C.size());

    // Padded run: every descriptor gets LLD + 3, passed through
    // ddlaDescInit's LLD argument (DESCINIT stores it verbatim), plus a
    // nonsense context that must be ignored.
    const int lldA = descA[ddla::DDLA_LLD_] + 3;
    const int lldB = descB[ddla::DDLA_LLD_] + 3;
    const int lldC = descC[ddla::DDLA_LLD_] + 3;
    int sdA[ddla::DDLA_DLEN_], sdB[ddla::DDLA_DLEN_], sdC[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(sdA, handle, m, k, nb, nb, g_test_irsrc, g_test_icsrc, lldA));
    DDLA_CHECK(ddlaDescInit(sdB, handle, k, n, nb, nb, g_test_irsrc, g_test_icsrc, lldB));
    DDLA_CHECK(ddlaDescInit(sdC, handle, m, n, nb, nb, g_test_irsrc, g_test_icsrc, lldC));
    sdA[ddla::DDLA_CTXT_] = kBogusContext;
    sdB[ddla::DDLA_CTXT_] = kBogusContext;
    sdC[ddla::DDLA_CTXT_] = kBogusContext;

    auto p_A = make_local_lld<Complex>(handle, descA, lldA, valA);
    auto p_B = make_local_lld<Complex>(handle, descB, lldB, valB);
    auto p_C = make_local_lld<Complex>(handle, descC, lldC, valC);
    DeviceBuffer<Complex> q_A(handle, p_A.size());
    DeviceBuffer<Complex> q_B(handle, p_B.size());
    DeviceBuffer<Complex> q_C(handle, p_C.size());
    upload(handle, q_A.ptr, p_A);
    upload(handle, q_B.ptr, p_B);
    upload(handle, q_C.ptr, p_C);
    check_ddla_sync(handle);
    ddla::pgemm<>(handle, 'N', 'N', m, n, k, alpha,
                  q_A.ptr, sdA, q_B.ptr, sdB,
                  beta, q_C.ptr, sdC);
    const auto padded = download(handle, q_C.ptr, p_C.size());

    const double err = diff_over_owned(handle, descC, padded, lldC,
                                       tight, descC[ddla::DDLA_LLD_]);
    require_close(handle, "pgemm honours padded LLD_A", err, 0.0);
}

#if DDLA_HAS_GPU
// --------------------------------------------------------------------------
// ppotrf with a bogus CTXT_ must agree with the tight run bit for bit.
// --------------------------------------------------------------------------
void check_ppotrf_bogus_ctxt(const ddla::DdlaHandle_t& handle,
                             const Shape& base)
{
    if(skip_non_square_grid(handle, "ppotrf bogus CTXT_")) return;

    const int n = square_size(handle, base);
    const int nb = base.nb;

    int descA[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, handle, n, n, nb, nb, g_test_irsrc, g_test_icsrc));
    int sdA[ddla::DDLA_DLEN_];
    std::copy(descA, descA + ddla::DDLA_DLEN_, sdA);
    sdA[ddla::DDLA_CTXT_] = kBogusContext;

    auto h_A = make_local<Complex>(handle, descA, [n](int i, int j){ return hpd_value(i, j, n); });

    DeviceBuffer<Complex> d_A(handle, h_A.size());
    upload(handle, d_A.ptr, h_A);
    check_ddla_sync(handle);
    int info = 0;
    (void)ddla::ppotrf<Complex>(handle, 'L', n, d_A.ptr, 1, 1, descA, info);
    const auto via_tight = download(handle, d_A.ptr, h_A.size());

    upload(handle, d_A.ptr, h_A);
    check_ddla_sync(handle);
    int info_desc = 0;
    (void)ddla::ppotrf<Complex>(handle, 'L', n, d_A.ptr, 1, 1, sdA, info_desc);
    const auto via_bogus = download(handle, d_A.ptr, h_A.size());

    double err = diff_over_owned(handle, descA, via_bogus, descA[ddla::DDLA_LLD_],
                                 via_tight, descA[ddla::DDLA_LLD_]);
    if(info != info_desc) err = 1.0;
    require_close(handle, "ppotrf ignores CTXT_", err, 0.0);
}

// --------------------------------------------------------------------------
// ptran honours a padded LLD the same way (no square-grid restriction).
// --------------------------------------------------------------------------
void check_ptran_padded_lld(const ddla::DdlaHandle_t& handle,
                            const Shape& base)
{
    const int n = square_size(handle, base);
    const int nb = base.nb;

    int descA[ddla::DDLA_DLEN_], descAT[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, handle, n, n, nb, nb, g_test_irsrc, g_test_icsrc));
    DDLA_CHECK(ddlaDescInit(descAT, handle, n, n, nb, nb, g_test_irsrc, g_test_icsrc));

    auto h_A  = make_local<Complex>(handle, descA, [n](int i, int j){ return hpd_value(i, j, n); });
    auto h_AT = make_local<Complex>(handle, descAT, [](int, int){ return Complex{}; });

    DeviceBuffer<Complex> d_A(handle, h_A.size());
    DeviceBuffer<Complex> d_AT(handle, h_AT.size());
    upload(handle, d_A.ptr, h_A);
    upload(handle, d_AT.ptr, h_AT);
    check_ddla_sync(handle);
    ddla::ptran<Complex>(handle, d_A.ptr, descA, d_AT.ptr, descAT, true);
    const auto tight = download(handle, d_AT.ptr, h_AT.size());

    const int lldA = descA[ddla::DDLA_LLD_] + 3;
    const int lldAT = descAT[ddla::DDLA_LLD_] + 3;
    int sdA[ddla::DDLA_DLEN_], sdAT[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(sdA, handle, n, n, nb, nb, g_test_irsrc, g_test_icsrc, lldA));
    DDLA_CHECK(ddlaDescInit(sdAT, handle, n, n, nb, nb, g_test_irsrc, g_test_icsrc, lldAT));
    sdA[ddla::DDLA_CTXT_] = kBogusContext;
    sdAT[ddla::DDLA_CTXT_] = kBogusContext;

    auto p_A  = make_local_lld<Complex>(handle, descA, lldA, [n](int i, int j){ return hpd_value(i, j, n); });
    auto p_AT = make_local_lld<Complex>(handle, descAT, lldAT, [](int, int){ return Complex{}; });
    DeviceBuffer<Complex> q_A(handle, p_A.size());
    DeviceBuffer<Complex> q_AT(handle, p_AT.size());
    upload(handle, q_A.ptr, p_A);
    upload(handle, q_AT.ptr, p_AT);
    check_ddla_sync(handle);
    ddla::ptran<Complex>(handle, q_A.ptr, sdA, q_AT.ptr, sdAT, true);
    const auto padded = download(handle, q_AT.ptr, p_AT.size());

    const double err = diff_over_owned(handle, descAT, padded, lldAT,
                                       tight, descAT[ddla::DDLA_LLD_]);
    require_close(handle, "ptran honours padded LLD_A", err, 0.0);
}
#endif // DDLA_HAS_GPU

} // namespace

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_scalapack_desc",
                         [](const ddla::DdlaHandle_t& handle, const Shape& base){
                             check_descinit_round_trip(handle, base);
                             check_descinit_rejects_bad_input(handle, base);
                             check_descinit_lld(handle, base);
                             check_routine_rejects_bad_desc(handle, base);
                             check_padded_lld(handle, base);
#if DDLA_HAS_GPU
                             check_ppotrf_bogus_ctxt(handle, base);
                             check_ptran_padded_lld(handle, base);
#endif
                         });
}
