#include "api_grid_test_common.h"

using namespace api_grid_test;

// Regression coverage for ppotrf/ppotrs's is_head/location handling.
//
// Neither test_api_grid_ppotrf.cpp nor test_api_grid_pposv.cpp exercises
// is_head=true at all (both only call ppotrf/pposv with the defaults). This
// file adds two cases:
//   1. location=-1: the head element is already the last global index --
//      the only path production code (LibRPA) currently invokes.
//   2. location=<interior index>: exercises ppotrf's internal pswap-based
//      relocation of the head element to the last index before
//      factorization. This path had a real bug (the second pswap call
//      swapped a column with itself, a no-op) that this test would have
//      caught: with the relocation broken, the un-relocated negative
//      diagonal is encountered by the *standard* (non-head-cased) Cholesky
//      step for a non-last block, which fails outright (info != 0) rather
//      than silently producing a wrong answer -- so this test fails loudly
//      on the pre-fix code, not just numerically.

namespace {

// Same diagonally-dominant Hermitian construction as hpd_value (see
// api_grid_test_common.h), except the diagonal entry at `head_idx`
// (0-based) is negated. The off-diagonal couplings are small relative to
// the (now negative) diagonal magnitude, so the Schur complement at that
// pivot goes negative, reliably exercising the head-correction (is_nega ==
// true) branch of ppotrf/ppotrs regardless of where head_idx sits.
inline Complex head_value(int i, int j, int n, int head_idx)
{
    if(i == j){
        const double diag = 5.0 + 0.2 * n + 0.05 * i;
        return Complex(i == head_idx ? -diag : diag, 0.0);
    }
    const int lo = std::min(i, j);
    const int hi = std::max(i, j);
    const Complex val(0.01 * ((lo + 2 * hi) % 5 - 2),
                      0.006 * ((3 * lo + hi) % 7 - 3));
    return i < j ? val : std::conj(val);
}

// build_rhs (api_grid_test_common.h) only accepts a plain function pointer,
// which can't capture head_idx -- inline the same B = A*X_known
// construction here instead.
inline std::vector<Complex> build_head_rhs(const ddla::DdlaDesc& descB, int n, int head_idx)
{
    return make_local<Complex>(descB, [&](int i, int j) -> Complex {
        Complex sum(0.0, 0.0);
        for(int l = 0; l < n; ++l){
            sum += head_value(i, l, n, head_idx) * x_value(l, j);
        }
        return sum;
    });
}

// ppotrs now applies the head-relocation permutation to B itself (rows for
// side='L', columns for side='R') when given the same `location` that was
// passed to ppotrf, so direct ppotrf + ppotrs head-correction callers do
// not need to permute B by hand.  check_head_case below verifies exactly
// that: ppotrf is called with is_head=true and location, and ppotrs is
// called with the same location -- no manual permute_rhs_rows() calls.
void check_head_case(const ddla::DdlaHandle_t& handle, int n, int nrhs, int nb,
                     int head_idx_1based, const std::string& label)
{
    ddla::DdlaDesc descA(handle), descB(handle);
    descA.init(n, n, nb, nb, 0, 0);
    descB.init(n, nrhs, nb, nb, 0, 0);

    const int head_idx0 = head_idx_1based - 1; // 0-based, for the value functions
    auto h_A = make_local<Complex>(descA, [=](int i, int j){
        return head_value(i, j, n, head_idx0);
    });
    auto h_B = build_head_rhs(descB, n, head_idx0);

    DeviceBuffer<Complex> d_A(handle, h_A.size());
    DeviceBuffer<Complex> d_B(handle, h_B.size());
    upload(handle, d_A.ptr, h_A);
    upload(handle, d_B.ptr, h_B);
    check_ddla_sync(handle);

    const int location = (head_idx_1based == n) ? -1 : head_idx_1based;

    int info = -1;
    const bool is_nega = ddla::ppotrf('L', n, d_A.ptr, 1, 1, descA, info, true, location);
    if(info != 0) MPI_Abort(ddla_get_communicator(handle), 1);
    // The construction is diagonally dominant everywhere except at
    // head_idx, so the head correction must fire for this test to be
    // meaningful; if this ever flips to false, head_value's diagonal
    // magnitude needs revisiting for this n.
    require_close(handle, label + " is_nega", is_nega ? 0.0 : 1.0, 0.0);

    // Same location forwarded to ppotrs; the B permutation (if any) happens
    // inside ppotrs.
    ddla::ppotrs('L', 'L', 'N', n, nrhs, d_A.ptr, descA, d_B.ptr, descB, is_nega, location);

    check_solution(handle, descB, d_B.ptr, h_B.size(), label, 5e-9);
}

} // namespace

void check_ppotrf_head(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    if(skip_non_square_grid(handle, "ppotrf_head")) return;

    const int nb = base.nb;
    const int n = square_size(handle, base);
    const int nrhs = nrhs_size(base);

    check_head_case(handle, n, nrhs, nb, n, "ppotrf_head(location=-1)");

    // Roughly a third of the way in, so it lands in an early block rather
    // than the trailing one for the shapes/grids this suite exercises --
    // deliberately not the same block as the last index, so this also
    // exercises pswap's cross-process communication path when the grid has
    // more than one process row/column.
    const int interior = std::max(1, n / 3);
    if(interior != n){
        check_head_case(handle, n, nrhs, nb, interior, "ppotrf_head(location=interior)");
    }
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_ppotrf_head", check_ppotrf_head);
}
