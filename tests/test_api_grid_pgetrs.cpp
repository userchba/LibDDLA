#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_pgetrs(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    const int n = square_size(handle, base);
    const int nrhs = nrhs_size(base);
    int descA[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, handle, n, n, nb, nb, 0, 0));

    auto h_A = make_local<Complex>(handle, descA, [=](int i, int j){ return dominant_value(i, j, n); });

    auto run_case = [&](char side, char trans){
        const std::string name = std::string("pgetrs(handle, ") + side + "," + trans + ")";
        // side='L': B is n x nrhs, solve op(A)*X = B;
        // side='R': B is nrhs x n, solve X*op(A) = B.
        const int b_rows = (side == 'L') ? n : nrhs;
        const int b_cols = (side == 'L') ? nrhs : n;
        int descB[ddla::DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(descB, handle, b_rows, b_cols, nb, nb, 0, 0));
        auto h_B = build_rhs_side(handle, descB, n, side, trans, dominant_value, n);

        DeviceBuffer<Complex> d_A(handle, h_A.size());
        DeviceBuffer<Complex> d_B(handle, h_B.size());
        upload(handle, d_A.ptr, h_A);
        upload(handle, d_B.ptr, h_B);
        check_ddla_sync(handle);

        std::vector<int> ipiv(ddla_test::m_loc(handle, descA));
        int info = -1;
        ddla::pgetrf(handle, n, n, d_A.ptr, 1, 1, descA, ipiv.data(), info);
        if(info != 0) MPI_Abort(ddlaGetCommunicator(handle), 1);
        ddla::pgetrs(handle, side, trans, n, nrhs, d_A.ptr, descA, ipiv.data(), d_B.ptr, descB);
        check_solution(handle, descB, d_B.ptr, h_B.size(), name, 5e-9);
    };

    run_case('L', 'N');
    run_case('L', 'T');
    run_case('L', 'C');
    run_case('R', 'N');
    run_case('R', 'T');
    run_case('R', 'C');

    // Pivots take the owner from A and the local index from B: a B whose
    // solved dimension is distributed differently must be rejected (it used
    // to write at local index -1).
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    std::vector<int> ipiv(std::max(1, ddla_test::m_loc(handle, descA)), 1);
    auto reject = [&](char side, int rsrc, int csrc, const std::string& what){
        const int b_rows = (side == 'L') ? n : nrhs;
        const int b_cols = (side == 'L') ? nrhs : n;
        int descB[ddla::DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(descB, handle, b_rows, b_cols, nb, nb, rsrc, csrc));
        expect_invalid(handle, std::string("pgetrs side=") + side + " B " + what, [&]{
            ddla::pgetrs(handle, side, 'N', n, nrhs, static_cast<Complex*>(nullptr), descA, ipiv.data(),
                         static_cast<Complex*>(nullptr), descB);
        });
    };
    if(nprows > 1) reject('L', 1, 0, "RSRC != A RSRC");
    if(npcols > 1) reject('R', 0, 1, "CSRC != A CSRC");
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pgetrs", check_pgetrs);
}
