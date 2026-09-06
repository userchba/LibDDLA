#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_pgesv_nopiv(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    const int n = square_size(handle, base);
    const int nrhs = nrhs_size(base);
    int descA[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, handle, n, n, nb, nb, 0, 0));

    auto h_A = make_local<Complex>(handle, descA, [=](int i, int j){ return dominant_value(i, j, n); });

    auto run_case = [&](char side, char trans){
        const std::string name = std::string("pgesv_nopiv(handle, ") + side + "," + trans + ")";
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

        ddla::pgesv_nopiv(handle, side, trans, n, nrhs, d_A.ptr, descA, d_B.ptr, descB);
        check_solution(handle, descB, d_B.ptr, h_B.size(), name, 5e-9);
    };

    run_case('L', 'N');
    run_case('L', 'T');
    run_case('L', 'C');
    run_case('R', 'N');
    run_case('R', 'T');
    run_case('R', 'C');
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pgesv_nopiv", check_pgesv_nopiv);
}
