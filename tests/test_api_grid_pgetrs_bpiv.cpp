#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_pgetrs_bpiv(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    const int n = square_size(handle, base);
    const int nrhs = nrhs_size(base);
    int descA[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, handle, n, n, nb, nb, 0, 0));

    auto run_case = [&](char side, char trans, Complex (*gen)(int, int, int)){
        const std::string name = std::string("pgetrs_bpiv(handle, ") + side + "," + trans + ")";
        // side='L': B is n x nrhs, solve op(A)*X = B;
        // side='R': B is nrhs x n, solve X*op(A) = B.
        const int b_rows = (side == 'L') ? n : nrhs;
        const int b_cols = (side == 'L') ? nrhs : n;
        auto h_A = make_local<Complex>(handle, descA, [&](int i, int j){ return gen(i, j, n); });
        int descB[ddla::DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(descB, handle, b_rows, b_cols, nb, nb, 0, 0));
        auto h_B = build_rhs_side(handle, descB, n, side, trans, gen, n);

        DeviceBuffer<Complex> d_A(handle, h_A.size());
        DeviceBuffer<Complex> d_B(handle, h_B.size());
        DeviceBuffer<int> d_ipiv(handle, std::max(1, ddla_test::m_loc(handle, descA)));
        upload(handle, d_A.ptr, h_A);
        upload(handle, d_B.ptr, h_B);
        check_ddla_sync(handle);

        int info = -1;
        ddla::pgetrf_bpiv(handle, n, n, d_A.ptr, descA, d_ipiv.ptr, info);
        if(info != 0) MPI_Abort(ddlaGetCommunicator(handle), 1);
        ddla::pgetrs_bpiv(handle, side, trans, n, nrhs, d_A.ptr, descA, d_ipiv.ptr, d_B.ptr, descB);
        check_solution(handle, descB, d_B.ptr, h_B.size(), name, 5e-9);
    };

    // Both the no-pivot path (dominant_value) and the real row-swap path
    // (cycled_value) must solve correctly.
    for(auto gen : {dominant_value, cycled_value}){
        run_case('L', 'N', gen);
        run_case('L', 'T', gen);
        run_case('L', 'C', gen);
        run_case('R', 'N', gen);
        run_case('R', 'T', gen);
        run_case('R', 'C', gen);
    }
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pgetrs_bpiv", check_pgetrs_bpiv);
}
