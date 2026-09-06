#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_ppotrs(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    if(skip_non_square_grid(handle, "ppotrs")) return;

    const int nb = base.nb;
    const int n = square_size(handle, base);
    const int nrhs = nrhs_size(base);
    int descA[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, handle, n, n, nb, nb, 0, 0));

    auto h_A = make_local<Complex>(handle, descA, [=](int i, int j){ return hpd_value(i, j, n); });

    auto run_case = [&](char side, char uplo, char trans){
        const std::string name = std::string("ppotrs(handle, ") + side + "," + uplo + "," + trans + ")";
        // side='L': B is n x nrhs, solve op(A)*X = B;
        // side='R': B is nrhs x n, solve X*op(A) = B.
        const int b_rows = (side == 'L') ? n : nrhs;
        const int b_cols = (side == 'L') ? nrhs : n;
        int descB[ddla::DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(descB, handle, b_rows, b_cols, nb, nb, 0, 0));
        auto h_B = build_rhs_side(handle, descB, n, side, trans, hpd_value, n);

        DeviceBuffer<Complex> d_A(handle, h_A.size());
        DeviceBuffer<Complex> d_B(handle, h_B.size());
        upload(handle, d_A.ptr, h_A);
        upload(handle, d_B.ptr, h_B);
        check_ddla_sync(handle);

        int info = -1;
        const bool is_nega = ddla::ppotrf(handle, uplo, n, d_A.ptr, 1, 1, descA, info);
        if(info != 0 || is_nega) MPI_Abort(ddlaGetCommunicator(handle), 1);
        ddla::ppotrs(handle, side, uplo, trans, n, nrhs, d_A.ptr, descA, d_B.ptr, descB, is_nega);
        check_solution(handle, descB, d_B.ptr, h_B.size(), name, 5e-9);
    };

    // A is Hermitian, so trans='C' (op(A)=A^H) and trans='N' are equivalent.
    // Full side x uplo x trans enumeration: 2 x 2 x 2 = 8 cases.
    const char sides[]  = {'L', 'R'};
    const char uplos[]  = {'L', 'U'};
    const char transes[] = {'N', 'C'};
    for(char side : sides)
        for(char uplo : uplos)
            for(char trans : transes)
                run_case(side, uplo, trans);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_ppotrs", check_ppotrs);
}
