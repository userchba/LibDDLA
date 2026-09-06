#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_pgeadd(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    const Complex alpha(2.0, -0.5);
    const Complex beta(-0.75, 1.25);
    auto run_case = [&](char transa, char transb, int m, int n){
        const int a_rows = transa == 'N' ? m : n;
        const int a_cols = transa == 'N' ? n : m;
        const int b_rows = transb == 'N' ? m : n;
        const int b_cols = transb == 'N' ? n : m;

        int descA[ddla::DDLA_DLEN_], descB[ddla::DDLA_DLEN_], descC[ddla::DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(descA, handle, a_rows, a_cols, nb, nb, 0, 0));
        DDLA_CHECK(ddlaDescInit(descB, handle, b_rows, b_cols, nb, nb, 0, 0));
        DDLA_CHECK(ddlaDescInit(descC, handle, m, n, nb, nb, 0, 0));

        auto h_A = make_local<Complex>(handle, descA, [](int i, int j){ return general_value(i, j, 4); });
        auto h_B = make_local<Complex>(handle, descB, [](int i, int j){ return general_value(i, j, 5); });
        std::vector<Complex> h_C(local_size(handle, descC), Complex(0.0, 0.0));

        DeviceBuffer<Complex> d_A(handle, h_A.size());
        DeviceBuffer<Complex> d_B(handle, h_B.size());
        DeviceBuffer<Complex> d_C(handle, h_C.size());
        upload(handle, d_A.ptr, h_A);
        upload(handle, d_B.ptr, h_B);
        upload(handle, d_C.ptr, h_C);
        check_ddla_sync(handle);

        ddla::pgeadd(handle, transa, transb, m, n, alpha, d_A.ptr, descA,
                     beta, d_B.ptr, descB, d_C.ptr, descC);
        auto out = download(handle, d_C.ptr, h_C.size());

        const double err = local_max_error<Complex>(handle, descC, out, [&](int i, int j){
            return alpha * op_value(transa, a_rows, a_cols, i, j, general_value, 4)
                 + beta * op_value(transb, b_rows, b_cols, i, j, general_value, 5);
        });
        std::string name = std::string("pgeadd(handle, ") + transa + "," + transb + ")";
        require_close(handle, name, err, 2e-10);
    };

    const int m_rect = round_up_for_grid(base.m, nb, nprows);
    const int n_rect = round_up_for_grid(base.n, nb, npcols);
    run_case('N', 'N', m_rect, n_rect);

    if(nprows == npcols){
        const int n_square = round_up_for_grid(std::max(base.m, base.n), nb, nprows);
        run_case('C', 'N', n_square, n_square);
        run_case('N', 'C', n_square, n_square);
        run_case('N', 'T', n_square, n_square);
        run_case('T', 'C', n_square, n_square);
    }
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pgeadd", check_pgeadd);
}
