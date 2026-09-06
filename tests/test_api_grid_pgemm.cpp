#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_pgemm(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    const int m = round_up_for_grid(base.m, nb, nprows);
    const int n = round_up_for_grid(base.n, nb, npcols);
    const int k = std::max(base.k, nb * std::max(nprows, npcols) + 1);
    const Complex alpha(0.8, -0.2);
    const Complex beta(-0.3, 0.1);

    for(char transa : {'N', 'T', 'C'}){
        for(char transb : {'N', 'T', 'C'}){
            const int a_rows = transa == 'N' ? m : k;
            const int a_cols = transa == 'N' ? k : m;
            const int b_rows = transb == 'N' ? k : n;
            const int b_cols = transb == 'N' ? n : k;

            int descA[ddla::DDLA_DLEN_], descB[ddla::DDLA_DLEN_], descC[ddla::DDLA_DLEN_];
            DDLA_CHECK(ddlaDescInit(descA, handle,
                                    a_rows, a_cols, nb, nb, g_test_irsrc, g_test_icsrc));
            DDLA_CHECK(ddlaDescInit(descB, handle,
                                    b_rows, b_cols, nb, nb, g_test_irsrc, g_test_icsrc));
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

            ddla::pgemm<>(handle, transa, transb, m, n, k, alpha,
                          d_A.ptr, descA, d_B.ptr, descB,
                          beta, d_C.ptr, descC);
            auto out = download(handle, d_C.ptr, h_C.size());

            const double err = local_max_error<Complex>(handle, descC, out, [&](int i, int j){
                Complex ref = beta * general_value(i, j, 3);
                for(int l = 0; l < k; ++l){
                    ref += alpha * op_value(transa, a_rows, a_cols, i, l, general_value, 1)
                         * op_value(transb, b_rows, b_cols, l, j, general_value, 2);
                }
                return ref;
            });
            std::string name = std::string("pgemm(handle, ") + transa + "," + transb + ")";
            require_close(handle, name, err, 2e-10);
        }
    }
}

// F2 regression: k==0 must reduce to C := beta*C with A/B never touched
// (the k-loop that used to perform the scale never runs when k==0).
void check_pgemm_k_zero(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    const int m = round_up_for_grid(base.m, nb, nprows);
    const int n = round_up_for_grid(base.n, nb, npcols);
    const Complex alpha(0.8, -0.2);
    const Complex beta(-0.3, 0.1);

    int descA[ddla::DDLA_DLEN_], descB[ddla::DDLA_DLEN_], descC[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, handle, m, 0, nb, nb, g_test_irsrc, g_test_icsrc));  // K == 0
    DDLA_CHECK(ddlaDescInit(descB, handle, 0, n, nb, nb, g_test_irsrc, g_test_icsrc));  // K == 0
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

    ddla::pgemm<>(handle, 'N', 'N', m, n, 0, alpha,
                  d_A.ptr, descA, d_B.ptr, descB,
                  beta, d_C.ptr, descC);
    auto out = download(handle, d_C.ptr, h_C.size());

    const double err = local_max_error<Complex>(handle, descC, out, [&](int i, int j){
        return beta * general_value(i, j, 3);
    });
    require_close(handle, "pgemm(handle, k=0)", err, 2e-10);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pgemm",
                          [](const ddla::DdlaHandle_t& handle, const Shape& base){
                              check_pgemm(handle, base);
                              check_pgemm_k_zero(handle, base);
                          });
}
