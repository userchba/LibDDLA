#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_ptran(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    const int m = round_up_for_grid(base.m, nb, nprows);
    const int n = round_up_for_grid(base.n, nb, npcols);
    int descA[ddla::DDLA_DLEN_], descAT[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, handle, m, n, nb, nb, 0, 0));
    DDLA_CHECK(ddlaDescInit(descAT, handle, n, m, nb, nb, 0, 0));

    auto h_A = make_local<Complex>(handle, descA, [](int i, int j){ return general_value(i, j, 6); });
    std::vector<Complex> h_AT(local_size(handle, descAT), Complex(0.0, 0.0));
    DeviceBuffer<Complex> d_A(handle, h_A.size());
    DeviceBuffer<Complex> d_AT(handle, h_AT.size());
    upload(handle, d_A.ptr, h_A);
    upload(handle, d_AT.ptr, h_AT);
    check_ddla_sync(handle);

    ddla::ptran(handle, d_A.ptr, descA, d_AT.ptr, descAT, true);
    auto out = download(handle, d_AT.ptr, h_AT.size());
    const double err = local_max_error<Complex>(handle, descAT, out, [&](int i, int j){
        return std::conj(general_value(j, i, 6));
    });
    require_close(handle, "ptran(handle, conj)", err, 1e-12);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_ptran", check_ptran);
}
