#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_pgetrf_nopiv(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    const int n = square_size(handle, base);
    const int nrhs = nrhs_size(base);
    int descA[ddla::DDLA_DLEN_], descB[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, handle, n, n, nb, nb, 0, 0));
    DDLA_CHECK(ddlaDescInit(descB, handle, n, nrhs, nb, nb, 0, 0));

    auto h_A = make_local<Complex>(handle, descA, [=](int i, int j){ return dominant_value(i, j, n); });
    auto h_B = build_rhs(handle, descB, n, dominant_value, n);

    DeviceBuffer<Complex> d_A(handle, h_A.size());
    DeviceBuffer<Complex> d_B(handle, h_B.size());
    upload(handle, d_A.ptr, h_A);
    upload(handle, d_B.ptr, h_B);
    check_ddla_sync(handle);

    int info = -1;
    ddla::pgetrf_nopiv(handle, n, n, d_A.ptr, descA, info);
    if(info != 0) MPI_Abort(ddlaGetCommunicator(handle), 1);
    ddla::pgetrs_nopiv(handle, 'L', 'N', n, nrhs, d_A.ptr, descA, d_B.ptr, descB);
    check_solution(handle, descB, d_B.ptr, h_B.size(), "pgetrf_nopiv", 5e-9);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pgetrf_nopiv", check_pgetrf_nopiv);
}
