#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_pgetrf_bpiv(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    const int n = square_size(handle, base);
    int descA[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, handle, n, n, nb, nb, 0, 0));

    // Run with the no-pivot matrix and with the row-cycled variant that
    // forces real row swaps in every diagonal block.
    for(auto gen : {dominant_value, cycled_value}){
        auto h_A = make_local<Complex>(handle, descA, [=](int i, int j){ return gen(i, j, n); });
        DeviceBuffer<Complex> d_A(handle, h_A.size());
        DeviceBuffer<int> d_ipiv(handle, std::max(1, ddla_test::m_loc(handle, descA)));
        upload(handle, d_A.ptr, h_A);
        check_ddla_sync(handle);

        int info = -1;
        ddla::pgetrf_bpiv(handle, n, n, d_A.ptr, descA, d_ipiv.ptr, info);
        if(info != 0) MPI_Abort(ddlaGetCommunicator(handle), 1);
        require_close(handle, "pgetrf_bpiv(handle, info)", 0.0, 0.0);
    }
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pgetrf_bpiv", check_pgetrf_bpiv);
}
