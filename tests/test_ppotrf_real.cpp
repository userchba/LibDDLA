#include "api_grid_test_common.h"
#include "test_desc_helpers.h"

using namespace api_grid_test;

// Real (float/double) regression coverage for the Cholesky family.
// ppotrf/ppotrs/pposv were historically complex-only; these checks exercise
// the real instantiations added for full scalar-type coverage.

template <typename T>
inline T spd_value(int i, int j)
{
    if(i == j) return T(6.0 + 0.05 * i);
    const int lo = std::min(i, j);
    const int hi = std::max(i, j);
    return T(0.01 * ((lo + 2 * hi) % 5 - 2));   // symmetric, diagonally dominant
}

template <typename T>
inline T x_value_real(int i, int j)
{
    return T(0.2 + 0.03 * i - 0.02 * j);
}

// ppotrf + ppotrs: solve A X = B with B = A * X_true; verify X == X_true.
template <typename T>
void check_ppotrf_real(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    if(skip_non_square_grid(handle, "ppotrf_real")) return;

    const int nb = base.nb;
    const int n = square_size(handle, base);
    const int nrhs = nrhs_size(base);
    int descA[DDLA_DLEN_], descB[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, handle, n, n, nb, nb, 0, 0));
    DDLA_CHECK(ddlaDescInit(descB, handle, n, nrhs, nb, nb, 0, 0));

    auto h_A = make_local<T>(handle, descA, [&](int i, int j){ return spd_value<T>(i, j); });
    auto h_B = make_local<T>(handle, descB, [&](int i, int j){
        T sum(0.0);
        for(int l = 0; l < n; ++l){
            sum += spd_value<T>(i, l) * x_value_real<T>(l, j);
        }
        return sum;
    });

    DeviceBuffer<T> d_A(handle, h_A.size());
    DeviceBuffer<T> d_B(handle, h_B.size());
    upload(handle, d_A.ptr, h_A);
    upload(handle, d_B.ptr, h_B);
    check_ddla_sync(handle);

    int info = -1;
    const bool is_nega = ddla::ppotrf(handle, 'L', n, d_A.ptr, 1, 1, descA, info);
    require_close(handle, "ppotrf_real(status)",
                  std::abs(info) + (is_nega ? 1.0 : 0.0), 0.0);
    check_ddla_sync(handle);

    ddla::ppotrs(handle, 'L', 'L', 'N', n, nrhs, d_A.ptr, descA, d_B.ptr, descB, is_nega);
    check_ddla_sync(handle);

    auto x = download(handle, d_B.ptr, h_B.size());
    const double sol_err = local_max_error<T>(handle, descB, x, [](int i, int j){
        return x_value_real<T>(i, j);
    });
    const double tol = std::is_same_v<T, float> ? 3e-3 : 1e-9;
    require_close(handle, "ppotrf_real(solution)", sol_err, tol);
}

// pposv: same solve through the driver; verify X == X_true.
template <typename T>
void check_pposv_real(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    if(skip_non_square_grid(handle, "pposv_real")) return;

    const int nb = base.nb;
    const int n = square_size(handle, base);
    const int nrhs = nrhs_size(base);
    int descA[DDLA_DLEN_], descB[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, handle, n, n, nb, nb, 0, 0));
    DDLA_CHECK(ddlaDescInit(descB, handle, n, nrhs, nb, nb, 0, 0));

    auto h_A = make_local<T>(handle, descA, [&](int i, int j){ return spd_value<T>(i, j); });
    auto h_B = make_local<T>(handle, descB, [&](int i, int j){
        T sum(0.0);
        for(int l = 0; l < n; ++l){
            sum += spd_value<T>(i, l) * x_value_real<T>(l, j);
        }
        return sum;
    });

    DeviceBuffer<T> d_A(handle, h_A.size());
    DeviceBuffer<T> d_B(handle, h_B.size());
    upload(handle, d_A.ptr, h_A);
    upload(handle, d_B.ptr, h_B);
    check_ddla_sync(handle);

    int info = -1;
    ddla::pposv(handle, 'L', 'L', 'N', n, nrhs,
                d_A.ptr, 1, 1, descA,
                d_B.ptr, 1, 1, descB, info);
    require_close(handle, "pposv_real(status)", std::abs(info), 0.0);
    check_ddla_sync(handle);

    auto x = download(handle, d_B.ptr, h_B.size());
    const double sol_err = local_max_error<T>(handle, descB, x, [](int i, int j){
        return x_value_real<T>(i, j);
    });
    const double tol = std::is_same_v<T, float> ? 3e-3 : 1e-9;
    require_close(handle, "pposv_real(solution)", sol_err, tol);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_ppotrf_real", [](const ddla::DdlaHandle_t& h, const Shape& base){
        check_ppotrf_real<double>(h, base);
        check_ppotrf_real<float>(h, base);
        check_pposv_real<double>(h, base);
        check_pposv_real<float>(h, base);
    });
}
