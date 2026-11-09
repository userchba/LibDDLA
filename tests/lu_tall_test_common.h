#pragma once
// Tall (m > n) regression for the block LU variants: the rows below the last
// diagonal block still need L21 = A21 * U1^{-1}.  The matrix is strongly
// diagonally dominant, so block-local pivoting never swaps and A == L*U holds
// with a unit lower-trapezoidal m x n L.  The factor is gathered on rank 0
// (test-sized matrices only) and multiplied out there.
#include "api_grid_test_common.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace api_grid_test {

template <class Factor>
void check_tall_lu(const ddla::DdlaHandle_t& handle, const Shape& base, const std::string& name, Factor factor)
{
    const int nb = base.nb;
    const int n = square_size(handle, base);
    const int m = n + 2 * nb + 3;
    int descA[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, handle, m, n, nb, nb, 0, 0));
    auto value = [=](int i, int j) { return dominant_value(i, j, m); };
    auto h_A = make_local<Complex>(handle, descA, value);
    DeviceBuffer<Complex> d_A(handle, h_A.size());
    upload(handle, d_A.ptr, h_A);
    check_ddla_sync(handle);

    int info = -1;
    factor(m, n, d_A.ptr, descA, info);

    const bool root = ddlaGetRank(handle) == 0;
    DeviceBuffer<Complex> d_full(handle, root ? std::size_t(m) * n : 1);
    ddla::gather(handle, m, n, d_A.ptr, 1, 1, descA, 0, root ? d_full.ptr : nullptr, m);
    check_ddla_sync(handle);

    double err = info == 0 ? 0.0 : std::numeric_limits<double>::infinity();
    if (root) {
        const auto f = download(handle, d_full.ptr, std::size_t(m) * n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                Complex s = 0;
                for (int k = 0; k <= std::min(i, j); ++k) {
                    const Complex l = k == i ? Complex(1) : f[i + std::size_t(k) * m];
                    s += l * f[k + std::size_t(j) * m];
                }
                const double e = std::abs(s - value(i, j));
                if (std::isnan(e) || e > err) err = e;
            }
    }
    require_close(handle, name + " m=" + std::to_string(m) + " n=" + std::to_string(n) + " (A == L*U)", err, 1e-10);
}

} // namespace api_grid_test
