#include "PBLAS/plapiv_impl.h"
#include "PBLAS/pswap_impl.h"
#include "PBLAS/ptrtrs_impl.h"
#include "ldlt_distributed.h"
#include "ldlt_pair_solve.h"
#include "host_parallel.h"
#include "instantiate.h"
#include <array>
#include <climits>
#include <cmath>
#include <limits>
#include <ddla/ddla.h>
#include <numeric>

namespace ddla::ldlt_impl
{
// LAPACK HETRF stores D's off-diagonal blocks and interleaved permutations
// in the triangle. Separate them as HETRS2/SYCONV do before unit TRSM.
template <DdlaBackend Backend, bool H, bool Pivot, class T>
void distributed_solve(const DdlaHandle_t& h, char u, int n, int nrhs, const T* a, const int* da,
                     const int* ipiv, T* b, const int* db, int& info)
{
    using R = typename real_type<T>::type;
    using D = std::conditional_t<H, R, T>;
    u = ldlt_impl::distributed_check<Backend, H, Pivot, true, T>(
        h, u, n, nrhs, const_cast<T*>(a), da, const_cast<int*>(ipiv), b, db, info);
    if (!n || !nrhs)
        return;
    const auto tiny = [](R magnitude) {
        return magnitude > R(0) && magnitude < std::numeric_limits<R>::min();
    };
    const auto diagonal_value = [](T value) -> D {
        if constexpr (H) return std::real(value);
        else return value;
    };
    if constexpr (Backend == DdlaBackend::CPU && Pivot)
    {
        if (h->nprocs == 1)
        {
            ldlt_impl::check_pivots(u, n, ipiv);
            // Native SYTRS/HETRS can multiply by 1/D for scalar pivots.
            // Its two-by-two solve divides the RHS by the off-diagonal first.
            bool needs_scaling = false;
            R smallest_scalar = R(1);
            const int ld = da[DDLA_LLD_];
            R pair_rhs_bound = std::numeric_limits<R>::max();
            for (int i = 0; i < n && !needs_scaling; ++i) {
                const D diagonal = diagonal_value(a[i + std::size_t(i) * ld]);
                if (ipiv[i] > 0) {
                    const R magnitude = std::abs(diagonal);
                    needs_scaling = tiny(magnitude) || magnitude > std::numeric_limits<R>::max() / R(4);
                    smallest_scalar = std::min(smallest_scalar, magnitude);
                } else {
                    const T coupling = u == 'L' ? a[i + 1 + std::size_t(i) * ld]
                                                : a[i + std::size_t(i + 1) * ld];
                    const D partner = diagonal_value(a[i + 1 + std::size_t(i + 1) * ld]);
                    needs_scaling = !ordinary_pair_value(pair_components<R>(diagonal)) ||
                                    !ordinary_pair_value(pair_components<R>(partner)) ||
                                    !ordinary_pair_value(pair_components<R>(coupling)) || coupling == T(0);
                    pair_rhs_bound = std::min(pair_rhs_bound,
                        std::min(R(1), std::abs(coupling)) * (std::numeric_limits<R>::max() / R(4)));
                    ++i;
                }
            }
            // Large RHS values can overflow reciprocal multiplication even
            // with normal D. Preserve the native shortcut for ordinary data.
            if ((smallest_scalar < R(1) || pair_rhs_bound < std::numeric_limits<R>::max()) &&
                !needs_scaling) {
                const R bound = std::min(pair_rhs_bound, smallest_scalar * (std::numeric_limits<R>::max() / R(4)));
                for (int j = 0; j < nrhs && !needs_scaling; ++j)
                    for (int i = 0; i < n && !needs_scaling; ++i) {
                        const T value = b[i + std::size_t(j) * db[DDLA_LLD_]];
                        needs_scaling = std::max(std::abs(std::real(value)), std::abs(std::imag(value))) > bound;
                    }
            }
            if (!needs_scaling) {
                ldlt_impl::solve(std::bool_constant<H>{}, u, n, nrhs, a, da[DDLA_LLD_], ipiv, b,
                                 db[DDLA_LLD_], info);
                return;
            }
        }
    }
    std::vector<int> piv, permutation, where, order, swaps;
    bool pairs = false, scalars = !Pivot, convert = false, permute = false;
    detail::agree_check(h, "phetrs metadata",
                        [&](detail::Agree&)
                        {
                            if constexpr (Pivot)
                            {
                                piv.resize(n);
                                permutation.resize(n);
                                where.resize(n);
                                order.resize(n);
                                swaps.resize(n);
                                std::iota(permutation.begin(), permutation.end(), 0);
                                if (!h->myid)
                                    std::copy_n(ipiv, n, piv.data());
                            }
                        });
    if constexpr (Pivot)
    {
        MPI_CHECK(MPI_Bcast(piv.data(), n, MPI_INT, 0, h->comm));
        ldlt_impl::check_pivots(u, n, piv.data());
        for (int done = 0; done < n;)
        {
            const int k = u == 'L' ? done : n - 1 - done;
            const int step = piv[k] < 0 ? 2 : 1;
            const int row = step == 2 ? k + (u == 'L' ? 1 : -1) : k;
            const int target = std::abs(piv[k]) - 1;
            pairs |= step == 2;
            scalars |= step == 1;
            convert |= step == 2 || row != target;
            std::swap(permutation[row], permutation[target]);
            done += step;
        }
    }
    // PLAPIV uses LU-style positive, row-cyclic pivots and omits a final
    // identity swap. Encode the complete BK permutation in that form; this
    // also covers upper factors whose first swap involves global row n-1.
    if constexpr (Pivot)
    {
        std::iota(order.begin(), order.end(), 0);
        where = order;
        for (int i = 0; i < n; ++i)
        {
            const int target = where[permutation[i]];
            swaps[i] = target + 1;
            permute |= target != i;
            const int old = order[i];
            std::swap(order[i], order[target]);
            where[old] = target;
            where[permutation[i]] = i;
        }
    }
    const bool square = da[DDLA_MB_] == da[DDLA_NB_] && da[DDLA_MB_] <= ldlt_impl::ldlt_panel_width;
    const bool copy_a = convert || !square || (n - 1) / da[DDLA_MB_] + 1 < h->nprows_ ||
                        (n - 1) / da[DDLA_NB_] + 1 < h->npcols_;
    int dt[DDLA_DLEN_], dx[DDLA_DLEN_], ds[DDLA_DLEN_];
    std::copy_n(da, DDLA_DLEN_, dt);
    vector<T> factors(h), rhs(h), spare(h);
    bool copy_rhs = false;
    T* triangle = const_cast<T*>(a);
    T* x = b;
    detail::agree_check(
        h, "phetrs distributed storage",
        [&](detail::Agree&)
        {
            if (copy_a)
            {
                const int block = square ? da[DDLA_MB_] : ldlt_impl::ldlt_panel_width;
                DDLA_CHECK(ddlaDescInit(dt, h, n, n, block, block, da[DDLA_RSRC_], da[DDLA_CSRC_]));
                factors.resize(
                    std::size_t(dt[DDLA_LLD_]) *
                    std::max(1, num_loc(n, block, h->mypcol_, dt[DDLA_CSRC_], h->npcols_)));
                triangle = factors.data();
                DDLA_CHECK(ddlaMemset(h, triangle, 0, factors.size() * sizeof(T)));
            }
            std::copy_n(db, DDLA_DLEN_, dx);
            copy_rhs = db[DDLA_MB_] != dt[DDLA_MB_] || db[DDLA_RSRC_] != dt[DDLA_RSRC_] ||
                       (n - 1) / db[DDLA_MB_] + 1 < h->nprows_ ||
                       (nrhs - 1) / db[DDLA_NB_] + 1 < h->npcols_;
            if (copy_rhs)
            {
                DDLA_CHECK(
                    ddlaDescInit(dx, h, n, nrhs, dt[DDLA_MB_],
                                 std::min({db[DDLA_NB_], ldlt_impl::ldlt_panel_width,
                                           std::max(1, (nrhs + h->npcols_ - 1) / h->npcols_)}),
                                 dt[DDLA_RSRC_], db[DDLA_CSRC_]));
                rhs.resize(std::size_t(dx[DDLA_LLD_]) *
                           std::max(1, num_loc(nrhs, dx[DDLA_NB_], h->mypcol_, dx[DDLA_CSRC_],
                                               h->npcols_)));
                x = rhs.data();
            }
            DDLA_CHECK(ddlaDescInit(ds, h, n, nrhs, dx[DDLA_MB_], dx[DDLA_NB_], dx[DDLA_RSRC_],
                                    dx[DDLA_CSRC_]));
            if (pairs)
                spare.resize(std::size_t(ds[DDLA_LLD_]) *
                             std::max(1, num_loc(nrhs, ds[DDLA_NB_], h->mypcol_, ds[DDLA_CSRC_],
                                                 h->npcols_)));
        });
    if (copy_a)
        ldlt_impl::copy_factor<Backend, H, T>(h, u, n, a, da, triangle, dt, false);

    std::vector<T> host_d;
    {
        vector<T> diagonal(h);
        detail::agree_check(h, "phetrs diagonal storage",
                            [&](detail::Agree&)
                            {
                                host_d.resize((Pivot ? 2 : 1) * std::size_t(n));
                                diagonal.resize(host_d.size());
                            });
        DDLA_CHECK(ddlaMemset(h, diagonal.data(), 0, diagonal.size() * sizeof(T)));
        for (int j = 0; j < n;)
        {
            const int width =
                std::min({n - j, ldlt_impl::ldlt_panel_width, dt[DDLA_MB_] - j % dt[DDLA_MB_],
                          dt[DDLA_NB_] - j % dt[DDLA_NB_]});
            const int row = indx_g2l_r(dt, h, j), col = indx_g2l_c(dt, h, j);
            if (row >= 0 && col >= 0)
                DDLA_CHECK(ddlaMemcpy2D(h, diagonal.data() + j, sizeof(T),
                                        triangle + row + std::size_t(col) * dt[DDLA_LLD_],
                                        std::size_t(dt[DDLA_LLD_] + 1) * sizeof(T), sizeof(T),
                                        width));
            j += width;
        }
        for (int i = 0; i < n; ++i)
            if constexpr (Pivot)
                if (piv[i] < 0)
                {
                    const int gr = u == 'L' ? i + 1 : i, gc = u == 'L' ? i : i + 1;
                    const int row = indx_g2l_r(dt, h, gr), col = indx_g2l_c(dt, h, gc);
                    if (row >= 0 && col >= 0)
                    {
                        T* entry = triangle + row + std::size_t(col) * dt[DDLA_LLD_];
                        DDLA_CHECK(ddlaMemcpy(h, diagonal.data() + n + i, entry, sizeof(T)));
                        DDLA_CHECK(ddlaMemset(h, entry, 0, sizeof(T)));
                    }
                    ++i;
                }
        // Complex all-reduce can use two real components per entry.
        for (std::size_t offset = 0; offset < diagonal.size();) {
            const int count = static_cast<int>(std::min(diagonal.size() - offset, std::size_t(INT_MAX / 2)));
            T* chunk = diagonal.data() + offset;
            commAllReduce<Backend>(h, CommScope::Grid, chunk, chunk, count, cclSum);
            offset += count;
        }
        DDLA_CHECK(ddlaMemcpy(h, host_d.data(), diagonal.data(), host_d.size() * sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
    }
    std::vector<D> local_diagonal;
    std::vector<PairRow<R>> pair_rows;
    std::vector<int> pair_swap;
    const int local_rows = num_loc(n, dx[DDLA_MB_], h->myprow_, dx[DDLA_RSRC_], h->nprows_);
    detail::agree_check(h, "phetrs diagonal coefficients", [&](detail::Agree&) {
        local_diagonal.resize(scalars ? local_rows : 0, D(1));
        if (pairs) {
            pair_rows.reserve(local_rows);
            pair_swap.resize(n);
        }
    });
    std::iota(pair_swap.begin(), pair_swap.end(), 1);
    const auto add_pair_row = [&](int global, D diagonal, D partner_diagonal, T coupling) {
        const int row = indx_g2l_r(dx, h, global);
        if (row >= 0)
            pair_rows.push_back({row, pair_components<R>(diagonal),
                                 pair_components<R>(partner_diagonal), pair_components<R>(coupling)});
    };
    for (int i = 0; i < n; ++i) {
        const D diagonal = diagonal_value(host_d[i]);
        if constexpr (!Pivot)
            if (diagonal == R(0)) {
                info = i + 1;
                return;
            }
        if constexpr (Pivot)
            if (piv[i] < 0) {
                const D partner = diagonal_value(host_d[i + 1]);
                const T stored = host_d[n + i];
                const T other = H ? detail::host_conj(stored) : stored;
                add_pair_row(i, diagonal, partner, u == 'U' ? stored : other);
                add_pair_row(i + 1, partner, diagonal, u == 'L' ? stored : other);
                pair_swap[i] = i + 2;
                ++i;
                continue;
            }
        const int row = indx_g2l_r(dx, h, i);
        if (row >= 0)
            local_diagonal[row] = diagonal;
    }
    std::size_t device_bytes = 0, host_bytes = 0;
    if (h->nprocs > 1)
        ptrtrs_impl::ptrtrs_workspace_size<T>(h, 'L', H ? 'C' : 'T', n, nrhs, dt, dx, &device_bytes,
                                              &host_bytes);
    std::size_t count = 0;
    if (convert)
    {
        pswap_impl::pswap_workspace_size<T>(h, dt, &count);
        device_bytes = std::max(device_bytes, count);
    }
    if (permute || pairs)
    {
        plapiv_impl::plapiv_workspace_size<T>(h, nrhs, dx, &count);
        device_bytes = std::max(device_bytes, count);
    }
    if (copy_rhs || pairs)
        for (const auto& route : std::array<std::pair<const int*, const int*>, 4>{
                 {{db, dx}, {dx, db}, {dx, ds}, {ds, dx}}})
        {
            std::size_t dbs = 0, hbs = 0;
            pgeadd_impl::pgeadd_workspace_size<T>(
                h, 'N', 'N', true, true, n, std::min(nrhs, ldlt_impl::ldlt_panel_width),
                route.second, &dbs, &hbs, route.first, route.second, true);
            device_bytes = std::max(device_bytes, dbs);
            host_bytes = std::max(host_bytes, hbs);
        }
    vector<char> work(h);
    std::vector<char> host;
    vector<D> device_diagonal(h);
    vector<PairRow<R>> device_pairs(h);
    std::vector<int> local_piv, local_pair;
    detail::agree_check(
        h, "phetrs solve workspace",
        [&](detail::Agree&)
        {
            work.resize(device_bytes);
            host.resize(host_bytes);
            device_diagonal.resize(local_diagonal.size());
            DDLA_CHECK(ddlaMemcpy(h, device_diagonal.data(), local_diagonal.data(),
                                  local_diagonal.size() * sizeof(D)));
            if constexpr (Backend == DdlaBackend::GPU)
                if (!pair_rows.empty()) {
                    device_pairs.resize(pair_rows.size());
                    DDLA_CHECK(ddlaMemcpy(h, device_pairs.data(), pair_rows.data(),
                                          pair_rows.size() * sizeof(PairRow<R>)));
                }
            const int rows = num_loc(n, dx[DDLA_MB_], h->myprow_, dx[DDLA_RSRC_], h->nprows_);
            local_piv.resize(permute ? rows : 0);
            local_pair.resize(pairs ? rows : 0);
            for (int i = 0; i < rows; ++i)
            {
                const int global = indx_l2g_r(dx, h, i);
                if (permute)
                    local_piv[i] = swaps[global];
                if (pairs)
                    local_pair[i] = pair_swap[global];
            }
        });
    auto exchange = [&](T* data, const int* d, int row, int other, int first, int width)
    {
        if (row != other && width)
            pswap_impl::pswap_run<Backend, T>(h, width, data, row + 1, first + 1, d, d[DDLA_M_],
                                              data, other + 1, first + 1, d, d[DDLA_M_],
                                              work.data(), work.size());
    };
    if constexpr (Pivot)
        if (convert)
            for (int done = 0; done < n;)
            {
                const int k = u == 'L' ? done : n - 1 - done, step = piv[k] < 0 ? 2 : 1;
                const int row = step == 2 ? k + (u == 'L' ? 1 : -1) : k;
                exchange(triangle, dt, row, std::abs(piv[k]) - 1, u == 'L' ? 0 : k + 1,
                         u == 'L' ? k : n - k - 1);
                done += step;
            }
    auto add = [&](const T* source, const int* source_desc, const T* second, const int* second_desc,
                   T* dest, const int* dest_desc, bool sum)
    {
        for (int j = 0; j < nrhs; j += ldlt_impl::ldlt_panel_width)
        {
            const int width = std::min(ldlt_impl::ldlt_panel_width, nrhs - j);
            pgeadd_impl::pgeadd_run<Backend, T>(h, 'N', 'N', n, width, T(1), source, 1, j + 1,
                                                source_desc, sum ? T(1) : T(0), second, 1, j + 1,
                                                second_desc, dest, 1, j + 1, dest_desc, work.data(),
                                                work.size(), host.data(), host.size());
        }
    };
    auto apply_pivots = [&](char direction, T* data, const int* d, const std::vector<int>& p)
    {
        plapiv_impl::plapiv_run<Backend, T>(h, direction, 'R', n, nrhs, data, d, p.data(), d,
                                            work.data(), work.size());
    };
    if (copy_rhs)
        add(b, db, nullptr, db, x, dx, false);
    if (permute)
        apply_pivots('F', x, dx, local_piv);
    auto triangular_solve = [&](char trans)
    {
        if (h->nprocs == 1)
            blas_impl::trsm_run<Backend, T>(h, 'L', u, trans, 'U', n, nrhs, T(1), triangle,
                                            dt[DDLA_LLD_], x, dx[DDLA_LLD_]);
        else
            ptrtrs_impl::ptrtrs_run<Backend, T>(h, 'L', u, trans, 'U', n, nrhs, triangle, dt, x, dx,
                                                work.data(), work.size(), host.data(), host.size());
    };
    triangular_solve('N');
    const int columns = num_loc(nrhs, dx[DDLA_NB_], h->mypcol_, dx[DDLA_CSRC_], h->npcols_);
    blas_impl::rscl_rows_run<Backend, T, D>(
        h, int(local_diagonal.size()), columns, device_diagonal.data(), x, dx[DDLA_LLD_]);
    if (pairs) {
        add(x, dx, nullptr, dx, spare.data(), ds, false);
        apply_pivots('F', spare.data(), ds, local_pair);
        // Each row reads its original value and the exchanged partner. Both
        // outputs can be written in place without seeing an updated RHS.
        if constexpr (Backend == DdlaBackend::CPU) {
            detail::host_parallel_for(columns, 1, h->host_threads, [&](long long first, long long end) {
                for (auto j = first; j < end; ++j) {
                    T* column = x + std::size_t(j) * dx[DDLA_LLD_];
                    const T* partner_column = spare.data() + std::size_t(j) * ds[DDLA_LLD_];
                    for (const auto& row : pair_rows)
                        column[row.row] = solve_pair_row<H>(row, column[row.row], partner_column[row.row]);
                }
            });
        } else {
#if DDLA_HAS_GPU
            solve_pair_rows_gpu<H>(h, static_cast<int>(pair_rows.size()), columns, device_pairs.data(),
                                   x, dx[DDLA_LLD_], spare.data(), ds[DDLA_LLD_]);
#endif
        }
    }
    triangular_solve(H ? 'C' : 'T');
    if (permute)
        apply_pivots('B', x, dx, local_piv);
    if (copy_rhs)
        add(x, dx, nullptr, dx, b, db, false);
    DDLA_CHECK(ddlaSynchronize(h));
}
#define ENGINE(B, T) \
    template void distributed_solve<B, false, true, T>(const DdlaHandle_t&, char, int, int, const T*, const int*, const int*, T*, const int*, int&); \
    template void distributed_solve<B, false, false, T>(const DdlaHandle_t&, char, int, int, const T*, const int*, const int*, T*, const int*, int&)
DDLA_FOR_EACH_BACKEND_TYPE(ENGINE)
#undef ENGINE
} // namespace ddla::ldlt_impl

namespace ddla {
template <DdlaBackend Backend, typename T>
void phetrs(const DdlaHandle_t& h, char u, int n, int nrhs, const T* a, const int* da,
            const int* piv, T* b, const int* db, int& info)
{
    ldlt_impl::distributed_solve<Backend, true, true, T>(h, u, n, nrhs, a, da, piv, b, db, info);
}
template <DdlaBackend Backend, typename T>
void phetrs_nopiv(const DdlaHandle_t& h, char u, int n, int nrhs, const T* a, const int* da, T* b,
                  const int* db, int& info)
{
    ldlt_impl::distributed_solve<Backend, true, false, T>(h, u, n, nrhs, a, da, nullptr, b, db, info);
}

#define INSTANCE(B, T)                                                                             \
    template void phetrs<B, T>(const DdlaHandle_t&, char, int, int, const T*, const int*,          \
                               const int*, T*, const int*, int&);                                  \
    template void phetrs_nopiv<B, T>(const DdlaHandle_t&, char, int, int, const T*, const int*,    \
                                     T*, const int*, int&)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANCE)
#undef INSTANCE
} // namespace ddla
