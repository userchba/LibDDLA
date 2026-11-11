#include "PBLAS/pdgmm_impl.h"
#include "PBLAS/plapiv_impl.h"
#include "PBLAS/pswap_impl.h"
#include "PBLAS/ptrtrs_impl.h"
#include "ldlt_distributed.h"
#include "instantiate.h"
#include <array>
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
    if constexpr (Backend == DdlaBackend::CPU && Pivot)
    {
        if (h->nprocs == 1)
        {
            ldlt_impl::check_pivots(u, n, ipiv);
            ldlt_impl::solve(std::bool_constant<H>{}, u, n, nrhs, a, da[DDLA_LLD_], ipiv, b, db[DDLA_LLD_],
                             info);
            return;
        }
    }
    std::vector<int> piv, permutation, where, order, swaps;
    bool pairs = false, convert = false, permute = false;
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
        vector<T> local(h), global(h);
        detail::agree_check(h, "phetrs diagonal storage",
                            [&](detail::Agree&)
                            {
                                host_d.resize((Pivot ? 2 : 1) * std::size_t(n));
                                local.resize(host_d.size());
                                global.resize(host_d.size());
                            });
        DDLA_CHECK(ddlaMemset(h, local.data(), 0, local.size() * sizeof(T)));
        for (int j = 0; j < n;)
        {
            const int width =
                std::min({n - j, ldlt_impl::ldlt_panel_width, dt[DDLA_MB_] - j % dt[DDLA_MB_],
                          dt[DDLA_NB_] - j % dt[DDLA_NB_]});
            const int row = indx_g2l_r(dt, h, j), col = indx_g2l_c(dt, h, j);
            if (row >= 0 && col >= 0)
                DDLA_CHECK(ddlaMemcpy2D(h, local.data() + j, sizeof(T),
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
                        DDLA_CHECK(ddlaMemcpy(h, local.data() + n + i, entry, sizeof(T)));
                        DDLA_CHECK(ddlaMemset(h, entry, 0, sizeof(T)));
                    }
                    ++i;
                }
        commReduceSum<Backend>(h, CommScope::Grid, local.data(), global.data(), global.size(), 0);
        commBcast<Backend>(h, CommScope::Grid, global.data(), global.size(), 0);
        DDLA_CHECK(ddlaMemcpy(h, host_d.data(), global.data(), host_d.size() * sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
    }
    std::vector<D> inverse;
    std::vector<T> off;
    std::vector<int> pair_swap;
    detail::agree_check(h, "phetrs diagonal coefficients",
                        [&](detail::Agree&)
                        {
                            inverse.resize(n);
                            off.resize(pairs ? n : 0, T(0));
                            pair_swap.resize(pairs ? n : 0);
                        });
    std::iota(pair_swap.begin(), pair_swap.end(), 1);
    for (int i = 0; i < n; ++i)
    {
        const auto diagonal = [&](int j) -> D {
            if constexpr (H)
                return std::real(host_d[j]);
            else
                return host_d[j];
        };
        const D d0 = diagonal(i);
        if constexpr (!Pivot)
            if (d0 == R(0))
            {
                info = i + 1;
                return;
            }
        if constexpr (Pivot)
            if (piv[i] < 0)
            {
                const T stored = host_d[n + i];
                const T e = H && u == 'U' ? detail::host_conj(stored) : stored;
                const R scale = std::abs(e);
                const D a0 = d0 / scale, a1 = diagonal(i + 1) / scale;
                const T z = e / scale;
                const D determinant = [&]() -> D {
                    if constexpr (H)
                        return a0 * a1 - R(1);
                    else
                        return a0 * a1 - z * z;
                }();
                const D multiplier = (D(1) / determinant) / scale;
                inverse[i] = a1 * multiplier;
                inverse[i + 1] = a0 * multiplier;
                off[i + 1] = -z * multiplier;
                off[i] = H ? detail::host_conj(off[i + 1]) : off[i + 1];
                pair_swap[i] = i + 2;
                ++i;
                continue;
            }
        inverse[i] = R(1) / d0;
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
    vector<D> device_inverse(h);
    vector<T> device_off(h);
    std::vector<int> local_piv, local_pair;
    detail::agree_check(
        h, "phetrs solve workspace",
        [&](detail::Agree&)
        {
            work.resize(device_bytes);
            host.resize(host_bytes);
            device_inverse.resize(n);
            device_off.resize(off.size());
            DDLA_CHECK(
                ddlaMemcpy(h, device_inverse.data(), inverse.data(), inverse.size() * sizeof(D)));
            if (pairs)
                DDLA_CHECK(ddlaMemcpy(h, device_off.data(), off.data(), off.size() * sizeof(T)));
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
    if (pairs)
    {
        add(x, dx, nullptr, dx, spare.data(), ds, false);
        apply_pivots('F', spare.data(), ds, local_pair);
        pdgmm_impl::pdgmm_run<Backend, T, T>(h, 'L', n, nrhs, spare.data(), ds, device_off.data(),
                                             1, spare.data(), ds);
    }
    pdgmm_impl::pdgmm_run<Backend, T, D>(h, 'L', n, nrhs, x, dx, device_inverse.data(), 1, x, dx);
    if (pairs)
        add(x, dx, spare.data(), ds, x, dx, true);
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
