#ifndef DDLA_LDLT_DISTRIBUTED_H
#define DDLA_LDLT_DISTRIBUTED_H
#include "BLAS/gemmt_impl.h"
#include "BLAS/hemmt_impl.h"
#include "PBLAS/pgeadd_impl.h"
#include "comm_traits.h"
#include "hetrf_native.h"
#include "ldlt_impl.h"
#include "matrix_transfer.h"
#include "plan_util.h"

namespace ddla::ldlt_impl
{
constexpr int ldlt_panel_width = 64;

// Map only this rank's part of a rectangle; global-order windows may be
// replicated vectors/panels. Reuses the gather/scatter mapping without MPI.
template <DdlaBackend B, class T>
void local_rectangle(const DdlaHandle_t& h, bool scatter, T* a, const int* d, int i, int j, int m,
                     int n, T* window, int ldw)
{
    const int r0 = num_loc(i, d[DDLA_MB_], h->myprow_, d[DDLA_RSRC_], h->nprows_);
    const int r1 = num_loc(i + m, d[DDLA_MB_], h->myprow_, d[DDLA_RSRC_], h->nprows_);
    const int c0 = num_loc(j, d[DDLA_NB_], h->mypcol_, d[DDLA_CSRC_], h->npcols_);
    const int c1 = num_loc(j + n, d[DDLA_NB_], h->mypcol_, d[DDLA_CSRC_], h->npcols_);
    if (r0 == r1 || c0 == c1)
        return;
    matrix_impl::Block<T> block{a + r0 + std::size_t(c0) * d[DDLA_LLD_],
                                d[DDLA_LLD_],
                                {r0, c0, r1 - r0, c1 - c0, h->myprow_, h->mypcol_}};
    matrix_impl::map_blocks<B, T>(h, scatter, &block, 1, d, i, j, m, n, window, ldw);
}

// Copy a stored triangle with bounded transport and optional Hermitian
// completion. Supports unrelated block sizes and sources, without touching
// the caller's unused triangle or gathering a dense matrix on one rank.
template <DdlaBackend B, class T>
void copy_factor(const DdlaHandle_t& h, char u, int n, const T* a, const int* da, T* f,
                 const int* df, bool complete)
{
    const int cap = std::min({ldlt_panel_width, da[DDLA_MB_], da[DDLA_NB_]});
    const bool same = da[DDLA_MB_] == df[DDLA_MB_] && da[DDLA_NB_] == df[DDLA_NB_] &&
                      da[DDLA_RSRC_] == df[DDLA_RSRC_] && da[DDLA_CSRC_] == df[DDLA_CSRC_];
    vector<char> device(h);
    std::vector<char> host;
    vector<T> diagonal(h);
    detail::agree_check(h, "LDLT triangle transport",
                        [&](detail::Agree&)
                        {
                            std::size_t db = 0, hb = 0;
                            for (char trans : {'N', 'C'})
                            {
                                if (trans == 'C' && !complete)
                                    continue;
                                std::size_t d = 0, s = 0;
                                pgeadd_impl::pgeadd_workspace_size<T>(
                                    h, trans, 'N', true, false, trans == 'N' ? n : cap,
                                    trans == 'N' ? cap : n, df, &d, &s, da, da, true);
                                db = std::max(db, d);
                                hb = std::max(hb, s);
                            }
                            device.resize(db);
                            host.resize(hb);
                            if (!same)
                                diagonal.resize(std::size_t(cap) * cap);
                        });
    for (int j = 0; j < n;)
    {
        const int b = std::min(
            {cap, n - j, da[DDLA_MB_] - j % da[DDLA_MB_], da[DDLA_NB_] - j % da[DDLA_NB_]});
        const int r = indx_g2l_r(da, h, j), c = indx_g2l_c(da, h, j);
        if (same)
        {
            if (r >= 0 && c >= 0)
                blas_impl::diagonal_block_run<B, T>(
                    h, u == 'U', true, b, a + r + std::size_t(c) * da[DDLA_LLD_], da[DDLA_LLD_],
                    f + r + std::size_t(c) * df[DDLA_LLD_], df[DDLA_LLD_]);
        }
        else
        {
            const int owner = h->rc_to_rank(indxg2p(j, da[DDLA_MB_], da[DDLA_RSRC_], h->nprows_),
                                            indxg2p(j, da[DDLA_NB_], da[DDLA_CSRC_], h->npcols_));
            if (h->myid == owner)
                blas_impl::diagonal_block_run<B, T>(h, u == 'U', true, b,
                                                    a + r + std::size_t(c) * da[DDLA_LLD_],
                                                    da[DDLA_LLD_], diagonal.data(), b);
            commBcast<B>(h, CommScope::Grid, diagonal.data(), std::size_t(b) * b, owner);
            local_rectangle<B, T>(h, true, f, df, j, j, b, b, diagonal.data(), b);
        }
        const int first = u == 'L' ? j + b : 0, count = u == 'L' ? n - first : j;
        if (count)
        {
            pgeadd_impl::pgeadd_run<B, T>(h, 'N', 'N', count, b, T(1), a, first + 1, j + 1, da,
                                          T(0), nullptr, 1, 1, da, f, first + 1, j + 1, df,
                                          device.data(), device.size(), host.data(), host.size());
            if (complete)
                pgeadd_impl::pgeadd_run<B, T>(h, 'C', 'N', b, count, T(1), a, first + 1, j + 1, da,
                                              T(0), nullptr, 1, 1, da, f, j + 1, first + 1, df,
                                              device.data(), device.size(), host.data(),
                                              host.size());
        }
        j += b;
    }
    DDLA_CHECK(ddlaSynchronize(h));
}
// Backend leaves shared by the two distributed panel methods. CPU uses
// existing BLAS/no-pivot kernels; GPU keeps the native fused operations.
template <DdlaBackend B, class T>
void diagonal_factor(const DdlaHandle_t& h, char u, int n, T* a, int ld, int* info)
{
    if constexpr (B == DdlaBackend::CPU)
    {
        *info = 0;
        factor_nopiv<true>(h, u, n, a, ld, *info);
    }
    else
        nopiv_diagonal(h, u, n, a, ld, info);
}
template <DdlaBackend B, class T>
void diagonal_divide(const DdlaHandle_t& h, int rows, int cols, const T* d, int ldd, const T* w,
                     int ldw, T* l, int ldl)
{
    if constexpr (B == DdlaBackend::CPU)
    {
        DDLA_CHECK(ddlaMemcpy2D(h, l, std::size_t(ldl) * sizeof(T), w, std::size_t(ldw) * sizeof(T),
                                std::size_t(rows) * sizeof(T), cols));
        for (int j = 0; j < cols; ++j)
            blas_impl::scal_run<B, T>(h, rows, T(1) / T(std::real(d[j + std::size_t(j) * ldd])),
                                      l + std::size_t(j) * ldl, 1);
    }
    else
        divide_panel(h, rows, cols, d, ldd, w, ldw, l, ldl);
}
template <DdlaBackend B, class T>
void column_record(const DdlaHandle_t& h, T* c, int first, int count, int diagonal,
                   PivotSummary* result)
{
    if constexpr (B == DdlaBackend::CPU)
    {
        const T d = T(std::real(c[diagonal]));
        c[diagonal] = T(0);
        int index = 0;
        blas_impl::iamax_run<B, T>(h, count, c + first, 1, index);
        index += first - 1;
        const T z = c[index];
        double maximum;
        if constexpr (std::is_arithmetic_v<T>)
            maximum = std::abs(z);
        else
            maximum = std::abs(z.real()) + std::abs(z.imag());
        c[diagonal] = d;
        *result = {double(std::abs(std::real(d))), maximum, index, 0};
    }
    else
        summarize_column(h, c, first, count, diagonal, result);
}
template <DdlaBackend B, class T>
void normalize_columns(const DdlaHandle_t& h, int first, int count, int current, int neighbor,
                       int step, int swap_a, int swap_b, const T* c0, const T* c1, T* l0, T* l1,
                       T* w0, T* w1)
{
    if constexpr (B == DdlaBackend::CPU)
    {
        DDLA_CHECK(ddlaMemcpy(h, w0 + first, c0 + first, std::size_t(count) * sizeof(T)));
        blas_impl::swap_run<B, T>(h, 1, w0 + swap_a, 1, w0 + swap_b, 1);
        if (step == 1)
        {
            DDLA_CHECK(ddlaMemcpy(h, l0 + first, w0 + first, std::size_t(count) * sizeof(T)));
            const auto d = std::real(w0[current]);
            if (d != 0)
                blas_impl::scal_run<B, T>(h, count, T(1) / T(d), l0 + first, 1);
            l0[current] = T(d);
        }
        else
        {
            DDLA_CHECK(ddlaMemcpy(h, w1 + first, c1 + first, std::size_t(count) * sizeof(T)));
            blas_impl::swap_run<B, T>(h, 1, w1 + swap_a, 1, w1 + swap_b, 1);
            const auto d0 = std::real(w0[current]), d1 = std::real(w1[neighbor]);
            const T e = w0[neighbor];
            const auto scale = std::abs(e);
            const auto a = d0 / scale, b = d1 / scale, m = (1 / (a * b - 1)) / scale;
            const T lower = -(e / scale) * m;
            const T inverse[4] = {T(b * m), lower, detail::host_conj(lower), T(a * m)};
            blas_impl::gemm_run<B, T>(h, 'N', 'N', count, 2, 2, T(1), w0 + first, int(w1 - w0),
                                      inverse, 2, T(0), l0 + first, int(l1 - l0));
            l0[current] = T(d0);
            l0[neighbor] = w0[neighbor];
            l1[current] = w1[current];
            l1[neighbor] = T(d1);
        }
    }
    else
        pivot_columns(h, first, count, current, neighbor, step, swap_a, swap_b, c0, c1, l0, l1, w0,
                      w1);
}
template <DdlaBackend B, class T>
void stored_update(const DdlaHandle_t& h, char u, int n, const int* d, const T* src, int lds,
                   T* dst, int ldd, int row0 = 0, int col0 = 0, int rows = -1, int cols = -1,
                   bool subtract = false)
{
    if constexpr (B == DdlaBackend::CPU)
    {
        if (rows < 0)
            rows = num_loc(n, d[DDLA_MB_], h->myprow_, d[DDLA_RSRC_], h->nprows_);
        if (cols < 0)
            cols = num_loc(n, d[DDLA_NB_], h->mypcol_, d[DDLA_CSRC_], h->npcols_);
        for (int j = 0; j < cols; ++j)
        {
            const int gj = indx_l2g_c(d, h, col0 + j);
            if (gj >= n)
                continue;
            const int begin = u == 'L' ? std::max(row0, num_loc(gj, d[DDLA_MB_], h->myprow_,
                                                                d[DDLA_RSRC_], h->nprows_))
                                       : row0;
            const int end = u == 'U'
                                ? std::min(row0 + rows, num_loc(gj + 1, d[DDLA_MB_], h->myprow_,
                                                                d[DDLA_RSRC_], h->nprows_))
                                : row0 + rows;
            if (end <= begin)
                continue;
            const T* source = src + begin - row0 + std::size_t(j) * lds;
            T* target = dst + begin + std::size_t(col0 + j) * ldd;
            if (subtract)
                blas_impl::axpy_run<B, T>(h, end - begin, T(-1), source, 1, target, 1);
            else
                DDLA_CHECK(ddlaMemcpy(h, target, source, std::size_t(end - begin) * sizeof(T)));
            const int diagonal = indx_g2l_r(d, h, gj);
            if (diagonal >= begin && diagonal < end)
                dst[diagonal + std::size_t(col0 + j) * ldd] =
                    T(std::real(dst[diagonal + std::size_t(col0 + j) * ldd]));
        }
    }
    else
        copy_stored(h, u, n, d, src, lds, dst, ldd, row0, col0, rows, cols, subtract);
}
// Selected-triangle symmetric interchange, using two replicated raw columns.
// No BLAS permutation supports the required conjugating row/column exchange.
template <DdlaBackend B, class T>
void symmetric_interchange(const DdlaHandle_t& h, char u, int n, const int* d, T* a, int ld,
                           int first, int count, int row_a, int row_b, const T* col_a,
                           const T* col_b)
{
    if constexpr (B == DdlaBackend::CPU)
    {
        for (int p = 0; p < 2; ++p)
            for (int i = first; i < first + count; ++i)
            {
                if (p && i == row_a)
                    continue;
                const int column = p ? row_b : row_a;
                const int old = i == row_a ? row_b : i == row_b ? row_a : i;
                T value = p ? col_a[old] : col_b[old];
                int r = i, c = column;
                if (u == 'L' ? r < c : r > c)
                {
                    std::swap(r, c);
                    value = detail::host_conj(value);
                }
                const int lr = indx_g2l_r(d, h, r), lc = indx_g2l_c(d, h, c);
                if (lr >= 0 && lc >= 0)
                    a[lr + std::size_t(lc) * ld] = r == c ? T(std::real(value)) : value;
            }
    }
    else
        swap_stored(h, u, n, d, a, ld, first, count, row_a, row_b, col_a, col_b);
}
} // namespace ddla::ldlt_impl
#endif
