#include "BLAS/gemmt_impl.h"
#include "BLAS/hemmt_impl.h"
#include "PBLAS/pgeadd_impl.h"
#include "PBLAS/pswap_impl.h"
#include "comm_traits.h"
#include "hetrf_native.h"
#include "ldlt_distributed.h"
#include "matrix_transfer.h"
#include "plan_util.h"
#include "instantiate.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <optional>

namespace ddla::ldlt_impl
{
namespace
{
constexpr int panel_width = 64;

// Gathered pivoting uses an extra matrix with the input's 2D ownership.
// In-place pivoting and no-pivot update the caller's triangle. L/W contain at
// most 64 replicated vectors; all remaining scratch is bounded by a panel.
template <DdlaBackend B, bool H, bool Pivot, class T> class DistributedFactor
{
    const DdlaHandle_t& h;
    const bool gathered;
    const char uplo;
    const int n, rows, cols, ld, capacity;
    int cache_begin = 0, cache_width = 0;
    int df[DDLA_DLEN_];
    vector<T> f, l, w, c0, c1, coeff, lr, wc, diagonal, product, cached, external, mirror, swap_a,
        swap_b;
    T* matrix;
    vector<int> status;
    vector<PivotSummary> summary;
    vector<char> transfer;
    std::vector<char> transfer_host;

  public:
    DistributedFactor(const DdlaHandle_t& handle, int order, T* input, const int* da, bool gather,
                      char u)
        : h(handle), gathered(gather), uplo(u), n(order),
          rows(num_loc(n, da[DDLA_MB_], h->myprow_, da[DDLA_RSRC_], h->nprows_)),
          cols(num_loc(n, da[DDLA_NB_], h->mypcol_, da[DDLA_CSRC_], h->npcols_)),
          ld(Pivot && gathered ? std::max(1, rows) : da[DDLA_LLD_]),
          capacity(Pivot ? std::min(n, panel_width)
                         : std::min({n, panel_width, da[DDLA_MB_], da[DDLA_NB_]})),
          f(h, Pivot && gathered ? std::size_t(ld) * cols : 0), l(h), w(h), c0(h), c1(h), coeff(h),
          lr(h), wc(h), diagonal(h), product(h), cached(h), external(h), mirror(h), swap_a(h),
          swap_b(h), matrix(Pivot && gathered ? f.data() : input), status(h), summary(h),
          transfer(h)
    {
        DDLA_CHECK(ddlaDescInit(df, h, n, n, da[DDLA_MB_], da[DDLA_NB_], da[DDLA_RSRC_],
                                da[DDLA_CSRC_], ld));
    }
    void prepare()
    {
        l.resize(std::size_t(n) * capacity);
        w.resize(std::size_t(n) * capacity);
        lr.resize(std::size_t(std::max(1, rows)) * capacity);
        wc.resize(std::size_t(std::max(1, cols)) * capacity);
        if constexpr (Pivot)
        {
            c0.resize(n);
            c1.resize(n);
            if (!h->myid)
            {
                coeff.resize(capacity);
                summary.resize(1);
            }
            std::size_t swap_bytes = 0;
            pswap_impl::pswap_workspace_size<T>(h, df, &swap_bytes);
            transfer.resize(std::max(
                swap_bytes, detail::padded_bytes<T>(std::size_t(n) * (gathered ? capacity : 1))));
            if (gathered)
            {
                if (!h->myid)
                {
                    cached.resize(std::size_t(n) * capacity);
                    external.resize(n);
                }
            }
            else
            {
                mirror.resize(n);
                swap_a.resize(n);
                swap_b.resize(n);
            }
        }
        else
        {
            diagonal.resize(capacity * capacity);
            DDLA_CHECK(ddlaMemset(h, diagonal.data(), 0, diagonal.size() * sizeof(T)));
            transfer.resize(detail::padded_bytes<T>(std::size_t(n) * capacity));
            status.resize(1);
        }
        if (!Pivot || gathered)
            transfer_host.resize(std::size_t(std::max(0, h->nprocs - 1)) *
                                 sizeof(matrix_impl::Block<T>));
        if (!Pivot || !gathered)
            product.resize(std::size_t(rows) * std::min(capacity, cols));
    }
    void initialize(char u, const T* a, const int* da)
    {
        if constexpr (!Pivot)
            return;
        if (gathered)
        {
            copy_factor<B, H, T>(h, u, n, a, da, matrix, df, true);
            return;
        }
        // Hermitian interchanges must never propagate ignored imaginary diagonals.
        if constexpr (H)
        {
            for (int j = 0; j < n;)
            {
                const int width = std::min({capacity, n - j, df[DDLA_MB_] - j % df[DDLA_MB_],
                                            df[DDLA_NB_] - j % df[DDLA_NB_]});
                const int row = indx_g2l_r(df, h, j), col = indx_g2l_c(df, h, j);
                if (row >= 0 && col >= 0)
                    blas_impl::real_diagonal<B, T>(h, width, matrix + row + std::size_t(col) * ld, ld);
                j += width;
            }
        }
    }
    void finish(char u, T* a, const int* da)
    {
        if constexpr (Pivot)
            if (gathered)
                stored_update<B, H, T>(h, u, n, da, matrix, ld, a, da[DDLA_LLD_]);
        DDLA_CHECK(ddlaSynchronize(h));
    }

  private:
    void gather(int m, int k, int i, int j, T* dst, int ldd, int root)
    {
        matrix_impl::gather_run<B, T>(h, m, k, matrix, i + 1, j + 1, df, root, dst, ldd,
                                      transfer.data(), transfer.size(), transfer_host.data(),
                                      transfer_host.size());
    }
    // The factors are already replicated panel vectors. Store directly on
    // each owner instead of sending them back through a root scatter.
    void local_panel(bool scatter, int first, int count, int j, int width, T* global)
    {
        local_rectangle<B, T>(h, scatter, matrix, df, first, j, count, width, global + first, n);
    }
    void prefetch(int done)
    {
        cache_width = std::min(capacity, n - done);
        cache_begin = uplo == 'L' ? done : n - done - cache_width;
        const int first = uplo == 'L' ? done : 0;
        gather(n - done, cache_width, first, cache_begin, h->myid ? nullptr : cached.data() + first,
               n, 0);
    }
    bool in_cache(int j) const
    {
        return gathered && j >= cache_begin && j < cache_begin + cache_width;
    }
    void read_column(int j, int first, int count, T* target)
    {
        if (in_cache(j))
        {
            if (!h->myid)
                DDLA_CHECK(ddlaMemcpy(h, target + first,
                                      cached.data() + first + std::size_t(j - cache_begin) * n,
                                      std::size_t(count) * sizeof(T)));
            return;
        }
        T* raw = reinterpret_cast<T*>(transfer.data());
        DDLA_CHECK(ddlaMemset(h, raw + first, 0, std::size_t(count) * sizeof(T)));
        if (gathered)
            local_panel(false, first, count, j, 1, raw);
        else
        {
            const int begin = uplo == 'L' ? j : first,
                      length = uplo == 'L' ? first + count - j : j - first + 1;
            local_rectangle<B, T>(h, false, matrix, df, begin, j, length, 1, raw + begin, n);
            const int row_first = uplo == 'L' ? first : j + 1,
                      row_count = uplo == 'L' ? j - first : first + count - j - 1;
            if (row_count)
            {
                DDLA_CHECK(ddlaMemset(h, mirror.data() + row_first, 0,
                                      std::size_t(row_count) * sizeof(T)));
                local_rectangle<B, T>(h, false, matrix, df, j, row_first, 1, row_count,
                                      mirror.data() + row_first, 1);
                blas_impl::geam_run<B, T>(h, H ? 'C' : 'T', H ? 'C' : 'T', row_count, 1, T(1),
                                          mirror.data() + row_first, 1, T(0),
                                          mirror.data() + row_first, 1, raw + row_first, row_count);
            }
        }
        commReduceSum<B>(h, CommScope::Grid, raw + first, target + first, std::size_t(count), 0);
        if (gathered && !h->myid)
            DDLA_CHECK(ddlaMemcpy(h, external.data() + first, target + first,
                                  std::size_t(count) * sizeof(T)));
    }
    // Only the root computes a pivot record. Carry its error flag and pivot
    // decision in the same broadcast; an all-rank reduction is unnecessary.
    template <class Decide>
    std::array<int, 4> column(int j, int first, int count, int q, T* target, PivotSummary& record,
                              Decide decide)
    {
        read_column(j, first, count, target);
        std::array<int, 4> control{};
        std::exception_ptr failure;
        if (!h->myid)
        {
            try
            {
                if (q)
                {
                    blas_impl::geam_run<B, T>(h, H ? 'C' : 'T', H ? 'C' : 'T', q, 1, T(1), w.data() + j, n, T(0),
                                              w.data() + j, n, coeff.data(), q);
                    blas_impl::gemv_run<B, T>(h, 'N', count, q, T(-1), l.data() + first, n,
                                              coeff.data(), 1, T(1), target + first, 1);
                }
                column_record<B, H, T>(h, target, first, count, j, summary.data());
                DDLA_CHECK(ddlaMemcpy(h, &record, summary.data(), sizeof(record)));
                DDLA_CHECK(ddlaSynchronize(h));
                decide(record, control);
            }
            catch (...)
            {
                failure = std::current_exception();
                control[0] = 1;
            }
        }
        MPI_CHECK(MPI_Bcast(control.data(), int(control.size()), MPI_INT, 0, h->comm));
        if (control[0])
        {
            if (failure)
                std::rethrow_exception(failure);
            throw std::runtime_error("phetrf: pivot evaluation failed on the root");
        }
        return control;
    }
    void swap_active(int first, int count, int a, int b, int q)
    {
        if (a == b)
            return;
        if (gathered)
        {
            pswap_impl::pswap_run<B, T>(h, count, matrix, a + 1, first + 1, df, n, matrix, b + 1,
                                        first + 1, df, n, transfer.data(), transfer.size());
            pswap_impl::pswap_run<B, T>(h, count, matrix, first + 1, a + 1, df, 1, matrix,
                                        first + 1, b + 1, df, 1, transfer.data(), transfer.size());
            if (!h->myid && cache_width)
            {
                blas_impl::swap_run<B, T>(h, cache_width, cached.data() + a, n, cached.data() + b,
                                          n);
                const bool ca = in_cache(a), cb = in_cache(b);
                if (ca && cb)
                    blas_impl::swap_run<B, T>(
                        h, count, cached.data() + first + std::size_t(a - cache_begin) * n, 1,
                        cached.data() + first + std::size_t(b - cache_begin) * n, 1);
                else if (ca || cb)
                {
                    T* column = cached.data() + std::size_t((ca ? a : b) - cache_begin) * n;
                    DDLA_CHECK(ddlaMemcpy(h, column + first, external.data() + first,
                                          std::size_t(count) * sizeof(T)));
                    blas_impl::swap_run<B, T>(h, 1, column + a, 1, column + b, 1);
                }
            }
        }
        else
        {
            const int lo = std::min(a, b), hi = std::max(a, b);
            read_column(lo, first, count, swap_a.data());
            read_column(hi, first, count, swap_b.data());
            commBcast<B>(h, CommScope::Grid, swap_a.data() + first, std::size_t(count), 0);
            commBcast<B>(h, CommScope::Grid, swap_b.data() + first, std::size_t(count), 0);
            symmetric_interchange<B, H, T>(h, uplo, n, df, matrix, ld, first, count, lo, hi,
                                        swap_a.data(), swap_b.data());
        }
        if (q)
        {
            blas_impl::swap_run<B, T>(h, q, l.data() + a, n, l.data() + b, n);
            blas_impl::swap_run<B, T>(h, q, w.data() + a, n, w.data() + b, n);
        }
    }
    void trailing(char uplo, int first, int count, int q)
    {
        if (!count || !q)
            return;
        const int r0 = num_loc(first, df[DDLA_MB_], h->myprow_, df[DDLA_RSRC_], h->nprows_);
        const int r1 = num_loc(first + count, df[DDLA_MB_], h->myprow_, df[DDLA_RSRC_], h->nprows_);
        const int cstart = num_loc(first, df[DDLA_NB_], h->mypcol_, df[DDLA_CSRC_], h->npcols_);
        const int cend =
            num_loc(first + count, df[DDLA_NB_], h->mypcol_, df[DDLA_CSRC_], h->npcols_);
        for (int g = first; g < first + count;)
        {
            const int length = std::min(first + count - g, df[DDLA_MB_] - g % df[DDLA_MB_]);
            if (indxg2p(g, df[DDLA_MB_], df[DDLA_RSRC_], h->nprows_) == h->myprow_)
            {
                const int local = indxg2l(g, df[DDLA_MB_], h->nprows_);
                DDLA_CHECK(ddlaMemcpy2D(
                    h, lr.data() + local, std::size_t(std::max(1, rows)) * sizeof(T), l.data() + g,
                    std::size_t(n) * sizeof(T), std::size_t(length) * sizeof(T), q));
            }
            g += length;
        }
        const int ldw = std::max(1, cols);
        for (int g = first; g < first + count;)
        {
            const int length = std::min(first + count - g, df[DDLA_NB_] - g % df[DDLA_NB_]);
            if (indxg2p(g, df[DDLA_NB_], df[DDLA_CSRC_], h->npcols_) == h->mypcol_)
            {
                const int local = indxg2l(g, df[DDLA_NB_], h->npcols_);
                DDLA_CHECK(ddlaMemcpy2D(h, wc.data() + local, std::size_t(ldw) * sizeof(T),
                                        w.data() + g, std::size_t(n) * sizeof(T),
                                        std::size_t(length) * sizeof(T), q));
            }
            g += length;
        }
        if (r1 > r0 && cend > cstart)
        {
            const int ldp = std::max(1, rows);
            if (Pivot && gathered)
                blas_impl::gemm_run<B, T>(h, 'N', H ? 'C' : 'T', r1 - r0, cend - cstart, q, T(-1),
                                          lr.data() + r0, ldp, wc.data() + cstart, ldw, T(1),
                                          matrix + r0 + std::size_t(cstart) * ld, ld);
            else
                for (int col = cstart; col < cend; col += capacity)
                {
                    const int width = std::min(capacity, cend - col);
                    blas_impl::gemm_run<B, T>(h, 'N', H ? 'C' : 'T', r1 - r0, width, q, T(1), lr.data() + r0,
                                              ldp, wc.data() + col, ldw, T(0), product.data(), ldp);
                    stored_update<B, H, T>(h, uplo, n, df, product.data(), ldp, matrix, ld, r0, col,
                                        r1 - r0, width, true);
                }
        }
    }

  public:
    void pivoted(char u, int* piv, int& info)
    {
        const double alpha = (1 + std::sqrt(17.0)) / 8;
        for (int done = 0; done < n;)
        {
            if (gathered)
                prefetch(done);
            int q = 0;
            while (done + q < n && q < capacity)
            {
                const int count = n - done - q;
                if (q == capacity - 1 && count > 1)
                    break;
                const int current = u == 'L' ? done + q : n - done - q - 1;
                const int first = u == 'L' ? current : 0;
                PivotSummary first_record{}, second_record{};
                const auto request =
                    column(current, first, count, q, c0.data(), first_record,
                           [&](const PivotSummary& record, std::array<int, 4>& control)
                           {
                               const bool singular = std::max(record.diagonal, record.maximum) == 0;
                               control[1] = !singular && record.diagonal < alpha * record.maximum
                                                ? record.index
                                                : -1;
                               control[2] = singular;
                           });
                const int imax = request[1];
                int decision[3] = {1, current, request[2]};
                if (imax >= 0)
                {
                    const auto selected = column(
                        imax, first, count, q, c1.data(), second_record,
                        [&](const PivotSummary& record, std::array<int, 4>& control)
                        {
                            control[1] = 1;
                            control[2] = current;
                            if (first_record.diagonal < alpha * first_record.maximum *
                                                            (first_record.maximum / record.maximum))
                            {
                                control[2] = imax;
                                if (record.diagonal < alpha * record.maximum)
                                    control[1] = 2;
                            }
                        });
                    decision[0] = selected[1];
                    decision[1] = selected[2];
                }
                const int step = decision[0], kp = decision[1];
                const int neighbor = current + (u == 'L' ? 1 : -1);
                const int interchange = step == 1 ? current : neighbor;
                if (decision[2] && !info)
                    info = current + 1;
                T* first_column = step == 1 && kp != current ? c1.data() : c0.data();
                commBcast<B>(h, CommScope::Grid, first_column + first, std::size_t(count), 0);
                if (step == 2)
                    commBcast<B>(h, CommScope::Grid, c1.data() + first, std::size_t(count), 0);
                swap_active(first, count, interchange, kp, q);
                T* l0 = l.data() + std::size_t(q) * n;
                T* l1 = step == 2 ? l0 + n : nullptr;
                T* w0 = w.data() + std::size_t(q) * n;
                T* w1 = step == 2 ? w0 + n : nullptr;
                normalize_columns<B, H, T>(h, first, count, current, neighbor, step, interchange, kp,
                                        first_column, c1.data(), l0, l1, w0, w1);
                local_panel(true, first, count, current, 1, l0);
                piv[current] = step == 1 ? kp + 1 : -(kp + 1);
                if (step == 2)
                {
                    const int next_first = u == 'L' ? neighbor : 0;
                    local_panel(true, next_first, count - 1, neighbor, 1, l1);
                    piv[neighbor] = -(kp + 1);
                }
                q += step;
            }
            trailing(u, u == 'L' ? done + q : 0, n - done - q, q);
            done += q;
        }
    }
    void unpivoted(char u, int& info)
    {
        for (int done = 0; done < n;)
        {
            const int end = n - done;
            const int b = u == 'L' ? std::min({capacity, end, df[DDLA_MB_] - done % df[DDLA_MB_],
                                               df[DDLA_NB_] - done % df[DDLA_NB_]})
                                   : std::min({capacity, end, (end - 1) % df[DDLA_MB_] + 1,
                                               (end - 1) % df[DDLA_NB_] + 1});
            const int start = u == 'L' ? done : end - b;
            const int owner =
                h->rc_to_rank(indxg2p(start, df[DDLA_MB_], df[DDLA_RSRC_], h->nprows_),
                              indxg2p(start, df[DDLA_NB_], df[DDLA_CSRC_], h->npcols_));
            const int row = indx_g2l_r(df, h, start), col = indx_g2l_c(df, h, start);
            int failure = 0;
            // Every diagonal intersection is local to its owner. Preserve the
            // original block until we know the successful prefix/suffix.
            detail::agree_check(
                h, "phetrf_nopiv diagonal",
                [&](detail::Agree&)
                {
                    if (h->myid != owner)
                        return;
                    blas_impl::diagonal_block_run<B, T>(h, u == 'U', H, b,
                                                        matrix + row + std::size_t(col) * ld, ld,
                                                        diagonal.data(), capacity);
                    diagonal_factor<B, H, T>(h, u, b, diagonal.data(), capacity, status.data());
                    DDLA_CHECK(ddlaMemcpy(h, &failure, status.data(), sizeof(failure)));
                    DDLA_CHECK(ddlaSynchronize(h));
                });
            MPI_CHECK(MPI_Bcast(&failure, 1, MPI_INT, owner, h->comm));
            const int q = !failure ? b : u == 'L' ? failure - 1 : b - failure;
            const int inside = u == 'L' ? 0 : b - q, pivot_start = start + inside;
            T* d = q ? diagonal.data() + inside + std::size_t(inside) * capacity : diagonal.data();
            if (q)
            {
                if (h->myid == owner)
                    stored_update<B, H, T>(h, u, n, df, d, capacity, matrix, ld, row + inside,
                                        col + inside, q, q);
                const int first = u == 'L' ? pivot_start + q : 0;
                const int count = u == 'L' ? n - first : pivot_start;
                if (count)
                {
                    gather(count, q, first, pivot_start, w.data() + first, n, owner);
                    detail::agree_check(h, "phetrf_nopiv panel",
                                        [&](detail::Agree&)
                                        {
                                            if (h->myid == owner)
                                                blas_impl::trsm_run<B, T>(
                                                    h, 'R', u, H ? 'C' : 'T', 'U', count, q, T(1), d,
                                                    capacity, w.data() + first, n);
                                        });
                    // One packet contains the weighted panel and D's diagonal.
                    // count*q+q <= n*capacity, so the existing transfer pool fits.
                    T* packet = reinterpret_cast<T*>(transfer.data());
                    if (h->myid == owner)
                    {
                        DDLA_CHECK(ddlaMemcpy2D(h, packet, std::size_t(count) * sizeof(T),
                                                w.data() + first, std::size_t(n) * sizeof(T),
                                                std::size_t(count) * sizeof(T), q));
                        DDLA_CHECK(ddlaMemcpy2D(h, packet + std::size_t(count) * q, sizeof(T), d,
                                                std::size_t(capacity + 1) * sizeof(T), sizeof(T),
                                                q));
                    }
                    commBcast<B>(h, CommScope::Grid, packet, std::size_t(count + 1) * q, owner);
                    if (h->myid != owner)
                    {
                        DDLA_CHECK(ddlaMemcpy2D(h, w.data() + first, std::size_t(n) * sizeof(T),
                                                packet, std::size_t(count) * sizeof(T),
                                                std::size_t(count) * sizeof(T), q));
                        DDLA_CHECK(ddlaMemcpy2D(h, d, std::size_t(capacity + 1) * sizeof(T),
                                                packet + std::size_t(count) * q, sizeof(T),
                                                sizeof(T), q));
                    }
                    diagonal_divide<B, H, T>(h, count, q, d, capacity, w.data() + first, n,
                                          l.data() + first, n);
                    local_panel(true, first, count, pivot_start, q, l.data());
                    trailing(u, first, count, q);
                }
            }
            if (failure)
            {
                const int failed = start + failure - 1;
                if (h->myid == owner)
                    DDLA_CHECK(ddlaMemcpy(
                        h, matrix + row + failure - 1 + std::size_t(col + failure - 1) * ld,
                        diagonal.data() + failure - 1 + std::size_t(failure - 1) * capacity,
                        sizeof(T)));
                info = failed + 1;
                return;
            }
            done += q;
        }
    }
};
} // namespace

template <DdlaBackend B, bool H, bool Pivot, class T>
void distributed_factor(const DdlaHandle_t& h, char u, int n, T* a, const int* da, int* piv,
                        int& info)
{
    u = distributed_check<B, H, Pivot, false, T>(h, u, n, 0, a, da, piv, nullptr, nullptr, info);
    bool gathered = true;
    if constexpr (Pivot)
        detail::agree_check(h, "phetrf panel method",
                            [&](detail::Agree& agree)
                            {
                                const char* value = std::getenv("DDLA_LDLT_PANEL");
                                if (value && *value)
                                {
                                    if (!std::strcmp(value, "inplace"))
                                        gathered = false;
                                    else if (std::strcmp(value, "gather"))
                                        throw std::invalid_argument(
                                            "DDLA_LDLT_PANEL must be gather or inplace");
                                }
                                agree.mode("DDLA_LDLT_PANEL", gathered ? 1 : 0);
                            });
    if (!n)
        return;
    const bool local = h->nprocs == 1;
    const bool owned = n <= da[DDLA_MB_] && n <= da[DDLA_NB_];
    if constexpr (B == DdlaBackend::CPU && Pivot)
    {
        if (local || owned)
        {
            const int root = h->rc_to_rank(da[DDLA_RSRC_], da[DDLA_CSRC_]);
            detail::agree_check(h, "phetrf local owner",
                                [&](detail::Agree&)
                                {
                                    if (h->myid == root)
                                        factor(std::bool_constant<H>{}, u, n, a, da[DDLA_LLD_], piv, info);
                                });
            MPI_CHECK(MPI_Bcast(&info, 1, MPI_INT, root, h->comm));
            MPI_CHECK(MPI_Bcast(piv, n, MPI_INT, root, h->comm));
            return;
        }
    }
#if DDLA_HAS_GPU
    if constexpr (B == DdlaBackend::GPU && H)
    {
        if constexpr (!Pivot || std::is_same_v<T, std::complex<double>>)
            if (local || owned)
            {
                const int root = h->rc_to_rank(da[DDLA_RSRC_], da[DDLA_CSRC_]);
                detail::agree_check(h, "phetrf local owner",
                                    [&](detail::Agree&)
                                    {
                                        if (h->myid != root)
                                            return;
                                        if constexpr (Pivot)
                                            hetrf<B, T>(h, u, n, a, da[DDLA_LLD_], piv, info);
                                        else
                                            hetrf_nopiv_gpu(h, u, n, a, da[DDLA_LLD_], info);
                                    });
                MPI_CHECK(MPI_Bcast(&info, 1, MPI_INT, root, h->comm));
                if constexpr (Pivot)
                    MPI_CHECK(MPI_Bcast(piv, n, MPI_INT, root, h->comm));
                return;
            }
    }
#endif
    std::optional<DistributedFactor<B, H, Pivot, T>> f;
    detail::agree_check(h, "phetrf distributed storage",
                        [&](detail::Agree&) { f.emplace(h, n, a, da, gathered, u); });
    f->initialize(u, a, da);
    detail::agree_check(h, "phetrf panel storage", [&](detail::Agree&) { f->prepare(); });
    if constexpr (Pivot)
        f->pivoted(u, piv, info);
    else
        f->unpivoted(u, info);
    f->finish(u, a, da);
}
#define ENGINE(B, T) \
    template void distributed_factor<B, false, true, T>(const DdlaHandle_t&, char, int, T*, const int*, int*, int&); \
    template void distributed_factor<B, false, false, T>(const DdlaHandle_t&, char, int, T*, const int*, int*, int&)
DDLA_FOR_EACH_BACKEND_TYPE(ENGINE)
#undef ENGINE
} // namespace ddla::ldlt_impl

namespace ddla
{
template <DdlaBackend B, class T>
void phetrf(const DdlaHandle_t& h, char u, int n, T* a, const int* da, int* p, int& info)
{
    ldlt_impl::distributed_factor<B, true, true, T>(h, u, n, a, da, p, info);
}
template <DdlaBackend B, class T>
void phetrf_nopiv(const DdlaHandle_t& h, char u, int n, T* a, const int* da, int& info)
{
    ldlt_impl::distributed_factor<B, true, false, T>(h, u, n, a, da, nullptr, info);
}

#define INSTANCE(B, T)                                                                             \
    template void phetrf<B, T>(const DdlaHandle_t&, char, int, T*, const int*, int*, int&);        \
    template void phetrf_nopiv<B, T>(const DdlaHandle_t&, char, int, T*, const int*, int&)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANCE)
#undef INSTANCE
} // namespace ddla
