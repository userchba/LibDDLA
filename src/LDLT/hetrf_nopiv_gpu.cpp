#include "BLAS/gemmt_impl.h"
#include "BLAS/hemmt_impl.h"
#include "hetrf_native.h"
#include "ldlt_impl.h"
#include <algorithm>

namespace ddla::ldlt_impl
{
namespace
{
constexpr auto gpu = DdlaBackend::GPU;
constexpr int panel_width = 64;

// Updates only selected diagonal tiles and their selected rectangular strips.
template <class T>
void update_triangle(const DdlaHandle_t& h, char u, int n, int k, const T* l, const T* w, int ldp,
                     T* a, int lda, void* scratch, std::size_t bytes)
{
    for (int j = 0; j < n; j += panel_width)
    {
        const int jb = std::min(panel_width, n - j);
        T* diagonal = a + j + std::size_t(j) * lda;
        blas_impl::real_diagonal<gpu, T>(h, jb, diagonal, lda);
        blas_impl::gemmt_run<gpu, T>(h, u, 'N', 'C', jb, k, T(-1), l + j, ldp, w + j, ldp, T(1),
                                     diagonal, lda, scratch, bytes);
        blas_impl::real_diagonal<gpu, T>(h, jb, diagonal, lda);
        const int first = u == 'U' ? 0 : j + jb, rows = u == 'U' ? j : n - j - jb;
        if (rows)
            blas_impl::gemm_run<gpu, T>(h, 'N', 'C', rows, jb, k, T(-1), l + first, ldp, w + j, ldp,
                                        T(1), a + first + std::size_t(j) * lda, lda);
    }
}
} // namespace

template <class T>
void hetrf_nopiv_gpu(const DdlaHandle_t& h, char u, int n, T* a, int lda, int& info)
{
    u = local_check<gpu, false, false, T>(h, u, n, 0, a, lda, nullptr, nullptr, 1, info);
    if (!n)
        return;
    if (n == 1)
    {
        blas_impl::real_diagonal<gpu, T>(h, 1, a, lda);
        T diagonal;
        DDLA_CHECK(ddlaMemcpy(h, &diagonal, a, sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
        info = std::real(diagonal) == 0 ? 1 : 0;
        return;
    }
    vector<int> status(h, 1);
    if (n <= 256)
    {
        nopiv_diagonal(h, u, n, a, lda, status.data());
        DDLA_CHECK(ddlaMemcpy(h, &info, status.data(), sizeof(info)));
        DDLA_CHECK(ddlaSynchronize(h));
        return;
    }
    vector<T> diagonal(h, panel_width * panel_width), l(h, std::size_t(n) * panel_width),
        w(h, std::size_t(n) * panel_width);
    std::size_t bytes = 0;
    for (int width = 1; width <= panel_width; ++width)
    {
        std::size_t current = 0;
        blas_impl::gemmt_workspace_size<T>(h, 'N', 'C', width, panel_width, &current);
        bytes = std::max(bytes, current);
    }
    vector<char> scratch(h, bytes);
    for (int done = 0; done < n;)
    {
        const int width = std::min(panel_width, n - done);
        const int start = u == 'U' ? n - done - width : done;
        T* source = a + start + std::size_t(start) * lda;
        blas_impl::diagonal_block_run<gpu, T>(h, u == 'U', true, width, source, lda,
                                              diagonal.data(), panel_width);
        nopiv_diagonal(h, u, width, diagonal.data(), panel_width, status.data());
        int failure = 0;
        DDLA_CHECK(ddlaMemcpy(h, &failure, status.data(), sizeof(failure)));
        DDLA_CHECK(ddlaSynchronize(h));
        const int q = !failure ? width : u == 'U' ? width - failure : failure - 1;
        const int inside = u == 'U' ? width - q : 0;
        const int pivot_start = start + inside;
        T* d = q ? diagonal.data() + inside + std::size_t(inside) * panel_width : diagonal.data();
        // Only copy successful factors. A failed diagonal block's remaining
        // original entries must receive the prefix update exactly once.
        for (int j = 0; j < q; ++j)
        {
            const int first = u == 'U' ? 0 : j, count = u == 'U' ? j + 1 : q - j;
            DDLA_CHECK(ddlaMemcpy(h, a + pivot_start + first + std::size_t(pivot_start + j) * lda,
                                  d + first + std::size_t(j) * panel_width,
                                  std::size_t(count) * sizeof(T)));
        }
        if (q)
        {
            const int first = u == 'U' ? 0 : pivot_start + q;
            const int rows = u == 'U' ? pivot_start : n - first;
            if (rows)
            {
                DDLA_CHECK(ddlaMemcpy2D(h, w.data(), std::size_t(n) * sizeof(T),
                                        a + first + std::size_t(pivot_start) * lda,
                                        std::size_t(lda) * sizeof(T), std::size_t(rows) * sizeof(T),
                                        q));
                blas_impl::trsm_run<gpu, T>(h, 'R', u, 'C', 'U', rows, q, T(1), d, panel_width,
                                            w.data(), n);
                divide_panel(h, rows, q, d, panel_width, w.data(), n, l.data(), n);
                DDLA_CHECK(ddlaMemcpy2D(
                    h, a + first + std::size_t(pivot_start) * lda, std::size_t(lda) * sizeof(T),
                    l.data(), std::size_t(n) * sizeof(T), std::size_t(rows) * sizeof(T), q));
                update_triangle(h, u, rows, q, l.data(), w.data(), n,
                                a + first + std::size_t(first) * lda, lda, scratch.data(), bytes);
            }
        }
        if (failure)
        {
            const int failed = u == 'U' ? pivot_start - 1 : pivot_start + q;
            DDLA_CHECK(ddlaMemcpy(
                h, a + failed + std::size_t(failed) * lda,
                diagonal.data() + failure - 1 + std::size_t(failure - 1) * panel_width, sizeof(T)));
            info = start + failure;
            break;
        }
        done += q;
    }
    DDLA_CHECK(ddlaSynchronize(h));
}
#define INSTANCE(T) template void hetrf_nopiv_gpu<T>(const DdlaHandle_t&, char, int, T*, int, int&);
INSTANCE(float)
INSTANCE(double)
INSTANCE(std::complex<float>)
INSTANCE(std::complex<double>)
#undef INSTANCE
} // namespace ddla::ldlt_impl
