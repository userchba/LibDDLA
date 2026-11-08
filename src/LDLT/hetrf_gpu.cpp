// Adapted from MAGMA src/zlahef_gpu.cpp and src/zhetrf_gpu.cpp.
// Upstream: icl-utk-edu/magma at 64a5264283984a360461cf89faed945b7fdca719.
// Authors: Ichi Yamazaki and Stan Tomov.
// Copyright (c) 2009-2023, The University of Tennessee.
// BSD-3-Clause terms and disclaimer are retained in MAGMA-LICENSE.
// Changes: DDLA backend operations, selected-triangle updates, 1-based INFO,
// standard Bunch-Kaufman alpha, ordered scalar downloads and a GPU final tail.
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "hetrf_gpu.h"
#include "ldlt_impl.h"
#include "BLAS/gemmt_impl.h"
#include <algorithm>
#include <cmath>

namespace ddla::ldlt_impl
{
namespace
{
using Z = std::complex<double>;
constexpr DdlaBackend gpu = DdlaBackend::GPU;
constexpr std::size_t zbytes = sizeof(Z);

// Pivot selection uses host scalars, never a host matrix or CPU panel factor.
Z read_pivot(const DdlaHandle_t& h, const Z* source)
{
    Z value;
    DDLA_CHECK(ddlaMemcpy(h, &value, source, sizeof(value)));
    DDLA_CHECK(ddlaSynchronize(h));
    return value;
}

double abs1(const Z& value)
{
    return std::abs(value.real()) + std::abs(value.imag());
}

void update_panel(const DdlaHandle_t& h, char uplo, int n, int nb, int kb, Z* a, int lda, Z* w,
                  int ldw, void* work, std::size_t bytes)
{
    const Z one(1), minus_one(-1);
    const auto A = [&](int i, int j) { return a + i + std::size_t(j) * lda; };
    const auto W = [&](int i, int j) { return w + i + std::size_t(j) * ldw; };
    if (uplo == 'U')
    {
        const int k = n - kb - 1;
        const int rank = kb;
        const int kw = nb - (n - k);
        for (int j = k >= 0 ? (k / nb) * nb : -1; j >= 0; j -= nb)
        {
            const int jb = std::min(nb, k - j + 1);
            blas_impl::real_diagonal<gpu, Z>(h, jb, A(j, j), lda);
            blas_impl::gemmt_run<gpu, Z>(h, 'U', 'N', 'T', jb, rank, minus_one, A(j, k + 1), lda,
                                         W(j, kw + 1), ldw, one, A(j, j), lda, work, bytes);
            blas_impl::real_diagonal<gpu, Z>(h, jb, A(j, j), lda);
            if (j > 0)
                blas_impl::gemm_run<gpu, Z>(h, 'N', 'T', j, jb, rank, minus_one, A(0, k + 1), lda,
                                            W(j, kw + 1), ldw, one, A(0, j), lda);
        }
    }
    else
    {
        const int k = kb;
        for (int j = k; j < n; j += nb)
        {
            const int jb = std::min(nb, n - j);
            blas_impl::real_diagonal<gpu, Z>(h, jb, A(j, j), lda);
            blas_impl::gemmt_run<gpu, Z>(h, 'L', 'N', 'T', jb, k, minus_one, A(j, 0), lda, W(j, 0),
                                         ldw, one, A(j, j), lda, work, bytes);
            blas_impl::real_diagonal<gpu, Z>(h, jb, A(j, j), lda);
            if (j + jb < n)
                blas_impl::gemm_run<gpu, Z>(h, 'N', 'T', n - j - jb, jb, k, minus_one, A(j + jb, 0),
                                            lda, W(j, 0), ldw, one, A(j + jb, j), lda);
        }
    }
}

// W stores conjugated completed columns, matching MAGMA's LAHEF convention.
// A partial panel leaves one free column for a possible 2x2 pivot. For the
// final panel n<=nb, factor every column here instead of calling CPU HETF2.
void panel(const DdlaHandle_t& h, char uplo, int n, int nb, int& kb, Z* a, int lda, int* ipiv, Z* w,
           int ldw, int& info, void* work, std::size_t bytes, int* device_piv, int* device_status,
           int offset)
{
    if (device_piv)
    {
        hetrf_panel_factor(h, uplo, n, nb, a, lda, device_piv, w, ldw, device_status, offset);
        int status[2];
        DDLA_CHECK(ddlaMemcpy(h, status, device_status, sizeof(status)));
        DDLA_CHECK(ddlaSynchronize(h));
        kb = status[0];
        info = status[1];
        update_panel(h, uplo, n, nb, kb, a, lda, w, ldw, work, bytes);
        if (n > nb)
            hetrf_panel_undo(h, uplo, n, kb, a, lda, device_piv, offset);
        return;
    }
    const Z one(1), minus_one(-1);
    const double alpha = (1.0 + std::sqrt(17.0)) / 8.0;
    const auto A = [&](int i, int j) { return a + i + std::size_t(j) * lda; };
    const auto W = [&](int i, int j) { return w + i + std::size_t(j) * ldw; };
    info = 0;

    if (uplo == 'U')
    {
        int k = n - 1, kw = 0;
        for (; k >= 0 && (n <= nb || k >= n - nb + 1);)
        {
            kw = nb - (n - k);
            DDLA_CHECK(ddlaMemcpy2D(h, W(0, kw), zbytes, A(0, k), zbytes, zbytes, k + 1));
            blas_impl::real_diagonal<gpu, Z>(h, 1, W(k, kw), 1);
            if (k + 1 < n)
            {
                blas_impl::gemv_run<gpu, Z>(h, 'N', k + 1, n - k - 1, minus_one, A(0, k + 1), lda,
                                            W(k, kw + 1), ldw, one, W(0, kw), 1);
                blas_impl::real_diagonal<gpu, Z>(h, 1, W(k, kw), 1);
            }

            int step = 1, kp = k, imax = 0;
            const double diagonal = std::abs(read_pivot(h, W(k, kw)).real());
            double colmax = 0;
            if (k > 0)
            {
                blas_impl::iamax_run<gpu, Z>(h, k, W(0, kw), 1, imax);
                --imax;
                colmax = abs1(read_pivot(h, W(imax, kw)));
            }
            if (std::max(diagonal, colmax) == 0)
            {
                if (!info)
                    info = k + 1;
                DDLA_CHECK(ddlaMemcpy2D(h, A(0, k), zbytes, W(0, kw), zbytes, zbytes, k + 1));
            }
            else
            {
                if (diagonal < alpha * colmax)
                {
                    DDLA_CHECK(ddlaMemcpy2D(h, W(0, kw - 1), zbytes, A(0, imax), zbytes, zbytes,
                                            imax + 1));
                    blas_impl::real_diagonal<gpu, Z>(h, 1, W(imax, kw - 1), 1);
                    hetrf_conj_copy(h, k - imax, A(imax, imax + 1), lda, W(imax + 1, kw - 1), 1);
                    if (k + 1 < n)
                    {
                        blas_impl::gemv_run<gpu, Z>(h, 'N', k + 1, n - k - 1, minus_one,
                                                    A(0, k + 1), lda, W(imax, kw + 1), ldw, one,
                                                    W(0, kw - 1), 1);
                        blas_impl::real_diagonal<gpu, Z>(h, 1, W(imax, kw - 1), 1);
                    }
                    const double imax_diagonal = std::abs(read_pivot(h, W(imax, kw - 1)).real());
                    int jmax = 0;
                    blas_impl::iamax_run<gpu, Z>(h, k - imax, W(imax + 1, kw - 1), 1, jmax);
                    jmax += imax;
                    double rowmax = abs1(read_pivot(h, W(jmax, kw - 1)));
                    if (imax > 0)
                    {
                        blas_impl::iamax_run<gpu, Z>(h, imax, W(0, kw - 1), 1, jmax);
                        --jmax;
                        rowmax = std::max(rowmax, abs1(read_pivot(h, W(jmax, kw - 1))));
                    }
                    if (diagonal >= alpha * colmax * (colmax / rowmax))
                    {
                        kp = k;
                    }
                    else if (imax_diagonal >= alpha * rowmax)
                    {
                        kp = imax;
                        DDLA_CHECK(
                            ddlaMemcpy2D(h, W(0, kw), zbytes, W(0, kw - 1), zbytes, zbytes, k + 1));
                    }
                    else
                    {
                        kp = imax;
                        step = 2;
                    }
                }
                const int kk = k - step + 1;
                const int kkw = nb - (n - kk);
                if (kp != kk)
                {
                    blas_impl::swap_run<gpu, Z>(h, n - kk, A(kk, kk), lda, A(kp, kk), lda);
                    blas_impl::swap_run<gpu, Z>(h, n - kk, W(kk, kkw), ldw, W(kp, kkw), ldw);
                    hetrf_conj_copy(h, kk - kp - 1, A(kp + 1, kk), 1, A(kp, kp + 1), lda);
                    DDLA_CHECK(ddlaMemcpy2D(h, A(0, kp), zbytes, A(0, kk), zbytes, zbytes, kp + 1));
                    blas_impl::real_diagonal<gpu, Z>(h, 1, A(kp, kp), 1);
                }
                if (step == 1)
                {
                    DDLA_CHECK(ddlaMemcpy2D(h, A(0, k), zbytes, W(0, kw), zbytes, zbytes, k + 1));
                    if (k > 0)
                    {
                        const double inverse = 1.0 / read_pivot(h, W(k, kw)).real();
                        blas_impl::scal_run<gpu, Z>(h, k, Z(inverse), A(0, k), 1);
                        hetrf_conj_copy(h, k, W(0, kw), 1, W(0, kw), 1);
                    }
                }
                else
                {
                    hetrf_scale_2x2(h, 'U', k - 1, W(0, kw - 1), ldw, A(0, k - 1), lda);
                    // Store only U's selected part of D; preserve the opposite triangle.
                    DDLA_CHECK(ddlaMemcpy(h, A(k - 1, k - 1), W(k - 1, kw - 1), zbytes));
                    DDLA_CHECK(ddlaMemcpy(h, A(k - 1, k), W(k - 1, kw), 2 * zbytes));
                    hetrf_conj_copy(h, k, W(0, kw), 1, W(0, kw), 1);
                    hetrf_conj_copy(h, k - 1, W(0, kw - 1), 1, W(0, kw - 1), 1);
                }
            }
            ipiv[k] = step == 1 ? kp + 1 : -(kp + 1);
            if (step == 2)
                ipiv[k - 1] = -(kp + 1);
            k -= step;
        }

        const int rank = n - k - 1;
        update_panel(h, uplo, n, nb, rank, a, lda, w, ldw, work, bytes);
        for (int j = k + 1; j < n;)
        {
            const int jj = j;
            int jp = ipiv[j];
            if (jp < 0)
            {
                jp = -jp;
                ++j;
            }
            ++j;
            --jp;
            if (jp != jj && j < n)
                blas_impl::swap_run<gpu, Z>(h, n - j, A(jp, j), lda, A(jj, j), lda);
        }
        kb = rank;
    }
    else
    {
        int k = 0;
        const int limit = n <= nb ? n : nb - 1;
        for (; k < limit;)
        {
            DDLA_CHECK(ddlaMemcpy2D(h, W(k, k), zbytes, A(k, k), zbytes, zbytes, n - k));
            blas_impl::real_diagonal<gpu, Z>(h, 1, W(k, k), 1);
            if (k > 0)
            {
                blas_impl::gemv_run<gpu, Z>(h, 'N', n - k, k, minus_one, A(k, 0), lda, W(k, 0), ldw,
                                            one, W(k, k), 1);
                blas_impl::real_diagonal<gpu, Z>(h, 1, W(k, k), 1);
            }
            int step = 1, kp = k, imax = 0;
            const double diagonal = std::abs(read_pivot(h, W(k, k)).real());
            double colmax = 0;
            if (k < n - 1)
            {
                blas_impl::iamax_run<gpu, Z>(h, n - k - 1, W(k + 1, k), 1, imax);
                imax += k;
                colmax = abs1(read_pivot(h, W(imax, k)));
            }
            if (std::max(diagonal, colmax) == 0)
            {
                if (!info)
                    info = k + 1;
                DDLA_CHECK(ddlaMemcpy2D(h, A(k, k), zbytes, W(k, k), zbytes, zbytes, n - k));
            }
            else
            {
                if (diagonal < alpha * colmax)
                {
                    hetrf_conj_copy(h, imax - k, A(imax, k), lda, W(k, k + 1), 1);
                    DDLA_CHECK(ddlaMemcpy2D(h, W(imax, k + 1), zbytes, A(imax, imax), zbytes,
                                            zbytes, n - imax));
                    blas_impl::real_diagonal<gpu, Z>(h, 1, W(imax, k + 1), 1);
                    if (k > 0)
                    {
                        blas_impl::gemv_run<gpu, Z>(h, 'N', n - k, k, minus_one, A(k, 0), lda,
                                                    W(imax, 0), ldw, one, W(k, k + 1), 1);
                        blas_impl::real_diagonal<gpu, Z>(h, 1, W(imax, k + 1), 1);
                    }
                    const double imax_diagonal = std::abs(read_pivot(h, W(imax, k + 1)).real());
                    int jmax = 0;
                    blas_impl::iamax_run<gpu, Z>(h, imax - k, W(k, k + 1), 1, jmax);
                    jmax += k - 1;
                    double rowmax = abs1(read_pivot(h, W(jmax, k + 1)));
                    if (imax < n - 1)
                    {
                        blas_impl::iamax_run<gpu, Z>(h, n - imax - 1, W(imax + 1, k + 1), 1, jmax);
                        jmax += imax;
                        rowmax = std::max(rowmax, abs1(read_pivot(h, W(jmax, k + 1))));
                    }
                    if (diagonal >= alpha * colmax * (colmax / rowmax))
                    {
                        kp = k;
                    }
                    else if (imax_diagonal >= alpha * rowmax)
                    {
                        kp = imax;
                        DDLA_CHECK(
                            ddlaMemcpy2D(h, W(k, k), zbytes, W(k, k + 1), zbytes, zbytes, n - k));
                    }
                    else
                    {
                        kp = imax;
                        step = 2;
                    }
                }
                const int kk = k + step - 1;
                if (kp != kk)
                {
                    hetrf_conj_copy(h, kp - kk, A(kk, kk), 1, A(kp, kk), lda);
                    DDLA_CHECK(
                        ddlaMemcpy2D(h, A(kp, kp), zbytes, A(kp, kk), zbytes, zbytes, n - kp));
                    blas_impl::real_diagonal<gpu, Z>(h, 1, A(kp, kp), 1);
                    blas_impl::swap_run<gpu, Z>(h, kk + 1, A(kk, 0), lda, A(kp, 0), lda);
                    blas_impl::swap_run<gpu, Z>(h, kk + 1, W(kk, 0), ldw, W(kp, 0), ldw);
                }
                if (step == 1)
                {
                    DDLA_CHECK(ddlaMemcpy2D(h, A(k, k), zbytes, W(k, k), zbytes, zbytes, n - k));
                    if (k < n - 1)
                    {
                        const double inverse = 1.0 / read_pivot(h, W(k, k)).real();
                        blas_impl::scal_run<gpu, Z>(h, n - k - 1, Z(inverse), A(k + 1, k), 1);
                        hetrf_conj_copy(h, n - k - 1, W(k + 1, k), 1, W(k + 1, k), 1);
                    }
                }
                else
                {
                    hetrf_scale_2x2(h, 'L', n - k - 2, W(k, k), ldw, A(k + 2, k), lda);
                    DDLA_CHECK(ddlaMemcpy(h, A(k, k), W(k, k), 2 * zbytes));
                    DDLA_CHECK(ddlaMemcpy(h, A(k + 1, k + 1), W(k + 1, k + 1), zbytes));
                    hetrf_conj_copy(h, n - k - 1, W(k + 1, k), 1, W(k + 1, k), 1);
                    hetrf_conj_copy(h, n - k - 2, W(k + 2, k + 1), 1, W(k + 2, k + 1), 1);
                }
            }
            ipiv[k] = step == 1 ? kp + 1 : -(kp + 1);
            if (step == 2)
                ipiv[k + 1] = -(kp + 1);
            k += step;
        }

        update_panel(h, uplo, n, nb, k, a, lda, w, ldw, work, bytes);
        for (int j = k; j > 0;)
        {
            const int jj = j;
            int jp = ipiv[j - 1];
            if (jp < 0)
            {
                jp = -jp;
                --j;
            }
            --j;
            if (jp != jj && j >= 1)
                blas_impl::swap_run<gpu, Z>(h, j, A(jp - 1, 0), lda, A(jj - 1, 0), lda);
        }
        kb = k;
    }
}
} // namespace

void hetrf_gpu(const DdlaHandle_t& h, char uplo, int n, Z* a, int lda, int* ipiv, int& info)
{
    uplo = local_check<gpu, true, false, Z>(h, uplo, n, 0, a, lda, ipiv, nullptr, 1, info);
    if (!n)
        return;
    if (n == 1)
    {
        blas_impl::real_diagonal<gpu, Z>(h, 1, a, lda);
        info = read_pivot(h, a).real() == 0 ? 1 : 0;
        ipiv[0] = 1;
        return;
    }
    const int block = std::min(n, 128);
    vector<Z> w(h, std::size_t(n) * block);
    // Avoid per-pivot host synchronization on small matrices.
    const bool fused = n <= 2048;
    vector<int> device_metadata(h, fused ? n + 2 : 0);
    int* const device_piv = fused ? device_metadata.data() : nullptr;
    int* const device_status = fused ? device_piv + n : nullptr;
    if (n <= block)
    {
        hetrf_panel_factor(h, uplo, n, n, a, lda, device_piv, w.data(), n, device_status, 0);
        std::vector<int> metadata(n + 2);
        DDLA_CHECK(ddlaMemcpy(h, metadata.data(), device_piv, metadata.size() * sizeof(int)));
        DDLA_CHECK(ddlaSynchronize(h));
        std::copy_n(metadata.data(), n, ipiv);
        info = metadata[n + 1];
        return;
    }
    std::size_t bytes = 0;
    for (int width = 1; width <= block; ++width)
    {
        std::size_t next = 0;
        blas_impl::gemmt_workspace_size<Z>(h, 'N', 'T', width, block, &next);
        bytes = std::max(bytes, next);
    }
    vector<char> work(h, bytes);
    if (uplo == 'U')
    {
        for (int active = n; active > 0;)
        {
            int kb = 0, partial_info = 0;
            panel(h, uplo, active, std::min(block, active), kb, a, lda, ipiv, w.data(), n,
                  partial_info, work.data(), bytes, device_piv, device_status, 0);
            if (!info && partial_info)
                info = partial_info;
            active -= kb;
        }
    }
    else
    {
        for (int offset = 0; offset < n;)
        {
            int kb = 0, partial_info = 0;
            panel(h, uplo, n - offset, std::min(block, n - offset), kb,
                  a + offset + std::size_t(offset) * lda, lda, ipiv + offset, w.data(), n,
                  partial_info, work.data(), bytes, fused ? device_piv + offset : nullptr,
                  device_status, offset);
            if (!info && partial_info)
                info = partial_info + offset;
            if (!fused)
                for (int i = offset; i < offset + kb; ++i)
                    ipiv[i] += ipiv[i] > 0 ? offset : -offset;
            offset += kb;
        }
    }
    if (fused)
        DDLA_CHECK(ddlaMemcpy(h, ipiv, device_piv, std::size_t(n) * sizeof(int)));
    // Work arrays retire only after all updates using them have completed.
    DDLA_CHECK(ddlaSynchronize(h));
}
} // namespace ddla::ldlt_impl
