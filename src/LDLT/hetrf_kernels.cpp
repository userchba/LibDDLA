// Adapted from MAGMA magmablas/zlascl_2x2.cu (Ichitaro Yamazaki) and
// zlahef_gpu.cpp. Copyright (c) 2009-2023, The University of Tennessee.
// BSD-3-Clause terms and disclaimer are retained in MAGMA-LICENSE.
#include "hetrf_gpu.h"
#include "ddla_stream.h"
#include "helpers.h"
#include "ldlt_gpu_max.h"
#include "gpu/gpu.h"
#include <algorithm>

namespace ddla::ldlt_impl
{
namespace
{
using Z = detail::DeviceScalarT<std::complex<double>>;

__global__ void conjugate_copy(int n, const Z* src, int incs, Z* dst, int incd)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        dst[std::size_t(i) * incd] = thrust::conj(src[std::size_t(i) * incs]);
}

__global__ void scale_pivot_pair(bool upper, int rows, const Z* w, int ldw, Z* a, int lda)
{
    const Z off = upper ? w[rows + ldw] : w[1];
    const Z d11 = upper ? w[rows + 1 + ldw] / thrust::conj(off) : w[1 + ldw] / off;
    const Z d22 = upper ? w[rows] / off : w[0] / thrust::conj(off);
    const double t = 1.0 / ((d11 * d22).real() - 1.0);
    const Z inverse_off = Z(t, 0) / off;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows; i += blockDim.x * gridDim.x)
    {
        const int wi = upper ? i : i + 2;
        const Z x = w[wi], y = w[wi + ldw];
        a[i] = (upper ? inverse_off : thrust::conj(inverse_off)) * (d11 * x - y);
        a[i + lda] = (upper ? thrust::conj(inverse_off) : inverse_off) * (d22 * y - x);
    }
}
} // namespace

void hetrf_conj_copy(const DdlaHandle_t& h, int n, const std::complex<double>* src, int incs,
                     std::complex<double>* dst, int incd)
{
    if (n <= 0)
        return;
    conjugate_copy<<<std::min((n + 127) / 128, 65535), 128, 0,
                     static_cast<gpuStream_t>(h->stream)>>>(n, reinterpret_cast<const Z*>(src),
                                                            incs, reinterpret_cast<Z*>(dst), incd);
    DDLA_CHECK(ddlaGetLastError(h));
}

void hetrf_scale_2x2(const DdlaHandle_t& h, char uplo, int rows, const std::complex<double>* w,
                     int ldw, std::complex<double>* a, int lda)
{
    if (rows <= 0)
        return;
    scale_pivot_pair<<<std::min((rows + 127) / 128, 65535), 128, 0,
                       static_cast<gpuStream_t>(h->stream)>>>(
        uplo == 'U', rows, reinterpret_cast<const Z*>(w), ldw, reinterpret_cast<Z*>(a), lda);
    DDLA_CHECK(ddlaGetLastError(h));
}
namespace
{
constexpr int panel_threads = 256;

// The existing BLAS panel needs several host reductions per pivot. For small
// matrices, fuse the same LAHEF operations into one cooperative GPU block.
// Row dot products and pivot reductions stay on the device; large trailing
// products still use the shared GEMMT/GEMM update in the host driver.
__device__ void finish_panel(bool upper, int n, int kb, Z* a, int lda, int* piv, int offset);

__global__ void panel_factor(bool upper, int n, int nb, Z* a, int lda, int* piv, Z* w, int ldw,
                             int* status, int offset)
{
    __shared__ double values[panel_threads];
    __shared__ int indices[panel_threads];
    __shared__ double diagonal, colmax, rowmax, inverse, cached[12];
    __shared__ int k, kw, kp, imax, step, zero, neighbor, first_info;
    Z* cache = reinterpret_cast<Z*>(cached);
    const double alpha = (1.0 + sqrt(17.0)) / 8.0;
    if (!threadIdx.x)
    {
        k = upper ? n - 1 : 0;
        first_info = 0;
    }
    __syncthreads();
    while (upper ? (k >= 0 && (n <= nb || k >= n - nb + 1)) : (k < (n <= nb ? n : nb - 1)))
    {
        if (!threadIdx.x)
        {
            kw = upper ? nb - (n - k) : k;
            step = 1;
            kp = k;
        }
        __syncthreads();
        const int first = upper ? 0 : k, last = upper ? k + 1 : n;
        for (int i = first + threadIdx.x; i < last; i += blockDim.x)
        {
            Z z = a[i + std::size_t(k) * lda];
            if (i == k)
                z.imag(0);
            if (upper)
            {
                for (int j = k + 1; j < n; ++j)
                    z -= a[i + std::size_t(j) * lda] * w[k + std::size_t(nb - n + j) * ldw];
            }
            else
            {
                for (int j = 0; j < k; ++j)
                    z -= a[i + std::size_t(j) * lda] * w[k + std::size_t(j) * ldw];
            }
            if (i == k)
                z.imag(0);
            w[i + std::size_t(kw) * ldw] = z;
        }
        __syncthreads();
        gpu_detail::max_abs_column(first, last, k, w + std::size_t(kw) * ldw, 1, values, indices);
        if (!threadIdx.x)
        {
            diagonal = fabs(w[k + std::size_t(kw) * ldw].real());
            colmax = fmax(0.0, values[0]);
            imax = indices[0];
            zero = fmax(diagonal, colmax) == 0;
            neighbor = !zero && diagonal < alpha * colmax;
            if (zero && !first_info)
                first_info = k + 1;
        }
        __syncthreads();
        const int next = upper ? kw - 1 : kw + 1;
        if (neighbor)
        {
            for (int i = first + threadIdx.x; i < last; i += blockDim.x)
            {
                Z z;
                if (upper)
                    z = i <= imax ? a[i + std::size_t(imax) * lda]
                                  : thrust::conj(a[imax + std::size_t(i) * lda]);
                else
                    z = i >= imax ? a[i + std::size_t(imax) * lda]
                                  : thrust::conj(a[imax + std::size_t(i) * lda]);
                if (i == imax)
                    z.imag(0);
                if (upper)
                {
                    for (int j = k + 1; j < n; ++j)
                        z -= a[i + std::size_t(j) * lda] * w[imax + std::size_t(nb - n + j) * ldw];
                }
                else
                {
                    for (int j = 0; j < k; ++j)
                        z -= a[i + std::size_t(j) * lda] * w[imax + std::size_t(j) * ldw];
                }
                if (i == imax)
                    z.imag(0);
                w[i + std::size_t(next) * ldw] = z;
            }
            __syncthreads();
            gpu_detail::max_abs_column(first, last, imax, w + std::size_t(next) * ldw, 1, values,
                                       indices);
            if (!threadIdx.x)
            {
                rowmax = fmax(0.0, values[0]);
                const double other_diagonal = fabs(w[imax + std::size_t(next) * ldw].real());
                if (diagonal >= alpha * colmax * (colmax / rowmax))
                    kp = k;
                else if (other_diagonal >= alpha * rowmax)
                    kp = imax;
                else
                {
                    kp = imax;
                    step = 2;
                }
            }
            __syncthreads();
            if (step == 1 && kp != k)
            {
                for (int i = first + threadIdx.x; i < last; i += blockDim.x)
                    w[i + std::size_t(kw) * ldw] = w[i + std::size_t(next) * ldw];
            }
            __syncthreads();
        }
        const int kk = upper ? k - step + 1 : k + step - 1;
        if (!zero && kp != kk)
        {
            if (upper)
            {
                const int kkw = nb - n + kk;
                for (int j = threadIdx.x; j < n - kk; j += blockDim.x)
                {
                    Z z = a[kk + std::size_t(kk + j) * lda];
                    a[kk + std::size_t(kk + j) * lda] = a[kp + std::size_t(kk + j) * lda];
                    a[kp + std::size_t(kk + j) * lda] = z;
                    z = w[kk + std::size_t(kkw + j) * ldw];
                    w[kk + std::size_t(kkw + j) * ldw] = w[kp + std::size_t(kkw + j) * ldw];
                    w[kp + std::size_t(kkw + j) * ldw] = z;
                }
                __syncthreads();
                for (int i = threadIdx.x; i < kk - kp - 1; i += blockDim.x)
                    a[kp + std::size_t(kp + 1 + i) * lda] =
                        thrust::conj(a[kp + 1 + i + std::size_t(kk) * lda]);
                __syncthreads();
                for (int i = threadIdx.x; i <= kp; i += blockDim.x)
                {
                    Z z = a[i + std::size_t(kk) * lda];
                    if (i == kp)
                        z.imag(0);
                    a[i + std::size_t(kp) * lda] = z;
                }
            }
            else
            {
                for (int i = threadIdx.x; i < kp - kk; i += blockDim.x)
                    a[kp + std::size_t(kk + i) * lda] =
                        thrust::conj(a[kk + i + std::size_t(kk) * lda]);
                __syncthreads();
                for (int i = kp + threadIdx.x; i < n; i += blockDim.x)
                {
                    Z z = a[i + std::size_t(kk) * lda];
                    if (i == kp)
                        z.imag(0);
                    a[i + std::size_t(kp) * lda] = z;
                }
                __syncthreads();
                for (int j = threadIdx.x; j <= kk; j += blockDim.x)
                {
                    Z z = a[kk + std::size_t(j) * lda];
                    a[kk + std::size_t(j) * lda] = a[kp + std::size_t(j) * lda];
                    a[kp + std::size_t(j) * lda] = z;
                    z = w[kk + std::size_t(j) * ldw];
                    w[kk + std::size_t(j) * ldw] = w[kp + std::size_t(j) * ldw];
                    w[kp + std::size_t(j) * ldw] = z;
                }
            }
            __syncthreads();
        }
        if (step == 1)
        {
            if (!threadIdx.x)
                inverse = zero ? 0.0 : 1.0 / w[k + std::size_t(kw) * ldw].real();
            __syncthreads();
            for (int i = first + threadIdx.x; i < last; i += blockDim.x)
            {
                const Z z = w[i + std::size_t(kw) * ldw];
                a[i + std::size_t(k) * lda] = i == k || zero ? z : z * inverse;
                if (i != k)
                    w[i + std::size_t(kw) * ldw] = thrust::conj(z);
            }
        }
        else
        {
            if (!threadIdx.x)
            {
                const int drow = upper ? k - 1 : k, dcol = upper ? kw - 1 : kw;
                const Z d00 = w[drow + std::size_t(dcol) * ldw];
                const Z d11 = w[drow + 1 + std::size_t(dcol + 1) * ldw];
                const Z off = upper ? w[drow + std::size_t(dcol + 1) * ldw]
                                    : w[drow + 1 + std::size_t(dcol) * ldw];
                cache[0] = upper ? d11 / thrust::conj(off) : d11 / off;
                cache[1] = upper ? d00 / off : d00 / thrust::conj(off);
                cache[2] = Z(1.0 / ((cache[0] * cache[1]).real() - 1.0), 0) / off;
                a[drow + std::size_t(drow) * lda] = d00;
                a[drow + 1 + std::size_t(drow + 1) * lda] = d11;
                if (upper)
                    a[drow + std::size_t(drow + 1) * lda] = off;
                else
                    a[drow + 1 + std::size_t(drow) * lda] = off;
            }
            __syncthreads();
            const int dcol = upper ? kw - 1 : kw, acol = upper ? k - 1 : k;
            for (int i = first + threadIdx.x; i < last; i += blockDim.x)
            {
                if (upper ? i < k - 1 : i > k + 1)
                {
                    const Z x = w[i + std::size_t(dcol) * ldw],
                            y = w[i + std::size_t(dcol + 1) * ldw];
                    a[i + std::size_t(acol) * lda] =
                        (upper ? cache[2] : thrust::conj(cache[2])) * (cache[0] * x - y);
                    a[i + std::size_t(acol + 1) * lda] =
                        (upper ? thrust::conj(cache[2]) : cache[2]) * (cache[1] * y - x);
                }
                if (upper ? i < k : i > k)
                    w[i + std::size_t(kw) * ldw] = thrust::conj(w[i + std::size_t(kw) * ldw]);
                if (upper ? i < k - 1 : i > k + 1)
                    w[i + std::size_t(next) * ldw] = thrust::conj(w[i + std::size_t(next) * ldw]);
            }
        }
        __syncthreads();
        if (!threadIdx.x)
        {
            piv[k] = step == 1 ? kp + 1 : -(kp + 1);
            if (step == 2)
                piv[upper ? k - 1 : k + 1] = -(kp + 1);
            k += upper ? -step : step;
        }
        __syncthreads();
    }
    if (!threadIdx.x)
    {
        status[0] = upper ? n - k - 1 : k;
        status[1] = first_info;
    }
    // No trailing update remains for a whole final panel, so finish its
    // permutations in this launch instead of returning to the host first.
    if (n <= nb)
        finish_panel(upper, n, n, a, lda, piv, offset);
}

__device__ void finish_panel(bool upper, int n, int kb, Z* a, int lda, int* piv, int offset)
{
    __shared__ int j, jj, jp, active;
    if (!threadIdx.x)
        j = upper ? n - kb : kb;
    __syncthreads();
    for (;;)
    {
        if (!threadIdx.x)
            active = upper ? j < n : j > 0;
        __syncthreads();
        if (!active)
            break;
        if (!threadIdx.x)
        {
            jj = j;
            jp = piv[upper ? j : j - 1];
            if (jp < 0)
            {
                jp = -jp;
                j += upper ? 1 : -1;
            }
            j += upper ? 1 : -1;
            if (upper)
                --jp;
        }
        __syncthreads();
        if (upper)
        {
            if (jp != jj && j < n)
                for (int col = j + threadIdx.x; col < n; col += blockDim.x)
                {
                    const Z z = a[jp + std::size_t(col) * lda];
                    a[jp + std::size_t(col) * lda] = a[jj + std::size_t(col) * lda];
                    a[jj + std::size_t(col) * lda] = z;
                }
        }
        else
        {
            if (jp != jj && j >= 1)
                for (int col = threadIdx.x; col < j; col += blockDim.x)
                {
                    const Z z = a[jp - 1 + std::size_t(col) * lda];
                    a[jp - 1 + std::size_t(col) * lda] = a[jj - 1 + std::size_t(col) * lda];
                    a[jj - 1 + std::size_t(col) * lda] = z;
                }
        }
        __syncthreads();
    }
    if (!upper && offset)
        for (int i = threadIdx.x; i < kb; i += blockDim.x)
            piv[i] += piv[i] > 0 ? offset : -offset;
}
__global__ void panel_undo(bool upper, int n, int kb, Z* a, int lda, int* piv, int offset)
{
    finish_panel(upper, n, kb, a, lda, piv, offset);
}
} // namespace

void hetrf_panel_factor(const DdlaHandle_t& h, char uplo, int n, int nb, std::complex<double>* a,
                        int lda, int* piv, std::complex<double>* w, int ldw, int* status,
                        int offset)
{
    panel_factor<<<1, panel_threads, 0, static_cast<gpuStream_t>(h->stream)>>>(
        uplo == 'U', n, nb, reinterpret_cast<Z*>(a), lda, piv, reinterpret_cast<Z*>(w), ldw, status,
        offset);
    DDLA_CHECK(ddlaGetLastError(h));
}

void hetrf_panel_undo(const DdlaHandle_t& h, char uplo, int n, int kb, std::complex<double>* a,
                      int lda, int* piv, int offset)
{
    panel_undo<<<1, panel_threads, 0, static_cast<gpuStream_t>(h->stream)>>>(
        uplo == 'U', n, kb, reinterpret_cast<Z*>(a), lda, piv, offset);
    DDLA_CHECK(ddlaGetLastError(h));
}

} // namespace ddla::ldlt_impl
