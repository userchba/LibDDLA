#include "BLAS/blas_impl.h"
#include "Eigensolver/chebyshev_impl.h"
#include "PBLAS/pgeadd_impl.h"
#include "PBLAS/pblas_workspace.h"
#include "backend_api.h"
#include "check_agree.h"
#include "comm_traits.h"
#include "ddla_desc.h"
#include "ddla_stream.h"
#include "helpers.h"
#include "host_parallel.h"
#include "instantiate.h"
#include "mpi_datatype.h"
#include "plan_util.h"
#include "require_backend.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <climits>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ddla
{
namespace spectral_impl
{
namespace chebyshev
{
namespace
{
constexpr int scan_blocks = 256, scan_threads = 256;
int global_index(int local, int block, int coordinate, int processes)
{
    return ((local / block) * processes + coordinate) * block + local % block;
}
template <class T> double re(T x)
{
    if constexpr (std::is_arithmetic<T>::value)
        return double(x);
    else
        return double(x.real());
}
template <class T> double im(T x)
{
    if constexpr (std::is_arithmetic<T>::value)
        return 0;
    else
        return double(x.imag());
}
template <class T> T number(double x, double y = 0)
{
    using R = typename real_type<T>::type;
    if constexpr (std::is_arithmetic<T>::value)
        return R(x);
    else
        return T(R(x), R(y));
}
template <class R> R shift(R x, int exponent)
{
    return std::scalbn(x, exponent);
}
std::size_t bytes(std::size_t count, std::size_t width)
{
    if (count > (std::numeric_limits<std::size_t>::max() - 255) / width)
        throw std::overflow_error("chebyshev: workspace overflow");
    return (count * width + 255) / 256 * 256;
}
void include(std::size_t b, std::size_t h, Plan &p)
{
    p.phase_backend = std::max(p.phase_backend, b);
    p.phase_host = std::max(p.phase_host, h);
}
void sum(const DdlaHandle_t &h, double *x, int count)
{
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, x, count, MPI_DOUBLE, MPI_SUM, h->comm));
}
void maximum(const DdlaHandle_t &h, double *x, int count)
{
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, x, count, MPI_DOUBLE, MPI_MAX, h->comm));
}
void stale(const DdlaHandle_t &h, const Plan &p)
{
    if (!p.ready || h->backend != p.backend || ddlaGetCommunication(h) != p.transport ||
        h->stream != p.compute || h->stream_data != p.data || h->host_threads != p.threads ||
        h->local_device != p.device || h->comm != p.communicator || h->nprows_ != p.pr ||
        h->npcols_ != p.pc || h->myprow_ != p.r || h->mypcol_ != p.c || h->major != p.major)
        throw std::invalid_argument("chebyshev: handle changed; prepare a new plan");
}
template <class T>
void add_bytes(const DdlaHandle_t &h, const Plan &p, char op, int m, int n, const int *a,
               const int *c, Plan &output)
{
    if (!m || !n)
        return;
    std::size_t b = 0, cpu = 0;
    pgeadd_impl::pgeadd_workspace_size<T>(h, op, 'N', true, false, m, n, c, &b, &cpu, a, nullptr,
                                          true);
    include(b, cpu, output);
}
template <class T>
void gemm_bytes(const DdlaHandle_t &h, Plan &p, pgemm_impl::Algorithm algorithm, int width, char ta,
                char tb, int m, int n, int k, const int *a, const int *b, const int *c)
{
    std::size_t backend = 0, host = 0;
    pgemm_impl::pgemm_workspace_size<T>(h, algorithm, width, false, ta, tb, m, n, k, false, a, b, c,
                                        &backend, &host);
    include(backend, host, p);
}
template <DdlaBackend B, class T>
void copy(const DdlaHandle_t &h, const Plan &p, const T *a, const int *da, T *c, const int *dc,
          int m, int n, void *backend, std::size_t bb, void *host, std::size_t hb)
{
    if (!m || !n)
        return;
    pgeadd_impl::pgeadd_run<B, T>(h, 'N', 'N', m, n, T(1), a, 1, 1, da, T(0), nullptr, 1, 1, dc, c,
                                  1, 1, dc, backend, bb, host, hb);
}
template <DdlaBackend B, class T>
void local_copy(const DdlaHandle_t &h, const Plan &p, const T *a, T *c)
{
    if (p.rows && p.columns)
        DDLA_CHECK(ddlaMemcpy2D(h, c, std::size_t(p.basis[DDLA_LLD_]) * sizeof(T), a,
                                std::size_t(p.basis[DDLA_LLD_]) * sizeof(T),
                                std::size_t(p.rows) * sizeof(T), p.columns));
}

// No existing BLAS wrapper supplies selected-uplo finite/roundtrip scanning,
// row absolute sums and column masks together. These helpers supply only that
// elementwise/reduction work; all matrix products/QR/eigensolves use DDLA.
#if DDLA_HAS_GPU
template <class D> __device__ double dreal(D x)
{
    if constexpr (std::is_arithmetic<D>::value)
        return double(x);
    else
        return double(x.real());
}
template <class D> __device__ double dimag(D x)
{
    if constexpr (std::is_arithmetic<D>::value)
        return 0;
    else
        return double(x.imag());
}
template <class D> __device__ D dnumber(double x, double y = 0)
{
    if constexpr (std::is_arithmetic<D>::value)
        return D(x);
    else
        return D(x, y);
}
__device__ int dindex(int i, int b, int c, int processes)
{
    return ((i / b) * processes + c) * b + i % b;
}
template <class D>
__global__ void matrix_scan(D *a, int lda, int nr, int nc, int b, int r, int c, int pr, int pc,
                            bool upper, int mode, int exponent, double *output)
{
    __shared__ double maxima[scan_threads];
    __shared__ int failures[scan_threads];
    double mx = 0;
    int bad = 0;
    const unsigned long long total = static_cast<unsigned long long>(nr) * nc;
    for (unsigned long long at =
             static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         at < total; at += static_cast<unsigned long long>(gridDim.x) * blockDim.x)
    {
        const int i = int(at % nr), j = int(at / nr), gi = dindex(i, b, r, pr),
                  gj = dindex(j, b, c, pc);
        if ((upper && gi > gj) || (!upper && gi < gj))
            continue;
        const D old = a[i + static_cast<unsigned long long>(j) * lda];
        const double vr = dreal(old), vi = gi == gj ? 0 : dimag(old);
        if (mode == 0)
        {
            bad |= !isfinite(vr) || !isfinite(vi);
            mx = fmax(mx, fmax(fabs(vr), fabs(vi)));
        }
        else
        {
            const D changed = dnumber<D>(scalbn(vr, -exponent), scalbn(vi, -exponent));
            if (mode == 1)
                bad |= scalbn(dreal(changed), exponent) != vr ||
                       scalbn(dimag(changed), exponent) != vi;
            else
                a[i + static_cast<unsigned long long>(j) * lda] = changed;
        }
    }
    if (mode == 2)
        return;
    maxima[threadIdx.x] = mx;
    failures[threadIdx.x] = bad;
    __syncthreads();
    for (int step = blockDim.x / 2; step; step /= 2)
    {
        if (threadIdx.x < step)
        {
            maxima[threadIdx.x] = fmax(maxima[threadIdx.x], maxima[threadIdx.x + step]);
            failures[threadIdx.x] |= failures[threadIdx.x + step];
        }
        __syncthreads();
    }
    if (!threadIdx.x)
    {
        output[2 * blockIdx.x] = maxima[0];
        output[2 * blockIdx.x + 1] = failures[0];
    }
}
// One row per thread, coalesced across adjacent rows on each column read.
template <class D>
__global__ void row_statistics(const D *a, int lda, int nr, int nc, int b, int r, int c, int pr,
                               int pc, double *output)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nr; i += blockDim.x * gridDim.x)
    {
        const int gi = dindex(i, b, r, pr);
        double total = 0, diagonal = 0;
        for (int j = 0; j < nc; ++j)
        {
            const D v = a[i + static_cast<unsigned long long>(j) * lda];
            total += hypot(dreal(v), dimag(v));
            if (gi == dindex(j, b, c, pc))
                diagonal = dreal(v);
        }
        output[i] = total;
        output[nr + i] = diagonal;
    }
}
template <class D>
__global__ void column_statistics(const D *x, const D *ax, int lda, int nr, int nc, int q, int nb,
                                  int pc, int c, int mode, const double *parameters, double *output)
{
    __shared__ double accum[4][scan_threads];
    const int j = blockIdx.x;
    if (j >= nc)
        return;
    const int gj = dindex(j, nb, c, pc);
    double v[4] = {0, 0, 0, 0};
    for (int i = threadIdx.x; i < nr; i += blockDim.x)
    {
        const D value = x[i + static_cast<unsigned long long>(j) * lda];
        const double xr = dreal(value), xi = dimag(value);
        if (mode == 0)
            v[0] = fmax(v[0], isfinite(xr) && isfinite(xi) ? fmax(fabs(xr), fabs(xi)) : INFINITY);
        else if (mode == 1)
        {
            const double scale = parameters[gj];
            if (scale > 0)
            {
                const double ar = xr / scale, ai = xi / scale;
                v[0] += ar * ar + ai * ai;
            }
        }
        else
        {
            const D action = ax[i + static_cast<unsigned long long>(j) * lda];
            const double ar = dreal(action), ai = dimag(action);
            const double rounded = parameters[gj], raw = parameters[q + gj];
            v[0] += xr * xr + xi * xi;
            v[1] += (ar - rounded * xr) * (ar - rounded * xr) +
                    (ai - rounded * xi) * (ai - rounded * xi);
            v[2] += (ar - raw * xr) * (ar - raw * xr) + (ai - raw * xi) * (ai - raw * xi);
            v[3] += !isfinite(xr) || !isfinite(xi) || !isfinite(ar) || !isfinite(ai);
        }
    }
    for (int k = 0; k < 4; ++k)
        accum[k][threadIdx.x] = v[k];
    __syncthreads();
    for (int step = blockDim.x / 2; step; step /= 2)
    {
        if (threadIdx.x < step)
            for (int k = 0; k < 4; ++k)
                accum[k][threadIdx.x] =
                    mode == 0 ? fmax(accum[k][threadIdx.x], accum[k][threadIdx.x + step])
                              : accum[k][threadIdx.x] + accum[k][threadIdx.x + step];
        __syncthreads();
    }
    if (!threadIdx.x)
        for (int k = 0; k < (mode == 2 ? 4 : 1); ++k)
            output[k * q + gj] = accum[k][0];
}
template <class D>
__global__ void transform_columns(D *x, const D *random, int lda, int nr, int nc, int n, int q,
                                  int mb, int nb, int r, int c, int pr, int pc, int mode, int nev,
                                  double delta, const double *parameters)
{
    const unsigned long long total = static_cast<unsigned long long>(nr) * nc;
    for (unsigned long long at =
             static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         at < total; at += static_cast<unsigned long long>(blockDim.x) * gridDim.x)
    {
        const int i = int(at % nr), j = int(at / nr), gj = dindex(j, nb, c, pc);
        const auto offset = i + static_cast<unsigned long long>(j) * lda;
        if (mode == 0)
            x[offset] = dnumber<D>(dindex(i, mb, r, pr) == gj ? 1 : 0);
        else if (mode == 1)
        {
            const double scale = parameters[gj], length = sqrt(parameters[q + gj]);
            x[offset] = dnumber<D>((dreal(x[offset]) / scale) / length,
                                   (dimag(x[offset]) / scale) / length);
        }
        else if (mode == 2)
        {
            if (gj < nev)
                x[offset] += dnumber<D>(delta) * random[offset];
        }
        else if (parameters[gj] != 0)
            x[offset] = random[offset];
    }
}
template <class D>
__global__ void qr_diagonal(const D *x, int lda, int nr, int nc, int q, int mb, int nb, int r,
                            int c, int pr, int pc, double *output)
{
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < nc; j += blockDim.x * gridDim.x)
    {
        const int gj = dindex(j, nb, c, pc);
        if (gj >= q || (gj / mb) % pr != r)
            continue;
        const int i = (gj / mb / pr) * mb + gj % mb;
        if (i < nr)
        {
            const D v = x[i + static_cast<unsigned long long>(j) * lda];
            output[gj] = hypot(dreal(v), dimag(v));
        }
    }
}
#endif

template <DdlaBackend B, class T> void fetch(const DdlaHandle_t &h, Resources<T> &r, int count)
{
    if constexpr (B == DdlaBackend::GPU)
    {
        DDLA_CHECK(ddlaMemcpy(h, r.stats.data(), r.device_stats.data(),
                              std::size_t(count) * sizeof(double)));
        DDLA_CHECK(ddlaSynchronize(h));
    }
}
template <DdlaBackend B, class T> void parameters(const DdlaHandle_t &h, Resources<T> &r, int count)
{
    if constexpr (B == DdlaBackend::GPU)
        DDLA_CHECK(ddlaMemcpy(h, r.device_parameters.data(), r.parameters.data(),
                              std::size_t(count) * sizeof(double)));
}
template <DdlaBackend B, class T>
std::pair<double, int> scan(const DdlaHandle_t &h, const Plan &p, Resources<T> &r, T *a, int mode,
                            int exponent)
{
    std::fill_n(r.stats.data(), 2 * scan_blocks, 0.);
    if constexpr (B == DdlaBackend::CPU)
    {
        const auto total = static_cast<long long>(p.rows) * num_loc(p.n, p.bA, p.c, 0, p.pc);
        detail::host_parallel_for(
            scan_blocks, 1, p.threads,
            [&](long long lo, long long hi)
            {
                for (int chunk = int(lo); chunk < int(hi); ++chunk)
                {
                    double mx = 0;
                    int bad = 0;
                    for (long long at = total * chunk / scan_blocks;
                         at < total * (chunk + 1) / scan_blocks; ++at)
                    {
                        const int i = int(at % p.rows), j = int(at / p.rows),
                                  gi = global_index(i, p.bA, p.r, p.pr),
                                  gj = global_index(j, p.bA, p.c, p.pc);
                        if ((p.uplo == 'U' && gi > gj) || (p.uplo == 'L' && gi < gj))
                            continue;
                        const T old = a[i + std::size_t(j) * p.a[DDLA_LLD_]];
                        const double vr = re(old), vi = gi == gj ? 0 : im(old);
                        if (!mode)
                        {
                            bad |= !std::isfinite(vr) || !std::isfinite(vi);
                            mx = std::max({mx, std::abs(vr), std::abs(vi)});
                        }
                        else
                        {
                            using R = typename real_type<T>::type;
                            const T changed =
                                number<T>(shift(R(vr), -exponent), shift(R(vi), -exponent));
                            if (mode == 1)
                                bad |= shift(R(re(changed)), exponent) != R(vr) ||
                                       shift(R(im(changed)), exponent) != R(vi);
                            else
                                a[i + std::size_t(j) * p.a[DDLA_LLD_]] = changed;
                        }
                    }
                    r.stats[2 * chunk] = mx;
                    r.stats[2 * chunk + 1] = bad;
                }
            });
    }
    else
    {
#if DDLA_HAS_GPU
        using D = detail::DeviceScalarT<T>;
        matrix_scan<D><<<scan_blocks, scan_threads, 0, static_cast<gpuStream_t>(h->stream)>>>(
            reinterpret_cast<D *>(a), p.a[DDLA_LLD_], p.rows, num_loc(p.n, p.bA, p.c, 0, p.pc),
            p.bA, p.r, p.c, p.pr, p.pc, p.uplo == 'U', mode, exponent, r.device_stats.data());
        GPU_CHECK(gpuGetLastError());
        if (mode != 2)
            fetch<B>(h, r, 2 * scan_blocks);
#endif
    }
    if (mode == 2)
        return {0, 0};
    double values[2] = {0, 0};
    for (int i = 0; i < scan_blocks; ++i)
    {
        values[0] = std::max(values[0], r.stats[2 * i]);
        values[1] = std::max(values[1], r.stats[2 * i + 1]);
    }
    maximum(h, values, 2);
    return {values[0], int(values[1])};
}
template <DdlaBackend B, class T>
void columns(const DdlaHandle_t &h, const Plan &p, Resources<T> &r, const T *x, const T *ax,
             int mode)
{
    const int count = (mode == 2 ? 4 : 1) * p.q;
    std::fill_n(r.stats.data(), count, 0.);
    if constexpr (B == DdlaBackend::CPU)
    {
        detail::host_parallel_for(
            p.columns, 1, p.threads,
            [&](long long lo, long long hi)
            {
                for (int j = int(lo); j < int(hi); ++j)
                {
                    const int gj = global_index(j, p.bq, p.c, p.pc);
                    double v[4] = {0, 0, 0, 0};
                    for (int i = 0; i < p.rows; ++i)
                    {
                        const auto at = i + std::size_t(j) * p.basis[DDLA_LLD_];
                        const double xr = re(x[at]), xi = im(x[at]);
                        if (mode == 0)
                            v[0] = std::max(v[0], std::isfinite(xr) && std::isfinite(xi)
                                                      ? std::max(std::abs(xr), std::abs(xi))
                                                      : std::numeric_limits<double>::infinity());
                        else if (mode == 1)
                        {
                            const double scale = r.parameters[gj];
                            if (scale > 0)
                            {
                                const double ar = xr / scale, ai = xi / scale;
                                v[0] += ar * ar + ai * ai;
                            }
                        }
                        else
                        {
                            const double ar = re(ax[at]), ai = im(ax[at]),
                                         rounded = r.parameters[gj], raw = r.parameters[p.q + gj];
                            v[0] += xr * xr + xi * xi;
                            v[1] += (ar - rounded * xr) * (ar - rounded * xr) +
                                    (ai - rounded * xi) * (ai - rounded * xi);
                            v[2] += (ar - raw * xr) * (ar - raw * xr) +
                                    (ai - raw * xi) * (ai - raw * xi);
                            v[3] += !std::isfinite(xr) || !std::isfinite(xi) ||
                                    !std::isfinite(ar) || !std::isfinite(ai);
                        }
                    }
                    for (int k = 0; k < (mode == 2 ? 4 : 1); ++k)
                        r.stats[k * p.q + gj] = v[k];
                }
            });
    }
    else
    {
#if DDLA_HAS_GPU
        DDLA_CHECK(ddlaMemset(h, r.device_stats.data(), 0, std::size_t(count) * sizeof(double)));
        if (p.columns)
        {
            using D = detail::DeviceScalarT<T>;
            column_statistics<D>
                <<<p.columns, scan_threads, 0, static_cast<gpuStream_t>(h->stream)>>>(
                    reinterpret_cast<const D *>(x), reinterpret_cast<const D *>(ax),
                    p.basis[DDLA_LLD_], p.rows, p.columns, p.q, p.bq, p.pc, p.c, mode,
                    r.device_parameters.data(), r.device_stats.data());
            GPU_CHECK(gpuGetLastError());
        }
        fetch<B>(h, r, count);
#endif
    }
    if (mode == 0)
        maximum(h, r.stats.data(), count);
    else
        sum(h, r.stats.data(), count);
}
template <DdlaBackend B, class T>
void transform(const DdlaHandle_t &h, const Plan &p, Resources<T> &r, T *x, const T *random,
               int mode, double delta = 0)
{
    if constexpr (B == DdlaBackend::CPU)
    {
        detail::host_parallel_for(
            p.columns, 1, p.threads,
            [&](long long lo, long long hi)
            {
                for (int j = int(lo); j < int(hi); ++j)
                {
                    const int gj = global_index(j, p.bq, p.c, p.pc);
                    for (int i = 0; i < p.rows; ++i)
                    {
                        const auto at = i + std::size_t(j) * p.basis[DDLA_LLD_];
                        if (!mode)
                            x[at] = T(global_index(i, p.bA, p.r, p.pr) == gj ? 1 : 0);
                        else if (mode == 1)
                            x[at] = number<T>(
                                (re(x[at]) / r.parameters[gj]) / std::sqrt(r.parameters[p.q + gj]),
                                (im(x[at]) / r.parameters[gj]) / std::sqrt(r.parameters[p.q + gj]));
                        else if (mode == 2)
                        {
                            if (gj < p.nev)
                                x[at] += T(delta) * random[at];
                        }
                        else if (r.parameters[gj])
                            x[at] = random[at];
                    }
                }
            });
    }
    else
    {
#if DDLA_HAS_GPU
        using D = detail::DeviceScalarT<T>;
        transform_columns<D><<<scan_blocks, scan_threads, 0, static_cast<gpuStream_t>(h->stream)>>>(
            reinterpret_cast<D *>(x), reinterpret_cast<const D *>(random), p.basis[DDLA_LLD_],
            p.rows, p.columns, p.n, p.q, p.bA, p.bq, p.r, p.c, p.pr, p.pc, mode, p.nev, delta,
            r.device_parameters.data());
        GPU_CHECK(gpuGetLastError());
#endif
    }
}
template <DdlaBackend B, class T>
bool normalize(const DdlaHandle_t &h, const Plan &p, Resources<T> &r, T *x)
{
    columns<B>(h, p, r, x, static_cast<const T *>(nullptr), 0);
    bool valid = true;
    for (int j = 0; j < p.q; ++j)
    {
        r.parameters[j] = r.stats[j];
        valid = valid && r.stats[j] > 0 && std::isfinite(r.stats[j]);
    }
    if (!valid)
        return false;
    parameters<B>(h, r, p.q);
    columns<B>(h, p, r, x, static_cast<const T *>(nullptr), 1);
    for (int j = 0; j < p.q; ++j)
    {
        r.parameters[p.q + j] = r.stats[j];
        valid = valid && r.stats[j] > 0 && std::isfinite(r.stats[j]);
    }
    if (!valid)
        return false;
    parameters<B>(h, r, 2 * p.q);
    transform<B>(h, p, r, x, static_cast<const T *>(nullptr), 1);
    return true;
}

template <DdlaBackend B, class T>
void diagonal(const DdlaHandle_t &h, const Plan &p, Resources<T> &r, const T *x)
{
    std::fill_n(r.stats.data(), p.q, 0.);
    if constexpr (B == DdlaBackend::CPU)
    {
        for (int j = 0; j < p.columns; ++j)
        {
            const int gj = global_index(j, p.bq, p.c, p.pc);
            if ((gj / p.bA) % p.pr != p.r)
                continue;
            const int i = (gj / p.bA / p.pr) * p.bA + gj % p.bA;
            if (i < p.rows)
                r.stats[gj] = std::hypot(re(x[i + std::size_t(j) * p.basis[DDLA_LLD_]]),
                                         im(x[i + std::size_t(j) * p.basis[DDLA_LLD_]]));
        }
    }
    else
    {
#if DDLA_HAS_GPU
        DDLA_CHECK(ddlaMemset(h, r.device_stats.data(), 0, std::size_t(p.q) * sizeof(double)));
        using D = detail::DeviceScalarT<T>;
        qr_diagonal<D><<<scan_blocks, scan_threads, 0, static_cast<gpuStream_t>(h->stream)>>>(
            reinterpret_cast<const D *>(x), p.basis[DDLA_LLD_], p.rows, p.columns, p.q, p.bA, p.bq,
            p.r, p.c, p.pr, p.pc, r.device_stats.data());
        GPU_CHECK(gpuGetLastError());
        fetch<B>(h, r, p.q);
#endif
    }
    maximum(h, r.stats.data(), p.q);
}
template <DdlaBackend B, class T>
bool orthogonalize(const DdlaHandle_t &h, const Plan &p, Resources<T> &r, T *original,
                   T *reflectors, T *output, void *backend, std::size_t bb, void *host,
                   std::size_t hb, Counters &count)
{
    using R = typename real_type<T>::type;
    const double threshold =
        64 * std::numeric_limits<R>::epsilon() * std::max(1., std::sqrt(double(p.n)));
    for (int retry = 0; retry < 4; ++retry)
    {
        columns<B>(h, p, r, original, static_cast<const T *>(nullptr), 0);
        bool deficient = false;
        for (int j = 0; j < p.q; ++j)
        {
            const bool bad = !(r.stats[j] > 0) || !std::isfinite(r.stats[j]);
            r.parameters[j] = bad ? 1 : 0;
            deficient = deficient || bad;
        }
        if (deficient)
        {
            random_impl::random_generate_run<B, T>(h, *r.rng, output,
                                                   std::int64_t(p.basis_elements));
            parameters<B>(h, r, p.q);
            transform<B>(h, p, r, original, output, 3);
            ++count.qr_retries;
        }
        if (!normalize<B>(h, p, r, original))
            return false;
        local_copy<B>(h, p, original, reflectors);
        qr_impl::pgeqrf_run<B, T>(h, 'U', p.qr_inplace, p.n, p.q, reflectors, p.basis.data(),
                                  r.local_tau.data(), backend, bb, host, hb);
        diagonal<B>(h, p, r, reflectors);
        deficient = false;
        for (int j = 0; j < p.q; ++j)
        {
            const bool bad = !(r.stats[j] > threshold) || !std::isfinite(r.stats[j]);
            r.parameters[j] = bad ? 1 : 0;
            deficient = deficient || bad;
        }
        if (deficient)
        {
            random_impl::random_generate_run<B, T>(h, *r.rng, output,
                                                   std::int64_t(p.basis_elements));
            parameters<B>(h, r, p.q);
            transform<B>(h, p, r, original, output, 3);
            ++count.qr_retries;
            continue;
        }
        std::fill(r.tau.begin(), r.tau.end(), T(0));
        for (int j = 0; j < p.columns; ++j)
            r.tau[global_index(j, p.bq, p.c, p.pc)] = r.local_tau[j];
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, r.tau.data(), p.q, detail::mpi_datatype<T>(), MPI_SUM,
                                h->row_comm));
        transform<B>(h, p, r, output, static_cast<const T *>(nullptr), 0);
        int info = 0;
        punmbr_run<B, T>(h, p.apply, reflectors, p.basis.data(), r.tau.data(), output,
                         p.basis.data(), info, backend, bb, host, hb);
        return info == 0;
    }
    return false;
}
template <DdlaBackend B, class T>
void action(const DdlaHandle_t &h, const Plan &p, T *a, const T *x, T *y, void *backend,
            std::size_t bb, void *host, std::size_t hb, Counters &count)
{
    pgemm_impl::pgemm_run<B, T>(h, p.action, p.action_width, 'N', 'N', p.n, p.q, p.n, T(1), a,
                                p.a.data(), x, p.basis.data(), T(0), y, p.basis.data(), backend, bb,
                                host, hb);
    ++count.operator_applications;
}
template <DdlaBackend B, class T>
void combine(const DdlaHandle_t &h, const Plan &p, T *destination, double alpha, const T *x,
             double beta, const T *y)
{
    if (p.rows && p.columns)
        blas_impl::geam_run<B, T>(h, 'N', 'N', p.rows, p.columns, T(alpha), x, p.basis[DDLA_LLD_],
                                  T(beta), y, p.basis[DDLA_LLD_], destination, p.basis[DDLA_LLD_]);
}
template <DdlaBackend B, class T>
void complete(const DdlaHandle_t &h, const Plan &p, T *a, void *backend, std::size_t bb, void *host,
              std::size_t hb)
{
    for (int k = 0; k < p.n;)
    {
        const int width = std::min({p.q, p.bA - k % p.bA, p.n - k});
        if ((k / p.bA) % p.pr == p.r && (k / p.bA) % p.pc == p.c)
        {
            const int i = num_loc(k, p.bA, p.r, 0, p.pr), j = num_loc(k, p.bA, p.c, 0, p.pc);
            ddla::herm_complete<B, T>(h, p.uplo, width, a + i + std::size_t(j) * p.a[DDLA_LLD_],
                                      p.a[DDLA_LLD_]);
        }
        const int rest = p.n - k - width;
        if (rest)
        {
            const int m = p.uplo == 'U' ? rest : width, n = p.uplo == 'U' ? width : rest;
            const int ia = p.uplo == 'U' ? k + 1 : k + width + 1,
                      ja = p.uplo == 'U' ? k + width + 1 : k + 1;
            const int ic = p.uplo == 'U' ? k + width + 1 : k + 1,
                      jc = p.uplo == 'U' ? k + 1 : k + width + 1;
            pgeadd_impl::pgeadd_run<B, T>(h, 'C', 'N', m, n, T(1), a, ia, ja, p.a.data(), T(0),
                                          nullptr, 1, 1, p.a.data(), a, ic, jc, p.a.data(), backend,
                                          bb, host, hb);
        }
        k += width;
    }
}
template <DdlaBackend B, class T>
std::array<double, 3> bounds(const DdlaHandle_t &h, const Plan &p, Resources<T> &r, const T *a)
{
    const int nc = num_loc(p.n, p.bA, p.c, 0, p.pc);
    if constexpr (B == DdlaBackend::CPU)
    {
        detail::host_parallel_for(p.rows, 64, p.threads,
                                  [&](long long lo, long long hi)
                                  {
                                      for (int i = int(lo); i < int(hi); ++i)
                                      {
                                          const int gi = global_index(i, p.bA, p.r, p.pr);
                                          double total = 0, diag = 0;
                                          for (int j = 0; j < nc; ++j)
                                          {
                                              const T x = a[i + std::size_t(j) * p.a[DDLA_LLD_]];
                                              total += std::hypot(re(x), im(x));
                                              if (gi == global_index(j, p.bA, p.c, p.pc))
                                                  diag = re(x);
                                          }
                                          r.stats[i] = total;
                                          r.stats[p.rows + i] = diag;
                                      }
                                  });
    }
    else
    {
#if DDLA_HAS_GPU
        using D = detail::DeviceScalarT<T>;
        row_statistics<D><<<scan_blocks, scan_threads, 0, static_cast<gpuStream_t>(h->stream)>>>(
            reinterpret_cast<const D *>(a), p.a[DDLA_LLD_], p.rows, nc, p.bA, p.r, p.c, p.pr, p.pc,
            r.device_stats.data());
        GPU_CHECK(gpuGetLastError());
        fetch<B>(h, r, 2 * p.rows);
#endif
    }
    MPI_CHECK(
        MPI_Allreduce(MPI_IN_PLACE, r.stats.data(), 2 * p.rows, MPI_DOUBLE, MPI_SUM, h->row_comm));
    double low = std::numeric_limits<double>::infinity(), high = -low, norm = 0;
    for (int i = 0; i < p.rows; ++i)
    {
        const double total = r.stats[i], diag = r.stats[p.rows + i],
                     off = std::max(0., total - std::abs(diag));
        low = std::min(low, diag - off);
        high = std::max(high, diag + off);
        norm = std::max(norm, total);
    }
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &low, 1, MPI_DOUBLE, MPI_MIN, h->comm));
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &high, 1, MPI_DOUBLE, MPI_MAX, h->comm));
    maximum(h, &norm, 1);
    return {low, high, norm};
}
} // namespace

template <class T>
Plan check(const DdlaHandle_t &h, DdlaBackend backend, char uplo, int n, int nev, int nex,
           const T *a, const int *da, const typename real_type<T>::type *w, const T *z,
           const int *dz, double tolerance, int max_iterations, int degree, bool warm)
{
    auto p = detail::agree_check(
        h, "chebyshev",
        [&](detail::Agree &) -> Plan
        {
            detail::require_backend(h, backend, "chebyshev");
            uplo = static_cast<char>(std::toupper(static_cast<unsigned char>(uplo)));
            if (n < 0 || n > INT_MAX / 64 || nev < 0 || nev > n || nex < 0 || nex > n - nev ||
                (uplo != 'U' && uplo != 'L') || !std::isfinite(tolerance) || tolerance <= 0 ||
                max_iterations < 0 || degree < 1)
                throw std::invalid_argument("chebyshev: invalid dimensions/options");
            check_desc(da, h);
            if (da[DDLA_M_] < n || da[DDLA_N_] < n || da[DDLA_MB_] != da[DDLA_NB_] ||
                da[DDLA_RSRC_] != 0 || da[DDLA_CSRC_] != 0)
                throw std::invalid_argument(
                    "chebyshev: A needs square source-zero blocks and n-by-n extent");
            Plan q;
            q.backend = backend;
            q.transport = ddlaGetCommunication(h);
            q.compute = h->stream;
            q.data = h->stream_data;
            q.major = h->major;
            q.device = h->local_device;
            q.communicator = h->comm;
            q.n = n;
            q.nev = nev;
            q.q = nev ? nev + nex : 0;
            q.degree = degree;
            q.max_iterations = max_iterations;
            q.warm = warm;
            q.tolerance = tolerance;
            q.uplo = uplo;
            q.pr = h->nprows_;
            q.pc = h->npcols_;
            q.r = h->myprow_;
            q.c = h->mypcol_;
            q.threads = h->host_threads;
            q.bA = da[DDLA_MB_];
            std::copy(da, da + DDLA_DLEN_, q.a.begin());
            q.rows = num_loc(n, q.bA, q.r, 0, q.pr);
            const int ac = num_loc(n, q.bA, q.c, 0, q.pc);
            if (nev)
            {
                if (!w || (q.rows && ac && !a))
                    throw std::invalid_argument("chebyshev: missing input/eigenvalues");
                check_desc(dz, h);
                if (dz[DDLA_M_] < n || dz[DDLA_N_] < nev)
                    throw std::invalid_argument("chebyshev: eigenvector extent");
                const int zr = num_loc(n, dz[DDLA_MB_], q.r, dz[DDLA_RSRC_], q.pr),
                          zc = num_loc(nev, dz[DDLA_NB_], q.c, dz[DDLA_CSRC_], q.pc);
                if (zr && zc && !z)
                    throw std::invalid_argument("chebyshev: missing active vectors");
                if (detail::matrix_overlaps(w, std::max(1, nev), nev, 1, z, dz[DDLA_LLD_], zr, zc))
                    throw std::invalid_argument("chebyshev: w/Z outputs overlap");
                std::copy(dz, dz + DDLA_DLEN_, q.z.begin());
            }
            q.bq = std::max(1, std::min(q.bA, q.q / std::max(1, h->nprocs)));
            q.columns = num_loc(q.q, q.bq, q.c, 0, q.pc);
            q.small_rows = num_loc(q.q, q.bq, q.r, 0, q.pr);
            q.small_columns = q.columns;
            DDLA_CHECK(
                ddlaDescInit(q.basis.data(), h, n, q.q, q.bA, q.bq, 0, 0, std::max(1, q.rows)));
            DDLA_CHECK(ddlaDescInit(q.projected.data(), h, q.q, q.q, q.bq, q.bq, 0, 0,
                                    std::max(1, q.small_rows)));
            q.basis_elements = std::size_t(q.basis[DDLA_LLD_]) * q.columns;
            q.projected_elements = std::size_t(q.projected[DDLA_LLD_]) * q.small_columns;
            q.stats_elements = std::max(
                {std::size_t(2 * scan_blocks), std::size_t(2) * q.rows, std::size_t(4) * q.q});
            bytes(q.basis_elements, sizeof(T));
            bytes(q.projected_elements, sizeof(T));
            return q;
        });
    int signature[29] = {int(p.backend), p.n,
                         p.nev,          nex,
                         p.degree,       p.max_iterations,
                         int(p.warm),    int(p.uplo),
                         int(sizeof(T)), int(std::is_arithmetic<T>::value)};
    std::memcpy(signature + 10, &p.tolerance, sizeof(p.tolerance));
    detail::descriptor_signature(signature + 12, da);
    if (nev)
        detail::descriptor_signature(signature + 19, dz);
    detail::agree_signature(h, "chebyshev", signature);
    return p;
}

template <class T>
Resources<T>::Resources(const DdlaHandle_t &handle, const Plan &p, std::uint64_t seed)
    : basis(handle), h(handle), s(handle), device_stats(handle), device_parameters(handle)
{
    if (!p.nev)
        return;
    basis.resize(p.basis_elements);
    h.resize(p.projected_elements);
    s.resize(p.projected_elements);
    if (p.backend == DdlaBackend::GPU)
    {
        device_stats.resize(p.stats_elements);
        device_parameters.resize(std::size_t(2) * p.q);
    }
    stats.resize(p.stats_elements);
    parameters.resize(std::size_t(2) * p.q);
    tau.resize(p.q);
    local_tau.resize(std::max(1, p.columns));
    values.resize(p.q);
    rng.emplace(handle, seed + static_cast<std::uint64_t>(handle->myid));
}
template <class T> std::size_t Resources<T>::retained_backend_bytes() const noexcept
{
    return (basis.size() + h.size() + s.size()) * sizeof(T) +
           (device_stats.size() + device_parameters.size()) * sizeof(double) +
           (eigen ? eigen->retained_backend_bytes() : 0);
}
template <class T> std::size_t Resources<T>::retained_host_bytes() const noexcept
{
    return (stats.size() + parameters.size()) * sizeof(double) +
           (tau.size() + local_tau.size()) * sizeof(T) + values.size() * sizeof(R);
}
template <DdlaBackend B, class T> void prepare(const DdlaHandle_t &h, Plan &p, Resources<T> &r)
{
    p.ready = false;
    if (!p.nev)
    {
        p.ready = true;
        return;
    }
    p.qr_inplace = qr_impl::pgeqrf_check<B>(h, 'U', p.n, p.q, p.basis.data(), r.local_tau.data());
    p.action =
        pgemm_impl::pgemm_check<B>(h, 'N', 'N', p.n, p.q, p.n, false, p.a.data(), p.basis.data(),
                                   p.basis.data(), pgemm_impl::Algorithm::BC, p.action_width);
    p.project = pgemm_impl::pgemm_check<B>(h, 'C', 'N', p.q, p.q, p.n, false, p.basis.data(),
                                           p.basis.data(), p.projected.data(),
                                           pgemm_impl::Algorithm::AB, p.project_width);
    p.rotate = pgemm_impl::pgemm_check<B>(h, 'N', 'N', p.n, p.q, p.q, false, p.basis.data(),
                                          p.projected.data(), p.basis.data(),
                                          pgemm_impl::Algorithm::AB, p.rotate_width);
    p.eigen = pheevd_driver_check<B, T>(
        h, h->nprocs == 1 ? detail::PheevdRoute::Local : detail::PheevdRoute::OneStage, 'V', 'L',
        p.q, p.q, r.h.data(), p.projected.data(), r.values.data(), r.s.data(), p.projected.data(),
        nullptr);
    if (p.eigen.one)
    {
        // PSTEDC derives leaf sizes, workers, resource events and scratch from
        // this field on every query/run. Cap BEFORE constructing its resources;
        // keep the independent strip/root-chunk policies unchanged and counted.
        auto &v = p.eigen.one->vectors;
        int share = 1;
        while (static_cast<long long>(share) * share < h->nprocs)
            ++share;
        // Halving a non-power-of-two problem down to singleton leaves would
        // create empty leaves (and an invalid zero LDZ in the shared leaf solve).
        // A 2-by-2 leaf floor is constant rounding, not a root q-by-q buffer.
        const int minimum_leaf = (p.q & (p.q - 1)) ? 2 : 1;
        v.leaf = std::min(std::max(v.leaf, minimum_leaf), std::max(minimum_leaf, p.q / share));
    }
    detail::agree_local(h, "chebyshev projected resources", [&] { r.eigen.emplace(h, p.eigen); });
    detail::agree_local(
        h, "chebyshev native queries",
        [&]
        {
            qr_impl::pgeqrf_workspace_size<T>(h, 'U', p.qr_inplace, p.n, p.q, r.basis.data(),
                                              p.basis.data(), r.local_tau.data(), &p.qr_backend,
                                              &p.qr_host);
            include(p.qr_backend, p.qr_host, p);
            p.apply =
                punmbr_geometry(h, B, 'Q', 'L', 'N', p.n, p.q, p.q, p.basis.data(), p.basis.data());
            std::size_t b = 0, cpu = 0;
            punmbr_workspace_size<T>(h, p.apply, &b, &cpu);
            include(b, cpu, p);
            p.eigen_work =
                pheevd_driver_workspace_size<T>(h, p.eigen, *r.eigen, r.h.data(), r.values.data());
            include(p.eigen_work.backend_bytes, p.eigen_work.host_bytes, p);
            gemm_bytes<T>(h, p, p.action, p.action_width, 'N', 'N', p.n, p.q, p.n, p.a.data(),
                          p.basis.data(), p.basis.data());
            gemm_bytes<T>(h, p, p.project, p.project_width, 'C', 'N', p.q, p.q, p.n, p.basis.data(),
                          p.basis.data(), p.projected.data());
            gemm_bytes<T>(h, p, p.rotate, p.rotate_width, 'N', 'N', p.n, p.q, p.q, p.basis.data(),
                          p.projected.data(), p.basis.data());
            add_bytes<T>(h, p, 'N', p.n, p.nev, p.z.data(), p.basis.data(), p);
            add_bytes<T>(h, p, 'N', p.n, p.nev, p.basis.data(), p.z.data(), p);
            for (int k = 0; k < p.n;)
            {
                const int width = std::min({p.q, p.bA - k % p.bA, p.n - k}), rest = p.n - k - width;
                if (rest)
                    add_bytes<T>(h, p, 'C', p.uplo == 'U' ? rest : width,
                                 p.uplo == 'U' ? width : rest, p.a.data(), p.a.data(), p);
                k += width;
            }
            const std::size_t prefix = 2 * bytes(p.basis_elements, sizeof(T));
            if (p.phase_backend > std::numeric_limits<std::size_t>::max() - prefix)
                throw std::overflow_error("chebyshev: workspace sum overflow");
            p.backend_bytes = prefix + p.phase_backend;
            p.host_bytes = p.phase_host;
        });
    p.ready = true;
}

template<class T>std::array<std::size_t,2> staging_bytes(const DdlaHandle_t& h,const Plan& p)
{
    stale(h,p);
    std::array<std::size_t,2> high{};
    if(p.backend!=DdlaBackend::GPU || h->communication()!=DdlaCommunication::HostStagedMPI || !p.nev)return high;
    auto copy=[&](char op,int m,int n,const int* a,int ia,int ja,const int* c,int ic,int jc) {
        pblas::staging_max(high,pgeadd_impl::pgeadd_staging_bytes(h,op,'N',true,false,m,n,
            a,ia,ja,c,1,1,c,ic,jc,sizeof(T)));
    };
    if(p.warm)copy('N',p.n,p.nev,p.z.data(),1,1,p.basis.data(),1,1);
    copy('N',p.n,p.nev,p.basis.data(),1,1,p.z.data(),1,1);
    for(int k=0;k<p.n;) {
        const int width=std::min({p.q,p.bA-k%p.bA,p.n-k}),rest=p.n-k-width;
        if(rest)copy('C',p.uplo=='U'?rest:width,p.uplo=='U'?width:rest,p.a.data(),
            p.uplo=='U'?k+1:k+width+1,p.uplo=='U'?k+width+1:k+1,p.a.data(),
            p.uplo=='U'?k+width+1:k+1,p.uplo=='U'?k+1:k+width+1);
        k+=width;
    }
    pblas::staging_max(high,pgemm_impl::pgemm_staging_bytes(h,p.action,true,'N','N',p.n,p.q,p.n,false,
        p.a.data(),p.basis.data(),p.basis.data(),sizeof(T),sizeof(T),sizeof(T)));
    pblas::staging_max(high,pgemm_impl::pgemm_staging_bytes(h,p.project,true,'C','N',p.q,p.q,p.n,false,
        p.basis.data(),p.basis.data(),p.projected.data(),sizeof(T),sizeof(T),sizeof(T)));
    pblas::staging_max(high,pgemm_impl::pgemm_staging_bytes(h,p.rotate,true,'N','N',p.n,p.q,p.q,false,
        p.basis.data(),p.projected.data(),p.basis.data(),sizeof(T),sizeof(T),sizeof(T)));
    pblas::staging_max(high,qr_impl::pgeqrf_staging_bytes<T>(h,'U',p.qr_inplace,p.n,p.q,p.basis.data()));
    pblas::staging_max(high,punmbr_staging_bytes<T>(h,p.apply));
    if(p.eigen.one)pblas::staging_max(high,one_stage_staging_bytes<T>(h,*p.eigen.one));
    return high;
}
#define STAGING_SIZE(T) template std::array<std::size_t,2> staging_bytes<T>(const DdlaHandle_t&,const Plan&)
DDLA_FOR_EACH_TYPE(STAGING_SIZE)
#undef STAGING_SIZE

template <DdlaBackend B, class T>
void run(const DdlaHandle_t &h, const Plan &p, Resources<T> &r, T *a,
         typename real_type<T>::type *w, T *z, int &iterations, int &converged, int &info,
         Counters *counters, void *backend, std::size_t backend_bytes, void *host,
         std::size_t host_bytes)
{
    using R = typename real_type<T>::type;
    detail::agree_local(h, "chebyshev prepared execution",
                        [&]
                        {
                            stale(h, p);
                            if (backend_bytes < p.backend_bytes || host_bytes < p.host_bytes ||
                                (p.backend_bytes && !backend) || (p.host_bytes && !host))
                                throw std::invalid_argument(
                                    "chebyshev: insufficient caller workspace");
                        });
    iterations = 0;
    converged = 0;
    info = success;
    Counters count;
    if (counters)
        *counters = count;
    if (!p.nev)
        return;

    char *cursor = static_cast<char *>(backend);
    T *basis[3] = {r.basis.data(), detail::take<T>(cursor, p.basis_elements),
                   detail::take<T>(cursor, p.basis_elements)};
    void *phase = cursor;
    const std::size_t phase_bytes = backend_bytes - 2 * bytes(p.basis_elements, sizeof(T));
    int current = 0;
    auto finish = [&]
    {
        // Borrowed arenas and PGEADD metadata must outlive queued work.
        DDLA_CHECK(ddlaSynchronize(h));
        if (counters)
            *counters = count;
    };

    // Validate and prove that componentwise binary scaling is reversible before
    // changing A. Unused triangle entries and imaginary diagonals are ignored.
    const auto input = scan<B>(h, p, r, a, 0, 0);
    if (input.second)
    {
        info = nonfinite;
        finish();
        return;
    }
    int exponent = 0;
    if (input.first)
        std::frexp(input.first, &exponent);
    if (scan<B>(h, p, r, a, 1, exponent).second)
    {
        info = scaling_loss;
        finish();
        return;
    }

    // Warm data can alias A. Read it before scaling or completing A, and check
    // every warm column before a zero/rank-deficient column may be replenished.
    random_impl::random_generate_run<B, T>(h, *r.rng, basis[0], std::int64_t(p.basis_elements));
    if (p.warm)
    {
        copy<B>(h, p, z, p.z.data(), basis[0], p.basis.data(), p.n, p.nev, phase, phase_bytes, host,
                host_bytes);
        columns<B>(h, p, r, basis[0], static_cast<const T *>(nullptr), 0);
        bool finite = true, empty = false;
        for (int j = 0; j < p.q; ++j)
        {
            finite = finite && std::isfinite(r.stats[j]);
            r.parameters[j] = r.stats[j] > 0 ? 0 : 1;
            empty = empty || r.parameters[j] != 0;
        }
        if (!finite)
        {
            info = nonfinite;
            finish();
            return;
        }
        if (empty)
        {
            random_impl::random_generate_run<B, T>(h, *r.rng, basis[1],
                                                   std::int64_t(p.basis_elements));
            parameters<B>(h, r, p.q);
            transform<B>(h, p, r, basis[0], basis[1], 3);
            ++count.qr_retries;
        }
        if (!normalize<B>(h, p, r, basis[0]))
        {
            info = rank_breakdown;
            finish();
            return;
        }
        random_impl::random_generate_run<B, T>(h, *r.rng, basis[1], std::int64_t(p.basis_elements));
        if (!normalize<B>(h, p, r, basis[1]))
        {
            info = rank_breakdown;
            finish();
            return;
        }
        const double perturbation =
            std::max(64. * std::numeric_limits<R>::epsilon(), std::sqrt(p.tolerance));
        transform<B>(h, p, r, basis[0], basis[1], 2, perturbation);
    }
    scan<B>(h, p, r, a, 2, exponent);
    complete<B>(h, p, a, phase, phase_bytes, host, host_bytes);
    const auto spectral = bounds<B>(h, p, r, a);
    const bool scalar = spectral[0] == spectral[1];
    const double inflation =
        16. * std::numeric_limits<R>::epsilon() * (1. + std::sqrt(double(p.n))) * spectral[2];
    const double lower = spectral[0] - inflation;
    const double upper = spectral[1] + inflation;
    const double matrix_norm = spectral[2];
    count.lower_bound = std::scalbn(lower, exponent);
    count.upper_bound = std::scalbn(upper, exponent);
    count.matrix_norm = std::scalbn(matrix_norm, exponent);
    bool have_ritz = false;
    bool pending_filter = false;

    for (;;)
    {
        // Retain the unmodified candidate until QR succeeds. Refill deficient
        // columns before the first Ritz step; filtering a zero-eigenvalue
        // replenishment immediately could erase the only new directions.
        const int old = current, reflectors = (old + 1) % 3, qslot = (old + 2) % 3;
        if (!orthogonalize<B>(h, p, r, basis[old], basis[reflectors], basis[qslot], phase,
                              phase_bytes, host, host_bytes, count))
        {
            info = rank_breakdown;
            break;
        }
        action<B>(h, p, a, basis[qslot], basis[old], phase, phase_bytes, host, host_bytes, count);
        pgemm_impl::pgemm_run<B, T>(h, p.project, p.project_width, 'C', 'N', p.q, p.q, p.n, T(1),
                                    basis[qslot], p.basis.data(), basis[old], p.basis.data(), T(0),
                                    r.h.data(), p.projected.data(), phase, phase_bytes, host,
                                    host_bytes);
        int projected_info = 0;
        pheevd_driver_run<B, T>(h, p.eigen, *r.eigen, p.eigen_work, r.h.data(), r.values.data(),
                                r.s.data(), projected_info, phase, phase_bytes, host, host_bytes);
        bool valid_ritz = projected_info == 0;
        for (int j = 0; j < p.q; ++j)
            valid_ritz = valid_ritz && std::isfinite(r.values[j]) &&
                         (j == 0 || r.values[j - 1] <= r.values[j]);
        if (!valid_ritz)
        {
            info = projected_failure;
            break;
        }
        pgemm_impl::pgemm_run<B, T>(h, p.rotate, p.rotate_width, 'N', 'N', p.n, p.q, p.q, T(1),
                                    basis[qslot], p.basis.data(), r.s.data(), p.projected.data(),
                                    T(0), basis[reflectors], p.basis.data(), phase, phase_bytes,
                                    host, host_bytes);
        pgemm_impl::pgemm_run<B, T>(h, p.rotate, p.rotate_width, 'N', 'N', p.n, p.q, p.q, T(1),
                                    basis[old], p.basis.data(), r.s.data(), p.projected.data(),
                                    T(0), basis[qslot], p.basis.data(), phase, phase_bytes, host,
                                    host_bytes);
        current = reflectors;
        have_ritz = true;
        if (pending_filter)
        {
            ++iterations;
            pending_filter = false;
        }

        // Certify the returned, rescaled R value, including its rounding. The
        // raw Ritz residual is used only to distinguish representability failure
        // from unfinished iteration; it never relaxes the public criterion.
        bool represented = true, all_raw = true;
        for (int j = 0; j < p.q; ++j)
        {
            const R value = std::scalbn(r.values[j], exponent);
            const bool valid = std::isfinite(value);
            if (j < p.nev)
                represented = represented && valid;
            r.parameters[j] = valid ? double(std::scalbn(value, -exponent)) : double(r.values[j]);
            r.parameters[p.q + j] = double(r.values[j]);
        }
        parameters<B>(h, r, 2 * p.q);
        columns<B>(h, p, r, basis[current], basis[qslot], 2);
        converged = 0;
        bool quantization_failure = false;
        count.maximum_residual = 0;
        for (int j = 0; j < p.nev; ++j)
        {
            const double xn = std::sqrt(r.stats[j]);
            const double denominator = (matrix_norm + std::abs(r.parameters[j])) * xn;
            const double raw_denominator = (matrix_norm + std::abs(double(r.values[j]))) * xn;
            const double residue = std::sqrt(r.stats[p.q + j]);
            const double raw_residue = std::sqrt(r.stats[2 * p.q + j]);
            const bool finite = r.stats[3 * p.q + j] == 0 && std::isfinite(xn) && xn > 0;
            const double relative =
                finite ? (denominator > 0
                              ? residue / denominator
                              : (residue == 0 ? 0 : std::numeric_limits<double>::infinity()))
                       : std::numeric_limits<double>::infinity();
            const double raw_relative =
                finite ? (raw_denominator > 0
                              ? raw_residue / raw_denominator
                              : (raw_residue == 0 ? 0 : std::numeric_limits<double>::infinity()))
                       : std::numeric_limits<double>::infinity();
            const bool valid_value = std::isfinite(std::scalbn(r.values[j], exponent));
            converged += valid_value && relative <= p.tolerance;
            all_raw = all_raw && raw_relative <= p.tolerance;
            // A reverse-triangle lower bound distinguishes unattainable scalar
            // rounding from an unfinished iteration (notably subnormal w).
            const double shift = std::abs(double(r.values[j]) - r.parameters[j]) * xn;
            const double floor =
                denominator > 0 ? std::max(0., shift - raw_residue) / denominator : 0.;
            quantization_failure = quantization_failure || floor > p.tolerance;
            count.maximum_residual = std::max(count.maximum_residual, relative);
        }
        if ((!represented || quantization_failure) && all_raw)
        {
            info = unrepresentable;
            break;
        }
        const bool may_accept = !p.warm || iterations > 0 || p.q == p.n || scalar;
        if (converged == p.nev && may_accept)
        {
            info = success;
            break;
        }
        if (iterations == p.max_iterations)
        {
            info = iteration_limit;
            break;
        }

        // A saturated cutoff still needs a genuine action. Its normalized
        // shifted-power limit is well-defined unless the GLOBAL interval is
        // collapsed; the latter has already been tested by a Ritz residual.
        const double cutoff = std::max(lower, std::min(upper, double(r.values.back())));
        const double center = (upper + cutoff) / 2, radius = (upper - cutoff) / 2;
        const double span = upper - lower;
        const bool power = radius <= 32. * std::numeric_limits<R>::epsilon() *
                                         std::max(1., std::max(std::abs(upper), std::abs(cutoff)));
        if (!(span > 0) || !std::isfinite(span))
        {
            info = rank_breakdown;
            break;
        }
        const double sigma1 = radius / (lower - center);
        double sigma = sigma1;
        int previous = -1;
        for (int step = 0; step < p.degree; ++step)
        {
            const int destination = previous < 0 ? (current + 1) % 3 : 3 - current - previous;
            action<B>(h, p, a, basis[current], basis[destination], phase, phase_bytes, host,
                      host_bytes, count);
            double alpha, beta, gamma = 0, next_sigma = sigma;
            if (power)
            {
                alpha = -1. / span;
                beta = upper / span;
            }
            else if (!step)
            {
                alpha = 1. / (lower - center);
                beta = -center * alpha;
            }
            else
            {
                next_sigma = 1. / (2. / sigma1 - sigma);
                alpha = 2. * next_sigma / radius;
                beta = -center * alpha;
                gamma = -sigma * next_sigma;
            }
            if (!std::isfinite(alpha) || !std::isfinite(beta) || !std::isfinite(gamma))
            {
                info = rank_breakdown;
                break;
            }
            combine<B>(h, p, basis[destination], alpha, basis[destination], beta, basis[current]);
            if (previous >= 0 && !power)
                combine<B>(h, p, basis[destination], 1., basis[destination], gamma,
                           basis[previous]);
            previous = current;
            current = destination;
            sigma = next_sigma;
        }
        if (info == rank_breakdown)
            break;
        pending_filter = true;
    }
    if (have_ritz && (info == success || info == iteration_limit || info == unrepresentable))
    {
        copy<B>(h, p, basis[current], p.basis.data(), z, p.z.data(), p.n, p.nev, phase, phase_bytes,
                host, host_bytes);
        DDLA_CHECK(ddlaSynchronize(h));
        for (int j = 0; j < p.nev; ++j)
            w[j] = std::scalbn(r.values[j], exponent);
    }
    finish();
}

#define INSTANTIATE_TYPE(T)                                                                        \
    template Plan check<T>(const DdlaHandle_t &, DdlaBackend, char, int, int, int, const T *,      \
                           const int *, const real_type<T>::type *, const T *, const int *,        \
                           double, int, int, bool);                                                \
    template class Resources<T>
DDLA_FOR_EACH_TYPE(INSTANTIATE_TYPE)
#undef INSTANTIATE_TYPE
#define INSTANTIATE(B, T)                                                                          \
    template void prepare<B, T>(const DdlaHandle_t &, Plan &, Resources<T> &);                     \
    template void run<B, T>(const DdlaHandle_t &, const Plan &, Resources<T> &, T *,               \
                            real_type<T>::type *, T *, int &, int &, int &, Counters *, void *,    \
                            std::size_t, void *, std::size_t)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE)
#undef INSTANTIATE
} // namespace chebyshev
} // namespace spectral_impl
} // namespace ddla


namespace ddla
{

template <DdlaBackend B, class T>
void pheevd_chebyshev(const DdlaHandle_t &h, char uplo, int n, int nev, int nex, T *a,
                      const int *da, typename real_type<T>::type *w, T *z, const int *dz,
                      double tolerance, int max_iterations, int degree, bool warm, int &iterations,
                      int &converged, int &info)
{
    using namespace spectral_impl::chebyshev;
    auto p =
        check<T>(h, B, uplo, n, nev, nex, a, da, w, z, dz, tolerance, max_iterations, degree, warm);
    std::optional<Resources<T>> resources;
    detail::agree_local(h, "chebyshev owned resources", [&] { resources.emplace(h, p); });
    prepare<B>(h, p, *resources);
    std::size_t backend_bytes = 0, host_bytes = 0;
    workspace_size(p, &backend_bytes, &host_bytes);
    ddla::vector<char> backend(h);
    std::vector<char> host;
    detail::agree_local(h, "chebyshev owned arena",
                        [&]
                        {
                            backend.resize(backend_bytes);
                            host.resize(host_bytes);
                            const auto slots=staging_bytes<T>(h,p);
                            if(slots[0]||slots[1])h->host_buffers<std::byte>(slots[0],slots[1]);
                        });
    try
    {
        run<B>(h, p, *resources, a, w, z, iterations, converged, info, nullptr, backend.data(),
               backend.size(), host.data(), host.size());
    }
    catch (...)
    {
        // The vendor may have queued work before reporting a local error.
        // Preserve owners until the compute queue has retired that work.
        ddlaSynchronize(h);
        throw;
    }
}

#define INSTANTIATE_PUBLIC(B, T)                                                                   \
    template void pheevd_chebyshev<B, T>(const DdlaHandle_t &, char, int, int, int, T *,           \
                                         const int *, real_type<T>::type *, T *, const int *,      \
                                         double, int, int, bool, int &, int &, int &)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_PUBLIC)
#undef INSTANTIATE_PUBLIC
} // namespace ddla
