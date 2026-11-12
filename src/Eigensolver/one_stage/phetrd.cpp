#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "ddla_connector.h"
#include "phetrd_impl.h"
#include "BLAS/blas_impl.h"
#include "check_agree.h"
#include "plan_util.h"
#include "Eigensolver/pvector_redist.h"
#include "helpers.h"
#include "instantiate.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <climits>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ddla {
namespace detail {
namespace {
template <class T> auto re(T x) {
    return std::real(x);
}

#if DDLA_HAS_GPU
template <class T> __device__ T dr(T x) {
    return x;
}

template <class R> __device__ R dr(thrust::complex<R> x) {
    return x.real();
}

template <class T> __device__ T di(T) {
    return T(0);
}

template <class R> __device__ R di(thrust::complex<R> x) {
    return x.imag();
}

// The column is short on each rank. One block copies it and forms a robust
// scaled norm plus its endpoint, replacing several BLAS launches and scalar
// transfers. Two passes avoid both squaring large entries and losing small
// entries before scaling. All lanes participate even on empty local slices.
template <class T, class R>
__global__ void norm_pack_kernel(const T* column, int active, int tail, int lr, bool owns_endpoint, T* v,
                                 R* packet, int inc, bool conjugate) {
    __shared__ R scratch[256];
    __shared__ int invalid[256];
    const int lane = threadIdx.x;
    R scale = 0;
    int bad = 0;
    for (int i = lane; i <= lr; i += 256) {
        const T value = i < active ? (conjugate ? device_conj(column[static_cast<std::size_t>(i) * inc])
                                                : column[static_cast<std::size_t>(i) * inc])
                                   : T(0);
        v[i] = value;
        if (i < tail) {
            const R ar = fabs(dr(value)), ai = fabs(di(value));
            bad |= (!isfinite(ar) || !isfinite(ai));
            scale = fmax(scale, fmax(ar, ai));
        }
    }
    scratch[lane] = scale;
    invalid[lane] = bad;
    __syncthreads();
    for (int step = 128; step; step /= 2) {
        if (lane < step) {
            scratch[lane] = fmax(scratch[lane], scratch[lane + step]);
            invalid[lane] |= invalid[lane + step];
        }
        __syncthreads();
    }
    scale = scratch[0];
    bad = invalid[0];
    // Preserve scratch[0] until every lane has loaded the common scale.
    __syncthreads();
    R sum = 0;
    if (scale > 0 && !bad)
        for (int i = lane; i < tail; i += 256) {
            const R ar = dr(v[i]) / scale, ai = di(v[i]) / scale;
            sum += ar * ar + ai * ai;
        }
    scratch[lane] = sum;
    __syncthreads();
    for (int step = 128; step; step /= 2) {
        if (lane < step)
            scratch[lane] += scratch[lane + step];
        __syncthreads();
    }
    if (lane == 0) {
        packet[0] = bad ? R(INFINITY) : scale * sqrt(scratch[0]);
        const T endpoint = owns_endpoint ? v[tail] : T(0);
        packet[1] = dr(endpoint);
        packet[2] = di(endpoint);
    }
}

// Native CCL coefficients stay on the device. Each packet contains a
// binary exponent, scaled tail sum, and endpoint. Byte AllGather preserves
// even subnormal endpoints before combination at a common binary scale.
template <class T, class R>
__global__ void scaled_pack_kernel(const T* column, int active, int tail, int lr, bool owns_endpoint, T* v,
                                   R* packet, int inc, bool conjugate) {
    __shared__ R scratch[256];
    __shared__ int invalid[256];
    const int lane = threadIdx.x;
    R largest = 0;
    int bad = 0;
    for (int i = lane; i <= lr; i += 256) {
        const T value = i < active ? (conjugate ? device_conj(column[static_cast<std::size_t>(i) * inc])
                                                : column[static_cast<std::size_t>(i) * inc])
                                   : T(0);
        v[i] = value;
        if (i < active) {
            const R ar = fabs(dr(value)), ai = fabs(di(value));
            bad |= (!isfinite(ar) || !isfinite(ai));
            largest = fmax(largest, fmax(ar, ai));
        }
    }
    scratch[lane] = largest;
    invalid[lane] = bad;
    __syncthreads();
    for (int step = 128; step; step /= 2) {
        if (lane < step) {
            scratch[lane] = fmax(scratch[lane], scratch[lane + step]);
            invalid[lane] |= invalid[lane + step];
        }
        __syncthreads();
    }
    largest = scratch[0];
    bad = invalid[0];
    int exponent = -4096;
    if (largest > 0 && !bad)
        (void)frexp(largest, &exponent);
    __syncthreads();
    R sum = 0;
    if (largest > 0 && !bad)
        for (int i = lane; i < tail; i += 256) {
            const R ar = scalbn(dr(v[i]), -exponent), ai = scalbn(di(v[i]), -exponent);
            sum += ar * ar + ai * ai;
        }
    scratch[lane] = sum;
    __syncthreads();
    for (int step = 128; step; step /= 2) {
        if (lane < step)
            scratch[lane] += scratch[lane + step];
        __syncthreads();
    }
    if (lane == 0) {
        packet[0] = bad ? R(INFINITY) : R(exponent);
        packet[1] = scratch[0];
        const T endpoint = owns_endpoint ? v[tail] : T(0);
        packet[2] = dr(endpoint);
        packet[3] = di(endpoint);
    }
}

template <class T, class R> __device__ T pair_value(R ar, R ai) {
    if constexpr (std::is_same<T, R>::value)
        return ar;
    else
        return T(ar, ai);
}

template <class T, class R>
__global__ void normalize_scaled_kernel(T* v, int lr, int active, int pivot, T* a, const R* packets,
                                        int ranks, int* invalid, int inc, bool conjugate) {
    __shared__ R coefficient[7];
    __shared__ int bad;
    if (threadIdx.x == 0) {
        // AllGather preserves packet bits, including subnormal endpoints.
        // Only this short reduction needs to interpret the packets. It cannot
        // be a plain SUM: local squared norms have different binary scales.
        R exponent = R(-4096), sum[3] = {0, 0, 0};
        bad = 0;
        for (int r = 0; r < ranks; ++r) {
            for (int k = 0; k < 4; ++k)
                bad |= !isfinite(packets[4 * r + k]);
            exponent = fmax(exponent, packets[4 * r]);
        }
        if (!bad)
            for (int r = 0; r < ranks; ++r) {
                const R* p = packets + 4 * r;
                sum[0] += scalbn(p[1], 2 * (static_cast<int>(p[0]) - static_cast<int>(exponent)));
                sum[1] += scalbn(p[2], -static_cast<int>(exponent));
                sum[2] += scalbn(p[3], -static_cast<int>(exponent));
            }
        bad |= (!isfinite(sum[0]) || !isfinite(sum[1]) || !isfinite(sum[2]));
        R beta = 0, beta_scaled = 1;
        T tau = T(0), denominator = T(1);
        if (!bad) {
            const T alpha = pair_value<T, R>(sum[1], sum[2]);
            const R tail = sqrt(sum[0]);
            if (tail != 0 || sum[2] != 0) {
                beta_scaled = -copysign(hypot(tail, hypot(sum[1], sum[2])), sum[1]);
                tau = device_conj(T(1) - alpha / beta_scaled);
                denominator = alpha / beta_scaled - T(1);
                beta = scalbn(beta_scaled, static_cast<int>(exponent));
            } else
                beta = scalbn(sum[1], static_cast<int>(exponent));
            bad = (!isfinite(beta) || !isfinite(dr(tau)) || !isfinite(di(tau)));
        }
        if (bad) {
            *invalid = 1;
            beta = 0;
            beta_scaled = 1;
            tau = T(0);
            denominator = T(1);
        }
        coefficient[0] = dr(tau);
        coefficient[1] = di(tau);
        coefficient[2] = dr(denominator);
        coefficient[3] = di(denominator);
        coefficient[4] = beta;
        coefficient[5] = beta_scaled;
        coefficient[6] = bad ? R(0) : exponent;
    }
    __syncthreads();
    const T denominator = pair_value<T, R>(coefficient[2], coefficient[3]);
    const int e = static_cast<int>(coefficient[6]);
    for (int i = threadIdx.x; i < lr; i += 256) {
        T value = T(0);
        if (i < active) {
            if (i == pivot)
                value = T(1);
            else if (!bad)
                value = pair_value<T, R>(scalbn(dr(v[i]), -e), scalbn(di(v[i]), -e)) / coefficient[5] /
                        denominator;
            if (a) {
                const T stored = i == pivot ? T(coefficient[4]) : value;
                a[static_cast<std::size_t>(i) * inc] = conjugate ? device_conj(stored) : stored;
            }
        }
        v[i] = value;
    }
    if (threadIdx.x == 0)
        v[lr] = pair_value<T, R>(coefficient[0], coefficient[1]);
}

template <class T>
__global__ void form_u_kernel(int n, const T* v, const T* y, const T* tau, const T* dot, T* u) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const T t = *tau;
        u[i] = device_conj(t) * y[i] - T(0.5) * t * device_conj(t) * dr(*dot) * v[i];
    }
}

template <class T, class R>
__global__ void normalize_kernel(T* v, int lr, int active, int pivot, T denominator, T tau_value, R beta,
                                 R beta_for_v, R scale, int rescales, T* a, T* tau, int inc, bool conjugate) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0)
        *tau = tau_value;
    if (i < lr) {
        T value = T(0);
        if (i < active) {
            if (i == pivot)
                value = T(1);
            else {
                value = v[i];
                for (int q = 0; q < rescales; ++q)
                    value *= scale;
                value /= beta_for_v;
                value /= denominator;
            }
            if (a) {
                const T stored = i == pivot ? T(beta) : value;
                a[static_cast<std::size_t>(i) * inc] = conjugate ? device_conj(stored) : stored;
            }
        }
        v[i] = value;
    }
}

// Vendor HEMV accepts a local square Hermitian matrix, not a rectangular
// block-cyclic shard whose scalar triangle follows global indices. Paired
// BLAS calls for each stored block dominated the measured GPU reduction.
// Fuse that distributed-layout operation; keep all rank-2k updates
// in BLAS. Each tile owns its two partial vectors, so no atomics or replicated
// matrix are needed. Scratch is about 1/16 of the local matrix (32-wide tiles).
template <class T>
__global__ void triangle_av_tiles_kernel(int nr, int nc, int nb, int mr, int mc, int pr, int pc, const T* a,
                                         int ld, bool lower, const T* vr, const T* vc, T* partial_r, int ldr,
                                         T* partial_c, int ldc) {
    constexpr int tile = 32;
    __shared__ T block[tile][tile + 1];
    __shared__ T sums_r[8][tile], sums_c[8][tile];
    __shared__ int global_r[tile], global_c[tile];
    const int x = threadIdx.x, y = threadIdx.y, r0 = blockIdx.x * tile, c0 = blockIdx.y * tile;
    const int r = r0 + x, c = c0 + x;
    const int gr0 = (r0 / nb * pr + mr) * nb + r0 % nb, gc0 = (c0 / nb * pc + mc) * nb + c0 % nb;
    const int rend = min(nr, r0 + tile) - 1, cend = min(nc, c0 + tile) - 1;
    const int grend = (rend / nb * pr + mr) * nb + rend % nb, gcend = (cend / nb * pc + mc) * nb + cend % nb;
    if (lower ? grend < gc0 : gr0 > gcend) {
        if (y == 0) {
            if (r < nr)
                partial_r[r + static_cast<std::size_t>(blockIdx.y) * ldr] = T(0);
            if (c < nc)
                partial_c[c + static_cast<std::size_t>(blockIdx.x) * ldc] = T(0);
        }
        return;
    }
    if (y == 0) {
        global_r[x] = r < nr ? (r / nb * pr + mr) * nb + r % nb : -1;
        global_c[x] = c < nc ? (c / nb * pc + mc) * nb + c % nb : -2;
    }
    __syncthreads();
    const int gr = global_r[x];
    for (int q = y; q < tile; q += 8) {
        const int j = c0 + q, gc = global_c[q];
        T value = T(0);
        if (r < nr && j < nc && (lower ? gr >= gc : gr <= gc)) {
            value = a[r + static_cast<std::size_t>(j) * ld];
            if (gr == gc)
                value = T(dr(value));
        }
        block[x][q] = value;
    }
    __syncthreads();
    T sr = T(0), sc = T(0);
    const int gc = global_c[x];
    for (int q = y; q < tile; q += 8) {
        if (c0 + q < nc)
            sr += block[x][q] * vc[c0 + q];
        if (r0 + q < nr) {
            const int i = r0 + q, gi = global_r[q];
            // A diagonal belongs to exactly one output layout.
            if (gi != gc)
                sc += device_conj(block[q][x]) * vr[i];
        }
    }
    sums_r[y][x] = sr;
    sums_c[y][x] = sc;
    __syncthreads();
    for (int step = 4; step; step /= 2) {
        if (y < step) {
            sums_r[y][x] += sums_r[y + step][x];
            sums_c[y][x] += sums_c[y + step][x];
        }
        __syncthreads();
    }
    if (y == 0) {
        if (r < nr)
            partial_r[r + static_cast<std::size_t>(blockIdx.y) * ldr] = sums_r[0][x];
        if (c < nc)
            partial_c[c + static_cast<std::size_t>(blockIdx.x) * ldc] = sums_c[0][x];
    }
}

template <class T>
__global__ void triangle_av_sum_kernel(int nr, int nc, const T* partial_r, int ldr, const T* partial_c,
                                       int ldc, T* yr, T* yc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nr) {
        T sum = T(0);
        for (int j = 0; j < (nc + 31) / 32; ++j)
            sum += partial_r[i + static_cast<std::size_t>(j) * ldr];
        yr[i] = sum;
    }
    if (i < nc) {
        T sum = T(0);
        for (int j = 0; j < (nr + 31) / 32; ++j)
            sum += partial_c[i + static_cast<std::size_t>(j) * ldc];
        yc[i] = sum;
    }
}

template <class T, class R>
__global__ void extract_kernel(int n, int nb, int row, int col, int rows, int cols, const T* a, int ld, R* de,
                               bool lower) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        const int lc = j / (nb * cols) * nb + j % nb;
        if (j / nb % cols == col) {
            if (j / nb % rows == row) {
                const int r = j / (nb * rows) * nb + j % nb;
                de[j] = dr(
                    a[lower ? lc + static_cast<std::size_t>(r) * ld : r + static_cast<std::size_t>(lc) * ld]);
            }
            if (j > 0 && (j - 1) / nb % rows == row) {
                const int r = (j - 1) / (nb * rows) * nb + (j - 1) % nb;
                de[n + j - 1] = dr(
                    a[lower ? lc + static_cast<std::size_t>(r) * ld : r + static_cast<std::size_t>(lc) * ld]);
            }
        }
    }
}
#endif

template <DdlaBackend B, class T>
void form_u(const DdlaHandle_t& h, int n, const T* v, const T* y, const T* tau, const T* dot, T* u) {
    if constexpr (B == DdlaBackend::CPU) {
        const T t = *tau;
        for (int i = 0; i < n; ++i)
            u[i] = detail::host_conj(t) * y[i] - T(0.5) * t * detail::host_conj(t) * re(*dot) * v[i];
    } else {
#if DDLA_HAS_GPU
        using D = DeviceScalarT<T>;
        if (n)
            form_u_kernel<D><<<(n + 255) / 256, 256, 0, static_cast<gpuStream_t>(h->stream)>>>(
                n, reinterpret_cast<const D*>(v), reinterpret_cast<const D*>(y),
                reinterpret_cast<const D*>(tau), reinterpret_cast<const D*>(dot), reinterpret_cast<D*>(u));
        DDLA_CHECK(ddlaGetLastError(h));
#endif
    }
}

// The one-column QL is a distributed LARFG, rather than a vendor QR
// factorization with per-call workspaces. Native CCL keeps coefficients on
// the device; CPU/staged paths exchange robust local norms and endpoints.
template <DdlaBackend B, class T>
void factor_one(const DdlaHandle_t& h, int j, T* a, bool lower, int nb, int ld, int pr, int pc, int mr, int mc, T* v, T* tau,
                typename real_type<T>::type* norm_work, typename real_type<T>::type* gathered, int* invalid) {
    using R = typename real_type<T>::type;
    const int nr = num_loc(j, nb, mr, 0, pr), inc = lower ? ld : 1;
    const int tail = num_loc(j - 1, nb, mr, 0, pr), pivot_owner = (j - 1) / nb % pr;
    const int owner = j / nb % pc, localj = indxg2l(j, nb, pc);
#if DDLA_HAS_GPU && defined(DDLA_USE_CCL) && !defined(DDLA_USE_GPU_CPU_TUNNEL)
    if constexpr (B == DdlaBackend::GPU) {
        using D = DeviceScalarT<T>;
        detail::agree_local(h,"one-stage eigensolver", [&] {
            if (mc == owner) {
                const T* column = nr ? a + triangle_offset(lower, ld, 0, localj) : nullptr;
                scaled_pack_kernel<D, R><<<1, 256, 0, static_cast<gpuStream_t>(h->stream)>>>(
                    reinterpret_cast<const D*>(column), nr, tail, nr, mr == pivot_owner,
                    reinterpret_cast<D*>(v), norm_work, inc, lower);
                DDLA_CHECK(ddlaGetLastError(h));
            }
        });
        detail::agree_local(h,"one-stage eigensolver", [&] {
            if (mc == owner) {
                const R* packets = norm_work;
                if (pr > 1) {
                    CCL_CHECK(gpucclAllGather(norm_work, norm_work + 4, 4 * sizeof(R), gpucclInt8,
                                              CommTraits<B>::comm(h, (lower ? CommScope::Row : CommScope::Col)),
                                              static_cast<gpuStream_t>(h->stream)));
                    packets = norm_work + 4;
                }
                T* column = nr ? a + triangle_offset(lower, ld, 0, localj) : nullptr;
                normalize_scaled_kernel<D, R><<<1, 256, 0, static_cast<gpuStream_t>(h->stream)>>>(
                    reinterpret_cast<D*>(v), nr, nr, mr == pivot_owner ? tail : -1,
                    reinterpret_cast<D*>(column), packets, pr, invalid, inc, lower);
                DDLA_CHECK(ddlaGetLastError(h));
            }
        });
        // Every row agrees on nr. Even an empty row receives its tau entry.
        commBcast<B>(h, (lower ? CommScope::Col : CommScope::Row), v, nr + 1, owner);
        DDLA_CHECK(ddlaMemcpy(h, tau, v + nr, sizeof(T)));
        return;
    }
#endif
    R packet[3] = {0, 0, 0};
    detail::agree_local(h,"one-stage eigensolver", [&] {
        // The row broadcast writes every live entry on nonowners. Neither
        // MPI nor the host-staged path consumes their receive buffer first.
        if (mc != owner)
            return;
        if constexpr (B == DdlaBackend::CPU) {
            DDLA_CHECK(ddlaMemset(h, v, 0, static_cast<std::size_t>(nr + 1) * sizeof(T)));
            if (nr) {
                if (lower)
                    blas_impl::geam_run<B, T>(h, 'C', 'C', nr, 1, T(1), a + triangle_offset(lower, ld, 0, localj),
                                              ld, T(0), a + triangle_offset(lower, ld, 0, localj), ld, v,
                                              nr);
                else
                    DDLA_CHECK(ddlaMemcpy(h, v, a + triangle_offset(lower, ld, 0, localj),
                                          static_cast<std::size_t>(nr) * sizeof(T)));
            }
            DDLA_CHECK(ddlaMemset(h, norm_work, 0, sizeof(R)));
            if (tail)
                blas_impl::nrm2_run<B, T>(h, tail, v, 1, norm_work);
            DDLA_CHECK(ddlaMemcpy(h, packet, norm_work, sizeof(R)));
            T endpoint = 0;
            if (mr == pivot_owner)
                DDLA_CHECK(ddlaMemcpy(h, &endpoint, v + tail, sizeof(T)));
            packet[1] = std::real(endpoint);
            packet[2] = std::imag(endpoint);
        } else {
#if DDLA_HAS_GPU
            using D = DeviceScalarT<T>;
            const T* column = nr ? a + triangle_offset(lower, ld, 0, localj) : nullptr;
            norm_pack_kernel<D, R><<<1, 256, 0, static_cast<gpuStream_t>(h->stream)>>>(
                reinterpret_cast<const D*>(column), nr, tail, nr, mr == pivot_owner, reinterpret_cast<D*>(v),
                norm_work, inc, lower);
            DDLA_CHECK(ddlaGetLastError(h));
            DDLA_CHECK(ddlaMemcpy(h, packet, norm_work, 3 * sizeof(R)));
            DDLA_CHECK(ddlaSynchronize(h));
#endif
        }
    });
    R xnorm = 0;
    T alpha = 0;
    if (mc == owner) {
        MPI_CHECK(MPI_Allgather(packet, 3, std::is_same<R, double>::value ? MPI_DOUBLE : MPI_FLOAT, gathered,
                                3, std::is_same<R, double>::value ? MPI_DOUBLE : MPI_FLOAT,
                                lower ? h->row_comm : h->col_comm));
        for (int r = 0; r < pr; ++r)
            xnorm = std::hypot(xnorm, gathered[3 * r]);
        if constexpr (std::is_same<T, R>::value)
            alpha = gathered[3 * pivot_owner + 1];
        else
            alpha = T(gathered[3 * pivot_owner + 1], gathered[3 * pivot_owner + 2]);
    }
    T t = 0, denominator = 1;
    R beta = std::real(alpha), beta_for_v = 1;
    int rescales = 0;
    const R safmin = std::numeric_limits<R>::min() / std::numeric_limits<R>::epsilon(), scale = R(1) / safmin;
    if (xnorm != 0 || std::imag(alpha) != 0) {
        beta = -std::copysign(std::hypot(xnorm, std::abs(alpha)), R(std::real(alpha)));
        while (std::abs(beta) < safmin && beta != 0 && rescales < 20) {
            ++rescales;
            alpha *= scale;
            xnorm *= scale;
            beta *= scale;
        }
        beta = -std::copysign(std::hypot(xnorm, std::abs(alpha)), R(std::real(alpha)));
        beta_for_v = beta;
        // Ratios avoid overflowing alpha-beta when the vector norm fits R.
        t = detail::host_conj(T(1) - alpha / beta);
        denominator = alpha / beta - T(1);
        for (int q = 0; q < rescales; ++q)
            beta *= safmin;
    }
    detail::agree_local(h,"one-stage eigensolver", [&] {
        if (!std::isfinite(std::abs(t)) || !std::isfinite(beta))
            throw std::runtime_error("one-stage Householder is not finite");
        T* column = mc == owner && nr ? a + triangle_offset(lower, ld, 0, localj) : nullptr;
        const int pivot = mr == pivot_owner ? tail : -1;
        // Only the owning process column computes the reflector. Broadcast
        // its exact coefficients afterwards; other columns do not repeat a
        // floating-point norm and independently choose a slightly different Q.
        T* packed_tau = v + nr; // first entry of the not-yet-live U column
        if (mc == owner) {
            if constexpr (B == DdlaBackend::CPU) {
                *packed_tau = t;
                for (int i = 0; i < nr; ++i) {
                    T value = 0;
                    if (i < nr) {
                        if (i == pivot)
                            value = 1;
                        else {
                            value = v[i];
                            for (int q = 0; q < rescales; ++q)
                                value *= scale;
                            value /= beta_for_v;
                            value /= denominator;
                        }
                        if (column) {
                            const T stored = i == pivot ? T(beta) : value;
                            column[static_cast<std::size_t>(i) * inc] = lower ? detail::host_conj(stored) : stored;
                        }
                    }
                    v[i] = value;
                }
            } else {
#if DDLA_HAS_GPU
                using D = DeviceScalarT<T>;
                normalize_kernel<D, R>
                    <<<std::max(1, (nr + 255) / 256), 256, 0, static_cast<gpuStream_t>(h->stream)>>>(
                        reinterpret_cast<D*>(v), nr, nr, pivot, to_device_scalar(denominator),
                        to_device_scalar(t), beta, beta_for_v, scale, rescales, reinterpret_cast<D*>(column),
                        reinterpret_cast<D*>(packed_tau), inc, lower);
                DDLA_CHECK(ddlaGetLastError(h));
#endif
            }
        }
    });
    // Tau occupies the first inactive V entry (or U[0] when nr reaches
    // capacity). The copy completes on the stream before that slot is reused.
    commBcast<B>(h, (lower ? CommScope::Col : CommScope::Row), v, nr + 1, owner);
    DDLA_CHECK(ddlaMemcpy(h, tau, v + nr, sizeof(T)));
}

template <DdlaBackend B, class T>
void transpose_vector(const DdlaHandle_t& h, const T* src, int ls, T* dst, int ld, int n,
                      bool lower, int nb, int pr, int pc, int mr, int mc, void* work, std::size_t bytes, bool reverse = false,
                      bool accumulate = false) {
    if (pr != pc) {
        const auto p = spectral_impl::ptranspose_vectors_geometry(h, B, n, 1, nb, ls, ld, accumulate,
                                                                  lower != reverse);
        spectral_impl::ptranspose_vectors_run<B, T>(h, p, src, dst, work, bytes);
        return;
    }
    const int count = num_loc(n, nb, reverse ? mr : mc, 0, pr);
    if (!count)
        return;
    if (mr == mc) {
        if (accumulate)
            blas_impl::axpy_run<B, T>(h, count, T(1), src, 1, dst, 1);
        else
            DDLA_CHECK(ddlaMemcpy(h, dst, src, static_cast<std::size_t>(count) * sizeof(T)));
    }
    commBcast<B>(h, reverse ? (lower ? CommScope::Col : CommScope::Row) : (lower ? CommScope::Row : CommScope::Col), dst, count, reverse ? mr : mc);
}

// A strict off-diagonal rectangle contributes to both vector layouts. Only
// a true global-diagonal square can be passed to a local Hermitian BLAS call.
template <DdlaBackend B, class T>
void triangle_av(const DdlaHandle_t& h, int n, const T* a, bool lower, int nb, int ld, int pr, int pc, int mr, int mc, const T* vr, const T* vc,
                 T* yr, T* yc, T* partial_r, int ldr, T* partial_c, int ldc) {
    const int nr = num_loc(n, nb, mr, 0, pr), nc = num_loc(n, nb, mc, 0, pc);
    // Equal row/column index sets make a square-grid diagonal shard a
    // genuine local Hermitian matrix on either backend.
    if (pr == pc && mr == mc) {
        if (nr)
            blas_impl::hemv_run<B, T>(h, lower ? 'L' : 'U', nr, T(1), a, ld, vr, 1, T(0), yr, 1);
        if (nc)
            DDLA_CHECK(ddlaMemset(h, yc, 0, static_cast<std::size_t>(nc) * sizeof(T)));
        return;
    }
    if constexpr (B == DdlaBackend::GPU) {
#if DDLA_HAS_GPU
        const int ar = lower ? nc : nr, ac = lower ? nr : nc;
        T* row_out = lower ? yc : yr;
        T* col_out = lower ? yr : yc;
        if (ar && ac) {
            using D = DeviceScalarT<T>;
            triangle_av_tiles_kernel<D><<<dim3((ar + 31) / 32, (ac + 31) / 32), dim3(32, 8), 0,
                                          static_cast<gpuStream_t>(h->stream)>>>(
                ar, ac, nb, lower ? mc : mr, lower ? mr : mc, lower ? pc : pr,
                lower ? pr : pc, reinterpret_cast<const D*>(a), ld, lower,
                reinterpret_cast<const D*>(lower ? vc : vr), reinterpret_cast<const D*>(lower ? vr : vc),
                reinterpret_cast<D*>(partial_r), ldr, reinterpret_cast<D*>(partial_c), ldc);
            DDLA_CHECK(ddlaGetLastError(h));
        }
        // Zero-length contractions still produce defined zero vectors.
        triangle_av_sum_kernel<DeviceScalarT<T>>
            <<<std::max(1, (std::max(ar, ac) + 255) / 256), 256, 0, static_cast<gpuStream_t>(h->stream)>>>(
                ar, ac, reinterpret_cast<const DeviceScalarT<T>*>(partial_r), ldr,
                reinterpret_cast<const DeviceScalarT<T>*>(partial_c), ldc,
                reinterpret_cast<DeviceScalarT<T>*>(row_out), reinterpret_cast<DeviceScalarT<T>*>(col_out));
        DDLA_CHECK(ddlaGetLastError(h));
        return;
#endif
    }
    if (nr)
        DDLA_CHECK(ddlaMemset(h, yr, 0, static_cast<std::size_t>(nr) * sizeof(T)));
    if (nc)
        DDLA_CHECK(ddlaMemset(h, yc, 0, static_cast<std::size_t>(nc) * sizeof(T)));
    for (int first = 0; first < n; first += nb) {
        if (first / nb % pc != mc)
            continue;
        const int width = std::min(nb, n - first), r = num_loc(first, nb, mr, 0, pr),
                  c = indxg2l(first, nb, pc);
        if (r) {
            const T* block = a + triangle_offset(lower, ld, 0, c);
            if (lower) {
                blas_impl::gemv_run<B, T>(h, 'C', width, r, T(1), block, ld, vc + c, 1, T(1), yr, 1);
                blas_impl::gemv_run<B, T>(h, 'N', width, r, T(1), block, ld, vr, 1, T(1), yc + c, 1);
            } else {
                blas_impl::gemv_run<B, T>(h, 'N', r, width, T(1), block, ld, vc + c, 1, T(1), yr, 1);
                blas_impl::gemv_run<B, T>(h, 'C', r, width, T(1), block, ld, vr, 1, T(1), yc + c, 1);
            }
        }
        if (first / nb % pr == mr)
            blas_impl::hemv_run<B, T>(h, lower ? 'L' : 'U', width, T(1), a + triangle_offset(lower, ld, r, c), ld, vr + r,
                                      1, T(1), yr + r, 1);
    }
}

// Recursively split at global block boundaries. Large strict rectangles keep
// GEMM throughput; HER2K alone handles diagonal squares and ignores their
// imaginary diagonal. The opposite scalar triangle is never a GEMM output.
template <DdlaBackend B, class T>
void triangle_update(const DdlaHandle_t& h, int lo, int hi, int pending, T* a, bool lower, int nb, int ld, int pr, int pc, int mr, int mc,
                     const T* rows, int lr, const T* cols, int lc) {
    using R = typename real_type<T>::type;
    if (lo >= hi)
        return;
    if (lo == 0 && pr == pc && mr == mc) {
        const int count = num_loc(hi, nb, mr, 0, pr), step = 2 * (lower ? lc : lr);
        const T* v = lower ? cols + lc : rows;
        const T* u = lower ? cols : rows + lr;
        if (count)
            blas_impl::her2k_run<B, T>(h, lower ? 'L' : 'U', 'N', count, pending, T(-1), v, step, u, step,
                                       R(1), a, ld);
        return;
    }
    const int first = lo / nb, last = (hi - 1) / nb;
    if (first == last) {
        if (first % pr == mr && first % pc == mc) {
            const int r = num_loc(lo, nb, mr, 0, pr), c = num_loc(lo, nb, mc, 0, pc);
            const T* v = lower ? cols + lc + c : rows + r;
            const T* u = lower ? cols + c : rows + lr + r;
            const int step = 2 * (lower ? lc : lr);
            blas_impl::her2k_run<B, T>(h, lower ? 'L' : 'U', 'N', hi - lo, pending, T(-1), v, step, u, step,
                                       R(1), a + triangle_offset(lower, ld, r, c), ld);
        }
        return;
    }
    const int mid = (first + (last - first + 1) / 2) * nb;
    const int r = num_loc(lo, nb, mr, 0, pr), c = num_loc(mid, nb, mc, 0, pc);
    const int nr = num_loc(mid, nb, mr, 0, pr) - r, nc = num_loc(hi, nb, mc, 0, pc) - c;
    if (nr && nc) {
        if (lower)
            blas_impl::gemm_run<B, T>(h, 'N', 'C', nc, nr, 2 * pending, T(-1), cols + c, lc, rows + r, lr,
                                      T(1), a + triangle_offset(lower, ld, r, c), ld);
        else
            blas_impl::gemm_run<B, T>(h, 'N', 'C', nr, nc, 2 * pending, T(-1), rows + r, lr, cols + c, lc,
                                      T(1), a + triangle_offset(lower, ld, r, c), ld);
    }
    triangle_update<B, T>(h, lo, mid, pending, a, lower, nb, ld, pr, pc, mr, mc, rows, lr, cols, lc);
    triangle_update<B, T>(h, mid, hi, pending, a, lower, nb, ld, pr, pc, mr, mc, rows, lr, cols, lc);
}

// The scratch of reduce on the logical grid of TriangleLayout (lr x lc local, swapped
// for the stored lower triangle).  Backend, in carving order: the pending rows [V|U]
// (lr x 2*cap) and columns [Uc|Vc] (lc x 2*cap), the A v partials and their sum (ly
// each: the column layout's lc for the full matrix on a square GPU grid, else lr), the
// dot partials and sums (2*cap each), the column-layout partials and sums (lc each,
// stored triangle), the partial vectors of the GPU triangle tiles (stored triangle off
// the square-grid diagonal), the norm packets (4 * (pr + 1)), the packed diagonals
// (2n), the device invalid flag (native CCL), then the vector transposes' (n x 1).
// Host: the gathered norm packets (3 * pr).
template <class T>
void reduce_workspace_size(const DdlaHandle_t& h, const spectral_impl::HetrdPlan& p,
                           std::size_t* backend_bytes, std::size_t* host_bytes) {
    using R = typename real_type<T>::type;
    constexpr std::size_t cap = 32;
    const bool lower = p.stored_only && p.uplo == 'L';
    const int pr = lower ? p.pc : p.pr, pc = lower ? p.pr : p.pc, mr = lower ? p.mc : p.mr,
              mc = lower ? p.mr : p.mc;
    const int nb = p.a[DDLA_NB_];
    const std::size_t lr = std::max(1, num_loc(p.n, nb, mr, 0, pr)),
                      lc = std::max(1, num_loc(p.n, nb, mc, 0, pc));
    const std::size_t ly = !p.stored_only && p.backend == DdlaBackend::GPU && pr == pc ? lc : lr;
    const std::size_t ldr = lower ? lc : lr, ldc = lower ? lr : lc;
    const bool tiles = p.stored_only && p.backend == DdlaBackend::GPU && (pr != pc || mr != mc);
    std::size_t transpose = 0;
    spectral_impl::ptranspose_vectors_workspace_size<T>(h, p.n, 1, nb, &transpose);
    *backend_bytes =
        padded_bytes<T>(lr * 2 * cap) + padded_bytes<T>(lc * 2 * cap) + 2 * padded_bytes<T>(ly) +
        2 * padded_bytes<T>(2 * cap) + (p.stored_only ? 2 * padded_bytes<T>(lc) : 0) +
        (tiles ? padded_bytes<T>(ldr * ((ldc + 31) / 32)) + padded_bytes<T>(ldc * ((ldr + 31) / 32)) : 0) +
        padded_bytes<R>(4 * (std::size_t(pr) + 1)) + padded_bytes<R>(2 * std::size_t(p.n)) + transpose;
#if defined(DDLA_USE_CCL) && !defined(DDLA_USE_GPU_CPU_TUNNEL)
    if (p.backend == DdlaBackend::GPU)
        *backend_bytes += padded_bytes<int>(1);
#endif
    *host_bytes = padded_bytes<R>(3 * std::size_t(pr));
}

template <bool StoredOnly, DdlaBackend B, class T>
void reduce(const DdlaHandle_t& h, char uplo, int n, T* a, const int* desc, T* tau,
            typename real_type<T>::type* de, spectral_impl::HetrdResources& resources, void* backend_work,
            std::size_t backend_bytes, void* host_work, std::size_t host_bytes) {
    constexpr int cap = 32;
    const bool lower = StoredOnly && uplo == 'L';
    int pr, pc, mr, mc;
    ddlaGetGridDims(h, pr, pc);
    ddlaGetGridCoords(h, mr, mc);
    if (lower) { std::swap(pr, pc); std::swap(mr, mc); }
    const int nb = desc[DDLA_NB_], ld = desc[DDLA_LLD_];
    const int lr = std::max(1, num_loc(n, nb, mr, 0, pr)), lc = std::max(1, num_loc(n, nb, mc, 0, pc));
    using R = typename real_type<T>::type;
    const bool adjoint_av = !StoredOnly && B == DdlaBackend::GPU && pr == pc;
    const int ly = adjoint_av ? lc : lr, ldr = lower ? lc : lr, ldc = lower ? lr : lc;
    const bool tiles = StoredOnly && B == DdlaBackend::GPU && (pr != pc || mr != mc);
    // Carved in the order reduce_workspace_size counts them; the transposes take the rest.
    char* const base = static_cast<char*>(backend_work);
    char* w = base;
    T* rows = take<T>(w, std::size_t(lr) * 2 * cap);
    T* cols = take<T>(w, std::size_t(lc) * 2 * cap);
    T* yp = take<T>(w, ly);
    T* y = take<T>(w, ly);
    T* dotp = take<T>(w, 2 * cap);
    T* dot = take<T>(w, 2 * cap);
    T* ycp = take<T>(w, StoredOnly ? lc : 0);
    T* yc = take<T>(w, StoredOnly ? lc : 0);
    T* partial_r = take<T>(w, tiles ? std::size_t(ldr) * ((ldc + 31) / 32) : 0);
    T* partial_c = take<T>(w, tiles ? std::size_t(ldc) * ((ldr + 31) / 32) : 0);
    R* norm_work = take<R>(w, 4 * (std::size_t(pr) + 1));
    R* packed = take<R>(w, std::size_t(2) * n);
    int* invalid = nullptr;
#if defined(DDLA_USE_CCL) && !defined(DDLA_USE_GPU_CPU_TUNNEL)
    if constexpr (B == DdlaBackend::GPU)
        invalid = take<int>(w, 1);
#endif
    assert(static_cast<std::size_t>(w - base) <= backend_bytes);
    void* const transpose_work = w;
    const std::size_t transpose_bytes = backend_bytes - static_cast<std::size_t>(w - base);
    char* hw = static_cast<char*>(host_work);
    R* gathered = take<R>(hw, std::size_t(3) * pr);
    assert(static_cast<std::size_t>(hw - static_cast<char*>(host_work)) <= host_bytes);
    (void)host_bytes;
    detail::agree_local(h,"one-stage eigensolver", [&] {
        resources.begin_reduction();
        if (invalid)
            DDLA_CHECK(ddlaMemset(h, invalid, 0, sizeof(int)));
        // Inactive slots remain defined while subsequent columns use prefixes.
        DDLA_CHECK(ddlaMemset(h, cols, 0, std::size_t(lc) * 2 * cap * sizeof(T)));
    });
    int pending = 0;
    for (int j = n - 1; j >= 1; --j) {
        const int nr = num_loc(j, nb, mr, 0, pr), nc = num_loc(j, nb, mc, 0, pc);
        const int owner = j / nb % pc, localj = indxg2l(j, nb, pc);
        if constexpr (StoredOnly) {
            // Restore only the stored column of the logical view. Handle its
            // diagonal separately: HER2K does not consume an imaginary diagonal.
            if (pending && mc == owner) {
                if (nr) {
                    if (lower)
                        blas_impl::gemm_run<B, T>(h, 'N', 'C', 1, nr, 2 * pending, T(-1), cols + localj, lc,
                                                  rows, lr, T(1), a + triangle_offset(lower, ld, 0, localj), ld);
                    else
                        blas_impl::gemm_run<B, T>(h, 'N', 'C', nr, 1, 2 * pending, T(-1), rows, lr,
                                                  cols + localj, lc, T(1), a + triangle_offset(lower, ld, 0, localj), ld);
                }
                if (j / nb % pr == mr) {
                    const int r = indxg2l(j, nb, pr);
                    const T* v = lower ? cols + lc + localj : rows + r;
                    const T* u = lower ? cols + localj : rows + lr + r;
                    const int step = 2 * (lower ? lc : lr);
                    blas_impl::her2k_run<B, T>(h, lower ? 'L' : 'U', 'N', 1, pending, T(-1), v, step,
                                               u, step, R(1), a + triangle_offset(lower, ld, r, localj), ld);
                }
            }
        } else {
            // Full-matrix method: keep the original single GEMM, including
            // the already-real diagonal, before factoring the next column.
            const int ni = num_loc(j + 1, nb, mr, 0, pr);
            if (pending && mc == owner && ni)
                blas_impl::gemm_run<B, T>(h, 'N', 'C', ni, 1, 2 * pending, T(-1), rows, lr, cols + localj, lc,
                                          T(1), a + static_cast<std::size_t>(localj) * ld, ld);
        }
        T* v = rows + static_cast<std::size_t>(2 * pending) * lr;
        T* u = v + lr;
        T* uc = cols + static_cast<std::size_t>(2 * pending) * lc;
        T* vc = uc + lc;
        factor_one<B, T>(h, j, a, lower, nb, ld, pr, pc, mr, mc, v, tau + j - 1, norm_work, gathered, invalid);
        transpose_vector<B, T>(h, v, lr, vc, lc, j, lower, nb, pr, pc, mr, mc, transpose_work, transpose_bytes);
        if constexpr (StoredOnly) {
            triangle_av<B, T>(h, j, a, lower, nb, ld, pr, pc, mr, mc, v, vc, yp, ycp, partial_r, ldr, partial_c, ldc);
            // Pending corrections are a full Hermitian product, independent of
            // which half stores A. Apply local column contributions before SUM.
            if (pending && nc) {
                blas_impl::gemv_run<B, T>(h, 'C', nc, 2 * pending, T(1), cols, lc, vc, 1, T(0), dotp, 1);
                if (nr)
                    blas_impl::gemv_run<B, T>(h, 'N', nr, 2 * pending, T(-1), rows, lr, dotp, 1, T(1), yp, 1);
            }
            commAllReduce<B>(h, (lower ? CommScope::Col : CommScope::Row), yp, y, nr, cclSum);
            commAllReduce<B>(h, (lower ? CommScope::Row : CommScope::Col), ycp, yc, nc, cclSum);
            transpose_vector<B, T>(h, yc, lc, y, lr, j, lower, nb, pr, pc, mr, mc, transpose_work, transpose_bytes, true, true);
            if (nr)
                blas_impl::dotc_run<B, T>(h, nr, v, 1, y, 1, dotp);
            else
                DDLA_CHECK(ddlaMemset(h, dotp, 0, sizeof(T)));
            commAllReduce<B>(h, (lower ? CommScope::Row : CommScope::Col), dotp, dot, 1, cclSum);
            form_u<B, T>(h, nr, v, y, tau + j - 1, dot, u);
            transpose_vector<B, T>(h, u, lr, uc, lc, j, lower, nb, pr, pc, mr, mc, transpose_work, transpose_bytes);
        } else {
            if (adjoint_av) {
                // A_current^H = A_base^H - Cols*Rows^H. Reduce the column
                // layout, then form U there before its single reverse broadcast.
                if (nr && nc)
                    blas_impl::gemv_run<B, T>(h, 'C', nr, nc, T(1), a, ld, v, 1, T(0), yp, 1);
                else if (nc)
                    DDLA_CHECK(ddlaMemset(h, yp, 0, static_cast<std::size_t>(nc) * sizeof(T)));
                if (pending && nr) {
                    blas_impl::gemv_run<B, T>(h, 'C', nr, 2 * pending, T(1), rows, lr, v, 1, T(0), dotp, 1);
                    if (nc)
                        blas_impl::gemv_run<B, T>(h, 'N', nc, 2 * pending, T(-1), cols, lc, dotp, 1, T(1), yp,
                                                  1);
                }
                commAllReduce<B>(h, CommScope::Col, yp, y, nc, cclSum);
                if (nc)
                    blas_impl::dotc_run<B, T>(h, nc, vc, 1, y, 1, dotp);
                else
                    DDLA_CHECK(ddlaMemset(h, dotp, 0, sizeof(T)));
                commAllReduce<B>(h, CommScope::Row, dotp, dot, 1, cclSum);
                form_u<B, T>(h, nc, vc, y, tau + j - 1, dot, uc);
                if (nr) {
                    if (mr == mc)
                        DDLA_CHECK(ddlaMemcpy(h, u, uc, static_cast<std::size_t>(nr) * sizeof(T)));
                    commBcast<B>(h, CommScope::Row, u, nr, mr);
                }
            } else {
                if (nr && nc)
                    blas_impl::gemv_run<B, T>(h, 'N', nr, nc, T(1), a, ld, vc, 1, T(0), yp, 1);
                else if (nr)
                    DDLA_CHECK(ddlaMemset(h, yp, 0, static_cast<std::size_t>(nr) * sizeof(T)));
                // A_current = A_base - Rows*Cols^H. Apply each process column's
                // correction before summing its partial Av; no separate coefficient
                // reduction is needed (ELPA's local correction schedule).
                if (pending && nc) {
                    blas_impl::gemv_run<B, T>(h, 'C', nc, 2 * pending, T(1), cols, lc, vc, 1, T(0), dotp, 1);
                    if (nr)
                        blas_impl::gemv_run<B, T>(h, 'N', nr, 2 * pending, T(-1), rows, lr, dotp, 1, T(1), yp,
                                                  1);
                }
                commAllReduce<B>(h, CommScope::Row, yp, y, nr, cclSum);
                if (nr)
                    blas_impl::dotc_run<B, T>(h, nr, v, 1, y, 1, dotp);
                else
                    DDLA_CHECK(ddlaMemset(h, dotp, 0, sizeof(T)));
                commAllReduce<B>(h, CommScope::Col, dotp, dot, 1, cclSum);
                form_u<B, T>(h, nr, v, y, tau + j - 1, dot, u);
                transpose_vector<B, T>(h, u, lr, uc, lc, j, lower, nb, pr, pc, mr, mc, transpose_work, transpose_bytes);
            }
        }
        ++pending;
        if (pending == cap || j == 1) {
            if constexpr (StoredOnly)
                triangle_update<B, T>(h, 0, j, pending, a, lower, nb, ld, pr, pc, mr, mc, rows, lr, cols, lc);
            else if (nr && nc)
                blas_impl::gemm_run<B, T>(h, 'N', 'C', nr, nc, 2 * pending, T(-1), rows, lr, cols, lc, T(1),
                                          a, ld);
            pending = 0;
        }
    }
    DDLA_CHECK(ddlaMemset(h, packed, 0, static_cast<std::size_t>(2) * n * sizeof(R)));
    if constexpr (B == DdlaBackend::CPU) {
        for (int j = 0; j < n; ++j)
            if (j / nb % pc == mc) {
                const int c = indxg2l(j, nb, pc);
                if (j / nb % pr == mr)
                    packed[j] = re(a[triangle_offset(lower, ld, indxg2l(j, nb, pr), c)]);
                if (j && (j - 1) / nb % pr == mr)
                    packed[n + j - 1] = re(a[triangle_offset(lower, ld, indxg2l(j - 1, nb, pr), c)]);
            }
    } else {
#if DDLA_HAS_GPU
        using D = DeviceScalarT<T>;
        extract_kernel<D, R><<<(n + 255) / 256, 256, 0, static_cast<gpuStream_t>(h->stream)>>>(
            n, nb, mr, mc, pr, pc, reinterpret_cast<const D*>(a), ld, packed, lower);
        DDLA_CHECK(ddlaGetLastError(h));
#endif
    }
    int failed = 0;
    detail::agree_local(h,"one-stage eigensolver", [&] {
        DDLA_CHECK(ddlaMemcpy(h, de, packed, static_cast<std::size_t>(2) * n * sizeof(R)));
        if (invalid)
            DDLA_CHECK(ddlaMemcpy(h, &failed, invalid, sizeof(int)));
        DDLA_CHECK(ddlaSynchronize(h));
    });
    if (invalid) {
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, ddlaGetCommunicator(h)));
        if (failed)
            throw std::runtime_error("one-stage Householder is not finite");
    }
    MPI_CHECK(::MPI_Allreduce(MPI_IN_PLACE, de, 2 * n,
                              std::is_same<R, double>::value ? MPI_DOUBLE : MPI_FLOAT, MPI_SUM,
                              ddlaGetCommunicator(h)));
    for (int i = 0; i < 2 * n; ++i)
        if (!std::isfinite(de[i]))
            throw std::runtime_error("one-stage tridiagonal is not finite");
}

} // namespace
} // namespace detail
namespace spectral_impl {
// Keep the CUDA 11.8--12.3 heuristic-cache workaround local to reduction,
// honor explicit vendor settings, and restore capacity on every exit.
HetrdResources::HetrdResources([[maybe_unused]] const DdlaHandle_t& h,
                                     [[maybe_unused]] const HetrdPlan& p) {
#if defined(DDLA_GPUBLAS_HAS_HEURISTICS_CACHE) && DDLA_GPUBLAS_HAS_HEURISTICS_CACHE
    if (p.backend != DdlaBackend::GPU || p.n < 1024 ||
        std::getenv("CUBLASLT_HEURISTICS_CACHE_CAPACITY"))
        return;
    int version = 0;
    BLAS_CHECK(gpublasGetVersion(static_cast<gpublasHandle_t>(h->blasH), &version));
    cache_eligible_ = version >= 111103 && version < 120304;
#endif
}

HetrdResources::~HetrdResources() {
    cancel_reduction();
}

void HetrdResources::begin_reduction() {
#if defined(DDLA_GPUBLAS_HAS_HEURISTICS_CACHE) && DDLA_GPUBLAS_HAS_HEURISTICS_CACHE
    if (!cache_eligible_)
        return;
    BLAS_CHECK(gpublasHeuristicsCacheGetCapacity(&cache_capacity_));
    if (cache_capacity_ != 8192)
        return;
    cache_changed_ = true;
    BLAS_CHECK(gpublasHeuristicsCacheSetCapacity(0));
#endif
}

void HetrdResources::end_reduction() {
#if defined(DDLA_GPUBLAS_HAS_HEURISTICS_CACHE) && DDLA_GPUBLAS_HAS_HEURISTICS_CACHE
    if (!cache_changed_)
        return;
    std::size_t current = 0;
    BLAS_CHECK(gpublasHeuristicsCacheGetCapacity(&current));
    if (current == 0)
        BLAS_CHECK(gpublasHeuristicsCacheSetCapacity(cache_capacity_));
    cache_changed_ = false;
#endif
}

void HetrdResources::cancel_reduction() noexcept {
#if defined(DDLA_GPUBLAS_HAS_HEURISTICS_CACHE) && DDLA_GPUBLAS_HAS_HEURISTICS_CACHE
    if (cache_changed_) {
        std::size_t current = 0;
        if (gpublasHeuristicsCacheGetCapacity(&current) == GPUBLAS_STATUS_SUCCESS && current == 0)
            (void)gpublasHeuristicsCacheSetCapacity(cache_capacity_);
        cache_changed_ = false;
    }
#endif
}


HetrdPlan phetrd_geometry(const DdlaHandle_t& h,DdlaBackend backend,char uplo,int n,const int* a,bool stored_only) {
    HetrdPlan p;p.backend=backend;p.uplo=uplo;p.n=n;p.stored_only=stored_only;
    ddlaGetGridDims(h,p.pr,p.pc);ddlaGetGridCoords(h,p.mr,p.mc);
    std::copy_n(a,DDLA_DLEN_,p.a.begin());return p;
}
template<class T>HetrdPlan phetrd_check(const DdlaHandle_t& h,DdlaBackend backend,char uplo,int n,
    const T* A,const int* a,const typename real_type<T>::type* d,const typename real_type<T>::type* e,const T* tau) {
    auto p=detail::agree_check(h,"phetrd",[&](detail::Agree&) {
        if(!h||ddlaGetBackend(h)!=backend)throw std::invalid_argument("phetrd: null handle or backend mismatch");
        const char u=static_cast<char>(std::toupper(static_cast<unsigned char>(uplo)));
        check_desc(a,h);
        if((u!='U'&&u!='L')||n<0||n>INT_MAX/2||n>a[DDLA_M_]||n>a[DDLA_N_]||
           a[DDLA_MB_]!=a[DDLA_NB_]||a[DDLA_RSRC_]!=0||a[DDLA_CSRC_]!=0)
            throw std::invalid_argument("phetrd: requires U/L, valid order and square source-zero blocks");
        auto result=phetrd_geometry(h,backend,u,n,a);
        const int rows=num_loc(n,a[DDLA_MB_],result.mr,0,result.pr),cols=num_loc(n,a[DDLA_NB_],result.mc,0,result.pc);
        if((n&&!d)||(n>1&&(!e||!tau))||(rows&&cols&&!A))
            throw std::invalid_argument("phetrd: missing active input/output");
        return result;
    });
    int sig[12]={int(backend),int(p.uplo),n,p.pr,p.pc};detail::descriptor_signature(sig+5,a);
    detail::agree_signature(h,"phetrd",sig);return p;
}
template<class T>void phetrd_workspace_size(const DdlaHandle_t& h,const HetrdPlan& p,std::size_t* backend,std::size_t* host) {
    *backend=*host=0;if(!p.n)return;detail::reduce_workspace_size<T>(h,p,backend,host);
}
template<DdlaBackend B,class T>void phetrd_run(const DdlaHandle_t& h,const HetrdPlan& p,HetrdResources& resources,
    T* A,T* tau,typename real_type<T>::type* de,int& info,void* work,std::size_t bytes,void* host,std::size_t host_bytes) {
    info=0;if(!p.n)return;
    struct Exit {HetrdResources& resources;~Exit(){resources.cancel_reduction();}} guard{resources};
    try {
        if(p.stored_only)detail::reduce<true,B,T>(h,p.uplo,p.n,A,p.a.data(),tau,de,resources,work,bytes,host,host_bytes);
        else detail::reduce<false,B,T>(h,p.uplo,p.n,A,p.a.data(),tau,de,resources,work,bytes,host,host_bytes);
#if defined(DDLA_GPUBLAS_HAS_HEURISTICS_CACHE) && DDLA_GPUBLAS_HAS_HEURISTICS_CACHE
        if constexpr(B==DdlaBackend::GPU)detail::agree_local(h,"one-stage eigensolver",[&]{resources.end_reduction();});
#endif
    } catch(...) {
        if constexpr(B==DdlaBackend::GPU){try{(void)ddlaSynchronize(h);}catch(...){}}
        throw;
    }
}
} // namespace spectral_impl

template<DdlaBackend B,class T>void phetrd(const DdlaHandle_t& h,char uplo,int n,T* A,const int* a,
    typename real_type<T>::type* d,typename real_type<T>::type* e,T* tau,int& info) {
    using R=typename real_type<T>::type;info=0;
    const auto p=spectral_impl::phetrd_check<T>(h,B,uplo,n,A,a,d,e,tau);
    if(!n)return;
    std::size_t bytes=0,host_bytes=0;
    vector<char> work(h);std::vector<char> host;std::vector<R> de;
    std::optional<spectral_impl::HetrdResources> resources;
    detail::agree_check(h,"phetrd allocation",[&](detail::Agree&){
        spectral_impl::phetrd_workspace_size<T>(h,p,&bytes,&host_bytes);
        work.resize(bytes);host.resize(host_bytes);de.resize(std::size_t(2)*n);resources.emplace(h,p);
    });
    spectral_impl::phetrd_run<B,T>(h,p,*resources,A,tau,de.data(),info,work.data(),bytes,host.data(),host_bytes);
    std::copy_n(de.data(),n,d);if(n>1)std::copy_n(de.data()+n,n-1,e);
}
#define DDLA_HETRD(B,T) \
 template void phetrd<B,T>(const DdlaHandle_t&,char,int,T*,const int*,typename real_type<T>::type*,typename real_type<T>::type*,T*,int&); \
 template void spectral_impl::phetrd_run<B,T>(const DdlaHandle_t&,const spectral_impl::HetrdPlan&,spectral_impl::HetrdResources&,T*,T*,typename real_type<T>::type*,int&,void*,std::size_t,void*,std::size_t)
DDLA_FOR_EACH_BACKEND_TYPE(DDLA_HETRD)
#define DDLA_HETRD_PLAN(T) \
 template spectral_impl::HetrdPlan spectral_impl::phetrd_check<T>(const DdlaHandle_t&,DdlaBackend,char,int,const T*,const int*,const typename real_type<T>::type*,const typename real_type<T>::type*,const T*); \
 template void spectral_impl::phetrd_workspace_size<T>(const DdlaHandle_t&,const spectral_impl::HetrdPlan&,std::size_t*,std::size_t*)
DDLA_FOR_EACH_TYPE(DDLA_HETRD_PLAN)
#undef DDLA_HETRD_PLAN
#undef DDLA_HETRD
} // namespace ddla
