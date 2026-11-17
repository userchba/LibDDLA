#include "ddla_connector.h"
// ---------------------------------------------------------------------------
// Backend reflector construction and GPU layout kernels for the pgeqrf panel.
// The panel itself is factored by geqrf; these helpers prepare its layout:
//
//   qr_reverse    -- local J-reversal for GPU geqrf with uplo = 'L'
//   qr_build_v    -- the explicit V (unit entries and zeros planted) from the window or in place
//   qr_adjoint_T  -- T -> T^H for the stage-1 hook
//
// Precedent: src/pgetf2_kernels.cpp (launcher + kernel split, explicit instantiation for the four
// scalar types).
// ---------------------------------------------------------------------------
#include "pgeqrf_kernels.h"
#include "ddla_stream.h"
#include "instantiate.h"
#include "host_parallel.h"

#include <algorithm>

namespace ddla {
namespace detail {

#if DDLA_HAS_GPU

namespace {

constexpr int kQrThreads = 128;
constexpr int kQrMaxBlocks = 1024;
// The fixed-order reduction trees halve the stride down from blockDim and the
// 2D launch grids divide row ranges by blockDim: both need a power of two.
static_assert(kQrThreads > 0 && (kQrThreads & (kQrThreads - 1)) == 0,
              "kQrThreads must be a power of two");

// Window element (r, col) of a <rt, Cplx>-templated panel: complex types are
// interleaved (rt pairs), real types plain.
template <bool Cplx>
DDLA_QR_HD inline std::size_t qr_elem(int r, int col, int ldw)
{
    const std::size_t e = static_cast<std::size_t>(r)
                        + static_cast<std::size_t>(col) * ldw;
    return Cplx ? 2 * e : e;
}

// Local row/column reversal for vendor QL (A' = J A J). Distributed
// panel packing and block-cyclic mapping live in matrix_transfer instead.
template <typename T>
__global__ void qr_reverse_kernel(int m, int n, const T* src, int ldsrc, T* dst, int lddst)
{
    for (int c = blockIdx.y; c < n; c += gridDim.y)
        for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < m;
             r += gridDim.x * blockDim.x)
            dst[std::size_t(m - 1 - r) + std::size_t(n - 1 - c) * lddst] =
                src[std::size_t(r) + std::size_t(c) * ldsrc];
}

// ---------------------------------------------------------------------------
// qr_build_v: explicit native V from a panel window (or in place from the
// freshly received window).  The window is in LAPACK storage, natural order.
// rev == 0 (QR, unit-lower V): entry (i, c) is 1 on the global diagonal
// (qr_l2g(lr0 + i) == j0 + c), 0 above it, the window value (g - j0, c) below
// it.  rev == 1 (QL, unit-UPPER V): the unit entry of column c sits at global
// row j0 + m_w - jb + c, 0 below it, the window value (g - j0, c) above it.
// in_place: source row i is local; src can alias v or be a separate panel.
// 2D grid (row block, column).
// ---------------------------------------------------------------------------
template <typename rt, bool Cplx>
__global__ void qr_build_v_kernel(int in_place, int rev, rt* v, int ldv, const rt* src,
                                  int ldsrc, int lrp, int jb, int j0, int m_w, int lr0,
                                  int mb, int myprow, int rsrc, int nprows)
{
    for (int c = blockIdx.y; c < jb; c += gridDim.y) {
        const int piv = rev ? j0 + m_w - jb + c : j0 + c;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < lrp;
             i += gridDim.x * blockDim.x) {
            const int g = qr_l2g(lr0 + i, mb, myprow, rsrc, nprows);
            const int d = g - j0;
            const int srow = in_place ? i : d;
            const int scol = c;
            rt* const p = v + qr_elem<Cplx>(i, c, ldv);
            const rt* const q = src + qr_elem<Cplx>(srow, scol, ldsrc);
            if (g == piv) {
                p[0] = rt(1);
                if constexpr (Cplx) p[1] = rt(0);
            } else if (rev ? g > piv : g < piv) {
                p[0] = rt(0);
                if constexpr (Cplx) p[1] = rt(0);
            } else {
                p[0] = q[0];
                if constexpr (Cplx) p[1] = q[1];
            }
        }
    }
}

// qr_adjoint_T: out(r, c) = conj(in(c, r)), r, c < jb.  One block; every output element is
// written by exactly one thread.  (A complex zero imaginary part becomes -0.0, which no
// consumer distinguishes.)
template <typename rt, bool Cplx>
__global__ void qr_adjoint_T_kernel(rt* out, int ldout, const rt* in, int jb)
{
    for (int idx = threadIdx.x; idx < jb * jb; idx += blockDim.x) {
        const int r = idx % jb, c = idx / jb;
        const rt* const q = in + qr_elem<Cplx>(c, r, jb);
        rt* const p = out + qr_elem<Cplx>(r, c, ldout);
        p[0] = q[0];
        if constexpr (Cplx) p[1] = -q[1];
    }
}


} // namespace

// --- launcher wrappers ------------------------------------------------------

// 2D launch geometry shared by the row-permutation kernels: blocks_x covers
// the rows, blocks_y the columns (capped, with an in-kernel stride loop as
// the gridDim.y guard).
static dim3 qr_rowcol_grid(int rows, int cols)
{
    const int bx = static_cast<int>(std::min<long long>(
        (static_cast<long long>(rows) + kQrThreads - 1) / kQrThreads,
        kQrMaxBlocks));
    const int by = cols < 65535 ? cols : 65535;
    return dim3(bx > 0 ? bx : 1, by > 0 ? by : 1);
}

template <typename T>
void qr_reverse(const DdlaHandle_t& handle, int m, int n, const T* src, int ldsrc, T* dst, int lddst)
{
    if (m <= 0 || n <= 0) return;
    qr_reverse_kernel<T><<<qr_rowcol_grid(m, n), kQrThreads, 0, static_cast<gpuStream_t>(handle->stream)>>>(m, n, src, ldsrc, dst, lddst);
    DDLA_CHECK(ddlaGetLastError(handle));
}

template <typename T>
void qr_adjoint_T(const DdlaHandle_t& handle, T* out, int ldout, const T* in, int jb)
{
    const gpuStream_t stream = static_cast<gpuStream_t>(handle->stream);
    using rt = typename real_type<T>::type;
    constexpr bool Cplx = !std::is_same<rt, T>::value;
    if (jb <= 0) return;
    qr_adjoint_T_kernel<rt, Cplx><<<1, kQrThreads, 0, stream>>>(
        reinterpret_cast<rt*>(out), ldout, reinterpret_cast<const rt*>(in), jb);
    DDLA_CHECK(ddlaGetLastError(handle));
}

// Explicit instantiations for the four scalar types.
#define DDLA_QR_INST_MAP_KERNELS(T)                                                \
    template void qr_reverse<T>(const DdlaHandle_t&, int, int, const T*, int, T*, int); \
    template void qr_adjoint_T<T>(const DdlaHandle_t&, T*, int, const T*, int)

DDLA_FOR_EACH_TYPE(DDLA_QR_INST_MAP_KERNELS)

#undef DDLA_QR_INST_MAP_KERNELS

#endif // DDLA_HAS_GPU


template <DdlaBackend Backend, typename T>
void qr_build_v(const DdlaHandle_t& handle, bool local_source, int rev, T* v, int ldv,
                const T* src, int ldsrc, int lrp, int jb, int j0, int m_w, int lr0,
                int mb, int myprow, int rsrc, int nprows)
{
    if (lrp <= 0 || jb <= 0) return;
    if constexpr (Backend == DdlaBackend::CPU) {
        // Step the block-cyclic rows without division inside the element loop.
        const int cycle = nprows * mb;
        const int shift = ((nprows + myprow - rsrc) % nprows) * mb;
        const int first_offset = lr0 % mb;
        const int first_base = cycle * (lr0 / mb) + shift;
        const bool copy_values = !local_source || src != v || ldsrc != ldv;
        const auto build_columns = [=](long long first, long long last) {
            for (int c = static_cast<int>(first); c < last; ++c) {
                const int pivot = rev ? j0 + m_w - jb + c : j0 + c;
                T* const column = v + std::size_t(c) * ldv;
                const T* const source = copy_values ? src + std::size_t(c) * ldsrc : nullptr;
                int offset = first_offset;
                int row_base = first_base;
                for (int i = 0; i < lrp; ++i) {
                    const int global_row = row_base + offset;
                    if (++offset == mb) {
                        offset = 0;
                        row_base += cycle;
                    }
                    if (global_row == pivot) column[i] = T(1);
                    else if (rev ? global_row > pivot : global_row < pivot) column[i] = T(0);
                    else if (copy_values) column[i] = source[local_source ? i : global_row - j0];
                }
            }
        };
        const int threads = std::max(1, handle->host_threads);
        const long long bytes = static_cast<long long>(lrp) * jb * sizeof(T);
        // Avoid team startup for small panels, and keep every worker occupied.
        if (threads > 1 && jb >= threads && bytes >= 1024 * 1024) {
            host_parallel_for(jb, 1, threads, build_columns);
        } else {
            build_columns(0, jb);
        }
    } else {
#if DDLA_HAS_GPU
        using rt = typename real_type<T>::type;
        constexpr bool Cplx = !std::is_same<rt, T>::value;
        const auto stream = static_cast<gpuStream_t>(handle->stream);
        qr_build_v_kernel<rt, Cplx><<<qr_rowcol_grid(lrp, jb), kQrThreads, 0, stream>>>(
            local_source ? 1 : 0, rev, reinterpret_cast<rt*>(v), ldv,
            reinterpret_cast<const rt*>(src), ldsrc, lrp, jb, j0, m_w, lr0, mb,
            myprow, rsrc, nprows);
        DDLA_CHECK(ddlaGetLastError(handle));
#endif
    }
}

#define INSTANTIATE_BUILD_V(BACKEND, TYPE) \
    template void qr_build_v<BACKEND, TYPE>(const DdlaHandle_t&, bool, int, TYPE*, int, \
        const TYPE*, int, int, int, int, int, int, int, int, int, int)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_BUILD_V)
#undef INSTANTIATE_BUILD_V

} // namespace detail
} // namespace ddla
