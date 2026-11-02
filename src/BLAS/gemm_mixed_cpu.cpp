#include "blas_impl.h"
#include "plan_util.h"
#include "gemm_mixed_cpu.h"
#include "ddla_stream.h"
#include "host_threads.h"

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstddef>
#include <memory>
#include <type_traits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace ddla { namespace detail {
namespace {

// Each worker owns complete output tiles, including conversion and the final
// alpha/beta update. Real BLAS supplies its tuned microkernel and packing; its
// internal team is held at one while the outer OpenMP team supplies parallelism.
// Scratch is bounded per worker, independent of the full matrix dimensions.
// The longer tile dimension reduces duplication of the complex-operand pack.
constexpr int mixed_short = 128, mixed_long = 512, mixed_kb = 512;

// RC: A [Br | Bi] = [Cr | Ci]. The two real planes are adjacent column panels,
// so one real GEMM reuses A for both components.
template <typename R>
void pack_right(char trans, int kk, int nn, const std::complex<R>* B, int ldb,
                R* packed)
{
    const bool conjugate = trans == 'C';
    for (int j = 0; j < nn; ++j) {
        R* re = packed + std::size_t(j)*kk;
        R* im = re + std::size_t(nn)*kk;
        for (int l = 0; l < kk; ++l) {
            const auto z = trans == 'N' ? B[l + std::size_t(j)*ldb]
                                       : B[j + std::size_t(l)*ldb];
            re[l] = z.real();
            im[l] = conjugate ? -z.imag() : z.imag();
        }
    }
}

// CR: [Ar; Ai] B = [Cr; Ci]. Stacking rows similarly lets real BLAS reuse B.
template <typename R>
void pack_left(char trans, int mm, int kk, const std::complex<R>* A, int lda,
               R* packed)
{
    const bool conjugate = trans == 'C';
    for (int l = 0; l < kk; ++l) {
        R* re = packed + std::size_t(l)*(2*mm);
        R* im = re + mm;
        for (int i = 0; i < mm; ++i) {
            const auto z = trans == 'N' ? A[i + std::size_t(l)*lda]
                                       : A[l + std::size_t(i)*lda];
            re[i] = z.real();
            im[i] = conjugate ? -z.imag() : z.imag();
        }
    }
}

template <typename R, bool CR>
void combine(int mm, int nn, const R* result, std::complex<R> alpha,
             std::complex<R> beta, std::complex<R>* C, int ldc)
{
    const int ldr = CR ? 2*mm : mm;
    const std::size_t imaginary = CR ? std::size_t(mm) : std::size_t(mm)*nn;
    const bool unit_alpha = alpha == R(1), zero_beta = beta == R(0);
    for (int j = 0; j < nn; ++j) {
        const R* re = result + std::size_t(j)*ldr;
        const R* im = re + imaginary;
        auto* out = C + std::size_t(j)*ldc;
        if (unit_alpha && zero_beta) {
            for (int i = 0; i < mm; ++i)
                out[i] = {re[i], im[i]};
        } else {
            for (int i = 0; i < mm; ++i) {
                const auto z = alpha * std::complex<R>(re[i], im[i]);
                // In particular, beta=0 must not read NaN-poisoned C.
                out[i] = zero_beta ? z : z + beta*out[i];
            }
        }
    }
}

// The tiling of one product, derived identically by the size and the run: the
// worker team, mb x nb output tiles over an nr x nc grid, kb-deep panels, and
// each worker's slot [ packed operand (pack_size) | real result (result_size) ].
void mixed_tiles(const DdlaHandle_t& h, int m, int n, int k, bool cr,
                 int& threads, int& mb, int& nb, int& kb, int& nr, int& nc,
                 std::size_t& pack_size, std::size_t& result_size)
{
    threads = 1;
    const int budget = std::max(1, h->host_threads);
#ifdef _OPENMP
    if (double(m)*n*k >= double(1 << 18)) threads = budget;
#else
    (void)budget;
#endif
    mb = std::min(m, cr ? mixed_short : mixed_long);
    nb = std::min(n, cr ? mixed_long : mixed_short);
    auto rows = [&] { return 1 + (m-1)/mb; };
    auto cols = [&] { return 1 + (n-1)/nb; };
    // Small matrices still need enough independent tiles for the entire team.
    // Count after each split; do not clamp a thread-grid dimension and silently
    // discard half the available workers. Rebalance the edge tiles afterwards.
    while (static_cast<long long>(rows())*cols() < threads && (mb > 1 || nb > 1)) {
        if ((mb >= nb && mb > 1) || nb <= 1) mb = (mb+1)/2;
        else nb = (nb+1)/2;
    }
    mb = 1 + (m-1)/rows();
    nb = 1 + (n-1)/cols();
    nr = rows(); nc = cols();
    threads = static_cast<int>(std::min<long long>(threads, static_cast<long long>(nr)*nc));
    // A single worker can use a deeper panel without multiplying the shared-cache footprint.
    kb = std::min(k, threads == 1 ? 2*mixed_kb : mixed_kb);
    pack_size = std::size_t(2)*kb*(cr ? mb : nb);
    result_size = std::size_t(2)*mb*nb;
}

template <typename R, bool CR>
void mixed_gemm(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
                std::complex<R> alpha,
                const std::conditional_t<CR, std::complex<R>, R>* A, int lda,
                const std::conditional_t<CR, R, std::complex<R>>* B, int ldb,
                std::complex<R> beta, std::complex<R>* C, int ldc, bool product,
                void* work, [[maybe_unused]] std::size_t bytes)
{
    const int budget = std::max(1, h->host_threads);
    if (!product) {
        if (beta == R(1)) return;
        const bool zero = beta == R(0);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(budget)
#endif
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                auto& c = C[i + std::size_t(j)*ldc];
                c = zero ? std::complex<R>(0) : beta*c;
            }
        return;
    }

    int threads, mb, nb, kb, nr, nc;
    std::size_t pack_size, result_size;
    mixed_tiles(h, m, n, k, CR, threads, mb, nb, kb, nr, nc, pack_size, result_size);
    const long long tiles = static_cast<long long>(nr) * nc;
    // [ worker slots: threads x (packed operand | real result) ]
    char* w = static_cast<char*>(work);
    R* const scratch = take<R>(w, static_cast<std::size_t>(threads) * (pack_size + result_size));
    assert(static_cast<std::size_t>(w - static_cast<char*>(work)) <= bytes);
    ScopedBlasThreads serial_blas(1);
#ifdef _OPENMP
    #pragma omp parallel num_threads(threads) if(threads > 1)
#endif
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        R* const packed = scratch + static_cast<std::size_t>(tid) * (pack_size + result_size);
        R* const result = packed + pack_size;
        // A work-sharing loop remains correct with dynamic or serialized nested
        // teams: all logical tiles execute even if fewer workers are granted.
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (long long tile = 0; tile < tiles; ++tile) {
            const int i0 = static_cast<int>(tile%nr)*mb;
            const int j0 = static_cast<int>(tile/nr)*nb;
            const int mm = std::min(mb, m-i0), nn = std::min(nb, n-j0);
            for (int l0 = 0; l0 < k;) {
                const int kk = std::min(kb, k-l0);
                const R accumulate = l0 == 0 ? R(0) : R(1);
                const auto* ap = A + (ta == 'N' ? std::size_t(i0) + std::size_t(l0)*lda
                                               : std::size_t(l0) + std::size_t(i0)*lda);
                const auto* bp = B + (tb == 'N' ? std::size_t(l0) + std::size_t(j0)*ldb
                                               : std::size_t(j0) + std::size_t(l0)*ldb);
                if constexpr (CR) {
                    pack_left(ta, mm, kk, ap, lda, packed);
                    blas_impl::gemm_run<DdlaBackend::CPU, R>(h, 'N', tb, 2*mm, nn, kk,
                        R(1), packed, 2*mm, bp, ldb, accumulate, result, 2*mm);
                } else {
                    pack_right(tb, kk, nn, bp, ldb, packed);
                    blas_impl::gemm_run<DdlaBackend::CPU, R>(h, ta, 'N', mm, 2*nn, kk,
                        R(1), ap, lda, packed, kk, accumulate, result, mm);
                }
                l0 += kk;
            }
            combine<R, CR>(mm, nn, result, alpha, beta,
                           C + i0 + std::size_t(j0)*ldc, ldc);
        }
    }
}

} // namespace

char* mixed_gemm_cpu_cached_workspace(std::size_t bytes)
{
    // No handle or asynchronous GPU use is retained in TLS.
    struct HostScratch { std::unique_ptr<char[]> data; std::size_t capacity=0; };
    thread_local HostScratch scratch;
    if (bytes>scratch.capacity) {
        std::unique_ptr<char[]> next(new char[bytes]);
        scratch.data=std::move(next);scratch.capacity=bytes;
    }
    return bytes ? scratch.data.get() : nullptr;
}

// [ worker slots: threads x (packed operand | real result) ], one region.
template <typename R>
void mixed_gemm_cpu_workspace_size(const DdlaHandle_t& h, int m, int n, int k, bool cr, std::size_t* backend_bytes)
{
    *backend_bytes = 0;
    if (m <= 0 || n <= 0 || k <= 0) return;
    int threads, mb, nb, kb, nr, nc;
    std::size_t pack_size, result_size;
    mixed_tiles(h, m, n, k, cr, threads, mb, nb, kb, nr, nc, pack_size, result_size);
    *backend_bytes = padded_bytes<R>(static_cast<std::size_t>(threads) * (pack_size + result_size));
}

template <typename R>
void mixed_gemm_cpu(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
                    std::complex<R> alpha, const R* A, int lda,
                    const std::complex<R>* B, int ldb, std::complex<R> beta,
                    std::complex<R>* C, int ldc, bool product, void* work, std::size_t bytes)
{
    mixed_gemm<R, false>(h,ta,tb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,product,work,bytes);
}
template <typename R>
void mixed_gemm_cpu(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
                    std::complex<R> alpha, const std::complex<R>* A, int lda,
                    const R* B, int ldb, std::complex<R> beta,
                    std::complex<R>* C, int ldc, bool product, void* work, std::size_t bytes)
{
    mixed_gemm<R, true>(h,ta,tb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,product,work,bytes);
}
#define INSTANTIATE_MIXED_CPU(R) \
    template void mixed_gemm_cpu<R>(const DdlaHandle_t&, char,char,int,int,int, \
        std::complex<R>,const R*,int,const std::complex<R>*,int,std::complex<R>,std::complex<R>*,int,bool,void*,std::size_t); \
    template void mixed_gemm_cpu<R>(const DdlaHandle_t&, char,char,int,int,int, \
        std::complex<R>,const std::complex<R>*,int,const R*,int,std::complex<R>,std::complex<R>*,int,bool,void*,std::size_t); \
    template void mixed_gemm_cpu_workspace_size<R>(const DdlaHandle_t&,int,int,int,bool,std::size_t*)
INSTANTIATE_MIXED_CPU(float);
INSTANTIATE_MIXED_CPU(double);
#undef INSTANTIATE_MIXED_CPU
}} // namespace ddla::detail
