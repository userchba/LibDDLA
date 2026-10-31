#ifndef DDLA_GEMM_MIXED_CPU_H
#define DDLA_GEMM_MIXED_CPU_H
#include <ddla/ddla.h>
#include "workspace.h"
namespace ddla { namespace detail {
struct MixedGemmWorkspaceSize {
    WorkspaceSize size{};
    int threads = 1;
    int mb = 0, nb = 0, kb = 0, nr = 0, nc = 0;
    std::size_t pack_size = 0, result_size = 0;
};
template <typename R>
MixedGemmWorkspaceSize mixed_gemm_cpu_workspace_size(const DdlaHandle_t&, int, int, int, bool complex_left, bool product);
// Checked CPU convenience calls preserve the historical caller-thread scratch
// cache. Host-only ownership is independent of handle lifetime and is acquired
// outside check/query/run; explicit caller workspace remains fully supported.
template <typename R> BufferView<R> mixed_gemm_cpu_cached_workspace(std::size_t count);
// Host-compiled blocked real-BLAS path, including in a CPU+GPU build. No communication;
// the handle supplies the thread budget established by ddlaSet.
template <typename R>
void mixed_gemm_cpu(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
                    std::complex<R> alpha, const R* A, int lda,
                    const std::complex<R>* B, int ldb, std::complex<R> beta,
                    std::complex<R>* C, int ldc, bool product, BufferView<R>, const MixedGemmWorkspaceSize&);
template <typename R>
void mixed_gemm_cpu(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
                    std::complex<R> alpha, const std::complex<R>* A, int lda,
                    const R* B, int ldb, std::complex<R> beta,
                    std::complex<R>* C, int ldc, bool product, BufferView<R>, const MixedGemmWorkspaceSize&);
}}
#endif
