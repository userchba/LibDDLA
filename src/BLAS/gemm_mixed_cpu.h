#ifndef DDLA_GEMM_MIXED_CPU_H
#define DDLA_GEMM_MIXED_CPU_H
#include <ddla/ddla.h>
#include <cstddef>
namespace ddla { namespace detail {
// Existing planar packing/combine operations, shared with CPU mixed batches.
template <typename R>
void mixed_pack_left(char trans,int m,int k,const std::complex<R>* A,int lda,R* packed);
template <typename R>
void mixed_pack_right(char trans,int k,int n,const std::complex<R>* B,int ldb,R* packed);
template <typename R,bool ComplexLeft>
void mixed_combine(int m,int n,const R* result,std::complex<R> alpha,
    std::complex<R> beta,std::complex<R>* C,int ldc);
// Bytes of the host-compiled path's scratch: the worker slots of the product's
// tiling, each [ packed operand | real result ]. The tiling depends on the
// handle's thread budget; m, n or k <= 0 needs nothing.
template <typename R>
void mixed_gemm_cpu_workspace_size(const DdlaHandle_t&, int m, int n, int k, bool complex_left,
                                   std::size_t* backend_bytes);
// The caller thread's cached scratch of at least `bytes`, for the checked CPU
// convenience calls: repeated products reuse it instead of paying the
// allocation and first touch of fresh memory per call. Host-only; no handle is
// retained.
char* mixed_gemm_cpu_cached_workspace(std::size_t bytes);
// Host-compiled blocked real-BLAS path, including in a CPU+GPU build. No communication;
// the handle supplies the thread budget established by ddlaSet.
template <typename R>
void mixed_gemm_cpu(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
                    std::complex<R> alpha, const R* A, int lda,
                    const std::complex<R>* B, int ldb, std::complex<R> beta,
                    std::complex<R>* C, int ldc, bool product, void* work, std::size_t bytes);
template <typename R>
void mixed_gemm_cpu(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
                    std::complex<R> alpha, const std::complex<R>* A, int lda,
                    const R* B, int ldb, std::complex<R> beta,
                    std::complex<R>* C, int ldc, bool product, void* work, std::size_t bytes);
}}
#endif
