#include "blas_impl.h"
#include "helpers.h"
#include "require_backend.h"
#include "ddla_stream.h"
#include "instantiate.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace ddla { namespace blas_impl {

#if DDLA_HAS_GPU
namespace {
// Vendor DGMM supports a single scalar type. A real diagonal scales each
// component of a complex element independently, including infinities and NaNs.
template <typename R>
__global__ void mixed_dgmm(bool left, int m, int n, const R* a, int lda,
                           const R* diagonal, int incx, R* c, int ldc)
{
    for (unsigned j = blockIdx.y * blockDim.y + threadIdx.y; j < unsigned(n);
         j += gridDim.y * blockDim.y) {
        for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < unsigned(m);
             i += gridDim.x * blockDim.x) {
            const R scale = diagonal[std::ptrdiff_t(left ? i : j) * incx];
            const auto src = 2 * (std::size_t(j) * lda + i);
            const auto dst = 2 * (std::size_t(j) * ldc + i);
            c[dst]     = a[src]     * scale;
            c[dst + 1] = a[src + 1] * scale;
        }
    }
}
} // namespace
#endif

template <DdlaBackend Backend, typename T, typename U>
void dgmm_check(const DdlaHandle_t& h, char side, int m, int n,
    const T* A, int lda, const U* x, int incx, T* C, int ldc, int diagonal_length)
{
    static_assert(std::is_same_v<U,T> ||
        std::is_same_v<U,typename real_type<T>::type>, "dgmm: incompatible diagonal type");
    detail::require_backend<Backend>(h,"dgmm");
    side=upper(side);
    if (side!='L' && side!='R')
        throw std::invalid_argument("dgmm: side must be L or R");
    if (m<0 || n<0 || lda<std::max(1,m) || ldc<std::max(1,m))
        throw std::invalid_argument("dgmm: invalid dimensions or leading dimensions");
    if (m==0 || n==0) return;
    if (!A || !x || !C)
        throw std::invalid_argument("dgmm: null active matrix or diagonal");
    const std::size_t a_count=std::size_t(n-1)*lda+m;
    const std::size_t c_count=std::size_t(n-1)*ldc+m;
    const std::size_t step=incx<0 ? -std::ptrdiff_t(incx) : incx;
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if (diagonal_length<1 || a_count>limit/sizeof(T) || c_count>limit/sizeof(T) ||
        (diagonal_length>1 && step>(limit/sizeof(U)-1)/std::size_t(diagonal_length-1)))
        throw std::invalid_argument("dgmm: matrix or diagonal span exceeds addressable storage");
    const std::size_t x_count=1+std::size_t(diagonal_length-1)*step;
    if ((A==C && lda!=ldc) ||
        (A!=C && detail::matrix_overlaps(A,lda,m,n,C,ldc,m,n)))
        throw std::invalid_argument("dgmm: matrices must be disjoint or identical with equal pitches");
    if (detail::memory_overlaps(x,x_count*sizeof(U),C,c_count*sizeof(T)))
        throw std::invalid_argument("dgmm: diagonal must not overlap the output");
}

template <DdlaBackend Backend, typename T, typename U>
void dgmm_run(const DdlaHandle_t& h, char side, int m, int n,
    const T* A, int lda, const U* x, int incx, T* C, int ldc)
{
    if (m==0 || n==0) return;
    side=upper(side);
    const int length=side=='L' ? m : n;
    const std::ptrdiff_t first=incx<0 ? std::ptrdiff_t(length-1)*-std::ptrdiff_t(incx) : 0;
    if constexpr (Backend==DdlaBackend::CPU) {
        const U* diagonal = x + first;
        const auto scale_columns = [&](int begin, int end) {
            for (int j = begin; j < end; ++j) {
                const T* input = A + std::size_t(j) * lda;
                T* output = C + std::size_t(j) * ldc;
                if (side == 'L') {
                    for (int i = 0; i < m; ++i)
                        output[i] = input[i] * diagonal[std::ptrdiff_t(i) * incx];
                } else {
                    const U scale = diagonal[std::ptrdiff_t(j) * incx];
                    for (int i = 0; i < m; ++i)
                        output[i] = input[i] * scale;
                }
            }
        };
        dgmm_columns(m, n, h->host_threads, scale_columns);
    } else {
#if DDLA_HAS_GPU
        const auto handle=static_cast<gpublasHandle_t>(h->blasH);
        if (length==1) incx=0;
        if constexpr (!std::is_same_v<T,U>) {
            // Right scaling can treat each complex column as 2m real rows.
            if (side=='R' && lda<=std::numeric_limits<int>::max()/2 &&
                ldc<=std::numeric_limits<int>::max()/2) {
                dgmm_run<Backend,U,U>(h,'R',2*m,n,reinterpret_cast<const U*>(A),2*lda,
                    x,incx,reinterpret_cast<U*>(C),2*ldc);
            } else {
                const dim3 block(32, 8);
                const dim3 grid(std::min(65535u, (unsigned(m) + block.x - 1) / block.x),
                                std::min(65535u, (unsigned(n) + block.y - 1) / block.y));
                mixed_dgmm<<<grid, block, 0, static_cast<gpuStream_t>(h->stream)>>>(
                    side == 'L', m, n, reinterpret_cast<const U*>(A), lda,
                    x + first, incx, reinterpret_cast<U*>(C), ldc);
                DDLA_CHECK(ddlaGetLastError(h));
            }
        } else if constexpr (std::is_same_v<T,float>) {
            BLAS_CHECK(gpublasSdgmm(handle,blas_impl::side(side),m,n,A,lda,x,incx,C,ldc));
        } else if constexpr (std::is_same_v<T,double>) {
            BLAS_CHECK(gpublasDdgmm(handle,blas_impl::side(side),m,n,A,lda,x,incx,C,ldc));
        } else if constexpr (std::is_same_v<T,std::complex<float>>) {
            using D=gpublasComplex;
            BLAS_CHECK(gpublasCdgmm(handle,blas_impl::side(side),m,n,
                reinterpret_cast<const D*>(A),lda,reinterpret_cast<const D*>(x),incx,
                reinterpret_cast<D*>(C),ldc));
        } else if constexpr (std::is_same_v<T,std::complex<double>>) {
            using D=gpublasDoubleComplex;
            BLAS_CHECK(gpublasZdgmm(handle,blas_impl::side(side),m,n,
                reinterpret_cast<const D*>(A),lda,reinterpret_cast<const D*>(x),incx,
                reinterpret_cast<D*>(C),ldc));
        }
#endif
    }
}

}} // namespace ddla::blas_impl

namespace ddla {
template <DdlaBackend Backend, typename T, typename U>
void dgmm(const DdlaHandle_t& h, char side, int m, int n,
    const T* A, int lda, const U* x, int incx, T* C, int ldc)
{
    blas_impl::dgmm_check<Backend,T,U>(h,side,m,n,A,lda,x,incx,C,ldc,
        blas_impl::upper(side)=='L' ? m : n);
    blas_impl::dgmm_run<Backend,T,U>(h,side,m,n,A,lda,x,incx,C,ldc);
}

#define INSTANTIATE(B,T,U) \
template void dgmm<B,T,U>(const DdlaHandle_t&,char,int,int,const T*,int,const U*,int,T*,int); \
template void blas_impl::dgmm_check<B,T,U>(const DdlaHandle_t&,char,int,int,const T*,int,const U*,int,T*,int,int); \
template void blas_impl::dgmm_run<B,T,U>(const DdlaHandle_t&,char,int,int,const T*,int,const U*,int,T*,int)
#define TYPES(B) \
INSTANTIATE(B,float,float); INSTANTIATE(B,double,double); \
INSTANTIATE(B,std::complex<float>,std::complex<float>); \
INSTANTIATE(B,std::complex<double>,std::complex<double>); \
INSTANTIATE(B,std::complex<float>,float); INSTANTIATE(B,std::complex<double>,double)
DDLA_FOR_EACH_BACKEND(TYPES)
#undef TYPES
#undef INSTANTIATE
} // namespace ddla
