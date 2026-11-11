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
        for (int j=0;j<n;++j) {
            for (int i=0;i<m;++i) {
                const int diagonal_index=side=='L' ? i : j;
                C[i+std::size_t(j)*ldc]=A[i+std::size_t(j)*lda]
                    *x[first+std::ptrdiff_t(diagonal_index)*incx];
            }
        }
    } else {
#if DDLA_GPU_VENDOR
        const auto handle=static_cast<gpublasHandle_t>(h->blasH);
        if (length==1) incx=0;
        if constexpr (!std::is_same_v<T,U>) {
            // A real diagonal scales both components equally. Right scaling
            // views the full matrix as 2m real rows. Left scaling views each
            // column as a 2-by-m real matrix and scales its columns.
            if (side=='R' && lda<=std::numeric_limits<int>::max()/2 &&
                ldc<=std::numeric_limits<int>::max()/2) {
                dgmm_run<Backend,U,U>(h,'R',2*m,n,reinterpret_cast<const U*>(A),2*lda,
                    x,incx,reinterpret_cast<U*>(C),2*ldc);
            } else {
                for (int j=0;j<n;++j) {
                    const U* diagonal=side=='L' ? x : x+first+std::ptrdiff_t(j)*incx;
                    dgmm_run<Backend,U,U>(h,'R',2,m,
                        reinterpret_cast<const U*>(A+std::size_t(j)*lda),2,
                        diagonal,side=='L' ? incx : 0,
                        reinterpret_cast<U*>(C+std::size_t(j)*ldc),2);
                }
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
