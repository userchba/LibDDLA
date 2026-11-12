#include "blas_impl.h"
#include "ddla_stream.h"
#include "host_threads.h"
#include "require_backend.h"
#include "instantiate.h"

#include <algorithm>
#include <complex>
#include <stdexcept>

namespace ddla { namespace blas_impl {
template <DdlaBackend Backend, typename T, typename Alpha>
void scalBatched_run(const DdlaHandle_t& h, int m, int n, const Alpha& alpha,
                     T* const* arrays, int lda, int count)
{
    if (m==0 || n==0 || count==0 || alpha==T(1)) return;
    if constexpr (Backend==DdlaBackend::CPU) {
        detail::for_each_product(h->host_threads,count,[&](int q) {
            if constexpr(std::is_same_v<T,Alpha>)
                geam_run<Backend,T>(h,'N','N',m,n,alpha,arrays[q],lda,T(0),nullptr,1,arrays[q],lda);
            else matrix_scal_run<Backend,T,Alpha>(h,m,n,alpha,arrays[q],lda);
        });
    } else {
#if DDLA_HAS_GPU
        matrix_scal_gpu_run<T,Alpha>(h,m,n,alpha,nullptr,lda,arrays,count);
#endif
    }
}
}}

namespace ddla {
template <DdlaBackend Backend, typename T>
void scalBatched(const DdlaHandle_t& h, int m, int n, const T& alpha,
                 T* const arrays[], int lda, int count)
{
    detail::require_backend<Backend>(h,"scalBatched");
    if (m<0 || n<0 || count<0 || lda<std::max(1,m))
        throw std::invalid_argument("scalBatched: invalid dimensions or leading dimension");
    if (m==0 || n==0 || count==0) return;
    if (!arrays) throw std::invalid_argument("scalBatched: null pointer array");
    if constexpr (Backend==DdlaBackend::CPU) {
        for (int q=0; q<count; ++q)
            if (!arrays[q]) throw std::invalid_argument("scalBatched: null matrix");
    }
    blas_impl::scalBatched_run<Backend,T>(h,m,n,alpha,arrays,lda,count);
}

template <DdlaBackend Backend,typename T,typename Alpha>
typename std::enable_if<
    std::is_same<T, std::complex<Alpha>>::value,
    void
>::type
scalBatched(const DdlaHandle_t& h, int m, int n, const Alpha& alpha,
                 T* const arrays[], int lda, int count)
{
    detail::require_backend<Backend>(h,"scalBatched");
    if (m<0 || n<0 || count<0 || lda<std::max(1,m))
        throw std::invalid_argument("scalBatched: invalid dimensions or leading dimension");
    if (m==0 || n==0 || count==0) return;
    if (!arrays) throw std::invalid_argument("scalBatched: null pointer array");
    if constexpr (Backend==DdlaBackend::CPU) {
        for (int q=0; q<count; ++q)
            if (!arrays[q]) throw std::invalid_argument("scalBatched: null matrix");
    }
    blas_impl::scalBatched_run<Backend,T,Alpha>(h,m,n,alpha,arrays,lda,count);
}

#define MIXED_BATCH(B,R) \
 template void scalBatched<B,std::complex<R>,R>(const DdlaHandle_t&,int,int,const R&,std::complex<R>* const*,int,int); \
 template void blas_impl::scalBatched_run<B,std::complex<R>,R>(const DdlaHandle_t&,int,int,const R&,std::complex<R>* const*,int,int)
MIXED_BATCH(DdlaBackend::CPU,float);MIXED_BATCH(DdlaBackend::CPU,double);
#if DDLA_HAS_GPU
MIXED_BATCH(DdlaBackend::GPU,float);MIXED_BATCH(DdlaBackend::GPU,double);
#endif
#undef MIXED_BATCH
#define INSTANTIATE(B,T) \
    template void scalBatched<B,T>(const DdlaHandle_t&,int,int,const T&,T* const*,int,int); \
    template void blas_impl::scalBatched_run<B,T>(const DdlaHandle_t&,int,int,const T&,T* const*,int,int)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE)
#undef INSTANTIATE
}
