#include "ddla_connector.h"
#include "blas_impl.h"
#include <ddla/ddla.h>
#include "blas_enums.h"
#include "ddla_stream.h"
#include "require_backend.h"
#include <type_traits>

#if DDLA_HAS_CPU
extern "C" {
void csymm_(const char*, const char*, const int*, const int*, const std::complex<float>*, const std::complex<float>*, const int*, const std::complex<float>*, const int*, const std::complex<float>*, std::complex<float>*, const int*);
void zsymm_(const char*, const char*, const int*, const int*, const std::complex<double>*, const std::complex<double>*, const int*, const std::complex<double>*, const int*, const std::complex<double>*, std::complex<double>*, const int*);
}
#endif

namespace ddla {
namespace blas_impl {
template <DdlaBackend Backend, typename T>
void symm_check(const DdlaHandle_t& h, char side, char uplo, int m, int n,
    const T& alpha, const T* A, int lda, const T* B, int ldb, const T& beta, T* C, int ldc)
{
    detail::require_backend<Backend>(h,"symm");
    if constexpr (std::is_arithmetic_v<T>) blas_impl::hemm_check<Backend,T>(h,side,uplo,m,n,alpha,A,lda,B,ldb,beta,C,ldc);
    if constexpr (Backend == DdlaBackend::GPU) {
        (void)detail::to_fill_mode(uplo);
        (void)detail::to_side_mode(side);
    }
}
template <DdlaBackend Backend, typename T>
void symm_run(const DdlaHandle_t& h, char side, char uplo, int m, int n,
    const T& alpha, const T* A, int lda, const T* B, int ldb, const T& beta, T* C, int ldc)
{
    if constexpr (std::is_arithmetic_v<T>) {
        blas_impl::hemm_run<Backend,T>(h,side,uplo,m,n,alpha,A,lda,B,ldb,beta,C,ldc);
    } else {
    if constexpr (Backend==DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        if constexpr (std::is_same_v<T,std::complex<float>>)
            csymm_(&side, &uplo,&m, &n,&alpha,A,&lda,B,&ldb,&beta,C,&ldc);
        else if constexpr (std::is_same_v<T,std::complex<double>>)
            zsymm_(&side, &uplo,&m, &n,&alpha,A,&lda,B,&ldb,&beta,C,&ldc);
#endif
    } else {
#if DDLA_GPU_VENDOR
        if constexpr (std::is_same_v<T,std::complex<float>>) {
            using D=gpublasComplex;
            BLAS_CHECK(gpublasCsymm(static_cast<gpublasHandle_t>(h->blasH),blas_impl::side(side),blas_impl::fill(uplo),m,n,
                reinterpret_cast<const D*>(&alpha),reinterpret_cast<const D*>(A),lda,
                reinterpret_cast<const D*>(B),ldb,
                reinterpret_cast<const D*>(&beta),reinterpret_cast<D*>(C),ldc));
        }
        else if constexpr (std::is_same_v<T,std::complex<double>>) {
            using D=gpublasDoubleComplex;
            BLAS_CHECK(gpublasZsymm(static_cast<gpublasHandle_t>(h->blasH),blas_impl::side(side),blas_impl::fill(uplo),m,n,
                reinterpret_cast<const D*>(&alpha),reinterpret_cast<const D*>(A),lda,
                reinterpret_cast<const D*>(B),ldb,
                reinterpret_cast<const D*>(&beta),reinterpret_cast<D*>(C),ldc));
        }
#endif
    }
    }
}
} // namespace blas_impl
template <DdlaBackend Backend, typename T>
void symm(const DdlaHandle_t& h, char side, char uplo, int m, int n,
    const T& alpha, const T* A, int lda, const T* B, int ldb, const T& beta, T* C, int ldc)
{
    blas_impl::symm_check<Backend,T>(h,side,uplo,m,n,alpha,A,lda,B,ldb,beta,C,ldc);
    blas_impl::symm_run<Backend,T>(h,side,uplo,m,n,alpha,A,lda,B,ldb,beta,C,ldc);
}

#define INSTANTIATE(B,T) template void symm<B,T>(const DdlaHandle_t&,char,char,int,int,const T&,const T*,int,const T*,int,const T&,T*,int); \
    template void blas_impl::symm_check<B,T>(const DdlaHandle_t&,char,char,int,int,const T&,const T*,int,const T*,int,const T&,T*,int); \
    template void blas_impl::symm_run<B,T>(const DdlaHandle_t&,char,char,int,int,const T&,const T*,int,const T*,int,const T&,T*,int);
#define TYPES(B) INSTANTIATE(B,float) INSTANTIATE(B,double) \
    INSTANTIATE(B,std::complex<float>) INSTANTIATE(B,std::complex<double>)
#if DDLA_HAS_CPU
TYPES(DdlaBackend::CPU)
#endif
#if DDLA_HAS_GPU
TYPES(DdlaBackend::GPU)
#endif
#undef TYPES
#undef INSTANTIATE
} // namespace ddla
