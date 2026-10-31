#include "ddla_connector.h"
#include "blas_impl.h"
#include <ddla/ddla.h>
#include "blas_enums.h"
#include "ddla_stream.h"
#include "require_backend.h"
#include <type_traits>

#if DDLA_HAS_CPU
extern "C" {
void ssyrk_(const char*, const char*, const int*, const int*, const float*, const float*, const int*, const float*, float*, const int*);
void dsyrk_(const char*, const char*, const int*, const int*, const double*, const double*, const int*, const double*, double*, const int*);
void csyrk_(const char*, const char*, const int*, const int*, const std::complex<float>*, const std::complex<float>*, const int*, const std::complex<float>*, std::complex<float>*, const int*);
void zsyrk_(const char*, const char*, const int*, const int*, const std::complex<double>*, const std::complex<double>*, const int*, const std::complex<double>*, std::complex<double>*, const int*);
}
#endif

namespace ddla {
namespace blas_impl {
template <DdlaBackend Backend, typename T>
void syrk_check(const DdlaHandle_t& h, char uplo, char trans, int n, int k,
    const T& alpha, const T* A, int lda, const T& beta, T* C, int ldc)
{
    detail::require_backend<Backend>(h,"syrk");
    if constexpr (Backend == DdlaBackend::GPU) {
        (void)detail::to_fill_mode(uplo);
        if constexpr (!std::is_arithmetic_v<T>) (void)detail::to_blas_op(trans);
    }
}
template <DdlaBackend Backend, typename T>
void syrk_run(const DdlaHandle_t& h, char uplo, char trans, int n, int k,
    const T& alpha, const T* A, int lda, const T& beta, T* C, int ldc)
{
    if constexpr (std::is_arithmetic_v<T>) trans=detail::trans_real(trans);
    if constexpr (Backend==DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        if constexpr (std::is_same_v<T,float>)
            ssyrk_(&uplo, &trans,&n, &k,&alpha,A,&lda,&beta,C,&ldc);
        else if constexpr (std::is_same_v<T,double>)
            dsyrk_(&uplo, &trans,&n, &k,&alpha,A,&lda,&beta,C,&ldc);
        else if constexpr (std::is_same_v<T,std::complex<float>>)
            csyrk_(&uplo, &trans,&n, &k,&alpha,A,&lda,&beta,C,&ldc);
        else if constexpr (std::is_same_v<T,std::complex<double>>)
            zsyrk_(&uplo, &trans,&n, &k,&alpha,A,&lda,&beta,C,&ldc);
#endif
    } else {
#if DDLA_GPU_VENDOR
        if constexpr (std::is_same_v<T,float>) {
            BLAS_CHECK(gpublasSsyrk(static_cast<gpublasHandle_t>(h->blasH),blas_impl::fill(uplo),blas_impl::op(trans),n,k,&alpha,A,lda,&beta,C,ldc));
        }
        else if constexpr (std::is_same_v<T,double>) {
            BLAS_CHECK(gpublasDsyrk(static_cast<gpublasHandle_t>(h->blasH),blas_impl::fill(uplo),blas_impl::op(trans),n,k,&alpha,A,lda,&beta,C,ldc));
        }
        else if constexpr (std::is_same_v<T,std::complex<float>>) {
            using D=gpublasComplex;
            BLAS_CHECK(gpublasCsyrk(static_cast<gpublasHandle_t>(h->blasH),blas_impl::fill(uplo),blas_impl::op(trans),n,k,
                reinterpret_cast<const D*>(&alpha),reinterpret_cast<const D*>(A),lda,
                reinterpret_cast<const D*>(&beta),reinterpret_cast<D*>(C),ldc));
        }
        else if constexpr (std::is_same_v<T,std::complex<double>>) {
            using D=gpublasDoubleComplex;
            BLAS_CHECK(gpublasZsyrk(static_cast<gpublasHandle_t>(h->blasH),blas_impl::fill(uplo),blas_impl::op(trans),n,k,
                reinterpret_cast<const D*>(&alpha),reinterpret_cast<const D*>(A),lda,
                reinterpret_cast<const D*>(&beta),reinterpret_cast<D*>(C),ldc));
        }
#endif
    }
}
} // namespace blas_impl
template <DdlaBackend Backend, typename T>
void syrk(const DdlaHandle_t& h, char uplo, char trans, int n, int k,
    const T& alpha, const T* A, int lda, const T& beta, T* C, int ldc)
{
    blas_impl::syrk_check<Backend,T>(h,uplo,trans,n,k,alpha,A,lda,beta,C,ldc);
    blas_impl::syrk_run<Backend,T>(h,uplo,trans,n,k,alpha,A,lda,beta,C,ldc);
}

#define INSTANTIATE(B,T) template void syrk<B,T>(const DdlaHandle_t&,char,char,int,int,const T&,const T*,int,const T&,T*,int); \
    template void blas_impl::syrk_check<B,T>(const DdlaHandle_t&,char,char,int,int,const T&,const T*,int,const T&,T*,int); \
    template void blas_impl::syrk_run<B,T>(const DdlaHandle_t&,char,char,int,int,const T&,const T*,int,const T&,T*,int);
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
