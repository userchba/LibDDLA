#include "blas_impl.h"
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <complex>
#include <climits>
#include <stdexcept>
#include <string>

#include "ddla_connector.h"
#include "ddla_stream.h"
#include "instantiate.h"
#include "require_backend.h"

// CPU BLAS Level-1 SCAL bindings (Fortran ABI), declared by hand rather than
// via a vendor header -- same convention as gemm.cpp's sgemm_/dgemm_/cgemm_/
// zgemm_ block.
extern "C" {
void sscal_(const int*, const float*, float*, const int*);
void dscal_(const int*, const double*, double*, const int*);
void cscal_(const int*, const std::complex<float>*, std::complex<float>*, const int*);
void zscal_(const int*, const std::complex<double>*, std::complex<double>*, const int*);
void csscal_(const int*, const float*, std::complex<float>*, const int*);
void zdscal_(const int*, const double*, std::complex<double>*, const int*);
}

namespace ddla {

inline void cpu_scal(int n, const float& alpha, float* x, int incx)
{
    sscal_(&n, &alpha, x, &incx);
}

inline void cpu_scal(int n, const double& alpha, double* x, int incx)
{
    dscal_(&n, &alpha, x, &incx);
}

inline void cpu_scal(int n, const std::complex<float>& alpha, std::complex<float>* x, int incx)
{
    cscal_(&n, &alpha, x, &incx);
}

inline void cpu_scal(int n, const std::complex<double>& alpha, std::complex<double>* x, int incx)
{
    zscal_(&n, &alpha, x, &incx);
}

#if DDLA_HAS_GPU
namespace {

// GPU arm: the vendor calls, one overload per scalar type (cuBLAS / hipBLAS scal).  Private to this translation unit --
// the public interface is the templated ddla::scal declared in <ddla/ddla.h>.
inline gpublasStatus_t gpublasScal(gpublasHandle_t handle, int64_t n, const float& alpha, float *x, int64_t incx)
{
    return gpublasSscal(handle, n, &alpha, x, incx);
}

inline gpublasStatus_t gpublasScal(gpublasHandle_t handle, int64_t n, const double& alpha, double *x, int64_t incx)
{
    return gpublasDscal(handle, n, &alpha, x, incx);
}

inline gpublasStatus_t gpublasScal(gpublasHandle_t handle, int64_t n, const std::complex<float>& alpha, std::complex<float> *x, int64_t incx)
{
    return gpublasCscal(handle, n, (gpublasComplex*)&alpha, (gpublasComplex*)x, incx);
}

inline gpublasStatus_t gpublasScal(gpublasHandle_t handle, int64_t n, const std::complex<double>& alpha, std::complex<double> *x, int64_t incx)
{
    return gpublasZscal(handle, n, (gpublasDoubleComplex*)&alpha, (gpublasDoubleComplex*)x, incx);
}

} // anonymous namespace
#endif // DDLA_HAS_GPU

namespace blas_impl {

template <DdlaBackend Backend, typename T>
void scal_check(const DdlaHandle_t& handle, int n, const T& alpha, T* x, int incx)
{
    detail::require_backend<Backend>(handle, "scal");
    if (n <= 0) return;

}

template <DdlaBackend Backend, typename T>
void scal_run(const DdlaHandle_t& handle, int n, const T& alpha, T* x, int incx)
{
    if (n <= 0) return;
    if constexpr (Backend == DdlaBackend::CPU) {
        cpu_scal(n, alpha, x, incx);
    } else {
#if DDLA_HAS_GPU
        BLAS_CHECK(gpublasScal(static_cast<gpublasHandle_t>(handle->blasH), static_cast<int64_t>(n), alpha, x,
                               static_cast<int64_t>(incx)));
#endif
    }
}

} // namespace blas_impl

template <DdlaBackend Backend, typename T>
void scal(const DdlaHandle_t& handle, int n, const T& alpha, T* x, int incx)
{
    blas_impl::scal_check<Backend, T>(handle, n, alpha, x, incx);
    blas_impl::scal_run<Backend, T>(handle, n, alpha, x, incx);
}

namespace blas_impl {
template <DdlaBackend Backend, typename T, typename Alpha>
void matrix_scal_run(const DdlaHandle_t& h,int m,int n,const Alpha& alpha,T* a,int lda)
{
    if (m==0 || n==0 || alpha==Alpha(1)) return;
    if constexpr (Backend==DdlaBackend::GPU) {
#if DDLA_HAS_GPU
        if (alpha==Alpha(0) || !std::is_same_v<T,Alpha>) {
            matrix_scal_gpu_run<T,Alpha>(h,m,n,alpha,a,lda,nullptr,1);
            return;
        }
#endif
    }
    if constexpr (std::is_same_v<T,Alpha>) {
        geam_run<Backend,T>(h,'N','N',m,n,alpha,a,lda,T(0),nullptr,1,a,lda);
    } else if constexpr (Backend==DdlaBackend::CPU) {
        using R=typename real_type<T>::type;
        static_assert(std::is_same_v<Alpha,R>);
        // std::complex components form a real matrix with twice the rows.
        if (m<=INT_MAX/2 && lda<=INT_MAX/2) {
            geam_run<Backend,R>(h,'N','N',2*m,n,alpha,reinterpret_cast<R*>(a),2*lda,
                R(0),nullptr,1,reinterpret_cast<R*>(a),2*lda);
        } else if (alpha==R(0)) {
            geam_run<Backend,T>(h,'N','N',m,n,T(0),a,lda,T(0),nullptr,1,a,lda);
        } else {
            // Avoid overflowing BLAS integer dimensions for a large pitch.
            const int one=1;
            for (int j=0;j<n;++j) {
                T* column=a+std::size_t(j)*lda;
                if constexpr(std::is_same_v<R,float>)csscal_(&m,&alpha,column,&one);
                else zdscal_(&m,&alpha,column,&one);
            }
        }
    }
}
}

// Matrix overloads share validation and a prechecked implementation.
template <DdlaBackend Backend, typename T>
void scal(const DdlaHandle_t& h,int m,int n,const T& alpha,T* a,int lda)
{
    if (m<0 || n<0 || lda<std::max(1,m))
        throw std::invalid_argument("scal(matrix): invalid dimensions or leading dimension");
    blas_impl::geam_check<Backend,T>(h,'N','N',m,n,alpha,a,lda,T(0),nullptr,1,a,lda);
    blas_impl::matrix_scal_run<Backend,T>(h,m,n,alpha,a,lda);
}
template <DdlaBackend Backend,typename T,typename Alpha>
typename std::enable_if<
    std::is_same<T, std::complex<Alpha>>::value,
    void
>::type
scal(const DdlaHandle_t& h,int m,int n,const Alpha& alpha,T* a,int lda)
{
    if (m<0 || n<0 || lda<std::max(1,m))
        throw std::invalid_argument("scal(matrix,mixed): invalid dimensions or leading dimension");
    blas_impl::geam_check<Backend,T>(h,'N','N',m,n,T(alpha),a,lda,T(0),nullptr,1,a,lda);
    blas_impl::matrix_scal_run<Backend,T,Alpha>(h,m,n,alpha,a,lda);
}
#define MIXED_SCAL(B,R) \
 template void scal<B,std::complex<R>,R>(const DdlaHandle_t&,int,int,const R&,std::complex<R>*,int); \
 template void blas_impl::matrix_scal_run<B,std::complex<R>,R>(const DdlaHandle_t&,int,int,const R&,std::complex<R>*,int)
#define MATRIX_RUN(B,T) template void blas_impl::matrix_scal_run<B,T,T>(const DdlaHandle_t&,int,int,const T&,T*,int)
#define MATRIX_RUN_TYPES(B) MATRIX_RUN(B,float);MATRIX_RUN(B,double); \
 MATRIX_RUN(B,std::complex<float>);MATRIX_RUN(B,std::complex<double>); \
 MIXED_SCAL(B,float);MIXED_SCAL(B,double)
DDLA_FOR_EACH_BACKEND(MATRIX_RUN_TYPES)
#undef MATRIX_RUN_TYPES
#undef MATRIX_RUN
#undef MIXED_SCAL

#define MATRIX_SCAL(B,T) template void scal<B,T>(const DdlaHandle_t&,int,int,const T&,T*,int)
DDLA_FOR_EACH_BACKEND_TYPE(MATRIX_SCAL)
#undef MATRIX_SCAL

#define INSTANTIATE_SCAL(BACKEND, TYPE)                                      \
    template void scal<BACKEND, TYPE>(                                       \
        const DdlaHandle_t&, int, const TYPE&, TYPE*, int)

DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_SCAL)

#undef INSTANTIATE_SCAL


#define DDLA_INSTANTIATE_PHASES(BACKEND, TYPE) \
    template void blas_impl::scal_check<BACKEND, TYPE>(const DdlaHandle_t& handle, int n, const TYPE& alpha, TYPE* x, int incx); \
    template void blas_impl::scal_run<BACKEND, TYPE>(const DdlaHandle_t& handle, int n, const TYPE& alpha, TYPE* x, int incx)
DDLA_FOR_EACH_BACKEND_TYPE(DDLA_INSTANTIATE_PHASES)
#undef DDLA_INSTANTIATE_PHASES
} // namespace ddla
