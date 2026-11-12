#include "lapack_impl.h"
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "host_threads.h"
#include "instantiate.h"
#include "require_backend.h"
#include <cctype>
#include <algorithm>
#include <complex>
#include <stdexcept>
#include <type_traits>
#include <vector>

extern "C" {
void ssyevd_(const char*, const char*, const int*, float*, const int*, float*, float*, const int*, int*, const int*, int*);
void dsyevd_(const char*, const char*, const int*, double*, const int*, double*, double*, const int*, int*, const int*, int*);
void cheevd_(const char*, const char*, const int*, std::complex<float>*, const int*, float*, std::complex<float>*, const int*, float*, const int*, int*, const int*, int*);
void zheevd_(const char*, const char*, const int*, std::complex<double>*, const int*, double*, std::complex<double>*, const int*, double*, const int*, int*, const int*, int*);
}

namespace ddla { namespace lapack_impl {
namespace {
template <class T>
void cpu_heevd(char job, char uplo, int n, T* a, int lda,
    typename real_type<T>::type* w, T* work, int lwork,
    typename real_type<T>::type* rwork, int lrwork, int* iwork, int liwork, int* info)
{
    if constexpr (std::is_same_v<T, float>)
        ssyevd_(&job, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, info);
    else if constexpr (std::is_same_v<T, double>)
        dsyevd_(&job, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, info);
    else if constexpr (std::is_same_v<T, std::complex<float>>)
        cheevd_(&job, &uplo, &n, a, &lda, w, work, &lwork, rwork, &lrwork, iwork, &liwork, info);
    else
        zheevd_(&job, &uplo, &n, a, &lda, w, work, &lwork, rwork, &lrwork, iwork, &liwork, info);
}

#if DDLA_HAS_GPU
// Overloads keep scalar-dependent vendor signatures out of the driver.
#define GPU_HEEVD(T, R, V, NAME) \
gpusolverStatus_t gpu_heevd_size(gpusolverHandle_t h, gpusolverEigMode_t job, \
    gpusolverFillMode_t uplo, int n, T* a, int lda, R* w, int* count) \
{ return NAME##_bufferSize(h, job, uplo, n, reinterpret_cast<V*>(a), lda, w, count); } \
gpusolverStatus_t gpu_heevd_call(gpusolverHandle_t h, gpusolverEigMode_t job, \
    gpusolverFillMode_t uplo, int n, T* a, int lda, R* w, T* work, int count, int* info) \
{ return NAME(h, job, uplo, n, reinterpret_cast<V*>(a), lda, w, reinterpret_cast<V*>(work), count, info); }
GPU_HEEVD(float, float, float, gpusolverSsyevd)
GPU_HEEVD(double, double, double, gpusolverDsyevd)
GPU_HEEVD(std::complex<float>, float, gpusolverComplex, gpusolverCheevd)
GPU_HEEVD(std::complex<double>, double, gpusolverDoubleComplex, gpusolverZheevd)
#undef GPU_HEEVD
#endif

} // namespace

template<DdlaBackend B>
void heevd_check(const DdlaHandle_t& h,char& job,char& uplo,int n,
    const void* a,int lda,const void* w,const int* info)
{
    detail::require_backend<B>(h,"heevd/syevd");
    job=static_cast<char>(std::toupper(static_cast<unsigned char>(job)));
    uplo=static_cast<char>(std::toupper(static_cast<unsigned char>(uplo)));
    if(n<0 || (job!='N'&&job!='V') || (uplo!='U'&&uplo!='L') || !info)
        throw std::invalid_argument("heevd/syevd: invalid order/job/uplo/info");
    if(n && (!a || !w || lda<std::max(1,n)))
        throw std::invalid_argument("heevd/syevd: invalid matrix/eigenvalues/leading dimension");
}
#define CHECK(B) template void heevd_check<B>(const DdlaHandle_t&,char&,char&,int,const void*,int,const void*,const int*)
DDLA_FOR_EACH_BACKEND(CHECK)
#undef CHECK

template <class T>
void heevd_workspace_size(const DdlaHandle_t& h, char job, char uplo, int n,
    T* a, int lda, typename real_type<T>::type* w, int& lwork, int& lrwork, int& liwork, std::size_t& padding_elements)
{
    using R = typename real_type<T>::type;
    lwork = lrwork = liwork = 0;
    padding_elements=0;
    if (!n) return;
    if (ddlaGetBackend(h) == DdlaBackend::CPU) {
        T work{};
        R rwork{};
        int iwork = 0, info = 0;
        cpu_heevd(job, uplo, n, a, lda, w, &work, -1, &rwork, -1, &iwork, -1, &info);
        if (info) throw std::runtime_error("pheevd: LAPACK workspace query failed");
        lwork = static_cast<int>(std::real(work));
        if constexpr (!std::is_same_v<T, R>) lrwork = static_cast<int>(rwork);
        liwork = iwork;
    } else {
#if DDLA_HAS_GPU
        SOLVER_CHECK(gpu_heevd_size(static_cast<gpusolverHandle_t>(h->solverH),
            job == 'V' ? GPUSOLVER_EIG_MODE_VECTOR : GPUSOLVER_EIG_MODE_NOVECTOR,
            uplo == 'U' ? GPUSOLVER_FILL_MODE_UPPER : GPUSOLVER_FILL_MODE_LOWER,
            n, a, lda, w, &lwork));
        padding_elements=std::size_t(lda-n)*n;
#endif
    }
}

template <DdlaBackend B, class T>
void heevd_run(const DdlaHandle_t& h, char job, char uplo, int n, T* a, int lda,
    typename real_type<T>::type* w, T* work, int lwork,
    typename real_type<T>::type* rwork, int lrwork, int* iwork, int liwork, int* info, T* padding)
{
    if (!n) {
        if constexpr (B==DdlaBackend::CPU) *info=0;
        else DDLA_CHECK(ddlaMemset(h,info,0,sizeof(int)));
        return;
    }
    if constexpr (B == DdlaBackend::CPU) {
        detail::ScopedBlasThreads threads(h->host_threads);
        cpu_heevd(job, uplo, n, a, lda, w, work, lwork, rwork, lrwork, iwork, liwork, info);
    } else {
#if DDLA_HAS_GPU
        const std::size_t width=std::size_t(lda-n)*sizeof(T);
        if(width)DDLA_CHECK(ddlaMemcpy2D(h,padding,width,a+n,std::size_t(lda)*sizeof(T),width,n));
        try {
#if defined(DDLA_USE_CUDA)
            // CUDA 11.8 ZHEEVD reads uninitialized caller workspace in its
            // SYTRD kernel (reproduced at n=65), including with valid input.
            if constexpr (std::is_same_v<T, std::complex<double>>)
                DDLA_CHECK(ddlaMemset(h, work, 0, std::size_t(lwork) * sizeof(T)));
#endif
            SOLVER_CHECK(gpu_heevd_call(static_cast<gpusolverHandle_t>(h->solverH),
                job == 'V' ? GPUSOLVER_EIG_MODE_VECTOR : GPUSOLVER_EIG_MODE_NOVECTOR,
                uplo == 'U' ? GPUSOLVER_FILL_MODE_UPPER : GPUSOLVER_FILL_MODE_LOWER,
                n, a, lda, w, work, lwork, info));
        } catch(...) {
            if(width) {
                (void)ddlaMemcpy2D(h,a+n,std::size_t(lda)*sizeof(T),padding,width,width,n);
                (void)ddlaSynchronize(h);
            }
            throw;
        }
        if(width)DDLA_CHECK(ddlaMemcpy2D(h,a+n,std::size_t(lda)*sizeof(T),padding,width,width,n));
#endif
    }
}

template<DdlaBackend B,class T>
void heevd_allocate(const DdlaHandle_t& h,char job,char uplo,int n,T* a,int lda,
               typename real_type<T>::type* w,int* info)
{
    heevd_check<B>(h,job,uplo,n,a,lda,w,info);
    int lwork=0,lrwork=0,liwork=0;std::size_t padding_elements=0;
    heevd_workspace_size<T>(h,job,uplo,n,a,lda,w,lwork,lrwork,liwork,padding_elements);
    ddla::vector<T> work(h,lwork),padding(h,padding_elements);
    std::vector<typename real_type<T>::type> rwork(lrwork);
    std::vector<int> iwork(liwork);
    heevd_run<B,T>(h,job,uplo,n,a,lda,w,work.data(),lwork,
        rwork.data(),lrwork,iwork.data(),liwork,info,padding.data());
}
#define SIZE(T) template void heevd_workspace_size<T>(const DdlaHandle_t&, char, char, int, T*, int, typename real_type<T>::type*, int&, int&, int&, std::size_t&)
#define RUN(B,T) \
    template void heevd_run<B,T>(const DdlaHandle_t&, char, char, int, T*, int, typename real_type<T>::type*, T*, int, typename real_type<T>::type*, int, int*, int, int*, T*); \
    template void heevd_allocate<B,T>(const DdlaHandle_t&,char,char,int,T*,int,typename real_type<T>::type*,int*)
DDLA_FOR_EACH_TYPE(SIZE)
DDLA_FOR_EACH_BACKEND_TYPE(RUN)
#undef RUN
#undef SIZE
} } // namespace ddla::lapack_impl

namespace ddla {
template<DdlaBackend B,class T>
void heevd(const DdlaHandle_t& h,char job,char uplo,int n,T* a,int lda,
           typename real_type<T>::type* w,int* info)
{
    static_assert(std::is_same_v<T,std::complex<float>> || std::is_same_v<T,std::complex<double>>,
                  "heevd requires complex scalars");
    lapack_impl::heevd_allocate<B,T>(h,job,uplo,n,a,lda,w,info);
}
#define HEEVD(B,T) template void heevd<B,T>(const DdlaHandle_t&,char,char,int,T*,int,typename real_type<T>::type*,int*)
#define HEEVD_TYPES(B) HEEVD(B,std::complex<float>); HEEVD(B,std::complex<double>)
DDLA_FOR_EACH_BACKEND(HEEVD_TYPES)
#undef HEEVD_TYPES
#undef HEEVD
} // namespace ddla
