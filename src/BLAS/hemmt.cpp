#include "hemmt_impl.h"
#include "blas_impl.h"
#include "gemmt_impl.h"
#include "helpers.h"
#include "require_backend.h"
#include "plan_util.h"
#include "ddla_stream.h"
#include <ddla/ddla_vector.h>
#include <algorithm>
#include <complex>
#include <stdexcept>
#include <type_traits>
namespace ddla { namespace blas_impl { namespace {
constexpr int hemmt_panel_size=512;
template <typename T> T diagonal_value(const T* x) { return *x; }
template <typename R> std::complex<R> diagonal_value(const std::complex<R>* x) {
    return {x->real(), R(0)};
}

#if DDLA_GPU_VENDOR
// Load complex input by its real components. A Hermitian imaginary diagonal
// may be uninitialized, so even a by-value complex load there is forbidden.
template <typename T>
__device__ T diagonal_load(const void* a,std::size_t p,bool diagonal)
{
    if constexpr (std::is_arithmetic_v<T>) return static_cast<const T*>(a)[p];
    else {
        using R=typename T::value_type;
        const R* x=static_cast<const R*>(a)+2*p;
        return {x[0],diagonal ? R(0) : x[1]};
    }
}

template <typename T>
__global__ void pack_diagonal(bool upper, bool hermitian, int width, const void* a, int lda, T* d, int ldd)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= width || j >= width) return;
    const bool stored = upper ? i <= j : i >= j;
    const int r = stored ? i : j, c = stored ? j : i;
    const T v = diagonal_load<T>(a, r + std::size_t(c)*lda, hermitian && i == j);
    d[i + std::size_t(j)*ldd] = detail::device_conj(v, hermitian && !stored);
}
#endif

} // namespace

template <DdlaBackend Backend, typename T>
void diagonal_block_run(const DdlaHandle_t& h, bool upper, bool hermitian, int width,
                    const T* a, int lda, T* d, int ldd)
{
    if constexpr (Backend == DdlaBackend::CPU) {
        for (int j=0; j<width; ++j) for (int i=0; i<width; ++i) {
            if (hermitian && i == j) d[i + std::size_t(j)*ldd] = diagonal_value(a+i+std::size_t(j)*lda);
            else {
                const bool stored = upper ? i <= j : i >= j;
                const T v = stored ? a[i+std::size_t(j)*lda] : a[j+std::size_t(i)*lda];
                d[i+std::size_t(j)*ldd] = hermitian && !stored ? detail::host_conj(v) : v;
            }
        }
    } else {
#if DDLA_GPU_VENDOR
        using D = detail::DeviceScalarT<T>;
        static_assert(sizeof(D) == sizeof(T));
        const dim3 threads(16,16), blocks((width-1)/16+1,(width-1)/16+1);
        pack_diagonal<D><<<blocks,threads,0,static_cast<gpuStream_t>(h->stream)>>>(upper,hermitian,width,a,lda,reinterpret_cast<D*>(d),ldd);
        DDLA_CHECK(ddlaGetLastError(h));
#endif
    }
}


template <DdlaBackend Backend,typename T>
bool hemmt_check(const DdlaHandle_t& h,char side,char uploA,char uploC,int n,
    const T& alpha,const T* A,int lda,const T* B,int ldb,const T& beta,T* C,int ldc)
{
    detail::require_backend<Backend>(h,"hemmt");
    side=upper(side);uploA=upper(uploA);uploC=upper(uploC);
    if ((side!='L' && side!='R') || (uploA!='U' && uploA!='L') || (uploC!='U' && uploC!='L'))
        throw std::invalid_argument("hemmt: expected side L/R and input/output triangles U/L");
    if (n<0 || lda<std::max(1,n) || ldb<std::max(1,n) || ldc<std::max(1,n))
        throw std::invalid_argument("hemmt: invalid dimensions or leading dimensions");
    const bool noop=n==0 || (alpha==T(0) && beta==T(1));
    if (!noop) {
        if (!C) throw std::invalid_argument("hemmt: null C");
        if (alpha!=T(0)) {
            if (!A || !B) throw std::invalid_argument("hemmt: null input");
            if (detail::matrix_overlaps(A,lda,n,n,C,ldc,n,n) || detail::matrix_overlaps(B,ldb,n,n,C,ldc,n,n))
                throw std::invalid_argument("hemmt: C must not overlap A or B");
        }
    }
    return noop;
}

template <typename T>
void hemmt_workspace_size(const DdlaHandle_t& h,int n,bool product,std::size_t* bytes)
{
    *bytes=0;
    if (n==0 || !product) return;
    const int width=std::min(n,hemmt_panel_size);
    std::size_t triangle=0;
    gemmt_workspace_size<T>(h,'N','N',width,n,&triangle);
    *bytes=detail::padded_bytes<T>(std::size_t(n)*width)+detail::padded_bytes<char>(triangle);
}

template <DdlaBackend Backend,typename T>
void hemmt_run(const DdlaHandle_t& h,char side,char uploA,char uploC,int n,
    const T& alpha,const T* A,int lda,const T* B,int ldb,const T& beta,T* C,int ldc,
    void* work,std::size_t bytes)
{
    if (n==0 || (alpha==T(0) && beta==T(1))) return;
    if (alpha==T(0)) {
        gemmt_run<Backend,T>(h,uploC,'N','N',n,0,T(0),nullptr,1,nullptr,1,beta,C,ldc,nullptr,0);
        return;
    }
    const bool left=side=='L',upper_input=uploA=='U',upper_output=uploC=='U';
    const int block=std::min(n,hemmt_panel_size);
    char* cursor=static_cast<char*>(work);
    T* panel=detail::take<T>(cursor,std::size_t(n)*block);
    const std::size_t triangle_bytes=bytes-std::size_t(cursor-static_cast<char*>(work));
    for (int first=0;first<n;) {
        const int width=std::min(block,n-first),after=first+width;
        const int ldp=left ? width : n;
        const auto copy=[&](int begin,int length,bool direct) {
            if (length==0) return;
            const T* source=left
                ? (direct ? A+first+std::size_t(begin)*lda : A+begin+std::size_t(first)*lda)
                : (direct ? A+begin+std::size_t(first)*lda : A+first+std::size_t(begin)*lda);
            T* target=left ? panel+std::size_t(begin)*width : panel+begin;
            geam_run<Backend,T>(h,direct ? 'N' : 'C','N',left ? width : length,left ? length : width,
                T(1),source,lda,T(0),nullptr,1,target,ldp);
        };
        copy(0,first,left ? !upper_input : upper_input);
        copy(after,n-after,left ? upper_input : !upper_input);
        diagonal_block_run<Backend,T>(h,upper_input,true,width,A+first+std::size_t(first)*lda,lda,
            left ? panel+std::size_t(first)*width : panel+first,ldp);
        const int begin=upper_output ? (left ? after : 0) : (left ? 0 : after);
        const int length=upper_output ? (left ? n-after : first) : (left ? first : n-after);
        if (length>0) {
            if (left) gemm_run<Backend,T>(h,'N','N',width,length,n,alpha,panel,ldp,
                B+std::size_t(begin)*ldb,ldb,beta,C+first+std::size_t(begin)*ldc,ldc);
            else gemm_run<Backend,T>(h,'N','N',length,width,n,alpha,B+begin,ldb,
                panel,ldp,beta,C+begin+std::size_t(first)*ldc,ldc);
        }
        gemmt_run<Backend,T>(h,uploC,'N','N',width,n,alpha,left ? panel : B+first,
            left ? ldp : ldb,left ? B+std::size_t(first)*ldb : panel,left ? ldb : ldp,
            beta,C+first+std::size_t(first)*ldc,ldc,cursor,triangle_bytes);
        first+=width;
    }
}
}} // namespace ddla::blas_impl
namespace ddla {
template <DdlaBackend Backend,typename T>
void hemmt(const DdlaHandle_t& h,char side,char uploA,char uploC,int n,
    const T& alpha,const T* A,int lda,const T* B,int ldb,const T& beta,T* C,int ldc)
{
    if (blas_impl::hemmt_check<Backend,T>(h,side,uploA,uploC,n,alpha,A,lda,B,ldb,beta,C,ldc)) return;
    std::size_t bytes=0;blas_impl::hemmt_workspace_size<T>(h,n,alpha!=T(0),&bytes);
    vector<char> work(h,bytes);
    blas_impl::hemmt_run<Backend,T>(h,blas_impl::upper(side),blas_impl::upper(uploA),blas_impl::upper(uploC),
        n,alpha,A,lda,B,ldb,beta,C,ldc,work.data(),bytes);
}
#define INSTANTIATE(B,T) \
 template void hemmt<B,T>(const DdlaHandle_t&,char,char,char,int,const T&,const T*,int,const T*,int,const T&,T*,int); \
 template bool blas_impl::hemmt_check<B,T>(const DdlaHandle_t&,char,char,char,int,const T&,const T*,int,const T*,int,const T&,T*,int); \
 template void blas_impl::hemmt_run<B,T>(const DdlaHandle_t&,char,char,char,int,const T&,const T*,int,const T*,int,const T&,T*,int,void*,std::size_t); \
 template void blas_impl::diagonal_block_run<B,T>(const DdlaHandle_t&,bool,bool,int,const T*,int,T*,int)
#define TYPES(B) \
 INSTANTIATE(B,float); INSTANTIATE(B,double); INSTANTIATE(B,std::complex<float>); INSTANTIATE(B,std::complex<double>)
TYPES(DdlaBackend::CPU);
#if DDLA_HAS_GPU
TYPES(DdlaBackend::GPU);
#endif
#define SIZE(T) template void blas_impl::hemmt_workspace_size<T>(const DdlaHandle_t&,int,bool,std::size_t*)
SIZE(float); SIZE(double); SIZE(std::complex<float>); SIZE(std::complex<double>);
#undef SIZE
#undef TYPES
#undef INSTANTIATE
}
