#include "hemmt_impl.h"
#include <string>
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


template <DdlaBackend Backend,typename T,typename TA,typename TB>
bool hemmt_check(const DdlaHandle_t& h,char side,char uploA,char uploC,int n,
    const T& alpha,const TA* A,int lda,const TB* B,int ldb,const T& beta,T* C,int ldc,
    const char* routine,int free_dimension)
{
    detail::require_backend<Backend>(h,routine);
    side=upper(side);uploA=upper(uploA);uploC=upper(uploC);
    const int other=free_dimension<0?n:free_dimension;
    const int rows=side=='L'?n:other,cols=side=='L'?other:n;
    if ((side!='L' && side!='R') || (uploA!='U' && uploA!='L') || (uploC!='U' && uploC!='L' && (uploC!='F' || free_dimension<0)))
        throw std::invalid_argument(std::string(routine)+": expected side L/R and input/output triangles U/L");
    if (n<0 || other<0 || (uploC!='F' && other!=n) || lda<std::max(1,n) || ldb<std::max(1,rows) || ldc<std::max(1,rows))
        throw std::invalid_argument(std::string(routine)+": invalid dimensions or leading dimensions");
    const bool noop=n==0 || other==0 || (alpha==T(0) && beta==T(1));
    if (!noop) {
        if (!C) throw std::invalid_argument(std::string(routine)+": null C");
        if (alpha!=T(0)) {
            if (!A || !B) throw std::invalid_argument(std::string(routine)+": null input");
            if (detail::matrix_overlaps(A,lda,n,n,C,ldc,rows,cols) || detail::matrix_overlaps(B,ldb,rows,cols,C,ldc,rows,cols))
                throw std::invalid_argument(std::string(routine)+": C must not overlap A or B");
        }
    }
    return noop;
}

template <typename T,typename TA,typename TB>
void hemmt_workspace_size(const DdlaHandle_t& h,int n,bool product,std::size_t* bytes,int free_dimension,char side,char uploC)
{
    *bytes=0;
    const int other=free_dimension<0?n:free_dimension;
    if (n==0 || other==0 || !product) return;
    const int width=std::min(n,hemmt_panel_size);
    std::size_t triangle=0;
    if constexpr (!std::is_same_v<T,TA> || !std::is_same_v<T,TB>) {
        for (int first=0;first<n;) {
            const int current=std::min(width,n-first),after=first+current;
            const int length=uploC=='F' ? other : uploC=='U' ? (side=='L'?n-after:first) : (side=='L'?first:n-after);
            std::size_t bytes=0;
            const bool complex_left=side=='L' ? std::is_same_v<T,TA> : std::is_same_v<T,TB>;
            if(complex_left && h->backend==DdlaBackend::GPU)bytes=detail::padded_bytes<T>(std::size_t(current)*std::min(length,n));
            else if(side=='L')local_gemm_workspace_size<T,TA,TB>(h,current,length,n,&bytes);
            else local_gemm_workspace_size<T,TB,TA>(h,length,current,n,&bytes);
            triangle=std::max(triangle,bytes);
            if (uploC!='F') {
                if (side=='L') gemmt_workspace_size<T,TA,TB>(h,'N','N',current,n,&bytes);
                else gemmt_workspace_size<T,TB,TA>(h,'N','N',current,n,&bytes);
                triangle=std::max(triangle,bytes);
            }
            first+=current;
        }
        *bytes=detail::padded_bytes<TA>(std::size_t(n)*width)+detail::padded_bytes<char>(triangle);
        return;
    }
    if (uploC!='F') {
        if (side=='L') gemmt_workspace_size<T,TA,TB>(h,'N','N',width,n,&triangle);
        else gemmt_workspace_size<T,TB,TA>(h,'N','N',width,n,&triangle);
    }
    std::size_t product_bytes=0;
    if (side=='L') local_gemm_workspace_size<T,TA,TB>(h,width,other,n,&product_bytes);
    else local_gemm_workspace_size<T,TB,TA>(h,other,width,n,&product_bytes);
    triangle=std::max(triangle,product_bytes);
    *bytes=detail::padded_bytes<TA>(std::size_t(n)*width)+detail::padded_bytes<char>(triangle);
}

template <DdlaBackend Backend,typename T,typename TA,typename TB>
void hemmt_run(const DdlaHandle_t& h,char side,char uploA,char uploC,int n,
    const T& alpha,const TA* A,int lda,const TB* B,int ldb,const T& beta,T* C,int ldc,
    void* work,std::size_t bytes,bool hermitian,int free_dimension)
{
    const int other=free_dimension<0?n:free_dimension;
    const int rows=side=='L'?n:other,cols=side=='L'?other:n;
    if (n==0 || other==0 || (alpha==T(0) && beta==T(1))) return;
    if (alpha==T(0)) {
        if (uploC!='F') gemmt_run<Backend,T,T,T>(h,uploC,'N','N',n,0,T(0),nullptr,1,nullptr,1,beta,C,ldc,nullptr,0);
        else if (beta==T(0)) for (int j=0;j<cols;++j)
            DDLA_CHECK(ddlaMemset(h,C+std::size_t(j)*ldc,0,std::size_t(rows)*sizeof(T)));
        else geam_run<Backend,T>(h,'N','N',rows,cols,beta,C,ldc,T(0),nullptr,1,C,ldc);
        return;
    }
    const bool left=side=='L',upper_input=uploA=='U',upper_output=uploC=='U';
    const int block=std::min(n,hemmt_panel_size);
    const char reflection=hermitian ? 'C' : 'T';
    char* cursor=static_cast<char*>(work);
    TA* panel=detail::take<TA>(cursor,std::size_t(n)*block);
    const std::size_t triangle_bytes=bytes-std::size_t(cursor-static_cast<char*>(work));
    for (int first=0;first<n;) {
        const int width=std::min(block,n-first),after=first+width;
        const int ldp=left ? width : n;
        const auto copy=[&](int begin,int length,bool direct) {
            if (length==0) return;
            const TA* source=left
                ? (direct ? A+first+std::size_t(begin)*lda : A+begin+std::size_t(first)*lda)
                : (direct ? A+begin+std::size_t(first)*lda : A+first+std::size_t(begin)*lda);
            TA* target=left ? panel+std::size_t(begin)*width : panel+begin;
            geam_run<Backend,TA>(h,direct ? 'N' : reflection,'N',left ? width : length,left ? length : width,
                TA(1),source,lda,TA(0),nullptr,1,target,ldp);
        };
        copy(0,first,left ? !upper_input : upper_input);
        copy(after,n-after,left ? upper_input : !upper_input);
        diagonal_block_run<Backend,TA>(h,upper_input,hermitian,width,A+first+std::size_t(first)*lda,lda,
            left ? panel+std::size_t(first)*width : panel+first,ldp);
        const int begin=uploC=='F' ? 0 : upper_output ? (left ? after : 0) : (left ? 0 : after);
        const int length=uploC=='F' ? other : upper_output ? (left ? n-after : first) : (left ? first : n-after);
        if (length>0) {
            if constexpr(Backend==DdlaBackend::GPU && (!std::is_same_v<T,TA> || !std::is_same_v<T,TB>)) {
                const bool complex_left=left ? std::is_same_v<T,TA> : std::is_same_v<T,TB>;
                if(complex_left) {
                    using R=typename real_type<T>::type;
                    const int free_block=n;
                    for(int offset=0;offset<length;) {
                        const int part=std::min(free_block,length-offset);
                        const int nr=left?width:part,nc=left?part:width;
                        const T* complex_input;
                        const R* real_input;
                        if constexpr(std::is_same_v<T,TA>) {
                            complex_input=panel;real_input=B+std::size_t(begin+offset)*ldb;
                        } else {
                            complex_input=B+begin+offset;real_input=panel;
                        }
                        // Interleaved complex rows are already a real 2*nr
                        // by K matrix. Native real GEMM needs no conversion.
                        T* result=reinterpret_cast<T*>(cursor);
                        gemm_run<Backend,R>(h,'N','N',2*nr,nc,n,R(1),
                            reinterpret_cast<const R*>(complex_input),2*(left?ldp:ldb),
                            real_input,left?ldb:ldp,R(0),reinterpret_cast<R*>(result),2*nr);
                        T* output=C+(left?first:begin+offset)+std::size_t(left?begin+offset:first)*ldc;
                        geam_run<Backend,T>(h,'N','N',nr,nc,alpha,result,nr,beta,output,ldc,output,ldc);
                        offset+=part;
                    }
                } else if(left)local_gemm_run<Backend,T,TA,TB>(h,'N','N',width,length,n,alpha,panel,ldp,
                    B+std::size_t(begin)*ldb,ldb,beta,C+first+std::size_t(begin)*ldc,ldc,cursor,triangle_bytes);
                else local_gemm_run<Backend,T,TB,TA>(h,'N','N',length,width,n,alpha,B+begin,ldb,
                    panel,ldp,beta,C+begin+std::size_t(first)*ldc,ldc,cursor,triangle_bytes);
            } else if(left)local_gemm_run<Backend,T,TA,TB>(h,'N','N',width,length,n,alpha,panel,ldp,
                B+std::size_t(begin)*ldb,ldb,beta,C+first+std::size_t(begin)*ldc,ldc,cursor,triangle_bytes);
            else local_gemm_run<Backend,T,TB,TA>(h,'N','N',length,width,n,alpha,B+begin,ldb,
                panel,ldp,beta,C+begin+std::size_t(first)*ldc,ldc,cursor,triangle_bytes);
        }
        if (uploC!='F') {
            if (left) gemmt_run<Backend,T,TA,TB>(h,uploC,'N','N',width,n,alpha,panel,ldp,
                B+std::size_t(first)*ldb,ldb,beta,C+first+std::size_t(first)*ldc,ldc,cursor,triangle_bytes);
            else gemmt_run<Backend,T,TB,TA>(h,uploC,'N','N',width,n,alpha,B+first,ldb,
                panel,ldp,beta,C+first+std::size_t(first)*ldc,ldc,cursor,triangle_bytes);
        }
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
 template bool blas_impl::hemmt_check<B,T>(const DdlaHandle_t&,char,char,char,int,const T&,const T*,int,const T*,int,const T&,T*,int,const char*,int); \
 template void blas_impl::hemmt_run<B,T>(const DdlaHandle_t&,char,char,char,int,const T&,const T*,int,const T*,int,const T&,T*,int,void*,std::size_t,bool,int); \
 template void blas_impl::diagonal_block_run<B,T>(const DdlaHandle_t&,bool,bool,int,const T*,int,T*,int)
#define TYPES(B) \
 INSTANTIATE(B,float); INSTANTIATE(B,double); INSTANTIATE(B,std::complex<float>); INSTANTIATE(B,std::complex<double>)
TYPES(DdlaBackend::CPU);
#if DDLA_HAS_GPU
TYPES(DdlaBackend::GPU);
#endif
#define SIZE(T) template void blas_impl::hemmt_workspace_size<T>(const DdlaHandle_t&,int,bool,std::size_t*,int,char,char)
SIZE(float); SIZE(double); SIZE(std::complex<float>); SIZE(std::complex<double>);
#undef SIZE
#undef TYPES
#undef INSTANTIATE
template <DdlaBackend Backend,typename R>
void hemm(const DdlaHandle_t& h,char side,char ua,int m,int n,const std::complex<R>& alpha,
    const R* A,int lda,const std::complex<R>* B,int ldb,const std::complex<R>& beta,std::complex<R>* C,int ldc)
{
    using T=std::complex<R>;
    side=blas_impl::upper(side);ua=blas_impl::upper(ua);
    if (m<0 || n<0) throw std::invalid_argument("hemm(mixed): negative dimension");
    const char uc='F';const int order=side=='L'?m:n,other=side=='L'?n:m;
    if (blas_impl::hemmt_check<Backend,T>(h,side,ua,uc,order,alpha,A,lda,B,ldb,beta,C,ldc,"hemm(mixed)",other)) return;
    std::size_t bytes=0;
    blas_impl::hemmt_workspace_size<T,R,std::complex<R>>(h,order,alpha!=T(0),&bytes,other,side,uc);
    vector<char> work(h,bytes);
    blas_impl::hemmt_run<Backend,T>(h,side,ua,uc,order,alpha,A,lda,B,ldb,beta,C,ldc,work.data(),bytes,true,other);
}

template <DdlaBackend Backend,typename R>
void hemm(const DdlaHandle_t& h,char side,char ua,int m,int n,const std::complex<R>& alpha,
    const std::complex<R>* A,int lda,const R* B,int ldb,const std::complex<R>& beta,std::complex<R>* C,int ldc)
{
    using T=std::complex<R>;
    side=blas_impl::upper(side);ua=blas_impl::upper(ua);
    if (m<0 || n<0) throw std::invalid_argument("hemm(mixed): negative dimension");
    const char uc='F';const int order=side=='L'?m:n,other=side=='L'?n:m;
    if (blas_impl::hemmt_check<Backend,T>(h,side,ua,uc,order,alpha,A,lda,B,ldb,beta,C,ldc,"hemm(mixed)",other)) return;
    std::size_t bytes=0;
    blas_impl::hemmt_workspace_size<T,std::complex<R>,R>(h,order,alpha!=T(0),&bytes,other,side,uc);
    vector<char> work(h,bytes);
    blas_impl::hemmt_run<Backend,T>(h,side,ua,uc,order,alpha,A,lda,B,ldb,beta,C,ldc,work.data(),bytes,true,other);
}

template <DdlaBackend Backend,typename R>
void hemmt(const DdlaHandle_t& h,char side,char ua,char uc,int n,const std::complex<R>& alpha,
    const R* A,int lda,const std::complex<R>* B,int ldb,const std::complex<R>& beta,std::complex<R>* C,int ldc)
{
    using T=std::complex<R>;
    side=blas_impl::upper(side);ua=blas_impl::upper(ua);
    uc=blas_impl::upper(uc);
    if (uc!='U' && uc!='L') throw std::invalid_argument("hemmt(mixed): expected output U/L");
    const int order=n,other=n;
    if (blas_impl::hemmt_check<Backend,T>(h,side,ua,uc,order,alpha,A,lda,B,ldb,beta,C,ldc,"hemmt(mixed)",other)) return;
    std::size_t bytes=0;
    blas_impl::hemmt_workspace_size<T,R,std::complex<R>>(h,order,alpha!=T(0),&bytes,other,side,uc);
    vector<char> work(h,bytes);
    blas_impl::hemmt_run<Backend,T>(h,side,ua,uc,order,alpha,A,lda,B,ldb,beta,C,ldc,work.data(),bytes,true,other);
}

template <DdlaBackend Backend,typename R>
void hemmt(const DdlaHandle_t& h,char side,char ua,char uc,int n,const std::complex<R>& alpha,
    const std::complex<R>* A,int lda,const R* B,int ldb,const std::complex<R>& beta,std::complex<R>* C,int ldc)
{
    using T=std::complex<R>;
    side=blas_impl::upper(side);ua=blas_impl::upper(ua);
    uc=blas_impl::upper(uc);
    if (uc!='U' && uc!='L') throw std::invalid_argument("hemmt(mixed): expected output U/L");
    const int order=n,other=n;
    if (blas_impl::hemmt_check<Backend,T>(h,side,ua,uc,order,alpha,A,lda,B,ldb,beta,C,ldc,"hemmt(mixed)",other)) return;
    std::size_t bytes=0;
    blas_impl::hemmt_workspace_size<T,std::complex<R>,R>(h,order,alpha!=T(0),&bytes,other,side,uc);
    vector<char> work(h,bytes);
    blas_impl::hemmt_run<Backend,T>(h,side,ua,uc,order,alpha,A,lda,B,ldb,beta,C,ldc,work.data(),bytes,true,other);
}

template void hemm<DdlaBackend::CPU,float>(const DdlaHandle_t&,char,char,int,int,const std::complex<float>&,const float*,int,const std::complex<float>*,int,const std::complex<float>&,std::complex<float>*,int);
template void hemmt<DdlaBackend::CPU,float>(const DdlaHandle_t&,char,char,char,int,const std::complex<float>&,const float*,int,const std::complex<float>*,int,const std::complex<float>&,std::complex<float>*,int);
template bool blas_impl::hemmt_check<DdlaBackend::CPU,std::complex<float>,float,std::complex<float>>(const DdlaHandle_t&,char,char,char,int,const std::complex<float>&,const float*,int,const std::complex<float>*,int,const std::complex<float>&,std::complex<float>*,int,const char*,int);
template void blas_impl::hemmt_run<DdlaBackend::CPU,std::complex<float>,float,std::complex<float>>(const DdlaHandle_t&,char,char,char,int,const std::complex<float>&,const float*,int,const std::complex<float>*,int,const std::complex<float>&,std::complex<float>*,int,void*,std::size_t,bool,int);
template void hemm<DdlaBackend::CPU,float>(const DdlaHandle_t&,char,char,int,int,const std::complex<float>&,const std::complex<float>*,int,const float*,int,const std::complex<float>&,std::complex<float>*,int);
template void hemmt<DdlaBackend::CPU,float>(const DdlaHandle_t&,char,char,char,int,const std::complex<float>&,const std::complex<float>*,int,const float*,int,const std::complex<float>&,std::complex<float>*,int);
template bool blas_impl::hemmt_check<DdlaBackend::CPU,std::complex<float>,std::complex<float>,float>(const DdlaHandle_t&,char,char,char,int,const std::complex<float>&,const std::complex<float>*,int,const float*,int,const std::complex<float>&,std::complex<float>*,int,const char*,int);
template void blas_impl::hemmt_run<DdlaBackend::CPU,std::complex<float>,std::complex<float>,float>(const DdlaHandle_t&,char,char,char,int,const std::complex<float>&,const std::complex<float>*,int,const float*,int,const std::complex<float>&,std::complex<float>*,int,void*,std::size_t,bool,int);
template void hemm<DdlaBackend::CPU,double>(const DdlaHandle_t&,char,char,int,int,const std::complex<double>&,const double*,int,const std::complex<double>*,int,const std::complex<double>&,std::complex<double>*,int);
template void hemmt<DdlaBackend::CPU,double>(const DdlaHandle_t&,char,char,char,int,const std::complex<double>&,const double*,int,const std::complex<double>*,int,const std::complex<double>&,std::complex<double>*,int);
template bool blas_impl::hemmt_check<DdlaBackend::CPU,std::complex<double>,double,std::complex<double>>(const DdlaHandle_t&,char,char,char,int,const std::complex<double>&,const double*,int,const std::complex<double>*,int,const std::complex<double>&,std::complex<double>*,int,const char*,int);
template void blas_impl::hemmt_run<DdlaBackend::CPU,std::complex<double>,double,std::complex<double>>(const DdlaHandle_t&,char,char,char,int,const std::complex<double>&,const double*,int,const std::complex<double>*,int,const std::complex<double>&,std::complex<double>*,int,void*,std::size_t,bool,int);
template void hemm<DdlaBackend::CPU,double>(const DdlaHandle_t&,char,char,int,int,const std::complex<double>&,const std::complex<double>*,int,const double*,int,const std::complex<double>&,std::complex<double>*,int);
template void hemmt<DdlaBackend::CPU,double>(const DdlaHandle_t&,char,char,char,int,const std::complex<double>&,const std::complex<double>*,int,const double*,int,const std::complex<double>&,std::complex<double>*,int);
template bool blas_impl::hemmt_check<DdlaBackend::CPU,std::complex<double>,std::complex<double>,double>(const DdlaHandle_t&,char,char,char,int,const std::complex<double>&,const std::complex<double>*,int,const double*,int,const std::complex<double>&,std::complex<double>*,int,const char*,int);
template void blas_impl::hemmt_run<DdlaBackend::CPU,std::complex<double>,std::complex<double>,double>(const DdlaHandle_t&,char,char,char,int,const std::complex<double>&,const std::complex<double>*,int,const double*,int,const std::complex<double>&,std::complex<double>*,int,void*,std::size_t,bool,int);
#if DDLA_HAS_GPU
template void hemm<DdlaBackend::GPU,float>(const DdlaHandle_t&,char,char,int,int,const std::complex<float>&,const float*,int,const std::complex<float>*,int,const std::complex<float>&,std::complex<float>*,int);
template void hemmt<DdlaBackend::GPU,float>(const DdlaHandle_t&,char,char,char,int,const std::complex<float>&,const float*,int,const std::complex<float>*,int,const std::complex<float>&,std::complex<float>*,int);
template bool blas_impl::hemmt_check<DdlaBackend::GPU,std::complex<float>,float,std::complex<float>>(const DdlaHandle_t&,char,char,char,int,const std::complex<float>&,const float*,int,const std::complex<float>*,int,const std::complex<float>&,std::complex<float>*,int,const char*,int);
template void blas_impl::hemmt_run<DdlaBackend::GPU,std::complex<float>,float,std::complex<float>>(const DdlaHandle_t&,char,char,char,int,const std::complex<float>&,const float*,int,const std::complex<float>*,int,const std::complex<float>&,std::complex<float>*,int,void*,std::size_t,bool,int);
template void hemm<DdlaBackend::GPU,float>(const DdlaHandle_t&,char,char,int,int,const std::complex<float>&,const std::complex<float>*,int,const float*,int,const std::complex<float>&,std::complex<float>*,int);
template void hemmt<DdlaBackend::GPU,float>(const DdlaHandle_t&,char,char,char,int,const std::complex<float>&,const std::complex<float>*,int,const float*,int,const std::complex<float>&,std::complex<float>*,int);
template bool blas_impl::hemmt_check<DdlaBackend::GPU,std::complex<float>,std::complex<float>,float>(const DdlaHandle_t&,char,char,char,int,const std::complex<float>&,const std::complex<float>*,int,const float*,int,const std::complex<float>&,std::complex<float>*,int,const char*,int);
template void blas_impl::hemmt_run<DdlaBackend::GPU,std::complex<float>,std::complex<float>,float>(const DdlaHandle_t&,char,char,char,int,const std::complex<float>&,const std::complex<float>*,int,const float*,int,const std::complex<float>&,std::complex<float>*,int,void*,std::size_t,bool,int);
template void hemm<DdlaBackend::GPU,double>(const DdlaHandle_t&,char,char,int,int,const std::complex<double>&,const double*,int,const std::complex<double>*,int,const std::complex<double>&,std::complex<double>*,int);
template void hemmt<DdlaBackend::GPU,double>(const DdlaHandle_t&,char,char,char,int,const std::complex<double>&,const double*,int,const std::complex<double>*,int,const std::complex<double>&,std::complex<double>*,int);
template bool blas_impl::hemmt_check<DdlaBackend::GPU,std::complex<double>,double,std::complex<double>>(const DdlaHandle_t&,char,char,char,int,const std::complex<double>&,const double*,int,const std::complex<double>*,int,const std::complex<double>&,std::complex<double>*,int,const char*,int);
template void blas_impl::hemmt_run<DdlaBackend::GPU,std::complex<double>,double,std::complex<double>>(const DdlaHandle_t&,char,char,char,int,const std::complex<double>&,const double*,int,const std::complex<double>*,int,const std::complex<double>&,std::complex<double>*,int,void*,std::size_t,bool,int);
template void hemm<DdlaBackend::GPU,double>(const DdlaHandle_t&,char,char,int,int,const std::complex<double>&,const std::complex<double>*,int,const double*,int,const std::complex<double>&,std::complex<double>*,int);
template void hemmt<DdlaBackend::GPU,double>(const DdlaHandle_t&,char,char,char,int,const std::complex<double>&,const std::complex<double>*,int,const double*,int,const std::complex<double>&,std::complex<double>*,int);
template bool blas_impl::hemmt_check<DdlaBackend::GPU,std::complex<double>,std::complex<double>,double>(const DdlaHandle_t&,char,char,char,int,const std::complex<double>&,const std::complex<double>*,int,const double*,int,const std::complex<double>&,std::complex<double>*,int,const char*,int);
template void blas_impl::hemmt_run<DdlaBackend::GPU,std::complex<double>,std::complex<double>,double>(const DdlaHandle_t&,char,char,char,int,const std::complex<double>&,const std::complex<double>*,int,const double*,int,const std::complex<double>&,std::complex<double>*,int,void*,std::size_t,bool,int);
#endif
template void blas_impl::hemmt_workspace_size<std::complex<float>,float,std::complex<float>>(const DdlaHandle_t&,int,bool,std::size_t*,int,char,char);
template void blas_impl::hemmt_workspace_size<std::complex<float>,std::complex<float>,float>(const DdlaHandle_t&,int,bool,std::size_t*,int,char,char);
template void blas_impl::hemmt_workspace_size<std::complex<double>,double,std::complex<double>>(const DdlaHandle_t&,int,bool,std::size_t*,int,char,char);
template void blas_impl::hemmt_workspace_size<std::complex<double>,std::complex<double>,double>(const DdlaHandle_t&,int,bool,std::size_t*,int,char,char);
}
