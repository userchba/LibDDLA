#include "blas_impl.h"
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "gemmt_impl.h"
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "helpers.h"
#include "require_backend.h"
#include "plan_util.h"

#include <algorithm>
#include <cctype>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

#if DDLA_CPU_HAS_GEMMT || DDLA_CPU_HAS_GEMMTR
extern "C" {
#if DDLA_CPU_HAS_GEMMT
char* openblas_get_config(void) __attribute__((weak));
#endif
#define DECLARE_GEMMT(NAME, T) \
void NAME(const char*, const char*, const char*, const int*, const int*, \
          const T*, const T*, const int*, const T*, const int*, const T*, T*, const int*)
#if DDLA_CPU_HAS_GEMMT
DECLARE_GEMMT(sgemmt_, float);
DECLARE_GEMMT(dgemmt_, double);
DECLARE_GEMMT(cgemmt_, std::complex<float>);
DECLARE_GEMMT(zgemmt_, std::complex<double>);
#else
DECLARE_GEMMT(sgemmtr_, float);
DECLARE_GEMMT(dgemmtr_, double);
DECLARE_GEMMT(cgemmtr_, std::complex<float>);
DECLARE_GEMMT(zgemmtr_, std::complex<double>);
#endif
#undef DECLARE_GEMMT
}
#endif

namespace ddla { namespace blas_impl { namespace {

constexpr int tile_size = 128;

// Native OpenBLAS 0.3.29 GEMMT is bypassed below for correctness/performance.
// Other providers keep their native triangular-product implementation.
bool openblas_0329()
{
#if DDLA_CPU_HAS_GEMMT
    const char* config=openblas_get_config ? openblas_get_config() : nullptr;
    constexpr char version[]="OpenBLAS 0.3.29";
    constexpr std::size_t length=sizeof(version)-1;
    return config && std::strncmp(config,version,length)==0 &&
        (config[length]=='\0' || std::isspace(static_cast<unsigned char>(config[length])));
#else
    return false;
#endif
}

template <typename T>
int batch_tile(const DdlaHandle_t& h,char ta,char tb,int n,int k)
{
    // Only measured profitable cases select the allocating public path.
    if (h->backend==DdlaBackend::GPU) {
        if constexpr (sizeof(typename real_type<T>::type)!=8) return 0;
        if (n<4096 || k<256 || upper(ta)!='N' || upper(tb)!='N') return 0;
        if constexpr (!std::is_arithmetic_v<T>) if (k>1024) return 0;
    } else {
        if (h->host_threads<8 || n<1024 || k<512) return 0;
#if DDLA_CPU_HAS_GEMMT
        if (!openblas_0329()) return 0;
#elif DDLA_CPU_HAS_GEMMTR
        return 0;
#endif
    }
    const std::size_t tiles=(std::size_t(n)+tile_size-1)/tile_size;
    if (tiles*(tiles+1)/2>std::size_t(std::numeric_limits<int>::max())) return 0;
    return tile_size;
}


template <typename T>
const T* a_rows(const T* a, int lda, char trans, int first)
{ return a + (trans == 'N' ? std::size_t(first) : std::size_t(first)*lda); }

template <typename T>
const T* b_cols(const T* b, int ldb, char trans, int first)
{ return b + (trans == 'N' ? std::size_t(first)*ldb : std::size_t(first)); }

// For BLAS providers without GEMMT/GEMMTR, split
// each diagonal square, issuing GEMM only for the selected off-diagonal
// rectangle. At the leaves each diagonal is one dot product, so no entry of
// the opposite triangle is computed or read.
template <DdlaBackend Backend, typename T>
void recursive_triangle(const DdlaHandle_t& h, char uplo, char ta, char tb,
    int n, int k, const T& alpha, const T* a, int lda, const T* b, int ldb,
    const T& beta, T* c, int ldc)
{
    if constexpr (Backend==DdlaBackend::GPU) {
        // All diagonal leaves are launched together after the large rectangles.
        if (n<=tile_size) return;
    } else if (n==1) {
        blas_impl::gemm_run<Backend,T>(h,ta,tb,1,1,k,alpha,a,lda,b,ldb,beta,c,ldc);
        return;
    }
    const int first = Backend==DdlaBackend::GPU ? ((n-1)/tile_size+1)/2*tile_size : n/2;
    const int second = n-first;
    const T* ar = a_rows(a,lda,ta,first);
    const T* bc = b_cols(b,ldb,tb,first);
    if (uplo == 'U')
        blas_impl::gemm_run<Backend,T>(h,ta,tb,first,second,k,alpha,a,lda,bc,ldb,beta,
                        c+std::size_t(first)*ldc,ldc);
    else
        blas_impl::gemm_run<Backend,T>(h,ta,tb,second,first,k,alpha,ar,lda,b,ldb,beta,c+first,ldc);
    recursive_triangle<Backend,T>(h,uplo,ta,tb,first,k,alpha,a,lda,b,ldb,beta,c,ldc);
    recursive_triangle<Backend,T>(h,uplo,ta,tb,second,k,alpha,ar,lda,bc,ldb,beta,
                                   c+first+std::size_t(first)*ldc,ldc);
}

#if DDLA_GPU_VENDOR
// std::complex need not have the alignment required by thrust::complex.
// Read/write its real components explicitly, as in the existing TRTRM path.
template <typename T> struct Global {
    __device__ static T load(const void* p, std::size_t i) { return static_cast<const T*>(p)[i]; }
    __device__ static void store(void* p, std::size_t i, T x) { static_cast<T*>(p)[i] = x; }
};
template <typename R> struct Global<thrust::complex<R>> {
    __device__ static thrust::complex<R> load(const void* p, std::size_t i) {
        const R* x = static_cast<const R*>(p)+2*i;
        return {x[0],x[1]};
    }
    __device__ static void store(void* p, std::size_t i, thrust::complex<R> x) {
        R* y = static_cast<R*>(p)+2*i;
        y[0]=x.real(); y[1]=x.imag();
    }
};

// GEAM works on rectangles. These small copy/scale kernels are needed to
// restrict access to one triangle, particularly with an undefined other half.
template <typename T, bool ReadC>
__global__ void scale_triangle_kernel(bool upper, int n, T beta, void* c, int ldc)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    if (i>=n || j>=n || (upper ? i>j : i<j)) return;
    const std::size_t p=i+std::size_t(j)*ldc;
    T x(0);
    if constexpr (ReadC) x=beta*Global<T>::load(c,p);
    Global<T>::store(c,p,x);
}

template <typename R>
__global__ void real_diagonal_kernel(int n, R* c, int ldc)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) c[2*(std::size_t(i)+std::size_t(i)*ldc)+1]=R(0);
}

// The supported CUDA/HIP BLAS interfaces have no general GEMMT for arbitrary
// partial products. SYRKX requires symmetric A*B^T, which does not hold for the
// k-panels of a Hermitian product. Only diagonal blocks use this kernel;
// strictly off-diagonal rectangles still use the optimized vendor GEMM.
// Every block outside the selected triangle returns uniformly. Boundary
// threads still load shared tiles and reach both barriers, but they never
// accumulate a dot product for an unselected output element.
template <typename T, typename R, bool ReadC>
__global__ void diagonal_gemm_kernel(bool upper, char ta, char tb, int n, int k,
    T alpha, const void* a, int lda, const void* b, int ldb, T beta, void* c, int ldc)
{
    constexpr int block_size=16, components=sizeof(T)/sizeof(R);
    // Independent diagonal leaves share one launch, rather than serializing
    // an under-filled kernel for every column panel.
    const std::size_t first=std::size_t(blockIdx.z)*tile_size;
    a=static_cast<const char*>(a)+(ta=='N' ? first : first*lda)*sizeof(T);
    b=static_cast<const char*>(b)+(tb=='N' ? first*ldb : first)*sizeof(T);
    c=static_cast<char*>(c)+(first+first*ldc)*sizeof(T);
    n=min(tile_size,n-int(first));
    if (upper ? blockIdx.x>blockIdx.y : blockIdx.x<blockIdx.y) return;
    __shared__ R as[block_size*block_size*components];
    __shared__ R bs[block_size*block_size*components];
    const int tx=threadIdx.x, ty=threadIdx.y;
    const int i=blockIdx.x*block_size+tx, j=blockIdx.y*block_size+ty;
    const bool selected=i<n && j<n && (upper ? i<=j : i>=j);
    // Shorten each floating-point dependency chain when several descriptor
    // blocks share one contraction panel, then combine the partial sums pairwise.
    T value[8]={};
    for (std::size_t first=0;first<std::size_t(k);first+=block_size) {
        const std::size_t pa=first+ty, pb=first+tx;
        T av(0),bv(0);
        if (i<n && pa<std::size_t(k)) {
            av=Global<T>::load(a,ta=='N' ? i+pa*lda : pa+std::size_t(i)*lda);
            av=detail::device_conj(av,ta=='C');
        }
        if (j<n && pb<std::size_t(k)) {
            bv=Global<T>::load(b,tb=='N' ? pb+std::size_t(j)*ldb : j+pb*ldb);
            bv=detail::device_conj(bv,tb=='C');
        }
        Global<T>::store(as,tx+ty*block_size,av);
        Global<T>::store(bs,tx+ty*block_size,bv);
        __syncthreads();
        if (selected) {
            #pragma unroll
            for (int p=0;p<block_size;++p)
                value[p%8]+=Global<T>::load(as,tx+p*block_size)*Global<T>::load(bs,p+ty*block_size);
        }
        __syncthreads();
    }
    if (selected) {
        const std::size_t at=i+std::size_t(j)*ldc;
        T result=alpha*(((value[0]+value[1])+(value[2]+value[3]))+((value[4]+value[5])+(value[6]+value[7])));
        if constexpr (ReadC) result+=beta*Global<T>::load(c,at);
        Global<T>::store(c,at,result);
    }
}
// The selected output tiles have three uniform groups and at most one
// partial diagonal tile. Build their pointer tables on the device so the
// public wrapper needs no temporary host storage or extra synchronization.
template <typename T>
__global__ void triangular_pointers(bool upper,int n,char ta,char tb,
    const T* a,int lda,const T* b,int ldb,T* c,int ldc,T* temporary,
    T** pointers,std::size_t count)
{
    const std::size_t at=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (at>=count) return;
    const int full=n/tile_size,tail=n%tile_size;
    const std::size_t rectangles=std::size_t(full)*(full-1)/2;
    const std::size_t edge=tail ? full : 0;
    int i,j;
    T* output;
    if (at<rectangles) {
        j=int((1+sqrt(1+8.0*double(at)))/2);
        while (std::size_t(j)*(j-1)/2>at) --j;
        while (std::size_t(j)*(j+1)/2<=at) ++j;
        i=int(at-std::size_t(j)*(j-1)/2);
        if (!upper) {const int t=i;i=j;j=t;}
        output=c+i*tile_size+std::size_t(j*tile_size)*ldc;
    } else if (at<rectangles+edge) {
        const int t=int(at-rectangles);
        i=upper ? t : full;j=upper ? full : t;
        output=c+i*tile_size+std::size_t(j*tile_size)*ldc;
    } else {
        i=j=int(at-rectangles-edge);
        output=temporary+std::size_t(i)*tile_size*tile_size;
    }
    pointers[at]=const_cast<T*>(a+(ta=='N' ? std::size_t(i)*tile_size : std::size_t(i)*tile_size*lda));
    pointers[count+at]=const_cast<T*>(b+(tb=='N' ? std::size_t(j)*tile_size*ldb : std::size_t(j)*tile_size));
    pointers[2*count+at]=output;
}

template <typename T,bool ReadC>
__global__ void copy_diagonal_triangle(bool upper,int n,const void* temporary,T beta,void* c,int ldc)
{
    const int first=blockIdx.z*tile_size;
    const int i=blockIdx.x*16+threadIdx.x,j=blockIdx.y*16+threadIdx.y;
    if (first+i>=n || first+j>=n || (upper ? i>j : i<j)) return;
    const std::size_t to=first+i+std::size_t(first+j)*ldc;
    T value=Global<T>::load(temporary,std::size_t(blockIdx.z)*tile_size*tile_size+i+std::size_t(j)*tile_size);
    if constexpr (ReadC) value+=beta*Global<T>::load(c,to);
    Global<T>::store(c,to,value);
}

#endif

template <DdlaBackend Backend, typename T>
void scale_triangle(const DdlaHandle_t& h, char uplo, int n, const T& beta, T* c, int ldc)
{
    if (beta==T(1)) return;
    if constexpr (Backend==DdlaBackend::CPU) {
        for (int j=0;j<n;++j) {
            const int begin=uplo=='U' ? 0 : j, end=uplo=='U' ? j+1 : n;
            for (int i=begin;i<end;++i) {
                T& x=c[i+std::size_t(j)*ldc];
                x=beta==T(0) ? T(0) : beta*x;
            }
        }
    } else {
#if DDLA_GPU_VENDOR
        using D=detail::DeviceScalarT<T>;
        const dim3 grid((n-1)/16+1,(n-1)/16+1), block(16,16);
        if (beta==T(0))
            scale_triangle_kernel<D,false><<<grid,block,0,static_cast<gpuStream_t>(h->stream)>>>(uplo=='U',n,D(0),c,ldc);
        else
            scale_triangle_kernel<D,true><<<grid,block,0,static_cast<gpuStream_t>(h->stream)>>>(uplo=='U',n,detail::to_device_scalar(beta),c,ldc);
        DDLA_CHECK(ddlaGetLastError(h));
#endif
    }
}

// Full diagonal products go to O(n*tile_size) scratch; only uplo is copied
// back. Rectangles write C directly, and every selected entry receives beta
// exactly once. This core allocates neither backend nor host memory.
template <DdlaBackend Backend,typename T>
void batched_triangle(const DdlaHandle_t& h,char uplo,char ta,char tb,int n,int k,
    const T& alpha,const T* a,int lda,const T* b,int ldb,const T& beta,T* c,int ldc,void* workspace)
{
    const bool upper_triangle=uplo=='U';
    const int full=n/tile_size,tail=n%tile_size;
    const std::size_t tiles=std::size_t(full)+(tail!=0),count=tiles*(tiles+1)/2;
    char* cursor=static_cast<char*>(workspace);
    T** pointers=detail::take<T*>(cursor,3*count);
    T* temporary=detail::take<T>(cursor,tiles*tile_size*tile_size);
    if constexpr (Backend==DdlaBackend::GPU) {
#if DDLA_GPU_VENDOR
        triangular_pointers<T><<<static_cast<unsigned>((count+255)/256),256,0,static_cast<gpuStream_t>(h->stream)>>>(
            upper_triangle,n,ta,tb,a,lda,b,ldb,c,ldc,temporary,pointers,count);
        DDLA_CHECK(ddlaGetLastError(h));
#endif
    } else {
        std::size_t at=0;
        auto put=[&](int i,int j,T* output) {
            pointers[at]=const_cast<T*>(a_rows(a,lda,ta,i*tile_size));
            pointers[count+at]=const_cast<T*>(b_cols(b,ldb,tb,j*tile_size));
            pointers[2*count+at]=output;++at;
        };
        for (int j=0;j<full;++j)
            for (int i=upper_triangle ? 0 : j+1;i<(upper_triangle ? j : full);++i)
                put(i,j,c+i*tile_size+std::size_t(j*tile_size)*ldc);
        if (tail) for (int q=0;q<full;++q) {
            const int i=upper_triangle ? q : full,j=upper_triangle ? full : q;
            put(i,j,c+i*tile_size+std::size_t(j*tile_size)*ldc);
        }
        for (int q=0;q<full;++q) put(q,q,temporary+std::size_t(q)*tile_size*tile_size);
        if (tail) put(full,full,temporary+std::size_t(full)*tile_size*tile_size);
    }
    const std::size_t counts[4]={std::size_t(full)*(full-1)/2,tail ? std::size_t(full) : 0,std::size_t(full),std::size_t(tail!=0)};
    std::size_t first=0;
    for (int group=0;group<4;++group) {
        const int batch=static_cast<int>(counts[group]);
        if (batch==0) continue;
        const int rows=group==1 ? (upper_triangle ? tile_size : tail) : group==3 ? tail : tile_size;
        const int cols=group==1 ? (upper_triangle ? tail : tile_size) : group==3 ? tail : tile_size;
        const int ldout=group<2 ? ldc : tile_size;
        const T scale=group<2 ? beta : T(0);
        // Pointer tables reside in backend memory, including for a single
        // product, so no host read or device-to-host synchronization is needed.
        blas_impl::gemmBatched_run<Backend,T>(h,ta,tb,rows,cols,k,alpha,
            pointers+first,lda,pointers+count+first,ldb,scale,pointers+2*count+first,ldout,batch);
        first+=counts[group];
    }
    if constexpr (Backend==DdlaBackend::GPU) {
#if DDLA_GPU_VENDOR
        using D=detail::DeviceScalarT<T>;
        const dim3 grid(tile_size/16,tile_size/16,static_cast<unsigned>(tiles)),block(16,16);
        if (beta==T(0)) copy_diagonal_triangle<D,false><<<grid,block,0,static_cast<gpuStream_t>(h->stream)>>>(upper_triangle,n,temporary,D(0),c,ldc);
        else copy_diagonal_triangle<D,true><<<grid,block,0,static_cast<gpuStream_t>(h->stream)>>>(upper_triangle,n,temporary,detail::to_device_scalar(beta),c,ldc);
        DDLA_CHECK(ddlaGetLastError(h));
#endif
    } else {
        for (int column=0;column<n;++column) {
            const int first_row=column/tile_size*tile_size,j=column%tile_size;
            const int width=std::min(tile_size,n-first_row),begin=upper_triangle ? 0 : j,end=upper_triangle ? j+1 : width;
            for (int i=begin;i<end;++i) {
                T value=temporary[std::size_t(column/tile_size)*tile_size*tile_size+i+std::size_t(j)*tile_size];
                T& output=c[first_row+i+std::size_t(column)*ldc];
                if (beta!=T(0)) value+=beta*output;
                output=value;
            }
        }
    }
}

}} // namespace blas_impl::<anonymous>

namespace blas_impl {

template <typename T>
void gemmt_workspace_size(const DdlaHandle_t& h,char ta,char tb,int n,int k,std::size_t* backend_bytes)
{
    const int block=batch_tile<T>(h,ta,tb,n,k);
    *backend_bytes=0;
    if (block==0) return;
    const std::size_t tiles=(std::size_t(n)+block-1)/block,count=tiles*(tiles+1)/2;
    *backend_bytes=detail::padded_bytes<T*>(3*count)+detail::padded_bytes<T>(tiles*block*block);
}

template <DdlaBackend Backend, typename T>
void gemmt_run(const DdlaHandle_t& h,
    char uplo, char ta, char tb, int n, int k,
    const T& alpha, const T* a, int lda, const T* b, int ldb,
    const T& beta, T* c, int ldc,void* backend_work,std::size_t backend_bytes)
{
    uplo=upper(uplo); ta=upper(ta); tb=upper(tb);
    if (n==0) return;
    if (alpha==T(0) || k==0) { scale_triangle<Backend,T>(h,uplo,n,beta,c,ldc); return; }
    if (backend_work && backend_bytes && batch_tile<T>(h,ta,tb,n,k)) {
        batched_triangle<Backend,T>(h,uplo,ta,tb,n,k,alpha,a,lda,b,ldb,beta,c,ldc,backend_work);
        return;
    }
    if constexpr (Backend==DdlaBackend::CPU) {
#if DDLA_CPU_HAS_GEMMT
        // OpenBLAS 0.3.29 GEMMT conjugates B in place for complex TRANSB=C
        // without restoring it. Keep const inputs immutable on that path.
        if constexpr (!std::is_arithmetic_v<T>) {
            if (tb=='C') {
                recursive_triangle<Backend,T>(h,uplo,ta,tb,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
                return;
            }
        }
        // This release implements GEMMT as repeated GEMV. Measured CPU
        // products are faster through the existing blocked GEMM fallback.
        // Keep other providers/releases on their native GEMMT implementation.
        if (openblas_0329()) {
            recursive_triangle<Backend,T>(h,uplo,ta,tb,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
            return;
        }
        if constexpr (std::is_same_v<T,float>)
            sgemmt_(&uplo,&ta,&tb,&n,&k,&alpha,a,&lda,b,&ldb,&beta,c,&ldc);
        else if constexpr (std::is_same_v<T,double>)
            dgemmt_(&uplo,&ta,&tb,&n,&k,&alpha,a,&lda,b,&ldb,&beta,c,&ldc);
        else if constexpr (std::is_same_v<T,std::complex<float>>)
            cgemmt_(&uplo,&ta,&tb,&n,&k,&alpha,a,&lda,b,&ldb,&beta,c,&ldc);
        else
            zgemmt_(&uplo,&ta,&tb,&n,&k,&alpha,a,&lda,b,&ldb,&beta,c,&ldc);
#elif DDLA_CPU_HAS_GEMMTR
        if constexpr (std::is_same_v<T,float>)
            sgemmtr_(&uplo,&ta,&tb,&n,&k,&alpha,a,&lda,b,&ldb,&beta,c,&ldc);
        else if constexpr (std::is_same_v<T,double>)
            dgemmtr_(&uplo,&ta,&tb,&n,&k,&alpha,a,&lda,b,&ldb,&beta,c,&ldc);
        else if constexpr (std::is_same_v<T,std::complex<float>>)
            cgemmtr_(&uplo,&ta,&tb,&n,&k,&alpha,a,&lda,b,&ldb,&beta,c,&ldc);
        else
            zgemmtr_(&uplo,&ta,&tb,&n,&k,&alpha,a,&lda,b,&ldb,&beta,c,&ldc);
#else
        recursive_triangle<Backend,T>(h,uplo,ta,tb,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
#endif
    } else {
#if DDLA_GPU_VENDOR
        recursive_triangle<Backend,T>(h,uplo,ta,tb,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
        using D=detail::DeviceScalarT<T>;
        using R=typename detail::DeviceScalar<T>::real;
        const dim3 grid((std::min(n,tile_size)-1)/16+1,
                        (std::min(n,tile_size)-1)/16+1,(n-1)/tile_size+1), block(16,16);
        if (beta==T(0))
            diagonal_gemm_kernel<D,R,false><<<grid,block,0,static_cast<gpuStream_t>(h->stream)>>>(
                uplo=='U',ta,tb,n,k,detail::to_device_scalar(alpha),a,lda,b,ldb,D(0),c,ldc);
        else
            diagonal_gemm_kernel<D,R,true><<<grid,block,0,static_cast<gpuStream_t>(h->stream)>>>(
                uplo=='U',ta,tb,n,k,detail::to_device_scalar(alpha),a,lda,b,ldb,detail::to_device_scalar(beta),c,ldc);
        DDLA_CHECK(ddlaGetLastError(h));
#endif
    }
}

template <DdlaBackend Backend, typename T>
void real_diagonal(const DdlaHandle_t& h, int n, T* c, int ldc)
{
    if constexpr (!std::is_arithmetic_v<T>) {
        if (n==0) return;
        using R=typename T::value_type;
        if constexpr (Backend==DdlaBackend::CPU) {
            for (int i=0;i<n;++i) c[i+std::size_t(i)*ldc].imag(R(0));
        } else {
#if DDLA_GPU_VENDOR
            real_diagonal_kernel<R><<<(n-1)/256+1,256,0,static_cast<gpuStream_t>(h->stream)>>>(n,reinterpret_cast<R*>(c),ldc);
            DDLA_CHECK(ddlaGetLastError(h));
#endif
        }
    }
}

template <DdlaBackend Backend, typename T>
void gemmt_check(const DdlaHandle_t& h, char uplo, char ta, char tb,
    int n, int k, const T& alpha, const T* a, int lda, const T* b, int ldb,
    const T& beta, T* c, int ldc)
{
    detail::require_backend<Backend>(h,"gemmt");
    uplo=static_cast<char>(std::toupper(static_cast<unsigned char>(uplo)));
    ta=static_cast<char>(std::toupper(static_cast<unsigned char>(ta)));
    tb=static_cast<char>(std::toupper(static_cast<unsigned char>(tb)));
    if ((uplo!='U' && uplo!='L') ||
        (ta!='N' && ta!='T' && ta!='C') || (tb!='N' && tb!='T' && tb!='C'))
        throw std::invalid_argument("gemmt: expected triangle U/L and operations N/T/C");
    if (n<0 || k<0 || lda<std::max(1,ta=='N' ? n : k) ||
        ldb<std::max(1,tb=='N' ? k : n) || ldc<std::max(1,n))
        throw std::invalid_argument("gemmt: invalid dimensions or leading dimensions");
    if (n==0) return;
    if (!c) throw std::invalid_argument("gemmt: null output");
    const bool product=alpha!=T(0) && k!=0;
    if (product) {
        if (!a || !b) throw std::invalid_argument("gemmt: null input");
        if (detail::matrix_overlaps(a,lda,ta=='N' ? n : k,ta=='N' ? k : n,c,ldc,n,n) ||
            detail::matrix_overlaps(b,ldb,tb=='N' ? k : n,tb=='N' ? n : k,c,ldc,n,n))
            throw std::invalid_argument("gemmt: output must not overlap either input");
    }
}
} // namespace blas_impl
template <DdlaBackend Backend, typename T>
void gemmt(const DdlaHandle_t& h, char uplo, char ta, char tb,
    int n, int k, const T& alpha, const T* a, int lda, const T* b, int ldb,
    const T& beta, T* c, int ldc)
{
    blas_impl::gemmt_check<Backend,T>(h,uplo,ta,tb,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
    std::size_t backend_bytes=0;
    if (alpha!=T(0) && k>0) blas_impl::gemmt_workspace_size<T>(h,ta,tb,n,k,&backend_bytes);
    vector<char> workspace(h,backend_bytes);
    blas_impl::gemmt_run<Backend,T>(h,uplo,ta,tb,n,k,alpha,a,lda,b,ldb,beta,c,ldc,workspace.data(),backend_bytes);
}

#define INSTANTIATE(B,T) \
template void gemmt<B,T>(const DdlaHandle_t&,char,char,char,int,int,const T&,const T*,int,const T*,int,const T&,T*,int); \
template void blas_impl::gemmt_check<B,T>(const DdlaHandle_t&,char,char,char,int,int,const T&,const T*,int,const T*,int,const T&,T*,int); \
template void blas_impl::gemmt_run<B,T>(const DdlaHandle_t&,char,char,char,int,int,const T&,const T*,int,const T*,int,const T&,T*,int,void*,std::size_t); \
template void blas_impl::real_diagonal<B,T>(const DdlaHandle_t&,int,T*,int)
#define INSTANTIATE_TYPES(B) \
INSTANTIATE(B,float); INSTANTIATE(B,double); \
INSTANTIATE(B,std::complex<float>); INSTANTIATE(B,std::complex<double>)
INSTANTIATE_TYPES(DdlaBackend::CPU);
#if DDLA_HAS_GPU
INSTANTIATE_TYPES(DdlaBackend::GPU);
#endif
#define INSTANTIATE_SIZE(T) template void blas_impl::gemmt_workspace_size<T>(const DdlaHandle_t&,char,char,int,int,std::size_t*)
INSTANTIATE_SIZE(float); INSTANTIATE_SIZE(double);
INSTANTIATE_SIZE(std::complex<float>); INSTANTIATE_SIZE(std::complex<double>);
#undef INSTANTIATE_SIZE
#undef INSTANTIATE_TYPES
#undef INSTANTIATE
} // namespace ddla
