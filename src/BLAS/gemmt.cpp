#include "blas_impl.h"
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "gemmt_impl.h"
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "helpers.h"
#include "require_backend.h"
#include "plan_util.h"
#include "instantiate.h"

#include <algorithm>
#include <cctype>
#include <complex>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace ddla { namespace blas_impl { namespace {

constexpr int tile_size = 128;

template <typename T>
int output_tile_size(const DdlaHandle_t& h)
{
    return h->backend==DdlaBackend::GPU && std::is_arithmetic_v<T> ? 2*tile_size : tile_size;
}

// Large rectangles cover the strictly selected coarse blocks. Their narrow
// diagonal boundary stays in 128-wide batches, so coalescing does not add
// discarded diagonal products or enlarge the temporary matrices.
void local_batch_counts(int n, int block, std::size_t (&counts)[6])
{
    const int full=n/block,edge=n%block,ratio=block/tile_size;
    const int small=edge/tile_size,tail=edge%tile_size;
    counts[0]=std::size_t(full)*(full-1)/2;
    counts[1]=edge ? full : 0;
    counts[2]=std::size_t(full)*ratio*(ratio-1)/2+std::size_t(small)*(small-1)/2;
    counts[3]=tail ? small : 0;
    counts[4]=n/tile_size;
    counts[5]=tail!=0;
}

template <typename T>
const T* a_rows(const T* a, int lda, char trans, int first)
{ return a + (trans == 'N' ? std::size_t(first) : std::size_t(first)*lda); }

template <typename T>
const T* b_cols(const T* b, int ldb, char trans, int first)
{ return b + (trans == 'N' ? std::size_t(first)*ldb : std::size_t(first)); }

#if DDLA_HAS_GPU
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

// Nonzero scaling must read only the selected triangle. Zero fills use
// ddlaMemsetTri instead of duplicating its no-read kernel.
template <typename T>
__global__ void scale_triangle_kernel(bool upper, int n, T beta, void* c, int ldc)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    if (i>=n || j>=n || (upper ? i>j : i<j)) return;
    const std::size_t p=i+std::size_t(j)*ldc;
    Global<T>::store(c,p,beta*Global<T>::load(c,p));
}

template <typename R>
__global__ void real_diagonal_kernel(int n, R* c, int ldc)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) c[2*(std::size_t(i)+std::size_t(i)*ldc)+1]=R(0);
}

// The selected output tiles have three uniform groups and at most one
// partial diagonal tile. Build their pointer tables on the device so the
// public wrapper needs no temporary host storage or extra synchronization.
template <typename T, typename TA, typename TB>
__global__ void triangular_pointers(bool upper,int n,int block,char ta,char tb,
    const TA* a,int lda,const TB* b,int ldb,T* c,int ldc,T* temporary,
    const TA** ap,const TB** bp,T** cp,std::size_t count)
{
    const int full=n/block,edge=n%block,ratio=block/tile_size;
    const int small=edge/tile_size,tail=edge%tile_size;
    const std::size_t big_rectangles=std::size_t(full)*(full-1)/2;
    const std::size_t big_edges=edge ? full : 0;
    const std::size_t per_block=std::size_t(ratio)*(ratio-1)/2;
    const std::size_t small_rectangles=std::size_t(full)*per_block+std::size_t(small)*(small-1)/2;
    const std::size_t small_edges=tail ? small : 0;
    for (std::size_t at=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
         at<count;at+=std::size_t(blockDim.x)*gridDim.x) {
        std::size_t index=at;
        int i,j;
        T* output;
        if (index<big_rectangles) {
            j=int((1+sqrt(1+8.0*double(index)))/2);
            while (std::size_t(j)*(j-1)/2>index) --j;
            while (std::size_t(j)*(j+1)/2<=index) ++j;
            i=int(index-std::size_t(j)*(j-1)/2);
            i*=block;j*=block;
        } else if ((index-=big_rectangles)<big_edges) {
            i=int(index)*block;j=full*block;
        } else if ((index-=big_edges)<small_rectangles) {
            const int base=int(index/per_block)*block;
            index%=per_block;
            j=int((1+sqrt(1+8.0*double(index)))/2);
            while (std::size_t(j)*(j-1)/2>index) --j;
            while (std::size_t(j)*(j+1)/2<=index) ++j;
            i=int(index-std::size_t(j)*(j-1)/2);
            i=base+i*tile_size;j=base+j*tile_size;
        } else if ((index-=small_rectangles)<small_edges) {
            i=full*block+int(index)*tile_size;j=n-tail;
        } else {
            index-=small_edges;
            i=j=int(index)*tile_size;
            output=temporary+index*tile_size*tile_size;
            ap[at]=(a+(ta=='N' ? std::size_t(i) : std::size_t(i)*lda));
            bp[at]=(b+(tb=='N' ? std::size_t(j)*ldb : std::size_t(j)));
            cp[at]=output;
            continue;
        }
        if (!upper) {const int swap=i;i=j;j=swap;}
        output=c+i+std::size_t(j)*ldc;
        ap[at]=(a+(ta=='N' ? std::size_t(i) : std::size_t(i)*lda));
        bp[at]=(b+(tb=='N' ? std::size_t(j)*ldb : std::size_t(j)));
        cp[at]=output;
    }
}

template <typename T,bool ReadC>
__global__ void copy_diagonal_triangle(bool upper,int block,int full,int tail,int first_tile,
    const void* temporary,T beta,void* c,int ldc,void* const* targets,const int* widths)
{
    const int tile=first_tile+blockIdx.z;
    const int width=widths ? widths[tile] : tile<full ? block : tail;
    const int i=blockIdx.x*16+threadIdx.x,j=blockIdx.y*16+threadIdx.y;
    if (i>=width || j>=width || (upper ? i>j : i<j)) return;
    void* output=targets ? targets[tile] : static_cast<char*>(c)+std::size_t(tile)*block*(std::size_t(ldc)+1)*sizeof(T);
    const std::size_t to=i+std::size_t(j)*ldc;
    T value=Global<T>::load(temporary,std::size_t(tile)*block*block+i+std::size_t(j)*block);
    if constexpr (ReadC) value+=beta*Global<T>::load(output,to);
    Global<T>::store(output,to,value);
}

#endif

template <DdlaBackend Backend, typename T>
void scale_triangle(const DdlaHandle_t& h, char uplo, int n, const T& beta, T* c, int ldc)
{
    if (beta==T(1)) return;
    if (beta==T(0)) {
        DDLA_CHECK(ddlaMemsetTri(h, uplo, n, n, c, ldc));
        return;
    }
    if constexpr (Backend==DdlaBackend::CPU) {
        for (int j=0;j<n;++j) {
            const int begin=uplo=='U' ? 0 : j, end=uplo=='U' ? j+1 : n;
            for (int i=begin;i<end;++i) {
                T& x=c[i+std::size_t(j)*ldc];
                x=beta*x;
            }
        }
    } else {
#if DDLA_HAS_GPU
        using D=detail::DeviceScalarT<T>;
        const dim3 grid((n-1)/16+1,(n-1)/16+1), block(16,16);
        scale_triangle_kernel<D><<<grid,block,0,static_cast<gpuStream_t>(h->stream)>>>(uplo=='U',n,detail::to_device_scalar(beta),c,ldc);
        DDLA_CHECK(ddlaGetLastError(h));
#endif
    }
}

// Full diagonal products go to O(n*tile_size) scratch; only uplo is copied
// back. Rectangles write C directly, and every selected entry receives beta
// exactly once. This core allocates neither backend nor host memory.
template <DdlaBackend Backend, typename T, typename TA, typename TB>
void batched_triangle(const DdlaHandle_t& h,char uplo,char ta,char tb,int n,int k,
    const T& alpha,const TA* a,int lda,const TB* b,int ldb,const T& beta,T* c,int ldc,void* workspace,std::size_t bytes)
{
    const bool upper_triangle=uplo=='U';
    const int block=output_tile_size<T>(h),full=n/block,edge=n%block;
    const int ratio=block/tile_size,small=edge/tile_size,tail=n%tile_size;
    std::size_t counts[6];local_batch_counts(n,block,counts);
    std::size_t count=0;for (auto size:counts) count+=size;
    const std::size_t diagonals=(std::size_t(n)+tile_size-1)/tile_size;
    char* cursor=static_cast<char*>(workspace);
    char* pointers=detail::take<char>(cursor,3*count*sizeof(T*));
    T* temporary=detail::take<T>(cursor,diagonals*tile_size*tile_size);
    const TA** ap=reinterpret_cast<const TA**>(pointers);
    const TB** bp=reinterpret_cast<const TB**>(pointers+count*sizeof(T*));
    T** cp=reinterpret_cast<T**>(pointers+2*count*sizeof(T*));
    const std::size_t mixed_bytes=bytes-std::size_t(cursor-static_cast<char*>(workspace));
    if constexpr (Backend==DdlaBackend::GPU) {
#if DDLA_HAS_GPU
        triangular_pointers<T,TA,TB><<<static_cast<unsigned>(std::min<std::size_t>((count+255)/256,65535)),256,0,static_cast<gpuStream_t>(h->stream)>>>(
            upper_triangle,n,block,ta,tb,a,lda,b,ldb,c,ldc,temporary,ap,bp,cp,count);
        DDLA_CHECK(ddlaGetLastError(h));
#endif
    } else {
        std::size_t at=0;
        auto put=[&](int row,int column,T* output) {
            ap[at]=a_rows(a,lda,ta,row);
            bp[at]=b_cols(b,ldb,tb,column);
            cp[at]=output;++at;
        };
        auto rectangle=[&](int row,int column) {
            if (!upper_triangle) std::swap(row,column);
            put(row,column,c+row+std::size_t(column)*ldc);
        };
        for (int j=0;j<full;++j) for (int i=0;i<j;++i) rectangle(i*block,j*block);
        if (edge) for (int i=0;i<full;++i) rectangle(i*block,full*block);
        for (int q=0;q<=full;++q) {
            const int width=q<full ? ratio : small;
            for (int j=0;j<width;++j) for (int i=0;i<j;++i)
                rectangle(q*block+i*tile_size,q*block+j*tile_size);
        }
        if (tail) for (int i=0;i<small;++i) rectangle(full*block+i*tile_size,n-tail);
        for (std::size_t i=0;i<diagonals;++i)
            put(int(i)*tile_size,int(i)*tile_size,temporary+i*tile_size*tile_size);
    }
    std::size_t first=0;
    for (int group=0;group<6;++group) {
        if (counts[group]==0) continue;
        const int size=group<2 ? block : tile_size;
        const int partial=group==1 ? edge : tail;
        const bool edge_group=group==1 || group==3;
        const int rows=edge_group ? (upper_triangle ? size : partial) : group==5 ? tail : size;
        const int cols=edge_group ? (upper_triangle ? partial : size) : group==5 ? tail : size;
        const int ldout=group<4 ? ldc : tile_size;
        const T scale=group<4 ? beta : T(0);
        for (std::size_t done=0;done<counts[group];) {
            const int batch=static_cast<int>(std::min<std::size_t>(counts[group]-done,std::numeric_limits<int>::max()));
            blas_impl::gemmBatched_run<Backend,T>(h,ta,tb,rows,cols,k,alpha,
                ap+first+done,lda,bp+first+done,ldb,scale,cp+first+done,ldout,batch,cursor,mixed_bytes);
            done+=batch;
        }
        first+=counts[group];
    }
    blas_impl::gemmt_copy_run<Backend,T>(h,uplo,tile_size,n/tile_size,tail,
        beta,temporary,c,ldc,nullptr);

}

}} // namespace blas_impl::<anonymous>

namespace blas_impl {

template <DdlaBackend Backend, typename T>
void gemmt_copy_run(const DdlaHandle_t& h,char uplo,int block,int full,int tail,
    const T& beta,const T* temporary,T* C,int ldc,T* const* targets,const int* widths)
{
    const int count=full+(tail!=0);
    if (count==0) return;
    const bool upper_triangle=upper(uplo)=='U';
    if constexpr (Backend==DdlaBackend::CPU) {
        for (int tile=0;tile<count;++tile) {
            const int width=widths ? widths[tile] : tile<full ? block : tail;
            T* output=targets ? targets[tile] : C+std::size_t(tile)*block*(std::size_t(ldc)+1);
            const T* input=temporary+std::size_t(tile)*block*block;
            for (int column=0;column<width;++column) {
                const int first=upper_triangle ? 0 : column;
                const int last=upper_triangle ? column+1 : width;
                for (int row=first;row<last;++row) {
                    T value=input[row+std::size_t(column)*block];
                    T& out=output[row+std::size_t(column)*ldc];
                    if (beta!=T(0)) value+=beta*out;
                    out=value;
                }
            }
        }
    } else {
#if DDLA_HAS_GPU
        using D=detail::DeviceScalarT<T>;
        for (int first=0;first<count;) {
            const int batch=std::min(count-first,65535);
            const dim3 grid((block-1)/16+1,(block-1)/16+1,batch),threads(16,16);
            if (beta==T(0)) copy_diagonal_triangle<D,false><<<grid,threads,0,static_cast<gpuStream_t>(h->stream)>>>(upper_triangle,block,full,tail,first,temporary,D(0),C,ldc,reinterpret_cast<void* const*>(targets),widths);
            else copy_diagonal_triangle<D,true><<<grid,threads,0,static_cast<gpuStream_t>(h->stream)>>>(upper_triangle,block,full,tail,first,temporary,detail::to_device_scalar(beta),C,ldc,reinterpret_cast<void* const*>(targets),widths);
            DDLA_CHECK(ddlaGetLastError(h));
            first+=batch;
        }
#endif
    }
}

template <typename T, typename TA, typename TB>
void gemmt_workspace_size(const DdlaHandle_t& h,char,char,int n,int k,std::size_t* backend_bytes)
{
    *backend_bytes=0;
    if (n==0 || k==0) return;
    const std::size_t tiles=(std::size_t(n)+tile_size-1)/tile_size;
    std::size_t counts[6];local_batch_counts(n,output_tile_size<T>(h),counts);
    std::size_t count=0;for (auto size:counts) count+=size;
    *backend_bytes=detail::padded_bytes<T*>(3*count)+detail::padded_bytes<T>(tiles*tile_size*tile_size);
    if constexpr (!std::is_same_v<T,TA> || !std::is_same_v<T,TB>) {
        std::size_t mixed=0;
        const int block=output_tile_size<T>(h),edge=n%block,tail=n%tile_size;
        for (int group=0;group<6;++group) if (counts[group]) {
            const int size=group<2 ? block : tile_size;
            const int partial=group==1 ? edge : tail;
            // U/L swaps rectangle extents; include both in the bound.
            const int a=group==1 || group==3 ? partial : group==5 ? tail : size;
            const int b=group==5 ? tail : size;
            for (bool swap : {false,true}) {
                std::size_t bytes=0;
                local_gemmBatched_workspace_size<T,TA,TB>(h,swap?a:b,swap?b:a,k,
                    int(std::min<std::size_t>(counts[group],std::numeric_limits<int>::max())),&bytes);
                mixed=std::max(mixed,bytes);
            }
        }
        *backend_bytes+=mixed;
    }
}

template <DdlaBackend Backend, typename T, typename TA, typename TB>
void gemmt_run(const DdlaHandle_t& h,
    char uplo, char ta, char tb, int n, int k,
    const T& alpha, const TA* a, int lda, const TB* b, int ldb,
    const T& beta, T* c, int ldc,void* backend_work,std::size_t bytes)
{
    uplo=upper(uplo); ta=upper(ta); tb=upper(tb);
    if (n==0) return;
    if (alpha==T(0) || k==0) { scale_triangle<Backend,T>(h,uplo,n,beta,c,ldc); return; }
    batched_triangle<Backend,T,TA,TB>(h,uplo,ta,tb,n,k,alpha,a,lda,b,ldb,beta,c,ldc,backend_work,bytes);
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
#if DDLA_HAS_GPU
            real_diagonal_kernel<R><<<(n-1)/256+1,256,0,static_cast<gpuStream_t>(h->stream)>>>(n,reinterpret_cast<R*>(c),ldc);
            DDLA_CHECK(ddlaGetLastError(h));
#endif
        }
    }
}

template <DdlaBackend Backend, typename T, typename TA, typename TB>
void gemmt_check(const DdlaHandle_t& h, char uplo, char ta, char tb,
    int n, int k, const T& alpha, const TA* a, int lda, const TB* b, int ldb,
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
    const bool product=alpha!=T(0) && k!=0;
    if (!product && beta==T(1)) return;
    if (!c) throw std::invalid_argument("gemmt: null output");
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
template void blas_impl::real_diagonal<B,T>(const DdlaHandle_t&,int,T*,int); \
template void blas_impl::gemmt_copy_run<B,T>(const DdlaHandle_t&,char,int,int,int,const T&,const T*,T*,int,T* const*,const int*)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE)
#define INSTANTIATE_SIZE(T) template void blas_impl::gemmt_workspace_size<T>(const DdlaHandle_t&,char,char,int,int,std::size_t*)
DDLA_FOR_EACH_TYPE(INSTANTIATE_SIZE)
#undef INSTANTIATE_SIZE
#undef INSTANTIATE
template <DdlaBackend Backend,typename R>
void gemmt(const DdlaHandle_t& h,char uplo,char ta,char tb,int n,int k,
    const std::complex<R>& alpha,const R* a,int lda,const std::complex<R>* b,int ldb,
    const std::complex<R>& beta,std::complex<R>* c,int ldc)
{
    using T=std::complex<R>;
    blas_impl::gemmt_check<Backend,T>(h,uplo,ta,tb,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
    std::size_t bytes=0;
    if (alpha!=T(0) && k>0) blas_impl::gemmt_workspace_size<T,R,std::complex<R>>(h,ta,tb,n,k,&bytes);
    vector<char> work(h,bytes);
    blas_impl::gemmt_run<Backend,T>(h,uplo,ta,tb,n,k,alpha,a,lda,b,ldb,beta,c,ldc,work.data(),bytes);
}

template <DdlaBackend Backend,typename R>
void gemmt(const DdlaHandle_t& h,char uplo,char ta,char tb,int n,int k,
    const std::complex<R>& alpha,const std::complex<R>* a,int lda,const R* b,int ldb,
    const std::complex<R>& beta,std::complex<R>* c,int ldc)
{
    using T=std::complex<R>;
    blas_impl::gemmt_check<Backend,T>(h,uplo,ta,tb,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
    std::size_t bytes=0;
    if (alpha!=T(0) && k>0) blas_impl::gemmt_workspace_size<T,std::complex<R>,R>(h,ta,tb,n,k,&bytes);
    vector<char> work(h,bytes);
    blas_impl::gemmt_run<Backend,T>(h,uplo,ta,tb,n,k,alpha,a,lda,b,ldb,beta,c,ldc,work.data(),bytes);
}

template void gemmt<DdlaBackend::CPU,float>(const DdlaHandle_t&,char,char,char,int,int,const std::complex<float>&,const float*,int,const std::complex<float>*,int,const std::complex<float>&,std::complex<float>*,int);
template void blas_impl::gemmt_check<DdlaBackend::CPU,std::complex<float>,float,std::complex<float>>(const DdlaHandle_t&,char,char,char,int,int,const std::complex<float>&,const float*,int,const std::complex<float>*,int,const std::complex<float>&,std::complex<float>*,int);
template void blas_impl::gemmt_run<DdlaBackend::CPU,std::complex<float>,float,std::complex<float>>(const DdlaHandle_t&,char,char,char,int,int,const std::complex<float>&,const float*,int,const std::complex<float>*,int,const std::complex<float>&,std::complex<float>*,int,void*,std::size_t);
template void gemmt<DdlaBackend::CPU,float>(const DdlaHandle_t&,char,char,char,int,int,const std::complex<float>&,const std::complex<float>*,int,const float*,int,const std::complex<float>&,std::complex<float>*,int);
template void blas_impl::gemmt_check<DdlaBackend::CPU,std::complex<float>,std::complex<float>,float>(const DdlaHandle_t&,char,char,char,int,int,const std::complex<float>&,const std::complex<float>*,int,const float*,int,const std::complex<float>&,std::complex<float>*,int);
template void blas_impl::gemmt_run<DdlaBackend::CPU,std::complex<float>,std::complex<float>,float>(const DdlaHandle_t&,char,char,char,int,int,const std::complex<float>&,const std::complex<float>*,int,const float*,int,const std::complex<float>&,std::complex<float>*,int,void*,std::size_t);
template void gemmt<DdlaBackend::CPU,double>(const DdlaHandle_t&,char,char,char,int,int,const std::complex<double>&,const double*,int,const std::complex<double>*,int,const std::complex<double>&,std::complex<double>*,int);
template void blas_impl::gemmt_check<DdlaBackend::CPU,std::complex<double>,double,std::complex<double>>(const DdlaHandle_t&,char,char,char,int,int,const std::complex<double>&,const double*,int,const std::complex<double>*,int,const std::complex<double>&,std::complex<double>*,int);
template void blas_impl::gemmt_run<DdlaBackend::CPU,std::complex<double>,double,std::complex<double>>(const DdlaHandle_t&,char,char,char,int,int,const std::complex<double>&,const double*,int,const std::complex<double>*,int,const std::complex<double>&,std::complex<double>*,int,void*,std::size_t);
template void gemmt<DdlaBackend::CPU,double>(const DdlaHandle_t&,char,char,char,int,int,const std::complex<double>&,const std::complex<double>*,int,const double*,int,const std::complex<double>&,std::complex<double>*,int);
template void blas_impl::gemmt_check<DdlaBackend::CPU,std::complex<double>,std::complex<double>,double>(const DdlaHandle_t&,char,char,char,int,int,const std::complex<double>&,const std::complex<double>*,int,const double*,int,const std::complex<double>&,std::complex<double>*,int);
template void blas_impl::gemmt_run<DdlaBackend::CPU,std::complex<double>,std::complex<double>,double>(const DdlaHandle_t&,char,char,char,int,int,const std::complex<double>&,const std::complex<double>*,int,const double*,int,const std::complex<double>&,std::complex<double>*,int,void*,std::size_t);
#if DDLA_HAS_GPU
template void gemmt<DdlaBackend::GPU,float>(const DdlaHandle_t&,char,char,char,int,int,const std::complex<float>&,const float*,int,const std::complex<float>*,int,const std::complex<float>&,std::complex<float>*,int);
template void blas_impl::gemmt_check<DdlaBackend::GPU,std::complex<float>,float,std::complex<float>>(const DdlaHandle_t&,char,char,char,int,int,const std::complex<float>&,const float*,int,const std::complex<float>*,int,const std::complex<float>&,std::complex<float>*,int);
template void blas_impl::gemmt_run<DdlaBackend::GPU,std::complex<float>,float,std::complex<float>>(const DdlaHandle_t&,char,char,char,int,int,const std::complex<float>&,const float*,int,const std::complex<float>*,int,const std::complex<float>&,std::complex<float>*,int,void*,std::size_t);
template void gemmt<DdlaBackend::GPU,float>(const DdlaHandle_t&,char,char,char,int,int,const std::complex<float>&,const std::complex<float>*,int,const float*,int,const std::complex<float>&,std::complex<float>*,int);
template void blas_impl::gemmt_check<DdlaBackend::GPU,std::complex<float>,std::complex<float>,float>(const DdlaHandle_t&,char,char,char,int,int,const std::complex<float>&,const std::complex<float>*,int,const float*,int,const std::complex<float>&,std::complex<float>*,int);
template void blas_impl::gemmt_run<DdlaBackend::GPU,std::complex<float>,std::complex<float>,float>(const DdlaHandle_t&,char,char,char,int,int,const std::complex<float>&,const std::complex<float>*,int,const float*,int,const std::complex<float>&,std::complex<float>*,int,void*,std::size_t);
template void gemmt<DdlaBackend::GPU,double>(const DdlaHandle_t&,char,char,char,int,int,const std::complex<double>&,const double*,int,const std::complex<double>*,int,const std::complex<double>&,std::complex<double>*,int);
template void blas_impl::gemmt_check<DdlaBackend::GPU,std::complex<double>,double,std::complex<double>>(const DdlaHandle_t&,char,char,char,int,int,const std::complex<double>&,const double*,int,const std::complex<double>*,int,const std::complex<double>&,std::complex<double>*,int);
template void blas_impl::gemmt_run<DdlaBackend::GPU,std::complex<double>,double,std::complex<double>>(const DdlaHandle_t&,char,char,char,int,int,const std::complex<double>&,const double*,int,const std::complex<double>*,int,const std::complex<double>&,std::complex<double>*,int,void*,std::size_t);
template void gemmt<DdlaBackend::GPU,double>(const DdlaHandle_t&,char,char,char,int,int,const std::complex<double>&,const std::complex<double>*,int,const double*,int,const std::complex<double>&,std::complex<double>*,int);
template void blas_impl::gemmt_check<DdlaBackend::GPU,std::complex<double>,std::complex<double>,double>(const DdlaHandle_t&,char,char,char,int,int,const std::complex<double>&,const std::complex<double>*,int,const double*,int,const std::complex<double>&,std::complex<double>*,int);
template void blas_impl::gemmt_run<DdlaBackend::GPU,std::complex<double>,std::complex<double>,double>(const DdlaHandle_t&,char,char,char,int,int,const std::complex<double>&,const std::complex<double>*,int,const double*,int,const std::complex<double>&,std::complex<double>*,int,void*,std::size_t);
#endif
template void blas_impl::gemmt_workspace_size<std::complex<float>,float,std::complex<float>>(const DdlaHandle_t&,char,char,int,int,std::size_t*);
template void blas_impl::gemmt_workspace_size<std::complex<float>,std::complex<float>,float>(const DdlaHandle_t&,char,char,int,int,std::size_t*);
template void blas_impl::gemmt_workspace_size<std::complex<double>,double,std::complex<double>>(const DdlaHandle_t&,char,char,int,int,std::size_t*);
template void blas_impl::gemmt_workspace_size<std::complex<double>,std::complex<double>,double>(const DdlaHandle_t&,char,char,int,int,std::size_t*);
} // namespace ddla
