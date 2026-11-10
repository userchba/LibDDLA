#include "hb2st_householder_gpu.h"
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "helpers.h"
#include "hb2st_householder_kernels.h"
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace ddla { namespace detail {
namespace {
// The row-major strip gives adjacent threads adjacent eigenvectors. Each
// thread retains one short reflector window while its start moves upward.
// Only one outgoing and one incoming row are transferred between reflectors.
// Tail entries beyond n are zero in both the packed vectors and the window.
template<class D, int B>
__global__ void apply_reflectors(D* x, int ld, int columns, int n, int first,
                               int count, const D* packed)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= columns) return;
    D window[B];
    #pragma unroll
    for (int r = 0; r < B; ++r)
        window[r] = first + r < n ? x[std::size_t(first + r) * ld + col] : D(0);
    #pragma unroll 1
    for (int q = 0; q < count; ++q) {
        const D* v = packed + std::size_t(q) * B;
        const D tau = v[0];
        if (tau != D(0)) {
            D dot = window[0];
            #pragma unroll
            for (int r = 1; r < B; ++r) dot += device_conj(v[r]) * window[r];
            dot *= tau;
            window[0] -= dot;
            #pragma unroll
            for (int r = 1; r < B; ++r) window[r] -= v[r] * dot;
        }
        if (q + 1 < count) {
            const int start = first - q;
            if (start + B - 1 < n)
                x[std::size_t(start + B - 1) * ld + col] = window[B - 1];
            #pragma unroll
            for (int r = B - 1; r > 0; --r) window[r] = window[r - 1];
            window[0] = x[std::size_t(start - 1) * ld + col];
        }
    }
    const int start = first - count + 1;
    #pragma unroll
    for (int r = 0; r < B; ++r)
        if (start + r < n) x[std::size_t(start + r) * ld + col] = window[r];
}

// Wider windows cooperate within a 32-lane warp instead of spilling a
// complete complex window per thread. Four neighboring columns share one
// block; every warp participates in the block barriers, including a partial
// final column group. Logical 32-lane shuffles work on CUDA and HIP.
template<class D,class R,int B>
__global__ void apply_reflectors_warp(D* x,int ld,int columns,int n,int first,
                                    int count,const D* packed)
{
    const int lane=threadIdx.x,group=threadIdx.y;
    const int col=blockIdx.x*4+group;
    const bool active=col<columns;
    __shared__ D window[4][B+1];
    __shared__ D dots[4];
    window[group][lane]=active && first+lane<n ? x[std::size_t(first+lane)*ld+col] : D(0);
    if constexpr(B==64)
        window[group][lane+32]=active && first+lane+32<n ? x[std::size_t(first+lane+32)*ld+col] : D(0);
    __syncthreads();
    for(int q=0;q<count;++q) {
        const D* v=packed+std::size_t(q)*B;
        const D v0=lane==0?D(1):v[lane];
        D a=window[group][lane];
        D dot=device_conj(v0)*a;
        D b=0,v1=0;
        if constexpr(B==64) {
            b=window[group][lane+32];v1=v[lane+32];
            dot+=device_conj(v1)*b;
        }
        #pragma unroll
        for(int offset=16;offset>0;offset/=2) {
            if constexpr(std::is_same<D,R>::value)dot+=gpuShflDown32(dot,offset);
            else dot+=D(gpuShflDown32(dot.real(),offset),gpuShflDown32(dot.imag(),offset));
        }
        if(lane==0) {
            R tau;
            if constexpr(std::is_same<D,R>::value)tau=v[0];
            else tau=v[0].real();
            dots[group]=tau*dot;
        }
        __syncthreads();
        a-=v0*dots[group];
        window[group][lane+1]=a;
        if constexpr(B==64) {
            b-=v1*dots[group];
            window[group][lane+33]=b;
        }
        const int start=first-q;
        if(active) {
            if(q+1==count) {
                if(start+lane<n)x[std::size_t(start+lane)*ld+col]=a;
                if constexpr(B==64)
                    if(start+lane+32<n)x[std::size_t(start+lane+32)*ld+col]=b;
            } else if(lane==31 && start+B-1<n) {
                if constexpr(B==64)x[std::size_t(start+B-1)*ld+col]=b;
                else x[std::size_t(start+B-1)*ld+col]=a;
            }
        }
        if(q+1<count && lane==0)
            window[group][0]=active ? x[std::size_t(start-1)*ld+col] : D(0);
        __syncthreads();
    }
}
}

template<class T>
void hh_gpu_panel_run(const DdlaHandle_t& h, T* rows, int ld, int columns,
                      int n, int band, int first, int count, const T* packed)
{
    if (!columns || !count) return;
    using D = DeviceScalarT<T>;
    using R = typename real_type<T>::type;
    const int threads = std::is_same<T, std::complex<double>>::value && band >= 32 ? 64 : 128;
    const dim3 blocks((columns + threads - 1) / threads);
    const auto stream = static_cast<gpuStream_t>(h->stream);
#define HH_PANEL(B) case B: apply_reflectors<D, B><<<blocks, threads, 0, stream>>>( \
    reinterpret_cast<D*>(rows), ld, columns, n, first, count, reinterpret_cast<const D*>(packed)); break
    switch (band) {
        HH_PANEL(2); HH_PANEL(4); HH_PANEL(8); HH_PANEL(16);
        case 32: apply_reflectors_warp<D,R,32><<<dim3((columns+3)/4),dim3(32,4),0,stream>>>(
            reinterpret_cast<D*>(rows),ld,columns,n,first,count,reinterpret_cast<const D*>(packed));break;
        case 64: apply_reflectors_warp<D,R,64><<<dim3((columns+3)/4),dim3(32,4),0,stream>>>(
            reinterpret_cast<D*>(rows),ld,columns,n,first,count,reinterpret_cast<const D*>(packed));break;
        default: throw std::logic_error("GPU Householder: unsupported compact panel width");
    }
#undef HH_PANEL
    DDLA_CHECK(ddlaGetLastError(h));
}
#define HH_INSTANTIATE(T) template void hh_gpu_panel_run<T>(const DdlaHandle_t&,T*,int,int,int,int,int,int,const T*)
HH_INSTANTIATE(float); HH_INSTANTIATE(double);
HH_INSTANTIATE(std::complex<float>); HH_INSTANTIATE(std::complex<double>);
#undef HH_INSTANTIATE
} }

// One fused block per independent chase event on the local band and halo.
// No history larger than a fixed number of wavefront levels resides on device.
namespace ddla { namespace detail {
namespace {
template<class D,class R>
__device__ __forceinline__ R chase_real(D x) {
    if constexpr(std::is_same<D,R>::value)return x;
    else return x.real();
}
template<class D,class R>
__device__ __forceinline__ R chase_scale(D x) {
    if constexpr(std::is_same<D,R>::value)return fabs(x);
    else return fmax(fabs(x.real()),fabs(x.imag()));
}
template<class D,class R>
__device__ __forceinline__ D chase_divide(D x,R y) {
    if constexpr(std::is_same<D,R>::value)return x/y;
    else return D(x.real()/y,x.imag()/y);
}
template<class D,class R>
__device__ __forceinline__ D chase_phase(D x) {
    if constexpr(std::is_same<D,R>::value)return x<R(0)?D(-1):D(1);
    else {
        const R scale=chase_scale<D,R>(x);
        if(scale==R(0))return D(1);
        const R re=x.real()/scale,im=x.imag()/scale,rho=sqrt(re*re+im*im);
        return D(re/rho,im/rho);
    }
}
template<class D,class R>
__device__ __forceinline__ bool chase_finite(D x) {
    if constexpr(std::is_same<D,R>::value)return isfinite(x);
    else return isfinite(x.real())&&isfinite(x.imag());
}
template<class D>
__device__ __forceinline__ D chase_get(const D* a,int ld,int origin,int row,int col) {
    return row>=col?a[std::size_t(col-origin)*ld+row-col]:device_conj(a[std::size_t(row-origin)*ld+col-row]);
}
template<class D,class R>
__global__ void reduce_wavefront(D* a,int n,int b,int ld,int lo,int level,int maxevents,
                                int packet_level,int origin,D* packet,int* failure) {
    const int stage=lo+blockIdx.x,sweep=level-3*stage;
    const int start=stage+1+sweep*b;
    const int ell=min(b,n-start),end=start+ell;
    const int target=sweep?start-b:stage;
    const int stop=min(n,end+b),tid=threadIdx.x;
    __shared__ D v[64],w[64];
    __shared__ R tau;
    if(tid==0) {
        R scale=0;
        for(int r=0;r<ell;++r){v[r]=chase_get(a,ld,origin,start+r,target);scale=fmax(scale,chase_scale<D,R>(v[r]));}
        if(scale==R(0)) {
            tau=0;v[0]=D(1);for(int r=1;r<ell;++r)v[r]=D(0);
        } else {
            R norm=0;
            for(int r=0;r<ell;++r){v[r]=chase_divide<D,R>(v[r],scale);norm+=chase_real<D,R>(device_conj(v[r])*v[r]);}
            const D alpha=-chase_phase<D,R>(v[0])*sqrt(norm),u0=v[0]-alpha;
            v[0]=D(1);norm=1;
            for(int r=1;r<ell;++r){v[r]/=u0;norm+=chase_real<D,R>(device_conj(v[r])*v[r]);}
            tau=R(2)/norm;
        }
    }
    __syncthreads();
    if(packet && tid<ell) {
        const std::size_t at=(std::size_t(packet_level)*maxevents+blockIdx.x)*b;
        packet[at+tid]=tid==0?D(tau):v[tid];
    }
    if(tid<ell && (!chase_finite<D,R>(v[tid])||!isfinite(tau)))atomicExch(failure,1);
    if(tau==R(0))return;
    if(tid<ell) {
        D sum=0;
        for(int c=0;c<ell;++c)sum+=chase_get(a,ld,origin,start+tid,start+c)*v[c];
        w[tid]=tau*sum;
    }
    __syncthreads();
    if(tid==0) {
        D dot=0;for(int r=0;r<ell;++r)dot+=device_conj(v[r])*w[r];
        const R correction=-R(0.5)*tau*chase_real<D,R>(dot);
        for(int r=0;r<ell;++r)w[r]+=correction*v[r];
    }
    __syncthreads();
    for(int q=tid;q<ell*ell;q+=blockDim.x) {
        const int r=q%ell,c=q/ell;
        if(r>=c) {
            D z=chase_get(a,ld,origin,start+r,start+c)-v[r]*device_conj(w[c])-w[r]*device_conj(v[c]);
            if(r==c)z=D(chase_real<D,R>(z));
            a[std::size_t(start+c-origin)*ld+r-c]=z;
            if(!chase_finite<D,R>(z))atomicExch(failure,1);
        }
    }
    for(int c=target+tid;c<start;c+=blockDim.x) {
        D dot=0;
        for(int r=0;r<ell;++r)dot+=device_conj(v[r])*chase_get(a,ld,origin,start+r,c);
        for(int r=0;r<ell;++r) {
            D& z=a[std::size_t(c-origin)*ld+start+r-c];z-=tau*v[r]*dot;
            if(!chase_finite<D,R>(z))atomicExch(failure,1);
        }
    }
    for(int r=end+tid;r<stop;r+=blockDim.x) {
        D dot=0;
        for(int c=0;c<ell;++c)dot+=chase_get(a,ld,origin,r,start+c)*v[c];
        for(int c=0;c<ell;++c) {
            D& z=a[std::size_t(start+c-origin)*ld+r-start-c];z-=tau*dot*device_conj(v[c]);
            if(!chase_finite<D,R>(z))atomicExch(failure,1);
        }
    }
    __syncthreads();
    if(tid>0 && tid<ell)a[std::size_t(target-origin)*ld+start+tid-target]=D(0);
}
}
template<class T>
void hh_gpu_reduce_levels(const DdlaHandle_t& h,int n,int b,int ld,int first,int levels,int maxevents,
                          int origin,int limit,T* band,T* packet,int* failure) {
    using D=DeviceScalarT<T>;using R=typename real_type<T>::type;
    for(int k=0;k<levels;++k) {
        int lo=0,hi=-1;
        if(!hh::local_level_bounds(n,b,first+k,origin,limit,2*b*k,lo,hi))continue;
        reduce_wavefront<D,R><<<hi-lo+1,128,0,static_cast<gpuStream_t>(h->stream)>>>(
            reinterpret_cast<D*>(band),n,b,ld,lo,first+k,maxevents,k,origin,reinterpret_cast<D*>(packet),failure);
    }
    DDLA_CHECK(ddlaGetLastError(h));
}
#define REDUCE(T) template void hh_gpu_reduce_levels<T>(const DdlaHandle_t&,int,int,int,int,int,int,int,int,T*,T*,int*)
REDUCE(float);REDUCE(double);REDUCE(std::complex<float>);REDUCE(std::complex<double>);
#undef REDUCE
} }
