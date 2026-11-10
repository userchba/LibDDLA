#include "phb2st_impl.h"
#include "hb2st_householder_kernels.h"
#include "ddla_stream.h"
#include "comm_traits.h"
#include "helpers.h"
#include <exception>
#include <type_traits>
#include <complex>
namespace ddla { namespace spectral_impl {
using detail::device_conj;
namespace {
template<class D,class R> __device__ __forceinline__ R gv_real(D x) {
    if constexpr(std::is_same<D,R>::value)return x;else return x.real();
}
template<class D,class R> __device__ __forceinline__ R gv_abs(D x) {
    if constexpr(std::is_same<D,R>::value)return fabs(x);else return hypot(x.real(),x.imag());
}
template<class D,class R> __device__ __forceinline__ bool gv_finite(D x) {
    if constexpr(std::is_same<D,R>::value)return isfinite(x);else return isfinite(x.real())&&isfinite(x.imag());
}
template<class D,class R> __device__ __forceinline__ D gv_phase(D x) {
    const R scale=fmax(fabs(x.real()),fabs(x.imag()));
    const R re=x.real()/scale,im=x.imag()/scale,rho=hypot(re,im);
    return D(re/rho,im/rho);
}
template<class D,class R> __device__ __forceinline__ void gv_generate(D f,D g,R& c,D& s,D& r) {
    if(g==D(0)){c=1;s=0;r=f;return;}
    if(f==D(0)) {
        c=0;
        if constexpr(std::is_same<D,R>::value){s=1;r=g;}
        else {s=device_conj(gv_phase<D,R>(g));r=D(gv_abs<D,R>(g));}
        return;
    }
    const R scale=fmax(gv_abs<D,R>(f),gv_abs<D,R>(g));
    const D ff=f/scale,gg=g/scale;
    const R af=gv_abs<D,R>(ff),ag=gv_abs<D,R>(gg),norm=hypot(af,ag);
    if constexpr(std::is_same<D,R>::value) {
        const R rr=copysign(norm,ff);c=ff/rr;s=gg/rr;r=rr*scale;
    } else {
        const D phase=gv_phase<D,R>(f);c=af/norm;s=phase*device_conj(gg)/norm;r=phase*(norm*scale);
    }
}
template<class D,class R> __device__ __forceinline__ void gv_pair(R c,D s,D& x,D& y,int* failure) {
    const D old=x;x=c*x+s*y;y=c*y-device_conj(s)*old;
    if(!gv_finite<D,R>(x)||!gv_finite<D,R>(y))atomicExch(failure,1);
}
__device__ __forceinline__ int gv_ceil(int x,int y) {return x>0?(x+y-1)/y:0;}
__device__ __forceinline__ bool gv_accept(int idx,int n,int b,int left,int right) {
    return idx>=1 && idx<n && idx-1>=left && min(n,idx+b+1)<=right;
}
template<class D,class R>
__global__ void givens_chase_steps(const D* input,D* output,D* tiles,int tile_columns,int blocks,
    int n,int b,int rank_first,int rank_last,int owned_first,int owned_last,
    int initial_i,int initial_k,int initial_j1,int initial_j2,int initial_nr,int levels,int maxevents,
    D* packet,int* failure) {
    const int ld=b+3,kdn=min(n-1,b),b1=b+1,tid=threadIdx.x;
    const int core_first=owned_first+int(static_cast<long long>(owned_last-owned_first)*blockIdx.x/blocks);
    const int core_last=owned_first+int(static_cast<long long>(owned_last-owned_first)*(blockIdx.x+1)/blocks);
    if(core_first==core_last)return;
    // Input/output separation prevents a fast block from publishing its new
    // core while another block is still loading its old temporal halo.
    const int origin=max(rank_first,core_first-levels*b1),limit=min(rank_last,core_last+levels*b1);
    D* state=tiles+std::size_t(blockIdx.x)*tile_columns*ld;
    for(int at=tid;at<(limit-origin)*ld;at+=blockDim.x)
        state[at]=input[std::size_t(origin-rank_first)*ld+at];
    __syncthreads();
    auto B=[&](int row,int col)->D&{return state[std::size_t(col-origin)*ld+row];};
    auto C=[&](int col)->D&{return B(b+1,col);};
    auto S=[&](int col)->D&{return B(b+2,col);};
    __shared__ R ann_cos;
    __shared__ D ann_sine;
    int i=initial_i,k=initial_k,j1=initial_j1,j2=initial_j2,nr=initial_nr;
    for(int step=0;step<levels;++step) {
        const int remain=levels-step;
        const int left=max(0,core_first-remain*b1),right=min(n,core_last+remain*b1);
        const int j1t=j1+kdn,j2t=j2+kdn;
        const bool ann=k>2,exit=j2t+kdn>n;
        const int mid=j1t-(ann?kdn+1:0),last=j2t-(exit?kdn+1:0),nrnew=nr+(ann?1:0);
        const int firstold=j1t-1,firstmid=mid-1;
        const int oldlo=max(0,gv_ceil(left+1-firstold,b1));
        const int oldhi=max(oldlo,min(nr,right==n?gv_ceil(n-firstold,b1):gv_ceil(right-b-firstold,b1)));
        const int oldcount=oldhi-oldlo;
        const int nrec=min(gv_ceil(j2t-firstmid,b1),gv_ceil(n-firstmid,b1));
        const int midlo=max(0,gv_ceil(left+1-firstmid,b1));
        const int midhi=max(midlo,min(nrnew,right==n?gv_ceil(n-firstmid,b1):gv_ceil(right-b-firstmid,b1)));
        // Generate existing bulges. All updates wait for this phase to finish.
        for(int t=oldlo+tid;t<oldhi;t+=blockDim.x) {
            const int idx=firstold+t*b1;R c;D sine,r;
            gv_generate<D,R>(B(0,idx-1),S(idx),c,sine,r);
            B(0,idx-1)=r;C(idx)=D(c);S(idx)=sine;
        }
        __syncthreads();
        for(int task=tid;task<oldcount*(b-1);task+=blockDim.x) {
            const int idx=firstold+(oldlo+task/(b-1))*b1,l=task%(b-1)+1;
            gv_pair<D,R>(gv_real<D,R>(C(idx)),S(idx),B(l,idx-1),B(l-1,idx),failure);
        }
        __syncthreads();
        const int aidx=i+k-2;
        const bool annihilate=ann&&k<=n-i+1&&gv_accept(aidx,n,b,left,right);
        if(tid==0&&annihilate) {
            R c;D sine,r;
            gv_generate<D,R>(B(b-k+2,aidx-1),B(b-k+1,aidx),c,sine,r);
            C(aidx)=D(c);S(aidx)=sine;ann_cos=c;ann_sine=sine;B(b-k+2,aidx-1)=r;
        }
        __syncthreads();
        if(annihilate)for(int l=tid;l<k-3;l+=blockDim.x)
            gv_pair<D,R>(ann_cos,ann_sine,B(b-k+3+l,aidx-1),B(b-k+2+l,aidx),failure);
        // Hermitian diagonal pair update. The annihilate right footprint is
        // strictly above these three entries and does not read its sine again.
        for(int t=midlo+tid;t<midhi;t+=blockDim.x) {
            const int idx=firstmid+t*b1;const R c=gv_real<D,R>(C(idx));const D sv=S(idx);
            const R x=gv_real<D,R>(B(b,idx-1)),y=gv_real<D,R>(B(b,idx));const D z=B(b-1,idx);
            if constexpr(std::is_same<D,R>::value) {
                const R t1=sv*z,t2=c*z,t3=t2-sv*x,t4=t2+sv*y,t5=c*x+t1,t6=c*y-t1;
                B(b,idx-1)=c*t5+sv*t4;B(b,idx)=c*t6-sv*t3;B(b-1,idx)=c*t4-sv*t5;
            } else {
                const R t1r=sv.real()*z.real()-sv.imag()*z.imag(),t1i=sv.real()*z.imag()+sv.imag()*z.real();
                const D t2=c*z,t3=t2-device_conj(sv)*x,t4=device_conj(t2)+sv*y;
                const R t5=c*x+t1r,t6=c*y-t1r;
                B(b,idx-1)=D(c*t5+sv.real()*t4.real()+sv.imag()*t4.imag());
                B(b,idx)=D(c*t6-sv.real()*t3.real()+sv.imag()*t3.imag());
                B(b-1,idx)=c*t3+device_conj(sv)*D(t6,t1i);
                S(idx)=device_conj(sv);
            }
            if(!gv_finite<D,R>(B(b,idx-1))||!gv_finite<D,R>(B(b,idx))||!gv_finite<D,R>(B(b-1,idx)))atomicExch(failure,1);
        }
        __syncthreads();
        const int midcount=midhi-midlo;
        for(int task=tid;task<midcount*(b-1);task+=blockDim.x) {
            const int t=midlo+task/(b-1),idx=firstmid+t*b1,lp=task%(b-1)+1;
            const int lmax=t==nrnew-1?min(b-1,n-j2t):b-1;
            if(lp<=lmax && idx+lp<n)
                gv_pair<D,R>(gv_real<D,R>(C(idx)),S(idx),B(b-1-lp,idx+lp),B(b-lp,idx+lp),failure);
        }
        __syncthreads();
        for(int t=midlo+tid;t<midhi;t+=blockDim.x) {
            const int idx=firstmid+t*b1;
            if(packet&&t<nrec&&idx>=core_first&&idx<core_last) {
                const std::size_t out=(std::size_t(step)*maxevents+(idx-rank_first)/b1)*2;
                packet[out]=C(idx);packet[out+1]=device_conj(S(idx));
            }
            if(idx<last&&idx+b<n) {S(idx+b)=S(idx)*B(0,idx+b);B(0,idx+b)=C(idx)*B(0,idx+b);}
        }
        __syncthreads();
        nr=nrnew-(exit?1:0);j1=mid;j2=last;
        if(--k<2){k=kdn+1;++i;}
    }
    for(int at=tid;at<(core_last-core_first)*ld;at+=blockDim.x)
        output[std::size_t(core_first-rank_first)*ld+at]=state[std::size_t(core_first-origin)*ld+at];
}
}

template<class T>
void Phb2stResources<T>::device_setup(const DdlaHandle_t& h,const Phb2stPlan& p) {
    std::exception_ptr error;
    try {
        const int n=p.input.n,b=p.input.band,P=h->nprocs;
        if(P>1&&n>2&&b>1&&(!team_||!team_->available()))
            throw std::invalid_argument("GPU Givens requires a shared band and record ring on one node");
        device_levels_=16;
        if(p.output==Phb2stOutput::History)
            device_levels_=std::max(1,std::min(16,int(p.local.record_capacity/(n/(b+1)+2))));
        device_core_first_=int(std::int64_t(n)*h->myid/P);
        device_core_last_=int(std::int64_t(n)*(h->myid+1)/P);
        const int halo=device_levels_*(b+1);
        device_first_=std::max(0,device_core_first_-halo);device_last_=std::min(n,device_core_last_+halo);
        device_max_events_=(device_last_-device_first_)/(b+1)+2;
        const auto state_count=std::max<std::size_t>(1,std::size_t(device_last_-device_first_)*(b+3));
        device_state_=std::make_unique<ddla::vector<T>>(h,state_count);
        device_output_=std::make_unique<ddla::vector<T>>(h,state_count);
        const int core=device_core_last_-device_core_first_;
        device_blocks_=std::max(1,std::min(8,(core+255)/256));
        device_tile_columns_=std::max(1,std::min(n,(core+device_blocks_-1)/device_blocks_+2*halo));
        device_tiles_=std::make_unique<ddla::vector<T>>(h,std::size_t(device_blocks_)*device_tile_columns_*(b+3));
        host_device_state_.resize(state_count);
        device_failure_=std::make_unique<ddla::vector<int>>(h,1);
        if(p.output==Phb2stOutput::History) {
            const std::size_t packet=std::size_t(device_levels_)*device_max_events_*2;
            device_packet_=std::make_unique<ddla::vector<T>>(h,packet);host_device_packet_.resize(packet);
            DDLA_CHECK(ddlaMemset(h,device_packet_->data(),0,packet*sizeof(T)));
            DDLA_CHECK(ddlaSynchronize(h));
        }
    }catch(...){error=std::current_exception();}
    int ok=error?0:1;MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,&ok,1,MPI_INT,MPI_MIN,h->comm));
    if(!ok){if(error)std::rethrow_exception(error);throw std::runtime_error("GPU Givens resource allocation failed on another rank");}
}

template<class T>
void Phb2stResources<T>::device_run(const DdlaHandle_t& h,const Phb2stPlan& p,T* band,
    hb2st_rot<T>* ring,long long capacity) {
    using D=detail::DeviceScalarT<T>;using R=typename real_type<T>::type;
    const int n=p.input.n,b=p.input.band,ld=b+3;
    const bool keep=p.output==Phb2stOutput::History;
    const int batch=keep?std::min(device_levels_,std::max(1,int(capacity/(n/(b+1)+2)))):device_levels_;
    for(int col=device_first_;col<device_last_;++col) {
        T* dst=host_device_state_.data()+std::size_t(col-device_first_)*ld;
        std::copy_n(band+std::size_t(col)*(b+1),b+1,dst);
        dst[b]=T(detail::hh::real_part(dst[b]));dst[b+1]=dst[b+2]=T(0);
    }
    DDLA_CHECK(ddlaMemcpy(h,device_state_->data(),host_device_state_.data(),host_device_state_.size()*sizeof(T)));
    DDLA_CHECK(ddlaMemset(h,device_failure_->data(),0,sizeof(int)));
    detail::ChaseCursor cursor(n,b);int nr=0;
    while(!cursor.done()) {
        const auto before=cursor;const int nr_before=nr;int steps=0;
        for(;steps<batch&&!cursor.done();++steps) {
            const int kdn=std::min(n-1,b);const bool exit=cursor.j2+2*kdn>n;
            nr+=cursor.k>2?1:0;if(exit)--nr;cursor.next();
        }
        std::exception_ptr error;
        try {
            givens_chase_steps<D,R><<<device_blocks_,256,0,static_cast<gpuStream_t>(h->stream)>>>(
                reinterpret_cast<const D*>(device_state_->data()),reinterpret_cast<D*>(device_output_->data()),
                reinterpret_cast<D*>(device_tiles_->data()),device_tile_columns_,device_blocks_,n,b,device_first_,device_last_,device_core_first_,device_core_last_,
                before.i,before.k,before.j1,before.j2,nr_before,steps,device_max_events_,
                keep?reinterpret_cast<D*>(device_packet_->data()):nullptr,device_failure_->data());
            DDLA_CHECK(ddlaGetLastError(h));
            if(keep)DDLA_CHECK(ddlaMemcpy(h,host_device_packet_.data(),device_packet_->data(),
                std::size_t(steps)*device_max_events_*2*sizeof(T)));
            int failure=0;DDLA_CHECK(ddlaMemcpy(h,&failure,device_failure_->data(),sizeof(int)));
            if(cursor.done())DDLA_CHECK(ddlaMemcpy2D(h,band+std::size_t(device_core_first_)*(b+1),(b+1)*sizeof(T),
                device_output_->data()+std::size_t(device_core_first_-device_first_)*ld,ld*sizeof(T),
                (b+1)*sizeof(T),device_core_last_-device_core_first_));
            DDLA_CHECK(ddlaSynchronize(h));
            if(failure)throw std::runtime_error("GPU Givens produced non-finite band entries");
            if(keep) {
                auto recorded=before;
                for(int step=0;step<steps;++step) {
                    const int first=recorded.first(),count=recorded.count();
                    const int lo=std::max(0,(device_core_first_>first?(device_core_first_-first+b)/(b+1):0));
                    for(int t=lo;t<count;++t) {
                        const int idx=first+t*(b+1);if(idx>=device_core_last_)break;
                        const std::size_t at=(std::size_t(step)*device_max_events_+(idx-device_first_)/(b+1))*2;
                        auto& out=ring[(recorded.base+t)%capacity];out={};out.j0=idx;
                        out.c=detail::hh::real_part(host_device_packet_[at]);out.s=host_device_packet_[at+1];
                    }
                    recorded.next();
                }
            }
        }catch(...){error=std::current_exception();}
        int ok=error?0:1;MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,&ok,1,MPI_INT,MPI_MIN,h->comm));
        if(!ok){if(error)std::rethrow_exception(error);throw std::runtime_error("GPU Givens reduction failed on another rank");}
        if(keep) {
            MPI_CHECK(MPI_Barrier(h->comm));
            history().capture(before.base,cursor.base,ring,capacity);
            MPI_CHECK(MPI_Barrier(h->comm));
        }
        device_state_.swap(device_output_);
        if(!cursor.done()) {
            commGroupStart<DdlaBackend::GPU>(h);
            for(int peer=0;peer<h->nprocs;++peer) {
                if(peer==h->myid)continue;
                const int a=int(std::int64_t(n)*peer/h->nprocs),z=int(std::int64_t(n)*(peer+1)/h->nprocs);
                const int halo=device_levels_*(b+1);
                const int s0=std::max(device_core_first_,std::max(0,a-halo)),s1=std::min(device_core_last_,std::min(n,z+halo));
                const int r0=std::max(device_first_,a),r1=std::min(device_last_,z);
                const std::size_t sc=std::size_t(std::max(0,s1-s0))*ld,rc=std::size_t(std::max(0,r1-r0))*ld;
                if(!sc&&!rc)continue;
                const T* send=device_state_->data()+(sc?std::size_t(s0-device_first_)*ld:0);
                T* recv=device_state_->data()+(rc?std::size_t(r0-device_first_)*ld:0);
                commSendrecv<DdlaBackend::GPU>(h,CommScope::Grid,send,sc,peer,recv,rc,peer);
            }
            commGroupEnd<DdlaBackend::GPU>(h);
        }
    }
    // With band<=1 there are no rotations and the input is already tridiagonal.
    MPI_CHECK(MPI_Barrier(h->comm));
}
#define GPU_GIVENS(T) template void Phb2stResources<T>::device_setup(const DdlaHandle_t&,const Phb2stPlan&); \
 template void Phb2stResources<T>::device_run(const DdlaHandle_t&,const Phb2stPlan&,T*,hb2st_rot<T>*,long long)
GPU_GIVENS(float);GPU_GIVENS(double);GPU_GIVENS(std::complex<float>);GPU_GIVENS(std::complex<double>);
#undef GPU_GIVENS

} }
