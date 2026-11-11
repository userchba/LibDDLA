#include "pdgmm_impl.h"
#include "BLAS/blas_impl.h"
#include "check_agree.h"
#include "ddla_desc.h"
#include "ddla_stream.h"
#include "instantiate.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>

namespace ddla { namespace pdgmm_impl {

template <DdlaBackend Backend, typename T, typename U>
void pdgmm_check(const DdlaHandle_t& h, char side, int m, int n,
    const T* A, const int* descA, const U* x, int incx, T* C, const int* descC)
{
    detail::agree_check(h,"pdgmm",[&](detail::Agree&) {
        check_desc(descA,h);
        check_desc(descC,h);
        if (m<0 || n<0 || m>descA[DDLA_M_] || n>descA[DDLA_N_] ||
            m>descC[DDLA_M_] || n>descC[DDLA_N_])
            throw std::invalid_argument("pdgmm: region exceeds descriptor dimensions");
        for (int field : {DDLA_MB_,DDLA_NB_,DDLA_RSRC_,DDLA_CSRC_}) {
            if (descA[field]!=descC[field])
                throw std::invalid_argument("pdgmm: input and output blocks and owners must match");
        }
        const int rows=num_loc(m,descA[DDLA_MB_],h->myprow_,descA[DDLA_RSRC_],h->nprows_);
        const int columns=num_loc(n,descA[DDLA_NB_],h->mypcol_,descA[DDLA_CSRC_],h->npcols_);
        blas_impl::dgmm_check<Backend,T,U>(h,side,rows,columns,A,descA[DDLA_LLD_],
            x,incx,C,descC[DDLA_LLD_],blas_impl::upper(side)=='L' ? m : n);
    });
    int signature[18]={blas_impl::upper(side),m,n,incx};
    detail::descriptor_signature(signature+4,descA);
    detail::descriptor_signature(signature+11,descC);
    detail::agree_signature(h,"pdgmm",signature);
}

template <DdlaBackend Backend, typename T, typename U>
void pdgmm_run(const DdlaHandle_t& h, char side, int m, int n,
    const T* A, const int* descA, const U* x, int incx, T* C, const int* descC)
{
    const int rows=num_loc(m,descA[DDLA_MB_],h->myprow_,descA[DDLA_RSRC_],h->nprows_);
    const int columns=num_loc(n,descA[DDLA_NB_],h->mypcol_,descA[DDLA_CSRC_],h->npcols_);
    if (rows==0 || columns==0) return;
    side=blas_impl::upper(side);
    const bool left=side=='L';
    const int length=left ? m : n;
    const int local_length=left ? rows : columns;
    const int block=descA[left ? DDLA_MB_ : DDLA_NB_];
    const int source=descA[left ? DDLA_RSRC_ : DDLA_CSRC_];
    const int coordinate=left ? h->myprow_ : h->mypcol_;
    const int processes=left ? h->nprows_ : h->npcols_;
    const std::ptrdiff_t step=incx<0 ? -std::ptrdiff_t(incx) : incx;
    for (int local=0;local<local_length;) {
        const int width=std::min(block,local_length-local);
        const int global=indxl2g(local,block,coordinate,source,processes);
        // DGMM starts a negative-stride slice at its highest address. Pass
        // the slice's lowest address so its first value is global[global].
        const std::ptrdiff_t diagonal_offset=incx<0 ?
            std::ptrdiff_t(length-global-width)*step : std::ptrdiff_t(global)*step;
        const std::size_t a_offset=left ? local : std::size_t(local)*descA[DDLA_LLD_];
        const std::size_t c_offset=left ? local : std::size_t(local)*descC[DDLA_LLD_];
        blas_impl::dgmm_run<Backend,T,U>(h,side,left ? width : rows,left ? columns : width,
            A+a_offset,descA[DDLA_LLD_],x+diagonal_offset,incx,C+c_offset,descC[DDLA_LLD_]);
        local+=width;
    }
}

}} // namespace ddla::pdgmm_impl

namespace ddla {
template <DdlaBackend Backend, typename T, typename U>
void pdgmm(const DdlaHandle_t& h, char side, int m, int n,
    const T* A, const int* descA, const U* x, int incx, T* C, const int* descC)
{
    pdgmm_impl::pdgmm_check<Backend,T,U>(h,side,m,n,A,descA,x,incx,C,descC);
    pdgmm_impl::pdgmm_run<Backend,T,U>(h,side,m,n,A,descA,x,incx,C,descC);
}

#define INSTANTIATE(B,T,U) \
template void pdgmm<B,T,U>(const DdlaHandle_t&,char,int,int,const T*,const int*,const U*,int,T*,const int*); \
template void pdgmm_impl::pdgmm_check<B,T,U>(const DdlaHandle_t&,char,int,int,const T*,const int*,const U*,int,T*,const int*); \
template void pdgmm_impl::pdgmm_run<B,T,U>(const DdlaHandle_t&,char,int,int,const T*,const int*,const U*,int,T*,const int*)
#define TYPES(B) \
INSTANTIATE(B,float,float); INSTANTIATE(B,double,double); \
INSTANTIATE(B,std::complex<float>,std::complex<float>); \
INSTANTIATE(B,std::complex<double>,std::complex<double>); \
INSTANTIATE(B,std::complex<float>,float); INSTANTIATE(B,std::complex<double>,double)
DDLA_FOR_EACH_BACKEND(TYPES)
#undef TYPES
#undef INSTANTIATE
} // namespace ddla
