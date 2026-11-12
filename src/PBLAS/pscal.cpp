#include <ddla/ddla.h>
#include "BLAS/blas_impl.h"
#include "check_agree.h"
#include "ddla_desc.h"
#include "ddla_stream.h"
#include "require_backend.h"
#include "instantiate.h"

#include <complex>
#include <stdexcept>

namespace ddla {
namespace {
template <DdlaBackend Backend, typename T, typename Alpha>
void pscal_matrix(const DdlaHandle_t& h, int m, int n, const Alpha& alpha,
           T* a, const int* desc)
{
    int rows=0,columns=0;
    detail::agree_check(h,"pscal",[&](detail::Agree&) {
        detail::require_backend<Backend>(h,"pscal");
        check_desc(desc,h);
        if (m<0 || n<0 || m>desc[DDLA_M_] || n>desc[DDLA_N_])
            throw std::invalid_argument("pscal: region exceeds descriptor dimensions");
        rows=num_loc(m,desc[DDLA_MB_],h->myprow_,desc[DDLA_RSRC_],h->nprows_);
        columns=num_loc(n,desc[DDLA_NB_],h->mypcol_,desc[DDLA_CSRC_],h->npcols_);
        blas_impl::geam_check<Backend,T>(h,'N','N',rows,columns,T(alpha),a,desc[DDLA_LLD_],
            T(0),nullptr,1,a,desc[DDLA_LLD_]);
    });
    int signature[9]={m,n};
    detail::descriptor_signature(signature+2,desc);
    detail::agree_signature(h,"pscal",signature);
    // Each rank updates only its owned entries. PGEADD's general alias
    // workspace is unnecessary for this identical-layout in-place operation.
    blas_impl::matrix_scal_run<Backend,T,Alpha>(h,rows,columns,alpha,a,desc[DDLA_LLD_]);
}

} // namespace
template <DdlaBackend Backend,typename T>
void pscal(const DdlaHandle_t& h,int m,int n,const T& alpha,T* a,const int* desc)
{pscal_matrix<Backend,T,T>(h,m,n,alpha,a,desc);}
template <DdlaBackend Backend,typename T,typename Alpha>
typename std::enable_if<
    std::is_same<T, std::complex<Alpha>>::value,
    void
>::type
pscal(const DdlaHandle_t& h,int m,int n,const Alpha& alpha,T* a,const int* desc)
{pscal_matrix<Backend,T,Alpha>(h,m,n,alpha,a,desc);}
#define MIXED_PSCAL(B,R) template void pscal<B,std::complex<R>,R>(const DdlaHandle_t&,int,int,const R&,std::complex<R>*,const int*)
MIXED_PSCAL(DdlaBackend::CPU,float);MIXED_PSCAL(DdlaBackend::CPU,double);
#if DDLA_HAS_GPU
MIXED_PSCAL(DdlaBackend::GPU,float);MIXED_PSCAL(DdlaBackend::GPU,double);
#endif
#undef MIXED_PSCAL

#define INSTANTIATE(B,T) template void pscal<B,T>(const DdlaHandle_t&,int,int,const T&,T*,const int*)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE)
#undef INSTANTIATE
}
