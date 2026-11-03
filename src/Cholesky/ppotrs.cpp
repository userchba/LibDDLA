#include "cholesky_workspace.h"
#include "check_agree.h"
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "ddla_desc.h"
#include <cassert>
#include <cstddef>
#include <vector>
#include "ddla_stream.h"
#include "require_backend.h"
namespace ddla{

namespace cholesky_impl {
template<DdlaBackend Backend>
void ppotrs_check(const DdlaHandle_t& h,char side,char uplo,char trans,int n,int nrhs,const void* A,const int* da,const void* B,const int* db,int location)
{
    detail::agree_check(h,"ppotrs",[&](detail::Agree&){
        check_desc(da,h);check_desc(db,h);detail::require_backend<Backend>(h,"ppotrs");
        if((side!='L' && side!='R') || (uplo!='L' && uplo!='U') || (trans!='N' && trans!='C') || n<0 || nrhs<0 || n>da[DDLA_M_] || n>da[DDLA_N_] || da[DDLA_MB_]!=da[DDLA_NB_])
            throw std::invalid_argument("ppotrs: invalid flags, dimensions or factor blocks");
        const int rows=side=='L'?n:nrhs,cols=side=='L'?nrhs:n;
        if(rows>db[DDLA_M_] || cols>db[DDLA_N_] || (location!=-1 && (location<1 || location>n)))throw std::invalid_argument("ppotrs: invalid right-hand side or head location");
        if(n && num_loc(n,da[DDLA_MB_],h->myprow_,da[DDLA_RSRC_],h->nprows_) && num_loc(n,da[DDLA_NB_],h->mypcol_,da[DDLA_CSRC_],h->npcols_) && !A)throw std::invalid_argument("ppotrs: null local factor");
        if(rows && cols && num_loc(rows,db[DDLA_MB_],h->myprow_,db[DDLA_RSRC_],h->nprows_) && num_loc(cols,db[DDLA_NB_],h->mypcol_,db[DDLA_CSRC_],h->npcols_) && !B)throw std::invalid_argument("ppotrs: null local right-hand side");
    });
}
template <DdlaBackend Backend, typename T>
void ppotrs_run(
    const DdlaHandle_t& handle, char side, char uplo, char trans,
    int n, int nrhs,
    T* A, const int* array_descA,
    T* B, const int* array_descB,
    bool is_nega, int location,
    void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes
)
{
    if(n==0 || nrhs==0)return;
    DdlaHandle_t ddla_handle=handle;
    // Head correction: when ppotrf was called with is_head=true and a
    // location != -1 (and != n), it relocated the head element to the last
    // global index by an in-place symmetric permutation of A.  The same
    // permutation must be applied to B on the side of the solve -- rows of B
    // for side='L' (B is n x nrhs), columns of B for side='R' (B is
    // nrhs x n) -- before the solve and again after (self-inverse), so X
    // comes back in the caller's original ordering.  This keeps direct
    // ppotrf+ppotrs head-correction users correct without external pswap
    // bookkeeping; pposv relies on it too.
    const bool needs_permute = (location != -1 && location != n);
    if(needs_permute){
        if(side == 'L')
            pswap_impl::pswap_run<Backend,T>(handle, nrhs, B, location, 1, array_descB, array_descB[DDLA_M_],
                        B, n,        1, array_descB, array_descB[DDLA_M_], backend_work, backend_bytes);
        else
            pswap_impl::pswap_run<Backend,T>(handle, array_descB[DDLA_M_], B, 1, location, array_descB, 1,
                        B, 1, n,        array_descB, 1, backend_work, backend_bytes);
    }

    // Solve op(A)*X=B (side='L') or X*op(A)=B (side='R') with the Cholesky
    // factor: for uplo='L', A = L*L^H.  Left solve applies L then L^H; right
    // solve applies L^H then L (the product order of the solves reverses).
    const bool left = (side == 'L');
    const char first_trans = (left == (uplo == 'L')) ? 'N' : 'C';
    const char second_trans = (left == (uplo == 'L')) ? 'C' : 'N';
    const int b_rows = left ? n : nrhs;
    const int b_cols = left ? nrhs : n;

    ptrtrs_impl::ptrtrs_run<Backend,T>(handle,
        side, uplo, first_trans, 'N', b_rows, b_cols,
        A, array_descA,
        B, array_descB,
        backend_work, backend_bytes, host_work, host_bytes
    );
    if(is_nega){
        int i_loc = indx_g2l_r(array_descA, handle, n - 1);
        int j_loc = indx_g2l_c(array_descA, handle, n - 1);
        if(i_loc >= 0 && j_loc >= 0){
            DDLA_CHECK(ddlaSynchronize(ddla_handle));
            T correction;
            DDLA_CHECK(ddlaMemcpy(ddla_handle, &correction, A + i_loc + j_loc * array_descA[DDLA_LLD_], sizeof(T)));
            DDLA_CHECK(ddlaSynchronize(ddla_handle));
            correction = -correction;
            DDLA_CHECK(ddlaMemcpy(ddla_handle, A + i_loc + j_loc * array_descA[DDLA_LLD_], &correction, sizeof(T)));
            DDLA_CHECK(ddlaSynchronize(ddla_handle));
        }
    }
    ptrtrs_impl::ptrtrs_run<Backend,T>(handle,
        side, uplo, second_trans, 'N', b_rows, b_cols,
        A, array_descA,
        B, array_descB,
        backend_work, backend_bytes, host_work, host_bytes
    );

    if(needs_permute){
        if(side == 'L')
            pswap_impl::pswap_run<Backend,T>(handle, nrhs, B, location, 1, array_descB, array_descB[DDLA_M_],
                        B, n,        1, array_descB, array_descB[DDLA_M_], backend_work, backend_bytes);
        else
            pswap_impl::pswap_run<Backend,T>(handle, array_descB[DDLA_M_], B, 1, location, array_descB, 1,
                        B, 1, n,        array_descB, 1, backend_work, backend_bytes);
    }
}
} // namespace cholesky_impl

template<DdlaBackend Backend,class T>
void ppotrs(const DdlaHandle_t& h,char side,char uplo,char trans,int n,int nrhs,T* A,const int* da,T* B,const int* db,bool negative,int location)
{
    cholesky_impl::ppotrs_check<Backend>(h,side,uplo,trans,n,nrhs,A,da,B,db,location);
    if(n==0 || nrhs==0)return;
    std::size_t backend_bytes=0,host_bytes=0;
    cholesky_impl::ppotrs_workspace_size<T>(h,side,n,nrhs,da,db,location,&backend_bytes,&host_bytes);
    vector<char> backend_work(h,backend_bytes);
    std::vector<char> host_work(host_bytes);
    cholesky_impl::ppotrs_run<Backend,T>(h,side,uplo,trans,n,nrhs,A,da,B,db,negative,location,
                                         backend_work.data(),backend_bytes,host_work.data(),host_bytes);
    DDLA_CHECK(ddlaSynchronize(h));
}

#define INSTANTIATE_PPOTRS(BACKEND, TYPE)                                   \
    template void ppotrs<BACKEND, TYPE>(const DdlaHandle_t&, char side, char uplo, char trans, \
    int n, int nrhs, \
    TYPE* A, const int* array_descA, \
    TYPE* B, const int* array_descB, \
    bool is_nega, int location); \
    template void cholesky_impl::ppotrs_run<BACKEND,TYPE>(const DdlaHandle_t&,char,char,char,int,int,TYPE*,const int*,TYPE*,const int*,bool,int, \
    void*,std::size_t,void*,std::size_t)

template void cholesky_impl::ppotrs_check<DdlaBackend::CPU>(const DdlaHandle_t&,char,char,char,int,int,const void*,const int*,const void*,const int*,int);
INSTANTIATE_PPOTRS(DdlaBackend::CPU, float);
INSTANTIATE_PPOTRS(DdlaBackend::CPU, double);
INSTANTIATE_PPOTRS(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PPOTRS(DdlaBackend::CPU, std::complex<double>);

#if DDLA_HAS_GPU
template void cholesky_impl::ppotrs_check<DdlaBackend::GPU>(const DdlaHandle_t&,char,char,char,int,int,const void*,const int*,const void*,const int*,int);
INSTANTIATE_PPOTRS(DdlaBackend::GPU, float);
INSTANTIATE_PPOTRS(DdlaBackend::GPU, double);
INSTANTIATE_PPOTRS(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PPOTRS(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_PPOTRS



}
