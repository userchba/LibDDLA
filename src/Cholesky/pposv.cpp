#include "cholesky_workspace.h"
#include "ddla_stream.h"
#include "check_agree.h"
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "ddla_desc.h"
#include <cassert>
#include <cstddef>
#include <vector>
#include "require_backend.h"

namespace ddla{

namespace cholesky_impl {
template<DdlaBackend Backend>
void pposv_check(const DdlaHandle_t& h,char side,char uplo,char trans,int n,int nrhs,const void* A,int ia,int ja,const int* da,const void* B,int ib,int jb,const int* db)
{
    ppotrf_check<Backend>(h,uplo,n,A,ia,ja,da);
    ppotrs_check<Backend>(h,side,uplo,trans,n,nrhs,A,da,B,db);
    detail::agree_check(h,"pposv",[&](detail::Agree&){if(ib!=1 || jb!=1)throw std::invalid_argument("pposv: offsets must be one");});
}
template <DdlaBackend Backend, typename T>
void pposv_run(
    const DdlaHandle_t& handle, char side, char uplo, char trans,
    int n, int nrhs,
    T* A, int ia, int ja, const int* array_descA,
    T* B, int ib, int jb, const int* array_descB,
    int& info, // host pointer
    void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes
)
{
    if(n==0){info=0;return;}
    // The factorization and the solve run one after the other on the whole scratch.
    ppotrf_run<Backend,T>(handle, uplo, n, A, ia, ja, array_descA, info,
                          backend_work, backend_bytes, host_work, host_bytes);
    if(info == 0)
        ppotrs_run<Backend,T>(handle, side, uplo, trans, n, nrhs, A, array_descA, B, array_descB,
                              backend_work, backend_bytes, host_work, host_bytes);
    return;
}

} // namespace cholesky_impl
template<DdlaBackend Backend,class T>
void pposv(const DdlaHandle_t& h,char side,char uplo,char trans,int n,int nrhs,T* A,int ia,int ja,const int* da,T* B,int ib,int jb,const int* db,int& info)
{
    cholesky_impl::pposv_check<Backend>(h,side,uplo,trans,n,nrhs,A,ia,ja,da,B,ib,jb,db);
    if(n==0){info=0;return;}
    std::size_t backend_bytes=0,host_bytes=0;
    cholesky_impl::pposv_workspace_size<T>(h,side,uplo,n,nrhs,A,da,db,&backend_bytes,&host_bytes);
    vector<char> backend_work(h,backend_bytes);
    std::vector<char> host_work(host_bytes);
    cholesky_impl::pposv_run<Backend,T>(h,side,uplo,trans,n,nrhs,A,ia,ja,da,B,ib,jb,db,info,
                                        backend_work.data(),backend_bytes,host_work.data(),host_bytes);
    DDLA_CHECK(ddlaSynchronize(h));
}

#define INSTANTIATE_PPOSV(BACKEND, TYPE)                                   \
    template void pposv<BACKEND, TYPE>(const DdlaHandle_t&, char side, char uplo, char trans, \
    int n, int nrhs, \
    TYPE* A, int ia, int ja, const int* array_descA, \
    TYPE* B, int ib, int jb, const int* array_descB, \
    int& info); \
    template void cholesky_impl::pposv_run<BACKEND,TYPE>(const DdlaHandle_t&,char,char,char,int,int,TYPE*,int,int,const int*,TYPE*,int,int,const int*,int&, \
    void*,std::size_t,void*,std::size_t)

template void cholesky_impl::pposv_check<DdlaBackend::CPU>(const DdlaHandle_t&,char,char,char,int,int,const void*,int,int,const int*,const void*,int,int,const int*);
INSTANTIATE_PPOSV(DdlaBackend::CPU, float);
INSTANTIATE_PPOSV(DdlaBackend::CPU, double);
INSTANTIATE_PPOSV(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PPOSV(DdlaBackend::CPU, std::complex<double>);

#if DDLA_HAS_GPU
template void cholesky_impl::pposv_check<DdlaBackend::GPU>(const DdlaHandle_t&,char,char,char,int,int,const void*,int,int,const int*,const void*,int,int,const int*);
INSTANTIATE_PPOSV(DdlaBackend::GPU, float);
INSTANTIATE_PPOSV(DdlaBackend::GPU, double);
INSTANTIATE_PPOSV(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PPOSV(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_PPOSV


} // namespace ddla
