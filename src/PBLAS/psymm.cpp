#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "phemm_impl.h"
#include <cctype>

#include <vector>

namespace ddla {

template <DdlaBackend Backend, typename T>
void psymm(const DdlaHandle_t& h, char side, char uplo, int m, int n,
    const T& alpha, const T* A, const int* da, const T* B, const int* db,
    const T& beta, T* C, const int* dc)
{
    const bool noop=phemm_impl::phemm_check<Backend,T>(h,side,uplo,m,n,alpha,A,da,B,db,beta,C,dc,false);
    if (noop) return;
    side=static_cast<char>(std::toupper(static_cast<unsigned char>(side)));
    uplo=static_cast<char>(std::toupper(static_cast<unsigned char>(uplo)));
    std::size_t backend_bytes=0,host_bytes=0;
    phemm_impl::phemm_workspace_size<T>(h,side,m,n,alpha==T(0),da,db,dc,false,&backend_bytes,&host_bytes);
    vector<char> backend_work(h,backend_bytes);
    std::vector<char> host_work(host_bytes);
    phemm_impl::phemm_run<Backend,T>(h,side,uplo,m,n,alpha,A,da,B,db,beta,C,dc,false,
        backend_work.data(),backend_bytes,host_work.data(),host_bytes);
}

#define INSTANTIATE_PSYMM(B,T) \
 template void psymm<B,T>(const DdlaHandle_t&,char,char,int,int,const T&,const T*,const int*,const T*,const int*,const T&,T*,const int*)
#define INSTANTIATE_PSYMM_TYPES(B) \
 INSTANTIATE_PSYMM(B,float); INSTANTIATE_PSYMM(B,double); \
 INSTANTIATE_PSYMM(B,std::complex<float>); INSTANTIATE_PSYMM(B,std::complex<double>)
INSTANTIATE_PSYMM_TYPES(DdlaBackend::CPU);
#if DDLA_HAS_GPU
INSTANTIATE_PSYMM_TYPES(DdlaBackend::GPU);
#endif
#undef INSTANTIATE_PSYMM_TYPES
#undef INSTANTIATE_PSYMM
} // namespace ddla
