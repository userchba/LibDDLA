#include <ddla/ddla.h>
#include "check_agree.h"
#include <ddla/ddla_vector.h>
#include "phemm_impl.h"
#include "instantiate.h"
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
    vector<char> backend_work(h);
    std::vector<char> host_work;
    auto prepare = [&] {
        phemm_impl::phemm_workspace_size<T>(h,side,m,n,alpha==T(0),da,db,dc,false,&backend_bytes,&host_bytes);
        backend_work.resize(backend_bytes);
        host_work.resize(host_bytes);
    };
    if (h->nprocs > 1)
        detail::agree_local(h, "psymm workspace preparation", prepare);
    else
        prepare();
    phemm_impl::phemm_run<Backend,T>(h,side,uplo,m,n,alpha,A,da,B,db,beta,C,dc,false,
        backend_work.data(),backend_bytes,host_work.data(),host_bytes);
}

#define INSTANTIATE_PSYMM(B,T) \
 template void psymm<B,T>(const DdlaHandle_t&,char,char,int,int,const T&,const T*,const int*,const T*,const int*,const T&,T*,const int*)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_PSYMM)
#undef INSTANTIATE_PSYMM
} // namespace ddla
