#include "pheevd_impl.h"
#include <ddla/ddla_vector.h>
#include "ddla_desc.h"
#include "ddla_stream.h"
#include "helpers.h"
#include "instantiate.h"
#include "lapack_impl.h"
#include "plan_util.h"
#include "require_backend.h"
#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ddla { namespace detail {
namespace {
template<class T> std::size_t local_bytes(std::size_t count) {
    if (count>(std::numeric_limits<std::size_t>::max()-255)/sizeof(T))
        throw std::length_error("pheevd: local workspace overflow");
    return padded_bytes<T>(count);
}
std::size_t local_sum(std::size_t a,std::size_t b) {
    if (b>std::numeric_limits<std::size_t>::max()-a)
        throw std::length_error("pheevd: local workspace overflow");
    return a+b;
}
} // namespace
template <DdlaBackend B, class T>
LocalPheevdPlan pheevd_local_check(const DdlaHandle_t& h, char job, char uplo, int n, int nev,
    const T* a, const int* da, const typename real_type<T>::type* w, const T* z, const int* dz)
{
    require_backend<B>(h,"pheevd");
    if (h->nprocs!=1 || n<0 || nev<0 || nev>n || (job!='N' && job!='V') || (uplo!='U' && uplo!='L'))
        throw std::invalid_argument("pheevd: invalid order/job/uplo");
    if (job=='V' && !nev) job='N';
    check_desc(da,h);
    if (da[DDLA_MB_]!=da[DDLA_NB_] || n>da[DDLA_M_] || n>da[DDLA_N_])
        throw std::invalid_argument("pheevd: square A blocks and valid extent required");
    if (n && (!a || !w)) throw std::invalid_argument("pheevd: missing active input/eigenvalues");
    LocalPheevdPlan p;
    p.backend=B; p.job=job; p.uplo=uplo; p.n=n; p.nev=nev; p.lda=da[DDLA_LLD_];
    if (job=='V') {
        check_desc(dz,h);
        if (dz[DDLA_MB_]!=da[DDLA_MB_] || dz[DDLA_NB_]!=da[DDLA_NB_] || n>dz[DDLA_M_] || nev>dz[DDLA_N_])
            throw std::invalid_argument("pheevd: incompatible output descriptor");
        if (n && !z) throw std::invalid_argument("pheevd: missing active output");
        p.ldz=dz[DDLA_LLD_];
        p.copy=a!=z || p.lda!=p.ldz;
        const auto span=[n](int ld,int cols) {
            if (!n || !cols) return std::size_t(0);
            const std::size_t limit=std::numeric_limits<std::size_t>::max()/sizeof(T);
            if (std::size_t(cols-1)>(limit-std::size_t(n))/std::size_t(ld))
                throw std::length_error("pheevd: local span overflow");
            return (std::size_t(cols-1)*ld+n)*sizeof(T);
        };
        p.overlap=p.copy && memory_overlaps(a,span(p.lda,n),z,span(p.ldz,nev));
    }
    return p;
}

template <class T>
LocalPheevdWorkspace pheevd_local_workspace_size(const DdlaHandle_t& h,const LocalPheevdPlan& p,
    T* a,typename real_type<T>::type* values)
{
    using R=typename real_type<T>::type;
    LocalPheevdWorkspace q;
    if (!p.n) return q;
    lapack_impl::heevd_workspace_size<T>(h,p.job,p.uplo,p.n,a,p.lda,values,
        q.lwork,q.lrwork,q.liwork,q.padding);
    if (q.lwork<0 || q.lrwork<0 || q.liwork<0)
        throw std::length_error("pheevd: invalid local vendor workspace count");
    q.backend_bytes=std::max(local_sum(local_bytes<T>(q.lwork),local_bytes<T>(q.padding)),
        local_bytes<T>(p.overlap?std::size_t(p.n)*p.nev:0));
    q.host_bytes=local_sum(local_bytes<R>(q.lrwork),local_bytes<int>(q.liwork));
    return q;
}

template <DdlaBackend B, class T>
void pheevd_local_run(const DdlaHandle_t& h,const LocalPheevdPlan& p,const LocalPheevdWorkspace& q,
    T* a,typename real_type<T>::type* w,T* z,int& info,typename real_type<T>::type* device_values,
    int* device_info,void* backend,std::size_t backend_bytes,void* host,std::size_t host_bytes)
{
    using R=typename real_type<T>::type;
    info=0;
    if (!p.n) return;
    assert(backend_bytes>=q.backend_bytes && host_bytes>=q.host_bytes);
    (void)backend_bytes; (void)host_bytes;
    char* at=static_cast<char*>(backend);
    T* work=q.lwork?take<T>(at,q.lwork):nullptr;
    T* padding=q.padding?take<T>(at,q.padding):nullptr;
    char* ht=static_cast<char*>(host);
    R* rwork=q.lrwork?take<R>(ht,q.lrwork):nullptr;
    int* iwork=q.liwork?take<int>(ht,q.liwork):nullptr;
    R* values=B==DdlaBackend::GPU?device_values:w;
    int* status=B==DdlaBackend::GPU?device_info:&info;
    try {
        lapack_impl::heevd_run<B,T>(h,p.job,p.uplo,p.n,a,p.lda,values,work,q.lwork,
            rwork,q.lrwork,iwork,q.liwork,status,padding);
        if constexpr (B==DdlaBackend::GPU) {
            DDLA_CHECK(ddlaMemcpy(h,&info,device_info,sizeof(int)));
            DDLA_CHECK(ddlaMemcpy(h,w,values,std::size_t(p.n)*sizeof(R)));
            DDLA_CHECK(ddlaSynchronize(h));
        }
        if (info) throw std::runtime_error("pheevd: local eigensolver failed, info="+std::to_string(info));
        if (p.copy) {
            const T* source=a;
            int source_ld=p.lda;
            if (p.overlap) {
                T* saved=static_cast<T*>(backend);
                DDLA_CHECK(ddlaMemcpy2D(h,saved,std::size_t(p.n)*sizeof(T),a,
                    std::size_t(p.lda)*sizeof(T),std::size_t(p.n)*sizeof(T),p.nev));
                source=saved; source_ld=p.n;
            }
            DDLA_CHECK(ddlaMemcpy2D(h,z,std::size_t(p.ldz)*sizeof(T),source,
                std::size_t(source_ld)*sizeof(T),std::size_t(p.n)*sizeof(T),p.nev));
        }
        if constexpr (B==DdlaBackend::GPU) {
            DDLA_CHECK(ddlaSynchronize(h));
        }
    } catch (...) {
        if constexpr (B==DdlaBackend::GPU) {
            try { (void)ddlaSynchronize(h); } catch (...) {}
        }
        throw;
    }
}

template <DdlaBackend B, class T>
void pheevd_local(const DdlaHandle_t& h,char job,char uplo,int n,int nev,T* a,const int* da,
    typename real_type<T>::type* w,T* z,const int* dz,int& info)
{
    using R=typename real_type<T>::type;
    info=0;
    const auto p=pheevd_local_check<B,T>(h,job,uplo,n,nev,a,da,w,z,dz);
    if (!n) return;
    // Explicit O(n) backend query/value resources: never query using host w
    // or a guessed alias of A on a GPU handle.
    ddla::vector<R> values(h,B==DdlaBackend::GPU?n:0);
    ddla::vector<int> status(h,B==DdlaBackend::GPU?1:0);
    const auto q=pheevd_local_workspace_size<T>(h,p,a,B==DdlaBackend::GPU?values.data():w);
    ddla::vector<char> backend(h,q.backend_bytes);
    std::vector<char> host(q.host_bytes);
    pheevd_local_run<B,T>(h,p,q,a,w,z,info,values.data(),status.data(),
        backend.data(),backend.size(),host.data(),host.size());
}
#define LOCAL_SIZE(T) template LocalPheevdWorkspace pheevd_local_workspace_size<T>(const DdlaHandle_t&,const LocalPheevdPlan&,T*,typename real_type<T>::type*)
DDLA_FOR_EACH_TYPE(LOCAL_SIZE)
#undef LOCAL_SIZE
#define LOCAL(B,T) \
 template LocalPheevdPlan pheevd_local_check<B,T>(const DdlaHandle_t&,char,char,int,int,const T*,const int*,const typename real_type<T>::type*,const T*,const int*); \
 template void pheevd_local_run<B,T>(const DdlaHandle_t&,const LocalPheevdPlan&,const LocalPheevdWorkspace&,T*,typename real_type<T>::type*,T*,int&,typename real_type<T>::type*,int*,void*,std::size_t,void*,std::size_t); \
 template void pheevd_local<B,T>(const DdlaHandle_t&,char,char,int,int,T*,const int*,typename real_type<T>::type*,T*,const int*,int&)
DDLA_FOR_EACH_BACKEND_TYPE(LOCAL)
#undef LOCAL
} } // namespace ddla::detail
