#include "stedc_impl.h"
#include "BLAS/blas_impl.h"
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "host_threads.h"
#include "instantiate.h"
#include "matrix_transfer.h"
#include "plan_util.h"
#include "require_backend.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

extern "C" {
void cstedc_(const char*, const int*, float*, float*, std::complex<float>*, const int*, std::complex<float>*, const int*, float*, const int*, int*, const int*, int*);
void zstedc_(const char*, const int*, double*, double*, std::complex<double>*, const int*, std::complex<double>*, const int*, double*, const int*, int*, const int*, int*);
}

namespace ddla { namespace spectral_impl {
namespace {
template<class T>
void cpu_stedc(char compz, int n, typename real_type<T>::type* d,
               typename real_type<T>::type* e, T* z, int ldz, T* work, int lwork,
               typename real_type<T>::type* rwork, int lrwork,
               int* iwork, int liwork, int& info)
{
    if constexpr (std::is_same_v<T, typename real_type<T>::type>)
        ddla::spectral_impl::stedc(compz,n,d,e,z,ldz,work,lwork,iwork,liwork,info);
    else if constexpr (std::is_same_v<T, std::complex<float>>)
        cstedc_(&compz, &n, d, e, z, &ldz, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
    else
        zstedc_(&compz, &n, d, e, z, &ldz, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
}

template<class R>
int workspace_count(R query, long double minimum = 1)
{
    double count = query;
    // A float query cannot represent every LP64 count above 2^24. Preserve
    // the queried optimum and enforce the checked, documented integer minimum.
    if constexpr (std::is_same_v<R,float>)
        if (std::isfinite(count) && count >= 16777216.)
            count = std::min(double(std::numeric_limits<int>::max()),
                std::max(count,static_cast<double>(minimum)));
    if (!std::isfinite(count) || count < 1 || count > std::numeric_limits<int>::max())
        throw std::length_error("stedc: LAPACK workspace exceeds LP64 range");
    return static_cast<int>(count);
}
} // namespace

template<class T>
StedcPlan stedc_plan(const DdlaHandle_t& h, char compz, int n, int ldz)
{
    using R = typename real_type<T>::type;
    using detail::padded_bytes;
    if (!h) throw std::invalid_argument("stedc: null handle");
    compz = static_cast<char>(std::toupper(static_cast<unsigned char>(compz)));
    if (n < 0 || (compz != 'N' && compz != 'I' && compz != 'V'))
        throw std::invalid_argument("stedc: invalid order or COMPZ");
    StedcPlan p;
    p.compz = compz;
    p.n = n;
    if (!n) return p;
    if (compz != 'N' && ldz < n)
        throw std::invalid_argument("stedc: leading dimension is too small");
    // Bound products before either LAPACK's INTEGER arithmetic or byte carving.
    const std::size_t nn = n;
    int lg = 0;
    for (std::size_t power = 1; power < nn; power *= 2) ++lg;
    const long double order = n;
    const long double max_work = compz == 'N' ? 1 :
        (compz == 'V' ? 1 + 3*order + 2*order*lg + 4*order*order :
         1 + 4*order + (std::is_same_v<T,R> ? 1 : 2)*order*order);
    const bool host_solve = h->backend == DdlaBackend::CPU || compz == 'N';
    if (host_solve) {
        if (max_work > std::size_t(std::numeric_limits<int>::max()))
            throw std::length_error("stedc: LAPACK workspace exceeds LP64 range");
        T query{}, z{};
        R rq{}, d{}, e{};
        int iq = 0, info = 0;
        cpu_stedc(compz, n, &d, &e, &z, std::max(1,n), &query, -1, &rq, -1, &iq, -1, info);
        if (info) throw std::runtime_error("stedc: LAPACK workspace query failed");
        const long double work_min = std::is_same_v<T,R> ? max_work :
            (compz == 'V' && n > 1 ? order*order : 1);
        p.lwork = workspace_count(std::real(query),work_min);
        if constexpr (!std::is_same_v<T,R>)
            // CSTEDC's REAL RWORK query does not use SROUNDUP_LWORK.
            p.lrwork = workspace_count(rq,max_work);
        p.liwork = workspace_count(iq);
        const auto bytes = padded_bytes<T>(p.lwork) + padded_bytes<R>(p.lrwork) + padded_bytes<int>(p.liwork);
        if (h->backend == DdlaBackend::CPU) {
            // The installed OpenBLAS 0.3.29 CSTEDC V can write a compact result
            // despite LDZ>n (direct LAPACK reproducer). Keep the vendor call
            // compact and copy only active rows back; never expose padding.
            p.pack_vectors = !std::is_same_v<T,R> && compz == 'V' && ldz != n && n > 1;
            p.backend_bytes = bytes + (p.pack_vectors ? padded_bytes<T>(nn*nn) : 0);
        } else p.host_bytes = bytes;
        return p;
    }
    if (nn > std::numeric_limits<std::size_t>::max() / nn / (8*sizeof(T)))
        throw std::length_error("stedc: matrix byte size overflow");
    p.rows.backend = DdlaBackend::GPU;
    p.rows.n = n;
    const bool real_identity = compz == 'I' && std::is_same_v<T,R>;
    p.rows.ldy = real_identity ? ldz : n;
    p.rows.borrow_q2 = !real_identity || ldz == n;
    p.rows.threads = h->host_threads;
    // Local solves use wider BLAS-3 updates than the memory-bounded distributed
    // default. Smaller local problems benefit from 128-row leaves; larger
    // problems retain the existing 1024-row bound.
    p.rows.leaf = stedc_gpu_leaf_size(n);
    p.rows.strip = std::min(n,512);
    stedc_rows_workspace_size<R>(h, p.rows, &p.engine_backend, &p.engine_host);
    p.backend_bytes = (real_identity ? 0 : padded_bytes<R>(nn*nn));
    if (compz == 'V') {
        if constexpr (!std::is_same_v<T,R>)
            blas_impl::mixed_gemm_workspace_size<R>(h,n,n,n,true,&p.multiply_bytes);
    }
    const auto multiply = compz == 'V' ? padded_bytes<T>(nn*nn) + p.multiply_bytes : 0;
    p.backend_bytes += std::max(p.engine_backend,multiply);
    p.host_bytes = p.engine_host;
    return p;
}

template<DdlaBackend B, class T>
void stedc_run(const DdlaHandle_t& h, const StedcPlan& p, PstedcResources& resources,
               typename real_type<T>::type* d, typename real_type<T>::type* e,
               T* z, int ldz, int& info, void* backend_work, void* host_work, typename real_type<T>::type* scratch)
{
    using R = typename real_type<T>::type;
    using detail::take;
    info = 0;
    if (!p.n) return;
    detail::ScopedBlasThreads threads(B == DdlaBackend::CPU ? h->host_threads : 1);
    char* hw = static_cast<char*>(B == DdlaBackend::CPU ? backend_work : host_work);
    if constexpr (B == DdlaBackend::CPU) {
        T* packed = p.pack_vectors ? take<T>(hw,std::size_t(p.n)*p.n) : nullptr;
        if (packed)
            DDLA_CHECK(ddlaMemcpy2D(h,packed,std::size_t(p.n)*sizeof(T),z,
                std::size_t(ldz)*sizeof(T),std::size_t(p.n)*sizeof(T),p.n));
        T* work = take<T>(hw,p.lwork);
        R* rwork = take<R>(hw,p.lrwork);
        int* iwork = take<int>(hw,p.liwork);
        // LAPACK does not reference Z for N, or E for n=1. Supply valid dummies
        // instead of passing null through a Fortran assumed-size argument.
        T dummy{};
        R off{};
        cpu_stedc(p.compz,p.n,d,e?e:&off,packed?packed:(z?z:&dummy),
                   packed?p.n:(p.compz=='N'?1:ldz),work,p.lwork,rwork,p.lrwork,iwork,p.liwork,info);
        if (packed)
            DDLA_CHECK(ddlaMemcpy2D(h,z,std::size_t(ldz)*sizeof(T),packed,
                std::size_t(p.n)*sizeof(T),std::size_t(p.n)*sizeof(T),p.n));
        return;
    } else {
#if DDLA_HAS_GPU
        if (p.compz == 'N') {
            T* work = take<T>(hw,p.lwork);
            R* rwork = take<R>(hw,p.lrwork);
            int* iwork = take<int>(hw,p.liwork);
            T dummy{};
            R off{};
            cpu_stedc('N',p.n,d,e?e:&off,&dummy,1,work,p.lwork,rwork,p.lrwork,iwork,p.liwork,info);
            return;
        }
        char* bw = static_cast<char*>(backend_work);
        const std::size_t square = std::size_t(p.n)*p.n;
        const bool real_identity = p.compz == 'I' && std::is_same_v<T,R>;
        R* vectors = real_identity ? reinterpret_cast<R*>(z) : take<R>(bw,square);
        char* multiply = bw;
        T* product = p.compz == 'V' ? take<T>(multiply,square) : nullptr;
        auto rows = p.rows;
        if (real_identity) rows.ldy = ldz;
        R off{};
        try {
            // Explicit one-rank policy: no MPI operations, even if h belongs
            // to a larger grid and other ranks solve different local matrices.
            stedc_rows_run<B,R>(h,rows,resources,d,e?e:&off,d,vectors,scratch,
                             bw,p.engine_backend,hw,p.engine_host);
        } catch (...) {
            info = 1;
            throw;
        }
        if (p.compz == 'V') {
            if constexpr (std::is_same_v<T,R>)
                blas_impl::gemm_run<B,T>(h,'N','N',p.n,p.n,p.n,T(1),z,ldz,vectors,p.n,T(0),product,p.n);
            else
                blas_impl::gemm_run<B,R>(h,'N','N',p.n,p.n,p.n,T(1),z,ldz,vectors,p.n,T(0),product,p.n,multiply,p.multiply_bytes);
            DDLA_CHECK(ddlaMemcpy2D(h,z,std::size_t(ldz)*sizeof(T),product,
                std::size_t(p.n)*sizeof(T),std::size_t(p.n)*sizeof(T),p.n));
        } else if constexpr (!std::is_same_v<T,R>) {
            matrix_impl::convert_rows<B,T>(h,vectors,p.n,z,ldz,p.n,p.n,false);
        }
        // The row engine already completed real I output. Conversion and V's
        // product/copy queue additional work which must retire before scratch
        // can be reused; real I needs no second completion fence.
        if (p.compz == 'V' || !std::is_same_v<T,R>)
            DDLA_CHECK(ddlaSynchronize(h));
#endif
    }
}

#define PLAN(T) template StedcPlan stedc_plan<T>(const DdlaHandle_t&,char,int,int)
#define RUN(B,T) template void stedc_run<B,T>(const DdlaHandle_t&,const StedcPlan&,PstedcResources&,typename real_type<T>::type*,typename real_type<T>::type*,T*,int,int&,void*,void*,typename real_type<T>::type*)
DDLA_FOR_EACH_TYPE(PLAN)
DDLA_FOR_EACH_BACKEND_TYPE(RUN)
#undef PLAN
#undef RUN
} } // namespace ddla::spectral_impl

namespace ddla {
template<DdlaBackend B, class T>
void stedc(const DdlaHandle_t& h, char compz, int n,
           typename real_type<T>::type* d, typename real_type<T>::type* e,
           T* z, int ldz, int& info)
{
    detail::require_backend<B>(h,"stedc");
    const auto p = spectral_impl::stedc_plan<T>(h,compz,n,ldz);
    if (n && (!d || (n>1 && !e) || (p.compz!='N' && (!z || ldz<std::max(1,n)))))
        throw std::invalid_argument("stedc: invalid diagonal/off-diagonal/eigenvector storage");
    if constexpr (B == DdlaBackend::GPU)
        if (n && p.compz!='N' && !h->initialized)
            throw std::invalid_argument("stedc: GPU handle must be initialized with ddlaSet");
    ddla::vector<char> backend(h,p.backend_bytes);
    std::vector<char> host(p.host_bytes);
    spectral_impl::PstedcResources resources(h,p.rows);
    spectral_impl::stedc_run<B,T>(h,p,resources,d,e,z,ldz,info,backend.data(),host.data());
}
#define PUBLIC(B,T) template void stedc<B,T>(const DdlaHandle_t&,char,int,typename real_type<T>::type*,typename real_type<T>::type*,T*,int,int&)
DDLA_FOR_EACH_BACKEND_TYPE(PUBLIC)
#undef PUBLIC
} // namespace ddla
