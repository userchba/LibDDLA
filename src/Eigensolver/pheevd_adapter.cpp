#include "pheevd_impl.h"
#include <ddla/ddla_vector.h>
#include "ddla_desc.h"
#include "ddla_stream.h"
#include "helpers.h"
#include "instantiate.h"
#include <stdexcept>
#include <type_traits>

namespace ddla { namespace detail {
namespace {
void local_check(const DdlaHandle_t& h, char job, char uplo, int n,
                  const void* a, const int* da, const void* w, const void* z, const int* dz)
{
    if (n < 0 || (job != 'N' && job != 'V') || (uplo != 'U' && uplo != 'L'))
        throw std::invalid_argument("pheevd: invalid order/job/uplo");
    check_desc(da, h);
    if (da[DDLA_MB_] != da[DDLA_NB_] || n > da[DDLA_M_] || n > da[DDLA_N_])
        throw std::invalid_argument("pheevd: square A blocks and valid extent required");
    if (n && (!a || !w))
        throw std::invalid_argument("pheevd: missing active input/eigenvalues");
    if (job == 'V') {
        check_desc(dz, h);
        if (dz[DDLA_MB_] != da[DDLA_MB_] || dz[DDLA_NB_] != da[DDLA_NB_] ||
            n > dz[DDLA_M_] || n > dz[DDLA_N_])
            throw std::invalid_argument("pheevd: incompatible output descriptor");
        if (n && !z) throw std::invalid_argument("pheevd: missing active output");
    }
}
} // namespace

template <DdlaBackend B, class T>
void pheevd_local(const DdlaHandle_t& h, char job, char uplo, int n, T* a, const int* da,
    typename real_type<T>::type* w, T* z, const int* dz, int& info)
{
    using R = typename real_type<T>::type;
    info = 0;
    local_check(h, job, uplo, n, a, da, w, z, dz);
    if (!n) return;
    const int lda = da[DDLA_LLD_];
    const int ldz = job == 'V' ? dz[DDLA_LLD_] : 1;
    const bool copy = job == 'V' && (a != z || lda != ldz);
    const auto span = [n](int ld) { return (std::size_t(n - 1) * ld + n) * sizeof(T); };
    const bool overlap = copy && memory_overlaps(a, span(lda), z, span(ldz));
    ddla::vector<T> saved(h, overlap ? std::size_t(n) * n : 0);
    ddla::vector<R> device_values(h, B == DdlaBackend::GPU ? n : 0);
    ddla::vector<int> device_info(h, B == DdlaBackend::GPU ? 1 : 0);
    R* values = B == DdlaBackend::GPU ? device_values.data() : w;
    int* status=B==DdlaBackend::GPU ? device_info.data() : &info;
    if constexpr (std::is_same_v<T,R>)
        syevd<B,T>(h,job,uplo,n,a,lda,values,status);
    else
        heevd<B,T>(h,job,uplo,n,a,lda,values,status);
    if constexpr (B == DdlaBackend::GPU) {
        DDLA_CHECK(ddlaMemcpy(h, &info, device_info.data(), sizeof(int)));
        DDLA_CHECK(ddlaMemcpy(h, w, values, std::size_t(n) * sizeof(R)));
        DDLA_CHECK(ddlaSynchronize(h));
    }
    if (info) throw std::runtime_error("pheevd: local eigensolver failed, info=" + std::to_string(info));
    if (copy) {
        const T* source = a;
        int source_ld = lda;
        if (overlap) {
            DDLA_CHECK(ddlaMemcpy2D(h, saved.data(), std::size_t(n) * sizeof(T), a,
                std::size_t(lda) * sizeof(T), std::size_t(n) * sizeof(T), n));
            source = saved.data();
            source_ld = n;
        }
        DDLA_CHECK(ddlaMemcpy2D(h, z, std::size_t(ldz) * sizeof(T), source,
            std::size_t(source_ld) * sizeof(T), std::size_t(n) * sizeof(T), n));
    }
    // Retire device workspace reads and any copy using saved before owners die.
    if constexpr (B == DdlaBackend::GPU) DDLA_CHECK(ddlaSynchronize(h));
}

#define ADAPTER(B,T) template void pheevd_local<B,T>(const DdlaHandle_t&,char,char,int,T*,const int*,typename real_type<T>::type*,T*,const int*,int&)
DDLA_FOR_EACH_BACKEND_TYPE(ADAPTER)
#undef ADAPTER
} } // namespace ddla::detail
