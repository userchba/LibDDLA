#include "diag_logdet_impl.h"
#include "backend_api.h"
#include "ddla_stream.h"
#include "instantiate.h"
#include "require_backend.h"
#include <algorithm>
#include <stdexcept>

namespace ddla {
namespace detail {

template<DdlaBackend B, class T>
DiagLogState diagonal_logsum(const DdlaHandle_t& h, const T* a, const DiagLogLayout& layout)
{
    if (layout.count == 0) return {};
    if constexpr (B == DdlaBackend::CPU) {
        double magnitude = 0, phase = 0;
        int flags = 0, negative = 0;
        const DiagLogTerm<T> term{a, layout};
        const int threads = std::max(1, h->host_threads);
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) if(layout.count >= 1024) \
    reduction(+:magnitude,phase) reduction(|:flags) reduction(^:negative)
#endif
        for (int item = 0; item < layout.count; ++item) {
            const auto value = term(item);
            magnitude += value.magnitude;
            phase += value.phase;
            flags |= value.flags & 3;
            negative ^= value.flags & 4;
        }
        return {magnitude, phase, flags | negative};
    } else {
#if DDLA_HAS_GPU
        return diagonal_logsum_gpu(h, a, layout);
#endif
    }
}

} // namespace detail

template<DdlaBackend B, class T>
void diag_logdet(const DdlaHandle_t& h, int n, const T* a, int lda,
                 std::complex<typename real_type<T>::type>& result)
{
    detail::require_backend<B>(h, "diag_logdet");
    if (n < 0 || lda < std::max(1, n) || (n > 0 && !a))
        throw std::invalid_argument("diag_logdet: invalid order, leading dimension or pointer");
    detail::check_diag_span<T>(n, n, lda);
    const detail::DiagLogLayout layout{n, n, lda, 1, 1, 1, 1, 0, 0, 0, 0, true};
    const auto state = detail::diagonal_logsum<B>(h, a, layout);
    if (state.flags & 1) throw std::domain_error("diag_logdet: nonfinite diagonal");
    result = detail::finish_diagonal_log<typename real_type<T>::type>(
        state.magnitude, state.phase, state.flags & 2, state.flags >> 2);
}

#define LOCAL(B,T) \
    template void diag_logdet<B,T>(const DdlaHandle_t&,int,const T*,int,std::complex<typename real_type<T>::type>&); \
    template detail::DiagLogState detail::diagonal_logsum<B,T>(const DdlaHandle_t&,const T*,const detail::DiagLogLayout&);
#define TYPES(B) LOCAL(B,float) LOCAL(B,double) LOCAL(B,std::complex<float>) LOCAL(B,std::complex<double>)
DDLA_FOR_EACH_BACKEND(TYPES)
#undef TYPES
#undef LOCAL
} // namespace ddla
