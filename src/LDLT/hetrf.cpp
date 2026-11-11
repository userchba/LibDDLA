#include <ddla/ddla.h>
#include "ldlt_impl.h"
#include "hetrf_gpu.h"
#include "hetrf_native.h"
#include "instantiate.h"

namespace ddla {

template <DdlaBackend Backend, typename T>
void hetrf(const DdlaHandle_t &h, char uplo, int n, T *A, int lda, int *ipiv, int &info) {
#if DDLA_HAS_GPU
    if constexpr (Backend == DdlaBackend::GPU && std::is_same_v<T, std::complex<double>>) {
        ldlt_impl::hetrf_gpu(h, uplo, n, A, lda, ipiv, info);
        return;
    }
#endif
    ldlt_impl::local<Backend, true, true, false, T>(h, uplo, n, 0, const_cast<T *>(A), lda,
                                                    const_cast<int *>(ipiv), nullptr, 1, info);
}

template <DdlaBackend Backend, typename T>
void hetrf_nopiv(const DdlaHandle_t &h, char uplo, int n, T *A, int lda, int &info) {
#if DDLA_HAS_GPU
    if constexpr (Backend == DdlaBackend::GPU) {
        ldlt_impl::hetrf_nopiv_gpu(h, uplo, n, A, lda, info);
        return;
    }
#endif
    ldlt_impl::local<Backend, true, false, false, T>(h, uplo, n, 0, const_cast<T *>(A), lda,
                                                     nullptr, nullptr, 1, info);
}

#define DDLA_INSTANTIATE(BACKEND_, TYPE_) \
    template void hetrf<BACKEND_, TYPE_>(const DdlaHandle_t &, char, int, TYPE_ *, int, int *, int &); \
    template void hetrf_nopiv<BACKEND_, TYPE_>(const DdlaHandle_t &, char, int, TYPE_ *, int, int &)
DDLA_FOR_EACH_BACKEND_TYPE(DDLA_INSTANTIATE)
#undef DDLA_INSTANTIATE

} // namespace ddla
