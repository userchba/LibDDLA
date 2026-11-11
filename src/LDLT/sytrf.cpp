#include <ddla/ddla.h>
#include "ldlt_impl.h"
#include "instantiate.h"

namespace ddla {

template <DdlaBackend Backend, typename T>
void sytrf(const DdlaHandle_t &h, char uplo, int n, T *A, int lda, int *ipiv, int &info) {
    ldlt_impl::local<Backend, false, true, false, T>(h, uplo, n, 0, const_cast<T *>(A), lda,
                                                     const_cast<int *>(ipiv), nullptr, 1, info);
}

template <DdlaBackend Backend, typename T>
void sytrf_nopiv(const DdlaHandle_t &h, char uplo, int n, T *A, int lda, int &info) {
    ldlt_impl::local<Backend, false, false, false, T>(h, uplo, n, 0, const_cast<T *>(A), lda,
                                                      nullptr, nullptr, 1, info);
}

#define DDLA_INSTANTIATE(BACKEND_, TYPE_) \
    template void sytrf<BACKEND_, TYPE_>(const DdlaHandle_t &, char, int, TYPE_ *, int, int *, int &); \
    template void sytrf_nopiv<BACKEND_, TYPE_>(const DdlaHandle_t &, char, int, TYPE_ *, int, int &)
DDLA_FOR_EACH_BACKEND_TYPE(DDLA_INSTANTIATE)
#undef DDLA_INSTANTIATE

} // namespace ddla
