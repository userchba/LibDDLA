#include <ddla/ddla.h>
#include "ldlt_impl.h"
#include "instantiate.h"

namespace ddla {

template <DdlaBackend Backend, typename T>
void sytrs(const DdlaHandle_t &h, char uplo, int n, int nrhs, const T *A, int lda, const int *ipiv,
           T *B, int ldb, int &info) {
    ldlt_impl::local<Backend, false, true, true, T>(h, uplo, n, nrhs, const_cast<T *>(A), lda,
                                                    const_cast<int *>(ipiv), B, ldb, info);
}

template <DdlaBackend Backend, typename T>
void sytrs_nopiv(const DdlaHandle_t &h, char uplo, int n, int nrhs, const T *A, int lda, T *B,
                 int ldb, int &info) {
    ldlt_impl::local<Backend, false, false, true, T>(h, uplo, n, nrhs, const_cast<T *>(A), lda,
                                                     nullptr, B, ldb, info);
}

#define DDLA_INSTANTIATE(BACKEND_, TYPE_) \
    template void sytrs<BACKEND_, TYPE_>(const DdlaHandle_t &, char, int, int, const TYPE_ *, int, const int *, TYPE_ *, int, int &); \
    template void sytrs_nopiv<BACKEND_, TYPE_>(const DdlaHandle_t &, char, int, int, const TYPE_ *, int, TYPE_ *, int, int &)
DDLA_FOR_EACH_BACKEND_TYPE(DDLA_INSTANTIATE)
#undef DDLA_INSTANTIATE

} // namespace ddla
