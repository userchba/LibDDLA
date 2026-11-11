#include <ddla/ddla.h>
#include "ldlt_impl.h"
#include "instantiate.h"

namespace ddla {

template <DdlaBackend Backend, typename T>
void psytrs(const DdlaHandle_t &h, char uplo, int n, int nrhs, const T *A, const int *descA,
            const int *ipiv, T *B, const int *descB, int &info) {
    ldlt_impl::distributed_solve<Backend, std::is_arithmetic_v<T>, true, T>(h, uplo, n, nrhs, A, descA, ipiv, B, descB, info);
}

template <DdlaBackend Backend, typename T>
void psytrs_nopiv(const DdlaHandle_t &h, char uplo, int n, int nrhs, const T *A, const int *descA,
                  T *B, const int *descB, int &info) {
    ldlt_impl::distributed_solve<Backend, std::is_arithmetic_v<T>, false, T>(h, uplo, n, nrhs, A, descA, nullptr, B, descB, info);
}

#define DDLA_INSTANTIATE(BACKEND_, TYPE_) \
    template void psytrs<BACKEND_, TYPE_>(const DdlaHandle_t &, char, int, int, const TYPE_ *, const int *, const int *, TYPE_ *, const int *, int &); \
    template void psytrs_nopiv<BACKEND_, TYPE_>(const DdlaHandle_t &, char, int, int, const TYPE_ *, const int *, TYPE_ *, const int *, int &)
DDLA_FOR_EACH_BACKEND_TYPE(DDLA_INSTANTIATE)
#undef DDLA_INSTANTIATE

} // namespace ddla
