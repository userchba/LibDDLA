#include <ddla/ddla.h>
#include "ldlt_impl.h"
#include "instantiate.h"

namespace ddla {

template <DdlaBackend Backend, typename T>
void psytrf(const DdlaHandle_t &h, char uplo, int n, T *A, const int *descA, int *ipiv, int &info) {
    ldlt_impl::distributed_factor<Backend, std::is_arithmetic_v<T>, true, T>(h, uplo, n, A, descA, ipiv, info);
}

template <DdlaBackend Backend, typename T>
void psytrf_nopiv(const DdlaHandle_t &h, char uplo, int n, T *A, const int *descA, int &info) {
    ldlt_impl::distributed_factor<Backend, std::is_arithmetic_v<T>, false, T>(h, uplo, n, A, descA, nullptr, info);
}

#define DDLA_INSTANTIATE(BACKEND_, TYPE_) \
    template void psytrf<BACKEND_, TYPE_>(const DdlaHandle_t &, char, int, TYPE_ *, const int *, int *, int &); \
    template void psytrf_nopiv<BACKEND_, TYPE_>(const DdlaHandle_t &, char, int, TYPE_ *, const int *, int &)
DDLA_FOR_EACH_BACKEND_TYPE(DDLA_INSTANTIATE)
#undef DDLA_INSTANTIATE

} // namespace ddla
