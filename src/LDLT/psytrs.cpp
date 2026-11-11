#include <ddla/ddla.h>
#include "ldlt_impl.h"
#include "instantiate.h"
#include <type_traits>

namespace ddla {

// Real SY solves use the distributed HE solves (see psytrf.cpp).
template <DdlaBackend Backend, typename T>
void psytrs(const DdlaHandle_t &h, char uplo, int n, int nrhs, const T *A, const int *descA,
            const int *ipiv, T *B, const int *descB, int &info) {
    if constexpr (std::is_floating_point_v<T>) {
        phetrs<Backend, T>(h, uplo, n, nrhs, A, descA, ipiv, B, descB, info);
    } else {
        ldlt_impl::distributed<Backend, false, true, true, T>(
            h, uplo, n, nrhs, const_cast<T *>(A), descA, const_cast<int *>(ipiv), B, descB, info);
    }
}

template <DdlaBackend Backend, typename T>
void psytrs_nopiv(const DdlaHandle_t &h, char uplo, int n, int nrhs, const T *A, const int *descA,
                  T *B, const int *descB, int &info) {
    if constexpr (std::is_floating_point_v<T>) {
        phetrs_nopiv<Backend, T>(h, uplo, n, nrhs, A, descA, B, descB, info);
    } else {
        ldlt_impl::distributed<Backend, false, false, true, T>(h, uplo, n, nrhs, const_cast<T *>(A),
                                                               descA, nullptr, B, descB, info);
    }
}

#define DDLA_INSTANTIATE(BACKEND_, TYPE_) \
    template void psytrs<BACKEND_, TYPE_>(const DdlaHandle_t &, char, int, int, const TYPE_ *, const int *, const int *, TYPE_ *, const int *, int &); \
    template void psytrs_nopiv<BACKEND_, TYPE_>(const DdlaHandle_t &, char, int, int, const TYPE_ *, const int *, TYPE_ *, const int *, int &)
DDLA_FOR_EACH_BACKEND_TYPE(DDLA_INSTANTIATE)
#undef DDLA_INSTANTIATE

} // namespace ddla
