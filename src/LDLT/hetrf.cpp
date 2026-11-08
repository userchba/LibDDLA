#include <ddla/ddla.h>
#include "ldlt_impl.h"
#include "hetrf_gpu.h"
#include "hetrf_native.h"

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

template void hetrf<DdlaBackend::CPU, float>(const DdlaHandle_t &, char, int, float *, int, int *,
                                             int &);
template void hetrf<DdlaBackend::CPU, double>(const DdlaHandle_t &, char, int, double *, int, int *,
                                              int &);
template void hetrf<DdlaBackend::CPU, std::complex<float>>(const DdlaHandle_t &, char, int,
                                                           std::complex<float> *, int, int *,
                                                           int &);
template void hetrf<DdlaBackend::CPU, std::complex<double>>(const DdlaHandle_t &, char, int,
                                                            std::complex<double> *, int, int *,
                                                            int &);
#if DDLA_HAS_GPU
template void hetrf<DdlaBackend::GPU, float>(const DdlaHandle_t &, char, int, float *, int, int *,
                                             int &);
template void hetrf<DdlaBackend::GPU, double>(const DdlaHandle_t &, char, int, double *, int, int *,
                                              int &);
template void hetrf<DdlaBackend::GPU, std::complex<float>>(const DdlaHandle_t &, char, int,
                                                           std::complex<float> *, int, int *,
                                                           int &);
template void hetrf<DdlaBackend::GPU, std::complex<double>>(const DdlaHandle_t &, char, int,
                                                            std::complex<double> *, int, int *,
                                                            int &);
#endif

template void hetrf_nopiv<DdlaBackend::CPU, float>(const DdlaHandle_t &, char, int, float *, int,
                                                   int &);
template void hetrf_nopiv<DdlaBackend::CPU, double>(const DdlaHandle_t &, char, int, double *, int,
                                                    int &);
template void hetrf_nopiv<DdlaBackend::CPU, std::complex<float>>(const DdlaHandle_t &, char, int,
                                                                 std::complex<float> *, int, int &);
template void hetrf_nopiv<DdlaBackend::CPU, std::complex<double>>(const DdlaHandle_t &, char, int,
                                                                  std::complex<double> *, int,
                                                                  int &);
#if DDLA_HAS_GPU
template void hetrf_nopiv<DdlaBackend::GPU, float>(const DdlaHandle_t &, char, int, float *, int,
                                                   int &);
template void hetrf_nopiv<DdlaBackend::GPU, double>(const DdlaHandle_t &, char, int, double *, int,
                                                    int &);
template void hetrf_nopiv<DdlaBackend::GPU, std::complex<float>>(const DdlaHandle_t &, char, int,
                                                                 std::complex<float> *, int, int &);
template void hetrf_nopiv<DdlaBackend::GPU, std::complex<double>>(const DdlaHandle_t &, char, int,
                                                                  std::complex<double> *, int,
                                                                  int &);
#endif

} // namespace ddla
