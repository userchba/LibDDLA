#include <ddla/ddla.h>
#include "ldlt_impl.h"

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

template void sytrf<DdlaBackend::CPU, float>(const DdlaHandle_t &, char, int, float *, int, int *,
                                             int &);
template void sytrf<DdlaBackend::CPU, double>(const DdlaHandle_t &, char, int, double *, int, int *,
                                              int &);
template void sytrf<DdlaBackend::CPU, std::complex<float>>(const DdlaHandle_t &, char, int,
                                                           std::complex<float> *, int, int *,
                                                           int &);
template void sytrf<DdlaBackend::CPU, std::complex<double>>(const DdlaHandle_t &, char, int,
                                                            std::complex<double> *, int, int *,
                                                            int &);
#if DDLA_HAS_GPU
template void sytrf<DdlaBackend::GPU, float>(const DdlaHandle_t &, char, int, float *, int, int *,
                                             int &);
template void sytrf<DdlaBackend::GPU, double>(const DdlaHandle_t &, char, int, double *, int, int *,
                                              int &);
template void sytrf<DdlaBackend::GPU, std::complex<float>>(const DdlaHandle_t &, char, int,
                                                           std::complex<float> *, int, int *,
                                                           int &);
template void sytrf<DdlaBackend::GPU, std::complex<double>>(const DdlaHandle_t &, char, int,
                                                            std::complex<double> *, int, int *,
                                                            int &);
#endif

template void sytrf_nopiv<DdlaBackend::CPU, float>(const DdlaHandle_t &, char, int, float *, int,
                                                   int &);
template void sytrf_nopiv<DdlaBackend::CPU, double>(const DdlaHandle_t &, char, int, double *, int,
                                                    int &);
template void sytrf_nopiv<DdlaBackend::CPU, std::complex<float>>(const DdlaHandle_t &, char, int,
                                                                 std::complex<float> *, int, int &);
template void sytrf_nopiv<DdlaBackend::CPU, std::complex<double>>(const DdlaHandle_t &, char, int,
                                                                  std::complex<double> *, int,
                                                                  int &);
#if DDLA_HAS_GPU
template void sytrf_nopiv<DdlaBackend::GPU, float>(const DdlaHandle_t &, char, int, float *, int,
                                                   int &);
template void sytrf_nopiv<DdlaBackend::GPU, double>(const DdlaHandle_t &, char, int, double *, int,
                                                    int &);
template void sytrf_nopiv<DdlaBackend::GPU, std::complex<float>>(const DdlaHandle_t &, char, int,
                                                                 std::complex<float> *, int, int &);
template void sytrf_nopiv<DdlaBackend::GPU, std::complex<double>>(const DdlaHandle_t &, char, int,
                                                                  std::complex<double> *, int,
                                                                  int &);
#endif

} // namespace ddla
