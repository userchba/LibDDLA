#include <ddla/ddla.h>
#include "ldlt_impl.h"

namespace ddla {

template <DdlaBackend Backend, typename T>
void psytrf(const DdlaHandle_t &h, char uplo, int n, T *A, const int *descA, int *ipiv, int &info) {
    ldlt_impl::distributed<Backend, false, true, false, T>(
        h, uplo, n, 0, const_cast<T *>(A), descA, const_cast<int *>(ipiv), nullptr, nullptr, info);
}

template <DdlaBackend Backend, typename T>
void psytrf_nopiv(const DdlaHandle_t &h, char uplo, int n, T *A, const int *descA, int &info) {
    ldlt_impl::distributed<Backend, false, false, false, T>(h, uplo, n, 0, const_cast<T *>(A),
                                                            descA, nullptr, nullptr, nullptr, info);
}

template void psytrf<DdlaBackend::CPU, float>(const DdlaHandle_t &, char, int, float *, const int *,
                                              int *, int &);
template void psytrf<DdlaBackend::CPU, double>(const DdlaHandle_t &, char, int, double *,
                                               const int *, int *, int &);
template void psytrf<DdlaBackend::CPU, std::complex<float>>(const DdlaHandle_t &, char, int,
                                                            std::complex<float> *, const int *,
                                                            int *, int &);
template void psytrf<DdlaBackend::CPU, std::complex<double>>(const DdlaHandle_t &, char, int,
                                                             std::complex<double> *, const int *,
                                                             int *, int &);
#if DDLA_HAS_GPU
template void psytrf<DdlaBackend::GPU, float>(const DdlaHandle_t &, char, int, float *, const int *,
                                              int *, int &);
template void psytrf<DdlaBackend::GPU, double>(const DdlaHandle_t &, char, int, double *,
                                               const int *, int *, int &);
template void psytrf<DdlaBackend::GPU, std::complex<float>>(const DdlaHandle_t &, char, int,
                                                            std::complex<float> *, const int *,
                                                            int *, int &);
template void psytrf<DdlaBackend::GPU, std::complex<double>>(const DdlaHandle_t &, char, int,
                                                             std::complex<double> *, const int *,
                                                             int *, int &);
#endif

template void psytrf_nopiv<DdlaBackend::CPU, float>(const DdlaHandle_t &, char, int, float *,
                                                    const int *, int &);
template void psytrf_nopiv<DdlaBackend::CPU, double>(const DdlaHandle_t &, char, int, double *,
                                                     const int *, int &);
template void psytrf_nopiv<DdlaBackend::CPU, std::complex<float>>(const DdlaHandle_t &, char, int,
                                                                  std::complex<float> *,
                                                                  const int *, int &);
template void psytrf_nopiv<DdlaBackend::CPU, std::complex<double>>(const DdlaHandle_t &, char, int,
                                                                   std::complex<double> *,
                                                                   const int *, int &);
#if DDLA_HAS_GPU
template void psytrf_nopiv<DdlaBackend::GPU, float>(const DdlaHandle_t &, char, int, float *,
                                                    const int *, int &);
template void psytrf_nopiv<DdlaBackend::GPU, double>(const DdlaHandle_t &, char, int, double *,
                                                     const int *, int &);
template void psytrf_nopiv<DdlaBackend::GPU, std::complex<float>>(const DdlaHandle_t &, char, int,
                                                                  std::complex<float> *,
                                                                  const int *, int &);
template void psytrf_nopiv<DdlaBackend::GPU, std::complex<double>>(const DdlaHandle_t &, char, int,
                                                                   std::complex<double> *,
                                                                   const int *, int &);
#endif

} // namespace ddla
