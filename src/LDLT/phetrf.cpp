#include <ddla/ddla.h>
#include "ldlt_impl.h"
#include "hetrf_native.h"

namespace ddla {

template <DdlaBackend Backend, typename T>
void phetrf(const DdlaHandle_t &h, char uplo, int n, T *A, const int *descA, int *ipiv, int &info) {
#if DDLA_HAS_GPU
    if constexpr (Backend == DdlaBackend::GPU) {
        ldlt_impl::phetrf_gpu<true, T>(h, uplo, n, A, descA, ipiv, info);
        return;
    }
#endif
    ldlt_impl::distributed<Backend, true, true, false, T>(
        h, uplo, n, 0, const_cast<T *>(A), descA, const_cast<int *>(ipiv), nullptr, nullptr, info);
}

template <DdlaBackend Backend, typename T>
void phetrf_nopiv(const DdlaHandle_t &h, char uplo, int n, T *A, const int *descA, int &info) {
#if DDLA_HAS_GPU
    if constexpr (Backend == DdlaBackend::GPU) {
        ldlt_impl::phetrf_gpu<false, T>(h, uplo, n, A, descA, nullptr, info);
        return;
    }
#endif
    ldlt_impl::distributed<Backend, true, false, false, T>(h, uplo, n, 0, const_cast<T *>(A), descA,
                                                           nullptr, nullptr, nullptr, info);
}

template void phetrf<DdlaBackend::CPU, float>(const DdlaHandle_t &, char, int, float *, const int *,
                                              int *, int &);
template void phetrf<DdlaBackend::CPU, double>(const DdlaHandle_t &, char, int, double *,
                                               const int *, int *, int &);
template void phetrf<DdlaBackend::CPU, std::complex<float>>(const DdlaHandle_t &, char, int,
                                                            std::complex<float> *, const int *,
                                                            int *, int &);
template void phetrf<DdlaBackend::CPU, std::complex<double>>(const DdlaHandle_t &, char, int,
                                                             std::complex<double> *, const int *,
                                                             int *, int &);
#if DDLA_HAS_GPU
template void phetrf<DdlaBackend::GPU, float>(const DdlaHandle_t &, char, int, float *, const int *,
                                              int *, int &);
template void phetrf<DdlaBackend::GPU, double>(const DdlaHandle_t &, char, int, double *,
                                               const int *, int *, int &);
template void phetrf<DdlaBackend::GPU, std::complex<float>>(const DdlaHandle_t &, char, int,
                                                            std::complex<float> *, const int *,
                                                            int *, int &);
template void phetrf<DdlaBackend::GPU, std::complex<double>>(const DdlaHandle_t &, char, int,
                                                             std::complex<double> *, const int *,
                                                             int *, int &);
#endif

template void phetrf_nopiv<DdlaBackend::CPU, float>(const DdlaHandle_t &, char, int, float *,
                                                    const int *, int &);
template void phetrf_nopiv<DdlaBackend::CPU, double>(const DdlaHandle_t &, char, int, double *,
                                                     const int *, int &);
template void phetrf_nopiv<DdlaBackend::CPU, std::complex<float>>(const DdlaHandle_t &, char, int,
                                                                  std::complex<float> *,
                                                                  const int *, int &);
template void phetrf_nopiv<DdlaBackend::CPU, std::complex<double>>(const DdlaHandle_t &, char, int,
                                                                   std::complex<double> *,
                                                                   const int *, int &);
#if DDLA_HAS_GPU
template void phetrf_nopiv<DdlaBackend::GPU, float>(const DdlaHandle_t &, char, int, float *,
                                                    const int *, int &);
template void phetrf_nopiv<DdlaBackend::GPU, double>(const DdlaHandle_t &, char, int, double *,
                                                     const int *, int &);
template void phetrf_nopiv<DdlaBackend::GPU, std::complex<float>>(const DdlaHandle_t &, char, int,
                                                                  std::complex<float> *,
                                                                  const int *, int &);
template void phetrf_nopiv<DdlaBackend::GPU, std::complex<double>>(const DdlaHandle_t &, char, int,
                                                                   std::complex<double> *,
                                                                   const int *, int &);
#endif

} // namespace ddla
