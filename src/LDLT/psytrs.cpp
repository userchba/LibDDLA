#include <ddla/ddla.h>
#include "ldlt_impl.h"
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

template void psytrs<DdlaBackend::CPU, float>(const DdlaHandle_t &, char, int, int, const float *,
                                              const int *, const int *, float *, const int *,
                                              int &);
template void psytrs<DdlaBackend::CPU, double>(const DdlaHandle_t &, char, int, int, const double *,
                                               const int *, const int *, double *, const int *,
                                               int &);
template void psytrs<DdlaBackend::CPU, std::complex<float>>(const DdlaHandle_t &, char, int, int,
                                                            const std::complex<float> *,
                                                            const int *, const int *,
                                                            std::complex<float> *, const int *,
                                                            int &);
template void psytrs<DdlaBackend::CPU, std::complex<double>>(const DdlaHandle_t &, char, int, int,
                                                             const std::complex<double> *,
                                                             const int *, const int *,
                                                             std::complex<double> *, const int *,
                                                             int &);
#if DDLA_HAS_GPU
template void psytrs<DdlaBackend::GPU, float>(const DdlaHandle_t &, char, int, int, const float *,
                                              const int *, const int *, float *, const int *,
                                              int &);
template void psytrs<DdlaBackend::GPU, double>(const DdlaHandle_t &, char, int, int, const double *,
                                               const int *, const int *, double *, const int *,
                                               int &);
template void psytrs<DdlaBackend::GPU, std::complex<float>>(const DdlaHandle_t &, char, int, int,
                                                            const std::complex<float> *,
                                                            const int *, const int *,
                                                            std::complex<float> *, const int *,
                                                            int &);
template void psytrs<DdlaBackend::GPU, std::complex<double>>(const DdlaHandle_t &, char, int, int,
                                                             const std::complex<double> *,
                                                             const int *, const int *,
                                                             std::complex<double> *, const int *,
                                                             int &);
#endif

template void psytrs_nopiv<DdlaBackend::CPU, float>(const DdlaHandle_t &, char, int, int,
                                                    const float *, const int *, float *,
                                                    const int *, int &);
template void psytrs_nopiv<DdlaBackend::CPU, double>(const DdlaHandle_t &, char, int, int,
                                                     const double *, const int *, double *,
                                                     const int *, int &);
template void psytrs_nopiv<DdlaBackend::CPU, std::complex<float>>(const DdlaHandle_t &, char, int,
                                                                  int, const std::complex<float> *,
                                                                  const int *,
                                                                  std::complex<float> *,
                                                                  const int *, int &);
template void
psytrs_nopiv<DdlaBackend::CPU, std::complex<double>>(const DdlaHandle_t &, char, int, int,
                                                     const std::complex<double> *, const int *,
                                                     std::complex<double> *, const int *, int &);
#if DDLA_HAS_GPU
template void psytrs_nopiv<DdlaBackend::GPU, float>(const DdlaHandle_t &, char, int, int,
                                                    const float *, const int *, float *,
                                                    const int *, int &);
template void psytrs_nopiv<DdlaBackend::GPU, double>(const DdlaHandle_t &, char, int, int,
                                                     const double *, const int *, double *,
                                                     const int *, int &);
template void psytrs_nopiv<DdlaBackend::GPU, std::complex<float>>(const DdlaHandle_t &, char, int,
                                                                  int, const std::complex<float> *,
                                                                  const int *,
                                                                  std::complex<float> *,
                                                                  const int *, int &);
template void
psytrs_nopiv<DdlaBackend::GPU, std::complex<double>>(const DdlaHandle_t &, char, int, int,
                                                     const std::complex<double> *, const int *,
                                                     std::complex<double> *, const int *, int &);
#endif

} // namespace ddla
