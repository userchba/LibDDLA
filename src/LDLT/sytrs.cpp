#include <ddla/ddla.h>
#include "ldlt_impl.h"

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

template void sytrs<DdlaBackend::CPU, float>(const DdlaHandle_t &, char, int, int, const float *,
                                             int, const int *, float *, int, int &);
template void sytrs<DdlaBackend::CPU, double>(const DdlaHandle_t &, char, int, int, const double *,
                                              int, const int *, double *, int, int &);
template void sytrs<DdlaBackend::CPU, std::complex<float>>(const DdlaHandle_t &, char, int, int,
                                                           const std::complex<float> *, int,
                                                           const int *, std::complex<float> *, int,
                                                           int &);
template void sytrs<DdlaBackend::CPU, std::complex<double>>(const DdlaHandle_t &, char, int, int,
                                                            const std::complex<double> *, int,
                                                            const int *, std::complex<double> *,
                                                            int, int &);
#if DDLA_HAS_GPU
template void sytrs<DdlaBackend::GPU, float>(const DdlaHandle_t &, char, int, int, const float *,
                                             int, const int *, float *, int, int &);
template void sytrs<DdlaBackend::GPU, double>(const DdlaHandle_t &, char, int, int, const double *,
                                              int, const int *, double *, int, int &);
template void sytrs<DdlaBackend::GPU, std::complex<float>>(const DdlaHandle_t &, char, int, int,
                                                           const std::complex<float> *, int,
                                                           const int *, std::complex<float> *, int,
                                                           int &);
template void sytrs<DdlaBackend::GPU, std::complex<double>>(const DdlaHandle_t &, char, int, int,
                                                            const std::complex<double> *, int,
                                                            const int *, std::complex<double> *,
                                                            int, int &);
#endif

template void sytrs_nopiv<DdlaBackend::CPU, float>(const DdlaHandle_t &, char, int, int,
                                                   const float *, int, float *, int, int &);
template void sytrs_nopiv<DdlaBackend::CPU, double>(const DdlaHandle_t &, char, int, int,
                                                    const double *, int, double *, int, int &);
template void sytrs_nopiv<DdlaBackend::CPU, std::complex<float>>(const DdlaHandle_t &, char, int,
                                                                 int, const std::complex<float> *,
                                                                 int, std::complex<float> *, int,
                                                                 int &);
template void sytrs_nopiv<DdlaBackend::CPU, std::complex<double>>(const DdlaHandle_t &, char, int,
                                                                  int, const std::complex<double> *,
                                                                  int, std::complex<double> *, int,
                                                                  int &);
#if DDLA_HAS_GPU
template void sytrs_nopiv<DdlaBackend::GPU, float>(const DdlaHandle_t &, char, int, int,
                                                   const float *, int, float *, int, int &);
template void sytrs_nopiv<DdlaBackend::GPU, double>(const DdlaHandle_t &, char, int, int,
                                                    const double *, int, double *, int, int &);
template void sytrs_nopiv<DdlaBackend::GPU, std::complex<float>>(const DdlaHandle_t &, char, int,
                                                                 int, const std::complex<float> *,
                                                                 int, std::complex<float> *, int,
                                                                 int &);
template void sytrs_nopiv<DdlaBackend::GPU, std::complex<double>>(const DdlaHandle_t &, char, int,
                                                                  int, const std::complex<double> *,
                                                                  int, std::complex<double> *, int,
                                                                  int &);
#endif

} // namespace ddla
