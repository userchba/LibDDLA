#ifndef DDLA_HETRF_GPU_H
#define DDLA_HETRF_GPU_H

#include <ddla/ddla.h>
#include <complex>

namespace ddla::ldlt_impl
{

// Local Hermitian Bunch-Kaufman factorization. Matrix arithmetic stays on the
// GPU; the host receives only scalar pivot decisions and owns LAPACK IPIV/INFO.
void hetrf_gpu(const DdlaHandle_t& h, char uplo, int n, std::complex<double>* a, int lda, int* ipiv,
               int& info);

void hetrf_conj_copy(const DdlaHandle_t& h, int n, const std::complex<double>* src, int incs,
                     std::complex<double>* dst, int incd);
void hetrf_scale_2x2(const DdlaHandle_t& h, char uplo, int rows, const std::complex<double>* w,
                     int ldw, std::complex<double>* a, int lda);

// Fused small-panel engine; status contains factored width and 1-based INFO.
void hetrf_panel_factor(const DdlaHandle_t& h, char uplo, int n, int nb, std::complex<double>* a,
                        int lda, int* piv, std::complex<double>* w, int ldw, int* status,
                        int offset);
void hetrf_panel_undo(const DdlaHandle_t& h, char uplo, int n, int kb, std::complex<double>* a,
                      int lda, int* piv, int offset);

} // namespace ddla::ldlt_impl
#endif
