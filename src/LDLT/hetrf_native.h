#ifndef DDLA_HETRF_NATIVE_H
#define DDLA_HETRF_NATIVE_H
#include <ddla/ddla.h>

namespace ddla::ldlt_impl
{
struct PivotSummary
{
    double diagonal, maximum;
    int index, reserved; // Initialize every byte downloaded to the host.
};
static_assert(sizeof(PivotSummary) == 2 * sizeof(double) + 2 * sizeof(int));
template <class T>
void summarize_column(const DdlaHandle_t& h, T* column, int first, int count, int diagonal,
                      PivotSummary* output);
template <class T>
void hetrf_nopiv_gpu(const DdlaHandle_t& h, char uplo, int n, T* a, int lda, int& info);
template <bool Pivot, class T>
void phetrf_gpu(const DdlaHandle_t& h, char uplo, int n, T* a, const int* desc, int* piv,
                int& info);

// Hermitian 1x1 pivots: local leaves up to256, distributed diagonal tiles up
// to64.
template <class T>
void nopiv_diagonal(const DdlaHandle_t& h, char uplo, int n, T* a, int lda, int* info);
template <class T>
void divide_panel(const DdlaHandle_t& h, int rows, int cols, const T* diagonal, int ldd,
                  const T* weighted, int ldw, T* factors, int ldf);
// Updated column(s) -> D and normalized factors, retaining weighted columns.
template <class T>
void pivot_columns(const DdlaHandle_t& h, int first, int count, int current, int neighbor, int step,
                   int swap_a, int swap_b, const T* c0, const T* c1, T* l0, T* l1, T* w0, T* w1);
// Same ownership, possibly different pitches; only the caller's triangle
// changes. Optional local offsets describe a bounded source tile. subtract
// accumulates a GEMM tile into that triangle without reading the unused output
// entries.
template <class T>
void copy_stored(const DdlaHandle_t& h, char uplo, int n, const int* desc, const T* src, int lds,
                 T* dst, int ldd, int row0 = 0, int col0 = 0, int rows = -1, int cols = -1,
                 bool subtract = false);
} // namespace ddla::ldlt_impl
#endif
