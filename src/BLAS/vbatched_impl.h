#ifndef DDLA_VBATCHED_IMPL_H
#define DDLA_VBATCHED_IMPL_H
#include <ddla/ddla.h>
#include "plan_util.h"
#include <cstddef>
namespace ddla { namespace blas_impl {
// Backend bytes of vbatched_dimensions_check: the device summary it reduces
// into, [ max m | max n | max k | first bad batch + 1 ] (4 ints). gemmVbatched2s
// checks its two stages one after the other on the same bytes. The runs need
// no scratch; an empty batch needs nothing.
inline void gemmVbatched_workspace_size(const DdlaHandle_t& handle, int batch_count, std::size_t* backend_bytes)
{
    (void)handle;
    *backend_bytes = batch_count > 0 ? detail::padded_bytes<int>(4) : 0;
}
// Reads the device dimension arrays back: rejects a negative size or a short
// leading dimension (naming the first bad batch) and returns the maxima the
// run launches with. Synchronizes the handle.
void vbatched_dimensions_check(const DdlaHandle_t&,char,char,int*,int*,int*,int*,int*,int*,int,
                                             void* backend_work,std::size_t backend_bytes,int& maximum_m,int& maximum_n,int& maximum_k);
template <DdlaBackend Backend, typename T>
void gemmVbatched_check(
    const DdlaHandle_t& handle,
    char transA, char transB,
    int* d_m, int* d_n, int* d_k,
    T alpha, const T* const* d_A_array, int* d_lda,
    const T* const* d_B_array, int* d_ldb,
    T beta, T** d_C_array, int* d_ldc,
    int batch_count);
template <DdlaBackend Backend, typename T>
void gemmVbatched_run(
    const DdlaHandle_t& handle, int maximum_m, int maximum_n, int maximum_k,
    char transA, char transB,
    int* d_m, int* d_n, int* d_k,
    T alpha, const T* const* d_A_array, int* d_lda,
    const T* const* d_B_array, int* d_ldb,
    T beta, T** d_C_array, int* d_ldc,
    int batch_count);
template <DdlaBackend Backend, typename T>
void gemmVbatched2s_check(
    const DdlaHandle_t& handle,
    char transA_0, char transB_0,
    int* d_m_0, int* d_n_0, int* d_k_0,
    T alpha_0, const T* const* d_A_array_0, int* d_lda_0,
    const T* const* d_B_array_0, int* d_ldb_0,
    T beta_0, T** d_C_array_0, int* d_ldc_0,
    char transA_1, char transB_1,
    int* d_m_1, int* d_n_1, int* d_k_1,
    T alpha_1, const T* const* d_AB_array_1,
    int* d_lda_1, int* d_ldb_1,
    T beta_1, T** d_C_array_1, int* d_ldc_1,
    bool C0_left, int batch_count,
    const int* segment_sizes, int segment_count);
template <DdlaBackend Backend, typename T>
void gemmVbatched2s_run(
    const DdlaHandle_t& handle, int maximum_m_0, int maximum_n_0, int maximum_k_0, int maximum_m_1, int maximum_n_1, int maximum_k_1,
    char transA_0, char transB_0,
    int* d_m_0, int* d_n_0, int* d_k_0,
    T alpha_0, const T* const* d_A_array_0, int* d_lda_0,
    const T* const* d_B_array_0, int* d_ldb_0,
    T beta_0, T** d_C_array_0, int* d_ldc_0,
    char transA_1, char transB_1,
    int* d_m_1, int* d_n_1, int* d_k_1,
    T alpha_1, const T* const* d_AB_array_1,
    int* d_lda_1, int* d_ldb_1,
    T beta_1, T** d_C_array_1, int* d_ldc_1,
    bool C0_left, int batch_count,
    const int* segment_sizes, int segment_count);
}} // namespace ddla::blas_impl
#endif
