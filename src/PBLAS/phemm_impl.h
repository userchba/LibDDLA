#ifndef DDLA_PHEMM_IMPL_H
#define DDLA_PHEMM_IMPL_H

#include <ddla/ddla.h>

#include <cstddef>

namespace ddla { namespace phemm_impl {


// Check is collective; the workspace size counts every temporary; run neither
// validates nor allocates. One scratch can be reused for the same geometry and
// nonzero-alpha calls, with different matrices and coefficients. Set
// hermitian=false for symmetric storage: reflection is a transpose and complex
// diagonals are kept. uploC selects full output F, or triangle U/L.
template <DdlaBackend Backend, typename T,typename TA=T,typename TB=T>
bool phemm_check(const DdlaHandle_t& h, char side, char uplo, int m, int n,
    const T& alpha, const TA* A, const int* da, const TB* B, const int* db,
    const T& beta, const T* C, const int* dc, bool hermitian = true, char uploC = 'F');

// Backend: the panel of A in B's distribution, the broadcast diagonal block,
// the transported piece and the buffer it is shifted or transposed into, then
// the scratch of the reflected panel's pgeadd; host: that pgeadd's.  Nothing
// for alpha == 0 or a one-process call delegated to local BLAS.
template <typename T,typename TA=T,typename TB=T>
void phemm_workspace_size(const DdlaHandle_t& h, char side, int m, int n,
    bool alpha_is_zero, const int* da, const int* db, const int* dc, bool hermitian,
    std::size_t* backend_bytes, std::size_t* host_bytes, char uploC = 'F');

template <DdlaBackend Backend, typename T,typename TA=T,typename TB=T>
void phemm_run(const DdlaHandle_t& h, char side, char uplo,
    int m, int n, const T& alpha, const TA* A, const int* da,
    const TB* B, const int* db, const T& beta, T* C, const int* dc, bool hermitian,
    void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes, char uploC = 'F');

}} // namespace ddla::phemm_impl
#endif
