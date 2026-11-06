#ifndef DDLA_PGEMMT_IMPL_H
#define DDLA_PGEMMT_IMPL_H

#include <ddla/ddla.h>
#include <cstddef>
#include "BLAS/gemmt_impl.h"

namespace ddla { namespace pgemmt_impl {

// Check is collective. The workspace size counts every buffer; run may be
// reused on one scratch for unchanged geometry, flags and coefficient modes
// without allocating.
template <DdlaBackend Backend, typename T,typename TA=T,typename TB=T>
bool check(const DdlaHandle_t& h, char uplo, char ta, char tb,
    int n, int k, const T& alpha, const TA* A, const int* da,
    const TB* B, const int* db, const T& beta, const T* C, const int* dc);

// Backend memory holds the communicated input panels, transfer scratch,
// batched pointers and temporary diagonal blocks. Host memory holds packing
// metadata and transfer scratch. Only the selected triangle is updated.
template <typename T,typename TA=T,typename TB=T>
void workspace_size(const DdlaHandle_t& h, char uplo, char ta, char tb, bool product, int n, int k,
    const int* da, const int* db, const int* dc,
    std::size_t* backend_bytes, std::size_t* host_bytes);

template <DdlaBackend Backend, typename T,typename TA=T,typename TB=T>
void run(const DdlaHandle_t& h, char uplo, char ta, char tb, bool product,
    int n, int k, const T& alpha, const TA* A, const int* da,
    const TB* B, const int* db, const T& beta, T* C, const int* dc,
    void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes);

}} // namespace ddla::pgemmt_impl
#endif
