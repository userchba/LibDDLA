#ifndef DDLA_PTRTRM_IMPL_H
#define DDLA_PTRTRM_IMPL_H

#include <ddla/ddla.h>
#include <cstddef>

namespace ddla { namespace ptrtrm_impl {

// Public ptrtrm composes check -> workspace size -> allocate -> run.
// Internal callers can size once and reuse caller-owned backend/host scratch.
// Flags are normalized U/L and N/T/C; uplo describes each stored operand.
template <DdlaBackend Backend, typename T>
void ptrtrm_check(const DdlaHandle_t& handle, char uplo_a, char transa, char uplo_b, char transb,
    int n, const T& alpha, const T* A, const int* descA, const T* B, const int* descB,
    const T& beta, const T* C, const int* descC);

// No validation or collectives. Includes the shared pgeadd workspace for
// transposed panels; NN needs only the original panel/batch/diagonal layout.
template <typename T>
void ptrtrm_workspace_size(const DdlaHandle_t& handle, char transa, char transb, int n,
    const int* descA, const int* descB, const int* descC,
    std::size_t* backend_bytes, std::size_t* host_bytes);

// No validation or allocation. Scratch must match the handle, flags and
// descriptor geometry used by workspace_size. Host scratch must be pageable:
// queued pointer-list and redistribution-schedule uploads consume it on return.
template <DdlaBackend Backend, typename T>
void ptrtrm_run(const DdlaHandle_t& handle, char uplo_a, char transa, char uplo_b, char transb,
    int n, const T& alpha, const T* A, const int* descA, const T* B, const int* descB,
    const T& beta, T* C, const int* descC, void* backend_work, std::size_t backend_bytes,
    void* host_work, std::size_t host_bytes);

}} // namespace ddla::ptrtrm_impl
#endif
