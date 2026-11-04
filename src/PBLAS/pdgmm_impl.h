#ifndef DDLA_PDGMM_IMPL_H
#define DDLA_PDGMM_IMPL_H
#include <ddla/ddla.h>

namespace ddla { namespace pdgmm_impl {
// Collective validation. The replicated diagonal lives in backend memory.
template <DdlaBackend Backend, typename T, typename U>
void pdgmm_check(const DdlaHandle_t& h, char side, int m, int n,
    const T* A, const int* descA, const U* x, int incx, T* C, const int* descC);

// Local execution only. No allocation, workspace, validation or collectives.
template <DdlaBackend Backend, typename T, typename U>
void pdgmm_run(const DdlaHandle_t& h, char side, int m, int n,
    const T* A, const int* descA, const U* x, int incx, T* C, const int* descC);
}} // namespace ddla::pdgmm_impl
#endif
