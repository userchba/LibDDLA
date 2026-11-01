#ifndef DDLA_REFLECTOR_BUILD_T_H
#define DDLA_REFLECTOR_BUILD_T_H

#include <ddla/ddla.h>

namespace ddla { namespace detail {

// Compact-WY factor from Gram = V^H V and tau. Forward: H(0)...H(k-1) =
// I - V T V^H, T upper triangular. Backward reverses the product and makes
// T lower triangular. Gram is k x k; only its strict upper triangle is read.
// T has leading dimension ldt >= k: both triangles are written, padding is
// preserved. Inputs and output are disjoint; k <= 0 touches nothing.
// CPU uses host pointers without the handle and is also available for host
// fallback in GPU-only builds. GPU uses backend pointers on the handle stream.
// No allocation or synchronization; GPU shared-memory limits match QR's kernel.
template <DdlaBackend Backend, typename T>
void reflector_build_T(const DdlaHandle_t& handle, int k, const T* gram,
                       const T* tau, T* tmat, int ldt, bool backward);

} } // namespace ddla::detail

#endif // DDLA_REFLECTOR_BUILD_T_H
