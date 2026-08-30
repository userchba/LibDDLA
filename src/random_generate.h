#ifndef DDLA_RANDOM_GENERATE_H
#define DDLA_RANDOM_GENERATE_H

// Public <ddla/ddla.h> must be included before any redeclaration: it
// owns the public default arguments (Backend = default_backend_v), and
// hipcc/clang rejects redeclarations that introduce defaults later.
#include <ddla/ddla.h>
#include "ddla_connector.h"
#include <ddla/ddla_handle_t.h>

namespace ddla {

/**
 * @brief Fill a buffer with independent uniform random values in [0, 1).
 *
 * Backend-templated in the same shape as gemm/scal/write_matrix.  The
 * template argument describes where the buffer lives:
 * `random_generate<DdlaBackend::CPU>` writes `data` directly as host memory,
 * while `random_generate<DdlaBackend::GPU>` generates the values on the
 * accelerator with the vendor RNG (curand on CUDA, hiprand on HIP).
 *
 * The CPU path uses std::mt19937_64 seeded from the system clock, so the
 * produced values are uniform in [0, 1) but are not bit-for-bit reproducible
 * with the GPU path.  Complex types are filled with 2N independent real
 * draws (real part first, then imaginary part).
 *
 * A non-positive `lengthOfData` is a safe no-op; `data` may be nullptr then.
 *
 * Supported T: float, double, std::complex<float>, std::complex<double>.
 */
template <DdlaBackend Backend, typename T>
void random_generate(T* data, const int64_t& lengthOfData);

} // namespace ddla

#endif // DDLA_RANDOM_GENERATE_H
