#ifndef DDLA_WRITE_MATRIX_H
#define DDLA_WRITE_MATRIX_H

// Public <ddla/ddla.h> must be included before any redeclaration: it
// owns the public default arguments (Backend = default_backend_v), and
// hipcc/clang rejects redeclarations that introduce defaults later.
#include <ddla/ddla.h>
#include "ddla_connector.h"
#include <ddla/ddla_handle_t.h>

namespace ddla {

/**
 * @brief Dump a column-major m x n matrix to a text file (debugging aid).
 *
 * Backend-templated in the same shape as gemm/scal/omatcopy, but here the
 * template argument describes *where the pointer lives*, not which BLAS to
 * call: `write_matrix<DdlaBackend::CPU>` reads `A` directly as host memory,
 * while `write_matrix<DdlaBackend::GPU>` first stages the matrix back to the
 * host with a synchronous device-to-host copy and then writes it. That makes
 * it usable for dumping a device buffer without the caller hand-rolling the
 * staging copy.
 *
 * Real types are written as plain numbers; complex types as "(re,im)".
 * Values whose magnitude is below 1e-10 are flushed to 0 so that the output
 * of a run is diffable -- this matches the behaviour of the original
 * complex<double>-only write_matrix this replaces.
 *
 * Supported T: float, double, std::complex<float>, std::complex<double>.
 */
template <DdlaBackend Backend, typename T>
void write_matrix(const T* A, const int& m, const int& n, const char* filename);

} // namespace ddla

#endif // DDLA_WRITE_MATRIX_H
