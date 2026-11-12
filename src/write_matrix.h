#ifndef DDLA_WRITE_MATRIX_H
#define DDLA_WRITE_MATRIX_H

// Public scalar declarations and private backend overloads share one
// declaration boundary; this header and all workspace helpers stay private.
#include "backend_api.h"
#include "ddla_connector.h"
#include <ddla/ddla_handle_t.h>
#include "plan_util.h"
#include <cstddef>
#include <iosfwd>

namespace ddla {

namespace matrix_io_impl {
// Keep the public debugging helper's silent no-op behavior for invalid input.
template<class T>bool write_matrix_check(const T* A,int m,int n,const char* filename) noexcept
{return A!=nullptr&&m>0&&n>0&&filename!=nullptr;}

// Host bytes of write_matrix_run: on the GPU the host copy of A (m x n); the
// CPU path reads A in place and needs none.
template<DdlaBackend Backend,class T>void write_matrix_workspace_size(int m,int n,std::size_t* host_bytes)
{
    *host_bytes=Backend==DdlaBackend::GPU&&m>0&&n>0?detail::padded_bytes<T>(std::size_t(m)*std::size_t(n)):0;
}

// The caller opens the stream; A and the dimensions are already checked. The
// GPU copy is synchronous, matching the original handle-free utility.
template<DdlaBackend Backend,class T>void write_matrix_run(const T* A,int m,int n,
    std::ostream& output,void* host_work,std::size_t host_bytes);
} // namespace matrix_io_impl

/**
 * @brief Dump a column-major m x n matrix to a text file (debugging aid).
 *
 * Backend-templated in the same shape as gemm/scal/geam, but here the
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
void write_matrix(const T* A, int m, int n, const char* filename);

} // namespace ddla

#endif // DDLA_WRITE_MATRIX_H
