#ifndef DDLA_WRITE_MATRIX_H
#define DDLA_WRITE_MATRIX_H

// Public <ddla/ddla.h> must be included before any redeclaration: it
// owns the public default arguments (Backend = default_backend_v), and
// hipcc/clang rejects redeclarations that introduce defaults later.
#include <ddla/ddla.h>
#include "ddla_connector.h"
#include <ddla/ddla_handle_t.h>
#include "workspace.h"
#include <iosfwd>

namespace ddla {

namespace matrix_io_impl {
// Keep the public debugging helper's silent no-op behavior for invalid input.
template<class T>bool write_matrix_check(const T* A,int m,int n,const char* filename) noexcept
{return A!=nullptr&&m>0&&n>0&&filename!=nullptr;}

// Only the GPU path needs staging. The zero CPU result is one supported path
// of this genuinely scratch-using operation, not a separate dummy component.
template<DdlaBackend Backend,class T>WorkspaceSize write_matrix_workspace_size(int m,int n)
{
    detail::WorkspaceLayout layout;
    if constexpr(Backend==DdlaBackend::GPU)
        if(m>0&&n>0)layout.host<T>(detail::workspace_mul(std::size_t(m),std::size_t(n)));
    return layout.size();
}

// The caller opens/prepares the stream and supplies live host objects for the
// GPU query's staging span. A, dimensions and capacity are already checked.
// GPU copies are synchronous, matching the original handle-free utility.
// Stream/file-buffer ownership and standard stream internals belong to caller.
template<DdlaBackend Backend,class T>void write_matrix_run(const T* A,int m,int n,
    std::ostream& output,BufferView<T> host_staging={});
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
