#ifndef DDLA_HANDLE_T_H
#define DDLA_HANDLE_T_H

#include <cstddef>
#include <mpi.h>
#include <ddla/ddla_config.h>

namespace ddla {

// ---------------------------------------------------------------------------
// Backend selection
// ---------------------------------------------------------------------------
enum class DdlaBackend {
    CPU,  ///< CPU-only (host BLAS, MPI communication)
    GPU   ///< GPU (CUDA or HIP, depending on build configuration)
};

/// Compile-time default used by backend-templated compute interfaces.
/// Dual CPU+GPU builds prefer GPU, matching ddla_init(handle).
constexpr DdlaBackend default_backend_v =
#if DDLA_HAS_GPU
    DdlaBackend::GPU;
#elif DDLA_HAS_CPU
    DdlaBackend::CPU;
#else
    // Unreachable: CMake requires at least one backend (CPU, CUDA, or HIP).
    // Kept as a hard error so the header stays self-consistent.
#error "LibDDLA requires at least one backend (CPU or GPU)"
#endif

/// Memory copy direction for ddla_memcpy.
enum class DdlaMemoryCopyKind {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice
};

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------
class DdlaStream;                      // concrete type in private header
using DdlaHandle_t = DdlaStream*;      // ABI: opaque pointer

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

/// Create an uninitialized handle using the compile-time default backend
/// (`default_backend_v`: GPU when compiled with GPU support, else CPU).
void ddla_init(DdlaHandle_t& handle);

/// Create a handle requesting a specific backend.
/// @throws std::runtime_error if the requested backend is unavailable.
void ddla_init(DdlaHandle_t& handle, DdlaBackend requested_backend);

/// Initialize process grid and allocate backend resources.
/// Backend is resolved and validated collectively.
void ddla_set(DdlaHandle_t handle, const MPI_Comm& comm = MPI_COMM_WORLD,
              const char& major = 'R');

/// Initialize with explicit grid dimensions.
void ddla_set(DdlaHandle_t handle, const MPI_Comm& comm,
              const int& nprows, const int& npcols,
              const char& major = 'R');

/// Release all resources; idempotent (safe to call multiple times).
void ddla_destroy(DdlaHandle_t& handle);

// ---------------------------------------------------------------------------
// Backend queries
// ---------------------------------------------------------------------------

/// True if the library was compiled with support for the given backend.
bool ddla_backend_available(DdlaBackend backend);

/// Return the resolved backend for this handle.
/// @throws std::runtime_error if @p handle is null.
DdlaBackend ddla_get_backend(const DdlaHandle_t& handle);

// ---------------------------------------------------------------------------
// Memory and synchronization (vendor-neutral)
// ---------------------------------------------------------------------------

/// Allocate backend-appropriate memory.
///
/// CPU handles allocate host memory (malloc).  GPU handles allocate device
/// memory on the handle's default stream.
///
/// - Returns 1 if @p ptr is null or @p handle is null.
/// - Zero-byte allocation: sets @p *ptr = nullptr and returns 0 (success).
///   No backend call is made.
/// - Nonzero CPU allocation: returns 0 on success, 1 if malloc fails.
/// - Nonzero GPU allocation: returns 0 on success, nonzero on device error.
///
/// @param[out] ptr   Set to the allocated pointer on success.
/// @param bytes      Allocation size in bytes.
/// @param handle     Backend handle (must not be null).
/// @returns 0 on success, nonzero on error.
int ddla_malloc(void** ptr, std::size_t bytes, const DdlaHandle_t& handle);

/// Free backend-appropriate memory.
///
/// - Returns 1 if @p handle is null.
/// - Freeing a null pointer is a valid no-op (returns 0).
/// - CPU handles free host memory; GPU handles free device memory on the
///   handle's default stream.
int ddla_free(void* ptr, const DdlaHandle_t& handle);

/// Copy data between host and device (or within host for CPU handles).
///
/// - Returns 1 if @p handle is null.
/// - Zero-byte copy: returns 0 without dereferencing @p src or @p dst.
/// - Nonzero copy: returns 1 if @p src or @p dst is null, or if @p kind
///   is not one of the three defined DdlaMemoryCopyKind enumerators.
/// - CPU handles perform an ordinary host memcpy (the @p kind is ignored
///   after validation).
/// - GPU handles use the appropriate runtimeMemcpyKind on the handle's
///   default stream.
///
/// CPU handles require host pointers.  GPU handles require pointers
/// allocated in the selected accelerator memory space.  No implicit
/// migration between address spaces is performed.
int ddla_memcpy(void* dst, const void* src, std::size_t bytes,
                DdlaMemoryCopyKind kind, const DdlaHandle_t& handle);

/// Synchronize the handle's default compute stream.
int ddla_synchronize(const DdlaHandle_t& handle);

// ---------------------------------------------------------------------------
// Public accessors (replace direct field access)
// ---------------------------------------------------------------------------

/// Return the duplicated MPI communicator owned by the handle.
MPI_Comm ddla_get_communicator(const DdlaHandle_t& handle);

/// Return the row communicator (processes in the same grid row).
MPI_Comm ddla_get_row_communicator(const DdlaHandle_t& handle);

/// Return the column communicator (processes in the same grid column).
MPI_Comm ddla_get_col_communicator(const DdlaHandle_t& handle);

/// Rank of this process within the handle's communicator.
int ddla_get_rank(const DdlaHandle_t& handle);

/// Size of the handle's communicator.
int ddla_get_size(const DdlaHandle_t& handle);

/// Process-grid row and column coordinates of this rank.
void ddla_get_grid_coords(const DdlaHandle_t& handle,
                          int& myprow, int& mypcol);

/// Number of process rows and columns in the grid.
void ddla_get_grid_dims(const DdlaHandle_t& handle,
                        int& nprows, int& npcols);

/// Convert a rank to (row, col) process-grid coordinates.
void ddla_rank_to_rc(const DdlaHandle_t& handle,
                     int rank, int& row, int& col);

/// Convert (row, col) process-grid coordinates to a rank.
int ddla_rc_to_rank(const DdlaHandle_t& handle, int row, int col);

/// Return the compute stream (GPU: device stream; CPU: nullptr/0).
void* ddla_get_stream(const DdlaHandle_t& handle);

} // namespace ddla

#endif // DDLA_HANDLE_T_H
