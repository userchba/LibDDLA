#ifndef DDLA_HANDLE_T_H
#define DDLA_HANDLE_T_H

#include <cstddef>
#include <stdexcept>
#include <string>
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
/// Dual CPU+GPU builds prefer GPU, matching ddlaInit(handle).
inline constexpr DdlaBackend default_backend_v =
#if DDLA_HAS_GPU
    DdlaBackend::GPU;
#elif DDLA_HAS_CPU
    DdlaBackend::CPU;
#else
    // Unreachable: CMake requires at least one backend (CPU, CUDA, or HIP).
    // Kept as a hard error so the header stays self-consistent.
#error "LibDDLA requires at least one backend (CPU or GPU)"
#endif

/// Status codes returned by the public API functions.
enum class ddlaStatus_t {
    DDLA_STATUS_SUCCESS = 0,        ///< Operation completed successfully.
    DDLA_STATUS_INVALID_HANDLE,     ///< Null or invalid handle.
    DDLA_STATUS_INVALID_VALUE,      ///< Null pointer, invalid enum value, etc.
    DDLA_STATUS_ALLOC_FAILED,       ///< Memory allocation failed.
    DDLA_STATUS_BACKEND_UNAVAILABLE,///< Requested backend not compiled in.
    DDLA_STATUS_INTERNAL_ERROR      ///< Vendor library / device error.
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
ddlaStatus_t ddlaInit(DdlaHandle_t& handle);

/// Create a handle requesting a specific backend.
/// @throws std::runtime_error if the requested backend is unavailable.
ddlaStatus_t ddlaInit(DdlaHandle_t& handle, DdlaBackend requested_backend);

/// Initialize process grid and allocate backend resources.
/// Backend is resolved and validated collectively.
ddlaStatus_t ddlaSet(DdlaHandle_t handle, const MPI_Comm& comm = MPI_COMM_WORLD,
                      char major = 'R');

/// Initialize with explicit grid dimensions.
ddlaStatus_t ddlaSet(DdlaHandle_t handle, const MPI_Comm& comm,
                      int nprows, int npcols,
                      char major = 'R');

/// Release all resources; idempotent (safe to call multiple times).
ddlaStatus_t ddlaDestroy(DdlaHandle_t& handle);

// ---------------------------------------------------------------------------
// Backend queries
// ---------------------------------------------------------------------------

/// True if the library was compiled with support for the given backend.
bool ddlaBackendAvailable(DdlaBackend backend);

/// Return the resolved backend for this handle.
/// @throws std::runtime_error if @p handle is null.
DdlaBackend ddlaGetBackend(const DdlaHandle_t& handle);

// ---------------------------------------------------------------------------
// Memory, copies, events and queries (vendor-neutral)
//
// Every function takes the handle FIRST and returns a ddlaStatus_t; no CUDA or HIP type and no stream appears in
// a signature.  The handle owns its work queue and every function below runs on it, in call order (the only
// functions that expose the vendor stream underneath are ddlaGetStream / ddlaSetStream further down).  A CPU
// handle runs the host equivalent (malloc, memcpy, no-op event calls); a GPU handle runs the vendor call.  All
// of them return DDLA_STATUS_INVALID_HANDLE for a null handle.
// ---------------------------------------------------------------------------

/// Opaque event: a marked point in the work queued on a handle (always null for a CPU handle).
using ddlaEvent_t = void*;

/// Allocate backend-appropriate memory.
///
/// CPU handles allocate host memory (malloc).  GPU handles allocate device memory, ordered on the handle's queue.
///
/// - Returns DDLA_STATUS_INVALID_VALUE if @p ptr is null,
///   DDLA_STATUS_INVALID_HANDLE if @p handle is null.
/// - Zero-byte allocation: sets @p *ptr = nullptr and returns
///   DDLA_STATUS_SUCCESS.  No backend call is made.
/// - Nonzero CPU allocation: DDLA_STATUS_SUCCESS on success,
///   DDLA_STATUS_ALLOC_FAILED if malloc fails.
/// - Nonzero GPU allocation: DDLA_STATUS_SUCCESS on success,
///   DDLA_STATUS_INTERNAL_ERROR on device error.
///
/// @param handle     Backend handle (must not be null).
/// @param[out] ptr   Set to the allocated pointer on success.
/// @param bytes      Allocation size in bytes.
ddlaStatus_t ddlaMalloc(const DdlaHandle_t& handle, void** ptr, std::size_t bytes);

/// Typed forms of ddlaMalloc: `double* p; ddlaMalloc(handle, &p, n * sizeof(double))` needs no cast.
template <typename T>
inline ddlaStatus_t ddlaMalloc(const DdlaHandle_t& handle, T** ptr, std::size_t bytes)
{
    return ddlaMalloc(handle, reinterpret_cast<void**>(ptr), bytes);
}

/// Free backend-appropriate memory.
///
/// - Returns DDLA_STATUS_INVALID_HANDLE if @p handle is null.
/// - Freeing a null pointer is a valid no-op (DDLA_STATUS_SUCCESS).
/// - CPU handles free host memory; GPU handles free device memory, ordered on the handle's queue.
ddlaStatus_t ddlaFree(const DdlaHandle_t& handle, void* ptr);

/// Allocate / free page-locked (pinned) host memory.  Copies from pinned memory are true asynchronous DMA,
/// while a copy from pageable memory first synchronizes the handle's queue.  A CPU handle uses malloc / free.
/// Zero bytes sets @p *ptr = nullptr and succeeds; freeing a null pointer succeeds.
ddlaStatus_t ddlaMallocHost(const DdlaHandle_t& handle, void** ptr, std::size_t bytes);
template <typename T>
inline ddlaStatus_t ddlaMallocHost(const DdlaHandle_t& handle, T** ptr, std::size_t bytes)
{
    return ddlaMallocHost(handle, reinterpret_cast<void**>(ptr), bytes);
}
ddlaStatus_t ddlaFreeHost(const DdlaHandle_t& handle, void* ptr);

/// Copy @p bytes contiguous bytes from @p src to @p dst on the handle's queue.
///
/// There is no direction argument.  A CPU handle copies host to host.  A GPU handle asks the GPU runtime
/// where @p src and @p dst live and picks the direction itself:
///
///   - host to host: an ordinary memcpy, performed immediately on the calling thread.  It is NOT ordered
///     after work already queued on the handle: call ddlaSynchronize first if a buffer is still the target
///     of an asynchronous copy or kernel.
///   - host to device, device to host, device to device: an asynchronous copy on the handle's queue
///     (call ddlaSynchronize before reading a device-to-host result).
///
/// Memory allocated by ddlaMalloc on a GPU handle (or by the vendor runtime's device allocator) and managed
/// memory count as device memory; pageable, pinned and registered host memory count as host memory.  The source and destination
/// regions must not overlap.
///
/// - Returns DDLA_STATUS_INVALID_HANDLE if @p handle is null.
/// - Zero-byte copy: returns DDLA_STATUS_SUCCESS without dereferencing @p src or @p dst.
/// - Nonzero copy: returns DDLA_STATUS_INVALID_VALUE if @p src or @p dst is null.
/// - A GPU handle returns DDLA_STATUS_INTERNAL_ERROR if the runtime cannot classify a pointer or reports
///   a device error.
ddlaStatus_t ddlaMemcpy(const DdlaHandle_t& handle, void* dst, const void* src, std::size_t bytes);

/// Copy a @p height-row, @p width-byte rectangle between two pitched
/// (row-major, byte-addressed) buffers, like cudaMemcpy2D.
///
/// @p dpitch and @p spitch are the distances in bytes between the starts
/// of consecutive rows of @p dst and @p src; @p width is the number of bytes
/// copied per row.  The direction is derived exactly as for ddlaMemcpy, from the handle and from where the
/// first byte of each rectangle lives.
///
/// - Returns DDLA_STATUS_INVALID_HANDLE if @p handle is null.
/// - Empty copy (@p width or @p height is zero): returns
///   DDLA_STATUS_SUCCESS without dereferencing @p src or @p dst.
/// - Otherwise returns DDLA_STATUS_INVALID_VALUE if @p src or @p dst is
///   null, or if @p dpitch or @p spitch is smaller than @p width.
/// - Ordering, the host-to-host case and CPU-handle behavior are the same as for ddlaMemcpy.
///   The source and destination rectangles must not overlap.
ddlaStatus_t ddlaMemcpy2D(const DdlaHandle_t& handle, void* dst, std::size_t dpitch,
                           const void* src, std::size_t spitch,
                           std::size_t width, std::size_t height);

/// Set @p bytes bytes of @p ptr to @p value on the handle's queue.  A GPU handle needs
/// device, managed or pinned memory; a CPU handle calls memset.  Zero bytes succeeds without touching
/// @p ptr; a null @p ptr with a nonzero size is DDLA_STATUS_INVALID_VALUE.
ddlaStatus_t ddlaMemset(const DdlaHandle_t& handle, void* ptr, int value, std::size_t bytes);

/// Block until everything queued on the handle has finished.
ddlaStatus_t ddlaSynchronize(const DdlaHandle_t& handle);

/// Block until all work on the handle's device has finished (a CPU handle: no-op).
ddlaStatus_t ddlaDeviceSynchronize(const DdlaHandle_t& handle);

/// Events (a CPU handle: no-ops).  ddlaEventRecord marks the current end of the work queued on the handle;
/// ddlaEventWait makes the work queued on the handle FROM NOW ON wait for the marked point without blocking the
/// calling thread; ddlaEventSynchronize blocks the calling thread until the marked point has completed.  This is
/// how the library orders two phases that it overlaps (the eigensolver
/// back-transform): no stream is ever exposed.
ddlaStatus_t ddlaEventCreate(const DdlaHandle_t& handle, ddlaEvent_t* event);
ddlaStatus_t ddlaEventDestroy(const DdlaHandle_t& handle, ddlaEvent_t event);
ddlaStatus_t ddlaEventRecord(const DdlaHandle_t& handle, ddlaEvent_t event);
ddlaStatus_t ddlaEventWait(const DdlaHandle_t& handle, ddlaEvent_t event);
ddlaStatus_t ddlaEventSynchronize(const DdlaHandle_t& handle, ddlaEvent_t event);

/// Return (and clear) the error left behind by an earlier asynchronous call or kernel launch on this thread:
/// the check after `kernel<<<...>>>(...)`.  A CPU handle always returns DDLA_STATUS_SUCCESS.
ddlaStatus_t ddlaGetLastError(const DdlaHandle_t& handle);

/// Make the handle's device current on the CALLING thread.  A thread created after ddlaSet starts on
/// device 0, so a helper thread that issues ddla* calls on a GPU handle calls this first.  CPU: no-op.
ddlaStatus_t ddlaSetDevice(const DdlaHandle_t& handle);

/// Number of devices visible to the process (a CPU handle: 1).
ddlaStatus_t ddlaGetDeviceCount(const DdlaHandle_t& handle, int* count);

/// Free and total device memory in bytes (a CPU handle: 0 and 0).
ddlaStatus_t ddlaMemGetInfo(const DdlaHandle_t& handle, std::size_t* free_bytes, std::size_t* total_bytes);

/// Name of a status code; for DDLA_STATUS_INTERNAL_ERROR it also carries the vendor message of the last
/// failed vendor call on the calling thread.  The pointer stays valid until the next call on this thread.
const char* ddlaGetErrorString(ddlaStatus_t status);

/// Set the compute stream of a GPU handle; subsequent library calls on this
/// handle are enqueued on @p stream.
///
/// The handle takes ownership of @p stream: it is destroyed by ddlaDestroy,
/// so the caller must not destroy it separately.
///
/// @throws std::runtime_error if the handle uses the CPU backend (CPU
/// handles have no device stream).
ddlaStatus_t ddlaSetStream(const DdlaHandle_t& handle, void* stream);

// ---------------------------------------------------------------------------
// Public accessors (replace direct field access)
// ---------------------------------------------------------------------------

/// Return the duplicated MPI communicator owned by the handle.
MPI_Comm ddlaGetCommunicator(const DdlaHandle_t& handle);

/// Return the row communicator (processes in the same grid row).
MPI_Comm ddlaGetRowCommunicator(const DdlaHandle_t& handle);

/// Return the column communicator (processes in the same grid column).
MPI_Comm ddlaGetColCommunicator(const DdlaHandle_t& handle);

/// Rank of this process within the handle's communicator.
int ddlaGetRank(const DdlaHandle_t& handle);

/// Size of the handle's communicator.
int ddlaGetSize(const DdlaHandle_t& handle);

/// Process-grid row and column coordinates of this rank.
void ddlaGetGridCoords(const DdlaHandle_t& handle,
                          int& myprow, int& mypcol);

/// Number of process rows and columns in the grid.
void ddlaGetGridDims(const DdlaHandle_t& handle,
                        int& nprows, int& npcols);

/// Convert a rank to (row, col) process-grid coordinates.
void ddlaRankToRc(const DdlaHandle_t& handle,
                     int rank, int& row, int& col);

/// Convert (row, col) process-grid coordinates to a rank.
int ddlaRcToRank(const DdlaHandle_t& handle, int row, int col);

/// Return the compute stream (GPU: device stream; CPU: nullptr/0).
void* ddlaGetStream(const DdlaHandle_t& handle);

/// Check a ddlaStatus_t returned by a public API function.
///
/// Throws std::runtime_error with file:line context (and, for a vendor failure, the vendor message from
/// ddlaGetErrorString) when @p status is not DDLA_STATUS_SUCCESS.
inline void DDLA_CHECK(ddlaStatus_t status,
                       const char* file = __builtin_FILE(),
                       int line = __builtin_LINE())
{
    if (status != ddlaStatus_t::DDLA_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("ddla error ") +
            std::to_string(static_cast<int>(status)) + " (" + ddlaGetErrorString(status) + ") at " + file + ":" +
            std::to_string(line));
    }
}

} // namespace ddla

#endif // DDLA_HANDLE_T_H
