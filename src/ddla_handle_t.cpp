#include "ddla_connector.h"
#include "ddla_stream.h"
#include "host_threads.h"
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <string>

namespace ddla {

// ---------------------------------------------------------------------------
// Backend availability (compile-time capabilities from generated config)
// ---------------------------------------------------------------------------
bool ddlaBackendAvailable(DdlaBackend backend)
{
    switch (backend) {
    case DdlaBackend::CPU:
#if DDLA_HAS_CPU
        return true;
#else
        return false;
#endif
    case DdlaBackend::GPU:
#if DDLA_HAS_GPU
        return true;
#else
        return false;
#endif
    }
    return false;
}

// ---------------------------------------------------------------------------
// ddlaInit — allocate opaque handle, validate backend
// ---------------------------------------------------------------------------
ddlaStatus_t ddlaInit(DdlaHandle_t& handle)
{
    return ddlaInit(handle, default_backend_v);
}

ddlaStatus_t ddlaInit(DdlaHandle_t& handle, DdlaBackend requested_backend)
{
    // Validate availability
    if (!ddlaBackendAvailable(requested_backend)) {
        throw std::runtime_error(
            "Requested backend is not available in this build of LibDDLA");
    }

    // Allocate fresh handle — do NOT read the incoming value (callers may
    // pass uninitialized pointer variables; reading them is UB).
    handle = new DdlaStream();
    handle->backend = requested_backend;
    return ddlaStatus_t::DDLA_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// ddlaGetBackend
// ---------------------------------------------------------------------------
DdlaBackend ddlaGetBackend(const DdlaHandle_t& handle)
{
    if (handle == nullptr) {
        throw std::runtime_error("ddlaGetBackend: handle is null");
    }
    return handle->backend;
}

// ---------------------------------------------------------------------------
// Shared by both ddlaSet overloads: validate the handle backend is
// available in this build, and collectively verify every MPI rank picked
// the same backend. Packs {value, -value} into a single MPI_MAX Allreduce
// to recover both the max and the min in one collective call instead of two.
// ---------------------------------------------------------------------------
static void resolve_and_validate_backend(DdlaHandle_t handle, const MPI_Comm& comm)
{
    const DdlaBackend resolved = handle->backend;

    if (!ddlaBackendAvailable(resolved)) {
        throw std::runtime_error(
            "Resolved backend is not available in this build of LibDDLA");
    }

    int send[2] = { static_cast<int>(resolved), -static_cast<int>(resolved) };
    int recv[2] = { 0, 0 };
    MPI_CHECK(MPI_Allreduce(send, recv, 2, MPI_INT, MPI_MAX, comm));
    int max_backend = recv[0];
    int min_backend = -recv[1];
    if (max_backend != min_backend) {
        throw std::runtime_error(
            "Not all MPI ranks selected the same backend");
    }
}

// ---------------------------------------------------------------------------
// ddlaSet — initialize process grid, verify backend consistency collectively
// ---------------------------------------------------------------------------
ddlaStatus_t ddlaSet(DdlaHandle_t handle, const MPI_Comm& comm, char major)
{
    if (handle == nullptr) {
        throw std::runtime_error("ddlaSet: handle is null");
    }

    resolve_and_validate_backend(handle, comm);

    // CPU handles must not execute GPU device setup
    handle->init(comm, major);
    if (handle->backend == DdlaBackend::CPU) handle->host_threads = detail::host_thread_budget(handle->comm);
    handle->initialized = true;
    return ddlaStatus_t::DDLA_STATUS_SUCCESS;
}

ddlaStatus_t ddlaSet(DdlaHandle_t handle, const MPI_Comm& comm,
              int nprows, int npcols, char major)
{
    if (handle == nullptr) {
        throw std::runtime_error("ddlaSet: handle is null");
    }

    resolve_and_validate_backend(handle, comm);

    handle->init(nprows, npcols, comm, major);
    if (handle->backend == DdlaBackend::CPU) handle->host_threads = detail::host_thread_budget(handle->comm);
    handle->initialized = true;
    return ddlaStatus_t::DDLA_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// ddlaDestroy — idempotent cleanup
// ---------------------------------------------------------------------------
ddlaStatus_t ddlaDestroy(DdlaHandle_t& handle)
{
    if (handle == nullptr) return ddlaStatus_t::DDLA_STATUS_SUCCESS;
    if (handle->destroyed) {
        handle = nullptr;
        return ddlaStatus_t::DDLA_STATUS_SUCCESS;
    }
    handle->clean();
    handle->destroyed = true;
    delete handle;
    handle = nullptr;
    return ddlaStatus_t::DDLA_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// ddlaGetStream — CPU returns nullptr; GPU returns the stream handle
// ---------------------------------------------------------------------------
void* ddlaGetStream(const DdlaHandle_t& handle)
{
    if (handle == nullptr) return nullptr;
    if (handle->backend == DdlaBackend::CPU) return nullptr;
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
    return handle->stream;
#else
    // GPU backend requested but this is a CPU-only build — unreachable
    // if ddlaBackendAvailable is checked correctly before init.
    return nullptr;
#endif
}

// ---------------------------------------------------------------------------
// Memory, copies, events and queries — dispatch by handle backend, with uniform validation
// ---------------------------------------------------------------------------
namespace {

// Vendor message of the most recent failed vendor call on this thread (read by ddlaGetErrorString).
thread_local std::string g_vendor_error;

// Where a copy goes, decided from the handle and the two pointers.
enum class CopyRoute { Host, HostToDevice, DeviceToHost, DeviceToDevice, Unknown };

// A CPU handle has no device: every copy is a host copy and the pointers are not inspected.  A GPU handle asks
// the runtime where each pointer lives (managed memory counts as device memory, see gpuPointerMemType).
CopyRoute route_copy(const DdlaHandle_t& handle, const void* dst, const void* src)
{
#if DDLA_GPU_VENDOR
    if (handle->backend == DdlaBackend::GPU) {
        const GpuMemType s = gpuPointerMemType(src);
        const GpuMemType d = gpuPointerMemType(dst);
        if (s == GpuMemType::Unknown || d == GpuMemType::Unknown) return CopyRoute::Unknown;
        if (s == GpuMemType::Host)
            return d == GpuMemType::Host ? CopyRoute::Host : CopyRoute::HostToDevice;
        return d == GpuMemType::Host ? CopyRoute::DeviceToHost : CopyRoute::DeviceToDevice;
    }
#else
    (void)handle; (void)dst; (void)src;
#endif
    return CopyRoute::Host;
}

constexpr ddlaStatus_t kOk = ddlaStatus_t::DDLA_STATUS_SUCCESS;
constexpr ddlaStatus_t kBadHandle = ddlaStatus_t::DDLA_STATUS_INVALID_HANDLE;
constexpr ddlaStatus_t kBadValue = ddlaStatus_t::DDLA_STATUS_INVALID_VALUE;
constexpr ddlaStatus_t kInternal = ddlaStatus_t::DDLA_STATUS_INTERNAL_ERROR;

inline bool is_gpu(const DdlaHandle_t& handle) { return handle->backend == DdlaBackend::GPU; }

#if DDLA_GPU_VENDOR
// Only meaningful for the three device routes.
gpuMemcpyKind vendor_kind(CopyRoute route)
{
    switch (route) {
    case CopyRoute::HostToDevice: return gpuMemcpyHostToDevice;
    case CopyRoute::DeviceToHost: return gpuMemcpyDeviceToHost;
    default:                      return gpuMemcpyDeviceToDevice;
    }
}

ddlaStatus_t to_status(gpuError_t err)
{
    if (err == gpuSuccess) return kOk;
    g_vendor_error = gpuGetErrorString(err);
    return kInternal;
}

inline gpuEvent_t vendor_event(ddlaEvent_t e) { return static_cast<gpuEvent_t>(e); }
#endif

// Host rectangle copy (the CPU-handle path and the host-to-host path of a GPU handle).
ddlaStatus_t host_memcpy2d(void* dst, std::size_t dpitch, const void* src, std::size_t spitch,
                           std::size_t width, std::size_t height)
{
    // Contiguous fast path: when both row pitches equal the row width the whole region is one block, so a
    // single memcpy avoids the per-row call overhead (stride-1 vectors can have a huge height).
    if (spitch == width && dpitch == width) {
        std::memcpy(dst, src, width * height);
        return kOk;
    }
    for (std::size_t i = 0; i < height; ++i)
        std::memcpy(static_cast<char*>(dst) + i * dpitch, static_cast<const char*>(src) + i * spitch, width);
    return kOk;
}

} // namespace

const char* ddlaGetErrorString(ddlaStatus_t status)
{
    const char* name = "DDLA_STATUS_UNKNOWN";
    switch (status) {
    case ddlaStatus_t::DDLA_STATUS_SUCCESS:             name = "DDLA_STATUS_SUCCESS"; break;
    case ddlaStatus_t::DDLA_STATUS_INVALID_HANDLE:      name = "DDLA_STATUS_INVALID_HANDLE"; break;
    case ddlaStatus_t::DDLA_STATUS_INVALID_VALUE:       name = "DDLA_STATUS_INVALID_VALUE"; break;
    case ddlaStatus_t::DDLA_STATUS_ALLOC_FAILED:        name = "DDLA_STATUS_ALLOC_FAILED"; break;
    case ddlaStatus_t::DDLA_STATUS_BACKEND_UNAVAILABLE: name = "DDLA_STATUS_BACKEND_UNAVAILABLE"; break;
    case ddlaStatus_t::DDLA_STATUS_INTERNAL_ERROR:      name = "DDLA_STATUS_INTERNAL_ERROR"; break;
    }
    thread_local std::string text;
    text = name;
    if (status == ddlaStatus_t::DDLA_STATUS_INTERNAL_ERROR && !g_vendor_error.empty())
        text += ": " + g_vendor_error;
    return text.c_str();
}

// -- allocation ------------------------------------------------------------------------------------------
ddlaStatus_t ddlaMalloc(const DdlaHandle_t& handle, void** ptr, std::size_t bytes)
{
    if (handle == nullptr) return kBadHandle;
    if (ptr == nullptr) return kBadValue;

    // Zero-byte allocation: set *ptr = nullptr, return success
    if (bytes == 0) {
        *ptr = nullptr;
        return kOk;
    }

    if (!is_gpu(handle)) {
        *ptr = std::malloc(bytes);
        return (*ptr != nullptr) ? kOk : ddlaStatus_t::DDLA_STATUS_ALLOC_FAILED;
    }
#if DDLA_GPU_VENDOR
    return to_status(gpuMallocAsync(ptr, bytes, static_cast<gpuStream_t>(handle->stream)));
#else
    return kInternal;
#endif
}

ddlaStatus_t ddlaFree(const DdlaHandle_t& handle, void* ptr)
{
    if (handle == nullptr) return kBadHandle;
    // Freeing nullptr is a valid no-op on both CPU and GPU
    if (ptr == nullptr) return kOk;
    if (!is_gpu(handle)) {
        std::free(ptr);
        return kOk;
    }
#if DDLA_GPU_VENDOR
    return to_status(gpuFreeAsync(ptr, static_cast<gpuStream_t>(handle->stream)));
#else
    return kInternal;
#endif
}

ddlaStatus_t ddlaMallocHost(const DdlaHandle_t& handle, void** ptr, std::size_t bytes)
{
    if (handle == nullptr) return kBadHandle;
    if (ptr == nullptr) return kBadValue;
    if (bytes == 0) {
        *ptr = nullptr;
        return kOk;
    }
    if (!is_gpu(handle)) {
        *ptr = std::malloc(bytes);
        return (*ptr != nullptr) ? kOk : ddlaStatus_t::DDLA_STATUS_ALLOC_FAILED;
    }
#if DDLA_GPU_VENDOR
    return to_status(gpuHostAlloc(ptr, bytes, gpuHostAllocDefault));
#else
    return kInternal;
#endif
}

ddlaStatus_t ddlaFreeHost(const DdlaHandle_t& handle, void* ptr)
{
    if (handle == nullptr) return kBadHandle;
    if (ptr == nullptr) return kOk;
    if (!is_gpu(handle)) {
        std::free(ptr);
        return kOk;
    }
#if DDLA_GPU_VENDOR
    return to_status(gpuFreeHost(ptr));
#else
    return kInternal;
#endif
}

// -- copies and memset -----------------------------------------------------------------------------------
namespace detail {

ddlaStatus_t memcpy_on(const DdlaHandle_t& handle, void* dst, const void* src, std::size_t bytes,
                       void* queue)
{
    if (handle == nullptr) return kBadHandle;

    // Zero-byte copy: no-op, do not dereference src/dst
    if (bytes == 0) return kOk;

    // Nonzero copy: reject null source or destination
    if (dst == nullptr || src == nullptr) return kBadValue;

    const CopyRoute route = route_copy(handle, dst, src);
    if (route == CopyRoute::Unknown) return kInternal;
    if (route == CopyRoute::Host) {
        std::memcpy(dst, src, bytes);
        return kOk;
    }
#if DDLA_GPU_VENDOR
    return to_status(gpuMemcpyAsync(dst, src, bytes, vendor_kind(route), static_cast<gpuStream_t>(queue)));
#else
    (void)queue;
    return kInternal;
#endif
}

} // namespace detail

ddlaStatus_t ddlaMemcpy(const DdlaHandle_t& handle, void* dst, const void* src, std::size_t bytes)
{
    return detail::memcpy_on(handle, dst, src, bytes, handle ? static_cast<gpuStream_t>(handle->stream) : nullptr);
}

ddlaStatus_t ddlaMemcpy2D(const DdlaHandle_t& handle, void* dst, std::size_t dpitch,
                          const void* src, std::size_t spitch,
                          std::size_t width, std::size_t height)
{
    if (handle == nullptr) return kBadHandle;

    // Empty rectangle: no-op, do not dereference src/dst
    if (width == 0 || height == 0) return kOk;

    if (dst == nullptr || src == nullptr) return kBadValue;

    // A pitch shorter than the row width would make rows overlap (and
    // cudaMemcpy2D rejects it too).
    if (dpitch < width || spitch < width) return kBadValue;

    const CopyRoute route = route_copy(handle, dst, src);
    if (route == CopyRoute::Unknown) return kInternal;
    if (route == CopyRoute::Host) return host_memcpy2d(dst, dpitch, src, spitch, width, height);
#if DDLA_GPU_VENDOR
    return to_status(gpuMemcpy2DAsync(dst, dpitch, src, spitch, width, height, vendor_kind(route),
                                      static_cast<gpuStream_t>(handle->stream)));
#else
    return kInternal;
#endif
}

ddlaStatus_t ddlaMemset(const DdlaHandle_t& handle, void* ptr, int value, std::size_t bytes)
{
    if (handle == nullptr) return kBadHandle;
    if (bytes == 0) return kOk;
    if (ptr == nullptr) return kBadValue;
    if (!is_gpu(handle)) {
        std::memset(ptr, value, bytes);
        return kOk;
    }
#if DDLA_GPU_VENDOR
    return to_status(gpuMemsetAsync(ptr, value, bytes, static_cast<gpuStream_t>(handle->stream)));
#else
    return kInternal;
#endif
}

// -- synchronization and events --------------------------------------------------------------------
namespace detail {

ddlaStatus_t synchronize_on(const DdlaHandle_t& handle, void* queue)
{
    if (handle == nullptr) return kBadHandle;
    if (!is_gpu(handle)) return kOk;
#if DDLA_GPU_VENDOR
    return to_status(gpuStreamSynchronize(static_cast<gpuStream_t>(queue)));
#else
    (void)queue;
    return kInternal;
#endif
}

} // namespace detail

ddlaStatus_t ddlaSynchronize(const DdlaHandle_t& handle)
{
    return detail::synchronize_on(handle, handle ? static_cast<gpuStream_t>(handle->stream) : nullptr);
}

ddlaStatus_t ddlaDeviceSynchronize(const DdlaHandle_t& handle)
{
    if (handle == nullptr) return kBadHandle;
    if (!is_gpu(handle)) return kOk;
#if DDLA_GPU_VENDOR
    return to_status(gpuDeviceSynchronize());
#else
    return kInternal;
#endif
}

ddlaStatus_t ddlaEventCreate(const DdlaHandle_t& handle, ddlaEvent_t* event)
{
    if (handle == nullptr) return kBadHandle;
    if (event == nullptr) return kBadValue;
    *event = nullptr;
    if (!is_gpu(handle)) return kOk;
#if DDLA_GPU_VENDOR
    gpuEvent_t e = nullptr;
    const ddlaStatus_t st = to_status(gpuEventCreate(&e));
    if (st == kOk) *event = e;
    return st;
#else
    return kInternal;
#endif
}

ddlaStatus_t ddlaEventDestroy(const DdlaHandle_t& handle, ddlaEvent_t event)
{
    if (handle == nullptr) return kBadHandle;
    if (!is_gpu(handle) || event == nullptr) return kOk;
#if DDLA_GPU_VENDOR
    return to_status(gpuEventDestroy(vendor_event(event)));
#else
    return kInternal;
#endif
}

namespace detail {

ddlaStatus_t event_record_on(const DdlaHandle_t& handle, ddlaEvent_t event, void* queue)
{
    if (handle == nullptr) return kBadHandle;
    if (!is_gpu(handle)) return kOk;
#if DDLA_GPU_VENDOR
    return to_status(gpuEventRecord(vendor_event(event), static_cast<gpuStream_t>(queue)));
#else
    (void)event; (void)queue;
    return kInternal;
#endif
}

} // namespace detail

ddlaStatus_t ddlaEventRecord(const DdlaHandle_t& handle, ddlaEvent_t event)
{
    return detail::event_record_on(handle, event, handle ? static_cast<gpuStream_t>(handle->stream) : nullptr);
}

namespace detail {

ddlaStatus_t event_wait_on(const DdlaHandle_t& handle, ddlaEvent_t event, void* queue)
{
    if (handle == nullptr) return kBadHandle;
    if (!is_gpu(handle)) return kOk;
#if DDLA_GPU_VENDOR
    return to_status(gpuStreamWaitEvent(static_cast<gpuStream_t>(queue), vendor_event(event), 0));
#else
    (void)event; (void)queue;
    return kInternal;
#endif
}

} // namespace detail

ddlaStatus_t ddlaEventWait(const DdlaHandle_t& handle, ddlaEvent_t event)
{
    return detail::event_wait_on(handle, event, handle ? static_cast<gpuStream_t>(handle->stream) : nullptr);
}

ddlaStatus_t ddlaEventSynchronize(const DdlaHandle_t& handle, ddlaEvent_t event)
{
    if (handle == nullptr) return kBadHandle;
    if (!is_gpu(handle)) return kOk;
#if DDLA_GPU_VENDOR
    return to_status(gpuEventSynchronize(vendor_event(event)));
#else
    (void)event;
    return kInternal;
#endif
}

// -- errors and device queries ---------------------------------------------------------------------------
ddlaStatus_t ddlaGetLastError(const DdlaHandle_t& handle)
{
    if (handle == nullptr) return kBadHandle;
    if (!is_gpu(handle)) return kOk;
#if DDLA_GPU_VENDOR
    return to_status(gpuGetLastError());
#else
    return kInternal;
#endif
}

ddlaStatus_t ddlaSetDevice(const DdlaHandle_t& handle)
{
    if (handle == nullptr) return kBadHandle;
    if (!is_gpu(handle)) return kOk;
#if DDLA_GPU_VENDOR
    return to_status(gpuSetDevice(handle->local_device));
#else
    return kInternal;
#endif
}

ddlaStatus_t ddlaGetDeviceCount(const DdlaHandle_t& handle, int* count)
{
    if (handle == nullptr) return kBadHandle;
    if (count == nullptr) return kBadValue;
    *count = 1;
    if (!is_gpu(handle)) return kOk;
#if DDLA_GPU_VENDOR
    return to_status(gpuGetDeviceCount(count));
#else
    return kInternal;
#endif
}

ddlaStatus_t ddlaMemGetInfo(const DdlaHandle_t& handle, std::size_t* free_bytes, std::size_t* total_bytes)
{
    if (handle == nullptr) return kBadHandle;
    if (free_bytes == nullptr || total_bytes == nullptr) return kBadValue;
    *free_bytes = 0;
    *total_bytes = 0;
    if (!is_gpu(handle)) return kOk;
#if DDLA_GPU_VENDOR
    return to_status(gpuMemGetInfo(free_bytes, total_bytes));
#else
    return kInternal;
#endif
}

// ---------------------------------------------------------------------------
// ddlaSetStream — GPU handles only; CPU handles have no device stream
// ---------------------------------------------------------------------------
ddlaStatus_t ddlaSetStream(const DdlaHandle_t& handle, void* stream)
{
    if (handle == nullptr) return ddlaStatus_t::DDLA_STATUS_INVALID_HANDLE;
    if (handle->backend == DdlaBackend::CPU) {
        throw std::runtime_error(
            "ddlaSetStream: setting a stream requires a GPU handle");
    }
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
    handle->stream = stream;
#endif
    // Re-bind the BLAS/solver handles so subsequent calls honor the new
    // stream (mirrors what DdlaStream::init does at creation time).
#if DDLA_GPU_VENDOR
    BLAS_CHECK(gpublasSetStream(static_cast<gpublasHandle_t>(handle->blasH), static_cast<gpuStream_t>(handle->stream)));
    SOLVER_CHECK(gpusolverSetStream(static_cast<gpusolverHandle_t>(handle->solverH), static_cast<gpuStream_t>(handle->stream)));
#endif
    return ddlaStatus_t::DDLA_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Public accessors
// ---------------------------------------------------------------------------
MPI_Comm ddlaGetCommunicator(const DdlaHandle_t& handle)
{
    if (handle == nullptr) return MPI_COMM_NULL;
    return handle->comm;
}

int ddlaGetRank(const DdlaHandle_t& handle)
{
    if (handle == nullptr) return -1;
    if (handle->comm == MPI_COMM_NULL) return -1;
    int rank;
    MPI_Comm_rank(handle->comm, &rank);
    return rank;
}

int ddlaGetSize(const DdlaHandle_t& handle)
{
    if (handle == nullptr) return 0;
    if (handle->comm == MPI_COMM_NULL) return 0;
    int size;
    MPI_Comm_size(handle->comm, &size);
    return size;
}

void ddlaGetGridCoords(const DdlaHandle_t& handle,
                          int& myprow, int& mypcol)
{
    if (handle == nullptr) {
        myprow = -1; mypcol = -1;
        return;
    }
    myprow = handle->myprow_;
    mypcol = handle->mypcol_;
}

void ddlaGetGridDims(const DdlaHandle_t& handle,
                        int& nprows, int& npcols)
{
    if (handle == nullptr) {
        nprows = 0; npcols = 0;
        return;
    }
    nprows = handle->nprows_;
    npcols = handle->npcols_;
}

void ddlaRankToRc(const DdlaHandle_t& handle,
                     int rank, int& row, int& col)
{
    if (handle == nullptr) {
        row = -1; col = -1;
        return;
    }
    handle->rank_to_rc(rank, row, col);
}

int ddlaRcToRank(const DdlaHandle_t& handle, int row, int col)
{
    if (handle == nullptr) return -1;
    return handle->rc_to_rank(row, col);
}

MPI_Comm ddlaGetRowCommunicator(const DdlaHandle_t& handle)
{
    if (handle == nullptr) return MPI_COMM_NULL;
    return handle->row_comm;
}

MPI_Comm ddlaGetColCommunicator(const DdlaHandle_t& handle)
{
    if (handle == nullptr) return MPI_COMM_NULL;
    return handle->col_comm;
}

} // namespace ddla
