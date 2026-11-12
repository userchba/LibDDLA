#include "ddla_connector.h"
#include "ddla_stream.h"
#include "host_threads.h"
#include "host_parallel.h"
#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <string>
#include <limits>
#if DDLA_HAS_GPU
#include "memory_tri_kernels.h"
#endif

namespace ddla {

// ---------------------------------------------------------------------------
// Backend availability (compile-time capabilities from generated config)
// ---------------------------------------------------------------------------
bool ddlaBackendAvailable(DdlaBackend backend)
{
    switch (backend) {
    case DdlaBackend::CPU:
        return true;
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
    handle->host_threads = detail::host_thread_budget(handle->comm, handle->node_ranks);
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
    handle->host_threads = detail::host_thread_budget(handle->comm, handle->node_ranks);
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
#if DDLA_HAS_GPU
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

#if DDLA_HAS_GPU
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
                           std::size_t width, std::size_t height, int threads)
{
    // Amortize worker startup over about 1 MiB per chunk. The synchronous
    // host helper joins before either buffer can be reused by the caller.
    const auto grain = static_cast<long long>(std::max<std::size_t>(1, (1 << 20) / width));
    const auto copy = [&](long long lo, long long hi) {
        auto* d = static_cast<char*>(dst) + static_cast<std::size_t>(lo) * dpitch;
        const auto* s = static_cast<const char*>(src) + static_cast<std::size_t>(lo) * spitch;
        if (spitch == width && dpitch == width) {
            std::memcpy(d, s, static_cast<std::size_t>(hi - lo) * width);
        } else {
            for (std::size_t i = 0; i < static_cast<std::size_t>(hi - lo); ++i)
                std::memcpy(d + i * dpitch, s + i * spitch, width);
        }
    };
    if (threads <= 1 || height <= static_cast<std::size_t>(grain))
        copy(0, static_cast<long long>(height));
    else
        detail::host_parallel_for(static_cast<long long>(height), grain, threads, copy);
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
#if DDLA_HAS_GPU
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
#if DDLA_HAS_GPU
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
#if DDLA_HAS_GPU
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
#if DDLA_HAS_GPU
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
#if DDLA_HAS_GPU
    return to_status(gpuMemcpyAsync(dst, src, bytes, vendor_kind(route), static_cast<gpuStream_t>(queue)));
#else
    (void)queue;
    return kInternal;
#endif
}

} // namespace detail

ddlaStatus_t ddlaMemcpy(const DdlaHandle_t& handle, void* dst, const void* src, std::size_t bytes)
{
    return detail::memcpy_on(handle, dst, src, bytes, handle ? handle->stream : nullptr);
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
    if (route == CopyRoute::Host) return host_memcpy2d(dst, dpitch, src, spitch, width, height, handle->host_threads);
#if DDLA_HAS_GPU
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
#if DDLA_HAS_GPU
    return to_status(gpuMemsetAsync(ptr, value, bytes, static_cast<gpuStream_t>(handle->stream)));
#else
    return kInternal;
#endif
}

namespace {
// All pointer arithmetic below is bounded by a representable byte span.
bool triangle_layout(int m, int n, int ld, std::size_t element_bytes, std::size_t& pitch)
{
    const auto limit = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if (ld < m || element_bytes == 0 || std::size_t(ld) > limit / element_bytes)
        return false;
    pitch = std::size_t(ld) * element_bytes;
    const auto width = std::size_t(m) * element_bytes;
    return std::size_t(n - 1) <= (limit - width) / pitch;
}

bool empty_triangle(char u, int m, int n, bool diagonal)
{
    return m == 0 || n == 0 || (!diagonal && (u == 'U' ? n == 1 : m == 1));
}

void triangle_range(bool upper, bool diagonal, int rows, int column, int& first, int& end)
{
    first = upper ? 0 : std::min(rows, column + (diagonal ? 0 : 1));
    end = upper ? std::min(rows, column + (diagonal ? 1 : 0)) : rows;
}

template <bool Copy>
void host_triangle(const DdlaHandle_t& h, bool upper, bool diagonal, int m, int n,
                    void* dst, std::size_t dpitch, const void* src, std::size_t spitch,
                    std::size_t element_bytes, int value)
{
    const auto grain = static_cast<long long>(std::max<std::size_t>(
        1, (1 << 20) / (std::size_t(m) * element_bytes)));
    const auto operation = [&](long long lo, long long hi) {
        for (int j = int(lo); j < hi; ++j) {
            int first, end;
            triangle_range(upper, diagonal, m, j, first, end);
            if (first == end)
                continue;
            auto* output = static_cast<char*>(dst) + std::size_t(j) * dpitch +
                           std::size_t(first) * element_bytes;
            const auto bytes = std::size_t(end - first) * element_bytes;
            if constexpr (Copy) {
                const auto* input = static_cast<const char*>(src) + std::size_t(j) * spitch +
                                    std::size_t(first) * element_bytes;
                std::memcpy(output, input, bytes);
            } else {
                std::memset(output, value, bytes);
            }
        }
    };
    detail::host_parallel_for(n, grain, h->host_threads, operation);
}
}

namespace detail {
ddlaStatus_t memset_tri(const DdlaHandle_t& h, char u, int m, int n, void* ptr, int ld,
                         int value, std::size_t element_bytes, bool diagonal)
{
    if (!h) return kBadHandle;
    if (u == 'u') u = 'U';
    if (u == 'l') u = 'L';
    if ((u != 'U' && u != 'L') || m < 0 || n < 0) return kBadValue;
    if (empty_triangle(u, m, n, diagonal)) return kOk;
    std::size_t pitch;
    if (!ptr || !triangle_layout(m, n, ld, element_bytes, pitch)) return kBadValue;
    if (!is_gpu(h)) {
        host_triangle<false>(h, u == 'U', diagonal, m, n, ptr, pitch, nullptr, 0,
                             element_bytes, value);
        return kOk;
    }
#if DDLA_HAS_GPU
    return to_status(memory_tri_gpu(h->stream, false, u == 'U', diagonal, m, n, ptr, pitch,
                                    nullptr, 0, element_bytes, value));
#else
    return kInternal;
#endif
}

ddlaStatus_t memcpy_tri(const DdlaHandle_t& h, char u, int m, int n, void* dst, int ldd,
                         const void* src, int lds, std::size_t element_bytes, bool diagonal)
{
    if (!h) return kBadHandle;
    if (u == 'u') u = 'U';
    if (u == 'l') u = 'L';
    if ((u != 'U' && u != 'L') || m < 0 || n < 0) return kBadValue;
    if (empty_triangle(u, m, n, diagonal)) return kOk;
    std::size_t dpitch, spitch;
    if (!dst || !src || !triangle_layout(m, n, ldd, element_bytes, dpitch) ||
        !triangle_layout(m, n, lds, element_bytes, spitch)) return kBadValue;
    if (dst == src && ldd == lds) return kOk;
    const CopyRoute route = route_copy(h, dst, src);
    if (route == CopyRoute::Unknown) return kInternal;
    if (route == CopyRoute::Host) {
        host_triangle<true>(h, u == 'U', diagonal, m, n, dst, dpitch, src, spitch,
                            element_bytes, 0);
        return kOk;
    }
#if DDLA_HAS_GPU
    if (route == CopyRoute::DeviceToDevice)
        return to_status(memory_tri_gpu(h->stream, true, u == 'U', diagonal, m, n, dst, dpitch,
                                        src, spitch, element_bytes, 0));
    for (int j = 0; j < n; ++j) {
        int first, end;
        triangle_range(u == 'U', diagonal, m, j, first, end);
        if (first == end) continue;
        const auto offset = std::size_t(first) * element_bytes;
        const auto status = to_status(gpuMemcpyAsync(
            static_cast<char*>(dst) + std::size_t(j) * dpitch + offset,
            static_cast<const char*>(src) + std::size_t(j) * spitch + offset,
            std::size_t(end - first) * element_bytes, vendor_kind(route),
            static_cast<gpuStream_t>(h->stream)));
        if (status != kOk) return status;
    }
    return kOk;
#else
    return kInternal;
#endif
}
}

// -- synchronization and events --------------------------------------------------------------------
namespace detail {

ddlaStatus_t synchronize_on(const DdlaHandle_t& handle, void* queue)
{
    if (handle == nullptr) return kBadHandle;
    if (!is_gpu(handle)) return kOk;
#if DDLA_HAS_GPU
    return to_status(gpuStreamSynchronize(static_cast<gpuStream_t>(queue)));
#else
    (void)queue;
    return kInternal;
#endif
}

} // namespace detail

ddlaStatus_t ddlaSynchronize(const DdlaHandle_t& handle)
{
    return detail::synchronize_on(handle, handle ? handle->stream : nullptr);
}

ddlaStatus_t ddlaDeviceSynchronize(const DdlaHandle_t& handle)
{
    if (handle == nullptr) return kBadHandle;
    if (!is_gpu(handle)) return kOk;
#if DDLA_HAS_GPU
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
#if DDLA_HAS_GPU
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
#if DDLA_HAS_GPU
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
#if DDLA_HAS_GPU
    return to_status(gpuEventRecord(vendor_event(event), static_cast<gpuStream_t>(queue)));
#else
    (void)event; (void)queue;
    return kInternal;
#endif
}

} // namespace detail

ddlaStatus_t ddlaEventRecord(const DdlaHandle_t& handle, ddlaEvent_t event)
{
    return detail::event_record_on(handle, event, handle ? handle->stream : nullptr);
}

namespace detail {

ddlaStatus_t event_wait_on(const DdlaHandle_t& handle, ddlaEvent_t event, void* queue)
{
    if (handle == nullptr) return kBadHandle;
    if (!is_gpu(handle)) return kOk;
#if DDLA_HAS_GPU
    return to_status(gpuStreamWaitEvent(static_cast<gpuStream_t>(queue), vendor_event(event), 0));
#else
    (void)event; (void)queue;
    return kInternal;
#endif
}

} // namespace detail

ddlaStatus_t ddlaEventWait(const DdlaHandle_t& handle, ddlaEvent_t event)
{
    return detail::event_wait_on(handle, event, handle ? handle->stream : nullptr);
}

ddlaStatus_t ddlaEventSynchronize(const DdlaHandle_t& handle, ddlaEvent_t event)
{
    if (handle == nullptr) return kBadHandle;
    if (!is_gpu(handle)) return kOk;
#if DDLA_HAS_GPU
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
#if DDLA_HAS_GPU
    return to_status(gpuGetLastError());
#else
    return kInternal;
#endif
}

ddlaStatus_t ddlaSetDevice(const DdlaHandle_t& handle)
{
    if (handle == nullptr) return kBadHandle;
    if (!is_gpu(handle)) return kOk;
#if DDLA_HAS_GPU
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
#if DDLA_HAS_GPU
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
#if DDLA_HAS_GPU
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
#if DDLA_HAS_GPU
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
