// Test-only native protocol observer. A missing fence is NOT a hardware
// corruption claim: this checks explicit retirement of the real H2D source.
#include "pgeadd_queue_trace_api.h"
#include "gpu/gpu.h"
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#if DDLA_HAS_GPU
using namespace ddla;
namespace {
std::atomic<bool> active{false};
std::atomic<int> record_fault{0}, create_ordinal{0};
std::atomic<uint64_t> uploads{0}, records{0}, waits{0}, drains{0}, wrong{0}, record_faults{0};
std::atomic<uint64_t> event_creates{0}, event_destroys{0}, stream_creates{0}, stream_destroys{0}, create_faults{0};
std::uintptr_t host_base = 0;
uint64_t host_bytes = 0, recorded = 0, retired = 0;
void* selected_queue = nullptr;
void* selected_event = nullptr;
template<class F> F next(const char* name)
{
    void* address = dlsym(RTLD_NEXT, name);
    if (!address) std::abort();
    F f;
    static_assert(sizeof(f) == sizeof(address), "dlsym ABI");
    std::memcpy(&f, &address, sizeof(f));
    return f;
}
bool managed(void* caller)
{
    Dl_info info{};
    if (!dladdr(caller, &info) || !info.dli_fname) return false;
    const char* base = std::strrchr(info.dli_fname, '/');
    return std::strstr(base ? base + 1 : info.dli_fname, "libddla.") != nullptr;
}
bool source(const void* pointer, std::size_t bytes)
{
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    return bytes && address >= host_base && address - host_base <= host_bytes &&
           bytes <= host_bytes - (address - host_base);
}
#define STRING_I(x) #x
#define STRING(x) STRING_I(x)
}
extern "C" void pgeadd_queue_trace_begin(const void* host, uint64_t bytes, void* queue, void* event, int fail)
{
    active.store(false);
    uploads = 0; records = 0; waits = 0; drains = 0; wrong = 0; record_faults = 0;
    recorded = retired = 0;
    host_base = reinterpret_cast<std::uintptr_t>(host); host_bytes = bytes;
    selected_queue = queue; selected_event = event; record_fault = fail;
    active.store(true);
}
extern "C" void pgeadd_queue_trace_end(pgeadd_queue_snapshot* out)
{
    active.store(false);
    *out = {uploads.load(), records.load(), waits.load(), drains.load(), wrong.load(),
            uploads.load() - retired, record_faults.load(), 0, 0, 0, 0, 0};
}
extern "C" void pgeadd_queue_trace_fail_create(int ordinal)
{
    event_creates = 0; event_destroys = 0; stream_creates = 0; stream_destroys = 0; create_faults = 0;
    create_ordinal = ordinal;
}
extern "C" void pgeadd_queue_trace_setup_snapshot(pgeadd_queue_snapshot* out)
{
    *out = {};
    out->event_creates = event_creates; out->event_destroys = event_destroys;
    out->stream_creates = stream_creates; out->stream_destroys = stream_destroys;
    out->create_faults = create_faults;
}
extern "C" gpuError_t gpuMemcpyAsync(void* dst, const void* src, std::size_t bytes,
                                     gpuMemcpyKind kind, gpuStream_t queue)
{
    static auto real = next<decltype(&gpuMemcpyAsync)>(STRING(gpuMemcpyAsync));
    const auto result = real(dst, src, bytes, kind, queue);
    if (active.load() && source(src, bytes) && result == gpuSuccess) {
        ++uploads;
        if (queue != selected_queue) ++wrong;
    }
    return result;
}
extern "C" gpuError_t gpuEventRecord(gpuEvent_t event, gpuStream_t queue)
{
    static auto real = next<decltype(&gpuEventRecord)>(STRING(gpuEventRecord));
    const bool observed = active.load() && queue == selected_queue &&
                          (!selected_event || event == selected_event);
    if (observed) {
        selected_event = event;
        if (record_fault.exchange(0)) { ++record_faults; return gpuErrorInvalidValue; }
    }
    const auto result = real(event, queue);
    if (observed && result == gpuSuccess) { ++records; recorded = uploads; }
    return result;
}
extern "C" gpuError_t gpuEventSynchronize(gpuEvent_t event)
{
    static auto real = next<decltype(&gpuEventSynchronize)>(STRING(gpuEventSynchronize));
    const auto result = real(event);
    if (active.load() && event == selected_event && result == gpuSuccess) {
        ++waits; retired = recorded;
    }
    return result;
}
extern "C" gpuError_t gpuStreamSynchronize(gpuStream_t queue)
{
    static auto real = next<decltype(&gpuStreamSynchronize)>(STRING(gpuStreamSynchronize));
    const auto result = real(queue);
    if (active.load() && queue == selected_queue && result == gpuSuccess) {
        ++drains; retired = uploads;
    }
    return result;
}
extern "C" gpuError_t gpuEventCreateWithFlags(gpuEvent_t* event, unsigned flags)
{
    static auto real = next<decltype(&gpuEventCreateWithFlags)>(STRING(gpuEventCreateWithFlags));
    const bool own = managed(__builtin_return_address(0));
    if (own && create_ordinal.load() > 0 && create_ordinal.fetch_sub(1) == 1) {
        ++create_faults; *event = nullptr; return gpuErrorInvalidValue;
    }
    const auto result = real(event, flags);
    if (own && result == gpuSuccess) ++event_creates;
    return result;
}
extern "C" gpuError_t gpuEventDestroy(gpuEvent_t event)
{
    static auto real = next<decltype(&gpuEventDestroy)>(STRING(gpuEventDestroy));
    const auto result = real(event);
    if (managed(__builtin_return_address(0)) && result == gpuSuccess) ++event_destroys;
    return result;
}
extern "C" gpuError_t gpuStreamCreate(gpuStream_t* queue)
{
    static auto real = next<decltype(&gpuStreamCreate)>(STRING(gpuStreamCreate));
    const auto result = real(queue);
    if (managed(__builtin_return_address(0)) && result == gpuSuccess) ++stream_creates;
    return result;
}
extern "C" gpuError_t gpuStreamCreateWithFlags(gpuStream_t* queue, unsigned flags)
{
    static auto real = next<decltype(&gpuStreamCreateWithFlags)>(STRING(gpuStreamCreateWithFlags));
    const auto result = real(queue, flags);
    if (managed(__builtin_return_address(0)) && result == gpuSuccess) ++stream_creates;
    return result;
}
extern "C" gpuError_t gpuStreamDestroy(gpuStream_t queue)
{
    static auto real = next<decltype(&gpuStreamDestroy)>(STRING(gpuStreamDestroy));
    const auto result = real(queue);
    if (managed(__builtin_return_address(0)) && result == gpuSuccess) ++stream_destroys;
    return result;
}
#endif
