// Native API interposition for test_stream_ownership (LD_PRELOAD).
#include "stream_ownership_trace_api.h"
#include "gpu/gpu.h"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

#if DDLA_HAS_GPU
using namespace ddla;
namespace {
enum Counter { Creates, Destroys, Synchronizes, DefaultSynchronizes, DefaultDestroys,
               BlasBinds, SolverBinds, BlasFaults, SolverFaults, CounterCount };
std::atomic<uint64_t> counts[CounterCount]{};
std::atomic<int> fault{StreamNoFault};

// Vendor libraries can own private streams. Count only calls made directly by
// LibDDLA or this test, so their implementation details do not affect balance.
bool managed(void* caller)
{
    Dl_info info{};
    if (!dladdr(caller, &info) || !info.dli_fname) return false;
    const char* path = std::strrchr(info.dli_fname, '/');
    const char* file = path ? path + 1 : info.dli_fname;
    return std::strstr(file, "libddla.") || std::strstr(file, "libddla-") ||
           std::strstr(file, "test_stream_ownership");
}
template <typename F> F next(const char* name)
{
    void* address = dlsym(RTLD_NEXT, name);
    if (!address) std::abort();
    F result;
    static_assert(sizeof(result) == sizeof(address), "dlsym function pointer ABI");
    std::memcpy(&result, &address, sizeof(result));
    return result;
}
void count(Counter which) { counts[which].fetch_add(1, std::memory_order_relaxed); }
bool fail(int expected)
{
    return fault.compare_exchange_strong(expected, StreamNoFault);
}
#define DDLA_STRING_I(x) #x
#define DDLA_STRING(x) DDLA_STRING_I(x)
}

extern "C" void stream_ownership_trace_reset()
{
    for (auto& value : counts) value.store(0);
    fault.store(StreamNoFault);
}
extern "C" void stream_ownership_trace_snapshot(stream_ownership_snapshot* out)
{
    *out = {counts[Creates].load(), counts[Destroys].load(), counts[Synchronizes].load(),
            counts[DefaultSynchronizes].load(), counts[DefaultDestroys].load(),
            counts[BlasBinds].load(), counts[SolverBinds].load(),
            counts[BlasFaults].load(), counts[SolverFaults].load()};
}
extern "C" void stream_ownership_trace_fail_once(int which) { fault.store(which); }

extern "C" gpuError_t gpuStreamCreate(gpuStream_t* stream)
{
    static auto real = next<decltype(&gpuStreamCreate)>(DDLA_STRING(gpuStreamCreate));
    const auto status = real(stream);
    if (status == gpuSuccess && managed(__builtin_return_address(0))) count(Creates);
    return status;
}
extern "C" gpuError_t gpuStreamCreateWithFlags(gpuStream_t* stream, unsigned flags)
{
    static auto real = next<decltype(&gpuStreamCreateWithFlags)>(DDLA_STRING(gpuStreamCreateWithFlags));
    const auto status = real(stream, flags);
    if (status == gpuSuccess && managed(__builtin_return_address(0))) count(Creates);
    return status;
}
extern "C" gpuError_t gpuStreamDestroy(gpuStream_t stream)
{
    static auto real = next<decltype(&gpuStreamDestroy)>(DDLA_STRING(gpuStreamDestroy));
    const bool ours = managed(__builtin_return_address(0));
    if (ours && !stream) count(DefaultDestroys);
    const auto status = real(stream);
    if (ours && status == gpuSuccess) count(Destroys);
    return status;
}
extern "C" gpuError_t gpuStreamSynchronize(gpuStream_t stream)
{
    static auto real = next<decltype(&gpuStreamSynchronize)>(DDLA_STRING(gpuStreamSynchronize));
    if (managed(__builtin_return_address(0))) {
        count(Synchronizes);
        if (!stream) count(DefaultSynchronizes);
    }
    return real(stream);
}
extern "C" gpublasStatus_t gpublasSetStream(gpublasHandle_t handle, gpuStream_t stream)
{
    static auto real = next<decltype(&gpublasSetStream)>(DDLA_STRING(gpublasSetStream));
    if (managed(__builtin_return_address(0))) {
        count(BlasBinds);
        if (fail(StreamBlasFault)) {
            count(BlasFaults);
            return DDLA_GPU_BLAS_CONST(STATUS_INTERNAL_ERROR);
        }
    }
    return real(handle, stream);
}
extern "C" gpusolverStatus_t gpusolverSetStream(gpusolverHandle_t handle, gpuStream_t stream)
{
    static auto real = next<decltype(&gpusolverSetStream)>(DDLA_STRING(gpusolverSetStream));
    if (managed(__builtin_return_address(0))) {
        count(SolverBinds);
        if (fail(StreamSolverFault)) {
            count(SolverFaults);
            return DDLA_GPU_SOLVER_CONST(STATUS_INTERNAL_ERROR);
        }
    }
    return real(handle, stream);
}
#endif
