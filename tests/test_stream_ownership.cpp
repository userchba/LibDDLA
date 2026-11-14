#include "api_grid_test_common.h"
#include "ddla_stream.h"
#include "stream_ownership_trace_api.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <thread>

using namespace ddla;
using namespace api_grid_test;
namespace {
void check(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
void cpu_cases()
{
    DdlaHandle_t h = nullptr;
    check(ddlaSetStream(h, nullptr) == ddlaStatus_t::DDLA_STATUS_INVALID_HANDLE,
          "null handle stream setter");
    DDLA_CHECK(ddlaInit(h, DdlaBackend::CPU));
    bool threw = false;
    try { ddlaSetStream(h, nullptr); } catch (const std::runtime_error&) { threw = true; }
    check(threw && ddlaGetStream(h) == nullptr, "CPU setter must reject streams");
    DDLA_CHECK(ddlaDestroy(h));
}
#if DDLA_HAS_GPU
template <typename F> F symbol(const char* name)
{
    void* address = dlsym(RTLD_DEFAULT, name);
    check(address != nullptr, "stream ownership preload is required for GPU validation");
    F result;
    static_assert(sizeof(result) == sizeof(address), "dlsym function pointer ABI");
    std::memcpy(&result, &address, sizeof(result));
    return result;
}
struct Trace {
    decltype(&stream_ownership_trace_reset) reset = symbol<decltype(reset)>("stream_ownership_trace_reset");
    decltype(&stream_ownership_trace_snapshot) snapshot = symbol<decltype(snapshot)>("stream_ownership_trace_snapshot");
    decltype(&stream_ownership_trace_fail_once) fail = symbol<decltype(fail)>("stream_ownership_trace_fail_once");
    stream_ownership_snapshot read() const { stream_ownership_snapshot s{}; snapshot(&s); return s; }
};
void bindings(DdlaHandle_t h, void* expected)
{
    gpuStream_t stream = nullptr;
    BLAS_CHECK(DDLA_GPU_BLAS(GetStream)(static_cast<gpublasHandle_t>(h->blasH), &stream));
    check(stream == expected, "BLAS stream binding");
    SOLVER_CHECK(gpusolverGetStream(static_cast<gpusolverHandle_t>(h->solverH), &stream));
    check(stream == expected, "solver stream binding");
    check(ddlaGetStream(h) == expected, "public stream binding");
}
gpuStream_t create_stream()
{
    gpuStream_t stream = nullptr;
    GPU_CHECK(gpuStreamCreateWithFlags(&stream, gpuStreamNonBlocking));
    return stream;
}

// The callback only uses host atomics; it calls neither MPI nor the GPU runtime.
// A bounded releaser avoids deadlock while proving replacement/teardown waits
// for queued work. Only the initializing thread invokes library/MPI calls.
struct Pending {
    std::atomic<bool> entered{false}, release{false}, finished{false};
    std::thread releaser;
    static void callback(void* opaque)
    {
        auto& p = *static_cast<Pending*>(opaque);
        p.entered.store(true);
        while (!p.release.load()) std::this_thread::yield();
        p.finished.store(true);
    }
    explicit Pending(void* stream)
    {
        GPU_CHECK(DDLA_GPU_RT(LaunchHostFunc)(static_cast<gpuStream_t>(stream), callback, this));
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!entered.load() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        if (!entered.load()) {
            release.store(true);
            GPU_CHECK(gpuStreamSynchronize(static_cast<gpuStream_t>(stream)));
            throw std::runtime_error("native stream callback did not start");
        }
        releaser = std::thread([this] {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            release.store(true);
        });
    }
    ~Pending()
    {
        release.store(true);
        if (releaser.joinable()) releaser.join();
    }
};
void numerical_case(DdlaHandle_t h)
{
    DeviceBuffer<double> a(h, 1), b(h, 1), c(h, 1);
    upload(h, a.ptr, std::vector<double>{4});
    upload(h, b.ptr, std::vector<double>{9});
    gemm(h, 'N', 'N', 1, 1, 1, 1.0, a.ptr, 1, b.ptr, 1, 0.0, c.ptr, 1);
    check(download(h, c.ptr, 1)[0] == 36, "GEMM after stream replacement");
    DeviceBuffer<int> info(h, 1);
    potrf(h, 'L', 1, c.ptr, 1, info.ptr);
    check(download(h, info.ptr, 1)[0] == 0 && download(h, c.ptr, 1)[0] == 6,
          "POTRF after stream replacement");
}
void failed_binding(DdlaHandle_t h, Trace& trace, int fault)
{
    void* old = ddlaGetStream(h);
    auto candidate = create_stream();
    const auto before = trace.read();
    trace.fail(fault);
    bool threw = false;
    try { DDLA_CHECK(ddlaSetStream(h, candidate)); }
    catch (const std::runtime_error&) { threw = true; }
    const auto after = trace.read();
    check(threw, "one-shot native binding failure did not propagate");
    check(after.blas_faults - before.blas_faults == (fault == StreamBlasFault) &&
          after.solver_faults - before.solver_faults == (fault == StreamSolverFault),
          "native binding fault must fire exactly once");
    check(after.blas_binds - before.blas_binds == 2 &&
          after.solver_binds - before.solver_binds == (fault == StreamBlasFault ? 1u : 2u),
          "both original vendor bindings must be restored");
    check(after.destroys == before.destroys, "failed setter destroyed caller/handle stream");
    bindings(h, old);
    // Caller still owns this native stream after the failed takeover.
    GPU_CHECK(gpuStreamSynchronize(candidate));
    GPU_CHECK(gpuStreamDestroy(candidate));
    numerical_case(h);
}
// Deliberately uses only the longstanding successful setter path so the same
// executable can expose the original stream leak in a pre-fix library.
void ownership_control()
{
    Trace trace;
    trace.reset();
    int ranks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    DdlaHandle_t h = nullptr;
    DDLA_CHECK(ddlaInit(h, DdlaBackend::GPU));
    DDLA_CHECK(ddlaSet(h, MPI_COMM_WORLD, 1, ranks));
    auto replacement = create_stream();
    DDLA_CHECK(ddlaSetStream(h, replacement));
    DDLA_CHECK(ddlaDestroy(h));
    const auto final = trace.read();
    std::cout << "OWNERSHIP_CONTROL creates=" << final.creates
              << " destroys=" << final.destroys << std::endl;
    check(final.creates >= 3, "ownership control preload did not observe native streams");
    check(final.creates == final.destroys,
          "OWNERSHIP_CONTROL_LEAK: replaced owned stream was not destroyed");
}
void gpu_cases()
{
    Trace trace;
    trace.reset();
    cpu_cases();
    check(trace.read().synchronizes == 0, "CPU cleanup touched GPU streams");
    DdlaHandle_t h = nullptr;
    DDLA_CHECK(ddlaInit(h, DdlaBackend::GPU));
    check(ddlaSetStream(h, nullptr) == ddlaStatus_t::DDLA_STATUS_INVALID_VALUE,
          "uninitialized GPU setter");
    DDLA_CHECK(ddlaDestroy(h));
    check(trace.read().synchronizes == 0, "uninitialized cleanup touched default stream");
    int ranks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    DDLA_CHECK(ddlaInit(h, DdlaBackend::GPU));
    DDLA_CHECK(ddlaSet(h, MPI_COMM_WORLD, 1, ranks));
    void* level = h->level_queue();
    check(trace.read().creates == 3, "preload must observe three owned queues");
    bindings(h, ddlaGetStream(h));
    const auto same_before = trace.read();
    DDLA_CHECK(ddlaSetStream(h, ddlaGetStream(h)));
    const auto same_after = trace.read();
    check(same_after.synchronizes == same_before.synchronizes &&
          same_after.blas_binds == same_before.blas_binds &&
          same_after.solver_binds == same_before.solver_binds &&
          same_after.destroys == same_before.destroys, "same-stream setter must be a no-op");
    check(ddlaSetStream(h, h->stream_data) == ddlaStatus_t::DDLA_STATUS_INVALID_VALUE &&
          ddlaSetStream(h, level) == ddlaStatus_t::DDLA_STATUS_INVALID_VALUE,
          "private queue aliases must be rejected");
    auto candidate = create_stream();
    h->communicator->group_start(h);
    const auto grouped = ddlaSetStream(h, candidate);
    h->communicator->group_end(h);
    check(grouped == ddlaStatus_t::DDLA_STATUS_INVALID_VALUE, "active group stream replacement");
    GPU_CHECK(gpuStreamDestroy(candidate));
    failed_binding(h, trace, StreamBlasFault);
    failed_binding(h, trace, StreamSolverFault);
    for (void* queue : {ddlaGetStream(h), h->stream_data, level}) {
        auto next = create_stream(); // Create before enqueuing the blocking callback.
        const auto before = trace.read();
        {
            Pending pending(queue);
            DDLA_CHECK(ddlaSetStream(h, next));
            check(pending.finished.load(), "replacement did not finish prior queued work");
        }
        check(trace.read().destroys == before.destroys + 1, "replacement leaked old stream");
        bindings(h, next);
        numerical_case(h);
    }
    const auto before_default = trace.read();
    DDLA_CHECK(ddlaSetStream(h, nullptr));
    check(trace.read().destroys == before_default.destroys + 1, "default selection leaked old stream");
    bindings(h, nullptr);
    numerical_case(h);
    const auto before_destroy = trace.read();
    {
        Pending pending(nullptr);
        DDLA_CHECK(ddlaDestroy(h));
        check(pending.finished.load(), "default stream work outlived handle teardown");
    }
    const auto final = trace.read();
    check(final.default_synchronizes == before_destroy.default_synchronizes + 1,
          "teardown must explicitly drain selected default stream");
    check(final.default_destroys == 0, "native default stream must never be destroyed");
    check(final.creates == final.destroys, "native stream create/destroy imbalance");
    check(final.blas_faults == 1 && final.solver_faults == 1, "native fault coverage");
}
#endif
}
int main(int argc, char** argv)
{
    ddla_test_mpi_init(&argc, &argv);
    try {
        cpu_cases();
#if DDLA_HAS_GPU
        if (argc == 2 && std::strcmp(argv[1], "--ownership-control") == 0)
            ownership_control();
        else
            gpu_cases();
#endif
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (!rank) std::cout << "STREAM_OWNERSHIP_PASS\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
}
