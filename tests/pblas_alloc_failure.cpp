// Test-only LD_PRELOAD allocator failures. Never linked into LibDDLA.
#include <ddla/ddla_handle_t.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <execinfo.h>
#include <new>
#include <unistd.h>

namespace {
std::atomic<int> armed{0}; // 1: ddlaMalloc (backend storage), 2: C++ host storage
std::atomic<unsigned> fired{0};
std::atomic<std::size_t> failed_bytes{0};
thread_local unsigned resolving = 0;
const char* own_path = nullptr;
struct ResolutionGuard {
    ResolutionGuard() { ++resolving; }
    ~ResolutionGuard() { --resolving; }
};
template <typename F> F next_own(F self)
{
    ResolutionGuard guard;
    void* address = nullptr;
    static_assert(sizeof(address) == sizeof(self), "function pointer ABI");
    std::memcpy(&address, &self, sizeof(address));
    Dl_info info{};
    if (!dladdr(address, &info) || !info.dli_sname) return nullptr;
    address = dlsym(RTLD_NEXT, info.dli_sname);
    F result = nullptr;
    std::memcpy(&result, &address, sizeof(result));
    return result;
}
bool library_caller()
{
    if (resolving) return false;
    ResolutionGuard guard;
    void* frames[32];
    const int count = backtrace(frames, 32);
    for (int i = 1; i < count; ++i) {
        Dl_info info{};
        if (!dladdr(frames[i], &info) || !info.dli_fname) continue;
        if (own_path && std::strcmp(info.dli_fname, own_path) == 0) continue;
        const char* file = std::strrchr(info.dli_fname, '/');
        file = file ? file + 1 : info.dli_fname;
        if (std::strstr(file, "libstdc++") || std::strstr(file, "libc++") ||
            std::strstr(file, "libgcc_s") || std::strstr(file, "libunwind") ||
            std::strncmp(file, "libc.so", 7) == 0 ||
            std::strncmp(file, "ld-linux", 8) == 0) continue;
        return std::strstr(file, "libddla.") || std::strstr(file, "libddla-");
    }
    return false;
}
bool fail(int kind, std::size_t bytes)
{
    if (resolving || !bytes || armed.load(std::memory_order_relaxed) != kind) return false;
    int expected = kind;
    if (!armed.compare_exchange_strong(expected, 0)) return false;
    // Disarm before constructing the exception or entering the caller's handler.
    failed_bytes.store(bytes);
    fired.fetch_add(1);
    char message[128];
    const int n = std::snprintf(message, sizeof(message),
        "PBLAS_ALLOCATION_FAULT kind=%d bytes=%zu\n", kind, bytes);
    if (n > 0) (void)write(STDERR_FILENO, message, std::size_t(n));
    return true;
}
void* allocate(std::size_t size, std::size_t alignment = 0)
{
    if (armed.load(std::memory_order_relaxed) == 2 && library_caller() && fail(2, size))
        throw std::bad_alloc();
    for (;;) {
        void* result = nullptr;
        if (alignment) {
            if (posix_memalign(&result, alignment, size ? size : 1) != 0) result = nullptr;
        } else result = std::malloc(size ? size : 1);
        if (result) return result;
        const auto handler = std::get_new_handler();
        if (!handler) throw std::bad_alloc();
        handler();
    }
}
}

void* operator new(std::size_t n) { return allocate(n); }
void* operator new[](std::size_t n) { return allocate(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept { try { return allocate(n); } catch (...) { return nullptr; } }
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept { try { return allocate(n); } catch (...) { return nullptr; } }
void* operator new(std::size_t n, std::align_val_t a) { return allocate(n, std::size_t(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return allocate(n, std::size_t(a)); }
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept { try { return allocate(n, std::size_t(a)); } catch (...) { return nullptr; } }
void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept { try { return allocate(n, std::size_t(a)); } catch (...) { return nullptr; } }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }

namespace ddla {
using MallocFn = ddlaStatus_t (*)(const DdlaHandle_t&, void**, std::size_t);
static MallocFn real_malloc()
{
    static MallocFn fn = next_own<MallocFn>(&ddlaMalloc);
    return fn;
}
ddlaStatus_t ddlaMalloc(const DdlaHandle_t& h, void** pointer, std::size_t bytes)
{
    if (fail(1, bytes)) {
        *pointer = nullptr;
        return ddlaStatus_t::DDLA_STATUS_ALLOC_FAILED;
    }
    const auto fn = real_malloc();
    return fn ? fn(h, pointer, bytes) : ddlaStatus_t::DDLA_STATUS_INTERNAL_ERROR;
}
}

extern "C" int pblas_alloc_failure_prewarm()
{
    ResolutionGuard guard;
    Dl_info info{};
    // Resolve stack walking and symbol lookup before enabling failures.
    void* frames[32];
    (void)backtrace(frames, 32);
    if (dladdr(frames[0], &info)) own_path = info.dli_fname;
    return ddla::real_malloc() && own_path ? 1 : 0;
}
extern "C" void pblas_alloc_failure_arm(int kind)
{
    fired.store(0);
    failed_bytes.store(0);
    armed.store(kind);
}
extern "C" void pblas_alloc_failure_disarm() { armed.store(0); }
extern "C" unsigned pblas_alloc_failure_fired() { return fired.load(); }
extern "C" std::size_t pblas_alloc_failure_bytes() { return failed_bytes.load(); }
