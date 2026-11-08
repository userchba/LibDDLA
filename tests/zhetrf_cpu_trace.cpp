// Test-only CPU LAPACK interposition. The numerical library has no hooks.
#include <atomic>
#include <complex>
#include <dlfcn.h>
#include <stdexcept>
namespace
{
std::atomic<long long> calls{0};
}
extern "C" void ddla_zhetrf_trace_reset()
{
    calls = 0;
}
extern "C" long long ddla_zhetrf_trace_count()
{
    return calls.load();
}
extern "C" void zhetrf_(const char* u, const int* n, std::complex<double>* a, const int* ld,
                        int* piv, std::complex<double>* w, const int* lw, int* info)
{
    using Function = void (*)(const char*, const int*, std::complex<double>*, const int*, int*,
                              std::complex<double>*, const int*, int*);
    static auto original = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "zhetrf_"));
    if (!original)
        throw std::runtime_error("Cannot resolve baseline LAPACK zhetrf_");
    ++calls;
    original(u, n, a, ld, piv, w, lw, info);
}

// MAGMA's original small-block fallback calls HETF2 directly.
extern "C" void zhetf2_(const char* u, const int* n, std::complex<double>* a, const int* ld,
                        int* piv, int* info)
{
    using Function =
        void (*)(const char*, const int*, std::complex<double>*, const int*, int*, int*);
    static auto original = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "zhetf2_"));
    if (!original)
        throw std::runtime_error("Cannot resolve baseline LAPACK zhetf2_");
    ++calls;
    original(u, n, a, ld, piv, info);
}
