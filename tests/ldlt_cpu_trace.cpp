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

extern "C" void ssytrf_(const char* u, const int* n, float* a, const int* lda, int* piv,
                        float* work, const int* lwork, int* info)
{
    using F = void (*)(const char*, const int*, float*, const int*, int*, float*, const int*, int*);
    static auto original = reinterpret_cast<F>(dlsym(RTLD_NEXT, "ssytrf_"));
    if (!original)
        throw std::runtime_error("CPU trace cannot resolve ssytrf_");
    ++calls;
    original(u, n, a, lda, piv, work, lwork, info);
}

extern "C" void dsytrf_(const char* u, const int* n, double* a, const int* lda, int* piv,
                        double* work, const int* lwork, int* info)
{
    using F =
        void (*)(const char*, const int*, double*, const int*, int*, double*, const int*, int*);
    static auto original = reinterpret_cast<F>(dlsym(RTLD_NEXT, "dsytrf_"));
    if (!original)
        throw std::runtime_error("CPU trace cannot resolve dsytrf_");
    ++calls;
    original(u, n, a, lda, piv, work, lwork, info);
}

extern "C" void chetrf_(const char* u, const int* n, std::complex<float>* a, const int* lda,
                        int* piv, std::complex<float>* work, const int* lwork, int* info)
{
    using F = void (*)(const char*, const int*, std::complex<float>*, const int*, int*,
                       std::complex<float>*, const int*, int*);
    static auto original = reinterpret_cast<F>(dlsym(RTLD_NEXT, "chetrf_"));
    if (!original)
        throw std::runtime_error("CPU trace cannot resolve chetrf_");
    ++calls;
    original(u, n, a, lda, piv, work, lwork, info);
}

extern "C" void ssyrk_(const char* u, const char* trans, const int* n, const int* k,
                       const float* alpha, const float* a, const int* lda, const float* beta,
                       float* c, const int* ldc)
{
    using F = void (*)(const char*, const char*, const int*, const int*, const float*, const float*,
                       const int*, const float*, float*, const int*);
    static auto original = reinterpret_cast<F>(dlsym(RTLD_NEXT, "ssyrk_"));
    if (!original)
        throw std::runtime_error("CPU trace cannot resolve ssyrk_");
    ++calls;
    original(u, trans, n, k, alpha, a, lda, beta, c, ldc);
}

extern "C" void dsyrk_(const char* u, const char* trans, const int* n, const int* k,
                       const double* alpha, const double* a, const int* lda, const double* beta,
                       double* c, const int* ldc)
{
    using F = void (*)(const char*, const char*, const int*, const int*, const double*,
                       const double*, const int*, const double*, double*, const int*);
    static auto original = reinterpret_cast<F>(dlsym(RTLD_NEXT, "dsyrk_"));
    if (!original)
        throw std::runtime_error("CPU trace cannot resolve dsyrk_");
    ++calls;
    original(u, trans, n, k, alpha, a, lda, beta, c, ldc);
}

extern "C" void cherk_(const char* u, const char* trans, const int* n, const int* k,
                       const float* alpha, const std::complex<float>* a, const int* lda,
                       const float* beta, std::complex<float>* c, const int* ldc)
{
    using F = void (*)(const char*, const char*, const int*, const int*, const float*,
                       const std::complex<float>*, const int*, const float*, std::complex<float>*,
                       const int*);
    static auto original = reinterpret_cast<F>(dlsym(RTLD_NEXT, "cherk_"));
    if (!original)
        throw std::runtime_error("CPU trace cannot resolve cherk_");
    ++calls;
    original(u, trans, n, k, alpha, a, lda, beta, c, ldc);
}

extern "C" void zherk_(const char* u, const char* trans, const int* n, const int* k,
                       const double* alpha, const std::complex<double>* a, const int* lda,
                       const double* beta, std::complex<double>* c, const int* ldc)
{
    using F = void (*)(const char*, const char*, const int*, const int*, const double*,
                       const std::complex<double>*, const int*, const double*,
                       std::complex<double>*, const int*);
    static auto original = reinterpret_cast<F>(dlsym(RTLD_NEXT, "zherk_"));
    if (!original)
        throw std::runtime_error("CPU trace cannot resolve zherk_");
    ++calls;
    original(u, trans, n, k, alpha, a, lda, beta, c, ldc);
}
