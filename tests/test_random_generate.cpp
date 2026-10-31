#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <mpi.h>

#include <ddla/ddla.h>
#include "ddla_connector.h"
#include "ddla_stream.h"

using namespace ddla;

//
// Helper: validate that every underlying real component of the generated
// data falls in [0, 1] (the gpurand uniform range).
//

template <typename T>
bool validate_uniform_range(const std::vector<T>& host_data, size_t count,
                            int myid, const char* type_name)
{
    bool ok = true;
    const double lo = 0.0;
    const double hi = 1.0;

    if constexpr (std::is_same_v<T, float>) {
        for (size_t i = 0; i < count; ++i) {
            double v = static_cast<double>(host_data[i]);
            if (!std::isfinite(v) || v < lo || v > hi) {
                fprintf(stderr, "[rank %d] %s[%zu] = %g  out of [%g,%g]\n",
                        myid, type_name, i, v, lo, hi);
                ok = false;
                break;
            }
        }
    } else if constexpr (std::is_same_v<T, double>) {
        for (size_t i = 0; i < count; ++i) {
            double v = host_data[i];
            if (!std::isfinite(v) || v < lo || v > hi) {
                fprintf(stderr, "[rank %d] %s[%zu] = %g  out of [%g,%g]\n",
                        myid, type_name, i, v, lo, hi);
                ok = false;
                break;
            }
        }
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        for (size_t i = 0; i < count; ++i) {
            float r = host_data[i].real();
            float im = host_data[i].imag();
            if (!std::isfinite(r) || r < lo || r > hi ||
                !std::isfinite(im) || im < lo || im > hi) {
                fprintf(stderr, "[rank %d] %s[%zu] = (%g,%g)  out of [%g,%g]\n",
                        myid, type_name, i, (double)r, (double)im, lo, hi);
                ok = false;
                break;
            }
        }
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        for (size_t i = 0; i < count; ++i) {
            double r = host_data[i].real();
            double im = host_data[i].imag();
            if (!std::isfinite(r) || r < lo || r > hi ||
                !std::isfinite(im) || im < lo || im > hi) {
                fprintf(stderr, "[rank %d] %s[%zu] = (%g,%g)  out of [%g,%g]\n",
                        myid, type_name, i, r, im, lo, hi);
                ok = false;
                break;
            }
        }
    }
    return ok;
}

//
// Test one scalar type on one backend: allocate where the handle's backend keeps memory (device memory on a GPU
// handle, host memory on a CPU handle), generate, copy back and validate.  The values must also differ from one
// element to the next (a constant fill is in [0, 1] too).
//

template <DdlaBackend Backend, typename T>
bool test_type(int64_t count, const DdlaHandle_t& handle, const char* type_name)
{
    int myid;
    MPI_Comm_rank(handle->comm, &myid);

    const size_t bytes = static_cast<size_t>(count) * sizeof(T);

    T* d_data = nullptr;
    DDLA_CHECK(ddlaMalloc(handle, &d_data, bytes));

    // Pre-fill with a value outside [0, 1], so an element the generator never wrote is noticed.
    std::vector<T> sentinel(count, T(2));
    DDLA_CHECK(ddlaMemcpy(handle, d_data, sentinel.data(), bytes));
    DDLA_CHECK(ddlaSynchronize(handle));

    random_generate<Backend>(handle, d_data, count);

    std::vector<T> h_data(count);
    DDLA_CHECK(ddlaMemcpy(handle, h_data.data(), d_data, bytes));
    DDLA_CHECK(ddlaSynchronize(handle));

    bool ok = validate_uniform_range(h_data, count, myid, type_name);
    bool varies = false;
    for (int64_t i = 1; i < count; ++i) {
        if (!(h_data[i] == h_data[0])) {
            varies = true;
            break;
        }
    }
    if (!varies) {
        fprintf(stderr, "[rank %d] %s: all %lld generated values are equal\n", myid, type_name, (long long)count);
        ok = false;
    }

    DDLA_CHECK(ddlaFree(handle, d_data));
    return ok;
}

//
// Zero-length call: must not crash or generate; nullptr is allowed.
//

template <DdlaBackend Backend>
bool test_zero_length(const DdlaHandle_t& handle)
{
    // Zero-length, null pointer -- a safe no-op.
    random_generate<Backend>(handle, static_cast<std::complex<double>*>(nullptr), int64_t(0));

    // Zero-length, non-null dummy pointer -- also a safe no-op.
    std::complex<double> dummy(-3.0, -4.0);
    random_generate<Backend>(handle, &dummy, int64_t(0));
    return dummy == std::complex<double>(-3.0, -4.0);
}

//
// Argument checks: null handle, null data for a nonzero count, and (when the other backend's handle is at hand) a
// backend mismatch.
//

template <DdlaBackend Backend>
bool test_rejections(const DdlaHandle_t& handle, const DdlaHandle_t& other)
{
    bool ok = true;
    auto expect_throw = [&](const std::function<void()>& fn, const char* what) {
        bool threw = false;
        try { fn(); }
        catch (const std::runtime_error& e) {
            threw = std::string(e.what()).find(what) != std::string::npos;
        }
        if (!threw) {
            fprintf(stderr, "random_generate: expected an exception containing \"%s\"\n", what);
            ok = false;
        }
    };
    double x[4] = {0, 0, 0, 0};
    expect_throw([&] { random_generate<Backend>(nullptr, x, int64_t(4)); }, "null handle");
    expect_throw([&] { random_generate<Backend>(handle, static_cast<double*>(nullptr), int64_t(4)); }, "null data");
    if (other != nullptr)
        expect_throw([&] { random_generate<Backend>(other, x, int64_t(4)); }, "does not match");
    return ok;
}

template <DdlaBackend Backend>
bool test_backend(const DdlaHandle_t& handle, const DdlaHandle_t& other, int myid)
{
    const int64_t count = 1024;
    bool ok = true;
    if (myid == 0) printf("--- Testing %s float ---\n", Backend == DdlaBackend::CPU ? "CPU" : "GPU");
    ok = test_type<Backend, float>(count, handle, "float") && ok;
    if (myid == 0) printf("--- Testing %s double ---\n", Backend == DdlaBackend::CPU ? "CPU" : "GPU");
    ok = test_type<Backend, double>(count, handle, "double") && ok;
    if (myid == 0) printf("--- Testing %s std::complex<float> ---\n", Backend == DdlaBackend::CPU ? "CPU" : "GPU");
    ok = test_type<Backend, std::complex<float>>(count, handle, "std::complex<float>") && ok;
    if (myid == 0) printf("--- Testing %s std::complex<double> ---\n", Backend == DdlaBackend::CPU ? "CPU" : "GPU");
    ok = test_type<Backend, std::complex<double>>(count, handle, "std::complex<double>") && ok;

    if (myid == 0) printf("--- Testing %s zero-length no-op ---\n", Backend == DdlaBackend::CPU ? "CPU" : "GPU");
    ok = test_zero_length<Backend>(handle) && ok;

    if (myid == 0) printf("--- Testing %s argument checks ---\n", Backend == DdlaBackend::CPU ? "CPU" : "GPU");
    ok = test_rejections<Backend>(handle, other) && ok;
    return ok;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // One handle per available backend: random_generate takes the handle first and the backend must match it.
    DdlaHandle_t cpu = nullptr;
    DdlaHandle_t gpu = nullptr;
#if DDLA_HAS_CPU
    ddlaInit(cpu, DdlaBackend::CPU);
    ddlaSet(cpu, MPI_COMM_WORLD);
#endif
#if DDLA_HAS_GPU
    ddlaInit(gpu, DdlaBackend::GPU);
    ddlaSet(gpu, MPI_COMM_WORLD);
#endif

    bool all_ok = true;
#if DDLA_HAS_GPU
    all_ok = test_backend<DdlaBackend::GPU>(gpu, cpu, myid) && all_ok;
#endif
#if DDLA_HAS_CPU
    all_ok = test_backend<DdlaBackend::CPU>(cpu, gpu, myid) && all_ok;
#endif

    // Aggregate failure across all ranks
    int local_ok = all_ok ? 1 : 0;
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (cpu != nullptr) ddlaDestroy(cpu);
    if (gpu != nullptr) ddlaDestroy(gpu);

    if (myid == 0) {
        if (!global_ok) {
            printf("test_random_generate: FAILED\n");
        } else {
            printf("test_random_generate: PASSED\n");
        }
    }

    MPI_Finalize();
    return global_ok ? 0 : 1;
}
