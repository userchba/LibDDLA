#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

#include <mpi.h>

#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include "ddla_stream_impl.h"

using namespace ddla;

//
// Helper: validate that every underlying real component of the generated
// data falls in [0, 1] (the derand uniform range).
//

// C++11 has no `if constexpr`: real and complex samples are validated by
// overloads on the scalar type -- the real overload handles float and double,
// the std::complex overload handles both complex precisions; overload
// resolution picks the right one at compile time.
template <typename T>
bool sample_in_range(const T& value, double lo, double hi,
                     int myid, const char* type_name, size_t i)
{
    const double v = static_cast<double>(value);
    if (std::isfinite(v) && v >= lo && v <= hi) return true;
    fprintf(stderr, "[rank %d] %s[%zu] = %g  out of [%g,%g]\n",
            myid, type_name, i, v, lo, hi);
    return false;
}

template <typename R>
bool sample_in_range(const std::complex<R>& value, double lo, double hi,
                     int myid, const char* type_name, size_t i)
{
    const double r = static_cast<double>(value.real());
    const double im = static_cast<double>(value.imag());
    if (std::isfinite(r) && r >= lo && r <= hi &&
        std::isfinite(im) && im >= lo && im <= hi) return true;
    fprintf(stderr, "[rank %d] %s[%zu] = (%g,%g)  out of [%g,%g]\n",
            myid, type_name, i, r, im, lo, hi);
    return false;
}

template <typename T>
bool validate_uniform_range(const std::vector<T>& host_data, size_t count,
                            int myid, const char* type_name)
{
    bool ok = true;
    const double lo = 0.0;
    const double hi = 1.0;

    for (size_t i = 0; i < count; ++i) {
        if (!sample_in_range(host_data[i], lo, hi, myid, type_name, i)) {
            ok = false;
            break;
        }
    }
    return ok;
}

#if DDLA_HAS_GPU
//
// Test one scalar type: allocate device memory, call random_generate,
// copy back and validate.
//

template <typename T>
bool test_type_gpu(int64_t count, const DdlaHandle_t& handle, const char* type_name)
{
    int myid;
    MPI_Comm_rank(handle->comm, &myid);

    const size_t bytes = static_cast<size_t>(count) * sizeof(T);

    // Allocate device memory
    T* d_data = nullptr;
    RUNTIME_CHECK(runtimeMalloc(&d_data, bytes));

    // Call the GPU backend
    random_generate<DdlaBackend::GPU>(d_data, count);

    // Copy back to host
    std::vector<T> h_data(count);
    RUNTIME_CHECK(runtimeMemcpy(h_data.data(), d_data, bytes, runtimeMemcpyDeviceToHost));

    // Validate
    bool ok = validate_uniform_range(h_data, count, myid, type_name);

    // Cleanup
    RUNTIME_CHECK(runtimeFree(d_data));

    return ok;
}

#endif

#if DDLA_HAS_CPU
//
// Test one scalar type on the CPU backend: fill a host vector directly and
// validate it on the host (no device staging needed).
//

template <typename T>
bool test_type_cpu(int64_t count, const DdlaHandle_t& handle, const char* type_name)
{
    int myid;
    MPI_Comm_rank(handle->comm, &myid);

    std::vector<T> h_data(static_cast<size_t>(count));
    random_generate<DdlaBackend::CPU>(h_data.data(), count);

    return validate_uniform_range(h_data, count, myid, type_name);
}
#endif

//
// Zero-length call: must not crash or generate; nullptr is allowed.
//

bool test_zero_length(const DdlaHandle_t& handle)
{
    int myid;
    MPI_Comm_rank(handle->comm, &myid);
    // Zero-length, null pointer – should be a safe no-op.
    random_generate((std::complex<double>*)nullptr, int64_t(0));

    // Zero-length, non-null dummy pointer – also safe no-op.
    std::complex<double> dummy;
    random_generate(&dummy, int64_t(0));

    if (myid == 0)
        printf("[rank %d] zero-length test passed\n", myid);
    return true;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    DdlaHandle_t handle;
    ddla_init(handle);
    ddla_set(handle);

    int myid;
    MPI_Comm_rank(handle->comm, &myid);

    bool all_ok = true;

    // ----------------------------------------------------------------
    // Test the four required types with a moderate element count.
    // ----------------------------------------------------------------

    const int64_t count = 1024;

#if DDLA_HAS_GPU
    if (myid == 0) printf("--- Testing float ---\n");
    all_ok = test_type_gpu<float>(count, handle, "float") && all_ok;

    if (myid == 0) printf("--- Testing double ---\n");
    all_ok = test_type_gpu<double>(count, handle, "double") && all_ok;

    if (myid == 0) printf("--- Testing std::complex<float> ---\n");
    all_ok = test_type_gpu<std::complex<float>>(count, handle, "std::complex<float>") && all_ok;

    if (myid == 0) printf("--- Testing std::complex<double> ---\n");
    all_ok = test_type_gpu<std::complex<double>>(count, handle, "std::complex<double>") && all_ok;
#endif


#if DDLA_HAS_CPU
    if (myid == 0) printf("--- Testing CPU float ---\n");
    all_ok = test_type_cpu<float>(count, handle, "float") && all_ok;

    if (myid == 0) printf("--- Testing CPU double ---\n");
    all_ok = test_type_cpu<double>(count, handle, "double") && all_ok;

    if (myid == 0) printf("--- Testing CPU std::complex<float> ---\n");
    all_ok = test_type_cpu<std::complex<float>>(count, handle, "std::complex<float>") && all_ok;

    if (myid == 0) printf("--- Testing CPU std::complex<double> ---\n");
    all_ok = test_type_cpu<std::complex<double>>(count, handle, "std::complex<double>") && all_ok;
#endif

    // Zero-length / null-pointer safety
    if (myid == 0) printf("--- Testing zero-length no-op ---\n");
    all_ok = test_zero_length(handle) && all_ok;

    // Aggregate failure across all ranks
    int local_ok = all_ok ? 1 : 0;
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, handle->comm);

    ddla_destroy(handle);

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
