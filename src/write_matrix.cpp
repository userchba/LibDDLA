#include <ddla/write_matrix.h>

#include <cmath>
#include <complex>
#include <fstream>
#include <type_traits>
#include <vector>

namespace ddla {

namespace {

// Flush denormal-ish noise to an exact 0 so successive runs diff cleanly.
template <typename R>
inline R flush_tiny(R v) { return (std::abs(v) < static_cast<R>(1e-10)) ? static_cast<R>(0) : v; }

// Scalar-type dispatch for value formatting. C++11 has no `if constexpr`:
// complex values print as (re,im), real values directly. The std::complex<>
// overload wins by overload resolution, so the `v.real()` call below is only
// ever instantiated for complex T.
template <typename R>
void write_value(std::ostream& out, const std::complex<R>& v)
{
    out << "(" << flush_tiny(v.real()) << "," << flush_tiny(v.imag()) << ") ";
}

template <typename T>
void write_value(std::ostream& out, const T& v)
{
    out << flush_tiny(v) << " ";
}

template <typename T>
void write_host_matrix(const T* A, int m, int n, const char* filename)
{
    std::ofstream outfile;
    outfile.open(filename, std::ios::out | std::ios::trunc);
    if (!outfile) return;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            const T v = A[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m];
            write_value(outfile, v);
        }
        outfile << "\n";
    }
    outfile.close();
}

} // anonymous namespace

template <DdlaBackend Backend, typename T>
void write_matrix(const T* A, const int& m, const int& n, const char* filename)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU write_matrix is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU write_matrix is not available in this LibDDLA build");

    if (A == nullptr || m <= 0 || n <= 0 || filename == nullptr) return;

    if (Backend == DdlaBackend::CPU) {
        write_host_matrix(A, m, n, filename);
    } else {
#if DDLA_HAS_GPU
        // Stage the device buffer back to the host, then reuse the same
        // formatter. A synchronous runtimeMemcpy is used deliberately: this is
        // a debugging dump, so a stream-ordered copy would only add a
        // synchronisation the caller then has to remember to perform.
        const std::size_t count = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
        std::vector<T> host(count);
        RUNTIME_CHECK<Backend>(runtimeMemcpy<Backend>(
            host.data(), A, count * sizeof(T),
            detail::RuntimeTraits<Backend>::device_to_host));
        write_host_matrix(host.data(), m, n, filename);
#endif
    }
}

#define INSTANTIATE_WRITE_MATRIX(BACKEND, TYPE)                              \
    template void write_matrix<BACKEND, TYPE>(                               \
        const TYPE*, const int&, const int&, const char*)

#if DDLA_HAS_CPU
INSTANTIATE_WRITE_MATRIX(DdlaBackend::CPU, float);
INSTANTIATE_WRITE_MATRIX(DdlaBackend::CPU, double);
INSTANTIATE_WRITE_MATRIX(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_WRITE_MATRIX(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_WRITE_MATRIX(DdlaBackend::GPU, float);
INSTANTIATE_WRITE_MATRIX(DdlaBackend::GPU, double);
INSTANTIATE_WRITE_MATRIX(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_WRITE_MATRIX(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_WRITE_MATRIX

} // namespace ddla
