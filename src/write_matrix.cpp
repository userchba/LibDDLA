#include "write_matrix.h"

#include <cmath>
#include <complex>
#include <fstream>
#include <type_traits>
#include <memory>

namespace ddla {

namespace {

template <typename T>
struct is_std_complex : std::false_type {};
template <typename U>
struct is_std_complex<std::complex<U>> : std::true_type {};

// Flush denormal-ish noise to an exact 0 so successive runs diff cleanly.
template <typename R>
inline R flush_tiny(R v) { return (std::abs(v) < static_cast<R>(1e-10)) ? static_cast<R>(0) : v; }

template <typename T>
void write_host_matrix(const T* A, int m, int n, std::ostream& outfile)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            const T v = A[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m];
            if constexpr (is_std_complex<T>::value) {
                outfile << "(" << flush_tiny(v.real()) << "," << flush_tiny(v.imag()) << ") ";
            } else {
                outfile << flush_tiny(v) << " ";
            }
        }
        outfile << "\n";
    }
}

} // anonymous namespace

namespace matrix_io_impl {
template<DdlaBackend Backend,class T>
void write_matrix_run(const T* A,int m,int n,std::ostream& output,BufferView<T> host_staging)
{
    if constexpr(Backend==DdlaBackend::CPU) {
        write_host_matrix(A,m,n,output);
    } else {
#if DDLA_HAS_GPU
        // This utility has no handle/stream. Finish the D2H read before the
        // formatter accesses staging and before the caller can reuse it.
        const auto count=std::size_t(m)*std::size_t(n);
        GPU_CHECK(gpuMemcpy(host_staging.data(),A,count*sizeof(T),gpuMemcpyDeviceToHost));
        write_host_matrix(host_staging.data(),m,n,output);
#endif
    }
}
} // namespace matrix_io_impl

template <DdlaBackend Backend, typename T>
void write_matrix(const T* A, int m, int n, const char* filename)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU write_matrix is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU write_matrix is not available in this LibDDLA build");
    if(!matrix_io_impl::write_matrix_check(A,m,n,filename))return;
    const auto query=matrix_io_impl::write_matrix_workspace_size<Backend,T>(m,n);
    const auto count=query.host_bytes/sizeof(T);
    std::unique_ptr<T[]> host;
    if(count)host.reset(new T[count]);
    std::ofstream output(filename,std::ios::out|std::ios::trunc);
    if(!output)return;
    matrix_io_impl::write_matrix_run<Backend,T>(A,m,n,output,{host.get(),count});
}

#define INSTANTIATE_WRITE_MATRIX(BACKEND, TYPE)                              \
    template void write_matrix<BACKEND, TYPE>(                               \
        const TYPE*, int, int, const char*);                            \
    template void matrix_io_impl::write_matrix_run<BACKEND, TYPE>(             \
        const TYPE*, int, int, std::ostream&, BufferView<TYPE>)

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
