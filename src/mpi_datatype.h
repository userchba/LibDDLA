#ifndef DDLA_MPI_DATATYPE_H
#define DDLA_MPI_DATATYPE_H

// ---------------------------------------------------------------------------
// Private header: MPI_Datatype for each DDLA scalar type. Shared by any
// translation unit that issues MPI collectives or point-to-point calls.
// Always pass an element count with the matching MPI_Datatype here -- never
// a byte count with MPI_BYTE, which silently overflows `int` well before a
// realistic panel size and forgoes MPI's own type checking (F6).
// ---------------------------------------------------------------------------

#include <mpi.h>
#include <complex>
#if defined(DDLA_USE_GPU_CPU_TUNNEL) && (defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP))
#include <thrust/complex.h>
#endif

namespace ddla {
namespace detail {

template <typename T>
static inline MPI_Datatype mpi_datatype();

template <> inline MPI_Datatype mpi_datatype<int>() { return MPI_INT; }
template <> inline MPI_Datatype mpi_datatype<float>() { return MPI_FLOAT; }
template <> inline MPI_Datatype mpi_datatype<double>() { return MPI_DOUBLE; }
template <> inline MPI_Datatype mpi_datatype<std::complex<float>>() { return MPI_CXX_FLOAT_COMPLEX; }
template <> inline MPI_Datatype mpi_datatype<std::complex<double>>() { return MPI_CXX_DOUBLE_COMPLEX; }

#if defined(DDLA_USE_GPU_CPU_TUNNEL) && (defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP))
// Device packing kernels also produce thrust complex panels. Host staging
// preserves their two-real layout, so MPI uses the matching complex type.
template <> inline MPI_Datatype mpi_datatype<thrust::complex<float>>() {
    static_assert(sizeof(thrust::complex<float>) == sizeof(std::complex<float>));
    return MPI_CXX_FLOAT_COMPLEX;
}
template <> inline MPI_Datatype mpi_datatype<thrust::complex<double>>() {
    static_assert(sizeof(thrust::complex<double>) == sizeof(std::complex<double>));
    return MPI_CXX_DOUBLE_COMPLEX;
}
#endif

} // namespace detail
} // namespace ddla

#endif // DDLA_MPI_DATATYPE_H
