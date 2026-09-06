#include <ddla/ddla.h>
#include "ddla_desc.h"
#include "ddla_connector.h"
#include "ddla_stream_impl.h"
#include "require_gpu.h"
#include <thrust/complex.h>
#include <type_traits>
#include <cassert>

namespace ddla {

namespace detail {

// Map std::complex to thrust::complex for device code, keep real types as-is.
template <typename T>
struct device_scalar {
    using type = T;
};

template <>
struct device_scalar<std::complex<float>> {
    using type = thrust::complex<float>;
};

template <>
struct device_scalar<std::complex<double>> {
    using type = thrust::complex<double>;
};

} // namespace detail

/**
 * @brief Device kernel: add alpha to each globally owned diagonal element.
 *
 * The global diagonal index i maps to a 2D block-cyclic local element via
 * the standard ScaLAPACK formulas.  Each thread checks ownership and updates
 * A[ilo + jlo*lld] in place.
 */
template <typename T1, typename T2>
__global__ void pdam_kernel(const T1* alpha, T2* A,
                            int n, int mb, int nb,
                            int irsrc, int icsrc,
                            int myprow, int mypcol,
                            int nprows, int npcols,
                            int lld)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    // ownership check for row i and column i
    const int row_owner = (irsrc + i / mb) % nprows;
    if (row_owner != myprow) return;

    const int col_owner = (icsrc + i / nb) % npcols;
    if (col_owner != mypcol) return;

    // local indices (ScaLAPACK block-cyclic mapping)
    const int ilo = mb * (i / (mb * nprows)) + i % mb;
    const int jlo = nb * (i / (nb * npcols)) + i % nb;

    A[ilo + jlo * lld] += *alpha;
}

template <typename T1, typename T2>
void pdam(const DdlaHandle_t& handle, const T1& alpha, T2* d_A, const int* array_descA, const int& n)
{
    check_desc(array_descA, handle);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);


    // Logical sub-matrix order: n < 0 means the whole matrix (descriptor dims).
    const int n_eff = (n < 0) ? array_descA[DDLA_M_] : n;
    assert(n_eff >= 0);
    assert(n_eff <= array_descA[DDLA_M_] && n_eff <= array_descA[DDLA_N_]);
    if (n_eff <= 0) return;

    detail::require_gpu_backend(handle, "pdam");
    runtimeStream_t stream = handle->stream;

    using deviceT1 = typename detail::device_scalar<T1>::type;
    using deviceT2 = typename detail::device_scalar<T2>::type;

    deviceT1* d_alpha = nullptr;
    RUNTIME_CHECK(runtimeMallocAsync(reinterpret_cast<void**>(&d_alpha), sizeof(deviceT1), stream));
    RUNTIME_CHECK(runtimeMemcpyAsync(d_alpha, &alpha, sizeof(deviceT1), runtimeMemcpyHostToDevice, stream));

    const int blockSize = 256;
    const int gridSize = (n_eff + blockSize - 1) / blockSize;

    deviceT2* d_A_dev = reinterpret_cast<deviceT2*>(d_A);
    pdam_kernel<deviceT1, deviceT2><<<gridSize, blockSize, 0, stream>>>(
        d_alpha, d_A_dev,
        n_eff,
        array_descA[DDLA_MB_], array_descA[DDLA_NB_],
        array_descA[DDLA_RSRC_], array_descA[DDLA_CSRC_],
        myprow, mypcol,
        nprows, npcols,
        array_descA[DDLA_LLD_]);
    RUNTIME_CHECK(runtimeGetLastError());

    RUNTIME_CHECK(runtimeFreeAsync(d_alpha, stream));
}

// Supported type combinations match LibRPA's DeviceConnector::pdam.
template void pdam<float, float>(const DdlaHandle_t&, const float&, float*, const int*, const int&);
template void pdam<double, double>(const DdlaHandle_t&, const double&, double*, const int*, const int&);
template void pdam<float, std::complex<float>>(const DdlaHandle_t&, const float&, std::complex<float>*, const int*, const int&);
template void pdam<std::complex<float>, std::complex<float>>(const DdlaHandle_t&, const std::complex<float>&, std::complex<float>*, const int*, const int&);
template void pdam<double, std::complex<double>>(const DdlaHandle_t&, const double&, std::complex<double>*, const int*, const int&);
template void pdam<std::complex<double>, std::complex<double>>(const DdlaHandle_t&, const std::complex<double>&, std::complex<double>*, const int*, const int&);

} // namespace ddla
