#include <ddla/ddla.h>
#include "ddla_desc.h"
#include "ddla_connector.h"
#include "ddla_stream.h"
#include "helpers.h"
#include "require_backend.h"
#include "check_agree.h"
#include <complex>
#include <stdexcept>
#include <type_traits>

namespace ddla {

#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)

/**
 * @brief Device kernel: add alpha to each globally owned diagonal element.
 *
 * The global diagonal index i maps to a 2D block-cyclic local element via
 * the standard ScaLAPACK formulas.  Each thread checks ownership and updates
 * A[ilo + jlo*lld] in place.
 */
template <typename T1, typename T2>
__global__ void pdam_kernel(const T1 alpha, T2* A,
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

    A[ilo + jlo * lld] += alpha;
}

#endif // DDLA_USE_CUDA || DDLA_USE_HIP

namespace pdam_impl {

// Component 1.  Collective: every rank throws if any rank does.  An empty
// order is a no-op and is accepted before the backend is looked at.
template <DdlaBackend Backend>
int pdam_check(const DdlaHandle_t& handle, int n, const int* array_descA)
{
    return detail::agree_check(handle, "pdam", [&](detail::Agree&) {
        check_desc(array_descA, handle);
        // Logical sub-matrix order: n < 0 means the whole matrix (descriptor dims).
        const int n_eff = (n < 0) ? array_descA[DDLA_M_] : n;
        if (n_eff > array_descA[DDLA_M_] || n_eff > array_descA[DDLA_N_])
            throw std::invalid_argument("pdam: n exceeds the descriptor dimensions");
        if (n_eff > 0) detail::require_backend<Backend>(handle, "pdam");
        return n_eff;
    });
}

// Scratch-free execution: alpha reaches the kernel by value.
template <DdlaBackend Backend, typename T1, typename T2>
void pdam_run(const DdlaHandle_t& handle, int n_eff, const T1& alpha, T2* A, const int* array_descA)
{
    if(n_eff<=0)return;
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);

    if constexpr (Backend == DdlaBackend::CPU) {
        // the same ownership test and block-cyclic local index as pdam_kernel, one global diagonal element at a time
        const int mb = array_descA[DDLA_MB_], nb = array_descA[DDLA_NB_];
        const int irsrc = array_descA[DDLA_RSRC_], icsrc = array_descA[DDLA_CSRC_];
        const int lld = array_descA[DDLA_LLD_];
        for (int i = 0; i < n_eff; ++i) {
            if ((irsrc + i / mb) % nprows != myprow) continue;
            if ((icsrc + i / nb) % npcols != mypcol) continue;
            const int ilo = mb * (i / (mb * nprows)) + i % mb;
            const int jlo = nb * (i / (nb * npcols)) + i % nb;
            A[ilo + static_cast<std::size_t>(jlo) * lld] += alpha;
        }
    } else {
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
        const gpuStream_t stream = static_cast<gpuStream_t>(handle->stream);

        using deviceT1 = detail::DeviceScalarT<T1>;
        using deviceT2 = detail::DeviceScalarT<T2>;

        const int blockSize = 256;
        const int gridSize = (n_eff + blockSize - 1) / blockSize;

        deviceT2* d_A_dev = reinterpret_cast<deviceT2*>(A);
        pdam_kernel<deviceT1, deviceT2><<<gridSize, blockSize, 0, stream>>>(
            detail::to_device_scalar(alpha), d_A_dev,
            n_eff,
            array_descA[DDLA_MB_], array_descA[DDLA_NB_],
            array_descA[DDLA_RSRC_], array_descA[DDLA_CSRC_],
            myprow, mypcol,
            nprows, npcols,
            array_descA[DDLA_LLD_]);
        DDLA_CHECK(ddlaGetLastError(handle));
#endif
    }
}

} // namespace pdam_impl

template <DdlaBackend Backend, typename T1, typename T2>
void pdam(const DdlaHandle_t& handle, const T1& alpha, T2* A, const int* array_descA, int n)
{
    const int order = pdam_impl::pdam_check<Backend>(handle, n, array_descA);
    if (order <= 0) return;
    pdam_impl::pdam_run<Backend, T1, T2>(handle, order, alpha, A, array_descA);
}

// Supported type combinations match LibRPA's DeviceConnector::pdam.
#define INSTANTIATE_PDAM(BACKEND, T1, T2) \
    template void pdam<BACKEND, T1, T2>(const DdlaHandle_t&, const T1&, T2*, const int*, int); \
    template void pdam_impl::pdam_run<BACKEND, T1, T2>(const DdlaHandle_t&, int, const T1&, T2*, const int*)

#define INSTANTIATE_PDAM_CHECK(BACKEND) \
    template int pdam_impl::pdam_check<BACKEND>(const DdlaHandle_t&, int, const int*)

#define INSTANTIATE_PDAM_ALL(BACKEND)                                        \
    INSTANTIATE_PDAM(BACKEND, float, float);                                 \
    INSTANTIATE_PDAM(BACKEND, double, double);                               \
    INSTANTIATE_PDAM(BACKEND, float, std::complex<float>);                   \
    INSTANTIATE_PDAM(BACKEND, std::complex<float>, std::complex<float>);     \
    INSTANTIATE_PDAM(BACKEND, double, std::complex<double>);                 \
    INSTANTIATE_PDAM(BACKEND, std::complex<double>, std::complex<double>)

#if DDLA_HAS_CPU
INSTANTIATE_PDAM_CHECK(DdlaBackend::CPU);
INSTANTIATE_PDAM_ALL(DdlaBackend::CPU);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_PDAM_CHECK(DdlaBackend::GPU);
INSTANTIATE_PDAM_ALL(DdlaBackend::GPU);
#endif

#undef INSTANTIATE_PDAM_CHECK
#undef INSTANTIATE_PDAM_ALL
#undef INSTANTIATE_PDAM

} // namespace ddla
