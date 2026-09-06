#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>
#include "ddla_connector.h"
#include <ddla/ddla_config.h>
#include <ddla/ddla_desc.h>
#include "ddla_stream_impl.h"
#include <vector>
#include "transport_block.h"
#include "gemm.h"
#include "scal.h"

// CPU-only includes (no GPU vendor headers pulled in by these; safe in any
// build -- DDLA_HAS_CPU below still gates whether the CPU pgemm/transport
// paths are ever compiled).
#include <cstdlib>
#include <complex>
#include <new>

namespace ddla {

inline const char* pgemm_backend_name(DdlaBackend backend)
{
    return backend == DdlaBackend::CPU ? "CPU" : "GPU";
}


template <DdlaBackend Backend, typename T>
void pgemm(
    const DdlaHandle_t& handle, const char& transa, const char& transb,
    const int& m, const int& n, const int& k,
    const T& alpha,
    const T* A, const int* descA,
    const T* B, const int* descB,
    const T& beta,
    T* C, const int* descC)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU pgemm is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU pgemm is not available in this LibDDLA build");

    DdlaHandle_t h = handle;

    if (h == nullptr) {
        throw std::runtime_error("pgemm: null handle");
    }
    // Each descriptor is validated against this handle's process grid
    // (dense block-cyclic type, slot ranges, LLD vs LOCr); a descriptor
    // built against a different process grid is rejected there.
    check_desc(descA, handle);
    check_desc(descB, handle);
    check_desc(descC, handle);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);

    const DdlaBackend actual_backend = ddlaGetBackend(h);
    if (actual_backend != Backend) {
        throw std::runtime_error(
            std::string("pgemm: template backend ") + pgemm_backend_name(Backend) +
            " does not match descriptor handle backend " +
            pgemm_backend_name(actual_backend));
    }

    // ---- SUMMA implementation (double-buffered panel pipeline) -------------
    assert(transa == 'N' || transa == 'T' || transa == 'C');
    assert(transb == 'N' || transb == 'T' || transb == 'C');
    assert(m >= 0 && n >= 0 && k >= 0);

    const int opA_m = (transa == 'N') ? descA[DDLA_M_] : descA[DDLA_N_];
    const int opA_n = (transa == 'N') ? descA[DDLA_N_] : descA[DDLA_M_];
    const int opB_m = (transb == 'N') ? descB[DDLA_M_] : descB[DDLA_N_];
    const int opB_n = (transb == 'N') ? descB[DDLA_N_] : descB[DDLA_M_];
    assert(m <= opA_m);
    assert(k <= opA_n);
    assert(k <= opB_m);
    assert(n <= opB_n);
    assert(m <= descC[DDLA_M_] && n <= descC[DDLA_N_]);
    {
        int mbA, kbA, kbB, nbB, mbC, nbC;
        mbC = descC[DDLA_MB_];
        nbC = descC[DDLA_NB_];
        if (transa == 'N') {
            mbA = descA[DDLA_MB_];
            kbA = descA[DDLA_NB_];
        } else {
            mbA = descA[DDLA_NB_];
            kbA = descA[DDLA_MB_];
        }
        if (transb == 'N') {
            kbB = descB[DDLA_MB_];
            nbB = descB[DDLA_NB_];
        } else {
            kbB = descB[DDLA_NB_];
            nbB = descB[DDLA_MB_];
        }
        assert(mbA == mbC);
        assert(kbA == kbB);
        assert(nbB == nbC);
    }

    int nb;
    if (transa == 'N')
        nb = descA[DDLA_NB_];
    else
        nb = descA[DDLA_MB_];

    const int m_loc_C = num_loc(m, descC[DDLA_MB_], myprow, descC[DDLA_RSRC_], nprows);
    const int n_loc_C = num_loc(n, descC[DDLA_NB_], mypcol, descC[DDLA_CSRC_], npcols);
    const int lldC = descC[DDLA_LLD_];

    // ---- Degenerate-size fast paths (F2) -----------------------------------
    // ScaLAPACK's PxGEMM contract: an empty output (m==0 or n==0) is a no-op,
    // and k==0 reduces to C := beta*C with A/B untouched. Both cases used to
    // fall through the k-loop below, which never executes when k==0, so C
    // was silently left unscaled by beta -- this restores that contract for
    // both backends without ever allocating the SUMMA panel buffers.
    if (m == 0 || n == 0) {
        return;
    }
    if (k == 0) {
        // F2 fast path: C := beta*C via one unified ddla::scal<Backend,T>
        // call per column.
        if (m_loc_C > 0 && n_loc_C > 0) {
            for (int j = 0; j < n_loc_C; ++j) {
                ddla::scal<Backend, T>(
                    h, m_loc_C, beta,
                    C + static_cast<std::size_t>(j) * lldC, 1);
            }
        }
        return;
    }

    // Buffer sizing
    const int m_loc_A = num_loc((transa == 'N') ? m : k, descA[DDLA_MB_], myprow, descA[DDLA_RSRC_], nprows);
    const int n_loc_A = num_loc((transa == 'N') ? k : m, descA[DDLA_NB_], mypcol, descA[DDLA_CSRC_], npcols);
    const int m_loc_B = num_loc((transb == 'N') ? k : n, descB[DDLA_MB_], myprow, descB[DDLA_RSRC_], nprows);
    const int n_loc_B = num_loc((transb == 'N') ? n : k, descB[DDLA_NB_], mypcol, descB[DDLA_CSRC_], npcols);

    // Scratch buffers (GPU stream-scratch or CPU heap) for the SUMMA panel
    // pipeline, freed explicitly below.  RUNTIME_CHECK throws std::runtime_error
    // on failure (uncaught it terminates the rank), and the transport_block/
    // gemm/scal calls below can throw on a null handle or a backend mismatch --
    // both validated above, but a failure between these allocations and the
    // frees at the end leaks the scratch buffers on that rank.  The process
    // terminates via the uncaught exception, so the OS reclaims them; this is
    // not a runaway leak in practice.
    const int buffer_max = 2;
    T* d_A_temp[buffer_max] = {nullptr, nullptr};
    T* d_B_temp[buffer_max] = {nullptr, nullptr};
    int count_a = ((transa == 'N') ? std::max(m_loc_A, m_loc_C) : std::max(n_loc_A, m_loc_C)) * nb;
    int count_b = nb * ((transb == 'N') ? std::max(n_loc_B, n_loc_C) : std::max(m_loc_B, n_loc_C));
    count_a = std::max(1, count_a);
    count_b = std::max(1, count_b);
    for (int i = 0; i < buffer_max; i++) {
        RUNTIME_CHECK<Backend>(runtimeMalloc<Backend>(
            reinterpret_cast<void**>(&d_A_temp[i]), count_a * sizeof(T)));
        RUNTIME_CHECK<Backend>(runtimeMalloc<Backend>(
            reinterpret_cast<void**>(&d_B_temp[i]), count_b * sizeof(T)));
    }

    int temp_buffer = 0;
    int k_s = 0, kb = 0;
    auto get_data = [&](int ks) {
        kb = std::min(nb, k - ks);
        if (kb <= 0) return;
        char sData_a;
        int m_a, n_a, g_ia, g_ja;
        if (transa != 'N') {
            sData_a = 'R';
            m_a = kb;
            n_a = m;
            g_ia = ks;
            g_ja = 0;
        } else {
            sData_a = 'C';
            m_a = m;
            n_a = kb;
            g_ia = 0;
            g_ja = ks;
        }
        ddla::transport_block<Backend, T>(handle, 
            sData_a, transa,
            m_a, n_a,
            A, g_ia, g_ja, descA,
            d_A_temp[temp_buffer]);

        char sData_b;
        int m_b, n_b, g_ib, g_jb;
        if (transb != 'N') {
            sData_b = 'C';
            m_b = n;
            n_b = kb;
            g_ib = 0;
            g_jb = ks;
        } else {
            sData_b = 'R';
            m_b = kb;
            n_b = n;
            g_ib = ks;
            g_jb = 0;
        }
        ddla::transport_block<Backend, T>(handle, 
            sData_b, transb,
            m_b, n_b,
            B, g_ib, g_jb, descB,
            d_B_temp[temp_buffer]);
    };

    get_data(k_s);
    bool first_gemm = true;
    for (k_s = 0; k_s < k; k_s += nb) {
        if constexpr (Backend == DdlaBackend::CPU) {
            // No-op: CPU BLAS calls are synchronous and every MPI call in
            // transport_block<DdlaBackend::CPU> is blocking, so there is no
            // in-flight work left to wait on here.
        } else {
            RUNTIME_CHECK<Backend>(runtimeStreamSynchronize<Backend>(h->stream));
            RUNTIME_CHECK<Backend>(runtimeStreamSynchronize<Backend>(h->stream_data));
        }
        if (m_loc_C > 0 && n_loc_C > 0) {
            const T gemm_beta = first_gemm ? beta : static_cast<T>(1.0);
            ddla::gemm<Backend>(
                h, transa, 'N',
                m_loc_C, n_loc_C, kb,
                alpha,
                d_A_temp[temp_buffer], transa == 'N' ? static_cast<int>(m_loc_C) : kb,
                d_B_temp[temp_buffer], kb,
                gemm_beta,
                C, lldC);
            first_gemm = false;
        }
        temp_buffer = (temp_buffer + 1) % buffer_max;
        get_data(k_s + nb);
    }
    if constexpr (Backend == DdlaBackend::CPU) {
        // No-op: CPU BLAS calls are synchronous and every MPI call in
        // transport_block<DdlaBackend::CPU> is blocking, so there is no
        // in-flight work left to wait on here.
    } else {
        RUNTIME_CHECK<Backend>(runtimeStreamSynchronize<Backend>(h->stream));
        RUNTIME_CHECK<Backend>(runtimeStreamSynchronize<Backend>(h->stream_data));
    }
    for (int i = 0; i < buffer_max; i++) {
        RUNTIME_CHECK<Backend>(runtimeFree<Backend>(d_A_temp[i]));
        RUNTIME_CHECK<Backend>(runtimeFree<Backend>(d_B_temp[i]));
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------
#define INSTANTIATE_PGEMM(BACKEND, TYPE)                                     \
    template void pgemm<BACKEND, TYPE>(                              const DdlaHandle_t&, \
        const char&, const char&, const int&, const int&, const int&,          \
        const TYPE&, const TYPE*, const int*,                            \
        const TYPE*, const int*,                                         \
        const TYPE&, TYPE*, const int*)

#if DDLA_HAS_CPU
INSTANTIATE_PGEMM(DdlaBackend::CPU, float);
INSTANTIATE_PGEMM(DdlaBackend::CPU, double);
INSTANTIATE_PGEMM(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PGEMM(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_PGEMM(DdlaBackend::GPU, float);
INSTANTIATE_PGEMM(DdlaBackend::GPU, double);
INSTANTIATE_PGEMM(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PGEMM(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_PGEMM

} // namespace ddla
