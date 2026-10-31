#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>
#include "ddla_connector.h"
#include <ddla/ddla_config.h>
#include <ddla/ddla_desc.h>
#include "ddla_stream.h"
#include <vector>
#include "transport_block.h"
#include "pgemm_impl.h"

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


namespace pgemm_impl {

template <DdlaBackend Backend, typename T>
void ab(
    const DdlaHandle_t& handle, char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const T* A, const int* descA,
    const T* B, const int* descB,
    const T& beta,
    T* C, const int* descC)
{
    const DdlaHandle_t h = handle;
    int nprows, npcols, myprow, mypcol;
    ddlaGetGridDims(h, nprows, npcols);
    ddlaGetGridCoords(h, myprow, mypcol);
    const int nb = transa == 'N' ? descA[DDLA_NB_] : descA[DDLA_MB_];
    const int m_loc_C = num_loc(m, descC[DDLA_MB_], myprow, descC[DDLA_RSRC_], nprows);
    const int n_loc_C = num_loc(n, descC[DDLA_NB_], mypcol, descC[DDLA_CSRC_], npcols);
    const int lldC = descC[DDLA_LLD_];
    // Buffer sizing
    const int m_loc_A = num_loc((transa == 'N') ? m : k, descA[DDLA_MB_], myprow, descA[DDLA_RSRC_], nprows);
    const int n_loc_A = num_loc((transa == 'N') ? k : m, descA[DDLA_NB_], mypcol, descA[DDLA_CSRC_], npcols);
    const int m_loc_B = num_loc((transb == 'N') ? k : n, descB[DDLA_MB_], myprow, descB[DDLA_RSRC_], nprows);
    const int n_loc_B = num_loc((transb == 'N') ? n : k, descB[DDLA_NB_], mypcol, descB[DDLA_CSRC_], npcols);

    // ---- k-panel blocking: one local gemm per run of consecutive k-blocks ---
    // A local gemm is cheaper per flop when it covers a run of k-blocks instead
    // of a single one: with nb = 128 and a 1024-wide k panel that is 8 panel
    // transports (8 broadcasts) feeding ONE local gemm of k = 1024. The target
    // panel width defaults to DDLA_PGEMM_KPANEL (1024); a value below nb
    // degenerates to the classic one-block-per-gemm pipeline. The transposed
    // transport path has no multi-block gather yet, so any transposed operand
    // stays one block per gemm.
    constexpr int kDefaultPanelWidth = 1024;
    const int kfull = k / nb;
    const int ktail = k - kfull * nb;
    const int nb_eff = (nb > 0) ? nb : 1;
    int panel_width = kDefaultPanelWidth;
    {
        const char* s = std::getenv("DDLA_PGEMM_KPANEL");
        if (s != nullptr && *s != '\0') {
            const int v = std::atoi(s);
            if (v > 0) panel_width = v;
        }
    }
    int blocks_per_gemm = std::max(1, panel_width / nb_eff);
    // The transposed transport path (trans != 'N') has no multi-block gather
    // yet -- it would move only the first block of a run and leave the rest of
    // the panel uninitialized -- so those operands stay one block per gemm.
    if (transa != 'N' || transb != 'N') blocks_per_gemm = 1;

    struct KStage { int start; int cnt; int kk; };  // [start, start+kk) slice of k
    std::vector<KStage> stages;
    stages.reserve(static_cast<std::size_t>(kfull / blocks_per_gemm + 2));
    for (int b = 0; b < kfull; b += blocks_per_gemm) {
        const int cnt = std::min(blocks_per_gemm, kfull - b);
        stages.push_back(KStage{b * nb, cnt, cnt * nb});
    }
    if (ktail > 0) stages.push_back(KStage{kfull * nb, 1, ktail});
    const int panel_max = blocks_per_gemm * nb;  // widest stage (>= nb)

    // The handle selects host/device storage and orders release after queued work.
    const int buffer_max = 2;
    vector<T> A_temp[buffer_max] = {vector<T>(h), vector<T>(h)};
    vector<T> B_temp[buffer_max] = {vector<T>(h), vector<T>(h)};
    int count_a = ((transa == 'N') ? std::max(m_loc_A, m_loc_C) : std::max(n_loc_A, m_loc_C)) * panel_max;
    int count_b = panel_max * ((transb == 'N') ? std::max(n_loc_B, n_loc_C) : std::max(m_loc_B, n_loc_C));
    count_a = std::max(1, count_a);
    count_b = std::max(1, count_b);
    // One-block staging tile. The B panel is stored as (kk x n_loc) with
    // ld = kk so one gemm sees the whole run, but transport_block writes a
    // (w x n_loc) tile with ld = w; every block is staged here and copied
    // into the panel with the panel pitch.
    int count_blk = nb * ((transb == 'N') ? std::max(n_loc_B, n_loc_C) : std::max(m_loc_B, n_loc_C));
    count_blk = std::max(1, count_blk);  // keep B_blk a real allocation even at zero local extent
    vector<T> B_blk(h, count_blk);
    for (int i = 0; i < buffer_max; i++) {
        A_temp[i].resize(count_a);
        B_temp[i].resize(count_b);
    }

    // Wait out the in-flight panel transport before the local gemm reads it
    // (GPU only: CPU BLAS is synchronous and every MPI call in
    // transport_block<DdlaBackend::CPU> is blocking).
    auto sync_panel = [&]() {
        if constexpr (Backend != DdlaBackend::CPU) {
            DDLA_CHECK(ddlaSynchronize(h));
            DDLA_CHECK(detail::synchronize_on(h, static_cast<gpuStream_t>(h->stream_data)));
        }
    };

    int temp_buffer = 0;

    auto get_data = [&](int s) {
        if (s < 0 || s >= static_cast<int>(stages.size())) return;
        const int ks = stages[s].start;
        const int cnt = stages[s].cnt;
        const int kk = stages[s].kk;

        // Gather the stage's blocks one transport_block call each. The A tile
        // (sData 'C', transa 'N') is (rows x w) with ld = rows, so it lands
        // straight in the panel at column offset b * w; the B tile (sData 'R',
        // transb 'N') is (w x cols) with ld = w, which only matches the panel
        // pitch kk when the stage is one block wide -- otherwise it is staged
        // in B_blk.data() and copied in with the panel pitch. w = nb for every full
        // block; the tail stage has w = ktail.
        for (int b = 0; b < cnt; ++b) {
            const int g = ks + b * nb;
            const int w = std::min(nb, k - g);

            if (transa == 'N') {
                ddla::transport_block<Backend, T>(handle,
                    'C', transa,
                    m, w,
                    A, 0, g, descA,
                    A_temp[temp_buffer].data() + (std::size_t)b * w * m_loc_A);
            } else {
                ddla::transport_block<Backend, T>(handle,
                    'R', transa,
                    w, m,
                    A, g, 0, descA,
                    A_temp[temp_buffer].data());
            }

            if (transb == 'N') {
                if (cnt != 1) {  // multi-block: tile ld w != panel ld kk, stage + repack
                    ddla::transport_block<Backend, T>(handle,
                        'R', transb,
                        w, n,
                        B, g, 0, descB,
                        B_blk.data());
                    DDLA_CHECK(ddlaMemcpy2D(h,
                        B_temp[temp_buffer].data() + (std::size_t)b * w, (std::size_t)kk * sizeof(T),
                        B_blk.data(), (std::size_t)w * sizeof(T),
                        (std::size_t)w * sizeof(T), (std::size_t)n_loc_B));
                } else {
                    ddla::transport_block<Backend, T>(handle,
                        'R', transb,
                        w, n,
                        B, g, 0, descB,
                        B_temp[temp_buffer].data());
                }
            } else {
                ddla::transport_block<Backend, T>(handle,
                    'C', transb,
                    n, w,
                    B, 0, g, descB,
                    B_temp[temp_buffer].data());
            }
        }
    };

    get_data(0);
    bool first_gemm = true;
    for (std::size_t s = 0; s < stages.size(); ++s) {
        const int kb = stages[s].kk;
        sync_panel();
        if (m_loc_C > 0 && n_loc_C > 0) {
            const T gemm_beta = first_gemm ? beta : static_cast<T>(1.0);
            ddla::gemm<Backend>(
                h, transa, 'N',
                m_loc_C, n_loc_C, kb,
                alpha,
                A_temp[temp_buffer].data(), transa == 'N' ? static_cast<int>(m_loc_C) : kb,
                B_temp[temp_buffer].data(), kb,
                gemm_beta,
                C, lldC);
            first_gemm = false;
        }
        temp_buffer = (temp_buffer + 1) % buffer_max;
        get_data(static_cast<int>(s) + 1);
    }
    sync_panel();
    for (int i = 0; i < buffer_max; i++) {
        A_temp[i].reset();
        B_temp[i].reset();
    }
    B_blk.reset();
}

template <DdlaBackend Backend, typename T>
void run(
    const DdlaHandle_t& handle, char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const T* A, const int* descA,
    const T* B, const int* descB,
    const T& beta,
    T* C, const int* descC, Algorithm algorithm)
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

    // Shared validation for every algorithm.
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

    const int m_loc_C = num_loc(m, descC[DDLA_MB_], myprow, descC[DDLA_RSRC_], nprows);
    const int n_loc_C = num_loc(n, descC[DDLA_NB_], mypcol, descC[DDLA_CSRC_], npcols);
    const int lldC = descC[DDLA_LLD_];

    // ---- Degenerate-size fast paths (F2) -----------------------------------
    // Empty outputs are untouched. A zero product must not read A or B,
    // and beta == 0 must not read the previous (possibly uninitialized) C.
    if (m == 0 || n == 0) {
        return;
    }
    if (k == 0 || alpha == T(0)) {
        if (m_loc_C > 0 && n_loc_C > 0) {
            for (int j = 0; j < n_loc_C; ++j) {
                T* col = C + static_cast<std::size_t>(j) * lldC;
                if (beta == T(0)) {
                    DDLA_CHECK(ddlaMemset(h, col, 0, sizeof(T) * m_loc_C));
                } else if (beta != T(1)) {
                    ddla::scal<Backend, T>(h, m_loc_C, beta, col, 1);
                }
            }
        }
        return;
    }

    if (algorithm == Algorithm::Auto)
        algorithm = select(transa, transb, m, n, k, descA, descB, descC, nprows, npcols);
    if (algorithm == Algorithm::AB) {
        ab<Backend>(handle, transa, transb, m, n, k, alpha, A, descA, B, descB, beta, C, descC);
    } else {
        stationary<Backend>(handle, algorithm == Algorithm::BC, transa, transb,
                            m, n, k, alpha, A, descA, B, descB, beta, C, descC);
    }
}

} // namespace pgemm_impl

template <DdlaBackend Backend, typename T>
void pgemm(
    const DdlaHandle_t& handle, char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const T* A, const int* descA,
    const T* B, const int* descB,
    const T& beta,
    T* C, const int* descC)
{
    pgemm_impl::run<Backend>(handle, transa, transb, m, n, k, alpha,
                            A, descA, B, descB, beta, C, descC, pgemm_impl::Algorithm::Auto);
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------
#define INSTANTIATE_PGEMM(BACKEND, TYPE)                                     \
    template void pgemm<BACKEND, TYPE>(                              const DdlaHandle_t&, \
        char, char, int, int, int,          \
        const TYPE&, const TYPE*, const int*,                            \
        const TYPE*, const int*,                                         \
        const TYPE&, TYPE*, const int*);                               \
    template void pgemm_impl::run<BACKEND, TYPE>(const DdlaHandle_t&, \
        char, char, int, int, int, \
        const TYPE&, const TYPE*, const int*, const TYPE*, const int*, \
        const TYPE&, TYPE*, const int*, pgemm_impl::Algorithm)

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
