#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <algorithm>
#include <initializer_list>
#include <cassert>
#include <stdexcept>
#include <string>
#include "ddla_connector.h"
#include <ddla/ddla_config.h>
#include <ddla/ddla_desc.h>
#include <ddla/ddla_vector.h>
#include "ddla_stream.h"
#include <vector>
#include "pblas_workspace.h"
#include "pgemm_impl.h"
#include "check_agree.h"
#include "plan_util.h"
#include "instantiate.h"
#include "require_backend.h"

// Host includes used by the CPU paths in every build; no GPU vendor headers.
#include <cstdlib>
#include <complex>
#include <new>

namespace ddla {


namespace pgemm_impl {

namespace {

// ---- k-panel blocking: one local gemm per run of consecutive k-blocks ---
// A local gemm is cheaper per flop when it covers a run of k-blocks instead
// of a single one: with nb = 128 and a 1024-wide k panel that is 8 panel
// transports (8 broadcasts) feeding ONE local gemm of k = 1024. The target
// panel width defaults to DDLA_PGEMM_KPANEL (1024); a value below nb
// degenerates to the classic one-block-per-gemm pipeline. The transposed
// transport path has no multi-block gather yet, so any transposed operand
// stays one block per gemm.
constexpr int kDefaultPanelWidth = 1024;

int kpanel_width()
{
    int panel_width = kDefaultPanelWidth;
    const char* s = std::getenv("DDLA_PGEMM_KPANEL");
    if (s != nullptr && *s != '\0') {
        const int v = std::atoi(s);
        if (v > 0) panel_width = v;
    }
    return panel_width;
}

// The k block of the SUMMA branch: A's column (or, transposed, row) block.
int ab_block(char transa, const int* descA)
{
    return transa == 'N' ? descA[DDLA_NB_] : descA[DDLA_MB_];
}

// The k-blocks one local gemm covers.  The transposed transport path
// (trans != 'N') has no multi-block gather yet -- it would move only the
// first block of a run and leave the rest of the panel uninitialized -- so
// those operands stay one block per gemm.
int ab_blocks_per_gemm(char transa, char transb, int nb, int panel_width)
{
    return transa != 'N' || transb != 'N' ? 1 : std::max(1, panel_width / nb);
}

// The k slices of the SUMMA branch: runs of bpg whole k-blocks, then the
// partial tail block.  ab_stages counts them, ab_stage(s) is slice s.
int ab_stages(int k, int nb, int bpg)
{
    const int kfull = k / nb;
    return (kfull + bpg - 1) / bpg + (k % nb > 0 ? 1 : 0);
}

KStage ab_stage(int s, int k, int nb, int bpg)
{
    const int kfull = k / nb;
    if (s < (kfull + bpg - 1) / bpg) {
        const int cnt = std::min(bpg, kfull - s * bpg);
        return KStage{s * bpg * nb, cnt, cnt * nb};
    }
    return KStage{kfull * nb, 1, k - kfull * nb};
}

// Element counts of the SUMMA branch's staging, in carving order: the A panel
// (TA) and the B panel (TB) of one k slice, and the one-block B tile (TB).
// The B panel is stored as (kk x n_loc) with ld = kk so one gemm sees the
// whole run, but the panel copy writes a (w x n_loc) tile with ld = w; every
// block of a multi-block run is staged in the tile and copied into the panel
// with the panel pitch.  Each is a real allocation (>= 1 element) even at
// zero local extent.
void ab_counts(const DdlaHandle_t& h, char transa, char transb, int m, int n, int k,
               const int* descA, const int* descB, const int* descC, int panel_width,
               std::size_t (&count)[3])
{
    const int nprows = h->nprows_, npcols = h->npcols_, myprow = h->myprow_, mypcol = h->mypcol_;
    const int nb = ab_block(transa, descA);
    const std::size_t panel_max = std::size_t(ab_blocks_per_gemm(transa, transb, nb, panel_width)) * nb;
    const int m_loc_C = num_loc(m, descC[DDLA_MB_], myprow, descC[DDLA_RSRC_], nprows);
    const int n_loc_C = num_loc(n, descC[DDLA_NB_], mypcol, descC[DDLA_CSRC_], npcols);
    const int m_loc_A = num_loc((transa == 'N') ? m : k, descA[DDLA_MB_], myprow, descA[DDLA_RSRC_], nprows);
    const int n_loc_A = num_loc((transa == 'N') ? k : m, descA[DDLA_NB_], mypcol, descA[DDLA_CSRC_], npcols);
    const int m_loc_B = num_loc((transb == 'N') ? k : n, descB[DDLA_MB_], myprow, descB[DDLA_RSRC_], nprows);
    const int n_loc_B = num_loc((transb == 'N') ? n : k, descB[DDLA_NB_], mypcol, descB[DDLA_CSRC_], npcols);
    const std::size_t rows_a = (transa == 'N') ? std::max(m_loc_A, m_loc_C) : std::max(n_loc_A, m_loc_C);
    const std::size_t cols_b = (transb == 'N') ? std::max(n_loc_B, n_loc_C) : std::max(m_loc_B, n_loc_C);
    count[0] = std::max(std::size_t(1), rows_a * panel_max);
    count[1] = std::max(std::size_t(1), panel_max * cols_b);
    count[2] = std::max(std::size_t(1), std::size_t(nb) * cols_b);
}

// Query redistribution once for the largest A/B panel. Aligned N panels use
// the direct copy/broadcast path and need no additional scratch.
template <typename TA, typename TB>
void ab_panel_bytes(const DdlaHandle_t& h, char transa, char transb, int m, int n, int nb,
                    const int* descA, const int* descB, const int* descC,
                    std::size_t (&panel_a)[2], std::size_t (&panel_b)[2])
{
    pblas::panel_workspace_size<TA>(h,true,transa,transa=='N'?m:nb,transa=='N'?nb:m,
        descA,descC,&panel_a[0],&panel_a[1]);
    pblas::panel_workspace_size<TB>(h,false,transb,transb=='N'?nb:n,transb=='N'?n:nb,
        descB,descC,&panel_b[0],&panel_b[1]);
}

template <DdlaBackend Backend, typename T, typename TA, typename TB>
void ab_run(
    const DdlaHandle_t& handle, char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const TA* A, const int* descA,
    const TB* B, const int* descB,
    const T& beta,
    T* C, const int* descC, int panel_width,
    void* backend_work, [[maybe_unused]] std::size_t backend_bytes,
    void* host_work, [[maybe_unused]] std::size_t host_bytes)
{
    const DdlaHandle_t h = handle;
    int nprows, npcols, myprow, mypcol;
    ddlaGetGridDims(h, nprows, npcols);
    ddlaGetGridCoords(h, myprow, mypcol);
    const int nb = ab_block(transa, descA);
    const int m_loc_C = num_loc(m, descC[DDLA_MB_], myprow, descC[DDLA_RSRC_], nprows);
    const int n_loc_C = num_loc(n, descC[DDLA_NB_], mypcol, descC[DDLA_CSRC_], npcols);
    const int lldC = descC[DDLA_LLD_];
    const int m_loc_A = num_loc((transa == 'N') ? m : k, descA[DDLA_MB_], myprow, descA[DDLA_RSRC_], nprows);
    const int n_loc_B = num_loc((transb == 'N') ? n : k, descB[DDLA_NB_], mypcol, descB[DDLA_CSRC_], npcols);
    const int bpg = ab_blocks_per_gemm(transa, transb, nb, panel_width);
    const int stages = ab_stages(k, nb, bpg);

    // The scratch, in the order pgemm_workspace_size counts it.
    std::size_t count[3], panel_a_bytes[2], panel_b_bytes[2];
    ab_counts(h, transa, transb, m, n, k, descA, descB, descC, panel_width, count);
    ab_panel_bytes<TA, TB>(h, transa, transb, m, n, nb, descA, descB, descC, panel_a_bytes, panel_b_bytes);
    char* bw = static_cast<char*>(backend_work);
    char* hw = static_cast<char*>(host_work);
    TA* A_temp[2];
    TB* B_temp[2];
    for (int i = 0; i < 2; ++i) {
        A_temp[i] = detail::take<TA>(bw, count[0]);
        B_temp[i] = detail::take<TB>(bw, count[1]);
    }
    TB* const B_blk = detail::take<TB>(bw, count[2]);
    char* const tr_a = detail::take<char>(bw, panel_a_bytes[0]);
    char* const tr_b = detail::take<char>(bw, panel_b_bytes[0]);
    char* const tr_a_host = detail::take<char>(hw, panel_a_bytes[1]);
    char* const tr_b_host = detail::take<char>(hw, panel_b_bytes[1]);
    const std::size_t used = static_cast<std::size_t>(bw - static_cast<char*>(backend_work));
    assert(used <= backend_bytes);
    assert(static_cast<std::size_t>(hw - static_cast<char*>(host_work)) <= host_bytes);
    // The rest is the local gemm's (a mixed product's) scratch.
    void* const gemm_work = bw;
    const std::size_t gemm_bytes = backend_bytes - used;
    constexpr int buffer_max = 2;

    // Wait out the in-flight panel transport before the local gemm reads it
    // (GPU only: CPU BLAS is synchronous and every MPI call in
    // CPU panel communication is blocking).
    auto sync_panel = [&]() {
        if constexpr (Backend != DdlaBackend::CPU) {
            DDLA_CHECK(ddlaSynchronize(h));
            DDLA_CHECK(detail::synchronize_on(h, static_cast<gpuStream_t>(h->stream_data)));
        }
    };

    int temp_buffer = 0;

    auto get_data = [&](int s) {
        if (s < 0 || s >= stages) return;
        const KStage stage = ab_stage(s, k, nb, bpg);
        const int ks = stage.start;
        const int cnt = stage.cnt;
        const int kk = stage.kk;

        // Gather the stage's blocks one panel copy each. The A tile
        // (sData 'C', transa 'N') is (rows x w) with ld = rows, so it lands
        // straight in the panel at column offset b * w; the B tile (sData 'R',
        // transb 'N') is (w x cols) with ld = w, which only matches the panel
        // pitch kk when the stage is one block wide -- otherwise it is staged
        // in B_blk and copied in with the panel pitch. w = nb for every full
        // block; the tail stage has w = ktail.
        for (int b = 0; b < cnt; ++b) {
            const int g = ks + b * nb;
            const int w = std::min(nb, k - g);

            pblas::panel_run<Backend,TA>(handle,true,transa,
                transa=='N'?m:w,transa=='N'?w:m,A,transa=='N'?0:g,transa=='N'?g:0,
                descA,descC,A_temp[temp_buffer]+std::size_t(b)*w*m_loc_C,
                tr_a,panel_a_bytes[0],tr_a_host,panel_a_bytes[1]);
            TB* const destination = cnt==1 ? B_temp[temp_buffer] : B_blk;
            pblas::panel_run<Backend,TB>(handle,false,transb,
                transb=='N'?w:n,transb=='N'?n:w,B,transb=='N'?g:0,transb=='N'?0:g,
                descB,descC,destination,tr_b,panel_b_bytes[0],tr_b_host,panel_b_bytes[1]);
            if (cnt!=1)
                DDLA_CHECK(ddlaMemcpy2D(h,B_temp[temp_buffer]+std::size_t(b)*w,
                    std::size_t(kk)*sizeof(TB),B_blk,std::size_t(w)*sizeof(TB),
                    std::size_t(w)*sizeof(TB),n_loc_C));
        }
    };

    get_data(0);
    bool first_gemm = true;
    for (int s = 0; s < stages; ++s) {
        const int kb = ab_stage(s, k, nb, bpg).kk;
        sync_panel();
        if (m_loc_C > 0 && n_loc_C > 0) {
            const T gemm_beta = first_gemm ? beta : static_cast<T>(1.0);
            local_gemm_run<Backend, T, TA, TB>(
                h, 'N', 'N',
                m_loc_C, n_loc_C, kb,
                alpha,
                A_temp[temp_buffer], m_loc_C,
                B_temp[temp_buffer], kb,
                gemm_beta,
                C, lldC, gemm_work, gemm_bytes);
            first_gemm = false;
        }
        temp_buffer = (temp_buffer + 1) % buffer_max;
        get_data(s + 1);
    }
    sync_panel();
}

} // namespace

// Component 1.
template <DdlaBackend Backend>
Algorithm pgemm_check(
    const DdlaHandle_t& h, char transa, char transb, int m, int n, int k,
    bool alpha_is_zero, const int* descA, const int* descB, const int* descC,
    Algorithm algorithm, int& panel_width)
{
    return detail::agree_check(h, "pgemm", [&](detail::Agree& agree) {
        detail::require_backend<Backend>(h, "pgemm");
        // Each descriptor is validated against this handle's process grid
        // (dense block-cyclic type, slot ranges, LLD vs LOCr); a descriptor
        // built against a different process grid is rejected there.
        check_desc(descA, h);
        check_desc(descB, h);
        check_desc(descC, h);
        int nprows = 0, npcols = 0;
        ddlaGetGridDims(h, nprows, npcols);


        // Shared validation for every algorithm.
        if (transa != 'N' && transa != 'T' && transa != 'C')
            throw std::invalid_argument("pgemm: transa must be 'N', 'T' or 'C'");
        if (transb != 'N' && transb != 'T' && transb != 'C')
            throw std::invalid_argument("pgemm: transb must be 'N', 'T' or 'C'");
        if (m < 0 || n < 0 || k < 0)
            throw std::invalid_argument("pgemm: negative dimension");

        const int opA_m = (transa == 'N') ? descA[DDLA_M_] : descA[DDLA_N_];
        const int opA_n = (transa == 'N') ? descA[DDLA_N_] : descA[DDLA_M_];
        const int opB_m = (transb == 'N') ? descB[DDLA_M_] : descB[DDLA_N_];
        const int opB_n = (transb == 'N') ? descB[DDLA_N_] : descB[DDLA_M_];
        if (m > opA_m || k > opA_n || k > opB_m || n > opB_n ||
            m > descC[DDLA_M_] || n > descC[DDLA_N_])
            throw std::invalid_argument("pgemm: m x n x k exceeds the descriptors");
        {
            const int mbC = descC[DDLA_MB_];
            const int nbC = descC[DDLA_NB_];
            const int mbA = (transa == 'N') ? descA[DDLA_MB_] : descA[DDLA_NB_];
            const int kbA = (transa == 'N') ? descA[DDLA_NB_] : descA[DDLA_MB_];
            const int kbB = (transb == 'N') ? descB[DDLA_MB_] : descB[DDLA_NB_];
            const int nbB = (transb == 'N') ? descB[DDLA_NB_] : descB[DDLA_MB_];
            if (mbA != mbC || kbA != kbB || nbB != nbC)
                throw std::invalid_argument("pgemm: block sizes of A, B and C do not match");
        }

        panel_width = kDefaultPanelWidth;
        if (m == 0 || n == 0 || k == 0 || alpha_is_zero) return Algorithm::AB;

        Algorithm chosen = algorithm;
        if (chosen == Algorithm::Auto) {
            chosen = select(transa, transb, m, n, k, descA, descB, descC, nprows, npcols);
            // The cost model ignores source coordinates; SUMMA (AB) accepts any
            // alignment, so an unaligned stationary pick falls back to it.
            if (chosen != Algorithm::AB &&
                !stationary_aligned(chosen == Algorithm::BC, transa, transb, descA, descB, descC))
                chosen = Algorithm::AB;
        } else if (chosen != Algorithm::AB &&
                   !stationary_aligned(chosen == Algorithm::BC, transa, transb, descA, descB, descC)) {
            throw std::invalid_argument("pgemm: stationary branch requires aligned non-transposed panel owners");
        }
        panel_width = kpanel_width();
        agree.mode("DDLA_PGEMM_KPANEL", panel_width);
        return chosen;
    });
}


// Component 2.  The SUMMA branch (AB): backend [ A panel (TA) | B panel (TB) ]
// twice (double-buffered) | B tile (TB) | transport of a transposed A | of a
// transposed B | the local gemm's scratch for the largest of its shapes (a
// mixed product only) ], host [ transport of a transposed A | of a transposed
// B ].  The stationary branches: stationary_workspace_size.
template <typename T, typename TA, typename TB>
void pgemm_workspace_size(
    const DdlaHandle_t& h, Algorithm algorithm, int panel_width, bool alpha_is_zero, char transa, char transb,
    int m, int n, int k, bool beta_is_nonzero,
    const int* descA, const int* descB, const int* descC,
    std::size_t* backend_bytes, std::size_t* host_bytes)
{
    *backend_bytes = 0;
    *host_bytes = 0;
    if (alpha_is_zero || m == 0 || n == 0 || k == 0) return;
    if (h->nprocs == 1 && std::is_same_v<T,TA> && std::is_same_v<T,TB>) {
        local_gemm_workspace_size<T,TA,TB>(h,m,n,k,backend_bytes);
        return;
    }
    if (algorithm != Algorithm::AB) {
        stationary_workspace_size<T, TA, TB>(h, algorithm == Algorithm::BC, transa, transb,
                                             m, n, k, beta_is_nonzero, descA, descB, descC,
                                             backend_bytes, host_bytes);
        return;
    }

    const int nb = ab_block(transa, descA);
    const int bpg = ab_blocks_per_gemm(transa, transb, nb, panel_width);
    std::size_t count[3], panel_a_bytes[2], panel_b_bytes[2];
    ab_counts(h, transa, transb, m, n, k, descA, descB, descC, panel_width, count);
    ab_panel_bytes<TA, TB>(h, transa, transb, m, n, nb, descA, descB, descC, panel_a_bytes, panel_b_bytes);
    // The local gemm runs a full stage, a shorter last run of whole blocks and the tail block.
    const int m_loc_C = num_loc(m, descC[DDLA_MB_], h->myprow_, descC[DDLA_RSRC_], h->nprows_);
    const int n_loc_C = num_loc(n, descC[DDLA_NB_], h->mypcol_, descC[DDLA_CSRC_], h->npcols_);
    const int kfull = k / nb;
    std::size_t gemm_bytes = 0;
    for (const int kb : {std::min(bpg, kfull) * nb, (kfull % bpg) * nb, k % nb}) {
        std::size_t bytes = 0;
        local_gemm_workspace_size<T, TA, TB>(h, m_loc_C, n_loc_C, kb, &bytes);
        gemm_bytes = std::max(gemm_bytes, bytes);
    }
    *backend_bytes = 2 * (detail::padded_bytes<TA>(count[0]) + detail::padded_bytes<TB>(count[1]))
                   + detail::padded_bytes<TB>(count[2])
                   + detail::padded_bytes<char>(panel_a_bytes[0]) + detail::padded_bytes<char>(panel_b_bytes[0])
                   + gemm_bytes;
    *host_bytes = detail::padded_bytes<char>(panel_a_bytes[1]) + detail::padded_bytes<char>(panel_b_bytes[1]);
}

// Component 3.
template <DdlaBackend Backend, typename T, typename TA, typename TB>
void pgemm_run(
    const DdlaHandle_t& h, Algorithm algorithm, int panel_width, char transa, char transb,
    int m, int n, int k, const T& alpha,
    const TA* A, const int* descA, const TB* B, const int* descB,
    const T& beta, T* C, const int* descC,
    void* backend_work, std::size_t backend_bytes,
    void* host_work, std::size_t host_bytes)
{
    try {
        if (m == 0 || n == 0) {
            return;
        }
        // Empty outputs are untouched. A zero product must not read A or B,
        // and beta == 0 must not read the previous (possibly uninitialized) C.
        if (k == 0 || alpha == T(0)) {
            if (beta == T(1)) return;
            int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
            ddlaGetGridDims(h, nprows, npcols);
            ddlaGetGridCoords(h, myprow, mypcol);
            const int m_loc_C = num_loc(m, descC[DDLA_MB_], myprow, descC[DDLA_RSRC_], nprows);
            const int n_loc_C = num_loc(n, descC[DDLA_NB_], mypcol, descC[DDLA_CSRC_], npcols);
            const int lldC = descC[DDLA_LLD_];
            blas_impl::matrix_scal_run<Backend,T>(h,m_loc_C,n_loc_C,beta,C,lldC);
            return;
        }

        if (h->nprocs == 1 && std::is_same_v<T,TA> && std::is_same_v<T,TB>) {
            local_gemm_run<Backend,T,TA,TB>(h,transa,transb,m,n,k,alpha,A,descA[DDLA_LLD_],
                B,descB[DDLA_LLD_],beta,C,descC[DDLA_LLD_],backend_work,backend_bytes);
            return;
        }

        if (algorithm == Algorithm::AB) {
            ab_run<Backend, T, TA, TB>(h, transa, transb, m, n, k, alpha, A, descA, B, descB, beta, C, descC,
                                       panel_width, backend_work, backend_bytes, host_work, host_bytes);
        } else {
            stationary_run<Backend, T, TA, TB>(h, algorithm == Algorithm::BC, transa, transb,
                                       m, n, k, alpha, A, descA, B, descB, beta, C, descC,
                                       backend_work, backend_bytes, host_work, host_bytes);
        }
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(h) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(h); } catch (...) {}
        throw;
    }
}

template <DdlaBackend Backend, typename T, typename TA, typename TB>
void run(
    const DdlaHandle_t& handle, char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const TA* A, const int* descA,
    const TB* B, const int* descB,
    const T& beta,
    T* C, const int* descC, Algorithm algorithm)
{
    int panel_width = kDefaultPanelWidth;
    algorithm = pgemm_check<Backend>(handle, transa, transb, m, n, k, alpha == T(0),
                                    descA, descB, descC, algorithm, panel_width);
    if (m == 0 || n == 0) return;
    const bool scale_only = k == 0 || alpha == T(0);
    if constexpr (!std::is_same_v<T, TA> || !std::is_same_v<T, TB>) {
        detail::agree_check(handle, "pgemm(mixed)", [&](detail::Agree&) {
            int pr, pc, r, c;
            ddlaGetGridDims(handle, pr, pc); ddlaGetGridCoords(handle, r, c);
            auto owns = [&](int rows, int cols, const int* d) {
                return num_loc(rows,d[DDLA_MB_],r,d[DDLA_RSRC_],pr)>0 &&
                       num_loc(cols,d[DDLA_NB_],c,d[DDLA_CSRC_],pc)>0;
            };
            if (!(scale_only && beta == T(1)) && owns(m,n,descC) && C==nullptr)
                throw std::invalid_argument("pgemm(mixed): null local C");
            if (!scale_only &&
                ((owns(transa=='N'?m:k,transa=='N'?k:m,descA) && A==nullptr) ||
                 (owns(transb=='N'?k:n,transb=='N'?n:k,descB) && B==nullptr)))
                throw std::invalid_argument("pgemm(mixed): null local input");
        });
    }
    std::size_t backend_bytes = 0, host_bytes = 0;
    pgemm_workspace_size<T, TA, TB>(handle, algorithm, panel_width, alpha == T(0), transa, transb, m, n, k, beta != T(0),
                                    descA, descB, descC, &backend_bytes, &host_bytes);
    vector<char> backend_work(handle, backend_bytes);
    std::vector<char> host_work(host_bytes);
    pgemm_run<Backend, T, TA, TB>(handle, algorithm, panel_width, transa, transb, m, n, k, alpha,
                          A, descA, B, descB, beta, C, descC,
                          backend_work.data(), backend_bytes, host_work.data(), host_bytes);
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

template <DdlaBackend Backend, typename R>
void pgemm(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
           const std::complex<R>& alpha, const R* a, const int* da,
           const std::complex<R>* b, const int* db, const std::complex<R>& beta,
           std::complex<R>* c, const int* dc)
{
    pgemm_impl::run<Backend>(h,ta,tb,m,n,k,alpha,a,da,b,db,beta,c,dc,pgemm_impl::Algorithm::Auto);
}

template <DdlaBackend Backend, typename R>
void pgemm(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
           const std::complex<R>& alpha, const std::complex<R>* a, const int* da,
           const R* b, const int* db, const std::complex<R>& beta,
           std::complex<R>* c, const int* dc)
{
    pgemm_impl::run<Backend>(h,ta,tb,m,n,k,alpha,a,da,b,db,beta,c,dc,pgemm_impl::Algorithm::Auto);
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
        const TYPE&, TYPE*, const int*, pgemm_impl::Algorithm); \
    template void pgemm_impl::pgemm_run<BACKEND, TYPE>(const DdlaHandle_t&, \
        pgemm_impl::Algorithm, int, char, char, int, int, int, \
        const TYPE&, const TYPE*, const int*, const TYPE*, const int*, \
        const TYPE&, TYPE*, const int*, void*, std::size_t, void*, std::size_t)

#define INSTANTIATE_PGEMM_CHECK(BACKEND) \
    template pgemm_impl::Algorithm pgemm_impl::pgemm_check<BACKEND>(const DdlaHandle_t&, \
        char, char, int, int, int, bool, const int*, const int*, const int*, pgemm_impl::Algorithm, int&)

#define INSTANTIATE_PGEMM_SIZE(TYPE, TA, TB) \
    template void pgemm_impl::pgemm_workspace_size<TYPE, TA, TB>(const DdlaHandle_t&, \
        pgemm_impl::Algorithm, int, bool, char, char, int, int, int, bool, \
        const int*, const int*, const int*, std::size_t*, std::size_t*)

// Mixed-domain instantiations retain real input storage and complex output.
#define INSTANTIATE_PGEMM_MIXED(BACKEND, R, TA, TB) \
    template void pgemm<BACKEND, R>(const DdlaHandle_t&, char, char, int, int, int, \
        const std::complex<R>&, const TA*, const int*, const TB*, const int*, \
        const std::complex<R>&, std::complex<R>*, const int*); \
    template void pgemm_impl::run<BACKEND, std::complex<R>, TA, TB>(const DdlaHandle_t&, \
        char, char, int, int, int, const std::complex<R>&, const TA*, const int*, const TB*, const int*, \
        const std::complex<R>&, std::complex<R>*, const int*, pgemm_impl::Algorithm); \
    template void pgemm_impl::pgemm_run<BACKEND, std::complex<R>, TA, TB>(const DdlaHandle_t&, \
        pgemm_impl::Algorithm, int, char, char, int, int, int, \
        const std::complex<R>&, const TA*, const int*, const TB*, const int*, \
        const std::complex<R>&, std::complex<R>*, const int*, void*, std::size_t, void*, std::size_t)

#define INSTANTIATE_PGEMM_MIXED_BACKEND(BACKEND) \
    INSTANTIATE_PGEMM_MIXED(BACKEND, float, float, std::complex<float>); \
    INSTANTIATE_PGEMM_MIXED(BACKEND, float, std::complex<float>, float); \
    INSTANTIATE_PGEMM_MIXED(BACKEND, double, double, std::complex<double>); \
    INSTANTIATE_PGEMM_MIXED(BACKEND, double, std::complex<double>, double)

DDLA_FOR_EACH_BACKEND(INSTANTIATE_PGEMM_CHECK)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_PGEMM)
DDLA_FOR_EACH_BACKEND(INSTANTIATE_PGEMM_MIXED_BACKEND)
INSTANTIATE_PGEMM_SIZE(float, float, float);
INSTANTIATE_PGEMM_SIZE(double, double, double);
INSTANTIATE_PGEMM_SIZE(std::complex<float>, std::complex<float>, std::complex<float>);
INSTANTIATE_PGEMM_SIZE(std::complex<double>, std::complex<double>, std::complex<double>);
INSTANTIATE_PGEMM_SIZE(std::complex<float>, float, std::complex<float>);
INSTANTIATE_PGEMM_SIZE(std::complex<float>, std::complex<float>, float);
INSTANTIATE_PGEMM_SIZE(std::complex<double>, double, std::complex<double>);
INSTANTIATE_PGEMM_SIZE(std::complex<double>, std::complex<double>, double);

#undef INSTANTIATE_PGEMM_MIXED_BACKEND
#undef INSTANTIATE_PGEMM_MIXED
#undef INSTANTIATE_PGEMM_SIZE
#undef INSTANTIATE_PGEMM_CHECK
#undef INSTANTIATE_PGEMM

} // namespace ddla
