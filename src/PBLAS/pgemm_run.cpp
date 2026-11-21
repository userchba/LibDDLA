#include "pgemm_geometry.h"
#include "comm_traits.h"
#include "instantiate.h"

#include <cassert>

namespace ddla { namespace pgemm_impl {
namespace {

int ab_full_stages(int kfull, int bpg)
{
    return kfull / bpg + (kfull % bpg != 0);
}
// The k slices of the SUMMA branch: runs of bpg whole k-blocks, then the
// partial tail block.  ab_stages counts them, ab_stage(s) is slice s.
int ab_stages(int k, int nb, int bpg)
{
    const int kfull = k / nb;
    return ab_full_stages(kfull, bpg) + (k % nb > 0 ? 1 : 0);
}

KStage ab_stage(int s, int k, int nb, int bpg)
{
    const int kfull = k / nb;
    if (s < ab_full_stages(kfull, bpg)) {
        const int cnt = std::min(bpg, kfull - s * bpg);
        return KStage{s * bpg * nb, cnt, cnt * nb};
    }
    return KStage{kfull * nb, 1, k - kfull * nb};
}
// Keep panel copies on the calling CPU thread. Starting a second OpenMP
// pool here can leave spinning workers competing with the threaded BLAS.
// The BLAS thread count is untouched, and the handle budget is restored.
template <DdlaBackend Backend> class HostPanelThreads {
    DdlaHandle_t h_;
    int saved_;
    bool enabled_;
public:
    explicit HostPanelThreads(DdlaHandle_t h, bool enabled) : h_(h), saved_(h->host_threads), enabled_(enabled) {
        if constexpr (Backend == DdlaBackend::CPU) { if (enabled_) h_->host_threads = 1; }
    }
    ~HostPanelThreads() {
        if constexpr (Backend == DdlaBackend::CPU) { if (enabled_) h_->host_threads = saved_; }
    }
    HostPanelThreads(const HostPanelThreads&) = delete;
    HostPanelThreads& operator=(const HostPanelThreads&) = delete;
};
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
    const int bpg = ab_blocks_per_gemm(h, k, nb, panel_width);
    const int stages = ab_stages(k, nb, bpg);

    // The scratch, in the order pgemm_workspace_size counts it.
    std::size_t count[3], panel_a_bytes[2], panel_b_bytes[2];
    ab_counts(h, transa, transb, m, n, k, descA, descB, descC, panel_width, count);
    ab_panel_bytes<TA, TB>(h, transa, transb, m, n, nb, descA, descB, descC, panel_a_bytes, panel_b_bytes);
    char* bw = static_cast<char*>(backend_work);
    char* hw = static_cast<char*>(host_work);
    constexpr int buffer_max = Backend == DdlaBackend::CPU ? 1 : 2;
    TA* A_temp[2];
    TB* B_temp[2];
    for (int i = 0; i < buffer_max; ++i) {
        A_temp[i] = detail::take<TA>(bw, count[0]);
        B_temp[i] = detail::take<TB>(bw, count[1]);
    }
    TB* const B_blk = detail::take<TB>(bw, count[2]);
    char* const tr_a = detail::take<char>(bw, panel_a_bytes[0]);
    char* const tr_b = detail::take<char>(bw, panel_b_bytes[0]);
    char* const tr_a_host = panel_a_bytes[1] ? detail::take<char>(hw, panel_a_bytes[1]) : nullptr;
    char* const tr_b_host = panel_b_bytes[1] ? detail::take<char>(hw, panel_b_bytes[1]) : nullptr;
    const std::size_t used = static_cast<std::size_t>(bw - static_cast<char*>(backend_work));
    assert(used <= backend_bytes);
    assert(host_bytes == 0 || static_cast<std::size_t>(hw - static_cast<char*>(host_work)) <= host_bytes);
    // The rest is the local gemm's (a mixed product's) scratch.
    void* const gemm_work = bw;
    const std::size_t gemm_bytes = backend_bytes - used;

    // stream_data is nonblocking. The input event covers preceding input
    // uploads and async scratch allocations on the compute queue, including
    // a caller-selected legacy default compute stream.
    void* const queue = Backend == DdlaBackend::GPU ? h->stream_data : nullptr;
    if constexpr (Backend == DdlaBackend::GPU) {
        DDLA_CHECK(detail::event_record_on(h, h->summa_input, h->stream));
        DDLA_CHECK(detail::event_wait_on(h, h->summa_input, queue));
    }
    bool reuse_pending[buffer_max] = {};

    auto get_data = [&](int s, int slot) {
        if (s >= stages) return;
        if constexpr (Backend == DdlaBackend::GPU) {
            if (reuse_pending[slot])
                DDLA_CHECK(detail::event_wait_on(h, h->summa_reuse[slot], queue));
        }
        const KStage stage = ab_stage(s, k, nb, bpg);
        const int ks = stage.start;
        const int kk = stage.kk;

        // Transport one descriptor block at a time so changing source owners
        // and conjugation are resolved before concatenation. A is contiguous
        // by logical k columns. B needs a pitched append because its block
        // tile has leading dimension w, while the complete panel has ld=kk.
        for (int b = 0; b < stage.cnt; ++b) {
            const int g = ks + b * nb;
            const int w = std::min(nb, k - g);
            const std::size_t q = static_cast<std::size_t>(g - ks);
            pblas::panel_run<Backend, TA>(h, true, transa,
                transa == 'N' ? m : w, transa == 'N' ? w : m,
                A, transa == 'N' ? 0 : g, transa == 'N' ? g : 0,
                descA, descC, A_temp[slot] + q * m_loc_C,
                tr_a, panel_a_bytes[0], tr_a_host, panel_a_bytes[1], queue,
                Backend == DdlaBackend::GPU ? h->summa_metadata[0] : nullptr);
            TB* const destination = stage.cnt == 1 ? B_temp[slot] : B_blk;
            pblas::panel_run<Backend, TB>(h, false, transb,
                transb == 'N' ? w : n, transb == 'N' ? n : w,
                B, transb == 'N' ? g : 0, transb == 'N' ? 0 : g,
                descB, descC, destination,
                tr_b, panel_b_bytes[0], tr_b_host, panel_b_bytes[1], queue,
                Backend == DdlaBackend::GPU ? h->summa_metadata[1] : nullptr);
            if (stage.cnt != 1)
                DDLA_CHECK(detail::memcpy2d_on(h, B_temp[slot] + q,
                    std::size_t(kk) * sizeof(TB), B_blk, std::size_t(w) * sizeof(TB),
                    std::size_t(w) * sizeof(TB), n_loc_C, queue));
        }
        if constexpr (Backend == DdlaBackend::GPU)
            DDLA_CHECK(detail::event_record_on(h, h->summa_ready[slot], queue));
    };

    get_data(0, 0);
    bool first_gemm = true;
    for (int s = 0; s < stages; ++s) {
        const int slot = s % buffer_max;
        const int kb = ab_stage(s, k, nb, bpg).kk;
        if constexpr (Backend == DdlaBackend::GPU)
            DDLA_CHECK(detail::event_wait_on(h, h->summa_ready[slot], h->stream));
        if (m_loc_C > 0 && n_loc_C > 0) {
            const T gemm_beta = first_gemm ? beta : static_cast<T>(1.0);
            local_gemm_run<Backend, T, TA, TB>(
                h, 'N', 'N', m_loc_C, n_loc_C, kb, alpha,
                A_temp[slot], m_loc_C, B_temp[slot], kb, gemm_beta,
                C, lldC, gemm_work, gemm_bytes);
            first_gemm = false;
        }
        if constexpr (Backend == DdlaBackend::GPU) {
            // Empty-output ranks still record retirement after their ready
            // wait, so slot reuse never depends on an unrecorded event.
            DDLA_CHECK(detail::event_record_on(h, h->summa_reuse[slot], h->stream));
            reuse_pending[slot] = true;
        }
        get_data(s + 1, (slot + 1) % buffer_max);
    }
    if constexpr (Backend == DdlaBackend::GPU) {
        // The borrowed arenas and host metadata are reusable on return.
        // No compute drain separates GEMMs; MPI and host-metadata contracts
        // can still require a data-queue wait during panel preparation.
        DDLA_CHECK(ddlaSynchronize(h));
        DDLA_CHECK(detail::synchronize_on(h, queue));
    }
}

} // namespace

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
    // Check before entering the cleanup path: an open CCL group may hold
    // submissions that a queue drain cannot complete. Raw callers must keep
    // the collectively prepared no-group precondition on every rank.
    if constexpr (Backend == DdlaBackend::GPU) {
        if (m > 0 && n > 0 && k > 0 && alpha != T(0) &&
            h->communication() == DdlaCommunication::CCL && h->communication_group_depth)
            throw std::invalid_argument("pgemm: active CCL group must be closed before a GPU product");
    }
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

        if (h->nprocs == 1) {
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
        if (ddlaGetBackend(h) == DdlaBackend::GPU) {
            try { (void)ddlaSynchronize(h); } catch (...) {}
            try { (void)detail::synchronize_on(h, h->stream_data); } catch (...) {}
        }
        throw;
    }
}
template <bool keep_a, DdlaBackend Backend, typename T, typename TA, typename TB>
void stationary_run_fixed(const DdlaHandle_t& h, char ta, char tb,
                    int m, int n, int k, const T& alpha,
                    const TA* a, const int* da, const TB* b, const int* db,
                    const T& beta, T* c, const int* dc,
                    void* backend_work, [[maybe_unused]] std::size_t backend_bytes,
                    void* host_work, [[maybe_unused]] std::size_t host_bytes)
{
    using TX = std::conditional_t<keep_a, TB, TA>;

    // Mixed local GEMM owns an OpenMP team; do not suppress its handle budget.
    HostPanelThreads<Backend> panel_threads(h, std::is_same_v<T,TA> && std::is_same_v<T,TB>);
    const int r=h->myprow_,col=h->mypcol_;
    const int* thin=keep_a?db:da;

    const TX* x = [&]() { if constexpr (keep_a) return b; else return a; }();
    const char tx = keep_a ? tb : ta;
    const CommScope reduction = (keep_a?ta!='N':tb=='N') ? CommScope::Col : CommScope::Row;

    // The scratch, in the order stationary_workspace_size counts it.
    std::size_t count[4], in[2], out[2];
    panel_scratch<T, TX>(h, keep_a, ta, tb, m, n, k, beta != T(0), da, db, dc, count, in, out);
    char* w = static_cast<char*>(backend_work);
    char* hw = static_cast<char*>(host_work);
    T* const partial = detail::take<T>(w, count[0]);
    TX* const panel = detail::take<TX>(w, count[1]);
    TX* const aligned = detail::take<TX>(w, count[2]);
    T* const temporary = detail::take<T>(w, count[3]);
    char* const add_in = detail::take<char>(w, in[0]);
    char* const add_out = detail::take<char>(w, out[0]);
    char* const add_in_host = detail::take<char>(hw, in[1]);
    char* const add_out_host = detail::take<char>(hw, out[1]);
    const std::size_t used = static_cast<std::size_t>(w - static_cast<char*>(backend_work));
    assert(used <= backend_bytes);
    assert(static_cast<std::size_t>(hw - static_cast<char*>(host_work)) <= host_bytes);
    // The rest is the local gemm's (a mixed product's) scratch.
    void* const gemm_work = w;
    const std::size_t gemm_bytes = backend_bytes - used;

    stationary_panels(h,keep_a,ta,tb,m,n,k,da,db,dc,[&](const StationaryGeometry& g,const StationaryPanel& p) {
        const bool k_rows=g.k_rows,thin_k_rows=g.thin_k_rows,transpose_output=g.transpose_output;
        const int kloc=g.kloc,floc=g.floc;
        const int width=p.width,root=p.root,owner=p.input_owner,offset=p.input_offset;
        const int* view=p.view;
        const int* aligned_desc=p.aligned;
        const int* temporary_desc=p.temporary;
        const int* result_desc=p.result;
        const TX* input = x;
        if (input != nullptr && (thin_k_rows ? col : r) == owner)
            input += thin_k_rows ? std::size_t(offset) * thin[DDLA_LLD_] : offset;
        const int* input_desc = view;
        char panel_op = tx;
        if (thin_k_rows != k_rows) {
            // The thin panel's k runs along the other process dimension. One
            // transpose lays it out like the stationary k blocks (any source
            // coordinates), so the broadcast below is always untransposed.
            pgeadd_impl::pgeadd_run<Backend, TX>(h, tx=='C'?'C':'T', 'N',
                aligned_desc[DDLA_M_], aligned_desc[DDLA_N_],
                TX(1), input, 1, 1, view, TX(0), static_cast<const TX*>(nullptr), 1, 1, aligned_desc,
                aligned, 1, 1, aligned_desc, add_in, in[0], add_in_host, in[1]);
            input = aligned;
            input_desc = aligned_desc;
            panel_op = tx == 'N' ? 'T' : 'N';
        }
        // An untransposed transport stages nothing.
        pblas::broadcast_panel<Backend, TX>(h, k_rows, input_desc[DDLA_M_], input_desc[DDLA_N_], input, 0, 0,
            input_desc, panel);
        const int panel_ld = k_rows ? std::max(1, kloc) : width;

        if (floc > 0 && kloc > 0) {
            if constexpr (keep_a)
                local_gemm_run<Backend, T, TA, TB>(h, ta, panel_op, floc, width, kloc, alpha, a, da[DDLA_LLD_],
                              panel, panel_ld, T(0), partial, floc, gemm_work, gemm_bytes);
            else
                local_gemm_run<Backend, T, TA, TB>(h, panel_op, tb, width, floc, kloc, alpha, panel, panel_ld,
                              b, db[DDLA_LLD_], T(0), partial, width, gemm_work, gemm_bytes);
        } else {
            DDLA_CHECK(ddlaMemset(h, partial, 0, std::size_t(floc)*width*sizeof(T)));
        }
        // In place: the root's partial becomes the reduced panel.
        commReduceSum<Backend>(h, reduction, partial, partial, std::size_t(floc)*width, root);

        const int* output=p.output;
        const int cm=p.cm,cn=p.cn;
        const T* result = partial;
        const int* rdesc = result_desc;
        char result_op = 'N';

        if (transpose_output) {
            // A transposed stationary input leaves the result distributed in
            // the opposite direction. Transpose the local reduced slab; the
            // final pgeadd redistributes it back into C's orientation.
            if ((k_rows ? r : col) == root && floc > 0)
                blas_impl::geam_run<Backend>(h, 'T', 'N', keep_a ? width : floc, keep_a ? floc : width,
                    T(1), partial, keep_a ? floc : width, T(0), static_cast<const T*>(nullptr), 1,
                    temporary, temporary_desc[DDLA_LLD_]);
            result = temporary;
            rdesc = temporary_desc;
            result_op = 'T';
        }

        // Every rank joins because a transposed result moves between ranks.
        // Only C's owners hold part of this panel, so beta is applied once,
        // and the descriptor view keeps the rest of C, including padding, untouched.
        const int co=p.output_offset;
        T* dest = c;
        if (dest != nullptr) dest += keep_a ? std::size_t(co)*dc[DDLA_LLD_] : co;
        pgeadd_impl::pgeadd_run<Backend, T>(h, result_op, 'N', cm, cn, T(1), result, 1, 1, rdesc,
                        beta, dest, 1, 1, output, dest, 1, 1, output, add_out, out[0], add_out_host, out[1]);
    });
}
template <DdlaBackend Backend, typename T, typename TA, typename TB>
void stationary_run(const DdlaHandle_t& h, bool keep_a, char ta,char tb,
                    int m,int n,int k,const T& alpha,const TA* a,const int* da,
                    const TB* b,const int* db,const T& beta,T* c,const int* dc,
                    void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes)
{
    try {
        if (keep_a) stationary_run_fixed<true,Backend>(h,ta,tb,m,n,k,alpha,a,da,b,db,beta,c,dc,
                                                       backend_work,backend_bytes,host_work,host_bytes);
        else stationary_run_fixed<false,Backend>(h,ta,tb,m,n,k,alpha,a,da,b,db,beta,c,dc,
                                                 backend_work,backend_bytes,host_work,host_bytes);
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(h) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(h); } catch (...) {}
        throw;
    }
}
#define INSTANTIATE_PGEMM_RUN(B, T, TA, TB) \
    template void pgemm_run<B,T,TA,TB>(const DdlaHandle_t&, \
        Algorithm, int, char, char, int, int, int, \
        const T&, const TA*, const int*, const TB*, const int*, \
        const T&, T*, const int*, void*, std::size_t, void*, std::size_t)
#define INSTANTIATE_PGEMM_RUN_BACKEND(B) \
    INSTANTIATE_PGEMM_RUN(B, float, float, float); \
    INSTANTIATE_PGEMM_RUN(B, double, double, double); \
    INSTANTIATE_PGEMM_RUN(B, std::complex<float>, std::complex<float>, std::complex<float>); \
    INSTANTIATE_PGEMM_RUN(B, std::complex<double>, std::complex<double>, std::complex<double>); \
    INSTANTIATE_PGEMM_RUN(B, std::complex<float>, float, std::complex<float>); \
    INSTANTIATE_PGEMM_RUN(B, std::complex<float>, std::complex<float>, float); \
    INSTANTIATE_PGEMM_RUN(B, std::complex<double>, double, std::complex<double>); \
    INSTANTIATE_PGEMM_RUN(B, std::complex<double>, std::complex<double>, double)
#define DDLA_INSTANTIATE_STATIONARY(B, T, TA, TB) \
    template void stationary_run<B,T,TA,TB>(const DdlaHandle_t&, bool, char, char, int, int, int, \
        const T&, const TA*, const int*, const TB*, const int*, const T&, T*, const int*, \
        void*, std::size_t, void*, std::size_t)
#define DDLA_INSTANTIATE_STATIONARY_BACKEND(B) \
    DDLA_INSTANTIATE_STATIONARY(B, float, float, float); \
    DDLA_INSTANTIATE_STATIONARY(B, double, double, double); \
    DDLA_INSTANTIATE_STATIONARY(B, std::complex<float>, std::complex<float>, std::complex<float>); \
    DDLA_INSTANTIATE_STATIONARY(B, std::complex<double>, std::complex<double>, std::complex<double>); \
    DDLA_INSTANTIATE_STATIONARY(B, std::complex<float>, float, std::complex<float>); \
    DDLA_INSTANTIATE_STATIONARY(B, std::complex<float>, std::complex<float>, float); \
    DDLA_INSTANTIATE_STATIONARY(B, std::complex<double>, double, std::complex<double>); \
    DDLA_INSTANTIATE_STATIONARY(B, std::complex<double>, std::complex<double>, double)

DDLA_FOR_EACH_BACKEND(INSTANTIATE_PGEMM_RUN_BACKEND)
DDLA_FOR_EACH_BACKEND(DDLA_INSTANTIATE_STATIONARY_BACKEND)

#undef DDLA_INSTANTIATE_STATIONARY_BACKEND
#undef DDLA_INSTANTIATE_STATIONARY
#undef INSTANTIATE_PGEMM_RUN_BACKEND
#undef INSTANTIATE_PGEMM_RUN

}} // namespace ddla::pgemm_impl
