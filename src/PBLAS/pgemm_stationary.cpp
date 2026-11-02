#include "pgemm_impl.h"
#include "pgeadd_impl.h"
#include "pblas_workspace.h"
#include "comm_traits.h"
#include "transport_block.h"
#include "plan_util.h"
#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <stdexcept>

namespace ddla { namespace pgemm_impl {
namespace {

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

// The storage of an `rows x cols` helper matrix with the given blocks and
// source process, laid out as its descriptor says: LLD x local columns.
std::size_t described_size(const DdlaHandle_t& h, int* desc, int rows, int cols, int mb, int nb, int rs, int cs)
{
    pblas::describe(desc, h, rows, cols, mb, nb, rs, cs);
    return std::max(std::size_t(1), std::size_t(desc[DDLA_LLD_]) * num_loc(cols, nb, h->mypcol_, cs, h->npcols_));
}

// The scratch of the panel loop, sized for the widest panel (the loop's width
// never exceeds panel_block; every size is monotone in the width).  count[]
// holds the element counts of the branch's own buffers, in carving order:
// partial (T) | panel (the thin operand's type) | aligned (the thin
// operand's transposed copy, when its k runs along the other process
// dimension) | temporary (T, when the result is transposed back).  in[] and
// out[] hold the (backend, host) bytes of the two pgeadd calls of every panel:
// the thin operand's transpose and the block of C, alpha*op(result) + beta*C.
template <typename T, typename TX>
void panel_scratch(const DdlaHandle_t& h, bool keep_a, char ta, char tb,
                   int m, int n, int k, bool beta_is_nonzero, const int* da, const int* db, const int* dc,
                   std::size_t (&count)[4], std::size_t (&in)[2], std::size_t (&out)[2])
{
    const int* fixed = keep_a ? da : db;
    const int* thin = keep_a ? db : da;
    const bool k_rows = keep_a ? ta != 'N' : tb == 'N';
    const bool thin_k_rows = keep_a ? tb == 'N' : ta != 'N';
    const int kb = fixed[k_rows ? DDLA_MB_ : DDLA_NB_];
    const int ks = fixed[k_rows ? DDLA_RSRC_ : DDLA_CSRC_];
    const int fb = fixed[k_rows ? DDLA_NB_ : DDLA_MB_];
    const int fs = fixed[k_rows ? DDLA_CSRC_ : DDLA_RSRC_];
    const bool transpose_output = keep_a == k_rows;
    const int kloc = num_loc(k, kb, k_rows ? h->myprow_ : h->mypcol_, ks, k_rows ? h->nprows_ : h->npcols_);
    const int floc = num_loc(keep_a ? m : n, fb, k_rows ? h->mypcol_ : h->myprow_, fs, k_rows ? h->npcols_ : h->nprows_);
    const int panels = keep_a ? n : m;
    const int panel_block = dc[keep_a ? DDLA_NB_ : DDLA_MB_];
    const char tx = keep_a ? tb : ta;
    const int width = std::min(panel_block, panels);
    count[0] = std::max(std::size_t(1), std::size_t(floc) * width);
    count[1] = std::max(std::size_t(1), std::size_t(kloc) * width);
    count[2] = count[3] = 0;
    in[0] = in[1] = 0;
    if (thin_k_rows != k_rows) {
        const int view_m = thin_k_rows ? k : width;
        const int view_n = thin_k_rows ? width : k;
        int aligned_desc[DDLA_DLEN_];
        count[2] = described_size(h, aligned_desc, view_n, view_m,
            thin[DDLA_NB_], thin[DDLA_MB_], k_rows ? ks : 0, k_rows ? 0 : ks);
        pgeadd_impl::pgeadd_workspace_size<TX>(h, tx == 'C' ? 'C' : 'T', 'N', true, false,
            aligned_desc[DDLA_M_], aligned_desc[DDLA_N_], aligned_desc, &in[0], &in[1]);
    }
    const int cm = keep_a ? m : width, cn = keep_a ? width : n;
    if (transpose_output) {
        int temporary_desc[DDLA_DLEN_];
        count[3] = described_size(h, temporary_desc, cn, cm, dc[DDLA_NB_], dc[DDLA_MB_],
            k_rows ? 0 : fs, k_rows ? fs : 0);
    }
    pgeadd_impl::pgeadd_workspace_size<T>(h, transpose_output ? 'T' : 'N', 'N', true, beta_is_nonzero,
                                          cm, cn, dc, &out[0], &out[1]);
}

} // namespace

bool stationary_aligned(bool keep_a, char ta, char tb, const int* da, const int* db, const int* dc)
{
    const int* fixed = keep_a ? da : db;
    const int* thin = keep_a ? db : da;
    const bool k_rows = keep_a ? ta != 'N' : tb == 'N';
    const bool thin_k_rows = keep_a ? tb == 'N' : ta != 'N';
    const int kb = fixed[k_rows ? DDLA_MB_ : DDLA_NB_];
    const int ks = fixed[k_rows ? DDLA_RSRC_ : DDLA_CSRC_];
    const int fb = fixed[k_rows ? DDLA_NB_ : DDLA_MB_];
    const int fs = fixed[k_rows ? DDLA_CSRC_ : DDLA_RSRC_];
    const bool transpose_output = keep_a == k_rows;
    return !((thin_k_rows == k_rows && thin[k_rows ? DDLA_RSRC_ : DDLA_CSRC_] != ks) ||
             (!transpose_output && fs != dc[keep_a ? DDLA_RSRC_ : DDLA_CSRC_]));
}

// Backend [ partial | panel | aligned | temporary (panel_scratch) | the thin
// operand's pgeadd | C's pgeadd | the local gemm's scratch for the full and
// the last panel width (a mixed product only) ], host [ the thin operand's
// pgeadd | C's pgeadd ].
template <typename T, typename TA, typename TB>
void stationary_workspace_size(const DdlaHandle_t& h, bool keep_a, char ta, char tb,
                               int m, int n, int k, bool beta_is_nonzero,
                               const int* da, const int* db, const int* dc,
                               std::size_t* backend_bytes, std::size_t* host_bytes)
{
    const int* fixed = keep_a ? da : db;
    const int* thin = keep_a ? db : da;
    const bool k_rows = keep_a ? ta != 'N' : tb == 'N';
    const bool thin_k_rows = keep_a ? tb == 'N' : ta != 'N';
    const int kb = fixed[k_rows ? DDLA_MB_ : DDLA_NB_];
    const int ks = fixed[k_rows ? DDLA_RSRC_ : DDLA_CSRC_];
    const int fb = fixed[k_rows ? DDLA_NB_ : DDLA_MB_];
    const int fs = fixed[k_rows ? DDLA_CSRC_ : DDLA_RSRC_];
    const bool transpose_output = keep_a == k_rows;
    const int kloc = num_loc(k, kb, k_rows ? h->myprow_ : h->mypcol_, ks, k_rows ? h->nprows_ : h->npcols_);
    const int floc = num_loc(keep_a ? m : n, fb, k_rows ? h->mypcol_ : h->myprow_, fs, k_rows ? h->npcols_ : h->nprows_);
    const int panels = keep_a ? n : m;
    const int panel_block = dc[keep_a ? DDLA_NB_ : DDLA_MB_];
    std::size_t count[4], in[2], out[2], thin_bytes;
    if (keep_a) {
        panel_scratch<T, TB>(h, keep_a, ta, tb, m, n, k, beta_is_nonzero, da, db, dc, count, in, out);
        thin_bytes = detail::padded_bytes<TB>(count[1]) + detail::padded_bytes<TB>(count[2]);
    } else {
        panel_scratch<T, TA>(h, keep_a, ta, tb, m, n, k, beta_is_nonzero, da, db, dc, count, in, out);
        thin_bytes = detail::padded_bytes<TA>(count[1]) + detail::padded_bytes<TA>(count[2]);
    }
    const std::size_t bytes = detail::padded_bytes<T>(count[0]) + thin_bytes + detail::padded_bytes<T>(count[3])
                            + detail::padded_bytes<char>(in[0]) + detail::padded_bytes<char>(out[0]);
    const int width = std::min(panel_block, panels), tail = panels % panel_block;
    std::size_t gemm_bytes = 0;
    for (const int w : {width, tail}) {
        std::size_t q = 0;
        local_gemm_workspace_size<T, TA, TB>(h, keep_a ? floc : w, keep_a ? w : floc, kloc, &q);
        gemm_bytes = std::max(gemm_bytes, q);
    }
    *backend_bytes = bytes + gemm_bytes;
    *host_bytes = detail::padded_bytes<char>(in[1]) + detail::padded_bytes<char>(out[1]);
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
    const int pr = h->nprows_, pc = h->npcols_, r = h->myprow_, col = h->mypcol_;
    const int* fixed = keep_a ? da : db;
    const int* thin = keep_a ? db : da;
    const bool k_rows = keep_a ? ta != 'N' : tb == 'N';
    const bool thin_k_rows = keep_a ? tb == 'N' : ta != 'N';
    const int kb = fixed[k_rows ? DDLA_MB_ : DDLA_NB_];
    const int ks = fixed[k_rows ? DDLA_RSRC_ : DDLA_CSRC_];
    const int fb = fixed[k_rows ? DDLA_NB_ : DDLA_MB_];
    const int fs = fixed[k_rows ? DDLA_CSRC_ : DDLA_RSRC_];
    const bool transpose_output = keep_a == k_rows;
    const int kloc = num_loc(k, kb, k_rows ? h->myprow_ : h->mypcol_, ks, k_rows ? h->nprows_ : h->npcols_);
    const int floc = num_loc(keep_a ? m : n, fb, k_rows ? h->mypcol_ : h->myprow_, fs, k_rows ? h->npcols_ : h->nprows_);
    const int panels = keep_a ? n : m;
    const int panel_block = dc[keep_a ? DDLA_NB_ : DDLA_MB_];

    const TX* x = [&]() { if constexpr (keep_a) return b; else return a; }();
    const char tx = keep_a ? tb : ta;
    const CommScope reduction = k_rows ? CommScope::Col : CommScope::Row;

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

    int aligned_desc[DDLA_DLEN_], temporary_desc[DDLA_DLEN_], result_desc[DDLA_DLEN_];
    for (int p0 = 0; p0 < panels; ) {
        const int width = std::min(panel_block, panels - p0);
        const int c_owner = indxg2p(p0, panel_block,
                                   dc[keep_a ? DDLA_CSRC_ : DDLA_RSRC_], keep_a ? pc : pr);
        const int root = transpose_output ? 0 : c_owner;

        // A descriptor view of this one physical row/column block. Preserve
        // the input LLD, and shift only the pointer on ranks owning the block.
        int view[DDLA_DLEN_];
        std::copy(thin, thin + DDLA_DLEN_, view);
        view[DDLA_M_] = thin_k_rows ? k : width;
        view[DDLA_N_] = thin_k_rows ? width : k;
        const int owner = indxg2p(p0, panel_block,
            thin[thin_k_rows ? DDLA_CSRC_ : DDLA_RSRC_], thin_k_rows ? pc : pr);
        view[thin_k_rows ? DDLA_CSRC_ : DDLA_RSRC_] = owner;
        const int offset = num_loc(p0, panel_block, thin_k_rows ? col : r,
            thin[thin_k_rows ? DDLA_CSRC_ : DDLA_RSRC_], thin_k_rows ? pc : pr);
        const TX* input = x;
        if (input != nullptr && (thin_k_rows ? col : r) == owner)
            input += thin_k_rows ? std::size_t(offset) * thin[DDLA_LLD_] : offset;
        const int* input_desc = view;
        char panel_op = tx;
        if (thin_k_rows != k_rows) {
            // The thin panel's k runs along the other process dimension. One
            // transpose lays it out like the stationary k blocks (any source
            // coordinates), so the broadcast below is always untransposed.
            pblas::describe(aligned_desc, h, view[DDLA_N_], view[DDLA_M_], view[DDLA_NB_], view[DDLA_MB_],
                            k_rows ? ks : 0, k_rows ? 0 : ks);
            pgeadd_impl::pgeadd_run<Backend, TX>(h, tx=='C'?'C':'T', 'N',
                aligned_desc[DDLA_M_], aligned_desc[DDLA_N_],
                TX(1), input, 1, 1, view, TX(0), static_cast<const TX*>(nullptr), 1, 1, aligned_desc,
                aligned, 1, 1, aligned_desc, add_in, in[0], add_in_host, in[1]);
            input = aligned;
            input_desc = aligned_desc;
            panel_op = tx == 'N' ? 'T' : 'N';
        }
        // An untransposed transport stages nothing.
        transport_impl::transport_run<Backend, TX>(h, k_rows ? 'C' : 'R', 'N',
            input_desc[DDLA_M_], input_desc[DDLA_N_], input, 0, 0, input_desc, panel,
            nullptr, 0, nullptr, 0);
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

        int output[DDLA_DLEN_];
        std::copy(dc, dc + DDLA_DLEN_, output);
        const int cm = keep_a ? m : width, cn = keep_a ? width : n;
        output[DDLA_M_] = cm;
        output[DDLA_N_] = cn;
        output[keep_a ? DDLA_CSRC_ : DDLA_RSRC_] = c_owner;
        // Without a transpose the roots are C's owners and their partials
        // already have C's local layout; only a descriptor is needed.
        pblas::describe(result_desc, h, cm, cn, dc[DDLA_MB_], dc[DDLA_NB_],
                                output[DDLA_RSRC_], output[DDLA_CSRC_]);
        const T* result = partial;
        const int* rdesc = result_desc;
        char result_op = 'N';

        if (transpose_output) {
            // A transposed stationary input leaves the result distributed in
            // the opposite direction. Transpose the local reduced slab; the
            // final pgeadd redistributes it back into C's orientation.
            pblas::describe(temporary_desc, h, cn, cm, dc[DDLA_NB_], dc[DDLA_MB_],
                            k_rows ? root : fs, k_rows ? fs : root);
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
        const int co = num_loc(p0, panel_block, keep_a ? col : r,
                              dc[keep_a ? DDLA_CSRC_ : DDLA_RSRC_], keep_a ? pc : pr);
        T* dest = c;
        if (dest != nullptr) dest += keep_a ? std::size_t(co)*dc[DDLA_LLD_] : co;
        pgeadd_impl::pgeadd_run<Backend, T>(h, result_op, 'N', cm, cn, T(1), result, 1, 1, rdesc,
                        beta, dest, 1, 1, output, dest, 1, 1, output, add_out, out[0], add_out_host, out[1]);
        p0 += width;
    }
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
#define DDLA_INSTANTIATE_STATIONARY_SIZE(T, TA, TB) \
    template void stationary_workspace_size<T,TA,TB>(const DdlaHandle_t&, bool, char, char, int, int, int, bool, \
        const int*, const int*, const int*, std::size_t*, std::size_t*)
#if DDLA_HAS_CPU
DDLA_INSTANTIATE_STATIONARY_BACKEND(DdlaBackend::CPU);
#endif
#if DDLA_HAS_GPU
DDLA_INSTANTIATE_STATIONARY_BACKEND(DdlaBackend::GPU);
#endif
DDLA_INSTANTIATE_STATIONARY_SIZE(float, float, float);
DDLA_INSTANTIATE_STATIONARY_SIZE(double, double, double);
DDLA_INSTANTIATE_STATIONARY_SIZE(std::complex<float>, std::complex<float>, std::complex<float>);
DDLA_INSTANTIATE_STATIONARY_SIZE(std::complex<double>, std::complex<double>, std::complex<double>);
DDLA_INSTANTIATE_STATIONARY_SIZE(std::complex<float>, float, std::complex<float>);
DDLA_INSTANTIATE_STATIONARY_SIZE(std::complex<float>, std::complex<float>, float);
DDLA_INSTANTIATE_STATIONARY_SIZE(std::complex<double>, double, std::complex<double>);
DDLA_INSTANTIATE_STATIONARY_SIZE(std::complex<double>, std::complex<double>, double);
#undef DDLA_INSTANTIATE_STATIONARY_SIZE
#undef DDLA_INSTANTIATE_STATIONARY_BACKEND
#undef DDLA_INSTANTIATE_STATIONARY

}} // namespace ddla::pgemm_impl
