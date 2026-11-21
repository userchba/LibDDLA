#include "pgemm_geometry.h"
#include "instantiate.h"

#include <initializer_list>

namespace ddla { namespace pgemm_impl {
namespace {

template <typename T>
std::size_t ab_bytes(std::size_t count)
{
    if (count > (std::numeric_limits<std::size_t>::max() - 255) / sizeof(T))
        throw std::length_error("pgemm: panel workspace overflow");
    return detail::padded_bytes<T>(count);
}

void ab_add_bytes(std::size_t& total, std::size_t bytes)
{
    if (bytes > std::numeric_limits<std::size_t>::max() - total)
        throw std::length_error("pgemm: workspace sum overflow");
    total += bytes;
}

} // namespace

// Component 2.  The SUMMA branch (AB): backend [ A panel (TA) | B panel (TB) ]
// in two GPU slots or one CPU slot | B tile (TB) | transport of a transposed A | of a
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
    if (h->nprocs == 1) {
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
    const int bpg = ab_blocks_per_gemm(h, k, nb, panel_width);
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
    for (int slot = 0; slot < (h->backend == DdlaBackend::CPU ? 1 : 2); ++slot) {
        ab_add_bytes(*backend_bytes, ab_bytes<TA>(count[0]));
        ab_add_bytes(*backend_bytes, ab_bytes<TB>(count[1]));
    }
    ab_add_bytes(*backend_bytes, ab_bytes<TB>(count[2]));
    ab_add_bytes(*backend_bytes, ab_bytes<char>(panel_a_bytes[0]));
    ab_add_bytes(*backend_bytes, ab_bytes<char>(panel_b_bytes[0]));
    ab_add_bytes(*backend_bytes, gemm_bytes);
    ab_add_bytes(*host_bytes, ab_bytes<char>(panel_a_bytes[1]));
    ab_add_bytes(*host_bytes, ab_bytes<char>(panel_b_bytes[1]));
}

std::array<std::size_t,2> pgemm_staging_bytes(const DdlaHandle_t& h,Algorithm algorithm,
    bool product,char ta,char tb,int m,int n,int k,bool beta_is_nonzero,
    const int* da,const int* db,const int* dc,
    std::size_t a_bytes,std::size_t b_bytes,std::size_t c_bytes)
{
    std::array<std::size_t,2> bytes{};
    if(h->backend!=DdlaBackend::GPU || h->communication()!=DdlaCommunication::HostStagedMPI ||
       !product || !m || !n || !k)return bytes;
    // Every supported domain uses native local GEMM on one process.
    if(h->nprocs==1)return bytes;
    if(algorithm!=Algorithm::AB)
        return stationary_staging_bytes(h,algorithm==Algorithm::BC,ta,tb,m,n,k,beta_is_nonzero,
            da,db,dc,algorithm==Algorithm::BC?b_bytes:a_bytes,c_bytes);
    const int block=ab_block(ta,da);
    // Exactly the descriptor blocks transported by SUMMA, independent of how
    // many such blocks a local GEMM aggregates. Only geometry is visited.
    for(int g=0;g<k;) {
        const int width=std::min(block,k-g);
        pblas::staging_max(bytes,pblas::panel_staging_bytes(h,true,ta,
            ta=='N'?m:width,ta=='N'?width:m,ta=='N'?0:g,ta=='N'?g:0,da,dc,a_bytes));
        pblas::staging_max(bytes,pblas::panel_staging_bytes(h,false,tb,
            tb=='N'?width:n,tb=='N'?n:width,tb=='N'?g:0,tb=='N'?0:g,db,dc,b_bytes));
        g+=width;
    }
    return bytes;
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
std::array<std::size_t,2> stationary_staging_bytes(const DdlaHandle_t& h,bool keep_a,
    char ta,char tb,int m,int n,int k,bool beta_is_nonzero,
    const int* da,const int* db,const int* dc,std::size_t thin_bytes,std::size_t result_bytes)
{
    std::array<std::size_t,2> bytes{};
    if(h->backend!=DdlaBackend::GPU || h->communication()!=DdlaCommunication::HostStagedMPI || !m || !n || !k)
        return bytes;
    const char tx=keep_a?tb:ta;
    stationary_panels(h,keep_a,ta,tb,m,n,k,da,db,dc,[&](const StationaryGeometry& g,const StationaryPanel& p) {
        const int* input=p.view;
        if(g.thin_k_rows!=g.k_rows) {
            pblas::staging_max(bytes,pgeadd_impl::pgeadd_staging_bytes(h,tx=='C'?'C':'T','N',true,false,
                p.aligned[DDLA_M_],p.aligned[DDLA_N_],p.view,1,1,p.aligned,1,1,p.aligned,1,1,thin_bytes));
            input=p.aligned;
        }
        pblas::staging_max(bytes,pblas::broadcast_staging_bytes(h,g.k_rows,
            input[DDLA_M_],input[DDLA_N_],0,0,input,thin_bytes));
        // reduce_sum stages both input and output, even on a singleton scope.
        const auto reduced=pblas::communication_bytes(std::size_t(g.floc)*p.width,result_bytes);
        pblas::staging_max(bytes,{reduced,reduced});
        pblas::staging_max(bytes,pgeadd_impl::pgeadd_staging_bytes(h,g.transpose_output?'T':'N','N',true,beta_is_nonzero,
            p.cm,p.cn,g.transpose_output?p.temporary:p.result,1,1,p.output,1,1,p.output,1,1,result_bytes));
    });
    return bytes;
}
#define INSTANTIATE_PGEMM_SIZE(TYPE, TA, TB) \
    template void pgemm_impl::pgemm_workspace_size<TYPE, TA, TB>(const DdlaHandle_t&, \
        pgemm_impl::Algorithm, int, bool, char, char, int, int, int, bool, \
        const int*, const int*, const int*, std::size_t*, std::size_t*)
#define DDLA_INSTANTIATE_STATIONARY_SIZE(T, TA, TB) \
    template void stationary_workspace_size<T,TA,TB>(const DdlaHandle_t&, bool, char, char, int, int, int, bool, \
        const int*, const int*, const int*, std::size_t*, std::size_t*)

INSTANTIATE_PGEMM_SIZE(float, float, float);
INSTANTIATE_PGEMM_SIZE(double, double, double);
INSTANTIATE_PGEMM_SIZE(std::complex<float>, std::complex<float>, std::complex<float>);
INSTANTIATE_PGEMM_SIZE(std::complex<double>, std::complex<double>, std::complex<double>);
INSTANTIATE_PGEMM_SIZE(std::complex<float>, float, std::complex<float>);
INSTANTIATE_PGEMM_SIZE(std::complex<float>, std::complex<float>, float);
INSTANTIATE_PGEMM_SIZE(std::complex<double>, double, std::complex<double>);
INSTANTIATE_PGEMM_SIZE(std::complex<double>, std::complex<double>, double);
DDLA_INSTANTIATE_STATIONARY_SIZE(float, float, float);
DDLA_INSTANTIATE_STATIONARY_SIZE(double, double, double);
DDLA_INSTANTIATE_STATIONARY_SIZE(std::complex<float>, std::complex<float>, std::complex<float>);
DDLA_INSTANTIATE_STATIONARY_SIZE(std::complex<double>, std::complex<double>, std::complex<double>);
DDLA_INSTANTIATE_STATIONARY_SIZE(std::complex<float>, float, std::complex<float>);
DDLA_INSTANTIATE_STATIONARY_SIZE(std::complex<float>, std::complex<float>, float);
DDLA_INSTANTIATE_STATIONARY_SIZE(std::complex<double>, double, std::complex<double>);
DDLA_INSTANTIATE_STATIONARY_SIZE(std::complex<double>, std::complex<double>, double);

#undef DDLA_INSTANTIATE_STATIONARY_SIZE
#undef INSTANTIATE_PGEMM_SIZE

}} // namespace ddla::pgemm_impl
