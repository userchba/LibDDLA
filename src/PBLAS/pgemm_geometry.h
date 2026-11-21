#ifndef DDLA_PGEMM_GEOMETRY_H
#define DDLA_PGEMM_GEOMETRY_H

#include "pgemm_impl.h"
#include "pblas_workspace.h"
#include "pgeadd_impl.h"
#include "plan_util.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ddla { namespace pgemm_impl {
namespace {

// Shared panel geometry keeps workspace queries and scratch carving identical.
// The k block of the SUMMA branch: A's column (or, transposed, row) block.
inline int ab_block(char transa, const int* descA)
{
    return transa == 'N' ? descA[DDLA_NB_] : descA[DDLA_MB_];
}

// Descriptor-block transport is independent of the local GEMM depth. All
// N/T/C combinations append the same consecutive logical k blocks.
inline int ab_blocks_per_gemm(const DdlaHandle_t& h, int k, int nb, int panel_width)
{
    int blocks = std::max(1, panel_width / nb);
    // Bound total operand-panel storage by four block-rounded rank shares,
    // including 1 x P and P x 1 grids. GPU needs two slots for overlap;
    // synchronous CPU execution uses one larger panel to preserve GEMM depth.
    // A descriptor block remains the lower bound for concentrated inputs.
    const int peers = std::max(h->nprows_, h->npcols_);
    const int owned_blocks = k / peers / nb;
    const long long share = std::max(1, owned_blocks);
    // CPU uses at most four shares in its single slot, still capped by the
    // requested width. The peer bound prevents a full operand replica;
    // block-rounded grids retain their one-block lower bound.
    const int shares = h->backend == DdlaBackend::CPU && owned_blocks > 0
        ? std::min(4, std::max(1, peers - 1)) : 1;
    blocks = static_cast<int>(std::min<long long>(blocks, share * shares));
    return blocks;
}
inline int ab_panel_width(int k, int nb, int bpg)
{
    // The partial tail is a separate stage. Never reserve the requested
    // panel width when the entire contraction is smaller than that target.
    return std::max(std::min(k / nb, bpg) * nb, k % nb);
}
inline std::size_t ab_elements(std::size_t rows, std::size_t columns)
{
    if (columns && rows > std::numeric_limits<std::size_t>::max() / columns)
        throw std::length_error("pgemm: panel element count overflow");
    return std::max(std::size_t(1), rows * columns);
}
// Element counts of the SUMMA branch's staging, in carving order: the A panel
// (TA) and the B panel (TB) of one k slice, and the one-block B tile (TB).
// The B panel is stored as (kk x n_loc) with ld = kk so one gemm sees the
// whole run, but the panel copy writes a (w x n_loc) tile with ld = w; every
// block of a multi-block run is staged in the tile and copied into the panel
// with the panel pitch.  Each is a real allocation (>= 1 element) even at
// zero local extent.
inline void ab_counts(const DdlaHandle_t& h, char transa, char transb, int m, int n, int k,
               const int* descA, const int* descB, const int* descC, int panel_width,
               std::size_t (&count)[3])
{
    const int nprows = h->nprows_, npcols = h->npcols_, myprow = h->myprow_, mypcol = h->mypcol_;
    const int nb = ab_block(transa, descA);
    const std::size_t panel_max = ab_panel_width(k, nb, ab_blocks_per_gemm(h, k, nb, panel_width));
    const int m_loc_C = num_loc(m, descC[DDLA_MB_], myprow, descC[DDLA_RSRC_], nprows);
    const int n_loc_C = num_loc(n, descC[DDLA_NB_], mypcol, descC[DDLA_CSRC_], npcols);
    const int m_loc_A = num_loc((transa == 'N') ? m : k, descA[DDLA_MB_], myprow, descA[DDLA_RSRC_], nprows);
    const int n_loc_A = num_loc((transa == 'N') ? k : m, descA[DDLA_NB_], mypcol, descA[DDLA_CSRC_], npcols);
    const int m_loc_B = num_loc((transb == 'N') ? k : n, descB[DDLA_MB_], myprow, descB[DDLA_RSRC_], nprows);
    const int n_loc_B = num_loc((transb == 'N') ? n : k, descB[DDLA_NB_], mypcol, descB[DDLA_CSRC_], npcols);
    const std::size_t rows_a = (transa == 'N') ? std::max(m_loc_A, m_loc_C) : std::max(n_loc_A, m_loc_C);
    const std::size_t cols_b = (transb == 'N') ? std::max(n_loc_B, n_loc_C) : std::max(m_loc_B, n_loc_C);
    count[0] = ab_elements(rows_a, panel_max);
    count[1] = ab_elements(panel_max, cols_b);
    count[2] = ab_elements(std::min(nb, k), cols_b);
}

// Query redistribution once for the largest A/B panel. Aligned N panels use
// the direct copy/broadcast path and need no additional scratch.
template <typename TA, typename TB>
inline void ab_panel_bytes(const DdlaHandle_t& h, char transa, char transb, int m, int n, int nb,
                    const int* descA, const int* descB, const int* descC,
                    std::size_t (&panel_a)[2], std::size_t (&panel_b)[2])
{
    pblas::panel_workspace_size<TA>(h,true,transa,transa=='N'?m:nb,transa=='N'?nb:m,
        descA,descC,&panel_a[0],&panel_a[1]);
    pblas::panel_workspace_size<TB>(h,false,transb,transb=='N'?nb:n,transb=='N'?n:nb,
        descB,descC,&panel_b[0],&panel_b[1]);
}
// The storage of an `rows x cols` helper matrix with the given blocks and
// source process, laid out as its descriptor says: LLD x local columns.
inline std::size_t described_size(const DdlaHandle_t& h, int* desc, int rows, int cols, int mb, int nb, int rs, int cs)
{
    pblas::describe(desc, h, rows, cols, mb, nb, rs, cs);
    return std::max(std::size_t(1), std::size_t(desc[DDLA_LLD_]) * num_loc(cols, nb, h->mypcol_, cs, h->npcols_));
}

// Shared pure descriptor geometry. Preparation and execution traverse the
// same panel views; no second redistribution schedule is maintained.
struct StationaryGeometry {
    bool k_rows,thin_k_rows,transpose_output;
    int ks,fs,kloc,floc,panels,block;
};
inline StationaryGeometry stationary_geometry(const DdlaHandle_t& h,bool keep_a,char ta,char tb,
    int m,int n,int k,const int* da,const int* db,const int* dc)
{
    const int* fixed=keep_a?da:db;
    const bool k_rows=keep_a?ta!='N':tb=='N';
    const int kb=fixed[k_rows?DDLA_MB_:DDLA_NB_],ks=fixed[k_rows?DDLA_RSRC_:DDLA_CSRC_];
    const int fb=fixed[k_rows?DDLA_NB_:DDLA_MB_],fs=fixed[k_rows?DDLA_CSRC_:DDLA_RSRC_];
    return {k_rows,keep_a?tb=='N':ta!='N',keep_a==k_rows,ks,fs,
        num_loc(k,kb,k_rows?h->myprow_:h->mypcol_,ks,k_rows?h->nprows_:h->npcols_),
        num_loc(keep_a?m:n,fb,k_rows?h->mypcol_:h->myprow_,fs,k_rows?h->npcols_:h->nprows_),
        keep_a?n:m,dc[keep_a?DDLA_NB_:DDLA_MB_]};
}
struct StationaryPanel {
    int width,cm,cn,root,input_owner,input_offset,output_offset;
    int view[DDLA_DLEN_],aligned[DDLA_DLEN_],output[DDLA_DLEN_];
    int result[DDLA_DLEN_],temporary[DDLA_DLEN_];
};
template<class Visit>
void stationary_panels(const DdlaHandle_t& h,bool keep_a,char ta,char tb,
    int m,int n,int k,const int* da,const int* db,const int* dc,Visit visit)
{
    const auto g=stationary_geometry(h,keep_a,ta,tb,m,n,k,da,db,dc);
    const int* thin=keep_a?db:da;
    for(int p0=0;p0<g.panels;) {
        StationaryPanel p;
        p.width=std::min(g.block,g.panels-p0);
        const int c_owner=indxg2p(p0,g.block,dc[keep_a?DDLA_CSRC_:DDLA_RSRC_],keep_a?h->npcols_:h->nprows_);
        p.root=g.transpose_output?0:c_owner;
        std::copy(thin,thin+DDLA_DLEN_,p.view);
        p.view[DDLA_M_]=g.thin_k_rows?k:p.width;
        p.view[DDLA_N_]=g.thin_k_rows?p.width:k;
        p.input_owner=indxg2p(p0,g.block,thin[g.thin_k_rows?DDLA_CSRC_:DDLA_RSRC_],g.thin_k_rows?h->npcols_:h->nprows_);
        p.view[g.thin_k_rows?DDLA_CSRC_:DDLA_RSRC_]=p.input_owner;
        p.input_offset=num_loc(p0,g.block,g.thin_k_rows?h->mypcol_:h->myprow_,
            thin[g.thin_k_rows?DDLA_CSRC_:DDLA_RSRC_],g.thin_k_rows?h->npcols_:h->nprows_);
        if(g.thin_k_rows!=g.k_rows)
            pblas::describe(p.aligned,h,p.view[DDLA_N_],p.view[DDLA_M_],p.view[DDLA_NB_],p.view[DDLA_MB_],
                g.k_rows?g.ks:0,g.k_rows?0:g.ks);
        p.cm=keep_a?m:p.width;p.cn=keep_a?p.width:n;
        std::copy(dc,dc+DDLA_DLEN_,p.output);
        p.output[DDLA_M_]=p.cm;p.output[DDLA_N_]=p.cn;
        p.output[keep_a?DDLA_CSRC_:DDLA_RSRC_]=c_owner;
        pblas::describe(p.result,h,p.cm,p.cn,dc[DDLA_MB_],dc[DDLA_NB_],p.output[DDLA_RSRC_],p.output[DDLA_CSRC_]);
        if(g.transpose_output)
            pblas::describe(p.temporary,h,p.cn,p.cm,dc[DDLA_NB_],dc[DDLA_MB_],
                g.k_rows?p.root:g.fs,g.k_rows?g.fs:p.root);
        p.output_offset=num_loc(p0,g.block,keep_a?h->mypcol_:h->myprow_,
            dc[keep_a?DDLA_CSRC_:DDLA_RSRC_],keep_a?h->npcols_:h->nprows_);
        visit(g,p);
        p0+=p.width;
    }
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
}} // namespace ddla::pgemm_impl

#endif // DDLA_PGEMM_GEOMETRY_H
