#ifndef DDLA_PBLAS_WORKSPACE_H
#define DDLA_PBLAS_WORKSPACE_H

#include <ddla/ddla.h>
#include <ddla/ddla_desc.h>
#include "ddla_stream.h"
#include "comm_traits.h"
#include "pgeadd_impl.h"

#include <algorithm>

namespace ddla { namespace pblas {

// Group local GPU updates without changing descriptor-block communication.
inline int contraction_width(const DdlaHandle_t& h,int nb,int k) noexcept
{return std::min(k,h->backend==DdlaBackend::GPU && k>=1024 ? std::max(nb,(1024/nb)*nb) : nb);}

// Derived descriptors in a trusted algorithm need index geometry only. The
// public descriptor constructor remains responsible for validating user input.
inline void describe(int* d,DdlaHandle_t h,int m,int n,int mb,int nb,int rs,int cs,int lld=-1) noexcept
{
    d[DDLA_DTYPE_]=DDLA_BLOCK_CYCLIC_2D;d[DDLA_CTXT_]=0;
    d[DDLA_M_]=m;d[DDLA_N_]=n;d[DDLA_MB_]=mb;d[DDLA_NB_]=nb;
    d[DDLA_RSRC_]=rs;d[DDLA_CSRC_]=cs;
    d[DDLA_LLD_]=std::max(lld,std::max(1,num_loc(m,mb,h->myprow_,rs,h->nprows_)));
}

// Replicate a one-block-wide, untransposed panel. Packing is a pitched copy
// only on its owning process row/column. A compact panel already in place
// needs just the broadcast.
template <DdlaBackend Backend, typename T>
void broadcast_panel(const DdlaHandle_t& h, bool column_panel, int m, int n,
                     const T* input, int i, int j, const int* d, T* output)
{
    if (m == 0 || n == 0) return;
    const int row0 = num_loc(i, d[DDLA_MB_], h->myprow_, d[DDLA_RSRC_], h->nprows_);
    const int col0 = num_loc(j, d[DDLA_NB_], h->mypcol_, d[DDLA_CSRC_], h->npcols_);
    const int rows = num_loc(i + m, d[DDLA_MB_], h->myprow_, d[DDLA_RSRC_], h->nprows_) - row0;
    const int cols = num_loc(j + n, d[DDLA_NB_], h->mypcol_, d[DDLA_CSRC_], h->npcols_) - col0;
    const int root = column_panel
        ? indxg2p(j, d[DDLA_NB_], d[DDLA_CSRC_], h->npcols_)
        : indxg2p(i, d[DDLA_MB_], d[DDLA_RSRC_], h->nprows_);
    const int packed_rows = column_panel ? rows : m;
    const int packed_cols = column_panel ? n : cols;
    const int coordinate = column_panel ? h->mypcol_ : h->myprow_;
    if (packed_rows == 0 || packed_cols == 0) return;
    if (coordinate == root) {
        const T* source = input + row0 + std::size_t(col0) * d[DDLA_LLD_];
        if (source != output || d[DDLA_LLD_] != packed_rows)
            DDLA_CHECK(ddlaMemcpy2D(h, output, std::size_t(packed_rows) * sizeof(T),
                source, std::size_t(d[DDLA_LLD_]) * sizeof(T),
                std::size_t(packed_rows) * sizeof(T), packed_cols));
    }
    commBcast<Backend>(h, column_panel ? CommScope::Row : CommScope::Col,
        output, std::size_t(packed_rows) * packed_cols, root);
}

inline bool panel_local_layout(bool column_panel, const int* source, const int* target)
{
    const int block = column_panel ? DDLA_MB_ : DDLA_NB_;
    const int owner = column_panel ? DDLA_RSRC_ : DDLA_CSRC_;
    return source[block] == target[block] && source[owner] == target[owner];
}

// Algorithm panels start at descriptor-block boundaries. Their short dimension
// occupies one block. The compact view retains the target distribution along
// the long dimension and places the short dimension on one broadcast root.
inline void panel_descriptor(int* d, DdlaHandle_t h, bool column_panel, char op,
                             int m, int n, int i, int j,
                             const int* source, const int* target)
{
    const bool transposed = op != 'N';
    const int rows = transposed ? n : m;
    const int cols = transposed ? m : n;
    const int mb = column_panel ? target[DDLA_MB_] : source[transposed ? DDLA_NB_ : DDLA_MB_];
    const int nb = column_panel ? source[transposed ? DDLA_MB_ : DDLA_NB_] : target[DDLA_NB_];
    int rs = 0, cs = 0;
    if (column_panel) {
        rs = indxg2p(transposed ? j : i, mb, target[DDLA_RSRC_], h->nprows_);
        if (!transposed) cs = indxg2p(j, source[DDLA_NB_], source[DDLA_CSRC_], h->npcols_);
    } else {
        cs = indxg2p(transposed ? i : j, nb, target[DDLA_CSRC_], h->npcols_);
        if (!transposed) rs = indxg2p(i, source[DDLA_MB_], source[DDLA_RSRC_], h->nprows_);
    }
    describe(d, h, rows, cols, mb, nb, rs, cs);
}

template <typename T>
void panel_workspace_size(const DdlaHandle_t& h, bool column_panel, char op,
                          int m, int n, const int* source, const int* target,
                          std::size_t* backend_bytes, std::size_t* host_bytes)
{
    *backend_bytes = *host_bytes = 0;
    if (m == 0 || n == 0 || (op == 'N' && panel_local_layout(column_panel, source, target)))
        return;
    int d[DDLA_DLEN_];
    panel_descriptor(d, h, column_panel, op, m, n, 0, 0, source, target);
    const bool general = op == 'N'
        || source[op == 'N' ? DDLA_MB_ : DDLA_NB_] != d[DDLA_MB_]
        || source[op == 'N' ? DDLA_NB_ : DDLA_MB_] != d[DDLA_NB_];
    pgeadd_impl::pgeadd_workspace_size<T>(h, op, 'N', true, false,
        d[DDLA_M_], d[DDLA_N_], d, backend_bytes, host_bytes, source, d, general);
}

// Materialize op(input) into the target panel layout, then replicate its short
// dimension. All redistribution belongs to pgeadd. The N path keeps
// the direct pack/broadcast route and uses no extra workspace.
template <DdlaBackend Backend, typename T>
void panel_run(const DdlaHandle_t& h, bool column_panel, char op, int m, int n,
               const T* input, int i, int j, const int* source, const int* target,
               T* output, void* work, std::size_t bytes, void* host, std::size_t host_bytes)
{
    if (m == 0 || n == 0) return;
    if (op == 'N' && panel_local_layout(column_panel, source, target)) {
        broadcast_panel<Backend>(h, column_panel, m, n, input, i, j, source, output);
        return;
    }
    int d[DDLA_DLEN_];
    panel_descriptor(d, h, column_panel, op, m, n, i, j, source, target);
    pgeadd_impl::pgeadd_run<Backend,T>(h, op, 'N', d[DDLA_M_], d[DDLA_N_],
        T(1), input, i + 1, j + 1, source, T(0), nullptr, 1, 1, d,
        output, 1, 1, d, work, bytes, host, host_bytes);
    broadcast_panel<Backend>(h, column_panel, d[DDLA_M_], d[DDLA_N_], output, 0, 0, d, output);
}

}}
#endif
