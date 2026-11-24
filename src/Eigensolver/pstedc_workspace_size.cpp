#include "pstedc_geometry.h"

#include <algorithm>
#include <climits>
#include <stdexcept>

namespace ddla::spectral_impl {

// The buffers of pstedc_run in its carving order (listed in pstedc_impl.h).
template <class R>
void stedc_rows_workspace_size(const DdlaHandle_t& h, const PstedcVectorPlan& p, std::size_t* backend_bytes,
                                std::size_t* host_bytes)
{
    using detail::padded_bytes;
    *backend_bytes = 0;
    *host_bytes = 0;
    if (!p.n)
        return;
    const bool gpu = p.backend == DdlaBackend::GPU;
    const std::size_t n = p.n;
    const int leaves = pstedc_leaves(p), mine = pstedc_my_leaves(leaves, p.rank, p.ranks);
    const std::size_t m = pstedc_max_leaf(p), workers = std::min(p.threads, mine);
    const std::size_t slab = std::size_t(std::max(1, detail::RowShare(p.n, p.ranks).count(p.rank))) * n;
    std::size_t backend = p.borrow_q ? 0 : padded_bytes<R>(slab);
    if (leaves > 1 && !p.borrow_q2) backend += padded_bytes<R>(slab);
    std::size_t tail = leaves > 1 ? padded_bytes<R>(n * p.strip) : 0;
    if (gpu) {
        if (leaves > 1)
            tail += padded_bytes<R>(5*n) + padded_bytes<int>(2*n) + padded_bytes<R>(2*n);
        tail += padded_bytes<int>(n);
    }
    const auto local = pstedc_rank_plan<R>(h,p);
    if (gpu) tail = std::max(tail, local.backend_bytes);
    backend += tail;
    std::size_t retained = 0;
    for (int i = p.rank; i < leaves; i += p.ranks) {
        const std::size_t size = pstedc_leaf_geometry(p, i);
        retained += size * size;
    }
    const std::size_t kept =
        2 * padded_bytes<int>(leaves) + padded_bytes<int>(n) + padded_bytes<std::size_t>(mine + 1);
    std::size_t leaf = padded_bytes<R>(retained) + padded_bytes<R>(p.ranks > 1 ? m * m : 0) +
                             padded_bytes<int>(mine) + padded_bytes<R>(workers * (m * m + 6 * m + 1)) +
                             padded_bytes<int>(workers * (5 * m + 3));
    if (p.rank_leaf) {
        const auto native = !gpu && local.backend_bytes > padded_bytes<R>(slab)
            ? local.backend_bytes : 0;
        const auto publication = 2*padded_bytes<int>(p.ranks) +
            padded_bytes<R>(m * std::min(p.strip, std::max(1,INT_MAX / int(m))));
        leaf = std::max({publication, local.host_bytes, native});
    }
    std::size_t merge = 0;
    if (leaves > 1) {
        const std::size_t split = p.split ? n : 0, norm_workers = std::max(1, std::min(p.threads, p.n));
        merge = 3 * padded_bytes<R>(n) + padded_bytes<R>(2 * n) + 6 * padded_bytes<R>(n) +
                padded_bytes<R>(4 * n) + padded_bytes<R>(gpu ? 5 * n : 0) +
                padded_bytes<R>(n * std::max(std::size_t(p.root_chunk), norm_workers)) +
                2 * padded_bytes<R>(split) + 2 * padded_bytes<int>(n) + padded_bytes<int>(2 * n) +
                2 * padded_bytes<int>(n) + 2 * padded_bytes<int>(p.ranks) + padded_bytes<int>(p.root_chunk) +
                2 * padded_bytes<int>(split) + padded_bytes<char>(split);
    }
    std::size_t host = kept + std::max(leaf, merge);
    if (!gpu) {
        backend += host;
        host = 0;
    }
    *backend_bytes = backend;
    *host_bytes = host;
}

template void stedc_rows_workspace_size<float>(const DdlaHandle_t&, const PstedcVectorPlan&, std::size_t*,
                                                std::size_t*);
template void stedc_rows_workspace_size<double>(const DdlaHandle_t&, const PstedcVectorPlan&, std::size_t*,
                                                 std::size_t*);

template <class R>
void pstedc_workspace_size(const DdlaHandle_t& h, const PstedcVectorPlan& p,
                           std::size_t* backend_bytes, std::size_t* host_bytes)
{
    if (p.n && (local_stedc_cpu(p) || (p.backend == DdlaBackend::GPU && p.ranks == 1))) {
        const auto local = pstedc_local_plan<R>(h, p, 'I');
        *backend_bytes = local.backend_bytes;
        *host_bytes = local.host_bytes;
        if (p.backend == DdlaBackend::CPU)
            *backend_bytes += detail::padded_bytes<R>(std::max(1, p.n - 1));
        return;
    }
    stedc_rows_workspace_size<R>(h, p, backend_bytes, host_bytes);
}
template void pstedc_workspace_size<float>(const DdlaHandle_t&, const PstedcVectorPlan&, std::size_t*, std::size_t*);
template void pstedc_workspace_size<double>(const DdlaHandle_t&, const PstedcVectorPlan&, std::size_t*, std::size_t*);

// Values-only host scratch; the vector case uses exactly the row engine's layout.
template <class R>
void pstedc_workspace_size(const DdlaHandle_t& h, const PstedcPlan& p, std::size_t* backend_bytes,
                           std::size_t* host_bytes)
{
    using detail::padded_bytes;
    if (p.mode != PstedcMode::values) {
        pstedc_workspace_size<R>(h, p.rows, backend_bytes, host_bytes);
        return;
    }
    *backend_bytes = 0;
    *host_bytes = 0;
    const auto& a = p.rows;
    if (!a.n)
        return;
    if(local_stedc_cpu(a)) {
        *backend_bytes = pstedc_local_plan<R>(h, a, 'N').backend_bytes +
                         padded_bytes<R>(std::max(1, a.n - 1));
        return;
    }
    const std::size_t n = a.n;
    const int leaves = pstedc_leaves(a.n, a.leaf), mine = pstedc_my_leaves(leaves, a.rank, a.ranks);
    const std::size_t m = pstedc_leaf_geometry(a.n, leaves, leaves - 1);
    const std::size_t workers = std::min(1, mine);
    const bool single = leaves == 1;
    const std::size_t leaf_work = single ? n : m * m + 6 * m + 1, leaf_iwork = single ? 0 : 5 * m + 3;
    if (!single && (leaf_work - 2 * m > INT_MAX || leaf_iwork > INT_MAX))
        throw std::length_error("pstedc: leaf LAPACK workspace exceeds integer range");
    const std::size_t kept =
        2 * padded_bytes<int>(leaves) + padded_bytes<int>(n) + (single ? 0 : 2 * padded_bytes<R>(n));
    const std::size_t leaf = padded_bytes<R>(workers * leaf_work) + padded_bytes<int>(workers * leaf_iwork) +
                             padded_bytes<int>(mine) + padded_bytes<R>(single ? 0 : m * m);
    std::size_t merge = 0;
    if (!single)
        merge = 3 * padded_bytes<R>(n) + padded_bytes<R>(2 * n) + 2 * padded_bytes<R>(n) +
                padded_bytes<R>(dc_delta_count<R>(p)) + 2 * padded_bytes<R>(n) + padded_bytes<R>(3 * n) +
                2 * padded_bytes<int>(n) + padded_bytes<int>(2 * n) + 2 * padded_bytes<int>(n) +
                2 * padded_bytes<int>(a.ranks) + 4 * padded_bytes<R>(n);
    const std::size_t bytes = kept + std::max(leaf, merge);
    if (a.backend == DdlaBackend::CPU)
        *backend_bytes = bytes;
    else
        *host_bytes = bytes;
}

template void pstedc_workspace_size<float>(const DdlaHandle_t&, const PstedcPlan&, std::size_t*, std::size_t*);
template void pstedc_workspace_size<double>(const DdlaHandle_t&, const PstedcPlan&, std::size_t*, std::size_t*);

} // namespace ddla::spectral_impl
