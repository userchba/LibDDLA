#ifndef DDLA_PSTEDC_GEOMETRY_H
#define DDLA_PSTEDC_GEOMETRY_H

#include "plan_util.h"
#include "stedc_impl.h"

#include <algorithm>
#include <climits>
#include <stdexcept>

namespace ddla::spectral_impl {
namespace {

// The local LAPACK interface uses 32-bit workspace counts. Larger vector
// problems keep the existing bounded-leaf engine.
inline bool local_stedc_cpu(const PstedcVectorPlan& p) {
    return p.backend==DdlaBackend::CPU && p.ranks==1 &&
        std::size_t(p.n)*(std::size_t(p.n)+4)+1<=std::size_t(INT_MAX);
}
template<class R>
StedcPlan pstedc_local_plan(const DdlaHandle_t& h, const PstedcVectorPlan& p, char job)
{
    if (p.backend == DdlaBackend::GPU) {
        // Real I output already has the checked row geometry and borrowing
        // decisions. Use STEDC execution without changing the checked
        // small-problem footprint or the explicit strip policy.
        StedcPlan local;
        local.compz = job;
        local.n = p.n;
        local.rows = p;
        stedc_rows_workspace_size<R>(h, p, &local.engine_backend, &local.engine_host);
        local.backend_bytes = local.engine_backend;
        local.host_bytes = local.engine_host;
        return local;
    }
    // Keep the existing native real LAPACK workspace bound. Unlike the public
    // four-type planner this needs no vendor query during reusable execution.
    StedcPlan local;
    local.compz = job;
    local.n = p.n;
    local.lwork = job == 'I' ? int(std::size_t(p.n) * (std::size_t(p.n) + 4) + 1) : 1;
    local.liwork = job == 'I' ? 3 + 5 * p.n : 1;
    local.backend_bytes = detail::padded_bytes<R>(local.lwork) +
                          detail::padded_bytes<int>(local.liwork);
    return local;
}

// A local factor uses the two idle outer row slabs for its output and Q.
// The remaining local workspace shares the outer strip and host phase.
template<class R>
StedcPlan pstedc_rank_plan(const DdlaHandle_t& h, const PstedcVectorPlan& p)
{
    if (!p.rank_leaf || p.rank >= pstedc_leaves(p)) return {};
    const int m = pstedc_leaf_geometry(p, p.rank);
    if (p.backend == DdlaBackend::CPU) {
        PstedcVectorPlan local;
        local.n = m;
        if (std::size_t(m) * (std::size_t(m) + 4) + 1 > std::size_t(INT_MAX))
            throw std::length_error("pstedc: local LAPACK workspace exceeds integer range");
        return pstedc_local_plan<R>(h, local, 'I');
    }
    auto local = stedc_plan<R>(h, 'I', m, m);
    local.rows.borrow_q = true;
    stedc_rows_workspace_size<R>(h, local.rows, &local.engine_backend, &local.engine_host);
    local.backend_bytes = local.engine_backend;
    local.host_bytes = local.engine_host;
    return local;
}

// Bounded root-delta cache for eigenvalues only; at least one column is kept.
template <typename rt>
std::size_t dc_delta_count(const spectral_impl::PstedcPlan& p)
{
    const int ranks = p.rows.ranks;
    const std::size_t n = p.rows.n;
    const std::size_t limit = std::size_t(p.keep_mb) * (std::size_t(1) << 20) / sizeof(rt);
    // Recompute needs K entries. A kept non-top block has at most
    // ceil(n/2) rows, and its retained deltas never exceed the frozen
    // keep limit (K<=2 is the documented unconditional exception).
    // This bounds all data-dependent K without building a schedule or
    // inspecting the tridiagonal data in the size function.
    const std::size_t half = (n + 1) / 2, half_square = half * half;
    const std::size_t share = half * ((half + ranks - 1) / ranks + 1);
    const std::size_t small = std::min(n, std::size_t(32) * ranks);
    const std::size_t unsplit = std::min(half, small) * std::min(half, small);
    const std::size_t capacity = ranks == 1 ? half_square : std::min(half_square, std::max(share, unsplit));
    return std::max({n, std::size_t(4), std::min(limit, capacity)});
}

} // namespace
} // namespace ddla::spectral_impl

#endif // DDLA_PSTEDC_GEOMETRY_H
