#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "BLAS/blas_impl.h"
#include "check_agree.h"
#include "ddla_desc.h"
#include "ddla_stream.h"
#include "instantiate.h"
#include "pgeqrf_impl.h"
#include "require_backend.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ddla {
namespace {

// The adjoint of each local block belongs to (old column, old row) on the
// transposed grid, with opposite rank ordering. Borrow the same MPI/CCL
// communicators with row/column roles swapped; no resource is created/freed.
// A handle is already exclusive to its calling application thread. QR ends
// its queued work before returning, and this guard restores topology on throw.
class TransposedGrid {
    DdlaHandle_t h_;
    void transpose() noexcept
    {
        std::swap(h_->nprows_, h_->npcols_);
        std::swap(h_->myprow_, h_->mypcol_);
        std::swap(h_->row_comm, h_->col_comm);
        std::swap(h_->nccl_row_comm, h_->nccl_col_comm);
        h_->major = h_->major == 'R' ? 'C' : 'R';
    }
public:
    explicit TransposedGrid(const DdlaHandle_t& h) noexcept : h_(h) { transpose(); }
    ~TransposedGrid() noexcept { transpose(); }
    TransposedGrid(const TransposedGrid&) = delete;
    TransposedGrid& operator=(const TransposedGrid&) = delete;
};

template<DdlaBackend Backend, class T>
void factorize(const DdlaHandle_t& h, bool rq, int m, int n, T* a, const int* da, T* tau, int& info)
{
    info = 0;
    const char* name = rq ? "pgerqf" : "pgelqf";
    int ml = 0, nl = 0;
    const bool inplace = detail::agree_check(h, name, [&](detail::Agree& agree) {
        detail::require_backend<Backend>(h, name);
        check_desc(da, h);
        agree.mode("LQ/RQ factorization", rq ? 1 : 0);
        if (h->communication_group_depth)
            throw std::invalid_argument(std::string(name) + ": active communication group");
        if (m < 0 || n < 0 || m > da[DDLA_M_] || n > da[DDLA_N_])
            throw std::invalid_argument(std::string(name) + ": dimensions outside descriptor");
        ml = num_loc(m, da[DDLA_MB_], h->myprow_, da[DDLA_RSRC_], h->nprows_);
        nl = num_loc(n, da[DDLA_NB_], h->mypcol_, da[DDLA_CSRC_], h->npcols_);
        if (ml > 0 && nl > 0 && !a)
            throw std::invalid_argument(std::string(name) + ": null local matrix");
        const int first = rq ? m - std::min(m, n) : 0;
        const int last = first + std::min(m, n);
        if (!tau && num_loc(last, da[DDLA_MB_], h->myprow_, da[DDLA_RSRC_], h->nprows_) >
                    num_loc(first, da[DDLA_MB_], h->myprow_, da[DDLA_RSRC_], h->nprows_))
            throw std::invalid_argument(std::string(name) + ": null local tau");
        const bool panel = std::min(m, n) > 0 && qr_impl::qr_panel_inplace_env();
        agree.mode("DDLA_QR_PANEL", panel ? 1 : 0);
        return panel;
    });
    int signature[9] = {m, n};
    detail::descriptor_signature(signature + 2, da);
    detail::agree_signature(h, name, signature);
    if (m == 0 || n == 0) return;

    if (h->nprocs == 1 && !inplace) {
        const int k = std::min(m, n), first = rq ? m - k : 0;
        vector<T> device_tau(h);
        T* factors = tau + first;
        if constexpr (Backend == DdlaBackend::GPU) {
            device_tau.resize(k);
            factors = device_tau.data();
        }
        if (rq) gerqf(h, m, n, a, da[DDLA_LLD_], factors);
        else gelqf(h, m, n, a, da[DDLA_LLD_], factors);
        if constexpr (Backend == DdlaBackend::GPU) {
            DDLA_CHECK(ddlaMemcpy(h, tau + first, factors, std::size_t(k) * sizeof(T)));
            DDLA_CHECK(ddlaSynchronize(h));
        }
        return;
    }

    const int ld = std::max(1, nl);
    int dt[DDLA_DLEN_];
    std::copy(da, da + DDLA_DLEN_, dt);
    dt[DDLA_M_] = n;
    dt[DDLA_N_] = m;
    std::swap(dt[DDLA_MB_], dt[DDLA_NB_]);
    std::swap(dt[DDLA_RSRC_], dt[DDLA_CSRC_]);
    dt[DDLA_LLD_] = ld;
    vector<T> adjoint(h);
    detail::agree_local(h, "LQ/RQ local adjoint storage", [&] {
        adjoint.resize(std::size_t(ld) * ml);
    });
    try {
        if (ml > 0 && nl > 0)
            blas_impl::geam_run<Backend, T>(h, 'C', 'N', nl, ml, T(1), a, da[DDLA_LLD_],
                T(0), static_cast<const T*>(nullptr), nl, adjoint.data(), ld, h->stream);
        {
            TransposedGrid grid(h);
            const auto plan = qr_impl::qr_geometry(h, dt, rq ? 'L' : 'U', n, m, 0, /*panel_width_limit=*/64);
            std::size_t backend_bytes = 0, host_bytes = 0;
            vector<char> work(h);
            std::vector<char> host;
            detail::agree_local(h, "LQ/RQ factorization workspace", [&] {
                qr_impl::qr_block_workspace_size<T>(h, plan, false, false, inplace,
                    adjoint.data(), ld, tau, &backend_bytes, &host_bytes);
                work.resize(backend_bytes);
                host.resize(host_bytes);
                const auto staged = qr_impl::qr_block_staging_bytes<T>(h, plan, dt, false, inplace, false);
                if (staged[0] || staged[1]) h->host_buffers<std::byte>(staged[0], staged[1]);
            });
            try {
                const T* factors = qr_impl::qr_block_run<Backend, T>(h, adjoint.data(), dt, plan,
                    false, false, inplace, nullptr, 0, nullptr, 0,
                    work.data(), backend_bytes, host.data(), host_bytes);
                // Transposed local columns are the original local rows.
                const int lo = num_loc(plan.jr_lo, plan.nb, plan.mypcol, plan.csrc, plan.npcols);
                const int hi = num_loc(plan.jr_hi, plan.nb, plan.mypcol, plan.csrc, plan.npcols);
                if (hi > lo)
                    DDLA_CHECK(ddlaMemcpy(h, tau + lo, factors + lo, std::size_t(hi - lo) * sizeof(T)));
                DDLA_CHECK(ddlaSynchronize(h));
            } catch (...) {
                // Retire queued host reads before the metadata scratch is freed.
                if constexpr (Backend == DdlaBackend::GPU)
                    try { (void)ddlaSynchronize(h); } catch (...) {}
                throw;
            }
        }
        if (ml > 0 && nl > 0)
            blas_impl::geam_run<Backend, T>(h, 'C', 'N', ml, nl, T(1), adjoint.data(), ld,
                T(0), static_cast<const T*>(nullptr), ml, a, da[DDLA_LLD_], h->stream);
        DDLA_CHECK(ddlaSynchronize(h));
    } catch (...) {
        if constexpr (Backend == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(h); } catch (...) {}
        throw;
    }
}

} // namespace

template<DdlaBackend Backend, class T>
void pgelqf(const DdlaHandle_t& h, int m, int n, T* a, const int* da, T* tau, int& info)
{
    factorize<Backend, T>(h, false, m, n, a, da, tau, info);
}

template<DdlaBackend Backend, class T>
void pgerqf(const DdlaHandle_t& h, int m, int n, T* a, const int* da, T* tau, int& info)
{
    factorize<Backend, T>(h, true, m, n, a, da, tau, info);
}

#define INSTANTIATE_ROW_FACTORS(B, T) \
    template void pgelqf<B, T>(const DdlaHandle_t&, int, int, T*, const int*, T*, int&); \
    template void pgerqf<B, T>(const DdlaHandle_t&, int, int, T*, const int*, T*, int&)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_ROW_FACTORS)
#undef INSTANTIATE_ROW_FACTORS
} // namespace ddla
