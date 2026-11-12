#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "BLAS/blas_impl.h"
#include "check_agree.h"
#include "comm_traits.h"
#include "ddla_desc.h"
#include "helpers.h"
#include "lapack_impl.h"
#include "mpi_datatype.h"
#include "punmtr_impl.h"
#include "require_backend.h"
#include "instantiate.h"
#include <algorithm>
#include <climits>
#include <complex>
#include <type_traits>
#include <vector>

namespace ddla {
namespace qr_apply_impl {
struct Geometry {
    char side = 'L', trans = 'N';
    int m = 0, n = 0, k = 0, nq = 0, rows = 0, cols = 0, lda = 1, ldc = 1;
    int mb = 1, nb = 1, rsrc = 0, csrc = 0, cap = 0;
};

template <bool Distributed, class Function>
void checked_setup(const DdlaHandle_t &h, Function &&function) {
    if constexpr (Distributed)
        detail::agree_check(h, "qr application", [&](detail::Agree &) { function(); });
    else
        function();
}

template <DdlaBackend B, bool Distributed, bool Generate, class T>
Geometry check(const DdlaHandle_t &h, char side, char trans, int m, int n, int k, const T *a,
               int lda, const int *da, const T *tau, T *c, int ldc, const int *dc) {
    Geometry p;
    checked_setup<Distributed>(h, [&] {
        detail::require_backend<B>(h, "qr application");
        p.side = blas_impl::upper(side);
        p.trans = blas_impl::upper(trans);
        constexpr bool real = std::is_arithmetic_v<T>;
        if ((p.side != 'L' && p.side != 'R') ||
            (p.trans != 'N' && p.trans != 'C' && !(real && p.trans == 'T')))
            throw std::invalid_argument("qr application invalid side or transpose");
        if (real && p.trans == 'T')
            p.trans = 'C';
        p.m = m;
        p.n = n;
        p.k = k;
        p.nq = p.side == 'L' ? m : n;
        if (m < 0 || n < 0 || k < 0 || k > p.nq || (Generate && (n > m || k > n)))
            throw std::invalid_argument("qr application invalid dimensions");
        p.cap = std::min(k, 32);
        if (p.cap && p.nq > INT_MAX / 2 / p.cap)
            throw std::overflow_error("qr application panel collective exceeds int");
        int ar = p.nq, ac = k, tc = k;
        if constexpr (Distributed) {
            check_desc(da, h);
            check_desc(dc, h);
            if (p.nq > da[DDLA_M_] || k > da[DDLA_N_] || m > dc[DDLA_M_] || n > dc[DDLA_N_])
                throw std::invalid_argument("qr application extents exceed descriptors");
            p.lda = da[DDLA_LLD_];
            p.ldc = dc[DDLA_LLD_];
            p.mb = da[DDLA_MB_];
            p.nb = da[DDLA_NB_];
            p.rsrc = da[DDLA_RSRC_];
            p.csrc = da[DDLA_CSRC_];
            ar = num_loc(p.nq, p.mb, h->myprow_, p.rsrc, h->nprows_);
            ac = tc = num_loc(k, p.nb, h->mypcol_, p.csrc, h->npcols_);
            p.rows = num_loc(m, dc[DDLA_MB_], h->myprow_, dc[DDLA_RSRC_], h->nprows_);
            p.cols = num_loc(n, dc[DDLA_NB_], h->mypcol_, dc[DDLA_CSRC_], h->npcols_);
        } else {
            p.lda = lda;
            p.ldc = ldc;
            p.rows = m;
            p.cols = n;
            if (lda < std::max(1, p.nq) || ldc < std::max(1, m))
                throw std::invalid_argument("qr application invalid leading dimension");
        }
        if (p.cap && std::max(1, p.side == 'L' ? p.cols : p.rows) > INT_MAX / 2 / p.cap)
            throw std::overflow_error("qr application product collective exceeds int");
        if (!m || !n)
            return;
        if ((Generate || k) && p.rows && p.cols && !c)
            throw std::invalid_argument("qr application missing output");
        if (!k)
            return;
        if ((ar && ac && !a) || (tc && !tau))
            throw std::invalid_argument("qr application missing reflector data");
        if constexpr (!Generate) {
            if (detail::matrix_overlaps(a, p.lda, ar, ac, c, p.ldc, p.rows, p.cols) ||
                detail::memory_overlaps(
                    tau, std::size_t(tc) * sizeof(T), c,
                    p.rows && p.cols ? (std::size_t(p.cols - 1) * p.ldc + p.rows) * sizeof(T) : 0))
                throw std::invalid_argument("qr application output overlaps factors or tau");
        } else if (detail::memory_overlaps(
                       tau, std::size_t(tc) * sizeof(T), c,
                       p.rows && p.cols ? (std::size_t(p.cols - 1) * p.ldc + p.rows) * sizeof(T)
                                        : 0))
            throw std::invalid_argument("orgqr output overlaps tau");
    });
    if constexpr (Distributed) {
        const int sig[] = {int(B), Generate ? 1 : 0, int(p.side), int(p.trans), m, n,
                           k,      int(sizeof(T))};
        detail::agree_signature(h, "qr application", sig);
        int sa[7], sc[7];
        detail::descriptor_signature(sa, da);
        detail::descriptor_signature(sc, dc);
        detail::agree_signature(h, "qr reflector distribution", sa);
        detail::agree_signature(h, "qr output distribution", sc);
    }
    return p;
}

// Storage is prepared once. Panel execution reuses LARFT and PUNMTR scratch.
template <class T> struct Workspace {
    vector<T> panel, reduced, view, tmat, product, reduced_product, device_tau;
    vector<char> larft;
    std::vector<T> host_tau;
    std::size_t larft_bytes = 0;
    explicit Workspace(const DdlaHandle_t &h)
        : panel(h), reduced(h), view(h), tmat(h), product(h), reduced_product(h), device_tau(h),
          larft(h) {}
};

template <DdlaBackend B, bool Distributed, class T>
void prepare(const DdlaHandle_t &h, const Geometry &p, const T *tau, Workspace<T> &w) {
    checked_setup<Distributed>(h, [&] {
        if (!p.m || !p.n || !p.k)
            return;
        const auto count = std::size_t(p.nq) * p.cap;
        w.panel.resize(count);
        w.tmat.resize(std::size_t(p.cap) * p.cap);
        const auto work = std::size_t(p.cap) * std::max(1, p.side == 'L' ? p.cols : p.rows);
        w.product.resize(work);
        if constexpr (Distributed) {
            w.reduced.resize(count);
            w.view.resize(std::size_t(std::max(1, p.side == 'L' ? p.rows : p.cols)) * p.cap);
            w.reduced_product.resize(work);
            w.device_tau.resize(p.k);
            w.host_tau.assign(p.k, T(0));
        }
        lapack_impl::larft_workspace_size<T>(h, p.nq, p.cap, &w.larft_bytes);
        w.larft.resize(w.larft_bytes);
    });
    if constexpr (Distributed) {
        if (!p.m || !p.n || !p.k)
            return;
        for (int j = 0; j < p.k; ++j)
            if (indxg2p(j, p.nb, p.csrc, h->npcols_) == h->mypcol_)
                w.host_tau[j] = tau[indxg2l(j, p.nb, h->npcols_)];
        if(h->npcols_>1)MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, w.host_tau.data(), p.k, detail::mpi_datatype<T>(),
                                MPI_SUM, h->row_comm));
        DDLA_CHECK(
            ddlaMemcpy(h, w.device_tau.data(), w.host_tau.data(), std::size_t(p.k) * sizeof(T)));
    }
}

// Entries on/above R's diagonal are never copied. The unit entry is explicit.
template <bool Distributed, class T>
void pack(const DdlaHandle_t &h, const Geometry &p, int first, int width, const T *a, T *panel) {
    DDLA_CHECK(ddlaMemset(h, panel, 0, std::size_t(p.nq) * width * sizeof(T)));
    for (int j = 0; j < width; ++j) {
        const int column = first + j;
        if constexpr (Distributed) {
            if (indxg2p(column, p.nb, p.csrc, h->npcols_) != h->mypcol_)
                continue;
            const int lc = indxg2l(column, p.nb, h->npcols_);
            const int end = num_loc(p.nq, p.mb, h->myprow_, p.rsrc, h->nprows_);
            for (int i = num_loc(column + 1, p.mb, h->myprow_, p.rsrc, h->nprows_); i < end;) {
                const int global = indxl2g(i, p.mb, h->myprow_, p.rsrc, h->nprows_);
                const int count = std::min(end - i, p.mb - global % p.mb);
                DDLA_CHECK(ddlaMemcpy(h, panel + global + std::size_t(j) * p.nq,
                                      a + i + std::size_t(lc) * p.lda,
                                      std::size_t(count) * sizeof(T)));
                i += count;
            }
        } else if (column + 1 < p.nq)
            DDLA_CHECK(ddlaMemcpy(h, panel + column + 1 + std::size_t(j) * p.nq,
                                  a + column + 1 + std::size_t(column) * p.lda,
                                  std::size_t(p.nq - column - 1) * sizeof(T)));
    }
}

template <DdlaBackend B, bool Distributed, class T>
void apply_run(const DdlaHandle_t &h, const Geometry &p, const T *a, const T *tau, T *c,
               const int *dc, Workspace<T> &w) {
    if (!p.m || !p.n || !p.k)
        return;
    const T one = T(1);
    const bool backwards = (p.side == 'L') == (p.trans == 'N');
    const int panels = (p.k - 1) / p.cap + 1;
    for (int index = 0; index < panels; ++index) {
        const int first = (backwards ? panels - 1 - index : index) * p.cap;
        const int width = std::min(p.cap, p.k - first);
        pack<Distributed>(h, p, first, width, a, w.panel.data());
        T *full = w.panel.data();
        if constexpr (Distributed) {
            commAllReduce<B>(h, CommScope::Grid, w.panel.data(), w.reduced.data(),
                             std::size_t(p.nq) * width, cclSum);
            full = w.reduced.data();
        }
        for (int j = 0; j < width; ++j)
            DDLA_CHECK(ddlaMemcpy(h, full + first + j + std::size_t(j) * p.nq, &one, sizeof(T)));
        const T *scalars = Distributed ? w.device_tau.data() : tau;
        lapack_impl::larft_run<B, T>(h, 'F', p.nq - first, width, full + first, p.nq,
                                     scalars + first, w.tmat.data(), p.cap, w.larft.data(),
                                     w.larft_bytes);
        T *v = full;
        int ldv = p.nq;
        if constexpr (Distributed) {
            const bool left = p.side == 'L';
            const int block = dc[left ? DDLA_MB_ : DDLA_NB_],
                      source = dc[left ? DDLA_RSRC_ : DDLA_CSRC_];
            const int me = left ? h->myprow_ : h->mypcol_, ranks = left ? h->nprows_ : h->npcols_;
            const int count = left ? p.rows : p.cols;
            ldv = std::max(1, count);
            v = w.view.data();
            for (int i = 0; i < count;) {
                const int global = indxl2g(i, block, me, source, ranks),
                          run = std::min(count - i, block - global % block);
                DDLA_CHECK(ddlaMemcpy2D(h, v + i, std::size_t(ldv) * sizeof(T), full + global,
                                        std::size_t(p.nq) * sizeof(T), std::size_t(run) * sizeof(T),
                                        width));
                i += run;
            }
        }
        const bool reduce = Distributed && (p.side == 'L' ? h->nprows_ : h->npcols_) > 1;
        detail::punmtr_run<B, T>(h, p.side, 'U', p.trans, p.rows, p.cols, width, ldv, p.cap, p.ldc,
                                 p.side == 'L' ? width : std::max(1, p.rows), reduce, v,
                                 w.tmat.data(), c, w.product.data(), w.reduced_product.data());
    }
    DDLA_CHECK(ddlaSynchronize(h));
}

template <bool Distributed, class T>
void identity(const DdlaHandle_t &h, const Geometry &p, T *a, const int *da) {
    const T one = T(1);
    for (int j = 0; j < p.cols; ++j) {
        if (p.rows)
            DDLA_CHECK(
                ddlaMemset(h, a + std::size_t(j) * p.ldc, 0, std::size_t(p.rows) * sizeof(T)));
        const int global =
            Distributed ? indxl2g(j, da[DDLA_NB_], h->mypcol_, da[DDLA_CSRC_], h->npcols_) : j;
        if (global >= p.m)
            continue;
        if constexpr (Distributed) {
            if (indxg2p(global, da[DDLA_MB_], da[DDLA_RSRC_], h->nprows_) != h->myprow_)
                continue;
        }
        const int row = Distributed ? indxg2l(global, da[DDLA_MB_], h->nprows_) : global;
        DDLA_CHECK(ddlaMemcpy(h, a + row + std::size_t(j) * p.ldc, &one, sizeof(T)));
    }
    DDLA_CHECK(ddlaSynchronize(h));
}

template <DdlaBackend B, bool Distributed, class T>
void apply(const DdlaHandle_t &h, char side, char trans, int m, int n, int k, const T *a, int lda,
           const int *da, const T *tau, T *c, int ldc, const int *dc, int &info) {
    const auto p =
        check<B, Distributed, false>(h, side, trans, m, n, k, a, lda, da, tau, c, ldc, dc);
    info = 0;
    Workspace<T> w(h);
    prepare<B, Distributed>(h, p, tau, w);
    apply_run<B, Distributed>(h, p, a, tau, c, dc, w);
}

template <DdlaBackend B, bool Distributed, class T>
void generate(const DdlaHandle_t &h, int m, int n, int k, T *a, int lda, const int *da,
              const T *tau, int &info) {
    const auto p = check<B, Distributed, true>(h, 'L', 'N', m, n, k, a, lda, da, tau, a, lda, da);
    info = 0;
    if (!m || !n)
        return;
    Workspace<T> w(h);
    vector<T> saved(h);
    Geometry stored = p;
    checked_setup<Distributed>(h, [&] {
        const int columns = Distributed ? num_loc(k, p.nb, h->mypcol_, p.csrc, h->npcols_) : k;
        stored.lda = std::max(1, p.rows);
        saved.resize(std::size_t(stored.lda) * columns);
    });
    prepare<B, Distributed>(h, p, tau, w);
    // Save only tails before A becomes I. R and parent padding stay unread.
    for (int j = 0; j < k; ++j) {
        if constexpr (Distributed)
            if (indxg2p(j, p.nb, p.csrc, h->npcols_) != h->mypcol_)
                continue;
        const int column = Distributed ? indxg2l(j, p.nb, h->npcols_) : j;
        const int first =
            Distributed ? num_loc(j + 1, p.mb, h->myprow_, p.rsrc, h->nprows_) : j + 1;
        if (p.rows > first)
            DDLA_CHECK(ddlaMemcpy(h, saved.data() + first + std::size_t(column) * stored.lda,
                                  a + first + std::size_t(column) * p.lda,
                                  std::size_t(p.rows - first) * sizeof(T)));
    }
    identity<Distributed>(h, p, a, da);
    apply_run<B, Distributed>(h, stored, saved.data(), tau, a, da, w);
}
} // namespace qr_apply_impl

template <DdlaBackend B, class T>
void ormqr(const DdlaHandle_t &h, char side, char trans, int m, int n, int k, const T *a, int lda,
           const T *tau, T *c, int ldc, int &info) {
    qr_apply_impl::apply<B, false>(h, side, trans, m, n, k, a, lda, nullptr, tau, c, ldc, nullptr,
                                   info);
}
template <DdlaBackend B, class T>
void orgqr(const DdlaHandle_t &h, int m, int n, int k, T *a, int lda, const T *tau, int &info) {
    qr_apply_impl::generate<B, false>(h, m, n, k, a, lda, nullptr, tau, info);
}
template <DdlaBackend B, class T>
void pormqr(const DdlaHandle_t &h, char side, char trans, int m, int n, int k, const T *a,
            const int *da, const T *tau, T *c, const int *dc, int &info) {
    qr_apply_impl::apply<B, true>(h, side, trans, m, n, k, a, 1, da, tau, c, 1, dc, info);
}
template <DdlaBackend B, class T>
void porgqr(const DdlaHandle_t &h, int m, int n, int k, T *a, const int *da, const T *tau,
            int &info) {
    qr_apply_impl::generate<B, true>(h, m, n, k, a, 1, da, tau, info);
}
#define DDLA_QR_APPLY(B, T)                                                                        \
    template void ormqr<B, T>(const DdlaHandle_t &, char, char, int, int, int, const T *, int,     \
                              const T *, T *, int, int &);                                         \
    template void orgqr<B, T>(const DdlaHandle_t &, int, int, int, T *, int, const T *, int &);    \
    template void pormqr<B, T>(const DdlaHandle_t &, char, char, int, int, int, const T *,         \
                               const int *, const T *, T *, const int *, int &);                   \
    template void porgqr<B, T>(const DdlaHandle_t &, int, int, int, T *, const int *, const T *,   \
                               int &)
DDLA_FOR_EACH_BACKEND_TYPE(DDLA_QR_APPLY)
#undef DDLA_QR_APPLY
} // namespace ddla
