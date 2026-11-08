#ifndef DDLA_LDLT_IMPL_H
#define DDLA_LDLT_IMPL_H

#include <ddla/ddla.h>
#include <type_traits>
#include <ddla/ddla_vector.h>
#include "ddla_connector.h"
#include "ddla_desc.h"
#include "check_agree.h"
#include "require_backend.h"
#include "BLAS/blas_impl.h"
#include "helpers.h"
#include <algorithm>
#include <cctype>
#include <complex>
#include <stdexcept>
#include <vector>

extern "C" {
void ssytrf_(const char *, const int *, float *, const int *, int *, float *, const int *, int *);
void ssytrs_(const char *, const int *, const int *, const float *, const int *, const int *,
             float *, const int *, int *);
void dsytrf_(const char *, const int *, double *, const int *, int *, double *, const int *, int *);
void dsytrs_(const char *, const int *, const int *, const double *, const int *, const int *,
             double *, const int *, int *);
void csytrf_(const char *, const int *, std::complex<float> *, const int *, int *,
             std::complex<float> *, const int *, int *);
void csytrs_(const char *, const int *, const int *, const std::complex<float> *, const int *,
             const int *, std::complex<float> *, const int *, int *);
void chetrf_(const char *, const int *, std::complex<float> *, const int *, int *,
             std::complex<float> *, const int *, int *);
void chetrs_(const char *, const int *, const int *, const std::complex<float> *, const int *,
             const int *, std::complex<float> *, const int *, int *);
void zsytrf_(const char *, const int *, std::complex<double> *, const int *, int *,
             std::complex<double> *, const int *, int *);
void zsytrs_(const char *, const int *, const int *, const std::complex<double> *, const int *,
             const int *, std::complex<double> *, const int *, int *);
void zhetrf_(const char *, const int *, std::complex<double> *, const int *, int *,
             std::complex<double> *, const int *, int *);
void zhetrs_(const char *, const int *, const int *, const std::complex<double> *, const int *,
             const int *, std::complex<double> *, const int *, int *);
}
namespace ddla {
namespace ldlt_impl {
inline void factor(std::integral_constant<bool, false>, char u, int n, float *a, int ld, int *piv,
                   int &info) {
    float query;
    int lwork = -1;
    ssytrf_(&u, &n, a, &ld, piv, &query, &lwork, &info);
    if (info)
        return;
    lwork = static_cast<int>(std::real(query));
    std::vector<float> work(lwork);
    ssytrf_(&u, &n, a, &ld, piv, work.data(), &lwork, &info);
}
inline void solve(std::integral_constant<bool, false>, char u, int n, int nrhs, const float *a,
                  int ld, const int *piv, float *b, int ldb, int &info) {
    ssytrs_(&u, &n, &nrhs, a, &ld, piv, b, &ldb, &info);
}
inline void factor(std::integral_constant<bool, true>, char u, int n, float *a, int ld, int *piv,
                   int &info) {
    float query;
    int lwork = -1;
    ssytrf_(&u, &n, a, &ld, piv, &query, &lwork, &info);
    if (info)
        return;
    lwork = static_cast<int>(std::real(query));
    std::vector<float> work(lwork);
    ssytrf_(&u, &n, a, &ld, piv, work.data(), &lwork, &info);
}
inline void solve(std::integral_constant<bool, true>, char u, int n, int nrhs, const float *a,
                  int ld, const int *piv, float *b, int ldb, int &info) {
    ssytrs_(&u, &n, &nrhs, a, &ld, piv, b, &ldb, &info);
}
inline void factor(std::integral_constant<bool, false>, char u, int n, double *a, int ld, int *piv,
                   int &info) {
    double query;
    int lwork = -1;
    dsytrf_(&u, &n, a, &ld, piv, &query, &lwork, &info);
    if (info)
        return;
    lwork = static_cast<int>(std::real(query));
    std::vector<double> work(lwork);
    dsytrf_(&u, &n, a, &ld, piv, work.data(), &lwork, &info);
}
inline void solve(std::integral_constant<bool, false>, char u, int n, int nrhs, const double *a,
                  int ld, const int *piv, double *b, int ldb, int &info) {
    dsytrs_(&u, &n, &nrhs, a, &ld, piv, b, &ldb, &info);
}
inline void factor(std::integral_constant<bool, true>, char u, int n, double *a, int ld, int *piv,
                   int &info) {
    double query;
    int lwork = -1;
    dsytrf_(&u, &n, a, &ld, piv, &query, &lwork, &info);
    if (info)
        return;
    lwork = static_cast<int>(std::real(query));
    std::vector<double> work(lwork);
    dsytrf_(&u, &n, a, &ld, piv, work.data(), &lwork, &info);
}
inline void solve(std::integral_constant<bool, true>, char u, int n, int nrhs, const double *a,
                  int ld, const int *piv, double *b, int ldb, int &info) {
    dsytrs_(&u, &n, &nrhs, a, &ld, piv, b, &ldb, &info);
}
inline void factor(std::integral_constant<bool, false>, char u, int n, std::complex<float> *a,
                   int ld, int *piv, int &info) {
    std::complex<float> query;
    int lwork = -1;
    csytrf_(&u, &n, a, &ld, piv, &query, &lwork, &info);
    if (info)
        return;
    lwork = static_cast<int>(std::real(query));
    std::vector<std::complex<float>> work(lwork);
    csytrf_(&u, &n, a, &ld, piv, work.data(), &lwork, &info);
}
inline void solve(std::integral_constant<bool, false>, char u, int n, int nrhs,
                  const std::complex<float> *a, int ld, const int *piv, std::complex<float> *b,
                  int ldb, int &info) {
    csytrs_(&u, &n, &nrhs, a, &ld, piv, b, &ldb, &info);
}
inline void factor(std::integral_constant<bool, true>, char u, int n, std::complex<float> *a,
                   int ld, int *piv, int &info) {
    std::complex<float> query;
    int lwork = -1;
    chetrf_(&u, &n, a, &ld, piv, &query, &lwork, &info);
    if (info)
        return;
    lwork = static_cast<int>(std::real(query));
    std::vector<std::complex<float>> work(lwork);
    chetrf_(&u, &n, a, &ld, piv, work.data(), &lwork, &info);
}
inline void solve(std::integral_constant<bool, true>, char u, int n, int nrhs,
                  const std::complex<float> *a, int ld, const int *piv, std::complex<float> *b,
                  int ldb, int &info) {
    chetrs_(&u, &n, &nrhs, a, &ld, piv, b, &ldb, &info);
}
inline void factor(std::integral_constant<bool, false>, char u, int n, std::complex<double> *a,
                   int ld, int *piv, int &info) {
    std::complex<double> query;
    int lwork = -1;
    zsytrf_(&u, &n, a, &ld, piv, &query, &lwork, &info);
    if (info)
        return;
    lwork = static_cast<int>(std::real(query));
    std::vector<std::complex<double>> work(lwork);
    zsytrf_(&u, &n, a, &ld, piv, work.data(), &lwork, &info);
}
inline void solve(std::integral_constant<bool, false>, char u, int n, int nrhs,
                  const std::complex<double> *a, int ld, const int *piv, std::complex<double> *b,
                  int ldb, int &info) {
    zsytrs_(&u, &n, &nrhs, a, &ld, piv, b, &ldb, &info);
}
inline void factor(std::integral_constant<bool, true>, char u, int n, std::complex<double> *a,
                   int ld, int *piv, int &info) {
    std::complex<double> query;
    int lwork = -1;
    zhetrf_(&u, &n, a, &ld, piv, &query, &lwork, &info);
    if (info)
        return;
    lwork = static_cast<int>(std::real(query));
    std::vector<std::complex<double>> work(lwork);
    zhetrf_(&u, &n, a, &ld, piv, work.data(), &lwork, &info);
}
inline void solve(std::integral_constant<bool, true>, char u, int n, int nrhs,
                  const std::complex<double> *a, int ld, const int *piv, std::complex<double> *b,
                  int ldb, int &info) {
    zhetrs_(&u, &n, &nrhs, a, &ld, piv, b, &ldb, &info);
}

template <bool Hermitian, typename T>
void factor_nopiv(const DdlaHandle_t &h, char u, int n, T *a, int ld, int &info) {
    using R = typename real_type<T>::type;
    for (int step = 0; step < n; ++step) {
        const int j = u == 'L' ? step : n - 1 - step;
        T &d = a[j + std::size_t(j) * ld];
        if constexpr (Hermitian)
            d = T(std::real(d));
        if (d == T(0)) {
            info = j + 1;
            return;
        }
        const int first = u == 'L' ? j + 1 : 0, count = u == 'L' ? n - j - 1 : j;
        if (!count)
            continue;
        T *column = a + first + std::size_t(j) * ld;
        for (int i = 0; i < count; ++i)
            column[i] /= d;
        T *trailing = a + first + std::size_t(first) * ld;
        if constexpr (Hermitian)
            blas_impl::herk_run<DdlaBackend::CPU, T>(h, u, 'N', count, 1, -R(std::real(d)), column,
                                                     ld, R(1), trailing, ld);
        else
            blas_impl::syrk_run<DdlaBackend::CPU, T>(h, u, 'N', count, 1, -d, column, ld, T(1),
                                                     trailing, ld);
    }
}
template <bool Hermitian, typename T>
void solve_nopiv(const DdlaHandle_t &h, char u, int n, int nrhs, const T *a, int ld, T *b, int ldb,
                 int &info) {
    for (int i = 0; i < n; ++i)
        if (a[i + std::size_t(i) * ld] == T(0)) {
            info = i + 1;
            return;
        }
    blas_impl::trsm_run<DdlaBackend::CPU, T>(h, 'L', u, 'N', 'U', n, nrhs, T(1), const_cast<T *>(a),
                                             ld, b, ldb);
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < n; ++i)
            b[i + std::size_t(j) * ldb] /= a[i + std::size_t(i) * ld];
    blas_impl::trsm_run<DdlaBackend::CPU, T>(h, 'L', u, Hermitian ? 'C' : 'T', 'U', n, nrhs, T(1),
                                             const_cast<T *>(a), ld, b, ldb);
}
inline char triangle(char u) {
    u = static_cast<char>(std::toupper(static_cast<unsigned char>(u)));
    if (u != 'U' && u != 'L')
        throw std::invalid_argument("ldlt invalid triangle");
    return u;
}
// Copy only stored entries, including on GPU handles.
template <typename T>
void copy_triangle(const DdlaHandle_t &h, char u, int n, const T *src, int lds, T *dst, int ldd) {
    for (int j = 0; j < n; ++j) {
        const int first = u == 'L' ? j : 0, count = u == 'L' ? n - j : j + 1;
        DDLA_CHECK(ddlaMemcpy(h, dst + first + std::size_t(j) * ldd,
                              src + first + std::size_t(j) * lds, std::size_t(count) * sizeof(T)));
    }
    DDLA_CHECK(ddlaSynchronize(h));
}
inline void check_pivots(char u, int n, const int *piv) {
    for (int step = 0; step < n; ++step) {
        const int i = u == 'L' ? step : n - 1 - step;
        const long long value = piv[i];
        if (value == 0 || value > n || value < -static_cast<long long>(n))
            throw std::invalid_argument("ldlt invalid pivot index");
        if (value < 0) {
            const int other = u == 'L' ? i + 1 : i - 1;
            if (other < 0 || other >= n || piv[other] != value)
                throw std::invalid_argument("ldlt invalid 2x2 pivot pair");
            ++step;
        }
    }
}
template <DdlaBackend B, bool Pivot, bool Solve, typename T>
char local_check(const DdlaHandle_t &h, char u, int n, int nrhs, const T *a, int ld,
                 const int *piv, T *b, int ldb, int &info) {
    detail::require_backend<B>(h, "ldlt");
    u = triangle(u);
    info = 0;
    if (n < 0 || nrhs < 0 || ld < std::max(1, n) || (Solve && ldb < std::max(1, n)))
        throw std::invalid_argument("ldlt invalid extent/stride");
    if (!n || (Solve && !nrhs))
        return u;
    if (!a || (Pivot && !piv) || (Solve && !b))
        throw std::invalid_argument("ldlt null active data");
    if constexpr (Solve)
        if (detail::matrix_overlaps(a, ld, n, n, b, ldb, n, nrhs))
            throw std::invalid_argument("ldlt factors overlap RHS");
    return u;
}
template <DdlaBackend B, bool H, bool Pivot, bool Solve, typename T>
void local(const DdlaHandle_t &h, char u, int n, int nrhs, T *a, int ld, int *piv, T *b, int ldb,
           int &info) {
    u = local_check<B, Pivot, Solve, T>(h, u, n, nrhs, a, ld, piv, b, ldb, info);
    if (!n || (Solve && !nrhs))
        return;
    std::vector<T> host_a(std::size_t(n) * n), host_b(Solve ? std::size_t(n) * nrhs : 0);
    copy_triangle(h, u, n, a, ld, host_a.data(), n);
    if constexpr (Solve) {
        DDLA_CHECK(ddlaMemcpy2D(h, host_b.data(), std::size_t(n) * sizeof(T), b,
                                std::size_t(ldb) * sizeof(T), std::size_t(n) * sizeof(T), nrhs));
        DDLA_CHECK(ddlaSynchronize(h));
        if constexpr (Pivot) {
            check_pivots(u, n, piv);
            solve(std::bool_constant<H>{}, u, n, nrhs, host_a.data(), n, piv, host_b.data(), n,
                  info);
        } else
            solve_nopiv<H>(h, u, n, nrhs, host_a.data(), n, host_b.data(), n, info);
        if (!info) {
            DDLA_CHECK(ddlaMemcpy2D(h, b, std::size_t(ldb) * sizeof(T), host_b.data(),
                                    std::size_t(n) * sizeof(T), std::size_t(n) * sizeof(T), nrhs));
            DDLA_CHECK(ddlaSynchronize(h));
        }
    } else {
        if constexpr (Pivot)
            factor(std::bool_constant<H>{}, u, n, host_a.data(), n, piv, info);
        else
            factor_nopiv<H>(h, u, n, host_a.data(), n, info);
        copy_triangle(h, u, n, host_a.data(), n, a, ld);
    }
}
template <DdlaBackend B, bool H, bool Pivot, bool Solve, typename T>
char distributed_check(const DdlaHandle_t &h, char u, int n, int nrhs, T *a, const int *da,
                       int *piv, T *b, const int *db, int &info) {
    detail::agree_check(h, "pldlt", [&](detail::Agree &agree) {
        detail::require_backend<B>(h, "pldlt");
        u = triangle(u);
        agree.mode("operation", (H ? 1 : 0) + (Pivot ? 2 : 0) + (Solve ? 4 : 0));
        check_desc(da, h);
        if constexpr (Solve)
            check_desc(db, h);
        if (n < 0 || nrhs < 0 || n > da[DDLA_M_] || n > da[DDLA_N_] ||
            (Solve && (n > db[DDLA_M_] || nrhs > db[DDLA_N_])))
            throw std::invalid_argument("pldlt invalid extents");
        if (n && (!Solve || nrhs)) {
            if (Pivot && !piv)
                throw std::invalid_argument("pldlt missing host pivots");
            const int ar = num_loc(n, da[DDLA_MB_], h->myprow_, da[DDLA_RSRC_], h->nprows_);
            const int ac = num_loc(n, da[DDLA_NB_], h->mypcol_, da[DDLA_CSRC_], h->npcols_);
            if (ar && ac && !a)
                throw std::invalid_argument("pldlt missing factors");
            if constexpr (Solve) {
                const int br = num_loc(n, db[DDLA_MB_], h->myprow_, db[DDLA_RSRC_], h->nprows_);
                const int bc = num_loc(nrhs, db[DDLA_NB_], h->mypcol_, db[DDLA_CSRC_], h->npcols_);
                if (br && bc && !b)
                    throw std::invalid_argument("pldlt missing RHS");
                if (detail::matrix_overlaps(a, da[DDLA_LLD_], ar, ac, b, db[DDLA_LLD_], br, bc))
                    throw std::invalid_argument("pldlt factors overlap RHS");
            }
        }
    });
    // Two signatures avoid including rank-local leading dimensions.
    int common[7] = {int(u), n, nrhs, int(B), H, Pivot, Solve};
    detail::agree_signature(h, "pldlt", common);
    int sa[7];
    detail::descriptor_signature(sa, da);
    detail::agree_signature(h, "pldlt A", sa);
    if constexpr (Solve) {
        int sb[7];
        detail::descriptor_signature(sb, db);
        detail::agree_signature(h, "pldlt B", sb);
    }
    info = 0;
    return u;
}
// Root-host baseline. All failure-prone setup is agreed before transfers.
template <DdlaBackend B, bool H, bool Pivot, bool Solve, typename T>
void distributed(const DdlaHandle_t &h, char u, int n, int nrhs, T *a, const int *da, int *piv,
                 T *b, const int *db, int &info) {
    u = distributed_check<B, H, Pivot, Solve, T>(h, u, n, nrhs, a, da, piv, b, db, info);
    if (!n || (Solve && !nrhs)) return;
    vector<T> window(h), rhs(h);
    std::vector<T> ha, hb;
    std::vector<int> root_piv;
    detail::agree_check(h, "pldlt storage", [&](detail::Agree &) {
        if (h->myid == 0) {
            window.resize(std::size_t(n) * n);
            ha.resize(std::size_t(n) * n);
            if constexpr (Solve) {
                rhs.resize(std::size_t(n) * nrhs);
                hb.resize(std::size_t(n) * nrhs);
            }
            if constexpr (Pivot) {
                root_piv.resize(n);
                if constexpr (Solve)
                    std::copy_n(piv, n, root_piv.data());
            }
        }
    });
    for (int j = 0; j < n; ++j) {
        const int first = u == 'L' ? j : 0, count = u == 'L' ? n - j : j + 1;
        gather<B, T>(h, count, 1, a, first + 1, j + 1, da, 0,
                     h->myid == 0 ? window.data() + first + std::size_t(j) * n : nullptr, n);
    }
    detail::agree_check(h, "pldlt factor copy", [&](detail::Agree &) {
        if (h->myid == 0)
            copy_triangle(h, u, n, window.data(), n, ha.data(), n);
    });
    if constexpr (Solve) {
        gather<B, T>(h, n, nrhs, b, 1, 1, db, 0, rhs.data(), n);
        detail::agree_check(h, "pldlt RHS copy", [&](detail::Agree &) {
            if (h->myid == 0) {
                DDLA_CHECK(ddlaMemcpy(h, hb.data(), rhs.data(), std::size_t(n) * nrhs * sizeof(T)));
                DDLA_CHECK(ddlaSynchronize(h));
            }
        });
    }
    detail::agree_check(h, "pldlt arithmetic", [&](detail::Agree &) {
        if (h->myid == 0) {
            if constexpr (Solve) {
                if constexpr (Pivot) {
                    check_pivots(u, n, root_piv.data());
                    solve(std::bool_constant<H>{}, u, n, nrhs, ha.data(), n, root_piv.data(),
                          hb.data(), n, info);
                } else
                    solve_nopiv<H>(h, u, n, nrhs, ha.data(), n, hb.data(), n, info);
            } else {
                if constexpr (Pivot)
                    factor(std::bool_constant<H>{}, u, n, ha.data(), n, root_piv.data(), info);
                else
                    factor_nopiv<H>(h, u, n, ha.data(), n, info);
            }
        }
    });
    MPI_CHECK(MPI_Bcast(&info, 1, MPI_INT, 0, h->comm));
    if constexpr (Pivot && !Solve) {
        if (h->myid == 0)
            std::copy_n(root_piv.data(), n, piv);
        MPI_CHECK(MPI_Bcast(piv, n, MPI_INT, 0, h->comm));
    }
    if constexpr (Solve) {
        if (info)
            return;
        detail::agree_check(h, "pldlt RHS upload", [&](detail::Agree &) {
            if (h->myid == 0) {
                DDLA_CHECK(ddlaMemcpy(h, rhs.data(), hb.data(), std::size_t(n) * nrhs * sizeof(T)));
                DDLA_CHECK(ddlaSynchronize(h));
            }
        });
        scatter<B, T>(h, n, nrhs, b, 1, 1, db, 0, rhs.data(), n);
    } else {
        detail::agree_check(h, "pldlt factor upload", [&](detail::Agree &) {
            if (h->myid == 0)
                copy_triangle(h, u, n, ha.data(), n, window.data(), n);
        });
        for (int j = 0; j < n; ++j) {
            const int first = u == 'L' ? j : 0, count = u == 'L' ? n - j : j + 1;
            scatter<B, T>(h, count, 1, a, first + 1, j + 1, da, 0,
                          h->myid == 0 ? window.data() + first + std::size_t(j) * n : nullptr, n);
        }
    }
}
} // namespace ldlt_impl
} // namespace ddla

#endif // DDLA_LDLT_IMPL_H
