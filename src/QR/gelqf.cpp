#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "BLAS/blas_impl.h"
#include "ddla_stream.h"
#include "instantiate.h"
#include "lapack_impl.h"
#include "require_backend.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>

#define DECLARE_ROW_LAPACK(P, T) \
    void P##gelqf_(const int*, const int*, T*, const int*, T*, T*, const int*, int*); \
    void P##gerqf_(const int*, const int*, T*, const int*, T*, T*, const int*, int*)
extern "C" {
DECLARE_ROW_LAPACK(s, float);
DECLARE_ROW_LAPACK(d, double);
DECLARE_ROW_LAPACK(c, std::complex<float>);
DECLARE_ROW_LAPACK(z, std::complex<double>);
}
#undef DECLARE_ROW_LAPACK

namespace ddla {
namespace {

#define ROW_LAPACK(P, T) \
    void row_lapack(bool rq, int m, int n, T* a, int lda, T* tau, T* work, int size, int& info) \
    { \
        if (rq) P##gerqf_(&m, &n, a, &lda, tau, work, &size, &info); \
        else P##gelqf_(&m, &n, a, &lda, tau, work, &size, &info); \
    }
ROW_LAPACK(s, float)
ROW_LAPACK(d, double)
ROW_LAPACK(c, std::complex<float>)
ROW_LAPACK(z, std::complex<double>)
#undef ROW_LAPACK

// LQ/RQ on CPU use their native LAPACK routines. On GPU, factor A^H with
// QR/QL and adjoint the packed result back: row reflectors and tau then have
// LAPACK's LQ/RQ storage, including conjugated vector tails for complex T.
template<DdlaBackend Backend, class T>
void factorize(const DdlaHandle_t& h, bool rq, int m, int n, T* a, int lda, T* tau)
{
    const char* name = rq ? "gerqf" : "gelqf";
    detail::require_backend<Backend>(h, name);
    if (m < 0 || n < 0) throw std::invalid_argument(std::string(name) + ": negative matrix size");
    if (std::min(m, n) == 0) return;
    if (!a || !tau || lda < std::max(1, m))
        throw std::invalid_argument(std::string(name) + ": invalid A, tau or leading dimension");

    if constexpr (Backend == DdlaBackend::CPU) {
        T optimal{};
        int info = 0;
        row_lapack(rq, m, n, a, lda, tau, &optimal, -1, info);
        const auto size = std::real(optimal);
        if (info != 0 || !std::isfinite(size) || size < 1 ||
            static_cast<long double>(size) > std::numeric_limits<int>::max())
            throw std::runtime_error(std::string(name) + ": LAPACK workspace query failed");
        const int lwork = static_cast<int>(size);
        vector<T> work(h, lwork);
        row_lapack(rq, m, n, a, lda, tau, work.data(), lwork, info);
        if (info != 0) throw std::runtime_error(std::string(name) + ": LAPACK factorization failed");
    } else {
        vector<T> adjoint(h, std::size_t(m) * n);
        std::size_t bytes = 0;
        const char uplo = rq ? 'L' : 'U';
        lapack_impl::geqrf_workspace_size<T>(h, uplo, n, m, adjoint.data(), n, tau, &bytes);
        vector<char> work(h, bytes);
        try {
            blas_impl::geam_run<Backend, T>(h, 'C', 'N', n, m, T(1), a, lda,
                T(0), static_cast<const T*>(nullptr), n, adjoint.data(), n, h->stream);
            lapack_impl::geqrf_run<Backend, T>(h, uplo, n, m, adjoint.data(), n, tau,
                work.data(), bytes);
            blas_impl::geam_run<Backend, T>(h, 'C', 'N', m, n, T(1), adjoint.data(), n,
                T(0), static_cast<const T*>(nullptr), m, a, lda, h->stream);
        } catch (...) {
            try { (void)ddlaSynchronize(h); } catch (...) {}
            throw;
        }
    }
}

} // namespace

template<DdlaBackend Backend, class T>
void gelqf(const DdlaHandle_t& h, int m, int n, T* a, int lda, T* tau)
{
    factorize<Backend, T>(h, false, m, n, a, lda, tau);
}

template<DdlaBackend Backend, class T>
void gerqf(const DdlaHandle_t& h, int m, int n, T* a, int lda, T* tau)
{
    factorize<Backend, T>(h, true, m, n, a, lda, tau);
}

#define INSTANTIATE_ROW_FACTORS(B, T) \
    template void gelqf<B, T>(const DdlaHandle_t&, int, int, T*, int, T*); \
    template void gerqf<B, T>(const DdlaHandle_t&, int, int, T*, int, T*)
DDLA_FOR_EACH_BACKEND_TYPE(INSTANTIATE_ROW_FACTORS)
#undef INSTANTIATE_ROW_FACTORS
} // namespace ddla
