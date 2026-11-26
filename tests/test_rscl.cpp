#include "api_grid_test_common.h"
#include "ddla_stream.h"
#include <ddla/ddla_vector.h>
#include <cstring>

using namespace ddla;
using namespace api_grid_test;

namespace {
template <class T>
T value(typename real_type<T>::type real, typename real_type<T>::type imaginary = 0)
{
    if constexpr (std::is_arithmetic_v<T>) return real;
    else return {real, imaginary};
}

template <class R>
void check_component(R actual, long double expected)
{
    const long double error = std::abs(static_cast<long double>(actual) - expected);
    const long double bound = 32 * std::numeric_limits<R>::epsilon() * std::abs(expected) +
                              2 * static_cast<long double>(std::numeric_limits<R>::denorm_min());
    if (!std::isfinite(actual) || error > bound)
        throw std::runtime_error("RSCL quotient is nonfinite or inaccurate");
}

template <class T, class A>
void vector_case(DdlaHandle_t h, A divisor, T input, int incx, int n = 7)
{
    using R = typename real_type<T>::type;
    const T guard = value<T>(R(-73), R(19));
    std::vector<T> host(3 + n * incx, guard);
    for (int i = 0; i < n; ++i) host[1 + i * incx] = input;
    ddla::vector<T> x(h, host.size());
    DDLA_CHECK(ddlaMemcpy(h, x.data(), host.data(), host.size() * sizeof(T)));
    rscl<T, A>(h, n, divisor, x.data() + 1, incx);
    std::vector<T> actual(host.size());
    DDLA_CHECK(ddlaMemcpy(h, actual.data(), x.data(), actual.size() * sizeof(T)));
    DDLA_CHECK(ddlaSynchronize(h));
    // Extended precision is independent of the implementation's reciprocal
    // scaling and cannot overflow at the binary32/binary64 range boundary.
    const std::complex<long double> numerator(std::real(input), std::imag(input));
    const std::complex<long double> denominator(std::real(divisor), std::imag(divisor));
    const auto expected = numerator / denominator;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (i >= 1 && (i - 1) % incx == 0 && (i - 1) / incx < n) {
            check_component<R>(std::real(actual[i]), expected.real());
            check_component<R>(std::imag(actual[i]), expected.imag());
        } else if (std::memcmp(&actual[i], &guard, sizeof(T))) {
            throw std::runtime_error("RSCL modified a guard or stride gap");
        }
    }
}

template <class T>
void local_cases(DdlaHandle_t h)
{
    using R = typename real_type<T>::type;
    const R tiny = std::numeric_limits<R>::min() / R(8);
    const R maximum = std::numeric_limits<R>::max();
    vector_case<T, R>(h, tiny, value<T>(tiny, -tiny), 3, 131075);
    for (int stride : {1, 3}) {
        for (R divisor : {R(2), R(-2), tiny, -tiny, maximum, -maximum})
            vector_case<T, R>(h, divisor, value<T>(divisor, divisor / R(2)), stride);
        vector_case<T, R>(h, R(.5), value<T>(maximum / R(2)), stride);
        vector_case<T, R>(h, std::numeric_limits<R>::denorm_min(),
                         value<T>(std::numeric_limits<R>::denorm_min()), stride);
        if constexpr (std::is_same_v<R, double>) {
            vector_case<T, R>(h, 0x0.07e65dccc8f15p-1022,
                             value<T>(0x1.f99773323c53fp-4), stride);
            vector_case<T, R>(h, 0x1.eb16f53e79189p-8,
                             value<T>(0x1.eb16f53e79188p+1016), stride);
        }
        if constexpr (!std::is_arithmetic_v<T>) {
            vector_case<T, T>(h, T(R(2), R(-3)), T(R(4), R(-1)), stride);
            vector_case<T, T>(h, T(tiny, tiny), T(tiny, -tiny), stride);
            vector_case<T, T>(h, T(maximum, maximum), T(maximum, -maximum), stride);
            vector_case<T, T>(h, T(R(0), tiny), T(tiny, -tiny), stride);
            vector_case<T, T>(h, T(R(2), R(-.1)) / R(4.01),
                             T(R(.51) * maximum, R(.3) * maximum), stride);
            const R lower = std::ldexp(R(1), std::numeric_limits<R>::min_exponent / 4 + 2);
            const R upper = std::ldexp(R(1), std::numeric_limits<R>::max_exponent / 4 - 2);
            vector_case<T, T>(h, T(lower, upper), T(upper, lower), stride);
            vector_case<T, T>(h, T(lower, -lower), T(upper, upper), stride);
        }
    }
    rscl<T, R>(h, 0, R(0), nullptr, 0);
    if (h->backend == DdlaBackend::CPU) {
        T aliased[] = {T(tiny), T(tiny * R(2)), T(-tiny * R(2))};
        rscl<T, T>(h, 3, aliased[0], aliased, 1);
        check_component<R>(std::real(aliased[0]), 1);
        check_component<R>(std::real(aliased[1]), 2);
        check_component<R>(std::real(aliased[2]), -2);
    }
    ddla::vector<T> x(h, 1);
    for (int which = 0; which < 6; ++which) {
        bool rejected = false;
        try {
            rscl<T, R>(h, which == 0 ? -1 : 1,
                       which == 1 ? R(0) : which == 2 ? std::numeric_limits<R>::infinity() :
                       which == 3 ? std::numeric_limits<R>::quiet_NaN() : R(2),
                       which == 4 ? nullptr : x.data(), which == 5 ? 0 : 1);
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("RSCL accepted invalid active arguments");
    }
}

template <class T>
void scalar_factors(DdlaHandle_t h)
{
    using R = typename real_type<T>::type;
    constexpr int n = 3, nrhs = 2;
    int da[DDLA_DLEN_], db[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(da, h, n, n, 1, 1, 0, 0));
    DDLA_CHECK(ddlaDescInit(db, h, n, nrhs, 1, 1, 0, 0));
    const R tiny = std::numeric_limits<R>::min() / R(8);
    const R d0 = std::is_same_v<R, double> ? R(0x0.07e65dccc8f15p-1022) : tiny;
    const R y0 = std::is_same_v<R, double> ? R(0x1.f99773323c53fp-4) : tiny;
    for (char uplo : {'L', 'U'}) {
        const auto factor = make_local<T>(h, da, [&](int i, int j) {
            return i != j ? T(0) : T(i == 0 ? d0 : i == 1 ? -tiny : R(2));
        });
        const auto input = make_local<T>(h, db, [&](int i, int j) {
            return value<T>(i == 0 ? y0 : i == 1 ? -tiny * R(j + 1) : R(4), R(0));
        });
        DeviceBuffer<T> a(h, factor.size()), b(h, input.size());
        upload(h, a.ptr, factor);
        const int pivots[] = {1, 2, 3};
        for (bool symmetric : {false, true})
            for (bool pivoted : {false, true}) {
                upload(h, b.ptr, input);
                int info = -1;
                if (symmetric && pivoted)
                    psytrs<T>(h, uplo, n, nrhs, a.ptr, da, pivots, b.ptr, db, info);
                else if (symmetric)
                    psytrs_nopiv<T>(h, uplo, n, nrhs, a.ptr, da, b.ptr, db, info);
                else if (pivoted)
                    phetrs<T>(h, uplo, n, nrhs, a.ptr, da, pivots, b.ptr, db, info);
                else
                    phetrs_nopiv<T>(h, uplo, n, nrhs, a.ptr, da, b.ptr, db, info);
                if (info) throw std::runtime_error("Scalar LDLT solve returned nonzero INFO");
                const auto actual = download(h, b.ptr, input.size());
                for (int j = 0; j < nrhs; ++j)
                    for (int i = 0; i < n; ++i) {
                        const int row = indx_g2l_r(db, h, i), column = indx_g2l_c(db, h, j);
                        if (row < 0 || column < 0) continue;
                        const auto entry = actual[row + std::size_t(column) * db[DDLA_LLD_]];
                        const long double expected = i == 0 ? static_cast<long double>(y0) / d0 :
                                                     i == 1 ? j + 1 : 2;
                        check_component<R>(std::real(entry), expected);
                        check_component<R>(std::imag(entry), 0);
                    }
                const auto preserved = download(h, a.ptr, factor.size());
                if (std::memcmp(preserved.data(), factor.data(), factor.size() * sizeof(T)))
                    throw std::runtime_error("Scalar LDLT solve modified its input factors");
            }
    }
}

template <class R>
void complex_symmetric_factor(DdlaHandle_t h)
{
    using T = std::complex<R>;
    const R maximum = std::numeric_limits<R>::max();
    const T divisor = T(R(2), R(-.1)) / R(4.01);
    const T input(R(.51) * maximum, R(.3) * maximum);
    const auto expected = std::complex<long double>(input.real(), input.imag()) /
                          std::complex<long double>(divisor.real(), divisor.imag());
    int desc[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(desc, h, 1, 1, 1, 1, 0, 0));
    const auto factor = make_local<T>(h, desc, [&](int, int) { return divisor; });
    const auto rhs = make_local<T>(h, desc, [&](int, int) { return input; });
    DeviceBuffer<T> a(h, factor.size()), b(h, rhs.size());
    upload(h, a.ptr, factor);
    const int pivot = 1;
    for (char uplo : {'L', 'U'})
        for (bool pivoted : {false, true}) {
            upload(h, b.ptr, rhs);
            int info = -1;
            if (pivoted) psytrs<T>(h, uplo, 1, 1, a.ptr, desc, &pivot, b.ptr, desc, info);
            else psytrs_nopiv<T>(h, uplo, 1, 1, a.ptr, desc, b.ptr, desc, info);
            if (info) throw std::runtime_error("Complex scalar solve returned nonzero INFO");
            const auto out = download(h, b.ptr, rhs.size());
            if (indx_g2l_r(desc, h, 0) >= 0 && indx_g2l_c(desc, h, 0) >= 0) {
                check_component<R>(out[0].real(), expected.real());
                check_component<R>(out[0].imag(), expected.imag());
            }
        }
}

template <class T>
void normal_scalar_factor(DdlaHandle_t h)
{
    const double divisor = 0x1.eb16f53e79189p-8;
    const double input = 0x1.eb16f53e79188p+1016;
    for (int n : {1, 2}) {
        int desc[DDLA_DLEN_];
        DDLA_CHECK(ddlaDescInit(desc, h, n, n, 1, 1, 0, 0));
        const auto factor = make_local<T>(h, desc, [&](int i, int j) {
            return i == j ? T(i == 0 ? divisor : 2) : T(0);
        });
        const auto rhs = make_local<T>(h, desc, [&](int i, int) { return T(i == 0 ? input : 0); });
        DeviceBuffer<T> a(h, factor.size()), b(h, rhs.size());
        upload(h, a.ptr, factor);
        const int pivots[] = {1, 2};
        for (char uplo : {'L', 'U'})
            for (bool symmetric : {false, true})
                for (bool pivoted : {false, true}) {
                    upload(h, b.ptr, rhs);
                    int info = -1;
                    if (symmetric && pivoted) psytrs<T>(h, uplo, n, n, a.ptr, desc, pivots, b.ptr, desc, info);
                    else if (symmetric) psytrs_nopiv<T>(h, uplo, n, n, a.ptr, desc, b.ptr, desc, info);
                    else if (pivoted) phetrs<T>(h, uplo, n, n, a.ptr, desc, pivots, b.ptr, desc, info);
                    else phetrs_nopiv<T>(h, uplo, n, n, a.ptr, desc, b.ptr, desc, info);
                    if (info) throw std::runtime_error("Normal scalar boundary returned nonzero INFO");
                    const auto actual = download(h, b.ptr, rhs.size());
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < n; ++i) {
                            const int row = indx_g2l_r(desc, h, i), column = indx_g2l_c(desc, h, j);
                            if (row < 0 || column < 0) continue;
                            const T x = actual[row + std::size_t(column) * desc[DDLA_LLD_]];
                            check_component<double>(std::real(x), i == 0 ? std::numeric_limits<double>::max() : 0);
                            check_component<double>(std::imag(x), 0);
                        }
                }
    }
}
} // namespace

int main(int argc, char** argv)
{
    ddla_test_mpi_init(&argc, &argv);
    try {
        int size = 1, rank = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        const std::string requested = argc > 1 ? argv[1] : "all";
        for (DdlaBackend backend : {DdlaBackend::CPU, DdlaBackend::GPU}) {
#if !DDLA_HAS_GPU
            if (backend == DdlaBackend::GPU) continue;
#endif
            if ((requested == "cpu" && backend != DdlaBackend::CPU) ||
                (requested == "gpu" && backend != DdlaBackend::GPU)) continue;
            DdlaHandle_t h = nullptr;
            DDLA_CHECK(ddlaInit(h, backend));
            DDLA_CHECK(ddlaSet(h, MPI_COMM_WORLD, size, 1));
#if DDLA_HAS_GPU
            gpublasPointerMode_t saved{};
            if (backend == DdlaBackend::GPU) {
                gpuStream_t alternate = nullptr;
                GPU_CHECK(gpuStreamCreateWithFlags(&alternate, gpuStreamNonBlocking));
                DDLA_CHECK(ddlaSetStream(h, alternate));
                BLAS_CHECK(gpublasGetPointerMode(static_cast<gpublasHandle_t>(h->blasH), &saved));
                BLAS_CHECK(gpublasSetPointerMode(static_cast<gpublasHandle_t>(h->blasH), GPUBLAS_POINTER_MODE_DEVICE));
            }
#endif
            local_cases<float>(h); local_cases<double>(h);
            local_cases<std::complex<float>>(h); local_cases<std::complex<double>>(h);
#if DDLA_HAS_GPU
            if (backend == DdlaBackend::GPU) {
                gpublasPointerMode_t actual;
                BLAS_CHECK(gpublasGetPointerMode(static_cast<gpublasHandle_t>(h->blasH), &actual));
                if (actual != GPUBLAS_POINTER_MODE_DEVICE)
                    throw std::runtime_error("RSCL changed the incoming BLAS pointer mode");
                BLAS_CHECK(gpublasSetPointerMode(static_cast<gpublasHandle_t>(h->blasH), saved));
            }
#endif
            scalar_factors<float>(h); scalar_factors<double>(h);
            scalar_factors<std::complex<float>>(h); scalar_factors<std::complex<double>>(h);
            complex_symmetric_factor<float>(h); complex_symmetric_factor<double>(h);
            normal_scalar_factor<double>(h); normal_scalar_factor<std::complex<double>>(h);
            if (size > 1) {
                DDLA_CHECK(ddlaSet(h, MPI_COMM_WORLD, 1, size));
                scalar_factors<double>(h); scalar_factors<std::complex<double>>(h);
            }
            DDLA_CHECK(ddlaDestroy(h));
            if (!rank) std::cout << "RSCL_PASS backend=" << int(backend) << " ranks=" << size << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
}
