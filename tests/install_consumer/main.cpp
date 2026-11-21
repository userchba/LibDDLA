#if __cplusplus != 201103L
#error "The installed public-header consumer must compile as C++11"
#endif

#include <cmath>
#include <complex>
#include <ddla/ddla.h>
#include <ddla/ddla_config.h>
#include <ddla/ddla_desc.h>
#include <ddla/ddla_handle_t.h>
#include <ddla/ddla_vector.h>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <stdexcept>
#include <type_traits>
#include <vector>

// Public scalar-only function pointers require neither private declarations nor
// vendor definitions. The same entry point dispatches from its supplied handle.
using Syevd = void (*)(const ddla::DdlaHandle_t &, char, char, int, double *, int, double *, int *);
using Heevd = void (*)(const ddla::DdlaHandle_t &, char, char, int, std::complex<double> *, int,
                       double *, int *);
using Phegst = void (*)(const ddla::DdlaHandle_t &, char, int, double *, const int *,
                        const double *, const int *);
using Scal = void (*)(const ddla::DdlaHandle_t &, int, const double &, double *, int);
Phegst volatile public_phegst = &ddla::phegst<double>;
Scal volatile public_scal = static_cast<Scal>(&ddla::scal<double>);
Syevd volatile public_syevd = &ddla::syevd<double>;
Heevd volatile public_heevd = &ddla::heevd<std::complex<double>>;

using PheevdChebyshev = void (*)(const ddla::DdlaHandle_t &, char, int, int, int, double *,
                                 const int *, double *, double *, const int *, double, int, int,
                                 bool, int &, int &, int &);
PheevdChebyshev volatile public_pheevd_chebyshev = &ddla::pheevd_chebyshev<double>;

template <class T> std::complex<double> cheb_wide(T x)
{
    return std::complex<double>(double(x), 0);
}
template <class R> std::complex<double> cheb_wide(std::complex<R> x)
{
    return std::complex<double>(x.real(), x.imag());
}
// Installed headers and a real owning call, all four scalar symbols, C++11.
// The analytic diagonal oracle uses O(n*nev) output plus a tiny local input.
template <class T> void check_chebyshev(const ddla::DdlaHandle_t &handle)
{
    typedef typename ddla::real_type<T>::type R;
    const int n = 4, nev = 2;
    int da[ddla::DDLA_DLEN_], dz[ddla::DDLA_DLEN_];
    ddla::DDLA_CHECK(ddla::ddlaDescInit(da, handle, n, n, 1, 1, 0, 0));
    ddla::DDLA_CHECK(ddla::ddlaDescInit(dz, handle, n, nev, 1, 1, 0, 0));
    std::vector<T> a(n * n), z(n * nev, T(std::numeric_limits<R>::quiet_NaN()));
    const double eigenvalues[] = {-2., -1., 3., 4.};
    const double tolerance = std::is_same<R, float>::value ? 4e-5 : 2e-11;
    for (int repeat = 0; repeat < 2; ++repeat)
    {
        const double shift = .125 * repeat;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                a[i + n * j] = i < j ? T(std::numeric_limits<R>::quiet_NaN())
                                     : T(i == j ? eigenvalues[i] + shift : 0);
        R w[nev + 2] = {R(-313), R(0), R(0), R(-313)};
        int iterations = -1, converged = -1, info = -1;
        ddla::pheevd_chebyshev<T>(handle, 'L', n, nev, 1, a.data(), da, w + 1, z.data(), dz,
                                  tolerance, 80, 12, repeat != 0, iterations, converged, info);
        if (info || converged != nev || w[0] != R(-313) || w[nev + 1] != R(-313) ||
            (repeat && iterations < 1))
            throw std::runtime_error("installed Chebyshev status/nev guard/warm-filter failure");
        for (int j = 0; j < nev; ++j)
        {
            if (!std::isfinite(w[j + 1]) || std::abs(double(w[j + 1]) - eigenvalues[j] - shift) >
                                                (std::is_same<R, float>::value ? 1e-3 : 1e-9))
                throw std::runtime_error("installed Chebyshev finite/lowest-value failure");
            double norm = 0, residual = 0;
            for (int i = 0; i < n; ++i)
            {
                const std::complex<double> x = cheb_wide(z[i + n * j]);
                if (!std::isfinite(x.real()) || !std::isfinite(x.imag()))
                    throw std::runtime_error("installed Chebyshev nonfinite vector");
                norm += std::norm(x);
                residual += std::norm((eigenvalues[i] + shift - double(w[j + 1])) * x);
            }
            const double denominator = (4. + shift + std::abs(double(w[j + 1]))) * std::sqrt(norm);
            if (!(norm > 0) || !std::isfinite(norm) || !std::isfinite(residual) ||
                !std::isfinite(denominator) || std::sqrt(residual) > 4 * tolerance * denominator)
                throw std::runtime_error("installed Chebyshev independent residual failure");
            for (int k = 0; k <= j; ++k)
            {
                std::complex<double> dot = 0;
                for (int i = 0; i < n; ++i)
                    dot += std::conj(cheb_wide(z[i + n * k])) * cheb_wide(z[i + n * j]);
                if (!std::isfinite(dot.real()) || !std::isfinite(dot.imag()) ||
                    std::abs(dot - std::complex<double>(j == k ? 1 : 0, 0)) >
                        (std::is_same<R, float>::value ? 1e-3 : 1e-10))
                    throw std::runtime_error("installed Chebyshev orthogonality failure");
            }
        }
    }
}


template<class R>
void installed_mixed_batches(ddla::DdlaHandle_t h)
{
    typedef std::complex<R> Z;
    int dims[]={1};
    R real[]={R(2)};
    Z complex[]={Z(1,3)}, intermediate[]={Z(0)}, output[]={Z(0)};
    const R* rp[]={real};
    const Z* zp[]={complex};
    Z* cp[]={output};
    Z* tp[]={intermediate};
    ddla::gemmVbatched<R>(h,'N','N',dims,dims,dims,Z(1),rp,dims,zp,dims,Z(0),cp,dims,1);
    if (std::abs(output[0]-Z(2,6))>R(1e-5))
        throw std::runtime_error("installed mixed RC variable batch is incorrect");
    ddla::gemmVbatched<R>(h,'N','N',dims,dims,dims,Z(1),zp,dims,rp,dims,Z(0),cp,dims,1);
    if (std::abs(output[0]-Z(2,6))>R(1e-5))
        throw std::runtime_error("installed mixed CR variable batch is incorrect");
    ddla::gemmVbatched2s<R>(h,'N','N',dims,dims,dims,Z(1),rp,dims,zp,dims,Z(0),tp,dims,
        'N','N',dims,dims,dims,Z(1),rp,dims,dims,Z(0),cp,dims,true,1,dims,1);
    if (std::abs(output[0]-Z(4,12))>R(1e-5))
        throw std::runtime_error("installed mixed RC two-stage batch is incorrect");
    ddla::gemmVbatched2s<R>(h,'N','N',dims,dims,dims,Z(1),zp,dims,rp,dims,Z(0),tp,dims,
        'N','N',dims,dims,dims,Z(1),rp,dims,dims,Z(0),cp,dims,false,1,dims,1);
    if (std::abs(output[0]-Z(4,12))>R(1e-5))
        throw std::runtime_error("installed mixed CR two-stage batch is incorrect");
    ddla::gemmVbatched2s<R>(h,'N','N',dims,dims,dims,Z(1),rp,dims,zp,dims,Z(0),tp,dims,
        'N','N',dims,dims,dims,Z(1),zp,dims,dims,Z(0),cp,dims,true,1,dims,1);
    if (std::abs(output[0]-Z(-16,12))>R(1e-5))
        throw std::runtime_error("installed RCC two-stage batch is incorrect");
    ddla::gemmVbatched2s<R>(h,'N','N',dims,dims,dims,Z(1),zp,dims,rp,dims,Z(0),tp,dims,
        'N','N',dims,dims,dims,Z(1),zp,dims,dims,Z(0),cp,dims,false,1,dims,1);
    if (std::abs(output[0]-Z(-16,12))>R(1e-5))
        throw std::runtime_error("installed CRC two-stage batch is incorrect");
    ddla::gemmVbatched2s<R>(h,'N','N',dims,dims,dims,Z(1),zp,dims,zp,dims,Z(0),tp,dims,
        'N','N',dims,dims,dims,Z(1),rp,dims,dims,Z(0),cp,dims,true,1,dims,1);
    if (std::abs(output[0]-Z(-16,12))>R(1e-5))
        throw std::runtime_error("installed CCR two-stage batch is incorrect");
    R real_intermediate[1]={R(0)};
    R* rtp[]={real_intermediate};
    ddla::gemmVbatched2s<R>(h,'N','N',dims,dims,dims,R(1),rp,dims,rp,dims,R(0),rtp,dims,
        'N','N',dims,dims,dims,Z(1),zp,dims,dims,Z(0),cp,dims,false,1,dims,1);
    if (std::abs(output[0]-Z(4,12))>R(1e-5))
        throw std::runtime_error("installed RRC two-stage batch is incorrect");

}


int main(int argc, char **argv)
{
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS)
    {
        std::cerr << "MPI_Init_thread(FUNNELED) failed" << std::endl;
        return 1;
    }
    if (provided < MPI_THREAD_FUNNELED)
    {
        std::cerr << "MPI_THREAD_FUNNELED required; MPI provided " << provided << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    ddla::DdlaHandle_t handle = nullptr;
    int failed = 0;
    try
    {
        const bool has_gpu = ddla::ddlaBackendAvailable(ddla::DdlaBackend::GPU);
        if (has_gpu != bool(DDLA_HAS_GPU))
            throw std::runtime_error("installed header/library capability mismatch");
        ddla::DDLA_CHECK(ddla::ddlaInit(handle, ddla::DdlaBackend::CPU));
        ddla::DDLA_CHECK(ddla::ddlaSet(handle, MPI_COMM_WORLD, 1, 1));
        installed_mixed_batches<float>(handle);
        installed_mixed_batches<double>(handle);
        double values[] = {1.0, -2.0};
        ddla::scal(handle, 2, 3.0, values, 1);
        if (values[0] != 3.0 || values[1] != -6.0)
            throw std::runtime_error(
                "installed handle-dispatched CPU call returned incorrect values");
        double a[] = {2., 1., 1., 2.}, w[2] = {};
        std::complex<double> logdet;
        ddla::diag_logdet(handle, 2, a, 2, logdet);
        if (std::abs(logdet - std::complex<double>(std::log(4.), 0.)) > 1e-12)
            throw std::runtime_error("installed diagonal log determinant is incorrect");
        int diag_desc[ddla::DDLA_DLEN_];
        ddla::DDLA_CHECK(ddla::ddlaDescInit(diag_desc,handle,2,2,1,1,0,0));
        ddla::pdiag_logdet(handle,2,a,diag_desc,logdet);
        if (std::abs(logdet - std::complex<double>(std::log(4.), 0.)) > 1e-12)
            throw std::runtime_error("installed distributed diagonal log determinant is incorrect");
        int info = -1;
        ddla::syevd(handle, 'N', 'L', 2, a, 2, w, &info);
        if (info || std::abs(w[0] - 1.) > 1e-12 || std::abs(w[1] - 3.) > 1e-12)
            throw std::runtime_error("installed SYEVD returned incorrect eigenvalues");
        std::complex<double> z[] = {{2., 0.}, {0., -1.}, {0., 1.}, {2., 0.}};
        ddla::heevd(handle, 'V', 'U', 2, z, 2, w, &info);
        if (info || std::abs(w[0] - 1.) > 1e-12 || std::abs(w[1] - 3.) > 1e-12)
            throw std::runtime_error("installed HEEVD returned incorrect eigenvalues");
        for (int j = 0; j < 2; ++j)
        {
            if (std::abs(2. * z[2 * j] + std::complex<double>(0., 1.) * z[1 + 2 * j] -
                         w[j] * z[2 * j]) > 1e-12 ||
                std::abs(std::complex<double>(0., -1.) * z[2 * j] + 2. * z[1 + 2 * j] -
                         w[j] * z[1 + 2 * j]) > 1e-12)
                throw std::runtime_error("installed HEEVD returned incorrect eigenvectors");
        }
        int desc[ddla::DDLA_DLEN_];
        ddla::DDLA_CHECK(ddla::ddlaDescInit(desc, handle, 2, 2, 1, 1, 0, 0));
        double metric[] = {4., 2., 2., 10.};
        ddla::ppotrf<double>(handle, 'L', 2, metric, 1, 1, desc, info);
        if (info)
            throw std::runtime_error("installed factorization failed");
        double factor_before[4];
        for (int i = 0; i < 4; ++i)
            factor_before[i] = metric[i];
        double reduced[] = {16., 14., 14., 55.};
        const double *fixed_factor = metric;
        ddla::phegst(handle, 'L', 2, reduced, desc, fixed_factor, desc);
        const double expected[] = {4., 1., 1., 5.};
        const auto accepted = [&](const double* result) {
            for (int i = 0; i < 4; ++i)
                if (!std::isfinite(result[i]) || std::abs(result[i] - expected[i]) > 1e-12 ||
                    metric[i] != factor_before[i])
                    return false;
            return true;
        };
        if (!accepted(reduced))
            throw std::runtime_error("installed PHEGST changed its fixed factor or returned incorrect values");
        double invalid[4];
        for (int i = 0; i < 4; ++i) invalid[i] = reduced[i];
        invalid[1] = std::numeric_limits<double>::quiet_NaN();
        if (accepted(invalid))
            throw std::runtime_error("installed PHEGST acceptance predicate accepted NaN");
        std::cout << "installed PHEGST NaN rejection PASS\n";
        check_chebyshev<float>(handle);
        check_chebyshev<double>(handle);
        check_chebyshev<std::complex<float>>(handle);
        check_chebyshev<std::complex<double>>(handle);
        std::cout << "installed Chebyshev CPU calls PASS all four types cold/warm" << '\n';
        if (!public_scal || !public_syevd || !public_heevd || !public_phegst ||
            !public_pheevd_chebyshev)
            throw std::runtime_error("installed scalar-only public entry is unavailable");
        std::cout << "installed public-header consumer PASS DDLA_HAS_GPU=" << DDLA_HAS_GPU << '\n';
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        failed = 1;
    }
    if (handle)
        ddla::ddlaDestroy(handle);
    MPI_Finalize();
    return failed;
}
