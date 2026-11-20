#if __cplusplus != 201103L
#error "The installed public-header consumer must compile as C++11"
#endif

#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>
#include <ddla/ddla_desc.h>
#include <ddla/ddla_vector.h>
#include <ddla/ddla_config.h>
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <limits>

// Public scalar-only function pointers require neither private declarations nor
// vendor definitions. The same entry point dispatches from its supplied handle.
using Syevd = void (*)(const ddla::DdlaHandle_t&, char, char, int, double*, int, double*, int*);
using Heevd = void (*)(const ddla::DdlaHandle_t&, char, char, int, std::complex<double>*, int, double*, int*);
using Phegst = void (*)(const ddla::DdlaHandle_t &, char, int, double *, const int *,
                        const double *, const int *);
using Scal = void (*)(const ddla::DdlaHandle_t&, int, const double&, double*, int);
Phegst volatile public_phegst = &ddla::phegst<double>;
Scal volatile public_scal = static_cast<Scal>(&ddla::scal<double>);
Syevd volatile public_syevd = &ddla::syevd<double>;
Heevd volatile public_heevd = &ddla::heevd<std::complex<double>>;

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

int main(int argc, char** argv)
{
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        std::cerr << "MPI_Init_thread(FUNNELED) failed" << std::endl;
        return 1;
    }
    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "MPI_THREAD_FUNNELED required; MPI provided " << provided << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    ddla::DdlaHandle_t handle = nullptr;
    int failed = 0;
    try {
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
            throw std::runtime_error("installed handle-dispatched CPU call returned incorrect values");
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
        for (int j = 0; j < 2; ++j) {
            if (std::abs(2. * z[2*j] + std::complex<double>(0., 1.) * z[1+2*j] - w[j] * z[2*j]) > 1e-12 ||
                std::abs(std::complex<double>(0., -1.) * z[2*j] + 2. * z[1+2*j] - w[j] * z[1+2*j]) > 1e-12)
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
        if (!public_scal || !public_syevd || !public_heevd || !public_phegst)
            throw std::runtime_error("installed scalar-only public entry is unavailable");
        std::cout << "installed public-header consumer PASS DDLA_HAS_GPU=" << DDLA_HAS_GPU << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        failed = 1;
    }
    if (handle)
        ddla::ddlaDestroy(handle);
    MPI_Finalize();
    return failed;
}
