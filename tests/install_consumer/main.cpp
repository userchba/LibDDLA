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

// No private implementation header or vendor header is needed. A dual library
// also exposes its GPU specialization to a CXX consumer, without initializing
// a GPU or requiring one to be present for this CPU-only execution.
using Syevd = void (*)(const ddla::DdlaHandle_t&, char, char, int, double*, int, double*, int*);
using Heevd = void (*)(const ddla::DdlaHandle_t&, char, char, int, std::complex<double>*, int, double*, int*);
using Scal = void (*)(const ddla::DdlaHandle_t&, int, const double&, double*, int);
#if DDLA_HAS_GPU
Scal volatile gpu_scal = static_cast<Scal>(&ddla::scal<ddla::DdlaBackend::GPU, double>);
Syevd volatile gpu_syevd = &ddla::syevd<ddla::DdlaBackend::GPU, double>;
Heevd volatile gpu_heevd = &ddla::heevd<ddla::DdlaBackend::GPU, std::complex<double>>;
#endif

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
        if (has_gpu != bool(DDLA_HAS_GPU) ||
            ddla::default_backend_v != (has_gpu ? ddla::DdlaBackend::GPU : ddla::DdlaBackend::CPU))
            throw std::runtime_error("installed header/library capability mismatch");
        ddla::DDLA_CHECK(ddla::ddlaInit(handle, ddla::DdlaBackend::CPU));
        ddla::DDLA_CHECK(ddla::ddlaSet(handle, MPI_COMM_WORLD, 1, 1));
        double values[] = {1.0, -2.0};
        ddla::scal<ddla::DdlaBackend::CPU>(handle, 2, 3.0, values, 1);
        if (values[0] != 3.0 || values[1] != -6.0)
            throw std::runtime_error("installed CPU specialization returned incorrect values");
        double a[] = {2., 1., 1., 2.}, w[2] = {};
        int info = -1;
        ddla::syevd<ddla::DdlaBackend::CPU>(handle, 'N', 'L', 2, a, 2, w, &info);
        if (info || std::abs(w[0] - 1.) > 1e-12 || std::abs(w[1] - 3.) > 1e-12)
            throw std::runtime_error("installed SYEVD returned incorrect eigenvalues");
        std::complex<double> z[] = {{2., 0.}, {0., -1.}, {0., 1.}, {2., 0.}};
        ddla::heevd<ddla::DdlaBackend::CPU>(handle, 'V', 'U', 2, z, 2, w, &info);
        if (info || std::abs(w[0] - 1.) > 1e-12 || std::abs(w[1] - 3.) > 1e-12)
            throw std::runtime_error("installed HEEVD returned incorrect eigenvalues");
        for (int j = 0; j < 2; ++j) {
            if (std::abs(2. * z[2*j] + std::complex<double>(0., 1.) * z[1+2*j] - w[j] * z[2*j]) > 1e-12 ||
                std::abs(std::complex<double>(0., -1.) * z[2*j] + 2. * z[1+2*j] - w[j] * z[1+2*j]) > 1e-12)
                throw std::runtime_error("installed HEEVD returned incorrect eigenvectors");
        }
#if DDLA_HAS_GPU
        if (!gpu_scal || !gpu_syevd || !gpu_heevd)
            throw std::runtime_error("installed GPU specialization is unavailable");
#endif
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
