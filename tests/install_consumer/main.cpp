#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>
#include <ddla/ddla_desc.h>
#include <ddla/ddla_vector.h>
#include <ddla/ddla_config.h>
#include <mpi.h>
#include <iostream>
#include <stdexcept>

// No private implementation header or vendor header is needed. A dual library
// also exposes its GPU specialization to a CXX consumer, without initializing
// a GPU or requiring one to be present for this CPU-only execution.
using Scal = void (*)(const ddla::DdlaHandle_t&, int, const double&, double*, int);
#if DDLA_HAS_GPU
Scal volatile gpu_scal = static_cast<Scal>(&ddla::scal<ddla::DdlaBackend::GPU, double>);
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
#if DDLA_HAS_GPU
        if (!gpu_scal)
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
