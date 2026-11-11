#include "api_grid_test_common.h"
#include <ddla/ddla_vector.h>
#include <complex>
#include <cstring>
using namespace ddla;
int main(int argc, char** argv)
{
    ddla_test_mpi_init(&argc, &argv);
    int size = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (size < 2)
    {
        MPI_Finalize();
        return 77;
    }
    try
    {
        DdlaHandle_t h = nullptr;
        DDLA_CHECK(ddlaInit(h, DdlaBackend::GPU));
        DDLA_CHECK(ddlaSet(h, MPI_COMM_WORLD, 1, size));
        {
            int d[DDLA_DLEN_];
            DDLA_CHECK(ddlaDescInit(d, h, 65, 65, 16, 24, 0, size - 1));
            using T = std::complex<double>;
            vector<T> a(h, std::size_t(d[DDLA_LLD_]) * ddla_test::n_loc(h, d));
            DDLA_CHECK(ddlaMemset(h, a.data(), 0, a.size() * sizeof(T)));
            int piv[65], info = 0, caught = 0;
            try
            {
                phetrf<DdlaBackend::GPU, T>(h, 'L', 65, a.data(), d, piv, info);
            }
            catch (const std::runtime_error& error)
            {
                caught = std::strstr(error.what(), rank == 0 ? "injected root pivot failure"
                                                             : "failed on the root") != nullptr;
            }
            int total = 0;
            MPI_Allreduce(&caught, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if (total != size)
                throw std::runtime_error("Root failure did not reach every rank");
            // A subsequent local GPU factor verifies that the error path retired
            // its queued communication before releasing temporary buffers.
            vector<T> one(h, 1);
            const T value(2, 3);
            DDLA_CHECK(ddlaMemcpy(h, one.data(), &value, sizeof(value)));
            hetrf_nopiv<DdlaBackend::GPU, T>(h, 'U', 1, one.data(), 1, info);
            if (info)
                throw std::runtime_error("GPU stream unusable after collective failure");
        }
        DDLA_CHECK(ddlaDestroy(h));
        if (!rank)
            std::cout << "LDLT_ROOT_FAILURE_PROPAGATION_PASS" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return 0;
}
