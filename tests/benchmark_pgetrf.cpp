#include <algorithm>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include <mpi.h>

#include "benchmark_grid_options.h"

#include <ddla/ddla.h>
#include "test_desc_helpers.h"
#include "ddla_connector.h"
#include "ddla_stream_impl.h"
#include "scal.h"

using namespace ddla;

using Complex = std::complex<double>;

constexpr unsigned long long kRandomSeed = 20260710ULL;

void fill_matrix(int n, const int* desc, Complex* d_A, const DdlaHandle_t& handle)
{
    const size_t nelem = static_cast<size_t>(ddla_test::m_loc(handle, desc)) * ddla_test::n_loc(handle, desc);
    derandGenerator_t generator;
    DERAND_CHECK(derandCreateGenerator(&generator, DERAND_RNG_PSEUDO_DEFAULT));
    DERAND_CHECK(derandSetPseudoRandomGeneratorSeed(
        generator, kRandomSeed + static_cast<unsigned long long>(handle->myid)));
    DERAND_CHECK(derandGenerateUniformDouble(
        generator, reinterpret_cast<double*>(d_A), nelem * 2));
    DERAND_CHECK(derandDestroyGenerator(generator));
    BLAS_CHECK(deblasScal(handle->blasH, nelem, Complex(1.0e-4, 0.0), d_A, 1));

    const Complex diag(2.0, 0.0);
    for(int i = 0; i < n; ++i){
        const int iloc = indx_g2l_r(desc, handle, i);
        const int jloc = indx_g2l_c(desc, handle, i);
        if(iloc >= 0 && jloc >= 0){
            RUNTIME_CHECK(runtimeMemcpyAsync(d_A + iloc + jloc * desc[DDLA_LLD_], &diag,
                                          sizeof(Complex), runtimeMemcpyHostToDevice,
                                          handle->stream));
        }
    }
    RUNTIME_CHECK(runtimeStreamSynchronize(handle->stream));
}

double benchmark_pgetrf(int n, const DdlaHandle_t& handle,
                        const benchmark_cli::Options& options, bool warmup)
{
    const int nb = std::min(128, n);
    int desc[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(desc, handle, n, n, nb, nb, 0, 0));

    const size_t nelem = static_cast<size_t>(ddla_test::m_loc(handle, desc)) * ddla_test::n_loc(handle, desc);
    Complex* d_A = nullptr;
    RUNTIME_CHECK(runtimeMallocAsync(reinterpret_cast<void**>(&d_A),
                                  nelem * sizeof(Complex), handle->stream));
    fill_matrix(n, desc, d_A, handle);

    std::vector<int> ipiv(ddla_test::m_loc(handle, desc));
    int info = -1;

    MPI_Barrier(handle->comm);
    const double start = MPI_Wtime();
    pgetrf(handle, n, n, d_A, desc, ipiv.data(), info);
    RUNTIME_CHECK(runtimeStreamSynchronize(handle->stream));
    MPI_Barrier(handle->comm);
    const double elapsed = MPI_Wtime() - start;

    int global_info = 0;
    MPI_Allreduce(&info, &global_info, 1, MPI_INT, MPI_MAX, handle->comm);
    if(global_info != 0){
        if(handle->myid == 0){
            std::cerr << "pgetrf failed for n=" << n << ", info=" << global_info << std::endl;
        }
        MPI_Abort(handle->comm, 1);
    }

    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, handle->comm);

    RUNTIME_CHECK(runtimeFreeAsync(d_A, handle->stream));
    RUNTIME_CHECK(runtimeStreamSynchronize(handle->stream));

    if(handle->myid == 0){
        std::cout << (warmup ? "WARMUP" : "RESULT") << " n=" << n
                  << " type=complex<double>"
                  << " grid=" << benchmark_cli::grid_name(options)
                  << " ranks=" << handle->nprocs
                  << " nb=" << nb
                  << " seed=" << kRandomSeed
                  << " time_s=" << std::fixed << std::setprecision(6)
                  << max_elapsed
                  << std::endl;
    }
    return max_elapsed;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int nprocs = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    benchmark_cli::Options options;
    std::string option_error;
    if(!benchmark_cli::parse(argc, argv, false, 1, options, option_error)){
        if(rank == 0){
            std::cerr << "Error: " << option_error << std::endl;
            std::cerr << benchmark_cli::usage(argv[0], false) << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if(nprocs != options.nprows * options.npcols){
        if(rank == 0){
            std::cerr << "--grid " << benchmark_cli::grid_name(options)
                      << " requires " << options.nprows * options.npcols
                      << " MPI ranks, but this run has " << nprocs << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    DdlaHandle_t handle = nullptr;
    ddlaInit(handle);
    ddlaSet(handle, MPI_COMM_WORLD, options.nprows, options.npcols);

    if(handle->myid == 0){
        std::cout << "=== pgetrf benchmark: complex<double>, " << nprocs
                  << " MPI ranks, " << benchmark_cli::grid_name(options)
                  << " grid, seed=" << kRandomSeed << " ===" << std::endl;
    }

    for(size_t i = 0; i < options.sizes.size(); ++i){
        const bool warmup = i == 0 && options.sizes[i] == 500;
        benchmark_pgetrf(options.sizes[i], handle, options, warmup);
    }

    ddlaDestroy(handle);
    MPI_Finalize();
    return 0;
}
