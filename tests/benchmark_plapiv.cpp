#include <algorithm>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include <mpi.h>

#include <ddla/ddla.h>
#include "test_desc_helpers.h"
#include "ddla_connector.h"
#include "ddla_stream_impl.h"

using namespace ddla;

using Complex = std::complex<double>;

Complex value(int i, int j)
{
    return Complex(10.0 * i + j, -0.5 * i + 0.25 * j);
}

// Fill a distributed matrix from a host generator.
void fill_local(int rows, int cols, const int* desc, Complex* d_A,
                const DdlaHandle_t& handle)
{
    std::vector<Complex> local(static_cast<size_t>(desc[DDLA_LLD_]) * ddla_test::n_loc(handle, desc),
                               Complex(0.0, 0.0));
    for(int jloc = 0; jloc < ddla_test::n_loc(handle, desc); ++jloc){
        const int j = indx_l2g_c(desc, handle, jloc);
        if(j >= cols) continue;
        for(int iloc = 0; iloc < ddla_test::m_loc(handle, desc); ++iloc){
            const int i = indx_l2g_r(desc, handle, iloc);
            if(i >= rows) continue;
            local[iloc + jloc * desc[DDLA_LLD_]] = value(i, j);
        }
    }
    RUNTIME_CHECK(runtimeMemcpyAsync(d_A, local.data(), local.size() * sizeof(Complex),
                                   runtimeMemcpyHostToDevice, handle->stream));
    RUNTIME_CHECK(runtimeStreamSynchronize(handle->stream));
}

// Time plapiv(direc, rowcol, 'C', m, n) on an m x n matrix with a pivot
// vector that swaps every other row (a non-trivial, communication-exercising
// permutation).  Returns the max time across ranks.
double benchmark_plapiv(char direc, char rowcol, int m, int n,
                        const DdlaHandle_t& handle)
{
    const int nb = std::min(128, std::min(m, n));
    int desc[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(desc, handle, m, n, nb, nb, 0, 0));

    const size_t nelem = static_cast<size_t>(desc[DDLA_LLD_]) * ddla_test::n_loc(handle, desc);
    Complex* d_A = nullptr;
    RUNTIME_CHECK(runtimeMallocAsync(reinterpret_cast<void**>(&d_A),
                                  std::max<size_t>(1, nelem) * sizeof(Complex),
                                  handle->stream));
    fill_local(m, n, desc, d_A, handle);

    // Pivot vector: swap row/col k with m-1-k for k in [0, m/2).  Local entry
    // for global k stores the 1-based target.
    std::vector<int> ipiv(ddla_test::m_loc(handle, desc), 0);
    for(int iloc = 0; iloc < ddla_test::m_loc(handle, desc); ++iloc){
        const int k = indx_l2g_r(desc, handle, iloc);
        if(k < m - 1 - k)
            ipiv[iloc] = (m - 1 - k) + 1;
        else
            ipiv[iloc] = k + 1;
    }

    MPI_Barrier(handle->comm);
    const double start = MPI_Wtime();
    plapiv(handle, direc, rowcol, 'C', m, n, d_A, desc, ipiv.data(), desc, nullptr);
    RUNTIME_CHECK(runtimeStreamSynchronize(handle->stream));
    MPI_Barrier(handle->comm);
    const double elapsed = MPI_Wtime() - start;

    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, handle->comm);

    RUNTIME_CHECK(runtimeFreeAsync(d_A, handle->stream));
    RUNTIME_CHECK(runtimeStreamSynchronize(handle->stream));

    if(handle->myid == 0){
        std::cout << "RESULT m=" << m
                  << " n=" << n
                  << " type=complex<double>"
                  << " op=plapiv(handle, " << direc << "," << rowcol << ",C)"
                  << " grid=2x2"
                  << " ranks=4"
                  << " nb=" << nb
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
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(nprocs != 4){
        if(rank == 0){
            std::cerr << "benchmark_plapiv requires exactly 4 MPI ranks for a 2x2 grid"
                      << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    DdlaHandle_t handle = nullptr;
    ddlaInit(handle);
    ddlaSet(handle, MPI_COMM_WORLD, 2, 2);

    std::vector<int> sizes = {500, 5000, 10000, 15000};
    if(argc > 1){
        sizes.clear();
        for(int i = 1; i < argc; ++i){
            sizes.push_back(std::atoi(argv[i]));
        }
    }

    if(handle->myid == 0){
        std::cout << "=== plapiv benchmark: complex<double>, 4 MPI ranks, 2x2 grid ==="
                  << std::endl;
        std::cout << "=== all four (direc, rowcol) combos, pivot swaps k <-> m-1-k ==="
                  << std::endl;
    }

    for(int n : sizes){
        benchmark_plapiv('F', 'R', n, n, handle);
        benchmark_plapiv('B', 'R', n, n, handle);
        benchmark_plapiv('F', 'C', n, n, handle);
        benchmark_plapiv('B', 'C', n, n, handle);
    }

    ddlaDestroy(handle);
    MPI_Finalize();
    return 0;
}
