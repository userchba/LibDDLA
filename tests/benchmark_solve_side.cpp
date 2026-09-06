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

Complex hpd_value(int i, int j, int n)
{
    if(i == j){
        return Complex(5.0 + 0.2 * n + 0.05 * i, 0.0);
    }
    const int lo = std::min(i, j);
    const int hi = std::max(i, j);
    const Complex val(0.01 * ((lo + 2 * hi) % 5 - 2),
                      0.006 * ((3 * lo + hi) % 7 - 3));
    return i < j ? val : std::conj(val);
}

Complex dominant_value(int i, int j, int n)
{
    if(i == j){
        return Complex(4.0 + 0.1 * i, 0.0);
    }
    return Complex(0.015 * ((i + 2 * j) % 5 - 2), 0.01 * ((2 * i + j) % 7 - 3));
}

Complex rhs_value(int i, int j, int n)
{
    return Complex(0.01 * ((i + 2 * j + n) % 11 - 5),
                   0.008 * ((3 * i + j) % 13 - 6));
}

template <typename Fn>
void fill_local(int rows, int cols, const int* desc, Complex* d_A,
                const DdlaHandle_t& handle, Fn value)
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

// Time ppotrs(side,'L','N') only; the ppotrf factorization is not timed.
double benchmark_ppotrs(char side, int n, int nrhs, const DdlaHandle_t& handle)
{
    const int nb = std::min(128, n);
    int descA[DDLA_DLEN_], descB[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, handle, n, n, nb, nb, 0, 0));
    if(side == 'L'){
        DDLA_CHECK(ddlaDescInit(descB, handle, n, nrhs, nb, nb, 0, 0));
    }else{
        DDLA_CHECK(ddlaDescInit(descB, handle, nrhs, n, nb, nb, 0, 0));
    }

    const size_t a_nelem = static_cast<size_t>(descA[DDLA_LLD_]) * ddla_test::n_loc(handle, descA);
    const size_t b_nelem = static_cast<size_t>(descB[DDLA_LLD_]) * ddla_test::n_loc(handle, descB);
    Complex* d_A = nullptr;
    Complex* d_B = nullptr;
    RUNTIME_CHECK(runtimeMallocAsync(reinterpret_cast<void**>(&d_A),
                                  std::max<size_t>(1, a_nelem) * sizeof(Complex),
                                  handle->stream));
    RUNTIME_CHECK(runtimeMallocAsync(reinterpret_cast<void**>(&d_B),
                                  std::max<size_t>(1, b_nelem) * sizeof(Complex),
                                  handle->stream));
    fill_local(n, n, descA, d_A, handle, [&](int i, int j){ return hpd_value(i, j, n); });
    const int b_rows = (side == 'L') ? n : nrhs;
    const int b_cols = (side == 'L') ? nrhs : n;
    fill_local(b_rows, b_cols, descB, d_B, handle, [&](int i, int j){ return rhs_value(i, j, n); });

    int info = -1;
    ppotrf(handle, 'L', n, d_A, 1, 1, descA, info);
    RUNTIME_CHECK(runtimeStreamSynchronize(handle->stream));

    MPI_Barrier(handle->comm);
    const double start = MPI_Wtime();
    ppotrs(handle, side, 'L', 'N', n, nrhs, d_A, descA, d_B, descB);
    RUNTIME_CHECK(runtimeStreamSynchronize(handle->stream));
    MPI_Barrier(handle->comm);
    const double elapsed = MPI_Wtime() - start;

    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, handle->comm);

    RUNTIME_CHECK(runtimeFreeAsync(d_A, handle->stream));
    RUNTIME_CHECK(runtimeFreeAsync(d_B, handle->stream));
    RUNTIME_CHECK(runtimeStreamSynchronize(handle->stream));

    if(handle->myid == 0){
        std::cout << "RESULT n=" << n
                  << " nrhs=" << nrhs
                  << " type=complex<double>"
                  << " op=ppotrs(handle, " << side << ",L,N)"
                  << " grid=2x2"
                  << " ranks=4"
                  << " nb=" << nb
                  << " time_s=" << std::fixed << std::setprecision(6)
                  << max_elapsed
                  << std::endl;
    }
    return max_elapsed;
}

// Time pgetrs(side,'N') only; the pgetrf factorization is not timed.
double benchmark_pgetrs(char side, int n, int nrhs, const DdlaHandle_t& handle)
{
    const int nb = std::min(128, n);
    int descA[DDLA_DLEN_], descB[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, handle, n, n, nb, nb, 0, 0));
    if(side == 'L'){
        DDLA_CHECK(ddlaDescInit(descB, handle, n, nrhs, nb, nb, 0, 0));
    }else{
        DDLA_CHECK(ddlaDescInit(descB, handle, nrhs, n, nb, nb, 0, 0));
    }

    const size_t a_nelem = static_cast<size_t>(descA[DDLA_LLD_]) * ddla_test::n_loc(handle, descA);
    const size_t b_nelem = static_cast<size_t>(descB[DDLA_LLD_]) * ddla_test::n_loc(handle, descB);
    Complex* d_A = nullptr;
    Complex* d_B = nullptr;
    RUNTIME_CHECK(runtimeMallocAsync(reinterpret_cast<void**>(&d_A),
                                  std::max<size_t>(1, a_nelem) * sizeof(Complex),
                                  handle->stream));
    RUNTIME_CHECK(runtimeMallocAsync(reinterpret_cast<void**>(&d_B),
                                  std::max<size_t>(1, b_nelem) * sizeof(Complex),
                                  handle->stream));
    fill_local(n, n, descA, d_A, handle, [&](int i, int j){ return dominant_value(i, j, n); });
    const int b_rows = (side == 'L') ? n : nrhs;
    const int b_cols = (side == 'L') ? nrhs : n;
    fill_local(b_rows, b_cols, descB, d_B, handle, [&](int i, int j){ return rhs_value(i, j, n); });

    std::vector<int> ipiv(ddla_test::m_loc(handle, descA));
    int info = -1;
    pgetrf(handle, n, n, d_A, descA, ipiv.data(), info);
    RUNTIME_CHECK(runtimeStreamSynchronize(handle->stream));

    MPI_Barrier(handle->comm);
    const double start = MPI_Wtime();
    pgetrs(handle, side, 'N', n, nrhs, d_A, descA, ipiv.data(), d_B, descB);
    RUNTIME_CHECK(runtimeStreamSynchronize(handle->stream));
    MPI_Barrier(handle->comm);
    const double elapsed = MPI_Wtime() - start;

    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, handle->comm);

    RUNTIME_CHECK(runtimeFreeAsync(d_A, handle->stream));
    RUNTIME_CHECK(runtimeFreeAsync(d_B, handle->stream));
    RUNTIME_CHECK(runtimeStreamSynchronize(handle->stream));

    if(handle->myid == 0){
        std::cout << "RESULT n=" << n
                  << " nrhs=" << nrhs
                  << " type=complex<double>"
                  << " op=pgetrs(handle, " << side << ",N)"
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
            std::cerr << "benchmark_solve_side requires exactly 4 MPI ranks for a 2x2 grid"
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
        std::cout << "=== solve benchmark: complex<double>, 4 MPI ranks, 2x2 grid, nrhs=n ==="
                  << std::endl;
        std::cout << "=== side='L' vs side='R' for ppotrs and pgetrs(handle, trans='N'), solve-only timing ==="
                  << std::endl;
    }

    for(int n : sizes){
        benchmark_ppotrs('L', n, n, handle);
        benchmark_ppotrs('R', n, n, handle);
        benchmark_pgetrs('L', n, n, handle);
        benchmark_pgetrs('R', n, n, handle);
    }

    ddlaDestroy(handle);
    MPI_Finalize();
    return 0;
}
