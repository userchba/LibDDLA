#include <mpi.h>
#include <iostream>
#include <vector>
#include <complex>
#include <iomanip>
#include <algorithm>
#include <ddla/ddla.h>
#include "test_desc_helpers.h"
#include "ddla_connector.h"
#include "ddla_stream_impl.h"

using namespace ddla;

/**
 * Benchmark pgemm with complex<double> on a 2x2 process grid (1 node, 4 GPU).
 * Compares 4 transpose combinations: (N,N), (C,N), (N,C), (C,C)
 * for matrix dimensions 100, 5000, 10000, 15000.
 */

struct BenchResult {
    int n;
    char transa, transb;
    double time;
};

void benchmark_pgemm(int n, char transa, char transb,
                     const DdlaHandle_t& ddla_handle,
                     std::vector<BenchResult>& results)
{
    int nb = std::min(128, n);
    int descA[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, ddla_handle, n, n, nb, nb, 0, 0));
    int descB[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descB, ddla_handle, n, n, nb, nb, 0, 0));
    int descC[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descC, ddla_handle, n, n, nb, nb, 0, 0));

    int myid = ddla_test::mypcol(ddla_handle) + ddla_test::myprow(ddla_handle) * ddla_test::npcols(ddla_handle);
    const size_t nelem = static_cast<size_t>(ddla_test::m_loc(ddla_handle, descA)) * ddla_test::n_loc(ddla_handle, descA);
    const size_t size = nelem * sizeof(std::complex<double>);

    std::complex<double>* d_A = nullptr;
    std::complex<double>* d_B = nullptr;
    std::complex<double>* d_C = nullptr;
    RUNTIME_CHECK(runtimeMalloc(&d_A, size));
    RUNTIME_CHECK(runtimeMalloc(&d_B, size));
    RUNTIME_CHECK(runtimeMalloc(&d_C, size));

    random_generate(d_A, nelem);
    random_generate(d_B, nelem);
    random_generate(d_C, nelem);

    std::complex<double> alpha(1.0, 0.0);
    std::complex<double> beta(0.0, 0.0);

    // Warm up
    pgemm(ddla_handle, transa, transb, n, n, n, alpha, d_A, descA, d_B, descB, beta, d_C, descC);
    RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(MPI_COMM_WORLD);

    // Timed runs: more iterations for small matrices
    int niter = (n <= 1000) ? 10 : (n <= 5000 ? 3 : 1);
    double start = MPI_Wtime();
    for (int iter = 0; iter < niter; iter++) {
        pgemm(ddla_handle, transa, transb, n, n, n, alpha, d_A, descA, d_B, descB, beta, d_C, descC);
    }
    RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = (MPI_Wtime() - start) / niter;

    double max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, ddla_handle->comm);

    if (myid == 0) {
        std::cout << "n=" << std::setw(6) << n
                  << "  opA=" << transa << " opB=" << transb
                  << "  time=" << std::fixed << std::setprecision(4)
                  << max_elapsed << "s"
                  << "  (iters=" << niter << ")"
                  << std::endl;
        results.push_back({n, transa, transb, max_elapsed});
    }

    RUNTIME_CHECK(runtimeFree(d_A));
    RUNTIME_CHECK(runtimeFree(d_B));
    RUNTIME_CHECK(runtimeFree(d_C));
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int myid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    DdlaHandle_t ddla_handle = nullptr;
    ddlaInit(ddla_handle);
    ddlaSet(ddla_handle, MPI_COMM_WORLD, 2, 2);

    if (myid == 0) {
        std::cout << "=== pgemm benchmark (complex<double>, " << nprocs
                  << " MPI, 2x2 grid, V100) ===" << std::endl;
    }

    std::vector<int> sizes = {500, 5000, 10000, 15000};
    if (argc > 1) {
        sizes.clear();
        for (int i = 1; i < argc; ++i)
            sizes.push_back(std::atoi(argv[i]));
    }

    struct OpPair { char a, b; const char* label; };
    std::vector<OpPair> ops = {
        {'N','N',"(N,N)"},
        {'C','N',"(C,N)"},
        {'N','C',"(N,C)"},
        {'C','C',"(C,C)"},
    };

    std::vector<BenchResult> results;

    for (int n : sizes) {
        for (auto& op : ops) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (myid == 0)
                std::cout << "  running n=" << n << " " << op.label << " ..." << std::flush;
            benchmark_pgemm(n, op.a, op.b, ddla_handle, results);
            if (myid == 0)
                std::cout << " done" << std::endl;
        }
    }

    // Print summary table
    if (myid == 0) {
        std::cout << std::endl;
        std::cout << "=== Summary (time in seconds) ===" << std::endl;
        std::cout << std::setw(8) << "n";
        for (auto& op : ops)
            std::cout << std::setw(12) << op.label;
        std::cout << std::endl;
        std::cout << std::string(8 + 12*ops.size(), '-') << std::endl;

        for (int n : sizes) {
            std::cout << std::setw(8) << n;
            for (auto& op : ops) {
                double t = -1;
                for (auto& r : results) {
                    if (r.n == n && r.transa == op.a && r.transb == op.b) {
                        t = r.time;
                        break;
                    }
                }
                if (t >= 0)
                    std::cout << std::setw(12) << std::fixed << std::setprecision(4) << t;
                else
                    std::cout << std::setw(12) << "N/A";
            }
            std::cout << std::endl;
        }
    }

    ddlaDestroy(ddla_handle);
    MPI_Finalize();
    return 0;
}
