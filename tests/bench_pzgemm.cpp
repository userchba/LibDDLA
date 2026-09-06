#include <cassert>
#include <cmath>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <complex>
#include <ddla/ddla.h>
#include "test_desc_helpers.h"
#include "ddla_connector.h"
#include "ddla_stream_impl.h"

using namespace ddla;

namespace {

// Complex multiply FLOP count: each complex mul-add = 6 real flops (3 mul + 3 add,
// using (a+bi)(c+di) = ac-bd + (ad+bc)i). ScaLAPACK convention counts 8.
// We use the standard 8 flops per complex mul-add for GFLOPS reporting.
constexpr double FLOPS_PER_ZGEMM_ELEM = 8.0;

std::complex<double> op_value(char trans, const std::vector<std::complex<double>>& mat,
                              int m, int n, int i, int j)
{
    if(trans == 'N'){
        return mat[i + j * m];
    }else if(trans == 'T'){
        return mat[j + i * m];
    }else{
        return std::conj(mat[j + i * m]);
    }
}

// Lightweight correctness check for small matrices only.
void check_correctness(char transa, char transb,
                       int m, int n, int k,
                       const DdlaHandle_t& handle, const int* descA, const int* descB, const int* descC,
                       const std::vector<std::complex<double>>& h_A,
                       const std::vector<std::complex<double>>& h_B,
                       const std::vector<std::complex<double>>& h_C_out,
                       int myid, int nprocs)
{
    auto gather_global = [&](const int* desc, const std::vector<std::complex<double>>& local)
                         -> std::vector<std::complex<double>>
    {
        int mg = desc[DDLA_M_];
        int ng = desc[DDLA_N_];
        std::vector<std::complex<double>> global(mg * ng);
        std::vector<int> recvcounts(nprocs), displs(nprocs);
        int loc_size = static_cast<int>(local.size());
        MPI_Allgather(&loc_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        displs[0] = 0;
        for(int i=1;i<nprocs;i++) displs[i] = displs[i-1] + recvcounts[i-1];
        std::vector<std::complex<double>> all_local(displs[nprocs-1] + recvcounts[nprocs-1]);
        MPI_Allgatherv(local.data(), loc_size, MPI_C_DOUBLE_COMPLEX,
                       all_local.data(), recvcounts.data(), displs.data(), MPI_C_DOUBLE_COMPLEX,
                       MPI_COMM_WORLD);
        int npcols = ddla_test::npcols(handle);
        for(int src=0; src<nprocs; src++){
            int prow = src / npcols;
            int pcol = src % npcols;
            int offset = displs[src];
            int count = recvcounts[src];
            if(count == 0) continue;
            int m_loc = num_loc(mg, desc[DDLA_MB_], prow, desc[DDLA_RSRC_], ddla_test::nprows(handle));
            int n_loc = num_loc(ng, desc[DDLA_NB_], pcol, desc[DDLA_CSRC_], npcols);
            if(m_loc * n_loc != count) continue;
            for(int j_loc=0; j_loc<n_loc; j_loc++){
                int j_g = indxl2g(j_loc, desc[DDLA_NB_], pcol, desc[DDLA_CSRC_], npcols);
                for(int i_loc=0; i_loc<m_loc; i_loc++){
                    int i_g = indxl2g(i_loc, desc[DDLA_MB_], prow, desc[DDLA_RSRC_], ddla_test::nprows(handle));
                    global[i_g + j_g * mg] = all_local[offset + i_loc + j_loc * m_loc];
                }
            }
        }
        return global;
    };

    auto g_A = gather_global(descA, h_A);
    auto g_B = gather_global(descB, h_B);
    auto g_C = gather_global(descC, h_C_out);

    int ma = (transa == 'N') ? m : k;
    int na = (transa == 'N') ? k : m;
    int mb_ = (transb == 'N') ? k : n;
    int nb_ = (transb == 'N') ? n : k;

    double max_err = 0.0;
    for(int j=0; j<n; j++){
        for(int i=0; i<m; i++){
            std::complex<double> ref(0.0, 0.0);
            for(int l=0; l<k; l++){
                ref += op_value(transa, g_A, ma, na, i, l) *
                       op_value(transb, g_B, mb_, nb_, l, j);
            }
            std::complex<double> diff = g_C[i + j * m] - ref;
            max_err = std::max(max_err, std::abs(diff));
        }
    }

    double global_max_err;
    MPI_Reduce(&max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(myid == 0){
        std::cout << "    [verify] max_err = " << global_max_err
                  << (global_max_err < 1e-8 ? "  PASS" : "  FAIL") << std::endl;
    }
    if(global_max_err > 1e-8 && myid == 0){
        std::cerr << "FAIL: correctness check failed" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void bench_one(char transa, char transb, int N, int nb,
               const DdlaHandle_t& ddla_handle, int n_warmup, int n_iter)
{
    int m = N, n = N, k = N;
    int myid = ddla_handle->myid;
    int nprocs = ddla_handle->nprocs;

    int descA[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, ddla_handle, m, k, nb, nb, 0, 0));
    int descB[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descB, ddla_handle, k, n, nb, nb, 0, 0));
    int descC[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descC, ddla_handle, m, n, nb, nb, 0, 0));

    // Initialize with deterministic pseudo-random data
    std::vector<std::complex<double>> h_A(ddla_test::m_loc(ddla_handle, descA) * ddla_test::n_loc(ddla_handle, descA));
    std::vector<std::complex<double>> h_B(ddla_test::m_loc(ddla_handle, descB) * ddla_test::n_loc(ddla_handle, descB));
    std::vector<std::complex<double>> h_C(ddla_test::m_loc(ddla_handle, descC) * ddla_test::n_loc(ddla_handle, descC));

    // Simple LCG for reproducible data (avoid heavy mt19937 for large sizes)
    auto gen_val = [](uint64_t seed) -> std::complex<double> {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        double r1 = static_cast<double>((seed >> 33) & 0xFFFFFF) / 0xFFFFFF - 0.5;
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        double r2 = static_cast<double>((seed >> 33) & 0xFFFFFF) / 0xFFFFFF - 0.5;
        return std::complex<double>(r1 * 2.0, r2 * 2.0);
    };
    uint64_t base = static_cast<uint64_t>(myid) * 1000003ULL + 42;
    for(size_t i = 0; i < h_A.size(); i++) h_A[i] = gen_val(base + i);
    for(size_t i = 0; i < h_B.size(); i++) h_B[i] = gen_val(base + i + 7);
    for(size_t i = 0; i < h_C.size(); i++) h_C[i] = gen_val(base + i + 13);

    std::complex<double>* d_A = nullptr;
    std::complex<double>* d_B = nullptr;
    std::complex<double>* d_C = nullptr;
    RUNTIME_CHECK(runtimeMalloc(&d_A, sizeof(std::complex<double>) * h_A.size()));
    RUNTIME_CHECK(runtimeMalloc(&d_B, sizeof(std::complex<double>) * h_B.size()));
    RUNTIME_CHECK(runtimeMalloc(&d_C, sizeof(std::complex<double>) * h_C.size()));

    RUNTIME_CHECK(runtimeMemcpy(d_A, h_A.data(), sizeof(std::complex<double>) * h_A.size(), runtimeMemcpyHostToDevice));
    RUNTIME_CHECK(runtimeMemcpy(d_B, h_B.data(), sizeof(std::complex<double>) * h_B.size(), runtimeMemcpyHostToDevice));
    RUNTIME_CHECK(runtimeMemcpy(d_C, h_C.data(), sizeof(std::complex<double>) * h_C.size(), runtimeMemcpyHostToDevice));

    std::complex<double> alpha(1.0, 0.0);
    std::complex<double> beta(0.0, 0.0);

    // Warmup
    for(int it = 0; it < n_warmup; it++){
        pgemm(ddla_handle, transa, transb, m, n, k, alpha, d_A, descA, d_B, descB, beta, d_C, descC);
        RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
    }

    // Timed iterations
    double t_start = MPI_Wtime();
    for(int it = 0; it < n_iter; it++){
        pgemm(ddla_handle, transa, transb, m, n, k, alpha, d_A, descA, d_B, descB, beta, d_C, descC);
        RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
    }
    double t_end = MPI_Wtime();
    double elapsed = (t_end - t_start) / n_iter;

    // Correctness check for small sizes only
    if(N <= 1000){
        RUNTIME_CHECK(runtimeMemcpy(h_C.data(), d_C, sizeof(std::complex<double>) * h_C.size(), runtimeMemcpyDeviceToHost));
        check_correctness(transa, transb, m, n, k, ddla_handle, descA, descB, descC,
                          h_A, h_B, h_C, myid, nprocs);
    }

    double flops = FLOPS_PER_ZGEMM_ELEM * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    double gflops = flops / elapsed / 1e9;

    // Report max time across ranks for consistency
    double max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(myid == 0){
        std::cout << "  pgemm(ddla_handle, " << transa << "," << transb << ")"
                  << "  N=" << N
                  << "  time=" << max_elapsed << "s"
                  << "  perf=" << (flops / max_elapsed / 1e9) << " GFLOPS"
                  << std::endl;
    }

    RUNTIME_CHECK(runtimeFree(d_A));
    RUNTIME_CHECK(runtimeFree(d_B));
    RUNTIME_CHECK(runtimeFree(d_C));
}

} // anonymous namespace

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    std::vector<std::pair<int,int>> grids;
    if(nprocs == 4){
        grids = {{2,2}};
    }else if(nprocs == 6){
        grids = {{2,3}};
    }else if(nprocs == 8){
        grids = {{2,4}};
    }else if(nprocs == 1){
        grids = {{1,1}};
    }else{
        grids = {{-1,-1}};
    }

    std::vector<int> sizes = {500, 5000, 10000, 15000};
    std::vector<std::pair<char,char>> trans_pairs = {
        {'N','N'}, {'C','N'}, {'N','C'}, {'C','C'}
    };

    for(const auto& grid : grids){
        DdlaHandle_t ddla_handle;
        ddlaInit(ddla_handle);
        if(grid.first < 0){
            ddlaSet(ddla_handle);
        }else{
            ddlaSet(ddla_handle, MPI_COMM_WORLD, grid.first, grid.second);
        }

        int myid = ddla_handle->myid;
        if(myid == 0){
            std::cout << "========================================================" << std::endl;
            std::cout << "LibDDLA pzgemm benchmark (complex<double>)" << std::endl;
            std::cout << "Grid: " << ddla_handle->nprows_ << "x" << ddla_handle->npcols_
                      << "  (" << nprocs << " processes)" << std::endl;
            std::cout << "========================================================" << std::endl;
        }

        int nb = 128;

        for(int N : sizes){
            if(myid == 0){
                std::cout << "\n--- N = " << N << " (nb=" << nb << ") ---" << std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);

            // Fewer iterations for large sizes (they take longer)
            int n_warmup = (N >= 10000) ? 1 : 2;
            int n_iter   = (N >= 10000) ? 2 : 3;

            for(const auto& [ta, tb] : trans_pairs){
                try {
                    bench_one(ta, tb, N, nb, ddla_handle, n_warmup, n_iter);
                } catch(const std::exception& e) {
                    if(myid == 0){
                        std::cerr << "  pgemm(ddla_handle, " << ta << "," << tb << ") N=" << N
                                  << " FAILED: " << e.what() << std::endl;
                    }
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }

        ddlaDestroy(ddla_handle);
    }

    if(MPI_COMM_NULL != MPI_COMM_WORLD){
        int myid;
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        if(myid == 0){
            std::cout << "\n========================================================" << std::endl;
            std::cout << "Benchmark complete." << std::endl;
            std::cout << "========================================================" << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
