#include <cassert>
#include <cmath>
#include <mpi.h>
#include <time.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <complex>
#include <ddla/ddla.h>
#include "test_desc_helpers.h"
#include "ddla_connector.h"
#include <random>
#include "ddla_stream_impl.h"

using namespace ddla;

namespace {

std::complex<double> op_value(char trans, const std::vector<std::complex<double>>& mat,
                              int m, int n, int i, int j)
{
    // mat is stored column-major, m rows x n cols
    if(trans == 'N'){
        return mat[i + j * m];
    }else if(trans == 'T'){
        return mat[j + i * m];
    }else{ // 'C'
        return std::conj(mat[j + i * m]);
    }
}

void check_pgemm(char transa, char transb,
                 int m, int n, int k, int nb,
                 const DdlaHandle_t& ddla_handle)
{
    int descA[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, ddla_handle, m, k, nb, nb, 0, 0));
    int descB[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descB, ddla_handle, k, n, nb, nb, 0, 0));
    int descC[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descC, ddla_handle, m, n, nb, nb, 0, 0));

    int myid = ddla_test::mypcol(ddla_handle) + ddla_test::myprow(ddla_handle) * ddla_test::npcols(ddla_handle);

    std::vector<std::complex<double>> h_A(ddla_test::m_loc(ddla_handle, descA) * ddla_test::n_loc(ddla_handle, descA));
    std::vector<std::complex<double>> h_B(ddla_test::m_loc(ddla_handle, descB) * ddla_test::n_loc(ddla_handle, descB));
    std::vector<std::complex<double>> h_C(ddla_test::m_loc(ddla_handle, descC) * ddla_test::n_loc(ddla_handle, descC));

    std::mt19937 gen(42 + myid);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for(auto& v : h_A) v = std::complex<double>(dist(gen), dist(gen));
    for(auto& v : h_B) v = std::complex<double>(dist(gen), dist(gen));
    for(auto& v : h_C) v = std::complex<double>(dist(gen), dist(gen));

    std::complex<double>* d_A;
    std::complex<double>* d_B;
    std::complex<double>* d_C;
    RUNTIME_CHECK(runtimeMalloc(&d_A, sizeof(std::complex<double>) * h_A.size()));
    RUNTIME_CHECK(runtimeMalloc(&d_B, sizeof(std::complex<double>) * h_B.size()));
    RUNTIME_CHECK(runtimeMalloc(&d_C, sizeof(std::complex<double>) * h_C.size()));

    RUNTIME_CHECK(runtimeMemcpy(d_A, h_A.data(), sizeof(std::complex<double>) * h_A.size(), runtimeMemcpyHostToDevice));
    RUNTIME_CHECK(runtimeMemcpy(d_B, h_B.data(), sizeof(std::complex<double>) * h_B.size(), runtimeMemcpyHostToDevice));
    RUNTIME_CHECK(runtimeMemcpy(d_C, h_C.data(), sizeof(std::complex<double>) * h_C.size(), runtimeMemcpyHostToDevice));

    std::complex<double> alpha(1.0, 0.0);
    std::complex<double> beta(0.0, 0.0);

    double start = MPI_Wtime();
    pgemm(ddla_handle, transa, transb, m, n, k, alpha, d_A, descA, d_B, descB, beta, d_C, descC);
    RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
    double elapsed = MPI_Wtime() - start;

    std::vector<std::complex<double>> h_C_out(ddla_test::m_loc(ddla_handle, descC) * ddla_test::n_loc(ddla_handle, descC));
    RUNTIME_CHECK(runtimeMemcpy(h_C_out.data(), d_C, sizeof(std::complex<double>) * h_C_out.size(), runtimeMemcpyDeviceToHost));

    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    auto gather_global = [&](const int* desc, const std::vector<std::complex<double>>& local)->std::vector<std::complex<double>>{
        int mg = desc[DDLA_M_];
        int ng = desc[DDLA_N_];
        std::vector<std::complex<double>> global(mg * ng);
        std::vector<int> recvcounts(nprocs);
        std::vector<int> displs(nprocs);
        int loc_size = local.size();
        MPI_Allgather(&loc_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        displs[0] = 0;
        for(int i=1;i<nprocs;i++) displs[i] = displs[i-1] + recvcounts[i-1];
        std::vector<std::complex<double>> all_local(displs[nprocs-1] + recvcounts[nprocs-1]);
        MPI_Allgatherv(local.data(), loc_size, MPI_C_DOUBLE_COMPLEX,
                       all_local.data(), recvcounts.data(), displs.data(), MPI_C_DOUBLE_COMPLEX,
                       MPI_COMM_WORLD);
        int npcols = ddla_test::npcols(ddla_handle);
        for(int src=0; src<nprocs; src++){
            int prow = src / npcols;
            int pcol = src % npcols;
            int offset = displs[src];
            int count = recvcounts[src];
            if(count == 0) continue;
            int m_loc = num_loc(mg, desc[DDLA_MB_], prow, desc[DDLA_RSRC_], ddla_test::nprows(ddla_handle));
            int n_loc = num_loc(ng, desc[DDLA_NB_], pcol, desc[DDLA_CSRC_], npcols);
            if(m_loc * n_loc != count) continue;
            for(int j_loc=0; j_loc<n_loc; j_loc++){
                int j_g = indxl2g(j_loc, desc[DDLA_NB_], pcol, desc[DDLA_CSRC_], npcols);
                for(int i_loc=0; i_loc<m_loc; i_loc++){
                    int i_g = indxl2g(i_loc, desc[DDLA_MB_], prow, desc[DDLA_RSRC_], ddla_test::nprows(ddla_handle));
                    global[i_g + j_g * mg] = all_local[offset + i_loc + j_loc * m_loc];
                }
            }
        }
        return global;
    };

    std::vector<std::complex<double>> g_A = gather_global(descA, h_A);
    std::vector<std::complex<double>> g_B = gather_global(descB, h_B);
    std::vector<std::complex<double>> g_C = gather_global(descC, h_C_out);

    int ma = (transa == 'N') ? m : k;
    int na = (transa == 'N') ? k : m;
    int mb_ = (transb == 'N') ? k : n;
    int nb_ = (transb == 'N') ? n : k;

    double max_err = 0.0;
    for(int j=0; j<n; j++){
        for(int i=0; i<m; i++){
            std::complex<double> ref(0.0, 0.0);
            for(int l=0; l<k; l++){
                ref += alpha * op_value(transa, g_A, ma, na, i, l) * op_value(transb, g_B, mb_, nb_, l, j);
            }
            std::complex<double> diff = g_C[i + j * m] - ref;
            max_err = std::max(max_err, std::abs(diff));
        }
    }

    double global_max_err;
    MPI_Reduce(&max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(myid == 0){
        std::cout << "pgemm(ddla_handle, " << transa << "," << transb << ") "
                  << m << "x" << n << "x" << k
                  << " grid " << ddla_test::nprows(ddla_handle) << "x" << ddla_test::npcols(ddla_handle)
                  << " time " << elapsed << "s"
                  << " max_err " << global_max_err << std::endl;
    }

    if(global_max_err > 1e-10){
        std::cerr << "FAIL: pgemm(ddla_handle, " << transa << "," << transb << ") error too large" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
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
        grids = {{1,4}, {2,2}, {4,1}};
    }else if(nprocs == 6){
        grids = {{2,3}, {3,2}};
    }else if(nprocs == 8){
        grids = {{2,4}, {4,2}};
    }else{
        grids = {{-1,-1}};
    }

    for(const auto& grid : grids){
        DdlaHandle_t ddla_handle;
        ddlaInit(ddla_handle);
        if(grid.first < 0){
            ddlaSet(ddla_handle);
        }else{
            ddlaSet(ddla_handle, MPI_COMM_WORLD, grid.first, grid.second);
        }

        int m = 100, n = 100, k = 100;
        int nb = 16;
        std::vector<char> trans_opts = {'N', 'T', 'C'};
        for(char transa : trans_opts){
            for(char transb : trans_opts){
                check_pgemm(transa, transb, m, n, k, nb, ddla_handle);
            }
        }

        ddlaDestroy(ddla_handle);
    }

    MPI_Finalize();
    return 0;
}
