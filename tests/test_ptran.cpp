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
#include "ptran.h"

using namespace ddla;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    DdlaHandle_t handle;
    ddlaInit(handle);
    ddlaSet(handle, MPI_COMM_WORLD, 2, 3);

    int m = 12, n = 12;
    int nb = 4;

    int descA[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descA, handle, m, n, nb, nb, 0, 0));
    int descAT[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descAT, handle, n, m, nb, nb, 0, 0));

    int loc_size_A = ddla_test::m_loc(handle, descA) * ddla_test::n_loc(handle, descA);
    int loc_size_AT = ddla_test::m_loc(handle, descAT) * ddla_test::n_loc(handle, descAT);

    std::vector<std::complex<double>> h_A(loc_size_A);
    for(int i=0;i<loc_size_A;i++){
        h_A[i] = std::complex<double>(i, -i);
    }

    std::complex<double>* d_A;
    std::complex<double>* d_AT;
    RUNTIME_CHECK(runtimeMalloc(&d_A, sizeof(std::complex<double>) * loc_size_A));
    RUNTIME_CHECK(runtimeMalloc(&d_AT, sizeof(std::complex<double>) * loc_size_AT));
    RUNTIME_CHECK(runtimeMemcpy(d_A, h_A.data(), sizeof(std::complex<double>) * loc_size_A, runtimeMemcpyHostToDevice));
    std::vector<std::complex<double>> h_zero_AT(loc_size_AT);
    RUNTIME_CHECK(runtimeMemcpy(d_AT, h_zero_AT.data(), sizeof(std::complex<double>) * loc_size_AT, runtimeMemcpyHostToDevice));

    ptran(handle, d_A, descA, d_AT, descAT);

    std::vector<std::complex<double>> h_AT(loc_size_AT);
    RUNTIME_CHECK(runtimeMemcpy(h_AT.data(), d_AT, sizeof(std::complex<double>) * loc_size_AT, runtimeMemcpyDeviceToHost));

    // Gather global A and AT on host
    auto gather = [&](const int* desc, const std::vector<std::complex<double>>& local){
        std::vector<int> recvcounts(nprocs), displs(nprocs);
        int sz = local.size();
        MPI_Allgather(&sz, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        displs[0] = 0;
        for(int i=1;i<nprocs;i++) displs[i] = displs[i-1] + recvcounts[i-1];
        std::vector<std::complex<double>> all(displs[nprocs-1] + recvcounts[nprocs-1]);
        MPI_Allgatherv(local.data(), sz, MPI_C_DOUBLE_COMPLEX,
                       all.data(), recvcounts.data(), displs.data(), MPI_C_DOUBLE_COMPLEX,
                       MPI_COMM_WORLD);
        std::vector<std::complex<double>> global(desc[DDLA_M_] * desc[DDLA_N_]);
        int npcols = ddla_test::npcols(handle);
        for(int src=0; src<nprocs; src++){
            int prow = src / npcols;
            int pcol = src % npcols;
            int off = displs[src];
            int ml = num_loc(desc[DDLA_M_], desc[DDLA_MB_], prow, desc[DDLA_RSRC_], ddla_test::nprows(handle));
            int nl = num_loc(desc[DDLA_N_], desc[DDLA_NB_], pcol, desc[DDLA_CSRC_], npcols);
            if(ml * nl != recvcounts[src]){
                std::cerr << "size mismatch src=" << src << " ml*nl=" << ml*nl << " recv=" << recvcounts[src] << std::endl;
                continue;
            }
            for(int j=0;j<nl;j++){
                int gj = indxl2g(j, desc[DDLA_NB_], pcol, desc[DDLA_CSRC_], npcols);
                for(int i=0;i<ml;i++){
                    int gi = indxl2g(i, desc[DDLA_MB_], prow, desc[DDLA_RSRC_], ddla_test::nprows(handle));
                    global[gi + gj * desc[DDLA_M_]] = all[off + i + j * ml];
                }
            }
        }
        return global;
    };

    auto g_A = gather(descA, h_A);
    auto g_AT = gather(descAT, h_AT);

    double max_err = 0;
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++){
            std::complex<double> ref = g_A[j + i * n]; // A^T[i,j] = A[j,i]
            std::complex<double> diff = g_AT[i + j * m] - ref;
            max_err = std::max(max_err, std::abs(diff));
        }
    }

    double global_max;
    MPI_Reduce(&max_err, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(ddla_test::myprow(handle) == 0 && ddla_test::mypcol(handle) == 0){
        std::cout << "ptran max_err " << global_max << std::endl;
    }

    RUNTIME_CHECK(runtimeFree(d_A));
    RUNTIME_CHECK(runtimeFree(d_AT));
    ddlaDestroy(handle);
    MPI_Finalize();
    return 0;
}
