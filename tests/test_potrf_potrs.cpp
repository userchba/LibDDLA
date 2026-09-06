#include <cassert>
#include <cmath>
#include <mpi.h>
#include <time.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <map>
#include <complex>
#include <string>
#include <ddla/ddla.h>
#include "test_desc_helpers.h"
#include "ddla_connector.h"
#include <random>
#include "ddla_stream_impl.h"
#include <fstream>

using namespace ddla;

const static std::string type_name = "cholesky_dcu";
static std::string filename = type_name;

void check_ppotrf(int n, const DdlaHandle_t& ddla_handle, bool is_write = false)
{
    std::ofstream outfile;
    if(is_write && ddla_handle->myid == 0)
        outfile.open(filename, std::ios::app);
    int matrix_desc[DDLA_DLEN_];
    ddla_test::init_square_blk(matrix_desc, n, n, 0, 0, ddla_handle);
    int nb = std::min(128, matrix_desc[DDLA_MB_]);
    DDLA_CHECK(ddlaDescInit(matrix_desc, ddla_handle, n, n, nb, nb, 0, 0));

    printf("myid:%d, m_loc:%d, n_loc:%d, mb:%d, nb:%d, m:%d, n:%d\n", ddla_handle->myid, ddla_test::m_loc(ddla_handle, matrix_desc), ddla_test::n_loc(ddla_handle, matrix_desc), matrix_desc[DDLA_MB_], matrix_desc[DDLA_NB_], matrix_desc[DDLA_M_], matrix_desc[DDLA_N_]);
    bool verbose = false;

    std::complex<double>* d_A;
    std::complex<double>* d_A_copy;
    
    MPI_Barrier(MPI_COMM_WORLD);
    ddla_handle->check_memory();
    MPI_Barrier(MPI_COMM_WORLD);

    const size_t size = ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc)*sizeof(std::complex<double>);
    RUNTIME_CHECK(runtimeMallocAsync((void**)&d_A, size, ddla_handle->stream));
    RUNTIME_CHECK(runtimeMallocAsync(&d_A_copy, size, ddla_handle->stream));
    
    RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
    ddla_handle->check_memory();
    MPI_Barrier(MPI_COMM_WORLD);
    random_generate(d_A, ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc));
    std::complex<double> ten = 1000.0;
    for(int i=0;i<matrix_desc[DDLA_M_];i++){
        int i_loc = indx_g2l_r(matrix_desc, ddla_handle, i);
        if(i_loc<0) continue;
        int j_loc = indx_g2l_c(matrix_desc, ddla_handle, i);
        if(j_loc<0) continue;
        RUNTIME_CHECK(runtimeMemcpyAsync(d_A+i_loc+j_loc*matrix_desc[DDLA_LLD_], &ten, sizeof(std::complex<double>), runtimeMemcpyHostToDevice, ddla_handle->stream));
        std::complex<double> one = -1.0;
        if(i == n - 1)
        RUNTIME_CHECK(runtimeMemcpyAsync(d_A+i_loc+j_loc*matrix_desc[DDLA_LLD_], &one, sizeof(std::complex<double>), runtimeMemcpyHostToDevice, ddla_handle->stream));
    }

    RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time_geadd = MPI_Wtime();
    pgeadd(ddla_handle, 
        'C', 'N',
        n, n,
        {1.0, 0.0},
        d_A, matrix_desc,
        {1.0, 0.0},
        d_A, matrix_desc,
        d_A_copy, matrix_desc
    );
    RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
    printf("myid:%d, pgeadd time:%lf\n", ddla_handle->myid, MPI_Wtime() - start_time_geadd);

    RUNTIME_CHECK(runtimeMemcpyAsync(d_A, d_A_copy, size, runtimeMemcpyDeviceToDevice, ddla_handle->stream));
    
    RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
    
    int info;
    RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
    {
        if(verbose)
        { 
            std::vector<std::complex<double>> a(ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc));
            RUNTIME_CHECK(runtimeMemcpyAsync(a.data(), d_A_copy, sizeof(std::complex<double>)*ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc), runtimeMemcpyDeviceToHost, ddla_handle->stream));
            RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
            std::string filename = "ppotrf_myid_";
            filename += std::to_string(ddla_handle->myid);
            filename += ".txt";
            write_matrix<DdlaBackend::CPU>(a.data(), ddla_test::m_loc(ddla_handle, matrix_desc), ddla_test::n_loc(ddla_handle, matrix_desc), filename.c_str());
        }

        RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
        MPI_Barrier(MPI_COMM_WORLD);
        printf("myid:%d, start ppotrf\n",ddla_handle->myid);
        double start_time_ppotrf = MPI_Wtime();
        bool is_nega = ppotrf(ddla_handle, 
            'L', n,
            d_A, 1, 1, matrix_desc,
            info,
            true, -1
        );
        RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
        double time_for_cholesky = MPI_Wtime() - start_time_ppotrf;
        printf("myid:%d, ppotrf time:%lf, is_nega:%d, info:%d\n", ddla_handle->myid, time_for_cholesky, is_nega, info);
        outfile << n << " " << time_for_cholesky << std::endl;
        outfile.close();
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time_ppotrs = MPI_Wtime();
        ppotrs(ddla_handle, 
            'L', 'L', 'N',
            n, n,
            d_A, matrix_desc,
            d_A_copy, matrix_desc,
            is_nega
        );
        RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
        printf("myid:%d, ppotrs time:%lf\n", ddla_handle->myid, MPI_Wtime() - start_time_ppotrs);
    }
    if(verbose)
    {
        std::vector<std::complex<double>> a(ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc));
        RUNTIME_CHECK(runtimeMemcpyAsync(a.data(), d_A_copy, sizeof(std::complex<double>)*ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc), runtimeMemcpyDeviceToHost, ddla_handle->stream));
        RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
        std::string filename = "after_ppotrs_myid_";
        filename += std::to_string(ddla_handle->myid);
        filename += ".txt";
        write_matrix<DdlaBackend::CPU>(a.data(), ddla_test::m_loc(ddla_handle, matrix_desc), ddla_test::n_loc(ddla_handle, matrix_desc), filename.c_str());
    }
    {
        // check the data from scalapack and bcast
        printf("myid:%d, start check identity result for ppotrf and ppotrs\n", ddla_handle->myid);        
        std::vector<std::complex<double>> temp_bcast(ddla_test::m_loc(ddla_handle, matrix_desc) * ddla_test::n_loc(ddla_handle, matrix_desc));
        RUNTIME_CHECK(runtimeMemcpyAsync(temp_bcast.data(), d_A_copy, sizeof(std::complex<double>)*ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc), runtimeMemcpyDeviceToHost, ddla_handle->stream));
        RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
        for(int i=0;i<matrix_desc[DDLA_M_];i++){
            int i_loc = indx_g2l_r(matrix_desc, ddla_handle, i);
            if(i_loc<0)
                continue;
            for(int j=0;j<matrix_desc[DDLA_N_];j++){
                int j_loc = indx_g2l_c(matrix_desc, ddla_handle, j);
                if(j_loc<0)
                    continue;
                double diff_abs;
                if(i==j)
                    diff_abs = std::abs(1. - temp_bcast[i_loc+j_loc*ddla_test::m_loc(ddla_handle, matrix_desc)]);
                else
                    diff_abs = std::abs(temp_bcast[i_loc+j_loc*ddla_test::m_loc(ddla_handle, matrix_desc)]);
                if(diff_abs>1e-6){
                    printf("myid:%d, check failed at global index (%d,%d), identity value=(%lf,%lf)\n",
                        ddla_handle->myid, i, j,
                        temp_bcast[i_loc+j_loc*ddla_test::m_loc(ddla_handle, matrix_desc)].real(),temp_bcast[i_loc+j_loc*ddla_test::m_loc(ddla_handle, matrix_desc)].imag()
                    );
                    break;
                }
                
            }
        }
        printf("end check pztrtrs result between scalapack and bcast\n");
    }
    RUNTIME_CHECK(runtimeFreeAsync(d_A_copy, ddla_handle->stream));
    RUNTIME_CHECK(runtimeFreeAsync(d_A, ddla_handle->stream));
    RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
}
int main(int argc, char* argv[]){  
    MPI_Init(&argc, &argv);
    printf("before stream init\n");
    DdlaHandle_t ddla_handle = nullptr;
    ddlaInit(ddla_handle);
    ddlaSet(ddla_handle, MPI_COMM_WORLD, 'R');
    printf("after stream init\n");
    // std::ofstream outfile;
    // filename += "_" + std::to_string(ddla_handle->nprocs) + ".dat";
    // if(ddla_handle->myid == 0)
    //     outfile.open(filename.c_str(), std::ios::out | std::ios::trunc);
    // outfile << type_name << std::endl;
    // outfile.close();
    check_ppotrf(1000, ddla_handle);

    std::map<int, int> mpi_to_size;
    mpi_to_size[1] = 15000;
    mpi_to_size[4] = 20000;
    mpi_to_size[16] = 60000; 

    for(int i = 5000; i <= mpi_to_size.at(ddla_handle->nprocs);i += 5000){
        RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
        MPI_Barrier(MPI_COMM_WORLD);
        printf("testing matrix size: %d\n",i);
        check_ppotrf(i, ddla_handle, false);
    }
    ddlaDestroy(ddla_handle);
    MPI_Finalize();
    return 0;
}