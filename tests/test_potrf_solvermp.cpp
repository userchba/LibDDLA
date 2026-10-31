#include "ddla_connector.h"
#include <cassert>
#include "test_desc_helpers.h"
#include <cmath>
#include <mpi.h>
#include <time.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <complex>
#include <string>
#include <ddla.h>
#include <ddla_connector.h>
#include <random>
#include "ddla_stream.h"
#include <cusolverMp.h>
#include "helpers.h"
using namespace ddla;

void check_ppotrf(int n, const DdlaHandle_t& ddla_handle)
{

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
    DDLA_CHECK(ddlaMalloc(ddla_handle, (void**)&d_A, size));
    DDLA_CHECK(ddlaMalloc(ddla_handle, (void**)&d_A_copy, size));
    
    DDLA_CHECK(ddlaSynchronize(ddla_handle));
    ddla_handle->check_memory();
    MPI_Barrier(MPI_COMM_WORLD);
    random_generate(ddla_handle, d_A, ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc));
    std::complex<double> ten = 1000.0;
    for(int i=0;i<matrix_desc[DDLA_M_];i++){
        int i_loc = indx_g2l_r(matrix_desc, ddla_handle, i);
        if(i_loc<0) continue;
        int j_loc = indx_g2l_c(matrix_desc, ddla_handle, i);
        if(j_loc<0) continue;
        DDLA_CHECK(ddlaMemcpy(ddla_handle, d_A+i_loc+j_loc*matrix_desc[DDLA_LLD_], &ten,
                              sizeof(std::complex<double>)));
    }
    DDLA_CHECK(ddlaSynchronize(ddla_handle));
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time_geadd = MPI_Wtime();
    pgeadd(ddla_handle, 
        'C', 'N',
        n, n,
        {1.0, 0.0},
        d_A, 1, 1, matrix_desc,
        {1.0, 0.0},
        d_A, 1, 1, matrix_desc,
        d_A_copy, 1, 1, matrix_desc
    );
    DDLA_CHECK(ddlaSynchronize(ddla_handle));
    printf("myid:%d, pgeadd time:%lf\n", ddla_handle->myid, MPI_Wtime() - start_time_geadd);
    DDLA_CHECK(ddlaMemcpy(ddla_handle, d_A, d_A_copy, size));
    std::complex<double>* h_A = NULL;
    if (ddla_handle->myid == 0)
    {
        h_A = (std::complex<double>*)malloc(matrix_desc[DDLA_M_] * matrix_desc[DDLA_N_] * sizeof(std::complex<double>));
        memset(h_A, 0xFF, matrix_desc[DDLA_M_] * matrix_desc[DDLA_N_] * sizeof(std::complex<double>));
        generate_diagonal_dominant_symmetric_matrix(matrix_desc[DDLA_M_], h_A, matrix_desc[DDLA_N_]);
    }

    if(verbose)
    {
        std::vector<std::complex<double>> a(ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc));
        DDLA_CHECK(ddlaMemcpy(ddla_handle, a.data(), d_A,
                              sizeof(std::complex<double>)*ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc)));
        DDLA_CHECK(ddlaSynchronize(ddla_handle));
        std::string filename = "before_potrf_myid_";
        filename += std::to_string(ddla_handle->myid);
        filename += ".txt";
        write_matrix<DdlaBackend::CPU>(a.data(), ddla_test::m_loc(ddla_handle, matrix_desc), ddla_test::n_loc(ddla_handle, matrix_desc), filename.c_str());
    }
    
    // DDLA_CHECK(ddlaMemcpy(ddla_handle, d_A_copy, d_A, size));
    DDLA_CHECK(ddlaSynchronize(ddla_handle));
    
    int * d_info;
    DDLA_CHECK(ddlaMalloc(ddla_handle, (void**)&d_info, sizeof(int)));
    
    if(verbose)
    { 
        std::vector<std::complex<double>> a(ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc));
        DDLA_CHECK(ddlaMemcpy(ddla_handle, a.data(), d_A_copy,
                              sizeof(std::complex<double>)*ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc)));
        DDLA_CHECK(ddlaSynchronize(ddla_handle));
        std::string filename = "ppotrf_myid_";
        filename += std::to_string(ddla_handle->myid);
        filename += ".txt";
        write_matrix<DdlaBackend::CPU>(a.data(), ddla_test::m_loc(ddla_handle, matrix_desc), ddla_test::n_loc(ddla_handle, matrix_desc), filename.c_str());
    }
    DDLA_CHECK(ddlaSynchronize(ddla_handle));
    {
        cusolverMpHandle_t cusolverMpHandle = NULL;
        cal_comm_t         cal_comm = NULL;

        cusolverStatus_t cusolverStat;
        calError_t       calStat;

        cal_comm_create_params_t params;
        params.allgather    = allgather;
        params.req_test     = request_test;
        params.req_free     = request_free;
        params.data         = (void*)(MPI_COMM_WORLD);
        params.rank         = ddla_handle->myid;
        params.nranks       = ddla_handle->nprocs;
        params.local_device = ddla_handle->local_device;

        calStat = cal_comm_create(params, &cal_comm);
        assert(calStat == CAL_OK);

        SOLVER_CHECK(cusolverMpCreate(&cusolverMpHandle, ddla_handle->local_device, static_cast<gpuStream_t>(ddla_handle->stream)));

        cusolverMpGrid_t grid = NULL;
        cusolverMpMatrixDescriptor_t descr = NULL;
        void* d_work = NULL;
        void* h_work = NULL;
        size_t workspaceInBytesOnDevice = 0;
        size_t workspaceInBytesOnHost = 0;

        cusolverStat = cusolverMpCreateDeviceGrid(
            cusolverMpHandle, &grid, cal_comm, ddla_test::nprows(ddla_handle), ddla_test::npcols(ddla_handle), CUSOLVERMP_GRID_MAPPING_ROW_MAJOR);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* Create matrix descriptor */
        cusolverStat = cusolverMpCreateMatrixDesc(
            &descr, grid, CUDA_C_64F, matrix_desc[DDLA_M_], matrix_desc[DDLA_N_], matrix_desc[DDLA_MB_], matrix_desc[DDLA_NB_], matrix_desc[DDLA_RSRC_], matrix_desc[DDLA_CSRC_], matrix_desc[DDLA_LLD_]);
        printf("myid:%d,m:%d,n:%d,mb:%d,nb:%d,irsrc:%d,icsrc:%d,lld:%d\n",ddla_handle->myid,matrix_desc[DDLA_M_],matrix_desc[DDLA_N_],matrix_desc[DDLA_MB_],matrix_desc[DDLA_NB_],matrix_desc[DDLA_RSRC_],matrix_desc[DDLA_CSRC_],matrix_desc[DDLA_LLD_]);
        
        // SOLVER_CHECK(cusolverMpMatrixScatterH2D(
        //     cusolverMpHandle, matrix_desc[DDLA_M_], matrix_desc[DDLA_N_], 
        //     (void*)d_A_copy, 1, 1, descr,
        //     0, (void*)h_A, matrix_desc[DDLA_M_]
        // ));
        cusolverStat = cusolverMpPotrf_bufferSize(
            cusolverMpHandle, GPUBLAS_FILL_MODE_LOWER, matrix_desc[DDLA_M_],
            d_A_copy, 1, 1, descr, CUDA_C_64F,
            &workspaceInBytesOnDevice, &workspaceInBytesOnHost
        );
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        printf("getrf rank:%d, workspaceInBytesOnDevice:%lf GiB, workspaceInBytesOnHost:%lf GiB\n",ddla_handle->myid,workspaceInBytesOnDevice/(1024.*1024.*1024.),workspaceInBytesOnHost/(1024.*1024.*1024.));

        if(workspaceInBytesOnDevice > 0)
            GPU_CHECK(cudaMalloc((void**)&d_work, workspaceInBytesOnDevice));
        
        h_work = (void*)malloc(workspaceInBytesOnHost);
        assert(h_work != NULL);

        calStat = cal_stream_sync(cal_comm, static_cast<gpuStream_t>(ddla_handle->stream));
        assert(calStat == CAL_OK);
        DDLA_CHECK(ddlaMemcpy(ddla_handle, d_A, d_A_copy,
                              ddla_test::m_loc(ddla_handle, matrix_desc) * ddla_test::n_loc(ddla_handle, matrix_desc) * sizeof(std::complex<double>)));
        DDLA_CHECK(ddlaSynchronize(ddla_handle));
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time=MPI_Wtime();
        cusolverStat = cusolverMpPotrf(
            cusolverMpHandle, GPUBLAS_FILL_MODE_LOWER, matrix_desc[DDLA_M_],
            d_A_copy, 1, 1, descr, CUDA_C_64F,
            d_work, workspaceInBytesOnDevice,
            h_work, workspaceInBytesOnHost,
            d_info
        );
        // ddla::potrf(ddla_handle, 'L', matrix_desc[DDLA_M_], d_A_copy, matrix_desc[DDLA_LLD_], d_info);
        if(cusolverStat != CUSOLVER_STATUS_SUCCESS){
            printf("myid:%d, cusolverStat:%d\n",ddla_handle->myid,cusolverStat);
        }
        DDLA_CHECK(ddlaSynchronize(ddla_handle));
        printf("myid:%d, cusolverMpPotrf time:%lf\n", ddla_handle->myid, MPI_Wtime()-start_time);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        MPI_Barrier(MPI_COMM_WORLD);
        printf("myid:%d, start ppotrf\n",ddla_handle->myid);
        double start_time_ppotrf = MPI_Wtime();
        ppotrf(ddla_handle, 
            'L', n,
            d_A, 1, 1, matrix_desc,
            d_info
        );
        printf("1234\n");
        DDLA_CHECK(ddlaSynchronize(ddla_handle));
        printf("myid:%d, ppotrf time:%lf\n",ddla_handle->myid,MPI_Wtime()-start_time_ppotrf);
        MPI_Barrier(MPI_COMM_WORLD);

        SOLVER_CHECK(cusolverMpDestroy(cusolverMpHandle));
        SOLVER_CHECK(cusolverMpDestroyMatrixDesc(descr));
        SOLVER_CHECK(cusolverMpDestroyGrid(grid));

        calStat = cal_comm_barrier(cal_comm, static_cast<gpuStream_t>(ddla_handle->stream));
        assert(calStat == CAL_OK);

        /* destroy CAL communicator */
        calStat = cal_comm_destroy(cal_comm);
        assert(calStat == CAL_OK);

        if (d_work != NULL)
        {
            DDLA_CHECK(ddlaFree(ddla_handle, d_work));
            d_work = NULL;
        }
        if (h_work != NULL) 
        {
            free(h_work);
            h_work = NULL;
        } 
    }
    DDLA_CHECK(ddlaSynchronize(ddla_handle));
    if(verbose)
    {
        std::vector<std::complex<double>> a(ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc));
        DDLA_CHECK(ddlaMemcpy(ddla_handle, a.data(), d_A_copy,
                              sizeof(std::complex<double>)*ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc)));
        DDLA_CHECK(ddlaSynchronize(ddla_handle));
        std::string filename = "potrf_myid_";
        filename += std::to_string(ddla_handle->myid);
        filename += ".txt";
        write_matrix<DdlaBackend::CPU>(a.data(), ddla_test::m_loc(ddla_handle, matrix_desc), ddla_test::n_loc(ddla_handle, matrix_desc), filename.c_str());
    }
    {
        printf("myid:%d, start check potrf\n", ddla_handle->myid);
        std::vector<std::complex<double>> a(ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc));
        std::vector<std::complex<double>> b(ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc));
        DDLA_CHECK(ddlaMemcpy(ddla_handle, a.data(), d_A,
                              sizeof(std::complex<double>)*ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc)));
        DDLA_CHECK(ddlaMemcpy(ddla_handle, b.data(), d_A_copy,
                              sizeof(std::complex<double>)*ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc)));
        DDLA_CHECK(ddlaSynchronize(ddla_handle));
        for(int i=0;i<matrix_desc[DDLA_M_];i++){
            int i_loc = indx_g2l_r(matrix_desc, ddla_handle, i);
            if(i_loc < 0) continue;
            for(int j = 0; j < matrix_desc[DDLA_N_]; j++){
                int j_loc = indx_g2l_c(matrix_desc, ddla_handle, j);
                if(j_loc < 0) continue;
                auto diff = a[i_loc + j_loc * matrix_desc[DDLA_LLD_]]-b[i_loc + j_loc * matrix_desc[DDLA_LLD_]];
                if(std::abs(diff)>1e-10){
                    printf("myid:%d, i:%d, j:%d, diff:(%lf,%lf), b:%lf\n",ddla_handle->myid,i,j,diff.real(),diff.imag(),1e-6);
                    break;
                }
            }
        }
        printf("myid:%d, check potrf pass\n", ddla_handle->myid);
    }
    if(h_A){
        free(h_A);
        h_A = nullptr;
    }
    DDLA_CHECK(ddlaFree(ddla_handle, d_A_copy));
    DDLA_CHECK(ddlaFree(ddla_handle, d_A));
    DDLA_CHECK(ddlaFree(ddla_handle, d_info));

    
}
int main(int argc, char* argv[]) {  
    MPI_Init(&argc, &argv);
    printf("before stream init\n");
    DdlaHandle_t ddla_handle = nullptr;
    ddlaInit(ddla_handle);
    ddlaSet(ddla_handle, MPI_COMM_WORLD, 'R');
    // DDLA_CHECK(ddlaSynchronize(ddla_handle));
    printf("after stream init\n");
    check_ppotrf(5000, ddla_handle);
    for(int i = 1000; i <= 1000 + 10 * 100; i += 100){
        DDLA_CHECK(ddlaSynchronize(ddla_handle));
        MPI_Barrier(MPI_COMM_WORLD);
        printf("testing matrix size: %d\n",i);
        check_ppotrf(i,ddla_handle);
    }
    ddlaDestroy(ddla_handle);
    MPI_Finalize();
    return 0;
}