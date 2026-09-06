#include <cassert>
#include <cmath>
#include <mpi.h>
#include <time.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <complex>
#include <string>
#include <ddla/ddla.h>
#include "test_desc_helpers.h"
#include "ddla_connector.h"
#include "scal.h"
#include <random>
#include "ddla_stream_impl.h"

using namespace ddla;


void check_pzgetrf(int n, const DdlaHandle_t& ddla_handle)
{

    int matrix_desc[DDLA_DLEN_];
    ddla_test::init_square_blk(matrix_desc, n, n, 0, 0, ddla_handle);
    int nb = std::min(128, matrix_desc[DDLA_MB_]);
    DDLA_CHECK(ddlaDescInit(matrix_desc, ddla_handle, n, n, nb, nb, 0, 0));

    int myid = ddla_test::mypcol(ddla_handle) + ddla_test::myprow(ddla_handle)*ddla_test::npcols(ddla_handle);
    printf("myid:%d, m_loc:%d, n_loc:%d, mb:%d, nb:%d, m:%d, n:%d\n", myid, ddla_test::m_loc(ddla_handle, matrix_desc), ddla_test::n_loc(ddla_handle, matrix_desc), matrix_desc[DDLA_MB_], matrix_desc[DDLA_NB_], matrix_desc[DDLA_M_], matrix_desc[DDLA_N_]);
    bool verbose = false;

    std::complex<double> *d_A, *d_A_copy;
    int *d_ipiv;
    
    std::vector<int>ipiv(ddla_test::m_loc(ddla_handle, matrix_desc));
    ddla_handle->check_memory();
    MPI_Barrier(MPI_COMM_WORLD);

    const size_t size = ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc)*sizeof(std::complex<double>);

    RUNTIME_CHECK(runtimeMallocAsync((void**)&d_A, size, ddla_handle->stream));
    RUNTIME_CHECK(runtimeMallocAsync((void**)&d_A_copy, size, ddla_handle->stream));
    RUNTIME_CHECK(runtimeMallocAsync((void**)&d_ipiv, ddla_test::m_loc(ddla_handle, matrix_desc) * sizeof(int), ddla_handle->stream));
    
    RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
    ddla_handle->check_memory();
    MPI_Barrier(MPI_COMM_WORLD);
    random_generate(d_A, ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc));
    BLAS_CHECK(deblasScal(ddla_handle->blasH, ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc), 0.01, d_A, 1));
    std::complex<double> cons_i = 2.0;
    for(int i = 0; i < matrix_desc[DDLA_M_]; i++){
        int i_loc = indx_g2l_r(matrix_desc, ddla_handle, i);
        if(i_loc < 0 ) continue;
        int j_loc = indx_g2l_c(matrix_desc, ddla_handle, i);
        if(j_loc < 0 ) continue;
        RUNTIME_CHECK(runtimeMemcpy(d_A + i_loc + j_loc * matrix_desc[DDLA_LLD_], &cons_i, sizeof(std::complex<double>), runtimeMemcpyHostToDevice));
    }
    
    RUNTIME_CHECK(runtimeMemcpyAsync(d_A_copy, d_A, sizeof(std::complex<double>)*ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc), runtimeMemcpyDeviceToDevice, ddla_handle->stream));

    if(verbose)
    {
        std::vector<std::complex<double>> a(ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc));
        RUNTIME_CHECK(runtimeMemcpyAsync(a.data(), d_A, sizeof(std::complex<double>)*ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc), runtimeMemcpyDeviceToHost, ddla_handle->stream));
        RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
        std::string filename = "before_trf_myid_";
        filename += std::to_string(myid);
        filename += ".txt";
        write_matrix<DdlaBackend::CPU>(a.data(), ddla_test::m_loc(ddla_handle, matrix_desc), ddla_test::n_loc(ddla_handle, matrix_desc), filename.c_str());
    }
    RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(MPI_COMM_WORLD);
    printf("myid:%d, start sv\n",myid);
    int info = -1;
    double start_time_trf = MPI_Wtime();
    pgetrf(ddla_handle, n, n, d_A, matrix_desc, ipiv.data(), info);
    RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time_trf = MPI_Wtime();
    assert(info == 0);
    printf("myid:%d, start bpiv\n", myid);
    info = -1;
    pgetrf_bpiv(ddla_handle, n, n, d_A_copy, matrix_desc, d_ipiv, info);
    printf("myid:%d, piv of lu time:%lf, bpiv of lu time:%lf\n", myid, end_time_trf - start_time_trf, MPI_Wtime() - end_time_trf);
    assert(info == 0);
    if(verbose)
    {
        std::vector<std::complex<double>> a(ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc));
        RUNTIME_CHECK(runtimeMemcpyAsync(a.data(), d_A, sizeof(std::complex<double>)*ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc), runtimeMemcpyDeviceToHost, ddla_handle->stream));
        RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
        std::string filename = "identity_myid_";
        filename += std::to_string(myid);
        filename += ".txt";
        write_matrix<DdlaBackend::CPU>(a.data(), ddla_test::m_loc(ddla_handle, matrix_desc), ddla_test::n_loc(ddla_handle, matrix_desc), filename.c_str());
    }
    {
        std::vector<std::complex<double>> a(ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc));
        std::vector<std::complex<double>> b(ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc));
        RUNTIME_CHECK(runtimeMemcpyAsync(a.data(), d_A, sizeof(std::complex<double>)*ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc), runtimeMemcpyDeviceToHost, ddla_handle->stream));
        RUNTIME_CHECK(runtimeMemcpyAsync(b.data(), d_A_copy, sizeof(std::complex<double>)*ddla_test::m_loc(ddla_handle, matrix_desc)*ddla_test::n_loc(ddla_handle, matrix_desc), runtimeMemcpyDeviceToHost, ddla_handle->stream));
        printf("myid:%d, start check trf result\n", myid);
        int loc_pi;
        std::complex<double> tmp;
        std::complex<double> ln_det_loc_a = 0.0;
        std::complex<double> ln_det_all_a = 0.0;
        std::complex<double> ln_det_loc_b = 0.0;
        std::complex<double> ln_det_all_b = 0.0;
        for (int ig = 0; ig != n; ig++)
        {
            int locr = indx_g2l_r(matrix_desc, ddla_handle, ig);
            int locc = indx_g2l_c(matrix_desc, ddla_handle, ig);
            if (locr >= 0 && locc >= 0)
            {
                tmp = a[locr + locc * matrix_desc[DDLA_LLD_]];
                ln_det_loc_a += tmp.real() > 0 ? std::log(tmp) : std::log(-tmp);
                tmp = b[locr + locc * matrix_desc[DDLA_LLD_]];
                ln_det_loc_b += tmp.real() > 0 ? std::log(tmp) : std::log(-tmp);
            }
        }
        MPI_Allreduce(&ln_det_loc_a,&ln_det_all_a,1,MPI_DOUBLE_COMPLEX,MPI_SUM, ddla_handle->comm);
        MPI_Allreduce(&ln_det_loc_b,&ln_det_all_b,1,MPI_DOUBLE_COMPLEX,MPI_SUM, ddla_handle->comm);
        MPI_Barrier(ddla_handle->comm);
        printf("myid:%d, ln_det_a:%lf+i%lf, ln_det_b:%lf+i%lf\n", myid, ln_det_all_a.real(), ln_det_all_a.imag(), ln_det_all_b.real(), ln_det_all_b.imag());
    }
    RUNTIME_CHECK(runtimeFreeAsync(d_ipiv, ddla_handle->stream));
    RUNTIME_CHECK(runtimeFreeAsync(d_A, ddla_handle->stream));
    RUNTIME_CHECK(runtimeFreeAsync(d_A_copy, ddla_handle->stream));
    RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
}
int main(int argc, char* argv[]) {  
    MPI_Init(&argc, &argv);
    printf("before stream init\n");
    DdlaHandle_t ddla_handle = nullptr;
    ddlaInit(ddla_handle);
    ddlaSet(ddla_handle);
    // RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
    printf("after stream init\n");
    check_pzgetrf(5000, ddla_handle);
    for(int i = 5000; i <= 4 * 5000; i += 5000){
        RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
        MPI_Barrier(MPI_COMM_WORLD);
        printf("testing matrix size: %d\n",i);
        check_pzgetrf(i, ddla_handle);
    }
    ddlaDestroy(ddla_handle);
    MPI_Finalize();
    return 0;
}