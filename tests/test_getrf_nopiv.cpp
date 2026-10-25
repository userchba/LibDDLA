#include <cmath>
#include "LU/lu_internal.h"
#include <complex>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <vector>

#include <ddla/ddla.h>
#include "ddla_connector.h"
#include "ddla_stream_impl.h"

using namespace ddla;

// Sum of log(|diag|) of the in-place LU factors.  For an LU decomposition
// (with or without pivoting) the absolute value of the determinant is the
// product of the absolute values of the diagonal entries of U.
template <typename T>
double log_abs_det(const std::vector<T>& a, int n, int lda)
{
    double logdet = 0.0;
    for (int i = 0; i < n; ++i) {
        logdet += std::log(std::abs(a[i + i * lda]));
    }
    return logdet;
}

void check_getrf_nopiv(int n, const DdlaHandle_t& ddla_handle)
{
    const int lda = n;
    const size_t nelem = static_cast<size_t>(n) * n;
    const size_t bytes = nelem * sizeof(std::complex<double>);

    std::complex<double> *d_A, *d_A_ref;
    int *d_info, *d_info_ref, *d_ipiv;

    DDLA_CHECK(ddlaMalloc(ddla_handle, (void**)&d_A, bytes));
    DDLA_CHECK(ddlaMalloc(ddla_handle, (void**)&d_A_ref, bytes));
    DDLA_CHECK(ddlaMalloc(ddla_handle, (void**)&d_info, sizeof(int)));
    DDLA_CHECK(ddlaMalloc(ddla_handle, (void**)&d_info_ref, sizeof(int)));
    DDLA_CHECK(ddlaMalloc(ddla_handle, (void**)&d_ipiv, n * sizeof(int)));

    // Fill with random values on device, then copy to the reference buffer.
    random_generate(ddla_handle, d_A, nelem);
    ddla::scal(ddla_handle, n * n, std::complex<double>(0.01, 0.0), d_A, 1);
    std::complex<double> cons_i = 2.0;
    for(int i = 0; i < n; i++){
        DDLA_CHECK(ddlaMemcpy(ddla_handle, d_A + i + i * n, &cons_i, sizeof(std::complex<double>)));
    }
    DDLA_CHECK(ddlaMemcpy(ddla_handle, d_A_ref, d_A, bytes));
    DDLA_CHECK(ddlaMemset(ddla_handle, d_info, 0, sizeof(int)));
    DDLA_CHECK(ddlaMemset(ddla_handle, d_info_ref, 0, sizeof(int)));

    // Run the local nopiv LU factorization.
    DDLA_CHECK(ddlaSynchronize(ddla_handle));
    double t_nopiv = MPI_Wtime();
    getrf_nopiv(n, n, d_A, lda, d_info, ddla_handle);
    DDLA_CHECK(ddlaSynchronize(ddla_handle));
    t_nopiv = MPI_Wtime() - t_nopiv;

    // Run cuSOLVER/hipSOLVER LU with partial pivoting as the reference.
    double t_ref = MPI_Wtime();
    ddla::getrf(ddla_handle, n, n, d_A_ref, lda, d_ipiv, d_info_ref);
    DDLA_CHECK(ddlaSynchronize(ddla_handle));
    t_ref = MPI_Wtime() - t_ref;

    // Copy results back to host.
    std::vector<std::complex<double>> h_A(nelem);
    std::vector<std::complex<double>> h_A_ref(nelem);
    DDLA_CHECK(ddlaMemcpy(ddla_handle, h_A.data(), d_A, bytes));
    DDLA_CHECK(ddlaMemcpy(ddla_handle, h_A_ref.data(), d_A_ref, bytes));
    DDLA_CHECK(ddlaSynchronize(ddla_handle));

    int info = 0, info_ref = 0;
    DDLA_CHECK(ddlaMemcpy(ddla_handle, &info, d_info, sizeof(int)));
    DDLA_CHECK(ddlaSynchronize(ddla_handle));
    DDLA_CHECK(ddlaMemcpy(ddla_handle, &info_ref, d_info_ref, sizeof(int)));
    DDLA_CHECK(ddlaSynchronize(ddla_handle));

    double logdet_nopiv = log_abs_det(h_A, n, lda);
    double logdet_ref   = log_abs_det(h_A_ref, n, lda);
    double diff = std::abs(logdet_nopiv - logdet_ref);

    std::cout << "n=" << n
              << "  getrf_nopiv time=" << t_nopiv
              << "  reference LU time=" << t_ref
              << "  info=" << info
              << "  info_ref=" << info_ref
              << "  log|det|_nopiv=" << logdet_nopiv
              << "  log|det|_ref=" << logdet_ref
              << "  diff=" << diff
              << std::endl;

    if (info != 0 || info_ref != 0) {
        std::cerr << "Error: non-zero info (nopiv=" << info << ", ref=" << info_ref << ")" << std::endl;
        std::exit(1);
    }
    if (diff > 1e-6) {
        std::cerr << "Error: log-determinant mismatch" << std::endl;
        std::exit(1);
    }

    DDLA_CHECK(ddlaFree(ddla_handle, d_A));
    DDLA_CHECK(ddlaFree(ddla_handle, d_A_ref));
    DDLA_CHECK(ddlaFree(ddla_handle, d_info));
    DDLA_CHECK(ddlaFree(ddla_handle, d_info_ref));
    DDLA_CHECK(ddlaFree(ddla_handle, d_ipiv));
    DDLA_CHECK(ddlaSynchronize(ddla_handle));
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    DdlaHandle_t ddla_handle = nullptr;
    ddlaInit(ddla_handle);
    ddlaSet(ddla_handle, MPI_COMM_WORLD);
    check_getrf_nopiv(100, ddla_handle);
    std::cout << "Testing getrf_nopiv against cuSOLVER/hipSOLVER LU" << std::endl;

    for (int n = 5000; n <= 20000; n += 5000) {
        DDLA_CHECK(ddlaSynchronize(ddla_handle));
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout << "Testing matrix size: " << n << std::endl;
        check_getrf_nopiv(n, ddla_handle);
    }

    ddlaDestroy(ddla_handle);
    MPI_Finalize();
    return 0;
}
