#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <cassert>
#include "ddla_stream_impl.h"
#include "require_gpu.h"

namespace ddla {

/**
 * @brief Distributed linear-system solver without pivoting (driver).
 *
 * Convenience wrapper: pgetrf_nopiv (LU) + pgetrs_nopiv (solve).
 * Solves A * X = B without pivoting.
 *
 * @tparam T   Scalar type.
 * @param n       Order of square matrix A.
 * @param nrhs    Number of right-hand sides.
 * @param d_A     Device pointer to A (input: coefficient; output: LU factors).
 * @param array_descA  ScaLAPACK int[9] descriptor for A.
 * @param d_B     Device pointer to RHS / solution B (input/output).
 * @param array_descB  ScaLAPACK int[9] descriptor for B.
 * @throws std::runtime_error if LU factorization fails (info != 0).
 */
template <typename T>
void pgesv_nopiv(
    const DdlaHandle_t& handle, const char& side, const char& trans, const int& n, const int& nrhs,
    T* d_A, const int* array_descA,
    T* d_B, const int* array_descB
)
{
    check_desc(array_descB, handle);
    check_desc(array_descA, handle);
    DdlaHandle_t ddla_handle = handle;
    detail::require_gpu_backend(ddla_handle, "pgesv_nopiv");
    assert(n <= array_descA[DDLA_M_] && n <= array_descA[DDLA_N_]);
    int info = 1;
    pgetrf_nopiv(handle, n, n, d_A, array_descA, info);
    if (info != 0) {
        throw std::runtime_error("pgesv_nopiv: pgetrf_nopiv returned info = " + std::to_string(info));
    }
    pgetrs_nopiv(handle, side, trans, n, nrhs, d_A, array_descA, d_B, array_descB);
}

template void pgesv_nopiv<float>(
    const DdlaHandle_t&, const char& side, const char& trans, const int& n, const int& nrhs,
    float* d_A, const int* array_descA,
    float* d_B, const int* array_descB
);
template void pgesv_nopiv<double>(
    const DdlaHandle_t&, const char& side, const char& trans, const int& n, const int& nrhs,
    double* d_A, const int* array_descA,
    double* d_B, const int* array_descB
);
template void pgesv_nopiv<std::complex<float>>(
    const DdlaHandle_t&, const char& side, const char& trans, const int& n, const int& nrhs,
    std::complex<float>* d_A, const int* array_descA,
    std::complex<float>* d_B, const int* array_descB
);
template void pgesv_nopiv<std::complex<double>>(
    const DdlaHandle_t&, const char& side, const char& trans, const int& n, const int& nrhs,
    std::complex<double>* d_A, const int* array_descA,
    std::complex<double>* d_B, const int* array_descB
);

} // namespace ddla
