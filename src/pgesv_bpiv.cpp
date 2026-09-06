#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <cassert>
#include <stdexcept>
#include <string>
#include "ddla_stream_impl.h"
#include "require_gpu.h"

namespace ddla {

/**
 * @brief Distributed linear-system solver using block-partial-pivoting LU
 *        (driver): solve op(A) * X = B (side='L') or X * op(A) = B
 *        (side='R').
 *
 * Convenience wrapper: pgetrf_bpiv (block LU with partial pivoting within
 * each block row) + pgetrs_bpiv (solve).  Corresponds to the same
 * ScaLAPACK-style interface as pgesv, but with block-partial pivoting.
 *
 * @tparam T   Scalar type.
 * @param side    'L' -- solve op(A)*X = B (B is n x nrhs);
 *                'R' -- solve X*op(A) = B (B is nrhs x n).
 * @param trans   'N', 'T' or 'C' -- operation applied to A.
 * @param n       Order of square matrix A.
 * @param nrhs    Number of right-hand sides.
 * @param d_A     Device pointer to A (input: coefficient; output: LU factors).
 * @param array_descA  ScaLAPACK int[9] descriptor for A.
 * @param d_B     Device pointer to RHS / solution B (input/output).
 * @param array_descB  ScaLAPACK int[9] descriptor for B.
 * @throws std::runtime_error if LU factorization fails (info != 0).
 */
template <typename T>
void pgesv_bpiv(
    const DdlaHandle_t& handle, const char& side, const char& trans, const int& n, const int& nrhs,
    T* d_A, const int* array_descA,
    T* d_B, const int* array_descB
)
{
    check_desc(array_descB, handle);
    check_desc(array_descA, handle);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);


    DdlaHandle_t ddla_handle = handle;
    detail::require_gpu_backend(ddla_handle, "pgesv_bpiv");
    assert(n <= array_descA[DDLA_M_] && n <= array_descA[DDLA_N_]);
    const int m_loc_A = num_loc(n, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
    int* d_ipiv = nullptr;
    RUNTIME_CHECK(runtimeMallocAsync(reinterpret_cast<void**>(&d_ipiv),
                                     sizeof(int) * std::max(1, m_loc_A),
                                     ddla_handle->stream));
    int info = 0;
    pgetrf_bpiv(handle, n, n, d_A, array_descA, d_ipiv, info);
    if(info != 0){
        RUNTIME_CHECK(runtimeFreeAsync(d_ipiv, ddla_handle->stream));
        throw std::runtime_error("pgesv_bpiv: pgetrf_bpiv returned info = " + std::to_string(info));
    }
    pgetrs_bpiv(handle, side, trans, n, nrhs, d_A, array_descA, d_ipiv, d_B, array_descB);
    RUNTIME_CHECK(runtimeFreeAsync(d_ipiv, ddla_handle->stream));
}

template void pgesv_bpiv<float>(
    const DdlaHandle_t&, const char& side, const char& trans, const int& n, const int& nrhs,
    float* d_A, const int* array_descA,
    float* d_B, const int* array_descB
);
template void pgesv_bpiv<double>(
    const DdlaHandle_t&, const char& side, const char& trans, const int& n, const int& nrhs,
    double* d_A, const int* array_descA,
    double* d_B, const int* array_descB
);
template void pgesv_bpiv<std::complex<float>>(
    const DdlaHandle_t&, const char& side, const char& trans, const int& n, const int& nrhs,
    std::complex<float>* d_A, const int* array_descA,
    std::complex<float>* d_B, const int* array_descB
);
template void pgesv_bpiv<std::complex<double>>(
    const DdlaHandle_t&, const char& side, const char& trans, const int& n, const int& nrhs,
    std::complex<double>* d_A, const int* array_descA,
    std::complex<double>* d_B, const int* array_descB
);

} // namespace ddla
