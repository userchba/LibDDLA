#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <cassert>
#include "ddla_stream_impl.h"
#include "require_gpu.h"

namespace ddla {

/**
 * @brief Distributed LU solve without pivoting.
 *
 * Solves A * X = B using the factors from pgetrf_nopiv.  Because no
 * pivoting was performed, the solve is simply two triangular solves:
 *   1) L * Y = B  (forward, lower triangular, unit diagonal)
 *   2) U * X = Y  (backward, upper triangular, non-unit diagonal)
 *
 * Only trans='N' is supported.
 *
 * @tparam T   Scalar type.
 * @param trans   'N' -- no transpose (only 'N' supported).
 * @param n       Order of matrix A.
 * @param nrhs    Number of right-hand sides.
 * @param d_A     Device pointer to LU factors (from pgetrf_nopiv).
 * @param array_descA  ScaLAPACK int[9] descriptor for A.
 * @param d_B     Device pointer to RHS / solution B (input/output).
 * @param array_descB  ScaLAPACK int[9] descriptor for B.
 */
template<typename T>
void pgetrs_nopiv(
    const DdlaHandle_t& handle, const char& side, const char& trans, const int& n, const int& nrhs,
    T* d_A, const int* array_descA,
    T* d_B, const int* array_descB
)
{
    check_desc(array_descB, handle);
    check_desc(array_descA, handle);
    DdlaHandle_t ddla_handle = handle;
    detail::require_gpu_backend(ddla_handle, "pgetrs_nopiv");
    assert(side == 'L' || side == 'R');
    assert(trans == 'N' || trans == 'T' || trans == 'C');
    // Leading-block sub-matrix: descriptors may be larger than the logical
    // sub-matrix (anchored at global (0,0)).
    if(side == 'L'){
        assert(n <= array_descA[DDLA_M_] && n <= array_descA[DDLA_N_]);
        assert(n <= array_descB[DDLA_M_]);
    }else{
        assert(n <= array_descA[DDLA_M_] && n <= array_descA[DDLA_N_]);
        assert(n <= array_descB[DDLA_N_]);
    }
    const int b_rows = (side == 'L') ? n : nrhs;
    const int b_cols = (side == 'L') ? nrhs : n;

    if(side == 'L'){
        if(trans == 'N'){
            // A = L*U => X = U^-1 * L^-1 * B: solve L then U.
            ptrtrs(handle, 'L', 'L', 'N', 'U', b_rows, b_cols,
                   d_A, array_descA, d_B, array_descB);
            ptrtrs(handle, 'L', 'U', 'N', 'N', b_rows, b_cols,
                   d_A, array_descA, d_B, array_descB);
        }else{
            // A^T = U^T * L^T => X = L^-T * U^-T * B: solve U^T then L^T.
            ptrtrs(handle, 'L', 'U', trans, 'N', b_rows, b_cols,
                   d_A, array_descA, d_B, array_descB);
            ptrtrs(handle, 'L', 'L', trans, 'U', b_rows, b_cols,
                   d_A, array_descA, d_B, array_descB);
        }
    }else{
        if(trans == 'N'){
            // X * L * U = B => X = B * U^-1 * L^-1: solve U then L.
            ptrtrs(handle, 'R', 'U', 'N', 'N', b_rows, b_cols,
                   d_A, array_descA, d_B, array_descB);
            ptrtrs(handle, 'R', 'L', 'N', 'U', b_rows, b_cols,
                   d_A, array_descA, d_B, array_descB);
        }else{
            // X * U^T * L^T = B => X = B * L^-T * U^-T: solve L^T then U^T.
            ptrtrs(handle, 'R', 'L', trans, 'U', b_rows, b_cols,
                   d_A, array_descA, d_B, array_descB);
            ptrtrs(handle, 'R', 'U', trans, 'N', b_rows, b_cols,
                   d_A, array_descA, d_B, array_descB);
        }
    }
}

template void pgetrs_nopiv<float>(
    const DdlaHandle_t&, const char& side, const char& trans, const int& n, const int& nrhs,
    float* d_A, const int* array_descA,
    float* d_B, const int* array_descB
);
template void pgetrs_nopiv<double>(
    const DdlaHandle_t&, const char& side, const char& trans, const int& n, const int& nrhs,
    double* d_A, const int* array_descA,
    double* d_B, const int* array_descB
);
template void pgetrs_nopiv<std::complex<float>>(
    const DdlaHandle_t&, const char& side, const char& trans, const int& n, const int& nrhs,
    std::complex<float>* d_A, const int* array_descA,
    std::complex<float>* d_B, const int* array_descB
);
template void pgetrs_nopiv<std::complex<double>>(
    const DdlaHandle_t&, const char& side, const char& trans, const int& n, const int& nrhs,
    std::complex<double>* d_A, const int* array_descA,
    std::complex<double>* d_B, const int* array_descB
);

} // namespace ddla
