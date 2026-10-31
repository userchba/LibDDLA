#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <cassert>
#include "ddla_stream.h"
#include "require_backend.h"

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
 * @param A     Pointer in the selected backend to LU factors (from pgetrf_nopiv).
 * @param array_descA  ScaLAPACK int[9] descriptor for A.
 * @param B     Pointer in the selected backend to RHS / solution B (input/output).
 * @param array_descB  ScaLAPACK int[9] descriptor for B.
 */
template <DdlaBackend Backend, typename T>
void pgetrs_nopiv(
    const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
    T* A, const int* array_descA,
    T* B, const int* array_descB
)
{
    check_desc(array_descB, handle);
    check_desc(array_descA, handle);
    DdlaHandle_t ddla_handle = handle;
    detail::require_backend<Backend>(ddla_handle, "pgetrs_nopiv");
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
            ptrtrs<Backend>(handle, 'L', 'L', 'N', 'U', b_rows, b_cols,
                   A, array_descA, B, array_descB);
            ptrtrs<Backend>(handle, 'L', 'U', 'N', 'N', b_rows, b_cols,
                   A, array_descA, B, array_descB);
        }else{
            // A^T = U^T * L^T => X = L^-T * U^-T * B: solve U^T then L^T.
            ptrtrs<Backend>(handle, 'L', 'U', trans, 'N', b_rows, b_cols,
                   A, array_descA, B, array_descB);
            ptrtrs<Backend>(handle, 'L', 'L', trans, 'U', b_rows, b_cols,
                   A, array_descA, B, array_descB);
        }
    }else{
        if(trans == 'N'){
            // X * L * U = B => X = B * U^-1 * L^-1: solve U then L.
            ptrtrs<Backend>(handle, 'R', 'U', 'N', 'N', b_rows, b_cols,
                   A, array_descA, B, array_descB);
            ptrtrs<Backend>(handle, 'R', 'L', 'N', 'U', b_rows, b_cols,
                   A, array_descA, B, array_descB);
        }else{
            // X * U^T * L^T = B => X = B * L^-T * U^-T: solve L^T then U^T.
            ptrtrs<Backend>(handle, 'R', 'L', trans, 'U', b_rows, b_cols,
                   A, array_descA, B, array_descB);
            ptrtrs<Backend>(handle, 'R', 'U', trans, 'N', b_rows, b_cols,
                   A, array_descA, B, array_descB);
        }
    }
}

#define INSTANTIATE_PGETRS_NOPIV(BACKEND, TYPE)                                   \
    template void pgetrs_nopiv<BACKEND, TYPE>(const DdlaHandle_t&, char side, char trans, int n, int nrhs, \
    TYPE* A, const int* array_descA, \
    TYPE* B, const int* array_descB)

#if DDLA_HAS_CPU
INSTANTIATE_PGETRS_NOPIV(DdlaBackend::CPU, float);
INSTANTIATE_PGETRS_NOPIV(DdlaBackend::CPU, double);
INSTANTIATE_PGETRS_NOPIV(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PGETRS_NOPIV(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_PGETRS_NOPIV(DdlaBackend::GPU, float);
INSTANTIATE_PGETRS_NOPIV(DdlaBackend::GPU, double);
INSTANTIATE_PGETRS_NOPIV(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PGETRS_NOPIV(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_PGETRS_NOPIV


} // namespace ddla
