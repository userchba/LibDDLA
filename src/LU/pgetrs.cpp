#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <cassert>
#include "ddla_stream.h"
#include "require_backend.h"
namespace ddla{

template <DdlaBackend Backend, typename T>
void pgetrs(
    const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
    T* A, const int* array_descA,
    const int* ipiv, // host
    T* B, const int* array_descB
)
{
    check_desc(array_descB, handle);
    check_desc(array_descA, handle);
    DdlaHandle_t ddla_handle = handle;
    detail::require_backend<Backend>(ddla_handle, "pgetrs");
    assert(side == 'L' || side == 'R');
    assert(trans == 'N' || trans == 'T' || trans == 'C');
    // side='L': B is n x nrhs, solves op(A)*X = B;
    // side='R': B is nrhs x n, solves X*op(A) = B.
    // Descriptors may describe matrices larger than the logical sub-matrix
    // (leading-block); only the leading n columns of A and the corresponding
    // leading block of B are referenced.
    if(side == 'L'){
        assert(n <= array_descA[DDLA_M_] && n <= array_descA[DDLA_N_]);
        assert(n <= array_descB[DDLA_M_]);
    }else{
        assert(n <= array_descA[DDLA_M_] && n <= array_descA[DDLA_N_]);
        assert(n <= array_descB[DDLA_N_]);
    }
    const int b_rows = (side == 'L') ? n : nrhs;
    const int b_cols = (side == 'L') ? nrhs : n;

    // Apply the pgetrf pivot permutation to B via plapiv.  With
    // A = P*L*U and P = P(0)*...*P(n-2), P(k) swapping rows k and
    // ipiv(k)-1, forward application computes P^T*B (rows) or B*P (columns),
    // backward application computes P*Z (rows) or Z*P^T (columns).  rowcol
    // selects rows of B (side='L') or columns of B (side='R'); direc selects
    // the application order.  The pivot vector follows array_descA's row
    // distribution, so array_descIP is dA.
    if(side == 'L'){
        if(trans == 'N'){
            // A = P*L*U => X = U^-1 * L^-1 * (P^T * B)
            plapiv<Backend>(handle, 'F', 'R', 'C', n, nrhs, B, array_descB, ipiv, array_descA, nullptr);
            ptrtrs<Backend>(handle, 'L', 'L', 'N', 'U', b_rows, b_cols, A, array_descA, B, array_descB);
            ptrtrs<Backend>(handle, 'L', 'U', 'N', 'N', b_rows, b_cols, A, array_descA, B, array_descB);
        }else{
            // op(A)^T = U^T * L^T * P^T => X = P * L^-T * U^-T * B
            ptrtrs<Backend>(handle, 'L', 'U', trans, 'N', b_rows, b_cols, A, array_descA, B, array_descB);
            ptrtrs<Backend>(handle, 'L', 'L', trans, 'U', b_rows, b_cols, A, array_descA, B, array_descB);
            plapiv<Backend>(handle, 'B', 'R', 'C', n, nrhs, B, array_descB, ipiv, array_descA, nullptr);
        }
    }else{
        if(trans == 'N'){
            // X * P * L * U = B => X = B * U^-1 * L^-1 * P^T
            ptrtrs<Backend>(handle, 'R', 'U', 'N', 'N', b_rows, b_cols, A, array_descA, B, array_descB);
            ptrtrs<Backend>(handle, 'R', 'L', 'N', 'U', b_rows, b_cols, A, array_descA, B, array_descB);
            plapiv<Backend>(handle, 'B', 'C', 'C', n, nrhs, B, array_descB, ipiv, array_descA, nullptr);
        }else{
            // X * U^T * L^T * P^T = B => X = B * P * L^-T * U^-T
            plapiv<Backend>(handle, 'F', 'C', 'C', n, nrhs, B, array_descB, ipiv, array_descA, nullptr);
            ptrtrs<Backend>(handle, 'R', 'L', trans, 'U', b_rows, b_cols, A, array_descA, B, array_descB);
            ptrtrs<Backend>(handle, 'R', 'U', trans, 'N', b_rows, b_cols, A, array_descA, B, array_descB);
        }
    }
}

#define INSTANTIATE_PGETRS(BACKEND, TYPE)                                   \
    template void pgetrs<BACKEND, TYPE>(const DdlaHandle_t&, char side, char trans, int n, int nrhs, \
    TYPE* A, const int* array_descA, \
    const int* ipiv, \
    TYPE* B, const int* array_descB)

#if DDLA_HAS_CPU
INSTANTIATE_PGETRS(DdlaBackend::CPU, float);
INSTANTIATE_PGETRS(DdlaBackend::CPU, double);
INSTANTIATE_PGETRS(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PGETRS(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_PGETRS(DdlaBackend::GPU, float);
INSTANTIATE_PGETRS(DdlaBackend::GPU, double);
INSTANTIATE_PGETRS(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PGETRS(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_PGETRS



} // namespace ddla
