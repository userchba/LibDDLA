#include <ddla/ddla.h>
#include "lu_internal.h"
#include "ddla_desc.h"
#include <cassert>
#include <stdexcept>
#include <string>
#include "ddla_stream.h"
#include "require_backend.h"

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
 * @param A     Pointer in the selected backend to A (input: coefficient; output: LU factors).
 * @param array_descA  ScaLAPACK int[9] descriptor for A.
 * @param B     Pointer in the selected backend to RHS / solution B (input/output).
 * @param array_descB  ScaLAPACK int[9] descriptor for B.
 * @throws std::runtime_error if LU factorization fails (info != 0).
 */
template <DdlaBackend Backend, typename T>
void pgesv_bpiv(
    const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
    T* A, const int* array_descA,
    T* B, const int* array_descB
)
{
    check_desc(array_descB, handle);
    check_desc(array_descA, handle);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);


    DdlaHandle_t ddla_handle = handle;
    detail::require_backend<Backend>(ddla_handle, "pgesv_bpiv");
    assert(n <= array_descA[DDLA_M_] && n <= array_descA[DDLA_N_]);
    const int m_loc_A = num_loc(n, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
    int* pivots = nullptr;
    DDLA_CHECK(ddlaMalloc(ddla_handle, reinterpret_cast<void**>(&pivots), sizeof(int) * std::max(1, m_loc_A)));
    int info = 0;
    pgetrf_bpiv<Backend>(handle, n, n, A, array_descA, pivots, info);
    if(info != 0){
        DDLA_CHECK(ddlaFree(ddla_handle, pivots));
        throw std::runtime_error("pgesv_bpiv: pgetrf_bpiv returned info = " + std::to_string(info));
    }
    pgetrs_bpiv<Backend>(handle, side, trans, n, nrhs, A, array_descA, pivots, B, array_descB);
    DDLA_CHECK(ddlaFree(ddla_handle, pivots));
}

#define INSTANTIATE_PGESV_BPIV(BACKEND, TYPE)                                   \
    template void pgesv_bpiv<BACKEND, TYPE>(const DdlaHandle_t&, char side, char trans, int n, int nrhs, \
    TYPE* A, const int* array_descA, \
    TYPE* B, const int* array_descB)

#if DDLA_HAS_CPU
INSTANTIATE_PGESV_BPIV(DdlaBackend::CPU, float);
INSTANTIATE_PGESV_BPIV(DdlaBackend::CPU, double);
INSTANTIATE_PGESV_BPIV(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PGESV_BPIV(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_PGESV_BPIV(DdlaBackend::GPU, float);
INSTANTIATE_PGESV_BPIV(DdlaBackend::GPU, double);
INSTANTIATE_PGESV_BPIV(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PGESV_BPIV(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_PGESV_BPIV


} // namespace ddla
