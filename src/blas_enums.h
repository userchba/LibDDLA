#ifndef DDLA_BLAS_ENUMS_H
#define DDLA_BLAS_ENUMS_H

// Validate and normalize the LAPACK character flags shared by both backends.
// Vendor enum conversion belongs to GPU execution in BLAS/blas_impl.h.

#include <stdexcept>

namespace ddla {
namespace detail {

inline char check_uplo(char uplo)
{
    if (uplo == 'U' || uplo == 'u') return 'U';
    if (uplo == 'L' || uplo == 'l') return 'L';
    throw std::runtime_error("ddla: invalid uplo flag (expected 'U' or 'L')");
}

inline char check_side(char side)
{
    if (side == 'L' || side == 'l') return 'L';
    if (side == 'R' || side == 'r') return 'R';
    throw std::runtime_error("ddla: invalid side flag (expected 'L' or 'R')");
}

inline char check_diag(char diag)
{
    if (diag == 'U' || diag == 'u') return 'U';
    if (diag == 'N' || diag == 'n') return 'N';
    throw std::runtime_error("ddla: invalid diag flag (expected 'U' or 'N')");
}

inline char check_trans(char trans)
{
    if (trans == 'N' || trans == 'n') return 'N';
    if (trans == 'T' || trans == 't') return 'T';
    if (trans == 'C' || trans == 'c') return 'C';
    throw std::runtime_error("ddla: invalid trans flag (expected 'N', 'T' or 'C')");
}

// The symmetric/Hermitian rank-k families (xSYRK, xHERK, xSYR2K, xHER2K)
// accept only 'N' or 'T'/'C' semantics -- there is no "no-transpose vs
// transpose" distinction beyond that, and LAPACK itself normalises any
// conjugate-transpose request to a plain transpose for the real types.  This
// mirrors what the pre-unification wrappers did with a local `trans_real`.
inline char trans_real(char trans)
{
    return (trans == 'N' || trans == 'n') ? 'N' : 'T';
}

} // namespace detail
} // namespace ddla

#endif // DDLA_BLAS_ENUMS_H
