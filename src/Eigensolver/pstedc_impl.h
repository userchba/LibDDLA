#ifndef DDLA_PSTEDC_IMPL_H
#define DDLA_PSTEDC_IMPL_H

// Internal entry of the distributed tridiagonal solve with a ROW-SLAB eigenvector matrix (stage 3 of pheevd
// on CPU and GPU handles). Public <ddla/ddla.h> is included first.
#include <ddla/ddla.h>
#include <cstddef>

namespace ddla {

// Internal row-layout entry for composite solvers and eigenvalues-only calls.
template<class T>void pstedc(const DdlaHandle_t&,char,int,
    typename real_type<T>::type*,typename real_type<T>::type*,typename real_type<T>::type*,
    typename real_type<T>::type*,int,int&);


/// Every eigenpair of the replicated real symmetric tridiagonal (d, e) by Cuppen divide and conquer; d and e are
/// destroyed.  w (HOST, length n) receives the eigenvalues in ascending order on every rank.  The eigenvector
/// matrix is NOT replicated: rank q of the handle's communicator owns the rows [lo(q), hi(q)) of Y, the
/// contiguous split of detail::RowShare (row_share.h), all n columns, column-major in backend memory with
/// leading dimension ldy >= max(1, hi(q) - lo(q)); column j belongs to w[j].  That is n^2/P elements per rank --
/// and the same per-rank memory in the solver itself (Q and the merge scratch are row slabs too, the secular
/// eigenvector strips are generated on the fly from O(K) data, so nothing of size n^2 is ever replicated or
/// communicated).  Collective over the handle's communicator.  Throws on a failed leaf solve
/// or secular root (consistently on all ranks).
template <typename rt>
void pstedc(const DdlaHandle_t& handle, int n, rt* d, rt* e, rt* w, rt* Y, int ldy);

/// Optional workspace for the private driver. The caller owns scratch, a contiguous
/// span of scratch_elements real entries in the handle's backend memory. All entries
/// may be overwritten, and the span must contain no output padding. On CPU,
/// real components of a std::complex<rt> array are also valid storage via the
/// array access guaranteed by C++17 [complex.numbers]/4. The solver
/// borrows it for Q only when it is large enough for this rank's compact row slab,
/// has suitable alignment, and
/// does not overlap Y or d/e/w. Otherwise it allocates Q as usual. Zero-row ranks
/// do not touch scratch. The pointer is never freed here; all use is complete when
/// this function returns. Collective over the grid;
/// the pointer and capacity may differ between ranks.
template <typename rt>
void pstedc(const DdlaHandle_t& handle, int n, rt* d, rt* e, rt* w, rt* Y, int ldy,
                 rt* scratch, std::size_t scratch_elements);

} // namespace ddla

#endif // DDLA_PSTEDC_IMPL_H
