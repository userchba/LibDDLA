#ifndef DDLA_PLAN_UTIL_H
#define DDLA_PLAN_UTIL_H

// ---------------------------------------------------------------------------
// Internal helpers of the "plan" phase.  Every routine is split into
//   1. <r>_check -- argument validation,
//   2. <r>_plan  -- geometry + size and allocate every temporary,
//   3. <r>_run   -- the algorithm; no validation, no allocation,
// (see ptrtrm_impl.h for the model).  A plan that has to serve calls whose
// offsets and extents differ from one call to the next (a panel loop that
// plans its callee once) sizes its buffers with the bounds below instead of
// the exact local extents of one call.
// ---------------------------------------------------------------------------

#include <algorithm>

namespace ddla {
namespace detail {

// Number of blocks of size b that an index range of length len can touch,
// wherever the range starts.
inline int tilesmax(int len, int b)
{
    return len > 0 ? (len + b - 1) / b + 1 : 0;
}

// Upper bound on the number of entries of an index range of length len that
// one process owns in a block-cyclic distribution (block size b over P
// processes), for every start offset and every owner coordinate.  The
// range's blocks are consecutive, so a process owns at most
// ceil(tilesmax / P) of them, and never more than len entries.
inline int locmax(int len, int b, int P)
{
    if (len <= 0) return 0;
    const long long tiles = tilesmax(len, b);
    const long long owned = (tiles + P - 1) / P * b;
    return static_cast<int>(std::min<long long>(len, owned));
}

} // namespace detail
} // namespace ddla

#endif // DDLA_PLAN_UTIL_H
