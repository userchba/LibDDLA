#ifndef DDLA_ROW_SHARE_H
#define DDLA_ROW_SHARE_H

// Contiguous row shares of an n-row matrix over the P ranks of the handle's communicator: rank q owns the rows
// [lo(q), hi(q)), the first n % P ranks one row more.  This is the split of the eigensolver and SVD row-distributed
// matrices -- the Q_band accumulation of Hb2stAccum (hb2st_accum.cpp) and the row-slab eigenvector matrix of the
// distributed tridiagonal solve (pstedc.cpp) -- so that a slab's rows line up with the other side's columns
// when the two are multiplied.  Deliberately independent of the 2D process grid (see the layout note in
// hb2st_accum.h).

namespace ddla {
namespace detail {

struct RowShare {
    int n = 0, P = 1, base = 0, rem = 0;

    RowShare() = default;
    RowShare(int n_, int P_) : n(n_), P(P_ < 1 ? 1 : P_), base(n_ / (P_ < 1 ? 1 : P_)), rem(n_ % (P_ < 1 ? 1 : P_)) {}

    int count(int q) const { return base + (q < rem ? 1 : 0); }
    int lo(int q) const { return q * base + (q < rem ? q : rem); }
    int hi(int q) const { return lo(q) + count(q); }

    /// The rank owning global row i, 0 <= i < n.
    int owner(int i) const
    {
        const int big = rem * (base + 1);   // rows held by the ranks with one extra row
        if (i < big) return i / (base + 1);
        return rem + (i - big) / (base > 0 ? base : 1);
    }
};

} // namespace detail
} // namespace ddla

#endif // DDLA_ROW_SHARE_H
