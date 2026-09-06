#ifndef DDLA_SRC_DESC_H
#define DDLA_SRC_DESC_H

// Internal descriptor validation. The public surface (include/ddla/ddla_desc.h)
// is the plain ScaLAPACK int[9]; routine bodies index it directly the way
// ScaLAPACK's own sources use DESCA(MB_), and always take the process grid
// from the DdlaHandle_t passed alongside -- nothing about a descriptor is
// cached or bound.

#include <ddla/ddla_desc.h>

namespace ddla{

/// Validate a ScaLAPACK-layout descriptor array against @p handle's process
/// grid.
///
/// @p desc must be a length-9 dense block-cyclic descriptor exactly as
/// DESCINIT produces it. desc[DDLA_CTXT_] is deliberately ignored: the grid,
/// communicators and stream come from @p handle.
///
/// @throws std::invalid_argument if @p desc or @p handle is null, if the
///         descriptor is not dense block-cyclic, if M/N is negative, MB/NB is
///         not positive, RSRC/CSRC does not address a process of the grid, or
///         desc[DDLA_LLD_] is smaller than max(1, LOCr(M_A)) on this grid
///         (the usual symptom of a descriptor built against a different
///         process grid).
void check_desc(const int* desc, const DdlaHandle_t& handle);

} // namespace ddla

#endif // DDLA_SRC_DESC_H
