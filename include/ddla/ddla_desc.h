#ifndef DDLA_DESC_H
#define DDLA_DESC_H

#include "ddla_handle_t.h"

#include <algorithm>

namespace ddla{

inline int indxg2p(const int &indxglob, const int &nb,
                              const int &isrcproc, const int &nprocs)
{
    return (isrcproc + indxglob / nb) % nprocs;
}

inline int indxg2l(const int &indxglob, const int &nb, const int &nprocs)
{
    return nb * (indxglob / (nb * nprocs)) + indxglob % nb;
}
inline int indxl2g(const int &indxloc, const int &nb, const int &iproc, const int &isrcproc, const int &nprocs)
{
    return nprocs * nb * (indxloc / nb) + indxloc % nb +
            ((nprocs + iproc - isrcproc) % nprocs) * nb;
}

inline int num_loc(const int& n, const int& nb, const int& iproc, const int& srcproc, const int& nprocs)
{
    int count = n / (nb * nprocs) * nb;
    int rest = n % (nb * nprocs);
    if (rest > nb * ((iproc + nprocs - srcproc) % nprocs))
    {
        if (rest - nb * ((iproc + nprocs - srcproc) % nprocs) >= nb)
            count += nb;
        else
            count += rest - nb * ((iproc + nprocs - srcproc) % nprocs);
    }
    return count;
}

// ---------------------------------------------------------------------------
// ScaLAPACK-layout descriptor
// ---------------------------------------------------------------------------
// Field offsets of a dense block-cyclic ScaLAPACK descriptor -- the array that
// DESCINIT fills. The DDLA_ prefix keeps them from colliding with the bare
// DTYPE_ / CTXT_ / M_ / ... macros a ScaLAPACK header may already define.
//
// Every distributed LibDDLA routine takes its process grid through a
// DdlaHandle_t and its matrix layout through one of these length-9 int
// arrays, so a caller can hand LibDDLA the very array ScaLAPACK's DESCINIT
// produced with no translation step in between.
inline constexpr int DDLA_DTYPE_ = 0;  ///< descriptor type; 1 == dense block-cyclic
inline constexpr int DDLA_CTXT_  = 1;  ///< BLACS context -- ignored, see below
inline constexpr int DDLA_M_     = 2;  ///< global rows
inline constexpr int DDLA_N_     = 3;  ///< global columns
inline constexpr int DDLA_MB_    = 4;  ///< row block size
inline constexpr int DDLA_NB_    = 5;  ///< column block size
inline constexpr int DDLA_RSRC_  = 6;  ///< process row owning the first block row
inline constexpr int DDLA_CSRC_  = 7;  ///< process column owning the first block column
inline constexpr int DDLA_LLD_   = 8;  ///< local leading dimension
inline constexpr int DDLA_DLEN_  = 9;  ///< number of ints in a descriptor

/// Value of desc[DDLA_DTYPE_] for the dense block-cyclic descriptors LibDDLA
/// understands (ScaLAPACK's BLOCK_CYCLIC_2D).
inline constexpr int DDLA_BLOCK_CYCLIC_2D = 1;

/// Fill @p desc the way ScaLAPACK's DESCINIT does.
///
/// Writes DTYPE_, M_, N_, MB_, NB_, RSRC_, CSRC_ and LLD_ into @p desc (a
/// caller-owned array of at least DDLA_DLEN_ ints) and zeroes CTXT_, so a
/// filled descriptor can be copied or compared wholesale. A caller handing
/// the array to ScaLAPACK must store its own BLACS context in CTXT_
/// afterwards.
///
/// The process grid comes from @p handle -- desc[DDLA_CTXT_] is deliberately
/// NOT consulted. A BLACS context integer is an index into a table private to
/// the BLACS library that minted it, so it cannot be resolved without linking
/// BLACS, and BLACS exposes no way to recover an MPI communicator from one in
/// any case. Every distributed LibDDLA routine therefore takes the handle
/// that owns the grid alongside the descriptor.
///
/// LLD follows ScaLAPACK's DESCINIT store semantics (TOOLS/descinit.f):
/// @p lld is a suggestion, stored as MAX(lld, MAX(1, LOCr(M_A))) on @p
/// handle's grid. A caller with an over-allocated local buffer passes its
/// padded leading dimension and it survives verbatim; an undersized one is
/// raised to the tight bound -- DESCINIT reports INFO = -9 in that case but
/// still stores a usable value, and LibDDLA keeps only the store, not the
/// error. The default @p lld < 0 -- a LibDDLA-only convenience, not
/// ScaLAPACK behaviour -- simply derives MAX(1, LOCr(M_A)). The reading side
/// (check_desc) still rejects a descriptor whose stored LLD_ is below the
/// bound, which is what catches a descriptor built against a different
/// process grid.
///
/// The argument order deliberately departs from DESCINIT: the handle comes
/// second, matching LibDDLA's handle-first convention across the distributed
/// routines, instead of standing in for ICTXT at DESCINIT's eighth position.
/// Only the LLD *semantics* are taken from DESCINIT.
///
/// @return DDLA_STATUS_SUCCESS, or DDLA_STATUS_INVALID_HANDLE if @p handle
///         is null or carries no process grid, or DDLA_STATUS_INVALID_VALUE
///         if @p desc is null, M/N is negative, MB/NB is not positive, or
///         RSRC/CSRC does not address a process of @p handle's grid.
inline ddlaStatus_t ddlaDescInit(int* desc, const DdlaHandle_t& handle,
                                 int m, int n, int mb, int nb,
                                 int irsrc, int icsrc, int lld = -1)
{
    if (desc == nullptr)
        return ddlaStatus_t::DDLA_STATUS_INVALID_VALUE;
    if (handle == nullptr)
        return ddlaStatus_t::DDLA_STATUS_INVALID_HANDLE;

    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
    if (nprows <= 0 || npcols <= 0)
        return ddlaStatus_t::DDLA_STATUS_INVALID_HANDLE;

    if (m < 0 || n < 0 || mb <= 0 || nb <= 0)
        return ddlaStatus_t::DDLA_STATUS_INVALID_VALUE;
    if (irsrc < 0 || irsrc >= nprows || icsrc < 0 || icsrc >= npcols)
        return ddlaStatus_t::DDLA_STATUS_INVALID_VALUE;

    // DESCINIT stores MAX(LLD, LLD_MIN): an undersized lld is a suggestion,
    // raised to the tight bound rather than rejected; a padded value
    // survives verbatim; the negative default derives the bound.
    const int lld_min = std::max(1, num_loc(m, mb, myprow, irsrc, nprows));

    desc[DDLA_DTYPE_] = DDLA_BLOCK_CYCLIC_2D;
    desc[DDLA_CTXT_]  = 0;
    desc[DDLA_M_]     = m;
    desc[DDLA_N_]     = n;
    desc[DDLA_MB_]    = mb;
    desc[DDLA_NB_]    = nb;
    desc[DDLA_RSRC_]  = irsrc;
    desc[DDLA_CSRC_]  = icsrc;
    desc[DDLA_LLD_]   = std::max(lld, lld_min);
    return ddlaStatus_t::DDLA_STATUS_SUCCESS;
}

/// Desc-level index mapping over a ScaLAPACK descriptor. Each call re-fetches
/// the process grid from @p handle, so inside a per-element loop prefer
/// hoisting one ddlaGetGridDims/ddlaGetGridCoords pair and calling the free
/// indxg2p/indxg2l/indxl2g/num_loc functions directly.
///
/// Local row index of global row @p gindx, or -1 if this rank does not own it.
///
/// Desc-level replacement for the removed DdlaDesc member of the same name:
/// the block size, source row and global row count come from @p desc, the
/// process grid from @p handle.
inline int indx_g2l_r(const int* desc, const DdlaHandle_t& handle, int gindx)
{
    if (handle == nullptr) return -1;
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
    if (myprow != indxg2p(gindx, desc[DDLA_MB_], desc[DDLA_RSRC_], nprows)
        || gindx >= desc[DDLA_M_])
        return -1;
    return indxg2l(gindx, desc[DDLA_MB_], nprows);
}

/// Local column index of global column @p gindx, or -1 if this rank does not own it.
inline int indx_g2l_c(const int* desc, const DdlaHandle_t& handle, int gindx)
{
    if (handle == nullptr) return -1;
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
    if (mypcol != indxg2p(gindx, desc[DDLA_NB_], desc[DDLA_CSRC_], npcols)
        || gindx >= desc[DDLA_N_])
        return -1;
    return indxg2l(gindx, desc[DDLA_NB_], npcols);
}

/// Global row index of local row @p lindx.
inline int indx_l2g_r(const int* desc, const DdlaHandle_t& handle, int lindx)
{
    if (handle == nullptr) return -1;
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
    return indxl2g(lindx, desc[DDLA_MB_], myprow, desc[DDLA_RSRC_], nprows);
}

/// Global column index of local column @p lindx.
inline int indx_l2g_c(const int* desc, const DdlaHandle_t& handle, int lindx)
{
    if (handle == nullptr) return -1;
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
    return indxl2g(lindx, desc[DDLA_NB_], mypcol, desc[DDLA_CSRC_], npcols);
}

} // namespace ddla


#endif // DDLA_DESC_H
