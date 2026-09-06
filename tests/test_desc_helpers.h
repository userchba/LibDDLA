#ifndef DDLA_TEST_DESC_HELPERS_H
#define DDLA_TEST_DESC_HELPERS_H

// Test-side convenience over the ScaLAPACK int[9] descriptor contract.
// LibDDLA deliberately keeps descriptors as plain int[9] arrays: the process
// grid comes from the handle, and ScaLAPACK does not store local extents in a
// descriptor at all. Production callers derive them with num_loc() plus
// ddlaGetGridDims/ddlaGetGridCoords; these helpers wrap exactly that so the
// tests can size buffers and map indices without repeating the grid fetches.
//
// Call them qualified (ddla_test::m_loc(...)) -- several tests declare local
// variables named m_loc/n_loc, so a `using namespace ddla_test` would shadow
// these functions in those scopes.

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <ddla/ddla.h>

namespace ddla_test {

/// Number of locally owned rows: num_loc over desc's row layout.
inline int m_loc(const ddla::DdlaHandle_t& handle, const int* desc)
{
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
    return ddla::num_loc(desc[ddla::DDLA_M_], desc[ddla::DDLA_MB_],
                         myprow, desc[ddla::DDLA_RSRC_], nprows);
}

/// Number of locally owned columns: num_loc over desc's column layout.
inline int n_loc(const ddla::DdlaHandle_t& handle, const int* desc)
{
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
    return ddla::num_loc(desc[ddla::DDLA_N_], desc[ddla::DDLA_NB_],
                         mypcol, desc[ddla::DDLA_CSRC_], npcols);
}

inline int nprows(const ddla::DdlaHandle_t& handle)
{
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    return nprows;
}

inline int npcols(const ddla::DdlaHandle_t& handle)
{
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    return npcols;
}

inline int myprow(const ddla::DdlaHandle_t& handle)
{
    int myprow = -1, mypcol = -1;
    ddlaGetGridCoords(handle, myprow, mypcol);
    return myprow;
}

inline int mypcol(const ddla::DdlaHandle_t& handle)
{
    int myprow = -1, mypcol = -1;
    ddlaGetGridCoords(handle, myprow, mypcol);
    return mypcol;
}

/// The removed DdlaDesc::init_square_blk: choose mb == nb as the smaller of
/// the per-dimension ceil counts, then initialize the descriptor with it.
inline void init_square_blk(int* desc, int m, int n, int irsrc, int icsrc,
                            const ddla::DdlaHandle_t& handle)
{
    const int mb = static_cast<int>(std::ceil(static_cast<double>(m)
                                              / nprows(handle)));
    const int nb = static_cast<int>(std::ceil(static_cast<double>(n)
                                              / npcols(handle)));
    const int nb_real = std::min(mb, nb);
    if (ddla::ddlaDescInit(desc, handle, m, n, nb_real, nb_real, irsrc, icsrc)
        != ddla::ddlaStatus_t::DDLA_STATUS_SUCCESS)
        throw std::invalid_argument("ddla_test: init_square_blk failed");
}

} // namespace ddla_test

#endif // DDLA_TEST_DESC_HELPERS_H
