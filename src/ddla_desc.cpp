#include "ddla_desc.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ddla{

void check_desc(const int* desc, const DdlaHandle_t& handle)
{
    if (desc == nullptr)
        throw std::invalid_argument("ddla: null ScaLAPACK descriptor");
    if (handle == nullptr)
        throw std::invalid_argument(
            "ddla: null handle for ScaLAPACK descriptor");
    if (desc[DDLA_DTYPE_] != DDLA_BLOCK_CYCLIC_2D)
        throw std::invalid_argument(
            "ddla: unsupported descriptor type "
            + std::to_string(desc[DDLA_DTYPE_])
            + "; only dense block-cyclic (DTYPE_ == 1) is supported");

    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
    if (nprows <= 0 || npcols <= 0)
        throw std::invalid_argument(
            "ddla: handle carries no process grid");

    if (desc[DDLA_M_] < 0 || desc[DDLA_N_] < 0)
        throw std::invalid_argument(
            "ddla: descriptor M/N must be non-negative, got M="
            + std::to_string(desc[DDLA_M_]) + " N="
            + std::to_string(desc[DDLA_N_]));
    if (desc[DDLA_MB_] <= 0 || desc[DDLA_NB_] <= 0)
        throw std::invalid_argument(
            "ddla: descriptor MB/NB must be positive, got MB="
            + std::to_string(desc[DDLA_MB_]) + " NB="
            + std::to_string(desc[DDLA_NB_]));
    if (desc[DDLA_RSRC_] < 0 || desc[DDLA_RSRC_] >= nprows
        || desc[DDLA_CSRC_] < 0 || desc[DDLA_CSRC_] >= npcols)
        throw std::invalid_argument(
            "ddla: descriptor RSRC/CSRC outside the process grid ("
            + std::to_string(nprows) + "x" + std::to_string(npcols)
            + "), got RSRC=" + std::to_string(desc[DDLA_RSRC_])
            + " CSRC=" + std::to_string(desc[DDLA_CSRC_]));

    // ScaLAPACK allows LLD_A to be larger than max(1, LOCr(m)), for callers
    // whose local buffer is over-allocated, so accept any value at least that
    // big and reject one that would index inside a column -- which is also
    // the check that catches a descriptor built against a different process
    // grid.
    const int required_lld =
        std::max(1, num_loc(desc[DDLA_M_], desc[DDLA_MB_], myprow,
                            desc[DDLA_RSRC_], nprows));
    if (desc[DDLA_LLD_] < required_lld)
        throw std::invalid_argument(
            "ddla: ScaLAPACK descriptor LLD_A=" + std::to_string(desc[DDLA_LLD_])
            + " is smaller than the required max(1, LOCr(M_A))="
            + std::to_string(required_lld)
            + "; the descriptor does not match this handle's process grid");
}

}
