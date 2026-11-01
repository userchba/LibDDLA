#ifndef DDLA_PLAN_UTIL_H
#define DDLA_PLAN_UTIL_H

// ---------------------------------------------------------------------------
// Internal routines that need temporary memory have three components, as the
// cuSOLVERMp / cuBLASMp routines do:
//   1. <r>_check          -- argument validation,
//   2. <r>_workspace_size -- the scratch in bytes: backend memory (device on
//                            GPU handles, host on CPU handles) and host memory,
//                            only the kinds the routine needs; no allocation,
//   3. <r>_run            -- the algorithm on the caller's scratch
//                            (pointer + bytes of each kind); no validation,
//                            no allocation.
// The caller allocates the bytes with ddla::vector<char> (backend) and
// std::vector<char> (host).  Scratch-free routines have check/run only.
// A run carves its buffers from the front of its scratch with take(); a run
// that calls other runs takes its own buffers first and hands the rest on.
// A size function that serves calls whose offsets and extents differ from one
// call to the next (a panel loop) sizes its buffers with the bounds below.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cstddef>

namespace ddla {
namespace detail {

// Bytes take<T>(cursor, count) consumes: count elements, padded to 256 bytes
// so that the next buffer stays aligned.
template <typename T>
constexpr std::size_t padded_bytes(std::size_t count)
{
    return (count * sizeof(T) + 255) / 256 * 256;
}

// The next count elements of T at cursor; advances cursor past them.
template <typename T>
inline T* take(char*& cursor, std::size_t count)
{
    T* p = reinterpret_cast<T*>(cursor);
    cursor += padded_bytes<T>(count);
    return p;
}

// Number of blocks of size b that an index range of length len can touch,
// wherever the range starts.
inline int tilesmax(int len, int b)
{
    return len > 0 ? static_cast<int>(1 + (static_cast<long long>(len) + b - 2) / b) : 0;
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
