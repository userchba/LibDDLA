#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "api_grid_test_common.h"
#include "Eigensolver/pherm_complete_workspace.h"

// ---------------------------------------------------------------------------
// pherm_complete (stage 0 of pheevd): the full Hermitian matrix must be rebuilt
// from the ONE stored triangle.
//
// The input holds the stored triangle of a Hermitian matrix with unique hashed
// entries; the opposite strict triangle and, for complex types, the imaginary
// part of the diagonal are POISON (NaN, or a large finite value) -- they must
// never be read.  After the call every entry of the leading n x n block must
// EQUAL the full matrix bit for bit (the pass only copies and conjugates), the
// diagonal must be real, and everything outside the leading block (the
// descriptor may be larger than n) must be untouched.
//
// Mutation-tested (see .claude_jobs/dist/p1_*_gate.sh): flipped triangle, missing conjugation, wrong diagonal
// half, kept imaginary diagonal, skipped local blocks, wrong pack/unpack offsets, mismatched send/receive
// order, wrong edge-block extents -- every one must make this test fail on at least one grid.
// Coverage: both uplo values, all four types, both poison kinds, block sizes
// that straddle the 32-wide kernel tile, ragged n (n % nb != 0, n < nb, n == 1),
// and every grid the harness enumerates (square and non-square; blocks whose
// source and destination ranks coincide, differ, and are the same process).
// ---------------------------------------------------------------------------

using namespace api_grid_test;

namespace {

double hash01(std::uint64_t i, int tag)
{
    std::uint64_t x = (i + 1) * 0x9E3779B97F4A7C15ull ^ static_cast<std::uint64_t>(tag) * 0xC2B2AE3D27D4EB4Full;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27; x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    return static_cast<double>(x >> 11) / 9007199254740992.0 - 0.5;
}

template <typename T> constexpr bool is_real_v = std::is_same_v<T, typename ddla::real_type<T>::type>;

// The full Hermitian matrix: unique entries, real diagonal.
template <typename T>
T full_value(int i, int j)
{
    using rt = typename ddla::real_type<T>::type;
    if (i == j) return T(static_cast<rt>(1.0 + hash01(static_cast<std::uint64_t>(i), 11)));
    const int a = std::max(i, j), b = std::min(i, j);   // the lower element (a > b)
    const std::uint64_t key = static_cast<std::uint64_t>(a) * 1048576ull + static_cast<std::uint64_t>(b);
    const rt re = static_cast<rt>(hash01(key, 21));
    if constexpr (is_real_v<T>) {
        return T(re);
    } else {
        const rt im = static_cast<rt>(hash01(key, 22));
        return (i > j) ? T(re, im) : T(re, -im);
    }
}

template <typename T>
T poison(int i, int j, bool nan_kind)
{
    using rt = typename ddla::real_type<T>::type;
    const rt v = nan_kind ? std::numeric_limits<rt>::quiet_NaN()
                          : static_cast<rt>(1.0e6 * (1.0 + ((i * 3 + j) % 5)));
    if constexpr (is_real_v<T>) return T(v);
    else return T(v, nan_kind ? v : static_cast<rt>(-0.5e6));
}

// outside the leading block: a value the pass must leave alone
template <typename T>
T sentinel(int i, int j)
{
    using rt = typename ddla::real_type<T>::type;
    const rt re = static_cast<rt>(100.0 + (i % 17) + 0.25 * (j % 13));
    if constexpr (is_real_v<T>) return T(re);
    else return T(re, static_cast<rt>(-3.0 - (i % 5)));
}

template <typename T>
void check_one(const ddla::DdlaHandle_t& handle, int n, int nb, int extra, bool upper, bool nan_kind,
               int scratch_mode = -1, int lld_padding = 0)
{
    const int N0 = n + extra;   // descriptor order; the logical order is n
    int descA[ddla::DDLA_DLEN_];
    ddlaDescInit(descA, handle, N0, N0, nb, nb, 0, 0);
    descA[ddla::DDLA_LLD_] += lld_padding;
    DeviceBuffer<T> d_A(handle, local_size(handle, descA));

    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
    const int m_loc = ddla::num_loc(N0, nb, myprow, 0, nprows);
    const int n_loc = ddla::num_loc(N0, nb, mypcol, 0, npcols);
    const int lld = descA[ddla::DDLA_LLD_];

    std::vector<T> local(local_size(handle, descA), T{});
    for (int jl = 0; jl < n_loc; ++jl) {
        const int gj = ddla::indxl2g(jl, nb, mypcol, 0, npcols);
        for (int il = 0; il < m_loc; ++il) {
            const int gi = ddla::indxl2g(il, nb, myprow, 0, nprows);
            T v;
            if (gi >= n || gj >= n) {
                v = sentinel<T>(gi, gj);
            } else if (upper ? gi > gj : gi < gj) {
                v = poison<T>(gi, gj, nan_kind);               // opposite triangle: never read
            } else if (gi == gj && !is_real_v<T>) {
                const T p = poison<T>(gi, gj, nan_kind);        // diagonal: real part valid, imaginary poison
                if constexpr (!is_real_v<T>) v = T(std::real(full_value<T>(gi, gj)), std::imag(p));
            } else {
                v = full_value<T>(gi, gj);                      // stored triangle (and the real diagonal)
            }
            local[il + static_cast<std::size_t>(jl) * lld] = v;
        }
        for (int il = m_loc; il < lld; ++il)
            local[il + static_cast<std::size_t>(jl) * lld] = sentinel<T>(il + 10000, jl);
    }
    upload<T>(handle, d_A.ptr, local);
    check_ddla_sync(handle);

    // Count the local cells whose transposed partner is remote, independently
    // of the library's block plan. Each is exactly one send or receive entry.
    std::size_t needed = 0;
    for (int jl = 0; jl < n_loc; ++jl) {
        const int gj = ddla::indxl2g(jl, nb, mypcol, 0, npcols);
        if (gj >= n) continue;
        for (int il = 0; il < m_loc; ++il) {
            const int gi = ddla::indxl2g(il, nb, myprow, 0, nprows);
            if (gi < n && ((gj / nb) % nprows != myprow || (gi / nb) % npcols != mypcol))
                ++needed;
        }
    }
    constexpr std::size_t guard = 7;
    std::optional<DeviceBuffer<T>> scratch;
    std::vector<T> scratch_before;
    std::size_t capacity = needed;
    if (scratch_mode >= 0) {
        // 0 exact, 1 oversized, 2 short, 3 mixed borrowing/fallback by rank,
        // 4 null pointer with a nonzero advertised capacity.
        if (scratch_mode == 1 || scratch_mode == 4) capacity += 17;
        if (needed && (scratch_mode == 2 ||
                       (scratch_mode == 3 && ddlaGetRank(handle) % 2))) --capacity;
        scratch.emplace(handle, capacity + 2 * guard);
        scratch_before.assign(capacity + 2 * guard, poison<T>(0, 0, true));
        for (std::size_t k = 0; k < guard; ++k) {
            scratch_before[k] = sentinel<T>(static_cast<int>(k), 100);
            scratch_before[guard + capacity + k] = sentinel<T>(static_cast<int>(k), 200);
        }
        upload(handle, scratch->ptr, scratch_before);
        check_ddla_sync(handle);
        ddla::detail::pherm_complete_with_workspace(
            handle, upper ? 'U' : 'L', n, d_A.ptr, descA,
            scratch_mode == 4 ? nullptr : scratch->ptr + guard, capacity);
    } else {
        ddla::pherm_complete(handle, upper ? 'U' : 'L', n, d_A.ptr, descA);
    }
    check_ddla_sync(handle);

    const std::vector<T> got = download<T>(handle, d_A.ptr, local.size());
    double bad = 0.0, first_i = -1, first_j = -1;
    for (int jl = 0; jl < n_loc; ++jl) {
        const int gj = ddla::indxl2g(jl, nb, mypcol, 0, npcols);
        for (int il = 0; il < m_loc; ++il) {
            const int gi = ddla::indxl2g(il, nb, myprow, 0, nprows);
            const T want = (gi >= n || gj >= n) ? sentinel<T>(gi, gj) : full_value<T>(gi, gj);
            if (!(got[il + static_cast<std::size_t>(jl) * lld] == want)) {   // NaN never equals
                if (bad == 0.0) { first_i = gi; first_j = gj; }
                bad += 1.0;
            }
        }
        for (int il = m_loc; il < lld; ++il) {
            const std::size_t k = il + static_cast<std::size_t>(jl) * lld;
            if (!(got[k] == local[k])) ++bad;
        }
    }
    const std::string tag = std::string("pherm_complete ") + (upper ? "U" : "L") + (nan_kind ? " nan" : " big") +
                            " n=" + std::to_string(n) + " nb=" + std::to_string(nb) +
                            " extra=" + std::to_string(extra) + " scratch=" + std::to_string(scratch_mode) +
                            " pad=" + std::to_string(lld_padding);
    if (bad > 0.0)
        std::cerr << "[rank " << ddlaGetRank(handle) << "] " << tag << ": first mismatch at global ("
                  << first_i << "," << first_j << ")" << std::endl;
    require_close(handle, tag + " mismatching entries", bad, 0.5);
    if (scratch) {
        const std::vector<T> packed = download(handle, scratch->ptr, scratch_before.size());
        const bool borrowed = ddlaGetBackend(handle) == ddla::DdlaBackend::GPU &&
                              scratch_mode != 4 && capacity >= needed;
        double scratch_bad = 0;
        for (std::size_t k = 0; k < packed.size(); ++k) {
            if (borrowed && k >= guard && k - guard < needed) {
                // Both packs must overwrite their poison before being read.
                if (!std::isfinite(std::real(packed[k])) || !std::isfinite(std::imag(packed[k])))
                    ++scratch_bad;
            } else if (std::memcmp(&packed[k], &scratch_before[k], sizeof(T)) != 0) {
                ++scratch_bad;
            }
        }
        require_close(handle, tag + " scratch guards/coverage", scratch_bad, 0.5);
    }
}

template <typename T>
void check_type(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    std::vector<int> nbs{base.nb, 2, 5, 16, 33};
    std::sort(nbs.begin(), nbs.end());
    nbs.erase(std::unique(nbs.begin(), nbs.end()), nbs.end());
    for (const int nb : nbs) {
        // 7 and 9+ block rows: a (source, destination) rank pair then exchanges several blocks whose
        // row order and column order DISAGREE (both sides must sort by the source block's origin), and on
        // small-lcm grids blocks whose source and destination coincide (the local path) appear
        std::vector<int> ns{1, 2, nb - 1, nb, nb + 1, 2 * nb, 3 * nb + 2, 4 * nb + nb / 2,
                            7 * nb + 1, 9 * nb + nb / 2};
        std::sort(ns.begin(), ns.end());
        ns.erase(std::unique(ns.begin(), ns.end()), ns.end());
        for (const int n : ns) {
            if (n < 1) continue;
            for (const bool upper : {true, false}) {
                check_one<T>(handle, n, nb, 0, upper, true);
                check_one<T>(handle, n, nb, 0, upper, false);
            }
        }
        // a descriptor larger than the logical order: the leading n x n block only
        for (const bool upper : {true, false}) {
            check_one<T>(handle, 2 * nb + 1, nb, nb + 1, upper, true);
            check_one<T>(handle, nb + 1, nb, 2 * nb + 3, upper, false);
            for (int mode = 0; mode != 5; ++mode)
                check_one<T>(handle, 7 * nb + 1, nb, nb + 2, upper, true, mode, 3);
            // Empty local extents, zero communication, and a partial edge block.
            check_one<T>(handle, 1, nb, 0, upper, true, 0);
            check_one<T>(handle, nb + 1, nb, 0, upper, false, 0);
            check_one<T>(handle, 0, nb, nb + 1, upper, true, 1, 2);
        }
    }
}

} // namespace

static void check_body(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    check_type<float>(handle, base);
    check_type<double>(handle, base);
    check_type<std::complex<float>>(handle, base);
    check_type<std::complex<double>>(handle, base);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pherm_complete", check_body);
}
