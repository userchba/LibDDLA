#include "pstedc_geometry.h"

#include "check_agree.h"
#include "ddla_stream.h"
#include "eig_params.h"
#include "helpers.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace ddla::spectral_impl {
namespace {

// CPU merges only: 1 = split the merge GEMM by the children's row support
// (the dlaed2/dlaed3 trick; default), 0 = one dense GEMM (the phase-1 port,
// kept for A/B), 2 = reordered columns but one dense GEMM (debug: isolates
// a reorder bug from a sub-GEMM-range bug).  Agreed by broadcast like the
// leaf size -- the summation order differs between the two, so replicas
// must all pick the same one.
int dc_split_gemm_mode()
{
    const char* s = std::getenv("DDLA_EIG_DC_SPLIT");
    if (s != nullptr && *s != '\0') {
        const int v = std::atoi(s);
        if (v == 0 || v == 1 || v == 2)
            return v;
    }
    return 1;
}

// Default leaf size of the eigenvalue-only tree (DcRowsStore; 'V' keeps 1024).
// A leaf costs O(m^3) because it needs its two eigenvector rows from a full
// ?STEDC('I'), while a merge only costs O(K^2): small leaves are cheaper.
constexpr int kValsLeaf = 128;

// Eigenvalue-only merges keep the deltas of their roots (one ?LAED4 pass) when
// they fit this many MiB per rank, and otherwise recompute them in a second
// pass (bit-identical: ?LAED4 is deterministic).  0 = always recompute.  Read
// on rank 0 and broadcast like the leaf size.
int pstedc_keep_mb()
{
    const char* s = std::getenv("DDLA_EIG_DC_KEEP_MB");
    if (s != nullptr && *s != '\0') {
        char* end = nullptr;
        const long v = std::strtol(s, &end, 10);
        if (end != s && *end == '\0' && v >= 0 && v <= 1048576)
            return static_cast<int>(v);
    }
    return 64;
}

} // namespace

template <class R>
PstedcVectorPlan pstedc_policy(const DdlaHandle_t& h, int n, int nblk)
{
    PstedcVectorPlan p;
    p.backend = h->backend;
    p.n = n;
    p.nblk = nblk;
    p.scalar_bytes = sizeof(R);
    p.ranks = h->nprocs;
    p.rank_leaf = h->nprocs > 1 && n > 1;
    p.block_leaf = p.rank_leaf && (n / nblk + (n % nblk != 0)) >= std::min(n, h->nprocs);
    // Oversized blocks must not concentrate a quadratic factor on one rank.
    if (p.block_leaf) {
        const std::size_t m = pstedc_max_leaf(p);
        p.block_leaf = m * m <= std::size_t(std::max(1, n / h->nprocs)) * n;
    }
    p.rank = h->myid;
    if (!n)
        return p;
    int policy[3] = {1024, 0, 0};
    if (p.rank == 0) {
        // Keep the old one-leaf device footprint through 1024. Above it,
        // local STEDC's smaller leaves reduce host work without adding a strip.
        const int leaf = p.backend == DdlaBackend::GPU && p.ranks == 1 && n > 1024
            ? stedc_gpu_leaf_size(n) : 1024;
        policy[0] = leaf;
        policy[1] = p.backend == DdlaBackend::CPU ? dc_split_gemm_mode() : 0;
        if (const char* env = std::getenv("DDLA_EIG_DC_STRIP")) {
            const long v = std::strtol(env, nullptr, 10);
            if (v >= 16)
                policy[2] = int(std::min<long>(v, INT_MAX));
        }
    }
    if(p.ranks>1)MPI_CHECK(MPI_Bcast(policy, 3, MPI_INT, 0, h->comm));
    p.leaf = policy[0];
    p.split = policy[1];
    p.threads = h->host_threads;
    p.profile = eig_profile_enabled();
    const auto rows = (static_cast<unsigned long long>(n) + p.ranks - 1) / p.ranks;
    auto width =
        p.backend == DdlaBackend::CPU ? std::min(512ull, ((std::max(128ull, rows) + 63) / 64) * 64) : 512ull;
    if (policy[2])
        width = static_cast<unsigned long long>(policy[2]);
    p.strip = std::max(16, static_cast<int>(std::min(
                               width, (256ull << 20) / (sizeof(R) * static_cast<unsigned long long>(n)))));
    p.root_chunk = std::max(
        32, std::min({4 * p.threads, 256,
                      static_cast<int>((128ull << 20) / (sizeof(R) * static_cast<std::size_t>(n) + 1))}));
    if (p.backend == DdlaBackend::GPU && p.ranks == 1)
        p.strip = std::min(p.strip, n);
    if (p.ranks > 1 && n > 1) {
        const int share = n / p.ranks + (n % p.ranks != 0);
        // At most a rank's share of the secular matrix is formed at once.
        // The fixed 2x2 merge uses four scalars and needs two strip columns.
        p.strip = std::min(p.strip, std::max(2, share));
        p.root_chunk = std::min(p.root_chunk, std::max(1, share));
    }
    return p;
}

template PstedcVectorPlan pstedc_policy<float>(const DdlaHandle_t&, int, int);
template PstedcVectorPlan pstedc_policy<double>(const DdlaHandle_t&, int, int);

template <class R>
PstedcPlan pstedc_policy(const DdlaHandle_t& h, char jobz, int n, int nblk, int ldz)
{
    PstedcPlan p;
    p.rows = pstedc_policy<R>(h, n, nblk);
    p.rows.ldy = ldz;
    if (jobz == 'N') p.rows.rank_leaf = false;
    p.mode = jobz == 'N' ? PstedcMode::values : PstedcMode::vectors;
    if (n && jobz == 'N') {
        int knobs[2]{};
        if (p.rows.rank == 0) {
            knobs[0] = kValsLeaf;
            knobs[1] = pstedc_keep_mb();
        }
        MPI_CHECK(MPI_Bcast(knobs, 2, MPI_INT, 0, h->comm));
        p.rows.leaf = knobs[0];
        p.rows.split = 0;
        p.keep_mb = knobs[1];
    }
    return p;
}

template <class R>
PstedcPlan pstedc_check(const DdlaHandle_t& h, char jobz, int n, int nblk, const R* d, const R* e, const R* w,
                        const R* z, int ldz)
{
    detail::agree_check(h, "pstedc", [&](detail::Agree& agreement) {
        agreement.mode("spectral operation", 3112);
        if (!h)
            throw std::invalid_argument("pstedc: null handle");
        if (h->backend == DdlaBackend::GPU && !DDLA_HAS_GPU)
            throw std::invalid_argument("pstedc: backend not built");
        if ((jobz != 'N' && jobz != 'V') || n < 0 || nblk < 1)
            throw std::invalid_argument("pstedc: invalid jobz or dimension");
        if (n && (!d || !w || (n > 1 && !e)))
            throw std::invalid_argument("pstedc: invalid input/output pointer");
    });
    const int signature[] = {3112, int(h->backend), int(jobz), n, nblk, int(sizeof(R))};
    detail::agree_signature(h, "pstedc", signature);
    if (jobz == 'V') {
        PstedcPlan p;
        p.mode = PstedcMode::vectors;
        p.rows = pstedc_check<R>(h, n, nblk, d, e, w, z, ldz);
        return p;
    }
    return pstedc_policy<R>(h, jobz, n, nblk, ldz);
}

template <class R>
PstedcVectorPlan pstedc_check(const DdlaHandle_t& h, int n, int nblk, const R* d, const R* e, const R* w, const R* Y,
                                 int ldy, const R* scratch, std::size_t scratch_elements)
{
    auto p = detail::agree_check(h, "pstedc", [&](detail::Agree& agreement) {
        agreement.mode("spectral operation", 3111);
        PstedcVectorPlan result;
        if (h == nullptr)
            throw std::invalid_argument("pstedc: null handle");
        result.backend = ddlaGetBackend(h);
        result.n = n;
        result.ldy = ldy;
        result.ranks = h->nprocs;
        result.rank = h->myid;
        if (n < 0 || nblk < 1)
            throw std::invalid_argument("pstedc: invalid n or nblk");
        if (result.ranks < 1 || result.rank < 0 || result.rank >= result.ranks)
            throw std::invalid_argument("pstedc: invalid communicator rank");
        const detail::RowShare rows(n, result.ranks);
        if (n && (!d || !w || (!Y && rows.count(result.rank) > 0) || (n > 1 && !e) ||
                  ldy < std::max(1, rows.count(result.rank))))
            throw std::invalid_argument("pstedc: invalid input/output pointer or row stride");
        return result;
    });
    const int signature[] = {3111, int(p.backend), n, nblk, p.ranks, int(sizeof(R))};
    detail::agree_signature(h, "pstedc", signature);
    if (!n)
        return p;
    p = pstedc_policy<R>(h, n, nblk);
    p.ldy = ldy;
    const detail::RowShare share(n, p.ranks);
    const std::size_t slab = std::size_t(std::max(1, share.count(p.rank))) * n;
    p.borrow_q = scratch && share.count(p.rank) > 0 && scratch_elements >= slab &&
                 reinterpret_cast<std::uintptr_t>(scratch) % alignof(R) == 0 &&
                 !detail::array_overlaps(scratch, slab, Y, std::size_t(ldy) * n) &&
                 !detail::array_overlaps(scratch, slab, d, n) &&
                 !detail::array_overlaps(scratch, slab, e, n - 1) && !detail::array_overlaps(scratch, slab, w, n);
    // Native one-rank LAPACK writes Y directly and needs no separate Q.
    // Do not report a borrowed factor which this path never touches.
    if (local_stedc_cpu(p))
        p.borrow_q = false;
    p.borrow_q2 = share.count(p.rank) > 0 && ldy == std::max(1, share.count(p.rank)) &&
                  !detail::array_overlaps(Y, slab, d, n) &&
                  !detail::array_overlaps(Y, slab, e, n - 1) &&
                  !detail::array_overlaps(Y, slab, w, n);

    return p;
}

template PstedcVectorPlan pstedc_check<float>(const DdlaHandle_t&, int, int, const float*, const float*,
                                                 const float*, const float*, int, const float*, std::size_t);
template PstedcVectorPlan pstedc_check<double>(const DdlaHandle_t&, int, int, const double*, const double*,
                                                  const double*, const double*, int, const double*,
                                                  std::size_t);

template PstedcPlan pstedc_policy<float>(const DdlaHandle_t&, char, int, int, int);
template PstedcPlan pstedc_policy<double>(const DdlaHandle_t&, char, int, int, int);
template PstedcPlan pstedc_check<float>(const DdlaHandle_t&, char, int, int, const float*,
                                       const float*, const float*, const float*, int);
template PstedcPlan pstedc_check<double>(const DdlaHandle_t&, char, int, int, const double*,
                                        const double*, const double*, const double*, int);

} // namespace ddla::spectral_impl
