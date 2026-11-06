#ifndef DDLA_PSTEDC_IMPL_H
#define DDLA_PSTEDC_IMPL_H

// Internal entry of the distributed tridiagonal solve with a ROW-SLAB eigenvector matrix (stage 3 of pheevd
// on CPU and GPU handles). Public <ddla/ddla.h> is included first.
#include <ddla/ddla.h>
#include <cstddef>

namespace ddla
{

/// Every eigenpair of the replicated real symmetric tridiagonal (d, e) by Cuppen divide and conquer; d and e
/// are destroyed.  w (HOST, length n) receives the eigenvalues in ascending order on every rank.  The
/// eigenvector matrix is NOT replicated: rank q of the handle's communicator owns the rows [lo(q), hi(q)) of
/// Y, the contiguous split of detail::RowShare (row_share.h), all n columns, column-major in backend memory
/// with leading dimension ldy >= max(1, hi(q) - lo(q)); column j belongs to w[j].  That is n^2/P elements per
/// rank -- and the same per-rank memory in the solver itself (Q and the merge scratch are row slabs too, the
/// secular eigenvector strips are generated on the fly from O(K) data, so nothing of size n^2 is ever
/// replicated or communicated).  Collective over the handle's communicator.  Throws on a failed leaf solve or
/// secular root (consistently on all ranks).
template <typename rt> void pstedc(const DdlaHandle_t &handle, int n, rt *d, rt *e, rt *w, rt *Y, int ldy);

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
void pstedc(const DdlaHandle_t &handle, int n, rt *d, rt *e, rt *w, rt *Y, int ldy, rt *scratch,
            std::size_t scratch_elements);

} // namespace ddla

#include <ddla/ddla.h>
#include "row_share.h"
#include <cassert>
#include <cstddef>
#include <mpi.h>

namespace ddla
{
namespace spectral_impl
{

// Policy and alias decisions belong to the collective checker. The scratch
// below is a function of this POD alone, including on zero-row ranks.
struct PstedcVectorPlan
{
    DdlaBackend backend{};
    int n = 0, ranks = 1, rank = 0, threads = 1, leaf = 1024, split = 0, strip = 16, root_chunk = 32;
    int ldy = 1;
    bool borrow_q = false, borrow_q2 = false, profile = false;
};
template <class R>
PstedcVectorPlan pstedc_check(const DdlaHandle_t &, int, const R *, const R *, const R *, const R *, int,
                              const R * = nullptr, std::size_t = 0);
// Collective policy extraction for already-validated composite geometry.
// No input/output pointer is inspected; alias/borrow decisions remain with
// the enclosing checker. Never call this from the size function or the run.
template <class R> PstedcVectorPlan pstedc_policy(const DdlaHandle_t &, int);

// Reproduce LAED0's recursive integer subdivision without an owning tree.
inline int pstedc_leaf_geometry(int n, int leaves, int index, int *start = nullptr) noexcept
{
    int offset = 0;
    for (int bit = leaves / 2; bit; bit /= 2)
    {
        const int left = n / 2;
        if (index & bit)
        {
            offset += left;
            n -= left;
        }
        else
            n = left;
    }
    if (start)
        *start = offset;
    return n;
}
// Leaves of that subdivision: halved until the largest (the last) has at most leaf rows.
inline int pstedc_leaves(int n, int leaf) noexcept
{
    int leaves = 1;
    while (pstedc_leaf_geometry(n, leaves, leaves - 1) > leaf)
        leaves *= 2;
    return leaves;
}
// Leaves rank solves (rank, rank + ranks, ...).
inline int pstedc_my_leaves(int leaves, int rank, int ranks) noexcept
{
    return rank < leaves ? 1 + (leaves - 1 - rank) / ranks : 0;
}

// The scratch of pstedc_run on this rank (none for n == 0), in carving order.
// Backend: Q (this rank's rows of all n columns, unless the caller lends it) and, with more
// than one leaf, Q2 (the same, unless it is Y), the strip of S (n x strip) and on a GPU handle
// the uploaded vectors zhat | dlamda | dbase | tau | nrm (5n) and the Givens list (2n columns,
// 2n numbers); on a GPU handle the device permutation (n).
// Host: the leaf sizes and starts, indxq (n) and the offsets of this rank's leaves; then the
// leaf phase and the merge phase share the rest.  Leaf phase: this rank's leaf factors, the
// receive buffer of one leaf (P > 1), the leaf errors and the ?STEDC work and iwork of every
// leaf worker.  Merge phase (more than one leaf): z, dlamda, wv, givnum (2n), lam, wp, zhat,
// tau, dbase, nrm, the root records (4n), the GPU upload (5n), a chunk of root deltas
// (n x max(root_chunk, normalisation workers), reused by normalisation), zho and dlo
// (split GEMM); iq, perm, givcol (2n),
// indxp, indx, counts and displs (P), the chunk errors, order and perm2 (split
// GEMM); the column classes (split GEMM).
// On a CPU handle, where backend memory is host memory, the host part follows the backend part
// and *host_bytes is 0.
template <class R>
void pstedc_workspace_size(const DdlaHandle_t &, const PstedcVectorPlan &, std::size_t *backend_bytes,
                           std::size_t *host_bytes);

// Owns the reusable GPU copy fence and its pending-source-read state.
// Construct/destroy outside check, size and run. No event is needed
// for CPU or a single-leaf problem.
class PstedcResources
{
    DdlaHandle_t handle_ = nullptr;
    ddlaEvent_t uploaded_ = nullptr;
    bool pending_ = false;

  public:
    PstedcResources(const DdlaHandle_t &handle, const PstedcVectorPlan &p) : handle_(handle)
    {
        if (p.n && p.backend == DdlaBackend::GPU && pstedc_leaves(p.n, p.leaf) > 1)
            DDLA_CHECK(ddlaEventCreate(handle_, &uploaded_));
    }
    ~PstedcResources()
    {
        if (pending_)
            try
            {
                (void)ddlaEventSynchronize(handle_, uploaded_);
            }
            catch (...)
            {
            }
        if (uploaded_)
            try
            {
                (void)ddlaEventDestroy(handle_, uploaded_);
            }
            catch (...)
            {
            }
    }
    PstedcResources(const PstedcResources &) = delete;
    PstedcResources &operator=(const PstedcResources &) = delete;
    void wait()
    {
        if (pending_)
        {
            DDLA_CHECK(ddlaEventSynchronize(handle_, uploaded_));
            pending_ = false;
        }
    }
    void record()
    {
        assert(uploaded_);
        DDLA_CHECK(ddlaEventRecord(handle_, uploaded_));
        pending_ = true;
    }
    void completed() noexcept
    {
        pending_ = false;
    }
};

// scratch: the caller's storage for Q when p.borrow_q (else unused).
template <DdlaBackend B, class R>
void pstedc_run(const DdlaHandle_t &, const PstedcVectorPlan &, PstedcResources &, R *d, R *e, R *w, R *Y,
                R *scratch, void *backend_work, std::size_t backend_bytes, void *host_work,
                std::size_t host_bytes);

enum class PstedcMode
{
    values,
    vectors
};
struct PstedcPlan
{
    PstedcVectorPlan rows{};
    PstedcMode mode = PstedcMode::values;
    int keep_mb = 64;
};
// Policy-only helper for validated composite dimensions. Public check also
// validates actual pointers/strides and agrees the operation signature.
template <class R> PstedcPlan pstedc_policy(const DdlaHandle_t &, char, int, int);
template <class R> PstedcPlan pstedc_values_policy(const DdlaHandle_t &h, int n)
{
    return pstedc_policy<R>(h, 'N', n, 1);
}
template <class R>
PstedcPlan pstedc_check(const DdlaHandle_t &, char, int, const R *, const R *, const R *, const R *, int);
// Vectors use pstedc_workspace_size directly. Eigenvalues only retain
// O(n) boundary rows and a bounded root-delta cache, plus leaf scratch.
template <class R>
void pstedc_workspace_size(const DdlaHandle_t &, const PstedcPlan &, std::size_t *backend_bytes,
                           std::size_t *host_bytes);

// z: the eigenvector matrix for jobz == 'V' (nullptr for 'N'). Only a row-vector
// solve needs the prepared upload fence; values-only needs no resources.
template <DdlaBackend B, class R>
void pstedc_run(const DdlaHandle_t &, const PstedcPlan &, R *d, R *e, R *w, R *z, void *backend_work,
                std::size_t backend_bytes, void *host_work, std::size_t host_bytes,
                PstedcResources *rows = nullptr);

} // namespace spectral_impl
} // namespace ddla

#endif // DDLA_PSTEDC_IMPL_H
