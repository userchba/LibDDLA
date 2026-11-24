// Tridiagonal diagonalization by Cuppen divide and conquer. Eigenvector rows
// stay distributed as contiguous row slabs throughout the tree.
// d, e and the ascending eigenvalues remain replicated host vectors.
// pstedc('N') retains only first/last eigenvector rows needed by later merges.

#include "helpers.h"
#include <ddla/ddla.h>

#include "ddla_connector.h"
#include "comm_traits.h"
#include "dc_common.h"
#include "ddla_stream.h"
#include "host_parallel.h"
#include "eig_params.h"
#include "row_share.h"
#include "host_threads.h"
#include "mpi_datatype.h"
#include "plan_util.h"
#include "pstedc_impl.h"
#include "pstedc_geometry.h"
#include "pstedc_numeric.h"
#include "check_agree.h"
#include "BLAS/blas_impl.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// Host LAPACK through the Fortran ABI, in src/gemm.cpp's extern "C" style.
// The hidden trailing CHARACTER-length arguments are omitted: correct on this
// project's gfortran + x86-64 SysV toolchain (lengths travel in registers and
// an extra unread one is harmless), not portable to toolchains that pass them
// differently.  (src/pgesvd.cpp and src/SVD/pbdsdc.cpp refer here.)
//   ?STEDC  leaf solves
//   ?LAED8  deflation decisions (ICOMPQ = 0: never touches Q/Q2)
//   ?LAED4  one secular-equation root and its delta vector
//   ?LAMRG  merge permutation of two sorted runs (prototypes: dc_common.h)
extern "C" {
void sstedc_(const char*, const int*, float*, float*, float*, const int*, float*, const int*, int*,
             const int*, int*);
void dstedc_(const char*, const int*, double*, double*, double*, const int*, double*, const int*, int*,
             const int*, int*);
// (ICOMPQ, K, N, QSIZ, D, Q, LDQ, INDXQ, RHO, CUTPNT, Z, DLAMDA, Q2, LDQ2, W,
//  PERM, GIVPTR, GIVCOL, GIVNUM, INDXP, INDX, INFO)
void slaed8_(const int*, int*, const int*, const int*, float*, float*, const int*, int*, float*, const int*,
             float*, float*, float*, const int*, float*, int*, int*, int*, float*, int*, int*, int*);
void dlaed8_(const int*, int*, const int*, const int*, double*, double*, const int*, int*, double*,
             const int*, double*, double*, double*, const int*, double*, int*, int*, int*, double*, int*,
             int*, int*);
void slaed4_(const int*, const int*, const float*, const float*, float*, const float*, float*, int*);
void dlaed4_(const int*, const int*, const double*, const double*, double*, const double*, double*, int*);
}

namespace ddla {

namespace spectral_impl {

// Typed LAPACK dispatch (overloads on the real type).
void stedc(char compz, int n, float* d, float* e, float* z, int ldz, float* work, int lwork, int* iwork,
           int liwork, int& info)
{
    sstedc_(&compz, &n, d, e, z, &ldz, work, &lwork, iwork, &liwork, &info);
}

void stedc(char compz, int n, double* d, double* e, double* z, int ldz, double* work, int lwork, int* iwork,
           int liwork, int& info)
{
    dstedc_(&compz, &n, d, e, z, &ldz, work, &lwork, iwork, &liwork, &info);
}

namespace {
template<DdlaBackend B, class R>
void pstedc_local_run(const DdlaHandle_t& h, const PstedcVectorPlan& p, char job,
                      R* d, R* e, R* w, R* z, R* scratch, void* backend_work,
                      void* host_work, PstedcResources* resources)
{
    const auto local = pstedc_local_plan<R>(h, p, job);
    char* work = static_cast<char*>(backend_work);
    R* off = e;
    if constexpr (B == DdlaBackend::CPU) {
        off = detail::take<R>(work, std::max(1, p.n - 1));
        if (p.n > 1)
            std::copy_n(e, p.n - 1, off);
    }
    std::memmove(w, d, std::size_t(p.n) * sizeof(R));
    int info = 0;
    if constexpr (B == DdlaBackend::CPU) {
        PstedcResources unused(h, local.rows);
        stedc_run<B,R>(h, local, unused, w, off, z, p.ldy, info, work, host_work);
    } else {
        assert(resources);
        stedc_run<B,R>(h, local, *resources, w, off, z, p.ldy, info, work, host_work, scratch);
    }
    if (info)
        throw std::runtime_error("pstedc: local STEDC failed, info=" + std::to_string(info));
}
}

struct PstedcLocalResources
{
    StedcPlan plan;
    PstedcResources resources;
    PstedcLocalResources(const DdlaHandle_t& h, const PstedcVectorPlan& p)
        : plan(p.scalar_bytes == sizeof(float) ? pstedc_rank_plan<float>(h,p)
                                              : pstedc_rank_plan<double>(h,p)),
          resources(h, plan.rows) {}
};

PstedcResources::PstedcResources(const DdlaHandle_t& h, const PstedcVectorPlan& p) : handle_(h)
{
    // Prepare nested resources before creating our own event, so a failed
    // local preparation cannot leave an outer event behind.
    if (p.rank_leaf && p.rank < pstedc_leaves(p))
        local = std::make_unique<PstedcLocalResources>(h,p);
    if (p.n && p.backend == DdlaBackend::GPU && pstedc_leaves(p) > 1)
        DDLA_CHECK(ddlaEventCreate(handle_, &uploaded_));
}
PstedcResources::~PstedcResources()
{
    if (pending_)
        try { (void)ddlaEventSynchronize(handle_, uploaded_); } catch (...) {}
    if (uploaded_)
        try { (void)ddlaEventDestroy(handle_, uploaded_); } catch (...) {}
}

// ?LAED8 with ICOMPQ = 0: Q and Q2 are not referenced (a dummy is passed),
// but LDQ/LDQ2 must still be >= max(1, n).
template <typename rt, typename F>
void laed8_decide_impl(F fn, int& k, int n, rt* d, int* indxq, rt& rho, int cutpnt, rt* z, rt* dlamda, rt* w,
                       int* perm, int& givptr, int* givcol, rt* givnum, int* indxp, int* indx, int& info)
{
    const int icompq = 0;
    const int ld = n > 1 ? n : 1;
    rt dummy = 0;
    fn(&icompq, &k, &n, &n, d, &dummy, &ld, indxq, &rho, &cutpnt, z, dlamda, &dummy, &ld, w, perm, &givptr,
       givcol, givnum, indxp, indx, &info);
}

void laed8_decide(int& k, int n, float* d, int* indxq, float& rho, int cutpnt, float* z, float* dlamda,
                  float* w, int* perm, int& givptr, int* givcol, float* givnum, int* indxp, int* indx,
                  int& info)
{
    laed8_decide_impl<float>(slaed8_, k, n, d, indxq, rho, cutpnt, z, dlamda, w, perm, givptr, givcol, givnum,
                             indxp, indx, info);
}

void laed8_decide(int& k, int n, double* d, int* indxq, double& rho, int cutpnt, double* z, double* dlamda,
                  double* w, int* perm, int& givptr, int* givcol, double* givnum, int* indxp, int* indx,
                  int& info)
{
    laed8_decide_impl<double>(dlaed8_, k, n, d, indxq, rho, cutpnt, z, dlamda, w, perm, givptr, givcol,
                              givnum, indxp, indx, info);
}

void laed4(int n, int i, const float* d, const float* z, float* delta, const float& rho, float& dlam,
           int& info)
{
    slaed4_(&n, &i, d, z, delta, &rho, &dlam, &info);
}

void laed4(int n, int i, const double* d, const double* z, double* delta, const double& rho, double& dlam,
           int& info)
{
    dlaed4_(&n, &i, d, z, delta, &rho, &dlam, &info);
}

} // namespace spectral_impl

namespace {

// Both trees use the same scaling and rank-one tears. Keep the arithmetic
// passes in this order so their leaf inputs remain bit-for-bit identical.
template <typename rt>
inline rt pstedc_scale(int n, rt* d, rt* e)
{
    rt orgnrm = 0;
    for (int i = 0; i < n; ++i)
        orgnrm = std::max(orgnrm, static_cast<rt>(std::fabs(d[i])));
    for (int i = 0; i + 1 < n; ++i)
        orgnrm = std::max(orgnrm, static_cast<rt>(std::fabs(e[i])));
    if (orgnrm == rt(0))
        orgnrm = rt(1); // zero matrix: leaves give Y = I
    for (int i = 0; i < n; ++i)
        d[i] /= orgnrm;
    for (int i = 0; i + 1 < n; ++i)
        e[i] /= orgnrm;
    return orgnrm;
}

template <typename rt>
inline void pstedc_split_leaves(int n, int nsub, rt* d, const rt* e, int* sizes, int* start)
{
    for (int i = 0; i < nsub; ++i)
        sizes[i] = spectral_impl::pstedc_leaf_geometry(n, nsub, i, start + i);
    for (int i = 1; i < nsub; ++i) {
        const int b = start[i];
        const rt ae = std::fabs(e[b - 1]);
        d[b - 1] -= ae;
        d[b] -= ae;
    }
}

// Deflation must agree before either tree enters data-dependent collectives.
inline void pstedc_agree_deflation(MPI_Comm comm, int info, int K)
{
    int tri[3] = {info != 0 ? 1 : 0, K, -K};
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, tri, 3, MPI_INT, MPI_MAX, comm));
    if (tri[0] != 0)
        throw std::runtime_error("pstedc: ?LAED8 failed (info=" + std::to_string(info) + ")");
    if (tri[1] != K || tri[2] != -K)
        throw std::runtime_error("pstedc: ?LAED8 K disagreement across ranks (mine " +
                                 std::to_string(K) + ", max " + std::to_string(tri[1]) + ")");
}


// Merges with fewer secular roots per rank than this are not split: the
// all-reduce and broadcast latency would exceed the O(K^2) host work saved.
constexpr int kMinRootsPerRank = 32;

// Classify the merge block's columns for the CPU split GEMM.  cls[c] (block
// position c, 0-based) starts as the child the column came from (0 = left
// [0, m1), 1 = right); walking the Givens list in order, a rotation pairing
// the halves -- or touching an already mixed column -- marks both columns
// mixed (2), because the pair then has support in both children's rows.
// Rotations within one half with two unmixed columns preserve the support.
void dc_split_classes(const int* givcol, int givptr, int m, int m1, char* cls)
{
    for (int c = 0; c < m; ++c)
        cls[c] = (c < m1) ? 0 : 1;
    for (int g = 0; g < givptr; ++g) {
        const int c1 = givcol[2 * g] - 1;
        const int c2 = givcol[2 * g + 1] - 1;
        if (c1 < 0 || c2 < 0 || c1 >= m || c2 >= m)
            continue; // defensive
        if (cls[c1] == 2 || cls[c2] == 2 || (cls[c1] ^ cls[c2]) != 0) {
            cls[c1] = 2;
            cls[c2] = 2;
        }
    }
}

struct PhaseClock
{
    using clock = std::chrono::steady_clock;
    clock::time_point t0 = clock::now();

    double lap()
    {
        const clock::time_point t1 = clock::now();
        const double s = std::chrono::duration<double>(t1 - t0).count();
        t0 = t1;
        return s;
    }
};

// ---------------------------------------------------------------------------
// DcRowsStore: the eigenvalue-only store (jobz == 'N' on a CPU handle).  A
// merge only ever reads the FIRST and LAST ROW of a solved block's eigenvector
// matrix (z = [last row of the left child ; first row of the right child]) and
// only needs those two rows of the merged block, which are the two rows of Q2
// times S.  So instead of Q (n x n) this store keeps F_ and L_ (length n each):
// F_[s+j] / L_[s+j] are the first / last row entry of column j of the block
// that starts at s, indexed by the block's own physical column positions
// exactly like Q's columns.  State is O(n) (plus the leaf scratch); there is
// no merge GEMM and no Q2. Host-only: it needs
// the communicator and nothing else, so it exists in every build.
// ---------------------------------------------------------------------------
template <typename rt>
class DcRowsStore
{
  public:
    // No backend buffers.  A single leaf needs no rows at all (?STEDC('N')).
    DcRowsStore(const DdlaHandle_t& h, const spectral_impl::PstedcPlan& p)
        : h_(h), n_(p.rows.n), P_(p.rows.ranks), leaves_(spectral_impl::pstedc_leaves(p.rows.n, p.rows.leaf))
    {
    }

    // F and L
    void take_kept(char*& kept)
    {
        const std::size_t rows = leaves_ > 1 ? n_ : 0;
        F_ = detail::take<rt>(kept, rows);
        L_ = detail::take<rt>(kept, rows);
    }

    // the leaf block; Fb, Lb, F2 and L2
    void take_phases(char*& leaf, char*& merge)
    {
        if (leaves_ <= 1)
            return;
        const std::size_t m = spectral_impl::pstedc_leaf_geometry(n_, leaves_, leaves_ - 1);
        Z_ = detail::take<rt>(leaf, m * m);
        Fb_ = detail::take<rt>(merge, n_);
        Lb_ = detail::take<rt>(merge, n_);
        F2_ = detail::take<rt>(merge, n_);
        L2_ = detail::take<rt>(merge, n_);
    }

    static constexpr const char* kTag = "[pstedc_cpu_vals]";

    void initialize()
    {
        const std::size_t rows = leaves_ > 1 ? n_ : 0;
        std::fill_n(F_, rows, rt(0));
        std::fill_n(L_, rows, rt(0));
    }

    // host scratch the leaf's ?STEDC('I') writes into (m x m, ld m); only its
    // first and last row survive (leaf_commit)
    rt* leaf_target(int m)
    {
        std::fill_n(Z_, std::size_t(m) * m, rt(0));
        return Z_;
    }

    void leaf_commit(int s0, int m)
    {
        for (int k = 0; k < m; ++k) {
            F_[static_cast<std::size_t>(s0) + k] = Z_[static_cast<std::size_t>(k) * m];
            L_[static_cast<std::size_t>(s0) + k] = Z_[(m - 1) + static_cast<std::size_t>(k) * m];
        }
    }

    // 2m numbers per leaf instead of the m x m block
    void share_leaves(const int* sizes, const int* start, int nsub)
    {
        if (P_ <= 1)
            return;
        const MPI_Datatype mt = detail::mpi_datatype<rt>();
        for (int i = 0; i < nsub; ++i) {
            const int m = sizes[static_cast<std::size_t>(i)];
            const std::size_t s0 = static_cast<std::size_t>(start[static_cast<std::size_t>(i)]);
            MPI_CHECK(MPI_Bcast(F_ + s0, m, mt, i % P_, h_->comm));
            MPI_CHECK(MPI_Bcast(L_ + s0, m, mt, i % P_, h_->comm));
        }
        // The overlaid leaf scratch is dead after this broadcast.
    }

    // (the driver calls read_z before rotate_gather: m1 is remembered for it)
    void read_z(int s, int m, int m1, rt* z)
    {
        m1_ = m1;
        for (int j = 0; j < m1; ++j)
            z[j] = L_[static_cast<std::size_t>(s) + j];
        for (int j = m1; j < m; ++j)
            z[j] = F_[static_cast<std::size_t>(s) + j];
    }

    // Fb = first row of blkdiag(Q_L, Q_R) = [f_L ; 0], Lb = last row = [0 ; l_R],
    // built in ZEROED scratch (F_[s+m1:s+m) still holds f_R and L_[s:s+m1)
    // still holds l_L, which a cross-half Givens rotation must not pull in);
    // then the deflation rotations (the same column operation as
    // the eigenvector row engine) and F2/L2 = Fb/Lb(PERM).
    void rotate_gather(int s, int m, int givptr, const int* givcol, const rt* givnum, const int* perm)
    {
        for (int j = 0; j < m; ++j) {
            const bool left = j < m1_;
            Fb_[static_cast<std::size_t>(j)] = left ? F_[static_cast<std::size_t>(s) + j] : rt(0);
            Lb_[static_cast<std::size_t>(j)] = left ? rt(0) : L_[static_cast<std::size_t>(s) + j];
        }
        for (int g = 0; g < givptr; ++g) {
            const std::size_t a = static_cast<std::size_t>(givcol[2 * g] - 1);
            const std::size_t b = static_cast<std::size_t>(givcol[2 * g + 1] - 1);
            const rt c = givnum[2 * g];
            const rt ss = givnum[2 * g + 1];
            const rt fx = Fb_[a], fy = Fb_[b];
            Fb_[a] = c * fx + ss * fy;
            Fb_[b] = c * fy - ss * fx;
            const rt lx = Lb_[a], ly = Lb_[b];
            Lb_[a] = c * lx + ss * ly;
            Lb_[b] = c * ly - ss * lx;
        }
        for (int j = 0; j < m; ++j) {
            F2_[static_cast<std::size_t>(j)] = Fb_[static_cast<std::size_t>(perm[j] - 1)];
            L2_[static_cast<std::size_t>(j)] = Lb_[static_cast<std::size_t>(perm[j] - 1)];
        }
    }

    // nothing to solve: the sorted, rotated rows are the merged rows
    void restore_all(int s, int m)
    {
        for (int j = 0; j < m; ++j) {
            F_[static_cast<std::size_t>(s) + j] = F2_[static_cast<std::size_t>(j)];
            L_[static_cast<std::size_t>(s) + j] = L2_[static_cast<std::size_t>(j)];
        }
    }

    // the new rows: fn/ln for the K secular vectors, then the deflated entries
    void place_rows(int s, int m, int K, const rt* fn, const rt* ln)
    {
        for (int j = 0; j < K; ++j) {
            F_[static_cast<std::size_t>(s) + j] = fn[j];
            L_[static_cast<std::size_t>(s) + j] = ln[j];
        }
        for (int j = K; j < m; ++j) {
            F_[static_cast<std::size_t>(s) + j] = F2_[static_cast<std::size_t>(j)];
            L_[static_cast<std::size_t>(s) + j] = L2_[static_cast<std::size_t>(j)];
        }
    }

    const rt* F2() const
    {
        return F2_;
    }

    const rt* L2() const
    {
        return L2_;
    }

  private:
    const DdlaHandle_t& h_;
    int n_;
    int P_ = 1;
    int leaves_ = 1;
    int m1_ = 0;
    rt* F_ = nullptr; // first / last row of every solved block
    rt* L_ = nullptr;
    rt* Fb_ = nullptr; // the merged block's rows before rotation
    rt* Lb_ = nullptr;
    rt* F2_ = nullptr; // ... after rotation and the PERM gather
    rt* L2_ = nullptr;
    rt* Z_ = nullptr; // leaf scratch
};

// Eigenvalues-only driver: retain boundary rows rather than full matrices.

// ---------------------------------------------------------------------------
// The row-slab solver (pstedc, src/pstedc_impl.h): the same Cuppen tree and the same host decisions as
// the boundary-row driver above, with the full eigenvector matrix distributed by ROWS. One template for both
// backends: only the data movement and the three bulk kernels (Givens, column gather, strip generation + GEMM)
// differ -- device kernels and cuBLAS on GPU handles, row-chunked host loops, parallel column copies and the
// BLAS gemm on CPU handles.
//
// Layout.  Rank q owns the rows [lo(q), hi(q)) of Q (detail::RowShare, all n columns), a (nrl x n) column-major
// slab with leading dimension nrl (device memory on GPU handles, host memory on CPU handles).  Q is block
// diagonal during the tree exactly as before: a solved subproblem [s, s+m) owns the dense block
// Q(s:s+m, s:s+m), and a merge touches only that block -- on each rank only the rows of the block that the rank
// owns.  Every operation on the block is a combination of ROWS (a Givens rotation mixes two columns row by row,
// the column permutation gathers whole columns, the merge product C = Q2 * S is row-independent), so all of
// them are rank-local on the slab:
//
//   z       the two rows s+m1-1 and s+m1 live on their owners; each copies its piece out and one
//           sum-allreduce (all other entries are zeros: exact) replicates z -- O(m) numbers
//   decide  ?LAED8 on every rank, identical inputs (as before)
//   rotate  Givens + column gather on the rank's rows of the block
//   roots   a contiguous share per rank when it pays; per root the LAED4 delta gives the Loewner partial product
//           and the root's ORIGIN pole and tau, delta(origin) = -tau (LAED4's own representation:
//           delta(k) = (dlamda(k) - dlamda(origin)) - tau).  Roots are processed in chunks: a chunk's deltas are
//           computed in parallel, then its Loewner updates are applied partitioned over the ROWS, each element
//           multiplying the roots in ascending order -- so the result does not depend on the thread count
//   z-hat   one MPI_PROD all-reduce (the distributed ?LAED9); then each rank normalises its roots (it rebuilds
//           delta from (origin, tau), which costs no memory) and one all-gather shares (lambda, dbase, tau, nrm)
//   S       NEVER stored: the eigenvector strips S(:, J0:J0+W) = zhat ./ delta ./ nrm are generated from those
//           O(K) vectors (ELPA builds its strips the same way), W <= 512 columns at a time
//   GEMM    Q(rows of the block, J0:J0+W) = Q2(rows, 1:K) * S strip -- local, written straight into Q
//           (Q2 is a separate copy of the permuted block, so the update is not in place)
//
// Per-rank memory: Q and Q2 slabs (n^2/P each) plus the strip (n x W) and O(n) vectors.  Communication: a few
// O(K) collectives per merge, plus one broadcast per leaf (n * leaf elements in total).  Nothing of size n^2 is
// replicated or exchanged.  A rank that owns no row of a merge's block skips its bulk work but joins its
// collectives, so merges of disjoint row ranges overlap in time across ranks.
// ---------------------------------------------------------------------------

#if DDLA_HAS_GPU

// S(k, jj) = ((zhat_k / ((dlamda_k - dbase_j) - tau_j)) / nrm_j) for the strip of roots j = j0 + jj
template <typename rt>
__global__ void pstedc_fill_s_kernel(rt* __restrict__ S, int lds, const rt* __restrict__ zhat,
                                     const rt* __restrict__ dlamda, const rt* __restrict__ dbase,
                                     const rt* __restrict__ tau, const rt* __restrict__ nrm, int K, int j0,
                                     int w)
{
    const long long total = static_cast<long long>(K) * w;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int k = static_cast<int>(idx % K);
        const int jj = static_cast<int>(idx / K);
        const int j = j0 + jj;
        const rt dk = (dlamda[k] - dbase[j]) - tau[j];
        S[static_cast<std::size_t>(jj) * lds + k] = (zhat[k] / dk) / nrm[j];
    }
}

#endif // CUDA/HIP

} // namespace

namespace spectral_impl {
template <DdlaBackend B, typename rt>
void stedc_rows_run(const DdlaHandle_t& handle, const PstedcVectorPlan& policy, PstedcResources& resources,
                     rt* d, rt* e, rt* w, rt* dY, rt* scratch, void* backend_work,
                     [[maybe_unused]] std::size_t backend_bytes, void* host_work,
                     [[maybe_unused]] std::size_t host_bytes)
{
    using detail::take;
    constexpr bool kGpu = (B == DdlaBackend::GPU);
    const int n = policy.n, ldy = policy.ldy;
    if (!n)
        return;
    MPI_Comm comm = handle->comm;
    const int P = policy.ranks, me = policy.rank;
    const MPI_Datatype mt = detail::mpi_datatype<rt>();
#if DDLA_HAS_GPU
    gpuStream_t st = nullptr;
    if constexpr (kGpu)
        st = static_cast<gpuStream_t>(handle->stream);
#endif
    auto sync = [&]() {
#if DDLA_HAS_GPU
        if constexpr (kGpu)
            DDLA_CHECK(ddlaSynchronize(handle));
#endif
    };
    // host threads of this rank (collective on its first use); every threaded loop below keeps the per-element
    // order of the serial one, so the results do not depend on the thread count.  GPU runs follow the same
    // OMP_NUM_THREADS / DDLA_EIG_THREADS rule (the cluster jobs pin it to 1).
    const int nth = policy.threads;
    const bool prof = policy.profile;
    PhaseClock clk;
    double t_leaf = 0, t_decide = 0, t_roots = 0, t_zhat = 0, t_dev = 0;
    int n_merge = 0, n_split = 0;
    long long sum_k = 0, top_k = 0;

    const detail::RowShare rs(n, P);
    const int lo = rs.lo(me);
    const int nrl = rs.count(me);
    const int ld = std::max(nrl, 1);


    // ---- scale by the max-norm (as ?STEDC does per unreduced block) ---------
    const rt orgnrm = pstedc_scale(n, d, e);

    // ---- scratch (pstedc_workspace_size) ---------------------------------------------------------------
    // The backend buffers, then the host arrays: host memory on a GPU handle, after the backend buffers on a CPU
    // handle (where backend memory is host memory).  The leaf phase and the merge phase overlay.
    const int nsub = pstedc_leaves(policy), mine = pstedc_my_leaves(nsub, me, P);
    const int norm_workers = std::max(1, std::min(nth, n));
    const int Wcap = policy.strip, kRootChunk = policy.root_chunk, split_mode = policy.split;
    const std::size_t slab = std::size_t(ld) * n;
    char* const backend_base = static_cast<char*>(backend_work);
    char* bw = backend_base;
    rt* const dQ = policy.borrow_q ? scratch : take<rt>(bw, slab);
    rt* dQ2 = nullptr;
    rt* dS = nullptr;   // the strip: S(:, J0:J0+W), device memory on GPU handles, host memory on CPU handles
    rt* dVec = nullptr; // GPU: zhat | dlamda | dbase | tau | nrm
    int* dGcol = nullptr; // GPU
    rt* dGnum = nullptr;  // GPU
    int* dPerm = nullptr; // GPU
    if (nsub > 1)
        dQ2 = policy.borrow_q2 ? dY : take<rt>(bw, slab);
    char* const tail = bw;
    if (nsub > 1) {
        dS = take<rt>(bw, std::size_t(n) * Wcap);
        if constexpr (kGpu) {
            dVec = take<rt>(bw, 5 * std::size_t(n));
            dGcol = take<int>(bw, 2 * std::size_t(n));
            dGnum = take<rt>(bw, 2 * std::size_t(n));
        }
    }
    if constexpr (kGpu)
        dPerm = take<int>(bw, n);
    if constexpr (kGpu)
        if (resources.local)
            bw = tail + std::max(std::size_t(bw - tail), resources.local->plan.backend_bytes);
    char* const host_base = kGpu ? static_cast<char*>(host_work) : bw;
    [[maybe_unused]] const auto fits = [&](const char* end) {
        return kGpu ? std::size_t(bw - backend_base) <= backend_bytes &&
                          std::size_t(end - host_base) <= host_bytes
                    : std::size_t(end - backend_base) <= backend_bytes;
    };
    char* hw = host_base;
    int* const sizes = take<int>(hw, nsub);
    int* const start = take<int>(hw, nsub);
    int* const indxq = take<int>(hw, n);
    std::size_t* const offsets = take<std::size_t>(hw, mine + 1);
    char* const phase = hw;

    for (int i = 0; i < nsub; ++i)
        sizes[i] = pstedc_leaf_geometry(policy, i, start + i);
    for (int i = 1; i < nsub; ++i) {
        const int cut = start[i];
        const rt rho = std::abs(e[cut - 1]);
        d[cut - 1] -= rho;
        d[cut] -= rho;
    }

    // The outer Q slab is idle workspace until the rank-local solve finishes.
    if (!policy.rank_leaf)
        DDLA_CHECK(ddlaMemset(handle, dQ, 0, slab * sizeof(rt)));

    // ---- local solves and bounded publication -------------------------------
    if (policy.rank_leaf) {
        // Each nonempty rank makes exactly one backend-selected local solve.
        // Q is idle until publication; Q2 is idle until the outer merges.
        detail::agree_local(handle, "pstedc: local STEDC", [&] {
            if (!resources.local) return;
            auto& local = *resources.local;
            const int m = sizes[me], first = start[me];
            void* work = tail;
            if constexpr (B != DdlaBackend::GPU)
                work = local.plan.backend_bytes <= detail::padded_bytes<rt>(slab)
                    ? static_cast<void*>(dQ) : static_cast<void*>(phase);
            int info = 0;
            stedc_run<B,rt>(handle,local.plan,local.resources,d + first,e + first,
                dQ2,m,info,work,phase,dQ);
            if (info) throw std::runtime_error("pstedc: local STEDC failed, info=" + std::to_string(info));
        });
        DDLA_CHECK(ddlaMemset(handle,dQ,0,slab*sizeof(rt)));
        char* publication = phase;
        int* const counts = take<int>(publication,P);
        int* const displacements = take<int>(publication,P);
        auto* const panel = reinterpret_cast<rt*>(publication);
        for (int i = 0; i < nsub; ++i) {
            const int m = sizes[i], first = start[i];
            MPI_CHECK(MPI_Bcast(d + first,m,mt,i,comm));
            const int width = std::min(Wcap,std::max(1,INT_MAX / m));
            const int ra = std::max(lo,first), rb = std::min(lo + nrl,first + m);
            // Most rows already belong to the local solver's rank. Keep them
            // on the backend; only block-rounding boundary rows need transport.
            if (me == i && rb > ra)
                DDLA_CHECK(ddlaMemcpy2D(handle,dQ + (ra-lo) + std::size_t(first)*ld,
                    std::size_t(ld)*sizeof(rt),dQ2 + (ra-first),std::size_t(m)*sizeof(rt),
                    std::size_t(rb-ra)*sizeof(rt),m));
            for (int j = 0; j < m; j += width) {
                const int columns = std::min(width,m-j);
                int total = 0;
                for (int q = 0; q < P; ++q) {
                    const int begin = std::max(rs.lo(q),first);
                    const int end = std::min(rs.hi(q),first+m);
                    counts[q] = q == i ? 0 : std::max(0,end-begin)*columns;
                    displacements[q] = total;
                    total += counts[q];
                    if (me == i && counts[q])
                        DDLA_CHECK(ddlaMemcpy2D(handle,panel + displacements[q],
                            std::size_t(end-begin)*sizeof(rt),
                            dQ2 + (begin-first) + std::size_t(j)*m,std::size_t(m)*sizeof(rt),
                            std::size_t(end-begin)*sizeof(rt),columns));
                }
                if (!total) continue;
                if (me == i) sync(); // MPI must see the completed host pack.
                MPI_CHECK(MPI_Scatterv(panel,counts,displacements,mt,
                    me == i ? MPI_IN_PLACE : panel,counts[me],mt,i,comm));
                if (counts[me]) {
                    DDLA_CHECK(ddlaMemcpy2D(handle,dQ + (ra-lo) + std::size_t(first+j)*ld,
                        std::size_t(ld)*sizeof(rt),panel,std::size_t(rb-ra)*sizeof(rt),
                        std::size_t(rb-ra)*sizeof(rt),columns));
                    sync(); // The next scatter overwrites the host receive panel.
                }
            }
            for (int j = 0; j < m; ++j) indxq[first+j] = j+1;
        }
    } else
    {
        const int maxsz = pstedc_leaf_geometry(n, nsub, nsub - 1), workers = std::min(nth, mine);
        const std::size_t leaf_work_count = std::size_t(maxsz) * maxsz + 6 * std::size_t(maxsz) + 1;
        const std::size_t leaf_iwork_count = 5 * std::size_t(maxsz) + 3;
        offsets[0] = 0;
        for (int j = 0; j < mine; ++j) {
            const int m = sizes[me + j * P];
            offsets[j + 1] = offsets[j] + std::size_t(m) * m;
        }
        // the leaf phase: this rank's leaf factors, one received leaf, the errors, the ?STEDC scratch per worker
        char* lw = phase;
        rt* const blocks = take<rt>(lw, offsets[mine]);
        rt* const recv = take<rt>(lw, P > 1 ? std::size_t(maxsz) * maxsz : 0);
        int* const errs = take<int>(lw, mine);
        rt* const work = take<rt>(lw, std::size_t(workers) * leaf_work_count);
        int* const iwork = take<int>(lw, std::size_t(workers) * leaf_iwork_count);
        assert(fits(lw));
        {
            std::optional<detail::ScopedBlasThreads> serial_blas;
            if constexpr (B != DdlaBackend::GPU)
                if (nth > 1 && mine > 1)
                    serial_blas.emplace(1);
            detail::host_parallel_for(workers, 1, nth, [&](long long wa, long long wb) {
                for (long long worker = wa; worker < wb; ++worker) {
                    rt* base = work + std::size_t(worker) * leaf_work_count;
                    rt* dl = base;
                    rt* el = dl + maxsz;
                    rt* lapack = el + maxsz;
                    int* iw = iwork + std::size_t(worker) * leaf_iwork_count;
                    for (int j = int(worker); j < mine; j += workers) {
                        const int i = me + j * P, m = sizes[i], s0 = start[i];
                        std::copy_n(d + s0, m, dl);
                        std::fill_n(el, m, rt(0));
                        std::copy_n(e + s0, std::max(0, m - 1), el);
                        rt* block = blocks + offsets[j];
                        // STEDC('I') initializes its output eigenvectors.
                        int li = 0;
                        stedc('I', m, dl, el, block, m, lapack, int(leaf_work_count - 2 * maxsz), iw,
                              int(leaf_iwork_count), li);
                        std::copy_n(dl, m, d + s0);
                        errs[j] = li;
                    }
                }
            });
        }
        int leaf_err = 0;
        for (int j = 0; j < mine; ++j)
            if (errs[j] && !leaf_err)
                leaf_err = errs[j];
        if (P > 1)
            agree_or_throw(comm, leaf_err, "pstedc", "leaf ?STEDC");
        else if (leaf_err)
            throw std::runtime_error("stedc: leaf ?STEDC failed (info=" + std::to_string(leaf_err) + ")");
        if (P > 1)
            for (int i = 0; i < nsub; ++i)
                MPI_CHECK(MPI_Bcast(d + start[i], sizes[i], mt, i % P, comm));
        for (int i = 0; i < nsub; ++i) {
            const int m = sizes[i], s0 = start[i], root = i % P;
            rt* blk = me == root ? blocks + offsets[i / P] : recv;
            if (P > 1)
                MPI_CHECK(MPI_Bcast(blk, m * m, mt, root, comm));
            const int ra = std::max(lo, s0), rb = std::min(lo + nrl, s0 + m);
            if (rb > ra) {
                DDLA_CHECK(ddlaMemcpy2D(handle, dQ + (ra - lo) + std::size_t(s0) * ld,
                                        std::size_t(ld) * sizeof(rt), blk + (ra - s0),
                                        std::size_t(m) * sizeof(rt), std::size_t(rb - ra) * sizeof(rt),
                                        std::size_t(m)));
                // The next broadcast reuses this host leaf block.
                if constexpr (kGpu)
                    DDLA_CHECK(ddlaSynchronize(handle));
            }
            for (int j = 0; j < m; ++j)
                indxq[s0 + j] = j + 1;
        }
    }
    t_leaf = clk.lap();

    // ---- merges ---------------------------------------------------------------
    // the merge phase (nothing with one leaf); the leaf phase it overlays is dead
    char* mw = phase;
    const auto rv = [&](std::size_t count) {
        return take<rt>(mw, nsub > 1 ? count : 0);
    };
    const auto iv = [&](std::size_t count) {
        return take<int>(mw, nsub > 1 ? count : 0);
    };
    const std::size_t nn = n, split_n = split_mode ? nn : 0;
    rt* const z = rv(nn);
    rt* const dlamda = rv(nn);
    rt* const wv = rv(nn);
    rt* const givnum = rv(2 * nn);
    rt* const lam = rv(nn);
    rt* const wp = rv(nn);
    rt* const zhat = rv(nn);
    rt* const tauv = rv(nn);
    rt* const dbasev = rv(nn);
    rt* const nrmv = rv(nn);
    rt* const rec = rv(4 * nn);
    [[maybe_unused]] rt* const up = rv(kGpu ? 5 * nn : 0); // GPU
    // Root deltas expire before normalisation; both passes reuse this storage.
    rt* const dchunk = rv(nn * std::max(kRootChunk, norm_workers));
    rt* const normcols = dchunk;
    rt* const zho = rv(split_n);
    rt* const dlo = rv(split_n);
    int* const iq = iv(nn);
    int* const perm = iv(nn);
    int* const givcol = iv(2 * nn);
    int* const indxp = iv(nn);
    int* const indx = iv(nn);
    int* const rcount = iv(P);
    int* const rdispl = iv(P);
    int* const ce = iv(kRootChunk);
    int* const order = iv(split_n);
    int* const perm2 = iv(split_n);
    char* const cls = take<char>(mw, nsub > 1 ? split_n : 0);
    assert(fits(mw));
    rt S2[4] = {0, 0, 0, 0};

    auto merge = [&](int s, int m, int m1) {
        // the rows of the block this rank owns, as local rows of its slab
        const int ra = std::max(lo, s), rb = std::min(lo + nrl, s + m);
        const int mr = std::max(rb - ra, 0);
        const int r0 = ra - lo;
        rt* const Qb = dQ + r0 + static_cast<std::size_t>(s) * ld;
        rt* const Q2b = dQ2 + r0;

        // z: last row of the left child's columns, first row of the right's -- each from the rank that owns
        // that row; every other entry of the sum is zero
        {
            std::fill(z, z + m, rt(0));
            const int rowA = s + m1 - 1, rowB = s + m1;
            bool pending = false;
            auto get_row = [&](rt* zdst, int row_g, int col0, int count) {
                const rt* src = dQ + (row_g - lo) + static_cast<std::size_t>(col0) * ld;
                if constexpr (kGpu) {
#if DDLA_HAS_GPU
                    DDLA_CHECK(ddlaMemcpy2D(handle, zdst, sizeof(rt), src,
                                            static_cast<std::size_t>(ld) * sizeof(rt), sizeof(rt),
                                            static_cast<std::size_t>(count)));
#endif
                } else {
                    for (int j = 0; j < count; ++j)
                        zdst[j] = src[static_cast<std::size_t>(j) * ld];
                }
                pending = true;
            };
            if (rs.owner(rowA) == me)
                get_row(z, rowA, s, m1);
            if (rs.owner(rowB) == me)
                get_row(z + m1, rowB, s + m1, m - m1);
            if (pending)
                sync();
            if (P > 1)
                MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, z, m, mt, MPI_SUM, comm));
        }

        // Protect host metadata uploaded by the previous merge before LAED8
        // rewrites its host arrays. The event excludes later merge GEMMs.
        if constexpr (kGpu)
            resources.wait();
        // decide (identical on every rank: identical inputs, scalar code)
        rt rho = e[s + m1 - 1];
        std::copy(indxq + s, indxq + s + m, iq);
        int K = 0, givptr = 0, i8 = 0;
        laed8_decide(K, m, d + s, iq, rho, m1, z, dlamda, wv, perm, givptr, givcol, givnum, indxp, indx, i8);
        if (P > 1)
            pstedc_agree_deflation(comm, i8, K);
        else if (i8)
            throw std::runtime_error("stedc: ?LAED8 failed (info=" + std::to_string(i8) + ")");
        t_decide += clk.lap();

        // the split GEMM reorders the kept columns [left | mixed | right] (and S's rows with them); the skipped GEMM blocks are
        // exact zeros, so the result is the same up to summation order.  K <= 2 keeps the explicit pole-ordered K x K S.
        const int do_split = (split_mode != 0 && K > 2) ? split_mode : 0;
        int kl = K, km = 0;
        if (do_split) {
            dc_split_classes(givcol, givptr, m, m1, cls);
            int n_l = 0, n_r = K;
            for (int I = 0; I < K; ++I) {
                const char c = cls[static_cast<std::size_t>(perm[static_cast<std::size_t>(I)] - 1)];
                if (c == 0)
                    order[static_cast<std::size_t>(n_l++)] = I;
                else if (c == 1)
                    order[static_cast<std::size_t>(--n_r)] = I;
            }
            kl = n_l;
            km = n_r - n_l;
            for (int I = 0; I < K; ++I)
                if (cls[static_cast<std::size_t>(perm[static_cast<std::size_t>(I)] - 1)] == 2)
                    order[static_cast<std::size_t>(n_l++)] = I;
            for (int j = 0; j < K; ++j)
                perm2[static_cast<std::size_t>(j)] =
                    perm[static_cast<std::size_t>(order[static_cast<std::size_t>(j)])];
            for (int j = 0; j < K; ++j)
                perm[static_cast<std::size_t>(j)] = perm2[static_cast<std::size_t>(j)];
        }

        // the deflation rotations, then Q2 = Qblk(:, PERM) -- on the rows this rank owns
        if (mr > 0) {
            if constexpr (kGpu) {
#if DDLA_HAS_GPU
                if (givptr > 0) {
                    DDLA_CHECK(ddlaMemcpy(handle, dGcol, givcol,
                                          2 * static_cast<std::size_t>(givptr) * sizeof(int)));
                    DDLA_CHECK(
                        ddlaMemcpy(handle, dGnum, givnum, 2 * static_cast<std::size_t>(givptr) * sizeof(rt)));
                    dc_givens_kernel<rt><<<(mr + 255) / 256, 256, 0, st>>>(Qb, ld, mr, dGcol, dGnum, givptr);
                    DDLA_CHECK(ddlaGetLastError(handle));
                }
                DDLA_CHECK(ddlaMemcpy(handle, dPerm, perm, static_cast<std::size_t>(m) * sizeof(int)));
                resources.record();
                launch_gather<rt>(handle, Q2b, ld, Qb, ld, mr, dPerm, m);
#endif
            } else {
                // rows are independent: each chunk of rows applies the whole rotation list to its rows (the
                // two columns of a rotation are contiguous runs of the chunk)
                if (givptr > 0)
                    detail::host_parallel_for(mr, 256, nth, [&](long long ra0, long long ra1) {
                        for (int g = 0; g < givptr; ++g) {
                            rt* pa =
                                Qb +
                                static_cast<std::size_t>(givcol[2 * static_cast<std::size_t>(g)] - 1) * ld;
                            rt* pb = Qb + static_cast<std::size_t>(
                                              givcol[2 * static_cast<std::size_t>(g) + 1] - 1) *
                                              ld;
                            const rt c = givnum[2 * static_cast<std::size_t>(g)];
                            const rt sn = givnum[2 * static_cast<std::size_t>(g) + 1];
                            for (long long r = ra0; r < ra1; ++r) {
                                const rt x = pa[r], y = pb[r];
                                pa[r] = c * x + sn * y;
                                pb[r] = c * y - sn * x;
                            }
                        }
                    });
                detail::host_parallel_for(m, 8, nth, [&](long long ca, long long cb) {
                    for (long long c = ca; c < cb; ++c)
                        std::memcpy(Q2b + static_cast<std::size_t>(c) * ld,
                                    Qb + static_cast<std::size_t>(perm[static_cast<std::size_t>(c)] - 1) * ld,
                                    static_cast<std::size_t>(mr) * sizeof(rt));
                });
            }
        }
        ++n_merge;
        sum_k += K;
        top_k = K;

        if (K == 0) {
            // nothing to solve: ?LAED8 sorted D and PERM sorts the columns
            if (mr > 0)
                DDLA_CHECK(ddlaMemcpy2D(handle, Qb, static_cast<std::size_t>(ld) * sizeof(rt), Q2b,
                                        static_cast<std::size_t>(ld) * sizeof(rt),
                                        static_cast<std::size_t>(mr) * sizeof(rt),
                                        static_cast<std::size_t>(m)));
            for (int k = 0; k < m; ++k)
                indxq[static_cast<std::size_t>(s) + k] = k + 1;
            return;
        }

        // roots: a contiguous share per rank when it pays
        const bool split = (P > 1) && (K > 2) && (K >= kMinRootsPerRank * P);
        auto jlo = [&](int p) {
            return split ? static_cast<int>(static_cast<long long>(K) * p / P) : 0;
        };
        const int J0 = jlo(me);
        const int J1 = split ? jlo(me + 1) : K;
        const int nJ = J1 - J0;
        if (split)
            ++n_split;

        // pass 1: the roots (LAED4), each one's origin pole and tau, and my share of the Loewner product.  A chunk
        // of roots is solved in parallel (private delta columns), then the chunk's Loewner updates are applied
        // partitioned over the ROWS I: wp[I] multiplies its roots in ascending order, exactly the serial order,
        // so the product is bitwise independent of the thread count
        if (K > 2)
            std::fill(wp, wp + K, rt(1));
        int e4 = 0;
        for (int jj0 = 0; jj0 < nJ; jj0 += kRootChunk) {
            const int cn = std::min(kRootChunk, nJ - jj0);
            detail::host_parallel_for(cn, 1, nth, [&](long long qa, long long qb) {
                for (long long q = qa; q < qb; ++q) {
                    const int J = J0 + jj0 + static_cast<int>(q);
                    rt* col = dchunk + static_cast<std::size_t>(q) * K;
                    int i4 = 0;
                    laed4(K, J + 1, dlamda, wv, col, rho, lam[static_cast<std::size_t>(J)], i4);
                    ce[q] = i4;
                    if (i4 != 0)
                        continue;
                    if (K > 2) {
                        // delta(origin) = -tau exactly (LAED4 forms delta(k) = (D(k) - D(origin)) - tau); the
                        // origin is the closer of the two poles around the root
                        int o = J;
                        if (J + 1 < K && std::fabs(col[static_cast<std::size_t>(J) + 1]) <
                                             std::fabs(col[static_cast<std::size_t>(J)]))
                            o = J + 1;
                        tauv[static_cast<std::size_t>(J)] = -col[static_cast<std::size_t>(o)];
                        dbasev[static_cast<std::size_t>(J)] = dlamda[static_cast<std::size_t>(o)];
                    } else {
                        // K <= 2: the delta already is the (normalised) eigenvector, S(:, J) = delta
                        for (int I = 0; I < K; ++I)
                            S2[J * K + I] = col[static_cast<std::size_t>(I)];
                    }
                }
            });
            for (int q = 0; q < cn; ++q)
                if (ce[q] != 0 && e4 == 0)
                    e4 = ce[q];
            if (K > 2 && e4 == 0)
                detail::host_parallel_for(K, 512, nth, [&](long long i0, long long i1) {
                    for (int q = 0; q < cn; ++q) {
                        const int J = J0 + jj0 + q;
                        const rt* col = dchunk + static_cast<std::size_t>(q) * K;
                        const rt lJ = dlamda[static_cast<std::size_t>(J)];
                        for (long long I = i0; I < i1; ++I) {
                            if (I == J)
                                wp[static_cast<std::size_t>(I)] *= col[I];
                            else
                                wp[static_cast<std::size_t>(I)] *=
                                    col[I] / (dlamda[static_cast<std::size_t>(I)] - lJ);
                        }
                    }
                });
            if (e4 != 0)
                break; // the failure is agreed (or thrown) below, without finishing the other chunks
        }
        if (split)
            agree_or_throw(comm, e4, "pstedc", "secular root (?LAED4)");
        else if (e4 != 0)
            throw std::runtime_error("pstedc: secular root (?LAED4) failed (info=" + std::to_string(e4) +
                                     ")");
        t_roots += clk.lap();

        // pass 2: z-hat, then the normalisation of my roots' eigenvectors from (origin, tau)
        if (K > 2) {
            if (split)
                MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, wp, K, mt, MPI_PROD, comm));
            for (int I = 0; I < K; ++I)
                zhat[static_cast<std::size_t>(I)] = std::copysign(std::sqrt(-wp[static_cast<std::size_t>(I)]),
                                                                  wv[static_cast<std::size_t>(I)]);
            const int workers = std::min(norm_workers, nJ);
            detail::host_parallel_for(workers, 1, nth, [&](long long wa, long long wb) {
                for (long long worker = wa; worker < wb; ++worker) {
                    rt* col = normcols + std::size_t(worker) * n;
                    for (int jj = int(worker); jj < nJ; jj += workers) {
                        const int J = J0 + jj;
                        const rt dbJ = dbasev[J], tJ = tauv[J];
                        for (int k = 0; k < K; ++k)
                            col[k] = zhat[k] / ((dlamda[k] - dbJ) - tJ);
                        nrmv[J] = scaled_nrm2(col, K);
                    }
                }
            });
        }
        if (split) {
            for (int jj = 0; jj < nJ; ++jj) {
                const std::size_t J = static_cast<std::size_t>(J0 + jj);
                rec[4 * J] = lam[J];
                rec[4 * J + 1] = dbasev[J];
                rec[4 * J + 2] = tauv[J];
                rec[4 * J + 3] = nrmv[J];
            }
            for (int p = 0; p < P; ++p) {
                rdispl[static_cast<std::size_t>(p)] = 4 * jlo(p);
                rcount[static_cast<std::size_t>(p)] = 4 * (jlo(p + 1) - jlo(p));
            }
            MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, mt, rec, rcount, rdispl, mt, comm));
            for (int J = 0; J < K; ++J) {
                lam[static_cast<std::size_t>(J)] = rec[4 * static_cast<std::size_t>(J)];
                dbasev[static_cast<std::size_t>(J)] = rec[4 * static_cast<std::size_t>(J) + 1];
                tauv[static_cast<std::size_t>(J)] = rec[4 * static_cast<std::size_t>(J) + 2];
                nrmv[static_cast<std::size_t>(J)] = rec[4 * static_cast<std::size_t>(J) + 3];
            }
        }
        // the merged block's eigenvalues: new ones first, ?LAED8's deflated ones (already in d[s+K:s+m]) after
        std::copy(lam, lam + K, d + s);
        t_zhat += clk.lap();

        // the update: Q(rows, 0:K) = Q2(rows, 0:K) * S, S generated in strips (K <= 2: the explicit K x K)
        if (mr > 0) {
            if (K > 2) {
                if constexpr (kGpu) {
#if DDLA_HAS_GPU
                    for (int I = 0; I < K; ++I) {
                        up[static_cast<std::size_t>(I)] = zhat[static_cast<std::size_t>(I)];
                        up[static_cast<std::size_t>(K) + I] = dlamda[static_cast<std::size_t>(I)];
                        up[2 * static_cast<std::size_t>(K) + I] = dbasev[static_cast<std::size_t>(I)];
                        up[3 * static_cast<std::size_t>(K) + I] = tauv[static_cast<std::size_t>(I)];
                        up[4 * static_cast<std::size_t>(K) + I] = nrmv[static_cast<std::size_t>(I)];
                    }
                    DDLA_CHECK(ddlaMemcpy(handle, dVec, up, 5 * static_cast<std::size_t>(K) * sizeof(rt)));
                    resources.record();
#endif
                }
                // S's rows follow the (possibly reordered) columns of Q2
                const rt* zS = zhat;
                const rt* dlS = dlamda;
                if (do_split) {
                    for (int j = 0; j < K; ++j) {
                        zho[static_cast<std::size_t>(j)] =
                            zhat[static_cast<std::size_t>(order[static_cast<std::size_t>(j)])];
                        dlo[static_cast<std::size_t>(j)] =
                            dlamda[static_cast<std::size_t>(order[static_cast<std::size_t>(j)])];
                    }
                    zS = zho;
                    dlS = dlo;
                }
                const int kr = K - kl - km;
                const int mu =
                    std::max(0, std::min(rb, s + m1) - ra); // my rows inside the left child's row range
                const int ml = mr - mu;
                for (int j0 = 0; j0 < K; j0 += Wcap) {
                    const int wj = std::min(Wcap, K - j0);
                    if constexpr (kGpu) {
#if DDLA_HAS_GPU
                        const long long total = static_cast<long long>(K) * wj;
                        const unsigned blocks =
                            static_cast<unsigned>(std::min<long long>(65535, (total + 255) / 256));
                        pstedc_fill_s_kernel<rt><<<blocks, 256, 0, st>>>(
                            dS, K, dVec, dVec + K, dVec + 2 * K, dVec + 3 * K, dVec + 4 * K, K, j0, wj);
                        DDLA_CHECK(ddlaGetLastError(handle));
#endif
                    } else {
                        detail::host_parallel_for(wj, 4, nth, [&](long long ja, long long jb) {
                            for (long long jj = ja; jj < jb; ++jj) {
                                const std::size_t J =
                                    static_cast<std::size_t>(j0) + static_cast<std::size_t>(jj);
                                const rt dbJ = dbasev[J], tJ = tauv[J], nJv = nrmv[J];
                                rt* sc = dS + static_cast<std::size_t>(jj) * K;
                                for (int k = 0; k < K; ++k)
                                    sc[k] = (zS[static_cast<std::size_t>(k)] /
                                             ((dlS[static_cast<std::size_t>(k)] - dbJ) - tJ)) /
                                            nJv;
                            }
                        });
                    }
                    if (do_split != 1) {
                        blas_impl::gemm_run<B, rt>(handle, 'N', 'N', mr, wj, K, rt(1), Q2b, ld, dS, K, rt(0),
                                                   Qb + static_cast<std::size_t>(j0) * ld, ld);
                    } else {
                        // rows of the left child: the [left | mixed] columns; rows of the right child: [mixed | right]
                        // (an empty column range means an all-zero result block, which must still be written)
                        auto zero_cols = [&](rt* c0, int rows) {
                            for (int c = 0; c < wj; ++c)
                                std::memset(c0 + static_cast<std::size_t>(c) * ld, 0,
                                            static_cast<std::size_t>(rows) * sizeof(rt));
                        };
                        rt* const Cb = Qb + static_cast<std::size_t>(j0) * ld;
                        if (mu > 0) {
                            if (kl + km > 0)
                                blas_impl::gemm_run<B, rt>(handle, 'N', 'N', mu, wj, kl + km, rt(1), Q2b, ld,
                                                           dS, K, rt(0), Cb, ld);
                            else
                                zero_cols(Cb, mu);
                        }
                        if (ml > 0) {
                            if (km + kr > 0)
                                blas_impl::gemm_run<B, rt>(handle, 'N', 'N', ml, wj, km + kr, rt(1),
                                                           Q2b + mu + static_cast<std::size_t>(kl) * ld, ld,
                                                           dS + kl, K, rt(0), Cb + mu, ld);
                            else
                                zero_cols(Cb + mu, ml);
                        }
                    }
                }
            } else {
                if constexpr (kGpu) {
#if DDLA_HAS_GPU
                    DDLA_CHECK(ddlaMemcpy(handle, dS, S2, static_cast<std::size_t>(K) * K * sizeof(rt)));
                    resources.record();
#endif
                } else {
                    std::memcpy(dS, S2, static_cast<std::size_t>(K) * K * sizeof(rt));
                }
                blas_impl::gemm_run<B, rt>(handle, 'N', 'N', mr, K, K, rt(1), Q2b, ld, dS, K, rt(0), Qb, ld);
            }
            // the deflated columns, already in order
            if (m > K)
                DDLA_CHECK(ddlaMemcpy2D(
                    handle, Qb + static_cast<std::size_t>(K) * ld, static_cast<std::size_t>(ld) * sizeof(rt),
                    Q2b + static_cast<std::size_t>(K) * ld, static_cast<std::size_t>(ld) * sizeof(rt),
                    static_cast<std::size_t>(mr) * sizeof(rt), static_cast<std::size_t>(m - K)));
        }
        // next level's sort order of this block
        lamrg(K, m - K, d + s, 1, -1, indxq + s);
    };

    // Reuse the leaf sizes in place for the successive tree levels.
    for (int count = nsub; count > 1; count = (count + 1) / 2) {
        int s = 0;
        for (int i = 0; i < count; i += 2) {
            const int m1 = sizes[i], m = m1 + (i + 1 < count ? sizes[i + 1] : 0);
            if (i + 1 < count)
                merge(s, m, m1);
            sizes[i / 2] = m;
            s += m;
        }
    }
    // The final gather follows the merges on the same stream; the completion
    // fence below also retires their host uploads.

    // ---- ascending eigenvalues and the matching columns -----------------------
    // Local STEDC overwrites D with the sorted values. The final permutation
    // must read an intact source; the last merge's lambda scratch is dead now.
    const rt* values = d;
    if (w == d && nsub > 1) {
        std::copy_n(d, n, lam);
        values = lam;
    }
    for (int i = 0; i < n; ++i)
        w[i] = values[indxq[static_cast<std::size_t>(i)] - 1] * orgnrm;
    if (nrl > 0) {
        if constexpr (kGpu) {
#if DDLA_HAS_GPU
            DDLA_CHECK(ddlaMemcpy(handle, dPerm, indxq, static_cast<std::size_t>(n) * sizeof(int)));
            launch_gather<rt>(handle, dY, ldy, dQ, ld, nrl, dPerm, n);
#endif
        } else {
            detail::host_parallel_for(n, 8, nth, [&](long long ca, long long cb) {
                for (long long c = ca; c < cb; ++c)
                    std::memcpy(dY + static_cast<std::size_t>(c) * ldy,
                                dQ + static_cast<std::size_t>(indxq[static_cast<std::size_t>(c)] - 1) * ld,
                                static_cast<std::size_t>(nrl) * sizeof(rt));
            });
        }
    }
    sync();
    resources.completed();
    t_dev += clk.lap();

    if (prof && me == 0)
        std::printf(
            "[pstedc_slab] n=%d P=%d threads=%d leaf=%d nleaf=%d merges=%d split=%d topK=%lld sumK=%lld | "
            "leaf %.3f decide %.3f roots %.3f zhat+S %.3f dev %.3f s\n",
            n, P, nth, pstedc_max_leaf(policy), nsub, n_merge, n_split, top_k, sum_k, t_leaf, t_decide, t_roots, t_zhat, t_dev);
}

#define PSTEDC_VECTORS_RUN(B, R)                                                                                \
    template void stedc_rows_run<B, R>(const DdlaHandle_t&, const PstedcVectorPlan&, PstedcResources&,    \
                                        R*, R*, R*, R*, R*, void*, std::size_t, void*, std::size_t)
PSTEDC_VECTORS_RUN(DdlaBackend::CPU, float);
PSTEDC_VECTORS_RUN(DdlaBackend::CPU, double);
#if DDLA_HAS_GPU
PSTEDC_VECTORS_RUN(DdlaBackend::GPU, float);
PSTEDC_VECTORS_RUN(DdlaBackend::GPU, double);
#endif
#undef PSTEDC_VECTORS_RUN

template <DdlaBackend B, class R>
void pstedc_run(const DdlaHandle_t& h, const PstedcVectorPlan& p, PstedcResources& resources,
                R* d, R* e, R* w, R* z, R* scratch, void* backend_work,
                std::size_t backend_bytes, void* host_work, std::size_t host_bytes)
{
    if (!p.n)
        return;
    if (local_stedc_cpu(p) || (p.backend == DdlaBackend::GPU && p.ranks == 1)) {
        pstedc_local_run<B,R>(h, p, 'I', d, e, w, z, scratch, backend_work, host_work, &resources);
        return;
    }
    stedc_rows_run<B,R>(h, p, resources, d, e, w, z, scratch, backend_work,
                        backend_bytes, host_work, host_bytes);
}
#define PSTEDC_LOCAL_DISPATCH(B,R) \
    template void pstedc_run<B,R>(const DdlaHandle_t&,const PstedcVectorPlan&,PstedcResources&, \
                                  R*,R*,R*,R*,R*,void*,std::size_t,void*,std::size_t)
PSTEDC_LOCAL_DISPATCH(DdlaBackend::CPU,float);
PSTEDC_LOCAL_DISPATCH(DdlaBackend::CPU,double);
#if DDLA_HAS_GPU
PSTEDC_LOCAL_DISPATCH(DdlaBackend::GPU,float);
PSTEDC_LOCAL_DISPATCH(DdlaBackend::GPU,double);
#endif
#undef PSTEDC_LOCAL_DISPATCH

template <DdlaBackend B, typename rt>
void pstedc_run(const DdlaHandle_t& handle, const PstedcPlan& policy, rt* d, rt* e, rt* w, rt* vectors,
                void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes,
                PstedcResources* rows)
{
    try {
        if (!policy.rows.n)
            return;
        if (policy.mode != PstedcMode::values) {
            assert(rows);
            pstedc_run<B, rt>(handle, policy.rows, *rows, d, e, w, vectors, nullptr, backend_work,
                             backend_bytes, host_work, host_bytes);
            return;
        }

        if constexpr(B==DdlaBackend::CPU) {
            if(local_stedc_cpu(policy.rows)) {
                pstedc_local_run<B,rt>(handle,policy.rows,'N',d,e,w,static_cast<rt*>(nullptr),
                    static_cast<rt*>(nullptr),backend_work,host_work,nullptr);
                return;
            }
        }

        using detail::take;
        const int n = policy.rows.n;
        MPI_Comm comm = handle->comm;
        const int P = policy.rows.ranks, me = policy.rows.rank;
        const MPI_Datatype mt = detail::mpi_datatype<rt>();
        const bool prof = policy.rows.profile;
        PhaseClock clk;
        double t_leaf = 0, t_decide = 0, t_roots = 0, t_zhat = 0, t_dev = 0, t_share = 0;
        int n_merge = 0, n_split = 0;
        long long sum_k = 0, top_k = 0;
        // eigenvalue-only coverage counters (printed with DDLA_EIG_PROFILE=1)
        long long c_rot = 0, c_mixed = 0, c_k0 = 0, c_recomp = 0, c_lamonly = 0;

        // ---- scale by the max-norm (as ?STEDC does per unreduced block) -------
        const rt orgnrm = pstedc_scale(n, d, e);

        // Frozen leaf size and bounded root-delta cache.
        const int leaf = policy.rows.leaf, keep_mb = policy.keep_mb;
        const std::size_t keep_elems = std::size_t(keep_mb) * (std::size_t(1) << 20) / sizeof(rt);
        const int nth = policy.rows.threads;

        // ---- scratch (pstedc_workspace_size) -----------------------------------
        // The store's backend buffers, then the host arrays: host memory on a GPU
        // handle, after the backend buffers on a CPU handle (where backend memory
        // is host memory).  The leaf phase and the merge phase overlay.
        const int nsub = pstedc_leaves(n, leaf), maxsz = pstedc_leaf_geometry(n, nsub, nsub - 1);
        const int mine = pstedc_my_leaves(nsub, me, P);
        const int workers = std::min(1, mine);
        const bool vals_single = nsub == 1;
        const std::size_t leaf_work_count =
            vals_single ? std::size_t(n) : std::size_t(maxsz) * maxsz + 6 * std::size_t(maxsz) + 1;
        const std::size_t leaf_iwork_count = vals_single ? 0 : 5 * std::size_t(maxsz) + 3;
        const std::size_t delta_count = dc_delta_count<rt>(policy);
        char* const backend_base = static_cast<char*>(backend_work);
        DcRowsStore<rt> store(handle, policy);
        char* const host_base = B == DdlaBackend::GPU ? static_cast<char*>(host_work) : backend_base;
        char* hw = host_base;
        int* const sizes = take<int>(hw, nsub);
        int* const start = take<int>(hw, nsub);
        int* const indxq = take<int>(hw, n);
        store.take_kept(hw);
        char* const phase = hw;
        char* lw = phase;
        rt* const leaf_work = take<rt>(lw, std::size_t(workers) * leaf_work_count);
        int* const leaf_iwork = take<int>(lw, std::size_t(workers) * leaf_iwork_count);
        int* const errs = take<int>(lw, mine);
        char* mw = phase;
        const auto real = [&](std::size_t count) {
            return take<rt>(mw, nsub > 1 ? count : 0);
        };
        const auto integer = [&](std::size_t count) {
            return take<int>(mw, nsub > 1 ? count : 0);
        };
        const std::size_t nn = n;
        rt* const z = real(nn);
        rt* const dlamda = real(nn);
        rt* const wv = real(nn);
        rt* const givnum = real(2 * nn);
        rt* const lam = real(nn);
        rt* const wp = real(nn);
        rt* const delta = real(delta_count);
        rt* const fnv = real(nn);
        rt* const lnv = real(nn);
        rt* const tri = real(3 * nn);
        int* const iq = integer(nn);
        int* const perm = integer(nn);
        int* const givcol = integer(2 * nn);
        int* const indxp = integer(nn);
        int* const indx = integer(nn);
        int* const rcount = integer(P);
        int* const rdispl = integer(P);
        store.take_phases(lw, mw);
        assert(B == DdlaBackend::GPU ? std::size_t(std::max(lw, mw) - host_base) <= host_bytes
                                     : std::size_t(std::max(lw, mw) - backend_base) <= backend_bytes);

        // ---- ?LAED0 subdivision and rank-one tears ----------------------------
        pstedc_split_leaves(n, nsub, d, e, sizes, start);

        store.initialize();

        // ---- one leaf, eigenvalues only: ?STEDC('N') (= ?STERF), no Z anywhere ----
        if (vals_single) {
            int li = 0;
            if (me == 0) {
                rt* const el = leaf_work;
                std::fill_n(el, n, rt(0));
                std::copy_n(e, std::max(0, n - 1), el);
                rt work = 0;
                int iwork = 0;
                // COMPZ=N uses STERF: both LAPACK work lengths are exactly one.
                stedc('N', n, d, el, static_cast<rt*>(nullptr), 1, &work, 1, &iwork, 1, li);
            }
            agree_or_throw(comm, li, "pstedc", "leaf ?STEDC");
            if (P > 1)
                MPI_CHECK(MPI_Bcast(d, n, mt, 0, comm));
            for (int k = 0; k < n; ++k)
                indxq[static_cast<std::size_t>(k)] = k + 1;
        }
        // ---- leaves: host ?STEDC round-robin over the ranks, then broadcast ---
        else {
            auto solve_leaf = [&](int i, rt* base, int* iw) -> int {
                const int m = sizes[i], s0 = start[i];
                rt* dl = base;
                rt* el = dl + maxsz;
                rt* lapack = el + maxsz;
                std::copy_n(d + s0, m, dl);
                std::fill_n(el, m, rt(0));
                std::copy_n(e + s0, std::max(0, m - 1), el);
                int li = 0;
                rt* Zt = store.leaf_target(m);
                stedc('I', m, dl, el, Zt, m, lapack, int(leaf_work_count - 2 * maxsz), iw,
                      int(leaf_iwork_count), li);
                std::copy_n(dl, m, d + s0);
                store.leaf_commit(s0, m);
                return li;
            };
            {
                detail::host_parallel_for(workers, 1, nth, [&](long long wa, long long wb) {
                    for (long long worker = wa; worker < wb; ++worker) {
                        rt* base = leaf_work + std::size_t(worker) * leaf_work_count;
                        int* iw = leaf_iwork + std::size_t(worker) * leaf_iwork_count;
                        for (int j = int(worker); j < mine; j += workers)
                            errs[j] = solve_leaf(me + j * P, base, iw);
                    }
                });
            }
            int leaf_err = 0;
            for (int j = 0; j < mine; ++j)
                if (errs[j] && !leaf_err)
                    leaf_err = errs[j];
            agree_or_throw(comm, leaf_err, "pstedc", "leaf ?STEDC");
            if (P > 1)
                for (int i = 0; i < nsub; ++i)
                    MPI_CHECK(MPI_Bcast(d + start[static_cast<std::size_t>(i)],
                                        sizes[static_cast<std::size_t>(i)], mt, i % P, comm));
            store.share_leaves(sizes, start, nsub);
            for (int i = 0; i < nsub; ++i)
                for (int k = 0; k < sizes[static_cast<std::size_t>(i)]; ++k)
                    indxq[static_cast<std::size_t>(start[static_cast<std::size_t>(i)]) + k] = k + 1;
        }
        t_leaf = clk.lap();

        // ---- merges -------------------------------------------------------------
        auto merge = [&](int s, int m, int m1) {
            // z: last row of the left child's columns, first row of the right's
            store.read_z(s, m, m1, z);

            // decide (identical on every rank: identical inputs, scalar code)
            rt rho = e[s + m1 - 1];
            std::copy(indxq + s, indxq + s + m, iq);
            int K = 0, givptr = 0, i8 = 0;
            laed8_decide(K, m, d + s, iq, rho, m1, z, dlamda, wv, perm, givptr, givcol, givnum, indxp, indx,
                         i8);
            // robustness: every rank must agree on K (and on failure) before the
            // merge continues into mismatched collectives
            pstedc_agree_deflation(comm, i8, K);
            t_decide += clk.lap();

            store.rotate_gather(s, m, givptr, givcol, givnum, perm);
            ++n_merge;
            sum_k += K;
            top_k = K;
            {
                c_rot += givptr;
                for (int g = 0; g < givptr; ++g)
                    if ((givcol[2 * g] - 1 < m1) != (givcol[2 * g + 1] - 1 < m1))
                        ++c_mixed;
            }

            if (K == 0) {
                ++c_k0;
                store.restore_all(s, m);
                for (int k = 0; k < m; ++k)
                    indxq[static_cast<std::size_t>(s) + k] = k + 1;
                return;
            }

            // roots: a contiguous share per rank when it pays
            const bool split = (P > 1) && (K > 2) && (K >= kMinRootsPerRank * P);
            auto jlo = [&](int p) {
                return split ? static_cast<int>(static_cast<long long>(K) * p / P) : 0;
            };
            const int J0 = jlo(me);
            const int J1 = split ? jlo(me + 1) : K;
            const int nJ = J1 - J0;
            if (split)
                ++n_split;

            // ---- eigenvalue-only merge: the merged block's first/last row are
            // F2*S and L2*S over the K secular vectors; S is never stored in
            // full and there is no GEMM.  The top merge (m == n) feeds nothing
            // afterwards, so it only needs the roots themselves.
            const bool lam_only = (m == n);
            // keep (one ?LAED4 pass) or recompute (two): decided from the largest
            // per-rank share so every rank agrees; K <= 2 always keeps
            const std::size_t nj_max =
                split ? static_cast<std::size_t>((K + P - 1) / P) : static_cast<std::size_t>(K);
            const bool keep = !lam_only && (K <= 2 || static_cast<std::size_t>(K) * nj_max <= keep_elems);
            if (lam_only)
                ++c_lamonly;
            else if (!keep)
                ++c_recomp;
            assert((keep ? std::size_t(K) * std::max(nJ, 1) : std::size_t(K)) <= delta_count);
            auto col_of = [&](int jj) {
                return delta + (keep ? static_cast<std::size_t>(jj) * K : 0);
            };
            if (!lam_only && K > 2)
                std::fill(wp, wp + K, rt(1));
            int e4 = 0;
            for (int jj = 0; jj < nJ; ++jj) {
                int i4 = 0;
                rt* col = col_of(jj);
                laed4(K, J0 + jj + 1, dlamda, wv, col, rho, lam[J0 + jj], i4);
                if (i4 != 0 && e4 == 0)
                    e4 = i4;
                if (i4 == 0 && !lam_only && K > 2) {
                    // my share of the Loewner z-hat product (as in the GEMM branch)
                    const int J = J0 + jj;
                    const rt lJ = dlamda[J];
                    for (int I = 0; I < J; ++I)
                        wp[static_cast<std::size_t>(I)] *= col[I] / (dlamda[I] - lJ);
                    wp[static_cast<std::size_t>(J)] *= col[J];
                    for (int I = J + 1; I < K; ++I)
                        wp[static_cast<std::size_t>(I)] *= col[I] / (dlamda[I] - lJ);
                }
            }
            if (split)
                agree_or_throw(comm, e4, "pstedc", "secular root (?LAED4)");
            else if (e4 != 0)
                throw std::runtime_error("pstedc: secular root (?LAED4) failed (info=" + std::to_string(e4) +
                                         ")");
            t_roots += clk.lap();

            if (lam_only) {
                if (split) {
                    for (int p = 0; p < P; ++p) {
                        rdispl[static_cast<std::size_t>(p)] = jlo(p);
                        rcount[static_cast<std::size_t>(p)] = jlo(p + 1) - jlo(p);
                    }
                    MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, mt, lam, rcount, rdispl, mt, comm));
                }
                std::copy(lam, lam + K, d + s);
            } else {
                if (K > 2) {
                    if (split)
                        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, wp, K, mt, MPI_PROD, comm));
                    for (int I = 0; I < K; ++I)
                        wp[static_cast<std::size_t>(I)] = std::copysign(
                            std::sqrt(-wp[static_cast<std::size_t>(I)]), wv[static_cast<std::size_t>(I)]);
                }
                // pass 2: S(:, J) = z-hat ./ delta_J, normalised (K <= 2: the delta
                // already is the eigenvector), then the two dot products.  Plain
                // loops: every rank that computes a root must get the same bits.
                const rt* Fr = store.F2();
                const rt* Lr = store.L2();
                for (int jj = 0; jj < nJ; ++jj) {
                    rt* col = col_of(jj);
                    if (!keep) {
                        // the same call as pass 1 on the same inputs: cannot fail now
                        int i4 = 0;
                        rt ldummy = 0;
                        laed4(K, J0 + jj + 1, dlamda, wv, col, rho, ldummy, i4);
                        if (i4 != 0)
                            throw std::logic_error("pstedc: ?LAED4 recompute disagreed with its first pass");
                    }
                    if (K > 2) {
                        for (int I = 0; I < K; ++I)
                            col[I] = wp[static_cast<std::size_t>(I)] / col[I];
                        const rt nrm = scaled_nrm2(col, K);
                        for (int I = 0; I < K; ++I)
                            col[I] /= nrm;
                    }
                    rt fa = 0, la = 0;
                    for (int I = 0; I < K; ++I) {
                        fa += Fr[I] * col[I];
                        la += Lr[I] * col[I];
                    }
                    fnv[static_cast<std::size_t>(J0 + jj)] = fa;
                    lnv[static_cast<std::size_t>(J0 + jj)] = la;
                }
                if (split) {
                    for (int jj = 0; jj < nJ; ++jj) {
                        const std::size_t J = static_cast<std::size_t>(J0 + jj);
                        tri[3 * J] = lam[J];
                        tri[3 * J + 1] = fnv[J];
                        tri[3 * J + 2] = lnv[J];
                    }
                    for (int p = 0; p < P; ++p) {
                        rdispl[static_cast<std::size_t>(p)] = 3 * jlo(p);
                        rcount[static_cast<std::size_t>(p)] = 3 * (jlo(p + 1) - jlo(p));
                    }
                    MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, mt, tri, rcount, rdispl, mt, comm));
                    for (int J = 0; J < K; ++J) {
                        lam[static_cast<std::size_t>(J)] = tri[3 * static_cast<std::size_t>(J)];
                        fnv[static_cast<std::size_t>(J)] = tri[3 * static_cast<std::size_t>(J) + 1];
                        lnv[static_cast<std::size_t>(J)] = tri[3 * static_cast<std::size_t>(J) + 2];
                    }
                }
                std::copy(lam, lam + K, d + s);
                store.place_rows(s, m, K, fnv, lnv);
            }
            t_zhat += clk.lap();
            // next level's sort order of this block
            lamrg(K, m - K, d + s, 1, -1, indxq + s);
        };

        for (int current = nsub; current > 1; current /= 2) {
            int s = 0;
            for (int i = 0; i < current; i += 2) {
                const int m1 = sizes[i], m = m1 + sizes[i + 1];
                merge(s, m, m1);
                sizes[i / 2] = m;
                s += m;
            }
        }
        t_dev += clk.lap();

        // ---- ascending eigenvalues and the matching columns ---------------------
        for (int i = 0; i < n; ++i)
            w[i] = d[indxq[static_cast<std::size_t>(i)] - 1] * orgnrm;
        t_dev += clk.lap();

        if (prof && me == 0)
            std::printf("%s n=%d P=%d leaf=%d nleaf=%d merges=%d split=%d topK=%lld "
                        "sumK=%lld | leaf %.3f decide %.3f roots %.3f zhat+S %.3f dev %.3f "
                        "share %.3f s\n",
                        DcRowsStore<rt>::kTag, n, P, leaf, nsub, n_merge, n_split, top_k, sum_k, t_leaf,
                        t_decide, t_roots, t_zhat, t_dev, t_share);
        if (prof && me == 0)
            std::printf("%s keep_mb=%d rotations=%lld mixed=%lld k0_merges=%lld recomputed=%lld "
                        "lam_only=%lld\n",
                        DcRowsStore<rt>::kTag, keep_mb, c_rot, c_mixed, c_k0, c_recomp, c_lamonly);
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(handle) == DdlaBackend::GPU)
            try {
                (void)ddlaSynchronize(handle);
            } catch (...) {
            }
        throw;
    }
}

#define PSTEDC_PHASE_RUN(B, R)                                                                               \
    template void pstedc_run<B, R>(const DdlaHandle_t&, const PstedcPlan&, R*, R*, R*, R*, void*,            \
                                   std::size_t, void*, std::size_t, PstedcResources*)
PSTEDC_PHASE_RUN(DdlaBackend::CPU, float);
PSTEDC_PHASE_RUN(DdlaBackend::CPU, double);
#if DDLA_HAS_GPU
PSTEDC_PHASE_RUN(DdlaBackend::GPU, float);
PSTEDC_PHASE_RUN(DdlaBackend::GPU, double);
#endif
#undef PSTEDC_PHASE_RUN

} // namespace spectral_impl

} // namespace ddla
