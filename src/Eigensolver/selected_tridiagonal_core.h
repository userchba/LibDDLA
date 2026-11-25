#ifndef DDLA_PRIVATE_SELECTED_TRIDIAGONAL_CORE_H
#define DDLA_PRIVATE_SELECTED_TRIDIAGONAL_CORE_H
// EXPERIMENTAL private selected solver; no public API/default adoption.
#include <mpi.h>
#include <cstddef>
#include <cstdint>
namespace ddla { namespace spectral_impl { namespace selected {
enum class Code { success, nonfinite_input, sterf_failed, stebz_failed,
    count_mismatch, value_mismatch, factor_failed, solve_failed,
    no_convergence, unrepresentable_values, scaling_loss, unresolved_selection };
struct Status {
    Code code=Code::success;
    int eigenvector=-1, native_info=0, scale_exponent=0;
    std::uint64_t solves=0, dot_reductions=0, allgathers=0;
    double maximum_residual=0, maximum_orthogonality=0;
    double last_residual=0, last_orthogonality=0;
    double cutoff_gap=0, selection_budget=0, subspace_residual_ratio=0;
    std::uint64_t growth_checks=0;
    int last_seed_solves=0, last_seed_growth=0, first_five_solves=0;
    unsigned first_five_growth_mask=0;
    double last_native_maximum=0, last_post_reorth_maximum=0, last_growth_magnitude=0;
    bool candidate_available=false, last_normwise_accepted=false;
    int last_candidate_index=-1;
};
struct Plan {
    MPI_Comm comm=MPI_COMM_NULL;
    int n=0, nev=0, rank=0, ranks=1, row0=0, rows=0, ldy=1;
    int max_iterations=24, max_restarts=4;
    std::size_t bytes=0, ds=0, es=0, values=0, phase=0;
    std::size_t blocks=0, splits=0, order=0, integer_phase=0, counts=0, displacements=0;
};
// Pure geometry for an enclosing checker that already validated collective
// dimensions and pointer/alias contracts. No MPI, input access or native query.
template<class R> Plan geometry(MPI_Comm,int n,int nev,int rank,int ranks,int ldy,
    int max_iterations=24,int max_restarts=4);
// Collective preparation; communicator and pointers remain caller-owned.
// d/e are replicated HOST input. w may overlap either input; both logical
// inputs are snapshotted before any w write. Y and w must be disjoint outputs.
// Input/output must not overlap the separately allocated work span.
template<class R> Plan check(MPI_Comm,int n,int nev,const R* d,const R* e,R* w,
    R* Y,int ldy,int max_iterations=24,int max_restarts=4);
// Pure rank-local arithmetic result, frozen by check. No LAPACK query.
inline std::size_t workspace_bytes(const Plan& p) noexcept { return p.bytes; }
// Unchecked preallocated execution; never allocates or creates resources.
// Lowest nev vectors: rows [row0,row0+rows), column-major, stride ldy.
// All n ascending eigenvalues are returned in host w. Published values are the
// successful STERF values, uniformly rescaled; bisection never replaces a prefix.
// Numerical failures are consistent on every rank; no D&C fallback.
// scaling_loss returns before any w/Y write. Growth-based targeted inverse
// iteration plus normwise certificates define success; no gap/angle guarantee.
// Cutoff metrics are informational and never refuse a solve. A nontrivial
// block needs three successful raw-growth checks, within fixed 24x4 bounds.
// MPI calls are made exclusively by the calling thread (FUNNELED contract).
template<class R> Status run(const Plan&,const R* d,const R* e,R* w,R* Y,void* work);
// Diagnostic view only: the last normalized current vector, not committed Y.
// The caller may inspect it after a one-solve nonconvergence result, but only
// when candidate_available is true. It borrows the real allocated x region;
// validity ends on scratch reuse. Never interpret this as successful output.
template<class R> const R* last_current(const Plan& p,const Status& s,const void* work) noexcept {
    if (!s.candidate_available || !p.n) return nullptr;
    return reinterpret_cast<const R*>(static_cast<const unsigned char*>(work)+p.phase)+std::size_t(4)*p.n;
}
struct Certificate { bool accepted=false; double norm=0,residual=0,orthogonality=0; };
// Collective allocation-free certificate. d/e/lambda share safe scaled units;
// x is a replicated full current vector. Q contains this rank's previous rows.
// coefficients holds previous scratch elements for the batched dot reduction.
template<class R> Certificate certify_scaled(const Plan&,const R* d,const R* e,R lambda,
    const R* x,const R* Q,int previous,R* coefficients);
} } } // namespace ddla::spectral_impl::selected
#endif
