#ifndef DDLA_EXPERIMENTAL_CHEBYSHEV_IMPL_H
#define DDLA_EXPERIMENTAL_CHEBYSHEV_IMPL_H
// Experimental isolated component. No installed API or automatic selection.
#include "Eigensolver/pheevd_driver.h"
#include "PBLAS/pgemm_impl.h"
#include "QR/pgeqrf_impl.h"
#include "SVD/svd_workspace.h"
#include "random_generate_impl.h"
#include <array>
#include <optional>
#include <vector>

namespace ddla
{
namespace spectral_impl
{
namespace chebyshev
{
enum Status
{
    success = 0,
    iteration_limit = 1,
    rank_breakdown = 2,
    nonfinite = 3,
    scaling_loss = 4,
    projected_failure = 5,
    unrepresentable = 6
};
struct Plan
{
    DdlaBackend backend{};
    DdlaCommunication transport{};
    void *compute = nullptr;
    void *data = nullptr;
    char uplo = 'L', major = 'R';
    MPI_Comm communicator = MPI_COMM_NULL;
    int device = -1;
    int n = 0, nev = 0, q = 0, degree = 0, max_iterations = 0, threads = 1;
    int pr = 1, pc = 1, r = 0, c = 0, rows = 0, columns = 0, small_rows = 0, small_columns = 0;
    int bA = 1, bq = 1, qr_width = 1;
    bool warm = false, ready = false, qr_inplace = false;
    double tolerance = 0;
    std::array<int, DDLA_DLEN_> a{}, z{}, basis{}, projected{};
    pgemm_impl::Algorithm action = pgemm_impl::Algorithm::BC;
    pgemm_impl::Algorithm project = pgemm_impl::Algorithm::AB;
    pgemm_impl::Algorithm rotate = pgemm_impl::Algorithm::AB;
    int action_width = 1, project_width = 1, rotate_width = 1;
    PunmbrPlan apply{};
    PheevdDriverPlan eigen{};
    PheevdDriverWorkspace eigen_work{};
    std::size_t basis_elements = 0, projected_elements = 0, stats_elements = 0;
    std::size_t qr_backend = 0, qr_host = 0, phase_backend = 0, phase_host = 0;
    std::size_t backend_bytes = 0, host_bytes = 0;
};
struct Counters
{
    int qr_retries = 0, operator_applications = 0;
    double maximum_residual = 0, lower_bound = 0, upper_bound = 0, matrix_norm = 0;
};
template <class T> class Resources
{
  public:
    using R = typename real_type<T>::type;
    ddla::vector<T> basis, h, s;
    ddla::vector<double> device_stats, device_parameters;
    std::vector<double> stats, parameters;
    std::vector<T> tau, local_tau;
    std::vector<R> values;
    std::optional<random_impl::PreparedRng> rng;
    std::optional<PheevdDriverResources<T>> eigen;
    // Local allocation only. The outer owner must agree this step on all ranks.
    Resources(const DdlaHandle_t &, const Plan &, std::uint64_t seed = 0x44544c4143484542ULL);
    std::size_t retained_backend_bytes() const noexcept;
    std::size_t retained_host_bytes() const noexcept;
};
// Metadata/options only; no matrix read. A is destroyed, w has nev entries.
// Warm Z is copied before A changes. Cold Z is never read. w/Z must be disjoint.
template <class T>
Plan check(const DdlaHandle_t &, DdlaBackend, char, int, int, int, const T *, const int *,
           const typename real_type<T>::type *, const T *, const int *, double, int, int, bool);
// Collective preparation uses real owned matrices for native workspace queries.
// The prepared resources and this plan must remain together until all runs end.
template <DdlaBackend B, class T> void prepare(const DdlaHandle_t &, Plan &, Resources<T> &);
// Pure integer query AFTER preparation. Includes two arena-owned basis buffers;
// Resources owns the third basis, H/S and separately reported linear storage.
inline void workspace_size(const Plan &p, std::size_t *backend, std::size_t *host)
{
    if (!p.ready)
        throw std::invalid_argument("chebyshev: prepare before sizing");
    *backend = p.backend_bytes;
    *host = p.host_bytes;
}
// Pure, exact two-slot HostStagedMPI high waters of this prepared composite.
// The owner must reserve these slots under allocation agreement BEFORE run;
// the numerical run neither reserves nor agrees per-message cache growth.
template<class T>std::array<std::size_t,2> staging_bytes(const DdlaHandle_t&,const Plan&);
// No owners/queries/setup in raw execution. Existing vendor/HostStaged caches
// may initialize on first use: distinguish vendor warm-up from a cold reserved
// transport cache. HostStagedMPI raw borrowing requires staging_bytes reserved.
// Adequate randomized starting-space overlap is assumed. Residual convergence
// alone is not a deterministic lowest-index certificate.
template <DdlaBackend B, class T>
void run(const DdlaHandle_t &, const Plan &, Resources<T> &, T *, typename real_type<T>::type *,
         T *, int &, int &, int &, Counters *, void *, std::size_t, void *, std::size_t);
} // namespace chebyshev
} // namespace spectral_impl
} // namespace ddla
#endif
