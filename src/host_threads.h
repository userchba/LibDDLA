#ifndef DDLA_HOST_THREADS_H
#define DDLA_HOST_THREADS_H

// The host-loop and eigensolver budget is prepared collectively by ddlaSet
// and retained by that handle. Disjoint rank bindings retain their CPUs;
// ranks whose masks overlap share those CPUs. OMP_NUM_THREADS >= 1 wins
// over the default, and DDLA_EIG_THREADS >= 1 wins over both (cap 512).

#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <thread>

#if defined(__linux__)
#include <sched.h>
#endif

// Optional BLAS thread controls (weak: null unless the linked BLAS exports them), used by ScopedBlasThreads below.
extern "C" {
void openblas_set_num_threads(int) __attribute__((weak));
int  openblas_get_num_threads(void) __attribute__((weak));
void mkl_set_num_threads(int) __attribute__((weak));
int  mkl_get_max_threads(void) __attribute__((weak));
}

namespace ddla {
namespace detail {

// CPUs this process may run on: the affinity mask honours a Slurm cpuset or an mpirun binding.
inline int host_allowed_cpus()
{
#if defined(__linux__) && defined(CPU_COUNT)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        const int n = CPU_COUNT(&set);
        if (n > 0) return n;
    }
#endif
    const unsigned hw = std::thread::hardware_concurrency();
    return hw > 0 ? static_cast<int>(hw) : 1;
}

// Collective over comm, with no process-global cache. Called at handle setup,
// never from a kernel that only a subset of ranks may reach.
int host_thread_budget(MPI_Comm comm, int& node_ranks);

// ---------------------------------------------------------------------------
// ScopedBlasThreads: hold the BLAS library's OWN thread count at `n` for the lifetime of the object, then put it back.
//
// A host loop that gives every OpenMP thread whole BLAS calls (gemmBatched: one matrix product per core) must keep the BLAS
// from starting a thread pool of its own inside each call.  An OpenMP-built OpenBLAS already drops to one thread inside a parallel
// region, but a pthreads-built one does not, and MKL needs the same call.  So, as LibRI
// does for MKL (mkl_get_max_threads / mkl_set_num_threads around its loop), the count is saved, set to `n`, and restored -- here by
// the destructor, so an exception cannot leave the BLAS pinned.
//
// The entry points are reached through WEAK symbols: they are called only if the BLAS this library is linked with exports them
// (OpenBLAS: openblas_{get,set}_num_threads; MKL: mkl_get_max_threads / mkl_set_num_threads).  With any other BLAS the object does
// nothing, and no build option or configure-time probe is needed.  The setting is process-global: make the object once, outside the
// parallel region, in code no other host thread of the process runs BLAS from at the same time.
// ---------------------------------------------------------------------------
class ScopedBlasThreads {
public:
    explicit ScopedBlasThreads(int n)
    {
        if (openblas_get_num_threads != nullptr && openblas_set_num_threads != nullptr) {
            openblas_saved_ = openblas_get_num_threads();
            openblas_set_num_threads(n);
            openblas_ = true;
        }
        if (mkl_get_max_threads != nullptr && mkl_set_num_threads != nullptr) {
            mkl_saved_ = mkl_get_max_threads();
            mkl_set_num_threads(n);
            mkl_ = true;
        }
    }
    ~ScopedBlasThreads()
    {
        if (openblas_) openblas_set_num_threads(openblas_saved_);
        if (mkl_) mkl_set_num_threads(mkl_saved_);
    }
    ScopedBlasThreads(const ScopedBlasThreads&) = delete;
    ScopedBlasThreads& operator=(const ScopedBlasThreads&) = delete;

    // The thread count the BLAS reports right now, 0 when it cannot be queried (a BLAS that is neither OpenBLAS nor MKL).
    static int current()
    {
        if (openblas_get_num_threads != nullptr) return openblas_get_num_threads();
        if (mkl_get_max_threads != nullptr) return mkl_get_max_threads();
        return 0;
    }

private:
    bool openblas_ = false, mkl_ = false;
    int openblas_saved_ = 0, mkl_saved_ = 0;
};

// Calls call(i) for every product i of the batch.  With fewer than two products or two threads there is nothing to spread and the loop is
// serial with the BLAS keeping its own threads (one big product still gets a threaded BLAS); otherwise min(nthreads, count) OpenMP threads
// each run whole products while the BLAS is pinned to one thread.  Without OpenMP the loop stays serial and the BLAS is left alone.
template <typename Call>
void for_each_product(int nthreads, int count, Call&& call)
{
#ifdef _OPENMP
    const int nt = std::min(nthreads, count);
#else
    const int nt = 1;
    (void)nthreads;
#endif
    if (nt < 2) {
        for (int i = 0; i < count; ++i) call(i);
        return;
    }
    detail::ScopedBlasThreads one_blas_thread(1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nt)
#endif
    for (int i = 0; i < count; ++i) call(i);
}


} // namespace detail
} // namespace ddla

#endif // DDLA_HOST_THREADS_H
