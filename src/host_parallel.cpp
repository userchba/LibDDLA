// Shared host-only threading used by Hermitian completion and eigensolver kernels.
#include "host_parallel.h"
#include "host_threads.h"
#include "ddla_connector.h"
#include <algorithm>
#include <cstdlib>
#include <exception>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace ddla { namespace detail {
int host_thread_budget(MPI_Comm comm)
{
    MPI_Comm node = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node));
    int ranks = 1;
    MPI_CHECK(MPI_Comm_size(node, &ranks));
    int budget = std::max(1, host_allowed_cpus() / std::max(1, ranks));
#if defined(__linux__) && defined(CPU_COUNT)
    cpu_set_t mask;
    CPU_ZERO(&mask);
    int valid = sched_getaffinity(0, sizeof(mask), &mask) == 0 && CPU_COUNT(&mask) > 0;
    int all_valid = 0;
    MPI_CHECK(MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN, node));
    if (all_valid) {
        // Each CPU contributes one worker divided among the ranks allowed
        // to run on it. This also covers partially overlapping bindings.
        int mine[CPU_SETSIZE]{}, sharing[CPU_SETSIZE]{};
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            mine[cpu] = CPU_ISSET(cpu, &mask) ? 1 : 0;
        MPI_CHECK(MPI_Allreduce(mine, sharing, CPU_SETSIZE, MPI_INT, MPI_SUM, node));
        double available = 0;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (mine[cpu]) available += 1.0 / sharing[cpu];
        budget = std::max(1, static_cast<int>(available + 1e-9));
    }
#endif
    MPI_CHECK(MPI_Comm_free(&node));
    if (const char* env = std::getenv("OMP_NUM_THREADS")) {
        const int requested = std::atoi(env);
        if (requested >= 1) budget = requested;
    }
    if (const char* env = std::getenv("DDLA_EIG_THREADS")) {
        const int requested = std::atoi(env);
        if (requested >= 1) budget = requested;
    }
    return std::max(1, std::min(budget, 512));
}

void host_parallel_for(long long n,long long grain,int nthreads,void* context,
                       void (*function)(void*,long long,long long))
{
    if (n <= 0) return;
    grain = std::max<long long>(1, grain);
    const long long nchunks = (n + grain - 1) / grain;
    const int nt = static_cast<int>(std::min<long long>(std::max(1, nthreads), nchunks));
    if (nt <= 1) {
        function(context,0,n);
        return;
    }
    std::exception_ptr first;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
#endif
    for (long long c = 0; c < nchunks; ++c) {
        try {
            function(context,c*grain,std::min(n,(c+1)*grain));
        } catch (...) {
#ifdef _OPENMP
#pragma omp critical(ddla_host_parallel_for)
#endif
            if (!first) first = std::current_exception();
        }
    }
    if (first) std::rethrow_exception(first);
}

void host_parallel_for(long long n,long long grain,int nthreads,
    const std::function<void(long long,long long)>& function)
{
    host_parallel_for(n,grain,nthreads,const_cast<void*>(static_cast<const void*>(&function)),
        [](void* context,long long lo,long long hi) {
            (*static_cast<const std::function<void(long long,long long)>*>(context))(lo,hi);
        });
}

}}
