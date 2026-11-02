// Shared host-only threading used by Hermitian completion and eigensolver kernels.
#include "host_parallel.h"
#include "host_threads.h"
#include <algorithm>
#include <cstdlib>
#include <exception>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace ddla { namespace detail {
namespace {
int& host_worker_cache()
{
    static int cached = 0;
    return cached;
}
}

int host_omp_threads_for_node(int ranks_on_node)
{
    (void)ranks_on_node;
    int& cached=host_worker_cache();
    if (cached > 0) return cached;
    int t = 1;
#ifdef _OPENMP
    if (const char* env = std::getenv("OMP_NUM_THREADS")) {
        const int v = std::atoi(env);
        t = v >= 1 ? v : omp_get_max_threads();
    } else {
        t = std::max(1, host_allowed_cpus() / std::max(1, ranks_on_node));
    }
#endif
    if (const char* env = std::getenv("DDLA_EIG_THREADS")) {
        const int v = std::atoi(env);
        if (v >= 1) t = v;
    }
    cached = std::max(1, std::min(t, 512));
    return cached;
}

int host_omp_threads(MPI_Comm comm)
{
    if(host_worker_cache()>0)return host_worker_cache();
    int ranks_on_node=1;
    MPI_Comm node=MPI_COMM_NULL;
    if(MPI_Comm_split_type(comm,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node)==MPI_SUCCESS) {
        MPI_Comm_size(node,&ranks_on_node);
        MPI_Comm_free(&node);
    }
    return host_omp_threads_for_node(ranks_on_node);
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
