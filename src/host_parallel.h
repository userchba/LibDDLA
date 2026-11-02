#ifndef DDLA_HOST_PARALLEL_H
#define DDLA_HOST_PARALLEL_H
#include <mpi.h>
#include <functional>
#include <memory>
#include <type_traits>
namespace ddla { namespace detail {
/// Threads for this rank's host kernels.  Collective over comm the first time it is called
/// (one MPI_Comm_split_type), cached afterwards.
int host_omp_threads(MPI_Comm comm);

// Reads/initializes the same worker cache without MPI. The caller supplies
// its already computed shared-communicator size. This permits collective
// preflight when ranks have different prior cache histories.
int host_omp_threads_for_node(int ranks_on_node);

/// f(lo, hi) over [0, n) in chunks of `grain` indices (dynamic schedule) on up to nthreads OpenMP threads,
/// the caller included; nthreads <= 1 or a single chunk runs f inline.  f runs concurrently, so it may only
/// write disjoint output ranges; it must not call MPI.  An exception thrown by f is rethrown here.
/// Callers in nvcc-compiled files use this instead of OpenMP pragmas.
void host_parallel_for(long long n,long long grain,int nthreads,void* context,
                       void (*function)(void*,long long,long long));
// The callback is borrowed until the synchronous OpenMP region has joined.
// A lambda conversion to std::function may allocate; this adapter cannot.
template<class F>
void host_parallel_for(long long n,long long grain,int nthreads,F&& function)
{
    using Callable=typename std::remove_reference<F>::type;
    auto* context=const_cast<void*>(static_cast<const void*>(std::addressof(function)));
    host_parallel_for(n,grain,nthreads,context,[](void* pointer,long long lo,long long hi) {
        (*static_cast<Callable*>(pointer))(lo,hi);
    });
}
void host_parallel_for(long long n,long long grain,int nthreads,
                       const std::function<void(long long,long long)>& function);

}}
#endif
