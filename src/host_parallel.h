#ifndef DDLA_HOST_PARALLEL_H
#define DDLA_HOST_PARALLEL_H
#include <mpi.h>
#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>
namespace ddla { namespace detail {
constexpr std::size_t kHostMemsetParallelBytes = 2 * 1024 * 1024;
// Host storage only; synchronous, disjoint worker ranges and no nested teams.
void host_memset(void* pointer,int value,std::size_t bytes,int nthreads);
/// f(lo, hi) over [0, n) in chunks of `grain` indices (dynamic schedule) on a team of nthreads OpenMP threads,
/// the caller included (the team keeps that size when there are fewer chunks, so the runtime's thread pool is
/// not resized between regions); nthreads <= 1 or a single chunk runs f inline.  f runs concurrently, so it may only
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
