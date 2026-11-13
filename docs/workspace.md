# Internal workspace interfaces

These private interfaces let a composite algorithm validate once, allocate
scratch once, and reuse it through a panel loop. Public BLAS/LAPACK and
distributed entry points keep their operation arguments and arrange temporary
storage internally. Public computational templates take scalar types only and
select the backend from the handle. The backend template arguments below
belong to private execution interfaces, where the enclosing algorithm has
already selected a backend; they are not public-call syntax. Private
declarations can change between releases.

## Check, size and run

A routine that uses scratch has three components:

1. `<routine>_check` validates arguments and freezes policy and geometry.
   Distributed checks preserve their collective error agreement.
2. `<routine>_workspace_size` writes `std::size_t` byte counts through output
   parameters: backend bytes first, host bytes second. A routine using only
   one kind reports only that kind. The query does not allocate, communicate,
   copy matrix data, launch kernels or synchronize. Vendor buffer-size queries
   and host LAPACK `lwork=-1` queries are permitted.
3. `<routine>_run` takes raw pointers and their byte counts after its operation
   arguments. It executes with the supplied scratch, without repeating argument
   validation, vendor workspace queries or LibDDLA scratch allocation.

Scratch-free operations, including homogeneous local GEMM and AXPY, have
check/run only. They have no size routine or dummy scratch argument.

There is no common workspace object or per-routine workspace-size struct.
Callers use `ddla::vector<char>` for backend memory and `std::vector<char>` for
host memory. Geometry and resource objects remain where they carry actual
algorithm policy, communicator state, MPI mappings, workers, events or pinned
upload slots. They are prepared outside raw execution.

## Local LAPACK example

This helper assumes an initialized handle and valid backend pointers. The
factorization overwrites A. It demonstrates the private declarations in
`src/lapack_impl.h`, which is available only to internal project code.

```cpp
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "lapack_impl.h"

template<ddla::DdlaBackend B, class T>
void local_lu(const ddla::DdlaHandle_t& h, int m, int n,
              T* a, int lda, int* pivots, int* info)
{
    ddla::lapack_impl::getrf_check<B>(h,m,n,a,lda,pivots,info);
    std::size_t backend_bytes=0;
    ddla::lapack_impl::getrf_workspace_size<T>(h,m,n,a,lda,&backend_bytes);
    ddla::vector<char> backend(h,backend_bytes);
    ddla::lapack_impl::getrf_run<B,T>(h,m,n,a,lda,pivots,info,
                                     backend.data(),backend_bytes);
}
```

`getrf` needs no host scratch on either backend; its size query returns zero
backend bytes on CPU. Routines such as QR still need CPU LAPACK work arrays.
A run derives the vendor `lwork` from its supplied byte capacity, bounded by
that vendor interface's integer range. A larger compatible buffer can serve a
smaller shape. Size each required shape outside the loop and retain the maximum.
Do not assume that querying one arbitrary shape bounds every other shape.

## Reusing distributed GEMM scratch

All ranks participate in the check, allocation agreement and every execution.
The example keeps dimensions, descriptors, policies and scalar zero/nonzero
cases fixed across repeats. Matrix contents may change between calls.

```cpp
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "check_agree.h"
#include "PBLAS/pgemm_impl.h"
#include <vector>

template<ddla::DdlaBackend B, class T>
void repeat_gemm(const ddla::DdlaHandle_t& h,int m,int n,int k,
                 const T& alpha,const T* a,const int* da,
                 const T* b,const int* db,const T& beta,T* c,const int* dc,
                 int repeats)
{
    using namespace ddla;
    const auto options=pgemm_impl::pgemm_check<B>(
        h,'N','N',m,n,k,alpha==T(0),da,db,dc,pgemm_impl::Algorithm::Auto);
    if(options.noop)return;
    std::size_t backend_bytes=0,host_bytes=0;
    vector<char> backend(h);
    std::vector<char> host;
    detail::agree_check(h,"GEMM scratch setup",[&](detail::Agree&){
        pgemm_impl::pgemm_workspace_size<T>(
            h,options,'N','N',m,n,k,beta!=T(0),da,db,dc,
            &backend_bytes,&host_bytes);
        backend.resize(backend_bytes);
        host.resize(host_bytes);
    });
    for(int pass=0;pass<repeats;++pass){
        pgemm_impl::pgemm_run<B,T>(h,options,'N','N',m,n,k,
            alpha,a,da,b,db,beta,c,dc,
            backend.data(),backend_bytes,host.data(),host_bytes);
        DDLA_CHECK(ddlaSynchronize(h)); // host metadata may be refilled next
    }
}
```

The surrounding agreement belongs to allocation/setup, not the pure query.
It prevents an allocation failure on one rank from stranding its peers in
later communication. Existing public routines perform their own setup.

## Memory layout and composition

Each rank sizes its own scratch. Do not substitute a root rank's size, sum
sizes across ranks, or alias the two regions merely because both are CPU
memory on CPU handles. Backend scratch follows the handle's memory space;
host scratch must remain CPU-accessible on GPU handles too.

`src/plan_util.h` provides `detail::padded_bytes<T>(count)` and
`detail::take<T>(cursor,count)`. The first reports an element array's bytes
rounded to a 256-byte boundary; the second advances a raw cursor by exactly
that amount. Padding preserves alignment relative to the buffer's base.
Routines needing stronger base alignment include slack in their size and
align the base themselves. Ordinary vector allocations supply the alignment
required by the standard scalar and metadata types used in the other paths.

Size and execution must carve buffers in the same order. A composite run
places its persistent buffers first and passes the remainder to its children.
Sequential stages share the maximum of their scratch requirements. Buffers
with overlapping lifetimes require disjoint regions. These rules preserve
panel reuse and the eigensolver/SVD stage overlays without persistent offset
objects. A run may recompute pure integer geometry to split regions; it must
not repeat a vendor workspace query or allocate inside a panel loop.

Raw execution trusts the checked geometry and byte capacities. Passing too
little storage or incompatible policy is a caller error. Keep host object
lifetimes valid when interpreting byte storage as typed arrays, and do not
place owning containers in disposable scratch.

## Resources, ordering and errors

Keep the handle, matrices, scratch and prepared resources alive until every
queued consumer is finished. GPU calls retain their normal stream semantics.
Synchronize before a host consumer reads a GPU result or before host scratch
is overwritten or released. Pageable memory alone does not guarantee that an
asynchronous copy has finished reading its host source.

Raw paths that previously protected host scratch on exceptional exits retain
that protection with explicit error cleanup: queued work is drained before
the exception reaches the caller. This adds no synchronization to successful
execution. Operational GPU/BLAS/MPI errors and numerical `info` reporting
remain active.

Pinned replay upload rings, events, shared-memory chase resources and workers
remain explicit prepared resources. Byte counts describe numerical scratch;
they do not include every MPI mapping or persistent resource allocation.
Prepare and release collective resources in the same order on every rank.
Cancellation must stop workers and retire uploads before the caller reuses
storage. Changing shape, descriptors, backend, process topology or frozen
policy requires new sizing and preparation.

The runtime `HostStagedMPI` communication path uses the handle's reusable host
staging vectors, regardless of its build-time default. They can grow on demand on a cold handle; this is distinct
from caller-owned numerical scratch. Each tunnel wrapper completes its staged
communication before reusing these buffers. Do not infer cold-path allocation
freedom from allocation tests of the direct GPU collective path.

Select communication with `ddlaSetCommunication` between top-level operations,
before checking/sizing/preparing the next transport-dependent algorithm. After
changing mode, re-query/reprepare any private plan or resource whose geometry,
scratch or communication schedule depends on that mode. A switch after
`ddlaSet` is collective, drains pending handle work and must not run inside an
active communication group. It does not change the handle's matrix backend,
process grid or descriptor distribution. Capability queries do not establish
that the installed MPI accepts device pointers.

## Validation

The `test_workspace_*` programs exercise pure size queries, repeat execution,
exact-sized buffer guards, independent numerical references and resource
reuse. Load `libworkspace_trace.so` and set
`DDLA_WORKSPACE_TRACE_REQUIRED=1` to enable their allocation/query checks;
ordinary untraced runs alone cannot prove those properties. The diagnostic
library belongs to the tests, and adds no instrumentation to LibDDLA.

Compare the same public-API executable against baseline and candidate
libraries when checking numerical preservation. Use restored inputs and the
same hardware, rank/thread placement, warm-ups and repetitions for timing.
