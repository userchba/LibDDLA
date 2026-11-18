#ifndef DDLA_PGEADD_IMPL_H
#define DDLA_PGEADD_IMPL_H

#include <ddla/ddla.h>

#include <cstddef>
#include <array>

namespace ddla { namespace pgeadd_impl {

// pgeadd split into the three components of plan_util.h: check, workspace
// size, run.  Library-internal callers (ptrmm, phemm, pgemmt, pgemm's
// stationary branches) size the scratch once for the largest panel they will
// add and call pgeadd_run for every panel, with no argument check and no
// allocation of their explicit scratch. GPU HostStagedMPI borrowers must
// additionally reserve pgeadd_staging_bytes in their owning preparation, or
// accept the legacy handle cache's on-demand allocation at communication.

// Component 1: validate every argument exactly like the public entry.  A
// null or mismatched handle throws std::runtime_error, everything else
// std::invalid_argument.  Collective: every rank throws if any rank does.
template <DdlaBackend Backend>
void pgeadd_check(const DdlaHandle_t& h, char transa, char transb, int m, int n,
                        const int* da, int ia, int ja,
                        const int* db, int ib, int jb,
                        const int* dc, int ic, int jc);

// Coefficient-aware validation used by the public wrapper. Unused inputs still
// satisfy the descriptor/region contract, but do not need MPI count capacity.
template <DdlaBackend Backend>
void pgeadd_check(const DdlaHandle_t& h, char transa, char transb, int m, int n,
                        const int* da, int ia, int ja,
                        const int* db, int ib, int jb,
                        const int* dc, int ic, int jc, bool use_a, bool use_b);

// Component 2: the scratch of calls of at most m x n. Source descriptors
// specify their block geometry when provided. use_a / use_b say whether
// alpha / beta can be nonzero (a zero
// coefficient reads no operand and needs no buffer).  Backend: the staging of
// the redistribution and of the prepared operands, and on a GPU handle the
// device copy of the redistribution schedule; host: that schedule (only for
// calls that can take the redistribution route).  No validation, no
// collectives.
template <typename T>
void pgeadd_workspace_size(const DdlaHandle_t& h, char ta, char tb, bool use_a, bool use_b,
                           int m, int n, const int* dc,
                           std::size_t* backend_bytes, std::size_t* host_bytes,
                           const int* da = nullptr, const int* db = nullptr,
                           bool allow_unaligned = false);

// Pass source descriptors and allow_unaligned=true when block sizes, phases
// or N owners may differ. Omitted descriptors retain the aligned-panel sizing
// contract used by existing internal callers. Bounds cover every start offset.

// Exact high-water bytes of the handle's two HostStagedMPI communication
// slots for this checked call. Geometry only: no allocation, MPI, or snapshot.
// Local alias snapshots already belong to backend_work and are not MPI payload.
// CPU and other transports return zero. Coordinates are one-based.
std::array<std::size_t,2> pgeadd_staging_bytes(const DdlaHandle_t& h,
    char ta, char tb, bool use_a, bool use_b, int m, int n,
    const int* da, int ia, int ja, const int* db, int ib, int jb,
    const int* dc, int ic, int jc, std::size_t element_bytes);

// Component 3: C := alpha*op(A) + beta*op(B). No validation or explicit scratch
// allocation; HostStagedMPI is allocation-free only after staging reservation.
// ta/tb are the normalized flags of pgeadd_check.  The host scratch holds
// schedule records uploaded to the device. A prepared handle event explicitly
// retires every host read before return or reuse for the second operand.
// Device arithmetic may remain pending on the compute queue. GPU raw callers
// must not have an active CCL communication group (the checker agrees this).
template <DdlaBackend Backend, typename T>
void pgeadd_run(const DdlaHandle_t& h, char ta, char tb, int m, int n,
                const T& alpha, const T* a, int ia, int ja, const int* da,
                const T& beta, const T* b, int ib, int jb, const int* db,
                T* c, int ic, int jc, const int* dc,
                void* backend_work, std::size_t backend_bytes,
                void* host_work, std::size_t host_bytes);

// Unit C := op(A) on an explicitly selected queue, with no shared BLAS-handle
// rebinding. Query pgeadd_workspace_size(op,'N',true,false,...) as usual.
// metadata_event is a caller-owned prepared handle event; host scratch has
// been consumed on return, including pageable or pinned metadata. Device
// copies/communication may remain pending: the caller must order panel use
// and retire backend scratch on this same queue. No event is created in run.
// Call outside an active CCL communication group, so nested work has actually
// been submitted before the metadata/ready events are recorded.
template <DdlaBackend Backend, typename T>
void pgeadd_copy_run(const DdlaHandle_t& h, char op, int m, int n,
    const T* a, int ia, int ja, const int* da, T* c, int ic, int jc, const int* dc,
    void* backend_work, std::size_t backend_bytes, void* host_work, std::size_t host_bytes,
    void* queue, ddlaEvent_t metadata_event);

}} // namespace ddla::pgeadd_impl

#endif // DDLA_PGEADD_IMPL_H
