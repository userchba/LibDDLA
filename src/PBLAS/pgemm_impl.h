#ifndef DDLA_PGEMM_IMPL_H
#define DDLA_PGEMM_IMPL_H

#include <ddla/ddla.h>

#include <cstddef>
#include <array>
#include <type_traits>

#include "BLAS/blas_impl.h"

namespace ddla { namespace pgemm_impl {

// Private entry points shared with the regression tests and benchmark. Public
// pgemm always selects Auto; there is no process-wide algorithm setting.
enum class Algorithm { Auto, AB, AC, BC };

Algorithm select(char ta, char tb, int m, int n, int k,
                 const int* da, const int* db, const int* dc, int pr, int pc);

// pgemm split into the three components of plan_util.h: check, workspace
// size, run.

// One run of consecutive k-blocks of the SUMMA branch: [start, start+kk) of k.
struct KStage {
    int start;
    int cnt;
    int kk;
};


// Component 1: validate every argument exactly like the public entry, resolve
// Auto, and read DDLA_PGEMM_KPANEL.  A null or mismatched handle throws
// std::runtime_error, everything else std::invalid_argument.  Collective:
// every rank throws if any rank does, and the environment-driven choice must
// agree across ranks. Active CCL groups are rejected for a
// nonzero product before panel events or copies can be submitted.
template <DdlaBackend Backend>
Algorithm pgemm_check(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
                      bool alpha_is_zero, const int* da, const int* db, const int* dc,
                      Algorithm algorithm, int& panel_width);

// Component 2: the scratch of the branch pgemm_check chose, for this m, n, k
// and these descriptors.  The SUMMA branch (AB) stages the k panels of A and B
// in two GPU slots or one CPU slot, plus one block tile of B, plus the transport scratch of a
// transposed operand. Consecutive logical k blocks are concatenated for all
// N/T/C combinations. GPU grouping is capped at a block-rounded rank share. A synchronous CPU
// panel uses up to four rank shares, capped below a full operand when
// descriptor blocks permit; block-rounded grids retain their one-block bound. Allocation covers only the
// largest actual stage. The handle prepares its
// GPU panel events before borrowed execution; no event creation occurs in
// size/run. The stationary branches (AC, BC) stage one panel of the
// thin operand and of the partial product, the transposed copies, and the
// scratch of the two pgeadd calls of every panel.  A mixed real/complex
// product adds the local GEMM's scratch.  beta_is_nonzero says whether the
// stationary branches read C.  No validation, no collectives.
template <typename T, typename TA = T, typename TB = T>
void pgemm_workspace_size(const DdlaHandle_t& h, Algorithm algorithm, int panel_width, bool alpha_is_zero, char ta, char tb,
                          int m, int n, int k, bool beta_is_nonzero,
                          const int* da, const int* db, const int* dc,
                          std::size_t* backend_bytes, std::size_t* host_bytes);

// Exact two-slot HostStagedMPI high water for the selected branch. Pure
// descriptor geometry, no allocation or collectives; CPU/CCL/MPI return zero.
std::array<std::size_t,2> pgemm_staging_bytes(const DdlaHandle_t& h,Algorithm algorithm,
    bool product,char ta,char tb,int m,int n,int k,bool beta_is_nonzero,
    const int* da,const int* db,const int* dc,
    std::size_t a_bytes,std::size_t b_bytes,std::size_t c_bytes);

// Component 3: C := alpha*op(A)*op(B) + beta*C. No geometry validation or algorithm
// allocation. GPU SUMMA transports on stream_data and computes on stream;
// prepared input/ready/reuse events protect operands and the two panel slots.
// Both touched queues are retired before SUMMA returns or rethrows. Existing
// vendor initialization may still require warm-up before strict allocation
// measurements. HostStagedMPI raw borrowers must reserve pgemm_staging_bytes
// in their owning preparation (or accept legacy on-demand cache growth).
// Public PGEMM reserves under its existing preparation agreement. GPU products
// require no active CCL group, as agreed by the checker; raw callers
// must preserve that collective precondition. Zero-product no-ops need no events.
template <DdlaBackend Backend, typename T, typename TA = T, typename TB = T>
void pgemm_run(const DdlaHandle_t& h, Algorithm algorithm, int panel_width, char ta, char tb,
               int m, int n, int k, const T& alpha,
               const TA* a, const int* da, const TB* b, const int* db,
               const T& beta, T* c, const int* dc,
               void* backend_work, std::size_t backend_bytes,
               void* host_work, std::size_t host_bytes);

// check -> size -> run with an explicit branch (the tests and the benchmark).
template <DdlaBackend Backend, typename T, typename TA = T, typename TB = T>
void run(const DdlaHandle_t& h, char ta, char tb,
         int m, int n, int k, const T& alpha,
         const TA* a, const int* da, const TB* b, const int* db,
         const T& beta, T* c, const int* dc, Algorithm algorithm);

// Whether the stationary branch (keep_a: BC, otherwise AC) can run: the thin
// operand's and C's panel owners must line up (pgemm_check throws otherwise).
bool stationary_aligned(bool keep_a, char ta, char tb, const int* da, const int* db, const int* dc);

// The stationary branches' part of pgemm_workspace_size and pgemm_run.
template <typename T, typename TA = T, typename TB = T>
void stationary_workspace_size(const DdlaHandle_t& h, bool keep_a, char ta, char tb,
                               int m, int n, int k, bool beta_is_nonzero,
                               const int* da, const int* db, const int* dc,
                               std::size_t* backend_bytes, std::size_t* host_bytes);

std::array<std::size_t,2> stationary_staging_bytes(const DdlaHandle_t& h,bool keep_a,
    char ta,char tb,int m,int n,int k,bool beta_is_nonzero,
    const int* da,const int* db,const int* dc,std::size_t thin_bytes,std::size_t result_bytes);

template <DdlaBackend Backend, typename T, typename TA = T, typename TB = T>
void stationary_run(const DdlaHandle_t& h, bool keep_a, char ta, char tb,
                    int m, int n, int k, const T& alpha,
                    const TA* a, const int* da, const TB* b, const int* db,
                    const T& beta, T* c, const int* dc,
                    void* backend_work, std::size_t backend_bytes,
                    void* host_work, std::size_t host_bytes);

using blas_impl::local_gemm_workspace_size;
using blas_impl::local_gemm_run;

}} // namespace ddla::pgemm_impl
#endif
