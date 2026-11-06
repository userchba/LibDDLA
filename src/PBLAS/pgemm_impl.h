#ifndef DDLA_PGEMM_IMPL_H
#define DDLA_PGEMM_IMPL_H

#include <ddla/ddla.h>

#include <cstddef>
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
// agree across ranks.
template <DdlaBackend Backend>
Algorithm pgemm_check(const DdlaHandle_t& h, char ta, char tb, int m, int n, int k,
                      bool alpha_is_zero, const int* da, const int* db, const int* dc,
                      Algorithm algorithm, int& panel_width);

// Component 2: the scratch of the branch pgemm_check chose, for this m, n, k
// and these descriptors.  The SUMMA branch (AB) stages the k panels of A and B
// double-buffered and one block tile of B, plus the transport scratch of a
// transposed operand; the stationary branches (AC, BC) stage one panel of the
// thin operand and of the partial product, the transposed copies, and the
// scratch of the two pgeadd calls of every panel.  A mixed real/complex
// product adds the local GEMM's scratch.  beta_is_nonzero says whether the
// stationary branches read C.  No validation, no collectives.
template <typename T, typename TA = T, typename TB = T>
void pgemm_workspace_size(const DdlaHandle_t& h, Algorithm algorithm, int panel_width, bool alpha_is_zero, char ta, char tb,
                          int m, int n, int k, bool beta_is_nonzero,
                          const int* da, const int* db, const int* dc,
                          std::size_t* backend_bytes, std::size_t* host_bytes);

// Component 3: C := alpha*op(A)*op(B) + beta*C.  No validation, no allocation.
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
