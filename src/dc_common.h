#ifndef DDLA_DC_COMMON_H
#define DDLA_DC_COMMON_H

// ---------------------------------------------------------------------------
// Plain helpers shared by the two distributed divide-and-conquer drivers:
//   src/pstedc.cpp      symmetric tridiagonal eigenproblem (stage 3 of pheevd)
//   src/pbdsdc_dc.cpp   bidiagonal SVD (stage 2 of pgesvd)
// Everything lives in an ANONYMOUS namespace.  NOTE: nvcc-compiled TUs export
// anonymous-namespace functions as WEAK dynamic symbols (nm -D shows them), so
// a same-named, same-signature helper with a DIFFERENT body in two TUs would
// silently collapse to one definition at link time.  These helpers are
// therefore required to be identical in both users -- that is what sharing
// them here guarantees (the cuobjdump -sass of the moved kernels was verified
// identical before and after the extraction); keep identical compile flags on
// both TUs (no per-TU -fmad option).  Algorithm-specific code (root splitting,
// the z-hat formula, the GEMM blocks) is NOT shared: LAED* and LASD* differ.
// ---------------------------------------------------------------------------

#include "ddla_connector.h"
#include "mpi_datatype.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>

// Host ?LAMRG through the Fortran ABI (hidden CHARACTER lengths: none).
extern "C" {
void slamrg_(const int*, const int*, const float*, const int*, const int*, int*);
void dlamrg_(const int*, const int*, const double*, const int*, const int*, int*);
}

namespace ddla {

namespace {

inline void lamrg(int n1, int n2, const float* d, int dtrd1, int dtrd2, int* perm)
{
    slamrg_(&n1, &n2, d, &dtrd1, &dtrd2, perm);
}
inline void lamrg(int n1, int n2, const double* d, int dtrd1, int dtrd2, int* perm)
{
    dlamrg_(&n1, &n2, d, &dtrd1, &dtrd2, perm);
}

// Leaf size of the D&C tree: the value of environment variable `env` when it
// is an integer >= 2, else `dflt`.  Callers read it on rank 0 only and
// broadcast it, so a node with a different environment cannot build a
// different tree.
inline int dc_leaf_size(const char* env, int dflt)
{
    const char* s = std::getenv(env);
    if (s != nullptr && *s != '\0') {
        const int v = std::atoi(s);
        if (v >= 2) return v;
    }
    return dflt;
}

// Every rank must agree on a failure before anyone throws: a lone throw would
// leave the others blocked in the next collective.  The message reads
// "<who>: <what> failed (info=<err>)".
inline void agree_or_throw(MPI_Comm comm, int local_err, const char* who, const char* what)
{
    int err = local_err;
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_INT, MPI_MAX, comm));
    if (err != 0)
        throw std::runtime_error(std::string(who) + ": " + what +
                                 " failed (info=" + std::to_string(err) + ")");
}

#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)

// Givens rotations on column pairs of an m-row block, in list order (a
// column may appear in several rotations, so the list is sequential; rows
// are independent).  givcol holds 1-BASED column pairs (as ?LAED8 / the
// ?LASD2 port produce them).
template <typename rt>
__global__ void dc_givens_kernel(rt* __restrict__ q, int ldq, int m,
                                 const int* __restrict__ givcol,
                                 const rt* __restrict__ givnum, int ng)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m) return;
    for (int g = 0; g < ng; ++g) {
        rt* pa = q + static_cast<std::size_t>(givcol[2 * g] - 1) * ldq + r;
        rt* pb = q + static_cast<std::size_t>(givcol[2 * g + 1] - 1) * ldq + r;
        const rt c = givnum[2 * g];
        const rt s = givnum[2 * g + 1];
        const rt x = *pa;
        const rt y = *pb;
        *pa = c * x + s * y;
        *pb = c * y - s * x;
    }
}

// dst(:, j) = src(:, perm[j] - 1) for j < ncols, rows [0, m); perm is 1-based.
template <typename rt>
__global__ void dc_gather_kernel(rt* __restrict__ dst, int ldd,
                                 const rt* __restrict__ src, int lds, int m,
                                 const int* __restrict__ perm, int ncols)
{
    for (int j = blockIdx.y; j < ncols; j += gridDim.y) {
        const rt* sc = src + static_cast<std::size_t>(perm[j] - 1) * lds;
        rt* dc = dst + static_cast<std::size_t>(j) * ldd;
        for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < m;
             r += gridDim.x * blockDim.x)
            dc[r] = sc[r];
    }
}

template <typename rt>
void launch_gather(const DdlaHandle_t& handle, rt* dst, int ldd, const rt* src, int lds, int m,
                   const int* perm, int ncols)
{
    const gpuStream_t st = static_cast<gpuStream_t>(handle->stream);
    if (m <= 0 || ncols <= 0) return;
    const dim3 grid(static_cast<unsigned>(std::min(8, (m + 255) / 256)),
                    static_cast<unsigned>(std::min(ncols, 65535)));
    dc_gather_kernel<rt><<<grid, 256, 0, st>>>(dst, ldd, src, lds, m, perm, ncols);
    DDLA_CHECK(ddlaGetLastError(handle));
}

#endif // CUDA/HIP

} // namespace

} // namespace ddla

#endif // DDLA_DC_COMMON_H
