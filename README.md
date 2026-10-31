# LibDDLA — Distributed Dense Linear Algebra

LibDDLA 0.0.6 is a C++17 template library for **distributed dense linear algebra**.
It provides ScaLAPACK-style APIs with 2D block-cyclic data distribution over an
MPI process grid, with a CPU (OpenBLAS/vendor BLAS), CUDA, or HIP backend
selected at build time, including an optional dual CPU+GPU build that compiles
both backends into one library and selects between them per handle at
runtime. All functions live in the `ddla` namespace.

## Key facts and caveats

- **Backend selection.** Enable `DDLA_USE_CPU`, `DDLA_USE_CUDA`, or
  `DDLA_USE_HIP` (at least one is required; `DDLA_USE_CUDA` and `DDLA_USE_HIP`
  are mutually exclusive). Combining `DDLA_USE_CPU` with `DDLA_USE_CUDA` or
  `DDLA_USE_HIP` builds a **dual** library containing both backends,
  selectable per `DdlaHandle_t` at runtime.
- **CPU and GPU coverage.** Both backends provide the local BLAS/LAPACK
  wrappers, including `gemm`, `trmm`, `trtrm`, `gemmBatched`, `getrf`, `potrf`,
  `geqrf` and `larft`; distributed GEMM, LU, Cholesky, QR/QL, triangular
  multiplication/solve, and data movement; and `random_generate`.
  These routines are templated on the
  backend, like `pgemm`: `pgetrf<DdlaBackend::CPU>(handle, ...)` runs on a CPU
  handle (host BLAS/LAPACK, MPI), `<DdlaBackend::GPU>` on a GPU handle, and the
  default is the build's default backend. Only the variable-size batched GEMMs
  (`gemmVbatched`, `gemmVbatched2s`) need `DDLA_USE_CUDA` or `DDLA_USE_HIP`
  (alone or in a dual build).
  `DDLA_USE_CCL` and `DDLA_USE_GPU_CPU_TUNNEL` are not supported with a
  CPU-only build, since there is no GPU backend to communicate with.
- **Supported scalar types** are `float`, `double`, `std::complex<float>`, and
  `std::complex<double>`. The distributed Cholesky family (`ppotrf` /
  `ppotrs` / `pposv`) is instantiated for all four types.
- **Submatrix support depends on the routine.** Most factorization/solve routines
  accept descriptors larger than the logical sub-matrix and operate only on
  the leading block anchored at global (0,0) (the same convention as
  `pgemm`); local extents are derived from the logical dimensions, not the
  descriptor's full size. `pdam` takes an optional `n` (default `-1` = the
  whole matrix). `pgetrf`, `pgeadd`, `gather` and `scatter` support explicit
  one-based global origins; the `ia/ja` arguments of `ppotrf` and the
  `ia/ja/ib/jb` arguments of `pposv` must currently be 1.
- **Matrix storage is caller-owned.** Use host pointers for CPU handles and
  device pointers for GPU handles. Individual routines may allocate temporary
  storage internally. `ddla::vector<T>` provides optional ownership of an
  uninitialized buffer in the handle's memory space; the handle must outlive it.
- **Integer dimensions (ScaLAPACK-compatible).** Global/local dimensions,
  strides, and indices use `int`, matching ScaLAPACK's descriptor convention.
  Matrices whose element count exceeds `INT_MAX` are not supported; extending
  to `int64_t` global dimensions is a deliberate future work item and is not
  part of the current API.
- An **installable CMake package config** is provided: after
  `cmake --install` (or `make install`), downstream builds can use
  `find_package(LibDDLA)` and link `ddla::ddla`. The package config locates
  MPI automatically and carries the include/library directories.

## Prerequisites

- **CMake** ≥ 3.18 (the source lists use `TARGET_DIRECTORY` properties)
- **C++17** compiler
- **MPI** (Open MPI or MPICH)
- **CPU backend:** a BLAS library (e.g. OpenBLAS) providing the standard
  Fortran `?gemm`/`?scal`/`?axpy`/`i?amax`/`?ger`/`?geru`/`?swap` symbols,
  the CBLAS `cblas_cdotc_sub` / `cblas_zdotc_sub` complex-dot interfaces, and
  the LAPACK `?getrf`/`?potrf`/`?laswp`/`?geqrf`/`?geqlf`/`?larft` ones. OpenBLAS
  and MKL carry LAPACK in the same library. Point CMake at it with
  `-DDDLA_CPU_BLAS_LIBRARY=/path/to/libopenblas.so`, or leave it unset to fall
  back to `find_package(BLAS REQUIRED)`.
- **CUDA backend:** CUDA Toolkit with cuBLAS, cuSOLVER, cuRAND
- **HIP backend:** ROCm / DTK with hipBLAS, hipSOLVER, hipRAND, and a CMake
  version with first-class HIP language support
- **Host LAPACK:** required by every source build, including GPU-only builds.
  Use `-DDDLA_LAPACK_LIBRARY=/path/to/libopenblas.so` when it is not already
  supplied by `DDLA_CPU_BLAS_LIBRARY` or found by `find_package(LAPACK)`.
  CMake also accepts an existing library path from the `OPENBLAS` environment
  variable when neither library option is set.
- **CCL mode:** NCCL or RCCL for direct inter-GPU collectives (enable
  `-DDDLA_USE_CCL=ON`)

The current HIP link configuration includes RCCL in its backend library list,
so RCCL must be available to HIP builds even when another communication mode is
selected.

The build system does **not** enforce a minimum CUDA/ROCm/NCCL version; use
toolchains that are compatible with C++17, the GPU hardware, and the required
math/solver libraries.

## Quick build

Architecture values (`CMAKE_CUDA_ARCHITECTURES`, `CMAKE_HIP_ARCHITECTURES`)
are hardware-specific — use the SM number for your NVIDIA GPU (e.g. `"80"` for
A100, `"70"` for V100) or the gfx target for your AMD GPU (e.g. `"gfx90a"` for
MI200 series).

### CPU-only

```bash
cmake -S . -B build-cpu                                 \
  -DDDLA_USE_CPU=ON                                      \
  -DDDLA_CPU_BLAS_LIBRARY=/path/to/libopenblas.so         \
  -DBUILD_TESTS=ON                                        \
  -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build-cpu -j
cmake --build build-cpu --target install
```

### CUDA

```bash
cmake -S . -B build-cuda                               \
  -DDDLA_USE_CUDA=ON                                   \
  -DDDLA_LAPACK_LIBRARY=/path/to/libopenblas.so         \
  -DCMAKE_CUDA_ARCHITECTURES="80"                      \
  -DBUILD_TESTS=ON                                     \
  -DDDLA_USE_CCL=ON                                    \
  -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build-cuda -j
cmake --build build-cuda --target install
```

### HIP / ROCm

```bash
cmake -S . -B build-hip                                \
  -DDDLA_USE_HIP=ON                                    \
  -DDDLA_LAPACK_LIBRARY=/path/to/libopenblas.so         \
  -DCMAKE_HIP_ARCHITECTURES="gfx90a"                   \
  -DBUILD_TESTS=ON                                     \
  -DDDLA_USE_CCL=ON                                    \
  -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build-hip -j
cmake --build build-hip --target install
```

### Dual CPU + GPU

Combine `DDLA_USE_CPU` with `DDLA_USE_CUDA` (or `DDLA_USE_HIP`) to build both
backends into one library, selectable per `DdlaHandle_t` at runtime:

The private `DdlaStream` has the same layout for CPU and GPU handles. Streams,
BLAS/solver handles and CCL communicators occupy opaque `void*` slots; CPU
handles leave them null. `src/ddla_stream.cpp` provides backend-templated
resource setup and cleanup in one translation unit. Dual builds compile each
library source once and generate both backend specializations in that pass;
they do not build a second CPU copy of the shared sources.

```bash
cmake -S . -B build-dual                                \
  -DDDLA_USE_CPU=ON                                      \
  -DDDLA_USE_CUDA=ON                                      \
  -DDDLA_CPU_BLAS_LIBRARY=/path/to/libopenblas.so         \
  -DCMAKE_CUDA_ARCHITECTURES="80"                         \
  -DBUILD_TESTS=ON                                        \
  -DDDLA_USE_CCL=ON                                       \
  -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build-dual -j
cmake --build build-dual --target install
```

The installed layout uses `include/ddla/*.h`, the platform library directory
(normally `lib/libddla.so` on Linux), and a CMake package under
`lib/cmake/LibDDLA`. Downstream builds can use:

```cmake
find_package(LibDDLA REQUIRED)
target_link_libraries(myapp PRIVATE ddla::ddla)
```

The installed `ddla::ddla` target supplies the include path, MPI dependency
and configured compile definitions. Public headers obtain build capabilities
from the generated `<ddla/ddla_config.h>` (`DDLA_HAS_CPU`, `DDLA_HAS_GPU`), so
applications using only the public API do not need private vendor headers.
In a dual build, `ddlaInit(handle)` and an omitted compute-template backend
default to GPU. For CPU calls, create a CPU handle and select
`DdlaBackend::CPU` explicitly on backend-templated compute routines.

## CMake options

| Option                     | Default | Description |
|---------------------------|---------|-------------|
| `BUILD_TESTS`             | OFF     | Build test executables |
| `DDLA_USE_CPU`            | OFF     | Build with the CPU backend (OpenBLAS/vendor BLAS); combine with `DDLA_USE_CUDA`/`DDLA_USE_HIP` for a dual CPU+GPU build |
| `DDLA_USE_CUDA`           | OFF     | Build for NVIDIA CUDA GPUs |
| `DDLA_USE_HIP`            | OFF     | Build for AMD HIP/ROCm GPUs |
| `DDLA_CPU_BLAS_LIBRARY`   | (empty) | Path to a BLAS library for the CPU backend; falls back to `find_package(BLAS REQUIRED)` if unset |
| `DDLA_LAPACK_LIBRARY`     | (empty) | Host LAPACK for all source builds; otherwise use `DDLA_CPU_BLAS_LIBRARY`, the `OPENBLAS` environment path, or LAPACK discovery |
| `DDLA_USE_DEBUG`          | OFF     | Enable `DDLA_USE_DEBUG` preprocessor macro |
| `DDLA_USE_CCL`            | OFF     | Use NCCL (CUDA) or RCCL (HIP) for device collectives (GPU backend required) |
| `DDLA_USE_GPU_CPU_TUNNEL` | OFF     | Route communication through host staging buffers (D2H → MPI → H2D) (GPU backend required) |
| `DDLA_ENABLE_FORTRAN`    | OFF     | Build `ddla_c`, the `ddla_fortran` library and Fortran module `ddla`; requires a Fortran compiler |
| `DDLA_ENABLE_PYTHON`     | OFF     | Build `ddla_c` and stage the Python package; requires a Python 3 interpreter |

When both `DDLA_USE_CCL` and `DDLA_USE_GPU_CPU_TUNNEL` are enabled, the
GPU-CPU tunnel path takes precedence for communication.  NCCL/RCCL libraries
are still linked, and CMake emits a warning noting that CCL will not be used.

## Communication modes

The library selects one of three communication paths at compile time:

| Path                             | Preprocessor                     | Behaviour |
|----------------------------------|----------------------------------|-----------|
| NCCL / RCCL                      | `DDLA_USE_CCL`                   | Direct device collectives on GPU-resident buffers |
| GPU-CPU tunnel                   | `DDLA_USE_GPU_CPU_TUNNEL`        | D2H copy → MPI collective → H2D copy |
| Synchronized MPI on device ptrs  | *(neither)*                      | `deviceStreamSynchronize` then GPU-aware MPI on device pointers |

All ranks must execute collectives in the same order regardless of the chosen
path.

All three paths are implemented once in `src/comm_traits.h` (`commSend`,
`commRecv`, `commBcast`, `commAlltoallv`), selected by the same
`DDLA_USE_CCL` / `DDLA_USE_GPU_CPU_TUNNEL` macros.

## Data model and lifecycle

### Handle

```cpp
#include <ddla/ddla.h>
#include <mpi.h>

ddla::DdlaHandle_t handle = nullptr;   // opaque pointer (DdlaStream*)
ddla::DDLA_CHECK(ddla::ddlaInit(handle));   // select backend and allocate handle
ddla::DDLA_CHECK(ddla::ddlaSet(handle, MPI_COMM_WORLD, 'R')); // create resources and auto grid
// or: ddla::DDLA_CHECK(ddla::ddlaSet(handle, MPI_COMM_WORLD, nprows, npcols, 'R'));
```

- `ddlaSet` stores the process-grid dimensions and communicator on the handle.
  The automatic form (`'R'`) derives a 2D grid from the number of MPI ranks.
- Initialize MPI before setting up the handle. In a dual build,
  `ddlaInit(handle, ddla::DdlaBackend::CPU)` selects the CPU backend;
  compute calls must use the matching template backend.
- **Destroy** with `ddla::DDLA_CHECK(ddla::ddlaDestroy(handle))` when done.

### Distributed matrix multiplication

`pgemm` selects its communication algorithm automatically from the matrix
shape, transpose flags, block sizes, and process grid. The AB branch broadcasts
panels of A and B while keeping C in place. For a thin output, BC keeps A in
place and communicates panels of B and C; AC keeps B in place and communicates
panels of A and C. The latter branches sum partial products to their output
owners, so C retains its ordinary two-dimensional block-cyclic distribution.
All branches support real/complex scalars and `N`, `T`, and `C` operations.

Selection uses ScaLAPACK 2.2.2's communication-volume estimates, with a preference
for AB when estimated costs are close. There is no public branch override.
`DDLA_PGEMM_KPANEL` continues to control K-panel batching in AB only; AC and BC
use one output descriptor block per panel, bounding temporary storage.

### Descriptor (ScaLAPACK `int[9]`)

Distributed routines describe matrix layouts with a plain length-9 int array
in ScaLAPACK's `DESCINIT` layout, plus the handle that owns the process grid.
Slot constants live in `<ddla/ddla_desc.h>`: `DDLA_DTYPE_`, `DDLA_CTXT_`,
`DDLA_M_`, `DDLA_N_`, `DDLA_MB_`, `DDLA_NB_`, `DDLA_RSRC_`, `DDLA_CSRC_`,
`DDLA_LLD_`, `DDLA_DLEN_`.

**Important:** `desc[DDLA_CTXT_]` is ignored. A BLACS context is an index into
a table private to the BLACS library that minted it, so the process grid,
communicators and stream always come from the `DdlaHandle_t` passed alongside
the descriptor. Every descriptor is validated against that grid, and a
mismatch (usually an `LLD_A` too small for the local row count) throws
`std::invalid_argument`.

**Important:** most factorization and solve routines require square blocks
(`mb == nb`).

```cpp
int m = 4096, n = 64;
int desc[ddla::DDLA_DLEN_];
ddla::DDLA_CHECK(ddla::ddlaDescInit(desc, handle, m, n, 64, 64, 0, 0));
// Over-allocated local buffer? Raise the leading dimension afterwards:
// desc[ddla::DDLA_LLD_] += extra_rows;
```

`ddlaDescInit` fills the descriptor the way `DESCINIT` does and derives the
tight `LLD = max(1, LOCr(M_A))` from the handle's grid.

Free index-mapping helpers are in `<ddla/ddla_desc.h>`:
`indxg2p`, `indxg2l`, `indxl2g`, `num_loc` (the ScaLAPACK convention), plus
desc-level `indx_g2l_r` / `indx_g2l_c` / `indx_l2g_r` / `indx_l2g_c`, each
taking `(const int* desc, const DdlaHandle_t& handle, index)`. Local extents
are not stored in a descriptor; derive them with
`num_loc(desc[DDLA_M_], desc[DDLA_MB_], myprow, desc[DDLA_RSRC_], nprows)`
after fetching the grid with `ddlaGetGridDims` / `ddlaGetGridCoords`.

### Local matrix storage

Each MPI rank owns a contiguous local submatrix of size `m_loc × n_loc` stored
in host memory for a CPU handle or device memory for a GPU handle. The caller
allocates and frees this memory, with helper calls taking the handle first.

```cpp
#include <ddla/ddla.h>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    ddla::DdlaHandle_t handle = nullptr;
    ddla::DDLA_CHECK(ddla::ddlaInit(handle));
    ddla::DDLA_CHECK(ddla::ddlaSet(handle, MPI_COMM_WORLD, 'R'));

    int n = 4096, nrhs = 64;
    int descA[ddla::DDLA_DLEN_], descB[ddla::DDLA_DLEN_];
    // Square blocks (mb == nb): use block size 64 for both matrices.
    ddla::DDLA_CHECK(ddla::ddlaDescInit(descA, handle, n, n, 64, 64, 0, 0));
    ddla::DDLA_CHECK(ddla::ddlaDescInit(descB, handle, n, nrhs, 64, 64, 0, 0));

    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddla::ddlaGetGridDims(handle, nprows, npcols);
    ddla::ddlaGetGridCoords(handle, myprow, mypcol);
    const long m_locA = ddla::num_loc(n, 64, myprow, 0, nprows);
    const long n_locA = ddla::num_loc(n, 64, mypcol, 0, npcols);
    const long m_locB = ddla::num_loc(n, 64, myprow, 0, nprows);
    const long n_locB = ddla::num_loc(nrhs, 64, mypcol, 0, npcols);

    double *d_A = nullptr, *d_B = nullptr;
    ddla::DDLA_CHECK(ddla::ddlaMalloc(handle, &d_A, m_locA * n_locA * sizeof(double)));
    ddla::DDLA_CHECK(ddla::ddlaMalloc(handle, &d_B, m_locB * n_locB * sizeof(double)));

    // ... fill matrices, call ddla routines ...

    ddla::DDLA_CHECK(ddla::ddlaFree(handle, d_A));
    ddla::DDLA_CHECK(ddla::ddlaFree(handle, d_B));
    ddla::DDLA_CHECK(ddla::ddlaDestroy(handle));
    MPI_Finalize();
    return 0;
}
```

The `ddlaMalloc` / `ddlaFree` helpers declared in
`<ddla/ddla_handle_t.h>` (pulled in by `<ddla/ddla.h>`) dispatch to the
handle's backend (host memory for CPU handles, device memory on the handle's
stream for GPU handles) and safely handle zero-byte requests (zero bytes sets
`*ptr = nullptr`).

Copies also take the handle first: `ddlaMemcpy(handle, dst, src, bytes)` and
`ddlaMemcpy2D(handle, dst, dpitch, src, spitch, width, height)`. There is no
copy-direction argument. GPU copies involving device memory are queued on
the handle; call `ddlaSynchronize(handle)` before reading their results on
the host. Pitches and copy sizes are in bytes.

## API overview

### Handle and descriptor

| Symbol | Description |
|--------|-------------|
| `DdlaHandle_t` | Opaque handle type (`DdlaStream*`) |
| `ddlaInit` | Allocate a handle and select its backend |
| `ddlaSet` | Configure the grid (auto `'R'` or explicit `nprows`×`npcols`) and create backend resources |
| `ddlaGetBackend` / `ddlaBackendAvailable` | Query the handle's backend / compiled backend availability |
| `ddlaDestroy` | Tear down handle (idempotent) |
| `ddlaGetStream` | Return the default device stream from a handle (CPU: `nullptr`) |
| `ddlaDescInit` | Fill a ScaLAPACK-layout `int[9]` descriptor (`DESCINIT` equivalent) |
| `int[9]` descriptor | 2D block-cyclic matrix layout, slots `DDLA_DTYPE_` … `DDLA_LLD_` |
| `ddlaGetGridDims` / `ddlaGetGridCoords` | Process grid of a handle |
| `indxg2p` / `indxg2l` / `indxl2g` / `num_loc` | Free index-mapping functions |
| `indx_g2l_r` / `indx_g2l_c` / `indx_l2g_r` / `indx_l2g_c` | Desc-level index mapping `(desc, handle, index)` |

### Distributed BLAS and data movement

| Function | Description |
|----------|-------------|
| `pgemm` | `C = α·op(A)·op(B) + β·C`; automatic AB/SUMMA, AC, or BC communication |
| `pgeadd` | `C = α·op(A) + β·op(B)` with required one-based `ia,ja`, `ib,jb`, `ic,jc`; aligned block layouts only (block sizes and within-block offsets match after `N/T/C`; `N` also requires matching owners); direct local `N,N` addition and one-destination local-region transpose on square grids (`m*n <= INT_MAX`); transpose-copy with `α=1`, `β=0` |
| `pdam` | Add scalar to the diagonal of a distributed matrix (leading-block, optional `n`; default `-1` = whole matrix) |
| `ptrmm` | In-place triangular multiply: `B = α·op(A)·B` or `B = α·B·op(A)` |
| `gather` / `scatter` | Move a submatrix between distributed storage and a column-major buffer on one root rank; explicit one-based origins, host buffers on CPU and device buffers on GPU |
| `transport_block` | Extract/transpose a contiguous block from a distributed matrix into a local buffer |

### Local BLAS and triangular products

Local calls perform no inter-rank communication. Their column-major matrices
must live in the handle's memory space, and GPU work uses the handle's stream.

| Function | Description |
|----------|-------------|
| `gemm` | `C = α·op(A)·op(B) + β·C`; also has a real-A × complex-B overload with complex C, α and β |
| `trmm` | Overwrite B with `α·op(A)·B` or `α·B·op(A)` |
| `trtrm` | `C = α·A·B + β·C` in the selected upper or lower triangle; both inputs use that same triangle |
| `symm` / `hemm` | Symmetric / Hermitian matrix multiplication |
| `syrk` / `syr2k` / `herk` / `her2k` | Symmetric / Hermitian rank-k and rank-2k updates |
| `gemmBatched` | Fixed-size GEMMs over arrays of pointers; arrays and matrices use the handle's memory space |

**TRMM interface change.** Local `trmm` now follows the in-place BLAS form:

```cpp
ddla::trmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
```

The separate output arguments of the former `trmm(..., B, ldb, C, ldc)`
interface were removed. To preserve B, first copy it into a destination,
then multiply that destination in place. A must not overlap B. The C ABI,
Fortran module and generated Python ctypes signatures use the same in-place
form; rebuild bindings and update old calls. Distributed `ptrmm` already
overwrites B and retains its existing signature.

`trtrm(handle, uplo, n, alpha, A, lda, B, ldb, beta, C, ldc)` supports
`uplo='U'` for upper × upper and `'L'` for lower × lower. Both diagonals are
explicit; neither operand is transposed or conjugated. The opposite triangle
of C and leading-dimension padding are preserved. A and B may coincide,
but C must not overlap either input.

### LU factorization, solve, and drivers

| Function | Description |
|----------|-------------|
| `pgetrf` | LU with partial (row) pivoting on an explicit one-based `ia,ja` submatrix |
| `pgetrf_bpiv` | Block LU with partial pivoting per block row (device pivot array) |
| `pgetrf_nopiv` | Multi-process LU without pivoting |
| `getrf_nopiv` | Local (single-process) LU without pivoting |
| `pgetrs` | Solve using pivoted LU factors: `op(A)·X = B` (side='L') or `X·op(A) = B` (side='R'), trans='N'/'T'/'C' |
| `pgetrs_nopiv` | Solve using no-pivot LU factors (same side/trans options) |
| `pgetrs_bpiv` | Solve using block-LU factors from `pgetrf_bpiv` (side, trans) |
| `pgesv` | Driver: LU + solve with pivoting (side, trans) |
| `pgesv_nopiv` | Driver: LU + solve without pivoting (side, trans) |
| `pgesv_bpiv` | Driver: block-LU + solve (side, trans) |
| `ptrtrs` | Distributed triangular solve (side × uplo × trans × diag) |
| `plapiv` | Apply pivot permutation to rows or columns, forward or backward |
| `pswap` | Swap rows or columns between two distributed matrices |

`pgetrf` uses a private blocked panel factorization, `pgetf2`; it is not a
public API. `DDLA_LU_PANEL=inplace` (default) or `gather` selects the panel
method, and all ranks must use the same setting.

### Cholesky

| Function | Description |
|----------|-------------|
| `ppotrf` | Standard Cholesky factorization (all four scalar types) |
| `ppotrs` | Solve using Cholesky factor (side='L'/'R'; trans='N'/'C') |
| `pposv` | Driver: Cholesky + solve (side='L'/'R'; trans='N'/'C') |
| `potrf_bottom_right` | Local bottom-right Cholesky (all four scalar types) |
| `ppotrf_bottom_right` | Distributed bottom-right Cholesky (all four scalar types) |

### QR / QL factorization

| Function | Description |
|----------|-------------|
| `pgeqrf(handle, uplo, m, n, ...)` | Distributed Householder QR (`'U'`: `A = Q·R`) or QL (`'L'`: `A = Q·L`); all four scalar types, with host `tau` indexed by local column |

Panels use a gathered local `geqrf` by default (`DDLA_QR_PANEL=gather`),
or the distributed `inplace` variant. All ranks must use the same selector.
Only QR and QL are implemented; right-side LQ and RQ forms are not provided.


**Solve semantics.** For `side='L'` the right-hand side B is `n × nrhs` and
the system is `op(A)·X = B`; for `side='R'` B is `nrhs × n` and the system is
`X·op(A) = B`.  The LU-family solves (`pgetrs`, `pgetrs_nopiv`, `pgetrs_bpiv`,
`pgesv`, `pgesv_nopiv`, `pgesv_bpiv`) accept `trans` = 'N', 'T' or 'C'.  The Cholesky solves
(`ppotrs`, `pposv`) accept `trans` = 'N' or 'C', which are equivalent for a
Hermitian matrix ('T' is not supported).  `plapiv` applies the pivot
permutation to either rows (`rowcol`='R') or columns (`rowcol`='C'), in
forward (`direc`='F') or backward (`direc`='B') order.  `ppotrs` applies the
head-correction relocation permutation to B itself (rows for side='L',
columns for side='R') when given the same `location` passed to `ppotrf`.

### Auxiliary and advanced

| Function | Description |
|----------|-------------|
| `geqrf(handle, uplo, m, n, A, lda, tau)` | Local QR / QL factorization, `uplo` = `'U'` (`A = Q·R`, LAPACK `?geqrf`) or `'L'` (`A = Q·L`, LAPACK `?geqlf`; cuSOLVER / hipSOLVER have no QL, so the GPU runs the vendor `geqrf` on the reversed matrix `J A J`); `tau` lives with the data |
| `larft(handle, direct, storev, n, k, V, ldv, tau, T, ldt)` | Local triangular factor `T` of a block reflector (LAPACK `?larft`, `storev` = `'C'`): `H = I - V·T·V^H` for `k` reflectors stored as `geqrf` / `geqlf` leave them, `direct` = `'F'` (`T` upper, `H = H(1)…H(k)`) or `'B'` (`T` lower, `H = H(k)…H(1)`); `T` is written whole (zeros in the other triangle).  The GPU needs no vendor routine (cuSOLVER has `Xlarft` only from CUDA 12.4, hipSOLVER none): `herk` forms `V^H V` and one small kernel runs the column recurrence |
| `gemmVbatched` | Batch of GEMMs with variable dimensions (device-resident dim arrays) |
| `gemmVbatched2s` | Two-stage variable-batch GEMM with reusable temporary |
| `random_generate` | Fill a backend-memory buffer with uniform random values |

`random_generate<Backend, T>` is declared in `<ddla/ddla.h>` (as are
`write_matrix` and all other backend-templated functions), implemented in
`src/random_generate.cpp`, and explicitly instantiated for `float`, `double`,
`std::complex<float>`, and `std::complex<double>`.

## Optional language bindings

`-DDDLA_ENABLE_FORTRAN=ON` builds the C-ABI bridge `ddla_c`, the Fortran
library `ddla_fortran`, and module `ddla`. The installed module is placed
under `include/ddla/`; link the matching Fortran and C-ABI libraries.

`-DDDLA_ENABLE_PYTHON=ON` builds `ddla_c` and stages the Python package.
Python does not require a Fortran compiler unless the Fortran option is also
enabled. Install the Python package separately and set `DDLA_LIB_PATH` to
the matching `libddla_c.so`; see [Python setup and examples](python/README.md).

## Testing

Build with `-DBUILD_TESTS=ON`.  Test executables are MPI programs and must be
launched with `mpirun`.  The total number of ranks must equal the product of
process-grid rows and columns (`nprows × npcols`), which defaults
automatically.  You can override the grid with `--grid`:

```bash
# Run a single test on 4 ranks (auto grid)
mpirun -np 4 build-cuda/tests/test_random_generate

# Explicit 2×2 grid for a test that supports --grid
mpirun -np 4 build-cuda/tests/test_api_grid_ptrtrs --grid 2x2
```

The files `tests/test_cuda.sh` and `tests/test_hip.sh` are cluster-specific
Slurm batch scripts.  They load modules, build, and run the test suite on
particular machines; adapt their module/partition settings to your own cluster.

On Slurm, request the MPI tasks before launching tests and keep the launched
rank count within that allocation. CTest's `PROCESSORS` property counts each
test's ranks; limit concurrent tests to the allocated rank budget. Single-node
CPU performance measurements should use all physical cores (for example,
4 ranks × 10 cores on HPC3 `p1`, or 4 × 12 on its 48-core nodes). Correctness
CI can use a smaller allocation.

The [HPC3 validation guide](.ci/hpc3/README.md) describes the CPU and four-/six-GPU
jobs, their resource requests and result logs. The
[KSSG validation guide](.ci/kssg/README.md) describes the manual HIP workflow.

## Repository layout

```
LibDDLA/
├── include/ddla/         Public headers
│   ├── ddla.h            Main API declarations (all Doxygen comments)
│   ├── ddla_handle_t.h   Handle type, init/set/destroy, memory & stream helpers
│   ├── ddla_desc.h       ScaLAPACK int[9] descriptor contract and index mapping helpers
│   ├── ddla_vector.h     Owned, uninitialized storage in the handle's memory space
│   └── ddla_config.h.in  Generated config header template (ddla_config.h at build time)
├── src/                  Implementation (one routine per .cpp) and private headers
│   └── comm_traits.h     Unified communication layer (commSend/commRecv/
│                         commBcast/commAlltoallv)
├── tests/                Integration tests (MPI executables)
├── benchmarks/           Benchmark data and supporting artifacts
├── install_scripts/      Backend-specific build/install scripts
├── .ci/                  Cluster CI control files (hpc3: CUDA, kssg: HIP/DCU)
├── cmake/                CMake helper modules
├── CMakeLists.txt        Top-level build
└── LICENSE               LGPL-3.0 license text
```

Implementation headers are private under `src/`. BLAS and LAPACK entry
checks and execution helpers are declared in `src/BLAS/blas_impl.h` and
`src/lapack_impl.h`; callers use the public functions in `<ddla/ddla.h>`.
The stream implementation, workspace helpers and vendor aliases also stay
private. Operation sources are grouped under `src/BLAS`, `src/PBLAS`,
`src/LU`, `src/Cholesky`, `src/QR` and `src/vbatched`.

The public `DDLA_HAS_CPU` / `DDLA_HAS_GPU` macros describe library
capabilities. Private `DDLA_USE_CUDA` / `DDLA_USE_HIP` definitions select the
vendor API available in a source file; `src/gpu/gpu.h` derives
`DDLA_GPU_VENDOR` from them. These conditions are distinct: a CPU-only
consumer can link a GPU-enabled library. `gpu_cpu.h` supplies compatibility
types for private shared code when neither vendor is selected.

## Version

**0.0.6** — defined in `src/version.h`.  The shared library SONAME tracks the
major version.

## License

LGPL v3.  See [LICENSE](./LICENSE).
