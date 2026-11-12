# LibDDLA — Distributed Dense Linear Algebra

LibDDLA 0.0.6 is a C++17 template library for **distributed dense linear algebra**.
It provides ScaLAPACK-style APIs with 2D block-cyclic data distribution over an
MPI process grid. Every build includes the CPU backend (OpenBLAS/vendor
BLAS). CUDA or HIP can add GPU support to the same library, with backend
selection per handle at runtime. All functions live in the `ddla` namespace.

## Key facts and caveats

- **Backend selection.** The CPU backend is always built. Enable
  `DDLA_USE_CUDA` or `DDLA_USE_HIP` to add GPU support; the two GPU vendors
  are mutually exclusive. With neither enabled the build is CPU-only.
  There is no GPU-only configuration or `DDLA_USE_CPU` build option.
  CPU+GPU libraries select the backend per `DdlaHandle_t` at runtime.
- **CPU and GPU coverage.** Both backends provide the local BLAS/LAPACK
  wrappers, including `gemm`, `trmm`, `trtrm`, `gemmBatched`, `getrf`, `potrf`,
  `geqrf` and `larft`; distributed GEMM, LU, Cholesky, QR/QL, triangular
  multiplication/solve, and data movement; and `random_generate`.
  Public compute calls select the backend from their handle: the same
  `pgetrf(handle, ...)` call uses host BLAS/LAPACK and MPI on a CPU handle
  or the GPU implementation on a GPU handle. Scalar types are inferred from
  arguments or supplied explicitly, such as `pgetrf<double>(handle, ...)`.
  Only the variable-size batched GEMMs
  (`gemmVbatched`, `gemmVbatched2s`) need `DDLA_USE_CUDA` or `DDLA_USE_HIP`
  in a CPU+GPU build.
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
  Public convenience routines arrange their
  temporary storage. Internal callers can check arguments once, query local
  host/backend workspace byte counts without allocation, and reuse caller-owned
  `std::vector<char>` / `ddla::vector<char>` buffers through
  unchecked execution. Scratch-free routines such as AXPY have check/run only.
  See [internal workspace interfaces](docs/workspace.md) for examples and
  memory/resource lifetime requirements.
- **Integer dimensions (ScaLAPACK-compatible).** Global/local dimensions,
  strides, and indices use `int`, matching ScaLAPACK's descriptor convention.
  Matrices whose element count exceeds `INT_MAX` are not supported; extending
  to `int64_t` global dimensions is a deliberate future work item and is not
  part of the current API.
- An **installable CMake package config** is provided: after
  `cmake --install` (or `make install`), downstream builds can use
  `find_package(LibDDLA)` and link `ddla::ddla`. The package config locates
  MPI automatically and exports the real library target, including its SONAME,
  C++17 requirement and link dependencies. Private backend macros are not
  required by public headers.

## Prerequisites

- **CMake** ≥ 3.18 (the source lists use `TARGET_DIRECTORY` properties)
- **C++17** compiler
- **MPI** (Open MPI or MPICH)
- **Every build:** a BLAS library (e.g. OpenBLAS) providing the standard
  Fortran `?gemm`/`?scal`/`?axpy`/`i?amax`/`?ger`/`?geru`/`?swap` symbols,
  the CBLAS `cblas_cdotc_sub` / `cblas_zdotc_sub` complex-dot interfaces, and
  the LAPACK `?getrf`/`?potrf`/`?laswp`/`?geqrf`/`?geqlf`/`?larft` ones. OpenBLAS
  and MKL carry LAPACK in the same library. Point CMake at it with
  `-DDDLA_CPU_BLAS_LIBRARY=/path/to/libopenblas.so`, or leave it unset to fall
  back to `find_package(BLAS REQUIRED)`.
  All builds require both CBLAS DOT interfaces at configuration time;
  complex dot products always call the library, with no scalar fallback.
- **CUDA backend:** CUDA Toolkit with cuBLAS, cuSOLVER, cuRAND
- **HIP backend:** ROCm / DTK with hipBLAS, hipSOLVER, hipRAND, and a CMake
  version ≥ 3.21 for first-class HIP language support
- **Host LAPACK:** required by every source build.
  Use `-DDDLA_LAPACK_LIBRARY=/path/to/libopenblas.so` when it is not already
  supplied by `DDLA_CPU_BLAS_LIBRARY` or found by `find_package(LAPACK)`.
  CMake also accepts an existing BLAS/LAPACK library path from the `OPENBLAS`
  environment variable when `DDLA_CPU_BLAS_LIBRARY` is unset.
- **CCL mode:** NCCL or RCCL for direct inter-GPU collectives (enable
  `-DDDLA_USE_CCL=ON`)

The current HIP link configuration includes RCCL in its backend library list,
so RCCL must be available to HIP builds even when another communication mode is
selected.

The build system does **not** enforce a minimum CUDA/ROCm/NCCL version; use
toolchains that are compatible with C++17, the GPU hardware, and the required
math/solver libraries.

CUDA dependency discovery prefers `CUDA::cublas`, `CUDA::cublasLt`,
`CUDA::curand`, `CUDA::cusolver` and `CUDA::cudart` from `FindCUDAToolkit`.
Explicit `CUBLAS_LIBRARY`, `CUBLASLT_LIBRARY`, `CURAND_LIBRARY`,
`CUSOLVER_LIBRARY` and `CUDART_LIBRARY` paths override those selections.
For a split NVHPC layout, `DDLA_CUDA_MATH_LIBRARY_DIR` can supply a fallback
math-library directory; standard CUDA searches remain enabled. Configuration
reports `DDLA_CUDA_LIBRARY_PROVIDERS` for inspection.

## Quick build

Architecture values (`CMAKE_CUDA_ARCHITECTURES`, `CMAKE_HIP_ARCHITECTURES`)
are hardware-specific — use the SM number for your NVIDIA GPU (e.g. `"80"` for
A100, `"70"` for V100) or the gfx target for your AMD GPU (e.g. `"gfx90a"` for
MI200 series).

### CPU-only

```bash
cmake -S . -B build-cpu                                 \
  -DDDLA_CPU_BLAS_LIBRARY=/path/to/libopenblas.so         \
  -DBUILD_TESTS=ON                                        \
  -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build-cpu -j
cmake --build build-cpu --target install
```

#### Hermitian eigensolver method

The implementation is organized under `src/Eigensolver/one_stage/` and
`src/Eigensolver/two_stage/`. The public dispatcher and shared operations stay
in `src/Eigensolver/`.

`pheevd` reads `DDLA_EIGEN_METHOD` on every call:

- Unset, empty or `auto` (default): select from the measured one-stage/two-stage policy.
- `two_stage`: force full → band → tridiagonal.
- `one_stage`: force direct Householder reduction to tridiagonal.

Local `syevd` (real) and `heevd` (complex) use the same handle-driven
interface as the other single-process solvers:
`(handle, jobz, uplo, n, A, lda, w, info)`. A, w and the `int* info` are in the
handle's backend memory; GPU callers synchronize before copying device results to the host.
The private check, workspace query and unchecked execution can reuse scratch.

On a single-rank grid, automatic calls use LAPACK SYEVD/HEEVD on CPU or
cuSOLVER/hipSOLVER on GPU directly. The leading active block and UPLO are passed
to the local solver; eigenvalues still return in host memory, and eigenvectors
are copied to `Z` when necessary. Padding is preserved unless it overlaps active
`Z`. Exact in-place output needs no matrix copy; other overlapping A/Z layouts
use one compact temporary matrix. Explicit one-stage/two-stage choices retain
their algorithms, including on one rank.

Automatic selection on multiple ranks is conservative: it considers the scalar type, matrix order,
eigenvalues/eigenvectors request, block/grid layout, cached host thread budget,
shared-memory topology and GPU support. Unmeasured profiles retain two-stage.
Nonempty two-stage tuning options, such as `DDLA_EIG_NBW`, keep the two-stage
route; common tridiagonal-solver and host-thread controls apply to either method.
Selection performs no matrix allocation and does not change either algorithm's
workspace ownership. See the measured-profile table below for the calibrated
limits; these are heuristics, not a promise that one method always wins.

| CUDA scalar type | Eigenvalues only (`N`) | Eigenvectors (`V`) |
| --- | --- | --- |
| float | two-stage | one-stage for 4096–16384 |
| double | two-stage | one-stage for 3072–16384 |
| complex float | one-stage for 1536–16384 | one-stage for 1024–16384 |
| complex double | one-stage for 1024–16384 | one-stage for 1024–16384 |

These one-stage ranges require one shared-memory domain, eight CUDA SM70 GPUs
with NCCL (without host staging), a row-major 2×4 grid, square blocks of64,
source coordinates zero, matching output blocks, and one configured host and
recognized BLAS worker per rank. Other cases retain two-stage, including CPU,
HIP, host-staged transport, and orders above16384. CPU thresholds are not
inferred from GPU timings. Manual method choices remain available everywhere.

For example, `export DDLA_EIGEN_METHOD=one_stage` selects the one-stage solver
on CPU or GPU. All MPI ranks must select the same method; invalid values and
inconsistent selections are rejected collectively. `pheevd_one_stage` and
`pheevd_two_stage` select their algorithm explicitly and ignore this variable.
Both read only the selected input triangle and destroy A. The one-stage solver
completes the other triangle, then uses both halves as workspace for full-matrix
GEMV/GEMM; `uplo` selects the input triangle, not a separate reduction method.

#### MPI thread support

Initialize MPI with `MPI_Init_thread(..., MPI_THREAD_FUNNELED, &provided)` and
check that `provided >= MPI_THREAD_FUNNELED`. Call MPI-using LibDDLA routines
from the thread that initialized MPI. Internal CPU workers perform local
numerical work; communication and shared-window MPI calls stay on the caller.

The C++ API leaves MPI initialization and finalization to the application.
The bindings request FUNNELED when they initialize MPI automatically. If MPI
is already initialized, its thread level remains the application's responsibility.

#### CPU threads (OpenMP)

The CPU eigensolver (`pheevd`) scales with threads as well as ranks.  The host kernels that BLAS does not
cover (the stage-4 rotation replay, the stage-3 divide-and-conquer loops, the stage-2 chase team) run on
OpenMP threads (`DDLA_USE_OPENMP`, default ON when OpenMP is found; without it they run serially with
identical results).  Rules for a threaded run:

- Set `OMP_NUM_THREADS` to the threads per rank; `DDLA_EIG_THREADS` takes precedence.
  Without either positive override, each rank derives its budget from its allowed
  CPU affinity: each CPU contributes one divided by the number of node-local
  ranks allowed on that CPU, and the total is rounded down (minimum one).
  Disjoint ten-core bindings retain ten workers per rank; four ranks sharing
  forty allowed CPUs also receive ten. Partially overlapping masks share only
  their common CPUs. The budget is fixed when `ddlaSet` sets up the handle;
  later environment or affinity changes take effect after another `ddlaSet`.
- On CPU handles, `pheevd` / `pstedc` temporarily set recognized BLAS thread controls to one while
  multiple LAPACK STEDC leaves run on multiple host threads. The previous count is restored after the
  joined leaf loop, before merge GEMMs, and when that loop exits by exception. One-leaf and one-host-thread
  paths retain the caller's setting. This guard applies to CPU handles in CPU-only or dual builds;
  it does not change GPU-handle behavior.
- The vendor thread setting is process-global. Run one solver call per process, with no overlapping
  solves or unrelated concurrent BLAS calls while the guard is active. The guard recognizes
  OpenBLAS/MKL get/set control pairs; unsupported libraries retain their configuration. The validation
  below covers OpenBLAS and does not establish a threading guarantee for other backends.
- Bind each MPI rank to its allocated cores. Under Slurm request `P` tasks with `T` CPUs per task
  (`--ntasks=P --cpus-per-task=T`) and use `mpirun -np P --map-by slot:PE=T --bind-to core`.
- Results are bitwise independent of the thread count for the library's own loops; a threaded BLAS may
  change the last bits of the GEMM-based stages, as it does between BLAS builds.
- `geam` (matrix add / transpose-copy) on a CPU handle uses the same per-handle
  budget fixed by `ddlaSet`; its result is bitwise independent of the thread count.
- `gemmBatched` on a CPU handle uses the same budget: each OpenMP thread computes whole matrix products, so
  every core handles one multiplication at a time.  For the duration of that loop the BLAS library's own
  thread count (OpenBLAS `openblas_set_num_threads`, MKL `mkl_set_num_threads`, found at run time) is held at
  one and then restored, so even a pthreads OpenBLAS does not start a thread pool inside each call.  A single
  product or a one-thread budget runs serially and leaves the BLAS threads alone.
- The CPU two-stage solver can use Householder reflectors or Givens rotations.
  With square blocks and band width 64, its automatic Householder path supports
  one shared-memory node, n >= 2048, at most 64 ranks and 1–64 host workers per rank.
  The existing `DDLA_EIG_CPU_ACCUM=1` / `=0` controls select the Givens path's
  accumulation or replay engine and bypass automatic Householder selection.

With pthreads OpenBLAS 0.3.29 on HPC3, the following CPU threading setup was validated with one MPI
rank bound to eight cores. Use default eigensolver tuning for this example. The native test observed
BLAS1 inside parallel STEDC leaves, restoration to BLAS8 after success and exception unwind, correct
numerical results, and unchanged affinity. These settings are a tested configuration; performance depends on the matrix, process grid
and hardware.

```bash
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export OPENBLAS_MAIN_FREE=1 OMP_PROC_BIND=false
export OMP_DYNAMIC=false OMP_WAIT_POLICY=PASSIVE
# In a Slurm allocation with one task and eight CPUs per task:
mpirun -np 1 --map-by slot:PE=8 --bind-to core ./your_cpu_eigensolver
```

The GPU eigensolver applies distributed Householder reflectors or Givens rotations
without forming the dense band-to-tridiagonal factor. The tested eight-V100
CUDA/NCCL profile can select Householder automatically; other GPU profiles retain
conservative fallbacks. See the [`pheevd_two_stage` contract](include/ddla/ddla.h)
for bandwidth, topology, thread-budget and CPU/GPU stage-2 selection rules.
For multiple ranks with separate, matching compact input/output layouts, it reuses A and Z between
stages and temporarily backs up A in distributed host memory. Padded or different layouts use separate
distributed workspaces. The Givens stage-2 path distributes band cores between nodes
and rotation history between ranks; native GPU Householder stage 2 currently
requires one shared-memory node. Eigenvalues and tridiagonal diagonals remain replicated. A and Z may overlap: the solver preserves
the reflectors needed by its final back-transform.

### CPU + CUDA

```bash
cmake -S . -B build-cuda                               \
  -DDDLA_USE_CUDA=ON                                   \
  -DDDLA_CPU_BLAS_LIBRARY=/path/to/libopenblas.so       \
  -DCMAKE_CUDA_ARCHITECTURES="80"                      \
  -DBUILD_TESTS=ON                                     \
  -DDDLA_USE_CCL=ON                                    \
  -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build-cuda -j
cmake --build build-cuda --target install
```

### CPU + HIP / ROCm

```bash
cmake -S . -B build-hip                                \
  -DDDLA_USE_HIP=ON                                    \
  -DDDLA_CPU_BLAS_LIBRARY=/path/to/libopenblas.so       \
  -DCMAKE_HIP_ARCHITECTURES="gfx90a"                   \
  -DBUILD_TESTS=ON                                     \
  -DDDLA_USE_CCL=ON                                    \
  -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build-hip -j
cmake --build build-hip --target install
```

### Runtime backend selection

Every CUDA or HIP build contains both CPU and GPU specializations. No
additional CPU build switch is needed. Legacy caches with `DDLA_USE_CPU=OFF`
are migrated to CPU-enabled builds during configuration.

The private `DdlaStream` has the same layout for CPU and GPU handles. Streams,
BLAS/solver handles and CCL communicators occupy opaque `void*` slots; CPU
handles leave them null. `src/ddla_stream.cpp` provides backend-templated
resource setup and cleanup in one translation unit. Dual builds compile each
library source once and generate both backend specializations in that pass;
they do not build a second CPU copy of the shared sources.

The installed layout uses `include/ddla/*.h`, the platform library directory
(normally `lib/libddla.so` on Linux), and a CMake package under
`lib/cmake/LibDDLA`. Downstream builds can use:

```cmake
find_package(LibDDLA REQUIRED)
target_link_libraries(myapp PRIVATE ddla::ddla)
```

The installed `ddla::ddla` target supplies the include path, MPI dependency
and configured compile definitions. CPU support is unconditional; public headers
obtain GPU availability from `<ddla/ddla_config.h>` (`DDLA_HAS_GPU`), so
applications using only the public API do not need private vendor headers.
In a dual build, `ddlaInit(handle)` defaults to GPU. Use
`ddlaInit(handle, ddla::DdlaBackend::CPU)` for CPU data. Public computational
templates take scalar types only and always use the handle's backend; there
is no separate backend template argument to keep in sync. Rebuild C++ clients
after removing the backend template argument, for example using
`gemm<double>(handle, ...)` with a CPU handle for host data. This backend
selection change retains the Fortran/Python opaque-handle interface. The
separate Cholesky cleanup removes the sign/location arguments: regenerate
and rebuild native bindings together with their Fortran/Python consumers.

## CMake options

| Option                     | Default | Description |
|---------------------------|---------|-------------|
| `BUILD_TESTS`             | OFF     | Build test executables |
| `DDLA_USE_CUDA`           | OFF     | Add NVIDIA CUDA support to the CPU backend |
| `DDLA_USE_HIP`            | OFF     | Add AMD HIP/ROCm support to the CPU backend |
| `DDLA_CPU_BLAS_LIBRARY`   | (empty) | Path to a BLAS library for the CPU backend; falls back to `find_package(BLAS REQUIRED)` if unset |
| `DDLA_USE_OPENMP`         | ON      | OpenMP threads for the host kernels of the CPU eigensolver (`src/Eigensolver/eig_cpu_kernels.cpp`, compiled with the host compiler in every build); OFF or no OpenMP: serial, same results |
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
  subsequent compute calls use that handle's backend automatically.
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

### Products updating one triangle

`gemmt` (local) and `pgemmt` (distributed) compute only the selected triangle
of `C = alpha * op(A) * op(B) + beta * C`. The opposite triangle stays
untouched. A and B are general matrices; the result need not be symmetric or
Hermitian. C uses ordinary column-major storage, not packed triangular storage.

```cpp
// Lower triangle only; op(A) is n-by-k and op(B) is k-by-n.
ddla::gemmt(h, 'L', 'N', 'N', n, k,
    alpha, A, lda, B, ldb, beta, C, ldc);
ddla::pgemmt(h, 'L', 'N', 'N', n, k,
    alpha, A_local, descA, B_local, descB, beta, C_local, descC);
```

Both `'U'` and `'L'` and independent `'N'/'T'/'C'` operand flags are supported.
CPU and GPU handles use the same interface with pointers in the matching memory
space. Complex alpha, beta and diagonal entries are unrestricted.
`beta=0` reads no old C; `alpha=0` or `k=0` reads no A/B.
C must not overlap the inputs. The distributed routine supports rectangular
process grids and independent descriptor source coordinates, with a common
square block size and leading logical regions. Matrices remain distributed;
workspace consists of panels rather than replicated full matrices.


### Products of triangular matrices

`trtrm` and `ptrtrm` accept independent input triangles. Upper times lower and
lower times upper produce a full, generally nonsymmetric matrix. Matching
triangles update only the corresponding triangle of C. All diagonal entries
are explicit; unused input triangles need not be initialized.

```cpp
// Full C = alpha * U * L + beta * C.
ddla::trtrm(h, 'U', 'N', 'L', 'N', n,
    alpha, U, ldu, L, ldl, beta, C, ldc);
ddla::ptrtrm(h, 'U', 'N', 'L', 'N', n,
    alpha, U_local, descU, L_local, descL, beta, C_local, descC);
```

Both routines use the flag order `uplo_a, transa, uplo_b, transb`, with
`N`, `T`, and `C` for unchanged, transpose, and conjugate transpose. The uplo
flags describe stored triangles; transposition flips the effective triangle.
The effective triangles determine whether one triangle or all of C is updated.
Distributed `ptrtrm` redistributes one panel at a time for transposed operands;
its descriptors require a common square block size and matching source
coordinates. Both routines expose only this four-flag interface.

`benchmark_products cpu|gpu local|distributed s|d|c|z n [nb=128]
[warmups=3] [repeats=9]` compares these products and symmetric/Hermitian
products against GEMM/PGEMM. Run it under MPI; local mode requires one rank.
Set `DDLA_BENCH_CASES=sym` for symmetric/Hermitian cases or `tri` for
upper-times-lower and lower-times-upper cases; the default `all` runs both.
Set `DDLA_BENCH_GRID=2x2` for an explicit distributed grid. Each sample times
the complete public call, including internal allocation and communication,
with backend synchronization and the slowest-rank elapsed time. Input setup
and correctness checks are outside the timed interval. Raw samples include
both full-output comparisons and independent scalar-reference error checks.

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
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS)
        return 1;
    if (provided < MPI_THREAD_FUNNELED) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

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

The `ddlaMalloc` / `ddlaFree` / `ddlaMemcpy` / `ddlaMemcpy2D` helpers declared
in `<ddla/ddla_handle_t.h>` (pulled in by `<ddla/ddla.h>`) all take the handle
as their first argument and dispatch to the handle's backend (host memory for
CPU handles, device memory on the handle's stream for GPU handles). Zero-byte
requests are safe (a zero-byte `ddlaMalloc` sets `*ptr = nullptr`).

`ddlaMemcpy(handle, dst, src, bytes)` copies a contiguous range and
`ddlaMemcpy2D(handle, dst, dpitch, src, spitch, width, height)` copies a
pitched rectangle (pitches and width in bytes, as in `cudaMemcpy2D`). There is
no direction argument: a CPU handle copies host to host, and a GPU handle asks
the GPU runtime where each pointer lives (host memory or device memory, managed
memory counts as device) and chooses host to host, host to device, device to
host or device to device itself. A host-to-host copy is a plain `memcpy` that
runs immediately on the calling thread and is not ordered after the handle's
stream; the other directions are enqueued asynchronously on the handle's
stream, so call `ddlaSynchronize(handle)` before using a device-to-host
result.

`ddlaMemcpyTri(handle, uplo, m, n, dst, ldd, src, lds, include_diagonal=true)`
copies only a local column-major upper/lower triangle. Leading dimensions are
in elements; the other triangle and padding remain unchanged. The matching
`ddlaMemsetTri(handle, uplo, m, n, ptr, lda, value=0, include_diagonal=true)`
repeats a byte value, just like `ddlaMemset` (use zero for numerical zero).
Pass `false` to preserve the diagonal. Both accept trivially copyable element
types, including real and complex scalars. Large host operations use the
handle's OpenMP thread budget; GPU fills and device-to-device copies use one
kernel on the handle stream without scratch allocation. Host/device triangular
copies enqueue selected contiguous column segments, with the same pointer
direction and synchronization rules as `ddlaMemcpy`. GPU fills require device
or managed memory. These are local memory helpers, not distributed operations
or transpose/conjugation routines.

The same family covers the rest of what the library does with the device, so
that no CUDA or HIP type appears in your code either:

- `ddlaMalloc` also takes a typed pointer (`double* p; ddlaMalloc(handle, &p, bytes)`).
- `ddlaMemset(handle, ptr, value, bytes)` fills device memory (host memory on a
  CPU handle).
- `ddlaMallocHost` / `ddlaFreeHost` allocate page-locked host memory, the
  source and target of true asynchronous copies.
- `ddlaEventCreate` / `ddlaEventDestroy` / `ddlaEventRecord` / `ddlaEventWait` /
  `ddlaEventSynchronize` mark a point in the handle's queue and order work against it.
- `ddlaGetLastError(handle)` is the check after your own kernel launch,
  `ddlaSetDevice(handle)` selects the handle's device on a helper thread,
  `ddlaGetDeviceCount` / `ddlaMemGetInfo` query the device.

All of them return a `ddlaStatus_t`; `DDLA_CHECK(status)` throws on anything but
`DDLA_STATUS_SUCCESS`, with the vendor error text from `ddlaGetErrorString`
for a device failure.  On a CPU handle the event calls are no-ops.

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
| `ddlaMalloc` / `ddlaFree` | Allocate / free backend memory: `ddlaMalloc(handle, &ptr, bytes)`, `ddlaFree(handle, ptr)` |
| `ddlaMemcpy` / `ddlaMemcpy2D` | Contiguous / pitched copy: `ddlaMemcpy(handle, dst, src, bytes)`, `ddlaMemcpy2D(handle, dst, dpitch, src, spitch, width, height)`; the direction is derived from the handle and the pointers |
| `ddlaMemset` / `ddlaMallocHost` / `ddlaFreeHost` | Fill device memory / allocate and free pinned host memory |
| `ddlaEventCreate` / `ddlaEventDestroy` / `ddlaEventRecord` / `ddlaEventWait` / `ddlaEventSynchronize` | Mark a point in the handle's queue and order work against it |
| `ddlaGetLastError` / `ddlaSetDevice` / `ddlaGetDeviceCount` / `ddlaMemGetInfo` / `ddlaGetErrorString` | Launch-error check, device selection and queries, status text |
| `ddlaSynchronize` | Wait for the handle's stream (needed after an async `DeviceToHost` copy) |
| `ddlaDescInit` | Fill a ScaLAPACK-layout `int[9]` descriptor (`DESCINIT` equivalent) |
| `int[9]` descriptor | 2D block-cyclic matrix layout, slots `DDLA_DTYPE_` … `DDLA_LLD_` |
| `ddlaGetGridDims` / `ddlaGetGridCoords` | Process grid of a handle |
| `indxg2p` / `indxg2l` / `indxl2g` / `num_loc` | Free index-mapping functions |
| `indx_g2l_r` / `indx_g2l_c` / `indx_l2g_r` / `indx_l2g_c` | Desc-level index mapping `(desc, handle, index)` |

### Distributed BLAS and data movement

| Function | Description |
|----------|-------------|
| `pgemm` | `C = α·op(A)·op(B) + β·C`; automatic AB/SUMMA, AC, or BC communication |
| `pgeadd` | `C = α·op(A) + β·op(B)` with required one-based `ia,ja`, `ib,jb`, `ic,jc`. Supports independent block sizes, within-block offsets, and owners for `N/T/C`. Matching layouts retain direct local addition and whole-slab redistribution, including transpose on square grids (`m*n <= INT_MAX`). Other layouts use block-intersection redistribution. |
| `pdam` | Add scalar to the diagonal of a distributed matrix (leading-block, optional `n`; default `-1` = whole matrix) |
| `ptrmm` | In-place triangular multiply: `B = α·op(A)·B` or `B = α·B·op(A)` |
| `gather` / `scatter` | Move a submatrix between distributed storage and a column-major buffer on one root rank; explicit one-based origins, host buffers on CPU and device buffers on GPU |

### Local BLAS and triangular products

Local calls perform no inter-rank communication. Their column-major matrices
must live in the handle's memory space, and GPU work uses the handle's stream.

| Function | Description |
|----------|-------------|
| `gemm` | `C = α·op(A)·op(B) + β·C`; also has a real-A × complex-B overload with complex C, α and β |
| `trmm` | Overwrite B with `α·op(A)·B` or `α·B·op(A)` |
| `trtrm` | `C = α·op(A)·op(B) + β·C` with independent triangle and transpose flags for A and B |
| `symm` / `hemm` | Symmetric / Hermitian matrix multiplication |
| `symmt` / `hemmt` | Symmetric / Hermitian A times general B (or B times A), updating only the selected triangle of C |
| `syrk` / `syr2k` / `herk` / `her2k` | Symmetric / Hermitian rank-k and rank-2k updates |
| `gemmBatched` | Fixed-size GEMMs over arrays of pointers; arrays and matrices use the handle's memory space |

`trtrm(handle, uplo_a, transa, uplo_b, transb, n, alpha, A, lda, B, ldb, beta, C, ldc)`
reads the `U` or `L` stored triangle of each operand; `transa` and `transb` select
`N`, `T` or `C`. Both diagonals are explicit. Matching effective triangles update
only that triangle of C; mixed effective triangles update all of C. The unused
triangle of C and leading-dimension padding are preserved. A and B may coincide,
but C must not overlap either input. The previous single-`uplo` interface is
replaced by passing `uplo, 'N', uplo, 'N'`.

`symmt(handle, side, uploA, uploC, n, alpha, A, lda, B, ldb, beta, C, ldc)`
computes the selected `U` or `L` triangle of `C = alpha*A*B + beta*C`
(`side='L'`) or `C = alpha*B*A + beta*C` (`side='R'`). All matrices are
n-by-n. Only `uploA` of symmetric A is read; B is general. Complex symmetric
A is reflected by transpose and retains its complex diagonal. `hemmt` uses
conjugate reflection and a real Hermitian diagonal instead. The other triangle
and padding of C are preserved. C must not overlap A or B.

`psymmt(handle, side, uploA, uploC, n, alpha, A, descA, B, descB, beta, C, descC)`
is the distributed counterpart, with block-cyclic descriptors and the same
layout requirements as `psymm`; C uses square blocks. It shares `phemmt`'s
triangle-only update schedule. Both routines support CPU and GPU backends and
all four scalar types. Zero alpha skips A/B; zero beta skips reading the selected
C entries; n=0 or alpha=0,beta=1 permits null data pointers.

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
| `pgeqrf(handle, m, n, ...)` | Distributed Householder QR, `A = Q·R`; all four scalar types, with host `tau` indexed by local column |
| `pgeqlf(handle, m, n, ...)` | Distributed Householder QL, `A = Q·L`; host `tau` uses local-column indices, including the final reflector columns of wide matrices |

Panels use a gathered local `geqrf` by default (`DDLA_QR_PANEL=gather`),
or the distributed `inplace` variant. All ranks must use the same selector.
Only QR and QL are implemented; right-side LQ and RQ forms are not provided.

### Singular value decomposition

| Function | Description |
|----------|-------------|
| `pgesvd` | Driver: `A = U·diag(s)·V^H` (ScaLAPACK `PxGESVD` semantics: economy `U` m×k and `VT` k×n, `jobu`/`jobvt` = 'V'/'N', host `s` descending and replicated; U's rows aligned with A's, VT's columns aligned with A's; handle-selected backend, all four scalar types; the bidiagonal vector stage uses row-distributed divide and conquer on CPU and GPU, with O(k²/P) factor storage per rank and bounded leaf solves; `DDLA_SVD_DC_LEAF` selects the leaf size, default 128, capped at 256; values-only jobs use O(k) dqds storage. The obsolete `DDLA_SVD_BDSOLVER` selector is ignored; replicated `lapack`/`qr` modes have been removed) |
| `pbdsdc` | Bidiagonal divide-and-conquer SVD `B = U·diag(s)·V^T`, upper/lower real bidiagonal, float/double, CPU/GPU. Host `d`, `e`, `s` are replicated; `s` is descending. Both `U` and `V` return contiguous row slabs over all communicator ranks with local leading dimension `ld`; no descriptor-output or replicated-factor variant. Matrix-sized storage is distributed; bounded panels and scalar vectors may be replicated. |
| `pgebrd` | Householder bidiagonalization `A = Q·B·P^H` (ScaLAPACK `PxGEBRD`; upper bidiagonal for m >= n, lower for m < n; LAPACK `?GEBRD` storage; host `d`, `e`, `tauq`, `taup` of global length, replicated) |
| `punmbr` | Apply `Q`/`Q^H` from the left or `P`/`P^H` from the right (ScaLAPACK `PxUNMBR`; vect='Q' with side='L', vect='P' with side='R') |

**Solve semantics.** For `side='L'` the right-hand side B is `n × nrhs` and
the system is `op(A)·X = B`; for `side='R'` B is `nrhs × n` and the system is
`X·op(A) = B`.  The LU-family solves (`pgetrs`, `pgetrs_nopiv`, `pgetrs_bpiv`,
`pgesv`, `pgesv_nopiv`, `pgesv_bpiv`) accept `trans` = 'N', 'T' or 'C'.  The Cholesky solves
(`ppotrs`, `pposv`) accept `trans` = 'N' or 'C', which are equivalent for a
Hermitian matrix ('T' is not supported).  `plapiv` applies the pivot
permutation to either rows (`rowcol`='R') or columns (`rowcol`='C'), in
forward (`direc`='F') or backward (`direc`='B') order. Cholesky factorization
reports a nonpositive pivot through `info`; the driver solves only when
factorization succeeds. The factor passed to `ppotrs` is read-only.

### Auxiliary and advanced

| Function | Description |
|----------|-------------|
| `swap` | Swap two vectors (BLAS-1 `?swap`; strides may be negative) |
| `getrf` | Local LU with partial pivoting, `A = P·L·U` (LAPACK `?getrf`; `ipiv` and `info` live with the data) |
| `potrf` | Local Cholesky factorization, `uplo` = `'U'`/`'L'` (LAPACK `?potrf`; the other triangle is not touched) |
| `laswp` | Local row interchanges from a `getrf` pivot array (LAPACK `?laswp`; any nonzero `incx`) |
| `geqrf(handle, uplo, m, n, A, lda, tau)` | Local QR / QL factorization, `uplo` = `'U'` (`A = Q·R`, LAPACK `?geqrf`) or `'L'` (`A = Q·L`, LAPACK `?geqlf`; cuSOLVER / hipSOLVER have no QL, so the GPU runs the vendor `geqrf` on the reversed matrix `J A J`); `tau` lives with the data |
| `larft(handle, direct, storev, n, k, V, ldv, tau, T, ldt)` | Local triangular factor `T` of a block reflector (LAPACK `?larft`, `storev` = `'C'`): `H = I - V·T·V^H` for `k` reflectors stored as `geqrf` / `geqlf` leave them, `direct` = `'F'` (`T` upper, `H = H(1)…H(k)`) or `'B'` (`T` lower, `H = H(k)…H(1)`); `T` is written whole (zeros in the other triangle).  The GPU needs no vendor routine (cuSOLVER has `Xlarft` only from CUDA 12.4, hipSOLVER none): `herk` forms `V^H V` and one small kernel runs the column recurrence |
| `gemmVbatched(handle, ...)` | Batch of GEMMs with variable dimensions (device-resident dim arrays; GPU only) |
| `gemmVbatched2s(handle, ...)` | Two-stage variable-batch GEMM with reusable temporary (GPU only) |
| `random_generate(handle, data, n)` | Fill a buffer with uniform random values |

`swap`, `getrf`, `potrf`, `laswp`, `geqrf`, `larft`, `random_generate`,
`gemm`, `geam` and `write_matrix` are scalar templates declared in
`<ddla/ddla.h>`. These compute and matrix-output calls take the handle first;
backend selection comes from that handle. For example,
`write_matrix(handle, A, m, n, "matrix.txt")` consumes host A on a CPU handle
or device A on a GPU handle. Use a CPU handle when writing a host reference
array, even in a GPU-enabled program. Supported instantiations cover `float`,
`double`, `std::complex<float>` and `std::complex<double>`, subject to each
routine's real/complex constraints.

Pointers use the memory domain documented by each routine. Local BLAS/LAPACK
buffers live with the handle's backend -- host memory on CPU, device memory on
GPU -- including local `ipiv` and `info`. GPU calls are asynchronous on the
handle's stream, so read `info` after `ddlaSynchronize`. Distributed routines
may document host metadata separately. `gemmVbatched` / `gemmVbatched2s` also
use the handle-driven scalar interface but require a GPU handle; a CPU handle
is rejected with `std::invalid_argument`.

### Fortran bindings

Built with `-DDDLA_ENABLE_FORTRAN=ON` (needs a Fortran compiler, off by
default): `ddla_c` contains the C-ABI bridge (`fortran/ddla_c_api.cpp`), and
`ddla_fortran` supplies the Fortran module (`fortran/ddla.f90`, `bind(C)`
interfaces) and links to that bridge. Installing places both libraries next
to `libddla` and `ddla.mod` under `include/ddla`. The signature table in
`fortran/gen_bindings.py` generates the C-ABI, Fortran and Python ctypes files.
Enable `DDLA_ENABLE_PYTHON` to build the bridge and stage the Python package;
see [Python setup and examples](python/README.md).

Routines keep their names under a LAPACK-style type prefix: `ddla_s_*`,
`ddla_d_*`, `ddla_c_*` and `ddla_z_*` take `real(c_float)`, `real(c_double)`,
`complex(c_float)` and `complex(c_double)` data (`ddla_d_pgemm`,
`ddla_z_pgesv`, ...).  There are deliberately no generic names: most routines
pass their matrices as `type(c_ptr)`, which a Fortran generic cannot resolve
by scalar type.  Every binding returns `integer(c_int)` status, 0 = success
(`ddla_check(status)` aborts with the text from `ddla_error_text(status)`);
a C++ exception from the library surfaces as `DDLA_STATUS_EXCEPTION` with its
`what()` text.

The ScaLAPACK block-cyclic data model carries over unchanged: `ddla_descinit`
fills the same `integer(c_int) :: desc(9)` array ScaLAPACK's `DESCINIT` would
(`ddla_numloc` is the `LOCr`/`LOCc` helper), the process grid comes from the
handle (`ddla_init` + `ddla_set(handle, fcomm, nprows, npcols, 'R')`, with
0, 0 requesting the automatic grid), and the local pieces are plain Fortran
arrays -- `c_loc(array)` on a CPU handle, `ddla_malloc`/`ddla_memcpy` device
buffers on a GPU handle.  Complex scalars travel as the `bind(C)` derived
types `ddla_complex_float` / `ddla_complex_double` (the layout of
`complex(c_float)` / `complex(c_double)`).

```fortran
use ddla
include 'mpif.h'
type(c_ptr) :: h
integer(c_int) :: st, desca(9)
st = ddla_init(h)
st = ddla_set(h, int(MPI_COMM_WORLD, c_int), 0, 0, 'R')
st = ddla_descinit(desca, h, n, n, mb, mb, 0, 0, -1)
st = ddla_d_pgesv(h, 'L', 'N', n, nrhs, pa, desca, pb, descb)
```

Bound: the distributed family (`pgemm`, `pgetrf` + `_nopiv`/`_bpiv`,
`pgetrs` + `_nopiv`/`_bpiv`, `pgesv` + `_nopiv`/`_bpiv`, `ppotrf`, `ppotrs`,
`pposv`, `ppotrf_bottom_right`, `potrf_bottom_right`, `pgeqrf`, `pgeadd`,
`pdam`, `ptrtrs`, `ptrmm`, `plapiv`, `pswap`, `gather`, `scatter`), the local
BLAS/LAPACK wrappers (`gemm`, `gemv`, `trsm`, `trmm`, `trmv`, `syrk`, `herk`,
`her2k`, `hemv`, `hemm`, `her2`, `geam`, `trtrm`, `scal`, `axpy`, `swap`,
`geru`, `gerc`, `iamax`, `nrm2`, `dotc`, `getrf`, `potrf`, `laswp`, `geqrf`,
`larft`) and the lifecycle / descriptor / memory calls (`ddla_init`,
`ddla_init_backend`, `ddla_set`, `ddla_destroy`, `ddla_grid_dims`,
`ddla_grid_coords`, `ddla_rank`, `ddla_size`, `ddla_descinit`, `ddla_numloc`,
`ddla_malloc`, `ddla_free`, `ddla_memcpy`, `ddla_memcpy2d`, `ddla_memset`,
`ddla_synchronize`).  Not bound: the pointer-array APIs (`gemmBatched`,
`gemmVbatched`, `gemmVbatched2s`), the utilities
`random_generate` / `write_matrix`, and the mixed real-by-complex `gemm`
overload -- call those from C++.

`tests/test_fortran_api.f90` registers as the `test_fortran_api_mpi` ctest
when the option is on: handle lifecycle, descriptors, memory round trips, and
`pgemm` / `pgesv` / `ppotrf`+`pdam`+`ppotrs` / `pgeqrf` residual checks for
real and complex data on the build's default backend (device buffers +
`ddla_memcpy` on a GPU handle).

### In-place triangular multiplication

Both local `trmm` and distributed `ptrmm` overwrite `B`, following BLAS/PBLAS:

```cpp
trmm<T>(handle, side, uplo, trans, diag, m, n,
                 alpha, A, lda, B, ldb);
ptrmm<T>(handle, side, uplo, trans, diag, m, n,
                  alpha, A, descA, B, descB);
```

`A` is read-only and must not overlap `B`. The unused triangle and implicit
unit diagonal are not referenced. With `alpha == 0`, the logical entries of
`B` become zero without reading `A` or the original `B`; padding is preserved.
The former local `trmm(..., B, ldb, C, ldc)` interface has been removed.
Callers that need to preserve `B` must first copy it to their destination,
then call in-place `trmm` on that destination. Rebuild C++ clients and migrate
C, Fortran, and Python calls to the same shorter argument list. The distributed
`ptrmm` signature is unchanged. Internal QR, SVD, triangular-product and
Householder consumers use their existing work buffers in place.

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

To check an installation independently of the source build, run
`tests/install_consumer/check_install.sh <build-dir> <new-scratch-dir>` in the
validation allocation. It installs and relocates a temporary prefix, then uses
only `find_package(LibDDLA)` and `ddla::ddla` to build a separate C++ consumer.
The one-rank smoke checks the installed capability header and CPU computation;
a dual installation also links a GPU specialization without initializing a GPU.
See [the consumer fixture](tests/install_consumer/README.md) for audit outputs.

## Repository layout

```
LibDDLA/
├── include/ddla/         Public headers
│   ├── ddla.h            Main API declarations (all Doxygen comments)
│   ├── ddla_handle_t.h   Handle type, init/set/destroy, memory & stream helpers
│   ├── ddla_desc.h       ScaLAPACK int[9] descriptor contract and index mapping helpers
│   ├── ddla_vector.h     Owned, uninitialized storage in the handle's memory space
│   └── ddla_config.h.in  Generated config header template (ddla_config.h at build time)
├── src/                  Implementation (one routine per .cpp) and private headers,
│                         grouped per algorithm: LU/ Cholesky/ LDLT/ QR/ Eigensolver/ SVD/
│                         BLAS/ PBLAS/ (plus gpu/ and vbatched/); shared
│                         infrastructure (comm_traits.h, ddla_connector.h, ...) at root
│   └── comm_traits.h     Unified communication layer (commSend/commRecv/
│                         commBcast/commAlltoallv)
├── fortran/              Fortran bindings: C-ABI shim, bind(C) module, signature-table generator
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
`src/LU`, `src/Cholesky`, `src/LDLT`, `src/QR`, `src/Eigensolver`,
`src/SVD` and `src/vbatched`.

CPU support is unconditional. The generated `DDLA_HAS_GPU` macro gates GPU
code in the public headers and library sources. Private `DDLA_USE_CUDA` /
`DDLA_USE_HIP` definitions select the vendor API in `src/gpu/gpu.h`; a GPU
library build supplies exactly one. CPU-only builds use no dummy GPU types
or no-op vendor calls. Consumers use the public headers and can select the
CPU backend of a GPU-enabled library without vendor definitions.

## Version

**0.0.6** — defined in `src/version.h`.  The shared library SONAME tracks the
major version.

## License

LGPL v3.  See [LICENSE](./LICENSE).

`pstedc<T>(handle, jobz, n, d, e, w, z, ldz, info)` has two modes:
`jobz='N'` computes eigenvalues only; `jobz='V'` also returns eigenvectors as
contiguous row slabs in backend memory. For communicator rank q among P ranks,
`local_rows = n/P + (q < n%P)` and the first global row is
`q*(n/P) + min(q,n%P)`. Each rank stores all n columns with
`ldz >= max(1,local_rows)`. Column j corresponds to w[j]. Leading-dimension
padding and columns beyond n are preserved; zero-row ranks may pass null z.
The replicated host vectors d and e are destroyed, and w receives ascending
eigenvalues on every rank. For 'N', z and ldz are ignored and z may be null.

The former `descZ` overload and its automatic row-to-2D redistribution have
been removed. Callers must allocate the row layout described above.
`pstedc` uses the existing row-distributed divide-and-conquer engine, with
secular factors generated in bounded strips. The eigenvalues-only tree keeps
only the boundary rows needed by later merges.

Vector solves default to a requested leaf size of 1024. On multiple ranks,
the actual size is also limited by `ceil(n/2)` and
`max(256, ceil(n/P))`, including when `DDLA_EIG_DC_LEAF` requests larger leaves.
This distributes large leaf solves across ranks while avoiding extra merges
for small problems. Root calculation and vector normalization reuse the same
scratch because their lifetimes do not overlap.

The Hermitian one-stage and two-stage drivers use the same row-distributed
D&C solver. Each rank owns a contiguous share of rows and all
columns, keeping eigenvector storage distributed as O(n²/P). One-stage
redistributes bounded real panels into the output's 2D descriptor, widens
locally for complex output, and applies its stored reflectors. Two-stage
feeds the row slabs directly to its replay, Householder, or accumulated-Q
back-transform. Both drivers return eigenvectors in the caller's 2D output
descriptor and reuse caller-sized workspace across these phases. No dense
secular factor is replicated.

The GPU Householder route supports native host or GPU band-to-tridiagonal
reduction and a device back-transform. Reflector history is partitioned across
MPI shared windows on one node; each GPU receives bounded panels. Narrow
panels use register windows, wider panels use cooperative warps, and a GPU
BLAS/WY reference supports comparisons.

Automatic Householder selection is scoped to the measured CUDA SM70/NCCL
profile: one node, eight ranks in a row-major 2×4 grid, one host worker per
rank and block size 64, with full eigenvectors. Double defaults to
Householder/band64 from n=4096; complex double from n=16384. Stage 2 uses CPU
below n=16384 and GPU from n=16384. An explicit `DDLA_EIG_NBW=64` also
permits complex double from n=8192; explicit band16 selects Householder only
for double from n=32768. Other widths, scalar types, eigenvalues-only calls
and GPU profiles keep their existing policy. These thresholds describe this
scoped dispatch policy, not a performance guarantee on other hardware.

`test_pheevd_householder_gpu` checks numerical and layout behavior. Private
comparison entry points are the backend-templated `detail::pheevd_givens`,
`detail::pheevd_householder` and `detail::pheevd_householder_stage4`, declared
in `src/Eigensolver/two_stage/pheevd_impl.h`; the reduction-backend argument
selects CPU or GPU stage 2 independently of matrix storage.

Panel extraction and redistribution use `pgeadd` followed by row/column
broadcasts. `pgeadd` accepts independent block sizes, owners, and submatrix
offsets for N/T/C. Aligned panel copies retain the direct packing path.
The former public `transport_block` interface has been removed.
