# LibDDLA — Distributed Dense Linear Algebra

LibDDLA 0.0.5 is a C++17 template library for **distributed dense linear algebra**.
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
- **The CPU-only backend has a reduced surface.** It currently covers the
  BLAS-1/2/3 wrappers (`gemm`, `scal`, `omatcopy`, `axpy`, `iamax`, `geru`) and
  `pgemm`. The distributed factorization/solve routines (`pgetrf`, `pgesv`,
  `ppotrf`, `pposv`, `ptran`, batched GEMM, etc.) are GPU-only and require
  `DDLA_USE_CUDA` or `DDLA_USE_HIP` (alone or in a dual build).
  `DDLA_USE_CCL` and `DDLA_USE_GPU_CPU_TUNNEL` are not supported with a
  CPU-only build, since there is no GPU backend to communicate with.
- **Supported scalar types** are `float`, `double`, `std::complex<float>`, and
  `std::complex<double>`. The distributed Cholesky family (`ppotrf` /
  `ppotrs` / `pposv`) is instantiated for all four types.
- **Leading-block sub-matrix support.** Most factorization/solve routines
  accept descriptors larger than the logical sub-matrix and operate only on
  the leading block anchored at global (0,0) (the same convention as
  `pgemm`); local extents are derived from the logical dimensions, not the
  descriptor's full size. `pdam` takes an optional `n` (default `-1` = the
  whole matrix), and `pgetf2` / `pgetf2_panel` take an explicit `n`.
- **Matrix storage is caller-owned.** LibDDLA routines operate on device pointers
  supplied by the caller. Individual routines may allocate and release temporary
  device workspaces internally.
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

- **CMake** ≥ 3.13
- **C++17** compiler
- **MPI** (Open MPI or MPICH)
- **CPU backend:** a BLAS library (e.g. OpenBLAS) providing the standard
  Fortran `?gemm`/`?scal`/`?axpy`/`i?amax`/`?ger`/`?geru` symbols and
  `cblas_?omatcopy`. Point CMake at it with
  `-DDDLA_CPU_BLAS_LIBRARY=/path/to/libopenblas.so`, or leave it unset to fall
  back to `find_package(BLAS REQUIRED)`.
- **CUDA backend:** CUDA Toolkit with cuBLAS, cuSOLVER, cuRAND
- **HIP backend:** ROCm / DTK with hipBLAS, hipSOLVER, hipRAND, and a CMake
  version with first-class HIP language support
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

Downstream translation units must still define the same backend macro(s) used
to build LibDDLA (`DDLA_USE_CPU`, `DDLA_USE_CUDA`, and/or `DDLA_USE_HIP`,
matching the build), because the backend selection is compile-time.

## CMake options

| Option                     | Default | Description |
|---------------------------|---------|-------------|
| `BUILD_TESTS`             | OFF     | Build test executables |
| `DDLA_USE_CPU`            | OFF     | Build with the CPU backend (OpenBLAS/vendor BLAS); combine with `DDLA_USE_CUDA`/`DDLA_USE_HIP` for a dual CPU+GPU build |
| `DDLA_USE_CUDA`           | OFF     | Build for NVIDIA CUDA GPUs |
| `DDLA_USE_HIP`            | OFF     | Build for AMD HIP/ROCm GPUs |
| `DDLA_CPU_BLAS_LIBRARY`   | (empty) | Path to a BLAS library for the CPU backend; falls back to `find_package(BLAS REQUIRED)` if unset |
| `DDLA_USE_DEBUG`          | OFF     | Enable `DDLA_USE_DEBUG` preprocessor macro |
| `DDLA_USE_CCL`            | OFF     | Use NCCL (CUDA) or RCCL (HIP) for device collectives (GPU backend required) |
| `DDLA_USE_GPU_CPU_TUNNEL` | OFF     | Route communication through host staging buffers (D2H → MPI → H2D) (GPU backend required) |

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
`DDLA_USE_CCL` / `DDLA_USE_GPU_CPU_TUNNEL` macros. This replaced the former
per-file `include/ddla/ddla_comm.h` idiom.

## Data model and lifecycle

### Handle

```cpp
#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>

ddla::DdlaHandle_t handle;          // opaque pointer (DdlaStream*)
ddla::ddla_init(handle);            // create BLAS/Solver handles and device streams
ddla::ddla_set(handle, MPI_COMM_WORLD, 'R');   // auto process grid (row-major)
// or: ddla::ddla_set(handle, MPI_COMM_WORLD, nprows, npcols, 'R');
```

- `ddla_set` stores the process-grid dimensions and communicator on the handle.
  The automatic form (`'R'`) derives a 2D grid from the number of MPI ranks.
- **Destroy** with `ddla::ddla_destroy(handle)` when done.

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
DDLA_CHECK(ddlaDescInit(desc, handle, m, n, 64, 64, 0, 0));  // mb == nb: square blocks
// Over-allocated local buffer? Raise the leading dimension afterwards:
// desc[DDLA_LLD_] += extra_rows;
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

### Local device matrices

Each MPI rank owns a contiguous local submatrix of size `m_loc × n_loc` stored
in GPU device memory.  The caller allocates and frees this memory.

```cpp
#include <ddla/ddla.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    ddla::DdlaHandle_t handle;
    ddla::ddlaInit(handle);
    ddla::ddlaSet(handle, MPI_COMM_WORLD, 'R');

    int n = 4096, nrhs = 64;
    int descA[ddla::DDLA_DLEN_], descB[ddla::DDLA_DLEN_];
    // Square blocks (mb == nb): use block size 64 for both matrices.
    ddla::DDLA_CHECK(ddlaDescInit(descA, handle, n, n, 64, 64, 0, 0));
    ddla::DDLA_CHECK(ddlaDescInit(descB, handle, n, nrhs, 64, 64, 0, 0));

    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddla::ddlaGetGridDims(handle, nprows, npcols);
    ddla::ddlaGetGridCoords(handle, myprow, mypcol);
    const long m_locA = ddla::num_loc(n, 64, myprow, 0, nprows);
    const long n_locB = ddla::num_loc(nrhs, 64, mypcol, 0, npcols);

    double *d_A = nullptr, *d_B = nullptr;
    ddla::ddlaMalloc(&d_A, m_locA * 64 * sizeof(double), handle);
    ddla::ddlaMalloc(&d_B, n_locB * 64 * sizeof(double), handle);

    // ... fill matrices, call ddla routines ...

    ddla::ddlaFree(d_A, handle);
    ddla::ddlaFree(d_B, handle);
    ddla::ddlaDestroy(handle);
    MPI_Finalize();
    return 0;
}
```

The `ddla::ddla_malloc` / `ddla::ddla_free` helpers declared in
`<ddla/ddla_handle_t.h>` dispatch to the handle's backend (host memory for
CPU handles, device memory on the handle's stream for GPU handles) and safely
handle zero-byte requests (zero bytes sets `*ptr = nullptr`).

## API overview

### Handle and descriptor

| Symbol | Description |
|--------|-------------|
| `DdlaHandle_t` | Opaque handle type (`DdlaStream*`) |
| `ddla_init` | Allocate handle (streams, BLAS/Solver handles, optional CCL comms) |
| `ddla_set` | Configure process grid (auto `'R'` or explicit `nprows`×`npcols`) |
| `ddla_destroy` | Tear down handle |
| `ddla_get_stream` | Return the default device stream from a handle |
| `ddlaDescInit` | Fill a ScaLAPACK-layout `int[9]` descriptor (`DESCINIT` equivalent) |
| `int[9]` descriptor | 2D block-cyclic matrix layout, slots `DDLA_DTYPE_` … `DDLA_LLD_` |
| `ddlaGetGridDims` / `ddlaGetGridCoords` | Process grid of a handle |
| `indxg2p` / `indxg2l` / `indxl2g` / `num_loc` | Free index-mapping functions |
| `indx_g2l_r` / `indx_g2l_c` / `indx_l2g_r` / `indx_l2g_c` | Desc-level index mapping `(desc, handle, index)` |

### Distributed BLAS and data movement

| Function | Description |
|----------|-------------|
| `pgemm` | `C = α·op(A)·op(B) + β·C` (SUMMA algorithm) |
| `pgeadd` | `C = α·op(A) + β·op(B)` |
| `pdam` | Add scalar to the diagonal of a distributed matrix (leading-block, optional `n`; default `-1` = whole matrix) |
| `ptran` | Out-of-place distributed transpose (with optional conjugate) |
| `transport_block` | Extract/transpose a contiguous block from a distributed matrix into a local buffer |

### LU factorization, solve, and drivers

| Function | Description |
|----------|-------------|
| `pgetf2` | Unblocked panel LU (inner kernel, host pivot array) |
| `pgetf2_panel` | Alternative panel LU (rank-revealing variant) |
| `pgetrf` | LU with partial (row) pivoting |
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

### Cholesky

| Function | Description |
|----------|-------------|
| `ppotrf` | Standard Cholesky factorization (all four scalar types) |
| `ppotrs` | Solve using Cholesky factor (side='L'/'R'; trans='N'/'C') |
| `pposv` | Driver: Cholesky + solve (side='L'/'R'; trans='N'/'C') |
| `potrf_bottom_right` | Local bottom-right Cholesky (all four scalar types) |
| `ppotrf_bottom_right` | Distributed bottom-right Cholesky (all four scalar types) |

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
| `gemmVbatched` | Batch of GEMMs with variable dimensions (device-resident dim arrays) |
| `gemmVbatched2s` | Two-stage variable-batch GEMM with reusable temporary |
| `random_generate` | Fill device buffer with uniform random values |

`random_generate<Backend, T>` is declared in `<ddla/random_generate.h>` as a
backend-templated function (the same shape as `gemm` / `scal` /
`write_matrix`), implemented in `src/random_generate.cpp`, and explicitly
instantiated for `float`, `double`, `std::complex<float>`, and
`std::complex<double>`.

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

## Repository layout

```
LibDDLA/
├── include/ddla/         Public headers
│   ├── ddla.h            Main API declarations (all Doxygen comments)
│   ├── ddla_handle_t.h   Handle type, init/set/destroy, ddla_malloc/ddla_free
│   ├── ddla_desc.h       ScaLAPACK int[9] descriptor contract and index mapping helpers
│   ├── ddla_connector.h  CUDA/HIP type aliases, macros, CHECK utilities, runtime alloc/copy wrappers
│   ├── ddla_stream.h     DdlaStream (internal)
│   ├── transport_block.h Distributed-to-local block extraction
│   ├── ptran.h           Distributed matrix transpose
│   ├── random_generate.h Uniform random fill (backend-templated)
│   ├── gemmVbatched.h    Variable-size batched GEMM
│   └── Backend wrappers  gemm.h, trsm.h, scal.h, axpy.h, swap.h, geru.h,
│                         iamax.h, geam.h, herk.h, syrk.h, trmm.h,
│                         gemmBatched.h, getrf.h, potrf.h, laswp.h
├── src/                  Implementation (one routine per .cpp)
│   └── comm_traits.h     Unified communication layer (commSend/commRecv/
│                         commBcast/commAlltoallv; replaces the former
│                         include/ddla/ddla_comm.h idiom)
├── tests/                Integration tests (MPI executables)
├── benchmarks/           Benchmark data and supporting artifacts
├── install_scripts/      Backend-specific build/install scripts
├── .ci/                  Cluster CI control files (hpc3: CUDA, kssg: HIP/DCU)
├── cmake/                CMake helper modules
├── CMakeLists.txt        Top-level build
└── LICENSE               LGPL-3.0 license text
```

## Version

**0.0.5** — defined in `src/version.h`.  The shared library SONAME tracks the
major version.

## License

LGPL v3.  See [LICENSE](./LICENSE).
