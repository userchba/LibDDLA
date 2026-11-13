# LibDDLA Python Bindings (`ddla`)

High-performance Python bindings for **LibDDLA** (Distributed Dense Linear Algebra).
Designed with native **NumPy** array interoperability and **mpi4py** integration, following the architectural paradigms of **ELPA (pyelpa in LibRPA)** and ScaLAPACK.

## Key Features

- **2D Block-Cyclic Distributed Matrix (`DistributedMatrix`)**:
  - Automatically derives process grids and ScaLAPACK `int[9]` descriptors.
  - Interoperates with `numpy.ndarray` on CPU (`order='F'`) with zero-copy data passing.
  - `gather` and `scatter` between CPU distributed partitions and global NumPy arrays.
  - GPU device buffers with explicit `from_host` / `to_host` copies and handle synchronization.
- **Process Grid Management (`ProcessorLayout`)**:
  - Automatically factors communicator ranks into balanced 2D grid dimensions (`np_rows` x `np_cols`).
  - Seamlessly accepts `mpi4py.MPI.Comm` or falls back to built-in MPI.
- **High-Level Linear Algebra Operations**:
  - `pgemm`: Distributed matrix-matrix multiplication ($C \leftarrow \alpha \mathrm{op}(A)\mathrm{op}(B) + \beta C$).
  - `pgetrf` / `pgetrs` / `pgesv`: Distributed LU factorization and general linear system solve.
  - `ppotrf` / `ppotrs` / `pposv`: Distributed Cholesky factorization and symmetric/Hermitian positive-definite solve.
    `ppotrf` returns `(A, info)` and `pposv` returns `(B, info)`; `ppotrs` returns the solution matrix.
  - `pgeqrf`: Distributed Householder QR factorization.
  - `pgeqlf`: Distributed Householder QL factorization.
  - `ptrtrs` / `ptrmm`: Distributed triangular solve and triangular matrix multiplication.
  - `pgeadd` / `pdam`: Distributed matrix addition and addition of a scalar to the diagonal.
- **Four Standard Precision Types**:
  - Single precision real (`float32` / `'s'`)
  - Double precision real (`float64` / `'d'`)
  - Single precision complex (`complex64` / `'c'`)
  - Double precision complex (`complex128` / `'z'`)

## Build and library discovery

From the repository root, build the native C-ABI library before installing
the Python package. A Python-only binding build needs no Fortran compiler:

```bash
cmake -S . -B build-python \
  -DDDLA_CPU_BLAS_LIBRARY=/path/to/libopenblas.so \
  -DDDLA_ENABLE_PYTHON=ON
cmake --build build-python --target ddla_python --parallel 4
python3 -m pip install ./python

export DDLA_LIB_PATH="$PWD/build-python/fortran/libddla_c.so"
export LD_LIBRARY_PATH="$PWD/build-python/fortran:$PWD/build-python/src${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

The package requires NumPy; install `./python[mpi]` to add optional `mpi4py`.
`DDLA_LIB_PATH` names the native shared library itself. For an installed
native prefix, `DDLA_DIR=/path/to/install` searches its `lib` and `lib64`
directories. Load the matching MPI and backend runtime libraries as well.
The native library must include the backend requested by `Handle`. The
C++ public API now also selects computation from the handle; this does not
change Python `Handle`, type prefixes or C error statuses. The Cholesky
interface cleanup is described below. The generated C bridge calls the scalar-only C++ entry points
without a second CPU/GPU execution dispatcher.

Distributed operations are collective over the handle's communicator. Launch the example
below with an MPI rank count covered by the scheduler allocation, for example
`mpirun -n 4 python3 example.py`. It uses CPU handles and NumPy host arrays.
For GPU local-buffer copies, call `handle.synchronize()` before consuming
results returned by `to_host()`.

## Quickstart Example

```python
import numpy as np
import ddla

# 1. Initialize handle and 2D process grid
handle = ddla.Handle(backend=ddla.Backend.CPU)

# 2. Create distributed matrices (or scatter from global NumPy arrays)
N = 1000
nb = 64
np.random.seed(42)
global_A = np.random.randn(N, N)
global_B = np.random.randn(N, N)

A = ddla.DistributedMatrix.from_global_matrix(handle, global_A, mb=nb, nb=nb)
B = ddla.DistributedMatrix.from_global_matrix(handle, global_B, mb=nb, nb=nb)

# 3. Perform distributed matrix multiplication: C = A @ B
C = ddla.pgemm(A, B)

# 4. Gather result back to global NumPy array on rank 0
global_C = C.to_global_matrix(root=0)

if handle.rank == 0:
    diff = np.max(np.abs(global_C - global_A @ global_B))
    print(f"Max absolute difference vs NumPy: {diff:.2e}")

# 5. Clean up
handle.destroy()
```

## TRMM compatibility

The generated C-ABI ctypes bindings for local `trmm` now overwrite B and
take arguments through `B, ldb`; the former separate `C, ldc` output
arguments were removed. Update direct `_c_api` calls and use the generated
bindings from the same source revision as the native library. The high-level
distributed `ddla.ptrmm` interface already overwrites B and is unchanged.

## Cholesky compatibility

`ppotrf` returns `(A, info)` and `pposv` returns `(B, info)`; they no longer
return a sign flag. `ppotrs(A, B, uplo='L')` no longer accepts `is_nega` or
`location`, and its factor A is read-only. A nonpositive pivot is reported
through positive `info`; the driver leaves the right-hand side unchanged
when factorization fails. Use the regenerated ctypes/Fortran bindings and
native C library from the same source revision, since the native Cholesky
argument lists changed as well.

`Handle` requires an explicit `backend=Backend.CPU` or `backend=Backend.GPU`; omitting it or passing `None` is rejected.

## Per-handle communication

`handle.communication` reports the current `ddla.Communication` value. CPU
handles support `MPI`; GPU handles support `MPI` and `HOST_STAGED_MPI`, plus
`CCL` if the native library includes NCCL/RCCL. The initial GPU policy is
CCL when built, otherwise MPI. Host staging is always selectable at runtime. Communication is independent of the matrix backend.

```python
# All ranks using this GPU handle call the setter together after construction.
if handle.is_communication_available(ddla.Communication.HOST_STAGED_MPI):
    handle.set_communication(ddla.Communication.HOST_STAGED_MPI)
print(handle.communication)
```

The setter is collective on the handle's communicator. Use identical modes
and call order on every rank, between operations, outside communication groups
and without concurrent use of the handle. Invalid/mismatched/unavailable
requests raise the mapped DDLA exception and preserve the previous policy.
The choice persists across grid reconfiguration. Queries are local. On GPU,
`MPI` means device pointers are sent to MPI:
availability does not probe GPU-aware MPI support or guarantee its performance.
Use host staging when device-pointer MPI support has not been verified.

The Python communication tests default to CPU coverage. Set
`DDLA_TEST_GPU_COMMUNICATION=1` only in an allocation with one assigned GPU per
rank to enable the GPU setter/getter test; it switches policies without probing
GPU-aware MPI with device traffic.
