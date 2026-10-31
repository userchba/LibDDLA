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
  - `pgeqrf`: Distributed Householder QR factorization.
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
  -DDDLA_USE_CPU=ON \
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
The native library must include the backend requested by `Handle`.

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
