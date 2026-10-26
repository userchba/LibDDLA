# LibDDLA Python Bindings (`ddla`)

High-performance Python bindings for **LibDDLA** (Distributed Dense Linear Algebra).
Designed with native **NumPy** array interoperability and **mpi4py** integration, following the architectural paradigms of **ELPA (pyelpa in LibRPA)** and ScaLAPACK.

## Key Features

- **2D Block-Cyclic Distributed Matrix (`DistributedMatrix`)**:
  - Automatically derives process grids and ScaLAPACK `int[9]` descriptors.
  - Interoperates with `numpy.ndarray` on CPU (`order='F'`) with zero-copy data passing.
  - Native `gather` and `scatter` between local distributed partitions and global NumPy arrays.
  - GPU backend support for CUDA/HIP device memory buffers and stream synchronization.
- **Process Grid Management (`ProcessorLayout`)**:
  - Automatically factors communicator ranks into balanced 2D grid dimensions (`np_rows` x `np_cols`).
  - Seamlessly accepts `mpi4py.MPI.Comm` or falls back to built-in MPI.
- **High-Level Linear Algebra Operations**:
  - `pgemm`: Distributed matrix-matrix multiplication ($C \leftarrow \alpha \mathrm{op}(A)\mathrm{op}(B) + \beta C$).
  - `pgetrf` / `pgetrs` / `pgesv`: Distributed LU factorization and general linear system solve.
  - `ppotrf` / `ppotrs` / `pposv`: Distributed Cholesky factorization and symmetric/Hermitian positive-definite solve.
  - `pgeqrf`: Distributed Householder QR factorization.
  - `ptrtrs` / `ptrmm`: Distributed triangular solve and triangular matrix multiplication.
  - `pgeadd` / `pdam`: Distributed matrix addition and diagonal scaling.
- **Four Standard Precision Types**:
  - Single precision real (`float32` / `'s'`)
  - Double precision real (`float64` / `'d'`)
  - Single precision complex (`complex64` / `'c'`)
  - Double precision complex (`complex128` / `'z'`)

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
