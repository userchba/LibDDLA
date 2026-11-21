# Changelog

## 0.0.7 — in preparation

Changes since `v0.0.6`. This release changes public signatures and memory
contracts. Rebuild downstream applications; see the
[migration and release guide](docs/releases/0.0.7.md) before upgrading.

### Added

- Mixed real/complex fixed-size batched GEMM, mixed variable-size batched
  GEMM, triangular-output GEMM, and symmetric/Hermitian products, including
  distributed variants. Matching real coefficients can scale complex matrix
  regions through `scal`, `pscal` and `scalBatched`.
- Local/distributed diagonal multiplication, triangular products, triangular
  memory operations, symmetric/Hermitian completion and diagonal log determinants.
- Local/distributed pivoted LDLT factorizations and solves; distributed QR
  least-squares; Q application/generation; local/distributed LQ and RQ.
- Distributed SVD with distributed bidiagonal factors and an opt-in QR
  reduction for eligible tall/wide problems.
- Distributed one-stage/two-stage Hermitian eigensolvers with adaptive method
  selection and a required eigenvector count, local SYEVD/HEEVD/STEDC, and
  generalized Hermitian-definite solvers. `phegst` can reuse a Cholesky factor.
- Opt-in Chebyshev-filtered partial Hermitian solver and opt-in selected
  tridiagonal eigenvectors for partial one-stage/two-stage solves.
- Per-handle runtime communication selection: MPI, host-staged MPI and
  compiled-in GPU collectives. Explicit-queue transfers support panel pipelining.
- Reusable internal check/query/run workspaces, transport-cache sizing and
  allocation diagnostics; generated public dispatch consistency checks and
  relocated installed-package tests.

### Changed and incompatible

Spectral APIs added after the 0.0.6 tag also evolved during development; their
final contracts below supersede intermediate `develop` interfaces.

- Every build includes CPU support; CUDA or HIP optionally adds one GPU backend.
  Public computational templates take scalar types only. Handles require an
  explicit CPU/GPU choice at initialization. Public consumers need C++11;
  building the implementation still requires C++17.
- Public GPU capability is expressed by `DDLA_HAS_GPU`; obsolete CPU capability
  guards and private backend definitions are unnecessary for public consumers.
- Distributed LU pivots use handle-selected memory, including GPU device memory.
- `ppotrf` returns `void` and reports through `info`; obsolete head correction
  arguments and behavior are removed. Rectangular process grids are supported.
- Distributed QR and QL have separate public entries. `transport_block` is
  removed; panel redistribution uses `pgeadd`.
- `pheevd` and its one-/two-stage entries require `nev` after `n`. They still
  return all `n` eigenvalues and only the requested lowest `nev` eigenvectors.
- `pstedc` takes a required `nblk`, supports values/vectors modes and returns
  row-distributed eigenvectors. Descriptor-output overloads and the old outer
  `DDLA_EIG_DC_LEAF` policy are removed. Multi-rank leaves use local STEDC.
- Replicated/root-gather spectral solver routes are removed. Global matrix-sized
  storage stays distributed, subject to descriptor/block and panel overhead.

### Fixes and performance work

- Collective preparation/failure agreement, transactional GPU stream replacement,
  subcommunicator device selection and asynchronous sink-counter lifetimes.
- NaN-safe zero-product handling in active GPU variable-batch GEMM paths and
  mixed batched leading-dimension overflow checks.
- Bounded QR logical panels independently of descriptor blocks; phased solver
  workspace reuse and host-staging reservation before numerical updates.
- CPU packing/thread policies, mixed GEMM layout reuse, SUMMA panel pipelining,
  and hardware-scoped eigensolver routing. These are workload-dependent changes,
  not a release-wide speedup guarantee. A small large-case CPU PSTEDC regression
  was accepted; see the [performance notes](docs/releases/0.0.7.md#performance-and-limitations).

Release-wide validation and publication remain pending; the release guide
records the scope of checks actually run for this preparation.

## 0.0.6

Previous release baseline: tag `v0.0.6`. This changelog starts with 0.0.7;
consult Git history for earlier changes.
