# HPC3 validation

The `HPC3 validation` workflow runs on **every push** to `main` or `develop`.
Manual runs can select the validation profile and an optional `source_sha`.
By default, both push and manual runs test the commit that triggered the
workflow. The workflow sends that checkout to the HPC3 cluster
(`hpc3.chinahpc.com:1014`, login `renxg`) over SSH, builds the CUDA backend,
runs the CPU suite and registered GPU tests through Slurm (submitted in
parallel, independent build trees), and publishes one table row per test in
the GitHub job summary. Raw logs are kept as an artifact. This mirrors the
KSSG setup in `.ci/kssg/`; HPC3 is a bare-metal (non-apptainer) cluster, so
this directory uses plain `module load` toolchains instead of a rootfs.

## GitHub setup

Repository secret required:

- `HPC3_SSH_PRIVATE_KEY` — private key for the `renxg` account on HPC3
  (public half appended to `/data/home/renxg/.ssh/authorized_keys`).

Host verification pins the committed `known_hosts` entry; do not disable it.

The workflow stages the selected source checkout in a fresh directory on HPC3
(`~/libddla_hpc3_ci/runs/github/<run_id>-<attempt>/`), so the pre-existing
canonical tree at `/data/home/renxg/app/github/LibDDLA` is never used or
modified by CI.

The CI control scripts come from the workflow's own commit
(`github.workflow_sha`), independently of the selected source ref.

## What runs

- **Quick**, used on pushes, runs the CPU and four-GPU CTest suites. With four
  ranks, fixtures using the shared API grid harness run on 2×2 by default.
  Explicit grid arguments, rank-specific tests and other fixtures follow their
  own configuration.
- **Full**, the default for manual runs, exercises each fixture's usual grid
  sweep and adds the six-GPU indexed addition and GEMM branch tests.

| Stage | Partitions | Allocation | Toolchain |
|---|---|---|---|
| CPU | `debug`, `48cp1-3`, `p1` | 4 ranks × 2 cores | GCC 11.4.0, OpenMPI 4.1.1, CMake 3.25.3, OpenBLAS |
| GPU 4-rank | `v100m3`, `v100g32fat`, `v100g32` | 4 V100s, 3 CPU cores per rank | GCC 11.3.0, CUDA-aware OpenMPI 4.1.8, CMake 3.25.3, NVHPC 25.5 |
| GPU 6-rank | Same GPU partitions | 6 V100s, 3 CPU cores per rank | Same GPU toolchain |

The controller submits the selected jobs together. Slurm picks an available
partition from each lane's configuration in `config.ini`. If `debug` rejects
the CPU submission because its per-user slot is occupied, the controller retries
on the CPU fallback partitions. Full CPU validation has a two-hour allocation
to accommodate the full grid sweep. CTest uses `PROCESSORS` to schedule within
the allocated rank count. The GPU wrapper maps MPI ranks to devices.

Both GPU jobs run on one node. The wrapper selects OpenMPI's
`self,vader,tcp` transports to avoid crashes in HPC3's `openib` component
during `MPI_Init`, including in single-rank tests. Local MPI traffic uses
shared memory with TCP available as a fallback; GPU collectives use NCCL.

What keeps each job short (measured 2026-10-24 on HEAD 1087342, same
node before/after):

- **Build tree and `TMPDIR` on the node-local `/tmp`**, removed when the
  job exits. On GPFS, CMake's configure stalled for up to 9 min, nvcc writes
  its intermediates to `TMPDIR`, and every `mpirun` creates its OpenMPI
  session directory there: the GPU suite took 523 s with `TMPDIR` on GPFS
  and 246 s on `/tmp`.
- **GPU jobs: NVHPC runtime libraries copied to `/tmp`.** cuBLAS(Lt),
  cuSOLVER, cuRAND and NCCL (2.5 GB) live on GPFS, where the loader pages
  them in at ~17 MB/s: on a node that had not loaded them lately the first
  test took 152 s instead of 2.4 s. `stage_gpu_libs` (`setup_gpu.sh`)
  copies them while CMake configures and builds, and the tests load the
  copies.
- **CPU job: `OMP_NUM_THREADS` = CPUs per rank (2).** Unset, each rank
  starts one thread per allocated CPU (8) and the 39 CPU tests take 303 s
  instead of 37 s.
- **GPU 4-rank suite: CTest parallelism matches the four allocated ranks.**
  CTest uses each test's `PROCESSORS` property, so two four-rank tests do
  not run simultaneously. On the host `getrf_nopiv` alone holds ~51 GB;
  the job retains its 128 GB allocation.

The CPU and four-GPU jobs set `DDLA_BUILD_BENCHMARKS=OFF` to keep standalone
benchmark binaries out of the default build. An explicit benchmark target
can still be built with `cmake --build <dir> --target <name>`.

Each submitted stage writes a result JSON and a summary table. The GitHub
check requires all submitted stages to finish successfully. Build trees are not kept: to debug a failure,
rebuild from `runs/github/<run>/source`.

## Reading failures

Results live under `~/libddla_hpc3_ci/runs/github/<run_id>-<attempt>/results/`
and are uploaded as GitHub Actions artifacts. The final **Fail when remote
validation failed** step reports an earlier remote failure; inspect:

- `jobs.txt` for each Slurm job ID, final state and exit code.
- `cpu-ctest.log`, `gpu4-ctest.log` and `gpu6-ctest.log`, plus their JUnit XML,
  for individual test failures. Test counts come from the selected source's
  CMake registrations and can change between revisions.
- `cpu-configure.log` / `cpu-build.log`, `gpu-configure.log` / `gpu-build.log`,
  and `gpu6-configure.log` / `gpu6-build.log` for failures before testing.
- `gpu4-slurm.log` and `gpu6-slurm.log` for the combined job output.
  The CPU template retains its `cpu-slurm-<job_id>.log` name.

An `mca_btl_openib.so` stack trace ending in `MPI_Init` identifies an MPI
startup failure before the numerical test runs. The GPU wrapper's
`self,vader,tcp` selection avoids that component in these single-node jobs.
