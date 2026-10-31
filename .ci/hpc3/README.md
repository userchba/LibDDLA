# HPC3 validation

The `HPC3 validation` workflow runs on **every push** to `main` or `develop`.
It also supports manual `workflow_dispatch` runs. The source checkout uses
the requested `source_sha`, or `develop` when that input is omitted. It sends
that checkout to the HPC3 cluster
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

| Stage | Partitions (first to start wins) | Job | Toolchain |
|---|---|---|---|
| CPU | `debug`, `48cp1-3`, `p1` (4 ranks × 2 cores) | configure + build + `ctest -L cpu` | gcc/11.4.0, openmpi/4.1.1, cmake/3.25.3, OpenBLAS |
| GPU 4-rank suite | `v100m3`, `v100g32fat`, `v100g32` (4 ranks × 4 cores, 4×V100) | configure + build + the default four-rank CTest suite | gcc/11.3.0, openmpi/4.1.8-cuda, cmake/3.25.3, NVHPC 25.5 |
| GPU 6-rank suite | same (6 ranks × 4 cores, 6×V100) | build indexed addition/GEMM tests + CTest at six ranks | same |

The three jobs are submitted together and run in parallel. Each lists
several partitions (`config.ini`) and Slurm starts it in whichever frees
up first; a single partition used to leave the CPU job pending on `p1` for
hours and the GitHub job ran into its 6 h timeout. `debug` takes one job
per user, so when that slot is busy `submit.sh` resubmits the CPU job on
its `fallback_partition` list. The one-hour limits are ~10× the run time
and let the backfill scheduler slot the jobs into gaps on busy partitions.

The default suite registers every test at `DDLA_TEST_NP=4` (a few cases
at 1), so it runs on 4 GPUs; indexed `pgeadd` and GEMM branch tests
additionally run at `DDLA_TEST_NP=6` in their own job. `gpurank_wrapper.sh`
pins rank *r* to GPU *r*.

The GPU jobs request four and six Slurm tasks respectively, with four CPU
cores per task. MPI launches use only the allocated slots. The four-GPU
job limits CTest's concurrent rank count to `SLURM_NTASKS`; the six-GPU
tests run sequentially.

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

Each stage writes `result_{cpu,gpu4,gpu6}.json` +
`summary_{cpu,gpu4,gpu6}.md`; all three jobs must finish with exit 0 for
the GitHub check to be green. Build trees are not kept: to debug a failure,
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
