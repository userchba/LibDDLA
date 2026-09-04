# HPC3 validation

The `HPC3 validation` workflow runs on **every push** to `main` or `develop`.
It sends the pushed commit to the HPC3 cluster
(`hpc3.chinahpc.com:1014`, login `renxg`) over SSH, builds the CUDA backend,
runs the CPU suite and the full 43-case GPU suite through Slurm (submitted in
parallel, independent build trees), and publishes one table row per test in
the GitHub job summary. Raw logs are kept as an artifact. This mirrors the
KSSG setup in `.ci/kssg/`; HPC3 is a bare-metal (non-apptainer) cluster, so
this directory uses plain `module load` toolchains instead of a rootfs.

## GitHub setup

Repository secret required:

- `HPC3_SSH_PRIVATE_KEY` — private key for the `renxg` account on HPC3
  (public half appended to `/data/home/renxg/.ssh/authorized_keys`).

Host verification pins the committed `known_hosts` entry; do not disable it.

The workflow checks out the pushed commit in a fresh directory on HPC3
(`~/libddla_hpc3_ci/runs/github/<run_id>-<attempt>/`), so the pre-existing
canonical tree at `/data/home/renxg/app/github/LibDDLA` is never used or
modified by CI.

## What runs

| Stage | Partition | Job | Toolchain |
|---|---|---|---|
| CPU | `p1` (8 cores) | configure + build + `ctest -L cpu` (5 cases) | gcc/11.4.0, openmpi/4.1.1, cmake/3.25.3, OpenBLAS |
| GPU 4-rank suite | `v100g32fat` (4×V100) | configure + build + ctest minus `test_ptran_mpi` (42 cases) | gcc/11.3.0, openmpi/4.1.8-cuda, cmake/3.25.3, NVHPC 25.5 |
| GPU 6-rank suite | `v100m3` (6×V100) | build `test_ptran` + ctest `-R ^test_ptran_mpi$` (1 case) | same |

The GPU side is split into **two parallel jobs on different partitions**
because `v100g32` is chronologically congested (the CI's own 8-GPU job has
sat pending there for days): the default suite registers every test at
`DDLA_TEST_NP=4` (two `potrf_bottom_right` cases at 1), so it runs on 4
GPUs on `v100g32fat`; the single 6-rank exception `test_ptran` gets its own
6-GPU job on `v100m3` (same V100 nodes, more idle capacity). One rank per
GPU in both jobs via `gpurank_wrapper.sh`.

Each stage writes `result_{cpu,gpu,gpu6}.json` +
`summary_{cpu,gpu,gpu6}.md`; all three jobs must finish with exit 0 for the
GitHub check to be green. Runtime budget: CPU ≲ 30 min on a p1 node (but
queue time on p1 varies with cluster load), each GPU job ≲ 2.5 h (observed
43/43 pass in ~3 min + build when partitions are quiet). The GitHub job
timeout is 6 h; on rare days when the p1 queue exceeds that, re-run the
workflow instead of changing the allocation.
