# HPC3 validation

The `HPC3 validation` workflow runs on **every push** to `main` or `develop`.
It sends the pushed commit to the HPC3 cluster
(`hpc3.chinahpc.com:1014`, login `renxg`) over SSH, builds the CUDA backend,
runs the CPU suite and the full 43-case GPU suite through Slurm (submitted in
parallel, independent build trees), and publishes one table row per test in
the GitHub job summary. Raw logs are kept as an artifact. This mirrors the
SAI setup in `.ci/slurm/`; HPC3 is a bare-metal (non-apptainer) cluster, so
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
| GPU | `v100g32` (8×V100) | configure + build + full ctest (43 cases) | gcc/11.3.0, openmpi/4.1.8-cuda, cmake/3.25.3, NVHPC 25.5 |

Each stage writes `result_{cpu,gpu}.json` + `summary_{cpu,gpu}.md`; both jobs
must finish with exit 0 for the GitHub check to be green. Runtime budget:
CPU ≲ 30 min on a p1 node (but queue time on p1 varies with cluster load),
GPU ≲ 2.5 h on 8 V100s (observed 43/43 pass in ~3 min + build). The GitHub
job timeout is 6 h; on rare days when the p1 queue exceeds that, re-run the
workflow instead of changing the allocation.
