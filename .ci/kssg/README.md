# KSSG validation

The `KSSG validation` workflow is **manual only** (`workflow_dispatch`);
pushes do not start it. Select **Actions → KSSG validation → Run workflow**
when the cluster is available. It tests the requested `source_sha`, or
`develop` when that input is omitted, and sends that checkout to the KSSG cluster
(`cancon.hpccube.com:65023`, login `ac5g1n561p`) over SSH, builds the
HIP/DCU backend, runs the registered HIP GPU suite through Slurm, and
publishes one table row per test in the GitHub job summary. Raw logs are
kept as an artifact. This mirrors the HPC3 setup in `.ci/hpc3/`; **CPU
validation is owned by the HPC3 workflow** and is deliberately not run
here (the retired SAI setup that lived in `.ci/slurm/` was replaced by
this directory).

## GitHub setup

Repository secret required:

- `KSSG_SSH_PRIVATE_KEY` — private key for the `ac5g1n561p` account on KSSG
  (public half appended to `/public/home/ac5g1n561p/.ssh/authorized_keys`,
  comment `github-actions-ci@libddla`).

Host verification pins the committed `known_hosts` entry; do not disable it.

The workflow stages the selected source checkout in a fresh directory on KSSG
(`~/libddla_kssg_ci/runs/github/<run_id>-<attempt>/`), so the pre-existing
manual tree at `/public/home/ac5g1n561p/app/github/LibDDLA` is never used or
modified by CI.

CI control files come from `github.workflow_sha`, independently of the
selected source ref. Test counts depend on that source's CMake registrations.

## What runs

| Stage | Partition | Job | Toolchain |
|---|---|---|---|
| GPU (HIP) | `kshdnormal` (4 ranks, 4 DCUs) | configure + build + `ctest -L gpu -E "test_getrf_nopiv_mpi"` | gcc/11.2.0 + HPC-X + DTK 25.04.4 (gfx906) + RCCL (`DDLA_USE_CCL=ON`) |

KSSG notes baked into the control files:

- Every Slurm job must request `--gres=dcu:1` minimum (QOS rule).
- Slurm caps memory at `DefMemPerCPU = 3500` MB per allocated CPU: the
  ceiling is 3.5G × (ntasks × cpus-per-task); the working shape here is
  `--ntasks=4 --cpus-per-task=8 --gres=dcu:4 --mem=96G` (112G ceiling).
  24G OOM-kills `test_pgetrf_nopiv`'s n=20000 case (multi-GB host-side
  log-det verification); 64G/16 CPUs → "Requested node configuration is
  not available".
- KSSG has **no python3** on login or compute nodes (Python 2.7.5 only), so
  `submit.sh` is pure bash/awk — the summary table is rendered on the GitHub
  runner from the JUnit XML file, not on the cluster.
- DTK 25.04.4 + CMake 3.25 traps handled in `setup_gpu.sh`: `HIP_PATH` must
  be the DTK root (no `/hip` suffix), `CMAKE_PREFIX_PATH` must be a single
  entry, and the default `dtk-22.10.1` module must be unloaded first or
  module loads silently no-op (`conflict compiler`).
- `rccl.h` and `thrust/` ship under `$ROCM_PATH/include`; the configure adds
  `-I$ROCM_PATH/include` for `DDLA_USE_CCL=ON` builds. Never add the
  DTK-bundled CUDA include tree (`$ROCM_PATH/cuda/cuda-12/...`) — its
  `vector_types.h` conflicts with HIP's vector types and breaks the build.
- RCCL is present (`$ROCM_PATH/lib/librccl.so`) and auto-detected by
  LibDDLA's HIP lib discovery; `DDLA_USE_CCL=ON` is mandatory — direct MPI
  on device pointers is pathologically slow (RCCL is ~1300× faster on
  panel-size broadcasts).
- `test_getrf_nopiv_mpi` is excluded on the GPU suite: its n=15000 case
  is pathologically slow on gfx906. Its single-GPU sibling
  `test_getrf_nopiv` stays in the suite.

The whole validation is **one** sbatch job: ~2.5 min build + ~7 min tests
(the `test_pgetrf_nopiv` sweep with RCCL is 31 s) on a kshdnormal node;
the GitHub job timeout is 3 h.

The job writes `result_gpu.json` + `summary.md`; the GitHub check is green
only when the HIP suite passes.
