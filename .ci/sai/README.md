# SAI V100 validation

The manually dispatched `SAI CUDA validation` workflow builds an approved commit
with CUDA 12.9.1 and executes the complete CUDA-enabled CTest inventory on SAI
`16V100`. It then reuses the same binaries for six-rank `pgeadd_indexed` and
`pgemm_branches` tests on a `1×6` process grid. The CUDA build also exercises
CPU-backend cases. Optional Fortran/Python bindings are disabled.

The VM runs the Actions listener, Git transport and reporting. Compilation and
tests run only inside Slurm allocations. The workflow does not upload controller
code to SAI: an administrator installs and approves that code separately.

## Configuration

| Component | Version or resource |
| --- | --- |
| CUDA / C++ compiler | CUDA 12.9.1 / GCC 13.3.0, `sm_70`, Release |
| MPI | Open MPI 5.0.10, NVHPC 26.3 GNU CUDA12 AVX2 |
| NCCL | SAI 2.29.3 for CUDA 12.9, CCL enabled |
| BLAS / CMake | SAIBLAS 2603 AVX2 / CMake 3.31.6 |
| Full suite | One node, 4 GPUs, default 4 ranks, 90-minute limit |
| Six-rank cases | One node, 8 allocated GPUs, 6 used, 25-minute limit |
| VM listener | Dedicated `libddla-runner` account; 2 GiB service memory cap |

The partition assigns host memory. Neither submission scripts nor their
environment set any explicit Slurm memory request. `rush-gpu` requires a GPU
allocation divisible by four, hence the eight-GPU allocation for six ranks.
Each local rank sees one device from Slurm's actual allocated mask; a runtime
probe checks the number of unique GPU UUIDs before tests start.

Auto-architecture modules are loaded inside each compute allocation. Tests run
serially so independent one-rank CTests do not compete for the same GPU. The
six-rank stage never reconfigures or rebuilds the four-rank build directory.

## Trigger and source provenance

1. Merge/review the controller and workflow changes. Dispatch is initially
   enabled only from `main` or `develop`; GitHub needs the workflow on the
   default branch before it exposes manual dispatch.
2. Have the SAI administrator add the full source commit SHA to
   `/etc/libddla-ci/policy.json`. Review applies to CMake and tests as well as
   library code. The example policy allows no commits.
3. Dispatch with that exact 40-character SHA and approve the `sai-libddla`
   GitHub Environment. Branch names, PR numbers and arbitrary repositories
   are not accepted as source inputs.

The relay fetches the commit from `userchba/LibDDLA` into a bare repository and
sends a Git bundle. The endpoint imports the bundle with object validation and
checks that its commit is exactly the administrator-approved SHA before
extracting source. A caller cannot relabel different source as an approved
SHA. Links, submodules and special files are currently rejected. No source
checkout, CMake hook or build is executed on the VM.

The recipe tests that commit as-is. It does not inject local patches or suppress
known failures. A fixed commit needs a new explicit approval when source changes.

## Failure and evidence handling

Each attempt has a unique `gh-RUN_ID-ATTEMPT` directory. A persistent submission
intent is written before `sbatch`, and its job receipt is reused on repeated
requests. An intent without a receipt stops instead of resubmitting: the site
administrator must reconcile Slurm by the stored job name. This handles the
ambiguous case where SSH fails after Slurm accepted a job.

The relay continues to the six-rank stage if the full suite fails but its build
and dependency/GPU probes succeeded. It caps total elapsed time, including
queueing, at 165 minutes. Cancellation targets only this run's recorded job IDs
after checking Slurm owner and name. The fixed Slurm wall times still apply if
the VM loses connectivity. If cancellation cannot be confirmed, logs say so.

The artifact includes CTest inventory, JUnit, raw logs, dependency paths, GPU
mapping, configuration, source/controller hashes and Slurm states. A passing
summary requires the exact complete inventory, no failed or skipped cases,
zero test/script exits, and a `COMPLETED|0:0` Slurm state for both stages.
Missing accounting, missing logs and incomplete tests remain failures.
GitHub artifact retention is 30 days; SAI run cleanup is an administrator task
restricted to terminal runs. The controller never deletes active runs.

## Initial status and rollout

The initial CUDA 12.9 baseline at commit
`1ff05d91559e52cf921ded8ad2aba34217e10f4d` had 146/152 passing CTests and 2/2
passing six-rank cases. The failures include a library lifetime defect, a
test allocation-attribution issue, and two cuSOLVER non-positive-definite
Cholesky error-code checks. The first two have separate proposed fixes; the
CUDA 12.9 Cholesky compatibility work is outstanding.

Start with manual runs and retain the actual failing status. Do not add the SAI
check to branch-protection requirements until the selected source has passed
and the deployment checks have completed. Automatic PR execution, cross-node
MPI, HIP and untrusted-code sandboxing are outside this initial workflow.

See [deployment/README.md](deployment/README.md) for owner and administrator
steps. Local checks require Python 3.10+ and Git:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s .ci/sai/tests -v
for script in .ci/sai/*.sh .ci/sai/*.sbatch .ci/sai/deployment/*.sh; do
    bash -n "$script"
done
```
