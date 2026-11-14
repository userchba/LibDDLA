# SAI GPU validation

The manual `SAI CUDA validation` workflow runs an approved LibDDLA commit on
SAI `16V100` with CUDA 12.9.1, GCC 13.3.0, Open MPI 5.0.10 and NCCL 2.29.3.
It builds with `sm_70` and runs the full CUDA-enabled CTest suite with four MPI
ranks, then reuses the same binaries for six-rank `pgeadd_indexed` and
`pgemm_branches` tests on a `1x6` process grid. The partition assigns host memory;
neither stage requests explicit Slurm memory.

## Runner setup

SAI administrators provide a dedicated `sai-libddla` runner and install the
[site controller](https://github.com/Stardust0831/LibDDLA/tree/66b2c419de961618671c9fbd99298dc6c4b2d255/.ci/sai)
separately. On the VM, place `relay.py`, `endpoint.py` and `report.py` together in
`/home/libddla-runner/ci/bin/`, owned by the administrator and read-only to the
runner. The dedicated SSH configuration is
`/home/libddla-runner/ci/ssh/config`. Install the controller and enforced SSH
command restriction on SAI before enabling the workflow.

The relay accepts `--source-sha`, `--run-id`, `--ssh-config`, `--work`, and
`--results`; `--cancel-only` cancels this attempt's recorded Slurm jobs. It
validates the administrator-approved full source SHA, transfers Git objects,
submits fixed four-/six-rank jobs and collects results. Compilation and tests
run inside Slurm allocations. Missing, skipped, failed or incomplete tests and
unsuccessful Slurm states must return a nonzero exit status. Results include
JUnit, raw logs, dependency/GPU details and Slurm accounting.

The repository owner registers the runner and configures Environment
`sai-libddla` with required reviewers and deployment branches `main`/`develop`.
The job needs only `contents: read`; no SSH key or repository-write token is
stored in this repository. Dedicated accounts, resource limits and controller
installation remain site administration responsibilities. Native builds run
as the SAI account, so approve trusted source only.

## Running

Once the workflow is on the default branch, select **Run workflow** from `main`
or `develop`, enter the exact 40-character commit SHA approved by the SAI
administrator, and approve the Environment. Each run uses a separate attempt
directory and uploads its evidence even when tests fail. The relay limits total
elapsed time, including queueing, to 165 minutes.

The workflow tests the selected commit as-is. Keep initial runs manual and
outside required merge checks while the two CUDA 12.9 Cholesky error-code
failures and deployment acceptance remain outstanding. Before activation,
verify the installed account restrictions, a complete Actions run with artifact
collection, and cancellation of that run's Slurm jobs.
